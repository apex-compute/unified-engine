"""OllApeX: a demo chat interface for a model that runs tensor-parallel on Apex Compute FPGA cards.

    python ollapex_server.py            # http://0.0.0.0:3000

  GET  /                 the web interface
  GET  /api/status       engine state, cards, limits, last run's technical details
  POST /api/chat         {model, messages, options, think} -> newline-delimited JSON events
  POST /api/file         raw file body (X-Filename header) -> extracted text and its token count,
                         or for an image: an id to attach to a message and its token cost
  POST /api/load         {model, num_ctx}  (re)load a model onto the cards
  POST /api/abort        stop the running generation

One request generates at a time (the cards hold one conversation state); others wait.
"""
import hashlib
import io
import json
import mimetypes
import multiprocessing as mp
import os
import platform
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import ollapex_engine as eng
import ollapex_vision as vis

VERSION = "0.2.0"
HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
DEFAULT_MODEL = os.environ.get("OLLAPEX_MODEL", "qwen3.8:27b")
DEFAULT_CTX = int(os.environ.get("OLLAPEX_CTX", "1024"))
PORT = int(os.environ.get("OLLAPEX_PORT", "3000"))
CTX_CHOICES = [256, 512, 1024, 2048, 4096, 8192, 16384]
DEFAULT_PREDICT = 1024
MAX_UPLOAD = 16 * 1024 * 1024
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
UPLOADS = "/dev/shm/ollapex_uploads"
DEFAULT_DETAIL = "low"
MAX_IMAGES = 64
IMAGES = {}                                       # image id -> {name, path, width, height, enc: {detail: encoded}}


def host_info():
    mem = 0
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal"):
            mem = int(line.split()[1]) / 1048576.0
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")
    return {"hostname": platform.node(), "kernel": platform.release(), "cpu": cpu, "cpus": os.cpu_count(), "ram_gb": round(mem, 1)}


class Engine:
    """The driver process plus what the server knows about it."""

    def __init__(self):
        self.lock = threading.Lock()              # one load / one generation at a time
        self.ctx = mp.get_context("spawn")
        self.abort = self.ctx.Event()
        self.conn = None
        self.proc = None
        self.status = {"state": "unloaded", "model": None, "num_ctx": None, "progress": None, "error": None,
                       "info": None, "last": None, "waiting": 0}
        self.tok = {}
        self.last_turn = None                     # for continuing a conversation without re-reading it
        self._start()

    def _start(self):
        self.conn, child = self.ctx.Pipe()
        # not a daemon: the driver has child processes of its own (one per card); it exits when this pipe closes
        self.proc = self.ctx.Process(target=eng.driver_main, args=(child, self.abort))
        self.proc.start()
        child.close()

    def tokenizer(self, model):
        if model not in self.tok:
            from transformers import AutoTokenizer
            self.tok[model] = AutoTokenizer.from_pretrained(eng.model_paths(model)["hf"], local_files_only=True)
        return self.tok[model]

    def _recv(self):
        while not self.conn.poll(1.0):
            if not self.proc.is_alive():
                self._start()
                self.status.update(state="unloaded", model=None, info=None, error="the engine process ended unexpectedly")
                raise RuntimeError(self.status["error"])
        return self.conn.recv()

    def _fail(self, ev):
        fatal = len(ev) > 2 and ev[2]
        if fatal:
            self.status.update(state="unloaded", model=None, num_ctx=None, info=None, error=ev[1])
            self.last_turn = None
            return RuntimeError(ev[1].split("\n")[0])
        return ValueError(ev[1])

    def ensure(self, model, num_ctx):
        """Caller holds the lock. Load (or reload) so that `model` is resident with `num_ctx`."""
        st = self.status
        if st["state"] in ("ready", "busy") and st["model"] == model and st["num_ctx"] == num_ctx:
            return 0.0
        t0 = time.time()
        st.update(state="loading", model=model, num_ctx=num_ctx, error=None, info=None,
                  progress={"ready": 0, "cards": 0, "since": t0})
        self.last_turn = None
        self.conn.send(("load", model, num_ctx))
        while True:
            ev = self._recv()
            if ev[0] == "progress":
                st["progress"] = dict(ev[1], since=t0)
            elif ev[0] == "ready":
                st.update(state="ready", info=ev[1], progress=None)
                return time.time() - t0
            elif ev[0] == "error":
                raise self._fail((ev[0], ev[1], True))

    def encode_image(self, path, W, H, level=None):
        """Caller holds the lock and the model is loaded. Runs the image encoder on the cards."""
        self.conn.send(("vision", path, W, H) + ((level,) if level else ()))
        while True:
            ev = self._recv()
            if ev[0] == "vision_done":
                return np.frombuffer(ev[1], dtype=np.int16).reshape(ev[2]), ev[3]
            if ev[0] == "error":
                raise self._fail(ev)

    def run(self, model, num_ctx, prepare):
        """Generator of engine events for one generation. Holds the lock for its whole life.
        prepare() is a generator run under the lock with the model loaded: it may yield events
        (image encoding progress) and returns the request for the engine."""
        self.status["waiting"] += 1
        self.lock.acquire()
        self.status["waiting"] -= 1
        try:
            load_s = self.ensure(model, num_ctx)
            self.status["state"] = "busy"
            self.abort.clear()
            req = yield from prepare()
            self.conn.send(("generate", req))
            while True:
                ev = self._recv()
                if ev[0] == "error":
                    raise self._fail(ev)
                if ev[0] == "done":
                    ev[1]["load_s"] = load_s
                    ev[1]["model"] = model
                    self.status["last"] = ev[1]
                yield ev
                if ev[0] == "done":
                    return
        finally:
            if self.status["state"] == "busy":
                self.status["state"] = "ready"
            self.lock.release()

    def load_async(self, model, num_ctx):
        def job():
            with self.lock:
                try:
                    self.ensure(model, num_ctx)
                except Exception as e:
                    self.status["error"] = str(e)
        threading.Thread(target=job, daemon=True).start()


ENGINE = None
HOST = {}


def clamp_ctx(v):
    v = int(v or DEFAULT_CTX)
    return max(c for c in CTX_CHOICES if c <= max(v, CTX_CHOICES[0]))


def resolve_model(name):
    models = eng.available_models()
    name = name or ENGINE.status["model"] or DEFAULT_MODEL
    if name not in models:
        raise ValueError("model not found: %s (available: %s)" % (name, ", ".join(models)))
    return name


def normalize(messages):
    return [{"role": m.get("role", "user"), "content": str(m.get("content") or ""),
             "images": [str(i) for i in (m.get("images") or [])]} for m in messages]


def for_template(m):
    """A message as the chat template wants it: images first, then the text."""
    if not m["images"]:
        return {"role": m["role"], "content": m["content"]}
    return {"role": m["role"], "content": [{"type": "image"}] * len(m["images"]) + [{"type": "text", "text": m["content"]}]}


def build_prompt(model, msgs, think, encoded):
    """Token ids for a chat, plus the image vectors and their positions. `encoded` maps an image id
    to its encoder output. When the request is the previous exchange plus one new user message,
    the ids are built as an exact continuation of what the cards already hold, so only the new
    turn has to be read (the cards cannot rewind; they can only continue)."""
    tok = ENGINE.tokenizer(model)
    if eng.is_gemma4(model):
        # Gemma 4 chat template (it starts with <bos>); "think" turns its thinking channel on. The template
        # leaves one <|image|> per image: it becomes <|image> + one <|image|> per soft token + <image|>, and
        # those positions are fed the encoder's soft tokens instead of a token embedding.
        def expand_g(text, ms):
            imgs = [encoded[i]["tokens"] for m in ms for i in m["images"]]
            parts = text.split("<|image|>")
            if len(parts) != len(imgs) + 1:
                raise ValueError("the chat template placed %d image slots for %d images" % (len(parts) - 1, len(imgs)))
            text = parts[0] + "".join("<|image>" + "<|image|>" * len(t) + "<image|>" + p for t, p in zip(imgs, parts[1:]))
            ids = [int(v) for v in tok.encode(text, add_special_tokens=False)]
            img_id = tok.convert_tokens_to_ids("<|image|>")
            rows = [r for t in imgs for r in t]
            at = [i for i, v in enumerate(ids) if v == img_id]
            if len(at) != len(rows):
                raise ValueError("image tokens in the prompt: %d, soft tokens: %d" % (len(at), len(rows)))
            return ids, {i: r.tobytes() for i, r in zip(at, rows)}

        def render(ms):
            return tok.apply_chat_template([for_template(m) for m in ms], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=bool(think))

        # The template renders past model turns without the (empty) thinking block it opens each new turn with,
        # so a re-rendered chat never continues what the cards hold. A follow-up (the previous exchange plus one
        # user message) is therefore built as the exact continuation, as for Qwen above.
        lt = ENGINE.last_turn
        eot = tok.convert_tokens_to_ids("<turn|>")
        if (lt and lt["model"] == model and lt["think"] == think and len(msgs) == len(lt["messages"]) + 2
                and msgs[:-2] == lt["messages"] and msgs[-2]["role"] == "assistant" and msgs[-1]["role"] == "user"
                and msgs[-2]["content"].strip() == lt["reply"].strip()):
            tail = render([msgs[-1]])
            tail = tail[len("<bos>"):] if tail.startswith("<bos>") else tail
            out = list(lt["out"])
            if not out or out[-1] != eot:
                out.append(eot)
            base = lt["ids"] + out
            tids, tvecs = expand_g("\n" + tail, [msgs[-1]])
            vecs = dict(lt["vecs"])
            vecs.update({len(base) + i: v for i, v in tvecs.items()})
            return base + tids, vecs, [], eot
        ids, vecs = expand_g(render(msgs), msgs)
        return ids, vecs, [], eot
    if eng.is_gptoss(model):
        # harmony chat template; "think" picks the reasoning effort (the model always writes an analysis
        # first). No continuation shortcut: the engine still reuses whatever prefix the cards already hold.
        if any(m["images"] for m in msgs):
            raise ValueError("gpt-oss cannot read images.")
        text = tok.apply_chat_template([{"role": m["role"], "content": m["content"]} for m in msgs], tokenize=False,
                                       add_generation_prompt=True, reasoning_effort="medium" if think else "low")
        return [int(v) for v in tok.encode(text, add_special_tokens=False)], {}, [], None
    pad = tok.convert_tokens_to_ids("<|image_pad|>")
    lt = ENGINE.last_turn
    eot = tok.convert_tokens_to_ids("<|im_end|>")

    def expand(ids, ms):
        return vis.expand_image_tokens(ids, pad, [encoded[i] for m in ms for i in m["images"]])

    if (lt and lt["model"] == model and lt["think"] == think and len(msgs) == len(lt["messages"]) + 2
            and msgs[:-2] == lt["messages"] and msgs[-2]["role"] == "assistant" and msgs[-1]["role"] == "user"
            and msgs[-2]["content"].strip() == lt["reply"].strip()):
        tail = tok.apply_chat_template([for_template(msgs[-1])], tokenize=False, add_generation_prompt=True, enable_thinking=think)
        out = list(lt["out"])
        if not out or out[-1] != eot:
            out.append(eot)
        base = lt["ids"] + out
        tids, tvecs, tblocks = expand([int(v) for v in tok.encode("\n" + tail, add_special_tokens=False)], [msgs[-1]])
        vecs = dict(lt["vecs"])
        vecs.update({len(base) + i: v for i, v in tvecs.items()})
        blocks = list(lt["blocks"]) + [(len(base) + s, gh, gw) for s, gh, gw in tblocks]
        return base + tids, vecs, blocks, eot
    text = tok.apply_chat_template([for_template(m) for m in msgs], tokenize=False, add_generation_prompt=True, enable_thinking=think)
    ids, vecs, blocks = expand([int(v) for v in tok.encode(text, add_special_tokens=True)], msgs)
    return ids, vecs, blocks, eot


class Detok:
    """Incremental detokenizer: emits text as soon as it is complete UTF-8, and separates the
    thinking part (before </think>) from the answer when thinking is on."""

    def __init__(self, tok, think, skip):
        self.tok, self.ids, self.sent, self.skip = tok, [], 0, set(skip)
        self.in_think = think
        self.hold = ""

    def push(self, token):
        if token in self.skip:
            return "", ""
        self.ids.append(token)
        text = self.tok.decode(self.ids, skip_special_tokens=False)
        if text.endswith("Fffd"):
            return "", ""
        piece, self.sent = text[self.sent:], len(text)
        if not self.in_think:
            return piece, ""
        self.hold += piece
        if "</think>" in self.hold:
            th, rest = self.hold.split("</think>", 1)
            self.in_think, self.hold = False, ""
            return rest.lstrip("\n"), th
        keep = max(0, len(self.hold) - 8)          # keep a tail in case the tag is split across tokens
        th, self.hold = self.hold[:keep], self.hold[keep:]
        return "", th

    def flush(self):
        th, self.hold = (self.hold, "") if self.in_think else ("", "")
        return th


class GemmaDetok:
    """Gemma 4 output: an optional <|channel>thought\n ... <channel|> block (the thinking), then the answer.
    Same push/flush interface as Detok."""

    def __init__(self, tok, skip):
        self.tok, self.skip = tok, set(skip)
        self.OPEN, self.CLOSE = tok.convert_tokens_to_ids("<|channel>"), tok.convert_tokens_to_ids("<channel|>")
        self.thinking, self.ids, self.sent, self.head = False, [], 0, False

    def push(self, token):
        if token in self.skip:
            return "", ""
        if token == self.OPEN:
            self.thinking, self.ids, self.sent, self.head = True, [], 0, True
            return "", ""
        if token == self.CLOSE:
            self.thinking, self.ids, self.sent, self.head = False, [], 0, False
            return "", ""
        self.ids.append(token)
        text = self.tok.decode(self.ids, skip_special_tokens=True)
        if text.endswith("\ufffd"):
            return "", ""
        if self.head:                                  # the channel name ("thought") and its newline
            if "\n" not in text:
                return "", ""
            self.head = False
            self.sent = text.index("\n") + 1
        piece, self.sent = text[self.sent:], len(text)
        return ("", piece) if self.thinking else (piece, "")

    def flush(self):
        return ""


class HarmonyDetok:
    """GPT-OSS (harmony format) output: <|channel|>NAME<|message|>TEXT<|end|><|start|>assistant<|channel|>...
    Text of the analysis channel is the thinking, the final (and commentary) channel is the answer.
    Same push/flush interface as Detok."""

    def __init__(self, tok, skip):
        self.tok, self.skip = tok, set(skip)
        sid = tok.convert_tokens_to_ids
        self.CH, self.MSG = sid("<|channel|>"), sid("<|message|>")
        self.HDR = {sid(t) for t in ("<|start|>", "<|end|>", "<|return|>", "<|call|>", "<|constrain|>")}
        self.mode, self.chan, self.ids, self.sent = "header", [], [], 0
        self.thinking = True

    def push(self, token):
        if token in self.skip:
            return "", ""
        if token == self.CH:
            self.mode, self.chan = "channel", []
            return "", ""
        if token == self.MSG:
            name = self.tok.decode(self.chan).strip()
            self.thinking = name.startswith("analysis")
            self.mode, self.ids, self.sent = "body", [], 0
            return "", ""
        if token in self.HDR:
            self.mode = "header"
            return "", ""
        if self.mode == "channel":
            self.chan.append(token)
            return "", ""
        if self.mode != "body":
            return "", ""                              # the role name after <|start|>
        self.ids.append(token)
        text = self.tok.decode(self.ids, skip_special_tokens=False)
        if text.endswith("�"):
            return "", ""
        piece, self.sent = text[self.sent:], len(text)
        return ("", piece) if self.thinking else (piece, "")

    def flush(self):
        return ""


def run_chat(model, messages, options, think, send):
    """send(obj) writes one event to the client. Events: status, thinking, content, done."""
    num_ctx = clamp_ctx(options.get("num_ctx") or ENGINE.status["num_ctx"] or DEFAULT_CTX)
    detail = options.get("image_detail") if options.get("image_detail") in vis.DETAIL else DEFAULT_DETAIL
    msgs = normalize(messages)
    tok = ENGINE.tokenizer(model)
    harmony = eng.is_gptoss(model)
    gemma = eng.is_gemma4(model)
    if gemma:
        # <turn|> ends the model turn; <eos> too
        stop_ids = [tok.convert_tokens_to_ids("<turn|>"), tok.convert_tokens_to_ids("<eos>")]
    elif harmony:
        # <|end|> closes the analysis message and the final answer follows; the turn ends at <|return|>
        stop_ids = [tok.convert_tokens_to_ids(t) for t in ("<|return|>", "<|call|>")]
    else:
        eot = tok.convert_tokens_to_ids("<|im_end|>")
        stop_ids = [eot] + [t for t in (tok.convert_tokens_to_ids("<|endoftext|>"),) if isinstance(t, int) and t >= 0]
    opts = {"num_predict": options.get("num_predict", DEFAULT_PREDICT), "temperature": options.get("temperature", 0.0),
            "top_k": options.get("top_k", eng.MAX_RANK), "top_p": options.get("top_p", 1.0), "seed": options.get("seed")}
    det = (GemmaDetok(tok, stop_ids) if gemma else HarmonyDetok(tok, stop_ids) if harmony
           else Detok(tok, think, stop_ids))
    content, out_ids, stats = [], [], None
    built = {}
    image_report = []

    def prepare():
        """Runs under the engine lock with the model loaded: encode the images, then build the prompt."""
        wanted = [i for m in msgs for i in m["images"]]
        if wanted and not (ENGINE.status.get("info") or {}).get("vision"):
            raise ValueError("The loaded model cannot read images.")
        encoded = {}
        for iid in wanted:
            im = IMAGES.get(iid)
            if im is None:
                raise ValueError("An attached image is no longer available on the server; attach it again.")
            if gemma:
                gkey = "gemma:" + detail                  # the Qwen encoder's output is cached under the bare level
                if gkey in im["enc"]:
                    encoded[iid] = im["enc"][gkey]
                    continue
                S = detail_levels()[detail]
                plan = image_plans(im)[detail]
                yield ("vision", {"name": im["name"], "width": plan["width"], "height": plan["height"],
                                  "tokens": plan["tokens"], "cards": True})
                toks, meta = ENGINE.encode_image(im["path"], plan["width"], plan["height"], S)
                im["enc"][gkey] = encoded[iid] = {"tokens": toks}
                image_report.append({"name": im["name"], "source": [im["width"], im["height"]],
                                     "encoded": [plan["width"], plan["height"]], "detail": detail,
                                     "patches": meta["patches"], "tokens": meta["tokens"], "card": meta["card"],
                                     "card_s": meta["card_s"], "compile_s": 0.0, "weights_upload_s": 0.0,
                                     "instructions": meta["instructions"], "segments": meta["segments"]})
                continue
            if detail not in im["enc"]:
                W, H = vis.plan_size(im["width"], im["height"], detail)
                yield ("vision", {"name": im["name"], "width": W, "height": H, "tokens": vis.tokens_for(W, H)})
                toks, meta = ENGINE.encode_image(im["path"], W, H)
                im["enc"][detail] = {"tokens": toks, "gh": meta["grid"][0] // 2, "gw": meta["grid"][1] // 2}
                image_report.append({"name": im["name"], "source": [im["width"], im["height"]], "encoded": [W, H], "detail": detail,
                                     "patches": meta["patches"], "tokens": meta["tokens"], "card": meta["card"],
                                     "card_s": meta["card_s"], "compile_s": meta["compile_s"],
                                     "weights_upload_s": meta["weights_upload_s"], "instructions": meta["instructions"]})
            encoded[iid] = im["enc"][detail]
        ids, vecs, blocks, _ = build_prompt(model, msgs, think, encoded)
        built.update(ids=ids, vecs=vecs, blocks=blocks)
        if gemma:                                     # 1D positions; each image is one block of soft tokens
            runs = []
            for i in sorted(vecs):
                if runs and runs[-1][0] + runs[-1][1] == i:
                    runs[-1][1] += 1
                else:
                    runs.append([i, 1])
            return {"ids": ids, "vecs": vecs, "opts": opts, "stop_ids": stop_ids, "image_blocks": runs}
        return {"ids": ids, "vecs": vecs, "opts": opts, "stop_ids": stop_ids, "rope": vis.rope_positions(len(ids), blocks)}

    gen = ENGINE.run(model, num_ctx, prepare)
    try:
        for ev in gen:
            if ev[0] == "token":
                out_ids.append(ev[1])
                c, t = det.push(ev[1])
                if t:
                    send({"type": "thinking", "text": t})
                if c:
                    content.append(c)
                    send({"type": "content", "text": c})
            elif ev[0] in ("start", "prefill", "vision"):
                send(dict(ev[1], type="status", event=ev[0]))
            elif ev[0] == "done":
                stats = ev[1]
    except (BrokenPipeError, ConnectionResetError):
        ENGINE.abort.set()                           # the browser went away: stop the cards, then let the engine finish
        for ev in gen:
            pass
        raise
    t = det.flush()
    if t:
        send({"type": "thinking", "text": t})
    ENGINE.last_turn = {"model": model, "think": think, "messages": msgs, "reply": "".join(content), "out": out_ids,
                        "ids": built["ids"], "vecs": built["vecs"], "blocks": built["blocks"]}
    if stats is not None:
        stats["thinking"] = think
        stats["image_tokens"] = len(built["vecs"])
        stats["images_encoded"] = image_report
    send({"type": "done", "stats": stats})


def add_image(name, data):
    """Keep an uploaded image for a later chat request; report what it costs at each detail level."""
    if not (ENGINE.status.get("info") or {}).get("vision"):
        raise ValueError("The loaded model cannot read images.")
    from PIL import Image
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        raise ValueError("This image could not be read.")
    img = img.convert("RGB")
    iid = hashlib.md5(data).hexdigest()[:16]
    if iid not in IMAGES:
        os.makedirs(UPLOADS, exist_ok=True)
        while len(IMAGES) >= MAX_IMAGES:                    # oldest out
            old = IMAGES.pop(next(iter(IMAGES)))
            try:
                os.remove(old["path"])
            except OSError:
                pass
        path = os.path.join(UPLOADS, iid + ".png")
        img.save(path)
        IMAGES[iid] = {"name": name, "path": path, "width": img.size[0], "height": img.size[1], "enc": {}}
    im = IMAGES[iid]
    plans = image_plans(im)
    return {"name": name, "kind": "image", "image_id": iid, "width": im["width"], "height": im["height"], "plans": plans,
            "text": "", "tokens": plans[DEFAULT_DETAIL]["tokens"], "num_ctx": ENGINE.status["num_ctx"] or clamp_ctx(DEFAULT_CTX)}


def detail_levels():
    """Detail level -> soft tokens per image when the loaded model sets them (Gemma 4), else None (Qwen: pixels)."""
    return (ENGINE.status.get("info") or {}).get("vision_levels") or None


def image_plans(im):
    """What the image becomes at each detail level: encoded size and prompt tokens."""
    levels = detail_levels()
    if not levels:
        plans = {}
        for d in vis.DETAIL:
            W, H = vis.plan_size(im["width"], im["height"], d)
            plans[d] = {"width": W, "height": H, "tokens": vis.tokens_for(W, H), "ready": d in im["enc"]}
        return plans
    key = tuple(sorted(levels.items()))
    if im.get("gplans_key") != key:
        # the model's own image processor (resize + patches only): the side lengths and the soft tokens
        from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor
        from PIL import Image
        img = Image.open(im["path"]).convert("RGB")
        im["gplans"] = {}
        for d, S in levels.items():
            pp = Gemma4ImageProcessor(max_soft_tokens=S)(images=[img], return_tensors="np")["image_position_ids"][0]
            pp = pp[(pp >= 0).all(-1)]
            g0, g1 = int(pp[:, 0].max()) + 1, int(pp[:, 1].max()) + 1
            im["gplans"][d] = {"width": 16 * g0, "height": 16 * g1, "tokens": len(pp) // 9 + 2}
        im["gplans_key"] = key
    return {d: dict(p, ready="gemma:" + d in im["enc"]) for d, p in im["gplans"].items()}


def extract_file(name, data):
    """Text of an uploaded file plus how many tokens it costs in the loaded model's context."""
    low = name.lower()
    if low.endswith(IMAGE_EXT):
        return add_image(name, data)
    if low.endswith(".pdf"):
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(data))
        text = "\n\n".join((p.extract_text() or "").strip() for p in reader.pages).strip()
        kind = "pdf, %d pages" % len(reader.pages)
        if not text:
            raise ValueError("No text could be extracted from this PDF (it may be a scanned image).")
    else:
        if b"\x00" in data[:4096]:
            raise ValueError("This file is not text. Attach a text, code or PDF file.")
        text = data.decode("utf-8", errors="replace").replace("\r\n", "\n").strip()
        kind = "text"
    model = ENGINE.status["model"] or DEFAULT_MODEL
    tokens = len(ENGINE.tokenizer(model).encode(text, add_special_tokens=False))
    return {"name": name, "kind": kind, "text": text, "chars": len(text), "tokens": tokens,
            "num_ctx": ENGINE.status["num_ctx"] or clamp_ctx(DEFAULT_CTX)}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "OllApeX/" + VERSION

    def log_message(self, fmt, *args):
        if "/api/status" not in (args[0] if args else ""):
            sys.stderr.write("%s %s %s\n" % (time.strftime("%H:%M:%S"), self.address_string(), fmt % args))

    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def chunk(self, obj):
        data = (json.dumps(obj, ensure_ascii=False) + "\n").encode()
        self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
        self.wfile.flush()

    def raw_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            raise ValueError("File too large (max %d MB)." % (MAX_UPLOAD // 1048576))
        return self.rfile.read(n) if n else b""

    def do_GET(self):
        path = self.path.split("?")[0]
        try:
            if path in ("/", "/index.html"):
                return self.static("index.html")
            if path.startswith("/static/"):
                return self.static(path[len("/static/"):])
            if path == "/api/status":
                st = dict(ENGINE.status)
                if st.get("progress"):
                    st["progress"] = dict(st["progress"], elapsed_s=time.time() - st["progress"]["since"])
                st.update({"version": VERSION, "host": HOST, "models": eng.available_models(),
                           "limits": {"ctx_choices": CTX_CHOICES, "max_ctx": eng.MAX_CTX, "max_top_k": eng.MAX_RANK,
                                      "default_num_predict": DEFAULT_PREDICT, "max_upload_mb": MAX_UPLOAD // 1048576},
                           "vision": bool((st.get("info") or {}).get("vision")), "default_model": DEFAULT_MODEL,
                           "image_detail": {"choices": detail_levels() or vis.DETAIL, "default": DEFAULT_DETAIL,
                                            "unit": "tokens" if detail_levels() else "px"}})
                return self.send_json(st)
            self.send_json({"error": "not found"}, 404)
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def static(self, rel):
        full = os.path.realpath(os.path.join(STATIC, rel))
        if not full.startswith(os.path.realpath(STATIC) + os.sep) or not os.path.isfile(full):
            return self.send_json({"error": "not found"}, 404)
        data = open(full, "rb").read()
        self.send_response(200)
        self.send_header("Content-Type", (mimetypes.guess_type(full)[0] or "application/octet-stream")
                         + ("; charset=utf-8" if full.endswith((".html", ".js", ".css", ".svg")) else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        path = self.path.split("?")[0]
        streaming = False
        try:
            if path == "/api/file":
                name = urllib.parse.unquote(self.headers.get("X-Filename") or "file.txt")
                return self.send_json(extract_file(os.path.basename(name), self.raw_body()))
            req = json.loads(self.raw_body() or b"{}")
            if path == "/api/chat":
                model = resolve_model(req.get("model"))
                messages = list(req.get("messages") or [])
                if not messages:
                    raise ValueError("no messages")
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                streaming = True
                run_chat(model, messages, dict(req.get("options") or {}), bool(req.get("think", False)), self.chunk)
                self.wfile.write(b"0\r\n\r\n")
                return
            if path == "/api/load":
                name = resolve_model(req.get("model"))
                ENGINE.load_async(name, clamp_ctx(req.get("num_ctx")))
                return self.send_json({"status": "loading", "model": name})
            if path == "/api/abort":
                ENGINE.abort.set()
                return self.send_json({"status": "ok"})
            self.send_json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            try:
                if streaming:
                    self.chunk({"type": "error", "message": str(e)})
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    self.send_json({"error": str(e)}, 400 if isinstance(e, ValueError) else 500)
            except Exception:
                pass


def main():
    global ENGINE, HOST
    HOST = host_info()
    ENGINE = Engine()
    if os.environ.get("OLLAPEX_AUTOLOAD", "1") != "0":
        ENGINE.load_async(DEFAULT_MODEL, clamp_ctx(DEFAULT_CTX))
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.daemon_threads = True
    print("OllApeX %s  http://0.0.0.0:%d  (model %s, num_ctx %d)" % (VERSION, PORT, DEFAULT_MODEL, clamp_ctx(DEFAULT_CTX)), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            ENGINE.conn.send(("quit",))
            ENGINE.proc.join(timeout=20)
        except Exception:
            pass


if __name__ == "__main__":
    main()
