#!/usr/bin/env python3
"""Unified web UI for the models that run on the board.

One page, one model dropdown:

* Gemma 3 1B            -- text chat
* Gemma 4 E2B / E4B     -- chat with text, an image and/or an audio clip
* FLUX.2 [klein] 4B     -- text-to-image (the flux2_klein_4b_server.py flow)

Standard library only. The board is one resource, so every request goes
through one scheduler thread, one job at a time. A chat model runs in a
resident ``llm_worker.py`` process that keeps the weights and the program image
on the board between messages; selecting another model (or "release the
board") stops it. FLUX jobs run exactly as in flux2_klein_4b_server.py (one
process per image, or its resident ``--serve`` runner).

Every answer carries the board-side metrics: per phase (vision / audio encoder,
prefill, decode; or TE / DiT / VAE) the engine time from the on-device latency
counter, the host wall time, tokens/s, GFLOPS and the share of the raw peak
(128 FLOP/cycle/engine at the HW_INFO clock).

    python models/unified_server/unified_server.py --port 8080
    python models/unified_server/unified_server.py --host 0.0.0.0 --port 8080 --engines 8

API: GET /api/models, POST /api/chat {model, messages, image?, audio?, max_new_tokens},
POST /api/generate (FLUX, as flux2_klein_4b_server.py), GET /api/jobs/<id>,
POST /api/jobs/<id>/stop, GET /api/jobs/<id>/image.png, GET /api/jobs, POST /api/release.
"""
from __future__ import annotations

import argparse
import base64
import collections
import glob
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(MODELS_DIR)
WORKER = os.path.join(HERE, "llm_worker.py")
OUT_DIR = os.path.join(HERE, "server_out")             # uploads, FLUX images (git-ignored)
UPLOAD_DIR = os.path.join(OUT_DIR, "uploads")
MAX_PROMPT_CHARS = 8000
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
AUDIO_NATIVE = (".wav", ".flac", ".ogg")               # what soundfile reads; others go through ffmpeg
# Upload extensions: the saved file's suffix comes from these tables, never
# from the client's string (the name only selects an entry).
IMAGE_EXT = {e: e for e in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")}
AUDIO_EXT = {e: e for e in AUDIO_NATIVE + (".mp3", ".m4a", ".webm", ".aac", ".opus", ".mp4")}

sys.path.insert(0, os.path.join(MODELS_DIR, "flux2_klein_4b"))
import flux2_klein_4b_server as flux  # noqa: E402  (FLUX.2 configs, runner and job)


# ---------------------------------------------------------------------------
# Model registry and the static figures the page shows on selection
# ---------------------------------------------------------------------------

def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _json(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


CHAT_MODELS = {
    "gemma3": {"title": "Gemma 3 1B", "dir": "gemma3", "modalities": ["text"],
               "blurb": "Text chat. IF4 weights; prefill and decode N-sharded over the engines."},
    "gemma4_e2b": {"title": "Gemma 4 E2B", "dir": "gemma4_e2b", "modalities": ["text", "image", "audio"],
                   "blurb": "Text, image and audio in. Vision encoder, audio encoder, prefill and decode all on the FPGA."},
    "gemma4_e4b": {"title": "Gemma 4 E4B", "dir": "gemma4_e4b", "modalities": ["text", "image", "audio"],
                   "blurb": "Gemma 4 E2B's code with E4B's geometry (2560 wide, 42 layers, 2 KV heads)."},
}


def chat_static(model: str) -> dict:
    """Geometry, weight and program-image figures from the files on disk (no board access)."""
    spec = CHAT_MODELS[model]
    d = os.path.join(MODELS_DIR, spec["dir"])
    cfg = _json(os.path.join(d, f"{model}_config.json"))
    fi, m, paths = cfg.get("file_info", {}), cfg.get("model", {}), cfg.get("paths", {})
    out = {"layers": fi.get("num_layers"), "hidden": fi.get("hidden_size"), "vocab": fi.get("embedding_vocab"),
           "max_context": m.get("max_context_size"),
           "prefill_max": m.get("prefill_max_seq_len"), "programs": [], "notes": []}
    if model == "gemma3":
        out["layers"] = out["layers"] or len(cfg.get("layers", []) or []) or 26
        wb = os.path.join(d, "gemma3_bin", "weights_gemma3_hf.bin")
        out["weights_file"] = {"name": os.path.basename(wb), "bytes": _size(wb)}
        metas = [p for p in glob.glob(os.path.join(d, "gemma3_bin", "gemma3*_program.json")) if "profile" not in p]
        if metas:
            meta = _json(max(metas, key=os.path.getmtime))
            out["programs"] = [{"name": "Prefill", "bytes": meta.get("prefill_program_size", 0),
                                "flops": meta.get("prefill_template_flops"),
                                "flops_note": f"at {meta.get('prefill_template_seq_len')} tokens"},
                               {"name": "Decoder (one token)", "bytes": meta.get("decoder_program_size", 0),
                                "flops": meta.get("decoder_total_flops"),
                                "flops_note": f"at {meta.get('decoder_attn_aligned_seq_len')}-token KV"}]
        out["ready"] = out["weights_file"]["bytes"] > 0
        out["notes"].append(f"Prompts up to {(out['prefill_max'] or 0) + 1} tokens (one prefill program); "
                            f"older turns are dropped to fit.")
    else:
        wb = os.path.join(d, paths.get("weights_bin", f"{model}_bin/params.bin"))
        out["weights_file"] = {"name": os.path.relpath(wb, d), "bytes": _size(wb)}
        pj = _json(os.path.join(d, f"{model}_bin", "programs.json"))
        groups: dict[str, list[int]] = {}
        for name, s in pj.get("sections", {}).items():
            key = re.sub(r"\d+$", "", name) if "worker" in name else name   # decode_worker3 -> decode_worker
            groups.setdefault(key, []).append(int(s.get("size", 0)))
        label = {"lm": "LM: prefill + decoder (engine 0)", "vision": "Vision encoder (engine 0)",
                 "audio": "Audio encoder", "prefill_worker": "Prefill workers", "decode_worker": "Decode workers",
                 "vision_worker": "Vision workers"}
        for key in sorted(groups, key=lambda k: (k not in label, list(label).index(k) if k in label else 0, k)):
            sizes = groups[key]
            name = label.get(key, key) + (f" ({len(sizes)} engines)" if "worker" in key else "")
            out["programs"].append({"name": name, "bytes": sum(sizes)})
        out["ready"] = out["weights_file"]["bytes"] > 0
        if not out["ready"]:
            out["notes"].append("params.bin is not generated yet: the first load converts the checkpoint "
                                "(downloading it if needed), which takes several minutes.")
        out["notes"].append("Image / audio turns are sent on their own (the encoders build their own prompt); "
                            "text turns carry the conversation.")
    return out


# ---------------------------------------------------------------------------
# Resident chat worker
# ---------------------------------------------------------------------------

class LLMWorker:
    """A llm_worker.py process: one model on the board, JSON lines on stdout, log on stderr."""

    def __init__(self, cmd: list[str], model: str, engines: int, sink):
        self.model, self.engines = model, engines
        self.started = time.time()
        self.info: dict | None = None
        self.sink = sink                           # callable(line) -> the running job's log
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1, encoding="utf-8",
                                     errors="replace")
        threading.Thread(target=self._pump_stderr, daemon=True).start()

    def _pump_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            sink = self.sink
            if sink is not None:
                sink(line)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def send(self, obj: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def events(self):
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

    def stop(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()            # EOF ends the request loop
            self.proc.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()
            self.proc.wait()


class ChatJob:
    kind = "chat"

    def __init__(self, model: str, engines: int, messages: list[dict], image: str | None, audio: str | None,
                 max_new_tokens: int, attachments: dict):
        self.id = uuid.uuid4().hex[:12]
        self.model, self.engines = model, engines
        self.messages, self.image, self.audio = messages, image, audio
        self.max_new_tokens = max_new_tokens
        self.attachments = attachments              # names shown on the page
        self.state = "queued"                       # queued | running | done | error | cancelled
        self.status = ""
        self.text = ""
        self.metrics: dict | None = None
        self.error: str | None = None
        self.log: collections.deque[str] = collections.deque(maxlen=400)
        self.created = time.time()
        self.started: float | None = None
        self.finished: float | None = None
        self.first_token: float | None = None
        self.n_tokens = 0
        self.status_since = time.time()
        self.stop_requested = False

    def set_status(self, text: str) -> None:
        if text != self.status:
            self.status, self.status_since = text, time.time()

    def live(self) -> dict:
        """What the token stream sends besides the text: progress, no log."""
        now = time.time()
        return {"state": self.state, "status": self.status,
                "status_s": round(now - self.status_since, 1) if self.status else 0.0,
                "tokens": self.n_tokens,
                "decode_s": round(now - self.first_token, 3) if self.first_token else None,
                "elapsed": round((self.finished or now) - (self.started or now), 1) if self.started else 0.0,
                "ttft_s": round(self.first_token - self.started, 2) if (self.first_token and self.started) else None}

    def public(self) -> dict:
        now = time.time()
        return {"id": self.id, "kind": self.kind, "model": self.model, "engines": self.engines,
                "state": self.state, "status": self.status, "text": self.text, "metrics": self.metrics,
                "error": self.error, "attachments": self.attachments,
                "prompt": self.messages[-1]["content"] if self.messages else "",
                "elapsed": round((self.finished or now) - (self.started or now), 1) if self.started else 0.0,
                "ttft_s": round(self.first_token - self.started, 2) if (self.first_token and self.started) else None,
                "log": "".join(list(self.log)[-200:])}


class Board(threading.Thread):
    """The single scheduler: FLUX jobs and chat jobs, one at a time, one resident process at most."""

    def __init__(self, python: str, dev: str, engines: int):
        super().__init__(daemon=True)
        self.python, self.dev, self.engines = python, dev, engines
        self.q: "queue.Queue" = queue.Queue()
        self.jobs: dict[str, object] = {}
        self.order: list[str] = []
        self.lock = threading.Lock()
        self.flux = flux.Runner(python, None, dev)   # used for its job methods; its thread never starts
        self.llm: LLMWorker | None = None
        self.release_requested = False
        self.loading: dict | None = None             # model being brought up

    # -- bookkeeping --------------------------------------------------------
    def submit(self, job) -> None:
        with self.lock:
            self.jobs[job.id] = job
            self.order.append(job.id)
        self.q.put(job)

    def position(self, job) -> int:
        with self.lock:
            return sum(1 for j in self.jobs.values() if j.state == "queued" and j.created < job.created)

    def resident(self) -> dict | None:
        w = self.llm
        if w is not None and w.alive() and w.info is not None:
            return {"kind": "chat", "model": w.model, "engines": w.engines, "since": w.started, "info": w.info}
        fw = self.flux.worker_state()
        if fw:
            return {"kind": "image", "model": "flux2_klein_4b", "engines": fw["engines"], "since": fw["since"],
                    "loaded": fw["loaded"]}
        return None

    def stop_llm(self) -> None:
        w, self.llm = self.llm, None
        if w is not None:
            w.stop()

    def stop_all(self) -> None:
        self.stop_llm()
        self.flux.stop_worker()

    # -- chat ---------------------------------------------------------------
    def _ensure_llm(self, job: ChatJob) -> LLMWorker:
        w = self.llm
        if w is not None and (not w.alive() or w.model != job.model or w.engines != job.engines):
            self.stop_llm()
            w = None
        if w is None:
            self.flux.stop_worker()                # the device is one resource
            title = CHAT_MODELS[job.model]["title"]
            job.set_status(f"loading {title} onto the board")
            job.log.append(f"starting the resident {title} runner on {job.engines} engines "
                           f"(device, weights, program image)...\n")
            self.loading = {"model": job.model, "since": time.time()}
            cmd = [self.python, WORKER, "--model", job.model, "--engines", str(job.engines),
                                    "--dev", self.dev]
            w = LLMWorker(cmd, job.model, job.engines, job.log.append)
            try:
                for ev in w.events():
                    if ev.get("event") == "ready":
                        w.info = ev.get("info") or {}
                        break
                    if ev.get("event") == "fatal":
                        raise RuntimeError(f"{title} failed to load: {ev.get('error')}")
                if w.info is None:
                    raise RuntimeError(f"the {title} runner exited during start-up")
            except Exception:
                w.stop()
                raise
            finally:
                self.loading = None
            self.llm = w
        w.sink = job.log.append
        return w

    def _run_chat(self, job: ChatJob) -> None:
        w = self._ensure_llm(job)
        if job.stop_requested:
            raise RuntimeError("stopped")
        job.set_status("starting")
        w.send({"id": job.id, "messages": job.messages, "image": job.image, "audio": job.audio,
                "max_new_tokens": job.max_new_tokens})
        for ev in w.events():
            e = ev.get("event")
            if ev.get("id") not in (None, job.id):
                continue
            if e == "token":
                if job.first_token is None:
                    job.first_token = time.time()
                job.n_tokens += 1
                job.set_status("decode")
                job.text += ev.get("text", "")
            elif e == "status":
                job.set_status(ev.get("text", ""))
            elif e == "result":
                job.metrics = ev.get("metrics") or {}
                job.metrics["resident"] = w.info
                if job.metrics.get("text"):
                    job.text = job.metrics["text"]
                return
            elif e == "error":
                raise RuntimeError(ev.get("error") or "the request failed")
        self.stop_llm()
        raise RuntimeError("the chat runner exited during the request")

    def request_stop(self, job) -> bool:
        if getattr(job, "kind", "") != "chat":
            return False
        job.stop_requested = True
        if job.state == "queued":
            return True
        w = self.llm
        if job.state == "running" and w is not None and w.alive():
            try:
                w.send({"cmd": "stop"})
            except OSError:
                pass
        return True

    # -- FLUX (flux2_klein_4b_server.Runner's job body) ----------------------
    def _run_flux(self, job) -> None:
        self.stop_llm()
        if job.no_reload:
            self.flux._run_resident(job)
        else:
            self.flux._run_oneshot(job)
        if not os.path.exists(job.image):
            raise RuntimeError("runner finished but wrote no image")
        if job.result:
            flux.save_board(job.result)

    def run(self) -> None:
        while True:
            try:
                job = self.q.get(timeout=1.0)
            except queue.Empty:
                if self.release_requested:
                    self.release_requested = False
                    self.stop_all()
                continue
            if job.state == "cancelled" or getattr(job, "stop_requested", False):
                job.state, job.finished = "cancelled", time.time()
                continue
            job.state, job.started = "running", time.time()
            try:
                os.makedirs(OUT_DIR, exist_ok=True)
                if getattr(job, "kind", "image") == "chat":
                    self._run_chat(job)
                else:
                    self._run_flux(job)
                job.state = "done"
            except Exception as e:  # noqa: BLE001 - report anything to the page
                job.state, job.error = "error", str(e)
            finally:
                job.finished = time.time()
                if getattr(job, "kind", "") == "chat":
                    job.status = ""


def flux_public(job, board: Board) -> dict:
    d = job.public()
    d["kind"] = "image"
    d["model"] = "flux2_klein_4b"
    d["image"] = f"/api/jobs/{job.id}/image.png" if job.state == "done" else None
    d["position"] = board.position(job)
    return d


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------

def save_upload(obj, kind: str, job_tag: str) -> tuple[str, str]:
    """{"name", "data": base64 or data: URL} -> (path on disk, display name)."""
    name = os.path.basename(str(obj.get("name") or kind))
    data = str(obj.get("data") or "")
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data, validate=False)
    if not raw:
        raise ValueError(f"empty {kind}")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError(f"{kind} larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    asked = os.path.splitext(name)[1].lower()
    ext = IMAGE_EXT.get(asked, ".png") if kind == "image" else AUDIO_EXT.get(asked, ".wav")
    if kind not in ("image", "audio") or not job_tag.isalnum():
        raise ValueError("bad upload")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, f"{job_tag}_{kind}{ext}")
    with open(path, "wb") as f:
        f.write(raw)
    if kind == "audio" and ext not in AUDIO_NATIVE:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise ValueError(f"{ext} audio needs ffmpeg on the server; upload WAV, FLAC or OGG instead")
        wav = os.path.splitext(path)[0] + ".wav"
        r = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", "16000", wav],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise ValueError(f"could not decode the audio: {r.stderr.strip()[-200:]}")
        path = wav
    return path, name


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

PAGE_PATH = os.path.join(HERE, "page.html")

# The page's Markdown / math libraries, fetched once from the CDN and served from here,
# so the browser needs no internet access (the page falls back to plain text without them).
VENDOR_CDN = "https://cdn.jsdelivr.net/npm/"
VENDOR_ALLOWED = ("marked@12.0.2/marked.min.js", "dompurify@3.1.6/dist/purify.min.js", "katex@0.16.11/dist/")
VENDOR_DIR = os.path.join(OUT_DIR, "vendor")
VENDOR_TYPES = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
                ".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf"}
_vendor_lock = threading.Lock()


def vendor_file(rel: str) -> tuple[bytes, str] | None:
    """(bytes, content type) of an allow-listed CDN file, cached under server_out/vendor/."""
    ext = os.path.splitext(rel)[1]
    if (".." in rel.split("/") or ext not in VENDOR_TYPES
            or not any(rel == a or (a.endswith("/") and rel.startswith(a)) for a in VENDOR_ALLOWED)):
        return None
    root = os.path.realpath(VENDOR_DIR)
    path = os.path.realpath(os.path.join(root, *rel.split("/")))
    if not path.startswith(root + os.sep):
        return None
    with _vendor_lock:
        if not os.path.exists(path):
            import urllib.request
            try:
                with urllib.request.urlopen(VENDOR_CDN + rel, timeout=15) as r:
                    data = r.read()
            except OSError:
                return None
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".part", "wb") as f:
                f.write(data)
            os.replace(path + ".part", path)
        with open(path, "rb") as f:
            return f.read(), VENDOR_TYPES[ext]


class Handler(BaseHTTPRequestHandler):
    board: Board            # set by main()
    page: bytes = b""

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(status, json.dumps(obj).encode(), "application/json")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", "0"))
        if n > 4 * MAX_UPLOAD_BYTES:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        b = self.board
        if path == "/":
            return self._send(200, self.page, "text/html; charset=utf-8")
        if path.startswith("/vendor/"):
            got = vendor_file(path[len("/vendor/"):])
            if got is None:
                return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            self.send_response(200)
            self.send_header("Content-Type", got[1])
            self.send_header("Content-Length", str(len(got[0])))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(got[0])
            return
        if path == "/api/models":
            fboard = flux.load_board()
            fboard["weights"] = flux.weight_files()
            fboard["worker"] = b.flux.worker_state()
            models = [{"id": k, "kind": "chat", "title": v["title"], "modalities": v["modalities"],
                       "blurb": v["blurb"], "static": chat_static(k)} for k, v in CHAT_MODELS.items()]
            models.append({"id": "flux2_klein_4b", "kind": "image", "title": "FLUX.2 [klein] 4B",
                           "modalities": ["text→image"],
                           "blurb": "Text-to-image. Text encoder, diffusion transformer and VAE decoder all on the FPGA."})
            return self._json({"models": models, "engines": b.engines, "resident": b.resident(),
                               "loading": b.loading, "flux": {"configs": flux.scan_configs(), "board": fboard}})
        if path == "/api/jobs":
            with b.lock:
                jobs = [b.jobs[i] for i in b.order]
            out = []
            for j in jobs:
                d = flux_public(j, b) if getattr(j, "kind", "image") != "chat" else j.public()
                d.pop("log", None)
                out.append(d)
            return self._json(out)
        if path.startswith("/api/jobs/") and path.endswith("/stream"):
            job = b.jobs.get(path[len("/api/jobs/"):-len("/stream")])
            if job is None or getattr(job, "kind", "image") != "chat":
                return self._json({"error": "no such chat job"}, HTTPStatus.NOT_FOUND)
            return self._stream(job)
        if path.startswith("/api/jobs/"):
            parts = path[len("/api/jobs/"):].split("/")
            job = b.jobs.get(parts[0])
            if job is None:
                return self._json({"error": "no such job"}, HTTPStatus.NOT_FOUND)
            if len(parts) == 1:
                if getattr(job, "kind", "image") == "chat":
                    d = job.public()
                    d["position"] = b.position(job)
                    return self._json(d)
                return self._json(flux_public(job, b))
            if parts[1] == "image.png" and getattr(job, "kind", "image") != "chat" and job.state == "done":
                with open(job.image, "rb") as f:
                    return self._send(200, f.read(), "image/png")
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _stream(self, job: "ChatJob") -> None:
        """Server-sent events: 'delta' {text, ...progress} as tokens arrive, 'done' with the whole job."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def send(event: str, data: dict) -> None:
            self.wfile.write(f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()

        prev, last_beat = "", 0.0
        try:
            while True:
                finished = job.state in ("done", "error", "cancelled")
                text = job.text
                now = time.time()
                if text != prev or now - last_beat > 0.5:
                    d = job.live()
                    if text.startswith(prev):
                        d["text"] = text[len(prev):]
                    else:                            # the final text replaced the streamed one
                        d["text"], d["reset"] = text, True
                    d["position"] = self.board.position(job) if job.state == "queued" else 0
                    send("delta", d)
                    prev, last_beat = text, now
                if finished:
                    send("done", job.public())
                    return
                time.sleep(0.02)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_POST(self) -> None:
        b = self.board
        path = self.path.split("?", 1)[0]
        if path == "/api/release":
            b.release_requested = True              # the scheduler stops it between jobs
            return self._json({"ok": True}, HTTPStatus.ACCEPTED)
        if path.startswith("/api/jobs/") and path.endswith("/stop"):
            job = b.jobs.get(path[len("/api/jobs/"):-len("/stop")])
            if job is None:
                return self._json({"error": "no such job"}, HTTPStatus.NOT_FOUND)
            return self._json({"ok": b.request_stop(job)})
        try:
            body = self._body()
        except (ValueError, json.JSONDecodeError) as e:
            return self._json({"error": f"bad request: {e}"}, HTTPStatus.BAD_REQUEST)
        if path == "/api/chat":
            return self._chat(body)
        if path == "/api/generate":
            return self._flux(body)
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _chat(self, body: dict) -> None:
        model = body.get("model")
        if model not in CHAT_MODELS:
            return self._json({"error": "unknown chat model"}, HTTPStatus.BAD_REQUEST)
        msgs = []
        for m in body.get("messages") or []:
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            msgs.append({"role": role, "content": str(m.get("content", ""))})
        if not msgs or msgs[-1]["role"] != "user":
            return self._json({"error": "the last message must be the user's"}, HTTPStatus.BAD_REQUEST)
        if sum(len(m["content"]) for m in msgs) > MAX_PROMPT_CHARS:
            return self._json({"error": f"conversation longer than {MAX_PROMPT_CHARS} characters"},
                              HTTPStatus.BAD_REQUEST)
        try:
            max_new = int(body.get("max_new_tokens") or 256)
        except (TypeError, ValueError):
            return self._json({"error": "max_new_tokens must be an integer"}, HTTPStatus.BAD_REQUEST)
        if not 1 <= max_new <= 8192:
            return self._json({"error": "max_new_tokens must be in [1, 8192]"}, HTTPStatus.BAD_REQUEST)
        mods = CHAT_MODELS[model]["modalities"]
        tag = uuid.uuid4().hex[:12]
        image = audio = None
        att = {}
        try:
            if body.get("image"):
                if "image" not in mods:
                    raise ValueError(f"{CHAT_MODELS[model]['title']} takes no images")
                image, att["image"] = save_upload(body["image"], "image", tag)
            if body.get("audio"):
                if "audio" not in mods:
                    raise ValueError(f"{CHAT_MODELS[model]['title']} takes no audio")
                audio, att["audio"] = save_upload(body["audio"], "audio", tag)
        except (ValueError, TypeError, base64.binascii.Error) as e:
            return self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
        if (image or audio) and not msgs[-1]["content"].strip():
            msgs[-1]["content"] = "Describe this image." if image and not audio else (
                "Transcribe the speech exactly." if audio and not image else "Describe what you see and hear.")
        if not msgs[-1]["content"].strip():
            return self._json({"error": "empty message"}, HTTPStatus.BAD_REQUEST)
        if image or audio:
            msgs = msgs[-1:]                        # the encoders build a single-turn prompt
        job = ChatJob(model, self.board.engines, msgs, image, audio, max_new, att)
        self.board.submit(job)
        self._json({"id": job.id, "position": self.board.position(job)}, HTTPStatus.ACCEPTED)

    def _flux(self, body: dict) -> None:
        try:
            prompt = str(body.get("prompt", "")).strip()
            seed = int(body.get("seed", 0))
            no_reload = bool(body.get("no_reload", False))
            cfg = next((c for c in flux.scan_configs() if c["id"] == body.get("config")), None)
        except (ValueError, TypeError) as e:
            return self._json({"error": f"bad request: {e}"}, HTTPStatus.BAD_REQUEST)
        if not prompt:
            return self._json({"error": "empty prompt"}, HTTPStatus.BAD_REQUEST)
        if len(prompt) > flux.MAX_PROMPT_CHARS:
            return self._json({"error": f"prompt longer than {flux.MAX_PROMPT_CHARS} characters"},
                              HTTPStatus.BAD_REQUEST)
        if cfg is None:
            return self._json({"error": "unknown configuration"}, HTTPStatus.BAD_REQUEST)
        if seed < 0:
            return self._json({"error": "seed must be >= 0"}, HTTPStatus.BAD_REQUEST)
        job = flux.Job(prompt, cfg, seed, no_reload)
        job.kind = "image"
        self.board.submit(job)
        self._json({"id": job.id, "position": self.board.position(job)}, HTTPStatus.ACCEPTED)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Unified web UI for the models on the board")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 to reach it from other machines)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--python", default=sys.executable, help="interpreter for the model runners")
    ap.add_argument("--dev", default="xdma0")
    ap.add_argument("--engines", type=int, default=8, help="engines for the chat models (FLUX uses its configs')")
    ap.add_argument("--no-reload", action="store_true",
                    help="FLUX: tick 'keep the board loaded between runs' by default")
    args = ap.parse_args(argv)

    board = Board(args.python, args.dev, args.engines)
    board.start()
    Handler.board = board
    with open(PAGE_PATH, encoding="utf-8") as f:
        Handler.page = f.read().replace("__NO_RELOAD__", "checked" if args.no_reload else "").encode()
    for k in CHAT_MODELS:
        s = chat_static(k)
        print(f"  {k}: weights {s['weights_file']['bytes'] / 1e9:.2f} GB"
              f"{'' if s['ready'] else ' (not generated yet)'}, context {s['max_context']}", file=sys.stderr)
    print(f"  flux2_klein_4b: {len(flux.scan_configs())} compiled program sets", file=sys.stderr)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"Unified Engine model server on http://{args.host}:{args.port}/  (runners: {args.python}, "
          f"{args.engines} engines for chat)", file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        board.stop_all()                            # free the device
    return 0


if __name__ == "__main__":
    sys.exit(main())
