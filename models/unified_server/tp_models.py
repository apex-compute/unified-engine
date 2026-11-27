"""Unified-server catalog and chat adapter for the five-card FPGA runners."""
import importlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "apex-tp-llm"
# id -> display name, runner module, slice set, context limit
MODELS = {
    "q35_9b": ("Qwen3.5 9B", "run_tp", "10fp4", 16384),
    "q38_27b": ("Qwen3.8 27B", "run_tp", "10fp4", 8192),
    "gptoss20b": ("GPT-OSS 20B", "gptoss_tp", "10", 8192),
    "gemma4_26b": ("Gemma 4 26B-A4B", "gemma4_tp", "10", 8192),
}


def work_dir(model):
    default = Path(os.environ.get("APEX_TP_WORK", "~/apex_tp")).expanduser() / "models" / model
    override = {"gptoss20b": "GPTOSS_WORK", "gemma4_26b": "GEMMA4_WORK"}.get(model)
    return Path(os.environ.get(override, default) if override else default).expanduser()


def plan_file(model):
    return work_dir(model) / ("tp" + MODELS[model][2]) / "plan.json"


def static(model):
    path = plan_file(model)
    try:
        plan = json.loads(path.read_text())
    except (OSError, ValueError):
        plan = {}
    if not isinstance(plan, dict):
        plan = {}
    ready = plan.get("P") == 10
    return {"layers": plan.get("layers"), "hidden": plan.get("hidden"), "max_context": MODELS[model][3],
            "engines": 10, "programs": [], "ready": ready, "weights_file": {"name": str(path.parent), "bytes": None},
            "notes": ["Five cards, two engines per card. Weights stay resident between messages.",
                      "Ten-shard plan found." if ready else f"Prepare ten weight shards and the tokenizer. Expected {path}."]}


def run(model, devices, serve):
    title, module, slices, context = MODELS[model]
    path = plan_file(model)
    if not path.is_file():
        raise FileNotFoundError(f"{title} requires prepared weights at {path}. See models/apex-tp-llm/README.md.")
    plan = json.loads(path.read_text())
    if not isinstance(plan, dict) or plan.get("P") != 10:
        raise ValueError(f"{title} requires a ten-shard weight plan at {path}")
    sys.path[:0] = [str(ROOT), str(ROOT / "gemma4")]
    if module == "run_tp":
        os.environ["Q_MODEL"] = model
        if model == "q38_27b":
            os.environ.setdefault("APEX_TP_PARAM_FACTOR", "1.0")
            os.environ.setdefault("APEX_TP_PARAM_SLACK_MB", "64")
    elif model == "gptoss20b":
        os.environ["GPTOSS_WORK"] = str(work_dir(model))
    else:
        if (work_dir(model) / "vision/vision.pt").is_file():
            os.environ.setdefault("GEMMA4_VISION_SHARDS", "8")
            os.environ.setdefault("GEMMA4_MAX_BLOCK", "288")
    runner = importlib.import_module(module)
    available = runner.cards_in_pci_order()
    selected = [d.strip() for d in devices.split(",")] if devices else available[:5]
    if len(selected) != 5 or len(set(selected)) != 5 or not set(selected) <= set(available):
        raise ValueError(f"{title} needs five distinct XDMA cards. Visible: {', '.join(available) or 'none'}.")
    return runner.main(["--tp", slices, "--engines", "2", "--max-context", str(context),
                        "--devs", ",".join(selected)], serve=serve)


class Answer:
    """Stream visible text only, including GPT-OSS Harmony's final channel."""
    def __init__(self, tokenizer, model):
        self.tok = tokenizer
        self.model = model
        self.mode = "header" if model == "gptoss20b" else "text"
        self.channel = []
        self.ids = []
        self.emitted = ""
        self.answer = ""

    def push(self, token):
        piece = self.tok.convert_ids_to_tokens(token)
        if self.model == "gptoss20b":
            if piece == "<|channel|>":
                self.mode, self.channel = "channel", []
                return ""
            if piece == "<|message|>":
                self.mode = "text" if self.tok.decode(self.channel).strip() == "final" else "hidden"
                self.ids, self.emitted = [], ""
                return ""
            if piece in ("<|end|>", "<|start|>"):
                self.mode = "header"
                return ""
            if self.mode == "channel":
                self.channel.append(token)
                return ""
        elif self.model == "gemma4_26b" and piece in ("<|channel>", "<channel|>"):
            self.mode = "hidden" if piece == "<|channel>" else "text"
            return ""
        if self.mode != "text":
            return ""
        self.ids.append(token)
        text = self.tok.decode(self.ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if text.endswith("\ufffd"):
            return ""
        delta = text[len(self.emitted):]
        self.emitted = text
        self.answer += delta
        return delta


class Chat:
    def __init__(self, model, stop, status, *, tokenizer, reset, park, step, stops, devices, cards, load_s,
                 encode_image=None, block_step=None, positions=None):
        self.model, self.stop, self.status = model, stop, status
        self.tok, self.reset, self.step, self.stops = tokenizer, reset, step, stops
        self.park = park
        self.park()
        self.encode_image, self.block_step = encode_image, block_step
        self.positions = positions
        self.limit = MODELS[model][3]
        self.last_chat, self.start = None, 0
        self.history, self.images = [], {}
        self.stats = {"engines": len(cards), "devices": devices, "hw_info": ", ".join(devices),
                      "load_s": load_s,
                      "weights_dram_bytes": int(sum(c["params_mb"] for c in cards) * 1e6)}

    def info(self):
        return self.stats

    def prompt(self, messages, images, start):
        turns = [dict(m) for m in messages[start:]]
        rows = [img for i, img in sorted(images.items()) if i >= start]
        for i in images:
            if i >= start:
                m = turns[i - start]
                caption = m["content"] if m["content"].strip() else "Describe this image."
                m["content"] = [{"type": "image"}, {"type": "text", "text": caption}]
        opts = {"reasoning_effort": "low"} if self.model == "gptoss20b" else {"enable_thinking": False}
        text = self.tok.apply_chat_template(turns, tokenize=False, add_generation_prompt=True, **opts)
        ids = self.tok.encode(text, add_special_tokens=self.model.startswith("q3"))
        if not rows:
            return ids, {}, []
        qwen = self.positions is not None
        marker = self.tok.convert_tokens_to_ids("<|image_pad|>" if qwen else "<|image|>")
        if ids.count(marker) != len(rows):
            raise ValueError("The image placeholders do not match the attached images")
        if not qwen:
            begin, end = [self.tok.convert_tokens_to_ids(s) for s in ("<|image>", "<image|>")]
        out, blocks, grids = [], {}, []
        pending = iter(rows)
        for token in ids:
            if token != marker:
                out.append(token)
                continue
            image = next(pending)
            if qwen:
                image, (gh, gw) = image
                if len(image) != gh * gw:
                    raise RuntimeError("Qwen image dimensions do not match the encoded rows")
                grids.append((len(out), gh, gw))
            else:
                out.append(begin)
            blocks[len(out)] = image
            out.extend([marker] * len(image))
            if not qwen:
                out.append(end)
        return out, blocks, grids

    def generate(self, req, on_token):
        try:
            result = self._generate(req, on_token)
        except ValueError:
            self.park()
            raise
        self.park()
        return result

    def _generate(self, req, on_token):
        started = time.monotonic()
        messages = req["messages"]
        chat = req.get("conversation")
        continuation = chat and chat == self.last_chat and self.history == messages[:-1]
        images = dict(self.images) if continuation else {}
        self.reset()
        if req.get("image"):
            if self.encode_image is None:
                raise ValueError("Prepare the vision weights for this model before attaching an image.")
            self.status("vision encoder")
            images[len(messages) - 1] = self.encode_image(req["image"])
        start = self.start if continuation else 0
        while True:
            ids, blocks, grids = self.prompt(messages, images, start)
            if 0 < len(ids) < self.limit:
                break
            if start + 2 >= len(messages):
                raise ValueError(f"The prompt exceeds the {self.limit}-token context")
            start += 2
        if self.positions:
            self.positions(len(ids), grids)
        self.status("prefill")
        pos, nxt = 0, None
        while pos < len(ids) and not self.stop.is_set():
            if pos in blocks:
                block = blocks[pos]
                self.block_step(block, pos)
                pos += len(block)
            else:
                nxt = self.step(ids[pos], pos, False)
                pos += 1
        prefilled = time.monotonic()
        answer, count = Answer(self.tok, self.model), 0
        limit = min(int(req["max_new_tokens"]), self.limit - len(ids))
        while count < limit and not self.stop.is_set():
            if count:
                nxt = self.step(nxt, len(ids) + count - 1, False)
            count += 1
            ended = nxt in self.stops
            if on_token(nxt, "" if ended else answer.push(nxt)) or ended:
                break
        end = time.monotonic()
        decode_s = end - prefilled
        self.history = [*messages, {"role": "assistant", "content": answer.answer}]
        self.images = {k: v for k, v in images.items() if k >= start}
        self.last_chat, self.start = chat, start
        return {"text": answer.answer, "prompt_tokens": len(ids), "generated_tokens": count,
                "image_tokens": sum(len(v) for v in blocks.values()), "history_dropped": start // 2,
                "context_tokens": pos + max(0, count - 1), "max_context": self.limit,
                "ttft_wall_s": prefilled - started if count else None, "wall_s": end - started,
                "decode_wall_tok_s": (count - 1) / decode_s if count > 1 and decode_s > 0 else None,
                "phases": []}
