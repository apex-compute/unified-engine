#!/usr/bin/env python3
"""Resident chat runner for the unified web server: one Gemma model on the board.

Loads one model once (device, weights to DRAM, program image), then answers
chat requests read as JSON lines on stdin until EOF. Everything the model code
prints goes to stderr (the server shows it as the run log); this script's own
protocol goes to the original stdout as JSON lines:

    {"event": "ready", "info": {...}}          once the model is on the board
    {"event": "token", "id": ..., "text": "..."}  one per generated token
    {"event": "status", "id": ..., "text": "..."} phase changes (encoders, prefill, ...)
    {"event": "result", "id": ..., "metrics": {...}}
    {"event": "error", "id": ..., "error": "..."}

Requests:

    {"id": "...", "messages": [{"role": "user"|"assistant", "content": "..."}, ...],
     "image": "/path.jpg" | null, "audio": "/path.wav" | null, "max_new_tokens": 256}
    {"cmd": "stop"}      stops the generation in flight after the current token

    python models/unified_server/llm_worker.py --model gemma3 --engines 8 --dev xdma0
    python models/unified_server/llm_worker.py --model gemma4_e2b --engines 8
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.dirname(HERE)
ROOT = os.path.dirname(MODELS)

# Protocol channel = the original stdout; every other write (the model code's
# prints, including the decoded text it echoes without newlines) goes to stderr.
_PROTO = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
os.dup2(2, 1)
sys.stdout = sys.stderr
_PROTO_LOCK = threading.Lock()


def emit(event: str, **kw) -> None:
    with _PROTO_LOCK:
        _PROTO.write(json.dumps({"event": event, **kw}, ensure_ascii=False) + "\n")
        _PROTO.flush()


def log(msg: str) -> None:
    sys.stderr.write(f"[worker] {msg}\n")
    sys.stderr.flush()


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _peak(clock_mhz: float, engines: int) -> float:
    """Raw peak GFLOPS: 128 FLOP/cycle/engine."""
    return clock_mhz * 0.128 * engines


def _phase(name, *, tokens=None, hw_s=None, wall_s=None, flops=None, gflops=None, engines=1,
           clock_mhz=0.0, note=None) -> dict:
    if gflops is None and flops and hw_s:
        gflops = flops / hw_s / 1e9
    peak = _peak(clock_mhz, engines)
    return {"name": name, "tokens": tokens, "hw_s": hw_s, "wall_s": wall_s,
            "tok_s": (tokens / hw_s) if (tokens and hw_s) else None,
            "flops": flops, "gflops": gflops, "engines": engines, "peak_gflops": peak,
            "utilization": (gflops / peak) if (gflops and peak) else None, "note": note}


# ---------------------------------------------------------------------------
# Gemma 3 1B (text)
# ---------------------------------------------------------------------------

class Gemma3Backend:
    title = "Gemma 3 1B"
    modalities = ("text",)

    def __init__(self, engines: int, dev: str):
        sys.path.insert(0, os.path.join(MODELS, "gemma3"))
        sys.path.insert(0, ROOT)
        import user_dma_core
        import gemma3_test as g3
        self.g3, self.udc, self.engines = g3, user_dma_core, engines
        user_dma_core.set_dma_device(dev)
        for attr in ("DMA_DEVICE_H2C", "DMA_DEVICE_C2H", "DMA_DEVICE_USER"):
            setattr(g3, attr, getattr(user_dma_core, attr))
        user_dma_core.configure_clock_from_hardware()
        log(user_dma_core.hardware_info_summary())
        from multi_engine_decode import reset_engine_queues
        reset_engine_queues(engines)
        user_dma_core.UnifiedEngine(BASE_ADDR=user_dma_core.UE_0_BASE_ADDR).clear_dram()
        t0 = time.perf_counter()
        ue = g3.Gemma3_UnifiedEngine(multi_core=engines)
        ue.set_prefill_seq(None)
        ue.setup_multi_core()
        self.load_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        ue.compile_gemma3(bin_reuse=True)
        self.compile_s = time.perf_counter() - t0
        self.program_cursor = ue._next_program_dram_addr   # every request loads the image here
        self.ue = ue
        self.clock_mhz = user_dma_core.configured_hardware_info().frequency_mhz

    def info(self) -> dict:
        ue = self.ue
        return {"model": "gemma3", "title": self.title, "engines": self.engines,
                "clock_mhz": self.clock_mhz, "peak_gflops": _peak(self.clock_mhz, self.engines),
                "hw_info": self.udc.hardware_info_summary(),
                "max_context": int(ue.MAX_CONTEXT_SIZE), "max_prompt_tokens": int(ue.PREFILL_MAX_SEQ_LEN) + 1,
                "weights_dram_bytes": int(ue.get_params_dram_usage()),
                "load_s": self.load_s, "compile_s": self.compile_s, "modalities": list(self.modalities)}

    def _encode(self, messages: list[dict]) -> tuple[list[int], int]:
        """Chat-template the conversation (the same encode gemma3_test.set_prefill_seq does),
        dropping the oldest turns until it fits the prefill program. Returns (ids, dropped)."""
        tok, limit = self.ue.tokenizer, int(self.ue.PREFILL_MAX_SEQ_LEN) + 1
        msgs = list(messages)
        dropped = 0
        while True:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = list(tok.encode(text, add_special_tokens=True))
            if len(ids) <= limit or len(msgs) <= 1:
                break
            msgs = msgs[2:] if len(msgs) > 2 else msgs[1:]
            dropped += 1
        if len(ids) > limit:
            raise ValueError(f"the prompt is {len(ids)} tokens; Gemma 3's prefill program takes at most {limit}")
        if len(ids) < 2:
            raise ValueError("empty prompt")
        return ids, dropped

    def generate(self, req: dict, on_token) -> dict:
        if req.get("image") or req.get("audio"):
            raise ValueError("Gemma 3 1B is text only")
        ue = self.ue
        ids, dropped = self._encode(req["messages"])
        ue.prefill_seq = tuple(ids)
        room = int(ue.MAX_CONTEXT_SIZE) - (len(ids) - 1)
        ue.max_new_tokens = max(1, min(int(req.get("max_new_tokens") or room), room))
        ue.on_token = on_token
        ue._next_program_dram_addr = self.program_cursor
        emit("status", id=req["id"], text="prefill")
        t0 = time.perf_counter()
        ctx = ue.run_gemma3_prefill()
        emit("status", id=req["id"], text="decode")
        r = ue.run_gemma3_decode(ctx)
        wall = time.perf_counter() - t0
        ue.on_token = None
        n = int(r["tokens_decoded"])
        dec_hw = r["decode_avg_hw_ms"] * n / 1e3
        dec_wall = (r["decode_avg_cpu_ms"] * n / 1e3) if n else None
        c = self.clock_mhz
        phases = [
            _phase("Prefill", tokens=r["prefill_tokens"], hw_s=r["prefill_hw_ms"] / 1e3,
                   wall_s=r["prefill_cpu_ms"] / 1e3, gflops=r.get("prefill_gflops"), engines=1, clock_mhz=c,
                   note="engine 0 runs prefill"),
            _phase("Decode", tokens=n, hw_s=dec_hw, wall_s=dec_wall, gflops=r.get("decoder_gflops") or None,
                   engines=self.engines, clock_mhz=c, note=f"N-sharded over {self.engines} engines"),
        ]
        return {
            "prompt_tokens": len(ids), "generated_tokens": n, "history_dropped": dropped,
            "phases": phases,
            "ttft_hw_s": r["prefill_hw_ms"] / 1e3,          # as Gemma 4's run_result: up to the prefill
            "ttft_wall_s": r["prefill_cpu_ms"] / 1e3,
            "decode_first_tok_s": r.get("peak_tokens_per_s"),
            "decode_hw_tok_s": (n / dec_hw) if dec_hw else None,
            "decode_wall_tok_s": r.get("avg_tokens_per_s"),
            "compile_s": 0.0, "wall_s": wall, "final_seq_len": r.get("total_tokens"),
            "text": r.get("decoded_text", ""),
        }


# ---------------------------------------------------------------------------
# Gemma 4 E2B / E4B (text, image, audio)
# ---------------------------------------------------------------------------

_G4_PER_REQUEST = ("_media_counts", "_vis_last_latency_us", "_vis_e2e_s", "_vis_frames", "_vis_num_soft_tokens",
                   "_audio_hw_latency_us", "_audio_e2e_s", "_audio_seconds",
                   "_image_features", "_audio_features", "_mm_types", "_prefill_latency_hw_us",
                   "_prefill_seq_len", "_decode_generated_n", "_decode_hw_latency_us", "_decoded_text")


SHORT_PROMPT_SYSTEM = "You are a helpful assistant."


class Gemma4Backend:
    modalities = ("text", "image", "audio")

    def __init__(self, variant: str, engines: int, dev: str):
        e2b = os.path.join(MODELS, "gemma4_e2b")
        sys.path.insert(0, e2b)
        sys.path.insert(0, ROOT)
        import gemma4_e2b_test as g4
        if variant == "gemma4_e4b":
            g4.set_model_variant("gemma4_e4b", os.path.join(MODELS, "gemma4_e4b"), "Gemma4 E4B")
            self.title = "Gemma 4 E4B"
        else:
            self.title = "Gemma 4 E2B"
        import user_dma_core
        self.g4, self.udc, self.variant, self.engines = g4, user_dma_core, variant, engines
        parser = g4.build_arg_parser()
        args = parser.parse_args(["--dev", dev, "--multi-core", str(engines), "--bin-reuse"])
        kwargs = g4.resolve_engine_config(parser, args)
        from user_hw_test import software_reset_test
        software_reset_test(cores=engines)
        g4.poison_dram("zero", None, quiet=True)        # what gemma4_e2b_test.main() does before the model
        t0 = time.perf_counter()
        ue = g4.Gemma4_UnifiedEngine(**kwargs)
        self.load_s = time.perf_counter() - t0
        ue._dram_read_speed_mbps = None
        # Build (or reuse) the program image once at start-up for a short prompt.
        ue.prefill_seq = tuple(self._template_on(ue, [{"role": "user", "content": "Write a haiku about the ocean."}]))
        # Every compile places its programs at the engines' current program cursors
        # and advances them; a resident runner recompiles per prompt length, so it
        # rewinds them to where the first compile started (as prepare_prefill_chunks
        # does for later chunks) -- otherwise they walk off the ISA windows.
        self._cursors = self._program_cursors(ue)
        t0 = time.perf_counter()
        ue.compile_gemma4()
        ue.prepare_prefill_chunks()
        self.compile_s = time.perf_counter() - t0
        self.ue = ue
        self.clock_mhz = 1000.0 / ue._clock_period_ns

    @staticmethod
    def _program_cursors(ue) -> list[tuple[object, int]]:
        engines = [ue]
        for get in ("_ensure_prefill_scheduler", "_ensure_lm_scheduler"):
            sched = getattr(ue, get)() if hasattr(ue, get) else None
            for w in (getattr(sched, "workers", None) or []):
                if all(w is not e for e in engines):
                    engines.append(w)
        return [(e, int(e._next_program_dram_addr)) for e in engines]

    def _rewind_cursors(self) -> None:
        for e, addr in self._cursors:
            e._next_program_dram_addr = addr

    @staticmethod
    def _template_on(ue, msgs):
        text = ue.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return ue.tokenizer.encode(text, add_special_tokens=True)

    def info(self) -> dict:
        ue = self.ue
        return {"model": self.variant, "title": self.title, "engines": self.engines,
                "clock_mhz": self.clock_mhz, "peak_gflops": _peak(self.clock_mhz, self.engines),
                "hw_info": self.udc.hardware_info_summary(),
                "max_context": int(ue.MAX_CONTEXT_SIZE), "max_prompt_tokens": int(ue.MAX_CONTEXT_SIZE) - 1,
                "prefill_chunk_tokens": int(ue.max_prefill_seq_len),
                "weights_dram_bytes": int(ue.get_params_dram_usage()),
                "load_s": self.load_s, "compile_s": self.compile_s, "modalities": list(self.modalities)}

    def _encode_text(self, messages: list[dict], max_new: int) -> tuple[list[int], int]:
        tok, ue = self.ue.tokenizer, self.ue
        limit = int(ue.MAX_CONTEXT_SIZE) - max_new
        msgs, dropped = list(messages), 0
        while True:
            ids = self._template(msgs)
            if len(ids) <= limit or len(msgs) <= 1:
                break
            msgs = msgs[2:] if len(msgs) > 2 else msgs[1:]
            dropped += 1
        if len(ids) - 1 < self.engines:
            # Multi-engine prefill shards the prompt rows: it needs one per engine,
            # so a very short message ("hi") gets a one-line system preamble.
            ids = self._template([{"role": "system", "content": SHORT_PROMPT_SYSTEM}] + msgs)
        return ids, dropped

    def _template(self, msgs: list[dict]) -> list[int]:
        tok = self.ue.tokenizer
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return list(tok.encode(text, add_special_tokens=True))   # as Gemma4LMMixin.set_prefill_seq

    def generate(self, req: dict, on_token) -> dict:
        ue, rid = self.ue, req["id"]
        for name in _G4_PER_REQUEST:
            setattr(ue, name, None)
        max_new = max(1, int(req.get("max_new_tokens") or 256))
        image, audio = req.get("image"), req.get("audio")
        prompt = req["messages"][-1]["content"] if req["messages"] else ""
        dropped = 0
        t_req = time.perf_counter()
        if image and audio:
            emit("status", id=rid, text="vision + audio encoders")
            ue.set_prefill_seq_media(image_path=image, audio_path=audio, prompt=prompt or None)
        elif image:
            emit("status", id=rid, text="vision encoder")
            ue.set_prefill_seq_vlm(image, prompt or None)
        elif audio:
            emit("status", id=rid, text="audio encoder")
            ue.set_prefill_seq_audio(audio, prompt or None)
        else:
            ids, dropped = self._encode_text(req["messages"], max_new)
            ue.prefill_seq = tuple(ids)
        prompt_tokens = len(ue.prefill_seq)
        mm = list(getattr(ue, "_mm_types", None) or [])          # prefill clears it; count now
        n_img, n_aud = sum(t == 1 for t in mm), sum(t == 3 for t in mm)
        room = int(ue.MAX_CONTEXT_SIZE) - prompt_tokens
        if room < 1:
            raise ValueError(f"the prompt is {prompt_tokens} tokens; the context holds {ue.MAX_CONTEXT_SIZE}")
        ue.max_new_tokens = min(max_new, room)
        # The prefill program's FLOP accounting (and the chunk offsets) follow the prompt
        # length, so a new length recompiles the LM section in-process (weights stay put).
        emit("status", id=rid, text="program image")
        t0 = time.perf_counter()
        self._rewind_cursors()
        ue.compile_gemma4()
        ue.prepare_prefill_chunks()
        compile_s = time.perf_counter() - t0
        reused = bool(getattr(ue, "_lm_compile_reused", False))
        emit("status", id=rid, text="prefill")
        ue.on_token = on_token
        try:
            ue.run_gemma4()
        finally:
            ue.on_token = None
        wall = time.perf_counter() - t_req
        r = ue.run_result()
        c, e = self.clock_mhz, self.engines
        phases = []
        if r.get("vision_hw_s"):
            vf = ue._model_flops_vision()
            phases.append(_phase(f"Vision encoder ({r.get('image_frames') or 1} image)", tokens=r.get("image_tokens") or n_img,
                                 hw_s=r["vision_hw_s"], wall_s=r.get("vision_wall_s"), flops=vf, engines=e,
                                 clock_mhz=c, note="soft tokens out"))
        if r.get("audio_hw_s"):
            secs = r.get("audio_seconds")
            phases.append(_phase("Audio encoder" + (f" ({secs:.1f} s clip)" if secs else ""),
                                 tokens=r.get("audio_tokens") or n_aud, hw_s=r["audio_hw_s"], wall_s=r.get("audio_wall_s"),
                                 flops=ue._model_flops_audio(), engines=e, clock_mhz=c,
                                 note="frames split across the engines; soft tokens out"))
        pf_model = ue._model_flops_prefill()
        chunks = r.get("prefill_chunks") or 1
        phases.append(_phase("Prefill" + (f" ({chunks} chunks)" if chunks > 1 else ""),
                             tokens=r.get("prefill_tokens"), hw_s=r.get("prefill_hw_s"),
                             wall_s=r.get("prefill_wall_s"), flops=pf_model, gflops=r.get("prefill_gflops_model"),
                             engines=e, clock_mhz=c, note="model FLOPs"))
        n = r.get("decode_tokens") or 0
        dec_us = getattr(ue, "_decode_hw_latency_us", None)
        dec_model = ue._model_flops_decode(n, ue.seq_len) if n else None
        phases.append(_phase("Decode", tokens=n, hw_s=(dec_us / 1e6) if dec_us else None,
                             wall_s=getattr(ue, "_decode_e2e_s", None), flops=dec_model,
                             gflops=r.get("decode_gflops_model"), engines=e, clock_mhz=c, note="model FLOPs"))
        text = r.get("decoded_text") or ""
        return {
            "prompt_tokens": prompt_tokens, "generated_tokens": n, "history_dropped": dropped,
            "image_tokens": r.get("image_tokens") or n_img, "audio_tokens": r.get("audio_tokens") or n_aud,
            "phases": phases,
            "ttft_hw_s": r.get("ttft_hw_s"), "ttft_wall_s": r.get("ttft_wall_s"),
            "decode_first_tok_s": r.get("decode_first_token_tok_s"),
            "decode_hw_tok_s": r.get("decode_hw_tok_s"), "decode_wall_tok_s": r.get("decode_avg_tok_s"),
            "compile_s": compile_s, "compile_reused": reused, "wall_s": wall,
            "final_seq_len": r.get("final_seq_len"), "text": text,
        }


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=("gemma3", "gemma4_e2b", "gemma4_e4b"))
    ap.add_argument("--engines", type=int, default=8)
    ap.add_argument("--dev", default="xdma0")
    args = ap.parse_args(argv)
    os.chdir(ROOT)

    try:
        if args.model == "gemma3":
            backend = Gemma3Backend(args.engines, args.dev)
        else:
            backend = Gemma4Backend(args.model, args.engines, args.dev)
    except BaseException as e:  # noqa: BLE001 - report start-up failures to the server
        traceback.print_exc()
        emit("fatal", error=f"{type(e).__name__}: {e}")
        return 1
    emit("ready", info=backend.info())

    requests: "queue.Queue[dict | None]" = queue.Queue()
    stop = threading.Event()

    def reader() -> None:                       # stdin -> requests; "stop" acts at once
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("cmd") == "stop":
                stop.set()
            else:
                requests.put(msg)
        requests.put(None)

    threading.Thread(target=reader, daemon=True).start()
    while True:
        req = requests.get()
        if req is None:
            break
        stop.clear()
        rid = req.get("id")

        def on_token(token_id, text, rid=rid):
            emit("token", id=rid, text=text)
            return stop.is_set()

        try:
            metrics = backend.generate(req, on_token)
            metrics["stopped"] = stop.is_set()
            emit("result", id=rid, metrics=metrics)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            emit("error", id=rid, error=f"{type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
