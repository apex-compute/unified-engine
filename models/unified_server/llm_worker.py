#!/usr/bin/env python3
"""Resident chat runner for the unified web server: one chat model on the board.

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
     "image": "/path.jpg" | null, "audio": "/path.wav" | null, "max_new_tokens": 256,
     "conversation": "<page chat id>"}
    {"cmd": "stop"}      stops the generation in flight after the current token

    python models/unified_server/llm_worker.py --model gemma3 --engines 8 --dev xdma0
    python models/unified_server/llm_worker.py --model gemma4_e2b --engines 8
    python models/unified_server/llm_worker.py --model qwen2.5_omni_7b --engines 8
"""
from __future__ import annotations

import argparse
import collections
import copy
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

class _Conversation:
    """The K/V cache keeps the conversation (shared by the Gemma backends).

    After a turn the cache holds positions [0, self.cached) (prompts + fed
    answers); the answer's last sampled token (self.unfed) was never fed back.
    The next message of the same conversation (same page conversation id, one
    new user message) prefills only [unfed, end-of-turn?, "\n", new user turn]
    at offset self.cached, on top of the cache. New chat, another model, or a
    full context starts over.
    """
    conv_id: str | None = None
    cached: int = 0
    unfed: int | None = None
    turns: int = 0                     # messages the cache covers (user + assistant)

    def _forget(self) -> None:
        self.conv_id, self.cached, self.unfed, self.turns = None, 0, None, 0

    def _continues(self, req: dict) -> bool:
        conv = req.get("conversation") or ""
        return bool(conv and conv == self.conv_id and self.unfed is not None
                    and len(req["messages"]) == self.turns + 1)

    def _joiner(self) -> list[int]:
        """Tokens that close the previous answer: the unfed last token, the
        end-of-turn marker if the answer stopped before it, and the newline the
        chat template puts between turns."""
        eot, nl = self._eot, self._nl
        if self.unfed == eot:
            return [eot, nl]
        if self.unfed == 1:                 # <eos>: close the turn instead
            return [eot, nl]
        return [self.unfed, eot, nl]

    def _strip_bos(self, ids: list[int], mm: list[int] | None):
        # Every leading <bos>: Gemma 3's template already starts with one and the
        # encode adds another, which must not land in the middle of a conversation.
        bos, n = self.ue.tokenizer.bos_token_id, 0
        while n < len(ids) and ids[n] == bos:
            n += 1
        return ids[n:], (mm[n:] if mm is not None else None)


class Gemma3Backend(_Conversation):
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
        self._eot = int(ue._end_of_turn_token_id)
        self._nl = int(ue.tokenizer.encode("\n", add_special_tokens=False)[0])
        self.clock_mhz = user_dma_core.configured_hardware_info().frequency_mhz

    def info(self) -> dict:
        ue = self.ue
        return {"model": "gemma3", "title": self.title, "engines": self.engines,
                "clock_mhz": self.clock_mhz, "peak_gflops": _peak(self.clock_mhz, self.engines),
                "hw_info": self.udc.hardware_info_summary(),
                "max_context": int(ue.MAX_CONTEXT_SIZE), "max_prompt_tokens": int(ue.MAX_CONTEXT_SIZE) - self.MIN_ROOM,
                "prefill_chunk_tokens": int(ue.PREFILL_MAX_SEQ_LEN),
                "weights_dram_bytes": int(ue.get_params_dram_usage()),
                "load_s": self.load_s, "compile_s": self.compile_s, "modalities": list(self.modalities)}

    MIN_ROOM = 32                      # context kept free for the answer when a prompt is fitted

    def _template(self, msgs: list[dict]) -> list[int]:
        tok = self.ue.tokenizer
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return list(tok.encode(text, add_special_tokens=True))

    def _encode(self, messages: list[dict]) -> tuple[list[int], int]:
        """Chat-template the conversation (the same encode gemma3_test.set_prefill_seq does),
        dropping the oldest turns until it fits the context. Returns (ids, dropped)."""
        tok, limit = self.ue.tokenizer, int(self.ue.MAX_CONTEXT_SIZE) - self.MIN_ROOM
        msgs = list(messages)
        dropped = 0
        while True:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = list(tok.encode(text, add_special_tokens=True))
            if len(msgs) <= 1 or len(ids) <= (limit if not dropped else _HISTORY_SHARE * limit):
                break
            msgs = msgs[2:] if len(msgs) > 2 else msgs[1:]
            dropped += 1
        if len(ids) > limit:
            raise ValueError(f"the prompt is {len(ids)} tokens; Gemma 3's {self.ue.MAX_CONTEXT_SIZE}-token "
                             f"context takes at most {limit}")
        if len(ids) < 2:
            raise ValueError("empty prompt")
        return ids, dropped

    def generate(self, req: dict, on_token) -> dict:
        if req.get("image") or req.get("audio"):
            raise ValueError("Gemma 3 1B is text only")
        ue = self.ue
        start, dropped = 0, 0
        if self._continues(req):
            turn, _ = self._strip_bos(self._template([req["messages"][-1]]), None)
            delta = self._joiner() + turn
            if self.cached + len(delta) + self.MIN_ROOM <= int(ue.MAX_CONTEXT_SIZE):
                start = self.cached
                ue.prefill_seq = tuple([0] * start + delta)   # history rows are not re-read
        if not start:
            ids, dropped = self._encode(req["messages"])
            ue.prefill_seq = tuple(ids)
        ue._prefill_start = start
        context_tokens = len(ue.prefill_seq)
        room = int(ue.MAX_CONTEXT_SIZE) - (context_tokens - 1)
        ue.max_new_tokens = max(1, min(int(req.get("max_new_tokens") or room), room))
        ue.on_token = on_token
        ue._next_program_dram_addr = self.program_cursor
        emit("status", id=req["id"], text="prefill")
        self._forget()                      # a failed run leaves the cache undefined
        t0 = time.perf_counter()
        try:
            ctx = ue.run_gemma3_prefill()
            emit("status", id=req["id"], text="decode")
            r = ue.run_gemma3_decode(ctx)
        finally:
            ue.on_token = None
            ue._prefill_start = 0
        wall = time.perf_counter() - t0
        self.conv_id = req.get("conversation") or None
        self.cached, self.unfed = int(ue.seq_len), getattr(ue, "_decode_last_token_id", None)
        self.turns = len(req["messages"]) + 1
        n = int(r["tokens_decoded"])
        dec_hw = r["decode_avg_hw_ms"] * n / 1e3
        dec_wall = (r["decode_avg_cpu_ms"] * n / 1e3) if n else None
        c = self.clock_mhz
        phases = [
            _phase("Prefill" + (f" ({ue._prefill_chunks} chunks)" if ue._prefill_chunks > 1 else ""),
                   tokens=ue._prefill_rows, hw_s=r["prefill_hw_ms"] / 1e3,
                   wall_s=r["prefill_cpu_ms"] / 1e3, gflops=r.get("prefill_gflops"),
                   engines=self.engines if ue._prefill_sharded() else 1, clock_mhz=c,
                   note=((f"new tokens at positions {start}+ on the cached conversation; " if start else "")
                         + (f"projections N-sharded over {self.engines} engines; attention on engine 0"
                            if ue._prefill_sharded() else "engine 0 runs prefill"))),
            _phase("Decode", tokens=n, hw_s=dec_hw, wall_s=dec_wall, gflops=r.get("decoder_gflops") or None,
                   engines=self.engines, clock_mhz=c, note=f"N-sharded over {self.engines} engines"),
        ]
        return {
            "prompt_tokens": context_tokens - start, "generated_tokens": n, "history_dropped": dropped,
            "context_tokens": int(ue.seq_len), "cached_tokens": start, "max_context": int(ue.MAX_CONTEXT_SIZE),
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

# When a conversation no longer fits and has to drop its oldest turns, keep at
# most this share of the context for what remains, so the next turns have room
# to continue on the cache instead of overflowing (and re-prefilling) every time.
_HISTORY_SHARE = 0.5


class Gemma4Backend(_Conversation):
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
        self._eot = int(ue._end_of_turn_token_id)
        self._nl = int(ue.tokenizer.encode("\n", add_special_tokens=False)[0])
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

    # Context kept free for the answer when the history is fitted. The answer
    # itself runs up to max_new_tokens or the end of the context, whichever is
    # first: reserving the whole max_new_tokens (8192 by default) would leave
    # no room for any history.
    MIN_ROOM = 64

    def _encode_text(self, messages: list[dict], max_new: int) -> tuple[list[int], int]:
        tok, ue = self.ue.tokenizer, self.ue
        limit = int(ue.MAX_CONTEXT_SIZE) - self.MIN_ROOM
        msgs, dropped = list(messages), 0
        while True:
            ids = self._template(msgs)
            if len(msgs) <= 1 or len(ids) <= (limit if not dropped else _HISTORY_SHARE * limit):
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
        max_new = max(1, int(req.get("max_new_tokens") or int(self.ue.MAX_CONTEXT_SIZE)))
        image, audio = req.get("image"), req.get("audio")
        prompt = req["messages"][-1]["content"] if req["messages"] else ""
        dropped = 0
        t_req = time.perf_counter()
        cont = self._continues(req)
        if image and audio:
            emit("status", id=rid, text="vision + audio encoders")
            ue.set_prefill_seq_media(image_path=image, audio_path=audio, prompt=prompt or None)
        elif image:
            emit("status", id=rid, text="vision encoder")
            ue.set_prefill_seq_vlm(image, prompt or None)
        elif audio:
            emit("status", id=rid, text="audio encoder")
            ue.set_prefill_seq_audio(audio, prompt or None)
        elif cont:
            ue.prefill_seq = tuple(self._template([{"role": "user", "content": prompt}]))
        else:
            ids, dropped = self._encode_text(req["messages"], max_new)
            ue.prefill_seq = tuple(ids)
        start = 0
        if cont:
            turn, mm_turn = self._strip_bos(list(ue.prefill_seq), getattr(ue, "_mm_types", None))
            delta = self._joiner() + turn
            if self.cached + len(delta) + self.MIN_ROOM <= int(ue.MAX_CONTEXT_SIZE):
                start = self.cached
                ue.prefill_seq = tuple([0] * start + delta)   # history rows are not re-read
                if mm_turn is not None:
                    ue._mm_types = [0] * (start + len(delta) - len(mm_turn)) + list(mm_turn)
            else:
                cont = False                                  # full: start the conversation over
                dropped = self.turns // 2
                if not (image or audio):
                    ids, dropped = self._encode_text(req["messages"][-1:], max_new)
                    ue.prefill_seq = tuple(ids)
        if not cont:
            self._forget()
        ue._prefill_start = start
        context_tokens = len(ue.prefill_seq)                       # cached history + this turn
        prompt_tokens = context_tokens - start                     # what this request prefills
        mm = list(getattr(ue, "_mm_types", None) or [])          # prefill clears it; count now
        n_img, n_aud = sum(t == 1 for t in mm), sum(t == 3 for t in mm)
        room = int(ue.MAX_CONTEXT_SIZE) - context_tokens
        if room < 1:
            raise ValueError(f"the prompt is {context_tokens} tokens; the context holds {ue.MAX_CONTEXT_SIZE}")
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
        self._forget()                      # a failed run leaves the cache undefined
        try:
            ue.run_gemma4()
        finally:
            ue.on_token = None
            ue._prefill_start = 0
        self.conv_id = req.get("conversation") or None
        self.cached, self.unfed = int(ue.seq_len), getattr(ue, "_decode_last_token_id", None)
        self.turns = len(req["messages"]) + 1
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
        pf_model, pf_gflops, pf_tokens = ue._model_flops_prefill(), r.get("prefill_gflops_model"), r.get("prefill_tokens")
        if start:
            # Only rows [start, end) ran: price them as the difference of two prefills.
            end = ue._prefill_seq_len
            ue._prefill_seq_len = start
            before = ue._model_flops_prefill()
            ue._prefill_seq_len = end
            pf_model = (pf_model - before) if (pf_model and before is not None) else None
            pf_gflops = (pf_model / r["prefill_hw_s"] / 1e9) if (pf_model and r.get("prefill_hw_s")) else None
            pf_tokens = end - start
        chunks = r.get("prefill_chunks") or 1
        phases.append(_phase("Prefill" + (f" ({chunks} chunks)" if chunks > 1 else ""),
                             tokens=pf_tokens, hw_s=r.get("prefill_hw_s"),
                             wall_s=r.get("prefill_wall_s"), flops=pf_model, gflops=pf_gflops,
                             engines=e, clock_mhz=c,
                             note=(f"new tokens at positions {start}+ on the cached conversation; model FLOPs"
                                   if start else "model FLOPs")))
        n = r.get("decode_tokens") or 0
        dec_us = getattr(ue, "_decode_hw_latency_us", None)
        dec_model = ue._model_flops_decode(n, ue.seq_len) if n else None
        phases.append(_phase("Decode", tokens=n, hw_s=(dec_us / 1e6) if dec_us else None,
                             wall_s=getattr(ue, "_decode_e2e_s", None), flops=dec_model,
                             gflops=r.get("decode_gflops_model"), engines=e, clock_mhz=c, note="model FLOPs"))
        text = r.get("decoded_text") or ""
        return {
            "prompt_tokens": prompt_tokens, "generated_tokens": n, "history_dropped": dropped,
            "context_tokens": int(ue.seq_len), "cached_tokens": start,
            "max_context": int(ue.MAX_CONTEXT_SIZE),
            "image_tokens": r.get("image_tokens") or n_img, "audio_tokens": r.get("audio_tokens") or n_aud,
            "phases": phases,
            "ttft_hw_s": r.get("ttft_hw_s"), "ttft_wall_s": r.get("ttft_wall_s"),
            "decode_first_tok_s": r.get("decode_first_token_tok_s"),
            "decode_hw_tok_s": r.get("decode_hw_tok_s"), "decode_wall_tok_s": r.get("decode_avg_tok_s"),
            "compile_s": compile_s, "compile_reused": reused, "wall_s": wall,
            "final_seq_len": r.get("final_seq_len"), "text": text,
        }


# ---------------------------------------------------------------------------
# Qwen2.5-Omni 7B (text, image, audio)
# ---------------------------------------------------------------------------

_QWEN_PER_REQUEST = ("_vis_latency_us", "_vis_wall_s", "_vis_total_flops", "_vis_gflops", "_vis_num_tokens",
                     "_audio_latency_us", "_audio_wall_s", "_audio_total_flops", "_audio_gflops",
                     "_audio_num_tokens", "_latency_prefill_us", "_prefill_wall_s", "_prefill_flops",
                     "_decode_step_us", "_decode_total_us", "_decode_wall_s", "_decode_n")


class QwenBackend:
    """Qwen2.5-Omni 7B Thinker, resident: weights go to the board once.

    The test script's own setup (flat DRAM plan, engine, artifacts) runs once
    with every stage the server can use ({lm, vision, audio}), so the plan --
    and with it every weight pool -- never changes, and the three weight sets
    load once. Each message then builds its programs for its own prompt shape
    (_prepare_unified, compile only; the weight inits are no-ops), after the
    engines' program cursors are rewound to where the first build started, and
    runs the encoders, prefill and decode as the script's main() does.

    History: every message re-prefills the whole conversation (the chat
    template over the text turns, the current turn's image / audio encoded);
    Qwen's prefill has no offset and its K/V cache is re-carved per message.
    """
    title = "Qwen2.5-Omni 7B"
    modalities = ("text", "image", "audio")
    MIN_ROOM = 64

    def __init__(self, engines: int, dev: str):
        import importlib.util
        path = os.path.join(MODELS, "qwen2.5_omni_7b", "qwen2.5_omni_7b_test.py")
        spec = importlib.util.spec_from_file_location("qwen25_omni_test", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["qwen25_omni_test"] = mod
        spec.loader.exec_module(mod)
        self.mod, self.engines = mod, engines
        parser = mod.build_arg_parser()
        self.base_args = parser.parse_args(["--multi-core", str(engines), "--dev", dev, "--no-summary"])
        # One process owns the board (and params.bin / programs.bin) for its lifetime.
        self._lock = mod._exclusive_run_lock()
        self._lock.__enter__()
        t0 = time.perf_counter()
        kw = mod.resolve_engine_config(parser, self.base_args)
        if not mod.dram_interleaved():
            raise RuntimeError("the resident Qwen runner needs the flat DRAM layout (U50)")
        a = self.base_args
        self.cfg = cfg = mod.apply_vision_resolution(
            mod.Qwen25OmniUnifiedEngine.load_config(script_dir=mod.SCRIPT_DIR), a.vision_res)
        params_path = mod._weight_mod.ensure_params_bin(mod.SCRIPT_DIR)
        self.params_bytes = _size(str(params_path))
        self.processor_dir = mod._weight_mod.ensure_processor_bundle(mod.SCRIPT_DIR, verbose=False)
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.processor_dir, trust_remote_code=True,
                                                       local_files_only=True)
        self.plan = mod._layout_mod.plan_flat(
            {"lm", "vision", "audio"}, a.vision_res, a.max_context,
            device_bytes=mod.user_dma_core.AVAILABLE_DRAM_SIZE_GB * 2**30)
        for line in self.plan.lines():
            log(line)
        fpga_build = mod.reset_selected_engines()
        ue = mod.Qwen25OmniUnifiedEngine(script_dir=mod.SCRIPT_DIR, fpga_build=fpga_build,
                                         vision_res=a.vision_res, fpga_talker=False,
                                         layout="flat", flat_plan=self.plan, **kw)
        ue.speak_as = None
        ue.configure_runtime_artifacts(params_path, self.processor_dir)
        ue.tokenizer = self.processor.tokenizer
        ue.processor = self.processor
        log(ue.describe_dram_map())
        # Weights, once: the flat plan gives every stage its own pool.
        ue.vision_weight_init()
        ue.audio_weight_init()
        ue.lm_weight_init()
        layers = int(ue._lm_dims()["NL"])
        ue._ensure_decode_shards(ue._ensure_stage_scheduler("decode"), layers)
        self.load_s = time.perf_counter() - t0
        self.ue = ue
        # Program cursors of every engine before any program is built: each
        # message's build starts here again (programs are rebuilt per prompt
        # shape, weights stay where they are).
        self._cursors = [(e, int(e._next_program_dram_addr)) for e in [ue] + list(ue._worker_pool or [])]
        self._fresh = {k: copy.deepcopy(getattr(ue, k)) for k in self._BUILD_STATE if hasattr(ue, k)}
        self.clock_mhz = 1000.0 / float(ue._clock_period_ns)
        self.compile_s = 0.0
        # Programs kept between messages (_prepare): the vision encoder (one
        # input shape, always the first stage, so always at the same address)
        # and the decode program + launch tables + flag-clear, keyed by where
        # they land (decode follows prefill, whose size rarely changes).
        self._vision_cache: dict | None = None
        self._decode_cache: dict = {}
        # Prefill programs by (prompt length, address): a repeated length (the
        # same question again, a new chat) skips its ~3 s compile. Bounded.
        self._prefill_cache: "collections.OrderedDict" = collections.OrderedDict()

    # Per-build bookkeeping the program builders accumulate (restored per message).
    _BUILD_STATE = ("_compile_state", "_raw_stages", "_packaged_program_stages", "_resident_stages",
                    "_resident_hashes", "_executed_program_stages", "_stage_tensor_state",
                    "_worker_isa_used")

    def _rewind(self) -> None:
        ue = self.ue
        for e, addr in self._cursors:
            e._next_program_dram_addr = addr
        for k, v in self._fresh.items():
            setattr(ue, k, copy.deepcopy(v))
        # The program bundle merges a process's later stores into the generation it
        # started; each message's build is a new generation of its own.
        bundle_mod = sys.modules.get(type(ue._program_bundle).__module__)
        if bundle_mod is not None:
            bundle_mod._STARTED_BUNDLES.pop(ue._program_bundle._run_key, None)
        for k in _QWEN_PER_REQUEST:
            if hasattr(ue, k):
                setattr(ue, k, None)
        # Stage images restored from memory shadow the compiled ones; and the
        # resident-image registry makes the DMA guard skip a write whose
        # (address, size) it has seen -- a new program of the same size at the
        # same address would then never reach the board. Both start empty.
        ue._bin_stage_state.clear()
        for regs in ue._resident_images.values():
            regs.clear()

    # -- per-message programs ------------------------------------------------
    def _engines(self):
        return [self.ue] + list(self.ue._worker_pool or [])

    def _cursor_state(self) -> tuple:
        return tuple(int(e._next_program_dram_addr) for e in self._engines())

    def _set_cursors(self, cursors: tuple) -> None:
        for e, addr in zip(self._engines(), cursors):
            e._next_program_dram_addr = addr

    def _stage_images(self, stage: str) -> dict:
        """{engine index: (address, bytes)} of a stage, as use_bin_stage takes them."""
        addr, master, workers = self.ue._program_stage_state(stage)
        images = {0: (int(addr), bytes(master))}
        for idx, _w, waddr, blob in workers:
            images[int(idx)] = (int(waddr), bytes(blob))
        return images

    def _prepare(self, args, processed, context) -> None:
        """The test script's _prepare_unified for this message, minus the work
        that does not change between messages.

        Stage order and placement are the script's (vision, audio, prefill,
        decode, decode launch tables, flag-clear, each after the previous one's
        ISA gap). The vision encoder is compiled once and restored afterwards,
        the way --run_from_bin restores it (compile state, tensor layout,
        images); decode and its launch tables are restored whenever they land
        at the addresses they were built for. Prefill (built for this prompt's
        length) and audio (this clip's length) are compiled. Nothing is stored
        to programs_flat.bin: every image goes to the board from host memory.
        """
        mod, ue = self.mod, self.ue
        started = time.perf_counter()
        worker_map = {i: w for i, w in enumerate(ue._worker_pool or [], start=1)}
        stages: list[str] = []
        vision_restored = False
        if args.image:
            vc = self._vision_cache
            vision_restored = vc is not None
            if vc is None:
                mod._run_vision(ue, processed, False, phase="prepare")
                self._vision_cache = {
                    "images": self._stage_images("vision"),
                    "compile_state": {k: ue._compile_state[k] for k in ("vision", "vision_extra")},
                    "tensor_state": copy.deepcopy(ue._stage_tensor_state["vision"]),
                    "isa_used": copy.deepcopy(ue._worker_isa_used),
                    "cursors": self._cursor_state(),
                }
            else:
                ue._compile_state.update(vc["compile_state"])
                mod._run_vision(ue, processed, False, phase="prepare_bin")   # inputs + tensors, no compile
                ue._stage_tensor_state["vision"] = copy.deepcopy(vc["tensor_state"])
                ue._worker_isa_used = copy.deepcopy(vc["isa_used"])
                ue.use_bin_stage("vision", vc["images"], worker_map)
                self._set_cursors(vc["cursors"])
            stages.append("vision")
        if args.audio:
            mod._run_audio(ue, processed, phase="prepare")
            stages.append("audio")
        before = dict(ue._dram_addresses)
        ue.lm_weight_init()
        ue.lm_tensor_init()
        layers = int(ue._lm_dims()["NL"])
        prefill_base = self._cursor_state()
        pkey = (len(context), prefill_base)
        pc = self._prefill_cache.get(pkey)
        if pc is None:
            ue.compile_or_restore("prefill", lambda: ue.compile_prefill(len(context)), False, "prefill")
            self._prefill_cache[pkey] = {"images": self._stage_images("prefill"),
                                         "compile_state": ue._compile_state["prefill"],
                                         "cursors": self._cursor_state()}
            while len(self._prefill_cache) > 24:
                self._prefill_cache.popitem(last=False)
        else:
            self._prefill_cache.move_to_end(pkey)
            ue._compile_state["prefill"] = pc["compile_state"]
            ue.compile_or_restore("prefill", None, True, "prefill")    # what the compile left behind
            ue.use_bin_stage("prefill", pc["images"], worker_map)
            self._set_cursors(pc["cursors"])
        ue._ensure_decode_shards(ue._ensure_stage_scheduler("decode"), layers)
        # Prefill's image size moves a little with the prompt length; give it a
        # fixed slot on every engine so decode (built right after it) always
        # lands at the same addresses and is built once, not per message.
        used = [c - b for c, b in zip(self._cursor_state(), prefill_base)]
        slot = getattr(self, "_prefill_slot", None)
        if slot is None or any(u > s_ for u, s_ in zip(used, slot)):
            mib = 2 ** 20
            self._prefill_slot = slot = tuple(-(-(u + u // 4 + mib) // mib) * mib for u in used)
        self._set_cursors(tuple(b + s_ for b, s_ in zip(prefill_base, slot)))
        key = self._cursor_state()
        dc = self._decode_cache.get(key)
        if dc is None:
            ue.compile_or_restore("decode", lambda: ue.compile_decoder(), False, "decode")
            ue.check_master_isa()
            ue.snapshot_stage_tensors("lm", before)
            ue.install_decode_launch_tables()
            ue.reserve_isa_gap("prefill", "decode")
            self._decode_cache[key] = {
                "images": self._stage_images("decode"),
                "compile_state": ue._compile_state["decode"],
                "lm_tensor_state": copy.deepcopy(ue._stage_tensor_state["lm"]),
                "launch": dict(ue._decode_launch) if getattr(ue, "_decode_launch", None) else None,
                "launch_raw": dict(ue._raw_stages["decode_launch"]) if "decode_launch" in ue._raw_stages else None,
                "cursors": self._cursor_state(),
            }
        else:
            ue._compile_state["decode"] = dc["compile_state"]
            ue.compile_or_restore("decode", None, True, "decode")     # what the compile left behind
            ue.use_bin_stage("decode", dc["images"], worker_map)
            ue._stage_tensor_state["lm"] = copy.deepcopy(dc["lm_tensor_state"])
            if dc["launch"] is not None:
                ue._decode_launch = dict(dc["launch"])
                ue._raw_stages["decode_launch"] = dict(dc["launch_raw"])
            self._set_cursors(dc["cursors"])
        stages += ["prefill", "decode"]
        if "decode_launch" in ue._raw_stages:
            stages.append("decode_launch")
        ue.install_flag_clear_programs()
        stages.append("flag_clear")
        self._install(stages)
        log(f"  [prepare] {', '.join(stages)} in {time.perf_counter() - started:.2f}s "
            f"(vision {('cached' if vision_restored else 'built') if args.image else '-'}, "
            f"prefill {'cached' if pc is not None else 'built'}, "
            f"decode {'cached' if dc is not None else 'built'})")

    def _install(self, stages: list[str]) -> None:
        """install_resident_programs from host memory: every stage image to the
        board, registered as resident (so the runners neither re-upload nor read
        programs_flat.bin)."""
        import hashlib
        ue = self.ue
        h2c = self.mod.user_dma_core.DMA_DEVICE_H2C
        for stage in stages:
            addr, master, workers = ue._program_stage_state(stage)
            items = [(ue, int(addr), bytes(master))] + [(w, int(a), bytes(b)) for _i, w, a, b in workers]
            for engine, a, blob in items:
                if engine.dma_write(h2c, a, blob, len(blob)) != len(blob):
                    raise IOError(f"{stage}: short program DMA at 0x{a:X}")
                ue._resident_images.setdefault(id(engine), {}).setdefault(a, set()).add(len(blob))
                who = "master" if engine is ue else f"worker@{id(engine) % 9973}"
                ue._resident_hashes[(stage, who, a)] = (engine, len(blob), hashlib.sha1(blob).hexdigest())
            if stage == "vision":
                ue._resident_images[id(ue)][int(addr)].add(len(ue._vis_encoder_program_bytes))
            ue._resident_stages.add(stage)
            ue._packaged_program_stages.add(stage)

    def info(self) -> dict:
        ue = self.ue
        return {"model": "qwen2.5_omni_7b", "title": self.title, "engines": self.engines,
                "clock_mhz": self.clock_mhz, "peak_gflops": _peak(self.clock_mhz, self.engines),
                "hw_info": self.mod.user_dma_core.hardware_info_summary(),
                "max_context": int(self.plan.max_context), "max_prompt_tokens": int(self.plan.max_context) - self.MIN_ROOM,
                "weights_dram_bytes": int(self.params_bytes),
                "load_s": self.load_s, "compile_s": self.compile_s, "modalities": list(self.modalities)}

    def _inputs(self, messages: list[dict], image: str | None, audio: str | None):
        """The processor over the conversation: earlier turns as text, the last
        user turn with its media ([audio] [image] text, as the test script)."""
        from PIL import Image
        mod, cfg, proc = self.mod, self.cfg, self.processor
        images = audio_in = None
        last = []
        if audio:
            audio_in = [mod._load_audio(audio, int(cfg["audio"]["sample_rate"]))]
            last.append({"type": "audio", "audio": audio})
        if image:
            size = int(cfg["vision"]["image_size"])
            with Image.open(image) as src:
                images = [src.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)]
            last.append({"type": "image", "image": image})
        last.append({"type": "text", "text": messages[-1]["content"]})
        history = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                   for m in messages[:-1]]
        limit = int(self.plan.max_context) - self.MIN_ROOM
        dropped = 0
        while True:
            msgs = history + [{"role": "user", "content": last}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            kw = {"text": text, "padding": True, "return_tensors": "pt"}
            if images is not None:
                kw["images"] = images
            if audio_in is not None:
                kw["audio"] = audio_in
            processed = proc(**kw)
            tokens = [int(v) for v in processed["input_ids"][0].tolist()]
            if len(tokens) <= limit or not history:
                break
            history = history[2:]
            dropped += 1
        if len(tokens) > limit:
            raise ValueError(f"the prompt is {len(tokens)} tokens; the context holds {self.plan.max_context}")
        return processed, tokens, dropped

    def generate(self, req: dict, on_token) -> dict:
        mod, ue, rid = self.mod, self.ue, req["id"]
        image, audio = req.get("image"), req.get("audio")
        processed, tokens, dropped = self._inputs(req["messages"], image, audio)
        context, seed = tokens[:-1], tokens[-1]
        room = int(self.plan.max_context) - len(tokens)
        args = copy.copy(self.base_args)
        args.image, args.audio = image, audio
        args.prompt = req["messages"][-1]["content"]
        args.max_new_tokens = max(1, min(int(req.get("max_new_tokens") or room), room))
        t_req = time.perf_counter()
        emit("status", id=rid, text="program image")
        self._rewind()
        self._prepare(args, processed, context)
        compile_s = time.perf_counter() - t_req
        image_emb = audio_emb = audio_meta = None
        if image:
            emit("status", id=rid, text="vision encoder")
            image_emb = mod._run_vision(ue, processed, phase="run")
        if audio:
            emit("status", id=rid, text="audio encoder")
            audio_emb, audio_meta = mod._run_audio(ue, processed, phase="run")
        positions, rope_delta = mod.build_multimodal_positions(
            tokens,
            image_grid_thw=processed.get("image_grid_thw") if image else None,
            audio_token_lengths=audio_meta["output_lengths"] if audio_meta is not None else None,
            image_token_id=int(self.cfg["tokens"]["image_token_id"]),
            audio_token_id=int(self.cfg["tokens"]["audio_token_id"]),
            video_token_id=int(self.cfg["tokens"]["video_token_id"]),
            spatial_merge_size=int(self.cfg["vision"]["spatial_merge_size"]),
            position_ids_per_second=int(self.cfg["vision"]["tokens_per_second"]))
        ue._rope_offset = int(rope_delta)
        emit("status", id=rid, text="prefill")
        ue.enter_stage_tensors("lm", ue.lm_tensor_init)
        ue.run_prefill(context, image_embeddings=image_emb, audio_embeddings=audio_emb,
                       positions=positions[: len(context)])
        ue.activate_decode_shared_weights()
        emit("status", id=rid, text="decode")
        ue.on_token = on_token
        try:
            _, text = ue.run_decoder(seed, max_new_tokens=args.max_new_tokens)
        finally:
            ue.on_token = None
        wall = time.perf_counter() - t_req
        rows = ue.stage_metrics(args)
        e, c = self.engines, self.clock_mhz
        phases, by = [], {}
        steps = list(getattr(ue, "_decode_step_us", ()) or ())
        for row in rows:
            hw = (row.get("us") or 0.0) / 1e6
            detail = str(row.get("detail", ""))
            if row["stage"] == "Prefill":
                tokens_n = len(context)
            elif row["stage"] == "Decode":
                tokens_n = len(steps)
            else:
                digits = [int(t) for t in detail.replace("->", " ").split() if t.isdigit()]
                tokens_n = digits[-1] if digits else None
            p = _phase(row["stage"], tokens=tokens_n, hw_s=hw or None, wall_s=row.get("wall"),
                       flops=row.get("model_flops"), gflops=row.get("model_gflops"), engines=e, clock_mhz=c,
                       note=(f"{detail}; issued work at {row.get('speedup', 0):.1f}x one engine's peak"
                             if row.get("speedup") else detail))
            phases.append(p)
            by[row["stage"]] = p
        ttft = [by[s] for s in ("Vision encoder", "Audio encoder", "Prefill") if s in by]
        dec = by.get("Decode")
        dec_wall = float(getattr(ue, "_decode_wall_s", 0.0) or 0.0)
        return {
            "prompt_tokens": len(tokens), "generated_tokens": int(getattr(ue, "_decode_n", len(steps)) or 0),
            "history_dropped": dropped, "context_tokens": len(tokens) + len(steps),
            "max_context": int(self.plan.max_context),
            "phases": phases,
            "ttft_hw_s": sum(p["hw_s"] or 0 for p in ttft) or None,
            "ttft_wall_s": sum(p["wall_s"] or 0 for p in ttft) or None,
            "decode_first_tok_s": (1e6 / steps[0]) if steps else None,
            "decode_hw_tok_s": dec["tok_s"] if dec else None,
            "decode_wall_tok_s": (len(steps) / dec_wall) if dec_wall else None,
            "compile_s": compile_s, "compile_reused": False, "wall_s": wall,
            "text": text or "",
        }


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=("gemma3", "gemma4_e2b", "gemma4_e4b", "qwen2.5_omni_7b"))
    ap.add_argument("--engines", type=int, default=8)
    ap.add_argument("--dev", default="xdma0")
    args = ap.parse_args(argv)
    os.chdir(ROOT)

    try:
        if args.model == "gemma3":
            backend = Gemma3Backend(args.engines, args.dev)
        elif args.model == "qwen2.5_omni_7b":
            backend = QwenBackend(args.engines, args.dev)
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
