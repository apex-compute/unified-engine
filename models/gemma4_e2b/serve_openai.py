#!/usr/bin/env python3
"""OpenAI-compatible chat server for Gemma 4 E2B on the Unified Engine.

Long-running sibling of gemma4_e2b_run_from_bin.py: loads the precompiled
bins once (weights DMA to card DRAM at startup), then serves
POST /v1/chat/completions (streaming SSE and non-streaming), GET /v1/models
and GET /health. Any OpenAI-compatible client works: curl, the openai SDK,
chat UIs.

Single card, single request at a time; concurrent requests get 503 while
the engine is busy. Decode is greedy by construction — the LM head ends in
the hardware argmax, logits never cross PCIe — so temperature > 0 is
rejected unless --force-greedy is passed.

Usage (from the repo root, same host requirements as run_from_bin):
    python3 models/gemma4_e2b/serve_openai.py --port 8080

Requires gemma4_e2b_bin/ artifacts (programs.bin/.json, params.bin,
tokenizer/) built once by gemma4_e2b_test.py.
"""
import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))  # repo root
sys.path.insert(0, SCRIPT_DIR)

# Globals bound at startup.
ENGINE = None
MANIFEST = None
RFB = None               # gemma4_e2b_run_from_bin module
UDC = None               # user_dma_core module
TORCH = None
MODEL_NAME = "gemma-4-E2B-it"
EOS_TOKENS = set()
PREFILL_MAX = 0
MAX_CTX = 0
FORCE_GREEDY = False
GEN_LOCK = threading.Lock()


def log(msg: str) -> None:
    # gemma4_e2b_run_from_bin overrides builtins.print for library
    # silencing; write to stderr directly so server logs always land.
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


def flatten_content(content):
    """OpenAI message content -> plain text. Rejects non-text parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            else:
                raise ValueError(
                    "only text content is supported on this endpoint (the "
                    "vision/audio encoders are not wired into the server yet)")
        return "".join(parts)
    raise ValueError(f"unsupported content type: {type(content).__name__}")


def messages_to_prompt_ids(messages):
    """Apply the Gemma 4 chat template via the engine tokenizer (multi-turn
    version of set_prefill_seq). Gemma has no system role; system text is
    folded into the first user turn."""
    system_parts, conv = [], []
    for msg in messages:
        role = msg.get("role", "user")
        text = flatten_content(msg.get("content"))
        if role == "system":
            system_parts.append(text)
        elif role in ("user", "assistant"):
            conv.append({"role": role, "content": text})
        else:
            raise ValueError(f"unsupported role: {role!r}")
    if not conv:
        raise ValueError("no user/assistant messages in request")
    if system_parts:
        for entry in conv:
            if entry["role"] == "user":
                entry["content"] = "\n".join(system_parts) + "\n\n" + entry["content"]
                break
        else:
            conv.insert(0, {"role": "user", "content": "\n".join(system_parts)})
    templated = ENGINE.tokenizer.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=True)
    return tuple(ENGINE.tokenizer.encode(templated, add_special_tokens=True))


def generate(prompt_ids, max_tokens):
    """Prefill then greedy decode. Yields (token_id, token_text). Caller
    must hold GEN_LOCK. The decode loop mirrors run_decoder (dynamic PBI,
    single decoder program, HW argmax) minus the terminal status bar and
    env-driven penalty knobs."""
    ue = ENGINE
    ue.prefill_seq = tuple(prompt_ids)
    RFB._SILENT_MODE = True
    try:
        ue.run_prefill_bucketed(MANIFEST)

        decoder_addr = MANIFEST["_decoder_addr_int"]
        flops_tok = MANIFEST["decoder_total_flops"]
        token_id = ue.prefill_seq[-1]

        # The decoder's LM-head matmul adds PENALTY_BIAS_DRAM as its C term;
        # zero it so the HW argmax is pure greedy.
        ue.dma_to_accelerator_memory(
            ue.PENALTY_BIAS_DRAM,
            TORCH.zeros(1, ue.EMBEDDING_ELEMENTS, dtype=TORCH.bfloat16))

        # Prime gpr_seq_len once; the decoder program self-advances it.
        ue.isa_add_set_core(ue.gpr_seq_len, ue.seq_len)

        for _ in range(max_tokens):
            if ue.seq_len >= ue.MAX_CONTEXT_SIZE:
                return
            ue.seq_len += 1
            aligned = ((ue.seq_len + 63) // 64) * 64
            ue.isa_add_set_core(ue.gpr_bucket_idx, aligned // UDC.UE_VECTOR_SIZE)

            embedding = ue.get_embedding_for_tokens([token_id])
            ue.dma_to_accelerator_memory(ue.LAYER0_INPUT_DRAM, embedding)
            per_layer = ue._compute_per_layer_inputs([token_id], embedding)
            ue.dma_to_accelerator_memory(
                ue.PER_LAYER_INPUTS_DRAM, per_layer.permute(1, 0, 2).contiguous())

            full_bias = TORCH.full((1, aligned), -1e36, dtype=TORCH.bfloat16)
            full_bias[0, :ue.seq_len] = 0.0
            ue.dma_to_accelerator_memory(ue.LAYER0_FLASH_BIAS_FULL_DRAM, full_bias)
            if ue.seq_len <= ue.sliding_window:
                sliding_bias = full_bias
            else:
                sliding_bias = TORCH.full((1, aligned), -1e36, dtype=TORCH.bfloat16)
                sliding_bias[0, ue.seq_len - ue.sliding_window:ue.seq_len] = 0.0
            ue.dma_to_accelerator_memory(ue.LAYER0_FLASH_BIAS_SLIDING_DRAM, sliding_bias)

            ue.program_execute(decoder_addr, flops=flops_tok)
            token_id = int(ue.get_arg_max_index())
            yield token_id, ue.tokenizer.decode([token_id])
            if token_id in EOS_TOKENS:
                return
    finally:
        RFB._SILENT_MODE = False


def make_chunk(request_id, model, delta_content=None, finish_reason=None):
    choice = {"index": 0, "delta": {}, "finish_reason": finish_reason}
    if delta_content is not None:
        choice["delta"]["content"] = delta_content
    if finish_reason is None and delta_content is None:
        choice["delta"]["role"] = "assistant"
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [choice],
    }


def make_response(request_id, model, content, finish_reason, prompt_tokens, completion_tokens):
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class APIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        log(fmt % args)

    def _cors_headers(self):
        return {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        }

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        for k, v in self._cors_headers().items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, code, message):
        self._send_json(code, {
            "error": {"message": message, "type": "invalid_request_error", "code": code}
        })

    def do_OPTIONS(self):
        self.send_response(204)
        for k, v in self._cors_headers().items():
            self.send_header(k, v)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {
                "status": "ok",
                "model": MODEL_NAME,
                "backend": "apex-unified-engine",
                "prefill_max_seq_len": PREFILL_MAX,
                "max_context_size": MAX_CTX,
                "busy": GEN_LOCK.locked(),
            })
        elif self.path == "/v1/models":
            self._send_json(200, {
                "object": "list",
                "data": [{
                    "id": MODEL_NAME,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "apex-compute",
                }],
            })
        else:
            self._send_error_json(404, "not found")

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send_error_json(404, "not found")
            return
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._send_error_json(400, "empty request body")
            return
        try:
            body = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as e:
            self._send_error_json(400, f"invalid JSON: {e}")
            return

        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            self._send_error_json(400, "'messages' is required and must be a list")
            return

        temperature = body.get("temperature", 0.0) or 0.0
        if temperature > 0 and not FORCE_GREEDY:
            self._send_error_json(400,
                "temperature > 0 is not supported: decode uses the on-card "
                "hardware argmax (logits never leave the card). Send "
                "temperature 0, or start the server with --force-greedy to "
                "accept any temperature and decode greedily.")
            return

        max_tokens = int(body.get("max_tokens", 256))
        stream = bool(body.get("stream", False))
        stop = body.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        try:
            prompt_ids = messages_to_prompt_ids(messages)
        except ValueError as e:
            self._send_error_json(400, str(e))
            return

        # run_prefill_bucketed consumes prompt_ids[:-1] through the static
        # template; the last token seeds the first decode step.
        if len(prompt_ids) < 2:
            self._send_error_json(400, "prompt too short after templating")
            return
        if len(prompt_ids) - 1 > PREFILL_MAX:
            self._send_error_json(400,
                f"prompt is {len(prompt_ids)} tokens; this instruction bin "
                f"was compiled with prefill_max_seq_len={PREFILL_MAX}. "
                f"Shorten the conversation or rebuild with a larger "
                f"prefill_max_seq_len in gemma4_e2b_config.json.")
            return
        max_tokens = max(1, min(max_tokens, MAX_CTX - len(prompt_ids)))

        if not GEN_LOCK.acquire(timeout=0.5):
            self._send_error_json(503, "engine busy with another request, try again")
            return
        try:
            if stream:
                self._handle_stream(request_id, prompt_ids, max_tokens, stop)
            else:
                self._handle_sync(request_id, prompt_ids, max_tokens, stop)
        except BrokenPipeError:
            log(f"{request_id}: client disconnected mid-stream")
        except Exception as e:
            log(f"{request_id}: FAILED: {type(e).__name__}: {e}")
            try:
                self._send_error_json(500, f"{type(e).__name__}: {e}")
            except Exception:
                pass
            raise
        finally:
            GEN_LOCK.release()

    def _handle_stream(self, request_id, prompt_ids, max_tokens, stop):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # SSE body has no Content-Length; connection close delimits it.
        self.send_header("Connection", "close")
        self.close_connection = True
        for k, v in self._cors_headers().items():
            self.send_header(k, v)
        self.end_headers()

        chunk = make_chunk(request_id, MODEL_NAME)
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.flush()

        finish_reason = "length"
        completion_tokens = 0
        accumulated = ""
        t0 = time.perf_counter()
        for token_id, token_text in generate(prompt_ids, max_tokens):
            if token_id in EOS_TOKENS:
                finish_reason = "stop"
                break
            completion_tokens += 1
            accumulated += token_text
            chunk = make_chunk(request_id, MODEL_NAME, delta_content=token_text)
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            if stop and any(s in accumulated for s in stop):
                finish_reason = "stop"
                break

        chunk = make_chunk(request_id, MODEL_NAME, finish_reason=finish_reason)
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        dt = time.perf_counter() - t0
        log(f"{request_id}: {completion_tokens} tokens in {dt:.1f}s "
            f"({completion_tokens / dt:.2f} tok/s incl. prefill)")

    def _handle_sync(self, request_id, prompt_ids, max_tokens, stop):
        parts = []
        finish_reason = "length"
        completion_tokens = 0
        for token_id, token_text in generate(prompt_ids, max_tokens):
            if token_id in EOS_TOKENS:
                finish_reason = "stop"
                break
            parts.append(token_text)
            completion_tokens += 1
            if stop and any(s in "".join(parts) for s in stop):
                finish_reason = "stop"
                break
        self._send_json(200, make_response(
            request_id, MODEL_NAME, "".join(parts), finish_reason,
            len(prompt_ids), completion_tokens))


def main():
    global ENGINE, MANIFEST, RFB, UDC, TORCH
    global MODEL_NAME, EOS_TOKENS, PREFILL_MAX, MAX_CTX, FORCE_GREEDY

    parser = argparse.ArgumentParser(
        description="OpenAI-compatible chat server for Gemma 4 E2B on the "
                    "Unified Engine (execute-only, from precompiled bins).")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--dev", default="xdma0",
                        help="DMA device name (e.g., xdma0, xdma1). Default: xdma0")
    parser.add_argument("--cycle", type=float, default=5.62,
                        help="Clock cycle time in nanoseconds (default: 5.62; use 2.5 for Alveo)")
    parser.add_argument("--local-weights", action="store_true",
                        help="Use gemma4_e2b_bin/params.bin instead of generated params.bin")
    parser.add_argument("--model-name", default=None,
                        help="Model id reported by /v1/models (default: gemma-4-E2B-it)")
    parser.add_argument("--force-greedy", action="store_true",
                        help="Accept temperature > 0 and decode greedily anyway "
                             "(default: reject with 400)")
    args = parser.parse_args()

    FORCE_GREEDY = args.force_greedy
    if args.model_name:
        MODEL_NAME = args.model_name

    import user_dma_core as UDC_mod
    import gemma4_e2b_run_from_bin as RFB_mod
    import torch
    RFB, UDC, TORCH = RFB_mod, UDC_mod, torch

    UDC.set_dma_device(args.dev)
    RFB.DMA_DEVICE_H2C = UDC.DMA_DEVICE_H2C
    RFB.DMA_DEVICE_C2H = UDC.DMA_DEVICE_C2H
    RFB.DMA_DEVICE_USER = UDC.DMA_DEVICE_USER
    UDC.CLOCK_CYCLE_TIME_NS = args.cycle
    log(f"DMA device {args.dev} (h2c={UDC.DMA_DEVICE_H2C}), cycle={args.cycle}ns")
    if not os.path.exists(UDC.DMA_DEVICE_H2C):
        raise SystemExit(
            f"ERROR: {UDC.DMA_DEVICE_H2C} not present. Install the card and "
            f"load the XDMA driver first (see the repo README).")

    bin_dir = os.path.join(SCRIPT_DIR, "gemma4_e2b_bin")
    for artifact in ("programs.bin", "programs.json"):
        if not os.path.exists(os.path.join(bin_dir, artifact)):
            raise SystemExit(
                f"ERROR: {os.path.join(bin_dir, artifact)} missing. Build the "
                f"bins once with gemma4_e2b_test.py, then start the server.")

    log("initializing engine (weights DMA to card DRAM, takes a few minutes)")
    t0 = time.time()
    ENGINE = RFB.Gemma4_UnifiedEngine(local_weights=args.local_weights)
    MANIFEST = ENGINE.load_instruction_bin()
    log(f"engine ready in {time.time() - t0:.0f}s")

    PREFILL_MAX = MANIFEST["prefill_max_seq_len"]
    MAX_CTX = ENGINE.MAX_CONTEXT_SIZE
    EOS_TOKENS = {1, ENGINE._end_of_turn_token_id}

    server = ThreadingHTTPServer((args.host, args.port), APIHandler)
    server.daemon_threads = True
    log(f"serving on {args.host}:{args.port}")
    log(f"  model: {MODEL_NAME}  prefill_max={PREFILL_MAX}  max_ctx={MAX_CTX}")
    log("  POST /v1/chat/completions   GET /v1/models   GET /health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("shutting down, clearing card DRAM scratch")
        ENGINE.clear_dram()
        server.shutdown()


if __name__ == "__main__":
    main()
