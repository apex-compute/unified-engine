#!/usr/bin/env python3
"""Gemma3-1B BF16 CUDA inference and FPGA-aligned profiling."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "gemma3_bin" / "gemma-3-1b-it"
DEFAULT_PROMPT = "x+3=5, what is x?"
DEFAULT_REPORT = SCRIPT_DIR / "gemma3_host_bf16_performance.md"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max-new-tokens", type=int, default=76)
    p.add_argument("--profile", action="store_true",
                   help="Measure prefill/decode and write Markdown.")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return p.parse_args()


def sync():
    torch.cuda.synchronize()


def timed_generate(model, inputs, n):
    sync(); started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=n,
                                use_cache=True)
    sync()
    return output, time.perf_counter() - started


def timed_prefill(model, inputs):
    sync(); started = time.perf_counter()
    with torch.inference_mode():
        output = model(**inputs, use_cache=True)
    sync()
    elapsed = time.perf_counter() - started
    del output
    return elapsed


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; refusing to silently benchmark CPU.")
    if not args.model.is_dir():
        raise SystemExit(f"Model directory not found: {args.model}")
    if args.max_new_tokens < 1 or args.runs < 1 or args.warmup < 0:
        raise SystemExit("Invalid token, run, or warmup count.")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    print("Gemma3-1B host CUDA run")
    print(f"GPU: {gpu_name}; PyTorch {torch.__version__}; CUDA {torch.version.cuda}")
    print("Weight / activation precision: BF16 / BF16")

    load_t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    sync(); load_s = time.perf_counter() - load_t0

    conversation = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(conversation, tokenize=False,
                                         add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_tokens = int(inputs["input_ids"].shape[1])
    fpga_prefill_tokens = input_tokens - 1
    print(f"Prompt: {args.prompt!r}")
    print(f"Input IDs: {input_tokens}; FPGA-equivalent prefill compute tokens: {fpga_prefill_tokens}")
    print(f"Model load + transfer: {load_s:.3f} s")

    if not args.profile:
        output, elapsed = timed_generate(model, inputs, args.max_new_tokens)
        generated = output[0, input_tokens:]
        print(f"\nDecoded text ({generated.numel()} tokens, {elapsed:.3f} s):\n")
        print(tokenizer.decode(generated, skip_special_tokens=True))
        return

    for _ in range(args.warmup):
        timed_generate(model, inputs, args.max_new_tokens)
    torch.cuda.reset_peak_memory_stats(device)
    prefill_times, first_times, full_times = [], [], []
    generated_tokens = 0
    for _ in range(args.runs):
        prefill_times.append(timed_prefill(model, inputs))
        one, first_s = timed_generate(model, inputs, 1)
        out, full_s = timed_generate(model, inputs, args.max_new_tokens)
        first_times.append(first_s); full_times.append(full_s)
        generated_tokens = int(out.shape[1] - input_tokens)
        del one, out

    prefill_s = statistics.median(prefill_times)
    first_s = statistics.median(first_times)
    full_s = statistics.median(full_times)
    decode_s = max(0.0, full_s - prefill_s)
    decode_tps = generated_tokens / decode_s
    total_tps = generated_tokens / full_s
    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    props = torch.cuda.get_device_properties(device)
    peak_pct = 100 * peak_gib * 1024**3 / props.total_memory

    report = f"""# Gemma3-1B host GPU performance — BF16

## Test setup

| Field | Value |
|---|---|
| GPU | {gpu_name} |
| Architecture / compute capability | Blackwell / {props.major}.{props.minor} |
| CUDA cores / Tensor cores | 6,144 / 192 (5th generation) |
| CUDA / PyTorch | {torch.version.cuda} / {torch.__version__} |
| Advertised peak AI throughput | 988 AI TOPS (FP4 sparse) |
| VRAM / theoretical DRAM bandwidth | {props.total_memory/1024**3:.2f} GiB GDDR7 / 672 GB/s |
| Weight / activation precision | BF16 / BF16 |
| Prompt | {args.prompt} |
| Host input IDs / FPGA prefill compute tokens | {input_tokens} / {fpga_prefill_tokens} |
| Generated / total host tokens | {generated_tokens} / {input_tokens+generated_tokens} |
| Timed runs | {args.runs} median, CUDA synchronized |
| Peak allocated VRAM | {peak_gib:.2f} GiB ({peak_pct:.1f}%) |

## Performance

| Stage | Metric |
|---|---:|
| Prefill forward | {prefill_s:.4f} s |
| End-to-end first token | {first_s:.4f} s ({1/first_s:.2f} tok/s) |
| Decode after prefill ({generated_tokens} tokens) | {decode_s:.4f} s ({decode_tps:.2f} tok/s) |
| Full generation ({generated_tokens} tokens) | {full_s:.4f} s ({total_tps:.2f} tok/s) |

The FPGA runner processes all but the final prompt token in its prefill program;
that final token enters its decoder. The host model receives the full chat-template
input, so both token counts are shown explicitly.
"""
    args.report.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote Markdown report: {args.report}")


if __name__ == "__main__":
    main()
