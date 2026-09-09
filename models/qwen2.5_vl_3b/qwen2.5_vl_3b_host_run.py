#!/usr/bin/env python3
"""Qwen2.5-VL-3B BF16 CUDA inference and FPGA-aligned profiling."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MODEL = SCRIPT_DIR / "qwen2.5_vl_3b_bin" / "Qwen2.5-VL-3B-Instruct"
DEFAULT_IMAGE = REPO_ROOT / "test_samples" / "yosemite.jpg"
DEFAULT_PROMPT = "Describe the picture in details."
DEFAULT_REPORT = SCRIPT_DIR / "qwen2.5_vl_3b_host_bf16_performance.md"
IMAGE_SIZE = (336, 336)
EXPECTED_PATCHES = 576
EXPECTED_IMAGE_TOKENS = 144


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--profile", action="store_true",
                   help="Measure vision/prefill/decode and write Markdown.")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return p.parse_args()


def sync():
    torch.cuda.synchronize()


def timed_generate(model, inputs, n):
    sync()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=n,
                                use_cache=True)
    sync()
    return output, time.perf_counter() - started


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; refusing to silently benchmark CPU.")
    if not args.model.is_dir() or not args.image.is_file():
        raise SystemExit(f"Missing model or image: {args.model}, {args.image}")
    if args.max_new_tokens < 1 or args.runs < 1 or args.warmup < 0:
        raise SystemExit("Invalid token, run, or warmup count.")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    print("Qwen2.5-VL-3B host CUDA run")
    print(f"GPU: {gpu_name}; PyTorch {torch.__version__}; CUDA {torch.version.cuda}")
    print("Weight / activation precision: BF16 / BF16")

    load_t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    sync()
    load_s = time.perf_counter() - load_t0

    prep_t0 = time.perf_counter()
    image = Image.open(args.image).convert("RGB").resize(IMAGE_SIZE, Image.BICUBIC)
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": args.prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    prep_s = time.perf_counter() - prep_t0
    input_ids = inputs["input_ids"]
    grid = inputs["image_grid_thw"]
    patches = int(grid[0].prod().item())
    image_tokens = patches // 4
    if patches != EXPECTED_PATCHES or image_tokens != EXPECTED_IMAGE_TOKENS:
        raise RuntimeError(f"FPGA shape mismatch: {patches} patches, {image_tokens} tokens")
    gpu_inputs = {k: (v.to(device=device, dtype=torch.bfloat16)
                      if torch.is_floating_point(v) else v.to(device))
                  for k, v in inputs.items()}
    print(f"Image: {args.image} resized to 336x336; grid {grid.tolist()}")
    print(f"Vision patches / merged tokens: {patches} / {image_tokens}")
    print(f"Prompt: {args.prompt!r}; prefill tokens: {input_ids.shape[1]}")
    print(f"Model load: {load_s:.3f} s; preprocessing: {prep_s:.3f} s")

    if not args.profile:
        output, elapsed = timed_generate(model, gpu_inputs, args.max_new_tokens)
        generated = output[0, input_ids.shape[1]:]
        print(f"\nDecoded text ({generated.numel()} tokens, {elapsed:.3f} s):\n")
        print(processor.decode(generated, skip_special_tokens=True))
        return

    for _ in range(args.warmup):
        timed_generate(model, gpu_inputs, args.max_new_tokens)
    torch.cuda.reset_peak_memory_stats(device)
    first, full = [], []
    generated_tokens = 0
    for _ in range(args.runs):
        one, t1 = timed_generate(model, gpu_inputs, 1)
        out, tall = timed_generate(model, gpu_inputs, args.max_new_tokens)
        first.append(t1); full.append(tall)
        generated_tokens = int(out.shape[1] - input_ids.shape[1])
        del one, out

    def vision_once():
        sync(); t0 = time.perf_counter()
        with torch.inference_mode():
            visual = getattr(model, "visual", None)
            if visual is None:
                visual = model.model.visual
            visual(gpu_inputs["pixel_values"],
                   grid_thw=gpu_inputs["image_grid_thw"])
        sync(); return time.perf_counter() - t0

    vision_s = statistics.median([vision_once() for _ in range(args.runs)])
    first_s = statistics.median(first)
    full_s = statistics.median(full)
    prefill_s = max(0.0, first_s - vision_s)
    decode_steps = max(0, generated_tokens - 1)
    decode_s = max(0.0, full_s - first_s)
    decode_tps = decode_steps / decode_s
    total_tps = generated_tokens / full_s
    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    props = torch.cuda.get_device_properties(device)
    peak_pct = 100 * peak_gib * 1024**3 / props.total_memory

    report = f"""# Qwen2.5-VL-3B host GPU performance — BF16

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
| Image | `{args.image}`; resized to 336×336 |
| Vision patches / merged tokens | {patches} / {image_tokens} |
| Prompt | {args.prompt} |
| Prefill / generated / total tokens | {input_ids.shape[1]} / {generated_tokens} / {input_ids.shape[1]+generated_tokens} |
| Timed runs | {args.runs} median, CUDA synchronized |
| Peak allocated VRAM | {peak_gib:.2f} GiB ({peak_pct:.1f}%) |

## Performance

| Stage | Metric |
|---|---:|
| Vision encoder | {vision_s:.4f} s |
| VLM prefill + first token | {first_s:.4f} s ({1/first_s:.2f} tok/s) |
| Estimated LM prefill + first token | {prefill_s:.4f} s |
| Decode ({decode_steps} remaining tokens) | {decode_s:.4f} s ({decode_tps:.2f} tok/s) |
| Full generation ({generated_tokens} tokens) | {full_s:.4f} s ({total_tps:.2f} tok/s) |

Vision is measured independently. The estimated LM-prefill figure subtracts
that vision measurement from the end-to-end first-token measurement.
"""
    args.report.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote Markdown report: {args.report}")


if __name__ == "__main__":
    main()
