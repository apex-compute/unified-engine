#!/usr/bin/env python3
"""Gemma 4 E2B CUDA host inference, aligned with the FPGA VLM test.

Defaults deliberately match ``gemma4_e2b_test.py --image``:
  * test_samples/yosemite.jpg, resized to 896 x 896
  * prompt: ``Describe this image in detail.``
  * 2,520 vision patches / 256 image soft tokens
  * 384 generated tokens

The Hugging Face checkpoint is loaded only from the repository-local model
directory; this benchmark does not download a model or silently fall back to
CPU. Normal mode prints the decoded response. ``--profile`` measures
CUDA-synchronized stage timings and writes an FPGA-style Markdown report.
"""

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
DEFAULT_MODEL = SCRIPT_DIR / "gemma4_e2b_bin" / "gemma-4-E2B-it"
DEFAULT_IMAGE = REPO_ROOT / "test_samples" / "yosemite.jpg"
DEFAULT_PROMPT = "Describe this image in detail."
VISION_CANONICAL_SIZE = (896, 896)
VISION_FIXED_NUM_PATCHES = 2520
VISION_FIXED_SOFT_TOKENS = 256
DEFAULT_NEW_TOKENS = 384
DEFAULT_REPORT = SCRIPT_DIR / "gemma4_e2b_host_bf16_performance.md"


def _sync() -> None:
    torch.cuda.synchronize()


def _timed_generate(model, inputs: dict[str, torch.Tensor], max_new_tokens: int):
    """Generate deterministically and return synchronized wall-clock seconds."""
    _sync()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    _sync()
    return generated, time.perf_counter() - started


def _median(values: list[float]) -> float:
    return statistics.median(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Local Hugging Face Gemma 4 E2B checkpoint directory.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE,
                        help="Input image; resized to 896x896 to match the FPGA test.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,
                        help="Prompt text, following the same single-image chat template.")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_NEW_TOKENS,
                        help="Number of deterministic output tokens (default: 384).")
    parser.add_argument("--profile", action="store_true",
                        help="Benchmark vision/prefill/decode and write a Markdown report.")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Untimed full generations before measurement.")
    parser.add_argument("--runs", type=int, default=3,
                        help="Timed full generations; median is reported.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT,
                        help="Markdown output used by --profile.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1 or args.warmup < 0 or args.runs < 1:
        raise SystemExit("--max-new-tokens and --runs must be positive; --warmup cannot be negative.")
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. Install a CUDA-enabled PyTorch build in this Python "
            "environment; this benchmark intentionally refuses a CPU comparison."
        )
    if not args.model.is_dir():
        raise SystemExit(f"Model directory not found: {args.model}")
    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    gpu_name = torch.cuda.get_device_name(device)

    print("Gemma 4 E2B host CUDA benchmark")
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
    print(f"Model: {args.model}")
    print(f"Image: {args.image}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Image resize: {VISION_CANONICAL_SIZE[0]}x{VISION_CANONICAL_SIZE[1]}")
    print("Weight precision: BF16")
    print(f"Requested generation: {args.max_new_tokens} tokens")

    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device).eval()
    _sync()
    print(f"Model load + transfer: {time.perf_counter() - load_started:.3f} s")

    preprocess_started = time.perf_counter()
    image = Image.open(args.image).convert("RGB").resize(VISION_CANONICAL_SIZE, Image.BICUBIC)
    conversation = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": args.prompt},
    ]}]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text_prompt], images=[[image]], return_tensors="pt")
    preprocess_s = time.perf_counter() - preprocess_started

    pixel_values = inputs["pixel_values"]
    input_ids = inputs["input_ids"]
    mm_types = inputs["mm_token_type_ids"]
    image_positions = int((mm_types == 1).sum().item())
    print("\nInput contract")
    print(f"pixel_values: {tuple(pixel_values.shape)}")
    print(f"image_position_ids: {tuple(inputs['image_position_ids'].shape)}")
    print(f"prefill sequence: {input_ids.shape[1]} tokens; image positions: {image_positions}")
    print(f"CPU preprocessing: {preprocess_s:.3f} s")
    if pixel_values.shape[1] != VISION_FIXED_NUM_PATCHES:
        raise RuntimeError(
            f"Expected {VISION_FIXED_NUM_PATCHES} vision patches, got {pixel_values.shape[1]}.")
    if image_positions != VISION_FIXED_SOFT_TOKENS:
        raise RuntimeError(
            f"Expected {VISION_FIXED_SOFT_TOKENS} image soft-token positions, got {image_positions}.")

    gpu_inputs: dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if torch.is_floating_point(value):
            gpu_inputs[key] = value.to(device=device, dtype=torch.bfloat16)
        else:
            gpu_inputs[key] = value.to(device)

    if not args.profile:
        output, elapsed = _timed_generate(model, gpu_inputs, args.max_new_tokens)
        generated = output[0, input_ids.shape[1]:]
        print(f"\nDecoded text ({generated.numel()} tokens, {elapsed:.3f} s):\n")
        print(processor.decode(generated, skip_special_tokens=True))
        return

    for _ in range(args.warmup):
        _timed_generate(model, gpu_inputs, args.max_new_tokens)

    torch.cuda.reset_peak_memory_stats(device)
    first_token_times: list[float] = []
    full_times: list[float] = []
    generated_tokens = None
    for _ in range(args.runs):
        one_token, first_token_s = _timed_generate(model, gpu_inputs, 1)
        output, full_s = _timed_generate(model, gpu_inputs, args.max_new_tokens)
        first_token_times.append(first_token_s)
        full_times.append(full_s)
        generated_tokens = int(output.shape[1] - input_ids.shape[1])
        del one_token, output

    first_token_s = _median(first_token_times)
    full_s = _median(full_times)
    # A one-token generate includes the VLM prefill and first-token selection.
    # The residual approximates the 383 remaining decode steps, matching the
    # FPGA report's first-token and average-decode figures.
    decode_steps = max(0, (generated_tokens or 0) - 1)
    decode_s = max(0.0, full_s - first_token_s)
    decode_toks = (decode_steps / decode_s) if decode_s else float("inf")
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    def _vision_once() -> float:
        _sync()
        started = time.perf_counter()
        with torch.inference_mode():
            model.model.get_image_features(
                pixel_values=gpu_inputs["pixel_values"],
                image_position_ids=gpu_inputs["image_position_ids"],
            )
        _sync()
        return time.perf_counter() - started

    vision_s = _median([_vision_once() for _ in range(args.runs)])
    estimated_prefill_s = max(0.0, first_token_s - vision_s)

    print("\nTiming (median of %d timed run%s; CUDA-synchronized)" % (
        args.runs, "" if args.runs == 1 else "s"))
    print(f"Vision encoder: {vision_s:.4f} s")
    print(f"VLM prefill + first token: {first_token_s:.4f} s ({1 / first_token_s:.2f} tok/s)")
    print(f"Estimated LM prefill + first token: {estimated_prefill_s:.4f} s")
    print(f"Remaining decode: {decode_s:.4f} s for {decode_steps} tokens ({decode_toks:.2f} tok/s)")
    print(f"Full generation: {full_s:.4f} s for {generated_tokens} tokens "
          f"({generated_tokens / full_s:.2f} tok/s)")
    print(f"Peak allocated CUDA memory: {peak_gib:.2f} GiB")
    print("\nFPGA comparison note: this run used %d prefill tokens and %d total tokens. "
          "Saved FPGA reports may show 272/656 for the same visible input; compare "
          "the printed input contract when tokenizer versions differ." %
          (input_ids.shape[1], input_ids.shape[1] + generated_tokens))
    if args.report:
        props = torch.cuda.get_device_properties(device)
        allocated_pct = 100.0 * peak_gib * (1024 ** 3) / props.total_memory
        report = f'''# Gemma4 E2B host GPU performance — BF16

## Test setup

| Field | Value |
|---|---|
| GPU | {gpu_name} |
| CUDA / PyTorch | {torch.version.cuda} / {torch.__version__} |
| Architecture / compute capability | Blackwell / {props.major}.{props.minor} |
| CUDA cores | 6,144 |
| Tensor cores | 192, 5th generation |
| Advertised peak AI throughput | 988 AI TOPS (FP4 sparse) |
| VRAM | {props.total_memory / (1024 ** 3):.2f} GiB GDDR7 |
| Theoretical DRAM bandwidth | 672 GB/s |
| Weight / activation precision | BF16 / BF16 |
| Image | `{args.image}`; resized to {VISION_CANONICAL_SIZE[0]}×{VISION_CANONICAL_SIZE[1]} |
| Vision patches / soft tokens | {pixel_values.shape[1]} / {image_positions} |
| Prompt | {args.prompt} |
| Prefill / generated / total tokens | {input_ids.shape[1]} / {generated_tokens} / {input_ids.shape[1] + generated_tokens} |
| Timed runs | {args.runs} median, CUDA synchronized |
| Peak allocated VRAM | {peak_gib:.2f} GiB ({allocated_pct:.1f}% of device memory) |

## Performance

| Stage | Metric |
|---|---:|
| Vision encoder | {vision_s:.4f} s |
| VLM prefill + first token | {first_token_s:.4f} s ({1 / first_token_s:.2f} tok/s) |
| Estimated LM prefill + first token | {estimated_prefill_s:.4f} s |
| Decode (remaining {decode_steps} tokens) | {decode_s:.4f} s ({decode_toks:.2f} tok/s) |
| Full generation ({generated_tokens} tokens) | {full_s:.4f} s ({generated_tokens / full_s:.2f} tok/s) |

Vision is measured independently. The VLM first-token time includes vision; subtracting the independent vision measurement gives the displayed estimated LM-prefill figure.
'''
        args.report.write_text(report, encoding="utf-8")
        print(f"Wrote Markdown report: {args.report}")


if __name__ == "__main__":
    main()
