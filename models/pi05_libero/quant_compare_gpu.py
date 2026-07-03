#!/usr/bin/env python3
"""GPU-only (torch, no FPGA) weight-fidelity comparison for pi05_libero.

For every matmul-shaped weight in weights_export/, quantizes with each scheme
below and measures reconstruction error against the bf16 baseline:

  - IF4        adaptive per-block INT4/FP4 min-MSE pick (quant_lib.quantize_if4)
  - TQ4        TurboQuant: per-64-block Walsh-Hadamard rotation + Lloyd-Max
               4-bit codebook (turbo_quant.quantize_tq4)

Weights are reshaped to 2D (N, K) with K = last axis. Only weights whose K is
a multiple of 64 (the HW block size) are quantized; everything else (biases,
norms, scales, and the head_dim=72 attention kernels) is skipped and reported
as such rather than silently dropped.

Run:
    conda activate apex-compute   # or any env with torch+cuda
    python quant_compare_gpu.py
"""
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import quant_lib

sys.path.insert(0, "/home/rohit/apex-compute-ML/simple-llm/src/quantization")
import turbo_quant

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights_export")
MANIFEST = os.path.join(WEIGHTS_DIR, "manifest.json")
BLOCK_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Only these are real matmul kernels -- skip biases/norms/scales/embeddings
# that aren't literal weight matrices participating in a contraction.
KERNEL_SUFFIXES = (".kernel", ".w", "gating_einsum", "einsum.w", ".linear")


def is_kernel(name: str) -> bool:
    if "pos_embedding" in name:
        return False
    return any(name.endswith(s) or s in name for s in KERNEL_SUFFIXES) or name.endswith("input_embedding")


def snr_db(ref: torch.Tensor, approx: torch.Tensor) -> float:
    ref_f = ref.float()
    err_f = (ref.float() - approx.float())
    signal_power = (ref_f ** 2).sum().item()
    noise_power = (err_f ** 2).sum().item()
    if noise_power == 0:
        return float("inf")
    return 10.0 * math.log10(signal_power / noise_power)


def mse(ref: torch.Tensor, approx: torch.Tensor) -> float:
    return ((ref.float() - approx.float()) ** 2).mean().item()


def quantize_if4_roundtrip(w_bf16_2d: torch.Tensor) -> torch.Tensor:
    N, K = w_bf16_2d.shape
    data, scale = quant_lib.quantize_if4(w_bf16_2d, block_size=BLOCK_SIZE)
    return quant_lib.dequantize_if4(data, scale, N, K, block_size=BLOCK_SIZE).to(w_bf16_2d.device)


def quantize_tq4_roundtrip(w_bf16_2d: torch.Tensor) -> torch.Tensor:
    N, K = w_bf16_2d.shape
    codes, scales, codebook, orig_shape = turbo_quant.quantize_tq4(w_bf16_2d, block_size=BLOCK_SIZE)
    rotated = turbo_quant.dequant_tq4(codes, scales, codebook, orig_shape, block_size=BLOCK_SIZE)
    # dequant_tq4 returns weights still in Walsh-Hadamard-rotated space; the
    # normalized WHT is self-inverse (H/sqrt(n) is orthogonal & symmetric),
    # so applying it a second time per 64-block undoes the rotation.
    blocked = rotated.reshape(N, K // BLOCK_SIZE, BLOCK_SIZE)
    unrotated = turbo_quant._wht_1d(blocked)
    return unrotated.reshape(N, K)


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)

    rows = []
    skipped = []

    for name, meta in manifest.items():
        if not is_kernel(name):
            continue
        shape = meta["shape"]
        if len(shape) < 2:
            skipped.append((name, shape, "not 2D+"))
            continue
        K = shape[-1]
        if K % BLOCK_SIZE != 0:
            skipped.append((name, shape, f"K={K} not a multiple of {BLOCK_SIZE}"))
            continue

        arr = np.load(os.path.join(WEIGHTS_DIR, meta["file"]))

        # TQ4's Lloyd-Max fit does a torch.randperm over the full flattened
        # tensor before subsampling, which itself spikes memory (e.g. 526M
        # elements -> ~4.2GB int64) on top of the resident weight set.
        # Route oversized tensors to CPU to avoid CUDA OOM; small/medium
        # tensors stay on GPU for speed.
        tensor_device = "cpu" if arr.size > 50_000_000 else DEVICE

        N_full = arr.size // K
        w = torch.from_numpy(arr).reshape(N_full, K).to(tensor_device)

        # Tensors above ~150M elements (e.g. mlp.gating_einsum at 1.2B) make
        # TQ4's per-tensor 50-iteration Lloyd-Max fit + full-tensor
        # nearest-codebook quantization pass impractically slow on CPU.
        # Sample a representative slice of rows instead of the full matrix;
        # SNR/MSE are computed only over the sampled rows and flagged below.
        MAX_ROWS_FULL = 150_000_000 // K
        sampled = N_full > MAX_ROWS_FULL
        if sampled:
            row_idx = torch.randperm(N_full)[:MAX_ROWS_FULL]
            w = w[row_idx]
        N = w.shape[0]
        w_bf16 = w.to(torch.bfloat16)

        if_deq = quantize_if4_roundtrip(w_bf16)
        tq_deq = quantize_tq4_roundtrip(w_bf16)
        del w
        if tensor_device == "cuda":
            torch.cuda.empty_cache()

        rows.append({
            "name": name,
            "shape": shape,
            "numel": arr.size,
            "sampled": sampled,
            "if4_mse": mse(w_bf16, if_deq),
            "if4_snr": snr_db(w_bf16, if_deq),
            "tq4_mse": mse(w_bf16, tq_deq),
            "tq4_snr": snr_db(w_bf16, tq_deq),
        })
        tag = " (sampled)" if sampled else ""
        print(f"{name:70s}{tag:10s} IF4 {rows[-1]['if4_snr']:6.2f} dB   TQ4 {rows[-1]['tq4_snr']:6.2f} dB")

    print("\n" + "=" * 100)
    print(f"{'weight':70s} {'shape':>20s} {'IF4 MSE':>12s} {'IF4 SNR(dB)':>12s} {'TQ4 MSE':>12s} {'TQ4 SNR(dB)':>12s}")
    print("-" * 100)
    for r in rows:
        tag = "*" if r["sampled"] else ""
        print(f"{r['name']+tag:70s} {str(r['shape']):>20s} {r['if4_mse']:12.3e} {r['if4_snr']:12.2f} {r['tq4_mse']:12.3e} {r['tq4_snr']:12.2f}")
    if any(r["sampled"] for r in rows):
        print("\n* = row-sampled (>150M elements), MSE/SNR computed over a subsample, not the full tensor")

    total_numel = sum(r["numel"] for r in rows)
    agg_if4_mse = sum(r["if4_mse"] * r["numel"] for r in rows) / total_numel
    agg_tq4_mse = sum(r["tq4_mse"] * r["numel"] for r in rows) / total_numel
    if4_wins = sum(1 for r in rows if r["if4_snr"] >= r["tq4_snr"])
    tq4_wins = len(rows) - if4_wins

    print("-" * 100)
    print(f"{'AGGREGATE (numel-weighted)':70s} {'':>20s} {agg_if4_mse:12.3e} {'':>12s} {agg_tq4_mse:12.3e}")
    print(f"\nIF4 wins on {if4_wins}/{len(rows)} tensors, TQ4 wins on {tq4_wins}/{len(rows)} tensors")

    if skipped:
        print(f"\nSkipped {len(skipped)} tensors (not block-quantizable):")
        for name, shape, reason in skipped:
            print(f"  {name:70s} {str(shape):>20s}  {reason}")


if __name__ == "__main__":
    main()
