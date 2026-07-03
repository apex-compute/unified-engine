#!/usr/bin/env python3
"""Master GPU quant comparison suite for pi05_libero, torch-only, no FPGA.

Two levels, both bf16 vs IF4 vs TQ4 (TurboQuant):
  1. Per-weight reconstruction fidelity          -> see quant_compare_gpu.py
  2. End-to-end action-chunk fidelity (this file) -- runs the *actual policy
     forward pass* (vision encoder, both LM stacks, flow-matching denoising
     loop) with every eligible nn.Linear weight fake-quantized in place, on
     the same sample observation and same fixed noise as the bf16 baseline.

Schemes:
  - IF4  adaptive per-block INT4/FP4 min-MSE pick      (quant_lib.quantize_if4)
  - TQ4  TurboQuant: per-64-block Walsh-Hadamard        (turbo_quant.quantize_tq4)
         rotation + Lloyd-Max 4-bit codebook

Setup (one-time):
    conda activate pi05
    python examples/convert_jax_model_to_pytorch.py \\
        --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \\
        --output_path <repo>/models/pi05_libero/pi05_libero_pytorch \\
        --config_name pi05_libero
    cp -r ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets \\
        <repo>/models/pi05_libero/pi05_libero_pytorch/

Run:
    conda activate pi05
    python quant_master_suite_gpu.py
"""
import dataclasses
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import quant_lib

sys.path.insert(0, "/home/rohit/apex-compute-ML/simple-llm/src/quantization")
import turbo_quant

from openpi.training import config as _config
from openpi.policies import policy_config

CKPT_DIR = Path(__file__).parent / "pi05_libero_pytorch"
SAMPLE_DIR = Path("/home/rohit/apex-compute-ML/simple-llm/src/models/pi0_5/sample_data")
BLOCK_SIZE = 64
ACTION_HORIZON = 10
ACTION_DIM = 32
SEED = 0


def load_example(idx: int) -> dict:
    meta = json.loads((SAMPLE_DIR / "meta.json").read_text())
    return {
        "observation/image": np.load(SAMPLE_DIR / f"sample_{idx}_image.npy"),
        "observation/wrist_image": np.load(SAMPLE_DIR / f"sample_{idx}_wrist_image.npy"),
        "observation/state": np.array(meta["state_example"], dtype=np.float32),
        "prompt": "pick up the object",
    }


def snr_db(ref: np.ndarray, approx: np.ndarray) -> float:
    signal_power = float(np.sum(ref.astype(np.float64) ** 2))
    noise_power = float(np.sum((ref.astype(np.float64) - approx.astype(np.float64)) ** 2))
    if noise_power == 0:
        return float("inf")
    return 10.0 * math.log10(signal_power / noise_power)


# ---------------------------------------------------------------------------
# Per-scheme weight quantizers -- each takes an (N, K) bf16 weight and
# returns its dequantized bf16 reconstruction, same shape.
# ---------------------------------------------------------------------------

def _if4_roundtrip(w_bf16: torch.Tensor) -> torch.Tensor:
    N, K = w_bf16.shape
    data, scale = quant_lib.quantize_if4(w_bf16, block_size=BLOCK_SIZE)
    return quant_lib.dequantize_if4(data, scale, N, K, block_size=BLOCK_SIZE).to(w_bf16.device)


def _tq4_roundtrip(w_bf16: torch.Tensor) -> torch.Tensor:
    N, K = w_bf16.shape
    codes, scales, codebook, orig_shape = turbo_quant.quantize_tq4(w_bf16, block_size=BLOCK_SIZE)
    rotated = turbo_quant.dequant_tq4(codes, scales, codebook, orig_shape, block_size=BLOCK_SIZE)
    # dequant_tq4 leaves weights in WHT-rotated space; the normalized WHT is
    # self-inverse, so applying it again per 64-block undoes the rotation.
    blocked = rotated.reshape(N, K // BLOCK_SIZE, BLOCK_SIZE)
    unrotated = turbo_quant._wht_1d(blocked)
    return unrotated.reshape(N, K)


SCHEMES = {
    "IF4": _if4_roundtrip,
    "TQ4": _tq4_roundtrip,
}


def quantize_model_(model: nn.Module, scheme_fn, scheme_name: str) -> list:
    skipped = []
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    print(f"Quantizing {len(linears)} nn.Linear layers to {scheme_name} (block_size={BLOCK_SIZE})...")
    for i, (name, module) in enumerate(linears):
        K = module.weight.shape[1]
        if K % BLOCK_SIZE != 0:
            skipped.append((name, tuple(module.weight.shape), f"in_features={K} not a multiple of {BLOCK_SIZE}"))
            continue
        # _wht_1d (used by TQ4) upcasts to float32 and its recursive
        # stack/reshape needs ~2x the tensor's fp32 footprint as scratch.
        # For huge single tensors (e.g. lm_head.weight at 257152x1024 =
        # 263M params) that scratch alone can tip an already-near-full card
        # into OOM. Route those to CPU for the quantize step only.
        oversized = module.weight.numel() > 50_000_000
        compute_device = "cpu" if oversized else module.weight.device
        with torch.no_grad():
            w_bf16 = module.weight.data.to(device=compute_device, dtype=torch.bfloat16)
            deq = scheme_fn(w_bf16)
            module.weight.data.copy_(deq.to(device=module.weight.device, dtype=module.weight.dtype))
        del w_bf16, deq
        # quantize_tq4's WHT + Lloyd-Max fit leaves sizeable scratch tensors
        # per call; across 458 Linear layers the CUDA allocator fragments
        # badly enough to OOM near the end without periodic release.
        if torch.cuda.is_available() and i % 20 == 0:
            torch.cuda.empty_cache()
    return skipped


def report(name: str, actions_bf16: np.ndarray, actions_q: np.ndarray, skipped: list):
    print("\n" + "=" * 70)
    print(f"ACTION-LEVEL bf16 vs {name} (full policy forward pass, GPU, real weights)")
    print("=" * 70)
    np.set_printoptions(precision=6, suppress=True)
    print(f"{name} actions:\n", actions_q)
    print("\nabs diff (bf16 - " + name + "):\n", np.abs(actions_bf16 - actions_q))
    overall_snr = snr_db(actions_bf16, actions_q)
    overall_mse = float(np.mean((actions_bf16 - actions_q) ** 2))
    print(f"\nOverall action SNR: {overall_snr:.2f} dB")
    print(f"Overall action MSE: {overall_mse:.3e}")
    print("Per-timestep SNR (dB):")
    for t in range(actions_bf16.shape[0]):
        t_snr = snr_db(actions_bf16[t], actions_q[t])
        t_mse = float(np.mean((actions_bf16[t] - actions_q[t]) ** 2))
        print(f"  t={t:2d}  SNR {t_snr:7.2f} dB   MSE {t_mse:.3e}")
    if skipped:
        print(f"Skipped {len(skipped)} Linear layers (in_features not a multiple of {BLOCK_SIZE}):")
        for lname, shape, reason in skipped[:10]:
            print(f"  {lname:60s} {str(shape):>20s}  {reason}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")
    return {"scheme": name, "overall_snr": overall_snr, "overall_mse": overall_mse}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = _config.get_config("pi05_libero")
    # Pi0Config defaults pytorch_compile_mode to "max-autotune", which wraps
    # sample_actions in torch.compile. Its autotune benchmarking retains
    # large workspace buffers across repeated calls with changing weight
    # values (bf16 -> IF4 -> TQ4), which is what pinned VRAM near the 11.5GB
    # ceiling and OOM'd on the TQ4 pass even after torch.cuda.empty_cache().
    # We don't need compiled-kernel throughput for a correctness/SNR sweep,
    # so force eager mode here.
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, pytorch_compile_mode=None))

    print(f"Loading bf16 baseline policy from {CKPT_DIR} on {device}...")
    policy = policy_config.create_trained_policy(config, CKPT_DIR, pytorch_device=device)

    example = load_example(0)
    rng = np.random.default_rng(SEED)
    fixed_noise = rng.standard_normal((ACTION_HORIZON, ACTION_DIM)).astype(np.float32)

    print("Running bf16 baseline inference...")
    actions_bf16 = policy.infer(example, noise=fixed_noise)["actions"]
    print(f"bf16 action_chunk shape: {actions_bf16.shape}")
    print("bf16 actions:\n", actions_bf16)

    # Snapshot original bf16 weights so each scheme starts from the same
    # baseline instead of compounding quantization from the previous scheme.
    # Keep the snapshot on CPU -- a second GPU-resident copy of the full
    # model would double VRAM use and OOM on an 11.5GB card.
    original_state = {k: v.detach().cpu().clone() for k, v in policy._model.state_dict().items()}

    summary = []
    for scheme_name, scheme_fn in SCHEMES.items():
        policy._model.load_state_dict(original_state)
        skipped = quantize_model_(policy._model, scheme_fn, scheme_name)
        print(f"Running {scheme_name} inference...")
        actions_q = policy.infer(example, noise=fixed_noise)["actions"]
        summary.append(report(scheme_name, actions_bf16, actions_q, skipped))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("SUMMARY (action-level, bf16 reference)")
    print("=" * 70)
    print(f"{'scheme':10s} {'SNR (dB)':>12s} {'MSE':>14s}")
    for s in summary:
        print(f"{s['scheme']:10s} {s['overall_snr']:12.2f} {s['overall_mse']:14.3e}")


if __name__ == "__main__":
    main()
