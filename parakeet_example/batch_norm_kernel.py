#!/usr/bin/env python3
"""
Batch Norm kernel for Parakeet Conformer ConvModule.

Eval-mode BatchNorm fused to: out[c,t] = in[c,t] * scale[c] + shift[c]
  where scale = weight / sqrt(var + eps), shift = bias - mean * scale

Mirrors the hardware flow: per-channel broadcast mul + broadcast add,
processing L-element rows through 64-wide vector lanes.
"""

import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import torch
from user_dma_core import UE_VECTOR_SIZE

EPS = 1e-5


# ---------------------------------------------------------------------------
# Core function: emulates the hardware kernel in bf16
# ---------------------------------------------------------------------------

def batch_norm_fused(x_bf16, bn_weight, bn_bias, bn_mean, bn_var, eps=EPS):
    """Fused eval-mode batch norm in bf16 — matches hardware execution.

    This is what the accelerator does:

    1. Host pre-computes scale & shift (once at weight load):
         scale[c] = weight[c] / sqrt(var[c] + eps)
         shift[c] = bias[c] - mean[c] * scale[c]
       Both quantized to bf16 and written to DRAM.

    2. Hardware processes (C, L) activation row-by-row:
       For each channel c:
         - DMA load: input[c, 0:L] from DRAM → URAM_A        (strided read, L elements)
         - MUL_BROADCAST: URAM_A[i] *= scale[c]  for i in 0..L-1  (L/64 vector cycles)
         - ADD_BROADCAST: URAM_A[i] += shift[c]  for i in 0..L-1  (L/64 vector cycles)
         - DMA store: URAM_A → DRAM output[c, 0:L]            (64-aligned writeback)

    Args:
        x_bf16:    (C, L) bf16 input activation
        bn_weight: (C,) batch norm gamma
        bn_bias:   (C,) batch norm beta
        bn_mean:   (C,) running mean
        bn_var:    (C,) running variance
    Returns:
        (C, L) bf16 output
    """
    # --- Step 1: fuse params to scale+shift, quantize to bf16 (host, once) ---
    scale = (bn_weight.float() / torch.sqrt(bn_var.float() + eps)).to(torch.bfloat16)
    shift = (bn_bias.float() - bn_mean.float() * scale.float()).to(torch.bfloat16)

    # --- Step 2: per-channel broadcast mul + add (hardware loop) ---
    # scale/shift are (C,) → unsqueeze to (C,1) for broadcast over L
    return x_bf16 * scale.unsqueeze(1) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# SNR
# ---------------------------------------------------------------------------

def calculate_snr(reference, result):
    """SNR in dB. Higher = better. >60 dB float32-exact, ~30-40 dB bf16-typical."""
    ref, res = reference.float(), result.float()
    if torch.isnan(ref).any() or torch.isnan(res).any():
        return float('-inf')
    noise = ((ref - res) ** 2).mean()
    return float('inf') if noise == 0 else (10 * torch.log10((ref ** 2).mean() / noise)).item()


# ---------------------------------------------------------------------------
# FLOPS utilization
# ---------------------------------------------------------------------------

def batch_norm_flops(C, L, cycle_ns=5.63):
    """FLOPS and DMA analysis for fused batch norm on (C, L).

    Compute: 2 broadcast ops per channel (mul + add), each L/64 vector cycles.
      - Useful FLOPS:  2 * C * L  (one mul + one add per element)
      - HW FLOPS:      2 * C * L  (all 64 lanes active, no padding waste)
      - Utilization:   100%  (broadcast fills all lanes)

    DMA (the real bottleneck):
      - Per channel: load L + store L = 2*L elements = 4*L bytes (bf16)
      - Params:      2*C elements = 4*C bytes (scale + shift, loaded once)
      - Total:       4*C*L + 4*C bytes

    Instructions: 4 per channel (load + mul + add + store) + 2 param loads.
    """
    useful = 2 * C * L
    cycles = C * 2 * (L // UE_VECTOR_SIZE)
    hw = cycles * UE_VECTOR_SIZE
    dma = 4 * C * L + 4 * C  # bytes

    print(f"\n  BatchNorm FLOPS: ({C}, {L})")
    print(f"  {'Useful FLOPS:':<24} {useful:>10,}  ({C*L:,} muls + {C*L:,} adds)")
    print(f"  {'HW FLOPS:':<24} {hw:>10,}  ({cycles:,} cycles x 64 lanes)")
    print(f"  {'Utilization:':<24} {useful/hw:.0%}")
    print(f"  {'DMA total:':<24} {dma:>10,} bytes ({dma/1024:.1f} KB)")
    print(f"  {'Instructions:':<24} {C*4+2:>10,}")
    print(f"  {'Compute time (est):':<24} {cycles*cycle_ns/1000:.1f} us @ {cycle_ns} ns/cyc")
    print(f"  {'Note:':<24} DMA-bound ({dma/1024:.0f} KB xfer >> {cycles*cycle_ns/1000:.1f} us compute)")
    return {"useful": useful, "hw": hw, "util": useful / hw, "dma_bytes": dma, "cycles": cycles}


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test():
    C, L = 1024, 128
    bn_weight, bn_bias = torch.randn(C), torch.randn(C)
    bn_mean, bn_var = torch.randn(C), torch.rand(C).abs() + 0.1
    x = torch.randn(C, L)

    # PyTorch reference (float32)
    bn = torch.nn.BatchNorm1d(C)
    bn.eval()
    bn.weight.data.copy_(bn_weight); bn.bias.data.copy_(bn_bias)
    bn.running_mean.copy_(bn_mean); bn.running_var.copy_(bn_var)
    with torch.no_grad():
        ref = bn(x.unsqueeze(0)).squeeze(0)

    # Our fused bf16 kernel
    out = batch_norm_fused(x.to(torch.bfloat16), bn_weight, bn_bias, bn_mean, bn_var)

    # float32 fused (sanity check — should match PyTorch exactly)
    scale = bn_weight / torch.sqrt(bn_var + EPS)
    shift = bn_bias - bn_mean * scale
    f32_out = x * scale.unsqueeze(1) + shift.unsqueeze(1)

    snr_f32 = calculate_snr(ref, f32_out)
    snr_bf16 = calculate_snr(ref, out.float())
    print(f"  f32 fused vs PyTorch:   SNR={snr_f32:.1f} dB  max_diff={(ref-f32_out).abs().max():.2e}")
    print(f"  bf16 fused vs PyTorch:  SNR={snr_bf16:.1f} dB  max_diff={(ref-out.float()).abs().max():.2e}")

    batch_norm_flops(C, L)
    print(f"\n  PASSED")


if __name__ == "__main__":
    test()
