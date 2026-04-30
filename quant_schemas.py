"""Block-scale 4-bit quantization schemas for the unified-engine HW.

Three schemas (INT4 / FP4 / IF4) x two directions (quantize / dequantize)
= six functions, one per combination. Operates on any 2D (N, K) bf16
tensor with K % block_size == 0 -- not weight-specific, though current
callers all happen to be weight packers. Single utility shared across all
model templates and the inner-dim-hardware eval. Pairs with
src/quantization/kv_quant_schemas.py (KV-cache schemas for SW simulation).

The HW dispatches the INT4 vs FP4 codebook by the sign bit of the per-block
bf16 scale: negative -> INT4 codes (-8..7), positive -> FP4 E2M1. This
module bakes that convention in so callers never re-derive it.

Contract (all functions):
  - 2D tensor (N, K) only. K % block_size == 0 (block_size is fixed at 64
    for the current HW; exposed as a parameter for forward-compat).
  - Blocking is along the last axis (K).
  - Returns flat byte buffers: data is row-major (N, K/2) packed nibbles
    (low nibble of byte = even K index, high nibble = odd K index); scale
    is row-major (N, K/64) bf16.
  - No padding, no axis selection, no multi-dim auto-flatten. Callers that
    want per-head / N-axis / transposed layouts reshape and concatenate
    themselves -- those are bin-layout concerns, not quantization concerns.

Functions:
  quantize_int4 / dequantize_int4 -- INT4 codes (-8..7), all-negative scales.
  quantize_fp4  / dequantize_fp4  -- FP4 E2M1, all-positive scales.
  quantize_if4  / dequantize_if4  -- per-block min-MSE pick between INT4
                                     and FP4; sign bit set per block. The
                                     dequant routes per-block by sign.
"""

import numpy as np
import torch


# FP4 E2M1 codebooks. _FP4_VALUES is the value table in argmin order
# (used during quantization). _FP4_NIBBLES[i] is the HW storage code that
# decodes back to value i. _FP4_DECODE indexes by storage code.
_FP4_VALUES_F32 = np.array(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0,
      0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0],
    dtype=np.float32,
)
_FP4_NIBBLES = np.array(
    [0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8,
     0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7],
    dtype=np.uint8,
)
_FP4_DECODE_F32 = np.array(
    [+0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def _check_2d_blocked(weight: torch.Tensor, block_size: int) -> tuple[int, int, int]:
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D (N, K); got shape {tuple(weight.shape)}")
    N, K = weight.shape
    if K % block_size != 0:
        raise ValueError(f"K={K} must be a multiple of block_size={block_size}")
    return N, K, K // block_size


def _pack_nibbles(nibbles: np.ndarray) -> bytes:
    low = nibbles[:, 0::2].astype(np.uint16)
    high = nibbles[:, 1::2].astype(np.uint16)
    packed = ((high << 4) | low).astype(np.uint8)
    return packed.tobytes()


def _unpack_nibbles(data_bytes: bytes, N: int, K: int) -> np.ndarray:
    packed = np.frombuffer(data_bytes, dtype=np.uint8)
    packed = packed[: N * (K // 2)].reshape(N, K // 2)
    low_nib = (packed & 0x0F).astype(np.int16)
    high_nib = ((packed >> 4) & 0x0F).astype(np.int16)
    return np.stack([low_nib, high_nib], axis=-1).reshape(N, K)


# ---------------------------------------------------------------------------
# INT4: codes -8..7, scale stored negative (HW INT4 dispatch).
# ---------------------------------------------------------------------------

def quantize_int4(weight: torch.Tensor, block_size: int = 64) -> tuple[bytes, bytes]:
    """Pack a 2D bf16/float weight as INT4 (codes -8..7) per K-block.
    Scales are written with a flipped sign so the HW dispatches as INT4.
    """
    N, K, num_blocks_k = _check_2d_blocked(weight, block_size)
    w = weight.detach().cpu().float()
    blocks = w.view(N, num_blocks_k, block_size)
    abs_max = blocks.abs().amax(dim=-1).clamp(min=1e-8)

    scale_f = abs_max / 7.0
    scale_bf_view = scale_f.to(torch.bfloat16).float()
    q = (blocks / scale_bf_view.unsqueeze(-1)).round().clamp(-8, 7)

    scale_bf16 = (-scale_f).to(torch.bfloat16)
    nibbles = (q.to(torch.int8).numpy().astype(np.int16) & 0x0F).astype(np.uint8)
    nibbles = nibbles.reshape(N, K)

    data_bytes = _pack_nibbles(nibbles)
    scale_bytes = scale_bf16.contiguous().view(torch.uint8).numpy().tobytes()
    return (data_bytes, scale_bytes)


def dequantize_int4(
    data_bytes: bytes, scale_bytes: bytes, N: int, K: int, block_size: int = 64
) -> torch.Tensor:
    """Inverse of :func:`quantize_int4`. Scale magnitude is used; sign is
    ignored (producer always writes negative)."""
    if K % block_size != 0:
        raise ValueError(f"K={K} must be a multiple of block_size={block_size}")
    num_blocks_k = K // block_size

    scale = torch.frombuffer(
        bytearray(scale_bytes), dtype=torch.bfloat16
    ).clone().reshape(N, num_blocks_k)

    nibbles = _unpack_nibbles(data_bytes, N, K)
    int4_vals = np.where(nibbles >= 8, nibbles - 16, nibbles).astype(np.float32)
    decoded = int4_vals.reshape(N, K)

    abs_scale_expanded = scale.abs().float().repeat_interleave(block_size, dim=1).to(torch.bfloat16)
    return torch.from_numpy(decoded).to(torch.bfloat16) * abs_scale_expanded


# ---------------------------------------------------------------------------
# FP4: E2M1, positive scales (HW FP4 dispatch).
# ---------------------------------------------------------------------------

def quantize_fp4(weight: torch.Tensor, block_size: int = 64) -> tuple[bytes, bytes]:
    """Pack a 2D bf16/float weight as FP4 E2M1 per K-block. Scales are
    written positive so the HW dispatches as FP4."""
    N, K, num_blocks_k = _check_2d_blocked(weight, block_size)
    w = weight.detach().cpu().float()
    blocks = w.view(N, num_blocks_k, block_size)
    abs_max = blocks.abs().amax(dim=-1).clamp(min=1e-8)

    fp4_t = torch.from_numpy(_FP4_VALUES_F32)
    scale_f = abs_max / 6.0
    scale_bf_view = scale_f.to(torch.bfloat16).float()
    scaled = blocks / scale_bf_view.unsqueeze(-1)
    dist = (scaled.unsqueeze(-1) - fp4_t).abs()
    idx = dist.argmin(dim=-1)

    scale_bf16 = scale_f.to(torch.bfloat16)
    nibbles = _FP4_NIBBLES[idx.numpy()].reshape(N, K)

    data_bytes = _pack_nibbles(nibbles)
    scale_bytes = scale_bf16.contiguous().view(torch.uint8).numpy().tobytes()
    return (data_bytes, scale_bytes)


def dequantize_fp4(
    data_bytes: bytes, scale_bytes: bytes, N: int, K: int, block_size: int = 64
) -> torch.Tensor:
    """Inverse of :func:`quantize_fp4`. Scale magnitude is used; sign is
    ignored (producer always writes positive)."""
    if K % block_size != 0:
        raise ValueError(f"K={K} must be a multiple of block_size={block_size}")
    num_blocks_k = K // block_size

    scale = torch.frombuffer(
        bytearray(scale_bytes), dtype=torch.bfloat16
    ).clone().reshape(N, num_blocks_k)

    nibbles = _unpack_nibbles(data_bytes, N, K)
    decoded = _FP4_DECODE_F32[nibbles].reshape(N, K)

    abs_scale_expanded = scale.abs().float().repeat_interleave(block_size, dim=1).to(torch.bfloat16)
    return torch.from_numpy(decoded).to(torch.bfloat16) * abs_scale_expanded


# ---------------------------------------------------------------------------
# IF4: per-block min-MSE pick between INT4 and FP4. Sign bit dispatches.
# ---------------------------------------------------------------------------

def quantize_if4(weight: torch.Tensor, block_size: int = 64) -> tuple[bytes, bytes]:
    """Pack a 2D bf16/float weight as IF4: per K-block, pick INT4 vs FP4
    by min reconstruction MSE. Scale sign carries the per-block codebook
    selection (negative=INT4, positive=FP4)."""
    N, K, num_blocks_k = _check_2d_blocked(weight, block_size)
    w = weight.detach().cpu().float()
    blocks = w.view(N, num_blocks_k, block_size)
    abs_max = blocks.abs().amax(dim=-1).clamp(min=1e-8)

    q4_scale_f = abs_max / 7.0
    q4_scale_bf = q4_scale_f.to(torch.bfloat16).float()
    q4_q = (blocks / q4_scale_bf.unsqueeze(-1)).round().clamp(-8, 7)
    q4_recon = q4_q * q4_scale_bf.unsqueeze(-1)
    q4_err = ((blocks - q4_recon) ** 2).sum(dim=-1)

    fp4_t = torch.from_numpy(_FP4_VALUES_F32)
    fp4_scale_f = abs_max / 6.0
    fp4_scale_bf = fp4_scale_f.to(torch.bfloat16).float()
    scaled = blocks / fp4_scale_bf.unsqueeze(-1)
    dist = (scaled.unsqueeze(-1) - fp4_t).abs()
    fp4_idx = dist.argmin(dim=-1)
    fp4_recon = fp4_t[fp4_idx] * fp4_scale_bf.unsqueeze(-1)
    fp4_err = ((blocks - fp4_recon) ** 2).sum(dim=-1)

    use_q4 = q4_err <= fp4_err
    chosen_scale_f = torch.where(use_q4, q4_scale_f, fp4_scale_f)
    signed_scale_f = torch.where(use_q4, -chosen_scale_f, chosen_scale_f)
    scale_bf16 = signed_scale_f.to(torch.bfloat16)

    q4_nib = (q4_q.to(torch.int8).numpy().astype(np.int16) & 0x0F).astype(np.uint8)
    fp4_nib = _FP4_NIBBLES[fp4_idx.numpy()]
    use_q4_b = use_q4.numpy()[..., None]
    nibbles = np.where(use_q4_b, q4_nib, fp4_nib).reshape(N, K)

    data_bytes = _pack_nibbles(nibbles)
    scale_bytes = scale_bf16.contiguous().view(torch.uint8).numpy().tobytes()
    return (data_bytes, scale_bytes)


def dequantize_if4(
    data_bytes: bytes, scale_bytes: bytes, N: int, K: int, block_size: int = 64
) -> torch.Tensor:
    """Inverse of :func:`quantize_if4`. Routes each block by the sign of
    its bf16 scale: negative -> INT4 codes, positive -> FP4 E2M1."""
    if K % block_size != 0:
        raise ValueError(f"K={K} must be a multiple of block_size={block_size}")
    num_blocks_k = K // block_size

    scale = torch.frombuffer(
        bytearray(scale_bytes), dtype=torch.bfloat16
    ).clone().reshape(N, num_blocks_k)

    nibbles = _unpack_nibbles(data_bytes, N, K)
    int4_vals = np.where(nibbles >= 8, nibbles - 16, nibbles).astype(np.float32)
    fp4_vals = _FP4_DECODE_F32[nibbles]

    int4_blocked = int4_vals.reshape(N, num_blocks_k, block_size)
    fp4_blocked = fp4_vals.reshape(N, num_blocks_k, block_size)
    use_int4 = (scale.float().numpy() < 0)[..., None]
    decoded = np.where(use_int4, int4_blocked, fp4_blocked).reshape(N, K)

    abs_scale_expanded = scale.abs().float().repeat_interleave(block_size, dim=1).to(torch.bfloat16)
    return torch.from_numpy(decoded).to(torch.bfloat16) * abs_scale_expanded


__all__ = [
    "quantize_int4", "dequantize_int4",
    "quantize_fp4",  "dequantize_fp4",
    "quantize_if4",  "dequantize_if4",
]
