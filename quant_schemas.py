"""Block-scale quantization for the unified-engine HW.

Six codebooks (one quantize/dequantize pair each) plus a string-dispatch
wrapper. Same scale-sign convention across all codecs: negative bf16 scale
selects the INT path on chip, positive selects the FP path. Magnitude is
the effective multiplier.

  ``int4`` -- INT4 codes (-8..7), 4-bit nibbles, all-negative bf16 scales.
  ``fp4``  -- FP4 E2M1, 4-bit nibbles, all-positive bf16 scales.
  ``if4``  -- per-block min-MSE pick between INT4 and FP4; the bf16
              scale's sign bit picks the codebook per block on chip.
  ``int8`` -- INT8 codes (-128..127), 1-byte raw, all-negative bf16 scales.
  ``fp8``  -- FP8 E4M3FN (254 finite values), 1-byte raw,
              all-positive bf16 scales.
  ``if8``  -- per-block min-MSE pick between INT8 and FP8; same per-block
              sign-bit dispatch as if4.

Customer-facing entry points: ``quantize(precision, w)`` and
``dequant(precision, ...)``. Templates that may switch codebooks during
development call these so the precision is config-driven (flip a config
string -- no code change). The typed packers are also public for direct
calls when only one codebook is needed.

Wire format (all functions): 2D tensor (N, K) with K % block_size == 0.
Returns flat byte buffers -- 4-bit codecs emit (N, K/2) packed nibbles
(low nibble = even K, high nibble = odd K); 8-bit codecs emit (N, K) raw
bytes. Scale is row-major (N, K/64) bf16. Per-head, N-axis, transposed,
or arbitrary-shape layouts are caller responsibility -- those are
bin-layout concerns, not codec concerns.

Code mappings and value tables match the HW reference in user_dma_core.py
(``UnifiedEngine.quantize_weight`` / ``quantize_weight_simulate``):
the FP4 nibble code table, the FP8 E4M3FN value-by-code table, and the
sign-bit dispatch convention all come from there. Stay byte-compatible
with that file; do not edit code mappings here without a matching HW
update.

Optional ``int_variant`` flag on ``if4`` / ``if8`` (and on the typed
``quantize_if4`` / ``quantize_if8`` packers):
  ``int_variant=None``  (default) -- per-block min-MSE selection.
  ``int_variant=True``  -- force every block to INT (uniform pure INT4/INT8).
  ``int_variant=False`` -- force every block to FP (uniform pure FP4/FP8).
The same flag is exposed at the SW boundary in
``user_dma_core.UnifiedEngine.quantize_weight`` so the two APIs stay in
sync. ``int_variant`` is meaningful only for ``if4`` / ``if8``; passing
it for any other precision raises ``ValueError``.
"""

import math

import numpy as np
import torch


# FP4 E2M1 codebook. Layout matches the HW reference in user_dma_core.py:
#   _FP4_VALUES   -- sorted neg-to-pos (argmin distance table during quantize)
#   _FP4_NIBBLES  -- argmin-index -> HW storage nibble code
#   _FP4_DECODE   -- nibble code -> decoded value (used during dequant)
_FP4_VALUES_BF16 = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0,
      0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0],
    dtype=torch.bfloat16,
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


# FP8 E4M3FN codebook. Indexed by byte code 0x00..0xFF. All 254 finite
# values listed exactly (powers-of-two and 1/8-step fractions thereof;
# every entry is representable in bf16 without rounding). Codes 0x7F /
# 0xFF are NaN under E4M3FN; the snap-to-grid table built below filters
# them out so quantization can never emit a NaN code.
# Negative half (0x80..0xFF) mirrors the positive half with 0x80 = -0.
# Verbatim copy from user_dma_core.UnifiedEngine._FP8_E4M3FN_VALUE_BY_CODE
# -- do not edit one without the other.
_FP8_E4M3FN_VALUE_BY_CODE = (
    # 0x00..0x07  (positive subnormals, E=0)
    0.0, 1/512, 2/512, 3/512, 4/512, 5/512, 6/512, 7/512,
    # 0x08..0x0F  (E=1)
    1.0/64, 1.125/64, 1.25/64, 1.375/64, 1.5/64, 1.625/64, 1.75/64, 1.875/64,
    # 0x10..0x17  (E=2)
    1.0/32, 1.125/32, 1.25/32, 1.375/32, 1.5/32, 1.625/32, 1.75/32, 1.875/32,
    # 0x18..0x1F  (E=3)
    1.0/16, 1.125/16, 1.25/16, 1.375/16, 1.5/16, 1.625/16, 1.75/16, 1.875/16,
    # 0x20..0x27  (E=4)
    1.0/8, 1.125/8, 1.25/8, 1.375/8, 1.5/8, 1.625/8, 1.75/8, 1.875/8,
    # 0x28..0x2F  (E=5)
    1.0/4, 1.125/4, 1.25/4, 1.375/4, 1.5/4, 1.625/4, 1.75/4, 1.875/4,
    # 0x30..0x37  (E=6)
    1.0/2, 1.125/2, 1.25/2, 1.375/2, 1.5/2, 1.625/2, 1.75/2, 1.875/2,
    # 0x38..0x3F  (E=7, bias point: 1.0 .. 1.875)
    1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875,
    # 0x40..0x47  (E=8)
    2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75,
    # 0x48..0x4F  (E=9)
    4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,
    # 0x50..0x57  (E=10)
    8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    # 0x58..0x5F  (E=11)
    16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0,
    # 0x60..0x67  (E=12)
    32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0,
    # 0x68..0x6F  (E=13)
    64.0, 72.0, 80.0, 88.0, 96.0, 104.0, 112.0, 120.0,
    # 0x70..0x77  (E=14)
    128.0, 144.0, 160.0, 176.0, 192.0, 208.0, 224.0, 240.0,
    # 0x78..0x7F  (E=15; 0x7F = +NaN)
    256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, math.nan,
    # 0x80..0x87  (negative subnormals, E=0)
    -0.0, -1/512, -2/512, -3/512, -4/512, -5/512, -6/512, -7/512,
    # 0x88..0x8F
    -1.0/64, -1.125/64, -1.25/64, -1.375/64, -1.5/64, -1.625/64, -1.75/64, -1.875/64,
    # 0x90..0x97
    -1.0/32, -1.125/32, -1.25/32, -1.375/32, -1.5/32, -1.625/32, -1.75/32, -1.875/32,
    # 0x98..0x9F
    -1.0/16, -1.125/16, -1.25/16, -1.375/16, -1.5/16, -1.625/16, -1.75/16, -1.875/16,
    # 0xA0..0xA7
    -1.0/8, -1.125/8, -1.25/8, -1.375/8, -1.5/8, -1.625/8, -1.75/8, -1.875/8,
    # 0xA8..0xAF
    -1.0/4, -1.125/4, -1.25/4, -1.375/4, -1.5/4, -1.625/4, -1.75/4, -1.875/4,
    # 0xB0..0xB7
    -1.0/2, -1.125/2, -1.25/2, -1.375/2, -1.5/2, -1.625/2, -1.75/2, -1.875/2,
    # 0xB8..0xBF
    -1.0, -1.125, -1.25, -1.375, -1.5, -1.625, -1.75, -1.875,
    # 0xC0..0xC7
    -2.0, -2.25, -2.5, -2.75, -3.0, -3.25, -3.5, -3.75,
    # 0xC8..0xCF
    -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -7.0, -7.5,
    # 0xD0..0xD7
    -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0,
    # 0xD8..0xDF
    -16.0, -18.0, -20.0, -22.0, -24.0, -26.0, -28.0, -30.0,
    # 0xE0..0xE7
    -32.0, -36.0, -40.0, -44.0, -48.0, -52.0, -56.0, -60.0,
    # 0xE8..0xEF
    -64.0, -72.0, -80.0, -88.0, -96.0, -104.0, -112.0, -120.0,
    # 0xF0..0xF7
    -128.0, -144.0, -160.0, -176.0, -192.0, -208.0, -224.0, -240.0,
    # 0xF8..0xFF  (0xFF = -NaN)
    -256.0, -288.0, -320.0, -352.0, -384.0, -416.0, -448.0, math.nan,
)

# Numpy decode table indexed by byte code (NaN at 0x7F / 0xFF preserved
# but never emitted by quantize_fp8 because the snap-to-grid table below
# filters them out).
_FP8_DECODE_F32 = np.array(_FP8_E4M3FN_VALUE_BY_CODE, dtype=np.float32)


def _fp8_e4m3fn_value_code_table(device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize the FP8 E4M3FN snap-to-grid lookup as torch tensors.

    Returns ``(values_t, codes_t)`` where ``values_t`` is a float32 tensor
    of the 254 finite FP8 E4M3FN values from ``_FP8_E4M3FN_VALUE_BY_CODE``
    and ``codes_t`` is the matching uint8 tensor of byte codes. The +/-NaN
    entries at 0x7F / 0xFF are filtered out so the table can be used
    directly for absmax snap-to-grid quantization without ever emitting a
    NaN code. Verbatim port of
    ``user_dma_core.UnifiedEngine._fp8_e4m3fn_value_code_table``.
    """
    values = [v for v in _FP8_E4M3FN_VALUE_BY_CODE if not math.isnan(v)]
    codes = [c for c, v in enumerate(_FP8_E4M3FN_VALUE_BY_CODE) if not math.isnan(v)]
    return (torch.tensor(values, dtype=torch.float32, device=device),
            torch.tensor(codes, dtype=torch.uint8, device=device))


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
    return ((high << 4) | low).astype(np.uint8).tobytes()


def _unpack_nibbles(data_bytes: bytes, N: int, K: int) -> np.ndarray:
    packed = np.frombuffer(data_bytes, dtype=np.uint8)[: N * (K // 2)].reshape(N, K // 2)
    low_nib = (packed & 0x0F).astype(np.int16)
    high_nib = ((packed >> 4) & 0x0F).astype(np.int16)
    return np.stack([low_nib, high_nib], axis=-1).reshape(N, K)


def _scale_bytes(scale_bf16: torch.Tensor) -> bytes:
    return scale_bf16.contiguous().view(torch.uint8).numpy().tobytes()


def _read_scale(scale_bytes: bytes, N: int, num_blocks_k: int) -> torch.Tensor:
    return torch.frombuffer(
        bytearray(scale_bytes), dtype=torch.bfloat16
    ).clone().reshape(N, num_blocks_k)


def _expand_abs_scale(scale: torch.Tensor, block_size: int) -> torch.Tensor:
    return scale.abs().float().repeat_interleave(block_size, dim=1).to(torch.bfloat16)


# Chunked FP-grid snap-to-grid lookup. The naive form
# ``(scaled.float().unsqueeze(-1) - values).abs().argmin(dim=-1)`` materializes
# a ``(num_blocks, block_size, num_grid_values)`` fp32 tensor. For the LM head
# of a multi-billion-parameter model and the FP8 grid (254 values) that's
# hundreds of GB and OOMs on workstation RAM. Chunk along ``num_blocks`` so
# peak temporary memory stays bounded.
def _snap_to_grid_argmin(scaled: torch.Tensor, values_t: torch.Tensor) -> torch.Tensor:
    """``scaled``: (num_blocks, block_size) bf16 or fp32.
    ``values_t``: (num_grid_values,) fp32. Returns LongTensor (num_blocks, block_size).

    Implementation: deduplicate then sort the codebook once, binary-search
    each query's insertion point, and pick the nearest of the two adjacent
    sorted entries. O(log V) per element vs O(V) brute-force argmin. The
    monotonic codebook makes this lossless: the nearest neighbour is
    provably one of the two adjacent sorted entries.

    Deduplication handles the FP8 E4M3FN ±0 duplicate (code 0x00 = +0.0,
    code 0x80 = -0.0): we keep the lowest-original-index copy of each
    unique value, so the lookup matches ``torch.argmin``'s "lowest index"
    tie-breaking convention. Result is byte-equal to the brute-force
    version on real weights.
    """
    flat = scaled.float().reshape(-1)
    # Deduplicate: for each unique value, keep the lowest original index.
    # Iterating once at codec load time on a 254-entry codebook is trivial.
    seen: dict = {}
    keep: list = []
    for i, v in enumerate(values_t.tolist()):
        if v not in seen:
            seen[v] = i
            keep.append(i)
    keep_idx = torch.tensor(keep, dtype=torch.long)
    values_unique = values_t[keep_idx]

    # Sort deduped codebook ascending; sort_idx maps sorted position -> dedup position.
    values_sorted, sort_idx = torch.sort(values_unique)
    n_vals = values_sorted.numel()

    # Binary-search each query's insertion point (right=False: ties go to the left).
    hi = torch.searchsorted(values_sorted, flat).clamp(max=n_vals - 1)
    lo = (hi - 1).clamp(min=0)
    d_hi = (values_sorted[hi] - flat).abs()
    d_lo = (flat - values_sorted[lo]).abs()
    dedup_lo = sort_idx[lo]
    dedup_hi = sort_idx[hi]
    # On exact distance ties, pick the lower deduped index (which corresponds
    # to the lower original-codebook index since dedup keeps lowest first).
    pick_lo = (d_lo < d_hi) | ((d_lo == d_hi) & (dedup_lo <= dedup_hi))
    chosen_dedup = torch.where(pick_lo, dedup_lo, dedup_hi)
    # Map dedup index back to the original codebook index.
    out = keep_idx[chosen_dedup]
    return out.view(scaled.shape).long()


# ---------------------------------------------------------------------------
# INT4: codes -8..7, scale stored negative -> HW INT4 dispatch.
# ---------------------------------------------------------------------------

def quantize_int4(weight: torch.Tensor, block_size: int = 64) -> tuple[bytes, bytes]:
    """Pack a 2D weight as INT4 (codes -8..7) per K-block. The bf16 scale
    is stored with a flipped sign so HW dispatches as INT4.

    bf16-throughout division matches the HW reference in
    ``user_dma_core.UnifiedEngine.quantize_weight`` (TYPE.IF4, int_variant=True)
    so output bytes are bit-equal to that path."""
    N, K, _ = _check_2d_blocked(weight, block_size)
    blocks = weight.detach().cpu().to(torch.bfloat16).flatten().view(-1, block_size)

    int4_max = torch.tensor(7.0, dtype=torch.bfloat16)
    scales = (blocks.abs().max(dim=1).values / int4_max).clamp(min=1e-8).to(torch.bfloat16)
    scaled = (blocks / scales[:, None]).to(torch.bfloat16)
    quantized = scaled.round().clamp(-8, 7).to(torch.int8).numpy()

    nibbles = (quantized.astype(np.uint8) & 0x0F).reshape(N, K)
    scale_bf16 = -scales  # negative -> INT4 dispatch
    return _pack_nibbles(nibbles), _scale_bytes(scale_bf16)


def dequantize_int4(
    data_bytes: bytes, scale_bytes: bytes, N: int, K: int, block_size: int = 64
) -> torch.Tensor:
    """Inverse of :func:`quantize_int4`. Scale magnitude is used; sign is
    ignored (producer always writes negative)."""
    if K % block_size != 0:
        raise ValueError(f"K={K} must be a multiple of block_size={block_size}")
    scale = _read_scale(scale_bytes, N, K // block_size)
    nibbles = _unpack_nibbles(data_bytes, N, K)
    decoded = np.where(nibbles >= 8, nibbles - 16, nibbles).astype(np.float32).reshape(N, K)
    return torch.from_numpy(decoded).to(torch.bfloat16) * _expand_abs_scale(scale, block_size)


# ---------------------------------------------------------------------------
# FP4: E2M1, scale stored positive -> HW FP4 dispatch.
# ---------------------------------------------------------------------------

def quantize_fp4(weight: torch.Tensor, block_size: int = 64) -> tuple[bytes, bytes]:
    """Pack a 2D weight as FP4 E2M1 per K-block. The bf16 scale is stored
    positive so HW dispatches as FP4."""
    N, K, _ = _check_2d_blocked(weight, block_size)
    blocks = weight.detach().cpu().to(torch.bfloat16).flatten().view(-1, block_size)

    # Clamp must be applied AFTER the /6 division so the stored scale matches
    # the HW-released convention -- bf16(1e-8) for all-zero blocks.
    fp4_max = torch.tensor(6.0, dtype=torch.bfloat16)
    scales = (blocks.abs().max(dim=1).values / fp4_max).clamp(min=1e-8).to(torch.bfloat16)
    scaled = (blocks / scales[:, None]).to(torch.bfloat16)
    # argmin against neg-to-pos value table, then remap index -> HW nibble.
    idx = torch.argmin(torch.abs(scaled.unsqueeze(-1) - _FP4_VALUES_BF16), dim=-1)
    nibbles = _FP4_NIBBLES[idx.numpy()].reshape(N, K)
    return _pack_nibbles(nibbles), _scale_bytes(scales)


def dequantize_fp4(
    data_bytes: bytes, scale_bytes: bytes, N: int, K: int, block_size: int = 64
) -> torch.Tensor:
    """Inverse of :func:`quantize_fp4`. Scale magnitude is used; sign is
    ignored (producer always writes positive)."""
    if K % block_size != 0:
        raise ValueError(f"K={K} must be a multiple of block_size={block_size}")
    scale = _read_scale(scale_bytes, N, K // block_size)
    nibbles = _unpack_nibbles(data_bytes, N, K)
    decoded = _FP4_DECODE_F32[nibbles].reshape(N, K)
    return torch.from_numpy(decoded).to(torch.bfloat16) * _expand_abs_scale(scale, block_size)


# ---------------------------------------------------------------------------
# IF4: per-block min-MSE pick between INT4 and FP4. Scale sign dispatches.
# ---------------------------------------------------------------------------

def quantize_if4(weight: torch.Tensor, block_size: int = 64,
                 int_variant: "bool | None" = None) -> tuple[bytes, bytes]:
    """Pack a 2D weight as IF4: per K-block, pick INT4 or FP4 by min
    reconstruction MSE. Scale sign carries the per-block selection
    (negative -> INT4, positive -> FP4).

    ``int_variant``:
      ``None`` (default) -- per-block MixMSE selection.
      ``True``  -- force every block to INT4 (uniform; bytes byte-equal
                    to ``quantize_int4``).
      ``False`` -- force every block to FP4 (uniform; bytes byte-equal
                    to ``quantize_fp4``).
    """
    N, K, num_blocks_k = _check_2d_blocked(weight, block_size)
    w_bf = weight.detach().cpu().to(torch.bfloat16)
    blocks_f = w_bf.float().view(N, num_blocks_k, block_size)
    blocks_bf = w_bf.view(N, num_blocks_k, block_size)
    abs_max = blocks_f.abs().amax(dim=-1).clamp(min=1e-8)

    # INT4 candidate. The intermediate division is done in fp32
    int4_scale_bf = (abs_max / 7.0).to(torch.bfloat16)
    # int4_scaled = (blocks_bf / int4_scale_bf.unsqueeze(-1)).to(torch.bfloat16)
    int4_scaled = blocks_f / int4_scale_bf.float().unsqueeze(-1)
    int4_q = int4_scaled.round().clamp(-8, 7)
    int4_recon = int4_q * int4_scale_bf.float().unsqueeze(-1)
    int4_err = ((blocks_f - int4_recon) ** 2).sum(dim=-1)

    # FP4 candidate: argmin against neg-to-pos value table, then remap to HW nibble.
    fp4_v = _FP4_VALUES_BF16.float()
    fp4_scale_bf = (abs_max / 6.0).to(torch.bfloat16)
    # fp4_scaled = (blocks_bf / fp4_scale_bf.unsqueeze(-1)).to(torch.bfloat16)
    fp4_scaled = blocks_f / fp4_scale_bf.float().unsqueeze(-1)
    fp4_idx = (fp4_scaled.float().unsqueeze(-1) - fp4_v).abs().argmin(dim=-1)
    fp4_recon = fp4_v[fp4_idx] * fp4_scale_bf.float().unsqueeze(-1)
    fp4_err = ((blocks_f - fp4_recon) ** 2).sum(dim=-1)

    use_q4 = int4_err <= fp4_err
    if int_variant is True:
        use_q4 = torch.ones_like(use_q4)
    elif int_variant is False:
        use_q4 = torch.zeros_like(use_q4)

    scale_bf16 = torch.where(use_q4, -int4_scale_bf, fp4_scale_bf)

    q4_nib = (int4_q.to(torch.int8).numpy().astype(np.int16) & 0x0F).astype(np.uint8)
    fp4_nib = _FP4_NIBBLES[fp4_idx.numpy()]
    nibbles = np.where(use_q4.numpy()[..., None], q4_nib, fp4_nib).reshape(N, K)
    return _pack_nibbles(nibbles), _scale_bytes(scale_bf16)


def dequantize_if4(
    data_bytes: bytes, scale_bytes: bytes, N: int, K: int, block_size: int = 64
) -> torch.Tensor:
    """Inverse of :func:`quantize_if4`. Per-block routing by scale sign:
    negative -> INT4 codes, positive -> FP4 E2M1."""
    if K % block_size != 0:
        raise ValueError(f"K={K} must be a multiple of block_size={block_size}")
    num_blocks_k = K // block_size
    scale = _read_scale(scale_bytes, N, num_blocks_k)

    nibbles = _unpack_nibbles(data_bytes, N, K)
    int4_blocked = np.where(nibbles >= 8, nibbles - 16, nibbles).astype(np.float32).reshape(
        N, num_blocks_k, block_size)
    fp4_blocked = _FP4_DECODE_F32[nibbles].reshape(N, num_blocks_k, block_size)

    use_int4 = (scale.float().numpy() < 0)[..., None]
    decoded = np.where(use_int4, int4_blocked, fp4_blocked).reshape(N, K)
    return torch.from_numpy(decoded).to(torch.bfloat16) * _expand_abs_scale(scale, block_size)


# ---------------------------------------------------------------------------
# INT8: codes -128..127, raw 1-byte storage, scale stored negative
# -> HW INT8 dispatch.
# ---------------------------------------------------------------------------

def quantize_int8(weight: torch.Tensor, block_size: int = 64) -> tuple[bytes, bytes]:
    """Pack a 2D weight as INT8 (codes -128..127) per K-block. The bf16
    scale is stored with a flipped sign so HW dispatches as INT8. Storage
    is raw bytes (no nibble packing).

    bf16-throughout division matches the HW reference in
    ``user_dma_core.UnifiedEngine.quantize_weight`` (TYPE.IF8, int_variant=True)
    so output bytes are bit-equal to that path."""
    N, K, _ = _check_2d_blocked(weight, block_size)
    blocks = weight.detach().cpu().to(torch.bfloat16).flatten().view(-1, block_size)

    int8_max = torch.tensor(127.0, dtype=torch.bfloat16)
    scales = (blocks.abs().max(dim=1).values / int8_max).clamp(min=1e-8).to(torch.bfloat16)
    scaled = (blocks / scales[:, None]).to(torch.bfloat16)
    quantized = scaled.round().clamp(-128, 127).to(torch.int8)

    # int8 storage shares byte pattern with uint8 (two's complement).
    data_bytes = quantized.numpy().reshape(N, K).tobytes()
    scale_bf16 = -scales  # negative -> INT8 dispatch
    return data_bytes, _scale_bytes(scale_bf16)


def dequantize_int8(
    data_bytes: bytes, scale_bytes: bytes, N: int, K: int, block_size: int = 64
) -> torch.Tensor:
    """Inverse of :func:`quantize_int8`. Scale magnitude is used; sign is
    ignored (producer always writes negative)."""
    if K % block_size != 0:
        raise ValueError(f"K={K} must be a multiple of block_size={block_size}")
    scale = _read_scale(scale_bytes, N, K // block_size)
    decoded = (np.frombuffer(data_bytes, dtype=np.int8)[: N * K]
               .astype(np.float32).reshape(N, K))
    return torch.from_numpy(decoded).to(torch.bfloat16) * _expand_abs_scale(scale, block_size)


# ---------------------------------------------------------------------------
# FP8: E4M3FN, raw 1-byte storage, scale stored positive
# -> HW FP8 dispatch.
# ---------------------------------------------------------------------------

def quantize_fp8(weight: torch.Tensor, block_size: int = 64) -> tuple[bytes, bytes]:
    """Pack a 2D weight as FP8 E4M3FN per K-block. Snap to the 254-entry
    finite-value grid (NaN codes 0x7F / 0xFF are never emitted). Storage
    is raw bytes (no packing). Scale is stored positive so HW dispatches
    as FP8."""
    N, K, _ = _check_2d_blocked(weight, block_size)
    blocks = weight.detach().cpu().to(torch.bfloat16).flatten().view(-1, block_size)

    # Per-block absmax / 448 (E4M3FN finite max), clamped to bf16(1e-8) so
    # all-zero blocks have a well-defined non-zero scale.
    fp8_max = torch.tensor(448.0, dtype=torch.bfloat16)
    scales = (blocks.abs().max(dim=1).values / fp8_max).clamp(min=1e-8).to(torch.bfloat16)
    scaled = (blocks / scales[:, None]).to(torch.bfloat16)

    # Snap to FP8 E4M3FN finite grid; emit the matching byte code.
    fp8_v_t, fp8_c_t = _fp8_e4m3fn_value_code_table()
    idx = _snap_to_grid_argmin(scaled, fp8_v_t)
    codes = fp8_c_t.numpy()[idx.numpy()].astype(np.uint8).reshape(N, K)
    return codes.tobytes(), _scale_bytes(scales)


def dequantize_fp8(
    data_bytes: bytes, scale_bytes: bytes, N: int, K: int, block_size: int = 64
) -> torch.Tensor:
    """Inverse of :func:`quantize_fp8`. Scale magnitude is used; sign is
    ignored (producer always writes positive)."""
    if K % block_size != 0:
        raise ValueError(f"K={K} must be a multiple of block_size={block_size}")
    scale = _read_scale(scale_bytes, N, K // block_size)
    codes = np.frombuffer(data_bytes, dtype=np.uint8)[: N * K].reshape(N, K)
    decoded = _FP8_DECODE_F32[codes]   # NaN at 0x7F/0xFF preserved if present
    return torch.from_numpy(decoded.copy()).to(torch.bfloat16) * _expand_abs_scale(scale, block_size)


# ---------------------------------------------------------------------------
# IF8: per-block min-MSE pick between INT8 and FP8. Scale sign dispatches
# (same convention as IF4).
# ---------------------------------------------------------------------------

def quantize_if8(weight: torch.Tensor, block_size: int = 64,
                 int_variant: "bool | None" = None) -> tuple[bytes, bytes]:
    """Pack a 2D weight as IF8: per K-block, pick INT8 or FP8 by min
    reconstruction MSE. Scale sign carries the per-block selection
    (negative -> INT8, positive -> FP8). Storage is raw bytes.

    ``int_variant``:
      ``None`` (default) -- per-block MixMSE selection.
      ``True``  -- force every block to INT8 (uniform; bytes byte-equal
                    to ``quantize_int8``).
      ``False`` -- force every block to FP8 (uniform; bytes byte-equal
                    to ``quantize_fp8``).
    """
    N, K, num_blocks_k = _check_2d_blocked(weight, block_size)
    w_bf = weight.detach().cpu().to(torch.bfloat16)
    blocks_f = w_bf.float().view(N, num_blocks_k, block_size)
    blocks_bf = w_bf.view(N, num_blocks_k, block_size)
    abs_max = blocks_f.abs().amax(dim=-1).clamp(min=1e-8)

    # INT8 candidate: bf16-throughout division (matches user_dma_core).
    int8_scale_bf = (abs_max / 127.0).to(torch.bfloat16)
    int8_scaled = (blocks_bf / int8_scale_bf.unsqueeze(-1)).to(torch.bfloat16)
    int8_q = int8_scaled.round().clamp(-128, 127)
    int8_recon = int8_q * int8_scale_bf.float().unsqueeze(-1)
    int8_err = ((blocks_f - int8_recon) ** 2).sum(dim=-1)

    # FP8 candidate: snap to E4M3FN finite grid (also bf16-throughout).
    fp8_v_t, fp8_c_t = _fp8_e4m3fn_value_code_table()
    fp8_scale_bf = (abs_max / 448.0).to(torch.bfloat16)
    fp8_scaled = (blocks_bf / fp8_scale_bf.unsqueeze(-1)).to(torch.bfloat16)
    # Chunked snap-to-grid -- the naive (..., 254)-fp32 distance tensor is
    # hundreds of GB on LM-head-sized inputs; chunk along the block axis.
    fp8_idx_flat = _snap_to_grid_argmin(
        fp8_scaled.reshape(-1, block_size), fp8_v_t
    ).view(N, num_blocks_k, block_size)
    fp8_idx = fp8_idx_flat
    fp8_recon = fp8_v_t[fp8_idx] * fp8_scale_bf.float().unsqueeze(-1)
    fp8_err = ((blocks_f - fp8_recon) ** 2).sum(dim=-1)

    use_int = int8_err <= fp8_err
    if int_variant is True:
        use_int = torch.ones_like(use_int)
    elif int_variant is False:
        use_int = torch.zeros_like(use_int)

    scale_bf16 = torch.where(use_int, -int8_scale_bf, fp8_scale_bf)

    # int8 storage shares byte pattern with uint8 (two's complement), so a
    # plain numpy reinterpret cast preserves bits.
    int8_codes = int8_q.to(torch.int8).numpy().astype(np.uint8)
    fp8_codes = fp8_c_t.numpy()[fp8_idx.numpy()]
    bytes_arr = np.where(use_int.numpy()[..., None], int8_codes, fp8_codes).astype(np.uint8).reshape(N, K)
    return bytes_arr.tobytes(), _scale_bytes(scale_bf16)


def dequantize_if8(
    data_bytes: bytes, scale_bytes: bytes, N: int, K: int, block_size: int = 64
) -> torch.Tensor:
    """Inverse of :func:`quantize_if8`. Per-block routing by scale sign:
    negative -> INT8 codes, positive -> FP8 E4M3FN."""
    if K % block_size != 0:
        raise ValueError(f"K={K} must be a multiple of block_size={block_size}")
    num_blocks_k = K // block_size
    scale = _read_scale(scale_bytes, N, num_blocks_k)

    int8_blocked = (np.frombuffer(data_bytes, dtype=np.int8)[: N * K]
                    .reshape(N, num_blocks_k, block_size).astype(np.float32))
    codes = (np.frombuffer(data_bytes, dtype=np.uint8)[: N * K]
             .reshape(N, num_blocks_k, block_size))
    fp8_blocked = _FP8_DECODE_F32[codes]

    use_int = (scale.float().numpy() < 0)[..., None]
    decoded = np.where(use_int, int8_blocked, fp8_blocked).reshape(N, K)
    return torch.from_numpy(decoded.copy()).to(torch.bfloat16) * _expand_abs_scale(scale, block_size)


# ---------------------------------------------------------------------------
# String-dispatched wrappers (config-driven precision).
# ---------------------------------------------------------------------------
# Templates under development call these so the codebook is selected by a
# config string. To switch between codecs, change the config value, not
# the code.
#
# Example:
#     from quant_schemas import quantize, dequant
#     data, scale = quantize(cfg["precision"], weight)   # int4 / fp4 / if4 / int8 / fp8 / if8
#     w_back      = dequant(cfg["precision"], data, scale, N, K)

_QUANTIZERS = {
    "int4": quantize_int4,
    "fp4":  quantize_fp4,
    "if4":  quantize_if4,
    "int8": quantize_int8,
    "fp8":  quantize_fp8,
    "if8":  quantize_if8,
}
_DEQUANTIZERS = {
    "int4": dequantize_int4,
    "fp4":  dequantize_fp4,
    "if4":  dequantize_if4,
    "int8": dequantize_int8,
    "fp8":  dequantize_fp8,
    "if8":  dequantize_if8,
}


def quantize(precision: str, weight: torch.Tensor,
             block_size: int = 64,
             int_variant: "bool | None" = None) -> tuple[bytes, bytes]:
    """Pack ``weight`` using the named codebook.
    ``precision`` ∈ {int4, fp4, if4, int8, fp8, if8}.

    ``int_variant`` (only valid for ``if4`` / ``if8``):
      ``None`` (default) -- per-block MixMSE selection.
      ``True``  -- force every block to the INT path.
      ``False`` -- force every block to the FP path.
    """
    try:
        fn = _QUANTIZERS[precision]
    except KeyError:
        raise ValueError(
            f"unknown precision {precision!r}; valid: {sorted(_QUANTIZERS)}"
        ) from None
    if precision in ("if4", "if8"):
        return fn(weight, block_size=block_size, int_variant=int_variant)
    if int_variant is not None:
        raise ValueError(
            f"int_variant is only valid for 'if4' / 'if8'; got precision={precision!r}"
        )
    return fn(weight, block_size=block_size)


def dequant(precision: str, data_bytes: bytes, scale_bytes: bytes,
            N: int, K: int, block_size: int = 64) -> torch.Tensor:
    """Inverse of :func:`quantize`.
    ``precision`` ∈ {int4, fp4, if4, int8, fp8, if8}."""
    try:
        fn = _DEQUANTIZERS[precision]
    except KeyError:
        raise ValueError(
            f"unknown precision {precision!r}; valid: {sorted(_DEQUANTIZERS)}"
        ) from None
    return fn(data_bytes, scale_bytes, N, K, block_size=block_size)


__all__ = [
    # Recommended customer entry points (config-driven precision).
    "quantize", "dequant",
    # Typed packers (direct access to a single codebook).
    "quantize_int4", "dequantize_int4",
    "quantize_fp4",  "dequantize_fp4",
    "quantize_if4",  "dequantize_if4",
    "quantize_int8", "dequantize_int8",
    "quantize_fp8",  "dequantize_fp8",
    "quantize_if8",  "dequantize_if8",
]
