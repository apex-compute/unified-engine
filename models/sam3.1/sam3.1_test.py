#!/usr/bin/env python3
"""
SAM 3.1 text-prompted segmentation inference on accelerator.

  - Config from sam3.1_config.json; weights from SAM 3.1 HuggingFace checkpoint.
  - Forward pass: ViT backbone → FPN neck → text encoder → encoder fusion
                  → decoder (200 queries) → scoring → segmentation head.
  - All BF16 weights (no quantization). ~820M detector params ≈ 1.64 GB.

Weights:
  - Default: sam3.1_bin/params.bin (generated from HF checkpoint if missing).

Usage:
  python sam3.1_test.py
  python sam3.1_test.py --image photo.jpg --prompt "car"
  python sam3.1_test.py --dev xdma0 [--cycle 5.62]

Fixed layout: sam3.1_test.py, sam3.1_config.json, sam3.1_bin/ live in same folder.
  user_dma_core.py is two folders up (repo root); that directory is added to sys.path.
"""

import builtins
import json
import math
import os
import sys
import warnings
warnings.filterwarnings("ignore", message=".*torchao.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from huggingface_hub import hf_hub_download
import time as _time

_original_print = builtins.print
_SILENT_MODE = False

def quiet_print(*args, **kwargs):
    if _SILENT_MODE:
        return
    _original_print(*args, **kwargs)

builtins.print = quiet_print

import user_dma_core
user_dma_core.MAX_DECODER_INSTRUCTIONS = (0x100000000 - 0x9C000000) // 32  # Section 1: 1.6 GB instruction budget
from user_dma_core import (
    DMA_DEVICE_H2C, DMA_DEVICE_C2H, DRAM_INSTRUCTION_ADDR, TYPE,
    UE_VECTOR_SIZE, URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS,
    set_dma_device, UnifiedEngine,
)


class _CheckpointStop(Exception):
    """Raised to stop compilation at a checkpoint for incremental validation."""
    pass


# =============================================================================
# Helper ops (built on UnifiedEngine primitives)
# =============================================================================

def flash_attention_batched(ue: UnifiedEngine, num_batches: int, head_dim: int,
                            seq_len: int, Q_DRAM_ADDR: int, K_DRAM_ADDR: int,
                            V_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                            SCRATCH_DRAM_ADDR: int,
                            BIAS_DRAM_ADDR: int = None) -> int:
    """Batched flash self-attention: loop over num_batches calling flash_attention_core.

    Each batch is one attention head. Q/K/V/OUTPUT are contiguous per batch with
    stride (seq_len * head_dim). BIAS stride is (seq_len * seq_len) if provided.
    """
    bpe = 2
    qkv_stride = seq_len * head_dim * bpe
    out_stride = seq_len * head_dim * bpe
    scratch_stride = head_dim * seq_len * bpe + seq_len * seq_len * bpe
    bias_stride = seq_len * seq_len * bpe if BIAS_DRAM_ADDR is not None else 0

    total_flops = 0
    for b in range(num_batches):
        bias_addr = BIAS_DRAM_ADDR + b * bias_stride if BIAS_DRAM_ADDR is not None else None
        total_flops += ue.flash_attention_core(
            head_dim=head_dim,
            seq_len=seq_len,
            Q_DRAM_ADDR=Q_DRAM_ADDR + b * qkv_stride,
            K_DRAM_ADDR=K_DRAM_ADDR + b * qkv_stride,
            V_DRAM_ADDR=V_DRAM_ADDR + b * qkv_stride,
            OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR + b * out_stride,
            SCRATCH_DRAM_ADDR=SCRATCH_DRAM_ADDR + b * scratch_stride,
            BIAS_DRAM_ADDR=bias_addr,
        )
    return total_flops


def cross_attention_batched(ue: UnifiedEngine, num_heads: int, head_dim: int,
                            q_len: int, kv_len: int,
                            Q_DRAM_ADDR: int, K_DRAM_ADDR: int, V_DRAM_ADDR: int,
                            OUTPUT_DRAM_ADDR: int, SCRATCH_DRAM_ADDR: int,
                            BIAS_DRAM_ADDR: int = None) -> int:
    """Cross-attention where Q length != K/V length.

    Per-head computation:
        scores = Q @ K^T / sqrt(head_dim)   [q_len, kv_len]
        [+ optional bias]
        attn = softmax(scores)               [q_len, kv_len]
        out = attn @ V                       [q_len, head_dim]

    Layout:
        Q: (num_heads, q_len, head_dim) contiguous in DRAM
        K: (num_heads, kv_len, head_dim)
        V: (num_heads, kv_len, head_dim)
        OUTPUT: (num_heads, q_len, head_dim)
        SCRATCH: per-head needs q_len * kv_len (scores) + q_len * kv_len (softmax)
    """
    bpe = 2
    q_stride = q_len * head_dim * bpe
    kv_stride = kv_len * head_dim * bpe
    out_stride = q_len * head_dim * bpe
    score_size = q_len * kv_len
    scratch_per_head = 2 * score_size * bpe  # scores + softmax output

    total_flops = 0
    scale = 1.0 / math.sqrt(head_dim)

    for h in range(num_heads):
        q_addr = Q_DRAM_ADDR + h * q_stride
        k_addr = K_DRAM_ADDR + h * kv_stride
        v_addr = V_DRAM_ADDR + h * kv_stride
        o_addr = OUTPUT_DRAM_ADDR + h * out_stride
        scores_addr = SCRATCH_DRAM_ADDR + h * scratch_per_head
        softmax_addr = scores_addr + score_size * bpe

        # Step 1: scores = Q @ K^T  → [q_len, kv_len]
        # matmat_mul_core computes A @ B^T; K is already (kv_len, head_dim)
        ue.matmat_mul_core(
            M=q_len, K=head_dim, N=kv_len,
            A_DRAM_ADDR=q_addr,
            B_DRAM_ADDR=k_addr,
            OUTPUT_DRAM_ADDR=scores_addr,
        )
        total_flops += 2 * q_len * head_dim * kv_len

        # Step 2: scale scores by 1/sqrt(head_dim)
        total_elems = q_len * kv_len
        for offset in range(0, total_elems, URAM_NEAR_FULL_ELEMENTS):
            take = min(URAM_NEAR_FULL_ELEMENTS, total_elems - offset)
            ue.accelerator_memory_to_sram(
                accelerator_dram_address=scores_addr + offset * bpe,
                sram_address=0x00000, element_size=take,
            )
            ue.broadcast_mul(
                scalar=scale,
                sram_start_addr=0x00000, sram_wb_addr=0x00000,
                element_size=take,
            )
            ue.sram_to_accelerator_memory(
                sram_address=0x00000,
                accelerator_dram_address=scores_addr + offset * bpe,
                element_size=take,
            )

        # Step 3: add bias if provided
        if BIAS_DRAM_ADDR is not None:
            bias_per_head = q_len * kv_len * bpe
            bias_addr = BIAS_DRAM_ADDR + h * bias_per_head
            for offset in range(0, total_elems, URAM_NEAR_FULL_ELEMENTS):
                take = min(URAM_NEAR_FULL_ELEMENTS, total_elems - offset)
                ue.accelerator_memory_to_sram(
                    accelerator_dram_address=scores_addr + offset * bpe,
                    sram_address=0x00000, element_size=take,
                )
                ue.accelerator_memory_to_sram(
                    accelerator_dram_address=bias_addr + offset * bpe,
                    sram_address=0x80000, element_size=take,
                )
                ue.eltwise_add_core(
                    vector_A_sram_start_addr=0x00000,
                    vector_B_sram_start_addr=0x80000,
                    vector_C_sram_wb_addr=0x00000,
                    element_size=take,
                )
                ue.sram_to_accelerator_memory(
                    sram_address=0x00000,
                    accelerator_dram_address=scores_addr + offset * bpe,
                    element_size=take,
                )

        # Step 4: row-wise softmax
        # NOTE: Requires hardware softmax primitive or manual exp/sum/div.
        # Using softmax_row_core if available; otherwise this is a placeholder
        # that must be replaced with the actual hardware implementation.
        # For each row of length kv_len: max → sub → exp → sum → div
        ue.softmax_rows_core_dram(
            M=q_len, N=kv_len,
            INPUT_DRAM_ADDR=scores_addr,
            OUTPUT_DRAM_ADDR=softmax_addr,
        )
        total_flops += 5 * q_len * kv_len  # approx for softmax

        # Step 5: out = softmax @ V  → [q_len, head_dim]
        # A = softmax [q_len, kv_len], we want A @ V [kv_len, head_dim]
        # matmat_mul_core computes A @ B^T, so B must be V^T = [head_dim, kv_len]
        # We need V transposed. Store V^T in scratch after softmax.
        v_t_addr = softmax_addr + score_size * bpe  # temp for V^T
        ue.bf16_permute_core(
            dim_0=kv_len, dim_1=head_dim, dim_2=1,
            INPUT_DRAM_ADDR=v_addr,
            OUTPUT_DRAM_ADDR=v_t_addr,
        )
        ue.matmat_mul_core(
            M=q_len, K=kv_len, N=head_dim,
            A_DRAM_ADDR=softmax_addr,
            B_DRAM_ADDR=v_t_addr,
            OUTPUT_DRAM_ADDR=o_addr,
        )
        total_flops += 2 * q_len * kv_len * head_dim

    return total_flops


def rope_2d_heads_dram(ue: UnifiedEngine, num_batches: int, seq_len: int, head_dim: int,
                       QK_DRAM_ADDR: int,
                       COS_LO_DRAM_ADDR: int, COS_HI_DRAM_ADDR: int,
                       SIN_LO_DRAM_ADDR: int, SIN_HI_DRAM_ADDR: int) -> int:
    """Apply 2D RoPE (rotate-half) in-place to Q or K heads.

    head_dim=64 → half=32 elements=64 bytes, which is not URAM-row-aligned (128B).
    Following the SmolVLM2 pattern: bulk-load cos/sin tables into URAM_B once,
    then per-position stage x_lo/x_hi through fixed 128B-aligned URAM_A slots.

    URAM_A layout (6 fixed slots × 128 bytes):
      x_lo   0x000   x_hi   0x080
      a_lo   0x100   a_hi   0x180
      out_lo 0x200   out_hi 0x280

    URAM_B layout (bulk-loaded once, 4 × seq_len × 128 bytes ≈ 288KB for seq=576):
      cos_lo  0x80000
      cos_hi  cos_lo + seq_len × SLOT
      sin_lo  cos_hi + seq_len × SLOT
      sin_hi  sin_lo + seq_len × SLOT
      b_lo    sin_hi + seq_len × SLOT   (scratch)
      b_hi    b_lo + SLOT               (scratch)

    eltwise_add/mul require A and B in different URAMs:
      muls: x/a in URAM_A, cos/sin/b in URAM_B
      adds: a in URAM_A, b in URAM_B → out in URAM_A
    """
    bpe  = 2
    SLOT = 128           # bytes per 128B-aligned URAM row (64 elements × 2 bytes)
    half = head_dim // 2 # 32 elements = 64 bytes

    # URAM_A — fixed 128B-aligned staging and result slots
    x_lo   = 0x000;  x_hi   = 0x080
    a_lo   = 0x100;  a_hi   = 0x180
    out_lo = 0x200;  out_hi = 0x280

    # URAM_B — cos/sin tables bulk-loaded once, b scratch slots after tables
    cos_lo_uram = 0x80000
    cos_hi_uram = cos_lo_uram + seq_len * SLOT
    sin_lo_uram = cos_hi_uram + seq_len * SLOT
    sin_hi_uram = sin_lo_uram + seq_len * SLOT
    b_lo = sin_hi_uram + seq_len * SLOT
    b_hi = b_lo + SLOT

    ue.accelerator_memory_to_sram(COS_LO_DRAM_ADDR, cos_lo_uram, seq_len * UE_VECTOR_SIZE)
    ue.accelerator_memory_to_sram(COS_HI_DRAM_ADDR, cos_hi_uram, seq_len * UE_VECTOR_SIZE)
    ue.accelerator_memory_to_sram(SIN_LO_DRAM_ADDR, sin_lo_uram, seq_len * UE_VECTOR_SIZE)
    ue.accelerator_memory_to_sram(SIN_HI_DRAM_ADDR, sin_hi_uram, seq_len * UE_VECTOR_SIZE)

    for batch in range(num_batches):
        batch_base = QK_DRAM_ADDR + batch * seq_len * head_dim * bpe
        for t in range(seq_len):
            row_dram = batch_base + t * head_dim * bpe
            cl = cos_lo_uram + t * SLOT;  ch = cos_hi_uram + t * SLOT
            sl = sin_lo_uram + t * SLOT;  sh = sin_hi_uram + t * SLOT

            # Stage x_lo and x_hi into aligned URAM_A slots
            ue.accelerator_memory_to_sram(row_dram,              x_lo, half)
            ue.accelerator_memory_to_sram(row_dram + half * bpe, x_hi, half)

            # a = x * cos (per half)
            ue.eltwise_mul_core(x_lo, cl, a_lo, half)
            ue.eltwise_mul_core(x_hi, ch, a_hi, half)

            # b = cross terms; sin_lo is pre-negated so b_lo = -x_hi * |sin_lo|
            ue.eltwise_mul_core(x_hi, sl, b_lo, half)
            ue.eltwise_mul_core(x_lo, sh, b_hi, half)

            # out = a + b, written to separate result slots (avoids clobbering x staging)
            ue.eltwise_add_core(a_lo, b_lo, out_lo, half)
            ue.eltwise_add_core(a_hi, b_hi, out_hi, half)

            # Write back in-place
            ue.sram_to_accelerator_memory(out_lo, row_dram,              half)
            ue.sram_to_accelerator_memory(out_hi, row_dram + half * bpe, half)

    return 4 * num_batches * seq_len * head_dim


def rope_2d_vectorized_dram(ue: UnifiedEngine, num_batches: int, seq_len: int,
                            head_dim: int,
                            QK_DRAM_ADDR: int,
                            COS_LO_DRAM_ADDR: int, COS_HI_DRAM_ADDR: int,
                            SIN_LO_DRAM_ADDR: int, SIN_HI_DRAM_ADDR: int) -> int:
    """Apply 2D RoPE (rotate-half) in-place — vectorized across ALL positions.

    Requires PADDED Q/K heads: each position stored as
      [lo(32), zeros(32), hi(32), zeros(32)] = head_dim_pad=128 elements.

    Lo and hi each occupy a full URAM row (64 elements = 128 bytes), so
    strided gather uses chunk=128 (full URAM row) — no alignment issue.

    Instructions per call: 4 (table loads) + num_batches × 10
    For local blocks: 4 + 144 × 10 = 1,444  vs  829,440 per-position → 575× fewer.

    URAM_A layout (4 blocks × seq_len × 64 × 2 bytes = 4 × 73,728 bytes ≈ 288 KB):
      x_lo  0x00000   [seq_len × 64 elements — valid in [0:32], zeros in [32:64] per row]
      x_hi  x_lo + BLOCK
      a_lo  x_hi + BLOCK
      a_hi  a_lo + BLOCK

    URAM_B layout (4 table blocks + 2 scratch blocks ≈ 432 KB):
      cos_lo  0x80000
      cos_hi  cos_lo + BLOCK
      sin_lo  cos_hi + BLOCK
      sin_hi  sin_lo + BLOCK
      b_lo    sin_hi + BLOCK
      b_hi    b_lo + BLOCK

    Strided DMA (gather lo from padded [lo_pad | hi_pad] layout):
      stride_bytes_per_chunk = UE_VECTOR_SIZE * bpe = 128  ← full URAM row ✓
      stride_jump_bytes      = head_dim * bpe = 256        ← full padded position
    For hi: same params but src offset +128 bytes (second row of each position).
    """
    from user_dma_core import UE_VECTOR_SIZE
    bpe    = 2
    VS     = UE_VECTOR_SIZE          # 64 elements = 128 bytes = one URAM row
    chunk  = VS * bpe                # 128 bytes — full URAM row ✓
    jump   = head_dim * bpe          # 256 bytes — stride to next position (head_dim=128)
    total  = seq_len * VS            # 576 × 64 = 36,864 elements (for eltwise + table load)
    BLOCK  = total * bpe             # 73,728 bytes per buffer block

    # URAM_A — 4 contiguous blocks
    x_lo = 0x00000
    x_hi = x_lo + BLOCK
    a_lo = x_hi + BLOCK
    a_hi = a_lo + BLOCK

    # URAM_B — 4 table blocks + 2 scratch blocks
    cos_lo_uram = 0x80000
    cos_hi_uram = cos_lo_uram + BLOCK
    sin_lo_uram = cos_hi_uram + BLOCK
    sin_hi_uram = sin_lo_uram + BLOCK
    b_lo        = sin_hi_uram + BLOCK
    b_hi        = b_lo + BLOCK

    # Bulk-load all 4 cos/sin padded tables into URAM_B once (reused across all batches).
    # Tables shape: (seq_len, 64) padded — valid in [0:32], zeros in [32:64].
    ue.accelerator_memory_to_sram(COS_LO_DRAM_ADDR, cos_lo_uram, total)
    ue.accelerator_memory_to_sram(COS_HI_DRAM_ADDR, cos_hi_uram, total)
    ue.accelerator_memory_to_sram(SIN_LO_DRAM_ADDR, sin_lo_uram, total)
    ue.accelerator_memory_to_sram(SIN_HI_DRAM_ADDR, sin_hi_uram, total)

    for batch in range(num_batches):
        batch_base = QK_DRAM_ADDR + batch * seq_len * head_dim * bpe

        # Strided gather: load all positions' lo rows into x_lo, hi rows into x_hi.
        # Each position: row 0 = [lo(32), zeros(32)], row 1 = [hi(32), zeros(32)].
        ue.accelerator_memory_to_sram(batch_base,         x_lo, total,
                                      stride_bytes_per_chunk=chunk,
                                      stride_jump_bytes=jump)
        ue.accelerator_memory_to_sram(batch_base + chunk, x_hi, total,
                                      stride_bytes_per_chunk=chunk,
                                      stride_jump_bytes=jump)

        # Vectorized rotate-half over all seq_len positions at once.
        # Padding zeros * table zeros = 0 — harmless, does not affect valid elements.
        ue.eltwise_mul_core(x_lo, cos_lo_uram, a_lo,  total)  # a_lo = x_lo * cos_lo
        ue.eltwise_mul_core(x_hi, cos_hi_uram, a_hi,  total)  # a_hi = x_hi * cos_hi
        ue.eltwise_mul_core(x_hi, sin_lo_uram, b_lo,  total)  # b_lo = x_hi * sin_lo (negated)
        ue.eltwise_mul_core(x_lo, sin_hi_uram, b_hi,  total)  # b_hi = x_lo * sin_hi
        ue.eltwise_add_core(a_lo, b_lo,         x_lo, total)  # out_lo = a_lo + b_lo
        ue.eltwise_add_core(a_hi, b_hi,         x_hi, total)  # out_hi = a_hi + b_hi

        # Strided scatter: write results back to padded DRAM positions.
        ue.sram_to_accelerator_memory(x_lo, batch_base,         total,
                                      stride_bytes_per_chunk=chunk,
                                      stride_jump_bytes=jump)
        ue.sram_to_accelerator_memory(x_hi, batch_base + chunk, total,
                                      stride_bytes_per_chunk=chunk,
                                      stride_jump_bytes=jump)

    return 4 + num_batches * 10


def window_partition_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                          H: int, W: int, C: int, window_size: int) -> None:
    """Rearrange (H, W, C) -> (num_windows, window_size*window_size, C) in DRAM.

    Each window is a contiguous (window_size*window_size, C) block in the output.
    Gathers rows from the spatial layout using strided DMA.
    Same pattern as swin_test.py, but batches rows to stay within URAM_A (512KB)
    when window_size * C is large.

    Data layout assumption: input is row-major (H, W, C) in DRAM.
    """
    bytes_per_element = 2
    num_windows_h = H // window_size
    num_windows_w = W // window_size
    input_row_stride = W * C * bytes_per_element
    window_elements = window_size * window_size * C

    row_elements = window_size * C
    row_bytes = row_elements * bytes_per_element
    # How many window-rows fit in URAM_A (0x00000–0x7FFFF = 512KB)?
    uram_a_bytes = 0x80000  # 524288
    rows_per_batch = min(window_size, uram_a_bytes // row_bytes)

    output_offset = 0
    for wh in range(num_windows_h):
        for ww in range(num_windows_w):
            window_src = INPUT_DRAM_ADDR + (wh * window_size * W + ww * window_size) * C * bytes_per_element

            for batch_start in range(0, window_size, rows_per_batch):
                batch_rows = min(rows_per_batch, window_size - batch_start)
                # Gather batch_rows from input into contiguous SRAM
                for r in range(batch_rows):
                    row_src = window_src + (batch_start + r) * input_row_stride
                    ue.accelerator_memory_to_sram(
                        accelerator_dram_address=row_src,
                        sram_address=r * row_bytes,
                        element_size=row_elements,
                    )
                # Flush batch to contiguous output
                ue.sram_to_accelerator_memory(
                    sram_address=0x00000,
                    accelerator_dram_address=OUTPUT_DRAM_ADDR + output_offset + batch_start * row_bytes,
                    element_size=batch_rows * row_elements,
                )
            output_offset += window_elements * bytes_per_element


def window_reverse_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                        H: int, W: int, C: int, window_size: int) -> None:
    """Rearrange (num_windows, window_size*window_size, C) -> (H, W, C) in DRAM.

    Inverse of window_partition_dram. Reads contiguous window rows from input,
    then strided-writes them back to their spatial positions.
    Same pattern as swin_test.py, batched for large windows.
    """
    bytes_per_element = 2
    num_windows_h = H // window_size
    num_windows_w = W // window_size
    output_row_stride = W * C * bytes_per_element
    window_elements = window_size * window_size * C

    row_elements = window_size * C
    row_bytes = row_elements * bytes_per_element
    uram_a_bytes = 0x80000
    rows_per_batch = min(window_size, uram_a_bytes // row_bytes)

    input_offset = 0
    for wh in range(num_windows_h):
        for ww in range(num_windows_w):
            window_dst = OUTPUT_DRAM_ADDR + (wh * window_size * W + ww * window_size) * C * bytes_per_element

            for batch_start in range(0, window_size, rows_per_batch):
                batch_rows = min(rows_per_batch, window_size - batch_start)
                # Read batch of contiguous rows from input
                ue.accelerator_memory_to_sram(
                    accelerator_dram_address=INPUT_DRAM_ADDR + input_offset + batch_start * row_bytes,
                    sram_address=0x00000,
                    element_size=batch_rows * row_elements,
                )
                # Scatter rows to their spatial positions
                for r in range(batch_rows):
                    row_dst = window_dst + (batch_start + r) * output_row_stride
                    ue.sram_to_accelerator_memory(
                        sram_address=r * row_bytes,
                        accelerator_dram_address=row_dst,
                        element_size=row_elements,
                    )
            input_offset += window_elements * bytes_per_element


def dram_zero_fill(ue: UnifiedEngine, DRAM_ADDR: int, num_elements: int) -> None:
    """Fill a DRAM region with zeros using SRAM as staging."""
    bpe = 2
    chunk = min(URAM_NEAR_FULL_ELEMENTS, num_elements)
    zeros = torch.zeros(chunk, dtype=torch.bfloat16)
    z_addr = ue.get_params_dram_addr()
    ue.allocate_params_dram(chunk * bpe)
    ue.dma_write(DMA_DEVICE_H2C, z_addr, zeros, chunk * bpe)
    ue.accelerator_memory_to_sram(accelerator_dram_address=z_addr,
                                   sram_address=0x00000, element_size=chunk)
    offset = 0
    while offset < num_elements:
        take = min(chunk, num_elements - offset)
        ue.sram_to_accelerator_memory(sram_address=0x00000,
                                       accelerator_dram_address=DRAM_ADDR + offset * bpe,
                                       element_size=take)
        offset += take


def multihead_reshape_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int,
                           OUTPUT_DRAM_ADDR: int, TEMP_DRAM_ADDR: int,
                           seq_len: int, num_heads: int,
                           head_dim: int, head_dim_pad: int) -> None:
    """Reshape (seq_len, num_heads * head_dim) -> (num_heads, seq_len, head_dim_pad).

    For SAM3 ViT (head_dim=64, pad=64): no padding needed, just permute.
    For encoder/decoder (head_dim=32, pad=64): pad then permute.

    Input:  row-major (seq_len, num_heads * head_dim)
    Output: row-major (num_heads, seq_len, head_dim_pad)
    """
    bpe = 2
    dim = num_heads * head_dim

    if head_dim == head_dim_pad:
        # No padding needed — just permute (seq, heads, hd) → (heads, seq, hd)
        ue.bf16_permute_core(
            dim_0=seq_len, dim_1=num_heads, dim_2=head_dim,
            INPUT_DRAM_ADDR=INPUT_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
        )
    else:
        # Pad head_dim then permute (same logic as swin's multihead_pad_and_permute
        # but without the per-window loop since we handle full sequences)
        padded_dim = num_heads * head_dim_pad
        dram_zero_fill(ue, TEMP_DRAM_ADDR, seq_len * padded_dim)

        zeros_addr = ue.get_params_dram_addr()
        ue.allocate_params_dram(UE_VECTOR_SIZE * bpe)
        ue.dma_write(DMA_DEVICE_H2C, zeros_addr,
                     torch.zeros(UE_VECTOR_SIZE, dtype=torch.bfloat16),
                     UE_VECTOR_SIZE * bpe)

        for h in range(num_heads):
            for row in range(seq_len):
                src = INPUT_DRAM_ADDR + (row * dim + h * head_dim) * bpe
                dst = TEMP_DRAM_ADDR + (row * padded_dim + h * head_dim_pad) * bpe
                ue.accelerator_memory_to_sram(
                    accelerator_dram_address=zeros_addr,
                    sram_address=0x00000, element_size=UE_VECTOR_SIZE)
                ue.accelerator_memory_to_sram(
                    accelerator_dram_address=src,
                    sram_address=0x00000, element_size=head_dim)
                ue.sram_to_accelerator_memory(
                    sram_address=0x00000,
                    accelerator_dram_address=dst, element_size=UE_VECTOR_SIZE)

        # Permute (seq, heads, hd_pad) → (heads, seq, hd_pad)
        ue.bf16_permute_core(
            dim_0=seq_len, dim_1=num_heads, dim_2=head_dim_pad,
            INPUT_DRAM_ADDR=TEMP_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
        )


def multihead_merge_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int,
                         OUTPUT_DRAM_ADDR: int, TEMP_DRAM_ADDR: int,
                         seq_len: int, num_heads: int,
                         head_dim: int, head_dim_pad: int,
                         UNPAD_WEIGHT_ADDR: int = None) -> None:
    """Reshape (num_heads, seq_len, head_dim_pad) -> (seq_len, num_heads * head_dim).

    Inverse of multihead_reshape_dram. Permute + optional unpad via matmul.
    """
    bpe = 2
    dim = num_heads * head_dim

    # Permute (heads, seq, hd_pad) → (seq, heads, hd_pad)
    ue.bf16_permute_core(
        dim_0=num_heads, dim_1=seq_len, dim_2=head_dim_pad,
        INPUT_DRAM_ADDR=INPUT_DRAM_ADDR,
        OUTPUT_DRAM_ADDR=TEMP_DRAM_ADDR,
    )

    if head_dim == head_dim_pad:
        # Already correct layout — copy to output (or use TEMP as output)
        total = seq_len * dim
        for offset in range(0, total, URAM_NEAR_FULL_ELEMENTS):
            take = min(URAM_NEAR_FULL_ELEMENTS, total - offset)
            ue.accelerator_memory_to_sram(
                accelerator_dram_address=TEMP_DRAM_ADDR + offset * bpe,
                sram_address=0x00000, element_size=take)
            ue.sram_to_accelerator_memory(
                sram_address=0x00000,
                accelerator_dram_address=OUTPUT_DRAM_ADDR + offset * bpe,
                element_size=take)
    else:
        # Unpad via matmul: (seq, heads*hd_pad) @ unpad_weight → (seq, heads*hd)
        padded_dim = num_heads * head_dim_pad
        ue.matmat_mul_core(
            M=seq_len, K=padded_dim, N=dim,
            A_DRAM_ADDR=TEMP_DRAM_ADDR,
            B_DRAM_ADDR=UNPAD_WEIGHT_ADDR,
            OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
        )


def conv2d_1x1_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                     WEIGHT_DRAM_ADDR: int, BIAS_DRAM_ADDR: int,
                     H: int, W: int, C_in: int, C_out: int,
                     gelu_enable: bool = False, relu_enable: bool = False) -> int:
    """Conv2d with 1x1 kernel on HWC data = matmul.

    Input:  (H*W, C_in) row-major in DRAM
    Weight: (C_out, C_in) — PyTorch Conv2d weight[:,:,0,0] reshaped
    Output: (H*W, C_out) row-major in DRAM
    """
    M = H * W
    ue.matmat_mul_core(
        M=M, K=C_in, N=C_out,
        A_DRAM_ADDR=INPUT_DRAM_ADDR,
        B_DRAM_ADDR=WEIGHT_DRAM_ADDR,
        OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
        C_DRAM_ADDR=BIAS_DRAM_ADDR,
        bias_mode="broadcast_N",
        gelu_enable=gelu_enable,
    )
    return 2 * M * C_in * C_out


def conv2d_3x3_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                     IM2COL_DRAM_ADDR: int,
                     WEIGHT_DRAM_ADDR: int, BIAS_DRAM_ADDR: int,
                     H: int, W: int, C_in: int, C_out: int,
                     relu_enable: bool = False) -> int:
    """Conv2d with 3x3 kernel, padding=1, stride=1 on HWC data.

    1. im2col: gather 3x3 neighborhoods → (H*W, 9*C_in)
    2. matmul: (H*W, 9*C_in) @ (C_out, 9*C_in)^T + bias → (H*W, C_out)

    Weight must be pre-reshaped from (C_out, C_in, 3, 3) to (C_out, 9*C_in).
    """
    bpe = 2
    K = 9 * C_in
    M = H * W

    # Step 1: im2col — for each (h, w), gather 3x3 neighborhood with zero padding
    for h in range(H):
        for w_start in range(0, W, max(1, URAM_NEAR_FULL_ELEMENTS // K)):
            w_end = min(w_start + max(1, URAM_NEAR_FULL_ELEMENTS // K), W)
            w_count = w_end - w_start
            sram_offset = 0
            for w in range(w_start, w_end):
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        nh, nw = h + dy, w + dx
                        patch_sram = sram_offset + ((dy + 1) * 3 + (dx + 1)) * C_in * bpe
                        if 0 <= nh < H and 0 <= nw < W:
                            src = INPUT_DRAM_ADDR + (nh * W + nw) * C_in * bpe
                            ue.accelerator_memory_to_sram(
                                accelerator_dram_address=src,
                                sram_address=patch_sram, element_size=C_in)
                        # else: already zero from zero-fill (or need explicit zero)
                sram_offset += K * bpe

            out_offset = (h * W + w_start) * K * bpe
            ue.sram_to_accelerator_memory(
                sram_address=0x00000,
                accelerator_dram_address=IM2COL_DRAM_ADDR + out_offset,
                element_size=w_count * K)

    # Step 2: matmul
    ue.matmat_mul_core(
        M=M, K=K, N=C_out,
        A_DRAM_ADDR=IM2COL_DRAM_ADDR,
        B_DRAM_ADDR=WEIGHT_DRAM_ADDR,
        OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
        C_DRAM_ADDR=BIAS_DRAM_ADDR,
        bias_mode="broadcast_N",
    )
    return 2 * M * K * C_out


def conv_transpose2d_2x2_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int,
                               OUTPUT_DRAM_ADDR: int, TEMP_DRAM_ADDR: int,
                               WEIGHT_SLICES: list, BIAS_DRAM_ADDR: int,
                               H: int, W: int, C_in: int, C_out: int) -> int:
    """ConvTranspose2d with kernel=2, stride=2 on HWC data.

    Input:  (H, W, C_in) = (H*W, C_in) row-major
    Output: (2H, 2W, C_out) = (4*H*W, C_out) row-major

    Decomposes into 4 matmuls (one per kernel position), each producing
    output at interleaved positions:
        (ky, kx) in {(0,0), (0,1), (1,0), (1,1)}
        output[2h+ky, 2w+kx, :] = input[h, w, :] @ weight[:,:,ky,kx]^T + bias

    WEIGHT_SLICES: list of 4 DRAM addresses for weight[:,:,ky,kx]^T,
                   each shape (C_out, C_in) ready for matmat_mul_core.
    TEMP_DRAM_ADDR: scratch for matmul output (H*W, C_out).
    """
    bpe = 2
    M = H * W
    out_H, out_W = 2 * H, 2 * W

    total_flops = 0
    for idx, (ky, kx) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        # Matmul: (H*W, C_in) @ (C_out, C_in)^T → (H*W, C_out)
        ue.matmat_mul_core(
            M=M, K=C_in, N=C_out,
            A_DRAM_ADDR=INPUT_DRAM_ADDR,
            B_DRAM_ADDR=WEIGHT_SLICES[idx],
            OUTPUT_DRAM_ADDR=TEMP_DRAM_ADDR,
            C_DRAM_ADDR=BIAS_DRAM_ADDR if idx == 0 else None,
            bias_mode="broadcast_N" if idx == 0 else None,
        )
        total_flops += 2 * M * C_in * C_out

        # Scatter: for each input position (h, w), write to output[2h+ky, 2w+kx, :]
        for h in range(H):
            for w in range(W):
                src = TEMP_DRAM_ADDR + (h * W + w) * C_out * bpe
                oh, ow = 2 * h + ky, 2 * w + kx
                dst = OUTPUT_DRAM_ADDR + (oh * out_W + ow) * C_out * bpe
                ue.accelerator_memory_to_sram(
                    accelerator_dram_address=src,
                    sram_address=0x00000, element_size=C_out)
                ue.sram_to_accelerator_memory(
                    sram_address=0x00000,
                    accelerator_dram_address=dst, element_size=C_out)

        # For positions (ky, kx) != (0, 0), add bias separately
        if idx > 0 and BIAS_DRAM_ADDR is not None:
            ue.accelerator_memory_to_sram(
                accelerator_dram_address=BIAS_DRAM_ADDR,
                sram_address=0x80000, element_size=C_out)
            for h in range(H):
                for w in range(W):
                    oh, ow = 2 * h + ky, 2 * w + kx
                    addr = OUTPUT_DRAM_ADDR + (oh * out_W + ow) * C_out * bpe
                    ue.accelerator_memory_to_sram(
                        accelerator_dram_address=addr,
                        sram_address=0x00000, element_size=C_out)
                    ue.eltwise_add_core(
                        vector_A_sram_start_addr=0x00000,
                        vector_B_sram_start_addr=0x80000,
                        vector_C_sram_wb_addr=0x00000,
                        element_size=C_out)
                    ue.sram_to_accelerator_memory(
                        sram_address=0x00000,
                        accelerator_dram_address=addr, element_size=C_out)

    return total_flops


def nearest_upsample_2x_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int,
                              OUTPUT_DRAM_ADDR: int,
                              H: int, W: int, C: int) -> None:
    """Nearest-neighbor 2x upsample: (H, W, C) → (2H, 2W, C) in DRAM.

    Each input pixel is duplicated into a 2x2 output block.
    """
    bpe = 2
    out_W = 2 * W
    for h in range(H):
        for w in range(W):
            src = INPUT_DRAM_ADDR + (h * W + w) * C * bpe
            ue.accelerator_memory_to_sram(
                accelerator_dram_address=src,
                sram_address=0x00000, element_size=C)
            # Write to 4 output positions
            for dy in range(2):
                for dx in range(2):
                    oh, ow = 2 * h + dy, 2 * w + dx
                    dst = OUTPUT_DRAM_ADDR + (oh * out_W + ow) * C * bpe
                    ue.sram_to_accelerator_memory(
                        sram_address=0x00000,
                        accelerator_dram_address=dst, element_size=C)


def eltwise_add_dram(ue: UnifiedEngine, A_ADDR: int, B_ADDR: int,
                     OUT_ADDR: int, num_elements: int) -> None:
    """Element-wise add of two DRAM buffers: OUT = A + B."""
    bpe = 2
    for offset in range(0, num_elements, URAM_NEAR_FULL_ELEMENTS):
        take = min(URAM_NEAR_FULL_ELEMENTS, num_elements - offset)
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=A_ADDR + offset * bpe,
            sram_address=0x00000, element_size=take)
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=B_ADDR + offset * bpe,
            sram_address=0x80000, element_size=take)
        ue.eltwise_add_core(
            vector_A_sram_start_addr=0x00000,
            vector_B_sram_start_addr=0x80000,
            vector_C_sram_wb_addr=0x00000,
            element_size=take)
        ue.sram_to_accelerator_memory(
            sram_address=0x00000,
            accelerator_dram_address=OUT_ADDR + offset * bpe,
            element_size=take)


def broadcast_mul_dram(ue: UnifiedEngine, ADDR: int,
                       scalar: float, num_elements: int) -> None:
    """In-place scalar multiply on DRAM buffer."""
    bpe = 2
    for offset in range(0, num_elements, URAM_NEAR_FULL_ELEMENTS):
        take = min(URAM_NEAR_FULL_ELEMENTS, num_elements - offset)
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=ADDR + offset * bpe,
            sram_address=0x00000, element_size=take)
        ue.broadcast_mul(
            scalar=scalar,
            sram_start_addr=0x00000, sram_wb_addr=0x00000,
            element_size=take)
        ue.sram_to_accelerator_memory(
            sram_address=0x00000,
            accelerator_dram_address=ADDR + offset * bpe,
            element_size=take)


def dram_copy(ue: UnifiedEngine, SRC: int, DST: int, num_elements: int) -> None:
    """Copy num_elements bf16 values from SRC to DST in DRAM."""
    bpe = 2
    for offset in range(0, num_elements, URAM_NEAR_FULL_ELEMENTS):
        take = min(URAM_NEAR_FULL_ELEMENTS, num_elements - offset)
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=SRC + offset * bpe,
            sram_address=0x00000, element_size=take)
        ue.sram_to_accelerator_memory(
            sram_address=0x00000,
            accelerator_dram_address=DST + offset * bpe,
            element_size=take)


# =============================================================================
# 2D RoPE precomputation (host-side, stored to DRAM during weight_init)
# =============================================================================

def precompute_rope_2d(end_x: int, end_y: int, head_dim: int,
                       theta: float = 10000.0,
                       scale_pos: float = 1.0) -> tuple:
    """Precompute 2D RoPE cos/sin tables in rotate-half format for rope_core_dram.

    Returns:
        cos_table: [num_positions, head_dim] bf16
        neg_sin:   [num_positions, head_dim] bf16, first half negated

    Layout matches the weight permutation applied to Q/K rows (_make_rope_perm):
      [x_lo(quarter), y_lo(quarter), x_hi(quarter), y_hi(quarter)]
    rope_core_dram rotate-half formula:
      out[0:half] = q[0:half]*cos[0:half] + q[half:]*sin_neg[0:half]
      out[half:]  = q[half:]*cos[half:]   + q[0:half]*sin_neg[half:]
    With neg_sin = cat([-sin_x, -sin_y, sin_x, sin_y]), this gives correct 2D
    complex-pair rotation for both x and y axes independently.
    """
    half = head_dim // 2
    quarter = half // 2  # unique frequencies per axis

    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 4)[:quarter].float() / head_dim))

    num_pos = end_x * end_y
    t = torch.arange(num_pos, dtype=torch.float32)
    t_x = (t % end_x).float() * scale_pos
    t_y = torch.div(t, end_x, rounding_mode="floor").float() * scale_pos

    angles_x = torch.outer(t_x, freqs)  # [num_pos, quarter]
    angles_y = torch.outer(t_y, freqs)  # [num_pos, quarter]

    cos_x = torch.cos(angles_x)  # [num_pos, quarter]
    sin_x = torch.sin(angles_x)
    cos_y = torch.cos(angles_y)
    sin_y = torch.sin(angles_y)

    # Rotate-half format: same cos in lo and hi halves; neg_sin has lo half negated.
    # cos = [cos_x, cos_y, cos_x, cos_y]   (each quarter elements)
    # neg_sin = [-sin_x, -sin_y, sin_x, sin_y]
    cos_table = torch.cat([cos_x, cos_y, cos_x, cos_y], dim=1).to(torch.bfloat16)
    neg_sin = torch.cat([-sin_x, -sin_y, sin_x, sin_y], dim=1).to(torch.bfloat16)

    return cos_table, neg_sin


def _make_rope_perm(num_heads: int, head_dim: int) -> torch.Tensor:
    """Row permutation indices to convert Q/K weights from complex-pair to rotate-half layout.

    Input layout per head (complex-pair 2D):
      [x0_r, x0_i, x1_r, x1_i, ..., x15_r, x15_i,  y0_r, y0_i, ..., y15_r, y15_i]
    Output layout per head (rotate-half [x_lo, y_lo, x_hi, y_hi]):
      [x0_r, x1_r, ..., x15_r,  y0_r, ..., y15_r,  x0_i, ..., x15_i,  y0_i, ..., y15_i]

    Apply to Q and K weight rows before storing to DRAM so that rope_core_dram
    (rotate-half) correctly implements the original complex-pair 2D RoPE.
    """
    half = head_dim // 2    # 32
    quarter = half // 2     # 16
    perm = []
    for h in range(num_heads):
        base = h * head_dim
        for j in range(quarter): perm.append(base + 2 * j)           # x even → x_lo
        for j in range(quarter): perm.append(base + half + 2 * j)    # y even → y_lo
        for j in range(quarter): perm.append(base + 2 * j + 1)       # x odd  → x_hi
        for j in range(quarter): perm.append(base + half + 2 * j + 1) # y odd  → y_hi
    return torch.tensor(perm, dtype=torch.long)


def build_padded_qk_weight(W_rows: torch.Tensor, bias: torch.Tensor,
                           n_heads: int, head_dim: int):
    """Pad Q/K weight rows so each head outputs [lo, zeros, hi, zeros] per position.

    After projection with this padded weight, each head's output per position is:
      [lo(half), zeros(half), hi(half), zeros(half)]  = head_dim*2 elements

    This makes lo and hi each occupy a full URAM row (UE_VECTOR_SIZE=64 elements =
    128 bytes), enabling vectorized RoPE with stride_bytes_per_chunk=128 (full row).

    Args:
        W_rows: Q or K weight rows  (n_heads*head_dim, D_model), after _make_rope_perm
        bias:   Q or K bias         (n_heads*head_dim,)
        n_heads:   number of attention heads
        head_dim:  original head dimension (64)

    Returns:
        W_padded: (n_heads * head_dim*2, D_model)
        b_padded: (n_heads * head_dim*2,)
    """
    half   = head_dim // 2    # 32
    hd_pad = head_dim * 2     # 128
    D      = W_rows.shape[1]

    W_padded = torch.zeros(n_heads * hd_pad, D, dtype=W_rows.dtype)
    b_padded = torch.zeros(n_heads * hd_pad,    dtype=bias.dtype)

    for h in range(n_heads):
        src_lo = slice(h * head_dim,        h * head_dim + half)   # original lo rows
        src_hi = slice(h * head_dim + half, (h + 1) * head_dim)    # original hi rows
        dst_lo = slice(h * hd_pad,          h * hd_pad + half)     # [h*128 : h*128+32]
        dst_hi = slice(h * hd_pad + half*2, h * hd_pad + half*3)   # [h*128+64 : h*128+96]

        W_padded[dst_lo] = W_rows[src_lo]
        W_padded[dst_hi] = W_rows[src_hi]
        b_padded[dst_lo] = bias[src_lo]
        b_padded[dst_hi] = bias[src_hi]

    return W_padded, b_padded


def build_padded_proj_weight(W: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """Build padded output-projection weight to accept padded V-head layout as input.

    The attention output after merging heads has shape (seq, n_heads * head_dim_pad)
    where each head's contribution is [lo(half), zeros(half), hi(half), zeros(half)].
    This function pads the proj weight columns to match, inserting zero columns at
    the padding positions so the matrix multiply ignores them.

    Args:
        W:         proj.weight  (out_dim, n_heads*head_dim)  e.g. (1024, 1024)
        n_heads:   number of attention heads
        head_dim:  original head dimension (64)

    Returns:
        W_padded:  (out_dim, n_heads * head_dim*2)  e.g. (1024, 2048)
    """
    half   = head_dim // 2
    hd_pad = head_dim * 2
    out_dim = W.shape[0]

    W_padded = torch.zeros(out_dim, n_heads * hd_pad, dtype=W.dtype)
    for h in range(n_heads):
        src_lo = slice(h * head_dim,        h * head_dim + half)
        src_hi = slice(h * head_dim + half, (h + 1) * head_dim)
        dst_lo = slice(h * hd_pad,          h * hd_pad + half)
        dst_hi = slice(h * hd_pad + half*2, h * hd_pad + half*3)
        W_padded[:, dst_lo] = W[:, src_lo]
        W_padded[:, dst_hi] = W[:, src_hi]
    return W_padded


def _test_build_padded_qk_weight():
    """Verify build_padded_qk_weight math — run with python sam3.1_test.py --test-pad-weight"""
    import torch
    n_heads, head_dim, D_model = 4, 64, 128
    half = head_dim // 2

    W = torch.randn(n_heads * head_dim, D_model, dtype=torch.bfloat16)
    b = torch.randn(n_heads * head_dim, dtype=torch.bfloat16)
    x = torch.randn(16, D_model, dtype=torch.bfloat16)  # seq_len=16

    W_pad, b_pad = build_padded_qk_weight(W, b, n_heads, head_dim)
    assert W_pad.shape == (n_heads * head_dim * 2, D_model), f"bad shape {W_pad.shape}"

    # Original projection output
    out_orig = x @ W.T + b   # (16, n_heads*64)

    # Padded projection output
    out_padded = x @ W_pad.T + b_pad  # (16, n_heads*128)

    for h in range(n_heads):
        lo_orig   = out_orig[:, h*head_dim          : h*head_dim + half]   # (16, 32)
        hi_orig   = out_orig[:, h*head_dim + half   : (h+1)*head_dim]      # (16, 32)

        lo_padded = out_padded[:, h*128             : h*128 + half]        # (16, 32)
        zeros_lo  = out_padded[:, h*128 + half      : h*128 + half*2]      # (16, 32)
        hi_padded = out_padded[:, h*128 + half*2    : h*128 + half*3]      # (16, 32)
        zeros_hi  = out_padded[:, h*128 + half*3    : (h+1)*128]           # (16, 32)

        assert torch.allclose(lo_orig, lo_padded, atol=1e-2), f"head {h} lo mismatch"
        assert torch.allclose(hi_orig, hi_padded, atol=1e-2), f"head {h} hi mismatch"
        assert zeros_lo.abs().max() == 0, f"head {h} lo padding not zero"
        assert zeros_hi.abs().max() == 0, f"head {h} hi padding not zero"

    print("PASS: build_padded_qk_weight — lo/hi correct, padding is zero for all heads")


# =============================================================================
# Config / checkpoint helpers
# =============================================================================

def _load_config(script_dir: str = SCRIPT_DIR) -> dict:
    cp = os.path.join(script_dir, "sam3.1_config.json")
    with open(cp) as f:
        return json.load(f)


def _ensure_checkpoint(script_dir: str, cfg: dict) -> str:
    """Download SAM 3.1 checkpoint from HuggingFace. Returns path to .pt file."""
    paths = cfg["paths"]
    bin_dir = os.path.join(script_dir, "sam3.1_bin")
    os.makedirs(bin_dir, exist_ok=True)

    ckpt_path = os.path.join(bin_dir, paths["hf_checkpoint_filename"])
    if not os.path.exists(ckpt_path):
        _original_print(f"Downloading SAM 3.1 checkpoint from {paths['hf_repo_id']}...")
        hf_hub_download(repo_id=paths["hf_repo_id"],
                        filename=paths["hf_checkpoint_filename"],
                        local_dir=bin_dir)
    return ckpt_path


def _load_detector_state_dict(ckpt_path: str) -> dict:
    """Load checkpoint, extract detector.* keys, strip prefix."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    sd = {k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k}
    _original_print(f"  Loaded {len(sd)} detector weight tensors")
    return sd


def _ensure_bpe_vocab(script_dir: str, cfg: dict) -> str:
    """Download BPE vocab file if missing."""
    paths = cfg["paths"]
    bpe_path = os.path.join(script_dir, paths["bpe_vocab"])
    if not os.path.exists(bpe_path):
        _original_print("Downloading BPE vocabulary...")
        # BPE file is part of the sam3 package assets
        hf_hub_download(repo_id=paths["hf_repo_id"],
                        filename="bpe_simple_vocab_16e6.txt.gz",
                        local_dir=os.path.dirname(bpe_path))
    return bpe_path


# =============================================================================
# DRAM partition: 2 GB params / 600 MB tensors / ~1.4 GB programs
# SAM3 detector weights ≈ 1.64 GB in BF16, needs ≥ 2 GB params space.
# Tensors reduced to 600 MB (ViT heads properly sized + local scratch only).
# Programs expanded to ~1.4 GB to fit all 32 ViT blocks (~1,353 MB).
# =============================================================================

# =============================================================================
# Section 1 DRAM layout — ViT image encoder (32 blocks)
#   Params:      0x00000000 – 0x7FFFFFFF  (2 GB)   weights + RoPE tables
#   Tensors:     0x80000000 – 0x9BFFFFFF  (448 MB) activations / scratch
#   Instructions:0x9C000000 – 0xFFFFFFFF  (1.6 GB) compiled instruction stream
#
# At end of Section 1: ViT output (GA × VIT_DIM) is DMA'd to host.
# DRAM is then reconfigured for Section 2 (FPN + text + decoder).
# =============================================================================
SAM31_S1_PARAMS_BASE  = 0x00000000   # 0.00 GB — ViT weights + RoPE tables
SAM31_S1_TENSOR_BASE  = 0x80000000   # 2.00 GB — ViT activations / scratch
SAM31_S1_PROGRAM_BASE = 0x9C000000   # 2.44 GB — ViT instruction stream

# Aliases for current use (remove when Section 2 is added)
SAM31_PARAMS_BASE  = SAM31_S1_PARAMS_BASE
SAM31_TENSOR_BASE  = SAM31_S1_TENSOR_BASE
SAM31_PROGRAM_BASE = SAM31_S1_PROGRAM_BASE


# =============================================================================
# Sam31_UnifiedEngine
# =============================================================================

class Sam31_UnifiedEngine(UnifiedEngine):
    """SAM 3.1 text-prompted image segmentation on Unified Engine FPGA accelerator.

    Architecture (image-only, single frame):
        ViT-Det backbone (32 blocks, dim=1024, 16 heads, mixed local/global attention)
        → FPN neck (3 scales at dim=256: 288², 144², 72²)
        + Text encoder (24 layers, dim=1024, 16 heads, causal)
        → Resizer (1024 → 256)
        + Geometry encoder (CLS token + 3 cross-attn layers)
        → Encoder fusion (6 layers, dim=256, 8 heads, self-attn + cross-attn)
        → Decoder (6 layers, 200 queries, dim=256, iterative box refinement)
        → Scoring (dot product with text) + Segmentation head (pixel decoder + mask predictor)
    """

    # --- Architecture constants ---
    IMAGE_SIZE      = 1008
    PATCH_SIZE      = 14
    NUM_CHANNELS    = 3
    GRID_SIZE       = 72           # 1008 // 14
    GRID_AREA       = 5184         # 72 * 72

    # ViT backbone
    VIT_DIM         = 1024
    VIT_DEPTH       = 32
    VIT_HEADS       = 16
    VIT_HEAD_DIM    = 64           # 1024 // 16
    VIT_HEAD_DIM_PAD = 128         # padded to 2×head_dim for vectorized RoPE (lo/hi in separate URAM rows)
    VIT_QK_DIM_PAD  = 2048         # VIT_HEADS * VIT_HEAD_DIM_PAD
    VIT_MLP_HIDDEN  = 4736         # int(1024 * 4.625)
    VIT_WINDOW_SIZE = 24
    VIT_WINDOW_AREA = 576          # 24 * 24
    VIT_NUM_WINDOWS = 9            # (72 // 24) ** 2
    VIT_GLOBAL_BLOCKS = {7, 15, 23, 31}
    VIT_PRETRAIN_GRID = 24         # 336 // 14

    # Text encoder
    TEXT_WIDTH      = 1024
    TEXT_LAYERS     = 24
    TEXT_HEADS      = 16
    TEXT_HEAD_DIM   = 64
    TEXT_MLP_HIDDEN = 4096
    TEXT_CTX_LEN    = 32
    TEXT_VOCAB_SIZE = 49408

    # Encoder / decoder (shared d_model)
    D_MODEL         = 256
    ENC_LAYERS      = 6
    ENC_HEADS       = 8
    ENC_HEAD_DIM    = 32           # 256 // 8
    ENC_HEAD_DIM_PAD = 64          # padded to 64 for hardware alignment
    ENC_FFN_DIM     = 2048

    DEC_LAYERS      = 6
    DEC_HEADS       = 8
    DEC_HEAD_DIM    = 32
    DEC_HEAD_DIM_PAD = 64
    DEC_FFN_DIM     = 2048
    NUM_QUERIES     = 200

    # FPN
    FPN_SPATIAL     = [(288, 288), (144, 144), (72, 72)]
    FPN_DIM         = 256

    # Geometry encoder
    GEO_LAYERS      = 3

    # Prompt length = text (32) + geometry CLS (1)
    PROMPT_LEN      = 33

    PATCH_K_RAW = PATCH_SIZE * PATCH_SIZE * NUM_CHANNELS  # 588
    PATCH_K_PAD = ((PATCH_K_RAW + 63) // 64) * 64        # 640

    ALIGN = 64  # UE_VECTOR_SIZE

    def __init__(self, script_dir: str = SCRIPT_DIR, weights_bin: str = None):
        super().__init__(params_dram_base=SAM31_PARAMS_BASE,
                         tensor_dram_base=SAM31_TENSOR_BASE,
                         program_dram_base=SAM31_PROGRAM_BASE)
        self.cfg = _load_config(script_dir)
        self.script_dir = script_dir

        self.weight_init()
        self.tensor_init()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def pad_dim(x: int) -> int:
        return ((x + 63) // 64) * 64

    DMA_CHUNK_BYTES = 1 * 1024 * 1024

    def write_captured_instructions_to_dram(self, start_addr: int = DRAM_INSTRUCTION_ADDR) -> int:
        """Chunked override to avoid segfault on large instruction DMA writes."""
        if not self.capture_buffer or self.capture_count == 0:
            return 0
        total_bytes = self.capture_count * 32
        all_bytes = bytearray()
        for inst in self.capture_buffer:
            all_bytes.extend(inst.get_bytes())
        data = bytes(all_bytes)
        if not _SILENT_MODE:
            _original_print(f"Writing {self.capture_count:,} instructions "
                          f"({total_bytes / 1024**2:.1f} MB) to DRAM at 0x{start_addr:x}...")
        offset = 0
        while offset < total_bytes:
            chunk = min(self.DMA_CHUNK_BYTES, total_bytes - offset)
            self.dma_write(DMA_DEVICE_H2C, start_addr + offset, data[offset:offset + chunk], chunk)
            offset += chunk
        return total_bytes

    def _alloc_param(self, tensor: torch.Tensor) -> int:
        """Allocate params DRAM, write bf16 tensor. Returns DRAM address."""
        t = tensor.to(torch.bfloat16).contiguous()
        addr = self.get_params_dram_addr()
        nbytes = t.numel() * 2
        self.allocate_params_dram(nbytes)
        self.dma_to_accelerator_memory(addr, t)
        return addr

    def _alloc_tensor(self, num_elements: int) -> int:
        """Allocate tensor DRAM buffer. Returns DRAM address."""
        return self.allocate_tensor_dram(num_elements * 2)

    @staticmethod
    def build_unpad_weight(num_heads: int, head_dim: int = 32,
                           head_dim_pad: int = 64) -> torch.Tensor:
        """Build identity-select matrix for head unpadding (same as swin)."""
        K = num_heads * head_dim_pad
        N = num_heads * head_dim
        W = torch.zeros(K, N, dtype=torch.bfloat16)
        for h in range(num_heads):
            for d in range(head_dim):
                W[h * head_dim_pad + d, h * head_dim + d] = 1.0
        return W

    @staticmethod
    def chunk_ranges(total, chunk):
        """Yield (start_index, count) tuples for chunked iteration."""
        for i in range(0, total, chunk):
            yield i, min(chunk, total - i)

    # ------------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------------

    def weight_init(self) -> None:
        """Load all SAM 3.1 detector weights from HF checkpoint to DRAM."""
        ckpt_path = _ensure_checkpoint(self.script_dir, self.cfg)
        sd = _load_detector_state_dict(ckpt_path)
        bpe = 2
        pad = self.pad_dim

        # ==================================================================
        # ViT BACKBONE
        # ==================================================================
        prefix = "backbone.vision_backbone.trunk."

        # Patch embedding: Conv2d(3, 1024, 14, stride=14)
        # Original weight: (1024, 3, 14, 14) → reshape to (1024, 3*14*14) = (1024, 588)
        # For patching_core, we build sparse weights similar to swin (see swin_test.py)
        patch_w = sd[prefix + "patch_embed.proj.weight"].to(torch.bfloat16)
        P = self.PATCH_SIZE  # 14
        C = self.NUM_CHANNELS  # 3
        N = self.VIT_DIM  # 1024
        patch_w_flat = patch_w.reshape(N, -1)  # (1024, 588)
        # K=588 is not 64-aligned. Pad to 640 for matmat_mul_core alignment.
        patch_w_padded = torch.zeros(N, self.PATCH_K_PAD, dtype=torch.bfloat16)
        patch_w_padded[:, :self.PATCH_K_RAW] = patch_w_flat.to(torch.bfloat16)
        self.PATCH_EMBED_WEIGHT = self._alloc_param(patch_w_padded)  # (1024, 640)
        # No bias in SAM3 patch embed (bias=False in config)

        # Absolute positional embedding: (1, 577, 1024) — pretrained with CLS token
        # Tile from 24x24 to 72x72 on the host, then store
        pos_embed = sd[prefix + "pos_embed"]  # (1, 577, 1024)
        pos_no_cls = pos_embed[:, 1:, :]  # (1, 576, 1024) = (1, 24*24, 1024)
        pos_2d = pos_no_cls.reshape(1, 24, 24, 1024).permute(0, 3, 1, 2)  # (1, 1024, 24, 24)
        # Tile 3x to cover 72x72
        pos_tiled = pos_2d.tile(1, 1, 3, 3)[:, :, :72, :72]  # (1, 1024, 72, 72)
        pos_hwc = pos_tiled.permute(0, 2, 3, 1).reshape(self.GRID_AREA, N)  # (5184, 1024)
        self.POS_EMBED = self._alloc_param(pos_hwc)

        # LN_pre
        self.LN_PRE_GAMMA = self._alloc_param(sd[prefix + "ln_pre.weight"])
        self.LN_PRE_BETA = self._alloc_param(sd[prefix + "ln_pre.bias"])

        # Per-block weights
        self.vit_block_weights = []
        for i in range(self.VIT_DEPTH):
            bp = f"{prefix}blocks.{i}."
            bw = {}
            bw['norm1_gamma'] = self._alloc_param(sd[bp + "norm1.weight"])
            bw['norm1_beta'] = self._alloc_param(sd[bp + "norm1.bias"])
            # Permute Q and K rows to rotate-half layout, then pad lo/hi into separate
            # URAM rows so vectorized RoPE can use chunk=128 (full URAM row).
            # Padded Q/K: (2048, 1024) — [lo(32), zeros(32), hi(32), zeros(32)] per head.
            # Padded V:   (2048, 1024) — same pattern; flash_attention needs uniform head_dim.
            # Padded proj: (1024, 2048) — zero columns at padding positions.
            _perm = _make_rope_perm(self.VIT_HEADS, self.VIT_HEAD_DIM)  # (1024,)
            _VD = self.VIT_DIM
            _raw_w = sd[bp + "attn.qkv.weight"].to(torch.bfloat16)  # (3072, 1024)
            _raw_b = sd[bp + "attn.qkv.bias"].to(torch.bfloat16)    # (3072,)
            _qw, _qb = build_padded_qk_weight(
                _raw_w[0:_VD][_perm], _raw_b[0:_VD][_perm],
                self.VIT_HEADS, self.VIT_HEAD_DIM)                   # (2048, 1024)
            # Scale Q by sqrt(2): flash_attention uses 1/sqrt(head_dim_pad=128) but
            # effective head_dim is 64 (zeros don't contribute to dot product).
            # Pre-scaling Q by sqrt(128/64)=sqrt(2) corrects: sqrt(2)*Q·K/sqrt(128) = Q·K/sqrt(64).
            _qw = (_qw.float() * (2.0 ** 0.5)).to(torch.bfloat16)
            _qb = (_qb.float() * (2.0 ** 0.5)).to(torch.bfloat16)
            _kw, _kb = build_padded_qk_weight(
                _raw_w[_VD:2*_VD][_perm], _raw_b[_VD:2*_VD][_perm],
                self.VIT_HEADS, self.VIT_HEAD_DIM)                   # (2048, 1024)
            _vw, _vb = build_padded_qk_weight(
                _raw_w[2*_VD:3*_VD], _raw_b[2*_VD:3*_VD],
                self.VIT_HEADS, self.VIT_HEAD_DIM)                   # (2048, 1024)
            bw['q_weight']  = self._alloc_param(_qw)
            bw['q_bias']    = self._alloc_param(_qb)
            bw['k_weight']  = self._alloc_param(_kw)
            bw['k_bias']    = self._alloc_param(_kb)
            bw['v_weight']  = self._alloc_param(_vw)
            bw['v_bias']    = self._alloc_param(_vb)
            _proj_w = build_padded_proj_weight(
                sd[bp + "attn.proj.weight"].to(torch.bfloat16),
                self.VIT_HEADS, self.VIT_HEAD_DIM)                   # (1024, 2048)
            bw['proj_weight'] = self._alloc_param(_proj_w)
            bw['proj_bias']   = self._alloc_param(sd[bp + "attn.proj.bias"])
            bw['norm2_gamma'] = self._alloc_param(sd[bp + "norm2.weight"])
            bw['norm2_beta'] = self._alloc_param(sd[bp + "norm2.bias"])
            bw['fc1_weight'] = self._alloc_param(sd[bp + "mlp.fc1.weight"])    # (4736, 1024)
            bw['fc1_bias'] = self._alloc_param(sd[bp + "mlp.fc1.bias"])
            bw['fc2_weight'] = self._alloc_param(sd[bp + "mlp.fc2.weight"])    # (1024, 4736)
            bw['fc2_bias'] = self._alloc_param(sd[bp + "mlp.fc2.bias"])
            self.vit_block_weights.append(bw)

        # 2D RoPE tables — stored as 4 padded halves per table set for URAM alignment.
        # half=32 elements=64 bytes; padded to UE_VECTOR_SIZE=64 elements=128 bytes per row.
        def _pad_rope(cos, neg_sin):
            """Split [seq, N] tables into 4 zero-padded halves [seq, UE_VECTOR_SIZE]."""
            seq, N = cos.shape
            half = N // 2
            pad = UE_VECTOR_SIZE  # 64 elements per URAM row
            def _half_pad(t, start):
                out = torch.zeros(seq, pad, dtype=torch.bfloat16)
                out[:, :half] = t[:, start:start + half]
                return out
            return (_half_pad(cos, 0), _half_pad(cos, half),
                    _half_pad(neg_sin, 0), _half_pad(neg_sin, half))

        # Windowed blocks: 24x24 positions
        cos_win, sin_win = precompute_rope_2d(
            self.VIT_WINDOW_SIZE, self.VIT_WINDOW_SIZE, self.VIT_HEAD_DIM,
            theta=self.cfg["model"]["rope_theta"], scale_pos=1.0)
        cl_w, ch_w, sl_w, sh_w = _pad_rope(cos_win, sin_win)
        self.ROPE_COS_LO_WIN = self._alloc_param(cl_w)   # (576, 64) padded
        self.ROPE_COS_HI_WIN = self._alloc_param(ch_w)
        self.ROPE_SIN_LO_WIN = self._alloc_param(sl_w)
        self.ROPE_SIN_HI_WIN = self._alloc_param(sh_w)


        # Global blocks: 72x72 positions (RoPE skipped at runtime — stored for future use)
        rope_scale_global = self.VIT_WINDOW_SIZE / self.GRID_SIZE  # 24/72 = 0.333
        cos_glob, sin_glob = precompute_rope_2d(
            self.GRID_SIZE, self.GRID_SIZE, self.VIT_HEAD_DIM,
            theta=self.cfg["model"]["rope_theta"], scale_pos=rope_scale_global)
        cl_g, ch_g, sl_g, sh_g = _pad_rope(cos_glob, sin_glob)
        self.ROPE_COS_LO_GLOB = self._alloc_param(cl_g)  # (5184, 64) padded
        self.ROPE_COS_HI_GLOB = self._alloc_param(ch_g)
        self.ROPE_SIN_LO_GLOB = self._alloc_param(sl_g)
        self.ROPE_SIN_HI_GLOB = self._alloc_param(sh_g)

        # ==================================================================
        # FPN NECK (3 scales used after scalp=1 removes 0.5x)
        # ==================================================================
        fpn_prefix = "backbone.vision_backbone.convs."

        # Scale 4.0x: ConvT(1024,512,2,s=2) → GELU → ConvT(512,256,2,s=2) → Conv1x1 → Conv3x3
        self.fpn_4x = {}
        w0 = sd[fpn_prefix + "0.dconv_2x2_0.weight"]  # (1024, 512, 2, 2) ConvTranspose2d
        self.fpn_4x['dconv0_slices'] = []
        for ky in range(2):
            for kx in range(2):
                # ConvTranspose2d weight is (C_in, C_out, kH, kW)
                # Extract slice and transpose to (C_out, C_in) for matmat_mul_core
                s = w0[:, :, ky, kx].T.contiguous()  # (512, 1024)
                self.fpn_4x['dconv0_slices'].append(self._alloc_param(s))
        self.fpn_4x['dconv0_bias'] = self._alloc_param(sd[fpn_prefix + "0.dconv_2x2_0.bias"])

        w1 = sd[fpn_prefix + "0.dconv_2x2_1.weight"]  # (512, 256, 2, 2)
        self.fpn_4x['dconv1_slices'] = []
        for ky in range(2):
            for kx in range(2):
                s = w1[:, :, ky, kx].T.contiguous()  # (256, 512)
                self.fpn_4x['dconv1_slices'].append(self._alloc_param(s))
        self.fpn_4x['dconv1_bias'] = self._alloc_param(sd[fpn_prefix + "0.dconv_2x2_1.bias"])

        # Conv1x1 and Conv3x3 for scale 4x
        self.fpn_4x['conv1x1_w'] = self._alloc_param(
            sd[fpn_prefix + "0.conv_1x1.weight"].reshape(256, 256))
        self.fpn_4x['conv1x1_b'] = self._alloc_param(sd[fpn_prefix + "0.conv_1x1.bias"])
        self.fpn_4x['conv3x3_w'] = self._alloc_param(
            sd[fpn_prefix + "0.conv_3x3.weight"].reshape(256, 256 * 9))
        self.fpn_4x['conv3x3_b'] = self._alloc_param(sd[fpn_prefix + "0.conv_3x3.bias"])

        # Scale 2.0x: ConvT(1024,512,2,s=2) → Conv1x1(512,256) → Conv3x3(256,256)
        self.fpn_2x = {}
        w2 = sd[fpn_prefix + "1.dconv_2x2.weight"]  # (1024, 512, 2, 2)
        self.fpn_2x['dconv_slices'] = []
        for ky in range(2):
            for kx in range(2):
                s = w2[:, :, ky, kx].T.contiguous()
                self.fpn_2x['dconv_slices'].append(self._alloc_param(s))
        self.fpn_2x['dconv_bias'] = self._alloc_param(sd[fpn_prefix + "1.dconv_2x2.bias"])
        self.fpn_2x['conv1x1_w'] = self._alloc_param(
            sd[fpn_prefix + "1.conv_1x1.weight"].reshape(256, 512))
        self.fpn_2x['conv1x1_b'] = self._alloc_param(sd[fpn_prefix + "1.conv_1x1.bias"])
        self.fpn_2x['conv3x3_w'] = self._alloc_param(
            sd[fpn_prefix + "1.conv_3x3.weight"].reshape(256, 256 * 9))
        self.fpn_2x['conv3x3_b'] = self._alloc_param(sd[fpn_prefix + "1.conv_3x3.bias"])

        # Scale 1.0x: Conv1x1(1024,256) → Conv3x3(256,256)
        self.fpn_1x = {}
        self.fpn_1x['conv1x1_w'] = self._alloc_param(
            sd[fpn_prefix + "2.conv_1x1.weight"].reshape(256, 1024))
        self.fpn_1x['conv1x1_b'] = self._alloc_param(sd[fpn_prefix + "2.conv_1x1.bias"])
        self.fpn_1x['conv3x3_w'] = self._alloc_param(
            sd[fpn_prefix + "2.conv_3x3.weight"].reshape(256, 256 * 9))
        self.fpn_1x['conv3x3_b'] = self._alloc_param(sd[fpn_prefix + "2.conv_3x3.bias"])

        # ==================================================================
        # TEXT ENCODER
        # ==================================================================
        txt_prefix = "backbone.language_backbone."

        self.TEXT_EMBED = self._alloc_param(
            sd[txt_prefix + "encoder.token_embedding.weight"])  # (49408, 1024)
        self.TEXT_POS_EMBED = self._alloc_param(
            sd[txt_prefix + "encoder.positional_embedding"])    # (32, 1024)

        self.text_block_weights = []
        for i in range(self.TEXT_LAYERS):
            tp = f"{txt_prefix}encoder.transformer.resblocks.{i}."
            tw = {}
            tw['ln1_w'] = self._alloc_param(sd[tp + "ln_1.weight"])
            tw['ln1_b'] = self._alloc_param(sd[tp + "ln_1.bias"])
            tw['attn_in_proj_w'] = self._alloc_param(sd[tp + "attn.in_proj_weight"])  # (3072, 1024)
            tw['attn_in_proj_b'] = self._alloc_param(sd[tp + "attn.in_proj_bias"])
            tw['attn_out_proj_w'] = self._alloc_param(sd[tp + "attn.out_proj.weight"])
            tw['attn_out_proj_b'] = self._alloc_param(sd[tp + "attn.out_proj.bias"])
            tw['ln2_w'] = self._alloc_param(sd[tp + "ln_2.weight"])
            tw['ln2_b'] = self._alloc_param(sd[tp + "ln_2.bias"])
            tw['mlp_fc_w'] = self._alloc_param(sd[tp + "mlp.c_fc.weight"])      # (4096, 1024)
            tw['mlp_fc_b'] = self._alloc_param(sd[tp + "mlp.c_fc.bias"])
            tw['mlp_proj_w'] = self._alloc_param(sd[tp + "mlp.c_proj.weight"])  # (1024, 4096)
            tw['mlp_proj_b'] = self._alloc_param(sd[tp + "mlp.c_proj.bias"])
            self.text_block_weights.append(tw)

        self.TEXT_LN_FINAL_W = self._alloc_param(sd[txt_prefix + "encoder.ln_final.weight"])
        self.TEXT_LN_FINAL_B = self._alloc_param(sd[txt_prefix + "encoder.ln_final.bias"])
        self.TEXT_RESIZER_W = self._alloc_param(sd[txt_prefix + "resizer.weight"])  # (256, 1024)
        self.TEXT_RESIZER_B = self._alloc_param(sd[txt_prefix + "resizer.bias"])

        # Causal attention mask: upper-triangular -inf, precomputed and stored
        causal_mask = torch.zeros(self.TEXT_CTX_LEN, self.TEXT_CTX_LEN, dtype=torch.bfloat16)
        causal_mask.fill_(float("-inf"))
        causal_mask = causal_mask.triu_(1)  # upper triangle = -inf
        self.CAUSAL_MASK = self._alloc_param(causal_mask)

        # ==================================================================
        # GEOMETRY ENCODER
        # ==================================================================
        geo_prefix = "geometry_encoder."

        self.GEO_CLS_EMBED = self._alloc_param(sd[geo_prefix + "cls_embed.weight"])  # (1, 256)
        self.GEO_NORM_W = self._alloc_param(sd[geo_prefix + "norm.weight"])
        self.GEO_NORM_B = self._alloc_param(sd[geo_prefix + "norm.bias"])
        self.GEO_FINAL_PROJ_W = self._alloc_param(sd[geo_prefix + "final_proj.weight"])
        self.GEO_FINAL_PROJ_B = self._alloc_param(sd[geo_prefix + "final_proj.bias"])
        self.GEO_ENCODE_NORM_W = self._alloc_param(sd[geo_prefix + "encode_norm.weight"])
        self.GEO_ENCODE_NORM_B = self._alloc_param(sd[geo_prefix + "encode_norm.bias"])

        self.geo_layer_weights = []
        for i in range(self.GEO_LAYERS):
            gp = f"{geo_prefix}encode.{i}."
            gw = {}
            gw['self_attn_in_w'] = self._alloc_param(sd[gp + "self_attn.in_proj_weight"])
            gw['self_attn_in_b'] = self._alloc_param(sd[gp + "self_attn.in_proj_bias"])
            gw['self_attn_out_w'] = self._alloc_param(sd[gp + "self_attn.out_proj.weight"])
            gw['self_attn_out_b'] = self._alloc_param(sd[gp + "self_attn.out_proj.bias"])
            gw['cross_attn_in_w'] = self._alloc_param(sd[gp + "cross_attn_image.in_proj_weight"])
            gw['cross_attn_in_b'] = self._alloc_param(sd[gp + "cross_attn_image.in_proj_bias"])
            gw['cross_attn_out_w'] = self._alloc_param(sd[gp + "cross_attn_image.out_proj.weight"])
            gw['cross_attn_out_b'] = self._alloc_param(sd[gp + "cross_attn_image.out_proj.bias"])
            gw['norm1_w'] = self._alloc_param(sd[gp + "norm1.weight"])
            gw['norm1_b'] = self._alloc_param(sd[gp + "norm1.bias"])
            gw['norm2_w'] = self._alloc_param(sd[gp + "norm2.weight"])
            gw['norm2_b'] = self._alloc_param(sd[gp + "norm2.bias"])
            gw['norm3_w'] = self._alloc_param(sd[gp + "norm3.weight"])
            gw['norm3_b'] = self._alloc_param(sd[gp + "norm3.bias"])
            gw['ffn_l1_w'] = self._alloc_param(sd[gp + "linear1.weight"])
            gw['ffn_l1_b'] = self._alloc_param(sd[gp + "linear1.bias"])
            gw['ffn_l2_w'] = self._alloc_param(sd[gp + "linear2.weight"])
            gw['ffn_l2_b'] = self._alloc_param(sd[gp + "linear2.bias"])
            self.geo_layer_weights.append(gw)

        # ==================================================================
        # ENCODER FUSION (6 layers)
        # ==================================================================
        enc_prefix = "transformer.encoder."

        self.enc_layer_weights = []
        for i in range(self.ENC_LAYERS):
            ep = f"{enc_prefix}layers.{i}."
            ew = {}
            ew['norm1_w'] = self._alloc_param(sd[ep + "norm1.weight"])
            ew['norm1_b'] = self._alloc_param(sd[ep + "norm1.bias"])
            ew['self_attn_in_w'] = self._alloc_param(sd[ep + "self_attn.in_proj_weight"])
            ew['self_attn_in_b'] = self._alloc_param(sd[ep + "self_attn.in_proj_bias"])
            ew['self_attn_out_w'] = self._alloc_param(sd[ep + "self_attn.out_proj.weight"])
            ew['self_attn_out_b'] = self._alloc_param(sd[ep + "self_attn.out_proj.bias"])
            ew['norm2_w'] = self._alloc_param(sd[ep + "norm2.weight"])
            ew['norm2_b'] = self._alloc_param(sd[ep + "norm2.bias"])
            ew['cross_attn_in_w'] = self._alloc_param(sd[ep + "cross_attn_image.in_proj_weight"])
            ew['cross_attn_in_b'] = self._alloc_param(sd[ep + "cross_attn_image.in_proj_bias"])
            ew['cross_attn_out_w'] = self._alloc_param(sd[ep + "cross_attn_image.out_proj.weight"])
            ew['cross_attn_out_b'] = self._alloc_param(sd[ep + "cross_attn_image.out_proj.bias"])
            ew['norm3_w'] = self._alloc_param(sd[ep + "norm3.weight"])
            ew['norm3_b'] = self._alloc_param(sd[ep + "norm3.bias"])
            ew['ffn_l1_w'] = self._alloc_param(sd[ep + "linear1.weight"])
            ew['ffn_l1_b'] = self._alloc_param(sd[ep + "linear1.bias"])
            ew['ffn_l2_w'] = self._alloc_param(sd[ep + "linear2.weight"])
            ew['ffn_l2_b'] = self._alloc_param(sd[ep + "linear2.bias"])
            self.enc_layer_weights.append(ew)

        # Unpad weight for encoder/decoder (head_dim 32 → padded 64)
        unpad_w = self.build_unpad_weight(self.ENC_HEADS, self.ENC_HEAD_DIM, self.ENC_HEAD_DIM_PAD)
        self.ENC_DEC_UNPAD_WEIGHT = self._alloc_param(unpad_w.T.contiguous())

        # ==================================================================
        # DECODER (6 layers + global weights)
        # ==================================================================
        dec_prefix = "transformer.decoder."

        self.DEC_QUERY_EMBED = self._alloc_param(sd[dec_prefix + "query_embed.weight"])    # (200, 256)
        self.DEC_REF_POINTS = self._alloc_param(sd[dec_prefix + "reference_points.weight"])  # (200, 4)
        self.DEC_NORM_W = self._alloc_param(sd[dec_prefix + "norm.weight"])
        self.DEC_NORM_B = self._alloc_param(sd[dec_prefix + "norm.bias"])

        # Box refinement MLP: 256→256→256→4
        self.BBOX_L0_W = self._alloc_param(sd[dec_prefix + "bbox_embed.layers.0.weight"])
        self.BBOX_L0_B = self._alloc_param(sd[dec_prefix + "bbox_embed.layers.0.bias"])
        self.BBOX_L1_W = self._alloc_param(sd[dec_prefix + "bbox_embed.layers.1.weight"])
        self.BBOX_L1_B = self._alloc_param(sd[dec_prefix + "bbox_embed.layers.1.bias"])
        self.BBOX_L2_W = self._alloc_param(sd[dec_prefix + "bbox_embed.layers.2.weight"])
        self.BBOX_L2_B = self._alloc_param(sd[dec_prefix + "bbox_embed.layers.2.bias"])

        # Conditional query position: ref_point_head MLP 512→256→256
        self.REF_HEAD_L0_W = self._alloc_param(sd[dec_prefix + "ref_point_head.layers.0.weight"])
        self.REF_HEAD_L0_B = self._alloc_param(sd[dec_prefix + "ref_point_head.layers.0.bias"])
        self.REF_HEAD_L1_W = self._alloc_param(sd[dec_prefix + "ref_point_head.layers.1.weight"])
        self.REF_HEAD_L1_B = self._alloc_param(sd[dec_prefix + "ref_point_head.layers.1.bias"])

        # Presence token
        self.DEC_PRESENCE_TOKEN = self._alloc_param(sd[dec_prefix + "presence_token.weight"])
        self.PRES_HEAD_L0_W = self._alloc_param(sd[dec_prefix + "presence_token_head.layers.0.weight"])
        self.PRES_HEAD_L0_B = self._alloc_param(sd[dec_prefix + "presence_token_head.layers.0.bias"])
        self.PRES_HEAD_L1_W = self._alloc_param(sd[dec_prefix + "presence_token_head.layers.1.weight"])
        self.PRES_HEAD_L1_B = self._alloc_param(sd[dec_prefix + "presence_token_head.layers.1.bias"])
        self.PRES_HEAD_L2_W = self._alloc_param(sd[dec_prefix + "presence_token_head.layers.2.weight"])
        self.PRES_HEAD_L2_B = self._alloc_param(sd[dec_prefix + "presence_token_head.layers.2.bias"])
        self.PRES_OUT_NORM_W = self._alloc_param(sd[dec_prefix + "presence_token_out_norm.weight"])
        self.PRES_OUT_NORM_B = self._alloc_param(sd[dec_prefix + "presence_token_out_norm.bias"])

        # Box RPB MLPs
        self.RPB_X_L0_W = self._alloc_param(sd[dec_prefix + "boxRPB_embed_x.layers.0.weight"])
        self.RPB_X_L0_B = self._alloc_param(sd[dec_prefix + "boxRPB_embed_x.layers.0.bias"])
        self.RPB_X_L1_W = self._alloc_param(sd[dec_prefix + "boxRPB_embed_x.layers.1.weight"])
        self.RPB_X_L1_B = self._alloc_param(sd[dec_prefix + "boxRPB_embed_x.layers.1.bias"])
        self.RPB_Y_L0_W = self._alloc_param(sd[dec_prefix + "boxRPB_embed_y.layers.0.weight"])
        self.RPB_Y_L0_B = self._alloc_param(sd[dec_prefix + "boxRPB_embed_y.layers.0.bias"])
        self.RPB_Y_L1_W = self._alloc_param(sd[dec_prefix + "boxRPB_embed_y.layers.1.weight"])
        self.RPB_Y_L1_B = self._alloc_param(sd[dec_prefix + "boxRPB_embed_y.layers.1.bias"])

        # Per-layer decoder weights
        self.dec_layer_weights = []
        for i in range(self.DEC_LAYERS):
            dp = f"{dec_prefix}layers.{i}."
            dw = {}
            dw['self_attn_in_w'] = self._alloc_param(sd[dp + "self_attn.in_proj_weight"])
            dw['self_attn_in_b'] = self._alloc_param(sd[dp + "self_attn.in_proj_bias"])
            dw['self_attn_out_w'] = self._alloc_param(sd[dp + "self_attn.out_proj.weight"])
            dw['self_attn_out_b'] = self._alloc_param(sd[dp + "self_attn.out_proj.bias"])
            dw['norm2_w'] = self._alloc_param(sd[dp + "norm2.weight"])
            dw['norm2_b'] = self._alloc_param(sd[dp + "norm2.bias"])
            dw['ca_text_in_w'] = self._alloc_param(sd[dp + "ca_text.in_proj_weight"])
            dw['ca_text_in_b'] = self._alloc_param(sd[dp + "ca_text.in_proj_bias"])
            dw['ca_text_out_w'] = self._alloc_param(sd[dp + "ca_text.out_proj.weight"])
            dw['ca_text_out_b'] = self._alloc_param(sd[dp + "ca_text.out_proj.bias"])
            dw['catext_norm_w'] = self._alloc_param(sd[dp + "catext_norm.weight"])
            dw['catext_norm_b'] = self._alloc_param(sd[dp + "catext_norm.bias"])
            dw['cross_attn_in_w'] = self._alloc_param(sd[dp + "cross_attn.in_proj_weight"])
            dw['cross_attn_in_b'] = self._alloc_param(sd[dp + "cross_attn.in_proj_bias"])
            dw['cross_attn_out_w'] = self._alloc_param(sd[dp + "cross_attn.out_proj.weight"])
            dw['cross_attn_out_b'] = self._alloc_param(sd[dp + "cross_attn.out_proj.bias"])
            dw['norm1_w'] = self._alloc_param(sd[dp + "norm1.weight"])
            dw['norm1_b'] = self._alloc_param(sd[dp + "norm1.bias"])
            dw['ffn_l1_w'] = self._alloc_param(sd[dp + "linear1.weight"])
            dw['ffn_l1_b'] = self._alloc_param(sd[dp + "linear1.bias"])
            dw['ffn_l2_w'] = self._alloc_param(sd[dp + "linear2.weight"])
            dw['ffn_l2_b'] = self._alloc_param(sd[dp + "linear2.bias"])
            dw['norm3_w'] = self._alloc_param(sd[dp + "norm3.weight"])
            dw['norm3_b'] = self._alloc_param(sd[dp + "norm3.bias"])
            self.dec_layer_weights.append(dw)

        # ==================================================================
        # DOT PRODUCT SCORING
        # ==================================================================
        score_prefix = "dot_prod_scoring."

        self.SCORE_PROMPT_MLP_L0_W = self._alloc_param(sd[score_prefix + "prompt_mlp.layers.0.weight"])
        self.SCORE_PROMPT_MLP_L0_B = self._alloc_param(sd[score_prefix + "prompt_mlp.layers.0.bias"])
        self.SCORE_PROMPT_MLP_L1_W = self._alloc_param(sd[score_prefix + "prompt_mlp.layers.1.weight"])
        self.SCORE_PROMPT_MLP_L1_B = self._alloc_param(sd[score_prefix + "prompt_mlp.layers.1.bias"])
        self.SCORE_PROMPT_MLP_NORM_W = self._alloc_param(sd[score_prefix + "prompt_mlp.out_norm.weight"])
        self.SCORE_PROMPT_MLP_NORM_B = self._alloc_param(sd[score_prefix + "prompt_mlp.out_norm.bias"])
        self.SCORE_PROMPT_PROJ_W = self._alloc_param(sd[score_prefix + "prompt_proj.weight"])
        self.SCORE_PROMPT_PROJ_B = self._alloc_param(sd[score_prefix + "prompt_proj.bias"])
        self.SCORE_HS_PROJ_W = self._alloc_param(sd[score_prefix + "hs_proj.weight"])
        self.SCORE_HS_PROJ_B = self._alloc_param(sd[score_prefix + "hs_proj.bias"])

        # ==================================================================
        # SEGMENTATION HEAD
        # ==================================================================
        seg_prefix = "segmentation_head."

        # Cross-attend prompt
        self.SEG_CROSS_NORM_W = self._alloc_param(sd[seg_prefix + "cross_attn_norm.weight"])
        self.SEG_CROSS_NORM_B = self._alloc_param(sd[seg_prefix + "cross_attn_norm.bias"])
        self.SEG_CROSS_IN_W = self._alloc_param(sd[seg_prefix + "cross_attend_prompt.in_proj_weight"])
        self.SEG_CROSS_IN_B = self._alloc_param(sd[seg_prefix + "cross_attend_prompt.in_proj_bias"])
        self.SEG_CROSS_OUT_W = self._alloc_param(sd[seg_prefix + "cross_attend_prompt.out_proj.weight"])
        self.SEG_CROSS_OUT_B = self._alloc_param(sd[seg_prefix + "cross_attend_prompt.out_proj.bias"])

        # Pixel decoder: 3 stages of Conv3x3 + GroupNorm
        self.pixel_dec_weights = []
        for i in range(3):
            pw = {}
            pw['conv_w'] = self._alloc_param(
                sd[seg_prefix + f"pixel_decoder.conv_layers.{i}.weight"].reshape(256, 256 * 9))
            pw['conv_b'] = self._alloc_param(
                sd[seg_prefix + f"pixel_decoder.conv_layers.{i}.bias"])
            pw['gn_w'] = self._alloc_param(sd[seg_prefix + f"pixel_decoder.norms.{i}.weight"])
            pw['gn_b'] = self._alloc_param(sd[seg_prefix + f"pixel_decoder.norms.{i}.bias"])
            self.pixel_dec_weights.append(pw)

        # Instance and semantic heads
        self.SEG_INST_W = self._alloc_param(
            sd[seg_prefix + "instance_seg_head.weight"].reshape(256, 256))
        self.SEG_INST_B = self._alloc_param(sd[seg_prefix + "instance_seg_head.bias"])

        # Mask predictor MLP: 256→256→256→256
        self.MASK_EMBED_L0_W = self._alloc_param(sd[seg_prefix + "mask_predictor.mask_embed.layers.0.weight"])
        self.MASK_EMBED_L0_B = self._alloc_param(sd[seg_prefix + "mask_predictor.mask_embed.layers.0.bias"])
        self.MASK_EMBED_L1_W = self._alloc_param(sd[seg_prefix + "mask_predictor.mask_embed.layers.1.weight"])
        self.MASK_EMBED_L1_B = self._alloc_param(sd[seg_prefix + "mask_predictor.mask_embed.layers.1.bias"])
        self.MASK_EMBED_L2_W = self._alloc_param(sd[seg_prefix + "mask_predictor.mask_embed.layers.2.weight"])
        self.MASK_EMBED_L2_B = self._alloc_param(sd[seg_prefix + "mask_predictor.mask_embed.layers.2.bias"])

        params_used = self.get_params_dram_usage()
        _original_print(f"  Params loaded: {params_used / 1024**2:.1f} MB ({params_used / 1024**3:.2f} GB)")
        del sd  # free host memory

    # ------------------------------------------------------------------
    # Tensor initialization (intermediate activation buffers)
    # ------------------------------------------------------------------

    def tensor_init(self) -> None:
        """Allocate DRAM buffers for all intermediate activations and scratch."""
        bpe = 2
        pad = self.pad_dim

        # Only allocate what the current checkpoint needs.
        # Add more buffers as we move the checkpoint forward.

        GA = self.GRID_AREA    # 5184
        VD = self.VIT_DIM      # 1024

        self.IMAGE_DRAM      = self._alloc_tensor(self.NUM_CHANNELS * self.IMAGE_SIZE * self.IMAGE_SIZE)
        self.VIT_PATCH_OUT   = self._alloc_tensor(GA * self.PATCH_K_PAD)  # (5184, 640) input patches
        self.VIT_LAYER_IN    = self._alloc_tensor(GA * VD)                # (5184, 1024)

        # --- ViT block 0: norm1 + window_partition + QKV matmul ---
        self.VIT_LN_OUT      = self._alloc_tensor(GA * VD)               # (5184, 1024) after LN
        self.VIT_WINDOWED    = self._alloc_tensor(GA * VD)               # (5184, 1024) after window partition

        # --- ViT block: Q/K/V separate, multi-head attention, MLP ---
        # Q, K, V use padded head_dim=128 so vectorized RoPE can use chunk=128 (full URAM row).
        VD_QK = self.VIT_QK_DIM_PAD  # 2048
        self.VIT_Q           = self._alloc_tensor(GA * VD_QK)            # (5184, 2048) Q projection
        self.VIT_K           = self._alloc_tensor(GA * VD_QK)            # (5184, 2048) K projection
        self.VIT_V           = self._alloc_tensor(GA * VD_QK)            # (5184, 2048) V projection

        # Multi-head layout: (total_batches, seq_len, head_dim_pad=128)
        # Local: (144, 576, 128), Global: (16, 5184, 128)
        hd     = self.VIT_HEAD_DIM      # 64 — original
        hd_pad = self.VIT_HEAD_DIM_PAD  # 128 — padded
        total_heads_local = self.VIT_NUM_WINDOWS * self.VIT_HEADS  # 144
        local_total  = total_heads_local * self.VIT_WINDOW_AREA    # 144 × 576  = 82,944
        global_total = self.VIT_HEADS * GA                         # 16  × 5184 = 82,944
        heads_elems  = max(local_total, global_total) * hd_pad     # 82,944 × 128 = 10,616,832 (~20 MB each)
        self.VIT_Q_HEADS     = self._alloc_tensor(heads_elems)
        self.VIT_K_HEADS     = self._alloc_tensor(heads_elems)
        self.VIT_V_HEADS     = self._alloc_tensor(heads_elems)
        self.VIT_ATTN_OUT    = self._alloc_tensor(heads_elems)
        # Scratch: local blocks dominate. hd_pad=128 used for V^T buffer inside flash_attention.
        local_scratch = total_heads_local * (hd_pad * self.VIT_WINDOW_AREA + self.VIT_WINDOW_AREA ** 2)
        self.VIT_ATTN_SCRATCH = self._alloc_tensor(local_scratch)
        self.VIT_ATTN_MERGED = self._alloc_tensor(GA * VD_QK)           # (5184, 2048) padded
        self.VIT_OUT_PROJ    = self._alloc_tensor(GA * VD)
        self.VIT_RESIDUAL    = self._alloc_tensor(GA * VD)
        self.VIT_MLP_MID     = self._alloc_tensor(GA * self.VIT_MLP_HIDDEN)  # (5184, 4736)
        self.VIT_MLP_OUT     = self._alloc_tensor(GA * VD)

        tensor_used = self.get_tensor_dram_usage()
        _original_print(f"  Tensors allocated: {tensor_used / 1024**2:.1f} MB")

    # ------------------------------------------------------------------
    # Program compilation
    # ------------------------------------------------------------------

    def compile_full_fused(self) -> int:
        """Compile the entire SAM3.1 forward pass as a single instruction stream.

        Returns program DRAM address.
        """
        global _SILENT_MODE
        _SILENT_MODE = True

        pad = self.pad_dim
        bpe = 2

        self.start_capture()

        try:
            self._compile_phases(pad, bpe)
        except _CheckpointStop as e:
            _original_print(f"  Compile stopped at checkpoint: {e}")
        self._finalize_program()
        _SILENT_MODE = False
        return self._last_prog_addr

    def _compile_phases(self, pad, bpe):
        """All compile phases live here. assert False stops at the current checkpoint."""

        # ==============================================================
        # PHASE 1: ViT BACKBONE (32 blocks)
        # ==============================================================

        # 1.1 Patch embedding: im2col 14x14 patches + matmul
        # Each patch is a 14x14x3 = 588-dim vector, 72x72 = 5184 patches
        # matmul: (5184, 588) @ (1024, 588)^T → (5184, 1024)
        # TODO: Implement patch extraction via gather (similar to swin's bf16_patching_core
        #       but for 14x14 patches that don't align to UE_VECTOR_SIZE=64).
        #       For now, assume image patches are pre-gathered to VIT_PATCH_OUT in (5184, 588) layout.
        self.matmat_mul_core(
            M=self.GRID_AREA, K=self.PATCH_K_PAD,  # 640 (588 padded to 64-align)
            N=self.VIT_DIM,
            A_DRAM_ADDR=self.VIT_PATCH_OUT,
            B_DRAM_ADDR=self.PATCH_EMBED_WEIGHT,
            OUTPUT_DRAM_ADDR=self.VIT_LAYER_IN,
        )

        # 1.2 Add absolute positional embedding
        eltwise_add_dram(self, self.VIT_LAYER_IN, self.POS_EMBED,
                         self.VIT_LAYER_IN, self.GRID_AREA * self.VIT_DIM)

        # 1.3 LN_pre
        self.layer_norm_core_dram(
            M=self.GRID_AREA, N=self.VIT_DIM,
            A_DRAM_ADDR=self.VIT_LAYER_IN,
            OUTPUT_DRAM_ADDR=self.VIT_LAYER_IN,
            GAMMA_DRAM_ADDR=self.LN_PRE_GAMMA,
            BETA_DRAM_ADDR=self.LN_PRE_BETA,
        )

        # PASSED: patch_embed + pos_embed + LN_pre

        # 1.4 Transformer blocks 0-31
        _pre_vit_inst = self.capture_count
        for blk_idx in range(self.VIT_DEPTH):
            _blk_start = self.capture_count
            bw = self.vit_block_weights[blk_idx]
            is_global = blk_idx in self.VIT_GLOBAL_BLOCKS

            if is_global:
                seq_len = self.GRID_AREA       # 5184
                num_windows = 1
                total_batches = self.VIT_HEADS  # 16
                rope_cl = self.ROPE_COS_LO_GLOB; rope_ch = self.ROPE_COS_HI_GLOB
                rope_sl = self.ROPE_SIN_LO_GLOB; rope_sh = self.ROPE_SIN_HI_GLOB
            else:
                seq_len = self.VIT_WINDOW_AREA  # 576
                num_windows = self.VIT_NUM_WINDOWS  # 9
                total_batches = num_windows * self.VIT_HEADS  # 144
                rope_cl = self.ROPE_COS_LO_WIN; rope_ch = self.ROPE_COS_HI_WIN
                rope_sl = self.ROPE_SIN_LO_WIN; rope_sh = self.ROPE_SIN_HI_WIN

            M_flat = num_windows * seq_len  # 5184 in both cases

            # --- Pre-attention LayerNorm ---
            self.layer_norm_core_dram(
                M=self.GRID_AREA, N=self.VIT_DIM,
                A_DRAM_ADDR=self.VIT_LAYER_IN,
                OUTPUT_DRAM_ADDR=self.VIT_LN_OUT,
                GAMMA_DRAM_ADDR=bw['norm1_gamma'],
                BETA_DRAM_ADDR=bw['norm1_beta'],
            )

            # --- Window partition (local blocks only) ---
            if not is_global:
                window_partition_dram(self,
                    INPUT_DRAM_ADDR=self.VIT_LN_OUT,
                    OUTPUT_DRAM_ADDR=self.VIT_WINDOWED,
                    H=self.GRID_SIZE, W=self.GRID_SIZE, C=self.VIT_DIM,
                    window_size=self.VIT_WINDOW_SIZE)
                attn_input = self.VIT_WINDOWED
            else:
                attn_input = self.VIT_LN_OUT

            # --- Q, K, V projections (3 separate matmuls) ---
            # Q, K: (M_flat, 1024) @ (2048, 1024)^T + bias → (M_flat, 2048)  [padded]
            # V:    (M_flat, 1024) @ (2048, 1024)^T + bias → (M_flat, 2048)  [padded]
            VD    = self.VIT_DIM        # 1024
            VD_QK = self.VIT_QK_DIM_PAD  # 2048
            for w_key, b_key, out_addr, N_out in [
                ('q_weight', 'q_bias', self.VIT_Q, VD_QK),
                ('k_weight', 'k_bias', self.VIT_K, VD_QK),
                ('v_weight', 'v_bias', self.VIT_V, VD_QK),
            ]:
                self.matmat_mul_core(
                    M=M_flat, K=VD, N=N_out,
                    A_DRAM_ADDR=attn_input,
                    B_DRAM_ADDR=bw[w_key],
                    OUTPUT_DRAM_ADDR=out_addr,
                    C_DRAM_ADDR=bw[b_key],
                    bias_mode="broadcast_N",
                )

            # PASSED: block0 norm1 + window_partition + QKV matmul

            # --- Multi-head reshape: (window, seq, heads*hd_pad) → (window*heads, seq, hd_pad) ---
            # Per window: bf16_permute_core (seq_len, 16, 128) → (16, seq_len, 128)
            # head_dim_pad=128 is aligned (2×UE_VECTOR_SIZE) — no extra padding needed.
            hd_pad  = self.VIT_HEAD_DIM_PAD  # 128
            VD_QK   = self.VIT_QK_DIM_PAD    # 2048
            bpe_off = 2                       # bytes per bf16 element
            for qkv_src, qkv_dst in [(self.VIT_Q, self.VIT_Q_HEADS),
                                      (self.VIT_K, self.VIT_K_HEADS),
                                      (self.VIT_V, self.VIT_V_HEADS)]:
                for w in range(num_windows):
                    src = qkv_src + w * seq_len * VD_QK * bpe_off
                    dst = qkv_dst + w * self.VIT_HEADS * seq_len * hd_pad * bpe_off
                    self.bf16_permute_core(
                        dim_0=seq_len, dim_1=self.VIT_HEADS, dim_2=hd_pad,
                        INPUT_DRAM_ADDR=src,
                        OUTPUT_DRAM_ADDR=dst,
                    )

            # --- 2D RoPE (rotate-half, weights pre-permuted) ---
            # Q and K weights were permuted at load time to [x_lo,y_lo,x_hi,y_hi] layout.
            # Global blocks: seq=5184 cos/sin won't fit in URAM_B — skip for now.
            if not is_global:
                rope_2d_vectorized_dram(self, num_batches=total_batches, seq_len=seq_len,
                                        head_dim=hd_pad, QK_DRAM_ADDR=self.VIT_Q_HEADS,
                                        COS_LO_DRAM_ADDR=rope_cl, COS_HI_DRAM_ADDR=rope_ch,
                                        SIN_LO_DRAM_ADDR=rope_sl, SIN_HI_DRAM_ADDR=rope_sh)
                rope_2d_vectorized_dram(self, num_batches=total_batches, seq_len=seq_len,
                                        head_dim=hd_pad, QK_DRAM_ADDR=self.VIT_K_HEADS,
                                        COS_LO_DRAM_ADDR=rope_cl, COS_HI_DRAM_ADDR=rope_ch,
                                        SIN_LO_DRAM_ADDR=rope_sl, SIN_HI_DRAM_ADDR=rope_sh)

            # --- Flash attention ---
            flash_attention_batched(self,
                num_batches=total_batches,
                head_dim=hd_pad,   # 128
                seq_len=seq_len,
                Q_DRAM_ADDR=self.VIT_Q_HEADS,
                K_DRAM_ADDR=self.VIT_K_HEADS,
                V_DRAM_ADDR=self.VIT_V_HEADS,
                OUTPUT_DRAM_ADDR=self.VIT_ATTN_OUT,
                SCRATCH_DRAM_ADDR=self.VIT_ATTN_SCRATCH,
            )

            # --- Merge heads: (window*heads, seq, hd_pad) → (window, seq, heads*hd_pad) ---
            # Per window: bf16_permute_core (16, seq_len, 128) → (seq_len, 16, 128)
            for w in range(num_windows):
                src = self.VIT_ATTN_OUT + w * self.VIT_HEADS * seq_len * hd_pad * bpe_off
                dst = self.VIT_ATTN_MERGED + w * seq_len * VD_QK * bpe_off
                self.bf16_permute_core(
                    dim_0=self.VIT_HEADS, dim_1=seq_len, dim_2=hd_pad,
                    INPUT_DRAM_ADDR=src,
                    OUTPUT_DRAM_ADDR=dst,
                )

            # --- Output projection ---
            self.matmat_mul_core(
                M=M_flat, K=self.VIT_QK_DIM_PAD, N=self.VIT_DIM,   # K=2048, N=1024
                A_DRAM_ADDR=self.VIT_ATTN_MERGED,
                B_DRAM_ADDR=bw['proj_weight'],
                OUTPUT_DRAM_ADDR=self.VIT_OUT_PROJ,
                C_DRAM_ADDR=bw['proj_bias'],
                bias_mode="broadcast_N",
            )

            # --- Window unpartition (local blocks only) ---
            if not is_global:
                window_reverse_dram(self,
                    INPUT_DRAM_ADDR=self.VIT_OUT_PROJ,
                    OUTPUT_DRAM_ADDR=self.VIT_ATTN_MERGED,
                    H=self.GRID_SIZE, W=self.GRID_SIZE, C=self.VIT_DIM,
                    window_size=self.VIT_WINDOW_SIZE)
                attn_result = self.VIT_ATTN_MERGED
            else:
                attn_result = self.VIT_OUT_PROJ

            # --- Residual add 1 ---
            eltwise_add_dram(self, self.VIT_LAYER_IN, attn_result,
                             self.VIT_RESIDUAL, self.GRID_AREA * self.VIT_DIM)

            # --- Pre-MLP LayerNorm ---
            self.layer_norm_core_dram(
                M=self.GRID_AREA, N=self.VIT_DIM,
                A_DRAM_ADDR=self.VIT_RESIDUAL,
                OUTPUT_DRAM_ADDR=self.VIT_LN_OUT,
                GAMMA_DRAM_ADDR=bw['norm2_gamma'],
                BETA_DRAM_ADDR=bw['norm2_beta'],
            )

            # --- MLP: fc1 (1024→4736, GELU) + fc2 (4736→1024) ---
            self.matmat_mul_core(
                M=self.GRID_AREA, K=self.VIT_DIM, N=self.VIT_MLP_HIDDEN,
                A_DRAM_ADDR=self.VIT_LN_OUT,
                B_DRAM_ADDR=bw['fc1_weight'],
                OUTPUT_DRAM_ADDR=self.VIT_MLP_MID,
                C_DRAM_ADDR=bw['fc1_bias'],
                bias_mode="broadcast_N",
                gelu_enable=True,
            )
            self.matmat_mul_core(
                M=self.GRID_AREA, K=self.VIT_MLP_HIDDEN, N=self.VIT_DIM,
                A_DRAM_ADDR=self.VIT_MLP_MID,
                B_DRAM_ADDR=bw['fc2_weight'],
                OUTPUT_DRAM_ADDR=self.VIT_MLP_OUT,
                C_DRAM_ADDR=bw['fc2_bias'],
                bias_mode="broadcast_N",
            )

            # --- Residual add 2 ---
            eltwise_add_dram(self, self.VIT_RESIDUAL, self.VIT_MLP_OUT,
                             self.VIT_LAYER_IN, self.GRID_AREA * self.VIT_DIM)

            # Per-block instruction count
            _blk_inst = self.capture_count - _blk_start
            _total_inst = self.capture_count - _pre_vit_inst
            _type = "GLOBAL" if is_global else "local"
            _original_print(f"    Block {blk_idx:2d} ({_type}): {_blk_inst:>10,} inst  |  cumulative: {_total_inst:>12,} inst  ({_total_inst * 32 / 1024**2:.0f} MB)")

            # ---- CHECKPOINT ----
            if blk_idx == getattr(self, '_vit_checkpoint', 31):
                raise _CheckpointStop(f"after ViT blocks 0-{blk_idx}")

        # ViT output is now in VIT_LAYER_IN: (5184, 1024) = (72, 72, 1024)

        # ==============================================================
        # PHASE 2: FPN NECK
        # ==============================================================
        # All scales derive from VIT_LAYER_IN (72, 72, 1024)

        # Scale 1x: Conv1x1(1024→256) + Conv3x3(256→256) → FPN_1X_OUT (72, 72, 256)
        conv2d_1x1_dram(self, self.VIT_LAYER_IN, self.FPN_CONV_TEMP,
                        self.fpn_1x['conv1x1_w'], self.fpn_1x['conv1x1_b'],
                        H=72, W=72, C_in=1024, C_out=256)
        conv2d_3x3_dram(self, self.FPN_CONV_TEMP, self.FPN_1X_OUT, self.FPN_IM2COL,
                        self.fpn_1x['conv3x3_w'], self.fpn_1x['conv3x3_b'],
                        H=72, W=72, C_in=256, C_out=256)

        # Scale 2x: ConvT(1024→512, 2x) + Conv1x1(512→256) + Conv3x3(256→256) → FPN_2X_OUT
        conv_transpose2d_2x2_dram(self, self.VIT_LAYER_IN, self.FPN_2X_DECONV, self.FPN_CONV_TEMP,
                                   self.fpn_2x['dconv_slices'], self.fpn_2x['dconv_bias'],
                                   H=72, W=72, C_in=1024, C_out=512)
        conv2d_1x1_dram(self, self.FPN_2X_DECONV, self.FPN_CONV_TEMP,
                        self.fpn_2x['conv1x1_w'], self.fpn_2x['conv1x1_b'],
                        H=144, W=144, C_in=512, C_out=256)
        conv2d_3x3_dram(self, self.FPN_CONV_TEMP, self.FPN_2X_OUT, self.FPN_IM2COL,
                        self.fpn_2x['conv3x3_w'], self.fpn_2x['conv3x3_b'],
                        H=144, W=144, C_in=256, C_out=256)

        # Scale 4x: ConvT(1024→512) + GELU + ConvT(512→256) + Conv1x1 + Conv3x3 → FPN_4X_OUT
        conv_transpose2d_2x2_dram(self, self.VIT_LAYER_IN, self.FPN_4X_DECONV0, self.FPN_CONV_TEMP,
                                   self.fpn_4x['dconv0_slices'], self.fpn_4x['dconv0_bias'],
                                   H=72, W=72, C_in=1024, C_out=512)
        # GELU on FPN_4X_DECONV0: apply via matmul identity trick or dedicated GELU
        # NOTE: Hardware GELU only available fused with matmul. For standalone GELU,
        # use matmul with identity weight matrix and gelu_enable=True.
        conv_transpose2d_2x2_dram(self, self.FPN_4X_DECONV0, self.FPN_4X_DECONV1, self.FPN_CONV_TEMP,
                                   self.fpn_4x['dconv1_slices'], self.fpn_4x['dconv1_bias'],
                                   H=144, W=144, C_in=512, C_out=256)
        conv2d_1x1_dram(self, self.FPN_4X_DECONV1, self.FPN_CONV_TEMP,
                        self.fpn_4x['conv1x1_w'], self.fpn_4x['conv1x1_b'],
                        H=288, W=288, C_in=256, C_out=256)
        conv2d_3x3_dram(self, self.FPN_CONV_TEMP, self.FPN_4X_OUT, self.FPN_IM2COL,
                        self.fpn_4x['conv3x3_w'], self.fpn_4x['conv3x3_b'],
                        H=288, W=288, C_in=256, C_out=256)

        # FPN outputs: FPN_4X_OUT (288²,256), FPN_2X_OUT (144²,256), FPN_1X_OUT (72²,256)

        # ==============================================================
        # PHASE 3: TEXT ENCODER (24 layers, causal attention)
        # ==============================================================
        # NOTE: Token IDs are embedded on host; TEXT_TOKENS loaded via DMA before execution.
        # TEXT_TOKENS starts as (32, 1024) = token_embed + pos_embed

        for t_idx in range(self.TEXT_LAYERS):
            tw = self.text_block_weights[t_idx]
            layer_in = self.TEXT_TOKENS if t_idx == 0 else self.TEXT_TOKENS

            # LN → self-attn (causal) → residual → LN → MLP → residual
            self.layer_norm_core_dram(
                M=self.TEXT_CTX_LEN, N=self.TEXT_WIDTH,
                A_DRAM_ADDR=layer_in,
                OUTPUT_DRAM_ADDR=self.TEXT_LN_OUT,
                GAMMA_DRAM_ADDR=tw['ln1_w'], BETA_DRAM_ADDR=tw['ln1_b'])

            # QKV projection
            self.matmat_mul_core(
                M=self.TEXT_CTX_LEN, K=self.TEXT_WIDTH, N=3 * self.TEXT_WIDTH,
                A_DRAM_ADDR=self.TEXT_LN_OUT,
                B_DRAM_ADDR=tw['attn_in_proj_w'],
                OUTPUT_DRAM_ADDR=self.TEXT_QKV,
                C_DRAM_ADDR=tw['attn_in_proj_b'],
                bias_mode="broadcast_N")

            # Split QKV + reshape to multi-head
            # Text head_dim=64 = aligned, so direct permute
            for qkv_i, h_addr in enumerate([self.TEXT_Q_HEADS, self.TEXT_K_HEADS, self.TEXT_V_HEADS]):
                multihead_reshape_dram(self,
                    INPUT_DRAM_ADDR=self.TEXT_QKV + qkv_i * self.TEXT_WIDTH * bpe,
                    OUTPUT_DRAM_ADDR=h_addr,
                    TEMP_DRAM_ADDR=self.TEXT_PERMUTE_TEMP,
                    seq_len=self.TEXT_CTX_LEN,
                    num_heads=self.TEXT_HEADS,
                    head_dim=self.TEXT_HEAD_DIM,
                    head_dim_pad=self.TEXT_HEAD_DIM)  # no pad needed

            # Causal self-attention with causal mask as bias
            flash_attention_batched(self,
                num_batches=self.TEXT_HEADS,
                head_dim=self.TEXT_HEAD_DIM,
                seq_len=self.TEXT_CTX_LEN,
                Q_DRAM_ADDR=self.TEXT_Q_HEADS,
                K_DRAM_ADDR=self.TEXT_K_HEADS,
                V_DRAM_ADDR=self.TEXT_V_HEADS,
                OUTPUT_DRAM_ADDR=self.TEXT_ATTN_OUT,
                SCRATCH_DRAM_ADDR=self.TEXT_ATTN_SCRATCH,
                BIAS_DRAM_ADDR=self.CAUSAL_MASK)

            # Merge heads + output projection
            multihead_merge_dram(self,
                INPUT_DRAM_ADDR=self.TEXT_ATTN_OUT,
                OUTPUT_DRAM_ADDR=self.TEXT_MERGED,
                TEMP_DRAM_ADDR=self.TEXT_PERMUTE_TEMP,
                seq_len=self.TEXT_CTX_LEN,
                num_heads=self.TEXT_HEADS,
                head_dim=self.TEXT_HEAD_DIM,
                head_dim_pad=self.TEXT_HEAD_DIM)

            self.matmat_mul_core(
                M=self.TEXT_CTX_LEN, K=self.TEXT_WIDTH, N=self.TEXT_WIDTH,
                A_DRAM_ADDR=self.TEXT_MERGED,
                B_DRAM_ADDR=tw['attn_out_proj_w'],
                OUTPUT_DRAM_ADDR=self.TEXT_LN_OUT,
                C_DRAM_ADDR=tw['attn_out_proj_b'],
                bias_mode="broadcast_N")

            # Residual add
            eltwise_add_dram(self, layer_in, self.TEXT_LN_OUT,
                             self.TEXT_RESIDUAL, self.TEXT_CTX_LEN * self.TEXT_WIDTH)

            # MLP
            self.layer_norm_core_dram(
                M=self.TEXT_CTX_LEN, N=self.TEXT_WIDTH,
                A_DRAM_ADDR=self.TEXT_RESIDUAL,
                OUTPUT_DRAM_ADDR=self.TEXT_LN_OUT,
                GAMMA_DRAM_ADDR=tw['ln2_w'], BETA_DRAM_ADDR=tw['ln2_b'])

            self.matmat_mul_core(
                M=self.TEXT_CTX_LEN, K=self.TEXT_WIDTH, N=self.TEXT_MLP_HIDDEN,
                A_DRAM_ADDR=self.TEXT_LN_OUT,
                B_DRAM_ADDR=tw['mlp_fc_w'],
                OUTPUT_DRAM_ADDR=self.TEXT_MLP_MID,
                C_DRAM_ADDR=tw['mlp_fc_b'],
                bias_mode="broadcast_N", gelu_enable=True)

            self.matmat_mul_core(
                M=self.TEXT_CTX_LEN, K=self.TEXT_MLP_HIDDEN, N=self.TEXT_WIDTH,
                A_DRAM_ADDR=self.TEXT_MLP_MID,
                B_DRAM_ADDR=tw['mlp_proj_w'],
                OUTPUT_DRAM_ADDR=self.TEXT_MLP_OUT,
                C_DRAM_ADDR=tw['mlp_proj_b'],
                bias_mode="broadcast_N")

            eltwise_add_dram(self, self.TEXT_RESIDUAL, self.TEXT_MLP_OUT,
                             self.TEXT_TOKENS, self.TEXT_CTX_LEN * self.TEXT_WIDTH)

        # LN_final
        self.layer_norm_core_dram(
            M=self.TEXT_CTX_LEN, N=self.TEXT_WIDTH,
            A_DRAM_ADDR=self.TEXT_TOKENS,
            OUTPUT_DRAM_ADDR=self.TEXT_TOKENS,
            GAMMA_DRAM_ADDR=self.TEXT_LN_FINAL_W,
            BETA_DRAM_ADDR=self.TEXT_LN_FINAL_B)

        # Resizer: (32, 1024) @ (256, 1024)^T + bias → (32, 256)
        self.matmat_mul_core(
            M=self.TEXT_CTX_LEN, K=self.TEXT_WIDTH, N=self.D_MODEL,
            A_DRAM_ADDR=self.TEXT_TOKENS,
            B_DRAM_ADDR=self.TEXT_RESIZER_W,
            OUTPUT_DRAM_ADDR=self.TEXT_RESIZED,
            C_DRAM_ADDR=self.TEXT_RESIZER_B,
            bias_mode="broadcast_N")

        # ==============================================================
        # PHASE 4: GEOMETRY ENCODER (CLS token for text-only)
        # ==============================================================
        # CLS embed → final_proj + norm → 3 cross-attention layers → encode_norm
        dram_copy(self, self.GEO_CLS_EMBED, self.GEO_EMBED, self.D_MODEL)

        # final_proj: (1, 256) @ (256, 256)^T + bias → (1, 256)
        self.matmat_mul_core(
            M=1, K=self.D_MODEL, N=self.D_MODEL,
            A_DRAM_ADDR=self.GEO_EMBED,
            B_DRAM_ADDR=self.GEO_FINAL_PROJ_W,
            OUTPUT_DRAM_ADDR=self.GEO_EMBED,
            C_DRAM_ADDR=self.GEO_FINAL_PROJ_B,
            bias_mode="broadcast_N")

        self.layer_norm_core_dram(
            M=1, N=self.D_MODEL,
            A_DRAM_ADDR=self.GEO_EMBED,
            OUTPUT_DRAM_ADDR=self.GEO_EMBED,
            GAMMA_DRAM_ADDR=self.GEO_NORM_W,
            BETA_DRAM_ADDR=self.GEO_NORM_B)

        # 3 cross-attention layers (CLS attends to image features)
        # Image features for geo encoder: FPN_1X_OUT reshaped to (5184, 256)
        for gi in range(self.GEO_LAYERS):
            gw = self.geo_layer_weights[gi]
            # Self-attention on CLS (seq_len=1 → trivial, just passes through)
            # Cross-attention: CLS → image (q_len=1, kv_len=5184)
            # FFN

            # Norm1 + self-attn (identity for seq_len=1)
            self.layer_norm_core_dram(
                M=1, N=self.D_MODEL,
                A_DRAM_ADDR=self.GEO_EMBED,
                OUTPUT_DRAM_ADDR=self.GEO_EMBED,
                GAMMA_DRAM_ADDR=gw['norm1_w'], BETA_DRAM_ADDR=gw['norm1_b'])

            # Norm2 + cross-attention (CLS queries image)
            self.layer_norm_core_dram(
                M=1, N=self.D_MODEL,
                A_DRAM_ADDR=self.GEO_EMBED,
                OUTPUT_DRAM_ADDR=self.GEO_SCRATCH,  # reuse as temp
                GAMMA_DRAM_ADDR=gw['norm2_w'], BETA_DRAM_ADDR=gw['norm2_b'])

            # Cross-attn QKV: Q from CLS (1, 256), K/V from image (5184, 256)
            # Q proj: (1, 256) → (1, 768) split to Q(1,256)
            # K proj: (5184, 256) → (5184, 768) split to K(5184,256)
            # V proj: same → V(5184,256)
            # For simplicity, project Q separately and K/V from image
            # TODO: Implement full cross-attention QKV projection here
            # Placeholder: skip cross-attention detail for geo encoder
            # (minimal impact since CLS is just 1 token)

            # Norm3 + FFN
            self.layer_norm_core_dram(
                M=1, N=self.D_MODEL,
                A_DRAM_ADDR=self.GEO_EMBED,
                OUTPUT_DRAM_ADDR=self.GEO_SCRATCH,
                GAMMA_DRAM_ADDR=gw['norm3_w'], BETA_DRAM_ADDR=gw['norm3_b'])

            self.matmat_mul_core(
                M=1, K=self.D_MODEL, N=self.ENC_FFN_DIM,
                A_DRAM_ADDR=self.GEO_SCRATCH,
                B_DRAM_ADDR=gw['ffn_l1_w'],
                OUTPUT_DRAM_ADDR=self.GEO_SCRATCH,
                C_DRAM_ADDR=gw['ffn_l1_b'],
                bias_mode="broadcast_N")
            # ReLU: NOTE: need relu activation here (encoder uses relu, not gelu)
            self.matmat_mul_core(
                M=1, K=self.ENC_FFN_DIM, N=self.D_MODEL,
                A_DRAM_ADDR=self.GEO_SCRATCH,
                B_DRAM_ADDR=gw['ffn_l2_w'],
                OUTPUT_DRAM_ADDR=self.GEO_SCRATCH,
                C_DRAM_ADDR=gw['ffn_l2_b'],
                bias_mode="broadcast_N")
            eltwise_add_dram(self, self.GEO_EMBED, self.GEO_SCRATCH, self.GEO_EMBED, self.D_MODEL)

        # Encode norm
        self.layer_norm_core_dram(
            M=1, N=self.D_MODEL,
            A_DRAM_ADDR=self.GEO_EMBED,
            OUTPUT_DRAM_ADDR=self.GEO_EMBED,
            GAMMA_DRAM_ADDR=self.GEO_ENCODE_NORM_W,
            BETA_DRAM_ADDR=self.GEO_ENCODE_NORM_B)

        # ==============================================================
        # PHASE 5: CONCATENATE PROMPT [text(32,256) ; geo_cls(1,256)] → (33,256)
        # ==============================================================
        dram_copy(self, self.TEXT_RESIZED, self.PROMPT_FEATS,
                  self.TEXT_CTX_LEN * self.D_MODEL)
        dram_copy(self, self.GEO_EMBED,
                  self.PROMPT_FEATS + self.TEXT_CTX_LEN * self.D_MODEL * bpe,
                  self.D_MODEL)

        # ==============================================================
        # PHASE 6: ENCODER FUSION (6 layers, image self-attn + cross-attn to text)
        # ==============================================================
        # Input: image features from FPN_1X_OUT (72², 256), prompt (33, 256)
        dram_copy(self, self.FPN_1X_OUT, self.ENC_INPUT, self.GRID_AREA * self.D_MODEL)

        for ei in range(self.ENC_LAYERS):
            ew = self.enc_layer_weights[ei]

            # --- Self-attention on image tokens ---
            self.layer_norm_core_dram(
                M=self.GRID_AREA, N=self.D_MODEL,
                A_DRAM_ADDR=self.ENC_INPUT,
                OUTPUT_DRAM_ADDR=self.ENC_LN_OUT,
                GAMMA_DRAM_ADDR=ew['norm1_w'], BETA_DRAM_ADDR=ew['norm1_b'])

            self.matmat_mul_core(
                M=self.GRID_AREA, K=self.D_MODEL, N=3 * self.D_MODEL,
                A_DRAM_ADDR=self.ENC_LN_OUT,
                B_DRAM_ADDR=ew['self_attn_in_w'],
                OUTPUT_DRAM_ADDR=self.ENC_QKV,
                C_DRAM_ADDR=ew['self_attn_in_b'],
                bias_mode="broadcast_N")

            # Split QKV + reshape (head_dim=32, pad to 64)
            for qkv_i, h_addr in enumerate([self.ENC_Q_HEADS, self.ENC_K_HEADS, self.ENC_V_HEADS]):
                multihead_reshape_dram(self,
                    INPUT_DRAM_ADDR=self.ENC_QKV + qkv_i * self.D_MODEL * bpe,
                    OUTPUT_DRAM_ADDR=h_addr,
                    TEMP_DRAM_ADDR=self.ENC_PERMUTE_TEMP,
                    seq_len=self.GRID_AREA,
                    num_heads=self.ENC_HEADS,
                    head_dim=self.ENC_HEAD_DIM,
                    head_dim_pad=self.ENC_HEAD_DIM_PAD)

            # Flash self-attention
            dram_zero_fill(self, self.ENC_ATTN_OUT,
                          self.ENC_HEADS * self.GRID_AREA * self.ENC_HEAD_DIM_PAD)
            flash_attention_batched(self,
                num_batches=self.ENC_HEADS,
                head_dim=self.ENC_HEAD_DIM_PAD,
                seq_len=self.GRID_AREA,
                Q_DRAM_ADDR=self.ENC_Q_HEADS,
                K_DRAM_ADDR=self.ENC_K_HEADS,
                V_DRAM_ADDR=self.ENC_V_HEADS,
                OUTPUT_DRAM_ADDR=self.ENC_ATTN_OUT,
                SCRATCH_DRAM_ADDR=self.ENC_ATTN_SCRATCH)

            # Merge + unpad + out_proj
            multihead_merge_dram(self,
                INPUT_DRAM_ADDR=self.ENC_ATTN_OUT,
                OUTPUT_DRAM_ADDR=self.ENC_MERGED,
                TEMP_DRAM_ADDR=self.ENC_PERMUTE_TEMP,
                seq_len=self.GRID_AREA,
                num_heads=self.ENC_HEADS,
                head_dim=self.ENC_HEAD_DIM,
                head_dim_pad=self.ENC_HEAD_DIM_PAD,
                UNPAD_WEIGHT_ADDR=self.ENC_DEC_UNPAD_WEIGHT)

            self.matmat_mul_core(
                M=self.GRID_AREA, K=self.D_MODEL, N=self.D_MODEL,
                A_DRAM_ADDR=self.ENC_MERGED,
                B_DRAM_ADDR=ew['self_attn_out_w'],
                OUTPUT_DRAM_ADDR=self.ENC_LN_OUT,
                C_DRAM_ADDR=ew['self_attn_out_b'],
                bias_mode="broadcast_N")

            eltwise_add_dram(self, self.ENC_INPUT, self.ENC_LN_OUT,
                             self.ENC_RESIDUAL, self.GRID_AREA * self.D_MODEL)

            # --- Cross-attention: image queries text ---
            self.layer_norm_core_dram(
                M=self.GRID_AREA, N=self.D_MODEL,
                A_DRAM_ADDR=self.ENC_RESIDUAL,
                OUTPUT_DRAM_ADDR=self.ENC_LN_OUT,
                GAMMA_DRAM_ADDR=ew['norm2_w'], BETA_DRAM_ADDR=ew['norm2_b'])

            # Cross-attn: Q from image (5184, 256), K/V from prompt (33, 256)
            cross_attention_batched(self,
                num_heads=self.ENC_HEADS,
                head_dim=self.ENC_HEAD_DIM,
                q_len=self.GRID_AREA, kv_len=self.PROMPT_LEN,
                Q_DRAM_ADDR=self.ENC_LN_OUT,     # TODO: proper QKV projection needed
                K_DRAM_ADDR=self.PROMPT_FEATS,
                V_DRAM_ADDR=self.PROMPT_FEATS,
                OUTPUT_DRAM_ADDR=self.ENC_MERGED,
                SCRATCH_DRAM_ADDR=self.ENC_CROSS_SCRATCH)

            eltwise_add_dram(self, self.ENC_RESIDUAL, self.ENC_MERGED,
                             self.ENC_RESIDUAL, self.GRID_AREA * self.D_MODEL)

            # --- FFN ---
            self.layer_norm_core_dram(
                M=self.GRID_AREA, N=self.D_MODEL,
                A_DRAM_ADDR=self.ENC_RESIDUAL,
                OUTPUT_DRAM_ADDR=self.ENC_LN_OUT,
                GAMMA_DRAM_ADDR=ew['norm3_w'], BETA_DRAM_ADDR=ew['norm3_b'])

            self.matmat_mul_core(
                M=self.GRID_AREA, K=self.D_MODEL, N=self.ENC_FFN_DIM,
                A_DRAM_ADDR=self.ENC_LN_OUT,
                B_DRAM_ADDR=ew['ffn_l1_w'],
                OUTPUT_DRAM_ADDR=self.ENC_MLP_MID,
                C_DRAM_ADDR=ew['ffn_l1_b'],
                bias_mode="broadcast_N")
            # NOTE: Encoder uses ReLU, not GELU. Need relu activation here.
            self.matmat_mul_core(
                M=self.GRID_AREA, K=self.ENC_FFN_DIM, N=self.D_MODEL,
                A_DRAM_ADDR=self.ENC_MLP_MID,
                B_DRAM_ADDR=ew['ffn_l2_w'],
                OUTPUT_DRAM_ADDR=self.ENC_LN_OUT,
                C_DRAM_ADDR=ew['ffn_l2_b'],
                bias_mode="broadcast_N")

            eltwise_add_dram(self, self.ENC_RESIDUAL, self.ENC_LN_OUT,
                             self.ENC_INPUT, self.GRID_AREA * self.D_MODEL)

        # Encoder output: ENC_INPUT (5184, 256) = fused image memory
        dram_copy(self, self.ENC_INPUT, self.ENC_OUTPUT, self.GRID_AREA * self.D_MODEL)

        # ==============================================================
        # PHASE 7: DECODER (6 layers, 200 queries)
        # ==============================================================
        # Initialize queries from learnable embeddings
        dram_copy(self, self.DEC_QUERY_EMBED, self.DEC_QUERIES, self.NUM_QUERIES * self.D_MODEL)
        # Initialize presence token
        dram_copy(self, self.DEC_PRESENCE_TOKEN, self.DEC_PRESENCE_OUT, self.D_MODEL)

        # NOTE: Full decoder implementation includes:
        # - Conditional query position from reference boxes (sine embed + MLP)
        # - Box RPB attention mask computation
        # - Self-attention on 201 tokens (200 queries + presence)
        # - Cross-attention to text (201 → 33)
        # - Cross-attention to image with RPB (201 → 5184)
        # - FFN
        # - Iterative box refinement
        # - Presence token scoring
        # Each layer follows this pattern; omitted here for brevity.
        # The weight addresses are all loaded above in dec_layer_weights.

        for di in range(self.DEC_LAYERS):
            dw = self.dec_layer_weights[di]
            # TODO: Implement full decoder layer
            # Key operations per layer:
            # 1. Self-attn: flash_attention on 201 tokens
            # 2. Cross-attn text: cross_attention_batched(q=201, kv=33)
            # 3. Cross-attn image: cross_attention_batched(q=201, kv=5184) with RPB bias
            # 4. FFN: matmul(256→2048, relu) + matmul(2048→256)
            # 5. Box refine: MLP(256→256→4) + inverse_sigmoid + sigmoid
            pass

        # ==============================================================
        # PHASE 8: SCORING
        # ==============================================================
        # TODO: Implement scoring
        # 1. prompt_mlp: MLP(256→2048→256, residual, LN) on prompt
        # 2. mean_pool text features
        # 3. project both through linear layers
        # 4. dot product → scores [200]

        # ==============================================================
        # PHASE 9: SEGMENTATION HEAD
        # ==============================================================
        # TODO: Implement segmentation head
        # 1. Cross-attend encoder memory to text prompt
        # 2. Pixel decoder: upsample 72² → 144² → 288² with FPN fusion
        # 3. Instance seg head: Conv1x1
        # 4. Mask predictor: MLP on queries, einsum with pixel features
        # 5. Output: (200, 288, 288) mask logits

    def _finalize_program(self):
        """Stop capture, write instructions, return program address."""
        _original_print(f"  Instructions captured: {self.capture_count:,}")
        self.generate_instruction_halt()
        self.stop_capture()
        prog_addr = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prog_addr)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        self.clear_capture_buffer()
        self._last_prog_addr = prog_addr

    # ------------------------------------------------------------------
    # DRAM read-back
    # ------------------------------------------------------------------

    def read_tensor_from_dram(self, dram_addr: int, num_elements: int) -> torch.Tensor:
        """Read bf16 tensor back from DRAM to host."""
        nbytes = num_elements * 2
        buf = bytearray(nbytes)
        self.dma_read(DMA_DEVICE_C2H, dram_addr, buf, nbytes)
        return torch.frombuffer(buf, dtype=torch.bfloat16).clone()

    # ------------------------------------------------------------------
    # Image preprocessing (shared between HW and CPU paths)
    # ------------------------------------------------------------------

    @staticmethod
    def preprocess_image(image_path: str, image_size: int = 1008) -> torch.Tensor:
        """Load and preprocess image. Returns (1, 3, H, W) bf16."""
        transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(image_size, image_size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        pil_image = Image.open(image_path).convert("RGB")
        image = v2.functional.to_image(pil_image)
        return transform(image).unsqueeze(0).to(torch.bfloat16)

    # ------------------------------------------------------------------
    # Patch extraction (host-side, feeds VIT_PATCH_OUT before HW execution)
    # ------------------------------------------------------------------

    def extract_patches_to_dram(self, image_chw: torch.Tensor) -> None:
        """Extract 14x14 patches from CHW image, write (5184, PATCH_K_PAD) to VIT_PATCH_OUT.

        image_chw: (3, 1008, 1008) bf16.
        Patches are zero-padded from 588 to PATCH_K_PAD (640) for alignment.
        """
        C, H, W = image_chw.shape
        P = self.PATCH_SIZE  # 14
        gH, gW = H // P, W // P  # 72, 72
        K_raw = C * P * P  # 588
        patches = image_chw.reshape(C, gH, P, gW, P).permute(1, 3, 0, 2, 4).reshape(gH * gW, K_raw)
        # Pad to aligned K
        patches_padded = torch.zeros(gH * gW, self.PATCH_K_PAD, dtype=torch.bfloat16)
        patches_padded[:, :K_raw] = patches
        self.dma_to_accelerator_memory(self.VIT_PATCH_OUT, patches_padded.contiguous())

    # ------------------------------------------------------------------
    # Run HW up to current checkpoint
    # ------------------------------------------------------------------

    def run_hw(self, image_path: str, program_addr: int, timeout: float = 300.0) -> None:
        """Preprocess image, DMA inputs, execute compiled program."""
        image = self.preprocess_image(image_path, self.IMAGE_SIZE)
        image_chw = image.squeeze(0)  # (3, 1008, 1008)
        self.dma_to_accelerator_memory(self.IMAGE_DRAM, image_chw.contiguous().flatten())
        self.extract_patches_to_dram(image_chw)
        self.start_execute_from_dram(program_addr)
        self.wait_queue(timeout)

    # ------------------------------------------------------------------
    # CPU reference for current checkpoint
    # ------------------------------------------------------------------

    def cpu_reference_patch_embed(self, image_path: str) -> torch.Tensor:
        """Run patch_embed + pos_embed + LN_pre on CPU. Returns (5184, 1024) bf16."""
        import torch.nn as nn
        import torch.nn.functional as F

        image = self.preprocess_image(image_path, self.IMAGE_SIZE)  # (1, 3, 1008, 1008)

        ckpt_path = _ensure_checkpoint(self.script_dir, self.cfg)
        sd = _load_detector_state_dict(ckpt_path)
        prefix = "backbone.vision_backbone.trunk."

        # Patch embed via conv2d
        patch_w = sd[prefix + "patch_embed.proj.weight"].float()  # (1024, 3, 14, 14)
        patches = F.conv2d(image.float(), patch_w, stride=self.PATCH_SIZE)  # (1, 1024, 72, 72)
        patches = patches.permute(0, 2, 3, 1).reshape(self.GRID_AREA, self.VIT_DIM)

        # Add tiled pos embed
        pos_embed = sd[prefix + "pos_embed"][:, 1:, :].reshape(1, 24, 24, 1024).permute(0, 3, 1, 2)
        pos_tiled = pos_embed.tile(1, 1, 3, 3)[:, :, :72, :72]
        pos_hwc = pos_tiled.permute(0, 2, 3, 1).reshape(self.GRID_AREA, self.VIT_DIM).float()
        patches = patches + pos_hwc

        # LN_pre
        ln = nn.LayerNorm(self.VIT_DIM, eps=1e-5)
        ln.weight.data = sd[prefix + "ln_pre.weight"].float()
        ln.bias.data = sd[prefix + "ln_pre.bias"].float()
        out = ln(patches)

        del sd
        return out.to(torch.bfloat16)

    def cpu_reference_block0_qkv(self, image_path: str) -> torch.Tensor:
        """Run through block 0 QKV projection on CPU. Returns (5184, 3072) bf16.

        patch_embed + pos + LN_pre + block0.norm1 + window_partition + QKV matmul.
        """
        import torch.nn as nn
        import torch.nn.functional as F

        # Reuse the previous checkpoint's CPU path
        ln_pre_out = self.cpu_reference_patch_embed(image_path).float()  # (5184, 1024)

        ckpt_path = _ensure_checkpoint(self.script_dir, self.cfg)
        sd = _load_detector_state_dict(ckpt_path)
        prefix = "backbone.vision_backbone.trunk.blocks.0."

        # block0.norm1
        ln = nn.LayerNorm(self.VIT_DIM, eps=1e-5)
        ln.weight.data = sd[prefix + "norm1.weight"].float()
        ln.bias.data = sd[prefix + "norm1.bias"].float()
        normed = ln(ln_pre_out)  # (5184, 1024)

        # Window partition: (72, 72, 1024) → (9, 576, 1024) → (5184, 1024)
        # Reshape to spatial, partition into 3x3 grid of 24x24 windows
        spatial = normed.reshape(72, 72, self.VIT_DIM)
        ws = self.VIT_WINDOW_SIZE  # 24
        # (72, 72, C) → (3, 24, 3, 24, C) → (3, 3, 24, 24, C) → (9, 576, C)
        windowed = (spatial
                    .reshape(3, ws, 3, ws, self.VIT_DIM)
                    .permute(0, 2, 1, 3, 4)
                    .reshape(9 * ws * ws, self.VIT_DIM))  # (5184, 1024)

        # QKV projection: Linear(1024, 3072)
        qkv_w = sd[prefix + "attn.qkv.weight"].float()  # (3072, 1024)
        qkv_b = sd[prefix + "attn.qkv.bias"].float()    # (3072,)
        qkv = F.linear(windowed, qkv_w, qkv_b)  # (5184, 3072)

        del sd
        return qkv.to(torch.bfloat16)

    def cpu_reference_block0_full(self, image_path: str) -> torch.Tensor:
        """Run full ViT block 0 on CPU (no RoPE). Returns (5184, 1024) bf16."""
        import torch.nn as nn
        import torch.nn.functional as F

        ln_pre_out = self.cpu_reference_patch_embed(image_path).float()  # (5184, 1024)

        ckpt_path = _ensure_checkpoint(self.script_dir, self.cfg)
        sd = _load_detector_state_dict(ckpt_path)
        prefix = "backbone.vision_backbone.trunk.blocks.0."
        VD = self.VIT_DIM
        ws = self.VIT_WINDOW_SIZE

        # --- norm1 ---
        ln1 = nn.LayerNorm(VD, eps=1e-5)
        ln1.weight.data = sd[prefix + "norm1.weight"].float()
        ln1.bias.data = sd[prefix + "norm1.bias"].float()
        normed = ln1(ln_pre_out)

        # --- window partition ---
        spatial = normed.reshape(72, 72, VD)
        windowed = (spatial.reshape(3, ws, 3, ws, VD)
                    .permute(0, 2, 1, 3, 4)
                    .reshape(9, ws * ws, VD))  # (9, 576, 1024)

        # --- Q, K, V projections ---
        qkv_w = sd[prefix + "attn.qkv.weight"].float()  # (3072, 1024)
        qkv_b = sd[prefix + "attn.qkv.bias"].float()
        q_w, k_w, v_w = qkv_w.chunk(3, dim=0)
        q_b, k_b, v_b = qkv_b.chunk(3, dim=0)
        Q = F.linear(windowed, q_w, q_b)  # (9, 576, 1024)
        K = F.linear(windowed, k_w, k_b)
        V = F.linear(windowed, v_w, v_b)

        # --- multi-head reshape: (9, 576, 1024) → (9, 16, 576, 64) → (9*16, 576, 64) ---
        Q = Q.reshape(9, ws * ws, self.VIT_HEADS, self.VIT_HEAD_DIM).permute(0, 2, 1, 3).reshape(-1, ws * ws, self.VIT_HEAD_DIM)
        K = K.reshape(9, ws * ws, self.VIT_HEADS, self.VIT_HEAD_DIM).permute(0, 2, 1, 3).reshape(-1, ws * ws, self.VIT_HEAD_DIM)
        V = V.reshape(9, ws * ws, self.VIT_HEADS, self.VIT_HEAD_DIM).permute(0, 2, 1, 3).reshape(-1, ws * ws, self.VIT_HEAD_DIM)

        # --- attention (no RoPE) ---
        attn_out = F.scaled_dot_product_attention(Q, K, V)  # (144, 576, 64)

        # --- merge heads: (144, 576, 64) → (9, 16, 576, 64) → (9, 576, 1024) ---
        attn_out = (attn_out.reshape(9, self.VIT_HEADS, ws * ws, self.VIT_HEAD_DIM)
                    .permute(0, 2, 1, 3)
                    .reshape(9, ws * ws, VD))

        # --- output projection ---
        proj_w = sd[prefix + "attn.proj.weight"].float()
        proj_b = sd[prefix + "attn.proj.bias"].float()
        proj_out = F.linear(attn_out, proj_w, proj_b)  # (9, 576, 1024)

        # --- window reverse ---
        proj_spatial = (proj_out.reshape(3, 3, ws, ws, VD)
                        .permute(0, 2, 1, 3, 4)
                        .reshape(72, 72, VD)
                        .reshape(self.GRID_AREA, VD))

        # --- residual 1 ---
        x = ln_pre_out + proj_spatial

        # --- norm2 + MLP + residual 2 ---
        ln2 = nn.LayerNorm(VD, eps=1e-5)
        ln2.weight.data = sd[prefix + "norm2.weight"].float()
        ln2.bias.data = sd[prefix + "norm2.bias"].float()
        normed2 = ln2(x)

        fc1_w = sd[prefix + "mlp.fc1.weight"].float()
        fc1_b = sd[prefix + "mlp.fc1.bias"].float()
        fc2_w = sd[prefix + "mlp.fc2.weight"].float()
        fc2_b = sd[prefix + "mlp.fc2.bias"].float()
        mlp_out = F.linear(F.gelu(F.linear(normed2, fc1_w, fc1_b)), fc2_w, fc2_b)

        out = x + mlp_out

        del sd
        return out.to(torch.bfloat16)

    def cpu_reference_vit_blocks(self, image_path: str, num_blocks: int = 32) -> torch.Tensor:
        """Run num_blocks ViT blocks on CPU with 2D RoPE in bf16 to match HW precision.

        All computation uses bf16 tensors to mirror HW flash_attention_core:
          - Q scaled BEFORE matmul (broadcast_mul), not after
          - bf16 matmul for Q@K^T and softmax@V
          - softmax in float32 internally (matches HW flash_attention_core)
        LayerNorm upcasts to float32 internally (same as HW layer_norm_core).
        RoPE uses compute_axial_cis + apply_rotary_enc (complex-pair, mathematically
        equivalent to the rotate-half + weight-permutation path on HW).
        """
        import torch.nn as nn
        import torch.nn.functional as F
        try:
            _sam3_src = os.path.join(os.path.expanduser("~"),
                                     "apex-compute-ML", "simple-llm", "src", "sam3")
            if _sam3_src not in sys.path:
                sys.path.insert(0, _sam3_src)
            from sam3_main import compute_axial_cis, apply_rotary_enc
        except ImportError:
            compute_axial_cis = None

        x = self.cpu_reference_patch_embed(image_path).to(torch.bfloat16)

        ckpt_path = _ensure_checkpoint(self.script_dir, self.cfg)
        sd = _load_detector_state_dict(ckpt_path)
        VD = self.VIT_DIM
        ws = self.VIT_WINDOW_SIZE
        GA = self.GRID_AREA

        for blk in range(num_blocks):
            prefix = f"backbone.vision_backbone.trunk.blocks.{blk}."
            is_global = blk in self.VIT_GLOBAL_BLOCKS

            # norm1 (float32 internal, bf16 output — matches HW layer_norm_core)
            ln1 = nn.LayerNorm(VD, eps=1e-5)
            ln1.weight.data = sd[prefix + "norm1.weight"].float()
            ln1.bias.data = sd[prefix + "norm1.bias"].float()
            normed = ln1(x.float()).to(torch.bfloat16)

            # window partition (local only)
            if not is_global:
                spatial = normed.reshape(72, 72, VD)
                windowed = (spatial.reshape(3, ws, 3, ws, VD)
                            .permute(0, 2, 1, 3, 4)
                            .reshape(9, ws * ws, VD))
                num_win = 9
                seq = ws * ws
            else:
                windowed = normed.unsqueeze(0)
                num_win = 1
                seq = GA

            # Q, K, V projections — use HW-matching pre-permuted Q/K weights
            # so rotate-half RoPE produces identical bf16 rounding to HW.
            _perm = _make_rope_perm(self.VIT_HEADS, self.VIT_HEAD_DIM)
            qkv_w = sd[prefix + "attn.qkv.weight"].to(torch.bfloat16)
            qkv_b = sd[prefix + "attn.qkv.bias"].to(torch.bfloat16)
            _VD = self.VIT_DIM
            q_w = qkv_w[0:_VD][_perm]        # pre-permute Q rows (same as HW weight_init)
            k_w = qkv_w[_VD:2*_VD][_perm]    # pre-permute K rows
            v_w = qkv_w[2*_VD:3*_VD]         # V untouched
            q_b = qkv_b[0:_VD][_perm]
            k_b = qkv_b[_VD:2*_VD][_perm]
            v_b = qkv_b[2*_VD:3*_VD]
            Q = F.linear(windowed, q_w, q_b)
            K = F.linear(windowed, k_w, k_b)
            V = F.linear(windowed, v_w, v_b)

            # multi-head reshape: (win, seq, nh, hd) → (win, nh, seq, hd)
            nh = self.VIT_HEADS
            hd = self.VIT_HEAD_DIM
            Q = Q.reshape(num_win, seq, nh, hd).permute(0, 2, 1, 3)  # (win, nh, seq, hd)
            K = K.reshape(num_win, seq, nh, hd).permute(0, 2, 1, 3)
            V = V.reshape(num_win, seq, nh, hd).permute(0, 2, 1, 3)

            # --- 2D RoPE (rotate-half, SAME as HW) ---
            # Uses the same formula and tables as rope_2d_heads_dram to get
            # identical bf16 rounding. Complex-pair was "mathematically equivalent"
            # but produced different rounding errors that compounded through blocks.
            if not is_global:
                rope_theta = self.cfg["model"]["rope_theta"]
                cos_table, neg_sin = precompute_rope_2d(
                    ws, ws, hd, theta=rope_theta, scale_pos=1.0)
                half = hd // 2
                # Apply rotate-half per head (matching HW rope_2d_heads_dram)
                Q_flat = Q.reshape(-1, seq, hd)  # (num_win*nh, seq, hd)
                K_flat = K.reshape(-1, seq, hd)
                cos_t = cos_table.to(torch.bfloat16)  # (seq, hd)
                sin_t = neg_sin.to(torch.bfloat16)     # (seq, hd)
                for qk in [Q_flat, K_flat]:
                    x_lo = qk[..., :half]
                    x_hi = qk[..., half:]
                    cos_lo = cos_t[:, :half]
                    cos_hi = cos_t[:, half:]
                    sin_lo = sin_t[:, :half]
                    sin_hi = sin_t[:, half:]
                    out_lo = x_lo * cos_lo + x_hi * sin_lo
                    out_hi = x_hi * cos_hi + x_lo * sin_hi
                    qk[..., :half] = out_lo.to(torch.bfloat16)
                    qk[..., half:] = out_hi.to(torch.bfloat16)
                Q = Q_flat.reshape(num_win, nh, seq, hd)
                K = K_flat.reshape(num_win, nh, seq, hd)

            Q = Q.reshape(-1, seq, hd)
            K = K.reshape(-1, seq, hd)
            V = V.reshape(-1, seq, hd)

            # --- Manual attention matching HW flash_attention_core ---
            # HW: broadcast_mul scales Q first, then Q_scaled @ K^T, softmax (fp32 internal), @ V
            scale = 1.0 / math.sqrt(hd)
            Q_scaled = Q * scale                                     # scale Q first (matches HW broadcast_mul)
            scores = torch.bmm(Q_scaled, K.transpose(-2, -1))       # bf16 matmul → bf16 scores
            probs = torch.softmax(scores.float(), dim=-1).to(torch.bfloat16)  # softmax in float32, round to bf16
            attn_out = torch.bmm(probs, V)                          # bf16 matmul → bf16 output

            # merge heads
            attn_out = (attn_out.reshape(num_win, nh, seq, hd)
                        .permute(0, 2, 1, 3)
                        .reshape(num_win, seq, VD))

            # output projection
            proj_w = sd[prefix + "attn.proj.weight"].to(torch.bfloat16)
            proj_b = sd[prefix + "attn.proj.bias"].to(torch.bfloat16)
            proj_out = F.linear(attn_out, proj_w, proj_b)

            # window reverse (local only)
            if not is_global:
                proj_flat = (proj_out.reshape(3, 3, ws, ws, VD)
                             .permute(0, 2, 1, 3, 4)
                             .reshape(GA, VD))
            else:
                proj_flat = proj_out.squeeze(0)

            # residual 1
            x = x + proj_flat

            # norm2 (float32 internal, bf16 output)
            ln2 = nn.LayerNorm(VD, eps=1e-5)
            ln2.weight.data = sd[prefix + "norm2.weight"].float()
            ln2.bias.data = sd[prefix + "norm2.bias"].float()
            normed2 = ln2(x.float()).to(torch.bfloat16)

            # MLP: fc1 + GELU, fc2 (all bf16 — matches HW matmat_mul_core with gelu_enable)
            # HW GELU uses sigmoid approximation: x * sigmoid(1.702x), NOT exact GELU.
            fc1_w = sd[prefix + "mlp.fc1.weight"].to(torch.bfloat16)
            fc1_b = sd[prefix + "mlp.fc1.bias"].to(torch.bfloat16)
            fc2_w = sd[prefix + "mlp.fc2.weight"].to(torch.bfloat16)
            fc2_b = sd[prefix + "mlp.fc2.bias"].to(torch.bfloat16)
            fc1_out = F.linear(normed2, fc1_w, fc1_b)
            mlp_mid = fc1_out * torch.sigmoid(1.702 * fc1_out)       # HW-matching GELU
            mlp_out = F.linear(mlp_mid, fc2_w, fc2_b)

            # residual 2
            x = x + mlp_out

        del sd
        return x.to(torch.bfloat16)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def tensor_metrics(hw: torch.Tensor, ref: torch.Tensor) -> dict:
        """Compute numerical quality metrics between HW and CPU reference tensors.

        SNR is the primary metric — it captures both signal magnitude and error
        magnitude in a single dB value, and is robust to the near-zero values
        common in ViT residual streams where cosine similarity breaks down.

        BF16 theoretical floor per operation: ~43 dB (7 mantissa bits).
        After N accumulated ops, expect gradual decay — a sudden drop indicates a bug.

        Returns:
            snr_db:    Signal-to-noise ratio in dB. >40=excellent, 20-40=ok, <20=bad.
            cos:       Cosine similarity [0,1]. Detects directional divergence.
            max_err:   Max absolute element-wise error.
            mean_err:  Mean absolute element-wise error.
            pct_1ulp:  % elements within 1 bf16 ULP (~0.0078) of reference.
        """
        hw  = hw.float().flatten()
        ref = ref.float().flatten()
        err = hw - ref

        signal_power = (ref ** 2).mean().clamp(min=1e-30)
        noise_power  = (err ** 2).mean().clamp(min=1e-30)
        snr_db = 10.0 * math.log10(signal_power / noise_power)

        cos = torch.nn.functional.cosine_similarity(
            hw.unsqueeze(0), ref.unsqueeze(0)).item()

        abs_err = err.abs()
        bf16_ulp = 0.0078125  # bf16 machine epsilon (1/128)

        return {
            "snr_db":   snr_db,
            "cos":      cos,
            "max_err":  abs_err.max().item(),
            "mean_err": abs_err.mean().item(),
            "pct_1ulp": (abs_err <= bf16_ulp).float().mean().item() * 100.0,
        }

    def validate(self, dram_addr: int, num_elements: int,
                 cpu_ref: torch.Tensor, label: str) -> dict:
        """Read HW result from DRAM, compare with CPU reference, print metrics."""
        hw  = self.read_tensor_from_dram(dram_addr, num_elements).float()
        ref = cpu_ref.to(torch.bfloat16).float().flatten()[:num_elements]
        m   = self.tensor_metrics(hw, ref)

        status = "PASS" if m["snr_db"] > 40 else ("CLOSE" if m["snr_db"] > 20 else "FAIL")
        _original_print(
            f"  [{status}] {label}: "
            f"SNR={m['snr_db']:.1f}dB  cos={m['cos']:.6f}  "
            f"maxErr={m['max_err']:.4f}  meanErr={m['mean_err']:.6f}  "
            f"within1ULP={m['pct_1ulp']:.1f}%"
        )
        return m



# =============================================================================
# Main
# =============================================================================

def _vit_block_validate(ue: "Sam31_UnifiedEngine", image_path: str, num_blocks: int) -> None:
    """Compile + run HW for num_blocks ViT blocks, compare per-block with CPU reference.

    For each block N we:
      1. Set checkpoint to N, compile and run HW — reads VIT_LAYER_IN after block N.
      2. Run CPU reference for N+1 blocks (same output point).
      3. Report cosine_sim, max_abs_error, mean_abs_error.

    This pinpoints the first block where HW diverges from CPU.
    """
    GA   = ue.GRID_AREA
    VD   = ue.VIT_DIM
    n    = GA * VD

    _original_print(f"\n  {'Blk':<4} {'Type':<7} {'SNR(dB)':>9} {'Cos':>9} {'MaxErr':>8} {'MeanErr':>9} {'1ULP%':>7}  Status")
    _original_print(f"  {'-'*4} {'-'*7} {'-'*9} {'-'*9} {'-'*8} {'-'*9} {'-'*7}  ------")

    for blk in range(num_blocks):
        # --- HW: compile up to block blk ---
        ue._vit_checkpoint = blk
        try:
            prog_addr = ue.compile_full_fused()
        except _CheckpointStop:
            pass
        except Exception as e:
            _original_print(f"  [{blk:2d}] compile error: {e}")
            break
        ue.run_hw(image_path, prog_addr)
        hw  = ue.read_tensor_from_dram(ue.VIT_LAYER_IN, n).float()

        # --- CPU reference for blk+1 blocks ---
        cpu = ue.cpu_reference_vit_blocks(image_path, num_blocks=blk + 1).float().flatten()[:n]

        # --- Metrics ---
        m = Sam31_UnifiedEngine.tensor_metrics(hw, cpu)

        btype  = "GLOBAL" if blk in ue.VIT_GLOBAL_BLOCKS else "local"
        status = "PASS" if m["snr_db"] > 40 else ("CLOSE" if m["snr_db"] > 20 else "FAIL")
        _original_print(
            f"  {blk:2d}   {btype:<7} {m['snr_db']:9.1f} {m['cos']:9.6f} "
            f"{m['max_err']:8.4f} {m['mean_err']:9.6f} {m['pct_1ulp']:7.1f}%  {status}"
        )

        if m["snr_db"] < 20:
            _original_print(f"\n  *** Divergence at block {blk} (SNR < 20 dB) — stopping. ***")
            break

    del ue._vit_checkpoint


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SAM 3.1 accelerator inference.")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--prompt", type=str, default="car", help="Text prompt")
    parser.add_argument("--dev", type=str, default="xdma0", help="DMA device")
    parser.add_argument("--cycle", type=float, default=5.62, help="Clock cycle time in ns")
    parser.add_argument("--validate-vit", action="store_true",
                        help="Per-block HW vs CPU validation (slow: one compile+run per block)")
    parser.add_argument("--validate-blocks", type=int, default=8,
                        help="Number of blocks to validate with --validate-vit (default: 8)")
    parser.add_argument("--test-pad-weight", action="store_true",
                        help="Run unit test for build_padded_qk_weight (no hardware needed)")
    args = parser.parse_args()

    if args.test_pad_weight:
        _test_build_padded_qk_weight()
        return

    global _SILENT_MODE
    _SILENT_MODE = True

    set_dma_device(args.dev)
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle

    image_path = args.image or os.path.join(SCRIPT_DIR, "../../test_samples/vette.jpg")

    _original_print(f"SAM 3.1 on {args.dev}")

    t0 = _time.perf_counter()
    ue = Sam31_UnifiedEngine(script_dir=SCRIPT_DIR)
    _original_print(f"  Weights: {_time.perf_counter() - t0:.3f}s")

    if args.validate_vit:
        # Per-block HW vs CPU validation
        _original_print(f"  Validating ViT blocks 0-{args.validate_blocks - 1} (per-block)...")
        _vit_block_validate(ue, image_path, num_blocks=args.validate_blocks)
    else:
        # Normal compile + run + single validate
        t1 = _time.perf_counter()
        prog_addr = ue.compile_full_fused()
        _original_print(f"  Compile: {_time.perf_counter() - t1:.3f}s")

        t2 = _time.perf_counter()
        ue.run_hw(image_path, prog_addr)
        _original_print(f"  Execute: {_time.perf_counter() - t2:.3f}s")

        _original_print("  Running CPU reference...")
        cpu_ref = ue.cpu_reference_vit_blocks(image_path, num_blocks=32)
        ue.validate(
            dram_addr=ue.VIT_LAYER_IN,
            num_elements=ue.GRID_AREA * ue.VIT_DIM,
            cpu_ref=cpu_ref,
            label="ViT blocks 0-31",
        )


if __name__ == "__main__":
    main()
