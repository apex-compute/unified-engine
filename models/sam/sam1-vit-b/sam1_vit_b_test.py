#!/usr/bin/env python3
"""
SAM 1 ViT-B segmentation inference on accelerator.

  - Config from sam1_vit_b_config.json; weights from facebook/sam-vit-base (HuggingFace).
  - Forward pass: ViT-B backbone (12 blocks, dim=768, 12 heads, window=14)
                  → neck (1×1 conv + LN + 3×3 conv + LN → 256-dim features).
  - All BF16 weights (no quantization). ~86M image encoder params ≈ 172 MB.
  - No RoPE: SAM1 uses decomposed 2D relative positional embeddings (rel_pos).
    NOTE: rel_pos is not yet implemented on-chip; initial pass skips it.
  - Window padding: 64×64 grid padded to 70×70 (next multiple of window_size=14).
    Boundary windows (9 of 25) include padding tokens with non-zero bias — minor
    accuracy impact; unpad step discards padding outputs before residual add.

Weights:
  - Default: sam1_vit_b_bin/model.safetensors (downloaded from facebook/sam-vit-base).

Usage:
  python sam1_vit_b_test.py
  python sam1_vit_b_test.py --image photo.jpg
  python sam1_vit_b_test.py --dev xdma0 [--cycle 5.62]

Fixed layout: sam1_vit_b_test.py, sam1_vit_b_config.json, sam1_vit_b_bin/ live in same folder.
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
# user_dma_core lives at repo root: models/sam/sam1-vit-b/ → ../../.. = repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))

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
# Section 2 uses 1 GB instructions (0xC0000000–0xFFFFFFFF); override before instantiating S2 engine.
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

def flash_attention_global_tiled(ue: UnifiedEngine, num_heads: int, head_dim: int,
                                  seq_len: int,
                                  Q_DRAM_ADDR: int, K_DRAM_ADDR: int,
                                  V_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                                  SCRATCH_DRAM_ADDR: int) -> int:
    """Global self-attention processing one head at a time to bound scratch usage.

    flash_attention_core needs (head_dim * seq_len + seq_len²) scratch per head.
    For seq=4096, head_dim=64: ~4 MB per head — fits in local scratch (111 MB)
    when scratch is reused across heads instead of allocated per-head simultaneously.
    """
    bpe = 2
    head_stride = seq_len * head_dim * bpe

    total_flops = 0
    for h in range(num_heads):
        total_flops += ue.flash_attention_core(
            head_dim=head_dim,
            seq_len=seq_len,
            Q_DRAM_ADDR=Q_DRAM_ADDR      + h * head_stride,
            K_DRAM_ADDR=K_DRAM_ADDR      + h * head_stride,
            V_DRAM_ADDR=V_DRAM_ADDR      + h * head_stride,
            OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR + h * head_stride,
            SCRATCH_DRAM_ADDR=SCRATCH_DRAM_ADDR,  # reused each head
        )
    return total_flops


def flash_attention_batched(ue: UnifiedEngine, num_batches: int, head_dim: int,
                            seq_len: int, Q_DRAM_ADDR: int, K_DRAM_ADDR: int,
                            V_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                            SCRATCH_DRAM_ADDR: int,
                            BIAS_DRAM_ADDR: int = None,
                            bias_shared: bool = False) -> int:
    """Batched flash self-attention: loop over num_batches calling flash_attention_core.

    Each batch is one attention head. Q/K/V/OUTPUT are contiguous per batch with
    stride (seq_len * head_dim). BIAS stride is (seq_len * seq_len) if provided,
    or 0 if bias_shared=True (single mask broadcast across all batches).
    """
    bpe = 2
    qkv_stride = seq_len * head_dim * bpe
    out_stride = seq_len * head_dim * bpe
    scratch_stride = head_dim * seq_len * bpe + seq_len * seq_len * bpe
    bias_stride = 0 if (BIAS_DRAM_ADDR is None or bias_shared) else seq_len * seq_len * bpe

    total_flops = 0
    for b in range(num_batches):
        bias_addr = BIAS_DRAM_ADDR if (BIAS_DRAM_ADDR is not None) else None
        if bias_addr is not None and not bias_shared:
            bias_addr = BIAS_DRAM_ADDR + b * bias_stride
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
    scratch_per_head = score_size * bpe  # one slot: fused scores+softmax output

    total_flops = 0
    scale = 1.0 / math.sqrt(head_dim)

    for h in range(num_heads):
        q_addr = Q_DRAM_ADDR + h * q_stride
        k_addr = K_DRAM_ADDR + h * kv_stride
        v_addr = V_DRAM_ADDR + h * kv_stride
        o_addr = OUTPUT_DRAM_ADDR + h * out_stride
        scores_addr = SCRATCH_DRAM_ADDR + h * scratch_per_head

        # Step 1: pre-scale Q_h in place by 1/sqrt(head_dim)
        # matmat_mul_core(softmax_enable=True) fuses scores+softmax; pre-scaling Q
        # avoids a separate score-scaling pass.
        q_elems = q_len * head_dim
        for offset in range(0, q_elems, URAM_NEAR_FULL_ELEMENTS):
            take = min(URAM_NEAR_FULL_ELEMENTS, q_elems - offset)
            ue.accelerator_memory_to_sram(
                accelerator_dram_address=q_addr + offset * bpe,
                sram_address=0x00000, element_size=take,
            )
            ue.broadcast_mul(
                scalar=scale,
                sram_start_addr=0x00000, sram_wb_addr=0x00000,
                element_size=take,
            )
            ue.sram_to_accelerator_memory(
                sram_address=0x00000,
                accelerator_dram_address=q_addr + offset * bpe,
                element_size=take,
            )

        # Step 2: add bias to scores_addr before fused matmul+softmax (if provided)
        if BIAS_DRAM_ADDR is not None:
            # Compute raw scores first, then add bias, then softmax below
            ue.matmat_mul_core(
                M=q_len, K=head_dim, N=kv_len,
                A_DRAM_ADDR=q_addr,
                B_DRAM_ADDR=k_addr,
                OUTPUT_DRAM_ADDR=scores_addr,
            )
            total_flops += 2 * q_len * head_dim * kv_len
            bias_per_head = q_len * kv_len * bpe
            bias_addr = BIAS_DRAM_ADDR + h * bias_per_head
            total_elems = q_len * kv_len
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
            # softmax on the bias-added scores (read scores_addr, write scores_addr)
            ue.matmat_mul_core(
                M=q_len, K=1, N=q_len * kv_len,
                A_DRAM_ADDR=scores_addr,
                B_DRAM_ADDR=scores_addr,
                OUTPUT_DRAM_ADDR=scores_addr,
                softmax_enable=True,
            )
        else:
            # Step 3: fused scores + softmax: softmax(Q_scaled @ K^T) → scores_addr
            ue.matmat_mul_core(
                M=q_len, K=head_dim, N=kv_len,
                A_DRAM_ADDR=q_addr,
                B_DRAM_ADDR=k_addr,
                OUTPUT_DRAM_ADDR=scores_addr,
                softmax_enable=True,
            )
            total_flops += 2 * q_len * head_dim * kv_len
        total_flops += 5 * q_len * kv_len  # approx for softmax

        # Step 4: out = softmax @ V  → [q_len, head_dim]
        # matmat_mul_core computes A @ B^T; transpose V → V^T stored after softmax.
        v_t_addr = scores_addr + score_size * bpe  # temp for V^T
        ue.bf16_permute_core(
            dim_0=kv_len, dim_1=head_dim, dim_2=1,
            INPUT_DRAM_ADDR=v_addr,
            OUTPUT_DRAM_ADDR=v_t_addr,
        )
        ue.matmat_mul_core(
            M=q_len, K=kv_len, N=head_dim,
            A_DRAM_ADDR=scores_addr,
            B_DRAM_ADDR=v_t_addr,
            OUTPUT_DRAM_ADDR=o_addr,
        )
        total_flops += 2 * q_len * kv_len * head_dim

    return total_flops


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

    For SAM1 ViT (head_dim=64, pad=64): no padding needed, just permute.
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
                     relu_enable: bool = False,
                     ZERO_PAD_DRAM_ADDR: int = None) -> int:
    """Conv2d with 3x3 kernel, padding=1, stride=1 on HWC data.

    Processes one row at a time: im2col for row h into IM2COL_DRAM (reused),
    then matmul for that row directly into OUTPUT. IM2COL_DRAM only needs
    W * 9 * C_in elements (one row), not H*W * 9 * C_in.

    Weight must be pre-reshaped from (C_out, C_in, 3, 3) to (C_out, 9*C_in).
    """
    bpe = 2
    K = 9 * C_in

    for h in range(H):
        # Step 1: im2col for row h — gather 3x3 neighborhoods into IM2COL_DRAM[0..W*K]
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
                        elif ZERO_PAD_DRAM_ADDR is not None:
                            ue.accelerator_memory_to_sram(
                                accelerator_dram_address=ZERO_PAD_DRAM_ADDR,
                                sram_address=patch_sram, element_size=C_in)
                sram_offset += K * bpe

            row_offset = w_start * K * bpe
            ue.sram_to_accelerator_memory(
                sram_address=0x00000,
                accelerator_dram_address=IM2COL_DRAM_ADDR + row_offset,
                element_size=w_count * K)

        # Step 2: matmul for row h — (W, K) @ (C_out, K)^T + bias → OUTPUT[h*W*C_out..]
        ue.matmat_mul_core(
            M=W, K=K, N=C_out,
            A_DRAM_ADDR=IM2COL_DRAM_ADDR,
            B_DRAM_ADDR=WEIGHT_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR + h * W * C_out * bpe,
            C_DRAM_ADDR=BIAS_DRAM_ADDR,
            bias_mode="broadcast_N",
        )
    return 2 * H * W * K * C_out


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
# Feature map pad / unpad helpers (SAM1 ViT-B: 64×64 grid → 70×70 for window=14)
# =============================================================================

def pad_feature_map_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                          H: int, W: int, H_PAD: int, W_PAD: int, C: int) -> None:
    """Pad (H, W, C) feature map to (H_PAD, W_PAD, C) in DRAM.
    Places input in top-left; remaining rows/cols zero-filled.
    H_PAD and W_PAD must both be multiples of the window_size.
    """
    bpe = 2
    dram_zero_fill(ue, OUTPUT_DRAM_ADDR, H_PAD * W_PAD * C)
    for h in range(H):
        src = INPUT_DRAM_ADDR + h * W * C * bpe
        dst = OUTPUT_DRAM_ADDR + h * W_PAD * C * bpe
        dram_copy(ue, src, dst, W * C)


def unpad_feature_map_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                            H: int, W: int, W_PAD: int, C: int) -> None:
    """Extract top-left (H, W, C) from (H, W_PAD, C) padded feature map in DRAM."""
    bpe = 2
    for h in range(H):
        src = INPUT_DRAM_ADDR + h * W_PAD * C * bpe
        dst = OUTPUT_DRAM_ADDR + h * W * C * bpe
        dram_copy(ue, src, dst, W * C)


# =============================================================================
# Config / checkpoint helpers
# =============================================================================

def _load_config(script_dir: str = SCRIPT_DIR) -> dict:
    cp = os.path.join(script_dir, "sam1_vit_b_config.json")
    with open(cp) as f:
        return json.load(f)


def _ensure_checkpoint(script_dir: str, cfg: dict) -> str:
    """Download SAM1 ViT-B checkpoint from HuggingFace. Returns path to safetensors file."""
    paths = cfg["paths"]
    bin_dir = os.path.join(script_dir, paths.get("bin_dir", "sam1_vit_b_bin"))
    os.makedirs(bin_dir, exist_ok=True)

    ckpt_path = os.path.join(bin_dir, paths["hf_checkpoint_filename"])
    if not os.path.exists(ckpt_path):
        _original_print(f"Downloading SAM1 ViT-B checkpoint from {paths['hf_repo_id']}...")
        hf_hub_download(repo_id=paths["hf_repo_id"],
                        filename=paths["hf_checkpoint_filename"],
                        local_dir=bin_dir)
    return ckpt_path


def _load_sam1_state_dict(ckpt_path: str) -> dict:
    """Load SAM1 checkpoint. Supports safetensors and torch formats."""
    try:
        from safetensors.torch import load_file
        sd = load_file(ckpt_path)
    except (ImportError, Exception):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
    _original_print(f"  Loaded {len(sd)} weight tensors")
    return sd



# =============================================================================
# DRAM layout — SAM1 ViT-B
#   Params:      0x00000000 – 0x7FFFFFFF  (2 GB)   weights
#   Tensors:     0x80000000 – 0x9BFFFFFF  (448 MB) activations / scratch
#   Instructions:0x9C000000 – 0xFFFFFFFF  (1.6 GB) compiled instruction stream
# =============================================================================
SAM1_PARAMS_BASE  = 0x00000000
SAM1_TENSOR_BASE  = 0x80000000
SAM1_PROGRAM_BASE = 0x9C000000


# =============================================================================
# Sam1VitB_UnifiedEngine
# =============================================================================

class Sam1VitB_UnifiedEngine(UnifiedEngine):
    """SAM 1 ViT-B image segmentation on Unified Engine FPGA accelerator.

    Architecture:
        ViT-B backbone (12 blocks, dim=768, 12 heads, window=14, 4 global blocks)
        → Neck (Conv 1×1 → LN → Conv 3×3 → LN → 256-dim feature map at 64×64)

    NOTE: rel_pos (decomposed 2D relative positional embeddings) is not yet
    implemented on-chip. Attention runs without positional bias — accuracy is
    reduced but architecture and weight loading are otherwise correct.
    """

    # --- Architecture constants ---
    IMAGE_SIZE       = 1024
    PATCH_SIZE       = 16
    NUM_CHANNELS     = 3
    GRID_SIZE        = 64           # 1024 // 16
    GRID_AREA        = 4096         # 64 * 64
    GRID_SIZE_PAD    = 70           # ceil(64/14)*14 — window partition padding
    GRID_AREA_PAD    = 4900         # 70 * 70

    # ViT backbone
    VIT_DIM          = 768
    VIT_DEPTH        = 12
    VIT_HEADS        = 12
    VIT_HEAD_DIM     = 64           # 768 // 12
    VIT_HEAD_DIM_PAD = 64           # no padding needed: head_dim == UE_VECTOR_SIZE
    VIT_QK_DIM_PAD   = 768          # VIT_HEADS * VIT_HEAD_DIM_PAD (== VIT_DIM)
    VIT_MLP_HIDDEN   = 3072         # 768 * 4
    VIT_WINDOW_SIZE  = 14
    VIT_WINDOW_AREA  = 196          # 14 * 14
    VIT_WINDOW_AREA_PAD = 256       # ceil(196/64)*64 — next 64-aligned seq for flash_attention SRAM
    VIT_NUM_WINDOWS  = 25           # (70 // 14) ** 2 = 25 (padded grid)
    VIT_GLOBAL_BLOCKS = {2, 5, 8, 11}
    VIT_PRETRAIN_GRID = 64          # pos_embed already at 64×64, no resize needed

    # SAM1 neck output
    NECK_DIM         = 256

    # SAM1 mask decoder
    NUM_MASKS        = 3            # num_multimask_outputs
    DEC_DIM          = 256          # transformer_dim
    DEC_HEADS        = 8            # num_heads for self-attn
    DEC_HEAD_DIM     = 32           # 256 // 8
    DEC_INTERNAL_DIM = 128          # cross-attn internal dim (downsample_rate=2)
    DEC_INTERNAL_HD  = 16           # 128 // 8
    DEC_MLP_DIM      = 2048
    DEC_LAYERS       = 2
    # Tokens for a single-point prompt: 1 iou + 4 mask + 1 fg point + 1 not-a-point pad
    DEC_NUM_TOKENS   = 7

    PATCH_K_RAW = PATCH_SIZE * PATCH_SIZE * NUM_CHANNELS  # 16*16*3 = 768
    PATCH_K_PAD = ((PATCH_K_RAW + 63) // 64) * 64        # 768 (already 64-aligned)

    ALIGN = 64  # UE_VECTOR_SIZE

    def __init__(self, script_dir: str = SCRIPT_DIR, weights_bin: str = None):
        super().__init__(params_dram_base=SAM1_PARAMS_BASE,
                         tensor_dram_base=SAM1_TENSOR_BASE,
                         program_dram_base=SAM1_PROGRAM_BASE)
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
        """Load SAM1 ViT-B weights from HuggingFace checkpoint to DRAM."""
        ckpt_path = _ensure_checkpoint(self.script_dir, self.cfg)
        sd = _load_sam1_state_dict(ckpt_path)
        bpe = 2

        # ==================================================================
        # ViT BACKBONE  (keys: vision_encoder.*)
        # ==================================================================
        vit = "vision_encoder."

        # Patch embedding: Conv2d(3, 768, 16, stride=16)
        # Weight: (768, 3, 16, 16) → reshape to (768, 768) for matmat_mul_core.
        # PATCH_K_RAW = 768, already 64-aligned — no padding needed.
        N = self.VIT_DIM  # 768
        patch_w = sd[vit + "patch_embed.projection.weight"].to(torch.bfloat16)
        self.PATCH_EMBED_WEIGHT = self._alloc_param(patch_w.reshape(N, self.PATCH_K_RAW))
        self.PATCH_EMBED_BIAS   = self._alloc_param(sd[vit + "patch_embed.projection.bias"])

        # Absolute positional embedding: (1, 64, 64, 768) — already at target resolution.
        pos_embed = sd[vit + "pos_embed"]  # (1, 64, 64, 768)
        pos_hwc = pos_embed.reshape(self.GRID_AREA, N).to(torch.bfloat16)
        self.POS_EMBED = self._alloc_param(pos_hwc)  # (4096, 768)

        # SAM1 has no LN_pre (trunk norm is applied per-block only).

        # Per-block weights
        # SAM1 uses combined attn.qkv weight (2304, 768) — split into Q/K/V (768, 768) each.
        # No RoPE, no head_dim padding: head_dim=64 == UE_VECTOR_SIZE, directly aligned.
        # Q scale: flash_attention uses 1/sqrt(head_dim=64) correctly — no correction needed.
        self.vit_block_weights = []
        for i in range(self.VIT_DEPTH):
            bp = f"{vit}layers.{i}."
            bw = {}
            bw['norm1_gamma'] = self._alloc_param(sd[bp + "layer_norm1.weight"])
            bw['norm1_beta']  = self._alloc_param(sd[bp + "layer_norm1.bias"])
            _VD = self.VIT_DIM  # 768
            _raw_w = sd[bp + "attn.qkv.weight"].to(torch.bfloat16)  # (2304, 768)
            _raw_b = sd[bp + "attn.qkv.bias"].to(torch.bfloat16)    # (2304,)
            bw['q_weight']    = self._alloc_param(_raw_w[0:_VD])         # (768, 768)
            bw['q_bias']      = self._alloc_param(_raw_b[0:_VD])
            bw['k_weight']    = self._alloc_param(_raw_w[_VD:2*_VD])     # (768, 768)
            bw['k_bias']      = self._alloc_param(_raw_b[_VD:2*_VD])
            bw['v_weight']    = self._alloc_param(_raw_w[2*_VD:3*_VD])   # (768, 768)
            bw['v_bias']      = self._alloc_param(_raw_b[2*_VD:3*_VD])
            bw['proj_weight'] = self._alloc_param(sd[bp + "attn.proj.weight"])  # (768, 768)
            bw['proj_bias']   = self._alloc_param(sd[bp + "attn.proj.bias"])
            bw['norm2_gamma'] = self._alloc_param(sd[bp + "layer_norm2.weight"])
            bw['norm2_beta']  = self._alloc_param(sd[bp + "layer_norm2.bias"])
            bw['fc1_weight']  = self._alloc_param(sd[bp + "mlp.lin1.weight"])  # (3072, 768)
            bw['fc1_bias']    = self._alloc_param(sd[bp + "mlp.lin1.bias"])
            bw['fc2_weight']  = self._alloc_param(sd[bp + "mlp.lin2.weight"])  # (768, 3072)
            bw['fc2_bias']    = self._alloc_param(sd[bp + "mlp.lin2.bias"])
            self.vit_block_weights.append(bw)

        # ==================================================================
        # SAM1 NECK: Conv1×1(768→256) → LN → Conv3×3(256→256) → LN
        # Neck convolutions have bias=False in SAM1.
        # ==================================================================
        neck = vit + "neck."
        # neck.conv1: Conv2d(768, 256, 1×1, bias=False) weight: (256, 768, 1, 1) → (256, 768)
        self.NECK_CONV1_W = self._alloc_param(
            sd[neck + "conv1.weight"].to(torch.bfloat16).reshape(self.NECK_DIM, self.VIT_DIM))
        # neck.layer_norm1: LayerNorm(256)
        self.NECK_LN1_W = self._alloc_param(sd[neck + "layer_norm1.weight"])
        self.NECK_LN1_B = self._alloc_param(sd[neck + "layer_norm1.bias"])
        # neck.conv2: Conv2d(256, 256, 3×3, padding=1, bias=False) weight: (256,256,3,3) → (256, 9*256)
        self.NECK_CONV3_W = self._alloc_param(
            sd[neck + "conv2.weight"].to(torch.bfloat16)
            .permute(0, 2, 3, 1).reshape(self.NECK_DIM, 9 * self.NECK_DIM).contiguous())
        # neck.layer_norm2: LayerNorm(256)
        self.NECK_LN2_W = self._alloc_param(sd[neck + "layer_norm2.weight"])
        self.NECK_LN2_B = self._alloc_param(sd[neck + "layer_norm2.bias"])

        # Zero-pad buffer for conv2d_3x3_dram border padding (NECK_DIM elements)
        self.ZERO_PAD = self._alloc_param(torch.zeros(self.NECK_DIM, dtype=torch.bfloat16))

        # Window attention bias mask: (256, 256) bf16 with -inf at key positions 196..255.
        # Shared across all 300 local-attention batches (25 windows × 12 heads).
        # Prevents the 60 seq-align padding tokens from contributing softmax weight.
        _wa_pad = self.VIT_WINDOW_AREA_PAD   # 256
        _wa     = self.VIT_WINDOW_AREA       # 196
        _win_bias = torch.zeros(_wa_pad, _wa_pad, dtype=torch.bfloat16)
        _win_bias[:, _wa:] = float("-inf")
        self.VIT_WINDOW_ATTN_BIAS = self._alloc_param(_win_bias)

        # ==================================================================
        # PROMPT ENCODER weights (CPU-side encoding; stored for reference)
        # ==================================================================
        pe = "prompt_encoder."
        self.PE_GAUSSIAN_MATRIX = self._alloc_param(
            sd["shared_image_embedding.positional_embedding"].to(torch.bfloat16))  # (2, 128)
        self.PE_POINT_EMBED_FG  = self._alloc_param(
            sd[pe + "point_embed.0.weight"].to(torch.bfloat16))        # (1, 256)
        self.PE_POINT_EMBED_BG  = self._alloc_param(
            sd[pe + "point_embed.1.weight"].to(torch.bfloat16))        # (1, 256)
        self.PE_NOT_A_POINT     = self._alloc_param(
            sd[pe + "not_a_point_embed.weight"].to(torch.bfloat16))    # (1, 256)
        self.PE_NO_MASK         = self._alloc_param(
            sd[pe + "no_mask_embed.weight"].to(torch.bfloat16))        # (1, 256)

        # Image positional encoding: constant 64×64 grid PE stored as param.
        # Used as key_pe in every cross-attention call.
        _pe_mat = sd["shared_image_embedding.positional_embedding"].float()  # (2, 128)
        _gy, _gx = torch.meshgrid(
            torch.arange(64, dtype=torch.float32),
            torch.arange(64, dtype=torch.float32), indexing='ij')
        _grid = torch.stack([(_gx + 0.5) / 64, (_gy + 0.5) / 64], dim=-1).reshape(-1, 2)  # (4096, 2)
        _proj = 2 * math.pi * (_grid @ _pe_mat)  # (4096, 128)
        _pos_src = torch.cat([torch.sin(_proj), torch.cos(_proj)], dim=-1).to(torch.bfloat16)
        self.DEC_POS_SRC = self._alloc_param(_pos_src)  # (4096, 256) constant image PE

        # ==================================================================
        # MASK DECODER weights
        # ==================================================================
        md = "mask_decoder."
        self.DEC_IOU_TOKEN   = self._alloc_param(
            sd[md + "iou_token.weight"].to(torch.bfloat16))            # (1, 256)
        self.DEC_MASK_TOKENS = self._alloc_param(
            sd[md + "mask_tokens.weight"].to(torch.bfloat16))          # (4, 256)

        self.dec_layer_weights = []
        for i in range(self.DEC_LAYERS):
            lp = md + f"transformer.layers.{i}."
            lw = {}
            # Self-attention on tokens (dim=256, head_dim=32)
            for proj in ("q", "k", "v"):
                lw[f'sa_{proj}_w'] = self._alloc_param(sd[lp + f"self_attn.{proj}_proj.weight"].to(torch.bfloat16))
                lw[f'sa_{proj}_b'] = self._alloc_param(sd[lp + f"self_attn.{proj}_proj.bias"].to(torch.bfloat16))
            lw['sa_out_w'] = self._alloc_param(sd[lp + "self_attn.out_proj.weight"].to(torch.bfloat16))
            lw['sa_out_b'] = self._alloc_param(sd[lp + "self_attn.out_proj.bias"].to(torch.bfloat16))
            # Cross-attn token→image (internal_dim=128, head_dim=16)
            for proj in ("q", "k", "v"):
                lw[f'ca_t2i_{proj}_w'] = self._alloc_param(sd[lp + f"cross_attn_token_to_image.{proj}_proj.weight"].to(torch.bfloat16))
                lw[f'ca_t2i_{proj}_b'] = self._alloc_param(sd[lp + f"cross_attn_token_to_image.{proj}_proj.bias"].to(torch.bfloat16))
            lw['ca_t2i_out_w'] = self._alloc_param(sd[lp + "cross_attn_token_to_image.out_proj.weight"].to(torch.bfloat16))
            lw['ca_t2i_out_b'] = self._alloc_param(sd[lp + "cross_attn_token_to_image.out_proj.bias"].to(torch.bfloat16))
            # Cross-attn image→token
            for proj in ("q", "k", "v"):
                lw[f'ca_i2t_{proj}_w'] = self._alloc_param(sd[lp + f"cross_attn_image_to_token.{proj}_proj.weight"].to(torch.bfloat16))
                lw[f'ca_i2t_{proj}_b'] = self._alloc_param(sd[lp + f"cross_attn_image_to_token.{proj}_proj.bias"].to(torch.bfloat16))
            lw['ca_i2t_out_w'] = self._alloc_param(sd[lp + "cross_attn_image_to_token.out_proj.weight"].to(torch.bfloat16))
            lw['ca_i2t_out_b'] = self._alloc_param(sd[lp + "cross_attn_image_to_token.out_proj.bias"].to(torch.bfloat16))
            # Layer norms (4 per layer)
            for n in ("1", "2", "3", "4"):
                lw[f'norm{n}_w'] = self._alloc_param(sd[lp + f"layer_norm{n}.weight"].to(torch.bfloat16))
                lw[f'norm{n}_b'] = self._alloc_param(sd[lp + f"layer_norm{n}.bias"].to(torch.bfloat16))
            # MLP (mlp_dim=2048)
            lw['mlp_lin1_w'] = self._alloc_param(sd[lp + "mlp.lin1.weight"].to(torch.bfloat16))  # (2048, 256)
            lw['mlp_lin1_b'] = self._alloc_param(sd[lp + "mlp.lin1.bias"].to(torch.bfloat16))
            lw['mlp_lin2_w'] = self._alloc_param(sd[lp + "mlp.lin2.weight"].to(torch.bfloat16))  # (256, 2048)
            lw['mlp_lin2_b'] = self._alloc_param(sd[lp + "mlp.lin2.bias"].to(torch.bfloat16))
            self.dec_layer_weights.append(lw)

        # Final cross-attention (after both layers)
        fp = md + "transformer.final_attn_token_to_image."
        self.dec_final_attn = {}
        for proj in ("q", "k", "v"):
            self.dec_final_attn[f'{proj}_w'] = self._alloc_param(sd[fp + f"{proj}_proj.weight"].to(torch.bfloat16))
            self.dec_final_attn[f'{proj}_b'] = self._alloc_param(sd[fp + f"{proj}_proj.bias"].to(torch.bfloat16))
        self.dec_final_attn['out_w'] = self._alloc_param(sd[fp + "out_proj.weight"].to(torch.bfloat16))
        self.dec_final_attn['out_b'] = self._alloc_param(sd[fp + "out_proj.bias"].to(torch.bfloat16))
        nfa = md + "transformer.layer_norm_final_attn."
        self.dec_final_norm = {
            'w': self._alloc_param(sd[nfa + "weight"].to(torch.bfloat16)),
            'b': self._alloc_param(sd[nfa + "bias"].to(torch.bfloat16)),
        }

        params_used = self.get_params_dram_usage()
        _original_print(f"  Params loaded: {params_used / 1024**2:.1f} MB ({params_used / 1024**3:.2f} GB)")
        del sd  # free host memory

    # ------------------------------------------------------------------
    # Tensor initialization (intermediate activation buffers)
    # ------------------------------------------------------------------

    def tensor_init(self) -> None:
        """Allocate DRAM buffers for all intermediate activations and scratch."""
        bpe = 2

        GA     = self.GRID_AREA      # 4096
        GA_PAD = self.GRID_AREA_PAD  # 4900 (70×70 padded grid for window partition)
        VD     = self.VIT_DIM        # 768
        VD_QK  = self.VIT_QK_DIM_PAD # 768 (no head_dim padding)
        hd_pad = self.VIT_HEAD_DIM_PAD  # 64

        self.IMAGE_DRAM      = self._alloc_tensor(self.NUM_CHANNELS * self.IMAGE_SIZE * self.IMAGE_SIZE)
        self.VIT_PATCH_OUT   = self._alloc_tensor(GA * self.PATCH_K_PAD)  # (4096, 768) patches
        self.VIT_LAYER_IN    = self._alloc_tensor(GA * VD)                # (4096, 768)

        # Pre-attention LayerNorm output and spatial-pad buffer for local blocks
        self.VIT_LN_OUT      = self._alloc_tensor(GA * VD)       # (4096, 768) — also used for unpad
        self.VIT_PADDED      = self._alloc_tensor(GA_PAD * VD)   # (4900, 768) padded 70×70 grid
        self.VIT_WINDOWED    = self._alloc_tensor(GA_PAD * VD)   # (4900, 768) window-partitioned

        # Q, K, V projections — sized for padded grid (local: 4900, global: 4096 ≤ 4900)
        self.VIT_Q           = self._alloc_tensor(GA_PAD * VD_QK)  # (4900, 768)
        self.VIT_K           = self._alloc_tensor(GA_PAD * VD_QK)
        self.VIT_V           = self._alloc_tensor(GA_PAD * VD_QK)

        # Multi-head layout: (total_batches, seq_len_pad, head_dim=64)
        # Local: 25 windows × 12 heads = 300 batches, seq_pad=256 → 300×256×64 = 4,915,200
        # Global: 12 heads, seq=4096 → 12×4096×64 = 3,145,728
        total_heads_local = self.VIT_NUM_WINDOWS * self.VIT_HEADS  # 25 × 12 = 300
        wa_pad       = self.VIT_WINDOW_AREA_PAD                    # 256 (64-aligned for flash attn)
        local_total_pad  = total_heads_local * wa_pad              # 300 × 256 = 76,800
        global_total = self.VIT_HEADS * GA                         # 12  × 4096 = 49,152
        heads_elems  = max(local_total_pad, global_total) * hd_pad # 76,800 × 64 = 4,915,200
        # Compact intermediate (300 × 196 × 64) for scatter/gather around flash attention
        compact_elems = total_heads_local * self.VIT_WINDOW_AREA * hd_pad  # 3,763,200
        self.VIT_COMPACT_HEADS = self._alloc_tensor(compact_elems)
        self.VIT_Q_HEADS     = self._alloc_tensor(heads_elems)
        self.VIT_K_HEADS     = self._alloc_tensor(heads_elems)
        self.VIT_V_HEADS     = self._alloc_tensor(heads_elems)
        self.VIT_ATTN_OUT    = self._alloc_tensor(heads_elems)
        # Scratch: local — 300 × (64×256 + 256²) = 300 × 81,920 = 24,576,000 elements
        local_scratch = total_heads_local * (hd_pad * wa_pad + wa_pad ** 2)
        self.VIT_ATTN_SCRATCH = self._alloc_tensor(local_scratch)
        self.VIT_ATTN_MERGED = self._alloc_tensor(GA_PAD * VD_QK)  # (4900, 768) merged heads
        self.VIT_OUT_PROJ    = self._alloc_tensor(GA_PAD * VD)      # (4900, 768) output proj
        self.VIT_RESIDUAL    = self._alloc_tensor(GA * VD)          # (4096, 768)
        self.VIT_MLP_MID     = self._alloc_tensor(GA * self.VIT_MLP_HIDDEN)  # (4096, 3072)
        self.VIT_MLP_OUT     = self._alloc_tensor(GA * VD)          # (4096, 768)

        # Neck buffers
        ND = self.NECK_DIM  # 256
        self.VIT_NECK_OUT    = self._alloc_tensor(GA * ND)           # (4096, 256) after 1×1 conv + LN
        self.NECK_IM2COL     = self._alloc_tensor(self.GRID_SIZE * 9 * ND)  # (64, 9×256) = 147,456
        self.NECK_OUT        = self._alloc_tensor(GA * ND)           # (4096, 256) final neck output

        # Mask decoder buffers
        DD  = self.DEC_DIM         # 256
        NT  = self.DEC_NUM_TOKENS  # 7
        ID  = self.DEC_INTERNAL_DIM  # 128 (cross-attn internal dim)
        self.DEC_TOKENS      = self._alloc_tensor(NT * DD)   # (7, 256) prompt + output tokens
        self.DEC_SRC         = self._alloc_tensor(GA * DD)   # (4096, 256) image src
        self.DEC_TOKENS_NORM = self._alloc_tensor(NT * DD)   # (7, 256) LN output scratch
        self.DEC_Q           = self._alloc_tensor(NT * DD)   # (7, 256) self-attn Q
        self.DEC_K           = self._alloc_tensor(NT * DD)   # (7, 256) self-attn K
        self.DEC_V           = self._alloc_tensor(NT * DD)   # (7, 256) self-attn V

        tensor_used = self.get_tensor_dram_usage()
        _original_print(f"  Tensors allocated: {tensor_used / 1024**2:.1f} MB")

    # ------------------------------------------------------------------
    # Program compilation
    # ------------------------------------------------------------------

    def compile_full_fused(self) -> int:
        """Compile SAM1 ViT-B forward pass (ViT backbone + neck) as instruction stream.

        Returns program DRAM address.
        """
        global _SILENT_MODE
        _SILENT_MODE = True

        pad = self.pad_dim
        bpe = 2

        self.start_capture()

        try:
            self._compile_phases(pad, bpe)
        except (_CheckpointStop, AssertionError) as e:
            _original_print(f"  Compile stopped at checkpoint: {e}")
        self._finalize_program()
        _SILENT_MODE = False
        return self._last_prog_addr

    def _compile_phases(self, pad, bpe):
        """All compile phases live here."""

        VD    = self.VIT_DIM         # 768
        VD_QK = self.VIT_QK_DIM_PAD  # 768 (no head_dim padding)
        hd    = self.VIT_HEAD_DIM_PAD # 64
        bpe   = 2

        # ==============================================================
        # PHASE 1: PATCH EMBEDDING
        # 16×16 patches, PATCH_K_RAW=768 already 64-aligned.
        # matmul: (4096, 768) @ (768, 768)^T + bias → (4096, 768)
        # Patches are pre-extracted to VIT_PATCH_OUT by extract_patches_to_dram().
        # ==============================================================
        self.matmat_mul_core(
            M=self.GRID_AREA, K=self.PATCH_K_PAD, N=VD,
            A_DRAM_ADDR=self.VIT_PATCH_OUT,
            B_DRAM_ADDR=self.PATCH_EMBED_WEIGHT,
            OUTPUT_DRAM_ADDR=self.VIT_LAYER_IN,
            C_DRAM_ADDR=self.PATCH_EMBED_BIAS,
            bias_mode="broadcast_N",
        )

        # Add absolute positional embedding (already at 64×64 resolution)
        eltwise_add_dram(self, self.VIT_LAYER_IN, self.POS_EMBED,
                         self.VIT_LAYER_IN, self.GRID_AREA * VD)

        # SAM1 has no LN_pre.

        # ==============================================================
        # PHASE 2: ViT BLOCKS (12 blocks)
        # Local blocks (0-1, 3-4, 6-7, 9-10): window=14, 25 windows
        # Global blocks (2, 5, 8, 11):         full seq=4096
        # No RoPE — SAM1 uses rel_pos (not yet implemented on-chip).
        # ==============================================================
        _pre_vit_inst = self.capture_count
        for blk_idx in range(self.VIT_DEPTH):
            _blk_start = self.capture_count
            bw = self.vit_block_weights[blk_idx]
            is_global = blk_idx in self.VIT_GLOBAL_BLOCKS

            if is_global:
                seq_len      = self.GRID_AREA        # 4096
                num_windows  = 1
                total_batches = self.VIT_HEADS        # 12
                M_flat       = self.GRID_AREA         # 4096
            else:
                seq_len      = self.VIT_WINDOW_AREA   # 196
                num_windows  = self.VIT_NUM_WINDOWS   # 25
                total_batches = num_windows * self.VIT_HEADS  # 300
                M_flat       = self.GRID_AREA_PAD     # 4900 (padded 70×70)

            # --- Pre-attention LayerNorm (always on unpadded GRID_AREA tokens) ---
            self.layer_norm_core_dram(
                M=self.GRID_AREA, N=VD,
                A_DRAM_ADDR=self.VIT_LAYER_IN,
                OUTPUT_DRAM_ADDR=self.VIT_LN_OUT,
                GAMMA_DRAM_ADDR=bw['norm1_gamma'],
                BETA_DRAM_ADDR=bw['norm1_beta'],
            )

            if blk_idx == 0 and getattr(self, '_ln1_checkpoint', False):
                raise _CheckpointStop("after block 0 pre-attn LayerNorm")

            # --- Window partition (local blocks only) ---
            # Pad 64×64 → 70×70 (next multiple of 14), then partition into 25 windows.
            if not is_global:
                pad_feature_map_dram(self,
                    INPUT_DRAM_ADDR=self.VIT_LN_OUT,
                    OUTPUT_DRAM_ADDR=self.VIT_PADDED,
                    H=self.GRID_SIZE, W=self.GRID_SIZE,
                    H_PAD=self.GRID_SIZE_PAD, W_PAD=self.GRID_SIZE_PAD,
                    C=VD)
                window_partition_dram(self,
                    INPUT_DRAM_ADDR=self.VIT_PADDED,
                    OUTPUT_DRAM_ADDR=self.VIT_WINDOWED,
                    H=self.GRID_SIZE_PAD, W=self.GRID_SIZE_PAD, C=VD,
                    window_size=self.VIT_WINDOW_SIZE)
                attn_input = self.VIT_WINDOWED
            else:
                attn_input = self.VIT_LN_OUT

            # --- Q, K, V projections ---
            # (M_flat, 768) @ (768, 768)^T + bias → (M_flat, 768)
            for w_key, b_key, out_addr in [
                ('q_weight', 'q_bias', self.VIT_Q),
                ('k_weight', 'k_bias', self.VIT_K),
                ('v_weight', 'v_bias', self.VIT_V),
            ]:
                self.matmat_mul_core(
                    M=M_flat, K=VD, N=VD_QK,
                    A_DRAM_ADDR=attn_input,

                    B_DRAM_ADDR=bw[w_key],
                    OUTPUT_DRAM_ADDR=out_addr,
                    C_DRAM_ADDR=bw[b_key],
                    bias_mode="broadcast_N",
                )


            # --- Multi-head reshape: (window, seq, heads*64) → (window*heads, seq_pad, 64) ---
            # Local: seq=196 is not 64-aligned; SRAM addresses must be 128-byte (64-elem) aligned.
            # Pad to seq_pad=256. Steps per Q/K/V:
            #   1. permute each window (196,12,64)→(12,196,64) into compact temp VIT_COMPACT_HEADS
            #   2. zero-fill padded heads buffer (total_batches × 256 × 64)
            #   3. scatter compact[b](196×64) → padded[b][0..196×64) for b in range(total_batches)
            if not is_global:
                wa     = seq_len         # 196
                wa_pad = self.VIT_WINDOW_AREA_PAD  # 256
                for qkv_src, qkv_dst in [(self.VIT_Q, self.VIT_Q_HEADS),
                                          (self.VIT_K, self.VIT_K_HEADS),
                                          (self.VIT_V, self.VIT_V_HEADS)]:
                    # Step 1: permute → compact
                    for w in range(num_windows):
                        self.bf16_permute_core(
                            dim_0=wa, dim_1=self.VIT_HEADS, dim_2=hd,
                            INPUT_DRAM_ADDR=qkv_src + w * wa * VD_QK * bpe,
                            OUTPUT_DRAM_ADDR=self.VIT_COMPACT_HEADS + w * self.VIT_HEADS * wa * hd * bpe,
                        )
                    # Step 2: zero-fill padded dest
                    dram_zero_fill(self, qkv_dst, total_batches * wa_pad * hd)
                    # Step 3: scatter compact → padded (insert real rows at padded stride)
                    for b in range(total_batches):
                        dram_copy(self,
                            self.VIT_COMPACT_HEADS + b * wa * hd * bpe,
                            qkv_dst + b * wa_pad * hd * bpe,
                            wa * hd)
            else:
                # Global: seq=4096, 4096 % 64 = 0 — no padding needed
                for qkv_src, qkv_dst in [(self.VIT_Q, self.VIT_Q_HEADS),
                                          (self.VIT_K, self.VIT_K_HEADS),
                                          (self.VIT_V, self.VIT_V_HEADS)]:
                    for w in range(num_windows):
                        src = qkv_src + w * seq_len * VD_QK * bpe
                        dst = qkv_dst + w * self.VIT_HEADS * seq_len * hd * bpe
                        self.bf16_permute_core(
                            dim_0=seq_len, dim_1=self.VIT_HEADS, dim_2=hd,
                            INPUT_DRAM_ADDR=src,
                            OUTPUT_DRAM_ADDR=dst,
                        )

            # --- No RoPE (SAM1 uses rel_pos — TODO: implement decomposed rel_pos on-chip) ---

            # --- Flash attention ---
            if is_global:
                # Global seq=4096: 4096 % 64 = 0, no padding needed
                flash_attention_global_tiled(self,
                    num_heads=self.VIT_HEADS,
                    head_dim=hd,
                    seq_len=seq_len,
                    Q_DRAM_ADDR=self.VIT_Q_HEADS,
                    K_DRAM_ADDR=self.VIT_K_HEADS,
                    V_DRAM_ADDR=self.VIT_V_HEADS,
                    OUTPUT_DRAM_ADDR=self.VIT_ATTN_OUT,
                    SCRATCH_DRAM_ADDR=self.VIT_ATTN_SCRATCH,
                )
            else:
                # Local: run attention on seq_pad=256 (real 196 tokens + 60 zero-padded).
                # Shared bias mask sets key positions 196..255 to -inf so seq-align
                # padding tokens get zero softmax weight.
                #
                # ACCURACY NOTE (validated 2026-04-10):
                # SAM1 ViT-B uses window_size=14 → window_area=196. Hardware flash_attention_core
                # requires SRAM addresses to be 128-byte (64-element) aligned, so seq_len must be
                # a multiple of 64. The nearest multiple is 256 (196 padded with 60 zeros).
                # Running flash_attention on seq=256 instead of seq=196 introduces numerical
                # differences even with the -inf bias mask, because BF16 softmax accumulation
                # over 256 terms differs from accumulation over 196 terms.
                #
                # Measured accuracy cascade (block 0):
                #   attn_merged (post flash_attn):  SNR=39.6 dB, within1ULP=99.9%, maxErr=0.031
                #   out_proj + win_reverse + add:   SNR=43.1 dB, within1ULP=98.0%, maxErr=0.094
                #   LN2 output:                     SNR=36.9 dB, within1ULP=77.3%, maxErr=0.375
                #     ↑ LN2 amplifies maxErr because a single element error shifts the mean for
                #       all 768 elements in that row; large gamma weights further amplify.
                #   fc1+GELU:                       SNR=35.4 dB, within1ULP=86.4%, maxErr=0.129
                #   fc2 (MLP out):                  SNR=35.8 dB, within1ULP=38.3%, maxErr=0.563
                #   block 0 final output:           SNR=36.4 dB, within1ULP=38.9%
                #   all 12 blocks final output:     SNR=33.0 dB, cos=0.9998, maxErr=1.875
                #
                # This is the fundamental accuracy limit of SAM1 ViT-B on this hardware.
                # SAM3.1 avoids this entirely: window_size=24 → area=576=9×64, already aligned.
                flash_attention_batched(self,
                    num_batches=total_batches,  # 300
                    head_dim=hd,
                    seq_len=wa_pad,             # 256 (64-aligned)
                    Q_DRAM_ADDR=self.VIT_Q_HEADS,
                    K_DRAM_ADDR=self.VIT_K_HEADS,
                    V_DRAM_ADDR=self.VIT_V_HEADS,
                    OUTPUT_DRAM_ADDR=self.VIT_ATTN_OUT,
                    SCRATCH_DRAM_ADDR=self.VIT_ATTN_SCRATCH,
                    BIAS_DRAM_ADDR=self.VIT_WINDOW_ATTN_BIAS,
                    bias_shared=True,
                )

            # --- Merge heads: (window*heads, seq, 64) → (window, seq, heads*64) ---
            if not is_global:
                # Gather valid rows: (total_batches, 256, 64) → compact (total_batches, 196, 64)
                for b in range(total_batches):
                    dram_copy(self,
                        self.VIT_ATTN_OUT + b * wa_pad * hd * bpe,
                        self.VIT_COMPACT_HEADS + b * wa * hd * bpe,
                        wa * hd)
                # Permute compact (12, 196, 64) → (196, 12, 64) per window → VIT_ATTN_MERGED
                for w in range(num_windows):
                    self.bf16_permute_core(
                        dim_0=self.VIT_HEADS, dim_1=wa, dim_2=hd,
                        INPUT_DRAM_ADDR=self.VIT_COMPACT_HEADS + w * self.VIT_HEADS * wa * hd * bpe,
                        OUTPUT_DRAM_ADDR=self.VIT_ATTN_MERGED + w * wa * VD_QK * bpe,
                    )
            else:
                for w in range(num_windows):
                    src = self.VIT_ATTN_OUT + w * self.VIT_HEADS * seq_len * hd * bpe
                    dst = self.VIT_ATTN_MERGED + w * seq_len * VD_QK * bpe
                    self.bf16_permute_core(
                        dim_0=self.VIT_HEADS, dim_1=seq_len, dim_2=hd,
                        INPUT_DRAM_ADDR=src,
                        OUTPUT_DRAM_ADDR=dst,
                    )


            # --- Output projection: (M_flat, 768) @ (768, 768)^T + bias → (M_flat, 768) ---
            self.matmat_mul_core(
                M=M_flat, K=VD_QK, N=VD,
                A_DRAM_ADDR=self.VIT_ATTN_MERGED,
                B_DRAM_ADDR=bw['proj_weight'],
                OUTPUT_DRAM_ADDR=self.VIT_OUT_PROJ,
                C_DRAM_ADDR=bw['proj_bias'],
                bias_mode="broadcast_N",
            )


            # --- Window unpartition + unpad (local blocks only) ---
            if not is_global:
                window_reverse_dram(self,
                    INPUT_DRAM_ADDR=self.VIT_OUT_PROJ,
                    OUTPUT_DRAM_ADDR=self.VIT_ATTN_MERGED,
                    H=self.GRID_SIZE_PAD, W=self.GRID_SIZE_PAD, C=VD,
                    window_size=self.VIT_WINDOW_SIZE)
                # Unpad: extract top-left 64×64 from 70×70 → VIT_LN_OUT (reused as temp)
                unpad_feature_map_dram(self,
                    INPUT_DRAM_ADDR=self.VIT_ATTN_MERGED,
                    OUTPUT_DRAM_ADDR=self.VIT_LN_OUT,
                    H=self.GRID_SIZE, W=self.GRID_SIZE,
                    W_PAD=self.GRID_SIZE_PAD, C=VD)
                attn_result = self.VIT_LN_OUT
            else:
                attn_result = self.VIT_OUT_PROJ


            # --- Residual add 1 ---
            eltwise_add_dram(self, self.VIT_LAYER_IN, attn_result,
                             self.VIT_RESIDUAL, self.GRID_AREA * VD)


            # --- Pre-MLP LayerNorm ---
            self.layer_norm_core_dram(
                M=self.GRID_AREA, N=VD,
                A_DRAM_ADDR=self.VIT_RESIDUAL,
                OUTPUT_DRAM_ADDR=self.VIT_LN_OUT,
                GAMMA_DRAM_ADDR=bw['norm2_gamma'],
                BETA_DRAM_ADDR=bw['norm2_beta'],
            )

            if blk_idx == 0 and getattr(self, '_ln2_checkpoint', False):
                raise _CheckpointStop("after block 0 pre-MLP LayerNorm")

            # --- MLP: fc1 (768→3072, GELU) + fc2 (3072→768) ---
            self.matmat_mul_core(
                M=self.GRID_AREA, K=VD, N=self.VIT_MLP_HIDDEN,
                A_DRAM_ADDR=self.VIT_LN_OUT,
                B_DRAM_ADDR=bw['fc1_weight'],
                OUTPUT_DRAM_ADDR=self.VIT_MLP_MID,
                C_DRAM_ADDR=bw['fc1_bias'],
                bias_mode="broadcast_N",
                gelu_enable=True,
            )

            if blk_idx == 0 and getattr(self, '_fc1_checkpoint', False):
                raise _CheckpointStop("after block 0 fc1+GELU")
            self.matmat_mul_core(
                M=self.GRID_AREA, K=self.VIT_MLP_HIDDEN, N=VD,
                A_DRAM_ADDR=self.VIT_MLP_MID,
                B_DRAM_ADDR=bw['fc2_weight'],
                OUTPUT_DRAM_ADDR=self.VIT_MLP_OUT,
                C_DRAM_ADDR=bw['fc2_bias'],
                bias_mode="broadcast_N",
            )

            if blk_idx == 0 and getattr(self, '_mlp_checkpoint', False):
                raise _CheckpointStop("after block 0 MLP (pre residual-2)")

            # --- Residual add 2 ---
            eltwise_add_dram(self, self.VIT_RESIDUAL, self.VIT_MLP_OUT,
                             self.VIT_LAYER_IN, self.GRID_AREA * VD)

            # Per-block instruction count
            _blk_inst  = self.capture_count - _blk_start
            _total_inst = self.capture_count - _pre_vit_inst
            _type = "GLOBAL" if is_global else "local"
            _original_print(
                f"    Block {blk_idx:2d} ({_type}): {_blk_inst:>10,} inst  |  "
                f"cumulative: {_total_inst:>12,} inst  ({_total_inst * 32 / 1024**2:.0f} MB)")

            if blk_idx == getattr(self, '_vit_checkpoint', self.VIT_DEPTH - 1):
                raise _CheckpointStop(f"after ViT blocks 0-{blk_idx}")

        # ViT output: VIT_LAYER_IN contains (4096, 768) = (64, 64, 768)

        # ==============================================================
        # PHASE 3: NECK
        # Conv 1×1 (768→256) → LN → Conv 3×3 (256→256) → LN
        # ==============================================================
        ND = self.NECK_DIM  # 256
        GS = self.GRID_SIZE  # 64

        # neck.0: Conv 1×1 — matmul: (4096, 768) @ (256, 768)^T → (4096, 256)
        self.matmat_mul_core(
            M=self.GRID_AREA, K=VD, N=ND,
            A_DRAM_ADDR=self.VIT_LAYER_IN,
            B_DRAM_ADDR=self.NECK_CONV1_W,
            OUTPUT_DRAM_ADDR=self.VIT_NECK_OUT,
        )

        # neck.1: LayerNorm(256) on (4096, 256)
        self.layer_norm_core_dram(
            M=self.GRID_AREA, N=ND,
            A_DRAM_ADDR=self.VIT_NECK_OUT,
            OUTPUT_DRAM_ADDR=self.VIT_NECK_OUT,
            GAMMA_DRAM_ADDR=self.NECK_LN1_W,
            BETA_DRAM_ADDR=self.NECK_LN1_B,
        )

        # neck.2: Conv 3×3 (256→256, padding=1) on 64×64 feature map
        conv2d_3x3_dram(self,
            INPUT_DRAM_ADDR=self.VIT_NECK_OUT,
            OUTPUT_DRAM_ADDR=self.NECK_OUT,
            IM2COL_DRAM_ADDR=self.NECK_IM2COL,
            WEIGHT_DRAM_ADDR=self.NECK_CONV3_W,
            BIAS_DRAM_ADDR=None,             # bias=False in SAM1 neck
            H=GS, W=GS, C_in=ND, C_out=ND,
            ZERO_PAD_DRAM_ADDR=self.ZERO_PAD,
        )

        # neck.3: LayerNorm(256) on (4096, 256)
        self.layer_norm_core_dram(
            M=self.GRID_AREA, N=ND,
            A_DRAM_ADDR=self.NECK_OUT,
            OUTPUT_DRAM_ADDR=self.NECK_OUT,
            GAMMA_DRAM_ADDR=self.NECK_LN2_W,
            BETA_DRAM_ADDR=self.NECK_LN2_B,
        )

        # NECK_OUT now contains the image embedding: (4096, 256) = (64, 64, 256)

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
    def preprocess_image(image_path: str, image_size: int = 1024) -> torch.Tensor:
        """Load and preprocess image using SAM1 normalization. Returns (1, 3, H, W) bf16."""
        transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(image_size, image_size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        pil_image = Image.open(image_path).convert("RGB")
        image = v2.functional.to_image(pil_image)
        return transform(image).unsqueeze(0).to(torch.bfloat16)

    # ------------------------------------------------------------------
    # Patch extraction (host-side, feeds VIT_PATCH_OUT before HW execution)
    # ------------------------------------------------------------------

    def extract_patches_to_dram(self, image_chw: torch.Tensor) -> None:
        """Extract 16×16 patches from CHW image, write (4096, 768) to VIT_PATCH_OUT.

        image_chw: (3, 1024, 1024) bf16.
        PATCH_K_RAW = 768 is already 64-aligned — no padding needed.
        """
        C, H, W = image_chw.shape
        P = self.PATCH_SIZE  # 16
        gH, gW = H // P, W // P  # 64, 64
        K_raw = C * P * P  # 768
        patches = image_chw.reshape(C, gH, P, gW, P).permute(1, 3, 0, 2, 4).reshape(gH * gW, K_raw)
        self.dma_to_accelerator_memory(self.VIT_PATCH_OUT, patches.contiguous())

    # ------------------------------------------------------------------
    # Run HW up to current checkpoint
    # ------------------------------------------------------------------

    def run_hw(self, image_path: str, program_addr: int, timeout: float = 600.0) -> None:
        """Preprocess image, DMA inputs, execute compiled program."""
        image = self.preprocess_image(image_path, self.IMAGE_SIZE)
        image_chw = image.squeeze(0)  # (3, 1024, 1024)
        self.extract_patches_to_dram(image_chw)
        self.start_execute_from_dram(program_addr)
        self.wait_queue(timeout)

    # ------------------------------------------------------------------
    # CPU reference for current checkpoint
    # ------------------------------------------------------------------

    def cpu_reference_patch_embed(self, image_path: str) -> torch.Tensor:
        """Run patch_embed + pos_embed on CPU. Returns (4096, 768) bf16. (SAM1 has no LN_pre.)"""
        import torch.nn.functional as F

        image = self.preprocess_image(image_path, self.IMAGE_SIZE)  # (1, 3, 1024, 1024)

        ckpt_path = _ensure_checkpoint(self.script_dir, self.cfg)
        sd = _load_sam1_state_dict(ckpt_path)

        # Patch embed via conv2d (16×16 patches, bias present)
        patch_w = sd["vision_encoder.patch_embed.projection.weight"].float()  # (768, 3, 16, 16)
        patch_b = sd["vision_encoder.patch_embed.projection.bias"].float()    # (768,)
        patches = F.conv2d(image.float(), patch_w, bias=patch_b,
                           stride=self.PATCH_SIZE)  # (1, 768, 64, 64)
        patches = patches.permute(0, 2, 3, 1).reshape(self.GRID_AREA, self.VIT_DIM)  # (4096, 768)

        # Add pos embed (already at 64×64 resolution, no tiling needed)
        pos_embed = sd["vision_encoder.pos_embed"].reshape(self.GRID_AREA, self.VIT_DIM).float()
        patches = patches + pos_embed

        del sd
        return patches.to(torch.bfloat16)

    def cpu_reference_block0_qkv(self, image_path: str) -> torch.Tensor:
        """Run through block 0 QKV projection on CPU. Returns (4900, 2304) bf16.

        patch_embed + pos + block0.layer_norm1 + pad_to_70 + window_partition + QKV matmul.
        """
        import torch.nn as nn
        import torch.nn.functional as F

        x = self.cpu_reference_patch_embed(image_path).float()  # (4096, 768)

        ckpt_path = _ensure_checkpoint(self.script_dir, self.cfg)
        sd = _load_sam1_state_dict(ckpt_path)
        prefix = "vision_encoder.layers.0."

        # block0.layer_norm1
        ln = nn.LayerNorm(self.VIT_DIM, eps=1e-6)
        ln.weight.data = sd[prefix + "layer_norm1.weight"].float()
        ln.bias.data = sd[prefix + "layer_norm1.bias"].float()
        normed = ln(x)  # (4096, 768)

        # Pad 64×64 → 70×70 (next multiple of window_size=14)
        VD = self.VIT_DIM
        ws = self.VIT_WINDOW_SIZE   # 14
        nw = self.VIT_NUM_WINDOWS   # 25 = 5×5
        nw_side = 5
        spatial = normed.reshape(self.GRID_SIZE, self.GRID_SIZE, VD)  # (64, 64, 768)
        padded = F.pad(spatial.permute(2, 0, 1).unsqueeze(0),
                       (0, self.GRID_SIZE_PAD - self.GRID_SIZE,
                        0, self.GRID_SIZE_PAD - self.GRID_SIZE)).squeeze(0).permute(1, 2, 0)
        # (70, 70, C) → (5, 14, 5, 14, C) → (25, 196, C)
        windowed = (padded
                    .reshape(nw_side, ws, nw_side, ws, VD)
                    .permute(0, 2, 1, 3, 4)
                    .reshape(nw * ws * ws, VD))  # (4900, 768)

        # QKV projection: Linear(768, 2304)
        qkv_w = sd[prefix + "attn.qkv.weight"].float()  # (2304, 768)
        qkv_b = sd[prefix + "attn.qkv.bias"].float()    # (2304,)
        qkv = F.linear(windowed, qkv_w, qkv_b)  # (4900, 2304)

        del sd
        return qkv.to(torch.bfloat16)

    def cpu_reference_block0_full(self, image_path: str) -> torch.Tensor:
        """Run full ViT block 0 on CPU (no rel_pos). Returns (4096, 768) bf16."""
        import torch.nn as nn
        import torch.nn.functional as F

        x = self.cpu_reference_patch_embed(image_path).float()  # (4096, 768)

        ckpt_path = _ensure_checkpoint(self.script_dir, self.cfg)
        sd = _load_sam1_state_dict(ckpt_path)
        prefix = "vision_encoder.layers.0."
        VD = self.VIT_DIM
        ws = self.VIT_WINDOW_SIZE   # 14
        nw_side = 5
        GA = self.GRID_AREA         # 4096

        # --- layer_norm1 ---
        ln1 = nn.LayerNorm(VD, eps=1e-6)
        ln1.weight.data = sd[prefix + "layer_norm1.weight"].float()
        ln1.bias.data = sd[prefix + "layer_norm1.bias"].float()
        normed = ln1(x)

        # --- pad 64→70, window partition → (25, 196, 768) ---
        spatial = normed.reshape(self.GRID_SIZE, self.GRID_SIZE, VD)
        padded = F.pad(spatial.permute(2, 0, 1).unsqueeze(0),
                       (0, self.GRID_SIZE_PAD - self.GRID_SIZE,
                        0, self.GRID_SIZE_PAD - self.GRID_SIZE)).squeeze(0).permute(1, 2, 0)
        windowed = (padded
                    .reshape(nw_side, ws, nw_side, ws, VD)
                    .permute(0, 2, 1, 3, 4)
                    .reshape(self.VIT_NUM_WINDOWS, ws * ws, VD))  # (25, 196, 768)

        # --- Q, K, V projections ---
        qkv_w = sd[prefix + "attn.qkv.weight"].float()  # (2304, 768)
        qkv_b = sd[prefix + "attn.qkv.bias"].float()
        q_w, k_w, v_w = qkv_w.chunk(3, dim=0)
        q_b, k_b, v_b = qkv_b.chunk(3, dim=0)
        Q = F.linear(windowed, q_w, q_b)  # (25, 196, 768)
        K = F.linear(windowed, k_w, k_b)
        V = F.linear(windowed, v_w, v_b)

        # --- multi-head reshape → (25*12, 196, 64) ---
        nh, hd = self.VIT_HEADS, self.VIT_HEAD_DIM
        nw = self.VIT_NUM_WINDOWS
        Q = Q.reshape(nw, ws*ws, nh, hd).permute(0, 2, 1, 3).reshape(-1, ws*ws, hd)
        K = K.reshape(nw, ws*ws, nh, hd).permute(0, 2, 1, 3).reshape(-1, ws*ws, hd)
        V = V.reshape(nw, ws*ws, nh, hd).permute(0, 2, 1, 3).reshape(-1, ws*ws, hd)

        # --- attention (no rel_pos — TODO) ---
        attn_out = F.scaled_dot_product_attention(Q, K, V)  # (300, 196, 64)

        # --- merge heads → (25, 196, 768) ---
        attn_out = (attn_out.reshape(nw, nh, ws*ws, hd)
                    .permute(0, 2, 1, 3)
                    .reshape(nw, ws*ws, VD))

        # --- output projection ---
        proj_w = sd[prefix + "attn.proj.weight"].float()
        proj_b = sd[prefix + "attn.proj.bias"].float()
        proj_out = F.linear(attn_out, proj_w, proj_b)  # (25, 196, 768)

        # --- window reverse + unpad 70→64 → (4096, 768) ---
        proj_spatial = (proj_out.reshape(nw_side, nw_side, ws, ws, VD)
                        .permute(0, 2, 1, 3, 4)
                        .reshape(self.GRID_SIZE_PAD, self.GRID_SIZE_PAD, VD))
        proj_flat = proj_spatial[:self.GRID_SIZE, :self.GRID_SIZE, :].reshape(GA, VD)

        # --- residual 1 ---
        x = x + proj_flat

        # --- layer_norm2 + MLP + residual 2 ---
        ln2 = nn.LayerNorm(VD, eps=1e-6)
        ln2.weight.data = sd[prefix + "layer_norm2.weight"].float()
        ln2.bias.data = sd[prefix + "layer_norm2.bias"].float()
        normed2 = ln2(x)

        lin1_w = sd[prefix + "mlp.lin1.weight"].float()
        lin1_b = sd[prefix + "mlp.lin1.bias"].float()
        lin2_w = sd[prefix + "mlp.lin2.weight"].float()
        lin2_b = sd[prefix + "mlp.lin2.bias"].float()
        mlp_out = F.linear(F.gelu(F.linear(normed2, lin1_w, lin1_b)), lin2_w, lin2_b)

        out = x + mlp_out

        del sd
        return out.to(torch.bfloat16)

    def cpu_reference_vit_blocks(self, image_path: str, num_blocks: int = 12) -> torch.Tensor:
        """Run num_blocks ViT blocks on CPU in bf16 to match HW precision.

        All computation uses bf16 tensors to mirror HW flash_attention_core:
          - Q scaled BEFORE matmul (broadcast_mul), not after
          - bf16 matmul for Q@K^T and softmax@V
          - softmax in float32 internally (matches HW flash_attention_core)
        LayerNorm upcasts to float32 internally (same as HW layer_norm_core).
        No RoPE (SAM1 uses rel_pos, currently skipped — same as HW).
        """
        import torch.nn as nn
        import torch.nn.functional as F

        x = self.cpu_reference_patch_embed(image_path).to(torch.bfloat16)
        torch.set_grad_enabled(False)

        ckpt_path = _ensure_checkpoint(self.script_dir, self.cfg)
        sd = _load_sam1_state_dict(ckpt_path)
        VD = self.VIT_DIM
        ws = self.VIT_WINDOW_SIZE   # 14
        GA = self.GRID_AREA         # 4096
        nw_side = 5
        nw_local = self.VIT_NUM_WINDOWS  # 25

        for blk in range(num_blocks):
            prefix = f"vision_encoder.layers.{blk}."
            is_global = blk in self.VIT_GLOBAL_BLOCKS

            # layer_norm1 (float32 internal, bf16 output — matches HW layer_norm_core)
            ln1 = nn.LayerNorm(VD, eps=1e-6)
            ln1.weight.data = sd[prefix + "layer_norm1.weight"].float()
            ln1.bias.data = sd[prefix + "layer_norm1.bias"].float()
            normed = ln1(x.float()).to(torch.bfloat16)

            if not is_global:
                # pad 64→70, window partition → (25, 196, 768)
                spatial = normed.reshape(self.GRID_SIZE, self.GRID_SIZE, VD)
                padded = F.pad(spatial.permute(2, 0, 1).unsqueeze(0),
                               (0, self.GRID_SIZE_PAD - self.GRID_SIZE,
                                0, self.GRID_SIZE_PAD - self.GRID_SIZE)
                               ).squeeze(0).permute(1, 2, 0)
                windowed = (padded
                            .reshape(nw_side, ws, nw_side, ws, VD)
                            .permute(0, 2, 1, 3, 4)
                            .reshape(nw_local, ws * ws, VD))
                num_win = nw_local
                seq = ws * ws
            else:
                windowed = normed.unsqueeze(0)  # (1, 4096, 768)
                num_win = 1
                seq = GA

            # Q, K, V projections (no weight permutation — no RoPE)
            nh = self.VIT_HEADS
            hd = self.VIT_HEAD_DIM
            qkv_w = sd[prefix + "attn.qkv.weight"].to(torch.bfloat16)
            qkv_b = sd[prefix + "attn.qkv.bias"].to(torch.bfloat16)
            Q = F.linear(windowed, qkv_w[:VD],    qkv_b[:VD])
            K = F.linear(windowed, qkv_w[VD:2*VD], qkv_b[VD:2*VD])
            V = F.linear(windowed, qkv_w[2*VD:],  qkv_b[2*VD:])

            # multi-head reshape → (num_win*nh, seq, hd)
            Q = Q.reshape(num_win, seq, nh, hd).permute(0, 2, 1, 3).reshape(-1, seq, hd)
            K = K.reshape(num_win, seq, nh, hd).permute(0, 2, 1, 3).reshape(-1, seq, hd)
            V = V.reshape(num_win, seq, nh, hd).permute(0, 2, 1, 3).reshape(-1, seq, hd)

            # Manual attention matching HW flash_attention_core
            scale = 1.0 / math.sqrt(hd)
            scores = torch.bmm(Q * scale, K.transpose(-2, -1))
            probs = torch.softmax(scores.float(), dim=-1).to(torch.bfloat16)
            attn_out = torch.bmm(probs, V)

            # merge heads → (num_win, seq, VD)
            attn_out = (attn_out.reshape(num_win, nh, seq, hd)
                        .permute(0, 2, 1, 3)
                        .reshape(num_win, seq, VD))

            # output projection
            proj_w = sd[prefix + "attn.proj.weight"].to(torch.bfloat16)
            proj_b = sd[prefix + "attn.proj.bias"].to(torch.bfloat16)
            proj_out = F.linear(attn_out, proj_w, proj_b)

            if not is_global:
                # window reverse + unpad 70→64
                proj_spatial = (proj_out.reshape(nw_side, nw_side, ws, ws, VD)
                                .permute(0, 2, 1, 3, 4)
                                .reshape(self.GRID_SIZE_PAD, self.GRID_SIZE_PAD, VD))
                proj_flat = proj_spatial[:self.GRID_SIZE, :self.GRID_SIZE, :].reshape(GA, VD)
            else:
                proj_flat = proj_out.squeeze(0)

            # residual 1
            x = x + proj_flat

            # layer_norm2 — HW computes mean in BF16/BF19; round mean to BF16 to match
            _ln2_gamma = sd[prefix + "layer_norm2.weight"].to(torch.bfloat16)
            _ln2_beta  = sd[prefix + "layer_norm2.bias"].to(torch.bfloat16)
            _mean = x.float().mean(dim=-1, keepdim=True).to(torch.bfloat16)
            _cent = (x - _mean).to(torch.bfloat16)
            _rms  = _cent.float().pow(2).mean(dim=-1, keepdim=True).sqrt().to(torch.bfloat16)
            normed2 = (_cent.float() / _rms.float()).to(torch.bfloat16) * _ln2_gamma + _ln2_beta

            # MLP: HW accumulates in BF19/TF32-like precision — use float32 linear to match
            lin1_w = sd[prefix + "mlp.lin1.weight"].to(torch.bfloat16)
            lin1_b = sd[prefix + "mlp.lin1.bias"].to(torch.bfloat16)
            lin2_w = sd[prefix + "mlp.lin2.weight"].to(torch.bfloat16)
            lin2_b = sd[prefix + "mlp.lin2.bias"].to(torch.bfloat16)
            fc1_out = F.linear(normed2.float(), lin1_w.float(), lin1_b.float()).to(torch.bfloat16)
            mlp_mid = (fc1_out * torch.sigmoid(1.702 * fc1_out.float()).to(torch.bfloat16)).to(torch.bfloat16)
            mlp_out = F.linear(mlp_mid.float(), lin2_w.float(), lin2_b.float()).to(torch.bfloat16)

            # residual 2
            x = x + mlp_out

        del sd
        torch.set_grad_enabled(True)
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

        status = ("PASS" if m["snr_db"] > 40 or m["pct_1ulp"] > 99.0
                  else ("CLOSE" if m["snr_db"] > 20 else "FAIL"))
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

def _vit_block_validate(ue: "Sam1VitB_UnifiedEngine", image_path: str, num_blocks: int) -> None:
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
        m = Sam1VitB_UnifiedEngine.tensor_metrics(hw, cpu)

        btype  = "GLOBAL" if blk in ue.VIT_GLOBAL_BLOCKS else "local"
        status = ("PASS" if m["snr_db"] > 40 or m["pct_1ulp"] > 99.0
                  else ("CLOSE" if m["snr_db"] > 20 else "FAIL"))
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
    parser = argparse.ArgumentParser(description="SAM 1 ViT-B accelerator inference.")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--dev", type=str, default="xdma0", help="DMA device")
    parser.add_argument("--cycle", type=float, default=5.62, help="Clock cycle time in ns")
    parser.add_argument("--validate-vit", action="store_true",
                        help="Per-block HW vs CPU validation (slow: one compile+run per block)")
    parser.add_argument("--validate-blocks", type=int, default=8,
                        help="Number of blocks to validate with --validate-vit (default: 8)")
    parser.add_argument("--no-step", action="store_true",
                        help="With --validate-vit: single compile+run to block N, one CPU ref (faster)")
    parser.add_argument("--validate-block0", action="store_true",
                        help="Validate ViT block 0 full output HW vs CPU ref, then assert False (DEBUG)")
    parser.add_argument("--validate-ln1", action="store_true",
                        help="Validate block 0 pre-attn LayerNorm (VIT_LN_OUT, 4096×768) HW vs CPU ref, then assert False (DEBUG)")
    parser.add_argument("--validate-ln2", action="store_true",
                        help="Validate block 0 pre-MLP LayerNorm (VIT_LN_OUT, 4096×768) HW vs CPU ref, then assert False (DEBUG)")
    parser.add_argument("--validate-fc1", action="store_true",
                        help="Validate block 0 fc1+GELU (VIT_MLP_MID, 4096×3072) HW vs CPU ref, then assert False (DEBUG)")
    parser.add_argument("--validate-mlp", action="store_true",
                        help="Validate block 0 MLP output (VIT_MLP_OUT, 4096×768) HW vs CPU ref, then assert False (DEBUG)")
    parser.add_argument("--validate-dec-input", action="store_true",
                        help="Validate decoder token assembly (DEC_TOKENS) and image src (DEC_SRC) vs CPU ref")
    args = parser.parse_args()

    global _SILENT_MODE
    _SILENT_MODE = True

    set_dma_device(args.dev)
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle

    image_path = args.image or os.path.join(SCRIPT_DIR, "../../../test_samples/vette.jpg")

    _original_print(f"SAM 1 ViT-B on {args.dev}")

    t0 = _time.perf_counter()
    ue = Sam1VitB_UnifiedEngine(script_dir=SCRIPT_DIR)
    _original_print(f"  Weights: {_time.perf_counter() - t0:.3f}s")

    if args.validate_ln1:
        _original_print("  Validating block 0 pre-attn LayerNorm (VIT_LN_OUT, 4096×768)...")
        ue._ln1_checkpoint = True
        try:
            prog_addr = ue.compile_full_fused()
        except _CheckpointStop:
            pass
        ue.run_hw(image_path, prog_addr)

        import torch.nn as nn
        ckpt_path = _ensure_checkpoint(SCRIPT_DIR, ue.cfg)
        sd = _load_sam1_state_dict(ckpt_path)
        prefix = "vision_encoder.layers.0."
        x = ue.cpu_reference_patch_embed(image_path).to(torch.bfloat16)
        # HW layer_norm_core: LayerNorm without eps — mean subtracted, then /sqrt(var)
        _gamma = sd[prefix + "layer_norm1.weight"].to(torch.bfloat16)
        _beta  = sd[prefix + "layer_norm1.bias"].to(torch.bfloat16)
        _xf = x.float()
        _mean = _xf.mean(dim=-1, keepdim=True)
        _cent = _xf - _mean
        _rms  = _cent.pow(2).mean(dim=-1, keepdim=True).sqrt()
        ln1_cpu = (_cent / _rms).to(torch.bfloat16) * _gamma + _beta
        del sd
        ue.validate(
            dram_addr=ue.VIT_LN_OUT,
            num_elements=ue.GRID_AREA * ue.VIT_DIM,
            cpu_ref=ln1_cpu,
            label="block 0 pre-attn LN1 (4096×768)",
        )
        assert False, "DEBUG"

    # ── Shared CPU ref builder for block 0 MLP checkpoints ──────────────────
    def _cpu_ref_block0_mlp(ue, image_path):
        """Return dict of block 0 MLP-stage CPU tensors (all bf16)."""
        import torch.nn as nn
        import torch.nn.functional as F
        ckpt_path = _ensure_checkpoint(SCRIPT_DIR, ue.cfg)
        sd = _load_sam1_state_dict(ckpt_path)
        prefix = "vision_encoder.layers.0."
        VD, ws, nw_side = ue.VIT_DIM, ue.VIT_WINDOW_SIZE, 5
        nh, hd = ue.VIT_HEADS, ue.VIT_HEAD_DIM
        nw = ue.VIT_NUM_WINDOWS   # 25
        GA = ue.GRID_AREA         # 4096

        x = ue.cpu_reference_patch_embed(image_path).to(torch.bfloat16)
        ln1 = nn.LayerNorm(VD, eps=1e-6)
        ln1.weight.data = sd[prefix + "layer_norm1.weight"].float()
        ln1.bias.data   = sd[prefix + "layer_norm1.bias"].float()
        normed = ln1(x.float()).to(torch.bfloat16)

        spatial = normed.reshape(ue.GRID_SIZE, ue.GRID_SIZE, VD)
        padded  = F.pad(spatial.permute(2,0,1).unsqueeze(0),
                        (0, ue.GRID_SIZE_PAD - ue.GRID_SIZE,
                         0, ue.GRID_SIZE_PAD - ue.GRID_SIZE)).squeeze(0).permute(1,2,0)
        windowed = (padded.reshape(nw_side, ws, nw_side, ws, VD)
                          .permute(0,2,1,3,4).reshape(nw * ws*ws, VD))

        qkv_w = sd[prefix + "attn.qkv.weight"].to(torch.bfloat16)
        qkv_b = sd[prefix + "attn.qkv.bias"].to(torch.bfloat16)
        Q = F.linear(windowed, qkv_w[:VD],     qkv_b[:VD])
        K = F.linear(windowed, qkv_w[VD:2*VD], qkv_b[VD:2*VD])
        V = F.linear(windowed, qkv_w[2*VD:],   qkv_b[2*VD:])
        Q = Q.reshape(nw, ws*ws, nh, hd).permute(0,2,1,3).reshape(nw*nh, ws*ws, hd)
        K = K.reshape(nw, ws*ws, nh, hd).permute(0,2,1,3).reshape(nw*nh, ws*ws, hd)
        V = V.reshape(nw, ws*ws, nh, hd).permute(0,2,1,3).reshape(nw*nh, ws*ws, hd)
        scores = torch.bmm(Q * (1.0 / math.sqrt(hd)), K.transpose(-2,-1))
        probs  = torch.softmax(scores.float(), dim=-1).to(torch.bfloat16)
        attn_out = torch.bmm(probs, V)
        attn_merged = (attn_out.reshape(nw, nh, ws*ws, hd)
                                .permute(0,2,1,3).reshape(nw * ws*ws, VD))

        proj_w = sd[prefix + "attn.proj.weight"].to(torch.bfloat16)
        proj_b = sd[prefix + "attn.proj.bias"].to(torch.bfloat16)
        proj_out = F.linear(attn_merged.reshape(nw, ws*ws, VD), proj_w, proj_b)
        proj_spatial = (proj_out.reshape(nw_side, nw_side, ws, ws, VD)
                        .permute(0,2,1,3,4).reshape(ue.GRID_SIZE_PAD, ue.GRID_SIZE_PAD, VD))
        proj_flat = proj_spatial[:ue.GRID_SIZE, :ue.GRID_SIZE, :].reshape(GA, VD)

        residual1 = (x + proj_flat).to(torch.bfloat16)

        # HW computes mean in BF16/BF19; round mean to BF16 before subtracting
        # to match HW rounding on residual1's value distribution
        ln2_gamma = sd[prefix + "layer_norm2.weight"].to(torch.bfloat16)
        ln2_beta  = sd[prefix + "layer_norm2.bias"].to(torch.bfloat16)
        _mean2 = residual1.float().mean(dim=-1, keepdim=True).to(torch.bfloat16)
        _centered2 = (residual1 - _mean2).to(torch.bfloat16)
        _rms2 = _centered2.float().pow(2).mean(dim=-1, keepdim=True).sqrt().to(torch.bfloat16)
        normed2 = (_centered2.float() / _rms2.float()).to(torch.bfloat16) * ln2_gamma + ln2_beta

        # HW accumulates in BF19/TF32-like precision — use float32 linear to match
        lin1_w = sd[prefix + "mlp.lin1.weight"].to(torch.bfloat16)
        lin1_b = sd[prefix + "mlp.lin1.bias"].to(torch.bfloat16)
        lin2_w = sd[prefix + "mlp.lin2.weight"].to(torch.bfloat16)
        lin2_b = sd[prefix + "mlp.lin2.bias"].to(torch.bfloat16)
        fc1_out = F.linear(normed2.float(), lin1_w.float(), lin1_b.float()).to(torch.bfloat16)
        mlp_mid = (fc1_out * torch.sigmoid(1.702 * fc1_out.float()).to(torch.bfloat16)).to(torch.bfloat16)
        mlp_out = F.linear(mlp_mid.float(), lin2_w.float(), lin2_b.float()).to(torch.bfloat16)

        del sd
        return {
            "ln2":     normed2,   # VIT_LN_OUT   (4096, 768)
            "fc1":     mlp_mid,   # VIT_MLP_MID  (4096, 3072)
            "mlp_out": mlp_out,   # VIT_MLP_OUT  (4096, 768)
        }

    if args.validate_ln2:
        _original_print("  Validating block 0 pre-MLP LayerNorm (VIT_LN_OUT, 4096×768)...")
        ue._ln2_checkpoint = True
        try:
            prog_addr = ue.compile_full_fused()
        except _CheckpointStop:
            pass
        ue.run_hw(image_path, prog_addr)
        refs = _cpu_ref_block0_mlp(ue, image_path)
        ue.validate(
            dram_addr=ue.VIT_LN_OUT,
            num_elements=ue.GRID_AREA * ue.VIT_DIM,
            cpu_ref=refs["ln2"],
            label="block 0 pre-MLP LN2 (4096×768)",
        )
        assert False, "DEBUG"

    if args.validate_fc1:
        _original_print("  Validating block 0 fc1+GELU (VIT_MLP_MID, 4096×3072)...")
        ue._fc1_checkpoint = True
        try:
            prog_addr = ue.compile_full_fused()
        except _CheckpointStop:
            pass
        ue.run_hw(image_path, prog_addr)
        refs = _cpu_ref_block0_mlp(ue, image_path)
        ue.validate(
            dram_addr=ue.VIT_MLP_MID,
            num_elements=ue.GRID_AREA * ue.VIT_MLP_HIDDEN,
            cpu_ref=refs["fc1"],
            label="block 0 fc1+GELU (4096×3072)",
        )
        assert False, "DEBUG"

    if args.validate_mlp:
        _original_print("  Validating block 0 MLP output (VIT_MLP_OUT, 4096×768)...")
        ue._mlp_checkpoint = True
        try:
            prog_addr = ue.compile_full_fused()
        except _CheckpointStop:
            pass
        ue.run_hw(image_path, prog_addr)
        refs = _cpu_ref_block0_mlp(ue, image_path)
        ue.validate(
            dram_addr=ue.VIT_MLP_OUT,
            num_elements=ue.GRID_AREA * ue.VIT_DIM,
            cpu_ref=refs["mlp_out"],
            label="block 0 MLP out (4096×768)",
        )
        assert False, "DEBUG"

    if args.validate_dec_input:
        _original_print("  Validating decoder layer 0 norm1 (LayerNorm on tokens before self-attn)...")

        # Run ViT + neck to populate NECK_OUT in DRAM
        prog_addr = ue.compile_full_fused()
        ue.run_hw(image_path, prog_addr)

        ckpt_path = _ensure_checkpoint(SCRIPT_DIR, ue.cfg)
        sd = _load_sam1_state_dict(ckpt_path)

        # ── CPU: assemble decoder token input ──
        TEST_POINT = (512.0, 512.0)   # (x, y) pixel coords on 1024×1024 image
        _pe_mat = sd["shared_image_embedding.positional_embedding"].float()  # (2, 128)

        def _point_pe(x, y):
            coord = torch.tensor([[x / ue.IMAGE_SIZE, y / ue.IMAGE_SIZE]], dtype=torch.float32)
            proj  = 2 * math.pi * (coord @ _pe_mat)
            return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1).to(torch.bfloat16)  # (1, 256)

        fg_emb  = (_point_pe(*TEST_POINT).float() + sd["prompt_encoder.point_embed.0.weight"].float()).to(torch.bfloat16)
        pad_emb = (_point_pe(0.0, 0.0).float()    + sd["prompt_encoder.not_a_point_embed.weight"].float()).to(torch.bfloat16)
        iou_tok   = sd["mask_decoder.iou_token.weight"].to(torch.bfloat16)    # (1, 256)
        mask_toks = sd["mask_decoder.mask_tokens.weight"].to(torch.bfloat16)  # (4, 256)
        cpu_tokens = torch.cat([iou_tok, mask_toks, fg_emb, pad_emb], dim=0)  # (7, 256)

        # ── CPU: LayerNorm on tokens (norm1 — same HW layer_norm_core convention) ──
        _g = sd["mask_decoder.transformer.layers.0.layer_norm1.weight"].float()  # (256,)
        _b = sd["mask_decoder.transformer.layers.0.layer_norm1.bias"].float()
        _x = cpu_tokens.float()
        _mean = _x.mean(dim=-1, keepdim=True)
        _cent = _x - _mean
        _rms  = _cent.pow(2).mean(dim=-1, keepdim=True).sqrt()
        cpu_norm1 = ((_cent / _rms).to(torch.bfloat16) * _g.to(torch.bfloat16) + _b.to(torch.bfloat16))  # (7, 256)

        # ── CPU: self-attn Q/K/V projections on norm1 output ──
        import torch.nn.functional as F
        sa_q_w = sd["mask_decoder.transformer.layers.0.self_attn.q_proj.weight"].to(torch.bfloat16)
        sa_q_b = sd["mask_decoder.transformer.layers.0.self_attn.q_proj.bias"].to(torch.bfloat16)
        sa_k_w = sd["mask_decoder.transformer.layers.0.self_attn.k_proj.weight"].to(torch.bfloat16)
        sa_k_b = sd["mask_decoder.transformer.layers.0.self_attn.k_proj.bias"].to(torch.bfloat16)
        sa_v_w = sd["mask_decoder.transformer.layers.0.self_attn.v_proj.weight"].to(torch.bfloat16)
        sa_v_b = sd["mask_decoder.transformer.layers.0.self_attn.v_proj.bias"].to(torch.bfloat16)
        cpu_Q = F.linear(cpu_norm1.float(), sa_q_w.float(), sa_q_b.float()).to(torch.bfloat16)  # (7, 256)
        cpu_K = F.linear(cpu_norm1.float(), sa_k_w.float(), sa_k_b.float()).to(torch.bfloat16)
        cpu_V = F.linear(cpu_norm1.float(), sa_v_w.float(), sa_v_b.float()).to(torch.bfloat16)
        del sd

        # ── HW: DMA tokens, norm1, then Q/K/V projections ──
        ue.dma_to_accelerator_memory(ue.DEC_TOKENS, cpu_tokens.flatten())

        ue.start_capture()
        lw0 = ue.dec_layer_weights[0]
        # norm1
        ue.layer_norm_core_dram(
            M=ue.DEC_NUM_TOKENS, N=ue.DEC_DIM,
            A_DRAM_ADDR=ue.DEC_TOKENS,
            OUTPUT_DRAM_ADDR=ue.DEC_TOKENS_NORM,
            GAMMA_DRAM_ADDR=lw0['norm1_w'],
            BETA_DRAM_ADDR=lw0['norm1_b'],
        )
        # Q projection: (7, 256) @ (256, 256)^T + bias → (7, 256)
        ue.matmat_mul_core(
            M=ue.DEC_NUM_TOKENS, K=ue.DEC_DIM, N=ue.DEC_DIM,
            A_DRAM_ADDR=ue.DEC_TOKENS_NORM,
            B_DRAM_ADDR=lw0['sa_q_w'],
            OUTPUT_DRAM_ADDR=ue.DEC_Q,
            C_DRAM_ADDR=lw0['sa_q_b'],
            bias_mode="broadcast_N",
        )
        # K projection
        ue.matmat_mul_core(
            M=ue.DEC_NUM_TOKENS, K=ue.DEC_DIM, N=ue.DEC_DIM,
            A_DRAM_ADDR=ue.DEC_TOKENS_NORM,
            B_DRAM_ADDR=lw0['sa_k_w'],
            OUTPUT_DRAM_ADDR=ue.DEC_K,
            C_DRAM_ADDR=lw0['sa_k_b'],
            bias_mode="broadcast_N",
        )
        # V projection
        ue.matmat_mul_core(
            M=ue.DEC_NUM_TOKENS, K=ue.DEC_DIM, N=ue.DEC_DIM,
            A_DRAM_ADDR=ue.DEC_TOKENS_NORM,
            B_DRAM_ADDR=lw0['sa_v_w'],
            OUTPUT_DRAM_ADDR=ue.DEC_V,
            C_DRAM_ADDR=lw0['sa_v_b'],
            bias_mode="broadcast_N",
        )
        ue._finalize_program()
        ue.start_execute_from_dram(ue._last_prog_addr)
        ue.wait_queue()

        ue.validate(dram_addr=ue.DEC_TOKENS_NORM,
                    num_elements=ue.DEC_NUM_TOKENS * ue.DEC_DIM,
                    cpu_ref=cpu_norm1,
                    label="dec layer0 norm1 (7×256)")
        ue.validate(dram_addr=ue.DEC_Q,
                    num_elements=ue.DEC_NUM_TOKENS * ue.DEC_DIM,
                    cpu_ref=cpu_Q,
                    label="dec layer0 self-attn Q (7×256)")
        ue.validate(dram_addr=ue.DEC_K,
                    num_elements=ue.DEC_NUM_TOKENS * ue.DEC_DIM,
                    cpu_ref=cpu_K,
                    label="dec layer0 self-attn K (7×256)")
        ue.validate(dram_addr=ue.DEC_V,
                    num_elements=ue.DEC_NUM_TOKENS * ue.DEC_DIM,
                    cpu_ref=cpu_V,
                    label="dec layer0 self-attn V (7×256)")
        assert False, "DEBUG"

    if args.validate_block0:
        _original_print("  Validating ViT block 0 (post-residual output)...")
        ue._vit_checkpoint = 0
        try:
            prog_addr = ue.compile_full_fused()
        except _CheckpointStop:
            pass
        ue.run_hw(image_path, prog_addr)
        cpu_ref = ue.cpu_reference_vit_blocks(image_path, num_blocks=1)
        ue.validate(
            dram_addr=ue.VIT_LAYER_IN,
            num_elements=ue.GRID_AREA * ue.VIT_DIM,
            cpu_ref=cpu_ref,
            label="ViT block 0",
        )
        assert False, "DEBUG"

    if args.validate_vit and args.no_step:
        # Single compile to block N, one CPU reference
        n = args.validate_blocks
        _original_print(f"  Validating ViT blocks 0-{n - 1} (single compile)...")
        ue._vit_checkpoint = n - 1
        try:
            prog_addr = ue.compile_full_fused()
        except _CheckpointStop:
            pass
        ue.run_hw(image_path, prog_addr)
        cpu_ref = ue.cpu_reference_vit_blocks(image_path, num_blocks=n)
        ue.validate(
            dram_addr=ue.VIT_LAYER_IN,
            num_elements=ue.GRID_AREA * ue.VIT_DIM,
            cpu_ref=cpu_ref,
            label=f"ViT blocks 0-{n - 1}",
        )
        assert False, "DEBUG"
    elif args.validate_vit:
        # Per-block HW vs CPU validation
        _original_print(f"  Validating ViT blocks 0-{args.validate_blocks - 1} (per-block)...")
        _vit_block_validate(ue, image_path, num_blocks=args.validate_blocks)
    else:
        # ── ViT + neck forward pass ──────────────────────────────────────
        neck_cache = os.path.join(SCRIPT_DIR, "sam1_vit_b_bin", "neck_output.pt")
        n_elems    = ue.GRID_AREA * ue.NECK_DIM  # 4096 * 256

        if False and os.path.exists(neck_cache):
            _original_print(f"  Loading cached neck output from {neck_cache}...")
            neck_out = torch.load(neck_cache, weights_only=True)  # (4096, 256) bf16
        else:
            t1 = _time.perf_counter()
            prog_addr = ue.compile_full_fused()
            _original_print(f"  Compile: {_time.perf_counter() - t1:.3f}s")

            t2 = _time.perf_counter()
            ue.run_hw(image_path, prog_addr)
            _original_print(f"  Execute: {_time.perf_counter() - t2:.3f}s")

            neck_out = ue.read_tensor_from_dram(ue.NECK_OUT, n_elems)
            neck_out = neck_out.reshape(ue.GRID_AREA, ue.NECK_DIM)
            os.makedirs(os.path.dirname(neck_cache), exist_ok=True)
            torch.save(neck_out, neck_cache)
            _original_print(f"  Saved neck output → {neck_cache}")

        _original_print(f"  Neck output shape: {neck_out.shape}, dtype: {neck_out.dtype}")
        _original_print(f"  |neck_out| mean={neck_out.float().abs().mean():.4f}  "
                        f"max={neck_out.float().abs().max():.4f}")


if __name__ == "__main__":
    main()
