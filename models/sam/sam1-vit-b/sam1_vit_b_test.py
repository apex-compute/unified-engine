#!/usr/bin/env python3
"""
SAM 1 ViT-B segmentation inference on accelerator.

  - Config from sam1_vit_b_config.json; weights from facebook/sam-vit-base (HuggingFace).
  - Forward pass: ViT-B backbone (12 blocks, dim=768, 12 heads, window=14)
                  → neck (1×1 conv + LN + 3×3 conv + LN → 256-dim features).
  - All BF16 weights (no quantization). ~86M image encoder params ≈ 172 MB.
  - No RoPE: SAM1 uses decomposed 2D relative positional embeddings (rel_pos).
    Implemented via precomputed (sz, sz, 64) Rh/Rw tables + matmul-based scatter.
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
user_dma_core.MAX_DECODER_INSTRUCTIONS = (0x100000000 - 0x3C000000) // 32  # 3.1 GB instruction budget
# Section 2 uses 1 GB instructions (0xC0000000–0xFFFFFFFF); override before instantiating S2 engine.
from user_dma_core import (
    DMA_DEVICE_H2C, DMA_DEVICE_C2H, DRAM_INSTRUCTION_ADDR, TYPE,
    UE_VECTOR_SIZE, URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS,
    set_dma_device, UnifiedEngine,
)
from model_lib_core import eltwise_mul_core_dram


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
                                  SCRATCH_DRAM_ADDR: int,
                                  RH_DRAM_ADDR: int = None,
                                  RW_DRAM_ADDR: int = None,
                                  ONES_SCATTER_ADDR: int = None,
                                  BIAS_DRAM_ADDR: int = None) -> int:
    """Global self-attention processing one head at a time to bound scratch usage.

    flash_attention_core needs (head_dim * seq_len + seq_len²) scratch per head.
    For seq=4096, head_dim=64: ~4 MB per head — fits in local scratch (111 MB)
    when scratch is reused across heads instead of allocated per-head simultaneously.

    If RH_DRAM_ADDR / RW_DRAM_ADDR / ONES_SCATTER_ADDR / BIAS_DRAM_ADDR are provided,
    decomposed 2D relative positional bias is computed per head before flash_attention_core.
    BIAS_DRAM_ADDR must hold (seq_len, seq_len) bf16 scratch (zeroed per head here).
    SCRATCH_DRAM_ADDR is reused as scratch for add_rel_pos_bias_dram (VIT_COMPACT_HEADS).
    """
    bpe = 2
    sz = int(seq_len ** 0.5)  # 64 for seq_len=4096
    head_stride = seq_len * head_dim * bpe
    use_rel_pos = (RH_DRAM_ADDR is not None and RW_DRAM_ADDR is not None
                   and ONES_SCATTER_ADDR is not None and BIAS_DRAM_ADDR is not None)

    total_flops = 0
    for h in range(num_heads):
        q_h = Q_DRAM_ADDR + h * head_stride
        bias_addr = None
        if use_rel_pos:
            # Zero-fill the per-head bias buffer (seq_len × seq_len)
            dram_zero_fill(ue, BIAS_DRAM_ADDR, seq_len * seq_len)
            # Compute decomposed rel_pos bias for this one head
            # Q layout for this head: (seq_len, head_dim) at q_h; seq_pad = seq_len (no padding)
            add_rel_pos_bias_dram(ue,
                Q_DRAM_ADDR=q_h,
                RH_DRAM_ADDR=RH_DRAM_ADDR,
                RW_DRAM_ADDR=RW_DRAM_ADDR,
                BIAS_DRAM_ADDR=BIAS_DRAM_ADDR,
                ONES_SCATTER_ADDR=ONES_SCATTER_ADDR,
                sz=sz,
                sz_pad=64,
                num_batches=1,
                seq_pad=seq_len,
                SCRATCH_DRAM_ADDR=SCRATCH_DRAM_ADDR,
            )
            bias_addr = BIAS_DRAM_ADDR
        total_flops += ue.flash_attention_core(
            head_dim=head_dim,
            seq_len=seq_len,
            Q_DRAM_ADDR=q_h,
            K_DRAM_ADDR=K_DRAM_ADDR      + h * head_stride,
            V_DRAM_ADDR=V_DRAM_ADDR      + h * head_stride,
            OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR + h * head_stride,
            SCRATCH_DRAM_ADDR=SCRATCH_DRAM_ADDR,  # reused each head
            BIAS_DRAM_ADDR=bias_addr,
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
    vt_size = kv_len * head_dim  # V^T temp needed per head
    scratch_per_head = (score_size + vt_size) * bpe  # scores + V^T

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
# Decomposed 2D relative positional bias (SAM1 ViT-B)
# =============================================================================

def add_rel_pos_bias_dram(ue: UnifiedEngine,
                           Q_DRAM_ADDR: int,
                           RH_DRAM_ADDR: int,
                           RW_DRAM_ADDR: int,
                           BIAS_DRAM_ADDR: int,
                           ONES_SCATTER_ADDR: int,
                           sz: int,
                           sz_pad: int,
                           num_batches: int,
                           seq_pad: int,
                           SCRATCH_DRAM_ADDR: int) -> None:
    """Add decomposed 2D relative positional bias into BIAS_DRAM_ADDR in-place.

    sz      = actual grid size (14 for local windows, 64 for global)
    sz_pad  = padded key-position dim for matmuls (64; N≥32 requirement of matmat_mul_core)
    seq_pad = bias row width in DRAM (256 for local, 4096 for global)
    seq_sq  = sz * sz = real token count per window (196 or 4096)

    Rh/Rw stored as (sz, sz_pad, 64): key-position dim zero-padded to sz_pad.
    ONES_SCATTER stored as (seq_pad, sz_pad): [kh*sz+kw, kh]=1 for kh,kw<sz; rest 0.

    Algorithm per (batch b, grid row qh):
      rel_h     = q_row @ Rh[qh].T          (sz, sz_pad); first sz cols valid
      scatter_h = rel_h @ ONES_SCATTER.T    (sz, seq_pad)
                  scatter_h[qw, kh*sz+kw] = rel_h[qw, kh]
      per qw:
        rel_w = q_row[qw] @ Rw[qw].T        (1, sz_pad); first sz entries valid
        tile first sz entries → RW_TILE[qw, kh*sz:...+sz] for each kh
      per qw: scatter_h[qw, 0:seq_sq] += RW_TILE[qw, 0:seq_sq]
      per qw: bias[b, qh*sz+qw, 0:seq_sq] += scatter_h[qw, 0:seq_sq]

    Scratch layout in SCRATCH_DRAM_ADDR (VIT_COMPACT_HEADS or VIT_ATTN_SCRATCH):
      [0                        : sz*sz_pad)       → SCR_REL_H   (sz, sz_pad)
      [sz*sz_pad                : +sz*seq_pad)     → SCR_SCATTER (sz, seq_pad)
      [sz*sz_pad+sz*seq_pad     : +sz_pad)         → SCR_REL_W   (sz_pad,) per-qw
      [sz*sz_pad+sz*seq_pad+sz_pad : +sz*seq_sq)   → SCR_RW_TILE (sz, seq_sq)
    Total local : 14*64 + 14*256 + 64 + 14*196 = 896+3584+64+2744 = 7288 elements
    Total global: 64*64 + 64*4096 + 64 + 64*4096 = 4096+262144+64+262144 = 528448 elements
    """
    bpe = 2
    hd = 64  # head_dim, always 64 for SAM1 ViT-B
    seq_sq = sz * sz

    SCR_REL_H   = SCRATCH_DRAM_ADDR
    SCR_SCATTER = SCRATCH_DRAM_ADDR + sz * sz_pad * bpe
    SCR_REL_W   = SCRATCH_DRAM_ADDR + sz * sz_pad * bpe + sz * seq_pad * bpe
    SCR_RW_TILE = SCRATCH_DRAM_ADDR + sz * sz_pad * bpe + sz * seq_pad * bpe + sz_pad * bpe

    bias_batch_stride = seq_pad * seq_pad

    for b in range(num_batches):
        q_batch_addr = Q_DRAM_ADDR    + b * seq_pad * hd * bpe
        bias_base    = BIAS_DRAM_ADDR + b * bias_batch_stride * bpe

        for qh in range(sz):
            q_row_addr = q_batch_addr + qh * sz * hd * bpe
            rh_qh_addr = RH_DRAM_ADDR + qh * sz_pad * hd * bpe  # Rh[qh]: (sz_pad, 64)

            # Step 1: rel_h = q_row @ Rh[qh].T → (sz, sz_pad)
            # A=(sz,64), B=(sz_pad,64) stored → C = A@B^T = (sz, sz_pad)
            ue.matmat_mul_core(
                M=sz, K=hd, N=sz_pad,
                A_DRAM_ADDR=q_row_addr,
                B_DRAM_ADDR=rh_qh_addr,
                OUTPUT_DRAM_ADDR=SCR_REL_H,
            )

            # Step 2: scatter_h = rel_h @ ONES_SCATTER^T → (sz, seq_pad)
            # A=(sz, sz_pad), B=(seq_pad, sz_pad) stored → C = A@B^T = (sz, seq_pad)
            # scatter_h[qw, kh*sz+kw] = rel_h[qw, kh]  for kh<sz, kw<sz
            ue.matmat_mul_core(
                M=sz, K=sz_pad, N=seq_pad,
                A_DRAM_ADDR=SCR_REL_H,
                B_DRAM_ADDR=ONES_SCATTER_ADDR,
                OUTPUT_DRAM_ADDR=SCR_SCATTER,
            )

            # Step 3: per-qw: compute rel_w and tile first sz entries into RW_TILE
            for qw in range(sz):
                q_single_addr = q_row_addr + qw * hd * bpe
                rw_qw_addr    = RW_DRAM_ADDR + qw * sz_pad * hd * bpe  # Rw[qw]: (sz_pad, 64)
                # rel_w = q_single @ Rw[qw].T → (1, sz_pad); first sz entries valid
                ue.matmat_mul_core(
                    M=1, K=hd, N=sz_pad,
                    A_DRAM_ADDR=q_single_addr,
                    B_DRAM_ADDR=rw_qw_addr,
                    OUTPUT_DRAM_ADDR=SCR_REL_W,
                )
                # Tile first sz entries of SCR_REL_W into RW_TILE[qw, kh*sz:kh*sz+sz]
                rw_tile_row = SCR_RW_TILE + qw * seq_sq * bpe
                for kh in range(sz):
                    dram_copy(ue, SCR_REL_W, rw_tile_row + kh * sz * bpe, sz)

            # Step 4: per-qw: merge rel_w tile into scatter_h (first seq_sq cols only)
            for qw in range(sz):
                eltwise_add_dram(ue,
                    SCR_SCATTER + qw * seq_pad * bpe,
                    SCR_RW_TILE + qw * seq_sq  * bpe,
                    SCR_SCATTER + qw * seq_pad * bpe,
                    seq_sq)

            # Step 5: per-qw: add combined bias into BIAS_DRAM_ADDR rows
            for qw in range(sz):
                bias_row_addr = bias_base + (qh * sz + qw) * seq_pad * bpe
                eltwise_add_dram(ue,
                    bias_row_addr,
                    SCR_SCATTER + qw * seq_pad * bpe,
                    bias_row_addr,
                    seq_sq)


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
#   Params:      0x00000000 – 0x1FFFFFFF  (512 MB)  weights
#   Tensors:     0x20000000 – 0x3BFFFFFF  (448 MB)  activations / scratch
#   Instructions:0x3C000000 – 0xFFFFFFFF  (3.1 GB)  compiled instruction stream
# =============================================================================
SAM1_PARAMS_BASE  = 0x00000000
SAM1_TENSOR_BASE  = 0x20000000
SAM1_PROGRAM_BASE = 0x3C000000


# =============================================================================
# Sam1VitB_UnifiedEngine
# =============================================================================

class Sam1VitB_UnifiedEngine(UnifiedEngine):
    """SAM 1 ViT-B image segmentation on Unified Engine FPGA accelerator.

    Architecture:
        ViT-B backbone (12 blocks, dim=768, 12 heads, window=14, 4 global blocks)
        → Neck (Conv 1×1 → LN → Conv 3×3 → LN → 256-dim feature map at 64×64)

    Decomposed 2D relative positional embeddings (rel_pos) are implemented via
    precomputed (sz, sz, 64) Rh/Rw tables in params and matmul-based scatter on-chip.
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

        # matmat_mul_core requires N ≥ N_chunk (typically 32 or 64).
        # sz=14 (local window) is too small. We pad the k-position dimension to sz_pad=64
        # so all rel_pos matmuls use N=64 or N=seq_sq_pad≥64.
        _SZ_PAD = 64  # common pad target — equals head_dim and global sz

        # ONES_SCATTER matrices for rel_pos scatter via matmul.
        # Shape: (seq_sq_pad, sz_pad) so matmat_mul_core (A @ B^T) computes
        #   rel_h (sz, sz_pad) @ ONES_SCATTER^T (sz_pad, seq_sq_pad) → (sz, seq_sq_pad).
        # ONES_SCATTER[kh*sz+kw, kh] = 1.0 for kh<sz, kw<sz; rest 0.
        # Result: scatter[qw, kh*sz+kw] = rel_h[qw, kh]  (broadcast over kw). ✓
        def _make_ones_scatter(sz: int, seq_sq_pad: int, sz_pad: int) -> torch.Tensor:
            mat = torch.zeros(seq_sq_pad, sz_pad, dtype=torch.bfloat16)
            for kh in range(sz):
                mat[kh * sz : kh * sz + sz, kh] = 1.0
            return mat

        _ws_local     = self.VIT_WINDOW_SIZE   # 14
        _ws_global    = 64                     # global block: 64×64 grid
        _seq_pad_local = self.VIT_WINDOW_AREA_PAD  # 256 (bias row width for local)
        _seq_pad_glob  = self.GRID_AREA             # 4096 (no padding for global)
        self.ONES_SCATTER_LOCAL  = self._alloc_param(
            _make_ones_scatter(_ws_local,  _seq_pad_local, _SZ_PAD))  # (256, 64)
        self.ONES_SCATTER_GLOBAL = self._alloc_param(
            _make_ones_scatter(_ws_global, _seq_pad_glob,  _SZ_PAD))  # (4096, 64)

        def _get_rel_pos(q_size: int, raw: torch.Tensor, sz_pad: int) -> torch.Tensor:
            """Build (q_size, sz_pad, head_dim) relative positional lookup table.

            raw: checkpoint tensor of shape (max_rel_dist, head_dim).
            The second dimension (key positions) is zero-padded from q_size to sz_pad
            so matmat_mul_core (which requires N ≥ 32) can handle it as N=sz_pad.
            """
            import torch.nn.functional as F
            max_dist = 2 * q_size - 1
            raw = raw.float()
            if raw.shape[0] != max_dist:
                raw = F.interpolate(
                    raw.T.unsqueeze(0),  # (1, head_dim, src_len)
                    size=max_dist, mode="linear", align_corners=False,
                ).squeeze(0).T  # (max_dist, head_dim)
            q_coords = torch.arange(q_size, dtype=torch.long)[:, None]
            k_coords = torch.arange(q_size, dtype=torch.long)[None, :]
            rel = (q_coords - k_coords + (q_size - 1))
            result = raw[rel].to(torch.bfloat16)  # (q_size, q_size, head_dim)
            if sz_pad > q_size:
                padded = torch.zeros(q_size, sz_pad, result.shape[2], dtype=torch.bfloat16)
                padded[:, :q_size, :] = result
                return padded  # (q_size, sz_pad, head_dim)
            return result

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
            # Relative positional bias tables: (sz, sz_pad=64, head_dim=64) lookup tables.
            # Key-position dim padded to sz_pad=64 so matmat_mul_core N≥32 requirement is met.
            # Local blocks: sz=14 → (14, 64, 64) = 115 KB each.
            # Global blocks: sz=64 → (64, 64, 64) = 524 KB each.
            _is_global_blk = i in self.VIT_GLOBAL_BLOCKS
            _sz = 64 if _is_global_blk else self.VIT_WINDOW_SIZE  # 64 or 14
            _rph = sd[bp + "attn.rel_pos_h"]  # checkpoint shape, e.g. (27, 64) for local
            _rpw = sd[bp + "attn.rel_pos_w"]
            bw['rel_pos_h'] = self._alloc_param(_get_rel_pos(_sz, _rph, _SZ_PAD))
            bw['rel_pos_w'] = self._alloc_param(_get_rel_pos(_sz, _rpw, _SZ_PAD))
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
            sd[pe + "point_embed.1.weight"].to(torch.bfloat16))        # (1, 256)  foreground label=1
        self.PE_POINT_EMBED_BG  = self._alloc_param(
            sd[pe + "point_embed.0.weight"].to(torch.bfloat16))        # (1, 256)  background label=0
        self.PE_NOT_A_POINT     = self._alloc_param(
            sd[pe + "not_a_point_embed.weight"].to(torch.bfloat16))    # (1, 256)
        self.PE_NO_MASK         = self._alloc_param(
            sd[pe + "no_mask_embed.weight"].to(torch.bfloat16))        # (1, 256)

        # Prompt encoder sin/cos lookup tables for hardware PE computation.
        # Host selects 4 rows (128 bf16 each) for (x_px, y_px) before execution.
        _pe_mat_f = sd["shared_image_embedding.positional_embedding"].float()  # (2, 128)
        _coords_x = torch.arange(1024, dtype=torch.float32) / 1024.0           # (1024,)
        _coords_y = torch.arange(1024, dtype=torch.float32) / 1024.0           # (1024,)
        _proj_x = 2 * math.pi * _coords_x.unsqueeze(1) * _pe_mat_f[0:1, :]     # (1024, 128)
        _proj_y = 2 * math.pi * _coords_y.unsqueeze(1) * _pe_mat_f[1:2, :]     # (1024, 128)
        self.PE_SIN_X = self._alloc_param(torch.sin(_proj_x).to(torch.bfloat16))  # (1024, 128)
        self.PE_COS_X = self._alloc_param(torch.cos(_proj_x).to(torch.bfloat16))  # (1024, 128)
        self.PE_SIN_Y = self._alloc_param(torch.sin(_proj_y).to(torch.bfloat16))  # (1024, 128)
        self.PE_COS_Y = self._alloc_param(torch.cos(_proj_y).to(torch.bfloat16))  # (1024, 128)

        # Image positional encoding: constant 64×64 grid PE stored as param.
        # Used as key_pe in every cross-attention call.
        _pe_mat = sd["shared_image_embedding.positional_embedding"].float()  # (2, 128)
        _gy, _gx = torch.meshgrid(
            torch.arange(64, dtype=torch.float32),
            torch.arange(64, dtype=torch.float32), indexing='ij')
        _grid = torch.stack([(_gx + 0.5) / 64, (_gy + 0.5) / 64], dim=-1).reshape(-1, 2)  # (4096, 2)
        _grid = 2.0 * _grid - 1.0  # normalize [0,1] → [-1,1] to match CPU reference
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
            # Pad Q/K/V projection weights (DH*HD, DD) → (DH*HD_PAD, DD).
            # Each head's HD=32 real rows go to [h*HD_PAD : h*HD_PAD+HD]; rest zero.
            # Q rows are also scaled by sqrt(2) so that cross_attention_batched's
            # 1/sqrt(HD_PAD=64) gives the correct 1/sqrt(HD=32) total scale.
            _DH, _HD, _HD_PAD = self.DEC_HEADS, self.DEC_HEAD_DIM, 64
            for proj in ("q", "k", "v"):
                w_raw = sd[lp + f"self_attn.{proj}_proj.weight"].to(torch.bfloat16)
                b_raw = sd[lp + f"self_attn.{proj}_proj.bias"].to(torch.bfloat16)
                scale = 1.0
                w_pad = torch.zeros(_DH * _HD_PAD, w_raw.shape[1], dtype=torch.bfloat16)
                b_pad = torch.zeros(_DH * _HD_PAD, dtype=torch.bfloat16)
                for _h in range(_DH):
                    w_pad[_h*_HD_PAD : _h*_HD_PAD+_HD] = w_raw[_h*_HD : (_h+1)*_HD] * scale
                    b_pad[_h*_HD_PAD : _h*_HD_PAD+_HD] = b_raw[_h*_HD : (_h+1)*_HD]
                lw[f'sa_{proj}_w'] = self._alloc_param(w_pad)
                lw[f'sa_{proj}_b'] = self._alloc_param(b_pad)
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

        # Self-attn: head_dim=32 padded to 64 (bf16_permute_core requires dim_2 multiple of 64)
        self.DEC_SA_UNPAD_W  = self._alloc_param(
            self.build_unpad_weight(self.DEC_HEADS, self.DEC_HEAD_DIM, 64).T.contiguous())
        # Cross-attn: head_dim=16 padded to 64 (same dim_2 constraint)
        self.DEC_CA_UNPAD_W  = self._alloc_param(
            self.build_unpad_weight(self.DEC_HEADS, self.DEC_INTERNAL_HD, 64).T.contiguous())

        # Cross-attn Q/K/V projection weights padded: (DH*HD=128, DD=256) → (DH*HD_PAD=512, DD=256)
        # Same padding trick as self-attn: each head's HD=16 real rows in [h*64:h*64+16], rest zero.
        _CA_DH, _CA_HD, _CA_HDP = self.DEC_HEADS, self.DEC_INTERNAL_HD, 64
        _CA_ID = self.DEC_INTERNAL_DIM  # 128
        _CA_DD = self.DEC_DIM           # 256
        for i in range(self.DEC_LAYERS):
            lp = "mask_decoder." + f"transformer.layers.{i}."
            lw = self.dec_layer_weights[i]
            for attn_name in ("cross_attn_token_to_image", "cross_attn_image_to_token"):
                for proj in ("q", "k", "v"):
                    w_raw = sd[lp + f"{attn_name}.{proj}_proj.weight"].to(torch.bfloat16)  # (128, 256)
                    b_raw = sd[lp + f"{attn_name}.{proj}_proj.bias"].to(torch.bfloat16)    # (128,)
                    w_pad = torch.zeros(_CA_DH * _CA_HDP, _CA_DD, dtype=torch.bfloat16)
                    b_pad = torch.zeros(_CA_DH * _CA_HDP, dtype=torch.bfloat16)
                    for _h in range(_CA_DH):
                        w_pad[_h*_CA_HDP : _h*_CA_HDP+_CA_HD] = w_raw[_h*_CA_HD : (_h+1)*_CA_HD]
                        b_pad[_h*_CA_HDP : _h*_CA_HDP+_CA_HD] = b_raw[_h*_CA_HD : (_h+1)*_CA_HD]
                    prefix = "t2i" if attn_name == "cross_attn_token_to_image" else "i2t"
                    lw[f'ca_{prefix}_{proj}_w'] = self._alloc_param(w_pad)
                    lw[f'ca_{prefix}_{proj}_b'] = self._alloc_param(b_pad)

        # Bias for token→image cross-attn: mask query rows [7:64] (zero-padded queries)
        NT_PAD = 64
        NT_tok = self.DEC_NUM_TOKENS  # 7
        _t2i_bias = torch.zeros(self.DEC_HEADS, NT_PAD, NT_PAD, dtype=torch.bfloat16)
        _t2i_bias[:, NT_tok:, :] = float('-inf')  # mask padded query rows
        self.DEC_CA_T2I_BIAS = self._alloc_param(_t2i_bias.reshape(-1))

        # Bias for image→token cross-attn: mask kv columns [7:64] (zero-padded keys)
        _i2t_bias = torch.zeros(self.DEC_HEADS, NT_PAD, NT_PAD, dtype=torch.bfloat16)
        _i2t_bias[:, :, NT_tok:] = float('-inf')
        self.DEC_CA_I2T_BIAS = self._alloc_param(_i2t_bias.reshape(-1))

        # Per-row kv-column mask for i2t: shape (NT_PAD=64,), 0.0 at [0:7], -inf at [7:64].
        # Broadcast-added to every query row of the (GA, NT_PAD) score matrix before softmax.
        _i2t_bias_row = torch.zeros(NT_PAD, dtype=torch.bfloat16)
        _i2t_bias_row[NT_tok:] = float('-inf')
        self.DEC_CA_I2T_BIAS_ROW = self._alloc_param(_i2t_bias_row)

        # Self-attn attention bias: (DEC_HEADS, NT, NT_PAD) with -inf at kv positions NT:NT_PAD.
        # Hardware requires seq_len to be a multiple of UE_VECTOR_SIZE=64.
        # NT=7 pads to NT_PAD=64. Bias cancels padded key positions in softmax.
        NT_PAD = 64  # next multiple of UE_VECTOR_SIZE
        NT_tok = self.DEC_NUM_TOKENS  # 7
        # Bias (8, 64, 64): -inf at kv positions [7:64] for all query positions
        _sa_bias = torch.zeros(self.DEC_HEADS, NT_PAD, NT_PAD, dtype=torch.bfloat16)
        _sa_bias[:, :, NT_tok:] = float('-inf')
        self.DEC_SA_ATTN_BIAS = self._alloc_param(_sa_bias.reshape(-1))

        # Pad final_attn Q/K/V weights: same (DH*HD=128, DD=256) → (DH*HDP=512, DD=256) pattern
        for proj in ("q", "k", "v"):
            w_raw = sd[fp + f"{proj}_proj.weight"].to(torch.bfloat16)  # (128, 256)
            b_raw = sd[fp + f"{proj}_proj.bias"].to(torch.bfloat16)    # (128,)
            w_pad = torch.zeros(_CA_DH * _CA_HDP, _CA_DD, dtype=torch.bfloat16)
            b_pad = torch.zeros(_CA_DH * _CA_HDP, dtype=torch.bfloat16)
            for _h in range(_CA_DH):
                w_pad[_h*_CA_HDP : _h*_CA_HDP+_CA_HD] = w_raw[_h*_CA_HD : (_h+1)*_CA_HD]
                b_pad[_h*_CA_HDP : _h*_CA_HDP+_CA_HD] = b_raw[_h*_CA_HD : (_h+1)*_CA_HD]
            self.dec_final_attn[f'{proj}_w'] = self._alloc_param(w_pad)
            self.dec_final_attn[f'{proj}_b'] = self._alloc_param(b_pad)

        # Output upscaling: upscale_conv1 (256→64), upscale_layer_norm, upscale_conv2 (64→32)
        md = "mask_decoder."
        # ConvTranspose2d 0: weight (256, 64, 2, 2) → 4 sub-matrices (N=64, K=256) for (dr,dc) positions
        w_up0 = sd[md + "upscale_conv1.weight"].to(torch.bfloat16)  # (256, 64, 2, 2)
        self.DEC_UP0_W = []
        for dr in range(2):
            row = []
            for dc in range(2):
                sub = w_up0[:, :, dr, dc].T.contiguous()  # (C_out=64, C_in=256)
                row.append(self._alloc_param(sub))
            self.DEC_UP0_W.append(row)
        if md + "upscale_conv1.bias" in sd:
            self.DEC_UP0_B = self._alloc_param(sd[md + "upscale_conv1.bias"].to(torch.bfloat16))
        else:
            self.DEC_UP0_B = None
        # LayerNorm after first ConvTranspose2d
        self.DEC_UP_LN_W = self._alloc_param(sd[md + "upscale_layer_norm.weight"].to(torch.bfloat16))  # (64,)
        self.DEC_UP_LN_B = self._alloc_param(sd[md + "upscale_layer_norm.bias"].to(torch.bfloat16))    # (64,)
        # ConvTranspose2d 1: weight (64, 32, 2, 2) → 4 sub-matrices (N=32, K=64) for (dr,dc) positions
        w_up1 = sd[md + "upscale_conv2.weight"].to(torch.bfloat16)  # (64, 32, 2, 2)
        self.DEC_UP1_W = []
        for dr in range(2):
            row = []
            for dc in range(2):
                sub_raw = w_up1[:, :, dr, dc].T.contiguous()  # (C_out=32, C_in=64)
                sub = torch.zeros(64, 64, dtype=torch.bfloat16)
                sub[:32, :] = sub_raw
                row.append(self._alloc_param(sub))
            self.DEC_UP1_W.append(row)
        if md + "upscale_conv2.bias" in sd:
            b_raw = sd[md + "upscale_conv2.bias"].to(torch.bfloat16)
            b_pad = torch.zeros(64, dtype=torch.bfloat16)
            b_pad[:32] = b_raw
            self.DEC_UP1_B = self._alloc_param(b_pad)
        else:
            self.DEC_UP1_B = None

        # Output hypernetwork MLPs: 4 MLPs, each proj_in + layers.0 + proj_out (256→256→256→32)
        NUM_MASKS = 4
        self.dec_hyper_weights = []
        for m in range(NUM_MASKS):
            hp = md + f"output_hypernetworks_mlps.{m}."
            hw_dict = {
                'l0_w': self._alloc_param(sd[hp + "proj_in.weight"].to(torch.bfloat16)),
                'l0_b': self._alloc_param(sd[hp + "proj_in.bias"].to(torch.bfloat16)),
                'l1_w': self._alloc_param(sd[hp + "layers.0.weight"].to(torch.bfloat16)),
                'l1_b': self._alloc_param(sd[hp + "layers.0.bias"].to(torch.bfloat16)),
            }
            w_raw = sd[hp + "proj_out.weight"].to(torch.bfloat16)  # (32, 256)
            w_pad = torch.zeros(64, 256, dtype=torch.bfloat16)
            w_pad[:32, :] = w_raw
            hw_dict['l2_w'] = self._alloc_param(w_pad)
            b_raw = sd[hp + "proj_out.bias"].to(torch.bfloat16)
            b_pad = torch.zeros(64, dtype=torch.bfloat16)
            b_pad[:32] = b_raw
            hw_dict['l2_b'] = self._alloc_param(b_pad)
            self.dec_hyper_weights.append(hw_dict)

        # IoU prediction head: proj_in + layers.0 + proj_out (256→256→256→4)
        ip = md + "iou_prediction_head."
        self.dec_iou_weights = {
            'l0_w': self._alloc_param(sd[ip + "proj_in.weight"].to(torch.bfloat16)),
            'l0_b': self._alloc_param(sd[ip + "proj_in.bias"].to(torch.bfloat16)),
            'l1_w': self._alloc_param(sd[ip + "layers.0.weight"].to(torch.bfloat16)),
            'l1_b': self._alloc_param(sd[ip + "layers.0.bias"].to(torch.bfloat16)),
        }
        w_raw = sd[ip + "proj_out.weight"].to(torch.bfloat16)  # (4, 256)
        w_pad = torch.zeros(64, 256, dtype=torch.bfloat16)
        w_pad[:4, :] = w_raw
        self.dec_iou_weights['l2_w'] = self._alloc_param(w_pad)
        b_raw = sd[ip + "proj_out.bias"].to(torch.bfloat16)
        b_pad = torch.zeros(64, dtype=torch.bfloat16)
        b_pad[:4] = b_raw
        self.dec_iou_weights['l2_b'] = self._alloc_param(b_pad)

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
        # Rel-pos bias buffer for local blocks: (300, 256, 256) = 19,660,800 elements ≈ 39.3 MB
        # Per-batch bias (bias_shared=False) after window mask is broadcast in.
        self.VIT_REL_POS_BIAS = self._alloc_tensor(total_heads_local * wa_pad * wa_pad)
        # Rel-pos bias buffer for global blocks: (4096, 4096) = 16,777,216 elements ≈ 33.6 MB
        # One head at a time; zeroed and filled before each flash_attention_core call.
        self.VIT_GLOBAL_BIAS = self._alloc_tensor(GA * GA)
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
        DD   = self.DEC_DIM          # 256
        NT   = self.DEC_NUM_TOKENS   # 7
        DH   = self.DEC_HEADS        # 8
        HD   = self.DEC_HEAD_DIM     # 32
        HDP  = 64                    # padded head_dim for self-attn
        self.DEC_TOKENS      = self._alloc_tensor(NT * DD)   # (7, 256) prompt + output tokens
        self.DEC_SRC         = self._alloc_tensor(GA * DD)   # (4096, 256) image src
        self.DEC_TOKENS_NORM = self._alloc_tensor(NT * DD)   # (7, 256) LN scratch
        self.DEC_Q           = self._alloc_tensor(NT * DH * 64)  # (7, 512) padded proj Q
        self.DEC_K           = self._alloc_tensor(NT * DH * 64)  # (7, 512) padded proj K
        self.DEC_V           = self._alloc_tensor(NT * DH * 64)  # (7, 512) padded proj V
        # Self-attn multihead buffers.
        # Hardware requires seq_len multiple of UE_VECTOR_SIZE=64. NT=7 → NT_PAD=64.
        # Q stays (8, 7, 32) — q_len=7 is the query side, no kv padding needed.
        # K/V padded to (8, 64, 32) — kv positions 7:64 are zero, cancelled by -inf bias.
        NT_PAD = 64
        HD_PAD = 64  # head_dim padded to 64 for bf16_permute_core
        # Q/K/V/OUT all (8, 64, 64) — Q seq also padded to 64 to use flash_attention_batched
        self.DEC_SA_Q_HEADS   = self._alloc_tensor(DH * NT_PAD * HD_PAD)   # (8, 64, 64) = 32768
        self.DEC_SA_K_HEADS   = self._alloc_tensor(DH * NT_PAD * HD_PAD)   # (8, 64, 64) = 32768
        self.DEC_SA_V_HEADS   = self._alloc_tensor(DH * NT_PAD * HD_PAD)   # (8, 64, 64) = 32768
        self.DEC_SA_OUT_HEADS = self._alloc_tensor(DH * NT_PAD * HD_PAD)   # (8, 64, 64) = 32768
        # flash_attention_batched scratch: (head_dim*seq_len + seq_len*seq_len) per head
        self.DEC_SA_SCRATCH   = self._alloc_tensor(DH * (HD_PAD * NT_PAD + NT_PAD * NT_PAD))  # 65536
        self.DEC_SA_TEMP      = self._alloc_tensor(DH * NT * HD_PAD)  # compact (8,7,64) for scatter
        self.DEC_SA_MERGED    = self._alloc_tensor(NT * DD)            # merged (7, 256)

        # Cross-attn positional encodings (assembled in hardware by the prompt encoder phase)
        self.DEC_QUERY_PE    = self._alloc_tensor(NT * DD)             # (7, 256) token PE
        self.DEC_KEY_PE      = self._alloc_tensor(GA * DD)             # (4096, 256) dense image PE

        # Prompt encoder PE scratch: 4 × 128 bf16 slots for host-selected rows
        # Host writes SIN_X[x], COS_X[x], SIN_Y[y], COS_Y[y] before execution.
        self.PE_SX_SLOT = self._alloc_tensor(128)   # SIN_X[x_px, :]
        self.PE_CX_SLOT = self._alloc_tensor(128)   # COS_X[x_px, :]
        self.PE_SY_SLOT = self._alloc_tensor(128)   # SIN_Y[y_px, :]
        self.PE_CY_SLOT = self._alloc_tensor(128)   # COS_Y[y_px, :]
        # Intermediate PE computation buffers (128 elements each)
        self.PE_TMP1    = self._alloc_tensor(128)   # sin(A)*cos(B)
        self.PE_TMP2    = self._alloc_tensor(128)   # cos(A)*sin(B)
        self.PE_TMP3    = self._alloc_tensor(128)   # cos(A)*cos(B)
        self.PE_TMP4    = self._alloc_tensor(128)   # sin(A)*sin(B)
        self.PE_SIN_OUT = self._alloc_tensor(128)   # pe_sin = sin(A+B)
        self.PE_COS_OUT = self._alloc_tensor(128)   # pe_cos = cos(A+B)
        self.PE_FG_TOK  = self._alloc_tensor(256)   # final fg_token (pe_sin||pe_cos + point_embed_1)

        # t2i cross-attn buffers
        # K and V projections for image side: (4096, 512) each
        CA_HDP = 64   # head_dim padded
        CA_ID  = self.DEC_INTERNAL_DIM   # 128
        self.DEC_CA_T2I_K      = self._alloc_tensor(GA * DH * CA_HDP)  # (4096, 512)
        self.DEC_CA_T2I_V      = self._alloc_tensor(GA * DH * CA_HDP)  # (4096, 512)
        self.DEC_CA_T2I_K_HEADS = self._alloc_tensor(DH * GA * CA_HDP)  # (8, 4096, 64)
        self.DEC_CA_T2I_V_HEADS = self._alloc_tensor(DH * GA * CA_HDP)  # (8, 4096, 64)
        # Per-head scratch (reused across heads sequentially)
        self.DEC_CA_SCORES     = self._alloc_tensor(NT * GA)            # (7, 4096) per-head scores
        self.DEC_CA_VT         = self._alloc_tensor(CA_HDP * GA)        # (64, 4096) per-head V^T
        # Per-head output accumulator: (8, 7, 64)
        self.DEC_CA_OUT_HEADS  = self._alloc_tensor(DH * NT * CA_HDP)   # (8, 7, 64)
        # Merged output after unpad: (7, 128) — fits inside DEC_SA_MERGED
        self.DEC_CA_T2I_MERGED = self._alloc_tensor(NT * CA_ID)         # (7, 128)

        # i2t cross-attn buffers (image attends to tokens; token side padded to NT_PAD_I=64)
        NT_PAD_I = 64
        self.DEC_CA_I2T_K_PAD    = self._alloc_tensor(NT_PAD_I * DH * CA_HDP)   # (64, 512) padded K proj
        self.DEC_CA_I2T_V_PAD    = self._alloc_tensor(NT_PAD_I * DH * CA_HDP)   # (64, 512) padded V proj
        self.DEC_CA_I2T_SCORES   = self._alloc_tensor(GA * NT_PAD_I)             # (4096, 64) per-head scores
        self.DEC_CA_I2T_VT       = self._alloc_tensor(CA_HDP * NT_PAD_I)         # (64, 64) per-head V^T
        self.DEC_CA_I2T_OUT_HEADS = self._alloc_tensor(DH * GA * CA_HDP)         # (8, 4096, 64) = 2M
        self.DEC_CA_I2T_MERGED   = self._alloc_tensor(GA * CA_ID)                # (4096, 128)

        # MLP intermediate: (7, 2048)
        MLP_DIM = self.DEC_MLP_DIM  # 2048
        self.DEC_MLP_INTER = self._alloc_tensor(NT * MLP_DIM)                    # (7, 2048)

        # Mask upscaling scratch and output buffers
        # ConvTranspose2d 0: (4096, 256) → (16384, 64); scratch holds 4 sub-position results
        self.DEC_UP0_SCRATCH = self._alloc_tensor(4 * GA * 64)                   # (16384, 64)
        self.DEC_UP0_OUT     = self._alloc_tensor(4 * GA * 64)                   # (16384, 64) pixel-shuffled

        # ConvTranspose2d 1: (16384, 64) → (65536, 32); scratch holds 4 sub-position results
        # Padded to N=64 lanes for hardware alignment (only first 32 channels real).
        self.DEC_UP1_SCRATCH = self._alloc_tensor(4 * 16384 * 64)                # (65536, 64)
        self.DEC_UP1_OUT     = self._alloc_tensor(65536 * 64)                    # (65536, 64) pixel-shuffled

        # Hypernetwork MLP outputs: padded to (64, 64); only first 4 rows × 32 cols real
        NUM_MASKS = 4
        self.DEC_HYPER_OUT   = self._alloc_tensor(64 * 64)                       # (64, 64)

        # IoU head output: padded to 64 lanes; only first 4 real
        self.DEC_IOU_OUT     = self._alloc_tensor(64)                            # (64,) iou scores

        # Final mask logits: padded to (65536, 64); only first 4 cols per row real
        self.DEC_MASK_LOGITS = self._alloc_tensor(65536 * 64)                    # (65536, 64)

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

        _t_compile_start = _time.perf_counter()
        try:
            self._compile_phases(pad, bpe)
        except (_CheckpointStop, AssertionError) as e:
            _original_print(f"  Compile stopped at checkpoint: {e}")
        _original_print(f"  [compile] total: {_time.perf_counter() - _t_compile_start:.3f}s")
        self._finalize_program()
        _SILENT_MODE = False
        return self._last_prog_addr

    def _compile_phases(self, pad, bpe):
        """All compile phases live here."""

        VD    = self.VIT_DIM         # 768
        VD_QK = self.VIT_QK_DIM_PAD  # 768 (no head_dim padding)
        hd    = self.VIT_HEAD_DIM_PAD # 64
        bpe   = 2

        _tp = _time.perf_counter()

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

        _original_print(f"  [compile] patch-embed: {_time.perf_counter() - _tp:.3f}s"); _tp = _time.perf_counter()

        # ==============================================================
        # PHASE 2: ViT BLOCKS (12 blocks)
        # Local blocks (0-1, 3-4, 6-7, 9-10): window=14, 25 windows
        # Global blocks (2, 5, 8, 11):         full seq=4096
        # No RoPE — SAM1 uses decomposed 2D rel_pos (implemented via Rh/Rw tables + matmul).
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

            # --- Flash attention (with decomposed rel_pos bias for local blocks) ---
            if is_global:
                # Global seq=4096: compute decomposed rel_pos bias into VIT_GLOBAL_BIAS,
                # then run flash attention with the bias. Now fits within the expanded
                # ~97M-instruction program budget (was skipped under the old 52.4M cap).
                # Zero-fill the global bias buffer (4096 × 4096).
                dram_zero_fill(self, self.VIT_GLOBAL_BIAS, self.GRID_AREA * self.GRID_AREA)
                # Compute decomposed rel_pos bias for all 12 heads into VIT_GLOBAL_BIAS.
                # num_batches=VIT_HEADS=12 (one per head; no windows for global).
                # sz=64 (64×64 global grid), sz_pad=64, seq_pad=4096.
                # VIT_COMPACT_HEADS is free at this point (used as scratch after QKV permute).
                add_rel_pos_bias_dram(self,
                    Q_DRAM_ADDR=self.VIT_Q_HEADS,
                    RH_DRAM_ADDR=bw['rel_pos_h'],
                    RW_DRAM_ADDR=bw['rel_pos_w'],
                    BIAS_DRAM_ADDR=self.VIT_GLOBAL_BIAS,
                    ONES_SCATTER_ADDR=self.ONES_SCATTER_GLOBAL,
                    sz=64,
                    sz_pad=64,
                    num_batches=self.VIT_HEADS,   # 12
                    seq_pad=self.GRID_AREA,       # 4096
                    SCRATCH_DRAM_ADDR=self.VIT_COMPACT_HEADS,
                )
                flash_attention_global_tiled(self,
                    num_heads=self.VIT_HEADS,
                    head_dim=hd,
                    seq_len=seq_len,
                    Q_DRAM_ADDR=self.VIT_Q_HEADS,
                    K_DRAM_ADDR=self.VIT_K_HEADS,
                    V_DRAM_ADDR=self.VIT_V_HEADS,
                    OUTPUT_DRAM_ADDR=self.VIT_ATTN_OUT,
                    SCRATCH_DRAM_ADDR=self.VIT_ATTN_SCRATCH,
                    RH_DRAM_ADDR=bw['rel_pos_h'],
                    RW_DRAM_ADDR=bw['rel_pos_w'],
                    ONES_SCATTER_ADDR=self.ONES_SCATTER_GLOBAL,
                    BIAS_DRAM_ADDR=self.VIT_GLOBAL_BIAS,
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
                # Initialize per-batch bias: zero-fill then broadcast window mask into each batch.
                # Window mask sets key positions wa:wa_pad to -inf to suppress padding tokens.
                dram_zero_fill(self, self.VIT_REL_POS_BIAS, total_batches * wa_pad * wa_pad)
                for _b in range(total_batches):
                    dram_copy(self,
                        self.VIT_WINDOW_ATTN_BIAS,
                        self.VIT_REL_POS_BIAS + _b * wa_pad * wa_pad * bpe,
                        wa_pad * wa_pad)
                # Add decomposed rel_pos bias into each batch's bias slice.
                # After QKV permute, VIT_COMPACT_HEADS is free to use as scratch.
                add_rel_pos_bias_dram(self,
                    Q_DRAM_ADDR=self.VIT_Q_HEADS,
                    RH_DRAM_ADDR=bw['rel_pos_h'],
                    RW_DRAM_ADDR=bw['rel_pos_w'],
                    BIAS_DRAM_ADDR=self.VIT_REL_POS_BIAS,
                    ONES_SCATTER_ADDR=self.ONES_SCATTER_LOCAL,
                    sz=self.VIT_WINDOW_SIZE,       # 14
                    sz_pad=64,
                    num_batches=total_batches,     # 300
                    seq_pad=wa_pad,                # 256
                    SCRATCH_DRAM_ADDR=self.VIT_COMPACT_HEADS,
                )
                flash_attention_batched(self,
                    num_batches=total_batches,  # 300
                    head_dim=hd,
                    seq_len=wa_pad,             # 256 (64-aligned)
                    Q_DRAM_ADDR=self.VIT_Q_HEADS,
                    K_DRAM_ADDR=self.VIT_K_HEADS,
                    V_DRAM_ADDR=self.VIT_V_HEADS,
                    OUTPUT_DRAM_ADDR=self.VIT_ATTN_OUT,
                    SCRATCH_DRAM_ADDR=self.VIT_ATTN_SCRATCH,
                    BIAS_DRAM_ADDR=self.VIT_REL_POS_BIAS,
                    bias_shared=False,
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

            if hasattr(self, '_vit_checkpoint') and blk_idx == self._vit_checkpoint:
                raise _CheckpointStop(f"after ViT blocks 0-{blk_idx}")

        # ViT output: VIT_LAYER_IN contains (4096, 768) = (64, 64, 768)
        _original_print(f"  [compile] ViT blocks (12): {_time.perf_counter() - _tp:.3f}s"); _tp = _time.perf_counter()

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
        _original_print(f"  [compile] neck: {_time.perf_counter() - _tp:.3f}s"); _tp = _time.perf_counter()

        # ==============================================================
        # PHASE 3b: PROMPT ENCODER — assemble DEC_TOKENS in hardware
        #
        # Inputs (written by host before execution):
        #   PE_SX_SLOT: SIN_X[x_px, :]  (128 bf16)
        #   PE_CX_SLOT: COS_X[x_px, :]  (128 bf16)
        #   PE_SY_SLOT: SIN_Y[y_px, :]  (128 bf16)
        #   PE_CY_SLOT: COS_Y[y_px, :]  (128 bf16)
        #
        # Computes fg_token = [sin(A+B), cos(A+B)] + PE_POINT_EMBED_FG
        # Assembles DEC_TOKENS rows:
        #   [0]   = iou_token
        #   [1:5] = mask_tokens  (4 rows × 256)
        #   [5]   = fg_token
        #   [6]   = not_a_point_embed (no PE for padding token)
        # Then: DEC_QUERY_PE = DEC_TOKENS
        #       DEC_KEY_PE   = DEC_POS_SRC
        # ==============================================================
        _DD = self.DEC_DIM   # 256

        # Step 1: sin(A+B) = SIN_X[x]*COS_Y[y] + COS_X[x]*SIN_Y[y]
        eltwise_mul_core_dram(self, 128, self.PE_SX_SLOT, self.PE_CY_SLOT, self.PE_TMP1)
        eltwise_mul_core_dram(self, 128, self.PE_CX_SLOT, self.PE_SY_SLOT, self.PE_TMP2)
        eltwise_add_dram(self, self.PE_TMP1, self.PE_TMP2, self.PE_SIN_OUT, 128)

        # Step 2: cos(A+B) = COS_X[x]*COS_Y[y] - SIN_X[x]*SIN_Y[y]
        eltwise_mul_core_dram(self, 128, self.PE_CX_SLOT, self.PE_CY_SLOT, self.PE_TMP3)
        eltwise_mul_core_dram(self, 128, self.PE_SX_SLOT, self.PE_SY_SLOT, self.PE_TMP4)
        broadcast_mul_dram(self, self.PE_TMP4, -1.0, 128)
        eltwise_add_dram(self, self.PE_TMP3, self.PE_TMP4, self.PE_COS_OUT, 128)

        # Step 3: assemble pe = [pe_sin, pe_cos] into PE_FG_TOK, then add point_embed.1.weight (FG)
        dram_copy(self, self.PE_SIN_OUT, self.PE_FG_TOK,           128)
        dram_copy(self, self.PE_COS_OUT, self.PE_FG_TOK + 128*bpe, 128)
        eltwise_add_dram(self, self.PE_FG_TOK, self.PE_POINT_EMBED_FG, self.PE_FG_TOK, _DD)

        # Step 4: assemble DEC_TOKENS[7, 256] from params + fg_token
        dram_copy(self, self.DEC_IOU_TOKEN,   self.DEC_TOKENS + 0*_DD*bpe, _DD)       # row 0
        dram_copy(self, self.DEC_MASK_TOKENS, self.DEC_TOKENS + 1*_DD*bpe, 4*_DD)     # rows 1-4
        dram_copy(self, self.PE_FG_TOK,       self.DEC_TOKENS + 5*_DD*bpe, _DD)       # row 5
        dram_copy(self, self.PE_NOT_A_POINT,  self.DEC_TOKENS + 6*_DD*bpe, _DD)       # row 6 (NO PE)

        # Step 5: DEC_QUERY_PE = DEC_TOKENS (tokens serve as their own PE in cross-attn)
        dram_copy(self, self.DEC_TOKENS, self.DEC_QUERY_PE, self.DEC_NUM_TOKENS * _DD)

        # Step 6: DEC_KEY_PE = DEC_POS_SRC (precomputed at weight_init)
        dram_copy(self, self.DEC_POS_SRC, self.DEC_KEY_PE, self.GRID_AREA * _DD)

        _original_print(f"  [compile] prompt encoder: {_time.perf_counter() - _tp:.3f}s"); _tp = _time.perf_counter()

        # ==============================================================
        # PHASE 4: MASK DECODER — layer 0, self-attention
        #
        # src = NECK_OUT + no_mask_embed (broadcast) → DEC_SRC
        # tokens = DMA'd externally (cpu_tokens) → DEC_TOKENS
        #
        # Layer 0: skip_first_layer_pe=True → q=k=v=tokens (no PE added)
        #   Q/K/V proj (7,256)→(7,256), reshape→(8,7,64)
        #   cross_attention_batched, merge, out_proj, residual, norm1 → DEC_TOKENS
        # ==============================================================
        DD     = self.DEC_DIM        # 256
        NT     = self.DEC_NUM_TOKENS # 7
        NT_PAD = 64                  # next multiple of UE_VECTOR_SIZE — required for HW attn
        DH     = self.DEC_HEADS      # 8
        HD     = self.DEC_HEAD_DIM   # 32
        GA     = self.GRID_AREA      # 4096
        lw0    = self.dec_layer_weights[0]

        # Init DEC_SRC: copy NECK_OUT then add no_mask_embed (broadcast add)
        dram_copy(self, self.NECK_OUT, self.DEC_SRC, self.GRID_AREA * DD)
        for row in range(0, self.GRID_AREA, URAM_NEAR_FULL_ELEMENTS // DD):
            take = min(URAM_NEAR_FULL_ELEMENTS // DD, self.GRID_AREA - row)
            self.accelerator_memory_to_sram(
                accelerator_dram_address=self.PE_NO_MASK,
                sram_address=0x80000, element_size=DD)
            for r in range(take):
                self.accelerator_memory_to_sram(
                    accelerator_dram_address=self.DEC_SRC + (row + r) * DD * 2,
                    sram_address=0x00000, element_size=DD)
                self.eltwise_add_core(
                    vector_A_sram_start_addr=0x00000,
                    vector_B_sram_start_addr=0x80000,
                    vector_C_sram_wb_addr=0x00000,
                    element_size=DD)
                self.sram_to_accelerator_memory(
                    sram_address=0x00000,
                    accelerator_dram_address=self.DEC_SRC + (row + r) * DD * 2,
                    element_size=DD)

        # Layer 0 self-attn: skip_first_layer_pe → q=k=v=tokens (no PE)
        # Projections output (7, 512) = (7, DH*HD_PAD) — weights pre-padded in weight_init
        # Q weight also pre-scaled by sqrt(2) so cross_attention_batched's 1/sqrt(64)=1/sqrt(32)
        N_PAD = DH * 64  # 512
        self.matmat_mul_core(
            M=NT, K=DD, N=N_PAD,
            A_DRAM_ADDR=self.DEC_TOKENS,
            B_DRAM_ADDR=lw0['sa_q_w'],
            OUTPUT_DRAM_ADDR=self.DEC_Q,
            C_DRAM_ADDR=lw0['sa_q_b'],
            bias_mode="broadcast_N",
        )
        self.matmat_mul_core(
            M=NT, K=DD, N=N_PAD,
            A_DRAM_ADDR=self.DEC_TOKENS,
            B_DRAM_ADDR=lw0['sa_k_w'],
            OUTPUT_DRAM_ADDR=self.DEC_K,
            C_DRAM_ADDR=lw0['sa_k_b'],
            bias_mode="broadcast_N",
        )
        self.matmat_mul_core(
            M=NT, K=DD, N=N_PAD,
            A_DRAM_ADDR=self.DEC_TOKENS,
            B_DRAM_ADDR=lw0['sa_v_w'],
            OUTPUT_DRAM_ADDR=self.DEC_V,
            C_DRAM_ADDR=lw0['sa_v_b'],
            bias_mode="broadcast_N",
        )
        HD_PAD = 64
        bpe = 2
        # Permute Q/K/V: (7,8,64) → compact (8,7,64) in TEMP, zero-fill (8,64,64), scatter
        for proj, heads in [(self.DEC_Q, self.DEC_SA_Q_HEADS),
                            (self.DEC_K, self.DEC_SA_K_HEADS),
                            (self.DEC_V, self.DEC_SA_V_HEADS)]:
            self.bf16_permute_core(
                dim_0=NT, dim_1=DH, dim_2=HD_PAD,
                INPUT_DRAM_ADDR=proj,
                OUTPUT_DRAM_ADDR=self.DEC_SA_TEMP,
            )
            dram_zero_fill(self, heads, DH * NT_PAD * HD_PAD)
            for h in range(DH):
                dram_copy(self,
                    self.DEC_SA_TEMP + h * NT * HD_PAD * bpe,
                    heads              + h * NT_PAD * HD_PAD * bpe,
                    NT * HD_PAD)

        # Pre-scale Q by sqrt(2): flash_attention_batched scales by 1/sqrt(64),
        # true scale is 1/sqrt(32). sqrt(2)/sqrt(64) = 1/sqrt(32). ✓
        broadcast_mul_dram(self, self.DEC_SA_Q_HEADS, math.sqrt(2.0), DH * NT_PAD * HD_PAD)

        # Self-attention via flash_attention_batched (HW primitive, seq_len=64 square)
        # bias (8,64,64): -inf at kv positions [7:64] cancels zero-padded keys
        flash_attention_batched(self,
            num_batches=DH, head_dim=HD_PAD, seq_len=NT_PAD,
            Q_DRAM_ADDR=self.DEC_SA_Q_HEADS,
            K_DRAM_ADDR=self.DEC_SA_K_HEADS,
            V_DRAM_ADDR=self.DEC_SA_V_HEADS,
            OUTPUT_DRAM_ADDR=self.DEC_SA_OUT_HEADS,
            SCRATCH_DRAM_ADDR=self.DEC_SA_SCRATCH,
            BIAS_DRAM_ADDR=self.DEC_SA_ATTN_BIAS,
        )
        # Extract first NT=7 rows per head from (8,64,64) output into compact (8,7,64)
        for h in range(DH):
            dram_copy(self,
                self.DEC_SA_OUT_HEADS + h * NT_PAD * HD_PAD * bpe,
                self.DEC_SA_TEMP      + h * NT * HD_PAD * bpe,
                NT * HD_PAD)
        # Merge: (8,7,64) → (7,256) with head_dim unpad 64→32
        multihead_merge_dram(self,
            INPUT_DRAM_ADDR=self.DEC_SA_TEMP,
            OUTPUT_DRAM_ADDR=self.DEC_SA_MERGED,
            TEMP_DRAM_ADDR=self.DEC_SA_Q_HEADS,
            seq_len=NT, num_heads=DH,
            head_dim=HD, head_dim_pad=HD_PAD,
            UNPAD_WEIGHT_ADDR=self.DEC_SA_UNPAD_W,
        )

        # out_proj: (7,256) @ (256,256)^T + bias → DEC_TOKENS_NORM (scratch)
        self.matmat_mul_core(
            M=NT, K=DD, N=DD,
            A_DRAM_ADDR=self.DEC_SA_MERGED,
            B_DRAM_ADDR=lw0['sa_out_w'],
            OUTPUT_DRAM_ADDR=self.DEC_TOKENS_NORM,
            C_DRAM_ADDR=lw0['sa_out_b'],
            bias_mode="broadcast_N",
        )
        # residual: DEC_TOKENS += out_proj result
        eltwise_add_dram(self, self.DEC_TOKENS, self.DEC_TOKENS_NORM, self.DEC_TOKENS, NT * DD)
        # norm1: LayerNorm in-place on DEC_TOKENS
        self.layer_norm_core_dram(
            M=NT, N=DD,
            A_DRAM_ADDR=self.DEC_TOKENS,
            OUTPUT_DRAM_ADDR=self.DEC_TOKENS,
            GAMMA_DRAM_ADDR=lw0['norm1_w'],
            BETA_DRAM_ADDR=lw0['norm1_b'],
        )

        # ── Layer 0: t2i cross-attention ───────────────────────────────────
        # q = tokens + query_pe, k = src + key_pe, v = src (no PE)
        # internal_dim=128, num_heads=8, head_dim=16 padded to 64
        CA_HDP = 64
        CA_ID  = self.DEC_INTERNAL_DIM   # 128
        CA_HD  = self.DEC_INTERNAL_HD    # 16
        N_CA   = DH * CA_HDP             # 512

        # Q: (tokens + query_pe) → project → (7, 512)
        eltwise_add_dram(self, self.DEC_TOKENS, self.DEC_QUERY_PE,
                         self.DEC_TOKENS_NORM, NT * DD)
        self.matmat_mul_core(
            M=NT, K=DD, N=N_CA,
            A_DRAM_ADDR=self.DEC_TOKENS_NORM,
            B_DRAM_ADDR=lw0['ca_t2i_q_w'],
            OUTPUT_DRAM_ADDR=self.DEC_Q,
            C_DRAM_ADDR=lw0['ca_t2i_q_b'],
            bias_mode="broadcast_N",
        )

        # K: (src + key_pe) @ W_k^T = src @ W_k^T + key_pe @ W_k^T
        # Use DEC_CA_T2I_V as temp for the key_pe partial result
        self.matmat_mul_core(
            M=GA, K=DD, N=N_CA,
            A_DRAM_ADDR=self.DEC_SRC,
            B_DRAM_ADDR=lw0['ca_t2i_k_w'],
            OUTPUT_DRAM_ADDR=self.DEC_CA_T2I_K,
            C_DRAM_ADDR=lw0['ca_t2i_k_b'],
            bias_mode="broadcast_N",
        )
        self.matmat_mul_core(
            M=GA, K=DD, N=N_CA,
            A_DRAM_ADDR=self.DEC_POS_SRC,
            B_DRAM_ADDR=lw0['ca_t2i_k_w'],
            OUTPUT_DRAM_ADDR=self.DEC_CA_T2I_V,  # temp
        )
        eltwise_add_dram(self, self.DEC_CA_T2I_K, self.DEC_CA_T2I_V,
                         self.DEC_CA_T2I_K, GA * N_CA)

        # V: src (no PE) → project → (4096, 512)
        self.matmat_mul_core(
            M=GA, K=DD, N=N_CA,
            A_DRAM_ADDR=self.DEC_SRC,
            B_DRAM_ADDR=lw0['ca_t2i_v_w'],
            OUTPUT_DRAM_ADDR=self.DEC_CA_T2I_V,
            C_DRAM_ADDR=lw0['ca_t2i_v_b'],
            bias_mode="broadcast_N",
        )

        # Reshape to heads: (seq, 8, 64) → (8, seq, 64) via permute
        # Q: (7, 8, 64) → (8, 7, 64) in DEC_SA_TEMP
        self.bf16_permute_core(
            dim_0=NT, dim_1=DH, dim_2=CA_HDP,
            INPUT_DRAM_ADDR=self.DEC_Q,
            OUTPUT_DRAM_ADDR=self.DEC_SA_TEMP,
        )
        # K: (4096, 8, 64) → (8, 4096, 64) in DEC_CA_T2I_K_HEADS
        self.bf16_permute_core(
            dim_0=GA, dim_1=DH, dim_2=CA_HDP,
            INPUT_DRAM_ADDR=self.DEC_CA_T2I_K,
            OUTPUT_DRAM_ADDR=self.DEC_CA_T2I_K_HEADS,
        )
        # V: (4096, 8, 64) → (8, 4096, 64) in DEC_CA_T2I_V_HEADS
        self.bf16_permute_core(
            dim_0=GA, dim_1=DH, dim_2=CA_HDP,
            INPUT_DRAM_ADDR=self.DEC_CA_T2I_V,
            OUTPUT_DRAM_ADDR=self.DEC_CA_T2I_V_HEADS,
        )

        # Pre-scale Q by 1/sqrt(CA_HD=16): padded zeros don't contribute to dot products
        broadcast_mul_dram(self, self.DEC_SA_TEMP,
                           1.0 / math.sqrt(CA_HD), DH * NT * CA_HDP)

        # Per-head: scores = softmax(Q_h @ K_h^T), out_h = scores @ V_h
        for h in range(DH):
            q_addr = self.DEC_SA_TEMP + h * NT * CA_HDP * bpe
            k_addr = self.DEC_CA_T2I_K_HEADS + h * GA * CA_HDP * bpe
            v_addr = self.DEC_CA_T2I_V_HEADS + h * GA * CA_HDP * bpe
            o_addr = self.DEC_CA_OUT_HEADS + h * NT * CA_HDP * bpe

            # scores: (7, 64) @ (64, 4096)^T → (7, 4096), softmax row-wise
            self.matmat_mul_core(
                M=NT, K=CA_HDP, N=GA,
                A_DRAM_ADDR=q_addr,
                B_DRAM_ADDR=k_addr,
                OUTPUT_DRAM_ADDR=self.DEC_CA_SCORES,
                softmax_enable=True,
            )
            # V^T: (4096, 64) → (64, 4096)
            self.bf16_transpose_core(
                M=GA, N=CA_HDP,
                INPUT_DRAM_ADDR=v_addr,
                OUTPUT_DRAM_ADDR=self.DEC_CA_VT,
            )
            # out: (7, 4096) @ (4096, 64)^T → (7, 4096) @ (4096, 64) using V^T as B
            # matmat_mul_core computes A @ B^T where B is (N, K)
            # B = V^T stored as (N=64, K=4096) → B^T = V → out = scores @ V ✓
            self.matmat_mul_core(
                M=NT, K=GA, N=CA_HDP,
                A_DRAM_ADDR=self.DEC_CA_SCORES,
                B_DRAM_ADDR=self.DEC_CA_VT,
                OUTPUT_DRAM_ADDR=o_addr,
            )

        # Merge: (8, 7, 64) → (7, 128) with head_dim unpad 64→16
        multihead_merge_dram(self,
            INPUT_DRAM_ADDR=self.DEC_CA_OUT_HEADS,
            OUTPUT_DRAM_ADDR=self.DEC_CA_T2I_MERGED,
            TEMP_DRAM_ADDR=self.DEC_SA_Q_HEADS,
            seq_len=NT, num_heads=DH,
            head_dim=CA_HD, head_dim_pad=CA_HDP,
            UNPAD_WEIGHT_ADDR=self.DEC_CA_UNPAD_W,
        )

        # out_proj: (7, 128) → (7, 256) + residual + norm2
        self.matmat_mul_core(
            M=NT, K=CA_ID, N=DD,
            A_DRAM_ADDR=self.DEC_CA_T2I_MERGED,
            B_DRAM_ADDR=lw0['ca_t2i_out_w'],
            OUTPUT_DRAM_ADDR=self.DEC_TOKENS_NORM,
            C_DRAM_ADDR=lw0['ca_t2i_out_b'],
            bias_mode="broadcast_N",
        )
        eltwise_add_dram(self, self.DEC_TOKENS, self.DEC_TOKENS_NORM,
                         self.DEC_TOKENS, NT * DD)
        self.layer_norm_core_dram(
            M=NT, N=DD,
            A_DRAM_ADDR=self.DEC_TOKENS,
            OUTPUT_DRAM_ADDR=self.DEC_TOKENS,
            GAMMA_DRAM_ADDR=lw0['norm2_w'],
            BETA_DRAM_ADDR=lw0['norm2_b'],
        )

        # ── Helper: i2t cross-attention (image attends to tokens) ──────────────
        # Shared constants already defined above: CA_HDP=64, CA_ID=128, CA_HD=16, N_CA=512
        NT_PAD_I = 64   # token sequence padded to 64 for matmat_mul_core N-alignment
        MLP_DIM  = self.DEC_MLP_DIM  # 2048

        def _compile_i2t(lw):
            """Compile one image→token cross-attn block + residual + norm4.
            Updates DEC_SRC in-place. Uses lw for the current layer weights."""
            # Q: (src + key_pe) @ W_q_i2t → (4096, 512)  [stored temporarily in DEC_CA_T2I_K]
            eltwise_add_dram(self, self.DEC_SRC, self.DEC_POS_SRC,
                             self.DEC_CA_T2I_K, GA * DD)
            self.matmat_mul_core(
                M=GA, K=DD, N=N_CA,
                A_DRAM_ADDR=self.DEC_CA_T2I_K,
                B_DRAM_ADDR=lw['ca_i2t_q_w'],
                OUTPUT_DRAM_ADDR=self.DEC_CA_T2I_V,   # temp for Q proj
                C_DRAM_ADDR=lw['ca_i2t_q_b'],
                bias_mode="broadcast_N",
            )

            # K: (tokens + query_pe) @ W_k_i2t → (7, 512), zero-padded to (64, 512)
            eltwise_add_dram(self, self.DEC_TOKENS, self.DEC_QUERY_PE,
                             self.DEC_TOKENS_NORM, NT * DD)
            self.matmat_mul_core(
                M=NT, K=DD, N=N_CA,
                A_DRAM_ADDR=self.DEC_TOKENS_NORM,
                B_DRAM_ADDR=lw['ca_i2t_k_w'],
                OUTPUT_DRAM_ADDR=self.DEC_CA_I2T_K_PAD,
                C_DRAM_ADDR=lw['ca_i2t_k_b'],
                bias_mode="broadcast_N",
            )
            # Zero-fill padding rows [7:64] in K_PAD: (NT_PAD_I - NT) * N_CA elements
            dram_zero_fill(self,
                self.DEC_CA_I2T_K_PAD + NT * N_CA * bpe,
                (NT_PAD_I - NT) * N_CA)

            # V: tokens @ W_v_i2t → (7, 512), zero-padded to (64, 512)
            self.matmat_mul_core(
                M=NT, K=DD, N=N_CA,
                A_DRAM_ADDR=self.DEC_TOKENS,
                B_DRAM_ADDR=lw['ca_i2t_v_w'],
                OUTPUT_DRAM_ADDR=self.DEC_CA_I2T_V_PAD,
                C_DRAM_ADDR=lw['ca_i2t_v_b'],
                bias_mode="broadcast_N",
            )
            dram_zero_fill(self,
                self.DEC_CA_I2T_V_PAD + NT * N_CA * bpe,
                (NT_PAD_I - NT) * N_CA)

            # Permute Q: (4096, 8, 64) → (8, 4096, 64) into DEC_CA_T2I_K_HEADS
            self.bf16_permute_core(
                dim_0=GA, dim_1=DH, dim_2=CA_HDP,
                INPUT_DRAM_ADDR=self.DEC_CA_T2I_V,      # Q proj lives here
                OUTPUT_DRAM_ADDR=self.DEC_CA_T2I_K_HEADS,
            )
            # Permute K: (64, 8, 64) → (8, 64, 64) into DEC_SA_K_HEADS (reused)
            self.bf16_permute_core(
                dim_0=NT_PAD_I, dim_1=DH, dim_2=CA_HDP,
                INPUT_DRAM_ADDR=self.DEC_CA_I2T_K_PAD,
                OUTPUT_DRAM_ADDR=self.DEC_SA_K_HEADS,
            )
            # Permute V: (64, 8, 64) → (8, 64, 64) into DEC_SA_V_HEADS (reused)
            self.bf16_permute_core(
                dim_0=NT_PAD_I, dim_1=DH, dim_2=CA_HDP,
                INPUT_DRAM_ADDR=self.DEC_CA_I2T_V_PAD,
                OUTPUT_DRAM_ADDR=self.DEC_SA_V_HEADS,
            )

            # Scale Q heads by 1/sqrt(CA_HD=16)
            broadcast_mul_dram(self, self.DEC_CA_T2I_K_HEADS,
                               1.0 / math.sqrt(CA_HD), DH * GA * CA_HDP)

            # Per-head attention: Q_h (4096,64) × K_h^T (64,64) → scores (4096,64) → softmax → × V_h
            # Zero-padded K/V rows [7:64] produce near-zero (not -inf) scores;
            # since V_h[7:64,:]=0 their contribution to the output is exactly zero.
            # Identity matrix for standalone row-wise softmax: softmax(S @ I^T) = softmax(S)
            _i2t_sm_id = torch.eye(NT_PAD_I, dtype=torch.bfloat16)
            _i2t_sm_id_w = self._alloc_param(_i2t_sm_id)
            for h in range(DH):
                q_addr = self.DEC_CA_T2I_K_HEADS + h * GA * CA_HDP * bpe
                k_addr = self.DEC_SA_K_HEADS      + h * NT_PAD_I * CA_HDP * bpe
                v_addr = self.DEC_SA_V_HEADS      + h * NT_PAD_I * CA_HDP * bpe
                o_addr = self.DEC_CA_I2T_OUT_HEADS + h * GA * CA_HDP * bpe

                # Step A: raw scores Q_h @ K_h^T → (GA, NT_PAD_I), no softmax
                self.matmat_mul_core(
                    M=GA, K=CA_HDP, N=NT_PAD_I,
                    A_DRAM_ADDR=q_addr,
                    B_DRAM_ADDR=k_addr,
                    OUTPUT_DRAM_ADDR=self.DEC_CA_I2T_SCORES,
                )
                # Step B: add kv-column bias row (0 at [0:7], -inf at [7:64]) to every
                # query row of the score matrix so softmax suppresses padded keys.
                # Load the 64-element bias row once into B-side SRAM, then for each of
                # the GA=4096 rows: load row → A, add → write back.
                self.accelerator_memory_to_sram(
                    accelerator_dram_address=self.DEC_CA_I2T_BIAS_ROW,
                    sram_address=0x80000,
                    element_size=NT_PAD_I,
                )
                for _row in range(GA):
                    _row_addr = self.DEC_CA_I2T_SCORES + _row * NT_PAD_I * bpe
                    self.accelerator_memory_to_sram(
                        accelerator_dram_address=_row_addr,
                        sram_address=0x00000,
                        element_size=NT_PAD_I,
                    )
                    self.eltwise_add_core(
                        vector_A_sram_start_addr=0x00000,
                        vector_B_sram_start_addr=0x80000,
                        vector_C_sram_wb_addr=0x00000,
                        element_size=NT_PAD_I,
                    )
                    self.sram_to_accelerator_memory(
                        sram_address=0x00000,
                        accelerator_dram_address=_row_addr,
                        element_size=NT_PAD_I,
                    )
                # Step C: standalone softmax on the bias-added scores.
                # Identity-matrix trick: softmax(scores @ I^T) = softmax(scores).
                # K=NT_PAD_I=64 satisfies the 64-ALU minimum (K=1 is invalid).
                self.matmat_mul_core(
                    M=GA, K=NT_PAD_I, N=NT_PAD_I,
                    A_DRAM_ADDR=self.DEC_CA_I2T_SCORES,
                    B_DRAM_ADDR=_i2t_sm_id_w,
                    OUTPUT_DRAM_ADDR=self.DEC_CA_I2T_SCORES,
                    softmax_enable=True,
                )
                self.bf16_transpose_core(
                    M=NT_PAD_I, N=CA_HDP,
                    INPUT_DRAM_ADDR=v_addr,
                    OUTPUT_DRAM_ADDR=self.DEC_CA_I2T_VT,
                )
                self.matmat_mul_core(
                    M=GA, K=NT_PAD_I, N=CA_HDP,
                    A_DRAM_ADDR=self.DEC_CA_I2T_SCORES,
                    B_DRAM_ADDR=self.DEC_CA_I2T_VT,
                    OUTPUT_DRAM_ADDR=o_addr,
                )

            # Merge: (8, 4096, 64) → permute → (4096, 512) → unpad → (4096, 128)
            # Use DEC_CA_T2I_K as the permute temp (GA*N_CA = 4096*512 elements, fits)
            multihead_merge_dram(self,
                INPUT_DRAM_ADDR=self.DEC_CA_I2T_OUT_HEADS,
                OUTPUT_DRAM_ADDR=self.DEC_CA_I2T_MERGED,
                TEMP_DRAM_ADDR=self.DEC_CA_T2I_K,
                seq_len=GA, num_heads=DH,
                head_dim=CA_HD, head_dim_pad=CA_HDP,
                UNPAD_WEIGHT_ADDR=self.DEC_CA_UNPAD_W,
            )

            # out_proj: (4096, 128) → (4096, 256), residual, norm4
            self.matmat_mul_core(
                M=GA, K=CA_ID, N=DD,
                A_DRAM_ADDR=self.DEC_CA_I2T_MERGED,
                B_DRAM_ADDR=lw['ca_i2t_out_w'],
                OUTPUT_DRAM_ADDR=self.DEC_TOKENS_NORM,
                C_DRAM_ADDR=lw['ca_i2t_out_b'],
                bias_mode="broadcast_N",
            )
            eltwise_add_dram(self, self.DEC_SRC, self.DEC_TOKENS_NORM,
                             self.DEC_SRC, GA * DD)
            self.layer_norm_core_dram(
                M=GA, N=DD,
                A_DRAM_ADDR=self.DEC_SRC,
                OUTPUT_DRAM_ADDR=self.DEC_SRC,
                GAMMA_DRAM_ADDR=lw['norm4_w'],
                BETA_DRAM_ADDR=lw['norm4_b'],
            )

        def _compile_mlp(lw):
            """Compile one MLP block (lin1 + GELU + lin2) + residual + norm3 on DEC_TOKENS."""
            self.matmat_mul_core(
                M=NT, K=DD, N=MLP_DIM,
                A_DRAM_ADDR=self.DEC_TOKENS,
                B_DRAM_ADDR=lw['mlp_lin1_w'],
                OUTPUT_DRAM_ADDR=self.DEC_MLP_INTER,
                C_DRAM_ADDR=lw['mlp_lin1_b'],
                bias_mode="broadcast_N",
                gelu_enable=True,
            )
            self.matmat_mul_core(
                M=NT, K=MLP_DIM, N=DD,
                A_DRAM_ADDR=self.DEC_MLP_INTER,
                B_DRAM_ADDR=lw['mlp_lin2_w'],
                OUTPUT_DRAM_ADDR=self.DEC_TOKENS_NORM,
                C_DRAM_ADDR=lw['mlp_lin2_b'],
                bias_mode="broadcast_N",
            )
            eltwise_add_dram(self, self.DEC_TOKENS, self.DEC_TOKENS_NORM,
                             self.DEC_TOKENS, NT * DD)
            self.layer_norm_core_dram(
                M=NT, N=DD,
                A_DRAM_ADDR=self.DEC_TOKENS,
                OUTPUT_DRAM_ADDR=self.DEC_TOKENS,
                GAMMA_DRAM_ADDR=lw['norm3_w'],
                BETA_DRAM_ADDR=lw['norm3_b'],
            )

        def _compile_t2i(lw, is_final=False, out_norm_w=None, out_norm_b=None):
            """Compile one token→image cross-attn block.
            If is_final=True: no residual add, apply out_norm instead of norm2."""
            # Q: (tokens + query_pe) → project → (7, 512)
            eltwise_add_dram(self, self.DEC_TOKENS, self.DEC_QUERY_PE,
                             self.DEC_TOKENS_NORM, NT * DD)
            self.matmat_mul_core(
                M=NT, K=DD, N=N_CA,
                A_DRAM_ADDR=self.DEC_TOKENS_NORM,
                B_DRAM_ADDR=lw['ca_t2i_q_w'],
                OUTPUT_DRAM_ADDR=self.DEC_Q,
                C_DRAM_ADDR=lw['ca_t2i_q_b'],
                bias_mode="broadcast_N",
            )

            # K: (src + key_pe) @ W_k
            self.matmat_mul_core(
                M=GA, K=DD, N=N_CA,
                A_DRAM_ADDR=self.DEC_SRC,
                B_DRAM_ADDR=lw['ca_t2i_k_w'],
                OUTPUT_DRAM_ADDR=self.DEC_CA_T2I_K,
                C_DRAM_ADDR=lw['ca_t2i_k_b'],
                bias_mode="broadcast_N",
            )
            self.matmat_mul_core(
                M=GA, K=DD, N=N_CA,
                A_DRAM_ADDR=self.DEC_POS_SRC,
                B_DRAM_ADDR=lw['ca_t2i_k_w'],
                OUTPUT_DRAM_ADDR=self.DEC_CA_T2I_V,   # temp
            )
            eltwise_add_dram(self, self.DEC_CA_T2I_K, self.DEC_CA_T2I_V,
                             self.DEC_CA_T2I_K, GA * N_CA)

            # V: src (no PE) → project
            self.matmat_mul_core(
                M=GA, K=DD, N=N_CA,
                A_DRAM_ADDR=self.DEC_SRC,
                B_DRAM_ADDR=lw['ca_t2i_v_w'],
                OUTPUT_DRAM_ADDR=self.DEC_CA_T2I_V,
                C_DRAM_ADDR=lw['ca_t2i_v_b'],
                bias_mode="broadcast_N",
            )

            # Permute Q: (7, 8, 64) → (8, 7, 64) into DEC_SA_TEMP
            self.bf16_permute_core(
                dim_0=NT, dim_1=DH, dim_2=CA_HDP,
                INPUT_DRAM_ADDR=self.DEC_Q,
                OUTPUT_DRAM_ADDR=self.DEC_SA_TEMP,
            )
            # Permute K: (4096, 8, 64) → (8, 4096, 64)
            self.bf16_permute_core(
                dim_0=GA, dim_1=DH, dim_2=CA_HDP,
                INPUT_DRAM_ADDR=self.DEC_CA_T2I_K,
                OUTPUT_DRAM_ADDR=self.DEC_CA_T2I_K_HEADS,
            )
            # Permute V: (4096, 8, 64) → (8, 4096, 64)
            self.bf16_permute_core(
                dim_0=GA, dim_1=DH, dim_2=CA_HDP,
                INPUT_DRAM_ADDR=self.DEC_CA_T2I_V,
                OUTPUT_DRAM_ADDR=self.DEC_CA_T2I_V_HEADS,
            )

            broadcast_mul_dram(self, self.DEC_SA_TEMP,
                               1.0 / math.sqrt(CA_HD), DH * NT * CA_HDP)

            for h in range(DH):
                q_addr = self.DEC_SA_TEMP        + h * NT * CA_HDP * bpe
                k_addr = self.DEC_CA_T2I_K_HEADS + h * GA * CA_HDP * bpe
                v_addr = self.DEC_CA_T2I_V_HEADS + h * GA * CA_HDP * bpe
                o_addr = self.DEC_CA_OUT_HEADS   + h * NT * CA_HDP * bpe

                self.matmat_mul_core(
                    M=NT, K=CA_HDP, N=GA,
                    A_DRAM_ADDR=q_addr,
                    B_DRAM_ADDR=k_addr,
                    OUTPUT_DRAM_ADDR=self.DEC_CA_SCORES,
                    softmax_enable=True,
                )
                self.bf16_transpose_core(
                    M=GA, N=CA_HDP,
                    INPUT_DRAM_ADDR=v_addr,
                    OUTPUT_DRAM_ADDR=self.DEC_CA_VT,
                )
                self.matmat_mul_core(
                    M=NT, K=GA, N=CA_HDP,
                    A_DRAM_ADDR=self.DEC_CA_SCORES,
                    B_DRAM_ADDR=self.DEC_CA_VT,
                    OUTPUT_DRAM_ADDR=o_addr,
                )

            multihead_merge_dram(self,
                INPUT_DRAM_ADDR=self.DEC_CA_OUT_HEADS,
                OUTPUT_DRAM_ADDR=self.DEC_CA_T2I_MERGED,
                TEMP_DRAM_ADDR=self.DEC_SA_Q_HEADS,
                seq_len=NT, num_heads=DH,
                head_dim=CA_HD, head_dim_pad=CA_HDP,
                UNPAD_WEIGHT_ADDR=self.DEC_CA_UNPAD_W,
            )

            self.matmat_mul_core(
                M=NT, K=CA_ID, N=DD,
                A_DRAM_ADDR=self.DEC_CA_T2I_MERGED,
                B_DRAM_ADDR=lw['ca_t2i_out_w'],
                OUTPUT_DRAM_ADDR=self.DEC_TOKENS_NORM,
                C_DRAM_ADDR=lw['ca_t2i_out_b'],
                bias_mode="broadcast_N",
            )

            if not is_final:
                eltwise_add_dram(self, self.DEC_TOKENS, self.DEC_TOKENS_NORM,
                                 self.DEC_TOKENS, NT * DD)
                self.layer_norm_core_dram(
                    M=NT, N=DD,
                    A_DRAM_ADDR=self.DEC_TOKENS,
                    OUTPUT_DRAM_ADDR=self.DEC_TOKENS,
                    GAMMA_DRAM_ADDR=lw['norm2_w'],
                    BETA_DRAM_ADDR=lw['norm2_b'],
                )
            else:
                # Final attn: add residual then apply final layer norm
                eltwise_add_dram(self, self.DEC_TOKENS, self.DEC_TOKENS_NORM,
                                 self.DEC_TOKENS, NT * DD)
                self.layer_norm_core_dram(
                    M=NT, N=DD,
                    A_DRAM_ADDR=self.DEC_TOKENS,
                    OUTPUT_DRAM_ADDR=self.DEC_TOKENS,
                    GAMMA_DRAM_ADDR=out_norm_w,
                    BETA_DRAM_ADDR=out_norm_b,
                )

        def _compile_sa(lw):
            """Compile one self-attention block + out_proj + residual + norm1 on DEC_TOKENS."""
            HD     = self.DEC_HEAD_DIM   # 32
            HD_PAD = 64
            NT_PAD = 64
            N_PAD  = DH * HD_PAD         # 512

            # Q/K/V projections (padded weights)
            self.matmat_mul_core(
                M=NT, K=DD, N=N_PAD,
                A_DRAM_ADDR=self.DEC_TOKENS,
                B_DRAM_ADDR=lw['sa_q_w'],
                OUTPUT_DRAM_ADDR=self.DEC_Q,
                C_DRAM_ADDR=lw['sa_q_b'],
                bias_mode="broadcast_N",
            )
            self.matmat_mul_core(
                M=NT, K=DD, N=N_PAD,
                A_DRAM_ADDR=self.DEC_TOKENS,
                B_DRAM_ADDR=lw['sa_k_w'],
                OUTPUT_DRAM_ADDR=self.DEC_K,
                C_DRAM_ADDR=lw['sa_k_b'],
                bias_mode="broadcast_N",
            )
            self.matmat_mul_core(
                M=NT, K=DD, N=N_PAD,
                A_DRAM_ADDR=self.DEC_TOKENS,
                B_DRAM_ADDR=lw['sa_v_w'],
                OUTPUT_DRAM_ADDR=self.DEC_V,
                C_DRAM_ADDR=lw['sa_v_b'],
                bias_mode="broadcast_N",
            )

            # Permute Q/K/V: (7,8,64) → compact (8,7,64) in TEMP, zero-fill (8,64,64), scatter
            for proj, heads in [(self.DEC_Q, self.DEC_SA_Q_HEADS),
                                (self.DEC_K, self.DEC_SA_K_HEADS),
                                (self.DEC_V, self.DEC_SA_V_HEADS)]:
                self.bf16_permute_core(
                    dim_0=NT, dim_1=DH, dim_2=HD_PAD,
                    INPUT_DRAM_ADDR=proj,
                    OUTPUT_DRAM_ADDR=self.DEC_SA_TEMP,
                )
                dram_zero_fill(self, heads, DH * NT_PAD * HD_PAD)
                for h in range(DH):
                    dram_copy(self,
                        self.DEC_SA_TEMP + h * NT * HD_PAD * bpe,
                        heads             + h * NT_PAD * HD_PAD * bpe,
                        NT * HD_PAD)

            # Scale Q by sqrt(2): effective scale = sqrt(2)/sqrt(64) = 1/sqrt(32) = 1/sqrt(HD)
            broadcast_mul_dram(self, self.DEC_SA_Q_HEADS, math.sqrt(2.0), DH * NT_PAD * HD_PAD)

            flash_attention_batched(self,
                num_batches=DH, head_dim=HD_PAD, seq_len=NT_PAD,
                Q_DRAM_ADDR=self.DEC_SA_Q_HEADS,
                K_DRAM_ADDR=self.DEC_SA_K_HEADS,
                V_DRAM_ADDR=self.DEC_SA_V_HEADS,
                OUTPUT_DRAM_ADDR=self.DEC_SA_OUT_HEADS,
                SCRATCH_DRAM_ADDR=self.DEC_SA_SCRATCH,
                BIAS_DRAM_ADDR=self.DEC_SA_ATTN_BIAS,
            )

            # Extract first NT=7 rows per head from (8,64,64) into compact (8,7,64)
            for h in range(DH):
                dram_copy(self,
                    self.DEC_SA_OUT_HEADS + h * NT_PAD * HD_PAD * bpe,
                    self.DEC_SA_TEMP      + h * NT * HD_PAD * bpe,
                    NT * HD_PAD)

            multihead_merge_dram(self,
                INPUT_DRAM_ADDR=self.DEC_SA_TEMP,
                OUTPUT_DRAM_ADDR=self.DEC_SA_MERGED,
                TEMP_DRAM_ADDR=self.DEC_SA_Q_HEADS,
                seq_len=NT, num_heads=DH,
                head_dim=HD, head_dim_pad=HD_PAD,
                UNPAD_WEIGHT_ADDR=self.DEC_SA_UNPAD_W,
            )

            self.matmat_mul_core(
                M=NT, K=DD, N=DD,
                A_DRAM_ADDR=self.DEC_SA_MERGED,
                B_DRAM_ADDR=lw['sa_out_w'],
                OUTPUT_DRAM_ADDR=self.DEC_TOKENS_NORM,
                C_DRAM_ADDR=lw['sa_out_b'],
                bias_mode="broadcast_N",
            )
            eltwise_add_dram(self, self.DEC_TOKENS, self.DEC_TOKENS_NORM,
                             self.DEC_TOKENS, NT * DD)
            self.layer_norm_core_dram(
                M=NT, N=DD,
                A_DRAM_ADDR=self.DEC_TOKENS,
                OUTPUT_DRAM_ADDR=self.DEC_TOKENS,
                GAMMA_DRAM_ADDR=lw['norm1_w'],
                BETA_DRAM_ADDR=lw['norm1_b'],
            )

        # ── Layer 0: remaining blocks (MLP + i2t) ─────────────────────────────
        _compile_mlp(lw0)
        _compile_i2t(lw0)

        # ── Layer 1: sa1 + t2i1 + MLP1 + i2t1 ────────────────────────────────
        lw1 = self.dec_layer_weights[1]
        _compile_sa(lw1)
        _compile_t2i(lw1)
        _compile_mlp(lw1)
        _compile_i2t(lw1)

        # ── Final token→image cross-attention + layer_norm_final_attn ─────────
        # Uses dec_final_attn (padded Q/K/V weights) — no i2t, no MLP after this.
        # Output updates DEC_TOKENS; DEC_SRC (last i2t1 output) feeds mask upscaling.
        # Adapt dec_final_attn keys to the ca_t2i_* scheme _compile_t2i expects.
        fa = self.dec_final_attn
        _final_lw = {
            'ca_t2i_q_w': fa['q_w'], 'ca_t2i_q_b': fa['q_b'],
            'ca_t2i_k_w': fa['k_w'], 'ca_t2i_k_b': fa['k_b'],
            'ca_t2i_v_w': fa['v_w'], 'ca_t2i_v_b': fa['v_b'],
            'ca_t2i_out_w': fa['out_w'], 'ca_t2i_out_b': fa['out_b'],
        }
        _compile_t2i(
            lw=_final_lw,
            is_final=True,
            out_norm_w=self.dec_final_norm['w'],
            out_norm_b=self.dec_final_norm['b'],
        )

        # ── Hypernetwork MLPs + IoU head ───────────────────────────────────────
        # Tokens layout after transformer:
        #   row 0 : iou_token  → IoU head (256→256→256→4)
        #   rows 1-4: mask_tokens → 4 hypernetwork MLPs (256→256→256→32)
        NUM_MASKS = 4

        # IoU head: 3-layer MLP with ReLU on first two layers
        # Input: DEC_TOKENS row 0 (1, 256); output: DEC_IOU_OUT (1, 4)
        iou_w = self.dec_iou_weights
        # Layer 0: (1,256) → (1,256) + ReLU
        self.matmat_mul_core(
            M=1, K=DD, N=DD,
            A_DRAM_ADDR=self.DEC_TOKENS,
            B_DRAM_ADDR=iou_w['l0_w'],
            OUTPUT_DRAM_ADDR=self.DEC_TOKENS_NORM,
            C_DRAM_ADDR=iou_w['l0_b'],
            bias_mode="broadcast_N",
            relu_enable=True,
        )
        # Layer 1: (1,256) → (1,256) + ReLU
        self.matmat_mul_core(
            M=1, K=DD, N=DD,
            A_DRAM_ADDR=self.DEC_TOKENS_NORM,
            B_DRAM_ADDR=iou_w['l1_w'],
            OUTPUT_DRAM_ADDR=self.DEC_TOKENS_NORM,
            C_DRAM_ADDR=iou_w['l1_b'],
            bias_mode="broadcast_N",
            relu_enable=True,
        )
        # Layer 2: (1,256) → (1,4) — padded to N=64 for hardware lane alignment
        self.matmat_mul_core(
            M=1, K=DD, N=64,
            A_DRAM_ADDR=self.DEC_TOKENS_NORM,
            B_DRAM_ADDR=iou_w['l2_w'],
            OUTPUT_DRAM_ADDR=self.DEC_IOU_OUT,
            C_DRAM_ADDR=iou_w['l2_b'],
            bias_mode="broadcast_N",
        )

        # Hypernetwork MLPs: 4 mask tokens (rows 1-4) → 4 × (1,32) mask weight vectors
        # Each MLP: ReLU on first two layers; no activation on final layer
        for m in range(NUM_MASKS):
            hw = self.dec_hyper_weights[m]
            src_addr = self.DEC_TOKENS + (m + 1) * DD * bpe  # row (m+1) of DEC_TOKENS
            out_addr = self.DEC_HYPER_OUT + m * 64 * bpe     # row m of DEC_HYPER_OUT (padded to 64 cols)

            # Layer 0: (1,256) → (1,256) + ReLU
            self.matmat_mul_core(
                M=1, K=DD, N=DD,
                A_DRAM_ADDR=src_addr,
                B_DRAM_ADDR=hw['l0_w'],
                OUTPUT_DRAM_ADDR=self.DEC_TOKENS_NORM,
                C_DRAM_ADDR=hw['l0_b'],
                bias_mode="broadcast_N",
                relu_enable=True,
            )
            # Layer 1: (1,256) → (1,256) + ReLU
            self.matmat_mul_core(
                M=1, K=DD, N=DD,
                A_DRAM_ADDR=self.DEC_TOKENS_NORM,
                B_DRAM_ADDR=hw['l1_w'],
                OUTPUT_DRAM_ADDR=self.DEC_TOKENS_NORM,
                C_DRAM_ADDR=hw['l1_b'],
                bias_mode="broadcast_N",
                relu_enable=True,
            )
            # Layer 2: (1,256) → (1,32) — padded to N=64 for hardware lane alignment
            self.matmat_mul_core(
                M=1, K=DD, N=64,
                A_DRAM_ADDR=self.DEC_TOKENS_NORM,
                B_DRAM_ADDR=hw['l2_w'],
                OUTPUT_DRAM_ADDR=out_addr,
                C_DRAM_ADDR=hw['l2_b'],
                bias_mode="broadcast_N",
            )

        # Zero-fill padded rows 4..63 of DEC_HYPER_OUT so they don't contribute
        # garbage to the final (65536,64) @ (64,64)^T mask-logit matmul.
        dram_zero_fill(self, self.DEC_HYPER_OUT + 4 * 64 * bpe, 60 * 64)

        # ── Mask upscaling: ConvTranspose2d × 2 ───────────────────────────────
        # Input: DEC_SRC (4096, 256) — final image features after i2t1 + norm4
        # Stage 0: ConvTranspose2d(256→64, k=2,s=2) → (16384, 64); LayerNorm2d(64); GELU
        C_IN0, C_OUT0 = DD, 64           # 256 → 64
        H_IN0, W_IN0  = 64, 64
        H_OUT0, W_OUT0 = 128, 128

        # Compute 4 sub-position matmuls: for each (dr, dc), output is (GA, C_OUT0)
        # Bias added here — each output pixel gets contribution from exactly one sub-position
        for dr in range(2):
            for dc in range(2):
                sub_idx = dr * 2 + dc
                self.matmat_mul_core(
                    M=GA, K=C_IN0, N=C_OUT0,
                    A_DRAM_ADDR=self.DEC_SRC,
                    B_DRAM_ADDR=self.DEC_UP0_W[dr][dc],
                    OUTPUT_DRAM_ADDR=self.DEC_UP0_SCRATCH + sub_idx * GA * C_OUT0 * bpe,
                    C_DRAM_ADDR=self.DEC_UP0_B,
                    bias_mode="broadcast_N" if self.DEC_UP0_B is not None else None,
                )

        # Pixel-shuffle scatter: (4, 4096, 64) → (16384, 64)
        # src_row = sub_idx*GA + r*W_IN0 + c; dst_row = (2*r+dr)*W_OUT0 + (2*c+dc)
        for r in range(H_IN0):
            for c in range(W_IN0):
                for dr in range(2):
                    for dc in range(2):
                        sub_idx  = dr * 2 + dc
                        src_row  = sub_idx * GA + r * W_IN0 + c
                        dst_row  = (2 * r + dr) * W_OUT0 + (2 * c + dc)
                        src_byte = self.DEC_UP0_SCRATCH + src_row * C_OUT0 * bpe
                        dst_byte = self.DEC_UP0_OUT     + dst_row * C_OUT0 * bpe
                        self.accelerator_memory_to_sram(src_byte, 0x0000, C_OUT0)
                        self.sram_to_accelerator_memory(0x0000, dst_byte, C_OUT0)

        # LayerNorm2d(64): normalize each spatial position (row) along the channel dim
        self.layer_norm_core_dram(
            M=H_OUT0 * W_OUT0, N=C_OUT0,
            A_DRAM_ADDR=self.DEC_UP0_OUT,
            OUTPUT_DRAM_ADDR=self.DEC_UP0_OUT,
            GAMMA_DRAM_ADDR=self.DEC_UP_LN_W,
            BETA_DRAM_ADDR=self.DEC_UP_LN_B,
        )

        # GELU activation: reuse matmat_mul_core with gelu_enable.
        # Identity weight (N=C_OUT0, K=C_OUT0) so output = GELU(input).
        # Build identity matrix for GELU passthrough (stored in params)
        _identity_64 = torch.eye(C_OUT0, dtype=torch.bfloat16)
        _gelu_id_w = self._alloc_param(_identity_64)
        self.matmat_mul_core(
            M=H_OUT0 * W_OUT0, K=C_OUT0, N=C_OUT0,
            A_DRAM_ADDR=self.DEC_UP0_OUT,
            B_DRAM_ADDR=_gelu_id_w,
            OUTPUT_DRAM_ADDR=self.DEC_UP0_OUT,
            gelu_enable=True,
        )

        # Stage 1: ConvTranspose2d(64→32, k=2,s=2) → (65536, 32)
        C_IN1, C_OUT1  = C_OUT0, 32        # 64 → 32
        C_OUT1_PAD     = 64                # hardware lane alignment (N must be multiple of 64)
        H_IN1, W_IN1   = H_OUT0, W_OUT0    # 128 × 128
        H_OUT1, W_OUT1 = 256, 256
        GA1 = H_IN1 * W_IN1                # 16384

        for dr in range(2):
            for dc in range(2):
                sub_idx = dr * 2 + dc
                self.matmat_mul_core(
                    M=GA1, K=C_IN1, N=C_OUT1_PAD,
                    A_DRAM_ADDR=self.DEC_UP0_OUT,
                    B_DRAM_ADDR=self.DEC_UP1_W[dr][dc],
                    OUTPUT_DRAM_ADDR=self.DEC_UP1_SCRATCH + sub_idx * GA1 * C_OUT1_PAD * bpe,
                    C_DRAM_ADDR=self.DEC_UP1_B,
                    bias_mode="broadcast_N" if self.DEC_UP1_B is not None else None,
                )

        # Pixel-shuffle scatter: (4, 16384, 64) → (65536, 64)
        for r in range(H_IN1):
            for c in range(W_IN1):
                for dr in range(2):
                    for dc in range(2):
                        sub_idx  = dr * 2 + dc
                        src_row  = sub_idx * GA1 + r * W_IN1 + c
                        dst_row  = (2 * r + dr) * W_OUT1 + (2 * c + dc)
                        src_byte = self.DEC_UP1_SCRATCH + src_row * C_OUT1_PAD * bpe
                        dst_byte = self.DEC_UP1_OUT     + dst_row * C_OUT1_PAD * bpe
                        self.accelerator_memory_to_sram(src_byte, 0x0000, C_OUT1_PAD)
                        self.sram_to_accelerator_memory(0x0000, dst_byte, C_OUT1_PAD)

        # GELU after second ConvTranspose2d (SAM output_upscaling index 4)
        _identity_64_1 = torch.eye(64, dtype=torch.bfloat16)
        _gelu_id_w1 = self._alloc_param(_identity_64_1)
        self.matmat_mul_core(
            M=H_OUT1 * W_OUT1, K=C_OUT1_PAD, N=C_OUT1_PAD,
            A_DRAM_ADDR=self.DEC_UP1_OUT,
            B_DRAM_ADDR=_gelu_id_w1,
            OUTPUT_DRAM_ADDR=self.DEC_UP1_OUT,
            gelu_enable=True,
        )

        # ── Final mask logits: (65536, 64) @ (64, 64)^T → (65536, 64) ──────────
        # DEC_HYPER_OUT is padded (64, 64) — first 4 rows real, rest zero.
        # Only first 4 output columns per row are real mask logits.
        self.matmat_mul_core(
            M=H_OUT1 * W_OUT1, K=C_OUT1_PAD, N=64,
            A_DRAM_ADDR=self.DEC_UP1_OUT,
            B_DRAM_ADDR=self.DEC_HYPER_OUT,
            OUTPUT_DRAM_ADDR=self.DEC_MASK_LOGITS,
        )
        _original_print(f"  [compile] mask decoder: {_time.perf_counter() - _tp:.3f}s")

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

    def cpu_reference_neck(self, vit_out: torch.Tensor) -> torch.Tensor:
        """CPU reference for SAM1 neck: Conv1×1 → LN → Conv3×3 → LN.

        Args:
            vit_out: (4096, 768) bf16 ViT backbone output (VIT_LAYER_IN contents).
        Returns:
            (4096, 256) bf16 neck output matching NECK_OUT.
        """
        import torch.nn as nn
        import torch.nn.functional as F

        ckpt_path = _ensure_checkpoint(self.script_dir, self.cfg)
        sd = _load_sam1_state_dict(ckpt_path)
        neck = "vision_encoder.neck."
        ND = self.NECK_DIM   # 256
        VD = self.VIT_DIM    # 768
        GS = self.GRID_SIZE  # 64

        x = vit_out.to(torch.bfloat16)  # (4096, 768)

        # Conv 1×1: (4096, 768) @ (256, 768)^T → (4096, 256)
        w1 = sd[neck + "conv1.weight"].to(torch.bfloat16).reshape(ND, VD)
        x = (x.float() @ w1.float().T).to(torch.bfloat16)

        # LayerNorm 1
        ln1 = nn.LayerNorm(ND)
        ln1.weight.data = sd[neck + "layer_norm1.weight"].float()
        ln1.bias.data   = sd[neck + "layer_norm1.bias"].float()
        x = ln1(x.float()).to(torch.bfloat16)  # (4096, 256)

        # Conv 3×3 (padding=1): treat (4096, 256) as (1, 256, 64, 64) spatial map
        w3 = sd[neck + "conv2.weight"].to(torch.bfloat16)  # (256, 256, 3, 3)
        x_spatial = x.reshape(GS, GS, ND).permute(2, 0, 1).unsqueeze(0)  # (1, 256, 64, 64)
        x_spatial = F.conv2d(x_spatial.float(), w3.float(), bias=None, padding=1)  # (1, 256, 64, 64)
        x = x_spatial.squeeze(0).permute(1, 2, 0).reshape(GS * GS, ND).to(torch.bfloat16)

        # LayerNorm 2
        ln2 = nn.LayerNorm(ND)
        ln2.weight.data = sd[neck + "layer_norm2.weight"].float()
        ln2.bias.data   = sd[neck + "layer_norm2.bias"].float()
        x = ln2(x.float()).to(torch.bfloat16)  # (4096, 256)

        del sd
        return x

    def cpu_reference_dec_sa0(self, cpu_tokens: torch.Tensor,
                               neck_out: torch.Tensor) -> tuple:
        """CPU reference for decoder src init + layer-0 self-attention.

        Matches Phase 4 compile:
          src = neck_out + no_mask_embed (broadcast)
          tokens: layer 0 skip_first_layer_pe → q=k=v=tokens, self-attn,
                  out_proj, residual, norm1

        Returns:
            (tokens_after_norm1, src)  both (NT, 256) / (4096, 256) bf16
        """
        import torch.nn as nn
        import torch.nn.functional as F

        ckpt_path = _ensure_checkpoint(self.script_dir, self.cfg)
        sd = _load_sam1_state_dict(ckpt_path)
        md = "mask_decoder."
        lp = md + "transformer.layers.0."

        tokens = cpu_tokens.to(torch.bfloat16)   # (7, 256)
        src    = neck_out.to(torch.bfloat16)      # (4096, 256)

        # src += no_mask_embed (broadcast)
        no_mask = sd["prompt_encoder.no_mask_embed.weight"].to(torch.bfloat16)  # (1, 256)
        src = src + no_mask  # (4096, 256)

        # Self-attn: skip_first_layer_pe → q=k=v=tokens (no PE)
        q_w = sd[lp + "self_attn.q_proj.weight"].to(torch.bfloat16)
        q_b = sd[lp + "self_attn.q_proj.bias"].to(torch.bfloat16)
        k_w = sd[lp + "self_attn.k_proj.weight"].to(torch.bfloat16)
        k_b = sd[lp + "self_attn.k_proj.bias"].to(torch.bfloat16)
        v_w = sd[lp + "self_attn.v_proj.weight"].to(torch.bfloat16)
        v_b = sd[lp + "self_attn.v_proj.bias"].to(torch.bfloat16)
        out_w = sd[lp + "self_attn.out_proj.weight"].to(torch.bfloat16)
        out_b = sd[lp + "self_attn.out_proj.bias"].to(torch.bfloat16)

        NT, DD, DH, HD = self.DEC_NUM_TOKENS, self.DEC_DIM, self.DEC_HEADS, self.DEC_HEAD_DIM
        Q = F.linear(tokens.float(), q_w.float(), q_b.float()).to(torch.bfloat16)  # (7, 256)
        K = F.linear(tokens.float(), k_w.float(), k_b.float()).to(torch.bfloat16)
        V = F.linear(tokens.float(), v_w.float(), v_b.float()).to(torch.bfloat16)

        # Multi-head reshape: (7, 256) → (8, 7, 32)
        Q = Q.reshape(NT, DH, HD).permute(1, 0, 2)  # (8, 7, 32)
        K = K.reshape(NT, DH, HD).permute(1, 0, 2)
        V = V.reshape(NT, DH, HD).permute(1, 0, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(HD)
        scores = torch.bmm(Q * scale, K.transpose(-2, -1))         # (8, 7, 7)
        probs  = torch.softmax(scores.float(), dim=-1).to(torch.bfloat16)
        attn   = torch.bmm(probs, V)                                # (8, 7, 32)

        # Merge heads: (8, 7, 32) → (7, 256)
        attn_merged = attn.permute(1, 0, 2).reshape(NT, DD)

        # out_proj + residual + norm1
        attn_out = F.linear(attn_merged.float(), out_w.float(), out_b.float()).to(torch.bfloat16)
        tokens = tokens + attn_out

        ln1_w = sd[lp + "layer_norm1.weight"].float()
        ln1_b = sd[lp + "layer_norm1.bias"].float()
        _mean = tokens.float().mean(dim=-1, keepdim=True)
        _cent = tokens.float() - _mean
        _rms  = _cent.pow(2).mean(dim=-1, keepdim=True).sqrt()
        tokens = (_cent / _rms).to(torch.bfloat16) * ln1_w.to(torch.bfloat16) + ln1_b.to(torch.bfloat16)

        del sd
        return tokens.to(torch.bfloat16), src.to(torch.bfloat16)

    def cpu_reference_dec_t2i0(self, tokens_after_norm1: torch.Tensor,
                                src: torch.Tensor,
                                query_pe: torch.Tensor,
                                key_pe: torch.Tensor) -> torch.Tensor:
        """CPU reference for decoder layer-0 token→image cross-attention.

        Args:
            tokens_after_norm1: (7, 256) bf16 — output of self-attn + norm1
            src:                (4096, 256) bf16 — image features + no_mask_embed
            query_pe:           (7, 256) bf16 — prompt token positional encoding
            key_pe:             (4096, 256) bf16 — dense image positional encoding
        Returns:
            tokens after t2i attn + residual + norm2, shape (7, 256) bf16
        """
        import torch.nn.functional as F

        ckpt_path = _ensure_checkpoint(self.script_dir, self.cfg)
        sd = _load_sam1_state_dict(ckpt_path)
        lp = "mask_decoder.transformer.layers.0."

        tokens = tokens_after_norm1.to(torch.bfloat16)  # (7, 256)
        src    = src.to(torch.bfloat16)                  # (4096, 256)
        qpe    = query_pe.to(torch.bfloat16)             # (7, 256)
        kpe    = key_pe.to(torch.bfloat16)               # (4096, 256)

        NT, DD = self.DEC_NUM_TOKENS, self.DEC_DIM
        DH, HD = self.DEC_HEADS, self.DEC_INTERNAL_HD    # 8, 16
        ID     = self.DEC_INTERNAL_DIM                   # 128

        q_w  = sd[lp + "cross_attn_token_to_image.q_proj.weight"].to(torch.bfloat16)
        q_b  = sd[lp + "cross_attn_token_to_image.q_proj.bias"].to(torch.bfloat16)
        k_w  = sd[lp + "cross_attn_token_to_image.k_proj.weight"].to(torch.bfloat16)
        k_b  = sd[lp + "cross_attn_token_to_image.k_proj.bias"].to(torch.bfloat16)
        v_w  = sd[lp + "cross_attn_token_to_image.v_proj.weight"].to(torch.bfloat16)
        v_b  = sd[lp + "cross_attn_token_to_image.v_proj.bias"].to(torch.bfloat16)
        out_w = sd[lp + "cross_attn_token_to_image.out_proj.weight"].to(torch.bfloat16)
        out_b = sd[lp + "cross_attn_token_to_image.out_proj.bias"].to(torch.bfloat16)
        ln2_w = sd[lp + "layer_norm2.weight"].to(torch.bfloat16)
        ln2_b = sd[lp + "layer_norm2.bias"].to(torch.bfloat16)

        GA = self.GRID_AREA  # 4096

        # q = tokens + query_pe, k = src + key_pe, v = src
        q = (tokens.float() + qpe.float()).to(torch.bfloat16)
        k = (src.float()    + kpe.float()).to(torch.bfloat16)
        v = src

        Q = F.linear(q.float(), q_w.float(), q_b.float()).to(torch.bfloat16)  # (7, 128)
        K = F.linear(k.float(), k_w.float(), k_b.float()).to(torch.bfloat16)  # (4096, 128)
        V = F.linear(v.float(), v_w.float(), v_b.float()).to(torch.bfloat16)  # (4096, 128)

        # Multi-head reshape: Q (7, 8, 16), K/V (4096, 8, 16)
        Q = Q.reshape(NT, DH, HD).permute(1, 0, 2)   # (8, 7, 16)
        K = K.reshape(GA, DH, HD).permute(1, 0, 2)   # (8, 4096, 16)
        V = V.reshape(GA, DH, HD).permute(1, 0, 2)   # (8, 4096, 16)

        scale = 1.0 / math.sqrt(HD)
        scores = torch.bmm(Q.float() * scale, K.float().transpose(-2, -1)).to(torch.bfloat16)  # (8, 7, 4096)
        probs  = torch.softmax(scores.to(torch.bfloat16), dim=-1)   # bf16 softmax to match HW
        attn   = torch.bmm(probs.float(), V.float()).to(torch.bfloat16)  # (8, 7, 16)

        attn_merged = attn.permute(1, 0, 2).reshape(NT, ID)   # (7, 128)
        attn_out = F.linear(attn_merged.float(), out_w.float(), out_b.float()).to(torch.bfloat16)

        tokens = tokens + attn_out  # residual

        # norm2
        _mean = tokens.float().mean(dim=-1, keepdim=True)
        _cent = tokens.float() - _mean
        _rms  = _cent.pow(2).mean(dim=-1, keepdim=True).sqrt()
        tokens = (_cent / _rms).to(torch.bfloat16) * ln2_w + ln2_b

        del sd
        return tokens.to(torch.bfloat16)

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

    _t_total = _time.perf_counter()
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

        fg_emb  = (_point_pe(*TEST_POINT).float() + sd["prompt_encoder.point_embed.1.weight"].float()).to(torch.bfloat16)
        pad_emb = sd["prompt_encoder.not_a_point_embed.weight"].to(torch.bfloat16)  # no PE add for padding token
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

            # Prompt encoder: write 4 sin/cos table rows for (x_px, y_px) to fixed DRAM slots.
            # The compiled program assembles DEC_TOKENS, DEC_QUERY_PE, DEC_KEY_PE from these.
            TEST_POINT = (512.0, 512.0)
            x_px = int(TEST_POINT[0])
            y_px = int(TEST_POINT[1])
            bpe = 2
            sin_x_row = ue.read_tensor_from_dram(ue.PE_SIN_X + x_px * 128 * bpe, 128)
            cos_x_row = ue.read_tensor_from_dram(ue.PE_COS_X + x_px * 128 * bpe, 128)
            sin_y_row = ue.read_tensor_from_dram(ue.PE_SIN_Y + y_px * 128 * bpe, 128)
            cos_y_row = ue.read_tensor_from_dram(ue.PE_COS_Y + y_px * 128 * bpe, 128)
            ue.dma_to_accelerator_memory(ue.PE_SX_SLOT, sin_x_row.contiguous())
            ue.dma_to_accelerator_memory(ue.PE_CX_SLOT, cos_x_row.contiguous())
            ue.dma_to_accelerator_memory(ue.PE_SY_SLOT, sin_y_row.contiguous())
            ue.dma_to_accelerator_memory(ue.PE_CY_SLOT, cos_y_row.contiguous())

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

        # Validate neck against CPU reference
        _original_print("  Running CPU reference for neck...")
        vit_out_raw = ue.read_tensor_from_dram(ue.VIT_LAYER_IN, ue.GRID_AREA * ue.VIT_DIM)
        vit_out_raw = vit_out_raw.reshape(ue.GRID_AREA, ue.VIT_DIM)
        neck_ref = ue.cpu_reference_neck(vit_out_raw)
        ue.validate(ue.NECK_OUT, ue.GRID_AREA * ue.NECK_DIM, neck_ref, "Neck output (4096,256)")

        # ── Read and postprocess final outputs ────────────────────────────────
        _t_post = _time.perf_counter()
        # Mask logits: stored (65536, 64) padded — only first 4 cols per row are real
        mask_logits = ue.read_tensor_from_dram(
            ue.DEC_MASK_LOGITS, 256 * 256 * 64).reshape(256 * 256, 64).float()
        mask_logits = mask_logits[:, :4].T.contiguous().reshape(4, 256, 256)
        _original_print(f"  Mask logit stats: min={mask_logits.min():.4f}  max={mask_logits.max():.4f}  "
                        f"mean={mask_logits.mean():.4f}  positive_pct={100*(mask_logits>0).float().mean():.1f}%")

        # IoU scores: stored padded (64,) — only first 4 real
        iou_scores = ue.read_tensor_from_dram(ue.DEC_IOU_OUT, 64).float()[:4]
        _original_print(f"  IoU scores: {iou_scores.tolist()}")

        # Sigmoid → probabilities, select best mask by IoU
        mask_probs = torch.sigmoid(mask_logits)          # (4, 256, 256)
        best_idx   = int(iou_scores.argmax())
        best_mask  = mask_probs[best_idx]                # (256, 256)
        _original_print(f"  Best mask idx: {best_idx}  IoU: {iou_scores[best_idx]:.4f}")

        # Upsample 256×256 → 1024×1024 (4× bilinear, matches SAM's original resolution)
        mask_1024 = torch.nn.functional.interpolate(
            best_mask.unsqueeze(0).unsqueeze(0),
            size=(1024, 1024), mode="bilinear", align_corners=False,
        ).squeeze()                                      # (1024, 1024)

        # Threshold at 0.5 → binary mask
        binary_mask = (mask_1024 > 0.5).numpy().astype("uint8") * 255

        # Save output
        import numpy as np
        out_dir  = os.path.join(SCRIPT_DIR, "sam1_vit_b_bin")
        out_path = os.path.join(out_dir, "mask_output.png")
        os.makedirs(out_dir, exist_ok=True)
        Image.fromarray(binary_mask).save(out_path)
        _original_print(f"  Mask saved → {out_path}")
        _original_print(f"  Postprocess (readback+sigmoid+upsample+save): {_time.perf_counter() - _t_post:.3f}s")

        # Overlay on original image for visualization
        try:
            orig_img = Image.open(image_path).convert("RGB").resize((1024, 1024))
            overlay  = np.array(orig_img).copy()
            overlay[binary_mask > 0] = (overlay[binary_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype("uint8")
            overlay_path = os.path.join(out_dir, "mask_overlay.png")
            Image.fromarray(overlay).save(overlay_path)
            _original_print(f"  Overlay saved → {overlay_path}")
        except Exception as e:
            _original_print(f"  Overlay failed: {e}")

        _original_print(f"  Total end-to-end: {_time.perf_counter() - _t_total:.3f}s")


if __name__ == "__main__":
    main()
