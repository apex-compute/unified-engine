#!/usr/bin/env python3
"""MobileSAM mask decoder — hardware accelerator implementation.

Encoder + mask decoder compiled into instruction streams and
executed on the Unified Engine accelerator.

Usage:
    python mobilesam_test.py [--dev xdma0] [--device kintex7] [--cycle 5.15]
"""

import argparse
import builtins
import itertools
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_original_print = builtins.print
builtins.print = lambda *a, **k: None  # silence all compile/init noise; use _original_print for real output

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, SCRIPT_DIR)

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, DMA_DEVICE_C2H, DRAM_INSTRUCTION_ADDR,
    UE_VECTOR_SIZE, URAM_NEAR_FULL_ELEMENTS, UE_FMAX_CONTEXT_SIZE,
    set_dma_device, UnifiedEngine, calculate_snr,
)
import warnings as _warnings
_warnings.filterwarnings("ignore")
import torch.nn as _nn

COMPUTE_DTYPE = torch.bfloat16


def nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float):
    """Non-maximum suppression. Returns list of kept indices."""
    if boxes.numel() == 0:
        return []
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True).tolist()
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        remaining = []
        for j in order:
            inter = max(0.0, min(x2[i].item(), x2[j].item()) - max(x1[i].item(), x1[j].item())) * \
                    max(0.0, min(y2[i].item(), y2[j].item()) - max(y1[i].item(), y1[j].item()))
            union = areas[i].item() + areas[j].item() - inter
            if union > 0 and inter / union <= threshold:
                remaining.append(j)
        order = remaining
    return keep


_prompt_weights = None


def _get_prompt_weights():
    """Lazy-load prompt encoder weights from checkpoint (no mobile_sam package needed)."""
    global _prompt_weights
    if _prompt_weights is not None:
        return _prompt_weights
    _sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    _prompt_weights = {
        "gauss_matrix": _sd["prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"],
        "no_mask_embed": _sd["prompt_encoder.no_mask_embed.weight"],
        "not_a_point": _sd["prompt_encoder.not_a_point_embed.weight"],
        "point_0": _sd["prompt_encoder.point_embeddings.0.weight"],
        "point_1": _sd["prompt_encoder.point_embeddings.1.weight"],
    }
    return _prompt_weights

# ---------------------------------------------------------------------------
# Local HW helpers (built on user_dma_core primitives only)
# ---------------------------------------------------------------------------

_BPE = 2  # bytes per bf16


def dram_zero_fill(ue: UnifiedEngine, DRAM_ADDR: int, num_elements: int) -> None:
    chunk = min(URAM_NEAR_FULL_ELEMENTS, num_elements)
    z_addr = ue.get_params_dram_addr()
    ue.allocate_params_dram(chunk * _BPE)
    ue.dma_write(DMA_DEVICE_H2C, z_addr,
                 torch.zeros(chunk, dtype=torch.bfloat16), chunk * _BPE)
    ue.accelerator_memory_to_sram(z_addr, 0x00000, chunk)
    offset = 0
    while offset < num_elements:
        take = min(chunk, num_elements - offset)
        ue.sram_to_accelerator_memory(0x00000, DRAM_ADDR + offset * _BPE, take)
        offset += take


def dram_copy(ue: UnifiedEngine, SRC: int, DST: int, num_elements: int) -> None:
    for offset in range(0, num_elements, URAM_NEAR_FULL_ELEMENTS):
        take = min(URAM_NEAR_FULL_ELEMENTS, num_elements - offset)
        ue.accelerator_memory_to_sram(SRC + offset * _BPE, 0x00000, take)
        ue.sram_to_accelerator_memory(0x00000, DST + offset * _BPE, take)


def eltwise_add_dram(ue: UnifiedEngine, A_ADDR: int, B_ADDR: int,
                     OUT_ADDR: int, num_elements: int) -> None:
    for offset in range(0, num_elements, URAM_NEAR_FULL_ELEMENTS):
        take = min(URAM_NEAR_FULL_ELEMENTS, num_elements - offset)
        ue.accelerator_memory_to_sram(A_ADDR + offset * _BPE, 0x00000, take)
        ue.accelerator_memory_to_sram(B_ADDR + offset * _BPE, 0x80000, take)
        ue.eltwise_add_core(0x00000, 0x80000, 0x00000, take)
        ue.sram_to_accelerator_memory(0x00000, OUT_ADDR + offset * _BPE, take)


def broadcast_mul_dram(ue: UnifiedEngine, ADDR: int,
                       scalar: float, num_elements: int) -> None:
    for offset in range(0, num_elements, URAM_NEAR_FULL_ELEMENTS):
        take = min(URAM_NEAR_FULL_ELEMENTS, num_elements - offset)
        ue.accelerator_memory_to_sram(ADDR + offset * _BPE, 0x00000, take)
        ue.broadcast_mul(scalar=scalar, sram_start_addr=0x00000,
                         sram_wb_addr=0x00000, element_size=take)
        ue.sram_to_accelerator_memory(0x00000, ADDR + offset * _BPE, take)


def multihead_reshape_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int,
                            OUTPUT_DRAM_ADDR: int, TEMP_DRAM_ADDR: int,
                            seq_len: int, num_heads: int,
                            head_dim: int, head_dim_pad: int,
                            input_row_stride: int = None) -> None:
    """(seq_len, input_row_stride) → (num_heads, seq_len, head_dim_pad).
    input_row_stride defaults to num_heads*head_dim (packed); pass C_PAD when rows are wider."""
    if input_row_stride is None:
        input_row_stride = num_heads * head_dim
    if head_dim == head_dim_pad and input_row_stride == num_heads * head_dim:
        ue.bf16_permute_core(dim_0=seq_len, dim_1=num_heads, dim_2=head_dim,
                             INPUT_DRAM_ADDR=INPUT_DRAM_ADDR,
                             OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR)
    else:
        padded_dim = num_heads * head_dim_pad
        dram_zero_fill(ue, TEMP_DRAM_ADDR, seq_len * padded_dim)
        z_addr = ue.get_params_dram_addr()
        ue.allocate_params_dram(UE_VECTOR_SIZE * _BPE)
        ue.dma_write(DMA_DEVICE_H2C, z_addr,
                     torch.zeros(UE_VECTOR_SIZE, dtype=torch.bfloat16),
                     UE_VECTOR_SIZE * _BPE)
        for h in range(num_heads):
            for row in range(seq_len):
                src = INPUT_DRAM_ADDR + (row * input_row_stride + h * head_dim) * _BPE
                dst = TEMP_DRAM_ADDR + (row * padded_dim + h * head_dim_pad) * _BPE
                ue.accelerator_memory_to_sram(z_addr, 0x00000, UE_VECTOR_SIZE)
                ue.accelerator_memory_to_sram(src, 0x00000, head_dim)
                ue.sram_to_accelerator_memory(0x00000, dst, UE_VECTOR_SIZE)
        ue.bf16_permute_core(dim_0=seq_len, dim_1=num_heads, dim_2=head_dim_pad,
                             INPUT_DRAM_ADDR=TEMP_DRAM_ADDR,
                             OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR)


def multihead_merge_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int,
                          OUTPUT_DRAM_ADDR: int, TEMP_DRAM_ADDR: int,
                          seq_len: int, num_heads: int,
                          head_dim: int, head_dim_pad: int,
                          UNPAD_WEIGHT_ADDR: int = None) -> None:
    """(num_heads, seq_len, head_dim_pad) → (seq_len, num_heads*head_dim)."""
    ue.bf16_permute_core(dim_0=num_heads, dim_1=seq_len, dim_2=head_dim_pad,
                         INPUT_DRAM_ADDR=INPUT_DRAM_ADDR,
                         OUTPUT_DRAM_ADDR=TEMP_DRAM_ADDR)
    if head_dim == head_dim_pad:
        total = seq_len * num_heads * head_dim
        dram_copy(ue, TEMP_DRAM_ADDR, OUTPUT_DRAM_ADDR, total)
    else:
        padded_dim = num_heads * head_dim_pad
        dim = num_heads * head_dim
        ue.matmat_mul_core(M=seq_len, K=padded_dim, N=dim,
                           A_DRAM_ADDR=TEMP_DRAM_ADDR,
                           B_DRAM_ADDR=UNPAD_WEIGHT_ADDR,
                           OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR)


def flash_attention_batched(ue: UnifiedEngine, num_batches: int, head_dim: int,
                             seq_len: int, Q_DRAM_ADDR: int, K_DRAM_ADDR: int,
                             V_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                             SCRATCH_DRAM_ADDR: int,
                             BIAS_DRAM_ADDR: int = None,
                             bias_shared: bool = False) -> None:
    stride = seq_len * head_dim * _BPE
    scratch_stride = head_dim * seq_len * _BPE + seq_len * seq_len * _BPE
    bias_stride = 0 if (BIAS_DRAM_ADDR is None or bias_shared) else seq_len * seq_len * _BPE
    for b in range(num_batches):
        bias_addr = (BIAS_DRAM_ADDR + b * bias_stride
                     if (BIAS_DRAM_ADDR is not None and not bias_shared)
                     else BIAS_DRAM_ADDR)
        ue.flash_attention_core(
            head_dim=head_dim, seq_len=seq_len,
            Q_DRAM_ADDR=Q_DRAM_ADDR + b * stride,
            K_DRAM_ADDR=K_DRAM_ADDR + b * stride,
            V_DRAM_ADDR=V_DRAM_ADDR + b * stride,
            OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR + b * stride,
            SCRATCH_DRAM_ADDR=SCRATCH_DRAM_ADDR + b * scratch_stride,
            BIAS_DRAM_ADDR=bias_addr,
        )


def cross_attn_flash_single_head(ue: UnifiedEngine, head_dim: int,
                                  q_len: int, kv_len: int,
                                  Q_DRAM_ADDR: int, K_DRAM_ADDR: int,
                                  V_T_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                                  SCORES_SCRATCH_ADDR: int,
                                  BIAS_DRAM_ADDR: int = None) -> None:
    assert q_len <= UE_FMAX_CONTEXT_SIZE
    assert head_dim % UE_VECTOR_SIZE == 0
    assert kv_len % UE_VECTOR_SIZE == 0
    K = head_dim
    N = kv_len
    N_chunk = min(N, (URAM_NEAR_FULL_ELEMENTS // K) // UE_VECTOR_SIZE * UE_VECTOR_SIZE)
    scores_sram = K * _BPE        # offset in URAM_A past the Q row
    for r in range(q_len):
        ue.accelerator_memory_to_sram(Q_DRAM_ADDR + r * K * _BPE, 0x00000, K)
        for j, n_take in ue.chunk_ranges(N, N_chunk):
            ue.accelerator_memory_to_sram(K_DRAM_ADDR + j * K * _BPE, 0x80000, n_take * K)
            ue.start_queue_for_bf16_matvec_operation(
                max_clear_en=1 if j == 0 else 0,
                fmax_context_addr=0,
                vector_sram_start_addr=0x00000,
                matrix_sram_start_addr=0x80000,
                output_sram_wb_addr=scores_sram + j * _BPE,
                K=K, N=n_take)
        ue.sram_to_accelerator_memory(scores_sram, SCORES_SCRATCH_ADDR + r * N * _BPE, N)
        if BIAS_DRAM_ADDR is not None:
            ue.accelerator_memory_to_sram(BIAS_DRAM_ADDR, 0x80000, N)
            ue.eltwise_add_core(scores_sram, 0x80000, scores_sram, N)
        ue.start_queue_for_bf16_softmax_operation(
            fmax_context_addr=0, vector_sram_start_addr=scores_sram,
            output_sram_wb_addr=scores_sram, N=N)
        v_chunk = max(UE_VECTOR_SIZE,
                      (URAM_NEAR_FULL_ELEMENTS // N // UE_VECTOR_SIZE) * UE_VECTOR_SIZE)
        out_sram = scores_sram + N * _BPE
        for v_col, v_take in ue.chunk_ranges(head_dim, v_chunk):
            ue.accelerator_memory_to_sram(
                V_T_DRAM_ADDR + v_col * N * _BPE, 0x80000, v_take * N)
            ue.start_queue_for_bf16_matvec_operation(
                max_clear_en=1 if v_col == 0 else 0,
                fmax_context_addr=1,
                vector_sram_start_addr=scores_sram,
                matrix_sram_start_addr=0x80000,
                output_sram_wb_addr=out_sram,
                K=N, N=v_take)
            ue.sram_to_accelerator_memory(
                out_sram, OUTPUT_DRAM_ADDR + r * head_dim * _BPE + v_col * _BPE, v_take)


def conv_transpose2d_2x2_dram(
    ue: UnifiedEngine,
    INPUT_DRAM_ADDR: int,
    OUTPUT_DRAM_ADDR: int,
    TEMP_DRAM_ADDR: int,
    WEIGHT_SLICES: list,   # list of 4 DRAM addrs: [(kh=0,kw=0),(kh=0,kw=1),(kh=1,kw=0),(kh=1,kw=1)]
    BIAS_DRAM_ADDR: int,
    H: int, W: int,
    C_in: int, C_out: int,
) -> None:
    """ConvTranspose2d kernel=2 stride=2, no padding, no overlap.

    Input:  (H*W, C_in)     stored row-major in INPUT_DRAM_ADDR
    Output: (2H*2W, C_out)  stored row-major in OUTPUT_DRAM_ADDR (must be pre-zeroed)
    TEMP:   (H*W, C_out)    scratch for per-slice matmul output

    C_out must already be padded to a multiple of 64 at the call site.
    Each of the 4 weight slices is (C_out, C_in) in DRAM.
    Bias (C_out,) is added in every slice matmul; since stride=kernel=2 each output pixel
    receives exactly one slice contribution, so bias is applied exactly once per pixel.
    """
    for s, (kh, kw) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        ue.matmat_mul_core(
            M=H * W, K=C_in, N=C_out,
            A_DRAM_ADDR=INPUT_DRAM_ADDR,
            B_DRAM_ADDR=WEIGHT_SLICES[s],
            OUTPUT_DRAM_ADDR=TEMP_DRAM_ADDR,
            C_DRAM_ADDR=BIAS_DRAM_ADDR,
            bias_mode="broadcast_N",
        )
        for i in range(H * W):
            h_i, w_i = i // W, i % W
            src     = TEMP_DRAM_ADDR + i * C_out * _BPE
            out_row = (2 * h_i + kh) * (2 * W) + (2 * w_i + kw)
            dst     = OUTPUT_DRAM_ADDR + out_row * C_out * _BPE
            ue.accelerator_memory_to_sram(src, 0x00000, C_out)
            ue.sram_to_accelerator_memory(0x00000, dst, C_out)

def conv2d_1x1_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                    WEIGHT_DRAM_ADDR: int, BIAS_DRAM_ADDR: int,
                    H: int, W: int, C_in: int, C_out: int,
                    gelu_enable: bool = False) -> None:
    """Conv2d(kernel=1) = matmul. HWC layout. Weight (C_out, C_in)."""
    ue.matmat_mul_core(
        M=H * W, K=C_in, N=C_out,
        A_DRAM_ADDR=INPUT_DRAM_ADDR, B_DRAM_ADDR=WEIGHT_DRAM_ADDR,
        OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
        C_DRAM_ADDR=BIAS_DRAM_ADDR, bias_mode="broadcast_N",
        gelu_enable=gelu_enable)


def conv2d_3x3_stride1_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                             IM2COL_DRAM_ADDR: int, WEIGHT_DRAM_ADDR: int, BIAS_DRAM_ADDR: int,
                             ZERO_PAD_DRAM_ADDR: int,
                             H: int, W: int, C_in: int, C_out: int,
                             gelu_enable: bool = False) -> None:
    """Conv2d(kernel=3, stride=1, pad=1) via row-by-row im2col + matmul. HWC layout.
    Weight pre-reshaped (C_out, 9*C_in). C_in and C_out multiples of 64.
    IM2COL_DRAM needs W*9*C_in elements (reused per row).
    """
    K = 9 * C_in
    W_CHUNK = max(1, URAM_NEAR_FULL_ELEMENTS // K)
    for h in range(H):
        for w_start in range(0, W, W_CHUNK):
            w_end = min(w_start + W_CHUNK, W)
            w_count = w_end - w_start
            sram_off = 0
            for w in range(w_start, w_end):
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        nh, nw = h + dy, w + dx
                        patch_sram = sram_off + ((dy + 1) * 3 + (dx + 1)) * C_in * _BPE
                        if 0 <= nh < H and 0 <= nw < W:
                            ue.accelerator_memory_to_sram(
                                INPUT_DRAM_ADDR + (nh * W + nw) * C_in * _BPE,
                                patch_sram, C_in)
                        else:
                            ue.accelerator_memory_to_sram(ZERO_PAD_DRAM_ADDR, patch_sram, C_in)
                sram_off += K * _BPE
            ue.sram_to_accelerator_memory(0x00000, IM2COL_DRAM_ADDR, w_count * K)
            ue.matmat_mul_core(
                M=w_count, K=K, N=C_out,
                A_DRAM_ADDR=IM2COL_DRAM_ADDR,
                B_DRAM_ADDR=WEIGHT_DRAM_ADDR,
                OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR + (h * W + w_start) * C_out * _BPE,
                C_DRAM_ADDR=BIAS_DRAM_ADDR, bias_mode="broadcast_N",
                gelu_enable=gelu_enable)


def conv2d_3x3_stride2_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                             IM2COL_DRAM_ADDR: int, WEIGHT_DRAM_ADDR: int, BIAS_DRAM_ADDR: int,
                             ZERO_PAD_DRAM_ADDR: int,
                             H_in: int, W_in: int, C_in: int, C_out: int,
                             gelu_enable: bool = False) -> None:
    """Conv2d(kernel=3, stride=2, pad=1) via row-by-row im2col + matmul. HWC layout.
    Weight pre-reshaped (C_out, 9*C_in). C_in and C_out multiples of 64.
    IM2COL_DRAM needs W_out*9*C_in elements (reused per row).
    """
    K = 9 * C_in
    H_out = (H_in - 1) // 2 + 1
    W_out = (W_in - 1) // 2 + 1
    W_CHUNK = max(1, URAM_NEAR_FULL_ELEMENTS // K)
    for h_out in range(H_out):
        h_in = h_out * 2
        for w_start in range(0, W_out, W_CHUNK):
            w_end = min(w_start + W_CHUNK, W_out)
            w_count = w_end - w_start
            sram_off = 0
            for w_out in range(w_start, w_end):
                w_in = w_out * 2
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        nh, nw = h_in + dy, w_in + dx
                        patch_sram = sram_off + ((dy + 1) * 3 + (dx + 1)) * C_in * _BPE
                        if 0 <= nh < H_in and 0 <= nw < W_in:
                            ue.accelerator_memory_to_sram(
                                INPUT_DRAM_ADDR + (nh * W_in + nw) * C_in * _BPE,
                                patch_sram, C_in)
                        else:
                            ue.accelerator_memory_to_sram(ZERO_PAD_DRAM_ADDR, patch_sram, C_in)
                sram_off += K * _BPE
            ue.sram_to_accelerator_memory(0x00000, IM2COL_DRAM_ADDR, w_count * K)
            ue.matmat_mul_core(
                M=w_count, K=K, N=C_out,
                A_DRAM_ADDR=IM2COL_DRAM_ADDR,
                B_DRAM_ADDR=WEIGHT_DRAM_ADDR,
                OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR + (h_out * W_out + w_start) * C_out * _BPE,
                C_DRAM_ADDR=BIAS_DRAM_ADDR, bias_mode="broadcast_N",
                gelu_enable=gelu_enable)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BIN_DIR = os.path.join(SCRIPT_DIR, "mobilesam_bin")
WEIGHTS = os.path.join(BIN_DIR, "mobile_sam.pt")

# ---------------------------------------------------------------------------
# Architecture constants
# ---------------------------------------------------------------------------

DEC_DIM     = 256          # token embed dim
DEC_HEADS   = 8            # attention heads
DEC_SA_HD   = 32           # self-attn head_dim  (internal_dim=256 / heads=8)
DEC_CA_HD   = 16           # cross-attn head_dim (internal_dim=128 / heads=8)
DEC_HD_PAD  = 64           # all head_dims padded to this (= UE_VECTOR_SIZE)
DEC_MLP_DIM = 2048         # MLP hidden dim (256 * 8)
DEC_LAYERS  = 2            # TwoWayTransformer layers
IMG_H = IMG_W = 64         # image feature map spatial size
GA          = IMG_H * IMG_W  # 4096 image tokens
NT          = 7            # 1 iou + 4 mask + 2 sparse (point + not-a-point pad)
NT_PAD      = 64           # token seq padded to multiple of UE_VECTOR_SIZE
BPE         = 2            # bytes per bf16 element

# Scale corrections: hw attention uses 1/sqrt(DEC_HD_PAD=64), but true scale is
# 1/sqrt(DEC_SA_HD=32) and 1/sqrt(DEC_CA_HD=16) respectively.
# Pre-multiply Q by this factor before calling flash_attention functions.
SA_SCALE_CORRECTION = math.sqrt(DEC_HD_PAD) / math.sqrt(DEC_SA_HD)  # sqrt(2)
CA_SCALE_CORRECTION = 1.0 / math.sqrt(DEC_CA_HD)  # 0.25 — raw matvec kernel, no internal sqrt scaling

ROW_CHUNK = UE_FMAX_CONTEXT_SIZE   # 64  (FMAX context depth for i2t chunking)

# ---------------------------------------------------------------------------
# Image encoder constants
# ---------------------------------------------------------------------------

ENC_IN_H = ENC_IN_W = 1024   # raw input image spatial size
ENC_C0   = 64                 # patch_embed output channels (real)
ENC_C0P  = 64                 # already 64-aligned, no extra padding needed
ENC_S0_H = ENC_S0_W = 256    # patch_embed output spatial (1024 / 2 / 2)

# C_in=3 padded to 64 for SRAM row alignment
ENC_CIN_PAD = 64
ENC_S0_C_EXP = 256   # Stage 0 MBConv expand dim (64 * 4)

# PatchMerging 0→1 and Stage 1 constants
ENC_S1_H = ENC_S1_W = 128   # spatial after PM 0→1
ENC_S1_C = 128               # channels after PM 0→1 (2 blocks of 64)
ENC_S1_WS       = 7          # window size
ENC_S1_PAD      = (ENC_S1_WS - ENC_S1_H % ENC_S1_WS) % ENC_S1_WS   # 5
ENC_S1_PH       = ENC_S1_H + ENC_S1_PAD   # 133
ENC_S1_NH       = ENC_S1_NW = ENC_S1_PH // ENC_S1_WS  # 19
ENC_S1_NWIN     = ENC_S1_NH * ENC_S1_NW   # 361
ENC_S1_WIN_REAL = ENC_S1_WS * ENC_S1_WS   # 49
ENC_S1_WIN_PAD  = 64
ENC_S1_HEADS    = 4
ENC_S1_HEAD_DIM = 32
ENC_S1_HEAD_PAD = 64
ENC_S1_MLP_HID  = 512

# PatchMerging 1→2 and Stage 2 constants
ENC_S2_H = ENC_S2_W = 64    # spatial after PM 1→2
ENC_S2_C = 160               # true channel count
ENC_S2_C_PAD = 192           # padded to next multiple of 64 (3×64)
ENC_S2_WS       = 14         # window size
ENC_S2_PAD      = (ENC_S2_WS - ENC_S2_H % ENC_S2_WS) % ENC_S2_WS   # 6
ENC_S2_PH       = ENC_S2_H + ENC_S2_PAD
ENC_S2_NH       = ENC_S2_NW = ENC_S2_PH // ENC_S2_WS
ENC_S2_NWIN     = ENC_S2_NH * ENC_S2_NW
ENC_S2_WIN_REAL = ENC_S2_WS * ENC_S2_WS   # 196
ENC_S2_WIN_PAD  = 256                      # next multiple of 64 ≥ 196
ENC_S2_HEADS    = 5
ENC_S2_HEAD_DIM = 32
ENC_S2_HEAD_PAD = 64
ENC_S2_MLP_HID  = 640

# PatchMerging 2→3 and Stage 3 constants
# PM23: stride=1, spatial stays 64×64, C: 160→320
ENC_S3_H = ENC_S3_W = 64    # spatial (unchanged by PM23 stride=1)
ENC_S3_C = 320               # 5×64, no padding needed
ENC_S3_WS       = 7
ENC_S3_PAD      = (ENC_S3_WS - ENC_S3_H % ENC_S3_WS) % ENC_S3_WS   # (7 - 64%7)%7 = 6
ENC_S3_PH       = ENC_S3_H + ENC_S3_PAD    # 70
ENC_S3_NH       = ENC_S3_NW = ENC_S3_PH // ENC_S3_WS   # 10
ENC_S3_NWIN     = ENC_S3_NH * ENC_S3_NW    # 100
ENC_S3_WIN_REAL = ENC_S3_WS * ENC_S3_WS    # 49
ENC_S3_WIN_PAD  = 64                        # next multiple of 64 ≥ 49
ENC_S3_HEADS    = 10
ENC_S3_HEAD_DIM = 32
ENC_S3_HEAD_PAD = 64
ENC_S3_MLP_HID  = 1280


# ---------------------------------------------------------------------------
# Static helper: unpad selection weight
# ---------------------------------------------------------------------------

def _build_unpad_weight(num_heads: int, head_dim: int, head_dim_pad: int) -> torch.Tensor:
    """(K=num_heads*head_dim_pad, N=num_heads*head_dim) identity-block matrix.

    matmul(padded_output [seq, K], W [K, N]) → unpadded_output [seq, N].
    Stored as W.T (N, K) for matmat_mul_core which computes A @ B^T.
    """
    K = num_heads * head_dim_pad
    N = num_heads * head_dim
    W = torch.zeros(K, N, dtype=torch.bfloat16)
    for h in range(num_heads):
        for d in range(head_dim):
            W[h * head_dim_pad + d, h * head_dim + d] = 1.0
    return W   # caller does .T.contiguous() before _alloc_param









# ---------------------------------------------------------------------------
# HW class
# ---------------------------------------------------------------------------

class MobileSAM_UE(UnifiedEngine):
    """MobileSAM mask decoder on the Unified Engine FPGA accelerator.

    Separation of compile and execute:
        prog = ue.compile_decoder()
        masks_hw, iou_hw = ue.run_decoder(prog, tokens_t, image_emb_t, image_pe_t, dense_t)

    Inputs (all bf16, batch=1):
        tokens_t    : (NT_PAD, 256) — pre-assembled token sequence (iou+mask+sparse), padded
        image_emb_t : (4096, 256)   — image embedding (H*W, C) HWC layout
        image_pe_t  : (4096, 256)   — image positional encoding
        dense_t     : (4096, 256)   — dense prompt (no_mask_embed broadcast)
    """

    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.init_unified_engine()
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.weight_init(sd)
        self._enc_init(sd)
        del sd
        self.tensor_init()

    # ------------------------------------------------------------------
    # DRAM helpers
    # ------------------------------------------------------------------

    def _alloc_param(self, tensor: torch.Tensor) -> int:
        t = tensor.to(torch.bfloat16).contiguous()
        addr = self.get_params_dram_addr()
        self.allocate_params_dram(t.numel() * BPE)
        self.dma_to_accelerator_memory(addr, t)
        return addr

    def _alloc_tensor(self, n_elements: int) -> int:
        return self.allocate_tensor_dram(n_elements * BPE)

    # ------------------------------------------------------------------
    # Weight init
    # ------------------------------------------------------------------

    def weight_init(self, sd: dict) -> None:
        md = "mask_decoder."

        # ---- Per-layer transformer weights (2 layers) ----
        self.dec_layer_weights = []
        for i in range(DEC_LAYERS):
            lp = md + f"transformer.layers.{i}."
            lw = {}

            # Self-attention: internal_dim=256, head_dim=32 → pad to 64
            for proj in ("q", "k", "v"):
                w_raw = sd[lp + f"self_attn.{proj}_proj.weight"].to(torch.bfloat16)  # (256, 256)
                b_raw = sd[lp + f"self_attn.{proj}_proj.bias"].to(torch.bfloat16)    # (256,)
                w_pad = torch.zeros(DEC_HEADS * DEC_HD_PAD, DEC_DIM, dtype=torch.bfloat16)
                b_pad = torch.zeros(DEC_HEADS * DEC_HD_PAD, dtype=torch.bfloat16)
                for h in range(DEC_HEADS):
                    w_pad[h*DEC_HD_PAD : h*DEC_HD_PAD+DEC_SA_HD] = \
                        w_raw[h*DEC_SA_HD : (h+1)*DEC_SA_HD]
                    b_pad[h*DEC_HD_PAD : h*DEC_HD_PAD+DEC_SA_HD] = \
                        b_raw[h*DEC_SA_HD : (h+1)*DEC_SA_HD]
                lw[f'sa_{proj}_w'] = self._alloc_param(w_pad)
                lw[f'sa_{proj}_b'] = self._alloc_param(b_pad)
            lw['sa_out_w'] = self._alloc_param(
                sd[lp + "self_attn.out_proj.weight"].to(torch.bfloat16))   # (256, 256)
            lw['sa_out_b'] = self._alloc_param(
                sd[lp + "self_attn.out_proj.bias"].to(torch.bfloat16))

            # Cross-attn t2i: internal_dim=128, head_dim=16 → pad to 64
            for proj in ("q", "k", "v"):
                w_raw = sd[lp + f"cross_attn_token_to_image.{proj}_proj.weight"].to(torch.bfloat16)
                b_raw = sd[lp + f"cross_attn_token_to_image.{proj}_proj.bias"].to(torch.bfloat16)
                w_pad = torch.zeros(DEC_HEADS * DEC_HD_PAD, DEC_DIM, dtype=torch.bfloat16)
                b_pad = torch.zeros(DEC_HEADS * DEC_HD_PAD, dtype=torch.bfloat16)
                for h in range(DEC_HEADS):
                    w_pad[h*DEC_HD_PAD : h*DEC_HD_PAD+DEC_CA_HD] = \
                        w_raw[h*DEC_CA_HD : (h+1)*DEC_CA_HD]
                    b_pad[h*DEC_HD_PAD : h*DEC_HD_PAD+DEC_CA_HD] = \
                        b_raw[h*DEC_CA_HD : (h+1)*DEC_CA_HD]
                lw[f't2i_{proj}_w'] = self._alloc_param(w_pad)
                lw[f't2i_{proj}_b'] = self._alloc_param(b_pad)
            lw['t2i_out_w'] = self._alloc_param(
                sd[lp + "cross_attn_token_to_image.out_proj.weight"].to(torch.bfloat16))
            lw['t2i_out_b'] = self._alloc_param(
                sd[lp + "cross_attn_token_to_image.out_proj.bias"].to(torch.bfloat16))

            # Cross-attn i2t: same dims as t2i
            for proj in ("q", "k", "v"):
                w_raw = sd[lp + f"cross_attn_image_to_token.{proj}_proj.weight"].to(torch.bfloat16)
                b_raw = sd[lp + f"cross_attn_image_to_token.{proj}_proj.bias"].to(torch.bfloat16)
                w_pad = torch.zeros(DEC_HEADS * DEC_HD_PAD, DEC_DIM, dtype=torch.bfloat16)
                b_pad = torch.zeros(DEC_HEADS * DEC_HD_PAD, dtype=torch.bfloat16)
                for h in range(DEC_HEADS):
                    w_pad[h*DEC_HD_PAD : h*DEC_HD_PAD+DEC_CA_HD] = \
                        w_raw[h*DEC_CA_HD : (h+1)*DEC_CA_HD]
                    b_pad[h*DEC_HD_PAD : h*DEC_HD_PAD+DEC_CA_HD] = \
                        b_raw[h*DEC_CA_HD : (h+1)*DEC_CA_HD]
                lw[f'i2t_{proj}_w'] = self._alloc_param(w_pad)
                lw[f'i2t_{proj}_b'] = self._alloc_param(b_pad)
            lw['i2t_out_w'] = self._alloc_param(
                sd[lp + "cross_attn_image_to_token.out_proj.weight"].to(torch.bfloat16))
            lw['i2t_out_b'] = self._alloc_param(
                sd[lp + "cross_attn_image_to_token.out_proj.bias"].to(torch.bfloat16))

            # Layer norms
            for n in range(1, 5):
                lw[f'norm{n}_w'] = self._alloc_param(
                    sd[lp + f"norm{n}.weight"].to(torch.bfloat16))
                lw[f'norm{n}_b'] = self._alloc_param(
                    sd[lp + f"norm{n}.bias"].to(torch.bfloat16))

            # MLP (lin1: 256→2048 + RELU, lin2: 2048→256)
            lw['mlp_lin1_w'] = self._alloc_param(
                sd[lp + "mlp.lin1.weight"].to(torch.bfloat16))   # (2048, 256)
            lw['mlp_lin1_b'] = self._alloc_param(
                sd[lp + "mlp.lin1.bias"].to(torch.bfloat16))
            lw['mlp_lin2_w'] = self._alloc_param(
                sd[lp + "mlp.lin2.weight"].to(torch.bfloat16))   # (256, 2048)
            lw['mlp_lin2_b'] = self._alloc_param(
                sd[lp + "mlp.lin2.bias"].to(torch.bfloat16))

            self.dec_layer_weights.append(lw)

        # ---- Final cross-attn (t2i) + norm_final_attn ----
        fp = md + "transformer.final_attn_token_to_image."
        self.dec_final_attn = {}
        for proj in ("q", "k", "v"):
            w_raw = sd[fp + f"{proj}_proj.weight"].to(torch.bfloat16)
            b_raw = sd[fp + f"{proj}_proj.bias"].to(torch.bfloat16)
            w_pad = torch.zeros(DEC_HEADS * DEC_HD_PAD, DEC_DIM, dtype=torch.bfloat16)
            b_pad = torch.zeros(DEC_HEADS * DEC_HD_PAD, dtype=torch.bfloat16)
            for h in range(DEC_HEADS):
                w_pad[h*DEC_HD_PAD : h*DEC_HD_PAD+DEC_CA_HD] = \
                    w_raw[h*DEC_CA_HD : (h+1)*DEC_CA_HD]
                b_pad[h*DEC_HD_PAD : h*DEC_HD_PAD+DEC_CA_HD] = \
                    b_raw[h*DEC_CA_HD : (h+1)*DEC_CA_HD]
            self.dec_final_attn[f'{proj}_w'] = self._alloc_param(w_pad)
            self.dec_final_attn[f'{proj}_b'] = self._alloc_param(b_pad)
        self.dec_final_attn['out_w'] = self._alloc_param(
            sd[fp + "out_proj.weight"].to(torch.bfloat16))
        self.dec_final_attn['out_b'] = self._alloc_param(
            sd[fp + "out_proj.bias"].to(torch.bfloat16))
        nfa = md + "transformer.norm_final_attn."
        self.dec_final_norm = {
            'w': self._alloc_param(sd[nfa + "weight"].to(torch.bfloat16)),
            'b': self._alloc_param(sd[nfa + "bias"].to(torch.bfloat16)),
        }

        # ---- Unpad weights (constructed, not from checkpoint) ----
        # SA:  (heads*HD_PAD=512, heads*HD=256) stored as (256, 512) for matmat
        self.DEC_SA_UNPAD_W = self._alloc_param(
            _build_unpad_weight(DEC_HEADS, DEC_SA_HD, DEC_HD_PAD).T.contiguous())
        # CA:  (heads*HD_PAD=512, heads*HD=128) stored as (128, 512)
        self.DEC_CA_UNPAD_W = self._alloc_param(
            _build_unpad_weight(DEC_HEADS, DEC_CA_HD, DEC_HD_PAD).T.contiguous())

        # ---- Attention biases (padding masks) ----
        # Self-attn: mask padded kv positions [NT:NT_PAD] with -inf
        sa_bias = torch.zeros(DEC_HEADS, NT_PAD, NT_PAD, dtype=torch.bfloat16)
        sa_bias[:, :, NT:] = float('-inf')
        self.DEC_SA_BIAS = self._alloc_param(sa_bias.reshape(-1))

        # i2t: per-row 1-D kv mask applied inside cross_attn_flash_single_head
        # masks padded token positions [NT:NT_PAD] in kv dim
        i2t_kv_mask = torch.zeros(NT_PAD, dtype=torch.bfloat16)
        i2t_kv_mask[NT:] = float('-inf')
        self.DEC_I2T_KV_MASK = self._alloc_param(i2t_kv_mask)

        # ---- Output upscaling ----
        # ConvTranspose2d(256→64, kernel=2, stride=2): weight (256, 64, 2, 2)
        # Decomposed into 4 sub-weights (C_out=64, C_in=256) for conv_transpose2d_2x2_dram
        w_up0 = sd[md + "output_upscaling.0.weight"].to(torch.bfloat16)  # (256, 64, 2, 2)
        self.DEC_UP0_W = []
        for ky in range(2):
            for kx in range(2):
                self.DEC_UP0_W.append(
                    self._alloc_param(w_up0[:, :, ky, kx].T.contiguous()))  # (64, 256)
        self.DEC_UP0_B = self._alloc_param(
            sd[md + "output_upscaling.0.bias"].to(torch.bfloat16))   # (64,)

        # LayerNorm2d(64) between the two ConvTranspose2d
        self.DEC_UP_LN_W = self._alloc_param(
            sd[md + "output_upscaling.1.weight"].to(torch.bfloat16))  # (64,)
        self.DEC_UP_LN_B = self._alloc_param(
            sd[md + "output_upscaling.1.bias"].to(torch.bfloat16))

        # ConvTranspose2d(64→32, kernel=2, stride=2): weight (64, 32, 2, 2)
        # Pad C_out 32→64 so matmul N is aligned
        w_up1 = sd[md + "output_upscaling.3.weight"].to(torch.bfloat16)  # (64, 32, 2, 2)
        self.DEC_UP1_W = []
        for ky in range(2):
            for kx in range(2):
                sub_raw = w_up1[:, :, ky, kx].T.contiguous()   # (32, 64)
                sub = torch.zeros(64, 64, dtype=torch.bfloat16)
                sub[:32] = sub_raw
                self.DEC_UP1_W.append(self._alloc_param(sub))  # (64, 64)
        b_raw = sd[md + "output_upscaling.3.bias"].to(torch.bfloat16)  # (32,)
        b_pad = torch.zeros(64, dtype=torch.bfloat16)
        b_pad[:32] = b_raw
        self.DEC_UP1_B = self._alloc_param(b_pad)

        # ---- Hypernetwork MLPs (4 × MLP3: 256→256→256→32) ----
        # layers.0: (256,256) relu, layers.1: (256,256) relu, layers.2: (32,256) linear
        self.dec_hyper_weights = []
        for m in range(4):
            hp = md + f"output_hypernetworks_mlps.{m}.layers."
            hw = {
                'l0_w': self._alloc_param(sd[hp + "0.weight"].to(torch.bfloat16)),
                'l0_b': self._alloc_param(sd[hp + "0.bias"].to(torch.bfloat16)),
                'l1_w': self._alloc_param(sd[hp + "1.weight"].to(torch.bfloat16)),
                'l1_b': self._alloc_param(sd[hp + "1.bias"].to(torch.bfloat16)),
            }
            w_raw = sd[hp + "2.weight"].to(torch.bfloat16)   # (32, 256)
            w_pad = torch.zeros(64, 256, dtype=torch.bfloat16)
            w_pad[:32] = w_raw
            hw['l2_w'] = self._alloc_param(w_pad)
            b_raw = sd[hp + "2.bias"].to(torch.bfloat16)
            b_pad = torch.zeros(64, dtype=torch.bfloat16)
            b_pad[:32] = b_raw
            hw['l2_b'] = self._alloc_param(b_pad)
            self.dec_hyper_weights.append(hw)

        # ---- IoU prediction head (MLP3: 256→256→256→4) ----
        ip = md + "iou_prediction_head.layers."
        iou_w = {
            'l0_w': self._alloc_param(sd[ip + "0.weight"].to(torch.bfloat16)),
            'l0_b': self._alloc_param(sd[ip + "0.bias"].to(torch.bfloat16)),
            'l1_w': self._alloc_param(sd[ip + "1.weight"].to(torch.bfloat16)),
            'l1_b': self._alloc_param(sd[ip + "1.bias"].to(torch.bfloat16)),
        }
        w_raw = sd[ip + "2.weight"].to(torch.bfloat16)   # (4, 256)
        w_pad = torch.zeros(64, 256, dtype=torch.bfloat16)
        w_pad[:4] = w_raw
        iou_w['l2_w'] = self._alloc_param(w_pad)
        b_raw = sd[ip + "2.bias"].to(torch.bfloat16)
        b_pad = torch.zeros(64, dtype=torch.bfloat16)
        b_pad[:4] = b_raw
        iou_w['l2_b'] = self._alloc_param(b_pad)
        self.dec_iou_weights = iou_w

        # GELU identity weight: used in output upscaling between ConvTranspose2d layers
        gelu_id_w = torch.eye(64, dtype=torch.bfloat16)
        self.GELU_ID_ADDR = self._alloc_param(gelu_id_w)

        print(f"  Params DRAM: {self.get_params_dram_usage() / 1024**2:.1f} MB")

    # ------------------------------------------------------------------
    # Tensor init
    # ------------------------------------------------------------------

    def tensor_init(self) -> None:
        # Input tensors (DMAed before each execution)
        self.TOKENS_DRAM    = self._alloc_tensor(NT_PAD * DEC_DIM)   # (64, 256) mutable tokens
        self.TOKENS_PE_DRAM = self._alloc_tensor(NT_PAD * DEC_DIM)   # (64, 256) constant token PE
        self.SRC_DRAM       = self._alloc_tensor(GA * DEC_DIM)        # (4096, 256) image src, mutable
        self.KEY_PE_DRAM    = self._alloc_tensor(GA * DEC_DIM)        # (4096, 256) image PE, constant

        # Temporaries for PE-added sequences
        self.Q_IN_DRAM   = self._alloc_tensor(NT_PAD * DEC_DIM)  # tokens + query_pe (for layer 1)
        self.SRC_PE_DRAM = self._alloc_tensor(GA * DEC_DIM)       # src + key_pe  (for t2i Q, i2t K)
        self.TOK_PE_DRAM = self._alloc_tensor(NT_PAD * DEC_DIM)   # tokens + query_pe (for i2t K/K_fin)

        # ---- Self-attention buffers (seq=64, head_dim=64) ----
        SA_PROJ = NT_PAD * DEC_HEADS * DEC_HD_PAD   # 64 * 512
        self.SA_Q_PROJ  = self._alloc_tensor(SA_PROJ)
        self.SA_K_PROJ  = self._alloc_tensor(SA_PROJ)
        self.SA_V_PROJ  = self._alloc_tensor(SA_PROJ)  # V from tokens (no PE)

        SA_HEADS = DEC_HEADS * NT_PAD * DEC_HD_PAD    # 8 * 64 * 64
        self.SA_Q_HEADS   = self._alloc_tensor(SA_HEADS)
        self.SA_K_HEADS   = self._alloc_tensor(SA_HEADS)
        self.SA_V_HEADS   = self._alloc_tensor(SA_HEADS)
        self.SA_ATTN_OUT  = self._alloc_tensor(SA_HEADS)
        # flash_attention_batched scratch: (head_dim * seq_len + seq_len^2) per head
        SA_SCRATCH = DEC_HEADS * (DEC_HD_PAD * NT_PAD + NT_PAD * NT_PAD)
        self.SA_SCRATCH   = self._alloc_tensor(SA_SCRATCH)
        self.SA_MERGE_TMP = self._alloc_tensor(SA_HEADS)    # permute temp for multihead_merge
        self.SA_MERGED    = self._alloc_tensor(NT_PAD * DEC_DIM)      # (64, 256) after unpad
        self.SA_OUT       = self._alloc_tensor(NT_PAD * DEC_DIM)      # after out_proj

        # ---- Cross-attn t2i: Q=tokens(64), K/V=image(4096), head_dim=64 ----
        self.T2I_Q_PROJ   = self._alloc_tensor(NT_PAD * DEC_HEADS * DEC_HD_PAD)   # (64, 512)
        self.T2I_K_PROJ   = self._alloc_tensor(GA * DEC_HEADS * DEC_HD_PAD)        # (4096, 512)
        self.T2I_V_PROJ   = self._alloc_tensor(GA * DEC_HEADS * DEC_HD_PAD)        # (4096, 512)
        self.T2I_Q_HEADS  = self._alloc_tensor(DEC_HEADS * NT_PAD * DEC_HD_PAD)    # (8, 64, 64)
        self.T2I_K_HEADS  = self._alloc_tensor(DEC_HEADS * GA * DEC_HD_PAD)        # (8, 4096, 64)
        self.T2I_V_HEADS  = self._alloc_tensor(DEC_HEADS * GA * DEC_HD_PAD)        # (8, 4096, 64)
        self.T2I_VT       = self._alloc_tensor(DEC_HEADS * DEC_HD_PAD * GA)        # (8, 64, 4096)
        self.T2I_ATTN_OUT = self._alloc_tensor(DEC_HEADS * NT_PAD * DEC_HD_PAD)    # (8, 64, 64)
        self.T2I_SCORES   = self._alloc_tensor(NT_PAD * GA)                         # scratch (64*4096)
        self.T2I_MERGE_TMP= self._alloc_tensor(DEC_HEADS * NT_PAD * DEC_HD_PAD)
        self.T2I_MERGED   = self._alloc_tensor(NT_PAD * DEC_HEADS * DEC_CA_HD)     # (64, 128)
        self.T2I_OUT      = self._alloc_tensor(NT_PAD * DEC_DIM)                    # (64, 256)

        # ---- Cross-attn i2t: Q=image(4096), K/V=tokens(64), head_dim=64 ----
        self.I2T_Q_PROJ   = self._alloc_tensor(GA * DEC_HEADS * DEC_HD_PAD)         # (4096, 512)
        self.I2T_K_PROJ   = self._alloc_tensor(NT_PAD * DEC_HEADS * DEC_HD_PAD)     # (64, 512)
        self.I2T_V_PROJ   = self._alloc_tensor(NT_PAD * DEC_HEADS * DEC_HD_PAD)     # (64, 512)
        self.I2T_Q_HEADS  = self._alloc_tensor(DEC_HEADS * GA * DEC_HD_PAD)         # (8, 4096, 64)
        self.I2T_K_HEADS  = self._alloc_tensor(DEC_HEADS * NT_PAD * DEC_HD_PAD)     # (8, 64, 64)
        self.I2T_V_HEADS  = self._alloc_tensor(DEC_HEADS * NT_PAD * DEC_HD_PAD)     # (8, 64, 64)
        self.I2T_VT       = self._alloc_tensor(DEC_HEADS * DEC_HD_PAD * NT_PAD)     # (8, 64, 64)
        self.I2T_ATTN_OUT = self._alloc_tensor(DEC_HEADS * GA * DEC_HD_PAD)         # (8, 4096, 64)
        self.I2T_SCORES   = self._alloc_tensor(GA * NT_PAD)                          # scratch (4096*64)
        self.I2T_MERGE_TMP= self._alloc_tensor(DEC_HEADS * GA * DEC_HD_PAD)
        self.I2T_MERGED   = self._alloc_tensor(GA * DEC_HEADS * DEC_CA_HD)          # (4096, 128)
        self.I2T_OUT      = self._alloc_tensor(GA * DEC_DIM)                         # (4096, 256)

        # ---- Norm + MLP scratch (token side) ----
        self.TOK_NORM = self._alloc_tensor(NT_PAD * DEC_DIM)
        self.MLP_MID  = self._alloc_tensor(NT_PAD * DEC_MLP_DIM)
        self.MLP_OUT  = self._alloc_tensor(NT_PAD * DEC_DIM)

        # ---- Norm scratch (image side, for norm4) ----
        self.SRC_NORM = self._alloc_tensor(GA * DEC_DIM)

        # ---- Output upscaling ----
        UP0_H = IMG_H * 2    # 128
        UP0_W = IMG_W * 2    # 128
        UP1_H = IMG_H * 4    # 256
        UP1_W = IMG_W * 4    # 256
        self.UP0_OUT    = self._alloc_tensor(UP0_H * UP0_W * 64)   # (128, 128, 64)
        self.UP0_SCATTER= self._alloc_tensor(GA * 64)               # scatter temp for conv_t2d
        self.UP_LN_OUT  = self._alloc_tensor(UP0_H * UP0_W * 64)   # (128, 128, 64)
        self.UP1_OUT    = self._alloc_tensor(UP1_H * UP1_W * 64)   # (256, 256, 64) padded C
        self.UP1_SCATTER= self._alloc_tensor(UP0_H * UP0_W * 64)   # scatter temp

        # ---- Hypernetwork + IoU scratch ----
        self.HYPER_MID1 = self._alloc_tensor(DEC_DIM)   # single-token MLP intermediate
        self.HYPER_MID2 = self._alloc_tensor(DEC_DIM)
        self.HYPER_OUT  = self._alloc_tensor(4 * 64)     # 4 mask outputs, 64 padded each
        self.IOU_MID1   = self._alloc_tensor(DEC_DIM)
        self.IOU_MID2   = self._alloc_tensor(DEC_DIM)
        self.IOU_OUT    = self._alloc_tensor(64)          # (1, 64) padded, real 4 values

        # ---- Final masks: (4, H*W) for H=W=256 ----
        self.MASK_OUT = self._alloc_tensor(4 * UP1_H * UP1_W)   # (4, 65536)

        print(f"  Tensor DRAM:  {self.get_tensor_dram_usage() / 1024**2:.1f} MB")

    # ------------------------------------------------------------------
    # Internal compile helpers
    # ------------------------------------------------------------------

    def _self_attn(self, lw: dict, skip_pe: bool) -> None:
        """Compile one self-attention block on TOKENS_DRAM.

        If skip_pe: Q=K=TOKENS_DRAM (layer 0, no PE).
        If not skip_pe: Q=K=Q_IN_DRAM (TOKENS + TOKENS_PE, must be pre-computed).
        V always uses TOKENS_DRAM (without PE).
        Output: SA_OUT (NT_PAD, DEC_DIM).
        """
        q_src = self.TOKENS_DRAM if skip_pe else self.Q_IN_DRAM

        # Q, K projections from q_src; V projection from tokens (no PE)
        for proj, src_addr, dst_addr in [
            ('q', q_src,              self.SA_Q_PROJ),
            ('k', q_src,              self.SA_K_PROJ),
            ('v', self.TOKENS_DRAM,   self.SA_V_PROJ),
        ]:
            self.matmat_mul_core(
                M=NT_PAD, K=DEC_DIM, N=DEC_HEADS * DEC_HD_PAD,
                A_DRAM_ADDR=src_addr,
                B_DRAM_ADDR=lw[f'sa_{proj}_w'],
                OUTPUT_DRAM_ADDR=dst_addr,
                C_DRAM_ADDR=lw[f'sa_{proj}_b'],
                bias_mode="broadcast_N",
            )

        # Reshape (NT_PAD, heads*hd_pad) → (heads, NT_PAD, hd_pad)
        # matmul already outputs interleaved padded layout; only permute needed
        for proj_addr, heads_addr in [
            (self.SA_Q_PROJ, self.SA_Q_HEADS),
            (self.SA_K_PROJ, self.SA_K_HEADS),
            (self.SA_V_PROJ, self.SA_V_HEADS),
        ]:
            self.bf16_permute_core(
                dim_0=NT_PAD, dim_1=DEC_HEADS, dim_2=DEC_HD_PAD,
                INPUT_DRAM_ADDR=proj_addr, OUTPUT_DRAM_ADDR=heads_addr,
            )

        # Pre-scale Q by SA_SCALE_CORRECTION so flash_attn (1/sqrt(64)) gives 1/sqrt(32)
        broadcast_mul_dram(self, self.SA_Q_HEADS,
                           SA_SCALE_CORRECTION,
                           DEC_HEADS * NT_PAD * DEC_HD_PAD)

        # flash_attention_batched: (8 batches, seq=64, head_dim=64)
        dram_zero_fill(self, self.SA_ATTN_OUT, DEC_HEADS * NT_PAD * DEC_HD_PAD)
        flash_attention_batched(
            self,
            num_batches=DEC_HEADS,
            head_dim=DEC_HD_PAD,
            seq_len=NT_PAD,
            Q_DRAM_ADDR=self.SA_Q_HEADS,
            K_DRAM_ADDR=self.SA_K_HEADS,
            V_DRAM_ADDR=self.SA_V_HEADS,
            OUTPUT_DRAM_ADDR=self.SA_ATTN_OUT,
            SCRATCH_DRAM_ADDR=self.SA_SCRATCH,
            BIAS_DRAM_ADDR=self.DEC_SA_BIAS,
        )

        # Merge (heads, NT_PAD, hd_pad) → (NT_PAD, 256) via permute + unpad matmul
        multihead_merge_dram(
            self,
            INPUT_DRAM_ADDR=self.SA_ATTN_OUT,
            OUTPUT_DRAM_ADDR=self.SA_MERGED,
            TEMP_DRAM_ADDR=self.SA_MERGE_TMP,
            seq_len=NT_PAD, num_heads=DEC_HEADS,
            head_dim=DEC_SA_HD, head_dim_pad=DEC_HD_PAD,
            UNPAD_WEIGHT_ADDR=self.DEC_SA_UNPAD_W,
        )

        # out_proj: (NT_PAD, 256) @ (256, 256)^T + bias → (NT_PAD, 256)
        self.matmat_mul_core(
            M=NT_PAD, K=DEC_DIM, N=DEC_DIM,
            A_DRAM_ADDR=self.SA_MERGED,
            B_DRAM_ADDR=lw['sa_out_w'],
            OUTPUT_DRAM_ADDR=self.SA_OUT,
            C_DRAM_ADDR=lw['sa_out_b'],
            bias_mode="broadcast_N",
        )

    def _cross_attn_t2i(self, q_w: int, q_b: int, k_w: int, k_b: int,
                         v_w: int, v_b: int, out_w: int, out_b: int,
                         Q_SRC: int, KV_SRC: int, V_SRC: int,
                         q_len: int) -> None:
        """Token-to-image cross-attn. Output written to T2I_OUT.

        Q_SRC: (q_len, 256) — tokens (+ pe if applicable), pre-assembled
        KV_SRC: (4096, 256) — image src + key_pe  (for K)
        V_SRC:  (4096, 256) — image src without pe (for V)
        q_len: padded token count (NT_PAD = 64)
        """
        # Q proj: (q_len, 256) → (q_len, 512)
        self.matmat_mul_core(
            M=q_len, K=DEC_DIM, N=DEC_HEADS * DEC_HD_PAD,
            A_DRAM_ADDR=Q_SRC, B_DRAM_ADDR=q_w,
            OUTPUT_DRAM_ADDR=self.T2I_Q_PROJ,
            C_DRAM_ADDR=q_b, bias_mode="broadcast_N",
        )
        # K proj: (4096, 256) → (4096, 512)
        self.matmat_mul_core(
            M=GA, K=DEC_DIM, N=DEC_HEADS * DEC_HD_PAD,
            A_DRAM_ADDR=KV_SRC, B_DRAM_ADDR=k_w,
            OUTPUT_DRAM_ADDR=self.T2I_K_PROJ,
            C_DRAM_ADDR=k_b, bias_mode="broadcast_N",
        )
        # V proj: (4096, 256) → (4096, 512)
        self.matmat_mul_core(
            M=GA, K=DEC_DIM, N=DEC_HEADS * DEC_HD_PAD,
            A_DRAM_ADDR=V_SRC, B_DRAM_ADDR=v_w,
            OUTPUT_DRAM_ADDR=self.T2I_V_PROJ,
            C_DRAM_ADDR=v_b, bias_mode="broadcast_N",
        )

        # Reshape → multi-head layout (permute only; matmul output already interleaved padded)
        self.bf16_permute_core(
            dim_0=q_len, dim_1=DEC_HEADS, dim_2=DEC_HD_PAD,
            INPUT_DRAM_ADDR=self.T2I_Q_PROJ, OUTPUT_DRAM_ADDR=self.T2I_Q_HEADS,
        )
        self.bf16_permute_core(
            dim_0=GA, dim_1=DEC_HEADS, dim_2=DEC_HD_PAD,
            INPUT_DRAM_ADDR=self.T2I_K_PROJ, OUTPUT_DRAM_ADDR=self.T2I_K_HEADS,
        )
        self.bf16_permute_core(
            dim_0=GA, dim_1=DEC_HEADS, dim_2=DEC_HD_PAD,
            INPUT_DRAM_ADDR=self.T2I_V_PROJ, OUTPUT_DRAM_ADDR=self.T2I_V_HEADS,
        )

        # Pre-scale Q: correction so combined scale = 1/sqrt(DEC_CA_HD=16)
        broadcast_mul_dram(self, self.T2I_Q_HEADS,
                           CA_SCALE_CORRECTION,
                           DEC_HEADS * q_len * DEC_HD_PAD)

        # Transpose V per head: (kv_len, hd_pad) → (hd_pad, kv_len)
        q_stride  = q_len * DEC_HD_PAD
        kv_stride = GA * DEC_HD_PAD
        for h in range(DEC_HEADS):
            self.bf16_transpose_core(
                M=GA, N=DEC_HD_PAD,
                INPUT_DRAM_ADDR=self.T2I_V_HEADS + h * kv_stride * BPE,
                OUTPUT_DRAM_ADDR=self.T2I_VT    + h * DEC_HD_PAD * GA * BPE,
            )

        # cross_attn_flash_single_head per head: q=64, kv=4096, hd=64
        for h in range(DEC_HEADS):
            cross_attn_flash_single_head(
                ue=self, head_dim=DEC_HD_PAD,
                q_len=q_len, kv_len=GA,
                Q_DRAM_ADDR     = self.T2I_Q_HEADS + h * q_stride  * BPE,
                K_DRAM_ADDR     = self.T2I_K_HEADS + h * kv_stride * BPE,
                V_T_DRAM_ADDR   = self.T2I_VT      + h * DEC_HD_PAD * GA * BPE,
                OUTPUT_DRAM_ADDR= self.T2I_ATTN_OUT+ h * q_stride  * BPE,
                SCORES_SCRATCH_ADDR=self.T2I_SCORES,
            )

        # Merge + unpad: (heads, q_len, hd_pad) → (q_len, heads*hd) = (64, 128)
        multihead_merge_dram(
            self,
            INPUT_DRAM_ADDR=self.T2I_ATTN_OUT,
            OUTPUT_DRAM_ADDR=self.T2I_MERGED,
            TEMP_DRAM_ADDR=self.T2I_MERGE_TMP,
            seq_len=q_len, num_heads=DEC_HEADS,
            head_dim=DEC_CA_HD, head_dim_pad=DEC_HD_PAD,
            UNPAD_WEIGHT_ADDR=self.DEC_CA_UNPAD_W,
        )

        # out_proj: (q_len, 128) @ (256, 128)^T + bias → (q_len, 256)
        self.matmat_mul_core(
            M=q_len, K=DEC_HEADS * DEC_CA_HD, N=DEC_DIM,
            A_DRAM_ADDR=self.T2I_MERGED,
            B_DRAM_ADDR=out_w,
            OUTPUT_DRAM_ADDR=self.T2I_OUT,
            C_DRAM_ADDR=out_b,
            bias_mode="broadcast_N",
        )

    def _cross_attn_i2t(self, q_w: int, q_b: int, k_w: int, k_b: int,
                         v_w: int, v_b: int, out_w: int, out_b: int,
                         Q_SRC: int, KV_SRC: int, V_SRC: int) -> None:
        """Image-to-token cross-attn. Output written to I2T_OUT.

        Q_SRC: (4096, 256) — image src + key_pe
        KV_SRC: (NT_PAD, 256) — tokens + query_pe  (for K)
        V_SRC:  (NT_PAD, 256) — tokens without pe  (for V)
        """
        # Q proj: (4096, 256) → (4096, 512)
        self.matmat_mul_core(
            M=GA, K=DEC_DIM, N=DEC_HEADS * DEC_HD_PAD,
            A_DRAM_ADDR=Q_SRC, B_DRAM_ADDR=q_w,
            OUTPUT_DRAM_ADDR=self.I2T_Q_PROJ,
            C_DRAM_ADDR=q_b, bias_mode="broadcast_N",
        )
        # K proj: (NT_PAD, 256) → (NT_PAD, 512)
        self.matmat_mul_core(
            M=NT_PAD, K=DEC_DIM, N=DEC_HEADS * DEC_HD_PAD,
            A_DRAM_ADDR=KV_SRC, B_DRAM_ADDR=k_w,
            OUTPUT_DRAM_ADDR=self.I2T_K_PROJ,
            C_DRAM_ADDR=k_b, bias_mode="broadcast_N",
        )
        # V proj: (NT_PAD, 256) → (NT_PAD, 512)
        self.matmat_mul_core(
            M=NT_PAD, K=DEC_DIM, N=DEC_HEADS * DEC_HD_PAD,
            A_DRAM_ADDR=V_SRC, B_DRAM_ADDR=v_w,
            OUTPUT_DRAM_ADDR=self.I2T_V_PROJ,
            C_DRAM_ADDR=v_b, bias_mode="broadcast_N",
        )

        # Reshape → multi-head (permute only; matmul output already interleaved padded)
        self.bf16_permute_core(
            dim_0=GA, dim_1=DEC_HEADS, dim_2=DEC_HD_PAD,
            INPUT_DRAM_ADDR=self.I2T_Q_PROJ, OUTPUT_DRAM_ADDR=self.I2T_Q_HEADS,
        )
        self.bf16_permute_core(
            dim_0=NT_PAD, dim_1=DEC_HEADS, dim_2=DEC_HD_PAD,
            INPUT_DRAM_ADDR=self.I2T_K_PROJ, OUTPUT_DRAM_ADDR=self.I2T_K_HEADS,
        )
        self.bf16_permute_core(
            dim_0=NT_PAD, dim_1=DEC_HEADS, dim_2=DEC_HD_PAD,
            INPUT_DRAM_ADDR=self.I2T_V_PROJ, OUTPUT_DRAM_ADDR=self.I2T_V_HEADS,
        )

        # Pre-scale Q
        broadcast_mul_dram(self, self.I2T_Q_HEADS,
                           CA_SCALE_CORRECTION,
                           DEC_HEADS * GA * DEC_HD_PAD)

        # Transpose V per head: (NT_PAD, hd_pad) → (hd_pad, NT_PAD)
        q_stride  = GA    * DEC_HD_PAD
        kv_stride = NT_PAD * DEC_HD_PAD
        for h in range(DEC_HEADS):
            self.bf16_transpose_core(
                M=NT_PAD, N=DEC_HD_PAD,
                INPUT_DRAM_ADDR=self.I2T_V_HEADS + h * kv_stride * BPE,
                OUTPUT_DRAM_ADDR=self.I2T_VT     + h * DEC_HD_PAD * NT_PAD * BPE,
            )

        # cross_attn_flash_single_head per head, chunked by ROW_CHUNK=64
        # q=4096, kv=64, hd=64
        for h in range(DEC_HEADS):
            q_h_addr   = self.I2T_Q_HEADS + h * q_stride  * BPE
            k_h_addr   = self.I2T_K_HEADS + h * kv_stride * BPE
            vt_h_addr  = self.I2T_VT      + h * DEC_HD_PAD * NT_PAD * BPE
            out_h_addr = self.I2T_ATTN_OUT+ h * q_stride  * BPE

            for row_base in range(0, GA, ROW_CHUNK):
                row_take = min(ROW_CHUNK, GA - row_base)
                cross_attn_flash_single_head(
                    ue=self, head_dim=DEC_HD_PAD,
                    q_len=row_take, kv_len=NT_PAD,
                    Q_DRAM_ADDR     = q_h_addr   + row_base * DEC_HD_PAD * BPE,
                    K_DRAM_ADDR     = k_h_addr,
                    V_T_DRAM_ADDR   = vt_h_addr,
                    OUTPUT_DRAM_ADDR= out_h_addr + row_base * DEC_HD_PAD * BPE,
                    SCORES_SCRATCH_ADDR=self.I2T_SCORES,
                    BIAS_DRAM_ADDR  = self.DEC_I2T_KV_MASK,
                )

        # Merge + unpad: (heads, 4096, hd_pad) → (4096, 128)
        multihead_merge_dram(
            self,
            INPUT_DRAM_ADDR=self.I2T_ATTN_OUT,
            OUTPUT_DRAM_ADDR=self.I2T_MERGED,
            TEMP_DRAM_ADDR=self.I2T_MERGE_TMP,
            seq_len=GA, num_heads=DEC_HEADS,
            head_dim=DEC_CA_HD, head_dim_pad=DEC_HD_PAD,
            UNPAD_WEIGHT_ADDR=self.DEC_CA_UNPAD_W,
        )

        # out_proj: (4096, 128) @ (256, 128)^T + bias → (4096, 256)
        self.matmat_mul_core(
            M=GA, K=DEC_HEADS * DEC_CA_HD, N=DEC_DIM,
            A_DRAM_ADDR=self.I2T_MERGED,
            B_DRAM_ADDR=out_w,
            OUTPUT_DRAM_ADDR=self.I2T_OUT,
            C_DRAM_ADDR=out_b,
            bias_mode="broadcast_N",
        )

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------

    def compile_decoder(self) -> int:
        """Compile the full mask decoder to a single instruction stream.

        Returns the program DRAM address.
        """
        self.start_capture()

        # === Step 0: src = image_emb + dense (written to SRC_DRAM before execute) ===
        # NOTE: image_emb + dense is computed CPU-side and DMAed into SRC_DRAM directly.
        # SRC_DRAM holds the result. KEY_PE_DRAM and TOKENS_PE_DRAM are constant.

        # === TwoWayTransformer: 2 layers ===
        for layer_i in range(DEC_LAYERS):
            lw = self.dec_layer_weights[layer_i]
            skip_pe = (layer_i == 0)

            # ---- Prepare q_in (tokens + query_pe if not skip_pe) ----
            if not skip_pe:
                eltwise_add_dram(self,
                    self.TOKENS_DRAM, self.TOKENS_PE_DRAM,
                    self.Q_IN_DRAM, NT_PAD * DEC_DIM)

            # ---- Self-attention on tokens ----
            self._self_attn(lw, skip_pe)

            # queries += SA_OUT  then norm1
            eltwise_add_dram(self,
                self.TOKENS_DRAM, self.SA_OUT,
                self.TOKENS_DRAM, NT_PAD * DEC_DIM)
            self.layer_norm_core_dram(
                M=NT_PAD, N=DEC_DIM,
                A_DRAM_ADDR=self.TOKENS_DRAM,
                OUTPUT_DRAM_ADDR=self.TOKENS_DRAM,
                GAMMA_DRAM_ADDR=lw['norm1_w'],
                BETA_DRAM_ADDR=lw['norm1_b'],
            )

            # ---- Cross-attn token→image ----
            # Q = tokens + query_pe  →  always add pe before t2i
            eltwise_add_dram(self,
                self.TOKENS_DRAM, self.TOKENS_PE_DRAM,
                self.Q_IN_DRAM, NT_PAD * DEC_DIM)
            # K/V source: src + key_pe (for K), src (for V)
            eltwise_add_dram(self,
                self.SRC_DRAM, self.KEY_PE_DRAM,
                self.SRC_PE_DRAM, GA * DEC_DIM)

            self._cross_attn_t2i(
                q_w=lw['t2i_q_w'], q_b=lw['t2i_q_b'],
                k_w=lw['t2i_k_w'], k_b=lw['t2i_k_b'],
                v_w=lw['t2i_v_w'], v_b=lw['t2i_v_b'],
                out_w=lw['t2i_out_w'], out_b=lw['t2i_out_b'],
                Q_SRC=self.Q_IN_DRAM,
                KV_SRC=self.SRC_PE_DRAM,
                V_SRC=self.SRC_DRAM,
                q_len=NT_PAD,
            )

            # queries += T2I_OUT  then norm2
            eltwise_add_dram(self,
                self.TOKENS_DRAM, self.T2I_OUT,
                self.TOKENS_DRAM, NT_PAD * DEC_DIM)
            self.layer_norm_core_dram(
                M=NT_PAD, N=DEC_DIM,
                A_DRAM_ADDR=self.TOKENS_DRAM,
                OUTPUT_DRAM_ADDR=self.TOKENS_DRAM,
                GAMMA_DRAM_ADDR=lw['norm2_w'],
                BETA_DRAM_ADDR=lw['norm2_b'],
            )

            # ---- MLP on tokens (relu) ----
            self.matmat_mul_core(
                M=NT_PAD, K=DEC_DIM, N=DEC_MLP_DIM,
                A_DRAM_ADDR=self.TOKENS_DRAM,
                B_DRAM_ADDR=lw['mlp_lin1_w'],
                OUTPUT_DRAM_ADDR=self.MLP_MID,
                C_DRAM_ADDR=lw['mlp_lin1_b'],
                bias_mode="broadcast_N",
                clamp_enable=True,
            )
            self.matmat_mul_core(
                M=NT_PAD, K=DEC_MLP_DIM, N=DEC_DIM,
                A_DRAM_ADDR=self.MLP_MID,
                B_DRAM_ADDR=lw['mlp_lin2_w'],
                OUTPUT_DRAM_ADDR=self.MLP_OUT,
                C_DRAM_ADDR=lw['mlp_lin2_b'],
                bias_mode="broadcast_N",
            )

            # queries += MLP_OUT  then norm3
            eltwise_add_dram(self,
                self.TOKENS_DRAM, self.MLP_OUT,
                self.TOKENS_DRAM, NT_PAD * DEC_DIM)
            self.layer_norm_core_dram(
                M=NT_PAD, N=DEC_DIM,
                A_DRAM_ADDR=self.TOKENS_DRAM,
                OUTPUT_DRAM_ADDR=self.TOKENS_DRAM,
                GAMMA_DRAM_ADDR=lw['norm3_w'],
                BETA_DRAM_ADDR=lw['norm3_b'],
            )

            # ---- Cross-attn image→token ----
            # Q_i2t = src + key_pe (already in SRC_PE_DRAM from above)
            # K_i2t = tokens + query_pe
            eltwise_add_dram(self,
                self.TOKENS_DRAM, self.TOKENS_PE_DRAM,
                self.TOK_PE_DRAM, NT_PAD * DEC_DIM)

            self._cross_attn_i2t(
                q_w=lw['i2t_q_w'], q_b=lw['i2t_q_b'],
                k_w=lw['i2t_k_w'], k_b=lw['i2t_k_b'],
                v_w=lw['i2t_v_w'], v_b=lw['i2t_v_b'],
                out_w=lw['i2t_out_w'], out_b=lw['i2t_out_b'],
                Q_SRC=self.SRC_PE_DRAM,
                KV_SRC=self.TOK_PE_DRAM,
                V_SRC=self.TOKENS_DRAM,
            )

            # keys += I2T_OUT  then norm4
            eltwise_add_dram(self,
                self.SRC_DRAM, self.I2T_OUT,
                self.SRC_DRAM, GA * DEC_DIM)
            self.layer_norm_core_dram(
                M=GA, N=DEC_DIM,
                A_DRAM_ADDR=self.SRC_DRAM,
                OUTPUT_DRAM_ADDR=self.SRC_DRAM,
                GAMMA_DRAM_ADDR=lw['norm4_w'],
                BETA_DRAM_ADDR=lw['norm4_b'],
            )

        # === Final cross-attn (token→image) + norm_final_attn ===
        fa = self.dec_final_attn

        eltwise_add_dram(self,
            self.TOKENS_DRAM, self.TOKENS_PE_DRAM,
            self.Q_IN_DRAM, NT_PAD * DEC_DIM)
        eltwise_add_dram(self,
            self.SRC_DRAM, self.KEY_PE_DRAM,
            self.SRC_PE_DRAM, GA * DEC_DIM)

        self._cross_attn_t2i(
            q_w=fa['q_w'], q_b=fa['q_b'],
            k_w=fa['k_w'], k_b=fa['k_b'],
            v_w=fa['v_w'], v_b=fa['v_b'],
            out_w=fa['out_w'], out_b=fa['out_b'],
            Q_SRC=self.Q_IN_DRAM,
            KV_SRC=self.SRC_PE_DRAM,
            V_SRC=self.SRC_DRAM,
            q_len=NT_PAD,
        )

        # queries += T2I_OUT  then norm_final_attn
        eltwise_add_dram(self,
            self.TOKENS_DRAM, self.T2I_OUT,
            self.TOKENS_DRAM, NT_PAD * DEC_DIM)
        self.layer_norm_core_dram(
            M=NT_PAD, N=DEC_DIM,
            A_DRAM_ADDR=self.TOKENS_DRAM,
            OUTPUT_DRAM_ADDR=self.TOKENS_DRAM,
            GAMMA_DRAM_ADDR=self.dec_final_norm['w'],
            BETA_DRAM_ADDR=self.dec_final_norm['b'],
        )
        # TOKENS_DRAM[0]   = hs (iou token)
        # TOKENS_DRAM[1:5] = mask tokens

        # === Output upscaling on SRC_DRAM (4096, 256) → (256, 256, 64→32) ===
        # ConvTranspose2d(256→64, 2×2): input (64,64,256) HWC, output (128,128,64) HWC
        dram_zero_fill(self, self.UP0_OUT, 128 * 128 * 64)
        conv_transpose2d_2x2_dram(
            self,
            INPUT_DRAM_ADDR=self.SRC_DRAM,
            OUTPUT_DRAM_ADDR=self.UP0_OUT,
            TEMP_DRAM_ADDR=self.UP0_SCATTER,
            WEIGHT_SLICES=self.DEC_UP0_W,
            BIAS_DRAM_ADDR=self.DEC_UP0_B,
            H=IMG_H, W=IMG_W, C_in=DEC_DIM, C_out=64,
        )

        # LayerNorm2d(64) on (128*128, 64): same as layer_norm_core_dram with no beta usage
        # LayerNorm2d normalizes over channel dim (C), not last dim—same as layer_norm_core_dram
        self.layer_norm_core_dram(
            M=128 * 128, N=64,
            A_DRAM_ADDR=self.UP0_OUT,
            OUTPUT_DRAM_ADDR=self.UP_LN_OUT,
            GAMMA_DRAM_ADDR=self.DEC_UP_LN_W,
            BETA_DRAM_ADDR=self.DEC_UP_LN_B,
        )

        # GELU in-place on UP_LN_OUT (128*128, 64) via identity matmul with gelu_enable
        self.matmat_mul_core(
            M=128 * 128, K=64, N=64,
            A_DRAM_ADDR=self.UP_LN_OUT,
            B_DRAM_ADDR=self.GELU_ID_ADDR,
            OUTPUT_DRAM_ADDR=self.UP_LN_OUT,
            gelu_enable=True,
        )

        # ConvTranspose2d(64→32 padded to 64, 2×2): (128,128,64) → (256,256,64)
        dram_zero_fill(self, self.UP1_OUT, 256 * 256 * 64)
        conv_transpose2d_2x2_dram(
            self,
            INPUT_DRAM_ADDR=self.UP_LN_OUT,
            OUTPUT_DRAM_ADDR=self.UP1_OUT,
            TEMP_DRAM_ADDR=self.UP1_SCATTER,
            WEIGHT_SLICES=self.DEC_UP1_W,
            BIAS_DRAM_ADDR=self.DEC_UP1_B,
            H=128, W=128, C_in=64, C_out=64,   # C_out=64 (real 32, padded)
        )
        # UP1_OUT: (256*256, 64) with real mask logit dim in [0:32]

        # === Hypernetwork MLPs: 4 mask tokens → (4, 32) hyper vectors ===
        for m in range(4):
            hw = self.dec_hyper_weights[m]
            tok_addr = self.TOKENS_DRAM + (1 + m) * DEC_DIM * BPE  # tokens[1+m]

            # l0: (1, 256) → (1, 256) relu
            self.matmat_mul_core(
                M=1, K=DEC_DIM, N=DEC_DIM,
                A_DRAM_ADDR=tok_addr,
                B_DRAM_ADDR=hw['l0_w'],
                OUTPUT_DRAM_ADDR=self.HYPER_MID1,
                C_DRAM_ADDR=hw['l0_b'], bias_mode="broadcast_N",
                clamp_enable=True,
            )
            # l1: (1, 256) → (1, 256) relu
            self.matmat_mul_core(
                M=1, K=DEC_DIM, N=DEC_DIM,
                A_DRAM_ADDR=self.HYPER_MID1,
                B_DRAM_ADDR=hw['l1_w'],
                OUTPUT_DRAM_ADDR=self.HYPER_MID2,
                C_DRAM_ADDR=hw['l1_b'], bias_mode="broadcast_N",
                clamp_enable=True,
            )
            # l2: (1, 256) → (1, 64) padded (real 32)
            self.matmat_mul_core(
                M=1, K=DEC_DIM, N=64,
                A_DRAM_ADDR=self.HYPER_MID2,
                B_DRAM_ADDR=hw['l2_w'],
                OUTPUT_DRAM_ADDR=self.HYPER_OUT + m * 64 * BPE,
                C_DRAM_ADDR=hw['l2_b'], bias_mode="broadcast_N",
            )

        # === Mask logits: hyper[m] (1, 64) @ up_flat (256*256, 64)^T → (1, 65536) ===
        UP_FLAT_ELEMS = 256 * 256   # 65536
        for m in range(4):
            self.matmat_mul_core(
                M=1, K=64, N=UP_FLAT_ELEMS,
                A_DRAM_ADDR=self.HYPER_OUT + m * 64 * BPE,
                B_DRAM_ADDR=self.UP1_OUT,
                OUTPUT_DRAM_ADDR=self.MASK_OUT + m * UP_FLAT_ELEMS * BPE,
            )

        # === IoU prediction head: tokens[0] → (1, 4) ===
        iw = self.dec_iou_weights
        iou_tok_addr = self.TOKENS_DRAM   # tokens[0] = iou token

        self.matmat_mul_core(
            M=1, K=DEC_DIM, N=DEC_DIM,
            A_DRAM_ADDR=iou_tok_addr,
            B_DRAM_ADDR=iw['l0_w'],
            OUTPUT_DRAM_ADDR=self.IOU_MID1,
            C_DRAM_ADDR=iw['l0_b'], bias_mode="broadcast_N",
            clamp_enable=True,
        )
        self.matmat_mul_core(
            M=1, K=DEC_DIM, N=DEC_DIM,
            A_DRAM_ADDR=self.IOU_MID1,
            B_DRAM_ADDR=iw['l1_w'],
            OUTPUT_DRAM_ADDR=self.IOU_MID2,
            C_DRAM_ADDR=iw['l1_b'], bias_mode="broadcast_N",
            clamp_enable=True,
        )
        self.matmat_mul_core(
            M=1, K=DEC_DIM, N=64,
            A_DRAM_ADDR=self.IOU_MID2,
            B_DRAM_ADDR=iw['l2_w'],
            OUTPUT_DRAM_ADDR=self.IOU_OUT,
            C_DRAM_ADDR=iw['l2_b'], bias_mode="broadcast_N",
        )

        self.stop_capture()
        self.generate_instruction_halt()

        prog_addr = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prog_addr)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        self.clear_capture_buffer()

        print(f"  Program: {self.get_program_dram_usage() / 1024**2:.1f} MB")
        return prog_addr

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def preload_decoder_image(self, image_emb_t: torch.Tensor,
                              image_pe_t: torch.Tensor,
                              dense_t: torch.Tensor) -> None:
        """DMA image embedding + PE into DRAM once before an AMG loop."""
        src_t = (image_emb_t + dense_t).to(torch.bfloat16).contiguous()
        self.dma_to_accelerator_memory(self.SRC_DRAM,    src_t.flatten())
        self.dma_to_accelerator_memory(self.KEY_PE_DRAM, image_pe_t.to(torch.bfloat16).contiguous().flatten())

    def run_decoder_tokens(self, prog_addr: int, tokens_t: torch.Tensor,
                           timeout: float = 120.0):
        """Run decoder assuming image/PE already loaded via preload_decoder_image."""
        toks = tokens_t.to(torch.bfloat16).contiguous()
        self.dma_to_accelerator_memory(self.TOKENS_DRAM,    toks.flatten())
        self.dma_to_accelerator_memory(self.TOKENS_PE_DRAM, toks.flatten())
        self.start_execute_from_dram(prog_addr)
        self.wait_queue(timeout)
        masks = self.dma_from_accelerator_memory(self.MASK_OUT, (4, 256 * 256)).float().reshape(4, 256, 256)
        iou   = self.dma_from_accelerator_memory(self.IOU_OUT, (64,)).float()[:4]
        return masks, iou

    def run_decoder(self, prog_addr: int,
                    tokens_t: torch.Tensor,
                    image_emb_t: torch.Tensor,
                    image_pe_t: torch.Tensor,
                    dense_t: torch.Tensor,
                    timeout: float = 120.0):
        """Run the compiled mask decoder.

        Args:
            tokens_t    : (NT_PAD, 256) bf16 — assembled tokens, zero-padded to 64
            image_emb_t : (4096, 256)   bf16 — image embedding HWC
            image_pe_t  : (4096, 256)   bf16 — image positional encoding HWC
            dense_t     : (4096, 256)   bf16 — dense prompt

        Returns:
            masks_hw : (4, 256, 256) float32
            iou_hw   : (4,)          float32
        """
        # src = image_emb + dense (CPU-side, then DMA in)
        src_t = (image_emb_t + dense_t).to(torch.bfloat16).contiguous()

        self.dma_to_accelerator_memory(self.TOKENS_DRAM,    tokens_t.to(torch.bfloat16).contiguous().flatten())
        self.dma_to_accelerator_memory(self.TOKENS_PE_DRAM, tokens_t.to(torch.bfloat16).contiguous().flatten())
        self.dma_to_accelerator_memory(self.SRC_DRAM,       src_t.flatten())
        self.dma_to_accelerator_memory(self.KEY_PE_DRAM,    image_pe_t.to(torch.bfloat16).contiguous().flatten())

        self.start_execute_from_dram(prog_addr)
        self.wait_queue(timeout)

        # Read masks: (4, 65536) → (4, 256, 256)
        masks_hw = self.dma_from_accelerator_memory(
            self.MASK_OUT, (4, 256 * 256)
        ).float().reshape(4, 256, 256)

        # Read IoU: (1, 64) padded → real (4,)
        iou_hw = self.dma_from_accelerator_memory(self.IOU_OUT, (64,)).float()[:4]

        return masks_hw, iou_hw

    def _enc_init(self, sd):
        # ---- patch_embed conv 0: Conv2d_BN(3→32, ks=3, stride=2, pad=1) + GELU ----
        # BN-fold, then zero-pad C_in 3→64 and C_out 32→64 for HW alignment.
        w0 = sd["image_encoder.patch_embed.seq.0.c.weight"].float()   # (32, 3, 3, 3)
        w0_f, b0_f = _bn_fold(
            w0,
            sd["image_encoder.patch_embed.seq.0.bn.weight"].float(),
            sd["image_encoder.patch_embed.seq.0.bn.bias"].float(),
            sd["image_encoder.patch_embed.seq.0.bn.running_mean"].float(),
            sd["image_encoder.patch_embed.seq.0.bn.running_var"].float(),
        )
        # reshape to (C_out_pad, 9*C_in_pad) with per-neighbor interleaved padding.
        # im2col K layout: [neighbor0 × C_in_pad, neighbor1 × C_in_pad, ...]
        # so real channels for neighbor ki go at cols [ki*C_in_pad : ki*C_in_pad + C_in_real].
        w0_perm = w0_f.permute(0, 2, 3, 1)                            # (32, kH, kW, 3)
        w0_pad  = torch.zeros(ENC_C0P, 9 * ENC_CIN_PAD, dtype=torch.bfloat16)
        for kh in range(3):
            for kw in range(3):
                ki = kh * 3 + kw
                w0_pad[:32, ki * ENC_CIN_PAD : ki * ENC_CIN_PAD + 3] = w0_perm[:, kh, kw, :]
        b0_pad  = torch.zeros(ENC_C0P, dtype=torch.bfloat16)
        b0_pad[:32] = b0_f

        # ---- patch_embed conv 2: Conv2d_BN(32→64, ks=3, stride=2, pad=1) ----
        # Input C_in is padded 64 (from conv0 output), C_out=64 already aligned.
        w2 = sd["image_encoder.patch_embed.seq.2.c.weight"].float()   # (64, 32, 3, 3)
        w2_f, b2_f = _bn_fold(
            w2,
            sd["image_encoder.patch_embed.seq.2.bn.weight"].float(),
            sd["image_encoder.patch_embed.seq.2.bn.bias"].float(),
            sd["image_encoder.patch_embed.seq.2.bn.running_mean"].float(),
            sd["image_encoder.patch_embed.seq.2.bn.running_var"].float(),
        )
        # C_in=32 → padded to ENC_C0P=64 per neighbor block
        w2_perm = w2_f.permute(0, 2, 3, 1)                            # (64, kH, kW, 32)
        w2_pad  = torch.zeros(ENC_C0P, 9 * ENC_C0P, dtype=torch.bfloat16)
        for kh in range(3):
            for kw in range(3):
                ki = kh * 3 + kw
                w2_pad[:64, ki * ENC_C0P : ki * ENC_C0P + 32] = w2_perm[:, kh, kw, :]
        b2_pad  = b2_f.to(torch.bfloat16)

        # ---- DRAM allocations ----
        # params
        self.PE_W0_DRAM  = self.allocate_params_dram(ENC_C0P * 9 * ENC_CIN_PAD * _BPE, "pe_w0")
        self.PE_B0_DRAM  = self.allocate_params_dram(ENC_C0P * _BPE,                   "pe_b0")
        self.PE_W2_DRAM  = self.allocate_params_dram(ENC_C0P * 9 * ENC_C0P * _BPE,    "pe_w2")
        self.PE_B2_DRAM  = self.allocate_params_dram(ENC_C0P * _BPE,                   "pe_b2")
        self.PE_ZERO_DRAM = self.allocate_params_dram(ENC_CIN_PAD * _BPE,              "pe_zero")

        # activations: input image padded to (H*W, ENC_CIN_PAD), inter, output
        PE_H, PE_W = ENC_IN_H, ENC_IN_W
        PE_H1 = PE_H // 2; PE_W1 = PE_W // 2   # after conv0: 512×512
        PE_H2 = PE_H1 // 2; PE_W2 = PE_W1 // 2 # after conv2: 256×256

        self.PE_IN_DRAM   = self.allocate_tensor_dram(PE_H * PE_W * ENC_CIN_PAD * _BPE,  "pe_in")
        self.PE_MID_DRAM  = self.allocate_tensor_dram(PE_H1 * PE_W1 * ENC_C0P * _BPE,   "pe_mid")
        self.PE_OUT_DRAM  = self.allocate_tensor_dram(PE_H2 * PE_W2 * ENC_C0P * _BPE,   "pe_out")
        self.PE_IM2COL    = self.allocate_tensor_dram(
            max(PE_W1, PE_W2) * 9 * max(ENC_CIN_PAD, ENC_C0P) * _BPE, "pe_im2col")

        # upload weights
        self.dma_to_accelerator_memory(self.PE_W0_DRAM, w0_pad.reshape(-1))
        self.dma_to_accelerator_memory(self.PE_B0_DRAM, b0_pad)
        self.dma_to_accelerator_memory(self.PE_W2_DRAM, w2_pad.reshape(-1))
        self.dma_to_accelerator_memory(self.PE_B2_DRAM, b2_pad)
        self.dma_to_accelerator_memory(self.PE_ZERO_DRAM, torch.zeros(ENC_CIN_PAD, dtype=torch.bfloat16))

        # 64×64 identity matrix for in-place GELU pass
        identity = torch.eye(ENC_C0P, dtype=torch.bfloat16)
        self.IDENTITY_DRAM = self.allocate_params_dram(ENC_C0P * ENC_C0P * _BPE, "identity64")
        self.dma_to_accelerator_memory(self.IDENTITY_DRAM, identity.reshape(-1))

        # ---- Stage 0 MBConv shared zero pad (256 channels) ----
        self.S0_ZERO_DRAM = self.allocate_params_dram(ENC_S0_C_EXP * _BPE, "s0_zero")
        self.dma_to_accelerator_memory(self.S0_ZERO_DRAM,
                                       torch.zeros(ENC_S0_C_EXP, dtype=torch.bfloat16))

        # Stage 0 activations: (256*256, 256) expand buffer, (256*256, 64) output
        self.S0_EXP_DRAM = self.allocate_tensor_dram(
            ENC_S0_H * ENC_S0_W * ENC_S0_C_EXP * _BPE, "s0_exp")
        _dw_chunk = max(1, URAM_NEAR_FULL_ELEMENTS // (9 * 64 + 64))
        self.S0_IM2COL   = self.allocate_tensor_dram(
            (_dw_chunk * 9 * 64 + _dw_chunk * 64) * _BPE, "s0_im2col")

        def _load_s0_conv1(bi):
            pfx = f"image_encoder.layers.0.blocks.{bi}.conv1"
            w = sd[f"{pfx}.c.weight"].float()   # (256, 64, 1, 1)
            wf, bf = _bn_fold(w, sd[f"{pfx}.bn.weight"].float(),
                               sd[f"{pfx}.bn.bias"].float(),
                               sd[f"{pfx}.bn.running_mean"].float(),
                               sd[f"{pfx}.bn.running_var"].float())
            # 1×1 conv: weight is already (C_out, C_in) after reshape
            w_flat = wf.reshape(ENC_S0_C_EXP, ENC_C0P).to(torch.bfloat16)
            w_addr = self.allocate_params_dram(w_flat.numel() * _BPE, f"s0b{bi}c1w")
            b_addr = self.allocate_params_dram(ENC_S0_C_EXP * _BPE,  f"s0b{bi}c1b")
            self.dma_to_accelerator_memory(w_addr, w_flat.reshape(-1))
            self.dma_to_accelerator_memory(b_addr, bf.to(torch.bfloat16))
            return w_addr, b_addr

        self.S0B0_C1W, self.S0B0_C1B = _load_s0_conv1(0)

        # Stage 0 Block 0 conv2: depthwise 3×3 (256→256) + GELU
        def _load_s0_conv2(bi):
            pfx = f"image_encoder.layers.0.blocks.{bi}.conv2"
            w = sd[f"{pfx}.c.weight"].float()   # (256, 1, 3, 3)
            wf, bf = _bn_fold(w, sd[f"{pfx}.bn.weight"].float(),
                               sd[f"{pfx}.bn.bias"].float(),
                               sd[f"{pfx}.bn.running_mean"].float(),
                               sd[f"{pfx}.bn.running_var"].float())
            # build 4 block-diagonal matrices (64, 9*64), stored contiguously
            W_bd = torch.zeros(4, 64, 9 * 64, dtype=torch.bfloat16)
            for blk in range(4):
                for c in range(64):
                    c_abs = blk * 64 + c
                    for ki in range(9):
                        W_bd[blk, c, ki * 64 + c] = wf[c_abs, 0, ki // 3, ki % 3]
            w_addr = self.allocate_params_dram(W_bd.numel() * _BPE, f"s0b{bi}c2w")
            b_addr = self.allocate_params_dram(ENC_S0_C_EXP * _BPE,  f"s0b{bi}c2b")
            self.dma_to_accelerator_memory(w_addr, W_bd.reshape(-1))
            self.dma_to_accelerator_memory(b_addr, bf.to(torch.bfloat16))
            return w_addr, b_addr

        self.S0B0_C2W, self.S0B0_C2B = _load_s0_conv2(0)

        # Stage 0 Block 0 conv3: 1×1 project (256→64)
        def _load_s0_conv3(bi):
            pfx = f"image_encoder.layers.0.blocks.{bi}.conv3"
            w = sd[f"{pfx}.c.weight"].float()   # (64, 256, 1, 1)
            wf, bf = _bn_fold(w, sd[f"{pfx}.bn.weight"].float(),
                               sd[f"{pfx}.bn.bias"].float(),
                               sd[f"{pfx}.bn.running_mean"].float(),
                               sd[f"{pfx}.bn.running_var"].float())
            w_flat = wf.reshape(ENC_C0P, ENC_S0_C_EXP).to(torch.bfloat16)
            w_addr = self.allocate_params_dram(w_flat.numel() * _BPE, f"s0b{bi}c3w")
            b_addr = self.allocate_params_dram(ENC_C0P * _BPE,        f"s0b{bi}c3b")
            self.dma_to_accelerator_memory(w_addr, w_flat.reshape(-1))
            self.dma_to_accelerator_memory(b_addr, bf.to(torch.bfloat16))
            return w_addr, b_addr

        self.S0B0_C3W, self.S0B0_C3B = _load_s0_conv3(0)
        self.S0B1_C1W, self.S0B1_C1B = _load_s0_conv1(1)
        self.S0B1_C2W, self.S0B1_C2B = _load_s0_conv2(1)
        self.S0B1_C3W, self.S0B1_C3B = _load_s0_conv3(1)

        # ---- PatchMerging 0→1: (256×256,64) → (128×128,128) ----
        # conv1: 1×1 (64→128) + GELU
        pm01_pfx = "image_encoder.layers.0.downsample"
        pm01_w1 = sd[f"{pm01_pfx}.conv1.c.weight"].float()  # (128, 64, 1, 1)
        pm01_w1f, pm01_b1f = _bn_fold(
            pm01_w1, sd[f"{pm01_pfx}.conv1.bn.weight"].float(),
            sd[f"{pm01_pfx}.conv1.bn.bias"].float(),
            sd[f"{pm01_pfx}.conv1.bn.running_mean"].float(),
            sd[f"{pm01_pfx}.conv1.bn.running_var"].float())
        # (128, 64): already aligned, use as-is
        pm01_w1_flat = pm01_w1f.reshape(ENC_S1_C, ENC_C0P).to(torch.bfloat16)

        # conv2: dw 3×3 stride2 (128→128) + GELU; C=128 = 2 blocks of 64
        pm01_w2 = sd[f"{pm01_pfx}.conv2.c.weight"].float()  # (128, 1, 3, 3)
        pm01_w2f, pm01_b2f = _bn_fold(
            pm01_w2, sd[f"{pm01_pfx}.conv2.bn.weight"].float(),
            sd[f"{pm01_pfx}.conv2.bn.bias"].float(),
            sd[f"{pm01_pfx}.conv2.bn.running_mean"].float(),
            sd[f"{pm01_pfx}.conv2.bn.running_var"].float())
        PM01_NBLK = ENC_S1_C // 64   # 2
        W_pm01_bd = torch.zeros(PM01_NBLK, 64, 9 * 64, dtype=torch.bfloat16)
        for blk in range(PM01_NBLK):
            for c in range(64):
                c_abs = blk * 64 + c
                for ki in range(9):
                    W_pm01_bd[blk, c, ki * 64 + c] = pm01_w2f[c_abs, 0, ki // 3, ki % 3]

        # conv3: 1×1 (128→128) — no GELU
        pm01_w3 = sd[f"{pm01_pfx}.conv3.c.weight"].float()  # (128, 128, 1, 1)
        pm01_w3f, pm01_b3f = _bn_fold(
            pm01_w3, sd[f"{pm01_pfx}.conv3.bn.weight"].float(),
            sd[f"{pm01_pfx}.conv3.bn.bias"].float(),
            sd[f"{pm01_pfx}.conv3.bn.running_mean"].float(),
            sd[f"{pm01_pfx}.conv3.bn.running_var"].float())
        pm01_w3_flat = pm01_w3f.reshape(ENC_S1_C, ENC_S1_C).to(torch.bfloat16)

        # upload PM01 weights
        def _ap(t, tag):
            a = self.allocate_params_dram(t.numel() * _BPE, tag)
            self.dma_to_accelerator_memory(a, t.reshape(-1))
            return a

        self.PM01_W1_DRAM = _ap(pm01_w1_flat, "pm01_w1")
        self.PM01_B1_DRAM = _ap(pm01_b1f.to(torch.bfloat16), "pm01_b1")
        self.PM01_W2_DRAM = _ap(W_pm01_bd, "pm01_w2")
        self.PM01_B2_DRAM = _ap(pm01_b2f.to(torch.bfloat16), "pm01_b2")
        self.PM01_W3_DRAM = _ap(pm01_w3_flat, "pm01_w3")
        self.PM01_B3_DRAM = _ap(pm01_b3f.to(torch.bfloat16), "pm01_b3")
        self.PM01_ZERO_DRAM = _ap(torch.zeros(ENC_S1_C, dtype=torch.bfloat16), "pm01_zero")

        # depthwise output buffer: (256×256, 256)
        self.S0_DW_DRAM = self.allocate_tensor_dram(
            ENC_S0_H * ENC_S0_W * ENC_S0_C_EXP * _BPE, "s0_dw")
        # conv3 + residual output: (256×256, 64) — reuse PE_OUT_DRAM after patch_embed is done
        # allocate a separate S0_OUT_DRAM to hold block 0 output before block 1 input
        self.S0B0_OUT_DRAM = self.allocate_tensor_dram(
            ENC_S0_H * ENC_S0_W * ENC_C0P * _BPE, "s0b0_out")
        self.S0B1_OUT_DRAM = self.allocate_tensor_dram(
            ENC_S0_H * ENC_S0_W * ENC_C0P * _BPE, "s0b1_out")

        # PM01 activation buffers
        # conv1 out: (256×256, 128)
        self.PM01_EXP_DRAM = self.allocate_tensor_dram(
            ENC_S0_H * ENC_S0_W * ENC_S1_C * _BPE, "pm01_exp")
        # conv2 out: (128×128, 128)
        self.PM01_DW_DRAM = self.allocate_tensor_dram(
            ENC_S1_H * ENC_S1_W * ENC_S1_C * _BPE, "pm01_dw")
        # conv3 out: (128×128, 128)
        self.PM01_OUT_DRAM = self.allocate_tensor_dram(
            ENC_S1_H * ENC_S1_W * ENC_S1_C * _BPE, "pm01_out")
        # im2col scratch for PM01 depthwise (128-channel, stride-2)
        _pm01_dw_chunk = max(1, URAM_NEAR_FULL_ELEMENTS // (9 * 64 + 64))
        self.PM01_IM2COL_DRAM = self.allocate_tensor_dram(
            (_pm01_dw_chunk * 9 * 64 + _pm01_dw_chunk * 64) * _BPE, "pm01_im2col")

        # ---- Stage 1: 2× TinyViTBlock (128×128, C=128, heads=4, ws=7) ----
        # Precompute ab_idxs for relative position bias
        _ws = ENC_S1_WS; _N = _ws * _ws
        _pts = list(itertools.product(range(_ws), range(_ws)))
        _offsets = {}; _idxs = []
        for p1 in _pts:
            for p2 in _pts:
                off = (abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))
                if off not in _offsets: _offsets[off] = len(_offsets)
                _idxs.append(_offsets[off])
        _ab_idxs = torch.tensor(_idxs).view(_N, _N)

        def _load_s1_block(bi):
            pfx = f"image_encoder.layers.1.blocks.{bi}"
            # attn norm
            an_g = sd[f"{pfx}.attn.norm.weight"].to(torch.bfloat16)
            an_b = sd[f"{pfx}.attn.norm.bias"].to(torch.bfloat16)
            # qkv: split into W_q, W_k, W_v each (C, C)
            qkv_w = sd[f"{pfx}.attn.qkv.weight"].float()   # (384, 128)
            qkv_b = sd[f"{pfx}.attn.qkv.bias"].float()     # (384,)
            W3 = qkv_w.reshape(ENC_S1_HEADS, 3, ENC_S1_HEAD_DIM, ENC_S1_C)  # (4,3,32,128)
            b3 = qkv_b.reshape(ENC_S1_HEADS, 3, ENC_S1_HEAD_DIM)
            W_q = W3[:, 0].reshape(ENC_S1_C, ENC_S1_C).to(torch.bfloat16)
            W_k = W3[:, 1].reshape(ENC_S1_C, ENC_S1_C).to(torch.bfloat16)
            W_v = W3[:, 2].reshape(ENC_S1_C, ENC_S1_C).to(torch.bfloat16)
            b_q = b3[:, 0].reshape(ENC_S1_C).to(torch.bfloat16)
            b_k = b3[:, 1].reshape(ENC_S1_C).to(torch.bfloat16)
            b_v = b3[:, 2].reshape(ENC_S1_C).to(torch.bfloat16)
            # relative position bias: (heads, WIN_REAL, WIN_REAL) → pad to (heads, WIN_PAD, WIN_PAD)
            ab = sd[f"{pfx}.attn.attention_biases"].float()  # (4, 49)
            bias_real = ab[:, _ab_idxs]   # (4, 49, 49)
            bias_pad = torch.full((ENC_S1_HEADS, ENC_S1_WIN_PAD, ENC_S1_WIN_PAD),
                                  float('-inf'), dtype=torch.float32)
            bias_pad[:, :_N, :_N] = bias_real
            bias_pad[:, :_N, _N:] = float('-inf')
            bias_pad[:, _N:, :]   = 0.0
            bias_pad = bias_pad.to(torch.bfloat16)
            # proj
            proj_w = sd[f"{pfx}.attn.proj.weight"].to(torch.bfloat16)   # (128, 128)
            proj_b = sd[f"{pfx}.attn.proj.bias"].to(torch.bfloat16)     # (128,)
            # unpad weight: (4*64, 4*32) = (256, 128)
            unpad_w = _build_unpad_weight(ENC_S1_HEADS, ENC_S1_HEAD_DIM, ENC_S1_HEAD_PAD)
            # local_conv: depthwise 3×3, BN-fold, block-diagonal
            lc_w = sd[f"{pfx}.local_conv.c.weight"].float()  # (128, 1, 3, 3)
            lc_wf, lc_bf = _bn_fold(
                lc_w, sd[f"{pfx}.local_conv.bn.weight"].float(),
                sd[f"{pfx}.local_conv.bn.bias"].float(),
                sd[f"{pfx}.local_conv.bn.running_mean"].float(),
                sd[f"{pfx}.local_conv.bn.running_var"].float())
            _nblk = ENC_S1_C // 64
            W_lc = torch.zeros(_nblk, 64, 9*64, dtype=torch.bfloat16)
            for blk in range(_nblk):
                for c in range(64):
                    c_abs = blk * 64 + c
                    for ki in range(9):
                        W_lc[blk, c, ki*64+c] = lc_wf[c_abs, 0, ki//3, ki%3]
            # mlp norm + fc1 + fc2
            mn_g = sd[f"{pfx}.mlp.norm.weight"].to(torch.bfloat16)
            mn_b = sd[f"{pfx}.mlp.norm.bias"].to(torch.bfloat16)
            fc1_w = sd[f"{pfx}.mlp.fc1.weight"].to(torch.bfloat16)   # (512, 128)
            fc1_b = sd[f"{pfx}.mlp.fc1.bias"].to(torch.bfloat16)     # (512,)
            fc2_w = sd[f"{pfx}.mlp.fc2.weight"].to(torch.bfloat16)   # (128, 512)
            fc2_b = sd[f"{pfx}.mlp.fc2.bias"].to(torch.bfloat16)     # (128,)
            # upload
            d = {}
            d["an_g"]   = _ap(an_g, f"s1b{bi}_an_g")
            d["an_b"]   = _ap(an_b, f"s1b{bi}_an_b")
            d["W_q"]    = _ap(W_q,  f"s1b{bi}_Wq")
            d["b_q"]    = _ap(b_q,  f"s1b{bi}_bq")
            d["W_k"]    = _ap(W_k,  f"s1b{bi}_Wk")
            d["b_k"]    = _ap(b_k,  f"s1b{bi}_bk")
            d["W_v"]    = _ap(W_v,  f"s1b{bi}_Wv")
            d["b_v"]    = _ap(b_v,  f"s1b{bi}_bv")
            d["bias"]   = _ap(bias_pad, f"s1b{bi}_bias")
            d["proj_w"] = _ap(proj_w, f"s1b{bi}_projw")
            d["proj_b"] = _ap(proj_b, f"s1b{bi}_projb")
            d["unpad"]  = _ap(unpad_w.T.contiguous(), f"s1b{bi}_unpad")
            d["lc_w"]   = _ap(W_lc, f"s1b{bi}_lcw")
            d["lc_b"]   = _ap(lc_bf.to(torch.bfloat16), f"s1b{bi}_lcb")
            d["lc_zero"]= _ap(torch.zeros(ENC_S1_C, dtype=torch.bfloat16), f"s1b{bi}_lczero")
            d["mn_g"]   = _ap(mn_g, f"s1b{bi}_mn_g")
            d["mn_b"]   = _ap(mn_b, f"s1b{bi}_mn_b")
            d["fc1_w"]  = _ap(fc1_w, f"s1b{bi}_fc1w")
            d["fc1_b"]  = _ap(fc1_b, f"s1b{bi}_fc1b")
            d["fc2_w"]  = _ap(fc2_w, f"s1b{bi}_fc2w")
            d["fc2_b"]  = _ap(fc2_b, f"s1b{bi}_fc2b")
            return d

        self.S1B = [_load_s1_block(0), _load_s1_block(1)]

        # Stage 1 activation buffers (shared across both blocks)
        _NTOK = ENC_S1_NWIN * ENC_S1_WIN_PAD   # 361*64 = 23104
        _HW1  = ENC_S1_H * ENC_S1_W            # 16384
        self.S1_WIN_DRAM    = self.allocate_tensor_dram(_NTOK * ENC_S1_C * _BPE,           "s1_win")
        self.S1_Q_DRAM      = self.allocate_tensor_dram(_NTOK * ENC_S1_C * _BPE,           "s1_q")
        self.S1_K_DRAM      = self.allocate_tensor_dram(_NTOK * ENC_S1_C * _BPE,           "s1_k")
        self.S1_V_DRAM      = self.allocate_tensor_dram(_NTOK * ENC_S1_C * _BPE,           "s1_v")
        # after multihead_reshape: (heads, NTOK, HEAD_PAD)
        _HEAD_BUF = ENC_S1_HEADS * _NTOK * ENC_S1_HEAD_PAD
        self.S1_Q_HEAD_DRAM = self.allocate_tensor_dram(_HEAD_BUF * _BPE,                  "s1_q_head")
        self.S1_K_HEAD_DRAM = self.allocate_tensor_dram(_HEAD_BUF * _BPE,                  "s1_k_head")
        self.S1_V_HEAD_DRAM = self.allocate_tensor_dram(_HEAD_BUF * _BPE,                  "s1_v_head")
        self.S1_ATTN_DRAM   = self.allocate_tensor_dram(_HEAD_BUF * _BPE,                  "s1_attn")
        # scratch for flash attention: scratch_stride = HEAD_PAD*WIN_PAD + WIN_PAD*WIN_PAD = 8192
        _SCRATCH = ENC_S1_NWIN * (ENC_S1_HEAD_PAD*ENC_S1_WIN_PAD + ENC_S1_WIN_PAD**2)
        self.S1_FLASH_SCRATCH = self.allocate_tensor_dram(_SCRATCH * _BPE,                 "s1_fscr")
        # multihead reshape/merge temp: (NTOK, heads*HEAD_PAD)
        self.S1_MH_TEMP     = self.allocate_tensor_dram(_NTOK * ENC_S1_HEADS * ENC_S1_HEAD_PAD * _BPE, "s1_mh_tmp")
        self.S1_MERGED_DRAM = self.allocate_tensor_dram(_NTOK * ENC_S1_C * _BPE,           "s1_merged")
        self.S1_PROJ_DRAM   = self.allocate_tensor_dram(_NTOK * ENC_S1_C * _BPE,           "s1_proj")
        self.S1_REV_DRAM    = self.allocate_tensor_dram(_HW1  * ENC_S1_C * _BPE,           "s1_rev")
        # local_conv im2col scratch
        _lc_chunk = max(1, URAM_NEAR_FULL_ELEMENTS // (9*64 + 64))
        self.S1_LC_IM2COL   = self.allocate_tensor_dram((_lc_chunk*9*64 + _lc_chunk*64)*_BPE, "s1_lc_im2col")
        self.S1_LN_DRAM     = self.allocate_tensor_dram(_NTOK * ENC_S1_C * _BPE,           "s1_ln")
        self.S1_MLP_FC1_DRAM= self.allocate_tensor_dram(_HW1 * ENC_S1_MLP_HID * _BPE,     "s1_fc1")
        # block outputs: (HW, C)
        self.S1B0_OUT_DRAM  = self.allocate_tensor_dram(_HW1 * ENC_S1_C * _BPE,            "s1b0_out")
        self.S1B1_OUT_DRAM  = self.allocate_tensor_dram(_HW1 * ENC_S1_C * _BPE,            "s1b1_out")

        # ---- PatchMerging 1→2: (128×128,128) → (64×64,160) padded to 192 ----
        # C=160 is not a multiple of 64; pad all weights/buffers to C_PAD=192 throughout PM12+S2.
        pm12_pfx = "image_encoder.layers.1.downsample"
        _C_in = ENC_S1_C; _C_real = ENC_S2_C; _CP = ENC_S2_C_PAD  # 128, 160, 192

        pm12_w1 = sd[f"{pm12_pfx}.conv1.c.weight"].float()   # (160,128,1,1)
        pm12_w1f, pm12_b1f = _bn_fold(
            pm12_w1, sd[f"{pm12_pfx}.conv1.bn.weight"].float(),
            sd[f"{pm12_pfx}.conv1.bn.bias"].float(),
            sd[f"{pm12_pfx}.conv1.bn.running_mean"].float(),
            sd[f"{pm12_pfx}.conv1.bn.running_var"].float())
        # pad output dim 160→192: zero rows for channels 160-191
        W_pm12_w1 = torch.zeros(_CP, _C_in, dtype=torch.bfloat16)
        W_pm12_w1[:_C_real] = pm12_w1f.reshape(_C_real, _C_in).to(torch.bfloat16)
        B_pm12_b1 = torch.zeros(_CP, dtype=torch.bfloat16)
        B_pm12_b1[:_C_real] = pm12_b1f.to(torch.bfloat16)

        pm12_w2 = sd[f"{pm12_pfx}.conv2.c.weight"].float()   # (160,1,3,3)
        pm12_w2f, pm12_b2f = _bn_fold(
            pm12_w2, sd[f"{pm12_pfx}.conv2.bn.weight"].float(),
            sd[f"{pm12_pfx}.conv2.bn.bias"].float(),
            sd[f"{pm12_pfx}.conv2.bn.running_mean"].float(),
            sd[f"{pm12_pfx}.conv2.bn.running_var"].float())
        # 3 blocks of 64; last block channels 160-191 get zero weights
        PM12_NBLK = _CP // 64   # 3
        W_pm12_bd = torch.zeros(PM12_NBLK, 64, 9 * 64, dtype=torch.bfloat16)
        for blk in range(PM12_NBLK):
            for c in range(64):
                c_abs = blk * 64 + c
                if c_abs < _C_real:
                    for ki in range(9):
                        W_pm12_bd[blk, c, ki * 64 + c] = pm12_w2f[c_abs, 0, ki // 3, ki % 3]
        B_pm12_b2 = torch.zeros(_CP, dtype=torch.bfloat16)
        B_pm12_b2[:_C_real] = pm12_b2f.to(torch.bfloat16)

        pm12_w3 = sd[f"{pm12_pfx}.conv3.c.weight"].float()   # (160,160,1,1)
        pm12_w3f, pm12_b3f = _bn_fold(
            pm12_w3, sd[f"{pm12_pfx}.conv3.bn.weight"].float(),
            sd[f"{pm12_pfx}.conv3.bn.bias"].float(),
            sd[f"{pm12_pfx}.conv3.bn.running_mean"].float(),
            sd[f"{pm12_pfx}.conv3.bn.running_var"].float())
        # pad (160,160)→(192,192): real weights in top-left, zeros elsewhere
        W_pm12_w3 = torch.zeros(_CP, _CP, dtype=torch.bfloat16)
        W_pm12_w3[:_C_real, :_C_real] = pm12_w3f.reshape(_C_real, _C_real).to(torch.bfloat16)
        B_pm12_b3 = torch.zeros(_CP, dtype=torch.bfloat16)
        B_pm12_b3[:_C_real] = pm12_b3f.to(torch.bfloat16)

        self.PM12_W1_DRAM  = _ap(W_pm12_w1,                              "pm12_w1")
        self.PM12_B1_DRAM  = _ap(B_pm12_b1,                              "pm12_b1")
        self.PM12_W2_DRAM  = _ap(W_pm12_bd,                              "pm12_w2")
        self.PM12_B2_DRAM  = _ap(B_pm12_b2,                              "pm12_b2")
        self.PM12_W3_DRAM  = _ap(W_pm12_w3,                              "pm12_w3")
        self.PM12_B3_DRAM  = _ap(B_pm12_b3,                              "pm12_b3")
        self.PM12_ZERO_DRAM= _ap(torch.zeros(_CP, dtype=torch.bfloat16), "pm12_zero")

        # PM12 activation buffers — all use C_PAD=192
        _HW2 = ENC_S2_H * ENC_S2_W
        self.PM12_EXP_DRAM = self.allocate_tensor_dram(_HW1 * _CP * _BPE,  "pm12_exp")   # (128×128,192)
        self.PM12_DW_DRAM  = self.allocate_tensor_dram(_HW2 * _CP * _BPE,  "pm12_dw")    # (64×64,192)
        self.PM12_OUT_DRAM = self.allocate_tensor_dram(_HW2 * _CP * _BPE,  "pm12_out")   # (64×64,192)
        _pm12_dw_chunk = max(1, URAM_NEAR_FULL_ELEMENTS // (9 * 64 + 64))
        self.PM12_IM2COL_DRAM = self.allocate_tensor_dram(
            (_pm12_dw_chunk * 9 * 64 + _pm12_dw_chunk * 64) * _BPE, "pm12_im2col")

        # ---- Stage 2: 6× TinyViTBlock (64×64, C=160 padded to 192, heads=5, ws=14) ----
        import math as _math
        _S2_LN_SCALE = _math.sqrt(ENC_S2_C / ENC_S2_C_PAD)   # sqrt(160/192)
        _CP2 = ENC_S2_C_PAD    # 192
        _C2  = ENC_S2_C        # 160
        _HD2 = ENC_S2_HEAD_DIM # 32
        _HP2 = ENC_S2_HEAD_PAD # 64
        _H2  = ENC_S2_HEADS    # 5
        _N2  = ENC_S2_WS * ENC_S2_WS  # 196

        _pts2 = list(itertools.product(range(ENC_S2_WS), range(ENC_S2_WS)))
        _offsets2 = {}; _idxs2 = []
        for _p1 in _pts2:
            for _p2 in _pts2:
                _o = (abs(_p1[0]-_p2[0]), abs(_p1[1]-_p2[1]))
                if _o not in _offsets2: _offsets2[_o] = len(_offsets2)
                _idxs2.append(_offsets2[_o])
        _ab_idxs2 = torch.tensor(_idxs2).view(_N2, _N2)

        def _pad_s2_w(w_raw, out_dim=_CP2, in_dim=_CP2):
            w = torch.zeros(out_dim, in_dim, dtype=torch.bfloat16)
            w[:w_raw.shape[0], :w_raw.shape[1]] = w_raw.to(torch.bfloat16)
            return w

        def _load_s2_block(bi):
            pfx = f"image_encoder.layers.2.blocks.{bi}"
            # attn norm — gamma scaled by sqrt(160/192) to correct LN RMS with 32 zero-padded channels
            an_g = torch.zeros(_CP2, dtype=torch.bfloat16)
            an_g[:_C2] = (sd[f"{pfx}.attn.norm.weight"].float() * _S2_LN_SCALE).to(torch.bfloat16)
            an_b = torch.zeros(_CP2, dtype=torch.bfloat16)
            an_b[:_C2] = sd[f"{pfx}.attn.norm.bias"].to(torch.bfloat16)
            # qkv: (480, 160) → split W_q/W_k/W_v each padded (192, 192)
            qkv_w = sd[f"{pfx}.attn.qkv.weight"].float()   # (480, 160)
            qkv_b = sd[f"{pfx}.attn.qkv.bias"].float()     # (480,)
            W3 = qkv_w.reshape(_H2, 3, _HD2, _C2)
            b3 = qkv_b.reshape(_H2, 3, _HD2)
            W_q = _pad_s2_w(W3[:, 0].reshape(_C2, _C2))
            W_k = _pad_s2_w(W3[:, 1].reshape(_C2, _C2))
            W_v = _pad_s2_w(W3[:, 2].reshape(_C2, _C2))
            def _pad_b2(b_raw):
                b = torch.zeros(_CP2, dtype=torch.bfloat16)
                b[:_C2] = b_raw.reshape(_C2).to(torch.bfloat16)
                return b
            b_q = _pad_b2(b3[:, 0])
            b_k = _pad_b2(b3[:, 1])
            b_v = _pad_b2(b3[:, 2])
            # attention bias: (5, num_unique) → padded (5, WIN_PAD, WIN_PAD)
            ab = sd[f"{pfx}.attn.attention_biases"].float()
            bias_real = ab[:, _ab_idxs2]   # (5, 196, 196)
            bias_pad = torch.full((_H2, ENC_S2_WIN_PAD, ENC_S2_WIN_PAD), float('-inf'), dtype=torch.float32)
            bias_pad[:, :_N2, :_N2] = bias_real
            bias_pad[:, :_N2, _N2:] = float('-inf')
            bias_pad[:, _N2:, :]    = 0.0
            bias_pad = bias_pad.to(torch.bfloat16)
            # proj: (160,160) → (192,192)
            proj_w = _pad_s2_w(sd[f"{pfx}.attn.proj.weight"].float().reshape(_C2, _C2))
            proj_b = _pad_b2(sd[f"{pfx}.attn.proj.bias"].float())
            # unpad weight: (5*64=320, 160) → pad output to 192, store .T
            unpad_w_real = _build_unpad_weight(_H2, _HD2, _HP2)  # (320, 160)
            unpad_w = torch.zeros(_H2 * _HP2, _CP2, dtype=torch.bfloat16)
            unpad_w[:, :_C2] = unpad_w_real.to(torch.bfloat16)
            # local_conv: (160,1,3,3) → 3 blocks of 64 (last block all-zero)
            lc_w = sd[f"{pfx}.local_conv.c.weight"].float()
            lc_wf, lc_bf = _bn_fold(
                lc_w, sd[f"{pfx}.local_conv.bn.weight"].float(),
                sd[f"{pfx}.local_conv.bn.bias"].float(),
                sd[f"{pfx}.local_conv.bn.running_mean"].float(),
                sd[f"{pfx}.local_conv.bn.running_var"].float())
            _nblk2 = _CP2 // 64   # 3
            W_lc = torch.zeros(_nblk2, 64, 9*64, dtype=torch.bfloat16)
            for blk in range(_nblk2):
                for c in range(64):
                    c_abs = blk*64 + c
                    if c_abs < _C2:
                        for ki in range(9):
                            W_lc[blk, c, ki*64+c] = lc_wf[c_abs, 0, ki//3, ki%3]
            # mlp norm — gamma scaled
            mn_g = torch.zeros(_CP2, dtype=torch.bfloat16)
            mn_g[:_C2] = (sd[f"{pfx}.mlp.norm.weight"].float() * _S2_LN_SCALE).to(torch.bfloat16)
            mn_b = torch.zeros(_CP2, dtype=torch.bfloat16)
            mn_b[:_C2] = sd[f"{pfx}.mlp.norm.bias"].to(torch.bfloat16)
            # fc1: (640,160) → (640,192): zeros for input cols 160-191
            fc1_w = torch.zeros(ENC_S2_MLP_HID, _CP2, dtype=torch.bfloat16)
            fc1_w[:, :_C2] = sd[f"{pfx}.mlp.fc1.weight"].to(torch.bfloat16)
            fc1_b = sd[f"{pfx}.mlp.fc1.bias"].to(torch.bfloat16)   # (640,)
            # fc2: (160,640) → (192,640): zeros for output rows 160-191
            fc2_w = torch.zeros(_CP2, ENC_S2_MLP_HID, dtype=torch.bfloat16)
            fc2_w[:_C2, :] = sd[f"{pfx}.mlp.fc2.weight"].to(torch.bfloat16)
            fc2_b = torch.zeros(_CP2, dtype=torch.bfloat16)
            fc2_b[:_C2] = sd[f"{pfx}.mlp.fc2.bias"].to(torch.bfloat16)
            d = {}
            d["an_g"]    = _ap(an_g,  f"s2b{bi}_an_g")
            d["an_b"]    = _ap(an_b,  f"s2b{bi}_an_b")
            d["W_q"]     = _ap(W_q,   f"s2b{bi}_Wq")
            d["b_q"]     = _ap(b_q,   f"s2b{bi}_bq")
            d["W_k"]     = _ap(W_k,   f"s2b{bi}_Wk")
            d["b_k"]     = _ap(b_k,   f"s2b{bi}_bk")
            d["W_v"]     = _ap(W_v,   f"s2b{bi}_Wv")
            d["b_v"]     = _ap(b_v,   f"s2b{bi}_bv")
            d["bias"]    = _ap(bias_pad, f"s2b{bi}_bias")
            d["proj_w"]  = _ap(proj_w,  f"s2b{bi}_projw")
            d["proj_b"]  = _ap(proj_b,  f"s2b{bi}_projb")
            d["unpad"]   = _ap(unpad_w.T.contiguous(), f"s2b{bi}_unpad")
            d["lc_w"]    = _ap(W_lc,  f"s2b{bi}_lcw")
            d["lc_b"]    = _ap(lc_bf.to(torch.bfloat16), f"s2b{bi}_lcb")
            d["lc_zero"] = _ap(torch.zeros(_CP2, dtype=torch.bfloat16), f"s2b{bi}_lczero")
            d["mn_g"]    = _ap(mn_g,  f"s2b{bi}_mn_g")
            d["mn_b"]    = _ap(mn_b,  f"s2b{bi}_mn_b")
            d["fc1_w"]   = _ap(fc1_w, f"s2b{bi}_fc1w")
            d["fc1_b"]   = _ap(fc1_b, f"s2b{bi}_fc1b")
            d["fc2_w"]   = _ap(fc2_w, f"s2b{bi}_fc2w")
            d["fc2_b"]   = _ap(fc2_b, f"s2b{bi}_fc2b")
            return d

        self.S2B = [_load_s2_block(bi) for bi in range(6)]

        # Stage 2 activation buffers (shared across all 6 blocks)
        _NTOK2 = ENC_S2_NWIN * ENC_S2_WIN_PAD   # 25 * 256 = 6400
        _HW2   = ENC_S2_H * ENC_S2_W             # 4096
        _HEAD_BUF2 = ENC_S2_HEADS * _NTOK2 * ENC_S2_HEAD_PAD  # 5*6400*64
        self.S2_WIN_DRAM     = self.allocate_tensor_dram(_NTOK2 * _CP2 * _BPE,   "s2_win")
        self.S2_Q_DRAM       = self.allocate_tensor_dram(_NTOK2 * _CP2 * _BPE,   "s2_q")
        self.S2_K_DRAM       = self.allocate_tensor_dram(_NTOK2 * _CP2 * _BPE,   "s2_k")
        self.S2_V_DRAM       = self.allocate_tensor_dram(_NTOK2 * _CP2 * _BPE,   "s2_v")
        self.S2_Q_HEAD_DRAM  = self.allocate_tensor_dram(_HEAD_BUF2 * _BPE,      "s2_q_head")
        self.S2_K_HEAD_DRAM  = self.allocate_tensor_dram(_HEAD_BUF2 * _BPE,      "s2_k_head")
        self.S2_V_HEAD_DRAM  = self.allocate_tensor_dram(_HEAD_BUF2 * _BPE,      "s2_v_head")
        self.S2_ATTN_DRAM    = self.allocate_tensor_dram(_HEAD_BUF2 * _BPE,      "s2_attn")
        _SCRATCH2 = ENC_S2_NWIN * (ENC_S2_HEAD_PAD * ENC_S2_WIN_PAD + ENC_S2_WIN_PAD ** 2)
        self.S2_FLASH_SCRATCH = self.allocate_tensor_dram(_SCRATCH2 * _BPE,      "s2_fscr")
        self.S2_MH_TEMP      = self.allocate_tensor_dram(_NTOK2 * ENC_S2_HEADS * ENC_S2_HEAD_PAD * _BPE, "s2_mh_tmp")
        self.S2_MERGED_DRAM  = self.allocate_tensor_dram(_NTOK2 * _CP2 * _BPE,   "s2_merged")
        self.S2_PROJ_DRAM    = self.allocate_tensor_dram(_NTOK2 * _CP2 * _BPE,   "s2_proj")
        self.S2_REV_DRAM     = self.allocate_tensor_dram(_HW2   * _CP2 * _BPE,   "s2_rev")
        _lc2_chunk = max(1, URAM_NEAR_FULL_ELEMENTS // (9*64 + 64))
        self.S2_LC_IM2COL    = self.allocate_tensor_dram((_lc2_chunk*9*64 + _lc2_chunk*64)*_BPE, "s2_lc_im2col")
        self.S2_LN_DRAM      = self.allocate_tensor_dram(_HW2   * _CP2 * _BPE,   "s2_ln")
        self.S2_MLP_FC1_DRAM = self.allocate_tensor_dram(_HW2   * ENC_S2_MLP_HID * _BPE, "s2_fc1")
        for _bi in range(6):
            setattr(self, f"S2B{_bi}_OUT_DRAM",
                    self.allocate_tensor_dram(_HW2 * _CP2 * _BPE, f"s2b{_bi}_out"))

        # ---- PatchMerging 2→3: (64×64, 160 padded to 192) → (64×64, 320) ----
        # stride=1: no spatial downsampling; 320 = 5×64, no output padding needed
        pm23_pfx = "image_encoder.layers.2.downsample"
        _C23_in  = ENC_S2_C      # 160
        _C23_inp = ENC_S2_C_PAD  # 192 (actual buffer width)
        _C23_out = ENC_S3_C      # 320

        pm23_w1 = sd[f"{pm23_pfx}.conv1.c.weight"].float()   # (320, 160, 1, 1)
        pm23_w1f, pm23_b1f = _bn_fold(
            pm23_w1, sd[f"{pm23_pfx}.conv1.bn.weight"].float(),
            sd[f"{pm23_pfx}.conv1.bn.bias"].float(),
            sd[f"{pm23_pfx}.conv1.bn.running_mean"].float(),
            sd[f"{pm23_pfx}.conv1.bn.running_var"].float())
        # pad input dim 160→192: zeros for cols 160-191
        W_pm23_w1 = torch.zeros(_C23_out, _C23_inp, dtype=torch.bfloat16)
        W_pm23_w1[:, :_C23_in] = pm23_w1f.reshape(_C23_out, _C23_in).to(torch.bfloat16)
        B_pm23_b1 = pm23_b1f.to(torch.bfloat16)   # (320,)

        pm23_w2 = sd[f"{pm23_pfx}.conv2.c.weight"].float()   # (320, 1, 3, 3) dw
        pm23_w2f, pm23_b2f = _bn_fold(
            pm23_w2, sd[f"{pm23_pfx}.conv2.bn.weight"].float(),
            sd[f"{pm23_pfx}.conv2.bn.bias"].float(),
            sd[f"{pm23_pfx}.conv2.bn.running_mean"].float(),
            sd[f"{pm23_pfx}.conv2.bn.running_var"].float())
        _pm23_nblk = _C23_out // 64   # 5
        W_pm23_bd = torch.zeros(_pm23_nblk, 64, 9*64, dtype=torch.bfloat16)
        for blk in range(_pm23_nblk):
            for c in range(64):
                c_abs = blk*64 + c
                for ki in range(9):
                    W_pm23_bd[blk, c, ki*64+c] = pm23_w2f[c_abs, 0, ki//3, ki%3]
        B_pm23_b2 = pm23_b2f.to(torch.bfloat16)   # (320,)

        pm23_w3 = sd[f"{pm23_pfx}.conv3.c.weight"].float()   # (320, 320, 1, 1)
        pm23_w3f, pm23_b3f = _bn_fold(
            pm23_w3, sd[f"{pm23_pfx}.conv3.bn.weight"].float(),
            sd[f"{pm23_pfx}.conv3.bn.bias"].float(),
            sd[f"{pm23_pfx}.conv3.bn.running_mean"].float(),
            sd[f"{pm23_pfx}.conv3.bn.running_var"].float())
        W_pm23_w3 = pm23_w3f.reshape(_C23_out, _C23_out).to(torch.bfloat16)
        B_pm23_b3 = pm23_b3f.to(torch.bfloat16)

        self.PM23_W1_DRAM  = _ap(W_pm23_w1,                               "pm23_w1")
        self.PM23_B1_DRAM  = _ap(B_pm23_b1,                               "pm23_b1")
        self.PM23_W2_DRAM  = _ap(W_pm23_bd,                               "pm23_w2")
        self.PM23_B2_DRAM  = _ap(B_pm23_b2,                               "pm23_b2")
        self.PM23_W3_DRAM  = _ap(W_pm23_w3,                               "pm23_w3")
        self.PM23_B3_DRAM  = _ap(B_pm23_b3,                               "pm23_b3")
        self.PM23_ZERO_DRAM= _ap(torch.zeros(_C23_out, dtype=torch.bfloat16), "pm23_zero")

        # PM23 activation buffers (stride=1: all (HW2, 320))
        self.PM23_EXP_DRAM = self.allocate_tensor_dram(_HW2 * _C23_out * _BPE, "pm23_exp")
        self.PM23_DW_DRAM  = self.allocate_tensor_dram(_HW2 * _C23_out * _BPE, "pm23_dw")
        self.PM23_OUT_DRAM = self.allocate_tensor_dram(_HW2 * _C23_out * _BPE, "pm23_out")
        _pm23_dw_chunk = max(1, URAM_NEAR_FULL_ELEMENTS // (9*64 + 64))
        self.PM23_IM2COL_DRAM = self.allocate_tensor_dram(
            (_pm23_dw_chunk*9*64 + _pm23_dw_chunk*64)*_BPE, "pm23_im2col")

        # ---- Stage 3: 2× TinyViTBlock (64×64, C=320, heads=10, ws=7) ----
        # C=320=5×64, no padding needed anywhere
        _C3 = ENC_S3_C   # 320

        _ws3 = ENC_S3_WS; _N3 = _ws3 * _ws3
        _pts3 = list(itertools.product(range(_ws3), range(_ws3)))
        _offsets3 = {}; _idxs3 = []
        for _p1 in _pts3:
            for _p2 in _pts3:
                _o = (abs(_p1[0]-_p2[0]), abs(_p1[1]-_p2[1]))
                if _o not in _offsets3: _offsets3[_o] = len(_offsets3)
                _idxs3.append(_offsets3[_o])
        _ab_idxs3 = torch.tensor(_idxs3).view(_N3, _N3)

        def _load_s3_block(bi):
            pfx = f"image_encoder.layers.3.blocks.{bi}"
            an_g = sd[f"{pfx}.attn.norm.weight"].to(torch.bfloat16)
            an_b = sd[f"{pfx}.attn.norm.bias"].to(torch.bfloat16)
            qkv_w = sd[f"{pfx}.attn.qkv.weight"].float()   # (960, 320)
            qkv_b = sd[f"{pfx}.attn.qkv.bias"].float()     # (960,)
            W3 = qkv_w.reshape(ENC_S3_HEADS, 3, ENC_S3_HEAD_DIM, _C3)
            b3 = qkv_b.reshape(ENC_S3_HEADS, 3, ENC_S3_HEAD_DIM)
            W_q = W3[:, 0].reshape(_C3, _C3).to(torch.bfloat16)
            W_k = W3[:, 1].reshape(_C3, _C3).to(torch.bfloat16)
            W_v = W3[:, 2].reshape(_C3, _C3).to(torch.bfloat16)
            b_q = b3[:, 0].reshape(_C3).to(torch.bfloat16)
            b_k = b3[:, 1].reshape(_C3).to(torch.bfloat16)
            b_v = b3[:, 2].reshape(_C3).to(torch.bfloat16)
            ab = sd[f"{pfx}.attn.attention_biases"].float()
            bias_real = ab[:, _ab_idxs3]   # (10, 49, 49)
            bias_pad = torch.full((ENC_S3_HEADS, ENC_S3_WIN_PAD, ENC_S3_WIN_PAD),
                                  float('-inf'), dtype=torch.float32)
            bias_pad[:, :_N3, :_N3] = bias_real
            bias_pad[:, :_N3, _N3:] = float('-inf')
            bias_pad[:, _N3:, :]    = 0.0
            bias_pad = bias_pad.to(torch.bfloat16)
            proj_w = sd[f"{pfx}.attn.proj.weight"].to(torch.bfloat16)   # (320, 320)
            proj_b = sd[f"{pfx}.attn.proj.bias"].to(torch.bfloat16)
            unpad_w = _build_unpad_weight(ENC_S3_HEADS, ENC_S3_HEAD_DIM, ENC_S3_HEAD_PAD)
            lc_w = sd[f"{pfx}.local_conv.c.weight"].float()  # (320, 1, 3, 3)
            lc_wf, lc_bf = _bn_fold(
                lc_w, sd[f"{pfx}.local_conv.bn.weight"].float(),
                sd[f"{pfx}.local_conv.bn.bias"].float(),
                sd[f"{pfx}.local_conv.bn.running_mean"].float(),
                sd[f"{pfx}.local_conv.bn.running_var"].float())
            _nblk3 = _C3 // 64   # 5
            W_lc = torch.zeros(_nblk3, 64, 9*64, dtype=torch.bfloat16)
            for blk in range(_nblk3):
                for c in range(64):
                    c_abs = blk*64 + c
                    for ki in range(9):
                        W_lc[blk, c, ki*64+c] = lc_wf[c_abs, 0, ki//3, ki%3]
            mn_g = sd[f"{pfx}.mlp.norm.weight"].to(torch.bfloat16)
            mn_b = sd[f"{pfx}.mlp.norm.bias"].to(torch.bfloat16)
            fc1_w = sd[f"{pfx}.mlp.fc1.weight"].to(torch.bfloat16)   # (1280, 320)
            fc1_b = sd[f"{pfx}.mlp.fc1.bias"].to(torch.bfloat16)
            fc2_w = sd[f"{pfx}.mlp.fc2.weight"].to(torch.bfloat16)   # (320, 1280)
            fc2_b = sd[f"{pfx}.mlp.fc2.bias"].to(torch.bfloat16)
            d = {}
            d["an_g"]    = _ap(an_g,  f"s3b{bi}_an_g")
            d["an_b"]    = _ap(an_b,  f"s3b{bi}_an_b")
            d["W_q"]     = _ap(W_q,   f"s3b{bi}_Wq")
            d["b_q"]     = _ap(b_q,   f"s3b{bi}_bq")
            d["W_k"]     = _ap(W_k,   f"s3b{bi}_Wk")
            d["b_k"]     = _ap(b_k,   f"s3b{bi}_bk")
            d["W_v"]     = _ap(W_v,   f"s3b{bi}_Wv")
            d["b_v"]     = _ap(b_v,   f"s3b{bi}_bv")
            d["bias"]    = _ap(bias_pad, f"s3b{bi}_bias")
            d["proj_w"]  = _ap(proj_w, f"s3b{bi}_projw")
            d["proj_b"]  = _ap(proj_b, f"s3b{bi}_projb")
            d["unpad"]   = _ap(unpad_w.T.contiguous(), f"s3b{bi}_unpad")
            d["lc_w"]    = _ap(W_lc,  f"s3b{bi}_lcw")
            d["lc_b"]    = _ap(lc_bf.to(torch.bfloat16), f"s3b{bi}_lcb")
            d["lc_zero"] = _ap(torch.zeros(_C3, dtype=torch.bfloat16), f"s3b{bi}_lczero")
            d["mn_g"]    = _ap(mn_g,  f"s3b{bi}_mn_g")
            d["mn_b"]    = _ap(mn_b,  f"s3b{bi}_mn_b")
            d["fc1_w"]   = _ap(fc1_w, f"s3b{bi}_fc1w")
            d["fc1_b"]   = _ap(fc1_b, f"s3b{bi}_fc1b")
            d["fc2_w"]   = _ap(fc2_w, f"s3b{bi}_fc2w")
            d["fc2_b"]   = _ap(fc2_b, f"s3b{bi}_fc2b")
            return d

        self.S3B = [_load_s3_block(0), _load_s3_block(1)]

        # Stage 3 activation buffers
        _NTOK3 = ENC_S3_NWIN * ENC_S3_WIN_PAD    # 100 * 64 = 6400
        _HW3   = ENC_S3_H * ENC_S3_W             # 4096
        _HEAD_BUF3 = ENC_S3_HEADS * _NTOK3 * ENC_S3_HEAD_PAD  # 10*6400*64
        self.S3_WIN_DRAM     = self.allocate_tensor_dram(_NTOK3 * _C3 * _BPE,    "s3_win")
        self.S3_Q_DRAM       = self.allocate_tensor_dram(_NTOK3 * _C3 * _BPE,    "s3_q")
        self.S3_K_DRAM       = self.allocate_tensor_dram(_NTOK3 * _C3 * _BPE,    "s3_k")
        self.S3_V_DRAM       = self.allocate_tensor_dram(_NTOK3 * _C3 * _BPE,    "s3_v")
        self.S3_Q_HEAD_DRAM  = self.allocate_tensor_dram(_HEAD_BUF3 * _BPE,      "s3_q_head")
        self.S3_K_HEAD_DRAM  = self.allocate_tensor_dram(_HEAD_BUF3 * _BPE,      "s3_k_head")
        self.S3_V_HEAD_DRAM  = self.allocate_tensor_dram(_HEAD_BUF3 * _BPE,      "s3_v_head")
        self.S3_ATTN_DRAM    = self.allocate_tensor_dram(_HEAD_BUF3 * _BPE,      "s3_attn")
        _SCRATCH3 = ENC_S3_NWIN * (ENC_S3_HEAD_PAD * ENC_S3_WIN_PAD + ENC_S3_WIN_PAD ** 2)
        self.S3_FLASH_SCRATCH = self.allocate_tensor_dram(_SCRATCH3 * _BPE,      "s3_fscr")
        self.S3_MH_TEMP      = self.allocate_tensor_dram(_NTOK3 * ENC_S3_HEADS * ENC_S3_HEAD_PAD * _BPE, "s3_mh_tmp")
        self.S3_MERGED_DRAM  = self.allocate_tensor_dram(_NTOK3 * _C3 * _BPE,    "s3_merged")
        self.S3_PROJ_DRAM    = self.allocate_tensor_dram(_NTOK3 * _C3 * _BPE,    "s3_proj")
        self.S3_REV_DRAM     = self.allocate_tensor_dram(_HW3   * _C3 * _BPE,    "s3_rev")
        _lc3_chunk = max(1, URAM_NEAR_FULL_ELEMENTS // (9*64 + 64))
        self.S3_LC_IM2COL    = self.allocate_tensor_dram((_lc3_chunk*9*64 + _lc3_chunk*64)*_BPE, "s3_lc_im2col")
        self.S3_LN_DRAM      = self.allocate_tensor_dram(_HW3   * _C3 * _BPE,    "s3_ln")
        self.S3_MLP_FC1_DRAM = self.allocate_tensor_dram(_HW3   * ENC_S3_MLP_HID * _BPE, "s3_fc1")
        self.S3B0_OUT_DRAM   = self.allocate_tensor_dram(_HW3   * _C3 * _BPE,    "s3b0_out")
        self.S3B1_OUT_DRAM   = self.allocate_tensor_dram(_HW3   * _C3 * _BPE,    "s3b1_out")

        # ---- Neck: Conv1x1(320→256) → LN → Conv3x3(256→256) → LN ----
        _NECK_C_IN  = ENC_S3_C   # 320
        _NECK_C_OUT = 256        # multiple of 64 ✓
        _NECK_HW    = ENC_S3_H * ENC_S3_W  # 4096

        neck_c1_w  = sd["image_encoder.neck.0.weight"].reshape(_NECK_C_OUT, _NECK_C_IN).to(torch.bfloat16)
        neck_ln1_g = sd["image_encoder.neck.1.weight"].to(torch.bfloat16)
        neck_ln1_b = sd["image_encoder.neck.1.bias"].to(torch.bfloat16)
        # neck.2: (256, 256, 3, 3) → (256, 9*256)
        neck_c3_w  = sd["image_encoder.neck.2.weight"].permute(0, 2, 3, 1).reshape(_NECK_C_OUT, 9 * _NECK_C_OUT).to(torch.bfloat16)
        neck_ln2_g = sd["image_encoder.neck.3.weight"].to(torch.bfloat16)
        neck_ln2_b = sd["image_encoder.neck.3.bias"].to(torch.bfloat16)

        self.NECK_C1_W   = _ap(neck_c1_w,                                          "neck_c1_w")
        self.NECK_LN1_G  = _ap(neck_ln1_g,                                         "neck_ln1_g")
        self.NECK_LN1_B  = _ap(neck_ln1_b,                                         "neck_ln1_b")
        self.NECK_C3_W   = _ap(neck_c3_w,                                          "neck_c3_w")
        self.NECK_LN2_G  = _ap(neck_ln2_g,                                         "neck_ln2_g")
        self.NECK_LN2_B  = _ap(neck_ln2_b,                                         "neck_ln2_b")
        self.NECK_ZERO   = _ap(torch.zeros(_NECK_C_OUT, dtype=torch.bfloat16),     "neck_zero")

        # Activation buffers
        self.NECK_BUF1_DRAM  = self.allocate_tensor_dram(_NECK_HW * _NECK_C_OUT * _BPE, "neck_buf1")
        self.NECK_OUT_DRAM   = self.allocate_tensor_dram(_NECK_HW * _NECK_C_OUT * _BPE, "neck_out")
        _neck_chunk = max(1, URAM_NEAR_FULL_ELEMENTS // (9 * _NECK_C_OUT))
        self.NECK_IM2COL_DRAM = self.allocate_tensor_dram(
            _neck_chunk * 9 * _NECK_C_OUT * _BPE, "neck_im2col")

    def _alloc_param(self, t: torch.Tensor) -> int:
        addr = self.allocate_params_dram(t.numel() * _BPE)
        self.dma_to_accelerator_memory(addr, t.reshape(-1))
        return addr



    def _run_s1_block(self, w: dict, INPUT_DRAM: int, OUTPUT_DRAM: int) -> None:
        """Run one TinyViTBlock: attn → residual → local_conv → mlp+residual."""
        _HW   = ENC_S1_H * ENC_S1_W
        _NTOK = ENC_S1_NWIN * ENC_S1_WIN_PAD
        _WPAD = ENC_S1_WIN_PAD
        _HP   = ENC_S1_HEAD_PAD

        # attn norm on spatial tokens BEFORE window partition (no zero-padding, avoids hardware LN NaN)
        _LN_BATCH = 512
        for _off in range(0, _HW, _LN_BATCH):
            _take = min(_LN_BATCH, _HW - _off)
            self.layer_norm_core_dram(
                M=_take, N=ENC_S1_C,
                A_DRAM_ADDR=INPUT_DRAM + _off * ENC_S1_C * _BPE,
                OUTPUT_DRAM_ADDR=self.S1_LN_DRAM + _off * ENC_S1_C * _BPE,
                GAMMA_DRAM_ADDR=w["an_g"], BETA_DRAM_ADDR=w["an_b"])

        # window partition LN output
        window_partition_dram(self, self.S1_LN_DRAM, self.S1_WIN_DRAM,
                              ENC_S1_H, ENC_S1_W, ENC_S1_C, ENC_S1_WS)

        # Q, K, V projections (from windowed LN tokens)
        self.matmat_mul_core(M=_NTOK, K=ENC_S1_C, N=ENC_S1_C,
                             A_DRAM_ADDR=self.S1_WIN_DRAM, B_DRAM_ADDR=w["W_q"],
                             OUTPUT_DRAM_ADDR=self.S1_Q_DRAM,
                             C_DRAM_ADDR=w["b_q"], bias_mode="broadcast_N")
        self.matmat_mul_core(M=_NTOK, K=ENC_S1_C, N=ENC_S1_C,
                             A_DRAM_ADDR=self.S1_WIN_DRAM, B_DRAM_ADDR=w["W_k"],
                             OUTPUT_DRAM_ADDR=self.S1_K_DRAM,
                             C_DRAM_ADDR=w["b_k"], bias_mode="broadcast_N")
        self.matmat_mul_core(M=_NTOK, K=ENC_S1_C, N=ENC_S1_C,
                             A_DRAM_ADDR=self.S1_WIN_DRAM, B_DRAM_ADDR=w["W_v"],
                             OUTPUT_DRAM_ADDR=self.S1_V_DRAM,
                             C_DRAM_ADDR=w["b_v"], bias_mode="broadcast_N")

        # multihead reshape: (NTOK, C) → (heads, NTOK, HEAD_PAD)
        multihead_reshape_dram(self, self.S1_Q_DRAM, self.S1_Q_HEAD_DRAM, self.S1_MH_TEMP,
                               _NTOK, ENC_S1_HEADS, ENC_S1_HEAD_DIM, _HP)
        multihead_reshape_dram(self, self.S1_K_DRAM, self.S1_K_HEAD_DRAM, self.S1_MH_TEMP,
                               _NTOK, ENC_S1_HEADS, ENC_S1_HEAD_DIM, _HP)
        multihead_reshape_dram(self, self.S1_V_DRAM, self.S1_V_HEAD_DRAM, self.S1_MH_TEMP,
                               _NTOK, ENC_S1_HEADS, ENC_S1_HEAD_DIM, _HP)
        # flash_attn uses 1/sqrt(head_dim_pad=64); true scale is 1/sqrt(head_dim=32)
        broadcast_mul_dram(self, self.S1_Q_HEAD_DRAM, SA_SCALE_CORRECTION,
                           ENC_S1_HEADS * _NTOK * _HP)

        # flash attention: process one head at a time, bias shared across windows
        _head_stride = ENC_S1_NWIN * _WPAD * _HP * _BPE
        for h in range(ENC_S1_HEADS):
            flash_attention_batched(
                self, num_batches=ENC_S1_NWIN, head_dim=_HP, seq_len=_WPAD,
                Q_DRAM_ADDR=self.S1_Q_HEAD_DRAM + h * _head_stride,
                K_DRAM_ADDR=self.S1_K_HEAD_DRAM + h * _head_stride,
                V_DRAM_ADDR=self.S1_V_HEAD_DRAM + h * _head_stride,
                OUTPUT_DRAM_ADDR=self.S1_ATTN_DRAM + h * _head_stride,
                SCRATCH_DRAM_ADDR=self.S1_FLASH_SCRATCH,
                BIAS_DRAM_ADDR=w["bias"] + h * _WPAD * _WPAD * _BPE,
                bias_shared=True)

        # multihead merge: (heads, NTOK, HEAD_PAD) → (NTOK, C)
        multihead_merge_dram(self, self.S1_ATTN_DRAM, self.S1_MERGED_DRAM, self.S1_MH_TEMP,
                             _NTOK, ENC_S1_HEADS, ENC_S1_HEAD_DIM, _HP,
                             UNPAD_WEIGHT_ADDR=w["unpad"])

        # proj
        self.matmat_mul_core(M=_NTOK, K=ENC_S1_C, N=ENC_S1_C,
                             A_DRAM_ADDR=self.S1_MERGED_DRAM, B_DRAM_ADDR=w["proj_w"],
                             OUTPUT_DRAM_ADDR=self.S1_PROJ_DRAM,
                             C_DRAM_ADDR=w["proj_b"], bias_mode="broadcast_N")

        # window reverse → S1_REV_DRAM (zero-fill first, then scatter)
        dram_zero_fill(self, self.S1_REV_DRAM, _HW * ENC_S1_C)
        window_reverse_dram(self, self.S1_PROJ_DRAM, self.S1_REV_DRAM,
                            ENC_S1_H, ENC_S1_W, ENC_S1_C, ENC_S1_WS,
                            ENC_S1_NH, ENC_S1_NW, ENC_S1_WIN_PAD)

        # attn residual: x_in + window_reverse → OUTPUT_DRAM (temporarily, reuse as post_attn)
        eltwise_add_dram(self, INPUT_DRAM, self.S1_REV_DRAM, OUTPUT_DRAM, _HW * ENC_S1_C)

        # local_conv (replaces x — no residual): depthwise 3×3 stride1
        conv2d_3x3_dw_dram(self, OUTPUT_DRAM, self.S1_REV_DRAM, self.S1_LC_IM2COL,
                           w["lc_w"], w["lc_b"], w["lc_zero"],
                           ENC_S1_H, ENC_S1_W, ENC_S1_C)

        # mlp norm in batches
        for _off in range(0, _HW, _LN_BATCH):
            _take = min(_LN_BATCH, _HW - _off)
            self.layer_norm_core_dram(
                M=_take, N=ENC_S1_C,
                A_DRAM_ADDR=self.S1_REV_DRAM + _off * ENC_S1_C * _BPE,
                OUTPUT_DRAM_ADDR=OUTPUT_DRAM  + _off * ENC_S1_C * _BPE,
                GAMMA_DRAM_ADDR=w["mn_g"], BETA_DRAM_ADDR=w["mn_b"])
        self.matmat_mul_core(M=_HW, K=ENC_S1_C, N=ENC_S1_MLP_HID,
                             A_DRAM_ADDR=OUTPUT_DRAM, B_DRAM_ADDR=w["fc1_w"],
                             OUTPUT_DRAM_ADDR=self.S1_MLP_FC1_DRAM,
                             C_DRAM_ADDR=w["fc1_b"], bias_mode="broadcast_N",
                             gelu_enable=True)
        self.matmat_mul_core(M=_HW, K=ENC_S1_MLP_HID, N=ENC_S1_C,
                             A_DRAM_ADDR=self.S1_MLP_FC1_DRAM, B_DRAM_ADDR=w["fc2_w"],
                             OUTPUT_DRAM_ADDR=OUTPUT_DRAM,
                             C_DRAM_ADDR=w["fc2_b"], bias_mode="broadcast_N")
        # mlp residual: local_conv_out + mlp_out → OUTPUT_DRAM
        eltwise_add_dram(self, self.S1_REV_DRAM, OUTPUT_DRAM, OUTPUT_DRAM, _HW * ENC_S1_C)

    def _run_s2_block(self, w: dict, INPUT_DRAM: int, OUTPUT_DRAM: int) -> None:
        """Run one Stage-2 TinyViTBlock: C=160 padded to 192, 5 heads, ws=14."""
        _HW   = ENC_S2_H * ENC_S2_W          # 4096
        _NTOK = ENC_S2_NWIN * ENC_S2_WIN_PAD  # 6400
        _WPAD = ENC_S2_WIN_PAD                 # 256
        _HP   = ENC_S2_HEAD_PAD                # 64
        _CP   = ENC_S2_C_PAD                   # 192
        _LN_BATCH = 512

        # attn norm on spatial tokens BEFORE window partition
        for _off in range(0, _HW, _LN_BATCH):
            _take = min(_LN_BATCH, _HW - _off)
            self.layer_norm_core_dram(
                M=_take, N=_CP,
                A_DRAM_ADDR=INPUT_DRAM + _off * _CP * _BPE,
                OUTPUT_DRAM_ADDR=self.S2_LN_DRAM + _off * _CP * _BPE,
                GAMMA_DRAM_ADDR=w["an_g"], BETA_DRAM_ADDR=w["an_b"])

        window_partition_dram(self, self.S2_LN_DRAM, self.S2_WIN_DRAM,
                              ENC_S2_H, ENC_S2_W, _CP, ENC_S2_WS)

        self.matmat_mul_core(M=_NTOK, K=_CP, N=_CP,
                             A_DRAM_ADDR=self.S2_WIN_DRAM, B_DRAM_ADDR=w["W_q"],
                             OUTPUT_DRAM_ADDR=self.S2_Q_DRAM,
                             C_DRAM_ADDR=w["b_q"], bias_mode="broadcast_N")
        self.matmat_mul_core(M=_NTOK, K=_CP, N=_CP,
                             A_DRAM_ADDR=self.S2_WIN_DRAM, B_DRAM_ADDR=w["W_k"],
                             OUTPUT_DRAM_ADDR=self.S2_K_DRAM,
                             C_DRAM_ADDR=w["b_k"], bias_mode="broadcast_N")
        self.matmat_mul_core(M=_NTOK, K=_CP, N=_CP,
                             A_DRAM_ADDR=self.S2_WIN_DRAM, B_DRAM_ADDR=w["W_v"],
                             OUTPUT_DRAM_ADDR=self.S2_V_DRAM,
                             C_DRAM_ADDR=w["b_v"], bias_mode="broadcast_N")

        multihead_reshape_dram(self, self.S2_Q_DRAM, self.S2_Q_HEAD_DRAM, self.S2_MH_TEMP,
                               _NTOK, ENC_S2_HEADS, ENC_S2_HEAD_DIM, _HP, input_row_stride=_CP)
        multihead_reshape_dram(self, self.S2_K_DRAM, self.S2_K_HEAD_DRAM, self.S2_MH_TEMP,
                               _NTOK, ENC_S2_HEADS, ENC_S2_HEAD_DIM, _HP, input_row_stride=_CP)
        multihead_reshape_dram(self, self.S2_V_DRAM, self.S2_V_HEAD_DRAM, self.S2_MH_TEMP,
                               _NTOK, ENC_S2_HEADS, ENC_S2_HEAD_DIM, _HP, input_row_stride=_CP)
        broadcast_mul_dram(self, self.S2_Q_HEAD_DRAM, SA_SCALE_CORRECTION,
                           ENC_S2_HEADS * _NTOK * _HP)

        _head_stride = ENC_S2_NWIN * _WPAD * _HP * _BPE
        for h in range(ENC_S2_HEADS):
            flash_attention_batched(
                self, num_batches=ENC_S2_NWIN, head_dim=_HP, seq_len=_WPAD,
                Q_DRAM_ADDR=self.S2_Q_HEAD_DRAM + h * _head_stride,
                K_DRAM_ADDR=self.S2_K_HEAD_DRAM + h * _head_stride,
                V_DRAM_ADDR=self.S2_V_HEAD_DRAM + h * _head_stride,
                OUTPUT_DRAM_ADDR=self.S2_ATTN_DRAM + h * _head_stride,
                SCRATCH_DRAM_ADDR=self.S2_FLASH_SCRATCH,
                BIAS_DRAM_ADDR=w["bias"] + h * _WPAD * _WPAD * _BPE,
                bias_shared=True)

        # merge: permute (heads,NTOK,HP) → (NTOK,heads*HP), then matmul → (NTOK, C_PAD=192)
        self.bf16_permute_core(dim_0=ENC_S2_HEADS, dim_1=_NTOK, dim_2=_HP,
                               INPUT_DRAM_ADDR=self.S2_ATTN_DRAM,
                               OUTPUT_DRAM_ADDR=self.S2_MH_TEMP)
        self.matmat_mul_core(M=_NTOK, K=ENC_S2_HEADS * _HP, N=_CP,
                             A_DRAM_ADDR=self.S2_MH_TEMP,
                             B_DRAM_ADDR=w["unpad"],
                             OUTPUT_DRAM_ADDR=self.S2_MERGED_DRAM)

        self.matmat_mul_core(M=_NTOK, K=_CP, N=_CP,
                             A_DRAM_ADDR=self.S2_MERGED_DRAM, B_DRAM_ADDR=w["proj_w"],
                             OUTPUT_DRAM_ADDR=self.S2_PROJ_DRAM,
                             C_DRAM_ADDR=w["proj_b"], bias_mode="broadcast_N")

        dram_zero_fill(self, self.S2_REV_DRAM, _HW * _CP)
        window_reverse_dram(self, self.S2_PROJ_DRAM, self.S2_REV_DRAM,
                            ENC_S2_H, ENC_S2_W, _CP, ENC_S2_WS,
                            ENC_S2_NH, ENC_S2_NW, ENC_S2_WIN_PAD)

        eltwise_add_dram(self, INPUT_DRAM, self.S2_REV_DRAM, OUTPUT_DRAM, _HW * _CP)

        conv2d_3x3_dw_dram(self, OUTPUT_DRAM, self.S2_REV_DRAM, self.S2_LC_IM2COL,
                           w["lc_w"], w["lc_b"], w["lc_zero"],
                           ENC_S2_H, ENC_S2_W, _CP)

        for _off in range(0, _HW, _LN_BATCH):
            _take = min(_LN_BATCH, _HW - _off)
            self.layer_norm_core_dram(
                M=_take, N=_CP,
                A_DRAM_ADDR=self.S2_REV_DRAM + _off * _CP * _BPE,
                OUTPUT_DRAM_ADDR=OUTPUT_DRAM  + _off * _CP * _BPE,
                GAMMA_DRAM_ADDR=w["mn_g"], BETA_DRAM_ADDR=w["mn_b"])

        self.matmat_mul_core(M=_HW, K=_CP, N=ENC_S2_MLP_HID,
                             A_DRAM_ADDR=OUTPUT_DRAM, B_DRAM_ADDR=w["fc1_w"],
                             OUTPUT_DRAM_ADDR=self.S2_MLP_FC1_DRAM,
                             C_DRAM_ADDR=w["fc1_b"], bias_mode="broadcast_N",
                             gelu_enable=True)
        self.matmat_mul_core(M=_HW, K=ENC_S2_MLP_HID, N=_CP,
                             A_DRAM_ADDR=self.S2_MLP_FC1_DRAM, B_DRAM_ADDR=w["fc2_w"],
                             OUTPUT_DRAM_ADDR=OUTPUT_DRAM,
                             C_DRAM_ADDR=w["fc2_b"], bias_mode="broadcast_N")
        eltwise_add_dram(self, self.S2_REV_DRAM, OUTPUT_DRAM, OUTPUT_DRAM, _HW * _CP)

    def _run_s3_block(self, w: dict, INPUT_DRAM: int, OUTPUT_DRAM: int) -> None:
        """Run one Stage-3 TinyViTBlock: C=320, 10 heads, ws=7. No padding needed."""
        _HW   = ENC_S3_H * ENC_S3_W           # 4096
        _NTOK = ENC_S3_NWIN * ENC_S3_WIN_PAD   # 6400
        _WPAD = ENC_S3_WIN_PAD                  # 64
        _HP   = ENC_S3_HEAD_PAD                 # 64
        _C    = ENC_S3_C                        # 320
        _LN_BATCH = 512

        for _off in range(0, _HW, _LN_BATCH):
            _take = min(_LN_BATCH, _HW - _off)
            self.layer_norm_core_dram(
                M=_take, N=_C,
                A_DRAM_ADDR=INPUT_DRAM + _off * _C * _BPE,
                OUTPUT_DRAM_ADDR=self.S3_LN_DRAM + _off * _C * _BPE,
                GAMMA_DRAM_ADDR=w["an_g"], BETA_DRAM_ADDR=w["an_b"])

        window_partition_dram(self, self.S3_LN_DRAM, self.S3_WIN_DRAM,
                              ENC_S3_H, ENC_S3_W, _C, ENC_S3_WS)

        self.matmat_mul_core(M=_NTOK, K=_C, N=_C,
                             A_DRAM_ADDR=self.S3_WIN_DRAM, B_DRAM_ADDR=w["W_q"],
                             OUTPUT_DRAM_ADDR=self.S3_Q_DRAM,
                             C_DRAM_ADDR=w["b_q"], bias_mode="broadcast_N")
        self.matmat_mul_core(M=_NTOK, K=_C, N=_C,
                             A_DRAM_ADDR=self.S3_WIN_DRAM, B_DRAM_ADDR=w["W_k"],
                             OUTPUT_DRAM_ADDR=self.S3_K_DRAM,
                             C_DRAM_ADDR=w["b_k"], bias_mode="broadcast_N")
        self.matmat_mul_core(M=_NTOK, K=_C, N=_C,
                             A_DRAM_ADDR=self.S3_WIN_DRAM, B_DRAM_ADDR=w["W_v"],
                             OUTPUT_DRAM_ADDR=self.S3_V_DRAM,
                             C_DRAM_ADDR=w["b_v"], bias_mode="broadcast_N")

        multihead_reshape_dram(self, self.S3_Q_DRAM, self.S3_Q_HEAD_DRAM, self.S3_MH_TEMP,
                               _NTOK, ENC_S3_HEADS, ENC_S3_HEAD_DIM, _HP)
        multihead_reshape_dram(self, self.S3_K_DRAM, self.S3_K_HEAD_DRAM, self.S3_MH_TEMP,
                               _NTOK, ENC_S3_HEADS, ENC_S3_HEAD_DIM, _HP)
        multihead_reshape_dram(self, self.S3_V_DRAM, self.S3_V_HEAD_DRAM, self.S3_MH_TEMP,
                               _NTOK, ENC_S3_HEADS, ENC_S3_HEAD_DIM, _HP)
        broadcast_mul_dram(self, self.S3_Q_HEAD_DRAM, SA_SCALE_CORRECTION,
                           ENC_S3_HEADS * _NTOK * _HP)

        _head_stride = ENC_S3_NWIN * _WPAD * _HP * _BPE
        for h in range(ENC_S3_HEADS):
            flash_attention_batched(
                self, num_batches=ENC_S3_NWIN, head_dim=_HP, seq_len=_WPAD,
                Q_DRAM_ADDR=self.S3_Q_HEAD_DRAM + h * _head_stride,
                K_DRAM_ADDR=self.S3_K_HEAD_DRAM + h * _head_stride,
                V_DRAM_ADDR=self.S3_V_HEAD_DRAM + h * _head_stride,
                OUTPUT_DRAM_ADDR=self.S3_ATTN_DRAM + h * _head_stride,
                SCRATCH_DRAM_ADDR=self.S3_FLASH_SCRATCH,
                BIAS_DRAM_ADDR=w["bias"] + h * _WPAD * _WPAD * _BPE,
                bias_shared=True)

        multihead_merge_dram(self, self.S3_ATTN_DRAM, self.S3_MERGED_DRAM, self.S3_MH_TEMP,
                             _NTOK, ENC_S3_HEADS, ENC_S3_HEAD_DIM, _HP,
                             UNPAD_WEIGHT_ADDR=w["unpad"])

        self.matmat_mul_core(M=_NTOK, K=_C, N=_C,
                             A_DRAM_ADDR=self.S3_MERGED_DRAM, B_DRAM_ADDR=w["proj_w"],
                             OUTPUT_DRAM_ADDR=self.S3_PROJ_DRAM,
                             C_DRAM_ADDR=w["proj_b"], bias_mode="broadcast_N")

        dram_zero_fill(self, self.S3_REV_DRAM, _HW * _C)
        window_reverse_dram(self, self.S3_PROJ_DRAM, self.S3_REV_DRAM,
                            ENC_S3_H, ENC_S3_W, _C, ENC_S3_WS,
                            ENC_S3_NH, ENC_S3_NW, ENC_S3_WIN_PAD)

        eltwise_add_dram(self, INPUT_DRAM, self.S3_REV_DRAM, OUTPUT_DRAM, _HW * _C)

        conv2d_3x3_dw_dram(self, OUTPUT_DRAM, self.S3_REV_DRAM, self.S3_LC_IM2COL,
                           w["lc_w"], w["lc_b"], w["lc_zero"],
                           ENC_S3_H, ENC_S3_W, _C)

        for _off in range(0, _HW, _LN_BATCH):
            _take = min(_LN_BATCH, _HW - _off)
            self.layer_norm_core_dram(
                M=_take, N=_C,
                A_DRAM_ADDR=self.S3_REV_DRAM + _off * _C * _BPE,
                OUTPUT_DRAM_ADDR=OUTPUT_DRAM  + _off * _C * _BPE,
                GAMMA_DRAM_ADDR=w["mn_g"], BETA_DRAM_ADDR=w["mn_b"])

        self.matmat_mul_core(M=_HW, K=_C, N=ENC_S3_MLP_HID,
                             A_DRAM_ADDR=OUTPUT_DRAM, B_DRAM_ADDR=w["fc1_w"],
                             OUTPUT_DRAM_ADDR=self.S3_MLP_FC1_DRAM,
                             C_DRAM_ADDR=w["fc1_b"], bias_mode="broadcast_N",
                             gelu_enable=True)
        self.matmat_mul_core(M=_HW, K=ENC_S3_MLP_HID, N=_C,
                             A_DRAM_ADDR=self.S3_MLP_FC1_DRAM, B_DRAM_ADDR=w["fc2_w"],
                             OUTPUT_DRAM_ADDR=OUTPUT_DRAM,
                             C_DRAM_ADDR=w["fc2_b"], bias_mode="broadcast_N")
        eltwise_add_dram(self, self.S3_REV_DRAM, OUTPUT_DRAM, OUTPUT_DRAM, _HW * _C)

    def compile_encoder(self) -> int:
        """Compile all encoder instructions into a single DRAM program. Returns prog addr."""
        self.start_capture()
        self._encoder_ops(lambda: None)  # no flush — one continuous instruction stream
        self.stop_capture()
        self.generate_instruction_halt()
        prog = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prog)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        self.clear_capture_buffer()
        return prog

    def execute_encoder(self, prog: int, image_t: torch.Tensor):
        """Execute pre-compiled encoder program. Result lands in self.NECK_OUT_DRAM."""
        img_hwc = image_t[0].permute(1, 2, 0).contiguous()
        img_pad = torch.zeros(ENC_IN_H * ENC_IN_W, ENC_CIN_PAD, dtype=torch.bfloat16)
        img_pad[:, :3] = img_hwc.reshape(-1, 3)
        self.dma_to_accelerator_memory(self.PE_IN_DRAM, img_pad.reshape(-1))
        self.start_execute_from_dram(prog)
        self.wait_queue(120.0)

    def encode(self, image_t: torch.Tensor):
        """Compile + execute in one call — single instruction stream."""
        prog = self.compile_encoder()
        self.execute_encoder(prog, image_t)

    # ------------------------------------------------------------------
    # Debug: step through encoder stage by stage, assert no NaNs
    # ------------------------------------------------------------------

    def _compile_run(self, ops_fn, image_t: torch.Tensor | None = None, timeout: float = 120.0) -> None:
        """Compile ops_fn() as a single program, execute, wait."""
        self.start_capture()
        ops_fn()
        self.stop_capture()
        self.generate_instruction_halt()
        prog = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prog)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        self.clear_capture_buffer()
        if image_t is not None:
            img_hwc = image_t[0].permute(1, 2, 0).contiguous()
            img_pad = torch.zeros(ENC_IN_H * ENC_IN_W, ENC_CIN_PAD, dtype=torch.bfloat16)
            img_pad[:, :3] = img_hwc.reshape(-1, 3)
            self.dma_to_accelerator_memory(self.PE_IN_DRAM, img_pad.reshape(-1))
        self.start_execute_from_dram(prog)
        self.wait_queue(timeout)

    def _read_tensor(self, addr: int, n_elements: int) -> torch.Tensor:
        """DMA-read n_elements bf16 values from addr."""
        buf = bytearray(n_elements * 2)
        self.dma_read(DMA_DEVICE_C2H, addr, buf, len(buf))
        return torch.frombuffer(buf, dtype=torch.bfloat16).clone()

    def _assert_no_nan(self, label: str, addr: int, n_elements: int) -> None:
        t = self._read_tensor(addr, n_elements)
        n_nan = torch.isnan(t).sum().item()
        n_inf = torch.isinf(t).sum().item()
        assert n_nan == 0 and n_inf == 0, (
            f"NaN/Inf after {label}: {n_nan} NaNs, {n_inf} Infs "
            f"(max={t[~torch.isnan(t)].abs().max().item():.4f})"
        )
        _original_print(f"  [OK] {label}: max_abs={t.abs().max().item():.4f}")

    def debug_encode(self, image_t: torch.Tensor) -> None:
        """Run encoder stage-by-stage, asserting no NaNs at each checkpoint."""
        _original_print("debug_encode: stepping through encoder ...")

        # conv0 + conv2 (patch embed)
        self._compile_run(lambda: (
            conv2d_3x3_stride2_dram(
                self, self.PE_IN_DRAM, self.PE_MID_DRAM, self.PE_IM2COL,
                self.PE_W0_DRAM, self.PE_B0_DRAM, self.PE_ZERO_DRAM,
                ENC_IN_H, ENC_IN_W, ENC_CIN_PAD, ENC_C0P, gelu_enable=True),
            conv2d_3x3_stride2_dram(
                self, self.PE_MID_DRAM, self.PE_OUT_DRAM, self.PE_IM2COL,
                self.PE_W2_DRAM, self.PE_B2_DRAM, self.PE_ZERO_DRAM,
                ENC_IN_H // 2, ENC_IN_W // 2, ENC_C0P, ENC_C0P),
        ), image_t=image_t)
        self._assert_no_nan("patch_embed (PE_OUT)", self.PE_OUT_DRAM, ENC_S0_H * ENC_S0_W * ENC_C0P)

        # Stage 0 Block 0
        def _s0b0():
            conv2d_1x1_dram(self, self.PE_OUT_DRAM, self.S0_EXP_DRAM,
                            self.S0B0_C1W, self.S0B0_C1B,
                            ENC_S0_H, ENC_S0_W, ENC_C0P, ENC_S0_C_EXP, gelu_enable=True)
            conv2d_3x3_dw_dram(self, self.S0_EXP_DRAM, self.S0_DW_DRAM, self.S0_IM2COL,
                               self.S0B0_C2W, self.S0B0_C2B, self.S0_ZERO_DRAM,
                               ENC_S0_H, ENC_S0_W, ENC_S0_C_EXP, gelu_enable=True)
            conv2d_1x1_dram(self, self.S0_DW_DRAM, self.S0B0_OUT_DRAM,
                            self.S0B0_C3W, self.S0B0_C3B,
                            ENC_S0_H, ENC_S0_W, ENC_S0_C_EXP, ENC_C0P)
            eltwise_add_dram(self, self.PE_OUT_DRAM, self.S0B0_OUT_DRAM,
                             self.S0B0_OUT_DRAM, ENC_S0_H * ENC_S0_W * ENC_C0P)
            self.matmat_mul_core(
                M=ENC_S0_H * ENC_S0_W, K=ENC_C0P, N=ENC_C0P,
                A_DRAM_ADDR=self.S0B0_OUT_DRAM, B_DRAM_ADDR=self.IDENTITY_DRAM,
                OUTPUT_DRAM_ADDR=self.S0B0_OUT_DRAM, gelu_enable=True)
        self._compile_run(_s0b0)
        self._assert_no_nan("S0B0_OUT", self.S0B0_OUT_DRAM, ENC_S0_H * ENC_S0_W * ENC_C0P)

        # Stage 0 Block 1
        def _s0b1():
            conv2d_1x1_dram(self, self.S0B0_OUT_DRAM, self.S0_EXP_DRAM,
                            self.S0B1_C1W, self.S0B1_C1B,
                            ENC_S0_H, ENC_S0_W, ENC_C0P, ENC_S0_C_EXP, gelu_enable=True)
            conv2d_3x3_dw_dram(self, self.S0_EXP_DRAM, self.S0_DW_DRAM, self.S0_IM2COL,
                               self.S0B1_C2W, self.S0B1_C2B, self.S0_ZERO_DRAM,
                               ENC_S0_H, ENC_S0_W, ENC_S0_C_EXP, gelu_enable=True)
            conv2d_1x1_dram(self, self.S0_DW_DRAM, self.S0B1_OUT_DRAM,
                            self.S0B1_C3W, self.S0B1_C3B,
                            ENC_S0_H, ENC_S0_W, ENC_S0_C_EXP, ENC_C0P)
            eltwise_add_dram(self, self.S0B0_OUT_DRAM, self.S0B1_OUT_DRAM,
                             self.S0B1_OUT_DRAM, ENC_S0_H * ENC_S0_W * ENC_C0P)
            self.matmat_mul_core(
                M=ENC_S0_H * ENC_S0_W, K=ENC_C0P, N=ENC_C0P,
                A_DRAM_ADDR=self.S0B1_OUT_DRAM, B_DRAM_ADDR=self.IDENTITY_DRAM,
                OUTPUT_DRAM_ADDR=self.S0B1_OUT_DRAM, gelu_enable=True)
        self._compile_run(_s0b1)
        self._assert_no_nan("S0B1_OUT", self.S0B1_OUT_DRAM, ENC_S0_H * ENC_S0_W * ENC_C0P)

        # PatchMerging 0→1
        def _pm01():
            conv2d_1x1_dram(self, self.S0B1_OUT_DRAM, self.PM01_EXP_DRAM,
                            self.PM01_W1_DRAM, self.PM01_B1_DRAM,
                            ENC_S0_H, ENC_S0_W, ENC_C0P, ENC_S1_C, gelu_enable=True)
            conv2d_3x3_dw_stride2_dram(
                self, self.PM01_EXP_DRAM, self.PM01_DW_DRAM, self.PM01_IM2COL_DRAM,
                self.PM01_W2_DRAM, self.PM01_B2_DRAM, self.PM01_ZERO_DRAM,
                ENC_S0_H, ENC_S0_W, ENC_S1_C, gelu_enable=True)
            conv2d_1x1_dram(self, self.PM01_DW_DRAM, self.PM01_OUT_DRAM,
                            self.PM01_W3_DRAM, self.PM01_B3_DRAM,
                            ENC_S1_H, ENC_S1_W, ENC_S1_C, ENC_S1_C)
        self._compile_run(_pm01)
        self._assert_no_nan("PM01_OUT", self.PM01_OUT_DRAM, ENC_S1_H * ENC_S1_W * ENC_S1_C)

        # Stage 1 blocks
        for bi, (inp, out, label) in enumerate([
            (self.PM01_OUT_DRAM, self.S1B0_OUT_DRAM, "S1B0_OUT"),
            (self.S1B0_OUT_DRAM, self.S1B1_OUT_DRAM, "S1B1_OUT"),
        ]):
            self._compile_run(lambda i=bi, _in=inp, _out=out: self._run_s1_block(self.S1B[i], _in, _out))
            self._assert_no_nan(label, out, ENC_S1_H * ENC_S1_W * ENC_S1_C)

        # PatchMerging 1→2
        _CP = ENC_S2_C_PAD
        def _pm12():
            conv2d_1x1_dram(self, self.S1B1_OUT_DRAM, self.PM12_EXP_DRAM,
                            self.PM12_W1_DRAM, self.PM12_B1_DRAM,
                            ENC_S1_H, ENC_S1_W, ENC_S1_C, _CP, gelu_enable=True)
            conv2d_3x3_dw_stride2_dram(self, self.PM12_EXP_DRAM, self.PM12_DW_DRAM,
                                       self.PM12_IM2COL_DRAM,
                                       self.PM12_W2_DRAM, self.PM12_B2_DRAM, self.PM12_ZERO_DRAM,
                                       ENC_S1_H, ENC_S1_W, _CP, gelu_enable=True)
            conv2d_1x1_dram(self, self.PM12_DW_DRAM, self.PM12_OUT_DRAM,
                            self.PM12_W3_DRAM, self.PM12_B3_DRAM,
                            ENC_S2_H, ENC_S2_W, _CP, _CP, gelu_enable=False)
        self._compile_run(_pm12)
        self._assert_no_nan("PM12_OUT", self.PM12_OUT_DRAM, ENC_S2_H * ENC_S2_W * _CP)

        # Stage 2 blocks
        _s2_io = [
            (self.PM12_OUT_DRAM, self.S2B0_OUT_DRAM, "S2B0_OUT"),
            (self.S2B0_OUT_DRAM, self.S2B1_OUT_DRAM, "S2B1_OUT"),
            (self.S2B1_OUT_DRAM, self.S2B2_OUT_DRAM, "S2B2_OUT"),
            (self.S2B2_OUT_DRAM, self.S2B3_OUT_DRAM, "S2B3_OUT"),
            (self.S2B3_OUT_DRAM, self.S2B4_OUT_DRAM, "S2B4_OUT"),
            (self.S2B4_OUT_DRAM, self.S2B5_OUT_DRAM, "S2B5_OUT"),
        ]
        for bi, (_in, _out, label) in enumerate(_s2_io):
            self._compile_run(lambda i=bi, _i=_in, _o=_out: self._run_s2_block(self.S2B[i], _i, _o))
            self._assert_no_nan(label, _out, ENC_S2_H * ENC_S2_W * _CP)

        # PatchMerging 2→3
        _C23 = ENC_S3_C
        def _pm23():
            conv2d_1x1_dram(self, self.S2B5_OUT_DRAM, self.PM23_EXP_DRAM,
                            self.PM23_W1_DRAM, self.PM23_B1_DRAM,
                            ENC_S2_H, ENC_S2_W, _CP, _C23, gelu_enable=True)
            conv2d_3x3_dw_dram(self, self.PM23_EXP_DRAM, self.PM23_DW_DRAM, self.PM23_IM2COL_DRAM,
                               self.PM23_W2_DRAM, self.PM23_B2_DRAM, self.PM23_ZERO_DRAM,
                               ENC_S2_H, ENC_S2_W, _C23, gelu_enable=True)
            conv2d_1x1_dram(self, self.PM23_DW_DRAM, self.PM23_OUT_DRAM,
                            self.PM23_W3_DRAM, self.PM23_B3_DRAM,
                            ENC_S2_H, ENC_S2_W, _C23, _C23, gelu_enable=False)
        self._compile_run(_pm23)
        self._assert_no_nan("PM23_OUT", self.PM23_OUT_DRAM, ENC_S2_H * ENC_S2_W * _C23)

        # Stage 3 blocks
        for bi, (_in, _out, label) in enumerate([
            (self.PM23_OUT_DRAM, self.S3B0_OUT_DRAM, "S3B0_OUT"),
            (self.S3B0_OUT_DRAM, self.S3B1_OUT_DRAM, "S3B1_OUT"),
        ]):
            self._compile_run(lambda i=bi, _i=_in, _o=_out: self._run_s3_block(self.S3B[i], _i, _o))
            self._assert_no_nan(label, _out, ENC_S3_H * ENC_S3_W * ENC_S3_C)

        # Neck
        _NECK_C_IN  = ENC_S3_C
        _NECK_C_OUT = 256
        _NECK_HW    = ENC_S3_H * ENC_S3_W
        def _neck():
            self.matmat_mul_core(
                M=_NECK_HW, K=_NECK_C_IN, N=_NECK_C_OUT,
                A_DRAM_ADDR=self.S3B1_OUT_DRAM, B_DRAM_ADDR=self.NECK_C1_W,
                OUTPUT_DRAM_ADDR=self.NECK_BUF1_DRAM)
            self.layer_norm_core_dram(
                M=_NECK_HW, N=_NECK_C_OUT,
                A_DRAM_ADDR=self.NECK_BUF1_DRAM, OUTPUT_DRAM_ADDR=self.NECK_BUF1_DRAM,
                GAMMA_DRAM_ADDR=self.NECK_LN1_G, BETA_DRAM_ADDR=self.NECK_LN1_B)
        self._compile_run(_neck)
        self._assert_no_nan("neck_conv1+ln1", self.NECK_BUF1_DRAM, _NECK_HW * _NECK_C_OUT)

        _original_print("debug_encode: all checkpoints passed — NaN must be in neck conv2 or later")

    # ------------------------------------------------------------------
    # Binary dump (for run_from_bin)
    # ------------------------------------------------------------------

    def dump_params_to_file(self, bin_dir: str):
        """Dump params + tensor layout to bin_dir/params.bin + params.json."""
        os.makedirs(bin_dir, exist_ok=True)
        bin_path = os.path.join(bin_dir, "params.bin")
        meta_path = os.path.join(bin_dir, "params.json")
        total = self.get_params_dram_usage()
        CHUNK = 1 * 1024 * 1024
        with open(bin_path, "wb") as f:
            offset = 0
            while offset < total:
                sz = min(CHUNK, total - offset)
                buf = bytearray(sz)
                self.dma_read(DMA_DEVICE_C2H, self._params_dram_base + offset, buf, sz)
                f.write(buf)
                offset += sz
        # Tensor offsets relative to tensor DRAM base (re-allocated by run_from_bin)
        tensor_ofs = {
            "pe_in": self.PE_IN_DRAM - self._tensor_dram_base,
            "tokens": self.TOKENS_DRAM - self._tensor_dram_base,
            "tokens_pe": self.TOKENS_PE_DRAM - self._tensor_dram_base,
            "src": self.SRC_DRAM - self._tensor_dram_base,
            "key_pe": self.KEY_PE_DRAM - self._tensor_dram_base,
            "neck_out": self.NECK_OUT_DRAM - self._tensor_dram_base,
            "mask_out": self.MASK_OUT - self._tensor_dram_base,
            "iou_out": self.IOU_OUT - self._tensor_dram_base,
        }
        tensor_sizes = {
            "pe_in": ENC_IN_H * ENC_IN_W * ENC_CIN_PAD * BPE,
            "tokens": NT_PAD * DEC_DIM * BPE,
            "tokens_pe": NT_PAD * DEC_DIM * BPE,
            "src": GA * DEC_DIM * BPE,
            "key_pe": GA * DEC_DIM * BPE,
            "neck_out": 4096 * 256 * BPE,
            "mask_out": 4 * 65536 * BPE,
            "iou_out": 64 * BPE,
        }
        with open(meta_path, "w") as f:
            json.dump({"size": total, "tensors": tensor_ofs, "tensor_sizes": tensor_sizes}, f)
        _original_print(f"  Params: {total / 1024**2:.1f} MB → {bin_path}")

    def dump_programs_to_file(self, enc_prog_addr: int, dec_prog_addr: int, bin_dir: str):
        """Read encoder+decoder programs from DRAM and save to bin_dir/programs.bin + programs.json.

        Program sizes are inferred from sequential DRAM layout
        (encoder compiled first, decoder second).
        """
        os.makedirs(bin_dir, exist_ok=True)
        bin_path = os.path.join(bin_dir, "programs.bin")
        meta_path = os.path.join(bin_dir, "programs.json")
        total_usage = self.get_program_dram_usage()
        enc_size = dec_prog_addr - enc_prog_addr
        dec_size = total_usage - enc_size
        programs = [
            ("encoder", enc_prog_addr, enc_size),
            ("decoder", dec_prog_addr, dec_size),
        ]
        manifest = {"programs": {}}
        all_bytes = bytearray()
        CHUNK = 1 * 1024 * 1024
        for name, addr, size in programs:
            offset_in_file = len(all_bytes)
            remaining = size
            while remaining > 0:
                sz = min(CHUNK, remaining)
                buf = bytearray(sz)
                self.dma_read(DMA_DEVICE_C2H, addr + (size - remaining), buf, sz)
                all_bytes.extend(buf)
                remaining -= sz
            manifest["programs"][name] = {"offset": offset_in_file, "size": size}
        with open(bin_path, "wb") as f:
            f.write(all_bytes)
        with open(meta_path, "w") as f:
            json.dump(manifest, f)
        _original_print(f"  Programs: {len(all_bytes)} bytes → {bin_path}")

    def _encoder_ops(self, flush):
        """All encoder instruction ops."""
        # conv0: (1024×1024, 64) → (512×512, 64)  [Conv-BN-GELU]
        conv2d_3x3_stride2_dram(
            self, self.PE_IN_DRAM, self.PE_MID_DRAM, self.PE_IM2COL,
            self.PE_W0_DRAM, self.PE_B0_DRAM, self.PE_ZERO_DRAM,
            ENC_IN_H, ENC_IN_W, ENC_CIN_PAD, ENC_C0P, gelu_enable=True)

        # conv2: (512×512, 64) → (256×256, 64)  [Conv-BN]
        conv2d_3x3_stride2_dram(
            self, self.PE_MID_DRAM, self.PE_OUT_DRAM, self.PE_IM2COL,
            self.PE_W2_DRAM, self.PE_B2_DRAM, self.PE_ZERO_DRAM,
            ENC_IN_H // 2, ENC_IN_W // 2, ENC_C0P, ENC_C0P)

        flush()

        # Stage 0 Block 0 conv1: 1×1 expand (256×256, 64) → (256×256, 256) + GELU
        conv2d_1x1_dram(self, self.PE_OUT_DRAM, self.S0_EXP_DRAM,
                        self.S0B0_C1W, self.S0B0_C1B,
                        ENC_S0_H, ENC_S0_W, ENC_C0P, ENC_S0_C_EXP, gelu_enable=True)

        # Stage 0 Block 0 conv2: depthwise 3×3 (256×256, 256→256) + GELU
        conv2d_3x3_dw_dram(
            self, self.S0_EXP_DRAM, self.S0_DW_DRAM, self.S0_IM2COL,
            self.S0B0_C2W, self.S0B0_C2B, self.S0_ZERO_DRAM,
            ENC_S0_H, ENC_S0_W, ENC_S0_C_EXP, gelu_enable=True)

        flush()

        # Stage 0 Block 0 conv3: 1×1 project (256×256, 256→64)
        conv2d_1x1_dram(self, self.S0_DW_DRAM, self.S0B0_OUT_DRAM,
                        self.S0B0_C3W, self.S0B0_C3B,
                        ENC_S0_H, ENC_S0_W, ENC_S0_C_EXP, ENC_C0P)

        # residual add: x + conv3_out → S0B0_OUT_DRAM (PE_OUT_DRAM holds patch_embed_out)
        eltwise_add_dram(self, self.PE_OUT_DRAM, self.S0B0_OUT_DRAM,
                         self.S0B0_OUT_DRAM, ENC_S0_H * ENC_S0_W * ENC_C0P)
        # GELU in-place: re-run conv3 with gelu fused using identity weight on sum
        # Simpler: matmul with identity (C,C) to apply GELU row by row
        self.matmat_mul_core(
            M=ENC_S0_H * ENC_S0_W, K=ENC_C0P, N=ENC_C0P,
            A_DRAM_ADDR=self.S0B0_OUT_DRAM,
            B_DRAM_ADDR=self.IDENTITY_DRAM,
            OUTPUT_DRAM_ADDR=self.S0B0_OUT_DRAM,
            gelu_enable=True)

        flush()

        # ---- Stage 0 Block 1 ----
        conv2d_1x1_dram(self, self.S0B0_OUT_DRAM, self.S0_EXP_DRAM,
                        self.S0B1_C1W, self.S0B1_C1B,
                        ENC_S0_H, ENC_S0_W, ENC_C0P, ENC_S0_C_EXP, gelu_enable=True)
        conv2d_3x3_dw_dram(self, self.S0_EXP_DRAM, self.S0_DW_DRAM, self.S0_IM2COL,
                           self.S0B1_C2W, self.S0B1_C2B, self.S0_ZERO_DRAM,
                           ENC_S0_H, ENC_S0_W, ENC_S0_C_EXP, gelu_enable=True)
        conv2d_1x1_dram(self, self.S0_DW_DRAM, self.S0B1_OUT_DRAM,
                        self.S0B1_C3W, self.S0B1_C3B,
                        ENC_S0_H, ENC_S0_W, ENC_S0_C_EXP, ENC_C0P)
        eltwise_add_dram(self, self.S0B0_OUT_DRAM, self.S0B1_OUT_DRAM,
                         self.S0B1_OUT_DRAM, ENC_S0_H * ENC_S0_W * ENC_C0P)
        self.matmat_mul_core(
            M=ENC_S0_H * ENC_S0_W, K=ENC_C0P, N=ENC_C0P,
            A_DRAM_ADDR=self.S0B1_OUT_DRAM,
            B_DRAM_ADDR=self.IDENTITY_DRAM,
            OUTPUT_DRAM_ADDR=self.S0B1_OUT_DRAM,
            gelu_enable=True)
        flush()

        # ---- PatchMerging 0→1 ----
        conv2d_1x1_dram(self, self.S0B1_OUT_DRAM, self.PM01_EXP_DRAM,
                        self.PM01_W1_DRAM, self.PM01_B1_DRAM,
                        ENC_S0_H, ENC_S0_W, ENC_C0P, ENC_S1_C, gelu_enable=True)
        conv2d_3x3_dw_stride2_dram(
            self, self.PM01_EXP_DRAM, self.PM01_DW_DRAM, self.PM01_IM2COL_DRAM,
            self.PM01_W2_DRAM, self.PM01_B2_DRAM, self.PM01_ZERO_DRAM,
            ENC_S0_H, ENC_S0_W, ENC_S1_C, gelu_enable=True)
        conv2d_1x1_dram(self, self.PM01_DW_DRAM, self.PM01_OUT_DRAM,
                        self.PM01_W3_DRAM, self.PM01_B3_DRAM,
                        ENC_S1_H, ENC_S1_W, ENC_S1_C, ENC_S1_C)
        flush()

        # ---- Stage 1: 2× TinyViTBlock ----
        _HW1 = ENC_S1_H * ENC_S1_W
        self._run_s1_block(self.S1B[0], self.PM01_OUT_DRAM, self.S1B0_OUT_DRAM)
        flush()
        self._run_s1_block(self.S1B[1], self.S1B0_OUT_DRAM, self.S1B1_OUT_DRAM)
        flush()

        # ---- PatchMerging 1→2 ----
        _HW2 = ENC_S2_H * ENC_S2_W
        _CP  = ENC_S2_C_PAD
        conv2d_1x1_dram(self, self.S1B1_OUT_DRAM, self.PM12_EXP_DRAM,
                        self.PM12_W1_DRAM, self.PM12_B1_DRAM,
                        ENC_S1_H, ENC_S1_W, ENC_S1_C, _CP, gelu_enable=True)
        conv2d_3x3_dw_stride2_dram(self, self.PM12_EXP_DRAM, self.PM12_DW_DRAM,
                                   self.PM12_IM2COL_DRAM,
                                   self.PM12_W2_DRAM, self.PM12_B2_DRAM, self.PM12_ZERO_DRAM,
                                   ENC_S1_H, ENC_S1_W, _CP, gelu_enable=True)
        conv2d_1x1_dram(self, self.PM12_DW_DRAM, self.PM12_OUT_DRAM,
                        self.PM12_W3_DRAM, self.PM12_B3_DRAM,
                        ENC_S2_H, ENC_S2_W, _CP, _CP, gelu_enable=False)
        flush()

        # ---- Stage 2: 6× TinyViTBlock ----
        _s2_io = [
            (self.PM12_OUT_DRAM, self.S2B0_OUT_DRAM),
            (self.S2B0_OUT_DRAM, self.S2B1_OUT_DRAM),
            (self.S2B1_OUT_DRAM, self.S2B2_OUT_DRAM),
            (self.S2B2_OUT_DRAM, self.S2B3_OUT_DRAM),
            (self.S2B3_OUT_DRAM, self.S2B4_OUT_DRAM),
            (self.S2B4_OUT_DRAM, self.S2B5_OUT_DRAM),
        ]
        for _bi, (_in, _out) in enumerate(_s2_io):
            self._run_s2_block(self.S2B[_bi], _in, _out)
            flush()

        # ---- PatchMerging 2→3 ----
        _C23 = ENC_S3_C
        conv2d_1x1_dram(self, self.S2B5_OUT_DRAM, self.PM23_EXP_DRAM,
                        self.PM23_W1_DRAM, self.PM23_B1_DRAM,
                        ENC_S2_H, ENC_S2_W, _CP, _C23, gelu_enable=True)
        conv2d_3x3_dw_dram(self, self.PM23_EXP_DRAM, self.PM23_DW_DRAM, self.PM23_IM2COL_DRAM,
                           self.PM23_W2_DRAM, self.PM23_B2_DRAM, self.PM23_ZERO_DRAM,
                           ENC_S2_H, ENC_S2_W, _C23, gelu_enable=True)
        conv2d_1x1_dram(self, self.PM23_DW_DRAM, self.PM23_OUT_DRAM,
                        self.PM23_W3_DRAM, self.PM23_B3_DRAM,
                        ENC_S2_H, ENC_S2_W, _C23, _C23, gelu_enable=False)
        flush()

        # ---- Stage 3: 2× TinyViTBlock ----
        _C3  = ENC_S3_C
        _HW3 = ENC_S3_H * ENC_S3_W
        self._run_s3_block(self.S3B[0], self.PM23_OUT_DRAM, self.S3B0_OUT_DRAM)
        flush()
        self._run_s3_block(self.S3B[1], self.S3B0_OUT_DRAM, self.S3B1_OUT_DRAM)
        flush()

        # ---- Neck ----
        _NECK_C_IN  = ENC_S3_C    # 320
        _NECK_C_OUT = 256
        _NECK_H = _NECK_W = ENC_S3_H   # 64
        _NECK_HW = _NECK_H * _NECK_W   # 4096

        # neck.0: Conv1x1(320→256), no bias
        self.matmat_mul_core(
            M=_NECK_HW, K=_NECK_C_IN, N=_NECK_C_OUT,
            A_DRAM_ADDR=self.S3B1_OUT_DRAM,
            B_DRAM_ADDR=self.NECK_C1_W,
            OUTPUT_DRAM_ADDR=self.NECK_BUF1_DRAM,
        )
        # neck.1: LayerNorm2d(256) — normalizes over 256 channels per pixel
        self.layer_norm_core_dram(
            M=_NECK_HW, N=_NECK_C_OUT,
            A_DRAM_ADDR=self.NECK_BUF1_DRAM,
            OUTPUT_DRAM_ADDR=self.NECK_BUF1_DRAM,
            GAMMA_DRAM_ADDR=self.NECK_LN1_G,
            BETA_DRAM_ADDR=self.NECK_LN1_B,
        )
        flush()

        # neck.2: Conv3x3(256→256, pad=1), no bias
        conv2d_3x3_stride1_dram(
            self,
            INPUT_DRAM_ADDR=self.NECK_BUF1_DRAM,
            OUTPUT_DRAM_ADDR=self.NECK_OUT_DRAM,
            IM2COL_DRAM_ADDR=self.NECK_IM2COL_DRAM,
            WEIGHT_DRAM_ADDR=self.NECK_C3_W,
            BIAS_DRAM_ADDR=self.NECK_ZERO,
            ZERO_PAD_DRAM_ADDR=self.NECK_ZERO,
            H=_NECK_H, W=_NECK_W, C_in=_NECK_C_OUT, C_out=_NECK_C_OUT,
        )
        # neck.3: LayerNorm2d(256)
        self.layer_norm_core_dram(
            M=_NECK_HW, N=_NECK_C_OUT,
            A_DRAM_ADDR=self.NECK_OUT_DRAM,
            OUTPUT_DRAM_ADDR=self.NECK_OUT_DRAM,
            GAMMA_DRAM_ADDR=self.NECK_LN2_G,
            BETA_DRAM_ADDR=self.NECK_LN2_B,
        )
        flush()





# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _assemble_tokens(checkpoint_path: str, sparse_t: torch.Tensor) -> torch.Tensor:
    """Build (NT_PAD, 256) padded token tensor from iou+mask weights + sparse prompt.

    sparse_t: (2, 256) bf16 — [point_token, not-a-point-pad] from prompt_encoder.
    """
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    iou_tok   = sd["mask_decoder.iou_token.weight"].to(torch.bfloat16)    # (1, 256)
    mask_toks = sd["mask_decoder.mask_tokens.weight"].to(torch.bfloat16)  # (4, 256)

    tokens = torch.zeros(NT_PAD, DEC_DIM, dtype=torch.bfloat16)
    tokens[0]   = iou_tok[0]
    tokens[1:5] = mask_toks
    tokens[5:7] = sparse_t.to(torch.bfloat16).reshape(2, DEC_DIM)
    return tokens


def conv2d_3x3_dw_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                        IM2COL_DRAM_ADDR: int, WEIGHT_DRAM_ADDR: int, BIAS_DRAM_ADDR: int,
                        ZERO_PAD_DRAM_ADDR: int,
                        H: int, W: int, C: int,
                        gelu_enable: bool = False) -> None:
    """Depthwise Conv2d(kernel=3, stride=1, pad=1, groups=C). HWC layout.
    C must be a multiple of 64. Processes 64 channels at a time as a block-diagonal matmul.
    WEIGHT_DRAM_ADDR holds (C//64) block-diagonal matrices each (64, 9*64), contiguous.
    For each block the matmul output (H*W, 64) is scattered into the correct channel
    columns of OUTPUT_DRAM (H*W, C) via per-pixel SRAM copy.
    """
    BLK = 64
    K   = 9 * BLK
    W_CHUNK = max(1, URAM_NEAR_FULL_ELEMENTS // (K + BLK))
    n_blocks = C // BLK
    for blk in range(n_blocks):
        c_off  = blk * BLK
        W_ADDR = WEIGHT_DRAM_ADDR + blk * BLK * K * _BPE
        B_ADDR = BIAS_DRAM_ADDR   + c_off * _BPE
        TEMP_OUT = IM2COL_DRAM_ADDR + W_CHUNK * K * _BPE
        for h in range(H):
            for w_start in range(0, W, W_CHUNK):
                w_end   = min(w_start + W_CHUNK, W)
                w_count = w_end - w_start
                sram_off = 0
                for w in range(w_start, w_end):
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            nh, nw = h + dy, w + dx
                            patch_sram = sram_off + ((dy + 1) * 3 + (dx + 1)) * BLK * _BPE
                            if 0 <= nh < H and 0 <= nw < W:
                                ue.accelerator_memory_to_sram(
                                    INPUT_DRAM_ADDR + (nh * W + nw) * C * _BPE + c_off * _BPE,
                                    patch_sram, BLK)
                            else:
                                ue.accelerator_memory_to_sram(ZERO_PAD_DRAM_ADDR, patch_sram, BLK)
                    sram_off += K * _BPE
                ue.sram_to_accelerator_memory(0x00000, IM2COL_DRAM_ADDR, w_count * K)
                ue.matmat_mul_core(
                    M=w_count, K=K, N=BLK,
                    A_DRAM_ADDR=IM2COL_DRAM_ADDR,
                    B_DRAM_ADDR=W_ADDR,
                    OUTPUT_DRAM_ADDR=TEMP_OUT,
                    C_DRAM_ADDR=B_ADDR, bias_mode="broadcast_N",
                    gelu_enable=gelu_enable)
                # strided write: scatter (w_count, BLK) temp into correct channel columns
                ue.accelerator_memory_to_sram(TEMP_OUT, 0x00000, w_count * BLK)
                ue.sram_to_accelerator_memory(
                    0x00000,
                    OUTPUT_DRAM_ADDR + (h * W + w_start) * C * _BPE + c_off * _BPE,
                    w_count * BLK,
                    stride_bytes_per_chunk=BLK * _BPE,
                    stride_jump_bytes=C * _BPE)


def conv2d_3x3_dw_stride2_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                                IM2COL_DRAM_ADDR: int, WEIGHT_DRAM_ADDR: int, BIAS_DRAM_ADDR: int,
                                ZERO_PAD_DRAM_ADDR: int,
                                H_in: int, W_in: int, C: int,
                                gelu_enable: bool = False) -> None:
    """Depthwise Conv2d(kernel=3, stride=2, pad=1, groups=C). HWC layout.
    C must be a multiple of 64. Block-diagonal matmul + strided scatter, same as
    conv2d_3x3_dw_dram but with stride=2 in the spatial gather loop.
    """
    BLK = 64
    K   = 9 * BLK
    H_out = (H_in - 1) // 2 + 1
    W_out = (W_in - 1) // 2 + 1
    W_CHUNK = max(1, URAM_NEAR_FULL_ELEMENTS // (K + BLK))
    n_blocks = C // BLK
    for blk in range(n_blocks):
        c_off  = blk * BLK
        W_ADDR = WEIGHT_DRAM_ADDR + blk * BLK * K * _BPE
        B_ADDR = BIAS_DRAM_ADDR   + c_off * _BPE
        TEMP_OUT = IM2COL_DRAM_ADDR + W_CHUNK * K * _BPE
        for h_out in range(H_out):
            h_in = h_out * 2
            for w_start in range(0, W_out, W_CHUNK):
                w_end   = min(w_start + W_CHUNK, W_out)
                w_count = w_end - w_start
                sram_off = 0
                for w_out in range(w_start, w_end):
                    w_in = w_out * 2
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            nh, nw = h_in + dy, w_in + dx
                            patch_sram = sram_off + ((dy + 1) * 3 + (dx + 1)) * BLK * _BPE
                            if 0 <= nh < H_in and 0 <= nw < W_in:
                                ue.accelerator_memory_to_sram(
                                    INPUT_DRAM_ADDR + (nh * W_in + nw) * C * _BPE + c_off * _BPE,
                                    patch_sram, BLK)
                            else:
                                ue.accelerator_memory_to_sram(ZERO_PAD_DRAM_ADDR, patch_sram, BLK)
                    sram_off += K * _BPE
                ue.sram_to_accelerator_memory(0x00000, IM2COL_DRAM_ADDR, w_count * K)
                ue.matmat_mul_core(
                    M=w_count, K=K, N=BLK,
                    A_DRAM_ADDR=IM2COL_DRAM_ADDR,
                    B_DRAM_ADDR=W_ADDR,
                    OUTPUT_DRAM_ADDR=TEMP_OUT,
                    C_DRAM_ADDR=B_ADDR, bias_mode="broadcast_N",
                    gelu_enable=gelu_enable)
                ue.accelerator_memory_to_sram(TEMP_OUT, 0x00000, w_count * BLK)
                ue.sram_to_accelerator_memory(
                    0x00000,
                    OUTPUT_DRAM_ADDR + (h_out * W_out + w_start) * C * _BPE + c_off * _BPE,
                    w_count * BLK,
                    stride_bytes_per_chunk=BLK * _BPE,
                    stride_jump_bytes=C * _BPE)


def window_partition_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                          H: int, W: int, C: int, ws: int) -> tuple:
    """(H*W, C) HWC → (nH*nW, WIN_PAD, C). Returns (nH, nW, WIN_PAD)."""
    pad_h  = (ws - H % ws) % ws
    pad_w  = (ws - W % ws) % ws
    pH, pW = H + pad_h, W + pad_w
    nH, nW = pH // ws, pW // ws
    WIN_REAL = ws * ws
    WIN_PAD  = ((WIN_REAL + 63) // 64) * 64
    total_out = nH * nW * WIN_PAD * C
    chunk = min(URAM_NEAR_FULL_ELEMENTS, total_out)
    z_addr = ue.get_params_dram_addr()
    ue.allocate_params_dram(chunk * _BPE)
    ue.dma_write(DMA_DEVICE_H2C, z_addr, torch.zeros(chunk, dtype=torch.bfloat16), chunk * _BPE)
    ue.accelerator_memory_to_sram(z_addr, 0x00000, chunk)
    offset = 0
    while offset < total_out:
        take = min(chunk, total_out - offset)
        ue.sram_to_accelerator_memory(0x00000, OUTPUT_DRAM_ADDR + offset * _BPE, take)
        offset += take
    for wh in range(nH):
        for ww in range(nW):
            win_idx = wh * nW + ww
            for dy in range(ws):
                src_h = wh * ws + dy
                if src_h >= H:
                    continue
                src_w    = ww * ws
                actual_w = min(ws, W - src_w)
                if actual_w <= 0:
                    continue
                n_elem   = actual_w * C
                src_dram = INPUT_DRAM_ADDR + (src_h * W + src_w) * C * _BPE
                dst_dram = OUTPUT_DRAM_ADDR + (win_idx * WIN_PAD + dy * ws) * C * _BPE
                ue.accelerator_memory_to_sram(src_dram, 0x00000, n_elem)
                ue.sram_to_accelerator_memory(0x00000, dst_dram, n_elem)
    return nH, nW, WIN_PAD


def window_reverse_dram(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                        H: int, W: int, C: int, ws: int, nH: int, nW: int, WIN_PAD: int) -> None:
    """(nH*nW, WIN_PAD, C) → (H*W, C). Clips boundary windows."""
    for wh in range(nH):
        for ww in range(nW):
            win_idx = wh * nW + ww
            for dy in range(ws):
                src_h    = wh * ws + dy
                if src_h >= H:
                    continue
                src_w    = ww * ws
                actual_w = min(ws, W - src_w)
                if actual_w <= 0:
                    continue
                n_elem   = actual_w * C
                src_dram = INPUT_DRAM_ADDR + (win_idx * WIN_PAD + dy * ws) * C * _BPE
                dst_dram = OUTPUT_DRAM_ADDR + (src_h * W + src_w) * C * _BPE
                ue.accelerator_memory_to_sram(src_dram, 0x00000, n_elem)
                ue.sram_to_accelerator_memory(0x00000, dst_dram, n_elem)


def _build_unpad_weight(num_heads: int, head_dim: int, head_dim_pad: int) -> torch.Tensor:
    """Selection matrix (num_heads*head_dim_pad, num_heads*head_dim) for stripping head_dim padding."""
    K = num_heads * head_dim_pad
    N = num_heads * head_dim
    W = torch.zeros(K, N, dtype=torch.bfloat16)
    for h in range(num_heads):
        for d in range(head_dim):
            W[h * head_dim_pad + d, h * head_dim + d] = 1.0
    return W


def _bn_fold(w_conv, bn_w, bn_b, bn_mean, bn_var, eps=1e-5):
    """Fold BatchNorm into conv weight and bias at load time.

    Returns (w_fused, b_fused) where:
      w_fused[c] = w_conv[c] * (bn_w[c] / sqrt(bn_var[c] + eps))
      b_fused[c] = bn_b[c]   - bn_mean[c] * bn_w[c] / sqrt(bn_var[c] + eps)
    """
    scale = bn_w / (bn_var + eps).sqrt()                        # (C_out,)
    w_fused = w_conv * scale[:, None, None, None]               # broadcast over kH,kW,C_in
    b_fused = bn_b - bn_mean * scale
    return w_fused.to(torch.bfloat16), b_fused.to(torch.bfloat16)
def _clock_ns_default_for_device(device: str) -> float:
    """Return default clock period (ns) for FPGA type — mirrors user_hw_test.py."""
    if device == "kintex7":                       return 5.1594
    if device in ("rk", "puzhi"):                 return 3.0
    if device in ("bittware", "bittware_256"):     return 3.3333
    if device == "alveo":                          return 4.0
    return 10.0


def main():
    parser = argparse.ArgumentParser(description="MobileSAM mask decoder accelerator test")
    parser.add_argument("--dev",   default="xdma0")
    parser.add_argument("--cycle", type=float, default=None, help='Clock cycle time in ns. Overrides --device default.')
    parser.add_argument("--device", type=str, default="kintex7", help='FPGA board profile (kintex7, rk, puzhi, bittware, bittware_256, alveo).')
    parser.add_argument("--point", nargs=2, type=int, metavar=("X", "Y"),
                        default=[512, 512],
                        help="Single-point inference: encode image, run decoder for this point, save best mask")
    parser.add_argument("--debug-encode", action="store_true",
                        help="Step through encoder stage-by-stage, asserting no NaNs at each checkpoint")
    args = parser.parse_args()

    set_dma_device(args.dev)
    axi_width_bits = 512 if args.device in ("bittware", "rk") else 256
    os.environ["UE_AXI_DATA_WIDTH_BITS"] = str(axi_width_bits)
    user_dma_core.UE_AXI_DATA_WIDTH_BITS = axi_width_bits
    clock = args.cycle if args.cycle is not None else _clock_ns_default_for_device(args.device)
    user_dma_core.CLOCK_CYCLE_TIME_NS = clock
    user_dma_core.UE_PEAK_GFLOPS = 0.128 / clock
    _original_print(f"FPGA profile: device={args.device}, clock={clock:.4f} ns, UE_AXI_DATA_WIDTH_BITS={axi_width_bits}")

    if not os.path.exists(WEIGHTS):
        import urllib.request
        _original_print(f"Weights not found, downloading to {WEIGHTS} …")
        os.makedirs(BIN_DIR, exist_ok=True)
        urllib.request.urlretrieve(
            "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
            WEIGHTS,
        )
        _original_print("  Download complete.")

    _original_print("MobileSAM — HW test")
    _original_print(f"  Device: {args.dev}  clock={clock:.4f} ns")

    # ---- Inputs ----
    from PIL import Image as _PIL_Image
    import numpy as _np
    _img = _PIL_Image.open(os.path.join(SCRIPT_DIR, "../../test_samples/vette.jpg")).convert("RGB")
    _img = _img.resize((ENC_IN_W, ENC_IN_H), _PIL_Image.BILINEAR)
    _img_arr = torch.from_numpy(_np.array(_img)).float().permute(2, 0, 1) / 255.0
    image_t  = _img_arr.unsqueeze(0).bfloat16()

    # Load checkpoint weights directly (no mobile_sam package needed)
    _ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    _gauss = _ckpt["prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"]
    dense_t = _ckpt["prompt_encoder.no_mask_embed.weight"].reshape(1, DEC_DIM).expand(GA, -1).bfloat16().contiguous()
    # Positional encoding (PositionEmbeddingRandom.forward equivalent)
    _tmp = torch.ones(IMG_H, IMG_W)
    _gy = (_tmp.cumsum(dim=0) - 0.5) / IMG_H
    _gx = (_tmp.cumsum(dim=1) - 0.5) / IMG_W
    _coords = 2 * torch.stack([_gx, _gy], dim=-1) - 1
    _pe_raw = 2 * math.pi * (_coords @ _gauss)
    _pe = torch.cat([torch.sin(_pe_raw), torch.cos(_pe_raw)], dim=-1)
    image_pe_t = _pe.permute(2, 0, 1)[None][0].permute(1, 2, 0).reshape(GA, DEC_DIM).bfloat16()

    # ---- Compile or load from bins ----
    bins_exist = (os.path.exists(os.path.join(BIN_DIR, "params.bin"))
                  and os.path.exists(os.path.join(BIN_DIR, "programs.bin")))
    if bins_exist:
        _original_print("\nLoading from pre-compiled bins …")
        from mobilesam_run_from_bin import MobileSAM_UE_Run
        ue = MobileSAM_UE_Run()
        ue.load_params()
        with open(os.path.join(BIN_DIR, "params.json")) as f:
            _pm = json.load(f)
        _tb = ue._tensor_dram_base
        # Set all tensor attrs the rest of the flow expects
        # (some names omit the _DRAM suffix, handle those explicitly)
        for name, ofs in _pm["tensors"].items():
            attr = "MASK_OUT" if name == "mask_out" else \
                   "IOU_OUT" if name == "iou_out" else \
                   name.upper() + "_DRAM"
            setattr(ue, attr, _tb + ofs)
        ue.pe_in_dram = ue.PE_IN_DRAM
        progs = ue.load_programs()
        enc_prog = progs["encoder"]
        dec_prog = progs["decoder"]
        # Stitch on the decoder interface methods the rest of the flow expects
        import types
        def _preload(self, image_emb_t, image_pe_t, dense_t):
            src_t = (image_emb_t + dense_t).to(torch.bfloat16).contiguous()
            self.dma_to_accelerator_memory(self.SRC_DRAM, src_t.flatten())
            self.dma_to_accelerator_memory(self.KEY_PE_DRAM, image_pe_t.flatten())
        def _run_dec(self, prog_addr, tokens_t, timeout=120.0):
            toks = tokens_t.to(torch.bfloat16).contiguous()
            self.dma_to_accelerator_memory(self.TOKENS_DRAM, toks.flatten())
            self.dma_to_accelerator_memory(self.TOKENS_PE_DRAM, toks.flatten())
            self.start_execute_from_dram(prog_addr)
            self.wait_queue(timeout)
            masks = self.dma_from_accelerator_memory(self.MASK_OUT, (4, 256*256)).float().reshape(4,256,256)
            iou = self.dma_from_accelerator_memory(self.IOU_OUT, (64,)).float()[:4]
            return masks, iou
        ue.preload_decoder_image = types.MethodType(_preload, ue)
        ue.run_decoder_tokens = types.MethodType(_run_dec, ue)
        _original_print("  Loaded from bins.")
    else:
        # IMPORTANT: create both UE objects first (params+tensors alloc), THEN compile.
        # Both share the same physical DRAM. If we compiled enc before creating ue,
        # ue's __init__ param allocs would overwrite enc's compiled programs.
        _original_print("\nCompiling … (this will take a moment...)")
        import threading as _th
        _t0 = time.perf_counter()
        _stop_timer = False
        def _roll_timer():
            while not _stop_timer:
                _original_print(f"\r  Elapsed: {time.perf_counter() - _t0:.1f}s", end="", flush=True)
                time.sleep(0.2)
        _timer = _th.Thread(target=_roll_timer, daemon=True)
        _timer.start()
        ue = MobileSAM_UE(WEIGHTS)
        enc_prog = ue.compile_encoder()
        _original_print(f"\r  Encoder: 1 program  ({time.perf_counter() - _t0:.1f}s)")
        dec_prog = ue.compile_decoder()
        _stop_timer = True
        _timer.join()
        _original_print(f"  Decoder: done  ({time.perf_counter() - _t0:.1f}s total)")

        # ---- Save bins ----
        _original_print("\nSaving bins …")
        ue.dump_params_to_file(BIN_DIR)
        ue.dump_programs_to_file(enc_prog, dec_prog, BIN_DIR)
        _original_print("  Done saving bins.")

    if args.debug_encode:
        if bins_exist:
            _original_print("--debug-encode requires a fresh compile (bins found — delete mobilesam_bin/ and re-run)")
        else:
            ue.software_reset()
            ue.debug_encode(image_t)
        return

    if args.point:
        px, py = args.point
        _original_print(f"\nSingle-point inference at ({px}, {py}) …")
        from PIL import Image as _PIL
        import numpy as _np
        _pw = _get_prompt_weights()
        coord = torch.tensor([[float(px), float(py)]])
        sparse_tok = _amg_encode_point(coord, _pw)  # (2, 256)
        tokens_t = _assemble_tokens(WEIGHTS, sparse_tok)

        t_enc = time.perf_counter()
        ue.execute_encoder(enc_prog, image_t)
        _original_print(f"  HW encoder done in {time.perf_counter() - t_enc:.2f}s")
        image_emb_t = ue.dma_from_accelerator_memory(ue.NECK_OUT_DRAM, (GA, DEC_DIM)).bfloat16()
        ue.preload_decoder_image(image_emb_t, image_pe_t, dense_t)
        t_dec = time.perf_counter()
        masks_hw, iou_hw = ue.run_decoder_tokens(dec_prog, tokens_t)
        _original_print(f"  HW decoder done in {time.perf_counter() - t_dec:.3f}s")
        best = iou_hw.argmax().item()
        _original_print(f"  HW  IOU: {[round(x,4) for x in iou_hw.tolist()]}  best={best}")

        def _save_point_overlay(mask_256, path, color):
            mask_1024 = _np.array(_PIL.fromarray(mask_256).resize((1024, 1024), _PIL.NEAREST))
            img = image_t[0].float().permute(1, 2, 0).numpy()
            img = ((img - img.min()) / (img.max() - img.min() + 1e-6) * 255).astype(_np.uint8)
            overlay = img.copy().astype(_np.float32)
            overlay[mask_1024] = overlay[mask_1024] * 0.4 + _np.array(color, dtype=_np.float32) * 0.6
            overlay = overlay.astype(_np.uint8)
            for dy in range(-6, 7):
                for dx in range(-6, 7):
                    if abs(dy) + abs(dx) <= 6:
                        ry, rx = _np.clip(py + dy, 0, 1023), _np.clip(px + dx, 0, 1023)
                        overlay[ry, rx] = [255, 50, 50]
            _PIL.fromarray(overlay).save(path)
            _original_print(f"  Saved {path}")

        _save_point_overlay((masks_hw[best] > _MASK_THRESHOLD).cpu().numpy(), "mask_point.png", [0, 255, 100])
        return

    # ---- Execute (HW encoder + AMG decoder) ----
    _original_print("\nExecuting …")
    t0 = time.perf_counter()
    ue.execute_encoder(enc_prog, image_t)
    _original_print(f"  HW encoder done in {time.perf_counter() - t0:.2f}s")
    if hasattr(ue, 'S3B1_OUT_DRAM'):
        s3b1_t = ue.dma_from_accelerator_memory(ue.S3B1_OUT_DRAM,
                     (ENC_S3_H * ENC_S3_W, ENC_S3_C)).bfloat16().float()
        _original_print(f"  S3B1_OUT:    mean={s3b1_t.mean():.4f} std={s3b1_t.std():.4f} "
                        f"min={s3b1_t.min():.4f} max={s3b1_t.max():.4f}")
    image_emb_t = ue.dma_from_accelerator_memory(ue.NECK_OUT_DRAM, (GA, DEC_DIM)).bfloat16()
    _original_print(f"  NECK_OUT:    mean={image_emb_t.float().mean():.4f} std={image_emb_t.float().std():.4f} "
                    f"min={image_emb_t.float().min():.4f} max={image_emb_t.float().max():.4f}")
    _original_print("Running HW AMG …")
    t0 = time.perf_counter()
    hw_masks = _amg_run_hw(ue, dec_prog, image_emb_t, image_pe_t, dense_t, _get_prompt_weights(), points_per_side=4)
    _original_print(f"  {len(hw_masks)} masks in {time.perf_counter() - t0:.2f}s")

    # ---- Overlays ----
    _original_print("\nSaving overlays …")
    _save_amg_overlay(image_t, hw_masks,  "mask_hw.png")


# ---------------------------------------------------------------------------
# AMG helpers
# ---------------------------------------------------------------------------

_AMG_POINTS_PER_SIDE = 16
_MASK_THRESHOLD      = 0.0
_PRED_IOU_THRESH     = 0.88
_STABILITY_THRESH    = 0.95
_STABILITY_OFFSET    = 1.0
_BOX_NMS_THRESH      = 0.7


def _amg_grid_points(points_per_side: int, img_size: int = 1024):
    step = 1.0 / points_per_side
    return torch.tensor([
        [int((x + 0.5) * step * img_size), int((y + 0.5) * step * img_size)]
        for y in range(points_per_side) for x in range(points_per_side)
    ], dtype=torch.float32)   # (N, 2)


def _amg_encode_point(coord_xy: torch.Tensor, pw: dict) -> torch.Tensor:
    """Build (2, 256) bf16 tokens for one foreground point. Uses prompt weights dict."""
    pts = coord_xy.reshape(1, 1, 2).float() + 0.5
    lbs = torch.ones(1, 1, dtype=torch.long)
    # Pad with a not-a-point (SAM always appends this when no boxes)
    pad_pt = torch.zeros(1, 1, 2)
    pad_lb = -torch.ones(1, 1, dtype=torch.long)
    pts = torch.cat([pts, pad_pt], dim=1)
    lbs = torch.cat([lbs, pad_lb], dim=1)
    # Positional encoding via forward_with_coords
    coords = pts.clone()
    coords[:, :, 0] /= 1024.0
    coords[:, :, 1] /= 1024.0
    coords = 2 * coords - 1
    pe = coords @ pw["gauss_matrix"]  # (1, 2, 128)
    pe = 2 * math.pi * pe
    point_embedding = torch.cat([torch.sin(pe), torch.cos(pe)], dim=-1)  # (1, 2, 256)
    # Apply point embeddings (labels: 1=foreground, -1=not-a-point)
    nap = (lbs == -1)[0]  # (2,) — squeeze batch dim for boolean indexing
    fg = (lbs == 1)[0]    # (2,)
    point_embedding[:, nap, :] = 0.0
    point_embedding[:, nap, :] += pw["not_a_point"]
    point_embedding[:, fg, :] += pw["point_1"]
    return point_embedding[0].bfloat16()  # (2, 256)


def _amg_filter_and_nms(all_logits: list, all_iou: list, orig_hw, scaled_hw,
                         pred_iou_thresh=_PRED_IOU_THRESH, stab_thresh=_STABILITY_THRESH,
                         tag=""):
    """Concat, filter by IOU + stability, NMS. Returns list of bool np arrays at orig_hw."""
    import numpy as _np
    from PIL import Image as _PIL

    logits     = torch.cat(all_logits, 0)   # (N*4, 256, 256)
    iou_scores = torch.cat(all_iou,    0)   # (N*4,)

    if tag:
        _original_print(f"  [{tag}] IOU scores: min={iou_scores.min():.3f} max={iou_scores.max():.3f} "
                        f"mean={iou_scores.mean():.3f} >thresh={(iou_scores > pred_iou_thresh).sum().item()}/{len(iou_scores)}")

    keep = iou_scores > pred_iou_thresh
    logits, iou_scores = logits[keep], iou_scores[keep]

    hi   = (logits > (_MASK_THRESHOLD + _STABILITY_OFFSET)).float().sum(-1).sum(-1)
    lo   = (logits > (_MASK_THRESHOLD - _STABILITY_OFFSET)).float().sum(-1).sum(-1)
    stab = hi / (lo + 1e-6)
    keep = stab >= stab_thresh
    logits, iou_scores = logits[keep], iou_scores[keep]

    if tag:
        _original_print(f"  [{tag}] after stability filter: {logits.shape[0]} masks remain")

    if logits.shape[0] == 0:
        return []

    sh, sw = scaled_hw
    oh, ow = orig_hw
    masks_bin  = logits > _MASK_THRESHOLD
    masks_crop = masks_bin[:, :sh, :sw]
    masks_orig = torch.stack([
        torch.from_numpy(_np.array(
            _PIL.fromarray(m.cpu().numpy()).resize((ow, oh), _PIL.NEAREST)
        )) for m in masks_crop
    ])

    # box NMS
    n = masks_orig.shape[0]
    boxes = torch.zeros(n, 4)
    for i, m in enumerate(masks_orig):
        ys, xs = torch.where(m)
        if len(ys):
            boxes[i] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()], dtype=torch.float32)
    kept_idx = nms(boxes, iou_scores.cpu(), _BOX_NMS_THRESH)
    return [masks_orig[i].numpy() for i in kept_idx]


def _amg_run_hw(ue: 'MobileSAM_UE', dec_prog: int,
                image_emb_t: torch.Tensor, image_pe_t: torch.Tensor, dense_t: torch.Tensor,
                pw: dict, points_per_side: int = _AMG_POINTS_PER_SIDE):
    """HW AMG: encoder output already in NECK_OUT_DRAM. Runs decoder per grid point."""
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    iou_tok   = sd["mask_decoder.iou_token.weight"].to(torch.bfloat16)
    mask_toks = sd["mask_decoder.mask_tokens.weight"].to(torch.bfloat16)

    ue.preload_decoder_image(image_emb_t, image_pe_t, dense_t)
    grid = _amg_grid_points(points_per_side)
    all_logits, all_iou = [], []
    for coord in grid:
        point_tok = _amg_encode_point(coord, pw)  # (2, 256)
        tokens_t  = torch.zeros(NT_PAD, DEC_DIM, dtype=torch.bfloat16)
        tokens_t[0]   = iou_tok[0]
        tokens_t[1:5] = mask_toks
        tokens_t[5:7] = point_tok
        masks, iou = ue.run_decoder_tokens(dec_prog, tokens_t)
        all_logits.append(masks)   # (4, 256, 256)
        all_iou.append(iou)        # (4,)
    return _amg_filter_and_nms(all_logits, all_iou,
                                orig_hw=(1024, 1024), scaled_hw=(256, 256),
                                pred_iou_thresh=0.7, stab_thresh=0.85, tag="HW")


def _save_amg_overlay(image_t: torch.Tensor, masks: list, path: str) -> None:
    """Overlay all AMG masks on image_t (1,3,H,W bf16), save to path."""
    import numpy as _np
    from PIL import Image as _PIL

    img = image_t[0].float().permute(1, 2, 0).numpy()
    img = ((img - img.min()) / (img.max() - img.min() + 1e-6) * 255).astype(_np.uint8)
    overlay = img.copy().astype(_np.float32)
    rng = _np.random.default_rng(42)
    for m in masks:
        color = rng.integers(60, 255, 3).astype(_np.float32)
        overlay[m] = overlay[m] * 0.4 + color * 0.6
    _PIL.fromarray(overlay.astype(_np.uint8)).save(path)
    _original_print(f"  {path}  ({len(masks)} masks)")


if __name__ == "__main__":
    main()
