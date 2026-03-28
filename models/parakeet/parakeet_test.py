#!/usr/bin/env python3
"""
Parakeet-TDT-0.6B inference on FPGA accelerator (BF16).

Subsampling, encoder, and decoder all run on-device.
CPU only does mel extraction and decoder loop orchestration
(register writes, program dispatch, argmax reads).

Usage:
  python parakeet_test.py
  python parakeet_test.py --audio test.wav
  python parakeet_test.py --dev xdma0 [--cycle 5.63]
"""

import json
import math
import os
import builtins
import struct
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

import numpy as np
import torch
import torch.nn.functional as F

# Silent mode: suppress internal prints during compilation
_original_print = builtins.print
_SILENT_MODE = False

def quiet_print(*args, **kwargs):
    if _SILENT_MODE:
        return
    _original_print(*args, **kwargs)

builtins.print = quiet_print

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, DMA_DEVICE_C2H, DRAM_INSTRUCTION_ADDR, TYPE, UE_VECTOR_SIZE,
    URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS, UnifiedEngine, set_dma_device,
    UE_MODE, BROADCAST_MODE, LALU_MODE, MEMCPY_TYPE, URAM_SECTION, UE_ARGMAX1_INDEX, UE_ARGMAX2_INDEX
)

URAM_A_BASE = 0x00000
URAM_B_BASE = 0x80000
EPS = 1e-5

def _make_add_set_bytes(dst_reg, immediate_value):
    """Build raw 32-byte ADD_SET instruction: dst_reg = immediate_value."""
    INSTRUCTION_ADD = 2
    INST_ADD_SET = 4
    w = [0] * 8
    w[0] = ((INST_ADD_SET & 0xF) << 0) | ((dst_reg & 0xF) << 4) | ((dst_reg & 0xF) << 8)
    w[1] = immediate_value & 0xFFFFFFFF
    w[7] = (INSTRUCTION_ADD & 0x7) << 29
    result = bytearray(32)
    for i in range(8):
        result[i*4:(i+1)*4] = struct.pack('<I', w[i] & 0xFFFFFFFF)
    return bytes(result)

WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "parakeet_bin", "model_weights.ckpt")
TOKENIZER_PATH = os.path.join(SCRIPT_DIR, "parakeet_bin", "tokenizer.model")
# ---------------------------------------------------------------------------
# Core functions: custom ops for Parakeet Conformer
# ---------------------------------------------------------------------------
def batch_norm_fuse_params(ue: UnifiedEngine, bn_weight, bn_bias, bn_mean, bn_var, eps=EPS):
    """Pre-fuse BN params to scale+shift, write to DRAM. Call once at weight load.
    Returns (scale_dram_addr, shift_dram_addr, C).
    """
    scale = (bn_weight.float() / torch.sqrt(bn_var.float() + eps)).to(torch.bfloat16).contiguous()
    shift = (bn_bias.float() - bn_mean.float() * scale.float()).to(torch.bfloat16).contiguous()
    C = scale.shape[0]
    scale_addr = ue.get_params_dram_addr()
    ue.allocate_params_dram(C * 2)
    ue.dma_write(DMA_DEVICE_H2C, scale_addr, scale, C * 2)
    shift_addr = ue.get_params_dram_addr()
    ue.allocate_params_dram(C * 2)
    ue.dma_write(DMA_DEVICE_H2C, shift_addr, shift, C * 2)
    return scale_addr, shift_addr, C
def batch_norm_prepare_tiled(ue, C, L, SCALE_DRAM_ADDR, SHIFT_DRAM_ADDR):
    """Pre-tile BN scale/shift vectors into (C, L) matrices for bulk ops.
    Returns (tiled_scale_addr, tiled_shift_addr).
    Call once before compile, not inside instruction capture.
    """
    bpe = 2
    # Read scale and shift from DRAM
    scale_host = torch.zeros(C, dtype=torch.bfloat16)
    shift_host = torch.zeros(C, dtype=torch.bfloat16)
    ue.dma_read(DMA_DEVICE_C2H, SCALE_DRAM_ADDR, scale_host, C * bpe)
    ue.dma_read(DMA_DEVICE_C2H, SHIFT_DRAM_ADDR, shift_host, C * bpe)
    # Tile: each scalar repeated L times per row
    scale_tiled = scale_host.unsqueeze(1).expand(C, L).contiguous()
    shift_tiled = shift_host.unsqueeze(1).expand(C, L).contiguous()
    # Write to DRAM
    scale_tiled_addr = ue.get_params_dram_addr()
    ue.allocate_params_dram(C * L * bpe)
    ue.dma_to_accelerator_memory(scale_tiled_addr, scale_tiled)
    shift_tiled_addr = ue.get_params_dram_addr()
    ue.allocate_params_dram(C * L * bpe)
    ue.dma_to_accelerator_memory(shift_tiled_addr, shift_tiled)
    return scale_tiled_addr, shift_tiled_addr
def batch_norm_core_dram(ue: UnifiedEngine, C: int, L: int,
                         A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                         SCALE_DRAM_ADDR: int, SHIFT_DRAM_ADDR: int,
                         tiled_scale_addr: int = None,
                         tiled_shift_addr: int = None) -> None:
    """Emit instructions for fused eval-mode batch norm on (C, L) tensor.
    If tiled_scale_addr and tiled_shift_addr are provided, uses bulk ops (6 instructions).
    Otherwise falls back to per-channel loop.
    """
    assert L % UE_VECTOR_SIZE == 0, f"L={L} must be a multiple of {UE_VECTOR_SIZE}"
    if tiled_scale_addr is not None and tiled_shift_addr is not None:
        total_elems = C * L
        # Load input, load tiled scale, multiply
        ue.accelerator_memory_to_sram(A_DRAM_ADDR, URAM_A_BASE, total_elems)
        ue.accelerator_memory_to_sram(tiled_scale_addr, URAM_B_BASE, total_elems)
        ue.eltwise_mul_core(vector_A_sram_start_addr=URAM_A_BASE,
                           vector_B_sram_start_addr=URAM_B_BASE,
                           vector_C_sram_wb_addr=URAM_A_BASE, element_size=total_elems)
        # Load tiled shift, add
        ue.accelerator_memory_to_sram(tiled_shift_addr, URAM_B_BASE, total_elems)
        ue.eltwise_add_core(vector_A_sram_start_addr=URAM_A_BASE,
                           vector_B_sram_start_addr=URAM_B_BASE,
                           vector_C_sram_wb_addr=URAM_A_BASE, element_size=total_elems)
        # Write result
        ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR, total_elems)
        return
    # Fallback: per-channel (original behavior)
    row_bytes = L * 2
    scale_host = torch.zeros(C, dtype=torch.bfloat16)
    shift_host = torch.zeros(C, dtype=torch.bfloat16)
    ue.dma_read(DMA_DEVICE_C2H, SCALE_DRAM_ADDR, scale_host, C * 2)
    ue.dma_read(DMA_DEVICE_C2H, SHIFT_DRAM_ADDR, shift_host, C * 2)
    for c in range(C):
        ue.accelerator_memory_to_sram(A_DRAM_ADDR + c * row_bytes, URAM_A_BASE, L)
        ue.broadcast_mul(scalar=scale_host[c].float().item(), sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=L)
        ue.broadcast_add(scalar=shift_host[c].float().item(), sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=L)
        ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR + c * row_bytes, L)
def allocate_identity(ue: UnifiedEngine, N: int):
    """Allocate and write an (N, N) bf16 identity matrix to DRAM. Call once at init.
    Returns DRAM address. Reused by tanh_core_dram and glu_core_dram.
    """
    assert N % UE_VECTOR_SIZE == 0, f"N={N} must be a multiple of {UE_VECTOR_SIZE}"
    identity = torch.eye(N, dtype=torch.bfloat16).contiguous()
    addr = ue.get_params_dram_addr()
    ue.allocate_params_dram(N * N * 2)
    ue.dma_write(DMA_DEVICE_H2C, addr, identity, N * N * 2)
    return addr
def tanh_core_dram(ue: UnifiedEngine, M: int, N: int,
                   A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                   IDENTITY_DRAM_ADDR: int) -> None:
    """Emit instructions for tanh(x) = 2*sigmoid(2x) - 1 on (M, N) tensor.
    Steps:
        1. MUL_BROADCAST scalar=2.0 on each row         →  2x
        2. matmat_mul_core (M,N)@(N,N) sigmoid_enable   →  sigmoid(2x)
        3. MUL_BROADCAST scalar=2.0 on each row         →  2*sigmoid(2x)
        4. ADD_BROADCAST scalar=-1.0 on each row        →  tanh(x)
    """
    assert N % UE_VECTOR_SIZE == 0, f"N={N} must be a multiple of {UE_VECTOR_SIZE}"
    row_bytes = N * 2
    # Step 1: 2x → OUTPUT as temp
    for m in range(M):
        ue.accelerator_memory_to_sram(A_DRAM_ADDR + m * row_bytes, URAM_A_BASE, N)
        ue.broadcast_mul(scalar=2.0, sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=N)
        ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR + m * row_bytes, N)
    # Step 2: sigmoid(2x) via identity matmul with LALU sigmoid
    ue.matmat_mul_core(M=M, K=N, N=N,
                       A_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                       B_DRAM_ADDR=IDENTITY_DRAM_ADDR,
                       OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                       sigmoid_enable=True)
    # Steps 3+4: 2*sigmoid(2x) - 1
    for m in range(M):
        ue.accelerator_memory_to_sram(OUTPUT_DRAM_ADDR + m * row_bytes, URAM_A_BASE, N)
        ue.broadcast_mul(scalar=2.0, sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=N)
        ue.broadcast_add(scalar=-1.0, sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=N)
        ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR + m * row_bytes, N)
def silu_core_dram(ue: UnifiedEngine, M: int, N: int,
                   A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                   IDENTITY_DRAM_ADDR: int) -> None:
    """Emit instructions for standalone SiLU: output = x * sigmoid(x) on (M, N) tensor.
    Steps:
        1. Copy input to OUTPUT as temp (preserve x for step 3)
        2. matmat_mul_core (M,N)@I(N,N) sigmoid_enable on temp  → sigmoid(x)
        3. eltwise_mul row-by-row: x * sigmoid(x) → OUTPUT
    """
    assert N % UE_VECTOR_SIZE == 0, f"N={N} must be a multiple of {UE_VECTOR_SIZE}"
    total_elems = M * N
    # Step 1: bulk copy input to OUTPUT (so we can sigmoid in-place there)
    ue.accelerator_memory_to_sram(A_DRAM_ADDR, URAM_A_BASE, total_elems)
    ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR, total_elems)
    # Step 2: sigmoid(x) via identity matmul with LALU sigmoid → OUTPUT in-place
    ue.matmat_mul_core(M=M, K=N, N=N, A_DRAM_ADDR=OUTPUT_DRAM_ADDR, B_DRAM_ADDR=IDENTITY_DRAM_ADDR, OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR, sigmoid_enable=True)
    # Step 3: bulk x * sigmoid(x)
    ue.accelerator_memory_to_sram(A_DRAM_ADDR, URAM_A_BASE, total_elems)
    ue.accelerator_memory_to_sram(OUTPUT_DRAM_ADDR, URAM_B_BASE, total_elems)
    ue.eltwise_mul_core(vector_A_sram_start_addr=URAM_A_BASE, vector_B_sram_start_addr=URAM_B_BASE, vector_C_sram_wb_addr=URAM_A_BASE, element_size=total_elems)
    ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR, total_elems)
def glu_core_dram(ue: UnifiedEngine, M: int, C: int, A_DRAM_ADDR: int, B_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, IDENTITY_DRAM_ADDR: int) -> None:
    """Emit instructions for GLU: output = a * sigmoid(b).
    Assumes PW Conv1 was split into two matmuls writing a and b separately:
        PW Conv1a: (M, D) @ W_a(D, C) → a at A_DRAM_ADDR     (M, C)
        PW Conv1b: (M, D) @ W_b(D, C) → b at B_DRAM_ADDR     (M, C)
    Steps:
        1. matmat_mul_core (M,C)@I(C,C) sigmoid_enable on b  → sigmoid(b) in-place
        2. eltwise_mul row-by-row: a * sigmoid(b) → OUTPUT
    """
    assert C % UE_VECTOR_SIZE == 0, f"C={C} must be a multiple of {UE_VECTOR_SIZE}"
    total_elems = M * C
    # Step 1: sigmoid(b) via identity matmul
    ue.matmat_mul_core(M=M, K=C, N=C, A_DRAM_ADDR=B_DRAM_ADDR, B_DRAM_ADDR=IDENTITY_DRAM_ADDR, OUTPUT_DRAM_ADDR=B_DRAM_ADDR, sigmoid_enable=True)
    # Step 2: bulk a * sigmoid(b)
    ue.accelerator_memory_to_sram(A_DRAM_ADDR, URAM_A_BASE, total_elems)
    ue.accelerator_memory_to_sram(B_DRAM_ADDR, URAM_B_BASE, total_elems)
    ue.eltwise_mul_core(vector_A_sram_start_addr=URAM_A_BASE, vector_B_sram_start_addr=URAM_B_BASE, vector_C_sram_wb_addr=URAM_A_BASE, element_size=total_elems)
    ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR, total_elems)
def rel_shift_core_dram(ue: UnifiedEngine, L: int, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                        input_row_stride: int = None) -> None:
    """Emit instructions for rel_shift: extract (L, L) from (L, P_pad) positional scores.
    Row i of output = input[i, (L-1-i) : (L-1-i)+L]
    Each is a contiguous L-element DMA copy at a computed source offset.
    No arithmetic — pure memory rearrangement.
    Args:
        ue: UnifiedEngine instance
        L: sequence length (L_pad, bucketed, must be multiple of UE_VECTOR_SIZE)
        INPUT_DRAM_ADDR: (L, input_row_stride) bf16 positional score matrix
        OUTPUT_DRAM_ADDR: (L, L) bf16 output
        input_row_stride: actual number of elements per row in input (P_pad from matmul).
                          Defaults to 2*L-1 for backwards compatibility.
    """
    assert L % UE_VECTOR_SIZE == 0, f"L={L} must be a multiple of {UE_VECTOR_SIZE}"
    P_stride = input_row_stride if input_row_stride is not None else (2 * L - 1)
    bpe = 2          # bf16
    for i in range(L):
        src = INPUT_DRAM_ADDR + (i * P_stride + (L - 1 - i)) * bpe
        dst = OUTPUT_DRAM_ADDR + i * L * bpe
        ue.accelerator_memory_to_sram(src, URAM_A_BASE, L)
        ue.sram_to_accelerator_memory(URAM_A_BASE, dst, L)
def chunked_transpose_core_dram(ue: UnifiedEngine, M: int, N: int,
                                 input_dram_addr: int, output_dram_addr: int,
                                 identity_dram_addr: int, temp_dram_addr: int) -> None:
    """Transpose (M, N) -> (N, M) by processing N in UE_VECTOR_SIZE-column chunks.

    bf16_smart_permute_core's dot-product transpose accumulates across
    N_transpose // UE_VECTOR_SIZE groups, which corrupts results when
    N > UE_VECTOR_SIZE.  This helper splits into chunks where each sub-transpose
    has N_transpose = UE_VECTOR_SIZE (=64), avoiding the bug.

    Input at input_dram_addr: (M, N) contiguous bf16.
    Output at output_dram_addr: (N, M_aligned) contiguous bf16,
        where M_aligned = pad_to_multiple(M, UE_VECTOR_SIZE).
    """
    bpe = 2
    VS = UE_VECTOR_SIZE  # 64
    M_aligned = ((M - 1) // VS + 1) * VS
    n_chunks = (N + VS - 1) // VS

    for c in range(n_chunks):
        col_start = c * VS
        col_end = min(col_start + VS, N)
        chunk_cols = col_end - col_start
        chunk_cols_pad = ((chunk_cols - 1) // VS + 1) * VS  # = VS for full chunks

        # Extract columns [col_start:col_end] from each of M rows via strided DMA
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=input_dram_addr + col_start * bpe,
            sram_address=URAM_A_BASE,
            element_size=M * chunk_cols,
            stride_bytes_per_chunk=chunk_cols * bpe,
            stride_jump_bytes=N * bpe)
        ue.sram_to_accelerator_memory(
            sram_address=URAM_A_BASE,
            accelerator_dram_address=temp_dram_addr,
            element_size=M * chunk_cols_pad)

        # Transpose (M, chunk_cols_pad) -> (chunk_cols_pad, M_aligned)
        # N_transpose = chunk_cols_pad = VS, so dot product has 1 group. Safe.
        bf16_smart_permute_core(ue,
            dims=[M, chunk_cols_pad], permute_indices=[1, 0],
            input_dram_addr=temp_dram_addr,
            output_dram_addr=output_dram_addr + col_start * M_aligned * bpe,
            params_dram_addr=identity_dram_addr,
            temp_dram_start=temp_dram_addr + M * chunk_cols_pad * bpe)
def half_step_residual_core_dram(ue: UnifiedEngine, M: int, N: int,
                                 RESIDUAL_DRAM_ADDR: int, FF_DRAM_ADDR: int,
                                 OUTPUT_DRAM_ADDR: int) -> None:
    """Emit instructions for half-step residual: output = residual + 0.5 * ff_output.
    Args:
        ue: UnifiedEngine instance
        M: number of rows
        N: vector dimension (must be multiple of UE_VECTOR_SIZE)
        RESIDUAL_DRAM_ADDR: (M, N) bf16 — original input x
        FF_DRAM_ADDR: (M, N) bf16 — feed-forward output, modified in-place
        OUTPUT_DRAM_ADDR: (M, N) bf16 — result
    """
    assert N % UE_VECTOR_SIZE == 0, f"N={N} must be a multiple of {UE_VECTOR_SIZE}"
    total_elems = M * N
    # Load FF output, scale by 0.5
    ue.accelerator_memory_to_sram(FF_DRAM_ADDR, URAM_A_BASE, total_elems)
    ue.broadcast_mul(scalar=0.5, sram_start_addr=URAM_A_BASE,
                     sram_wb_addr=URAM_A_BASE, element_size=total_elems)
    # Load residual, add
    ue.accelerator_memory_to_sram(RESIDUAL_DRAM_ADDR, URAM_B_BASE, total_elems)
    ue.eltwise_add_core(vector_A_sram_start_addr=URAM_A_BASE,
                        vector_B_sram_start_addr=URAM_B_BASE,
                        vector_C_sram_wb_addr=URAM_A_BASE,
                        element_size=total_elems)
    # Write result
    ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR, total_elems)
def bf16_smart_permute_core(ue, dims, permute_indices, input_dram_addr, output_dram_addr,
                             params_dram_addr, temp_dram_start):
    """Permute via DMA gather + batched transpose decomposition."""
    from user_dma_core import (UE_VECTOR_SIZE, UE_MODE, URAM_FULL_ELEMENTS,
                               URAM_NEAR_FULL_ELEMENTS, URAM_HALF_ELEMENTS,
                               URAM_SECTION, URAM_WRITE_SRC, URAM_START_ADDR, LALU_MODE)
    n = len(dims) - 1
    bpe = 2
    total_elements = 1
    for d in dims:
        total_elements *= d
    k = permute_indices[n]
    inst_id = 0

    # Case 1: last dim stays fixed (P[n] == n) — pure DMA gather
    if k == n:
        last_dim = dims[n]
        output_shape = tuple(dims[permute_indices[i]] for i in range(len(dims)))
        permute_a = torch.arange(total_elements, dtype=torch.int32).reshape(*dims)
        permute_a = permute_a.permute(*permute_indices).contiguous().flatten()

        if last_dim < UE_VECTOR_SIZE:
            for j in range(total_elements // last_dim):
                src_idx = permute_a[j * last_dim].item()
                ue.ue_memcpy_from_dram(input_dram_addr + src_idx * bpe, last_dim * bpe, 0,
                    URAM_START_ADDR, URAM_SECTION.URAM_A.value, inst_id)
                ue.wait_queue(); inst_id += 1
                ue.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, URAM_START_ADDR,
                    output_dram_addr + j * last_dim * bpe, last_dim * bpe, inst_id)
                ue.wait_queue(); inst_id += 1
            return (1, output_shape)

        out_addr = output_dram_addr
        remaining = total_elements
        aligned = (URAM_NEAR_FULL_ELEMENTS // (UE_VECTOR_SIZE * last_dim)) * UE_VECTOR_SIZE * last_dim
        i = 0
        while remaining > 0:
            cur = min(aligned, remaining)
            n_blocks = cur // last_dim
            for j in range(n_blocks):
                src_idx = permute_a[i + j * last_dim].item()
                ue.ue_memcpy_from_dram(input_dram_addr + src_idx * bpe, last_dim * bpe, 0,
                    URAM_START_ADDR + (j * last_dim) // UE_VECTOR_SIZE,
                    URAM_SECTION.URAM_A.value, inst_id)
                ue.wait_queue(); inst_id += 1
            ue.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, URAM_START_ADDR,
                out_addr, n_blocks * last_dim * bpe, inst_id)
            ue.wait_queue(); inst_id += 1
            remaining -= cur; out_addr += n_blocks * last_dim * bpe; i += cur
        return (1, output_shape)

    # Case 2: last dim changes — Q1 + transpose + Q3
    remaining_for_q1 = [i for i in range(n + 1) if i != k and i != n]
    q1 = remaining_for_q1 + [k, n]
    q1_is_identity = all(q1[i] == i for i in range(n + 1))
    dims_after_q1 = [dims[q1[i]] for i in range(n + 1)]

    M_transpose = dims_after_q1[n - 1]
    N_transpose = dims_after_q1[n]
    M_aligned = ((M_transpose - 1) // UE_VECTOR_SIZE + 1) * UE_VECTOR_SIZE
    batch_size = 1
    for i in range(n - 1):
        batch_size *= dims_after_q1[i]
    dims_after_transpose = list(dims_after_q1[:n - 1]) + [N_transpose, M_aligned]

    current_dim_at_pos = list(q1[:n - 1]) + [n, k]
    pos_of_orig_dim = [0] * (n + 1)
    for pos, orig_dim in enumerate(current_dim_at_pos):
        pos_of_orig_dim[orig_dim] = pos
    q3 = [pos_of_orig_dim[permute_indices[i]] for i in range(n + 1)]
    q3_is_identity = all(q3[i] == i for i in range(n + 1))
    output_shape = tuple(dims_after_transpose[q3[i]] for i in range(n + 1))

    transposed_total = batch_size * N_transpose * M_aligned
    safe_temp = temp_dram_start
    if q1_is_identity:
        p2_in = input_dram_addr
    else:
        p2_in = safe_temp; safe_temp += total_elements * bpe
    if q3_is_identity:
        p2_out = output_dram_addr
    else:
        p2_out = safe_temp

    # Phase 1: Q1 permute (last-dim-fixed DMA gather)
    if not q1_is_identity:
        q1_pa = torch.arange(total_elements, dtype=torch.int32).reshape(*dims).permute(*q1).contiguous().flatten()
        last_dim = dims[n]
        out_addr = p2_in; remaining = total_elements
        aligned = (URAM_NEAR_FULL_ELEMENTS // (UE_VECTOR_SIZE * last_dim)) * UE_VECTOR_SIZE * last_dim
        i = 0
        while remaining > 0:
            cur = min(aligned, remaining); n_blocks = cur // last_dim
            for j in range(n_blocks):
                ue.ue_memcpy_from_dram(input_dram_addr + q1_pa[i + j * last_dim].item() * bpe,
                    last_dim * bpe, 0, URAM_START_ADDR + (j * last_dim) // UE_VECTOR_SIZE,
                    URAM_SECTION.URAM_A.value, inst_id)
                ue.wait_queue(); inst_id += 1
            ue.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, URAM_START_ADDR,
                out_addr, n_blocks * last_dim * bpe, inst_id)
            ue.wait_queue(); inst_id += 1
            remaining -= cur; out_addr += n_blocks * last_dim * bpe; i += cur

    # Phase 2: Batched transpose (identity dot-product)
    input_uram_addr = URAM_START_ADDR
    ue.ue_memcpy_from_dram(params_dram_addr, UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe,
        0, input_uram_addr, URAM_SECTION.URAM_A.value, inst_id)
    ue.wait_queue(); inst_id += 1

    max_N_chunk = min(((URAM_NEAR_FULL_ELEMENTS // N_transpose) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE, M_aligned)
    max_M_chunk = min(N_transpose, URAM_HALF_ELEMENTS // N_transpose, URAM_HALF_ELEMENTS // max_N_chunk)
    in_stride = M_transpose * N_transpose * bpe
    out_stride = M_aligned * N_transpose * bpe

    for batch in range(batch_size):
        cur_in = p2_in + batch * in_stride
        cur_out = p2_out + batch * out_stride
        remaining_M = N_transpose; start_vec = 0; out_chunk = cur_out

        while remaining_M > 0:
            cur_M = min(max_M_chunk, remaining_M)
            output_uram = UE_VECTOR_SIZE; remaining_N = M_aligned
            weight_addr = cur_in; out_offset = out_chunk

            while remaining_N > 0:
                cur_N = min(max_N_chunk, remaining_N)
                ue.ue_memcpy_from_dram(weight_addr, cur_N * N_transpose * bpe,
                    0, URAM_START_ADDR, URAM_SECTION.URAM_B.value, inst_id)
                ue.wait_queue(); inst_id += 1

                for i in range(cur_M):
                    abs_row = start_vec + i
                    vec_idx = abs_row % UE_VECTOR_SIZE
                    col_block = abs_row // UE_VECTOR_SIZE
                    ue.start_queue(0, 0, N_transpose // UE_VECTOR_SIZE, LALU_MODE.BYPASS.value, 0, 0,
                        URAM_SECTION.URAM_A.value, 0, 0, output_uram, URAM_WRITE_SRC.URAM_WRITE_BACK.value,
                        UE_MODE.BF16_DOT_PRODUCT, 0, input_uram_addr + vec_idx, URAM_START_ADDR + col_block,
                        1, 0, cur_N * N_transpose, cur_N, inst_id)
                    inst_id += 1; ue.wait_queue()
                    ue.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, output_uram,
                        out_offset + i * M_aligned * bpe, cur_N * bpe, inst_id)
                    ue.wait_queue(); inst_id += 1

                remaining_N -= cur_N; out_offset += cur_N * bpe; weight_addr += cur_N * N_transpose * bpe
            out_chunk += cur_M * M_aligned * bpe; remaining_M -= cur_M; start_vec += cur_M

    # Phase 3: Q3 permute (last-dim-fixed DMA gather)
    if not q3_is_identity:
        q3_pa = torch.arange(transposed_total, dtype=torch.int32).reshape(*dims_after_transpose).permute(*q3).contiguous().flatten()
        last_dim = M_aligned
        out_addr = output_dram_addr; remaining = transposed_total
        aligned = (URAM_NEAR_FULL_ELEMENTS // (UE_VECTOR_SIZE * last_dim)) * UE_VECTOR_SIZE * last_dim
        i = 0
        while remaining > 0:
            cur = min(aligned, remaining); n_blocks = cur // last_dim
            for j in range(n_blocks):
                ue.ue_memcpy_from_dram(p2_out + q3_pa[i + j * last_dim].item() * bpe,
                    last_dim * bpe, 0, URAM_START_ADDR + (j * last_dim) // UE_VECTOR_SIZE,
                    URAM_SECTION.URAM_A.value, inst_id)
                ue.wait_queue(); inst_id += 1
            ue.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, URAM_START_ADDR,
                out_addr, n_blocks * last_dim * bpe, inst_id)
            ue.wait_queue(); inst_id += 1
            remaining -= cur; out_addr += n_blocks * last_dim * bpe; i += cur

    return (2, output_shape)
# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def load_config(config_path=None):
    """Load parakeet_config.json."""
    if config_path is None:
        config_path = os.path.join(SCRIPT_DIR, "parakeet_config.json")
    with open(config_path, "r") as f:
        return json.load(f)
def pad_to_multiple(n, multiple):
    return ((n + multiple - 1) // multiple) * multiple
def conv2d_outsize(H, k=3, s=2, p=1):
    """Output spatial dim for conv2d/pool with given kernel, stride, padding."""
    return (H + 2 * p - k) // s + 1
def hw_vs_cpu(name, hw_tensor, ref_tensor, cos_threshold=0.99):
    """Compare HW result to CPU reference. Asserts on NaN or cosine failure."""
    h = hw_tensor.float().flatten()
    r = ref_tensor.float().flatten()
    assert not torch.isnan(h).any(), f"{name}: HW output contains NaN"
    assert not torch.isnan(r).any(), f"{name}: CPU reference contains NaN"
    n = min(len(h), len(r))
    cos = F.cosine_similarity(h[:n].unsqueeze(0), r[:n].unsqueeze(0)).item()
    mae = (h[:n] - r[:n]).abs().mean().item()
    max_err = (h[:n] - r[:n]).abs().max().item()
    print(f"  [{name}] cos={cos:.6f}  mae={mae:.6f}  max_err={max_err:.6f}  "
          f"hw_norm={h[:n].norm():.4f}  ref_norm={r[:n].norm():.4f}")
    assert cos >= cos_threshold, (
        f"MISMATCH {name}: cos={cos:.6f} < {cos_threshold}  "
        f"mae={mae:.6f}  max_err={max_err:.6f}")
def rel_shift(x):
    """Relative position shift (skew trick): (B, H, T, 2T-1) -> (B, H, T, T)."""
    B, H, T, P = x.shape
    zero_pad = torch.zeros(B, H, T, 1, device=x.device, dtype=x.dtype)
    x_padded = torch.cat([zero_pad, x], dim=-1)
    x_padded = x_padded.view(B, H, P + 1, T)
    x = x_padded[:, :, 1:].reshape(B, H, T, P)
    x = x[:, :, :, :P // 2 + 1]
    return x
def compile_and_run(engine, emit_fn):
    """Compile a block of HW instructions and execute it."""
    engine.clear_capture_buffer()
    engine.start_capture()
    engine.generate_instruction_flag_clear()
    emit_fn()
    engine.stop_capture()
    engine.generate_instruction_halt()
    prog = engine.get_program_dram_addr()
    engine.write_captured_instructions_to_dram(prog)
    engine.allocate_program_dram(engine.get_capture_instruction_size_bytes())
    engine.program_execute(prog)
def read_dram(engine, addr, numel):
    """Read bf16 tensor from accelerator DRAM."""
    buf = torch.zeros(numel, dtype=torch.bfloat16)
    engine.dma_read(DMA_DEVICE_C2H, addr, buf, numel * engine.bytes_per_element)
    return buf
# ---------------------------------------------------------------------------
# Host-side mel spectrogram (runs on CPU, not accelerator)
# ---------------------------------------------------------------------------
def compute_mel_spectrogram(waveform, cfg, ckpt_sd=None):
    """waveform: (B, samples) float32 → (B, T_mel, 128) bf16.

    Args:
        ckpt_sd: checkpoint state_dict — used to load the mel filterbank
                 and STFT window from 'preprocessor.featurizer.fb' and
                 'preprocessor.featurizer.window'.
    """
    pre = cfg["preprocessing"]
    n_fft = pre["n_fft"]
    hop_length = pre["hop_length"]
    win_length = pre["win_length"]
    if ckpt_sd is not None:
        fb = ckpt_sd["preprocessor.featurizer.fb"].float()       # (1, 128, 257)
        window = ckpt_sd["preprocessor.featurizer.window"].float()  # (400,)
    else:
        raise RuntimeError("Checkpoint state_dict required for mel filterbank and window")
    stft = torch.stft(waveform.float(), n_fft, hop_length, win_length,
                       window=window, center=True, pad_mode="reflect",
                       return_complex=True)
    mag = stft.abs()
    power = mag * mag
    mel = torch.matmul(fb, power)                      # (B, 128, T)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    mean = mel.mean(dim=-1, keepdim=True)
    var = mel.var(dim=-1, keepdim=True, unbiased=True)
    std = torch.clamp(torch.sqrt(var), min=1e-5)
    mel = (mel - mean) / std
    return mel.transpose(1, 2).to(torch.bfloat16)      # (B, T_mel, 128)
# ---------------------------------------------------------------------------
# Parakeet Unified Engine
# ---------------------------------------------------------------------------
# Parakeet DRAM partition over 4GB address space:
#   2.5 GB params / 750 MB tensors / 750 MB programs
PARAKEET_PARAMS_BASE  = 0x00000000   # 3 GB for weights + identities + Toeplitz DW conv matrices
PARAKEET_TENSOR_BASE  = 0xC0000000   # 768 MB for intermediate activations + im2col temp buffers
PARAKEET_PROGRAM_BASE = 0xF0000000   # 256 MB for compiled instruction programs
class Parakeet_UnifiedEngine(UnifiedEngine):
    """UnifiedEngine subclass for Parakeet-TDT-0.6B."""

    def __init__(self, script_dir=None, clock_period_ns=None):
        super().__init__(BASE_ADDR=user_dma_core.UE_0_BASE_ADDR, params_dram_base=PARAKEET_PARAMS_BASE, tensor_dram_base=PARAKEET_TENSOR_BASE, program_dram_base=PARAKEET_PROGRAM_BASE, clock_period_ns=clock_period_ns)
        self.script_dir = script_dir or SCRIPT_DIR
        # Hang prevention: stop stale execution, write HALT to program base
        self.dram_inst_running(False)
        self.start_capture()
        self.generate_instruction_halt()
        self.stop_capture()
        halt_bytes = bytearray()
        for inst in self.capture_buffer:
            halt_bytes.extend(inst.get_bytes())
        self.dma_write(DMA_DEVICE_H2C, PARAKEET_PROGRAM_BASE, halt_bytes, len(halt_bytes))
        self.clear_capture_buffer()
        self._cfg = load_config()
        enc = self._cfg["encoder"]
        pred = self._cfg["predictor"]
        jnt = self._cfg["joint"]
        hw = self._cfg["hardware"]
        self.d_model = enc["d_model"]           # 1024
        self.num_layers = enc["num_layers"]     # 24
        self.num_heads = enc["num_heads"]       # 8
        self.head_dim = enc["head_dim"]         # 128
        self.ff_dim = enc["ff_dim"]             # 4096
        self.conv_kernel = enc["conv_kernel"]   # 9
        self.conv_pad = enc["conv_pad"]         # 4
        self.sub_channels = enc["sub_channels"] # 256
        self.n_mels = enc["n_mels"]             # 128
        self.pred_hidden = pred["hidden_size"]  # 640
        self.vocab_size = pred["vocab_size"]    # 8193
        self.joint_hidden = jnt["hidden_size"]  # 640
        self.joint_output_padded = jnt["output_size_padded"]
        self.blank_id = jnt["blank_id"]
        self.tdt_durations = jnt["tdt_durations"]
        self.max_symbols_per_step = jnt["max_symbols_per_step"]
        self.block_size = hw["block_size"]      # 64
        self.bytes_per_element = enc["bytes_per_element"]
        # ISA register assignments for decoder dynamic addressing
        regs = self._cfg.get("fixed_isa_regs", {})
        self.TOKEN_REG = regs.get("TOKEN_REG", 1)
        self.ENC_T_REG = regs.get("ENC_T_REG", 2)
        self.TMP_REG = regs.get("TMP_REG", 3)
    def overwrite_instruction_with_general_register(self, general_register):
        """Modify the last captured instruction to use a general register for DRAM address."""
        if not self.capture_buffer or self.capture_count == 0:
            raise RuntimeError("capture_buffer is empty")
        if general_register <= 0 or general_register > 15:
            raise ValueError(f"general_register must be in [1, 15], got {general_register}")
        inst = self.capture_buffer[self.capture_count - 1]
        w = inst.words
        w[0] = ((0 & 0xF) << 0) | ((general_register & 0xF) << 4) | ((0 & 0xF) << 8) | ((0 & 0xF) << 12)
        w[7] = (w[7] & 0x1FFFFFFF) | ((user_dma_core.INSTRUCTION_REG_REWRITE & 0x7) << 29)
    def isa_add_set_core(self, dst_reg_idx, immediate_value, timeout_s=10.0):
        """Set one ISA register to an immediate value via minimal program execution."""
        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_add_set(dst_reg_idx, immediate_value)
        self.stop_capture()
        self.generate_instruction_halt()
        prog = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prog)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        self.clear_capture_buffer()
        self.start_execute_from_dram(prog)
        self.wait_queue(timeout_s)
    # XDMA driver has a transfer size limit (~16-64MB per os.write call).
    # Chunk large DMA uploads to avoid silent failures.
    DMA_CHUNK_BYTES = 1 * 1024 * 1024  # 1 MB per chunk
    def dma_to_accelerator_memory(self, dma_address, data):
        """Chunked DMA write that handles large transfers."""
        assert data.dtype == torch.bfloat16, "Data must be in bf16 format"
        flat = data.contiguous().flatten()
        total_bytes = flat.numel() * 2
        if total_bytes <= self.DMA_CHUNK_BYTES:
            self.dma_write(DMA_DEVICE_H2C, dma_address, flat, total_bytes)
            return
        elems_per_chunk = self.DMA_CHUNK_BYTES // 2
        offset = 0
        while offset < flat.numel():
            chunk_elems = min(elems_per_chunk, flat.numel() - offset)
            chunk = flat[offset:offset + chunk_elems].contiguous()
            addr = dma_address + offset * 2
            self.dma_write(DMA_DEVICE_H2C, addr, chunk, chunk_elems * 2)
            offset += chunk_elems
    def _alloc_write(self, tensor):
        """Allocate params DRAM, write bf16 tensor via chunked DMA. Returns DRAM address."""
        t = tensor.to(torch.bfloat16).contiguous()
        addr = self.get_params_dram_addr()
        nbytes = t.numel() * 2
        self.allocate_params_dram(nbytes)
        self.dma_to_accelerator_memory(addr, t)
        return addr
    def _stage_dw_conv1d(self, weight):
        """(C,1,K) -> (C,64) im2col padded."""
        C, _, K = weight.shape
        padded_K = pad_to_multiple(K, self.block_size)
        w_flat = weight.reshape(C, K).to(torch.bfloat16)
        w_padded = torch.zeros(C, padded_K, dtype=torch.bfloat16)
        w_padded[:, :K] = w_flat
        return self._alloc_write(w_padded)
    def _split_pw_conv1(self, weight):
        """(2D,D,1) -> (addr_a, addr_b) each (D,D)."""
        w = weight.squeeze(-1).to(torch.bfloat16)
        D = w.shape[0] // 2
        return self._alloc_write(w[:D].contiguous()), self._alloc_write(w[D:].contiguous())
    def _stage_sub_conv2d(self, weight, out_ch, in_ch, k=3):
        """Stage 2D conv weight for im2col: (out_ch, in_ch, k, k) -> (out_ch, k*k*in_ch padded to 64)."""
        w = weight.to(torch.bfloat16).reshape(out_ch, -1)
        flat_k = w.shape[1]
        padded = pad_to_multiple(flat_k, self.block_size)
        w_pad = torch.zeros(out_ch, padded, dtype=torch.bfloat16)
        w_pad[:, :flat_k] = w
        return self._alloc_write(w_pad), flat_k, padded
    @staticmethod
    def ensure_model_files():
        """Download Parakeet-TDT-0.6B from HuggingFace if not present."""
        if os.path.exists(WEIGHTS_PATH) and os.path.exists(TOKENIZER_PATH):
            return
        model_dir = os.path.dirname(WEIGHTS_PATH)
        os.makedirs(model_dir, exist_ok=True)
        print(f"Model files not found, downloading to {model_dir} ...")
        from huggingface_hub import hf_hub_download
        import tarfile
        nemo_path = hf_hub_download(
            repo_id="nvidia/parakeet-tdt-0.6b-v3",
            filename="parakeet-tdt-0.6b-v3.nemo",
            cache_dir=model_dir)
        print(f"  Extracting from {nemo_path} ...")
        with tarfile.open(nemo_path, "r") as tar:
            for member in tar.getmembers():
                if member.name.endswith("model_weights.ckpt"):
                    member.name = "model_weights.ckpt"
                    tar.extract(member, model_dir)
                elif "tokenizer.model" in member.name:
                    member.name = "tokenizer.model"
                    tar.extract(member, model_dir)
        if not os.path.exists(WEIGHTS_PATH):
            sys.exit(f"Failed to extract model_weights.ckpt")
        if not os.path.exists(TOKENIZER_PATH):
            sys.exit(f"Failed to extract tokenizer.model")
        print("  Download complete.")
    def weight_init(self):
        """Load checkpoint, stage all weights to DRAM."""
        self.ensure_model_files()
        ckpt_path = WEIGHTS_PATH
        print(f"Loading weights from {ckpt_path} ...")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        print(f"  {sum(v.numel() for v in sd.values() if hasattr(v, 'numel')):,} parameters")
        D, FF, H = self.d_model, self.ff_dim, self.pred_hidden
        self.w = {}

        # Identity matrices for LALU activation (sigmoid, softmax, etc.)
        self.w["IDENTITY_1024"] = allocate_identity(self, D)
        self.w["IDENTITY_4096"] = allocate_identity(self, FF)
        self.w["IDENTITY_640"] = allocate_identity(self, self.joint_hidden)
        # 64x64 identity for bf16_smart_permute_core transpose operations.
        self.w["IDENTITY_64"] = allocate_identity(self, UE_VECTOR_SIZE)

        # Subsampling weights for hardware im2col
        # Stage 0: Conv2d(1->256, k=3, s=2, p=1)
        self.w["SUB_CONV0_W"], self.sub0_flat_k, self.sub0_padded_k = \
            self._stage_sub_conv2d(sd["encoder.pre_encode.conv.0.weight"], 256, 1, 3)
        self.w["SUB_CONV0_B"] = self._alloc_write(sd["encoder.pre_encode.conv.0.bias"])
        # Im2col gathering matrices G: stored as (N=W_out*64, K=3*W_in) for B convention.
        # G[ow*64 + kh*3 + kw, kh*W_in + col] = 1 where col = ow*2 - 1 + kw (if valid).
        def _build_im2col_G(W_in, W_out):
            K_g = pad_to_multiple(3 * W_in, self.block_size)  # padded for matmul K alignment
            N_g = W_out * 64
            G = torch.zeros(N_g, K_g, dtype=torch.bfloat16)
            for kh in range(3):
                for kw in range(3):
                    for ow in range(W_out):
                        col = ow * 2 - 1 + kw
                        if 0 <= col < W_in:
                            G[ow * 64 + kh * 3 + kw, kh * W_in + col] = 1.0
            return G
        W_out_s0 = (self.n_mels + 2 - 3) // 2 + 1  # 64
        G_s0 = _build_im2col_G(self.n_mels, W_out_s0)
        self.w["IM2COL_G_S0"] = self._alloc_write(G_s0)
        print(f"  im2col G_s0: {tuple(G_s0.shape)} = {G_s0.numel()*2/1024:.0f} KB")
        W0_sub = (self.n_mels + 2 - 3) // 2 + 1  # 64 (= W_out of stage 0 = W_in of stage 1)
        W1_sub = (W0_sub + 2 - 3) // 2 + 1       # 32
        W2_sub = (W1_sub + 2 - 3) // 2 + 1        # 16
        G_s1 = _build_im2col_G(W0_sub, W1_sub)
        self.w["IM2COL_G_S1"] = self._alloc_write(G_s1)
        print(f"  im2col G_s1: {tuple(G_s1.shape)} = {G_s1.numel()*2/1024:.0f} KB")
        G_s2 = _build_im2col_G(W1_sub, W2_sub)
        self.w["IM2COL_G_S2"] = self._alloc_write(G_s2)
        print(f"  im2col G_s2: {tuple(G_s2.shape)} = {G_s2.numel()*2/1024:.0f} KB")
        # Stage 1: DW Conv2d(256, k=3, s=2, p=1) + PW Conv2d(256->256)
        self.w["SUB_DW1_W"] = self._stage_dw_conv1d(
            sd["encoder.pre_encode.conv.2.weight"].reshape(256, 1, 9))
        self.w["SUB_DW1_B"] = self._alloc_write(sd["encoder.pre_encode.conv.2.bias"])
        self.w["SUB_PW1_W"] = self._alloc_write(sd["encoder.pre_encode.conv.3.weight"].reshape(256, 256))
        self.w["SUB_PW1_B"] = self._alloc_write(sd["encoder.pre_encode.conv.3.bias"])
        # Stage 2: DW Conv2d(256, k=3, s=2, p=1) + PW Conv2d(256->256)
        self.w["SUB_DW2_W"] = self._stage_dw_conv1d(
            sd["encoder.pre_encode.conv.5.weight"].reshape(256, 1, 9))
        self.w["SUB_DW2_B"] = self._alloc_write(sd["encoder.pre_encode.conv.5.bias"])
        self.w["SUB_PW2_W"] = self._alloc_write(sd["encoder.pre_encode.conv.6.weight"].reshape(256, 256))
        self.w["SUB_PW2_B"] = self._alloc_write(sd["encoder.pre_encode.conv.6.bias"])
        # Final linear — original weight and permuted version for direct read from (N, SC) layout
        W_out_orig = sd["encoder.pre_encode.out.weight"]  # (D=1024, SC*W2=4096)
        self.w["SUB_OUT_W"] = self._alloc_write(W_out_orig)
        self.w["SUB_OUT_B"] = self._alloc_write(sd["encoder.pre_encode.out.bias"])
        # Permuted weight: absorbs the (H2,W2,SC)->(H2,SC,W2) flatten permute into the weight.
        # Original expects input ordered as [c*W2+w], permuted expects [w*SC+c].
        # W_perm[d, w*SC+c] = W_orig[d, c*W2+w]
        W2_sub = (W1_sub + 2 - 3) // 2 + 1  # 16
        W_perm = W_out_orig.reshape(self.d_model, self.sub_channels, W2_sub).permute(0, 2, 1).reshape(self.d_model, -1).contiguous()
        self.w["SUB_OUT_W_PERM"] = self._alloc_write(W_perm)

        # Per-layer conformer weights
        self.layer_addrs = []
        for i in range(self.num_layers):
            la = {}
            pfx = f"encoder.layers.{i}"
            ln_map = {"LN_FF1": "norm_feed_forward1", "LN_ATTN": "norm_self_att",
                      "LN_CONV": "norm_conv", "LN_FF2": "norm_feed_forward2", "LN_OUT": "norm_out"}
            for our_key, nemo_key in ln_map.items():
                la[f"{our_key}_WEIGHT"] = self._alloc_write(sd[f"{pfx}.{nemo_key}.weight"])
                la[f"{our_key}_BIAS"] = self._alloc_write(sd[f"{pfx}.{nemo_key}.bias"])
            w1 = sd[f"{pfx}.feed_forward1.linear1.weight"]  # (4096, 1024)
            la["FF1_W1_LO"] = self._alloc_write(w1[:FF//2, :].contiguous())
            la["FF1_W1_HI"] = self._alloc_write(w1[FF//2:, :].contiguous())
            w2 = sd[f"{pfx}.feed_forward1.linear2.weight"]  # (1024, 4096)
            la["FF1_W2_LO"] = self._alloc_write(w2[:, :FF//2].contiguous())
            la["FF1_W2_HI"] = self._alloc_write(w2[:, FF//2:].contiguous())
            w1 = sd[f"{pfx}.feed_forward2.linear1.weight"]
            la["FF2_W1_LO"] = self._alloc_write(w1[:FF//2, :].contiguous())
            la["FF2_W1_HI"] = self._alloc_write(w1[FF//2:, :].contiguous())
            w2 = sd[f"{pfx}.feed_forward2.linear2.weight"]
            la["FF2_W2_LO"] = self._alloc_write(w2[:, :FF//2].contiguous())
            la["FF2_W2_HI"] = self._alloc_write(w2[:, FF//2:].contiguous())
            la["ATTN_Q_W"] = self._alloc_write(sd[f"{pfx}.self_attn.linear_q.weight"])
            la["ATTN_K_W"] = self._alloc_write(sd[f"{pfx}.self_attn.linear_k.weight"])
            la["ATTN_V_W"] = self._alloc_write(sd[f"{pfx}.self_attn.linear_v.weight"])
            la["ATTN_POS_W"] = self._alloc_write(sd[f"{pfx}.self_attn.linear_pos.weight"])
            la["ATTN_OUT_W"] = self._alloc_write(sd[f"{pfx}.self_attn.linear_out.weight"])
            pw1_a, pw1_b = self._split_pw_conv1(sd[f"{pfx}.conv.pointwise_conv1.weight"])
            la["CONV_PW1A_W"], la["CONV_PW1B_W"] = pw1_a, pw1_b
            la["CONV_PW2_W"] = self._alloc_write(sd[f"{pfx}.conv.pointwise_conv2.weight"].squeeze(-1))
            la["ATTN_BIAS_U"] = self._alloc_write(sd[f"{pfx}.self_attn.pos_bias_u"].reshape(-1))
            la["ATTN_BIAS_V"] = self._alloc_write(sd[f"{pfx}.self_attn.pos_bias_v"].reshape(-1))
            la["CONV_DW_W"] = self._stage_dw_conv1d(sd[f"{pfx}.conv.depthwise_conv.weight"])
            bn_w = sd[f"{pfx}.conv.batch_norm.weight"]
            bn_b = sd[f"{pfx}.conv.batch_norm.bias"]
            bn_m = sd[f"{pfx}.conv.batch_norm.running_mean"]
            bn_v = sd[f"{pfx}.conv.batch_norm.running_var"]
            s_addr, sh_addr, _ = batch_norm_fuse_params(self, bn_w, bn_b, bn_m, bn_v)
            la["CONV_BN_SCALE"], la["CONV_BN_SHIFT"] = s_addr, sh_addr
            self.layer_addrs.append(la)
            print(f"\r  Layer {i:2d}/{self.num_layers - 1} staged", end="", flush=True)

        print()
        # Predictor weights
        self.w["EMBED"] = self._alloc_write(sd["decoder.prediction.embed.weight"])
        for i in range(2):
            self.w[f"LSTM_WIH{i}"] = self._alloc_write(sd[f"decoder.prediction.dec_rnn.lstm.weight_ih_l{i}"])
            self.w[f"LSTM_WHH{i}"] = self._alloc_write(sd[f"decoder.prediction.dec_rnn.lstm.weight_hh_l{i}"])
            self.w[f"LSTM_BIH{i}"] = self._alloc_write(sd[f"decoder.prediction.dec_rnn.lstm.bias_ih_l{i}"])
            self.w[f"LSTM_BHH{i}"] = self._alloc_write(sd[f"decoder.prediction.dec_rnn.lstm.bias_hh_l{i}"])

        # Joint weights
        self.w["JOINT_ENC_W"] = self._alloc_write(sd["joint.enc.weight"])
        self.w["JOINT_ENC_B"] = self._alloc_write(sd["joint.enc.bias"])
        self.w["JOINT_PRED_W"] = self._alloc_write(sd["joint.pred.weight"])
        self.w["JOINT_PRED_B"] = self._alloc_write(sd["joint.pred.bias"])

        # Split joint output: token (8256x640) + duration (64x640)
        out_w = sd["joint.joint_net.2.weight"].to(torch.bfloat16)
        out_b = sd["joint.joint_net.2.bias"].to(torch.bfloat16)
        N_tok, N_tok_pad = self.vocab_size, pad_to_multiple(self.vocab_size, self.block_size)
        N_dur, N_dur_pad = len(self.tdt_durations), pad_to_multiple(len(self.tdt_durations), self.block_size)
        w_tok = torch.zeros(N_tok_pad, self.joint_hidden, dtype=torch.bfloat16)
        w_tok[:N_tok] = out_w[:N_tok]
        # Padding biases must be -inf so padding positions never win argmax.
        b_tok = torch.full((N_tok_pad,), -1e4, dtype=torch.bfloat16)
        b_tok[:N_tok] = out_b[:N_tok]
        self.w["JOINT_OUT_TOK_W"] = self._alloc_write(w_tok)
        self.w["JOINT_OUT_TOK_B"] = self._alloc_write(b_tok)
        w_dur = torch.zeros(N_dur_pad, self.joint_hidden, dtype=torch.bfloat16)
        w_dur[:N_dur] = out_w[N_tok:N_tok + N_dur]
        b_dur = torch.full((N_dur_pad,), -1e4, dtype=torch.bfloat16)
        b_dur[:N_dur] = out_b[N_tok:N_tok + N_dur]
        self.w["JOINT_OUT_DUR_W"] = self._alloc_write(w_dur)
        self.w["JOINT_OUT_DUR_B"] = self._alloc_write(b_dur)
        # Store checkpoint for mel spectrogram (filterbank + window)
        self._ckpt_sd = sd
        params_used = self.get_params_dram_usage()
        params_limit = self._tensor_dram_base - self._params_dram_base
        print(f"  Weight staging complete: {params_used / 1024**2:.1f} MB in DRAM "
              f"(budget: {params_limit / 1024**2:.0f} MB)")
        assert params_used <= params_limit, (
            f"PARAMS DRAM OVERFLOW: {params_used/1024**2:.1f} MB used > "
            f"{params_limit/1024**2:.0f} MB budget. "
            f"Params bleed into tensor region — corrupts activations!"
        )
    def tensor_init(self, L_pad):
        """Allocate intermediate DRAM buffers."""
        # Zero entire tensor DRAM region up front to prevent stale NaN.
        tensor_budget = self._program_dram_base - self._tensor_dram_base
        print(f"  Zeroing tensor DRAM ({tensor_budget / 1024**2:.0f} MB budget)...")
        ZERO_CHUNK = 512 * 1024  # 512K elements = 1MB per chunk
        offset = 0
        total_elems = tensor_budget // 2
        while offset < total_elems:
            chunk_elems = min(ZERO_CHUNK, total_elems - offset)
            z = torch.zeros(chunk_elems, dtype=torch.bfloat16)
            self.dma_to_accelerator_memory(self._tensor_dram_base + offset * 2, z)
            offset += chunk_elems
        print(f"  Tensor DRAM zeroed.")

        bpe = self.bytes_per_element
        D, FF, H = self.d_model, self.ff_dim, self.pred_hidden
        dk = self.head_dim
        SC = self.sub_channels  # 256
        P_pad = pad_to_multiple(2 * L_pad - 1, self.block_size)
        N_tok_pad = pad_to_multiple(self.vocab_size, self.block_size)
        N_dur_pad = pad_to_multiple(len(self.tdt_durations), self.block_size)
        # Encoder intermediates
        self.INPUT_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.LN_OUT_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.FF_MID_DRAM = self.allocate_tensor_dram(L_pad * FF * bpe)
        self.FF_OUT_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.RESIDUAL_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.Q_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.K_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.V_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.POS_PROJ_DRAM = self.allocate_tensor_dram(P_pad * D * bpe)  # P_pad rows, not L_pad
        self.SCORE_DRAM = self.allocate_tensor_dram(L_pad * L_pad * bpe)
        self.ATTN_OUT_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.POS_EMB_DRAM = self.allocate_tensor_dram(P_pad * D * bpe)
        self.REL_SHIFT_DRAM = self.allocate_tensor_dram(L_pad * L_pad * bpe)
        self.CONV_A_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.CONV_B_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.CONV_T_DRAM = self.allocate_tensor_dram(D * L_pad * bpe)
        self.CONV_DW_DRAM = self.allocate_tensor_dram(D * L_pad * bpe)
        self.CONV_OUT_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.ATTN_VT_DRAM = self.allocate_tensor_dram(dk * L_pad * bpe)
        self.ATTN_MASK_DRAM = self.allocate_tensor_dram(L_pad * L_pad * bpe)
        self.PERMUTE_TEMP_DRAM = self.allocate_tensor_dram(D * L_pad * bpe)
        self.ENC_OUT_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        # L_pad-sized identity for softmax and SiLU in conv module
        self.IDENTITY_LPAD_DRAM = allocate_identity(self, L_pad)
        # Subsampling intermediates
        T_mel_max = L_pad * 8
        self.MEL_DRAM = self.allocate_tensor_dram(T_mel_max * self.n_mels * bpe)
        # Temp buffer for R_combined (stage 0 im2col row selection)
        H0_max = (T_mel_max + 2 - 3) // 2 + 1
        self.IM2COL_R_DRAM = self.allocate_tensor_dram(H0_max * 3 * self.n_mels * bpe)
        H0, W0 = T_mel_max // 2, self.n_mels // 2  # after stage 0
        H1, W1 = H0 // 2, W0 // 2                  # after stage 1
        N0 = H0 * W0
        N1 = H1 * W1
        N1_pad = pad_to_multiple(N1, self.block_size)
        H2, W2_tmp = H1 // 2, W1 // 2
        N2 = H2 * W2_tmp
        N2_pad = pad_to_multiple(N2, self.block_size)
        N_dw_max_pad = max(N1_pad, N2_pad)
        max_patch_size = max(N0 * 64, SC * N_dw_max_pad * 64)
        self.SUB_PATCH_DRAM = self.allocate_tensor_dram(max_patch_size * bpe)
        # Temp buffers for depthwise im2col (stages 1, 2)
        max_N_dw_in = max(N0, N1_pad)  # max spatial positions for DW input
        max_M_aligned = pad_to_multiple(max_N_dw_in, self.block_size)
        self.IM2COL_TRANSPOSE_DRAM = self.allocate_tensor_dram(max_M_aligned * SC * bpe)
        self.IM2COL_PERMUTE_TEMP_DRAM = self.allocate_tensor_dram(
            SC * pad_to_multiple(max_N_dw_in, self.block_size) * bpe)
        # R_all: SC * H_out_pad * K_g per DW stage (K_g = pad(3*W_in, 64))
        H1_pad_max = N1_pad // W1 if W1 > 0 else 0
        H2_pad_max = N2_pad // W2_tmp if W2_tmp > 0 else 0
        K_g_s1 = pad_to_multiple(3 * W0, self.block_size)
        K_g_s2 = pad_to_multiple(3 * W1, self.block_size)
        r_all_max = max(SC * H1_pad_max * K_g_s1, SC * H2_pad_max * K_g_s2) if W1 > 0 else 0
        self.IM2COL_R_DW_DRAM = self.allocate_tensor_dram(r_all_max * bpe)
        self.SUB_OUT0_DRAM = self.allocate_tensor_dram(N0 * SC * bpe)
        self.SUB_DW_OUT_DRAM = self.allocate_tensor_dram(SC * N_dw_max_pad * bpe)
        self.SUB_PW_IN_DRAM = self.allocate_tensor_dram(N_dw_max_pad * SC * bpe)
        self.SUB_FLAT_DRAM = self.allocate_tensor_dram(L_pad * 4096 * bpe)
        # Decoder intermediates
        self.PRED_EMB_DRAM = self.allocate_tensor_dram(H * bpe)
        self.PRED_GATES_DRAM = self.allocate_tensor_dram(4 * H * bpe)
        self.PRED_GATES2_DRAM = self.allocate_tensor_dram(4 * H * bpe)
        self.PRED_H0_DRAM = self.allocate_tensor_dram(H * bpe)
        self.PRED_C0_DRAM = self.allocate_tensor_dram(H * bpe)
        self.PRED_H1_DRAM = self.allocate_tensor_dram(H * bpe)
        self.PRED_C1_DRAM = self.allocate_tensor_dram(H * bpe)
        self.PRED_OUT_DRAM = self.allocate_tensor_dram(H * bpe)
        # On-device LSTM state save buffers (for blank rollback without host DMA)
        self.SAVED_H0_DRAM = self.allocate_tensor_dram(H * bpe)
        self.SAVED_C0_DRAM = self.allocate_tensor_dram(H * bpe)
        self.SAVED_H1_DRAM = self.allocate_tensor_dram(H * bpe)
        self.SAVED_C1_DRAM = self.allocate_tensor_dram(H * bpe)
        self.JOINT_ENC_DRAM = self.allocate_tensor_dram(D * bpe)
        self.JOINT_PRED_DRAM = self.allocate_tensor_dram(self.joint_hidden * bpe)
        self.JOINT_SUM_DRAM = self.allocate_tensor_dram(self.joint_hidden * bpe)
        self.JOINT_TOK_DRAM = self.allocate_tensor_dram(N_tok_pad * bpe)
        self.JOINT_DUR_DRAM = self.allocate_tensor_dram(N_dur_pad * bpe)
        tensor_used = self.get_tensor_dram_usage()
        tensor_limit = self._program_dram_base - self._tensor_dram_base
        print(f"  Tensor alloc: {tensor_used / 1024**2:.1f} MB "
              f"(budget: {tensor_limit / 1024**2:.0f} MB)")
        assert tensor_used <= tensor_limit, (
            f"TENSOR DRAM OVERFLOW: {tensor_used/1024**2:.1f} MB used > "
            f"{tensor_limit/1024**2:.0f} MB budget. "
            f"Tensors bleed into program region!"
        )
    def compile_im2col_s0(self, T_mel):
        """Compile accelerator program for stage-0 im2col via permutation-matrix matmul.

        Builds R_combined(H_out, 3*W_in) in IM2COL_R_DRAM via strided DMA (row
        selection from MEL_DRAM), then matmuls R @ G → (N0, 64) at SUB_PATCH_DRAM.

        Must be called after mel is uploaded to MEL_DRAM.
        Returns (program_addr, H_out, W_out).
        """
        from user_dma_core import URAM_START_ADDR, URAM_SECTION, URAM_NEAR_FULL_SIZE
        bpe = 2
        W_in = self.n_mels  # 128
        stride = 2
        padding = 1
        H_out = (T_mel + 2 * padding - 3) // stride + 1
        W_out = (W_in + 2 * padding - 3) // stride + 1
        N0 = H_out * W_out
        K_g = pad_to_multiple(3 * W_in, self.block_size)  # 384 (already aligned)
        N_g = W_out * 64      # 4096
        row_bytes = W_in * bpe  # 256 bytes per mel row — well above 32-byte AXI min

        # Pre-zero R_combined (padding rows stay zero)
        self.dma_to_accelerator_memory(self.IM2COL_R_DRAM,
            torch.zeros(H_out * K_g, dtype=torch.bfloat16))

        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()
        inst_id = 0

        # Build R_combined(H_out, 3*W_in) via strided DMA.
        # Row oh = [mel_row[oh*2-1] | mel_row[oh*2] | mel_row[oh*2+1]]
        for kh in range(3):
            # First valid oh where input row oh*2 - 1 + kh >= 0
            oh_start = max(0, (padding - kh + stride - 1) // stride)
            # Last valid oh (exclusive) where input row < T_mel
            oh_end = min(H_out, (T_mel + padding - kh + stride - 1) // stride)
            n_rows = oh_end - oh_start
            if n_rows <= 0:
                continue

            first_input_row = oh_start * stride - padding + kh
            read_bytes = n_rows * row_bytes

            # Chunk to fit URAM (contiguous packing of n_rows * W_in elements)
            max_read = (URAM_NEAR_FULL_SIZE // row_bytes) * row_bytes
            offset = 0
            while offset < read_bytes:
                chunk = min(read_bytes - offset, max_read)
                rows_in_chunk = chunk // row_bytes
                src_row = first_input_row + (offset // row_bytes) * stride
                oh_base = oh_start + offset // row_bytes

                # Strided read: gather every-other mel row into URAM
                src = self.MEL_DRAM + src_row * W_in * bpe
                self.ue_memcpy_from_dram(
                    src, chunk, 0, URAM_START_ADDR,
                    URAM_SECTION.URAM_A.value, inst_id,
                    stride_bytes_per_chunk=row_bytes,
                    stride_jump_bytes=stride * row_bytes)
                self.wait_queue(); inst_id += 1

                # Strided write: scatter into kh-th column block of R_combined
                dst = (self.IM2COL_R_DRAM
                       + oh_base * K_g * bpe
                       + kh * W_in * bpe)
                self.ue_memcpy_to_dram(
                    0, URAM_SECTION.URAM_A.value, URAM_START_ADDR,
                    dst, chunk, inst_id,
                    stride_bytes_per_chunk=row_bytes,
                    stride_jump_bytes=K_g * bpe)
                self.wait_queue(); inst_id += 1

                offset += chunk

        # Matmul: R_combined(H_out, K_g) @ G(K_g, N_g) → (H_out, N_g) = (N0, 64)
        self.matmat_mul_core(M=H_out, K=K_g, N=N_g,
            A_DRAM_ADDR=self.IM2COL_R_DRAM,
            B_DRAM_ADDR=self.w["IM2COL_G_S0"],
            OUTPUT_DRAM_ADDR=self.SUB_PATCH_DRAM)

        self.stop_capture()
        self.generate_instruction_halt()
        prog = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prog)
        prog_bytes = self.get_capture_instruction_size_bytes()
        self.allocate_program_dram(prog_bytes)
        return prog, H_out, W_out
    def compile_im2col_dw(self, H_in, W_in, N_out_pad, input_dram_addr, g_key):
        """Compile accelerator program for depthwise im2col via transpose + permutation-matrix matmul.

        Input: (H_in * W_in, SC) at input_dram_addr (spatial-major, SC=256 channels interleaved).
        Output: (SC, N_out_pad, 64) at SUB_PATCH_DRAM.

        Steps compiled into one program:
        1. Transpose (N_in, SC) → (SC, N_in) via bf16_smart_permute_core
        2. Build R_all (SC * H_out_pad, 3 * W_in) via strided DMA (per-channel row selection)
        3. Matmul: R_all @ G → (SC * H_out_pad, W_out * 64) = (SC, N_out_pad, 64)

        Returns (program_addr, H_out, W_out).
        """
        from user_dma_core import URAM_START_ADDR, URAM_SECTION, URAM_NEAR_FULL_SIZE
        bpe = 2
        SC = self.sub_channels  # 256
        stride = 2
        padding = 1
        N_in = H_in * W_in
        H_out = (H_in + 2 * padding - 3) // stride + 1
        W_out = (W_in + 2 * padding - 3) // stride + 1
        N_out = H_out * W_out
        # Pad H_out so H_out_pad * W_out == N_out_pad (keeps (SC, N_out_pad, 64) layout aligned)
        H_out_pad = N_out_pad // W_out
        K_g = pad_to_multiple(3 * W_in, self.block_size)  # must be multiple of 64 for matmul
        N_g = W_out * 64
        row_bytes = W_in * bpe  # bytes per spatial row (one channel)

        # Pre-zero R_all (padding rows stay zero — extra K_g columns beyond 3*W_in also zero)
        self.dma_to_accelerator_memory(self.IM2COL_R_DW_DRAM,
            torch.zeros(SC * H_out_pad * K_g, dtype=torch.bfloat16))

        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()

        # Step 1: Transpose (N_in, SC) → (SC, N_in)
        bf16_smart_permute_core(self,
            dims=[N_in, SC],
            permute_indices=[1, 0],
            input_dram_addr=input_dram_addr,
            output_dram_addr=self.IM2COL_TRANSPOSE_DRAM,
            params_dram_addr=self.w["IDENTITY_64"],
            temp_dram_start=self.IM2COL_PERMUTE_TEMP_DRAM)

        # Step 2: Build R_all (SC * H_out_pad, 3 * W_in) via strided DMA.
        # After transpose, the output is (SC, M_aligned) where M_aligned = pad(N_in, 64).
        # Channel c starts at IM2COL_TRANSPOSE_DRAM + c * M_aligned * bpe.
        M_aligned = pad_to_multiple(N_in, self.block_size)
        uram_row_bytes = UE_VECTOR_SIZE * bpe  # 128 bytes — minimum for strided DMA
        use_stride = (row_bytes >= uram_row_bytes)
        inst_id = 0
        max_read = (URAM_NEAR_FULL_SIZE // row_bytes) * row_bytes
        for c in range(SC):
            ch_base = self.IM2COL_TRANSPOSE_DRAM + c * M_aligned * bpe
            r_base = self.IM2COL_R_DW_DRAM + c * H_out_pad * K_g * bpe
            for kh in range(3):
                oh_start = max(0, (padding - kh + stride - 1) // stride)
                oh_end = min(H_out, (H_in + padding - kh + stride - 1) // stride)
                n_rows = oh_end - oh_start
                if n_rows <= 0:
                    continue
                first_input_row = oh_start * stride - padding + kh

                if use_stride:
                    # Batched strided DMA (row_bytes >= 1 URAM row)
                    read_bytes = n_rows * row_bytes
                    offset = 0
                    while offset < read_bytes:
                        chunk = min(read_bytes - offset, max_read)
                        src_row = first_input_row + (offset // row_bytes) * stride
                        oh_base = oh_start + offset // row_bytes
                        src = ch_base + src_row * W_in * bpe
                        self.ue_memcpy_from_dram(
                            src, chunk, 0, URAM_START_ADDR,
                            URAM_SECTION.URAM_A.value, inst_id,
                            stride_bytes_per_chunk=row_bytes,
                            stride_jump_bytes=stride * row_bytes)
                        self.wait_queue(); inst_id += 1
                        dst = r_base + oh_base * K_g * bpe + kh * W_in * bpe
                        self.ue_memcpy_to_dram(
                            0, URAM_SECTION.URAM_A.value, URAM_START_ADDR,
                            dst, chunk, inst_id,
                            stride_bytes_per_chunk=row_bytes,
                            stride_jump_bytes=K_g * bpe)
                        self.wait_queue(); inst_id += 1
                        offset += chunk
                else:
                    # Individual row DMA (row_bytes < 1 URAM row)
                    for j in range(n_rows):
                        src_row = first_input_row + j * stride
                        src = ch_base + src_row * W_in * bpe
                        self.ue_memcpy_from_dram(
                            src, row_bytes, 0, URAM_START_ADDR,
                            URAM_SECTION.URAM_A.value, inst_id)
                        self.wait_queue(); inst_id += 1
                        dst = r_base + (oh_start + j) * K_g * bpe + kh * W_in * bpe
                        self.ue_memcpy_to_dram(
                            0, URAM_SECTION.URAM_A.value, URAM_START_ADDR,
                            dst, row_bytes, inst_id)
                        self.wait_queue(); inst_id += 1

        # Step 3: Matmul R_all @ G → (SC * H_out_pad, W_out * 64) = (SC, N_out_pad, 64)
        self.matmat_mul_core(M=SC * H_out_pad, K=K_g, N=N_g,
            A_DRAM_ADDR=self.IM2COL_R_DW_DRAM,
            B_DRAM_ADDR=self.w[g_key],
            OUTPUT_DRAM_ADDR=self.SUB_PATCH_DRAM)

        self.stop_capture()
        self.generate_instruction_halt()
        prog = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prog)
        prog_bytes = self.get_capture_instruction_size_bytes()
        self.allocate_program_dram(prog_bytes)
        return prog, H_out, W_out
    def compile_sub_stage0(self, N0, padded_k, SC):
        """HW program: im2col patches(N0,64) @ W(64,256) + bias, ReLU. Patches pre-built on host."""
        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()
        self.matmat_mul_core(M=N0, K=padded_k, N=SC,
            A_DRAM_ADDR=self.SUB_PATCH_DRAM, B_DRAM_ADDR=self.w["SUB_CONV0_W"],
            OUTPUT_DRAM_ADDR=self.SUB_OUT0_DRAM,
            C_DRAM_ADDR=self.w["SUB_CONV0_B"], relu_enable=True)
        self.stop_capture()
        self.generate_instruction_halt()
        prog = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prog)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        return prog, 2 * N0 * padded_k * SC
    def compile_sub_stage_dw_pw(self, N_in, N_out, SC, dw_patch_addr, dw_w_key, dw_b_key, pw_w_key, pw_b_key):
        """HW program: DW patches(N_out,64) @ kernels via per-channel matmul, then PW(N_out,256)@(256,256)+bias+ReLU.
        DW im2col patches pre-built on host at dw_patch_addr as (SC, N_out_pad, 64).
        N_out is padded to N_out_pad (next multiple of block_size) for aligned matmuls."""
        bpe = self.bytes_per_element
        N_out_pad = pad_to_multiple(N_out, self.block_size)
        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()
        flops = 0
        # DW: per channel, kernel(1,64) @ patches(N_out_pad,64)^T = (1, N_out_pad)
        for ch in range(SC):
            kernel_addr = self.w[dw_w_key] + ch * 64 * bpe
            patch_addr = dw_patch_addr + ch * N_out_pad * 64 * bpe
            out_addr = self.SUB_DW_OUT_DRAM + ch * N_out_pad * bpe
            self.matmat_mul_core(M=1, K=64, N=N_out_pad,
                A_DRAM_ADDR=kernel_addr, B_DRAM_ADDR=patch_addr, OUTPUT_DRAM_ADDR=out_addr)
            flops += 2 * 64 * N_out_pad
        # DW bias: broadcast_add per channel
        dw_bias = torch.zeros(SC, dtype=torch.bfloat16)
        self.dma_read(DMA_DEVICE_C2H, self.w[dw_b_key], dw_bias, SC * bpe)
        for ch in range(SC):
            out_addr = self.SUB_DW_OUT_DRAM + ch * N_out_pad * bpe
            self.accelerator_memory_to_sram(out_addr, URAM_A_BASE, N_out_pad)
            self.broadcast_add(scalar=dw_bias[ch].float().item(),
                sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=N_out_pad)
            self.sram_to_accelerator_memory(URAM_A_BASE, out_addr, N_out_pad)
        # Chunked on-device transpose: (SC, N_out_pad) -> (N_out_pad, SC)
        chunk = SC  # 256
        for c_start in range(0, N_out_pad, chunk):
            c_end = min(c_start + chunk, N_out_pad)
            c_len = c_end - c_start
            c_len_pad = pad_to_multiple(c_len, self.block_size)
            for ch in range(SC):
                src = self.SUB_DW_OUT_DRAM + (ch * N_out_pad + c_start) * bpe
                dst = self.PERMUTE_TEMP_DRAM + ch * c_len_pad * bpe
                self.accelerator_memory_to_sram(src, URAM_A_BASE, c_len)
                self.sram_to_accelerator_memory(URAM_A_BASE, dst, c_len)
            bf16_smart_permute_core(self,
                dims=[SC, c_len_pad], permute_indices=[1, 0],
                input_dram_addr=self.PERMUTE_TEMP_DRAM,
                output_dram_addr=self.SUB_PW_IN_DRAM + c_start * SC * bpe,
                params_dram_addr=self.w["IDENTITY_64"],
                temp_dram_start=self.PERMUTE_TEMP_DRAM + SC * c_len_pad * bpe)
        # PW: (N_out_pad, 256) @ (256, 256) + bias, ReLU
        self.matmat_mul_core(M=N_out_pad, K=SC, N=SC,
            A_DRAM_ADDR=self.SUB_PW_IN_DRAM, B_DRAM_ADDR=self.w[pw_w_key],
            OUTPUT_DRAM_ADDR=self.SUB_PW_IN_DRAM,
            C_DRAM_ADDR=self.w[pw_b_key], relu_enable=True)
        flops += 2 * N_out * SC * SC
        self.stop_capture()
        self.generate_instruction_halt()
        prog = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prog)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        return prog, flops
    def compile_sub_flatten_linear(self, H2, W2):
        """HW program: read stage-2 output (N2, SC) directly from SUB_PW_IN_DRAM as (H2, W2*SC)
        and multiply by permuted weight to produce (H2, D) at INPUT_DRAM.
        The column permute is absorbed into the weight matrix (SUB_OUT_W_PERM)."""
        D = self.d_model
        K = W2 * self.sub_channels  # W2*SC = 4096
        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()
        self.matmat_mul_core(M=H2, K=K, N=D,
            A_DRAM_ADDR=self.SUB_PW_IN_DRAM,
            B_DRAM_ADDR=self.w["SUB_OUT_W_PERM"],
            OUTPUT_DRAM_ADDR=self.INPUT_DRAM,
            C_DRAM_ADDR=self.w["SUB_OUT_B"])
        self.stop_capture()
        self.generate_instruction_halt()
        prog = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prog)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        return prog, 2 * H2 * K * D
    def prepare_attention_tiled_biases(self, L_pad):
        """Pre-tile attention bias_u and bias_v for bulk eltwise_add.
        For each layer and head, creates (L_pad, dk) tiled bias in DRAM.
        Must be called after weight_init() and before compile_encoder().
        """
        dk = self.head_dim  # 128
        bpe = self.bytes_per_element
        for layer_idx in range(self.num_layers):
            la = self.layer_addrs[layer_idx]
            for bias_key in ("ATTN_BIAS_U", "ATTN_BIAS_V"):
                # Read the full (H_heads * dk) bias from DRAM
                full_bias = torch.zeros(self.num_heads * dk, dtype=torch.bfloat16)
                self.dma_read(DMA_DEVICE_C2H, la[bias_key], full_bias, self.num_heads * dk * bpe)
                tiled_key = f"{bias_key}_TILED"
                head_addrs = []
                for h in range(self.num_heads):
                    head_bias = full_bias[h * dk:(h + 1) * dk]
                    # Tile: repeat dk-element bias L_pad times
                    tiled = head_bias.unsqueeze(0).expand(L_pad, dk).contiguous()
                    addr = self._alloc_write(tiled)
                    head_addrs.append(addr)
                la[tiled_key] = head_addrs
    def _enc_matmul(self, M, K, N, A_DRAM_ADDR, w_addr, OUTPUT_DRAM_ADDR, **kw):
        """Encoder matmul (bf16)."""
        self.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=A_DRAM_ADDR,
            B_DRAM_ADDR=w_addr, OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR, **kw)
    def compile_encoder(self, L_pad, toeplitz_addrs, bn_tiled):
        """Compile full 24-layer conformer encoder. Returns (program_addr, program_bytes).
        Must call prepare_attention_tiled_biases(L_pad) before this.
        Args:
            L_pad: padded sequence length
            toeplitz_addrs: list of 24 DRAM addresses for Toeplitz DW conv matrices
            bn_tiled: list of 24 (tiled_scale_addr, tiled_shift_addr) tuples
        """
        D, FF, H_heads, dk = self.d_model, self.ff_dim, self.num_heads, self.head_dim
        bpe = self.bytes_per_element
        P = 2 * L_pad - 1
        P_pad = pad_to_multiple(P, self.block_size)

        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()
        total_flops = 0

        for layer_idx in range(self.num_layers):
            la = self.layer_addrs[layer_idx]

            # ===== FF1 (half-step residual, K-split) =====
            FF_HALF = FF // 2
            self.layer_norm_core_dram(M=L_pad, N=D,A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.LN_OUT_DRAM,GAMMA_DRAM_ADDR=la["LN_FF1_WEIGHT"], BETA_DRAM_ADDR=la["LN_FF1_BIAS"])
            # Up-proj split: two (L_pad, D) @ (D, FF/2) = (L_pad, FF/2) with SiLU
            self._enc_matmul(L_pad, D, FF_HALF, self.LN_OUT_DRAM, la["FF1_W1_LO"], self.FF_MID_DRAM, silu_enable=True)
            self._enc_matmul(L_pad, D, FF_HALF, self.LN_OUT_DRAM, la["FF1_W1_HI"], self.FF_MID_DRAM + L_pad * FF_HALF * bpe, silu_enable=True)
            total_flops += 2 * L_pad * D * FF
            # Down-proj split: two (L_pad, FF/2) @ (FF/2, D) + accumulate
            self._enc_matmul(L_pad, FF_HALF, D, self.FF_MID_DRAM, la["FF1_W2_LO"], self.FF_OUT_DRAM)
            self._enc_matmul(L_pad, FF_HALF, D, self.FF_MID_DRAM + L_pad * FF_HALF * bpe, la["FF1_W2_HI"], self.LN_OUT_DRAM)
            self.accelerator_memory_to_sram(self.FF_OUT_DRAM, URAM_A_BASE, L_pad * D)
            self.accelerator_memory_to_sram(self.LN_OUT_DRAM, URAM_B_BASE, L_pad * D)
            self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * D)
            self.sram_to_accelerator_memory(URAM_A_BASE, self.FF_OUT_DRAM, L_pad * D)
            total_flops += 2 * L_pad * FF * D
            half_step_residual_core_dram(self, M=L_pad, N=D,
                RESIDUAL_DRAM_ADDR=self.INPUT_DRAM, FF_DRAM_ADDR=self.FF_OUT_DRAM,
                OUTPUT_DRAM_ADDR=self.INPUT_DRAM)
            total_flops += 2 * L_pad * D

            # ===== Self-attention =====
            self.layer_norm_core_dram(M=L_pad, N=D, A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.LN_OUT_DRAM, GAMMA_DRAM_ADDR=la["LN_ATTN_WEIGHT"], BETA_DRAM_ADDR=la["LN_ATTN_BIAS"])
            self._enc_matmul(L_pad, D, D, self.LN_OUT_DRAM, la["ATTN_Q_W"], self.Q_DRAM)
            self._enc_matmul(L_pad, D, D, self.LN_OUT_DRAM, la["ATTN_K_W"], self.K_DRAM)
            self._enc_matmul(L_pad, D, D, self.LN_OUT_DRAM, la["ATTN_V_W"], self.V_DRAM)
            total_flops += 3 * 2 * L_pad * D * D
            self._enc_matmul(P_pad, D, D, self.POS_EMB_DRAM, la["ATTN_POS_W"], self.POS_PROJ_DRAM)
            total_flops += 2 * P_pad * D * D

            for h in range(H_heads):
                h_off = h * dk * bpe
                # Q_h + bias_u (TILED)
                self.accelerator_memory_to_sram(self.Q_DRAM + h_off, URAM_A_BASE, L_pad * dk,
                    stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                self.accelerator_memory_to_sram(la["ATTN_BIAS_U_TILED"][h], URAM_B_BASE, L_pad * dk)
                self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * dk)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_A_DRAM, L_pad * dk)
                # K_h
                self.accelerator_memory_to_sram(self.K_DRAM + h_off, URAM_A_BASE, L_pad * dk,
                    stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_B_DRAM, L_pad * dk)
                # Content scores
                self.matmat_mul_core(M=L_pad, K=dk, N=L_pad,
                    A_DRAM_ADDR=self.CONV_A_DRAM, B_DRAM_ADDR=self.CONV_B_DRAM,
                    OUTPUT_DRAM_ADDR=self.SCORE_DRAM)
                total_flops += 2 * L_pad * dk * L_pad
                # Positional: Q_h + bias_v (TILED)
                self.accelerator_memory_to_sram(self.Q_DRAM + h_off, URAM_A_BASE, L_pad * dk, stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                self.accelerator_memory_to_sram(la["ATTN_BIAS_V_TILED"][h], URAM_B_BASE, L_pad * dk)
                self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * dk)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_A_DRAM, L_pad * dk)
                # P_h
                self.accelerator_memory_to_sram(self.POS_PROJ_DRAM + h_off, URAM_A_BASE, P_pad * dk, stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_B_DRAM, P_pad * dk)
                self.matmat_mul_core(M=L_pad, K=dk, N=P_pad, A_DRAM_ADDR=self.CONV_A_DRAM, B_DRAM_ADDR=self.CONV_B_DRAM, OUTPUT_DRAM_ADDR=self.CONV_OUT_DRAM)
                total_flops += 2 * L_pad * dk * P_pad
                rel_shift_core_dram(self, L=L_pad, INPUT_DRAM_ADDR=self.CONV_OUT_DRAM, OUTPUT_DRAM_ADDR=self.REL_SHIFT_DRAM, input_row_stride=P_pad)
                # Combine content + pos
                score_elems = L_pad * L_pad
                self.accelerator_memory_to_sram(self.SCORE_DRAM, URAM_A_BASE, score_elems)
                self.accelerator_memory_to_sram(self.REL_SHIFT_DRAM, URAM_B_BASE, score_elems)
                self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, score_elems)
                # Scale (data already in URAM_A from eltwise_add above)
                inv_sqrt_dk = 1.0 / math.sqrt(dk)
                self.broadcast_mul(scalar=inv_sqrt_dk, sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=score_elems)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.SCORE_DRAM, score_elems)
                # Softmax + mask
                self.matmat_mul_core(M=L_pad, K=L_pad, N=L_pad, A_DRAM_ADDR=self.SCORE_DRAM, B_DRAM_ADDR=self.IDENTITY_LPAD_DRAM, OUTPUT_DRAM_ADDR=self.SCORE_DRAM, softmax_enable=True, C_DRAM_ADDR=self.ATTN_MASK_DRAM, bias_mode="full_matrix")
                total_flops += 2 * L_pad * L_pad * L_pad  # softmax identity matmul
                # V_h -> transpose -> attn @ V_h
                self.accelerator_memory_to_sram(self.V_DRAM + h_off, URAM_A_BASE, L_pad * dk, stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_B_DRAM, L_pad * dk)
                chunked_transpose_core_dram(self, M=L_pad, N=dk, input_dram_addr=self.CONV_B_DRAM, output_dram_addr=self.ATTN_VT_DRAM, identity_dram_addr=self.w["IDENTITY_64"], temp_dram_addr=self.PERMUTE_TEMP_DRAM)
                total_flops += (dk // UE_VECTOR_SIZE) * dk * 2 * UE_VECTOR_SIZE * pad_to_multiple(L_pad, UE_VECTOR_SIZE)  # V transpose dot-products
                self.matmat_mul_core(M=L_pad, K=L_pad, N=dk, A_DRAM_ADDR=self.SCORE_DRAM, B_DRAM_ADDR=self.ATTN_VT_DRAM, OUTPUT_DRAM_ADDR=self.CONV_A_DRAM)
                total_flops += 2 * L_pad * L_pad * dk
                # Write head output (strided)
                self.accelerator_memory_to_sram(self.CONV_A_DRAM, URAM_A_BASE, L_pad * dk)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.ATTN_OUT_DRAM + h_off, L_pad * dk, stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
            # Output projection + residual
            self._enc_matmul(L_pad, D, D, self.ATTN_OUT_DRAM, la["ATTN_OUT_W"], self.FF_OUT_DRAM)
            total_flops += 2 * L_pad * D * D
            self.accelerator_memory_to_sram(self.INPUT_DRAM, URAM_A_BASE, L_pad * D)
            self.accelerator_memory_to_sram(self.FF_OUT_DRAM, URAM_B_BASE, L_pad * D)
            self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * D)
            self.sram_to_accelerator_memory(URAM_A_BASE, self.INPUT_DRAM, L_pad * D)
            total_flops += L_pad * D

            # ===== Conv module =====
            self.layer_norm_core_dram(M=L_pad, N=D, A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.LN_OUT_DRAM, GAMMA_DRAM_ADDR=la["LN_CONV_WEIGHT"], BETA_DRAM_ADDR=la["LN_CONV_BIAS"])
            self._enc_matmul(L_pad, D, D, self.LN_OUT_DRAM, la["CONV_PW1A_W"], self.CONV_A_DRAM)
            self._enc_matmul(L_pad, D, D, self.LN_OUT_DRAM, la["CONV_PW1B_W"], self.CONV_B_DRAM, sigmoid_enable=True)
            total_flops += 2 * 2 * L_pad * D * D
            # GLU: a * sigmoid(b) — sigmoid fused into PW1b above
            self.accelerator_memory_to_sram(self.CONV_A_DRAM, URAM_A_BASE, L_pad * D)
            self.accelerator_memory_to_sram(self.CONV_B_DRAM, URAM_B_BASE, L_pad * D)
            self.eltwise_mul_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * D)
            self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_A_DRAM, L_pad * D)
            # Transpose (L_pad, D) -> (D, L_pad) for channel-first DW conv.
            chunked_transpose_core_dram(self, M=L_pad, N=D, input_dram_addr=self.CONV_A_DRAM, output_dram_addr=self.CONV_T_DRAM, identity_dram_addr=self.w["IDENTITY_64"], temp_dram_addr=self.PERMUTE_TEMP_DRAM)
            total_flops += (D // UE_VECTOR_SIZE) * D * 2 * UE_VECTOR_SIZE * pad_to_multiple(L_pad, UE_VECTOR_SIZE)  # transpose dot-products
            # DW Conv1d via Toeplitz: per-channel matmul(1, L_pad) @ T(L_pad, L_pad)
            t_addr = toeplitz_addrs[layer_idx]
            for ch in range(D):
                self.matmat_mul_core(M=1, K=L_pad, N=L_pad, A_DRAM_ADDR=self.CONV_T_DRAM + ch * L_pad * bpe, B_DRAM_ADDR=t_addr + ch * L_pad * L_pad * bpe, OUTPUT_DRAM_ADDR=self.CONV_DW_DRAM + ch * L_pad * bpe)
            total_flops += D * 2 * 1 * L_pad * L_pad  # actual HW compute (full Toeplitz matmul, not just 9-tap)
            batch_norm_core_dram(self, C=D, L=L_pad, A_DRAM_ADDR=self.CONV_DW_DRAM, OUTPUT_DRAM_ADDR=self.CONV_DW_DRAM, SCALE_DRAM_ADDR=la["CONV_BN_SCALE"], SHIFT_DRAM_ADDR=la["CONV_BN_SHIFT"], tiled_scale_addr=bn_tiled[layer_idx][0], tiled_shift_addr=bn_tiled[layer_idx][1])
            silu_core_dram(self, M=D, N=L_pad, A_DRAM_ADDR=self.CONV_DW_DRAM, OUTPUT_DRAM_ADDR=self.CONV_OUT_DRAM, IDENTITY_DRAM_ADDR=self.IDENTITY_LPAD_DRAM)
            total_flops += 2 * D * L_pad * L_pad  # SiLU identity matmul
            # Transpose (D, L_pad) -> (L_pad, D) from SiLU output.
            chunked_transpose_core_dram(self, M=D, N=L_pad, input_dram_addr=self.CONV_OUT_DRAM, output_dram_addr=self.CONV_T_DRAM, identity_dram_addr=self.w["IDENTITY_64"], temp_dram_addr=self.PERMUTE_TEMP_DRAM)
            total_flops += (L_pad // UE_VECTOR_SIZE) * L_pad * 2 * UE_VECTOR_SIZE * pad_to_multiple(D, UE_VECTOR_SIZE)  # transpose dot-products
            self._enc_matmul(L_pad, D, D, self.CONV_T_DRAM, la["CONV_PW2_W"], self.CONV_OUT_DRAM)
            total_flops += 2 * L_pad * D * D
            self.accelerator_memory_to_sram(self.INPUT_DRAM, URAM_A_BASE, L_pad * D)
            self.accelerator_memory_to_sram(self.CONV_OUT_DRAM, URAM_B_BASE, L_pad * D)
            self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * D)
            self.sram_to_accelerator_memory(URAM_A_BASE, self.INPUT_DRAM, L_pad * D)
            total_flops += L_pad * D

            # ===== FF2 (half-step residual, K-split) =====
            self.layer_norm_core_dram(M=L_pad, N=D, A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.LN_OUT_DRAM, GAMMA_DRAM_ADDR=la["LN_FF2_WEIGHT"], BETA_DRAM_ADDR=la["LN_FF2_BIAS"])
            # Up-proj split
            self._enc_matmul(L_pad, D, FF_HALF, self.LN_OUT_DRAM, la["FF2_W1_LO"], self.FF_MID_DRAM, silu_enable=True)
            self._enc_matmul(L_pad, D, FF_HALF, self.LN_OUT_DRAM, la["FF2_W1_HI"], self.FF_MID_DRAM + L_pad * FF_HALF * bpe, silu_enable=True)
            total_flops += 2 * L_pad * D * FF
            # Down-proj split + accumulate
            self._enc_matmul(L_pad, FF_HALF, D, self.FF_MID_DRAM, la["FF2_W2_LO"], self.FF_OUT_DRAM)
            self._enc_matmul(L_pad, FF_HALF, D, self.FF_MID_DRAM + L_pad * FF_HALF * bpe, la["FF2_W2_HI"], self.LN_OUT_DRAM)
            self.accelerator_memory_to_sram(self.FF_OUT_DRAM, URAM_A_BASE, L_pad * D)
            self.accelerator_memory_to_sram(self.LN_OUT_DRAM, URAM_B_BASE, L_pad * D)
            self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * D)
            self.sram_to_accelerator_memory(URAM_A_BASE, self.FF_OUT_DRAM, L_pad * D)
            total_flops += 2 * L_pad * FF * D
            half_step_residual_core_dram(self, M=L_pad, N=D,
                RESIDUAL_DRAM_ADDR=self.INPUT_DRAM, FF_DRAM_ADDR=self.FF_OUT_DRAM,
                OUTPUT_DRAM_ADDR=self.INPUT_DRAM)
            total_flops += 2 * L_pad * D
            # ===== Final LayerNorm =====
            self.layer_norm_core_dram(M=L_pad, N=D, A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.INPUT_DRAM, GAMMA_DRAM_ADDR=la["LN_OUT_WEIGHT"], BETA_DRAM_ADDR=la["LN_OUT_BIAS"])
            _original_print(f"\r  Compiling encoder layer {layer_idx + 1}/{self.num_layers}", end="", flush=True)

        _original_print()  # newline after \r progress
        # Copy final encoder output
        self.accelerator_memory_to_sram(self.INPUT_DRAM, URAM_A_BASE, L_pad * D)
        self.sram_to_accelerator_memory(URAM_A_BASE, self.ENC_OUT_DRAM, L_pad * D)

        self.stop_capture()
        self.generate_instruction_halt()
        program_addr = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(program_addr)
        program_bytes = self.get_capture_instruction_size_bytes()
        self.allocate_program_dram(program_bytes)
        return program_addr, program_bytes
    def compile_decoder(self):
        """Compile decoder programs with on-device DMA (no host round-trips per step).

        Returns (pred_prog, tok_prog, dur_prog, state_restore_prog, total_flops).

        pred_prog: LSTM state save + embedding lookup (register-addressed) + predictor LSTM
        tok_prog: enc_out[t] copy (register-addressed) + pred_out copy + joint projections + token argmax
        dur_prog: duration matmul + argmax
        state_restore_prog: copy saved LSTM state back (for blank rollback)
        """
        H = self.pred_hidden
        D = self.d_model
        bpe = self.bytes_per_element
        N_tok_pad = pad_to_multiple(self.vocab_size, self.block_size)
        N_dur_pad = pad_to_multiple(len(self.tdt_durations), self.block_size)
        total_flops = 0
        # --- Predictor program: state save + embedding + LSTM ---
        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()
        # 1. Save LSTM state to on-device temp buffers
        for src, dst in [(self.PRED_H0_DRAM, self.SAVED_H0_DRAM),
                         (self.PRED_C0_DRAM, self.SAVED_C0_DRAM),
                         (self.PRED_H1_DRAM, self.SAVED_H1_DRAM),
                         (self.PRED_C1_DRAM, self.SAVED_C1_DRAM)]:
            self.accelerator_memory_to_sram(src, URAM_A_BASE, H)
            self.sram_to_accelerator_memory(URAM_A_BASE, dst, H)
        # 2. Embedding lookup: TMP_REG = TOKEN_REG + EMBED_base, DMA from TMP_REG
        self.generate_instruction_add_imm(self.TOKEN_REG, self.w["EMBED"], self.TMP_REG)
        self.accelerator_memory_to_sram(self.w["EMBED"], URAM_A_BASE, H)
        self.overwrite_instruction_with_general_register(self.TMP_REG)
        self.sram_to_accelerator_memory(URAM_A_BASE, self.PRED_EMB_DRAM, H)
        # 3. Predictor LSTM (2 layers)
        for i in range(2):
            h_addr = self.PRED_H0_DRAM if i == 0 else self.PRED_H1_DRAM
            c_addr = self.PRED_C0_DRAM if i == 0 else self.PRED_C1_DRAM
            x_addr = self.PRED_EMB_DRAM if i == 0 else self.PRED_H0_DRAM
            self.matmat_mul_core(M=1, K=H, N=4*H, A_DRAM_ADDR=x_addr, B_DRAM_ADDR=self.w[f"LSTM_WIH{i}"], OUTPUT_DRAM_ADDR=self.PRED_GATES_DRAM, C_DRAM_ADDR=self.w[f"LSTM_BIH{i}"])
            total_flops += 2 * H * 4 * H
            self.matmat_mul_core(M=1, K=H, N=4*H, A_DRAM_ADDR=h_addr, B_DRAM_ADDR=self.w[f"LSTM_WHH{i}"], OUTPUT_DRAM_ADDR=self.PRED_GATES2_DRAM, C_DRAM_ADDR=self.w[f"LSTM_BHH{i}"])
            total_flops += 2 * H * 4 * H
            self.accelerator_memory_to_sram(self.PRED_GATES_DRAM, URAM_A_BASE, 4*H)
            self.accelerator_memory_to_sram(self.PRED_GATES2_DRAM, URAM_B_BASE, 4*H)
            self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, 4*H)
            self.sram_to_accelerator_memory(URAM_A_BASE, self.PRED_GATES_DRAM, 4*H)
            self.matmat_mul_core(M=1, K=H, N=H, A_DRAM_ADDR=self.PRED_GATES_DRAM, B_DRAM_ADDR=self.w["IDENTITY_640"], OUTPUT_DRAM_ADDR=self.PRED_GATES_DRAM, sigmoid_enable=True)
            self.matmat_mul_core(M=1, K=H, N=H, A_DRAM_ADDR=self.PRED_GATES_DRAM + H * bpe, B_DRAM_ADDR=self.w["IDENTITY_640"], OUTPUT_DRAM_ADDR=self.PRED_GATES_DRAM + H * bpe, sigmoid_enable=True)
            tanh_core_dram(self, M=1, N=H, A_DRAM_ADDR=self.PRED_GATES_DRAM + 2 * H * bpe, OUTPUT_DRAM_ADDR=self.PRED_GATES_DRAM + 2 * H * bpe, IDENTITY_DRAM_ADDR=self.w["IDENTITY_640"])
            self.matmat_mul_core(M=1, K=H, N=H, A_DRAM_ADDR=self.PRED_GATES_DRAM + 3 * H * bpe, B_DRAM_ADDR=self.w["IDENTITY_640"], OUTPUT_DRAM_ADDR=self.PRED_GATES_DRAM + 3 * H * bpe, sigmoid_enable=True)
            self.accelerator_memory_to_sram(self.PRED_GATES_DRAM + H * bpe, URAM_A_BASE, H)
            self.accelerator_memory_to_sram(c_addr, URAM_B_BASE, H)
            self.eltwise_mul_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, H)
            self.sram_to_accelerator_memory(URAM_A_BASE, c_addr, H)
            self.accelerator_memory_to_sram(self.PRED_GATES_DRAM, URAM_A_BASE, H)
            self.accelerator_memory_to_sram(self.PRED_GATES_DRAM + 2 * H * bpe, URAM_B_BASE, H)
            self.eltwise_mul_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, H)
            self.accelerator_memory_to_sram(c_addr, URAM_B_BASE, H)
            self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, H)
            self.sram_to_accelerator_memory(URAM_A_BASE, c_addr, H)
            tanh_core_dram(self, M=1, N=H, A_DRAM_ADDR=c_addr, OUTPUT_DRAM_ADDR=self.PRED_OUT_DRAM, IDENTITY_DRAM_ADDR=self.w["IDENTITY_640"])
            self.accelerator_memory_to_sram(self.PRED_GATES_DRAM + 3 * H * bpe, URAM_A_BASE, H)
            self.accelerator_memory_to_sram(self.PRED_OUT_DRAM, URAM_B_BASE, H)
            self.eltwise_mul_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, H)
            self.sram_to_accelerator_memory(URAM_A_BASE, h_addr, H)
        self.accelerator_memory_to_sram(self.PRED_H1_DRAM, URAM_A_BASE, H)
        self.sram_to_accelerator_memory(URAM_A_BASE, self.PRED_OUT_DRAM, H)
        self.stop_capture()
        self.generate_instruction_halt()
        pred_prog = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(pred_prog)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        # --- Joint token program: enc_out copy + pred_out copy + projections + token argmax ---
        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()
        # enc_out[t] copy: TMP_REG = ENC_T_REG + ENC_OUT_DRAM, DMA from TMP_REG
        self.generate_instruction_add_imm(self.ENC_T_REG, self.ENC_OUT_DRAM, self.TMP_REG)
        self.accelerator_memory_to_sram(self.ENC_OUT_DRAM, URAM_A_BASE, D)
        self.overwrite_instruction_with_general_register(self.TMP_REG)
        self.sram_to_accelerator_memory(URAM_A_BASE, self.JOINT_ENC_DRAM, D)
        # pred_out copy (static addresses)
        self.accelerator_memory_to_sram(self.PRED_OUT_DRAM, URAM_A_BASE, H)
        self.sram_to_accelerator_memory(URAM_A_BASE, self.JOINT_PRED_DRAM, H)
        # Joint projections + ReLU
        self.matmat_mul_core(M=1, K=D, N=H, A_DRAM_ADDR=self.JOINT_ENC_DRAM, B_DRAM_ADDR=self.w["JOINT_ENC_W"], OUTPUT_DRAM_ADDR=self.JOINT_ENC_DRAM, C_DRAM_ADDR=self.w["JOINT_ENC_B"])
        total_flops += 2 * D * H
        self.matmat_mul_core(M=1, K=H, N=H, A_DRAM_ADDR=self.JOINT_PRED_DRAM, B_DRAM_ADDR=self.w["JOINT_PRED_W"], OUTPUT_DRAM_ADDR=self.JOINT_PRED_DRAM, C_DRAM_ADDR=self.w["JOINT_PRED_B"])
        total_flops += 2 * H * H
        self.accelerator_memory_to_sram(self.JOINT_ENC_DRAM, URAM_A_BASE, H)
        self.accelerator_memory_to_sram(self.JOINT_PRED_DRAM, URAM_B_BASE, H)
        self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, H)
        self.sram_to_accelerator_memory(URAM_A_BASE, self.JOINT_SUM_DRAM, H)
        self.matmat_mul_core(M=1, K=H, N=H, A_DRAM_ADDR=self.JOINT_SUM_DRAM, B_DRAM_ADDR=self.w["IDENTITY_640"], OUTPUT_DRAM_ADDR=self.JOINT_SUM_DRAM, relu_enable=True)
        # Token matmul + argmax
        self.matmat_mul_core(M=1, K=H, N=N_tok_pad, A_DRAM_ADDR=self.JOINT_SUM_DRAM, B_DRAM_ADDR=self.w["JOINT_OUT_TOK_W"], OUTPUT_DRAM_ADDR=self.JOINT_TOK_DRAM, C_DRAM_ADDR=self.w["JOINT_OUT_TOK_B"])
        total_flops += 2 * H * N_tok_pad
        self.stop_capture()
        self.generate_instruction_halt()
        tok_prog = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(tok_prog)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        # --- Joint duration program ---
        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()
        self.matmat_mul_core(M=1, K=H, N=N_dur_pad, A_DRAM_ADDR=self.JOINT_SUM_DRAM, B_DRAM_ADDR=self.w["JOINT_OUT_DUR_W"], OUTPUT_DRAM_ADDR=self.JOINT_DUR_DRAM, C_DRAM_ADDR=self.w["JOINT_OUT_DUR_B"])
        total_flops += 2 * H * N_dur_pad
        self.stop_capture()
        self.generate_instruction_halt()
        dur_prog = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(dur_prog)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        # --- State restore program (for blank rollback) ---
        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()
        for src, dst in [(self.SAVED_H0_DRAM, self.PRED_H0_DRAM),
                         (self.SAVED_C0_DRAM, self.PRED_C0_DRAM),
                         (self.SAVED_H1_DRAM, self.PRED_H1_DRAM),
                         (self.SAVED_C1_DRAM, self.PRED_C1_DRAM)]:
            self.accelerator_memory_to_sram(src, URAM_A_BASE, H)
            self.sram_to_accelerator_memory(URAM_A_BASE, dst, H)
        self.stop_capture()
        self.generate_instruction_halt()
        restore_prog = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(restore_prog)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())

        return pred_prog, tok_prog, dur_prog, restore_prog, total_flops
    def program_execute(self, program_addr, timeout=50.0, flops=None):
        """Execute compiled program. Returns (latency_us, gflops)."""
        self.start_execute_from_dram(program_addr)
        latency_us, gflops = 0, 0
        if timeout > 0:
            self.wait_queue(timeout)
            latency_us = self.report_latency_in_us()
            if flops:
                gflops = self.report_flop_rate_gflops(flops)
        return latency_us, gflops
    def get_arg_max_index1(self):
        """Read hardware argmax1 register."""
        return self.read_reg32(user_dma_core.UE_ARGMAX1_INDEX)
    def get_arg_max_index2(self):
        """Read hardware argmax2 register."""
        return self.read_reg32(user_dma_core.UE_ARGMAX2_INDEX)
    def make_rel_pos_emb(self, seq_len):
        """Generate relative positional encoding: (2*seq_len-1, D) bf16."""
        D = self.d_model
        max_len = 2 * seq_len
        pe = torch.zeros(max_len, D)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, D, 2, dtype=torch.float32) * -(math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe_pos = pe[:seq_len]
        pe_neg = torch.flip(pe[1:seq_len], [0])
        return torch.cat([pe_neg, pe_pos], dim=0).to(torch.bfloat16)
    # -----------------------------------------------------------------------
    # Bin dump / load
    # -----------------------------------------------------------------------
    def dump_params(self, L_pad):
        """Dump entire params DRAM to parakeet_bin/params.bin + params.json."""
        bin_dir = os.path.join(self.script_dir, "parakeet_bin")
        bin_path = os.path.join(bin_dir, "params.bin")
        meta_path = os.path.join(bin_dir, "params.json")
        total = self.get_params_dram_usage()
        # Read back entire params region in chunks
        CHUNK = 1 * 1024 * 1024  # 1 MB
        with open(bin_path, "wb") as f:
            offset = 0
            while offset < total:
                sz = min(CHUNK, total - offset)
                buf = bytearray(sz)
                self.dma_read(DMA_DEVICE_C2H, self._params_dram_base + offset, buf, sz)
                f.write(buf)
                offset += sz
        with open(meta_path, "w") as f:
            json.dump({"L_pad": L_pad, "size": total}, f)
        _original_print(f"  Params dumped: {total / 1024**2:.1f} MB → {bin_path}")
    def load_params(self, L_pad):
        """Load params DRAM from bin. Returns True if loaded, False if not found/mismatch."""
        bin_dir = os.path.join(self.script_dir, "parakeet_bin")
        bin_path = os.path.join(bin_dir, "params.bin")
        meta_path = os.path.join(bin_dir, "params.json")
        if not os.path.exists(bin_path) or not os.path.exists(meta_path):
            return False
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("L_pad") != L_pad:
            _original_print(f"  Params bin L_pad={meta.get('L_pad')} != {L_pad}, reloading weights")
            return False
        total = meta["size"]
        CHUNK = 1 * 1024 * 1024
        with open(bin_path, "rb") as f:
            offset = 0
            while offset < total:
                data = f.read(min(CHUNK, total - offset))
                self.dma_write(DMA_DEVICE_H2C, self._params_dram_base + offset, data, len(data))
                offset += len(data)
        self.allocate_params_dram(total)
        _original_print(f"  Loading Toeplitz Staged BF16 Weights... {total / 1024**2:.1f} MB from bin")
        return True
    def dump_programs(self, program_addrs, L_pad):
        """Dump compiled programs to bin + json manifest in parakeet_bin/.
        program_addrs: dict of {name: (dram_addr, size_bytes)}."""
        bin_dir = os.path.join(self.script_dir, "parakeet_bin")
        bin_path = os.path.join(bin_dir, "programs.bin")
        meta_path = os.path.join(bin_dir, "programs.json")
        manifest = {"L_pad": L_pad, "programs": {}}
        all_bytes = bytearray()
        for name, (addr, size) in program_addrs.items():
            buf = bytearray(size)
            self.dma_read(DMA_DEVICE_C2H, addr, buf, size)
            manifest["programs"][name] = {"offset": len(all_bytes), "size": size}
            all_bytes.extend(buf)
        with open(bin_path, "wb") as f:
            f.write(all_bytes)
        with open(meta_path, "w") as f:
            json.dump(manifest, f)
        _original_print(f"  Programs dumped: {len(all_bytes)} bytes → {bin_path}")
    def load_programs(self, L_pad):
        """Load compiled programs from bin. Returns dict of {name: dram_addr} or None if not found."""
        bin_dir = os.path.join(self.script_dir, "parakeet_bin")
        bin_path = os.path.join(bin_dir, "programs.bin")
        meta_path = os.path.join(bin_dir, "programs.json")
        if not os.path.exists(bin_path) or not os.path.exists(meta_path):
            return None
        with open(meta_path) as f:
            manifest = json.load(f)
        if manifest.get("L_pad") != L_pad:
            _original_print(f"  Bin L_pad={manifest.get('L_pad')} != current {L_pad}, recompiling")
            return None
        with open(bin_path, "rb") as f:
            all_bytes = f.read()
        addrs = {}
        for name, meta in manifest["programs"].items():
            data = all_bytes[meta["offset"]:meta["offset"] + meta["size"]]
            addr = self.get_program_dram_addr()
            self.dma_write(DMA_DEVICE_H2C, addr, data, len(data))
            self.allocate_program_dram(len(data))
            addrs[name] = addr
        _original_print(f"  Programs loaded: {len(all_bytes)} bytes from bin")
        return addrs
    # -----------------------------------------------------------------------
    # Execute: decoder
    # -----------------------------------------------------------------------
    def run_decode(self, enc_out_addr, L):
        """TDT greedy decode with on-device DMA. Host only sets registers and dispatches programs."""
        bpe = self.bytes_per_element
        H = self.pred_hidden
        D = self.d_model
        pred_prog = self.progs["pred"][0]
        tok_prog = self.progs["joint_tok"][0]
        dur_prog = self.progs["joint_dur"][0]
        restore_prog = self.progs["state_restore"][0]
        # Pre-zero all decoder intermediate buffers
        zeros_h = torch.zeros(H, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.PRED_H0_DRAM, zeros_h)
        self.dma_to_accelerator_memory(self.PRED_C0_DRAM, zeros_h)
        self.dma_to_accelerator_memory(self.PRED_H1_DRAM, zeros_h)
        self.dma_to_accelerator_memory(self.PRED_C1_DRAM, zeros_h)
        self.dma_to_accelerator_memory(self.PRED_EMB_DRAM, zeros_h)
        self.dma_to_accelerator_memory(self.PRED_OUT_DRAM, zeros_h)
        self.dma_to_accelerator_memory(self.PRED_GATES_DRAM, torch.zeros(4 * H, dtype=torch.bfloat16))
        self.dma_to_accelerator_memory(self.PRED_GATES2_DRAM, torch.zeros(4 * H, dtype=torch.bfloat16))
        self.dma_to_accelerator_memory(self.JOINT_ENC_DRAM, torch.zeros(D, dtype=torch.bfloat16))
        self.dma_to_accelerator_memory(self.JOINT_PRED_DRAM, zeros_h)
        self.dma_to_accelerator_memory(self.JOINT_SUM_DRAM, zeros_h)
        N_tok_pad = pad_to_multiple(self.vocab_size, self.block_size)
        N_dur_pad = pad_to_multiple(len(self.tdt_durations), self.block_size)
        self.dma_to_accelerator_memory(self.JOINT_TOK_DRAM, torch.zeros(N_tok_pad, dtype=torch.bfloat16))
        self.dma_to_accelerator_memory(self.JOINT_DUR_DRAM, torch.zeros(N_dur_pad, dtype=torch.bfloat16))
        tokens = []
        t = 0
        last_token = self.blank_id
        total_steps = 0
        while t < L:
            symbols = 0
            while symbols < self.max_symbols_per_step:
                # Set registers: TOKEN_REG = embedding offset, ENC_T_REG = encoder position offset
                self.isa_add_set_core(self.TOKEN_REG, last_token * H * bpe)
                self.isa_add_set_core(self.ENC_T_REG, t * D * bpe)
                # Predictor: state save + embedding lookup (register-addressed) + LSTM
                self.program_execute(pred_prog)
                # Joint token: enc_out[t] copy (register-addressed) + pred_out copy + projections + argmax
                self.program_execute(tok_prog)
                token_id = self.get_arg_max_index1()
                # Joint duration -> hardware argmax
                self.program_execute(dur_prog)
                dur_idx = self.get_arg_max_index1()
                dur = self.tdt_durations[dur_idx] if dur_idx < len(self.tdt_durations) else 0
                total_steps += 1
                if token_id == self.blank_id:
                    # Restore LSTM state from on-device save buffers
                    self.program_execute(restore_prog)
                    t += max(dur, 1)
                    break
                else:
                    tokens.append(token_id)
                    last_token = token_id
                    symbols += 1
                    if dur > 0:
                        t += dur
                        break
            else:
                t += 1
        _original_print(f"  Decode: {total_steps} joint steps, {len(tokens)} tokens emitted")
        return tokens
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parakeet-TDT-0.6B accelerator inference")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file (.wav, .flac, etc.)")
    parser.add_argument("--dev", type=str, default="xdma0", help="XDMA device")
    parser.add_argument("--cycle", type=float, default=5.63, help="Clock cycle in ns")
    args = parser.parse_args()

    global _SILENT_MODE
    _SILENT_MODE = True

    cfg = load_config()
    set_dma_device(args.dev)
    audio_path = args.audio or os.path.join(SCRIPT_DIR, cfg["defaults"]["default_audio"])

    import torchaudio
    import soundfile as sf
    assert os.path.exists(audio_path), f"Audio file not found: {audio_path}"
    data, sr = sf.read(audio_path, dtype="float32")
    waveform = torch.from_numpy(data).T if data.ndim > 1 else torch.from_numpy(data).unsqueeze(0)
    if sr != cfg["preprocessing"]["sample_rate"]:
        waveform = torchaudio.functional.resample(waveform, sr, cfg["preprocessing"]["sample_rate"])
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    audio_dur = waveform.shape[1] / cfg["preprocessing"]["sample_rate"]
    _original_print(f"Parakeet-TDT-0.6B on {args.dev} ({audio_dur:.1f}s audio)")

    # --- Init engine ---
    engine = Parakeet_UnifiedEngine(clock_period_ns=args.cycle)

    # --- Mel spectrogram (CPU) — needs checkpoint for filterbank ---
    Parakeet_UnifiedEngine.ensure_model_files()
    sd = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    mel = compute_mel_spectrogram(waveform, cfg, ckpt_sd=sd)
    T_mel = mel.shape[1]
    n_mels = cfg["encoder"]["n_mels"]  # 128

    # --- Compute correct output dimensions through 3 subsampling stages ---
    H0, W0 = conv2d_outsize(T_mel), conv2d_outsize(n_mels)      # stage 0
    H1, W1 = conv2d_outsize(H0), conv2d_outsize(W0)             # stage 1
    H2, W2 = conv2d_outsize(H1), conv2d_outsize(W1)             # stage 2
    N0 = H0 * W0
    L = H2
    L_pad = pad_to_multiple(L, cfg["hardware"]["block_size"])

    # --- Load weights + Toeplitz (from bin if available, else from checkpoint) ---
    params_loaded = engine.load_params(L_pad)
    if not params_loaded:
        _original_print("  Loading weights...", end=" ", flush=True)
        engine.weight_init()
        _original_print("done")

    # --- Allocate DRAM buffers ---
    engine.tensor_init(L_pad)

    SC = engine.sub_channels  # 256
    D = engine.d_model         # 1024
    bpe = engine.bytes_per_element

    N1 = H1 * W1
    N1_pad = pad_to_multiple(N1, engine.block_size)
    N2 = H2 * W2
    N2_pad = pad_to_multiple(N2, engine.block_size)
    P_pad = pad_to_multiple(2 * L_pad - 1, engine.block_size)

    if not params_loaded:
        # Toeplitz matrices for DW conv (all 24 layers)
        toeplitz_addrs = []
        for li in range(engine.num_layers):
            _original_print(f"\r  Staging Toeplitz layer {li + 1}/{engine.num_layers}", end="", flush=True)
            la_i = engine.layer_addrs[li]
            k_flat = torch.zeros(D * 64, dtype=torch.bfloat16)
            engine.dma_read(DMA_DEVICE_C2H, la_i["CONV_DW_W"], k_flat, D * 64 * bpe)
            kernel = k_flat.reshape(D, 64)
            toeplitz = torch.zeros(D, L_pad, L_pad, dtype=torch.bfloat16)
            for k in range(9):
                offset = k - 4
                t_idx = torch.arange(max(0, -offset), min(L_pad, L_pad - offset))
                toeplitz[:, t_idx, t_idx + offset] = kernel[:, k:k+1].expand(-1, len(t_idx))
            addr = engine.get_params_dram_addr()
            engine.allocate_params_dram(D * L_pad * L_pad * bpe)
            engine.dma_to_accelerator_memory(addr, toeplitz.reshape(-1).contiguous())
            toeplitz_addrs.append(addr)
        _original_print()

        # Pre-tile BN params and attention biases
        bn_tiled = []
        for li in range(engine.num_layers):
            la_i = engine.layer_addrs[li]
            s_addr, sh_addr = batch_norm_prepare_tiled(engine, D, L_pad,
                la_i["CONV_BN_SCALE"], la_i["CONV_BN_SHIFT"])
            bn_tiled.append((s_addr, sh_addr))
        engine.prepare_attention_tiled_biases(L_pad)

    # Position embedding + attention mask (tensor DRAM, always needed)
    rel_pe = engine.make_rel_pos_emb(L_pad)
    if rel_pe.shape[0] < P_pad:
        pe_padded = torch.zeros(P_pad, D, dtype=torch.bfloat16)
        pe_padded[:rel_pe.shape[0], :] = rel_pe
        rel_pe = pe_padded
    engine.dma_to_accelerator_memory(engine.POS_EMB_DRAM, rel_pe.contiguous())
    mask = torch.full((L_pad, L_pad), -1e38, dtype=torch.bfloat16)
    mask[:L, :L] = 0.0
    mask[L:, 0] = 0.0
    engine.dma_to_accelerator_memory(engine.ATTN_MASK_DRAM, mask.contiguous())

    # Dump params to bin (weights + Toeplitz + BN + attention biases)
    if not params_loaded:
        engine.dump_params(L_pad)

    # ==================================================================
    # Compile (or load from bin)
    # ==================================================================
    import time as _time
    import sentencepiece as spm

    loaded = engine.load_programs(L_pad)
    if loaded:
        im2col_s0 = loaded["im2col_s0"]
        prog_s0 = loaded["prog_s0"]
        im2col_s1 = loaded["im2col_s1"]
        prog_s1 = loaded["prog_s1"]
        im2col_s2 = loaded["im2col_s2"]
        prog_s2 = loaded["prog_s2"]
        prog_flatten_lin = loaded["prog_flatten_lin"]
        enc_prog_addr = loaded["encoder"]
        pred_prog = loaded["pred"]
        tok_prog = loaded["tok"]
        dur_prog = loaded["dur"]
        restore_prog = loaded["restore"]
    else:
        # Subsampling programs
        im2col_s0, _, _ = engine.compile_im2col_s0(T_mel)
        prog_s0, _ = engine.compile_sub_stage0(N0, 64, SC)
        im2col_s1, _, _ = engine.compile_im2col_dw(H0, W0, N1_pad, engine.SUB_OUT0_DRAM, "IM2COL_G_S1")
        prog_s1, _ = engine.compile_sub_stage_dw_pw(N0, N1, SC,
            engine.SUB_PATCH_DRAM, "SUB_DW1_W", "SUB_DW1_B", "SUB_PW1_W", "SUB_PW1_B")
        im2col_s2, _, _ = engine.compile_im2col_dw(H1, W1, N2_pad, engine.SUB_PW_IN_DRAM, "IM2COL_G_S2")
        prog_s2, _ = engine.compile_sub_stage_dw_pw(N1, N2, SC,
            engine.SUB_PATCH_DRAM, "SUB_DW2_W", "SUB_DW2_B", "SUB_PW2_W", "SUB_PW2_B")
        prog_flatten_lin, _ = engine.compile_sub_flatten_linear(H2, W2)
        # Encoder program
        enc_prog_addr, enc_prog_bytes = engine.compile_encoder(L_pad, toeplitz_addrs, bn_tiled)
        # Decoder programs
        pred_prog, tok_prog, dur_prog, restore_prog, _ = engine.compile_decoder()
        # Compute program sizes from consecutive DRAM addresses
        all_progs = [
            ("im2col_s0", im2col_s0), ("prog_s0", prog_s0),
            ("im2col_s1", im2col_s1), ("prog_s1", prog_s1),
            ("im2col_s2", im2col_s2), ("prog_s2", prog_s2),
            ("prog_flatten_lin", prog_flatten_lin), ("encoder", enc_prog_addr),
            ("pred", pred_prog), ("tok", tok_prog), ("dur", dur_prog),
            ("restore", restore_prog),
        ]
        prog_end = engine.get_program_dram_addr()  # next free address = end of last program
        sized = {}
        for i, (name, addr) in enumerate(all_progs):
            next_addr = all_progs[i + 1][1] if i + 1 < len(all_progs) else prog_end
            sized[name] = (addr, next_addr - addr)
        engine.dump_programs(sized, L_pad)
    engine.progs = {"pred": (pred_prog, 0), "joint_tok": (tok_prog, 0), "joint_dur": (dur_prog, 0), "state_restore": (restore_prog, 0)}

    import threading

    def _progress_timer(label, start_time, stop_event):
        """Print updating elapsed time until stop_event is set."""
        while not stop_event.wait(1.0):
            elapsed = _time.perf_counter() - start_time
            _original_print(f"\r  {label} ({elapsed:.0f}s)", end="", flush=True)

    t_start = _time.perf_counter()

    stop = threading.Event()
    timer = threading.Thread(target=_progress_timer, args=("Executing", t_start, stop), daemon=True)
    timer.start()

    # --- Subsampling ---
    mel_flat = mel.squeeze(0).to(torch.bfloat16).contiguous()
    engine.dma_to_accelerator_memory(engine.MEL_DRAM, mel_flat)
    engine.dma_to_accelerator_memory(engine.SUB_OUT0_DRAM, torch.zeros(N0 * SC, dtype=torch.bfloat16))
    engine.program_execute(im2col_s0)
    engine.program_execute(prog_s0)
    engine.program_execute(im2col_s1)
    engine.program_execute(prog_s1)
    engine.program_execute(im2col_s2)
    engine.program_execute(prog_s2)
    engine.dma_to_accelerator_memory(engine.INPUT_DRAM, torch.zeros(L_pad * D, dtype=torch.bfloat16))
    engine.program_execute(prog_flatten_lin)
    if L_pad > H2:
        ef = torch.zeros((L_pad - H2) * D, dtype=torch.bfloat16)
        ef[0::2] = 0.1; ef[1::2] = -0.1
        engine.dma_to_accelerator_memory(engine.INPUT_DRAM + H2 * D * bpe, ef)
    t_sub_done = _time.perf_counter()

    # --- Encoder ---
    engine.start_execute_from_dram(enc_prog_addr)
    engine.wait_queue(120.0)
    t_enc_done = _time.perf_counter()

    # --- Decoder ---
    hw_enc_out = read_dram(engine, engine.INPUT_DRAM, L_pad * D)
    engine.dma_to_accelerator_memory(engine.ENC_OUT_DRAM, hw_enc_out.contiguous())
    hw_tokens = engine.run_decode(engine.ENC_OUT_DRAM, L)
    t_dec_done = _time.perf_counter()

    stop.set()
    timer.join()
    _original_print(f"\r  Executing ({t_dec_done - t_start:.0f}s) done")

    # ==================================================================
    # Results
    # ==================================================================
    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER_PATH)
    vocab_sz = sp.GetPieceSize()
    hw_text = sp.DecodeIds([t for t in hw_tokens if 0 <= t < vocab_sz])

    total = t_dec_done - t_start

    _original_print(f"\n  >>> {hw_text}")
    _original_print(f"\n  Timing:")
    _original_print(f"    Subsampling:         {t_sub_done - t_start:.3f}s")
    _original_print(f"    Encoder (24 layers): {t_enc_done - t_sub_done:.3f}s")
    _original_print(f"    Decoder:             {t_dec_done - t_enc_done:.3f}s")
    _original_print(f"    Total:               {total:.3f}s")

if __name__ == "__main__":
    main()