#!/usr/bin/env python3
"""
Parakeet-TDT-0.6B inference on accelerator.

  - Config from parakeet_config.json; weights from NeMo checkpoint.
  - Mel spectrogram computed on host CPU.
  - Encoder (subsampling + 24x conformer) compiled and run on accelerator.
  - TDT decode loop (LSTM predictor + joint network) run on accelerator.

Usage:
  python parakeet_test.py
  python parakeet_test.py --audio test.wav
  python parakeet_test.py --dev xdma0 [--cycle 5.63]

Fixed layout: parakeet_test.py, parakeet_config.json, and parakeet_bin/ live in the same folder.
  user_dma_core.py is one folder up; that parent is added to sys.path.
"""

import json
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import numpy as np
import torch
import torch.nn.functional as F

import struct
import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, DMA_DEVICE_C2H, DRAM_INSTRUCTION_ADDR, TYPE, UE_VECTOR_SIZE,
    URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS, UnifiedEngine, set_dma_device,
    UE_MODE, BROADCAST_MODE, LALU_MODE, MEMCPY_TYPE, URAM_SECTION,
)

URAM_A_BASE = 0x00000
URAM_B_BASE = 0x80000
EPS = 1e-5
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
def batch_norm_core_dram(ue: UnifiedEngine, C: int, L: int,
                         A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                         SCALE_DRAM_ADDR: int, SHIFT_DRAM_ADDR: int) -> None:
    """Emit instructions for fused eval-mode batch norm on (C, L) tensor.
    For each channel c:
        output[c, :] = input[c, :] * scale[c] + shift[c]
    Uses broadcast_mul + broadcast_add per channel row.
    """
    assert L % UE_VECTOR_SIZE == 0, f"L={L} must be a multiple of {UE_VECTOR_SIZE}"
    row_bytes = L * 2
    # Read scale/shift tensors from DRAM to host to extract per-channel bf16 scalars
    scale_host = torch.zeros(C, dtype=torch.bfloat16)
    shift_host = torch.zeros(C, dtype=torch.bfloat16)
    ue.dma_read(DMA_DEVICE_C2H, SCALE_DRAM_ADDR, scale_host, C * 2)
    ue.dma_read(DMA_DEVICE_C2H, SHIFT_DRAM_ADDR, shift_host, C * 2)
    # Convert bf16 tensors to raw uint16 array for broadcast scalars
    scale_raw = scale_host.view(torch.int16).numpy()
    shift_raw = shift_host.view(torch.int16).numpy()
    for c in range(C):
        ue.accelerator_memory_to_sram(A_DRAM_ADDR + c * row_bytes, URAM_A_BASE, L)
        ue.broadcast_mul(scalar=int(scale_raw[c]) & 0xFFFF, sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=L)
        ue.broadcast_add(scalar=int(shift_raw[c]) & 0xFFFF, sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=L)
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
    scalar_2 = ue.float_to_bf16(2.0)
    scalar_neg1 = ue.float_to_bf16(-1.0)
    # Step 1: 2x → OUTPUT as temp
    for m in range(M):
        ue.accelerator_memory_to_sram(A_DRAM_ADDR + m * row_bytes, URAM_A_BASE, N)
        ue.broadcast_mul(scalar=scalar_2, sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=N)
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
        ue.broadcast_mul(scalar=scalar_2, sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=N)
        ue.broadcast_add(scalar=scalar_neg1, sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=N)
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
    row_bytes = N * 2
    # Step 1: copy input to OUTPUT (so we can sigmoid in-place there)
    for m in range(M):
        ue.accelerator_memory_to_sram(A_DRAM_ADDR + m * row_bytes, URAM_A_BASE, N)
        ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR + m * row_bytes, N)
    # Step 2: sigmoid(x) via identity matmul with LALU sigmoid → OUTPUT in-place
    ue.matmat_mul_core(M=M, K=N, N=N, A_DRAM_ADDR=OUTPUT_DRAM_ADDR, B_DRAM_ADDR=IDENTITY_DRAM_ADDR, OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR, sigmoid_enable=True)
    # Step 3: x * sigmoid(x) row-by-row
    for m in range(M):
        ue.accelerator_memory_to_sram(A_DRAM_ADDR + m * row_bytes, URAM_A_BASE, N)
        ue.accelerator_memory_to_sram(OUTPUT_DRAM_ADDR + m * row_bytes, URAM_B_BASE, N)
        ue.eltwise_mul_core(vector_A_sram_start_addr=URAM_A_BASE, vector_B_sram_start_addr=URAM_B_BASE, vector_C_sram_wb_addr=URAM_A_BASE, element_size=N)
        ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR + m * row_bytes, N)
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
    row_bytes = C * 2
    # Step 1: sigmoid(b) via identity matmul
    ue.matmat_mul_core(M=M, K=C, N=C, A_DRAM_ADDR=B_DRAM_ADDR, B_DRAM_ADDR=IDENTITY_DRAM_ADDR, OUTPUT_DRAM_ADDR=B_DRAM_ADDR, sigmoid_enable=True)
    # Step 2: a * sigmoid(b)
    for m in range(M):
        ue.accelerator_memory_to_sram(A_DRAM_ADDR + m * row_bytes, URAM_A_BASE, C)
        ue.accelerator_memory_to_sram(B_DRAM_ADDR + m * row_bytes, URAM_B_BASE, C)
        ue.eltwise_mul_core(vector_A_sram_start_addr=URAM_A_BASE, vector_B_sram_start_addr=URAM_B_BASE, vector_C_sram_wb_addr=URAM_A_BASE, element_size=C)
        ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR + m * row_bytes, C)
def rel_shift_core_dram(ue: UnifiedEngine, L: int, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int) -> None:
    """Emit instructions for rel_shift: extract (L, L) from (L, 2L-1) positional scores.
    Row i of output = input[i, (L-1-i) : (L-1-i)+L]
    Each is a contiguous L-element DMA copy at a computed source offset.
    No arithmetic — pure memory rearrangement.
    Args:
        ue: UnifiedEngine instance
        L: sequence length (L_pad, bucketed, must be multiple of UE_VECTOR_SIZE)
        INPUT_DRAM_ADDR: (L, 2L-1) bf16 positional score matrix
        OUTPUT_DRAM_ADDR: (L, L) bf16 output
    """
    assert L % UE_VECTOR_SIZE == 0, f"L={L} must be a multiple of {UE_VECTOR_SIZE}"
    P = 2 * L - 1   # input row width
    bpe = 2          # bf16
    for i in range(L):
        src = INPUT_DRAM_ADDR + (i * P + (L - 1 - i)) * bpe
        dst = OUTPUT_DRAM_ADDR + i * L * bpe
        ue.accelerator_memory_to_sram(src, URAM_A_BASE, L)
        ue.sram_to_accelerator_memory(URAM_A_BASE, dst, L)
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
    row_bytes = N * 2
    scalar_half = ue.float_to_bf16(0.5)
    for m in range(M):
        # Load ff_output into URAM_A, multiply by 0.5
        ue.accelerator_memory_to_sram(FF_DRAM_ADDR + m * row_bytes, URAM_A_BASE, N)
        ue.broadcast_mul(scalar=scalar_half, sram_start_addr=URAM_A_BASE,
                         sram_wb_addr=URAM_A_BASE, element_size=N)
        # Load residual into URAM_B, add
        ue.accelerator_memory_to_sram(RESIDUAL_DRAM_ADDR + m * row_bytes, URAM_B_BASE, N)
        ue.eltwise_add_core(vector_A_sram_start_addr=URAM_A_BASE,
                            vector_B_sram_start_addr=URAM_B_BASE,
                            vector_C_sram_wb_addr=URAM_A_BASE,
                            element_size=N)
        # Write result
        ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR + m * row_bytes, N)
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
# ---------------------------------------------------------------------------
# Host-side mel spectrogram (runs on CPU, not accelerator)
# ---------------------------------------------------------------------------
def compute_mel_spectrogram(waveform, cfg):
    """waveform: (B, samples) float32 → (B, T_mel, 128) bf16."""
    pre = cfg["preprocessing"]
    n_fft = pre["n_fft"]
    hop_length = pre["hop_length"]
    win_length = pre["win_length"]
    # TODO: load actual filterbank and window from weights
    fb = torch.zeros(1, pre["n_mels"], n_fft // 2 + 1)
    window = torch.hann_window(win_length)
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
class Parakeet_UnifiedEngine(UnifiedEngine):
    """UnifiedEngine subclass for Parakeet-TDT-0.6B."""

    def __init__(self, script_dir=None):
        super().__init__(BASE_ADDR=user_dma_core.UE_0_BASE_ADDR,
                         program_dram_base=DRAM_INSTRUCTION_ADDR)
        self.script_dir = script_dir or SCRIPT_DIR
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
    def _alloc_write(self, tensor):
        """Allocate params DRAM, write bf16 tensor. Returns DRAM address."""
        t = tensor.to(torch.bfloat16).contiguous()
        addr = self.get_params_dram_addr()
        nbytes = t.numel() * 2
        self.allocate_params_dram(nbytes)
        self.dma_write(DMA_DEVICE_H2C, addr, t, nbytes)
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

        # Identity matrices
        self.w["IDENTITY_1024"] = allocate_identity(self, D)
        self.w["IDENTITY_4096"] = allocate_identity(self, FF)
        self.w["IDENTITY_640"] = allocate_identity(self, self.joint_hidden)

        # Subsampling weights for hardware im2col
        # Stage 0: Conv2d(1->256, k=3, s=2, p=1)
        self.w["SUB_CONV0_W"], self.sub0_flat_k, self.sub0_padded_k = \
            self._stage_sub_conv2d(sd["encoder.pre_encode.conv.0.weight"], 256, 1, 3)
        self.w["SUB_CONV0_B"] = self._alloc_write(sd["encoder.pre_encode.conv.0.bias"])
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
        # Final linear
        self.w["SUB_OUT_W"] = self._alloc_write(sd["encoder.pre_encode.out.weight"])
        self.w["SUB_OUT_B"] = self._alloc_write(sd["encoder.pre_encode.out.bias"])

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
            la["FF1_W1"] = self._alloc_write(sd[f"{pfx}.feed_forward1.linear1.weight"])
            la["FF1_W2"] = self._alloc_write(sd[f"{pfx}.feed_forward1.linear2.weight"])
            la["FF2_W1"] = self._alloc_write(sd[f"{pfx}.feed_forward2.linear1.weight"])
            la["FF2_W2"] = self._alloc_write(sd[f"{pfx}.feed_forward2.linear2.weight"])
            la["ATTN_Q_W"] = self._alloc_write(sd[f"{pfx}.self_attn.linear_q.weight"])
            la["ATTN_K_W"] = self._alloc_write(sd[f"{pfx}.self_attn.linear_k.weight"])
            la["ATTN_V_W"] = self._alloc_write(sd[f"{pfx}.self_attn.linear_v.weight"])
            la["ATTN_POS_W"] = self._alloc_write(sd[f"{pfx}.self_attn.linear_pos.weight"])
            la["ATTN_OUT_W"] = self._alloc_write(sd[f"{pfx}.self_attn.linear_out.weight"])
            la["ATTN_BIAS_U"] = self._alloc_write(sd[f"{pfx}.self_attn.pos_bias_u"].reshape(-1))
            la["ATTN_BIAS_V"] = self._alloc_write(sd[f"{pfx}.self_attn.pos_bias_v"].reshape(-1))
            pw1_a, pw1_b = self._split_pw_conv1(sd[f"{pfx}.conv.pointwise_conv1.weight"])
            la["CONV_PW1A_W"], la["CONV_PW1B_W"] = pw1_a, pw1_b
            la["CONV_DW_W"] = self._stage_dw_conv1d(sd[f"{pfx}.conv.depthwise_conv.weight"])
            bn_w = sd[f"{pfx}.conv.batch_norm.weight"]
            bn_b = sd[f"{pfx}.conv.batch_norm.bias"]
            bn_m = sd[f"{pfx}.conv.batch_norm.running_mean"]
            bn_v = sd[f"{pfx}.conv.batch_norm.running_var"]
            s_addr, sh_addr, _ = batch_norm_fuse_params(self, bn_w, bn_b, bn_m, bn_v)
            la["CONV_BN_SCALE"], la["CONV_BN_SHIFT"] = s_addr, sh_addr
            la["CONV_PW2_W"] = self._alloc_write(sd[f"{pfx}.conv.pointwise_conv2.weight"].squeeze(-1))
            self.layer_addrs.append(la)
            print(f"  Layer {i:2d} staged")

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
        b_tok = torch.zeros(N_tok_pad, dtype=torch.bfloat16)
        b_tok[:N_tok] = out_b[:N_tok]
        self.w["JOINT_OUT_TOK_W"] = self._alloc_write(w_tok)
        self.w["JOINT_OUT_TOK_B"] = self._alloc_write(b_tok)
        w_dur = torch.zeros(N_dur_pad, self.joint_hidden, dtype=torch.bfloat16)
        w_dur[:N_dur] = out_w[N_tok:N_tok + N_dur]
        b_dur = torch.zeros(N_dur_pad, dtype=torch.bfloat16)
        b_dur[:N_dur] = out_b[N_tok:N_tok + N_dur]
        self.w["JOINT_OUT_DUR_W"] = self._alloc_write(w_dur)
        self.w["JOINT_OUT_DUR_B"] = self._alloc_write(b_dur)
        print(f"  Weight staging complete: {self.get_params_dram_usage() / 1024**2:.1f} MB in DRAM")
    def tensor_init(self, L_pad):
        """Allocate intermediate DRAM buffers."""
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
        self.POS_PROJ_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
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
        # Subsampling intermediates (host builds im2col patches, HW runs matmuls)
        T_mel_max = L_pad * 8
        H0, W0 = T_mel_max // 2, self.n_mels // 2  # after stage 0
        H1, W1 = H0 // 2, W0 // 2                  # after stage 1
        N0 = H0 * W0
        N1 = H1 * W1
        # Shared patch buffer: max(stage0 patches, stage1/2 DW patches)
        max_patch_size = max(N0 * 64, SC * N1 * 64)  # stage 0 or DW stage 1 (larger)
        self.SUB_PATCH_DRAM = self.allocate_tensor_dram(max_patch_size * bpe)
        self.SUB_OUT0_DRAM = self.allocate_tensor_dram(N0 * SC * bpe)  # stage 0 output
        self.SUB_DW_OUT_DRAM = self.allocate_tensor_dram(SC * N1 * bpe)  # DW output (reused)
        self.SUB_PW_IN_DRAM = self.allocate_tensor_dram(N1 * SC * bpe)  # after permute for PW
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
        self.JOINT_ENC_DRAM = self.allocate_tensor_dram(self.joint_hidden * bpe)
        self.JOINT_PRED_DRAM = self.allocate_tensor_dram(self.joint_hidden * bpe)
        self.JOINT_SUM_DRAM = self.allocate_tensor_dram(self.joint_hidden * bpe)
        self.JOINT_TOK_DRAM = self.allocate_tensor_dram(N_tok_pad * bpe)
        self.JOINT_DUR_DRAM = self.allocate_tensor_dram(N_dur_pad * bpe)
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
        DW im2col patches pre-built on host at dw_patch_addr as (SC, N_out, 64)."""
        bpe = self.bytes_per_element
        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()
        flops = 0
        # DW: per channel, kernel(1,64) @ patches(N_out,64)^T = (1, N_out)
        for ch in range(SC):
            kernel_addr = self.w[dw_w_key] + ch * 64 * bpe
            patch_addr = dw_patch_addr + ch * N_out * 64 * bpe
            out_addr = self.SUB_DW_OUT_DRAM + ch * N_out * bpe
            self.matmat_mul_core(M=1, K=64, N=N_out,
                A_DRAM_ADDR=kernel_addr, B_DRAM_ADDR=patch_addr, OUTPUT_DRAM_ADDR=out_addr)
            flops += 2 * 64 * N_out
        # DW bias: broadcast_add per channel
        dw_bias = torch.zeros(SC, dtype=torch.bfloat16)
        self.dma_read(DMA_DEVICE_C2H, self.w[dw_b_key], dw_bias, SC * bpe)
        dw_bias_raw = dw_bias.view(torch.int16).numpy()
        for ch in range(SC):
            out_addr = self.SUB_DW_OUT_DRAM + ch * N_out * bpe
            self.accelerator_memory_to_sram(out_addr, URAM_A_BASE, N_out)
            self.broadcast_add(scalar=int(dw_bias_raw[ch]) & 0xFFFF,
                sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=N_out)
            self.sram_to_accelerator_memory(URAM_A_BASE, out_addr, N_out)
        # Chunked on-device transpose: (SC, N_out) -> (N_out, SC)
        # Process N_out in chunks of SC (256) so each sub-transpose is (SC, SC) which fits URAM.
        chunk = SC  # 256
        for c_start in range(0, N_out, chunk):
            c_end = min(c_start + chunk, N_out)
            c_len = c_end - c_start
            c_len_pad = pad_to_multiple(c_len, self.block_size)
            # Extract (SC, c_len) slice from SUB_DW_OUT and transpose to (c_len_pad, SC)
            # Input: SC rows, each N_out wide, we want columns [c_start:c_end]
            # Use strided read per channel row: read c_len elements at offset c_start
            for ch in range(SC):
                src = self.SUB_DW_OUT_DRAM + (ch * N_out + c_start) * bpe
                # Write contiguously into temp as (SC, c_len_pad) for transpose
                dst = self.PERMUTE_TEMP_DRAM + ch * c_len_pad * bpe
                self.accelerator_memory_to_sram(src, URAM_A_BASE, c_len)
                self.sram_to_accelerator_memory(URAM_A_BASE, dst, c_len)
            # Transpose (SC, c_len_pad) -> (c_len_pad, SC) — both dims <= 256, fits URAM
            bf16_smart_permute_core(self,
                dims=[SC, c_len_pad], permute_indices=[1, 0],
                input_dram_addr=self.PERMUTE_TEMP_DRAM,
                output_dram_addr=self.SUB_PW_IN_DRAM + c_start * SC * bpe,
                params_dram_addr=self.w["IDENTITY_1024"],
                temp_dram_start=self.PERMUTE_TEMP_DRAM + SC * c_len_pad * bpe)
        # PW: (N_out, 256) @ (256, 256) + bias, ReLU
        self.matmat_mul_core(M=N_out, K=SC, N=SC,
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

    def compile_sub_linear(self, L_pad):
        """HW program: flatten linear (L_pad, 4096) @ (4096, 1024) + bias -> INPUT_DRAM."""
        D = self.d_model
        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()
        self.matmat_mul_core(M=L_pad, K=4096, N=D,
            A_DRAM_ADDR=self.SUB_FLAT_DRAM, B_DRAM_ADDR=self.w["SUB_OUT_W"],
            OUTPUT_DRAM_ADDR=self.INPUT_DRAM, C_DRAM_ADDR=self.w["SUB_OUT_B"])
        self.stop_capture()
        self.generate_instruction_halt()
        prog = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prog)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        return prog, 2 * L_pad * 4096 * D

    def _im2col_conv2d(self, input_tensor, H_in, W_in, stride=2, padding=1):
        """Host-side: build im2col patch matrix for 3x3 conv2d using F.unfold (vectorized).
        input_tensor: (H_in, W_in) or (C, H_in, W_in) bf16.
        Single-channel: returns (N_out, 64) bf16.  Depthwise: returns (C, N_out, 64).
        """
        k = 3
        if input_tensor.dim() == 2:
            # (H, W) -> (1, 1, H, W) for unfold
            x = input_tensor.float().unsqueeze(0).unsqueeze(0)
            # unfold: (1, 1*k*k, N_out) = (1, 9, N_out)
            cols = F.unfold(x, k, padding=padding, stride=stride)  # (1, 9, N)
            H_out = (H_in + 2*padding - k) // stride + 1
            W_out = (W_in + 2*padding - k) // stride + 1
            N = cols.shape[2]
            patches = torch.zeros(N, 64, dtype=torch.bfloat16)
            patches[:, :9] = cols[0].t().to(torch.bfloat16)  # (N, 9)
            return patches, H_out, W_out
        else:
            # (C, H, W) -> depthwise: treat each channel as (1, 1, H, W)
            C = input_tensor.shape[0]
            x = input_tensor.float().unsqueeze(1)  # (C, 1, H, W)
            cols = F.unfold(x.reshape(C, 1, H_in, W_in), k, padding=padding, stride=stride)  # (C, 9, N)
            H_out = (H_in + 2*padding - k) // stride + 1
            W_out = (W_in + 2*padding - k) // stride + 1
            N = cols.shape[2]
            patches = torch.zeros(C, N, 64, dtype=torch.bfloat16)
            patches[:, :, :9] = cols.permute(0, 2, 1).to(torch.bfloat16)  # (C, N, 9)
            return patches, H_out, W_out

    def compile_encoder(self, L_pad):
        """Monolithic encoder compile: subsampling + 24 conformer layers inline. Returns (prog, flops)."""
        D, FF, H_heads, dk = self.d_model, self.ff_dim, self.num_heads, self.head_dim
        bpe = self.bytes_per_element
        P = 2 * L_pad - 1
        P_pad = pad_to_multiple(P, self.block_size)

        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()
        total_flops = 0

        # --- 24 conformer layers, all ops inline ---
        for layer_idx in range(self.num_layers):
            la = self.layer_addrs[layer_idx]

            # ===== FF1 (half-step residual) =====
            self.layer_norm_core_dram(M=L_pad, N=D,
                A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.LN_OUT_DRAM,
                GAMMA_DRAM_ADDR=la["LN_FF1_WEIGHT"], BETA_DRAM_ADDR=la["LN_FF1_BIAS"])
            self.matmat_mul_core(M=L_pad, K=D, N=FF,
                A_DRAM_ADDR=self.LN_OUT_DRAM, B_DRAM_ADDR=la["FF1_W1"],
                OUTPUT_DRAM_ADDR=self.FF_MID_DRAM, silu_enable=True)
            total_flops += 2 * L_pad * D * FF
            self.matmat_mul_core(M=L_pad, K=FF, N=D,
                A_DRAM_ADDR=self.FF_MID_DRAM, B_DRAM_ADDR=la["FF1_W2"],
                OUTPUT_DRAM_ADDR=self.FF_OUT_DRAM)
            total_flops += 2 * L_pad * FF * D
            half_step_residual_core_dram(self, M=L_pad, N=D,
                RESIDUAL_DRAM_ADDR=self.INPUT_DRAM, FF_DRAM_ADDR=self.FF_OUT_DRAM,
                OUTPUT_DRAM_ADDR=self.INPUT_DRAM)
            total_flops += 2 * L_pad * D

            # ===== Self-attention =====
            self.layer_norm_core_dram(M=L_pad, N=D,
                A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.LN_OUT_DRAM,
                GAMMA_DRAM_ADDR=la["LN_ATTN_WEIGHT"], BETA_DRAM_ADDR=la["LN_ATTN_BIAS"])
            self.matmat_mul_core(M=L_pad, K=D, N=D,
                A_DRAM_ADDR=self.LN_OUT_DRAM, B_DRAM_ADDR=la["ATTN_Q_W"], OUTPUT_DRAM_ADDR=self.Q_DRAM)
            self.matmat_mul_core(M=L_pad, K=D, N=D,
                A_DRAM_ADDR=self.LN_OUT_DRAM, B_DRAM_ADDR=la["ATTN_K_W"], OUTPUT_DRAM_ADDR=self.K_DRAM)
            self.matmat_mul_core(M=L_pad, K=D, N=D,
                A_DRAM_ADDR=self.LN_OUT_DRAM, B_DRAM_ADDR=la["ATTN_V_W"], OUTPUT_DRAM_ADDR=self.V_DRAM)
            total_flops += 3 * 2 * L_pad * D * D
            self.matmat_mul_core(M=P_pad, K=D, N=D,
                A_DRAM_ADDR=self.POS_EMB_DRAM, B_DRAM_ADDR=la["ATTN_POS_W"],
                OUTPUT_DRAM_ADDR=self.POS_PROJ_DRAM)
            total_flops += 2 * P_pad * D * D

            for h in range(H_heads):
                h_off = h * dk * bpe
                # Q_h + bias_u
                self.accelerator_memory_to_sram(self.Q_DRAM + h_off, URAM_A_BASE, L_pad * dk,
                    stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                self.accelerator_memory_to_sram(la["ATTN_BIAS_U"] + h_off, URAM_B_BASE, dk)
                for row in range(L_pad):
                    self.eltwise_add_core(URAM_A_BASE + row * dk * bpe, URAM_B_BASE,
                        URAM_A_BASE + row * dk * bpe, dk)
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
                # Positional: Q_h + bias_v
                self.accelerator_memory_to_sram(self.Q_DRAM + h_off, URAM_A_BASE, L_pad * dk,
                    stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                self.accelerator_memory_to_sram(la["ATTN_BIAS_V"] + h_off, URAM_B_BASE, dk)
                for row in range(L_pad):
                    self.eltwise_add_core(URAM_A_BASE + row * dk * bpe, URAM_B_BASE,
                        URAM_A_BASE + row * dk * bpe, dk)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_A_DRAM, L_pad * dk)
                # P_h
                self.accelerator_memory_to_sram(self.POS_PROJ_DRAM + h_off, URAM_A_BASE, P_pad * dk,
                    stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_B_DRAM, P_pad * dk)
                self.matmat_mul_core(M=L_pad, K=dk, N=P_pad,
                    A_DRAM_ADDR=self.CONV_A_DRAM, B_DRAM_ADDR=self.CONV_B_DRAM,
                    OUTPUT_DRAM_ADDR=self.CONV_OUT_DRAM)
                total_flops += 2 * L_pad * dk * P_pad
                rel_shift_core_dram(self, L=L_pad,
                    INPUT_DRAM_ADDR=self.CONV_OUT_DRAM, OUTPUT_DRAM_ADDR=self.REL_SHIFT_DRAM)
                # Combine content + pos
                score_elems = L_pad * L_pad
                self.accelerator_memory_to_sram(self.SCORE_DRAM, URAM_A_BASE, score_elems)
                self.accelerator_memory_to_sram(self.REL_SHIFT_DRAM, URAM_B_BASE, score_elems)
                self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, score_elems)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.SCORE_DRAM, score_elems)
                # Scale
                self.accelerator_memory_to_sram(self.SCORE_DRAM, URAM_A_BASE, score_elems)
                inv_sqrt_dk = self.float_to_bf16(1.0 / math.sqrt(dk))
                for row in range(L_pad):
                    self.broadcast_mul(scalar=inv_sqrt_dk,
                        sram_start_addr=URAM_A_BASE + row * L_pad * bpe,
                        sram_wb_addr=URAM_A_BASE + row * L_pad * bpe, element_size=L_pad)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.SCORE_DRAM, score_elems)
                # Softmax + mask
                self.matmat_mul_core(M=L_pad, K=L_pad, N=L_pad,
                    A_DRAM_ADDR=self.SCORE_DRAM, B_DRAM_ADDR=self.IDENTITY_LPAD_DRAM,
                    OUTPUT_DRAM_ADDR=self.SCORE_DRAM, softmax_enable=True,
                    C_DRAM_ADDR=self.ATTN_MASK_DRAM, bias_mode="full_matrix")
                # V_h -> transpose -> attn @ V_h
                self.accelerator_memory_to_sram(self.V_DRAM + h_off, URAM_A_BASE, L_pad * dk,
                    stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_B_DRAM, L_pad * dk)
                bf16_smart_permute_core(self, dims=[L_pad, dk], permute_indices=[1, 0],
                    input_dram_addr=self.CONV_B_DRAM, output_dram_addr=self.ATTN_VT_DRAM,
                    params_dram_addr=self.w["IDENTITY_1024"], temp_dram_start=self.PERMUTE_TEMP_DRAM)
                self.matmat_mul_core(M=L_pad, K=L_pad, N=dk,
                    A_DRAM_ADDR=self.SCORE_DRAM, B_DRAM_ADDR=self.ATTN_VT_DRAM,
                    OUTPUT_DRAM_ADDR=self.CONV_A_DRAM)
                total_flops += 2 * L_pad * L_pad * dk
                # Write head output (strided)
                self.accelerator_memory_to_sram(self.CONV_A_DRAM, URAM_A_BASE, L_pad * dk)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.ATTN_OUT_DRAM + h_off, L_pad * dk,
                    stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
            # Output projection + residual
            self.matmat_mul_core(M=L_pad, K=D, N=D,
                A_DRAM_ADDR=self.ATTN_OUT_DRAM, B_DRAM_ADDR=la["ATTN_OUT_W"],
                OUTPUT_DRAM_ADDR=self.FF_OUT_DRAM)
            total_flops += 2 * L_pad * D * D
            self.accelerator_memory_to_sram(self.INPUT_DRAM, URAM_A_BASE, L_pad * D)
            self.accelerator_memory_to_sram(self.FF_OUT_DRAM, URAM_B_BASE, L_pad * D)
            self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * D)
            self.sram_to_accelerator_memory(URAM_A_BASE, self.INPUT_DRAM, L_pad * D)
            total_flops += L_pad * D
            # ===== Conv module =====
            self.layer_norm_core_dram(M=L_pad, N=D,
                A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.LN_OUT_DRAM,
                GAMMA_DRAM_ADDR=la["LN_CONV_WEIGHT"], BETA_DRAM_ADDR=la["LN_CONV_BIAS"])
            self.matmat_mul_core(M=L_pad, K=D, N=D,
                A_DRAM_ADDR=self.LN_OUT_DRAM, B_DRAM_ADDR=la["CONV_PW1A_W"],
                OUTPUT_DRAM_ADDR=self.CONV_A_DRAM)
            self.matmat_mul_core(M=L_pad, K=D, N=D,
                A_DRAM_ADDR=self.LN_OUT_DRAM, B_DRAM_ADDR=la["CONV_PW1B_W"],
                OUTPUT_DRAM_ADDR=self.CONV_B_DRAM)
            total_flops += 2 * 2 * L_pad * D * D
            glu_core_dram(self, M=L_pad, C=D,
                A_DRAM_ADDR=self.CONV_A_DRAM, B_DRAM_ADDR=self.CONV_B_DRAM,
                OUTPUT_DRAM_ADDR=self.CONV_A_DRAM, IDENTITY_DRAM_ADDR=self.w["IDENTITY_1024"])
            bf16_smart_permute_core(self, dims=[L_pad, D], permute_indices=[1, 0],
                input_dram_addr=self.CONV_A_DRAM, output_dram_addr=self.CONV_T_DRAM,
                params_dram_addr=self.w["IDENTITY_1024"], temp_dram_start=self.PERMUTE_TEMP_DRAM)
            # DW Conv1d via shift-multiply-accumulate (9 taps, padding=4)
            # For each channel: output[t] = sum_{k=0}^{8} kernel[k] * input[t + k - 4]
            # Read kernel scalars to host for broadcast_mul
            dw_kernel_host = torch.zeros(D, 64, dtype=torch.bfloat16)
            self.dma_read(DMA_DEVICE_C2H, la["CONV_DW_W"], dw_kernel_host, D * 64 * bpe)
            dw_kernel_raw = dw_kernel_host.view(torch.int16).numpy().reshape(D, 64)
            conv_pad = self.conv_pad  # 4
            K = 2 * conv_pad + 1     # 9
            for ch in range(D):
                ch_in = self.CONV_T_DRAM + ch * L_pad * bpe
                ch_out = self.CONV_DW_DRAM + ch * L_pad * bpe
                for k in range(K):
                    offset = k - conv_pad  # -4 to +4
                    # Source: input[max(0, offset) : min(L_pad, L_pad+offset)]
                    # Dest:   output[max(0, -offset) : min(L_pad, L_pad-offset)]
                    src_start = max(0, offset)
                    dst_start = max(0, -offset)
                    length = L_pad - abs(offset)
                    if length <= 0:
                        continue
                    length_aligned = pad_to_multiple(length, self.block_size)
                    # Read shifted input to SRAM_A
                    self.accelerator_memory_to_sram(ch_in + src_start * bpe, URAM_A_BASE, length)
                    # Multiply by kernel[k] scalar
                    scalar = int(dw_kernel_raw[ch, k]) & 0xFFFF
                    self.broadcast_mul(scalar=scalar, sram_start_addr=URAM_A_BASE,
                        sram_wb_addr=URAM_A_BASE, element_size=length)
                    if k == 0:
                        # First tap: write directly to output
                        self.sram_to_accelerator_memory(URAM_A_BASE, ch_out + dst_start * bpe, length)
                    else:
                        # Accumulate: read existing output, add, write back
                        self.accelerator_memory_to_sram(ch_out + dst_start * bpe, URAM_B_BASE, length)
                        self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, length)
                        self.sram_to_accelerator_memory(URAM_A_BASE, ch_out + dst_start * bpe, length)
                total_flops += 2 * K * L_pad
            batch_norm_core_dram(self, C=D, L=L_pad,
                A_DRAM_ADDR=self.CONV_DW_DRAM, OUTPUT_DRAM_ADDR=self.CONV_DW_DRAM,
                SCALE_DRAM_ADDR=la["CONV_BN_SCALE"], SHIFT_DRAM_ADDR=la["CONV_BN_SHIFT"])
            silu_core_dram(self, M=D, N=L_pad,
                A_DRAM_ADDR=self.CONV_DW_DRAM, OUTPUT_DRAM_ADDR=self.CONV_DW_DRAM,
                IDENTITY_DRAM_ADDR=self.IDENTITY_LPAD_DRAM)
            bf16_smart_permute_core(self, dims=[D, L_pad], permute_indices=[1, 0],
                input_dram_addr=self.CONV_DW_DRAM, output_dram_addr=self.CONV_T_DRAM,
                params_dram_addr=self.w["IDENTITY_1024"], temp_dram_start=self.PERMUTE_TEMP_DRAM)
            self.matmat_mul_core(M=L_pad, K=D, N=D,
                A_DRAM_ADDR=self.CONV_T_DRAM, B_DRAM_ADDR=la["CONV_PW2_W"],
                OUTPUT_DRAM_ADDR=self.CONV_OUT_DRAM)
            total_flops += 2 * L_pad * D * D
            self.accelerator_memory_to_sram(self.INPUT_DRAM, URAM_A_BASE, L_pad * D)
            self.accelerator_memory_to_sram(self.CONV_OUT_DRAM, URAM_B_BASE, L_pad * D)
            self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * D)
            self.sram_to_accelerator_memory(URAM_A_BASE, self.INPUT_DRAM, L_pad * D)
            total_flops += L_pad * D

            # ===== FF2 (half-step residual) =====
            self.layer_norm_core_dram(M=L_pad, N=D,
                A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.LN_OUT_DRAM,
                GAMMA_DRAM_ADDR=la["LN_FF2_WEIGHT"], BETA_DRAM_ADDR=la["LN_FF2_BIAS"])
            self.matmat_mul_core(M=L_pad, K=D, N=FF,
                A_DRAM_ADDR=self.LN_OUT_DRAM, B_DRAM_ADDR=la["FF2_W1"],
                OUTPUT_DRAM_ADDR=self.FF_MID_DRAM, silu_enable=True)
            total_flops += 2 * L_pad * D * FF
            self.matmat_mul_core(M=L_pad, K=FF, N=D,
                A_DRAM_ADDR=self.FF_MID_DRAM, B_DRAM_ADDR=la["FF2_W2"],
                OUTPUT_DRAM_ADDR=self.FF_OUT_DRAM)
            total_flops += 2 * L_pad * FF * D
            half_step_residual_core_dram(self, M=L_pad, N=D,
                RESIDUAL_DRAM_ADDR=self.INPUT_DRAM, FF_DRAM_ADDR=self.FF_OUT_DRAM,
                OUTPUT_DRAM_ADDR=self.INPUT_DRAM)
            total_flops += 2 * L_pad * D

            # ===== Final LayerNorm =====
            self.layer_norm_core_dram(M=L_pad, N=D,
                A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.INPUT_DRAM,
                GAMMA_DRAM_ADDR=la["LN_OUT_WEIGHT"], BETA_DRAM_ADDR=la["LN_OUT_BIAS"])

        # Copy final encoder output
        self.accelerator_memory_to_sram(self.INPUT_DRAM, URAM_A_BASE, L_pad * D)
        self.sram_to_accelerator_memory(URAM_A_BASE, self.ENC_OUT_DRAM, L_pad * D)

        self.stop_capture()
        self.generate_instruction_halt()
        program_addr = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(program_addr)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        return program_addr, total_flops

    def compile_decoder(self):
        """Compile predictor LSTM + split joint. Returns (pred_prog, tok_prog, dur_prog, flops)."""
        H = self.pred_hidden
        D = self.d_model
        bpe = self.bytes_per_element
        N_tok_pad = pad_to_multiple(self.vocab_size, self.block_size)
        N_dur_pad = pad_to_multiple(len(self.tdt_durations), self.block_size)
        total_flops = 0

        # --- Predictor program ---
        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()

        for i in range(2):
            h_addr = self.PRED_H0_DRAM if i == 0 else self.PRED_H1_DRAM
            c_addr = self.PRED_C0_DRAM if i == 0 else self.PRED_C1_DRAM
            x_addr = self.PRED_EMB_DRAM if i == 0 else self.PRED_H0_DRAM
            self.matmat_mul_core(M=1, K=H, N=4*H,
                A_DRAM_ADDR=x_addr, B_DRAM_ADDR=self.w[f"LSTM_WIH{i}"],
                OUTPUT_DRAM_ADDR=self.PRED_GATES_DRAM, C_DRAM_ADDR=self.w[f"LSTM_BIH{i}"])
            total_flops += 2 * H * 4 * H
            self.matmat_mul_core(M=1, K=H, N=4*H,
                A_DRAM_ADDR=h_addr, B_DRAM_ADDR=self.w[f"LSTM_WHH{i}"],
                OUTPUT_DRAM_ADDR=self.PRED_GATES2_DRAM, C_DRAM_ADDR=self.w[f"LSTM_BHH{i}"])
            total_flops += 2 * H * 4 * H
            self.accelerator_memory_to_sram(self.PRED_GATES_DRAM, URAM_A_BASE, 4*H)
            self.accelerator_memory_to_sram(self.PRED_GATES2_DRAM, URAM_B_BASE, 4*H)
            self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, 4*H)
            self.sram_to_accelerator_memory(URAM_A_BASE, self.PRED_GATES_DRAM, 4*H)
            # i sigmoid
            self.matmat_mul_core(M=1, K=H, N=H,
                A_DRAM_ADDR=self.PRED_GATES_DRAM, B_DRAM_ADDR=self.w["IDENTITY_640"],
                OUTPUT_DRAM_ADDR=self.PRED_GATES_DRAM, sigmoid_enable=True)
            # f sigmoid
            self.matmat_mul_core(M=1, K=H, N=H,
                A_DRAM_ADDR=self.PRED_GATES_DRAM + H * bpe, B_DRAM_ADDR=self.w["IDENTITY_640"],
                OUTPUT_DRAM_ADDR=self.PRED_GATES_DRAM + H * bpe, sigmoid_enable=True)
            # g tanh
            tanh_core_dram(self, M=1, N=H,
                A_DRAM_ADDR=self.PRED_GATES_DRAM + 2 * H * bpe,
                OUTPUT_DRAM_ADDR=self.PRED_GATES_DRAM + 2 * H * bpe,
                IDENTITY_DRAM_ADDR=self.w["IDENTITY_640"])
            # o sigmoid
            self.matmat_mul_core(M=1, K=H, N=H,
                A_DRAM_ADDR=self.PRED_GATES_DRAM + 3 * H * bpe, B_DRAM_ADDR=self.w["IDENTITY_640"],
                OUTPUT_DRAM_ADDR=self.PRED_GATES_DRAM + 3 * H * bpe, sigmoid_enable=True)
            # c_new = f*c + i*g
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
            # h_new = o * tanh(c_new)
            tanh_core_dram(self, M=1, N=H,
                A_DRAM_ADDR=c_addr, OUTPUT_DRAM_ADDR=self.PRED_OUT_DRAM,
                IDENTITY_DRAM_ADDR=self.w["IDENTITY_640"])
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
        # --- Joint token program ---
        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()
        self.matmat_mul_core(M=1, K=D, N=H,
            A_DRAM_ADDR=self.JOINT_ENC_DRAM, B_DRAM_ADDR=self.w["JOINT_ENC_W"],
            OUTPUT_DRAM_ADDR=self.JOINT_ENC_DRAM, C_DRAM_ADDR=self.w["JOINT_ENC_B"])
        total_flops += 2 * D * H
        self.matmat_mul_core(M=1, K=H, N=H,
            A_DRAM_ADDR=self.JOINT_PRED_DRAM, B_DRAM_ADDR=self.w["JOINT_PRED_W"],
            OUTPUT_DRAM_ADDR=self.JOINT_PRED_DRAM, C_DRAM_ADDR=self.w["JOINT_PRED_B"])
        total_flops += 2 * H * H
        self.accelerator_memory_to_sram(self.JOINT_ENC_DRAM, URAM_A_BASE, H)
        self.accelerator_memory_to_sram(self.JOINT_PRED_DRAM, URAM_B_BASE, H)
        self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, H)
        self.sram_to_accelerator_memory(URAM_A_BASE, self.JOINT_SUM_DRAM, H)
        self.matmat_mul_core(M=1, K=H, N=H,
            A_DRAM_ADDR=self.JOINT_SUM_DRAM, B_DRAM_ADDR=self.w["IDENTITY_640"],
            OUTPUT_DRAM_ADDR=self.JOINT_SUM_DRAM, relu_enable=True)
        self.matmat_mul_core(M=1, K=H, N=N_tok_pad,
            A_DRAM_ADDR=self.JOINT_SUM_DRAM, B_DRAM_ADDR=self.w["JOINT_OUT_TOK_W"],
            OUTPUT_DRAM_ADDR=self.JOINT_TOK_DRAM, C_DRAM_ADDR=self.w["JOINT_OUT_TOK_B"])
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

        self.matmat_mul_core(M=1, K=H, N=N_dur_pad,
            A_DRAM_ADDR=self.JOINT_SUM_DRAM, B_DRAM_ADDR=self.w["JOINT_OUT_DUR_W"],
            OUTPUT_DRAM_ADDR=self.JOINT_DUR_DRAM, C_DRAM_ADDR=self.w["JOINT_OUT_DUR_B"])
        total_flops += 2 * H * N_dur_pad

        self.stop_capture()
        self.generate_instruction_halt()
        dur_prog = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(dur_prog)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())

        return pred_prog, tok_prog, dur_prog, total_flops

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

    def get_arg_max_index(self):
        """Read hardware argmax register."""
        return self.read_reg32(user_dma_core.UE_ARGMAX_INDEX)

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
    # Compile all programs at once
    # -----------------------------------------------------------------------

    def compile_all(self, T_mel, L_pad):
        """Compile all programs up front: subsampling stages, encoder, decoder.
        Stores program addresses in self.progs dict.
        """
        SC = self.sub_channels
        D = self.d_model
        n_mels = self.n_mels
        H0, W0 = T_mel // 2, n_mels // 2
        H1, W1 = H0 // 2, W0 // 2
        H2, W2 = H1 // 2, W1 // 2
        N0, N1, N2 = H0 * W0, H1 * W1, H2 * W2
        self.progs = {}
        total_compile_flops = 0

        # Subsampling stage 0: im2col matmul + bias + ReLU
        prog, f = self.compile_sub_stage0(N0, 64, SC)
        self.progs["sub0"] = (prog, f)
        total_compile_flops += f
        print(f"  Compiled sub_stage0: {f/1e6:.1f}M flops")

        # Subsampling stage 1: DW + PW
        prog, f = self.compile_sub_stage_dw_pw(H0*W0, N1, SC,
            self.SUB_PATCH_DRAM, "SUB_DW1_W", "SUB_DW1_B", "SUB_PW1_W", "SUB_PW1_B")
        self.progs["sub1"] = (prog, f)
        total_compile_flops += f
        print(f"  Compiled sub_stage1: {f/1e6:.1f}M flops")

        # Subsampling stage 2: DW + PW
        prog, f = self.compile_sub_stage_dw_pw(H1*W1, N2, SC,
            self.SUB_PATCH_DRAM, "SUB_DW2_W", "SUB_DW2_B", "SUB_PW2_W", "SUB_PW2_B")
        self.progs["sub2"] = (prog, f)
        total_compile_flops += f
        print(f"  Compiled sub_stage2: {f/1e6:.1f}M flops")

        # Subsampling linear
        prog, f = self.compile_sub_linear(L_pad)
        self.progs["sub_lin"] = (prog, f)
        total_compile_flops += f
        print(f"  Compiled sub_linear: {f/1e6:.1f}M flops")

        # Encoder (24 conformer layers)
        prog, f = self.compile_encoder(L_pad)
        self.progs["encoder"] = (prog, f)
        total_compile_flops += f
        print(f"  Compiled encoder: {f/1e9:.1f}G flops, "
              f"{self.get_capture_instruction_size_bytes()/1024:.0f}KB")

        # Decoder (predictor + split joint)
        pred_prog, tok_prog, dur_prog, f = self.compile_decoder()
        self.progs["pred"] = (pred_prog, f)
        self.progs["joint_tok"] = (tok_prog, 0)
        self.progs["joint_dur"] = (dur_prog, 0)
        total_compile_flops += f
        print(f"  Compiled decoder: {f/1e6:.1f}M flops")

        print(f"  All programs compiled. Total flops: {total_compile_flops/1e9:.1f}G")

    # -----------------------------------------------------------------------
    # Execute: encoder
    # -----------------------------------------------------------------------

    def run_encoder(self, mel_bf16, L, L_pad):
        """Execute encoder: subsampling (host im2col + HW matmuls) + conformer layers.
        All programs already compiled via compile_all().
        Returns enc_out_addr.
        """
        bpe = self.bytes_per_element
        SC = self.sub_channels
        D = self.d_model
        T_mel = mel_bf16.shape[1]
        mel_2d = mel_bf16.squeeze(0)  # (T_mel, 128)

        # --- Subsampling stage 0 ---
        patches0, H0, W0 = self._im2col_conv2d(mel_2d, T_mel, self.n_mels)
        self.dma_to_accelerator_memory(self.SUB_PATCH_DRAM, patches0.contiguous())
        self.program_execute(self.progs["sub0"][0])

        # Read stage 0 output, build im2col for stage 1
        N0 = H0 * W0
        out0 = torch.zeros(N0 * SC, dtype=torch.bfloat16)
        self.dma_read(DMA_DEVICE_C2H, self.SUB_OUT0_DRAM, out0, N0 * SC * bpe)
        out0_3d = out0.reshape(H0, W0, SC).permute(2, 0, 1).contiguous()

        # --- Subsampling stage 1 ---
        dw_patches1, H1, W1 = self._im2col_conv2d(out0_3d, H0, W0)
        self.dma_to_accelerator_memory(self.SUB_PATCH_DRAM, dw_patches1.contiguous())
        self.program_execute(self.progs["sub1"][0])

        # Read stage 1 output, build im2col for stage 2
        N1 = H1 * W1
        out1 = torch.zeros(N1 * SC, dtype=torch.bfloat16)
        self.dma_read(DMA_DEVICE_C2H, self.SUB_PW_IN_DRAM, out1, N1 * SC * bpe)
        out1_3d = out1.reshape(H1, W1, SC).permute(2, 0, 1).contiguous()

        # --- Subsampling stage 2 ---
        dw_patches2, H2, W2 = self._im2col_conv2d(out1_3d, H1, W1)
        self.dma_to_accelerator_memory(self.SUB_PATCH_DRAM, dw_patches2.contiguous())
        self.program_execute(self.progs["sub2"][0])

        # Read stage 2 output, flatten to (L_pad, 4096)
        N2 = H2 * W2
        out2 = torch.zeros(N2 * SC, dtype=torch.bfloat16)
        self.dma_read(DMA_DEVICE_C2H, self.SUB_PW_IN_DRAM, out2, N2 * SC * bpe)
        # Reference: x.permute(0,2,1,3).reshape(B,T,C*Freq) -> (H2, SC, W2) -> (H2, SC*W2)
        flat = out2.reshape(H2, W2, SC).permute(0, 2, 1).reshape(H2, SC * W2).contiguous()
        if H2 < L_pad:
            flat_padded = torch.zeros(L_pad, 4096, dtype=torch.bfloat16)
            flat_padded[:H2, :] = flat
            flat = flat_padded
        self.dma_to_accelerator_memory(self.SUB_FLAT_DRAM, flat.contiguous())

        # --- Subsampling linear ---
        self.program_execute(self.progs["sub_lin"][0])
        print("  Subsampling done")

        # --- Write positional embedding ---
        P_pad = pad_to_multiple(2 * L_pad - 1, self.block_size)
        rel_pe = self.make_rel_pos_emb(L_pad)
        if rel_pe.shape[0] < P_pad:
            pe_padded = torch.zeros(P_pad, D, dtype=torch.bfloat16)
            pe_padded[:rel_pe.shape[0], :] = rel_pe
            rel_pe = pe_padded
        self.dma_to_accelerator_memory(self.POS_EMB_DRAM, rel_pe.contiguous())

        # --- Write attention mask ---
        mask = torch.full((L_pad, L_pad), float("-inf"), dtype=torch.bfloat16)
        mask[:L, :L] = 0.0
        self.dma_to_accelerator_memory(self.ATTN_MASK_DRAM, mask.contiguous())

        # --- Execute 24 conformer layers ---
        print("  Executing conformer layers...")
        enc_prog, enc_flops = self.progs["encoder"]
        latency_us, gflops = self.program_execute(enc_prog, flops=enc_flops)
        print(f"  Encoder: {latency_us:.0f} us, {gflops:.1f} GFLOPS")
        return self.ENC_OUT_DRAM

    # -----------------------------------------------------------------------
    # Execute: decoder
    # -----------------------------------------------------------------------

    def run_decode(self, enc_out_addr, L):
        """TDT greedy decode. All programs already compiled. Returns token list."""
        bpe = self.bytes_per_element
        H = self.pred_hidden
        D = self.d_model
        pred_prog = self.progs["pred"][0]
        tok_prog = self.progs["joint_tok"][0]
        dur_prog = self.progs["joint_dur"][0]
        # Init LSTM state
        zeros_h = torch.zeros(H, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.PRED_H0_DRAM, zeros_h)
        self.dma_to_accelerator_memory(self.PRED_C0_DRAM, zeros_h)
        self.dma_to_accelerator_memory(self.PRED_H1_DRAM, zeros_h)
        self.dma_to_accelerator_memory(self.PRED_C1_DRAM, zeros_h)
        tokens = []
        t = 0
        last_token = self.blank_id
        total_steps = 0
        while t < L:
            symbols = 0
            while symbols < self.max_symbols_per_step:
                # Embedding lookup (on-device DRAM -> SRAM -> DRAM)
                emb_src = self.w["EMBED"] + last_token * H * bpe
                self.accelerator_memory_to_sram(emb_src, URAM_A_BASE, H)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.PRED_EMB_DRAM, H)
                # Predictor
                self.program_execute(pred_prog)
                # Copy enc_out[t] and pred_out to joint inputs (on-device)
                self.accelerator_memory_to_sram(enc_out_addr + t * D * bpe, URAM_A_BASE, D)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.JOINT_ENC_DRAM, D)
                self.accelerator_memory_to_sram(self.PRED_OUT_DRAM, URAM_A_BASE, H)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.JOINT_PRED_DRAM, H)
                # Joint token -> hardware argmax
                self.program_execute(tok_prog)
                token_id = self.get_arg_max_index()
                # Joint duration -> hardware argmax
                self.program_execute(dur_prog)
                dur_idx = self.get_arg_max_index()
                dur = self.tdt_durations[dur_idx] if dur_idx < len(self.tdt_durations) else 0
                total_steps += 1
                if token_id == self.blank_id:
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
        print(f"  Decode: {total_steps} joint steps, {len(tokens)} tokens emitted")
        return tokens
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parakeet-TDT-0.6B accelerator inference")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file (.wav, .flac, etc.)")
    parser.add_argument("--dev", type=str, default="xdma0", help="XDMA device")
    parser.add_argument("--cycle", type=float, default=5.63, help="Clock cycle in ns")
    args = parser.parse_args()
    cfg = load_config()
    set_dma_device(args.dev)
    audio_path = args.audio or os.path.join(SCRIPT_DIR, cfg["defaults"]["default_audio"])
    print(f"Parakeet-TDT-0.6B on {args.dev}")
    print(f"  Encoder: {cfg['encoder']['num_layers']} conformer layers, d_model={cfg['encoder']['d_model']}")
    print(f"  Predictor: {cfg['predictor']['num_layers']}-layer LSTM, hidden={cfg['predictor']['hidden_size']}")
    print(f"  Joint: tok={cfg['joint']['output_size']} dur={len(cfg['joint']['tdt_durations'])}")
    print(f"  Block size: {cfg['hardware']['block_size']}")
    if audio_path and os.path.exists(audio_path):
        import torchaudio
        print(f"  Loading: {audio_path}")
        waveform, sr = torchaudio.load(audio_path)
        if sr != cfg["preprocessing"]["sample_rate"]:
            waveform = torchaudio.functional.resample(waveform, sr, cfg["preprocessing"]["sample_rate"])
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
    else:
        duration_s = 5
        waveform = torch.randn(1, cfg["preprocessing"]["sample_rate"] * duration_s)
        print(f"  Using dummy {duration_s}s audio")
    print(f"  Audio: {waveform.shape[1]} samples ({waveform.shape[1]/cfg['preprocessing']['sample_rate']:.1f}s)")
    mel = compute_mel_spectrogram(waveform, cfg)
    T_mel = mel.shape[1]
    L = T_mel // 8
    L_pad = pad_to_multiple(L, cfg["hardware"]["block_size"])
    print(f"  Mel: {T_mel} frames, L={L}, L_pad={L_pad}")
    engine = Parakeet_UnifiedEngine()
    engine.weight_init()

    # --- Compile all programs ---
    print(f"\n--- Compile ---")
    engine.tensor_init(L_pad)
    engine.compile_all(T_mel, L_pad)

    # --- Execute ---
    print(f"\n--- Encoder ---")
    enc_out_addr = engine.run_encoder(mel, L, L_pad)
    print(f"\n--- Decoder ---")
    tokens = engine.run_decode(enc_out_addr, L)
    tokenizer_path = TOKENIZER_PATH
    if os.path.exists(tokenizer_path):
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(tokenizer_path)
        vocab_sz = sp.GetPieceSize()
        valid_tokens = [t for t in tokens if 0 <= t < vocab_sz]
        text = sp.DecodeIds(valid_tokens)
        print(f"\n  >>> {text}\n")
    else:
        print(f"  Tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
        print(f"  (tokenizer not found at {tokenizer_path})")


if __name__ == "__main__":
    main()
