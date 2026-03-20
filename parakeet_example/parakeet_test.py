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
    DMA_DEVICE_H2C, DRAM_INSTRUCTION_ADDR, TYPE, UE_VECTOR_SIZE,
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
    scale_sram = URAM_B_BASE
    shift_sram = URAM_B_BASE + C * 2
    ue.accelerator_memory_to_sram(SCALE_DRAM_ADDR, scale_sram, C)
    ue.accelerator_memory_to_sram(SHIFT_DRAM_ADDR, shift_sram, C)
    scale_host = torch.zeros(C, dtype=torch.bfloat16)
    ue.dma_read(DMA_DEVICE_H2C.replace("H2C", "C2H") if hasattr(DMA_DEVICE_H2C, 'replace') else "/dev/xdma0_c2h_0",
                SCALE_DRAM_ADDR, scale_host, C * 2)
    shift_host = torch.zeros(C, dtype=torch.bfloat16)
    ue.dma_read("/dev/xdma0_c2h_0", SHIFT_DRAM_ADDR, shift_host, C * 2)
    for c in range(C):
        ue.accelerator_memory_to_sram(A_DRAM_ADDR + c * row_bytes, URAM_A_BASE, L)
        scale_bf16 = struct.unpack('H', scale_host[c].contiguous().view(torch.uint8).numpy().tobytes())[0]
        ue.broadcast_mul(scalar=scale_bf16, sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=L)
        shift_bf16 = struct.unpack('H', shift_host[c].contiguous().view(torch.uint8).numpy().tobytes())[0]
        ue.broadcast_add(scalar=shift_bf16, sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=L)
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


# ---------------------------------------------------------------------------
# Permute core (copied from smolvlm2 — general-purpose DMA gather + transpose)
# ---------------------------------------------------------------------------
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

        self.d_model = enc["d_model"]
        self.num_layers = enc["num_layers"]
        self.num_heads = enc["num_heads"]
        self.head_dim = enc["head_dim"]
        self.ff_dim = enc["ff_dim"]
        self.conv_kernel = enc["conv_kernel"]
        self.conv_pad = enc["conv_pad"]
        self.sub_channels = enc["sub_channels"]
        self.n_mels = enc["n_mels"]

        self.pred_hidden = pred["hidden_size"]
        self.vocab_size = pred["vocab_size"]

        self.joint_hidden = jnt["hidden_size"]
        self.joint_output_padded = jnt["output_size_padded"]
        self.blank_id = jnt["blank_id"]
        self.tdt_durations = jnt["tdt_durations"]
        self.max_symbols_per_step = jnt["max_symbols_per_step"]

        self.block_size = hw["block_size"]
        self.bytes_per_element = enc["bytes_per_element"]

    # -----------------------------------------------------------------------
    # Weight loading
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Helpers: allocate + write a bf16 tensor to DRAM
    # -----------------------------------------------------------------------

    def _alloc_write(self, tensor):
        """Allocate params DRAM and write a bf16 tensor. Returns DRAM address."""
        t = tensor.to(torch.bfloat16).contiguous()
        addr = self.get_params_dram_addr()
        nbytes = t.numel() * 2
        self.allocate_params_dram(nbytes)
        self.dma_write(DMA_DEVICE_H2C, addr, t, nbytes)
        return addr

    def _stage_dw_conv1d(self, weight):
        """Stage DW conv1d: (C, 1, K) → (C, 64) im2col padded. Write to DRAM."""
        C, _, K = weight.shape
        padded_K = pad_to_multiple(K, self.block_size)
        w_flat = weight.reshape(C, K).to(torch.bfloat16)
        w_padded = torch.zeros(C, padded_K, dtype=torch.bfloat16)
        w_padded[:, :K] = w_flat
        return self._alloc_write(w_padded)

    def _split_pw_conv1(self, weight):
        """Split PW Conv1 (2D, D, 1) → two (D, D) matrices for GLU split.
        weight shape: (2*D_MODEL, D_MODEL, 1) → W_a (D, D), W_b (D, D).
        Returns (addr_a, addr_b).
        """
        w = weight.squeeze(-1).to(torch.bfloat16)  # (2D, D)
        D = w.shape[0] // 2
        w_a = w[:D, :].contiguous()   # first half → a path
        w_b = w[D:, :].contiguous()   # second half → b path (sigmoid)
        return self._alloc_write(w_a), self._alloc_write(w_b)

    def _pad_joint_output(self, weight, bias):
        """Pad joint output from (8198, 640) → (8256, 640), bias (8198,) → (8256,)."""
        N, K = weight.shape
        N_pad = pad_to_multiple(N, self.block_size)
        w = weight.to(torch.bfloat16)
        b = bias.to(torch.bfloat16)
        if N_pad != N:
            w_padded = torch.zeros(N_pad, K, dtype=torch.bfloat16)
            w_padded[:N] = w
            b_padded = torch.zeros(N_pad, dtype=torch.bfloat16)
            b_padded[:N] = b
            w, b = w_padded, b_padded
        return self._alloc_write(w), self._alloc_write(b)

    # -----------------------------------------------------------------------
    # Weight loading
    # -----------------------------------------------------------------------

    def weight_init(self):
        """Load NeMo checkpoint, stage weights, write all to DRAM.
        Stores per-layer weight DRAM addresses in self.layer_addrs[layer_idx].
        Stores shared addresses in self.w (subsampling, predictor, joint, identity).
        """
        ckpt_path = WEIGHTS_PATH
        print(f"Loading weights from {ckpt_path} ...")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        print(f"  {sum(v.numel() for v in sd.values() if hasattr(v, 'numel')):,} parameters")

        bpe = self.bytes_per_element
        D = self.d_model       # 1024
        FF = self.ff_dim       # 4096
        H = self.pred_hidden   # 640

        self.w = {}

        # --- Identity matrices (shared, allocated once) ---
        self.w["IDENTITY_1024"] = allocate_identity(self, D)
        self.w["IDENTITY_4096"] = allocate_identity(self, FF)
        self.w["IDENTITY_640"] = allocate_identity(self, self.joint_hidden)

        # --- Subsampling weights (stored in self.w, used by CPU subsampling) ---
        # We write these to DRAM for completeness but subsampling runs on CPU.
        self.w["SUB_CONV0_W"] = self._alloc_write(sd["encoder.pre_encode.conv.0.weight"].reshape(256, -1))
        self.w["SUB_CONV0_B"] = self._alloc_write(sd["encoder.pre_encode.conv.0.bias"])
        self.w["SUB_CONV2_W"] = self._alloc_write(sd["encoder.pre_encode.conv.2.weight"].reshape(256, -1))
        self.w["SUB_CONV2_B"] = self._alloc_write(sd["encoder.pre_encode.conv.2.bias"])
        self.w["SUB_CONV3_W"] = self._alloc_write(sd["encoder.pre_encode.conv.3.weight"].reshape(256, 256))
        self.w["SUB_CONV3_B"] = self._alloc_write(sd["encoder.pre_encode.conv.3.bias"])
        self.w["SUB_CONV5_W"] = self._alloc_write(sd["encoder.pre_encode.conv.5.weight"].reshape(256, -1))
        self.w["SUB_CONV5_B"] = self._alloc_write(sd["encoder.pre_encode.conv.5.bias"])
        self.w["SUB_CONV6_W"] = self._alloc_write(sd["encoder.pre_encode.conv.6.weight"].reshape(256, 256))
        self.w["SUB_CONV6_B"] = self._alloc_write(sd["encoder.pre_encode.conv.6.bias"])
        self.w["SUB_OUT_W"] = self._alloc_write(sd["encoder.pre_encode.out.weight"])
        self.w["SUB_OUT_B"] = self._alloc_write(sd["encoder.pre_encode.out.bias"])

        # --- Per-layer conformer weights ---
        self.layer_addrs = []
        for i in range(self.num_layers):
            la = {}
            pfx = f"encoder.layers.{i}"

            # NeMo key → our DRAM key mapping for LayerNorms
            ln_map = {
                "LN_FF1":  "norm_feed_forward1",
                "LN_ATTN": "norm_self_att",
                "LN_CONV": "norm_conv",
                "LN_FF2":  "norm_feed_forward2",
                "LN_OUT":  "norm_out",
            }
            for our_key, nemo_key in ln_map.items():
                la[f"{our_key}_WEIGHT"] = self._alloc_write(sd[f"{pfx}.{nemo_key}.weight"])
                la[f"{our_key}_BIAS"] = self._alloc_write(sd[f"{pfx}.{nemo_key}.bias"])

            # FF1 / FF2
            la["FF1_W1"] = self._alloc_write(sd[f"{pfx}.feed_forward1.linear1.weight"])
            la["FF1_W2"] = self._alloc_write(sd[f"{pfx}.feed_forward1.linear2.weight"])
            la["FF2_W1"] = self._alloc_write(sd[f"{pfx}.feed_forward2.linear1.weight"])
            la["FF2_W2"] = self._alloc_write(sd[f"{pfx}.feed_forward2.linear2.weight"])

            # Attention projections
            la["ATTN_Q_W"] = self._alloc_write(sd[f"{pfx}.self_attn.linear_q.weight"])
            la["ATTN_K_W"] = self._alloc_write(sd[f"{pfx}.self_attn.linear_k.weight"])
            la["ATTN_V_W"] = self._alloc_write(sd[f"{pfx}.self_attn.linear_v.weight"])
            la["ATTN_POS_W"] = self._alloc_write(sd[f"{pfx}.self_attn.linear_pos.weight"])
            la["ATTN_OUT_W"] = self._alloc_write(sd[f"{pfx}.self_attn.linear_out.weight"])

            # Pos biases: (8, 128) → flatten to (1024,)
            la["ATTN_BIAS_U"] = self._alloc_write(sd[f"{pfx}.self_attn.pos_bias_u"].reshape(-1))
            la["ATTN_BIAS_V"] = self._alloc_write(sd[f"{pfx}.self_attn.pos_bias_v"].reshape(-1))

            # Conv module: split PW1 (2048, 1024, 1) → two (1024, 1024)
            pw1_a, pw1_b = self._split_pw_conv1(sd[f"{pfx}.conv.pointwise_conv1.weight"])
            la["CONV_PW1A_W"] = pw1_a
            la["CONV_PW1B_W"] = pw1_b

            # DW Conv1d staged: (1024, 1, 9) → im2col padded (1024, 64)
            la["CONV_DW_W"] = self._stage_dw_conv1d(sd[f"{pfx}.conv.depthwise_conv.weight"])

            # BN: fuse to scale + shift, write to DRAM
            bn_w = sd[f"{pfx}.conv.batch_norm.weight"]
            bn_b = sd[f"{pfx}.conv.batch_norm.bias"]
            bn_m = sd[f"{pfx}.conv.batch_norm.running_mean"]
            bn_v = sd[f"{pfx}.conv.batch_norm.running_var"]
            scale_addr, shift_addr, _ = batch_norm_fuse_params(
                self, bn_w, bn_b, bn_m, bn_v)
            la["CONV_BN_SCALE"] = scale_addr
            la["CONV_BN_SHIFT"] = shift_addr

            # PW Conv2: (1024, 1024, 1) → (1024, 1024)
            la["CONV_PW2_W"] = self._alloc_write(sd[f"{pfx}.conv.pointwise_conv2.weight"].squeeze(-1))

            self.layer_addrs.append(la)
            print(f"  Layer {i:2d} staged")

        # --- Predictor weights ---
        self.w["EMBED"] = self._alloc_write(sd["decoder.prediction.embed.weight"])
        for i in range(2):
            self.w[f"LSTM_WIH{i}"] = self._alloc_write(
                sd[f"decoder.prediction.dec_rnn.lstm.weight_ih_l{i}"])
            self.w[f"LSTM_WHH{i}"] = self._alloc_write(
                sd[f"decoder.prediction.dec_rnn.lstm.weight_hh_l{i}"])
            self.w[f"LSTM_BIH{i}"] = self._alloc_write(
                sd[f"decoder.prediction.dec_rnn.lstm.bias_ih_l{i}"])
            self.w[f"LSTM_BHH{i}"] = self._alloc_write(
                sd[f"decoder.prediction.dec_rnn.lstm.bias_hh_l{i}"])

        # --- Joint weights ---
        self.w["JOINT_ENC_W"] = self._alloc_write(sd["joint.enc.weight"])
        self.w["JOINT_ENC_B"] = self._alloc_write(sd["joint.enc.bias"])
        self.w["JOINT_PRED_W"] = self._alloc_write(sd["joint.pred.weight"])
        self.w["JOINT_PRED_B"] = self._alloc_write(sd["joint.pred.bias"])

        # Joint output: split into token logits + duration logits for separate argmax
        # Token: rows [0:8193] padded to 8256 × 640
        # Duration: rows [8193:8198] padded to 64 × 640
        out_w = sd["joint.joint_net.2.weight"].to(torch.bfloat16)  # (8198, 640)
        out_b = sd["joint.joint_net.2.bias"].to(torch.bfloat16)    # (8198,)

        N_tok = self.vocab_size                           # 8193
        N_tok_pad = pad_to_multiple(N_tok, self.block_size)  # 8256
        N_dur = len(self.tdt_durations)                   # 5
        N_dur_pad = pad_to_multiple(N_dur, self.block_size)  # 64

        # Token weights: (8256, 640)
        w_tok = torch.zeros(N_tok_pad, self.joint_hidden, dtype=torch.bfloat16)
        w_tok[:N_tok] = out_w[:N_tok]
        b_tok = torch.zeros(N_tok_pad, dtype=torch.bfloat16)
        b_tok[:N_tok] = out_b[:N_tok]
        self.w["JOINT_OUT_TOK_W"] = self._alloc_write(w_tok)
        self.w["JOINT_OUT_TOK_B"] = self._alloc_write(b_tok)

        # Duration weights: (64, 640)
        w_dur = torch.zeros(N_dur_pad, self.joint_hidden, dtype=torch.bfloat16)
        w_dur[:N_dur] = out_w[N_tok:N_tok + N_dur]
        b_dur = torch.zeros(N_dur_pad, dtype=torch.bfloat16)
        b_dur[:N_dur] = out_b[N_tok:N_tok + N_dur]
        self.w["JOINT_OUT_DUR_W"] = self._alloc_write(w_dur)
        self.w["JOINT_OUT_DUR_B"] = self._alloc_write(b_dur)

        total_params_bytes = self.get_params_dram_usage()
        print(f"  Weight staging complete: {total_params_bytes / 1024**2:.1f} MB in DRAM")

    # -----------------------------------------------------------------------
    # Tensor init: intermediate DRAM buffers (reused across layers)
    # -----------------------------------------------------------------------

    def tensor_init(self, L_pad):
        """Allocate intermediate DRAM buffers for encoder pipeline."""
        bpe = self.bytes_per_element
        D = self.d_model       # 1024
        FF = self.ff_dim       # 4096
        H = self.pred_hidden   # 640

        # Encoder intermediates (reused per layer)
        self.INPUT_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.LN_OUT_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.FF_MID_DRAM = self.allocate_tensor_dram(L_pad * FF * bpe)
        self.FF_OUT_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.RESIDUAL_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)

        # Attention intermediates
        self.Q_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.K_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.V_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        self.POS_PROJ_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)
        # Per-head score matrices: H heads × (L_pad × L_pad)
        # We process heads sequentially, reuse one buffer per head
        self.SCORE_DRAM = self.allocate_tensor_dram(L_pad * L_pad * bpe)
        self.ATTN_OUT_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)

        # Pos emb: (2L_pad-1, D)
        P_pad = pad_to_multiple(2 * L_pad - 1, self.block_size)
        self.POS_EMB_DRAM = self.allocate_tensor_dram(P_pad * D * bpe)
        # rel_shift output: (L_pad, L_pad)
        self.REL_SHIFT_DRAM = self.allocate_tensor_dram(L_pad * L_pad * bpe)

        # Conv module intermediates
        self.CONV_A_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)   # GLU a
        self.CONV_B_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)   # GLU b
        self.CONV_T_DRAM = self.allocate_tensor_dram(D * L_pad * bpe)   # transposed (D, L_pad)
        self.CONV_DW_DRAM = self.allocate_tensor_dram(D * L_pad * bpe)  # after DW conv (D, L_pad)
        self.CONV_OUT_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)

        # Attention: transposed V_h scratch and mask
        dk = self.head_dim  # 128
        self.ATTN_VT_DRAM = self.allocate_tensor_dram(dk * L_pad * bpe)  # V_h transposed (dk, L_pad)
        self.ATTN_MASK_DRAM = self.allocate_tensor_dram(L_pad * L_pad * bpe)  # padding mask

        # Permute scratch (needed by bf16_smart_permute_core)
        self.PERMUTE_TEMP_DRAM = self.allocate_tensor_dram(D * L_pad * bpe)

        # Encoder output (final)
        self.ENC_OUT_DRAM = self.allocate_tensor_dram(L_pad * D * bpe)

        # Decoder intermediates (fixed dims, allocated once)
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
        N_tok_pad = pad_to_multiple(self.vocab_size, self.block_size)  # 8256
        N_dur_pad = pad_to_multiple(len(self.tdt_durations), self.block_size)  # 64
        self.JOINT_TOK_DRAM = self.allocate_tensor_dram(N_tok_pad * bpe)
        self.JOINT_DUR_DRAM = self.allocate_tensor_dram(N_dur_pad * bpe)

    # -----------------------------------------------------------------------
    # Compile: feed-forward sub-module
    # -----------------------------------------------------------------------

    def _compile_ff(self, L_pad, layer_idx, ff_prefix, ln_prefix):
        """Emit FF1 or FF2: LN → matmul(L,1024)@(1024,4096) → SiLU → matmul(L,4096)@(4096,1024) → half-step residual."""
        la = self.layer_addrs[layer_idx]
        D, FF = self.d_model, self.ff_dim
        flops = 0

        # LayerNorm
        self.layer_norm_core_dram(M=L_pad, N=D,
            A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.LN_OUT_DRAM,
            GAMMA_DRAM_ADDR=la[f"{ln_prefix}_WEIGHT"], BETA_DRAM_ADDR=la[f"{ln_prefix}_BIAS"])

        # Linear1: (L_pad, 1024) @ (1024, 4096) with SiLU
        self.matmat_mul_core(M=L_pad, K=D, N=FF,
            A_DRAM_ADDR=self.LN_OUT_DRAM, B_DRAM_ADDR=la[f"{ff_prefix}_W1"],
            OUTPUT_DRAM_ADDR=self.FF_MID_DRAM, silu_enable=True)
        flops += 2 * L_pad * D * FF

        # Linear2: (L_pad, 4096) @ (4096, 1024)
        self.matmat_mul_core(M=L_pad, K=FF, N=D,
            A_DRAM_ADDR=self.FF_MID_DRAM, B_DRAM_ADDR=la[f"{ff_prefix}_W2"],
            OUTPUT_DRAM_ADDR=self.FF_OUT_DRAM)
        flops += 2 * L_pad * FF * D

        # Half-step residual: input + 0.5 * ff_out → INPUT (in-place)
        half_step_residual_core_dram(self, M=L_pad, N=D,
            RESIDUAL_DRAM_ADDR=self.INPUT_DRAM, FF_DRAM_ADDR=self.FF_OUT_DRAM,
            OUTPUT_DRAM_ADDR=self.INPUT_DRAM)
        flops += 2 * L_pad * D
        return flops

    # -----------------------------------------------------------------------
    # Compile: self-attention sub-module
    # -----------------------------------------------------------------------

    def _compile_attention(self, L_pad, layer_idx):
        """Emit relative-positional multi-head self-attention.

        matmat_mul_core computes C(M,N) = A(M,K) @ B(N,K)^T.
        For attn @ V we need scores(L,L) @ V_h(L,dk). Since the core does A@B^T,
        we transpose V_h to (dk, L) so: A(L,L) @ B(dk,L)^T = A @ V_h. ✓
        """
        la = self.layer_addrs[layer_idx]
        D = self.d_model
        H = self.num_heads
        dk = self.head_dim  # 128
        bpe = self.bytes_per_element
        flops = 0

        # LayerNorm
        self.layer_norm_core_dram(M=L_pad, N=D,
            A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.LN_OUT_DRAM,
            GAMMA_DRAM_ADDR=la["LN_ATTN_WEIGHT"], BETA_DRAM_ADDR=la["LN_ATTN_BIAS"])

        # Q, K, V projections: each (L_pad, 1024) @ (1024, 1024)
        self.matmat_mul_core(M=L_pad, K=D, N=D,
            A_DRAM_ADDR=self.LN_OUT_DRAM, B_DRAM_ADDR=la["ATTN_Q_W"],
            OUTPUT_DRAM_ADDR=self.Q_DRAM)
        self.matmat_mul_core(M=L_pad, K=D, N=D,
            A_DRAM_ADDR=self.LN_OUT_DRAM, B_DRAM_ADDR=la["ATTN_K_W"],
            OUTPUT_DRAM_ADDR=self.K_DRAM)
        self.matmat_mul_core(M=L_pad, K=D, N=D,
            A_DRAM_ADDR=self.LN_OUT_DRAM, B_DRAM_ADDR=la["ATTN_V_W"],
            OUTPUT_DRAM_ADDR=self.V_DRAM)
        flops += 3 * 2 * L_pad * D * D

        # Pos projection: (P_pad, 1024) @ (1024, 1024) → POS_PROJ
        P = 2 * L_pad - 1
        P_pad = pad_to_multiple(P, self.block_size)
        self.matmat_mul_core(M=P_pad, K=D, N=D,
            A_DRAM_ADDR=self.POS_EMB_DRAM, B_DRAM_ADDR=la["ATTN_POS_W"],
            OUTPUT_DRAM_ADDR=self.POS_PROJ_DRAM)
        flops += 2 * P_pad * D * D

        # --- Per-head attention (sequential over H=8 heads) ---
        for h in range(H):
            h_off = h * dk * bpe

            # --- Extract Q_h (L_pad, dk) via strided read ---
            self.accelerator_memory_to_sram(
                accelerator_dram_address=self.Q_DRAM + h_off,
                sram_address=URAM_A_BASE, element_size=L_pad * dk,
                stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)

            # Add bias_u_h (broadcast across rows)
            self.accelerator_memory_to_sram(la["ATTN_BIAS_U"] + h_off, URAM_B_BASE, dk)
            for row in range(L_pad):
                self.eltwise_add_core(
                    vector_A_sram_start_addr=URAM_A_BASE + row * dk * bpe,
                    vector_B_sram_start_addr=URAM_B_BASE,
                    vector_C_sram_wb_addr=URAM_A_BASE + row * dk * bpe,
                    element_size=dk)
            self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_A_DRAM, L_pad * dk)

            # --- Extract K_h (L_pad, dk) via strided read ---
            self.accelerator_memory_to_sram(
                accelerator_dram_address=self.K_DRAM + h_off,
                sram_address=URAM_A_BASE, element_size=L_pad * dk,
                stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
            self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_B_DRAM, L_pad * dk)

            # Content: (Q_h+u)(L,dk) @ K_h(L,dk)^T = (L,L) ✓
            self.matmat_mul_core(M=L_pad, K=dk, N=L_pad,
                A_DRAM_ADDR=self.CONV_A_DRAM, B_DRAM_ADDR=self.CONV_B_DRAM,
                OUTPUT_DRAM_ADDR=self.SCORE_DRAM)
            flops += 2 * L_pad * dk * L_pad

            # --- Positional: (Q_h + bias_v) @ P_h^T → rel_shift ---
            self.accelerator_memory_to_sram(
                accelerator_dram_address=self.Q_DRAM + h_off,
                sram_address=URAM_A_BASE, element_size=L_pad * dk,
                stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
            self.accelerator_memory_to_sram(la["ATTN_BIAS_V"] + h_off, URAM_B_BASE, dk)
            for row in range(L_pad):
                self.eltwise_add_core(
                    vector_A_sram_start_addr=URAM_A_BASE + row * dk * bpe,
                    vector_B_sram_start_addr=URAM_B_BASE,
                    vector_C_sram_wb_addr=URAM_A_BASE + row * dk * bpe,
                    element_size=dk)
            self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_A_DRAM, L_pad * dk)

            # Extract P_h (P_pad, dk) via strided read
            self.accelerator_memory_to_sram(
                accelerator_dram_address=self.POS_PROJ_DRAM + h_off,
                sram_address=URAM_A_BASE, element_size=P_pad * dk,
                stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
            self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_B_DRAM, P_pad * dk)

            # (L,dk) @ (P_pad,dk)^T = (L, P_pad)
            self.matmat_mul_core(M=L_pad, K=dk, N=P_pad,
                A_DRAM_ADDR=self.CONV_A_DRAM, B_DRAM_ADDR=self.CONV_B_DRAM,
                OUTPUT_DRAM_ADDR=self.CONV_OUT_DRAM)  # scratch for pos scores
            flops += 2 * L_pad * dk * P_pad

            # rel_shift: (L, P_pad) → (L, L)
            rel_shift_core_dram(self, L=L_pad,
                INPUT_DRAM_ADDR=self.CONV_OUT_DRAM, OUTPUT_DRAM_ADDR=self.REL_SHIFT_DRAM)

            # --- Combine: content + pos → scale → add mask → softmax ---
            score_elems = L_pad * L_pad
            self.accelerator_memory_to_sram(self.SCORE_DRAM, URAM_A_BASE, score_elems)
            self.accelerator_memory_to_sram(self.REL_SHIFT_DRAM, URAM_B_BASE, score_elems)
            self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, score_elems)
            self.sram_to_accelerator_memory(URAM_A_BASE, self.SCORE_DRAM, score_elems)

            # Scale: scores * (1/sqrt(dk)) and add attention mask via fused bias
            # matmat_mul_core: scores(L,L) @ I(L,L) with bias = ATTN_MASK (full_matrix)
            # This applies: output[i,j] = scores[i,j] * scale + mask[i,j]
            # Then softmax is fused on top.
            # We use broadcast_mul for scale first, then softmax matmul with mask bias.
            self.accelerator_memory_to_sram(self.SCORE_DRAM, URAM_A_BASE, score_elems)
            inv_sqrt_dk = self.float_to_bf16(1.0 / math.sqrt(dk))
            for row in range(L_pad):
                self.broadcast_mul(scalar=inv_sqrt_dk,
                    sram_start_addr=URAM_A_BASE + row * L_pad * bpe,
                    sram_wb_addr=URAM_A_BASE + row * L_pad * bpe,
                    element_size=L_pad)
            self.sram_to_accelerator_memory(URAM_A_BASE, self.SCORE_DRAM, score_elems)

            # Softmax with padding mask as bias (full_matrix mode):
            # C(L,L) = softmax(A(L,L) @ I(L,L) + mask(L,L))
            self.matmat_mul_core(M=L_pad, K=L_pad, N=L_pad,
                A_DRAM_ADDR=self.SCORE_DRAM,
                B_DRAM_ADDR=self.w["IDENTITY_1024"],
                OUTPUT_DRAM_ADDR=self.SCORE_DRAM,
                softmax_enable=True,
                C_DRAM_ADDR=self.ATTN_MASK_DRAM, bias_mode="full_matrix")

            # --- Value matmul: attn(L,L) @ V_h(L,dk) ---
            # Extract V_h (L_pad, dk) via strided read
            self.accelerator_memory_to_sram(
                accelerator_dram_address=self.V_DRAM + h_off,
                sram_address=URAM_A_BASE, element_size=L_pad * dk,
                stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
            self.sram_to_accelerator_memory(URAM_A_BASE, self.CONV_B_DRAM, L_pad * dk)

            # Transpose V_h: (L_pad, dk) → (dk, L_pad) so matmul does A @ B^T = A @ V_h
            bf16_smart_permute_core(self,
                dims=[L_pad, dk], permute_indices=[1, 0],
                input_dram_addr=self.CONV_B_DRAM,
                output_dram_addr=self.ATTN_VT_DRAM,
                params_dram_addr=self.w["IDENTITY_1024"],
                temp_dram_start=self.PERMUTE_TEMP_DRAM)

            # attn(L,L) @ V_h^T(dk,L)^T = attn @ V_h → (L, dk) ✓
            self.matmat_mul_core(M=L_pad, K=L_pad, N=dk,
                A_DRAM_ADDR=self.SCORE_DRAM, B_DRAM_ADDR=self.ATTN_VT_DRAM,
                OUTPUT_DRAM_ADDR=self.CONV_A_DRAM)
            flops += 2 * L_pad * L_pad * dk

            # Write head output back to ATTN_OUT at correct head offset (strided write)
            self.accelerator_memory_to_sram(self.CONV_A_DRAM, URAM_A_BASE, L_pad * dk)
            self.sram_to_accelerator_memory(URAM_A_BASE,
                self.ATTN_OUT_DRAM + h_off, L_pad * dk,
                stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)

        # Output projection: (L_pad, 1024) @ (1024, 1024)
        self.matmat_mul_core(M=L_pad, K=D, N=D,
            A_DRAM_ADDR=self.ATTN_OUT_DRAM, B_DRAM_ADDR=la["ATTN_OUT_W"],
            OUTPUT_DRAM_ADDR=self.FF_OUT_DRAM)
        flops += 2 * L_pad * D * D

        # Residual: input + attn_out → INPUT
        self.accelerator_memory_to_sram(self.INPUT_DRAM, URAM_A_BASE, L_pad * D)
        self.accelerator_memory_to_sram(self.FF_OUT_DRAM, URAM_B_BASE, L_pad * D)
        self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * D)
        self.sram_to_accelerator_memory(URAM_A_BASE, self.INPUT_DRAM, L_pad * D)
        flops += L_pad * D
        return flops

    # -----------------------------------------------------------------------
    # Compile: conv module
    # -----------------------------------------------------------------------

    def _compile_conv_module(self, L_pad, layer_idx):
        """Emit conv module: LN → PW1(split) → GLU → DW → BN → SiLU → PW2."""
        la = self.layer_addrs[layer_idx]
        D = self.d_model
        flops = 0

        # LayerNorm
        self.layer_norm_core_dram(M=L_pad, N=D,
            A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.LN_OUT_DRAM,
            GAMMA_DRAM_ADDR=la["LN_CONV_WEIGHT"], BETA_DRAM_ADDR=la["LN_CONV_BIAS"])

        # PW Conv1 split: a = LN_OUT @ W_a, b = LN_OUT @ W_b (each L_pad,1024)
        self.matmat_mul_core(M=L_pad, K=D, N=D,
            A_DRAM_ADDR=self.LN_OUT_DRAM, B_DRAM_ADDR=la["CONV_PW1A_W"],
            OUTPUT_DRAM_ADDR=self.CONV_A_DRAM)
        self.matmat_mul_core(M=L_pad, K=D, N=D,
            A_DRAM_ADDR=self.LN_OUT_DRAM, B_DRAM_ADDR=la["CONV_PW1B_W"],
            OUTPUT_DRAM_ADDR=self.CONV_B_DRAM)
        flops += 2 * 2 * L_pad * D * D

        # GLU: a * sigmoid(b) → CONV_A_DRAM  (L_pad, D)
        glu_core_dram(self, M=L_pad, C=D,
            A_DRAM_ADDR=self.CONV_A_DRAM, B_DRAM_ADDR=self.CONV_B_DRAM,
            OUTPUT_DRAM_ADDR=self.CONV_A_DRAM,
            IDENTITY_DRAM_ADDR=self.w["IDENTITY_1024"])

        # Transpose (L_pad, D) → (D, L_pad) for channel-wise DW conv
        bf16_smart_permute_core(self,
            dims=[L_pad, D], permute_indices=[1, 0],
            input_dram_addr=self.CONV_A_DRAM,
            output_dram_addr=self.CONV_T_DRAM,
            params_dram_addr=self.w["IDENTITY_1024"],
            temp_dram_start=self.PERMUTE_TEMP_DRAM)

        # DW Conv1d im2col on (D, L_pad) layout
        # Per channel: for each 64-element time block, build im2col patches
        # via strided DRAM read (free), then matmul (64,64)@(64,1) → (64,1)
        n_blocks = L_pad // self.block_size
        bpe = self.bytes_per_element
        for ch in range(D):
            kernel_addr = la["CONV_DW_W"] + ch * 64 * bpe
            ch_base = self.CONV_T_DRAM + ch * L_pad * bpe
            ch_out = self.CONV_DW_DRAM + ch * L_pad * bpe
            for blk in range(n_blocks):
                in_addr = ch_base + blk * self.block_size * bpe
                out_addr = ch_out + blk * self.block_size * bpe
                self.matmat_mul_core(M=64, K=64, N=1,
                    A_DRAM_ADDR=in_addr, B_DRAM_ADDR=kernel_addr,
                    OUTPUT_DRAM_ADDR=out_addr)
                flops += 2 * 64 * 64

        # BN + SiLU operate on (D, L_pad) layout — channel-major is fine
        batch_norm_core_dram(self, C=D, L=L_pad,
            A_DRAM_ADDR=self.CONV_DW_DRAM, OUTPUT_DRAM_ADDR=self.CONV_DW_DRAM,
            SCALE_DRAM_ADDR=la["CONV_BN_SCALE"], SHIFT_DRAM_ADDR=la["CONV_BN_SHIFT"])

        silu_core_dram(self, M=D, N=L_pad,
            A_DRAM_ADDR=self.CONV_DW_DRAM, OUTPUT_DRAM_ADDR=self.CONV_DW_DRAM,
            IDENTITY_DRAM_ADDR=self.w["IDENTITY_1024"])

        # Transpose back (D, L_pad) → (L_pad, D) for PW Conv2
        bf16_smart_permute_core(self,
            dims=[D, L_pad], permute_indices=[1, 0],
            input_dram_addr=self.CONV_DW_DRAM,
            output_dram_addr=self.CONV_T_DRAM,
            params_dram_addr=self.w["IDENTITY_1024"],
            temp_dram_start=self.PERMUTE_TEMP_DRAM)

        # PW Conv2: (L_pad, 1024) @ (1024, 1024)
        self.matmat_mul_core(M=L_pad, K=D, N=D,
            A_DRAM_ADDR=self.CONV_T_DRAM, B_DRAM_ADDR=la["CONV_PW2_W"],
            OUTPUT_DRAM_ADDR=self.CONV_OUT_DRAM)
        flops += 2 * L_pad * D * D

        # Residual: input + conv_out → INPUT
        self.accelerator_memory_to_sram(self.INPUT_DRAM, URAM_A_BASE, L_pad * D)
        self.accelerator_memory_to_sram(self.CONV_OUT_DRAM, URAM_B_BASE, L_pad * D)
        self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * D)
        self.sram_to_accelerator_memory(URAM_A_BASE, self.INPUT_DRAM, L_pad * D)
        flops += L_pad * D
        return flops

    # -----------------------------------------------------------------------
    # Compile: Conformer encoder (single layer)
    # -----------------------------------------------------------------------

    def compile_conformer_layer(self, L_pad, layer_idx):
        """Compile one conformer block. Returns total flops."""
        la = self.layer_addrs[layer_idx]
        D = self.d_model
        flops = 0

        # FF1 (half-step residual)
        flops += self._compile_ff(L_pad, layer_idx, "FF1", "LN_FF1")

        # Self-attention + residual
        flops += self._compile_attention(L_pad, layer_idx)

        # Conv module + residual
        flops += self._compile_conv_module(L_pad, layer_idx)

        # FF2 (half-step residual)
        flops += self._compile_ff(L_pad, layer_idx, "FF2", "LN_FF2")

        # Final LayerNorm
        self.layer_norm_core_dram(M=L_pad, N=D,
            A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.INPUT_DRAM,
            GAMMA_DRAM_ADDR=la["LN_OUT_WEIGHT"], BETA_DRAM_ADDR=la["LN_OUT_BIAS"])

        return flops

    # -----------------------------------------------------------------------
    # Compile: Subsampling (CPU for now — variable T_mel, runs once)
    # -----------------------------------------------------------------------

    def compile_subsampling(self, T_mel):
        """Subsampling runs on CPU (variable T_mel, runs once per utterance).
        Returns L = T_mel // 8.
        """
        L = T_mel // 8
        return L

    # -----------------------------------------------------------------------
    # Compile: Full encoder (bucketed by L_pad)
    # -----------------------------------------------------------------------

    def compile_encoder(self, L_pad):
        """Compile encoder conformer layers for a given L_pad bucket.
        Returns (program_addr, total_flops).
        """
        self.tensor_init(L_pad)

        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()

        total_flops = 0
        for layer_idx in range(self.num_layers):
            # Layer-to-layer: output of previous layer is already in INPUT_DRAM
            # (first layer: caller writes subsampling output to INPUT_DRAM)
            total_flops += self.compile_conformer_layer(L_pad, layer_idx)

        # Copy final output
        self.accelerator_memory_to_sram(self.INPUT_DRAM, URAM_A_BASE, L_pad * self.d_model)
        self.sram_to_accelerator_memory(URAM_A_BASE, self.ENC_OUT_DRAM, L_pad * self.d_model)

        self.stop_capture()
        self.generate_instruction_halt()

        program_addr = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(program_addr)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())

        return program_addr, total_flops

    # -----------------------------------------------------------------------
    # Compile: LSTM predictor step
    # -----------------------------------------------------------------------

    def compile_predictor_step(self):
        """Compile one LSTM predictor step (fixed dims, compile once).
        Input: embedding at PRED_EMB_DRAM (1, 640)
        State: PRED_H0, C0, H1, C1
        Output: PRED_OUT_DRAM (1, 640)
        Returns (program_addr, flops).
        """
        H = self.pred_hidden  # 640
        bpe = self.bytes_per_element

        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()
        flops = 0

        for i in range(2):
            h_addr = self.PRED_H0_DRAM if i == 0 else self.PRED_H1_DRAM
            c_addr = self.PRED_C0_DRAM if i == 0 else self.PRED_C1_DRAM
            x_addr = self.PRED_EMB_DRAM if i == 0 else self.PRED_H0_DRAM

            # gates_ih: (1, 640) @ (640, 2560)
            self.matmat_mul_core(M=1, K=H, N=4*H,
                A_DRAM_ADDR=x_addr, B_DRAM_ADDR=self.w[f"LSTM_WIH{i}"],
                OUTPUT_DRAM_ADDR=self.PRED_GATES_DRAM,
                C_DRAM_ADDR=self.w[f"LSTM_BIH{i}"])
            flops += 2 * H * 4 * H

            # gates_hh: (1, 640) @ (640, 2560)
            self.matmat_mul_core(M=1, K=H, N=4*H,
                A_DRAM_ADDR=h_addr, B_DRAM_ADDR=self.w[f"LSTM_WHH{i}"],
                OUTPUT_DRAM_ADDR=self.PRED_GATES2_DRAM,
                C_DRAM_ADDR=self.w[f"LSTM_BHH{i}"])
            flops += 2 * H * 4 * H

            # gates = gates_ih + gates_hh → PRED_GATES_DRAM
            self.accelerator_memory_to_sram(self.PRED_GATES_DRAM, URAM_A_BASE, 4*H)
            self.accelerator_memory_to_sram(self.PRED_GATES2_DRAM, URAM_B_BASE, 4*H)
            self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, 4*H)
            self.sram_to_accelerator_memory(URAM_A_BASE, self.PRED_GATES_DRAM, 4*H)

            # Gate activations:
            # i_gate = sigmoid(gates[0:H])
            # f_gate = sigmoid(gates[H:2H])
            # g_gate = tanh(gates[2H:3H])
            # o_gate = sigmoid(gates[3H:4H])
            #
            # sigmoid on i,f,o via identity matmul:
            # We apply sigmoid to the full (1, 4H) then fix g_gate separately.
            # Actually, let's apply sigmoid to i,f,o slices and tanh to g slice.
            # Simpler: sigmoid on all 4 gates into scratch, then fix g.

            # i_gate (offset 0): sigmoid via identity matmul on (1, H)
            self.matmat_mul_core(M=1, K=H, N=H,
                A_DRAM_ADDR=self.PRED_GATES_DRAM + 0,
                B_DRAM_ADDR=self.w["IDENTITY_640"],
                OUTPUT_DRAM_ADDR=self.PRED_GATES_DRAM + 0,
                sigmoid_enable=True)
            # f_gate (offset H)
            self.matmat_mul_core(M=1, K=H, N=H,
                A_DRAM_ADDR=self.PRED_GATES_DRAM + H * bpe,
                B_DRAM_ADDR=self.w["IDENTITY_640"],
                OUTPUT_DRAM_ADDR=self.PRED_GATES_DRAM + H * bpe,
                sigmoid_enable=True)
            # g_gate (offset 2H): tanh
            tanh_core_dram(self, M=1, N=H,
                A_DRAM_ADDR=self.PRED_GATES_DRAM + 2 * H * bpe,
                OUTPUT_DRAM_ADDR=self.PRED_GATES_DRAM + 2 * H * bpe,
                IDENTITY_DRAM_ADDR=self.w["IDENTITY_640"])
            # o_gate (offset 3H): sigmoid
            self.matmat_mul_core(M=1, K=H, N=H,
                A_DRAM_ADDR=self.PRED_GATES_DRAM + 3 * H * bpe,
                B_DRAM_ADDR=self.w["IDENTITY_640"],
                OUTPUT_DRAM_ADDR=self.PRED_GATES_DRAM + 3 * H * bpe,
                sigmoid_enable=True)

            # Cell update: c_new = f * c_prev + i * g
            # f * c_prev
            self.accelerator_memory_to_sram(self.PRED_GATES_DRAM + H * bpe, URAM_A_BASE, H)  # f
            self.accelerator_memory_to_sram(c_addr, URAM_B_BASE, H)  # c_prev
            self.eltwise_mul_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, H)
            # i * g
            self.accelerator_memory_to_sram(self.PRED_GATES_DRAM + 0, URAM_B_BASE, H)  # i (reuse B)
            # Need g in A... swap approach
            self.sram_to_accelerator_memory(URAM_A_BASE, c_addr, H)  # f*c_prev → c_addr temp
            self.accelerator_memory_to_sram(self.PRED_GATES_DRAM + 0, URAM_A_BASE, H)  # i
            self.accelerator_memory_to_sram(self.PRED_GATES_DRAM + 2 * H * bpe, URAM_B_BASE, H)  # g
            self.eltwise_mul_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, H)  # i*g
            # c_new = f*c_prev + i*g
            self.accelerator_memory_to_sram(c_addr, URAM_B_BASE, H)  # f*c_prev
            self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, H)
            self.sram_to_accelerator_memory(URAM_A_BASE, c_addr, H)  # c_new

            # Hidden: h_new = o * tanh(c_new)
            # tanh(c_new)
            tanh_core_dram(self, M=1, N=H,
                A_DRAM_ADDR=c_addr, OUTPUT_DRAM_ADDR=self.PRED_OUT_DRAM,
                IDENTITY_DRAM_ADDR=self.w["IDENTITY_640"])
            # o * tanh(c_new)
            self.accelerator_memory_to_sram(self.PRED_GATES_DRAM + 3 * H * bpe, URAM_A_BASE, H)
            self.accelerator_memory_to_sram(self.PRED_OUT_DRAM, URAM_B_BASE, H)
            self.eltwise_mul_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, H)
            self.sram_to_accelerator_memory(URAM_A_BASE, h_addr, H)  # h_new

        # Output = h1
        self.accelerator_memory_to_sram(self.PRED_H1_DRAM, URAM_A_BASE, H)
        self.sram_to_accelerator_memory(URAM_A_BASE, self.PRED_OUT_DRAM, H)

        self.stop_capture()
        self.generate_instruction_halt()

        program_addr = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(program_addr)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        return program_addr, flops

    # -----------------------------------------------------------------------
    # Compile: Joint network
    # -----------------------------------------------------------------------

    def compile_joint(self):
        """Compile joint network as two programs for separate hardware argmax.

        Program 1 (joint_tok): projections + add + ReLU + token logits matmul
          → hardware argmax register = token_id
        Program 2 (joint_dur): duration logits matmul (reuses JOINT_SUM from prog 1)
          → hardware argmax register = dur_idx

        Returns (tok_prog, dur_prog, flops).
        """
        D = self.d_model
        H = self.joint_hidden   # 640
        N_tok_pad = pad_to_multiple(self.vocab_size, self.block_size)  # 8256
        N_dur_pad = pad_to_multiple(len(self.tdt_durations), self.block_size)  # 64
        flops = 0

        # --- Program 1: projections + token logits ---
        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()

        # enc_proj: (1, 1024) @ (1024, 640) + bias
        self.matmat_mul_core(M=1, K=D, N=H,
            A_DRAM_ADDR=self.JOINT_ENC_DRAM, B_DRAM_ADDR=self.w["JOINT_ENC_W"],
            OUTPUT_DRAM_ADDR=self.JOINT_ENC_DRAM, C_DRAM_ADDR=self.w["JOINT_ENC_B"])
        flops += 2 * D * H

        # pred_proj: (1, 640) @ (640, 640) + bias
        self.matmat_mul_core(M=1, K=H, N=H,
            A_DRAM_ADDR=self.JOINT_PRED_DRAM, B_DRAM_ADDR=self.w["JOINT_PRED_W"],
            OUTPUT_DRAM_ADDR=self.JOINT_PRED_DRAM, C_DRAM_ADDR=self.w["JOINT_PRED_B"])
        flops += 2 * H * H

        # add + ReLU
        self.accelerator_memory_to_sram(self.JOINT_ENC_DRAM, URAM_A_BASE, H)
        self.accelerator_memory_to_sram(self.JOINT_PRED_DRAM, URAM_B_BASE, H)
        self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, H)
        self.sram_to_accelerator_memory(URAM_A_BASE, self.JOINT_SUM_DRAM, H)
        self.matmat_mul_core(M=1, K=H, N=H,
            A_DRAM_ADDR=self.JOINT_SUM_DRAM, B_DRAM_ADDR=self.w["IDENTITY_640"],
            OUTPUT_DRAM_ADDR=self.JOINT_SUM_DRAM, relu_enable=True)

        # Token logits: (1, 640) @ (640, 8256) + bias → hardware argmax = token_id
        self.matmat_mul_core(M=1, K=H, N=N_tok_pad,
            A_DRAM_ADDR=self.JOINT_SUM_DRAM, B_DRAM_ADDR=self.w["JOINT_OUT_TOK_W"],
            OUTPUT_DRAM_ADDR=self.JOINT_TOK_DRAM, C_DRAM_ADDR=self.w["JOINT_OUT_TOK_B"])
        flops += 2 * H * N_tok_pad

        self.stop_capture()
        self.generate_instruction_halt()
        tok_prog = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(tok_prog)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())

        # --- Program 2: duration logits (JOINT_SUM already computed) ---
        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()

        # Duration logits: (1, 640) @ (640, 64) + bias → hardware argmax = dur_idx
        self.matmat_mul_core(M=1, K=H, N=N_dur_pad,
            A_DRAM_ADDR=self.JOINT_SUM_DRAM, B_DRAM_ADDR=self.w["JOINT_OUT_DUR_W"],
            OUTPUT_DRAM_ADDR=self.JOINT_DUR_DRAM, C_DRAM_ADDR=self.w["JOINT_OUT_DUR_B"])
        flops += 2 * H * N_dur_pad

        self.stop_capture()
        self.generate_instruction_halt()
        dur_prog = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(dur_prog)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())

        return tok_prog, dur_prog, flops

    # -----------------------------------------------------------------------
    # Run: full pipeline
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Program execution helper
    # -----------------------------------------------------------------------

    def program_execute(self, program_addr, timeout=50.0, flops=None):
        """Execute a compiled program from DRAM. Returns (latency_us, gflops)."""
        self.start_execute_from_dram(program_addr)
        latency_us, gflops = 0, 0
        if timeout > 0:
            self.wait_queue(timeout)
            latency_us = self.report_latency_in_us()
            if flops:
                gflops = self.report_flop_rate_gflops(flops)
        return latency_us, gflops

    def get_arg_max_index(self):
        """Read hardware argmax register (set by last matmul output)."""
        return self.read_reg32(user_dma_core.UE_ARGMAX_INDEX)

    # -----------------------------------------------------------------------
    # CPU subsampling
    # -----------------------------------------------------------------------

    def run_subsampling_cpu(self, mel_bf16):
        """Run subsampling on CPU using staged weights in DRAM.
        mel_bf16: (1, T_mel, 128) bf16
        Returns: (1, L, 1024) bf16 where L = T_mel // 8
        """
        bpe = self.bytes_per_element
        B, T_mel, n_mels = mel_bf16.shape
        x = mel_bf16.unsqueeze(1).float()  # (1, 1, T_mel, 128)

        def read_weight(addr, shape):
            t = torch.zeros(shape, dtype=torch.bfloat16)
            self.dma_read("/dev/xdma0_c2h_0", addr, t, t.numel() * bpe)
            return t.float()

        # Stage 1: Conv2d(1→256, k=3, s=2) + ReLU
        # Using standard F.conv2d since this is CPU
        w0 = read_weight(self.w["SUB_CONV0_W"], (256, 9)).reshape(256, 1, 3, 3)
        b0 = read_weight(self.w["SUB_CONV0_B"], (256,))
        x = F.relu(F.conv2d(x, w0, b0, stride=2, padding=1))

        # Stage 2: DWConv2d(256, k=3, s=2) + PWConv2d(256→256) + ReLU
        w2 = read_weight(self.w["SUB_CONV2_W"], (256, 9)).reshape(256, 1, 3, 3)
        b2 = read_weight(self.w["SUB_CONV2_B"], (256,))
        x = F.conv2d(x, w2, b2, stride=2, padding=1, groups=256)
        w3 = read_weight(self.w["SUB_CONV3_W"], (256, 256)).reshape(256, 256, 1, 1)
        b3 = read_weight(self.w["SUB_CONV3_B"], (256,))
        x = F.relu(F.conv2d(x, w3, b3))

        # Stage 3: DWConv2d(256, k=3, s=2) + PWConv2d(256→256) + ReLU
        w5 = read_weight(self.w["SUB_CONV5_W"], (256, 9)).reshape(256, 1, 3, 3)
        b5 = read_weight(self.w["SUB_CONV5_B"], (256,))
        x = F.conv2d(x, w5, b5, stride=2, padding=1, groups=256)
        w6 = read_weight(self.w["SUB_CONV6_W"], (256, 256)).reshape(256, 256, 1, 1)
        b6 = read_weight(self.w["SUB_CONV6_B"], (256,))
        x = F.relu(F.conv2d(x, w6, b6))

        # Flatten: (B, 256, T/8, 16) → (B, T/8, 4096)
        B, C, T_out, Freq = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T_out, C * Freq)

        # Linear: (B, T/8, 4096) @ (1024, 4096)^T + bias → (B, T/8, 1024)
        w_out = read_weight(self.w["SUB_OUT_W"], (1024, 4096))
        b_out = read_weight(self.w["SUB_OUT_B"], (1024,))
        x = torch.matmul(x, w_out.t()) + b_out

        return x.to(torch.bfloat16)  # (1, L, 1024)

    # -----------------------------------------------------------------------
    # Positional embedding (CPU, sinusoidal)
    # -----------------------------------------------------------------------

    def make_rel_pos_emb(self, seq_len):
        """Generate relative positional encoding: (2*seq_len-1, D) bf16."""
        D = self.d_model
        max_len = max(seq_len, 2 * seq_len)
        pe = torch.zeros(max_len, D)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, D, 2, dtype=torch.float32)
                        * -(math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe_pos = pe[:seq_len]                               # (T, D)
        pe_neg = torch.flip(pe[1:seq_len], [0])             # (T-1, D)
        rel_pe = torch.cat([pe_neg, pe_pos], dim=0)         # (2T-1, D)
        return rel_pe.to(torch.bfloat16)

    # -----------------------------------------------------------------------
    # Run: encoder
    # -----------------------------------------------------------------------

    def run_encoder(self, mel_bf16):
        """Run full encoder: CPU subsampling → accelerator conformer layers.
        mel_bf16: (1, T_mel, 128) bf16 tensor
        Returns: (enc_out_addr, L, L_pad)
        """
        T_mel = mel_bf16.shape[1]
        L = T_mel // 8
        L_pad = pad_to_multiple(L, self.block_size)
        bpe = self.bytes_per_element
        D = self.d_model

        # 1. Compile encoder for this L_pad bucket
        enc_prog, enc_flops = self.compile_encoder(L_pad)
        print(f"  Encoder compiled: L={L}, L_pad={L_pad}, "
              f"program={self.get_capture_instruction_size_bytes()/1024:.0f}KB, "
              f"flops={enc_flops/1e9:.1f}G")

        # 2. CPU subsampling: mel → (1, L, 1024)
        print("  Running subsampling on CPU...")
        sub_out = self.run_subsampling_cpu(mel_bf16)  # (1, L, 1024)
        L_actual = sub_out.shape[1]

        # Pad to L_pad if needed
        if L_actual < L_pad:
            sub_padded = torch.zeros(1, L_pad, D, dtype=torch.bfloat16)
            sub_padded[:, :L_actual, :] = sub_out
            sub_out = sub_padded
        # Write subsampling output to INPUT_DRAM as (L_pad, D)
        self.dma_to_accelerator_memory(self.INPUT_DRAM,
            sub_out.squeeze(0).contiguous())  # (L_pad, D)

        # 3. Generate and write positional embedding
        P = 2 * L_pad - 1
        P_pad = pad_to_multiple(P, self.block_size)
        rel_pe = self.make_rel_pos_emb(L_pad)  # (2*L_pad-1, D)
        # Pad rows to P_pad
        if rel_pe.shape[0] < P_pad:
            pe_padded = torch.zeros(P_pad, D, dtype=torch.bfloat16)
            pe_padded[:rel_pe.shape[0], :] = rel_pe
            rel_pe = pe_padded
        self.dma_to_accelerator_memory(self.POS_EMB_DRAM, rel_pe.contiguous())

        # 4. Generate and write attention padding mask (L_pad, L_pad)
        # Conformer uses full bidirectional attention — mask only blocks padding.
        # mask[i,j] = 0 if i < L and j < L, else -inf
        mask = torch.full((L_pad, L_pad), float("-inf"), dtype=torch.bfloat16)
        mask[:L, :L] = 0.0
        self.dma_to_accelerator_memory(self.ATTN_MASK_DRAM, mask.contiguous())

        # 5. Execute conformer layers on accelerator
        print("  Executing encoder on accelerator...")
        latency_us, gflops = self.program_execute(enc_prog, flops=enc_flops)
        print(f"  Encoder: {latency_us:.0f} us, {gflops:.1f} GFLOPS")

        return self.ENC_OUT_DRAM, L, L_pad

    # -----------------------------------------------------------------------
    # Run: decoder (TDT greedy)
    # -----------------------------------------------------------------------

    def run_decode(self, enc_out_addr, L):
        """Run TDT greedy decode loop.
        Host-side control loop; data movement stays on-device (DRAM↔SRAM).
        Hardware argmax for both token and duration via split joint programs.
        Returns: list of token IDs
        """
        bpe = self.bytes_per_element
        H = self.pred_hidden
        D = self.d_model

        # Compile predictor + split joint (fixed dims, once)
        pred_prog, pred_flops = self.compile_predictor_step()
        tok_prog, dur_prog, joint_flops = self.compile_joint()
        print(f"  Decoder compiled: pred_flops={pred_flops/1e6:.1f}M, "
              f"joint_flops={joint_flops/1e6:.1f}M")

        # Init LSTM state to zeros
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
                # --- Embedding lookup: on-device DRAM → SRAM → DRAM ---
                emb_src = self.w["EMBED"] + last_token * H * bpe
                self.accelerator_memory_to_sram(emb_src, URAM_A_BASE, H)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.PRED_EMB_DRAM, H)

                # --- Run predictor ---
                self.program_execute(pred_prog, flops=pred_flops)

                # --- Copy enc_out[t] to JOINT_ENC_DRAM (on-device) ---
                self.accelerator_memory_to_sram(
                    enc_out_addr + t * D * bpe, URAM_A_BASE, D)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.JOINT_ENC_DRAM, D)

                # --- Copy pred_out to JOINT_PRED_DRAM (on-device) ---
                self.accelerator_memory_to_sram(self.PRED_OUT_DRAM, URAM_A_BASE, H)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.JOINT_PRED_DRAM, H)

                # --- Run joint token program → hardware argmax = token_id ---
                self.program_execute(tok_prog)
                token_id = self.get_arg_max_index()

                # --- Run joint duration program → hardware argmax = dur_idx ---
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
                t += 1  # max symbols reached

        print(f"  Decode: {total_steps} joint steps, {len(tokens)} tokens emitted")
        return tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parakeet-TDT-0.6B accelerator inference")
    parser.add_argument("--audio", type=str, default=None, help="Path to 16kHz WAV file")
    parser.add_argument("--dev", type=str, default="xdma0", help="XDMA device")
    parser.add_argument("--cycle", type=float, default=5.63, help="Clock cycle in ns")
    args = parser.parse_args()

    cfg = load_config()
    set_dma_device(args.dev)

    print(f"Parakeet-TDT-0.6B on {args.dev}")
    print(f"  Encoder: {cfg['encoder']['num_layers']} conformer layers, d_model={cfg['encoder']['d_model']}")
    print(f"  Predictor: {cfg['predictor']['num_layers']}-layer LSTM, hidden={cfg['predictor']['hidden_size']}")
    print(f"  Joint output: {cfg['joint']['output_size']} → {cfg['joint']['output_size_padded']} (padded)")
    print(f"  Hardware block size: {cfg['hardware']['block_size']}")

    # --- Load audio ---
    if args.audio:
        import torchaudio
        waveform, sr = torchaudio.load(args.audio)
        if sr != cfg["preprocessing"]["sample_rate"]:
            waveform = torchaudio.functional.resample(waveform, sr, cfg["preprocessing"]["sample_rate"])
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
    else:
        duration_s = 5
        waveform = torch.randn(1, cfg["preprocessing"]["sample_rate"] * duration_s)
        print(f"  Using dummy {duration_s}s audio")

    print(f"  Audio: {waveform.shape[1]} samples ({waveform.shape[1]/cfg['preprocessing']['sample_rate']:.1f}s)")

    # --- Mel spectrogram (CPU) ---
    mel = compute_mel_spectrogram(waveform, cfg)
    T_mel = mel.shape[1]
    L = T_mel // 8
    L_pad = pad_to_multiple(L, cfg["hardware"]["block_size"])
    print(f"  Mel frames: {T_mel}, encoder steps: L={L}, L_pad={L_pad}")

    # --- Init engine + load weights ---
    engine = Parakeet_UnifiedEngine()
    engine.weight_init()

    # --- Run encoder ---
    print(f"\n--- Encoder ---")
    enc_out_addr, L, L_pad = engine.run_encoder(mel)

    # --- Run decoder ---
    print(f"\n--- Decoder ---")
    tokens = engine.run_decode(enc_out_addr, L)

    # --- Detokenize ---
    tokenizer_path = TOKENIZER_PATH
    if os.path.exists(tokenizer_path):
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(tokenizer_path)
        text = sp.DecodeIds(tokens)
        print(f"\n  >>> {text}\n")
    else:
        print(f"  Tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
        print(f"  (tokenizer not found at {tokenizer_path})")


if __name__ == "__main__":
    main()
