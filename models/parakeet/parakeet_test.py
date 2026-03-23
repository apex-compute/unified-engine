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
  user_dma_core.py is at the repo root (two folders up); that directory is added to sys.path.
"""

import json
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

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

# ---------------------------------------------------------------------------
# Debug: compare hardware vs CPU reference
# ---------------------------------------------------------------------------
def check_nan(name, tensor):
    """Assert-halt if tensor contains NaN or Inf. Call on any intermediate result."""
    f = tensor.float().flatten()
    nan_c = torch.isnan(f).sum().item()
    inf_c = torch.isinf(f).sum().item()
    if nan_c or inf_c:
        total = f.numel()
        # Find first NaN/Inf index
        bad_mask = torch.isnan(f) | torch.isinf(f)
        first_bad = bad_mask.nonzero(as_tuple=True)[0][0].item()
        # Show surrounding context
        lo = max(0, first_bad - 2)
        hi = min(total, first_bad + 3)
        context = f[lo:hi].tolist()
        assert False, (
            f"NaN/Inf DETECTED in {name}: "
            f"{nan_c} NaN, {inf_c} Inf out of {total} elements. "
            f"First bad index: {first_bad}. "
            f"Values [{lo}:{hi}]: {context}"
        )


def compare_tensors(name, hw_tensor, ref_tensor, cos_threshold=0.99):
    """Compare hw vs ref tensors. Asserts on NaN or cosine similarity failure."""
    h = hw_tensor.float().flatten()
    r = ref_tensor.float().flatten()
    # NaN/Inf check before comparison
    check_nan(f"{name} (hw)", hw_tensor)
    check_nan(f"{name} (ref)", ref_tensor)
    min_len = min(len(h), len(r))
    h, r = h[:min_len], r[:min_len]
    cos = F.cosine_similarity(h.unsqueeze(0), r.unsqueeze(0)).item()
    mae = (h - r).abs().mean().item()
    max_err = (h - r).abs().max().item()
    r_norm = r.norm().item()
    h_norm = h.norm().item()
    info = (f"{name}: cos={cos:.6f}  mae={mae:.4f}  max_err={max_err:.4f}  "
            f"hw_norm={h_norm:.2f}  ref_norm={r_norm:.2f}")
    if cos >= cos_threshold:
        print(f"  [PASS] {info}")
    else:
        # Build detailed error message
        diff = (h - r).abs()
        worst_idx = diff.topk(min(5, len(diff))).indices
        details = "\n".join(
            f"    [{idx.item()}] hw={h[idx.item()]:.6f} ref={r[idx.item()]:.6f} diff={diff[idx.item()]:.6f}"
            for idx in worst_idx)
        assert False, f"MISMATCH {info}\n  Worst elements:\n{details}"
    return cos
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
    # broadcast_mul/add expect raw float scalars (start_queue_broadcast does bf16 encoding)
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

        # Extract columns [col_start:col_end] from each of M rows into temp as (M, chunk_cols_pad)
        for row in range(M):
            src = input_dram_addr + (row * N + col_start) * bpe
            dst = temp_dram_addr + row * chunk_cols_pad * bpe
            ue.accelerator_memory_to_sram(src, URAM_A_BASE, chunk_cols)
            ue.sram_to_accelerator_memory(URAM_A_BASE, dst, chunk_cols)

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
    row_bytes = N * 2
    for m in range(M):
        # Load ff_output into URAM_A, multiply by 0.5
        ue.accelerator_memory_to_sram(FF_DRAM_ADDR + m * row_bytes, URAM_A_BASE, N)
        ue.broadcast_mul(scalar=0.5, sram_start_addr=URAM_A_BASE,
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
PARAKEET_PARAMS_BASE  = 0x00000000   # 2.5 GB for weights + identities + staged
PARAKEET_TENSOR_BASE  = 0xA0000000   # 750 MB for intermediate activations
PARAKEET_PROGRAM_BASE = 0xCEE00000   # 750 MB for compiled instruction programs


class Parakeet_UnifiedEngine(UnifiedEngine):
    """UnifiedEngine subclass for Parakeet-TDT-0.6B."""

    def __init__(self, script_dir=None, debug=False):
        super().__init__(BASE_ADDR=user_dma_core.UE_0_BASE_ADDR,
                         params_dram_base=PARAKEET_PARAMS_BASE,
                         tensor_dram_base=PARAKEET_TENSOR_BASE,
                         program_dram_base=PARAKEET_PROGRAM_BASE)
        self.script_dir = script_dir or SCRIPT_DIR
        self.debug = debug
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

    # XDMA driver has a transfer size limit (~16-64MB per os.write call).
    # Chunk large DMA uploads to avoid silent failures.
    DMA_CHUNK_BYTES = 1 * 1024 * 1024  # 1 MB per chunk — XDMA driver may silently drop larger writes

    def dma_to_accelerator_memory(self, dma_address, data):
        """Chunked DMA write that handles large transfers."""
        assert data.dtype == torch.bfloat16, "Data must be in bf16 format"
        total_bytes = data.numel() * 2
        if total_bytes <= self.DMA_CHUNK_BYTES:
            ret = self.dma_write(DMA_DEVICE_H2C, dma_address, data, total_bytes)
            if ret != total_bytes:
                print(f"  WARNING: dma_write returned {ret}, expected {total_bytes}")
            return
        n_chunks = (total_bytes + self.DMA_CHUNK_BYTES - 1) // self.DMA_CHUNK_BYTES
        print(f"  [DMA chunked] {total_bytes/1024**2:.1f} MB → {n_chunks} chunks "
              f"of {self.DMA_CHUNK_BYTES/1024**2:.0f} MB to 0x{dma_address:08X}")
        flat = data.contiguous().flatten()
        offset = 0
        elems_per_chunk = self.DMA_CHUNK_BYTES // 2  # bf16 = 2 bytes
        chunk_idx = 0
        while offset < flat.numel():
            chunk_elems = min(elems_per_chunk, flat.numel() - offset)
            chunk = flat[offset:offset + chunk_elems].contiguous()
            addr = dma_address + offset * 2
            ret = self.dma_write(DMA_DEVICE_H2C, addr, chunk, chunk_elems * 2)
            if ret != chunk_elems * 2:
                print(f"    chunk {chunk_idx}: FAILED wrote {ret}/{chunk_elems*2} bytes "
                      f"at 0x{addr:08X}")
            chunk_idx += 1
            offset += chunk_elems
        print(f"    {chunk_idx} chunks written")

    def _alloc_write(self, tensor):
        """Allocate params DRAM, write bf16 tensor via chunked DMA. Returns DRAM address."""
        t = tensor.to(torch.bfloat16).contiguous()
        addr = self.get_params_dram_addr()
        nbytes = t.numel() * 2
        self.allocate_params_dram(nbytes)
        # Use chunked write to avoid XDMA driver silent failures on large tensors
        self._chunked_dma_write(addr, t)
        return addr

    def _chunked_dma_write(self, dma_address, data):
        """Write bf16 tensor to DRAM in 1MB chunks."""
        flat = data.contiguous().flatten()
        total_bytes = flat.numel() * 2
        chunk_bytes = self.DMA_CHUNK_BYTES
        if total_bytes <= chunk_bytes:
            ret = self.dma_write(DMA_DEVICE_H2C, dma_address, flat, total_bytes)
            if ret != total_bytes:
                print(f"  WARNING: dma_write returned {ret}, expected {total_bytes} "
                      f"at 0x{dma_address:08X}")
            return
        elems_per_chunk = chunk_bytes // 2
        offset = 0
        while offset < flat.numel():
            chunk_elems = min(elems_per_chunk, flat.numel() - offset)
            chunk = flat[offset:offset + chunk_elems].contiguous()
            addr = dma_address + offset * 2
            self.dma_write(DMA_DEVICE_H2C, addr, chunk, chunk_elems * 2)
            offset += chunk_elems
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
        # The permute function loads UE_VECTOR_SIZE * UE_VECTOR_SIZE (64*64)
        # bytes assuming 64-element rows. A larger identity (e.g. 1024x1024)
        # has 1024-element rows, so the loaded block is garbled.
        self.w["IDENTITY_64"] = allocate_identity(self, UE_VECTOR_SIZE)

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
        # Padding biases must be -inf so padding positions never win argmax.
        # Zero padding logits (0) beat negative real logits, causing the decoder
        # to always pick padding index 8193 instead of the correct token.
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

        # Verify conformer weights survived DMA upload
        bpe = self.bytes_per_element
        print("  Verifying conformer weights in DRAM...")
        for li in [0, 12, 23]:  # spot-check first, middle, last layers
            la = self.layer_addrs[li]
            for key, ckpt_key in [("FF1_W1", f"encoder.layers.{li}.feed_forward1.linear1.weight"),
                                   ("ATTN_Q_W", f"encoder.layers.{li}.self_attn.linear_q.weight")]:
                ref_w = sd[ckpt_key].to(torch.bfloat16).flatten()
                hw_w = torch.zeros(ref_w.numel(), dtype=torch.bfloat16)
                self.dma_read(DMA_DEVICE_C2H, la[key], hw_w, ref_w.numel() * bpe)
                match = (hw_w == ref_w).sum().item()
                total = ref_w.numel()
                if match != total:
                    print(f"    FAIL layer {li} {key}: {match}/{total} match")
                    print(f"      hw[:5]={hw_w[:5].float().tolist()} ref[:5]={ref_w[:5].float().tolist()}")
                else:
                    print(f"    OK   layer {li} {key}: {total} elements match")

    def tensor_init(self, L_pad):
        """Allocate intermediate DRAM buffers."""
        # Zero entire tensor DRAM region up front to prevent stale NaN.
        # Must happen BEFORE allocations so that identity matrices (written
        # during allocation) are not overwritten.
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
        # Subsampling intermediates (host builds im2col patches, HW runs matmuls)
        T_mel_max = L_pad * 8
        H0, W0 = T_mel_max // 2, self.n_mels // 2  # after stage 0
        H1, W1 = H0 // 2, W0 // 2                  # after stage 1
        N0 = H0 * W0
        N1 = H1 * W1
        # Pad spatial dims to block_size for aligned DW matmuls
        N1_pad = pad_to_multiple(N1, self.block_size)
        H2, W2_tmp = H1 // 2, W1 // 2
        N2 = H2 * W2_tmp
        N2_pad = pad_to_multiple(N2, self.block_size)
        N_dw_max_pad = max(N1_pad, N2_pad)
        # Shared patch buffer: max(stage0 patches, stage1/2 DW patches with padding)
        max_patch_size = max(N0 * 64, SC * N_dw_max_pad * 64)
        self.SUB_PATCH_DRAM = self.allocate_tensor_dram(max_patch_size * bpe)
        self.SUB_OUT0_DRAM = self.allocate_tensor_dram(N0 * SC * bpe)  # stage 0 output
        self.SUB_DW_OUT_DRAM = self.allocate_tensor_dram(SC * N_dw_max_pad * bpe)  # DW output (padded)
        self.SUB_PW_IN_DRAM = self.allocate_tensor_dram(N_dw_max_pad * SC * bpe)  # after permute for PW (padded)
        self.SUB_FLAT_DRAM = self.allocate_tensor_dram(L_pad * 4096 * bpe)
        # Debug: per-layer snapshots (allocated only when debug=True)
        self.DEBUG_LAYER_DRAM = []
        if self.debug:
            for _ in range(self.num_layers):
                self.DEBUG_LAYER_DRAM.append(
                    self.allocate_tensor_dram(L_pad * D * bpe))
        # Decoder intermediates
        self.PRED_EMB_DRAM = self.allocate_tensor_dram(H * bpe)
        self.PRED_GATES_DRAM = self.allocate_tensor_dram(4 * H * bpe)
        self.PRED_GATES2_DRAM = self.allocate_tensor_dram(4 * H * bpe)
        self.PRED_H0_DRAM = self.allocate_tensor_dram(H * bpe)
        self.PRED_C0_DRAM = self.allocate_tensor_dram(H * bpe)
        self.PRED_H1_DRAM = self.allocate_tensor_dram(H * bpe)
        self.PRED_C1_DRAM = self.allocate_tensor_dram(H * bpe)
        self.PRED_OUT_DRAM = self.allocate_tensor_dram(H * bpe)
        self.JOINT_ENC_DRAM = self.allocate_tensor_dram(D * bpe)  # holds raw enc_out[t] (1024) before projection
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
        # DW bias: broadcast_add per channel (pass raw float, not pre-encoded bf16)
        dw_bias = torch.zeros(SC, dtype=torch.bfloat16)
        self.dma_read(DMA_DEVICE_C2H, self.w[dw_b_key], dw_bias, SC * bpe)
        for ch in range(SC):
            out_addr = self.SUB_DW_OUT_DRAM + ch * N_out_pad * bpe
            self.accelerator_memory_to_sram(out_addr, URAM_A_BASE, N_out_pad)
            self.broadcast_add(scalar=dw_bias[ch].float().item(),
                sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=N_out_pad)
            self.sram_to_accelerator_memory(URAM_A_BASE, out_addr, N_out_pad)
        # Chunked on-device transpose: (SC, N_out_pad) -> (N_out_pad, SC)
        # Process N_out_pad in chunks of SC (256) so each sub-transpose is (SC, SC) which fits URAM.
        chunk = SC  # 256
        for c_start in range(0, N_out_pad, chunk):
            c_end = min(c_start + chunk, N_out_pad)
            c_len = c_end - c_start
            c_len_pad = pad_to_multiple(c_len, self.block_size)
            # Extract (SC, c_len) slice from SUB_DW_OUT and transpose to (c_len_pad, SC)
            # Input: SC rows, each N_out_pad wide, we want columns [c_start:c_end]
            # Use strided read per channel row: read c_len elements at offset c_start
            for ch in range(SC):
                src = self.SUB_DW_OUT_DRAM + (ch * N_out_pad + c_start) * bpe
                # Write contiguously into temp as (SC, c_len_pad) for transpose
                dst = self.PERMUTE_TEMP_DRAM + ch * c_len_pad * bpe
                self.accelerator_memory_to_sram(src, URAM_A_BASE, c_len)
                self.sram_to_accelerator_memory(URAM_A_BASE, dst, c_len)
            # Transpose (SC, c_len_pad) -> (c_len_pad, SC) — both dims <= 256, fits URAM
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
        Single-channel: returns (N_out, 64) bf16.  Depthwise: returns (C, N_out_pad, 64)
        where N_out_pad is padded to the next multiple of block_size for aligned matmuls.
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
            N_pad = pad_to_multiple(N, self.block_size)
            patches = torch.zeros(C, N_pad, 64, dtype=torch.bfloat16)
            patches[:, :N, :9] = cols.permute(0, 2, 1).to(torch.bfloat16)  # (C, N, 9)
            return patches, H_out, W_out

    def compile_encoder(self, L_pad, max_layers=None, max_sublayer=None):
        """Monolithic encoder compile: conformer layers inline. Returns (prog, flops).
        max_layers: if set, only compile this many layers (for debugging).
        max_sublayer: if set with max_layers=1, stop after this sub-module in layer 0:
            1=FF1, 2=+Attention, 3=+ConvModule, 4=+FF2, 5=+FinalLN (complete layer)
        """
        D, FF, H_heads, dk = self.d_model, self.ff_dim, self.num_heads, self.head_dim
        bpe = self.bytes_per_element
        P = 2 * L_pad - 1
        P_pad = pad_to_multiple(P, self.block_size)

        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()
        total_flops = 0

        # --- conformer layers, all ops inline ---
        n_layers = max_layers if max_layers is not None else self.num_layers
        print(f"  Compiling {n_layers}/{self.num_layers} conformer layers"
              f"{f' (sublayer stop at {max_sublayer})' if max_sublayer else ''}")
        for layer_idx in range(n_layers):
            la = self.layer_addrs[layer_idx]

            # ===== FF1 (half-step residual) =====
            self.layer_norm_core_dram(M=L_pad, N=D,
                A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.LN_OUT_DRAM,
                GAMMA_DRAM_ADDR=la["LN_FF1_WEIGHT"], BETA_DRAM_ADDR=la["LN_FF1_BIAS"])

            if max_sublayer is not None and max_sublayer <= 0 and layer_idx == 0:
                # Copy LN output to ENC_OUT so we can read it back
                self.accelerator_memory_to_sram(self.LN_OUT_DRAM, URAM_A_BASE, L_pad * D)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.INPUT_DRAM, L_pad * D)
                print(f"  [sublayer stop] after LayerNorm only"); break

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

            if max_sublayer is not None and max_sublayer <= 1 and layer_idx == 0:
                print(f"  [sublayer stop] after FF1"); break

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
                    INPUT_DRAM_ADDR=self.CONV_OUT_DRAM, OUTPUT_DRAM_ADDR=self.REL_SHIFT_DRAM,
                    input_row_stride=P_pad)
                # Combine content + pos
                score_elems = L_pad * L_pad
                self.accelerator_memory_to_sram(self.SCORE_DRAM, URAM_A_BASE, score_elems)
                self.accelerator_memory_to_sram(self.REL_SHIFT_DRAM, URAM_B_BASE, score_elems)
                self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, score_elems)
                self.sram_to_accelerator_memory(URAM_A_BASE, self.SCORE_DRAM, score_elems)
                # Scale
                self.accelerator_memory_to_sram(self.SCORE_DRAM, URAM_A_BASE, score_elems)
                inv_sqrt_dk = 1.0 / math.sqrt(dk)
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
                # Use chunked transpose: dk=128 > UE_VECTOR_SIZE=64, so
                # bf16_smart_permute_core's dot-product would corrupt the result.
                chunked_transpose_core_dram(self, M=L_pad, N=dk,
                    input_dram_addr=self.CONV_B_DRAM, output_dram_addr=self.ATTN_VT_DRAM,
                    identity_dram_addr=self.w["IDENTITY_64"],
                    temp_dram_addr=self.PERMUTE_TEMP_DRAM)
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

            if max_sublayer is not None and max_sublayer <= 2 and layer_idx == 0:
                print(f"  [sublayer stop] after Attention"); break

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
            # Transpose (L_pad, D) -> (D, L_pad) for channel-first DW conv.
            # D=1024 >> UE_VECTOR_SIZE=64, so use chunked transpose.
            chunked_transpose_core_dram(self, M=L_pad, N=D,
                input_dram_addr=self.CONV_A_DRAM, output_dram_addr=self.CONV_T_DRAM,
                identity_dram_addr=self.w["IDENTITY_64"],
                temp_dram_addr=self.PERMUTE_TEMP_DRAM)
            # DW Conv1d via shift-multiply-accumulate (9 taps, padding=4)
            # For each channel: output[t] = sum_{k=0}^{8} kernel[k] * input[t + k - 4]
            # Read kernel scalars to host for broadcast_mul
            dw_kernel_host = torch.zeros(D * 64, dtype=torch.bfloat16)
            self.dma_read(DMA_DEVICE_C2H, la["CONV_DW_W"], dw_kernel_host, D * 64 * bpe)
            dw_kernel_2d = dw_kernel_host.reshape(D, 64)
            conv_pad = self.conv_pad  # 4
            K = 2 * conv_pad + 1     # 9
            for ch in range(D):
                ch_in = self.CONV_T_DRAM + ch * L_pad * bpe
                ch_out = self.CONV_DW_DRAM + ch * L_pad * bpe
                # Zero-initialize output row (L_pad is already 64-aligned)
                self.accelerator_memory_to_sram(ch_in, URAM_A_BASE, L_pad)
                self.broadcast_mul(scalar=0.0, sram_start_addr=URAM_A_BASE,
                    sram_wb_addr=URAM_A_BASE, element_size=L_pad)
                self.sram_to_accelerator_memory(URAM_A_BASE, ch_out, L_pad)
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
                    # Read shifted input to SRAM_A (length elements from DRAM).
                    # SRAM region is length_aligned; extra tail elements are stale
                    # but will be multiplied and written — safe because output row
                    # was zero-initialized and dst_start+length_aligned <= L_pad.
                    self.accelerator_memory_to_sram(ch_in + src_start * bpe, URAM_A_BASE, length_aligned)
                    # Multiply by kernel[k] scalar (convert bf16 raw to float)
                    scalar = dw_kernel_2d[ch, k].float().item()
                    self.broadcast_mul(scalar=scalar, sram_start_addr=URAM_A_BASE,
                        sram_wb_addr=URAM_A_BASE, element_size=length_aligned)
                    # Accumulate: read existing output, add, write back
                    self.accelerator_memory_to_sram(ch_out + dst_start * bpe, URAM_B_BASE, length_aligned)
                    self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, length_aligned)
                    self.sram_to_accelerator_memory(URAM_A_BASE, ch_out + dst_start * bpe, length_aligned)
                total_flops += 2 * K * L_pad
            batch_norm_core_dram(self, C=D, L=L_pad,
                A_DRAM_ADDR=self.CONV_DW_DRAM, OUTPUT_DRAM_ADDR=self.CONV_DW_DRAM,
                SCALE_DRAM_ADDR=la["CONV_BN_SCALE"], SHIFT_DRAM_ADDR=la["CONV_BN_SHIFT"])
            # SiLU: x * sigmoid(x). Cannot be in-place because silu_core_dram
            # Step 2 (sigmoid) overwrites the input, destroying x before Step 3
            # can multiply x * sigmoid(x). Use CONV_OUT as temp for sigmoid output.
            silu_core_dram(self, M=D, N=L_pad,
                A_DRAM_ADDR=self.CONV_DW_DRAM, OUTPUT_DRAM_ADDR=self.CONV_OUT_DRAM,
                IDENTITY_DRAM_ADDR=self.IDENTITY_LPAD_DRAM)
            # Transpose (D, L_pad) -> (L_pad, D) directly from SiLU output.
            # No need to copy back to CONV_DW — read from CONV_OUT instead.
            chunked_transpose_core_dram(self, M=D, N=L_pad,
                input_dram_addr=self.CONV_OUT_DRAM, output_dram_addr=self.CONV_T_DRAM,
                identity_dram_addr=self.w["IDENTITY_64"],
                temp_dram_addr=self.PERMUTE_TEMP_DRAM)
            self.matmat_mul_core(M=L_pad, K=D, N=D,
                A_DRAM_ADDR=self.CONV_T_DRAM, B_DRAM_ADDR=la["CONV_PW2_W"],
                OUTPUT_DRAM_ADDR=self.CONV_OUT_DRAM)
            total_flops += 2 * L_pad * D * D
            self.accelerator_memory_to_sram(self.INPUT_DRAM, URAM_A_BASE, L_pad * D)
            self.accelerator_memory_to_sram(self.CONV_OUT_DRAM, URAM_B_BASE, L_pad * D)
            self.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * D)
            self.sram_to_accelerator_memory(URAM_A_BASE, self.INPUT_DRAM, L_pad * D)
            total_flops += L_pad * D

            if max_sublayer is not None and max_sublayer <= 3 and layer_idx == 0:
                print(f"  [sublayer stop] after ConvModule"); break

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

            if max_sublayer is not None and max_sublayer <= 4 and layer_idx == 0:
                print(f"  [sublayer stop] after FF2"); break

            # ===== Final LayerNorm =====
            self.layer_norm_core_dram(M=L_pad, N=D,
                A_DRAM_ADDR=self.INPUT_DRAM, OUTPUT_DRAM_ADDR=self.INPUT_DRAM,
                GAMMA_DRAM_ADDR=la["LN_OUT_WEIGHT"], BETA_DRAM_ADDR=la["LN_OUT_BIAS"])

            # Debug: snapshot this layer's output
            if self.debug and self.DEBUG_LAYER_DRAM:
                self.accelerator_memory_to_sram(self.INPUT_DRAM, URAM_A_BASE, L_pad * D)
                self.sram_to_accelerator_memory(URAM_A_BASE,
                    self.DEBUG_LAYER_DRAM[layer_idx], L_pad * D)

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

    def compile_all(self, T_mel, L_pad, max_layers=None, max_sublayer=None):
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

        # Encoder (conformer layers)
        prog, f = self.compile_encoder(L_pad, max_layers=max_layers, max_sublayer=max_sublayer)
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

        def _stat(name, tensor):
            """Print tensor stats for debugging."""
            f = tensor.float()
            nan_c = torch.isnan(f).sum().item()
            inf_c = torch.isinf(f).sum().item()
            if nan_c or inf_c:
                print(f"    {name}: numel={f.numel()} NaN={nan_c} Inf={inf_c}")
            else:
                print(f"    {name}: numel={f.numel()} "
                      f"norm={f.norm():.4f} min={f.min():.4f} max={f.max():.4f} "
                      f"mean={f.mean():.6f}")

        def _read_dram(addr, numel):
            buf = torch.zeros(numel, dtype=torch.bfloat16)
            self.dma_read(DMA_DEVICE_C2H, addr, buf, numel * bpe)
            return buf

        # --- CPU reference subsampling for comparison ---
        # Run host-side PyTorch to get expected values at each stage
        print("  --- CPU reference subsampling ---")
        ref_x = mel_bf16.unsqueeze(1).float()  # (B, 1, T_mel, 128) — add channel dim
        # Stage 0 reference: Conv2d(1->256, k=3, s=2) + bias + ReLU
        sd = self._ckpt_sd
        ref_s0 = F.conv2d(ref_x, sd["encoder.pre_encode.conv.0.weight"].float(),
                          sd["encoder.pre_encode.conv.0.bias"].float(),
                          stride=2, padding=1)
        ref_s0 = F.relu(ref_s0)  # (1, 256, H0_ref, W0_ref)
        _stat("REF stage0", ref_s0)

        # --- Subsampling stage 0 ---
        patches0, H0, W0 = self._im2col_conv2d(mel_2d, T_mel, self.n_mels)
        N0 = H0 * W0
        # Pre-zero output buffer to prevent stale NaN from accumulating into matmul result
        zero_buf = torch.zeros(N0 * SC, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.SUB_OUT0_DRAM, zero_buf)
        self.dma_to_accelerator_memory(self.SUB_PATCH_DRAM, patches0.contiguous())
        check_nan("sub_stage0_patches (host)", patches0)
        self.program_execute(self.progs["sub0"][0])

        out0 = torch.zeros(N0 * SC, dtype=torch.bfloat16)
        self.dma_read(DMA_DEVICE_C2H, self.SUB_OUT0_DRAM, out0, N0 * SC * bpe)
        _stat("HW  stage0", out0)
        check_nan("sub_stage0_output", out0)
        out0_3d = out0.reshape(H0, W0, SC).permute(2, 0, 1).contiguous()

        # --- Stages 1 & 2: run on CPU in bf16 ---
        # DW im2col patches are too large for DMA (~262MB for stage 1).
        # Run DW+PW conv on CPU using F.conv2d — cheap one-time ops.
        # Use bf16 throughout to match the reference model's precision.
        # (float32 intermediates cause subsampling cos mismatch with bf16 ref.)
        cpu_s0 = out0.reshape(H0, W0, SC).permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16)
        cpu_s1 = F.conv2d(cpu_s0, sd["encoder.pre_encode.conv.2.weight"].to(torch.bfloat16),
                          sd["encoder.pre_encode.conv.2.bias"].to(torch.bfloat16),
                          stride=2, padding=1, groups=SC)
        cpu_s1 = F.conv2d(cpu_s1, sd["encoder.pre_encode.conv.3.weight"].to(torch.bfloat16),
                          sd["encoder.pre_encode.conv.3.bias"].to(torch.bfloat16))
        cpu_s1 = F.relu(cpu_s1)

        cpu_s2 = F.conv2d(cpu_s1, sd["encoder.pre_encode.conv.5.weight"].to(torch.bfloat16),
                          sd["encoder.pre_encode.conv.5.bias"].to(torch.bfloat16),
                          stride=2, padding=1, groups=SC)
        cpu_s2 = F.conv2d(cpu_s2, sd["encoder.pre_encode.conv.6.weight"].to(torch.bfloat16),
                          sd["encoder.pre_encode.conv.6.bias"].to(torch.bfloat16))
        cpu_s2 = F.relu(cpu_s2)

        _, _, H2, W2 = cpu_s2.shape
        flat = cpu_s2.squeeze(0).permute(1, 2, 0).reshape(H2, SC * W2).to(torch.bfloat16).contiguous()
        if H2 < L_pad:
            flat_padded = torch.zeros(L_pad, 4096, dtype=torch.bfloat16)
            flat_padded[:H2, :] = flat
            flat = flat_padded

        self.dma_to_accelerator_memory(self.SUB_FLAT_DRAM, flat.contiguous())

        # Verify flat upload
        flat_verify = _read_dram(self.SUB_FLAT_DRAM, min(1024, flat.numel()))
        flat_match = (flat_verify == flat.flatten()[:flat_verify.numel()]).sum().item()
        print(f"    Flat upload verify: {flat_match}/{flat_verify.numel()} match")

        # --- Subsampling linear ---
        # Pre-zero INPUT_DRAM (output of linear) to prevent stale NaN accumulation
        zero_input = torch.zeros(L_pad * D, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.INPUT_DRAM, zero_input)
        self.program_execute(self.progs["sub_lin"][0])

        # Fill padding rows with values that have nonzero variance to prevent
        # LayerNorm NaN. A constant fill (like 1e-6) has var=0, causing
        # division by zero in LayerNorm. Alternating ±0.1 gives var > 0.
        if L_pad > L:
            pad_rows = L_pad - L
            # Alternating +0.1 / -0.1 pattern: mean ≈ 0, var ≈ 0.01
            epsilon_fill = torch.zeros(pad_rows * D, dtype=torch.bfloat16)
            epsilon_fill[0::2] = 0.1
            epsilon_fill[1::2] = -0.1
            pad_offset = L * D * bpe
            self.dma_to_accelerator_memory(self.INPUT_DRAM + pad_offset, epsilon_fill)

        lin_hw = _read_dram(self.INPUT_DRAM, L_pad * D)
        _stat("HW  linear_out", lin_hw[:L * D])
        check_nan("sub_linear_output", lin_hw[:L * D])
        # Save for debug comparison — conformer will overwrite INPUT_DRAM
        self._hw_sub_out = lin_hw.clone().reshape(L_pad, D)

        # CPU reference linear for comparison
        ref_linear = F.linear(flat[:H2].float(), sd["encoder.pre_encode.out.weight"].float(),
                              sd["encoder.pre_encode.out.bias"].float())
        _stat("REF linear_out", ref_linear)
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
        # Valid rows attend to valid columns; padding columns are blocked.
        # Padding rows (L..L_pad-1) must attend to at least one position to
        # avoid softmax(all -inf) = NaN. Let them attend to position 0
        # (harmless dummy attention — output is masked by downstream ops).
        mask = torch.full((L_pad, L_pad), -1e38, dtype=torch.bfloat16)
        mask[:L, :L] = 0.0                # valid positions attend normally
        mask[L:, 0] = 0.0                 # padding rows attend to pos 0 (prevents NaN)
        self.dma_to_accelerator_memory(self.ATTN_MASK_DRAM, mask.contiguous())

        # Verify layer 0 FF1 weights in DRAM
        sd = self._ckpt_sd
        la0 = self.layer_addrs[0]
        for wname, ckpt_key in [("FF1_W1", "encoder.layers.0.feed_forward1.linear1.weight"),
                                 ("FF1_W2", "encoder.layers.0.feed_forward1.linear2.weight"),
                                 ("LN_FF1_WEIGHT", "encoder.layers.0.norm_feed_forward1.weight"),
                                 ("LN_FF1_BIAS", "encoder.layers.0.norm_feed_forward1.bias")]:
            ref_w = sd[ckpt_key].to(torch.bfloat16).flatten()
            hw_w = torch.zeros(ref_w.numel(), dtype=torch.bfloat16)
            self.dma_read(DMA_DEVICE_C2H, la0[wname], hw_w, ref_w.numel() * bpe)
            match = (hw_w == ref_w).sum().item()
            hw_f = hw_w.float()
            ref_f = ref_w.float()
            print(f"    Weight {wname}: {match}/{ref_w.numel()} match  "
                  f"hw=[{hw_f[:3].tolist()}] ref=[{ref_f[:3].tolist()}]  "
                  f"hw_norm={hw_f.norm():.4f} ref_norm={ref_f.norm():.4f}")

        # Verify mask upload
        mask_read = _read_dram(self.ATTN_MASK_DRAM, L_pad * L_pad)
        mask_2d = mask_read.float().reshape(L_pad, L_pad)
        nan_mask = torch.isnan(mask_2d).sum().item()
        inf_mask = torch.isinf(mask_2d).sum().item()
        # Check padding row L has at least one non-negative-huge entry
        if L < L_pad:
            pad_row_max = mask_2d[L, :].max().item()
            pad_row_min = mask_2d[L, :].min().item()
            print(f"    Mask: nan={nan_mask} inf={inf_mask} "
                  f"valid_block[0,0]={mask_2d[0,0]:.1f} valid_block[0,L-1]={mask_2d[0,L-1]:.1f} "
                  f"pad_col[0,L]={mask_2d[0,L]:.1f} "
                  f"pad_row[L,0]={mask_2d[L,0]:.1f} pad_row[L,1]={mask_2d[L,1]:.1f}")
        else:
            print(f"    Mask: nan={nan_mask} inf={inf_mask} L==L_pad, no padding rows")

        # Verify INPUT_DRAM before conformer
        inp_pre = _read_dram(self.INPUT_DRAM, L_pad * D)
        _stat("INPUT pre-conformer", inp_pre[:L * D])
        if L < L_pad:
            _stat("INPUT padding rows", inp_pre[L * D:])

        # --- Pre-zero all encoder intermediate buffers to prevent stale NaN ---
        print("  Pre-zeroing encoder intermediate buffers...")
        zero_bufs = {
            "ENC_OUT":     (self.ENC_OUT_DRAM,     L_pad * D),
            "LN_OUT":      (self.LN_OUT_DRAM,      L_pad * D),
            "FF_MID":      (self.FF_MID_DRAM,      L_pad * self.ff_dim),
            "FF_OUT":      (self.FF_OUT_DRAM,       L_pad * D),
            "RESIDUAL":    (self.RESIDUAL_DRAM,     L_pad * D),
            "Q":           (self.Q_DRAM,            L_pad * D),
            "K":           (self.K_DRAM,            L_pad * D),
            "V":           (self.V_DRAM,            L_pad * D),
            "POS_PROJ":    (self.POS_PROJ_DRAM,     L_pad * D),
            "SCORE":       (self.SCORE_DRAM,        L_pad * L_pad),
            "ATTN_OUT":    (self.ATTN_OUT_DRAM,     L_pad * D),
            "REL_SHIFT":   (self.REL_SHIFT_DRAM,    L_pad * L_pad),
            "CONV_A":      (self.CONV_A_DRAM,       L_pad * D),
            "CONV_B":      (self.CONV_B_DRAM,       L_pad * D),
            "CONV_T":      (self.CONV_T_DRAM,       D * L_pad),
            "CONV_DW":     (self.CONV_DW_DRAM,      D * L_pad),
            "CONV_OUT":    (self.CONV_OUT_DRAM,      L_pad * D),
        }
        for buf_name, (addr, numel) in zero_bufs.items():
            z = torch.zeros(numel, dtype=torch.bfloat16)
            self.dma_to_accelerator_memory(addr, z)

        # --- Execute 24 conformer layers ---
        print("  Executing conformer layers...")
        enc_prog, enc_flops = self.progs["encoder"]
        latency_us, gflops = self.program_execute(enc_prog, flops=enc_flops)
        print(f"  Encoder: {latency_us:.0f} us, {gflops:.1f} GFLOPS")

        # Check encoder output
        enc_out = _read_dram(self.ENC_OUT_DRAM, L_pad * D)
        _stat("ENC_OUT", enc_out[:L * D])

        # --- Post-execution intermediate buffer NaN scan ---
        # These buffers retain whatever was last written during the program.
        # For sublayer=2 (FF1+Attention), this tells us exactly which attention
        # sub-operation first produced NaN.
        FF = self.ff_dim
        P_pad = pad_to_multiple(2 * L_pad - 1, self.block_size)
        dk = self.head_dim
        print("\n  --- Intermediate buffer NaN scan ---")
        scan_bufs = [
            ("LN_OUT",      self.LN_OUT_DRAM,      L_pad * D,       "last LayerNorm output"),
            ("FF_MID",      self.FF_MID_DRAM,       L_pad * FF,      "FF linear1+SiLU"),
            ("FF_OUT",      self.FF_OUT_DRAM,        L_pad * D,       "FF linear2 / attn out proj"),
            ("Q",           self.Q_DRAM,             L_pad * D,       "Q projection"),
            ("K",           self.K_DRAM,             L_pad * D,       "K projection"),
            ("V",           self.V_DRAM,             L_pad * D,       "V projection"),
            ("POS_PROJ",    self.POS_PROJ_DRAM,      L_pad * D,       "Pos projection (allocated size)"),
            ("SCORE",       self.SCORE_DRAM,         L_pad * L_pad,   "Attention scores (last head)"),
            ("REL_SHIFT",   self.REL_SHIFT_DRAM,     L_pad * L_pad,   "Rel shift output (last head)"),
            ("ATTN_OUT",    self.ATTN_OUT_DRAM,      L_pad * D,       "Multi-head concat"),
            ("CONV_A",      self.CONV_A_DRAM,        L_pad * dk,      "Temp A (last head attn@V)"),
            ("CONV_B",      self.CONV_B_DRAM,        L_pad * dk,      "Temp B (last head K/V)"),
            ("CONV_OUT",    self.CONV_OUT_DRAM,       L_pad * P_pad,   "Pos scores pre-relshift (last head)"),
            ("ATTN_VT",     self.ATTN_VT_DRAM,       dk * L_pad,      "V transposed (last head)"),
            ("ATTN_MASK",   self.ATTN_MASK_DRAM,     L_pad * L_pad,   "Attention mask"),
            ("INPUT",       self.INPUT_DRAM,          L_pad * D,       "Final encoder state"),
            ("IDENTITY_LP", self.IDENTITY_LPAD_DRAM,  L_pad * L_pad,   "Identity matrix (L_pad)"),
            ("POS_EMB",     self.POS_EMB_DRAM,        P_pad * D,       "Positional embeddings"),
        ]
        for name, addr, numel, desc in scan_bufs:
            buf = _read_dram(addr, numel)
            f = buf.float()
            nan_c = torch.isnan(f).sum().item()
            inf_c = torch.isinf(f).sum().item()
            if nan_c or inf_c:
                print(f"    {name:12s} [{numel:>8d}]: NaN={nan_c:>6d} Inf={inf_c:>4d}  -- {desc}")
            else:
                print(f"    {name:12s} [{numel:>8d}]: CLEAN  norm={f.norm():.2f}  "
                      f"min={f.min():.4f} max={f.max():.4f}  -- {desc}")

        check_nan("encoder_output", enc_out[:L * D])

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
        # Pre-zero all decoder intermediate buffers to prevent stale NaN
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
                # Embedding lookup via host DMA (on-device SRAM ops may not
                # sync with program execution, causing stale reads)
                emb_src = self.w["EMBED"] + last_token * H * bpe
                emb_buf = torch.zeros(H, dtype=torch.bfloat16)
                self.dma_read(DMA_DEVICE_C2H, emb_src, emb_buf, H * bpe)
                self.dma_to_accelerator_memory(self.PRED_EMB_DRAM, emb_buf)
                # Predictor
                self.program_execute(pred_prog)
                # Copy enc_out[t] and pred_out to joint inputs via HOST DMA
                # (direct SRAM ops may not sync with program execution)
                enc_t_buf = torch.zeros(D, dtype=torch.bfloat16)
                self.dma_read(DMA_DEVICE_C2H, enc_out_addr + t * D * bpe, enc_t_buf, D * bpe)
                self.dma_to_accelerator_memory(self.JOINT_ENC_DRAM, enc_t_buf)
                pred_buf = torch.zeros(H, dtype=torch.bfloat16)
                self.dma_read(DMA_DEVICE_C2H, self.PRED_OUT_DRAM, pred_buf, H * bpe)
                self.dma_to_accelerator_memory(self.JOINT_PRED_DRAM, pred_buf)
                # Joint token -> hardware argmax
                self.program_execute(tok_prog)
                token_id = self.get_arg_max_index()
                # Joint duration -> hardware argmax
                self.program_execute(dur_prog)
                dur_idx = self.get_arg_max_index()
                dur = self.tdt_durations[dur_idx] if dur_idx < len(self.tdt_durations) else 0
                total_steps += 1
                # Diagnostic: dump first 3 steps
                if total_steps <= 3:
                    # Read back key tensors for diagnosis
                    hw_pred = torch.zeros(H, dtype=torch.bfloat16)
                    self.dma_read(DMA_DEVICE_C2H, self.PRED_OUT_DRAM, hw_pred, H * bpe)
                    check_nan(f"decode_step{total_steps}_pred_out", hw_pred)
                    hw_enc_t = torch.zeros(D, dtype=torch.bfloat16)
                    self.dma_read(DMA_DEVICE_C2H, enc_out_addr + t * D * bpe, hw_enc_t, D * bpe)
                    check_nan(f"decode_step{total_steps}_enc_t{t}", hw_enc_t)
                    hw_tok = torch.zeros(N_tok_pad, dtype=torch.bfloat16)
                    self.dma_read(DMA_DEVICE_C2H, self.JOINT_TOK_DRAM, hw_tok, N_tok_pad * bpe)
                    check_nan(f"decode_step{total_steps}_tok_logits", hw_tok[:self.vocab_size])
                    hw_dur_logits = torch.zeros(N_dur_pad, dtype=torch.bfloat16)
                    self.dma_read(DMA_DEVICE_C2H, self.JOINT_DUR_DRAM, hw_dur_logits, N_dur_pad * bpe)
                    print(f"  [step {total_steps}] t={t} tok={token_id} dur={dur} "
                          f"blank={self.blank_id}")
                    print(f"    pred_out: norm={hw_pred.float().norm():.4f} "
                          f"mean={hw_pred.float().mean():.6f} "
                          f"min={hw_pred.float().min():.4f} max={hw_pred.float().max():.4f}")
                    print(f"    enc[t]:   norm={hw_enc_t.float().norm():.4f} "
                          f"mean={hw_enc_t.float().mean():.6f}")
                    tok_f = hw_tok[:self.vocab_size].float()
                    print(f"    tok_logits: argmax={tok_f.argmax().item()} "
                          f"max={tok_f.max():.4f} min={tok_f.min():.4f} "
                          f"blank_score={tok_f[self.blank_id]:.4f} "
                          f"top5={tok_f.topk(5).indices.tolist()}")
                    dur_f = hw_dur_logits[:len(self.tdt_durations)].float()
                    print(f"    dur_logits: {dur_f.tolist()} argmax={dur_f.argmax().item()}")
                    # Check if enc output varies across time
                    if total_steps == 1:
                        enc0 = hw_enc_t.clone()
                        hw_enc_t1 = torch.zeros(D, dtype=torch.bfloat16)
                        if L > 1:
                            self.dma_read(DMA_DEVICE_C2H, enc_out_addr + 1 * D * bpe,
                                          hw_enc_t1, D * bpe)
                            cos01 = F.cosine_similarity(
                                enc0.float().unsqueeze(0),
                                hw_enc_t1.float().unsqueeze(0)).item()
                            print(f"    enc cos(t=0,t=1): {cos01:.6f}")
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
def _load_reference_model():
    """Load the CPU reference ParakeetTDT model for debug comparison."""
    REF_DIR = "/home/rohit/apex-compute-ML/simple-llm/src/parakeet"
    sys.path.insert(0, REF_DIR)
    from parakeet_modules import ParakeetTDT
    from parakeet_main import load_checkpoint as ref_load_checkpoint, convert_to_bf16
    ref = ParakeetTDT()
    ref = ref_load_checkpoint(ref, WEIGHTS_PATH)
    ref = convert_to_bf16(ref)
    ref.eval()
    ref.stage_for_hardware()
    return ref


def debug_encoder(engine, ref_model, mel, L, L_pad, waveform=None):
    """Run encoder with per-stage CPU comparison."""
    bpe = engine.bytes_per_element
    D = engine.d_model
    print("\n=== DEBUG: Encoder Comparison ===")

    # --- Run reference encoder step by step ---
    with torch.no_grad():
        if waveform is not None:
            ref_mel = ref_model.preprocessor(waveform.float())
        else:
            # mel is already computed; use it directly as ref_mel
            ref_mel = mel if mel.dim() == 3 else mel.unsqueeze(0)
    compare_tensors("mel_spectrogram", mel.squeeze(0), ref_mel.squeeze(0))

    # --- Run reference subsampling step by step ---
    sub = ref_model.encoder.subsampling
    with torch.no_grad():
        ref_x = ref_mel.unsqueeze(1)  # (1, 1, T, 128)
        # Stage 0
        from parakeet_modules import conv2d_standard_staged, depthwise_conv2d_staged, pointwise_conv2d
        ref_s0 = conv2d_standard_staged(ref_x, sub._conv0_staged, sub.conv0_b,
                                         sub._conv0_patch_size, stride=2, padding=1)
        ref_s0 = F.relu(ref_s0)

    # --- Execute HW subsampling stage 0 ---
    enc_out_addr = engine.run_encoder(mel, L, L_pad)

    # --- Read back HW encoder output and compare ---
    enc_hw = torch.zeros(L_pad * D, dtype=torch.bfloat16)
    engine.dma_read(DMA_DEVICE_C2H, engine.ENC_OUT_DRAM, enc_hw, L_pad * D * bpe)
    enc_hw = enc_hw.reshape(L_pad, D)

    # Run full reference encoder
    with torch.no_grad():
        ref_enc = ref_model.encoder(ref_mel)  # (1, L_ref, 1024)
    ref_enc_2d = ref_enc.squeeze(0)  # (L_ref, 1024)
    L_ref = ref_enc_2d.shape[0]
    L_cmp = min(L, L_ref)

    print(f"\n--- Encoder Output (L={L}, L_ref={L_ref}) ---")
    # Don't assert here — report and continue to per-layer comparison
    h = enc_hw[:L_cmp].float().flatten()
    r = ref_enc_2d[:L_cmp].float().flatten()
    cos = F.cosine_similarity(h.unsqueeze(0), r.unsqueeze(0)).item()
    mae = (h - r).abs().mean().item()
    print(f"  encoder_output: cos={cos:.6f}  mae={mae:.4f}  "
          f"hw_norm={h.norm():.2f}  ref_norm={r.norm():.2f}")
    if cos < 0.99:
        print(f"  WARNING: encoder output mismatch (cos={cos:.4f}), checking per-layer...")

    # Check if HW encoder output varies across timesteps
    enc_hw_f = enc_hw[:L].float()
    ts_var = enc_hw_f.var(dim=0).mean().item()
    ts_cos_01 = F.cosine_similarity(enc_hw_f[0:1], enc_hw_f[1:2]).item() if L > 1 else 0
    ts_cos_0m = F.cosine_similarity(enc_hw_f[0:1], enc_hw_f[L//2:L//2+1]).item() if L > 2 else 0
    print(f"  HW enc temporal variance: {ts_var:.6f}")
    print(f"  HW enc cos(t=0, t=1): {ts_cos_01:.6f}")
    print(f"  HW enc cos(t=0, t=L/2): {ts_cos_0m:.6f}")

    # --- Per-layer debug if buffers were allocated ---
    if engine.debug and engine.DEBUG_LAYER_DRAM:
        print(f"\n--- Per-Layer Comparison (24 conformer layers) ---")
        # Use HW subsampling output as shared starting point for fair comparison.
        # This isolates conformer errors from subsampling differences.
        hw_sub = engine._hw_sub_out[:L].to(torch.bfloat16)  # (L, 1024)
        # Also run ref subsampling for comparison
        with torch.no_grad():
            ref_sub_out = ref_model.encoder.subsampling(ref_mel)  # (1, L_ref, 1024)
        L_ref = ref_sub_out.shape[1]
        L_cmp_sub = min(L, L_ref)
        sub_cos = F.cosine_similarity(
            hw_sub[:L_cmp_sub].float().flatten().unsqueeze(0),
            ref_sub_out.squeeze(0)[:L_cmp_sub].float().flatten().unsqueeze(0)).item()
        print(f"  Subsampling output cos(HW, REF): {sub_cos:.6f}  "
              f"(L_hw={L}, L_ref={L_ref})")

        from parakeet_modules import make_rel_pos_emb
        # Feed HW sub output to ref conformer layers for fair comparison
        ref_layer_x = hw_sub.unsqueeze(0).to(ref_sub_out.dtype)  # (1, L, 1024)
        ref_pos = make_rel_pos_emb(L).to(dtype=ref_layer_x.dtype, device=ref_layer_x.device)

        with torch.no_grad():
            for li in range(engine.num_layers):
                ref_layer_x = ref_model.encoder.layers[li](ref_layer_x, ref_pos)
                # Read HW layer output
                hw_layer = torch.zeros(L_pad * D, dtype=torch.bfloat16)
                engine.dma_read(DMA_DEVICE_C2H, engine.DEBUG_LAYER_DRAM[li],
                                hw_layer, L_pad * D * bpe)
                check_nan(f"hw_layer_{li:02d}", hw_layer[:L * D])
                hw_layer = hw_layer.reshape(L_pad, D)
                ref_layer_2d = ref_layer_x.squeeze(0)
                L_cmp = min(L, ref_layer_2d.shape[0])
                hw_flat = hw_layer[:L_cmp].float().flatten()
                ref_flat = ref_layer_2d[:L_cmp].float().flatten()
                min_len = min(len(hw_flat), len(ref_flat))
                cos = F.cosine_similarity(hw_flat[:min_len].unsqueeze(0),
                                          ref_flat[:min_len].unsqueeze(0)).item()
                mae = (hw_flat[:min_len] - ref_flat[:min_len]).abs().mean().item()
                status = "PASS" if cos >= 0.95 else ("WARN" if cos >= 0.5 else "FAIL")
                print(f"  [{status}] layer_{li:02d}: cos={cos:.6f}  mae={mae:.4f}  "
                      f"hw_norm={hw_flat.norm():.2f}  ref_norm={ref_flat.norm():.2f}")

    # --- Also compare subsampling linear output ---
    sub_lin_hw = torch.zeros(L_pad * D, dtype=torch.bfloat16)
    engine.dma_read(DMA_DEVICE_C2H, engine.INPUT_DRAM, sub_lin_hw, L_pad * D * bpe)
    # NOTE: INPUT_DRAM now holds the final encoder output (overwritten by conformer layers).
    # The subsampling linear output was INPUT_DRAM before conformer started.
    # We can't read it after the fact. But per-layer comparison covers this.

    return enc_out_addr


def debug_decoder(engine, ref_model, enc_out_addr, L):
    """Run decoder with per-step CPU comparison (first N steps)."""
    bpe = engine.bytes_per_element
    H = engine.pred_hidden
    D = engine.d_model
    MAX_DEBUG_STEPS = 5
    print(f"\n=== DEBUG: Decoder Comparison (first {MAX_DEBUG_STEPS} steps) ===")

    # Read HW encoder output for reference joint
    enc_hw = torch.zeros(L * D, dtype=torch.bfloat16)
    engine.dma_read(DMA_DEVICE_C2H, enc_out_addr, enc_hw, L * D * bpe)
    enc_hw = enc_hw.reshape(L, D)

    # Also run reference encoder for ref decoder
    # (reuse ref_model's encoder output for clean comparison)

    # HW decode state
    zeros_h = torch.zeros(H, dtype=torch.bfloat16)
    engine.dma_to_accelerator_memory(engine.PRED_H0_DRAM, zeros_h)
    engine.dma_to_accelerator_memory(engine.PRED_C0_DRAM, zeros_h)
    engine.dma_to_accelerator_memory(engine.PRED_H1_DRAM, zeros_h)
    engine.dma_to_accelerator_memory(engine.PRED_C1_DRAM, zeros_h)

    pred_prog = engine.progs["pred"][0]
    tok_prog = engine.progs["joint_tok"][0]
    dur_prog = engine.progs["joint_dur"][0]

    # Reference decode state
    ref_state = None
    ref_last_token = engine.blank_id
    hw_last_token = engine.blank_id
    t = 0

    from parakeet_modules import VOCAB_SIZE, NUM_TDT_DURATIONS, BLANK_ID

    for step in range(MAX_DEBUG_STEPS):
        if t >= L:
            break
        print(f"\n  --- Decode step {step}, t={t}, last_token={hw_last_token} ---")

        # HW: embedding + predictor
        emb_src = engine.w["EMBED"] + hw_last_token * H * bpe
        engine.accelerator_memory_to_sram(emb_src, URAM_A_BASE, H)
        engine.sram_to_accelerator_memory(URAM_A_BASE, engine.PRED_EMB_DRAM, H)
        engine.program_execute(pred_prog)

        # Read HW predictor output
        hw_pred = torch.zeros(H, dtype=torch.bfloat16)
        engine.dma_read(DMA_DEVICE_C2H, engine.PRED_OUT_DRAM, hw_pred, H * bpe)

        # Reference predictor
        with torch.no_grad():
            ref_pred, ref_new_state = ref_model.predictor.step(ref_last_token, ref_state)
        compare_tensors(f"  pred_out[{step}]", hw_pred, ref_pred.squeeze(0))

        # HW: copy enc_out[t] + pred_out to joint inputs
        engine.accelerator_memory_to_sram(enc_out_addr + t * D * bpe, URAM_A_BASE, D)
        engine.sram_to_accelerator_memory(URAM_A_BASE, engine.JOINT_ENC_DRAM, D)
        engine.accelerator_memory_to_sram(engine.PRED_OUT_DRAM, URAM_A_BASE, H)
        engine.sram_to_accelerator_memory(URAM_A_BASE, engine.JOINT_PRED_DRAM, H)

        # HW: joint token
        engine.program_execute(tok_prog)
        hw_token_id = engine.get_arg_max_index()

        # HW: joint duration
        engine.program_execute(dur_prog)
        hw_dur_idx = engine.get_arg_max_index()
        hw_dur = engine.tdt_durations[hw_dur_idx] if hw_dur_idx < len(engine.tdt_durations) else 0

        # Read HW joint logits for comparison
        N_tok_pad = pad_to_multiple(engine.vocab_size, engine.block_size)
        hw_tok_logits = torch.zeros(N_tok_pad, dtype=torch.bfloat16)
        engine.dma_read(DMA_DEVICE_C2H, engine.JOINT_TOK_DRAM, hw_tok_logits, N_tok_pad * bpe)

        # Reference joint
        with torch.no_grad():
            ref_enc_t = enc_hw[t:t+1].unsqueeze(0) if False else ref_pred.new_zeros(1, D)
            # Use HW encoder output for fair comparison
            ref_enc_t = enc_hw[t:t+1].float().to(ref_pred.dtype)
            ref_logits = ref_model.joint(ref_enc_t, ref_pred)
        ref_tok_logits = ref_logits[:, :VOCAB_SIZE]
        ref_token_id = ref_tok_logits.argmax(dim=-1).item()
        ref_dur_logits = ref_logits[:, VOCAB_SIZE:VOCAB_SIZE + NUM_TDT_DURATIONS]
        ref_dur = engine.tdt_durations[ref_dur_logits.argmax(dim=-1).item()]

        compare_tensors(f"  tok_logits[{step}]", hw_tok_logits[:VOCAB_SIZE], ref_tok_logits.squeeze(0))
        print(f"  HW: token={hw_token_id} dur={hw_dur}  |  REF: token={ref_token_id} dur={ref_dur}")
        assert hw_token_id == ref_token_id, (
            f"Token mismatch at step {step}: HW={hw_token_id} REF={ref_token_id}"
        )
        assert hw_dur == ref_dur, (
            f"Duration mismatch at step {step}: HW={hw_dur} REF={ref_dur}"
        )

        # Advance state
        if hw_token_id == engine.blank_id:
            t += max(hw_dur, 1)
        else:
            hw_last_token = hw_token_id
            ref_last_token = hw_token_id  # track HW path
            ref_state = ref_new_state
            if hw_dur > 0:
                t += hw_dur
            else:
                t += 1  # prevent infinite loop in debug


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parakeet-TDT-0.6B accelerator inference")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file (.wav, .flac, etc.)")
    parser.add_argument("--dev", type=str, default="xdma0", help="XDMA device")
    parser.add_argument("--cycle", type=float, default=5.63, help="Clock cycle in ns")
    parser.add_argument("--debug", action="store_true", help="Run CPU reference comparison at every checkpoint")
    parser.add_argument("--layers", type=int, default=None, help="Only compile N conformer layers (debug)")
    parser.add_argument("--sublayer", type=int, default=None,
                        help="With --layers 1: stop after sub-module N (1=FF1 2=Attn 3=Conv 4=FF2 5=full)")
    parser.add_argument("--isolate", action="store_true",
                        help="Auto-isolate first NaN: run layer 0 sublayers 1..5, then layers 2..24")
    args = parser.parse_args()
    cfg = load_config()
    set_dma_device(args.dev)
    audio_path = args.audio or os.path.join(SCRIPT_DIR, cfg["defaults"]["default_audio"])
    print(f"Parakeet-TDT-0.6B on {args.dev}")
    print(f"  Encoder: {cfg['encoder']['num_layers']} conformer layers, d_model={cfg['encoder']['d_model']}")
    print(f"  Predictor: {cfg['predictor']['num_layers']}-layer LSTM, hidden={cfg['predictor']['hidden_size']}")
    print(f"  Joint: tok={cfg['joint']['output_size']} dur={len(cfg['joint']['tdt_durations'])}")
    print(f"  Block size: {cfg['hardware']['block_size']}")
    if args.debug:
        print(f"  DEBUG MODE: CPU reference comparison enabled")
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
    engine = Parakeet_UnifiedEngine(debug=args.debug)
    engine.weight_init()
    mel = compute_mel_spectrogram(waveform, cfg, ckpt_sd=engine._ckpt_sd)
    T_mel = mel.shape[1]
    L = T_mel // 8
    L_pad = pad_to_multiple(L, cfg["hardware"]["block_size"])
    print(f"  Mel: {T_mel} frames, L={L}, L_pad={L_pad}")

    # --- Compile all programs ---
    print(f"\n--- Compile ---")
    engine.tensor_init(L_pad)
    engine.compile_all(T_mel, L_pad, max_layers=args.layers, max_sublayer=args.sublayer)

    if args.isolate:
        # --- Isolate mode: find exactly where NaN first appears ---
        print(f"\n{'='*60}")
        print(f"  ISOLATE MODE: auto-searching for first NaN")
        print(f"{'='*60}")
        D = engine.d_model
        bpe = engine.bytes_per_element
        sublayer_names = {1: "FF1", 2: "Attention", 3: "ConvModule", 4: "FF2", 5: "FinalLN"}

        # Load CPU reference model for comparison
        ref_model = _load_reference_model()
        from parakeet_modules import make_rel_pos_emb, FeedForward, RelPosAttention, ConvModule

        def _run_and_check(max_layers, max_sublayer, label):
            """Recompile with given limits, run encoder, return (nan_free, hw_out, cos)."""
            engine.reset_tensor_dram_addr()
            engine.reset_program_dram_addr()
            engine.tensor_init(L_pad)
            engine.compile_all(T_mel, L_pad, max_layers=max_layers, max_sublayer=max_sublayer)
            try:
                enc_addr = engine.run_encoder(mel, L, L_pad)
            except AssertionError:
                enc_addr = engine.ENC_OUT_DRAM
            out = torch.zeros(L_pad * D, dtype=torch.bfloat16)
            engine.dma_read(DMA_DEVICE_C2H, enc_addr, out, L_pad * D * bpe)
            hw_valid = out[:L * D].float()
            nan_c = torch.isnan(hw_valid).sum().item()
            valid = hw_valid[~torch.isnan(hw_valid)]
            norm = valid.norm().item() if valid.numel() > 0 else float('nan')
            return nan_c == 0, out.reshape(L_pad, D), norm

        # Get HW subsampling output as shared starting point
        engine.reset_tensor_dram_addr()
        engine.reset_program_dram_addr()
        engine.tensor_init(L_pad)
        engine.compile_all(T_mel, L_pad, max_layers=1, max_sublayer=1)
        engine.run_encoder(mel, L, L_pad)
        hw_sub = engine._hw_sub_out[:L].clone()  # (L, 1024) bf16

        # Run CPU reference sublayers on the SAME input
        ref_x = hw_sub.unsqueeze(0).to(torch.bfloat16)  # (1, L, D)
        ref_pos = make_rel_pos_emb(L).to(dtype=ref_x.dtype)
        ref_block = ref_model.encoder.layers[0]

        # Compute ref sublayer outputs step by step
        with torch.no_grad():
            # FF1
            ref_after_ff1 = ref_x + 0.5 * ref_block.ff1(ref_block.ln_ff1(ref_x))
            # Attention
            ref_after_attn = ref_after_ff1 + ref_block.self_attn(
                ref_block.ln_attn(ref_after_ff1), ref_pos)
            # ConvModule
            ref_after_conv = ref_after_attn + ref_block.conv(
                ref_block.ln_conv(ref_after_attn))
            # FF2
            ref_after_ff2 = ref_after_conv + 0.5 * ref_block.ff2(
                ref_block.ln_ff2(ref_after_conv))
            # Final LN
            ref_after_ln = ref_block.ln_out(ref_after_ff2)

        ref_stages = {
            1: ("FF1",        ref_after_ff1.squeeze(0)),
            2: ("Attention",  ref_after_attn.squeeze(0)),
            3: ("ConvModule", ref_after_conv.squeeze(0)),
            4: ("FF2",        ref_after_ff2.squeeze(0)),
            5: ("FinalLN",    ref_after_ln.squeeze(0)),
        }

        # Run each sublayer and collect results
        results = []
        for sub in range(1, 6):
            name, ref_out = ref_stages[sub]
            label = f"after {name}"
            ok, hw_out, norm = _run_and_check(max_layers=1, max_sublayer=sub, label=label)
            if not ok:
                results.append((sub, name, "NaN", 0.0, 0.0, 0.0, 0.0))
                continue
            hw_flat = hw_out[:L].float().flatten()
            ref_flat = ref_out[:L].float().flatten()
            min_len = min(len(hw_flat), len(ref_flat))
            cos = F.cosine_similarity(
                hw_flat[:min_len].unsqueeze(0),
                ref_flat[:min_len].unsqueeze(0)).item()
            mae = (hw_flat[:min_len] - ref_flat[:min_len]).abs().mean().item()
            results.append((sub, name, "OK", cos, mae, hw_flat.norm().item(), ref_flat.norm().item()))

        # Print summary table
        print(f"\n{'='*72}")
        print(f"  SUBLAYER COMPARISON SUMMARY — Layer 0")
        print(f"  (HW vs CPU reference, same subsampling input)")
        print(f"{'='*72}")
        print(f"  {'Sub':>3s}  {'Name':<12s}  {'Status':<6s}  {'Cosine':>8s}  {'MAE':>8s}  {'HW norm':>10s}  {'REF norm':>10s}")
        print(f"  {'-'*3}  {'-'*12}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")
        for sub, name, status, cos, mae, hw_n, ref_n in results:
            if status == "NaN":
                print(f"  {sub:3d}  {name:<12s}  {'NaN':>6s}")
            else:
                flag = "PASS" if cos >= 0.95 else ("WARN" if cos >= 0.5 else "FAIL")
                print(f"  {sub:3d}  {name:<12s}  {flag:<6s}  {cos:8.4f}  {mae:8.4f}  {hw_n:10.2f}  {ref_n:10.2f}")
        print(f"{'='*72}")
        # Identify first failure
        for sub, name, status, cos, *_ in results:
            if status != "NaN" and cos < 0.95:
                print(f"\n  >>> First divergence: sublayer {sub} ({name}), cos={cos:.4f}")
                break

        # --- Detailed Attention Diagnostic ---
        # Run sublayer=2 (FF1+Attention), read back all intermediates, compare with CPU reference
        print(f"\n{'='*72}")
        print(f"  ATTENTION DIAGNOSTIC — Layer 0")
        print(f"{'='*72}")

        # Run HW with sublayer=2
        engine.reset_tensor_dram_addr()
        engine.reset_program_dram_addr()
        engine.tensor_init(L_pad)
        engine.compile_all(T_mel, L_pad, max_layers=1, max_sublayer=2)
        try:
            engine.run_encoder(mel, L, L_pad)
        except AssertionError:
            pass

        dk = engine.head_dim
        H_heads = engine.num_heads

        def _read(addr, numel):
            buf = torch.zeros(numel, dtype=torch.bfloat16)
            engine.dma_read(DMA_DEVICE_C2H, addr, buf, numel * bpe)
            return buf

        def _cos(hw, ref, name):
            h = hw.float().flatten()
            r = ref.float().flatten()
            n = min(len(h), len(r))
            c = F.cosine_similarity(h[:n].unsqueeze(0), r[:n].unsqueeze(0)).item()
            mae = (h[:n] - r[:n]).abs().mean().item()
            flag = "PASS" if c >= 0.95 else ("WARN" if c >= 0.5 else "FAIL")
            print(f"  [{flag}] {name:<30s} cos={c:.6f}  mae={mae:.4f}  "
                  f"hw_norm={h[:n].norm():.2f}  ref_norm={r[:n].norm():.2f}")
            return c

        # Compute CPU reference attention intermediates on SAME input
        ref_ff1_out = ref_after_ff1  # (1, L, D) — input to attention
        ref_block_attn = ref_block.self_attn
        with torch.no_grad():
            ref_ln_attn = ref_block.ln_attn(ref_ff1_out)  # (1, L, D)
            ref_q = ref_block_attn.linear_q(ref_ln_attn)  # (1, L, D)
            ref_k = ref_block_attn.linear_k(ref_ln_attn)
            ref_v = ref_block_attn.linear_v(ref_ln_attn)
            ref_p = ref_block_attn.linear_pos(ref_pos)    # (1, 2L-1, D)

            # Multi-head reshape
            B_r, T_r, _ = ref_q.shape
            ref_q_h = ref_q.view(B_r, T_r, H_heads, dk).transpose(1, 2)  # (1, H, L, dk)
            ref_k_h = ref_k.view(B_r, T_r, H_heads, dk).transpose(1, 2)
            ref_v_h = ref_v.view(B_r, T_r, H_heads, dk).transpose(1, 2)
            ref_p_h = ref_p.view(1, -1, H_heads, dk).transpose(1, 2)     # (1, H, 2L-1, dk)

            # Head 0 content scores
            bias_u = ref_block_attn.pos_bias_u.unsqueeze(1)  # (H, 1, dk)
            bias_v = ref_block_attn.pos_bias_v.unsqueeze(1)
            ref_qu = ref_q_h + bias_u
            ref_content_h0 = torch.matmul(ref_qu[0, 0:1], ref_k_h[0, 0:1].transpose(-2, -1))  # (1, L, L)

            ref_qv = ref_q_h + bias_v
            ref_pos_raw_h0 = torch.matmul(ref_qv[0, 0:1], ref_p_h[0, 0:1].transpose(-2, -1))  # (1, L, 2L-1)
            from parakeet_modules import rel_shift
            ref_pos_h0 = rel_shift(ref_pos_raw_h0.unsqueeze(0)).squeeze(0)  # (1, L, L)

            ref_scores_h0 = (ref_content_h0 + ref_pos_h0) * (1.0 / math.sqrt(dk))
            ref_softmax_h0 = F.softmax(ref_scores_h0, dim=-1)  # (1, L, L)

            # Full attention output
            ref_attn_out = ref_block_attn(ref_ln_attn, ref_pos)  # (1, L, D)

        # Read HW intermediates
        hw_ln = _read(engine.LN_OUT_DRAM, L_pad * D).reshape(L_pad, D)
        hw_q = _read(engine.Q_DRAM, L_pad * D).reshape(L_pad, D)
        hw_k = _read(engine.K_DRAM, L_pad * D).reshape(L_pad, D)
        hw_v = _read(engine.V_DRAM, L_pad * D).reshape(L_pad, D)
        hw_score = _read(engine.SCORE_DRAM, L_pad * L_pad).reshape(L_pad, L_pad)
        hw_attn_out = _read(engine.ATTN_OUT_DRAM, L_pad * D).reshape(L_pad, D)
        hw_ff_out = _read(engine.FF_OUT_DRAM, L_pad * D).reshape(L_pad, D)

        print(f"\n  Attention intermediates (valid rows only, L={L}):")
        _cos(hw_ln[:L], ref_ln_attn.squeeze(0), "LN_attn output")
        _cos(hw_q[:L], ref_q.squeeze(0), "Q projection")
        _cos(hw_k[:L], ref_k.squeeze(0), "K projection")
        _cos(hw_v[:L], ref_v.squeeze(0), "V projection")

        # Head 0 scores: SCORE_DRAM has the LAST head (head 7), not head 0.
        # But we can check the last head against ref
        last_h = H_heads - 1
        with torch.no_grad():
            ref_content_last = torch.matmul(ref_qu[0, last_h:last_h+1],
                                             ref_k_h[0, last_h:last_h+1].transpose(-2, -1))
            ref_pos_raw_last = torch.matmul(ref_qv[0, last_h:last_h+1],
                                             ref_p_h[0, last_h:last_h+1].transpose(-2, -1))
            ref_pos_last = rel_shift(ref_pos_raw_last.unsqueeze(0)).squeeze(0)
            ref_scores_last = (ref_content_last + ref_pos_last) * (1.0 / math.sqrt(dk))
            ref_softmax_last = F.softmax(ref_scores_last, dim=-1)

        _cos(hw_score[:L, :L], ref_softmax_last.squeeze(0)[:L, :L],
             f"Softmax scores (head {last_h})")

        # V transpose verification: check CONV_B (V_h) and ATTN_VT (transposed)
        hw_conv_b = _read(engine.CONV_B_DRAM, L_pad * dk).reshape(L_pad, dk)
        hw_attn_vt = _read(engine.ATTN_VT_DRAM, dk * L_pad).reshape(dk, L_pad)
        # CONV_B should have V_h for last head
        ref_v_last = ref_v_h[0, last_h]  # (L, dk)
        _cos(hw_conv_b[:L], ref_v_last[:L], f"V_h (head {last_h}, before transpose)")
        # ATTN_VT should be V_h transposed: (dk, L_pad)
        # Compare: ATTN_VT[:, :L] should match V_h[:L, :]^T
        _cos(hw_attn_vt[:, :L].reshape(-1), ref_v_last[:L].t().reshape(-1),
             f"V_h^T (head {last_h}, after transpose)")
        print(f"    CONV_B norm={hw_conv_b[:L].float().norm():.2f}  "
              f"ATTN_VT norm={hw_attn_vt[:, :L].float().norm():.2f}  "
              f"(should be equal if transpose preserves data)")

        # Per-head attn@V result: CONV_A has last head's output
        hw_conv_a = _read(engine.CONV_A_DRAM, L_pad * dk).reshape(L_pad, dk)
        with torch.no_grad():
            ref_attn_v_last = torch.matmul(ref_softmax_last, ref_v_last.unsqueeze(0))  # (1, L, dk)
        _cos(hw_conv_a[:L], ref_attn_v_last.squeeze(0)[:L],
             f"attn@V (head {last_h}, single head)")

        # Check a few specific positions in ATTN_OUT to verify strided write
        print(f"\n  Strided write check (ATTN_OUT layout):")
        for h_check in [0, 4, 7]:
            h_start = h_check * dk
            hw_slice = hw_attn_out[0, h_start:h_start+4].float()  # first row, 4 elements of head h
            print(f"    ATTN_OUT[0, head{h_check}:head{h_check}+4] = {hw_slice.tolist()}")
        # If strided write works, different heads should have different values
        # If broken (contiguous write), head 0-6 would be zero or from head 7

        _cos(hw_attn_out[:L], ref_attn_out.squeeze(0)[:L],
             "Attention output (pre-proj)")

        # Output projection: FF_OUT_DRAM has W_out @ attn_concat
        with torch.no_grad():
            ref_out_proj = ref_block_attn.linear_out(ref_attn_out)
        _cos(hw_ff_out[:L], ref_out_proj.squeeze(0)[:L],
             "Output projection")

        # Check: what does residual look like?
        hw_input = _read(engine.INPUT_DRAM, L_pad * D).reshape(L_pad, D)
        _cos(hw_input[:L], ref_after_attn.squeeze(0)[:L],
             "After residual (x + attn)")

        # --- Reprint sublayer summary table at the end ---
        print(f"\n{'='*72}")
        print(f"  SUBLAYER COMPARISON SUMMARY — Layer 0  (L={L}, L_pad={L_pad})")
        print(f"  (HW vs CPU reference, same subsampling input)")
        print(f"{'='*72}")
        print(f"  {'Sub':>3s}  {'Name':<12s}  {'Status':<6s}  {'Cosine':>8s}  {'MAE':>8s}  {'HW norm':>10s}  {'REF norm':>10s}")
        print(f"  {'-'*3}  {'-'*12}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")
        for sub, name, status, cos, mae, hw_n, ref_n in results:
            if status == "NaN":
                print(f"  {sub:3d}  {name:<12s}  {'NaN':>6s}")
            else:
                flag = "PASS" if cos >= 0.95 else ("WARN" if cos >= 0.5 else "FAIL")
                print(f"  {sub:3d}  {name:<12s}  {flag:<6s}  {cos:8.4f}  {mae:8.4f}  {hw_n:10.2f}  {ref_n:10.2f}")
        print(f"{'='*72}")

        # --- Multi-layer degradation test ---
        # Run with increasing layer counts to find where cos drops
        print(f"\n{'='*72}")
        print(f"  MULTI-LAYER DEGRADATION TEST  (L={L}, L_pad={L_pad})")
        print(f"{'='*72}")
        # Feed HW sub output to ref for fair comparison
        ref_layer_x = hw_sub.unsqueeze(0).to(torch.bfloat16)
        ref_pos_full = make_rel_pos_emb(L).to(dtype=ref_layer_x.dtype)
        layer_counts = [1, 2, 4, 8, 12, 16, 20, 24]
        for n_layers in layer_counts:
            engine.reset_tensor_dram_addr()
            engine.reset_program_dram_addr()
            engine.tensor_init(L_pad)
            engine.compile_all(T_mel, L_pad, max_layers=n_layers, max_sublayer=None)
            try:
                engine.run_encoder(mel, L, L_pad)
            except AssertionError:
                pass
            hw_out = torch.zeros(L_pad * D, dtype=torch.bfloat16)
            engine.dma_read(DMA_DEVICE_C2H, engine.ENC_OUT_DRAM, hw_out, L_pad * D * bpe)
            # Run ref for same number of layers
            ref_x_tmp = ref_layer_x.clone()
            with torch.no_grad():
                for li in range(n_layers):
                    ref_x_tmp = ref_model.encoder.layers[li](ref_x_tmp, ref_pos_full)
            hw_f = hw_out[:L * D].float()
            ref_f = ref_x_tmp.squeeze(0)[:L].float().flatten()
            nan_c = torch.isnan(hw_f).sum().item()
            if nan_c > 0:
                print(f"  {n_layers:2d} layers: NaN={nan_c}")
                continue
            cos_val = F.cosine_similarity(hw_f.unsqueeze(0), ref_f.unsqueeze(0)).item()
            print(f"  {n_layers:2d} layers: cos={cos_val:.6f}  "
                  f"hw_norm={hw_f.norm():.2f}  ref_norm={ref_f.norm():.2f}")
        print(f"{'='*72}")

        sys.exit(0)

    elif args.debug:
        # --- Debug mode: run with CPU reference comparison ---
        ref_model = _load_reference_model()

        enc_out_addr = debug_encoder(engine, ref_model, mel, L, L_pad, waveform=waveform)
        debug_decoder(engine, ref_model, enc_out_addr, L)

        # Also run full HW decode for final output
        print(f"\n--- Full HW Decoder ---")
        tokens = engine.run_decode(enc_out_addr, L)
    else:
        # --- Normal mode ---
        print(f"\n--- Encoder ---")
        enc_out_addr = engine.run_encoder(mel, L, L_pad)
        print(f"\n--- Decoder ---")
        tokens = engine.run_decode(enc_out_addr, L)

    # --- BYPASS TEST: Feed ref subsampling to HW encoder, and ref enc to HW decoder ---
    print(f"\n--- Bypass Tests ---")
    ref_model = _load_reference_model()
    with torch.no_grad():
        ref_mel = ref_model.preprocessor(waveform.float())
        ref_sub = ref_model.encoder.subsampling(ref_mel)  # (1, L_ref, 1024)
    L_ref = ref_sub.shape[1]
    print(f"  REF subsampling: shape={ref_sub.shape}, L_ref={L_ref}")

    # Test 1: Feed ref sub output to HW encoder (bypass HW subsampling)
    print(f"\n  TEST 1: REF subsampling → HW conformer → HW decoder")
    ref_sub_padded = torch.zeros(L_pad, engine.d_model, dtype=torch.bfloat16)
    L_use = min(L_ref, L_pad)  # might differ by 1
    ref_sub_padded[:L_use] = ref_sub.squeeze(0)[:L_use].to(torch.bfloat16)
    # Fill remaining padding rows
    if L_use < L_pad:
        for r in range(L_use, L_pad):
            ref_sub_padded[r, 0::2] = 0.1
            ref_sub_padded[r, 1::2] = -0.1
    engine.dma_to_accelerator_memory(engine.INPUT_DRAM, ref_sub_padded.contiguous())
    # Re-execute JUST the encoder conformer (skip subsampling)
    enc_prog, enc_flops = engine.progs["encoder"]
    engine.program_execute(enc_prog, flops=enc_flops)
    bypass_enc = torch.zeros(L_pad * engine.d_model, dtype=torch.bfloat16)
    engine.dma_read(DMA_DEVICE_C2H, engine.ENC_OUT_DRAM, bypass_enc,
                    L_pad * engine.d_model * engine.bytes_per_element)
    bypass_enc_2d = bypass_enc.reshape(L_pad, engine.d_model)[:L_use]
    # Compare with ref encoder
    with torch.no_grad():
        ref_enc = ref_model.encoder(ref_mel)
    ref_enc_2d = ref_enc.squeeze(0)[:L_use]
    bypass_cos = F.cosine_similarity(
        bypass_enc_2d.float().flatten().unsqueeze(0),
        ref_enc_2d.float().flatten().unsqueeze(0)).item()
    print(f"  Bypass encoder cos: {bypass_cos:.6f}  "
          f"hw_norm={bypass_enc_2d.float().norm():.4f}  "
          f"ref_norm={ref_enc_2d.float().norm():.4f}")
    # Decode the bypass encoder output
    bypass_tokens = engine.run_decode(engine.ENC_OUT_DRAM, L_use)

    # Test 2: Feed ref encoder output to HW decoder (bypass everything)
    print(f"\n  TEST 2: REF encoder → HW decoder")
    ref_enc_padded = torch.zeros(L_pad, engine.d_model, dtype=torch.bfloat16)
    ref_enc_padded[:L_use] = ref_enc_2d.to(torch.bfloat16)
    engine.dma_to_accelerator_memory(engine.ENC_OUT_DRAM, ref_enc_padded.contiguous())
    ref2hw_tokens = engine.run_decode(engine.ENC_OUT_DRAM, L_use)

    # --- CPU reference decode using same encoder output for comparison ---
    print(f"\n--- CPU Reference Decode ---")
    bpe = engine.bytes_per_element
    D = engine.d_model

    # Read HW encoder output (from the normal run, not bypass)
    hw_enc = torch.zeros(L_pad * D, dtype=torch.bfloat16)
    engine.dma_read(DMA_DEVICE_C2H, enc_out_addr, hw_enc, L_pad * D * bpe)
    hw_enc_2d = hw_enc.reshape(L_pad, D)[:L]

    # ref_enc already computed in bypass tests; reuse ref_mel too
    with torch.no_grad():
        pass  # ref_mel and ref_enc already computed
        pass  # ref_enc already computed
    ref_enc_2d = ref_enc.squeeze(0)[:L_ref]

    print(f"  HW  encoder: shape=({L}, {D}) norm={hw_enc_2d.float().norm():.4f} "
          f"min={hw_enc_2d.float().min():.4f} max={hw_enc_2d.float().max():.4f}")
    print(f"  REF encoder: shape=({L_ref}, {D}) norm={ref_enc_2d.float().norm():.4f} "
          f"min={ref_enc_2d.float().min():.4f} max={ref_enc_2d.float().max():.4f}")
    L_cmp = min(L, L_ref)
    cos = F.cosine_similarity(
        hw_enc_2d[:L_cmp].float().flatten().unsqueeze(0),
        ref_enc_2d[:L_cmp].float().flatten().unsqueeze(0)).item()
    print(f"  Encoder cos similarity: {cos:.6f}")

    # CPU decode on HW encoder output (model is bf16, so keep bf16)
    print(f"\n  CPU decode on HW encoder output:")
    with torch.no_grad():
        hw_tokens = ref_model.decode(hw_enc_2d.unsqueeze(0).to(ref_enc.dtype))
    print(f"  HW-enc tokens: {hw_tokens[0][:20]}{'...' if len(hw_tokens[0]) > 20 else ''}")

    # CPU decode on CPU encoder output
    print(f"\n  CPU decode on REF encoder output:")
    with torch.no_grad():
        ref_tokens = ref_model.decode(ref_enc)
    print(f"  REF-enc tokens: {ref_tokens[0][:20]}{'...' if len(ref_tokens[0]) > 20 else ''}")

    # Decode all to text
    tokenizer_path = TOKENIZER_PATH
    if os.path.exists(tokenizer_path):
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(tokenizer_path)
        vocab_sz = sp.GetPieceSize()

        hw_valid = [t for t in tokens if 0 <= t < vocab_sz]
        hw_text = sp.DecodeIds(hw_valid)
        print(f"\n  HW  decode (HW enc):       >>> {hw_text}")

        hw_cpu_valid = [t for t in hw_tokens[0] if 0 <= t < vocab_sz]
        hw_cpu_text = sp.DecodeIds(hw_cpu_valid)
        print(f"  CPU decode (HW enc):       >>> {hw_cpu_text}")

        bypass_valid = [t for t in bypass_tokens if 0 <= t < vocab_sz]
        bypass_text = sp.DecodeIds(bypass_valid)
        print(f"  HW dec (REF sub→HW enc):   >>> {bypass_text}")

        ref2hw_valid = [t for t in ref2hw_tokens if 0 <= t < vocab_sz]
        ref2hw_text = sp.DecodeIds(ref2hw_valid)
        print(f"  HW dec (REF enc→HW dec):   >>> {ref2hw_text}")

        ref_valid = [t for t in ref_tokens[0] if 0 <= t < vocab_sz]
        ref_text = sp.DecodeIds(ref_valid)
        print(f"  CPU decode (REF enc):      >>> {ref_text}")
    else:
        print(f"  HW tokens:  {tokens[:20]}")
        print(f"  CPU-on-HW:  {hw_tokens[0][:20]}")
        print(f"  REF tokens: {ref_tokens[0][:20]}")

    print()


if __name__ == "__main__":
    main()
