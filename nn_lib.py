"""nn_lib.py

Shared helpers extracted from per-model bring-up files. Every public op uses the
``_core_dram`` naming convention for DRAM-in/DRAM-out wrappers around the SRAM
primitives in ``user_dma_core.UnifiedEngine``.

Staging ground for ops that will eventually graduate into ``user_dma_core.py``.
Functions here are intentionally thin — state lives on the ``ue`` (UnifiedEngine)
instance. This file never mutates ``user_dma_core``; callers still construct a
``UnifiedEngine`` subclass and pass it in.

Grouping:
  PART A — Variable seq_len ops (PBI candidates; will graduate to user_dma_core)
    §2  DRAM elementwise wrappers (eltwise_add/mul_core_dram)
    §2b Unified ND permute (smart_bf16_permute_core)
    §3  Fused post-add norm variants (rms/layer_norm + residual in one pass)
    §7  Activation / fused ops (tanh, silu, glu, half_step_residual, rel_shift,
        chunked_transpose, batch_norm)
    §8  RoPE (rope_hf_core_dram — prefill padded-split variant)
    §9  Flash attention (decode + prefill, smolvlm2-style)
  PART B — Fixed-shape host helpers (init-time, not PBI-relevant)
    §1  Weight staging / DRAM loaders (store_weight, load_weight_cache, etc.)
    §4  (moved to quant_lib.py — legacy concatenated-format codecs)
    §5  Weight-bin plumbing (manifest writer, name mapping hook)
    §6  Simple-LM boilerplate (config loading, HF download, quiet print)
"""
from __future__ import annotations

import builtins
import json
import math
import os
import struct
from typing import Callable

import numpy as np
import torch

from user_dma_core import (
    DMA_DEVICE_C2H,
    DMA_DEVICE_H2C,
    MEMCPY_TYPE,
    UE_VECTOR_SIZE,
    URAM_FULL_ELEMENTS,
    URAM_NEAR_FULL_ELEMENTS,
    ue_35bit_addr_shifter,
)

# Standard URAM section bases used across all _core_dram helpers.
URAM_A_BASE = 0x00000
URAM_B_BASE = 0x80000

# =============================================================================
# PART A — Variable seq_len ops (PBI candidates; eventual user_dma_core promotion)
# These ops have a runtime-variable dimension (seq_len, kv_len, M, or L) and are
# the targets for PBI-bucketed compilation. As they stabilize they should move
# down into user_dma_core.UnifiedEngine.
# =============================================================================
# =============================================================================
# §2 DRAM elementwise wrappers
# =============================================================================
def eltwise_add_core_dram(ue, size: int, A_DRAM_ADDR: int, B_DRAM_ADDR: int,
                          OUTPUT_DRAM_ADDR: int) -> int:
    """OUTPUT = A + B over ``size`` bf16 elements, streamed through URAM A/B."""
    bytes_per_element = 2
    uram_a_addr = 0x00000
    uram_b_addr = 0x80000
    chunk_size = min(URAM_NEAR_FULL_ELEMENTS, size)

    for start, take in ue.chunk_ranges(size, chunk_size):
        offset_bytes = start * bytes_per_element
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=A_DRAM_ADDR + offset_bytes,
            sram_address=uram_a_addr, element_size=take)
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=B_DRAM_ADDR + offset_bytes,
            sram_address=uram_b_addr, element_size=take)
        ue.eltwise_add_core(
            vector_A_sram_start_addr=uram_a_addr,
            vector_B_sram_start_addr=uram_b_addr,
            vector_C_sram_wb_addr=uram_a_addr, element_size=take)
        ue.sram_to_accelerator_memory(
            sram_address=uram_a_addr,
            accelerator_dram_address=OUTPUT_DRAM_ADDR + offset_bytes,
            element_size=take)
    return size

def eltwise_mul_core_dram(ue, size: int, A_DRAM_ADDR: int, B_DRAM_ADDR: int,
                          OUTPUT_DRAM_ADDR: int) -> int:
    """OUTPUT = A * B over ``size`` bf16 elements, streamed through URAM A/B."""
    bytes_per_element = 2
    uram_a_addr = 0x00000
    uram_b_addr = 0x80000
    chunk_size = min(URAM_NEAR_FULL_ELEMENTS, size)

    for start, take in ue.chunk_ranges(size, chunk_size):
        offset_bytes = start * bytes_per_element
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=A_DRAM_ADDR + offset_bytes,
            sram_address=uram_a_addr, element_size=take)
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=B_DRAM_ADDR + offset_bytes,
            sram_address=uram_b_addr, element_size=take)
        ue.eltwise_mul_core(
            vector_A_sram_start_addr=uram_a_addr,
            vector_B_sram_start_addr=uram_b_addr,
            vector_C_sram_wb_addr=uram_a_addr, element_size=take)
        ue.sram_to_accelerator_memory(
            sram_address=uram_a_addr,
            accelerator_dram_address=OUTPUT_DRAM_ADDR + offset_bytes,
            element_size=take)
    return size


# =============================================================================
# §2b Unified ND permute (DMA gather + batched transpose)
# =============================================================================
def smart_bf16_permute_core(ue, dims, permute_indices, input_dram_addr, output_dram_addr,
                            params_dram_addr=0, temp_dram_start=0):
    """ND permute via DMA gather + batched identity-dot-product transpose.

    Auto-detects an identity prefix in ``permute_indices``: leading dims that
    map to themselves are treated as outer-loop batch and strided over, so the
    inner decomposition only sees the dims that actually move.

    Subsumes the 3-D ``bf16_permute_core`` on ``UnifiedEngine`` (lands in Case 1
    fast gather) and handles non-``UE_VECTOR_SIZE``-aligned last dims.
    """
    from user_dma_core import (UE_VECTOR_SIZE, UE_MODE, URAM_NEAR_FULL_ELEMENTS,
                               URAM_HALF_ELEMENTS, URAM_SECTION, URAM_WRITE_SRC,
                               URAM_START_ADDR, LALU_MODE)
    bpe = 2

    # Peel identity prefix → outer batch loop.
    batch_prefix = 0
    while batch_prefix < len(permute_indices) and permute_indices[batch_prefix] == batch_prefix:
        batch_prefix += 1
    # Last dim must be part of the permuted suffix for the decomposition to make sense.
    if batch_prefix >= len(permute_indices) - 1:
        batch_prefix = 0

    if batch_prefix > 0:
        inner_dims = list(dims[batch_prefix:])
        inner_perm = [p - batch_prefix for p in permute_indices[batch_prefix:]]
        outer = 1
        for d in dims[:batch_prefix]:
            outer *= d
        inner_elems = 1
        for d in inner_dims:
            inner_elems *= d
        stride = inner_elems * bpe
        last_shape = None
        for b in range(outer):
            _, last_shape = smart_bf16_permute_core(
                ue, inner_dims, inner_perm,
                input_dram_addr + b * stride,
                output_dram_addr + b * stride,
                params_dram_addr, temp_dram_start,
            )
        return (1, tuple(dims[:batch_prefix]) + last_shape)

    n = len(dims) - 1
    total_elements = 1
    for d in dims:
        total_elements *= d
    k = permute_indices[n]
    # Case 1: last dim stays fixed — pure DMA gather.
    if k == n:
        last_dim = dims[n]
        output_shape = tuple(dims[permute_indices[i]] for i in range(len(dims)))
        permute_a = torch.arange(total_elements, dtype=torch.int32).reshape(*dims)
        permute_a = permute_a.permute(*permute_indices).contiguous().flatten()

        if last_dim < UE_VECTOR_SIZE or last_dim % UE_VECTOR_SIZE != 0:
            for j in range(total_elements // last_dim):
                src_idx = permute_a[j * last_dim].item()
                ue.ue_memcpy_from_dram(input_dram_addr + src_idx * bpe, last_dim * bpe, 0,
                    URAM_START_ADDR, URAM_SECTION.URAM_A.value)
                ue.wait_queue()
                ue.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, URAM_START_ADDR,
                    output_dram_addr + j * last_dim * bpe, last_dim * bpe)
                ue.wait_queue()
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
                    URAM_SECTION.URAM_A.value)
                ue.wait_queue()
            ue.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, URAM_START_ADDR,
                out_addr, n_blocks * last_dim * bpe)
            ue.wait_queue()
            remaining -= cur; out_addr += n_blocks * last_dim * bpe; i += cur
        return (1, output_shape)

    # Case 2: last dim changes — Q1 (last-dim-fixed) → transpose → Q3 (last-dim-fixed).
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
                    URAM_SECTION.URAM_A.value)
                ue.wait_queue()
            ue.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, URAM_START_ADDR,
                out_addr, n_blocks * last_dim * bpe)
            ue.wait_queue()
            remaining -= cur; out_addr += n_blocks * last_dim * bpe; i += cur

    input_uram_addr = URAM_START_ADDR
    ue.ue_memcpy_from_dram(params_dram_addr, UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe,
        0, input_uram_addr, URAM_SECTION.URAM_A.value)
    ue.wait_queue()

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
                    0, URAM_START_ADDR, URAM_SECTION.URAM_B.value)
                ue.wait_queue()

                for i in range(cur_M):
                    abs_row = start_vec + i
                    vec_idx = abs_row % UE_VECTOR_SIZE
                    col_block = abs_row // UE_VECTOR_SIZE
                    ue.start_queue_for_bf16_matvec_operation(
                        0, 0,
                        (input_uram_addr + vec_idx) * UE_VECTOR_SIZE * 2,
                        0x80000 + col_block * UE_VECTOR_SIZE * 2,
                        output_uram * UE_VECTOR_SIZE * 2,
                        N_transpose, cur_N)
                    ue.wait_queue()
                    ue.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, output_uram,
                        out_offset + i * M_aligned * bpe, cur_N * bpe)
                    ue.wait_queue()

                remaining_N -= cur_N; out_offset += cur_N * bpe; weight_addr += cur_N * N_transpose * bpe
            out_chunk += cur_M * M_aligned * bpe; remaining_M -= cur_M; start_vec += cur_M

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
                    URAM_SECTION.URAM_A.value)
                ue.wait_queue()
            ue.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, URAM_START_ADDR,
                out_addr, n_blocks * last_dim * bpe)
            ue.wait_queue()
            remaining -= cur; out_addr += n_blocks * last_dim * bpe; i += cur

    return (2, output_shape)


# =============================================================================
# §3 Fused post-add norm variants
# =============================================================================
def rms_norm_core_dram_post_add(ue, M: int, N: int,
                                A_DRAM_ADDR: int, B_DRAM_ADDR: int,
                                ADDOUTPUT_DRAM_ADDR: int, NORMOUTPUT_DRAM_ADDR: int,
                                GAMMA_DRAM_ADDR: int | None = None) -> int:
    """rms_norm(A + B). Writes both the pre-norm sum (residual) and the normed output."""
    gamma_sram_addr = 0x80000
    params_sram_addr = gamma_sram_addr

    if GAMMA_DRAM_ADDR is not None:
        ue.accelerator_memory_to_sram(accelerator_dram_address=GAMMA_DRAM_ADDR,
                                      sram_address=gamma_sram_addr, element_size=N)
        params_sram_addr += N * 2
    else:
        gamma_sram_addr = None

    vector_A_sram_addr = 0x00000
    vector_B_sram_addr = params_sram_addr
    uram_b_remaining_elements = URAM_NEAR_FULL_ELEMENTS - (params_sram_addr - 0x80000) // 2
    chunk_size = min(URAM_NEAR_FULL_ELEMENTS // N, uram_b_remaining_elements // N, M)
    assert chunk_size >= 1 and chunk_size <= M

    for i, m_take in ue.chunk_ranges(M, chunk_size):
        chunk_elements = m_take * N
        ue.accelerator_memory_to_sram(accelerator_dram_address=A_DRAM_ADDR + i * N * 2,
                                      sram_address=vector_A_sram_addr, element_size=chunk_elements)
        ue.accelerator_memory_to_sram(accelerator_dram_address=B_DRAM_ADDR + i * N * 2,
                                      sram_address=vector_B_sram_addr, element_size=chunk_elements)
        for j in range(m_take):
            ue.eltwise_add_core(vector_A_sram_addr + j * N * 2, vector_B_sram_addr + j * N * 2,
                                vector_A_sram_addr + j * N * 2, N)
        ue.sram_to_accelerator_memory(sram_address=vector_A_sram_addr,
                                      accelerator_dram_address=ADDOUTPUT_DRAM_ADDR + i * N * 2,
                                      element_size=chunk_elements)
        for j in range(m_take):
            ue.rms_norm_core(vector_A_sram_addr + j * N * 2, vector_A_sram_addr + j * N * 2,
                             N, gamma_sram_addr)
        ue.sram_to_accelerator_memory(sram_address=vector_A_sram_addr,
                                      accelerator_dram_address=NORMOUTPUT_DRAM_ADDR + i * N * 2,
                                      element_size=chunk_elements)

    total_flops = M * N + 3 * M * N
    if gamma_sram_addr is not None:
        total_flops += M * N
    return total_flops


def layer_norm_core_dram_post_add(ue, M: int, N: int,
                                  A_DRAM_ADDR: int, B_DRAM_ADDR: int,
                                  ADDOUTPUT_DRAM_ADDR: int, NORMOUTPUT_DRAM_ADDR: int,
                                  GAMMA_DRAM_ADDR: int | None = None,
                                  BETA_DRAM_ADDR: int | None = None) -> int:
    """layer_norm(A + B). Writes both the pre-norm sum and the normed output."""
    zeros_sram_addr = 0x80000

    zeros_dram_addr = ue.get_params_dram_addr()
    ue.allocate_params_dram(N * 2)
    ue.dma_write(DMA_DEVICE_H2C, zeros_dram_addr, torch.zeros(N, dtype=torch.bfloat16), N * 2)

    ue.accelerator_memory_to_sram(accelerator_dram_address=zeros_dram_addr,
                                  sram_address=zeros_sram_addr, element_size=N)
    params_sram_addr = zeros_sram_addr + N * 2

    gamma_sram_addr = None
    beta_sram_addr = None

    if GAMMA_DRAM_ADDR is not None:
        gamma_sram_addr = params_sram_addr
        ue.accelerator_memory_to_sram(accelerator_dram_address=GAMMA_DRAM_ADDR,
                                      sram_address=gamma_sram_addr, element_size=N)
        params_sram_addr += N * 2

    if BETA_DRAM_ADDR is not None:
        beta_sram_addr = params_sram_addr
        ue.accelerator_memory_to_sram(accelerator_dram_address=BETA_DRAM_ADDR,
                                      sram_address=beta_sram_addr, element_size=N)
        params_sram_addr += N * 2

    vector_A_sram_addr = 0x00000
    vector_B_sram_addr = params_sram_addr
    uram_b_remaining_elements = URAM_NEAR_FULL_ELEMENTS - (params_sram_addr - 0x80000) // 2
    chunk_size = min(URAM_NEAR_FULL_ELEMENTS // N, uram_b_remaining_elements // N, M)
    assert chunk_size >= 1 and chunk_size <= M

    for i, m_take in ue.chunk_ranges(M, chunk_size):
        chunk_elements = m_take * N
        ue.accelerator_memory_to_sram(accelerator_dram_address=A_DRAM_ADDR + i * N * 2,
                                      sram_address=vector_A_sram_addr, element_size=chunk_elements)
        ue.accelerator_memory_to_sram(accelerator_dram_address=B_DRAM_ADDR + i * N * 2,
                                      sram_address=vector_B_sram_addr, element_size=chunk_elements)
        for j in range(m_take):
            ue.eltwise_add_core(vector_A_sram_addr + j * N * 2, vector_B_sram_addr + j * N * 2,
                                vector_A_sram_addr + j * N * 2, N)
        ue.sram_to_accelerator_memory(sram_address=vector_A_sram_addr,
                                      accelerator_dram_address=ADDOUTPUT_DRAM_ADDR + i * N * 2,
                                      element_size=chunk_elements)
        for j in range(m_take):
            ue.layer_norm_core(vector_A_sram_addr + j * N * 2, vector_A_sram_addr + j * N * 2,
                               N, zeros_sram_addr, gamma_sram_addr, beta_sram_addr)
        ue.sram_to_accelerator_memory(sram_address=vector_A_sram_addr,
                                      accelerator_dram_address=NORMOUTPUT_DRAM_ADDR + i * N * 2,
                                      element_size=chunk_elements)

    total_flops = M * N + 5 * M * N
    if gamma_sram_addr is not None:
        total_flops += M * N
    if beta_sram_addr is not None:
        total_flops += M * N
    return total_flops


# =============================================================================
# §7 Activation / fused ops (parakeet-derived)
# =============================================================================
def batch_norm_core_dram(ue, C: int, L: int,
                         A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                         SCALE_DRAM_ADDR: int, SHIFT_DRAM_ADDR: int,
                         tiled_scale_addr: int = None,
                         tiled_shift_addr: int = None) -> None:
    """Eval-mode batch norm on (C, L). If ``tiled_scale_addr`` / ``tiled_shift_addr``
    are provided, runs the 6-instruction bulk path; otherwise falls back to a
    per-channel broadcast loop."""
    assert L % UE_VECTOR_SIZE == 0, f"L={L} must be a multiple of {UE_VECTOR_SIZE}"
    if tiled_scale_addr is not None and tiled_shift_addr is not None:
        total_elems = C * L
        ue.accelerator_memory_to_sram(A_DRAM_ADDR, URAM_A_BASE, total_elems)
        ue.accelerator_memory_to_sram(tiled_scale_addr, URAM_B_BASE, total_elems)
        ue.eltwise_mul_core(vector_A_sram_start_addr=URAM_A_BASE,
                            vector_B_sram_start_addr=URAM_B_BASE,
                            vector_C_sram_wb_addr=URAM_A_BASE, element_size=total_elems)
        ue.accelerator_memory_to_sram(tiled_shift_addr, URAM_B_BASE, total_elems)
        ue.eltwise_add_core(vector_A_sram_start_addr=URAM_A_BASE,
                            vector_B_sram_start_addr=URAM_B_BASE,
                            vector_C_sram_wb_addr=URAM_A_BASE, element_size=total_elems)
        ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR, total_elems)
        return

    row_bytes = L * 2
    scale_host = torch.zeros(C, dtype=torch.bfloat16)
    shift_host = torch.zeros(C, dtype=torch.bfloat16)
    ue.dma_read(DMA_DEVICE_C2H, SCALE_DRAM_ADDR, scale_host, C * 2)
    ue.dma_read(DMA_DEVICE_C2H, SHIFT_DRAM_ADDR, shift_host, C * 2)
    for c in range(C):
        ue.accelerator_memory_to_sram(A_DRAM_ADDR + c * row_bytes, URAM_A_BASE, L)
        ue.broadcast_mul(scalar=scale_host[c].float().item(),
                         sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=L)
        ue.broadcast_add(scalar=shift_host[c].float().item(),
                         sram_start_addr=URAM_A_BASE, sram_wb_addr=URAM_A_BASE, element_size=L)
        ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR + c * row_bytes, L)


def tanh_core_dram(ue, M: int, N: int,
                   A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                   IDENTITY_DRAM_ADDR: int) -> None:
    """tanh(x) = 2*sigmoid(2x) - 1 over an (M, N) tensor. Uses an identity-matmul
    sigmoid trick (LALU sigmoid is only exposed via matmat_mul_core)."""
    assert N % UE_VECTOR_SIZE == 0, f"N={N} must be a multiple of {UE_VECTOR_SIZE}"
    row_bytes = N * 2
    for m in range(M):
        ue.accelerator_memory_to_sram(A_DRAM_ADDR + m * row_bytes, URAM_A_BASE, N)
        ue.broadcast_mul(scalar=2.0, sram_start_addr=URAM_A_BASE,
                         sram_wb_addr=URAM_A_BASE, element_size=N)
        ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR + m * row_bytes, N)
    ue.matmat_mul_core(M=M, K=N, N=N,
                       A_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                       B_DRAM_ADDR=IDENTITY_DRAM_ADDR,
                       OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                       sigmoid_enable=True)
    for m in range(M):
        ue.accelerator_memory_to_sram(OUTPUT_DRAM_ADDR + m * row_bytes, URAM_A_BASE, N)
        ue.broadcast_mul(scalar=2.0, sram_start_addr=URAM_A_BASE,
                         sram_wb_addr=URAM_A_BASE, element_size=N)
        ue.broadcast_add(scalar=-1.0, sram_start_addr=URAM_A_BASE,
                         sram_wb_addr=URAM_A_BASE, element_size=N)
        ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR + m * row_bytes, N)


def silu_core_dram(ue, M: int, N: int,
                   A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                   IDENTITY_DRAM_ADDR: int,
                   gpr_M_reg: int = None) -> None:
    """SiLU: output = x * sigmoid(x) on an (M, N) tensor. Identity-matmul sigmoid
    then elementwise multiply against the original input."""
    assert N % UE_VECTOR_SIZE == 0, f"N={N} must be a multiple of {UE_VECTOR_SIZE}"
    total_elems = M * N
    ue.accelerator_memory_to_sram(A_DRAM_ADDR, URAM_A_BASE, total_elems)
    ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR, total_elems)
    ue.matmat_mul_core(M=M, K=N, N=N, A_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                       B_DRAM_ADDR=IDENTITY_DRAM_ADDR,
                       OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR, sigmoid_enable=True,
                       gpr_M_reg=gpr_M_reg)
    ue.accelerator_memory_to_sram(A_DRAM_ADDR, URAM_A_BASE, total_elems)
    ue.accelerator_memory_to_sram(OUTPUT_DRAM_ADDR, URAM_B_BASE, total_elems)
    ue.eltwise_mul_core(vector_A_sram_start_addr=URAM_A_BASE,
                        vector_B_sram_start_addr=URAM_B_BASE,
                        vector_C_sram_wb_addr=URAM_A_BASE, element_size=total_elems)
    ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR, total_elems)


def glu_core_dram(ue, M: int, C: int,
                  A_DRAM_ADDR: int, B_DRAM_ADDR: int,
                  OUTPUT_DRAM_ADDR: int, IDENTITY_DRAM_ADDR: int) -> None:
    """GLU: output = a * sigmoid(b). Assumes a/b are produced by two separate
    matmuls (PW Conv1 split) so they're already in DRAM at the two addresses.
    Note: B_DRAM_ADDR is modified in place with sigmoid(b)."""
    assert C % UE_VECTOR_SIZE == 0, f"C={C} must be a multiple of {UE_VECTOR_SIZE}"
    total_elems = M * C
    ue.matmat_mul_core(M=M, K=C, N=C, A_DRAM_ADDR=B_DRAM_ADDR,
                       B_DRAM_ADDR=IDENTITY_DRAM_ADDR,
                       OUTPUT_DRAM_ADDR=B_DRAM_ADDR, sigmoid_enable=True)
    ue.accelerator_memory_to_sram(A_DRAM_ADDR, URAM_A_BASE, total_elems)
    ue.accelerator_memory_to_sram(B_DRAM_ADDR, URAM_B_BASE, total_elems)
    ue.eltwise_mul_core(vector_A_sram_start_addr=URAM_A_BASE,
                        vector_B_sram_start_addr=URAM_B_BASE,
                        vector_C_sram_wb_addr=URAM_A_BASE, element_size=total_elems)
    ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR, total_elems)


def half_step_residual_core_dram(ue, M: int, N: int,
                                 RESIDUAL_DRAM_ADDR: int, FF_DRAM_ADDR: int,
                                 OUTPUT_DRAM_ADDR: int) -> None:
    """output = residual + 0.5 * ff_output. FF_DRAM_ADDR is read but not modified."""
    assert N % UE_VECTOR_SIZE == 0, f"N={N} must be a multiple of {UE_VECTOR_SIZE}"
    total_elems = M * N
    ue.accelerator_memory_to_sram(FF_DRAM_ADDR, URAM_A_BASE, total_elems)
    ue.broadcast_mul(scalar=0.5, sram_start_addr=URAM_A_BASE,
                     sram_wb_addr=URAM_A_BASE, element_size=total_elems)
    ue.accelerator_memory_to_sram(RESIDUAL_DRAM_ADDR, URAM_B_BASE, total_elems)
    ue.eltwise_add_core(vector_A_sram_start_addr=URAM_A_BASE,
                        vector_B_sram_start_addr=URAM_B_BASE,
                        vector_C_sram_wb_addr=URAM_A_BASE,
                        element_size=total_elems)
    ue.sram_to_accelerator_memory(URAM_A_BASE, OUTPUT_DRAM_ADDR, total_elems)


def rel_shift_core_dram(ue, L: int, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                        input_row_stride: int = None) -> None:
    """rel_shift: extract (L, L) from a (L, input_row_stride) positional-score matrix.
    Row i of output = input[i, (L-1-i) : (L-1-i)+L]. Pure memory rearrangement
    (no arithmetic). ``input_row_stride`` defaults to ``2*L - 1``."""
    assert L % UE_VECTOR_SIZE == 0, f"L={L} must be a multiple of {UE_VECTOR_SIZE}"
    P_stride = input_row_stride if input_row_stride is not None else (2 * L - 1)
    bpe = 2
    for i in range(L):
        src = INPUT_DRAM_ADDR + (i * P_stride + (L - 1 - i)) * bpe
        dst = OUTPUT_DRAM_ADDR + i * L * bpe
        ue.accelerator_memory_to_sram(src, URAM_A_BASE, L)
        ue.sram_to_accelerator_memory(URAM_A_BASE, dst, L)


def chunked_transpose_core_dram(ue, M: int, N: int,
                                input_dram_addr: int, output_dram_addr: int,
                                identity_dram_addr: int, temp_dram_addr: int) -> None:
    """Transpose (M, N) -> (N, M_aligned) by processing N in UE_VECTOR_SIZE-column
    chunks. Works around a bug in ``smart_bf16_permute_core`` where
    ``N_transpose > UE_VECTOR_SIZE`` accumulates across groups and corrupts output.
    ``M_aligned = pad_to_multiple(M, UE_VECTOR_SIZE)``."""
    bpe = 2
    VS = UE_VECTOR_SIZE
    M_aligned = ((M - 1) // VS + 1) * VS
    n_chunks = (N + VS - 1) // VS

    for c in range(n_chunks):
        col_start = c * VS
        col_end = min(col_start + VS, N)
        chunk_cols = col_end - col_start
        chunk_cols_pad = ((chunk_cols - 1) // VS + 1) * VS

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

        smart_bf16_permute_core(
            ue,
            dims=[M, chunk_cols_pad], permute_indices=[1, 0],
            input_dram_addr=temp_dram_addr,
            output_dram_addr=output_dram_addr + col_start * M_aligned * bpe,
            params_dram_addr=identity_dram_addr,
            temp_dram_start=temp_dram_addr + M * chunk_cols_pad * bpe)


# =============================================================================
# §8 RoPE (prefill padded-split variant; smolvlm2-derived)
# =============================================================================
def rope_hf_core_dram(ue, M: int, D: int,
                      X_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                      COS_DRAM_ADDR: int, SIN_DRAM_ADDR: int,
                      COS_LO_PAD_ADDR: int = 0, COS_HI_PAD_ADDR: int = 0,
                      SIN_LO_PAD_ADDR: int = 0, SIN_HI_PAD_ADDR: int = 0) -> int:
    """Prefill RoPE via padded-split bulk load (D < 128). The sin first half must
    be pre-negated on the host. Returns FLOP count."""
    bytes_per_element = 2
    row_bytes = D * bytes_per_element
    half_D = D // 2
    half_bytes = half_D * bytes_per_element

    SRAM_SLOT = 128
    uram_x_lo      = 0x00000
    uram_x_hi      = uram_x_lo      + SRAM_SLOT
    uram_a_lo      = uram_x_hi      + SRAM_SLOT
    uram_a_hi      = uram_a_lo      + SRAM_SLOT
    uram_result_lo = uram_a_hi      + SRAM_SLOT
    uram_result_hi = uram_result_lo + SRAM_SLOT

    uram_b_lo        = URAM_B_BASE
    uram_b_hi        = uram_b_lo         + SRAM_SLOT
    uram_cos_lo_bulk = uram_b_hi         + SRAM_SLOT
    uram_cos_hi_bulk = uram_cos_lo_bulk  + M * SRAM_SLOT
    uram_sin_lo_bulk = uram_cos_hi_bulk  + M * SRAM_SLOT
    uram_sin_hi_bulk = uram_sin_lo_bulk  + M * SRAM_SLOT

    ue.accelerator_memory_to_sram(accelerator_dram_address=COS_LO_PAD_ADDR,
                                  sram_address=uram_cos_lo_bulk, element_size=M * D)
    ue.accelerator_memory_to_sram(accelerator_dram_address=COS_HI_PAD_ADDR,
                                  sram_address=uram_cos_hi_bulk, element_size=M * D)
    ue.accelerator_memory_to_sram(accelerator_dram_address=SIN_LO_PAD_ADDR,
                                  sram_address=uram_sin_lo_bulk, element_size=M * D)
    ue.accelerator_memory_to_sram(accelerator_dram_address=SIN_HI_PAD_ADDR,
                                  sram_address=uram_sin_hi_bulk, element_size=M * D)

    for row in range(M):
        row_offset = row * row_bytes
        ue.accelerator_memory_to_sram(accelerator_dram_address=X_DRAM_ADDR + row_offset,
                                      sram_address=uram_x_lo, element_size=half_D)
        ue.accelerator_memory_to_sram(accelerator_dram_address=X_DRAM_ADDR + row_offset + half_bytes,
                                      sram_address=uram_x_hi, element_size=half_D)

        uram_cos_lo_i = uram_cos_lo_bulk + row * SRAM_SLOT
        uram_cos_hi_i = uram_cos_hi_bulk + row * SRAM_SLOT
        uram_sin_lo_i = uram_sin_lo_bulk + row * SRAM_SLOT
        uram_sin_hi_i = uram_sin_hi_bulk + row * SRAM_SLOT

        ue.eltwise_mul_core(uram_x_lo, uram_cos_lo_i, uram_a_lo, half_D)
        ue.eltwise_mul_core(uram_x_hi, uram_cos_hi_i, uram_a_hi, half_D)
        ue.eltwise_mul_core(uram_x_hi, uram_sin_lo_i, uram_b_lo, half_D)
        ue.eltwise_mul_core(uram_x_lo, uram_sin_hi_i, uram_b_hi, half_D)
        ue.eltwise_add_core(uram_a_lo, uram_b_lo, uram_result_lo, half_D)
        ue.eltwise_add_core(uram_a_hi, uram_b_hi, uram_result_hi, half_D)

        ue.sram_to_accelerator_memory(sram_address=uram_result_lo,
                                      accelerator_dram_address=OUTPUT_DRAM_ADDR + row_offset,
                                      element_size=half_D)
        ue.sram_to_accelerator_memory(sram_address=uram_result_hi,
                                      accelerator_dram_address=OUTPUT_DRAM_ADDR + row_offset + half_bytes,
                                      element_size=half_D)

    return M * D * 4


def rope_hf_core_dram_pbi(ue, M: int, D: int,
                          X_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, ROPE_PACKED_ADDR: int,
                          gpr_M_reg: int, tmp_reg: int, t_reg: int) -> int:
    """PBI (hardware-loop) prefill RoPE for head_dim D < 128 (padded-split scheme).

    Token count is a *runtime* trip count from ``gpr_M_reg`` (caller primes it via
    ADD_SET); ``M`` is FLOPs-accounting / loop-template only, so the captured program
    is seq_len-agnostic. Caller brackets with start_capture()/stop_capture().

    D=64 means each half is 32 elems = 64 bytes = half a 128-byte SRAM row, and SRAM
    cannot be addressed mid-row. We sidestep that entirely: every DMA uses a
    *register-computed* DRAM address (``general_reg_src``) — DRAM is byte-addressable —
    so the two 32-elem halves land in their own 128-byte-aligned SRAM rows. No PBI
    pointers are used (reads and writes are both register-addressed), so this stays
    clear of the >=4-advancing-pointer failure mode.

    ``ROPE_PACKED_ADDR`` points at a per-token packed table laid out as
    ``[cos_lo | cos_hi | sin_lo | sin_hi]``, each 32-elem half zero-padded to
    ``UE_VECTOR_SIZE`` (64) so it occupies one full 128-byte SRAM row. ``sin_lo`` must
    be pre-negated on the host (matches :func:`rope_hf_core_dram`). One DMA loads all
    four rows contiguously; element_size=32 slices read each row from its aligned start.

    Registers: ``gpr_M_reg`` (outer trip count), ``tmp_reg`` (address scratch),
    ``t_reg`` (per-token counter, set to 0 here and incremented each iteration).
    """
    assert D % 2 == 0, "D must be even for RoPE half layout"
    assert D < 128, "rope_hf_core_dram_pbi is the D<128 padded-split path; use the shared rope for D>=128"
    bpe = 2
    half_D = D // 2                       # 32
    half_bytes = half_D * bpe             # 64
    row_bytes = D * bpe                   # 128 (x / output row)
    PAD = UE_VECTOR_SIZE                  # 64 — pad each half to a full URAM row
    pad_row_bytes = PAD * bpe             # 128
    rope_stride = 4 * pad_row_bytes       # 512 — packed cos_lo|cos_hi|sin_lo|sin_hi

    # eltwise requires its two operands in DIFFERENT URAM banks (A: 0x00000+, B: 0x80000+).
    # URAM_A holds x + cos-products + results; URAM_B holds the packed cos/sin + sin-products.
    SRAM_SLOT = 128
    uram_x_lo      = 0x00000              # URAM_A
    uram_x_hi      = 0x00080
    uram_a_lo      = 0x00100
    uram_a_hi      = 0x00180
    uram_result_lo = 0x00200
    uram_result_hi = 0x00280

    uram_cos_lo    = 0x80000              # URAM_B — packed read lands the 4 rows here ...
    uram_cos_hi    = uram_cos_lo + SRAM_SLOT
    uram_sin_lo    = uram_cos_hi + SRAM_SLOT
    uram_sin_hi    = uram_sin_lo + SRAM_SLOT
    uram_b_lo      = uram_sin_hi + SRAM_SLOT
    uram_b_hi      = uram_b_lo   + SRAM_SLOT

    shift = ue_35bit_addr_shifter

    def _set_addr(base, stride):
        # tmp_reg = shift(base) + t_reg * shift(stride)  ==  shift(base + t*stride)
        # (linear because base & stride are 8-byte aligned; shift is >>3)
        ue.generate_instruction_reg_mul_imm(tmp_reg, t_reg, shift(stride))
        ue.generate_instruction_add_imm(tmp_reg, shift(base), tmp_reg)

    ue.generate_instruction_add_set(t_reg, 0)
    ue.loop_start(loop_cnt=M, gpr_loop_cnt=gpr_M_reg)

    # --- register-addressed reads (each half into its own aligned row) ---
    _set_addr(X_DRAM_ADDR, row_bytes)
    ue.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=uram_x_lo,
                                  element_size=half_D, general_reg_src=tmp_reg)
    _set_addr(X_DRAM_ADDR + half_bytes, row_bytes)
    ue.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=uram_x_hi,
                                  element_size=half_D, general_reg_src=tmp_reg)
    _set_addr(ROPE_PACKED_ADDR, rope_stride)
    ue.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=uram_cos_lo,
                                  element_size=4 * PAD, general_reg_src=tmp_reg)

    # --- compute (all operands start on 128-byte SRAM rows, element_size=32) ---
    ue.eltwise_mul_core(uram_x_lo, uram_cos_lo, uram_a_lo, half_D)   # a_lo = x_lo * cos_lo
    ue.eltwise_mul_core(uram_x_hi, uram_cos_hi, uram_a_hi, half_D)   # a_hi = x_hi * cos_hi
    ue.eltwise_mul_core(uram_x_hi, uram_sin_lo, uram_b_lo, half_D)   # b_lo = x_hi * (-sin)
    ue.eltwise_mul_core(uram_x_lo, uram_sin_hi, uram_b_hi, half_D)   # b_hi = x_lo * sin
    ue.eltwise_add_core(uram_a_lo, uram_b_lo, uram_result_lo, half_D)
    ue.eltwise_add_core(uram_a_hi, uram_b_hi, uram_result_hi, half_D)

    # --- register-addressed writes ---
    _set_addr(OUTPUT_DRAM_ADDR, row_bytes)
    ue.sram_to_accelerator_memory(sram_address=uram_result_lo, accelerator_dram_address=0,
                                  element_size=half_D, general_reg_src=tmp_reg)
    _set_addr(OUTPUT_DRAM_ADDR + half_bytes, row_bytes)
    ue.sram_to_accelerator_memory(sram_address=uram_result_hi, accelerator_dram_address=0,
                                  element_size=half_D, general_reg_src=tmp_reg)

    ue.generate_instruction_add_inc(t_reg)
    ue.loop_end()
    return M * D * 4


# =============================================================================
# §9 Flash attention (smolvlm2-derived; decode + prefill GQA-batched)
# =============================================================================
def decode_flash_attention_core(ue, head_dim: int, kv_len: int,
                                Q_DRAM_ADDR: int, K_DRAM_ADDR: int, V_DRAM_ADDR: int,
                                OUTPUT_DRAM_ADDR: int, SCRATCH_DRAM_ADDR: int,
                                IDENTITY_DRAM_ADDR: int = None,
                                BIAS_DRAM_ADDR: int = None,
                                debug_mode: bool = False, SM_OUTPUT_DRAM_ADDR: int = None,
                                num_q_heads: int = 1, bias_row_stride: int = None) -> int:
    """Decode attention: V^T transpose, scaled Q@K^T, softmax, sm@V^T. GQA via
    ``num_q_heads``. Returns FLOP count."""
    from user_dma_core import UE_FMAX_CONTEXT_SIZE
    bytes_per_element = 2
    bias_enable = BIAS_DRAM_ADDR is not None
    if debug_mode:
        assert SM_OUTPUT_DRAM_ADDR is not None

    SCRATCH_DRAM_PARTIAL_SM = SCRATCH_DRAM_ADDR + head_dim * kv_len * bytes_per_element
    M = head_dim
    K = head_dim
    N = kv_len

    identity_matrix_sram_start_addr = 0x00000
    identity_tensor = torch.eye(head_dim, dtype=torch.bfloat16)

    ue.accelerator_memory_to_sram(accelerator_dram_address=IDENTITY_DRAM_ADDR,
                                  sram_address=identity_matrix_sram_start_addr,
                                  element_size=UE_VECTOR_SIZE * UE_VECTOR_SIZE)
    usable_uram_a_start_addr = identity_matrix_sram_start_addr + UE_VECTOR_SIZE * UE_VECTOR_SIZE * bytes_per_element
    usable_uram_b_elements = URAM_NEAR_FULL_ELEMENTS
    N_chunk = min(N, (usable_uram_b_elements // K) // UE_VECTOR_SIZE * UE_VECTOR_SIZE)
    N_chunk_aligned = None
    if N_chunk < UE_VECTOR_SIZE:
        if (K * 32) <= usable_uram_b_elements:
            N_chunk = 32
        elif (K * 16) <= usable_uram_b_elements:
            N_chunk = 16
        else:
            assert False, f"K={K} too large for usable URAM={usable_uram_b_elements}"
        N_chunk_aligned = UE_VECTOR_SIZE
    usable_uram_a_elements = URAM_FULL_ELEMENTS - UE_VECTOR_SIZE * UE_VECTOR_SIZE
    output_N_size = N_chunk_aligned if N_chunk_aligned is not None else N_chunk
    M_chunk = min(M, usable_uram_a_elements // output_N_size)
    assert M_chunk >= 1 and M_chunk <= M

    output_sram_wb_addr = usable_uram_a_start_addr
    uram_b_start_addr = 0x80000
    for i, m_take in ue.chunk_ranges(M, M_chunk):
        for j, n_take in ue.chunk_ranges(N, N_chunk):
            ue.accelerator_memory_to_sram(accelerator_dram_address=V_DRAM_ADDR + j * K * bytes_per_element,
                                          sram_address=uram_b_start_addr,
                                          element_size=n_take * K)
            for output_row in range(m_take):
                if N_chunk_aligned is None:
                    out_sram_offset = output_row * n_take * bytes_per_element
                else:
                    out_sram_offset = output_row * N_chunk_aligned * bytes_per_element
                ones_idx = identity_tensor[output_row+i, :].reshape(-1, UE_VECTOR_SIZE).sum(axis=1).argmax(axis=0)
                vector_idx = identity_tensor[output_row+i, :].reshape(-1, UE_VECTOR_SIZE)[ones_idx, :].argmax(axis=0)
                ue.start_queue_for_bf16_matvec_operation(max_clear_en=0,
                                                        fmax_context_addr=0,
                                                        vector_sram_start_addr=0x00000 + vector_idx * UE_VECTOR_SIZE * bytes_per_element,
                                                        matrix_sram_start_addr=uram_b_start_addr + ones_idx * UE_VECTOR_SIZE * bytes_per_element,
                                                        output_sram_wb_addr=output_sram_wb_addr + out_sram_offset,
                                                        K=UE_VECTOR_SIZE,
                                                        N=n_take,
                                                        stride_z=m_take)
            start_dram_address_of_partial_matrix = SCRATCH_DRAM_ADDR + i * N * bytes_per_element + j * bytes_per_element

            if N_chunk_aligned is None:
                ue.sram_to_accelerator_memory(sram_address=output_sram_wb_addr,
                                              accelerator_dram_address=start_dram_address_of_partial_matrix,
                                              element_size=m_take * n_take,
                                              stride_bytes_per_chunk=n_take * bytes_per_element,
                                              stride_jump_bytes=N * bytes_per_element)
            else:
                for o_row_idx in range(m_take):
                    ue.sram_to_accelerator_memory(sram_address=output_sram_wb_addr + o_row_idx * N_chunk_aligned * bytes_per_element,
                                                  accelerator_dram_address=start_dram_address_of_partial_matrix + o_row_idx * N * bytes_per_element,
                                                  element_size=n_take)

    M = num_q_heads
    K = head_dim
    N = kv_len
    _bias_stride = bias_row_stride if bias_row_stride is not None else N
    usable_uram_b_elements = URAM_NEAR_FULL_ELEMENTS
    N_chunk = min(N, (usable_uram_b_elements // K) // UE_VECTOR_SIZE * UE_VECTOR_SIZE)
    N_chunk_aligned = None
    if N_chunk < UE_VECTOR_SIZE:
        if (K * 32) <= usable_uram_b_elements:
            N_chunk = 32
        elif (K * 16) <= usable_uram_b_elements:
            N_chunk = 16
        else:
            assert False, f"K={K} too large for usable URAM={usable_uram_b_elements}"
        N_chunk_aligned = UE_VECTOR_SIZE

    usable_uram_a_elements = URAM_FULL_ELEMENTS
    output_N_size = N_chunk_aligned if N_chunk_aligned is not None else N_chunk
    M_chunk = min(UE_FMAX_CONTEXT_SIZE, M, usable_uram_a_elements // (K + output_N_size))
    assert M_chunk >= 1 and M_chunk <= M

    uram_a_start_addr = 0x00000
    uram_b_start_addr = 0x80000
    for i, m_take in ue.chunk_ranges(M, M_chunk):
        ue.accelerator_memory_to_sram(accelerator_dram_address=Q_DRAM_ADDR + i * K * bytes_per_element,
                                      sram_address=uram_a_start_addr,
                                      element_size=m_take * K)

        ue.broadcast_mul(scalar=1 / math.sqrt(head_dim),
                         sram_start_addr=uram_a_start_addr,
                         sram_wb_addr=uram_a_start_addr,
                         element_size=m_take * K)

        output_sram_wb_addr = uram_a_start_addr + m_take * K * bytes_per_element
        assert output_sram_wb_addr < 0x80000

        clear_en = 1
        for j, n_take in ue.chunk_ranges(N, N_chunk):
            ue.accelerator_memory_to_sram(accelerator_dram_address=K_DRAM_ADDR + j * K * bytes_per_element,
                                          sram_address=uram_b_start_addr,
                                          element_size=n_take * K)

            assert m_take * K + n_take * m_take <= URAM_FULL_ELEMENTS

            for output_row in range(m_take):
                if bias_enable:
                    ue.accelerator_memory_to_bias_sram(
                        accelerator_dram_address=BIAS_DRAM_ADDR + ((i + output_row) * _bias_stride + j) * bytes_per_element,
                        element_size=n_take)

                if N_chunk_aligned is None:
                    out_sram_offset = output_row * n_take * bytes_per_element
                else:
                    out_sram_offset = output_row * N_chunk_aligned * bytes_per_element

                ue.start_queue_for_bf16_matvec_operation(max_clear_en=clear_en,
                                                        fmax_context_addr=output_row,
                                                        vector_sram_start_addr=uram_a_start_addr + output_row * K * bytes_per_element,
                                                        matrix_sram_start_addr=uram_b_start_addr,
                                                        output_sram_wb_addr=output_sram_wb_addr + out_sram_offset,
                                                        K=K,
                                                        N=n_take,
                                                        bias_enable=bias_enable)
                clear_en = 0

            start_dram_address_of_partial_matrix = SCRATCH_DRAM_PARTIAL_SM + j * bytes_per_element

            if N_chunk_aligned is None:
                ue.sram_to_accelerator_memory(sram_address=output_sram_wb_addr,
                                              accelerator_dram_address=start_dram_address_of_partial_matrix,
                                              element_size=m_take * n_take,
                                              stride_bytes_per_chunk=n_take * bytes_per_element,
                                              stride_jump_bytes=N * bytes_per_element)
            else:
                for o_row_idx in range(m_take):
                    ue.sram_to_accelerator_memory(sram_address=output_sram_wb_addr + o_row_idx * N_chunk_aligned * bytes_per_element,
                                                  accelerator_dram_address=start_dram_address_of_partial_matrix + o_row_idx * N * bytes_per_element,
                                                  element_size=n_take)

        max_m_take = min((URAM_FULL_ELEMENTS - UE_VECTOR_SIZE) // N, UE_FMAX_CONTEXT_SIZE)

        for m_take_chunk_idx, m_take_chunk_size in ue.chunk_ranges(m_take, max_m_take):
            ue.accelerator_memory_to_sram(accelerator_dram_address=SCRATCH_DRAM_PARTIAL_SM + m_take_chunk_idx * N * bytes_per_element,
                                          sram_address=uram_a_start_addr,
                                          element_size=m_take_chunk_size * N)

            for row_idx in range(m_take_chunk_size):
                ue.start_queue_for_bf16_softmax_operation(fmax_context_addr=row_idx + m_take_chunk_idx,
                                                         vector_sram_start_addr=uram_a_start_addr + row_idx * N * bytes_per_element,
                                                         output_sram_wb_addr=uram_a_start_addr + row_idx * N * bytes_per_element,
                                                         N=N)

            if debug_mode:
                ue.sram_to_accelerator_memory(sram_address=uram_a_start_addr,
                                              accelerator_dram_address=SM_OUTPUT_DRAM_ADDR + (i + m_take_chunk_idx) * N * bytes_per_element,
                                              element_size=m_take_chunk_size * N)

            v_tr_row_chunk_size = min((URAM_NEAR_FULL_ELEMENTS // kv_len // UE_VECTOR_SIZE) * UE_VECTOR_SIZE,
                                      ((URAM_FULL_ELEMENTS - m_take_chunk_size * kv_len) // m_take_chunk_size // UE_VECTOR_SIZE) * UE_VECTOR_SIZE,
                                      head_dim)

            v_tr_row_chunk_size_aligned = None
            if v_tr_row_chunk_size < UE_VECTOR_SIZE:
                v_tr_row_chunk_size_aligned = UE_VECTOR_SIZE
                if kv_len * 32 <= URAM_NEAR_FULL_ELEMENTS:
                    v_tr_row_chunk_size = 32
                elif kv_len * 16 <= URAM_NEAR_FULL_ELEMENTS:
                    v_tr_row_chunk_size = 16
                else:
                    assert False, "v_tr_row_chunk_size too large for URAM"

            v_t_sram_start_addr = 0x80000
            output_sram_wb_addr = uram_a_start_addr + m_take_chunk_size * kv_len * bytes_per_element

            for v_tr_column_idx, v_tr_column_take in ue.chunk_ranges(head_dim, v_tr_row_chunk_size):
                ue.accelerator_memory_to_sram(accelerator_dram_address=SCRATCH_DRAM_ADDR + v_tr_column_idx * kv_len * bytes_per_element,
                                              sram_address=v_t_sram_start_addr,
                                              element_size=v_tr_column_take * kv_len)

                for p_row_idx in range(m_take_chunk_size):
                    if v_tr_row_chunk_size_aligned is None:
                        output_sram_wb_offset = p_row_idx * v_tr_column_take * bytes_per_element
                    else:
                        output_sram_wb_offset = 0

                    ue.start_queue_for_bf16_matvec_operation(max_clear_en=0,
                                                            fmax_context_addr=0,
                                                            vector_sram_start_addr=uram_a_start_addr + p_row_idx * kv_len * bytes_per_element,
                                                            matrix_sram_start_addr=v_t_sram_start_addr,
                                                            output_sram_wb_addr=output_sram_wb_addr + output_sram_wb_offset,
                                                            K=kv_len,
                                                            N=v_tr_column_take)

                    if v_tr_row_chunk_size_aligned is not None:
                        ue.sram_to_accelerator_memory(sram_address=output_sram_wb_addr + output_sram_wb_offset,
                                                      accelerator_dram_address=OUTPUT_DRAM_ADDR + (i + m_take_chunk_idx) * head_dim * bytes_per_element
                                                                                                  + v_tr_column_idx * bytes_per_element
                                                                                                  + p_row_idx * head_dim * bytes_per_element,
                                                      element_size=v_tr_column_take)

                if v_tr_row_chunk_size_aligned is None:
                    ue.sram_to_accelerator_memory(sram_address=output_sram_wb_addr,
                                                  accelerator_dram_address=OUTPUT_DRAM_ADDR + (i + m_take_chunk_idx) * head_dim * bytes_per_element + v_tr_column_idx * bytes_per_element,
                                                  element_size=m_take_chunk_size * v_tr_column_take,
                                                  stride_bytes_per_chunk=v_tr_column_take * bytes_per_element,
                                                  stride_jump_bytes=head_dim * bytes_per_element)

    total_flops = 1 * head_dim
    total_flops += 2 * 1 * head_dim * kv_len
    if bias_enable:
        total_flops += 1 * kv_len
    total_flops += 1 * kv_len * 5
    total_flops += 2 * 1 * kv_len * head_dim
    return total_flops


def prefill_flash_attention_core(ue, head_dim: int, seq_len: int,
                                 Q_DRAM_ADDR: int, K_DRAM_ADDR: int, V_DRAM_ADDR: int,
                                 OUTPUT_DRAM_ADDR: int, SCRATCH_DRAM_ADDR: int,
                                 IDENTITY_DRAM_ADDR: int, BIAS_DRAM_ADDR: int = None,
                                 num_q_heads: int = 1) -> int:
    """GQA-batched prefill attention. V^T once, then Q@K^T + softmax + sm@V^T per
    Q head. Returns FLOP count."""
    from user_dma_core import UE_FMAX_CONTEXT_SIZE

    bpe = 2
    bias_enable = BIAS_DRAM_ADDR is not None
    head_bytes = seq_len * head_dim * bpe
    SCRATCH_VT = SCRATCH_DRAM_ADDR
    SCRATCH_SM = SCRATCH_DRAM_ADDR + head_dim * seq_len * bpe

    identity_tensor = torch.eye(head_dim, dtype=torch.bfloat16)

    identity_sram = 0x00000
    ue.accelerator_memory_to_sram(IDENTITY_DRAM_ADDR, identity_sram, UE_VECTOR_SIZE * UE_VECTOR_SIZE)

    usable_a_start = identity_sram + UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe
    M, K, N = head_dim, head_dim, seq_len

    usable_b_elements = URAM_NEAR_FULL_ELEMENTS
    N_chunk = min(N, (usable_b_elements // K) // UE_VECTOR_SIZE * UE_VECTOR_SIZE)
    N_chunk_aligned = None
    if N_chunk < UE_VECTOR_SIZE:
        if (K * 32) <= usable_b_elements:
            N_chunk = 32
        elif (K * 16) <= usable_b_elements:
            N_chunk = 16
        else:
            assert False, f"V^T: K={K} too large for URAM"
        N_chunk_aligned = UE_VECTOR_SIZE

    usable_a_elements = URAM_FULL_ELEMENTS - UE_VECTOR_SIZE * UE_VECTOR_SIZE
    out_N = N_chunk_aligned if N_chunk_aligned is not None else N_chunk
    M_chunk = min(M, usable_a_elements // out_N)
    assert M_chunk >= 1

    out_sram = usable_a_start
    b_sram = 0x80000
    for i, m_take in ue.chunk_ranges(M, M_chunk):
        for j, n_take in ue.chunk_ranges(N, N_chunk):
            ue.accelerator_memory_to_sram(V_DRAM_ADDR + j * K * bpe, b_sram, n_take * K)
            for r in range(m_take):
                sram_off = r * (N_chunk_aligned or n_take) * bpe
                ones_idx = identity_tensor[r + i, :].reshape(-1, UE_VECTOR_SIZE).sum(axis=1).argmax(axis=0)
                vec_idx = identity_tensor[r + i, :].reshape(-1, UE_VECTOR_SIZE)[ones_idx, :].argmax(axis=0)
                ue.start_queue_for_bf16_matvec_operation(
                    max_clear_en=0, fmax_context_addr=0,
                    vector_sram_start_addr=identity_sram + vec_idx * UE_VECTOR_SIZE * bpe,
                    matrix_sram_start_addr=b_sram + ones_idx * UE_VECTOR_SIZE * bpe,
                    output_sram_wb_addr=out_sram + sram_off, K=UE_VECTOR_SIZE, N=n_take, stride_z=m_take)
            dst = SCRATCH_VT + i * N * bpe + j * bpe
            if N_chunk_aligned is None:
                ue.sram_to_accelerator_memory(out_sram, dst, m_take * n_take,
                                              stride_bytes_per_chunk=n_take * bpe, stride_jump_bytes=N * bpe)
            else:
                for o in range(m_take):
                    ue.sram_to_accelerator_memory(out_sram + o * N_chunk_aligned * bpe,
                                                  dst + o * N * bpe, n_take)

    M_q, K_q, N_q = seq_len, head_dim, seq_len

    usable_b_elements = URAM_NEAR_FULL_ELEMENTS
    N_chunk = min(N_q, (usable_b_elements // K_q) // UE_VECTOR_SIZE * UE_VECTOR_SIZE)
    N_chunk_aligned = None
    if N_chunk < UE_VECTOR_SIZE:
        if (K_q * 32) <= usable_b_elements:
            N_chunk = 32
        elif (K_q * 16) <= usable_b_elements:
            N_chunk = 16
        else:
            assert False, f"Q@K^T: K={K_q} too large"
        N_chunk_aligned = UE_VECTOR_SIZE

    out_N = N_chunk_aligned if N_chunk_aligned is not None else N_chunk
    M_chunk = min(UE_FMAX_CONTEXT_SIZE, M_q, URAM_FULL_ELEMENTS // (K_q + out_N))
    assert M_chunk >= 1

    a_sram = 0x00000
    b_sram = 0x80000

    for g in range(num_q_heads):
        q_base = Q_DRAM_ADDR + g * head_bytes
        out_base = OUTPUT_DRAM_ADDR + g * head_bytes

        for i, m_take in ue.chunk_ranges(M_q, M_chunk):
            ue.accelerator_memory_to_sram(q_base + i * K_q * bpe, a_sram, m_take * K_q)
            ue.broadcast_mul(scalar=1 / math.sqrt(head_dim),
                             sram_start_addr=a_sram, sram_wb_addr=a_sram,
                             element_size=m_take * K_q)
            qkt_sram = a_sram + m_take * K_q * bpe
            assert qkt_sram < 0x80000

            clear_en = 1
            for j, n_take in ue.chunk_ranges(N_q, N_chunk):
                ue.accelerator_memory_to_sram(K_DRAM_ADDR + j * K_q * bpe, b_sram, n_take * K_q)
                for r in range(m_take):
                    if bias_enable:
                        ue.accelerator_memory_to_bias_sram(
                            BIAS_DRAM_ADDR + ((i + r) * N_q + j) * bpe, n_take)
                    sram_off = r * (N_chunk_aligned or n_take) * bpe
                    ue.start_queue_for_bf16_matvec_operation(
                        max_clear_en=clear_en, fmax_context_addr=r,
                        vector_sram_start_addr=a_sram + r * K_q * bpe,
                        matrix_sram_start_addr=b_sram,
                        output_sram_wb_addr=qkt_sram + sram_off,
                        K=K_q, N=n_take, bias_enable=bias_enable)
                    clear_en = 0
                dst = SCRATCH_SM + j * bpe
                if N_chunk_aligned is None:
                    ue.sram_to_accelerator_memory(qkt_sram, dst, m_take * n_take,
                                                  stride_bytes_per_chunk=n_take * bpe, stride_jump_bytes=N_q * bpe)
                else:
                    for o in range(m_take):
                        ue.sram_to_accelerator_memory(qkt_sram + o * N_chunk_aligned * bpe,
                                                      dst + o * N_q * bpe, n_take)

            max_sm = min((URAM_FULL_ELEMENTS - UE_VECTOR_SIZE) // N_q, UE_FMAX_CONTEXT_SIZE)
            for si, s_take in ue.chunk_ranges(m_take, max_sm):
                ue.accelerator_memory_to_sram(SCRATCH_SM + si * N_q * bpe, a_sram, s_take * N_q)
                for row in range(s_take):
                    ue.start_queue_for_bf16_softmax_operation(
                        fmax_context_addr=row + si,
                        vector_sram_start_addr=a_sram + row * N_q * bpe,
                        output_sram_wb_addr=a_sram + row * N_q * bpe, N=N_q)

                vt_chunk = min((URAM_NEAR_FULL_ELEMENTS // seq_len // UE_VECTOR_SIZE) * UE_VECTOR_SIZE,
                               ((URAM_FULL_ELEMENTS - s_take * seq_len) // s_take // UE_VECTOR_SIZE) * UE_VECTOR_SIZE,
                               head_dim)
                vt_aligned = None
                if vt_chunk < UE_VECTOR_SIZE:
                    vt_aligned = UE_VECTOR_SIZE
                    if seq_len * 32 <= URAM_NEAR_FULL_ELEMENTS:
                        vt_chunk = 32
                    elif seq_len * 16 <= URAM_NEAR_FULL_ELEMENTS:
                        vt_chunk = 16
                    else:
                        assert False, "vt_chunk too small"
                vt_sram = 0x80000
                sm_out_sram = a_sram + s_take * seq_len * bpe

                for vi, v_take in ue.chunk_ranges(head_dim, vt_chunk):
                    ue.accelerator_memory_to_sram(SCRATCH_VT + vi * seq_len * bpe, vt_sram, v_take * seq_len)
                    for pr in range(s_take):
                        wb_off = 0 if vt_aligned is not None else pr * v_take * bpe
                        ue.start_queue_for_bf16_matvec_operation(
                            max_clear_en=0, fmax_context_addr=0,
                            vector_sram_start_addr=a_sram + pr * seq_len * bpe,
                            matrix_sram_start_addr=vt_sram,
                            output_sram_wb_addr=sm_out_sram + wb_off,
                            K=seq_len, N=v_take)
                        if vt_aligned is not None:
                            ue.sram_to_accelerator_memory(sm_out_sram + wb_off,
                                                          out_base + (i + si) * head_dim * bpe + vi * bpe + pr * head_dim * bpe,
                                                          v_take)
                    if vt_aligned is None:
                        ue.sram_to_accelerator_memory(sm_out_sram,
                                                      out_base + (i + si) * head_dim * bpe + vi * bpe,
                                                      s_take * v_take,
                                                      stride_bytes_per_chunk=v_take * bpe,
                                                      stride_jump_bytes=head_dim * bpe)

    total_flops = head_dim * seq_len
    total_flops += num_q_heads * (seq_len * head_dim + 2 * seq_len * head_dim * seq_len)
    if bias_enable:
        total_flops += num_q_heads * seq_len * seq_len
    total_flops += num_q_heads * seq_len * seq_len * 5
    total_flops += num_q_heads * 2 * seq_len * seq_len * head_dim
    return total_flops

# =============================================================================
# PART B — Fixed-shape host helpers
# Weight staging, quantization codecs, weight-bin plumbing, and bring-up
# boilerplate. These do not depend on seq_len; they run once at init.
# =============================================================================
# =============================================================================
# §1 Weight staging / DRAM loaders
# =============================================================================
def store_weight(ue, tensor: torch.Tensor, padded_shape=None) -> int:
    """Pad, cast to bf16, DMA to params DRAM. Returns the DRAM address."""
    bf16 = tensor.to(torch.bfloat16)
    if padded_shape is not None:
        padded = torch.zeros(padded_shape, dtype=torch.bfloat16)
        slices = tuple(slice(0, s) for s in bf16.shape)
        padded[slices] = bf16
        bf16 = padded
    bf16 = bf16.contiguous()
    nbytes = bf16.numel() * 2
    addr = ue.get_params_dram_addr()
    ue.dma_write(DMA_DEVICE_H2C, addr, bf16.flatten(), nbytes)
    ue.allocate_params_dram(nbytes)
    return addr

def store_quantized_weight(ue, raw_data) -> tuple[int, int]:
    """Store Q4_64 raw GGUF data as (scales, packed) in DRAM. Returns (scale_addr, data_addr)."""
    raw_bytes = raw_data.tobytes() if hasattr(raw_data, "tobytes") else bytes(raw_data)
    n_blocks = len(raw_bytes) // 34
    scales_size = n_blocks * 2
    data_size = n_blocks * 32

    scales_np = np.frombuffer(raw_bytes[:scales_size], dtype=np.uint16).copy()
    scale_tensor = torch.from_numpy(scales_np).view(torch.bfloat16)
    scale_addr = ue.get_params_dram_addr()
    ue.dma_write(DMA_DEVICE_H2C, scale_addr, scale_tensor, scales_size)
    ue.allocate_params_dram(scales_size)

    data_np = np.frombuffer(raw_bytes[scales_size:scales_size + data_size], dtype=np.uint8).copy()
    data_tensor = torch.from_numpy(data_np)
    data_addr = ue.get_params_dram_addr()
    ue.dma_write(DMA_DEVICE_H2C, data_addr, data_tensor, data_size)
    ue.allocate_params_dram(data_size)

    return scale_addr, data_addr

def load_weight_cache(bin_path: str) -> dict:
    """Load bin + sibling json manifest. Returns ``{tensor_name: raw_uint8_numpy}``."""
    json_path = bin_path.rsplit(".", 1)[0] + ".json"
    with open(json_path) as f:
        manifest = json.load(f)
    with open(bin_path, "rb") as f:
        raw = f.read()
    cache = {}
    for name, meta in manifest.items():
        cache[name] = np.frombuffer(
            raw[meta["offset"]:meta["offset"] + meta["size"]], dtype=np.uint8
        ).copy()
    return cache

def store_identity_matrix(ue) -> int:
    """Write a UE_VECTOR_SIZE × UE_VECTOR_SIZE bf16 identity into params DRAM. Returns its address."""
    bpe = 2
    size = UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe
    addr = ue.get_params_dram_addr()
    ue.dma_write(DMA_DEVICE_H2C, addr, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16), size)
    ue.allocate_params_dram(size)
    return addr


# §4 Quantization helpers — MOVED to quant_lib.py. The legacy concatenated-
# format codecs (quantize_bf16_to_int4_packed, quantize_q4_64, quantize_fp4_64,
# _FP4_E2M1_TABLE) now live in quant_lib.py's "Legacy concatenated-format
# codecs" section. Import from quant_lib in new code.


# =============================================================================
# §5 Weight-bin plumbing
# =============================================================================
def write_weight_bin(bin_path: str,
                     model,
                     param_filter: Callable[[str], bool],
                     mode: str,
                     qtype: str,
                     qfn: Callable,
                     weight_key_fn: Callable[[str, str], str],
                     quant_suffixes: dict[str, set[str]],
                     transform_fn: Callable[[str, torch.Tensor], torch.Tensor] | None = None) -> None:
    """Write (possibly quantized) weights to ``bin_path`` with a sibling ``.json`` manifest.

    ``weight_key_fn(hf_name, mode)`` returns the short manifest key.
    ``quant_suffixes[qtype]`` is the set of HF-name suffixes that get quantized via ``qfn``.
    ``transform_fn`` optionally rewrites a tensor before bf16 packing (e.g. transpose).
    """
    json_path = bin_path.rsplit(".", 1)[0] + ".json"
    manifest: dict = {}
    count = 0
    with open(bin_path, "wb") as f:
        for pname, param in model.named_parameters():
            if not param_filter(pname):
                continue
            key = weight_key_fn(pname, mode)
            t = param.data
            if any(pname.endswith(s) for s in quant_suffixes[qtype]):
                data, _ = qfn(t)
                raw = data.tobytes()
                key = f"{key}.{qtype}"
            else:
                if transform_fn is not None:
                    t = transform_fn(pname, t)
                raw = t.to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes()
            offset = f.tell()
            f.write(raw)
            manifest[key] = {"offset": offset, "size": len(raw)}
            count += 1
    with open(json_path, "w") as f:
        json.dump(manifest, f)
    print(f"Weights: {count} tensors, {os.path.getsize(bin_path)/1048576:.1f} MB → {bin_path}")


# =============================================================================
# §6 Simple-LM boilerplate
# =============================================================================
_ORIGINAL_PRINT = builtins.print
_SILENT_MODE = False


def set_silent(silent: bool) -> None:
    """Toggle the global silent mode used by ``quiet_print``."""
    global _SILENT_MODE
    _SILENT_MODE = silent


def quiet_print(*args, **kwargs) -> None:
    """``print`` replacement that respects ``set_silent``."""
    if _SILENT_MODE:
        return
    _ORIGINAL_PRINT(*args, **kwargs)


def install_quiet_print() -> None:
    """Monkeypatch ``builtins.print`` with ``quiet_print`` (call once at module import)."""
    builtins.print = quiet_print


def parse_offset(val) -> int:
    """Parse offset/size JSON field: int or hex string like ``'0x24000000'``."""
    if isinstance(val, str):
        return int(val, 0)
    return int(val)


def ensure_hf_model(script_dir: str, cfg: dict, model_cls):
    """Download (if missing) and load an HF causal-LM. Returns ``(model, model_dir)``.

    ``model_cls`` is the class to call ``.from_pretrained`` on (e.g.
    ``transformers.AutoModelForCausalLM``). Kept as a parameter so this module
    does not depend on ``transformers`` at import time.
    """
    from huggingface_hub import snapshot_download

    model_dir = os.path.join(script_dir, cfg["paths"]["hf_model_dir"])
    hf_repo = cfg["paths"]["hf_model_repo"]
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        _ORIGINAL_PRINT(f"Downloading HF model {hf_repo} to {os.path.abspath(model_dir)} ...")
        snapshot_download(repo_id=hf_repo, local_dir=model_dir, local_dir_use_symlinks=False)
        _ORIGINAL_PRINT("Download complete.")
    model = model_cls.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map=None, trust_remote_code=True
    )
    return model, model_dir


def load_config_with_weight_defs(config_path: str) -> dict:
    """Load a per-model config JSON and derive the ``_weight_defs`` offset/size dict.

    Expects the config to have ``file_info.layer_size`` and ``regions`` /
    ``non_layer_regions`` maps of ``{key: {offset, size}}``. Attaches the flat
    ``weight_defs`` dict to the returned config under key ``_weight_defs``.
    """
    with open(config_path) as f:
        cfg = json.load(f)
    weight_defs = {"LAYER_WEIGHT_SIZE": cfg["file_info"]["layer_size"]}
    for key, r in cfg.get("regions", {}).items():
        weight_defs[key] = parse_offset(r["offset"])
        weight_defs[f"{key}_SIZE"] = r["size"]
    for key, r in cfg.get("non_layer_regions", {}).items():
        weight_defs[key] = parse_offset(r["offset"])
        weight_defs[f"{key}_SIZE"] = r["size"]
    cfg["_weight_defs"] = weight_defs
    return cfg


