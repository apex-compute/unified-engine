"""model_lib_core.py

Shared helpers extracted from per-model bring-up files. Every public op uses the
``_core_dram`` naming convention for DRAM-in/DRAM-out wrappers around the SRAM
primitives in ``user_dma_core.UnifiedEngine``.

Functions here are intentionally thin — state lives on the ``ue`` (UnifiedEngine)
instance. This file never mutates ``user_dma_core``; callers still construct a
``UnifiedEngine`` subclass and pass it in.

Grouping:
  §1 Weight staging / DRAM loaders
  §2 DRAM elementwise wrappers
  §3 Fused post-add norm variants (residual + norm in one pass)
  §4 Quantization helpers (bf16 → INT4 packed, Q4_64, FP4 E2M1)
  §5 Weight-bin plumbing (manifest writer, name mapping hook)
  §6 Simple-LM boilerplate (config loading, HF download, quiet print)
"""
from __future__ import annotations

import builtins
import json
import os
import struct
from typing import Callable

import numpy as np
import torch

from user_dma_core import (
    DMA_DEVICE_H2C,
    MEMCPY_TYPE,
    UE_VECTOR_SIZE,
    URAM_NEAR_FULL_ELEMENTS,
)

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
# §4 Quantization helpers
# =============================================================================
# FP4 E2M1 lookup table (16 values, codes 0-7 positive, 8-15 negative).
# Kept here as the canonical default; callers may override by passing their own
# table as ``fp4_table`` to ``quantize_fp4_64``.
_FP4_E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.bfloat16,
)


def quantize_bf16_to_int4_packed(weight_bf16: torch.Tensor,
                                 block_size: int = 64) -> tuple[bytes, bytes]:
    """Quantize bf16 (N_w, K_w) to INT4 packed + per-block bf16 scale along K.

    Returns ``(data_bytes, scale_bytes)``. ``K_w`` must be a multiple of ``block_size``.
    """
    w = weight_bf16.detach().cpu().float().reshape(-1)
    N_w, K_w = weight_bf16.shape
    assert K_w % block_size == 0
    w_blocks = w.reshape(N_w, K_w // block_size, block_size)
    scale = w_blocks.abs().amax(dim=-1).clamp(min=1e-8) / 7.0
    scale_bf16 = -scale.to(torch.bfloat16)
    w_int8 = (w_blocks / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8)
    w_nibbles = w_int8.numpy().astype(np.int16) & 0x0F
    low = w_nibbles[:, :, 0::2].reshape(N_w, -1)
    high = w_nibbles[:, :, 1::2].reshape(N_w, -1)
    packed = (high << 4) | low
    data_bytes = packed.astype(np.uint8).tobytes()
    scale_bytes = scale_bf16.contiguous().view(torch.uint8).numpy().tobytes()
    return data_bytes, scale_bytes


def quantize_q4_64(tensor: torch.Tensor) -> tuple[np.ndarray, int]:
    """INT4 quantization with 64-element blocks. Returns (packed_uint8_array, n_blocks).

    Layout: [bf16 scales per block][packed int4 data]. Matches GGUF Q4_64.
    """
    data = tensor.flatten().cpu().float().numpy()
    n_blocks = int(np.ceil(len(data) / 64))
    padded = np.pad(data, (0, n_blocks * 64 - len(data)))
    blocks = padded.reshape(n_blocks, 64)
    scales = np.max(np.abs(blocks), axis=1)
    scales[scales == 0] = 1.0
    scales /= 7.0
    quantized = np.clip(np.round(blocks / scales[:, None]), -8, 7).astype(np.int8)
    pairs = (quantized.astype(np.uint8) & 0x0F).reshape(n_blocks, 32, 2)
    packed = pairs[:, :, 0] | (pairs[:, :, 1] << 4)
    scale_bytes = torch.tensor(-scales, dtype=torch.float32).to(torch.bfloat16).view(torch.uint16).numpy()
    return np.frombuffer(scale_bytes.tobytes() + packed.tobytes(), dtype=np.uint8), n_blocks


def quantize_fp4_64(tensor: torch.Tensor,
                    fp4_table: torch.Tensor | None = None) -> tuple[np.ndarray, int]:
    """FP4 E2M1 quantization with 64-element blocks. Returns (packed_uint8_array, n_blocks)."""
    table = fp4_table if fp4_table is not None else _FP4_E2M1_TABLE
    x = tensor.to(torch.bfloat16).cpu().flatten()
    n_blocks = int(np.ceil(x.numel() / 64))
    if x.numel() % 64 != 0:
        x = torch.nn.functional.pad(x, (0, n_blocks * 64 - x.numel()))
    blocks = x.view(n_blocks, 64)
    fp4_max = torch.tensor(6.0, dtype=torch.bfloat16)
    scales = (blocks.abs().max(dim=1).values / fp4_max).clamp(min=1e-8).to(torch.bfloat16)
    scaled = (blocks / scales[:, None]).to(torch.bfloat16)
    codes = torch.argmin(torch.abs(scaled.unsqueeze(-1) - table), dim=-1).to(torch.uint8)
    codes_np = codes.numpy().flatten()
    if len(codes_np) % 2 != 0:
        codes_np = np.pad(codes_np, (0, 1))
    packed = (codes_np[0::2] & 0x0F) | ((codes_np[1::2] & 0x0F) << 4)
    scales_np = scales.view(torch.uint16).numpy()
    return np.frombuffer(scales_np.tobytes() + packed.tobytes(), dtype=np.uint8), n_blocks


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
