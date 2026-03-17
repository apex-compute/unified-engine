#!/usr/bin/env python3
"""SmolVLM2-500M on accelerator: fp4_e2m1 (vision) & q4_64 (language)."""
import builtins
import json
import math
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, _REPO_ROOT)

import numpy as np

# Suppress prints during decode/compile (same as Gemma3)
_original_print = builtins.print
_SILENT_MODE = False
def _quiet_print(*args, **kwargs):
    if not _SILENT_MODE:
        _original_print(*args, **kwargs)
builtins.print = _quiet_print
import torch
from huggingface_hub import snapshot_download

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, TYPE, UE_VECTOR_SIZE, UE_ARGMAX_INDEX,
    URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS,
    DRAM_INSTRUCTION_ADDR, INSTRUCTION_REG_REWRITE, MEMCPY_TYPE,
    UnifiedEngine,
)
def _load_smolvlm2_config(path: str | None = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smolvlm2_config.json")
    with open(path) as f:
        return json.load(f)

_SMOLVLM2_CFG = _load_smolvlm2_config()
HF_MODEL_REPO = _SMOLVLM2_CFG["paths"]["hf_model_repo"]
# =============================================================================
# Helper Methods for SmolVLM2
# =============================================================================
def store_weight(ue, tensor, padded_shape=None):
    """Pad, convert to bf16, DMA to device DRAM. Returns DRAM address."""
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
def store_quantized_weight(ue, raw_data):
    """Store Q4_64 raw GGUF data as scale + packed in DRAM. Returns (scale_addr, data_addr)."""
    raw_bytes = raw_data.tobytes() if hasattr(raw_data, 'tobytes') else bytes(raw_data)
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
def load_weight_cache(bin_path):
    """Load bin+json weight file. Returns {tensor_name: raw_numpy_data}."""
    json_path = bin_path.rsplit('.', 1)[0] + '.json'
    with open(json_path) as f:
        manifest = json.load(f)
    with open(bin_path, 'rb') as f:
        raw = f.read()
    cache = {}
    for name, meta in manifest.items():
        cache[name] = np.frombuffer(raw[meta['offset']:meta['offset'] + meta['size']], dtype=np.uint8).copy()
    return cache
def init_hang_prevention(ue) -> None:
    """Stop stale execution and write HALT to instruction DRAM base."""
    print("[Init] Hang prevention: disabling instruction execution...")
    ue.dram_inst_running(False)
    ue.start_capture()
    ue.generate_instruction_halt()
    ue.stop_capture()
    halt_bytes = bytearray()
    for inst in ue.capture_buffer:
        halt_bytes.extend(inst.get_bytes())
    ue.dma_write(DMA_DEVICE_H2C, DRAM_INSTRUCTION_ADDR, halt_bytes, len(halt_bytes))
    ue.clear_capture_buffer()
    print("[Init] HALT written to instruction DRAM base")
def isa_set_register(ue, dst_reg_idx: int, immediate_value: int, timeout_s: float = 10.0) -> None:
    """Set one ISA register to an immediate value via minimal program."""
    ue._inst_id = 0
    ue.start_capture()
    ue.generate_instruction_add_set(dst_reg_idx, immediate_value)
    ue.generate_instruction_halt()
    ue.stop_capture()
    program_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())
    ue.clear_capture_buffer()
    ue.start_execute_from_dram(program_addr)
    ue.wait_queue(timeout_s)
def _make_add_set_bytes(dst_reg: int, immediate_value: int) -> bytes:
    """Build raw 32-byte ADD_SET instruction: dst_reg = immediate_value."""
    import struct
    INSTRUCTION_ADD = 2
    INST_ADD_SET = 4
    w = [0] * 8
    w[0] = ((INST_ADD_SET & 0xF) << 0) | \
           ((dst_reg & 0xF) << 4) | \
           ((dst_reg & 0xF) << 8) | \
           ((0 & 0xF) << 12)
    w[1] = immediate_value & 0xFFFFFFFF
    w[7] = (INSTRUCTION_ADD & 0x7) << 29
    result = bytearray(32)
    for i in range(8):
        result[i*4:(i+1)*4] = struct.pack('<I', w[i] & 0xFFFFFFFF)
    return bytes(result)
def accelerator_memory_to_sram_reg(ue, accelerator_dram_address: int, sram_address: int,
                                    element_size: int, general_reg_src: int,
                                    stride_bytes_per_chunk: int = 0, stride_jump_bytes: int = 0) -> None:
    """accelerator_memory_to_sram with register-based DRAM address."""
    element_size_bytes = element_size * 2
    uram_type, uram_start_addr = ue.sram_address_to_uram_address(sram_address)
    ue.ue_memcpy_from_dram(accelerator_dram_address, element_size_bytes, MEMCPY_TYPE.URAM.value,
                           uram_start_addr, uram_type.value, ue._inst_id,
                           stride_bytes_per_chunk=stride_bytes_per_chunk,
                           stride_jump_bytes=stride_jump_bytes,
                           general_reg_src=general_reg_src)
    ue._inst_id += 1
def overwrite_last_instruction_with_general_register(ue, general_register: int) -> None:
    """Overwrite last captured instruction to use a general register for DRAM address."""
    if not ue.capture_buffer or ue.capture_count == 0:
        print("ERROR: overwrite_last_instruction_with_general_register() called but capture_buffer is empty!")
        return
    if general_register <= 0 or general_register > 15:
        raise ValueError(f"general_register must be in [1, 15], got {general_register}")

    inst = ue.capture_buffer[ue.capture_count - 1]
    w = inst.words
    w[0] = ((0 & 0xF) << 0) | \
           ((general_register & 0xF) << 4) | \
           ((0 & 0xF) << 8) | \
           ((0 & 0xF) << 12)
    w[7] = (w[7] & 0x1FFFFFFF) | ((INSTRUCTION_REG_REWRITE & 0x7) << 29)
def eltwise_add_core_dram(ue, size: int, A_DRAM_ADDR: int, B_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int) -> int:
    """OUTPUT = A + B (DRAM)."""
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
def eltwise_mul_core_dram(ue, size: int, A_DRAM_ADDR: int, B_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int) -> int:
    """OUTPUT = A * B (DRAM)."""
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
def capture_to_raw(ue):
    """Stop capture, extract raw instruction bytes (no halt), clear buffer."""
    ue.stop_capture()
    raw = bytearray()
    for inst in ue.capture_buffer:
        raw.extend(inst.get_bytes())
    ue.clear_capture_buffer()
    return bytes(raw)
def generate_halt_raw(ue):
    """Return raw bytes for a single HALT instruction."""
    ue.start_capture()
    ue.generate_instruction_halt()
    ue.stop_capture()
    raw = bytearray()
    for inst in ue.capture_buffer:
        raw.extend(inst.get_bytes())
    ue.clear_capture_buffer()
    return bytes(raw)
def rms_norm_core_dram_post_add(ue, M: int, N: int,
                                 A_DRAM_ADDR: int, B_DRAM_ADDR: int,
                                 ADDOUTPUT_DRAM_ADDR: int, NORMOUTPUT_DRAM_ADDR: int,
                                 GAMMA_DRAM_ADDR: int = None) -> int:
    """rms_norm(A + B) with residual output."""
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
def layer_norm_core_dram(ue, M: int, N: int,
                         A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                         GAMMA_DRAM_ADDR: int = None, BETA_DRAM_ADDR: int = None) -> int:
    """OUTPUT = LayerNorm(A)."""
    zeros_sram = 0x80000
    zeros_dram = ue.get_params_dram_addr()
    ue.allocate_params_dram(N * 2)
    ue.dma_write(DMA_DEVICE_H2C, zeros_dram, torch.zeros(N, dtype=torch.bfloat16), N * 2)
    ue.accelerator_memory_to_sram(zeros_dram, zeros_sram, N)
    params_sram = zeros_sram + N * 2

    gamma_sram = None
    if GAMMA_DRAM_ADDR is not None:
        gamma_sram = params_sram
        ue.accelerator_memory_to_sram(GAMMA_DRAM_ADDR, gamma_sram, N)
        params_sram += N * 2
    beta_sram = None
    if BETA_DRAM_ADDR is not None:
        beta_sram = params_sram
        ue.accelerator_memory_to_sram(BETA_DRAM_ADDR, beta_sram, N)
        params_sram += N * 2

    vector_sram = 0x00000
    uram_b_used = (params_sram - 0x80000) // 2
    chunk = min(URAM_NEAR_FULL_ELEMENTS // N, (URAM_NEAR_FULL_ELEMENTS - uram_b_used) // N, M)
    assert chunk >= 1

    for i, m_take in ue.chunk_ranges(M, chunk):
        ue.accelerator_memory_to_sram(A_DRAM_ADDR + i * N * 2, vector_sram, m_take * N)
        for j in range(m_take):
            ue.layer_norm_core(vector_sram + j * N * 2, vector_sram + j * N * 2, N, zeros_sram, gamma_sram, beta_sram)
        ue.sram_to_accelerator_memory(vector_sram, OUTPUT_DRAM_ADDR + i * N * 2, m_take * N)
    return 5 * M * N + (M * N if gamma_sram else 0) + (M * N if beta_sram else 0)
def layer_norm_core_dram_post_add(ue, M: int, N: int,
                                   A_DRAM_ADDR: int, B_DRAM_ADDR: int,
                                   ADDOUTPUT_DRAM_ADDR: int, NORMOUTPUT_DRAM_ADDR: int,
                                   GAMMA_DRAM_ADDR: int = None, BETA_DRAM_ADDR: int = None) -> int:
    """layer_norm(A + B) with residual output."""
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
def rope_hf_core_dram(ue, M: int, D: int,
                      X_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                      COS_DRAM_ADDR: int, SIN_DRAM_ADDR: int,
                      COS_LO_PAD_ADDR: int = 0, COS_HI_PAD_ADDR: int = 0,
                      SIN_LO_PAD_ADDR: int = 0, SIN_HI_PAD_ADDR: int = 0) -> int:
    """Prefill RoPE via padded-split bulk load (D<128). Sin first half must be pre-negated."""
    bytes_per_element = 2
    row_bytes = D * bytes_per_element
    half_D = D // 2
    half_bytes = half_D * bytes_per_element

    URAM_B_BASE = 0x80000
    SRAM_SLOT = 128
    uram_x_lo      = 0x00000
    uram_x_hi      = uram_x_lo      + SRAM_SLOT
    uram_a_lo      = uram_x_hi      + SRAM_SLOT
    uram_a_hi      = uram_a_lo      + SRAM_SLOT
    uram_result_lo = uram_a_hi      + SRAM_SLOT
    uram_result_hi = uram_result_lo + SRAM_SLOT

    uram_b_lo       = URAM_B_BASE
    uram_b_hi       = uram_b_lo       + SRAM_SLOT
    uram_cos_lo_bulk = uram_b_hi      + SRAM_SLOT
    uram_cos_hi_bulk = uram_cos_lo_bulk + M * SRAM_SLOT
    uram_sin_lo_bulk = uram_cos_hi_bulk + M * SRAM_SLOT
    uram_sin_hi_bulk = uram_sin_lo_bulk + M * SRAM_SLOT

    ue.accelerator_memory_to_sram(
        accelerator_dram_address=COS_LO_PAD_ADDR,
        sram_address=uram_cos_lo_bulk, element_size=M * D)
    ue.accelerator_memory_to_sram(
        accelerator_dram_address=COS_HI_PAD_ADDR,
        sram_address=uram_cos_hi_bulk, element_size=M * D)
    ue.accelerator_memory_to_sram(
        accelerator_dram_address=SIN_LO_PAD_ADDR,
        sram_address=uram_sin_lo_bulk, element_size=M * D)
    ue.accelerator_memory_to_sram(
        accelerator_dram_address=SIN_HI_PAD_ADDR,
        sram_address=uram_sin_hi_bulk, element_size=M * D)

    for row in range(M):
        row_offset = row * row_bytes
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=X_DRAM_ADDR + row_offset,
            sram_address=uram_x_lo, element_size=half_D)
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=X_DRAM_ADDR + row_offset + half_bytes,
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

        ue.sram_to_accelerator_memory(
            sram_address=uram_result_lo,
            accelerator_dram_address=OUTPUT_DRAM_ADDR + row_offset, element_size=half_D)
        ue.sram_to_accelerator_memory(
            sram_address=uram_result_hi,
            accelerator_dram_address=OUTPUT_DRAM_ADDR + row_offset + half_bytes, element_size=half_D)

    return M * D * 4
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

    # Case 2: last dim changes — decompose into Q1 (last-dim-fixed) + transpose + Q3 (last-dim-fixed)
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

    # Phase 2: Batched transpose (last two dims swapped via identity dot-product)
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
def store_identity_matrix(ue):
    """Store identity matrix in DRAM once. Returns DRAM address."""
    bpe = 2
    size = UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe
    addr = ue.get_params_dram_addr()
    ue.dma_write(DMA_DEVICE_H2C, addr, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16), size)
    ue.allocate_params_dram(size)
    return addr
def decode_flash_attention_core(ue, head_dim: int, kv_len: int,
                                 Q_DRAM_ADDR: int, K_DRAM_ADDR: int, V_DRAM_ADDR: int,
                                 OUTPUT_DRAM_ADDR: int, SCRATCH_DRAM_ADDR: int,
                                 IDENTITY_DRAM_ADDR: int = None,
                                 BIAS_DRAM_ADDR: int = None,
                                 debug_mode: bool = False, SM_OUTPUT_DRAM_ADDR: int = None,
                                 num_q_heads: int = 1, bias_row_stride: int = None) -> int:
    """Decode attention: V^T transpose, scaled Q@K^T, softmax, sm@V^T. GQA via num_q_heads."""
    from user_dma_core import UE_FMAX_CONTEXT_SIZE
    bytes_per_element = 2
    bias_enable = BIAS_DRAM_ADDR is not None
    if debug_mode:
        assert SM_OUTPUT_DRAM_ADDR is not None

    SCRATCH_DRAM_PARTIAL_SM = SCRATCH_DRAM_ADDR + head_dim * kv_len * bytes_per_element
    # I @ V^T: (head_dim, head_dim) @ (kv_len, head_dim)^T -> (head_dim, kv_len)
    M = head_dim
    K = head_dim
    N = kv_len

    identity_matrix_sram_start_addr = 0x00000
    identity_tensor = torch.eye(head_dim, dtype=torch.bfloat16)

    # Load pre-stored identity matrix from DRAM to SRAM (stored once by caller)
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

    # Q @ K^T: (M, head_dim) @ (head_dim, kv_len) -> (M, kv_len)
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

        # Softmax
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
                    assert False, f"v_tr_row_chunk_size too large for URAM"

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
    """GQA-batched prefill attention. V^T once, then Q@K^T + softmax + sm@V^T per Q head."""
    from user_dma_core import UE_FMAX_CONTEXT_SIZE

    bpe = 2
    bias_enable = BIAS_DRAM_ADDR is not None
    head_bytes = seq_len * head_dim * bpe
    SCRATCH_VT = SCRATCH_DRAM_ADDR
    SCRATCH_SM = SCRATCH_DRAM_ADDR + head_dim * seq_len * bpe

    identity_tensor = torch.eye(head_dim, dtype=torch.bfloat16)

    # ========== Step 1: V^T transpose (done ONCE for all Q heads) ==========
    # I @ V^T: [head_dim, head_dim] @ [seq_len, head_dim]^T → [head_dim, seq_len]
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

    # ========== Step 2: Per-head Q@K^T + softmax + sm@V^T ==========
    M_q, K_q, N_q = seq_len, head_dim, seq_len

    # Q@K^T tiling
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
            # Load Q chunk + scale
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

            # Softmax + sm@V^T (per M_chunk rows)
            max_sm = min((URAM_FULL_ELEMENTS - UE_VECTOR_SIZE) // N_q, UE_FMAX_CONTEXT_SIZE)
            for si, s_take in ue.chunk_ranges(m_take, max_sm):
                ue.accelerator_memory_to_sram(SCRATCH_SM + si * N_q * bpe, a_sram, s_take * N_q)
                for row in range(s_take):
                    ue.start_queue_for_bf16_softmax_operation(
                        fmax_context_addr=row + si,
                        vector_sram_start_addr=a_sram + row * N_q * bpe,
                        output_sram_wb_addr=a_sram + row * N_q * bpe, N=N_q)

                # sm @ V^T chunks
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

    total_flops = head_dim * seq_len  # V^T (once)
    total_flops += num_q_heads * (seq_len * head_dim + 2 * seq_len * head_dim * seq_len)  # scale + Q@K^T
    if bias_enable:
        total_flops += num_q_heads * seq_len * seq_len
    total_flops += num_q_heads * seq_len * seq_len * 5  # softmax
    total_flops += num_q_heads * 2 * seq_len * seq_len * head_dim  # sm@V^T
    return total_flops
# =============================================================================
# GGUF generation — quantization helpers
# =============================================================================
# FP4 E2M1 lookup table: 16 values (codes 0-7 positive, 8-15 negative)
_FP4_E2M1_TABLE = torch.tensor(_SMOLVLM2_CFG["quantization"]["fp4_e2m1"]["table"], dtype=torch.bfloat16)
def quantize_q4_64(tensor):
    """INT4 quantization with 64-element blocks. Returns (packed_bytes, n_blocks)."""
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
    scale_bytes = torch.tensor(scales, dtype=torch.float32).to(torch.bfloat16).view(torch.uint16).numpy()
    return np.frombuffer(scale_bytes.tobytes() + packed.tobytes(), dtype=np.uint8), n_blocks
def quantize_fp4_64(tensor):
    """FP4 E2M1 quantization with 64-element blocks. Returns (packed_bytes, n_blocks)."""
    x = tensor.to(torch.bfloat16).cpu().flatten()
    n_blocks = int(np.ceil(x.numel() / 64))
    if x.numel() % 64 != 0:
        x = torch.nn.functional.pad(x, (0, n_blocks * 64 - x.numel()))
    blocks = x.view(n_blocks, 64)
    fp4_max = torch.tensor(6.0, dtype=torch.bfloat16)
    scales = (blocks.abs().max(dim=1).values / fp4_max).clamp(min=1e-8).to(torch.bfloat16)
    scaled = (blocks / scales[:, None]).to(torch.bfloat16)
    # Nearest-neighbor lookup into FP4 table
    codes = torch.argmin(torch.abs(scaled.unsqueeze(-1) - _FP4_E2M1_TABLE), dim=-1).to(torch.uint8)
    codes_np = codes.numpy().flatten()
    if len(codes_np) % 2 != 0:
        codes_np = np.pad(codes_np, (0, 1))
    packed = (codes_np[0::2] & 0x0F) | ((codes_np[1::2] & 0x0F) << 4)
    scales_np = scales.view(torch.uint16).numpy()
    return np.frombuffer(scales_np.tobytes() + packed.tobytes(), dtype=np.uint8), n_blocks
# =============================================================================
# Weight generation — name mapping, quantization dispatch, bin+json writers
# =============================================================================
_LAYER_MAP = {
    'lm': {
        'self_attn.q_proj.weight': 'attn_q.weight', 'self_attn.k_proj.weight': 'attn_k.weight',
        'self_attn.v_proj.weight': 'attn_v.weight', 'self_attn.o_proj.weight': 'attn_output.weight',
        'mlp.gate_proj.weight': 'ffn_gate.weight', 'mlp.up_proj.weight': 'ffn_up.weight',
        'mlp.down_proj.weight': 'ffn_down.weight', 'input_layernorm.weight': 'attn_norm.weight',
        'post_attention_layernorm.weight': 'ffn_norm.weight',
    },
    'vision': {
        'layer_norm1.weight': 'ln1.weight', 'layer_norm1.bias': 'ln1.bias',
        'layer_norm2.weight': 'ln2.weight', 'layer_norm2.bias': 'ln2.bias',
        'self_attn.q_proj.weight': 'attn_q.weight', 'self_attn.q_proj.bias': 'attn_q.bias',
        'self_attn.k_proj.weight': 'attn_k.weight', 'self_attn.k_proj.bias': 'attn_k.bias',
        'self_attn.v_proj.weight': 'attn_v.weight', 'self_attn.v_proj.bias': 'attn_v.bias',
        'self_attn.out_proj.weight': 'attn_out.weight', 'self_attn.out_proj.bias': 'attn_out.bias',
        'mlp.fc1.weight': 'ffn_down.weight', 'mlp.fc1.bias': 'ffn_down.bias',
        'mlp.fc2.weight': 'ffn_up.weight', 'mlp.fc2.bias': 'ffn_up.bias',
    },
}
_TOP_MAP = {
    'lm': {'embed_tokens.weight': 'token_embd.weight', 'norm.weight': 'output_norm.weight',
            'lm_head.weight': 'output.weight'},
    'vision': {
        'vision_model.embeddings.patch_embedding.weight': 'v.patch_embd.weight',
        'vision_model.embeddings.patch_embedding.bias': 'v.patch_embd.bias',
        'vision_model.embeddings.position_embedding.weight': 'v.position_embd.weight',
        'vision_model.post_layernorm.weight': 'v.post_ln.weight',
        'vision_model.post_layernorm.bias': 'v.post_ln.bias',
        'connector.modality_projection.proj.weight': 'mm.model.fc.weight',
    },
}
_QUANT_SUFFIXES = {
    'q4_64': {'q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'o_proj.weight',
              'gate_proj.weight', 'up_proj.weight', 'down_proj.weight', 'lm_head.weight'},
    'fp4_64': {'q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'out_proj.weight',
               'fc1.weight', 'fc2.weight', 'modality_projection.proj.weight'},
}
def _weight_key(hf_name, mode):
    """Map HF param name → short weight key. mode='lm' or 'vision'."""
    name = hf_name
    for pfx in ('model.text_model.', 'model.', 'text_model.'):
        if name.startswith(pfx):
            name = name[len(pfx):]
            break
    if name in _TOP_MAP[mode]:
        return _TOP_MAP[mode][name]
    if mode == 'lm' and name.startswith('layers.'):
        p = name.split('.'); comp = '.'.join(p[2:])
        if comp in _LAYER_MAP['lm']:
            return f'blk.{p[1]}.{_LAYER_MAP["lm"][comp]}'
    elif mode == 'vision' and 'encoder.layers.' in name:
        p = name.split('.'); idx = p.index('layers') + 1; comp = '.'.join(p[idx+1:])
        if comp in _LAYER_MAP['vision']:
            return f'v.blk.{p[idx]}.{_LAYER_MAP["vision"][comp]}'
    return name
def _write_weight_bin(bin_path, model, param_filter, mode, qtype, qfn):
    """Write quantized weights to bin + json manifest (no GGUF dependency)."""
    json_path = bin_path.rsplit('.', 1)[0] + '.json'
    manifest = {}
    count = 0
    with open(bin_path, 'wb') as f:
        for pname, param in model.named_parameters():
            if not param_filter(pname):
                continue
            key = _weight_key(pname, mode)
            t = param.data
            if any(pname.endswith(s) for s in _QUANT_SUFFIXES[qtype]):
                data, _ = qfn(t)
                raw = data.tobytes()
                key = f'{key}.{qtype}'
            else:
                if 'position_embedding.weight' in pname and t.dim() == 2:
                    t = t.t().contiguous()
                raw = t.to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes()
            offset = f.tell()
            f.write(raw)
            manifest[key] = {'offset': offset, 'size': len(raw)}
            count += 1
    with open(json_path, 'w') as f:
        json.dump(manifest, f)
    print(f"Weights: {count} tensors, {os.path.getsize(bin_path)/1048576:.1f} MB → {bin_path}")
def generate_lm_weights(model, output_path):
    """Generate LM Q4_64 weight bin."""
    _write_weight_bin(output_path, model,
        lambda n: 'text_model' in n or 'lm_head' in n, 'lm', 'q4_64', quantize_q4_64)
def generate_vision_weights(model, output_path):
    """Generate vision+connector FP4_64 weight bin."""
    _write_weight_bin(output_path, model,
        lambda n: 'vision_model' in n or 'connector' in n, 'vision', 'fp4_64', quantize_fp4_64)

class SmolVLM2_UnifiedEngine(UnifiedEngine):
    """SmolVLM2-500M accelerator engine: weight loading, compile, inference."""
    # --- Model dimensions (SmolVLM2-500M) — loaded from smolvlm2_config.json ---
    _cfg = _SMOLVLM2_CFG
    HIDDEN_SIZE       = _cfg["lm"]["hidden_size"]
    NUM_LAYERS        = _cfg["lm"]["num_layers"]
    NUM_HEADS         = _cfg["lm"]["num_heads"]
    NUM_KV_HEADS      = _cfg["lm"]["num_kv_heads"]
    GROUP_SIZE        = _cfg["lm"]["group_size"]
    HEAD_DIM          = _cfg["lm"]["head_dim"]
    INTERMEDIATE_SIZE = _cfg["lm"]["intermediate_size"]
    VOCAB_SIZE        = _cfg["lm"]["vocab_size"]
    ROPE_THETA        = _cfg["lm"]["rope_theta"]
    RMS_NORM_EPS      = _cfg["lm"]["rms_norm_eps"]
    MAX_POSITION_EMBEDDINGS = _cfg["lm"]["max_position_embeddings"]
    # ISA register assignments (fixed, used by compile_decoder and run_decoder)
    V_CACHE_SIZE_REG = _cfg["fixed_isa_regs"]["V_CACHE_SIZE_REG"]
    ROPE_SIZE_REG    = _cfg["fixed_isa_regs"]["ROPE_SIZE_REG"]
    TMP_REG          = _cfg["fixed_isa_regs"]["TMP_REG"]

    def __init__(self, script_dir: str = None, lm_weights: str = None, vision_weights: str = None,
                 vision_bf16: bool = False, lm_bf16: bool = False):
        self.script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
        self._cfg = _SMOLVLM2_CFG
        # DRAM layout: adjust for BF16 weights (~968MB params) vs Q4 (~300MB)
        if lm_bf16 or vision_bf16:
            dl = self._cfg["dram_layout"]["bf16"]
            super().__init__(
                params_dram_base=int(dl["params_dram_base"], 16),
                tensor_dram_base=int(dl["tensor_dram_base"], 16),
                program_dram_base=int(dl["program_dram_base"], 16),
            )
        else:
            super().__init__()
        self._isa_reg_counter = 1
        self.vision_bf16 = vision_bf16
        self.lm_bf16 = lm_bf16

        # Weight bin paths (generated by --gen-weights)
        self.lm_weights_path = lm_weights or os.path.join(self.script_dir, self._cfg["paths"]["lm_weights"])
        self.vision_weights_path = vision_weights or os.path.join(self.script_dir, self._cfg["paths"]["vision_weights"])
    # --- ISA register helpers (same as Gemma3) ---
    def reset_isa_reg_counter(self) -> None:
        self._isa_reg_counter = 1
    def alloc_isa_reg(self, reset: bool = False) -> int:
        if reset:
            self._isa_reg_counter = 1
        if self._isa_reg_counter > 15:
            raise ValueError("Exceeded available ISA registers (max 15)")
        reg = self._isa_reg_counter
        self._isa_reg_counter += 1
        return reg
    def clear_inst_id(self) -> None:
        self._inst_id = 0
    def get_arg_max_index(self) -> int:
        return self.read_reg32(UE_ARGMAX_INDEX)
    def overwrite_instruction_with_general_register(self, general_register: int) -> None:
        """Overwrite last instruction to use general register for DRAM address."""
        overwrite_last_instruction_with_general_register(self, general_register)
    def isa_add_set_core(self, dst_reg_idx: int, immediate_value: int, timeout_s: float = 10.0) -> None:
        """Set ISA register to immediate value."""
        isa_set_register(self, dst_reg_idx, immediate_value, timeout_s)
    def rope_hf_core(self, N: int, input_dram_addr: int, output_dram_addr: int,
                     cos_dram_addr: int, sin_dram_addr: int,
                     rope_size_reg: int = None, output_addr_inc_reg: int = None,
                     tmp_reg: int = None) -> int:
        """Per-token decode RoPE (D=64) with register-addressed cos/sin."""
        use_reg = rope_size_reg is not None and tmp_reg is not None
        bpe = 2
        half = N // 2
        half_bytes = half * bpe
        SRAM_SLOT = 128
        uram_x_lo      = 0x00000
        uram_x_hi      = uram_x_lo + SRAM_SLOT
        uram_a_lo      = uram_x_hi + SRAM_SLOT
        uram_a_hi      = uram_a_lo + SRAM_SLOT
        uram_result_lo = uram_a_hi + SRAM_SLOT
        uram_result_hi = uram_result_lo + SRAM_SLOT
        uram_cos_lo    = 0x80000
        uram_cos_hi    = uram_cos_lo + SRAM_SLOT
        uram_sin_lo    = uram_cos_hi + SRAM_SLOT
        uram_sin_hi    = uram_sin_lo + SRAM_SLOT
        uram_b_lo      = uram_sin_hi + SRAM_SLOT
        uram_b_hi      = uram_b_lo + SRAM_SLOT

        # Load input halves
        self.accelerator_memory_to_sram(input_dram_addr, uram_x_lo, half)
        self.accelerator_memory_to_sram(input_dram_addr + half_bytes, uram_x_hi, half)
        # Load cos/sin (register-addressed for decode position)
        if use_reg:
            self.generate_instruction_add_imm(rope_size_reg, cos_dram_addr, tmp_reg)
            accelerator_memory_to_sram_reg(self, cos_dram_addr, uram_cos_lo, half, tmp_reg)
            self.generate_instruction_add_imm(rope_size_reg, cos_dram_addr + half_bytes, tmp_reg)
            accelerator_memory_to_sram_reg(self, cos_dram_addr + half_bytes, uram_cos_hi, half, tmp_reg)
            self.generate_instruction_add_imm(rope_size_reg, sin_dram_addr, tmp_reg)
            accelerator_memory_to_sram_reg(self, sin_dram_addr, uram_sin_lo, half, tmp_reg)
            self.generate_instruction_add_imm(rope_size_reg, sin_dram_addr + half_bytes, tmp_reg)
            accelerator_memory_to_sram_reg(self, sin_dram_addr + half_bytes, uram_sin_hi, half, tmp_reg)
        else:
            self.accelerator_memory_to_sram(cos_dram_addr, uram_cos_lo, half)
            self.accelerator_memory_to_sram(cos_dram_addr + half_bytes, uram_cos_hi, half)
            self.accelerator_memory_to_sram(sin_dram_addr, uram_sin_lo, half)
            self.accelerator_memory_to_sram(sin_dram_addr + half_bytes, uram_sin_hi, half)
        # RoPE: result = x * cos + rotate_half(x) * sin
        self.eltwise_mul_core(uram_x_lo, uram_cos_lo, uram_a_lo, half)
        self.eltwise_mul_core(uram_x_hi, uram_cos_hi, uram_a_hi, half)
        self.eltwise_mul_core(uram_x_hi, uram_sin_lo, uram_b_lo, half)
        self.eltwise_mul_core(uram_x_lo, uram_sin_hi, uram_b_hi, half)
        self.eltwise_add_core(uram_a_lo, uram_b_lo, uram_result_lo, half)
        self.eltwise_add_core(uram_a_hi, uram_b_hi, uram_result_hi, half)
        # Write output (register-addressed for KV cache position)
        if output_addr_inc_reg is not None:
            self.generate_instruction_add_imm(output_addr_inc_reg, output_dram_addr, tmp_reg)
            self.sram_to_accelerator_memory(uram_result_lo, output_dram_addr, half)
            self.overwrite_instruction_with_general_register(tmp_reg)
            self.generate_instruction_add_imm(output_addr_inc_reg, output_dram_addr + half_bytes, tmp_reg)
            self.sram_to_accelerator_memory(uram_result_hi, output_dram_addr + half_bytes, half)
            self.overwrite_instruction_with_general_register(tmp_reg)
        else:
            self.sram_to_accelerator_memory(uram_result_lo, output_dram_addr, half)
            self.sram_to_accelerator_memory(uram_result_hi, output_dram_addr + half_bytes, half)
        return 4 * N

    # --- Weight loading ---
    def weight_init(self) -> None:
        """Load GGUF weights to device DRAM."""
        from transformers import AutoTokenizer
        model_dir = os.path.join(self.script_dir, "smolvlm2_bin", "SmolVLM2-500M-Video-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        # --- Language model weights ---
        if self.lm_bf16:
            # BF16 precision: load directly from HF model
            from transformers import AutoModelForImageTextToText
            hf_model = AutoModelForImageTextToText.from_pretrained(
                model_dir, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map=None).eval()
            text_model = hf_model.model.text_model
            # Embedding
            self.embedding_weight = text_model.embed_tokens.weight.data.clone()
            self.embed_addr = store_weight(self, self.embedding_weight)
            # Per-layer weights
            self.lm_layer_addrs = []
            for i, layer in enumerate(text_model.layers):
                la = {}
                for proj, attr in [('q', 'q_proj'), ('k', 'k_proj'), ('v', 'v_proj'), ('o', 'o_proj')]:
                    la[f'{proj}_weight'] = store_weight(self, getattr(layer.self_attn, attr).weight.data)
                for proj, attr in [('gate', 'gate_proj'), ('up', 'up_proj'), ('down', 'down_proj')]:
                    la[f'{proj}_weight'] = store_weight(self, getattr(layer.mlp, attr).weight.data)
                la['ln1_gamma'] = store_weight(self, layer.input_layernorm.weight.data)
                la['ln2_gamma'] = store_weight(self, layer.post_attention_layernorm.weight.data)
                self.lm_layer_addrs.append(la)
            # Final norm
            self.final_norm_addr = store_weight(self, text_model.norm.weight.data)
            # LM head
            self.lm_head_weight = store_weight(self, hf_model.lm_head.weight.data)
            del hf_model
            print(f"LM weights loaded (BF16): {self.NUM_LAYERS} layers, params DRAM usage: {self.get_params_dram_usage()} bytes")
        else:
            # Q4_64 precision: load from weight bin
            lm_cache = load_weight_cache(self.lm_weights_path)
            # Embedding (BF16) — also keep CPU copy for token lookup
            embed_raw = lm_cache['token_embd.weight']
            embed_bf16 = torch.from_numpy(embed_raw.copy()).view(torch.bfloat16).reshape(self.VOCAB_SIZE, self.HIDDEN_SIZE)
            self.embedding_weight = embed_bf16.clone()
            self.embed_addr = store_weight(self, embed_bf16)
            # Per-layer weights
            self.lm_layer_addrs = []
            for i in range(self.NUM_LAYERS):
                la = {}
                for proj, wkey in [('q', 'attn_q'), ('k', 'attn_k'), ('v', 'attn_v'),
                                        ('o', 'attn_output'), ('gate', 'ffn_gate'),
                                        ('up', 'ffn_up'), ('down', 'ffn_down')]:
                    raw = lm_cache[f'blk.{i}.{wkey}.weight.q4_64']
                    la[f'{proj}_scale'], la[f'{proj}_data'] = store_quantized_weight(self, raw)
                for norm, wkey in [('ln1', 'attn_norm'), ('ln2', 'ffn_norm')]:
                    raw = lm_cache[f'blk.{i}.{wkey}.weight']
                    la[f'{norm}_gamma'] = store_weight(self, torch.from_numpy(raw.copy()).view(torch.bfloat16))
                self.lm_layer_addrs.append(la)
            raw = lm_cache['output_norm.weight']
            self.final_norm_addr = store_weight(self, torch.from_numpy(raw.copy()).view(torch.bfloat16))
            raw = lm_cache['output.weight.q4_64']
            self.lm_head_scale, self.lm_head_data = store_quantized_weight(self, raw)
            print(f"LM weights loaded (Q4): {self.NUM_LAYERS} layers, params DRAM usage: {self.get_params_dram_usage()} bytes")
        # Identity matrix for decode attention
        self.identity_addr = store_identity_matrix(self)
        # DRAM usage check
        params_used = self.get_params_dram_usage()
        params_limit = self._tensor_dram_base - self._params_dram_base
        _original_print(f"  Params DRAM: {params_used/1024/1024:.1f} MB used / {params_limit/1024/1024:.0f} MB available"
                        + (" OVERFLOW!" if params_used > params_limit else ""))
        # --- Vision encoder weights ---
        # Position IDs for SigLIP (shared by both bf16/fp4 paths)
        NPS = 32  # num_patches_per_side = 512 / 16
        boundaries = torch.arange(1.0 / NPS, 1.0, 1.0 / NPS, dtype=torch.float32)
        frac = torch.arange(NPS, dtype=torch.float32) / NPS * (1 - 1e-6)
        buckets = torch.bucketize(frac, boundaries, right=True)
        vis_position_ids = (buckets[:, None] * NPS + buckets[None, :]).flatten()

        if self.vision_bf16:
            # BF16 precision: load directly from HF model (no FP4 quantization)
            from transformers import AutoModelForImageTextToText
            model_dir = os.path.join(self.script_dir, "smolvlm2_bin", "SmolVLM2-500M-Video-Instruct")
            hf_model = AutoModelForImageTextToText.from_pretrained(
                model_dir, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map=None).eval()
            vis_enc = hf_model.model.vision_model
            self.vis_layer_addrs = []
            for i, layer in enumerate(vis_enc.encoder.layers):
                la = {}
                for proj, attr in [('q', 'q_proj'), ('k', 'k_proj'), ('v', 'v_proj'), ('o', 'out_proj')]:
                    linear = getattr(layer.self_attn, attr)
                    la[f'{proj}_weight'] = store_weight(self, linear.weight.data)
                    la[f'{proj}_bias'] = store_weight(self, linear.bias.data)
                for proj, attr in [('fc1', 'fc1'), ('fc2', 'fc2')]:
                    linear = getattr(layer.mlp, attr)
                    la[f'{proj}_weight'] = store_weight(self, linear.weight.data)
                    la[f'{proj}_bias'] = store_weight(self, linear.bias.data)
                for ln, attr in [('ln1', 'layer_norm1'), ('ln2', 'layer_norm2')]:
                    norm = getattr(layer, attr)
                    la[f'{ln}_weight'] = store_weight(self, norm.weight.data)
                    la[f'{ln}_bias'] = store_weight(self, norm.bias.data)
                self.vis_layer_addrs.append(la)
            vis_layers = len(vis_enc.encoder.layers)
            # Patch embedding
            self.patch_weight_addr = store_weight(self, vis_enc.embeddings.patch_embedding.weight.data.reshape(768, 768))
            self.patch_bias_addr = store_weight(self, vis_enc.embeddings.patch_embedding.bias.data)
            # Position embedding (with position ID lookup)
            pos_table = vis_enc.embeddings.position_embedding.weight.data  # [1024, 768]
            self.pos_embed_addr = store_weight(self, pos_table[vis_position_ids])
            # Post-layernorm
            self.vis_post_ln_weight = store_weight(self, vis_enc.post_layernorm.weight.data)
            self.vis_post_ln_bias = store_weight(self, vis_enc.post_layernorm.bias.data)
            # Connector (BF16)
            connector_weight = hf_model.model.connector.modality_projection.proj.weight.data
            self.connector_weight_addr = store_weight(self, connector_weight)
            del hf_model
            params_used = self.get_params_dram_usage()
            _original_print(f"  Vision BF16 loaded: {vis_layers} layers, total params: {params_used/1024/1024:.1f} MB")
        else:
            # FP4_64 precision: load from weight bin
            vis_cache = load_weight_cache(self.vision_weights_path)
            self.vis_layer_addrs = []
            vis_layers = sum(1 for k in vis_cache if k.startswith('v.blk.') and k.endswith('.ln1.weight'))
            for i in range(vis_layers):
                la = {}
                for proj, wkey in [('q', 'attn_q'), ('k', 'attn_k'), ('v', 'attn_v'), ('o', 'attn_out')]:
                    raw = vis_cache[f'v.blk.{i}.{wkey}.weight.fp4_64']
                    la[f'{proj}_scale'], la[f'{proj}_data'] = store_quantized_weight(self, raw)
                    bias_raw = vis_cache[f'v.blk.{i}.{wkey}.bias']
                    la[f'{proj}_bias'] = store_weight(self, torch.from_numpy(bias_raw.copy()).view(torch.bfloat16))
                for proj, wkey in [('fc1', 'ffn_down'), ('fc2', 'ffn_up')]:
                    raw = vis_cache[f'v.blk.{i}.{wkey}.weight.fp4_64']
                    la[f'{proj}_scale'], la[f'{proj}_data'] = store_quantized_weight(self, raw)
                    bias_raw = vis_cache[f'v.blk.{i}.{wkey}.bias']
                    la[f'{proj}_bias'] = store_weight(self, torch.from_numpy(bias_raw.copy()).view(torch.bfloat16))
                for ln, wkey in [('ln1', 'ln1'), ('ln2', 'ln2')]:
                    for suffix in ('weight', 'bias'):
                        raw = vis_cache[f'v.blk.{i}.{wkey}.{suffix}']
                        la[f'{ln}_{suffix}'] = store_weight(self, torch.from_numpy(raw.copy()).view(torch.bfloat16))
                self.vis_layer_addrs.append(la)
            self.patch_weight_addr = store_weight(self, torch.from_numpy(
                vis_cache['v.patch_embd.weight'].copy()).view(torch.bfloat16))
            self.patch_bias_addr = store_weight(self, torch.from_numpy(
                vis_cache['v.patch_embd.bias'].copy()).view(torch.bfloat16))
            pos_raw = torch.from_numpy(vis_cache['v.position_embd.weight'].copy()).view(torch.bfloat16)
            pos_table = pos_raw.reshape(768, 1024).t().contiguous()  # un-transpose from GGUF
            self.pos_embed_addr = store_weight(self, pos_table[vis_position_ids])
            self.vis_post_ln_weight = store_weight(self, torch.from_numpy(
                vis_cache['v.post_ln.weight'].copy()).view(torch.bfloat16))
            self.vis_post_ln_bias = store_weight(self, torch.from_numpy(
                vis_cache['v.post_ln.bias'].copy()).view(torch.bfloat16))
            raw = vis_cache['mm.model.fc.weight.fp4_64']
            self.connector_scale, self.connector_data = store_quantized_weight(self, raw)
            print(f"Vision weights loaded (FP4): {vis_layers} layers + connector, params DRAM usage: {self.get_params_dram_usage()} bytes")
    def tensor_init(self, max_seq_len: int = 512) -> None:
        """Allocate DRAM for activations, KV cache, masks, RoPE."""
        self.max_seq_len = max_seq_len
        seq_len = max_seq_len
        bpe = 2
        # KV cache offset constants (Gemma3-style flat layout: [layer, head, seq, dim])
        self.k_size = self.HEAD_DIM * bpe                             # 128 bytes per position per head
        self.KV_HEAD_STRIDE = seq_len * self.k_size                   # one head, all positions
        self.KV_LAYER_STRIDE = self.NUM_KV_HEADS * self.KV_HEAD_STRIDE  # all heads, one layer

        print(f"Allocate tensor dram start at DRAM address: 0x{self.get_tensor_dram_addr():X}")
        # KV cache: [NUM_LAYERS, NUM_KV_HEADS, max_seq_len, HEAD_DIM], zero-padded
        kv_cache_total = self.NUM_LAYERS * self.KV_LAYER_STRIDE
        self.LAYER0_K_DRAM = self.allocate_tensor_dram(kv_cache_total)
        self.LAYER0_V_DRAM = self.allocate_tensor_dram(kv_cache_total)
        kv_zeros = torch.zeros(self.NUM_LAYERS * self.NUM_KV_HEADS * seq_len * self.HEAD_DIM, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_K_DRAM, kv_zeros)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, kv_zeros)
        # Decoder layer intermediates (shared across 32 layers):
        self.LAYER0_INPUT_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_PRE_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_Q_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_K_PROJ_DRAM = self.allocate_tensor_dram(seq_len * self.NUM_KV_HEADS * self.HEAD_DIM * bpe)
        self.LAYER0_V_PROJ_DRAM = self.allocate_tensor_dram(seq_len * self.NUM_KV_HEADS * self.HEAD_DIM * bpe)
        self.LAYER0_Q_PERM_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_ATTN_OUT_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_ATTN_SCRATCH_DRAM = self.allocate_tensor_dram((self.HEAD_DIM * seq_len + max(self.HEAD_DIM, 64) * seq_len) * bpe)
        self.LAYER0_ATTN_RESULT_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_O_PROJ_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_RESIDUAL_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LAYER0_MLP_GATE_DRAM = self.allocate_tensor_dram(seq_len * self.INTERMEDIATE_SIZE * bpe)
        self.LAYER0_MLP_UP_DRAM = self.allocate_tensor_dram(seq_len * self.INTERMEDIATE_SIZE * bpe)
        self.LAYER0_MLP_MULT_DRAM = self.allocate_tensor_dram(seq_len * self.INTERMEDIATE_SIZE * bpe)
        self.LAYER0_MLP_DOWN_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.FINAL_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.HIDDEN_SIZE * bpe)
        self.LOGITS_DRAM = self.allocate_tensor_dram(1 * self.VOCAB_SIZE * bpe)
        # Causal mask for prefill (written by compile_prefill/load_prefill before use)
        self.CAUSAL_MASK_DRAM = self.allocate_tensor_dram(seq_len * seq_len * bpe)
        # Decode bias: [GROUP_SIZE, max_seq_len] — written each decode step
        self.DECODE_BIAS_DRAM = self.allocate_tensor_dram(self.GROUP_SIZE * seq_len * bpe)
        # RoPE cos/sin tables
        self._load_rope_tables()
        # Vision encoder intermediates (fixed seq=1024, hidden=768, intermediate=3072):
        VS, VH, VI = 1024, 768, 3072
        self.VIS_PIXEL_IN_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_PATCH_PERM_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_PATCH_PROJ_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_IO_A_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_IO_B_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_LN_OUT_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_Q_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_K_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_V_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_Q_PERM_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_K_PERM_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_V_PERM_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_ATTN_OUT_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_ATTN_SCRATCH_DRAM = self.allocate_tensor_dram((64 * VS + 64 * VS) * bpe)
        self.VIS_ATTN_RESULT_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_O_PROJ_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_RESIDUAL_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_MLP_INTER_DRAM = self.allocate_tensor_dram(VS * VI * bpe)
        self.VIS_MLP_OUT_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_POST_LN_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        # Connector: pixel shuffle [1024,768]→[64,12288] (scale_factor=4), linear→[64,960]
        self.VIS_SHUFFLED_DRAM = self.allocate_tensor_dram(64 * 12288 * bpe)
        self.VIS_CONNECTOR_DRAM = self.allocate_tensor_dram(64 * self.HIDDEN_SIZE * bpe)
        # Permute params (identity matrix + temp for bf16_smart_permute_core)
        permute_size = UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe
        self.PERMUTE_PARAMS_DRAM = self.get_params_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, self.PERMUTE_PARAMS_DRAM,
                       torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16), permute_size)
        self.allocate_params_dram(permute_size)
        self.PERMUTE_TEMP_DRAM = self.get_tensor_dram_addr()
        self.allocate_tensor_dram(VS * VH * bpe * 2)  # temp space for permute decomposition
        print(f"    Allocate tensor dram end at DRAM address: 0x{self.get_tensor_dram_addr():X}, usage: {self.get_tensor_dram_usage()} bytes")
    def _load_rope_tables(self) -> None:
        """Precompute cos/sin tables, pre-negate sin, DMA to device."""
        D, S, bpe = self.HEAD_DIM, self.max_seq_len, 2
        inv_freq = 1.0 / (self.ROPE_THETA ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
        freqs = torch.outer(torch.arange(S, dtype=torch.float32), inv_freq)  # [S, D/2]
        cos_full = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=1).to(torch.bfloat16).contiguous()
        sin_full = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=1).to(torch.bfloat16).contiguous()
        sin_full[:, :D // 2] = -sin_full[:, :D // 2]  # pre-negate for HW RoPE kernel
        table_size = S * D * bpe
        self.ROPE_COS_DRAM = self.get_params_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, self.ROPE_COS_DRAM, cos_full.flatten(), table_size)
        self.allocate_params_dram(table_size)
        self.ROPE_SIN_DRAM = self.get_params_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, self.ROPE_SIN_DRAM, sin_full.flatten(), table_size)
        self.allocate_params_dram(table_size)
        self._hw_cos, self._hw_sin = cos_full, sin_full  # CPU copies for decode position updates
        # Padded-split cos/sin for prefill RoPE (D<128 path):
        # Each 32-element half is zero-padded to 64 elements = 128 bytes = 1 URAM row
        half = D // 2
        if D < 128:
            cos_lo = cos_full[:, :half]  # [S, 32]
            cos_hi = cos_full[:, half:]  # [S, 32]
            sin_lo = sin_full[:, :half]  # [S, 32] (already pre-negated)
            sin_hi = sin_full[:, half:]  # [S, 32]
            pad_D = UE_VECTOR_SIZE  # 64 — pad each half to full URAM row
            for name, data in [('COS_LO', cos_lo), ('COS_HI', cos_hi),
                               ('SIN_LO', sin_lo), ('SIN_HI', sin_hi)]:
                padded = torch.zeros(S, pad_D, dtype=torch.bfloat16)
                padded[:, :half] = data
                padded = padded.contiguous()
                addr = self.get_params_dram_addr()
                self.dma_write(DMA_DEVICE_H2C, addr, padded.flatten(), S * pad_D * bpe)
                self.allocate_params_dram(S * pad_D * bpe)
                setattr(self, f'ROPE_{name}_PAD_DRAM', addr)
            print(f"    RoPE padded-split [{S}, {pad_D}] × 4 arrays DMA'd")
        print(f"    RoPE cos/sin [{S}, {D}] DMA'd, theta={self.ROPE_THETA}")

    # --- Compile ---
    def compile_encoder(self) -> int:
        """Compile vision encoder program. Returns DRAM address."""
        from user_dma_core import TYPE
        S, H, D, N_HEADS, I = 1024, 768, 64, 12, 3072
        bpe = 2
        head_stride = S * D * bpe
        permute_dims = [S, N_HEADS, D]          # [1024, 12, 64]
        inv_permute_dims = [N_HEADS, S, D]      # [12, 1024, 64]
        permute_indices = [1, 0, 2]             # swap first two dims

        self.start_capture()

        # === Patch embedding: pixels → [1024, 768] ===
        # Permute: [C=3, H_patches=32, P=16, W_patches=32, P=16] → [32, 32, 3, 16, 16]
        P = 16
        H_patches = 32  # 512 / 16
        bf16_smart_permute_core(self,dims=[3, H_patches, P, H_patches, P], permute_indices=[1, 3, 0, 2, 4],input_dram_addr=self.VIS_PIXEL_IN_DRAM, output_dram_addr=self.VIS_PATCH_PERM_DRAM,params_dram_addr=self.PERMUTE_PARAMS_DRAM, temp_dram_start=self.PERMUTE_TEMP_DRAM)

        # Patch projection: [1024, 768] @ weight + bias → [1024, 768]
        self.matmat_mul_core(M=S, K=H, N=H,A_DRAM_ADDR=self.VIS_PATCH_PERM_DRAM, B_DRAM_ADDR=self.patch_weight_addr,OUTPUT_DRAM_ADDR=self.VIS_PATCH_PROJ_DRAM,C_DRAM_ADDR=self.patch_bias_addr, bias_mode="broadcast_N")

        # Add position embeddings → first layer input
        eltwise_add_core_dram(self, size=S * H,A_DRAM_ADDR=self.VIS_PATCH_PROJ_DRAM, B_DRAM_ADDR=self.pos_embed_addr,OUTPUT_DRAM_ADDR=self.VIS_IO_A_DRAM)

        # === Encoder layers ===
        # Helper: dispatch matmul as BF16 or FP4 based on vision_bf16 flag
        def vis_matmul(M, K, N, A, proj, la, OUT, bias=None, **kw):
            if self.vision_bf16:
                self.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{proj}_weight'],
                    OUTPUT_DRAM_ADDR=OUT, C_DRAM_ADDR=bias, bias_mode="broadcast_N", **kw)
            else:
                self.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{proj}_data'],
                    OUTPUT_DRAM_ADDR=OUT, C_DRAM_ADDR=bias, bias_mode="broadcast_N",
                    is_B_quantized=True, data_type=TYPE.FP4, SCALE_DRAM_ADDR=la[f'{proj}_scale'], **kw)

        for layer_idx, la in enumerate(self.vis_layer_addrs):
            h_in  = self.VIS_IO_A_DRAM if layer_idx % 2 == 0 else self.VIS_IO_B_DRAM
            h_out = self.VIS_IO_B_DRAM if layer_idx % 2 == 0 else self.VIS_IO_A_DRAM
            # LN1
            layer_norm_core_dram(self, M=S, N=H, A_DRAM_ADDR=h_in, OUTPUT_DRAM_ADDR=self.VIS_LN_OUT_DRAM,
                GAMMA_DRAM_ADDR=la['ln1_weight'], BETA_DRAM_ADDR=la['ln1_bias'])
            # Q/K/V projections
            for proj, dst in [('q', self.VIS_Q_DRAM), ('k', self.VIS_K_DRAM), ('v', self.VIS_V_DRAM)]:
                vis_matmul(S, H, H, self.VIS_LN_OUT_DRAM, proj, la, dst, bias=la[f'{proj}_bias'])
            # Permute Q/K/V: [S, 768] → [12, S, 64]
            for src, dst in [(self.VIS_Q_DRAM, self.VIS_Q_PERM_DRAM),
                             (self.VIS_K_DRAM, self.VIS_K_PERM_DRAM),
                             (self.VIS_V_DRAM, self.VIS_V_PERM_DRAM)]:
                bf16_smart_permute_core(self, dims=permute_dims, permute_indices=permute_indices,
                    input_dram_addr=src, output_dram_addr=dst,
                    params_dram_addr=self.PERMUTE_PARAMS_DRAM, temp_dram_start=self.PERMUTE_TEMP_DRAM)
            # 12x flash attention (no causal mask, no GQA)
            for h in range(N_HEADS):
                self.flash_attention_core(head_dim=D, seq_len=S,
                    Q_DRAM_ADDR=self.VIS_Q_PERM_DRAM + h * head_stride,
                    K_DRAM_ADDR=self.VIS_K_PERM_DRAM + h * head_stride,
                    V_DRAM_ADDR=self.VIS_V_PERM_DRAM + h * head_stride,
                    OUTPUT_DRAM_ADDR=self.VIS_ATTN_OUT_DRAM + h * head_stride,
                    SCRATCH_DRAM_ADDR=self.VIS_ATTN_SCRATCH_DRAM)
            # Inverse permute: [12, S, 64] → [S, 768]
            bf16_smart_permute_core(self, dims=inv_permute_dims, permute_indices=permute_indices,
                input_dram_addr=self.VIS_ATTN_OUT_DRAM, output_dram_addr=self.VIS_ATTN_RESULT_DRAM,
                params_dram_addr=self.PERMUTE_PARAMS_DRAM, temp_dram_start=self.PERMUTE_TEMP_DRAM)
            # O projection + residual + LN2
            vis_matmul(S, H, H, self.VIS_ATTN_RESULT_DRAM, 'o', la, self.VIS_O_PROJ_DRAM, bias=la['o_bias'])
            layer_norm_core_dram_post_add(self, M=S, N=H, A_DRAM_ADDR=h_in, B_DRAM_ADDR=self.VIS_O_PROJ_DRAM,
                ADDOUTPUT_DRAM_ADDR=self.VIS_RESIDUAL_DRAM, NORMOUTPUT_DRAM_ADDR=self.VIS_LN_OUT_DRAM,
                GAMMA_DRAM_ADDR=la['ln2_weight'], BETA_DRAM_ADDR=la['ln2_bias'])
            # MLP: fc1 + GELU, fc2, residual
            vis_matmul(S, H, I, self.VIS_LN_OUT_DRAM, 'fc1', la, self.VIS_MLP_INTER_DRAM, bias=la['fc1_bias'], gelu_enable=True)
            vis_matmul(S, I, H, self.VIS_MLP_INTER_DRAM, 'fc2', la, self.VIS_MLP_OUT_DRAM, bias=la['fc2_bias'])
            eltwise_add_core_dram(self, size=S * H,
                A_DRAM_ADDR=self.VIS_RESIDUAL_DRAM, B_DRAM_ADDR=self.VIS_MLP_OUT_DRAM, OUTPUT_DRAM_ADDR=h_out)
        # Post-layernorm
        final_vis = self.VIS_IO_A_DRAM if len(self.vis_layer_addrs) % 2 == 0 else self.VIS_IO_B_DRAM
        layer_norm_core_dram(self, M=S, N=H, A_DRAM_ADDR=final_vis, OUTPUT_DRAM_ADDR=self.VIS_POST_LN_DRAM,
            GAMMA_DRAM_ADDR=self.vis_post_ln_weight, BETA_DRAM_ADDR=self.vis_post_ln_bias)
        # Pixel shuffle: [1024,768] → [64,12288]
        bf16_smart_permute_core(self, dims=[8, 4, 8, 4, H], permute_indices=[0, 2, 1, 3, 4],
            input_dram_addr=self.VIS_POST_LN_DRAM, output_dram_addr=self.VIS_SHUFFLED_DRAM,
            params_dram_addr=self.PERMUTE_PARAMS_DRAM, temp_dram_start=self.PERMUTE_TEMP_DRAM)
        # Connector: [64, 12288] → [64, 960]
        if self.vision_bf16:
            self.matmat_mul_core(M=64, K=12288, N=self.HIDDEN_SIZE,
                A_DRAM_ADDR=self.VIS_SHUFFLED_DRAM, B_DRAM_ADDR=self.connector_weight_addr,
                OUTPUT_DRAM_ADDR=self.VIS_CONNECTOR_DRAM)
        else:
            self.matmat_mul_core(M=64, K=12288, N=self.HIDDEN_SIZE,
                A_DRAM_ADDR=self.VIS_SHUFFLED_DRAM, B_DRAM_ADDR=self.connector_data,
                OUTPUT_DRAM_ADDR=self.VIS_CONNECTOR_DRAM,
                is_B_quantized=True, data_type=TYPE.FP4, SCALE_DRAM_ADDR=self.connector_scale)

        self.stop_capture()
        self.generate_instruction_halt()
        all_bytes = bytearray()
        for inst in self.capture_buffer:
            all_bytes.extend(inst.get_bytes())
        program_addr = self.get_program_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, program_addr, all_bytes, len(all_bytes))
        self.allocate_program_dram(len(all_bytes))
        self.clear_capture_buffer()
        bin_path = os.path.join(self.script_dir, "smolvlm2_bin", "encoder_program.bin")
        with open(bin_path, "wb") as f:
            f.write(all_bytes)
        print(f"    Vision encoder compiled: {len(all_bytes)} bytes → {bin_path}")
        return program_addr

    def compile_prefill(self, seq_len: int) -> None:
        """Compile prefill program (padded to 64). Embed/merge fused at runtime."""
        from user_dma_core import TYPE
        self.prefill_seq_len = seq_len  # actual token count (for causal mask / LM head)
        S = ((seq_len + 63) // 64) * 64  # padded for HW
        self._prefill_padded = S
        H = self.HIDDEN_SIZE           # 960
        KV = self.NUM_KV_HEADS * self.HEAD_DIM  # 320
        D = self.HEAD_DIM              # 64
        I = self.INTERMEDIATE_SIZE     # 2560
        bpe = 2
        head_stride = S * D * bpe

        # Causal mask for padded size: upper triangular -inf
        # Padding rows (seq_len..S-1) must attend to at least one position to avoid
        # softmax(all -inf) = NaN. Let them attend to position 0 (harmless dummy attention).
        causal = torch.full((S, S), -1e38, dtype=torch.bfloat16)
        causal = torch.triu(causal, diagonal=1)        # standard causal: lower tri = 0
        causal[:, seq_len:] = -1e38                     # can't attend to padded cols
        causal[seq_len:, :] = -1e38                     # padded rows: block everything...
        causal[seq_len:, 0] = 0.0                       # ...except position 0 (prevents NaN)
        self.dma_write(DMA_DEVICE_H2C, self.CAUSAL_MASK_DRAM, causal.flatten(), S * S * bpe)

        # Helper: dispatch matmul as BF16 or Q4_64
        def lm_matmul(M, K, N, A, proj, la, OUT, **kw):
            if self.lm_bf16:
                self.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{proj}_weight'],
                    OUTPUT_DRAM_ADDR=OUT, **kw)
            else:
                self.quantized_matmat_core(M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{proj}_data'],
                    OUTPUT_DRAM_ADDR=OUT, SCALE_DRAM_ADDR=la[f'{proj}_scale'], data_type=TYPE.INT4, **kw)

        self.start_capture()
        for layer_idx in range(self.NUM_LAYERS):
            la = self.lm_layer_addrs[layer_idx]
            h_in  = self.LAYER0_INPUT_DRAM if layer_idx % 2 == 0 else self.LAYER0_OUTPUT_DRAM
            h_out = self.LAYER0_OUTPUT_DRAM if layer_idx % 2 == 0 else self.LAYER0_INPUT_DRAM
            # Input layernorm (fused with previous layer's MLP residual for layers 1+)
            if layer_idx == 0:
                self.rms_norm_core_dram(M=S, N=H, A_DRAM_ADDR=h_in,OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=la['ln1_gamma'])
            else:
                rms_norm_core_dram_post_add(self, M=S, N=H,
                    A_DRAM_ADDR=self.LAYER0_RESIDUAL_DRAM, B_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                    ADDOUTPUT_DRAM_ADDR=h_in, NORMOUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                    GAMMA_DRAM_ADDR=la['ln1_gamma'])
            # Q/K/V projections
            lm_matmul(S, H, H, self.LAYER0_PRE_NORM_DRAM, 'q', la, self.LAYER0_Q_DRAM)
            lm_matmul(S, H, KV, self.LAYER0_PRE_NORM_DRAM, 'k', la, self.LAYER0_K_PROJ_DRAM)
            lm_matmul(S, H, KV, self.LAYER0_PRE_NORM_DRAM, 'v', la, self.LAYER0_V_PROJ_DRAM)
            # Permute Q [S,960]→[15,S,64] + RoPE per head
            for h in range(self.NUM_HEADS):
                q_head = self.LAYER0_Q_PERM_DRAM + h * head_stride
                self.accelerator_memory_to_sram(self.LAYER0_Q_DRAM + h * D * bpe, 0x00000, S * D,stride_bytes_per_chunk=D * bpe, stride_jump_bytes=H * bpe)
                self.sram_to_accelerator_memory(0x00000, q_head, S * D)
                rope_hf_core_dram(self, M=S, D=D, X_DRAM_ADDR=q_head, OUTPUT_DRAM_ADDR=q_head,
                    COS_DRAM_ADDR=self.ROPE_COS_DRAM, SIN_DRAM_ADDR=self.ROPE_SIN_DRAM,
                    COS_LO_PAD_ADDR=self.ROPE_COS_LO_PAD_DRAM, COS_HI_PAD_ADDR=self.ROPE_COS_HI_PAD_DRAM,
                    SIN_LO_PAD_ADDR=self.ROPE_SIN_LO_PAD_DRAM, SIN_HI_PAD_ADDR=self.ROPE_SIN_HI_PAD_DRAM)
            # Permute K [S,320]→KV cache [5,S,64] + RoPE per head
            for h in range(self.NUM_KV_HEADS):
                k_cache = self.LAYER0_K_DRAM + layer_idx * self.KV_LAYER_STRIDE + h * self.KV_HEAD_STRIDE
                self.accelerator_memory_to_sram(self.LAYER0_K_PROJ_DRAM + h * D * bpe, 0x00000, S * D,stride_bytes_per_chunk=D * bpe, stride_jump_bytes=KV * bpe)
                self.sram_to_accelerator_memory(0x00000, k_cache, S * D)
                rope_hf_core_dram(self, M=S, D=D, X_DRAM_ADDR=k_cache, OUTPUT_DRAM_ADDR=k_cache,
                    COS_DRAM_ADDR=self.ROPE_COS_DRAM, SIN_DRAM_ADDR=self.ROPE_SIN_DRAM,
                    COS_LO_PAD_ADDR=self.ROPE_COS_LO_PAD_DRAM, COS_HI_PAD_ADDR=self.ROPE_COS_HI_PAD_DRAM,
                    SIN_LO_PAD_ADDR=self.ROPE_SIN_LO_PAD_DRAM, SIN_HI_PAD_ADDR=self.ROPE_SIN_HI_PAD_DRAM)
            # Permute V [S,320]→KV cache [5,S,64] (no RoPE)
            for h in range(self.NUM_KV_HEADS):
                v_cache = self.LAYER0_V_DRAM + layer_idx * self.KV_LAYER_STRIDE + h * self.KV_HEAD_STRIDE
                self.accelerator_memory_to_sram(self.LAYER0_V_PROJ_DRAM + h * D * bpe, 0x00000, S * D,stride_bytes_per_chunk=D * bpe, stride_jump_bytes=KV * bpe)
                self.sram_to_accelerator_memory(0x00000, v_cache, S * D)
            # 5 GQA-batched flash attention (3 Q heads share 1 KV head, V^T done once per group)
            for kv_b in range(self.NUM_KV_HEADS):
                q_start = kv_b * self.GROUP_SIZE * head_stride
                prefill_flash_attention_core(self, head_dim=D, seq_len=S,Q_DRAM_ADDR=self.LAYER0_Q_PERM_DRAM + q_start,K_DRAM_ADDR=self.LAYER0_K_DRAM + layer_idx * self.KV_LAYER_STRIDE + kv_b * self.KV_HEAD_STRIDE,V_DRAM_ADDR=self.LAYER0_V_DRAM + layer_idx * self.KV_LAYER_STRIDE + kv_b * self.KV_HEAD_STRIDE,OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_OUT_DRAM + q_start,SCRATCH_DRAM_ADDR=self.LAYER0_ATTN_SCRATCH_DRAM,IDENTITY_DRAM_ADDR=self.identity_addr,BIAS_DRAM_ADDR=self.CAUSAL_MASK_DRAM,num_q_heads=self.GROUP_SIZE)
            # Inverse permute [15,S,64]→[S,960]
            for h in range(self.NUM_HEADS):
                self.accelerator_memory_to_sram(self.LAYER0_ATTN_OUT_DRAM + h * head_stride, 0x00000, S * D)
                self.sram_to_accelerator_memory(0x00000, self.LAYER0_ATTN_RESULT_DRAM + h * D * bpe, S * D,stride_bytes_per_chunk=D * bpe, stride_jump_bytes=H * bpe)
            # O projection + residual + RMS norm
            lm_matmul(S, H, H, self.LAYER0_ATTN_RESULT_DRAM, 'o', la, self.LAYER0_O_PROJ_DRAM)
            rms_norm_core_dram_post_add(self, M=S, N=H, A_DRAM_ADDR=h_in, B_DRAM_ADDR=self.LAYER0_O_PROJ_DRAM,
                ADDOUTPUT_DRAM_ADDR=self.LAYER0_RESIDUAL_DRAM, NORMOUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                GAMMA_DRAM_ADDR=la['ln2_gamma'])
            # MLP: gate+SiLU, up, gate*up, down
            lm_matmul(S, H, I, self.LAYER0_PRE_NORM_DRAM, 'gate', la, self.LAYER0_MLP_GATE_DRAM, silu_enable=True)
            lm_matmul(S, H, I, self.LAYER0_PRE_NORM_DRAM, 'up', la, self.LAYER0_MLP_UP_DRAM)
            eltwise_mul_core_dram(self, size=S * I,
                A_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM, B_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                OUTPUT_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM)
            lm_matmul(S, I, H, self.LAYER0_MLP_MULT_DRAM, 'down', la, self.LAYER0_MLP_DOWN_DRAM)
            # MLP residual (only last layer — others fused into next layer's input norm)
            if layer_idx == self.NUM_LAYERS - 1:
                eltwise_add_core_dram(self, size=S * H,
                    A_DRAM_ADDR=self.LAYER0_RESIDUAL_DRAM, B_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                    OUTPUT_DRAM_ADDR=h_out)
        # Final norm + LM head (last token only)
        final_buf = self.LAYER0_INPUT_DRAM if self.NUM_LAYERS % 2 == 0 else self.LAYER0_OUTPUT_DRAM
        self.rms_norm_core_dram(M=S, N=H, A_DRAM_ADDR=final_buf,
            OUTPUT_DRAM_ADDR=self.FINAL_NORM_DRAM, GAMMA_DRAM_ADDR=self.final_norm_addr)
        last_token = self.FINAL_NORM_DRAM + (self.prefill_seq_len - 1) * H * bpe
        if self.lm_bf16:
            self.matmat_mul_core(M=1, K=H, N=self.VOCAB_SIZE, A_DRAM_ADDR=last_token,
                B_DRAM_ADDR=self.lm_head_weight, OUTPUT_DRAM_ADDR=self.LOGITS_DRAM)
        else:
            self.quantized_matmat_core(M=1, K=H, N=self.VOCAB_SIZE, A_DRAM_ADDR=last_token,
                B_DRAM_ADDR=self.lm_head_data, OUTPUT_DRAM_ADDR=self.LOGITS_DRAM,
                SCALE_DRAM_ADDR=self.lm_head_scale, data_type=TYPE.INT4)
        # Extract raw instruction bytes (no halt) for runtime fusion
        self._prefill_raw = capture_to_raw(self)
        self._halt_raw = generate_halt_raw(self)

        # Estimate worst-case fused program size:
        # embed gather (2 insts × 32 bytes × max_seq_len) + vision merge (2 × 32 × 64) + prefill + halt
        max_embed_insts = seq_len * 2 * 32  # 2 DMA instructions per token
        max_merge_insts = IMAGE_SEQ_LEN * 2 * 32  # 64 image tokens
        worst_case = max_embed_insts + max_merge_insts + len(self._prefill_raw) + len(self._halt_raw)
        self._prefill_scratch_addr = self.get_program_dram_addr()
        self.allocate_program_dram(worst_case)

        bin_path = os.path.join(self.script_dir, "smolvlm2_bin", f"prefill_program_S{S}.bin")
        with open(bin_path, "wb") as f:
            f.write(self._prefill_raw)
        print(f"    Prefill compiled (S={S}): {len(self._prefill_raw)} bytes raw + scratch {worst_case} bytes → {bin_path}")

    def compile_decoder(self, kv_len_buckets=None) -> None:
        """Compile per-bucket decode programs + per-token embed programs."""
        from user_dma_core import TYPE
        if kv_len_buckets is None:
            kv_len_buckets = [64 * (i + 1) for i in range(self.max_seq_len // 64)]
        self._kv_len_buckets = kv_len_buckets
        H, KV, D, I = self.HIDDEN_SIZE, self.NUM_KV_HEADS * self.HEAD_DIM, self.HEAD_DIM, self.INTERMEDIATE_SIZE
        bpe = 2

        # Helper: dispatch matmul as BF16 or Q4_64
        def lm_matmul(M, K, N, A, proj, la, OUT, **kw):
            if self.lm_bf16:
                self.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{proj}_weight'],
                    OUTPUT_DRAM_ADDR=OUT, **kw)
            else:
                self.quantized_matmat_core(M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{proj}_data'],
                    OUTPUT_DRAM_ADDR=OUT, SCALE_DRAM_ADDR=la[f'{proj}_scale'], data_type=TYPE.INT4, **kw)

        # --- Pre-compile per-token embed programs (on-device DMA gather) ---
        embed_row_bytes = H * bpe
        print(f"    Pre-compiling {self.VOCAB_SIZE} embed programs (on-device DMA gather)...")
        self._decode_embed_raw = []
        for tid in range(self.VOCAB_SIZE):
            src_addr = self.embed_addr + tid * embed_row_bytes
            self.start_capture()
            self.accelerator_memory_to_sram(
                accelerator_dram_address=src_addr,
                sram_address=0x00000,
                element_size=H)
            self.sram_to_accelerator_memory(
                sram_address=0x00000,
                accelerator_dram_address=self.LAYER0_INPUT_DRAM,
                element_size=H)
            self._decode_embed_raw.append(capture_to_raw(self))

        # --- Compile bucket programs (one per kv_len, with halt) ---
        self._decode_bucket_raw = []
        self.clear_inst_id()
        for kv_len in kv_len_buckets:
            self.start_capture()
            for layer_idx in range(self.NUM_LAYERS):
                la = self.lm_layer_addrs[layer_idx]
                h_in  = self.LAYER0_INPUT_DRAM if layer_idx % 2 == 0 else self.LAYER0_OUTPUT_DRAM
                h_out = self.LAYER0_OUTPUT_DRAM if layer_idx % 2 == 0 else self.LAYER0_INPUT_DRAM
                # RMS norm (input_layernorm)
                self.rms_norm_core_dram(M=1, N=H, A_DRAM_ADDR=h_in,
                    OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=la['ln1_gamma'])
                # Q/K/V projections M=1
                lm_matmul(1, H, H, self.LAYER0_PRE_NORM_DRAM, 'q', la, self.LAYER0_Q_DRAM)
                lm_matmul(1, H, KV, self.LAYER0_PRE_NORM_DRAM, 'k', la, self.LAYER0_K_PROJ_DRAM)
                lm_matmul(1, H, KV, self.LAYER0_PRE_NORM_DRAM, 'v', la, self.LAYER0_V_PROJ_DRAM)
                # Store V [1,320]=[1,5*64] to KV cache at current pos (register-addressed)
                for h in range(self.NUM_KV_HEADS):
                    v_cache = self.LAYER0_V_DRAM + layer_idx * self.KV_LAYER_STRIDE + h * self.KV_HEAD_STRIDE
                    self.accelerator_memory_to_sram(self.LAYER0_V_PROJ_DRAM + h * D * bpe, 0x10000, D)
                    self.generate_instruction_add_imm(self.V_CACHE_SIZE_REG, v_cache, self.TMP_REG)
                    self.sram_to_accelerator_memory(0x10000, v_cache, D)
                    self.overwrite_instruction_with_general_register(self.TMP_REG)
                # RoPE on K (5 heads), store to KV cache at current pos
                for h in range(self.NUM_KV_HEADS):
                    k_cache = self.LAYER0_K_DRAM + layer_idx * self.KV_LAYER_STRIDE + h * self.KV_HEAD_STRIDE
                    self.rope_hf_core(N=D,input_dram_addr=self.LAYER0_K_PROJ_DRAM + h * D * bpe,output_dram_addr=k_cache,cos_dram_addr=self.ROPE_COS_DRAM, sin_dram_addr=self.ROPE_SIN_DRAM,rope_size_reg=self.ROPE_SIZE_REG, output_addr_inc_reg=self.V_CACHE_SIZE_REG,tmp_reg=self.TMP_REG)
                # RoPE on Q (15 heads) — Q is [1,960]=[1,15*64], output to Q_PERM
                for h in range(self.NUM_HEADS):
                    self.rope_hf_core(N=D,input_dram_addr=self.LAYER0_Q_DRAM + h * D * bpe,output_dram_addr=self.LAYER0_Q_PERM_DRAM + h * D * bpe,cos_dram_addr=self.ROPE_COS_DRAM, sin_dram_addr=self.ROPE_SIN_DRAM,rope_size_reg=self.ROPE_SIZE_REG, tmp_reg=self.TMP_REG)
                # 5 decode attention calls (GQA: 3 Q heads batched per KV head)
                for kv_b in range(self.NUM_KV_HEADS):
                    q_start = kv_b * self.GROUP_SIZE * D * bpe
                    decode_flash_attention_core(self, head_dim=D, kv_len=kv_len,Q_DRAM_ADDR=self.LAYER0_Q_PERM_DRAM + q_start,K_DRAM_ADDR=self.LAYER0_K_DRAM + layer_idx * self.KV_LAYER_STRIDE + kv_b * self.KV_HEAD_STRIDE,V_DRAM_ADDR=self.LAYER0_V_DRAM + layer_idx * self.KV_LAYER_STRIDE + kv_b * self.KV_HEAD_STRIDE,OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_OUT_DRAM + q_start,SCRATCH_DRAM_ADDR=self.LAYER0_ATTN_SCRATCH_DRAM,IDENTITY_DRAM_ADDR=self.identity_addr,BIAS_DRAM_ADDR=self.DECODE_BIAS_DRAM,num_q_heads=self.GROUP_SIZE,bias_row_stride=self.max_seq_len)
                # O projection M=1
                lm_matmul(1, H, H, self.LAYER0_ATTN_OUT_DRAM, 'o', la, self.LAYER0_O_PROJ_DRAM)
                # Fused residual + RMS norm
                rms_norm_core_dram_post_add(self, M=1, N=H, A_DRAM_ADDR=h_in, B_DRAM_ADDR=self.LAYER0_O_PROJ_DRAM,
                    ADDOUTPUT_DRAM_ADDR=self.LAYER0_RESIDUAL_DRAM, NORMOUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                    GAMMA_DRAM_ADDR=la['ln2_gamma'])
                # MLP M=1
                lm_matmul(1, H, I, self.LAYER0_PRE_NORM_DRAM, 'gate', la, self.LAYER0_MLP_GATE_DRAM, silu_enable=True)
                lm_matmul(1, H, I, self.LAYER0_PRE_NORM_DRAM, 'up', la, self.LAYER0_MLP_UP_DRAM)
                self.accelerator_memory_to_sram(self.LAYER0_MLP_GATE_DRAM, 0x10000, I)
                self.accelerator_memory_to_sram(self.LAYER0_MLP_UP_DRAM, 0x90000, I)
                self.eltwise_mul_core(0x10000, 0x90000, 0x10000, I)
                self.sram_to_accelerator_memory(0x10000, self.LAYER0_MLP_MULT_DRAM, I)
                lm_matmul(1, I, H, self.LAYER0_MLP_MULT_DRAM, 'down', la, self.LAYER0_MLP_DOWN_DRAM)
                # MLP residual M=1
                self.accelerator_memory_to_sram(self.LAYER0_RESIDUAL_DRAM, 0x10000, H)
                self.accelerator_memory_to_sram(self.LAYER0_MLP_DOWN_DRAM, 0x90000, H)
                self.eltwise_add_core(0x10000, 0x90000, 0x10000, H)
                self.sram_to_accelerator_memory(0x10000, h_out, H)
            # Final norm + LM head M=1
            final_buf = self.LAYER0_INPUT_DRAM if self.NUM_LAYERS % 2 == 0 else self.LAYER0_OUTPUT_DRAM
            self.rms_norm_core_dram(M=1, N=H, A_DRAM_ADDR=final_buf,
                OUTPUT_DRAM_ADDR=self.FINAL_NORM_DRAM, GAMMA_DRAM_ADDR=self.final_norm_addr)
            if self.lm_bf16:
                self.matmat_mul_core(M=1, K=H, N=self.VOCAB_SIZE, A_DRAM_ADDR=self.FINAL_NORM_DRAM,
                    B_DRAM_ADDR=self.lm_head_weight, OUTPUT_DRAM_ADDR=self.LOGITS_DRAM)
            else:
                self.quantized_matmat_core(M=1, K=H, N=self.VOCAB_SIZE, A_DRAM_ADDR=self.FINAL_NORM_DRAM,
                    B_DRAM_ADDR=self.lm_head_data, OUTPUT_DRAM_ADDR=self.LOGITS_DRAM,
                    SCALE_DRAM_ADDR=self.lm_head_scale, data_type=TYPE.INT4)
            self.generate_instruction_halt()
            bucket_raw = capture_to_raw(self)
            self._decode_bucket_raw.append(bucket_raw)

        # Allocate scratch for fused decode program: reg_set (64B) + embed + largest bucket
        max_embed = max(len(r) for r in self._decode_embed_raw)
        max_bucket = max(len(r) for r in self._decode_bucket_raw)
        worst_case = 64 + max_embed + max_bucket  # 64 bytes for 2 x 32-byte ADD_SET
        self._decode_scratch_addr = self.get_program_dram_addr()
        self.allocate_program_dram(worst_case)

        # Save to bin files for fast reload
        bin_dir = os.path.join(self.script_dir, "smolvlm2_bin")
        # Save bucket programs
        bucket_meta = []
        bucket_all = bytearray()
        for raw in self._decode_bucket_raw:
            bucket_meta.append({"offset": len(bucket_all), "size": len(raw)})
            bucket_all.extend(raw)
        bin_path = os.path.join(bin_dir, "decoder_program.bin")
        with open(bin_path, "wb") as f:
            f.write(bucket_all)
        # Save embed programs (all same size, stride-indexed)
        embed_stride = len(self._decode_embed_raw[0])
        embed_bin_path = os.path.join(bin_dir, "decoder_embed.bin")
        with open(embed_bin_path, "wb") as f:
            for raw in self._decode_embed_raw:
                f.write(raw)
        meta_path = os.path.join(bin_dir, "decoder_program.json")
        with open(meta_path, "w") as f:
            json.dump({
                "kv_len_buckets": kv_len_buckets,
                "bucket_meta": bucket_meta,
                "embed_stride": embed_stride,
                "embed_count": len(self._decode_embed_raw),
            }, f)
        embed_total = sum(len(r) for r in self._decode_embed_raw)
        print(f"    Decoder compiled: {len(kv_len_buckets)} buckets ({len(bucket_all)} bytes), "
              f"{self.VOCAB_SIZE} embed programs ({embed_total} bytes) → {bin_dir}")

    def _load_bin(self, bin_path: str) -> int:
        """Load a program bin file into program DRAM. Returns DRAM address."""
        with open(bin_path, "rb") as f:
            data = f.read()
        addr = self.get_program_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, addr, data, len(data))
        self.allocate_program_dram(len(data))
        print(f"    Loaded {len(data)} bytes from {os.path.basename(bin_path)}")
        return addr

    def load_encoder(self) -> int:
        """Load pre-compiled encoder from bin. Reproduces compile-time DRAM allocations."""
        N = 768  # vision hidden size
        N_HEADS = 12
        bpe = 2
        zeros = torch.zeros(N, dtype=torch.bfloat16)
        identity = torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16)
        identity_size = UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe

        num_vis_layers = len(self.vis_layer_addrs)
        for _ in range(num_vis_layers):
            # LN1 zeros
            addr = self.get_params_dram_addr()
            self.allocate_params_dram(N * bpe)
            self.dma_write(DMA_DEVICE_H2C, addr, zeros, N * bpe)
            # 12× flash_attention_core identity matrices
            for _ in range(N_HEADS):
                addr = self.get_params_dram_addr()
                self.allocate_params_dram(identity_size)
                self.dma_write(DMA_DEVICE_H2C, addr, identity, identity_size)
            # LN2 post-add zeros
            addr = self.get_params_dram_addr()
            self.allocate_params_dram(N * bpe)
            self.dma_write(DMA_DEVICE_H2C, addr, zeros, N * bpe)
        # Final post-layernorm zeros
        addr = self.get_params_dram_addr()
        self.allocate_params_dram(N * bpe)
        self.dma_write(DMA_DEVICE_H2C, addr, zeros, N * bpe)

        return self._load_bin(os.path.join(self.script_dir, "smolvlm2_bin", "encoder_program.bin"))

    def load_prefill(self, seq_len: int) -> None:
        """Load pre-compiled prefill raw bytes from bin. Sets up causal mask and scratch."""
        S = ((seq_len + 63) // 64) * 64
        self.prefill_seq_len = seq_len
        self._prefill_padded = S
        bpe = 2
        # Causal mask for the padded size
        causal = torch.full((S, S), -1e38, dtype=torch.bfloat16)
        causal = torch.triu(causal, diagonal=1)
        causal[:, seq_len:] = -1e38
        causal[seq_len:, :] = -1e38
        causal[seq_len:, 0] = 0.0
        self.dma_write(DMA_DEVICE_H2C, self.CAUSAL_MASK_DRAM, causal.flatten(), S * S * bpe)
        # Load prefill raw bytes (no halt)
        bin_path = os.path.join(self.script_dir, "smolvlm2_bin", f"prefill_program_S{S}.bin")
        with open(bin_path, "rb") as f:
            self._prefill_raw = f.read()
        self._halt_raw = generate_halt_raw(self)
        # Allocate scratch for fused program
        max_embed_insts = seq_len * 2 * 32
        max_merge_insts = IMAGE_SEQ_LEN * 2 * 32
        worst_case = max_embed_insts + max_merge_insts + len(self._prefill_raw) + len(self._halt_raw)
        self._prefill_scratch_addr = self.get_program_dram_addr()
        self.allocate_program_dram(worst_case)
        print(f"    Loaded prefill raw ({len(self._prefill_raw)} bytes) + scratch ({worst_case} bytes)")

    def load_decoder(self) -> None:
        """Load pre-compiled decoder from bin (embed programs + bucket programs)."""
        bin_dir = os.path.join(self.script_dir, "smolvlm2_bin")
        meta_path = os.path.join(bin_dir, "decoder_program.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self._kv_len_buckets = meta["kv_len_buckets"]
        # Load bucket programs
        bin_path = os.path.join(bin_dir, "decoder_program.bin")
        with open(bin_path, "rb") as f:
            bucket_all = f.read()
        self._decode_bucket_raw = []
        for bm in meta["bucket_meta"]:
            self._decode_bucket_raw.append(bucket_all[bm["offset"]:bm["offset"] + bm["size"]])
        # Load embed programs (stride-indexed)
        embed_bin_path = os.path.join(bin_dir, "decoder_embed.bin")
        with open(embed_bin_path, "rb") as f:
            embed_all = f.read()
        stride = meta["embed_stride"]
        count = meta["embed_count"]
        self._decode_embed_raw = [embed_all[i * stride:(i + 1) * stride] for i in range(count)]
        # Allocate scratch for fused decode program: reg_set (64B) + embed + largest bucket
        max_embed = max(len(r) for r in self._decode_embed_raw)
        max_bucket = max(len(r) for r in self._decode_bucket_raw)
        worst_case = 64 + max_embed + max_bucket  # 64 bytes for 2 x 32-byte ADD_SET
        self._decode_scratch_addr = self.get_program_dram_addr()
        self.allocate_program_dram(worst_case)
        print(f"    Loaded decoder: {len(self._decode_bucket_raw)} buckets, "
              f"{len(self._decode_embed_raw)} embed programs, scratch {worst_case} bytes")

    # --- Run ---
    def run_encoder(self, encoder_addr: int, pixel_values) -> None:
        """Run vision encoder. Output lands in VIS_CONNECTOR_DRAM."""
        pixels_bf16 = pixel_values.to(torch.bfloat16).contiguous().flatten()
        self.dma_to_accelerator_memory(self.VIS_PIXEL_IN_DRAM, pixels_bf16)
        # Encoder FLOPs: patch proj + pos add + N layers + post LN + connector
        VS, VH, VD, VN, VI = 1024, 768, 64, 12, 3072
        n_vis = len(self.vis_layer_addrs)
        attn_per_head = VS * VD + 2 * VS * VD * VS + VS * VS * 5 + 2 * VS * VS * VD
        per_layer = (7 * VS * VH + 3 * (2 * VS * VH * VH + VS * VH)
            + VN * attn_per_head + 2 * VS * VH * VH + VS * VH
            + 8 * VS * VH + 2 * VS * VH * VI + VS * VI
            + 2 * VS * VI * VH + VS * VH + VS * VH)
        enc_flops = (2 * VS * VH * VH + VS * VH + VS * VH
            + n_vis * per_layer + 7 * VS * VH
            + 2 * 64 * 12288 * self.HIDDEN_SIZE)
        self.program_execute(encoder_addr, timeout=30.0, total_flops=enc_flops)
    def run_prefill(self, token_ids, has_image: bool = False, total_flops: int = None) -> int:
        """On-device prefill: embed gather + vision merge + decoder layers. Returns argmax."""
        seq_len = len(token_ids)
        self.seq_len = seq_len
        S = self._prefill_padded
        H = self.HIDDEN_SIZE
        bpe = 2
        embed_row_bytes = H * bpe

        # Epsilon-fill padding rows to prevent RMS norm NaN on zero rows
        if S > seq_len:
            pad_rows = S - seq_len
            epsilon_fill = torch.full((pad_rows * H,), 1e-6, dtype=torch.bfloat16)
            pad_offset = seq_len * embed_row_bytes
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM + pad_offset, epsilon_fill)

        # Generate on-device embedding gather instructions
        self.start_capture()
        for t in range(seq_len):
            token_id = token_ids[t]
            src_addr = self.embed_addr + token_id * embed_row_bytes
            dst_addr = self.LAYER0_INPUT_DRAM + t * embed_row_bytes
            self.accelerator_memory_to_sram(
                accelerator_dram_address=src_addr,
                sram_address=0x00000,
                element_size=H)
            self.sram_to_accelerator_memory(
                sram_address=0x00000,
                accelerator_dram_address=dst_addr,
                element_size=H)

        # Generate on-device vision merge instructions (overwrite image token positions)
        if has_image:
            img_positions = [i for i, t in enumerate(token_ids) if t == IMAGE_TOKEN_ID]
            if len(img_positions) > 0:
                assert len(img_positions) == IMAGE_SEQ_LEN, \
                    f"Expected {IMAGE_SEQ_LEN} image tokens, got {len(img_positions)}"
                for i, pos in enumerate(img_positions):
                    src_addr = self.VIS_CONNECTOR_DRAM + i * embed_row_bytes
                    dst_addr = self.LAYER0_INPUT_DRAM + pos * embed_row_bytes
                    self.accelerator_memory_to_sram(
                        accelerator_dram_address=src_addr,
                        sram_address=0x00000,
                        element_size=H)
                    self.sram_to_accelerator_memory(
                        sram_address=0x00000,
                        accelerator_dram_address=dst_addr,
                        element_size=H)

        dynamic_raw = capture_to_raw(self)

        # Fuse: dynamic embed/merge + prefill layers + halt → single dispatch
        fused = bytearray()
        fused.extend(dynamic_raw)
        fused.extend(self._prefill_raw)
        fused.extend(self._halt_raw)

        # Write fused program to scratch and dispatch
        self.dma_write(DMA_DEVICE_H2C, self._prefill_scratch_addr, bytes(fused), len(fused))
        self.start_execute_from_dram(self._prefill_scratch_addr)
        self.wait_queue(30.0)
        if total_flops is not None:
            self._last_hw_gflops = self.report_flop_rate_gflops(total_flops)
            self._last_total_flops = total_flops
        else:
            self._last_hw_gflops = None
            self._last_total_flops = None
        return self.get_arg_max_index()
    def run_decoder(self, token_id: int, max_new_tokens: int = 512) -> list:
        """Auto-regressive decode loop. Returns generated token IDs."""
        bpe = 2
        global _SILENT_MODE
        generated = []
        H, D, I = self.HIDDEN_SIZE, self.HEAD_DIM, self.INTERMEDIATE_SIZE
        QH, KVH, G = self.NUM_HEADS, self.NUM_KV_HEADS, self.GROUP_SIZE
        self._decode_total_flops = 0
        self._decode_total_hw_ns = 0
        # All embed programs are the same size (2 DMA instructions)
        embed_size = len(self._decode_embed_raw[0])
        REG_SET_SIZE = 64  # 2 x 32-byte ADD_SET instructions
        bucket_offset = REG_SET_SIZE + embed_size  # bucket starts after reg_set + embed
        last_bucket_idx = -1  # track cached bucket to avoid re-DMA
        while len(generated) < max_new_tokens and self.seq_len < self.max_seq_len:
            _SILENT_MODE = True
            self.seq_len += 1
            pos = self.seq_len - 1
            # Decode attention bias: [GROUP_SIZE, max_seq_len] with proper stride
            # Each row must be max_seq_len wide to match bias_row_stride in attention
            aligned_kv = ((self.seq_len + 63) // 64) * 64
            bias = torch.full((self.GROUP_SIZE * self.max_seq_len,), -1e38, dtype=torch.bfloat16)
            for g in range(self.GROUP_SIZE):
                bias[g * self.max_seq_len:g * self.max_seq_len + self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.DECODE_BIAS_DRAM, bias)
            # Fuse: reg_set + embed + bucket → single dispatch (no separate isa_set_register)
            prog_idx = min((self.seq_len - 1) // 64, len(self._decode_bucket_raw) - 1)
            if prog_idx != last_bucket_idx:
                bucket_raw = self._decode_bucket_raw[prog_idx]
                self.dma_write(DMA_DEVICE_H2C,
                               self._decode_scratch_addr + bucket_offset,
                               bucket_raw, len(bucket_raw))
                last_bucket_idx = prog_idx
            # Build fused reg_set + embed bytes (64 + 64 = 128 bytes)
            reg_set_bytes = bytearray()
            reg_set_bytes.extend(_make_add_set_bytes(self.V_CACHE_SIZE_REG, pos * self.k_size))
            reg_set_bytes.extend(_make_add_set_bytes(self.ROPE_SIZE_REG, pos * self.HEAD_DIM * bpe))
            embed_raw = self._decode_embed_raw[token_id]
            fused_head = bytes(reg_set_bytes) + embed_raw
            # DMA fused reg_set + embed to start of scratch (bucket already cached after)
            self.dma_write(DMA_DEVICE_H2C, self._decode_scratch_addr, fused_head, len(fused_head))
            self.start_execute_from_dram(self._decode_scratch_addr)
            self.wait_queue(10.0)
            # Accumulate decode FLOPs from HW cycle counter
            hw_cycles = self.read_reg32(user_dma_core.UE_LATENCY_COUNT_ADDR)
            self._decode_total_hw_ns += hw_cycles * user_dma_core.CLOCK_CYCLE_TIME_NS
            per_layer = (4 * H + 2 * H * (H + 2 * KVH * D)
                + (QH + KVH) * D * 4
                + KVH * (D * aligned_kv + G * (D + 2 * D * aligned_kv
                    + aligned_kv + aligned_kv * 5 + 2 * aligned_kv * D))
                + 2 * H * H + 5 * H + 4 * H * I + I + 2 * I * H + H)
            self._decode_total_flops += self.NUM_LAYERS * per_layer + 4 * H + 2 * H * self.VOCAB_SIZE
            token_id = self.get_arg_max_index()
            generated.append(token_id)
            _SILENT_MODE = False
            if token_id in _SMOLVLM2_CFG["model"]["stop_token_ids"]:  # <|endoftext|>, BOS, PAD, <end_of_utterance>
                break
            _original_print(self.tokenizer.decode([token_id]), end="", flush=True)
        _SILENT_MODE = False
        _original_print("")
        self._decode_hw_gflops = (self._decode_total_flops / self._decode_total_hw_ns
                                  if self._decode_total_hw_ns > 0 else 0)
        return generated
    def program_execute(self, program_addr: int = user_dma_core.DRAM_INSTRUCTION_ADDR,
                        timeout: float = 10.0, total_flops: int = None) -> None:
        """Execute compiled program from DRAM."""
        self.start_execute_from_dram(program_addr)
        self.wait_queue(timeout)
        if total_flops is not None:
            self._last_hw_gflops = self.report_flop_rate_gflops(total_flops)
            self._last_total_flops = total_flops
        else:
            self._last_hw_gflops = None
            self._last_total_flops = None
# =============================================================================
# Image processing + input construction
# =============================================================================
IMAGE_TOKEN_ID = _SMOLVLM2_CFG["special_tokens"]["image_token_id"]
FAKE_TOKEN_ID = _SMOLVLM2_CFG["special_tokens"]["fake_token_id"]
GLOBAL_IMG_TOKEN_ID = _SMOLVLM2_CFG["special_tokens"]["global_img_token_id"]
IMAGE_SEQ_LEN = _SMOLVLM2_CFG["model"]["image_seq_len"]
def process_image(image_path: str, size: int = 512) -> torch.Tensor:
    """Load, resize, normalize image → [3, size, size] bf16."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # normalize to [-1, 1]
    return torch.from_numpy(arr).permute(2, 0, 1).to(torch.bfloat16)
def build_input_ids(tokenizer, prompt: str, has_image: bool = True) -> list:
    """Build token ID list with chat template and optional image token expansion."""
    if has_image:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    tokens = tokenizer.encode(prompt_text)
    if not has_image:
        return tokens
    # Expand single <image> token → <fake><global-img><image>×64<fake>
    expanded = []
    for t in tokens:
        if t == IMAGE_TOKEN_ID:
            expanded += [FAKE_TOKEN_ID, GLOBAL_IMG_TOKEN_ID] + [IMAGE_TOKEN_ID] * IMAGE_SEQ_LEN + [FAKE_TOKEN_ID]
        else:
            expanded.append(t)
    return expanded
# =============================================================================
# Main
# =============================================================================
def main():
    import argparse
    from user_dma_core import set_dma_device

    parser = argparse.ArgumentParser(description="SmolVLM2-500M on accelerator")
    parser.add_argument("--gen-weights", action="store_true", help="Generate quantized weight bins from HF weights")
    _d = _SMOLVLM2_CFG["defaults"]
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--prompt", type=str, default=_d["prompt"], help="Text prompt")
    parser.add_argument("--image", type=str, default=os.path.join(_root, _d["image"]), help="Path to image file (None for text-only)")
    parser.add_argument("--dev", type=str, default=_d["dev"], help="DMA device name")
    parser.add_argument("--cycle", type=float, default=_d["cycle_ns"], help="Clock cycle time in ns")
    parser.add_argument("--max-seq", type=int, default=_d["max_seq"], help="Max sequence length")
    parser.add_argument("--vision-bf16", action="store_true", help="Use BF16 (not FP4) for vision encoder")
    parser.add_argument("--lm-bf16", action="store_true", help="Use BF16 (not Q4_64) for LM decoder")
    parser.add_argument("--bin", action="store_true", help="Load from pre-compiled bin (skip compile)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, _SMOLVLM2_CFG["paths"]["hf_model_dir"])
    weights_dir = os.path.join(script_dir, _SMOLVLM2_CFG["paths"]["weights_dir"])
    # Download HF model if needed
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        print(f"Downloading {HF_MODEL_REPO} ...")
        snapshot_download(repo_id=HF_MODEL_REPO, local_dir=model_dir, local_dir_use_symlinks=False, ignore_patterns=["onnx/*"])
    # Generate quantized weight bins if requested
    if args.gen_weights:
        from transformers import AutoModelForImageTextToText
        print("Loading HF model for weight generation...")
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir, local_files_only=True, torch_dtype='auto', device_map=None, attn_implementation="eager")
        model.cpu().eval()
        generate_lm_weights(model, os.path.join(weights_dir, "smolvlm2_lm_q4.bin"))
        generate_vision_weights(model, os.path.join(weights_dir, "smolvlm2_vision_fp4.bin"))
        del model
        print("Weight generation complete.")
        return
    # Auto-generate weight bins if missing
    lm_bin = os.path.join(weights_dir, "smolvlm2_lm_q4.bin")
    vis_bin = os.path.join(weights_dir, "smolvlm2_vision_fp4.bin")
    if not os.path.exists(lm_bin) or not os.path.exists(vis_bin):
        from transformers import AutoModelForImageTextToText
        print("Weight files not found, generating...")
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir, local_files_only=True, torch_dtype='auto', device_map=None, attn_implementation="eager")
        model.cpu().eval()
        generate_lm_weights(model, lm_bin)
        generate_vision_weights(model, vis_bin)
        del model
        print("Weight generation complete.")

    # --- Hardware inference ---
    set_dma_device(args.dev)
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle
    global _SILENT_MODE
    _SILENT_MODE = True
    ue = SmolVLM2_UnifiedEngine(script_dir=script_dir, vision_bf16=args.vision_bf16, lm_bf16=args.lm_bf16)
    init_hang_prevention(ue)
    ue.weight_init()
    ue.tensor_init(max_seq_len=args.max_seq)
    has_image = args.image is not None
    token_ids = build_input_ids(ue.tokenizer, args.prompt, has_image=has_image)
    seq_len = len(token_ids)
    _SILENT_MODE = False
    if has_image:
        image_path = os.path.abspath(args.image)
        _original_print(f"Image: {image_path}")
    _original_print(f"Prompt: {args.prompt!r} ({seq_len} tokens, image={'yes' if has_image else 'no'})")
    # --- Compile (or load from bin) ---
    _original_print("Compiling...", end=" ", flush=True)
    _SILENT_MODE = True
    timer = time.perf_counter()
    if args.bin:
        if has_image:
            enc_addr = ue.load_encoder()
        ue.load_prefill(seq_len=seq_len)
        ue.load_decoder()
    else:
        if has_image:
            enc_addr = ue.compile_encoder()
        ue.compile_prefill(seq_len=seq_len)
        ue.compile_decoder()
    _SILENT_MODE = False
    _original_print(f"done in {time.perf_counter() - timer:.2f}s")
    # --- Vision encoder ---
    if has_image:
        _original_print("Running vision encoder...", end=" ", flush=True)
        timer = time.perf_counter()
        pixel_values = process_image(args.image)
        _SILENT_MODE = True
        ue.run_encoder(enc_addr, pixel_values)
        _SILENT_MODE = False
        enc_time = time.perf_counter() - timer
        enc_gflops = ue._last_hw_gflops
        enc_total_flops = ue._last_total_flops
        _original_print(f"done in {enc_time:.2f}s ({enc_gflops:.2f} GFLOPS, {enc_total_flops/1e9:.1f} GF)")
    # --- Prefill (on-device embed + vision merge + decoder layers, single dispatch) ---
    _original_print("Running prefill...", end=" ", flush=True)
    timer = time.perf_counter()
    padded_seq = ((seq_len + 63) // 64) * 64
    # Prefill FLOPs (S=padded, 32 layers of: RMS+QKV+RoPE+GQA_attn+O+RMS+MLP + final norm + LM head)
    S = padded_seq
    H, D, I = ue.HIDDEN_SIZE, ue.HEAD_DIM, ue.INTERMEDIATE_SIZE
    KVH, QH, G = ue.NUM_KV_HEADS, ue.NUM_HEADS, ue.GROUP_SIZE
    KV = KVH * D
    attn_per_group = (D * S + G * (S * D + 2 * S * D * S + S * S + S * S * 5 + 2 * S * S * D))
    per_layer = (4 * S * H + 2 * S * H * H + 2 * 2 * S * H * KV
        + (QH + KVH) * S * D * 4 + KVH * attn_per_group
        + 2 * S * H * H + 5 * S * H + 2 * (2 * S * H * I) + S * I + 2 * S * I * H + S * H)
    prefill_flops = ue.NUM_LAYERS * per_layer + 4 * S * H + 2 * 1 * H * ue.VOCAB_SIZE
    _SILENT_MODE = True
    hw_token = ue.run_prefill(token_ids, has_image=has_image, total_flops=prefill_flops)
    _SILENT_MODE = False
    prefill_time = time.perf_counter() - timer
    _original_print(f"done in {prefill_time:.2f}s → {ue.tokenizer.decode([hw_token])!r} ({ue._last_hw_gflops:.2f} GFLOPS, {ue._last_total_flops/1e9:.1f} GF)")
    # Zero out stale KV cache positions (seq_len..padded-1) left by padded prefill
    if padded_seq > seq_len:
        stale_size = (padded_seq - seq_len) * ue.HEAD_DIM
        stale_zeros = torch.zeros(stale_size, dtype=torch.bfloat16)
        for layer in range(ue.NUM_LAYERS):
            for h in range(ue.NUM_KV_HEADS):
                k_stale = ue.LAYER0_K_DRAM + layer * ue.KV_LAYER_STRIDE + h * ue.KV_HEAD_STRIDE + seq_len * ue.HEAD_DIM * 2
                v_stale = ue.LAYER0_V_DRAM + layer * ue.KV_LAYER_STRIDE + h * ue.KV_HEAD_STRIDE + seq_len * ue.HEAD_DIM * 2
                ue.dma_to_accelerator_memory(k_stale, stale_zeros)
                ue.dma_to_accelerator_memory(v_stale, stale_zeros)
    # --- Decode (on-device embed fused with decoder, single dispatch per token) ---
    max_new = args.max_seq - seq_len
    _original_print("Decoding...\n")
    decode_timer = time.perf_counter()
    hw_tokens = ue.run_decoder(hw_token, max_new_tokens=max_new)
    decode_time = time.perf_counter() - decode_timer
    total_time = prefill_time + decode_time
    n_generated = len(hw_tokens)
    _original_print(f"\n--- Performance ---")
    if has_image:
        _original_print(f"  Encoder:  {enc_time:.2f}s ({enc_gflops:.2f} GFLOPS, {enc_total_flops/1e9:.1f} GF)")
    _original_print(f"  Prefill:  {seq_len} tokens in {prefill_time:.2f}s ({prefill_flops/1e9:.1f} GF)")
    _original_print(f"  Decode:   {n_generated} tokens in {decode_time:.2f}s ({n_generated / decode_time:.2f} t/s, {ue._decode_hw_gflops:.2f} GFLOPS, {ue._decode_total_flops/1e9:.1f} GF)")
    _original_print(f"  Total:    {n_generated} tokens in {total_time:.2f}s ({n_generated / total_time:.2f} t/s incl. prefill)")
    if not args.bin:
        _original_print(f"\nTip: Use --bin flag to skip compilation on next run.")

if __name__ == "__main__":
    main()
