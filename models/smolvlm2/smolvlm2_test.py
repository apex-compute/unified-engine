#!/usr/bin/env python3
"""SmolVLM2-500M on accelerator: bf16 (vision) & q4_64 (language). Use --vision-fp4 for FP4 vision."""
import builtins
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
    DMA_DEVICE_H2C, DMA_DEVICE_C2H, TYPE, UE_VECTOR_SIZE, UE_ARGMAX_INDEX, UE_ARGMAX1_INDEX,
    URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS,
    DRAM_INSTRUCTION_ADDR, INSTRUCTION_REG_REWRITE, MEMCPY_TYPE,
    UnifiedEngine, ue_35bit_addr_shifter,
)
from nn_lib import (
    prefill_flash_attention_core,
    smart_bf16_permute_core,
    store_weight, store_quantized_weight, load_weight_cache, store_identity_matrix,
    eltwise_add_core_dram, eltwise_mul_core_dram,
    rms_norm_core_dram_post_add, layer_norm_core_dram_post_add,
)
from quant_lib import (
    quantize_q4_64 as _mlc_quantize_q4_64,
    quantize_fp4_64 as _mlc_quantize_fp4_64,
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
def init_hang_prevention(ue) -> None:
    """Stop stale execution and write HALT to instruction DRAM base."""
    print("[Init] Hang prevention: disabling instruction execution...")
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
    """Build raw 32-byte ADD_SET instruction: dst_reg = immediate_value.
    Encoding must match user_dma_core.generate_instruction / andromeda.c layout."""
    import struct
    INSTRUCTION_ADD = 2
    INST_ADD_SET = 4
    w = [0] * 8
    w[0] = (INSTRUCTION_ADD & 0xF) << 8
    w[1] = ((INST_ADD_SET & 0xF) << 0) | \
           ((dst_reg & 0xF) << 4) | \
           ((dst_reg & 0xF) << 8) | \
           ((0 & 0xF) << 12) | \
           ((immediate_value & 0xFFFF) << 16)
    w[2] = (immediate_value >> 16) & 0xFFFF
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
                           uram_start_addr, uram_type.value,
                           stride_bytes_per_chunk=stride_bytes_per_chunk,
                           stride_jump_bytes=stride_jump_bytes,
                           general_reg_src=general_reg_src,
                           inst_pointer_idx=ue._inst_id)
    ue._inst_id += 1
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
# =============================================================================
# GGUF generation — quantization helpers
# =============================================================================
# FP4 E2M1 lookup table: 16 values (codes 0-7 positive, 8-15 negative). Config-driven
# so smolvlm2_config.json stays authoritative; passed into the shared quantizer.
_FP4_E2M1_TABLE = torch.tensor(_SMOLVLM2_CFG["quantization"]["fp4_e2m1"]["table"], dtype=torch.bfloat16)

def quantize_q4_64(tensor):
    return _mlc_quantize_q4_64(tensor)

def quantize_fp4_64(tensor):
    return _mlc_quantize_fp4_64(tensor, fp4_table=_FP4_E2M1_TABLE)
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
    # Prefill PBI (seq-len-agnostic, gemma3-style): three runtime GPRs primed by run_prefill_v2.
    #   gpr_seq_len    — token count S (matmul/norm/rope/eltwise/gather row loops)
    #   gpr_q_seq_len  — S * GROUP_SIZE (token-major stacked Q rope row loop)
    #   gpr_bucket_idx — aligned(S*GROUP_SIZE)/64, 1-based flash bucket selector
    GPR_SEQ_LEN_REG    = _cfg["fixed_isa_regs"]["GPR_SEQ_LEN_REG"]
    GPR_Q_SEQ_LEN_REG  = _cfg["fixed_isa_regs"]["GPR_Q_SEQ_LEN_REG"]
    GPR_BUCKET_IDX_REG = _cfg["fixed_isa_regs"]["GPR_BUCKET_IDX_REG"]
    # Max prompt length the compile-once prefill program supports (sets flash bucket count).
    PREFILL_MAX_SEQ_LEN = _cfg["model"]["prefill_max_seq_len"]

    def __init__(self, script_dir: str = None, lm_weights: str = None, vision_weights: str = None,
                 vision_bf16: bool = True):
        self.script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
        self._cfg = _SMOLVLM2_CFG
        # DRAM layout: bf16 layout needed when vision uses BF16 weights
        if vision_bf16:
            dl = self._cfg["dram_layout"]["bf16"]
            super().__init__(
                params_dram_base=int(dl["params_dram_base"], 16),
                tensor_dram_base=int(dl["tensor_dram_base"], 16),
                program_dram_base=int(dl["program_dram_base"], 16),
            )
        else:
            super().__init__()
        self._isa_reg_counter = 1
        self.gpr_seq_len = self.GPR_SEQ_LEN_REG      # primed to S in run_prefill_v2
        self.gpr_q_seq_len = self.GPR_Q_SEQ_LEN_REG  # primed to S*GROUP_SIZE in run_prefill_v2
        self.gpr_bucket_idx = self.GPR_BUCKET_IDX_REG  # primed to flash bucket in run_prefill_v2
        self.vision_bf16 = vision_bf16
        self.lm_bf16 = False

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
    def dbg_dram(self, addr: int, n_elems: int, name: str, ref: "torch.Tensor" = None,
                 raise_on_nan: bool = True):
        """Read n_elems bf16 from DRAM and report NaN/Inf + stats (and max-abs-err / cosine vs an
        optional CPU reference). Used by the debug harness to localize where activations go bad."""
        buf = bytearray(n_elems * 2)
        self.dma_read(DMA_DEVICE_C2H, addr, buf, n_elems * 2)
        t = torch.frombuffer(bytes(buf), dtype=torch.bfloat16).float()
        n_nan = int(torch.isnan(t).sum()); n_inf = int(torch.isinf(t).sum())
        finite = t[torch.isfinite(t)]
        lo = float(finite.min()) if finite.numel() else float("nan")
        hi = float(finite.max()) if finite.numel() else float("nan")
        mn = float(finite.mean()) if finite.numel() else float("nan")
        info = {"t": t, "nan": n_nan, "inf": n_inf, "cos": None, "cos_excl8": None,
                "cos_tok": None, "relL2": None}
        msg = (f"[dbg] {name:30s} n={n_elems:<8d} nan={n_nan:<6d} inf={n_inf:<5d} "
               f"min={lo:+.3f} max={hi:+.3f} mean={mn:+.4f}")
        if ref is not None:
            rflat = ref.flatten().float()
            m = min(rflat.numel(), t.numel())
            d, r = t[:m], rflat[:m]
            err = float((d - r).abs().max())
            info["cos"] = float(torch.nn.functional.cosine_similarity(d, r, dim=0))
            info["relL2"] = float((d - r).norm() / (r.norm() + 1e-9))
            msg += f"  maxerr={err:.3f} cos={info['cos']:.4f} relL2={info['relL2']:.3f}"
            # Outlier-robust: SmolLM2 "massive activations" (a few huge channels) dominate plain
            # cosine. If ref is 2D [T,H], also report cosine with the top-8 |channel| dims masked
            # and the mean per-token cosine — these expose errors the big dims hide.
            if ref.dim() == 2 and d.numel() == ref.numel():
                T, Hd = ref.shape
                dd, rr = d.view(T, Hd), r.view(T, Hd)
                chan = rr.abs().mean(0)
                topk = torch.topk(chan, min(8, Hd)).indices
                mask = torch.ones(Hd); mask[topk] = 0.0
                info["cos_excl8"] = float(torch.nn.functional.cosine_similarity(
                    (dd * mask).flatten(), (rr * mask).flatten(), dim=0))
                info["cos_tok"] = float(torch.nn.functional.cosine_similarity(dd, rr, dim=1).mean())
                msg += f" cos_excl8={info['cos_excl8']:.4f} cos_tok={info['cos_tok']:.4f}"
        print(msg, flush=True)
        if raise_on_nan and (n_nan or n_inf):
            raise FloatingPointError(f"NaN/Inf detected in {name} (nan={n_nan}, inf={n_inf})")
        return info
    def isa_add_set_core(self, dst_reg_idx: int, immediate_value: int, timeout_s: float = 10.0) -> None:
        """Set ISA register to immediate value."""
        isa_set_register(self, dst_reg_idx, immediate_value, timeout_s)
    def rope_hf_core(self, N: int, input_dram_addr: int, output_dram_addr: int,
                     cos_dram_addr: int, sin_dram_addr: int,
                     rope_size_reg: int = None, output_addr_inc_reg: int = None,
                     tmp_reg: int = None) -> int:
        """Per-token decode RoPE for N=64 (half=32). Caller must bracket with start/stop_capture.

        Compute flow — splits N into lo/hi halves (32 each) to meet 128-byte SRAM alignment:
          1. Load x_lo[32], x_hi[32]       → sram_x_lo, sram_x_hi      (two 128-byte slots)
          2. Load cos_lo[32], cos_hi[32]   → sram_cos_lo, sram_cos_hi
             Load sin_lo[32], sin_hi[32]   → sram_sin_lo, sram_sin_hi   (sin pre-negated in table)
          3. a_lo[32] = x_lo * cos_lo      (eltwise_mul)
             a_hi[32] = x_hi * cos_hi      (eltwise_mul)
          4. b_lo[32] = x_hi * sin_lo      (eltwise_mul, rotate_half lo: x_hi * (-sin_lo))
             b_hi[32] = x_lo * sin_hi      (eltwise_mul, rotate_half hi: x_lo * sin_hi)
          5. result_lo[32] = a_lo + b_lo   (eltwise_add)
             result_hi[32] = a_hi + b_hi   (eltwise_add)
          6. Write result_lo, result_hi → output_dram_addr

        rope_size_reg: ISA register holding a per-position byte offset added to cos/sin addresses
          at runtime. tmp_reg required.
        output_addr_inc_reg: ISA register holding a per-position byte offset applied to output_dram_addr.
        Returns: approximate FLOPs (4*N).
        """
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
            self.generate_instruction_add_imm(rope_size_reg, ue_35bit_addr_shifter(cos_dram_addr), tmp_reg)
            accelerator_memory_to_sram_reg(self, cos_dram_addr, uram_cos_lo, half, tmp_reg)
            self.generate_instruction_add_imm(rope_size_reg, ue_35bit_addr_shifter(cos_dram_addr + half_bytes), tmp_reg)
            accelerator_memory_to_sram_reg(self, cos_dram_addr + half_bytes, uram_cos_hi, half, tmp_reg)
            self.generate_instruction_add_imm(rope_size_reg, ue_35bit_addr_shifter(sin_dram_addr), tmp_reg)
            accelerator_memory_to_sram_reg(self, sin_dram_addr, uram_sin_lo, half, tmp_reg)
            self.generate_instruction_add_imm(rope_size_reg, ue_35bit_addr_shifter(sin_dram_addr + half_bytes), tmp_reg)
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
            self.generate_instruction_add_imm(output_addr_inc_reg, ue_35bit_addr_shifter(output_dram_addr), tmp_reg)
            self.sram_to_accelerator_memory(uram_result_lo, output_dram_addr, half)
            self.overwrite_instruction_with_general_register(tmp_reg)
            self.generate_instruction_add_imm(output_addr_inc_reg, ue_35bit_addr_shifter(output_dram_addr + half_bytes), tmp_reg)
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
        self.bytes_per_element = bpe
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
        # --- gemma3-style stacked-GQA flash buffers (seq-len-agnostic prefill v2) ---
        # One bucket-dispatcher flash per kv-group operates on a square of aligned(S*GROUP_SIZE)
        # rows. Buffers are sized for the largest bucket = aligned(PREFILL_MAX_SEQ_LEN*GROUP_SIZE).
        G = self.GROUP_SIZE
        qmax = ((self.PREFILL_MAX_SEQ_LEN * G + 63) // 64) * 64
        self.PREFILL_QMAX = qmax
        self.FLASH_NUM_BUCKETS = qmax // UE_VECTOR_SIZE
        D = self.HEAD_DIM
        self.FLASH_Q_DRAM = self.allocate_tensor_dram(qmax * D * bpe)       # token-major stacked Q
        self.FLASH_K_DRAM = self.allocate_tensor_dram(qmax * D * bpe)       # K duplicated token-major
        self.FLASH_V_DRAM = self.allocate_tensor_dram(qmax * D * bpe)       # V duplicated token-major
        self.FLASH_OUT_DRAM = self.allocate_tensor_dram(qmax * D * bpe)     # attention output (stacked)
        self.FLASH_SCRATCH_DRAM = self.allocate_tensor_dram(
            UE_VECTOR_SIZE * qmax * bpe + D * qmax * bpe)                   # Vᵀ + softmax scratch
        self.FLASH_ATTN_P_DRAM = self.allocate_tensor_dram(qmax * qmax * bpe)  # fused QKᵀ+softmax probs
        self.FLASH_BIAS_DRAM = self.allocate_tensor_dram(qmax * qmax * bpe)    # block-diagonal causal bias
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
        # Contiguous packed table for the prefill d64 PBI RoPE (rope_hf_core_dram_d64_pbi):
        # per token [cos(D) || sin(D)] = 2D elems, sin first half pre-negated. The d64 path
        # asserts sin_dram_addr == cos_dram_addr + D*bpe, so cos/sin must be interleaved per
        # token (NOT two separate [S,D] tables like ROPE_COS_DRAM/ROPE_SIN_DRAM above).
        packed = torch.cat([cos_full, sin_full], dim=1).contiguous()  # [S, 2D]
        self.ROPE_PACKED_DRAM = self.get_params_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, self.ROPE_PACKED_DRAM, packed.flatten(), S * 2 * D * bpe)
        self.allocate_params_dram(S * 2 * D * bpe)
        # Token-major-DUPLICATED packed table for the gemma3-style stacked-Q rope. The stacked Q
        # for one kv-group is [S, GROUP_SIZE, D] flattened to rows r = t*G + g; roping it as one
        # M=S*G d64 loop needs row r to use token (r//G)'s cos/sin. There is no d64 GQA rope, so
        # we pre-duplicate each token's [cos||sin] row GROUP_SIZE times → row r holds packed[r//G].
        G = self.GROUP_SIZE
        packed_gqa = packed.repeat_interleave(G, dim=0).contiguous()  # [G*S, 2D]
        self.ROPE_PACKED_GQA_DRAM = self.get_params_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, self.ROPE_PACKED_GQA_DRAM, packed_gqa.flatten(), G * S * 2 * D * bpe)
        self.allocate_params_dram(G * S * 2 * D * bpe)
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
        smart_bf16_permute_core(self,dims=[3, H_patches, P, H_patches, P], permute_indices=[1, 3, 0, 2, 4],input_dram_addr=self.VIS_PIXEL_IN_DRAM, output_dram_addr=self.VIS_PATCH_PERM_DRAM,params_dram_addr=self.PERMUTE_PARAMS_DRAM, temp_dram_start=self.PERMUTE_TEMP_DRAM)

        # Patch projection: [1024, 768] @ weight + bias → [1024, 768]
        self.matmat_mul_core(M=S, K=H, N=H,A_DRAM_ADDR=self.VIS_PATCH_PERM_DRAM, B_DRAM_ADDR=self.patch_weight_addr,OUTPUT_DRAM_ADDR=self.VIS_PATCH_PROJ_DRAM,C_DRAM_ADDR=self.patch_bias_addr, bias_mode="broadcast_N")

        # Add position embeddings → first layer input
        eltwise_add_core_dram(self, size=S * H,A_DRAM_ADDR=self.VIS_PATCH_PROJ_DRAM, B_DRAM_ADDR=self.pos_embed_addr,OUTPUT_DRAM_ADDR=self.VIS_IO_A_DRAM)

        # === Encoder layers ===
        def vis_matmul(M, K, N, A, proj, la, OUT, bias=None, **kw):
            if self.vision_bf16:
                self.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{proj}_weight'],
                    OUTPUT_DRAM_ADDR=OUT, C_DRAM_ADDR=bias, bias_mode="broadcast_N", **kw)
            else:
                self.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{proj}_data'],
                    OUTPUT_DRAM_ADDR=OUT, C_DRAM_ADDR=bias, bias_mode="broadcast_N",
                    is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=la[f'{proj}_scale'], **kw)

        for layer_idx, la in enumerate(self.vis_layer_addrs):
            h_in  = self.VIS_IO_A_DRAM if layer_idx % 2 == 0 else self.VIS_IO_B_DRAM
            h_out = self.VIS_IO_B_DRAM if layer_idx % 2 == 0 else self.VIS_IO_A_DRAM
            # LN1
            self.layer_norm_core_dram(M=S, N=H, A_DRAM_ADDR=h_in, OUTPUT_DRAM_ADDR=self.VIS_LN_OUT_DRAM,
                GAMMA_DRAM_ADDR=la['ln1_weight'], BETA_DRAM_ADDR=la['ln1_bias'])
            # Q/K/V projections
            for proj, dst in [('q', self.VIS_Q_DRAM), ('k', self.VIS_K_DRAM), ('v', self.VIS_V_DRAM)]:
                vis_matmul(S, H, H, self.VIS_LN_OUT_DRAM, proj, la, dst, bias=la[f'{proj}_bias'])
            # Permute Q/K/V: [S, 768] → [12, S, 64]
            for src, dst in [(self.VIS_Q_DRAM, self.VIS_Q_PERM_DRAM),
                             (self.VIS_K_DRAM, self.VIS_K_PERM_DRAM),
                             (self.VIS_V_DRAM, self.VIS_V_PERM_DRAM)]:
                smart_bf16_permute_core(self, dims=permute_dims, permute_indices=permute_indices,
                    input_dram_addr=src, output_dram_addr=dst,
                    params_dram_addr=self.PERMUTE_PARAMS_DRAM, temp_dram_start=self.PERMUTE_TEMP_DRAM)
            # 12x flash attention (no causal mask, no GQA)
            for h in range(N_HEADS):
                self.flash_attention_core(head_dim=D, seq_len=S,
                    Q_DRAM_ADDR=self.VIS_Q_PERM_DRAM + h * head_stride,
                    K_DRAM_ADDR=self.VIS_K_PERM_DRAM + h * head_stride,
                    V_DRAM_ADDR=self.VIS_V_PERM_DRAM + h * head_stride,
                    OUTPUT_DRAM_ADDR=self.VIS_ATTN_OUT_DRAM + h * head_stride,
                    SCRATCH_DRAM_ADDR=self.VIS_ATTN_SCRATCH_DRAM,
                    IDENTITY_DRAM_ADDR=self.identity_addr)
            # Inverse permute: [12, S, 64] → [S, 768]
            smart_bf16_permute_core(self, dims=inv_permute_dims, permute_indices=permute_indices,
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
        self.layer_norm_core_dram(M=S, N=H, A_DRAM_ADDR=final_vis, OUTPUT_DRAM_ADDR=self.VIS_POST_LN_DRAM,
            GAMMA_DRAM_ADDR=self.vis_post_ln_weight, BETA_DRAM_ADDR=self.vis_post_ln_bias)
        # Pixel shuffle: [1024,768] → [64,12288]
        smart_bf16_permute_core(self, dims=[8, 4, 8, 4, H], permute_indices=[0, 2, 1, 3, 4],
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
                is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=self.connector_scale)

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
                    OUTPUT_DRAM_ADDR=OUT, SCALE_DRAM_ADDR=la[f'{proj}_scale'], data_type=TYPE.IF4, **kw)

        self.start_capture()
        # Prime gpr_seq_len for the d64 PBI RoPE hardware loop. S is known at compile time
        # (prefill is per-S compiled because flash stays legacy/static — see PBI flash
        # back-to-back bug), so this is a static ADD_SET at the top of the captured program.
        # ISA scratch-reg allocation must start past the fixed regs (1..4) so the rope's
        # internal alloc_isa_reg() (tmp_reg/t_reg) does not clobber gpr_seq_len.
        self._isa_reg_counter = max(self._cfg["fixed_isa_regs"].values()) + 1  # = 5
        self.generate_instruction_add_set(self.gpr_seq_len, S)
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
                self.rope_hf_core_dram(M=S, N=D, input_dram_addr=q_head, output_dram_addr=q_head,
                    cos_dram_addr=self.ROPE_PACKED_DRAM, sin_dram_addr=self.ROPE_PACKED_DRAM + D * bpe,
                    gpr_M_reg=self.gpr_seq_len)
            # Permute K [S,320]→KV cache [5,S,64] + RoPE per head
            for h in range(self.NUM_KV_HEADS):
                k_cache = self.LAYER0_K_DRAM + layer_idx * self.KV_LAYER_STRIDE + h * self.KV_HEAD_STRIDE
                self.accelerator_memory_to_sram(self.LAYER0_K_PROJ_DRAM + h * D * bpe, 0x00000, S * D,stride_bytes_per_chunk=D * bpe, stride_jump_bytes=KV * bpe)
                self.sram_to_accelerator_memory(0x00000, k_cache, S * D)
                self.rope_hf_core_dram(M=S, N=D, input_dram_addr=k_cache, output_dram_addr=k_cache,
                    cos_dram_addr=self.ROPE_PACKED_DRAM, sin_dram_addr=self.ROPE_PACKED_DRAM + D * bpe,
                    gpr_M_reg=self.gpr_seq_len)
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
                SCALE_DRAM_ADDR=self.lm_head_scale, data_type=TYPE.IF4)
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

    def compile_prefill_v2(self, halt_after_layer: int = None, debug_dump: bool = False) -> None:
        """Seq-len-agnostic gemma3-style prefill. Compiled ONCE for PREFILL_MAX; the runtime length
        is read by the bucket-dispatcher flash via gpr_bucket_idx. Permutes/matmuls/norms run static
        at PREFILL_MAX (shorter prompts padded with epsilon, masked by the block-diagonal bias);
        only the d64 RoPE (gpr) and the O(seq^2) flash (bucket) are runtime-sized.

        Per kv-group GQA into one flash: gather the 3 q-heads token-major [PM,G,D]; rope it (M=PM*G,
        token-major-duplicated table); rope the kv-head into the decoder KV cache then duplicate its
        rows xG token-major; one flash_attention_core(gpr_bucket_idx); un-stack the output back into
        [PM,H]. Ends after the 32 layers with HALT (no final-norm/LM head — those depend on the
        runtime last-token offset and are emitted by run_prefill_v2). Captured in place at a fixed
        program-DRAM address so the flash dispatcher's absolute jumps stay valid; run_prefill_v2's
        preamble primes gpr_bucket_idx and jump_abs-es into this program.
        """
        from user_dma_core import TYPE
        H, D, I = self.HIDDEN_SIZE, self.HEAD_DIM, self.INTERMEDIATE_SIZE
        KV = self.NUM_KV_HEADS * D
        G = self.GROUP_SIZE
        bpe = 2
        PM = self.PREFILL_MAX_SEQ_LEN          # static row count for non-attention ops
        qmax = self.PREFILL_QMAX                # aligned(PM*G) — stacked flash rows
        num_buckets = self.FLASH_NUM_BUCKETS

        def lm_matmul(M, K, N, A, proj, la, OUT, **kw):
            if self.lm_bf16:
                self.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{proj}_weight'],
                    OUTPUT_DRAM_ADDR=OUT, **kw)
            else:
                self.quantized_matmat_core(M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{proj}_data'],
                    OUTPUT_DRAM_ADDR=OUT, SCALE_DRAM_ADDR=la[f'{proj}_scale'], data_type=TYPE.IF4, **kw)

        def strided_copy(src, src_jump, dst, dst_jump, rows, width):
            # static [rows, width] copy: strided gather -> contiguous SRAM -> strided scatter
            self.accelerator_memory_to_sram(src, 0x00000, rows * width,
                stride_bytes_per_chunk=width * bpe, stride_jump_bytes=src_jump)
            self.sram_to_accelerator_memory(0x00000, dst, rows * width,
                stride_bytes_per_chunk=width * bpe, stride_jump_bytes=dst_jump)

        def duplicate_gqa_rows(src_sram_addr, dst_dram_addr):
            # token-major duplication: dst row t*G+g = src row t (kv head broadcast across the group)
            row_bytes = D * bpe
            row_uram_words = row_bytes // (UE_VECTOR_SIZE * bpe)
            _, src_uram_addr = self.sram_address_to_uram_address(src_sram_addr)
            ptr = self.alloc_inst_ptr()
            self.generate_instruction_pbi_init(dram_shared_addr=dst_dram_addr, dma_length=row_bytes,
                output_size=0, uram_length=0, uram_a_start_addr=src_uram_addr,
                uram_b_start_addr=src_uram_addr, uram_wb_addr=0, uram_dst_addr=0,
                fmax_context_addr=0, inst_pointer_idx=ptr)
            self.loop_start(loop_cnt=PM)
            self.loop_start(G)
            self.sram_to_accelerator_memory(sram_address=0, accelerator_dram_address=row_bytes,
                element_size=D, inst_pointer_idx=ptr, memcpy_length_bytes=0)
            self.loop_end()
            self.generate_instruction_pbi_inc(dram_shared_addr=0, dma_length=0, output_size=0,
                uram_length=0, uram_a_start_addr=row_uram_words, uram_b_start_addr=row_uram_words,
                uram_wb_addr=0, uram_dst_addr=0, fmax_context_addr=0, inst_pointer_idx=ptr)
            self.loop_end()
            self.release_inst_ptr(ptr)

        # Debug: per-layer hidden-state dump [NUM_LAYERS, PM, H] so one run captures every layer's
        # output for NaN/cosine comparison against a CPU reference (localizes the first bad layer).
        if debug_dump:
            self.DBG_LAYER_DUMP = self.allocate_tensor_dram(self.NUM_LAYERS * PM * H * bpe)
            print(f"    [dbg] layer dump @0x{self.DBG_LAYER_DUMP:X} ({self.NUM_LAYERS}x{PM}x{H})")
        # Scratch ISA regs start past the fixed gpr regs (1..6) so rope/flash scratch don't clobber them.
        self._isa_reg_counter = max(self._cfg["fixed_isa_regs"].values()) + 1
        self.start_capture()
        prefill_addr = self.get_program_dram_addr()
        # Constant runtime regs for the d64 RoPE loops (PM is compile-time fixed).
        self.generate_instruction_add_set(self.gpr_seq_len, PM)
        self.generate_instruction_add_set(self.gpr_q_seq_len, PM * G)
        for layer_idx in range(self.NUM_LAYERS):
            la = self.lm_layer_addrs[layer_idx]
            h_in  = self.LAYER0_INPUT_DRAM if layer_idx % 2 == 0 else self.LAYER0_OUTPUT_DRAM
            h_out = self.LAYER0_OUTPUT_DRAM if layer_idx % 2 == 0 else self.LAYER0_INPUT_DRAM
            # Input layernorm (fused with previous layer's MLP residual for layers 1+)
            if layer_idx == 0:
                self.rms_norm_core_dram(M=PM, N=H, A_DRAM_ADDR=h_in,
                    OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=la['ln1_gamma'])
            else:
                rms_norm_core_dram_post_add(self, M=PM, N=H,
                    A_DRAM_ADDR=self.LAYER0_RESIDUAL_DRAM, B_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                    ADDOUTPUT_DRAM_ADDR=h_in, NORMOUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                    GAMMA_DRAM_ADDR=la['ln1_gamma'])
            # Q/K/V projections
            lm_matmul(PM, H, H,  self.LAYER0_PRE_NORM_DRAM, 'q', la, self.LAYER0_Q_DRAM)
            lm_matmul(PM, H, KV, self.LAYER0_PRE_NORM_DRAM, 'k', la, self.LAYER0_K_PROJ_DRAM)
            lm_matmul(PM, H, KV, self.LAYER0_PRE_NORM_DRAM, 'v', la, self.LAYER0_V_PROJ_DRAM)
            # K per kv-head -> decoder KV cache (contiguous [PM,D]) + RoPE
            for h in range(self.NUM_KV_HEADS):
                k_cache = self.LAYER0_K_DRAM + layer_idx * self.KV_LAYER_STRIDE + h * self.KV_HEAD_STRIDE
                strided_copy(self.LAYER0_K_PROJ_DRAM + h * D * bpe, KV * bpe, k_cache, D * bpe, PM, D)
                self.rope_hf_core_dram(M=PM, N=D, input_dram_addr=k_cache, output_dram_addr=k_cache,
                    cos_dram_addr=self.ROPE_PACKED_DRAM, sin_dram_addr=self.ROPE_PACKED_DRAM + D * bpe,
                    gpr_M_reg=self.gpr_seq_len)
            # V per kv-head -> decoder KV cache (no RoPE)
            for h in range(self.NUM_KV_HEADS):
                v_cache = self.LAYER0_V_DRAM + layer_idx * self.KV_LAYER_STRIDE + h * self.KV_HEAD_STRIDE
                strided_copy(self.LAYER0_V_PROJ_DRAM + h * D * bpe, KV * bpe, v_cache, D * bpe, PM, D)
            # Per kv-group GQA: stack Q token-major, rope, duplicate K/V, one bucket flash, un-stack
            for kv_b in range(self.NUM_KV_HEADS):
                k_cache = self.LAYER0_K_DRAM + layer_idx * self.KV_LAYER_STRIDE + kv_b * self.KV_HEAD_STRIDE
                v_cache = self.LAYER0_V_DRAM + layer_idx * self.KV_LAYER_STRIDE + kv_b * self.KV_HEAD_STRIDE
                # gather the group's 3 q-heads into token-major FLASH_Q (row t*G+g = q-head kv_b*G+g)
                for g in range(G):
                    strided_copy(self.LAYER0_Q_DRAM + (kv_b * G + g) * D * bpe, H * bpe,
                                 self.FLASH_Q_DRAM + g * D * bpe, G * D * bpe, PM, D)
                # rope stacked Q over M=PM*G using the token-major-duplicated table
                self.rope_hf_core_dram(M=PM * G, N=D, input_dram_addr=self.FLASH_Q_DRAM,
                    output_dram_addr=self.FLASH_Q_DRAM, cos_dram_addr=self.ROPE_PACKED_GQA_DRAM,
                    sin_dram_addr=self.ROPE_PACKED_GQA_DRAM + D * bpe, gpr_M_reg=self.gpr_q_seq_len)
                # duplicate K/V rows token-major into the flash buffers
                self.accelerator_memory_to_sram(k_cache, 0x10000, PM * D)
                duplicate_gqa_rows(0x10000, self.FLASH_K_DRAM)
                self.accelerator_memory_to_sram(v_cache, 0x20000, PM * D)
                duplicate_gqa_rows(0x20000, self.FLASH_V_DRAM)
                # one bucket-dispatcher flash for the whole group
                self.flash_attention_core(head_dim=D, seq_len=qmax,
                    Q_DRAM_ADDR=self.FLASH_Q_DRAM, K_DRAM_ADDR=self.FLASH_K_DRAM,
                    V_DRAM_ADDR=self.FLASH_V_DRAM, OUTPUT_DRAM_ADDR=self.FLASH_OUT_DRAM,
                    SCRATCH_DRAM_ADDR=self.FLASH_SCRATCH_DRAM, ATTN_P_DRAM_ADDR=self.FLASH_ATTN_P_DRAM,
                    IDENTITY_DRAM_ADDR=self.identity_addr, BIAS_DRAM_ADDR=self.FLASH_BIAS_DRAM,
                    gpr_bucket_idx=self.gpr_bucket_idx, num_buckets=num_buckets)
                # un-stack FLASH_OUT [PM,G,D] -> ATTN_RESULT [PM,H] at this group's head columns
                for g in range(G):
                    strided_copy(self.FLASH_OUT_DRAM + g * D * bpe, G * D * bpe,
                                 self.LAYER0_ATTN_RESULT_DRAM + (kv_b * G + g) * D * bpe, H * bpe, PM, D)
            # O projection + residual + RMS norm
            lm_matmul(PM, H, H, self.LAYER0_ATTN_RESULT_DRAM, 'o', la, self.LAYER0_O_PROJ_DRAM)
            rms_norm_core_dram_post_add(self, M=PM, N=H, A_DRAM_ADDR=h_in, B_DRAM_ADDR=self.LAYER0_O_PROJ_DRAM,
                ADDOUTPUT_DRAM_ADDR=self.LAYER0_RESIDUAL_DRAM, NORMOUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                GAMMA_DRAM_ADDR=la['ln2_gamma'])
            # MLP: gate+SiLU, up, gate*up, down
            lm_matmul(PM, H, I, self.LAYER0_PRE_NORM_DRAM, 'gate', la, self.LAYER0_MLP_GATE_DRAM, silu_enable=True)
            lm_matmul(PM, H, I, self.LAYER0_PRE_NORM_DRAM, 'up', la, self.LAYER0_MLP_UP_DRAM)
            eltwise_mul_core_dram(self, size=PM * I,
                A_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM, B_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                OUTPUT_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM)
            lm_matmul(PM, I, H, self.LAYER0_MLP_MULT_DRAM, 'down', la, self.LAYER0_MLP_DOWN_DRAM)
            # Debug: dump this layer's output (residual + MLP) to its own slot; does not disturb the
            # fused path (next layer's input norm recomputes from RESIDUAL + MLP_DOWN).
            if debug_dump:
                eltwise_add_core_dram(self, size=PM * H,
                    A_DRAM_ADDR=self.LAYER0_RESIDUAL_DRAM, B_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                    OUTPUT_DRAM_ADDR=self.DBG_LAYER_DUMP + layer_idx * PM * H * bpe)
            # MLP residual: materialize the layer output (normally fused into the next layer's input
            # norm; done explicitly for the last layer, or the bisection halt layer, so it can be read).
            if layer_idx == self.NUM_LAYERS - 1 or layer_idx == halt_after_layer:
                eltwise_add_core_dram(self, size=PM * H,
                    A_DRAM_ADDR=self.LAYER0_RESIDUAL_DRAM, B_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                    OUTPUT_DRAM_ADDR=h_out)
            if halt_after_layer is not None and layer_idx == halt_after_layer:
                # Bisection: stop after this layer; h_out holds its output hidden state.
                self._prefill_v2_layer_out_buf = h_out
                break
        self.generate_instruction_halt()
        self.stop_capture()
        self.write_captured_instructions_to_dram(prefill_addr)
        size_bytes = self.get_capture_instruction_size_bytes()
        # Cache the full program to bin (gemma3 convention) so later runs load_prefill_v2 in ~1s
        # instead of recompiling. Only the canonical program (no debug dump / no early halt).
        if halt_after_layer is None and not debug_dump:
            raw = bytearray()
            for inst in self.capture_buffer:
                raw.extend(inst.get_bytes())
            bin_path = os.path.join(self.script_dir, "smolvlm2_bin", "prefill_v2_program.bin")
            with open(bin_path, "wb") as f:
                f.write(bytes(raw))
        self.allocate_program_dram(size_bytes)
        self.clear_capture_buffer()
        self._prefill_v2_addr = prefill_addr
        # Reserve preamble (embed/merge + reg prime + jump) and postamble (final-norm + LM head)
        # scratch regions; both are captured fresh per run.
        self._prefill_v2_preamble_addr = self.get_program_dram_addr()
        self.allocate_program_dram(PM * 2 * 32 + IMAGE_SEQ_LEN * 2 * 32 + 256)
        self._prefill_v2_postamble_addr = self.get_program_dram_addr()
        self.allocate_program_dram(4096)
        # Final hidden lands in the buffer matching the layer ping-pong parity.
        self._prefill_v2_final_buf = self.LAYER0_INPUT_DRAM if self.NUM_LAYERS % 2 == 0 else self.LAYER0_OUTPUT_DRAM
        print(f"    Prefill v2 compiled @0x{prefill_addr:X}: {size_bytes} bytes, "
              f"{num_buckets} buckets, PM={PM}, qmax={qmax}")

    def load_prefill_v2(self) -> None:
        """Load the cached seq-len-agnostic prefill program. Replays compile_prefill_v2's exact
        program-DRAM allocations so the body lands at the same address its flash absolute-jumps
        were baked against (must be called at the same point in the alloc sequence: after the
        decoder is loaded)."""
        PM = self.PREFILL_MAX_SEQ_LEN
        bin_path = os.path.join(self.script_dir, "smolvlm2_bin", "prefill_v2_program.bin")
        with open(bin_path, "rb") as f:
            raw = f.read()
        prefill_addr = self.get_program_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, prefill_addr, raw, len(raw))
        self.allocate_program_dram(len(raw))
        self._prefill_v2_addr = prefill_addr
        self._prefill_v2_preamble_addr = self.get_program_dram_addr()
        self.allocate_program_dram(PM * 2 * 32 + IMAGE_SEQ_LEN * 2 * 32 + 256)
        self._prefill_v2_postamble_addr = self.get_program_dram_addr()
        self.allocate_program_dram(4096)
        self._prefill_v2_final_buf = self.LAYER0_INPUT_DRAM if self.NUM_LAYERS % 2 == 0 else self.LAYER0_OUTPUT_DRAM
        print(f"    Loaded prefill v2 @0x{prefill_addr:X}: {len(raw)} bytes")

    def run_prefill_v2(self, token_ids, has_image: bool = False, total_flops: int = None,
                       skip_postamble: bool = False) -> int:
        """Run the seq-len-agnostic prefill: host prep (epsilon pad + block bias), a preamble that
        does on-device embed/merge + primes gpr_bucket_idx + jumps into the cached prefill, then a
        postamble that does final-norm + LM head on the runtime last token. Returns argmax.

        skip_postamble=True runs only the preamble->prefill (for bisection: read intermediate
        buffers afterward) and returns None."""
        from user_dma_core import TYPE
        seq_len = len(token_ids)
        self.seq_len = seq_len
        H, D, G, bpe = self.HIDDEN_SIZE, self.HEAD_DIM, self.GROUP_SIZE, 2
        PM = self.PREFILL_MAX_SEQ_LEN
        assert seq_len <= PM, f"prompt {seq_len} exceeds PREFILL_MAX_SEQ_LEN={PM}"
        qmax = self.PREFILL_QMAX
        qS = seq_len * G
        aligned_q = ((qS + 63) // 64) * 64
        bucket_idx = aligned_q // UE_VECTOR_SIZE
        embed_row_bytes = H * bpe

        # 1. epsilon-fill INPUT padding rows [seq_len:PM] (non-attention ops run static PM rows)
        if PM > seq_len:
            epsilon = torch.full(((PM - seq_len) * H,), 1e-6, dtype=torch.bfloat16)
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM + seq_len * embed_row_bytes, epsilon)
        # 2. fill flash Q/K/V tails finite so masked-out padded rows can't produce NaN in softmax
        eps_flash = torch.full((qmax * D,), 1e-6, dtype=torch.bfloat16)
        for buf in (self.FLASH_Q_DRAM, self.FLASH_K_DRAM, self.FLASH_V_DRAM):
            self.dma_to_accelerator_memory(buf, eps_flash)
        # 3. block-diagonal causal bias [aligned_q, aligned_q]: allow (t,g)->(t',g') iff g==g' & t'<=t
        rq = torch.arange(aligned_q).unsqueeze(1)
        rk = torch.arange(aligned_q).unsqueeze(0)
        allow = (rq < qS) & (rk < qS) & (rq % G == rk % G) & (rk // G <= rq // G)
        bias = torch.where(allow, torch.tensor(0.0), torch.tensor(-1e38)).to(torch.bfloat16)
        bias[torch.arange(aligned_q) >= qS, 0] = 0.0  # padded query rows attend row 0 (discarded)
        self.dma_to_accelerator_memory(self.FLASH_BIAS_DRAM, bias)

        # 4. Preamble: on-device embed gather + vision merge + prime gpr_bucket_idx + jump into prefill
        self.clear_inst_id()
        self.start_capture()
        for t in range(seq_len):
            src = self.embed_addr + token_ids[t] * embed_row_bytes
            self.accelerator_memory_to_sram(src, 0x00000, H)
            self.sram_to_accelerator_memory(0x00000, self.LAYER0_INPUT_DRAM + t * embed_row_bytes, H)
        if has_image:
            img_positions = [i for i, t in enumerate(token_ids) if t == IMAGE_TOKEN_ID]
            if len(img_positions) > 0:
                assert len(img_positions) == IMAGE_SEQ_LEN, \
                    f"Expected {IMAGE_SEQ_LEN} image tokens, got {len(img_positions)}"
                for i, pos in enumerate(img_positions):
                    self.accelerator_memory_to_sram(self.VIS_CONNECTOR_DRAM + i * embed_row_bytes, 0x00000, H)
                    self.sram_to_accelerator_memory(0x00000, self.LAYER0_INPUT_DRAM + pos * embed_row_bytes, H)
        self.generate_instruction_add_set(self.gpr_bucket_idx, bucket_idx)
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(self._prefill_v2_addr))
        self.stop_capture()
        self.write_captured_instructions_to_dram(self._prefill_v2_preamble_addr)
        self.clear_capture_buffer()
        self.start_execute_from_dram(self._prefill_v2_preamble_addr)
        self.wait_queue(180.0)
        if skip_postamble:
            return None  # bisection: caller reads intermediate buffers

        # 5. Postamble: final norm over ALL tokens (identical to the proven old prefill) + LM head on
        # the last real token. (RMSNorm is per-row, so M=S then index == M=1 on the last row; using
        # M=S removes the M=1 path as a variable.)
        last_off = (seq_len - 1) * H * bpe
        self.clear_inst_id()
        self.start_capture()
        self.rms_norm_core_dram(M=seq_len, N=H, A_DRAM_ADDR=self._prefill_v2_final_buf,
            OUTPUT_DRAM_ADDR=self.FINAL_NORM_DRAM, GAMMA_DRAM_ADDR=self.final_norm_addr)
        last_norm = self.FINAL_NORM_DRAM + last_off
        if self.lm_bf16:
            self.matmat_mul_core(M=1, K=H, N=self.VOCAB_SIZE, A_DRAM_ADDR=last_norm,
                B_DRAM_ADDR=self.lm_head_weight, OUTPUT_DRAM_ADDR=self.LOGITS_DRAM)
        else:
            self.quantized_matmat_core(M=1, K=H, N=self.VOCAB_SIZE, A_DRAM_ADDR=last_norm,
                B_DRAM_ADDR=self.lm_head_data, OUTPUT_DRAM_ADDR=self.LOGITS_DRAM,
                SCALE_DRAM_ADDR=self.lm_head_scale, data_type=TYPE.IF4)
        self.generate_instruction_halt()
        self.stop_capture()
        self.write_captured_instructions_to_dram(self._prefill_v2_postamble_addr)
        self.clear_capture_buffer()
        self.start_execute_from_dram(self._prefill_v2_postamble_addr)
        self.wait_queue(30.0)
        if total_flops is not None:
            self._last_hw_gflops, _ = self.report_flop_rate_gflops(total_flops)
            self._last_total_flops = total_flops
        else:
            self._last_hw_gflops = None
            self._last_total_flops = None
        return self.get_arg_max_index()

    def compile_decoder(self) -> None:
        """Compile single decoder program with runtime kv_len dispatch via gpr_bucket_idx."""
        from user_dma_core import TYPE
        H, D, I = self.HIDDEN_SIZE, self.HEAD_DIM, self.INTERMEDIATE_SIZE
        bpe = 2
        G = self.GROUP_SIZE
        num_buckets = (self.max_seq_len + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE

        def lm_matmul(M, K, N, A, proj, la, OUT, **kw):
            if self.lm_bf16:
                self.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{proj}_weight'],
                    OUTPUT_DRAM_ADDR=OUT, **kw)
            else:
                self.quantized_matmat_core(M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{proj}_data'],
                    OUTPUT_DRAM_ADDR=OUT, SCALE_DRAM_ADDR=la[f'{proj}_scale'], data_type=TYPE.IF4, **kw)

        # Guarantee counter starts at 1 regardless of prior compilations.
        self.reset_isa_reg_counter()
        # Reserve fixed ISA registers (1-6) so alloc_isa_reg() inside
        # decoder_group_attention_core_pbi / matmat_mul_core never clobbers
        # V_CACHE_SIZE_REG(1), ROPE_SIZE_REG(2), TMP_REG(3),
        # gpr_seq_len(4), gpr_q_seq_len(5), gpr_bucket_idx(6) at runtime.
        _NUM_FIXED_REGS = 6
        for _ in range(_NUM_FIXED_REGS):
            self.alloc_isa_reg()

        global _SILENT_MODE
        _SILENT_MODE = True
        self.clear_inst_id()
        self.start_capture()

        for layer_idx in range(self.NUM_LAYERS):
            la = self.lm_layer_addrs[layer_idx]
            h_in  = self.LAYER0_INPUT_DRAM if layer_idx % 2 == 0 else self.LAYER0_OUTPUT_DRAM
            h_out = self.LAYER0_OUTPUT_DRAM if layer_idx % 2 == 0 else self.LAYER0_INPUT_DRAM

            # RMS norm (input_layernorm)
            self.rms_norm_core_dram(M=1, N=H, A_DRAM_ADDR=h_in,
                OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=la['ln1_gamma'])
            # Q/K/V projections
            lm_matmul(1, H, H,                     self.LAYER0_PRE_NORM_DRAM, 'q', la, self.LAYER0_Q_DRAM)
            lm_matmul(1, H, self.NUM_KV_HEADS * D, self.LAYER0_PRE_NORM_DRAM, 'k', la, self.LAYER0_K_PROJ_DRAM)
            lm_matmul(1, H, self.NUM_KV_HEADS * D, self.LAYER0_PRE_NORM_DRAM, 'v', la, self.LAYER0_V_PROJ_DRAM)

            # Compute position-based register offsets once per layer
            # V_CACHE_SIZE_REG = gpr_seq_len * k_size  (byte offset into KV cache)
            # ROPE_SIZE_REG    = gpr_seq_len * D * bpe (byte offset into cos/sin tables)
            self.generate_instruction_reg_mul_imm(self.V_CACHE_SIZE_REG, self.gpr_seq_len, ue_35bit_addr_shifter(self.k_size))
            self.generate_instruction_reg_mul_imm(self.ROPE_SIZE_REG, self.gpr_seq_len, ue_35bit_addr_shifter(D * bpe))

            # Write V to KV cache at current position
            for h in range(self.NUM_KV_HEADS):
                v_base = self.LAYER0_V_DRAM + layer_idx * self.KV_LAYER_STRIDE + h * self.KV_HEAD_STRIDE
                self.accelerator_memory_to_sram(self.LAYER0_V_PROJ_DRAM + h * D * bpe, 0x10000, D)
                self.generate_instruction_add_imm(self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter(v_base), self.TMP_REG)
                self.sram_to_accelerator_memory(0x10000, 0, D, general_reg_src=self.TMP_REG)

            # RoPE K: write RoPE'd result directly to K cache via output_addr_inc_reg
            for h in range(self.NUM_KV_HEADS):
                k_base = self.LAYER0_K_DRAM + layer_idx * self.KV_LAYER_STRIDE + h * self.KV_HEAD_STRIDE
                self.rope_hf_core(N=D,
                    input_dram_addr=self.LAYER0_K_PROJ_DRAM + h * D * bpe,
                    output_dram_addr=k_base,
                    cos_dram_addr=self.ROPE_COS_DRAM, sin_dram_addr=self.ROPE_SIN_DRAM,
                    rope_size_reg=self.ROPE_SIZE_REG,
                    output_addr_inc_reg=self.V_CACHE_SIZE_REG,
                    tmp_reg=self.TMP_REG)

            # RoPE Q: output to Q_PERM_DRAM (attention reads from there)
            for h in range(self.NUM_HEADS):
                self.rope_hf_core(N=D,
                    input_dram_addr=self.LAYER0_Q_DRAM + h * D * bpe,
                    output_dram_addr=self.LAYER0_Q_PERM_DRAM + h * D * bpe,
                    cos_dram_addr=self.ROPE_COS_DRAM, sin_dram_addr=self.ROPE_SIN_DRAM,
                    rope_size_reg=self.ROPE_SIZE_REG,
                    tmp_reg=self.TMP_REG)

            # GQA decode attention: NUM_KV_HEADS groups × GROUP_SIZE Q heads each
            for kv_b in range(self.NUM_KV_HEADS):
                q_start = kv_b * G * D * bpe
                self.decoder_group_attention_core_pbi(
                    group_size=G,
                    head_dim=D,
                    Q_DRAM_ADDR=self.LAYER0_Q_PERM_DRAM + q_start,
                    K_DRAM_ADDR=self.LAYER0_K_DRAM + layer_idx * self.KV_LAYER_STRIDE + kv_b * self.KV_HEAD_STRIDE,
                    V_DRAM_ADDR=self.LAYER0_V_DRAM + layer_idx * self.KV_LAYER_STRIDE + kv_b * self.KV_HEAD_STRIDE,
                    OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_OUT_DRAM + q_start,
                    SCRATCH_DRAM_ADDR=self.LAYER0_ATTN_SCRATCH_DRAM,
                    BIAS_DRAM_ADDR=self.DECODE_BIAS_DRAM,
                    IDENTITY_DRAM_ADDR=self.identity_addr,
                    gpr_bucket_idx=self.gpr_bucket_idx,
                    num_buckets=num_buckets,
                )

            # O projection
            lm_matmul(1, H, H, self.LAYER0_ATTN_OUT_DRAM, 'o', la, self.LAYER0_O_PROJ_DRAM)
            # Post-attn residual + RMS norm
            rms_norm_core_dram_post_add(self, M=1, N=H,
                A_DRAM_ADDR=h_in, B_DRAM_ADDR=self.LAYER0_O_PROJ_DRAM,
                ADDOUTPUT_DRAM_ADDR=self.LAYER0_RESIDUAL_DRAM,
                NORMOUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                GAMMA_DRAM_ADDR=la['ln2_gamma'])
            # MLP
            lm_matmul(1, H, I, self.LAYER0_PRE_NORM_DRAM, 'gate', la, self.LAYER0_MLP_GATE_DRAM, silu_enable=True)
            lm_matmul(1, H, I, self.LAYER0_PRE_NORM_DRAM, 'up',   la, self.LAYER0_MLP_UP_DRAM)
            self.accelerator_memory_to_sram(self.LAYER0_MLP_GATE_DRAM, 0x10000, I)
            self.accelerator_memory_to_sram(self.LAYER0_MLP_UP_DRAM,   0x90000, I)
            self.eltwise_mul_core(0x10000, 0x90000, 0x10000, I)
            self.sram_to_accelerator_memory(0x10000, self.LAYER0_MLP_MULT_DRAM, I)
            lm_matmul(1, I, H, self.LAYER0_MLP_MULT_DRAM, 'down', la, self.LAYER0_MLP_DOWN_DRAM)
            # MLP residual
            self.accelerator_memory_to_sram(self.LAYER0_RESIDUAL_DRAM,  0x10000, H)
            self.accelerator_memory_to_sram(self.LAYER0_MLP_DOWN_DRAM,  0x90000, H)
            self.eltwise_add_core(0x10000, 0x90000, 0x10000, H)
            self.sram_to_accelerator_memory(0x10000, h_out, H)

        # Final norm + LM head
        final_buf = self.LAYER0_INPUT_DRAM if self.NUM_LAYERS % 2 == 0 else self.LAYER0_OUTPUT_DRAM
        self.rms_norm_core_dram(M=1, N=H, A_DRAM_ADDR=final_buf,
            OUTPUT_DRAM_ADDR=self.FINAL_NORM_DRAM, GAMMA_DRAM_ADDR=self.final_norm_addr)
        if self.lm_bf16:
            self.matmat_mul_core(M=1, K=H, N=self.VOCAB_SIZE, A_DRAM_ADDR=self.FINAL_NORM_DRAM,
                B_DRAM_ADDR=self.lm_head_weight, OUTPUT_DRAM_ADDR=self.LOGITS_DRAM)
        else:
            self.quantized_matmat_core(M=1, K=H, N=self.VOCAB_SIZE, A_DRAM_ADDR=self.FINAL_NORM_DRAM,
                B_DRAM_ADDR=self.lm_head_data, OUTPUT_DRAM_ADDR=self.LOGITS_DRAM,
                SCALE_DRAM_ADDR=self.lm_head_scale, data_type=TYPE.IF4)

        # Advance token position on-device so next decode step writes to the right KV slot
        self.generate_instruction_add_inc(self.gpr_seq_len)
        self.generate_instruction_halt()
        self.stop_capture()
        for _ in range(_NUM_FIXED_REGS):
            self.release_isa_reg()
        _SILENT_MODE = False

        raw = bytearray()
        for inst in self.capture_buffer:
            raw.extend(inst.get_bytes())
        self.clear_capture_buffer()

        bin_dir = os.path.join(self.script_dir, "smolvlm2_bin")
        bin_path = os.path.join(bin_dir, "decoder_program.bin")
        with open(bin_path, "wb") as f:
            f.write(bytes(raw))
        meta_path = os.path.join(bin_dir, "decoder_program.json")
        with open(meta_path, "w") as f:
            json.dump({"decoder_program_size": len(raw), "num_buckets": num_buckets, "version": 2}, f)

        # Load into program DRAM for immediate use (same session)
        self._decoder_program_addr = self.get_program_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, self._decoder_program_addr, bytes(raw), len(raw))
        self.allocate_program_dram(len(raw))
        self._decoder_preamble_addr = self.get_program_dram_addr()
        self.allocate_program_dram(256)
        self._decoder_num_buckets = num_buckets
        print(f"    Decoder compiled: single program {len(raw)} bytes, {num_buckets} attention buckets → {bin_dir}")

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

    def dump_snapshot(self) -> None:
        """Dump params DRAM + all runtime address metadata to smolvlm2_bin/params.bin + params.json."""
        bin_dir = os.path.join(self.script_dir, "smolvlm2_bin")
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
        addr_attrs = [
            "embed_addr", "ROPE_COS_DRAM", "ROPE_SIN_DRAM", "PERMUTE_PARAMS_DRAM",
            "LAYER0_K_DRAM", "LAYER0_V_DRAM", "LAYER0_INPUT_DRAM", "LAYER0_OUTPUT_DRAM",
            "LAYER0_PRE_NORM_DRAM", "LAYER0_Q_DRAM", "LAYER0_K_PROJ_DRAM", "LAYER0_V_PROJ_DRAM",
            "LAYER0_Q_PERM_DRAM", "LAYER0_ATTN_OUT_DRAM", "LAYER0_ATTN_SCRATCH_DRAM",
            "LAYER0_ATTN_RESULT_DRAM", "LAYER0_O_PROJ_DRAM", "LAYER0_RESIDUAL_DRAM",
            "LAYER0_MLP_GATE_DRAM", "LAYER0_MLP_UP_DRAM", "LAYER0_MLP_MULT_DRAM",
            "LAYER0_MLP_DOWN_DRAM", "FINAL_NORM_DRAM", "LOGITS_DRAM",
            "CAUSAL_MASK_DRAM", "DECODE_BIAS_DRAM",
            "VIS_PIXEL_IN_DRAM", "VIS_PATCH_PERM_DRAM", "VIS_PATCH_PROJ_DRAM",
            "VIS_IO_A_DRAM", "VIS_IO_B_DRAM", "VIS_LN_OUT_DRAM",
            "VIS_Q_DRAM", "VIS_K_DRAM", "VIS_V_DRAM",
            "VIS_Q_PERM_DRAM", "VIS_K_PERM_DRAM", "VIS_V_PERM_DRAM",
            "VIS_ATTN_OUT_DRAM", "VIS_ATTN_SCRATCH_DRAM", "VIS_ATTN_RESULT_DRAM",
            "VIS_O_PROJ_DRAM", "VIS_RESIDUAL_DRAM", "VIS_MLP_INTER_DRAM", "VIS_MLP_OUT_DRAM",
            "VIS_POST_LN_DRAM", "VIS_SHUFFLED_DRAM", "VIS_CONNECTOR_DRAM", "PERMUTE_TEMP_DRAM",
            # Prefill v2 (gemma3-style) RoPE tables + stacked-GQA flash buffers
            "ROPE_PACKED_DRAM", "ROPE_PACKED_GQA_DRAM",
            "FLASH_Q_DRAM", "FLASH_K_DRAM", "FLASH_V_DRAM", "FLASH_OUT_DRAM",
            "FLASH_SCRATCH_DRAM", "FLASH_ATTN_P_DRAM", "FLASH_BIAS_DRAM",
        ]
        meta = {
            "params_size": total,
            "tensor_size": self.get_tensor_dram_usage(),
            "max_seq_len": self.max_seq_len,
            "k_size": self.k_size,
            "bytes_per_element": self.bytes_per_element,
            "KV_HEAD_STRIDE": self.KV_HEAD_STRIDE,
            "KV_LAYER_STRIDE": self.KV_LAYER_STRIDE,
            "vision_bf16": self.vision_bf16,
            "_num_vis_layers": len(self.vis_layer_addrs),
            "PREFILL_QMAX": self.PREFILL_QMAX,
            "FLASH_NUM_BUCKETS": self.FLASH_NUM_BUCKETS,
        }
        for attr in addr_attrs:
            meta[attr] = getattr(self, attr)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        _original_print(f"  Snapshot dumped: {total / 1024**2:.1f} MB → {bin_path}")

    def load_snapshot(self) -> bool:
        """Load params DRAM from snapshot bin + restore all address metadata. Returns True if loaded."""
        bin_dir = os.path.join(self.script_dir, "smolvlm2_bin")
        bin_path = os.path.join(bin_dir, "params.bin")
        meta_path = os.path.join(bin_dir, "params.json")
        if not os.path.exists(bin_path) or not os.path.exists(meta_path):
            return False
        with open(meta_path) as f:
            meta = json.load(f)
        total = meta["params_size"]
        CHUNK = 1 * 1024 * 1024
        with open(bin_path, "rb") as f:
            offset = 0
            while offset < total:
                data = f.read(min(CHUNK, total - offset))
                self.dma_write(DMA_DEVICE_H2C, self._params_dram_base + offset, data, len(data))
                offset += len(data)
        self.allocate_params_dram(total)
        self.allocate_tensor_dram(meta["tensor_size"])
        for key, val in meta.items():
            if key not in ("params_size", "tensor_size"):
                setattr(self, key, val)
        self.bytes_per_element = 2  # always bf16; not in snapshot, needed by attention PBI body
        from transformers import AutoTokenizer
        model_dir = os.path.join(self.script_dir, "smolvlm2_bin", "SmolVLM2-500M-Video-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        kv_zeros = torch.zeros(self.NUM_LAYERS * self.NUM_KV_HEADS * self.max_seq_len * self.HEAD_DIM, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_K_DRAM, kv_zeros)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, kv_zeros)
        _original_print(f"  Snapshot loaded: {total / 1024**2:.1f} MB")
        return True

    def load_decoder(self) -> None:
        """Load pre-compiled decoder program from bin."""
        bin_dir = os.path.join(self.script_dir, "smolvlm2_bin")
        meta_path = os.path.join(bin_dir, "decoder_program.json")
        with open(meta_path) as f:
            meta = json.load(f)
        if "num_buckets" not in meta:
            raise RuntimeError(
                "decoder_program.json is from the old bucket format; "
                "delete decoder_program.bin and decoder_program.json to recompile"
            )
        bin_path = os.path.join(bin_dir, "decoder_program.bin")
        with open(bin_path, "rb") as f:
            raw = f.read()
        self._decoder_program_addr = self.get_program_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, self._decoder_program_addr, raw, len(raw))
        self.allocate_program_dram(len(raw))
        self._decoder_preamble_addr = self.get_program_dram_addr()
        self.allocate_program_dram(256)
        self._decoder_num_buckets = meta["num_buckets"]
        print(f"    Loaded decoder @0x{self._decoder_program_addr:X}: {len(raw)} bytes, {self._decoder_num_buckets} buckets")

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
        self.wait_queue(180.0)  # TEMP: diagnose d64-rope prefill slow-vs-hang (was 30.0)
        if total_flops is not None:
            self._last_hw_gflops, _ = self.report_flop_rate_gflops(total_flops)
            self._last_total_flops = total_flops
        else:
            self._last_hw_gflops = None
            self._last_total_flops = None
        return self.get_arg_max_index()
    def run_decoder(self, token_id: int, max_new_tokens: int = 512) -> list:
        """Auto-regressive decode loop. Returns generated token IDs."""
        global _SILENT_MODE
        generated = []
        H = self.HIDDEN_SIZE
        bpe = 2
        embed_row_bytes = H * bpe
        num_buckets = self._decoder_num_buckets
        decoder_program_addr = self._decoder_program_addr
        preamble_addr = self._decoder_preamble_addr

        while len(generated) < max_new_tokens and self.seq_len < self.max_seq_len:
            _SILENT_MODE = True
            self.seq_len += 1
            aligned_kv = ((self.seq_len + 63) // 64) * 64
            bucket_idx = min(aligned_kv // UE_VECTOR_SIZE, num_buckets)

            # Decode bias: [GROUP_SIZE, max_seq_len] with zeros at valid KV positions
            bias = torch.full((self.GROUP_SIZE, self.max_seq_len), -1e38, dtype=torch.bfloat16)
            bias[:, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.DECODE_BIAS_DRAM, bias)

            # Host-side embed DMA (same as gemma3) — avoids on-device C2H before decoder
            src_addr = self.embed_addr + token_id * embed_row_bytes
            embed_vec = self.dma_from_accelerator_memory(src_addr, torch.Size([H]))
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embed_vec)

            # Preamble: prime gpr_seq_len + gpr_bucket_idx + jump into decoder
            self.clear_inst_id()
            self.start_capture()
            self.generate_instruction_add_set(self.gpr_seq_len, self.seq_len - 1)
            self.generate_instruction_add_set(self.gpr_bucket_idx, bucket_idx)
            self.generate_instruction_jump_abs(ue_35bit_addr_shifter(decoder_program_addr))
            self.stop_capture()
            self.write_captured_instructions_to_dram(preamble_addr)
            self.clear_capture_buffer()

            self.start_execute_from_dram(preamble_addr)
            self.wait_queue(10.0)

            token_id = self.get_arg_max_index()
            # Debug: first 2 steps only
            if len(generated) < 2:
                logits = self.dma_from_accelerator_memory(self.LOGITS_DRAM, torch.Size([self.VOCAB_SIZE]))
                top5 = logits.topk(5)
                _original_print(f"\n[dbg step={len(generated)+1}] seq_len={self.seq_len} gpr_seq_len_set={self.seq_len-1} bucket={bucket_idx} argmax={token_id} top5_ids={top5.indices.tolist()} top5_vals={top5.values.tolist()}")
            generated.append(token_id)
            _SILENT_MODE = False
            if token_id in _SMOLVLM2_CFG["model"]["stop_token_ids"]:
                _original_print(f"\nStop token {token_id} reached.")
                break
            _original_print(self.tokenizer.decode([token_id]), end="", flush=True)
        _SILENT_MODE = False
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
def _hf_reference(model_dir, tokenizer, prompt, image_path, n_greedy=8):
    """HF CPU fp32 reference for --debug. Returns (token_ids, hidden[L+1] x [S,H], logits[S,V],
    image_features[64,H] or None, greedy_tokens[n_greedy]). Image mode runs the full multimodal
    forward so per-layer hidden states match the device's image-mode prefill."""
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.float32,
        device_map=None, attn_implementation="eager").cpu().eval()
    image_features = None
    if image_path:
        from PIL import Image
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        img = Image.open(image_path).convert("RGB")
        # Use the DEVICE token layout (build_input_ids) for both, so sequences align 1:1.
        token_ids = build_input_ids(tokenizer, prompt, has_image=True)
        pv = proc.image_processor(images=[img], return_tensors="pt")
        kw = {"input_ids": torch.tensor([token_ids]),
              "pixel_values": pv["pixel_values"].to(torch.float32)}
        if "pixel_attention_mask" in pv:
            kw["pixel_attention_mask"] = pv["pixel_attention_mask"]
        with torch.no_grad():
            try:
                feat = model.get_image_features(pixel_values=kw["pixel_values"],
                                                pixel_attention_mask=kw.get("pixel_attention_mask"))
                image_features = torch.as_tensor(feat).reshape(-1, model.config.text_config.hidden_size).float()
            except Exception as e:
                print(f"[dbg] get_image_features unavailable ({e}); vision checked by NaN/stats only")
            out = model(output_hidden_states=True, use_cache=False, **kw)
            gen = model.generate(max_new_tokens=n_greedy, do_sample=False, **kw)
    else:
        token_ids = build_input_ids(tokenizer, prompt, has_image=False)
        ids = torch.tensor([token_ids])
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            gen = model.generate(input_ids=ids, max_new_tokens=n_greedy, do_sample=False)
    hidden = [h[0].float() for h in out.hidden_states]
    logits = out.logits[0].float()
    greedy = gen[0][len(token_ids):].tolist()
    del model
    return token_ids, hidden, logits, image_features, greedy


def run_debug(ue, args, script_dir, has_image):
    """End-to-end NaN/cosine localizer (one device session). Verifies, with a HF CPU reference:
    (1) vision encoder, (2) prefill_v2 every layer, (3) first-token logits / postamble, (4) the
    decode loop. Compile chatter is suppressed; a single clear DIAGNOSIS is printed at the end.
    Use --image=None for the clean text-only LM check (no vision confound)."""
    import io, contextlib
    quiet = lambda: contextlib.redirect_stdout(io.StringIO())  # hide compile/run chatter

    model_dir = os.path.join(script_dir, _SMOLVLM2_CFG["paths"]["hf_model_dir"])
    H, bpe = ue.HIDDEN_SIZE, 2
    PM = ue.PREFILL_MAX_SEQ_LEN
    F = {}  # findings

    print("\nLoading HF reference (CPU fp32)…", flush=True)
    with quiet():
        token_ids, hidden, logits, image_features, hf_greedy = _hf_reference(
            model_dir, ue.tokenizer, args.prompt, args.image if has_image else None)
    seq_len = len(token_ids)
    gold = int(logits[-1].argmax())
    print(f"=== DEBUG  tokens={seq_len}  image={has_image}  prompt={args.prompt!r} ===")
    print(f"HF golden first token id={gold} {ue.tokenizer.decode([gold])!r}", flush=True)

    # --- Stage 1: vision encoder ---
    if has_image:
        print("\n--- Stage 1: vision encoder (VIS_CONNECTOR vs HF image features) ---", flush=True)
        with quiet():
            enc_addr = ue.compile_encoder()
            ue.run_encoder(enc_addr, process_image(args.image))
        vis = ue.dbg_dram(ue.VIS_CONNECTOR_DRAM, 64 * H, "VIS_CONNECTOR",
                          ref=image_features if image_features is not None else None, raise_on_nan=False)
        F["vision_cos_excl8"] = vis["cos_excl8"]
        F["vision_nan"] = vis["nan"] + vis["inf"]

    # --- Stage 2: prefill_v2 per-layer hidden vs HF (one run) ---
    print("\n--- Stage 2: prefill_v2 per-layer hidden vs HF (one run) ---", flush=True)
    print("  compiling prefill_v2 (~496K instrs, silent, ~30-90s)…", flush=True)
    with quiet():
        ue.compile_prefill_v2(debug_dump=True)
    print("  running prefill_v2 on device…", flush=True)
    with quiet():
        ue.run_prefill_v2(token_ids, has_image=has_image, skip_postamble=True)
    layer_excl8, layer_nan = [], 0
    for n in range(ue.NUM_LAYERS):
        r = ue.dbg_dram(ue.DBG_LAYER_DUMP + n * PM * H * bpe, seq_len * H, f"L{n:02d}_out",
                        ref=hidden[n + 1][:seq_len], raise_on_nan=False)
        layer_excl8.append(r["cos_excl8"]); layer_nan += r["nan"] + r["inf"]
    # L31 vs HF is raw-vs-final-normed (artifact) -> judge layers on L0..L30 only.
    early = [c for c in layer_excl8[:3] if c is not None]
    midlate = [c for c in layer_excl8[3:ue.NUM_LAYERS - 1] if c is not None]
    F["prefill_early_min"] = min(early) if early else None
    F["prefill_midlate_min"] = min(midlate) if midlate else None
    F["prefill_nan"] = layer_nan

    # --- Stage 3: full prefill -> first-token logits (postamble: final-norm + LM head) ---
    print("\n--- Stage 3: full prefill first token (postamble) ---", flush=True)
    with quiet():
        tok = ue.run_prefill_v2(token_ids, has_image=has_image)
    lg = ue.dbg_dram(ue.LOGITS_DRAM, ue.VOCAB_SIZE, "LOGITS", ref=logits[-1], raise_on_nan=False)
    F["first_tok"] = tok; F["first_tok_match"] = (tok == gold); F["logits_cos"] = lg["cos"]
    # Split postamble: compare the device's NORMED final hidden (FINAL_NORM, M=1 last token) to HF's
    # final-normed last hidden (hidden[-1] is post-final-norm). High cos -> norm/L31 fine, LM head is
    # the bug; low cos -> final-norm or layer-31 output is the bug.
    hf_fn = hidden[-1][seq_len - 1]
    print(f"   (HF final-normed last hidden range min={float(hf_fn.min()):+.2f} max={float(hf_fn.max()):+.2f})")
    fn = ue.dbg_dram(ue.FINAL_NORM_DRAM + (seq_len - 1) * ue.HIDDEN_SIZE * 2, ue.HIDDEN_SIZE,
                     "FINAL_NORM(last)", ref=hf_fn, raise_on_nan=False)
    F["final_norm_cos"] = fn["cos"]

    # --- Stage 4: decode loop vs HF greedy continuation ---
    print("\n--- Stage 4: decode loop vs HF greedy ---", flush=True)
    n_dec = len(hf_greedy)
    try:
        bin_dir = os.path.join(script_dir, "smolvlm2_bin")
        have_dec = os.path.exists(os.path.join(bin_dir, "decoder_program.bin"))
        print(f"  {'loading' if have_dec else 'compiling'} decoder"
              f"{' (~49K embed programs, slow, minutes)' if not have_dec else ''}…", flush=True)
        with quiet():
            (ue.load_decoder if have_dec else ue.compile_decoder)()
            dev_cont = ue.run_decoder(tok, max_new_tokens=n_dec - 1)
        dev_seq = [tok] + list(dev_cont)
        m = 0
        for a, b in zip(dev_seq, hf_greedy):
            if a == b: m += 1
            else: break
        F["decode_match"] = m; F["decode_total"] = n_dec
        F["dev_text"] = ue.tokenizer.decode(dev_seq)
        F["hf_text"] = ue.tokenizer.decode(hf_greedy)
    except Exception as e:
        F["decode_err"] = repr(e)

    # ---------------- DIAGNOSIS ----------------
    def ok(c, th=0.90): return c is not None and c >= th
    print("\n" + "=" * 70)
    print("                        D I A G N O S I S")
    print("=" * 70)
    if has_image:
        print(f"  vision encoder : cos_excl8={F.get('vision_cos_excl8')}  nan/inf={F.get('vision_nan')}")
    print(f"  prefill layers : L0-2 min cos_excl8={F['prefill_early_min']:.3f}  "
          f"L3-30 min={F['prefill_midlate_min']:.3f}  nan/inf={F['prefill_nan']}")
    print(f"  first token    : device={F['first_tok']} {ue.tokenizer.decode([F['first_tok']])!r}  "
          f"golden={gold} {ue.tokenizer.decode([gold])!r}  "
          f"{'MATCH' if F['first_tok_match'] else 'MISMATCH'}  (logits cos={F['logits_cos']:.4f})")
    fnc = F.get("final_norm_cos")
    print(f"  final-norm     : normed last-hidden cos vs HF = {fnc if fnc is None else round(fnc,4)}")
    if "decode_err" in F:
        print(f"  decode loop    : ERROR {F['decode_err']}")
    else:
        print(f"  decode loop    : matched {F['decode_match']}/{F['decode_total']} greedy tokens")
        print(f"     device: {F['dev_text']!r}")
        print(f"     golden: {F['hf_text']!r}")
    print("-" * 70)

    # Per-stage PASS/FAIL — evaluate EVERY stage independently (don't stop at the first failure).
    def line(name, passed, note=""):
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:16s} {note}")
    stages = []
    if has_image:
        v_ok = (F.get("vision_nan", 0) == 0) and ok(F.get("vision_cos_excl8"), 0.85)
        stages.append(("vision encoder", v_ok,
                       f"cos_excl8={F.get('vision_cos_excl8')} nan={F.get('vision_nan')} "
                       "(preprocessing differs; low cos may be benign)"))
    p_ok = (F["prefill_nan"] == 0) and ok(F["prefill_early_min"])
    stages.append(("prefill layers", p_ok,
                   f"L0-2 min cos_excl8={F['prefill_early_min']:.3f} nan={F['prefill_nan']}"))
    fn_ok = ok(fnc, 0.95)
    stages.append(("final-norm/L31", fn_ok, f"normed-hidden cos={fnc if fnc is None else round(fnc,4)}"))
    head_ok = F["first_tok_match"]
    stages.append(("LM head / token", head_ok,
                   f"device={F['first_tok']!r} golden={gold!r} logits_cos={F['logits_cos']:.3f}"))
    dec_ok = ("decode_err" not in F) and (F.get("decode_match", 0) == F.get("decode_total", -1))
    stages.append(("decoder loop", dec_ok,
                   F.get("decode_err", f"matched {F.get('decode_match')}/{F.get('decode_total')}")))
    for name, passed, note in stages:
        line(name, passed, note)
    print("-" * 70)
    failed = [n for n, p, _ in stages if not p]
    if not failed:
        print("  >>> ALL STAGES PASS — output matches HF.")
    else:
        print(f"  >>> FIRST FAILING STAGE: {failed[0].upper()}   (all failures: {', '.join(failed)})")
    print("=" * 70, flush=True)


# =============================================================================
# CPU decode verification (--verify_prefill)
# =============================================================================
def _verify_prefill_cpu(ue, token_ids, hw_first_token, model_dir, image_path, prompt, max_new=30):
    """Use AutoProcessor + model.generate() — the standard HF path — to get ground-truth output."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image

    _original_print("  Loading HF model + processor on CPU...")
    hf = AutoModelForImageTextToText.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map="cpu", attn_implementation="eager").eval()
    proc = AutoProcessor.from_pretrained(model_dir, local_files_only=True)

    has_image = image_path is not None
    if has_image:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        img = Image.open(image_path).convert("RGB")
        prompt_text = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs = proc(text=prompt_text, images=[img], return_tensors="pt")
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        prompt_text = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs = proc(text=prompt_text, return_tensors="pt")
    inputs = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in inputs.items()}

    _original_print(f"  hw first token: {hw_first_token} ({ue.tokenizer.decode([hw_first_token])})")
    _original_print("  CPU generate output: ", end="", flush=True)
    with torch.no_grad():
        out_ids = hf.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    new_ids = out_ids[0, inputs["input_ids"].shape[1]:]
    _original_print(proc.decode(new_ids, skip_special_tokens=True))
    del hf


# =============================================================================
# Main
# =============================================================================
def main():
    import argparse
    from user_dma_core import set_dma_device

    parser = argparse.ArgumentParser(description="SmolVLM2-500M on accelerator")
    parser.add_argument("--gen-weights", action="store_true", help="Generate quantized weight bins from HF weights")
    _d = _SMOLVLM2_CFG["defaults"]
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument("--prompt", type=str, default=_d["prompt"], help="Text prompt")
    parser.add_argument("--image", type=str, default=os.path.join(_root, _d["image"]), help="Path to image file (None for text-only)")
    parser.add_argument("--dev", type=str, default=_d["dev"], help="DMA device name")
    parser.add_argument("--cycle", type=float, default=_d["cycle_ns"], help="Clock cycle time in ns")
    parser.add_argument("--max-seq", type=int, default=_d["max_seq"], help="Max sequence length")
    parser.add_argument("--vision-fp4", action="store_true", help="Use FP4 quantized weights for vision encoder (default: BF16)")
    parser.add_argument("--debug", action="store_true", help="Staged NaN/cosine localizer: verify vision + per-layer prefill_v2 vs HF CPU reference, then exit (no decode)")
    parser.add_argument("--verify_prefill", action="store_true", help="After hardware prefill, run CPU decode using HF model to confirm model output (skips hardware decoder)")
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
    ue = SmolVLM2_UnifiedEngine(script_dir=script_dir, vision_bf16=not args.vision_fp4)
    init_hang_prevention(ue)
    if not ue.load_snapshot():
        ue.weight_init()
        ue.tensor_init(max_seq_len=args.max_seq)
        ue.dump_snapshot()
    has_image = args.image is not None and str(args.image).strip().lower() not in ("none", "")
    token_ids = build_input_ids(ue.tokenizer, args.prompt, has_image=has_image)
    seq_len = len(token_ids)
    _SILENT_MODE = False
    if has_image:
        image_path = os.path.abspath(args.image)
        _original_print(f"Image: {image_path}")
    _original_print(f"Prompt: {args.prompt!r} ({seq_len} tokens, image={'yes' if has_image else 'no'})")
    if args.debug:
        run_debug(ue, args, script_dir, has_image)
        return
    # --- Compile (or load from bin) ---
    _SILENT_MODE = True
    timer = time.perf_counter()
    bin_dir = os.path.join(script_dir, "smolvlm2_bin")
    S = ((seq_len + 63) // 64) * 64
    # All three (encoder + decoder + prefill v2) are cached to bin; recompile only when missing.
    use_bin = (os.path.exists(os.path.join(bin_dir, "decoder_program.bin"))
               and os.path.exists(os.path.join(bin_dir, "prefill_v2_program.bin")))
    import threading
    stop_compile = threading.Event()
    def _compile_progress():
        while not stop_compile.wait(1.0):
            elapsed = time.perf_counter() - timer
            _original_print(f"\r  Compiling ({elapsed:.0f}s)", end="", flush=True)
    if not use_bin:
        t = threading.Thread(target=_compile_progress, daemon=True)
        t.start()
    if use_bin:
        if has_image:
            enc_addr = ue.load_encoder()
        ue.load_decoder()
        # Prefill v2 loaded last so its program-DRAM region sits past encoder/decoder.
        ue.load_prefill_v2()
    else:
        if has_image:
            enc_addr = ue.compile_encoder()
        ue.compile_decoder()
        # Prefill v2 compiled last so its program-DRAM region sits past encoder/decoder.
        ue.compile_prefill_v2()
    stop_compile.set()
    _SILENT_MODE = False
    elapsed = time.perf_counter() - timer
    if use_bin:
        _original_print(f"  Loaded from bin in {elapsed:.2f}s")
    else:
        _original_print(f"\r  Compiling ({elapsed:.0f}s) done")
    # --- Vision encoder ---
    if has_image:
        timer = time.perf_counter()
        pixel_values = process_image(args.image)
        stop_enc = threading.Event()
        def _enc_progress():
            while not stop_enc.wait(1.0):
                _original_print(f"\r  Vision encoder ({time.perf_counter() - timer:.0f}s)", end="", flush=True)
        t_enc = threading.Thread(target=_enc_progress, daemon=True)
        t_enc.start()
        _SILENT_MODE = True
        ue.run_encoder(enc_addr, pixel_values)
        _SILENT_MODE = False
        stop_enc.set()
        t_enc.join()
        enc_time = time.perf_counter() - timer
        _original_print(f"\r  Vision encoder ({enc_time:.0f}s) done")
    # --- Prefill ---
    timer = time.perf_counter()
    padded_seq = ((seq_len + 63) // 64) * 64
    S = padded_seq
    H, D, I = ue.HIDDEN_SIZE, ue.HEAD_DIM, ue.INTERMEDIATE_SIZE
    KVH, QH, G = ue.NUM_KV_HEADS, ue.NUM_HEADS, ue.GROUP_SIZE
    KV = KVH * D
    attn_per_group = (D * S + G * (S * D + 2 * S * D * S + S * S + S * S * 5 + 2 * S * S * D))
    per_layer = (4 * S * H + 2 * S * H * H + 2 * 2 * S * H * KV
        + (QH + KVH) * S * D * 4 + KVH * attn_per_group
        + 2 * S * H * H + 5 * S * H + 2 * (2 * S * H * I) + S * I + 2 * S * I * H + S * H)
    prefill_flops = ue.NUM_LAYERS * per_layer + 4 * S * H + 2 * 1 * H * ue.VOCAB_SIZE
    stop_pf = threading.Event()
    def _pf_progress():
        while not stop_pf.wait(1.0):
            _original_print(f"\r  Prefill ({time.perf_counter() - timer:.0f}s)", end="", flush=True)
    t_pf = threading.Thread(target=_pf_progress, daemon=True)
    t_pf.start()
    _SILENT_MODE = True
    hw_token = ue.run_prefill_v2(token_ids, has_image=has_image, total_flops=prefill_flops)
    _SILENT_MODE = False
    stop_pf.set()
    t_pf.join()
    prefill_time = time.perf_counter() - timer
    _original_print(f"\r  Prefill ({prefill_time:.0f}s) done")
    # Zero out stale KV cache positions left by the static PREFILL_MAX prefill v2 (it writes cache
    # rows [seq_len:PREFILL_MAX] from epsilon-padded input — decode must not read those as real).
    zero_to = ue.PREFILL_MAX_SEQ_LEN
    if zero_to > seq_len:
        stale_size = (zero_to - seq_len) * ue.HEAD_DIM
        stale_zeros = torch.zeros(stale_size, dtype=torch.bfloat16)
        for layer in range(ue.NUM_LAYERS):
            for h in range(ue.NUM_KV_HEADS):
                k_stale = ue.LAYER0_K_DRAM + layer * ue.KV_LAYER_STRIDE + h * ue.KV_HEAD_STRIDE + seq_len * ue.HEAD_DIM * 2
                v_stale = ue.LAYER0_V_DRAM + layer * ue.KV_LAYER_STRIDE + h * ue.KV_HEAD_STRIDE + seq_len * ue.HEAD_DIM * 2
                ue.dma_to_accelerator_memory(k_stale, stale_zeros)
                ue.dma_to_accelerator_memory(v_stale, stale_zeros)
    if args.verify_prefill:
        _original_print(f"\n--- verify_prefill: CPU ground-truth decode ---")
        _verify_prefill_cpu(ue, token_ids, hw_token, model_dir,
                            image_path=args.image if has_image else None,
                            prompt=args.prompt, max_new=30)
        _original_print("verify_prefill done — exiting (no hardware decode).")
        return

    # --- Decode (on-device embed fused with decoder, single dispatch per token) ---
    max_new = args.max_seq - seq_len
    _original_print(f"\n--- Starting decoder ---")
    _original_print(ue.tokenizer.decode([hw_token]), end="", flush=True)
    decode_timer = time.perf_counter()
    hw_tokens = ue.run_decoder(hw_token, max_new_tokens=max_new)
    decode_time = time.perf_counter() - decode_timer
    total_time = prefill_time + decode_time
    n_generated = len(hw_tokens)
    _original_print(f"\nDecoder done in {total_time:.2f} seconds, total {n_generated} tokens.")
    _original_print("SmolVLM2 test ends.")

if __name__ == "__main__":
    main()
