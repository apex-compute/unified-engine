#!/usr/bin/env python3
"""Qwen2.5-VL-3B on accelerator. LM and vision precisions are
config-driven via qwen2.5_vl_3b_config.json::precision.{lm,vision}
(values: int4 / fp4 / if4). Defaults: lm=if4 (eval-winning text codec
per src/models/qwen2.5_VL_3b/compare/summary.md), vision=int4 (legacy
released codec, byte-identical to the prior Q4_64 path). All
quantization goes through src/template/quant_schemas.py."""

import json
import math
import mmap
import os
import sys

import numpy as np
import torch
from huggingface_hub import snapshot_download
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

import user_dma_core
from user_dma_core import DMA_DEVICE_H2C, TYPE, UE_FMAX_CONTEXT_SIZE, UE_VECTOR_SIZE, UE_ARGMAX1_INDEX, URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS, set_dma_device
from user_dma_core import UnifiedEngine, DRAM_INSTRUCTION_ADDR, INSTRUCTION_REG_REWRITE, MEMCPY_TYPE
from user_dma_core import ue_35bit_addr_shifter
import quant_schemas

import builtins

_original_print = builtins.print
_SILENT_MODE = False

def quiet_print(*args, **kwargs):
    if _SILENT_MODE:
        return
    _original_print(*args, **kwargs)

builtins.print = quiet_print

def _parse_offset(val) -> int:
    """Parse offset/size from JSON: int or hex string like '0x24000000'."""
    if isinstance(val, str):
        return int(val, 0)
    return int(val)

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
    cache = {}
    with open(bin_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for name, meta in manifest.items():
            cache[name] = np.frombuffer(mm[meta['offset']:meta['offset'] + meta['size']], dtype=np.uint8).copy()
        mm.close()
    return cache

def store_identity_matrix(ue):
    """Store identity matrix in DRAM once. Returns DRAM address."""
    bpe = 2
    size = UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe
    addr = ue.get_params_dram_addr()
    ue.dma_write(DMA_DEVICE_H2C, addr, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16), size)
    ue.allocate_params_dram(size)
    return addr

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

def bf16_permute_core_v2(ue, dims, permute_indices, input_dram_addr, output_dram_addr,
                          params_dram_addr, temp_dram_start):
    """Permute via DMA gather + batched transpose decomposition.
    Adapted from bf16_permute_core_v2 — handles non-64-aligned last_dim (e.g. 80)."""
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

        if last_dim < UE_VECTOR_SIZE or last_dim % UE_VECTOR_SIZE != 0:
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

def _qs_pack(precision: str, tensor: torch.Tensor):
    """Run a 64-block codec via quant_schemas and emit the scale-then-data
    byte layout the released wire format uses (consumed by
    store_quantized_weight() as 34 bytes per block: 2 B bf16 scale + 32 B
    nibbles). Accepts arbitrary input shape; flattens and zero-pads to a
    multiple of 64 to match the released helpers."""
    bf = tensor.detach().to(torch.bfloat16).cpu()
    # Fast path: 2D and K-aligned. Preserve (N, K) shape so the codec can
    # chunk along rows (large N inputs like the LM head would otherwise hit
    # the unbounded distance-tensor allocation when flattened to (1, N*K)).
    if bf.dim() == 2 and bf.shape[1] % 64 == 0:
        n_blocks = bf.numel() // 64
        data_bytes, scale_bytes = quant_schemas.quantize(precision, bf, block_size=64)
        return np.frombuffer(scale_bytes + data_bytes, dtype=np.uint8), n_blocks
    # Generic path: flatten + zero-pad to a multiple of 64. Used for
    # arbitrary-rank tensors (e.g. the 5D Conv3D patch_embed weight).
    bf = bf.flatten()
    n_blocks = (bf.numel() + 63) // 64
    if bf.numel() != n_blocks * 64:
        bf = torch.nn.functional.pad(bf, (0, n_blocks * 64 - bf.numel()))
    bf = bf.view(1, -1)
    data_bytes, scale_bytes = quant_schemas.quantize(precision, bf, block_size=64)
    return np.frombuffer(scale_bytes + data_bytes, dtype=np.uint8), n_blocks

_LAYER_MAP = {
    'lm': {
        'self_attn.q_proj.weight': 'attn_q.weight', 'self_attn.q_proj.bias': 'attn_q.bias',
        'self_attn.k_proj.weight': 'attn_k.weight', 'self_attn.k_proj.bias': 'attn_k.bias',
        'self_attn.v_proj.weight': 'attn_v.weight', 'self_attn.v_proj.bias': 'attn_v.bias',
        'self_attn.o_proj.weight': 'attn_output.weight',
        'mlp.gate_proj.weight': 'ffn_gate.weight', 'mlp.up_proj.weight': 'ffn_up.weight',
        'mlp.down_proj.weight': 'ffn_down.weight', 'input_layernorm.weight': 'attn_norm.weight',
        'post_attention_layernorm.weight': 'ffn_norm.weight',
    },
    'vision': {
        'attn.qkv.weight': 'attn_qkv.weight', 'attn.qkv.bias': 'attn_qkv.bias',
        'attn.proj.weight': 'attn_out.weight', 'attn.proj.bias': 'attn_out.bias',
        'mlp.gate_proj.weight': 'ffn_gate.weight', 'mlp.gate_proj.bias': 'ffn_gate.bias',
        'mlp.up_proj.weight': 'ffn_up.weight', 'mlp.up_proj.bias': 'ffn_up.bias',
        'mlp.down_proj.weight': 'ffn_down.weight', 'mlp.down_proj.bias': 'ffn_down.bias',
        'norm1.weight': 'norm1.weight', 'norm2.weight': 'norm2.weight',
    },
}
_TOP_MAP = {
    'lm': {'embed_tokens.weight': 'token_embd.weight', 'norm.weight': 'output_norm.weight'},
    'vision': {
        'patch_embed.proj.weight': 'v.patch_embd.weight',
        'patch_embed.proj.bias': 'v.patch_embd.bias',
        'merger.ln_q.weight': 'v.merger_ln_q.weight',
        'merger.mlp.0.weight': 'v.merger_mlp0.weight',
        'merger.mlp.0.bias': 'v.merger_mlp0.bias',
        'merger.mlp.2.weight': 'v.merger_mlp2.weight',
        'merger.mlp.2.bias': 'v.merger_mlp2.bias',
    },
}
# LM-side quant scope: q/k/gate/up/down (v_proj and o_proj stay BF16 for
# attention accuracy). Same set for every supported precision.
_LM_QUANT_LAYERS = {'q_proj.weight', 'k_proj.weight',
                    'gate_proj.weight', 'up_proj.weight', 'down_proj.weight'}

_VALID_PRECISIONS = ('int4', 'fp4', 'if4')

def _lm_precision(cfg: dict) -> str:
    """Read LM precision from the config, defaulting to the eval-winner 'if4'.
    Validates against the codecs the quant_schemas wrapper actually supports."""
    p = cfg.get("precision", {}).get("lm", "if4")
    if p not in _VALID_PRECISIONS:
        raise ValueError(f"config precision.lm={p!r} not in {_VALID_PRECISIONS}")
    return p

def _vision_precision(cfg: dict) -> str:
    """Read vision precision from the config. Default 'int4' matches the
    legacy released vision codec (Q4_64 = pure INT4 codes, all-negative
    bf16 scales — HW INT4 dispatch). Same precision string drives both the
    generator's codec call and the manifest suffix the loader looks for."""
    p = cfg.get("precision", {}).get("vision", "int4")
    if p not in _VALID_PRECISIONS:
        raise ValueError(f"config precision.vision={p!r} not in {_VALID_PRECISIONS}")
    return p

def _weight_key(hf_name, mode):
    """Map HF param name to short weight key. mode='lm' or 'vision'."""
    name = hf_name
    for pfx in ('model.', 'visual.'):
        if name.startswith(pfx):
            name = name[len(pfx):]
            break
    if name in _TOP_MAP[mode]:
        return _TOP_MAP[mode][name]
    if mode == 'lm' and name.startswith('layers.'):
        p = name.split('.'); comp = '.'.join(p[2:])
        if comp in _LAYER_MAP['lm']:
            return f'blk.{p[1]}.{_LAYER_MAP["lm"][comp]}'
    elif mode == 'vision' and name.startswith('blocks.'):
        p = name.split('.'); comp = '.'.join(p[2:])
        if comp in _LAYER_MAP['vision']:
            return f'v.blk.{p[1]}.{_LAYER_MAP["vision"][comp]}'
    return name

def _write_weight_bin(bin_path, model, param_filter, mode, suffix, quant_layers, qfn):
    """Write a weight bin + json manifest. Params whose name ends with one
    of ``quant_layers`` go through ``qfn`` (returning packed scale+nibble
    bytes) and get a ``.{suffix}`` manifest key; everything else is stored
    BF16 with no suffix."""
    json_path = bin_path.rsplit('.', 1)[0] + '.json'
    manifest = {}
    count = 0
    with open(bin_path, 'wb') as f:
        for pname, param in model.named_parameters():
            if not param_filter(pname):
                continue
            key = _weight_key(pname, mode)
            t = param.data
            if any(pname.endswith(s) for s in quant_layers):
                data, _ = qfn(t)
                raw = data.tobytes()
                key = f'{key}.{suffix}'
            else:
                raw = t.to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes()
            offset = f.tell()
            f.write(raw)
            manifest[key] = {'offset': offset, 'size': len(raw)}
            count += 1
    with open(json_path, 'w') as f:
        json.dump(manifest, f)
    print(f"Weights: {count} tensors, {os.path.getsize(bin_path)/1048576:.1f} MB -> {bin_path}")

def generate_lm_weights(model, output_path, precision: str = "if4"):
    """Generate LM weight bin using the given quant_schemas precision
    ('int4' / 'fp4' / 'if4'). The precision string is also the manifest
    suffix the runtime loader looks for.

    Also pre-quantizes the LM head (tied to the input embedding) and
    appends it to the bin as ``lm_head.weight.{precision}``. The runtime
    loads those bytes directly instead of re-quantizing 300M+ weights at
    weight_init, which OOMs memory-constrained devices like Pi 5."""
    if precision not in _VALID_PRECISIONS:
        raise ValueError(f"precision={precision!r} not in {_VALID_PRECISIONS}")
    _write_weight_bin(output_path, model,
        lambda n: 'model.layers' in n or 'model.embed_tokens' in n or 'model.norm' in n,
        'lm', precision, _LM_QUANT_LAYERS, lambda t: _qs_pack(precision, t))

    # Pre-quantize the LM head (tied to embedding) and append to the bin.
    embed_w = model.get_input_embeddings().weight.detach().to(torch.bfloat16)
    combined, _ = _qs_pack(precision, embed_w)
    combined_bytes = combined.tobytes()
    json_path = output_path.rsplit('.', 1)[0] + '.json'
    with open(json_path) as f:
        manifest = json.load(f)
    with open(output_path, 'ab') as f:
        offset = f.tell()
        f.write(combined_bytes)
    manifest[f'lm_head.weight.{precision}'] = {'offset': offset, 'size': len(combined_bytes)}
    with open(json_path, 'w') as f:
        json.dump(manifest, f)
    print(f"LM head ({precision}) appended: {len(combined_bytes)/1048576:.1f} MB at offset 0x{offset:X}")

def generate_vision_weights(model, output_path, precision: str = "int4"):
    """Generate vision weight bin with pre-padded QKV (80→128) and MLP
    (3420→3456). The given quant_schemas precision ('int4' / 'fp4' / 'if4')
    drives both the codec and the manifest suffix.

    The binary stores padded weights so the runtime doesn't need the HF model.
    QKV is stored as separate qk_padded (rearranged 128-dim) and v_padded
    (sequential 128-dim).
    """
    if precision not in _VALID_PRECISIONS:
        raise ValueError(f"precision={precision!r} not in {_VALID_PRECISIONS}")
    sfx = precision  # manifest suffix tag = precision string
    qpack = lambda t: _qs_pack(precision, t)
    VN, VD, VD_PAD, VH = 16, 80, 128, 1280
    VI = 3420
    VIS_MLP_PAD = ((VI + 63) // 64) * 64
    half_d = VD // 2
    json_path = output_path.rsplit('.', 1)[0] + '.json'
    manifest = {}
    count = 0
    with open(output_path, 'wb') as f:
        for i, block in enumerate(model.visual.blocks):
            prefix = f'visual.blocks.{i}'
            # QKV: pad and split into qk_padded (rearranged) + v_padded (sequential)
            qkv_w = block.attn.qkv.weight.detach().to(torch.bfloat16)
            qkv_b = block.attn.qkv.bias.detach().to(torch.bfloat16)
            qkv_w_3d = qkv_w.view(3, VN, VD, VH)
            qkv_b_3d = qkv_b.view(3, VN, VD)
            # Q/K rearranged padded
            qk_padded_w = torch.zeros(2 * VN * VD_PAD, VH, dtype=torch.bfloat16)
            qk_padded_b = torch.zeros(2 * VN * VD_PAD, dtype=torch.bfloat16)
            for proj in range(2):
                for h in range(VN):
                    hs = (proj * VN + h) * VD_PAD
                    qk_padded_w[hs:hs+half_d, :] = qkv_w_3d[proj, h, :half_d, :]
                    qk_padded_w[hs+64:hs+64+half_d, :] = qkv_w_3d[proj, h, half_d:, :]
                    qk_padded_b[hs:hs+half_d] = qkv_b_3d[proj, h, :half_d]
                    qk_padded_b[hs+64:hs+64+half_d] = qkv_b_3d[proj, h, half_d:]
            data, _ = qpack(qk_padded_w)
            raw = data.tobytes(); offset = f.tell(); f.write(raw)
            manifest[f'{prefix}.attn.qk_padded.weight.{sfx}'] = {'offset': offset, 'size': len(raw)}
            raw = qk_padded_b.contiguous().view(torch.uint16).cpu().numpy().tobytes(); offset = f.tell(); f.write(raw)
            manifest[f'{prefix}.attn.qk_padded.bias'] = {'offset': offset, 'size': len(raw)}
            # V sequential padded
            v_padded_w = torch.zeros(VN * VD_PAD, VH, dtype=torch.bfloat16)
            v_padded_b = torch.zeros(VN * VD_PAD, dtype=torch.bfloat16)
            for h in range(VN):
                hs = h * VD_PAD
                v_padded_w[hs:hs+VD, :] = qkv_w_3d[2, h, :, :]
                v_padded_b[hs:hs+VD] = qkv_b_3d[2, h, :]
            data, _ = qpack(v_padded_w)
            raw = data.tobytes(); offset = f.tell(); f.write(raw)
            manifest[f'{prefix}.attn.v_padded.weight.{sfx}'] = {'offset': offset, 'size': len(raw)}
            raw = v_padded_b.contiguous().view(torch.uint16).cpu().numpy().tobytes(); offset = f.tell(); f.write(raw)
            manifest[f'{prefix}.attn.v_padded.bias'] = {'offset': offset, 'size': len(raw)}
            # O proj (no padding needed)
            data, _ = qpack(block.attn.proj.weight.detach().to(torch.bfloat16))
            raw = data.tobytes(); offset = f.tell(); f.write(raw)
            manifest[f'{prefix}.attn.proj.weight.{sfx}'] = {'offset': offset, 'size': len(raw)}
            raw = block.attn.proj.bias.detach().to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes()
            offset = f.tell(); f.write(raw)
            manifest[f'{prefix}.attn.proj.bias'] = {'offset': offset, 'size': len(raw)}
            # MLP: pad 3420→3456
            for proj_name in ['gate_proj', 'up_proj']:
                w = getattr(block.mlp, proj_name).weight.detach().to(torch.bfloat16)
                w_padded = torch.zeros(VIS_MLP_PAD, VH, dtype=torch.bfloat16); w_padded[:w.shape[0]] = w
                data, _ = qpack(w_padded)
                raw = data.tobytes(); offset = f.tell(); f.write(raw)
                manifest[f'{prefix}.mlp.{proj_name}.weight.{sfx}'] = {'offset': offset, 'size': len(raw)}
                b = getattr(block.mlp, proj_name).bias.detach().to(torch.bfloat16)
                b_padded = torch.zeros(VIS_MLP_PAD, dtype=torch.bfloat16); b_padded[:b.shape[0]] = b
                raw = b_padded.contiguous().view(torch.uint16).cpu().numpy().tobytes(); offset = f.tell(); f.write(raw)
                manifest[f'{prefix}.mlp.{proj_name}.bias'] = {'offset': offset, 'size': len(raw)}
            down_w = block.mlp.down_proj.weight.detach().to(torch.bfloat16)
            down_padded = torch.zeros(VH, VIS_MLP_PAD, dtype=torch.bfloat16); down_padded[:, :down_w.shape[1]] = down_w
            data, _ = qpack(down_padded)
            raw = data.tobytes(); offset = f.tell(); f.write(raw)
            manifest[f'{prefix}.mlp.down_proj.weight.{sfx}'] = {'offset': offset, 'size': len(raw)}
            raw = block.mlp.down_proj.bias.detach().to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes()
            offset = f.tell(); f.write(raw)
            manifest[f'{prefix}.mlp.down_proj.bias'] = {'offset': offset, 'size': len(raw)}
            # Norms
            for norm_name in ['norm1', 'norm2']:
                raw = getattr(block, norm_name).weight.detach().to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes()
                offset = f.tell(); f.write(raw)
                manifest[f'{prefix}.{norm_name}.weight'] = {'offset': offset, 'size': len(raw)}
            count += 1
        # Patch embed (no padding needed)
        data, _ = qpack(model.visual.patch_embed.proj.weight.detach().to(torch.bfloat16))
        raw = data.tobytes(); offset = f.tell(); f.write(raw)
        manifest[f'visual.patch_embed.proj.weight.{sfx}'] = {'offset': offset, 'size': len(raw)}
        # Merger
        raw = model.visual.merger.ln_q.weight.detach().to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes()
        offset = f.tell(); f.write(raw)
        manifest['visual.merger.ln_q.weight'] = {'offset': offset, 'size': len(raw)}
        for mlp_idx in [0, 2]:
            data, _ = qpack(model.visual.merger.mlp[mlp_idx].weight.detach().to(torch.bfloat16))
            raw = data.tobytes(); offset = f.tell(); f.write(raw)
            manifest[f'visual.merger.mlp.{mlp_idx}.weight.{sfx}'] = {'offset': offset, 'size': len(raw)}
            raw = model.visual.merger.mlp[mlp_idx].bias.detach().to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes()
            offset = f.tell(); f.write(raw)
            manifest[f'visual.merger.mlp.{mlp_idx}.bias'] = {'offset': offset, 'size': len(raw)}
    with open(json_path, 'w') as f:
        json.dump(manifest, f)
    print(f"Vision weights (padded {precision}): {count} layers + merger, {os.path.getsize(output_path)/1048576:.1f} MB -> {output_path}")

def _load_config(script_dir: str) -> dict:
    """Load qwen2.5_vl_3b_config.json and build weight_defs (offset/size dict) from regions."""
    config_path = os.path.join(script_dir, "qwen2.5_vl_3b_config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)
    weight_defs = {"LAYER_WEIGHT_SIZE": cfg["file_info"]["layer_size"]}
    for key, r in cfg.get("regions", {}).items():
        weight_defs[key] = _parse_offset(r["offset"])
        weight_defs[f"{key}_SIZE"] = r["size"]
    for key, r in cfg.get("non_layer_regions", {}).items():
        weight_defs[key] = _parse_offset(r["offset"])
        weight_defs[f"{key}_SIZE"] = r["size"]
    cfg["_weight_defs"] = weight_defs
    return cfg

def _ensure_hf_model(script_dir: str, cfg: dict):
    """Ensure HF model is downloaded and loaded. Returns (model, model_dir)."""
    model_dir = os.path.join(script_dir, cfg["paths"]["hf_model_dir"])
    hf_repo = cfg["paths"]["hf_model_repo"]
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        _original_print(f"Downloading HF model {hf_repo} to {os.path.abspath(model_dir)} ...")
        snapshot_download(repo_id=hf_repo, local_dir=model_dir, local_dir_use_symlinks=False)
        _original_print("Download complete.")
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map=None, trust_remote_code=True
    )
    # Newer transformers nests vision encoder under model.model.visual;
    # expose it as model.visual so the rest of the code can access it directly.
    if not hasattr(model, 'visual') and hasattr(model, 'model') and hasattr(model.model, 'visual'):
        model.visual = model.model.visual
    return model, model_dir

def weight_bin_generate(script_dir: str | None = None, output_lm: str | None = None, output_vis: str | None = None) -> tuple[str, str]:
    """Generate LM and vision weight bins from Hugging Face model.
    Returns (lm_path, vis_path)."""
    script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
    cfg = _load_config(script_dir)
    paths = cfg["paths"]
    lm_path = output_lm or os.path.join(script_dir, paths["lm_weights"])
    vis_path = output_vis or os.path.join(script_dir, paths["vision_weights"])
    os.makedirs(os.path.dirname(lm_path), exist_ok=True)

    model, model_dir = _ensure_hf_model(script_dir, cfg)
    generate_lm_weights(model, lm_path, precision=_lm_precision(cfg))
    generate_vision_weights(model, vis_path, precision=_vision_precision(cfg))
    del model
    return lm_path, vis_path

class Qwen25VL3B_UnifiedEngine(UnifiedEngine):
    """UnifiedEngine for Qwen2.5-VL-3B-Instruct: config + weight bins, compile, run."""

    def __init__(self, script_dir: str | None = None, hf_model_dir: str | None = None):
        # Initialize identity-DMA dedup state before super().__init__() which calls dma_write.
        self._identity_dram_written = False
        self._identity_dram_addr = None
        self._IDENTITY_MAT_BYTES = UE_VECTOR_SIZE * UE_VECTOR_SIZE * 2  # bfloat16 bytes
        # Qwen2.5-VL-3B DRAM layout starting from 0x00000000 (4GB DRAM):
        #   params:  0x00000000 – 0x90000000 (~2.25 GB: NO embedding upload (host-side),
        #            36 LM layers INT4+BF16 V/O, INT4 lm_head 165MB, RoPE, vision FP4 ~350MB)
        #   tensors: 0x90000000 – 0xA0000000 (256 MB: LM+vision intermediates ~182MB)
        #   instructions: 0xA0000000 – 0x100000000 (1.5 GB: encoder ~648MB + LM ~94MB)
        super().__init__(
            params_dram_base=0x00000000,
            tensor_dram_base=0x90000000,
            program_dram_base=0xA0000000,
        )
        self.script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
        self._cfg = _load_config(self.script_dir)
        self.weight_defs = self._cfg["_weight_defs"]

        fi = self._cfg["file_info"]
        model = self._cfg["model"]
        paths = self._cfg["paths"]

        self.vector_length = fi["hidden_size"]
        self.head_dim = fi["head_dim"]
        self.actual_head_dim = fi["actual_head_dim"]
        self.num_kv_heads = fi["num_kv_heads"]
        self.bytes_per_element = fi["bytes_per_element"]
        self.group_size = fi["group_size"]
        self.mlp_elements = fi["mlp_elements"]
        self.hf_model_dir = hf_model_dir or os.path.join(self.script_dir, paths["hf_model_dir"])
        self.q_size = self.head_dim * self.group_size * self.bytes_per_element
        self.k_size = self.head_dim * self.bytes_per_element
        self.MAX_CONTEXT_SIZE = model["max_context_size"]
        self.LAYER_SIZE = fi["num_layers"]
        self.EMBEDDING_ELEMENTS = fi["embedding_vocab"]
        self._isa_reg_counter = 1
        fixed = self._cfg.get("fixed_isa_regs", {})
        self.V_CACHE_SIZE_REG = fixed["V_CACHE_SIZE_REG"]
        self.TMP_REG = fixed["TMP_REG"]
        self.ROPE_SIZE_REG = fixed["ROPE_SIZE_REG"]
        self.causal_mask_upper = False
        self._rope_global_layers = set(model["rope_global_layers"])  # empty for Qwen2.5-VL
        self._end_of_turn_token_id = model["end_of_turn_token_id"]
        self._gamma_bin_offset = self._cfg["special"]["rms_norm"]["gamma_offset"]

        # Qwen2.5-VL architecture flags
        self._has_q_k_norm = False       # No QK RMSNorm
        self._has_post_attn_norm = False  # no post-attention norm
        self._has_post_mlp_norm = False   # no post-FFN norm
        self._has_qkv_bias = True        # Q/K/V have bias

        # Vision config
        self._vis_cfg = self._cfg.get("vision", {})

        # Weight loading
        self.weight_init()
        self.tensor_init()

    _IDENTITY_MAT_BYTES = UE_VECTOR_SIZE * UE_VECTOR_SIZE * 2

    def _preallocate_identity_matrix(self) -> None:
        """Allocate the identity-matrix DRAM slot once and DMA-write it."""
        if self._identity_dram_addr is not None:
            return
        self._identity_dram_addr = super().allocate_params_dram(self._IDENTITY_MAT_BYTES)
        eye = torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16)
        super().dma_write(DMA_DEVICE_H2C, self._identity_dram_addr, eye, self._IDENTITY_MAT_BYTES)
        self._identity_dram_written = True
        _original_print(f"  Identity matrix pre-allocated at DRAM 0x{self._identity_dram_addr:X}")

    def _flash_attention_core_cached(self, **kwargs) -> int:
        """Wrap flash_attention_core so the identity-matrix slot is always reused."""
        saved = self._next_params_dram_addr
        self._next_params_dram_addr = self._identity_dram_addr
        result = self.flash_attention_core(**kwargs)
        self._next_params_dram_addr = saved
        return result

    def dma_write(self, device, addr, data, size):
        """Skip redundant identity-matrix DMA writes (already in DRAM from pre-allocation)."""
        if (self._identity_dram_written
                and self._identity_dram_addr is not None
                and addr == self._identity_dram_addr
                and size == self._IDENTITY_MAT_BYTES):
            return  # already written; same content, skip
        super().dma_write(device, addr, data, size)

    def reset_isa_reg_counter(self) -> None:
        """Reset the ISA register allocation counter to 1 (register 0 is hard-wired zero)."""
        self._isa_reg_counter = 1

    def alloc_isa_reg(self, reset: bool = False) -> int:
        """Allocate next ISA register (1-15). Reg 0 is hard-wired zero."""
        if reset:
            self._isa_reg_counter = 1

        if self._isa_reg_counter > 15:
            raise ValueError("Exceeded available ISA registers (max 15)")

        reg_idx = self._isa_reg_counter
        self._isa_reg_counter += 1
        return reg_idx

    # overwrite_instruction_with_general_register: use base class (user_dma_core.py)

    def isa_add_set_core(self, dst_reg_idx: int, immediate_value: int, timeout_s: float = 10.0) -> None:
        """Set one ISA register to an immediate value via ADD SET + HALT program."""
        self.clear_inst_id()
        self.start_capture()
        self.generate_instruction_add_set(dst_reg_idx, immediate_value)
        self.stop_capture()
        self.generate_instruction_halt()
        program_addr = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(program_addr)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        self.clear_capture_buffer()
        self.start_execute_from_dram(program_addr)
        self.wait_queue(timeout_s)

    def write_captured_instructions_to_file(self, start_addr: int, filename: str = "captured_instructions.bin") -> None:
        """Write all captured instructions to a binary file."""
        if not hasattr(self, 'capture_buffer') or not self.capture_buffer:
            print("Warning: No captured instructions to write to file.")
            return

        all_instructions_bytes = bytearray()
        for inst in self.capture_buffer:
            all_instructions_bytes.extend(inst.get_bytes())

        with open(filename, "wb") as f:
            f.write(all_instructions_bytes)

        print(f"Successfully wrote {len(self.capture_buffer)} captured instructions ({len(all_instructions_bytes)} bytes) to {filename}")

    def load_instructions(self, bin_path: str) -> tuple[int, int]:
        """Load decoder instruction bin from file into program DRAM. Returns (start_addr, total_size)."""
        with open(bin_path, "rb") as f:
            data = f.read()
        total_size = len(data)
        start_addr = self.allocate_program_dram(total_size)
        self.dma_write(DMA_DEVICE_H2C, start_addr, data, total_size)
        print(f"    Loaded {total_size} bytes from instruction.bin to DRAM at 0x{start_addr:x}")
        return start_addr, total_size

    def allocate_params_dram(self, size_bytes: int) -> int:
        """Allocate from params DRAM region. Returns address before increment."""
        params_dram_addr = self._next_params_dram_addr
        self._next_params_dram_addr += size_bytes
        return params_dram_addr

    def clear_inst_id(self) -> None:
        """Reset instruction ID counter for the next capture."""
        self._inst_id = 0

    def get_arg_max_index(self) -> int:
        """Get the arg max index from the Unified Engine"""
        return self.read_reg32(UE_ARGMAX1_INDEX)

    def rope_hf_core(self, N: int, input_dram_addr: int, output_dram_addr: int, cos_dram_addr: int, sin_dram_addr: int, rope_size_reg: int = None, output_addr_inc_reg: int = None, tmp_reg: int = None) -> int:
        """RoPE (HuggingFace style). Caller must have start_capture() before and stop_capture() after."""
        assert N % UE_VECTOR_SIZE == 0 and N >= 64, f"N must be a multiple of {UE_VECTOR_SIZE} and >= 64"
        assert N % 2 == 0, "N must be even for RoPE half layout"
        assert N >= 128, "N must be >= 128 so half-vector SRAM offsets are 128-byte aligned"
        half = N // 2
        bytes_per_elem = 2
        sram_x = 0x00000
        sram_a = 0x20000
        sram_d = 0x40000
        sram_cos = 0x80000
        sram_sin = 0x80000 + N * bytes_per_elem
        sram_bc = 0x80000 + N * bytes_per_elem * 2
        self.accelerator_memory_to_sram(accelerator_dram_address=input_dram_addr, sram_address=sram_x, element_size=N)
        if rope_size_reg is not None:
            self.generate_instruction_add_imm(rope_size_reg, ue_35bit_addr_shifter(cos_dram_addr), tmp_reg)
            self.accelerator_memory_to_sram(accelerator_dram_address=cos_dram_addr, sram_address=sram_cos, element_size=N)
            self.overwrite_instruction_with_general_register(tmp_reg)
            self.generate_instruction_add_imm(rope_size_reg, ue_35bit_addr_shifter(sin_dram_addr), tmp_reg)
            self.accelerator_memory_to_sram(accelerator_dram_address=sin_dram_addr, sram_address=sram_sin, element_size=N)
            self.overwrite_instruction_with_general_register(tmp_reg)
        else:
            self.accelerator_memory_to_sram(accelerator_dram_address=cos_dram_addr, sram_address=sram_cos, element_size=N)
            self.accelerator_memory_to_sram(accelerator_dram_address=sin_dram_addr, sram_address=sram_sin, element_size=N)
        self.eltwise_mul_core(vector_A_sram_start_addr=sram_x, vector_B_sram_start_addr=sram_cos, vector_C_sram_wb_addr=sram_a, element_size=N)
        self.eltwise_mul_core(vector_A_sram_start_addr=sram_x + half * bytes_per_elem, vector_B_sram_start_addr=sram_sin, vector_C_sram_wb_addr=sram_bc, element_size=half)
        self.eltwise_mul_core(vector_A_sram_start_addr=sram_x, vector_B_sram_start_addr=sram_sin + half * bytes_per_elem, vector_C_sram_wb_addr=sram_bc + half * bytes_per_elem, element_size=half)
        self.eltwise_add_core(vector_A_sram_start_addr=sram_a, vector_B_sram_start_addr=sram_bc, vector_C_sram_wb_addr=sram_d, element_size=N)
        if output_addr_inc_reg is not None:
            self.generate_instruction_add_imm(output_addr_inc_reg, ue_35bit_addr_shifter(output_dram_addr), tmp_reg)
            self.sram_to_accelerator_memory(sram_address=sram_d, accelerator_dram_address=output_dram_addr, element_size=N)
            self.overwrite_instruction_with_general_register(tmp_reg)
        else:
            self.sram_to_accelerator_memory(sram_address=sram_d, accelerator_dram_address=output_dram_addr, element_size=N)
        return 4 * N

    def decoder_attention_core(self, head_dim: int, seq_len: int, Q_DRAM_ADDR: int, K_DRAM_ADDR: int, V_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, SCRATCH_DRAM_ADDR: int, IDENTITY_DRAM_ADDR: int = None, BIAS_DRAM_ADDR: int = None,
                            debug_mode: bool = False, SM_OUTPUT_DRAM_ADDR: int = None) -> None:

        bytes_per_element = 2
        bias_enable = True if BIAS_DRAM_ADDR is not None else False

        if debug_mode: # DEBUG only, needs to be allocated in DRAM
            assert SM_OUTPUT_DRAM_ADDR is not None, "SM_OUTPUT_DRAM_ADDR is not set for debug mode"

        # SCRATCH_DRAM_ADDR is used for V^T
        SCRATCH_DRAM_PARTIAL_SM = SCRATCH_DRAM_ADDR + head_dim * seq_len * bytes_per_element # used for partial softmax output

        # I @ V^T: (head_dim, head_dim) @ (seq_len, head_dim)^T -> (head_dim, seq_len)
        M = head_dim
        K = head_dim
        N = seq_len

        # transfer identity matrix to URAM_A start
        self.accelerator_memory_to_sram(accelerator_dram_address=IDENTITY_DRAM_ADDR,
                                        sram_address=0,
                                        element_size=UE_VECTOR_SIZE * UE_VECTOR_SIZE)

        usable_uram_a_start_addr = UE_VECTOR_SIZE * UE_VECTOR_SIZE * bytes_per_element

        # URAM_B is used for V matrix, we need to chunk the V matrix into smaller chunks that can fit in URAM_B
        usable_uram_b_elements = URAM_NEAR_FULL_ELEMENTS
        N_chunk = min(N, (usable_uram_b_elements // K) // UE_VECTOR_SIZE * UE_VECTOR_SIZE)
        N_chunk_aligned = None
        if N_chunk < UE_VECTOR_SIZE:
            if (K * 32) <= usable_uram_b_elements:
                N_chunk = 32
            elif (K * 16) <= usable_uram_b_elements:
                N_chunk = 16
            else:
                assert False, f"K={K} is too large to fit in usable URAM elements={usable_uram_b_elements}"
            N_chunk_aligned = UE_VECTOR_SIZE

        usable_uram_a_elements = URAM_FULL_ELEMENTS - UE_VECTOR_SIZE * UE_VECTOR_SIZE
        output_N_size = N_chunk_aligned if N_chunk_aligned is not None else N_chunk
        M_chunk = min(M, usable_uram_a_elements // output_N_size)
        assert M_chunk >= 1 and M_chunk <= M

        print(f"M_chunk: {M_chunk}, N_chunk: {N_chunk}", f"N_chunk_aligned: {N_chunk_aligned}")
        print(f"URAM_A usage: {100 * (UE_VECTOR_SIZE * UE_VECTOR_SIZE + M_chunk * output_N_size) / URAM_FULL_ELEMENTS:.2f}% of URAM_NEAR_FULL_ELEMENTS")
        print(f"URAM_B usage: {100 * N_chunk * K / URAM_FULL_ELEMENTS:.2f}% of URAM_FULL_ELEMENTS")

        output_sram_wb_addr = usable_uram_a_start_addr
        uram_b_start_addr = 0x80000
        for i, m_take in self.chunk_ranges(M, M_chunk):
            for j, n_take in self.chunk_ranges(N, N_chunk):

                self.accelerator_memory_to_sram(accelerator_dram_address=V_DRAM_ADDR + j * K * bytes_per_element,
                                            sram_address=uram_b_start_addr,
                                            element_size=n_take * K)

                for output_row in range(m_take):
                    if N_chunk_aligned is None:
                        out_sram_offset = output_row * n_take * bytes_per_element
                    else:
                        out_sram_offset = output_row * N_chunk_aligned * bytes_per_element

                    ones_idx = (output_row + i) // UE_VECTOR_SIZE
                    vector_idx = (output_row + i) % UE_VECTOR_SIZE

                    self.start_queue_for_bf16_matvec_operation(max_clear_en=0,
                                                            fmax_context_addr=0,
                                                            vector_sram_start_addr=0x00000 + vector_idx * UE_VECTOR_SIZE * bytes_per_element,
                                                            matrix_sram_start_addr=uram_b_start_addr + ones_idx * UE_VECTOR_SIZE * bytes_per_element,
                                                            output_sram_wb_addr=output_sram_wb_addr + out_sram_offset,
                                                            K=UE_VECTOR_SIZE,
                                                            N=n_take,
                                                            stride_z=m_take)

                start_dram_address_of_partial_matrix = SCRATCH_DRAM_ADDR + i * N * bytes_per_element + j * bytes_per_element

                if N_chunk_aligned is None:
                    self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr,
                                                    accelerator_dram_address=start_dram_address_of_partial_matrix,
                                                    element_size=m_take * n_take,
                                                    stride_bytes_per_chunk=n_take * bytes_per_element,
                                                    stride_jump_bytes=N * bytes_per_element)
                else:
                    for o_row_idx in range(m_take):
                        self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr + o_row_idx * N_chunk_aligned * bytes_per_element,
                                                        accelerator_dram_address=start_dram_address_of_partial_matrix + o_row_idx * N * bytes_per_element,
                                                        element_size=n_take)

        # Q @ K^T: (1, head_dim) @ (head_dim, seq_len) -> (1, seq_len)
        M = 1
        K = head_dim
        N = seq_len
        usable_uram_b_elements = URAM_NEAR_FULL_ELEMENTS
        N_chunk = min(N, (usable_uram_b_elements // K) // UE_VECTOR_SIZE * UE_VECTOR_SIZE)
        N_chunk_aligned = None
        if N_chunk < UE_VECTOR_SIZE:
            if (K * 32) <= usable_uram_b_elements:
                N_chunk = 32
            elif (K * 16) <= usable_uram_b_elements:
                N_chunk = 16
            else:
                assert False, f"K={K} is too large to fit in usable URAM elements={usable_uram_b_elements}"
            N_chunk_aligned = UE_VECTOR_SIZE

        usable_uram_a_elements = URAM_FULL_ELEMENTS
        output_N_size = N_chunk_aligned if N_chunk_aligned is not None else N_chunk
        M_chunk = min(UE_FMAX_CONTEXT_SIZE, M, usable_uram_a_elements // (K + output_N_size))
        assert M_chunk >= 1 and M_chunk <= M

        print(f"M_chunk: {M_chunk}, N_chunk: {N_chunk}", f"N_chunk_aligned: {N_chunk_aligned}")
        print(f"URAM_A usage: {100 * (M_chunk * K + M_chunk * output_N_size) / URAM_FULL_ELEMENTS:.2f}% of URAM_NEAR_FULL_ELEMENTS")
        print(f"URAM_B usage: {100 * N_chunk * K / URAM_FULL_ELEMENTS:.2f}% of URAM_FULL_ELEMENTS")

        uram_a_start_addr = 0x00000
        uram_b_start_addr = 0x80000
        for i, m_take in self.chunk_ranges(M, M_chunk):
            self.accelerator_memory_to_sram(accelerator_dram_address=Q_DRAM_ADDR + i * K * bytes_per_element,
                                            sram_address=uram_a_start_addr,
                                            element_size=m_take * K)

            self.broadcast_mul(scalar=1 / math.sqrt(head_dim),
                                    sram_start_addr=uram_a_start_addr,
                                    sram_wb_addr=uram_a_start_addr,
                                    element_size=m_take * K)

            output_sram_wb_addr = uram_a_start_addr + m_take * K * bytes_per_element

            assert output_sram_wb_addr < 0x80000, f"output_sram_wb_addr={output_sram_wb_addr} is greater than 0x80000"

            clear_en = 1
            for j, n_take in self.chunk_ranges(N, N_chunk):
                self.accelerator_memory_to_sram(accelerator_dram_address=K_DRAM_ADDR + j * K * bytes_per_element,
                                            sram_address=uram_b_start_addr,
                                            element_size=n_take * K)

                if bias_enable:
                    self.accelerator_memory_to_bias_sram(accelerator_dram_address=BIAS_DRAM_ADDR + j * bytes_per_element,
                                                       element_size=n_take)

                assert m_take * K + n_take * m_take <= URAM_FULL_ELEMENTS

                for output_row in range(m_take):
                    # removed bias_enable as per causal mask drop

                    if N_chunk_aligned is None:
                        out_sram_offset = output_row * n_take * bytes_per_element
                    else:
                        out_sram_offset = output_row * N_chunk_aligned * bytes_per_element

                    self.start_queue_for_bf16_matvec_operation(max_clear_en=clear_en,
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
                    self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr,
                                                    accelerator_dram_address=start_dram_address_of_partial_matrix,
                                                    element_size=m_take * n_take,
                                                    stride_bytes_per_chunk=n_take * bytes_per_element,
                                                    stride_jump_bytes=N * bytes_per_element)
                else:
                    for o_row_idx in range(m_take):
                        self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr + o_row_idx * N_chunk_aligned * bytes_per_element,
                                                        accelerator_dram_address=start_dram_address_of_partial_matrix + o_row_idx * N * bytes_per_element,
                                                        element_size=n_take)


            # SOFTMAX CALCULATION
            max_m_take = min((URAM_FULL_ELEMENTS - UE_VECTOR_SIZE) // N, UE_FMAX_CONTEXT_SIZE) # worst case scenario, leave one row for output

            for m_take_chunk_idx, m_take_chunk_size in self.chunk_ranges(m_take, max_m_take):
                self.accelerator_memory_to_sram(accelerator_dram_address=SCRATCH_DRAM_PARTIAL_SM + m_take_chunk_idx * N * bytes_per_element,
                                            sram_address=uram_a_start_addr,
                                            element_size=m_take_chunk_size * N)

                # Reuse input sram_wb_addr for softmax output
                for row_idx in range(m_take_chunk_size):
                    self.start_queue_for_bf16_softmax_operation(fmax_context_addr=row_idx + m_take_chunk_idx,
                                                                vector_sram_start_addr=uram_a_start_addr + row_idx * N * bytes_per_element,
                                                                output_sram_wb_addr=uram_a_start_addr + row_idx * N * bytes_per_element,
                                                                N=N)


                # softmax output tap point - DEBUG only
                if debug_mode:
                    self.sram_to_accelerator_memory(sram_address=uram_a_start_addr,
                                    accelerator_dram_address=SM_OUTPUT_DRAM_ADDR + (i + m_take_chunk_idx) * N * bytes_per_element,
                                    element_size=m_take_chunk_size * N)

                v_tr_row_chunk_size = min((URAM_NEAR_FULL_ELEMENTS // seq_len // UE_VECTOR_SIZE) * UE_VECTOR_SIZE,
                                        ((URAM_FULL_ELEMENTS - m_take_chunk_size * seq_len) // m_take_chunk_size // UE_VECTOR_SIZE) * UE_VECTOR_SIZE,
                                        head_dim)

                v_tr_row_chunk_size_aligned = None
                if v_tr_row_chunk_size < UE_VECTOR_SIZE:
                    v_tr_row_chunk_size_aligned = UE_VECTOR_SIZE
                    if seq_len * 32 <= URAM_NEAR_FULL_ELEMENTS:
                        v_tr_row_chunk_size = 32
                    elif seq_len * 16 <= URAM_NEAR_FULL_ELEMENTS:
                        v_tr_row_chunk_size = 16
                    else:
                        assert False, f"v_tr_row_chunk_size={v_tr_row_chunk_size} is too large"

                v_t_sram_start_addr = 0x80000 # URAM_B start
                output_sram_wb_addr = uram_a_start_addr + m_take_chunk_size * seq_len * bytes_per_element

                for v_tr_column_idx, v_tr_column_take in self.chunk_ranges(head_dim, v_tr_row_chunk_size):
                    self.accelerator_memory_to_sram(accelerator_dram_address=SCRATCH_DRAM_ADDR + v_tr_column_idx * seq_len * bytes_per_element,
                                                sram_address=v_t_sram_start_addr,
                                                element_size=v_tr_column_take * seq_len)

                    for p_row_idx in range(m_take_chunk_size):
                        if v_tr_row_chunk_size_aligned is None:
                            output_sram_wb_offset = p_row_idx * v_tr_column_take * bytes_per_element
                        else:
                            output_sram_wb_offset = 0

                        self.start_queue_for_bf16_matvec_operation(max_clear_en=0,
                                                                fmax_context_addr=0,
                                                                vector_sram_start_addr=uram_a_start_addr + p_row_idx * seq_len * bytes_per_element,
                                                                matrix_sram_start_addr=v_t_sram_start_addr,
                                                                output_sram_wb_addr=output_sram_wb_addr + output_sram_wb_offset,
                                                                K=seq_len,
                                                                N=v_tr_column_take)

                        if v_tr_row_chunk_size_aligned is not None:
                            self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr + output_sram_wb_offset,
                                                            accelerator_dram_address=OUTPUT_DRAM_ADDR + (i + m_take_chunk_idx) * head_dim * bytes_per_element
                                                                                                        + v_tr_column_idx * bytes_per_element
                                                                                                        + p_row_idx * head_dim * bytes_per_element,
                                                            element_size=v_tr_column_take)


                    if v_tr_row_chunk_size_aligned is None:
                        self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr,
                                                        accelerator_dram_address=OUTPUT_DRAM_ADDR + (i + m_take_chunk_idx) * head_dim * bytes_per_element + v_tr_column_idx * bytes_per_element,
                                                        element_size=m_take_chunk_size * v_tr_column_take,
                                                        stride_bytes_per_chunk=v_tr_column_take * bytes_per_element,
                                                        stride_jump_bytes=head_dim * bytes_per_element)

        # Total Theoretical FLOPS
        total_flops = 1 * head_dim # q_scale
        total_flops += 2 * 1 * head_dim * seq_len # Q @ K^T
        total_flops += 1 * seq_len * 5 # softmax
        total_flops += 2 * 1 * seq_len * head_dim # sm @ v
        print(f"Total Theoretical FLOPS: {total_flops}")
        return total_flops

    def get_embedding_for_tokens(self, token_ids: list[int] | tuple) -> torch.Tensor:
        """Return (len(token_ids), vector_length) bfloat16 tensor from self.embedding_weight (no scaling)."""
        tid_t = torch.tensor(token_ids, dtype=torch.long)
        out = torch.zeros(len(token_ids), self.vector_length, dtype=torch.bfloat16)
        valid = tid_t < self.embedding_weight.shape[0]
        out[valid] = self.embedding_weight[tid_t[valid]]
        return out

    def _load_rope_host(self, rope_theta: float | None = None) -> None:
        """Generate per-position RoPE table and write to DRAM."""
        rope_cfg = self._cfg["special"]["rope"]
        theta = rope_theta if rope_theta is not None else rope_cfg["theta"]
        num_rope_positions = rope_cfg["num_positions"]
        D_per_head = self.actual_head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(D_per_head, dtype=torch.float32) / D_per_head))
        pos = torch.arange(num_rope_positions, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)                     # (num_positions, 64)
        cos_ = freqs.cos().to(torch.bfloat16)
        sin_ = freqs.sin().to(torch.bfloat16)
        # rope_hf_core(N=actual_head_dim=128): [cos(64), cos(64), -sin(64), sin(64)]
        rope_tensor = torch.cat([cos_, cos_, -sin_, sin_], dim=1)  # (num_positions, 256)
        rope_raw = rope_tensor.contiguous().view(torch.uint8).numpy().tobytes()
        rope_sz = self.weight_defs["ROPE_SIZE"]
        rope_raw_padded = (rope_raw + b"\x00" * rope_sz)[:rope_sz]
        rope_base = self.allocate_params_dram(rope_sz)
        self.dma_write(DMA_DEVICE_H2C, rope_base, rope_raw_padded, rope_sz)
        self.DRAM_ADDR_ROPE = rope_base
        # Store inv_freq for mrope reuse
        self._rope_inv_freq = inv_freq
        self._rope_theta = theta
        self._rope_table_bytes = rope_raw_padded  # save for restoring after mRoPE prefill

    def load_mrope_for_prefill(self, prefill_seq, image_grid_thw=None) -> None:
        """Overwrite DRAM RoPE table with per-token mRoPE for VLM prefill."""
        if image_grid_thw is None:
            return  # text-only, default RoPE table is fine

        mrope_sections = self._cfg["special"]["rope"].get("mrope_section", [16, 24, 24])
        spatial_merge_size = self._vis_cfg["spatial_merge_size"]
        image_pad_id = 151655  # <|image_pad|>
        vision_start_id = 151652  # <|vision_start|>
        seq_len = len(prefill_seq)
        tokens = list(prefill_seq)

        pos_t = torch.zeros(seq_len, dtype=torch.long)
        pos_h = torch.zeros(seq_len, dtype=torch.long)
        pos_w = torch.zeros(seq_len, dtype=torch.long)

        image_index = 0
        st = 0
        # Find vision_start tokens to locate images
        vision_starts = [i for i, t in enumerate(tokens) if t == vision_start_id]
        img_token_positions = [i for i, t in enumerate(tokens) if t == image_pad_id]

        if len(img_token_positions) > 0 and image_grid_thw is not None:
            # Text before image
            img_first = img_token_positions[0]
            text_before = img_first
            st_idx = 0
            for i in range(text_before):
                pos_t[i] = st_idx + i
                pos_h[i] = st_idx + i
                pos_w[i] = st_idx + i

            # Image tokens: grid-based positions
            t_grid, h_grid, w_grid = image_grid_thw[0].tolist()
            llm_h = h_grid // spatial_merge_size
            llm_w = w_grid // spatial_merge_size
            llm_t = t_grid  # 1 for single image

            text_start = text_before + st_idx  # position offset
            for idx, tok_pos in enumerate(img_token_positions):
                # idx maps to (t_i, h_i, w_i) in the merged grid
                t_i = idx // (llm_h * llm_w)
                hw_i = idx % (llm_h * llm_w)
                h_i = hw_i // llm_w
                w_i = hw_i % llm_w
                pos_t[tok_pos] = text_start  # temporal: constant for single image
                pos_h[tok_pos] = text_start + h_i
                pos_w[tok_pos] = text_start + w_i

            # Text after image: resume from max image position + 1
            max_img_pos = max(pos_t[img_token_positions].max(),
                              pos_h[img_token_positions].max(),
                              pos_w[img_token_positions].max()) + 1
            img_last = img_token_positions[-1]
            for i in range(img_last + 1, seq_len):
                offset = i - (img_last + 1)
                pos_t[i] = max_img_pos + offset
                pos_h[i] = max_img_pos + offset
                pos_w[i] = max_img_pos + offset
        else:
            # No image — uniform positions
            for i in range(seq_len):
                pos_t[i] = i
                pos_h[i] = i
                pos_w[i] = i

        D = self.actual_head_dim // 2
        inv_freq = self._rope_inv_freq  # [64]
        # Split inv_freq into 3 sections [16, 24, 24]
        s0, s1, s2 = mrope_sections  # [16, 24, 24]
        inv_sec0 = inv_freq[:s0]      # dims [0:16]
        inv_sec1 = inv_freq[s0:s0+s1]  # dims [16:40]
        inv_sec2 = inv_freq[s0+s1:]    # dims [40:64]

        # Compute per-token frequencies
        freqs = torch.zeros(seq_len, D, dtype=torch.float32)
        freqs[:, :s0]       = pos_t.float().unsqueeze(1) * inv_sec0.unsqueeze(0)
        freqs[:, s0:s0+s1]  = pos_h.float().unsqueeze(1) * inv_sec1.unsqueeze(0)
        freqs[:, s0+s1:]    = pos_w.float().unsqueeze(1) * inv_sec2.unsqueeze(0)

        cos_ = freqs.cos().to(torch.bfloat16)
        sin_ = freqs.sin().to(torch.bfloat16)
        # rope_hf_core(N=128): [cos(64), cos(64), -sin(64), sin(64)]
        rope_tensor = torch.cat([cos_, cos_, -sin_, sin_], dim=1)  # [seq_len, 256]

        # Overwrite the first seq_len rows of the DRAM RoPE table
        rope_bytes = rope_tensor.contiguous().view(torch.uint8).numpy().tobytes()
        self.dma_write(DMA_DEVICE_H2C, self.DRAM_ADDR_ROPE, rope_bytes, len(rope_bytes))

        # Compute rope_delta: decoder position = seq_len + rope_delta
        # (image tokens span fewer positions than their count, so delta is negative)
        max_pos = max(pos_t.max().item(), pos_h.max().item(), pos_w.max().item())
        self._mrope_delta = int(max_pos + 1 - seq_len)
        _original_print(f"    mRoPE table [{seq_len}, 256] written to DRAM (sections {mrope_sections}, rope_delta={self._mrope_delta})")

    def restore_rope_for_decoder(self) -> None:
        """Restore the original position-indexed RoPE table after mRoPE prefill.
        The decoder needs sequential positions, not the per-token mRoPE table."""
        if hasattr(self, '_rope_table_bytes'):
            self.dma_write(DMA_DEVICE_H2C, self.DRAM_ADDR_ROPE,
                           self._rope_table_bytes, len(self._rope_table_bytes))

    def weight_init(self) -> None:
        """Load LM + vision weights from bin files to DRAM."""
        from transformers import AutoTokenizer
        model_dir = os.path.join(self.script_dir, self._cfg["paths"]["hf_model_dir"])
        hf_repo = self._cfg["paths"]["hf_model_repo"]
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            _original_print(f"Downloading HF model {hf_repo} to {os.path.abspath(model_dir)} ...")
            snapshot_download(repo_id=hf_repo, local_dir=model_dir, local_dir_use_symlinks=False)
            _original_print("Download complete.")

        # Generate weight bins if missing
        lm_bin_path = os.path.join(self.script_dir, self._cfg["paths"]["lm_weights"])
        vis_bin_path = os.path.join(self.script_dir, self._cfg["paths"]["vision_weights"])
        if not os.path.exists(lm_bin_path) or not os.path.exists(vis_bin_path):
            _original_print("Weight bins not found, generating...")
            weight_bin_generate(script_dir=self.script_dir, output_lm=lm_bin_path, output_vis=vis_bin_path)

        # Load LM weights
        print(f"  Reading LM weight bin (memmap)...", flush=True)
        lm_cache = load_weight_cache(lm_bin_path)

        # Embedding (host-side only — NOT uploaded to FPGA DRAM)
        # Embedding lookup is done on host in get_embedding_for_tokens()
        embed_raw = lm_cache['language_model.embed_tokens.weight']
        embed_bf16 = torch.from_numpy(embed_raw.copy()).view(torch.bfloat16).reshape(self.EMBEDDING_ELEMENTS, self.vector_length)
        self.embedding_weight = embed_bf16.clone()  # host copy for token lookup

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        # Per-layer weights — config-driven precision (precision.lm) for
        # Q/K/gate/up/down, BF16 for V/O (all from binary). Default 'if4'
        # picks INT4 vs FP4 per 64-block by min weight-MSE; near-lossless
        # vs BF16 on this LM (PPL +0.9% on WikiText-2 vs +7.9% for pure FP4
        # and +17% for pure INT4 — see src/models/qwen2.5_VL_3b/compare/summary.md).
        # V/O stored as BF16 in binary for better attention accuracy.
        lm_prec = _lm_precision(self._cfg)
        print(f"  Loading {self.LAYER_SIZE} LM layers to FPGA DRAM ({lm_prec})...", flush=True)
        self.lm_layer_addrs = []
        for i in range(self.LAYER_SIZE):
            if i and (i % 8 == 0 or i == self.LAYER_SIZE - 1):
                print(f"    layer {i + 1}/{self.LAYER_SIZE} loaded", flush=True)
            la = {}
            prefix = f'language_model.layers.{i}'
            # Q, K, gate, up, down: precision.lm-quantized from binary
            for proj, hf_sub in [('q', 'self_attn.q_proj'), ('k', 'self_attn.k_proj'),
                                  ('gate', 'mlp.gate_proj'), ('up', 'mlp.up_proj'),
                                  ('down', 'mlp.down_proj')]:
                la[f'{proj}_scale'], la[f'{proj}_data'] = store_quantized_weight(self, lm_cache[f'{prefix}.{hf_sub}.weight.{lm_prec}'])
            # V and O as BF16 from binary
            la['v_weight'] = store_weight(self, torch.from_numpy(lm_cache[f'{prefix}.self_attn.v_proj.weight'].copy()).view(torch.bfloat16))
            la['o_weight'] = store_weight(self, torch.from_numpy(lm_cache[f'{prefix}.self_attn.o_proj.weight'].copy()).view(torch.bfloat16))
            # Biases from binary
            for proj, hf_sub in [('q', 'self_attn.q_proj'), ('k', 'self_attn.k_proj'),
                                  ('v', 'self_attn.v_proj')]:
                bias_key = f'{prefix}.{hf_sub}.bias'
                if bias_key in lm_cache:
                    la[f'{proj}_bias'] = store_weight(self, torch.from_numpy(lm_cache[bias_key].copy()).view(torch.bfloat16))
                else:
                    bias_size = self.vector_length if proj == 'q' else self.head_dim
                    la[f'{proj}_bias'] = store_weight(self, torch.zeros(bias_size, dtype=torch.bfloat16))
            # Norm weights from binary
            la['ln1_gamma'] = store_weight(self, torch.from_numpy(lm_cache[f'{prefix}.input_layernorm.weight'].copy()).view(torch.bfloat16))
            la['ln2_gamma'] = store_weight(self, torch.from_numpy(lm_cache[f'{prefix}.post_attention_layernorm.weight'].copy()).view(torch.bfloat16))
            self.lm_layer_addrs.append(la)

        # NOTE: hf_model kept alive for vision MLP BF16 weight extraction below

        # Final norm
        self.final_norm_addr = store_weight(self, torch.from_numpy(lm_cache['language_model.norm.weight'].copy()).view(torch.bfloat16))

        # LM head: pre-quantized at bin-gen time (tied to embedding) and
        # loaded directly from the bin. No runtime quantization -- if the
        # entry is missing, the bin is stale and must be regenerated.
        lm_head_key = f'lm_head.weight.{lm_prec}'
        if lm_head_key not in lm_cache:
            raise RuntimeError(
                f"LM head entry {lm_head_key!r} not found in bin "
                f"{lm_bin_path!r}. The bin was generated by an older "
                f"version of the script. Delete the bin and rerun to "
                f"regenerate with the LM head pre-quantized."
            )
        self.lm_head_scale, self.lm_head_data = store_quantized_weight(
            self, lm_cache[lm_head_key])

        # Identity matrix for decode attention
        self.identity_addr = store_identity_matrix(self)

        # RoPE
        self._load_rope_host()

        # Vision weights (all from binary — pre-padded QKV and MLP)
        self._load_vision_weights(vis_bin_path)

        print(f"    Allocate weights end at DRAM address: 0x{self.get_params_dram_addr():X}, usage: {self.get_params_dram_usage()} bytes")
        print("Tokenizer loaded successfully.")

    def _load_vision_weights(self, vis_bin_path: str, hf_model=None) -> None:
        """Load vision encoder weights from bin file. Manifest-key suffix
        comes from cfg['precision']['vision'] — same string the generator
        wrote, so changing precision in the config switches both ends."""
        vis_cache = load_weight_cache(vis_bin_path)
        vis_cfg = self._vis_cfg
        vis_depth = vis_cfg["depth"]
        VI = vis_cfg["intermediate_size"]
        VH = vis_cfg["hidden_size"]
        VIS_MLP_PAD = ((VI + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE

        VN = vis_cfg["num_heads"]
        VD = vis_cfg["head_dim"]
        VD_PAD = 128                                # padded head_dim (must be % 64 == 0)

        vis_prec = _vision_precision(self._cfg)
        print(f"  Loading {vis_depth} vision encoder layers to FPGA DRAM ({vis_prec})...", flush=True)
        self.vis_layer_addrs = []
        for i in range(vis_depth):
            if i and (i % 8 == 0 or i == vis_depth - 1):
                print(f"    layer {i + 1}/{vis_depth} loaded", flush=True)
            la = {}
            prefix = f'visual.blocks.{i}'
            # Q/K (rearranged padded) + V (sequential padded) — pre-padded in binary
            la['qk_scale'], la['qk_data'] = store_quantized_weight(self, vis_cache[f'{prefix}.attn.qk_padded.weight.{vis_prec}'])
            la['qk_bias'] = store_weight(self, torch.from_numpy(vis_cache[f'{prefix}.attn.qk_padded.bias'].copy()).view(torch.bfloat16))
            la['v_scale'], la['v_data'] = store_quantized_weight(self, vis_cache[f'{prefix}.attn.v_padded.weight.{vis_prec}'])
            la['v_bias'] = store_weight(self, torch.from_numpy(vis_cache[f'{prefix}.attn.v_padded.bias'].copy()).view(torch.bfloat16))
            # O proj
            la['o_scale'], la['o_data'] = store_quantized_weight(self, vis_cache[f'{prefix}.attn.proj.weight.{vis_prec}'])
            la['o_bias'] = store_weight(self, torch.from_numpy(vis_cache[f'{prefix}.attn.proj.bias'].copy()).view(torch.bfloat16))
            # MLP (pre-padded 3420→3456 in binary)
            for proj, hf_sub in [('gate', 'mlp.gate_proj'), ('up', 'mlp.up_proj'), ('down', 'mlp.down_proj')]:
                la[f'{proj}_scale'], la[f'{proj}_data'] = store_quantized_weight(self, vis_cache[f'{prefix}.{hf_sub}.weight.{vis_prec}'])
                la[f'{proj}_bias'] = store_weight(self, torch.from_numpy(vis_cache[f'{prefix}.{hf_sub}.bias'].copy()).view(torch.bfloat16))
            # RMSNorm (no bias)
            la['norm1_weight'] = store_weight(self, torch.from_numpy(vis_cache[f'{prefix}.norm1.weight'].copy()).view(torch.bfloat16))
            la['norm2_weight'] = store_weight(self, torch.from_numpy(vis_cache[f'{prefix}.norm2.weight'].copy()).view(torch.bfloat16))
            self.vis_layer_addrs.append(la)

        # Patch embedding (Conv3D weight [1280, 3, 2, 14, 14]); precision per cfg.
        self.patch_weight_scale, self.patch_weight_data = store_quantized_weight(self, vis_cache[f'visual.patch_embed.proj.weight.{vis_prec}'])

        # Merger
        self.merger_ln_q_weight = store_weight(self, torch.from_numpy(vis_cache['visual.merger.ln_q.weight'].copy()).view(torch.bfloat16))
        # Merger MLP — quantized per cfg.precision.vision.
        self.merger_mlp0_scale, self.merger_mlp0_data = store_quantized_weight(self, vis_cache[f'visual.merger.mlp.0.weight.{vis_prec}'])
        self.merger_mlp0_bias = store_weight(self, torch.from_numpy(vis_cache['visual.merger.mlp.0.bias'].copy()).view(torch.bfloat16))
        self.merger_mlp2_scale, self.merger_mlp2_data = store_quantized_weight(self, vis_cache[f'visual.merger.mlp.2.weight.{vis_prec}'])
        self.merger_mlp2_bias = store_weight(self, torch.from_numpy(vis_cache['visual.merger.mlp.2.bias'].copy()).view(torch.bfloat16))

        print(f"Vision weights loaded ({vis_prec} + padded MLP to {VIS_MLP_PAD}): {vis_depth} layers + merger, params DRAM usage: {self.get_params_dram_usage()} bytes")

    def tensor_init(self) -> None:
        """Allocate DRAM tensors for LM + vision intermediates."""
        seq_len = self.MAX_CONTEXT_SIZE
        # Qwen2.5-VL: q_seq_len = seq_len * group_size (8 Q heads per KV head)
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        ahd = self.actual_head_dim
        nkvh = self.num_kv_heads
        bpe = self.bytes_per_element

        print(f"Allocate tensor dram start at DRAM address: 0x{self.get_tensor_dram_addr():X}")

        zero_add = torch.zeros(seq_len * self.head_dim, dtype=torch.bfloat16)
        self.ZERO_DRAM_ADDR = self.allocate_tensor_dram(seq_len * self.head_dim * bpe)
        self.dma_to_accelerator_memory(self.ZERO_DRAM_ADDR, zero_add)
        self.IDENTITY_DRAM_ADDR = self.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe)
        self.dma_to_accelerator_memory(self.IDENTITY_DRAM_ADDR, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))

        # Per-head flash attention buffers: one KV head at a time, reused across heads
        self.LAYER0_FLASH_Q_DRAM = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        self.LAYER0_FLASH_K_DRAM = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        self.LAYER0_FLASH_V_DRAM = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        zero_pad = torch.zeros(aligned_seq_len * ahd, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_Q_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_K_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_V_DRAM, zero_pad)

        # Per-head flash output (seq_len * group_size, ahd); reused across KV heads
        self.LAYER0_FLASH_OUT_HEAD_DRAM = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        # Final assembled flash output (seq_len, head_dim * group_size) = (seq_len, 2048)
        self.LAYER0_FLASH_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.head_dim * self.group_size * bpe)
        self.LAYER0_FLASH_SCRATCH_DRAM = self.allocate_tensor_dram(max(ahd, UE_FMAX_CONTEXT_SIZE) * aligned_seq_len * 2 + ahd * aligned_seq_len * 2)
        aligned_tok = ((seq_len + 63) // 64) * 64
        self.LAYER0_FLASH_BIAS_DRAM = self.allocate_tensor_dram(aligned_tok * aligned_tok * bpe)

        # Layer intermediate tensors
        self.LAYER0_INPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_Q_DRAM = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_K_DRAM = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_K_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_Q_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.q_size)
        # V proj temp buffer
        self.LAYER0_V_PROJ_TEMP = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_ATTN_PROJ_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_ATTN_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_ATTN_RESIDUAL_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_MLP_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_MLP_GATE_DRAM = self.allocate_tensor_dram(seq_len * self.mlp_elements * 2)
        self.LAYER0_MLP_UP_DRAM = self.allocate_tensor_dram(seq_len * self.mlp_elements * 2)
        self.LAYER0_MLP_MULT_DRAM = self.allocate_tensor_dram(seq_len * self.mlp_elements * 2)
        self.LAYER0_MLP_DOWN_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_MLP_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.OUTPUT_NORM_DRAM = self.allocate_tensor_dram(1 * self.vector_length * bpe)
        self.LOGITS_DRAM = self.allocate_tensor_dram(1 * self.EMBEDDING_ELEMENTS * bpe)

        vis_cfg = self._vis_cfg
        self.NUM_MERGED_TOKENS = vis_cfg.get("num_merged_tokens", 256)
        VS = vis_cfg["num_patches"]
        VH = vis_cfg["hidden_size"]
        VN = vis_cfg["num_heads"]
        VD = vis_cfg["head_dim"]
        VI = vis_cfg["intermediate_size"]
        VIS_MLP_PAD = ((VI + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
        VD_PAD = 128                           # padded head_dim for flash attention (must be % 64 == 0)
        VH_OUT = vis_cfg["out_hidden_size"]
        VMERGE = vis_cfg["spatial_merge_size"]

        # Patch embedding input (host runs Conv3D, DMAs result here)
        self.VIS_PATCH_INPUT_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        # Double-buffered layer I/O
        self.VIS_IO_A_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        self.VIS_IO_B_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        # RMSNorm output
        self.VIS_NORM_OUT_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        # Q/K output [VS, 4096] (padded: 2 * VN * VD_PAD, FP4 matmul)
        self.VIS_QK_DRAM = self.allocate_tensor_dram(VS * VN * VD_PAD * 2 * bpe)
        # V output [VS, 2048] (padded: VN * VD_PAD, BF16 matmul)
        self.VIS_V_DRAM = self.allocate_tensor_dram(VS * VN * VD_PAD * bpe)
        # Padded Q/K/V for flash attention [16, VS, 128] — zero-filled at init
        self.VIS_Q_PAD_DRAM = self.allocate_tensor_dram(VN * VS * VD_PAD * bpe)
        self.VIS_K_PAD_DRAM = self.allocate_tensor_dram(VN * VS * VD_PAD * bpe)
        self.VIS_V_PAD_DRAM = self.allocate_tensor_dram(VN * VS * VD_PAD * bpe)
        vis_pad_zeros = torch.zeros(VN * VS * VD_PAD, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.VIS_Q_PAD_DRAM, vis_pad_zeros)
        self.dma_to_accelerator_memory(self.VIS_K_PAD_DRAM, vis_pad_zeros)
        self.dma_to_accelerator_memory(self.VIS_V_PAD_DRAM, vis_pad_zeros)
        # Flash attention output (padded) [16, 1024, 128]
        self.VIS_ATTN_OUT_DRAM = self.allocate_tensor_dram(VN * VS * VD_PAD * bpe)
        # Flash attention scratch
        self.VIS_ATTN_SCRATCH_DRAM = self.allocate_tensor_dram(
            (max(VD_PAD, UE_FMAX_CONTEXT_SIZE) * VS * 2 + VD_PAD * VS * 2))
        # Attention result after inverse permute [1024, 1280]
        self.VIS_ATTN_RESULT_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        # Unpermuted attention output (trimmed from 128 back to 80) [16, 1024, 80]
        self.VIS_ATTN_TRIM_DRAM = self.allocate_tensor_dram(VN * VS * VD * bpe)
        # O projection output
        self.VIS_O_PROJ_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        # Residual buffer
        self.VIS_RESIDUAL_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        # SwiGLU MLP buffers (padded intermediate_size for 64-alignment)
        self.VIS_MLP_GATE_DRAM = self.allocate_tensor_dram(VS * VIS_MLP_PAD * bpe)
        self.VIS_MLP_UP_DRAM = self.allocate_tensor_dram(VS * VIS_MLP_PAD * bpe)
        self.VIS_MLP_MULT_DRAM = self.allocate_tensor_dram(VS * VIS_MLP_PAD * bpe)
        self.VIS_MLP_DOWN_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        # Post-layers: merger RMSNorm output
        self.VIS_POST_NORM_DRAM = self.allocate_tensor_dram(VS * VH * bpe)
        # Spatial merge output [256, 5120]
        merge_dim = VMERGE * VMERGE * VH  # 2*2*1280 = 5120
        self.VIS_MERGED_DRAM = self.allocate_tensor_dram(self.NUM_MERGED_TOKENS * merge_dim * bpe)
        # Merger MLP intermediate [256, 5120]
        self.VIS_MERGER_INTER_DRAM = self.allocate_tensor_dram(self.NUM_MERGED_TOKENS * merge_dim * bpe)
        # Final encoder output [256, 2048]
        self.VIS_ENCODER_OUT_DRAM = self.allocate_tensor_dram(self.NUM_MERGED_TOKENS * VH_OUT * bpe)
        # 2D RoPE cos/sin tables (computed on host per image, DMA'd before encoder run)
        # Shape: [1024, 128] each — padded from 80 to 128 for aligned rope_hf_core
        self.VIS_ROPE_COS_DRAM = self.allocate_tensor_dram(VS * VD_PAD * bpe)
        self.VIS_ROPE_SIN_DRAM = self.allocate_tensor_dram(VS * VD_PAD * bpe)
        # Permute params (identity matrix for bf16_permute_core_v2)
        permute_size = UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe
        self.VIS_PERMUTE_PARAMS_DRAM = self.get_params_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, self.VIS_PERMUTE_PARAMS_DRAM,
                       torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16), permute_size)
        self.allocate_params_dram(permute_size)
        # Temp space for permute decomposition (needs at least 2x largest tensor)
        self.VIS_PERMUTE_TEMP_DRAM = self.allocate_tensor_dram(VN * VS * VD_PAD * bpe * 2)

        kv_cache_elems = self.LAYER_SIZE * nkvh * self.MAX_CONTEXT_SIZE * ahd
        kv_cache_bytes = kv_cache_elems * bpe
        self.LAYER0_V_DRAM = self.allocate_tensor_dram(kv_cache_bytes)
        self.LAYER0_K_ROPE_DRAM = self.allocate_tensor_dram(kv_cache_bytes)
        zero_kv = torch.zeros(kv_cache_elems, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, zero_kv)
        self.dma_to_accelerator_memory(self.LAYER0_K_ROPE_DRAM, zero_kv)

        print(f"    Allocate tensor dram end at DRAM address: 0x{self.get_tensor_dram_addr():X}, usage: {self.get_tensor_dram_usage()} bytes")
        # Pre-allocate the identity matrix used by flash_attention_core
        self._preallocate_identity_matrix()

    def _corrected_hw_latency_us(self, wall_time_s: float) -> float:
        """Read HW cycle counter and correct for 32-bit overflow.
        Counter wraps at 2^32 * cycle_ns = ~24.18s at 5.63ns."""
        raw_us = self.report_latency_in_us()
        overflow_us = (2**32) * self._clock_period_ns / 1e3
        # Add overflows until within reasonable range of wall time
        corrected = raw_us
        while corrected + overflow_us / 2 < wall_time_s * 1e6:
            corrected += overflow_us
        return corrected

    @staticmethod
    def _wait_with_heartbeat(wait_fn, label: str, interval_s: float = 5.0):
        """Call ``wait_fn`` (blocking) on a background thread and print
        ``[label] still running ({n}s)`` every ``interval_s`` so the demo
        doesn't look frozen during long HW executes. Returns wall time."""
        import threading
        stop = threading.Event()
        def _beat():
            n = 0
            while not stop.wait(interval_s):
                n += interval_s
                print(f"      [{label}] still running ({n:.0f}s)...", flush=True)
        t = threading.Thread(target=_beat, daemon=True)
        t0 = time.perf_counter()
        t.start()
        try:
            wait_fn()
        finally:
            stop.set()
            t.join(timeout=interval_s)
        return time.perf_counter() - t0

    def program_execute(self, program_start_addr: int = user_dma_core.DRAM_INSTRUCTION_ADDR, timeout: float = 120.0, gflops: float = None) -> None:
        """Execute compiled program from DRAM instruction memory."""
        print(f"  Running program on FPGA (DRAM addr 0x{program_start_addr:X})...", flush=True)
        self.start_execute_from_dram(program_start_addr)
        wall_s = self._wait_with_heartbeat(lambda: self.wait_queue(timeout), label="FPGA")
        latency_us = self._corrected_hw_latency_us(wall_s)
        print(f"    Total program execution latency = {latency_us:.0f} us")
        if gflops is not None:
            gflops_program = gflops / (latency_us * 1e-6) / 1e9
            self._last_hw_gflops = gflops_program
            self._last_total_flops = gflops
            print(f"    Throughput: {gflops_program:.2f} GFLOPS")

    def compile_encoder(self, num_layers: int = None) -> int:
        """Compile vision encoder FPGA program. Returns program DRAM address."""
        from user_dma_core import TYPE

        vis_cfg = self._vis_cfg
        VS = vis_cfg["num_patches"]
        VH = vis_cfg["hidden_size"]
        VN = vis_cfg["num_heads"]
        VD = vis_cfg["head_dim"]
        VI = vis_cfg["intermediate_size"]
        VIS_MLP_PAD = ((VI + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
        total_layers = num_layers if num_layers is not None else vis_cfg["depth"]
        VD_PAD = 128
        VH_OUT = vis_cfg["out_hidden_size"]
        VMERGE = vis_cfg["spatial_merge_size"]
        bpe = 2

        head_stride_pad = VS * VD_PAD * bpe
        head_stride_raw = VS * VD * bpe

        permute_dims_merge = [VN, VS, VD]        # [16, 576, 80] for head merge
        permute_idx = [1, 0, 2]                  # swap dim 0 and dim 1

        _bf16_mode = getattr(self, '_vision_bf16_mode', False)
        bin_dir = os.path.join(self.script_dir, "qwen2.5_vl_3b_bin")
        os.makedirs(bin_dir, exist_ok=True)
        cache_suffix = f"_{total_layers}L" if total_layers != vis_cfg["depth"] else ""
        if _bf16_mode:
            cache_suffix += "_bf16"
        enc_bin_path = os.path.join(bin_dir, f"encoder_program{cache_suffix}.bin")

        if os.path.exists(enc_bin_path):
            _original_print(f"  Loading cached encoder program from {enc_bin_path} ...")
            with open(enc_bin_path, "rb") as f:
                prog_bytes = f.read()
            program_addr = self.get_program_dram_addr()
            self.dma_write(DMA_DEVICE_H2C, program_addr, prog_bytes, len(prog_bytes))
            self.allocate_program_dram(len(prog_bytes))
            self._vis_program_addr = program_addr
            _original_print(f"    Loaded {len(prog_bytes)} bytes at 0x{program_addr:X}")
            return program_addr

        global _SILENT_MODE
        _SILENT_MODE = True

        _bf16_mode = getattr(self, '_vision_bf16_mode', False)
        def vis_matmul(M, K, N, A, la_key_prefix, la, OUT, bias=None, silu_en=False, gelu_en=False):
            if _bf16_mode and f'{la_key_prefix}_weight_bf16' in la:
                self.matmat_mul_core(
                    M=M, K=K, N=N, A_DRAM_ADDR=A,
                    B_DRAM_ADDR=la[f'{la_key_prefix}_weight_bf16'],
                    OUTPUT_DRAM_ADDR=OUT,
                    C_DRAM_ADDR=la.get(f'{la_key_prefix}_bias_bf16', bias), bias_mode="broadcast_N",
                    silu_enable=silu_en, gelu_enable=gelu_en)
            else:
                self.matmat_mul_core(
                    M=M, K=K, N=N, A_DRAM_ADDR=A,
                    B_DRAM_ADDR=la[f'{la_key_prefix}_data'],
                    OUTPUT_DRAM_ADDR=OUT,
                    C_DRAM_ADDR=bias, bias_mode="broadcast_N",
                    is_B_quantized=True, data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=la[f'{la_key_prefix}_scale'],
                    silu_enable=silu_en, gelu_enable=gelu_en)

        total_bytes = 0

        _original_print(f"  Compiling vision encoder: {total_layers} layers (single program)...")
        self.start_capture()

        for layer_idx in range(total_layers):
            _original_print(f"    vision layer {layer_idx + 1}/{total_layers}", end="\r", flush=True)
            la = self.vis_layer_addrs[layer_idx]
            h_in  = self.VIS_IO_A_DRAM if layer_idx % 2 == 0 else self.VIS_IO_B_DRAM
            h_out = self.VIS_IO_B_DRAM if layer_idx % 2 == 0 else self.VIS_IO_A_DRAM

            # Layer 0 copies patch input to IO_A
            if layer_idx == 0:
                for chunk_start in range(0, VS * VH, URAM_NEAR_FULL_ELEMENTS):
                    chunk_take = min(URAM_NEAR_FULL_ELEMENTS, VS * VH - chunk_start)
                    self.accelerator_memory_to_sram(
                        accelerator_dram_address=self.VIS_PATCH_INPUT_DRAM + chunk_start * bpe,
                        sram_address=0x00000, element_size=chunk_take)
                    self.sram_to_accelerator_memory(
                        sram_address=0x00000,
                        accelerator_dram_address=self.VIS_IO_A_DRAM + chunk_start * bpe,
                        element_size=chunk_take)

            self.rms_norm_core_dram(M=VS, N=VH,
                A_DRAM_ADDR=h_in, OUTPUT_DRAM_ADDR=self.VIS_NORM_OUT_DRAM,
                GAMMA_DRAM_ADDR=la['norm1_weight'])

            VN_VD_PAD = VN * VD_PAD
            if _bf16_mode and 'qk_weight_bf16' in la:
                self.matmat_mul_core(
                    M=VS, K=VH, N=VN_VD_PAD * 2,
                    A_DRAM_ADDR=self.VIS_NORM_OUT_DRAM,
                    B_DRAM_ADDR=la['qk_weight_bf16'],
                    OUTPUT_DRAM_ADDR=self.VIS_QK_DRAM,
                    C_DRAM_ADDR=la['qk_bias_bf16'], bias_mode="broadcast_N")
            else:
                self.matmat_mul_core(
                    M=VS, K=VH, N=VN_VD_PAD * 2,
                    A_DRAM_ADDR=self.VIS_NORM_OUT_DRAM,
                    B_DRAM_ADDR=la['qk_data'],
                    OUTPUT_DRAM_ADDR=self.VIS_QK_DRAM,
                    C_DRAM_ADDR=la['qk_bias'], bias_mode="broadcast_N",
                    is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=la['qk_scale'])

            if _bf16_mode and 'v_weight_bf16' in la:
                self.matmat_mul_core(
                    M=VS, K=VH, N=VN_VD_PAD,
                    A_DRAM_ADDR=self.VIS_NORM_OUT_DRAM,
                    B_DRAM_ADDR=la['v_weight_bf16'],
                    OUTPUT_DRAM_ADDR=self.VIS_V_DRAM,
                    C_DRAM_ADDR=la['v_bias_bf16'], bias_mode="broadcast_N")
            else:
                self.matmat_mul_core(
                    M=VS, K=VH, N=VN_VD_PAD,
                    A_DRAM_ADDR=self.VIS_NORM_OUT_DRAM,
                    B_DRAM_ADDR=la['v_data'],
                    OUTPUT_DRAM_ADDR=self.VIS_V_DRAM,
                    C_DRAM_ADDR=la['v_bias'], bias_mode="broadcast_N",
                    is_B_quantized=True, data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=la['v_scale'])

            qk_row_pad = VN_VD_PAD * 2
            extract_buf = self.VIS_PERMUTE_TEMP_DRAM
            perm_temp = self.VIS_PERMUTE_TEMP_DRAM + VS * VN_VD_PAD * bpe

            # Extract Q/K from interleaved QK output using strided DMA
            ext_chunk = URAM_NEAR_FULL_ELEMENTS // VN_VD_PAD
            for proj_idx, (proj_col_off, pad_dst) in enumerate([
                (0,              self.VIS_Q_PAD_DRAM),
                (VN_VD_PAD,      self.VIS_K_PAD_DRAM),
            ]):
                for t_start in range(0, VS, ext_chunk):
                    t_take = min(ext_chunk, VS - t_start)
                    src = self.VIS_QK_DRAM + t_start * qk_row_pad * bpe + proj_col_off * bpe
                    dst = extract_buf + t_start * VN_VD_PAD * bpe
                    self.accelerator_memory_to_sram(
                        src, 0x00000, t_take * VN_VD_PAD,
                        stride_bytes_per_chunk=VN_VD_PAD * bpe,
                        stride_jump_bytes=qk_row_pad * bpe)
                    self.sram_to_accelerator_memory(0x00000, dst, t_take * VN_VD_PAD)

                bf16_permute_core_v2(self,
                    dims=[VS, VN, VD_PAD], permute_indices=[1, 0, 2],
                    input_dram_addr=extract_buf,
                    output_dram_addr=pad_dst,
                    params_dram_addr=self.VIS_PERMUTE_PARAMS_DRAM,
                    temp_dram_start=perm_temp)

            bf16_permute_core_v2(self,
                dims=[VS, VN, VD_PAD], permute_indices=[1, 0, 2],
                input_dram_addr=self.VIS_V_DRAM,
                output_dram_addr=self.VIS_V_PAD_DRAM,
                params_dram_addr=self.VIS_PERMUTE_PARAMS_DRAM,
                temp_dram_start=perm_temp)

            # Batched RoPE: load all tokens for one head, bulk eltwise for x*cos,
            # then per-token cross-terms (x_hi*sin_lo, x_lo*sin_hi), store all at once.
            half = VD_PAD // 2
            q_all_elems = VS * VD_PAD
            # SRAM layout: URAM_A has Q + result buffer, URAM_B has cos + sin
            sram_q   = 0x00000                         # Q/K [VS, 128] = 73728 elems (URAM_A)
            sram_a   = sram_q + q_all_elems * bpe      # x*cos result [VS, 128] (URAM_A)
            sram_cos = 0x80000                         # cos [VS, 128] (URAM_B)
            sram_sin = sram_cos + q_all_elems * bpe    # sin [VS, 128] (URAM_B)
            sram_bc  = sram_sin + q_all_elems * bpe    # cross-term temp [128] (URAM_B)
            # Load cos/sin once per layer
            self.accelerator_memory_to_sram(self.VIS_ROPE_COS_DRAM, sram_cos, q_all_elems)
            self.accelerator_memory_to_sram(self.VIS_ROPE_SIN_DRAM, sram_sin, q_all_elems)
            for h in range(VN):
                for buf_dram in (self.VIS_Q_PAD_DRAM, self.VIS_K_PAD_DRAM):
                    head_addr = buf_dram + h * head_stride_pad
                    self.accelerator_memory_to_sram(head_addr, sram_q, q_all_elems)
                    # Bulk: a = x * cos (all 73728 elements at once)
                    self.eltwise_mul_core(sram_q, sram_cos, sram_a, q_all_elems)
                    # Per-token cross-terms and accumulate into a
                    for t in range(VS):
                        t_off = t * VD_PAD * bpe
                        sx = sram_q + t_off
                        ss = sram_sin + t_off
                        sa = sram_a + t_off
                        # bc_lo = x_hi * sin_lo; bc_hi = x_lo * sin_hi
                        self.eltwise_mul_core(sx + half * bpe, ss, sram_bc, half)
                        self.eltwise_mul_core(sx, ss + half * bpe, sram_bc + half * bpe, half)
                        # a += bc (result in sram_a, overwrites x*cos with final RoPE)
                        self.eltwise_add_core(sa, sram_bc, sa, VD_PAD)
                    self.sram_to_accelerator_memory(sram_a, head_addr, q_all_elems)

            # flash_attention_core scales by 1/sqrt(128) instead of 1/sqrt(80).
            # The softer attention (sqrt(128)) is more robust to INT4 quantization noise
            # and empirically gives better end-to-end results than correcting to sqrt(80).
            fullatt_indexes = set(vis_cfg.get("fullatt_block_indexes", [7, 15, 23, 31]))
            cu_win = getattr(self, '_cu_window_seqlens', [0, VS])
            is_full_attn = (layer_idx in fullatt_indexes)

            if is_full_attn:
                for h in range(VN):
                    self._flash_attention_core_cached(
                        head_dim=VD_PAD,
                        seq_len=VS,
                        Q_DRAM_ADDR=self.VIS_Q_PAD_DRAM + h * head_stride_pad,
                        K_DRAM_ADDR=self.VIS_K_PAD_DRAM + h * head_stride_pad,
                        V_DRAM_ADDR=self.VIS_V_PAD_DRAM + h * head_stride_pad,
                        OUTPUT_DRAM_ADDR=self.VIS_ATTN_OUT_DRAM + h * head_stride_pad,
                        SCRATCH_DRAM_ADDR=self.VIS_ATTN_SCRATCH_DRAM)
            else:
                for h in range(VN):
                    for w_idx in range(len(cu_win) - 1):
                        w_start = cu_win[w_idx]
                        w_len = cu_win[w_idx + 1] - w_start
                        if w_len <= 0:
                            continue
                        w_off = w_start * VD_PAD * bpe  # byte offset within head
                        self._flash_attention_core_cached(
                            head_dim=VD_PAD,
                            seq_len=w_len,
                            Q_DRAM_ADDR=self.VIS_Q_PAD_DRAM + h * head_stride_pad + w_off,
                            K_DRAM_ADDR=self.VIS_K_PAD_DRAM + h * head_stride_pad + w_off,
                            V_DRAM_ADDR=self.VIS_V_PAD_DRAM + h * head_stride_pad + w_off,
                            OUTPUT_DRAM_ADDR=self.VIS_ATTN_OUT_DRAM + h * head_stride_pad + w_off,
                            SCRATCH_DRAM_ADDR=self.VIS_ATTN_SCRATCH_DRAM)

            # Trim: extract first 80 from each 128-element row. Batch per head using strided DMA.
            trim_chunk = URAM_NEAR_FULL_ELEMENTS // VD  # rows per SRAM batch
            for h in range(VN):
                src_head = self.VIS_ATTN_OUT_DRAM + h * head_stride_pad
                dst_head = self.VIS_ATTN_TRIM_DRAM + h * head_stride_raw
                for t_start in range(0, VS, trim_chunk):
                    t_take = min(trim_chunk, VS - t_start)
                    self.accelerator_memory_to_sram(
                        src_head + t_start * VD_PAD * bpe, 0x00000, t_take * VD,
                        stride_bytes_per_chunk=VD * bpe,
                        stride_jump_bytes=VD_PAD * bpe)
                    self.sram_to_accelerator_memory(
                        0x00000, dst_head + t_start * VD * bpe, t_take * VD)

            bf16_permute_core_v2(self,
                dims=permute_dims_merge, permute_indices=permute_idx,
                input_dram_addr=self.VIS_ATTN_TRIM_DRAM,
                output_dram_addr=self.VIS_ATTN_RESULT_DRAM,
                params_dram_addr=self.VIS_PERMUTE_PARAMS_DRAM,
                temp_dram_start=self.VIS_PERMUTE_TEMP_DRAM)

            vis_matmul(VS, VH, VH, self.VIS_ATTN_RESULT_DRAM, 'o', la,
                       self.VIS_O_PROJ_DRAM, bias=la['o_bias'])

            rms_norm_core_dram_post_add(self, M=VS, N=VH,
                A_DRAM_ADDR=h_in, B_DRAM_ADDR=self.VIS_O_PROJ_DRAM,
                ADDOUTPUT_DRAM_ADDR=self.VIS_RESIDUAL_DRAM,
                NORMOUTPUT_DRAM_ADDR=self.VIS_NORM_OUT_DRAM,
                GAMMA_DRAM_ADDR=la['norm2_weight'])

            self.matmat_mul_core(is_B_quantized=True,
                M=VS, K=VH, N=VIS_MLP_PAD,
                A_DRAM_ADDR=self.VIS_NORM_OUT_DRAM,
                B_DRAM_ADDR=la['gate_data'],
                OUTPUT_DRAM_ADDR=self.VIS_MLP_GATE_DRAM,
                SCALE_DRAM_ADDR=la['gate_scale'], data_type=TYPE.IF4,
                C_DRAM_ADDR=la['gate_bias'], bias_mode="broadcast_N",
                silu_enable=True)
            self.matmat_mul_core(is_B_quantized=True,
                M=VS, K=VH, N=VIS_MLP_PAD,
                A_DRAM_ADDR=self.VIS_NORM_OUT_DRAM,
                B_DRAM_ADDR=la['up_data'],
                OUTPUT_DRAM_ADDR=self.VIS_MLP_UP_DRAM,
                SCALE_DRAM_ADDR=la['up_scale'], data_type=TYPE.IF4,
                C_DRAM_ADDR=la['up_bias'], bias_mode="broadcast_N")

            # gate * up chunked
            _M_CHUNK = min((0x80000 - 0x10000) // 2 // VIS_MLP_PAD, VS)
            for _t in range(0, VS, _M_CHUNK):
                _m_take = min(_M_CHUNK, VS - _t)
                _g_row = self.VIS_MLP_GATE_DRAM + _t * VIS_MLP_PAD * bpe
                _u_row = self.VIS_MLP_UP_DRAM   + _t * VIS_MLP_PAD * bpe
                _m_row = self.VIS_MLP_MULT_DRAM  + _t * VIS_MLP_PAD * bpe
                self.accelerator_memory_to_sram(_g_row, 0x10000, _m_take * VIS_MLP_PAD)
                self.accelerator_memory_to_sram(_u_row, 0x90000, _m_take * VIS_MLP_PAD)
                self.eltwise_mul_core(0x10000, 0x90000, 0x10000, _m_take * VIS_MLP_PAD)
                self.sram_to_accelerator_memory(0x10000, _m_row, _m_take * VIS_MLP_PAD)

            # down_proj
            self.matmat_mul_core(is_B_quantized=True,
                M=VS, K=VIS_MLP_PAD, N=VH,
                A_DRAM_ADDR=self.VIS_MLP_MULT_DRAM,
                B_DRAM_ADDR=la['down_data'],
                OUTPUT_DRAM_ADDR=self.VIS_MLP_DOWN_DRAM,
                SCALE_DRAM_ADDR=la['down_scale'], data_type=TYPE.IF4,
                C_DRAM_ADDR=la['down_bias'], bias_mode="broadcast_N")

            eltwise_add_core_dram(self, size=VS * VH,
                A_DRAM_ADDR=self.VIS_RESIDUAL_DRAM,
                B_DRAM_ADDR=self.VIS_MLP_DOWN_DRAM,
                OUTPUT_DRAM_ADDR=h_out)

        _original_print()  # newline after \r progress

        final_vis = self.VIS_IO_A_DRAM if total_layers % 2 == 0 else self.VIS_IO_B_DRAM

        self.rms_norm_core_dram(M=VS, N=VH,
            A_DRAM_ADDR=final_vis, OUTPUT_DRAM_ADDR=self.VIS_POST_NORM_DRAM,
            GAMMA_DRAM_ADDR=self.merger_ln_q_weight)

        merge_dim = VMERGE * VMERGE * VH
        for chunk_start in range(0, VS * VH, URAM_NEAR_FULL_ELEMENTS):
            chunk_take = min(URAM_NEAR_FULL_ELEMENTS, VS * VH - chunk_start)
            self.accelerator_memory_to_sram(
                self.VIS_POST_NORM_DRAM + chunk_start * bpe, 0x00000, chunk_take)
            self.sram_to_accelerator_memory(
                0x00000, self.VIS_MERGED_DRAM + chunk_start * bpe, chunk_take)

        if _bf16_mode and hasattr(self, 'merger_mlp0_weight_bf16'):
            self.matmat_mul_core(
                M=self.NUM_MERGED_TOKENS, K=merge_dim, N=merge_dim,
                A_DRAM_ADDR=self.VIS_MERGED_DRAM,
                B_DRAM_ADDR=self.merger_mlp0_weight_bf16,
                OUTPUT_DRAM_ADDR=self.VIS_MERGER_INTER_DRAM,
                C_DRAM_ADDR=self.merger_mlp0_bias_bf16, bias_mode="broadcast_N",
                gelu_enable=True)
            self.matmat_mul_core(
                M=self.NUM_MERGED_TOKENS, K=merge_dim, N=VH_OUT,
                A_DRAM_ADDR=self.VIS_MERGER_INTER_DRAM,
                B_DRAM_ADDR=self.merger_mlp2_weight_bf16,
                OUTPUT_DRAM_ADDR=self.VIS_ENCODER_OUT_DRAM,
                C_DRAM_ADDR=self.merger_mlp2_bias_bf16, bias_mode="broadcast_N")
        else:
            self.matmat_mul_core(
                M=self.NUM_MERGED_TOKENS, K=merge_dim, N=merge_dim,
                A_DRAM_ADDR=self.VIS_MERGED_DRAM,
                B_DRAM_ADDR=self.merger_mlp0_data,
                OUTPUT_DRAM_ADDR=self.VIS_MERGER_INTER_DRAM,
                C_DRAM_ADDR=self.merger_mlp0_bias, bias_mode="broadcast_N",
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.merger_mlp0_scale,
                gelu_enable=True)
            self.matmat_mul_core(
                M=self.NUM_MERGED_TOKENS, K=merge_dim, N=VH_OUT,
                A_DRAM_ADDR=self.VIS_MERGER_INTER_DRAM,
                B_DRAM_ADDR=self.merger_mlp2_data,
                OUTPUT_DRAM_ADDR=self.VIS_ENCODER_OUT_DRAM,
                C_DRAM_ADDR=self.merger_mlp2_bias, bias_mode="broadcast_N",
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.merger_mlp2_scale)

        # Finalize single program
        self.stop_capture()
        self.generate_instruction_halt()
        prog_bytes = bytearray()
        for inst in self.capture_buffer:
            prog_bytes.extend(inst.get_bytes())
        program_addr = self.get_program_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, program_addr, prog_bytes, len(prog_bytes))
        self.allocate_program_dram(len(prog_bytes))
        self.clear_capture_buffer()
        total_bytes = len(prog_bytes)

        _SILENT_MODE = False
        self._vis_program_addr = program_addr

        # Save compiled program to cache for fast reload next time
        with open(enc_bin_path, "wb") as f:
            f.write(prog_bytes)
        _original_print(f"    Vision encoder compiled: {total_bytes} total bytes, {total_layers} layers + merger (single program)")
        _original_print(f"    Program addr: 0x{program_addr:X}, cached to {enc_bin_path}")
        return program_addr

    def prepare_encoder_input(self, pixel_values: torch.Tensor,
                              image_grid_thw: torch.Tensor = None) -> dict:
        """Preprocess image: patch_embed, window reorder, RoPE tables. Call before compile_encoder()."""
        vis_cfg = self._vis_cfg
        VS = vis_cfg["num_patches"]
        VH = vis_cfg["hidden_size"]
        VD = vis_cfg["head_dim"]
        VD_PAD = 128
        model_dir = os.path.join(self.script_dir, self._cfg["paths"]["hf_model_dir"])

        if not hasattr(self, '_hf_model'):
            print("    Loading HF model for patch embedding...")
            from transformers import Qwen2_5_VLForConditionalGeneration
            self._hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_dir, torch_dtype=torch.bfloat16
            )
            if not hasattr(self._hf_model, 'visual') and hasattr(self._hf_model, 'model') and hasattr(self._hf_model.model, 'visual'):
                self._hf_model.visual = self._hf_model.model.visual
            self._hf_model.eval()
            print("    HF model loaded.")

        model = self._hf_model

        with torch.no_grad():
            if image_grid_thw is not None:
                pixel_values_hf = pixel_values.to(torch.bfloat16)
            else:
                from transformers import AutoTokenizer, Qwen2VLImageProcessor
                from PIL import Image
                img_np = pixel_values.float().permute(1, 2, 0).numpy()
                img_cfg = self._cfg.get("image_processing", {})
                mean = img_cfg.get("normalize_mean", [0.48145466, 0.4578275, 0.40821073])
                std = img_cfg.get("normalize_std", [0.26862954, 0.26130258, 0.27577711])
                for c in range(3):
                    img_np[:, :, c] = img_np[:, :, c] * std[c] + mean[c]
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                img_size = vis_cfg.get("image_size", 336)
                pil_img = pil_img.resize((img_size, img_size), Image.Resampling.BILINEAR)
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": "Describe this image."},
                ]}]
                tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
                text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_processor = Qwen2VLImageProcessor.from_pretrained(model_dir, trust_remote_code=True)
                img_inputs = image_processor(images=[pil_img], return_tensors="pt")
                text_enc = tokenizer(text_input, return_tensors="pt")
                pixel_values_hf = img_inputs["pixel_values"].to(torch.bfloat16)
                image_grid_thw = img_inputs["image_grid_thw"]

            visual = model.visual
            patch_embeds = visual.patch_embed(pixel_values_hf)
            assert patch_embeds.shape == (VS, VH), \
                f"Patch embed shape {patch_embeds.shape} != ({VS}, {VH})"
            rotary_pos_emb = visual.rot_pos_emb(image_grid_thw)

            window_index, cu_window_seqlens = visual.get_window_index(image_grid_thw)
            spatial_merge_unit = visual.spatial_merge_size ** 2
            patch_embeds = patch_embeds.reshape(VS // spatial_merge_unit, spatial_merge_unit, -1)
            patch_embeds = patch_embeds[window_index, :, :]
            patch_embeds = patch_embeds.reshape(VS, -1)
            rotary_pos_emb = rotary_pos_emb.reshape(VS // spatial_merge_unit, spatial_merge_unit, -1)
            rotary_pos_emb = rotary_pos_emb[window_index, :, :]
            rotary_pos_emb = rotary_pos_emb.reshape(VS, -1)
            self._vis_reverse_index = torch.argsort(window_index)

            if isinstance(cu_window_seqlens, torch.Tensor):
                self._cu_window_seqlens = cu_window_seqlens.unique_consecutive().tolist()
            else:
                self._cu_window_seqlens = [cu_window_seqlens[0]] + [
                    cu_window_seqlens[i] for i in range(1, len(cu_window_seqlens))
                    if cu_window_seqlens[i] != cu_window_seqlens[i-1]]

        return {
            'patch_embeds': patch_embeds,
            'rotary_pos_emb': rotary_pos_emb,
            'image_grid_thw': image_grid_thw,
        }

    def run_encoder(self, program_addr: int, pixel_values: torch.Tensor,
                    image_grid_thw: torch.Tensor = None) -> None:
        """Run vision encoder on FPGA. Output in VIS_ENCODER_OUT_DRAM."""
        vis_cfg = self._vis_cfg
        VS = vis_cfg["num_patches"]
        VH = vis_cfg["hidden_size"]
        VN = vis_cfg["num_heads"]
        VD = vis_cfg["head_dim"]
        VD_PAD = 128
        bpe = 2

        # Preprocess image (or reuse if prepare_encoder_input was already called)
        prep = self.prepare_encoder_input(pixel_values, image_grid_thw)
        patch_embeds = prep['patch_embeds']
        rotary_pos_emb = prep['rotary_pos_emb']
        image_grid_thw = prep['image_grid_thw']

        patch_bf16 = patch_embeds.to(torch.bfloat16).contiguous().flatten()
        self.dma_to_accelerator_memory(self.VIS_PATCH_INPUT_DRAM, patch_bf16)
        print(f"    Patch embeddings [{VS}, {VH}] DMA'd to 0x{self.VIS_PATCH_INPUT_DRAM:X}")

        # Re-zero the padded Q/K/V buffers (they accumulate stale data between runs)
        vis_pad_zeros = torch.zeros(VN * VS * VD_PAD, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.VIS_Q_PAD_DRAM, vis_pad_zeros)
        self.dma_to_accelerator_memory(self.VIS_K_PAD_DRAM, vis_pad_zeros)
        self.dma_to_accelerator_memory(self.VIS_V_PAD_DRAM, vis_pad_zeros)

        # RoPE cos/sin tables use rearranged padding layout [d0..d39, zeros(24), d40..d79, zeros(24)]
        half_d = VD // 2
        pad_gap = VD_PAD // 2 - half_d  # 64 - 40 = 24
        with torch.no_grad():
            cos_raw = rotary_pos_emb.cos().to(torch.bfloat16)  # [VS, 40]
            sin_raw = rotary_pos_emb.sin().to(torch.bfloat16)  # [VS, 40]

            # Build cos table: [VS, VD_PAD=128]
            cos_table = torch.ones(VS, VD_PAD, dtype=torch.bfloat16)
            cos_table[:, :half_d] = cos_raw           # [0:40] = cos
            # [40:64] stays ones (identity for zero-padded dims)
            cos_table[:, 64:64 + half_d] = cos_raw    # [64:104] = cos
            # [104:128] stays ones

            # Build sin table: [VS, VD_PAD=128]
            sin_table = torch.zeros(VS, VD_PAD, dtype=torch.bfloat16)
            sin_table[:, :half_d] = -sin_raw           # [0:40] = -sin
            # [40:64] stays zeros
            sin_table[:, 64:64 + half_d] = sin_raw     # [64:104] = +sin
            # [104:128] stays zeros

            self.dma_to_accelerator_memory(self.VIS_ROPE_COS_DRAM, cos_table.flatten())
            self.dma_to_accelerator_memory(self.VIS_ROPE_SIN_DRAM, sin_table.flatten())
        print(f"    Vision RoPE tables [{VS}, {VD_PAD}] DMA'd to FPGA")

        # Vision encoder FLOP estimate
        vis_cfg = self._vis_cfg
        total_layers = vis_cfg["depth"]
        VI = vis_cfg["intermediate_size"]
        VIS_MLP_PAD = ((VI + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
        cu_win = getattr(self, '_cu_window_seqlens', [0, VS])
        fullatt_idxs = set(vis_cfg.get("fullatt_block_indexes", [7, 15, 23, 31]))
        vis_flops = 0
        for li in range(total_layers):
            vis_flops += 3 * VS * VH                               # RMSNorm
            vis_flops += 2 * VS * VH * (VN * VD_PAD * 2)           # QK matmul
            vis_flops += 2 * VS * VH * (VN * VD_PAD)               # V matmul
            vis_flops += VN * VS * VD_PAD * 4 * 2                  # RoPE Q+K
            if li in fullatt_idxs:
                vis_flops += VN * (2*VS*VD_PAD*VS + VS*VS*5 + 2*VS*VS*VD_PAD)
            else:
                for w in range(len(cu_win)-1):
                    wl = cu_win[w+1] - cu_win[w]
                    vis_flops += VN * (2*wl*VD_PAD*wl + wl*wl*5 + 2*wl*wl*VD_PAD)
            vis_flops += 2 * VS * VH * VH                          # O proj
            vis_flops += 4 * VS * VH                                # residual+norm
            vis_flops += 2 * VS * VH * VIS_MLP_PAD * 2 + VS * VIS_MLP_PAD  # gate+up+mul
            vis_flops += 2 * VS * VIS_MLP_PAD * VH                 # down
            vis_flops += VS * VH                                    # residual
        VMERGE = vis_cfg["spatial_merge_size"]
        merge_dim = VMERGE * VMERGE * VH
        VH_OUT = vis_cfg["out_hidden_size"]
        vis_flops += 2 * self.NUM_MERGED_TOKENS * merge_dim * merge_dim  # merger MLP0
        vis_flops += 2 * self.NUM_MERGED_TOKENS * merge_dim * VH_OUT     # merger MLP2

        print(f"    Running vision encoder on FPGA (longest single step, ~40-60s)...", flush=True)
        self.start_execute_from_dram(program_addr)
        wall_s = self._wait_with_heartbeat(lambda: self.wait_queue(600.0), label="vision")
        latency_us = self._corrected_hw_latency_us(wall_s)
        vis_gflops = vis_flops / (latency_us * 1e-6) / 1e9
        self._vis_total_flops = vis_flops
        self._vis_gflops = vis_gflops
        print(f"    Vision encoder complete: {latency_us/1e6:.2f}s, {vis_gflops:.2f} GFLOPS")

        num_merged = self.NUM_MERGED_TOKENS
        out_hidden = self._vis_cfg["out_hidden_size"]
        if hasattr(self, '_vis_reverse_index'):
            merger_out = self.dma_from_accelerator_memory(
                self.VIS_ENCODER_OUT_DRAM, (num_merged, out_hidden)).cpu()
            merger_out = merger_out[self._vis_reverse_index].contiguous()
            self.dma_to_accelerator_memory(self.VIS_ENCODER_OUT_DRAM, merger_out.flatten())
        self._vis_num_tokens = num_merged
        self._image_grid_thw = image_grid_thw  # store for mRoPE computation
        print(f"    Vision encoder output: {self._vis_num_tokens} tokens at 0x{self.VIS_ENCODER_OUT_DRAM:X}")

    def compile_prefill(self, seq_len: int, layer_size: int | None = None) -> dict:
        """Compile prefill program for given seq_len. Returns (addr, flops)."""
        if layer_size is None:
            layer_size = self.LAYER_SIZE

        # Check for cached prefill program
        bin_dir = os.path.join(self.script_dir, "qwen2.5_vl_3b_bin")
        os.makedirs(bin_dir, exist_ok=True)
        cache_path = os.path.join(bin_dir, f"prefill_program_s{seq_len}.bin")
        meta_path = os.path.join(bin_dir, f"prefill_program_s{seq_len}.json")

        if os.path.exists(cache_path) and os.path.exists(meta_path):
            _original_print(f"  Loading cached prefill program (seq_len={seq_len-1}) from {cache_path} ...")
            with open(cache_path, "rb") as f:
                prog_bytes = f.read()
            with open(meta_path, "r") as f:
                meta = json.load(f)
            program_addr = self.get_program_dram_addr()
            self.dma_write(DMA_DEVICE_H2C, program_addr, prog_bytes, len(prog_bytes))
            self.allocate_program_dram(len(prog_bytes))
            self.seq_len = seq_len - 1
            return program_addr, meta["total_flops"]

        seq_len -= 1
        self.seq_len = seq_len
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        aligned_tok = ((seq_len + 63) // 64) * 64

        ahd  = self.actual_head_dim
        nkvh = self.num_kv_heads
        qpkv = self.group_size
        bpe  = self.bytes_per_element
        hd   = self.head_dim
        rope_row_bytes = ahd * 2 * bpe

        global _SILENT_MODE
        _SILENT_MODE = True
        self.start_capture()
        total_flops = 0
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        _original_print(f"  Compiling prefill seq_len={seq_len}, {layer_size} layers...")
        for layer_idx in range(layer_size):
            _original_print(f"    prefill layer {layer_idx + 1}/{layer_size}", end="\r", flush=True)
            layer_off = layer_idx * LAYER_WEIGHT_SIZE
            la = self.lm_layer_addrs[layer_idx]
            if layer_idx != 0:
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, sram_address=0x10000, element_size=seq_len * self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_INPUT_DRAM, element_size=seq_len * self.vector_length)

            # Pre-norm (input_layernorm)
            total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=la['ln1_gamma'])

            # Q, K, V projections with bias
            total_flops += self.matmat_mul_core(is_B_quantized=True,M=seq_len, K=self.vector_length, N=hd * qpkv,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=la['q_data'], OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                SCALE_DRAM_ADDR=la['q_scale'], data_type=TYPE.IF4,
                C_DRAM_ADDR=la['q_bias'])
            total_flops += self.matmat_mul_core(is_B_quantized=True,M=seq_len, K=self.vector_length, N=hd,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=la['k_data'], OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                SCALE_DRAM_ADDR=la['k_scale'], data_type=TYPE.IF4,
                C_DRAM_ADDR=la['k_bias'])
            # v_proj writes to temp buffer (seq_len, hd) in standard per-head interleaved layout (BF16)
            total_flops += self.matmat_mul_core(M=seq_len, K=self.vector_length, N=hd,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=la['v_weight'], OUTPUT_DRAM_ADDR=self.LAYER0_V_PROJ_TEMP,
                C_DRAM_ADDR=la['v_bias'])

            # Qwen2.5-VL: NO QK RMSNorm
            # Copy K_DRAM → K_NORM_DRAM (no scaling)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_K_DRAM, sram_address=0x10000, element_size=seq_len * hd)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_K_NORM_DRAM, element_size=seq_len * hd)
            # Copy Q_DRAM → Q_NORM_DRAM (no scaling needed)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_Q_DRAM, sram_address=0x10000, element_size=seq_len * hd * qpkv)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_Q_NORM_DRAM, element_size=seq_len * hd * qpkv)

            # RoPE per head per token: K uses single RoPE base
            ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE
            for t in range(seq_len):
                cos_addr = ROPE_WEIGHT_ADDR + t * rope_row_bytes
                sin_addr = cos_addr + ahd * bpe
                for kv_h in range(nkvh):
                    total_flops += self.rope_hf_core(
                        N=ahd,
                        input_dram_addr=self.LAYER0_K_NORM_DRAM + (t * nkvh + kv_h) * ahd * bpe,
                        output_dram_addr=self.LAYER0_K_NORM_DRAM + (t * nkvh + kv_h) * ahd * bpe,
                        cos_dram_addr=cos_addr,
                        sin_dram_addr=sin_addr)
                for q_h in range(nkvh * qpkv):
                    total_flops += self.rope_hf_core(
                        N=ahd,
                        input_dram_addr=self.LAYER0_Q_NORM_DRAM + (t * nkvh * qpkv + q_h) * ahd * bpe,
                        output_dram_addr=self.LAYER0_Q_NORM_DRAM + (t * nkvh * qpkv + q_h) * ahd * bpe,
                        cos_dram_addr=cos_addr,
                        sin_dram_addr=sin_addr)

            # Per-Q-head flash attention: one Q head at a time, K/V shared within KV group
            for kv_h in range(nkvh):
                k_cache_base = (self.LAYER0_K_ROPE_DRAM
                                + layer_idx * nkvh * self.MAX_CONTEXT_SIZE * ahd * bpe
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)
                v_cache_base = (self.LAYER0_V_DRAM
                                + layer_idx * nkvh * self.MAX_CONTEXT_SIZE * ahd * bpe
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)

                # Scatter roped K_h → KV cache + FLASH_K (ONE copy, no GQA dup)
                for t in range(seq_len):
                    k_src = self.LAYER0_K_NORM_DRAM + (t * nkvh + kv_h) * ahd * bpe
                    self.accelerator_memory_to_sram(k_src, 0x10000, ahd)
                    self.sram_to_accelerator_memory(0x10000, k_cache_base + t * ahd * bpe, ahd)
                    self.sram_to_accelerator_memory(0x10000, self.LAYER0_FLASH_K_DRAM + t * ahd * bpe, ahd)

                # Scatter V_h from V_PROJ_TEMP → KV cache + FLASH_V (ONE copy, no GQA dup)
                for t in range(seq_len):
                    v_src = self.LAYER0_V_PROJ_TEMP + (t * nkvh + kv_h) * ahd * bpe
                    self.accelerator_memory_to_sram(v_src, 0x20000, ahd)
                    self.sram_to_accelerator_memory(0x20000, v_cache_base + t * ahd * bpe, ahd)
                    self.sram_to_accelerator_memory(0x20000, self.LAYER0_FLASH_V_DRAM + t * ahd * bpe, ahd)

                # Per-Q-head: scatter one Q head, flash attention, assemble output
                for q_idx in range(qpkv):
                    # Scatter Q head (kv_h * qpkv + q_idx) for all tokens → FLASH_Q
                    for t in range(seq_len):
                        q_src = self.LAYER0_Q_NORM_DRAM + (t * nkvh * qpkv + kv_h * qpkv + q_idx) * ahd * bpe
                        self.accelerator_memory_to_sram(q_src, 0x30000, ahd)
                        self.sram_to_accelerator_memory(0x30000, self.LAYER0_FLASH_Q_DRAM + t * ahd * bpe, ahd)

                    # Flash attention for this single Q head (seq_len = aligned_tok, standard causal)
                    total_flops += self._flash_attention_core_cached(
                        head_dim=ahd,
                        seq_len=aligned_tok,
                        Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
                        K_DRAM_ADDR=self.LAYER0_FLASH_K_DRAM,
                        V_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                        OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUT_HEAD_DRAM,
                        SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                        BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_DRAM,
                    )

                    # Assemble: copy output to FLASH_OUTPUT_DRAM at correct head position
                    global_q_h = kv_h * qpkv + q_idx
                    for t in range(seq_len):
                        src = self.LAYER0_FLASH_OUT_HEAD_DRAM + t * ahd * bpe
                        dst = self.LAYER0_FLASH_OUTPUT_DRAM + t * hd * qpkv * bpe + global_q_h * ahd * bpe
                        self.accelerator_memory_to_sram(src, 0x40000, ahd)
                        self.sram_to_accelerator_memory(0x40000, dst, ahd)

            # o_proj: (seq_len, hd * qpkv) → (seq_len, vector_length) (BF16)
            total_flops += self.matmat_mul_core(M=seq_len, K=hd * qpkv, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM, B_DRAM_ADDR=la['o_weight'], OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM)

            # Qwen2.5-VL: no post-attention norm; add residual directly to o_proj output
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_INPUT_DRAM, sram_address=0x10000, element_size=seq_len * self.vector_length)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM, sram_address=0x90000, element_size=seq_len * self.vector_length)
            self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=seq_len * self.vector_length)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, element_size=seq_len * self.vector_length)

            # Qwen2.5-VL: post_attention_layernorm IS the pre-FFN norm
            total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=la['ln2_gamma'])

            # MLP: gate_proj with SiLU, up_proj, gate x up element-wise, down_proj
            total_flops += self.matmat_mul_core(is_B_quantized=True,M=seq_len, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=la['gate_data'], OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                SCALE_DRAM_ADDR=la['gate_scale'], data_type=TYPE.IF4, silu_enable=True)
            total_flops += self.matmat_mul_core(is_B_quantized=True,M=seq_len, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=la['up_data'], OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                SCALE_DRAM_ADDR=la['up_scale'], data_type=TYPE.IF4)

            # gate x up chunked: process M_CHUNK rows at a time
            _bpe = self.bytes_per_element
            _M_CHUNK = min((0x80000 - 0x10000) // 2 // self.mlp_elements, seq_len)
            for _t in range(0, seq_len, _M_CHUNK):
                _m_take = min(_M_CHUNK, seq_len - _t)
                _g_row = self.LAYER0_MLP_GATE_DRAM + _t * self.mlp_elements * _bpe
                _u_row = self.LAYER0_MLP_UP_DRAM   + _t * self.mlp_elements * _bpe
                _m_row = self.LAYER0_MLP_MULT_DRAM  + _t * self.mlp_elements * _bpe
                self.accelerator_memory_to_sram(_g_row, 0x10000, _m_take * self.mlp_elements)
                self.accelerator_memory_to_sram(_u_row, 0x90000, _m_take * self.mlp_elements)
                self.eltwise_mul_core(0x10000, 0x90000, 0x10000, _m_take * self.mlp_elements)
                self.sram_to_accelerator_memory(0x10000, _m_row, _m_take * self.mlp_elements)

            # down_proj
            total_flops += self.matmat_mul_core(is_B_quantized=True,M=seq_len, K=self.mlp_elements, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM, B_DRAM_ADDR=la['down_data'], OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                SCALE_DRAM_ADDR=la['down_scale'], data_type=TYPE.IF4)

            # Qwen2.5-VL: no post-FFN norm; add residual directly to down_proj output
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, sram_address=0x10000, element_size=seq_len * self.vector_length)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_DOWN_DRAM, sram_address=0x90000, element_size=seq_len * self.vector_length)
            self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=seq_len * self.vector_length)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, element_size=seq_len * self.vector_length)

        self.stop_capture()
        self.generate_instruction_halt()
        # Save program bytes before writing to DRAM (for caching)
        prog_bytes = bytearray()
        for inst in self.capture_buffer:
            prog_bytes.extend(inst.get_bytes())
        prefill_program_addr = self.get_program_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, prefill_program_addr, prog_bytes, len(prog_bytes))
        self.allocate_program_dram(len(prog_bytes))
        self.clear_capture_buffer()
        _SILENT_MODE = False
        _original_print()  # newline after \r layer progress
        # Cache to disk
        with open(cache_path, "wb") as f:
            f.write(prog_bytes)
        with open(meta_path, "w") as f:
            json.dump({"total_flops": total_flops, "seq_len": seq_len}, f)
        print(f"    Prefill program cached to {cache_path} ({len(prog_bytes)} bytes)")
        print(f"    Prefill program start at 0x{prefill_program_addr:X} end at 0x{self.get_program_dram_addr():X}, usage: {self.get_program_dram_usage()} bytes")

        return prefill_program_addr, total_flops

    def run_prefill(self, prefill_program_addr: int, prefill_seq, gflops: int = None, has_image: bool = False) -> dict:
        """Run prefill: gather embeddings, optionally merge vision tokens, then execute."""
        if prefill_seq is None:
            prefill_seq = tuple(self._cfg["default_prefill_tokens"])

        # Prefill processes all but the last token
        if len(prefill_seq) > 1:
            prefill_seq = prefill_seq[:-1]
            assert len(prefill_seq) == self.seq_len, f"Expected seq_len {self.seq_len}, but got {len(prefill_seq)}"
        else:
            raise ValueError("Prefill sequence must have at least 2 tokens.")

        seq_len = len(prefill_seq)
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        aligned_tok = ((seq_len + 63) // 64) * 64
        bpe = self.bytes_per_element
        embed_row_bytes = self.vector_length * bpe

        # Gather text embeddings
        embedding_tensor = self.get_embedding_for_tokens(prefill_seq)

        # Merge vision embeddings: overwrite image token positions
        if has_image:
            vision_token_id = 151655  # <|image_pad|>
            img_positions = [i for i, t in enumerate(prefill_seq) if t == vision_token_id]
            num_vis = getattr(self, '_vis_num_tokens', 0)
            if len(img_positions) > 0 and num_vis > 0:
                # Read vision output from device DRAM back to host for merge
                vis_embeddings = self.dma_from_accelerator_memory(
                    self.VIS_ENCODER_OUT_DRAM, (num_vis, self.vector_length)).cpu()

                # Overwrite image token positions with vision embeddings
                embed_reshaped = embedding_tensor.reshape(seq_len, self.vector_length)
                n_replace = min(len(img_positions), num_vis)
                for i in range(n_replace):
                    embed_reshaped[img_positions[i]] = vis_embeddings[i]
                embedding_tensor = embed_reshaped.flatten()
                print(f"    Merged {n_replace} vision tokens into prefill at positions {img_positions[:5]}{'...' if len(img_positions) > 5 else ''}")

        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)

        # Build causal bias mask (standard lower-triangular on aligned_tok x aligned_tok)
        bias_one_group = torch.full((aligned_tok, aligned_tok), float("-inf"), dtype=torch.bfloat16)
        valid_mask = torch.tril(torch.ones(aligned_tok, aligned_tok, dtype=torch.bool), diagonal=0) if not self.causal_mask_upper else torch.triu(torch.ones(aligned_tok, aligned_tok, dtype=torch.bool), diagonal=0)
        bias_one_group.masked_fill_(valid_mask, 0.0)
        bias_one_group[:, seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_one_group)
        self.program_execute(prefill_program_addr, gflops=gflops)

    def compile_decoder(self, layer_size: int | None = None) -> tuple[list[int], list[int]]:
        """Compile decoder programs for seq_len buckets; write decoder_program.bin and decoder_program.json.
        Returns (program_sizes[8], total_flops_list[8])."""
        if layer_size is None:
            layer_size = self.LAYER_SIZE
        paths_cfg = self._cfg.get("paths", {})
        decoder_bin_rel = paths_cfg.get("decoder_program_bin", "qwen2.5_vl_3b_bin/decoder_program.bin")
        decoder_meta_rel = paths_cfg.get("decoder_program_meta", "qwen2.5_vl_3b_bin/decoder_program.json")
        decoder_bin_path = os.path.join(self.script_dir, decoder_bin_rel)
        decoder_meta_path = os.path.join(self.script_dir, decoder_meta_rel)
        os.makedirs(os.path.dirname(decoder_bin_path), exist_ok=True)
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        segment_instruction_counts = []
        total_flops_list = []

        ahd  = self.actual_head_dim
        nkvh = self.num_kv_heads
        qpkv = self.group_size
        bpe  = self.bytes_per_element
        hd   = self.head_dim
        rope_row_bytes = ahd * 2 * bpe   # 512 bytes per position

        global _SILENT_MODE
        _SILENT_MODE = True
        self.clear_inst_id()
        self.start_capture()

        buckets = self._cfg["model"]["decoder_seq_len_buckets"]
        _original_print(f"  Compiling decoder {len(buckets)} buckets x {layer_size} layers...")
        for _bi, seq_len in enumerate(buckets):
            _original_print(f"    decoder bucket {_bi + 1}/{len(buckets)} seq_len={seq_len}...", flush=True)
            count_at_start = self.capture_count
            total_flops = 0

            for layer_idx in range(layer_size):
                layer_off = layer_idx * LAYER_WEIGHT_SIZE
                la = self.lm_layer_addrs[layer_idx]
                if layer_idx != 0:
                    self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, sram_address=0x10000, element_size=self.vector_length)
                    self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_INPUT_DRAM, element_size=self.vector_length)

                # Pre-norm
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=la['ln1_gamma'])

                # Q, K, V projections with bias
                total_flops += self.matmat_mul_core(is_B_quantized=True,M=1, K=self.vector_length, N=hd * qpkv,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=la['q_data'], OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                    SCALE_DRAM_ADDR=la['q_scale'], data_type=TYPE.IF4,
                    C_DRAM_ADDR=la['q_bias'])
                total_flops += self.matmat_mul_core(is_B_quantized=True,M=1, K=self.vector_length, N=hd,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=la['k_data'], OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                    SCALE_DRAM_ADDR=la['k_scale'], data_type=TYPE.IF4,
                    C_DRAM_ADDR=la['k_bias'])
                total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=hd,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=la['v_weight'], OUTPUT_DRAM_ADDR=self.LAYER0_V_PROJ_TEMP,
                    C_DRAM_ADDR=la['v_bias'])

                # Qwen2.5-VL: NO QK norm — use Q/K directly

                # RoPE per head (decode position set by ROPE_SIZE_REG at runtime)
                ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE
                for kv_h in range(nkvh):
                    total_flops += self.rope_hf_core(
                        N=ahd,
                        input_dram_addr=self.LAYER0_K_DRAM + kv_h * ahd * bpe,
                        output_dram_addr=self.LAYER0_K_DRAM + kv_h * ahd * bpe,
                        cos_dram_addr=ROPE_WEIGHT_ADDR,
                        sin_dram_addr=ROPE_WEIGHT_ADDR + ahd * bpe,
                        rope_size_reg=self.ROPE_SIZE_REG,
                        tmp_reg=self.TMP_REG)
                for q_h in range(nkvh * qpkv):
                    total_flops += self.rope_hf_core(
                        N=ahd,
                        input_dram_addr=self.LAYER0_Q_DRAM + q_h * ahd * bpe,
                        output_dram_addr=self.LAYER0_Q_DRAM + q_h * ahd * bpe,
                        cos_dram_addr=ROPE_WEIGHT_ADDR,
                        sin_dram_addr=ROPE_WEIGHT_ADDR + ahd * bpe,
                        rope_size_reg=self.ROPE_SIZE_REG,
                        tmp_reg=self.TMP_REG)

                # Per-KV-head: store new K/V to cache at decode position, then decoder_attention
                for kv_h in range(nkvh):
                    k_cache_base = (self.LAYER0_K_ROPE_DRAM
                                    + layer_idx * nkvh * self.MAX_CONTEXT_SIZE * ahd * bpe
                                    + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)
                    v_cache_base = (self.LAYER0_V_DRAM
                                    + layer_idx * nkvh * self.MAX_CONTEXT_SIZE * ahd * bpe
                                    + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)

                    # Store roped K_h to KV cache at decode position (via V_CACHE_SIZE_REG)
                    self.accelerator_memory_to_sram(self.LAYER0_K_DRAM + kv_h * ahd * bpe, 0x10000, ahd)
                    self.generate_instruction_add_imm(self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter(k_cache_base), self.TMP_REG)
                    self.sram_to_accelerator_memory(0x10000, 0, ahd)
                    self.overwrite_instruction_with_general_register(self.TMP_REG)

                    # Store V_h to KV cache at decode position (via V_CACHE_SIZE_REG)
                    self.accelerator_memory_to_sram(self.LAYER0_V_PROJ_TEMP + kv_h * ahd * bpe, 0x20000, ahd)
                    self.generate_instruction_add_imm(self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter(v_cache_base), self.TMP_REG)
                    self.sram_to_accelerator_memory(0x20000, 0, ahd)
                    self.overwrite_instruction_with_general_register(self.TMP_REG)

                    # For each Q head in this KV head's group: run decoder_attention
                    for q in range(qpkv):
                        q_src = self.LAYER0_Q_DRAM + (kv_h * qpkv + q) * ahd * bpe
                        flash_q_addr = self.LAYER0_FLASH_Q_DRAM + q * ahd * bpe
                        self.accelerator_memory_to_sram(q_src, 0x30000, ahd)
                        self.sram_to_accelerator_memory(0x30000, flash_q_addr, ahd)

                        total_flops += self.decoder_attention_core(
                            head_dim=ahd,
                            seq_len=seq_len,
                            Q_DRAM_ADDR=flash_q_addr,
                            K_DRAM_ADDR=k_cache_base,
                            V_DRAM_ADDR=v_cache_base,
                            OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM + (kv_h * qpkv + q) * ahd * bpe,
                            IDENTITY_DRAM_ADDR=self.IDENTITY_DRAM_ADDR,
                            SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                            BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_DRAM,
                        )

                # o_proj (BF16)
                total_flops += self.matmat_mul_core(M=1, K=hd * qpkv, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM, B_DRAM_ADDR=la['o_weight'], OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM)

                # Qwen2.5-VL: no post-attention norm; residual direct on o_proj output
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_INPUT_DRAM, sram_address=0x10000, element_size=self.vector_length)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM, sram_address=0x90000, element_size=self.vector_length)
                self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, element_size=self.vector_length)

                # Qwen2.5-VL: post_attention_layernorm IS the pre-FFN norm
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=la['ln2_gamma'])

                # MLP: SwiGLU
                total_flops += self.matmat_mul_core(is_B_quantized=True,M=1, K=self.vector_length, N=self.mlp_elements,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=la['gate_data'], OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                    SCALE_DRAM_ADDR=la['gate_scale'], data_type=TYPE.IF4, silu_enable=True)
                total_flops += self.matmat_mul_core(is_B_quantized=True,M=1, K=self.vector_length, N=self.mlp_elements,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=la['up_data'], OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                    SCALE_DRAM_ADDR=la['up_scale'], data_type=TYPE.IF4)

                # gate x up (M=1: mlp_elements fits in SRAM in one shot)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_GATE_DRAM, sram_address=0x10000, element_size=self.mlp_elements)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_UP_DRAM, sram_address=0x90000, element_size=self.mlp_elements)
                self.eltwise_mul_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.mlp_elements)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_MLP_MULT_DRAM, element_size=self.mlp_elements)

                # down_proj
                total_flops += self.matmat_mul_core(is_B_quantized=True,M=1, K=self.mlp_elements, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM, B_DRAM_ADDR=la['down_data'], OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                    SCALE_DRAM_ADDR=la['down_scale'], data_type=TYPE.IF4)

                # Qwen2.5-VL: no post-FFN norm; residual direct on down_proj output
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, sram_address=0x10000, element_size=self.vector_length)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_DOWN_DRAM, sram_address=0x90000, element_size=self.vector_length)
                self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, element_size=self.vector_length)

            if layer_size == self.LAYER_SIZE:
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_OUTPUT_DRAM,
                    OUTPUT_DRAM_ADDR=self.OUTPUT_NORM_DRAM, GAMMA_DRAM_ADDR=self.final_norm_addr)
                total_flops += self.matmat_mul_core(is_B_quantized=True, M=1, K=self.vector_length, N=self.EMBEDDING_ELEMENTS,
                    A_DRAM_ADDR=self.OUTPUT_NORM_DRAM, B_DRAM_ADDR=self.lm_head_data, OUTPUT_DRAM_ADDR=self.LOGITS_DRAM,
                    SCALE_DRAM_ADDR=self.lm_head_scale, data_type=TYPE.IF4)

            self.generate_instruction_halt()
            segment_instruction_counts.append(self.capture_count - count_at_start)
            total_flops_list.append(total_flops)

        self.stop_capture()
        _SILENT_MODE = False
        all_programs_bytes = bytearray()
        for inst in self.capture_buffer:
            all_programs_bytes.extend(inst.get_bytes())
        with open(decoder_bin_path, "wb") as f:
            f.write(all_programs_bytes)
        program_sizes = [c * 32 for c in segment_instruction_counts]
        with open(decoder_meta_path, "w") as f:
            json.dump({"instruction_counts": segment_instruction_counts, "program_sizes": program_sizes, "total_flops": total_flops_list}, f, indent=0)
        self.clear_capture_buffer()
        print(f"Decoder programs: {len(segment_instruction_counts)} segments written to {decoder_bin_path} ({len(all_programs_bytes)} bytes)")
        return program_sizes, total_flops_list

    def run_decoder(self, decoder_program_sizes: list[int], decoder_base_addr: int, token_id: int,
                    gflops_per_token: list[int] | None = None, repetition_penalty: float = 1.2) -> dict:
        """Run decode loop. Breaks on EOS/EOT tokens."""
        if token_id is None:
            print("No last token available for decode.")
            return {}

        _qwen25_stop_tokens = {151643, 151645, self._end_of_turn_token_id}

        ahd = self.actual_head_dim
        bpe = self.bytes_per_element
        _kv_stride  = ahd * bpe
        _rope_stride = ahd * 2 * bpe

        generated_tokens = set()

        global _SILENT_MODE
        max_seq_len = self.MAX_CONTEXT_SIZE
        while self.seq_len < max_seq_len:
            _SILENT_MODE = True
            self.seq_len += 1
            aligned_seq_len = ((self.seq_len + 63) // 64) * 64
            prog_idx = min((self.seq_len - 1) // 64, len(decoder_program_sizes) - 1)
            prog_addr = decoder_base_addr + sum(decoder_program_sizes[:prog_idx])

            self.isa_add_set_core(self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter((self.seq_len - 1) * _kv_stride))
            _rope_pos = self.seq_len - 1 + getattr(self, '_rope_offset', 0)
            self.isa_add_set_core(self.ROPE_SIZE_REG,    ue_35bit_addr_shifter(_rope_pos * _rope_stride))

            embedding_tensor = self.get_embedding_for_tokens([token_id])
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)

            bias_host = torch.full((1, aligned_seq_len), -1e36, dtype=torch.bfloat16)
            bias_host[0, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_host)

            self.start_execute_from_dram(prog_addr)
            self.wait_queue(10.0)

            if repetition_penalty != 1.0 and len(generated_tokens) > 0:
                logits = self.dma_from_accelerator_memory(
                    self.LOGITS_DRAM, (self.EMBEDDING_ELEMENTS,)).cpu().float()
                for prev_tok in generated_tokens:
                    if logits[prev_tok] > 0:
                        logits[prev_tok] /= repetition_penalty
                    else:
                        logits[prev_tok] *= repetition_penalty
                token_id = int(logits.argmax().item())
            else:
                token_id = self.get_arg_max_index()

            generated_tokens.add(token_id)
            token_char = self.tokenizer.decode([token_id])
            _SILENT_MODE = False
            if token_id in _qwen25_stop_tokens:
                print(f"\nStop token {token_id} reached.")
                break
            print(token_char, end="", flush=True)
        return self.seq_len

def process_image(image_path: str, size: int = 448) -> torch.Tensor:
    """Load, resize, normalize image -> [3, size, size] bf16."""
    from PIL import Image
    img_cfg = _load_config(SCRIPT_DIR).get("image_processing", {})
    resize = img_cfg.get("resize", size)
    mean = img_cfg.get("normalize_mean", [0.48145466, 0.4578275, 0.40821073])
    std = img_cfg.get("normalize_std", [0.26862954, 0.26130258, 0.27577711])
    img = Image.open(image_path).convert("RGB").resize((resize, resize), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    for c in range(3):
        arr[:, :, c] = (arr[:, :, c] - mean[c]) / std[c]
    return torch.from_numpy(arr).permute(2, 0, 1).to(torch.bfloat16)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Qwen2.5-VL-3B prefill + decode on accelerator.")
    parser.add_argument("--prompt", type=str, default="please describe the image in details.", help="Text prompt")
    parser.add_argument("--image", type=str, default=os.path.join(SCRIPT_DIR, "../../test_samples/yosemite.jpg"),
                        help="Path to image file (default: test_samples/yosemite.jpg, use 'none' for text-only)")
    parser.add_argument('--vision-cpu', action='store_true',
                        help='Run vision encoder on CPU (fp32) instead of FPGA — for quality comparison')
    parser.add_argument('--rep-penalty', type=float, default=1.05,
                        help='Repetition penalty (1.0=off, default: 1.05)')
    parser.add_argument('--dev', type=str, default='xdma0',
                        help='DMA device name (e.g., xdma0, xdma1). Default: xdma0')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    set_dma_device(args.dev)
    # Refresh local bindings shadowed at import time so DMA goes to the right device
    import sys as _sys, user_dma_core as _udc
    _mod = _sys.modules[__name__]
    _mod.DMA_DEVICE_H2C = _udc.DMA_DEVICE_H2C
    _mod.DMA_DEVICE_C2H = _udc.DMA_DEVICE_C2H
    _mod.DMA_DEVICE_USER = _udc.DMA_DEVICE_USER
    global DMA_DEVICE_H2C, DMA_DEVICE_C2H, DMA_DEVICE_USER
    DMA_DEVICE_H2C = user_dma_core.DMA_DEVICE_H2C
    DMA_DEVICE_C2H = user_dma_core.DMA_DEVICE_C2H
    DMA_DEVICE_USER = user_dma_core.DMA_DEVICE_USER
    user_dma_core.CLOCK_CYCLE_TIME_NS = 5.63

    ue = Qwen25VL3B_UnifiedEngine(script_dir=script_dir)

    # Stop any stale FPGA execution from a previous crashed/timed-out run
    ue.dram_inst_running(False)
    ue.start_capture()
    ue.generate_instruction_halt()
    ue.stop_capture()
    halt_bytes = bytearray()
    for inst in ue.capture_buffer:
        halt_bytes.extend(inst.get_bytes())
    ue.dma_write(DMA_DEVICE_H2C, DRAM_INSTRUCTION_ADDR, halt_bytes, len(halt_bytes))
    ue.clear_capture_buffer()

    cfg = _load_config(script_dir)
    if args.image and args.image.lower() == "none":
        args.image = None
    has_image = args.image is not None

    print(f"\n--- Configuration ---")
    print(f"  Image : {args.image if has_image else '(none)'}")
    print(f"  Prompt: {args.prompt!r}")

    if has_image and args.vision_cpu:
        # CPU vision encoder (fp32 precision, for quality comparison)
        print(f"\n--- Vision Encoder (CPU) ---")
        timer_vis = time.perf_counter()
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from PIL import Image
        model_dir = os.path.join(script_dir, cfg["paths"]["hf_model_dir"])
        hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, device_map="cpu")
        if not hasattr(hf_model, 'visual') and hasattr(hf_model, 'model') and hasattr(hf_model.model, 'visual'):
            hf_model.visual = hf_model.model.visual
        hf_model.eval()
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        img_size = cfg["vision"].get("image_size", 336)
        image = Image.open(args.image).convert("RGB").resize((img_size, img_size))
        messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
        pixel_values_hf = inputs["pixel_values"].to(torch.bfloat16)
        image_grid_thw = inputs["image_grid_thw"]
        with torch.no_grad():
            vis_output = hf_model.visual(pixel_values_hf, grid_thw=image_grid_thw)
        # DMA vision output to FPGA
        num_merged = vis_output.shape[0]
        ue.dma_to_accelerator_memory(ue.VIS_ENCODER_OUT_DRAM, vis_output.to(torch.bfloat16).flatten())
        ue._vis_num_tokens = num_merged
        ue._image_grid_thw = image_grid_thw
        t_total = time.perf_counter() - timer_vis
        t_compile = 0
        print(f"CPU vision encoder: {num_merged} tokens in {t_total:.2f}s")
        del hf_model
    elif has_image:
        # FPGA vision encoder
        print(f"\n--- Vision Encoder (FPGA) ---")
        timer_vis = time.perf_counter()
        pixel_values = process_image(args.image)
        print(f"Image loaded: {args.image} -> {pixel_values.shape}")
        ue.prepare_encoder_input(pixel_values)  # sets _cu_window_seqlens before compile
        encoder_addr = ue.compile_encoder()
        t_compile = time.perf_counter() - timer_vis
        ue.run_encoder(encoder_addr, pixel_values)
        t_total = time.perf_counter() - timer_vis
        print(f"Vision encoder: compile {t_compile:.2f}s + run {t_total - t_compile:.2f}s = {t_total:.2f}s total")

    if args.prompt is not None or has_image:
        tok_path = os.path.join(script_dir, cfg["paths"]["hf_model_dir"])
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
        prompt_text = args.prompt or "Describe this image."
        if has_image:
            # Build prompt with image placeholder tokens for Qwen2.5-VL
            # The vision_token_id (image_pad) placeholders will be replaced with vision embeddings
            num_vis_tokens = getattr(ue, '_vis_num_tokens', cfg["vision"]["num_merged_tokens"])
            # Qwen2.5-VL uses <|image_pad|> (151655) as the single placeholder in chat template;
            # we need to expand it to num_vis_tokens copies for the prefill sequence.
            # The image_pad tokens are what run_prefill() looks for to replace with vision embeddings.
            image_pad_id = 151655  # <|image_pad|>
            vision_start_id = 151652  # <|vision_start|>
            vision_end_id = 151653    # <|vision_end|>
            # Build text-only prompt then insert vision placeholders
            conversation_text = [{"role": "user", "content": prompt_text}]
            prompt_with_template = tokenizer.apply_chat_template(
                conversation_text, tokenize=False, add_generation_prompt=True
            )
            text_tokens = list(tokenizer.encode(prompt_with_template, add_special_tokens=False))
            # Find user content start (after "user\n")
            # Insert: <|vision_start|> + <|image_pad|>*N + <|vision_end|> + \n
            vis_placeholder = [vision_start_id] + [image_pad_id] * num_vis_tokens + [vision_end_id]
            # Insert after system header (first "user\n" = tokens [151644, 872, 198])
            # Find the user content start
            insert_pos = 0
            for idx in range(len(text_tokens) - 2):
                if text_tokens[idx:idx+3] == [151644, 872, 198]:  # <|im_start|>user\n
                    insert_pos = idx + 3
                    break
            prefill_seq = tuple(text_tokens[:insert_pos] + vis_placeholder + text_tokens[insert_pos:])
        else:
            conversation = [{"role": "user", "content": prompt_text}]
            prompt_with_template = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            prefill_seq = tuple(tokenizer.encode(prompt_with_template, add_special_tokens=False))
        print(f"Prefill from prompt ({len(prefill_seq)} tokens): {prompt_text!r}")
        print(f"Sequence ids: {prefill_seq[:20]}{'...' if len(prefill_seq) > 20 else ''}")
    else:
        prefill_seq = tuple(cfg["default_prefill_tokens"])

    if has_image:
        image_grid_thw = getattr(ue, '_image_grid_thw', None)
        ue.load_mrope_for_prefill(prefill_seq, image_grid_thw=image_grid_thw)

    # Compact token display
    _img_id = 151655
    _seq = list(prefill_seq)
    _parts, _i = [], 0
    while _i < len(_seq):
        if _seq[_i] == _img_id:
            _cnt = 0
            while _i < len(_seq) and _seq[_i] == _img_id:
                _cnt += 1; _i += 1
            _parts.append(f"<|image_pad|>*{_cnt}")
        else:
            _parts.append(str(_seq[_i])); _i += 1
    print(f"Prompt ({len(prefill_seq)} tokens): [{', '.join(_parts)}]")

    print(f"\n--- Prefill compile ---")
    timer = time.perf_counter()
    prefill_program_addr, gflops_prefill = ue.compile_prefill(seq_len=len(prefill_seq))
    t_prefill_compile = time.perf_counter() - timer
    print(f"  {t_prefill_compile:.2f}s")

    print(f"\n--- Prefill run ({len(prefill_seq)} tokens) ---")
    timer = time.perf_counter()
    ue.run_prefill(prefill_program_addr, prefill_seq=prefill_seq, gflops=gflops_prefill, has_image=has_image)
    latency_prefill = time.perf_counter() - timer
    prefill_hw_gflops = getattr(ue, '_last_hw_gflops', 0)
    print(f"  {latency_prefill:.2f}s")

    print(f"\n--- Decode compile ---")
    timer = time.perf_counter()
    decoder_bin_path = os.path.join(script_dir, "qwen2.5_vl_3b_bin", "decoder_program.bin")
    decoder_meta_path = os.path.join(script_dir, "qwen2.5_vl_3b_bin", "decoder_program.json")
    if os.path.exists(decoder_bin_path) and os.path.exists(decoder_meta_path):
        with open(decoder_meta_path, "r") as f:
            meta = json.load(f)
        if "instruction_counts" in meta:
            decoder_program_sizes = [c * 32 for c in meta["instruction_counts"]]
        else:
            decoder_program_sizes = meta["program_sizes"]
        gflops_per_token = meta["total_flops"]
        print(f"  loaded from cache")
    else:
        decoder_program_sizes, gflops_per_token = ue.compile_decoder()
        print(f"  compiled fresh")
    decoder_base_addr, _ = ue.load_instructions(decoder_bin_path)
    t_decode_compile = time.perf_counter() - timer
    print(f"  {t_decode_compile:.2f}s")

    # Restore RoPE for decoder
    if has_image:
        ue.restore_rope_for_decoder()
        ue._rope_offset = getattr(ue, '_mrope_delta', 0)
    else:
        ue._rope_offset = 0

    print(f"\n--- Decode run ---")
    timer = time.perf_counter()
    token_cnt_decoded = ue.run_decoder(decoder_program_sizes, decoder_base_addr, token_id=prefill_seq[-1], gflops_per_token=gflops_per_token, repetition_penalty=args.rep_penalty)
    latency_decoder = time.perf_counter() - timer
    n_new = token_cnt_decoded - len(prefill_seq)
    print(f"\n  {latency_decoder:.2f}s ({n_new} tokens, {latency_decoder/max(n_new,1):.2f}s/tok)")

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    if has_image:
        vis_flops = getattr(ue, '_vis_total_flops', 0)
        vis_gflops = getattr(ue, '_vis_gflops', 0)
        vis_run = t_total - t_compile
        print(f"  Vision compile:   {t_compile:.2f}s")
        print(f"  Vision run:       {vis_run:.2f}s  ({vis_gflops:.2f} GFLOPS)")
    print(f"  Prefill compile:  {t_prefill_compile:.2f}s")
    prefill_flops = gflops_prefill if isinstance(gflops_prefill, (int, float)) else 0
    print(f"  Prefill run:      {latency_prefill:.2f}s  ({prefill_hw_gflops:.2f} GFLOPS)  ({len(prefill_seq)} tokens)")
    print(f"  Decode compile:   {t_decode_compile:.2f}s")
    print(f"  Decode run:       {latency_decoder:.2f}s  ({n_new} tokens, {latency_decoder/max(n_new,1):.2f}s/tok)")
    total = (t_total if has_image else 0) + t_prefill_compile + latency_prefill + t_decode_compile + latency_decoder
    print(f"  ──────────────────────────")
    print(f"  Total:            {total:.2f}s")

if __name__ == "__main__":
    main()
