#!/usr/bin/env python3
"""Qwen2.5-VL-3B on accelerator. LM and vision precisions are
config-driven via qwen2.5_vl_3b_config.json::precision.{lm,vision}
(values: int4 / fp4 / if4). Defaults: lm=if4 (eval-winning text codec
per src/models/qwen2.5_VL_3b/compare/summary.md), vision=int4 (legacy
released codec, byte-identical to the prior Q4_64 path). All
quantization goes through src/template/quant_lib.py."""

import json
import math
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
from user_dma_core import DMA_DEVICE_H2C, TYPE, UE_FMAX_CONTEXT_SIZE, UE_VECTOR_SIZE, URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS, UE_MODE, set_dma_device
from nn_lib import eltwise_add_core_dram, eltwise_mul_core_dram, rms_norm_core_dram_post_add
from user_dma_core import UnifiedEngine, DRAM_INSTRUCTION_ADDR, INSTRUCTION_REG_REWRITE, MEMCPY_TYPE
from user_dma_core import ue_35bit_addr_shifter
import quant_lib

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

def load_weight_cache(bin_path, region=None):
    """Load bin+json weight file. Returns {tensor_name: raw_numpy_data}.

    When ``region`` is provided, ``bin_path`` is expected to be the unified
    ``params.bin`` whose sidecar ``params.json`` contains a ``regions`` map
    (each entry has ``offset``, ``size``, and a ``manifest`` of tensors with
    region-relative offsets). The selected region's bytes are sliced out and
    its sub-manifest is used. Without ``region``, the legacy flat layout
    (top-level manifest of {name: {offset, size}}) is honored.
    """
    json_path = bin_path.rsplit('.', 1)[0] + '.json'
    with open(json_path) as f:
        manifest = json.load(f)
    with open(bin_path, 'rb') as f:
        raw = f.read()
    if region is not None:
        regions = manifest.get('regions') or {}
        if region not in regions:
            raise KeyError(f"region {region!r} not in {json_path} (regions={list(regions)})")
        r = regions[region]
        base = int(r['offset'])
        size = int(r['size'])
        raw = raw[base:base + size]
        sub_manifest = r['manifest']
    else:
        sub_manifest = manifest
    cache = {}
    for name, meta in sub_manifest.items():
        cache[name] = np.frombuffer(raw[meta['offset']:meta['offset'] + meta['size']], dtype=np.uint8).copy()
    return cache

def store_identity_matrix(ue):
    """Store identity matrix in DRAM once. Returns DRAM address."""
    bpe = 2
    size = UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe
    addr = ue.get_params_dram_addr()
    ue.dma_write(DMA_DEVICE_H2C, addr, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16), size)
    ue.allocate_params_dram(size)
    return addr



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


def _qs_pack(precision: str, tensor: torch.Tensor):
    """Run a 64-block codec via quant_lib and emit the scale-then-data
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
        data_bytes, scale_bytes = quant_lib.quantize(precision, bf, block_size=64)
        return np.frombuffer(scale_bytes + data_bytes, dtype=np.uint8), n_blocks
    # Generic path: flatten + zero-pad to a multiple of 64. Used for
    # arbitrary-rank tensors (e.g. the 5D Conv3D patch_embed weight).
    bf = bf.flatten()
    n_blocks = (bf.numel() + 63) // 64
    if bf.numel() != n_blocks * 64:
        bf = torch.nn.functional.pad(bf, (0, n_blocks * 64 - bf.numel()))
    bf = bf.view(1, -1)
    data_bytes, scale_bytes = quant_lib.quantize(precision, bf, block_size=64)
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
    Validates against the codecs the quant_lib wrapper actually supports."""
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
    """Generate LM weight bin using the given quant_lib precision
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
    (3420→3456). The given quant_lib precision ('int4' / 'fp4' / 'if4')
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
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    missing_required = not os.path.exists(config_path)
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f).get("weight_map", {})
        required_files = set(weight_map.values())
        missing_required = missing_required or any(
            not os.path.exists(os.path.join(model_dir, name))
            for name in required_files)
    if missing_required:
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

def weight_bin_generate(script_dir: str | None = None, output_params: str | None = None) -> str:
    """Generate the unified params.bin (LM + vision) from a Hugging Face model.

    LM and vision weight bytes are generated into temporary sidecar files,
    then concatenated into a single ``params.bin`` whose ``params.json``
    sidecar carries a ``regions`` map ({lm,vision} → {offset, size,
    manifest}) where each region's manifest holds region-relative tensor
    offsets. Returns the unified params.bin path."""
    script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
    cfg = _load_config(script_dir)
    paths = cfg["paths"]
    params_path = output_params or os.path.join(script_dir, paths["params"])
    os.makedirs(os.path.dirname(params_path), exist_ok=True)
    bin_dir = os.path.dirname(params_path)

    lm_tmp = os.path.join(bin_dir, "_params_lm.tmp.bin")
    vis_tmp = os.path.join(bin_dir, "_params_vision.tmp.bin")

    model, model_dir = _ensure_hf_model(script_dir, cfg)
    generate_lm_weights(model, lm_tmp, precision=_lm_precision(cfg))
    generate_vision_weights(model, vis_tmp, precision=_vision_precision(cfg))
    del model

    lm_json = lm_tmp.rsplit('.', 1)[0] + '.json'
    vis_json = vis_tmp.rsplit('.', 1)[0] + '.json'
    with open(lm_json) as f:
        lm_manifest = json.load(f)
    with open(vis_json) as f:
        vis_manifest = json.load(f)
    lm_size = os.path.getsize(lm_tmp)
    vis_size = os.path.getsize(vis_tmp)

    # Concatenate LM then vision into params.bin (raw region dump).
    CHUNK = 4 * 1024 * 1024
    with open(params_path, 'wb') as out:
        for tmp in (lm_tmp, vis_tmp):
            with open(tmp, 'rb') as src:
                while True:
                    buf = src.read(CHUNK)
                    if not buf:
                        break
                    out.write(buf)

    regions = {
        "lm":     {"offset": 0,       "size": lm_size,  "manifest": lm_manifest},
        "vision": {"offset": lm_size, "size": vis_size, "manifest": vis_manifest},
    }
    params_json = params_path.rsplit('.', 1)[0] + '.json'
    with open(params_json, 'w') as f:
        json.dump({"regions": regions}, f)

    for tmp in (lm_tmp, vis_tmp, lm_json, vis_json):
        try:
            os.remove(tmp)
        except OSError:
            pass
    print(f"Unified params.bin: lm={lm_size/1048576:.1f} MB + vision={vis_size/1048576:.1f} MB -> {params_path}")
    return params_path

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
        self.PREFILL_MAX_SEQ_LEN = int(model.get("prefill_max_seq_len", 64))
        self.LAYER_SIZE = fi["num_layers"]
        self.EMBEDDING_ELEMENTS = fi["embedding_vocab"]
        # Dynamic-PBI GPR layout (mirrors Dave's qwen3 conversion):
        #   reg 1 = TMP_REG          — scratch for reg_mul_imm + add_imm address math
        #   reg 2 = GF_SEQ_LEN_REG   — runtime row count for matmul/norm ops (M=seq_len)
        #   reg 3 = GF_Q_SEQ_LEN_REG — for ops with M = seq_len * group_size
        #   reg 4 = GF_ALIGNED_SEQ_LEN_REG — active KV length rounded to 64
        # Dynamic GPR allocation (via alloc_isa_reg) starts at 5.
        fixed = self._cfg["fixed_isa_regs"]
        self.TMP_REG            = fixed["TMP_REG"]
        self.gf_seq_len         = fixed["GF_SEQ_LEN_REG"]
        self.gf_q_seq_len       = fixed["GF_Q_SEQ_LEN_REG"]
        self.gf_aligned_seq_len = fixed["GF_ALIGNED_SEQ_LEN_REG"]
        self._isa_reg_counter = 5
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

    def _emit_pbi_scatter_per_token(self, *, read_base, read_stride_bytes,
                                    write_specs, sram_byte_addr, element_count,
                                    gf_seq_len, template_seq_len):
        """PBI runtime-loop scatter: gf_seq_len iterations, each iter copies
        element_count rows from read_base + t*read_stride_bytes to each
        (dst_base, dst_stride) in write_specs. SRAM→DRAM scatters use pbi_init
        pointers that auto-advance by their stride per iteration. The captured
        bin is seq_len-agnostic up to MAX_CONTEXT_SIZE."""
        bpe = self.bytes_per_element
        bytes_per_call = element_count * bpe
        _, sram_words = self.sram_address_to_uram_address(sram_byte_addr)

        ptr_ws = [self.alloc_inst_ptr() for _ in write_specs]
        for ptr_w, (dst_base, _stride) in zip(ptr_ws, write_specs):
            self.generate_instruction_pbi_init(
                dram_shared_addr=dst_base,
                dma_length=bytes_per_call,
                uram_a_start_addr=sram_words,
                uram_b_start_addr=sram_words,
                inst_pointer_idx=ptr_w,
            )

        t_reg = self.alloc_isa_reg()
        self.generate_instruction_add_set(t_reg, 0)

        self.loop_start(loop_cnt=template_seq_len, gpr_loop_cnt=gf_seq_len)
        self.generate_instruction_reg_mul_imm(
            self.TMP_REG, t_reg, ue_35bit_addr_shifter(read_stride_bytes))
        self.generate_instruction_add_imm(
            self.TMP_REG, ue_35bit_addr_shifter(read_base), self.TMP_REG)
        self.accelerator_memory_to_sram(
            accelerator_dram_address=0,
            sram_address=sram_byte_addr,
            element_size=element_count,
            general_reg_src=self.TMP_REG,
        )
        for ptr_w, (_base, dst_stride) in zip(ptr_ws, write_specs):
            self.sram_to_accelerator_memory(
                sram_address=0,
                accelerator_dram_address=dst_stride,
                element_size=element_count,
                inst_pointer_idx=ptr_w,
                memcpy_length_bytes=0,
            )
        self.generate_instruction_add_inc(t_reg)
        self.loop_end()

        self.release_isa_reg()
        for ptr in reversed(ptr_ws):
            self.release_inst_ptr(ptr)

    def _flash_attention_core_cached(self, **kwargs) -> int:
        """Compatibility wrapper: translate legacy flash args to unified attention."""
        kwargs.pop("ATTN_P_DRAM_ADDR", None)
        kwargs.pop("gpr_bucket_idx", None)
        kwargs.pop("num_buckets", None)
        kwargs.pop("gpr_ret_id", None)
        seq_len = int(kwargs.pop("seq_len"))
        head_dim = int(kwargs["head_dim"])
        kwargs.setdefault("IDENTITY_DRAM_ADDR", self.IDENTITY_DRAM_ADDR)
        return self.unified_attention_core(
            batch=seq_len,
            aligned_seq_len=seq_len,
            head_dim=head_dim,
            Q_DRAM_ADDR=kwargs["Q_DRAM_ADDR"],
            K_DRAM_ADDR=kwargs["K_DRAM_ADDR"],
            V_DRAM_ADDR=kwargs["V_DRAM_ADDR"],
            BIAS_DRAM_ADDR=kwargs["BIAS_DRAM_ADDR"],
            OUTPUT_DRAM_ADDR=kwargs["OUTPUT_DRAM_ADDR"],
            SCRATCH_DRAM_ADDR=kwargs["SCRATCH_DRAM_ADDR"],
            IDENTITY_DRAM_ADDR=kwargs["IDENTITY_DRAM_ADDR"],
        )

    def dma_write(self, device, addr, data, size):
        """Skip redundant identity-matrix DMA writes (already in DRAM from pre-allocation)."""
        if (self._identity_dram_written
                and self._identity_dram_addr is not None
                and addr == self._identity_dram_addr
                and size == self._IDENTITY_MAT_BYTES):
            return size  # already written; report the requested transfer as satisfied
        return super().dma_write(device, addr, data, size)

    def reset_isa_reg_counter(self) -> None:
        """Reset PBI ISA-reg allocator back to 5 (regs 1-4 are fixed/reserved)."""
        self._isa_reg_counter = 5

    def alloc_isa_reg(self, reset: bool = False) -> int:
        """Allocate next ISA register. Regs 1-4 are fixed dynamic regs."""
        if reset:
            self._isa_reg_counter = 5

        if self._isa_reg_counter > 31:
            raise ValueError("Exceeded available ISA registers (max 31)")

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

        # Generate unified params.bin if missing
        params_bin_path = os.path.join(self.script_dir, self._cfg["paths"]["params"])
        params_json_path = params_bin_path.rsplit('.', 1)[0] + '.json'
        if not os.path.exists(params_bin_path) or not os.path.exists(params_json_path):
            _original_print("params.bin not found, generating...")
            weight_bin_generate(script_dir=self.script_dir, output_params=params_bin_path)

        # Load LM weights (lm region of unified params.bin)
        print(f"  Reading LM weight region from params.bin...", flush=True)
        lm_cache = load_weight_cache(params_bin_path, region="lm")

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
        self._load_vision_weights(params_bin_path)

        print(f"    Allocate weights end at DRAM address: 0x{self.get_params_dram_addr():X}, usage: {self.get_params_dram_usage()} bytes")
        print("Tokenizer loaded successfully.")

    def _load_vision_weights(self, params_bin_path: str, hf_model=None) -> None:
        """Load vision encoder weights from the unified params.bin (vision
        region). Manifest-key suffix comes from cfg['precision']['vision'] —
        same string the generator wrote, so changing precision in the
        config switches both ends."""
        vis_cache = load_weight_cache(params_bin_path, region="vision")
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
        prefill_q_seq = self.PREFILL_MAX_SEQ_LEN * self.group_size
        prefill_aligned = ((prefill_q_seq + 63) // 64) * 64
        lm_attn_batch = max(prefill_aligned, self.group_size)
        lm_attn_aligned = max(prefill_aligned, ((self.MAX_CONTEXT_SIZE + 63) // 64) * 64)
        lm_scratch_elems = ((ahd + lm_attn_aligned) * lm_attn_aligned
                            + lm_attn_batch * ahd)
        self.LAYER0_FLASH_SCRATCH_DRAM = self.allocate_tensor_dram(lm_scratch_elems * bpe)
        # PBI prefill packs FLASH_Q with qpkv rows per token (q_seq_len = PREFILL_MAX_SEQ_LEN * qpkv),
        # so BIAS and ATTN_P are sized to (prefill_aligned)^2 where prefill_aligned = PREFILL_MAX_SEQ_LEN*qpkv aligned to UE_VECTOR_SIZE.
        self.LAYER0_FLASH_BIAS_DRAM = self.allocate_tensor_dram(prefill_aligned * prefill_aligned * bpe)
        self.LAYER0_FLASH_ATTN_P_DRAM = self.allocate_tensor_dram(prefill_aligned * prefill_aligned * bpe)

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
        # Per-vocab additive repetition-penalty bias — the LM-head matmul's C term
        # (bias_mode="broadcast_N", write_back_disable). The HW argmax of
        # (logits + bias) returns the penalized token directly — NO logit readback
        # (llama3.2_1b style on-FPGA penalty). All-zero buffer = pure greedy.
        self.PENALTY_BIAS_DRAM = self.allocate_tensor_dram(1 * self.EMBEDDING_ELEMENTS * bpe)

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
        vis_scratch_elems = (VD_PAD + VS) * VS + VS * VD_PAD
        self.VIS_ATTN_SCRATCH_DRAM = self.allocate_tensor_dram(vis_scratch_elems * bpe)
        # §7 shared-subroutine vision flash: FIXED operand buffers the one flash
        # body reads/writes; each per-(head,window) call site marshals its segment
        # in/out. Sized for the largest segment (full attention = VS patches).
        self.VIS_FLASH_Q_DRAM   = self.allocate_tensor_dram(VS * VD_PAD * bpe)
        self.VIS_FLASH_K_DRAM   = self.allocate_tensor_dram(VS * VD_PAD * bpe)
        self.VIS_FLASH_V_DRAM   = self.allocate_tensor_dram(VS * VD_PAD * bpe)
        self.VIS_FLASH_OUT_DRAM = self.allocate_tensor_dram(VS * VD_PAD * bpe)
        # Bucket dispatcher softmax-intermediate scratch (VS×VS for full attention)
        self.VIS_ATTN_P_DRAM    = self.allocate_tensor_dram(VS * VS * bpe)
        self.VIS_ATTN_BIAS_DRAM = self.allocate_tensor_dram(VS * VS * bpe)
        self.dma_to_accelerator_memory(
            self.VIS_ATTN_BIAS_DRAM,
            torch.zeros(VS, VS, dtype=torch.bfloat16))
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
        # 2D RoPE table, interleaved per-token rows [cos(VD_PAD) || sin(VD_PAD)] so the
        # shared PBI rope core loads both with ONE auto-advancing pointer (sin = cos +
        # row_bytes). Padded from head_dim 80 to VD_PAD=128 for 64-aligned rope_hf_core_dram.
        # Tiled VN times (one block per head) so a single M=VN*VS rope call covers all
        # heads in one PBI hardware loop instead of 16 per-head calls (Tier-1 shrink).
        self.VIS_ROPE_DRAM = self.allocate_tensor_dram(VN * VS * 2 * VD_PAD * bpe)
        self.VIS_ROPE_COS_DRAM = self.VIS_ROPE_DRAM
        self.VIS_ROPE_SIN_DRAM = self.VIS_ROPE_DRAM + VD_PAD * bpe
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

    def program_execute(self, program_start_addr: int = user_dma_core.DRAM_INSTRUCTION_ADDR,
                        timeout: float = 120.0, flops: float = None, gflops: float = None,
                        verbose: bool = True) -> tuple[float, float]:
        """Execute compiled program from DRAM. Returns (latency_us, flop_rate_gflops).

        Either ``flops`` (raw FLOPs) or ``gflops`` (already in GFLOPs) may be passed
        for throughput reporting; both map to the same internal value. flop_rate is
        0.0 when neither is provided.
        """
        if verbose:
            print(f"  Running program on FPGA (DRAM addr 0x{program_start_addr:X})...", flush=True)
        self.start_execute_from_dram(program_start_addr)
        if verbose:
            wall_s = self._wait_with_heartbeat(lambda: self.wait_queue(timeout), label="FPGA")
        else:
            # Still measure wall time — _corrected_hw_latency_us reads the HW
            # cycle counter but needs wall time to fix 32-bit overflow wrap.
            _t0 = time.perf_counter()
            self.wait_queue(timeout)
            wall_s = time.perf_counter() - _t0
        latency_us = self._corrected_hw_latency_us(wall_s)
        total_flops = flops if flops is not None else gflops
        flop_rate = 0.0
        if total_flops is not None:
            flop_rate = total_flops / (latency_us * 1e-6) / 1e9
            self._last_hw_gflops = flop_rate
            self._last_total_flops = total_flops
            if verbose:
                print(f"    Total program execution latency = {latency_us:.0f} us")
                print(f"    Throughput: {flop_rate:.2f} GFLOPS")
        elif verbose:
            print(f"    Total program execution latency = {latency_us:.0f} us")
        return latency_us, flop_rate

    def _vis_transpose_10(self, input_addr: int, output_addr: int, d0: int, d1: int, last: int) -> None:
        """Transpose [d0, d1, last] -> [d1, d0, last] (permute [1,0,2]) via strided DMA.

        One gather DMA pair per OUTPUT block (d1 iterations) instead of
        bf16_permute_core_v2's per-row unroll (d0*d1 rows): for output block i1,
        gather the d0 strided `last`-chunks {input[i0, i1, :]} into contiguous SRAM,
        then store. For the head<->token swaps (one dim = VN = 16) this is ~16 or
        ~576 iters vs ~9216 — the dominant encoder bin-size cost. SRAM holds one
        d0*last slab (< URAM_NEAR_FULL_ELEMENTS for VS=576)."""
        # NOTE: looping the smaller dim with a strided WRITE (scatter) would cut the
        # merge from 576 to 16 iters, but a many-chunk strided write produces wrong
        # output on device (verified 2026-06-10: ATTN_RESULT cos=0.06) even though the
        # symmetric strided READ is correct. Stay on the gather (strided read) path.
        bpe = self.bytes_per_element
        for i1 in range(d1):
            self.accelerator_memory_to_sram(
                accelerator_dram_address=input_addr + i1 * last * bpe,
                sram_address=0x00000, element_size=d0 * last,
                stride_bytes_per_chunk=last * bpe, stride_jump_bytes=d1 * last * bpe)
            self.sram_to_accelerator_memory(
                sram_address=0x00000,
                accelerator_dram_address=output_addr + i1 * d0 * last * bpe,
                element_size=d0 * last)

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
        # NOTE: no standalone encoder_program.bin — the ONLY instruction bin is
        # the unified programs.bin written by compile_all. Direct callers
        # (e.g. the compare script's per-layer vision checks) just compile fresh
        # into program DRAM; nothing is cached to disk here.

        global _SILENT_MODE
        _SILENT_MODE = True

        _bf16_mode = getattr(self, '_vision_bf16_mode', False)
        def vis_matmul(M, K, N, A, la_key_prefix, la, OUT, bias=None, silu_en=False, gelu_en=False, gpr_M_reg=None):
            if _bf16_mode and f'{la_key_prefix}_weight_bf16' in la:
                self.matmat_mul_core(
                    M=M, K=K, N=N, A_DRAM_ADDR=A,
                    B_DRAM_ADDR=la[f'{la_key_prefix}_weight_bf16'],
                    OUTPUT_DRAM_ADDR=OUT,
                    C_DRAM_ADDR=la.get(f'{la_key_prefix}_bias_bf16', bias), bias_mode="broadcast_N",
                    silu_enable=silu_en, gelu_enable=gelu_en, gpr_M_reg=gpr_M_reg)
            else:
                self.matmat_mul_core(
                    M=M, K=K, N=N, A_DRAM_ADDR=A,
                    B_DRAM_ADDR=la[f'{la_key_prefix}_data'],
                    OUTPUT_DRAM_ADDR=OUT,
                    C_DRAM_ADDR=bias, bias_mode="broadcast_N",
                    is_B_quantized=True, data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=la[f'{la_key_prefix}_scale'],
                    silu_enable=silu_en, gelu_enable=gelu_en, gpr_M_reg=gpr_M_reg)

        total_bytes = 0

        _original_print(f"  Compiling vision encoder: {total_layers} layers (single program)...")
        self.start_capture()

        # Runtime row-count register (M=VS) shared by all M-agnostic PBI ops in the
        # encoder — matmuls, norm1, and the rope core. Constant across heads/layers;
        # primed once, not consumed by the hardware loop. Drives the per-layer ISA to
        # be M-agnostic (single hardware loop instead of VS-row unrolling).
        self.reset_isa_reg_counter()
        vis_M_reg = self.alloc_isa_reg()
        self.generate_instruction_add_set(vis_M_reg, VS)

        def _vis_flash_call(q_src, k_src, v_src, out_dst, seg_len):
            """Marshal one segment's Q/K/V into fixed buffers and run unified attention."""
            elems = seg_len * VD_PAD
            for src, dst in ((q_src, self.VIS_FLASH_Q_DRAM),
                             (k_src, self.VIS_FLASH_K_DRAM),
                             (v_src, self.VIS_FLASH_V_DRAM)):
                self.accelerator_memory_to_sram(src, 0x00000, elems)
                self.sram_to_accelerator_memory(0x00000, dst, elems)
            self.unified_attention_core(
                batch=seg_len,
                aligned_seq_len=seg_len,
                head_dim=VD_PAD,
                Q_DRAM_ADDR=self.VIS_FLASH_Q_DRAM,
                K_DRAM_ADDR=self.VIS_FLASH_K_DRAM,
                V_DRAM_ADDR=self.VIS_FLASH_V_DRAM,
                BIAS_DRAM_ADDR=self.VIS_ATTN_BIAS_DRAM,
                OUTPUT_DRAM_ADDR=self.VIS_FLASH_OUT_DRAM,
                SCRATCH_DRAM_ADDR=self.VIS_ATTN_SCRATCH_DRAM,
                IDENTITY_DRAM_ADDR=self.IDENTITY_DRAM_ADDR)
            self.accelerator_memory_to_sram(self.VIS_FLASH_OUT_DRAM, 0x00000, elems)
            self.sram_to_accelerator_memory(0x00000, out_dst, elems)

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
                GAMMA_DRAM_ADDR=la['norm1_weight'], gpr_M_reg=vis_M_reg)

            VN_VD_PAD = VN * VD_PAD
            if _bf16_mode and 'qk_weight_bf16' in la:
                self.matmat_mul_core(
                    M=VS, K=VH, N=VN_VD_PAD * 2,
                    A_DRAM_ADDR=self.VIS_NORM_OUT_DRAM,
                    B_DRAM_ADDR=la['qk_weight_bf16'],
                    OUTPUT_DRAM_ADDR=self.VIS_QK_DRAM,
                    C_DRAM_ADDR=la['qk_bias_bf16'], bias_mode="broadcast_N",
                    gpr_M_reg=vis_M_reg)
            else:
                self.matmat_mul_core(
                    M=VS, K=VH, N=VN_VD_PAD * 2,
                    A_DRAM_ADDR=self.VIS_NORM_OUT_DRAM,
                    B_DRAM_ADDR=la['qk_data'],
                    OUTPUT_DRAM_ADDR=self.VIS_QK_DRAM,
                    C_DRAM_ADDR=la['qk_bias'], bias_mode="broadcast_N",
                    is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=la['qk_scale'], gpr_M_reg=vis_M_reg)

            if _bf16_mode and 'v_weight_bf16' in la:
                self.matmat_mul_core(
                    M=VS, K=VH, N=VN_VD_PAD,
                    A_DRAM_ADDR=self.VIS_NORM_OUT_DRAM,
                    B_DRAM_ADDR=la['v_weight_bf16'],
                    OUTPUT_DRAM_ADDR=self.VIS_V_DRAM,
                    C_DRAM_ADDR=la['v_bias_bf16'], bias_mode="broadcast_N",
                    gpr_M_reg=vis_M_reg)
            else:
                self.matmat_mul_core(
                    M=VS, K=VH, N=VN_VD_PAD,
                    A_DRAM_ADDR=self.VIS_NORM_OUT_DRAM,
                    B_DRAM_ADDR=la['v_data'],
                    OUTPUT_DRAM_ADDR=self.VIS_V_DRAM,
                    C_DRAM_ADDR=la['v_bias'], bias_mode="broadcast_N",
                    is_B_quantized=True, data_type=TYPE.IF4,
                    SCALE_DRAM_ADDR=la['v_scale'], gpr_M_reg=vis_M_reg)

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

                # Token-major [VS, VN, VD_PAD] -> head-major [VN, VS, VD_PAD] via
                # strided DMA (16 iters), replacing the per-row-unrolled permute.
                self._vis_transpose_10(extract_buf, pad_dst, d0=VS, d1=VN, last=VD_PAD)

            self._vis_transpose_10(self.VIS_V_DRAM, self.VIS_V_PAD_DRAM,
                                   d0=VS, d1=VN, last=VD_PAD)

            # RoPE via the shared PBI core (rope_hf_core_dram). Per token it loads x plus
            # the interleaved [cos||sin] row through 2 auto-advancing load pointers (+1
            # store) — the proven-good pointer count, unlike the old 3-separate-load path
            # whose 3rd pointer didn't reliably advance and corrupted vision output.
            # cos/sin are position-only, so the same table is reused for every head.
            # One rope call per buffer over ALL heads (M=VN*VS, head-major Q/K_PAD).
            # The VN-tiled cos/sin table makes row h*VS+t use cos/sin[t]. Replaces the
            # old 16-heads x (Q,K) = 32 per-head calls with 2 PBI-looped calls.
            # Re-prime the shared M register to VN*VS for the rope, then back to VS
            # (no extra ISA register — the encoder is at the reg budget).
            self.generate_instruction_add_set(vis_M_reg, VN * VS)
            for buf_dram in (self.VIS_Q_PAD_DRAM, self.VIS_K_PAD_DRAM):
                self.rope_hf_core_dram(
                    M=VN * VS, N=VD_PAD,
                    input_dram_addr=buf_dram, output_dram_addr=buf_dram,
                    cos_dram_addr=self.VIS_ROPE_COS_DRAM,
                    sin_dram_addr=self.VIS_ROPE_SIN_DRAM,
                    gpr_M_reg=vis_M_reg)
            self.generate_instruction_add_set(vis_M_reg, VS)

            # flash_attention_core scales by 1/sqrt(128) instead of 1/sqrt(80).
            # The softer attention (sqrt(128)) is more robust to INT4 quantization noise
            # and empirically gives better end-to-end results than correcting to sqrt(80).
            fullatt_indexes = set(vis_cfg.get("fullatt_block_indexes", [7, 15, 23, 31]))
            cu_win = getattr(self, '_cu_window_seqlens', [0, VS])
            is_full_attn = (layer_idx in fullatt_indexes)

            if is_full_attn:
                for h in range(VN):
                    base = h * head_stride_pad
                    _vis_flash_call(self.VIS_Q_PAD_DRAM + base, self.VIS_K_PAD_DRAM + base,
                                    self.VIS_V_PAD_DRAM + base, self.VIS_ATTN_OUT_DRAM + base, VS)
            else:
                for h in range(VN):
                    for w_idx in range(len(cu_win) - 1):
                        w_start = cu_win[w_idx]
                        w_len = cu_win[w_idx + 1] - w_start
                        if w_len <= 0:
                            continue
                        off = h * head_stride_pad + w_start * VD_PAD * bpe
                        _vis_flash_call(self.VIS_Q_PAD_DRAM + off, self.VIS_K_PAD_DRAM + off,
                                        self.VIS_V_PAD_DRAM + off, self.VIS_ATTN_OUT_DRAM + off, w_len)

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

            # Head-major [VN, VS, VD] -> token-major [VS, VN, VD] via strided DMA.
            self._vis_transpose_10(self.VIS_ATTN_TRIM_DRAM, self.VIS_ATTN_RESULT_DRAM,
                                   d0=VN, d1=VS, last=VD)

            vis_matmul(VS, VH, VH, self.VIS_ATTN_RESULT_DRAM, 'o', la,
                       self.VIS_O_PROJ_DRAM, bias=la['o_bias'], gpr_M_reg=vis_M_reg)

            # norm2 kept LEGACY (static): its PBI variant uses a 2-load+2-store pointer
            # pattern that corrupts output on device (verified 2026-05-29) AND saved only
            # ~2MB — net loss. See [[project_qwenvl_rope_pbi_bug]].
            # post-attn residual + norm2, split into an efficient chunked eltwise-add
            # and an M-looped rms_norm instead of the fused post_add (whose legacy form
            # is M-unrolled over VS rows ≈ 2400 instr/layer, and whose PBI form corrupts
            # on device). Same result: residual = h_in + o_proj; norm = rms_norm(residual).
            eltwise_add_core_dram(self, size=VS * VH,
                A_DRAM_ADDR=h_in, B_DRAM_ADDR=self.VIS_O_PROJ_DRAM,
                OUTPUT_DRAM_ADDR=self.VIS_RESIDUAL_DRAM)
            self.rms_norm_core_dram(M=VS, N=VH,
                A_DRAM_ADDR=self.VIS_RESIDUAL_DRAM, OUTPUT_DRAM_ADDR=self.VIS_NORM_OUT_DRAM,
                GAMMA_DRAM_ADDR=la['norm2_weight'], gpr_M_reg=vis_M_reg)

            self.matmat_mul_core(is_B_quantized=True,
                M=VS, K=VH, N=VIS_MLP_PAD,
                A_DRAM_ADDR=self.VIS_NORM_OUT_DRAM,
                B_DRAM_ADDR=la['gate_data'],
                OUTPUT_DRAM_ADDR=self.VIS_MLP_GATE_DRAM,
                SCALE_DRAM_ADDR=la['gate_scale'], data_type=TYPE.IF4,
                C_DRAM_ADDR=la['gate_bias'], bias_mode="broadcast_N",
                silu_enable=True, gpr_M_reg=vis_M_reg)
            self.matmat_mul_core(is_B_quantized=True,
                M=VS, K=VH, N=VIS_MLP_PAD,
                A_DRAM_ADDR=self.VIS_NORM_OUT_DRAM,
                B_DRAM_ADDR=la['up_data'],
                OUTPUT_DRAM_ADDR=self.VIS_MLP_UP_DRAM,
                SCALE_DRAM_ADDR=la['up_scale'], data_type=TYPE.IF4,
                C_DRAM_ADDR=la['up_bias'], bias_mode="broadcast_N", gpr_M_reg=vis_M_reg)

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
                C_DRAM_ADDR=la['down_bias'], bias_mode="broadcast_N", gpr_M_reg=vis_M_reg)

            eltwise_add_core_dram(self, size=VS * VH,
                A_DRAM_ADDR=self.VIS_RESIDUAL_DRAM,
                B_DRAM_ADDR=self.VIS_MLP_DOWN_DRAM,
                OUTPUT_DRAM_ADDR=h_out)

        _original_print()  # newline after \r progress

        final_vis = self.VIS_IO_A_DRAM if total_layers % 2 == 0 else self.VIS_IO_B_DRAM

        self.rms_norm_core_dram(M=VS, N=VH,
            A_DRAM_ADDR=final_vis, OUTPUT_DRAM_ADDR=self.VIS_POST_NORM_DRAM,
            GAMMA_DRAM_ADDR=self.merger_ln_q_weight, gpr_M_reg=vis_M_reg)

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

        self.generate_instruction_halt()

        # Finalize single program
        self.stop_capture()
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

        if getattr(self, '_unified_active', False):
            # Unified single-bin mode: hand bytes to compile_all for programs.bin.
            self._seg_encoder = (program_addr, bytes(prog_bytes))
        _original_print(f"    Vision encoder compiled: {total_bytes} bytes at 0x{program_addr:X}, "
                        f"{total_layers} layers + merger")
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

            # Interleaved table [VS, 2, VD_PAD]: row t = [cos(128) || sin(128)] so the
            # PBI rope core loads cos+sin with a single auto-advancing pointer.
            rope_table = torch.empty(VS, 2, VD_PAD, dtype=torch.bfloat16)
            cos_table = rope_table[:, 0, :]
            sin_table = rope_table[:, 1, :]

            # cos: ones everywhere (identity for zero-padded dims), real cos in both halves
            cos_table.fill_(1.0)
            cos_table[:, :half_d] = cos_raw           # [0:40] = cos
            cos_table[:, 64:64 + half_d] = cos_raw    # [64:104] = cos; [40:64],[104:128] = 1

            # sin: zeros in pad gaps; -sin in lo half, +sin in hi half (rotate-half sign)
            sin_table.zero_()
            sin_table[:, :half_d] = -sin_raw           # [0:40] = -sin
            sin_table[:, 64:64 + half_d] = sin_raw     # [64:104] = +sin; gaps = 0

            # Tile VN times (one block per head): row h*VS+t = table[t], so a single
            # M=VN*VS rope call over the head-major Q_PAD/K_PAD applies cos/sin[t] to
            # every head's token t (cos/sin are position-only, shared across heads).
            self.dma_to_accelerator_memory(self.VIS_ROPE_DRAM, rope_table.repeat(VN, 1, 1).flatten())
        print(f"    Vision RoPE table [{VN}*{VS}, 2, {VD_PAD}] (tiled per head) DMA'd to FPGA")

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

    def _emit_prefill_program(self, seq_len: int, layer_size: int) -> int:
        """Emit ONE seq_len-agnostic prefill program (PBI single-bin pattern).

        Caller wraps in start_capture()/stop_capture(). Program ends with a halt
        so it can be jumped to via a runtime preamble that primes
        gf_seq_len / gf_q_seq_len / gf_aligned_seq_len.

        ``seq_len`` is a compile-time template for FLOPS bookkeeping and
        static ``M=`` args. K/V/Q scatters and FLASH_OUTPUT assembly use PBI
        runtime loops keyed off gf_seq_len. rms_norm/matmat/rope use
        ``gpr_M_reg=gf_seq_len`` (or TMP_REG holding gf_seq_len*mult for QK
        heads). Unified attention is bounded by gf_aligned_seq_len.

        Qwen2.5-VL specifics vs Dave's qwen3:
          - Q/K/V have biases (C_DRAM_ADDR=la['*_bias']).
          - NO QK RMSNorm — rope reads from K_DRAM/Q_DRAM directly.
          - o_proj has no quantization.
          - mRoPE table is pre-baked at DRAM_ADDR_ROPE in per-token rows
            (cos[ahd] || sin[ahd]) so sequential reads work.
        """
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        ahd  = self.actual_head_dim
        nkvh = self.num_kv_heads
        qpkv = self.group_size
        bpe  = self.bytes_per_element
        hd   = self.head_dim
        ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE

        self._isa_reg_counter = 5
        total_flops = 0

        for layer_idx in range(layer_size):
            _original_print(f"    prefill layer {layer_idx + 1}/{layer_size}", end="\r", flush=True)
            la = self.lm_layer_addrs[layer_idx]
            if layer_idx != 0:
                # Inter-layer copy via PBI per-token scatter to stay under URAM_A.
                self._emit_pbi_scatter_per_token(
                    read_base=self.LAYER0_OUTPUT_DRAM,
                    read_stride_bytes=self.vector_length * bpe,
                    write_specs=[(self.LAYER0_INPUT_DRAM, self.vector_length * bpe)],
                    sram_byte_addr=0x10000,
                    element_count=self.vector_length,
                    gf_seq_len=self.gf_seq_len,
                    template_seq_len=seq_len,
                )

            # Pre-norm (input_layernorm)
            total_flops += (self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=la['ln1_gamma'],
                              gpr_M_reg=self.gf_seq_len) or 0)

            # Q, K, V projections with bias
            total_flops += (self.matmat_mul_core(is_B_quantized=True, M=seq_len, K=self.vector_length, N=hd * qpkv,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=la['q_data'], OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                SCALE_DRAM_ADDR=la['q_scale'], data_type=TYPE.IF4,
                C_DRAM_ADDR=la['q_bias'], gpr_M_reg=self.gf_seq_len) or 0)
            total_flops += (self.matmat_mul_core(is_B_quantized=True, M=seq_len, K=self.vector_length, N=hd,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=la['k_data'], OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                SCALE_DRAM_ADDR=la['k_scale'], data_type=TYPE.IF4,
                C_DRAM_ADDR=la['k_bias'], gpr_M_reg=self.gf_seq_len) or 0)
            total_flops += (self.matmat_mul_core(M=seq_len, K=self.vector_length, N=hd,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=la['v_weight'], OUTPUT_DRAM_ADDR=self.LAYER0_V_PROJ_TEMP,
                C_DRAM_ADDR=la['v_bias'], gpr_M_reg=self.gf_seq_len) or 0)

            # RoPE in place on K_DRAM (nkvh groups per token) and Q_DRAM (nkvh*qpkv groups per token).
            # mRoPE table layout is per-token contiguous rows [cos(ahd) || sin(ahd)].
            total_flops += self.rope_hf_core_dram_gqa(
                M=seq_len, group_size=nkvh, N=ahd,
                input_dram_addr=self.LAYER0_K_DRAM,
                output_dram_addr=self.LAYER0_K_DRAM,
                cos_dram_addr=ROPE_WEIGHT_ADDR,
                sin_dram_addr=ROPE_WEIGHT_ADDR + ahd * bpe,
                gpr_M_reg=self.gf_seq_len)
            total_flops += self.rope_hf_core_dram_gqa(
                M=seq_len, group_size=nkvh * qpkv, N=ahd,
                input_dram_addr=self.LAYER0_Q_DRAM,
                output_dram_addr=self.LAYER0_Q_DRAM,
                cos_dram_addr=ROPE_WEIGHT_ADDR,
                sin_dram_addr=ROPE_WEIGHT_ADDR + ahd * bpe,
                gpr_M_reg=self.gf_seq_len)

            # Per-KV-head: PBI scatter K/V to cache + FLASH_K/V (duplicated qpkv times per token),
            # PBI scatter packed FLASH_Q (qpkv rows per token), bucketed flash, PBI assemble output.
            for kv_h in range(nkvh):
                k_cache_base = (self.LAYER0_K_ROPE_DRAM
                                + layer_idx * nkvh * self.MAX_CONTEXT_SIZE * ahd * bpe
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)
                v_cache_base = (self.LAYER0_V_DRAM
                                + layer_idx * nkvh * self.MAX_CONTEXT_SIZE * ahd * bpe
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)

                # K scatter: K_DRAM[t][kv_h] → K cache[kv_h][t] + qpkv copies into FLASH_K
                k_write_specs = [(k_cache_base, ahd * bpe)]
                for g in range(qpkv):
                    k_write_specs.append((self.LAYER0_FLASH_K_DRAM + g * ahd * bpe, qpkv * ahd * bpe))
                self._emit_pbi_scatter_per_token(
                    read_base=self.LAYER0_K_DRAM + kv_h * ahd * bpe,
                    read_stride_bytes=nkvh * ahd * bpe,
                    write_specs=k_write_specs,
                    sram_byte_addr=0x10000,
                    element_count=ahd,
                    gf_seq_len=self.gf_seq_len,
                    template_seq_len=seq_len,
                )

                # V scatter: V_PROJ_TEMP[t][kv_h] → V cache[kv_h][t] + qpkv copies into FLASH_V
                v_write_specs = [(v_cache_base, ahd * bpe)]
                for g in range(qpkv):
                    v_write_specs.append((self.LAYER0_FLASH_V_DRAM + g * ahd * bpe, qpkv * ahd * bpe))
                self._emit_pbi_scatter_per_token(
                    read_base=self.LAYER0_V_PROJ_TEMP + kv_h * ahd * bpe,
                    read_stride_bytes=nkvh * ahd * bpe,
                    write_specs=v_write_specs,
                    sram_byte_addr=0x20000,
                    element_count=ahd,
                    gf_seq_len=self.gf_seq_len,
                    template_seq_len=seq_len,
                )

                # Q scatter: Q_DRAM[t][kv_h*qpkv:kv_h*qpkv+qpkv] → FLASH_Q[t*qpkv:(t+1)*qpkv]
                self._emit_pbi_scatter_per_token(
                    read_base=self.LAYER0_Q_DRAM + kv_h * qpkv * ahd * bpe,
                    read_stride_bytes=nkvh * qpkv * ahd * bpe,
                    write_specs=[(self.LAYER0_FLASH_Q_DRAM, qpkv * ahd * bpe)],
                    sram_byte_addr=0x30000,
                    element_count=qpkv * ahd,
                    gf_seq_len=self.gf_seq_len,
                    template_seq_len=seq_len,
                )

                total_flops += (self.unified_attention_core(
                    batch=q_seq_len,
                    aligned_seq_len=aligned_seq_len,
                    head_dim=ahd,
                    Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
                    K_DRAM_ADDR=self.LAYER0_FLASH_K_DRAM,
                    V_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                    BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_DRAM,
                    OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUT_HEAD_DRAM,
                    SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                    IDENTITY_DRAM_ADDR=self.IDENTITY_DRAM_ADDR,
                    gpr_batch_reg=self.gf_q_seq_len,
                    gpr_aligned_seq_len_reg=self.gf_aligned_seq_len) or 0)

                # Assemble: per token, copy qpkv rows from FLASH_OUT_HEAD[t*qpkv:(t+1)*qpkv]
                # to FLASH_OUTPUT[t][kv_h*qpkv:(kv_h+1)*qpkv].
                self._emit_pbi_scatter_per_token(
                    read_base=self.LAYER0_FLASH_OUT_HEAD_DRAM,
                    read_stride_bytes=qpkv * ahd * bpe,
                    write_specs=[(self.LAYER0_FLASH_OUTPUT_DRAM + kv_h * qpkv * ahd * bpe,
                                  hd * qpkv * bpe)],
                    sram_byte_addr=0x40000,
                    element_count=qpkv * ahd,
                    gf_seq_len=self.gf_seq_len,
                    template_seq_len=seq_len,
                )

            # o_proj (no quantization params for qwen2.5_vl)
            total_flops += (self.matmat_mul_core(M=seq_len, K=hd * qpkv, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM, B_DRAM_ADDR=la['o_weight'], OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                gpr_M_reg=self.gf_seq_len) or 0)

            # Post-attn residual: layer_input + o_proj. PBI runtime loop.
            self.eltwise_core_dram(
                M=seq_len, N=self.vector_length,
                dram_a=self.LAYER0_INPUT_DRAM,
                dram_b=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                dram_out=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                mode=UE_MODE.ELTWISE_ADD,
                gpr_M_reg=self.gf_seq_len,
            )

            # post_attention_layernorm = pre-FFN norm
            total_flops += (self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=la['ln2_gamma'],
                              gpr_M_reg=self.gf_seq_len) or 0)

            # MLP gate (with SiLU), up, gate × up, down.
            total_flops += (self.matmat_mul_core(is_B_quantized=True, M=seq_len, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=la['gate_data'], OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                SCALE_DRAM_ADDR=la['gate_scale'], data_type=TYPE.IF4, silu_enable=True,
                gpr_M_reg=self.gf_seq_len) or 0)
            total_flops += (self.matmat_mul_core(is_B_quantized=True, M=seq_len, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=la['up_data'], OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                SCALE_DRAM_ADDR=la['up_scale'], data_type=TYPE.IF4,
                gpr_M_reg=self.gf_seq_len) or 0)

            self.eltwise_core_dram(
                M=seq_len, N=self.mlp_elements,
                dram_a=self.LAYER0_MLP_GATE_DRAM,
                dram_b=self.LAYER0_MLP_UP_DRAM,
                dram_out=self.LAYER0_MLP_MULT_DRAM,
                mode=UE_MODE.ELTWISE_MUL,
                gpr_M_reg=self.gf_seq_len,
            )

            total_flops += (self.matmat_mul_core(is_B_quantized=True, M=seq_len, K=self.mlp_elements, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM, B_DRAM_ADDR=la['down_data'], OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                SCALE_DRAM_ADDR=la['down_scale'], data_type=TYPE.IF4,
                gpr_M_reg=self.gf_seq_len) or 0)

            # Post-MLP residual: layer_output = POST_ATTN_RESIDUAL + MLP_DOWN.
            self.eltwise_core_dram(
                M=seq_len, N=self.vector_length,
                dram_a=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                dram_b=self.LAYER0_MLP_DOWN_DRAM,
                dram_out=self.LAYER0_OUTPUT_DRAM,
                mode=UE_MODE.ELTWISE_ADD,
                gpr_M_reg=self.gf_seq_len,
            )

        self.generate_instruction_halt()

        return total_flops

    def compile_prefill(self, layer_size: int | None = None) -> tuple[int, int, int]:
        """Emit the single PBI prefill program into program DRAM and return
        (program_addr, preamble_addr, total_flops). No standalone disk cache — the
        bytes are handed to compile_all (via self._seg_prefill) for the unified
        programs.bin. The program is seq_len-agnostic for actual_seq_len ≤
        PREFILL_MAX_SEQ_LEN, primed at runtime by the preamble in run_prefill.
        """
        if layer_size is None:
            layer_size = self.LAYER_SIZE
        template_seq_len = self.PREFILL_MAX_SEQ_LEN

        global _SILENT_MODE
        _SILENT_MODE = True
        _original_print(f"  Compiling PBI prefill (template={template_seq_len}, {layer_size} layers)...")
        self.clear_inst_id()
        self.start_capture()
        total_flops = self._emit_prefill_program(seq_len=template_seq_len, layer_size=layer_size)
        self.stop_capture()

        prog_bytes = bytearray()
        for inst in self.capture_buffer:
            prog_bytes.extend(inst.get_bytes())
        program_addr = self.get_program_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, program_addr, prog_bytes, len(prog_bytes))
        self.allocate_program_dram(len(prog_bytes))
        self.clear_capture_buffer()
        preamble_addr = self.allocate_program_dram(256)
        _SILENT_MODE = False
        _original_print()

        if getattr(self, '_unified_active', False):
            self._seg_prefill = (program_addr, bytes(prog_bytes))
        print(f"    Prefill compiled: {len(prog_bytes)} bytes at 0x{program_addr:X}, preamble 0x{preamble_addr:X}")
        return program_addr, preamble_addr, total_flops

    def run_prefill(self, prefill_program_addr: int, preamble_addr: int,
                    prefill_seq, gflops: int = None, has_image: bool = False) -> dict:
        """Run prefill via runtime preamble that primes gf_seq_len / gf_q_seq_len /
        gf_aligned_seq_len and JUMP_ABS into the cached prefill program. Handles any
        actual_seq_len ≤ PREFILL_MAX_SEQ_LEN."""
        if prefill_seq is None:
            prefill_seq = tuple(self._cfg["default_prefill_tokens"])
        if len(prefill_seq) <= 1:
            raise ValueError("Prefill sequence must have at least 2 tokens.")

        prefill_seq = prefill_seq[:-1]
        actual_seq_len = len(prefill_seq)
        if actual_seq_len > self.PREFILL_MAX_SEQ_LEN:
            raise ValueError(
                f"Prompt too long: actual_seq_len={actual_seq_len} > PREFILL_MAX_SEQ_LEN={self.PREFILL_MAX_SEQ_LEN}. "
                f"Rebuild with a larger prefill_max_seq_len in config."
            )
        self.seq_len = actual_seq_len

        q_seq_len = actual_seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64

        # Gather text embeddings and merge vision tokens if present.
        embedding_tensor = self.get_embedding_for_tokens(prefill_seq)
        if has_image:
            vision_token_id = 151655
            img_positions = [i for i, t in enumerate(prefill_seq) if t == vision_token_id]
            num_vis = getattr(self, '_vis_num_tokens', 0)
            if len(img_positions) > 0 and num_vis > 0:
                vis_embeddings = self.dma_from_accelerator_memory(
                    self.VIS_ENCODER_OUT_DRAM, (num_vis, self.vector_length)).cpu()
                embed_reshaped = embedding_tensor.reshape(actual_seq_len, self.vector_length)
                n_replace = min(len(img_positions), num_vis)
                for i in range(n_replace):
                    embed_reshaped[img_positions[i]] = vis_embeddings[i]
                embedding_tensor = embed_reshaped.flatten()
                print(f"    Merged {n_replace} vision tokens into prefill at positions {img_positions[:5]}{'...' if len(img_positions) > 5 else ''}")

        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)

        # Causal bias for packed FLASH_Q (q_seq_len rows of Q vs q_seq_len rows of K duplicated qpkv times per token).
        # Lower-triangular on aligned_seq_len × aligned_seq_len; mask K cols beyond actual q_seq_len.
        bias_one_group = torch.full((aligned_seq_len, aligned_seq_len), float("-inf"), dtype=torch.bfloat16)
        valid_mask = torch.tril(torch.ones(aligned_seq_len, aligned_seq_len, dtype=torch.bool), diagonal=0) if not self.causal_mask_upper else torch.triu(torch.ones(aligned_seq_len, aligned_seq_len, dtype=torch.bool), diagonal=0)
        bias_one_group.masked_fill_(valid_mask, 0.0)
        bias_one_group[:, q_seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_one_group)

        # Emit runtime preamble: ADD_SETs for the 3 GPRs + JUMP_ABS into prefill.
        self.clear_inst_id()
        self.start_capture()
        self.generate_instruction_add_set(self.gf_seq_len,         actual_seq_len)
        self.generate_instruction_add_set(self.gf_q_seq_len,       q_seq_len)
        self.generate_instruction_add_set(self.gf_aligned_seq_len, aligned_seq_len)
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(prefill_program_addr))
        self.stop_capture()
        self.write_captured_instructions_to_dram(preamble_addr)
        self.clear_capture_buffer()

        prefill_flops = gflops if isinstance(gflops, (int, float)) else 0
        latency_us, flop_rate = self.program_execute(preamble_addr, timeout=180.0,
                                                     flops=prefill_flops, verbose=False)
        self._latency_hw_prefill = latency_us
        self._flop_rate_hw_prefill = flop_rate
        self._last_hw_gflops = flop_rate
        print(f"    Prefill executed: actual_seq_len={actual_seq_len}, q_seq_len={q_seq_len}, aligned_seq_len={aligned_seq_len}")
        print(f"    HW latency: {latency_us/1e6:.2f}s, {flop_rate:.2f} GFLOPS")

    def _emit_decoder_program(self, layer_size: int) -> int:
        """Emit ONE decode-position-agnostic decoder program (PBI single-bin pattern).

        Caller wraps this in start_capture()/stop_capture(). The emitted program
        ends with ADD_INC gf_seq_len + halt so it can be jumped to via a runtime
        preamble that primes gf_q_seq_len, gf_aligned_seq_len, and gf_rope_cos_abs
        each step (and gf_seq_len as the cache write position).

        Address math: K/V cache write addresses are computed at runtime as
        ``base + gf_seq_len * (ahd*bpe)`` via reg_mul_imm + add_imm into TMP_REG,
        then passed as ``general_reg_src=TMP_REG`` to the scatter store.

        Decoder attention uses inline ``unified_attention_core``; one call per KV
        head processes all qpkv Q heads and is bounded by gf_aligned_seq_len.

        All bulk ops use ``gpr_M_reg=gf_one`` (M=1) for consistency.
        """
        ahd  = self.actual_head_dim
        nkvh = self.num_kv_heads
        qpkv = self.group_size
        bpe  = self.bytes_per_element
        hd   = self.head_dim
        rope_row_bytes = ahd * 2 * bpe   # 512 B per token

        ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE

        # Reset dynamic GPR allocator.
        self._isa_reg_counter = 5
        gf_one             = self.alloc_isa_reg()
        gf_rope_cos_abs    = self.alloc_isa_reg()
        # Stash for run_decoder preamble to use the same indices.
        self._gf_one          = gf_one
        self._gf_rope_cos_abs = gf_rope_cos_abs

        # gf_one is primed once inside the program (always = 1).
        self.generate_instruction_add_set(gf_one, 1)
        # gf_rope_cos_abs is primed per-step by run_decoder's preamble (depends on _rope_offset).

        total_flops = 0
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        for layer_idx in range(layer_size):
            layer_off = layer_idx * LAYER_WEIGHT_SIZE
            la = self.lm_layer_addrs[layer_idx]
            if layer_idx != 0:
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, sram_address=0x10000, element_size=self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_INPUT_DRAM, element_size=self.vector_length)

            # Pre-norm — M=1 via gf_one
            total_flops += (self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                          OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=la['ln1_gamma'],
                          gpr_M_reg=gf_one) or 0)

            # Q, K, V projections with bias.
            # §3h: q_proj, k_proj (IF4, K=2048<8192) use the M=1 GEMV kernel.
            # v_proj is BF16 — quantized_matmat_core asserts IF4/IF8/TQ4 only
            # (see [[bf16_models_no_quantized_gemv]]), so it stays on matmat_mul_core.
            total_flops += (self.quantized_matmat_core(M=1, K=self.vector_length, N=hd * qpkv,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=la['q_data'], OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                SCALE_DRAM_ADDR=la['q_scale'], data_type=TYPE.IF4,
                C_DRAM_ADDR=la['q_bias']) or 0)
            total_flops += (self.quantized_matmat_core(M=1, K=self.vector_length, N=hd,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=la['k_data'], OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                SCALE_DRAM_ADDR=la['k_scale'], data_type=TYPE.IF4,
                C_DRAM_ADDR=la['k_bias']) or 0)
            total_flops += (self.matmat_mul_core(M=1, K=self.vector_length, N=hd,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=la['v_weight'], OUTPUT_DRAM_ADDR=self.LAYER0_V_PROJ_TEMP,
                C_DRAM_ADDR=la['v_bias'], gpr_M_reg=gf_one) or 0)

            # Qwen2.5-VL: NO QK norm — Q/K used directly

            # RoPE per head — use gr_weight_dram=gf_rope_cos_abs (absolute cos base addr
            # primed by per-step preamble = ROPE_LOCAL + (seq_len-1 + _rope_offset) * rope_row_bytes).
            for kv_h in range(nkvh):
                total_flops += (self.rope_hf_core_decode(
                    N=ahd,
                    input_dram_addr=self.LAYER0_K_DRAM + kv_h * ahd * bpe,
                    output_dram_addr=self.LAYER0_K_DRAM + kv_h * ahd * bpe,
                    cos_dram_addr=ROPE_WEIGHT_ADDR,
                    sin_dram_addr=ROPE_WEIGHT_ADDR + ahd * bpe,
                    gr_weight_dram=gf_rope_cos_abs) or 0)
            for q_h in range(nkvh * qpkv):
                total_flops += (self.rope_hf_core_decode(
                    N=ahd,
                    input_dram_addr=self.LAYER0_Q_DRAM + q_h * ahd * bpe,
                    output_dram_addr=self.LAYER0_Q_DRAM + q_h * ahd * bpe,
                    cos_dram_addr=ROPE_WEIGHT_ADDR,
                    sin_dram_addr=ROPE_WEIGHT_ADDR + ahd * bpe,
                    gr_weight_dram=gf_rope_cos_abs) or 0)

            # Per-KV-head: store new K/V to cache at gf_seq_len position, then unified attention.
            for kv_h in range(nkvh):
                k_cache_base = (self.LAYER0_K_ROPE_DRAM
                                + layer_idx * nkvh * self.MAX_CONTEXT_SIZE * ahd * bpe
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)
                v_cache_base = (self.LAYER0_V_DRAM
                                + layer_idx * nkvh * self.MAX_CONTEXT_SIZE * ahd * bpe
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)

                # K cache write: TMP_REG = k_cache_base + gf_seq_len * (ahd*bpe)
                self.accelerator_memory_to_sram(self.LAYER0_K_DRAM + kv_h * ahd * bpe, 0x10000, ahd)
                self.generate_instruction_reg_mul_imm(
                    self.TMP_REG, self.gf_seq_len, ue_35bit_addr_shifter(ahd * bpe))
                self.generate_instruction_add_imm(
                    self.TMP_REG, ue_35bit_addr_shifter(k_cache_base), self.TMP_REG)
                self.sram_to_accelerator_memory(
                    sram_address=0x10000, accelerator_dram_address=0,
                    element_size=ahd, general_reg_src=self.TMP_REG)

                # V cache write
                self.accelerator_memory_to_sram(self.LAYER0_V_PROJ_TEMP + kv_h * ahd * bpe, 0x20000, ahd)
                self.generate_instruction_reg_mul_imm(
                    self.TMP_REG, self.gf_seq_len, ue_35bit_addr_shifter(ahd * bpe))
                self.generate_instruction_add_imm(
                    self.TMP_REG, ue_35bit_addr_shifter(v_cache_base), self.TMP_REG)
                self.sram_to_accelerator_memory(
                    sram_address=0x20000, accelerator_dram_address=0,
                    element_size=ahd, general_reg_src=self.TMP_REG)

                # Gather qpkv Q heads for this KV head into FLASH_Q (contiguous: 0, ahd*bpe, ...)
                for q in range(qpkv):
                    q_src = self.LAYER0_Q_DRAM + (kv_h * qpkv + q) * ahd * bpe
                    flash_q_addr = self.LAYER0_FLASH_Q_DRAM + q * ahd * bpe
                    self.accelerator_memory_to_sram(q_src, 0x30000, ahd)
                    self.sram_to_accelerator_memory(0x30000, flash_q_addr, ahd)

                # Marshal active K/V history into FLASH_K / FLASH_V. gf_q_seq_len is
                # the active KV token count; gf_aligned_seq_len bounds attention.
                self._emit_pbi_scatter_per_token(
                    read_base=k_cache_base,
                    read_stride_bytes=ahd * bpe,
                    write_specs=[(self.LAYER0_FLASH_K_DRAM, ahd * bpe)],
                    sram_byte_addr=0x50000,
                    element_count=ahd,
                    gf_seq_len=self.gf_q_seq_len,
                    template_seq_len=self.MAX_CONTEXT_SIZE,
                )
                self._emit_pbi_scatter_per_token(
                    read_base=v_cache_base,
                    read_stride_bytes=ahd * bpe,
                    write_specs=[(self.LAYER0_FLASH_V_DRAM, ahd * bpe)],
                    sram_byte_addr=0x60000,
                    element_count=ahd,
                    gf_seq_len=self.gf_q_seq_len,
                    template_seq_len=self.MAX_CONTEXT_SIZE,
                )

                total_flops += (self.unified_attention_core(
                    batch=qpkv,
                    aligned_seq_len=self.MAX_CONTEXT_SIZE,
                    head_dim=ahd,
                    Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
                    K_DRAM_ADDR=self.LAYER0_FLASH_K_DRAM,
                    V_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                    BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_DRAM,
                    OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUT_HEAD_DRAM,
                    SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                    IDENTITY_DRAM_ADDR=self.IDENTITY_DRAM_ADDR,
                    gpr_aligned_seq_len_reg=self.gf_aligned_seq_len) or 0)

                # Post-return: copy FLASH_OUT_HEAD (qpkv × ahd) → assembled FLASH_OUTPUT
                # at this KV head's offset. Decode M=1 → 2 KB per copy.
                self.accelerator_memory_to_sram(
                    self.LAYER0_FLASH_OUT_HEAD_DRAM, 0x40000, qpkv * ahd)
                self.sram_to_accelerator_memory(
                    0x40000, self.LAYER0_FLASH_OUTPUT_DRAM + kv_h * qpkv * ahd * bpe,
                    qpkv * ahd)

            # o_proj (BF16)
            total_flops += (self.matmat_mul_core(M=1, K=hd * qpkv, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM, B_DRAM_ADDR=la['o_weight'], OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                gpr_M_reg=gf_one) or 0)

            # Qwen2.5-VL: no post-attention norm; residual direct on o_proj output
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_INPUT_DRAM, sram_address=0x10000, element_size=self.vector_length)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM, sram_address=0x90000, element_size=self.vector_length)
            self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, element_size=self.vector_length)

            # Qwen2.5-VL: post_attention_layernorm IS the pre-FFN norm
            total_flops += (self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                          OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=la['ln2_gamma'],
                          gpr_M_reg=gf_one) or 0)

            # MLP: SwiGLU
            # §3h: gate_proj, up_proj (IF4, K=2048<8192) use the M=1 GEMV kernel.
            total_flops += (self.quantized_matmat_core(M=1, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=la['gate_data'], OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                SCALE_DRAM_ADDR=la['gate_scale'], data_type=TYPE.IF4, silu_enable=True) or 0)
            total_flops += (self.quantized_matmat_core(M=1, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=la['up_data'], OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                SCALE_DRAM_ADDR=la['up_scale'], data_type=TYPE.IF4) or 0)

            # gate x up (M=1: mlp_elements fits in SRAM in one shot)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_GATE_DRAM, sram_address=0x10000, element_size=self.mlp_elements)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_UP_DRAM, sram_address=0x90000, element_size=self.mlp_elements)
            self.eltwise_mul_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.mlp_elements)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_MLP_MULT_DRAM, element_size=self.mlp_elements)

            # down_proj — K=mlp_elements=11008 > SCALE_BRAM_ELEMENTS=8192 triggers
            # quantized_matmat_core N_chunk=0 compile-hang (§3h), so stay on matmat_mul_core.
            total_flops += (self.matmat_mul_core(is_B_quantized=True, M=1, K=self.mlp_elements, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM, B_DRAM_ADDR=la['down_data'], OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                SCALE_DRAM_ADDR=la['down_scale'], data_type=TYPE.IF4,
                gpr_M_reg=gf_one) or 0)

            # Qwen2.5-VL: no post-FFN norm; residual direct on down_proj output
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, sram_address=0x10000, element_size=self.vector_length)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_DOWN_DRAM, sram_address=0x90000, element_size=self.vector_length)
            self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, element_size=self.vector_length)

        # Final norm + LM head (only when running full model)
        if layer_size == self.LAYER_SIZE:
            total_flops += (self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_OUTPUT_DRAM,
                OUTPUT_DRAM_ADDR=self.OUTPUT_NORM_DRAM, GAMMA_DRAM_ADDR=self.final_norm_addr,
                gpr_M_reg=gf_one) or 0)
            # §3h LM head (M=1 GEMV). On-FPGA repetition penalty (llama3.2_1b style):
            # the per-vocab PENALTY_BIAS_DRAM is the matmul's C term (broadcast_N), so
            # the HW argmax of (logits + bias) returns the penalized token directly —
            # write_back_disable=True means LOGITS_DRAM is never written and there is
            # NO host logit readback. All-zero bias buffer = pure greedy.
            total_flops += (self.quantized_matmat_core(M=1, K=self.vector_length, N=self.EMBEDDING_ELEMENTS,
                A_DRAM_ADDR=self.OUTPUT_NORM_DRAM, B_DRAM_ADDR=self.lm_head_data, OUTPUT_DRAM_ADDR=self.LOGITS_DRAM,
                SCALE_DRAM_ADDR=self.lm_head_scale, data_type=TYPE.IF4,
                C_DRAM_ADDR=self.PENALTY_BIAS_DRAM, bias_mode="broadcast_N",
                write_back_disable=True) or 0)

        # End-of-body: bump gf_seq_len for next decode step, then halt.
        self.generate_instruction_add_inc(self.gf_seq_len)
        self.generate_instruction_halt()

        return total_flops

    def compile_decoder(self, layer_size: int | None = None) -> tuple[int, int]:
        """Emit the single PBI decoder program into program DRAM and allocate the
        per-step preamble slot. No standalone disk cache — bytes are handed to
        compile_all (via self._seg_decoder) for the unified programs.bin.
        ``self._decoder_preamble_addr`` is the small slot run_decoder rewrites each
        step. Returns (decoder_program_addr, total_flops).
        """
        if layer_size is None:
            layer_size = self.LAYER_SIZE

        global _SILENT_MODE
        _SILENT_MODE = True
        _original_print(f"  Compiling PBI decoder ({layer_size} layers)...")
        self.clear_inst_id()
        self.start_capture()
        total_flops = self._emit_decoder_program(layer_size)
        self.stop_capture()
        _SILENT_MODE = False

        decoder_program_addr = self.get_program_dram_addr()
        prog_bytes = bytearray()
        for inst in self.capture_buffer:
            prog_bytes.extend(inst.get_bytes())
        self.write_captured_instructions_to_dram(decoder_program_addr)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        self.clear_capture_buffer()

        if getattr(self, '_unified_active', False):
            self._seg_decoder = (decoder_program_addr, bytes(prog_bytes))
        print(f"  Decoder compiled: {len(prog_bytes)} bytes at 0x{decoder_program_addr:X}, "
              f"total_flops={total_flops}")

        # Allocate a small preamble slot (5 instructions × 32 B = 160 B; round up to 256 B).
        self._decoder_preamble_addr = self.get_program_dram_addr()
        self.allocate_program_dram(256)
        return decoder_program_addr, total_flops

    def _ensure_canonical_vision_layout(self) -> None:
        """Ensure self._cu_window_seqlens is set so the encoder ISA can be
        captured even on a text-only first compile. The vision window layout is
        a deterministic function of the canonical image size (336×336 → fixed
        576 patches), NOT of pixel content — so a synthetic black image yields
        the exact window layout any real image will use at runtime."""
        if getattr(self, '_cu_window_seqlens', None) is not None:
            return
        img_size = self._vis_cfg.get("image_size", 336)
        synthetic = torch.zeros(3, img_size, img_size, dtype=torch.bfloat16)
        _original_print(f"  [unified bin] deriving canonical vision window layout "
                        f"({img_size}×{img_size}) for the encoder section...")
        self.prepare_encoder_input(synthetic)

    def compile_all(self, layer_size: int | None = None) -> dict:
        """Build the COMPLETE single instruction bin (prefill + decoder + encoder),
        once, serving BOTH LM-only and VLM requests.

        Ordering is the fix (mirrors gemma4_e2b compile_instruction_bin):
        prefill + decoder are captured FIRST at the fixed instruction base, the
        encoder is appended AFTER. So the §7 shared-subroutine JUMP_ABS targets
        in prefill/decoder bake against a base that never depends on the encoder's
        presence (see fpga_pbi_jump_target_bake). LM-only loads the bin and runs
        prefill/decoder (skips encoder execution); VLM additionally runs the
        encoder. One mode-agnostic bin → instructions_s{tmpl}.bin.

        The encoder section is always compiled (canonical 336×336 window layout),
        so the first run builds the complete bin regardless of LM/VLM. Subsequent
        runs load it. Cache key: PREFILL_MAX_SEQ_LEN + layer_size.

        Returns dict: encoder_addr, prefill_addr, prefill_preamble, decoder_addr,
        decoder_preamble, prefill_flops, decoder_flops.
        """
        if layer_size is None:
            layer_size = self.LAYER_SIZE
        bin_dir = os.path.join(self.script_dir, "qwen2.5_vl_3b_bin")
        os.makedirs(bin_dir, exist_ok=True)
        tmpl = self.PREFILL_MAX_SEQ_LEN
        uni_bin  = os.path.join(bin_dir, "programs.bin")
        uni_meta = os.path.join(bin_dir, "programs.json")

        # ---- cache hit: load the one bin, DMA each segment at its stored addr ----
        # Filename is fixed (programs.bin), so validate the baked layout
        # signature — a bin built for a different prefill_max_seq_len / layer_size
        # has different bucket bodies and addresses (core_changes §6) and must be
        # rebuilt, not silently reused.
        _meta_ok = False
        if os.path.exists(uni_bin) and os.path.exists(uni_meta):
            with open(uni_meta) as f:
                meta = json.load(f)
            _meta_ok = (meta.get("template_seq_len") == tmpl
                        and meta.get("layer_size") == layer_size)
            if not _meta_ok:
                _original_print(f"  programs.bin was built for template_seq_len="
                                f"{meta.get('template_seq_len')}/layer_size={meta.get('layer_size')} "
                                f"≠ current {tmpl}/{layer_size} — rebuilding.")
        if _meta_ok:
            with open(uni_bin, "rb") as f:
                raw = f.read()
            for seg in meta["segments"]:
                b = raw[seg["off"]:seg["off"] + seg["size"]]
                self.dma_write(DMA_DEVICE_H2C, seg["addr"], b, len(b))
            self._next_program_dram_addr = meta["end_addr"]
            self._gf_one = meta["gf_one"]
            self._gf_rope_cos_abs = meta["gf_rope_cos_abs"]
            self._decoder_preamble_addr = meta["decoder_preamble"]
            self._vis_program_addr = meta["encoder_addr"]
            _original_print(f"  Loaded unified instruction bin "
                            f"({len(raw)/1024/1024:.1f} MB, {len(meta['segments'])} segments) from {uni_bin}")
            return {k: meta[k] for k in ("encoder_addr", "prefill_addr", "prefill_preamble",
                                          "decoder_addr", "decoder_preamble",
                                          "prefill_flops", "decoder_flops")}

        # ---- cache miss: compile the complete bin (LM first, encoder appended) ----
        _original_print(f"  Compiling complete instruction bin (prefill + decoder + encoder)...")
        self._ensure_canonical_vision_layout()
        self._unified_active = True
        self._seg_encoder = self._seg_prefill = self._seg_decoder = None
        try:
            # LM FIRST → prefill/decoder land at the fixed base; their §7 jumps
            # bake here and stay valid whether or not the encoder runs.
            prefill_addr, prefill_preamble, prefill_flops = self.compile_prefill(layer_size)
            decoder_addr, decoder_flops = self.compile_decoder(layer_size)
            decoder_preamble = self._decoder_preamble_addr
            # ENCODER LAST → appended after the LM section.
            enc_addr = self.compile_encoder()
        finally:
            self._unified_active = False
        end_addr = self.get_program_dram_addr()

        # Assemble one bin: concat segment bytes; meta records each at its abs addr.
        segments, raw = [], bytearray()
        for name, seg in (("prefill", self._seg_prefill),
                          ("decoder", self._seg_decoder),
                          ("encoder", self._seg_encoder)):
            if seg is None:
                continue
            addr, b = seg
            segments.append({"name": name, "addr": addr, "off": len(raw), "size": len(b)})
            raw.extend(b)
        meta = {
            "template_seq_len": tmpl, "layer_size": layer_size,
            "segments": segments, "end_addr": end_addr,
            "encoder_addr": enc_addr,
            "prefill_addr": prefill_addr, "prefill_preamble": prefill_preamble,
            "prefill_flops": prefill_flops,
            "decoder_addr": decoder_addr, "decoder_preamble": decoder_preamble,
            "decoder_flops": decoder_flops,
            "gf_one": self._gf_one, "gf_rope_cos_abs": self._gf_rope_cos_abs,
        }
        with open(uni_bin, "wb") as f:
            f.write(raw)
        with open(uni_meta, "w") as f:
            json.dump(meta, f)
        print(f"  Complete instruction bin: {len(raw)/1024/1024:.1f} MB, "
              f"{len(segments)} segments (prefill+decoder+encoder) → {uni_bin}")
        return {k: meta[k] for k in ("encoder_addr", "prefill_addr", "prefill_preamble",
                                      "decoder_addr", "decoder_preamble",
                                      "prefill_flops", "decoder_flops")}

    def _structural_token_ids(self) -> set:
        """Token ids that must NEVER be repetition-penalized: punctuation, whitespace,
        newline, special tokens. Penalizing these 'glue' tokens over a long generation
        starves a small model of grammar → word-salad. Computed once, cached."""
        cached = getattr(self, "_struct_ids_cache", None)
        if cached is not None:
            return cached
        import string
        allowed = set(string.punctuation) | set(string.whitespace) | set("—–’‘“”…·•‹›«»¡¿")
        ids = set(int(i) for i in (getattr(self.tokenizer, "all_special_ids", []) or []))
        for i in range(self.EMBEDDING_ELEMENTS):
            s = self.tokenizer.decode([i]).strip()
            if s == "" or all(ch in allowed for ch in s):
                ids.add(i)
        self._struct_ids_cache = ids
        return ids

    def _structural_ids_tensor(self) -> torch.Tensor:
        """1-D LongTensor of structural/special token ids (cached) for vectorized exemption."""
        t = getattr(self, "_struct_ids_tensor_cache", None)
        if t is None:
            t = torch.tensor(sorted(self._structural_token_ids()), dtype=torch.long)
            self._struct_ids_tensor_cache = t
        return t

    def _write_penalty_bias(self, prev_tokens) -> None:
        """On-FPGA repetition penalty: build the per-vocab additive bias from the windowed
        token frequency and DMA it to PENALTY_BIAS_DRAM (the LM-head matmul's C term).
        bias[t] = clamp(-alpha*count[t], min=-cap); structural tokens stay 0. The HW argmax
        of (logits + bias) then returns the penalized token — NO logit readback. One full-
        buffer DMA per step (push only). Mirrors llama3.2_1b._write_penalty_bias."""
        vocab = self.EMBEDDING_ELEMENTS
        alpha = float(getattr(self, "pen_alpha", 1.0))
        cap   = float(getattr(self, "pen_cap", 20.0))
        W     = int(getattr(self, "rep_window", 256))
        window = prev_tokens[-W:]
        count = torch.zeros(vocab, dtype=torch.float32)
        if window:
            win = torch.tensor(window, dtype=torch.long)
            count.index_add_(0, win, torch.ones(win.numel(), dtype=torch.float32))
            count[self._structural_ids_tensor()] = 0.0
        bias = (-alpha * count).clamp(min=-cap).to(torch.bfloat16).view(1, vocab)
        self.dma_to_accelerator_memory(self.PENALTY_BIAS_DRAM, bias)

    def run_decoder(self, decoder_program_addr: int, token_id: int,
                    gflops: int | None = None, latency_prefill_wall: float = 0.0) -> tuple[int, str]:
        """Run PBI decode loop. Per-step preamble primes dynamic GPRs, then JUMP_ABS
        into the single cached decoder program. Breaks on EOS/EOT.
        Accumulates HW counters (total_latency_us, total_flop_rate) and prints the
        Decoder/HW counter status block at the end (matches gemma3/llama3.2 format).
        """
        if token_id is None:
            print("No last token available for decode.")
            return self.seq_len, ""

        _qwen25_stop_tokens = {151643, 151645, self._end_of_turn_token_id}
        decoded_chars: list[str] = []
        ahd = self.actual_head_dim
        bpe = self.bytes_per_element
        rope_row_bytes = ahd * 2 * bpe
        ROPE_BASE = self.DRAM_ADDR_ROPE
        preamble_addr = self._decoder_preamble_addr

        per_token_flops = gflops if isinstance(gflops, (int, float)) else 0
        total_latency_us = 0.0
        total_flop_rate = 0.0
        decode_start = time.perf_counter()
        prefill_seq_start = self.seq_len  # seq_len after prefill, before any decode step

        # On-FPGA repetition penalty (llama3.2_1b style — NO logit readback). The
        # per-vocab bias is the LM-head matmul's C term; the HW argmax of
        # (logits + bias) returns the penalized token. Position-gated: pure greedy
        # (zero bias) for the first `greedy_until` decoded tokens, then the bias is
        # refreshed each step from the windowed token frequency. _generated_tokens
        # is seeded with the prompt by main().
        if not hasattr(self, '_generated_tokens'):
            self._generated_tokens = []
        fpga_penalty = bool(getattr(self, 'fpga_penalty', True))
        greedy_until = int(getattr(self, 'greedy_until', 512))
        self.dma_to_accelerator_memory(self.PENALTY_BIAS_DRAM,
            torch.zeros(1, self.EMBEDDING_ELEMENTS, dtype=torch.bfloat16))

        # Live status bar (matches llama3.2_1b): pin the bottom terminal row as a
        # status line via an ANSI scroll region; tokens stream above it and the
        # tok/s counter refreshes in place. Only on a real TTY (skip when piped).
        import shutil
        _use_status = sys.stdout.isatty()
        def _status_setup():
            rows = shutil.get_terminal_size().lines
            sys.stdout.write(f"\033[1;{rows - 1}r")    # scroll region = rows 1..rows-1
            sys.stdout.write(f"\033[{rows - 1};1H")    # park cursor at bottom of region
            sys.stdout.flush()
        def _status_update():
            rows = shutil.get_terminal_size().lines
            n = self.seq_len - prefill_seq_start
            elapsed = time.perf_counter() - decode_start
            rate = n / elapsed if elapsed > 0 else 0.0
            sys.stdout.write("\0337")                  # save cursor
            sys.stdout.write(f"\033[{rows};1H\033[2K") # bottom row, clear it
            sys.stdout.write(f" decoding… {n} tokens  (pos {self.seq_len}/{self.MAX_CONTEXT_SIZE})  "
                             f"{elapsed:.1f}s  {rate:.1f} tok/s")
            sys.stdout.write("\0338")                  # restore cursor
            sys.stdout.flush()
        def _status_teardown():
            rows = shutil.get_terminal_size().lines
            sys.stdout.write("\033[r")                 # reset scroll region
            sys.stdout.write(f"\033[{rows};1H\033[2K") # clear the status row
            sys.stdout.flush()
        if _use_status:
            _status_setup()

        global _SILENT_MODE
        max_seq_len = self.MAX_CONTEXT_SIZE
        while self.seq_len < max_seq_len:
            _SILENT_MODE = True
            self.seq_len += 1
            step_pos = self.seq_len - 1  # KV-fill index for THIS step
            aligned_seq_len = ((self.seq_len + 63) // 64) * 64
            rope_pos = step_pos + getattr(self, '_rope_offset', 0)
            cos_abs_addr = ROPE_BASE + rope_pos * rope_row_bytes

            embedding_tensor = self.get_embedding_for_tokens([token_id])
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)

            bias_host = torch.full((1, aligned_seq_len), -1e36, dtype=torch.bfloat16)
            bias_host[0, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(
                self.LAYER0_FLASH_BIAS_DRAM,
                bias_host.repeat(self.group_size, 1))

            # Refresh the LM-head penalty bias past the greedy gate (push only — the
            # HW argmax of logits+bias yields the penalized token, no readback).
            if fpga_penalty and (self.seq_len - prefill_seq_start) > greedy_until:
                self._write_penalty_bias(self._generated_tokens)

            # Per-step preamble: prime dynamic GPRs + JUMP_ABS into cached decoder program.
            self.clear_inst_id()
            self.start_capture()
            self.generate_instruction_add_set(self.gf_seq_len,         step_pos)
            self.generate_instruction_add_set(self.gf_q_seq_len,       self.seq_len)
            self.generate_instruction_add_set(self.gf_aligned_seq_len, aligned_seq_len)
            self.generate_instruction_add_set(self._gf_rope_cos_abs,   ue_35bit_addr_shifter(cos_abs_addr))
            self.generate_instruction_jump_abs(ue_35bit_addr_shifter(decoder_program_addr))
            self.stop_capture()
            self.write_captured_instructions_to_dram(preamble_addr)
            self.clear_capture_buffer()

            step_latency_us, step_flop_rate = self.program_execute(
                preamble_addr, timeout=10.0, flops=per_token_flops, verbose=False)
            total_latency_us += step_latency_us
            total_flop_rate += step_flop_rate

            # Token selection: HW argmax register holds argmax(logits + penalty bias).
            token_id = self.get_arg_max_index()
            self._generated_tokens.append(token_id)
            token_char = self.tokenizer.decode([token_id])
            _SILENT_MODE = False
            if token_id in _qwen25_stop_tokens:
                if _use_status:
                    _status_teardown()
                print(f"\nStop token {token_id} reached.")
                break
            decoded_chars.append(token_char)
            print(token_char, end="", flush=True)
            if _use_status:
                _status_update()
        else:
            if _use_status:
                _status_teardown()

        # Decoder/HW counter status block (matches gemma3 / llama3.2 format).
        latency_decoder_wall = time.perf_counter() - decode_start
        tokens_decoded = max(1, self.seq_len - prefill_seq_start)
        total_wall = latency_prefill_wall + latency_decoder_wall
        hw_total_s = (getattr(self, '_latency_hw_prefill', 0.0) + total_latency_us) / 1e6
        avg_hw_gflops = total_flop_rate / tokens_decoded
        print(f"\nDecoder done in {total_wall:.2f} seconds, "
              f"speed: {tokens_decoded / latency_decoder_wall:.2f} tokens/s, "
              f"total {self.seq_len} tokens.")
        print(f"HW counter: Latency: {hw_total_s:.2f} seconds, "
              f"decoder average Gflops: {avg_hw_gflops:.2f} Gflops")
        return self.seq_len, "".join(decoded_chars)

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

def _clock_ns_default_for_device(device: str) -> float:
    """Return default clock period (ns) for FPGA type — mirrors user_hw_test.py."""
    if device == "kintex7":                       return 5.1594
    if device in ("rk", "puzhi"):                 return 3.0
    if device in ("bittware", "bittware_256"):     return 3.3333
    if device == "alveo":                          return 4.0
    return 10.0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Qwen2.5-VL-3B prefill + decode on accelerator.")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt (default: describe the default image).")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to an image file -> VLM mode. Default uses "
                             "test_samples/yosemite.jpg; 'none' forces text-only.")
    parser.add_argument("--vision-enable", action="store_true",
                        help="Compatibility flag; default already uses the sample image "
                             "unless --image none is given.")
    # On-FPGA repetition penalty is the DEFAULT decode path: the penalty is folded into the LM-head
    # matmul bias so the HW argmax returns the penalized token directly — no logit readback,
    # fully deterministic. --pure-greedy disables it (the bias buffer is zeroed; one bin serves both).
    parser.add_argument('--pure-greedy', action='store_true',
                        help='Disable the on-FPGA repetition penalty entirely — plain greedy decode. '
                             'The penalty is ENABLED by default; use --pure-greedy only for the A/B '
                             'baseline and the compare/calibration tool.')
    pen_group = parser.add_argument_group('on-FPGA repetition penalty (active unless --pure-greedy)')
    pen_group.add_argument('--greedy-until', type=int, default=512,
                        help='Pure greedy for the first N decoded tokens (correct math/reasoning, '
                             'which lands early), then the penalty turns on to break long-context '
                             'loops. 0 = penalty from the start. Default 512.')
    pen_group.add_argument('--pen-alpha', type=float, default=1.0,
                        help='bias[t] = -alpha*count[t] (logit units). Default 1.0.')
    pen_group.add_argument('--pen-cap', type=float, default=20.0,
                        help='max |bias| per token (floor on -alpha*count). Default 20.')
    pen_group.add_argument('--rep-window', type=int, default=256,
                        help='count tokens over the last N (never penalizes punctuation/whitespace/'
                             'special tokens). Default 256.')
    parser.add_argument('--dev', type=str, default='xdma0',
                        help='DMA device name (e.g., xdma0, xdma1). Default: xdma0')
    parser.add_argument('--cycle', type=float, default=None, help='Clock cycle time in ns. Overrides --device default.')
    parser.add_argument('--device', type=str, default='kintex7', help='FPGA board profile (kintex7, rk, puzhi, bittware, bittware_256, alveo).')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    set_dma_device(args.dev)
    global DMA_DEVICE_H2C, DMA_DEVICE_C2H, DMA_DEVICE_USER
    DMA_DEVICE_H2C = user_dma_core.DMA_DEVICE_H2C
    DMA_DEVICE_C2H = user_dma_core.DMA_DEVICE_C2H
    DMA_DEVICE_USER = user_dma_core.DMA_DEVICE_USER
    axi_width_bits = 512 if args.device in ("bittware", "rk") else 256
    os.environ["UE_AXI_DATA_WIDTH_BITS"] = str(axi_width_bits)
    user_dma_core.UE_AXI_DATA_WIDTH_BITS = axi_width_bits
    clock = args.cycle if args.cycle is not None else _clock_ns_default_for_device(args.device)
    user_dma_core.CLOCK_CYCLE_TIME_NS = clock
    user_dma_core.UE_PEAK_GFLOPS = 0.128 / clock
    print(f"FPGA profile: device={args.device}, clock={clock:.4f} ns, UE_AXI_DATA_WIDTH_BITS={axi_width_bits}")

    ue = Qwen25VL3B_UnifiedEngine(script_dir=script_dir)

    ue.start_capture()
    ue.generate_instruction_halt()
    ue.stop_capture()
    halt_bytes = bytearray()
    for inst in ue.capture_buffer:
        halt_bytes.extend(inst.get_bytes())
    ue.dma_write(DMA_DEVICE_H2C, DRAM_INSTRUCTION_ADDR, halt_bytes, len(halt_bytes))
    ue.clear_capture_buffer()

    cfg = _load_config(script_dir)
    force_text_only = bool(args.image and args.image.lower() == "none")
    if force_text_only:
        args.image = None
    # Default is VLM with the sample image. Use --image none to force LM text-only.
    if args.image is None and not force_text_only:
        args.image = os.path.join(SCRIPT_DIR, "../../test_samples/yosemite.jpg")
    has_image = args.image is not None
    # Vision encoder always runs on FPGA when an image is given.
    fpga_vision = has_image
    if args.prompt is None:
        args.prompt = ("please describe the image in details." if has_image
                       else "What is the capital of France?")

    print(f"\n--- Configuration ---")
    print(f"  Mode  : {'VLM (vision + LM)' if has_image else 'LM (text-only)'}")
    print(f"  Image : {args.image if has_image else '(none)'}")
    print(f"  Prompt: {args.prompt!r}")

    t_compile = t_total = 0.0  # vision compile/total wall (set by the FPGA vision branch)

    if fpga_vision:
        # FPGA vision encoder — preprocess only; the encoder PROGRAM is compiled
        # as part of the unified instruction bin (compile_all) below, then run.
        print(f"\n--- Vision Encoder (FPGA) ---")
        timer_vis = time.perf_counter()
        _pixel_values = process_image(args.image)
        print(f"Image loaded: {args.image} -> {_pixel_values.shape}")
        ue.prepare_encoder_input(_pixel_values)  # sets _cu_window_seqlens before compile

    # ---- Single complete instruction bin: prefill + decoder + encoder ----
    # Built once (first run), serves BOTH LM-only and VLM. LM section is captured
    # first at a fixed base so its §7 JUMP_ABS targets never depend on the encoder;
    # the encoder is appended after and only executed for VLM. See compile_all.
    print(f"\n--- Compiling complete instruction bin ---")
    _t_uni = time.perf_counter()
    progs = ue.compile_all()
    t_uni_compile = time.perf_counter() - _t_uni
    print(f"  {t_uni_compile:.2f}s")

    if fpga_vision:
        t_compile = time.perf_counter() - timer_vis
        ue.run_encoder(progs["encoder_addr"], _pixel_values)
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

    # Prefill + decoder programs already live in the unified bin (compile_all).
    prefill_program_addr  = progs["prefill_addr"]
    prefill_preamble_addr = progs["prefill_preamble"]
    gflops_prefill        = progs["prefill_flops"]
    decoder_program_addr  = progs["decoder_addr"]
    gflops_per_token      = progs["decoder_flops"]
    t_prefill_compile = t_decode_compile = 0.0  # folded into the unified compile

    print(f"\n--- Prefill run ({len(prefill_seq)} tokens) ---")
    timer = time.perf_counter()
    ue.run_prefill(prefill_program_addr, prefill_preamble_addr, prefill_seq=prefill_seq, gflops=gflops_prefill, has_image=has_image)
    latency_prefill = time.perf_counter() - timer
    prefill_hw_gflops = getattr(ue, '_last_hw_gflops', 0)
    print(f"  {latency_prefill:.2f}s")

    # Restore RoPE for decoder
    if has_image:
        ue.restore_rope_for_decoder()
        ue._rope_offset = getattr(ue, '_mrope_delta', 0)
    else:
        ue._rope_offset = 0

    # On-FPGA repetition-penalty config (deterministic, no logit readback).
    ue.fpga_penalty = not args.pure_greedy
    ue.greedy_until = int(args.greedy_until)
    ue.pen_alpha    = float(args.pen_alpha)
    ue.pen_cap      = float(args.pen_cap)
    ue.rep_window   = int(args.rep_window)
    ue._generated_tokens = list(prefill_seq)  # seed the penalty window with the prompt
    if ue.fpga_penalty:
        print(f"Decode: on-FPGA penalty (LM-head bias) — pure greedy for {ue.greedy_until} tokens, "
              f"then alpha={ue.pen_alpha} cap={ue.pen_cap} window={ue.rep_window}")
    else:
        print("Decode: plain greedy (--pure-greedy)")

    print(f"\n--- Decode run ---")
    timer = time.perf_counter()
    token_cnt_decoded, decoded_text = ue.run_decoder(
        decoder_program_addr, token_id=prefill_seq[-1],
        gflops=gflops_per_token, latency_prefill_wall=latency_prefill)
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
    print(f"  Unified compile:  {t_uni_compile:.2f}s  (encoder+prefill+decoder, one bin)")
    prefill_flops = gflops_prefill if isinstance(gflops_prefill, (int, float)) else 0
    print(f"  Prefill run:      {latency_prefill:.2f}s  ({prefill_hw_gflops:.2f} GFLOPS)  ({len(prefill_seq)} tokens)")
    print(f"  Decode run:       {latency_decoder:.2f}s  ({n_new} tokens, {latency_decoder/max(n_new,1):.2f}s/tok)")
    total = (t_total if fpga_vision else 0) + t_uni_compile + latency_prefill + latency_decoder
    print(f"  ──────────────────────────")
    print(f"  Total:            {total:.2f}s")

    # Clear FPGA DRAM so stale scratch doesn't bleed into subsequent runs
    # (see [[fpga_compare_clear_dram_oracle.md]] — 400 max|d| floor without this).
    ue.clear_dram()
    print("Qwen2.5-VL-3B test ends.")

if __name__ == "__main__":
    main()
