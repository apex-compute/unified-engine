#!/usr/bin/env python3
"""LocateAnything-3B — full on-device path: FPGA(MoonViT + connector + decoder).

MoonViT, the MLP connector, and the Qwen2.5-3B decoder all run on the FPGA board.
No CUDA/GPU is required. A CPU-only PyTorch reference (own-PyTorch impl in
locateanything_3b_cpu_test.py) is built solely to supply weights/tokenizer and,
when --check-vision/--compare are passed, an optional parity reference.

The decoder reuses the qwen2.5_vl_3b engine verbatim (dimensionally identical LM):
  * config redirected to locateanything_3b_fpga_config.json (vocab 152681, 1D rope)
  * LM weights from params.bin region 'lm' (see gen_lm_bin)
  * vision runs on the board (MoonViT weights loaded separately)
  * image-token id remapped to the placeholder run_prefill scans for

Run (apex-compute env, board attached):
  python locateanything_3b_test.py --query car
"""

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
from user_dma_core import DMA_DEVICE_H2C, TYPE, UE_FMAX_CONTEXT_SIZE, UE_VECTOR_SIZE, URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS, UE_MODE, configure_device
from nn_lib import eltwise_add_core_dram, eltwise_mul_core_dram, rms_norm_core_dram_post_add
from nn_lib import store_weight, store_quantized_weight, smart_bf16_permute_core, layer_norm_core_dram_post_add
from quant_lib import quantize_q4_64
from user_dma_core import UnifiedEngine, DRAM_INSTRUCTION_ADDR, MEMCPY_TYPE
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


def _load_tokenizer(model_dir: str):
    """Load LocateAnything/Qwen tokenizer with the corrected Mistral regex when supported."""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


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


# ============================================================================
# Consolidated bin helpers: params.bin/params.json + programs.bin/programs.json
# All weights (lm, connector, ...) live in ONE params.bin with a region manifest.
# All compiled programs (encoder, prefill variants, decoder, connector variants)
# live in ONE programs.bin with a per-program manifest.
# ============================================================================

def _la_bin_dir(script_dir):
    d = os.path.join(script_dir, "locateanything_3b_bin")
    os.makedirs(d, exist_ok=True)
    return d


def _params_paths(script_dir):
    d = _la_bin_dir(script_dir)
    return os.path.join(d, "params.bin"), os.path.join(d, "params.json")


def _programs_paths(script_dir):
    d = _la_bin_dir(script_dir)
    return os.path.join(d, "programs.bin"), os.path.join(d, "programs.json")


def params_bin_write_region(script_dir, region_name, raw_bytes, sub_manifest):
    """Append/overwrite a region in params.bin. Region's sub_manifest entries
    have offsets relative to region start; we store absolute offsets in
    params.json under regions[region_name].entries.
    """
    bin_path, meta_path = _params_paths(script_dir)
    # Load existing manifest if any
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"regions": {}}
    manifest.setdefault("regions", {})

    # If region already present, we rewrite the whole file (simpler than splicing).
    regions = manifest["regions"]
    # Collect existing region blobs (excluding the one we're (re)writing)
    keep = []
    if os.path.exists(bin_path):
        with open(bin_path, "rb") as f:
            old = f.read()
        for rname, rmeta in regions.items():
            if rname == region_name:
                continue
            keep.append((rname, rmeta, old[rmeta["offset"]:rmeta["offset"] + rmeta["size"]]))

    # Rebuild
    out = bytearray()
    new_regions = {}
    for rname, rmeta, blob in keep:
        new_regions[rname] = {"offset": len(out), "size": len(blob), "entries": rmeta.get("entries", {})}
        out.extend(blob)
    # Append the new/updated region
    region_offset = len(out)
    new_regions[region_name] = {"offset": region_offset, "size": len(raw_bytes), "entries": sub_manifest}
    out.extend(raw_bytes)

    with open(bin_path, "wb") as f:
        f.write(bytes(out))
    manifest["regions"] = new_regions
    with open(meta_path, "w") as f:
        json.dump(manifest, f)


def params_bin_load_region(script_dir, region_name):
    """Return {entry_key: raw_numpy_uint8} for a region, or None if missing."""
    bin_path, meta_path = _params_paths(script_dir)
    if not (os.path.exists(bin_path) and os.path.exists(meta_path)):
        return None
    with open(meta_path) as f:
        manifest = json.load(f)
    regions = manifest.get("regions", {})
    if region_name not in regions:
        return None
    rmeta = regions[region_name]
    region_start = rmeta["offset"]
    with open(bin_path, "rb") as f:
        f.seek(region_start)
        blob = f.read(rmeta["size"])
    cache = {}
    for k, m in rmeta.get("entries", {}).items():
        cache[k] = np.frombuffer(blob[m["offset"]:m["offset"] + m["size"]], dtype=np.uint8).copy()
    # Also expose entry meta (e.g. shape) under a sentinel
    cache["__entries__"] = rmeta.get("entries", {})
    return cache


def programs_bin_load(script_dir, key):
    """Return (prog_bytes, meta_dict) for a program key, or None if absent."""
    bin_path, meta_path = _programs_paths(script_dir)
    if not (os.path.exists(bin_path) and os.path.exists(meta_path)):
        return None
    with open(meta_path) as f:
        manifest = json.load(f)
    progs = manifest.get("programs", {})
    if key not in progs:
        return None
    m = progs[key]
    with open(bin_path, "rb") as f:
        f.seek(m["offset"])
        data = f.read(m["size"])
    return data, m.get("meta", {})


def programs_bin_store(script_dir, key, prog_bytes, meta=None):
    """Append/overwrite a program entry in programs.bin + programs.json."""
    bin_path, meta_path = _programs_paths(script_dir)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"programs": {}}
    manifest.setdefault("programs", {})
    progs = manifest["programs"]

    # Rewrite to keep file in sync with manifest (drop the key if present).
    keep = []
    if os.path.exists(bin_path):
        with open(bin_path, "rb") as f:
            old = f.read()
        for pname, pmeta in progs.items():
            if pname == key:
                continue
            keep.append((pname, pmeta, old[pmeta["offset"]:pmeta["offset"] + pmeta["size"]]))

    out = bytearray()
    new_progs = {}
    for pname, pmeta, blob in keep:
        new_progs[pname] = {"offset": len(out), "size": len(blob), "meta": pmeta.get("meta", {})}
        out.extend(blob)
    new_progs[key] = {"offset": len(out), "size": len(prog_bytes), "meta": (meta or {})}
    out.extend(bytes(prog_bytes))

    with open(bin_path, "wb") as f:
        f.write(bytes(out))
    manifest["programs"] = new_progs
    with open(meta_path, "w") as f:
        json.dump(manifest, f)

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
# LM-side quant scope: q/k/o/gate/up/down quantized (if4); only v_proj stays BF16
# (tiny, attention-value sensitive). o_proj+down_proj moved BF16->if4 to cut the
# decode weight-bandwidth (down_proj 45MB->11MB/layer). if4 is near-lossless
# (~+0.9% PPL). Must stay in sync with _gen_lm_bin.py::_LM_QUANT_SUFFIXES.
_LM_QUANT_LAYERS = {'q_proj.weight', 'k_proj.weight', 'o_proj.weight',
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
        # LocateAnything keeps v/o/down in BF16, so LM weights reach ~3.08GB and
        # run past the old scratch base (0x90000000) and program base (0xA0000000)
        # -> runtime writes stomped layers ~28-35's weights (NaN at decode layer 29).
        # Relocate scratch + program arenas ABOVE the weight blob; the device is 4GB
        # (offset ceiling 0x100000000). With o_proj/down_proj now if4 the LM weights
        # are ~1.7GB (was 3.08GB), and ENABLING THE ON-DEVICE VISION ENCODER adds
        # ~212MB (if4) / ~800MB (bf16) of MoonViT weights into params PLUS large
        # VIS_* scratch buffers (~290MB) and a 377MB encoder program. The old
        # 0xC0/0xCE bases were sized for LM-only and the encoder's buffers/programs
        # collided near 0xCE000000 -> decode residual NaN. Spread the three arenas:
        #   params:  0x0          .. <0xB0000000  (2.75GB: LM ~1.7GB + conn + vision)
        #   tensor:  0xB0000000   .. <0xDC000000  (704MB scratch: LM ~208MB + VIS ~290MB)
        #   program: 0xDC000000   .. <0x100000000 (576MB: encoder 377MB + prefill/decode)
        # Invariant: params_end < tensor_dram_base < program_dram_base < 0x100000000.
        super().__init__(
            BASE_ADDR=user_dma_core.UE_0_BASE_ADDR,
            params_dram_base=user_dma_core.DRAM_START_ADDR,
            tensor_dram_base=0xB0000000,
            program_dram_base=0xDC000000,
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
        # Dynamic-PBI GPR layout:
        #   reg 1 = TMP_REG          — scratch for reg_mul_imm + add_imm address math
        #   reg 2 = GF_SEQ_LEN_REG   — runtime row count for matmul/norm ops (M=seq_len)
        #   reg 3 = GF_Q_SEQ_LEN_REG — for ops with M = seq_len * group_size
        #   reg 4 = GF_ALIGNED_SEQ_LEN_REG — runtime KV length for unified_attention_core
        # Dynamic GPR allocation (via alloc_isa_reg) starts at 5.
        fixed = self._cfg["fixed_isa_regs"]
        self.TMP_REG       = fixed["TMP_REG"]
        self.gf_seq_len    = fixed["GF_SEQ_LEN_REG"]
        self.gf_q_seq_len  = fixed["GF_Q_SEQ_LEN_REG"]
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

    def dma_write(self, device, addr, data, size):
        """Skip redundant identity-matrix DMA writes (already in DRAM from pre-allocation)."""
        if (self._identity_dram_written
                and self._identity_dram_addr is not None
                and addr == self._identity_dram_addr
                and size == self._IDENTITY_MAT_BYTES):
            return  # already written; same content, skip
        super().dma_write(device, addr, data, size)

    def reset_isa_reg_counter(self) -> None:
        """Reset PBI ISA-reg allocator back to 5 (regs 1-4 are fixed/reserved)."""
        self._isa_reg_counter = 5

    def alloc_isa_reg(self, reset: bool = False) -> int:
        """Allocate next ISA register (5-63). Regs 1-4 are fixed model preamble regs."""
        if reset:
            self._isa_reg_counter = 5

        if self._isa_reg_counter > 63:
            raise ValueError("Exceeded available ISA registers (max 63)")

        reg_idx = self._isa_reg_counter
        self._isa_reg_counter += 1
        return reg_idx

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
        model_dir = os.path.join(self.script_dir, self._cfg["paths"]["hf_model_dir"])
        hf_repo = self._cfg["paths"]["hf_model_repo"]
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            _original_print(f"Downloading HF model {hf_repo} to {os.path.abspath(model_dir)} ...")
            snapshot_download(repo_id=hf_repo, local_dir=model_dir, local_dir_use_symlinks=False)
            _original_print("Download complete.")

        # Load LM weights from consolidated params.bin (region "lm").
        # Auto-generate if the region is missing.
        print(f"  Reading LM weight region from params.bin ...", flush=True)
        lm_cache = params_bin_load_region(self.script_dir, "lm")
        if lm_cache is None:
            _original_print("LM region not found in params.bin, generating...")
            gen_lm_bin(precision=_lm_precision(self._cfg))
            lm_cache = params_bin_load_region(self.script_dir, "lm")
            assert lm_cache is not None, "gen_lm_bin did not produce 'lm' region"

        # Embedding (host-side only — NOT uploaded to FPGA DRAM)
        # Embedding lookup is done on host in get_embedding_for_tokens()
        embed_raw = lm_cache['language_model.embed_tokens.weight']
        embed_bf16 = torch.from_numpy(embed_raw.copy()).view(torch.bfloat16).reshape(self.EMBEDDING_ELEMENTS, self.vector_length)
        self.embedding_weight = embed_bf16.clone()  # host copy for token lookup

        self.tokenizer = _load_tokenizer(model_dir)

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
            # Q, K, gate, up: precision.lm-quantized from binary
            for proj, hf_sub in [('q', 'self_attn.q_proj'), ('k', 'self_attn.k_proj'),
                                  ('gate', 'mlp.gate_proj'), ('up', 'mlp.up_proj')]:
                la[f'{proj}_scale'], la[f'{proj}_data'] = store_quantized_weight(self, lm_cache[f'{prefix}.{hf_sub}.weight.{lm_prec}'])
            # V stays BF16 (small, attention-value sensitive); O + down_proj if4-quantized.
            la['v_weight'] = store_weight(self, torch.from_numpy(lm_cache[f'{prefix}.self_attn.v_proj.weight'].copy()).view(torch.bfloat16))
            la['o_scale'], la['o_data'] = store_quantized_weight(self, lm_cache[f'{prefix}.self_attn.o_proj.weight.{lm_prec}'])
            la['down_scale'], la['down_data'] = store_quantized_weight(self, lm_cache[f'{prefix}.mlp.down_proj.weight.{lm_prec}'])
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

        # Vision weights (all from binary — pre-padded QKV and MLP).
        # LADecoder overrides this to a no-op (FPGA MoonViT pulls live HF weights);
        # the base Qwen2.5-VL implementation expects a path argument and is unused
        # in LA paths.
        self._load_vision_weights(None)

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
        prefill_q_seq = self.PREFILL_MAX_SEQ_LEN * self.group_size
        prefill_aligned = ((prefill_q_seq + 63) // 64) * 64
        max_batch = max(prefill_q_seq, self.group_size)
        max_aligned_seq_len = max(prefill_aligned, self.MAX_CONTEXT_SIZE)
        self.LAYER0_FLASH_SCRATCH_DRAM = self.allocate_tensor_dram(
            ((ahd + max_aligned_seq_len) * max_aligned_seq_len + max_batch * ahd) * bpe)
        # PBI prefill packs FLASH_Q with qpkv rows per token (q_seq_len = PREFILL_MAX_SEQ_LEN * qpkv),
        # so BIAS is sized to (prefill_aligned)^2 where prefill_aligned = PREFILL_MAX_SEQ_LEN*qpkv aligned to UE_VECTOR_SIZE.
        self.LAYER0_FLASH_BIAS_DRAM = self.allocate_tensor_dram(prefill_aligned * prefill_aligned * bpe)

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
            ((VD_PAD + VS) * VS + VS * VD_PAD) * bpe)
        # Vision attention fixed buffers. Each per-(head,window) call site marshals
        # its segment in/out. Sized for the largest segment (full attention = VS patches).
        self.VIS_FLASH_Q_DRAM   = self.allocate_tensor_dram(VS * VD_PAD * bpe)
        self.VIS_FLASH_K_DRAM   = self.allocate_tensor_dram(VS * VD_PAD * bpe)
        self.VIS_FLASH_V_DRAM   = self.allocate_tensor_dram(VS * VD_PAD * bpe)
        self.VIS_FLASH_OUT_DRAM = self.allocate_tensor_dram(VS * VD_PAD * bpe)
        self.VIS_ATTN_BIAS_DRAM = self.allocate_tensor_dram(VS * VS * bpe)
        self.dma_to_accelerator_memory(self.VIS_ATTN_BIAS_DRAM, torch.zeros(VS * VS, dtype=torch.bfloat16))
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
        # Replicated VN× along rows so ONE rope_hf_core_dram call (M=VN*VS) can
        # rotate all heads at once (heads are contiguous in VIS_Q/K_PAD_DRAM).
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

        # On-FPGA repetition penalty: per-vocab additive bias (the LM-head matmul's C term,
        # bias_mode="broadcast_N"). Allocated LAST so adding it never shifts addresses
        # baked into the bin. All-zero = no penalty.
        self.PENALTY_BIAS_DRAM = self.allocate_tensor_dram(1 * self.EMBEDDING_ELEMENTS * bpe)
        self.dma_to_accelerator_memory(self.PENALTY_BIAS_DRAM,
                                       torch.zeros(1, self.EMBEDDING_ELEMENTS, dtype=torch.bfloat16))

        print(f"    Allocate tensor dram end at DRAM address: 0x{self.get_tensor_dram_addr():X}, usage: {self.get_tensor_dram_usage()} bytes")
        # Pre-allocate the identity matrix used by attention cores.
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

    def _vis_transpose_10(self, input_addr: int, output_addr: int, d0: int, d1: int, last: int) -> None:
        """Transpose [d0, d1, last] -> [d1, d0, last] (permute [1,0,2]) via strided DMA.

        One gather DMA pair per OUTPUT block (d1 iterations) instead of
        bf16_permute_core_v2's per-row unroll (d0*d1 rows). For the head<->token
        swaps (one dim = VN = 16) this is ~16 or ~576 iters vs ~9216.
        NOTE: strided WRITE (scatter) produces wrong output on device — stay on
        the gather (strided read) path."""
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
        cache_suffix = f"_{total_layers}L" if total_layers != vis_cfg["depth"] else ""
        if _bf16_mode:
            cache_suffix += "_bf16"
        enc_prog_key = f"encoder{cache_suffix}_unified_attention"

        _cached = programs_bin_load(self.script_dir, enc_prog_key)
        if _cached is not None:
            prog_bytes, _ = _cached
            _original_print(f"  Loading cached encoder program ({enc_prog_key}) from programs.bin ...")
            program_addr = self.get_program_dram_addr()
            self.dma_write(DMA_DEVICE_H2C, program_addr, prog_bytes, len(prog_bytes))
            self.allocate_program_dram(len(prog_bytes))
            self._vis_program_addr = program_addr
            _original_print(f"    Loaded {len(prog_bytes)} bytes at 0x{program_addr:X}")
            return program_addr

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
        # Separate M-reg for the head-collapsed rope (M=VN*VS over the contiguous
        # [VN,VS,VD_PAD] head buffer). One call/buf replaces the per-head loop.
        vis_rope_M_reg = self.alloc_isa_reg()
        self.generate_instruction_add_set(vis_rope_M_reg, VN * VS)

        def _vis_flash_call(q_src, k_src, v_src, out_dst, seg_len):
            """Marshal one segment's Q/K/V into fixed buffers, run unified attention, scatter to out_dst."""
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
            for buf_dram in (self.VIS_Q_PAD_DRAM, self.VIS_K_PAD_DRAM):
                self.rope_hf_core_dram(
                    M=VN * VS, N=VD_PAD,
                    input_dram_addr=buf_dram, output_dram_addr=buf_dram,
                    cos_dram_addr=self.VIS_ROPE_COS_DRAM,
                    sin_dram_addr=self.VIS_ROPE_SIN_DRAM,
                    gpr_M_reg=vis_rope_M_reg)

            # Attention uses the padded head dim scale 1/sqrt(128) instead of 1/sqrt(80).
            # The softer attention (sqrt(128)) is more robust to INT4 quantization noise
            # and empirically gives better end-to-end results than correcting to sqrt(80).
            fullatt_indexes = set(vis_cfg.get("fullatt_block_indexes", [7, 15, 23, 31]))
            cu_win = getattr(self, '_cu_window_seqlens', [0, VS])
            is_full_attn = (layer_idx in fullatt_indexes)

            # §7: each call site marshals its segment into VIS_FLASH_* and jumps
            # into the shared flash subroutine compiled after HALT below.
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

        _original_print()  # newline after \r progress

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

        # Save compiled program to consolidated programs.bin for fast reload next time
        programs_bin_store(self.script_dir, enc_prog_key, prog_bytes,
                           meta={"total_layers": total_layers, "bf16_mode": _bf16_mode})
        _original_print(f"    Vision encoder compiled: {total_bytes} total bytes, {total_layers} layers + merger (single program)")
        _original_print(f"    Program addr: 0x{program_addr:X}, cached to programs.bin[{enc_prog_key}]")
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
                from transformers import Qwen2VLImageProcessor
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
                tokenizer = _load_tokenizer(model_dir)
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

            # Replicate VN× along rows: block h = identical [VS,2,VD_PAD] copy, so a
            # single rope call over the contiguous [VN,VS,VD_PAD] head buffer reads the
            # correct position table for every (head, pos) row.
            rope_table_rep = rope_table.unsqueeze(0).expand(VN, VS, 2, VD_PAD).reshape(VN * VS, 2, VD_PAD)
            self.dma_to_accelerator_memory(self.VIS_ROPE_DRAM, rope_table_rep.flatten())
        print(f"    Vision RoPE table [{VN}x{VS}, 2, {VD_PAD}] (interleaved cos||sin, head-replicated) DMA'd to FPGA")

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
        ``gpr_M_reg=gf_seq_len``. Attention receives runtime batch and aligned
        sequence length through the fixed GPRs.

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

            # o_proj (if4-quantized)
            total_flops += (self.matmat_mul_core(is_B_quantized=True, M=seq_len, K=hd * qpkv, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM, B_DRAM_ADDR=la['o_data'], OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                SCALE_DRAM_ADDR=la['o_scale'], data_type=TYPE.IF4,
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
        """Compile single PBI prefill program. Returns (program_addr, preamble_addr, total_flops).

        Cache key: (PREFILL_MAX_SEQ_LEN, layer_size). The bin is seq_len-agnostic
        for actual_seq_len ≤ PREFILL_MAX_SEQ_LEN, primed at runtime via the
        preamble emitted in run_prefill.
        """
        if layer_size is None:
            layer_size = self.LAYER_SIZE

        template_seq_len = self.PREFILL_MAX_SEQ_LEN
        prefill_key = f"prefill_pbi_s{template_seq_len}"

        _cached = programs_bin_load(self.script_dir, prefill_key)
        if _cached is not None:
            prog_bytes, meta = _cached
            _original_print(f"  Loading cached PBI prefill program ({prefill_key}) from programs.bin ...")
            program_addr = self.get_program_dram_addr()
            self.dma_write(DMA_DEVICE_H2C, program_addr, prog_bytes, len(prog_bytes))
            self.allocate_program_dram(len(prog_bytes))
            preamble_addr = self.allocate_program_dram(256)
            return program_addr, preamble_addr, meta["total_flops"]

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

        programs_bin_store(self.script_dir, prefill_key, prog_bytes,
                           meta={"total_flops": total_flops, "template_seq_len": template_seq_len, "layer_size": layer_size})
        print(f"    PBI prefill cached to programs.bin[{prefill_key}] ({len(prog_bytes)} bytes)")
        print(f"    Prefill program at 0x{program_addr:X}, preamble slot at 0x{preamble_addr:X}")
        return program_addr, preamble_addr, total_flops

    def prefill_theoretical_flops(self, seq_len: int, layer_size: int | None = None) -> int:
        """Analytic forward-pass FLOPs for a `seq_len`-token prefill (2 flops/MAC).
        Uses the model's true dims (not the padded/packed engine internals) so the
        achieved-GFLOPS number is a meaningful MFU-style figure. Attention is
        non-causal (LocateAnything block decoding) -> full S^2."""
        L   = layer_size if layer_size is not None else self.LAYER_SIZE
        S   = seq_len
        d   = self.vector_length
        dff = self.mlp_elements
        hd  = self.actual_head_dim
        nkv = self.num_kv_heads
        nh  = nkv * self.group_size          # query heads
        per_layer = (
            2 * S * d * (nh * hd + 2 * nkv * hd)   # Q,K,V projections
            + 2 * 2 * S * S * nh * hd              # QK^T + scores·V (full, non-causal)
            + 2 * S * (nh * hd) * d                # output projection
            + 6 * S * d * dff                      # SwiGLU MLP (gate+up+down)
        )
        return per_layer * L

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
        # Use a FINITE mask sentinel (matching run_decoder's -1e36), NOT float("-inf").
        # The tiled flash softmax tracks a per-tile max; a fully-masked 64-wide tile
        # (common for early causal rows) gives tile_max=-inf -> exp(-inf-(-inf))=NaN,
        # which contaminates the running sum and propagates NaN through every layer.
        # A large finite negative keeps the tile max finite (exp(0)=1, normalized away).
        NEG_MASK = -1e36
        bias_one_group = torch.full((aligned_seq_len, aligned_seq_len), NEG_MASK, dtype=torch.bfloat16)
        valid_mask = torch.tril(torch.ones(aligned_seq_len, aligned_seq_len, dtype=torch.bool), diagonal=0) if not self.causal_mask_upper else torch.triu(torch.ones(aligned_seq_len, aligned_seq_len, dtype=torch.bool), diagonal=0)
        bias_one_group.masked_fill_(valid_mask, 0.0)
        bias_one_group[:, q_seq_len:] = NEG_MASK
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_one_group)

        # Emit runtime preamble: ADD_SETs for the 3 GPRs + JUMP_ABS into prefill.
        self.clear_inst_id()
        self.start_capture()
        self.generate_instruction_add_set(self.gf_seq_len, actual_seq_len)
        self.generate_instruction_add_set(self.gf_q_seq_len, q_seq_len)
        self.generate_instruction_add_set(self.gf_aligned_seq_len, aligned_seq_len)
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(prefill_program_addr))
        self.stop_capture()
        self.write_captured_instructions_to_dram(preamble_addr)
        self.clear_capture_buffer()

        self.start_execute_from_dram(preamble_addr)
        wall_s = self._wait_with_heartbeat(lambda: self.wait_queue(180.0), label="prefill")
        latency_us = self._corrected_hw_latency_us(wall_s)
        print(f"    Prefill executed: actual_seq_len={actual_seq_len}, q_seq_len={q_seq_len}, aligned_seq_len={aligned_seq_len}")

        # Throughput: theoretical model FLOPs at the ACTUAL seq len / measured HW latency.
        total_flops = self.prefill_theoretical_flops(actual_seq_len)
        hw_gflops = total_flops / (latency_us * 1e-6) / 1e9
        self._last_total_flops = total_flops
        self._last_hw_gflops = hw_gflops
        self._last_prefill_latency_us = latency_us
        print(f"    Prefill HW latency = {latency_us/1e3:.1f} ms  "
              f"({total_flops/1e9:.1f} GFLOP -> {hw_gflops:.1f} GFLOPS)")
        return dict(latency_us=latency_us, total_flops=total_flops, gflops=hw_gflops)

    # ------------------------------------------------------------------
    # NaN localizer: compile a K-layer prefill (NO cache), run it, and read
    # back intermediate DRAM buffers to find the first op that produces NaN.
    # ------------------------------------------------------------------
    def _compile_klayer_prefill(self, K: int):
        """Compile a fresh K-layer prefill program (bypasses the s384 cache).
        Returns (program_addr, preamble_addr)."""
        global _SILENT_MODE
        _SILENT_MODE = True
        self.clear_inst_id()
        self.start_capture()
        self._emit_prefill_program(seq_len=self.PREFILL_MAX_SEQ_LEN, layer_size=K)
        self.stop_capture()
        prog = bytearray()
        for inst in self.capture_buffer:
            prog.extend(inst.get_bytes())
        addr = self.get_program_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, addr, bytes(prog), len(prog))
        self.allocate_program_dram(len(prog))
        preamble = self.allocate_program_dram(256)
        self.clear_capture_buffer()
        _SILENT_MODE = False
        return addr, preamble

    def _scan(self, name: str, addr: int, M: int, N: int):
        """DMA back [M,N] bf16 from DRAM and report nan/inf/min/max."""
        t = self.dma_from_accelerator_memory(addr, (M, N)).cpu().float()
        nan = int(torch.isnan(t).sum())
        inf = int(torch.isinf(t).sum())
        finite = t[torch.isfinite(t)]
        lo = float(finite.min()) if finite.numel() else float("nan")
        hi = float(finite.max()) if finite.numel() else float("nan")
        flag = "  <-- FIRST NaN" if nan else ("  <-- inf" if inf else "")
        _original_print(f"    {name:26s} nan={nan:>7} inf={inf:>5} min={lo:+.3e} max={hi:+.3e}{flag}")
        return nan, inf

    def _layer_buffer_order(self):
        """Intermediate buffers in the exact order they are written within a
        layer. After a K-layer prefill these hold layer (K-1)'s compute."""
        ahd, qpkv, hd = self.actual_head_dim, self.group_size, self.head_dim
        V, MLP = self.vector_length, self.mlp_elements
        return [
            ("input (residual in)",   self.LAYER0_INPUT_DRAM,             V),
            ("pre_norm (ln1)",        self.LAYER0_PRE_NORM_DRAM,          V),
            ("q_proj",                self.LAYER0_Q_DRAM,                 hd * qpkv),
            ("k_proj+rope",           self.LAYER0_K_DRAM,                 hd),
            ("v_proj",                self.LAYER0_V_PROJ_TEMP,            hd),
            ("flash_output",          self.LAYER0_FLASH_OUTPUT_DRAM,      hd * qpkv),
            ("o_proj",                self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,  V),
            ("post_attn_residual",    self.LAYER0_POST_ATTN_RESIDUAL_DRAM, V),
            ("pre_mlp_norm (ln2)",    self.LAYER0_PRE_MLP_NORM_DRAM,      V),
            ("mlp_gate (silu)",       self.LAYER0_MLP_GATE_DRAM,          MLP),
            ("mlp_up",                self.LAYER0_MLP_UP_DRAM,            MLP),
            ("mlp_gate*up",           self.LAYER0_MLP_MULT_DRAM,          MLP),
            ("mlp_down",              self.LAYER0_MLP_DOWN_DRAM,          V),
            ("layer_output",          self.LAYER0_OUTPUT_DRAM,            V),
        ]

    def prefill_nan_probe(self, prefill_seq, max_layers: int = 36):
        """Walk layers 0..max_layers-1. For each layer j, run a (j+1)-layer
        prefill (so the LAYER0_* scratch holds layer j's compute) and scan every
        intermediate buffer in op order. Stop at the FIRST layer that produces a
        NaN/inf and report the exact op."""
        _original_print("\n=== NaN PROBE: per-layer op scan (stops at first NaN) ===")
        order = self._layer_buffer_order()
        for j in range(max_layers):
            prog, pre = self._compile_klayer_prefill(K=j + 1)
            self.run_prefill(prog, pre, prefill_seq=prefill_seq, has_image=True)
            M = self.seq_len
            _original_print(f"\n  --- layer {j} (ran K={j + 1} layers) ---")
            first_bad = None
            for name, addr, N in order:
                # For layers >0 the only meaningful "input" is the residual copy;
                # all op buffers belong to THIS layer j.
                nan, inf = self._scan(name, addr, M, N)
                if first_bad is None and (nan or inf):
                    first_bad = name
            if first_bad:
                _original_print(f"\n  >>> FIRST NaN: layer {j}, op '{first_bad}'")
                return True
        _original_print(f"\n  No NaN found through {max_layers} prefill layers.")
        _original_print("  => Prefill is finite. NaN must be in the readout/decode path.")
        return False

    def _compile_klayer_decoder(self, K: int):
        """Compile a fresh K-layer decoder program (bypasses cache). For K<36 it
        omits final-norm/lm_head, leaving layer K-1's output in LAYER0_OUTPUT_DRAM.
        Returns (program_addr, preamble_addr)."""
        global _SILENT_MODE
        _SILENT_MODE = True
        self.clear_inst_id()
        self.start_capture()
        self._emit_decoder_program(layer_size=K)
        self.stop_capture()
        prog = bytearray()
        for inst in self.capture_buffer:
            prog.extend(inst.get_bytes())
        addr = self.get_program_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, addr, bytes(prog), len(prog))
        self.allocate_program_dram(len(prog))
        preamble = self.allocate_program_dram(256)
        self.clear_capture_buffer()
        _SILENT_MODE = False
        return addr, preamble

    def _decode_one_step(self, program_addr, preamble_addr, token_id, seq_len_before):
        """Run exactly one decode step (token_id) at position seq_len_before,
        jumping into program_addr. Mirrors run_decoder's per-step body but does
        not read logits or advance persistent state beyond self.seq_len."""
        ahd = self.actual_head_dim
        bpe = self.bytes_per_element
        rope_row_bytes = ahd * 2 * bpe
        ROPE_BASE = self.DRAM_ADDR_ROPE

        self.seq_len = seq_len_before + 1
        step_pos = self.seq_len - 1
        aligned_seq_len = ((self.seq_len + 63) // 64) * 64
        rope_pos = step_pos + getattr(self, '_rope_offset', 0)
        cos_abs_addr = ROPE_BASE + rope_pos * rope_row_bytes

        emb = self.get_embedding_for_tokens([token_id])
        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, emb)
        bias_host = torch.full((self.group_size, aligned_seq_len), -1e36, dtype=torch.bfloat16)
        bias_host[:, :self.seq_len] = 0.0
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_host)

        self.clear_inst_id()
        self.start_capture()
        self.generate_instruction_add_set(self.gf_seq_len, step_pos)
        self.generate_instruction_add_set(self.gf_q_seq_len, self.seq_len)
        self.generate_instruction_add_set(self.gf_aligned_seq_len, aligned_seq_len)
        self.generate_instruction_add_set(self._gf_rope_cos_abs, ue_35bit_addr_shifter(cos_abs_addr))
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(program_addr))
        self.stop_capture()
        self.write_captured_instructions_to_dram(preamble_addr)
        self.clear_capture_buffer()
        self.start_execute_from_dram(preamble_addr)
        self.wait_queue(10.0)

    def decode_nan_probe(self, token_id, max_layers: int = 36):
        """After a full prefill, sweep K=1..36 decode layers: run ONE decode step
        through a K-layer decoder and scan layer (K-1)'s output. Stop at the first
        decode layer that goes NaN and dump its op buffers."""
        _original_print("\n=== DECODE NaN PROBE: per-layer single-step scan ===")
        seq0 = self.seq_len
        V = self.vector_length
        for j in range(max_layers):
            prog, pre = self._compile_klayer_decoder(K=j + 1)
            self._decode_one_step(prog, pre, token_id, seq_len_before=seq0)
            nan, inf = self._scan(f"decode layer {j} output", self.LAYER0_OUTPUT_DRAM, 1, V)
            if nan or inf:
                _original_print(f"\n  >>> first decode NaN at layer {j}. Op buffers for layer {j}:")
                for name, addr, N in self._layer_buffer_order():
                    self._scan(name, addr, 1, N)
                # Root-cause the ln1 blowup: dump gamma + input, recompute RMSNorm on host.
                la = self.lm_layer_addrs[j]
                gamma = self.dma_from_accelerator_memory(la['ln1_gamma'], (V,)).cpu().float()
                xin = self.dma_from_accelerator_memory(self.LAYER0_INPUT_DRAM, (1, V)).cpu().float().flatten()
                g_abs = gamma.abs()
                rms = (xin.pow(2).mean() + 1e-6).sqrt()
                host_norm = xin / rms * gamma
                gi = int(g_abs.argmax())
                _original_print(f"\n  [root] layer {j} ln1_gamma: min={gamma.min():+.3e} max={gamma.max():+.3e} "
                                f"|max|@dim={gi} (gamma={gamma[gi]:+.3e}, x={xin[gi]:+.3e})")
                _original_print(f"  [root] input rms={rms:.3e}  host RMSNorm: min={host_norm.min():+.3e} max={host_norm.max():+.3e}")
                _original_print(f"  [root] => {'GAMMA looks corrupt (huge entries)' if g_abs.max() > 1e3 else 'gamma normal -> HW norm path bug'}")
                # Address-map / boundary check across neighbor layers.
                _original_print("\n  [map] per-layer ln1_gamma DRAM addr + readback absmax (source=2.2):")
                for li in range(max(0, j - 5), min(self.LAYER_SIZE, j + 3)):
                    a = self.lm_layer_addrs[li]['ln1_gamma']
                    gv = self.dma_from_accelerator_memory(a, (V,)).cpu().float()
                    nanc = int(torch.isnan(gv).sum())
                    over = ">4GB-32b" if a > 0xFFFFFFFF else ""
                    _original_print(f"    layer {li:2d}: ln1_gamma @ 0x{a:011X} {over:8s} "
                                    f"absmax={gv.abs().max():.3e} nan={nanc}")
                _original_print(f"  [map] tensor LAYER0_INPUT_DRAM=0x{self.LAYER0_INPUT_DRAM:011X} "
                                f"LAYER0_OUTPUT_DRAM=0x{self.LAYER0_OUTPUT_DRAM:011X} "
                                f"params_base=0x{self._params_dram_base:011X} "
                                f"params_end=0x{self.get_params_dram_addr():011X}")
                _original_print(f"\n  >>> FIRST NaN: decode layer {j}")
                return
        _original_print("\n  No decode-layer NaN found (?!) — check final-norm/lm_head wiring.")

    def _emit_decoder_program(self, layer_size: int) -> int:
        """Emit ONE decode-position-agnostic decoder program (PBI single-bin pattern).

        Caller wraps this in start_capture()/stop_capture(). The emitted program
        ends with ADD_INC gf_seq_len + halt so it can be jumped to via a runtime
        preamble that primes gf_q_seq_len, gf_aligned_seq_len, and gf_rope_cos_abs
        each step.

        Address math: K/V cache write addresses are computed at runtime as
        ``base + gf_seq_len * (ahd*bpe)`` via reg_mul_imm + add_imm into TMP_REG,
        then passed as ``general_reg_src=TMP_REG`` to the scatter store.

        Decoder attention uses per-KV-head ``unified_attention_core``; each call
        processes all qpkv Q heads against the active KV range.

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
        # Two persistent regs survive the layer loop:
        #   gf_one          — gpr_M_reg for M=1 matmul/norm ops
        #   gf_rope_cos_abs — absolute cos-base addr, primed per step by preamble
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

            # Q, K, V projections with bias
            # Q/K (IF4, K=2048<8192) use quantized_matmat_core M=1 GEMV (fused dequant+dot ~2×)
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

                # Gather valid K history (incl. just-written token) from cache → FLASH_K.
                self._emit_pbi_scatter_per_token(
                    read_base=k_cache_base,
                    read_stride_bytes=ahd * bpe,
                    write_specs=[(self.LAYER0_FLASH_K_DRAM, ahd * bpe)],
                    sram_byte_addr=0x10000,
                    element_count=ahd,
                    gf_seq_len=self.gf_q_seq_len,
                    template_seq_len=self.MAX_CONTEXT_SIZE,
                )

                # V cache write
                self.accelerator_memory_to_sram(self.LAYER0_V_PROJ_TEMP + kv_h * ahd * bpe, 0x20000, ahd)
                self.generate_instruction_reg_mul_imm(
                    self.TMP_REG, self.gf_seq_len, ue_35bit_addr_shifter(ahd * bpe))
                self.generate_instruction_add_imm(
                    self.TMP_REG, ue_35bit_addr_shifter(v_cache_base), self.TMP_REG)
                self.sram_to_accelerator_memory(
                    sram_address=0x20000, accelerator_dram_address=0,
                    element_size=ahd, general_reg_src=self.TMP_REG)

                # Gather valid V history from cache → FLASH_V.
                self._emit_pbi_scatter_per_token(
                    read_base=v_cache_base,
                    read_stride_bytes=ahd * bpe,
                    write_specs=[(self.LAYER0_FLASH_V_DRAM, ahd * bpe)],
                    sram_byte_addr=0x20000,
                    element_count=ahd,
                    gf_seq_len=self.gf_q_seq_len,
                    template_seq_len=self.MAX_CONTEXT_SIZE,
                )

                # Gather qpkv Q heads for this KV head into FLASH_Q (contiguous: 0, ahd*bpe, ...)
                for q in range(qpkv):
                    q_src = self.LAYER0_Q_DRAM + (kv_h * qpkv + q) * ahd * bpe
                    flash_q_addr = self.LAYER0_FLASH_Q_DRAM + q * ahd * bpe
                    self.accelerator_memory_to_sram(q_src, 0x30000, ahd)
                    self.sram_to_accelerator_memory(0x30000, flash_q_addr, ahd)

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

                # Copy attention output (FLASH_OUT_HEAD) → this head's slot in FLASH_OUTPUT.
                self.accelerator_memory_to_sram(self.LAYER0_FLASH_OUT_HEAD_DRAM, 0x40000, qpkv * ahd)
                self.sram_to_accelerator_memory(
                    0x40000, self.LAYER0_FLASH_OUTPUT_DRAM + kv_h * qpkv * ahd * bpe, qpkv * ahd)

            # o_proj (IF4, K=hd*qpkv=2048<8192 → quantized_matmat_core M=1 GEMV)
            total_flops += (self.quantized_matmat_core(M=1, K=hd * qpkv, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM, B_DRAM_ADDR=la['o_data'], OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                SCALE_DRAM_ADDR=la['o_scale'], data_type=TYPE.IF4) or 0)

            # Qwen2.5-VL: no post-attention norm; residual direct on o_proj output
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_INPUT_DRAM, sram_address=0x10000, element_size=self.vector_length)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM, sram_address=0x90000, element_size=self.vector_length)
            self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, element_size=self.vector_length)

            # Qwen2.5-VL: post_attention_layernorm IS the pre-FFN norm
            total_flops += (self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                          OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=la['ln2_gamma'],
                          gpr_M_reg=gf_one) or 0)

            # MLP: SwiGLU (gate/up IF4, K=2048<8192 → quantized_matmat_core M=1 GEMV)
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

            # down_proj (if4-quantized)
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
            # LM head with on-FPGA repetition-penalty bias folded in as the matmul's C term
            # (bias_mode="broadcast_N"). HW argmax of (logits + bias) returns the penalized
            # token directly — no logit readback. write_back_disable=True always (no 304 KB
            # LOGITS_DRAM writeback needed). Bias buffer all-zero = pure greedy.
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
        """Compile-or-load single PBI decoder program; allocate per-step preamble slot.

        Cache hit: load bin from disk, DMA to program DRAM, restore stashed GPR indices.
        Cache miss: capture _emit_decoder_program, write to DRAM + bin/meta.

        Either way, ``self._decoder_preamble_addr`` is allocated as a small program slot
        rewritten each decode step by run_decoder.

        Returns (decoder_program_addr, total_flops).
        """
        if layer_size is None:
            layer_size = self.LAYER_SIZE
        # Cache key: "penalty_bias_argmax" identifies the on-FPGA penalty LM head
        # (C_DRAM_ADDR=PENALTY_BIAS_DRAM, write_back_disable=True). A stale bin from
        # the old host-side repetition-penalty build (missing or different key) is recompiled.
        lm_head_sig = "penalty_bias_argmax_unified_attention"
        cache_hit = False
        prog_bytes = None
        meta = {}
        _cached = programs_bin_load(self.script_dir, "decoder")
        if _cached is not None:
            prog_bytes, meta = _cached
            if "program_size" in meta and isinstance(meta.get("total_flops"), int) \
                    and meta.get("lm_head_sig") == lm_head_sig:
                cache_hit = True
            else:
                _original_print(f"  Decoder cache sig ({meta.get('lm_head_sig')!r}) != {lm_head_sig!r} — recompiling.")

        if cache_hit:
            decoder_program_addr = self.get_program_dram_addr()
            self.dma_write(DMA_DEVICE_H2C, decoder_program_addr, prog_bytes, len(prog_bytes))
            self.allocate_program_dram(len(prog_bytes))
            total_flops = meta["total_flops"]
            # Restore the GPR indices _emit_decoder_program would have allocated.
            self._gf_one          = 5
            self._gf_rope_cos_abs = 6
            print(f"  Decoder loaded from cache ({len(prog_bytes)} bytes) at 0x{decoder_program_addr:X}")
        else:
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

            programs_bin_store(self.script_dir, "decoder", prog_bytes,
                               meta={"program_size": len(prog_bytes), "total_flops": total_flops,
                                     "lm_head_sig": lm_head_sig})
            print(f"  Decoder program: 0x{decoder_program_addr:X} – 0x{self.get_program_dram_addr():X} "
                  f"({len(prog_bytes)} bytes), total_flops={total_flops}")

        # Allocate a small preamble slot (5 instructions × 32 B = 160 B; round up to 256 B).
        self._decoder_preamble_addr = self.get_program_dram_addr()
        self.allocate_program_dram(256)
        return decoder_program_addr, total_flops

    # ---- On-FPGA repetition penalty ----
    def _structural_token_ids(self) -> set:
        """Token ids that must never be repetition-penalized: punctuation, whitespace,
        newline, and special tokens. Cached on first call."""
        cached = getattr(self, "_struct_ids_cache", None)
        if cached is not None:
            return cached
        import string
        allowed = set(string.punctuation) | set(string.whitespace) | set("—–''""…·•‹›«»¡¿")
        ids = set(int(i) for i in (getattr(self.tokenizer, "all_special_ids", []) or []))
        for i in range(self.EMBEDDING_ELEMENTS):
            s = self.tokenizer.decode([i]).strip()
            if s == "" or all(ch in allowed for ch in s):
                ids.add(i)
        self._struct_ids_cache = ids
        return ids

    def _structural_ids_tensor(self) -> torch.Tensor:
        """1-D LongTensor of the structural/special token ids (cached)."""
        t = getattr(self, "_struct_ids_tensor_cache", None)
        if t is None:
            t = torch.tensor(sorted(self._structural_token_ids()), dtype=torch.long)
            self._struct_ids_tensor_cache = t
        return t

    def _write_penalty_bias(self, prev_tokens) -> None:
        """Build per-vocab additive bias from windowed token frequency and DMA it to
        PENALTY_BIAS_DRAM (LM-head matmul C term). bias[t] = clamp(-alpha*count[t], min=-cap);
        structural tokens stay 0. HW argmax of (logits + bias) returns the penalized token."""
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
                    gflops: int | None = None) -> int:
        """Run PBI decode loop. Per-step preamble primes gf_seq_len / gf_q_seq_len /
        gf_aligned_seq_len / gf_rope_cos_abs, then JUMP_ABS into the cached decoder program. Breaks on EOS/EOT.
        Token selection is always the HW argmax of (logits + PENALTY_BIAS_DRAM); no logit readback.
        """
        if token_id is None:
            print("No last token available for decode.")
            return self.seq_len

        _qwen25_stop_tokens = {151643, 151645, self._end_of_turn_token_id}

        ahd = self.actual_head_dim
        bpe = self.bytes_per_element
        rope_row_bytes = ahd * 2 * bpe
        ROPE_BASE = self.DRAM_ADDR_ROPE
        preamble_addr = self._decoder_preamble_addr

        if not hasattr(self, '_generated_tokens'):
            self._generated_tokens = []
        fpga_penalty   = bool(getattr(self, 'fpga_penalty', True))
        greedy_until   = int(getattr(self, 'greedy_until', 512))
        prefill_seq_start = len(self._generated_tokens)
        self.dma_to_accelerator_memory(self.PENALTY_BIAS_DRAM,
            torch.zeros(1, self.EMBEDDING_ELEMENTS, dtype=torch.bfloat16))

        global _SILENT_MODE
        max_seq_len = self.MAX_CONTEXT_SIZE

        # Live t/s counter (same as gemma3/smolvlm2)
        import shutil
        _use_status = sys.stdout.isatty()
        _timer = time.perf_counter()
        _decoder_token_cnt = 0
        def _status_setup():
            rows = shutil.get_terminal_size().lines
            sys.stdout.write(f"\033[1;{rows - 1}r")
            sys.stdout.write(f"\033[{rows - 1};1H")
            sys.stdout.flush()
        def _status_update():
            rows = shutil.get_terminal_size().lines
            elapsed = time.perf_counter() - _timer
            rate = _decoder_token_cnt / elapsed if elapsed > 0 else 0.0
            sys.stdout.write("\0337")
            sys.stdout.write(f"\033[{rows};1H\033[2K")
            sys.stdout.write(f" decoding… {_decoder_token_cnt} tokens  (pos {self.seq_len}/{max_seq_len})  "
                             f"{elapsed:.1f}s  {rate:.1f} tok/s")
            sys.stdout.write("\0338")
            sys.stdout.flush()
        def _status_teardown():
            rows = shutil.get_terminal_size().lines
            sys.stdout.write("\033[r")
            sys.stdout.write(f"\033[{rows};1H\033[2K")
            sys.stdout.flush()
        if _use_status:
            _status_setup()

        while self.seq_len < max_seq_len:
            _decoder_token_cnt += 1
            _SILENT_MODE = True
            self.seq_len += 1
            step_pos = self.seq_len - 1  # KV-fill index for THIS step
            aligned_seq_len = ((self.seq_len + 63) // 64) * 64
            rope_pos = step_pos + getattr(self, '_rope_offset', 0)
            cos_abs_addr = ROPE_BASE + rope_pos * rope_row_bytes

            embedding_tensor = self.get_embedding_for_tokens([token_id])
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)

            bias_host = torch.full((self.group_size, aligned_seq_len), -1e36, dtype=torch.bfloat16)
            bias_host[:, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_host)

            # Per-step preamble: prime runtime GPRs + JUMP_ABS into cached decoder program.
            self.clear_inst_id()
            self.start_capture()
            self.generate_instruction_add_set(self.gf_seq_len, step_pos)
            self.generate_instruction_add_set(self.gf_q_seq_len, self.seq_len)
            self.generate_instruction_add_set(self.gf_aligned_seq_len, aligned_seq_len)
            self.generate_instruction_add_set(self._gf_rope_cos_abs, ue_35bit_addr_shifter(cos_abs_addr))
            self.generate_instruction_jump_abs(ue_35bit_addr_shifter(decoder_program_addr))
            self.stop_capture()
            self.write_captured_instructions_to_dram(preamble_addr)
            self.clear_capture_buffer()

            # Refresh penalty bias past the greedy gate, BEFORE execute.
            if fpga_penalty and (len(self._generated_tokens) - prefill_seq_start) > greedy_until:
                self._write_penalty_bias(self._generated_tokens)

            self.start_execute_from_dram(preamble_addr)
            self.wait_queue(10.0)

            # Token selection: HW argmax already has penalty folded in on chip.
            token_id = self.get_arg_max_index()
            self._generated_tokens.append(token_id)
            token_char = self.tokenizer.decode([token_id])
            _SILENT_MODE = False
            if token_id in _qwen25_stop_tokens:
                if _use_status:
                    _status_teardown()
                print(f"\nStop token {token_id} reached.")
                break
            print(token_char, end="", flush=True)
            if _use_status:
                _status_update()
        else:
            if _use_status:
                _status_teardown()
        return self.seq_len


# ============================================================
#  LocateAnything-specific: vision prep + driver (engine inlined above)
# ============================================================
import argparse
from PIL import Image

LA_DIR = SCRIPT_DIR
REPO_ROOT = PROJECT_ROOT
# FPGA decoder engine config (inlined; formerly locateanything_3b_fpga_config.json).
# LM is dimensionally identical to Qwen2.5-VL-3B's; differences: vocab 152704 and
# PLAIN 1D rope (never load_mrope_for_prefill). Vision section only sizes (unused)
# DRAM buffers since MoonViT runs on GPU in stage 1.
_FPGA_CFG = {
    "fixed_isa_regs": {"TMP_REG": 1, "GF_SEQ_LEN_REG": 2, "GF_Q_SEQ_LEN_REG": 3, "GF_ALIGNED_SEQ_LEN_REG": 4},
    "file_info": {"layer_size": 40965632, "num_layers": 36, "head_dim": 256, "actual_head_dim": 128,
                  "num_kv_heads": 2, "hidden_size": 2048, "embedding_vocab": 152704, "group_size": 8,
                  "mlp_elements": 11008, "bytes_per_element": 2},
    "model": {"max_context_size": 512, "prefill_max_seq_len": 384, "end_of_turn_token_id": 151645,
              "rope_global_layers": []},
    "precision": {"lm": "if4"},
    "vision": {"hidden_size": 1280, "num_heads": 16, "head_dim": 80, "intermediate_size": 3420,
               "depth": 32, "patch_size": 14, "temporal_patch_size": 2, "spatial_merge_size": 2,
               "out_hidden_size": 2048, "window_size": 112, "fullatt_block_indexes": [7, 15, 23, 31],
               "image_size": 336, "num_patches": 576, "num_merged_tokens": 1024},
    "paths": {"hf_model_dir": "locateanything_3b_bin/LocateAnything-3B",
              "hf_model_repo": "nvidia/LocateAnything-3B"},
    "non_layer_regions": {"ROPE": {"offset": "0x86D98800", "size": 262144}},
    "special": {"rope": {"head_dim": 256, "actual_head_dim": 128, "num_positions": 512,
                         "theta": 1000000.0, "_mrope_section_unused": [16, 24, 24]},
                "rms_norm": {"gamma_offset": 0.0}},
}

# ---- Weight-bin generators (inlined; formerly _gen_lm_bin.py / _gen_connector_bin.py) ----
# Read HF safetensors directly and emit the wire format weight_init() consumes.
_GEN_MODEL_DIR = os.path.join(LA_DIR, "locateanything_3b_bin", "LocateAnything-3B")
_GEN_PAD_VOCAB = 152704   # 152681 -> 64*2386; pad rows are zeros (logit 0, harmless for argmax)
# if4-quantized weights (rest bf16). MUST match _LM_QUANT_LAYERS / weight_init loader.
_LM_GEN_QUANT_SUFFIXES = ('self_attn.q_proj.weight', 'self_attn.k_proj.weight',
                          'self_attn.o_proj.weight', 'mlp.gate_proj.weight',
                          'mlp.up_proj.weight', 'mlp.down_proj.weight')
_CONN_KEYS = ["mlp1.0.weight", "mlp1.0.bias", "mlp1.1.weight",
              "mlp1.1.bias", "mlp1.3.weight", "mlp1.3.bias"]


def _gen_pad_vocab(t):
    if t.shape[0] >= _GEN_PAD_VOCAB:
        return t
    pad = torch.zeros(_GEN_PAD_VOCAB - t.shape[0], t.shape[1], dtype=t.dtype)
    return torch.cat([t, pad], dim=0)


def gen_lm_bin(precision="if4"):
    """Generate the 'lm' region of params.bin: q/k/o/gate/up/down if4,
    v/norms/biases/embed bf16, tied lm_head appended."""
    import glob
    from safetensors.torch import load_file
    sd_full = {}
    for f in sorted(glob.glob(os.path.join(_GEN_MODEL_DIR, "*.safetensors"))):
        sd_full.update(load_file(f))
    items = {"language_model." + k[len("language_model.model."):]: v
             for k, v in sd_full.items() if k.startswith("language_model.model.")}
    assert items, "no language_model.model.* keys found"
    sub_manifest = {}
    region_buf = bytearray()
    count = 0
    for key, t in items.items():
        if key == "language_model.embed_tokens.weight":
            t = _gen_pad_vocab(t)
        if any(key.endswith(s) for s in _LM_GEN_QUANT_SUFFIXES):
            data, _ = _qs_pack(precision, t); raw = data.tobytes(); key = f'{key}.{precision}'
        else:
            raw = t.to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes()
        offset = len(region_buf); region_buf.extend(raw)
        sub_manifest[key] = {'offset': offset, 'size': len(raw)}; count += 1
    embed_w = _gen_pad_vocab(sd_full['language_model.model.embed_tokens.weight'].detach().to(torch.bfloat16))
    combined, _ = _qs_pack(precision, embed_w); combined_bytes = combined.tobytes()
    offset = len(region_buf); region_buf.extend(combined_bytes)
    sub_manifest[f'lm_head.weight.{precision}'] = {'offset': offset, 'size': len(combined_bytes)}
    params_bin_write_region(LA_DIR, "lm", bytes(region_buf), sub_manifest)
    print(f"LM region: {count + 1} tensors, {len(region_buf)/1048576:.1f} MB -> params.bin[lm]")


def gen_connector_bin():
    """Generate the 'connector' region of params.bin: mlp1 LayerNorm + 2 Linears, bf16."""
    import glob
    from safetensors.torch import load_file
    sd = {}
    for f in sorted(glob.glob(os.path.join(_GEN_MODEL_DIR, "*.safetensors"))):
        sd.update(load_file(f))
    missing = [k for k in _CONN_KEYS if k not in sd]
    assert not missing, f"connector keys missing from safetensors: {missing}"
    sub_manifest = {}
    region_buf = bytearray()
    for k in _CONN_KEYS:
        raw = sd[k].detach().to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes()
        offset = len(region_buf); region_buf.extend(raw)
        sub_manifest[k] = {"offset": offset, "size": len(raw), "shape": list(sd[k].shape)}
    params_bin_write_region(LA_DIR, "connector", bytes(region_buf), sub_manifest)
    print(f"connector region: {len(_CONN_KEYS)} tensors, {len(region_buf)/1048576:.1f} MB -> params.bin[connector]")
ENGINE_IMG_PLACEHOLDER = 151655   # what run_prefill() scans for; we remap onto it
LA_IMAGE_TOKEN = 151665           # <IMG_CONTEXT> in LocateAnything


def _remap_boxes_to_original(text, xform):
    """Map <box> coords from the letterboxed-square frame back to the original image's
    normalized [0,1000] frame. Identity when xform is None (no letterbox)."""
    if not xform:
        return text
    import re
    S = xform["side"]; px = xform["pad_x"]; py = xform["pad_y"]
    cw = xform["cw"]; ch = xform["ch"]

    def _mx(v):
        o = (int(v) / 1000.0 * S - px) / cw * 1000.0
        return max(0, min(1000, round(o)))

    def _my(v):
        o = (int(v) / 1000.0 * S - py) / ch * 1000.0
        return max(0, min(1000, round(o)))

    # 4-coord boxes first, then 2-coord points (won't match the 4-coord form).
    text = re.sub(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>",
                  lambda m: f"<box><{_mx(m[1])}><{_my(m[2])}><{_mx(m[3])}><{_my(m[4])}></box>",
                  text)
    text = re.sub(r"<box><(\d+)><(\d+)></box>",
                  lambda m: f"<box><{_mx(m[1])}><{_my(m[2])}></box>",
                  text)
    return text


def _gpu_vision(query, prompt_kind, image_path, device="cpu", compute_vit=True,
                keep_model=False, square=224, max_side=0):
    """Build the MoonViT model, process the image, tokenize. Optionally compute the
    GPU vit/connector reference. Returns a dict; the FPGA-vision path reuses the model
    + pixel_values to run MoonViT on the board instead.

    Resolution control (pick ONE):
      * square > 0  : letterbox (aspect-preserving, gray pad) to square×square BEFORE
        the patchifier. Every image -> the SAME patch grid -> the SAME compiled encoder
        program for ANY image. Predicted boxes come out in the padded frame, so they're
        remapped back to the original via `vis_xform` (returned).
      * max_side > 0: aspect-preserving cap on the longest side (no pad). Bounds cost but
        N still varies with aspect ratio (multiple cached programs). vis_xform=None.

    Keys: vit, conn_ref, prefill_seq, n_img, wh, model, pixel_values, grid_hw, cfg,
    vis_xform (letterbox geometry or None)."""
    import locateanything_3b_cpu_test as la
    cfg = json.load(open(os.path.join(LA_DIR, "locateanything_3b_config.json")))
    print("  loading weights ...", flush=True)
    model = la.LocateAnything(cfg).eval()
    la.load_weights(model, la.MODEL_DIR, device, torch.bfloat16)

    improc = la.ImageProcessor(merge_kernel_size=tuple(cfg["vision_config"]["merge_kernel_size"]))
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.width, img.height          # keep original for the box overlay
    vis_img = img
    vis_xform = None
    if square and square > 0:
        scale = square / float(max(orig_w, orig_h))
        cw, ch = max(1, round(orig_w * scale)), max(1, round(orig_h * scale))
        resized = img.resize((cw, ch), Image.BICUBIC)
        canvas = Image.new("RGB", (square, square), (127, 127, 127))
        pad_x, pad_y = (square - cw) // 2, (square - ch) // 2
        canvas.paste(resized, (pad_x, pad_y))
        vis_img = canvas
        vis_xform = dict(side=square, pad_x=pad_x, pad_y=pad_y, cw=cw, ch=ch)
        print(f"  vision letterbox {orig_w}x{orig_h} -> {square}x{square} "
              f"(content {cw}x{ch}, pad {pad_x},{pad_y})")
    elif max_side and max(orig_w, orig_h) > max_side:
        scale = max_side / float(max(orig_w, orig_h))
        new_wh = (max(1, round(orig_w * scale)), max(1, round(orig_h * scale)))
        vis_img = img.resize(new_wh, Image.BICUBIC)
        print(f"  vision input resized {orig_w}x{orig_h} -> {new_wh[0]}x{new_wh[1]} (max_side={max_side})")
    pv, grid_hw = improc(vis_img)
    mk = cfg["vision_config"]["merge_kernel_size"]
    n_img = (grid_hw[0] * grid_hw[1]) // (mk[0] * mk[1])

    vit = conn_ref = None
    if compute_vit:
        with torch.no_grad():
            vit = model.vision_features(pv.to(device, dtype=torch.bfloat16), grid_hw)  # [N,4608]
            conn_ref = model.mlp1(vit).to(torch.bfloat16).cpu()                        # [N,2048]
            vit = vit.to(torch.bfloat16).cpu()

    tok = _load_tokenizer(la.MODEL_DIR)
    text = la.build_prompt_text(la.PROMPTS[prompt_kind].format(q=query), n_img)
    ids = tok([text], return_tensors="pt").input_ids[0].tolist()
    # remap our <IMG_CONTEXT> onto the placeholder run_prefill replaces with vision
    ids = [ENGINE_IMG_PLACEHOLDER if t == LA_IMAGE_TOKEN else t for t in ids]

    if not keep_model:
        del model
        model = None
        if device == "cuda":
            torch.cuda.empty_cache()
    return dict(vit=vit, conn_ref=conn_ref, prefill_seq=tuple(ids), n_img=n_img,
                wh=(orig_w, orig_h), model=model, pixel_values=pv,
                grid_hw=grid_hw, cfg=cfg, vis_xform=vis_xform)


def _cpu_bf16_compare(args, hw_answer, prep):
    """Run the CPU bf16 reference (generate_slow) on the SAME image/query and
    return a comparison dict vs the HW decode. Reuses prep['conn_ref'] so the
    vision tower is NOT recomputed. Greedy decode on both sides => deterministic,
    so the honest metric is token-sequence agreement + per-box coordinate delta."""
    import locateanything_3b_cpu_test as la
    import torch, re
    device = "cpu"
    cfg = prep["cfg"]
    model = la.LocateAnything(cfg).eval()
    la.load_weights(model, la.MODEL_DIR, device, torch.bfloat16)
    tok = _load_tokenizer(la.MODEL_DIR)
    text = la.build_prompt_text(la.PROMPTS[args.prompt_kind].format(q=args.query), prep["n_img"])
    input_ids = tok([text], return_tensors="pt").input_ids
    conn = prep["conn_ref"]
    if conn is not None:
        conn = conn.to(device, dtype=torch.bfloat16)
    out_ids = la.generate_slow(model, tok, cfg, prep["pixel_values"], prep["grid_hw"],
                               input_ids, device, torch.bfloat16, conn=conn)
    cpu_answer = tok.decode(out_ids, skip_special_tokens=False)

    # token-level agreement (re-tokenize HW text so both are id sequences)
    hw_ids = tok(hw_answer, return_tensors="pt").input_ids[0].tolist()
    n = min(len(out_ids), len(hw_ids))
    agree = sum(1 for a, b in zip(out_ids[:n], hw_ids[:n]) if a == b)

    def _boxes(s):
        return [tuple(map(int, m)) for m in
                re.findall(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", s)]
    cb, hb = _boxes(cpu_answer), _boxes(hw_answer)
    nb = min(len(cb), len(hb))
    max_dl = max((max(abs(c - h) for c, h in zip(cb[i], hb[i])) for i in range(nb)),
                 default=0)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return dict(cpu_answer=cpu_answer.strip(), cpu_ids=out_ids, hw_ids=hw_ids,
                tok_agree=agree, tok_total=n, cpu_boxes=len(cb), hw_boxes=len(hb),
                box_match=nb, max_box_delta=max_dl,
                exact=(out_ids[:n] == hw_ids[:n] and len(out_ids) == len(hw_ids)))


# ============================================================
#  Quant comparison: bf16 vs if4 vs tq4 (TurboQuant), CPU reference
# ============================================================
# Effective-weight method: the WHT rotation is applied *inside* the quantizer
# (rotate -> Lloyd-Max quant -> dequant -> un-rotate), yielding a dense weight
# numerically identical to the folded HW version but needing NO model surgery.
# This isolates the quantization-quality question and is fully deterministic.
#   if4 scheme:  q,k,o,gate,up,down -> if4 ; v stays bf16     (matches the HW baseline)
#   tq4 scheme:  q,k,v,o,gate,up    -> tq4 ; down -> if4      (rotations cancel offline)
#   lm_head: if4 in if4-scheme, tq4 in tq4-scheme. embed stays bf16 in both.

def _qc_hadamard(n, device):
    """Normalized (orthonormal, symmetric) n×n Walsh-Hadamard matrix; n=2^k."""
    H = torch.tensor([[1.0]], device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / (n ** 0.5)


def _qc_if4_eff(W):
    """if4 round-trip -> reconstructed dense weight (same shape/dtype)."""
    from quant_lib import quantize_if4, dequantize_if4
    d, s = quantize_if4(W.detach().to(torch.bfloat16), block_size=64)
    return dequantize_if4(d, s, W.shape[0], W.shape[1], block_size=64).to(W.device).to(W.dtype)


_QC_CB = None  # cached (centroids, decision_boundaries) for the d=128 Lloyd-Max codebook


def _qc_tq4_eff(W, block=128):
    """TurboQuant round-trip: block-Hadamard rotate -> per-block-norm scale ->
    Lloyd-Max snap -> dequant -> un-rotate. Returns reconstructed dense weight."""
    from quant_lib import get_codebook
    global _QC_CB
    if _QC_CB is None:
        cb = get_codebook(block, 4)
        _QC_CB = (torch.tensor(cb["centroids"], dtype=torch.float32),
                  torch.tensor(cb["boundaries"][1:-1], dtype=torch.float32))
    out, K = W.shape
    assert K % block == 0, f"K={K} not divisible by Hadamard block {block}"
    dev = W.device
    cen, bnd = _QC_CB[0].to(dev), _QC_CB[1].to(dev)
    H = _qc_hadamard(block, dev)
    Wf = W.detach().float().reshape(out, K // block, block)
    Wr = Wf @ H                                   # rotate each block (H symmetric orthonormal)
    nrm = Wr.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    idx = torch.searchsorted(bnd, (Wr / nrm).contiguous())
    deq = cen[idx] * nrm                          # dequant in rotated space
    return (deq @ H).reshape(out, K).to(W.dtype)  # un-rotate


def _quant_compare(args, prep):
    """Run the CPU reference decoder bf16 / if4 / tq4 on the SAME vision output and
    return a metrics dict. Teacher-forced logits (aligned to the bf16 sequence) give
    clean logit MSE/cos/argmax; free greedy runs give exact-match + box deltas."""
    import locateanything_3b_cpu_test as la
    import torch, re
    import torch.nn.functional as F

    model = prep.get("model")
    cfg = prep["cfg"]
    if model is None:
        model = la.LocateAnything(cfg).eval()
        la.load_weights(model, la.MODEL_DIR, "cpu", torch.bfloat16)
    # Run the comparison on CPU: the GPU already holds the full 3B model (kept for
    # on-device vision) and won't fit the weight clones + forward (OOM). This is an
    # offline analysis, so correctness > speed.
    model = model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = torch.device("cpu")
    dtype = torch.bfloat16

    tok = _load_tokenizer(la.MODEL_DIR)
    text = la.build_prompt_text(la.PROMPTS[args.prompt_kind].format(q=args.query), prep["n_img"])
    input_ids = tok([text], return_tensors="pt").input_ids.to(device)
    n_prompt = input_ids.shape[1]

    with torch.no_grad():                                # vision/connector ONCE -> reused by all 3
        vit = model.vision_features(prep["pixel_values"].to(device, dtype), prep["grid_hw"])
        conn = model.mlp1(vit).to(dtype)

    LM = model.language_model
    layers = LM.model.layers
    n_layers = len(layers)

    def if4_targets():
        t = []
        for L in layers:
            for n in ("q_proj", "k_proj", "o_proj"):       t.append((getattr(L.self_attn, n), _qc_if4_eff))
            for n in ("gate_proj", "up_proj", "down_proj"): t.append((getattr(L.mlp, n), _qc_if4_eff))
        t.append((LM.lm_head, _qc_if4_eff))
        return t

    def tq4_targets():
        t = []
        for L in layers:
            for n in ("q_proj", "k_proj", "v_proj", "o_proj"): t.append((getattr(L.self_attn, n), _qc_tq4_eff))
            for n in ("gate_proj", "up_proj"):                 t.append((getattr(L.mlp, n), _qc_tq4_eff))
            t.append((L.mlp.down_proj, _qc_if4_eff))           # down stays if4 (silu⊙up blocks rotation)
        t.append((LM.lm_head, _qc_tq4_eff))
        return t

    def apply(targets):
        orig, sse, ssq = [], 0.0, 0.0
        with torch.no_grad():
            for mod, fn in targets:
                w = mod.weight; o = w.data.clone(); orig.append((w, o))
                nw = fn(o)
                sse += ((nw.float() - o.float()) ** 2).sum().item()
                ssq += (o.float() ** 2).sum().item()
                w.data.copy_(nw.to(w.dtype))
        return orig, (sse / ssq if ssq else 0.0)

    def restore(orig):
        with torch.no_grad():
            for w, o in orig:
                w.data.copy_(o)

    @torch.no_grad()
    def forced(full_ids):                                # teacher-forced logits for the gen positions
        ids = full_ids.to(device)
        emb = LM.model.embed_tokens(ids)
        sel = (ids[0] == model.image_token_index)
        emb[0, sel] = conn[: int(sel.sum())].to(emb.dtype)
        T = ids.shape[1]
        pos = torch.arange(T, device=device).unsqueeze(0)
        kv = [{"k": None, "v": None} for _ in range(n_layers)]
        h = LM.model(emb, pos, kv, causal=True)          # [1, T, H]
        hs = h[:, n_prompt - 1:T - 1, :]                 # only the gen-predicting positions
        return LM.lm_head(hs)[0].float()                 # [n_gen, V]

    @torch.no_grad()
    def freerun():
        return la.generate_slow(model, tok, cfg, prep["pixel_values"], prep["grid_hw"],
                                input_ids, device, dtype, conn=conn)

    def boxes(ids):
        s = tok.decode(ids, skip_special_tokens=False)
        return s, [tuple(map(int, m)) for m in
                   re.findall(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", s)]

    ref_ids = freerun()                                  # bf16 reference (free greedy)
    full = torch.cat([input_ids, torch.tensor([ref_ids], device=device)], dim=1)
    L_ref = forced(full)
    ref_s, ref_b = boxes(ref_ids)

    schemes = {}
    for name, build in (("if4", if4_targets), ("tq4", tq4_targets)):
        orig, wmse = apply(build())
        Lf = forced(full)
        own = freerun()
        restore(orig)
        own_s, own_b = boxes(own)
        d = Lf - L_ref
        nb = min(len(ref_b), len(own_b))
        schemes[name] = dict(
            wmse=wmse,
            max_abs=d.abs().max().item(),
            mse=(d ** 2).mean().item(),
            cos=F.cosine_similarity(Lf.flatten(), L_ref.flatten(), dim=0).item(),
            arg=(Lf.argmax(-1) == L_ref.argmax(-1)).float().mean().item() * 100.0,
            exact=(own == ref_ids),
            nboxes=len(own_b), box_match=nb,
            box_dmax=max((max(abs(a - b) for a, b in zip(ref_b[i], own_b[i])) for i in range(nb)), default=0),
            answer=own_s.strip(),
        )
    return dict(ref_boxes=len(ref_b), ref_answer=ref_s.strip(), n_gen=len(ref_ids), schemes=schemes)


def _run_decode_only():
    """Skip vision/connector; text-only prefill + decode to isolate the LM + decoder path."""
    print("=== Decode-only mode (no vision) ===")
    if params_bin_load_region(LA_DIR, "lm") is None:
        print("  LM region missing from params.bin -> generating ...")
        hf_dir = os.path.join(LA_DIR, "locateanything_3b_bin", "LocateAnything-3B")
        if not os.path.exists(os.path.join(hf_dir, "config.json")):
            print(f"  Downloading HF model nvidia/LocateAnything-3B ...")
            snapshot_download(repo_id="nvidia/LocateAnything-3B", local_dir=hf_dir, local_dir_use_symlinks=False)
        gen_lm_bin()

    def _la_load_config(script_dir):
        import copy
        cfg = copy.deepcopy(_FPGA_CFG)
        wd = {"LAYER_WEIGHT_SIZE": cfg["file_info"]["layer_size"]}
        for k, r in cfg.get("regions", {}).items():
            wd[k] = _parse_offset(r["offset"]); wd[f"{k}_SIZE"] = r["size"]
        for k, r in cfg.get("non_layer_regions", {}).items():
            wd[k] = _parse_offset(r["offset"]); wd[f"{k}_SIZE"] = r["size"]
        cfg["_weight_defs"] = wd
        return cfg

    class LADecoderTextOnly(Qwen25VL3B_UnifiedEngine):
        def _load_vision_weights(self, *a, **k):
            pass

    global _load_config
    _load_config = _la_load_config

    ue = LADecoderTextOnly(script_dir=LA_DIR)
    ue._rope_offset = 0

    # Use tokenizer from HF dir
    tok_dir = os.path.join(LA_DIR, "locateanything_3b_bin", "LocateAnything-3B")
    tokenizer = _load_tokenizer(tok_dir)
    ue.tokenizer = tokenizer

    prompt = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
    token_ids = tokenizer.encode(prompt)
    print(f"  Prompt: {prompt!r}")
    print(f"  Token ids ({len(token_ids)}): {token_ids}")

    print("\n--- Prefill compile ---")
    t0 = time.perf_counter()
    pa, pre, gflops = ue.compile_prefill()
    print(f"  {time.perf_counter()-t0:.2f}s")

    print(f"--- Prefill run ({len(token_ids)} tokens) ---")
    t0 = time.perf_counter()
    ue.run_prefill(pa, pre, prefill_seq=token_ids, gflops=gflops, has_image=False)
    t_pre = time.perf_counter() - t0
    print(f"  {t_pre:.2f}s")

    print("\n--- Decode compile ---")
    t0 = time.perf_counter()
    da, gpt = ue.compile_decoder()
    t_dc = time.perf_counter() - t0
    _dec_cached = programs_bin_load(LA_DIR, "decoder")
    _dec_kb = (len(_dec_cached[0]) / 1024) if _dec_cached else 0
    print(f"  {t_dc:.2f}s  (decoder size: {_dec_kb:.0f} KB)")

    print("--- Decode run ---")
    ue._generated_tokens = list(token_ids)
    t0 = time.perf_counter()
    tok_cnt = ue.run_decoder(da, token_id=token_ids[-1], gflops=gpt)
    t_dec = time.perf_counter() - t0
    n_new = (tok_cnt - len(token_ids)) if isinstance(tok_cnt, int) else 0
    print(f"\n  {t_dec:.2f}s  {n_new} tokens  {n_new/max(t_dec,1e-9):.2f} tok/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=str, default="xdma0",
                    help="DMA device name for non-Efinix profiles (e.g., xdma0, xdma1). Efinix uses /dev/pcie_dma0_* from its profile.")
    ap.add_argument("--device", type=str, default="kintex7",
                    help="FPGA board / bitstream profile. Use efinix for the Efinix 4GB DMA profile.")
    ap.add_argument("--cycle", type=float, default=None,
                    help="Clock cycle time in ns. Default: from --device, or existing runtime default.")
    ap.add_argument("--image", default=os.path.join(REPO_ROOT, "test_samples", "people.jpg"))
    ap.add_argument("--prompt-kind", default="detect",
                    choices=["detect", "ground_multi", "ground_one", "point"])
    ap.add_argument("--query", default="face")
    ap.add_argument("--square", type=int, default=224, choices=[0, 224, 384],
                    help="letterbox every image to square×square (aspect-preserving, gray pad) "
                         "before patchifying -> ONE compiled encoder program for any image. "
                         "384 (~196 tok, higher quality) or 224 (~64 tok, fastest). 0 = disable "
                         "(use --max-side instead). Boxes are remapped back to the original frame.")
    ap.add_argument("--max-side", type=int, default=0,
                    help="only when --square 0: aspect-preserving cap on the longest dim "
                         "(bounds cost but N varies with aspect ratio -> multiple cached programs).")
    ap.add_argument("--decode-only", action="store_true",
                    help="skip vision/connector entirely; prefill+decode a short text prompt to isolate the LM decode path")
    ap.add_argument("--check-connector", action="store_true",
                    help="diff FPGA connector output vs the bit-exact GPU connector")
    ap.add_argument("--check-vision", action="store_true",
                    help="parity-check FPGA vit[N,4608] vs the CPU reference")
    ap.add_argument("--vision-precision", default="if4", choices=["if4", "bf16"],
                    help="MoonViT encoder weight precision (if4 fits the params window; "
                         "bf16 ~0.98GB may overflow)")
    ap.add_argument("--vision-debug", action="store_true",
                    help="run MoonViT op-by-op (separate execute per op) with a NaN/Inf "
                         "check after each stage; halts at the first bad stage")
    ap.add_argument("--stop-after", default=None,
                    help="with --vision-debug: force a halt after this stage label "
                         "(e.g. L0.flash, L0.fc0_gelu, final_norm)")
    ap.add_argument("--vision-no-mask", action="store_true",
                    help="with --vision-debug: run flash WITHOUT the pad-key mask bias "
                         "(isolation: output will be wrong but should be NaN-free)")
    ap.add_argument("--silent", action="store_true",
                    help="suppress compile/engine chatter (URAM usage, captured-instruction "
                         "dumps, etc.); timing summary + output still print")
    ap.add_argument("--compare", action="store_true",
                    help="after decode, run the CPU bf16 reference on the same image/query "
                         "and print a token/box agreement report under the timing summary")
    ap.add_argument("--quant-compare", action="store_true",
                    help="run the CPU reference decoder bf16 vs if4 vs tq4 on the same vision "
                         "output and print a quant-fidelity comparison at the very bottom")
    args = ap.parse_args()

    profile = configure_device(args.device, dma_device=args.dev)
    global DMA_DEVICE_H2C
    DMA_DEVICE_H2C = user_dma_core.DMA_DEVICE_H2C
    axi_width_bits = profile.get("axi_data_width_bits")
    if axi_width_bits is not None:
        os.environ["UE_AXI_DATA_WIDTH_BITS"] = str(axi_width_bits)
        user_dma_core.UE_AXI_DATA_WIDTH_BITS = axi_width_bits
    clock = args.cycle if args.cycle is not None else profile.get("clock_period_ns")
    if clock is not None:
        user_dma_core.CLOCK_CYCLE_TIME_NS = clock
        user_dma_core.UE_PEAK_GFLOPS = 0.128 / clock
    effective_dma = "pcie_dma0" if profile["device"] == "efinix" else args.dev
    print(f"FPGA profile: device={profile['device']}, clock={user_dma_core.CLOCK_CYCLE_TIME_NS:.4f} ns, UE_AXI_DATA_WIDTH_BITS={user_dma_core.UE_AXI_DATA_WIDTH_BITS}")
    print(f"Using DMA device: {effective_dma}")
    print(f"  H2C: {user_dma_core.DMA_DEVICE_H2C}")
    print(f"  C2H: {user_dma_core.DMA_DEVICE_C2H}")
    print(f"  USER: {user_dma_core.DMA_DEVICE_USER}")
    print(f"  BASE: 0x{user_dma_core.UE_0_BASE_ADDR:08x}")
    print(
        f"  DRAM: start=0x{user_dma_core.DRAM_START_ADDR:08x}, "
        f"act=0x{user_dma_core.DRAM_ACTIVATION_ADDR:08x}, "
        f"inst=0x{user_dma_core.DRAM_INSTRUCTION_ADDR:08x}, "
        f"end=0x{user_dma_core.DRAM_END_ADDR:08x}"
    )

    # ==================================================================
    # DEBUG CONFIG (no flags needed). Flip these in code.
    # image / query / prompt_kind now come from the CLI (--image/--query/--prompt-kind).
    # ==================================================================
    PROBE_PREFILL_NAN = False        # True: layer-by-layer prefill sweep. False: fast prefill+decode.
    PROBE_DECODE_NAN  = False        # True: after prefill, sweep decode layers for the first NaN.
    PROBE_ADDR_MAP    = False        # fast: dump per-layer weight addrs + DRAM readback, then exit (no prefill)
    FORCE_DEBUG_LOGITS = True        # dump final_norm + lm_head NaN breakdown on first decode step
    # ==================================================================

    global _SILENT_MODE
    _SILENT_MODE = args.silent

    if args.decode_only:
        _run_decode_only()
        return

    # ---- 1. Vision prep (build model, process image, tokenize) ----
    fpga_vision = True                          # MoonViT always runs on the FPGA board
    want_cpu_vit = args.check_vision            # CPU reference only when parity-checking
    print("=== Vision: FPGA MoonViT -> FPGA connector + decoder ===")
    t0 = time.time()
    prep = _gpu_vision(args.query, args.prompt_kind, args.image,
                       compute_vit=want_cpu_vit, keep_model=fpga_vision,
                       square=args.square, max_side=args.max_side)
    vit = prep["vit"]; conn_ref = prep["conn_ref"]; prefill_seq = prep["prefill_seq"]
    n_img = prep["n_img"]; (W, H) = prep["wh"]; vis_cfg = prep["cfg"]
    print(f"  image_tokens={n_img}  prefill_len={len(prefill_seq)}  ({time.time()-t0:.1f}s)")

    # ---- 2. Build FPGA decoder ----
    print("\n=== Building FPGA decoder (loads LM bin to DRAM) ===")
    if params_bin_load_region(LA_DIR, "lm") is None:
        print("  LM region missing from params.bin -> generating (one-time, ~45s) ...")
        gen_lm_bin()

    def _la_load_config(script_dir):
        import copy
        cfg = copy.deepcopy(_FPGA_CFG)
        wd = {"LAYER_WEIGHT_SIZE": cfg["file_info"]["layer_size"]}
        for k, r in cfg.get("regions", {}).items():
            wd[k] = _parse_offset(r["offset"]); wd[f"{k}_SIZE"] = r["size"]
        for k, r in cfg.get("non_layer_regions", {}).items():
            wd[k] = _parse_offset(r["offset"]); wd[f"{k}_SIZE"] = r["size"]
        cfg["_weight_defs"] = wd
        return cfg
    global _load_config
    _load_config = _la_load_config

    class LADecoder(Qwen25VL3B_UnifiedEngine):
        def _load_vision_weights(self, *a, **k):
            print("  [stage2] MoonViT loaded separately -> skipping LM vision-weight load")

        def load_connector(self):
            """Load the connector (mlp1) bf16 weights into params DRAM
            from params.bin region 'connector'."""
            conn_cache = params_bin_load_region(LA_DIR, "connector")
            if conn_cache is None:
                print("  connector region missing in params.bin -> generating ...")
                gen_connector_bin()
                conn_cache = params_bin_load_region(LA_DIR, "connector")
                assert conn_cache is not None, "gen_connector_bin did not produce 'connector' region"
            man = conn_cache["__entries__"]

            def _ld(key):
                m = man[key]
                arr = np.frombuffer(conn_cache[key], dtype=np.uint16).copy()
                t = torch.from_numpy(arr).view(torch.bfloat16).reshape(m["shape"])
                return store_weight(self, t)

            self.conn_ln_g = _ld("mlp1.0.weight")   # LayerNorm gamma [4608]
            self.conn_ln_b = _ld("mlp1.0.bias")     # LayerNorm beta  [4608]
            self.conn_w0 = _ld("mlp1.1.weight")     # Linear0 [2048,4608]
            self.conn_b0 = _ld("mlp1.1.bias")       # Linear0 bias [2048]
            self.conn_w2 = _ld("mlp1.3.weight")     # Linear1 [2048,2048]
            self.conn_b2 = _ld("mlp1.3.bias")       # Linear1 bias [2048]
            self._conn_in, self._conn_out = man["mlp1.1.weight"]["shape"][1], man["mlp1.3.weight"]["shape"][0]
            self._conn_hidden = man["mlp1.1.weight"]["shape"][0]
            print(f"  connector loaded: LayerNorm({self._conn_in}) -> "
                  f"Linear({self._conn_in},{self._conn_hidden}) -> GELU -> "
                  f"Linear({self._conn_hidden},{self._conn_out})")

        def compile_connector(self, N):
            """Compile-or-load the connector program for N vision tokens.

            Allocates a permanent program slot keyed by N. Called from run_connector
            on first use so N is known. Prefill/decoder compile after this, which
            preserves program-arena ordering.
            """
            if getattr(self, "_connector_program_addr", None) is not None and self._connector_compiled_N == N:
                return  # already compiled for this N

            Cin = self._conn_in
            IN  = self.VIS_MERGED_DRAM
            LN  = self.VIS_MERGER_INTER_DRAM

            conn_key = f"connector_N{N}"
            cache_hit = False
            prog_bytes = None
            _cached = programs_bin_load(self.script_dir, conn_key)
            if _cached is not None:
                prog_bytes, meta = _cached
                if meta.get("num_tokens") == N and meta.get("conn_in") == Cin:
                    cache_hit = True

            if cache_hit:
                self._connector_program_addr = self.get_program_dram_addr()
                self.dma_write(DMA_DEVICE_H2C, self._connector_program_addr, prog_bytes, len(prog_bytes))
                self.allocate_program_dram(len(prog_bytes))
                self._connector_compiled_N = N
                print(f"  Connector loaded from cache ({len(prog_bytes)} bytes) at 0x{self._connector_program_addr:X}")
            else:
                _original_print(f"  Compiling connector program (N={N})...")
                self.clear_inst_id()
                self.start_capture()
                self.layer_norm_core_dram(M=N, N=Cin, A_DRAM_ADDR=IN, OUTPUT_DRAM_ADDR=LN,
                                          GAMMA_DRAM_ADDR=self.conn_ln_g, BETA_DRAM_ADDR=self.conn_ln_b)
                self.matmat_mul_core(M=N, K=Cin, N=self._conn_hidden,
                                     A_DRAM_ADDR=LN, B_DRAM_ADDR=self.conn_w0,
                                     OUTPUT_DRAM_ADDR=IN,
                                     C_DRAM_ADDR=self.conn_b0, bias_mode="broadcast_N",
                                     gelu_enable=True)
                self.matmat_mul_core(M=N, K=self._conn_hidden, N=self._conn_out,
                                     A_DRAM_ADDR=IN, B_DRAM_ADDR=self.conn_w2,
                                     OUTPUT_DRAM_ADDR=self.VIS_ENCODER_OUT_DRAM,
                                     C_DRAM_ADDR=self.conn_b2, bias_mode="broadcast_N")
                self.stop_capture()
                self.generate_instruction_halt()

                self._connector_program_addr = self.get_program_dram_addr()
                prog_bytes = bytearray()
                for inst in self.capture_buffer:
                    prog_bytes.extend(inst.get_bytes())
                self.write_captured_instructions_to_dram(self._connector_program_addr)
                self.allocate_program_dram(self.get_capture_instruction_size_bytes())
                self.clear_capture_buffer()
                self._connector_compiled_N = N

                programs_bin_store(self.script_dir, conn_key, prog_bytes,
                                   meta={"num_tokens": N, "conn_in": Cin, "program_size": len(prog_bytes)})
                _original_print(f"  Connector compiled ({len(prog_bytes)} bytes) at 0x{self._connector_program_addr:X}")

        def run_connector(self, vit):
            """Compile (once per N) and execute the connector program."""
            N, Cin = vit.shape
            assert Cin == self._conn_in, f"connector in {Cin} != {self._conn_in}"
            self.compile_connector(N)
            self.dma_to_accelerator_memory(self.VIS_MERGED_DRAM, vit.contiguous().flatten())
            self.start_execute_from_dram(self._connector_program_addr)
            self.wait_queue(120.0)
            self._vis_num_tokens = N

    ue = LADecoder(script_dir=LA_DIR)
    ue._debug_logits = bool(os.environ.get("LA_DEBUG_LOGITS")) or FORCE_DEBUG_LOGITS

    if PROBE_ADDR_MAP:
        # Fast: dump per-layer weight addresses + DRAM read-back (source ln1_gamma absmax=2.2),
        # no prefill/decode needed. Detects unwritten/garbage upper-region weights.
        V = ue.vector_length
        _original_print("\n=== ADDR MAP: ln1_gamma per layer (source absmax=2.20) ===")
        _original_print(f"  params_base=0x{ue._params_dram_base:011X}  params_end=0x{ue.get_params_dram_addr():011X}  "
                        f"usage={ue.get_params_dram_usage()/1e9:.3f}GB")
        _original_print(f"  tensor LAYER0_INPUT=0x{ue.LAYER0_INPUT_DRAM:011X}  LAYER0_OUTPUT=0x{ue.LAYER0_OUTPUT_DRAM:011X}")
        for li in range(ue.LAYER_SIZE):
            a = ue.lm_layer_addrs[li]['ln1_gamma']
            gv = ue.dma_from_accelerator_memory(a, (V,)).cpu().float()
            nanc = int(torch.isnan(gv).sum())
            over = ">4GB-32b" if a > 0xFFFFFFFF else ""
            bad = "  <-- BAD" if (nanc or gv.abs().max() > 1e3) else ""
            _original_print(f"    layer {li:2d}: 0x{a:011X} {over:9s} absmax={gv.abs().max():.3e} nan={nanc}{bad}")
        return

    ue.load_connector()

    # ---- 2b. FPGA MoonViT encoder (optional): produces vit[N_merged,4608] in DRAM ----
    t_moonvit_compile = t_moonvit_run = 0.0
    if fpga_vision:
        print("\n=== FPGA MoonViT encoder (stitched op flow) ===")
        model = prep["model"]
        N = prep["grid_hw"][0] * prep["grid_hw"][1]
        moonvit_load_weights(ue, model, vis_cfg, precision=args.vision_precision)
        moonvit_setup_dram(ue, N, vis_cfg)
        moonvit_prepare_input(ue, model, prep["pixel_values"], prep["grid_hw"], vis_cfg)
        if args.vision_debug:
            # op-by-op execute + NaN check; halts at the first bad stage (or --stop-after)
            moonvit_run_staged(ue, vis_cfg, prep["grid_hw"], stop_after=args.stop_after,
                                  mask_pad=not args.vision_no_mask)
        else:
            _t = time.perf_counter()
            prog = moonvit_compile_encoder(ue, vis_cfg, prep["grid_hw"])
            t_moonvit_compile = time.perf_counter() - _t
            print(f"  MoonViT encoder compile: {t_moonvit_compile:.2f}s (N={ue.MV_N})")
            _t = time.perf_counter()
            moonvit_run_encoder(ue, prog)
            t_moonvit_run = time.perf_counter() - _t
            mv_us = getattr(ue, '_last_moonvit_latency_us', 0)
            print(f"  MoonViT encoder run:     {t_moonvit_run:.2f}s wall  ({mv_us/1e3:.1f} ms HW)")
        # read vit[N_merged,4608] back from DRAM (also feeds the connector + optional parity)
        vit = ue.dma_from_accelerator_memory(ue.VIS_MERGED_DRAM, (ue.MV_N_MERGED, 4608)).cpu()
        if args.check_vision and prep["vit"] is not None:
            ref = prep["vit"].float()
            f = vit.float()
            diff = (f - ref).abs()
            denom = ref.abs().mean().clamp_min(1e-6)
            cos = torch.nn.functional.cosine_similarity(f.flatten(), ref.flatten(), dim=0)
            print(f"  [check-vision] FPGA vit vs CPU ref: max|Δ|={diff.max():.4f} "
                  f"mean|Δ|={diff.mean():.5f} rel={diff.mean()/denom:.4f} cos={cos:.5f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- 3. FPGA connector: vit[N,4608] -> VIS_ENCODER_OUT_DRAM[N,2048] ----
    print("\n=== FPGA connector (hw exec) ===")
    timer = time.perf_counter()
    ue.run_connector(vit)
    t_connector = time.perf_counter() - timer
    _original_print(f"  FPGA connector -> {n_img} vision tokens @ VIS_ENCODER_OUT_DRAM 0x{ue.VIS_ENCODER_OUT_DRAM:X}  ({t_connector:.2f}s)")

    # ---- DRAM layout guard: params (LM+conn+vision) must not overrun scratch,
    #      scratch must not overrun the program arena (the on-device-vision NaN
    #      bug). All addrs are now known (weights + all VIS_* buffers allocated).
    _pe = ue.get_params_dram_addr(); _te = ue._tensor_dram_addr
    _tb = ue._tensor_dram_base; _pb = ue._program_dram_base
    _original_print(f"  [dram] params_end=0x{_pe:09X} ({_pe/2**30:.2f}GB)  "
                    f"tensor_base=0x{_tb:09X}  tensor_end=0x{_te:09X} ({(_te-_tb)/2**20:.0f}MB)  "
                    f"program_base=0x{_pb:09X}  ceil=0x100000000")
    assert _pe <= _tb, f"OVERLAP: params_end 0x{_pe:X} overruns tensor_base 0x{_tb:X} (raise tensor_dram_base)"
    assert _te <= _pb, f"OVERLAP: tensor_end 0x{_te:X} overruns program_base 0x{_pb:X} (raise program_dram_base)"

    if args.check_connector:
        conn_fpga = ue.dma_from_accelerator_memory(
            ue.VIS_ENCODER_OUT_DRAM, (n_img, ue._conn_out)).cpu().float()
        ref = conn_ref.float()
        diff = (conn_fpga - ref).abs()
        denom = ref.abs().mean().clamp_min(1e-6)
        print(f"  [check] connector FPGA vs GPU: max|Δ|={diff.max():.4f}  "
              f"mean|Δ|={diff.mean():.5f}  rel={diff.mean()/denom:.4f}")

    # ---- 3b. NaN localizer (hardcoded debug path) ----
    if PROBE_PREFILL_NAN:
        ue._vis_num_tokens = n_img
        found = ue.prefill_nan_probe(prefill_seq)
        if found:
            return
        # Prefill clean -> fall through to decode with readout NaN dump enabled.
        ue._debug_logits = True

    # ---- 4. Prefill + decode on FPGA ----
    _original_print("\n=== Prefill ===")
    _original_print("--- Prefill compile ---")
    timer = time.perf_counter()
    pa, pre, gflops = ue.compile_prefill()
    t_prefill_compile = time.perf_counter() - timer
    _original_print(f"  {t_prefill_compile:.2f}s")

    _original_print(f"--- Prefill run ({len(prefill_seq)} tokens) ---")
    timer = time.perf_counter()
    ue.run_prefill(pa, pre, prefill_seq=prefill_seq, gflops=gflops, has_image=True)
    t_prefill_run = time.perf_counter() - timer
    prefill_hw_gflops = getattr(ue, '_last_hw_gflops', 0)
    _original_print(f"  {t_prefill_run:.2f}s  ({prefill_hw_gflops:.2f} GFLOPS)")

    if PROBE_DECODE_NAN:
        ue._rope_offset = 0  # plain 1D rope (decode positions sequential)
        ue.decode_nan_probe(prefill_seq[-1])
        return

    _original_print("\n=== Decode ===")
    _original_print("--- Decode compile ---")
    timer = time.perf_counter()
    da, gpt = ue.compile_decoder()
    t_decode_compile = time.perf_counter() - timer
    _original_print(f"  {t_decode_compile:.2f}s")
    ue._rope_offset = 0  # plain 1D rope, sequential positions (no mRoPE)
    # Tee decode stdout: stream to the terminal live AND capture it for the overlay.
    import io, sys, contextlib

    class _Tee(io.TextIOBase):
        def __init__(self, *streams):
            self.streams = streams
        def write(self, s):
            for st in self.streams:
                st.write(s); st.flush()
            return len(s)

    buf = io.StringIO()
    real = sys.stdout
    ue._generated_tokens = list(prefill_seq)   # seed on-FPGA penalty window with prompt
    _original_print("--- Decode run ---")
    timer = time.perf_counter()
    with contextlib.redirect_stdout(_Tee(real, buf)):
        tok_cnt = ue.run_decoder(da, token_id=prefill_seq[-1], gflops=gpt)
    t_decode_run = time.perf_counter() - timer
    answer_raw = buf.getvalue()                       # padded-frame coords (for --compare)
    # Map boxes out of the letterboxed square back to the original image frame.
    answer = _remap_boxes_to_original(answer_raw, prep.get("vis_xform"))
    n_new = (tok_cnt - len(prefill_seq)) if isinstance(tok_cnt, int) else 0
    tok_s = max(n_new, 0) / max(t_decode_run, 1e-9)
    _original_print(f"\n  {t_decode_run:.2f}s ({n_new} tokens, {t_decode_run/max(n_new,1):.3f}s/tok, "
          f"{tok_s:.2f} tok/s)")
    _original_print("\nOUTPUT:", answer.strip())

    import re
    n_boxes = len(re.findall(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer))
    boxes_s = n_boxes / max(t_decode_run, 1e-9)

    # ---- Timing summary ----
    total = (t_moonvit_compile + t_moonvit_run + t_connector + t_prefill_compile
             + t_prefill_run + t_decode_compile + t_decode_run)
    _original_print(f"\n{'='*60}\n  Timing summary\n{'='*60}")
    _original_print(f"  MoonViT compile:          {t_moonvit_compile:.2f}s")
    _original_print(f"  MoonViT run:              {t_moonvit_run:.2f}s")
    _original_print(f"  Connector (compile+exec): {t_connector:.2f}s")
    _original_print(f"  Prefill compile:          {t_prefill_compile:.2f}s")
    _original_print(f"  Prefill run:              {t_prefill_run:.2f}s  ({len(prefill_seq)} tokens)")
    _original_print(f"  Decode compile:           {t_decode_compile:.2f}s")
    _original_print(f"  Decode run:               {t_decode_run:.2f}s  ({n_new} tokens)")
    _original_print(f"  ──────────────────────────")
    _original_print(f"  Total (FPGA):             {total:.2f}s")
    _original_print(f"  Throughput:               {tok_s:.2f} tok/s, {boxes_s:.2f} boxes/s "
          f"({n_boxes} boxes in {t_decode_run:.2f}s decode)")

    # ---- CPU bf16 comparison report ----
    if args.compare:
        _original_print(f"\n{'='*60}\n  CPU bf16 comparison\n{'='*60}")
        try:
            c = _cpu_bf16_compare(args, answer_raw, prep)
            pct = 100.0 * c["tok_agree"] / max(c["tok_total"], 1)
            _original_print(f"  CPU output:   {c['cpu_answer']}")
            _original_print(f"  Token agree:  {c['tok_agree']}/{c['tok_total']} ({pct:.1f}%)"
                            f"   exact={'yes' if c['exact'] else 'no'}")
            _original_print(f"  Boxes:        HW={c['hw_boxes']}  CPU={c['cpu_boxes']}  "
                            f"matched={c['box_match']}  max|Δcoord|={c['max_box_delta']}")
        except Exception as e:
            import traceback
            _original_print(f"  [compare failed] {e}")
            traceback.print_exc()

    # ---- 5. Draw overlay (reuse cpu_test renderer) ----
    import locateanything_3b_cpu_test as la
    stem = os.path.splitext(os.path.basename(args.image))[0]
    out_path = os.path.join(LA_DIR, f"{stem}_boxes_fpga.jpg")
    path, nb, npt = la.draw_overlay(args.image, answer, out_path)
    print(f"\n{n_boxes} boxes")
    print(f"overlay saved -> {path}  ({nb} boxes, {npt} points)")

    prefill_tok_s = len(prefill_seq) / max(t_prefill_run, 1e-9)
    _original_print("TEST_RESULT:" + json.dumps({
        "decoded_text":        answer.strip(),
        "decoded_tokens":      n_new,
        "prefill_tokens":      len(prefill_seq),
        "prefill_speed_tok_s": round(prefill_tok_s, 2),
        "decode_speed_tok_s":  round(tok_s, 2),
    }))

    # ---- Quant fidelity comparison: bf16 vs if4 vs tq4 (CPU reference) ----
    if args.quant_compare:
        _original_print(f"\n{'='*72}\n  Quant comparison  (CPU reference, same vision output)\n{'='*72}")
        try:
            qc = _quant_compare(args, prep)
            s = qc["schemes"]
            _original_print(f"  bf16 reference: {qc['n_gen']} gen tokens, {qc['ref_boxes']} boxes")
            _original_print(f"  {'scheme':<6}{'wMSE':>10}{'logitMSE':>11}{'logit|Δ|max':>13}"
                            f"{'cos':>9}{'argmax%':>9}{'exact':>7}{'boxes':>7}{'boxΔmax':>9}")
            for name in ("if4", "tq4"):
                m = s[name]
                _original_print(f"  {name:<6}{m['wmse']:>10.2e}{m['mse']:>11.3e}{m['max_abs']:>13.3f}"
                                f"{m['cos']:>9.5f}{m['arg']:>9.1f}{('yes' if m['exact'] else 'no'):>7}"
                                f"{m['nboxes']:>7}{m['box_dmax']:>9}")
            # verdict: closer-to-bf16 = lower logit MSE, higher cos, higher argmax-agreement
            closer = min(("if4", "tq4"), key=lambda n: s[n]["mse"])
            _original_print(f"  ── closer to bf16 (lowest logit MSE): {closer.upper()}  "
                            f"[if4 cos={s['if4']['cos']:.5f} argmax={s['if4']['arg']:.1f}% | "
                            f"tq4 cos={s['tq4']['cos']:.5f} argmax={s['tq4']['arg']:.1f}%]")
        except Exception as e:
            import traceback
            _original_print(f"  [quant-compare failed] {e}")
            traceback.print_exc()



# ============================================================
#  MoonViT-SO-400M FPGA vision encoder (inlined; was moonvit_encoder.py)
# ============================================================
VD_PAD = 128  # padded head_dim (72 -> 128), must be a multiple of 64


def _vis_dims(cfg):
    v = cfg["vision_config"]
    return dict(
        depth=v["num_hidden_layers"],          # 27
        VH=v["hidden_size"],                    # 1152
        VN=v["num_attention_heads"],            # 16
        VD=v["head_dim"],                       # 72
        VI=v["intermediate_size"],              # 4304
        theta=v["rope_2d_theta"],               # 10000
        merge=v["merge_kernel_size"][0],        # 2
    )


# =====================================================================================
# Host-side weight preparation (padding + evens-odds reorder + q score-scale)
# =====================================================================================

def _qk_pad_reorder(W, b, VN, VD, scale):
    """[VN*VD, VH] q or k weight -> [VN*VD_PAD, VH] in the pad-128 rotate_half layout.

    rotate_half pairs lane i with lane i+64 (half of VD_PAD=128). The rope cos/sin table
    has the real angles at lanes [0:36] (lo) and [64:100] (hi). So the EVEN head_dim
    indices (the real part of each interleaved pair) go to lanes 0..35 and the ODD
    indices (imag part) go to lanes 64..99 — NOT contiguous 0..71. Pad lanes (36..63,
    100..127) stay zero. `scale` folds the sqrt(128/72) flash-scale correction into q."""
    VH = W.shape[1]
    half = VD // 2                       # 36
    hp = VD_PAD // 2                     # 64
    W = W.reshape(VN, VD, VH)
    b = b.reshape(VN, VD)
    W_even = W[:, 0:VD:2, :] * scale      # [VN, 36, VH] -> lo half lanes 0..35
    W_odd = W[:, 1:VD:2, :] * scale       # [VN, 36, VH] -> hi half lanes 64..99
    b_even = b[:, 0:VD:2] * scale
    b_odd = b[:, 1:VD:2] * scale
    Wp = torch.zeros(VN, VD_PAD, VH, dtype=torch.bfloat16)
    bp = torch.zeros(VN, VD_PAD, dtype=torch.bfloat16)
    Wp[:, :half, :] = W_even.to(torch.bfloat16)
    Wp[:, hp:hp + half, :] = W_odd.to(torch.bfloat16)
    bp[:, :half] = b_even.to(torch.bfloat16)
    bp[:, hp:hp + half] = b_odd.to(torch.bfloat16)
    return Wp.reshape(VN * VD_PAD, VH).contiguous(), bp.reshape(-1).contiguous()


def _o_pad(Wo, VN, VD):
    """o-proj weight [VH, VN*VD] -> [VH, VN*VD_PAD]: each head's VD input cols placed in
    the first VD of its 128-block, pad cols zero. Lets o-proj consume the 128-padded
    attention output directly (so no separate 72-trim / unaligned permute is needed)."""
    VH = Wo.shape[0]
    Wo = Wo.reshape(VH, VN, VD)
    Wp = torch.zeros(VH, VN, VD_PAD, dtype=torch.bfloat16)
    Wp[:, :, :VD] = Wo.to(torch.bfloat16)
    return Wp.reshape(VH, VN * VD_PAD).contiguous()


def _v_pad(W, b, VN, VD):
    """[VN*VD, VH] v weight -> [VN*VD_PAD, VH] natural order, zero-padded 72->128."""
    VH = W.shape[1]
    W = W.reshape(VN, VD, VH)
    b = b.reshape(VN, VD)
    Wp = torch.zeros(VN, VD_PAD, VH, dtype=torch.bfloat16)
    bp = torch.zeros(VN, VD_PAD, dtype=torch.bfloat16)
    Wp[:, :VD, :] = W.to(torch.bfloat16)
    bp[:, :VD] = b.to(torch.bfloat16)
    return Wp.reshape(VN * VD_PAD, VH).contiguous(), bp.reshape(-1).contiguous()


def _row_pad(W, b, target_rows):
    """Pad a [out, in] linear's OUTPUT rows (and bias) up to target_rows with zeros."""
    out, inn = W.shape
    Wp = torch.zeros(target_rows, inn, dtype=torch.bfloat16)
    Wp[:out] = W.to(torch.bfloat16)
    bp = torch.zeros(target_rows, dtype=torch.bfloat16)
    bp[:out] = b.to(torch.bfloat16)
    return Wp.contiguous(), bp.contiguous()


def _col_pad(W, target_cols):
    """Pad a [out, in] linear's INPUT cols (K dim) up to target_cols with zeros."""
    out, inn = W.shape
    Wp = torch.zeros(out, target_cols, dtype=torch.bfloat16)
    Wp[:, :inn] = W.to(torch.bfloat16)
    return Wp.contiguous()


_BIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "locateanything_3b_bin")


def _prepare_moonvit_weights(model, cfg, precision):
    """Model-dependent prep (pad/reorder/quantize), independent of DRAM placement.
    Returns an ordered list of (scope, name, kind, payload) entries replayed by
    _store_moonvit_weights. Order fixes the bump-allocator addresses, so the same list
    always lands weights at the same addresses (which the cached program bakes in).
    kind: 'mat'=precision-dependent matmul weight, 'bf16'=bias/norm. payload: q4-packed
    np.uint8 (mat+if4) or a bf16 tensor."""
    d = _vis_dims(cfg)
    VN, VD, VH, VI = d["VN"], d["VD"], d["VH"], d["VI"]
    VIS_MLP_PAD = ((VI + 63) // 64) * 64
    q_scale = math.sqrt(VD_PAD / VD)

    def _mat(W):  # matmul weight -> q4-packed bytes or bf16 tensor
        if precision == "if4":
            packed, _ = quantize_q4_64(W)
            return ('mat', packed)
        return ('mat', W.to(torch.bfloat16).contiguous())

    out = []
    for li, blk in enumerate(model.vision_model.encoder.blocks):
        Wqkv = blk.wqkv.weight.data.float(); bqkv = blk.wqkv.bias.data.float()
        qW, kW, vW = Wqkv[:VH], Wqkv[VH:2 * VH], Wqkv[2 * VH:]
        qb, kb, vb = bqkv[:VH], bqkv[VH:2 * VH], bqkv[2 * VH:]
        qWp, qbp = _qk_pad_reorder(qW, qb, VN, VD, scale=q_scale)
        kWp, kbp = _qk_pad_reorder(kW, kb, VN, VD, scale=1.0)
        vWp, vbp = _v_pad(vW, vb, VN, VD)
        fc0W, fc0b = _row_pad(blk.mlp.fc0.weight.data.float(), blk.mlp.fc0.bias.data.float(), VIS_MLP_PAD)
        fc1W = _col_pad(blk.mlp.fc1.weight.data.float(), VIS_MLP_PAD)
        bf = lambda t: t.to(torch.bfloat16).contiguous()
        out += [
            (li, 'q', *_mat(qWp)),   (li, 'q_bias', 'bf16', bf(qbp)),
            (li, 'k', *_mat(kWp)),   (li, 'k_bias', 'bf16', bf(kbp)),
            (li, 'v', *_mat(vWp)),   (li, 'v_bias', 'bf16', bf(vbp)),
            (li, 'o', *_mat(_o_pad(blk.wo.weight.data.float(), VN, VD))),
            (li, 'o_bias', 'bf16', bf(blk.wo.bias.data)),
            (li, 'norm0_w', 'bf16', bf(blk.norm0.weight.data)),
            (li, 'norm0_b', 'bf16', bf(blk.norm0.bias.data)),
            (li, 'norm1_w', 'bf16', bf(blk.norm1.weight.data)),
            (li, 'norm1_b', 'bf16', bf(blk.norm1.bias.data)),
            (li, 'fc0', *_mat(fc0W)), (li, 'fc0_bias', 'bf16', bf(fc0b)),
            (li, 'fc1', *_mat(fc1W)), (li, 'fc1_bias', 'bf16', bf(blk.mlp.fc1.bias.data)),
        ]
    fln = model.vision_model.encoder.final_layernorm
    out += [('final', 'ln_w', 'bf16', bf(fln.weight.data)),
            ('final', 'ln_b', 'bf16', bf(fln.bias.data))]
    return out


def _store_moonvit_weights(ue, prepared, cfg):
    """Replay a prepared list into params DRAM, recording addresses into
    ue.mv_layer_addrs / ue.mv_final_*. Deterministic store order = stable addresses."""
    depth = _vis_dims(cfg)["depth"]
    ue.mv_layer_addrs = [{} for _ in range(depth)]
    import numpy as np
    for scope, name, kind, payload in prepared:
        if kind == 'mat' and ue.mv_precision == "if4":
            arr = payload if isinstance(payload, np.ndarray) else np.frombuffer(payload, dtype=np.uint8)
            sa, da = store_quantized_weight(ue, arr)
            ue.mv_layer_addrs[scope][f'{name}_scale'] = sa
            ue.mv_layer_addrs[scope][f'{name}_data'] = da
        elif kind == 'mat':                      # bf16 matmul weight
            ue.mv_layer_addrs[scope][f'{name}_weight'] = store_weight(ue, payload)
        elif scope == 'final':
            setattr(ue, f'mv_final_{name}', store_weight(ue, payload))
        else:                                    # per-layer bf16 bias/norm
            ue.mv_layer_addrs[scope][name] = store_weight(ue, payload)


def moonvit_load_weights(ue, model, cfg, precision="if4"):
    """Prepare (or load cached) MoonViT weights, then store to params DRAM. The prepared
    blobs are cached to disk so repeat runs skip padding/reorder/quantization (and don't
    need `model` for weights). Big matmuls are if4 by default; biases/norms stay bf16."""
    ue.mv_precision = precision
    os.makedirs(_BIN_DIR, exist_ok=True)
    cache = os.path.join(_BIN_DIR, f"moonvit_weights_{precision}.pt")
    if os.path.exists(cache):
        prepared = torch.load(cache, weights_only=False)
        print(f"  MoonViT weights: loaded prepared cache {os.path.basename(cache)}")
    else:
        assert model is not None, "no weight cache and no model to prepare from"
        prepared = _prepare_moonvit_weights(model, cfg, precision)
        torch.save(prepared, cache)
        print(f"  MoonViT weights: prepared from model -> cached {os.path.basename(cache)}")
    _store_moonvit_weights(ue, prepared, cfg)
    print(f"  MoonViT weights stored: {len(ue.mv_layer_addrs)} layers ({precision}); "
          f"params DRAM usage: {ue.get_params_dram_usage()/1048576:.0f} MB")


# =====================================================================================
# DRAM region allocation (sized for a specific token count N)
# =====================================================================================

def moonvit_setup_dram(ue, N, cfg):
    """Allocate MoonViT activation regions for N patch tokens. The sequence length is
    padded up to a multiple of 64 (SKILL rule #1 — flash attention's seq_len drives the
    score-matrix row stride, which must be 128-byte aligned). Padded query rows are
    ignored at the merge; padded KEY columns are masked via the flash BIAS. Recompiled
    per image grid (N varies), so sizing to the actual N keeps the footprint small."""
    d = _vis_dims(cfg)
    VN, VD, VH, VI, merge = d["VN"], d["VD"], d["VH"], d["VI"], d["merge"]
    VIS_MLP_PAD = ((VI + 63) // 64) * 64
    bpe = 2
    N_real = N
    # Pad seq to a multiple of 128 (not just 64): flash stores partial scores with a
    # row stride of N elements, and the tiling appears to want 128-element alignment.
    N = ((N + 127) // 128) * 128                     # padded seq len (128-aligned)
    N_merged = N_real // (merge * merge)
    merge_dim = merge * merge * VH                   # 4*1152 = 4608

    ue.MV_N = N                                      # padded seq len (strides / M / flash)
    ue.MV_N_REAL = N_real                            # real patch count (merge reads these)
    ue.MV_N_MERGED = N_merged
    ue.MV_IO_A = ue.allocate_tensor_dram(N * VH * bpe)
    ue.MV_IO_B = ue.allocate_tensor_dram(N * VH * bpe)
    # zero the layer I/O so padded seq rows start clean (LayerNorm of a zero row -> beta,
    # no NaN; they stay isolated from real rows thereafter).
    io_zeros = torch.zeros(N * VH, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(ue.MV_IO_A, io_zeros)
    ue.dma_to_accelerator_memory(ue.MV_IO_B, io_zeros)
    ue.MV_NORM_OUT = ue.allocate_tensor_dram(N * VH * bpe)
    ue.MV_Q = ue.allocate_tensor_dram(N * VN * VD_PAD * bpe)
    ue.MV_K = ue.allocate_tensor_dram(N * VN * VD_PAD * bpe)
    ue.MV_V = ue.allocate_tensor_dram(N * VN * VD_PAD * bpe)
    ue.MV_Q_PAD = ue.allocate_tensor_dram(VN * N * VD_PAD * bpe)
    ue.MV_K_PAD = ue.allocate_tensor_dram(VN * N * VD_PAD * bpe)
    ue.MV_V_PAD = ue.allocate_tensor_dram(VN * N * VD_PAD * bpe)
    zeros_pad = torch.zeros(VN * N * VD_PAD, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(ue.MV_Q_PAD, zeros_pad)
    ue.dma_to_accelerator_memory(ue.MV_K_PAD, zeros_pad)
    ue.dma_to_accelerator_memory(ue.MV_V_PAD, zeros_pad)
    ue.MV_ATTN_OUT = ue.allocate_tensor_dram(VN * N * VD_PAD * bpe)
    # zero so a NaN read here means flash COMPUTED NaN, not that it failed to write
    ue.dma_to_accelerator_memory(ue.MV_ATTN_OUT, torch.zeros(VN * N * VD_PAD, dtype=torch.bfloat16))
    # debug tap: softmax(QK^T) dump [N, N] (flash debug_mode), zeroed for clean read
    ue.MV_SM_DEBUG = ue.allocate_tensor_dram(N * N * bpe)
    ue.dma_to_accelerator_memory(ue.MV_SM_DEBUG, torch.zeros(N * N, dtype=torch.bfloat16))
    ue.MV_ATTN_SCRATCH = ue.allocate_tensor_dram(
        ((VD_PAD + N) * N + N * VD_PAD) * bpe)
    # attention result kept in padded head layout [N, VN*VD_PAD]; o-proj weight zero-pads
    ue.MV_ATTN_RESULT = ue.allocate_tensor_dram(N * VN * VD_PAD * bpe)
    ue.MV_O_PROJ = ue.allocate_tensor_dram(N * VH * bpe)
    ue.MV_RESIDUAL = ue.allocate_tensor_dram(N * VH * bpe)
    ue.MV_MLP_INTER = ue.allocate_tensor_dram(N * VIS_MLP_PAD * bpe)
    ue.MV_MLP_OUT = ue.allocate_tensor_dram(N * VH * bpe)
    ue.MV_POST_NORM = ue.allocate_tensor_dram(N * VH * bpe)
    # Rope table tiled VN× so a single rope_hf_core_dram call at M=VN*N covers all
    # heads. Row h*N+t uses cos/sin[t] — identical across heads (position-only rope).
    ue.MV_ROPE = ue.allocate_tensor_dram(VN * N * 2 * VD_PAD * bpe)
    ue.MV_ROPE_COS = ue.MV_ROPE
    ue.MV_ROPE_SIN = ue.MV_ROPE + VD_PAD * bpe
    ue.MV_PERM_TEMP = ue.allocate_tensor_dram(VN * N * VD_PAD * bpe * 2)
    # MoonViT attention fixed operand buffers, one head at a time.
    ue.MV_FLASH_Q   = ue.allocate_tensor_dram(N * VD_PAD * bpe)
    ue.MV_FLASH_K   = ue.allocate_tensor_dram(N * VD_PAD * bpe)
    ue.MV_FLASH_V   = ue.allocate_tensor_dram(N * VD_PAD * bpe)
    ue.MV_FLASH_OUT = ue.allocate_tensor_dram(N * VD_PAD * bpe)
    # Flash BIAS [N, N] row-major (added to QK^T before softmax): mask padded KEY
    # columns (>= N_real) with a large negative so real queries ignore them. Same mask
    # for every head/layer (depends only on key index).
    ue.MV_ATTN_BIAS = ue.allocate_tensor_dram(N * N * bpe)
    bias = torch.zeros(N, N, dtype=torch.bfloat16)
    if N_real < N:
        bias[:, N_real:] = -1e38   # proven mask value (qwen/smolvlm flash bias); NOT -30000
    ue.dma_to_accelerator_memory(ue.MV_ATTN_BIAS, bias.flatten())
    # identity for the permute core (reuse inherited VIS_PERMUTE_PARAMS_DRAM if present)
    if not hasattr(ue, "VIS_PERMUTE_PARAMS_DRAM"):
        permute_size = UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe
        ue.VIS_PERMUTE_PARAMS_DRAM = ue.get_params_dram_addr()
        ue.dma_write(DMA_DEVICE_H2C, ue.VIS_PERMUTE_PARAMS_DRAM,
                     torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16), permute_size)
        ue.allocate_params_dram(permute_size)
    # merged output [N_merged, 4608] -> overwrite VIS_MERGED_DRAM so run_connector picks it up
    ue.MV_MERGED = ue.allocate_tensor_dram(N_merged * merge_dim * bpe)
    ue.VIS_MERGED_DRAM = ue.MV_MERGED
    print(f"  MoonViT DRAM allocated for N={N} (N_merged={N_merged}, merge_dim={merge_dim})")


# =====================================================================================
# Host-side input prep: patch embedding (+pos emb) and the 2D-RoPE cos/sin table
# =====================================================================================

def moonvit_prepare_input(ue, model, pixel_values, grid_hw, cfg, device="cpu"):
    """Run patch_embed (conv + learnable 2D-interp pos-emb) and build the rope table on
    host; DMA both to the board. Returns N (token count)."""
    d = _vis_dims(cfg)
    VD, VN = d["VD"], d["VN"]
    N_real = grid_hw[0] * grid_hw[1]
    N = ue.MV_N                       # padded (64-aligned) seq len; rope/patch sized to it
    half = VD // 2  # 36

    with torch.no_grad():
        # patch_embed already adds the interpolated learnable pos-emb -> [N_real, VH]
        patch = model.vision_model.patch_embed(
            pixel_values.to(device, dtype=torch.bfloat16), grid_hw).to(torch.bfloat16).cpu()
        assert patch.shape[0] == N_real, f"patch tokens {patch.shape[0]} != N_real={N_real}"
        # 2D RoPE freqs_cis: [N_real, half] complex (interleaved x/y per the reference)
        fc = model.vision_model.encoder.rope_2d.freqs_cis(grid_hw, torch.device(device))
        fc = fc.cpu()  # [N_real, half] complex64
        cos_raw = fc.real.to(torch.bfloat16)  # [N_real, half]
        sin_raw = fc.imag.to(torch.bfloat16)  # [N_real, half]

    # Fill the padded seq rows with copies of real patch rows, NOT zeros: a zero row has
    # zero variance, so LayerNorm does (0-0)/sqrt(0+eps) -> 0*inf = NaN, which then
    # poisons q/k/v -> V[pad] -> the whole attention output. Pad rows are masked out as
    # keys, so their (finite) content never affects the real tokens.
    pad = N - N_real
    if pad > 0:
        reps = (pad + N_real - 1) // N_real
        patch_full = torch.cat([patch] + [patch[:N_real]] * reps, dim=0)[:N]
    else:
        patch_full = patch
    ue.dma_to_accelerator_memory(ue.MV_IO_A, patch_full.contiguous().flatten())

    # Pad-128 rotate_half table (matches the validated vision_rope_hf_core_dram_test):
    # cos lanes [0:half] and [64:64+half] (gaps = 1.0 identity); sin lo pre-negated. Pad
    # seq rows (N_real..N) get identity rope (cos=1, sin=0) — they are masked-out keys.
    rope_table = torch.zeros(N, 2, VD_PAD, dtype=torch.bfloat16)
    cos_t = rope_table[:, 0, :]
    sin_t = rope_table[:, 1, :]
    cos_t.fill_(1.0)
    cos_t[:N_real, :half] = cos_raw
    cos_t[:N_real, 64:64 + half] = cos_raw
    sin_t[:N_real, :half] = -sin_raw
    sin_t[:N_real, 64:64 + half] = sin_raw
    # Tile VN× so the single per-buffer rope call (M=VN*N) reads the right row per token.
    rope_tiled = rope_table.unsqueeze(0).expand(VN, N, 2, VD_PAD).reshape(VN * N, 2, VD_PAD)
    ue.dma_to_accelerator_memory(ue.MV_ROPE, rope_tiled.contiguous().flatten())
    print(f"  MoonViT patch[{N_real},{patch.shape[1]}] + rope[{VN}x{N},2,{VD_PAD}] "
          f"(seq padded {N_real}->{N}) DMA'd")
    return N


# =====================================================================================
# Compile the encoder program (single capture stream)
# =====================================================================================

def moonvit_compile_encoder(ue, cfg, grid_hw):
    """Capture the full MoonViT transformer stack + final norm + 2x2 merge into one
    instruction stream. Returns the program DRAM address. Static M=N (recompiled per
    grid), so no PBI M-register games — correctness first."""
    d = _vis_dims(cfg)
    depth, VN, VD, VH, VI, merge = d["depth"], d["VN"], d["VD"], d["VH"], d["VI"], d["merge"]
    VIS_MLP_PAD = ((VI + 63) // 64) * 64
    bpe = 2
    N = ue.MV_N
    N_merged = ue.MV_N_MERGED
    head_stride_pad = N * VD_PAD * bpe
    VN_VD_PAD = VN * VD_PAD

    from user_dma_core import TYPE
    prec = getattr(ue, "mv_precision", "if4")

    # --- program (instruction) cache: keyed on seq + precision; bakes in the (deterministic)
    # weight/tensor addresses + absolute jumps, so reload is valid only when load_weights/
    # setup_dram ran first and the program lands at the same addr (deterministic alloc order). ---
    os.makedirs(_BIN_DIR, exist_ok=True)
    moonvit_key = f"moonvit_encoder_N{N}_{prec}_unified_attention"
    _cached = programs_bin_load(os.path.dirname(_BIN_DIR), moonvit_key)
    if _cached is not None:
        prog, _ = _cached
        program_addr = ue.get_program_dram_addr()
        ue.dma_write(DMA_DEVICE_H2C, program_addr, prog, len(prog))
        ue.allocate_program_dram(len(prog))
        print(f"  MoonViT encoder: loaded cached program programs.bin[{moonvit_key}] "
              f"({len(prog)} bytes = {len(prog)//32} instructions)")
        return program_addr

    def _mm(M, K, Nn, A, name, la, OUT, bias=None, gelu=False):
        """Precision-aware matmul: if4 (quantized B + scale) or bf16.

        PBI runtime-M: passes gpr_M_reg=vis_M_reg (primed with N) so matmat_mul_core
        routes to matmat_mul_core_pbi -> ONE templated M-iteration + hardware loop
        instead of unrolling ~M/M_chunk static tiles. This is the dominant program-
        size win (6 big matmuls/layer x 27 layers). Mirrors the LM prefill matmuls
        (locateanything_3b_test.py: q/k/v/o/gate/up all pass gpr_M_reg=gf_seq_len).
        Flash/permute/norm stay static (flash can't be PBI - back-to-back bug)."""
        if prec == "if4":
            ue.matmat_mul_core(M=M, K=K, N=Nn, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{name}_data'],
                               OUTPUT_DRAM_ADDR=OUT, C_DRAM_ADDR=bias, bias_mode="broadcast_N",
                               is_B_quantized=True, data_type=TYPE.IF4,
                               SCALE_DRAM_ADDR=la[f'{name}_scale'], gelu_enable=gelu,
                               gpr_M_reg=vis_M_reg)
        else:
            ue.matmat_mul_core(M=M, K=K, N=Nn, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{name}_weight'],
                               OUTPUT_DRAM_ADDR=OUT, C_DRAM_ADDR=bias, bias_mode="broadcast_N",
                               gelu_enable=gelu, gpr_M_reg=vis_M_reg)

    global _SILENT_MODE
    _SILENT_MODE = True
    ue.start_capture()

    ue.reset_isa_reg_counter()
    vis_M_reg = ue.alloc_isa_reg()
    ue.generate_instruction_add_set(vis_M_reg, N)

    def _mv_flash_call(q_src, k_src, v_src, out_dst):
        """Marshal one head's Q/K/V into fixed buffers and run unified attention."""
        elems = N * VD_PAD
        for src, dst in ((q_src, ue.MV_FLASH_Q), (k_src, ue.MV_FLASH_K), (v_src, ue.MV_FLASH_V)):
            ue.accelerator_memory_to_sram(src, 0x00000, elems)
            ue.sram_to_accelerator_memory(0x00000, dst, elems)
        ue.unified_attention_core(
            batch=N,
            aligned_seq_len=N,
            head_dim=VD_PAD,
            Q_DRAM_ADDR=ue.MV_FLASH_Q,
            K_DRAM_ADDR=ue.MV_FLASH_K,
            V_DRAM_ADDR=ue.MV_FLASH_V,
            BIAS_DRAM_ADDR=ue.MV_ATTN_BIAS,
            OUTPUT_DRAM_ADDR=ue.MV_FLASH_OUT,
            SCRATCH_DRAM_ADDR=ue.MV_ATTN_SCRATCH,
            IDENTITY_DRAM_ADDR=ue.IDENTITY_DRAM_ADDR)
        ue.accelerator_memory_to_sram(ue.MV_FLASH_OUT, 0x00000, elems)
        ue.sram_to_accelerator_memory(0x00000, out_dst, elems)

    for li in range(depth):
        la = ue.mv_layer_addrs[li]
        h_in = ue.MV_IO_A if li % 2 == 0 else ue.MV_IO_B
        h_out = ue.MV_IO_B if li % 2 == 0 else ue.MV_IO_A

        # --- norm0 (LayerNorm) ---
        ue.layer_norm_core_dram(M=N, N=VH, A_DRAM_ADDR=h_in, OUTPUT_DRAM_ADDR=ue.MV_NORM_OUT,
                                GAMMA_DRAM_ADDR=la['norm0_w'], BETA_DRAM_ADDR=la['norm0_b'])

        # --- q/k/v projections into per-head-128-padded layout [N, VN*128] ---
        for proj, dst in [('q', ue.MV_Q), ('k', ue.MV_K), ('v', ue.MV_V)]:
            _mm(N, VH, VN_VD_PAD, ue.MV_NORM_OUT, proj, la, dst, bias=la[f'{proj}_bias'])

        # --- permute [N, VN, 128] -> [VN, N, 128] for q/k/v via strided DMA ---
        # Each SRAM load: d0*last = N*VD_PAD = 1408*128 = 180224 elems = 352KB < 512KB SRAM.
        # Chunk size (last=VD_PAD=128 elems = 256 bytes) is a multiple of 128-byte SRAM rows.
        for src, dst in [(ue.MV_Q, ue.MV_Q_PAD), (ue.MV_K, ue.MV_K_PAD), (ue.MV_V, ue.MV_V_PAD)]:
            ue._vis_transpose_10(src, dst, d0=N, d1=VN, last=VD_PAD)

        # --- 2D RoPE: one call per buffer over ALL heads (M=VN*N, head-major Q/K_PAD) ---
        # Reprime vis_M_reg to VN*N for the rope calls, then restore to N for matmuls.
        ue.generate_instruction_add_set(vis_M_reg, VN * N)
        for buf in (ue.MV_Q_PAD, ue.MV_K_PAD):
            ue.rope_hf_core_dram(M=VN * N, N=VD_PAD,
                                 input_dram_addr=buf, output_dram_addr=buf,
                                 cos_dram_addr=ue.MV_ROPE_COS, sin_dram_addr=ue.MV_ROPE_SIN,
                                 gpr_M_reg=vis_M_reg)
        ue.generate_instruction_add_set(vis_M_reg, N)

        # --- per-head attention ---
        for h in range(VN):
            _mv_flash_call(ue.MV_Q_PAD + h * head_stride_pad,
                           ue.MV_K_PAD + h * head_stride_pad,
                           ue.MV_V_PAD + h * head_stride_pad,
                           ue.MV_ATTN_OUT + h * head_stride_pad)

        # --- inverse permute [VN, N, 128] -> [N, VN*128] via strided DMA ---
        # Each SRAM load: d0*last = VN*VD_PAD = 16*128 = 2048 elems = 4KB — trivial.
        ue._vis_transpose_10(ue.MV_ATTN_OUT, ue.MV_ATTN_RESULT, d0=VN, d1=N, last=VD_PAD)

        # --- o proj (K=VN*128, pad cols zeroed in weight) + residual + norm1 fused ---
        _mm(N, VN_VD_PAD, VH, ue.MV_ATTN_RESULT, 'o', la, ue.MV_O_PROJ, bias=la['o_bias'])
        layer_norm_core_dram_post_add(ue, M=N, N=VH,
                                      A_DRAM_ADDR=h_in, B_DRAM_ADDR=ue.MV_O_PROJ,
                                      ADDOUTPUT_DRAM_ADDR=ue.MV_RESIDUAL,
                                      NORMOUTPUT_DRAM_ADDR=ue.MV_NORM_OUT,
                                      GAMMA_DRAM_ADDR=la['norm1_w'], BETA_DRAM_ADDR=la['norm1_b'])

        # --- MLP: fc0 + gelu, fc1, residual add ---
        _mm(N, VH, VIS_MLP_PAD, ue.MV_NORM_OUT, 'fc0', la, ue.MV_MLP_INTER,
            bias=la['fc0_bias'], gelu=True)
        _mm(N, VIS_MLP_PAD, VH, ue.MV_MLP_INTER, 'fc1', la, ue.MV_MLP_OUT, bias=la['fc1_bias'])
        eltwise_add_core_dram(ue, size=N * VH, A_DRAM_ADDR=ue.MV_RESIDUAL,
                              B_DRAM_ADDR=ue.MV_MLP_OUT, OUTPUT_DRAM_ADDR=h_out)

    # --- final layernorm ---
    final_vis = ue.MV_IO_A if depth % 2 == 0 else ue.MV_IO_B
    ue.layer_norm_core_dram(M=N, N=VH, A_DRAM_ADDR=final_vis, OUTPUT_DRAM_ADDR=ue.MV_POST_NORM,
                            GAMMA_DRAM_ADDR=ue.mv_final_ln_w, BETA_DRAM_ADDR=ue.mv_final_ln_b)

    # --- 2x2 patch merge [gh,gw,VH] -> [N_merged, 4*VH] ---
    nh, nw = grid_hw[0] // merge, grid_hw[1] // merge
    smart_bf16_permute_core(ue, dims=[nh, merge, nw, merge, VH], permute_indices=[0, 2, 1, 3, 4],
                            input_dram_addr=ue.MV_POST_NORM, output_dram_addr=ue.MV_MERGED,
                            params_dram_addr=ue.VIS_PERMUTE_PARAMS_DRAM,
                            temp_dram_start=ue.MV_PERM_TEMP)

    ue.generate_instruction_halt()

    ue.stop_capture()
    _SILENT_MODE = False
    prog = bytearray()
    for inst in ue.capture_buffer:
        prog.extend(inst.get_bytes())
    program_addr = ue.get_program_dram_addr()
    ue.dma_write(DMA_DEVICE_H2C, program_addr, prog, len(prog))
    ue.allocate_program_dram(len(prog))
    ue.clear_capture_buffer()
    programs_bin_store(os.path.dirname(_BIN_DIR), moonvit_key, bytes(prog),
                       meta={"N": N, "precision": prec, "depth": depth})
    _original_print(f"  MoonViT encoder compiled: {len(prog)//1024//1024}MB "
                    f"({len(prog)//32} instructions, {depth} layers) -> cached programs.bin[{moonvit_key}]")
    return program_addr


def moonvit_run_encoder(ue, program_addr, timeout_s=300.0):
    """Execute the compiled MoonViT program. Output [N_merged, 4608] lands in
    ue.VIS_MERGED_DRAM, ready for run_connector."""
    ue.start_execute_from_dram(program_addr)
    wall_s = ue._wait_with_heartbeat(lambda: ue.wait_queue(timeout_s), label="moonvit")
    latency_us = ue._corrected_hw_latency_us(wall_s)
    ue._last_moonvit_latency_us = latency_us
    print(f"  MoonViT encoder executed -> VIS_MERGED_DRAM 0x{ue.VIS_MERGED_DRAM:X}  "
          f"(HW {latency_us/1e3:.1f} ms)")
    return latency_us


# =====================================================================================
# Staged (op-by-op) debug runner — execute one op, read it back, NaN-check, repeat.
# =====================================================================================

def _step(ue, label, emit, check_addr=None, check_shape=None, real_rows=None,
          stop=False, check=True, timeout_s=120.0):
    """Capture exactly one op group, execute it on hardware immediately, then read the
    result back and assert it is NaN/Inf-free. The NaN assert IS the movable stop: it
    halts at the first bad stage, so you fix it and re-run to land on the next one.
    Pass stop=True (or --stop-after <label>) to force a halt after a known-good stage."""
    ue.clear_inst_id()
    ue.start_capture()
    emit()
    ue.stop_capture()
    ue.generate_instruction_halt()
    prog = ue.get_program_dram_addr()          # transient: reuse same region each step
    ue.write_captured_instructions_to_dram(prog)
    ue.clear_capture_buffer()
    ue.start_execute_from_dram(prog)
    ue.wait_queue(timeout_s)

    if check and check_addr is not None:
        t = ue.dma_from_accelerator_memory(check_addr, check_shape).float()
        view = t[:real_rows] if real_rows is not None else t
        n = int(torch.isnan(view).sum()); i = int(torch.isinf(view).sum())
        mx = float(view.abs().max()) if view.numel() else 0.0
        mean = float(view.abs().mean()) if view.numel() else 0.0
        print(f"  [{label:18s}] {str(tuple(check_shape)):>16s}  nan={n:<6d} inf={i:<4d} "
              f"absmax={mx:.4g} absmean={mean:.4g}")
        assert n == 0 and i == 0, f"NaN/Inf detected at stage '{label}' — fix here"
    else:
        print(f"  [{label:18s}] executed")
    if stop:
        assert False, f"FORCED STOP after stage '{label}' (--stop-after)"


def moonvit_run_staged(ue, cfg, grid_hw, stop_after=None, check=True, timeout_s=120.0,
                       mask_pad=True):
    """Run the encoder ONE op at a time (separate capture+execute per op), NaN-checking
    each output. Mirrors moonvit_compile_encoder's op chain exactly. Output ends in
    ue.VIS_MERGED_DRAM (same as the single-stream path) if it runs to completion.

    Walk it down: leave stop_after=None and it halts at the first NaN/Inf; fix that op,
    re-run, it advances to the next. Or set stop_after='L0.flash' to force-stop earlier."""
    d = _vis_dims(cfg)
    depth, VN, VD, VH, VI, merge = d["depth"], d["VN"], d["VD"], d["VH"], d["VI"], d["merge"]
    VIS_MLP_PAD = ((VI + 63) // 64) * 64
    bpe = 2
    N = ue.MV_N
    N_real = ue.MV_N_REAL
    head_stride_pad = N * VD_PAD * bpe
    VN_VD_PAD = VN * VD_PAD
    from user_dma_core import TYPE
    prec = getattr(ue, "mv_precision", "if4")

    def _mm_emit(M, K, Nn, A, name, la, OUT, bias=None, gelu=False):
        def f():
            if prec == "if4":
                ue.matmat_mul_core(M=M, K=K, N=Nn, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{name}_data'],
                                   OUTPUT_DRAM_ADDR=OUT, C_DRAM_ADDR=bias, bias_mode="broadcast_N",
                                   is_B_quantized=True, data_type=TYPE.IF4,
                                   SCALE_DRAM_ADDR=la[f'{name}_scale'], gelu_enable=gelu)
            else:
                ue.matmat_mul_core(M=M, K=K, N=Nn, A_DRAM_ADDR=A, B_DRAM_ADDR=la[f'{name}_weight'],
                                   OUTPUT_DRAM_ADDR=OUT, C_DRAM_ADDR=bias, bias_mode="broadcast_N",
                                   gelu_enable=gelu)
        return f

    def step(label, emit, addr=None, shape=None):
        _step(ue, label, emit, addr, shape, real_rows=N_real,
              stop=(label == stop_after), check=check, timeout_s=timeout_s)

    print(f"=== MoonViT staged run (N={N_real}->pad {N}, prec={prec}, "
          f"stop_after={stop_after}) ===")
    for li in range(depth):
        la = ue.mv_layer_addrs[li]
        h_in = ue.MV_IO_A if li % 2 == 0 else ue.MV_IO_B
        h_out = ue.MV_IO_B if li % 2 == 0 else ue.MV_IO_A
        L = f"L{li}"

        step(f"{L}.norm0",
             lambda h_in=h_in, la=la: ue.layer_norm_core_dram(
                 M=N, N=VH, A_DRAM_ADDR=h_in, OUTPUT_DRAM_ADDR=ue.MV_NORM_OUT,
                 GAMMA_DRAM_ADDR=la['norm0_w'], BETA_DRAM_ADDR=la['norm0_b']),
             ue.MV_NORM_OUT, (N, VH))

        for proj, dst in [('q', ue.MV_Q), ('k', ue.MV_K), ('v', ue.MV_V)]:
            step(f"{L}.{proj}proj",
                 _mm_emit(N, VH, VN_VD_PAD, ue.MV_NORM_OUT, proj, la, dst, bias=la[f'{proj}_bias']),
                 dst, (N, VN_VD_PAD))

        for proj, src, dst in [('q', ue.MV_Q, ue.MV_Q_PAD), ('k', ue.MV_K, ue.MV_K_PAD),
                               ('v', ue.MV_V, ue.MV_V_PAD)]:
            step(f"{L}.perm_{proj}",
                 lambda src=src, dst=dst: smart_bf16_permute_core(
                     ue, dims=[N, VN, VD_PAD], permute_indices=[1, 0, 2],
                     input_dram_addr=src, output_dram_addr=dst,
                     params_dram_addr=ue.VIS_PERMUTE_PARAMS_DRAM, temp_dram_start=ue.MV_PERM_TEMP),
                 dst, (VN * N, VD_PAD))

        def rope_emit(la=la):
            ue.reset_isa_reg_counter()
            r = ue.alloc_isa_reg()
            ue.generate_instruction_add_set(r, N)
            for h in range(VN):
                for buf in (ue.MV_Q_PAD, ue.MV_K_PAD):
                    a = buf + h * head_stride_pad
                    ue.rope_hf_core_dram(M=N, N=VD_PAD, input_dram_addr=a, output_dram_addr=a,
                                         cos_dram_addr=ue.MV_ROPE_COS, sin_dram_addr=ue.MV_ROPE_SIN,
                                         gpr_M_reg=r)
        step(f"{L}.rope", rope_emit, ue.MV_Q_PAD, (VN * N, VD_PAD))

        # Attention per head. Staged debug no longer has the legacy softmax dump tap.
        bias_addr = ue.MV_ATTN_BIAS if mask_pad else ue.MV_SM_DEBUG
        for h in range(VN):
            def flash_emit(h=h):
                ue.unified_attention_core(
                    batch=N,
                    aligned_seq_len=N,
                    head_dim=VD_PAD,
                    Q_DRAM_ADDR=ue.MV_Q_PAD + h * head_stride_pad,
                    K_DRAM_ADDR=ue.MV_K_PAD + h * head_stride_pad,
                    V_DRAM_ADDR=ue.MV_V_PAD + h * head_stride_pad,
                    BIAS_DRAM_ADDR=bias_addr,
                    OUTPUT_DRAM_ADDR=ue.MV_ATTN_OUT + h * head_stride_pad,
                    SCRATCH_DRAM_ADDR=ue.MV_ATTN_SCRATCH,
                    IDENTITY_DRAM_ADDR=ue.IDENTITY_DRAM_ADDR)

            step(f"{L}.flash_h{h}", flash_emit, ue.MV_ATTN_OUT + h * head_stride_pad, (N, VD_PAD))

        step(f"{L}.inv_perm",
             lambda: smart_bf16_permute_core(
                 ue, dims=[VN, N, VD_PAD], permute_indices=[1, 0, 2],
                 input_dram_addr=ue.MV_ATTN_OUT, output_dram_addr=ue.MV_ATTN_RESULT,
                 params_dram_addr=ue.VIS_PERMUTE_PARAMS_DRAM, temp_dram_start=ue.MV_PERM_TEMP),
             ue.MV_ATTN_RESULT, (N, VN_VD_PAD))

        step(f"{L}.oproj",
             _mm_emit(N, VN_VD_PAD, VH, ue.MV_ATTN_RESULT, 'o', la, ue.MV_O_PROJ, bias=la['o_bias']),
             ue.MV_O_PROJ, (N, VH))

        step(f"{L}.norm1_postadd",
             lambda h_in=h_in, la=la: layer_norm_core_dram_post_add(
                 ue, M=N, N=VH, A_DRAM_ADDR=h_in, B_DRAM_ADDR=ue.MV_O_PROJ,
                 ADDOUTPUT_DRAM_ADDR=ue.MV_RESIDUAL, NORMOUTPUT_DRAM_ADDR=ue.MV_NORM_OUT,
                 GAMMA_DRAM_ADDR=la['norm1_w'], BETA_DRAM_ADDR=la['norm1_b']),
             ue.MV_NORM_OUT, (N, VH))

        step(f"{L}.fc0_gelu",
             _mm_emit(N, VH, VIS_MLP_PAD, ue.MV_NORM_OUT, 'fc0', la, ue.MV_MLP_INTER,
                      bias=la['fc0_bias'], gelu=True),
             ue.MV_MLP_INTER, (N, VIS_MLP_PAD))

        step(f"{L}.fc1",
             _mm_emit(N, VIS_MLP_PAD, VH, ue.MV_MLP_INTER, 'fc1', la, ue.MV_MLP_OUT, bias=la['fc1_bias']),
             ue.MV_MLP_OUT, (N, VH))

        step(f"{L}.residual",
             lambda h_out=h_out: eltwise_add_core_dram(
                 ue, size=N * VH, A_DRAM_ADDR=ue.MV_RESIDUAL, B_DRAM_ADDR=ue.MV_MLP_OUT,
                 OUTPUT_DRAM_ADDR=h_out),
             h_out, (N, VH))

    final_vis = ue.MV_IO_A if depth % 2 == 0 else ue.MV_IO_B
    step("final_norm",
         lambda: ue.layer_norm_core_dram(
             M=N, N=VH, A_DRAM_ADDR=final_vis, OUTPUT_DRAM_ADDR=ue.MV_POST_NORM,
             GAMMA_DRAM_ADDR=ue.mv_final_ln_w, BETA_DRAM_ADDR=ue.mv_final_ln_b),
         ue.MV_POST_NORM, (N, VH))

    nh, nw = grid_hw[0] // merge, grid_hw[1] // merge
    step("merge",
         lambda: smart_bf16_permute_core(
             ue, dims=[nh, merge, nw, merge, VH], permute_indices=[0, 2, 1, 3, 4],
             input_dram_addr=ue.MV_POST_NORM, output_dram_addr=ue.MV_MERGED,
             params_dram_addr=ue.VIS_PERMUTE_PARAMS_DRAM, temp_dram_start=ue.MV_PERM_TEMP),
         ue.MV_MERGED, (ue.MV_N_MERGED, merge * merge * VH))
    print("  MoonViT staged run complete — no NaN/Inf through all stages.")

if __name__ == "__main__":
    main()
