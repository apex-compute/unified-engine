#!/usr/bin/env python3
"""
Gemma4 E2B inference on accelerator: prefill + decode.

  - Config from gemma4_e2b_config.json; weights from a single bin (see below).
  - Prefill: compiled each run. Decoder: if gemma4_e2b_bin/decoder_instruction.bin and
    gemma4_e2b_bin/decoder_instruction_meta.json exist, skip decoder compile and load
    program sizes from meta; otherwise compile and write the bin + meta.
  - Run prefill then decode loop. For numeric verification use gemma4_e2b_numeric.py.

Weights:
  - Default: gemma4_e2b_bin/weights_gemma4_e2b_hf.bin (generated from HF model if missing).
  - --local-weights: use gemma4_e2b_bin/full_model_weights.bin instead.

Usage:
  python gemma4_e2b_test.py
  python gemma4_e2b_test.py --prompt "your prompt"
  python gemma4_e2b_test.py --image path/to/image.jpg
  python gemma4_e2b_test.py --image path/to/image.jpg --prompt "What is in this image?"
  python gemma4_e2b_test.py --dev xdma0 [--cycle 5.62]
  python gemma4_e2b_test.py --local-weights

Fixed layout: gemma4_e2b_test.py, gemma4_e2b_numeric.py, *.json, and gemma4_e2b_bin/ live in the same folder.
  user_dma_core.py is two folders up (repo root); that directory is added to sys.path.
"""

import json
import math
import os
import sys

# This file's folder: gemma4_e2b_bin/, *.json, decoder_program.json live here. user_dma_core is two levels up (repo root).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoTokenizer
from huggingface_hub import snapshot_download
import time
# pcie_utils imports (run from andromeda/pcie_utils or with PYTHONPATH)
import user_dma_core
from user_dma_core import DMA_DEVICE_H2C, DRAM_INSTRUCTION_ADDR, TYPE, UE_FMAX_CONTEXT_SIZE, UE_VECTOR_SIZE, UE_ARGMAX_INDEX, URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS, set_dma_device
from user_dma_core import UnifiedEngine

# Audio encoder primitives (Conformer ops ported from Parakeet)
from audio_primitives import (
    silu_core_dram as _aud_silu,
    glu_core_dram as _aud_glu,
    half_step_residual_core_dram as _aud_half_step,
    build_toeplitz_for_depthwise as _aud_build_toeplitz,
    depthwise_conv1d_core_dram as _aud_depthwise_conv1d,
    copy_dram_to_dram_chunked as _aud_copy_chunked,
    eltwise_add_core_dram as _aud_eltwise_add,
)

# --- BROAD PRINT SUPPRESSION FOR LIBRARIES ---
import builtins

_original_print = builtins.print
_SILENT_MODE = False

def quiet_print(*args, **kwargs):
    """Suppress prints when _SILENT_MODE is True; otherwise print normally."""
    if _SILENT_MODE:
        return
    _original_print(*args, **kwargs)

builtins.print = quiet_print
# ---------------------------------------------

def _parse_offset(val) -> int:
    """Parse offset/size from JSON: int or hex string like '0x24000000'."""
    if isinstance(val, str):
        return int(val, 0)
    return int(val)

_FP4_E2M1_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                                  -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.bfloat16)

def _quantize_bf16_to_fp4_packed(weight_bf16: torch.Tensor, block_size: int = 64) -> tuple[bytes, bytes]:
    """Quantize bf16 weight (N_w, K_w) to FP4 E2M1 packed + scale per block of 64 along K. Returns (data_bytes, scale_bytes)."""
    N_w, K_w = weight_bf16.shape
    assert K_w % block_size == 0
    w = weight_bf16.detach().cpu().to(torch.bfloat16).reshape(N_w, K_w // block_size, block_size)
    fp4_max = torch.tensor(6.0, dtype=torch.bfloat16)
    scale = (w.abs().amax(dim=-1).clamp(min=1e-8) / fp4_max).to(torch.bfloat16)
    scaled = (w / scale.unsqueeze(-1)).to(torch.bfloat16)
    codes = torch.argmin(torch.abs(scaled.unsqueeze(-1) - _FP4_E2M1_TABLE), dim=-1).to(torch.uint8)
    codes_np = codes.numpy().reshape(N_w, -1)
    low = codes_np[:, 0::2] & 0x0F
    high = codes_np[:, 1::2] & 0x0F
    packed = (high << 4) | low
    data_bytes = packed.astype(np.uint8).tobytes()
    scale_bytes = scale.contiguous().view(torch.uint8).numpy().tobytes()
    return (data_bytes, scale_bytes)

def _quantize_bf16_to_int4_packed(weight_bf16: torch.Tensor, block_size: int = 64) -> tuple[bytes, bytes]:
    """Quantize bf16 weight (N_w, K_w) to INT4 packed + scale per block of 64 along K. Returns (data_bytes, scale_bytes)."""
    w = weight_bf16.detach().cpu().float().reshape(-1)
    N_w, K_w = weight_bf16.shape
    assert K_w % block_size == 0
    w_blocks = w.reshape(N_w, K_w // block_size, block_size)
    scale = w_blocks.abs().amax(dim=-1).clamp(min=1e-8) / 7.0
    scale_bf16 = scale.to(torch.bfloat16)
    w_int8 = (w_blocks / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8)
    w_nibbles = w_int8.numpy().astype(np.int16) & 0x0F
    low = w_nibbles[:, :, 0::2].reshape(N_w, -1)
    high = w_nibbles[:, :, 1::2].reshape(N_w, -1)
    packed = (high << 4) | low
    data_bytes = packed.astype(np.uint8).tobytes()
    scale_bytes = scale_bf16.contiguous().view(torch.uint8).numpy().tobytes()
    return (data_bytes, scale_bytes)

def _host_rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS norm on host (for per-layer projection norm)."""
    x_f = x.float()
    rms = (x_f ** 2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return ((x_f / rms) * gamma.float()).to(x.dtype)

def weight_bin_generate(output_path: str | None = None, config_path: str | None = None) -> str:
    """Generate full_model_weights.bin from Hugging Face model per gemma4_e2b_config.json layout.
    Returns the path to the written file."""
    cfg = Gemma4_UnifiedEngine.load_config(config_path=config_path, script_dir=SCRIPT_DIR)
    weight_defs = cfg["_weight_defs"]
    paths = cfg["paths"]
    paths_full = os.path.join(SCRIPT_DIR, paths["weights_bin"])
    out_path = output_path or paths_full

    model, model_dir = _ensure_hf_model(SCRIPT_DIR, cfg)
    text_model = model.model.language_model
    gamma_offset = cfg["special"]["rms_norm"]["gamma_offset"]
    emb_cfg = cfg["special"]["embedding"]
    token_embd_offset = _parse_offset(emb_cfg["token_embd_offset"])
    token_embd_size = _parse_offset(emb_cfg["token_embd_size"])
    LAYER_WEIGHT_SIZE = weight_defs["LAYER_WEIGHT_SIZE"]
    base_layer0 = weight_defs["BLK0_ATTN_NORM_WEIGHT"]
    num_layers = cfg["file_info"]["num_layers"]
    head_dim = cfg["file_info"]["head_dim"]  # 512 (max / full attention)
    head_dim_sliding = cfg["file_info"]["head_dim_sliding"]  # 256
    hidden_size = cfg["file_info"]["hidden_size"]
    group_size = cfg["file_info"]["group_size"]
    full_attention_layers = set(cfg["model"]["full_attention_layers"])
    mlp_elements_wide = cfg["file_info"].get("mlp_elements_wide", cfg["file_info"]["mlp_elements"])
    blk0_structure = cfg["layers"]["structure"]

    # Compute total file size: max(offset + size) over all regions (layer regions use last layer)
    max_end = 0
    for key, r in cfg.get("regions", {}).items():
        off = weight_defs[key]
        max_end = max(max_end, off + (num_layers - 1) * LAYER_WEIGHT_SIZE + r["size"])
    for key, r in cfg.get("non_layer_regions", {}).items():
        off = weight_defs[key]
        max_end = max(max_end, off + r["size"])
    max_end = max(max_end, token_embd_offset + token_embd_size)
    buf = bytearray(max_end)

    def write_at(offset: int, data: bytes) -> None:
        buf[offset : offset + len(data)] = data[: len(buf) - offset]

    # Embedding: scale by sqrt(hidden_size)
    embed = text_model.embed_tokens.weight.detach().cpu().to(torch.bfloat16)
    embedding_scale = hidden_size ** 0.5
    emb_scaled = (embed.float() * embedding_scale).to(torch.bfloat16)
    raw_emb = emb_scaled.contiguous().view(torch.uint8).numpy().tobytes()
    write_at(token_embd_offset, raw_emb)

    # Layers
    for layer_idx in range(num_layers):
        layer = text_model.layers[layer_idx]
        attn = layer.self_attn
        is_full = layer_idx in full_attention_layers
        cur_head_dim = head_dim if is_full else head_dim_sliding
        cur_q_size = cur_head_dim * group_size
        cur_k_size = cur_head_dim

        gamma_in = (layer.input_layernorm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)

        # Q/K/V weights: actual sizes differ per layer, zero-pad to max (full attention) sizes
        q_w_actual = attn.q_proj.weight.detach().cpu().to(torch.bfloat16)  # [cur_q_size, hidden_size]
        k_w_actual = attn.k_proj.weight.detach().cpu().to(torch.bfloat16)  # [cur_k_size, hidden_size]
        v_w_actual = attn.v_proj.weight.detach().cpu().to(torch.bfloat16)  # [cur_k_size, hidden_size]
        o_w_actual = attn.o_proj.weight.detach().cpu().to(torch.bfloat16)  # [hidden_size, cur_q_size]

        # Pad Q/K/V rows to max sizes (N dimension padding — contiguous rows, safe for sub-N matmul).
        # O weight: do NOT pad K dimension — quantize with actual K so scale/data blocks align correctly.
        max_q_size = head_dim * group_size  # 4096
        max_k_size = head_dim  # 512
        q_w = torch.zeros(max_q_size, hidden_size, dtype=torch.bfloat16)
        q_w[:cur_q_size, :] = q_w_actual
        k_w = torch.zeros(max_k_size, hidden_size, dtype=torch.bfloat16)
        k_w[:cur_k_size, :] = k_w_actual
        v_w = torch.zeros(max_k_size, hidden_size, dtype=torch.bfloat16)
        v_w[:cur_k_size, :] = v_w_actual
        # O weight: use actual dimensions (no column padding) to keep INT4 scale blocks aligned
        o_w = o_w_actual  # [hidden_size, cur_q_size]

        # Q/K norm: pad to max head_dim
        gamma_q_actual = (attn.q_norm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
        gamma_k_actual = (attn.k_norm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
        gamma_q = torch.ones(head_dim, dtype=torch.bfloat16)  # default 1.0 (gamma_offset already applied)
        gamma_q[:cur_head_dim] = gamma_q_actual[:cur_head_dim]
        gamma_k = torch.ones(head_dim, dtype=torch.bfloat16)
        gamma_k[:cur_head_dim] = gamma_k_actual

        gamma_post = (layer.post_attention_layernorm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
        gamma_ffn = (layer.pre_feedforward_layernorm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
        gate_w_actual = layer.mlp.gate_proj.weight.detach().cpu().to(torch.bfloat16)
        up_w_actual = layer.mlp.up_proj.weight.detach().cpu().to(torch.bfloat16)
        down_w = layer.mlp.down_proj.weight.detach().cpu().to(torch.bfloat16)
        # Pad gate/up rows to max MLP width (N dimension padding, safe for sub-N matmul)
        cur_mlp = gate_w_actual.shape[0]
        gate_w = torch.zeros(mlp_elements_wide, hidden_size, dtype=torch.bfloat16)
        gate_w[:cur_mlp, :] = gate_w_actual
        up_w = torch.zeros(mlp_elements_wide, hidden_size, dtype=torch.bfloat16)
        up_w[:cur_mlp, :] = up_w_actual
        # Down weight: use actual K (no padding) — quantize as-is so scale/data blocks align
        gamma_post_ffn = (layer.post_feedforward_layernorm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)

        # Per-layer weights (BF16, no quantization)
        per_layer_input_gate_w = layer.per_layer_input_gate.weight.detach().cpu().to(torch.bfloat16)  # [256, 1536]
        per_layer_projection_w = layer.per_layer_projection.weight.detach().cpu().to(torch.bfloat16)  # [1536, 256]
        gamma_post_per_layer_input_norm = (layer.post_per_layer_input_norm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)  # [1536]
        layer_scalar_val = layer.layer_scalar.detach().cpu().to(torch.bfloat16)  # scalar
        # Pad layer_scalar to 64 bytes (32 bf16 elements)
        layer_scalar_tensor = torch.zeros(32, dtype=torch.bfloat16)
        layer_scalar_tensor[0] = layer_scalar_val

        # O weight: quantize with actual K (unpadded) so INT4 scale/data stride matches K
        region_writes = [
            (gamma_in, "bf16"),
            (q_w, "int4"),
            (k_w, "int4"),
            (v_w, "int4"),
            (gamma_q, "bf16"),
            (gamma_k, "bf16"),
            (o_w, "int4"),  # [hidden_size, cur_q_size] — actual dimensions
            (gamma_post, "bf16"),
            (gamma_ffn, "bf16"),
            (up_w, "int4"),
            (gate_w, "int4"),
            (down_w, "int4"),
            (gamma_post_ffn, "bf16"),
            (per_layer_input_gate_w, "bf16"),
            (per_layer_projection_w, "bf16"),
            (gamma_post_per_layer_input_norm, "bf16"),
            (layer_scalar_tensor, "bf16"),
        ]
        j = 0
        i = 0
        while i < len(blk0_structure):
            off_key = blk0_structure[i]["key"]
            sz_key = f"{off_key}_SIZE"
            off = weight_defs[off_key]
            sz = weight_defs[sz_key]
            file_off = off + layer_idx * LAYER_WEIGHT_SIZE
            tensor, kind = region_writes[j]
            if kind == "int4":
                next_key = blk0_structure[i + 1]["key"]
                data_sz = weight_defs[f"{next_key}_SIZE"]
                data_bytes, scale_bytes = _quantize_bf16_to_fp4_packed(tensor)
                scale_padded = (scale_bytes + b"\x00" * sz)[:sz]
                data_padded = (data_bytes + b"\x00" * data_sz)[:data_sz]
                write_at(file_off, scale_padded)
                data_off = weight_defs[next_key] + layer_idx * LAYER_WEIGHT_SIZE
                write_at(data_off, data_padded)
                i += 2
            else:
                t = tensor.detach().cpu().to(torch.bfloat16).contiguous()
                raw = (t.view(torch.uint8).numpy().tobytes() + b"\x00" * sz)[:sz]
                write_at(file_off, raw)
                i += 1
            j += 1

    # ROPE: two tables with different dimensions
    rope_cfg = cfg["special"]["rope"]
    theta = rope_cfg["theta"]
    local_base = rope_cfg["local_base"]
    num_positions = rope_cfg["num_positions"]
    partial_rotary_factor = rope_cfg["partial_rotary_factor_global"]

    # LOCAL RoPE: head_dim=256, full rotation, D=128
    D_local = head_dim_sliding // 2  # 128
    inv_freq_local = 1.0 / (local_base ** (torch.arange(D_local, dtype=torch.float32) / D_local))
    pos = torch.arange(num_positions, dtype=torch.float32)
    freqs_local = torch.outer(pos, inv_freq_local)
    cos_local = freqs_local.cos().to(torch.bfloat16)
    sin_local = freqs_local.sin().to(torch.bfloat16)
    rope_local = torch.cat([cos_local, cos_local, -sin_local, sin_local], dim=1)
    sz = weight_defs["ROPE_LOCAL_SIZE"]
    raw = (rope_local.contiguous().view(torch.uint8).numpy().tobytes() + b"\x00" * sz)[:sz]
    write_at(weight_defs["ROPE_LOCAL"], raw)

    # GLOBAL RoPE: head_dim=512, partial_rotary_factor=0.25, rotary_dims=128, D=64
    rotary_dims = int(head_dim * partial_rotary_factor)  # 128
    D_global = rotary_dims // 2  # 64
    inv_freq_global = 1.0 / (theta ** (torch.arange(D_global, dtype=torch.float32) / D_global))
    freqs_global = torch.outer(pos, inv_freq_global)
    cos_global = freqs_global.cos().to(torch.bfloat16)
    sin_global = freqs_global.sin().to(torch.bfloat16)
    rope_global = torch.cat([cos_global, cos_global, -sin_global, sin_global], dim=1)
    sz = weight_defs["ROPE_GLOBAL_SIZE"]
    raw = (rope_global.contiguous().view(torch.uint8).numpy().tobytes() + b"\x00" * sz)[:sz]
    write_at(weight_defs["ROPE_GLOBAL"], raw)

    # OUTPUT_NORM
    out_norm = text_model.norm.weight.detach().cpu().to(torch.bfloat16)
    sz = weight_defs["OUTPUT_NORM_WEIGHT_SIZE"]
    raw = (out_norm.contiguous().view(torch.uint8).numpy().tobytes() + b"\x00" * sz)[:sz]
    write_at(weight_defs["OUTPUT_NORM_WEIGHT"], raw)

    # PER_LAYER_MODEL_PROJ_WEIGHT: [1536, 8960] from model.model.language_model.per_layer_model_projection
    per_layer_model_proj_w = text_model.per_layer_model_projection.weight.detach().cpu().to(torch.bfloat16)
    sz = weight_defs["PER_LAYER_MODEL_PROJ_WEIGHT_SIZE"]
    raw = (per_layer_model_proj_w.contiguous().view(torch.uint8).numpy().tobytes() + b"\x00" * sz)[:sz]
    write_at(weight_defs["PER_LAYER_MODEL_PROJ_WEIGHT"], raw)

    # PER_LAYER_PROJ_NORM_WEIGHT: [256] from model.model.language_model.per_layer_projection_norm
    per_layer_proj_norm_w = (text_model.per_layer_projection_norm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
    sz = weight_defs["PER_LAYER_PROJ_NORM_WEIGHT_SIZE"]
    raw = (per_layer_proj_norm_w.contiguous().view(torch.uint8).numpy().tobytes() + b"\x00" * sz)[:sz]
    write_at(weight_defs["PER_LAYER_PROJ_NORM_WEIGHT"], raw)

    # LM_HEAD: tied with embed_tokens, so clone
    lm_head_w = model.lm_head.weight.detach().clone().cpu().to(torch.bfloat16)
    scale_sz = weight_defs["LM_HEAD_WEIGHT_SCALE_SIZE"]
    data_sz = weight_defs["LM_HEAD_WEIGHT_DATA_SIZE"]
    data_bytes, scale_bytes = _quantize_bf16_to_fp4_packed(lm_head_w)
    scale_padded = (scale_bytes + b"\x00" * scale_sz)[:scale_sz]
    data_padded = (data_bytes + b"\x00" * data_sz)[:data_sz]
    write_at(weight_defs["LM_HEAD_WEIGHT_SCALE"], scale_padded)
    write_at(weight_defs["LM_HEAD_WEIGHT_DATA"], data_padded)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(buf)
    print(f"Generated weights bin: {out_path} ({len(buf)} bytes)")
    return out_path

def _ensure_hf_model(script_dir: str, cfg: dict):
    """Ensure HF model is downloaded and loaded. Returns (model, model_dir). Single place for download + load."""
    model_dir = os.path.join(script_dir, cfg["paths"]["hf_model_dir"])
    hf_repo = cfg["paths"]["hf_model_repo"]
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        _original_print(f"Downloading HF model {hf_repo} to {os.path.abspath(model_dir)} ...")
        snapshot_download(repo_id=hf_repo, local_dir=model_dir, local_dir_use_symlinks=False)
        _original_print("Download complete.")
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map=None, trust_remote_code=True
    )
    return model, model_dir

# -----------------------------------------------------------------------------
# Gemma4 E2B unified engine
# -----------------------------------------------------------------------------
class Gemma4_UnifiedEngine(UnifiedEngine):
    """UnifiedEngine with Gemma4 E2B dims: loads config + weight bin, compile_prefill/compile_decoder, run_prefill/run_decoder. Numeric checks in gemma4_e2b_numeric.py."""

    def __init__(self, script_dir: str | None = None, local_weights: bool = False, dual_engine: bool = False, engine_slave: bool = False):
        engine_base = user_dma_core.UE_0_BASE_ADDR + 0x00010000 if engine_slave else user_dma_core.UE_0_BASE_ADDR
        # Compute DRAM layout to avoid overlaps:
        #   [0x80000000 ... params_end] = weights
        #   [tensor_base ... tensor_end] = tensors (~170 MB)
        #   [program_base ... ] = instruction programs (~50 MB max)
        _script_dir = script_dir or SCRIPT_DIR
        _cfg_tmp = self.load_config(script_dir=_script_dir)
        _fi = _cfg_tmp["file_info"]
        _layers_sz = _fi["num_layers"] * _fi["layer_size"]
        _non_layer_sz = sum(r["size"] for r in _cfg_tmp["non_layer_regions"].values())
        _params_estimate = _layers_sz + _non_layer_sz + 0x100000  # +1MB margin for RoPE/identity
        _tensor_base = ((0x80000000 + _params_estimate + 0xFFFFF) // 0x100000) * 0x100000  # 1MB aligned
        # Tensor DRAM budget must cover KV cache + activations + flash scratch +
        # flash bias buffers. Flash bias is the dominant cost: sized for
        # max_prefill_seq_len * group_size (see tensor_init). At
        # max_prefill_seq_len=512 and group_size=8, total tensor DRAM fits in
        # ~256 MB. We reserve 288 MB to leave headroom.
        _tensor_estimate = 0x12000000  # 288 MB
        _program_base = ((_tensor_base + _tensor_estimate + 0xFFFFF) // 0x100000) * 0x100000  # 1MB aligned
        if engine_slave:
            _program_base += 0x10000000
        super().__init__(BASE_ADDR=engine_base, program_dram_base=_program_base, tensor_dram_base=_tensor_base)
        self.dual_engine = dual_engine
        self.script_dir = script_dir or SCRIPT_DIR
        self._cfg = self.load_config(script_dir=self.script_dir)
        self.weight_defs = self._cfg["_weight_defs"]

        fi = self._cfg["file_info"]
        model = self._cfg["model"]
        paths = self._cfg["paths"]

        self.vector_length = fi["hidden_size"]
        self.head_dim = fi["head_dim"]  # 512 (max, for uniform sizing)
        self.head_dim_sliding = fi["head_dim_sliding"]  # 256
        self.per_layer_input_dim = fi["per_layer_input_dim"]  # 256
        self.bytes_per_element = fi["bytes_per_element"]
        self.group_size = fi["group_size"]
        self.mlp_elements = fi["mlp_elements"]
        self.mlp_elements_wide = fi.get("mlp_elements_wide", fi["mlp_elements"])
        self.q_size = self.head_dim * self.group_size * self.bytes_per_element
        self.k_size = self.head_dim * self.bytes_per_element
        self.MAX_CONTEXT_SIZE = model["max_context_size"]
        self.LAYER_SIZE = fi["num_layers"]
        self.EMBEDDING_ELEMENTS = fi["embedding_vocab"]
        self._isa_reg_counter = 1
        fixed = self._cfg.get("fixed_isa_regs", {})
        self.V_CACHE_SIZE_REG = fixed["V_CACHE_SIZE_REG"]
        self.TMP_REG = fixed["TMP_REG"]
        self.ROPE_SIZE_REG = fixed["ROPE_SIZE_REG"]  # for sliding layers (rope_row = head_dim_sliding * 2 * bpe)
        self.ROPE_GLOBAL_SIZE_REG = fixed.get("ROPE_GLOBAL_SIZE_REG", self.ROPE_SIZE_REG)  # for full attn layers
        self.causal_mask_upper = False
        self._rope_global_layers = set(model["rope_global_layers"])
        self._full_attention_layers = set(model["full_attention_layers"])
        self._double_wide_mlp_first = model.get("double_wide_mlp_first_layer", fi["num_layers"])
        self._end_of_turn_token_id = model["end_of_turn_token_id"]
        # Sliding-attention window. Sliding layers (i.e. layers NOT in
        # full_attention_layers) are limited to attending the last
        # `sliding_window` tokens. Default to MAX_CONTEXT_SIZE so older configs
        # without this key keep their old behaviour (no real windowing).
        self.sliding_window = model.get("sliding_window", model["max_context_size"])
        # max_prefill_seq_len caps the flash attention scratch/bias buffer sizes
        # independently of max_context_size. Flash buffers are only used at full
        # size during prefill (for the whole prompt); decode uses 1 row. Sizing
        # them for max_context_size wastes ~200+ MB of DRAM that the decoder
        # program bin later tries to occupy, causing overlap.
        self.max_prefill_seq_len = model.get("max_prefill_seq_len", model["max_context_size"])
        # KV sharing map: built from HF model during weight_init
        self._kv_shared_map = {}  # layer_idx -> reference_layer_idx (populated in weight_init)
        self._gamma_bin_offset = self._cfg["special"]["rms_norm"]["gamma_offset"]
        self._per_layer_model_proj_scale = model["per_layer_model_proj_scale"]
        self._per_layer_input_scale = model["per_layer_input_scale"]
        self.prefill_seq = None
        self.engine_slave = engine_slave
        # Identity matrix cache (fix for flash_attention_core DRAM leak)
        self._identity_dram_addr = None
        self._identity_dram_written = False

        self._weights_bin_rel = "gemma4_e2b_bin/full_model_weights.bin" if local_weights else paths["weights_bin"]
        self.weight_init()
        self.tensor_init()
        self._preallocate_identity_matrix()

    # ─── Identity matrix cache (fix for flash_attention_core DRAM leak) ─────
    _IDENTITY_MAT_BYTES = UE_VECTOR_SIZE * UE_VECTOR_SIZE * 2  # 8192

    def _preallocate_identity_matrix(self) -> None:
        if self._identity_dram_addr is not None:
            return
        self._identity_dram_addr = self.allocate_params_dram(self._IDENTITY_MAT_BYTES)
        eye = torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16)
        super().dma_write(DMA_DEVICE_H2C, self._identity_dram_addr, eye, self._IDENTITY_MAT_BYTES)
        self._identity_dram_written = True

    def _flash_attention_core_cached(self, **kwargs) -> int:
        saved = self._next_params_dram_addr
        self._next_params_dram_addr = self._identity_dram_addr
        result = self.flash_attention_core(**kwargs)
        self._next_params_dram_addr = saved
        return result

    def dma_write(self, device, addr, data, size):
        if (getattr(self, '_identity_dram_written', False)
                and self._identity_dram_addr is not None
                and addr == self._identity_dram_addr
                and size == self._IDENTITY_MAT_BYTES):
            return
        super().dma_write(device, addr, data, size)

    def _emit_sram_eltwise_chunked(self, kind: str,
                                    addr_A: int, addr_B: int, addr_out: int,
                                    num_elements: int,
                                    chunk: int = 131072) -> None:
        """Emit chunked eltwise_add/eltwise_mul instructions into the current
        capture buffer.

        URAM A (0x10000..0x90000) and URAM B (0x90000..0x110000) each hold
        0x80000 bytes = 262144 bf16 elements. A single unchunked eltwise_*
        therefore crashes into the other buffer once num_elements exceeds
        262144 (for Gemma4 VLM prefill this happens at ~43 image tokens with
        cur_mlp=6144, ~22 tokens with cur_mlp=12288, and ~170 tokens with
        vector_length=1536). Chunking to 131072 elements per call is a safe,
        uniform limit that leaves room for both operands in the two buffers.

        Intended for use INSIDE compile_prefill / compile_decoder, which
        capture all instructions in one program (unlike the compare script's
        _run_eltwise_*_chunked which compiles a fresh program per chunk).
        """
        bpe = self.bytes_per_element
        for off in range(0, num_elements, chunk):
            n = min(chunk, num_elements - off)
            self.accelerator_memory_to_sram(
                accelerator_dram_address=addr_A + off * bpe,
                sram_address=0x10000, element_size=n)
            self.accelerator_memory_to_sram(
                accelerator_dram_address=addr_B + off * bpe,
                sram_address=0x90000, element_size=n)
            if kind == "add":
                self.eltwise_add_core(
                    vector_A_sram_start_addr=0x10000,
                    vector_B_sram_start_addr=0x90000,
                    vector_C_sram_wb_addr=0x10000,
                    element_size=n)
            elif kind == "mul":
                self.eltwise_mul_core(
                    vector_A_sram_start_addr=0x10000,
                    vector_B_sram_start_addr=0x90000,
                    vector_C_sram_wb_addr=0x10000,
                    element_size=n)
            else:
                raise ValueError(f"unknown eltwise kind: {kind!r}")
            self.sram_to_accelerator_memory(
                sram_address=0x10000,
                accelerator_dram_address=addr_out + off * bpe,
                element_size=n)

    def _emit_sram_copy_chunked(self, src_addr: int, dst_addr: int,
                                 num_elements: int, chunk: int = 131072) -> None:
        """Emit a chunked DRAM→SRAM→DRAM copy into the capture buffer. Used to
        shuttle large activations when a direct DRAM-to-DRAM path isn't handy."""
        bpe = self.bytes_per_element
        for off in range(0, num_elements, chunk):
            n = min(chunk, num_elements - off)
            self.accelerator_memory_to_sram(
                accelerator_dram_address=src_addr + off * bpe,
                sram_address=0x10000, element_size=n)
            self.sram_to_accelerator_memory(
                sram_address=0x10000,
                accelerator_dram_address=dst_addr + off * bpe,
                element_size=n)

    def _emit_sram_broadcast_mul_chunked(self, src_addr: int, dst_addr: int,
                                          num_elements: int, scalar: float,
                                          chunk: int = 131072) -> None:
        """Emit a chunked in-place broadcast_mul (scalar multiply) through
        SRAM into the capture buffer."""
        bpe = self.bytes_per_element
        for off in range(0, num_elements, chunk):
            n = min(chunk, num_elements - off)
            self.accelerator_memory_to_sram(
                accelerator_dram_address=src_addr + off * bpe,
                sram_address=0x10000, element_size=n)
            self.broadcast_mul(scalar=scalar,
                               sram_start_addr=0x10000,
                               sram_wb_addr=0x10000,
                               element_size=n)
            self.sram_to_accelerator_memory(
                sram_address=0x10000,
                accelerator_dram_address=dst_addr + off * bpe,
                element_size=n)

    @staticmethod
    def load_config(config_path: str | None = None, script_dir: str | None = None) -> dict:
        """Load gemma4_e2b_config.json and build weight_defs (offset/size dict) from regions."""
        if config_path is None:
            script_dir = script_dir or SCRIPT_DIR
            config_path = os.path.join(script_dir, "gemma4_e2b_config.json")
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

    def set_prefill_seq(self, prompt: str | None = None) -> None:
        """Set self.prefill_seq from a text prompt (tokenize with chat template) or from config default."""
        if prompt is not None:
            conversation = [{"role": "user", "content": prompt}]
            prompt_with_template = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            self.prefill_seq = tuple(self.tokenizer.encode(prompt_with_template, add_special_tokens=True))
            print(f"Prefill from prompt ({len(self.prefill_seq)} tokens): {prompt!r}")
            print(f"Sequence ids: {self.prefill_seq}")
        else:
            self.prefill_seq = tuple(self._cfg["default_prefill_tokens"])
            decoded = self.tokenizer.decode(list(self.prefill_seq), skip_special_tokens=True)
            print(f"Prefill from default ({len(self.prefill_seq)} tokens): {decoded!r}")
            print(f"Sequence ids: {self.prefill_seq}")

    def _run_vision_encoder_fpga(self, image_path: str, prompt: str) -> tuple[torch.Tensor, list[int], list[int]]:
        """Run the full vision encoder on FPGA: patch embedder → 16 encoder
        layers → pooler → embed_vision projection. Returns
        (image_features, token_ids, mm_types).

        DRAM-layout note: vision reuses the LM tensor DRAM region via
        vision_tensor_init (same as the compare script). This clobbers
        IDENTITY_DRAM_ADDR, the KV cache zero-pad, and the LM flash scratch
        buffers that the LM decoder relies on — so we restore IDENTITY_DRAM
        and re-run the LM tensor zero-pads at the end. Prefill re-uploads
        its own input/bias tensors in run_prefill so they don't need to be
        preserved across this call.
        """
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForImageTextToText

        # Load HF model on CPU (no GPU move) — only used for weight upload
        # and for processor; all compute runs on the FPGA.
        hf_model, model_dir = _ensure_hf_model(self.script_dir, self._cfg)
        processor = AutoProcessor.from_pretrained(model_dir)
        hf_model.eval()

        # Tokenize + patch extraction on host via the processor.
        image = Image.open(image_path).convert("RGB")
        conversation = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text_prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text_prompt], images=[[image]], return_tensors="pt")

        pixel_values = inputs['pixel_values']                  # [1, S_patches, 768]
        pixel_position_ids = inputs['image_position_ids']       # [1, S_patches, 2]
        padding_positions = (pixel_position_ids == -1).all(dim=-1)  # [1, S_patches]
        num_patches = pixel_values.shape[1]

        # Save LM allocator state so we can restore it after vision finishes.
        _tensor_dram_save = self._tensor_dram_addr
        _prog_dram_addr_save = self._next_program_dram_addr
        _prog_dram_base_save = self._program_dram_base

        # Choose a safe program_base for vision ops. Vision weights go
        # immediately after LM tensors (at _tensor_dram_save); vision programs
        # must not overlap either those weights OR the LM weights region.
        # Put them 100 MB past the current tensor cursor — comfortably past
        # vision weights (~80 MB) and well above LM weights (0x80000000..).
        vision_prog_base = _tensor_dram_save + (100 * 1024 * 1024)

        # ---- FPGA vision ----
        self.vision_weight_init(hf_model)
        self.vision_tensor_init(num_patches, program_base=vision_prog_base)
        self.set_vision_attention_bias(padding_positions)

        # RoPE tables (2D, per image) — compute via the HF rotary module then
        # DMA the cos/sin tables into VIS_ROPE_COS/SIN.
        with torch.no_grad():
            vt = hf_model.model.vision_tower
            patch_embeds_hf = vt.patch_embedder(pixel_values, pixel_position_ids, padding_positions)
            cos_table, sin_table = vt.encoder.rotary_emb(patch_embeds_hf, pixel_position_ids)
        cos_2d = cos_table.squeeze(0).cpu().to(torch.bfloat16)
        sin_2d = sin_table.squeeze(0).cpu().to(torch.bfloat16)
        self.dma_to_accelerator_memory(self.VIS_ROPE_COS, cos_2d)
        self.dma_to_accelerator_memory(self.VIS_ROPE_SIN, sin_2d)

        # Patch embedder on FPGA → VIS_IO_A
        self.vision_patch_embed(pixel_values.cpu(), pixel_position_ids.cpu(),
                                 padding_positions.cpu())

        # 16 encoder layers on FPGA
        final_buf = self.VIS_IO_A
        for li in range(self.VIS_LAYERS):
            final_buf = self.run_vision_layer(li, cos_2d, sin_2d)
        encoder_out = self.dma_from_accelerator_memory(
            final_buf, (num_patches, self.VIS_H)).cpu()

        # Pooler + embed_vision tail on FPGA → [N_soft, 1536]
        image_features = self.vision_embed_project(
            encoder_out, pixel_position_ids.cpu(), padding_positions.cpu())
        image_features = image_features.to(torch.bfloat16)

        # ---- Restore LM state so decoder can run ----
        # Restore allocator pointers. Vision tensors clobbered the LM tensor
        # DRAM region (KV cache + flash buffers + IDENTITY), so we re-DMA the
        # pieces the LM decoder reads without re-writing first: IDENTITY (used
        # by decoder_attention_core) and the KV cache (positions beyond the
        # current seq_len are masked by the bias, but must contain safe
        # values — not NaN/Inf from vision intermediates — or bf16 Q·K will
        # overflow to NaN and destroy the softmax).
        self._tensor_dram_addr = _tensor_dram_save
        self._next_program_dram_addr = _prog_dram_addr_save
        self._program_dram_base = _prog_dram_base_save
        # Re-zero the KV cache slots.
        # NOTE: self.k_size is in BYTES (head_dim * bytes_per_element), but
        # torch.zeros(N, dtype=bfloat16) takes N as element count. Using
        # k_size as the element multiplier produces a tensor that's 2× the
        # actual KV slot size and overflows the next buffer. tensor_init has
        # the same bug but it's masked by subsequent initializations of the
        # buffers it overflows into. We must size correctly here because
        # we re-upload IDENTITY_DRAM right after and don't want to clobber it.
        kv_slot_elems = self._num_kv_slots * self.MAX_CONTEXT_SIZE * self.head_dim
        kv_zero_pad = torch.zeros(kv_slot_elems, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, kv_zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_K_ROPE_DRAM, kv_zero_pad)
        # Re-DMA identity matrix AFTER the KV zero-pads so it can't be
        # overwritten by overflow. decoder_attention_core reads this at
        # Stage 1 (I @ V^T); zeroing it produces all-zero attention output.
        self.dma_to_accelerator_memory(
            self.IDENTITY_DRAM_ADDR,
            torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))

        token_ids = inputs['input_ids'][0].tolist()
        mm_types = inputs['mm_token_type_ids'][0].tolist()

        del hf_model

        print(f"  [Vision] FPGA path complete: {num_patches} patches → "
              f"{image_features.shape[0]} soft tokens, "
              f"image_features {tuple(image_features.shape)}")
        return image_features, token_ids, mm_types

    def _run_audio_encoder_fpga(self, audio_path: str, prompt: str) -> tuple[torch.Tensor, list[int], list[int]]:
        """Run the full audio encoder on FPGA: feature extractor (host) →
        subsample stem (host) → 12 Conformer layers (FPGA) → output_proj +
        multimodal embedder (host) → audio soft tokens.

        DRAM-layout note: same pattern as _run_vision_encoder_fpga. Audio
        weights go after LM tensors; audio intermediates clobber LM tensor
        DRAM (KV cache, IDENTITY, flash buffers) which we re-zero / re-upload
        before returning. Audio programs go 100 MB past the LM tensor cursor
        so they don't overlap LM tensors or LM weights.

        Returns (audio_features [N_soft, 1536], token_ids, mm_types), where
        mm_types uses 3 at audio-soft-token positions (HF convention).
        """
        from transformers import AutoProcessor, AutoModelForImageTextToText

        # Load HF model for weights + processor for tokenization.
        hf_model, model_dir = _ensure_hf_model(self.script_dir, self._cfg)
        processor = AutoProcessor.from_pretrained(model_dir)
        hf_model.eval()

        # Load audio file with the feature extractor.
        try:
            import soundfile as sf
        except ImportError as e:
            raise RuntimeError(
                "soundfile is required for audio input. `pip install soundfile`.") from e
        audio_array, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=-1)  # mono
        target_sr = getattr(processor.feature_extractor, "sampling_rate", 16000)
        if sr != target_sr:
            t = torch.from_numpy(audio_array).float().unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t,
                              size=int(audio_array.shape[0] * target_sr / sr),
                              mode="linear", align_corners=False)
            audio_array = t.squeeze().numpy()
        print(f"  [Audio] loaded {audio_path}: {audio_array.shape[0]/target_sr:.2f}s @ {target_sr} Hz")

        # Build the conversation and tokenize with audio placeholders.
        conversation = [{"role": "user", "content": [
            {"type": "audio"},
            {"type": "text", "text": prompt},
        ]}]
        text_prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(
            text=[text_prompt],
            audio=[audio_array],
            sampling_rate=target_sr,
            return_tensors="pt",
        )
        input_features = inputs["input_features"]
        input_features_mask = inputs.get("input_features_mask")
        token_ids = inputs["input_ids"][0].tolist()
        mm_types = inputs["mm_token_type_ids"][0].tolist()
        print(f"  [Audio] {len(token_ids)} prompt tokens "
              f"({sum(1 for m in mm_types if m == 3)} audio, "
              f"{sum(1 for m in mm_types if m == 0)} text)")

        # Save LM allocator state so we can restore it after audio finishes.
        _tensor_dram_save = self._tensor_dram_addr
        _prog_dram_addr_save = self._next_program_dram_addr
        _prog_dram_base_save = self._program_dram_base

        # Audio program DRAM: 200 MB past current tensor cursor. Audio weight
        # uploads alone consume ~160 MB in FP4 q4_64, so the 100 MB cushion
        # vision uses isn't quite enough — we need 200 MB to keep the audio
        # programs clear of the weight region and well above LM weights (<0xE0500000).
        audio_prog_base = _tensor_dram_save + (200 * 1024 * 1024)

        # ---- FPGA audio ----
        # reset_allocator=False: keep LM tensor DRAM intact; audio weights go
        # immediately after LM tensor's current end.
        self.audio_weight_init(hf_model, reset_allocator=False)

        # audio_tensor_init allocates intermediates AFTER weights (current
        # cursor). Intermediates clobber nothing yet because we're past LM
        # tensors. But we also need to set _program_dram_base / _next_program
        # so captured audio instructions go to audio_prog_base, not into the
        # LM program area that's already in use.
        self._next_program_dram_addr = audio_prog_base
        self._program_dram_base = audio_prog_base

        # Subsample on host (Conv2d stem is tiny; not worth FPGA-ing).
        sub_hidden, _ = self.audio_subsample_host(input_features, input_features_mask)
        T_sub = int(sub_hidden.shape[1])

        # Allocate intermediates. This starts at current cursor (after audio
        # weights) and grows upward. Total audio footprint (weights + inter)
        # is ~200 MB, which lands around ~0xFE000000 — close to the 4 GB
        # boundary but safe because audio_prog_base is past it (will wrap?
        # no — audio_prog_base < tensor cursor if weights are bigger than
        # 200 MB. We'll allocate programs there during compile, and they
        # don't overlap tensors because program_base = tensor_save + 200 MB
        # and tensor cursor after audio weights ≈ tensor_save + 160 MB).
        self.audio_tensor_init(T_sub)

        # Seed AUD_IO_A with the subsampled hidden state (padded to L_pad).
        L_pad = self._aud_L_pad
        H = self.AUD_H
        seed = torch.zeros(L_pad, H, dtype=torch.bfloat16)
        seed[:T_sub] = sub_hidden[0, :T_sub].to(torch.bfloat16)
        self.dma_to_accelerator_memory(self.AUD_IO_A, seed.contiguous())

        # Run all 12 Conformer layers on FPGA.
        for li in range(self.AUD_LAYERS):
            self.run_audio_layer(li)

        # Read encoder output, then run output_proj + multimodal embedder on host.
        encoder_out = self.dma_from_accelerator_memory(self.AUD_IO_A, (L_pad, H)).cpu()[:T_sub]
        audio_features = self.audio_embed_project_host(encoder_out)
        # Sanity: number of audio soft tokens equals mm_types count of 3s.
        n_audio_slots = sum(1 for m in mm_types if m == 3)
        if audio_features.shape[0] != n_audio_slots:
            print(f"  [Audio] WARNING: encoder produced {audio_features.shape[0]} "
                  f"soft tokens but prompt has {n_audio_slots} audio slots; "
                  f"truncating/padding to fit.")
            if audio_features.shape[0] > n_audio_slots:
                audio_features = audio_features[:n_audio_slots]
            else:
                pad = torch.zeros(n_audio_slots - audio_features.shape[0],
                                  audio_features.shape[1], dtype=torch.bfloat16)
                audio_features = torch.cat([audio_features, pad], dim=0)

        # ---- Restore LM state so decoder can run ----
        # Same story as vision: audio intermediates clobbered LAYER0_V/K_ROPE
        # (KV cache) and may have touched IDENTITY_DRAM. Re-zero KV and
        # re-upload IDENTITY so the LM decoder path is clean.
        self._tensor_dram_addr = _tensor_dram_save
        self._next_program_dram_addr = _prog_dram_addr_save
        self._program_dram_base = _prog_dram_base_save
        kv_slot_elems = self._num_kv_slots * self.MAX_CONTEXT_SIZE * self.head_dim
        kv_zero_pad = torch.zeros(kv_slot_elems, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, kv_zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_K_ROPE_DRAM, kv_zero_pad)
        self.dma_to_accelerator_memory(
            self.IDENTITY_DRAM_ADDR,
            torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))

        del hf_model
        print(f"  [Audio] FPGA path complete: {T_sub} frames → "
              f"{audio_features.shape[0]} soft tokens, "
              f"audio_features {tuple(audio_features.shape)}")
        return audio_features, token_ids, mm_types

    def set_prefill_seq_audio(self, audio_path: str, prompt: str = None) -> None:
        """Set prefill_seq for audio input: run audio encoder on FPGA,
        merge audio features into the LM embedding stream during run_prefill
        at mm_types == 3 positions."""
        if prompt is None:
            prompt = "Describe what you hear."
        audio_features, token_ids, mm_types = self._run_audio_encoder_fpga(audio_path, prompt)
        self.prefill_seq = tuple(token_ids)
        self._audio_features = audio_features  # [N_soft_tokens, 1536]
        self._mm_types = mm_types
        n_audio = (torch.tensor(mm_types) == 3).sum().item()
        n_text = (torch.tensor(mm_types) == 0).sum().item()
        print(f"Audio prefill: {len(token_ids)} tokens ({n_audio} audio, {n_text} text)")
        print(f"Audio features: {tuple(audio_features.shape)}")

    def set_prefill_seq_vlm(self, image_path: str, prompt: str = None) -> None:
        """Set prefill_seq for VLM: run vision encoder on FPGA, merge image
        features into embeddings during run_prefill."""
        if prompt is None:
            prompt = "Describe this image in detail."

        image_features, token_ids, mm_types = self._run_vision_encoder_fpga(image_path, prompt)

        # Store for later use
        self.prefill_seq = tuple(token_ids)
        self._image_features = image_features  # [N_soft_tokens, 1536]
        self._mm_types = mm_types

        print(f"VLM prefill: {len(token_ids)} tokens ({(torch.tensor(mm_types) == 1).sum().item()} image, "
              f"{(torch.tensor(mm_types) == 0).sum().item()} text)")
        print(f"Image features: {image_features.shape}")

    def reset_isa_reg_counter(self) -> None:
        """Reset the ISA register allocation counter to 1 (register 0 is hard-wired zero)."""
        self._isa_reg_counter = 1

    def alloc_isa_reg(self, reset: bool = False) -> int:
        """
        Allocate the next available general-purpose ISA register.

        General-purpose ISA registers are 32 bits wide: regs 0..15.
        Register 0 is a hard-wired zero register, so allocation starts from 1.

        Args:
            reset: If True, reset the counter to 1 before allocation (default: False)

        Returns:
            The allocated register index (1-15)

        Raises:
            ValueError: If all available registers (1-15) have been allocated
        """
        if reset:
            self._isa_reg_counter = 1

        if self._isa_reg_counter > 15:
            raise ValueError("Exceeded available ISA registers (max 15)")

        reg_idx = self._isa_reg_counter
        self._isa_reg_counter += 1
        return reg_idx


    def isa_add_set_core(self, dst_reg_idx: int, immediate_value: int, timeout_s: float = 10.0) -> None:
        """
        Run a minimal program that sets one ISA register to an immediate value (ADD SET then HALT):
        start_capture -> generate_instruction_add_set -> stop_capture -> halt -> write to DRAM -> execute -> wait.
        Use e.g. isa_add_set_core(V_CACHE_SIZE_REG, self.seq_len * self.k_size).
        """
        self.isa_add_set_multi([(dst_reg_idx, immediate_value)], timeout_s=timeout_s)

    def isa_add_set_multi(self, reg_values: list[tuple[int, int]], timeout_s: float = 10.0) -> None:
        """Set multiple ISA registers in one program submission.

        Each capture → DMA → execute → wait cycle costs ~5–10 ms of round-trip
        overhead, so combining 3 register sets per decode token into a single
        program saves two round trips. Order of writes inside the program
        matches the order of `reg_values`.
        """
        self.clear_inst_id()
        self.start_capture()
        for dst_reg_idx, immediate_value in reg_values:
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
        """
        Write all captured instructions to a binary file.

        Args:
            start_addr: DRAM address where instructions are intended to be stored (used for logging/naming if needed)
            filename: Name of the file to write to
        """
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

    # Overwrite UnifiedEngine allocate_params_dram
    def allocate_params_dram(self, size_bytes: int) -> int:
        """
        Allocate memory from the params DRAM region incrementally.

        Args:
            size_bytes: Number of bytes to allocate

        Returns:
            The DRAM address of the allocated block (address before increment).
        """
        params_dram_addr = self._next_params_dram_addr
        self._next_params_dram_addr += size_bytes
        return params_dram_addr

    def clear_inst_id(self) -> None:
        """Reset instruction ID counter for the next capture."""
        self._inst_id = 0

    def get_arg_max_index(self) -> int:
        """Get the arg max index from the Unified Engine"""
        return self.read_reg32(UE_ARGMAX_INDEX)

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
            self.generate_instruction_add_imm(rope_size_reg, cos_dram_addr, tmp_reg)
            self.accelerator_memory_to_sram(accelerator_dram_address=cos_dram_addr, sram_address=sram_cos, element_size=N)
            self.overwrite_instruction_with_general_register(tmp_reg)
            self.generate_instruction_add_imm(rope_size_reg, sin_dram_addr, tmp_reg)
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
            self.generate_instruction_add_imm(output_addr_inc_reg, output_dram_addr, tmp_reg)
            self.sram_to_accelerator_memory(sram_address=sram_d, accelerator_dram_address=output_dram_addr, element_size=N)
            self.overwrite_instruction_with_general_register(tmp_reg)
        else:
            self.sram_to_accelerator_memory(sram_address=sram_d, accelerator_dram_address=output_dram_addr, element_size=N)
        return 4 * N

    def decoder_attention_core(self, head_dim: int, seq_len: int, Q_DRAM_ADDR: int, K_DRAM_ADDR: int, V_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, SCRATCH_DRAM_ADDR: int, IDENTITY_DRAM_ADDR: int =None, BIAS_DRAM_ADDR: int = None,
                            debug_mode: bool = False, SM_OUTPUT_DRAM_ADDR: int = None, q_rows: int = 1) -> None:
        """Decoder-side attention kernel for a single KV head (GQA).

        q_rows: number of Q rows to compute against this shared K/V in a
            single call. In a GQA layer with group_size query heads sharing
            one KV head, pass q_rows=group_size to process all group_size
            query heads in one call — K, V, V^T are loaded from DRAM once
            and reused across all q_rows. Q_DRAM_ADDR must point to q_rows
            contiguous head_dim-sized vectors, and OUTPUT_DRAM_ADDR must
            have room for q_rows contiguous head_dim-sized outputs.
            Default (q_rows=1) preserves the original per-head behaviour.
        """

        bytes_per_element = 2
        bias_enable = True if BIAS_DRAM_ADDR is not None else False

        if debug_mode: # DEBUG only, needs to be allocated in DRAM
            assert SM_OUTPUT_DRAM_ADDR is not None, "SM_OUTPUT_DRAM_ADDR is not set for debug mode"

        # SCRATCH_DRAM_ADDR is used for V^T
        SCRATCH_DRAM_PARTIAL_SM = SCRATCH_DRAM_ADDR + head_dim * seq_len * bytes_per_element # used for partial softmax output

        # ----------------------------------------------------------------------------------------------------------------
        # I @ V^T: (head_dim, head_dim) @ (seq_len, head_dim)^T -> (head_dim, seq_len)
        # Convention: first matrix I is (M, K), second V^T is (K, N), output  (M, N)
        M = head_dim   # identity length (rows of I)
        K = head_dim  # identity dimension (inner product dim)
        N = seq_len   # V length (columns of V^T)

        identity_tensor = torch.eye(head_dim, dtype=torch.bfloat16)

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
        assert M_chunk >= 1 and M_chunk <= M, f"M_chunk={M_chunk} must be greater than 0 and less than M={M}"

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

                    ones_idx = identity_tensor[output_row+i, :].reshape(-1, UE_VECTOR_SIZE).sum(axis=1).argmax(axis=0)
                    vector_idx = identity_tensor[output_row+i, :].reshape(-1, UE_VECTOR_SIZE)[ones_idx, :].argmax(axis=0)

                    self.start_queue_for_bf16_matvec_operation(max_clear_en=0,
                                                            fmax_context_addr=0,
                                                            vector_sram_start_addr=0x00000 + vector_idx * UE_VECTOR_SIZE * bytes_per_element,
                                                            matrix_sram_start_addr=uram_b_start_addr + ones_idx * UE_VECTOR_SIZE * bytes_per_element,
                                                            output_sram_wb_addr=output_sram_wb_addr + out_sram_offset,
                                                            K=UE_VECTOR_SIZE,
                                                            N=n_take,
                                                            stride_z=m_take)

                start_dram_address_of_partial_matrix = SCRATCH_DRAM_ADDR + i * N * bytes_per_element + j * bytes_per_element # the space needed is head_dim x seq_len

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

        # ----------------------------------------------------------------------------------------------------------------
        # Q @ K^T: (q_rows, head_dim) @ (head_dim, seq_len) -> (q_rows, seq_len)
        # Convention: first matrix Q is (M, K), second K^T is (K, N), output scores (M, N)
        # With q_rows > 1 (GQA head fusion), a single load of each K chunk
        # is reused across all q_rows Q rows — amortizes the K DMA.
        M = q_rows    # query length (rows of Q)
        K = head_dim  # head dimension (inner product dim)
        N = seq_len   # key length (columns of K^T)
        # Calculate N_chunk
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
        assert M_chunk >= 1 and M_chunk <= M, f"M_chunk={M_chunk} must be greater than 0 and less than M={M}"

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

            assert output_sram_wb_addr < 0x80000, f"output_sram_wb_addr={output_sram_wb_addr} is greater than 0x80000, which is the size of URAM_B"

            clear_en = 1
            for j, n_take in self.chunk_ranges(N, N_chunk):
                self.accelerator_memory_to_sram(accelerator_dram_address=K_DRAM_ADDR + j * K * bytes_per_element,
                                            sram_address=uram_b_start_addr,
                                            element_size=n_take * K)

                if bias_enable:
                    self.accelerator_memory_to_bias_sram(accelerator_dram_address=BIAS_DRAM_ADDR + j * bytes_per_element,
                                                       element_size=n_take)

                assert m_take * K + n_take * m_take<= URAM_FULL_ELEMENTS

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

                # TODO: if FMAX_CONTEXT_SIZE x seq_len can fit in URAM_A, then we can avoid copying to DRAM, create a special case for this
                start_dram_address_of_partial_matrix = SCRATCH_DRAM_PARTIAL_SM + j * bytes_per_element # the space needed is FMAX_CONTEXT_SIZE x seq_len

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
                        assert False, f"v_tr_row_chunk_size={v_tr_row_chunk_size} is too large to fit in URAM_NEAR_FULL_ELEMENTS={URAM_NEAR_FULL_ELEMENTS}"

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

    # ================================================================
    #  Vision encoder on FPGA
    # ================================================================

    # Vision encoder constants
    VIS_H = 768          # hidden size
    VIS_HEADS = 12       # number of attention heads
    VIS_HEAD_DIM = 64    # head_dim = 768 / 12
    VIS_MLP = 3072       # intermediate_size
    VIS_LAYERS = 16      # num_hidden_layers
    VIS_ROPE_DIM = 32    # half of head_dim for 2D RoPE (64 / 2)

    def vision_weight_init(self, hf_model) -> None:
        """Upload vision encoder weights to FPGA DRAM.

        Q/K/V/O and MLP gate/up/down are quantized to FP4 E2M1 (q4_64) with
        BF16 per-block scales (block_size=64 along K). Norms and the patch
        embedder stay BF16. Each quantized projection stores a dict
        {'data': addr, 'scale': addr} in the per-layer address table so the
        compile functions can pass SCALE_DRAM_ADDR alongside B_DRAM_ADDR.
        """
        vt = hf_model.model.vision_tower
        bpe = self.bytes_per_element
        H = self.VIS_H
        MLP = self.VIS_MLP
        HD = self.VIS_HEAD_DIM

        print("\n[Vision] Uploading vision encoder weights to DRAM (FP4 q4_64) ...")

        def _upload_q4_64(w_bf16: torch.Tensor) -> dict:
            """Quantize [N, K] bf16 weight to FP4 E2M1 (q4_64), DMA data + scale,
            return {'data': addr, 'scale': addr, 'shape': (N, K)}."""
            assert w_bf16.dim() == 2, f"expected 2D weight, got {w_bf16.shape}"
            N, K = w_bf16.shape
            assert K % 64 == 0, f"K={K} not divisible by 64"
            data_bytes, scale_bytes = _quantize_bf16_to_fp4_packed(w_bf16.contiguous(), block_size=64)
            scale_addr = self.allocate_tensor_dram(len(scale_bytes))
            self.dma_write(DMA_DEVICE_H2C, scale_addr, scale_bytes, len(scale_bytes))
            data_addr = self.allocate_tensor_dram(len(data_bytes))
            self.dma_write(DMA_DEVICE_H2C, data_addr, data_bytes, len(data_bytes))
            return {"data": data_addr, "scale": scale_addr, "shape": (N, K)}

        # Per-layer weights: pack all layers contiguously
        layer_weight_addrs = []
        for li in range(self.VIS_LAYERS):
            L = vt.encoder.layers[li]
            addrs = {}
            # Norms (BF16, small)
            for norm_name in ["input_layernorm", "post_attention_layernorm",
                              "pre_feedforward_layernorm", "post_feedforward_layernorm"]:
                w = getattr(L, norm_name).weight.detach().cpu().to(torch.bfloat16)
                addr = self.allocate_tensor_dram(w.numel() * bpe)
                self.dma_to_accelerator_memory(addr, w)
                addrs[norm_name] = addr

            # Q/K/V/O projections (FP4 q4_64)
            for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                w = getattr(L.self_attn, proj_name).linear.weight.detach().cpu().to(torch.bfloat16).contiguous()
                addrs[proj_name] = _upload_q4_64(w)

            # Q/K norms (BF16)
            for qk_norm in ["q_norm", "k_norm"]:
                w = getattr(L.self_attn, qk_norm).weight.detach().cpu().to(torch.bfloat16)
                addr = self.allocate_tensor_dram(w.numel() * bpe)
                self.dma_to_accelerator_memory(addr, w)
                addrs[qk_norm] = addr

            # MLP gate/up/down (FP4 q4_64)
            for mlp_name in ["gate_proj", "up_proj", "down_proj"]:
                w = getattr(L.mlp, mlp_name).linear.weight.detach().cpu().to(torch.bfloat16).contiguous()
                addrs[mlp_name] = _upload_q4_64(w)

            layer_weight_addrs.append(addrs)

        self._vis_weight_addrs = layer_weight_addrs

        # Patch embedder: input_proj Linear(768, 768) is also FP4 q4_64.
        # K=768, block_size=64 → 12 blocks per row, same as layer projections.
        pe = vt.patch_embedder
        w_proj = pe.input_proj.weight.detach().cpu().to(torch.bfloat16).contiguous()
        self.VIS_PATCH_PROJ_INFO = _upload_q4_64(w_proj)

        # Position embedding table [2, 10240, 768] kept on host — the lookup
        # is a gather and only runs once per image, not a hot path.
        self._vis_pos_embed_table = pe.position_embedding_table.detach().cpu().to(torch.bfloat16)

        # Embed vision (projector) weights:
        # - embedding_pre_projection_norm is a Gemma4RMSNorm(with_scale=False),
        #   i.e. pure x * rsqrt(mean(x^2)+eps). We upload an all-ones gamma so
        #   the standard rms_norm_core_dram (which multiplies by gamma) is
        #   equivalent to "no scale".
        # - embedding_projection is Linear(768, 1536), quantized to FP4 q4_64
        #   like every other vision matmul.
        ev = hf_model.model.embed_vision
        self.VIS_EMBED_NORM_HAS_SCALE = False
        ones_gamma = torch.ones(H, dtype=torch.bfloat16)
        self.VIS_EMBED_NORM_GAMMA = self.allocate_tensor_dram(ones_gamma.numel() * bpe)
        self.dma_to_accelerator_memory(self.VIS_EMBED_NORM_GAMMA, ones_gamma)

        w_eproj = ev.embedding_projection.weight.detach().cpu().to(torch.bfloat16).contiguous()
        # [out=text_hidden_size=1536, in=vision_hidden_size=768]
        self.VIS_EMBED_PROJ_INFO = _upload_q4_64(w_eproj)
        self.VIS_TEXT_H = w_eproj.shape[0]  # 1536 for Gemma4 E2B

        # Config values needed at inference time (stored to avoid passing
        # hf_model through every vision call).
        self.VIS_POOL_K = vt.config.pooling_kernel_size  # typically 3

        # Extract clip ranges from Gemma4ClippableLinear wrappers
        clip_ranges = []
        for li in range(self.VIS_LAYERS):
            L = vt.encoder.layers[li]
            cr = {}
            for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                proj = getattr(L.self_attn, proj_name)
                if proj.use_clipped_linears:
                    cr[proj_name] = {
                        "input": (proj.input_min.item(), proj.input_max.item()),
                        "output": (proj.output_min.item(), proj.output_max.item()),
                    }
                else:
                    cr[proj_name] = {
                        "input": (float("-inf"), float("inf")),
                        "output": (float("-inf"), float("inf")),
                    }
            for mlp_name in ["gate_proj", "up_proj", "down_proj"]:
                proj = getattr(L.mlp, mlp_name)
                if proj.use_clipped_linears:
                    cr[mlp_name] = {
                        "input": (proj.input_min.item(), proj.input_max.item()),
                        "output": (proj.output_min.item(), proj.output_max.item()),
                    }
                else:
                    cr[mlp_name] = {
                        "input": (float("-inf"), float("inf")),
                        "output": (float("-inf"), float("inf")),
                    }
            clip_ranges.append(cr)
        self._vis_clip_ranges = clip_ranges
        print(f"  Clip ranges loaded for {len(clip_ranges)} layers "
              f"(q_proj input: [{clip_ranges[0]['q_proj']['input'][0]:.2f}, "
              f"{clip_ranges[0]['q_proj']['input'][1]:.2f}])")

        # Save vision weight end address — program DRAM must start AFTER this
        self._vis_weight_end = self.get_tensor_dram_addr()
        total = self.get_tensor_dram_usage()
        print(f"  Vision weights uploaded. Tensor DRAM usage: {total} bytes")
        print(f"  Vision weight region: 0x{layer_weight_addrs[0]['input_layernorm']:X} .. 0x{self._vis_weight_end:X}")

    def vision_tensor_init(self, num_patches: int, *, program_base: int | None = None) -> None:
        """Allocate DRAM for vision encoder intermediate tensors.

        Reuses the LM tensor DRAM region since vision and LM don't run simultaneously.

        program_base: if None (compare-script default), vision programs go to
            DRAM_INSTRUCTION_ADDR (0xD0000000), which is inside the LM weights
            region. That's safe when no LM code runs after vision. The test
            script passes an explicit safe address (above LM tensors and clear
            of LM weights) so LM decode can run correctly post-vision.
        """
        bpe = self.bytes_per_element
        H = self.VIS_H
        HD = self.VIS_HEAD_DIM
        NH = self.VIS_HEADS
        MLP = self.VIS_MLP
        S = num_patches  # sequence length for vision

        # Reset tensor allocator — vision tensors reuse LM tensor DRAM region
        self.reset_tensor_dram_addr()
        if program_base is None:
            self._next_program_dram_addr = user_dma_core.DRAM_INSTRUCTION_ADDR
            self._program_dram_base = user_dma_core.DRAM_INSTRUCTION_ADDR
        else:
            self._next_program_dram_addr = program_base
            self._program_dram_base = program_base
        print(f"\n[Vision] Allocating vision tensor DRAM for {S} patches ...")
        print(f"  Program DRAM at 0x{self._program_dram_base:X}")

        # Layer I/O (double-buffered)
        self.VIS_IO_A = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_IO_B = self.allocate_tensor_dram(S * H * bpe)

        # Intermediates
        self.VIS_NORM_OUT = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_Q_DRAM = self.allocate_tensor_dram(S * NH * HD * bpe)
        self.VIS_K_DRAM = self.allocate_tensor_dram(S * NH * HD * bpe)
        self.VIS_V_DRAM = self.allocate_tensor_dram(S * NH * HD * bpe)
        self.VIS_Q_NORM = self.allocate_tensor_dram(S * NH * HD * bpe)
        self.VIS_K_NORM = self.allocate_tensor_dram(S * NH * HD * bpe)
        self.VIS_ATTN_OUT = self.allocate_tensor_dram(S * H * bpe)  # after o_proj
        self.VIS_POST_ATTN_NORM = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_POST_ATTN_RES = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_PRE_FFN_NORM = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_MLP_GATE = self.allocate_tensor_dram(S * MLP * bpe)
        self.VIS_MLP_UP = self.allocate_tensor_dram(S * MLP * bpe)
        self.VIS_MLP_MULT = self.allocate_tensor_dram(S * MLP * bpe)
        self.VIS_MLP_DOWN = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_POST_FFN_NORM = self.allocate_tensor_dram(S * H * bpe)

        # Flash attention buffers (per-head, process heads sequentially)
        aligned_S = ((S + 63) // 64) * 64
        self.VIS_FLASH_Q = self.allocate_tensor_dram(aligned_S * HD * bpe)
        self.VIS_FLASH_K = self.allocate_tensor_dram(aligned_S * HD * bpe)
        self.VIS_FLASH_V = self.allocate_tensor_dram(aligned_S * HD * bpe)
        self.VIS_FLASH_OUT = self.allocate_tensor_dram(aligned_S * HD * bpe)
        self.VIS_FLASH_SCRATCH = self.allocate_tensor_dram(
            max(HD, 512) * aligned_S * 2 + HD * aligned_S * 2)
        self.VIS_FLASH_BIAS = self.allocate_tensor_dram(aligned_S * aligned_S * bpe)

        # Bidirectional bias: all zeros, alignment padding columns masked.
        # NOTE: real image padding patches (position_ids == -1) still need to
        # be masked — call set_vision_attention_bias(padding_positions) before
        # running attention.
        bias = torch.zeros(aligned_S, aligned_S, dtype=torch.bfloat16)
        bias[:, S:] = float("-inf")  # alignment padding columns
        self.dma_to_accelerator_memory(self.VIS_FLASH_BIAS, bias)

        # RoPE tables
        self.VIS_ROPE_COS = self.allocate_tensor_dram(S * HD * bpe)  # [S, 64]
        self.VIS_ROPE_SIN = self.allocate_tensor_dram(S * HD * bpe)

        # Identity matrix for flash attention
        self.VIS_IDENTITY = self.allocate_tensor_dram(HD * HD * bpe)
        self.dma_to_accelerator_memory(self.VIS_IDENTITY,
                                        torch.eye(HD, dtype=torch.bfloat16))

        # Embed-vision (pooler tail) scratch. N_soft (post-pooler, post-mask)
        # is image-dependent but bounded above by S / pooling_kernel_size^2; we
        # size at S to be safe and allocate in BF16.
        text_h = getattr(self, "VIS_TEXT_H", 1536)
        self.VIS_EMBED_POOL = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_EMBED_NORMED = self.allocate_tensor_dram(S * H * bpe)
        self.VIS_EMBED_OUT = self.allocate_tensor_dram(S * text_h * bpe)

        self._vis_num_patches = S
        self._vis_aligned_S = aligned_S
        self._vis_padding_mask = None  # [S] bool, set by set_vision_attention_bias
        total = self.get_tensor_dram_usage()
        print(f"  Vision tensor DRAM allocated. Total usage: {total} bytes")

    def set_vision_attention_bias(self, padding_positions: torch.Tensor) -> None:
        """Rebuild VIS_FLASH_BIAS so attention masks BOTH real padding patches
        (position_ids == -1) and the alignment padding at the end.

        padding_positions: [B, S] or [S] bool — True where patch is padding.
        Must be called after vision_tensor_init and before running attention.
        """
        S = self._vis_num_patches
        aligned_S = self._vis_aligned_S
        if padding_positions.dim() == 2:
            pad = padding_positions[0].cpu().bool()
        else:
            pad = padding_positions.cpu().bool()
        assert pad.shape[0] == S, f"padding mask has {pad.shape[0]} entries, expected {S}"
        self._vis_padding_mask = pad
        n_pad = int(pad.sum().item())
        n_valid = S - n_pad

        # Mask every padding column: real image padding + alignment padding.
        col_mask = torch.zeros(aligned_S, dtype=torch.bool)
        col_mask[:S] = pad
        col_mask[S:] = True
        bias = torch.zeros(aligned_S, aligned_S, dtype=torch.bfloat16)
        bias[:, col_mask] = float("-inf")
        self.dma_to_accelerator_memory(self.VIS_FLASH_BIAS, bias)
        print(f"  Vision attention bias: {n_valid} valid / {n_pad} real padding / "
              f"{aligned_S - S} alignment padding")

    def _run_eltwise_add_chunked(self, a_addr: int, b_addr: int, out_addr: int, num_elements: int) -> None:
        """Element-wise add two DRAM tensors, one SRAM-sized chunk per FPGA program."""
        CHUNK = 65536  # 128KB per buffer, safe for SRAM gap 0x10000-0x90000
        bpe = self.bytes_per_element
        for off in range(0, num_elements, CHUNK):
            n = min(CHUNK, num_elements - off)
            def _fn(a=a_addr + off * bpe, b=b_addr + off * bpe, o=out_addr + off * bpe, sz=n):
                self.accelerator_memory_to_sram(a, 0x10000, sz)
                self.accelerator_memory_to_sram(b, 0x90000, sz)
                self.eltwise_add_core(0x10000, 0x90000, 0x10000, sz)
                self.sram_to_accelerator_memory(0x10000, o, sz)
            self._compile_and_run_single("eltwise_add_chunk", _fn)

    def _run_eltwise_mul_chunked(self, a_addr: int, b_addr: int, out_addr: int, num_elements: int) -> None:
        """Element-wise multiply two DRAM tensors, one SRAM-sized chunk per FPGA program."""
        CHUNK = 65536  # 128KB per buffer, safe for SRAM
        bpe = self.bytes_per_element
        for off in range(0, num_elements, CHUNK):
            n = min(CHUNK, num_elements - off)
            def _fn(a=a_addr + off * bpe, b=b_addr + off * bpe, o=out_addr + off * bpe, sz=n):
                self.accelerator_memory_to_sram(a, 0x10000, sz)
                self.accelerator_memory_to_sram(b, 0x90000, sz)
                self.eltwise_mul_core(0x10000, 0x90000, 0x10000, sz)
                self.sram_to_accelerator_memory(0x10000, o, sz)
            self._compile_and_run_single("eltwise_mul_chunk", _fn)

    def _compile_and_run_single(self, label: str, compile_fn) -> None:
        """Compile a single operation, run it. Reuses same program DRAM slot."""
        import builtins
        _orig_print = builtins.print
        builtins.print = lambda *a, **kw: None

        # Reset program DRAM to reuse space (each op is independent)
        self.reset_program_dram_addr()

        self.start_capture()
        compile_fn()
        self.stop_capture()
        self.generate_instruction_halt()
        prog_addr = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prog_addr)
        sz = self.get_capture_instruction_size_bytes()
        self.allocate_program_dram(sz)
        self.clear_capture_buffer()
        builtins.print = _orig_print

        self.start_execute_from_dram(prog_addr)
        self.wait_queue(120.0)

    def _host_clip_dram(self, addr: int, shape: tuple, clip_min: float, clip_max: float) -> None:
        """Read tensor from FPGA DRAM, clip on host, write back."""
        t = self.dma_from_accelerator_memory(addr, shape).cpu()
        t = t.clamp(min=clip_min, max=clip_max)
        self.dma_to_accelerator_memory(addr, t)

    def compile_vision_layer(self, layer_idx: int) -> int:
        """Run vision layer part A: pre_norm + Q/K/V projections + Q/K norms.

        Each operation compiled and run separately to avoid FPGA instruction limits.
        Gemma4 vision uses Gemma4ClippableLinear which clips input/output — we apply
        clipping on host between FPGA matmul calls.
        """
        S = self._vis_num_patches
        H = self.VIS_H
        HD = self.VIS_HEAD_DIM
        NH = self.VIS_HEADS
        w = self._vis_weight_addrs[layer_idx]
        clips = self._vis_clip_ranges[layer_idx]

        INPUT_DRAM = self.VIS_IO_A if layer_idx % 2 == 0 else self.VIS_IO_B

        # 1. Input RMSNorm
        self._compile_and_run_single("pre_norm", lambda: self.rms_norm_core_dram(
            M=S, N=H, A_DRAM_ADDR=INPUT_DRAM,
            OUTPUT_DRAM_ADDR=self.VIS_NORM_OUT,
            GAMMA_DRAM_ADDR=w["input_layernorm"]))

        # Read norm_out once on host, then upload clipped versions per projection
        norm_out = self.dma_from_accelerator_memory(self.VIS_NORM_OUT, (S, H)).cpu()

        # 2. Q projection (FP4 q4_64): clip input on host, upload, run matmul, clip output
        self.dma_to_accelerator_memory(self.VIS_NORM_OUT,
            norm_out.clamp(min=clips["q_proj"]["input"][0], max=clips["q_proj"]["input"][1]))
        self._compile_and_run_single("q_proj", lambda: self.matmat_mul_core(
            M=S, K=H, N=NH * HD,
            A_DRAM_ADDR=self.VIS_NORM_OUT,
            B_DRAM_ADDR=w["q_proj"]["data"],
            OUTPUT_DRAM_ADDR=self.VIS_Q_DRAM,
            is_B_quantized=True,
            data_type=TYPE.FP4,
            SCALE_DRAM_ADDR=w["q_proj"]["scale"]))
        self._host_clip_dram(self.VIS_Q_DRAM, (S, NH * HD), *clips["q_proj"]["output"])

        # 3. K projection (FP4 q4_64)
        self.dma_to_accelerator_memory(self.VIS_NORM_OUT,
            norm_out.clamp(min=clips["k_proj"]["input"][0], max=clips["k_proj"]["input"][1]))
        self._compile_and_run_single("k_proj", lambda: self.matmat_mul_core(
            M=S, K=H, N=NH * HD,
            A_DRAM_ADDR=self.VIS_NORM_OUT,
            B_DRAM_ADDR=w["k_proj"]["data"],
            OUTPUT_DRAM_ADDR=self.VIS_K_DRAM,
            is_B_quantized=True,
            data_type=TYPE.FP4,
            SCALE_DRAM_ADDR=w["k_proj"]["scale"]))
        self._host_clip_dram(self.VIS_K_DRAM, (S, NH * HD), *clips["k_proj"]["output"])

        # 4. V projection (FP4 q4_64)
        self.dma_to_accelerator_memory(self.VIS_NORM_OUT,
            norm_out.clamp(min=clips["v_proj"]["input"][0], max=clips["v_proj"]["input"][1]))
        self._compile_and_run_single("v_proj", lambda: self.matmat_mul_core(
            M=S, K=H, N=NH * HD,
            A_DRAM_ADDR=self.VIS_NORM_OUT,
            B_DRAM_ADDR=w["v_proj"]["data"],
            OUTPUT_DRAM_ADDR=self.VIS_V_DRAM,
            is_B_quantized=True,
            data_type=TYPE.FP4,
            SCALE_DRAM_ADDR=w["v_proj"]["scale"]))
        self._host_clip_dram(self.VIS_V_DRAM, (S, NH * HD), *clips["v_proj"]["output"])

        # Restore unclipped norm_out for comparison readback
        self.dma_to_accelerator_memory(self.VIS_NORM_OUT, norm_out)

        # 5. Q norm
        self._compile_and_run_single("q_norm", lambda: self.rms_norm_core_dram(
            M=S * NH, N=HD,
            A_DRAM_ADDR=self.VIS_Q_DRAM,
            OUTPUT_DRAM_ADDR=self.VIS_Q_NORM,
            GAMMA_DRAM_ADDR=w["q_norm"]))

        # 6. K norm
        self._compile_and_run_single("k_norm", lambda: self.rms_norm_core_dram(
            M=S * NH, N=HD,
            A_DRAM_ADDR=self.VIS_K_DRAM,
            OUTPUT_DRAM_ADDR=self.VIS_K_NORM,
            GAMMA_DRAM_ADDR=w["k_norm"]))

        print(f"  [Vision L{layer_idx}] Part A: proj+norm done (with clipping)")
        return 0

    def host_vision_v_norm_rope_gather(self, layer_idx: int,
                                        cos_2d: torch.Tensor,
                                        sin_2d: torch.Tensor) -> None:
        """Host-side: V norm, 2D RoPE on Q/K, gather per-head for attention.

        Done on host because:
        - V norm without scale: rms_norm_core_dram requires gamma, and per-element
          SRAM loop generates too many instructions for large S.
        - 2D RoPE: rope_hf_core requires N>=64, but vision uses N=32 per dim.
        - Head gather: per-token strided copy generates too many instructions.

        Reads Q_NORM, K_NORM, V_DRAM from FPGA. Applies V norm + RoPE on host.
        Then for each head, uploads [S, HD] Q/K/V to flash buffers for attention.
        """
        bpe = self.bytes_per_element
        S = self._vis_num_patches
        HD = self.VIS_HEAD_DIM
        NH = self.VIS_HEADS
        aligned_S = self._vis_aligned_S

        # Read Q_NORM, K_NORM, V from FPGA
        q_norm = self.dma_from_accelerator_memory(
            self.VIS_Q_NORM, (S * NH, HD)).cpu().float()
        k_norm = self.dma_from_accelerator_memory(
            self.VIS_K_NORM, (S * NH, HD)).cpu().float()
        v_raw = self.dma_from_accelerator_memory(
            self.VIS_V_DRAM, (S * NH, HD)).cpu().float()

        # V norm (RMS normalize without scale)
        v_rms = (v_raw ** 2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        v_normed = (v_raw / v_rms).to(torch.bfloat16)
        # Write back to FPGA (for readback comparison)
        self.dma_to_accelerator_memory(self.VIS_V_DRAM, v_normed)

        # 2D RoPE: split head_dim=64 into 2x32, apply 1D RoPE to each half
        # cos_2d, sin_2d: [S, 64] = [S, 32_row, 32_col]
        q_4d = q_norm.reshape(S, NH, HD)  # [S, NH, 64]
        k_4d = k_norm.reshape(S, NH, HD)  # [S, NH, 64]
        cos = cos_2d.float()  # [S, 64]
        sin = sin_2d.float()  # [S, 64]

        def rotate_half(x):
            x1 = x[..., :x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        # Apply 2D RoPE: split into two halves of 32 dims each
        for qk, name in [(q_4d, "Q"), (k_4d, "K")]:
            for dim_idx in range(2):
                d_start = dim_idx * 32
                d_end = d_start + 32
                x_part = qk[:, :, d_start:d_end]  # [S, NH, 32]
                cos_part = cos[:, d_start:d_end].unsqueeze(1)  # [S, 1, 32]
                sin_part = sin[:, d_start:d_end].unsqueeze(1)  # [S, 1, 32]
                rotated = (x_part * cos_part) + (rotate_half(x_part) * sin_part)
                qk[:, :, d_start:d_end] = rotated

        # Write Q/K after RoPE back to FPGA (for readback comparison)
        q_roped = q_4d.reshape(S * NH, HD).to(torch.bfloat16)
        k_roped = k_4d.reshape(S * NH, HD).to(torch.bfloat16)
        self.dma_to_accelerator_memory(self.VIS_Q_NORM, q_roped)
        self.dma_to_accelerator_memory(self.VIS_K_NORM, k_roped)

        self._vis_q_roped = q_4d  # [S, NH, HD] float for attention
        self._vis_k_roped = k_4d
        self._vis_v_normed = v_normed.reshape(S, NH, HD)

    def run_vision_attention_all_heads(self, layer_idx: int) -> None:
        """Run attention for all heads: host gather -> FPGA attn -> host scatter.

        Q pre-scale is done on host to avoid SRAM overflow (aligned_S * HD can
        exceed available SRAM for large patch counts).
        """
        bpe = self.bytes_per_element
        S = self._vis_num_patches
        HD = self.VIS_HEAD_DIM
        NH = self.VIS_HEADS
        aligned_S = self._vis_aligned_S

        q = self._vis_q_roped  # [S, NH, HD] float
        k = self._vis_k_roped
        v = self._vis_v_normed.float()

        attn_out_all = torch.zeros(S, NH, HD, dtype=torch.bfloat16)

        for h in range(NH):
            # Gather head h: [S, HD]
            # Q pre-scale on host: cancel flash_attention_core's internal 1/sqrt(d)
            q_h = (q[:, h, :] * math.sqrt(HD)).to(torch.bfloat16)
            k_h = k[:, h, :].to(torch.bfloat16)
            v_h = v[:, h, :].to(torch.bfloat16)

            # Pad to aligned_S
            q_pad = torch.zeros(aligned_S, HD, dtype=torch.bfloat16)
            k_pad = torch.zeros(aligned_S, HD, dtype=torch.bfloat16)
            v_pad = torch.zeros(aligned_S, HD, dtype=torch.bfloat16)
            q_pad[:S] = q_h
            k_pad[:S] = k_h
            v_pad[:S] = v_h

            # Upload to FPGA flash buffers
            self.dma_to_accelerator_memory(self.VIS_FLASH_Q, q_pad)
            self.dma_to_accelerator_memory(self.VIS_FLASH_K, k_pad)
            self.dma_to_accelerator_memory(self.VIS_FLASH_V, v_pad)
            self.dma_to_accelerator_memory(self.VIS_FLASH_OUT,
                                            torch.zeros(aligned_S, HD, dtype=torch.bfloat16))

            # Compile flash attention (reset program DRAM each head to prevent overflow)
            self._compile_and_run_single(f"attn_h{h}", lambda:
                self._flash_attention_core_cached(
                    head_dim=HD,
                    seq_len=aligned_S,
                    Q_DRAM_ADDR=self.VIS_FLASH_Q,
                    K_DRAM_ADDR=self.VIS_FLASH_K,
                    V_DRAM_ADDR=self.VIS_FLASH_V,
                    OUTPUT_DRAM_ADDR=self.VIS_FLASH_OUT,
                    SCRATCH_DRAM_ADDR=self.VIS_FLASH_SCRATCH,
                    BIAS_DRAM_ADDR=self.VIS_FLASH_BIAS))

            # Read back and scatter
            out_h = self.dma_from_accelerator_memory(
                self.VIS_FLASH_OUT, (aligned_S, HD)).cpu()[:S]
            attn_out_all[:, h, :] = out_h

        # Write full attention output [S, NH*HD] to VIS_Q_DRAM (reuse)
        self.dma_to_accelerator_memory(self.VIS_Q_DRAM,
                                        attn_out_all.reshape(S, NH * HD))

    def compile_vision_layer_post_attn(self, layer_idx: int) -> int:
        """Run post-attention: O proj + post_attn_norm + residual + MLP + output.

        Each operation compiled and run separately.
        """
        S = self._vis_num_patches
        H = self.VIS_H
        HD = self.VIS_HEAD_DIM
        NH = self.VIS_HEADS
        MLP = self.VIS_MLP
        w = self._vis_weight_addrs[layer_idx]

        INPUT_DRAM = self.VIS_IO_A if layer_idx % 2 == 0 else self.VIS_IO_B
        OUTPUT_DRAM = self.VIS_IO_B if layer_idx % 2 == 0 else self.VIS_IO_A

        clips = self._vis_clip_ranges[layer_idx]

        # O projection (FP4 q4_64, with clipping)
        self._host_clip_dram(self.VIS_Q_DRAM, (S, NH * HD), *clips["o_proj"]["input"])
        self._compile_and_run_single("o_proj", lambda: self.matmat_mul_core(
            M=S, K=NH * HD, N=H,
            A_DRAM_ADDR=self.VIS_Q_DRAM,
            B_DRAM_ADDR=w["o_proj"]["data"],
            OUTPUT_DRAM_ADDR=self.VIS_ATTN_OUT,
            is_B_quantized=True,
            data_type=TYPE.FP4,
            SCALE_DRAM_ADDR=w["o_proj"]["scale"]))
        self._host_clip_dram(self.VIS_ATTN_OUT, (S, H), *clips["o_proj"]["output"])

        # Post-attention norm
        self._compile_and_run_single("post_attn_norm", lambda: self.rms_norm_core_dram(
            M=S, N=H,
            A_DRAM_ADDR=self.VIS_ATTN_OUT,
            OUTPUT_DRAM_ADDR=self.VIS_POST_ATTN_NORM,
            GAMMA_DRAM_ADDR=w["post_attention_layernorm"]))

        # Residual: input + post_attn_norm (chunked, each chunk a separate program)
        sz_h = S * H
        self._run_eltwise_add_chunked(INPUT_DRAM, self.VIS_POST_ATTN_NORM,
                                       self.VIS_POST_ATTN_RES, sz_h)

        # Pre-FFN norm
        self._compile_and_run_single("pre_ffn_norm", lambda: self.rms_norm_core_dram(
            M=S, N=H,
            A_DRAM_ADDR=self.VIS_POST_ATTN_RES,
            OUTPUT_DRAM_ADDR=self.VIS_PRE_FFN_NORM,
            GAMMA_DRAM_ADDR=w["pre_feedforward_layernorm"]))

        # Read pre_ffn_norm once, clip copies for gate/up
        pre_ffn = self.dma_from_accelerator_memory(self.VIS_PRE_FFN_NORM, (S, H)).cpu()

        # MLP gate (FP4 q4_64, GELU) — with clipping
        self.dma_to_accelerator_memory(self.VIS_PRE_FFN_NORM,
            pre_ffn.clamp(min=clips["gate_proj"]["input"][0], max=clips["gate_proj"]["input"][1]))
        self._compile_and_run_single("gate_proj", lambda: self.matmat_mul_core(
            M=S, K=H, N=MLP,
            A_DRAM_ADDR=self.VIS_PRE_FFN_NORM,
            B_DRAM_ADDR=w["gate_proj"]["data"],
            OUTPUT_DRAM_ADDR=self.VIS_MLP_GATE,
            is_B_quantized=True,
            data_type=TYPE.FP4,
            SCALE_DRAM_ADDR=w["gate_proj"]["scale"],
            gelu_enable=True))
        self._host_clip_dram(self.VIS_MLP_GATE, (S, MLP), *clips["gate_proj"]["output"])

        # MLP up (FP4 q4_64) — with clipping
        self.dma_to_accelerator_memory(self.VIS_PRE_FFN_NORM,
            pre_ffn.clamp(min=clips["up_proj"]["input"][0], max=clips["up_proj"]["input"][1]))
        self._compile_and_run_single("up_proj", lambda: self.matmat_mul_core(
            M=S, K=H, N=MLP,
            A_DRAM_ADDR=self.VIS_PRE_FFN_NORM,
            B_DRAM_ADDR=w["up_proj"]["data"],
            OUTPUT_DRAM_ADDR=self.VIS_MLP_UP,
            is_B_quantized=True,
            data_type=TYPE.FP4,
            SCALE_DRAM_ADDR=w["up_proj"]["scale"]))
        self._host_clip_dram(self.VIS_MLP_UP, (S, MLP), *clips["up_proj"]["output"])

        # Restore unclipped pre_ffn_norm for comparison readback
        self.dma_to_accelerator_memory(self.VIS_PRE_FFN_NORM, pre_ffn)

        # gate * up (chunked, each chunk a separate program)
        self._run_eltwise_mul_chunked(self.VIS_MLP_GATE, self.VIS_MLP_UP,
                                       self.VIS_MLP_MULT, S * MLP)

        # MLP down (FP4 q4_64) — with clipping
        self._host_clip_dram(self.VIS_MLP_MULT, (S, MLP), *clips["down_proj"]["input"])
        self._compile_and_run_single("down_proj", lambda: self.matmat_mul_core(
            M=S, K=MLP, N=H,
            A_DRAM_ADDR=self.VIS_MLP_MULT,
            B_DRAM_ADDR=w["down_proj"]["data"],
            OUTPUT_DRAM_ADDR=self.VIS_MLP_DOWN,
            is_B_quantized=True,
            data_type=TYPE.FP4,
            SCALE_DRAM_ADDR=w["down_proj"]["scale"]))
        self._host_clip_dram(self.VIS_MLP_DOWN, (S, H), *clips["down_proj"]["output"])

        # Post-FFN norm
        self._compile_and_run_single("post_ffn_norm", lambda: self.rms_norm_core_dram(
            M=S, N=H,
            A_DRAM_ADDR=self.VIS_MLP_DOWN,
            OUTPUT_DRAM_ADDR=self.VIS_POST_FFN_NORM,
            GAMMA_DRAM_ADDR=w["post_feedforward_layernorm"]))

        # Post-FFN residual -> OUTPUT (chunked, each chunk a separate program)
        self._run_eltwise_add_chunked(self.VIS_POST_ATTN_RES, self.VIS_POST_FFN_NORM,
                                       OUTPUT_DRAM, sz_h)

        print(f"  [Vision L{layer_idx}] Part D: 10 ops (O proj, norms, MLP, residuals) done")
        return 0

    def run_vision_layer(self, layer_idx: int,
                          cos_2d: torch.Tensor,
                          sin_2d: torch.Tensor) -> int:
        """Run one full vision encoder layer on HW.

        Orchestrates: pre_norm + Q/K/V/O projections + Q/K norms (FPGA) →
        V norm + 2D RoPE + per-head gather (host) → flash attention per head
        (FPGA) → O proj + post_attn_norm + residual + MLP + post_ffn_norm +
        residual (FPGA).

        Input must already be in VIS_IO_A (if layer_idx even) or VIS_IO_B
        (if layer_idx odd). Output lands in the other of the two buffers.
        Returns the output DRAM address so callers can read back.
        """
        self.compile_vision_layer(layer_idx)
        self.host_vision_v_norm_rope_gather(layer_idx, cos_2d, sin_2d)
        self.run_vision_attention_all_heads(layer_idx)
        self.compile_vision_layer_post_attn(layer_idx)
        return self.VIS_IO_B if layer_idx % 2 == 0 else self.VIS_IO_A

    def vision_patch_embed(self,
                            pixel_values: torch.Tensor,
                            pixel_position_ids: torch.Tensor,
                            padding_positions: torch.Tensor) -> torch.Tensor:
        """Run the Gemma4 vision patch embedder on FPGA.

        Mirrors Gemma4VisionPatchEmbedder.forward:
          scaled = 2 * (pixel_values - 0.5)
          hidden = input_proj(scaled)               # FPGA, FP4 q4_64
          pos    = pos_table[0, x] + pos_table[1, y]  # host gather
          pos[padding] = 0
          return hidden + pos                        # FPGA, chunked eltwise add

        Uses VIS_IO_B as a pixel scratchpad and VIS_NORM_OUT as the position-
        embedding staging buffer (both are otherwise unused before layer 0
        runs). Final patch embeddings land in VIS_IO_A.

        Returns the patch embeddings [S, H] bf16 (read back for comparison).
        """
        S = self._vis_num_patches
        H = self.VIS_H

        pv = pixel_values
        if pv.dim() == 3 and pv.shape[0] == 1:
            pv = pv.squeeze(0)
        assert pv.shape == (S, H), f"pixels shape {pv.shape}, expected ({S}, {H})"
        pids = pixel_position_ids
        if pids.dim() == 3 and pids.shape[0] == 1:
            pids = pids.squeeze(0)
        pad = padding_positions
        if pad.dim() == 2 and pad.shape[0] == 1:
            pad = pad.squeeze(0)

        # Host: scale pixels, upload to VIS_IO_B as temp.
        scaled = (2.0 * (pv.float() - 0.5)).to(torch.bfloat16).contiguous()
        self.dma_to_accelerator_memory(self.VIS_IO_B, scaled)

        # FPGA: input_proj matmul (FP4 q4_64) → VIS_IO_A
        w = self.VIS_PATCH_PROJ_INFO
        self._compile_and_run_single("patch_input_proj", lambda: self.matmat_mul_core(
            M=S, K=H, N=H,
            A_DRAM_ADDR=self.VIS_IO_B,
            B_DRAM_ADDR=w["data"],
            OUTPUT_DRAM_ADDR=self.VIS_IO_A,
            is_B_quantized=True,
            data_type=TYPE.FP4,
            SCALE_DRAM_ADDR=w["scale"]))

        # Host: gather position embeddings [S, H], zero padding rows.
        table = self._vis_pos_embed_table.float()          # [2, P, H]
        clamped = pids.clamp(min=0).long().cpu()           # [S, 2]
        pe_sum = (table[0, clamped[:, 0]] + table[1, clamped[:, 1]])  # [S, H]
        pe_sum[pad.cpu()] = 0.0
        pe_bf16 = pe_sum.to(torch.bfloat16).contiguous()

        # Upload to VIS_NORM_OUT (scratch) and FPGA eltwise-add into VIS_IO_A
        self.dma_to_accelerator_memory(self.VIS_NORM_OUT, pe_bf16)
        self._run_eltwise_add_chunked(
            self.VIS_IO_A, self.VIS_NORM_OUT, self.VIS_IO_A, S * H)

        return self.dma_from_accelerator_memory(self.VIS_IO_A, (S, H)).cpu()

    def vision_embed_project(self,
                              hidden_states: torch.Tensor,
                              pixel_position_ids: torch.Tensor,
                              padding_positions: torch.Tensor) -> torch.Tensor:
        """Run the Gemma4 vision pooler + embed_vision tail.

        Mirrors Gemma4VisionModel.forward's tail:
          hidden = masked_fill(hidden, padding)
          hidden = _avg_pool_by_positions(hidden, ids, output_length)   # host
          hidden = hidden * sqrt(hidden_size)                            # host
          hidden = hidden[pooler_mask]                                   # host
          hidden = rms_norm(hidden, with_scale=False)                    # FPGA
          out    = embedding_projection @ hidden                          # FPGA (FP4)

        Pooling and gather/mask stay on host because they are indexed by
        position (hard to express with matmat cores) and only touch <~800
        rows. The per-element RMSNorm and the Linear(768, 1536) are on FPGA.

        hidden_states: [S, 768] or [1, S, 768] — output of the encoder.
        Returns image_features [N_final, 1536] bf16.
        """
        H = self.VIS_H
        text_h = self.VIS_TEXT_H
        pool_k = self.VIS_POOL_K

        # Bring everything to CPU — the HF model may be on GPU, but the
        # pooler math is cheap and the FPGA DMA path expects CPU tensors.
        hidden_states = hidden_states.detach().cpu()
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        S = hidden_states.shape[1]
        output_length = S // (pool_k * pool_k)

        pids = pixel_position_ids.detach().cpu()
        if pids.dim() == 2:
            pids = pids.unsqueeze(0)
        pad = padding_positions.detach().cpu()
        if pad.dim() == 1:
            pad = pad.unsqueeze(0)

        # Host: pooler.forward equivalent (zero padding, spatial avg pool,
        # scale by sqrt(hidden_size), strip masked rows).
        # This inlines Gemma4VisionPooler._avg_pool_by_positions so the method
        # has no dependency on the HF module tree at runtime.
        with torch.no_grad():
            h = hidden_states.float().clone()
            h.masked_fill_(pad.unsqueeze(-1), 0.0)
            input_seq_len = h.shape[1]
            k = int((input_seq_len // output_length) ** 0.5)
            k_squared = k * k
            if k_squared * output_length != input_seq_len:
                raise ValueError(
                    f"Cannot pool {h.shape} to {output_length}: k={k}^2 × length="
                    f"{output_length} must equal {input_seq_len}.")
            clamped = pids.clamp(min=0)                         # [1, S, 2]
            max_x = clamped[..., 0].max(dim=-1, keepdim=True)[0] + 1
            kernel_idxs = torch.div(clamped, k, rounding_mode="floor")
            kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
            weights = F.one_hot(kernel_idxs.long(), output_length).float() / k_squared  # [1, S, L]
            pooled = weights.transpose(1, 2) @ h                # [1, L, 768]
            pooler_mask = torch.logical_not((weights == 0).all(dim=1))  # [1, L]
            pooled = pooled * (H ** 0.5)
            pooled = pooled[pooler_mask]                        # [N_final, 768]
        N_final = int(pooled.shape[0])
        assert N_final <= S, f"pooler produced {N_final} rows, scratch sized for {S}"

        pooled_bf16 = pooled.to(torch.bfloat16).contiguous()
        self.dma_to_accelerator_memory(self.VIS_EMBED_POOL, pooled_bf16)

        # FPGA: RMSNorm (with_scale=False → gamma=ones)
        self._compile_and_run_single("embed_pre_norm", lambda: self.rms_norm_core_dram(
            M=N_final, N=H,
            A_DRAM_ADDR=self.VIS_EMBED_POOL,
            OUTPUT_DRAM_ADDR=self.VIS_EMBED_NORMED,
            GAMMA_DRAM_ADDR=self.VIS_EMBED_NORM_GAMMA))

        # FPGA: embedding_projection (FP4 q4_64)
        w = self.VIS_EMBED_PROJ_INFO
        self._compile_and_run_single("embed_projection", lambda: self.matmat_mul_core(
            M=N_final, K=H, N=text_h,
            A_DRAM_ADDR=self.VIS_EMBED_NORMED,
            B_DRAM_ADDR=w["data"],
            OUTPUT_DRAM_ADDR=self.VIS_EMBED_OUT,
            is_B_quantized=True,
            data_type=TYPE.FP4,
            SCALE_DRAM_ADDR=w["scale"]))

        return self.dma_from_accelerator_memory(self.VIS_EMBED_OUT, (N_final, text_h)).cpu()

    # ================================================================
    #  Audio encoder on FPGA  (Conformer, ported from Parakeet)
    # ================================================================
    #
    # Architecture (mirrors Gemma4AudioModel in HF):
    #   log-mel features → SubSampleConvProjection (host) → 12 × Gemma4AudioLayer
    #   → output_proj (1024 → 1536) → Gemma4MultimodalEmbedder (RMSNorm + Linear)
    #
    # Per layer (Gemma4AudioLayer.forward):
    #   FFN1 macaron half (RMSNorm in/out, Linear+SiLU+Linear, *0.5 residual)
    #   → norm_pre_attn → chunked self-attn (rel-pos, soft-cap) → norm_post_attn → +residual
    #   → lconv1d (RMSNorm, Linear→GLU, depthwise k=5, conv_norm, SiLU, Linear, +residual)
    #   → FFN2 macaron half
    #   → norm_out
    #
    # Phase 1 (current): host subsample + host rel-pos + FPGA FFN1 macaron half
    # Phase 2:           FPGA chunked self-attn (Q/K/V, rel-pos bias, soft-cap, mask, softmax)
    # Phase 3:           FPGA conv module (Linear/GLU/depthwise/SiLU/Linear)
    # Phase 4:           FPGA FFN2 + final norm (mostly reuses FFN1 code)
    # Phase 5:           audio_embed_project (output_proj + multimodal embedder)

    def audio_config_init(self) -> None:
        """Parse Gemma4AudioConfig values from the local HF model directory's
        config.json. Cached on self for use by audio_weight_init/tensor_init."""
        if hasattr(self, '_audio_cfg'):
            return  # already initialized
        import json as _json
        model_dir = os.path.join(self.script_dir, self._cfg["paths"]["hf_model_dir"])
        cfg_path = os.path.join(model_dir, "config.json")
        with open(cfg_path) as f:
            full = _json.load(f)
        ac = full.get("audio_config")
        if ac is None:
            raise RuntimeError(
                f"audio_config missing from {cfg_path}. "
                f"This model does not include the audio encoder.")
        self._audio_cfg = ac
        self.AUD_H = ac["hidden_size"]                      # 1024
        self.AUD_HEADS = ac["num_attention_heads"]          # 8
        self.AUD_HEAD_DIM = ac["hidden_size"] // ac["num_attention_heads"]  # 128
        self.AUD_FFN = ac["hidden_size"] * 4                # 4096
        self.AUD_LAYERS = ac["num_hidden_layers"]           # 12
        self.AUD_CONV_K = ac["conv_kernel_size"]            # 5
        self.AUD_CHUNK = ac["attention_chunk_size"]         # 12
        self.AUD_CTX_LEFT = ac["attention_context_left"]    # 13
        self.AUD_CTX_RIGHT = ac["attention_context_right"]  # 0
        # context_size = chunk + (left-1) + right
        self.AUD_CTX = self.AUD_CHUNK + self.AUD_CTX_LEFT - 1 + self.AUD_CTX_RIGHT
        self.AUD_SOFT_CAP = ac["attention_logit_cap"]       # 50
        self.AUD_RESIDUAL_W = ac["residual_weight"]         # 0.5
        self.AUD_RMS_EPS = ac["rms_norm_eps"]
        self.AUD_OUT_DIM = ac["output_proj_dims"]           # 1536 (LM hidden size)
        self.AUD_SUB_CHANS = ac["subsampling_conv_channels"]  # [128, 32]
        self.AUD_USE_CLIP = ac["use_clipped_linears"]
        self.AUD_INVALID_LOGIT = ac["attention_invalid_logits_value"]
        # q_scale = (head_dim^-0.5) / log(2),  k_scale = log(1 + e) / log(2)
        self.AUD_Q_SCALE = (self.AUD_HEAD_DIM ** -0.5) / math.log(2)
        self.AUD_K_SCALE = math.log(1.0 + math.e) / math.log(2)
        print(f"[Audio] config loaded: {self.AUD_LAYERS} layers, "
              f"H={self.AUD_H}, heads={self.AUD_HEADS}, FFN={self.AUD_FFN}, "
              f"conv_k={self.AUD_CONV_K}, chunk={self.AUD_CHUNK}, ctx={self.AUD_CTX}")

    def audio_weight_init(self, hf_model, *, reset_allocator: bool = True) -> None:
        """Upload Gemma4 audio encoder weights to FPGA DRAM.

        Quantization: matmul weights → FP4 q4_64 (matches vision encoder),
        norms / depthwise kernel / per_dim_scale → BF16.

        Stores per-layer DRAM addresses in self._aud_weight_addrs[i]:
            "FF1_W1" / "FF1_W2" / "Q_PROJ" / ... / "CONV_LIN_START" / ...
                Each is a {'data': ..., 'scale': ..., 'shape': (N, K)} dict
                for FP4 weights, OR a plain DRAM address int for BF16 weights.
        Per-layer ClippableLinear clip ranges in self._aud_clip_ranges[i].

        reset_allocator: if True (default for compare script), reset the
            tensor DRAM allocator to _tensor_base BEFORE uploading audio
            weights. This clobbers the LM tensor region, which is fine
            because the compare script doesn't run LM after audio. The
            test script (which interleaves audio + LM) should pass False
            and use the save/restore pattern in _run_audio_encoder_fpga.
        """
        self.audio_config_init()
        am = hf_model.model.audio_tower
        bpe = self.bytes_per_element

        if reset_allocator:
            # Save the LM tensor cursor — caller might want to restore it.
            self._aud_tensor_dram_save = self._tensor_dram_addr
            self.reset_tensor_dram_addr()
            print(f"\n[Audio] Resetting tensor allocator from 0x{self._aud_tensor_dram_save:X} "
                  f"to 0x{self._tensor_dram_addr:X} for audio weights")

        print("\n[Audio] Uploading audio encoder weights to DRAM (FP4 q4_64 + BF16) ...")

        def _upload_bf16(name: str, w: torch.Tensor) -> int:
            t = w.detach().cpu().to(torch.bfloat16).contiguous()
            sz = t.numel() * bpe
            addr = self.allocate_tensor_dram(sz)
            self.dma_to_accelerator_memory(addr, t)
            return addr

        def _upload_fp4(name: str, w: torch.Tensor) -> dict:
            """Quantize an [N, K] BF16 weight to FP4 E2M1 (q4_64) and DMA both
            data and scale tables. Returns the {'data','scale','shape'} dict
            our compile path expects to pass to matmat_mul_core."""
            w_bf16 = w.detach().cpu().to(torch.bfloat16).contiguous()
            assert w_bf16.dim() == 2, f"{name}: expected 2D weight, got {tuple(w_bf16.shape)}"
            N, K = w_bf16.shape
            assert K % 64 == 0, f"{name}: K={K} not divisible by 64 (FP4 block size)"
            data_bytes, scale_bytes = _quantize_bf16_to_fp4_packed(w_bf16, block_size=64)
            scale_addr = self.allocate_tensor_dram(len(scale_bytes))
            self.dma_write(DMA_DEVICE_H2C, scale_addr, scale_bytes, len(scale_bytes))
            data_addr = self.allocate_tensor_dram(len(data_bytes))
            self.dma_write(DMA_DEVICE_H2C, data_addr, data_bytes, len(data_bytes))
            return {"data": data_addr, "scale": scale_addr, "shape": (N, K)}

        # ---- Subsample conv weights (kept on host, used by audio_subsample_host)
        # Conv2d weights are tiny (~64 KB total) and we run subsampling on host
        # for Phase 1. Keep them as Python attrs so the host helper can call
        # F.conv2d directly without round-tripping through HF every time.
        sub = am.subsample_conv_projection
        self._aud_sub_w0_conv = sub.layer0.conv.weight.detach().cpu().to(torch.bfloat16)
        self._aud_sub_w0_norm = sub.layer0.norm.weight.detach().cpu().to(torch.bfloat16)
        self._aud_sub_w1_conv = sub.layer1.conv.weight.detach().cpu().to(torch.bfloat16)
        self._aud_sub_w1_norm = sub.layer1.norm.weight.detach().cpu().to(torch.bfloat16)
        self._aud_sub_proj_w  = sub.input_proj_linear.weight.detach().cpu().to(torch.bfloat16)

        # ---- Per-layer encoder weights ----
        # Cache BF16 host copies of weights used by host-fallback ops
        # (chunked attention, depthwise conv). These avoid having to
        # dequantize FP4 or re-load HF model later.
        hf_cache: list[dict] = []
        layer_addrs: list[dict] = []
        clip_ranges: list[dict] = []
        for li in range(self.AUD_LAYERS):
            L = am.layers[li]
            addrs: dict = {}
            crs: dict = {}
            hfc: dict = {}
            # Host-only weight cache for the chunked-attn and depthwise host fallbacks.
            hfc["rel_k_w"] = L.self_attn.relative_k_proj.weight.detach().cpu().to(torch.bfloat16)
            hfc["o_w"] = L.self_attn.post.linear.weight.detach().cpu().to(torch.bfloat16)
            hfc["dw_w"] = L.lconv1d.depthwise_conv1d.weight.detach().cpu().to(torch.bfloat16).squeeze(1)
            hf_cache.append(hfc)

            # FFN1 macaron half (Gemma4AudioFeedForward)
            ff1 = L.feed_forward1
            addrs["FF1_PRE_NORM"]  = _upload_bf16("ff1_pre_norm", ff1.pre_layer_norm.weight)
            addrs["FF1_W1"]        = _upload_fp4("ff1_w1", ff1.ffw_layer_1.linear.weight)
            addrs["FF1_W2"]        = _upload_fp4("ff1_w2", ff1.ffw_layer_2.linear.weight)
            addrs["FF1_POST_NORM"] = _upload_bf16("ff1_post_norm", ff1.post_layer_norm.weight)
            crs["FF1_W1"] = self._extract_clips(ff1.ffw_layer_1)
            crs["FF1_W2"] = self._extract_clips(ff1.ffw_layer_2)

            # Self-attention (Gemma4AudioAttention)
            sa = L.self_attn
            addrs["ATTN_PRE_NORM"] = _upload_bf16("attn_pre_norm", L.norm_pre_attn.weight)
            addrs["Q_PROJ"] = _upload_fp4("q_proj", sa.q_proj.linear.weight)
            addrs["K_PROJ"] = _upload_fp4("k_proj", sa.k_proj.linear.weight)
            addrs["V_PROJ"] = _upload_fp4("v_proj", sa.v_proj.linear.weight)
            addrs["O_PROJ"] = _upload_fp4("o_proj", sa.post.linear.weight)
            addrs["REL_K_PROJ"]    = _upload_fp4("rel_k_proj", sa.relative_k_proj.weight)
            addrs["PER_DIM_SCALE"] = _upload_bf16("per_dim_scale", sa.per_dim_scale)
            addrs["ATTN_POST_NORM"] = _upload_bf16("attn_post_norm", L.norm_post_attn.weight)
            crs["Q_PROJ"] = self._extract_clips(sa.q_proj)
            crs["K_PROJ"] = self._extract_clips(sa.k_proj)
            crs["V_PROJ"] = self._extract_clips(sa.v_proj)
            crs["O_PROJ"] = self._extract_clips(sa.post)

            # Conv module (Gemma4AudioLightConv1d)
            cv = L.lconv1d
            addrs["CONV_PRE_NORM"]  = _upload_bf16("conv_pre_norm", cv.pre_layer_norm.weight)
            addrs["CONV_LIN_START"] = _upload_fp4("conv_lin_start", cv.linear_start.linear.weight)
            # depthwise: weight shape [hidden, 1, k] → store as (hidden, k) BF16
            dw_w = cv.depthwise_conv1d.weight.detach().cpu().to(torch.bfloat16).squeeze(1)
            addrs["CONV_DW_W"]   = _upload_bf16("conv_dw_w", dw_w)
            addrs["CONV_NORM"]   = _upload_bf16("conv_norm", cv.conv_norm.weight)
            addrs["CONV_LIN_END"] = _upload_fp4("conv_lin_end", cv.linear_end.linear.weight)
            crs["CONV_LIN_START"] = self._extract_clips(cv.linear_start)
            crs["CONV_LIN_END"]   = self._extract_clips(cv.linear_end)

            # FFN2 macaron half
            ff2 = L.feed_forward2
            addrs["FF2_PRE_NORM"]  = _upload_bf16("ff2_pre_norm", ff2.pre_layer_norm.weight)
            addrs["FF2_W1"]        = _upload_fp4("ff2_w1", ff2.ffw_layer_1.linear.weight)
            addrs["FF2_W2"]        = _upload_fp4("ff2_w2", ff2.ffw_layer_2.linear.weight)
            addrs["FF2_POST_NORM"] = _upload_bf16("ff2_post_norm", ff2.post_layer_norm.weight)
            crs["FF2_W1"] = self._extract_clips(ff2.ffw_layer_1)
            crs["FF2_W2"] = self._extract_clips(ff2.ffw_layer_2)

            # Final per-layer norm
            addrs["NORM_OUT"] = _upload_bf16("norm_out", L.norm_out.weight)

            layer_addrs.append(addrs)
            clip_ranges.append(crs)

        self._aud_weight_addrs = layer_addrs
        self._aud_clip_ranges = clip_ranges
        self._aud_hf_layers = hf_cache

        # ---- Output projection (1024 → 1536) and multimodal embedder
        # output_proj has bias=True, the embedder embedding_projection has bias=False.
        # Multimodal embedder is on hf_model.model.embed_audio (mirrors embed_vision).
        self._aud_output_proj_w = am.output_proj.weight.detach().cpu().to(torch.bfloat16)
        self._aud_output_proj_b = am.output_proj.bias.detach().cpu().to(torch.bfloat16)
        ea = hf_model.model.embed_audio
        self._aud_embedder_proj_w = ea.embedding_projection.weight.detach().cpu().to(torch.bfloat16)

        print(f"[Audio] uploaded {self.AUD_LAYERS} layers + subsample + projector "
              f"(tensor DRAM at 0x{self.get_tensor_dram_addr():X})")

    @staticmethod
    def _extract_clips(clippable_linear) -> dict:
        """Pull (in_min, in_max, out_min, out_max) from a Gemma4ClippableLinear,
        defaulting to ±inf if `use_clipped_linears` is False or bounds aren't loaded.
        """
        cl = clippable_linear
        if not getattr(cl, "use_clipped_linears", False):
            return {"in_min": float("-inf"), "in_max": float("inf"),
                    "out_min": float("-inf"), "out_max": float("inf")}
        return {
            "in_min": float(cl.input_min.item()),
            "in_max": float(cl.input_max.item()),
            "out_min": float(cl.output_min.item()),
            "out_max": float(cl.output_max.item()),
        }

    def audio_tensor_init(self, num_frames: int) -> None:
        """Allocate intermediate DRAM buffers for the Conformer encoder. All
        sized for the *padded* L_pad = ceil(num_frames / 64) * 64 frames.

        Phase 1 only allocates buffers used by FFN1; Phase 2/3 add attn/conv.
        """
        self.audio_config_init()
        bpe = self.bytes_per_element
        H = self.AUD_H
        FF = self.AUD_FFN
        VS = UE_VECTOR_SIZE  # 64

        L_pad = ((num_frames + VS - 1) // VS) * VS
        self._aud_num_frames = num_frames
        self._aud_L_pad = L_pad
        print(f"\n[Audio] Allocating audio tensor DRAM for {num_frames} frames (L_pad={L_pad})")

        # Layer I/O double-buffered (so layer i reads from one and writes to the other)
        self.AUD_IO_A = self.allocate_tensor_dram(L_pad * H * bpe)
        self.AUD_IO_B = self.allocate_tensor_dram(L_pad * H * bpe)

        # Norm output (used as input to all post-norm matmuls in a layer)
        self.AUD_NORM_OUT = self.allocate_tensor_dram(L_pad * H * bpe)

        # Saved residual for half-step macaron
        self.AUD_RESIDUAL = self.allocate_tensor_dram(L_pad * H * bpe)

        # FFN intermediate (S × 4*H)
        self.AUD_FFN_MID = self.allocate_tensor_dram(L_pad * FF * bpe)
        # FFN second-stage output (back to S × H)
        self.AUD_FFN_OUT = self.allocate_tensor_dram(L_pad * H * bpe)

        # SiLU scratch — needed because silu_core_dram reads x WHILE writing
        # sigmoid(x), so input and output buffers must be distinct. Same size
        # as AUD_FFN_MID (L_pad × FF).
        self.AUD_SILU_OUT = self.allocate_tensor_dram(L_pad * FF * bpe)

        # Identity matrices for SiLU/GLU sigmoid-via-matmul. We need TWO
        # because matmat reads B with row stride = N (the matmul output dim).
        # If we passed an FFxFF identity to a 1024x1024 matmul, the row stride
        # mismatch would corrupt the result. Allocate one per N value used:
        #   AUD_IDENTITY_FF: 4096x4096 — used by FFN1/FFN2 SiLU
        #   AUD_IDENTITY_H : 1024x1024 — used by conv module SiLU and GLU
        self.AUD_IDENTITY_FF = self.allocate_tensor_dram(FF * FF * bpe)
        self.dma_to_accelerator_memory(self.AUD_IDENTITY_FF,
            torch.eye(FF, dtype=torch.bfloat16).contiguous())
        self.AUD_IDENTITY_H = self.allocate_tensor_dram(H * H * bpe)
        self.dma_to_accelerator_memory(self.AUD_IDENTITY_H,
            torch.eye(H, dtype=torch.bfloat16).contiguous())
        # Backwards-compat alias used by Phase 1 FFN1 code
        self.AUD_IDENTITY = self.AUD_IDENTITY_FF

        # Phase 2 buffers (attention scratch). These are small.
        HD = self.AUD_HEAD_DIM
        NH = self.AUD_HEADS
        self.AUD_Q   = self.allocate_tensor_dram(L_pad * NH * HD * bpe)
        self.AUD_K   = self.allocate_tensor_dram(L_pad * NH * HD * bpe)
        self.AUD_V   = self.allocate_tensor_dram(L_pad * NH * HD * bpe)
        self.AUD_REL_K_PROJ_OUT = self.allocate_tensor_dram(self.AUD_CTX_LEFT * NH * HD * bpe)
        self.AUD_ATTN_OUT = self.allocate_tensor_dram(L_pad * H * bpe)

        # Phase 3 (conv module) and Phase 5 (audio projector) buffers will be
        # allocated by their own helpers when those phases land. Skipping them
        # here saves the per-layer Toeplitz matrices (12 × 1024 × L_pad² × 2 B)
        # which dominate audio tensor DRAM at any non-trivial L_pad.

        print(f"[Audio] tensor DRAM end: 0x{self.get_tensor_dram_addr():X} "
              f"(usage: {self.get_tensor_dram_usage()/(1024*1024):.1f} MB)")

    def audio_subsample_host(self,
                              input_features: torch.Tensor,
                              input_features_mask: torch.Tensor | None = None
                              ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the SubSampleConvProjection stem on host. Mirrors
        Gemma4AudioSubSampleConvProjection.forward.

        Inputs:
            input_features: [B, T, n_mels] from the feature extractor (B=1).
            input_features_mask: [B, T] boolean valid-frame mask.

        Returns:
            hidden_states: [B, T_sub, hidden_size] post-subsample BF16
            mask:          [B, T_sub] downsampled mask
        """
        self.audio_config_init()
        if not hasattr(self, "_aud_sub_w0_conv"):
            raise RuntimeError("audio_weight_init must be called before audio_subsample_host")

        with torch.no_grad():
            x = input_features.to(torch.bfloat16).unsqueeze(1)  # [B, 1, T, n_mels]
            mask = input_features_mask
            # ---- layer 0 ----
            if mask is not None:
                x = x * mask[:, None, :, None].to(x.dtype)
            x = F.conv2d(x, self._aud_sub_w0_conv, bias=None, stride=2, padding=1)
            # LayerNorm over channel dim (out_channels). HF runs it as
            # norm(x.permute(0,2,3,1)).permute(0,3,1,2). bias=False.
            x_perm = x.permute(0, 2, 3, 1).float()
            x_perm = F.layer_norm(x_perm, (x_perm.shape[-1],),
                                  weight=self._aud_sub_w0_norm.float(), bias=None,
                                  eps=self._audio_cfg["rms_norm_eps"])
            x = x_perm.permute(0, 3, 1, 2).contiguous().to(torch.bfloat16)
            x = F.relu(x)
            if mask is not None:
                mask = mask[:, ::2]
            # ---- layer 1 ----
            if mask is not None:
                x = x * mask[:, None, :, None].to(x.dtype)
            x = F.conv2d(x, self._aud_sub_w1_conv, bias=None, stride=2, padding=1)
            x_perm = x.permute(0, 2, 3, 1).float()
            x_perm = F.layer_norm(x_perm, (x_perm.shape[-1],),
                                  weight=self._aud_sub_w1_norm.float(), bias=None,
                                  eps=self._audio_cfg["rms_norm_eps"])
            x = x_perm.permute(0, 3, 1, 2).contiguous().to(torch.bfloat16)
            x = F.relu(x)
            if mask is not None:
                mask = mask[:, ::2]
            # ---- final input_proj_linear ----
            B, _, T_sub, _ = x.shape
            x = x.permute(0, 2, 3, 1).contiguous().reshape(B, T_sub, -1)
            x = F.linear(x.float(), self._aud_sub_proj_w.float()).to(torch.bfloat16)
        return x, mask

    def audio_rel_pos_host(self) -> torch.Tensor:
        """Compute the Gemma4 audio relative-position encoding on host.
        Returns [context_size, hidden_size] BF16.

        Mirrors Gemma4AudioRelPositionalEncoding.forward but does not need a
        hidden_states tensor — the table only depends on context_size and
        hidden_size, both of which come from config.
        """
        self.audio_config_init()
        H = self.AUD_H
        # NOTE: Gemma4 hardcodes ``position_ids = arange(12, -1, -1)`` (13 positions)
        # in HF Gemma4AudioRelPositionalEncoding.forward; this matches the
        # default chunk=12, ctx_left=13, ctx_right=0 case where context_size = 24.
        # The "13" comes from chunk_size+1 (or context_left). For other configs
        # we follow the same pattern: arange(context_left, -1, -1).
        num_pos = self.AUD_CTX_LEFT
        with torch.no_grad():
            num_timescales = H // 2
            log_inc = math.log(10000.0 / 1.0) / max(num_timescales - 1, 1)
            inv_ts = torch.exp(torch.arange(num_timescales, dtype=torch.float32) * -log_inc)
            pos = torch.arange(num_pos - 1, -1, -1, dtype=torch.float32).unsqueeze(-1)  # [num_pos, 1]
            scaled = pos * inv_ts.unsqueeze(0)  # [num_pos, num_timescales]
            pe = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return pe.to(torch.bfloat16)

    def audio_embed_project_host(self, encoder_out: torch.Tensor) -> torch.Tensor:
        """Run output_proj (1024→1536, with bias) and the multimodal embedder
        (RMSNorm with_scale=False + Linear 1536→1536 bias=False) on host.

        Mirrors Gemma4AudioModel.output_proj followed by Gemma4MultimodalEmbedder.
        Phase 5 will move both stages onto FPGA.

        encoder_out: [N, 1024] from the encoder cumulative pass.
        Returns audio_features: [N, 1536] BF16, ready to merge into the LM
            embedding stream like image features.
        """
        if not hasattr(self, "_aud_output_proj_w"):
            raise RuntimeError("audio_weight_init must be called before audio_embed_project_host")
        with torch.no_grad():
            x = encoder_out.detach().cpu()
            if x.dim() == 3:
                x = x.squeeze(0)
            x = F.linear(x.float(), self._aud_output_proj_w.float(),
                         bias=self._aud_output_proj_b.float())  # [N, 1536]
            # Multimodal embedder: RMSNorm (with_scale=False) + Linear (no bias)
            ms = x.pow(2).mean(-1, keepdim=True) + self._audio_cfg["rms_norm_eps"]
            x = x * torch.pow(ms, -0.5)
            x = F.linear(x, self._aud_embedder_proj_w.float())
        return x.to(torch.bfloat16)

    def compile_audio_layer_ffn1(self, layer_idx: int) -> None:
        """Compile + run FFN1 macaron half for one Conformer layer.

        Reads from AUD_IO_A (or B depending on layer parity), writes back to
        the same buffer. The half-step residual is applied in place.

        Sequence (mirrors Gemma4AudioFeedForward.forward):
            residual = x
            x = clamp(x)                                  ← skipped, BF16 won't overflow
            x = pre_layer_norm(x)                         ← RMSNorm
            x = ffw_layer_1(x)                            ← Linear 1024→4096 (Clippable)
            x = SiLU(x)                                   ← x * sigmoid(x)
            x = ffw_layer_2(x)                            ← Linear 4096→1024 (Clippable)
            x = clamp(x)                                  ← skipped
            x = post_layer_norm(x)                        ← RMSNorm
            x = x * residual_weight  +  residual          ← *0.5 + residual
        """
        self.audio_config_init()
        H = self.AUD_H
        FF = self.AUD_FFN
        L_pad = self._aud_L_pad
        S = self._aud_num_frames  # only the first S rows are valid; rest are padding
        w = self._aud_weight_addrs[layer_idx]
        cr = self._aud_clip_ranges[layer_idx]

        IN_BUF = self.AUD_IO_A  # all layers run in place on IO_A; ping-pong was a misdesign

        # Save residual (for half-step add at the end)
        self._aud_copy_buf("ffn1_save_residual", IN_BUF, self.AUD_RESIDUAL, L_pad * H, row_n=H)

        # 1. pre_layer_norm
        self._compile_and_run_single("aud_ff1_pre_norm", lambda: self.rms_norm_core_dram(
            M=L_pad, N=H,
            A_DRAM_ADDR=IN_BUF,
            OUTPUT_DRAM_ADDR=self.AUD_NORM_OUT,
            GAMMA_DRAM_ADDR=w["FF1_PRE_NORM"]))

        # 2. ffw_layer_1 (clippable, FP4 q4_64). Apply input clip on host,
        #    run matmul, apply output clip on host.
        self._host_clip_dram(self.AUD_NORM_OUT, (L_pad, H),
                              cr["FF1_W1"]["in_min"], cr["FF1_W1"]["in_max"])
        ff1w1 = w["FF1_W1"]
        self._compile_and_run_single("aud_ff1_w1", lambda: self.matmat_mul_core(
            M=L_pad, K=H, N=FF,
            A_DRAM_ADDR=self.AUD_NORM_OUT,
            B_DRAM_ADDR=ff1w1["data"],
            OUTPUT_DRAM_ADDR=self.AUD_FFN_MID,
            is_B_quantized=True,
            data_type=TYPE.FP4,
            SCALE_DRAM_ADDR=ff1w1["scale"]))
        self._host_clip_dram(self.AUD_FFN_MID, (L_pad, FF),
                              cr["FF1_W1"]["out_min"], cr["FF1_W1"]["out_max"])

        # 3. SiLU on (L_pad, FF) — out goes to a SEPARATE buffer because the
        # silu helper does an in-place sigmoid on its OUTPUT and would
        # destroy x if A_DRAM == OUTPUT_DRAM.
        self._compile_and_run_single("aud_ff1_silu", lambda: _aud_silu(
            self, L_pad, FF, self.AUD_FFN_MID, self.AUD_SILU_OUT, self.AUD_IDENTITY_FF))

        # 4. ffw_layer_2 (clippable, FP4 q4_64). Reads from SiLU output.
        self._host_clip_dram(self.AUD_SILU_OUT, (L_pad, FF),
                              cr["FF1_W2"]["in_min"], cr["FF1_W2"]["in_max"])
        ff1w2 = w["FF1_W2"]
        self._compile_and_run_single("aud_ff1_w2", lambda: self.matmat_mul_core(
            M=L_pad, K=FF, N=H,
            A_DRAM_ADDR=self.AUD_SILU_OUT,
            B_DRAM_ADDR=ff1w2["data"],
            OUTPUT_DRAM_ADDR=self.AUD_FFN_OUT,
            is_B_quantized=True,
            data_type=TYPE.FP4,
            SCALE_DRAM_ADDR=ff1w2["scale"]))
        self._host_clip_dram(self.AUD_FFN_OUT, (L_pad, H),
                              cr["FF1_W2"]["out_min"], cr["FF1_W2"]["out_max"])

        # 5. post_layer_norm
        self._compile_and_run_single("aud_ff1_post_norm", lambda: self.rms_norm_core_dram(
            M=L_pad, N=H,
            A_DRAM_ADDR=self.AUD_FFN_OUT,
            OUTPUT_DRAM_ADDR=self.AUD_FFN_OUT,
            GAMMA_DRAM_ADDR=w["FF1_POST_NORM"]))

        # 6. Half-step residual: out = residual + 0.5 * ffn_out → IN_BUF
        self._compile_and_run_single("aud_ff1_half_residual", lambda: _aud_half_step(
            self, L_pad, H,
            self.AUD_RESIDUAL, self.AUD_FFN_OUT, IN_BUF))

    def compile_audio_layer_attn(self, layer_idx: int) -> None:
        """Run the self-attention block of one Conformer layer.

        Sequence (mirrors Gemma4AudioLayer.forward + Gemma4AudioAttention.forward):

            residual = x          # (the layer's running hidden, IN_BUF)
            x = norm_pre_attn(x)  # RMSNorm  ← FPGA
            q = q_proj(x); k = k_proj(x); v = v_proj(x)  ← FPGA (FP4)
            q = q * q_scale * softplus(per_dim_scale)
            k = k * k_scale
            ▼ chunked local attention (rel-pos, soft-cap, mask, softmax) ← HOST for now
            attn_out = post(attn_out)   ← FPGA (FP4)
            x = norm_post_attn(attn_out)  ← FPGA
            x = x + residual               ← FPGA eltwise

        IN_BUF parity: layer 0 reads from AUD_IO_A, layer 1 from AUD_IO_B,
        etc. We write the post-attn-residual result back to the SAME buffer
        the layer started in (so the conv module reads from there).
        """
        self.audio_config_init()
        H = self.AUD_H
        HD = self.AUD_HEAD_DIM
        NH = self.AUD_HEADS
        L_pad = self._aud_L_pad
        T = self._aud_num_frames
        w = self._aud_weight_addrs[layer_idx]
        cr = self._aud_clip_ranges[layer_idx]

        IN_BUF = self.AUD_IO_A  # all layers run in place on IO_A; ping-pong was a misdesign

        # Save residual: copy current IN_BUF state into AUD_RESIDUAL.
        # Note: AUD_RESIDUAL is reused by every sub-block (FFN1, attn, conv,
        # FFN2) within this layer. Each sub-block must save its own residual
        # before any FPGA writes happen.
        self._aud_copy_buf("attn_save_residual", IN_BUF, self.AUD_RESIDUAL,
                            L_pad * H, row_n=H)

        # 1. norm_pre_attn (RMSNorm with learned scale)
        self._compile_and_run_single("aud_attn_pre_norm", lambda: self.rms_norm_core_dram(
            M=L_pad, N=H,
            A_DRAM_ADDR=IN_BUF,
            OUTPUT_DRAM_ADDR=self.AUD_NORM_OUT,
            GAMMA_DRAM_ADDR=w["ATTN_PRE_NORM"]))

        # 2. Q / K / V projections (FP4 q4_64, ClippableLinear).
        # Writes go directly into AUD_Q / AUD_K / AUD_V which are sized
        # (L_pad, NH*HD) = (L_pad, H) bf16.
        for proj_name, addr_key, dst in [
            ("Q_PROJ", "Q_PROJ", self.AUD_Q),
            ("K_PROJ", "K_PROJ", self.AUD_K),
            ("V_PROJ", "V_PROJ", self.AUD_V),
        ]:
            self._host_clip_dram(self.AUD_NORM_OUT, (L_pad, H),
                                  cr[addr_key]["in_min"], cr[addr_key]["in_max"])
            wq = w[addr_key]
            label = f"aud_attn_{proj_name.lower()}"
            self._compile_and_run_single(label, lambda d=dst, ww=wq: self.matmat_mul_core(
                M=L_pad, K=H, N=H,
                A_DRAM_ADDR=self.AUD_NORM_OUT,
                B_DRAM_ADDR=ww["data"],
                OUTPUT_DRAM_ADDR=d,
                is_B_quantized=True, data_type=TYPE.FP4,
                SCALE_DRAM_ADDR=ww["scale"]))
            self._host_clip_dram(dst, (L_pad, H),
                                  cr[addr_key]["out_min"], cr[addr_key]["out_max"])

        # 3..N: chunked local attention on HOST (Phase 2 will move this to FPGA).
        # We pull Q, K, V back, run the exact HF attention math, and DMA the
        # post-projection result back to AUD_ATTN_OUT.
        attn_out_host = self._aud_chunked_attn_host(layer_idx, T)  # [T, H] bf16
        attn_out_padded = torch.zeros(L_pad, H, dtype=torch.bfloat16)
        attn_out_padded[:T] = attn_out_host
        self.dma_to_accelerator_memory(self.AUD_ATTN_OUT, attn_out_padded.contiguous())

        # 4. norm_post_attn (RMSNorm)
        self._compile_and_run_single("aud_attn_post_norm", lambda: self.rms_norm_core_dram(
            M=L_pad, N=H,
            A_DRAM_ADDR=self.AUD_ATTN_OUT,
            OUTPUT_DRAM_ADDR=self.AUD_ATTN_OUT,
            GAMMA_DRAM_ADDR=w["ATTN_POST_NORM"]))

        # 5. Residual add: out = norm_post_attn(attn) + saved residual → IN_BUF
        self._compile_and_run_single("aud_attn_residual", lambda: _aud_eltwise_add(
            self, L_pad, H, self.AUD_RESIDUAL, self.AUD_ATTN_OUT, IN_BUF))

    def _aud_chunked_attn_host(self, layer_idx: int, T: int) -> torch.Tensor:
        """HOST implementation of Gemma4 chunked local self-attention.

        Reads Q, K, V from FPGA DRAM, runs the exact HF attention math
        (rel-pos bias, soft-cap tanh, mask, softmax, attn@V), then runs the
        post (output) projection — also on host since the input length is
        small. Returns [T, H] bf16 (the OUT proj output).

        This is a temporary fallback while we figure out the chunked-local
        kernel layout. Future Phase will compile this onto FPGA.
        """
        H = self.AUD_H
        HD = self.AUD_HEAD_DIM
        NH = self.AUD_HEADS
        L_pad = self._aud_L_pad
        chunk_size = self.AUD_CHUNK
        max_past = self.AUD_CTX_LEFT - 1
        max_future = self.AUD_CTX_RIGHT
        context_size = chunk_size + max_past + max_future
        soft_cap = self.AUD_SOFT_CAP
        invalid_logit = self.AUD_INVALID_LOGIT
        w_layer = self._aud_weight_addrs[layer_idx]
        cr = self._aud_clip_ranges[layer_idx]

        # 1. Read Q, K, V from DRAM (only the first T rows are real)
        Q = self.dma_from_accelerator_memory(self.AUD_Q, (L_pad, H)).cpu()[:T].float()
        K = self.dma_from_accelerator_memory(self.AUD_K, (L_pad, H)).cpu()[:T].float()
        V = self.dma_from_accelerator_memory(self.AUD_V, (L_pad, H)).cpu()[:T].float()

        # Reshape to [B=1, T, NH, HD]
        Q = Q.view(1, T, NH, HD)
        K = K.view(1, T, NH, HD)
        V = V.view(1, T, NH, HD)

        # 2. Per-head scaling — match HF lines 220-221, 278-279.
        per_dim_scale = self.dma_from_accelerator_memory(
            w_layer["PER_DIM_SCALE"], (HD,)).cpu().float()
        q_scale = self.AUD_Q_SCALE
        k_scale = self.AUD_K_SCALE
        Q = Q * q_scale * F.softplus(per_dim_scale)
        K = K * k_scale

        # 3. _convert_to_block on Q
        num_blocks = (T + chunk_size - 1) // chunk_size
        pad = num_blocks * chunk_size - T
        Q_pad = F.pad(Q, (0, 0, 0, 0, 0, pad))
        Q_blocks = Q_pad.view(1, num_blocks, chunk_size, NH, HD).contiguous()

        # 4. _extract_block_context on K and V
        K_pad = F.pad(K, (0, 0, 0, 0, max_past, max_future + chunk_size - 1))
        V_pad = F.pad(V, (0, 0, 0, 0, max_past, max_future + chunk_size - 1))
        K_ctx = K_pad.unfold(1, context_size, chunk_size)  # [1, num_blocks, NH, HD, context_size]
        V_ctx = V_pad.unfold(1, context_size, chunk_size)
        K_ctx = torch.movedim(K_ctx, -1, 2).contiguous()   # [1, num_blocks, context_size, NH, HD]
        V_ctx = torch.movedim(V_ctx, -1, 2).contiguous()

        # 5. relative_k_proj on the relative position embedding
        pos_emb = self.audio_rel_pos_host().float()  # [num_pos, H]
        # relative_k_proj weight is FP4-quantized on FPGA. Pull a host BF16
        # version from HF for now (we have it via the linear's `linear.weight`
        # before it was quantized — but we didn't keep it). Re-extract from
        # HF model. For Phase 1 we accept this small inefficiency.
        # NOTE: simpler approach — recompute from HF model directly.
        # This avoids dequantizing the FP4 representation.
        rel_k_w = self._get_audio_rel_k_proj_weight(layer_idx).float()  # [H, H]
        rel_k = pos_emb @ rel_k_w.T  # [num_pos, H]
        rel_k = rel_k.view(-1, NH, HD)

        # 6. Q @ K^T
        queries = Q_blocks.permute(0, 3, 1, 2, 4)  # [B, NH, num_blocks, chunk, HD]
        K_ctx_perm = K_ctx.permute(0, 3, 1, 4, 2)  # [B, NH, num_blocks, HD, context]
        matrix_ac = queries @ K_ctx_perm  # [B, NH, num_blocks, chunk, context]

        # 7. Q @ rel_k
        queries_flat = queries.reshape(1, NH, -1, HD)  # [B, NH, num_blocks*chunk, HD]
        rel_k_perm = rel_k.permute(1, 2, 0)  # [NH, HD, num_pos]
        matrix_bd = queries_flat @ rel_k_perm  # [B, NH, num_blocks*chunk, num_pos]
        matrix_bd = matrix_bd.reshape(1, NH, num_blocks, chunk_size, -1)
        matrix_bd = self._aud_rel_shift_host(matrix_bd, context_size)

        # 8. add + soft-cap + mask + softmax
        attn_w = matrix_ac + matrix_bd
        attn_w = soft_cap * torch.tanh(attn_w / soft_cap)

        # Mask: build the same blocked 5D mask HF builds via
        # _convert_4d_mask_to_blocked_5d.
        mask_5d = self._aud_make_blocked_mask(T, num_blocks, chunk_size,
                                                max_past, max_future)
        attn_w = attn_w.masked_fill(~mask_5d, invalid_logit)
        attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(V_ctx.dtype)

        # 9. attn @ V
        V_ctx_perm = V_ctx.permute(0, 3, 1, 2, 4)  # [B, NH, num_blocks, context, HD]
        attn_out = attn_w @ V_ctx_perm  # [B, NH, num_blocks, chunk, HD]
        attn_out = attn_out.permute(0, 2, 3, 1, 4).reshape(1, num_blocks * chunk_size, -1)
        attn_out = attn_out[:, :T].contiguous()  # [1, T, H]

        # 10. post (output) projection — apply Clippable input/output clipping
        attn_out_clip = attn_out.squeeze(0).clamp(
            min=cr["O_PROJ"]["in_min"], max=cr["O_PROJ"]["in_max"])
        out_w = self._get_audio_o_proj_weight(layer_idx).float()  # [H, H]
        result = attn_out_clip.float() @ out_w.T
        result = result.clamp(min=cr["O_PROJ"]["out_min"], max=cr["O_PROJ"]["out_max"])
        return result.to(torch.bfloat16)

    def _aud_rel_shift_host(self, x: torch.Tensor, context_size: int) -> torch.Tensor:
        """Host port of Gemma4AudioAttention._rel_shift."""
        batch_size, num_heads, num_blocks, block_size, position_length = x.shape
        x = F.pad(x, (0, context_size + 1 - position_length))
        x = x.view(batch_size, num_heads, num_blocks, block_size * (context_size + 1))
        x = x[..., : block_size * context_size]
        return x.view(batch_size, num_heads, num_blocks, block_size, context_size)

    def _aud_make_blocked_mask(self, T: int, num_blocks: int, chunk_size: int,
                                max_past: int, max_future: int) -> torch.Tensor:
        """Build the 5D blocked attention mask matching HF's
        _convert_4d_mask_to_blocked_5d for a single sequence of length T,
        with no padding (B=1). Result: [1, 1, num_blocks, chunk_size, context_size].
        """
        padded_seq_len = num_blocks * chunk_size
        # 4D causal-by-distance mask: True if (q_idx - kv_idx) within window
        valid = torch.zeros(padded_seq_len, padded_seq_len, dtype=torch.bool)
        valid[:T, :T] = True
        # Sliding-window constraint
        i_idx = torch.arange(padded_seq_len).unsqueeze(1)
        j_idx = torch.arange(padded_seq_len).unsqueeze(0)
        dist = i_idx - j_idx
        within = ((dist >= 0) & (dist < max_past + 1)) | ((dist < 0) & (-dist <= max_future))
        valid = valid & within
        # Pad to [num_blocks * chunk_size + max_past + max_future] on the kv axis
        mask_5d = valid.view(1, 1, num_blocks, chunk_size, padded_seq_len)
        mask_5d = F.pad(mask_5d, (max_past, max_future), value=False)
        block_starts = torch.arange(num_blocks) * chunk_size
        offsets = torch.arange(chunk_size + max_past + max_future)
        kv_indices = block_starts[:, None] + offsets[None, :]
        kv_indices = kv_indices[None, None, :, None, :].expand(1, 1, -1, chunk_size, -1)
        return mask_5d.gather(-1, kv_indices)

    def _get_audio_rel_k_proj_weight(self, layer_idx: int) -> torch.Tensor:
        """Lazily fetch the BF16 relative_k_proj weight from the host-cached
        HF model (we kept references during audio_weight_init via _aud_hf)."""
        return self._aud_hf_layers[layer_idx]["rel_k_w"]

    def _get_audio_o_proj_weight(self, layer_idx: int) -> torch.Tensor:
        return self._aud_hf_layers[layer_idx]["o_w"]

    def compile_audio_layer_conv(self, layer_idx: int) -> None:
        """Run the lconv1d (light Conv1d) module of one Conformer layer.

        Sequence (mirrors Gemma4AudioLightConv1d.forward):
            residual = x
            x = pre_layer_norm(x)             ← FPGA RMSNorm
            x = linear_start(x)               ← FPGA matmul (1024 → 2048)
            x = GLU(x)                        ← FPGA helper (split halves + sigmoid + mul)
            x = depthwise_conv1d(x)           ← HOST (Phase 3 will move to FPGA)
            x = conv_norm(x)                  ← FPGA RMSNorm
            x = SiLU(x)                       ← FPGA helper
            x = linear_end(x)                 ← FPGA matmul (1024 → 1024)
            x = x + residual                  ← FPGA eltwise
        """
        self.audio_config_init()
        H = self.AUD_H
        L_pad = self._aud_L_pad
        T = self._aud_num_frames
        w = self._aud_weight_addrs[layer_idx]
        cr = self._aud_clip_ranges[layer_idx]
        IN_BUF = self.AUD_IO_A  # all layers run in place on IO_A; ping-pong was a misdesign

        self._aud_copy_buf("conv_save_residual", IN_BUF, self.AUD_RESIDUAL, L_pad * H, row_n=H)

        # 1. pre_layer_norm
        self._compile_and_run_single("aud_conv_pre_norm", lambda: self.rms_norm_core_dram(
            M=L_pad, N=H,
            A_DRAM_ADDR=IN_BUF,
            OUTPUT_DRAM_ADDR=self.AUD_NORM_OUT,
            GAMMA_DRAM_ADDR=w["CONV_PRE_NORM"]))

        # 2. linear_start (1024 → 2048). Output goes to AUD_FFN_MID temporarily
        # (which is L_pad × FF=4096 = enough for L_pad × 2H=2048).
        self._host_clip_dram(self.AUD_NORM_OUT, (L_pad, H),
                              cr["CONV_LIN_START"]["in_min"], cr["CONV_LIN_START"]["in_max"])
        cls = w["CONV_LIN_START"]
        self._compile_and_run_single("aud_conv_lin_start", lambda: self.matmat_mul_core(
            M=L_pad, K=H, N=2 * H,
            A_DRAM_ADDR=self.AUD_NORM_OUT,
            B_DRAM_ADDR=cls["data"],
            OUTPUT_DRAM_ADDR=self.AUD_FFN_MID,
            is_B_quantized=True, data_type=TYPE.FP4,
            SCALE_DRAM_ADDR=cls["scale"]))
        self._host_clip_dram(self.AUD_FFN_MID, (L_pad, 2 * H),
                              cr["CONV_LIN_START"]["out_min"], cr["CONV_LIN_START"]["out_max"])

        # 3. GLU: split (L_pad, 2H) into gate=(L_pad, H) and value=(L_pad, H),
        # then output = gate * sigmoid(value).
        # The linear output is laid out as [L_pad, 2H] row-major. Row r:
        #   gate[r]  = AUD_FFN_MID[r, 0:H]
        #   value[r] = AUD_FFN_MID[r, H:2H]
        # We can pass these as DRAM bases to glu_core_dram, but the helper
        # assumes both halves are in CONTIGUOUS L_pad×H buffers, not a single
        # interleaved buffer. Easiest fix: copy each half to its own buffer.
        # AUD_CONV_GATE/VALUE were dropped in tensor_init's Phase-3 cleanup;
        # reuse AUD_NORM_OUT (L_pad×H, free at this point) as the gate buffer
        # and AUD_RESIDUAL would clobber the residual we need later — so use
        # AUD_FFN_OUT (L_pad×H) as the value buffer instead.
        # Then the helper writes the GLU output back over AUD_NORM_OUT.
        # row_bytes_2H = 2*H*bpe; gate at offset 0, value at offset H*bpe per row.
        self._aud_split_2h_to_halves("conv_glu_split",
            self.AUD_FFN_MID, self.AUD_NORM_OUT, self.AUD_FFN_OUT, L_pad, H)
        self._compile_and_run_single("aud_conv_glu", lambda: _aud_glu(
            self, L_pad, H,
            self.AUD_NORM_OUT,  # GATE (a)
            self.AUD_FFN_OUT,   # VALUE (b, will be sigmoided in place)
            self.AUD_NORM_OUT,  # OUTPUT
            self.AUD_IDENTITY_H))  # H×H identity for K=N=H matmul

        # 4. depthwise_conv1d (HOST for Phase 1; will move to FPGA in Phase 3)
        glu_out = self.dma_from_accelerator_memory(self.AUD_NORM_OUT, (L_pad, H)).cpu()
        dw_w = self._aud_hf_layers[layer_idx]["dw_w"]  # [H, k]
        dw_out_host = self._aud_depthwise_conv1d_host(glu_out[:T], dw_w)
        dw_padded = torch.zeros(L_pad, H, dtype=torch.bfloat16)
        dw_padded[:T] = dw_out_host
        self.dma_to_accelerator_memory(self.AUD_NORM_OUT, dw_padded.contiguous())

        # 5. conv_norm (RMSNorm with learned scale)
        self._compile_and_run_single("aud_conv_norm", lambda: self.rms_norm_core_dram(
            M=L_pad, N=H,
            A_DRAM_ADDR=self.AUD_NORM_OUT,
            OUTPUT_DRAM_ADDR=self.AUD_NORM_OUT,
            GAMMA_DRAM_ADDR=w["CONV_NORM"]))

        # 6. SiLU
        self._compile_and_run_single("aud_conv_silu", lambda: _aud_silu(
            self, L_pad, H, self.AUD_NORM_OUT, self.AUD_FFN_OUT, self.AUD_IDENTITY_H))

        # 7. linear_end (1024 → 1024)
        self._host_clip_dram(self.AUD_FFN_OUT, (L_pad, H),
                              cr["CONV_LIN_END"]["in_min"], cr["CONV_LIN_END"]["in_max"])
        cle = w["CONV_LIN_END"]
        self._compile_and_run_single("aud_conv_lin_end", lambda: self.matmat_mul_core(
            M=L_pad, K=H, N=H,
            A_DRAM_ADDR=self.AUD_FFN_OUT,
            B_DRAM_ADDR=cle["data"],
            OUTPUT_DRAM_ADDR=self.AUD_FFN_OUT,
            is_B_quantized=True, data_type=TYPE.FP4,
            SCALE_DRAM_ADDR=cle["scale"]))
        self._host_clip_dram(self.AUD_FFN_OUT, (L_pad, H),
                              cr["CONV_LIN_END"]["out_min"], cr["CONV_LIN_END"]["out_max"])

        # 8. Residual: out = AUD_FFN_OUT + AUD_RESIDUAL → IN_BUF
        self._compile_and_run_single("aud_conv_residual", lambda: _aud_eltwise_add(
            self, L_pad, H, self.AUD_RESIDUAL, self.AUD_FFN_OUT, IN_BUF))

    def _aud_split_2h_to_halves(self, label: str,
                                 src_2h: int, dst_a: int, dst_b: int,
                                 L_pad: int, H: int) -> None:
        """Split (L_pad, 2H) row-major buffer into (L_pad, H) gate and value
        buffers via strided SRAM copies. Compiled as one program."""
        bpe = self.bytes_per_element
        def _fn():
            # Row-by-row strided copy from interleaved to two contiguous halves.
            # Use accelerator_memory_to_sram with stride for the gather.
            # Strided gather: read H elements from each row at offset 0, jump 2H per row
            self.accelerator_memory_to_sram(
                accelerator_dram_address=src_2h,
                sram_address=0x00000,
                element_size=L_pad * H,
                stride_bytes_per_chunk=H * bpe,
                stride_jump_bytes=2 * H * bpe,
            )
            self.sram_to_accelerator_memory(0x00000, dst_a, L_pad * H)
            self.accelerator_memory_to_sram(
                accelerator_dram_address=src_2h + H * bpe,
                sram_address=0x00000,
                element_size=L_pad * H,
                stride_bytes_per_chunk=H * bpe,
                stride_jump_bytes=2 * H * bpe,
            )
            self.sram_to_accelerator_memory(0x00000, dst_b, L_pad * H)
        self._compile_and_run_single(label, _fn)

    def _aud_depthwise_conv1d_host(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Host fallback for Gemma4AudioCausalConv1d.
        x: [T, H], w: [H, k]. Returns [T, H] bf16."""
        T, H = x.shape
        k = w.shape[1]
        # Causal padding: left_pad = k-1, no right pad
        x_t = x.transpose(0, 1).unsqueeze(0).float()  # [1, H, T]
        x_p = F.pad(x_t, (k - 1, 0))
        w_t = w.unsqueeze(1).float()  # [H, 1, k] for groups=H depthwise conv
        out = F.conv1d(x_p, w_t, bias=None, stride=1, padding=0, groups=H)
        return out.squeeze(0).transpose(0, 1).to(torch.bfloat16)  # [T, H]

    def compile_audio_layer_ffn2(self, layer_idx: int) -> None:
        """Run the FFN2 macaron half. Identical to FFN1 except for the
        weight keys (FF2_*) so we just call _compile_audio_ffn_macaron with
        the FF2 weight prefix."""
        self._compile_audio_ffn_macaron(layer_idx, prefix="FF2")

    def _compile_audio_ffn_macaron(self, layer_idx: int, *, prefix: str) -> None:
        """Generic Gemma4AudioFeedForward macaron half: works for FFN1 and
        FFN2. ``prefix`` selects the weight keys: 'FF1' or 'FF2'."""
        H = self.AUD_H
        FF = self.AUD_FFN
        L_pad = self._aud_L_pad
        w = self._aud_weight_addrs[layer_idx]
        cr = self._aud_clip_ranges[layer_idx]
        IN_BUF = self.AUD_IO_A  # all layers run in place on IO_A; ping-pong was a misdesign

        self._aud_copy_buf(f"{prefix.lower()}_save_residual",
                            IN_BUF, self.AUD_RESIDUAL, L_pad * H, row_n=H)

        self._compile_and_run_single(f"aud_{prefix.lower()}_pre_norm", lambda: self.rms_norm_core_dram(
            M=L_pad, N=H,
            A_DRAM_ADDR=IN_BUF,
            OUTPUT_DRAM_ADDR=self.AUD_NORM_OUT,
            GAMMA_DRAM_ADDR=w[f"{prefix}_PRE_NORM"]))

        self._host_clip_dram(self.AUD_NORM_OUT, (L_pad, H),
                              cr[f"{prefix}_W1"]["in_min"], cr[f"{prefix}_W1"]["in_max"])
        w1 = w[f"{prefix}_W1"]
        self._compile_and_run_single(f"aud_{prefix.lower()}_w1", lambda: self.matmat_mul_core(
            M=L_pad, K=H, N=FF,
            A_DRAM_ADDR=self.AUD_NORM_OUT,
            B_DRAM_ADDR=w1["data"],
            OUTPUT_DRAM_ADDR=self.AUD_FFN_MID,
            is_B_quantized=True, data_type=TYPE.FP4,
            SCALE_DRAM_ADDR=w1["scale"]))
        self._host_clip_dram(self.AUD_FFN_MID, (L_pad, FF),
                              cr[f"{prefix}_W1"]["out_min"], cr[f"{prefix}_W1"]["out_max"])

        self._compile_and_run_single(f"aud_{prefix.lower()}_silu", lambda: _aud_silu(
            self, L_pad, FF, self.AUD_FFN_MID, self.AUD_SILU_OUT, self.AUD_IDENTITY_FF))

        self._host_clip_dram(self.AUD_SILU_OUT, (L_pad, FF),
                              cr[f"{prefix}_W2"]["in_min"], cr[f"{prefix}_W2"]["in_max"])
        w2 = w[f"{prefix}_W2"]
        self._compile_and_run_single(f"aud_{prefix.lower()}_w2", lambda: self.matmat_mul_core(
            M=L_pad, K=FF, N=H,
            A_DRAM_ADDR=self.AUD_SILU_OUT,
            B_DRAM_ADDR=w2["data"],
            OUTPUT_DRAM_ADDR=self.AUD_FFN_OUT,
            is_B_quantized=True, data_type=TYPE.FP4,
            SCALE_DRAM_ADDR=w2["scale"]))
        self._host_clip_dram(self.AUD_FFN_OUT, (L_pad, H),
                              cr[f"{prefix}_W2"]["out_min"], cr[f"{prefix}_W2"]["out_max"])

        self._compile_and_run_single(f"aud_{prefix.lower()}_post_norm", lambda: self.rms_norm_core_dram(
            M=L_pad, N=H,
            A_DRAM_ADDR=self.AUD_FFN_OUT,
            OUTPUT_DRAM_ADDR=self.AUD_FFN_OUT,
            GAMMA_DRAM_ADDR=w[f"{prefix}_POST_NORM"]))

        self._compile_and_run_single(f"aud_{prefix.lower()}_half_residual",
            lambda: _aud_half_step(self, L_pad, H,
                self.AUD_RESIDUAL, self.AUD_FFN_OUT, IN_BUF))

    def compile_audio_layer_norm_out(self, layer_idx: int) -> None:
        """Final per-layer RMSNorm. Writes back into IN_BUF in place."""
        H = self.AUD_H
        L_pad = self._aud_L_pad
        w = self._aud_weight_addrs[layer_idx]
        IN_BUF = self.AUD_IO_A  # all layers run in place on IO_A; ping-pong was a misdesign
        self._compile_and_run_single("aud_norm_out", lambda: self.rms_norm_core_dram(
            M=L_pad, N=H,
            A_DRAM_ADDR=IN_BUF,
            OUTPUT_DRAM_ADDR=IN_BUF,
            GAMMA_DRAM_ADDR=w["NORM_OUT"]))

    def run_audio_layer(self, layer_idx: int) -> int:
        """Run a complete Conformer layer in place: FFN1 → attn → conv →
        FFN2 → norm_out. Returns the DRAM address holding the result.

        All sub-blocks read and write AUD_IO_A; we never ping-pong, so the
        return value is always AUD_IO_A. Caller can DMA from there.
        """
        self.compile_audio_layer_ffn1(layer_idx)
        self.compile_audio_layer_attn(layer_idx)
        self.compile_audio_layer_conv(layer_idx)
        self.compile_audio_layer_ffn2(layer_idx)
        self.compile_audio_layer_norm_out(layer_idx)
        return self.AUD_IO_A

    def _aud_copy_buf(self, label: str, src: int, dst: int, n_elems: int,
                       row_n: int | None = None) -> None:
        """Copy n_elems bf16 elements from src DRAM to dst DRAM via SRAM,
        chunked so URAM_A doesn't overflow on long buffers.

        row_n: optional row width for clean row-aligned chunking.
        """
        def _fn():
            _aud_copy_chunked(self, src, dst, n_elems, row_n=row_n)
        self._compile_and_run_single(label, _fn)

    def get_embedding_for_tokens(self, token_ids: list[int] | tuple) -> torch.Tensor:
        """Return (len(token_ids), vector_length) bfloat16 tensor from self.embedding_weight (HF, scale applied)."""
        tid_t = torch.tensor(token_ids, dtype=torch.long)
        out = torch.zeros(len(token_ids), self.vector_length, dtype=torch.bfloat16)
        valid = tid_t < self.embedding_weight.shape[0]
        out[valid] = self.embedding_weight[tid_t[valid]]
        return out

    def _load_rope_host(self, rope_theta: float | None = None, rope_local_base: float | None = None) -> None:
        """Generate RoPE (cos, cos, -sin, sin) on host and write to DRAM. Uses config for sizes and num_positions.
        LOCAL: head_dim=256, full rotation. GLOBAL: head_dim=512, partial rotation (first 128 dims)."""
        rope_cfg = self._cfg["special"]["rope"]
        theta = rope_theta if rope_theta is not None else rope_cfg["theta"]
        local_base = rope_local_base if rope_local_base is not None else rope_cfg["local_base"]
        num_rope_positions = rope_cfg["num_positions"]
        partial_rotary_factor = rope_cfg["partial_rotary_factor_global"]

        # LOCAL RoPE: head_dim_sliding=256, full rotation, D=128
        D_local = self.head_dim_sliding // 2  # 128
        inv_freq_local = 1.0 / (local_base ** (torch.arange(D_local, dtype=torch.float32) / D_local))
        pos = torch.arange(num_rope_positions, dtype=torch.float32)
        freqs_local = torch.outer(pos, inv_freq_local)
        cos_local = freqs_local.cos().to(torch.bfloat16)
        sin_local = freqs_local.sin().to(torch.bfloat16)
        rope_local = torch.cat([cos_local, cos_local, -sin_local, sin_local], dim=1)
        sz = self.weight_defs["ROPE_LOCAL_SIZE"]
        raw = rope_local.contiguous().view(torch.uint8).numpy().tobytes()
        raw = (raw + b"\x00" * sz)[:sz]
        addr = self.allocate_params_dram(sz)
        self.dma_write(DMA_DEVICE_H2C, addr, raw, sz)
        self.DRAM_ADDR_ROPE_LOCAL = addr

        # GLOBAL RoPE: head_dim=512, partial_rotary_factor=0.25, rotary_dims=128, D=64
        rotary_dims = int(self.head_dim * partial_rotary_factor)  # 128
        D_global = rotary_dims // 2  # 64
        inv_freq_global = 1.0 / (theta ** (torch.arange(D_global, dtype=torch.float32) / D_global))
        freqs_global = torch.outer(pos, inv_freq_global)
        cos_global = freqs_global.cos().to(torch.bfloat16)
        sin_global = freqs_global.sin().to(torch.bfloat16)
        rope_global = torch.cat([cos_global, cos_global, -sin_global, sin_global], dim=1)
        sz = self.weight_defs["ROPE_GLOBAL_SIZE"]
        raw = rope_global.contiguous().view(torch.uint8).numpy().tobytes()
        raw = (raw + b"\x00" * sz)[:sz]
        addr = self.allocate_params_dram(sz)
        self.dma_write(DMA_DEVICE_H2C, addr, raw, sz)
        self.DRAM_ADDR_ROPE_GLOBAL = addr

    def weight_init(self) -> None:
        """Ensure weight bin exists (generate from HF if missing), load it, then initialize DRAM: embedding, layers from bin, RoPE, OUTPUT_NORM/LM_HEAD.
        Also loads per-layer embedding and projection weights for host-side computation."""
        full_path = os.path.join(self.script_dir, self._weights_bin_rel)
        if os.path.exists(full_path):
            print(f"Weight bin exists, skip generation: {full_path}")
        else:
            print(f"Weight bin not found, generating: {full_path}")
            weight_bin_generate(output_path=full_path)
        with open(full_path, "rb") as f:
            self.weight_bin = f.read()

        model, model_dir = _ensure_hf_model(self.script_dir, self._cfg)
        text_model = model.model.language_model

        # Embedding: scale by sqrt(hidden_size)
        embed = text_model.embed_tokens.weight.detach().cpu().to(torch.bfloat16)
        embedding_scale = self.vector_length ** 0.5
        self.embedding_weight = (embed.float() * embedding_scale).to(torch.bfloat16)

        # Per-layer embedding for host-side lookup: [262144, 8960], pre-scaled by sqrt(per_layer_input_dim)
        per_layer_embed_scale = self.per_layer_input_dim ** 0.5  # sqrt(256) = 16.0
        self.embed_tokens_per_layer_weight = (text_model.embed_tokens_per_layer.weight.detach().cpu().float() * per_layer_embed_scale).to(torch.bfloat16)

        # Per-layer model projection weight for host-side computation: [8960, 1536] (transposed for matmul)
        self.per_layer_model_proj_weight = text_model.per_layer_model_projection.weight.detach().cpu().to(torch.bfloat16)  # [8960, 1536]

        # Per-layer projection norm weight for host-side computation (raw weight, no gamma_offset)
        # Host-side norm uses w directly; gamma_offset is only for HW norm (HW internally adds 1.0)
        self.per_layer_proj_norm_weight = text_model.per_layer_projection_norm.weight.detach().cpu().to(torch.bfloat16)  # [256]

        # Layer scalars and KV sharing map
        self._layer_scalars = []
        for layer_idx in range(self.LAYER_SIZE):
            layer = text_model.layers[layer_idx]
            self._layer_scalars.append(layer.layer_scalar.item())
            attn = layer.self_attn
            if attn.is_kv_shared_layer and attn.kv_shared_layer_index is not None:
                self._kv_shared_map[layer_idx] = attn.kv_shared_layer_index

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        base_layer0 = self.weight_defs["BLK0_ATTN_NORM_WEIGHT"]
        blk0_regions = [
            (s["key"], f"{s['key']}_SIZE", s["attr"])
            for s in self._cfg["layers"]["structure"]
        ]
        non_layer = [
            (s["key"], f"{s['key']}_SIZE", s["attr"])
            for s in self._cfg["layers"]["non_layer"]
            if s["key"] not in ("ROPE_LOCAL", "ROPE_GLOBAL")  # RoPE loaded via _load_rope_host()
        ]

        last_structure_key = self._cfg["layers"]["structure"][-1]["key"]
        layer0_end = (self.weight_defs[last_structure_key] - base_layer0
                      + self.weight_defs[f"{last_structure_key}_SIZE"])
        assert layer0_end <= LAYER_WEIGHT_SIZE, (
            f"Layer 0 size overflow: computed {layer0_end} > LAYER_WEIGHT_SIZE {LAYER_WEIGHT_SIZE}"
        )

        print(f"\n--- Weights DRAM allocation, start at DRAM address: 0x{self.get_params_dram_addr():X} ---")
        layers_total = self.LAYER_SIZE * LAYER_WEIGHT_SIZE
        layers_base_dram = self.allocate_params_dram(layers_total)
        for layer_idx in range(self.LAYER_SIZE):
            for off_key, sz_key, attr in blk0_regions:
                off = self.weight_defs[off_key]
                sz = self.weight_defs[sz_key]
                bin_off = off + layer_idx * LAYER_WEIGHT_SIZE
                raw = self.weight_bin[bin_off : bin_off + sz]
                offset_in_layer = off - base_layer0
                dram_addr = layers_base_dram + layer_idx * LAYER_WEIGHT_SIZE + offset_in_layer
                self.dma_write(DMA_DEVICE_H2C, dram_addr, raw, sz)
            if layer_idx == 0:
                for off_key, sz_key, attr in blk0_regions:
                    off = self.weight_defs[off_key]
                    offset_in_layer = off - base_layer0
                    setattr(self, attr, layers_base_dram + offset_in_layer)
        print(f"Layers 0..{self.LAYER_SIZE - 1} loaded: 0x{layers_base_dram:X} size {layers_total} (LAYER_WEIGHT_SIZE={LAYER_WEIGHT_SIZE})")

        for off_key, sz_key, attr in non_layer:
            off = self.weight_defs[off_key]
            sz = self.weight_defs[sz_key]
            raw = self.weight_bin[off : off + sz]
            addr = self.allocate_params_dram(sz)
            self.dma_write(DMA_DEVICE_H2C, addr, raw, sz)
            setattr(self, attr, addr)

        self._load_rope_host()
        print(f"    Allocate weights end at DRAM address: 0x{self.get_params_dram_addr():X}, usage: {self.get_params_dram_usage()} bytes")
        print("Tokenizer loaded successfully.")

    def tensor_init(self) -> None:
        """Initialize hardware DRAM for gemma4 E2B model (layer-wise overlap except for kv cache).
        KV cache uses max head_dim (512) for uniform sizing."""
        seq_len = self.MAX_CONTEXT_SIZE
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        # Flash attention scratch/bias buffers are only used at full capacity
        # during prefill. Sizing them for MAX_CONTEXT_SIZE wastes ~200+ MB that
        # the decoder program bin would otherwise overlap with. Use
        # max_prefill_seq_len (capped at MAX_CONTEXT_SIZE) instead.
        prefill_seq_len = min(self.max_prefill_seq_len, self.MAX_CONTEXT_SIZE)
        prefill_q_seq_len = prefill_seq_len * self.group_size
        prefill_aligned_seq_len = ((prefill_q_seq_len + 63) // 64) * 64

        print(f"Allocate tensor dram start at DRAM address: 0x{self.get_tensor_dram_addr():X}")
        # Build compact KV slot map: only layers that own KV state get a slot.
        # KV-shared layers point at their reference layer's slot, so L15-34 do not
        # consume cache space (saves ~40 MB at MAX_CONTEXT_SIZE=1024).
        non_shared_layers = [l for l in range(self.LAYER_SIZE) if l not in self._kv_shared_map]
        self._kv_slot_for_layer = {}
        for slot, l in enumerate(non_shared_layers):
            self._kv_slot_for_layer[l] = slot
        for shared_l, ref_l in self._kv_shared_map.items():
            self._kv_slot_for_layer[shared_l] = self._kv_slot_for_layer[ref_l]
        self._num_kv_slots = len(non_shared_layers)
        _kv_saved = (self.LAYER_SIZE - self._num_kv_slots) * self.MAX_CONTEXT_SIZE * self.k_size * 2  # K+V
        print(f"KV cache: {self._num_kv_slots} unique slots (of {self.LAYER_SIZE} layers), saved {_kv_saved / (1024*1024):.1f} MB via KV sharing")
        # Allocate shared memory for k v cache (k rope and v projection) and zero pad for decoder use:
        # Uses max head_dim (512) = self.k_size for uniform sizing
        self.LAYER0_V_DRAM = self.allocate_tensor_dram(self._num_kv_slots * self.MAX_CONTEXT_SIZE * self.k_size)
        self.LAYER0_K_ROPE_DRAM = self.allocate_tensor_dram(self._num_kv_slots * self.MAX_CONTEXT_SIZE * self.k_size)
        zero_pad = torch.zeros(self._num_kv_slots * self.MAX_CONTEXT_SIZE * self.k_size, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_K_ROPE_DRAM, zero_pad)
        # Allocate memory for constant zero tensor, identity matrix, and bias:
        zero_add = torch.zeros(seq_len * self.head_dim * self.bytes_per_element, dtype=torch.bfloat16)
        self.ZERO_DRAM_ADDR = self.allocate_tensor_dram(seq_len * self.head_dim * self.bytes_per_element)
        self.dma_to_accelerator_memory(self.ZERO_DRAM_ADDR, zero_add)
        self.IDENTITY_DRAM_ADDR = self.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * self.bytes_per_element)
        self.dma_to_accelerator_memory(self.IDENTITY_DRAM_ADDR, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))
        # Allocate memory for flash attention and zero pad (sized for prefill only):
        self.LAYER0_FLASH_Q_DRAM = self.allocate_tensor_dram(prefill_aligned_seq_len * self.head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_K_DRAM = self.allocate_tensor_dram(prefill_aligned_seq_len * self.head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_V_DRAM = self.allocate_tensor_dram(prefill_aligned_seq_len * self.head_dim * self.bytes_per_element)
        zero_pad = torch.zeros(prefill_aligned_seq_len * self.head_dim * self.bytes_per_element, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_Q_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_K_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_V_DRAM, zero_pad)
        # Allocate memory for layer intermediate tensors:
        self.LAYER0_INPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_Q_DRAM = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_K_DRAM = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_K_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_Q_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_FLASH_OUTPUT_DRAM = self.allocate_tensor_dram(prefill_aligned_seq_len * self.head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_SCRATCH_DRAM = self.allocate_tensor_dram(max(self.head_dim, UE_FMAX_CONTEXT_SIZE) * prefill_aligned_seq_len * 2 + self.head_dim * prefill_aligned_seq_len * 2)
        # Two flash-attention bias buffers: full-attention layers attend to
        # the entire causal window, sliding-attention layers are limited to
        # `sliding_window` tokens. compile_prefill / compile_decoder pick the
        # right address per layer; run_prefill / run_decoder upload both.
        # Sized for prefill (not MAX_CONTEXT_SIZE) — see comment above.
        self.LAYER0_FLASH_BIAS_FULL_DRAM = self.allocate_tensor_dram(prefill_aligned_seq_len * prefill_aligned_seq_len * self.bytes_per_element)
        self.LAYER0_FLASH_BIAS_SLIDING_DRAM = self.allocate_tensor_dram(prefill_aligned_seq_len * prefill_aligned_seq_len * self.bytes_per_element)
        # Backwards-compat alias (older callers use the singular name).
        self.LAYER0_FLASH_BIAS_DRAM = self.LAYER0_FLASH_BIAS_FULL_DRAM
        self.LAYER0_ATTN_PROJ_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_ATTN_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_ATTN_RESIDUAL_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_MLP_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        mlp_max = max(self.mlp_elements, self.mlp_elements_wide)
        self.LAYER0_MLP_GATE_DRAM = self.allocate_tensor_dram(seq_len * mlp_max * 2)
        self.LAYER0_MLP_UP_DRAM = self.allocate_tensor_dram(seq_len * mlp_max * 2)
        self.LAYER0_MLP_MULT_DRAM = self.allocate_tensor_dram(seq_len * mlp_max * 2)
        self.LAYER0_MLP_DOWN_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_MLP_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.OUTPUT_NORM_DRAM = self.allocate_tensor_dram(1 * self.vector_length * self.bytes_per_element)
        self.LOGITS_DRAM = self.allocate_tensor_dram(1 * self.EMBEDDING_ELEMENTS * self.bytes_per_element)

        # Per-layer input injection buffers
        # PER_LAYER_INPUTS_DRAM: holds per_layer_inputs for all layers: MAX_CONTEXT_SIZE x 35 x 256 x 2 bytes
        self.PER_LAYER_INPUTS_DRAM = self.allocate_tensor_dram(self.MAX_CONTEXT_SIZE * self.LAYER_SIZE * self.per_layer_input_dim * self.bytes_per_element)
        # Intermediate DRAMs for per-layer injection
        self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.per_layer_input_dim * self.bytes_per_element)
        self.LAYER0_PER_LAYER_PROJ_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * self.bytes_per_element)

        print(f"    Allocate tensor dram end at DRAM address: 0x{self.get_tensor_dram_addr():X}, usage: {self.get_tensor_dram_usage()} bytes")

    def program_execute(self, program_start_addr: int = user_dma_core.DRAM_INSTRUCTION_ADDR, timeout: float = 300.0, flops: float = None) -> None:
        """Execute compiled program from DRAM instruction memory.
        """
        print(f"Execute program start at 0x{program_start_addr:X}")
        self.start_execute_from_dram(program_start_addr)
        latency, flop_rate_program = 0, 0
        if timeout == 0:
            print("Program started")
        else:
            self.wait_queue(timeout)
            latency = self.report_latency_in_us()
            print(f"    Total program execution latency = {latency} us")
            if flops is not None:
                flop_rate_program, _ = self.report_flop_rate_gflops(flops)
                print(f"Report FLOPS for program execution: {flop_rate_program:.2f} GFLOPS")
        return latency, flop_rate_program

    def _get_layer_attention_dims(self, layer_idx: int) -> tuple[int, int, int]:
        """Return (cur_head_dim, cur_q_size, cur_k_size) for a given layer index."""
        if layer_idx in self._full_attention_layers:
            cur_head_dim = self.head_dim  # 512
            cur_q_size = cur_head_dim * self.group_size  # 4096
            cur_k_size = cur_head_dim  # 512
        else:
            cur_head_dim = self.head_dim_sliding  # 256
            cur_q_size = cur_head_dim * self.group_size  # 2048
            cur_k_size = cur_head_dim  # 256
        return cur_head_dim, cur_q_size, cur_k_size

    def _get_rope_dims(self, layer_idx: int) -> int:
        """Return the number of dims to apply RoPE on for a given layer.
        Sliding layers: full rotation on head_dim_sliding=256, so N=256.
        Full attention layers: partial rotation, only first 128 dims of head_dim=512, so N=128."""
        if layer_idx in self._full_attention_layers:
            partial_rotary_factor = self._cfg["special"]["rope"]["partial_rotary_factor_global"]
            return int(self.head_dim * partial_rotary_factor)  # 128
        else:
            return self.head_dim_sliding  # 256

    def _get_mlp_elements(self, layer_idx: int) -> int:
        """Return MLP intermediate size for a given layer (wide for KV-shared layers)."""
        if layer_idx >= self._double_wide_mlp_first:
            return self.mlp_elements_wide
        return self.mlp_elements

    def _compile_per_layer_injection(self, layer_idx: int, layer_off: int, seq_len: int) -> int:
        """Compile per-layer input injection block. Returns flops added."""
        total_flops = 0
        # gate = gelu(per_layer_input_gate @ hidden_state): Linear(1536->256) + GELU
        total_flops += self.matmat_mul_core(M=seq_len, K=self.vector_length, N=self.per_layer_input_dim,
            A_DRAM_ADDR=self.LAYER0_OUTPUT_DRAM,
            B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PER_LAYER_GATE + layer_off,
            OUTPUT_DRAM_ADDR=self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM,
            gelu_enable=True)
        # gated = gate * per_layer_input[layer_idx]  (small: seq_len * 256)
        per_layer_input_addr = self.PER_LAYER_INPUTS_DRAM + layer_idx * seq_len * self.per_layer_input_dim * self.bytes_per_element
        self._emit_sram_eltwise_chunked(
            "mul",
            self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM,
            per_layer_input_addr,
            self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM,
            seq_len * self.per_layer_input_dim)
        # projected = per_layer_projection @ gated: Linear(256->1536)
        total_flops += self.matmat_mul_core(M=seq_len, K=self.per_layer_input_dim, N=self.vector_length,
            A_DRAM_ADDR=self.LAYER0_PER_LAYER_GATE_OUTPUT_DRAM,
            B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PER_LAYER_PROJ + layer_off,
            OUTPUT_DRAM_ADDR=self.LAYER0_PER_LAYER_PROJ_OUTPUT_DRAM)

        # FUSED: rms_norm(projected) + (hidden + normed) + layer_scalar
        # Previously: rms_norm_core_dram wrote the normed output to DRAM,
        # then a chunked SRAM pass read it back for the residual add. This
        # DRAM write+read pair is eliminated by running the per-row RMS norm
        # on projection data that's ALREADY in URAM_A, then immediately
        # loading the residual hidden-state chunk into URAM_B and doing the
        # add + scalar-mul before the single final writeback.
        #
        # SRAM layout (chunked by row):
        #   URAM_A [0x10000..0x80000]: projection chunk in-place (M_chunk rows × N)
        #   URAM_B [0x80000..0x80000+N*bpe]: gamma (loaded once outside the loop)
        #   URAM_B [0x80000+N*bpe..]: hidden-state chunk (M_chunk rows × N)
        bpe = self.bytes_per_element
        N = self.vector_length  # 1536
        gamma_addr = self.DRAM_ADDR_LAYER0_POST_PER_LAYER_NORM_GAMMA + layer_off
        gamma_sram = 0x80000
        hidden_sram_base = 0x80000 + N * bpe  # leave room for gamma
        proj_sram_base = 0x10000
        # Max rows per chunk is bounded by whichever URAM region is smaller
        # after reserving Q (0x00000-0x10000), gamma, and accounting for
        # two-row-aligned layout.
        uram_a_free_elements = (0x80000 - proj_sram_base) // bpe   # 229376
        uram_b_free_elements = (0x100000 - hidden_sram_base) // bpe  # 259072
        max_rows_per_chunk = min(uram_a_free_elements, uram_b_free_elements) // N
        if max_rows_per_chunk < 1:
            max_rows_per_chunk = 1
        # Upload gamma once for all chunks
        self.accelerator_memory_to_sram(
            accelerator_dram_address=gamma_addr,
            sram_address=gamma_sram, element_size=N)
        for m_off in range(0, seq_len, max_rows_per_chunk):
            m_take = min(max_rows_per_chunk, seq_len - m_off)
            chunk_bytes_offset = m_off * N * bpe
            # Load projection chunk into URAM_A
            self.accelerator_memory_to_sram(
                accelerator_dram_address=self.LAYER0_PER_LAYER_PROJ_OUTPUT_DRAM + chunk_bytes_offset,
                sram_address=proj_sram_base, element_size=m_take * N)
            # Per-row RMS norm in-place with the URAM_B-resident gamma
            for row in range(m_take):
                row_sram = proj_sram_base + row * N * bpe
                self.rms_norm_core(row_sram, row_sram, N, gamma_sram)
            # Load hidden-state chunk into URAM_B (past gamma)
            self.accelerator_memory_to_sram(
                accelerator_dram_address=self.LAYER0_OUTPUT_DRAM + chunk_bytes_offset,
                sram_address=hidden_sram_base, element_size=m_take * N)
            # Residual add: normed (URAM_A) + hidden (URAM_B) → URAM_A
            self.eltwise_add_core(
                vector_A_sram_start_addr=proj_sram_base,
                vector_B_sram_start_addr=hidden_sram_base,
                vector_C_sram_wb_addr=proj_sram_base,
                element_size=m_take * N)
            # Multiply by layer_scalar in-place
            self.broadcast_mul(
                scalar=self._layer_scalars[layer_idx],
                sram_start_addr=proj_sram_base,
                sram_wb_addr=proj_sram_base,
                element_size=m_take * N)
            # Single writeback per chunk
            self.sram_to_accelerator_memory(
                sram_address=proj_sram_base,
                accelerator_dram_address=self.LAYER0_OUTPUT_DRAM + chunk_bytes_offset,
                element_size=m_take * N)
        return total_flops

    def compile_prefill(self, seq_len: int, layer_size: int = 35) -> dict:
        """
        Compile prefill for the given prefill sequence.

        Args:
            seq_len: The length of the prefill sequence.
            layer_size: The number of layers to compile.

        Returns:
            A tuple containing the address of the prefill program in DRAM and the number of GFLOPS.
        """
        seq_len -= 1
        self.seq_len = seq_len
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        if self.dual_engine:
            seq_len_engine0 = seq_len // 2
            seq_len_engine1 = seq_len - seq_len_engine0
        else:
            seq_len_engine0 = seq_len

        # --- Gemma4 E2B 35 layers: compile---
        global _SILENT_MODE
        _SILENT_MODE = True
        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()
        total_flops = 0
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        for layer_idx in range(layer_size):
            layer_off = layer_idx * LAYER_WEIGHT_SIZE
            cur_head_dim, cur_q_size, cur_k_size = self._get_layer_attention_dims(layer_idx)
            cur_mlp = self._get_mlp_elements(layer_idx)
            rope_n = self._get_rope_dims(layer_idx)

            if layer_idx != 0 and not self.engine_slave:
                self._emit_sram_copy_chunked(
                    self.LAYER0_OUTPUT_DRAM, self.LAYER0_INPUT_DRAM,
                    seq_len * self.vector_length)
            if not self.engine_slave:
                total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                                    OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off)
            if not self.engine_slave:
                # Master: set -> clear -> workload -> check | set -> clear -> workload -> check |
                self.generate_instruction_flag_set()
                self.generate_instruction_flag_clear()
                # Q projection: N = cur_q_size (actual per-layer Q output dim)
                total_flops += self.matmat_mul_core(M=seq_len_engine0, K=self.vector_length, N=cur_q_size,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.FP4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off,
                    )
                if layer_idx not in self._kv_shared_map:
                    # Non-shared layer: compute K/V projections normally.
                    # Shared layers skip entirely — their attention reads K/V directly
                    # from the reference layer's slot via _kv_slot_for_layer.
                    # K projection: N = cur_k_size (actual per-layer K output dim)
                    total_flops += self.matmat_mul_core(M=seq_len_engine0, K=self.vector_length, N=cur_k_size,
                        A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                        B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off,
                        OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                        is_B_quantized=True,
                        data_type=TYPE.FP4,
                        SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off,
                        )
                    # V projection: write to temp buffer first, then scatter to KV cache at k_size stride
                    total_flops += self.matmat_mul_core(M=seq_len_engine0, K=self.vector_length, N=cur_k_size,
                        A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                        B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off,
                        OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,  # temp buffer
                        is_B_quantized=True,
                        data_type=TYPE.FP4,
                        SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off,
                        )
                    # V norm + scatter to KV cache at k_size stride (matching decoder's V_CACHE_SIZE_REG)
                    v_cache_base = self.LAYER0_V_DRAM + self._kv_slot_for_layer[layer_idx] * self.MAX_CONTEXT_SIZE * self.k_size
                    for t in range(seq_len_engine0):
                        self.accelerator_memory_to_sram(self.LAYER0_FLASH_V_DRAM + t * cur_k_size * self.bytes_per_element, 0x10000, cur_k_size)
                        self.rms_norm_core(0x10000, 0x10000, cur_k_size)  # no gamma
                        self.sram_to_accelerator_memory(0x10000, v_cache_base + t * self.k_size, cur_k_size)
                if self.dual_engine:
                    self.generate_instruction_flag_check(target_engine_idx=1)
            else:
                # Slave(run first): check -> clear -> workload -> set | check -> clear -> workload -> set |
                self.generate_instruction_flag_check(target_engine_idx=0)
                self.generate_instruction_flag_clear()
                total_flops += self.matmat_mul_core(M=seq_len_engine1, K=self.vector_length, N=cur_q_size,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM + seq_len_engine0 * self.vector_length * self.bytes_per_element,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM + seq_len_engine0 * cur_q_size * self.bytes_per_element,
                    is_B_quantized=True,
                    data_type=TYPE.FP4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off,
                    )
                total_flops += self.matmat_mul_core(M=seq_len_engine1, K=self.vector_length, N=cur_k_size,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM + seq_len_engine0 * self.vector_length * self.bytes_per_element,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM + seq_len_engine0 * cur_k_size * self.bytes_per_element,
                    is_B_quantized=True,
                    data_type=TYPE.FP4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off,
                    )
                total_flops += self.matmat_mul_core(M=seq_len_engine1, K=self.vector_length, N=cur_k_size,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM + seq_len_engine0 * self.vector_length * self.bytes_per_element,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_V_DRAM + self._kv_slot_for_layer[layer_idx] * self.MAX_CONTEXT_SIZE * self.k_size + seq_len_engine0 * cur_k_size * self.bytes_per_element,
                    is_B_quantized=True,
                    data_type=TYPE.FP4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off,
                    )
                self.generate_instruction_flag_set()
            if not self.engine_slave:
                # Q norm always needed (Q is always computed fresh)
                total_flops += self.rms_norm_core_dram(M=seq_len * self.group_size, N=cur_head_dim, A_DRAM_ADDR=self.LAYER0_Q_DRAM,
                                OUTPUT_DRAM_ADDR=self.LAYER0_Q_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_NORM_GAMMA + layer_off)

                ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE_GLOBAL if layer_idx in self._rope_global_layers else self.DRAM_ADDR_ROPE_LOCAL

                if layer_idx not in self._kv_shared_map:
                    # Non-shared: K norm + K RoPE
                    total_flops += self.rms_norm_core_dram(M=seq_len, N=cur_head_dim, A_DRAM_ADDR=self.LAYER0_K_DRAM,
                                    OUTPUT_DRAM_ADDR=self.LAYER0_K_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_NORM_GAMMA + layer_off)
                    kv_slot_off = self._kv_slot_for_layer[layer_idx] * self.MAX_CONTEXT_SIZE * self.k_size
                    for t in range(seq_len):
                        total_flops += self.rope_hf_core(N=rope_n, input_dram_addr=self.LAYER0_K_NORM_DRAM + t * cur_head_dim * self.bytes_per_element, output_dram_addr=(self.LAYER0_K_ROPE_DRAM + kv_slot_off) + t * self.k_size,
                                    cos_dram_addr=ROPE_WEIGHT_ADDR + t * rope_n * 2 * self.bytes_per_element, sin_dram_addr=ROPE_WEIGHT_ADDR + t * rope_n * 2 * self.bytes_per_element + rope_n * self.bytes_per_element)
                # else: shared layers read K/V from reference layer's slot at attention time

                # Q RoPE: always runs (Q is always fresh)
                for t in range(seq_len):
                    for g in range(self.group_size):
                        total_flops += self.rope_hf_core(N=rope_n, input_dram_addr=self.LAYER0_Q_NORM_DRAM + (t * self.group_size + g) * cur_head_dim * self.bytes_per_element, output_dram_addr=self.LAYER0_FLASH_Q_DRAM + (t * self.group_size + g) * cur_head_dim * self.bytes_per_element,
                                    cos_dram_addr=ROPE_WEIGHT_ADDR + t * rope_n * 2 * self.bytes_per_element, sin_dram_addr=ROPE_WEIGHT_ADDR + t * rope_n * 2 * self.bytes_per_element + rope_n * self.bytes_per_element)

                # For full attention layers with partial rotation, copy non-rotated dims through
                # (The Q_NORM already has those dims, and rope_hf_core only wrote rope_n dims to FLASH_Q)
                if layer_idx in self._full_attention_layers and rope_n < cur_head_dim:
                    # Copy non-rotated dims (rope_n..cur_head_dim) from Q_NORM to FLASH_Q
                    for t in range(seq_len):
                        for g in range(self.group_size):
                            src = self.LAYER0_Q_NORM_DRAM + (t * self.group_size + g) * cur_head_dim * self.bytes_per_element + rope_n * self.bytes_per_element
                            dst = self.LAYER0_FLASH_Q_DRAM + (t * self.group_size + g) * cur_head_dim * self.bytes_per_element + rope_n * self.bytes_per_element
                            remaining = cur_head_dim - rope_n
                            self.accelerator_memory_to_sram(src, 0x10000, remaining)
                            self.sram_to_accelerator_memory(0x10000, dst, remaining)
                    # Copy non-rotated dims for K (use k_size stride for KV cache).
                    # Skip shared layers: their K_NORM holds stale data and the ref
                    # layer's slot already has the correct non-rotated dims.
                    if layer_idx not in self._kv_shared_map:
                        kv_slot_off = self._kv_slot_for_layer[layer_idx] * self.MAX_CONTEXT_SIZE * self.k_size
                        for t in range(seq_len):
                            src = self.LAYER0_K_NORM_DRAM + t * cur_head_dim * self.bytes_per_element + rope_n * self.bytes_per_element
                            dst = (self.LAYER0_K_ROPE_DRAM + kv_slot_off) + t * self.k_size + rope_n * self.bytes_per_element
                            remaining = cur_head_dim - rope_n
                            self.accelerator_memory_to_sram(src, 0x10000, remaining)
                            self.sram_to_accelerator_memory(0x10000, dst, remaining)

                # Duplicate k rope and v projection to GQA flash attention:
                # KV cache uses k_size stride per token (matching decoder's V_CACHE_SIZE_REG).
                # Shared layers read from reference layer's slot via _kv_slot_for_layer.
                kv_slot_off = self._kv_slot_for_layer[layer_idx] * self.MAX_CONTEXT_SIZE * self.k_size
                k_cache_base = self.LAYER0_K_ROPE_DRAM + kv_slot_off
                v_cache_base = self.LAYER0_V_DRAM + kv_slot_off
                for i in range(seq_len):
                    self.accelerator_memory_to_sram(k_cache_base + i * self.k_size, 0x10000 + i * cur_head_dim * self.bytes_per_element, cur_head_dim)
                for i in range(seq_len):
                    for g in range(self.group_size):
                        self.sram_to_accelerator_memory(0x10000 + i * cur_head_dim * self.bytes_per_element, self.LAYER0_FLASH_K_DRAM + (g + i * self.group_size) * cur_head_dim * self.bytes_per_element, cur_head_dim)

                for i in range(seq_len):
                    self.accelerator_memory_to_sram(v_cache_base + i * self.k_size, 0x20000 + i * cur_head_dim * self.bytes_per_element, cur_head_dim)
                for i in range(seq_len):
                    for g in range(self.group_size):
                        self.sram_to_accelerator_memory(0x20000 + i * cur_head_dim * self.bytes_per_element, self.LAYER0_FLASH_V_DRAM + (g + i * self.group_size) * cur_head_dim * self.bytes_per_element, cur_head_dim)

                # Gemma4 uses scaling=1.0 (no 1/sqrt(d) in attention scores).
                # flash_attention_core internally applies 1/sqrt(head_dim), so pre-scale Q by sqrt(head_dim) to cancel it.
                self._emit_sram_broadcast_mul_chunked(
                    self.LAYER0_FLASH_Q_DRAM, self.LAYER0_FLASH_Q_DRAM,
                    aligned_seq_len * cur_head_dim, math.sqrt(cur_head_dim))

                # Pick the per-layer bias: full attention layers see the
                # entire causal window; sliding-attention layers are limited
                # to `sliding_window` tokens (run_prefill builds both biases).
                bias_addr_layer = (self.LAYER0_FLASH_BIAS_FULL_DRAM
                                   if layer_idx in self._full_attention_layers
                                   else self.LAYER0_FLASH_BIAS_SLIDING_DRAM)
                total_flops += self._flash_attention_core_cached(
                    head_dim=cur_head_dim,
                    seq_len=aligned_seq_len,
                    Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
                    K_DRAM_ADDR=self.LAYER0_FLASH_K_DRAM,
                    V_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                    OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                    SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                    BIAS_DRAM_ADDR=bias_addr_layer,
                )
            if not self.engine_slave:
                # Master: set -> clear -> workload -> check | set -> clear -> workload -> check |
                self.generate_instruction_flag_set()
                self.generate_instruction_flag_clear()
                # O projection: INT4, K=cur_q_size
                total_flops += self.matmat_mul_core(M=seq_len_engine0, K=cur_q_size, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.FP4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off,
                    )
                if self.dual_engine:
                    self.generate_instruction_flag_check(target_engine_idx=1)
            else:
                # Slave(run first): check -> clear -> workload -> set | check -> clear -> workload -> set |
                self.generate_instruction_flag_check(target_engine_idx=0)
                self.generate_instruction_flag_clear()
                total_flops += self.matmat_mul_core(M=seq_len_engine1, K=cur_q_size, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM + seq_len_engine0 * cur_q_size * self.bytes_per_element,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM + seq_len_engine0 * self.vector_length * self.bytes_per_element,
                    is_B_quantized=True,
                    data_type=TYPE.FP4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off,
                    )
                self.generate_instruction_flag_set()
            if not self.engine_slave:
                total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                                OUTPUT_DRAM_ADDR=self.LAYER0_POST_ATTN_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_NORM_GAMMA + layer_off)
                self._emit_sram_eltwise_chunked(
                    "add", self.LAYER0_INPUT_DRAM, self.LAYER0_POST_ATTN_NORM_DRAM,
                    self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                    seq_len * self.vector_length)
                total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                                OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off)
            if not self.engine_slave:
                # Master: set -> clear -> workload -> check | set -> clear -> workload -> check |
                self.generate_instruction_flag_set()
                self.generate_instruction_flag_clear()
                total_flops += self.matmat_mul_core(M=seq_len_engine0, K=self.vector_length, N=cur_mlp,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.FP4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off,
                    gelu_enable=True,
                    )
                total_flops += self.matmat_mul_core(M=seq_len_engine0, K=self.vector_length, N=cur_mlp,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.FP4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off,
                    )
                if self.dual_engine:
                    self.generate_instruction_flag_check(target_engine_idx=1)
            else:
                # Slave: check -> clear -> workload -> set | check -> clear -> workload -> set |
                self.generate_instruction_flag_check(target_engine_idx=0)
                self.generate_instruction_flag_clear()
                total_flops += self.matmat_mul_core(M=seq_len_engine1, K=self.vector_length, N=cur_mlp,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM + seq_len_engine0 * self.vector_length * self.bytes_per_element,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM + seq_len_engine0 * cur_mlp * self.bytes_per_element,
                    is_B_quantized=True,
                    data_type=TYPE.FP4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off,
                    gelu_enable=True,
                    )
                total_flops += self.matmat_mul_core(M=seq_len_engine1, K=self.vector_length, N=cur_mlp,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM + seq_len_engine0 * self.vector_length * self.bytes_per_element,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM + seq_len_engine0 * cur_mlp * self.bytes_per_element,
                    is_B_quantized=True,
                    data_type=TYPE.FP4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off,
                    )
                self.generate_instruction_flag_set()
            if not self.engine_slave:
                self._emit_sram_eltwise_chunked(
                    "mul", self.LAYER0_MLP_GATE_DRAM, self.LAYER0_MLP_UP_DRAM,
                    self.LAYER0_MLP_MULT_DRAM,
                    seq_len * cur_mlp)
            if not self.engine_slave:
                # Master: set -> clear -> workload -> check | set -> clear -> workload -> check |
                self.generate_instruction_flag_set()
                self.generate_instruction_flag_clear()
                total_flops += self.matmat_mul_core(M=seq_len_engine0, K=cur_mlp, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.FP4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off,
                    )
                if self.dual_engine:
                    self.generate_instruction_flag_check(target_engine_idx=1)
            else:
                # Slave: check -> clear -> workload -> set | check -> clear -> workload -> set |
                self.generate_instruction_flag_check(target_engine_idx=0)
                self.generate_instruction_flag_clear()
                total_flops += self.matmat_mul_core(M=seq_len_engine1, K=cur_mlp, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM + seq_len_engine0 * cur_mlp * self.bytes_per_element,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM + seq_len_engine0 * self.vector_length * self.bytes_per_element,
                    is_B_quantized=True,
                    data_type=TYPE.FP4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off,
                    )
                self.generate_instruction_flag_set()
            if not self.engine_slave:
                total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                                OUTPUT_DRAM_ADDR=self.LAYER0_POST_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_FFW_NORM_GAMMA + layer_off)
                self._emit_sram_eltwise_chunked(
                    "add", self.LAYER0_POST_ATTN_RESIDUAL_DRAM, self.LAYER0_POST_MLP_NORM_DRAM,
                    self.LAYER0_OUTPUT_DRAM,
                    seq_len * self.vector_length)

                # Per-layer input injection (NEW for Gemma4 E2B)
                total_flops += self._compile_per_layer_injection(layer_idx, layer_off, seq_len)

        self.stop_capture()
        self.generate_instruction_halt()
        prefill_program_addr = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prefill_program_addr)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        _SILENT_MODE = False
        print(f"    Prefill program start at 0x{prefill_program_addr:X} end at 0x{self.get_program_dram_addr():X}, usage: {self.get_program_dram_usage()} bytes")

        return prefill_program_addr, total_flops

    def _compute_per_layer_inputs(self, token_ids, embedding_tensor: torch.Tensor) -> torch.Tensor:
        """Compute per-layer inputs on host side.
        Args:
            token_ids: token id sequence (list or tuple)
            embedding_tensor: (seq_len, hidden_size) bf16 tensor (already scaled)
        Returns:
            per_layer_inputs: (seq_len, LAYER_SIZE, per_layer_input_dim) bf16 tensor
        """
        seq_len = len(token_ids)
        tid_t = torch.tensor(token_ids, dtype=torch.long)

        # VLM mode: replace image token IDs with pad_token_id for per_layer_embed lookup
        if hasattr(self, '_mm_types') and self._mm_types is not None:
            mm_mask = torch.tensor(self._mm_types[:len(token_ids)])
            tid_t_for_pli = tid_t.clone()
            tid_t_for_pli[mm_mask == 1] = 0  # pad_token_id
        else:
            tid_t_for_pli = tid_t

        # per_layer_embed: lookup from embed_tokens_per_layer [262144, 8960] -> [seq_len, 8960] -> [seq_len, 35, 256]
        per_layer_embed = self.embed_tokens_per_layer_weight[tid_t_for_pli]  # [seq_len, 8960]
        per_layer_embed = per_layer_embed.reshape(seq_len, self.LAYER_SIZE, self.per_layer_input_dim)  # [seq_len, 35, 256]

        # per_layer_proj: (per_layer_model_projection @ embedding.T).T
        # per_layer_model_proj_weight is [8960, 1536], embedding_tensor is [seq_len, 1536]
        # We want [seq_len, 8960] = embedding_tensor @ per_layer_model_proj_weight.T
        # But need to undo the embedding scale first: use the unscaled embedding
        # Actually the spec says to use the embedding_tensor (already scaled), and apply proj_scale
        per_layer_proj = (embedding_tensor.float() @ self.per_layer_model_proj_weight.float().T)  # [seq_len, 8960]
        per_layer_proj = (per_layer_proj * self._per_layer_model_proj_scale).to(torch.bfloat16)
        per_layer_proj = per_layer_proj.reshape(seq_len, self.LAYER_SIZE, self.per_layer_input_dim)  # [seq_len, 35, 256]

        # rms_norm per_layer_proj along last dim with per_layer_proj_norm_weight
        per_layer_proj = _host_rms_norm(per_layer_proj, self.per_layer_proj_norm_weight)

        # per_layer_inputs = (per_layer_proj + per_layer_embed) * per_layer_input_scale
        per_layer_inputs = ((per_layer_proj.float() + per_layer_embed.float()) * self._per_layer_input_scale).to(torch.bfloat16)

        return per_layer_inputs  # [seq_len, 35, 256]

    def run_prefill(self, prefill_program_addr: int, prefill_seq=None, flops: int = None) -> dict:
        """
        Run prefill. Uses prefill_seq if provided, otherwise self.prefill_seq (set by set_prefill_seq()).

        Args:
            prefill_program_addr: The address of the prefill program in DRAM.
            prefill_seq: Optional sequence; if None, uses self.prefill_seq.
            flops: The number of FLOPS to use for the prefill.

        Returns:
            A tuple containing the latency and flop rate.
        """
        if prefill_seq is None:
            prefill_seq = self.prefill_seq
        if prefill_seq is None:
            prefill_seq = tuple(self._cfg["default_prefill_tokens"])
        # Prefill processes all but the last token
        if len(prefill_seq) > 1:
            prefill_seq = prefill_seq[:-1]
            assert len(prefill_seq) == self.seq_len, f"Expected seq_len {self.seq_len}, but got {len(prefill_seq)}"
        else:
            raise ValueError("Prefill sequence must have at least 2 tokens.")

        seq_len = len(prefill_seq)
        assert seq_len <= self.max_prefill_seq_len, (
            f"Prefill length {seq_len} exceeds max_prefill_seq_len {self.max_prefill_seq_len}. "
            f"Bump 'max_prefill_seq_len' in gemma4_e2b_config.json (and raise _tensor_estimate "
            f"in __init__ accordingly) to support longer prompts."
        )
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64

        embedding_tensor = self.get_embedding_for_tokens(prefill_seq)

        # Multimodal merge: replace image/audio placeholder embeddings with
        # encoder-produced soft-token features. Uses mm_token_type_ids where
        # 1=image, 3=audio (HF convention, see transformers/processing_utils.py
        # create_mm_token_type_ids).
        if hasattr(self, '_mm_types') and self._mm_types is not None:
            mm_types = torch.tensor(self._mm_types[:len(prefill_seq)])
            if hasattr(self, '_image_features') and self._image_features is not None:
                image_mask = (mm_types == 1)
                embedding_tensor[image_mask] = self._image_features[:image_mask.sum()].to(embedding_tensor.dtype)
                print(f"Merged {image_mask.sum().item()} image features into embeddings")
            if hasattr(self, '_audio_features') and self._audio_features is not None:
                audio_mask = (mm_types == 3)
                embedding_tensor[audio_mask] = self._audio_features[:audio_mask.sum()].to(embedding_tensor.dtype)
                print(f"Merged {audio_mask.sum().item()} audio features into embeddings")

        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)

        # Compute per-layer inputs on host and DMA to FPGA
        per_layer_inputs = self._compute_per_layer_inputs(prefill_seq, embedding_tensor)  # [seq_len, 35, 256]
        # Permute to [35, seq_len, 256] so each layer's data is contiguous in DRAM
        per_layer_inputs_flat = per_layer_inputs.permute(1, 0, 2).contiguous()  # [35, seq_len, 256]
        self.dma_to_accelerator_memory(self.PER_LAYER_INPUTS_DRAM, per_layer_inputs_flat)

        # Clear multimodal state now that prefill's per-layer-inputs have been computed.
        # Decode reuses _compute_per_layer_inputs with seq_len=1, and if _mm_types
        # is still set it would incorrectly treat every decode token as a
        # multimodal position (replacing its ID with pad_token_id=0 for
        # per_layer_embed lookup),
        # producing garbage per-layer injection and all-pad output. Mirror the
        # compare script's pattern: clear right after use.
        self._mm_types = None
        self._image_features = None
        self._audio_features = None

        # Build BOTH prefill bias matrices: full (causal) for full-attention
        # layers, and sliding (causal AND within `sliding_window` tokens) for
        # sliding-attention layers. compile_prefill picks per-layer.
        # Both biases are in q_seq_len space (each token has group_size query
        # heads, K is GQA-duplicated to match), so the window is converted
        # from token space to q-position space by multiplying by group_size.
        full_bias = torch.full((aligned_seq_len, aligned_seq_len), float("-inf"), dtype=torch.bfloat16)
        valid_mask = torch.tril(torch.ones(aligned_seq_len, aligned_seq_len, dtype=torch.bool), diagonal=0) if not self.causal_mask_upper else torch.triu(torch.ones(aligned_seq_len, aligned_seq_len, dtype=torch.bool), diagonal=0)
        full_bias.masked_fill_(valid_mask, 0.0)
        full_bias[:, q_seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_FULL_DRAM, full_bias)

        # Sliding bias is identical to full when seq_len ≤ sliding_window;
        # otherwise it additionally masks anything older than the window.
        if seq_len <= self.sliding_window:
            sliding_bias = full_bias
        else:
            window_q = self.sliding_window * self.group_size
            sliding_bias = torch.full((aligned_seq_len, aligned_seq_len), float("-inf"), dtype=torch.bfloat16)
            i_idx = torch.arange(aligned_seq_len).unsqueeze(1)
            j_idx = torch.arange(aligned_seq_len).unsqueeze(0)
            i_token = i_idx // self.group_size
            j_token = j_idx // self.group_size
            in_window = (i_token - j_token) < self.sliding_window
            sliding_mask = valid_mask & in_window
            sliding_bias.masked_fill_(sliding_mask, 0.0)
            sliding_bias[:, q_seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_SLIDING_DRAM, sliding_bias)
        latency, flop_rate_program = self.program_execute(prefill_program_addr, flops=flops)
        return latency, flop_rate_program

    def compile_decoder(self, layer_size: int = 35) -> tuple[str, list[int], list[int]]:
        """Compile decoder programs for seq_len buckets, or load from existing bin/meta.
        Returns (decoder_bin_path, program_sizes[8], total_flops_list[8])."""
        decoder_bin_path = os.path.join(self.script_dir, "gemma4_e2b_bin", "decoder_program.bin")
        decoder_meta_path = os.path.join(self.script_dir, "gemma4_e2b_bin", "decoder_program.json")
        if os.path.exists(decoder_bin_path) and os.path.exists(decoder_meta_path):
            with open(decoder_meta_path, "r") as f:
                meta = json.load(f)
            if "instruction_counts" in meta:
                program_sizes = [c * 32 for c in meta["instruction_counts"]]
            else:
                program_sizes = meta["program_sizes"]
            gflops_per_token = meta["total_flops"]
            print(f"Decoder bin found, skipped compile.")
            return decoder_bin_path, program_sizes, gflops_per_token

        print(f"Decoder bin not found, compiling...")
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        segment_instruction_counts = []
        total_flops_list = []

        global _SILENT_MODE
        _SILENT_MODE = True
        self.clear_inst_id()
        self.clear_capture_buffer()
        self.start_capture()
        for seq_len in self._cfg["model"]["decoder_seq_len_buckets"]:
            count_at_start = self.capture_count
            total_flops = 0
            for layer_idx in range(layer_size):
                layer_off = layer_idx * LAYER_WEIGHT_SIZE
                cur_head_dim, cur_q_size, cur_k_size = self._get_layer_attention_dims(layer_idx)
                cur_mlp = self._get_mlp_elements(layer_idx)
                rope_n = self._get_rope_dims(layer_idx)

                # Layer-input source:
                #   layer 0: LAYER0_INPUT_DRAM (uploaded by run_decoder each step)
                #   layer i>0: LAYER0_OUTPUT_DRAM (written by the previous layer's
                #     per_layer_injection). No copy needed — LAYER0_OUTPUT_DRAM is
                #     only overwritten at the end of the current layer (in the
                #     MLP residual add at line ~2955), which happens AFTER we
                #     consume it as the attention-residual source. So reading it
                #     here and for the attention residual below is safe.
                layer_input_addr = self.LAYER0_INPUT_DRAM if layer_idx == 0 else self.LAYER0_OUTPUT_DRAM
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=layer_input_addr,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off)
                # Q/K/V projections: use per-layer dims
                total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=cur_q_size,
                                                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                                                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off,
                                                    OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                                                    is_B_quantized=True,
                                                    data_type=TYPE.FP4,
                                                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off,
                                                    )
                if layer_idx in self._kv_shared_map:
                    ref_layer = self._kv_shared_map[layer_idx]
                    kv_layer_for_attn = ref_layer  # read from reference layer's KV cache
                else:
                    kv_layer_for_attn = layer_idx  # read from own KV cache
                    # K projection
                    total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=cur_k_size,
                        A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                        B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off,
                        OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                        is_B_quantized=True,
                        data_type=TYPE.FP4,
                        SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off,
                        )
                    # V projection
                    total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=cur_k_size,
                        A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                        B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off,
                        OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                        is_B_quantized=True,
                        data_type=TYPE.FP4,
                        SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off,
                        )
                    self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_FLASH_V_DRAM, sram_address=0x10000, element_size=cur_k_size)
                    # V norm (Gemma4: normalize V without learnable scale)
                    self.rms_norm_core(0x10000, 0x10000, cur_k_size)  # no gamma
                    self.generate_instruction_add_imm(self.V_CACHE_SIZE_REG, self.LAYER0_V_DRAM + self._kv_slot_for_layer[layer_idx] * self.MAX_CONTEXT_SIZE * self.k_size, self.TMP_REG)
                    self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=0, element_size=cur_k_size)
                    self.overwrite_instruction_with_general_register(self.TMP_REG)
                    # RMS norm on K
                    total_flops += self.rms_norm_core_dram(M=1, N=cur_head_dim, A_DRAM_ADDR=self.LAYER0_K_DRAM,
                                  OUTPUT_DRAM_ADDR=self.LAYER0_K_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_NORM_GAMMA + layer_off)

                # Q norm always runs
                total_flops += self.rms_norm_core_dram(M=self.group_size, N=cur_head_dim, A_DRAM_ADDR=self.LAYER0_Q_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_Q_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_NORM_GAMMA + layer_off)

                ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE_GLOBAL if layer_idx in self._rope_global_layers else self.DRAM_ADDR_ROPE_LOCAL

                # Select correct ROPE register: sliding uses ROPE_SIZE_REG, full uses ROPE_GLOBAL_SIZE_REG
                cur_rope_reg = self.ROPE_GLOBAL_SIZE_REG if layer_idx in self._full_attention_layers else self.ROPE_SIZE_REG

                if layer_idx not in self._kv_shared_map:
                    # RoPE on K: only rope_n dims (non-shared layers only)
                    total_flops += self.rope_hf_core(N=rope_n, input_dram_addr=self.LAYER0_K_NORM_DRAM, output_dram_addr=self.LAYER0_K_ROPE_DRAM + self._kv_slot_for_layer[layer_idx] * self.MAX_CONTEXT_SIZE * self.k_size,
                            cos_dram_addr=ROPE_WEIGHT_ADDR, sin_dram_addr=ROPE_WEIGHT_ADDR + rope_n * self.bytes_per_element,
                            rope_size_reg=cur_rope_reg, output_addr_inc_reg=self.V_CACHE_SIZE_REG, tmp_reg=self.TMP_REG)

                # RoPE on Q groups: always runs
                for g in range(self.group_size):
                    total_flops += self.rope_hf_core(N=rope_n, input_dram_addr=self.LAYER0_Q_NORM_DRAM + g * cur_head_dim * self.bytes_per_element, output_dram_addr=self.LAYER0_FLASH_Q_DRAM + g * cur_head_dim * self.bytes_per_element,
                            cos_dram_addr=ROPE_WEIGHT_ADDR, sin_dram_addr=ROPE_WEIGHT_ADDR + rope_n * self.bytes_per_element,
                            rope_size_reg=cur_rope_reg, tmp_reg=self.TMP_REG)

                # For full attention layers with partial rotation, copy non-rotated dims through
                if layer_idx in self._full_attention_layers and rope_n < cur_head_dim:
                    remaining = cur_head_dim - rope_n
                    # Copy non-rotated dims for Q
                    for g in range(self.group_size):
                        src = self.LAYER0_Q_NORM_DRAM + g * cur_head_dim * self.bytes_per_element + rope_n * self.bytes_per_element
                        dst = self.LAYER0_FLASH_Q_DRAM + g * cur_head_dim * self.bytes_per_element + rope_n * self.bytes_per_element
                        self.accelerator_memory_to_sram(src, 0x10000, remaining)
                        self.sram_to_accelerator_memory(0x10000, dst, remaining)
                    if layer_idx not in self._kv_shared_map:
                        # Copy non-rotated dims for K (decoder: single token, write to cache)
                        src = self.LAYER0_K_NORM_DRAM + rope_n * self.bytes_per_element
                        # Need to write to the K_ROPE cache at the current position
                        # The rotated part was already written by rope_hf_core with output_addr_inc_reg
                        # For the non-rotated part, we write to the same cache location + rope_n offset
                        dst_base = self.LAYER0_K_ROPE_DRAM + self._kv_slot_for_layer[layer_idx] * self.MAX_CONTEXT_SIZE * self.k_size + rope_n * self.bytes_per_element
                        self.accelerator_memory_to_sram(src, 0x10000, remaining)
                        self.generate_instruction_add_imm(self.V_CACHE_SIZE_REG, dst_base, self.TMP_REG)
                        self.sram_to_accelerator_memory(0x10000, dst_base, remaining)
                        self.overwrite_instruction_with_general_register(self.TMP_REG)

                # Q pre-scaling: scale Q by sqrt(head_dim) to cancel internal 1/sqrt(head_dim) in attention
                self.accelerator_memory_to_sram(self.LAYER0_FLASH_Q_DRAM, 0x30000, self.group_size * cur_head_dim)
                self.broadcast_mul(scalar=math.sqrt(cur_head_dim), sram_start_addr=0x30000, sram_wb_addr=0x30000, element_size=self.group_size * cur_head_dim)
                self.sram_to_accelerator_memory(0x30000, self.LAYER0_FLASH_Q_DRAM, self.group_size * cur_head_dim)

                # Gather K/V from KV cache (k_size stride) to contiguous flash buffers
                # decoder_attention_core expects contiguous [seq_len, head_dim] K/V
                kv_slot_off_read = self._kv_slot_for_layer[kv_layer_for_attn] * self.MAX_CONTEXT_SIZE * self.k_size
                kv_k_base = self.LAYER0_K_ROPE_DRAM + kv_slot_off_read
                kv_v_base = self.LAYER0_V_DRAM + kv_slot_off_read
                if cur_head_dim * self.bytes_per_element != self.k_size:
                    # Stride mismatch: gather per-token into contiguous buffer
                    for t in range(seq_len):
                        self.accelerator_memory_to_sram(kv_k_base + t * self.k_size, 0x10000, cur_head_dim)
                        self.sram_to_accelerator_memory(0x10000, self.LAYER0_FLASH_K_DRAM + t * cur_head_dim * self.bytes_per_element, cur_head_dim)
                        self.accelerator_memory_to_sram(kv_v_base + t * self.k_size, 0x20000, cur_head_dim)
                        self.sram_to_accelerator_memory(0x20000, self.LAYER0_FLASH_V_DRAM + t * cur_head_dim * self.bytes_per_element, cur_head_dim)
                    dec_k_addr = self.LAYER0_FLASH_K_DRAM
                    dec_v_addr = self.LAYER0_FLASH_V_DRAM
                else:
                    # No stride mismatch: read directly from KV cache
                    dec_k_addr = kv_k_base
                    dec_v_addr = kv_v_base

                # Per-layer bias: full attention layers see the entire causal
                # window, sliding-attention layers are limited to
                # `sliding_window` tokens. run_decoder uploads both biases
                # each step.
                bias_addr_layer = (self.LAYER0_FLASH_BIAS_FULL_DRAM
                                   if layer_idx in self._full_attention_layers
                                   else self.LAYER0_FLASH_BIAS_SLIDING_DRAM)
                # GQA head fusion: the group_size query heads in FLASH_Q all
                # share the same K, V, V^T (one KV head per group in Gemma4).
                # Calling decoder_attention_core once with q_rows=group_size
                # amortizes the V^T transpose, the K streaming, and the V^T
                # streaming across all heads (vs. running them group_size
                # times in a per-head loop).
                total_flops += self.decoder_attention_core(
                    head_dim=cur_head_dim,
                    seq_len=seq_len,
                    Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
                    K_DRAM_ADDR=dec_k_addr,
                    V_DRAM_ADDR=dec_v_addr,
                    OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                    IDENTITY_DRAM_ADDR=self.IDENTITY_DRAM_ADDR,
                    SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                    BIAS_DRAM_ADDR=bias_addr_layer,
                    q_rows=self.group_size,
                )
                # O projection: INT4, K=cur_q_size (actual per-layer attention output dim)
                total_flops += self.matmat_mul_core(M=1, K=cur_q_size, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.FP4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off,
                    )
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_POST_ATTN_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_NORM_GAMMA + layer_off)

                # Attention residual: use layer_input_addr (LAYER0_OUTPUT_DRAM
                # for layers > 0, LAYER0_INPUT_DRAM for layer 0) — same source
                # as the pre-norm above. This avoids the LAYER0_OUTPUT → LAYER0_INPUT
                # copy that used to run at the top of every layer.
                self.accelerator_memory_to_sram(accelerator_dram_address=layer_input_addr, sram_address=0x10000, element_size=self.vector_length)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_NORM_DRAM, sram_address=0x90000, element_size=self.vector_length)
                self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, element_size=self.vector_length)

                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off)

                total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=cur_mlp,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.FP4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off,
                    gelu_enable=True,
                    )
                total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=cur_mlp,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.FP4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off,
                    )

                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_GATE_DRAM, sram_address=0x10000, element_size=cur_mlp)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_UP_DRAM, sram_address=0x90000, element_size=cur_mlp)
                self.eltwise_mul_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=cur_mlp)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_MLP_MULT_DRAM, element_size=cur_mlp)

                total_flops += self.matmat_mul_core(M=1, K=cur_mlp, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.FP4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off,
                    )
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_POST_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_FFW_NORM_GAMMA + layer_off)

                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, sram_address=0x10000, element_size=self.vector_length)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_MLP_NORM_DRAM, sram_address=0x90000, element_size=self.vector_length)
                self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, element_size=self.vector_length)

                # Per-layer input injection (NEW for Gemma4 E2B) - decoder uses seq_len=1
                total_flops += self._compile_per_layer_injection(layer_idx, layer_off, 1)

            if layer_size == self.LAYER_SIZE:
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_OUTPUT_DRAM,
                    OUTPUT_DRAM_ADDR=self.OUTPUT_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_OUTPUT_NORM_GAMMA)
                total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=self.EMBEDDING_ELEMENTS,
                    A_DRAM_ADDR=self.OUTPUT_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_QUANT,
                    OUTPUT_DRAM_ADDR=self.LOGITS_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.FP4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_SCALE,
                    )

            self.generate_instruction_halt()
            segment_instruction_counts.append(self.capture_count - count_at_start)
            total_flops_list.append(total_flops)
        self.stop_capture()
        _SILENT_MODE = False
        all_programs_bytes = bytearray()
        for inst in self.capture_buffer:
            all_programs_bytes.extend(inst.get_bytes())
        os.makedirs(os.path.dirname(decoder_bin_path), exist_ok=True)
        with open(decoder_bin_path, "wb") as f:
            f.write(all_programs_bytes)
        program_sizes = [c * 32 for c in segment_instruction_counts]
        with open(decoder_meta_path, "w") as f:
            json.dump({"instruction_counts": segment_instruction_counts, "program_sizes": program_sizes, "total_flops": total_flops_list}, f, indent=0)
        self.clear_capture_buffer()
        print(f"Decoder programs: {len(segment_instruction_counts)} segments written to {decoder_bin_path} ({len(all_programs_bytes)} bytes)")
        return decoder_bin_path, program_sizes, total_flops_list

    def run_decoder(self, decoder_program_sizes: list[int], decoder_base_addr: int, token_id: int, flops_per_token: list[int] | None = None) -> dict:
        """Run decode loop. seq_len capped at MAX_CONTEXT_SIZE. Breaks on END_OF_TURN_TOKEN_ID."""
        if token_id is None:
            print("No last token available for decode.")
            return {}

        global _SILENT_MODE
        max_seq_len = self.MAX_CONTEXT_SIZE
        total_latency, total_flop_rate = 0, 0
        n_buckets = len(decoder_program_sizes)
        while self.seq_len < max_seq_len:
            _SILENT_MODE = True
            timer_start = time.perf_counter()
            self.seq_len += 1
            aligned_seq_len = ((self.seq_len + 63) // 64) * 64
            prog_idx = min((self.seq_len - 1) // 64, n_buckets - 1)
            prog_addr = decoder_base_addr + sum(decoder_program_sizes[:prog_idx])
            flops_per_token_idx = flops_per_token[prog_idx] if flops_per_token else None

            # Set V_CACHE_SIZE, ROPE_SIZE (sliding), ROPE_GLOBAL_SIZE (full attn) in
            # one program submission instead of three sequential ones, saving
            # ~10–15 ms of round-trip overhead per decoded token.
            decode_pos = self.seq_len - 1
            rope_local_row = self.head_dim_sliding * 2 * self.bytes_per_element  # 256 * 2 * 2 = 1024
            rope_global_row = int(self.head_dim * self._cfg["special"]["rope"]["partial_rotary_factor_global"]) * 2 * self.bytes_per_element  # 128 * 2 * 2 = 512
            self.isa_add_set_multi([
                (self.V_CACHE_SIZE_REG, decode_pos * self.k_size),
                (self.ROPE_SIZE_REG, decode_pos * rope_local_row),
                (self.ROPE_GLOBAL_SIZE_REG, decode_pos * rope_global_row),
            ])
            embedding_tensor = self.get_embedding_for_tokens([token_id])
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)

            # Compute per-layer inputs for single token and DMA
            per_layer_inputs = self._compute_per_layer_inputs([token_id], embedding_tensor)  # [1, 35, 256]
            # Permute to [35, 1, 256] so each layer's data is contiguous in DRAM
            self.dma_to_accelerator_memory(self.PER_LAYER_INPUTS_DRAM, per_layer_inputs.permute(1, 0, 2).contiguous())

            # Build BOTH decode bias rows. Full attention layers see all
            # positions [0, seq_len); sliding-attention layers see only the
            # last `sliding_window` tokens [max(0, seq_len-window), seq_len).
            full_bias_row = torch.full((1, aligned_seq_len), -1e36, dtype=torch.bfloat16)
            full_bias_row[0, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_FULL_DRAM, full_bias_row)
            if self.seq_len <= self.sliding_window:
                sliding_bias_row = full_bias_row
            else:
                sliding_bias_row = torch.full((1, aligned_seq_len), -1e36, dtype=torch.bfloat16)
                window_start = self.seq_len - self.sliding_window
                sliding_bias_row[0, window_start:self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_SLIDING_DRAM, sliding_bias_row)

            latency, flop_rate_program = self.program_execute(prog_addr, flops=flops_per_token_idx)
            total_latency += latency
            total_flop_rate += flop_rate_program
            token_id = self.get_arg_max_index()
            token_char = self.tokenizer.decode([token_id])
            _SILENT_MODE = False

            if token_id in [1, self._end_of_turn_token_id]:
                print(f"\nStop token {token_id} reached.")
                break
            print(token_char, end="", flush=True)
        return self.seq_len, total_latency, total_flop_rate

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Gemma4 E2B layer-0 prefill: run on accelerator, verify with torch ref.")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt. Default is a built-in prompt per mode "
                             "(LM: a test question; VLM: 'Describe this image in detail.'; "
                             "Audio: 'Describe what you hear.').")
    # Modality selection. Two ways to enable VLM or audio:
    #   (a) Pass a path via --image / --audio — that path is used AND the
    #       matching encoder is enabled automatically.
    #   (b) Pass --vision-enable / --audio-enable (no path) — the template
    #       uses a default example file shipped at ../../test_samples/
    #       (relative to this script) so the CLI works out-of-the-box.
    # Passing both modalities at once is rejected; LM-only is the default
    # when neither is selected.
    parser.add_argument("--image", type=str, default=None,
                        help="Path to image file for VLM inference. Implies --vision-enable.")
    parser.add_argument("--vision-enable", action="store_true",
                        help="Enable VLM mode with the default example image "
                             "(../../test_samples/yosemite.jpg, relative to this script). "
                             "Ignored if --image is also given.")
    parser.add_argument("--audio", type=str, default=None,
                        help="Path to audio file (.wav, .flac, etc.). Implies --audio-enable.")
    parser.add_argument("--audio-enable", action="store_true",
                        help="Enable audio mode with the default example audio "
                             "(../../test_samples/apex.wav, relative to this script). "
                             "Ignored if --audio is also given.")
    parser.add_argument("--local-weights", action="store_true", help="Use gemma4_e2b_bin/full_model_weights.bin instead of generated weights_gemma4_e2b_hf.bin")
    parser.add_argument('--dev', type=str, default='xdma0',
                        help='DMA device name (e.g., xdma0, xdma1). Default: xdma0')
    parser.add_argument('--cycle', type=float, default=5.62,
                        help='Clock cycle time in nanoseconds (default: 3.0, use 2.5 for alveo)')
    args = parser.parse_args()

    set_dma_device(args.dev)
    global DMA_DEVICE_H2C, DMA_DEVICE_C2H, DMA_DEVICE_USER
    DMA_DEVICE_H2C = user_dma_core.DMA_DEVICE_H2C
    DMA_DEVICE_C2H = user_dma_core.DMA_DEVICE_C2H
    DMA_DEVICE_USER = user_dma_core.DMA_DEVICE_USER
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle
    print(f"Using DMA device: {args.dev}")
    print(f"  H2C: {DMA_DEVICE_H2C}")
    print(f"  C2H: {DMA_DEVICE_C2H}")
    print(f"  USER: {DMA_DEVICE_USER}")
    print(f"Setting CLOCK_CYCLE_TIME_NS = {user_dma_core.CLOCK_CYCLE_TIME_NS}")

    # --- Resolve modality and input path -----------------------------------
    # The script has three modes: LM (text only), VLM (image + text),
    # and audio (audio + text). Exactly one encoder modality can be active
    # per run. Enabling rules:
    #   * --image PATH             → VLM with that image
    #   * --vision-enable (no path) → VLM with the shipped default image
    #   * --audio PATH             → audio with that file
    #   * --audio-enable (no path) → audio with the shipped default file
    #   * none of the above        → pure LM
    # Default example inputs live in a shared test_samples directory at the
    # template level (two levels up from this script). Multiple model folders
    # can share the same example files.
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _TEST_SAMPLES = os.path.normpath(os.path.join(_HERE, "..", "..", "test_samples"))
    DEFAULT_IMAGE = os.path.join(_TEST_SAMPLES, "yosemite.jpg")
    DEFAULT_AUDIO = os.path.join(_TEST_SAMPLES, "apex.wav")

    vision_on = bool(args.image) or args.vision_enable
    audio_on = bool(args.audio) or args.audio_enable
    if vision_on and audio_on:
        raise SystemExit(
            "Only one encoder modality per run. Choose either --image / --vision-enable "
            "OR --audio / --audio-enable, not both.")

    image_path = args.image or (DEFAULT_IMAGE if args.vision_enable else None)
    audio_path = args.audio or (DEFAULT_AUDIO if args.audio_enable else None)
    if vision_on and image_path and not os.path.exists(image_path):
        raise SystemExit(f"Image file not found: {image_path}")
    if audio_on and audio_path and not os.path.exists(audio_path):
        raise SystemExit(f"Audio file not found: {audio_path}")

    ue = Gemma4_UnifiedEngine(local_weights=args.local_weights)
    if vision_on:
        print(f"[Mode] VLM — image: {image_path}")
        ue.set_prefill_seq_vlm(image_path, prompt=args.prompt)
    elif audio_on:
        print(f"[Mode] Audio — audio: {audio_path}")
        ue.set_prefill_seq_audio(audio_path, prompt=args.prompt)
    elif args.prompt:
        print(f"[Mode] LM — prompt: {args.prompt!r}")
        ue.set_prefill_seq(args.prompt)
    else:
        print(f"[Mode] LM — default prompt")
        ue.set_prefill_seq()

    print(f"\n--- Compiling ---")
    timer = time.perf_counter()
    prefill_program_addr, flops_prefill = ue.compile_prefill(seq_len=len(ue.prefill_seq))
    print(f"Prefill compile done in {time.perf_counter() - timer:.2f} seconds")
    timer_dec = time.perf_counter()
    decoder_bin_path, decoder_program_sizes, flops_per_token = ue.compile_decoder()
    print(f"Decoder compile done in {time.perf_counter() - timer_dec:.2f} seconds.")
    # NOTE: the decoder bin is loaded into DRAM AFTER run_prefill completes,
    # reusing the prefill bin's DRAM region. For long prompts (e.g. VLM with
    # ~280 tokens) the prefill bin is ~180 MB and decoder bin is ~66 MB; holding
    # both at once would overflow the 4 GB DRAM boundary. Since the prefill bin
    # is dead code once run_prefill finishes, overwriting it with the decoder
    # bin keeps peak program-DRAM usage at max(prefill, decoder) instead of
    # prefill + decoder.

    print(f"\n--- Starting prefill ---")
    print(f"Prompt ({len(ue.prefill_seq)}) tokens: {ue.prefill_seq}")
    timer=time.perf_counter()
    latency_hw_prefill, flop_rate_hw_prefill = ue.run_prefill(prefill_program_addr, flops=flops_prefill)
    latency_prefill = time.perf_counter() - timer
    print(f"Prefill execute done in {latency_prefill:.2f} seconds, start decoding...\n")

    # Reuse the prefill bin's DRAM region for the decoder bin (see note above).
    ue._next_program_dram_addr = prefill_program_addr
    decoder_base_addr, _ = ue.load_instructions(decoder_bin_path)

    print(f"\n--- Starting decoder ---")
    timer=time.perf_counter()
    token_cnt_decoded, latency_hw_decoder, flop_rate_hw_decoder = ue.run_decoder(decoder_program_sizes, decoder_base_addr, token_id=ue.prefill_seq[-1], flops_per_token=flops_per_token)
    latency_decoder = time.perf_counter() - timer
    print(f"\nDecoder done in {latency_prefill + latency_decoder:.2f} seconds, speed: {(token_cnt_decoded - len(ue.prefill_seq) + 1) / latency_decoder:.2f} tokens/s, total {token_cnt_decoded} tokens.")
    print(f"HW counter: Latency: {(latency_hw_prefill + latency_hw_decoder) / 1e6:.2f} seconds, decoder average Gflops: {flop_rate_hw_decoder / (token_cnt_decoded - len(ue.prefill_seq) + 1):.2f} Gflops")
    print("Gemma4 E2B test ends.")

if __name__ == "__main__":
    main()
