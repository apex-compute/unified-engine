#!/usr/bin/env python3
"""
Llama-3.2-1B inference on accelerator: prefill + decode.

  - Config from llama3.2_1b_config.json; weights from a single bin (see below).
  - Prefill: compiled each run. Decoder: if llama3.2_1b_bin/decoder_program.bin and
    llama3.2_1b_bin/decoder_program.json exist, skip decoder compile and load
    program sizes from meta; otherwise compile and write the bin + meta.
  - Run prefill then decode loop.

Architecture differences vs Gemma3:
  - No per-head Q/K normalization (q_norm, k_norm absent).
  - No post-attention normalization (Gemma3 only).
  - No post-FFN normalization (Gemma3 only).
  - layer.post_attention_layernorm is the pre-FFN norm (not post-attn).
  - Embedding is NOT scaled by sqrt(hidden_size).
  - LM head weight is tied to the embedding weight.
  - gamma_offset = 0.0 (LLaMA uses w directly, not 1+w).

Weights:
  - Default: llama3.2_1b_bin/weights_llama3.2_1b_hf.bin (generated from HF model if missing).
  - --local-weights: use llama3.2_1b_bin/full_model_weights.bin instead.

Usage:
  python llama3.2_1b_test.py
  python llama3.2_1b_test.py --prompt "your prompt"
  python llama3.2_1b_test.py --dev xdma0 [--cycle 5.88]
  python llama3.2_1b_test.py --local-weights
"""

import json
import math
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import time

# This file's folder; user_dma_core.py is two folders up (repo root); that directory is added to sys.path.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

import user_dma_core
from user_dma_core import DMA_DEVICE_H2C, TYPE, UE_FMAX_CONTEXT_SIZE, UE_VECTOR_SIZE, URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS, set_dma_device, ue_35bit_addr_shifter
from user_dma_core import UnifiedEngine

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

def _quantize_bf16_to_int4_packed(weight_bf16: torch.Tensor, block_size: int = 64) -> tuple[bytes, bytes]:
    """Quantize bf16 weight (N_w, K_w) to INT4 packed + scale per block of 64 along K. Returns (data_bytes, scale_bytes)."""
    w = weight_bf16.detach().cpu().float().reshape(-1)
    N_w, K_w = weight_bf16.shape
    assert K_w % block_size == 0
    w_blocks = w.reshape(N_w, K_w // block_size, block_size)
    scale = w_blocks.abs().amax(dim=-1).clamp(min=1e-8) / 7.0
    # IF4 dispatches INT4 vs FP4 by bf16 scale sign (negative=INT4 codebook).
    scale_bf16 = (-scale).to(torch.bfloat16)
    w_int8 = (w_blocks / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8)
    w_nibbles = w_int8.numpy().astype(np.int16) & 0x0F
    low = w_nibbles[:, :, 0::2].reshape(N_w, -1)
    high = w_nibbles[:, :, 1::2].reshape(N_w, -1)
    packed = (high << 4) | low
    data_bytes = packed.astype(np.uint8).tobytes()
    scale_bytes = scale_bf16.contiguous().view(torch.uint8).numpy().tobytes()
    return (data_bytes, scale_bytes)

def _rope_kv_perm(num_kv_heads: int, actual_head_dim: int) -> torch.Tensor:
    """Return the 1-D index permutation that reorders a combined KV-head vector from
    standard layout  [h0[lo,hi], h1[lo,hi], ..., h_{N-1}[lo,hi]]
    to lo|hi layout  [h0[lo], ..., h_{N-1}[lo], h0[hi], ..., h_{N-1}[hi]].

    When k_proj / q_proj weight rows are permuted by this index before packing, the
    rope_hf_core (i, i+D/2) pairing maps exactly to per-head (j, j+head_dim/2) pairing
    within each 64-dim head rather than crossing head boundaries.

    The same permutation also generates the inverse by sorting:
        inv = torch.argsort(perm)
    """
    half = actual_head_dim // 2          # e.g. 32
    D    = num_kv_heads * actual_head_dim # e.g. 512
    perm = torch.empty(D, dtype=torch.long)
    for h in range(num_kv_heads):
        for j in range(half):
            perm[h * half + j]                   = h * actual_head_dim + j         # lo half
            perm[num_kv_heads * half + h * half + j] = h * actual_head_dim + half + j  # hi half
    return perm


def weight_bin_generate(script_dir: str | None = None, output_path: str | None = None) -> str:
    """Generate weights_llama3.2_1b_hf.bin from HuggingFace model per llama3.2_1b_config.json layout.
    Returns the path to the written file."""
    script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
    cfg = _load_config(script_dir)
    weight_defs = cfg["_weight_defs"]
    paths = cfg["paths"]
    paths_full = os.path.join(script_dir, paths["weights_bin"])
    out_path = output_path or paths_full

    model, model_dir = _ensure_hf_model(script_dir, cfg)
    gamma_offset = cfg["special"]["rms_norm"]["gamma_offset"]  # 0.0 for LLaMA
    emb_cfg = cfg["special"]["embedding"]
    token_embd_offset = _parse_offset(emb_cfg["token_embd_offset"])
    token_embd_size = _parse_offset(emb_cfg["token_embd_size"])
    LAYER_WEIGHT_SIZE = weight_defs["LAYER_WEIGHT_SIZE"]
    base_layer0 = weight_defs["BLK0_ATTN_NORM_WEIGHT"]
    num_layers = cfg["file_info"]["num_layers"]
    head_dim = cfg["file_info"]["head_dim"]
    vector_length = cfg["file_info"]["hidden_size"]
    group_size = cfg["file_info"]["group_size"]
    blk0_structure = cfg["layers"]["structure"]

    # [lo|hi] row permutation for k_proj/q_proj weights.
    # After permutation, rope_hf_core(N=head_dim=512) on the matmul output correctly
    # applies per-head RoPE using the 8-head tiled table (satisfies N>=128 constraint).
    head_dim_actual = 64
    num_kv_heads = head_dim // head_dim_actual  # 8
    kv_perm = _rope_kv_perm(num_kv_heads, head_dim_actual)   # size head_dim=512
    q_groups = group_size  # 4 groups of 512-dim (each group covers 2 KV heads' Q)
    q_perm = torch.cat([kv_perm + g * head_dim for g in range(q_groups)])  # size 2048

    # Compute total file size
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

    # Embedding: LLaMA does NOT scale by sqrt(hidden_size)
    embed = model.get_input_embeddings().weight.detach().cpu().to(torch.bfloat16)
    raw_emb = embed.contiguous().view(torch.uint8).numpy().tobytes()
    write_at(token_embd_offset, raw_emb)

    # Layers
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn

        # LLaMA norms: gamma_offset = 0.0 (weight stored as-is)
        gamma_in = (layer.input_layernorm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
        # [lo|hi] permutation on k_proj and q_proj rows so that rope_hf_core(N=512)
        # applies per-head RoPE correctly via the 8-head tiled frequency table.
        # After matmul, K output is [K0_lo..K7_lo, K0_hi..K7_hi] and Q is split
        # into 4 groups of 512 (covering 2 KV heads each), each in [lo|hi] layout.
        q_w = attn.q_proj.weight.detach().cpu().to(torch.bfloat16)[q_perm, :]
        k_w = attn.k_proj.weight.detach().cpu().to(torch.bfloat16)[kv_perm, :]
        v_w = attn.v_proj.weight.detach().cpu().to(torch.bfloat16)
        # LLaMA has no q_norm / k_norm: write zero placeholders (norm steps are skipped in pipeline)
        gamma_q = torch.zeros(head_dim, dtype=torch.bfloat16)
        gamma_k = torch.zeros(head_dim, dtype=torch.bfloat16)
        o_w = attn.o_proj.weight.detach().cpu().to(torch.bfloat16)
        # LLaMA has no post-attention norm: write zero placeholder (step skipped in pipeline)
        gamma_post = torch.zeros(vector_length, dtype=torch.bfloat16)
        # LLaMA's post_attention_layernorm IS the pre-FFN norm
        gamma_ffn = (layer.post_attention_layernorm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
        gate_w = layer.mlp.gate_proj.weight.detach().cpu().to(torch.bfloat16)
        up_w = layer.mlp.up_proj.weight.detach().cpu().to(torch.bfloat16)
        down_w = layer.mlp.down_proj.weight.detach().cpu().to(torch.bfloat16)
        # LLaMA has no post-FFN norm: write zero placeholder (step skipped in pipeline)
        gamma_post_ffn = torch.zeros(vector_length, dtype=torch.bfloat16)

        region_writes = [
            (gamma_in, "bf16"),
            (q_w, "int4"),
            (k_w, "int4"),
            (v_w, "int4"),
            (gamma_q, "bf16"),
            (gamma_k, "bf16"),
            (o_w, "int4"),
            (gamma_post, "bf16"),
            (gamma_ffn, "bf16"),
            (up_w, "int4"),
            (gate_w, "int4"),
            (down_w, "int4"),
            (gamma_post_ffn, "bf16"),
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
                data_bytes, scale_bytes = _quantize_bf16_to_int4_packed(tensor)
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

    # ROPE: LLaMA uses a single RoPE table (rope_global_layers is empty, all layers use local/default)
    rope_cfg = cfg["special"]["rope"]
    theta = rope_cfg["theta"]
    local_base = rope_cfg["local_base"]
    num_positions = rope_cfg["num_positions"]
    # Compute tiled RoPE for LLaMA: repeat per-head 32-freq pattern across all KV heads
    # head_dim (512) = num_kv_heads (8) x head_dim_actual (64); D_per_head = 32
    head_dim_actual = 64
    num_kv_heads = head_dim // head_dim_actual  # = 8
    D_per_head = head_dim_actual // 2           # = 32
    for name, theta_val, off_key, sz_key in [
        ("ROPE_LOCAL", local_base, "ROPE_LOCAL", "ROPE_LOCAL_SIZE"),
        ("ROPE_GLOBAL", theta, "ROPE_GLOBAL", "ROPE_GLOBAL_SIZE"),
    ]:
        inv_freq = 1.0 / (theta_val ** (torch.arange(D_per_head, dtype=torch.float32) / D_per_head))
        pos = torch.arange(num_positions, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)                     # (num_positions, 32)
        cos_head = freqs.cos().to(torch.bfloat16)              # (num_positions, 32)
        sin_head = freqs.sin().to(torch.bfloat16)              # (num_positions, 32)
        # Tile across num_kv_heads to get (num_positions, head_dim/2)
        cos_full = cos_head.repeat(1, num_kv_heads)            # (num_positions, 256)
        sin_full = sin_head.repeat(1, num_kv_heads)            # (num_positions, 256)
        # Layout expected by rope_hf_core: [cos_full, cos_full, -sin_full, sin_full]
        rope_tensor = torch.cat([cos_full, cos_full, -sin_full, sin_full], dim=1)
        sz = weight_defs[sz_key]
        raw = (rope_tensor.contiguous().view(torch.uint8).numpy().tobytes() + b"\x00" * sz)[:sz]
        write_at(weight_defs[off_key], raw)

    # OUTPUT_NORM
    out_norm = (model.model.norm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
    sz = weight_defs["OUTPUT_NORM_WEIGHT_SIZE"]
    raw = (out_norm.contiguous().view(torch.uint8).numpy().tobytes() + b"\x00" * sz)[:sz]
    write_at(weight_defs["OUTPUT_NORM_WEIGHT"], raw)

    # LM_HEAD: LLaMA ties lm_head weight to input embedding
    lm_head_w = model.get_input_embeddings().weight.detach().cpu().to(torch.bfloat16)
    scale_sz = weight_defs["LM_HEAD_WEIGHT_SCALE_SIZE"]
    data_sz = weight_defs["LM_HEAD_WEIGHT_DATA_SIZE"]
    data_bytes, scale_bytes = _quantize_bf16_to_int4_packed(lm_head_w)
    scale_padded = (scale_bytes + b"\x00" * scale_sz)[:scale_sz]
    data_padded = (data_bytes + b"\x00" * data_sz)[:data_sz]
    write_at(weight_defs["LM_HEAD_WEIGHT_SCALE"], scale_padded)
    write_at(weight_defs["LM_HEAD_WEIGHT_DATA"], data_padded)

    with open(out_path, "wb") as f:
        f.write(buf)
    print(f"Generated weights bin: {out_path} ({len(buf)} bytes)")
    return out_path

def _ensure_hf_model(script_dir: str, cfg: dict):
    """Ensure HF model is downloaded and loaded. Returns (model, model_dir)."""
    model_dir = os.path.join(script_dir, cfg["paths"]["hf_model_dir"])
    hf_repo = cfg["paths"]["hf_model_repo"]
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        _original_print(f"Downloading HF model {hf_repo} to {os.path.abspath(model_dir)} ...")
        snapshot_download(repo_id=hf_repo, local_dir=model_dir, local_dir_use_symlinks=False)
        _original_print("Download complete.")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map=None, trust_remote_code=True
    )
    return model, model_dir

def _load_config(script_dir: str) -> dict:
    """Load llama3.2_1b_config.json and build weight_defs (offset/size dict) from regions."""
    config_path = os.path.join(script_dir, "llama3.2_1b_config.json")
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

# -----------------------------------------------------------------------------
# Llama-3.2-1B unified engine
# -----------------------------------------------------------------------------
class Llama32_1b_UnifiedEngine(UnifiedEngine):
    """UnifiedEngine for Llama-3.2-1B: loads config + weight bin, compile_prefill/compile_decoder, run_prefill/run_decoder.

    Key architectural differences from Gemma3:
      - No Q/K per-head norm (q_norm, k_norm): compile pipeline skips those RMS norm steps.
      - No post-attention norm: residual is applied directly to o_proj output.
      - No post-FFN norm: residual is applied directly to down_proj output.
      - post_attention_layernorm in HF is the pre-FFN norm.
      - Embedding NOT scaled by sqrt(hidden_size).
      - LM head weight tied to input embedding.
    """

    def __init__(self, script_dir: str | None = None, hf_model_dir: str | None = None, weights_bin: str | None = None):
        super().__init__(program_dram_base=0xC0000000)
        self.script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
        self._cfg = _load_config(self.script_dir)
        self.weight_defs = self._cfg["_weight_defs"]

        fi = self._cfg["file_info"]
        model = self._cfg["model"]
        paths = self._cfg["paths"]

        self.vector_length = fi["hidden_size"]
        self.head_dim = fi["head_dim"]
        self.bytes_per_element = fi["bytes_per_element"]
        self.group_size = fi["group_size"]
        self.mlp_elements = fi["mlp_elements"]
        self.hf_model_dir = hf_model_dir or os.path.join(self.script_dir, paths["hf_model_dir"])
        self.q_size = self.head_dim * self.group_size * self.bytes_per_element
        self.k_size = self.head_dim * self.bytes_per_element
        # LLaMA 3.2 1B GQA: 8 KV heads × 64-dim per head = head_dim=512 combined
        self.actual_head_dim = 64
        self.num_kv_heads = self.head_dim // self.actual_head_dim  # = 8
        self.MAX_CONTEXT_SIZE = model["max_context_size"]
        self.LAYER_SIZE = fi["num_layers"]
        self.EMBEDDING_ELEMENTS = fi["embedding_vocab"]
        fixed = self._cfg.get("fixed_isa_regs", {})
        self.V_CACHE_SIZE_REG = fixed["V_CACHE_SIZE_REG"]
        self.TMP_REG = fixed["TMP_REG"]
        self.ROPE_SIZE_REG = fixed["ROPE_SIZE_REG"]
        self.gf_bucket_idx = fixed["GF_BUCKET_IDX_REG"]
        self._isa_reg_counter = max(fixed.values()) + 1  # must start past all fixed ISA regs
        self.causal_mask_upper = False
        self._rope_global_layers = set(model["rope_global_layers"])
        self._end_of_turn_token_id = model["end_of_turn_token_id"]
        self._gamma_bin_offset = self._cfg["special"]["rms_norm"]["gamma_offset"]

        # LLaMA architecture flags: these norms do not exist in LLaMA 3.2
        self._has_q_k_norm = False       # no per-head Q/K normalization
        self._has_post_attn_norm = False  # no post-attention normalization
        self._has_post_mlp_norm = False   # no post-FFN normalization

        bin_path = weights_bin or paths["weights_bin"]
        full_path = os.path.join(self.script_dir, bin_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Bin file not found: {full_path}")
        with open(full_path, "rb") as f:
            self.weight_bin = f.read()
        self.weight_init()
        self.tensor_init()

    def get_embedding_for_tokens(self, token_ids: list[int] | tuple) -> torch.Tensor:
        """Return (len(token_ids), vector_length) bfloat16 tensor from self.embedding_weight (HF, scale applied)."""
        tid_t = torch.tensor(token_ids, dtype=torch.long)
        out = torch.zeros(len(token_ids), self.vector_length, dtype=torch.bfloat16)
        valid = tid_t < self.embedding_weight.shape[0]
        out[valid] = self.embedding_weight[tid_t[valid]]
        return out

    def _load_rope_host(self, rope_theta: float | None = None, rope_local_base: float | None = None) -> None:
        """Generate N=512 tiled RoPE table (8 KV heads tiled) and write to DRAM.

        Used with [lo|hi]-permuted k/q_proj weights so rope_hf_core(N=512) correctly
        applies per-head RoPE. The 32-element per-head frequencies are tiled 8× to cover
        the full 256-element half of the 512-dim combined vector.

        Layout per position (1024 elements × 2 bytes = 2048 bytes):
            [cos_full(256), cos_full(256), -sin_full(256), sin_full(256)]
        rope_hf_core(N=512) reads cos at t*2048 and sin at t*2048+1024.

        This satisfies the N>=128 hardware alignment constraint (N=512 >> 128).
        """
        rope_cfg = self._cfg["special"]["rope"]
        theta = rope_theta if rope_theta is not None else rope_cfg["theta"]
        local_base = rope_local_base if rope_local_base is not None else rope_cfg["local_base"]
        num_rope_positions = rope_cfg["num_positions"]
        D_per_head = self.actual_head_dim // 2  # = 32 frequencies per KV head
        for name, theta_val, sz_key, attr in [
            ("ROPE_LOCAL", local_base, "ROPE_LOCAL_SIZE", "DRAM_ADDR_ROPE_LOCAL"),
            ("ROPE_GLOBAL", theta, "ROPE_GLOBAL_SIZE", "DRAM_ADDR_ROPE_GLOBAL"),
        ]:
            inv_freq = 1.0 / (theta_val ** (torch.arange(D_per_head, dtype=torch.float32) / D_per_head))
            pos = torch.arange(num_rope_positions, dtype=torch.float32)
            freqs = torch.outer(pos, inv_freq)
            cos_head = freqs.cos().to(torch.bfloat16)  # (num_pos, 32)
            sin_head = freqs.sin().to(torch.bfloat16)  # (num_pos, 32)
            # Tile 8× → (num_pos, 256): each of 8 KV heads uses same per-head frequencies
            cos_full = cos_head.repeat(1, self.num_kv_heads)   # (num_pos, 256)
            sin_full = sin_head.repeat(1, self.num_kv_heads)   # (num_pos, 256)
            # Layout for rope_hf_core(N=512): [cos_full, cos_full, -sin_full, sin_full]
            rope_tensor = torch.cat([cos_full, cos_full, -sin_full, sin_full], dim=1)  # (num_pos, 1024)
            sz = self.weight_defs[sz_key]
            raw = rope_tensor.contiguous().view(torch.uint8).numpy().tobytes()
            raw = (raw + b"\x00" * sz)[:sz]
            addr = self.allocate_params_dram(sz)
            self.dma_write(DMA_DEVICE_H2C, addr, raw, sz)
            setattr(self, attr, addr)

    def weight_init(self) -> None:
        """Initialize DRAM: load HF embedding+tokenizer, layer weights from bin, host-computed RoPE, OUTPUT_NORM/LM_HEAD from bin."""
        model, model_dir = _ensure_hf_model(self.script_dir, self._cfg)
        # LLaMA does NOT scale the embedding by sqrt(hidden_size)
        embed = model.get_input_embeddings().weight.detach().cpu().to(torch.bfloat16)
        self.embedding_weight = embed
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

        layer0_end = (self.weight_defs["BLK0_POST_FFW_NORM_WEIGHT"] - base_layer0
                      + self.weight_defs["BLK0_POST_FFW_NORM_WEIGHT_SIZE"])
        assert layer0_end == LAYER_WEIGHT_SIZE, (
            f"Layer 0 size mismatch: computed {layer0_end} != LAYER_WEIGHT_SIZE {LAYER_WEIGHT_SIZE}"
        )

        print(f"\n--- Weights DRAM allocation, start at DRAM address: {self.get_params_dram_addr()} ---")
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
        """Initialize hardware DRAM tensors for Llama-3.2-1B (layer-wise overlap except for kv cache)."""
        seq_len = self.MAX_CONTEXT_SIZE
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64

        print(f"Allocate tensor dram start at DRAM address: 0x{self.get_tensor_dram_addr():X}")
        # Allocate shared memory for k v cache (k rope and v projection) and zero pad for decoder use:
        self.LAYER0_V_DRAM = self.allocate_tensor_dram(self.LAYER_SIZE * self.MAX_CONTEXT_SIZE * self.k_size)
        self.LAYER0_K_ROPE_DRAM = self.allocate_tensor_dram(self.LAYER_SIZE * self.MAX_CONTEXT_SIZE * self.k_size)
        zero_pad = torch.zeros(self.LAYER_SIZE * self.MAX_CONTEXT_SIZE * self.k_size, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_K_ROPE_DRAM, zero_pad)
        # Allocate memory for constant zero tensor, identity matrix, and bias:
        zero_add = torch.zeros(seq_len * self.head_dim * self.bytes_per_element, dtype=torch.bfloat16)
        self.ZERO_DRAM_ADDR = self.allocate_tensor_dram(seq_len * self.head_dim * self.bytes_per_element)
        self.dma_to_accelerator_memory(self.ZERO_DRAM_ADDR, zero_add)
        self.IDENTITY_DRAM_ADDR = self.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * self.bytes_per_element)
        self.dma_to_accelerator_memory(self.IDENTITY_DRAM_ADDR, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))
        # Allocate memory for flash attention and zero pad:
        self.LAYER0_FLASH_Q_DRAM = self.allocate_tensor_dram(aligned_seq_len * self.head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_K_DRAM = self.allocate_tensor_dram(aligned_seq_len * self.head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_V_DRAM = self.allocate_tensor_dram(aligned_seq_len * self.head_dim * self.bytes_per_element)
        zero_pad = torch.zeros(aligned_seq_len * self.head_dim * self.bytes_per_element, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_Q_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_K_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_V_DRAM, zero_pad)
        # Allocate memory for layer intermediate tensors:
        self.LAYER0_INPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_Q_DRAM = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_K_DRAM = self.allocate_tensor_dram(seq_len * self.k_size)
        # K_NORM and Q_NORM aliases: for LLaMA these point to K_DRAM/Q_DRAM (no norm applied)
        self.LAYER0_K_NORM_DRAM = self.LAYER0_K_DRAM
        self.LAYER0_Q_NORM_DRAM = self.LAYER0_Q_DRAM
        # Temp buffer: v_proj interleaved output (T, 512) written during prefill before
        # per-head reorganization into the V KV cache (per-head layout).
        self.LAYER0_V_PROJ_TEMP = self.allocate_tensor_dram(seq_len * self.k_size)
        # Per-head flash output (T*group_size, actual_head_dim); reused across 8 KV heads.
        self.LAYER0_FLASH_OUT_HEAD_DRAM = self.allocate_tensor_dram(
            aligned_seq_len * self.actual_head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.head_dim * self.group_size * self.bytes_per_element)
        self.LAYER0_FLASH_SCRATCH_DRAM = self.allocate_tensor_dram(max(self.head_dim, UE_FMAX_CONTEXT_SIZE) * aligned_seq_len * 2 + self.head_dim * aligned_seq_len * 2)
        self.LAYER0_FLASH_BIAS_DRAM = self.allocate_tensor_dram(aligned_seq_len * aligned_seq_len * self.bytes_per_element)
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
        self.OUTPUT_NORM_DRAM = self.allocate_tensor_dram(1 * self.vector_length * self.bytes_per_element)
        self.LOGITS_DRAM = self.allocate_tensor_dram(1 * self.EMBEDDING_ELEMENTS * self.bytes_per_element)

        print(f"    Allocate tensor dram end at DRAM address: 0x{self.get_tensor_dram_addr():X}, usage: {self.get_tensor_dram_usage()} bytes")

    def program_execute(self, program_start_addr: int = user_dma_core.DRAM_INSTRUCTION_ADDR, timeout: float = 10.0, gflops: float = None) -> None:
        """Execute compiled program from DRAM instruction memory.
        """
        print(f"Execute program start at 0x{program_start_addr:X}")
        self.start_execute_from_dram(program_start_addr)
        self.wait_queue(timeout)
        latency = self.report_latency_in_us()
        print(f"    Total program execution latency = {latency} us")
        if gflops is not None:
            gflops_program, _ = self.report_flop_rate_gflops(gflops)
            print(f"Report FLOPS for program execution: {gflops_program:.2f} GFLOPS")

    def compile_prefill(self, seq_len: int, layer_size: int | None = None) -> dict:
        """Compile prefill for the given sequence length.

        LLaMA differences: Q/K norm steps are skipped; post-attn and post-MLP norm
        steps are skipped; K_DRAM/Q_DRAM are used directly for RoPE.
        """
        if layer_size is None:
            layer_size = self.LAYER_SIZE
        seq_len -= 1
        self.seq_len = seq_len
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64

        # --- LLaMA 3.2 16 layers: compile---
        global _SILENT_MODE
        _SILENT_MODE = True
        self.start_capture()
        total_flops = 0
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        for layer_idx in range(layer_size):
            layer_off = layer_idx * LAYER_WEIGHT_SIZE
            if layer_idx != 0:
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, sram_address=0x10000, element_size=seq_len * self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_INPUT_DRAM, element_size=seq_len * self.vector_length)
            total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off)

            total_flops += self.quantized_matmat_core(M=seq_len, K=self.vector_length, N=self.head_dim * self.group_size,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off, data_type=TYPE.IF4)
            total_flops += self.quantized_matmat_core(M=seq_len, K=self.vector_length, N=self.head_dim,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off, data_type=TYPE.IF4)
            # v_proj writes to interleaved temp (T, 512); per-head KV cache populated below.
            total_flops += self.quantized_matmat_core(M=seq_len, K=self.vector_length, N=self.head_dim,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_V_PROJ_TEMP,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off, data_type=TYPE.IF4)

            # LLaMA 8-head GQA: rope_hf_core(N=512) on [lo|hi]-permuted K and Q,
            # then scatter 64-dim per-head slices into per-head flash buffers and KV
            # cache, then run flash_attention(head_dim=64) × 8 KV heads.
            #
            # K/Q weight permutation (_rope_kv_perm) ensures that after k/q_proj matmul
            # the output is in [lo|hi] layout: [KV0_lo..KV7_lo, KV0_hi..KV7_hi].
            # rope_hf_core(N=512) with the 8-head tiled table then rotates each 64-dim
            # head correctly in-place.  After rope, K_h_roped = [K_h_lo(32), K_h_hi(32)]
            # is scattered from non-contiguous positions in K_DRAM to contiguous 64-dim
            # slots in the KV cache and FLASH_K_DRAM.
            ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE_GLOBAL if layer_idx in self._rope_global_layers else self.DRAM_ADDR_ROPE_LOCAL
            ahd      = self.actual_head_dim   # 64
            nkvh     = self.num_kv_heads      # 8
            qpkv     = self.group_size        # 4
            bpe      = self.bytes_per_element
            hd       = self.head_dim          # 512
            total_q_dim = hd * qpkv           # 2048 (o_proj input width)
            half_ahd = ahd // 2               # 32  (lo or hi slice width)
            rope_row = hd * 2 * bpe           # 2048 bytes per rope table row (N=512)

            # Phase 1: K rope in-place on LAYER0_K_DRAM (N=512, [lo|hi] layout)
            # After this, K_DRAM[t] = [K0_lo_r..K7_lo_r(256), K0_hi_r..K7_hi_r(256)]
            for t in range(seq_len):
                total_flops += self.rope_hf_core(
                    N=hd,
                    input_dram_addr=self.LAYER0_K_DRAM + t * hd * bpe,
                    output_dram_addr=self.LAYER0_K_DRAM + t * hd * bpe,
                    cos_dram_addr=ROPE_WEIGHT_ADDR + t * rope_row,
                    sin_dram_addr=ROPE_WEIGHT_ADDR + t * rope_row + hd * bpe)

            # Phase 2: Q rope in-place per 512-dim group (N=512, [lo|hi] layout)
            # 4 groups × T tokens; group g covers Q for KV heads 2g and 2g+1
            for g in range(qpkv):
                for t in range(seq_len):
                    q_t_g = self.LAYER0_Q_DRAM + t * total_q_dim * bpe + g * hd * bpe
                    total_flops += self.rope_hf_core(
                        N=hd,
                        input_dram_addr=q_t_g,
                        output_dram_addr=q_t_g,
                        cos_dram_addr=ROPE_WEIGHT_ADDR + t * rope_row,
                        sin_dram_addr=ROPE_WEIGHT_ADDR + t * rope_row + hd * bpe)

            # Phase 3: Per-KV-head scatter → KV cache + flash buffers → flash_attention
            for kv_h in range(nkvh):
                k_cache_base = (self.LAYER0_K_ROPE_DRAM
                                + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)
                v_cache_base = (self.LAYER0_V_DRAM
                                + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)

                # Scatter K_h_roped (64-dim) from [lo|hi] K_DRAM → KV cache + FLASH_K
                # K_DRAM[t][lo]: kv_h*half_ahd..kv_h*half_ahd+32
                # K_DRAM[t][hi]: hd//2+kv_h*half_ahd..hd//2+kv_h*half_ahd+32
                # SRAM slots: lo→0x10000, hi→0x10080 (each 64 bytes; must be 128-byte aligned)
                for t in range(seq_len):
                    k_t = self.LAYER0_K_DRAM + t * hd * bpe
                    self.accelerator_memory_to_sram(
                        k_t + kv_h * half_ahd * bpe, 0x10000, half_ahd)
                    self.accelerator_memory_to_sram(
                        k_t + (hd // 2 + kv_h * half_ahd) * bpe,
                        0x10080, half_ahd)
                    k_dst = k_cache_base + t * ahd * bpe
                    self.sram_to_accelerator_memory(0x10000, k_dst, half_ahd)
                    self.sram_to_accelerator_memory(0x10080, k_dst + half_ahd * bpe, half_ahd)
                    for g in range(qpkv):
                        k_f = self.LAYER0_FLASH_K_DRAM + (t * qpkv + g) * ahd * bpe
                        self.sram_to_accelerator_memory(0x10000, k_f, half_ahd)
                        self.sram_to_accelerator_memory(0x10080, k_f + half_ahd * bpe, half_ahd)

                # Scatter V_h (64-dim, standard layout) from V_PROJ_TEMP → KV cache + FLASH_V
                # V_PROJ_TEMP[t] = [V_KV0(64), V_KV1(64), ..., V_KV7(64)] = 512-dim
                for t in range(seq_len):
                    v_src = self.LAYER0_V_PROJ_TEMP + (t * nkvh + kv_h) * ahd * bpe
                    self.accelerator_memory_to_sram(v_src, 0x20000, ahd)
                    self.sram_to_accelerator_memory(
                        0x20000, v_cache_base + t * ahd * bpe, ahd)
                    for g in range(qpkv):
                        self.sram_to_accelerator_memory(
                            0x20000,
                            self.LAYER0_FLASH_V_DRAM + (t * qpkv + g) * ahd * bpe, ahd)

                # Scatter Q_h_q (64-dim) from [lo|hi] Q_DRAM → FLASH_Q
                # KV head kv_h → Q group g = kv_h//2, local_kv = kv_h%2
                # Sub-head q of KV head kv_h: sub_idx = local_kv*qpkv + q  (0..7 within group)
                g_for_kv = kv_h // 2
                local_kv = kv_h % 2
                for t in range(seq_len):
                    q_g_t = self.LAYER0_Q_DRAM + t * total_q_dim * bpe + g_for_kv * hd * bpe
                    for q in range(qpkv):
                        sub_idx = local_kv * qpkv + q
                        self.accelerator_memory_to_sram(
                            q_g_t + sub_idx * half_ahd * bpe, 0x30000, half_ahd)
                        self.accelerator_memory_to_sram(
                            q_g_t + (hd // 2 + sub_idx * half_ahd) * bpe,
                            0x30080, half_ahd)
                        q_dst = self.LAYER0_FLASH_Q_DRAM + (t * qpkv + q) * ahd * bpe
                        self.sram_to_accelerator_memory(0x30000, q_dst, half_ahd)
                        self.sram_to_accelerator_memory(0x30080, q_dst + half_ahd * bpe, half_ahd)

                # Flash attention for this KV head (head_dim=64)
                total_flops += self.flash_attention_core(
                    head_dim=ahd,
                    seq_len=aligned_seq_len,
                    Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
                    K_DRAM_ADDR=self.LAYER0_FLASH_K_DRAM,
                    V_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                    OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUT_HEAD_DRAM,
                    SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                    BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_DRAM,
                )

                # Assemble output into LAYER0_FLASH_OUTPUT_DRAM
                # Standard GQA layout per token: [kv0_q0(64), kv0_q1..q3, kv1_q0..q3, ...]
                out_h_base = kv_h * qpkv * ahd * bpe
                for t in range(seq_len):
                    for g in range(qpkv):
                        src = self.LAYER0_FLASH_OUT_HEAD_DRAM + (t * qpkv + g) * ahd * bpe
                        dst = (self.LAYER0_FLASH_OUTPUT_DRAM
                               + t * total_q_dim * bpe + out_h_base + g * ahd * bpe)
                        self.accelerator_memory_to_sram(src, 0x40000, ahd)
                        self.sram_to_accelerator_memory(0x40000, dst, ahd)

            total_flops += self.quantized_matmat_core(M=seq_len, K=self.head_dim * self.group_size, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off, data_type=TYPE.IF4)

            # LLaMA: no post-attention norm; add residual directly to o_proj output
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_INPUT_DRAM, sram_address=0x10000, element_size=seq_len * self.vector_length)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM, sram_address=0x90000, element_size=seq_len * self.vector_length)
            self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=seq_len * self.vector_length)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, element_size=seq_len * self.vector_length)

            # LLaMA: post_attention_layernorm IS the pre-FFN norm
            total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off)
            total_flops += self.quantized_matmat_core(M=seq_len, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off, data_type=TYPE.IF4, silu_enable=True)
            total_flops += self.quantized_matmat_core(M=seq_len, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off, data_type=TYPE.IF4)
            # gate × up — chunked row-by-row; seq_len*mlp_elements = 344064 elems = 688KB
            # which overflows SRAM if loaded in one shot at 0x10000/0x90000.  One row = 16KB ✓
            _bpe = self.bytes_per_element
            for _t in range(seq_len):
                _g_row = self.LAYER0_MLP_GATE_DRAM + _t * self.mlp_elements * _bpe
                _u_row = self.LAYER0_MLP_UP_DRAM   + _t * self.mlp_elements * _bpe
                _m_row = self.LAYER0_MLP_MULT_DRAM  + _t * self.mlp_elements * _bpe
                self.accelerator_memory_to_sram(_g_row, 0x10000, self.mlp_elements)
                self.accelerator_memory_to_sram(_u_row, 0x90000, self.mlp_elements)
                self.eltwise_mul_core(0x10000, 0x90000, 0x10000, self.mlp_elements)
                self.sram_to_accelerator_memory(0x10000, _m_row, self.mlp_elements)
            total_flops += self.quantized_matmat_core(M=seq_len, K=self.mlp_elements, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off, data_type=TYPE.IF4)

            # LLaMA: no post-FFN norm; add residual directly to down_proj output
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, sram_address=0x10000, element_size=seq_len * self.vector_length)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_DOWN_DRAM, sram_address=0x90000, element_size=seq_len * self.vector_length)
            self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=seq_len * self.vector_length)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, element_size=seq_len * self.vector_length)
        self.stop_capture()
        self.generate_instruction_halt()
        prefill_program_addr = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prefill_program_addr)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        self.clear_capture_buffer()
        _SILENT_MODE = False
        print(f"    Prefill program start at 0x{prefill_program_addr:X} end at 0x{self.get_program_dram_addr():X}, usage: {self.get_program_dram_usage()} bytes")

        return prefill_program_addr, total_flops

    def compile_prefill_buckets(self, layer_size: int | None = None) -> tuple[list[int], list[int]]:
        
        if layer_size is None:
            layer_size = self.LAYER_SIZE
        paths_cfg = self._cfg.get("paths", {})
        prefill_bin_rel  = paths_cfg.get("prefill_program_bin",  "llama3.2_1b_bin/prefill_program.bin")
        prefill_meta_rel = paths_cfg.get("prefill_program_meta", "llama3.2_1b_bin/prefill_program.json")
        prefill_bin_path  = os.path.join(self.script_dir, prefill_bin_rel)
        prefill_meta_path = os.path.join(self.script_dir, prefill_meta_rel)
        if os.path.exists(prefill_bin_path) and os.path.exists(prefill_meta_path):
            with open(prefill_meta_path) as f:
                meta = json.load(f)
            program_sizes = meta.get("program_sizes", [c * 32 for c in meta["instruction_counts"]])
            print(f"Prefill bin found, skipping compile.")
            return program_sizes, meta["total_flops"]
        os.makedirs(os.path.dirname(prefill_bin_path), exist_ok=True)
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        prefill_buckets = self._cfg["model"]["prefill_seq_len_buckets"]
        segment_instruction_counts = []
        total_flops_list = []

        global _SILENT_MODE
        _SILENT_MODE = True
        self.clear_inst_id()
        self.start_capture()
        for bucket in prefill_buckets:
            count_at_start = self.capture_count
            total_flops = 0
            seq_len = bucket - 1
            self.seq_len = seq_len
            q_seq_len = seq_len * self.group_size
            aligned_seq_len = ((q_seq_len + 63) // 64) * 64
            for layer_idx in range(layer_size):
                layer_off = layer_idx * LAYER_WEIGHT_SIZE
                if layer_idx != 0:
                    self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, sram_address=0x10000, element_size=seq_len * self.vector_length)
                    self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_INPUT_DRAM, element_size=seq_len * self.vector_length)
                total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                    OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off)
                total_flops += self.quantized_matmat_core(M=seq_len, K=self.vector_length, N=self.head_dim * self.group_size,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off, data_type=TYPE.IF4)
                total_flops += self.quantized_matmat_core(M=seq_len, K=self.vector_length, N=self.head_dim,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off, data_type=TYPE.IF4)
                total_flops += self.quantized_matmat_core(M=seq_len, K=self.vector_length, N=self.head_dim,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_V_PROJ_TEMP,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off, data_type=TYPE.IF4)
                ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE_GLOBAL if layer_idx in self._rope_global_layers else self.DRAM_ADDR_ROPE_LOCAL
                ahd = self.actual_head_dim
                nkvh = self.num_kv_heads
                qpkv = self.group_size
                bpe = self.bytes_per_element
                hd = self.head_dim
                total_q_dim = hd * qpkv
                half_ahd = ahd // 2
                rope_row = hd * 2 * bpe
                for t in range(seq_len):
                    total_flops += self.rope_hf_core(N=hd,
                        input_dram_addr=self.LAYER0_K_DRAM + t * hd * bpe,
                        output_dram_addr=self.LAYER0_K_DRAM + t * hd * bpe,
                        cos_dram_addr=ROPE_WEIGHT_ADDR + t * rope_row,
                        sin_dram_addr=ROPE_WEIGHT_ADDR + t * rope_row + hd * bpe)
                for g in range(qpkv):
                    for t in range(seq_len):
                        q_t_g = self.LAYER0_Q_DRAM + t * total_q_dim * bpe + g * hd * bpe
                        total_flops += self.rope_hf_core(N=hd, input_dram_addr=q_t_g, output_dram_addr=q_t_g,
                            cos_dram_addr=ROPE_WEIGHT_ADDR + t * rope_row,
                            sin_dram_addr=ROPE_WEIGHT_ADDR + t * rope_row + hd * bpe)
                for kv_h in range(nkvh):
                    k_cache_base = (self.LAYER0_K_ROPE_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)
                    v_cache_base = (self.LAYER0_V_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)
                    for t in range(seq_len):
                        k_t = self.LAYER0_K_DRAM + t * hd * bpe
                        self.accelerator_memory_to_sram(k_t + kv_h * half_ahd * bpe, 0x10000, half_ahd)
                        self.accelerator_memory_to_sram(k_t + (hd // 2 + kv_h * half_ahd) * bpe, 0x10080, half_ahd)
                        k_dst = k_cache_base + t * ahd * bpe
                        self.sram_to_accelerator_memory(0x10000, k_dst, half_ahd)
                        self.sram_to_accelerator_memory(0x10080, k_dst + half_ahd * bpe, half_ahd)
                        for g in range(qpkv):
                            k_f = self.LAYER0_FLASH_K_DRAM + (t * qpkv + g) * ahd * bpe
                            self.sram_to_accelerator_memory(0x10000, k_f, half_ahd)
                            self.sram_to_accelerator_memory(0x10080, k_f + half_ahd * bpe, half_ahd)
                    for t in range(seq_len):
                        v_src = self.LAYER0_V_PROJ_TEMP + (t * nkvh + kv_h) * ahd * bpe
                        self.accelerator_memory_to_sram(v_src, 0x20000, ahd)
                        self.sram_to_accelerator_memory(0x20000, v_cache_base + t * ahd * bpe, ahd)
                        for g in range(qpkv):
                            self.sram_to_accelerator_memory(0x20000, self.LAYER0_FLASH_V_DRAM + (t * qpkv + g) * ahd * bpe, ahd)
                    g_for_kv = kv_h // 2
                    local_kv = kv_h % 2
                    for t in range(seq_len):
                        q_g_t = self.LAYER0_Q_DRAM + t * total_q_dim * bpe + g_for_kv * hd * bpe
                        for q in range(qpkv):
                            sub_idx = local_kv * qpkv + q
                            self.accelerator_memory_to_sram(q_g_t + sub_idx * half_ahd * bpe, 0x30000, half_ahd)
                            self.accelerator_memory_to_sram(q_g_t + (hd // 2 + sub_idx * half_ahd) * bpe, 0x30080, half_ahd)
                            q_dst = self.LAYER0_FLASH_Q_DRAM + (t * qpkv + q) * ahd * bpe
                            self.sram_to_accelerator_memory(0x30000, q_dst, half_ahd)
                            self.sram_to_accelerator_memory(0x30080, q_dst + half_ahd * bpe, half_ahd)
                    total_flops += self.flash_attention_core(head_dim=ahd, seq_len=aligned_seq_len,
                        Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM, K_DRAM_ADDR=self.LAYER0_FLASH_K_DRAM,
                        V_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM, OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUT_HEAD_DRAM,
                        SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM, BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_DRAM)
                    out_h_base = kv_h * qpkv * ahd * bpe
                    for t in range(seq_len):
                        for g in range(qpkv):
                            src = self.LAYER0_FLASH_OUT_HEAD_DRAM + (t * qpkv + g) * ahd * bpe
                            dst = self.LAYER0_FLASH_OUTPUT_DRAM + t * total_q_dim * bpe + out_h_base + g * ahd * bpe
                            self.accelerator_memory_to_sram(src, 0x40000, ahd)
                            self.sram_to_accelerator_memory(0x40000, dst, ahd)
                total_flops += self.quantized_matmat_core(M=seq_len, K=self.head_dim * self.group_size, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off, data_type=TYPE.IF4)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_INPUT_DRAM, sram_address=0x10000, element_size=seq_len * self.vector_length)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM, sram_address=0x90000, element_size=seq_len * self.vector_length)
                self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=seq_len * self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, element_size=seq_len * self.vector_length)
                total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                    OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off)
                total_flops += self.quantized_matmat_core(M=seq_len, K=self.vector_length, N=self.mlp_elements,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off, data_type=TYPE.IF4, silu_enable=True)
                total_flops += self.quantized_matmat_core(M=seq_len, K=self.vector_length, N=self.mlp_elements,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off, data_type=TYPE.IF4)
                _bpe = self.bytes_per_element
                for _t in range(seq_len):
                    _g_row = self.LAYER0_MLP_GATE_DRAM + _t * self.mlp_elements * _bpe
                    _u_row = self.LAYER0_MLP_UP_DRAM   + _t * self.mlp_elements * _bpe
                    _m_row = self.LAYER0_MLP_MULT_DRAM  + _t * self.mlp_elements * _bpe
                    self.accelerator_memory_to_sram(_g_row, 0x10000, self.mlp_elements)
                    self.accelerator_memory_to_sram(_u_row, 0x90000, self.mlp_elements)
                    self.eltwise_mul_core(0x10000, 0x90000, 0x10000, self.mlp_elements)
                    self.sram_to_accelerator_memory(0x10000, _m_row, self.mlp_elements)
                total_flops += self.quantized_matmat_core(M=seq_len, K=self.mlp_elements, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off, data_type=TYPE.IF4)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, sram_address=0x10000, element_size=seq_len * self.vector_length)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_DOWN_DRAM, sram_address=0x90000, element_size=seq_len * self.vector_length)
                self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=seq_len * self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, element_size=seq_len * self.vector_length)
            self.generate_instruction_halt()
            segment_instruction_counts.append(self.capture_count - count_at_start)
            total_flops_list.append(total_flops)
        self.stop_capture()
        _SILENT_MODE = False
        all_programs_bytes = bytearray()
        for inst in self.capture_buffer:
            all_programs_bytes.extend(inst.get_bytes())
        with open(prefill_bin_path, "wb") as f:
            f.write(all_programs_bytes)
        program_sizes = [c * 32 for c in segment_instruction_counts]
        with open(prefill_meta_path, "w") as f:
            json.dump({"instruction_counts": segment_instruction_counts, "program_sizes": program_sizes,
                       "total_flops": total_flops_list, "buckets": prefill_buckets}, f, indent=0)
        self.clear_capture_buffer()
        print(f"Prefill programs: {len(segment_instruction_counts)} buckets written to {prefill_bin_path} ({len(all_programs_bytes)} bytes)")
        return program_sizes, total_flops_list

    def run_prefill(self, prefill_program_addr: int, prefill_seq, gflops: int = None) -> dict:
        """
        Run prefill for the given prefill sequence.
        
        Args:
            prefill_program_addr: The address of the prefill program in DRAM.
            prefill_seq: The prefill sequence.
            gflops: The number of GFLOPS to use for the prefill.
        
        Returns:
            A dictionary containing the results of the prefill.
        """
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

        embedding_tensor = self.get_embedding_for_tokens(prefill_seq)
        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
        # Block-causal mask: all group_size Q copies of token t may attend to ALL
        # group_size K copies of tokens 0..t (not just j <= i as in tril).
        # This is correct for GQA duplication where K[t*gs+g] = K_unique[t] for all g.
        bias_one_group = torch.full((aligned_seq_len, aligned_seq_len), float("-inf"), dtype=torch.bfloat16)
        rows = torch.arange(aligned_seq_len).unsqueeze(1)
        cols = torch.arange(aligned_seq_len).unsqueeze(0)
        valid_mask = (cols // self.group_size) <= (rows // self.group_size)
        bias_one_group.masked_fill_(valid_mask, 0.0)
        bias_one_group[:, q_seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_one_group)
        self.program_execute(prefill_program_addr, gflops=gflops)

    def compile_decoder(self, layer_size: int | None = None) -> tuple[int, int]:
        """Compile single-segment PBI decoder; write decoder_program.bin and decoder_program.json.
        Returns (program_size_bytes, total_flops)."""
        if layer_size is None:
            layer_size = self.LAYER_SIZE
        paths_cfg = self._cfg.get("paths", {})
        decoder_bin_rel = paths_cfg.get("decoder_program_bin", "llama3.2_1b_bin/decoder_program.bin")
        decoder_meta_rel = paths_cfg.get("decoder_program_meta", "llama3.2_1b_bin/decoder_program.json")
        decoder_bin_path = os.path.join(self.script_dir, decoder_bin_rel)
        decoder_meta_path = os.path.join(self.script_dir, decoder_meta_rel)
        os.makedirs(os.path.dirname(decoder_bin_path), exist_ok=True)
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        total_flops = 0

        global _SILENT_MODE
        _SILENT_MODE = True
        self.clear_inst_id()
        self.start_capture()
        for layer_idx in range(layer_size):
                layer_off = layer_idx * LAYER_WEIGHT_SIZE
                if layer_idx != 0:
                    self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, sram_address=0x10000, element_size=self.vector_length)
                    self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_INPUT_DRAM, element_size=self.vector_length)
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off)
                total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=self.head_dim * self.group_size,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off, data_type=TYPE.IF4)
                total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=self.head_dim,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off, data_type=TYPE.IF4)
                total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=self.head_dim,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off, data_type=TYPE.IF4)

                # LLaMA 8-head GQA decoder: rope_hf_core(N=512) on [lo|hi]-permuted K and Q
                # in-place, then scatter 64-dim per-head slices to KV cache (via
                # V_CACHE_SIZE_REG for decode position), then decoder_attention(head_dim=64).
                ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE_GLOBAL if layer_idx in self._rope_global_layers else self.DRAM_ADDR_ROPE_LOCAL
                ahd      = self.actual_head_dim   # 64
                nkvh     = self.num_kv_heads      # 8
                qpkv     = self.group_size        # 4
                bpe      = self.bytes_per_element
                hd       = self.head_dim          # 512
                total_q_dim = hd * qpkv           # 2048
                half_ahd = ahd // 2               # 32

                # Step 1: K rope in-place (N=512, uses ROPE_SIZE_REG for decode position)
                total_flops += self.rope_hf_core(
                    N=hd,
                    input_dram_addr=self.LAYER0_K_DRAM,
                    output_dram_addr=self.LAYER0_K_DRAM,
                    cos_dram_addr=ROPE_WEIGHT_ADDR,
                    sin_dram_addr=ROPE_WEIGHT_ADDR + hd * bpe,
                    rope_size_reg=self.ROPE_SIZE_REG,
                    tmp_reg=self.TMP_REG)

                # Step 2: Q rope in-place per 512-dim group (N=512, uses ROPE_SIZE_REG)
                for g in range(qpkv):
                    total_flops += self.rope_hf_core(
                        N=hd,
                        input_dram_addr=self.LAYER0_Q_DRAM + g * hd * bpe,
                        output_dram_addr=self.LAYER0_Q_DRAM + g * hd * bpe,
                        cos_dram_addr=ROPE_WEIGHT_ADDR,
                        sin_dram_addr=ROPE_WEIGHT_ADDR + hd * bpe,
                        rope_size_reg=self.ROPE_SIZE_REG,
                        tmp_reg=self.TMP_REG)

                # Step 3: Per-KV-head scatter K/V to cache + scatter Q → decoder_attention
                for kv_h in range(nkvh):
                    k_cache_base = (self.LAYER0_K_ROPE_DRAM
                                    + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size
                                    + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)
                    v_cache_base = (self.LAYER0_V_DRAM
                                    + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size
                                    + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)

                    # Scatter K_h_roped (64-dim) → KV cache at decode position
                    # lo→SRAM 0x10000, hi→SRAM 0x10080 (128-byte aligned slots)
                    self.accelerator_memory_to_sram(
                        self.LAYER0_K_DRAM + kv_h * half_ahd * bpe, 0x10000, half_ahd)
                    self.accelerator_memory_to_sram(
                        self.LAYER0_K_DRAM + (hd // 2 + kv_h * half_ahd) * bpe,
                        0x10080, half_ahd)
                    self.generate_instruction_add_imm(
                        self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter(k_cache_base), self.TMP_REG)
                    self.sram_to_accelerator_memory(0x10000, 0, half_ahd)
                    self.overwrite_instruction_with_general_register(self.TMP_REG)
                    self.generate_instruction_add_imm(
                        self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter(k_cache_base + half_ahd * bpe), self.TMP_REG)
                    self.sram_to_accelerator_memory(0x10080, 0, half_ahd)
                    self.overwrite_instruction_with_general_register(self.TMP_REG)

                    # Scatter V_h (64-dim, standard layout) → V cache at decode position
                    # v_proj output at LAYER0_FLASH_V_DRAM: [V_KV0(64)..V_KV7(64)] = 512-dim
                    self.accelerator_memory_to_sram(
                        self.LAYER0_FLASH_V_DRAM + kv_h * ahd * bpe, 0x20000, ahd)
                    self.generate_instruction_add_imm(
                        self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter(v_cache_base), self.TMP_REG)
                    self.sram_to_accelerator_memory(0x20000, 0, ahd)
                    self.overwrite_instruction_with_general_register(self.TMP_REG)

                    # Scatter Q_h_q (64-dim) from [lo|hi] Q_DRAM → FLASH_Q → decoder_attn
                    # KV head kv_h → Q group g = kv_h//2; sub_idx = (kv_h%2)*qpkv + q
                    g_for_kv = kv_h // 2
                    local_kv = kv_h % 2
                    q_g_addr = self.LAYER0_Q_DRAM + g_for_kv * hd * bpe
                    for q in range(qpkv):
                        sub_idx = local_kv * qpkv + q
                        flash_q_addr = self.LAYER0_FLASH_Q_DRAM + (kv_h * qpkv + q) * ahd * bpe
                        self.accelerator_memory_to_sram(
                            q_g_addr + sub_idx * half_ahd * bpe, 0x30000, half_ahd)
                        self.accelerator_memory_to_sram(
                            q_g_addr + (hd // 2 + sub_idx * half_ahd) * bpe,
                            0x30080, half_ahd)
                        self.sram_to_accelerator_memory(0x30000, flash_q_addr, half_ahd)
                        self.sram_to_accelerator_memory(0x30080, flash_q_addr + half_ahd * bpe, half_ahd)
                        attn_flops = self.decoder_group_attention_core(
                            group_size=1,
                            head_dim=ahd,
                            seq_len=self.MAX_CONTEXT_SIZE,
                            gf_bucket_idx=self.gf_bucket_idx,
                            num_buckets=8,
                            Q_DRAM_ADDR=flash_q_addr,
                            K_DRAM_ADDR=k_cache_base,
                            V_DRAM_ADDR=v_cache_base,
                            OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM + (kv_h * qpkv + q) * ahd * bpe,
                            IDENTITY_DRAM_ADDR=self.IDENTITY_DRAM_ADDR,
                            SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                            BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_DRAM,
                        )
                        total_flops += attn_flops[-1]
                total_flops += self.quantized_matmat_core(M=1, K=self.head_dim * self.group_size, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off, data_type=TYPE.IF4)

                # LLaMA: no post-attention norm; residual directly on o_proj output
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_INPUT_DRAM, sram_address=0x10000, element_size=self.vector_length)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM, sram_address=0x90000, element_size=self.vector_length)
                self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, element_size=self.vector_length)

                # LLaMA: post_attention_layernorm IS the pre-FFN norm
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off)

                total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=self.mlp_elements,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off, data_type=TYPE.IF4, silu_enable=True)
                total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=self.mlp_elements,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off, data_type=TYPE.IF4)

                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_GATE_DRAM, sram_address=0x10000, element_size=self.mlp_elements)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_UP_DRAM, sram_address=0x90000, element_size=self.mlp_elements)
                self.eltwise_mul_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.mlp_elements)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_MLP_MULT_DRAM, element_size=self.mlp_elements)

                total_flops += self.quantized_matmat_core(M=1, K=self.mlp_elements, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off, data_type=TYPE.IF4)

                # LLaMA: no post-FFN norm; residual directly on down_proj output
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, sram_address=0x10000, element_size=self.vector_length)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_DOWN_DRAM, sram_address=0x90000, element_size=self.vector_length)
                self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, element_size=self.vector_length)

        if layer_size == self.LAYER_SIZE:
            total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_OUTPUT_DRAM,
                OUTPUT_DRAM_ADDR=self.OUTPUT_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_OUTPUT_NORM_GAMMA)
            total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=self.EMBEDDING_ELEMENTS,
                A_DRAM_ADDR=self.OUTPUT_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_QUANT, OUTPUT_DRAM_ADDR=self.LOGITS_DRAM,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_SCALE, data_type=TYPE.IF4)

        self.generate_instruction_halt()
        self.stop_capture()
        _SILENT_MODE = False
        program_bytes = bytearray()
        for inst in self.capture_buffer:
            program_bytes.extend(inst.get_bytes())
        with open(decoder_bin_path, "wb") as f:
            f.write(program_bytes)
        program_size = len(program_bytes)
        with open(decoder_meta_path, "w") as f:
            json.dump({"program_size": program_size, "total_flops": total_flops}, f, indent=0)
        self.clear_capture_buffer()
        print(f"Decoder program written to {decoder_bin_path} ({program_size} bytes)")
        return program_size, total_flops

    def run_decoder(self, decoder_base_addr: int, token_id: int, total_flops: int | None = None) -> int:
        """Run decode loop. seq_len capped at MAX_CONTEXT_SIZE. Breaks on LLaMA EOS/EOT tokens."""
        if token_id is None:
            print("No last token available for decode.")
            return 0

        # LLaMA stop tokens: <|end_of_text|>=128001, <|eom_id|>=128008, <|eot_id|>=128009
        _llama_stop_tokens = {128001, 128008, self._end_of_turn_token_id}

        global _SILENT_MODE
        max_seq_len = self.MAX_CONTEXT_SIZE
        while self.seq_len < max_seq_len:
            _SILENT_MODE = True
            timer_start = time.perf_counter()
            self.seq_len += 1
            aligned_seq_len = ((self.seq_len + 63) // 64) * 64
            prog_addr = decoder_base_addr
            gflops = total_flops

            # GF_BUCKET_IDX_REG: 1-based bucket index (seq_len → bucket 1..8, each covers 64 tokens)
            self.isa_add_set_core(self.gf_bucket_idx, min((self.seq_len + 63) // 64, 8))
            # V_CACHE_SIZE_REG: decode_pos × actual_hd × bpe (per-head KV cache stride = 128B)
            # ROPE_SIZE_REG:    decode_pos × head_dim × 2 × bpe (N=512 rope row = 2048B)
            _kv_stride  = self.actual_head_dim * self.bytes_per_element  # 128 bytes/position
            _rope_row   = self.head_dim * 2 * self.bytes_per_element     # 2048 bytes/position
            self.isa_add_set_core(self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter((self.seq_len - 1) * _kv_stride))
            self.isa_add_set_core(self.ROPE_SIZE_REG,    ue_35bit_addr_shifter((self.seq_len - 1) * _rope_row))

            embedding_tensor = self.get_embedding_for_tokens([token_id])
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)

            bias_host = torch.full((1, aligned_seq_len), -1e36, dtype=torch.bfloat16)
            bias_host[0, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_host)

            self.start_execute_from_dram(prog_addr)
            self.wait_queue(10.0)
            token_id = self.get_arg_max_index()
            token_char = self.tokenizer.decode([token_id])
            _SILENT_MODE = False
            if token_id in _llama_stop_tokens:
                print(f"\nStop token {token_id} reached.")
                break
            print(token_char, end="", flush=True)

        return self.seq_len

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Llama-3.2-1B prefill + decode on accelerator.")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt: tokenizer encodes this to prefill_seq (overrides default)")
    parser.add_argument("--local-weights", action="store_true", help="Use llama3.2_1b_bin/full_model_weights.bin instead of generated weights_llama3.2_1b_hf.bin")
    parser.add_argument('--dev', type=str, default='xdma0',
                        help='DMA device name (e.g., xdma0, xdma1). Default: xdma0')
    parser.add_argument('--cycle', type=float, default=1/0.17,
                        help='Clock cycle time in nanoseconds (default: ~5.88ns)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.local_weights:
        weights_bin_rel = "llama3.2_1b_bin/full_model_weights.bin"
    else:
        weights_bin_rel = "llama3.2_1b_bin/weights_llama3.2_1b_hf.bin"
        weights_bin_full = os.path.join(script_dir, weights_bin_rel)
        if not os.path.exists(weights_bin_full):
            weight_bin_generate(script_dir=script_dir, output_path=weights_bin_full)

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
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle
    print(f"Using DMA device: {args.dev}")
    print(f"  H2C: {DMA_DEVICE_H2C}")
    print(f"  C2H: {DMA_DEVICE_C2H}")
    print(f"  USER: {DMA_DEVICE_USER}")
    print(f"Setting CLOCK_CYCLE_TIME_NS = {user_dma_core.CLOCK_CYCLE_TIME_NS}")

    ue = Llama32_1b_UnifiedEngine(script_dir=script_dir, weights_bin=weights_bin_rel)
    ue.software_reset()
    cfg = _load_config(script_dir)
    if args.prompt is not None:
        tok_path = os.path.join(script_dir, cfg["paths"]["hf_model_dir"])
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
        conversation = [{"role": "user", "content": args.prompt}]
        prompt_with_template = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        prefill_seq = tuple(tokenizer.encode(prompt_with_template, add_special_tokens=False))
        print(f"Prefill from prompt ({len(prefill_seq)} tokens): {args.prompt!r}")
        print(f"Sequence ids: {prefill_seq}")
    else:
        prefill_seq = tuple(cfg["default_prefill_tokens"])

    print(f"\n--- Compiling ---")
    timer = time.perf_counter()
    prefill_program_sizes, gflops_prefill_list = ue.compile_prefill_buckets()
    print(f"Prefill buckets ready in {time.perf_counter() - timer:.2f}s")

    decoder_bin_path = os.path.join(script_dir, "llama3.2_1b_bin", "decoder_program.bin")
    decoder_meta_path = os.path.join(script_dir, "llama3.2_1b_bin", "decoder_program.json")
    if os.path.exists(decoder_bin_path) and os.path.exists(decoder_meta_path):
        with open(decoder_meta_path, "r") as f:
            meta = json.load(f)
        # Support old multi-segment format: total_flops was a list; use max bucket value.
        raw_flops = meta["total_flops"]
        total_flops = raw_flops[-1] if isinstance(raw_flops, list) else raw_flops
        print(f"Decoder bin found, skipped compile ({time.perf_counter() - timer:.2f}s).")
    else:
        timer_dec = time.perf_counter()
        _, total_flops = ue.compile_decoder()
        print(f"Decoder compile done in {time.perf_counter() - timer_dec:.2f} seconds.")
    decoder_base_addr, _ = ue.load_program_instructions_from_file(decoder_bin_path)

    # Load prefill bin and pick bucket
    prefill_bin_path  = os.path.join(script_dir, "llama3.2_1b_bin", "prefill_program.bin")
    prefill_meta_path = os.path.join(script_dir, "llama3.2_1b_bin", "prefill_program.json")
    prefill_base_addr, _ = ue.load_instructions(prefill_bin_path)
    with open(prefill_meta_path) as f:
        prefill_meta = json.load(f)
    prefill_buckets = prefill_meta["buckets"]
    actual_seq_len = len(prefill_seq) - 1
    bucket_idx = next(i for i, b in enumerate(prefill_buckets) if b - 1 >= actual_seq_len)
    bucket_seq_len = prefill_buckets[bucket_idx] - 1
    prefill_program_addr = prefill_base_addr + sum(prefill_program_sizes[:bucket_idx])
    gflops_prefill = gflops_prefill_list[bucket_idx]
    ue.seq_len = actual_seq_len

    print(f"\n--- Starting prefill ({actual_seq_len} tokens, bucket={bucket_seq_len}) ---")
    timer = time.perf_counter()
    embedding_tensor = ue.get_embedding_for_tokens(list(prefill_seq[:-1]))
    if actual_seq_len < bucket_seq_len:
        pad = embedding_tensor[-1:].repeat(bucket_seq_len - actual_seq_len, 1)
        embedding_tensor = torch.cat([embedding_tensor, pad], dim=0)
    q_seq_len = bucket_seq_len * ue.group_size
    aligned_seq_len_q = ((q_seq_len + 63) // 64) * 64
    ue.dma_to_accelerator_memory(ue.LAYER0_INPUT_DRAM, embedding_tensor)
    bias_one_group = torch.full((aligned_seq_len_q, aligned_seq_len_q), float("-inf"), dtype=torch.bfloat16)
    rows = torch.arange(aligned_seq_len_q).unsqueeze(1)
    cols = torch.arange(aligned_seq_len_q).unsqueeze(0)
    valid_mask = (cols // ue.group_size) <= (rows // ue.group_size)
    bias_one_group.masked_fill_(valid_mask, 0.0)
    bias_one_group[:, q_seq_len:] = float("-inf")
    ue.dma_to_accelerator_memory(ue.LAYER0_FLASH_BIAS_DRAM, bias_one_group)
    ue.program_execute(prefill_program_addr, gflops=gflops_prefill)
    if actual_seq_len < bucket_seq_len:
        zero_vec = torch.zeros(ue.actual_head_dim, dtype=torch.bfloat16)
        bpe = ue.bytes_per_element
        ahd = ue.actual_head_dim
        for layer_idx in range(ue.LAYER_SIZE):
            for kv_h in range(ue.num_kv_heads):
                k_base = ue.LAYER0_K_ROPE_DRAM + layer_idx * ue.MAX_CONTEXT_SIZE * ue.k_size + kv_h * ue.MAX_CONTEXT_SIZE * ahd * bpe
                v_base = ue.LAYER0_V_DRAM + layer_idx * ue.MAX_CONTEXT_SIZE * ue.k_size + kv_h * ue.MAX_CONTEXT_SIZE * ahd * bpe
                for t in range(actual_seq_len, bucket_seq_len):
                    ue.dma_to_accelerator_memory(k_base + t * ahd * bpe, zero_vec)
                    ue.dma_to_accelerator_memory(v_base + t * ahd * bpe, zero_vec)
    latency_prefill = time.perf_counter() - timer
    print(f"Prefill execute done in {latency_prefill:.2f} seconds, start decoding...\n")

    print(f"\n--- Starting decoder ---")
    timer=time.perf_counter()
    token_cnt_decoded = ue.run_decoder(decoder_base_addr, token_id=prefill_seq[-1], total_flops=total_flops)
    latency_decoder = time.perf_counter() - timer
    print(f"\nDecoder done in {latency_prefill + latency_decoder:.2f} seconds, total {token_cnt_decoded} tokens.")
    print("Llama-3.2-1B test ends.")

if __name__ == "__main__":
    main()
