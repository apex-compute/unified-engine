#!/usr/bin/env python3
"""
Llama-3.2-1B IF8 inference on accelerator: prefill + decode.

This is the IF8 (8-bit adaptive block-scaled) variant of
``llama3.2_1b_test.py``, following the design used by
``models/gemma3/gemma3_test_IF8.py``:

  - Quantized weight data regions are doubled and repacked because IF8 stores
    one byte per element instead of two elements per byte.
  - Every quantized matmul uses ``TYPE.IF8``.
  - Weights use per-block MixMSE selection between INT8 and FP8 E4M3.
  - IF8 weights and programs live in ``llama3.2_1b_if8_bin/`` so they cannot
    collide with the IF4/FP4 artifacts.

  - Config from llama3.2_1b_config.json; its on-disk IF4 layout is rewritten
    in memory for IF8.
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
  - Default: llama3.2_1b_if8_bin/params.bin (generated from HF model if missing).
  - --local-weights: use llama3.2_1b_if8_bin/full_model_weights.bin instead.

Usage:
  python llama3.2_1b_IF8.py
  python llama3.2_1b_IF8.py --prompt "your prompt"
  python llama3.2_1b_IF8.py --dev xdma0 [--cycle 5.042]
  python llama3.2_1b_IF8.py --local-weights
"""

import hashlib
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
from user_dma_core import DMA_DEVICE_H2C, TYPE, UE_MODE, UE_FMAX_CONTEXT_SIZE, UE_VECTOR_SIZE, URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS, set_dma_device, ue_35bit_addr_shifter, INSTRUCTION_SIZE_BYTES
from user_dma_core import UnifiedEngine
# Canonical HW-aligned codec shared across all model templates. IF8 performs
# per-block MixMSE selection between INT8 and FP8 E4M3; the bf16 scale sign
# selects the matching hardware codebook.
from quant_lib import quantize

QUANT_PRECISION = "if8"

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


def _minimal_chat_prompt(prompt: str) -> str:
    """Render the canonical Llama user/assistant headers without the dated system block."""
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _if4_to_if8_layout(cfg: dict) -> dict:
    """Rewrite the IF4 config layout in memory for one-byte IF8 data.

    Scale and bf16 regions keep their sizes. Every quantized ``*_DATA`` region
    doubles, layer offsets are repacked contiguously, and non-layer regions are
    moved after the enlarged layer block. The shared JSON file is not modified.
    """
    regions = cfg.get("regions", {})
    non_layer_regions = cfg.get("non_layer_regions", {})

    base = 0
    if regions:
        ordered = sorted(
            regions, key=lambda key: _parse_offset(regions[key]["offset"])
        )
        base = _parse_offset(regions[ordered[0]]["offset"])
        cursor = base
        for key in ordered:
            region = regions[key]
            if key.endswith("_DATA"):
                region["size"] *= 2
            region["offset"] = f"0x{cursor:08X}"
            cursor += region["size"]
        cfg["file_info"]["layer_size"] = cursor - base

    if non_layer_regions:
        cursor = (
            base
            + cfg["file_info"]["num_layers"] * cfg["file_info"]["layer_size"]
        )
        ordered = sorted(
            non_layer_regions,
            key=lambda key: _parse_offset(non_layer_regions[key]["offset"]),
        )
        for key in ordered:
            region = non_layer_regions[key]
            if key.endswith("_DATA"):
                region["size"] *= 2
            region["offset"] = f"0x{cursor:08X}"
            cursor += region["size"]

    paths = cfg.get("paths", {})
    for key in ("weights_bin", "params_meta", "instruction_bin", "instruction_meta"):
        value = paths.get(key)
        if isinstance(value, str) and value.startswith("llama3.2_1b_bin/"):
            paths[key] = (
                "llama3.2_1b_if8_bin/"
                + value[len("llama3.2_1b_bin/"):]
            )
    return cfg

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
    """Generate IF8 params.bin from HuggingFace model and the rewritten layout.
    Returns the path to the written file."""
    script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
    cfg = _load_config(script_dir)
    weight_defs = cfg["_weight_defs"]
    paths = cfg["paths"]
    paths_full = os.path.join(script_dir, paths["weights_bin"])
    out_path = output_path or paths_full

    _q = cfg["special"]["quantization"]
    block_size = _q.get("block_size", 64)

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
            (q_w, "quant"),
            (k_w, "quant"),
            (v_w, "quant"),
            (gamma_q, "bf16"),
            (gamma_k, "bf16"),
            (o_w, "quant"),
            (gamma_post, "bf16"),
            (gamma_ffn, "bf16"),
            (up_w, "quant"),
            (gate_w, "quant"),
            (down_w, "quant"),
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
            if kind == "quant":
                next_key = blk0_structure[i + 1]["key"]
                data_sz = weight_defs[f"{next_key}_SIZE"]
                data_bytes, scale_bytes = quantize(
                    QUANT_PRECISION, tensor, block_size=block_size
                )
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
    data_bytes, scale_bytes = quantize(
        QUANT_PRECISION, lm_head_w, block_size=block_size
    )
    scale_padded = (scale_bytes + b"\x00" * scale_sz)[:scale_sz]
    data_padded = (data_bytes + b"\x00" * data_sz)[:data_sz]
    write_at(weight_defs["LM_HEAD_WEIGHT_SCALE"], scale_padded)
    write_at(weight_defs["LM_HEAD_WEIGHT_DATA"], data_padded)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(buf)
    meta_path = paths.get("params_meta")
    meta_full = os.path.join(script_dir, meta_path) if meta_path else os.path.splitext(out_path)[0] + ".json"
    with open(meta_full, "w") as f:
        json.dump({"size": len(buf), "num_layers": num_layers, "layer_size": LAYER_WEIGHT_SIZE}, f, indent=2)
    print(f"Generated weights bin: {out_path} ({len(buf)} bytes)")
    print(f"Generated params meta: {meta_full}")
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
    """Load the shared config, rewrite it for IF8, and build weight definitions."""
    config_path = os.path.join(script_dir, "llama3.2_1b_config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)
    cfg = _if4_to_if8_layout(cfg)
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
class Llama32_1b_IF8_UnifiedEngine(UnifiedEngine):
    """IF8 UnifiedEngine for Llama-3.2-1B: loads config + weight bin, compile_prefill/compile_decoder, run_prefill/run_decoder.

    Key architectural differences from Gemma3:
      - No Q/K per-head norm (q_norm, k_norm): compile pipeline skips those RMS norm steps.
      - No post-attention norm: residual is applied directly to o_proj output.
      - No post-FFN norm: residual is applied directly to down_proj output.
      - post_attention_layernorm in HF is the pre-FFN norm.
      - Embedding NOT scaled by sqrt(hidden_size).
      - LM head weight tied to input embedding.
    """

    def __init__(self, script_dir: str | None = None, hf_model_dir: str | None = None, weights_bin: str | None = None,
                 decoder_matmatmul: bool = False, stream_prefill: bool = False,
                 matmatmul: bool | None = None):
        # IF8 layout inside the 2 GB window 0x80000000..0xFFFFFFFF.
        # At max_context_size=1024 the loaded IF8 params use ~1.20 GiB and
        # tensors/KV use ~212 MiB. Reserve the final 10 MiB for instructions.
        #   params : 0x80000000 .. 0xD0000000  (1.25 GiB)
        #   tensor : 0xD0000000 .. 0xFF600000  (758 MiB)
        #   program: 0xFF600000 .. 0x100000000 (10 MiB)
        super().__init__(
            params_dram_base=0x80000000,
            tensor_dram_base=0xD0000000,
            program_dram_base=0xFF600000,
        )
        self.script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
        # ``matmatmul`` is the legacy constructor keyword. It has always selected
        # only the decoder kernel; retain it as an alias without letting the old
        # ambiguous name leak into the execution logic.
        if matmatmul is not None:
            decoder_matmatmul = bool(matmatmul)
        # Decode matmul kernel select (mirrors gemma3's --decoder-matmatmul):
        #   False (default) -> quantized_matmat_core: 1-pass streaming IF8 dot.
        #   True            -> matmat_mul_core(is_B_quantized=True): dequantize B to bf16
        #                      in URAM, then bf16 dot (2 passes over the weight bytes).
        self.decoder_matmatmul = decoder_matmatmul
        # IF8 prefill is fastest through the dequantize-to-BF16 matmul on the
        # current hardware. Keep the IF4 one-pass streaming path available for
        # A/B testing, but do not make its 8-bit bandwidth penalty the default.
        self.stream_prefill = stream_prefill
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
        self.PREFILL_CONTEXT_SIZE = model["prefill_context_size"]
        self.LAYER_SIZE = fi["num_layers"]
        self.EMBEDDING_ELEMENTS = fi["embedding_vocab"]
        fixed = self._cfg.get("fixed_isa_regs", {})
        self.V_CACHE_SIZE_REG = fixed["V_CACHE_SIZE_REG"]
        self.TMP_REG = fixed["TMP_REG"]
        self.ROPE_SIZE_REG = fixed["ROPE_SIZE_REG"]
        self.gpr_bucket_idx = fixed["GPR_BUCKET_IDX_REG"]
        self.gpr_seq_len = fixed["GPR_SEQ_LEN_REG"]
        # gpr_q_seq_len / gpr_aligned_seq_len feed unified_attention_core's dynamic
        # batch / aligned_seq_len GPRs (see _compile_prefill_program / _compile_decoder_program).
        self.gpr_q_seq_len = fixed["GPR_Q_SEQ_LEN_REG"]
        self.gpr_aligned_seq_len = fixed["GPR_ALIGNED_SEQ_LEN_REG"]
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
        if self.get_params_dram_addr() > self._tensor_dram_base:
            raise MemoryError(
                f"Parameter DRAM overlaps tensor DRAM: "
                f"params_end=0x{self.get_params_dram_addr():X}, "
                f"tensor_start=0x{self._tensor_dram_base:X}"
            )
        self.tensor_init()
        if self.get_tensor_dram_addr() > self._program_dram_base:
            raise MemoryError(
                f"Tensor DRAM overlaps program DRAM: "
                f"tensor_end=0x{self.get_tensor_dram_addr():X}, "
                f"program_start=0x{self._program_dram_base:X}"
            )

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
        self.LAYER0_FLASH_ATTN_P_DRAM = self.allocate_tensor_dram(aligned_seq_len * aligned_seq_len * self.bytes_per_element)
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
        # Per-vocab additive repetition-penalty bias (on-FPGA penalty, the default). The LM-head
        # matmul reads this as its C bias (bias_mode="broadcast_N") so the on-chip argmax already
        # returns the penalized token id — no logit readback. Host maintains it with +/-alpha writes
        # (see notes_repetition_penalty_fpga_bias.md); all-zero = no penalty.
        self.PENALTY_BIAS_DRAM = self.allocate_tensor_dram(1 * self.EMBEDDING_ELEMENTS * self.bytes_per_element)

        print(f"    Allocate tensor dram end at DRAM address: 0x{self.get_tensor_dram_addr():X}, usage: {self.get_tensor_dram_usage()} bytes")

    def _compile_prefill_program(self, template_seq_len: int, layer_size: int, profile: bool = False) -> dict:
        """Compile prefill into the active capture session.

        ``template_seq_len`` is used only for FLOPs accounting and static M= args;
        all runtime loop counts are driven by ``gpr_seq_len`` / ``gpr_bucket_idx``
        primed by the caller's preamble, so a single bin works for any seq_len.
        Returns dict with ``size_bytes``, ``flops`` and (profile only) ``checkpoints``.

        When ``profile`` is True, a HALT checkpoint is emitted after every major per-layer
        step so :meth:`run_llama_profile` can measure the per-step HW latency breakdown
        (summed over all layers). Checkpoint names carry an ``L<idx>_`` prefix that the
        profiler strips before rolling the per-layer HALTs up by step type.
        """
        if not getattr(self, "is_capture_on", False):
            raise RuntimeError("_compile_prefill_program() requires an active capture session")
        count_at_start = self.capture_count
        seq_len = template_seq_len
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        checkpoints: list[list] = []

        def _checkpoint(name: str) -> None:
            self.generate_instruction_halt()
            self.pad_capture_to_64b_boundary()
            resume = self.get_program_dram_addr() + self.capture_count * INSTRUCTION_SIZE_BYTES
            checkpoints.append([name, f"0x{resume:X}"])

        global _SILENT_MODE
        _SILENT_MODE = True
        total_flops = 0
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]

        # Layer-invariant constants
        ahd         = self.actual_head_dim   # 64
        nkvh        = self.num_kv_heads      # 8
        qpkv        = self.group_size        # 4
        bpe         = self.bytes_per_element
        hd          = self.head_dim          # 512
        total_q_dim = hd * qpkv             # 2048
        half_ahd    = ahd // 2              # 32
        rope_row    = hd * 2 * bpe          # bytes per rope table row

        # The host uploads prompt embeddings to OUTPUT_DRAM. Each layer consumes
        # that recurrent buffer and writes its final residual back to the same
        # buffer, eliminating the old OUTPUT->INPUT inter-layer copy.
        layer_input_addr = self.LAYER0_OUTPUT_DRAM

        def prefill_projection_core(M: int, K: int, N: int, **kwargs) -> int:
            """Select the measured-fast IF8 prefill kernel or streaming A/B path."""
            if self.stream_prefill:
                kwargs.pop("is_B_quantized", None)
                return self.quantized_matmat_core(M=M, K=K, N=N, **kwargs)
            return self.matmat_mul_core(M=M, K=K, N=N, **kwargs)

        for layer_idx in range(layer_size):
            layer_off = layer_idx * LAYER_WEIGHT_SIZE
            total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=layer_input_addr,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off,
                              gpr_M_reg=self.gpr_seq_len)
            if profile:
                _checkpoint(f"L{layer_idx}_pre_norm")

            total_flops += prefill_projection_core(M=seq_len, K=self.vector_length, N=self.head_dim * self.group_size,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                is_B_quantized=True, SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off, data_type=TYPE.IF8,
                gpr_M_reg=self.gpr_seq_len)
            total_flops += prefill_projection_core(M=seq_len, K=self.vector_length, N=self.head_dim,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                is_B_quantized=True, SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off, data_type=TYPE.IF8,
                gpr_M_reg=self.gpr_seq_len)
            # v_proj writes to interleaved temp (T, 512); per-head KV cache populated below.
            total_flops += prefill_projection_core(M=seq_len, K=self.vector_length, N=self.head_dim,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_V_PROJ_TEMP,
                is_B_quantized=True, SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off, data_type=TYPE.IF8,
                gpr_M_reg=self.gpr_seq_len)
            if profile:
                _checkpoint(f"L{layer_idx}_qkv_proj")

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

            # Phase 1: K rope in-place on LAYER0_K_DRAM (N=512, [lo|hi] layout).
            # K layout = [seq_len, hd]. PBI: outer M loop driven by gpr_seq_len.
            total_flops += self.rope_hf_core_dram(
                M=seq_len,
                N=hd,
                input_dram_addr=self.LAYER0_K_DRAM,
                output_dram_addr=self.LAYER0_K_DRAM,
                cos_dram_addr=ROPE_WEIGHT_ADDR,
                sin_dram_addr=ROPE_WEIGHT_ADDR + hd * bpe,
                gpr_M_reg=self.gpr_seq_len,
            )

            # Phase 2: Q rope in-place (N=512, [lo|hi] layout).
            # Q layout = [seq_len, qpkv=4, hd] — 4 sub-rows per token share one cos/sin.
            # PBI: outer M loop driven by gpr_seq_len; inner group loop is static.
            total_flops += self.rope_hf_core_dram_gqa(
                M=seq_len,
                group_size=qpkv,
                N=hd,
                input_dram_addr=self.LAYER0_Q_DRAM,
                output_dram_addr=self.LAYER0_Q_DRAM,
                cos_dram_addr=ROPE_WEIGHT_ADDR,
                sin_dram_addr=ROPE_WEIGHT_ADDR + hd * bpe,
                gpr_M_reg=self.gpr_seq_len,
            )
            if profile:
                _checkpoint(f"L{layer_idx}_rope")

            # Scale the full post-RoPE query tensor once. Otherwise each of the
            # eight per-KV-head attention calls launches its own scaling kernel.
            total_flops += self.eltwise_core_dram(
                M=seq_len,
                N=total_q_dim,
                dram_a=self.LAYER0_Q_DRAM,
                dram_b=None,
                dram_out=self.LAYER0_Q_DRAM,
                mode=UE_MODE.MUL_BROADCAST,
                scalar=1.0 / math.sqrt(ahd),
                gpr_M_reg=self.gpr_seq_len,
            )

            # Phase 3: Per-KV-head scatter → KV cache + flash buffers → flash_attention
            for kv_h in range(nkvh):
                k_cache_base = (self.LAYER0_K_ROPE_DRAM
                                + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)
                v_cache_base = (self.LAYER0_V_DRAM
                                + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)

                # Scatter K_h_roped (64-dim) from [lo|hi] K_DRAM → KV cache + FLASH_K
                # lo half and hi half are non-contiguous in K_DRAM → two PBI loops.
                k_flash_lo_specs = (
                    [(k_cache_base,               ahd * bpe)]
                    + [(self.LAYER0_FLASH_K_DRAM + g * ahd * bpe,               qpkv * ahd * bpe) for g in range(qpkv)]
                )
                k_flash_hi_specs = (
                    [(k_cache_base + half_ahd * bpe, ahd * bpe)]
                    + [(self.LAYER0_FLASH_K_DRAM + g * ahd * bpe + half_ahd * bpe, qpkv * ahd * bpe) for g in range(qpkv)]
                )
                self._emit_pbi_scatter_per_token(
                    read_base=self.LAYER0_K_DRAM + kv_h * half_ahd * bpe,
                    read_stride_bytes=hd * bpe,
                    write_specs=k_flash_lo_specs,
                    sram_byte_addr=0x10000,
                    element_count=half_ahd,
                    gpr_seq_len=self.gpr_seq_len,
                    template_seq_len=seq_len,
                )
                self._emit_pbi_scatter_per_token(
                    read_base=self.LAYER0_K_DRAM + (hd // 2 + kv_h * half_ahd) * bpe,
                    read_stride_bytes=hd * bpe,
                    write_specs=k_flash_hi_specs,
                    sram_byte_addr=0x10080,
                    element_count=half_ahd,
                    gpr_seq_len=self.gpr_seq_len,
                    template_seq_len=seq_len,
                )

                # Scatter V_h (64-dim, standard layout) from V_PROJ_TEMP → KV cache + FLASH_V
                # V_PROJ_TEMP layout: [seq_len, nkvh, ahd] → stride = nkvh*ahd*bpe per token.
                self._emit_pbi_scatter_per_token(
                    read_base=self.LAYER0_V_PROJ_TEMP + kv_h * ahd * bpe,
                    read_stride_bytes=nkvh * ahd * bpe,
                    write_specs=(
                        [(v_cache_base, ahd * bpe)]
                        + [(self.LAYER0_FLASH_V_DRAM + g * ahd * bpe, qpkv * ahd * bpe) for g in range(qpkv)]
                    ),
                    sram_byte_addr=0x20000,
                    element_count=ahd,
                    gpr_seq_len=self.gpr_seq_len,
                    template_seq_len=seq_len,
                )

                # Scatter Q_h_q (64-dim) from [lo|hi] Q_DRAM → FLASH_Q
                # Q group g_for_kv = kv_h//2; sub-heads 0..qpkv-1 within that group.
                g_for_kv = kv_h // 2
                local_kv = kv_h % 2
                for q in range(qpkv):
                    sub_idx = local_kv * qpkv + q
                    q_lo_base = (self.LAYER0_Q_DRAM
                                 + g_for_kv * hd * bpe
                                 + sub_idx * half_ahd * bpe)
                    q_hi_base = (self.LAYER0_Q_DRAM
                                 + g_for_kv * hd * bpe
                                 + (hd // 2 + sub_idx * half_ahd) * bpe)
                    flash_q_lo = self.LAYER0_FLASH_Q_DRAM + q * ahd * bpe
                    flash_q_hi = flash_q_lo + half_ahd * bpe
                    self._emit_pbi_scatter_per_token(
                        read_base=q_lo_base,
                        read_stride_bytes=total_q_dim * bpe,
                        write_specs=[(flash_q_lo, qpkv * ahd * bpe)],
                        sram_byte_addr=0x30000,
                        element_count=half_ahd,
                        gpr_seq_len=self.gpr_seq_len,
                        template_seq_len=seq_len,
                    )
                    self._emit_pbi_scatter_per_token(
                        read_base=q_hi_base,
                        read_stride_bytes=total_q_dim * bpe,
                        write_specs=[(flash_q_hi, qpkv * ahd * bpe)],
                        sram_byte_addr=0x30080,
                        element_count=half_ahd,
                        gpr_seq_len=self.gpr_seq_len,
                        template_seq_len=seq_len,
                    )

                # Scaled dot-product attention for this KV head's qpkv query heads,
                # called inline per KV head (unified_attention_core replaces the old
                # shared flash_attention_core subroutine + JUMP_ABS call-site pattern).
                attn_result = self.unified_attention_core(
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
                    gpr_batch_reg=self.gpr_q_seq_len,
                    gpr_aligned_seq_len_reg=self.gpr_aligned_seq_len,
                    q_pre_scaled=True,
                )
                total_flops += attn_result or 0

                # Assemble output into LAYER0_FLASH_OUTPUT_DRAM
                # Standard GQA layout per token: [kv0_q0(64), kv0_q1..q3, kv1_q0..q3, ...]
                out_h_base = kv_h * qpkv * ahd * bpe
                t_reg = self.alloc_isa_reg()
                src_off_reg = self.alloc_isa_reg()
                dst_off_reg = self.alloc_isa_reg()
                self.generate_instruction_add_set(t_reg, 0)
                self.loop_start(loop_cnt=seq_len, gpr_loop_cnt=self.gpr_seq_len)
                self.generate_instruction_reg_mul_imm(src_off_reg, t_reg, ue_35bit_addr_shifter(qpkv * ahd * bpe))
                self.generate_instruction_reg_mul_imm(dst_off_reg, t_reg, ue_35bit_addr_shifter(total_q_dim * bpe))
                for g in range(qpkv):
                    self.generate_instruction_add_imm(src_off_reg, ue_35bit_addr_shifter(self.LAYER0_FLASH_OUT_HEAD_DRAM + g * ahd * bpe), self.TMP_REG)
                    self.accelerator_memory_to_sram(0, 0x40000, ahd, general_reg_src=self.TMP_REG)
                    self.generate_instruction_add_imm(dst_off_reg, ue_35bit_addr_shifter(self.LAYER0_FLASH_OUTPUT_DRAM + out_h_base + g * ahd * bpe), self.TMP_REG)
                    self.sram_to_accelerator_memory(0x40000, 0, ahd, general_reg_src=self.TMP_REG)
                self.generate_instruction_add_inc(t_reg)
                self.loop_end()
                self.release_isa_reg()  # dst_off_reg
                self.release_isa_reg()  # src_off_reg
                self.release_isa_reg()  # t_reg
            if profile:
                _checkpoint(f"L{layer_idx}_attention")

            total_flops += prefill_projection_core(M=seq_len, K=self.head_dim * self.group_size, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                is_B_quantized=True, SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off, data_type=TYPE.IF8,
                gpr_M_reg=self.gpr_seq_len)

            # LLaMA: no post-attention norm; add residual directly to o_proj output.
            # Per-row PBI loop (gpr_seq_len trips) — no URAM cap on prefill length.
            self.eltwise_core_dram(
                M=seq_len, N=self.vector_length,
                dram_a=layer_input_addr,
                dram_b=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                dram_out=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                mode=UE_MODE.ELTWISE_ADD,
                gpr_M_reg=self.gpr_seq_len,
            )
            if profile:
                _checkpoint(f"L{layer_idx}_o_proj_residual")

            # LLaMA: post_attention_layernorm IS the pre-FFN norm
            total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off,
                              gpr_M_reg=self.gpr_seq_len)
            if profile:
                _checkpoint(f"L{layer_idx}_pre_ffn_norm")
            total_flops += prefill_projection_core(M=seq_len, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                is_B_quantized=True, SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off, data_type=TYPE.IF8, silu_enable=True,
                gpr_M_reg=self.gpr_seq_len)
            total_flops += prefill_projection_core(M=seq_len, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                is_B_quantized=True, SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off, data_type=TYPE.IF8,
                gpr_M_reg=self.gpr_seq_len)
            # gate × up — per-row PBI loop (gpr_seq_len trips); one row of mlp_elements per iter.
            self.eltwise_core_dram(
                M=seq_len, N=self.mlp_elements,
                dram_a=self.LAYER0_MLP_GATE_DRAM,
                dram_b=self.LAYER0_MLP_UP_DRAM,
                dram_out=self.LAYER0_MLP_MULT_DRAM,
                mode=UE_MODE.ELTWISE_MUL,
                gpr_M_reg=self.gpr_seq_len,
            )
            if profile:
                _checkpoint(f"L{layer_idx}_mlp_gateup_mul")
            total_flops += prefill_projection_core(M=seq_len, K=self.mlp_elements, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                is_B_quantized=True, SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off, data_type=TYPE.IF8,
                gpr_M_reg=self.gpr_seq_len)

            # LLaMA: no post-FFN norm; add residual directly to down_proj output.
            # Per-row PBI loop (gpr_seq_len trips) — layer_output = POST_ATTN_RESIDUAL + MLP_DOWN.
            self.eltwise_core_dram(
                M=seq_len, N=self.vector_length,
                dram_a=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                dram_b=self.LAYER0_MLP_DOWN_DRAM,
                dram_out=self.LAYER0_OUTPUT_DRAM,
                mode=UE_MODE.ELTWISE_ADD,
                gpr_M_reg=self.gpr_seq_len,
            )
            if profile:
                _checkpoint(f"L{layer_idx}_mlp_down_residual")
        self.generate_instruction_halt()
        prefill_program_size = (self.capture_count - count_at_start) * INSTRUCTION_SIZE_BYTES
        _SILENT_MODE = False
        return {"size_bytes": prefill_program_size, "flops": total_flops, "checkpoints": checkpoints}

    def _compile_decoder_program(self, layer_size: int, profile: bool = False) -> dict:
        """Compile decoder into the active capture session.
        Returns dict with ``program_size_bytes``, ``total_flops`` and (profile only) ``checkpoints``.

        When ``profile`` is True, a HALT checkpoint is emitted after every major per-layer
        step so :meth:`run_llama_profile` can measure the per-step HW latency breakdown
        (summed over all layers). The once-per-token final norm + LM head after the layer
        loop is drained as the trailing ``output_norm_lm_head`` segment by the profiler.
        """
        if not getattr(self, "is_capture_on", False):
            raise RuntimeError("_compile_decoder_program() requires an active capture session")
        count_at_start = self.capture_count
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        total_flops = 0
        decoder_aligned_seq_len = ((self.MAX_CONTEXT_SIZE + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
        checkpoints: list[list] = []

        def _checkpoint(name: str) -> None:
            self.generate_instruction_halt()
            self.pad_capture_to_64b_boundary()
            resume = self.get_program_dram_addr() + self.capture_count * INSTRUCTION_SIZE_BYTES
            checkpoints.append([name, f"0x{resume:X}"])

        def decoder_projection_core(K: int, N: int, force_stream: bool = False, **kwargs) -> int:
            """Select the streaming or compatibility decoder projection kernel.

            Both paths read the SAME IF8 weight; they differ in how the dot is computed:

            - default (``--decoder-matmatmul`` off) -> :meth:`quantized_matmat_core`: 1-pass streaming
              quantized dot (inline if8->bf19 unpack straight through the DOT_PRODUCT unit).
            - ``--decoder-matmatmul`` -> :meth:`matmat_mul_core` with ``is_B_quantized=True``:
              DEQUANTIZES B to bf16 in URAM, then a bf16 dot — two passes over the weight
              bytes, so ~2x the DRAM traffic (and ~2x slower) for higher-precision accumulate.

            ``force_stream`` pins an individual call to the streaming path regardless of the flag.
            """
            if not self.decoder_matmatmul or force_stream:
                kwargs.pop("is_B_quantized", None)
                return self.quantized_matmat_core(M=1, K=K, N=N, **kwargs)
            m_reg = self.alloc_isa_reg()
            self.generate_instruction_add_set(m_reg, 1)
            flops = self.matmat_mul_core(M=1, K=K, N=N, gpr_M_reg=m_reg, **kwargs)
            self.release_isa_reg()
            return flops

        global _SILENT_MODE
        _SILENT_MODE = True
        for layer_idx in range(layer_size):
                layer_off = layer_idx * LAYER_WEIGHT_SIZE
                layer_input_addr = self.LAYER0_INPUT_DRAM if layer_idx == 0 else self.LAYER0_OUTPUT_DRAM
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=layer_input_addr,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off)
                if profile:
                    _checkpoint(f"L{layer_idx}_pre_norm")
                total_flops += decoder_projection_core(K=self.vector_length, N=self.head_dim * self.group_size,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off, data_type=TYPE.IF8, is_B_quantized=True)
                total_flops += decoder_projection_core(K=self.vector_length, N=self.head_dim,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off, data_type=TYPE.IF8, is_B_quantized=True)
                total_flops += decoder_projection_core(K=self.vector_length, N=self.head_dim,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off, data_type=TYPE.IF8, is_B_quantized=True)
                if profile:
                    _checkpoint(f"L{layer_idx}_qkv_proj")

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
                total_flops += self.rope_hf_core_decode(
                    N=hd,
                    input_dram_addr=self.LAYER0_K_DRAM,
                    output_dram_addr=self.LAYER0_K_DRAM,
                    cos_dram_addr=ROPE_WEIGHT_ADDR,
                    sin_dram_addr=ROPE_WEIGHT_ADDR + hd * bpe,
                    rope_size_reg=self.ROPE_SIZE_REG,
                    tmp_reg=self.TMP_REG)

                # Step 2: Q rope in-place per 512-dim group (N=512, uses ROPE_SIZE_REG)
                for g in range(qpkv):
                    total_flops += self.rope_hf_core_decode(
                        N=hd,
                        input_dram_addr=self.LAYER0_Q_DRAM + g * hd * bpe,
                        output_dram_addr=self.LAYER0_Q_DRAM + g * hd * bpe,
                        cos_dram_addr=ROPE_WEIGHT_ADDR,
                        sin_dram_addr=ROPE_WEIGHT_ADDR + hd * bpe,
                        rope_size_reg=self.ROPE_SIZE_REG,
                        tmp_reg=self.TMP_REG)
                if profile:
                    _checkpoint(f"L{layer_idx}_rope")

                # Apply 1/sqrt(head_dim) once to the contiguous post-RoPE Q vector.
                # Each of the eight per-KV-head attention calls can then consume its
                # gathered query group directly instead of repeating this multiply.
                total_flops += self.eltwise_core_dram(
                    M=1, N=total_q_dim, dram_a=self.LAYER0_Q_DRAM, dram_b=None,
                    dram_out=self.LAYER0_Q_DRAM, mode=UE_MODE.MUL_BROADCAST,
                    scalar=1.0 / math.sqrt(ahd))

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
                    self.sram_to_accelerator_memory(0x10000, 0, half_ahd, general_reg_src=self.TMP_REG)
                    self.generate_instruction_add_imm(
                        self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter(k_cache_base + half_ahd * bpe), self.TMP_REG)
                    self.sram_to_accelerator_memory(0x10080, 0, half_ahd, general_reg_src=self.TMP_REG)

                    # Scatter V_h (64-dim, standard layout) → V cache at decode position
                    # v_proj output at LAYER0_FLASH_V_DRAM: [V_KV0(64)..V_KV7(64)] = 512-dim
                    self.accelerator_memory_to_sram(
                        self.LAYER0_FLASH_V_DRAM + kv_h * ahd * bpe, 0x20000, ahd)
                    self.generate_instruction_add_imm(
                        self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter(v_cache_base), self.TMP_REG)
                    self.sram_to_accelerator_memory(0x20000, 0, ahd, general_reg_src=self.TMP_REG)

                    # Scatter Q_h_q (64-dim) from [lo|hi] Q_DRAM → FLASH_Q base (no kv_h offset)
                    # KV head kv_h → Q group g = kv_h//2; sub_idx = (kv_h%2)*qpkv + q
                    g_for_kv = kv_h // 2
                    local_kv = kv_h % 2
                    q_g_addr = self.LAYER0_Q_DRAM + g_for_kv * hd * bpe
                    for q in range(qpkv):
                        sub_idx = local_kv * qpkv + q
                        flash_q_addr = self.LAYER0_FLASH_Q_DRAM + q * ahd * bpe
                        self.accelerator_memory_to_sram(
                            q_g_addr + sub_idx * half_ahd * bpe, 0x30000, half_ahd)
                        self.accelerator_memory_to_sram(
                            q_g_addr + (hd // 2 + sub_idx * half_ahd) * bpe,
                            0x30080, half_ahd)
                        self.sram_to_accelerator_memory(0x30000, flash_q_addr, half_ahd)
                        self.sram_to_accelerator_memory(0x30080, flash_q_addr + half_ahd * bpe, half_ahd)
                    # Each head's K/V cache is already laid out as
                    # [MAX_CONTEXT_SIZE, actual_head_dim]. Pass its base straight
                    # to the read-only attention core; staging the aligned prefix
                    # copied the same contiguous history twice per KV head, per
                    # layer, per token.
                    attn_result = self.unified_attention_core(
                        batch=qpkv,
                        aligned_seq_len=decoder_aligned_seq_len,
                        head_dim=ahd,
                        Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
                        K_DRAM_ADDR=k_cache_base,
                        V_DRAM_ADDR=v_cache_base,
                        BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_DRAM,
                        OUTPUT_DRAM_ADDR=(self.LAYER0_FLASH_OUTPUT_DRAM
                                          + kv_h * qpkv * ahd * bpe),
                        SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                        IDENTITY_DRAM_ADDR=self.IDENTITY_DRAM_ADDR,
                        gpr_aligned_seq_len_reg=self.gpr_aligned_seq_len,
                        q_pre_scaled=True,
                    )
                    total_flops += attn_result or 0
                if profile:
                    _checkpoint(f"L{layer_idx}_attention")
                total_flops += decoder_projection_core(K=self.head_dim * self.group_size, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off, data_type=TYPE.IF8, is_B_quantized=True)

                # LLaMA: no post-attention norm; residual directly on o_proj output
                self.accelerator_memory_to_sram(accelerator_dram_address=layer_input_addr, sram_address=0x10000, element_size=self.vector_length)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM, sram_address=0x90000, element_size=self.vector_length)
                self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, element_size=self.vector_length)
                if profile:
                    _checkpoint(f"L{layer_idx}_o_proj_residual")

                # LLaMA: post_attention_layernorm IS the pre-FFN norm
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off)
                if profile:
                    _checkpoint(f"L{layer_idx}_pre_ffn_norm")

                total_flops += decoder_projection_core(K=self.vector_length, N=self.mlp_elements,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off, data_type=TYPE.IF8, silu_enable=True, is_B_quantized=True)
                total_flops += decoder_projection_core(K=self.vector_length, N=self.mlp_elements,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off, data_type=TYPE.IF8, is_B_quantized=True)

                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_GATE_DRAM, sram_address=0x10000, element_size=self.mlp_elements)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_UP_DRAM, sram_address=0x90000, element_size=self.mlp_elements)
                self.eltwise_mul_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.mlp_elements)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_MLP_MULT_DRAM, element_size=self.mlp_elements)
                if profile:
                    _checkpoint(f"L{layer_idx}_mlp_gateup_mul")

                total_flops += decoder_projection_core(K=self.mlp_elements, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off, data_type=TYPE.IF8, is_B_quantized=True)

                # LLaMA: no post-FFN norm; residual directly on down_proj output
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, sram_address=0x10000, element_size=self.vector_length)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_DOWN_DRAM, sram_address=0x90000, element_size=self.vector_length)
                self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, element_size=self.vector_length)
                if profile:
                    _checkpoint(f"L{layer_idx}_mlp_down_residual")

        if layer_size == self.LAYER_SIZE:
            total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_OUTPUT_DRAM,
                OUTPUT_DRAM_ADDR=self.OUTPUT_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_OUTPUT_NORM_GAMMA)
            penalty_kwargs = dict(C_DRAM_ADDR=self.PENALTY_BIAS_DRAM, bias_mode="broadcast_N") \
                if bool(getattr(self, "fpga_penalty", False)) else {}
            # LM head uses the same dual-kernel dispatch as the layer matmuls (mirrors gemma3):
            # streaming IF8 by default, dequantize-to-bf16 dot under
            # --decoder-matmatmul. Both kernels
            # honor the folded repetition-penalty bias (broadcast_N) and the argmax-only
            # write_back_disable, so the HW argmax still returns the penalized token id.
            total_flops += decoder_projection_core(K=self.vector_length, N=self.EMBEDDING_ELEMENTS,
                A_DRAM_ADDR=self.OUTPUT_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_QUANT, OUTPUT_DRAM_ADDR=self.LOGITS_DRAM,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_SCALE, data_type=TYPE.IF8, is_B_quantized=True,
                write_back_disable=True, **penalty_kwargs)

        self.generate_instruction_halt()
        decoder_program_size = (self.capture_count - count_at_start) * INSTRUCTION_SIZE_BYTES
        _SILENT_MODE = False
        return {"program_size_bytes": decoder_program_size, "total_flops": total_flops, "checkpoints": checkpoints}

    def _instruction_paths(self) -> tuple[str, str]:
        """(bin, meta) cache paths for the current compile mode (mirrors gemma3's tagging).

        The captured image depends on the decode matmul kernel, prefill matmul
        kernel, and on-FPGA penalty mode, so every combination gets its own cache
        entry instead of silently reusing a binary compiled for another mode.
        """
        paths_cfg = self._cfg.get("paths", {})
        bin_rel = paths_cfg.get("instruction_bin", "llama3.2_1b_if8_bin/programs.bin")
        meta_rel = paths_cfg.get("instruction_meta", "llama3.2_1b_if8_bin/programs.json")
        tag = ("_decoder_matmatmul" if self.decoder_matmatmul else "")
        tag += ("_streamprefill" if self.stream_prefill else "")
        tag += ("" if bool(getattr(self, "fpga_penalty", False)) else "_puregreedy")
        if tag:
            b_root, b_ext = os.path.splitext(bin_rel)
            m_root, m_ext = os.path.splitext(meta_rel)
            bin_rel, meta_rel = f"{b_root}{tag}{b_ext}", f"{m_root}{tag}{m_ext}"
        return (os.path.join(self.script_dir, bin_rel), os.path.join(self.script_dir, meta_rel))

    def _instruction_compiler_fingerprint(self, layer_size: int) -> str:
        """Hash every local input that can change the captured instruction stream."""
        digest = hashlib.sha256()
        config_path = os.path.join(self.script_dir, "llama3.2_1b_config.json")
        for source_path in (__file__, user_dma_core.__file__, config_path):
            digest.update(os.path.abspath(source_path).encode())
            with open(source_path, "rb") as source_file:
                digest.update(source_file.read())
        digest.update(
            f"layers={layer_size};decoder_matmatmul={self.decoder_matmatmul};"
            f"stream_prefill={self.stream_prefill};"
            f"penalty={getattr(self, 'fpga_penalty', False)}".encode())
        return digest.hexdigest()

    def compile_llama(self, layer_size: int | None = None, profile: bool = False) -> None:
        """Compile prefill + decoder into a single combined instruction image.

        Layout in program DRAM:  [prefill][decoder]

        Prefill is compiled with a fixed template (UE_VECTOR_SIZE) — all runtime
        loop counts are driven by GPRs primed by the preamble, so the same bin
        works for any seq_len. A matching compiler fingerprint makes this a no-op.

        When ``profile`` is True, both programs are compiled with per-step HALT
        checkpoints into a separate profile image (so the normal bin is never
        overwritten with checkpoint HALTs). The checkpoint resume-address lists are
        stored in the meta for :meth:`run_llama_profile`.

        Writes:
          - paths.instruction_bin  : combined raw instruction stream
          - paths.instruction_meta : per-stage start addresses, sizes, FLOPs
        """
        if layer_size is None:
            layer_size = self.LAYER_SIZE
        if profile:
            instruction_bin_path = os.path.join(self.script_dir, "llama3.2_1b_if8_bin/llama3.2_1b_profile_program.bin")
            instruction_meta_path = os.path.join(self.script_dir, "llama3.2_1b_if8_bin/llama3.2_1b_profile_program.json")
        else:
            instruction_bin_path, instruction_meta_path = self._instruction_paths()
        self._instruction_bin_path = instruction_bin_path
        self._instruction_meta_path = instruction_meta_path
        compiler_fingerprint = self._instruction_compiler_fingerprint(layer_size)
        print(f"Decode matmul kernel: {'matmat_mul_core (dequantize->bf16 dot, 2-pass)' if self.decoder_matmatmul else 'quantized_matmat_core (streaming IF8, 1-pass)'}")
        print(f"Prefill matmul kernel: {'quantized_matmat_core (streaming IF8, 1-pass A/B)' if self.stream_prefill else 'matmat_mul_core (dequantize->bf16 dot, measured-fast IF8 default)'}")
        if os.path.exists(instruction_bin_path) and os.path.exists(instruction_meta_path):
            try:
                with open(instruction_meta_path, "r") as meta_file:
                    cached_meta = json.load(meta_file)
                cached_bin_size = os.path.getsize(instruction_bin_path)
            except (OSError, ValueError, TypeError):
                cached_meta = {}
                cached_bin_size = -1
            if (cached_meta.get("compiler_fingerprint") == compiler_fingerprint
                    and cached_meta.get("instruction_total_size") == cached_bin_size):
                print(f"Reusing validated instruction image at {instruction_bin_path}")
                return
            print(f"Rebuilding stale instruction image at {instruction_bin_path}")

        # Compile the prefill template at PREFILL_CONTEXT_SIZE (the max prompt length).
        # All loop counts are GPR-driven (gpr_seq_len), so the single cached program
        # handles any actual prompt length <= PREFILL_CONTEXT_SIZE; template_seq_len is
        # used only for FLOPs accounting and the flash bucket count.
        template_seq_len = self.PREFILL_CONTEXT_SIZE

        self.clear_inst_id()
        self.start_capture()

        print(f"Compiling prefill (template_seq_len={template_seq_len}; profile={profile})...")
        t0 = time.perf_counter()
        prefill_prog = self._compile_prefill_program(template_seq_len=template_seq_len, layer_size=layer_size, profile=profile)
        print(f"  prefill compiled: {prefill_prog['size_bytes']} bytes, {time.perf_counter() - t0:.1f}s")

        print("Compiling decoder...")
        t0 = time.perf_counter()
        decoder_prog = self._compile_decoder_program(layer_size=layer_size, profile=profile)
        print(f"  decoder compiled: {decoder_prog['program_size_bytes']} bytes, {time.perf_counter() - t0:.1f}s")

        self.stop_capture()

        os.makedirs(os.path.dirname(instruction_bin_path), exist_ok=True)
        instruction_bytes = bytearray()
        for inst in self.capture_buffer:
            instruction_bytes.extend(inst.get_bytes())
        if len(instruction_bytes) > 10 * 1024 * 1024 - 64 * 1024:
            raise MemoryError(
                f"Instruction image exceeds 10 MiB program region: "
                f"{len(instruction_bytes)} bytes"
            )
        with open(instruction_bin_path, "wb") as f:
            f.write(instruction_bytes)
        self.clear_capture_buffer()

        instruction_base_addr = self.get_program_dram_addr()
        prefill_program_addr = instruction_base_addr
        decoder_program_addr = instruction_base_addr + prefill_prog["size_bytes"]

        metadata = {
            "instruction_bin": os.path.relpath(instruction_bin_path, self.script_dir),
            "compiler_fingerprint": compiler_fingerprint,
            "instruction_base_addr": f"0x{instruction_base_addr:X}",
            "instruction_total_size": len(instruction_bytes),
            "prefill_template_seq_len": template_seq_len,
            "prefill_program_start_addr": f"0x{prefill_program_addr:X}",
            "prefill_program_size": prefill_prog["size_bytes"],
            "prefill_template_flops": prefill_prog["flops"],
            "decoder_program_start_addr": f"0x{decoder_program_addr:X}",
            "decoder_program_size": decoder_prog["program_size_bytes"],
            "decoder_total_flops": decoder_prog["total_flops"],
        }
        if profile:
            metadata["prefill_profile_checkpoints"] = prefill_prog["checkpoints"]
            metadata["decoder_profile_checkpoints"] = decoder_prog["checkpoints"]
        with open(instruction_meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Combined instruction image written to {instruction_bin_path} ({len(instruction_bytes)} bytes)")
        print(f"Metadata written to {instruction_meta_path}")

    def _structural_token_ids(self) -> set:
        """Token ids that must NEVER be repetition-penalized: punctuation, whitespace,
        newline, and special tokens. Precomputed once from the tokenizer vocab and cached.

        These 'glue' tokens recur constantly in any text; penalizing them over a long
        generation is what starves a small model of grammatical structure and produces
        word-salad. Exempting them lets the repetition penalty target only content tokens.
        """
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
        """1-D LongTensor of the structural/special token ids (cached) for vectorized
        exemption in the repetition penalty."""
        t = getattr(self, "_struct_ids_tensor_cache", None)
        if t is None:
            t = torch.tensor(sorted(self._structural_token_ids()), dtype=torch.long)
            self._struct_ids_tensor_cache = t
        return t

    def _write_penalty_bias(self, prev_tokens) -> None:
        """On-FPGA repetition penalty (the default): build the per-vocab additive bias from the
        windowed token frequency and DMA it to PENALTY_BIAS_DRAM (the LM-head matmul's C term,
        bias_mode="broadcast_N"). bias[t] = clamp(−alpha·count[t], min=−cap); structural tokens stay
        0. The HW argmax of (logits + bias) then returns the penalized token id — no logit readback.

        A single full-buffer DMA per step (one device open/write). Exactly matches the SW golden
        reference compare/compare_llama3.2_1b_penalty.py. Incremental ±alpha chunk writes were tried and measured both
        SLOWER (2 per-step DMAs each pay os.open/os.close, which dominates the tiny transfer) and
        lower quality (count-from-gate) — so the full rewrite is the production path.
        """
        vocab = self.EMBEDDING_ELEMENTS
        alpha = float(getattr(self, "pen_alpha", 1.0))
        cap = float(getattr(self, "pen_cap", 20.0))
        W = int(getattr(self, "rep_window", 256))
        window = prev_tokens[-W:]  # last W tokens
        count = torch.zeros(vocab, dtype=torch.float32)
        if window:
            win = torch.tensor(window, dtype=torch.long)
            count.index_add_(0, win, torch.ones(win.numel(), dtype=torch.float32))  # frequency of each token id
            count[self._structural_ids_tensor()] = 0.0  # never penalize punctuation/whitespace/specials
        bias = (-alpha * count).clamp(min=-cap).to(torch.bfloat16).view(1, vocab)  # bias[t] = clamp(−α · count[t], min = −cap)
        self.dma_to_accelerator_memory(self.PENALTY_BIAS_DRAM, bias)  # push to buffer

    def run_llama(self) -> None:
        """Load the unified instruction image and run prefill + decoder loop.

        Primes GPRs via a small captured preamble that jumps into the cached
        prefill or decoder program at runtime.
        """
        # Mode-specific bin/meta (decode matmul kernel + penalty) that compile_llama produced.
        meta_path = getattr(self, "_instruction_meta_path", None) or self._instruction_paths()[1]
        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.load_program_instructions_from_file(os.path.join(self.script_dir, meta["instruction_bin"]))
        preamble_addr = self.get_program_dram_addr()
        if preamble_addr + 64 * 1024 > 0x100000000:
            raise MemoryError(
                f"Instruction image exceeds 10 MiB program region: preamble=0x{preamble_addr:X}"
            )

        prefill_program_addr = int(meta["prefill_program_start_addr"], 16)
        decoder_program_addr = int(meta["decoder_program_start_addr"], 16)
        template_seq_len = int(meta["prefill_template_seq_len"])
        flops_prefill_template = meta["prefill_template_flops"]
        decoder_flops_per_token = meta["decoder_total_flops"]
        _max_gpr_bucket = (self.MAX_CONTEXT_SIZE + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
        _kv_stride = self.actual_head_dim * self.bytes_per_element
        _rope_row  = self.head_dim * 2 * self.bytes_per_element

        prefill_seq = self.prefill_seq
        if prefill_seq is None:
            prefill_seq = tuple(self._cfg["default_prefill_tokens"])
        if len(prefill_seq) < 2:
            raise ValueError("Prefill sequence must have at least 2 tokens.")
        prefill_seq = prefill_seq[:-1]  # last token starts the decoder
        prefill_seq_len = len(prefill_seq)
        self.seq_len = prefill_seq_len

        q_seq_len = prefill_seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        bucket_idx = aligned_seq_len // UE_VECTOR_SIZE
        flops_prefill = flops_prefill_template * prefill_seq_len // max(template_seq_len, 1)

        # Prefill preamble: prime gpr_seq_len + gpr_bucket_idx (+ gpr_q_seq_len / gpr_aligned_seq_len
        # for unified_attention_core's dynamic batch / aligned_seq_len), then jump into cached prefill.
        self.clear_inst_id()
        self.start_capture()
        self.generate_instruction_add_set(self.gpr_seq_len, prefill_seq_len)
        self.generate_instruction_add_set(self.gpr_bucket_idx, bucket_idx)
        self.generate_instruction_add_set(self.gpr_q_seq_len, q_seq_len)
        self.generate_instruction_add_set(self.gpr_aligned_seq_len, aligned_seq_len)
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(prefill_program_addr))
        self.stop_capture()
        self.write_captured_instructions_to_dram(preamble_addr)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        self.clear_capture_buffer()

        # Build every position-dependent decoder preamble once. Each 64-byte-aligned
        # entry is six instructions (four register sets, jump, unreachable padding NOP),
        # so the hot loop only selects an address instead of compiling and DMA-writing it.
        decoder_dispatch_addr = self.get_program_dram_addr()
        decoder_dispatch_stride = 6 * INSTRUCTION_SIZE_BYTES
        self.clear_inst_id()
        self.start_capture()
        for decode_pos in range(prefill_seq_len, self.MAX_CONTEXT_SIZE):
            runtime_seq_len = decode_pos + 1
            runtime_aligned_seq_len = ((runtime_seq_len + 63) // 64) * 64
            runtime_bucket_idx = min((runtime_seq_len + 63) // 64, _max_gpr_bucket)
            entry_start = self.capture_count
            self.clear_inst_id()
            self.generate_instruction_add_set(self.gpr_bucket_idx, runtime_bucket_idx)
            self.generate_instruction_add_set(self.gpr_aligned_seq_len, runtime_aligned_seq_len)
            self.generate_instruction_add_set(self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter(decode_pos * _kv_stride))
            self.generate_instruction_add_set(self.ROPE_SIZE_REG, ue_35bit_addr_shifter(decode_pos * _rope_row))
            self.generate_instruction_jump_abs(ue_35bit_addr_shifter(decoder_program_addr))
            self.generate_instruction_nop()
            entry_size = self.capture_count - entry_start
            assert entry_size == 6, (
                f"decoder dispatch entry at position {decode_pos} has {entry_size} instructions"
            )
        self.stop_capture()
        decoder_dispatch_size = self.get_capture_instruction_size_bytes()
        dispatch_bytes_written = self.write_captured_instructions_to_dram(decoder_dispatch_addr)
        if dispatch_bytes_written != decoder_dispatch_size:
            raise RuntimeError("Failed to write the complete decoder dispatch table")
        self.allocate_program_dram(decoder_dispatch_size)
        self.clear_capture_buffer()

        embedding_tensor = self.get_embedding_for_tokens(prefill_seq)
        self.dma_to_accelerator_memory(self.LAYER0_OUTPUT_DRAM, embedding_tensor)
        bias_one_group = torch.full((aligned_seq_len, aligned_seq_len), float("-inf"), dtype=torch.bfloat16)
        rows = torch.arange(aligned_seq_len).unsqueeze(1)
        cols = torch.arange(aligned_seq_len).unsqueeze(0)
        valid_mask = (cols // self.group_size) <= (rows // self.group_size)
        bias_one_group.masked_fill_(valid_mask, 0.0)
        bias_one_group[:, q_seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_one_group)

        print(f"\n--- Starting prefill (seq_len={prefill_seq_len}) ---")
        print(f"Prompt ({len(self.prefill_seq)}) tokens: {self.prefill_seq}")
        timer = time.perf_counter()
        hw_lat_prefill_us, _ = self.program_execute(preamble_addr, flops=flops_prefill)
        latency_prefill = time.perf_counter() - timer
        print(f"Prefill done in {latency_prefill:.2f}s\n")

        print("--- Starting decoder ---")
        hw_decode_lats_us: list[float] = []
        decoded_chars: list[str] = []
        decode_bias_host: torch.Tensor | None = None
        decode_bias_aligned_seq_len = 0
        timer = time.perf_counter()
        token_id = self.prefill_seq[-1]
        _llama_stop_tokens = {128001, 128008, self._end_of_turn_token_id}
        global _SILENT_MODE

        # Penalty window state: the on-FPGA penalty counts every token seen so far (prompt +
        # decoded), seeded by main() with the prompt ids. Falls back to the full prompt when run
        # without main().
        if not hasattr(self, "_generated_tokens"):
            self._generated_tokens = list(self.prefill_seq)
        # Position-gated hybrid decode (deterministic): PURE greedy (HW argmax) for the first
        # `greedy_until` decoded tokens — correct math/reasoning, which lands early — then the
        # on-FPGA repetition penalty turns on to break long-context loops.
        _greedy_until = int(getattr(self, "greedy_until", 0))
        # On-FPGA penalty: the LM-head matmul adds PENALTY_BIAS_DRAM (its C bias) so the HW argmax
        # already returns the penalized token. Zero the buffer first → pure greedy until the gate,
        # then refresh the full bias each step past the gate (_write_penalty_bias). Plain mode
        # (--pure-greedy, writeback-on bin for compare/baseline) leaves the buffer untouched.
        _fpga_penalty = bool(getattr(self, "fpga_penalty", False))
        if _fpga_penalty:
            self.dma_to_accelerator_memory(self.PENALTY_BIAS_DRAM,
                                           torch.zeros(1, self.EMBEDDING_ELEMENTS, dtype=torch.bfloat16))

        # Two-region live counter: pin the bottom terminal row as a status line via
        # an ANSI scroll region; tokens stream in the area above it and the counter
        # refreshes in place. All output is on stdout (tokens scroll inside rows
        # 1..rows-1; the status writes row `rows` with cursor save/restore), so
        # nothing clobbers the streamed token text. Only when stdout is a real TTY
        # (skip when piped/redirected). Mirrors gemma3_test.py.
        import shutil
        _use_status = sys.stdout.isatty()
        def _status_setup():
            rows = shutil.get_terminal_size().lines
            sys.stdout.write(f"\033[1;{rows - 1}r")   # scroll region = rows 1..rows-1
            sys.stdout.write(f"\033[{rows - 1};1H")   # park cursor at bottom of region
            sys.stdout.flush()
        def _status_update():
            rows = shutil.get_terminal_size().lines
            n = self.seq_len - prefill_seq_len
            elapsed = time.perf_counter() - timer
            rate = n / elapsed if elapsed > 0 else 0.0
            sys.stdout.write("\0337")                 # save cursor
            sys.stdout.write(f"\033[{rows};1H\033[2K") # bottom row, clear it
            sys.stdout.write(f" decoding… {n} tokens  (pos {self.seq_len}/{self.MAX_CONTEXT_SIZE})  "
                             f"{elapsed:.1f}s  {rate:.1f} tok/s")
            sys.stdout.write("\0338")                 # restore cursor
            sys.stdout.flush()
        def _status_teardown():
            rows = shutil.get_terminal_size().lines
            sys.stdout.write("\033[r")                # reset scroll region
            sys.stdout.write(f"\033[{rows};1H\033[2K") # clear the status row
            sys.stdout.flush()
        if _use_status:
            _status_setup()

        while self.seq_len < self.MAX_CONTEXT_SIZE:
            _SILENT_MODE = True
            self.seq_len += 1
            aligned_seq_len = ((self.seq_len + 63) // 64) * 64
            decode_pos = self.seq_len - 1

            if 0 <= token_id < self.embedding_weight.shape[0]:
                embedding_tensor = self.embedding_weight[token_id:token_id + 1]
            else:
                embedding_tensor = self.get_embedding_for_tokens([token_id])
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
            # unified_attention_core's dynamic path always uses bias_mode="full_matrix" (one bias
            # row per batch item); the decoder attention call's batch=qpkv query heads all share the
            # same causal mask. Reuse one contiguous buffer within each 64-token bucket.
            if aligned_seq_len != decode_bias_aligned_seq_len:
                decode_bias_host = torch.full(
                    (self.group_size, aligned_seq_len), -1e36, dtype=torch.bfloat16)
                decode_bias_host[:, :self.seq_len] = 0.0
                decode_bias_aligned_seq_len = aligned_seq_len
            else:
                decode_bias_host[:, self.seq_len - 1] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, decode_bias_host)

            # Select the prebuilt register setup + jump for this absolute position.
            decoder_step_addr = decoder_dispatch_addr + (
                decode_pos - prefill_seq_len) * decoder_dispatch_stride

            # On-FPGA penalty: refresh the per-vocab bias (this step's LM-head matmul C term) from the
            # windowed token frequency once past the gate, so the HW argmax of (logits + bias) returns
            # the penalized token directly — no logit readback. A single full-buffer DMA per step;
            # incremental ±α chunk writes were measured SLOWER (per-DMA device open/close dominates)
            # and lower quality — see notes_repetition_penalty_fpga_bias.md.
            if _fpga_penalty and (self.seq_len - prefill_seq_len) > _greedy_until:
                self._write_penalty_bias(self._generated_tokens)

            hw_lat_dec_us, _ = self.program_execute(decoder_step_addr, flops=decoder_flops_per_token)
            hw_decode_lats_us.append(hw_lat_dec_us)
            # Token selection: read the HW argmax register. In penalty mode the LM-head matmul
            # already added the bias, so the register holds the penalized token; in plain mode it's
            # pure greedy. Either way no logit readback.
            token_id = self.get_arg_max_index(rank=1)
            self._generated_tokens.append(token_id)
            token_char = self.tokenizer.decode([token_id])
            _SILENT_MODE = False
            if token_id in _llama_stop_tokens:
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

        latency_decoder = time.perf_counter() - timer
        tokens_decoded = self.seq_len - prefill_seq_len
        print(f"\nDecoder done in {latency_prefill + latency_decoder:.2f}s, "
              f"speed: {tokens_decoded / latency_decoder:.2f} tokens/s, "
              f"total {self.seq_len} tokens.")

        hw_decode_avg_ms = sum(hw_decode_lats_us) / len(hw_decode_lats_us) / 1e3 if hw_decode_lats_us else 0.0
        hw_decode_first_ms = hw_decode_lats_us[0] / 1e3 if hw_decode_lats_us else 0.0
        hw_decode_total_us = sum(hw_decode_lats_us)
        decoder_gflops = (
            decoder_flops_per_token * len(hw_decode_lats_us)
            / (hw_decode_total_us * 1e3)
            if hw_decode_total_us > 0 else 0.0
        )
        _original_print(
            f"Report FLOPS for decoder execution: {decoder_gflops:.2f} GFLOPS"
        )
        cpu_decode_avg_ms = latency_decoder * 1e3 / tokens_decoded if tokens_decoded else 0.0
        peak_tokens_per_s = 1000.0 / hw_decode_first_ms if hw_decode_first_ms > 0 else 0.0
        avg_tokens_per_s = tokens_decoded / latency_decoder if latency_decoder > 0 else 0.0
        _original_print("\n=== Performance Summary ===")
        _original_print(f"Instruction size  : prefill={meta['prefill_program_size']/1024:.1f} kB  decoder={meta['decoder_program_size']/1024:.1f} kB  total={(meta['prefill_program_size']+meta['decoder_program_size'])/1024:.1f} kB")
        _original_print(f"Prefill ({prefill_seq_len} tokens): HW={hw_lat_prefill_us/1e3:,.1f} ms  CPU={latency_prefill*1e3:,.1f} ms")
        _original_print(f"Decode 1st token  : HW={hw_decode_first_ms:,.1f} ms/tok  ({1000/hw_decode_first_ms:.2f} tok/s)")
        _original_print(f"Decode  ({tokens_decoded} tokens): HW={hw_decode_avg_ms:,.1f} ms/tok  CPU={cpu_decode_avg_ms:,.1f} ms/tok  ({tokens_decoded/latency_decoder:.2f} tok/s)")

        return {
            "prefill_tokens": prefill_seq_len,
            "decoded_text": "".join(decoded_chars),
            "decoded_tokens": tokens_decoded,
            # Compatibility aliases shared with Gemma3 and user_hw_test.py.
            "tokens_decoded": tokens_decoded,
            "avg_tokens_per_s": avg_tokens_per_s,
            "peak_tokens_per_s": peak_tokens_per_s,
            "prefill_speed_tok_s": round(prefill_seq_len / latency_prefill, 2),
            "decode_speed_tok_s": round(avg_tokens_per_s, 2),
            "decoder_gflops": round(decoder_gflops, 2),
            "prefill_size_kb": round(meta["prefill_program_size"] / 1024, 1),
            "decoder_size_kb": round(meta["decoder_program_size"] / 1024, 1),
            "prefill_hw_ms": hw_lat_prefill_us / 1e3,
            "prefill_cpu_ms": latency_prefill * 1e3,
            "decode_first_hw_ms": hw_decode_first_ms,
            "decode_avg_hw_ms": hw_decode_avg_ms,
            "decode_avg_cpu_ms": cpu_decode_avg_ms,
        }

    def _profile_execute(self, preamble_addr: int, checkpoints: list,
                         tail_label: str | None = None, timeout: float = 30.0) -> tuple[list, dict]:
        """Walk an unrolled program through its per-layer HALT checkpoints, summing each step's HW
        latency across all layers. Checkpoint names carry an ``L<idx>_`` prefix that is stripped so
        the per-layer HALTs roll up by step type. The post-loop fall-through segment (final norm +
        LM head for decode; the terminating HALT for prefill) is always drained and recorded under
        ``tail_label`` when one is given. Returns ``(ordered_step_names, {step: summed_ms})``.
        """
        from collections import OrderedDict
        step_ms: "OrderedDict[str, float]" = OrderedDict()
        self.start_execute_from_dram(preamble_addr)
        for name, resume_addr_hex in checkpoints:
            self.wait_queue(timeout)
            step = name.split("_", 1)[1] if name.startswith("L") and "_" in name else name
            step_ms[step] = step_ms.get(step, 0.0) + self.report_latency_in_us() / 1e3
            self.start_execute_from_dram(int(resume_addr_hex, 16))
        # Drain the post-loop fall-through (final norm + LM head for decode; bare HALT for prefill).
        self.wait_queue(timeout)
        tail_ms = self.report_latency_in_us() / 1e3
        if tail_label is not None:
            step_ms[tail_label] = tail_ms
        return list(step_ms.keys()), step_ms

    @staticmethod
    def _print_profile_table(title: str, order: list, step_ms: dict) -> float:
        """Print one per-step HW-latency table (summed over all layers) and return the total ms.

        Uses ``_original_print`` so the table is never swallowed by ``_SILENT_MODE`` (which is
        held True during the profiled execution to mute the per-instruction capture logging).
        """
        total_ms = sum(step_ms.values())
        _original_print(f"\n=== {title} (HW latency, summed over all layers) ===")
        _original_print(f"{'Step':<38} {'ms':>9}  {'%':>6}")
        _original_print("-" * 57)
        for name in order:
            ms = step_ms[name]
            pct = ms / total_ms * 100 if total_ms > 0 else 0.0
            _original_print(f"{name:<38} {ms:>9.3f}  {pct:>5.1f}%")
        _original_print("-" * 57)
        _original_print(f"{'Total':<38} {total_ms:>9.3f}  100.0%")
        return total_ms

    def run_llama_profile(self) -> None:
        """Load the profile instruction image and print ONE per-step table for prefill and ONE for
        the first decoded token. Each program is walked through its per-layer HALT checkpoints and
        each step is summed across all layers (:meth:`_profile_execute`), so the tables are per-step
        totals for the whole phase — with no per-layer breakdown (mirrors gemma3's profiler).
        """
        meta_path = getattr(self, "_instruction_meta_path", None) or \
            os.path.join(self.script_dir, "llama3.2_1b_if8_bin/llama3.2_1b_profile_program.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.load_program_instructions_from_file(os.path.join(self.script_dir, meta["instruction_bin"]))
        preamble_addr = self.get_program_dram_addr()
        if preamble_addr + 64 * 1024 > 0x100000000:
            raise MemoryError(
                f"Instruction image exceeds 10 MiB program region: preamble=0x{preamble_addr:X}"
            )

        prefill_program_addr = int(meta["prefill_program_start_addr"], 16)
        decoder_program_addr = int(meta["decoder_program_start_addr"], 16)
        prefill_checkpoints  = meta.get("prefill_profile_checkpoints", [])
        decoder_checkpoints  = meta.get("decoder_profile_checkpoints", [])
        _kv_stride = self.actual_head_dim * self.bytes_per_element
        _rope_row  = self.head_dim * 2 * self.bytes_per_element

        prefill_seq = self.prefill_seq or tuple(self._cfg["default_prefill_tokens"])
        if len(prefill_seq) < 2:
            raise ValueError("Prefill sequence must have at least 2 tokens.")
        prefill_seq = prefill_seq[:-1]  # last token starts the decoder
        prefill_seq_len = len(prefill_seq)
        self.seq_len = prefill_seq_len

        q_seq_len = prefill_seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        bucket_idx = aligned_seq_len // UE_VECTOR_SIZE

        # On-FPGA penalty needs a zeroed bias buffer so the profiled decode step is pure-greedy.
        if bool(getattr(self, "fpga_penalty", False)):
            self.dma_to_accelerator_memory(self.PENALTY_BIAS_DRAM,
                                           torch.zeros(1, self.EMBEDDING_ELEMENTS, dtype=torch.bfloat16))

        # --- Prefill preamble + inputs (mirrors run_llama) ---
        self.clear_inst_id()
        self.start_capture()
        self.generate_instruction_add_set(self.gpr_seq_len, prefill_seq_len)
        self.generate_instruction_add_set(self.gpr_bucket_idx, bucket_idx)
        self.generate_instruction_add_set(self.gpr_q_seq_len, q_seq_len)
        self.generate_instruction_add_set(self.gpr_aligned_seq_len, aligned_seq_len)
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(prefill_program_addr))
        self.stop_capture()
        self.write_captured_instructions_to_dram(preamble_addr)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        self.clear_capture_buffer()

        embedding_tensor = self.get_embedding_for_tokens(prefill_seq)
        self.dma_to_accelerator_memory(self.LAYER0_OUTPUT_DRAM, embedding_tensor)
        bias_one_group = torch.full((aligned_seq_len, aligned_seq_len), float("-inf"), dtype=torch.bfloat16)
        rows = torch.arange(aligned_seq_len).unsqueeze(1)
        cols = torch.arange(aligned_seq_len).unsqueeze(0)
        valid_mask = (cols // self.group_size) <= (rows // self.group_size)
        bias_one_group.masked_fill_(valid_mask, 0.0)
        bias_one_group[:, q_seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_one_group)

        global _SILENT_MODE
        _SILENT_MODE = True
        _original_print(f"\n--- Profiling: prefill (seq_len={prefill_seq_len}) ---")
        if prefill_checkpoints:
            order, step_ms = self._profile_execute(preamble_addr, prefill_checkpoints)
            prefill_total_ms = self._print_profile_table(f"Prefill  (seq_len={prefill_seq_len})", order, step_ms)
        else:
            prefill_total_ms = 0.0
            _original_print("  (no prefill checkpoints in meta — recompile with --profile to enable)")

        # --- Decoder preamble + inputs for the first decoded token (mirrors run_llama) ---
        token_id = self.prefill_seq[-1]
        self.seq_len += 1
        aligned_dec = ((self.seq_len + 63) // 64) * 64
        bucket_idx = min((self.seq_len + 63) // 64, (self.MAX_CONTEXT_SIZE + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE)
        decode_pos = self.seq_len - 1

        embedding_tensor = self.get_embedding_for_tokens([token_id])
        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
        bias_host = torch.full((1, aligned_dec), -1e36, dtype=torch.bfloat16)
        bias_host[0, :self.seq_len] = 0.0
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_host.repeat(self.group_size, 1))

        self.clear_inst_id()
        self.start_capture()
        self.generate_instruction_add_set(self.gpr_bucket_idx, bucket_idx)
        self.generate_instruction_add_set(self.gpr_aligned_seq_len, aligned_dec)
        self.generate_instruction_add_set(self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter(decode_pos * _kv_stride))
        self.generate_instruction_add_set(self.ROPE_SIZE_REG, ue_35bit_addr_shifter(decode_pos * _rope_row))
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(decoder_program_addr))
        self.stop_capture()
        self.write_captured_instructions_to_dram(preamble_addr)
        self.clear_capture_buffer()

        _original_print("\n--- Profiling: decoder (first decoded token) ---")
        order, step_ms = self._profile_execute(
            preamble_addr, decoder_checkpoints, tail_label="output_norm_lm_head")
        decoder_total_ms = self._print_profile_table("Decoder  (first token)", order, step_ms)
        _SILENT_MODE = False

        if prefill_total_ms > 0:
            _original_print(f"\nPrefill (HW): {prefill_total_ms:.2f} ms  ({prefill_seq_len} tokens)")
        _original_print(f"Decode  (HW): {decoder_total_ms:.2f} ms/tok  ({1000/decoder_total_ms:.2f} tok/s)")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def _clock_ns_default_for_device(device: str) -> float:
    """Return default clock period (ns) for FPGA type — mirrors user_hw_test.py."""
    if device == "kintex7":                       return 1000 / (1066 / 5.375)
    if device in ("rk", "puzhi"):                 return 3.0
    if device in ("bittware", "bittware_256"):     return 3.3333
    if device == "alveo":                          return 4.0
    if device == "efinix":                         return 4.0
    return 10.0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Llama-3.2-1B IF8 prefill + decode on accelerator.")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt")
    parser.add_argument(
        "--standard-chat-template",
        action="store_true",
        help="Use the tokenizer's full dated system prompt. Default: canonical user/assistant "
             "headers without the system/date block.",
    )
    parser.add_argument("--local-weights", action="store_true", help="Use llama3.2_1b_if8_bin/full_model_weights.bin")  # legacy dev path; not in standard bin set
    parser.add_argument('--dev', type=str, default='xdma0', help='DMA device name (default: xdma0)')
    parser.add_argument('--cycle', type=float, default=None, help='Clock cycle time in ns. Overrides --device default.')
    parser.add_argument('--device', type=str, default='kintex7', help='FPGA board profile (kintex7, rk, puzhi, bittware, bittware_256, alveo, efinix).')
    parser.add_argument('--decoder-matmatmul', '--matmatmul',
                        dest='decoder_matmatmul', action='store_true',
                        help='Use matmat_mul_core for the decoder quantized matmats (incl. the LM head) '
                             'instead of the default quantized_matmat_core (streaming) path. Same IF8 '
                             'weights either way: matmat_mul_core dequantizes B to bf16 in URAM and does '
                             'a bf16 dot (2 passes over the weight bytes, ~2x slower, higher-precision '
                             'accumulate); the default streams IF8 through the dot unit in 1 pass.')
    parser.add_argument(
        '--stream-prefill',
        action='store_true',
        help='A/B test the one-pass streaming IF8 prefill matmul. Default: the measured-faster '
             'dequantize-to-BF16 path used by Gemma IF8.',
    )
    parser.add_argument('--profile', action='store_true',
                        help='Compile a profile binary with per-step HALT checkpoints and print one '
                             'per-step HW-latency table (summed over all layers) for prefill and for '
                             'the first decoded token.')
    # On-FPGA repetition penalty is the DEFAULT decode path: the penalty is folded into the LM-head
    # matmul bias so the HW argmax returns the penalized token directly — no logit readback,
    # fully deterministic. --pure-greedy disables it entirely.
    parser.add_argument('--pure-greedy', action='store_true',
                        help='Disable the on-FPGA repetition penalty entirely — plain greedy decode '
                             '(writeback-on bin). The penalty is ENABLED by default; use --pure-greedy '
                             'only for the A/B baseline and the compare/calibration tool.')
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
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.local_weights:
        weights_bin_rel = "llama3.2_1b_if8_bin/full_model_weights.bin"
    else:
        weights_bin_rel = "llama3.2_1b_if8_bin/params.bin"
        weights_bin_full = os.path.join(script_dir, weights_bin_rel)
        if not os.path.exists(weights_bin_full):
            weight_bin_generate(script_dir=script_dir, output_path=weights_bin_full)

    set_dma_device("efinix" if args.device == "efinix" else args.dev)
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
    effective_dma = "pcie_dma0" if args.device == "efinix" else args.dev
    print(f"FPGA profile: device={args.device}, clock={clock:.4f} ns, UE_AXI_DATA_WIDTH_BITS={axi_width_bits}")

    ue = UnifiedEngine()
    ue.software_reset()

    ue = Llama32_1b_IF8_UnifiedEngine(
        script_dir=script_dir,
        weights_bin=weights_bin_rel,
        decoder_matmatmul=args.decoder_matmatmul,
        stream_prefill=args.stream_prefill,
    )
    cfg = _load_config(script_dir)

    if args.prompt is not None:
        tok_path = os.path.join(script_dir, cfg["paths"]["hf_model_dir"])
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
        if args.standard_chat_template:
            conversation = [{"role": "user", "content": args.prompt}]
            prompt_with_template = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True)
            template_name = "standard dated"
        else:
            prompt_with_template = _minimal_chat_prompt(args.prompt)
            template_name = "minimal"
        prefill_seq = tuple(tokenizer.encode(prompt_with_template, add_special_tokens=False))
        print(f"Prefill from prompt ({len(prefill_seq)} tokens, {template_name} template): {args.prompt!r}")
    else:
        prefill_seq = tuple(cfg["default_prefill_tokens"])

    max_prefill = ue.PREFILL_CONTEXT_SIZE
    if len(prefill_seq) < 2 or len(prefill_seq) > max_prefill + 1:
        print(f"WARNING: prompt length {len(prefill_seq)} out of range [2, {max_prefill + 1}], falling back to default.")
        prefill_seq = tuple(cfg["default_prefill_tokens"])

    ue.prefill_seq = prefill_seq

    # Decode config — deterministic, on-FPGA penalty only. Must be set BEFORE compile_llama() since
    # fpga_penalty changes the compiled LM-head matmul (bias on / writeback off).
    ue.greedy_until = int(args.greedy_until)
    ue.fpga_penalty = not bool(args.pure_greedy)
    ue.pen_alpha = float(args.pen_alpha)
    ue.pen_cap = float(args.pen_cap)
    ue.rep_window = int(args.rep_window)
    ue._generated_tokens = list(prefill_seq)   # seed the penalty window with the prompt
    if ue.fpga_penalty:
        print(f"Decode: ON-FPGA penalty (bias in LM-head matmul) — pure greedy for {ue.greedy_until} "
              f"tokens, then alpha={ue.pen_alpha} cap={ue.pen_cap} window={ue.rep_window}")
        # Precompute the structural-exemption set upfront (one vocab scan) so it doesn't stall the
        # first penalized decode step.
        _n = len(ue._structural_token_ids())
        print(f"  penalty exempts {_n} structural/special tokens (punctuation/whitespace/newline)")
    else:
        print("Decode: plain greedy (deterministic) — no penalty (writeback-on bin)")

    if args.profile:
        print("\n--- Compiling profile binary ---")
        timer = time.perf_counter()
        ue.compile_llama(profile=True)
        print(f"Compile done in {time.perf_counter() - timer:.2f}s")
        print("\n--- Running profile ---")
        ue.run_llama_profile()
        print("Decoder/prefill profile done.")
        return

    print("\n--- Compiling ---")
    timer = time.perf_counter()
    ue.compile_llama()
    print(f"Compile done in {time.perf_counter() - timer:.2f}s")

    print("\n--- Running ---")
    run_result = ue.run_llama()
    print("Llama-3.2-1B IF8 test ends.")
    _original_print(f"TEST_RESULT: {json.dumps(run_result)}")

if __name__ == "__main__":
    main()
