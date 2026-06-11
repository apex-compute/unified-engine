#!/usr/bin/env python3
"""
Gemma3 inference on accelerator: prefill + decode.

  - Config from gemma3_config.json; weights from a single bin (see below).
  - A single prefill program (sized to ``len(prompt) - 1``) plus every decoder seq_len bucket are
    compiled into ONE combined instruction image plus a metadata sidecar (per-stage start
    addresses, sizes, FLOPs). Both are runtime-generated tmp artifacts inside gemma3_bin/, with
    relative paths declared in gemma3_config.json (paths.instruction_bin / instruction_meta).
    The bin is tied to the prompt length used at compile time and is recompiled per run.
  - Run phase loads the combined bin once into program DRAM, executes the single prefill
    program, then dispatches per-token decoder programs by start address.

Weights:
  - Default: gemma3_bin/weights_gemma3_hf.bin (generated from HF model if missing).
  - --local-weights: use gemma3_bin/full_model_weights.bin instead.

Usage:
  python gemma3_test.py
  python gemma3_test.py --prompt "your prompt"
  python gemma3_test.py --dev xdma0 [--device kintex7] [--cycle 5.15]
  python gemma3_test.py --dev xdma0 --device bittware
  python gemma3_test.py --local-weights
  python gemma3_test.py --dual-engine

``--device`` matches ``user_hw_test.py`` (FPGA profile: ``UE_AXI_DATA_WIDTH_BITS`` and default clock).
``--dev`` is the XDMA device name (e.g. ``xdma0``). For Bittware use ``--device bittware``; override clock with ``--cycle`` if needed.

Fixed layout: gemma3_test.py, gemma3_numeric.py, *.json, and gemma3_bin/ live in the same folder.
  user_dma_core.py is one folder up; that parent is added to sys.path.
"""

import json
import math
import os
import sys

# This file's folder: gemma3_bin/, *.json live here. user_dma_core is two folders up.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import time
# pcie_utils imports (run from andromeda/pcie_utils or with PYTHONPATH)
import user_dma_core
from user_dma_core import DMA_DEVICE_H2C, DRAM_INSTRUCTION_ADDR, INSTRUCTION_SIZE_BYTES, TYPE, UE_FMAX_CONTEXT_SIZE, UE_MODE, UE_VECTOR_SIZE, UE_ARGMAX_INDEX, URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS, set_dma_device, ue_35bit_addr_shifter
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

# Codec lives in quant_schemas.py -- the single canonical HW-aligned implementation
# shared across all model templates. Parallels src/quantization/kv_quant_schemas.py
# (KV-cache schemas for SW simulation). Gemma3 calls the string-dispatched
# ``quantize`` directly so the codebook is config-driven; today's release uses "if4".
# To switch codebooks, change QUANT_PRECISION (``int4`` / ``fp4`` / ``if4``); to force
# a per-block variant within IF4, pass ``int_variant=True`` (uniform INT4) or ``False``
# (uniform FP4) to quantize().
from quant_lib import quantize

QUANT_PRECISION = "if4"

def weight_bin_generate(output_path: str | None = None, config_path: str | None = None) -> str:
    """Generate full_model_weights.bin from Hugging Face model per gemma3_config.json layout.

    Quantizable weights (Q/K/V/O projections, MLP up/gate/down, LM head) use
    IF4: per-K-block min-MSE selection between INT4 and FP4, dispatched on
    the FPGA via the bf16 scale's sign bit. Norms / embeddings stay BF16.

    Returns the path to the written file."""
    cfg = Gemma3_UnifiedEngine.load_config(config_path=config_path, script_dir=SCRIPT_DIR)
    weight_defs = cfg["_weight_defs"]
    paths = cfg["paths"]
    paths_full = os.path.join(SCRIPT_DIR, paths["weights_bin"])
    out_path = output_path or paths_full

    model, model_dir = _ensure_hf_model(SCRIPT_DIR, cfg)
    gamma_offset = cfg["special"]["rms_norm"]["gamma_offset"]
    emb_cfg = cfg["special"]["embedding"]
    token_embd_offset = _parse_offset(emb_cfg["token_embd_offset"])
    token_embd_size = _parse_offset(emb_cfg["token_embd_size"])
    LAYER_WEIGHT_SIZE = weight_defs["LAYER_WEIGHT_SIZE"]
    base_layer0 = weight_defs["BLK0_ATTN_NORM_WEIGHT"]
    num_layers = cfg["file_info"]["num_layers"]
    head_dim = cfg["file_info"]["head_dim"]
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

    # Embedding
    embed = model.get_input_embeddings().weight.detach().cpu().to(torch.bfloat16)
    embedding_scale = model.config.hidden_size ** 0.5
    emb_scaled = (embed.float() * embedding_scale).to(torch.bfloat16)
    raw_emb = emb_scaled.contiguous().view(torch.uint8).numpy().tobytes()
    write_at(token_embd_offset, raw_emb)

    # Layers
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn
        gamma_in = (layer.input_layernorm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
        q_w = attn.q_proj.weight.detach().cpu().to(torch.bfloat16)
        k_w = attn.k_proj.weight.detach().cpu().to(torch.bfloat16)
        v_w = attn.v_proj.weight.detach().cpu().to(torch.bfloat16)
        gamma_q = (attn.q_norm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)[:head_dim]
        gamma_k = (attn.k_norm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
        o_w = attn.o_proj.weight.detach().cpu().to(torch.bfloat16)
        gamma_post = (layer.post_attention_layernorm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
        gamma_ffn = (layer.pre_feedforward_layernorm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
        gate_w = layer.mlp.gate_proj.weight.detach().cpu().to(torch.bfloat16)
        up_w = layer.mlp.up_proj.weight.detach().cpu().to(torch.bfloat16)
        down_w = layer.mlp.down_proj.weight.detach().cpu().to(torch.bfloat16)
        gamma_post_ffn = (layer.post_feedforward_layernorm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)

        region_writes = [
            (gamma_in, "bf16"),
            (q_w, "if4"),
            (k_w, "if4"),
            (v_w, "if4"),
            (gamma_q, "bf16"),
            (gamma_k, "bf16"),
            (o_w, "if4"),
            (gamma_post, "bf16"),
            (gamma_ffn, "bf16"),
            (up_w, "if4"),
            (gate_w, "if4"),
            (down_w, "if4"),
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
            if kind == "if4":
                next_key = blk0_structure[i + 1]["key"]
                data_sz = weight_defs[f"{next_key}_SIZE"]
                data_bytes, scale_bytes = quantize(QUANT_PRECISION, tensor)
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

    # ROPE
    rope_cfg = cfg["special"]["rope"]
    theta = rope_cfg["theta"]
    local_base = rope_cfg["local_base"]
    num_positions = rope_cfg["num_positions"]
    D = head_dim // 2
    for name, theta_val, off_key, sz_key in [
        ("ROPE_LOCAL", local_base, "ROPE_LOCAL", "ROPE_LOCAL_SIZE"),
        ("ROPE_GLOBAL", theta, "ROPE_GLOBAL", "ROPE_GLOBAL_SIZE"),
    ]:
        inv_freq = 1.0 / (theta_val ** (torch.arange(D, dtype=torch.float32) / D))
        pos = torch.arange(num_positions, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        cos_ = freqs.cos().to(torch.bfloat16)
        sin_ = freqs.sin().to(torch.bfloat16)
        rope_tensor = torch.cat([cos_, cos_, -sin_, sin_], dim=1)
        sz = weight_defs[sz_key]
        raw = (rope_tensor.contiguous().view(torch.uint8).numpy().tobytes() + b"\x00" * sz)[:sz]
        write_at(weight_defs[off_key], raw)

    # OUTPUT_NORM
    out_norm = model.model.norm.weight.detach().cpu().to(torch.bfloat16)
    sz = weight_defs["OUTPUT_NORM_WEIGHT_SIZE"]
    raw = (out_norm.contiguous().view(torch.uint8).numpy().tobytes() + b"\x00" * sz)[:sz]
    write_at(weight_defs["OUTPUT_NORM_WEIGHT"], raw)

    # LM_HEAD
    lm_head_w = model.lm_head.weight.detach().cpu().to(torch.bfloat16)
    scale_sz = weight_defs["LM_HEAD_WEIGHT_SCALE_SIZE"]
    data_sz = weight_defs["LM_HEAD_WEIGHT_DATA_SIZE"]
    data_bytes, scale_bytes = quantize(QUANT_PRECISION, lm_head_w)
    scale_padded = (scale_bytes + b"\x00" * scale_sz)[:scale_sz]
    data_padded = (data_bytes + b"\x00" * data_sz)[:data_sz]
    write_at(weight_defs["LM_HEAD_WEIGHT_SCALE"], scale_padded)
    write_at(weight_defs["LM_HEAD_WEIGHT_DATA"], data_padded)

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
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map=None, trust_remote_code=True
    )
    return model, model_dir

# -----------------------------------------------------------------------------
# Gemma3 unified engine
# -----------------------------------------------------------------------------
class Gemma3_UnifiedEngine(UnifiedEngine):
    """UnifiedEngine with Gemma3 dims: loads config + weight bin; compile_prefill/compile_decoder;
    run_prefill/run_decoder (numeric checks in gemma3_numeric.py).

    Dual-engine: not verified end-to-end after the PBI migration. The compile path still carries
    sharding hooks (row split, flags, ``shard_dram_addr``) where practical so future dual-PBI work
    can build on the same structure; runtime enablement remains gated until validated.
    """

    def __init__(self, script_dir: str | None = None, local_weights: bool = False, dual_engine: bool = False, engine_slave: bool = False):
        program_dram_base = DRAM_INSTRUCTION_ADDR + 0x10000000 if engine_slave else DRAM_INSTRUCTION_ADDR
        engine_base = user_dma_core.UE_0_BASE_ADDR + 0x00010000 if engine_slave else user_dma_core.UE_0_BASE_ADDR
        super().__init__(BASE_ADDR=engine_base, program_dram_base=program_dram_base)
        self.dual_engine = dual_engine
        self.script_dir = script_dir or SCRIPT_DIR
        self._cfg = self.load_config(script_dir=self.script_dir)
        self.weight_defs = self._cfg["_weight_defs"]

        fi = self._cfg["file_info"]
        model = self._cfg["model"]
        paths = self._cfg["paths"]

        self.vector_length = fi["hidden_size"]
        self.head_dim = fi["head_dim"]
        self.bytes_per_element = fi["bytes_per_element"]
        self.group_size = fi["group_size"]
        self.mlp_elements = fi["mlp_elements"]
        self.q_size = self.head_dim * self.group_size * self.bytes_per_element
        self.k_size = self.head_dim * self.bytes_per_element
        self.MAX_CONTEXT_SIZE = model["max_context_size"]
        self.PREFILL_CONTEXT_SIZE = model["prefill_context_size"]
        self.PREFILL_MAX_SEQ_LEN = int(model["prefill_max_seq_len"])
        self.LAYER_SIZE = fi["num_layers"]
        self.EMBEDDING_ELEMENTS = fi["embedding_vocab"]
        fixed = self._cfg.get("fixed_isa_regs", {})
        self.TMP_REG = fixed["TMP_REG"]
        # Fixed GPR indices (see ``fixed_isa_regs`` in gemma3_config.json):
        #   GPR_SEQ_LEN — current token position; multiplied at runtime to derive K/V/RoPE offsets.
        #   GPR_Q_SEQ_LEN — row count for PBI Q-head RMS outer loops.
        #   GPR_BUCKET_IDX — flash-attention bucket selector (= seq_len // UE_VECTOR_SIZE).
        self.gpr_seq_len = fixed["GPR_SEQ_LEN_REG"]
        self.gpr_q_seq_len = fixed["GPR_Q_SEQ_LEN_REG"]
        self.gpr_bucket_idx = fixed["GPR_BUCKET_IDX_REG"]
        # Dynamic GPR allocation must not use 1–4 (TMP, GPR_SEQ_LEN, GPR_Q_SEQ_LEN, GPR_BUCKET_IDX).
        self._isa_reg_counter = 5
        self.causal_mask_upper = False
        self._rope_global_layers = set(model["rope_global_layers"])
        self._end_of_turn_token_id = model["end_of_turn_token_id"]
        self._gamma_bin_offset = self._cfg["special"]["rms_norm"]["gamma_offset"]
        self.prefill_seq = None 
        self.engine_slave = engine_slave

        self._weights_bin_rel = "gemma3_bin/full_model_weights.bin" if local_weights else paths["weights_bin"]
        self.weight_init()
        self.tensor_init()

    @staticmethod
    def load_config(config_path: str | None = None, script_dir: str | None = None) -> dict:
        """Load gemma3_config.json and build weight_defs (offset/size dict) from regions."""
        if config_path is None:
            script_dir = script_dir or SCRIPT_DIR
            config_path = os.path.join(script_dir, "gemma3_config.json")
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
        """Set self.prefill_seq.

        Prefill compiles a single program sized to ``len(self.prefill_seq) - 1`` (last token feeds
        the decoder), so the tokenized prompt length must lie in ``[2, prefill_max_seq_len + 1]``.
        Out-of-range prompts fall back to the default prompt with a warning.
        """
        max_prefill = self.PREFILL_MAX_SEQ_LEN
        default_tokens = tuple(self._cfg["default_prefill_tokens"])
        if prompt is not None:
            conversation = [{"role": "user", "content": prompt}]
            prompt_with_template = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            tokens = tuple(self.tokenizer.encode(prompt_with_template, add_special_tokens=True))
            if len(tokens) < 2 or len(tokens) > max_prefill + 1:
                print(
                    f"WARNING: Tokenized prompt has {len(tokens)} tokens; supported prompt length is "
                    f"[2, {max_prefill + 1}] (prefill seq_len in [1, {max_prefill}]). "
                    "Falling back to default prompt."
                )
                self.prefill_seq = default_tokens
            else:
                self.prefill_seq = tokens
                print(f"Prefill from prompt ({len(self.prefill_seq)} tokens): {prompt!r}")
        else:
            self.prefill_seq = default_tokens
        print(f"Sequence ids: {self.prefill_seq}")

    def get_embedding_for_tokens(self, token_ids: list[int] | tuple) -> torch.Tensor:
        """Return (len(token_ids), vector_length) bfloat16 tensor from self.embedding_weight (HF, scale applied)."""
        tid_t = torch.tensor(token_ids, dtype=torch.long)
        out = torch.zeros(len(token_ids), self.vector_length, dtype=torch.bfloat16)
        valid = tid_t < self.embedding_weight.shape[0]
        out[valid] = self.embedding_weight[tid_t[valid]]
        return out

    def _load_rope_host(self, rope_theta: float | None = None, rope_local_base: float | None = None) -> None:
        """Generate RoPE (cos, cos, -sin, sin) on host and write to DRAM. Uses config for sizes and num_positions."""
        rope_cfg = self._cfg["special"]["rope"]
        theta = rope_theta if rope_theta is not None else rope_cfg["theta"]
        local_base = rope_local_base if rope_local_base is not None else rope_cfg["local_base"]
        num_rope_positions = rope_cfg["num_positions"]
        D = self.head_dim // 2
        for name, theta_val, sz_key, attr in [
            ("ROPE_LOCAL", local_base, "ROPE_LOCAL_SIZE", "DRAM_ADDR_ROPE_LOCAL"),
            ("ROPE_GLOBAL", theta, "ROPE_GLOBAL_SIZE", "DRAM_ADDR_ROPE_GLOBAL"),
        ]:
            inv_freq = 1.0 / (theta_val ** (torch.arange(D, dtype=torch.float32) / D))
            pos = torch.arange(num_rope_positions, dtype=torch.float32)
            freqs = torch.outer(pos, inv_freq)
            cos_ = freqs.cos().to(torch.bfloat16)
            sin_ = freqs.sin().to(torch.bfloat16)
            rope_tensor = torch.cat([cos_, cos_, -sin_, sin_], dim=1)
            sz = self.weight_defs[sz_key]
            raw = rope_tensor.contiguous().view(torch.uint8).numpy().tobytes()
            raw = (raw + b"\x00" * sz)[:sz]
            addr = self.allocate_params_dram(sz)
            self.dma_write(DMA_DEVICE_H2C, addr, raw, sz)
            setattr(self, attr, addr)

    def weight_init(self) -> None:
        """Ensure weight bin exists (generate from HF if missing), load it, then initialize DRAM: embedding, layers from bin, RoPE, OUTPUT_NORM/LM_HEAD."""
        full_path = os.path.join(self.script_dir, self._weights_bin_rel)
        if os.path.exists(full_path):
            print(f"Weight bin exists, skip generation: {full_path}")
        else:
            print(f"Weight bin not found, generating: {full_path}")
            weight_bin_generate(output_path=full_path)
        with open(full_path, "rb") as f:
            self.weight_bin = f.read()
        model, model_dir = _ensure_hf_model(self.script_dir, self._cfg)
        embed = model.get_input_embeddings().weight.detach().cpu().to(torch.bfloat16)
        embedding_scale = model.config.hidden_size ** 0.5
        self.embedding_weight = (embed.float() * embedding_scale).to(torch.bfloat16)
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
        """Initialize hardware DRAM for gemma3 model (layer-wise overlap except for kv cache).
        """
        seq_len = self.MAX_CONTEXT_SIZE
        q_seq_len = seq_len * self.group_size
        aligned_seq_len_q = ((q_seq_len + 63) // 64) * 64

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
        self.IDENTITY_FULL_DRAM_ADDR = self.allocate_tensor_dram(self.head_dim * self.head_dim * self.bytes_per_element)
        self.dma_to_accelerator_memory(self.IDENTITY_FULL_DRAM_ADDR, torch.eye(self.head_dim, dtype=torch.bfloat16))
        # Allocate memory for flash attention and zero pad:
        self.LAYER0_FLASH_Q_DRAM = self.allocate_tensor_dram(aligned_seq_len_q * self.head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_K_DRAM = self.allocate_tensor_dram(aligned_seq_len_q * self.head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_V_DRAM = self.allocate_tensor_dram(aligned_seq_len_q * self.head_dim * self.bytes_per_element)
        zero_pad = torch.zeros(aligned_seq_len_q * self.head_dim * self.bytes_per_element, dtype=torch.bfloat16)
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
        self.LAYER0_FLASH_OUTPUT_DRAM = self.allocate_tensor_dram(aligned_seq_len_q * self.head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_SCRATCH_DRAM = self.allocate_tensor_dram(max(self.head_dim, UE_FMAX_CONTEXT_SIZE) * aligned_seq_len_q * 2 + self.head_dim * aligned_seq_len_q * 2)
        self.LAYER0_FLASH_ATTN_P_DRAM = self.allocate_tensor_dram(aligned_seq_len_q * aligned_seq_len_q * self.bytes_per_element)
        self.LAYER0_FLASH_BIAS_DRAM = self.allocate_tensor_dram(aligned_seq_len_q * aligned_seq_len_q * self.bytes_per_element)
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

    def _compile_prefill_program(self, prefill_seq_len: int, layer_size: int = 26, use_pbi: bool = False) -> dict:
        """
        Compile a single prefill program and return its metadata (start_addr, size_bytes, flops).

        ``prefill_seq_len`` is a **compile-time template value** used only for FLOPs accounting and
        the static ``M=`` / ``seq_len=`` arguments to inner ops (which all use ``gpr_M_reg`` /
        ``gpr_bucket_idx`` for the actual runtime row count, so the static M only affects FLOPs and
        asserts — not the captured program semantics). The captured program reads the *real*
        seq_len at execute time from three GPRs the **caller must prime before entering this
        program**: ``self.gpr_seq_len``, ``self.gpr_q_seq_len``, ``self.gpr_bucket_idx`` (1-based
        bucket selector). This function emits **no** ADD_SETs for these registers, so a single
        cached prefill bin works for any real prefill_seq_len.

        Args:
            prefill_seq_len: Compile-time template (typically ``UE_VECTOR_SIZE``); does not bound
                the runtime seq_len. Drives FLOPs estimate and inner-op static M args only.
            layer_size: Number of transformer layers to compile.
            use_pbi: Use pointer-backed instruction (PBI) descriptors.

        Note:
            **Dual-engine + PBI** has not been verified on hardware since the PBI update. Where
            practical, this compile path still applies the row shard and flag choreography so a
            future dual-PBI bring-up can extend it without rewriting the whole prefill graph.

        Captures into the active capture session; the caller is responsible for
        starting/stopping capture around one or more calls. Each program ends
        with a HALT so it can be jumped to independently at runtime.
        """
        if not getattr(self, "is_capture_on", False):
            raise RuntimeError("_compile_prefill_program() requires an active capture session")
        count_at_start = self.capture_count
        seq_len = prefill_seq_len
        q_seq_len = seq_len * self.group_size
        aligned_seq_len_q = ((q_seq_len + 63) // 64) * 64
        # flash_attention bucket dispatcher uses **1-based** indexing: gpr_bucket_idx = K runs the bucket body sized for seq_len = K * UE_VECTOR_SIZE. 
        bucket_idx = aligned_seq_len_q // UE_VECTOR_SIZE
        engine_master = not self.engine_slave
        row_offset = 0 if engine_master else seq_len // 2
        # Dual-engine: halve per-engine ``seq_len`` and offset DRAM views. Not validated with PBI
        # on hardware yet; kept at compile time for forward compatibility.
        if self.dual_engine:
            seq_len = (seq_len - row_offset if self.engine_slave else row_offset) 

        def shard_dram_addr(base_addr: int, row_offset: int, row_width: int) -> int:
            return base_addr + row_offset * row_width * self.bytes_per_element

        def begin_parallel_stage() -> None:
            if not self.dual_engine:
                return
            if self.engine_slave:
                self.generate_instruction_flag_check(target_engine_idx=0)
                self.generate_instruction_flag_clear()
            else:
                self.generate_instruction_flag_set()
                self.generate_instruction_flag_clear()

        def end_parallel_stage() -> None:
            if not self.dual_engine:
                return
            if self.engine_slave:
                self.generate_instruction_flag_set()
            else:
                self.generate_instruction_flag_check(target_engine_idx=1)

        def partitioned_matmat_mul_core(
            input_dram_addr: int,
            k_dim: int,
            n_dim: int,
            weight_dram_addr: int,
            output_dram_addr: int,
            scale_dram_addr: int,
            **kwargs,
        ) -> int:
            begin_parallel_stage()
            flops = self.matmat_mul_core(
                M=seq_len,
                K=k_dim,
                N=n_dim,
                A_DRAM_ADDR=shard_dram_addr(input_dram_addr, row_offset, k_dim),
                B_DRAM_ADDR=weight_dram_addr,
                OUTPUT_DRAM_ADDR=shard_dram_addr(output_dram_addr, row_offset, n_dim),
                is_B_quantized=True,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=scale_dram_addr,
                gpr_M_reg=self.gpr_seq_len if use_pbi else None,
                **kwargs,
            )
            end_parallel_stage()
            return flops

        def duplicate_gqa_rows_pbi(src_sram_addr: int, dst_dram_addr: int, gpr_seq_len: int = None) -> None:
            row_bytes = self.head_dim * self.bytes_per_element
            row_uram_words = row_bytes // (UE_VECTOR_SIZE * self.bytes_per_element)
            _, src_uram_addr = self.sram_address_to_uram_address(src_sram_addr)
            ptr = self.alloc_inst_ptr()
            self.generate_instruction_pbi_init(
                dram_shared_addr=dst_dram_addr,
                dma_length=row_bytes,
                output_size=0,
                uram_length=0,
                uram_a_start_addr=src_uram_addr,
                uram_b_start_addr=src_uram_addr,
                uram_wb_addr=0,
                uram_dst_addr=0,
                fmax_context_addr=0,
                inst_pointer_idx=ptr,
            )
            self.loop_start(loop_cnt=seq_len, gpr_loop_cnt=gpr_seq_len)
            self.loop_start(self.group_size)
            self.sram_to_accelerator_memory(
                sram_address=0,
                accelerator_dram_address=row_bytes,
                element_size=self.head_dim,
                inst_pointer_idx=ptr,
                memcpy_length_bytes=0,
            )
            self.loop_end()
            self.generate_instruction_pbi_inc(
                dram_shared_addr=0,
                dma_length=0,
                output_size=0,
                uram_length=0,
                uram_a_start_addr=row_uram_words,
                uram_b_start_addr=row_uram_words,
                uram_wb_addr=0,
                uram_dst_addr=0,
                fmax_context_addr=0,
                inst_pointer_idx=ptr,
            )
            self.loop_end()
            self.release_inst_ptr(ptr)

        # --- Gemma3 26 layers: compile---
        global _SILENT_MODE
        _SILENT_MODE = True
        if self.dual_engine:
            self.generate_instruction_flag_clear()
        total_flops = 0
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]

        # NOTE: gpr_seq_len / gpr_q_seq_len / gpr_bucket_idx are primed in the runtime preamble for dynamic seq_len
        for layer_idx in range(layer_size):
            layer_off = layer_idx * LAYER_WEIGHT_SIZE
            if layer_idx != 0 and engine_master:
                # change the first layer input to addr=LAYER0_OUTPUT_DRAM
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, sram_address=0x10000, element_size=self.PREFILL_CONTEXT_SIZE * self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_INPUT_DRAM, element_size=self.PREFILL_CONTEXT_SIZE * self.vector_length)
            if engine_master:
                total_flops += self.rms_norm_core_dram(
                    M=seq_len,
                    N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                    OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                    GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off,
                    gpr_M_reg=self.gpr_seq_len if use_pbi else None,
                )
            # Dual-engine PBI matmul sharding is only partially wired; full validation is TBD.
            total_flops += self.matmat_mul_core(
                M=seq_len,
                K=self.vector_length,
                N=self.head_dim * self.group_size,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                is_B_quantized=True,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off,
                gpr_M_reg=self.gpr_seq_len if use_pbi else None,
            )
            total_flops += self.matmat_mul_core(
                M=seq_len,
                K=self.vector_length,
                N=self.head_dim,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                is_B_quantized=True,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off,
                gpr_M_reg=self.gpr_seq_len if use_pbi else None,
            )
            total_flops += self.matmat_mul_core(
                M=seq_len,
                K=self.vector_length,
                N=self.head_dim,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_V_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size,
                is_B_quantized=True,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off,
                gpr_M_reg=self.gpr_seq_len if use_pbi else None,
            )
            # TODO: OUTPUT_DRAM_ADDR=temp addr. Then memcpy from temp addr to self.LAYER0_V_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size
            if engine_master:
                total_flops += self.rms_norm_core_dram(
                    M=seq_len,
                    N=self.head_dim,
                    A_DRAM_ADDR=self.LAYER0_K_DRAM,
                    OUTPUT_DRAM_ADDR=self.LAYER0_K_NORM_DRAM,
                    GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_NORM_GAMMA + layer_off,
                    gpr_M_reg=self.gpr_seq_len if use_pbi else None,
                )
                total_flops += self.rms_norm_core_dram(
                    M=seq_len * self.group_size,
                    N=self.head_dim,
                    A_DRAM_ADDR=self.LAYER0_Q_DRAM,
                    OUTPUT_DRAM_ADDR=self.LAYER0_Q_NORM_DRAM,
                    GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_NORM_GAMMA + layer_off,
                    gpr_M_reg=self.gpr_q_seq_len if use_pbi else None,
                )

                # ROPE weights are shared between layers
                # TODO: need to enumerate the two cases (global and local) and use jump_abs to switch between them
                ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE_GLOBAL if layer_idx in self._rope_global_layers else self.DRAM_ADDR_ROPE_LOCAL
                total_flops += self.rope_hf_core_dram(
                    M=seq_len,
                    N=self.head_dim,
                    input_dram_addr=self.LAYER0_K_NORM_DRAM,
                    output_dram_addr=self.LAYER0_K_ROPE_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size,
                    cos_dram_addr=ROPE_WEIGHT_ADDR,
                    sin_dram_addr=ROPE_WEIGHT_ADDR + self.head_dim * self.bytes_per_element,
                    gpr_M_reg=self.gpr_seq_len if use_pbi else None,
                )
                # TODO: output_dram_addr= fixed dram addr, then memcpy from temp addr to self.LAYER0_K_ROPE_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size,
                total_flops += self.rope_hf_core_dram_gqa(
                    M=seq_len,
                    group_size=self.group_size,
                    N=self.head_dim,
                    input_dram_addr=self.LAYER0_Q_NORM_DRAM,
                    output_dram_addr=self.LAYER0_FLASH_Q_DRAM,
                    cos_dram_addr=ROPE_WEIGHT_ADDR,
                    sin_dram_addr=ROPE_WEIGHT_ADDR + self.head_dim * self.bytes_per_element,
                    gpr_M_reg=self.gpr_seq_len if use_pbi else None,
                )
                
                # Pre-flash-attn layout:
                # Q: [seq_len, group_size, head_dim], [seq_len:max_seq_len, :] has been padded 0
                # K/V cache: [seq_len, head_dim]; duplicate each token row group_size times for GQA. [seq_len, group_size, head_dim], [seq_len:max_seq_len, :] has been padded 0
                # TODO: generate register for dram addr over layer_idx loop.
                self.accelerator_memory_to_sram(self.LAYER0_K_ROPE_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size, 0x10000, self.PREFILL_CONTEXT_SIZE * self.head_dim)
                duplicate_gqa_rows_pbi(0x10000, self.LAYER0_FLASH_K_DRAM, self.gpr_seq_len if use_pbi else None)

                self.accelerator_memory_to_sram(self.LAYER0_V_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size, 0x20000, self.PREFILL_CONTEXT_SIZE * self.head_dim)
                duplicate_gqa_rows_pbi(0x20000, self.LAYER0_FLASH_V_DRAM, self.gpr_seq_len if use_pbi else None)

                flash_attention_result = self.flash_attention_core(
                    head_dim=self.head_dim,
                    seq_len=aligned_seq_len_q,
                    Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
                    K_DRAM_ADDR=self.LAYER0_FLASH_K_DRAM,
                    V_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                    OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                    SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                    BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_DRAM,
                    IDENTITY_DRAM_ADDR=self.IDENTITY_DRAM_ADDR,
                    ATTN_P_DRAM_ADDR=self.LAYER0_FLASH_ATTN_P_DRAM if use_pbi else None,
                    gpr_bucket_idx=self.gpr_bucket_idx if use_pbi else None,
                    num_buckets=(self.PREFILL_MAX_SEQ_LEN * self.group_size + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE,
                )
                if use_pbi:
                    runtime_bucket_idx = aligned_seq_len_q // UE_VECTOR_SIZE  # 1-indexed
                    total_flops += flash_attention_result[runtime_bucket_idx - 1]
                else:
                    total_flops += flash_attention_result
                use_pbi = True
            total_flops += self.matmat_mul_core(
                M=seq_len,
                K=self.head_dim * self.group_size,
                N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                is_B_quantized=True,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off,
                gpr_M_reg=self.gpr_seq_len if use_pbi else None,
            )
            if engine_master:
                total_flops += self.rms_norm_core_dram(
                    M=seq_len,
                    N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                    OUTPUT_DRAM_ADDR=self.LAYER0_POST_ATTN_NORM_DRAM,
                    GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_NORM_GAMMA + layer_off,
                    gpr_M_reg=self.gpr_seq_len if use_pbi else None,
                )
                total_flops += self.eltwise_core_dram(
                    seq_len,
                    self.vector_length,
                    self.LAYER0_INPUT_DRAM,
                    self.LAYER0_POST_ATTN_NORM_DRAM,
                    self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                    UE_MODE.ELTWISE_ADD,
                    gpr_M_reg=self.gpr_seq_len if use_pbi else None,
                )
                total_flops += self.rms_norm_core_dram(
                    M=seq_len,
                    N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                    OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                    GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off,
                    gpr_M_reg=self.gpr_seq_len if use_pbi else None,
                )
            total_flops += self.matmat_mul_core(
                M=seq_len,
                K=self.vector_length,
                N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                is_B_quantized=True,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off,
                gelu_enable=True,
                gpr_M_reg=self.gpr_seq_len if use_pbi else None,
            )
            total_flops += self.matmat_mul_core(
                M=seq_len,
                K=self.vector_length,
                N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                is_B_quantized=True,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off,
                gpr_M_reg=self.gpr_seq_len if use_pbi else None,
            )
            if engine_master:
                # BF16 multiply of [seq_len, mlp_elements] tensors in DRAM; tiles inside
                # eltwise_core_dram so large prefill (up through model.prefill_max_seq_len, 64-aligned
                # with flash attn) is not limited by staging full tensors in flat URAM.
                total_flops += self.eltwise_core_dram(
                    seq_len,
                    self.mlp_elements,
                    self.LAYER0_MLP_GATE_DRAM,
                    self.LAYER0_MLP_UP_DRAM,
                    self.LAYER0_MLP_MULT_DRAM,
                    UE_MODE.ELTWISE_MUL,
                    gpr_M_reg=self.gpr_seq_len if use_pbi else None,
                )
            total_flops += self.matmat_mul_core(
                M=seq_len,
                K=self.mlp_elements,
                N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                is_B_quantized=True,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off,
                gpr_M_reg=self.gpr_seq_len if use_pbi else None,
            )
            if engine_master:
                total_flops += self.rms_norm_core_dram(
                    M=seq_len,
                    N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                    OUTPUT_DRAM_ADDR=self.LAYER0_POST_MLP_NORM_DRAM,
                    GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_FFW_NORM_GAMMA + layer_off,
                    gpr_M_reg=self.gpr_seq_len if use_pbi else None,
                )
                total_flops += self.eltwise_core_dram(
                    seq_len,
                    self.vector_length,
                    self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                    self.LAYER0_POST_MLP_NORM_DRAM,
                    self.LAYER0_OUTPUT_DRAM,
                    UE_MODE.ELTWISE_ADD,
                    gpr_M_reg=self.gpr_seq_len if use_pbi else None,
                )
        # NOTE: gpr_seq_len holds the current token position and is primed by the runtime preamble.
        # K/V and RoPE DRAM offsets are derived at runtime via MUL_IMM(gpr_seq_len, stride) + ADD_IMM(base).
        self.generate_instruction_halt()
        prefill_program_addr = self.get_program_dram_addr() + count_at_start * INSTRUCTION_SIZE_BYTES
        prefill_program_size = (self.capture_count - count_at_start) * INSTRUCTION_SIZE_BYTES
        _SILENT_MODE = False

        return {
            "prefill_seq_len": prefill_seq_len,
            "start_addr": prefill_program_addr,
            "size_bytes": prefill_program_size,
            "flops": total_flops,
        }
        
    def _compile_decoder_programs(self, layer_size: int = 26, use_pbi: bool = False, profile: bool = False) -> dict:
        """Compile a single decoder program; KV length is selected at runtime via ``gpr_bucket_idx``.

        Grouped attention uses :meth:`decoder_group_attention_core` with ``gpr_bucket_idx`` (same
        1-based convention as prefill flash attention). ``num_buckets`` is derived from
        ``max_context_size`` (``(max_context_size + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE``).
        """
        if not getattr(self, "is_capture_on", False):
            raise RuntimeError("_compile_decoder_programs() requires an active capture session")
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        num_buckets = (self.MAX_CONTEXT_SIZE + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE

        decoder_count_at_start = self.capture_count
        count_at_start = self.capture_count
        total_flops = 0
        checkpoints: list[list] = []

        def _checkpoint(name: str) -> None:
            self.generate_instruction_halt()
            self.pad_capture_to_64b_boundary()
            resume = self.get_program_dram_addr() + self.capture_count * INSTRUCTION_SIZE_BYTES
            checkpoints.append([name, f"0x{resume:X}"])

        global _SILENT_MODE
        _SILENT_MODE = True
        for layer_idx in range(layer_size):
            layer_off = layer_idx * LAYER_WEIGHT_SIZE
            if layer_idx != 0:
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, sram_address=0x10000, element_size=self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_INPUT_DRAM, element_size=self.vector_length)
            total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                          OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off)
            if profile:
                _checkpoint(f"L{layer_idx}_pre_norm")
            total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=self.head_dim * self.group_size,
                                                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                                                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off,
                                                OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                                                data_type=TYPE.IF4,
                                                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off,
                                                )
            total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=self.head_dim,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off,
                )
            total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=self.head_dim,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off,
                )
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_FLASH_V_DRAM, sram_address=0x10000, element_size=self.head_dim)
            self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(self.k_size))
            self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(self.LAYER0_V_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size), self.TMP_REG)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=0, element_size=self.head_dim, general_reg_src=self.TMP_REG)
            if profile:
                _checkpoint(f"L{layer_idx}_qkv_proj_vcache")
            total_flops += self.rms_norm_core_dram(M=1, N=self.head_dim, A_DRAM_ADDR=self.LAYER0_K_DRAM,
                          OUTPUT_DRAM_ADDR=self.LAYER0_K_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_NORM_GAMMA + layer_off)
            total_flops += self.rms_norm_core_dram(M=self.group_size, N=self.head_dim, A_DRAM_ADDR=self.LAYER0_Q_DRAM,
                          OUTPUT_DRAM_ADDR=self.LAYER0_Q_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_NORM_GAMMA + layer_off)

            ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE_GLOBAL if layer_idx in self._rope_global_layers else self.DRAM_ADDR_ROPE_LOCAL
            k_rope_layer_addr = self.LAYER0_K_ROPE_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size
            self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(self.k_size * 2))
            self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(ROPE_WEIGHT_ADDR), self.TMP_REG)
            total_flops += self.rope_hf_core_decode(N=self.head_dim, input_dram_addr=self.LAYER0_K_NORM_DRAM, output_dram_addr=k_rope_layer_addr,
                    gr_weight_dram=self.TMP_REG)
            self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(self.k_size))
            self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(k_rope_layer_addr), self.TMP_REG)
            self.accelerator_memcpy(k_rope_layer_addr, 0, self.k_size, gr_dst_addr=self.TMP_REG)
            self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gpr_seq_len, ue_35bit_addr_shifter(self.k_size * 2))
            self.generate_instruction_add_imm(self.TMP_REG, ue_35bit_addr_shifter(ROPE_WEIGHT_ADDR), self.TMP_REG)
            for g in range(self.group_size):
                total_flops += self.rope_hf_core_decode(N=self.head_dim, input_dram_addr=self.LAYER0_Q_NORM_DRAM + g * self.head_dim * self.bytes_per_element, output_dram_addr=self.LAYER0_FLASH_Q_DRAM + g * self.head_dim * self.bytes_per_element,
                        gr_weight_dram=self.TMP_REG)
            if profile:
                _checkpoint(f"L{layer_idx}_qk_norm_rope")
            attn_result = self.decoder_group_attention_core(
                group_size=self.group_size,
                head_dim=self.head_dim,
                seq_len=UE_VECTOR_SIZE,
                Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
                K_DRAM_ADDR=self.LAYER0_K_ROPE_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size,
                V_DRAM_ADDR=self.LAYER0_V_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size,
                OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                IDENTITY_DRAM_ADDR=self.IDENTITY_DRAM_ADDR,
                SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_DRAM,
                gpr_bucket_idx=self.gpr_bucket_idx if use_pbi else None,
                num_buckets=num_buckets,
            )
            total_flops += attn_result[-1] if use_pbi else attn_result
            if profile:
                _checkpoint(f"L{layer_idx}_attention")
            total_flops += self.quantized_matmat_core(M=1, K=self.head_dim * self.group_size, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off,
                )
            total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                          OUTPUT_DRAM_ADDR=self.LAYER0_POST_ATTN_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_NORM_GAMMA + layer_off)

            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_INPUT_DRAM, sram_address=0x10000, element_size=self.vector_length)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_NORM_DRAM, sram_address=0x90000, element_size=self.vector_length)
            self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, element_size=self.vector_length)
            if profile:
                _checkpoint(f"L{layer_idx}_o_proj_post_attn_norm_residual")

            total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                          OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off)
            if profile:
                _checkpoint(f"L{layer_idx}_pre_ffn_norm")

            total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off,
                gelu_enable=True,
                )
            total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off,
                )

            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_GATE_DRAM, sram_address=0x10000, element_size=self.mlp_elements)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_UP_DRAM, sram_address=0x90000, element_size=self.mlp_elements)
            self.eltwise_mul_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.mlp_elements)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_MLP_MULT_DRAM, element_size=self.mlp_elements)
            if profile:
                _checkpoint(f"L{layer_idx}_mlp_gateup_gelu_mul")

            total_flops += self.quantized_matmat_core(M=1, K=self.mlp_elements, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off,
                )
            total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                          OUTPUT_DRAM_ADDR=self.LAYER0_POST_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_FFW_NORM_GAMMA + layer_off)

            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, sram_address=0x10000, element_size=self.vector_length)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_MLP_NORM_DRAM, sram_address=0x90000, element_size=self.vector_length)
            self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, element_size=self.vector_length)
            if profile:
                _checkpoint(f"L{layer_idx}_mlp_down_post_ffn_norm_residual")

        if layer_size == self.LAYER_SIZE:
            total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_OUTPUT_DRAM,
                OUTPUT_DRAM_ADDR=self.OUTPUT_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_OUTPUT_NORM_GAMMA)
            total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=self.EMBEDDING_ELEMENTS,
                A_DRAM_ADDR=self.OUTPUT_NORM_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_QUANT,
                OUTPUT_DRAM_ADDR=self.LOGITS_DRAM,
                data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_SCALE,
                write_back_disable=True,
                )

        # Advance token position; K/V/RoPE offsets are derived as gpr_seq_len * stride at each usage site.
        self.generate_instruction_add_inc(self.gpr_seq_len)

        self.generate_instruction_halt()
        inst_count = self.capture_count - count_at_start
        _SILENT_MODE = False
        decoder_program_addr = self.get_program_dram_addr() + decoder_count_at_start * INSTRUCTION_SIZE_BYTES
        program_size_bytes = inst_count * INSTRUCTION_SIZE_BYTES
        print(
            f"Decoder program compiled at 0x{decoder_program_addr:X}, single segment with "
            f"PBI grouped attention ({num_buckets} attention buckets), size {program_size_bytes} bytes"
        )
        return {
            "program_size_bytes": program_size_bytes,
            "total_flops": total_flops,
            "checkpoints": checkpoints,
        }

    def compile_gemma3(self, layer_size: int = 26, use_pbi: bool = False, slave_engine = None, profile: bool = False) -> None:
        """Compile a single prefill program (seq_len-agnostic) and one decoder program into a
        combined instruction image.

        Layout in program DRAM (after loading the combined instruction image):
          [prefill][decoder]

        Prefill is compiled with a fixed template ``prefill_seq_len = UE_VECTOR_SIZE`` (only used
        for FLOPs accounting and static inner-op ``M=`` args — captured ops drive their loop counts
        off ``gpr_M_reg`` / ``gpr_bucket_idx`` at runtime, so the bin is valid for any real
        seq_len). The runtime preamble in :meth:`run_gemma3` primes three GPRs
        (gpr_seq_len, gpr_q_seq_len, gpr_bucket_idx) before entering the cached prefill program,
        so the same bin works across all prompt lengths and we only need to compile once.

        The decoder program is also captured once; grouped attention uses ``gpr_bucket_idx`` (same
        1-based convention as prefill flash attention). Each decode step rebuilds a tiny dispatch
        stub that sets ``gpr_bucket_idx`` then jumps into the cached decoder program.

        If both the bin and meta sidecar already exist, this is a no-op (reuse the cached image).

        Writes:
          - paths.instruction_bin : raw 32 B/instruction stream (HALT appends a trailing NOP when
            needed so each segment length is a multiple of 64 B; see :meth:`UnifiedEngine.generate_instruction_halt`).
          - paths.instruction_meta : per-stage start addresses, sizes, FLOPs.
        """
        if profile:
            instruction_bin_path  = os.path.join(self.script_dir, "gemma3_bin/gemma3_profile_instruction.bin")
            instruction_meta_path = os.path.join(self.script_dir, "gemma3_bin/gemma3_profile_instruction.json")
        else:
            instruction_bin_path  = os.path.join(self.script_dir, self._cfg["paths"]["instruction_bin"])
            instruction_meta_path = os.path.join(self.script_dir, self._cfg["paths"]["instruction_meta"])
        if os.path.exists(instruction_bin_path) and os.path.exists(instruction_meta_path):
            print(f"Reusing existing instruction image at {instruction_bin_path}")
            print(f"  delete {instruction_bin_path} to force recompile.")
            return

        # Fixed compile-time template; runtime preamble overrides the actual seq_len via GPRs.
        prefill_seq_len = UE_VECTOR_SIZE

        self.clear_inst_id()
        self.start_capture()
        print(f"Compiling prefill (seq_len-agnostic; template={prefill_seq_len})...")
        compile_t0 = time.perf_counter()
        prefill_prog = self._compile_prefill_program(
            prefill_seq_len=prefill_seq_len, layer_size=layer_size, use_pbi=True
        )
        print(f"  prefill compiled, size={prefill_prog['size_bytes']} bytes, "
              f"elapsed={time.perf_counter() - compile_t0:.1f}s")
        decoder_program = self._compile_decoder_programs(layer_size=layer_size, use_pbi=True, profile=profile)
        self.stop_capture()

        os.makedirs(os.path.dirname(instruction_bin_path), exist_ok=True)
        instruction_bytes = bytearray()
        for inst in self.capture_buffer:
            instruction_bytes.extend(inst.get_bytes())
        assert len(instruction_bytes) % 64 == 0, (
            "combined instruction image must be 64-byte aligned (HALT emits trailing NOP via generate_instruction_halt)"
        )
        with open(instruction_bin_path, "wb") as f:
            f.write(instruction_bytes)
        self.clear_capture_buffer()

        # Start address of the single decoder segment (after prefill in the combined image).
        instruction_base_addr = self.get_program_dram_addr()
        prefill_program_addr = instruction_base_addr
        decoder_program_addr = instruction_base_addr + prefill_prog["size_bytes"]

        # Only fields the compiler decides at capture time go in the meta (start addresses,
        # sizes, FLOPs). The prefill bin is seq_len-agnostic, so no prefill_seq_len cross-check is
        # stored — the runtime preamble in run_gemma3 sets the actual seq_len via GPRs.
        metadata = {
            "instruction_bin": os.path.relpath(instruction_bin_path, self.script_dir),
            "instruction_base_addr": f"0x{instruction_base_addr:X}",
            "instruction_total_size": len(instruction_bytes),
            "prefill_template_seq_len": prefill_seq_len,
            "prefill_program_start_addr": f"0x{prefill_program_addr:X}",
            "prefill_program_size": prefill_prog["size_bytes"],
            "prefill_template_flops": prefill_prog["flops"],
            "decoder_program_start_addr": f"0x{decoder_program_addr:X}",
            "decoder_program_size": decoder_program["program_size_bytes"],
            "decoder_total_flops": decoder_program["total_flops"],
        }
        if profile:
            metadata["decoder_profile_checkpoints"] = decoder_program["checkpoints"]

        if slave_engine is not None:
            # Dual-engine slave compiles the same single prefill program.
            slave_engine.clear_inst_id()
            slave_engine.start_capture()
            slave_prefill_prog = slave_engine._compile_prefill_program(
                prefill_seq_len=prefill_seq_len, layer_size=layer_size, use_pbi=use_pbi
            )
            slave_engine.stop_capture()
            slave_bin_path = os.path.join(self.script_dir, "gemma3_bin", "prefill_program_slave.bin")
            slave_program_bytes = bytearray()
            for inst in slave_engine.capture_buffer:
                slave_program_bytes.extend(inst.get_bytes())
            assert len(slave_program_bytes) % 64 == 0, (
                "slave prefill instruction image must be 64-byte aligned"
            )
            with open(slave_bin_path, "wb") as f:
                f.write(slave_program_bytes)
            slave_engine.clear_capture_buffer()
            slave_prefill_addr = slave_engine.get_program_dram_addr()
            metadata["dual_engine_slave_prefill"] = {
                "bin_path": os.path.relpath(slave_bin_path, self.script_dir),
                "start_addr": f"0x{slave_prefill_addr:X}",
                "flops": slave_prefill_prog["flops"],
            }

        with open(instruction_meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Combined gemma3 instruction image written to {instruction_bin_path} ({len(instruction_bytes)} bytes)")
        print(f"  prefill: seq_len-agnostic (template={prefill_seq_len}), {prefill_prog['size_bytes']} bytes")
        print(f"  decoder: {decoder_program['program_size_bytes']} bytes")
        print(f"Metadata written to {instruction_meta_path}")

    def _decode_profile_execute(self, preamble_addr: int, checkpoints: list, timeout: float = 30.0) -> list:
        """Execute one decoder step through profile checkpoints; return per-step HW latencies."""
        results = []
        self.start_execute_from_dram(preamble_addr)
        for name, resume_addr_hex in checkpoints:
            self.wait_queue(timeout)
            results.append((name, self.report_latency_in_us() / 1e3))
            self.start_execute_from_dram(int(resume_addr_hex, 16))
        self.wait_queue(timeout)
        results.append(("output_norm_lm_head", self.report_latency_in_us() / 1e3))
        return results

    def run_gemma3_profile(self) -> None:
        """Load the profile instruction image, run prefill + one profiled decoder step,
        and print a per-step latency breakdown for the decoder.
        """
        profile_meta_path = os.path.join(self.script_dir, "gemma3_bin/gemma3_profile_instruction.json")
        with open(profile_meta_path, "r") as f:
            meta = json.load(f)

        self.load_program_instructions_from_file(os.path.join(self.script_dir, meta["instruction_bin"]))
        preamble_addr = self.get_program_dram_addr()

        prefill_program_addr   = _parse_offset(meta["prefill_program_start_addr"])
        decoder_program_addr   = _parse_offset(meta["decoder_program_start_addr"])
        flops_prefill_template = meta["prefill_template_flops"]
        template_prefill_seq_len = int(meta["prefill_template_seq_len"])
        checkpoints            = meta["decoder_profile_checkpoints"]
        _max_gpr_bucket = (self.MAX_CONTEXT_SIZE + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE

        prefill_seq = self.prefill_seq or tuple(self._cfg["default_prefill_tokens"])
        prefill_seq = prefill_seq[:-1]
        prefill_seq_len = len(prefill_seq)
        self.seq_len = prefill_seq_len

        q_seq_len = prefill_seq_len * self.group_size
        aligned_seq_len_q = ((q_seq_len + 63) // 64) * 64
        bucket_idx = aligned_seq_len_q // UE_VECTOR_SIZE
        flops_prefill = flops_prefill_template * prefill_seq_len // max(template_prefill_seq_len, 1)

        self.clear_inst_id()
        self.start_capture()
        self.generate_instruction_add_set(self.gpr_seq_len, prefill_seq_len)
        self.generate_instruction_add_set(self.gpr_q_seq_len, q_seq_len)
        self.generate_instruction_add_set(self.gpr_bucket_idx, bucket_idx)
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(prefill_program_addr))
        self.stop_capture()
        self.write_captured_instructions_to_dram(preamble_addr)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        self.clear_capture_buffer()

        embedding_tensor = self.get_embedding_for_tokens(prefill_seq)
        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
        bias_one_group = torch.full((aligned_seq_len_q, aligned_seq_len_q), float("-inf"), dtype=torch.bfloat16)
        valid_mask = torch.tril(torch.ones(aligned_seq_len_q, aligned_seq_len_q, dtype=torch.bool))
        bias_one_group.masked_fill_(valid_mask, 0.0)
        bias_one_group[:, q_seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_one_group)

        print(f"\n--- Profiling: prefill (seq_len={prefill_seq_len}) ---")
        timer = time.perf_counter()
        self.program_execute(preamble_addr, flops=flops_prefill)
        print(f"Prefill done in {time.perf_counter() - timer:.2f}s")

        # Decoder preamble for the first decode token
        token_id = prefill_seq[-1]
        self.seq_len += 1
        aligned_dec = ((self.seq_len + 63) // 64) * 64
        bucket_idx = min(aligned_dec // UE_VECTOR_SIZE, _max_gpr_bucket)
        embedding_tensor = self.get_embedding_for_tokens([token_id])
        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
        bias_host = torch.full((1, aligned_dec), -1e36, dtype=torch.bfloat16)
        bias_host[0, :self.seq_len] = 0.0
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_host)

        self.clear_inst_id()
        self.start_capture()
        self.generate_instruction_add_set(self.gpr_bucket_idx, bucket_idx)
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(decoder_program_addr))
        self.stop_capture()
        self.write_captured_instructions_to_dram(preamble_addr)
        self.clear_capture_buffer()

        print("\n--- Profiling: decoder step breakdown ---")
        results = self._decode_profile_execute(preamble_addr, checkpoints)

        total_ms = sum(ms for _, ms in results)
        from collections import defaultdict
        step_totals: dict = defaultdict(float)
        for name, ms in results:
            step = name.split("_", 1)[1] if "_" in name else name
            step_totals[step] += ms

        print(f"\n{'Step (all layers)':<35} {'ms':>9}  {'%':>6}")
        print("-" * 54)
        for step, ms in step_totals.items():
            print(f"{step:<35} {ms:>9.2f}  {ms/total_ms*100:>5.1f}%")
        print("-" * 54)
        print(f"{'Total':<35} {total_ms:>9.2f}  100.0%")
        print(f"\nDecode speed (HW): {1000/total_ms:.2f} tok/s  ({total_ms:.1f} ms/tok)")

        print(f"\n{'Step':<40} {'ms':>8}")
        print("-" * 50)
        for name, ms in results:
            print(f"{name:<40} {ms:>8.2f}")

    def run_gemma3(self, slave_engine = None) -> dict:
        """Load the unified instruction image once, then run prefill + decoder loop.

        The cached gemma3_instruction.bin is seq_len-agnostic; the runtime prefill_seq_len is
        applied via a small preamble program compiled fresh per run that primes three GPRs
        (gpr_seq_len, gpr_q_seq_len, gpr_bucket_idx) and then unconditional-jumps into the cached
        prefill program.

        Each decode token captures the same short dispatch stub (``gpr_bucket_idx`` +
        jump into the cached decoder program), DMAs it over the **same** program-DRAM words
        as the prefill preamble (``preamble_addr``), and executes from that address again.
        The single ``allocate_program_dram`` after the preamble already reserved that
        region; the allocator tail is not moved for decode. ``decoder_total_flops`` is a
        single template value for GFLOPS reporting.
        """
        meta_path = os.path.join(self.script_dir, self._cfg["paths"]["instruction_meta"])
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.load_program_instructions_from_file(os.path.join(self.script_dir, meta["instruction_bin"]))
        preamble_addr = self.get_program_dram_addr()

        prefill_program_addr = _parse_offset(meta["prefill_program_start_addr"])
        # FLOPs in meta are scaled to compile-time template seq_len; we rescale per actual seq_len below.
        flops_prefill_template = meta["prefill_template_flops"]
        template_prefill_seq_len = int(meta["prefill_template_seq_len"])

        decoder_program_addr = _parse_offset(meta["decoder_program_start_addr"])
        decoder_flops_per_token = meta["decoder_total_flops"]
        _max_gpr_bucket = (self.MAX_CONTEXT_SIZE + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE

        prefill_seq = self.prefill_seq
        if prefill_seq is None:
            prefill_seq = tuple(self._cfg["default_prefill_tokens"])
        if len(prefill_seq) <= 1:
            raise ValueError("Prefill sequence must have at least 2 tokens.")
        # Prefill program processes all but the last token (the last one feeds the decoder).
        prefill_seq = prefill_seq[:-1]
        prefill_seq_len = len(prefill_seq)
        self.seq_len = prefill_seq_len

        # Rough FLOPs rescale (linear in seq_len for the bulk of the work; a couple of softmax /
        # quadratic terms in the compiled template don't scale cleanly, but this is fine for
        # back-of-envelope GFLOPS reporting).
        flops_prefill = flops_prefill_template * prefill_seq_len // max(template_prefill_seq_len, 1)

        slave_prefill_flops = None
        slave_prefill_addr = None
        slave_prefill = meta.get("dual_engine_slave_prefill")
        if slave_engine is not None and slave_prefill is not None:
            slave_bin_path = os.path.join(self.script_dir, slave_prefill["bin_path"])
            slave_engine.reset_program_dram_addr()
            slave_engine.load_program_instructions_from_file(slave_bin_path)
            slave_prefill_addr = _parse_offset(slave_prefill["start_addr"])
            slave_prefill_flops = slave_prefill["flops"]

        print(f"\n--- Starting prefill (seq_len={prefill_seq_len}) ---")
        print(f"Prompt ({len(self.prefill_seq)}) tokens: {self.prefill_seq}")
        timer = time.perf_counter()
        if slave_prefill_addr is not None:
            slave_engine.program_execute(slave_prefill_addr, timeout=0)

        seq_len = prefill_seq_len
        q_seq_len = seq_len * self.group_size
        aligned_seq_len_q = ((q_seq_len + 63) // 64) * 64
        bucket_idx = aligned_seq_len_q // UE_VECTOR_SIZE  # 1-based, matches flash_attention dispatcher

        # ----- Runtime preamble: prime three GPRs, then jump into the cached prefill -----
        self.clear_inst_id()
        self.start_capture()
        self.generate_instruction_add_set(self.gpr_seq_len, prefill_seq_len)
        self.generate_instruction_add_set(self.gpr_q_seq_len, q_seq_len)
        self.generate_instruction_add_set(self.gpr_bucket_idx, bucket_idx)
        # Unconditional absolute jump into the cached prefill program.
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(prefill_program_addr))
        self.stop_capture()
        self.write_captured_instructions_to_dram(preamble_addr)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        self.clear_capture_buffer()

        embedding_tensor = self.get_embedding_for_tokens(prefill_seq)
        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
        bias_one_group = torch.full((aligned_seq_len_q, aligned_seq_len_q), float("-inf"), dtype=torch.bfloat16)
        valid_mask = torch.tril(torch.ones(aligned_seq_len_q, aligned_seq_len_q, dtype=torch.bool), diagonal=0) if not self.causal_mask_upper else torch.triu(torch.ones(aligned_seq_len_q, aligned_seq_len_q, dtype=torch.bool), diagonal=0)
        bias_one_group.masked_fill_(valid_mask, 0.0)
        bias_one_group[:, q_seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_one_group)
        # Execute from the preamble; preamble jumps into cached prefill, which HALTs on completion.
        latency_hw_prefill, flop_rate_hw_prefill = self.program_execute(preamble_addr, flops=flops_prefill)
        latency_prefill = time.perf_counter() - timer
        if slave_prefill_flops is not None:
            print(f"Dual engine prefill gflops: {((flops_prefill + slave_prefill_flops) / (latency_hw_prefill * 1e3)):.2f} GFLOPS")
        print(f"Prefill execute done in {latency_prefill:.2f} seconds, start decoding...\n")

        print(f"\n--- Starting decoder ---")
        timer = time.perf_counter()
        token_id = self.prefill_seq[-1]
        global _SILENT_MODE
        max_seq_len = self.MAX_CONTEXT_SIZE
        total_latency, total_flop_rate = 0, 0
        decoder_token_cnt = 0
        hw_decode_lats_us: list[float] = []

        # Two-region live counter: pin the bottom terminal row as a status line via
        # an ANSI scroll region; tokens stream in the area above it and the counter
        # refreshes in place. Only when stdout is a real TTY (skip when piped/redirected).
        import shutil
        _use_status = sys.stdout.isatty()
        def _status_setup():
            rows = shutil.get_terminal_size().lines
            sys.stdout.write(f"\033[1;{rows - 1}r")   # scroll region = rows 1..rows-1
            sys.stdout.write(f"\033[{rows - 1};1H")   # park cursor at bottom of region
            sys.stdout.flush()
        def _status_update():
            rows = shutil.get_terminal_size().lines
            elapsed = time.perf_counter() - timer
            rate = decoder_token_cnt / elapsed if elapsed > 0 else 0.0
            sys.stdout.write("\0337")                 # save cursor
            sys.stdout.write(f"\033[{rows};1H\033[2K") # bottom row, clear it
            sys.stdout.write(f" decoding… {decoder_token_cnt} tokens  (pos {self.seq_len}/{max_seq_len})  "
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

        while self.seq_len < max_seq_len:
            decoder_token_cnt += 1
            _SILENT_MODE = True
            self.seq_len += 1
            aligned_seq_len_q = ((self.seq_len + 63) // 64) * 64
            bucket_idx = min(aligned_seq_len_q // UE_VECTOR_SIZE, _max_gpr_bucket)
            flops_hw = (
                decoder_flops_per_token
                if isinstance(decoder_flops_per_token, (int, float))
                else None
            )

            # Host interuption only for each decoded token to handle argmax result
            embedding_tensor = self.get_embedding_for_tokens([token_id])
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
            bias_host = torch.full((1, aligned_seq_len_q), -1e36, dtype=torch.bfloat16)
            bias_host[0, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_host)

            self.clear_inst_id()
            self.start_capture()
            self.generate_instruction_add_set(self.gpr_bucket_idx, bucket_idx)
            self.generate_instruction_jump_abs(ue_35bit_addr_shifter(decoder_program_addr))
            self.stop_capture()
            self.write_captured_instructions_to_dram(preamble_addr)
            self.clear_capture_buffer()

            latency, flop_rate_program = self.program_execute(preamble_addr, flops=flops_hw)
            hw_decode_lats_us.append(latency)
            total_latency += latency
            total_flop_rate += flop_rate_program
            token_id = self.get_arg_max_index()
            token_char = self.tokenizer.decode([token_id])
            _SILENT_MODE = False
            if token_id in [1, self._end_of_turn_token_id]:
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

        latency_decoder = time.perf_counter() - timer
        tokens_decoded = decoder_token_cnt
        print(f"\nDecoder done in {latency_prefill + latency_decoder:.2f}s, "
              f"speed: {tokens_decoded / latency_decoder:.2f} tokens/s, "
              f"total {self.seq_len} tokens.")

        hw_decode_avg_ms  = sum(hw_decode_lats_us) / len(hw_decode_lats_us) / 1e3 if hw_decode_lats_us else 0.0
        hw_decode_first_ms = hw_decode_lats_us[0] / 1e3 if hw_decode_lats_us else 0.0
        cpu_decode_avg_ms = latency_decoder * 1e3 / tokens_decoded if tokens_decoded else 0.0
        _original_print("\n=== Performance Summary ===")
        _original_print(f"Instruction size  : prefill={meta['prefill_program_size']/1024:.1f} kB  decoder={meta['decoder_program_size']/1024:.1f} kB  total={(meta['prefill_program_size']+meta['decoder_program_size'])/1024:.1f} kB")
        _original_print(f"Prefill ({prefill_seq_len} tokens): HW={latency_hw_prefill/1e3:,.1f} ms  CPU={latency_prefill*1e3:,.1f} ms")
        _original_print(f"Decode 1st token  : HW={hw_decode_first_ms:,.1f} ms/tok  ({1000/hw_decode_first_ms:.2f} tok/s)")
        _original_print(f"Decode  ({tokens_decoded} tokens): HW={hw_decode_avg_ms:,.1f} ms/tok  CPU={cpu_decode_avg_ms:,.1f} ms/tok  ({tokens_decoded/latency_decoder:.2f} tok/s)")
        
# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def _clock_ns_default_for_device(device: str) -> float:
    """Return default clock period (ns) for FPGA type — mirrors user_hw_test.py."""
    if device == "kintex7":                       return 5.1594
    if device in ("rk", "puzhi"):                 return 3.0
    if device in ("bittware", "bittware_256"):     return 3.3333
    if device == "alveo":                          return 4.0
    return 10.0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Gemma3 layer-0 prefill: run on accelerator, verify with torch ref.")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt: tokenizer encodes this to prefill_seq (overrides default)")
    parser.add_argument("--local-weights", action="store_true", help="Use gemma3_bin/full_model_weights.bin instead of generated weights_gemma3_hf.bin")
    parser.add_argument(
        "--dual-engine",
        action="store_true",
        help="Dual-engine path (compile-time sharding hooks exist; PBI + dual not verified end-to-end—CLI still rejects).",
    )
    parser.add_argument('--dev', type=str, default='xdma0',
                        help='DMA device name (e.g., xdma0, xdma1). Default: xdma0')
    parser.add_argument(
        '--device',
        type=str,
        default='kintex7',
        help='FPGA board / bitstream profile (same as user_hw_test.py): affects UE_AXI_DATA_WIDTH_BITS and default --cycle.',
    )
    parser.add_argument(
        '--cycle',
        type=float,
        default=None,
        help='Clock cycle time in nanoseconds. Default: from --device (kintex7=5.1594ns, bittware=3.3333ns, rk/puzhi=3.0ns, alveo=4.0ns).',
    )
    parser.add_argument('--profile', action='store_true',
                        help='Compile a profile binary with per-step HALT checkpoints and run one decode step to measure per-step latency breakdown.')
    args = parser.parse_args()

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
    print(f"Using DMA device: {args.dev}")
    print(f"  H2C: {DMA_DEVICE_H2C}")
    print(f"  C2H: {DMA_DEVICE_C2H}")
    print(f"  USER: {DMA_DEVICE_USER}")

    dual_engine = args.dual_engine
    assert dual_engine == False, (
        "Dual-engine Gemma3 PBI is not verified end-to-end yet; compile preserves sharding hooks for "
        "future work. Re-run without --dual-engine until validation lands."
    )
    ue = Gemma3_UnifiedEngine(local_weights=args.local_weights, dual_engine=dual_engine)
    ue.set_prefill_seq(args.prompt)

    if dual_engine:
        ue2 = Gemma3_UnifiedEngine(local_weights=args.local_weights, dual_engine=True, engine_slave=True)
        ue2.set_prefill_seq(args.prompt)

    print(f"\n--- Compiling ---")
    timer = time.perf_counter()
    ue.compile_gemma3(slave_engine=ue2 if dual_engine else None)
    print(f"Compile done in {time.perf_counter() - timer:.2f} seconds")

    if args.profile:
        print("\n--- Compiling profile binary ---")
        timer = time.perf_counter()
        ue.compile_gemma3(profile=True)
        print(f"Profile compile done in {time.perf_counter() - timer:.2f} seconds")
        ue.run_gemma3_profile()
        ue.clear_dram()
        print("Decoder profile done.")
        return

    run_result = ue.run_gemma3(slave_engine=ue2 if dual_engine else None)
    print("Gemma3 test ends.")

    ue.clear_dram()
    global _SILENT_MODE
    _SILENT_MODE = True
    ue.software_reset()

if __name__ == "__main__":
    main()
