#!/usr/bin/env python3
"""
Qwen3-4B inference on accelerator: prefill + decode.

  - Config from qwen3_4b_bin/qwen3_4b_config.json; weights from a single bin (see below).
  - Prefill: compiled each run. Decoder: if qwen3_4b_bin/decoder_program.bin and
    qwen3_4b_bin/decoder_program.json exist, skip decoder compile and load
    program sizes from meta; otherwise compile and write the bin + meta.
  - Run prefill then decode loop. For numeric verification use qwen3_4b_numeric.py.

Architecture notes vs Gemma3:
  - 36 layers, 32 Q heads / 8 KV heads (group_size=4), actual head_dim=128.
  - hidden_size=2560, mlp_intermediate=9728, vocab=151936.
  - QK RMSNorm per head (gamma_offset=0.0, like Gemma3 but no +1).
  - No post-attention norm; residual applied directly to o_proj output (like LLaMA).
  - No post-FFN norm; residual applied directly to down_proj output (like LLaMA).
  - post_attention_layernorm in HF IS the pre-FFN norm (like LLaMA).
  - Embedding NOT scaled by sqrt(hidden_size) (like LLaMA).
  - Separate lm_head weight (NOT tied to embedding).
  - SwiGLU activation: silu_enable=True on gate_proj (like LLaMA).
  - Single RoPE base theta=1_000_000 (no local/global split).
  - RoPE applied per head (N=128) without lo|hi permutation.

Weights:
  - Default: qwen3_4b_bin/weights_qwen3_4b_hf.bin (generated from HF model if missing).
  - --local-weights: use qwen3_4b_bin/full_model_weights.bin instead.

Usage:
  python qwen3_4b_test.py
  python qwen3_4b_test.py --prompt "your prompt"
  python qwen3_4b_test.py --dev xdma0 [--cycle 5.88]
  python qwen3_4b_test.py --local-weights
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
from user_dma_core import DMA_DEVICE_H2C, TYPE, UE_FMAX_CONTEXT_SIZE, UE_MODE, UE_VECTOR_SIZE, UE_ARGMAX_INDEX, set_dma_device, ue_35bit_addr_shifter
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

# down_proj K=6144 ≤ SCALE_BRAM_ELEMENTS=8192 — no K-split needed.
# (Previously was 8960 which required splitting; mlp_elements corrected to 6144.)

def _parse_offset(val) -> int:
    """Parse offset/size from JSON: int or hex string like '0x24000000'."""
    if isinstance(val, str):
        return int(val, 0)
    return int(val)

def _quantize_bf16_to_int4_packed(weight_bf16: torch.Tensor, block_size: int = 64) -> tuple[bytes, bytes]:
    """Quantize bf16 weight (N_w, K_w) to INT4 packed + scale per block of 64 along K. Returns (data_bytes, scale_bytes).
    Scale is stored with negative sign so HW IF4 path dispatches as INT4 (sign(bf16 scale)=neg → INT4 codebook)."""
    w = weight_bf16.detach().cpu().float().reshape(-1)
    N_w, K_w = weight_bf16.shape
    assert K_w % block_size == 0
    w_blocks = w.reshape(N_w, K_w // block_size, block_size)
    scale = w_blocks.abs().amax(dim=-1).clamp(min=1e-8) / 7.0
    scale_bf16 = (-scale).to(torch.bfloat16)
    w_int8 = (w_blocks / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8)
    w_nibbles = w_int8.numpy().astype(np.int16) & 0x0F
    low = w_nibbles[:, :, 0::2].reshape(N_w, -1)
    high = w_nibbles[:, :, 1::2].reshape(N_w, -1)
    packed = (high << 4) | low
    data_bytes = packed.astype(np.uint8).tobytes()
    scale_bytes = scale_bf16.contiguous().view(torch.uint8).numpy().tobytes()
    return (data_bytes, scale_bytes)

def weight_bin_generate(script_dir: str | None = None, output_path: str | None = None) -> str:
    """Generate weights_qwen3_4b_hf.bin from Hugging Face model per qwen3_4b_config.json layout.
    Returns the path to the written file. Use this bin to replace full_model_weights.bin."""
    script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
    cfg = _load_config(script_dir)
    weight_defs = cfg["_weight_defs"]
    paths = cfg["paths"]
    paths_full = os.path.join(script_dir, paths["weights_bin"])
    out_path = output_path or paths_full

    model, model_dir = _ensure_hf_model(script_dir, cfg)
    gamma_offset = cfg["special"]["rms_norm"]["gamma_offset"]  # 0.0 for Qwen3
    emb_cfg = cfg["special"]["embedding"]
    token_embd_offset = _parse_offset(emb_cfg["token_embd_offset"])
    token_embd_size = _parse_offset(emb_cfg["token_embd_size"])
    LAYER_WEIGHT_SIZE = weight_defs["LAYER_WEIGHT_SIZE"]
    base_layer0 = weight_defs["BLK0_ATTN_NORM_WEIGHT"]
    num_layers = cfg["file_info"]["num_layers"]
    actual_head_dim = cfg["file_info"]["actual_head_dim"]
    blk0_structure = cfg["layers"]["structure"]

    # Compute total file size: max(offset + size) over all regions
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
    raw_emb = embed.contiguous().view(torch.uint8).numpy().tobytes()
    write_at(token_embd_offset, raw_emb)

    # Layers
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn

        # Qwen3 norms: gamma_offset=0.0 (stored as-is)
        gamma_in = (layer.input_layernorm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
        q_w = attn.q_proj.weight.detach().cpu().to(torch.bfloat16)
        k_w = attn.k_proj.weight.detach().cpu().to(torch.bfloat16)
        v_w = attn.v_proj.weight.detach().cpu().to(torch.bfloat16)
        # Qwen3 has QK norm per actual head dim (128); store as (actual_head_dim,)
        gamma_q = (attn.q_norm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
        gamma_k = (attn.k_norm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
        o_w = attn.o_proj.weight.detach().cpu().to(torch.bfloat16)
        # Qwen3: no post-attention norm — write zero placeholder (pipeline skips this step)
        gamma_post = torch.zeros(cfg["file_info"]["hidden_size"], dtype=torch.bfloat16)
        # Qwen3: post_attention_layernorm IS the pre-FFN norm (same as LLaMA)
        gamma_ffn = (layer.post_attention_layernorm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
        gate_w = layer.mlp.gate_proj.weight.detach().cpu().to(torch.bfloat16)
        up_w = layer.mlp.up_proj.weight.detach().cpu().to(torch.bfloat16)
        down_w = layer.mlp.down_proj.weight.detach().cpu().to(torch.bfloat16)
        # Qwen3: no post-FFN norm — write zero placeholder (pipeline skips this step)
        gamma_post_ffn = torch.zeros(cfg["file_info"]["hidden_size"], dtype=torch.bfloat16)

        # K=6144 ≤ SCALE_BRAM_ELEMENTS=8192 — single down_proj (no split needed)
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

    # ROPE: single base theta for all layers (rope_global_layers is empty)
    # Per-head frequencies: D_per_head = actual_head_dim // 2 = 64
    rope_cfg = cfg["special"]["rope"]
    theta = rope_cfg["theta"]
    num_positions = rope_cfg["num_positions"]
    D_per_head = actual_head_dim // 2  # 64
    inv_freq = 1.0 / (theta ** (torch.arange(D_per_head, dtype=torch.float32) / D_per_head))
    pos = torch.arange(num_positions, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)         # (num_positions, 64)
    cos_ = freqs.cos().to(torch.bfloat16)      # (num_positions, 64)
    sin_ = freqs.sin().to(torch.bfloat16)      # (num_positions, 64)
    # rope_hf_core(N=actual_head_dim=128) layout: [cos(64), cos(64), -sin(64), sin(64)]
    rope_tensor = torch.cat([cos_, cos_, -sin_, sin_], dim=1)  # (num_positions, 256)
    for off_key, sz_key in [("ROPE_LOCAL", "ROPE_LOCAL_SIZE"), ("ROPE_GLOBAL", "ROPE_GLOBAL_SIZE")]:
        sz = weight_defs[sz_key]
        raw = (rope_tensor.contiguous().view(torch.uint8).numpy().tobytes() + b"\x00" * sz)[:sz]
        write_at(weight_defs[off_key], raw)

    # OUTPUT_NORM
    out_norm = (model.model.norm.weight.detach().cpu().to(torch.bfloat16).float() + gamma_offset).to(torch.bfloat16)
    sz = weight_defs["OUTPUT_NORM_WEIGHT_SIZE"]
    raw = (out_norm.contiguous().view(torch.uint8).numpy().tobytes() + b"\x00" * sz)[:sz]
    write_at(weight_defs["OUTPUT_NORM_WEIGHT"], raw)

    # LM_HEAD: Qwen3 has a separate lm_head weight (NOT tied to embedding)
    lm_head_w = model.lm_head.weight.detach().cpu().to(torch.bfloat16)
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

def _load_config(script_dir: str) -> dict:
    """Load qwen3_4b_config.json and build weight_defs (offset/size dict) from regions."""
    config_path = os.path.join(script_dir, "qwen3_4b_config.json")
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
# Qwen3-4B unified engine
# -----------------------------------------------------------------------------
class Qwen3_4b_UnifiedEngine(UnifiedEngine):
    """UnifiedEngine for Qwen3-4B: loads config + weight bin, compile_prefill/compile_decoder, run_prefill/run_decoder.

    Key architectural differences from Gemma3:
      - QK RMSNorm per actual head dim (128), gamma_offset=0.0.
      - No post-attention norm: residual applied directly to o_proj output.
      - No post-FFN norm: residual applied directly to down_proj output.
      - post_attention_layernorm in HF IS the pre-FFN norm.
      - Embedding NOT scaled by sqrt(hidden_size).
      - Separate lm_head weight (not tied to embedding).
      - SwiGLU (SiLU gate): silu_enable=True on gate_proj.
      - Single RoPE theta=1_000_000 for all layers; rope applied per head (N=128).
      - Per-KV-head flash attention (8 KV heads x 128 dim); group_size=2.
    """

    def __init__(self, script_dir: str | None = None, hf_model_dir: str | None = None, weights_bin: str | None = None):
        # Qwen3-4B DRAM layout (weights ~2.15 GB → needs more params region than 1.7B):
        #   params:       0x00000000 – 0x90000000 (~2.25 GB)
        #   tensors:      0x90000000 – 0xE0000000 (~1.25 GB, KV cache + activations)
        #   instructions: 0xE0000000 – 0x100000000 (~512 MB, compiled bin)
        super().__init__(
            params_dram_base=0x00000000,
            tensor_dram_base=0x90000000,
            program_dram_base=0xE0000000,
        )
        self.script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
        self._cfg = _load_config(self.script_dir)
        self.weight_defs = self._cfg["_weight_defs"]

        fi = self._cfg["file_info"]
        model = self._cfg["model"]
        paths = self._cfg["paths"]

        self.vector_length = fi["hidden_size"]           # 2048
        self.head_dim = fi["head_dim"]                   # 1024 (= num_kv_heads * actual_head_dim)
        self.actual_head_dim = fi["actual_head_dim"]     # 128
        self.num_kv_heads = fi["num_kv_heads"]           # 8
        self.bytes_per_element = fi["bytes_per_element"] # 2
        self.group_size = fi["group_size"]               # 2 (Q heads per KV head)
        self.mlp_elements = fi["mlp_elements"]           # 8960
        self.hf_model_dir = hf_model_dir or os.path.join(self.script_dir, paths["hf_model_dir"])
        # q_size = total Q output dim * bpe = (num_kv_heads * actual_head_dim * group_size) * bpe
        self.q_size = self.head_dim * self.group_size * self.bytes_per_element   # 1024*2*2 = 4096
        self.k_size = self.head_dim * self.bytes_per_element                     # 1024*2 = 2048
        self.MAX_CONTEXT_SIZE = model["max_context_size"]
        self.PREFILL_MAX_SEQ_LEN = int(model.get("prefill_max_seq_len", 256))
        self.LAYER_SIZE = fi["num_layers"]
        self.EMBEDDING_ELEMENTS = fi["embedding_vocab"]
        # Dynamic-PBI GPR layout (see core_changes.md §3a):
        #   reg 1 = TMP_REG          — scratch for reg_mul_imm + add_imm address math
        #   reg 2 = GF_SEQ_LEN_REG   — runtime row count for matmul/norm/rope (M=seq_len ops)
        #   reg 3 = GF_Q_SEQ_LEN_REG — for ops with M = seq_len * group_size (Q-side norms/rope)
        #   reg 4 = GF_BUCKET_IDX_REG— 1-based bucket selector for flash / group-attention kernels
        # Dynamic GPR allocation (via alloc_isa_reg) starts at 5; PBI op-internal loop counters
        # consume from there and release back at loop_end.
        fixed = self._cfg.get("fixed_isa_regs", {})
        self.TMP_REG       = fixed["TMP_REG"]
        self.gf_seq_len    = fixed["GF_SEQ_LEN_REG"]
        self.gf_q_seq_len  = fixed["GF_Q_SEQ_LEN_REG"]
        self.gf_bucket_idx = fixed["GF_BUCKET_IDX_REG"]
        self._isa_reg_counter = 5
        self.causal_mask_upper = False
        self._end_of_turn_token_id = model["end_of_turn_token_id"]

        bin_path = weights_bin or paths["weights_bin"]
        full_path = os.path.join(self.script_dir, bin_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Bin file not found: {full_path}")
        with open(full_path, "rb") as f:
            self.weight_bin = f.read()
        self.weight_init()
        self.tensor_init()

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

    def get_embedding_for_tokens(self, token_ids: list[int] | tuple) -> torch.Tensor:
        """Return (len(token_ids), vector_length) bfloat16 tensor from self.embedding_weight (no scaling)."""
        tid_t = torch.tensor(token_ids, dtype=torch.long)
        out = torch.zeros(len(token_ids), self.vector_length, dtype=torch.bfloat16)
        valid = tid_t < self.embedding_weight.shape[0]
        out[valid] = self.embedding_weight[tid_t[valid]]
        return out

    def _load_rope_host(self, rope_theta: float | None = None) -> None:
        """Generate per-head RoPE (cos, cos, -sin, sin) on host and write to DRAM.
        D_per_head = actual_head_dim // 2 = 64. Both ROPE_LOCAL and ROPE_GLOBAL use the
        same theta (rope_global_layers is empty for Qwen3)."""
        rope_cfg = self._cfg["special"]["rope"]
        theta = rope_theta if rope_theta is not None else rope_cfg["theta"]
        num_rope_positions = rope_cfg["num_positions"]
        D_per_head = self.actual_head_dim // 2   # 64
        inv_freq = 1.0 / (theta ** (torch.arange(D_per_head, dtype=torch.float32) / D_per_head))
        pos = torch.arange(num_rope_positions, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)                     # (num_positions, 64)
        cos_ = freqs.cos().to(torch.bfloat16)
        sin_ = freqs.sin().to(torch.bfloat16)
        # rope_hf_core(N=actual_head_dim=128): [cos(64), cos(64), -sin(64), sin(64)]
        rope_tensor = torch.cat([cos_, cos_, -sin_, sin_], dim=1)  # (num_positions, 256)
        # Single DMA: write ROPE_LOCAL and ROPE_GLOBAL back-to-back (same table, same theta)
        rope_raw = rope_tensor.contiguous().view(torch.uint8).numpy().tobytes()
        local_sz  = self.weight_defs["ROPE_LOCAL_SIZE"]
        global_sz = self.weight_defs["ROPE_GLOBAL_SIZE"]
        local_raw  = (rope_raw + b"\x00" * local_sz)[:local_sz]
        global_raw = (rope_raw + b"\x00" * global_sz)[:global_sz]
        rope_buf = local_raw + global_raw
        rope_base = self.allocate_params_dram(len(rope_buf))
        self.dma_write(DMA_DEVICE_H2C, rope_base, rope_buf, len(rope_buf))
        self.DRAM_ADDR_ROPE_LOCAL  = rope_base
        self.DRAM_ADDR_ROPE_GLOBAL = rope_base + local_sz

    def weight_init(self) -> None:
        """Initialize DRAM from weight bin: load embedding+tokenizer offline from
        cached files, layers from bin, host-computed RoPE, then OUTPUT_NORM/LM_HEAD
        from bin. The embedding is read straight out of ``self.weight_bin`` (no HF
        model needed at this stage); the tokenizer is loaded with
        ``local_files_only=True``. ``main()`` runs ``weight_bin_generate`` first
        on a fresh machine — it handles the HF download — so by the time we get
        here both the weight bin and the tokenizer files exist locally."""
        emb_cfg = self._cfg["special"]["embedding"]
        token_embd_offset = _parse_offset(emb_cfg["token_embd_offset"])
        vocab_size = emb_cfg["vocab_size"]
        emb_dim = emb_cfg["embedding_dim"]
        emb_bytes = vocab_size * emb_dim * self.bytes_per_element
        raw_emb = bytearray(self.weight_bin[token_embd_offset : token_embd_offset + emb_bytes])
        self.embedding_weight = torch.frombuffer(raw_emb, dtype=torch.bfloat16).reshape(vocab_size, emb_dim).clone()
        model_dir = os.path.join(self.script_dir, self._cfg["paths"]["hf_model_dir"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True, local_files_only=True,
        )

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
        assert layer0_end <= LAYER_WEIGHT_SIZE, (
            f"Layer 0 size mismatch: computed {layer0_end} > LAYER_WEIGHT_SIZE {LAYER_WEIGHT_SIZE}"
        )

        import time as _time
        print(f"\n--- Weights DRAM allocation, start at DRAM address: {self.get_params_dram_addr()} ---")
        layers_total = self.LAYER_SIZE * LAYER_WEIGHT_SIZE
        layers_base_dram = self.allocate_params_dram(layers_total)

        # Single large DMA for all layer weights: the bin file stores layers at base_layer0
        # with the same LAYER_WEIGHT_SIZE stride as DRAM, so data maps directly.
        # This avoids 28*20=560 small DMA calls (each with PCIe round-trip overhead).
        bin_layers_start = base_layer0
        print(f"  DMA layers: {layers_total // 1024 // 1024} MB → DRAM 0x{layers_base_dram:X}...", flush=True)
        _t0 = _time.perf_counter()
        self.dma_write(DMA_DEVICE_H2C, layers_base_dram,
                       self.weight_bin[bin_layers_start : bin_layers_start + layers_total],
                       layers_total)
        print(f"  DMA layers done in {_time.perf_counter() - _t0:.1f}s")
        # Set layer-0 DRAM attribute addresses from known offsets (same for all layers)
        for off_key, sz_key, attr in blk0_regions:
            offset_in_layer = self.weight_defs[off_key] - base_layer0
            setattr(self, attr, layers_base_dram + offset_in_layer)
        print(f"Layers 0..{self.LAYER_SIZE - 1} loaded: 0x{layers_base_dram:X} size {layers_total} (LAYER_WEIGHT_SIZE={LAYER_WEIGHT_SIZE})")

        # Single large DMA for all non-layer weights (OUTPUT_NORM + LM_HEAD).
        # Assemble a contiguous buffer then transfer in one shot.
        nl_slices = [self.weight_bin[self.weight_defs[k] : self.weight_defs[k] + self.weight_defs[s]]
                     for k, s, _ in non_layer]
        nl_buf = b"".join(nl_slices)
        print(f"  DMA non-layer weights: {len(nl_buf) // 1024 // 1024} MB...", flush=True)
        _t0 = _time.perf_counter()
        nl_base_dram = self.allocate_params_dram(len(nl_buf))
        self.dma_write(DMA_DEVICE_H2C, nl_base_dram, nl_buf, len(nl_buf))
        print(f"  DMA non-layer done in {_time.perf_counter() - _t0:.1f}s")
        nl_offset = 0
        for off_key, sz_key, attr in non_layer:
            setattr(self, attr, nl_base_dram + nl_offset)
            nl_offset += self.weight_defs[sz_key]

        self._load_rope_host()
        print(f"    Allocate weights end at DRAM address: 0x{self.get_params_dram_addr():X}, usage: {self.get_params_dram_usage()} bytes")
        print("Tokenizer loaded successfully.")

    def tensor_init(self) -> None:
        """Initialize hardware DRAM tensors for Qwen3-4B.

        KV cache layout (per layer, per KV head):
          LAYER0_V_DRAM[layer][kv_h][t]  (MAX_CONTEXT_SIZE * actual_head_dim per head)
          LAYER0_K_ROPE_DRAM[layer][kv_h][t]  (same shape)
        KV cache is placed last in the activation region so it can grow with max_context_size.
        """
        seq_len = self.MAX_CONTEXT_SIZE
        # Qwen3: q_seq_len = seq_len * group_size (2 Q heads per KV head)
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        ahd = self.actual_head_dim   # 128
        nkvh = self.num_kv_heads     # 8
        bpe = self.bytes_per_element

        print(f"Allocate tensor dram start at DRAM address: 0x{self.get_tensor_dram_addr():X}")

        # --- Fixed tensors (reused per layer, do not grow with KV cache) ---
        # Constant zero and identity buffers
        zero_add = torch.zeros(seq_len * self.head_dim * bpe, dtype=torch.bfloat16)
        self.ZERO_DRAM_ADDR = self.allocate_tensor_dram(seq_len * self.head_dim * bpe)
        self.dma_to_accelerator_memory(self.ZERO_DRAM_ADDR, zero_add)
        # 64×64 identity for decoder_attention_core / decoder_group_attention_core
        # (legacy and PBI paths still read this slot for their I @ V^T tile).
        self.IDENTITY_DRAM_ADDR = self.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe)
        self.dma_to_accelerator_memory(self.IDENTITY_DRAM_ADDR, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))
        # Flash-attention bucket-dispatcher scratch buffer. Sized for the
        # largest aligned_seq_len_q this engine instance might see:
        # PREFILL_MAX_SEQ_LEN * group_size, rounded up to UE_VECTOR_SIZE.
        aligned_q_max = ((self.PREFILL_MAX_SEQ_LEN * self.group_size + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
        self.LAYER0_FLASH_ATTN_P_DRAM = self.allocate_tensor_dram(aligned_q_max * aligned_q_max * bpe)

        # Per-head flash attention buffers: one KV head at a time, reused across heads
        # FLASH_Q: (q_seq_len_aligned, ahd) for one KV head's Q group (group_size=2 Q heads)
        self.LAYER0_FLASH_Q_DRAM = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        self.LAYER0_FLASH_K_DRAM = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        self.LAYER0_FLASH_V_DRAM = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        zero_pad = torch.zeros(aligned_seq_len * ahd * bpe, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_Q_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_K_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_V_DRAM, zero_pad)

        # Per-head flash output (seq_len * group_size, ahd); reused across KV heads
        self.LAYER0_FLASH_OUT_HEAD_DRAM = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        # Final assembled flash output (seq_len, head_dim * group_size) = (seq_len, 2048)
        self.LAYER0_FLASH_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.head_dim * self.group_size * bpe)
        self.LAYER0_FLASH_SCRATCH_DRAM = self.allocate_tensor_dram(max(ahd, UE_FMAX_CONTEXT_SIZE) * aligned_seq_len * 2 + ahd * aligned_seq_len * 2)
        self.LAYER0_FLASH_BIAS_DRAM = self.allocate_tensor_dram(aligned_seq_len * aligned_seq_len * bpe)

        # Layer intermediate tensors
        self.LAYER0_INPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_Q_DRAM = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_K_DRAM = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_K_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_Q_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.q_size)
        # V proj temp buffer: standard layout (seq_len, head_dim) = (seq_len, nkvh * ahd)
        self.LAYER0_V_PROJ_TEMP = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_ATTN_PROJ_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        # POST_ATTN_NORM_DRAM: Qwen3 has no post-attn norm; allocated but unused in pipeline
        self.LAYER0_POST_ATTN_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_ATTN_RESIDUAL_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_MLP_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_MLP_GATE_DRAM = self.allocate_tensor_dram(seq_len * self.mlp_elements * 2)
        self.LAYER0_MLP_UP_DRAM = self.allocate_tensor_dram(seq_len * self.mlp_elements * 2)
        self.LAYER0_MLP_MULT_DRAM = self.allocate_tensor_dram(seq_len * self.mlp_elements * 2)
        self.LAYER0_MLP_DOWN_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        # K=6144 fits in SCALE_BRAM directly — no split buffers needed
        # POST_MLP_NORM_DRAM: Qwen3 has no post-FFN norm; allocated but unused in pipeline
        self.LAYER0_POST_MLP_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.OUTPUT_NORM_DRAM = self.allocate_tensor_dram(1 * self.vector_length * bpe)
        self.LOGITS_DRAM = self.allocate_tensor_dram(1 * self.EMBEDDING_ELEMENTS * bpe)

        # --- KV cache (at tail of activation region, grows with MAX_CONTEXT_SIZE) ---
        # Per-head contiguous layout: [layer][kv_head][position] = ahd elements per position
        kv_cache_total = self.LAYER_SIZE * nkvh * self.MAX_CONTEXT_SIZE * ahd * bpe
        self.LAYER0_V_DRAM = self.allocate_tensor_dram(kv_cache_total)
        self.LAYER0_K_ROPE_DRAM = self.allocate_tensor_dram(kv_cache_total)
        zero_kv = torch.zeros(kv_cache_total, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, zero_kv)
        self.dma_to_accelerator_memory(self.LAYER0_K_ROPE_DRAM, zero_kv)

        print(f"    Allocate tensor dram end at DRAM address: 0x{self.get_tensor_dram_addr():X}, usage: {self.get_tensor_dram_usage()} bytes")

    def program_execute(self, program_start_addr: int = user_dma_core.DRAM_INSTRUCTION_ADDR, timeout: float = 120.0, gflops: float = None) -> None:
        """Execute compiled program from DRAM instruction memory.
        """
        print(f"Execute program start at 0x{program_start_addr:X}")
        self.start_execute_from_dram(program_start_addr)
        self.wait_queue(timeout)
        latency = self.report_latency_in_us()
        print(f"    Total program execution latency = {latency} us")
        if gflops is not None:
            gflops_program, _ = self.report_flop_rate_gflops(gflops)
            # High-precision so callers can sanity-check vs peak ( = 128 / clock_ns ):
            # if num_flops or clock period is mis-set the reported GFLOPS will drift.
            print(f"Report FLOPS for program execution: {gflops_program:.4f} GFLOPS "
                  f"(num_flops={int(gflops)}, latency_us={latency:.1f}, clock_ns={self._clock_period_ns})")

    def reset_isa_reg_counter(self) -> None:
        """Reset PBI ISA-reg allocator back to 5 (regs 1-4 are fixed/reserved)."""
        self._isa_reg_counter = 5

    def alloc_isa_reg(self, reset: bool = False) -> int:
        """Allocate the next free ISA register. Used by PBI ops for loop counters."""
        if reset:
            self._isa_reg_counter = 5
        if self._isa_reg_counter > 15:
            raise ValueError("Exceeded available ISA registers (max 15)")
        reg_idx = self._isa_reg_counter
        self._isa_reg_counter += 1
        return reg_idx

    def _emit_pbi_scatter_per_token(self, *, read_base, read_stride_bytes,
                                    write_specs, sram_byte_addr, element_count,
                                    gf_seq_len, template_seq_len):
        """Emit one PBI runtime loop that staged-copies one ``element_count``-row
        per outer iteration from ``read_base`` to each (base, stride) in
        ``write_specs``. The outer trip count is taken from GPR ``gf_seq_len`` so
        the body executes exactly ``actual_seq_len`` times at runtime — making
        the captured bin truly seq_len-agnostic up to ``MAX_CONTEXT_SIZE``.

        Read side uses register-computed addresses (``reg_mul_imm`` + ``add_imm``
        + ``general_reg_src=TMP_REG``) — the same pattern used by the decoder
        bin. The write side uses gemma3-style ``pbi_init`` pointers + per-call
        DRAM-delta DMAs, which is the proven SRAM→DRAM PBI scatter shape.

        Per-iteration t-counter is a locally-allocated GPR that increments by 1
        at end-of-body via ``add_inc``. Released after the loop.
        """
        bpe = self.bytes_per_element
        bytes_per_call = element_count * bpe
        _, sram_words = self.sram_address_to_uram_address(sram_byte_addr)

        # Allocate write PBI pointers (one per destination stream).
        ptr_ws = [self.alloc_inst_ptr() for _ in write_specs]
        for ptr_w, (dst_base, _stride) in zip(ptr_ws, write_specs):
            self.generate_instruction_pbi_init(
                dram_shared_addr=dst_base,
                dma_length=bytes_per_call,
                uram_a_start_addr=sram_words,
                uram_b_start_addr=sram_words,
                inst_pointer_idx=ptr_w,
            )

        # Per-token counter t (0, 1, 2, ...) — used to compute read DRAM addr.
        t_reg = self.alloc_isa_reg()
        self.generate_instruction_add_set(t_reg, 0)

        self.loop_start(loop_cnt=template_seq_len, gf_loop_cnt=gf_seq_len)
        # Read DRAM addr = read_base + t * read_stride_bytes
        self.generate_instruction_reg_mul_imm(
            self.TMP_REG, t_reg, ue_35bit_addr_shifter(read_stride_bytes))
        self.generate_instruction_add_imm(
            self.TMP_REG, ue_35bit_addr_shifter(read_base), self.TMP_REG)
        # DRAM→SRAM with runtime-computed source DRAM addr (delivered via TMP_REG).
        self.accelerator_memory_to_sram(
            accelerator_dram_address=0,
            sram_address=sram_byte_addr,
            element_size=element_count,
            general_reg_src=self.TMP_REG,
        )
        # SRAM→DRAM via PBI pointers (each advances its DRAM addr by its stride).
        for ptr_w, (_base, dst_stride) in zip(ptr_ws, write_specs):
            self.sram_to_accelerator_memory(
                sram_address=0,
                accelerator_dram_address=dst_stride,
                element_size=element_count,
                inst_pointer_idx=ptr_w,
                memcpy_length_bytes=0,
            )
        # t += 1
        self.generate_instruction_add_inc(t_reg)
        self.loop_end()

        self.release_isa_reg()  # t_reg
        for ptr in reversed(ptr_ws):
            self.release_inst_ptr(ptr)

    def _emit_prefill_program(self, seq_len: int, layer_size: int) -> int:
        """Emit ONE seq_len-agnostic prefill program (no capture-session boundary).

        Caller wraps this in start_capture()/stop_capture(). The emitted program
        ends with a halt so it can be jumped to via a runtime preamble that
        primes gf_seq_len / gf_q_seq_len / gf_bucket_idx GPRs.

        ``seq_len`` here is a **compile-time template** used only for FLOPS
        bookkeeping and as the static ``M=`` arg to PBI ops (overridden at
        runtime by ``gf_M_reg``). K/V/Q scatter, FLASH_K/V duplication, and the
        FLASH_OUTPUT assembly are emitted as **PBI runtime loops** keyed off
        ``gf_seq_len``, so the bin is truly seq_len-agnostic up to
        ``MAX_CONTEXT_SIZE`` regardless of the template value.

        All bulk ops (rms_norm, matmat, rope, eltwise) are PBI-dispatched via
        gf_M_reg=self.gf_seq_len. Per-token outer loops for matmul/norm/rope
        run gf_seq_len iterations at runtime regardless of the template value.
        Flash attention dispatches on gf_bucket_idx.
        """
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        # 1-based bucket selector at the static template; runtime overrides via gf_bucket_idx.
        num_buckets_prefill = max(1, aligned_seq_len // UE_VECTOR_SIZE)

        ahd  = self.actual_head_dim   # 128
        nkvh = self.num_kv_heads      # 8
        qpkv = self.group_size        # 2 (Q heads per KV head)
        bpe  = self.bytes_per_element
        hd   = self.head_dim          # 1024
        ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE_LOCAL   # single base for all layers

        # Reset dynamic GPR allocator so PBI-op-internal loop counters don't
        # accumulate across consecutive layers. (Also done in compile_instructions
        # but defensively reset here too.)
        self._isa_reg_counter = 5

        total_flops = 0
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        for layer_idx in range(layer_size):
            _original_print(f"    prefill layer {layer_idx + 1}/{layer_size}", end="\r", flush=True)
            layer_off = layer_idx * LAYER_WEIGHT_SIZE
            if layer_idx != 0:
                # Inter-layer copy: previous layer's OUTPUT → next layer's INPUT.
                # Per-token PBI loop (one row of vector_length per iter) to avoid the
                # URAM-A overflow that a single seq_len*vector_length SRAM stage hits
                # once seq_len * vector_length * bpe > 512 KB (URAM_A capacity).
                self._emit_pbi_scatter_per_token(
                    read_base=self.LAYER0_OUTPUT_DRAM,
                    read_stride_bytes=self.vector_length * bpe,
                    write_specs=[(self.LAYER0_INPUT_DRAM, self.vector_length * bpe)],
                    sram_byte_addr=0x10000,
                    element_count=self.vector_length,
                    gf_seq_len=self.gf_seq_len,
                    template_seq_len=seq_len,
                )

            # Pre-norm (input_layernorm) — M=seq_len rows at runtime via gf_seq_len
            total_flops += (self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off,
                              gf_M_reg=self.gf_seq_len) or 0)

            # Q, K, V projections — M=seq_len rows at runtime
            total_flops += (self.matmat_mul_core(M=seq_len, K=self.vector_length, N=hd * qpkv,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off, gf_M_reg=self.gf_seq_len) or 0)
            total_flops += (self.matmat_mul_core(M=seq_len, K=self.vector_length, N=hd,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off, gf_M_reg=self.gf_seq_len) or 0)
            total_flops += (self.matmat_mul_core(M=seq_len, K=self.vector_length, N=hd,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_V_PROJ_TEMP,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off, gf_M_reg=self.gf_seq_len) or 0)

            # QK RMSNorm per head — M = seq_len * nkvh and M = seq_len * nkvh * qpkv.
            # Compute M for each into TMP_REG via reg_mul_imm(gf_seq_len, multiplier).
            self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gf_seq_len, nkvh)
            total_flops += (self.rms_norm_core_dram(M=seq_len * nkvh, N=ahd, A_DRAM_ADDR=self.LAYER0_K_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_K_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_NORM_GAMMA + layer_off,
                              gf_M_reg=self.TMP_REG) or 0)
            self.generate_instruction_reg_mul_imm(self.TMP_REG, self.gf_seq_len, nkvh * qpkv)
            total_flops += (self.rms_norm_core_dram(M=seq_len * nkvh * qpkv, N=ahd, A_DRAM_ADDR=self.LAYER0_Q_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_Q_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_NORM_GAMMA + layer_off,
                              gf_M_reg=self.TMP_REG) or 0)

            # RoPE per head per token via rope_hf_core_dram_gqa:
            # K_NORM layout per token: nkvh groups sharing one cos/sin row.
            # Q_NORM layout per token: (nkvh*qpkv) groups sharing one cos/sin row.
            # Runtime row count = gf_seq_len tokens.
            total_flops += self.rope_hf_core_dram_gqa(
                M=seq_len, group_size=nkvh, N=ahd,
                input_dram_addr=self.LAYER0_K_NORM_DRAM,
                output_dram_addr=self.LAYER0_K_NORM_DRAM,
                cos_dram_addr=ROPE_WEIGHT_ADDR,
                sin_dram_addr=ROPE_WEIGHT_ADDR + ahd * bpe,
                gf_M_reg=self.gf_seq_len)
            total_flops += self.rope_hf_core_dram_gqa(
                M=seq_len, group_size=nkvh * qpkv, N=ahd,
                input_dram_addr=self.LAYER0_Q_NORM_DRAM,
                output_dram_addr=self.LAYER0_Q_NORM_DRAM,
                cos_dram_addr=ROPE_WEIGHT_ADDR,
                sin_dram_addr=ROPE_WEIGHT_ADDR + ahd * bpe,
                gf_M_reg=self.gf_seq_len)

            # Per-KV-head: scatter K/V to cache + flash buffers, scatter Q, then flash_attention.
            # All per-token scatters are PBI runtime loops (gf_seq_len trips) — the bin is
            # truly seq_len-agnostic up to MAX_CONTEXT_SIZE; no PREFILL_MAX_SEQ_LEN cap.
            for kv_h in range(nkvh):
                k_cache_base = (self.LAYER0_K_ROPE_DRAM
                                + layer_idx * nkvh * self.MAX_CONTEXT_SIZE * ahd * bpe
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)
                v_cache_base = (self.LAYER0_V_DRAM
                                + layer_idx * nkvh * self.MAX_CONTEXT_SIZE * ahd * bpe
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)

                # K scatter: K_NORM[t][kv_h] → K cache[kv_h][t] + FLASH_K[t*qpkv+g] for g in 0..qpkv-1
                k_write_specs = [(k_cache_base, ahd * bpe)]
                for g in range(qpkv):
                    k_write_specs.append((self.LAYER0_FLASH_K_DRAM + g * ahd * bpe, qpkv * ahd * bpe))
                self._emit_pbi_scatter_per_token(
                    read_base=self.LAYER0_K_NORM_DRAM + kv_h * ahd * bpe,
                    read_stride_bytes=nkvh * ahd * bpe,
                    write_specs=k_write_specs,
                    sram_byte_addr=0x10000,
                    element_count=ahd,
                    gf_seq_len=self.gf_seq_len,
                    template_seq_len=seq_len,
                )

                # V scatter: V_PROJ_TEMP[t][kv_h] → V cache[kv_h][t] + FLASH_V[t*qpkv+g]
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

                # Q scatter: Q_NORM[t][kv_h*qpkv:kv_h*qpkv+qpkv] → FLASH_Q[t*qpkv:(t+1)*qpkv]
                # Per token, copy qpkv contiguous Q rows (qpkv*ahd elements) in one DMA.
                self._emit_pbi_scatter_per_token(
                    read_base=self.LAYER0_Q_NORM_DRAM + kv_h * qpkv * ahd * bpe,
                    read_stride_bytes=nkvh * qpkv * ahd * bpe,
                    write_specs=[(self.LAYER0_FLASH_Q_DRAM, qpkv * ahd * bpe)],
                    sram_byte_addr=0x30000,
                    element_count=qpkv * ahd,
                    gf_seq_len=self.gf_seq_len,
                    template_seq_len=seq_len,
                )

                # Flash attention for this KV head (head_dim=128, GQA group_size=2).
                # Bucket dispatcher: bin contains num_buckets_prefill bodies (one per
                # 64-token block of aligned Q seq), runtime gf_bucket_idx GPR selects.
                # The new flash kernel inlines V^T materialization via PBI matmul —
                # no host-side identity DMA, so cache-hit replay is deterministic.
                flash_result = self.flash_attention_core(
                    head_dim=ahd,
                    seq_len=aligned_seq_len,
                    Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
                    K_DRAM_ADDR=self.LAYER0_FLASH_K_DRAM,
                    V_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                    OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUT_HEAD_DRAM,
                    SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                    BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_DRAM,
                    ATTN_P_DRAM_ADDR=self.LAYER0_FLASH_ATTN_P_DRAM,
                    gf_bucket_idx=self.gf_bucket_idx,
                    num_buckets=num_buckets_prefill,
                )
                # PBI flash returns a per-bucket FLOPS list; pick the bucket the
                # template targets for FLOPS bookkeeping.
                total_flops += (flash_result[num_buckets_prefill - 1]
                                if isinstance(flash_result, (list, tuple)) else (flash_result or 0))

                # Assemble per-head output into FLASH_OUTPUT_DRAM. Per token, copy
                # qpkv contiguous rows from FLASH_OUT_HEAD[t*qpkv:(t+1)*qpkv] to
                # FLASH_OUTPUT[t][kv_h*qpkv:(kv_h+1)*qpkv]. PBI runtime loop.
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

            # o_proj
            total_flops += (self.matmat_mul_core(M=seq_len, K=hd * qpkv, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off, gf_M_reg=self.gf_seq_len) or 0)

            # Qwen3: no post-attention norm; add residual directly to o_proj output.
            # PBI runtime loop (one row per iter, gf_seq_len trips) so seq_len > URAM-A capacity is safe.
            self.eltwise_core_dram(
                M=seq_len, N=self.vector_length,
                dram_a=self.LAYER0_INPUT_DRAM,
                dram_b=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                dram_out=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                mode=UE_MODE.ELTWISE_ADD,
                gf_M_reg=self.gf_seq_len,
            )

            # Qwen3: post_attention_layernorm IS the pre-FFN norm
            total_flops += (self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off,
                              gf_M_reg=self.gf_seq_len) or 0)

            # MLP: gate_proj with SiLU, up_proj, gate x up element-wise, down_proj
            total_flops += (self.matmat_mul_core(M=seq_len, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off, silu_enable=True,
                gf_M_reg=self.gf_seq_len) or 0)
            total_flops += (self.matmat_mul_core(M=seq_len, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off,
                gf_M_reg=self.gf_seq_len) or 0)

            # gate × up: PBI runtime loop, one row of mlp_elements per iter (gf_seq_len trips).
            self.eltwise_core_dram(
                M=seq_len, N=self.mlp_elements,
                dram_a=self.LAYER0_MLP_GATE_DRAM,
                dram_b=self.LAYER0_MLP_UP_DRAM,
                dram_out=self.LAYER0_MLP_MULT_DRAM,
                mode=UE_MODE.ELTWISE_MUL,
                gf_M_reg=self.gf_seq_len,
            )

            # down_proj: K=6144 ≤ SCALE_BRAM_ELEMENTS=8192, single call
            total_flops += (self.matmat_mul_core(M=seq_len, K=self.mlp_elements, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off,
                gf_M_reg=self.gf_seq_len) or 0)

            # Qwen3: no post-FFN norm; add residual directly to down_proj output
            # Post-MLP residual: layer_output = POST_ATTN_RESIDUAL + MLP_DOWN. PBI runtime loop.
            self.eltwise_core_dram(
                M=seq_len, N=self.vector_length,
                dram_a=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                dram_b=self.LAYER0_MLP_DOWN_DRAM,
                dram_out=self.LAYER0_OUTPUT_DRAM,
                mode=UE_MODE.ELTWISE_ADD,
                gf_M_reg=self.gf_seq_len,
            )

        # Per-bucket halt so each bucket-program is independently executable.
        self.generate_instruction_halt()
        return total_flops

    def run_prefill(self, prefill_program_addr: int, preamble_addr: int,
                    prefill_seq, gflops: int = None) -> dict:
        """Run prefill via dynamic-PBI runtime preamble.

        Emits a tiny preamble program at ``preamble_addr`` that primes the three
        runtime GPRs (gf_seq_len, gf_q_seq_len, gf_bucket_idx) and then
        unconditional-jumps into the cached prefill program. Single cached bin
        handles any actual_seq_len ≤ PREFILL_MAX_SEQ_LEN — no padding needed.

        Args:
            prefill_program_addr: address of the cached prefill program in DRAM.
            preamble_addr: pre-reserved DRAM slot for the runtime preamble.
                Caller (compile_instructions or run_from_bin's loader) reserved
                this via ``allocate_program_dram(SMALL_SIZE)`` once at startup.
            prefill_seq: full tokenized prompt; last token is decoder seed.
            gflops: FLOPS estimate (from meta).
        """
        if prefill_seq is None:
            raise ValueError("run_prefill: prefill_seq is required (caller must tokenize via apply_chat_template).")
        if len(prefill_seq) <= 1:
            raise ValueError("Prefill sequence must have at least 2 tokens.")

        # Prefill processes all but the last token (last token seeds the decoder).
        prefill_seq = prefill_seq[:-1]
        actual_seq_len = len(prefill_seq)
        if actual_seq_len > self.PREFILL_MAX_SEQ_LEN:
            raise ValueError(
                f"Prompt too long: actual_seq_len={actual_seq_len} > PREFILL_MAX_SEQ_LEN={self.PREFILL_MAX_SEQ_LEN}. "
                f"Rebuild the bin with a larger prefill_max_seq_len in config."
            )
        self.seq_len = actual_seq_len

        q_seq_len = actual_seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        bucket_idx = max(1, aligned_seq_len // UE_VECTOR_SIZE)

        # Zero the intermediate buffers that the prefill scatter loop reads at
        # ``template_seq_len`` rows. With ``gf_M_reg=gf_seq_len``, ops compute only
        # ``actual_seq_len`` rows but the static-unrolled scatter still iterates
        # ``template_seq_len`` rows, so slots [actual_seq_len..template_seq_len) would
        # otherwise pick up stale DRAM (potentially NaN bits from prior runs) and
        # propagate NaN through K/V cache → attention → all layers.
        ahd = self.actual_head_dim
        nkvh = self.num_kv_heads
        qpkv = self.group_size
        template = self.PREFILL_MAX_SEQ_LEN
        # K_NORM, Q_NORM, V_PROJ_TEMP are sized for MAX_CONTEXT_SIZE × per_token_K/Q
        # but only the first `template * (nkvh|nkvh*qpkv)` rows are read by the scatter.
        # Zero those leading regions to defeat stale-NaN.
        zero_kvq = torch.zeros(template * nkvh * qpkv * ahd, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_K_NORM_DRAM, zero_kvq[: template * nkvh * ahd])
        self.dma_to_accelerator_memory(self.LAYER0_Q_NORM_DRAM, zero_kvq[: template * nkvh * qpkv * ahd])
        self.dma_to_accelerator_memory(self.LAYER0_V_PROJ_TEMP, zero_kvq[: template * nkvh * ahd])

        # DMA inputs: embedding (actual_seq_len rows) and bias mask.
        embedding_tensor = self.get_embedding_for_tokens(prefill_seq)
        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
        bias_one_group = torch.full((aligned_seq_len, aligned_seq_len), float("-inf"), dtype=torch.bfloat16)
        valid_mask = torch.tril(torch.ones(aligned_seq_len, aligned_seq_len, dtype=torch.bool), diagonal=0) if not self.causal_mask_upper else torch.triu(torch.ones(aligned_seq_len, aligned_seq_len, dtype=torch.bool), diagonal=0)
        bias_one_group.masked_fill_(valid_mask, 0.0)
        bias_one_group[:, q_seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_one_group)

        # Emit runtime preamble: ADD_SETs for the 3 GPRs + JUMP_ABS into prefill.
        # Same slot is reused across calls (overwritten each time).
        self.clear_inst_id()
        self.start_capture()
        self.generate_instruction_add_set(self.gf_seq_len,    actual_seq_len)
        self.generate_instruction_add_set(self.gf_q_seq_len,  q_seq_len)
        self.generate_instruction_add_set(self.gf_bucket_idx, bucket_idx)
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(prefill_program_addr))
        self.stop_capture()
        self.write_captured_instructions_to_dram(preamble_addr)
        self.clear_capture_buffer()

        # Execute from the preamble — it jumps into the cached prefill, which halts.
        self.program_execute(preamble_addr, gflops=gflops)

    def _emit_decoder_program(self, layer_size: int) -> int:
        """Emit ONE decode-position-agnostic decoder program.

        Caller wraps this in start_capture()/stop_capture(). The emitted program
        ends with a halt so it can be jumped to via a runtime preamble that
        primes gf_bucket_idx (1-based KV context bucket). gf_seq_len is
        primed once by the prefill preamble and auto-incremented at the end
        of every decoder execution via ADD_INC.

        Address math: K/V cache write addresses are computed at runtime as
        ``base + gf_seq_len * (ahd*bpe)`` via reg_mul_imm + add_imm into TMP_REG,
        then passed as ``general_reg_src=TMP_REG`` to the scatter store.

        Decoder attention uses the PBI-bucketed
        ``decoder_group_attention_core`` (one body per KV head, processing all
        ``qpkv`` Q heads at once), with bucket selection driven by
        ``gf_bucket_idx``. Each bucket body covers exactly the active KV
        range — no stale-cache positions enter the softmax, eliminating the
        NaN cascade the legacy static-seq_len kernel produced.

        All bulk ops (matmul, rms_norm) use ``gf_M_reg`` GPRs (gf_one for M=1,
        gf_nkvh for M=nkvh, gf_nkvh_qpkv for M=nkvh*qpkv).
        """
        ahd  = self.actual_head_dim   # 128
        nkvh = self.num_kv_heads      # 8
        qpkv = self.group_size        # 2
        bpe  = self.bytes_per_element
        hd   = self.head_dim          # 1024
        rope_row_bytes = ahd * 2 * bpe   # 512 B per token

        # Decoder attention bucketing: each bucket body covers seq_len = i * UE_VECTOR_SIZE
        # for i = 1..num_buckets. num_buckets must cover MAX_CONTEXT_SIZE.
        num_buckets_decoder = (self.MAX_CONTEXT_SIZE + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
        ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE_LOCAL

        # Reset GPR allocator. Keep long-lived locals to a minimum — the PBI bucketed
        # decoder kernel internally allocates registers per bucket body (bucket_scratch,
        # m_reg, transpose's 4 + nested loop counters), so we have a tight budget.
        # Only two locals survive across the layer loop:
        #   gf_one          (5) — gf_M_reg for M=1 matmul/norm ops
        #   gf_rope_cos_abs (6) — absolute cos-base addr for this step's rope; reused
        #                         across all rope_hf_core calls in all layers
        self._isa_reg_counter = 5
        gf_one          = self.alloc_isa_reg()
        gf_rope_cos_abs = self.alloc_isa_reg()
        self.generate_instruction_add_set(gf_one, 1)
        # gf_rope_cos_abs = ROPE_LOCAL_BASE + gf_seq_len * rope_row_bytes (word address)
        self.generate_instruction_reg_mul_imm(
            gf_rope_cos_abs, self.gf_seq_len, ue_35bit_addr_shifter(rope_row_bytes))
        self.generate_instruction_add_imm(
            gf_rope_cos_abs, ue_35bit_addr_shifter(ROPE_WEIGHT_ADDR), gf_rope_cos_abs)

        total_flops = 0
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        for layer_idx in range(layer_size):
            layer_off = layer_idx * LAYER_WEIGHT_SIZE
            if layer_idx != 0:
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, sram_address=0x10000, element_size=self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_INPUT_DRAM, element_size=self.vector_length)

            # Pre-norm — M=1 via gf_one
            total_flops += (self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                          OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off,
                          gf_M_reg=gf_one) or 0)

            # Q, K, V projections
            total_flops += (self.matmat_mul_core(M=1, K=self.vector_length, N=hd * qpkv,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off, gf_M_reg=gf_one) or 0)
            total_flops += (self.matmat_mul_core(M=1, K=self.vector_length, N=hd,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off, gf_M_reg=gf_one) or 0)
            total_flops += (self.matmat_mul_core(M=1, K=self.vector_length, N=hd,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_V_PROJ_TEMP,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off, gf_M_reg=gf_one) or 0)

            # QK RMSNorm: M is a compile-time constant (nkvh for K, nkvh*qpkv for Q).
            # Pass static M; no gf_M_reg needed.
            total_flops += (self.rms_norm_core_dram(M=nkvh, N=ahd, A_DRAM_ADDR=self.LAYER0_K_DRAM,
                          OUTPUT_DRAM_ADDR=self.LAYER0_K_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_NORM_GAMMA + layer_off) or 0)
            total_flops += (self.rms_norm_core_dram(M=nkvh * qpkv, N=ahd, A_DRAM_ADDR=self.LAYER0_Q_DRAM,
                          OUTPUT_DRAM_ADDR=self.LAYER0_Q_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_NORM_GAMMA + layer_off) or 0)

            # RoPE per head — use base-class rope_hf_core with gr_weight_dram (single GPR
            # holding absolute cos-base address). gf_rope_cos_abs was primed once at the
            # start of the bin (= ROPE_LOCAL + gf_seq_len * rope_row_bytes).
            for kv_h in range(nkvh):
                total_flops += user_dma_core.UnifiedEngine.rope_hf_core(self,
                    N=ahd,
                    input_dram_addr=self.LAYER0_K_NORM_DRAM + kv_h * ahd * bpe,
                    output_dram_addr=self.LAYER0_K_NORM_DRAM + kv_h * ahd * bpe,
                    cos_dram_addr=ROPE_WEIGHT_ADDR,
                    gr_weight_dram=gf_rope_cos_abs)
            for q_h in range(nkvh * qpkv):
                total_flops += user_dma_core.UnifiedEngine.rope_hf_core(self,
                    N=ahd,
                    input_dram_addr=self.LAYER0_Q_NORM_DRAM + q_h * ahd * bpe,
                    output_dram_addr=self.LAYER0_Q_NORM_DRAM + q_h * ahd * bpe,
                    cos_dram_addr=ROPE_WEIGHT_ADDR,
                    gr_weight_dram=gf_rope_cos_abs)

            # Per-KV-head: store new K/V to cache at decode position, then per-Q-head attention.
            # K cache write addr = k_cache_base + gf_seq_len * (ahd*bpe); same for V.
            for kv_h in range(nkvh):
                k_cache_base = (self.LAYER0_K_ROPE_DRAM
                                + layer_idx * nkvh * self.MAX_CONTEXT_SIZE * ahd * bpe
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)
                v_cache_base = (self.LAYER0_V_DRAM
                                + layer_idx * nkvh * self.MAX_CONTEXT_SIZE * ahd * bpe
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)

                # K cache write
                self.accelerator_memory_to_sram(self.LAYER0_K_NORM_DRAM + kv_h * ahd * bpe, 0x10000, ahd)
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

                # Gather this KV head's Q group (qpkv contiguous head_dim rows) into FLASH_Q.
                for q in range(qpkv):
                    q_src = self.LAYER0_Q_NORM_DRAM + (kv_h * qpkv + q) * ahd * bpe
                    flash_q_addr = self.LAYER0_FLASH_Q_DRAM + q * ahd * bpe
                    self.accelerator_memory_to_sram(q_src, 0x30000, ahd)
                    self.sram_to_accelerator_memory(0x30000, flash_q_addr, ahd)

                # PBI-bucketed group attention: one call processes all qpkv Q heads for
                # this KV head. gf_bucket_idx selects the bucket body whose seq_len
                # exactly covers the current KV range — uninitialized cache slots are
                # never read, eliminating the NaN cascade the legacy kernel had.
                attn_result = self.decoder_group_attention_core(
                    group_size=qpkv,
                    head_dim=ahd,
                    seq_len=UE_VECTOR_SIZE,
                    Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
                    K_DRAM_ADDR=k_cache_base,
                    V_DRAM_ADDR=v_cache_base,
                    OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM + kv_h * qpkv * ahd * bpe,
                    IDENTITY_DRAM_ADDR=self.IDENTITY_DRAM_ADDR,
                    SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                    BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_DRAM,
                    gf_bucket_idx=self.gf_bucket_idx,
                    num_buckets=num_buckets_decoder,
                )
                total_flops += (attn_result[-1]
                                if isinstance(attn_result, (list, tuple)) else (attn_result or 0))

            # o_proj
            total_flops += (self.matmat_mul_core(M=1, K=hd * qpkv, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off, gf_M_reg=gf_one) or 0)

            # Qwen3: no post-attention norm; residual direct on o_proj output
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_INPUT_DRAM, sram_address=0x10000, element_size=self.vector_length)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM, sram_address=0x90000, element_size=self.vector_length)
            self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, element_size=self.vector_length)

            # Qwen3: post_attention_layernorm IS the pre-FFN norm
            total_flops += (self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                          OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off,
                          gf_M_reg=gf_one) or 0)

            # MLP: SwiGLU
            total_flops += (self.matmat_mul_core(M=1, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off, silu_enable=True,
                gf_M_reg=gf_one) or 0)
            total_flops += (self.matmat_mul_core(M=1, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off, gf_M_reg=gf_one) or 0)

            # gate x up (M=1: mlp_elements fits in SRAM in one shot)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_GATE_DRAM, sram_address=0x10000, element_size=self.mlp_elements)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_UP_DRAM, sram_address=0x90000, element_size=self.mlp_elements)
            self.eltwise_mul_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.mlp_elements)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_MLP_MULT_DRAM, element_size=self.mlp_elements)

            # down_proj: K=6144 ≤ SCALE_BRAM_ELEMENTS=8192, single call
            total_flops += (self.matmat_mul_core(M=1, K=self.mlp_elements, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off, gf_M_reg=gf_one) or 0)

            # Qwen3: no post-FFN norm; residual direct on down_proj output
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, sram_address=0x10000, element_size=self.vector_length)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_DOWN_DRAM, sram_address=0x90000, element_size=self.vector_length)
            self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, element_size=self.vector_length)

        if layer_size == self.LAYER_SIZE:
            total_flops += (self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_OUTPUT_DRAM,
                OUTPUT_DRAM_ADDR=self.OUTPUT_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_OUTPUT_NORM_GAMMA,
                gf_M_reg=gf_one) or 0)
            total_flops += (self.matmat_mul_core(M=1, K=self.vector_length, N=self.EMBEDDING_ELEMENTS,
                A_DRAM_ADDR=self.OUTPUT_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_QUANT, OUTPUT_DRAM_ADDR=self.LOGITS_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_SCALE, gf_M_reg=gf_one) or 0)

        # Advance gf_seq_len for next decode step (matches gemma3 convention).
        # Host preamble primes only gf_bucket_idx between steps; gf_seq_len carries
        # across via this in-bin increment.
        self.generate_instruction_add_inc(self.gf_seq_len)

        # End-of-program halt so the runtime preamble's JUMP_ABS returns control after execute.
        self.generate_instruction_halt()
        return total_flops

    def compile_instructions(self, layer_size: int | None = None) -> dict:
        """Compile a UNIFIED single-bin instruction image: ONE prefill program
        + ONE decoder program in one capture session. Writes
        ``qwen3_4b_instruction.bin`` + matching ``.json`` meta to disk.

        Both programs are **seq_len-agnostic**: matmul/norm/rope row counts come
        from gf_seq_len / gf_q_seq_len GPRs, and flash attention dispatches on
        gf_bucket_idx — all primed by the runtime preamble in run_prefill /
        run_decoder per call. The cached bin therefore works across any prompt
        length ≤ PREFILL_MAX_SEQ_LEN and any decode position < MAX_CONTEXT_SIZE.

        Cache-hit: if bin + meta already exist, returns the meta without
        recompile.

        Returns the meta dict:
          {
            "prefill_template_seq_len":  int,   # static template used at compile time
            "prefill_program_start_addr": "0x...",
            "prefill_program_size":      int,   # bytes
            "prefill_template_flops":    int,
            "decoder_program_start_addr": "0x...",
            "decoder_program_size":      int,
            "decoder_total_flops":       int,
          }
        """
        if layer_size is None:
            layer_size = self.LAYER_SIZE

        paths_cfg = self._cfg.get("paths", {})
        bin_rel  = paths_cfg.get("instruction_bin",  "qwen3_4b_bin/qwen3_4b_instruction.bin")
        meta_rel = paths_cfg.get("instruction_meta", "qwen3_4b_bin/qwen3_4b_instruction.json")
        bin_path  = os.path.join(self.script_dir, bin_rel)
        meta_path = os.path.join(self.script_dir, meta_rel)

        # Always run the full compile, even if the bin is cached, because
        # ``flash_attention_core_pbi`` and ``bf16_transpose_core_pbi`` do
        # **host-side identity-matrix DMA writes** during compile and the
        # captured bin references those exact DRAM addresses. Skipping the
        # compile would leave those addresses holding whatever any other
        # script wrote to them since the bin was generated (e.g. another
        # model's weights), which produces NaN attention and all-`!` decode.
        # The disk bin is still re-used as a cache for the *bytes* — we
        # skip the bin-write step if the existing file matches.
        bin_cached = os.path.exists(bin_path) and os.path.exists(meta_path)

        os.makedirs(os.path.dirname(bin_path), exist_ok=True)
        # Template seq_len: drives static M= args (overridden at runtime by gf_M_reg),
        # FLOPs estimate, and unrolled scatter-loop iteration count. Set to
        # PREFILL_MAX_SEQ_LEN so the bin can handle any actual_seq_len up to that.
        prefill_template_seq_len = self.PREFILL_MAX_SEQ_LEN

        global _SILENT_MODE
        _SILENT_MODE = True
        self.clear_inst_id()
        self.start_capture()

        # Track start address (= current program-DRAM allocator tail before any DMA).
        instruction_base_addr = self.get_program_dram_addr()

        _original_print(f"  Compiling prefill (template seq_len={prefill_template_seq_len}, {layer_size} layers)...")
        prefill_count_at_start = self.capture_count
        prefill_flops = self._emit_prefill_program(seq_len=prefill_template_seq_len, layer_size=layer_size)
        prefill_program_size = (self.capture_count - prefill_count_at_start) * 32

        _original_print(f"  Compiling decoder ({layer_size} layers)...")
        decoder_count_at_start = self.capture_count
        decoder_flops = self._emit_decoder_program(layer_size=layer_size)
        decoder_program_size = (self.capture_count - decoder_count_at_start) * 32

        self.stop_capture()
        _SILENT_MODE = False

        # Sanity check vs MAX_DECODER_INSTRUCTIONS cap.
        from user_dma_core import MAX_DECODER_INSTRUCTIONS
        if self.capture_count >= MAX_DECODER_INSTRUCTIONS:
            raise RuntimeError(
                f"Capture hit MAX_DECODER_INSTRUCTIONS cap "
                f"({MAX_DECODER_INSTRUCTIONS} instructions = "
                f"{MAX_DECODER_INSTRUCTIONS * 32 / 2**20:.0f} MiB). "
                f"Captured {self.capture_count} instructions = "
                f"{self.capture_count * 32 / 2**20:.0f} MiB. "
                f"Shrink PREFILL_MAX_SEQ_LEN or move more ops to PBI loop_start."
            )

        all_bytes = bytearray()
        for inst in self.capture_buffer:
            all_bytes.extend(inst.get_bytes())

        prefill_program_start_addr = instruction_base_addr
        decoder_program_start_addr = instruction_base_addr + prefill_program_size

        meta = {
            "instruction_base_addr":        f"0x{instruction_base_addr:X}",
            "instruction_total_size":       len(all_bytes),
            "prefill_template_seq_len":     prefill_template_seq_len,
            "prefill_program_start_addr":   f"0x{prefill_program_start_addr:X}",
            "prefill_program_size":         prefill_program_size,
            "prefill_template_flops":       prefill_flops,
            "decoder_program_start_addr":   f"0x{decoder_program_start_addr:X}",
            "decoder_program_size":         decoder_program_size,
            "decoder_total_flops":          decoder_flops,
        }
        if bin_cached:
            print(f"Compile re-run for identity-matrix DMAs; bin already on disk at {bin_path}.")
        else:
            with open(bin_path, "wb") as f:
                f.write(all_bytes)
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"Instruction bin written: {bin_path} ({len(all_bytes)} bytes; "
                  f"prefill {prefill_program_size} B + decoder {decoder_program_size} B)")
        self.clear_capture_buffer()
        return meta

    def run_decoder(self, decoder_program_addr: int, preamble_addr: int,
                    token_id: int, gflops_per_token: int | None = None) -> dict:
        """Run decode loop via dynamic-PBI runtime preamble.

        Each step emits a small preamble at ``preamble_addr`` that primes
        gf_seq_len (current decode position, ==previous token count) and
        gf_bucket_idx (1-based KV context bucket), then jump_abs's into the
        cached decoder program. Same preamble slot is overwritten per token.

        Args:
            decoder_program_addr: address of the cached decoder program in DRAM.
            preamble_addr: pre-reserved DRAM slot for runtime preamble (same
                slot reused as run_prefill).
            token_id: seed token (last token of the prompt).
            gflops_per_token: single FLOPS estimate for the decoder program.
        """
        if token_id is None:
            print("No last token available for decode.")
            return {}

        # Qwen3 stop tokens: <|im_end|>=151645, <|endoftext|>=151643
        _qwen3_stop_tokens = {151643, 151645, self._end_of_turn_token_id}

        global _SILENT_MODE
        # Decoder PBI buckets cover seq_len = i * UE_VECTOR_SIZE for i = 1..num_buckets,
        # spanning MAX_CONTEXT_SIZE. Bias buffer must match the maximum bucket size
        # since each bucket body reads its own static seq_len of bias rows.
        num_buckets_decoder = (self.MAX_CONTEXT_SIZE + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
        max_bucket_seq_len = num_buckets_decoder * UE_VECTOR_SIZE
        max_seq_len = self.MAX_CONTEXT_SIZE
        while self.seq_len < max_seq_len:
            _SILENT_MODE = True
            # self.seq_len at entry is the count of K/V already in cache; the
            # current decode token will be written at that index.
            decode_pos = self.seq_len               # K/V cache write index for this token
            new_ctx_len = decode_pos + 1            # KV positions [0..decode_pos] inclusive
            aligned_ctx = ((new_ctx_len + 63) // 64) * 64
            bucket_idx = max(1, aligned_ctx // UE_VECTOR_SIZE)

            embedding_tensor = self.get_embedding_for_tokens([token_id])
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
            # Bias sized for the maximum decoder bucket; positions
            # [new_ctx_len..max_bucket_seq_len-1] are -inf so the selected bucket
            # body never sums over uninitialized KV-cache slots.
            bias_host = torch.full((1, max_bucket_seq_len), -1e36, dtype=torch.bfloat16)
            bias_host[0, :new_ctx_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_host)

            # Decode preamble: prime gf_bucket_idx only; gf_seq_len is carried over
            # from prefill (or previous decode step) via the in-bin ADD_INC.
            self.clear_inst_id()
            self.start_capture()
            self.generate_instruction_add_set(self.gf_bucket_idx, bucket_idx)
            self.generate_instruction_jump_abs(ue_35bit_addr_shifter(decoder_program_addr))
            self.stop_capture()
            self.write_captured_instructions_to_dram(preamble_addr)
            self.clear_capture_buffer()

            self.start_execute_from_dram(preamble_addr)
            self.wait_queue(10.0)
            token_id = self.get_arg_max_index()
            token_char = self.tokenizer.decode([token_id])
            _SILENT_MODE = False
            self.seq_len += 1
            if token_id in _qwen3_stop_tokens:
                print(f"\nStop token {token_id} reached.")
                break
            print(token_char, end="", flush=True)
        return self.seq_len

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3-4B prefill + decode on accelerator.")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt: tokenizer encodes this to prefill_seq (overrides default)")
    parser.add_argument("--local-weights", action="store_true", help="Use qwen3_4b_bin/full_model_weights.bin instead of generated weights_qwen3_4b_hf.bin")
    parser.add_argument('--dev', type=str, default='xdma0',
                        help='DMA device name (e.g., xdma0, xdma1). Default: xdma0')
    parser.add_argument('--cycle', type=float, default=5.62,
                        help='Clock cycle time in nanoseconds (default: 5.62ns ≈ peak 22.8 GFLOPS)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.local_weights:
        weights_bin_rel = "qwen3_4b_bin/full_model_weights.bin"
    else:
        weights_bin_rel = "qwen3_4b_bin/weights_qwen3_4b_hf.bin"
        weights_bin_full = os.path.join(script_dir, weights_bin_rel)
        if not os.path.exists(weights_bin_full):
            weight_bin_generate(script_dir=script_dir, output_path=weights_bin_full)

    set_dma_device(args.dev)
    global DMA_DEVICE_H2C, DMA_DEVICE_C2H, DMA_DEVICE_USER
    DMA_DEVICE_H2C = user_dma_core.DMA_DEVICE_H2C
    DMA_DEVICE_C2H = user_dma_core.DMA_DEVICE_C2H
    DMA_DEVICE_USER = user_dma_core.DMA_DEVICE_USER
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle
    user_dma_core.UE_PEAK_GFLOPS = 0.128 / args.cycle
    print(f"Using DMA device: {args.dev}")
    print(f"  H2C: {DMA_DEVICE_H2C}")
    print(f"  C2H: {DMA_DEVICE_C2H}")
    print(f"  USER: {DMA_DEVICE_USER}")
    print(f"Setting CLOCK_CYCLE_TIME_NS = {user_dma_core.CLOCK_CYCLE_TIME_NS}, UE_PEAK_GFLOPS = {user_dma_core.UE_PEAK_GFLOPS:.4f}")

    ue = Qwen3_4b_UnifiedEngine(script_dir=script_dir, weights_bin=weights_bin_rel)
    cfg = _load_config(script_dir)
    # Always tokenize via apply_chat_template — no hardcoded ids in the config.
    user_prompt = args.prompt if args.prompt is not None else cfg.get("default_prompt", "What is 3 + 5?")
    system_prompt = cfg.get("default_system_prompt", "You are a helpful assistant.")
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    prompt_with_template = ue.tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    prefill_seq = tuple(ue.tokenizer.encode(prompt_with_template, add_special_tokens=False))
    print(f"User prompt ({len(prefill_seq)} tokens): {user_prompt!r}")
    print(f"Sequence ids: {prefill_seq}")

    print(f"\n--- Compiling unified instruction bin (1 prefill + 1 decoder, dynamic PBI) ---")
    timer = time.perf_counter()
    inst_meta = ue.compile_instructions()
    print(f"  compile_instructions done in {time.perf_counter() - timer:.2f}s")

    paths_cfg = cfg.get("paths", {})
    inst_bin_path = os.path.join(script_dir, paths_cfg.get("instruction_bin",
                                  "qwen3_4b_bin/qwen3_4b_instruction.bin"))
    base_addr, _ = ue.load_instructions(inst_bin_path)
    # Reserve a slot AFTER the loaded bin for the runtime preamble. Prefill preamble
    # is 4 instructions (3 ADD_SET + JUMP_ABS) = 128 B; decode preamble is 2 instructions
    # (ADD_SET gf_bucket_idx + JUMP_ABS) and overwrites the same slot.
    preamble_addr = ue.get_program_dram_addr()
    ue.allocate_program_dram(128)

    prefill_program_addr = _parse_offset(inst_meta["prefill_program_start_addr"])
    decoder_program_addr = _parse_offset(inst_meta["decoder_program_start_addr"])
    decoder_total_flops  = inst_meta["decoder_total_flops"]

    actual_seq_len = len(prefill_seq) - 1
    # Rescale prefill FLOPs from the compile-time template to the actual seq_len.
    # The meta records FLOPs for ``prefill_template_seq_len`` rows; runtime ops execute
    # ``actual_seq_len`` rows (via gf_seq_len-driven PBI loops), so linear scaling gives
    # the real work count for the GFLOPS report.
    template_seq_len = int(inst_meta["prefill_template_seq_len"])
    gflops_prefill = inst_meta["prefill_template_flops"] * actual_seq_len // max(template_seq_len, 1)
    print(f"\n--- Starting prefill (actual {actual_seq_len} tokens, dynamic seq_len) ---")
    print(f"Prompt tokens ({len(prefill_seq)}): {prefill_seq}")
    print(f"Prompt text: {ue.tokenizer.decode(prefill_seq, skip_special_tokens=False)!r}")
    timer = time.perf_counter()
    ue.run_prefill(prefill_program_addr, preamble_addr, prefill_seq=prefill_seq,
                   gflops=gflops_prefill)
    latency_prefill = time.perf_counter() - timer
    print(f"Prefill execute done in {latency_prefill:.2f} seconds, start decoding...\n")

    print(f"\n--- Starting decoder ---")
    timer = time.perf_counter()
    token_cnt_decoded = ue.run_decoder(decoder_program_addr, preamble_addr,
                                       token_id=prefill_seq[-1],
                                       gflops_per_token=decoder_total_flops)
    latency_decoder = time.perf_counter() - timer
    decoded_tokens = max(token_cnt_decoded - len(prefill_seq) + 1, 1)
    print(f"\nDecoder done in {latency_prefill + latency_decoder:.2f} seconds, total {token_cnt_decoded} tokens, "
          f"decode speed: {decoded_tokens / latency_decoder:.2f} tokens/s ({decoded_tokens} decoded tokens / {latency_decoder:.2f}s).")
    print("Qwen3-4B test ends.")

if __name__ == "__main__":
    main()
