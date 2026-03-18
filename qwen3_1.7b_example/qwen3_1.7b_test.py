#!/usr/bin/env python3
"""
Qwen3-1.7B inference on accelerator: prefill + decode.

  - Config from qwen3_1.7b_bin/qwen3_1.7b_config.json; weights from a single bin (see below).
  - Prefill: compiled each run. Decoder: if qwen3_1.7b_bin/decoder_program.bin and
    qwen3_1.7b_bin/decoder_program.json exist, skip decoder compile and load
    program sizes from meta; otherwise compile and write the bin + meta.
  - Run prefill then decode loop. For numeric verification use qwen3_1.7b_numeric.py.

Architecture notes vs Gemma3:
  - 28 layers, 16 Q heads / 8 KV heads, actual head_dim=128.
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
  - Default: qwen3_1.7b_bin/weights_qwen3_1.7b_hf.bin (generated from HF model if missing).
  - --local-weights: use qwen3_1.7b_bin/full_model_weights.bin instead.

Usage:
  python qwen3_1.7b_test.py
  python qwen3_1.7b_test.py --prompt "your prompt"
  python qwen3_1.7b_test.py --dev xdma0 [--cycle 5.88]
  python qwen3_1.7b_test.py --local-weights
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

# This file's folder; user_dma_core.py is one folder up; that parent is added to sys.path.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import user_dma_core
from user_dma_core import DMA_DEVICE_H2C, TYPE, UE_FMAX_CONTEXT_SIZE, UE_VECTOR_SIZE, UE_ARGMAX_INDEX, URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS, set_dma_device
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

def weight_bin_generate(script_dir: str | None = None, output_path: str | None = None) -> str:
    """Generate weights_qwen3_1.7b_hf.bin from Hugging Face model per qwen3_1.7b_config.json layout.
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
    """Load qwen3_1.7b_config.json and build weight_defs (offset/size dict) from regions."""
    config_path = os.path.join(script_dir, "qwen3_1.7b_config.json")
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
# Qwen3-1.7B unified engine
# -----------------------------------------------------------------------------
class Qwen3_1_7b_UnifiedEngine(UnifiedEngine):
    """UnifiedEngine for Qwen3-1.7B: loads config + weight bin, compile_prefill/compile_decoder, run_prefill/run_decoder.

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
        # Initialize identity-DMA dedup state before super().__init__() which calls dma_write.
        self._identity_dram_written = False
        self._identity_dram_addr = None
        self._IDENTITY_MAT_BYTES = UE_VECTOR_SIZE * UE_VECTOR_SIZE * 2  # bfloat16 bytes
        # Qwen3 DRAM layout starting from 0x00000000 (4GB DRAM):
        #   params: 0x00000000 – 0x58000000 (~1.4 GB, covers layers + LM_HEAD + ROPE)
        #   tensors: 0x58000000 – 0x98000000 (~1 GB, intermediates + KV cache)
        #   instructions: 0x98000000 – 0xA0000000 (128 MB)
        super().__init__(
            params_dram_base=0x00000000,
            tensor_dram_base=0x58000000,
            program_dram_base=0x98000000,
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
        self.LAYER_SIZE = fi["num_layers"]
        self.EMBEDDING_ELEMENTS = fi["embedding_vocab"]
        self._isa_reg_counter = 1
        fixed = self._cfg.get("fixed_isa_regs", {})
        self.V_CACHE_SIZE_REG = fixed["V_CACHE_SIZE_REG"]
        self.TMP_REG = fixed["TMP_REG"]
        self.ROPE_SIZE_REG = fixed["ROPE_SIZE_REG"]
        self.causal_mask_upper = False
        self._rope_global_layers = set(model["rope_global_layers"])  # empty for Qwen3
        self._end_of_turn_token_id = model["end_of_turn_token_id"]
        self._gamma_bin_offset = self._cfg["special"]["rms_norm"]["gamma_offset"]  # 0.0

        # Qwen3 architecture flags
        self._has_q_k_norm = True        # QK RMSNorm per head
        self._has_post_attn_norm = False  # no post-attention norm
        self._has_post_mlp_norm = False   # no post-FFN norm

        bin_path = weights_bin or paths["weights_bin"]
        full_path = os.path.join(self.script_dir, bin_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Bin file not found: {full_path}")
        with open(full_path, "rb") as f:
            self.weight_bin = f.read()
        self.weight_init()
        self.tensor_init()

    # ------------------------------------------------------------------ #
    #  Identity-matrix DRAM caching  (fixes compile_prefill hang)        #
    # ------------------------------------------------------------------ #
    # flash_attention_core in user_dma_core does:                        #
    #   addr = self.get_params_dram_addr()                               #
    #   self.allocate_params_dram(8192)   ← advances params pointer      #
    #   self.dma_write(..., addr, eye, 8192)  ← REAL hardware DMA        #
    # This is NOT inside a capture block, so it fires on every call:     #
    # 8 KV-heads × 28 layers = 224 blocking PCIe writes during compile.  #
    # Fix: allocate the slot once (first call), skip all subsequent      #
    # allocations and DMA writes.                                         #

    _IDENTITY_MAT_BYTES = UE_VECTOR_SIZE * UE_VECTOR_SIZE * 2  # 8192

    def _preallocate_identity_matrix(self) -> None:
        """Allocate the identity-matrix DRAM slot once and DMA-write it.

        Must be called AFTER weight_init (params DRAM is fully laid out) and
        BEFORE the first compile pass.  Subsequent calls are no-ops.
        """
        if self._identity_dram_addr is not None:
            return
        self._identity_dram_addr = super().allocate_params_dram(self._IDENTITY_MAT_BYTES)
        eye = torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16)
        super().dma_write(DMA_DEVICE_H2C, self._identity_dram_addr, eye, self._IDENTITY_MAT_BYTES)
        self._identity_dram_written = True
        _original_print(f"  Identity matrix pre-allocated at DRAM 0x{self._identity_dram_addr:X}")

    def _flash_attention_core_cached(self, **kwargs) -> int:
        """Wrap flash_attention_core so the identity-matrix slot is always reused.

        flash_attention_core calls get_params_dram_addr() + allocate_params_dram(8192)
        + dma_write on EVERY invocation — a real blocking PCIe write outside any capture
        block.  By temporarily resetting _next_params_dram_addr to the pre-allocated slot
        before the call (and restoring it after), every subsequent call reuses the same
        DRAM address.  The dma_write override then skips the redundant write.
        """
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

    def overwrite_instruction_with_general_register(self, general_register: int) -> None:
        """
        Overwrite the most recently captured instruction to use a general register
        for rewriting the DRAM source/destination address.

        This modifies the instruction at capture_buffer[capture_count - 1] to:
        - Set inst_src_reg_idx in word 0 (bits 4-7)
        - Set inst_type to INSTRUCTION_REG_REWRITE in word 7 (bits 29-31)
        """
        if self.capture_buffer is None or len(self.capture_buffer) == 0:
            print("ERROR: overwrite_instruction_with_general_register() called but capture_buffer is empty!")
            return
        if self.capture_count == 0:
            print("ERROR: overwrite_instruction_with_general_register() called but capture_count is 0!")
            return
        if general_register <= 0 or general_register > 15:
            raise ValueError(f"general_register must be in [1, 15], got {general_register}")

        inst = self.capture_buffer[self.capture_count - 1]
        w = inst.words

        # Overwrite word 0: set inst_src_reg_idx in bits 4-7
        w[0] = ((0 & 0xF) << 0) | \
               ((general_register & 0xF) << 4) | \
               ((0 & 0xF) << 8) | \
               ((0 & 0xF) << 12)

        # Overwrite word 7: preserve bits 0-28, set inst_type to INSTRUCTION_REG_REWRITE in bits 29-31
        w[7] = (w[7] & 0x1FFFFFFF) | ((user_dma_core.INSTRUCTION_REG_REWRITE & 0x7) << 29)

    def isa_add_set_core(self, dst_reg_idx: int, immediate_value: int, timeout_s: float = 10.0) -> None:
        """
        Run a minimal program that sets one ISA register to an immediate value (ADD SET then HALT):
        start_capture -> generate_instruction_add_set -> stop_capture -> halt -> write to DRAM -> execute -> wait.
        Use e.g. isa_add_set_core(V_CACHE_SIZE_REG, self.seq_len * self.k_size).
        """
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

    def decoder_attention_core(self, head_dim: int, seq_len: int, Q_DRAM_ADDR: int, K_DRAM_ADDR: int, V_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, SCRATCH_DRAM_ADDR: int, IDENTITY_DRAM_ADDR: int = None, BIAS_DRAM_ADDR: int = None,
                            debug_mode: bool = False, SM_OUTPUT_DRAM_ADDR: int = None) -> None:

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

        # ----------------------------------------------------------------------------------------------------------------
        # Q @ K^T: (1, head_dim) @ (head_dim, seq_len) -> (1, seq_len)
        # Convention: first matrix Q is (M, K), second K^T is (K, N), output scores (M, N)
        M = 1         # query length (rows of Q)
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
            #print(f"softmax rows: {m_take * N} elements vs {URAM_FULL_ELEMENTS} elements")
            # DEBUG to get seq_len x seq_len sm(QK^T) results are copied to DRAM
            # start_dram_address_of_partial_row_complete_matrix = SM_OUTPUT_DRAM_ADDR + i * N * bytes_per_element #  make only FMAX_CONTEXT_SIZE x seq_len sm(QK^T) results are copied to DRAM
            
            # if m_take * N is greater than the space available in URAM_A, copy the matrix to DRAM
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
        """Initialize DRAM from weight bin: load HF embedding+tokenizer, layers from bin, host-computed RoPE, then OUTPUT_NORM/LM_HEAD from bin."""
        model, model_dir = _ensure_hf_model(self.script_dir, self._cfg)
        # Qwen3 does NOT scale the embedding by sqrt(hidden_size)
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
        """Initialize hardware DRAM tensors for Qwen3-1.7B.

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
        self.IDENTITY_DRAM_ADDR = self.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe)
        self.dma_to_accelerator_memory(self.IDENTITY_DRAM_ADDR, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))

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
        # Pre-allocate the identity matrix used by flash_attention_core so that the
        # 224 DMA writes during compile (8 KV-heads × 28 layers) collapse to 1.
        self._preallocate_identity_matrix()

    def program_execute(self, program_start_addr: int = user_dma_core.DRAM_INSTRUCTION_ADDR, timeout: float = 10.0, gflops: float = None) -> None:
        """Execute compiled program from DRAM instruction memory.
        """
        print(f"Execute program start at 0x{program_start_addr:X}")
        self.start_execute_from_dram(program_start_addr)
        self.wait_queue(timeout)
        latency = self.report_latency_in_us()
        print(f"    Total program execution latency = {latency} us")
        if gflops is not None:
            gflops_program = self.report_flop_rate_gflops(gflops)
            print(f"Report FLOPS for program execution: {gflops_program:.2f} GFLOPS")

    def compile_prefill(self, seq_len: int, layer_size: int | None = None) -> dict:
        """
        Compile prefill for the given prefill sequence.

        Qwen3 differences from Gemma3:
          - QK norm per head (M = seq_len * num_kv_heads, N = actual_head_dim).
          - RoPE applied per head per token (N = actual_head_dim = 128).
          - Per-KV-head scatter + flash_attention(head_dim=128) x 8 KV heads.
          - No post-attention norm; residual direct on o_proj output.
          - No post-FFN norm; residual direct on down_proj output.
          - SwiGLU (silu_enable=True); gate x up row-by-row to avoid SRAM overflow.

        Args:
            seq_len: Length of the prefill sequence (including last token).
            layer_size: Number of layers to compile (default: all).

        Returns:
            (prefill_program_addr, total_flops)
        """
        if layer_size is None:
            layer_size = self.LAYER_SIZE
        seq_len -= 1
        self.seq_len = seq_len
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64

        ahd  = self.actual_head_dim   # 128
        nkvh = self.num_kv_heads      # 8
        qpkv = self.group_size        # 2 (Q heads per KV head)
        bpe  = self.bytes_per_element
        hd   = self.head_dim          # 1024
        # Rope table row size for N=ahd=128: [cos(64), cos(64), -sin(64), sin(64)] = 256 elements
        rope_row_bytes = ahd * 2 * bpe   # 128*2*2 = 512 bytes per position per call

        # --- Qwen3 28 layers: compile ---
        global _SILENT_MODE
        _SILENT_MODE = True
        self.start_capture()
        total_flops = 0
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        _original_print(f"  Compiling prefill seq_len={seq_len}, {layer_size} layers...")
        for layer_idx in range(layer_size):
            _original_print(f"    prefill layer {layer_idx + 1}/{layer_size}", end="\r", flush=True)
            layer_off = layer_idx * LAYER_WEIGHT_SIZE
            if layer_idx != 0:
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, sram_address=0x10000, element_size=seq_len * self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_INPUT_DRAM, element_size=seq_len * self.vector_length)

            # Pre-norm (input_layernorm)
            total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off)

            # Q, K, V projections
            total_flops += self.quantized_matmat_core(M=seq_len, K=self.vector_length, N=hd * qpkv,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off, data_type=TYPE.INT4)
            total_flops += self.quantized_matmat_core(M=seq_len, K=self.vector_length, N=hd,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off, data_type=TYPE.INT4)
            # v_proj writes to temp buffer (seq_len, hd) in standard per-head interleaved layout
            total_flops += self.quantized_matmat_core(M=seq_len, K=self.vector_length, N=hd,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_V_PROJ_TEMP,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off, data_type=TYPE.INT4)

            # QK RMSNorm per head: treat (seq_len, hd) as (seq_len * nkvh, ahd) for K
            # and (seq_len, hd * qpkv) as (seq_len * nkvh * qpkv, ahd) for Q
            total_flops += self.rms_norm_core_dram(M=seq_len * nkvh, N=ahd, A_DRAM_ADDR=self.LAYER0_K_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_K_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_NORM_GAMMA + layer_off)
            total_flops += self.rms_norm_core_dram(M=seq_len * nkvh * qpkv, N=ahd, A_DRAM_ADDR=self.LAYER0_Q_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_Q_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_NORM_GAMMA + layer_off)

            # RoPE per head per token: K uses single RoPE base (rope_global_layers is empty)
            # K_NORM layout: (seq_len, hd) = (seq_len, nkvh * ahd) — heads are contiguous per token
            # rope_hf_core(N=ahd=128) reads cos/sin from table at position t
            ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE_LOCAL   # single base for all layers
            for t in range(seq_len):
                cos_addr = ROPE_WEIGHT_ADDR + t * rope_row_bytes
                sin_addr = cos_addr + ahd * bpe              # sin row starts after cos row (each 64 × 2 B)
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
            # Per-KV-head: scatter K/V to cache + flash buffers, scatter Q, then flash_attention
            for kv_h in range(nkvh):
                k_cache_base = (self.LAYER0_K_ROPE_DRAM
                                + layer_idx * nkvh * self.MAX_CONTEXT_SIZE * ahd * bpe
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)
                v_cache_base = (self.LAYER0_V_DRAM
                                + layer_idx * nkvh * self.MAX_CONTEXT_SIZE * ahd * bpe
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)

                # Scatter roped K_h (standard layout) → KV cache + FLASH_K (GQA dup x qpkv)
                # K_NORM layout per token t: [h0(128), h1(128), ..., h7(128)]
                for t in range(seq_len):
                    k_src = self.LAYER0_K_NORM_DRAM + (t * nkvh + kv_h) * ahd * bpe
                    self.accelerator_memory_to_sram(k_src, 0x10000, ahd)
                    self.sram_to_accelerator_memory(0x10000, k_cache_base + t * ahd * bpe, ahd)
                    for g in range(qpkv):
                        self.sram_to_accelerator_memory(0x10000, self.LAYER0_FLASH_K_DRAM + (t * qpkv + g) * ahd * bpe, ahd)

                # Scatter V_h from V_PROJ_TEMP → KV cache + FLASH_V (GQA dup x qpkv)
                # V_PROJ_TEMP layout per token t: [h0(128), h1(128), ..., h7(128)]
                for t in range(seq_len):
                    v_src = self.LAYER0_V_PROJ_TEMP + (t * nkvh + kv_h) * ahd * bpe
                    self.accelerator_memory_to_sram(v_src, 0x20000, ahd)
                    self.sram_to_accelerator_memory(0x20000, v_cache_base + t * ahd * bpe, ahd)
                    for g in range(qpkv):
                        self.sram_to_accelerator_memory(0x20000, self.LAYER0_FLASH_V_DRAM + (t * qpkv + g) * ahd * bpe, ahd)

                # Scatter Q heads for this KV head → FLASH_Q
                # Q_NORM layout per token t: [q0(128), q1(128), ..., q15(128)] = (nkvh*qpkv) heads
                # KV head kv_h maps to Q heads: kv_h*qpkv .. kv_h*qpkv + qpkv - 1
                for t in range(seq_len):
                    for q in range(qpkv):
                        q_src = self.LAYER0_Q_NORM_DRAM + (t * nkvh * qpkv + kv_h * qpkv + q) * ahd * bpe
                        self.accelerator_memory_to_sram(q_src, 0x30000, ahd)
                        self.sram_to_accelerator_memory(0x30000, self.LAYER0_FLASH_Q_DRAM + (t * qpkv + q) * ahd * bpe, ahd)

                # Flash attention for this KV head (head_dim=128, GQA group_size=2)
                # Use _flash_attention_core_cached to reuse the pre-allocated identity
                # matrix slot — avoids 224 blocking PCIe DMA writes during compile.
                total_flops += self._flash_attention_core_cached(
                    head_dim=ahd,
                    seq_len=aligned_seq_len,
                    Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
                    K_DRAM_ADDR=self.LAYER0_FLASH_K_DRAM,
                    V_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                    OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUT_HEAD_DRAM,
                    SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                    BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_DRAM,
                )

                # Assemble per-head output into FLASH_OUTPUT_DRAM
                # Output layout: (seq_len, nkvh * qpkv * ahd) = (seq_len, 2048)
                out_h_base = kv_h * qpkv * ahd * bpe
                for t in range(seq_len):
                    for g in range(qpkv):
                        src = self.LAYER0_FLASH_OUT_HEAD_DRAM + (t * qpkv + g) * ahd * bpe
                        dst = self.LAYER0_FLASH_OUTPUT_DRAM + t * hd * qpkv * bpe + out_h_base + g * ahd * bpe
                        self.accelerator_memory_to_sram(src, 0x40000, ahd)
                        self.sram_to_accelerator_memory(0x40000, dst, ahd)

            # o_proj: (seq_len, hd * qpkv) → (seq_len, vector_length)
            total_flops += self.quantized_matmat_core(M=seq_len, K=hd * qpkv, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off, data_type=TYPE.INT4)

            # Qwen3: no post-attention norm; add residual directly to o_proj output
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_INPUT_DRAM, sram_address=0x10000, element_size=seq_len * self.vector_length)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM, sram_address=0x90000, element_size=seq_len * self.vector_length)
            self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=seq_len * self.vector_length)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, element_size=seq_len * self.vector_length)

            # Qwen3: post_attention_layernorm IS the pre-FFN norm
            total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off)

            # MLP: gate_proj with SiLU, up_proj, gate x up element-wise, down_proj
            total_flops += self.quantized_matmat_core(M=seq_len, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off, data_type=TYPE.INT4, silu_enable=True)
            total_flops += self.quantized_matmat_core(M=seq_len, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off, data_type=TYPE.INT4)

            # gate x up chunked: process M_CHUNK rows at a time
            # Each row = mlp_elements = 8960 elems × 2 bytes = 17.9 KB
            # Bank A (0x10000..0x7FFFF = 458 KB) holds gate chunk; bank B (0x90000..0xFFFFF = 458 KB) holds up chunk
            # 458 KB / 17.9 KB per row = 25 rows → M_CHUNK=25 (reduces iterations ~25×)
            _bpe = self.bytes_per_element
            _M_CHUNK = min(25, seq_len)
            for _t in range(0, seq_len, _M_CHUNK):
                _m_take = min(_M_CHUNK, seq_len - _t)
                _g_row = self.LAYER0_MLP_GATE_DRAM + _t * self.mlp_elements * _bpe
                _u_row = self.LAYER0_MLP_UP_DRAM   + _t * self.mlp_elements * _bpe
                _m_row = self.LAYER0_MLP_MULT_DRAM  + _t * self.mlp_elements * _bpe
                self.accelerator_memory_to_sram(_g_row, 0x10000, _m_take * self.mlp_elements)
                self.accelerator_memory_to_sram(_u_row, 0x90000, _m_take * self.mlp_elements)
                self.eltwise_mul_core(0x10000, 0x90000, 0x10000, _m_take * self.mlp_elements)
                self.sram_to_accelerator_memory(0x10000, _m_row, _m_take * self.mlp_elements)

            # down_proj: K=6144 ≤ SCALE_BRAM_ELEMENTS=8192, single call (no split)
            total_flops += self.quantized_matmat_core(M=seq_len, K=self.mlp_elements, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off, data_type=TYPE.INT4)

            # Qwen3: no post-FFN norm; add residual directly to down_proj output
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
        _original_print()  # newline after \r layer progress
        print(f"    Prefill program start at 0x{prefill_program_addr:X} end at 0x{self.get_program_dram_addr():X}, usage: {self.get_program_dram_usage()} bytes")

        return prefill_program_addr, total_flops

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
        bias_one_group = torch.full((aligned_seq_len, aligned_seq_len), float("-inf"), dtype=torch.bfloat16)
        valid_mask = torch.tril(torch.ones(aligned_seq_len, aligned_seq_len, dtype=torch.bool), diagonal=0) if not self.causal_mask_upper else torch.triu(torch.ones(aligned_seq_len, aligned_seq_len, dtype=torch.bool), diagonal=0)
        bias_one_group.masked_fill_(valid_mask, 0.0)
        bias_one_group[:, q_seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_one_group)
        self.program_execute(prefill_program_addr, gflops=gflops)

    def compile_decoder(self, layer_size: int | None = None) -> tuple[list[int], list[int]]:
        """Compile decoder programs for seq_len buckets; write decoder_program.bin and decoder_program.json.
        Returns (program_sizes[8], total_flops_list[8])."""
        if layer_size is None:
            layer_size = self.LAYER_SIZE
        paths_cfg = self._cfg.get("paths", {})
        decoder_bin_rel = paths_cfg.get("decoder_program_bin", "qwen3_1.7b_bin/decoder_program.bin")
        decoder_meta_rel = paths_cfg.get("decoder_program_meta", "qwen3_1.7b_bin/decoder_program.json")
        decoder_bin_path = os.path.join(self.script_dir, decoder_bin_rel)
        decoder_meta_path = os.path.join(self.script_dir, decoder_meta_rel)
        os.makedirs(os.path.dirname(decoder_bin_path), exist_ok=True)
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        segment_instruction_counts = []
        total_flops_list = []

        ahd  = self.actual_head_dim   # 128
        nkvh = self.num_kv_heads      # 8
        qpkv = self.group_size        # 2
        bpe  = self.bytes_per_element
        hd   = self.head_dim          # 1024
        rope_row_bytes = ahd * 2 * bpe   # 512 bytes per position

        global _SILENT_MODE
        _SILENT_MODE = True
        self.clear_inst_id()
        self.start_capture()

        buckets = self._cfg["model"]["decoder_seq_len_buckets"]
        _original_print(f"  Compiling decoder {len(buckets)} buckets × {layer_size} layers...")
        for _bi, seq_len in enumerate(buckets):
            _original_print(f"    decoder bucket {_bi + 1}/{len(buckets)} seq_len={seq_len}...", flush=True)
            count_at_start = self.capture_count
            total_flops = 0

            for layer_idx in range(layer_size):
                layer_off = layer_idx * LAYER_WEIGHT_SIZE
                if layer_idx != 0:
                    self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, sram_address=0x10000, element_size=self.vector_length)
                    self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_INPUT_DRAM, element_size=self.vector_length)

                # Pre-norm
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off)

                # Q, K projections; V to FLASH_V_DRAM temp (single token)
                total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=hd * qpkv,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off, data_type=TYPE.INT4)
                total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=hd,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off, data_type=TYPE.INT4)
                total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=hd,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_V_PROJ_TEMP,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off, data_type=TYPE.INT4)

                # QK RMSNorm per head (M=1 token * nkvh heads / nkvh*qpkv heads)
                total_flops += self.rms_norm_core_dram(M=nkvh, N=ahd, A_DRAM_ADDR=self.LAYER0_K_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_K_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_NORM_GAMMA + layer_off)
                total_flops += self.rms_norm_core_dram(M=nkvh * qpkv, N=ahd, A_DRAM_ADDR=self.LAYER0_Q_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_Q_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_NORM_GAMMA + layer_off)

                # RoPE per head (decode position set by ROPE_SIZE_REG at runtime)
                ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE_LOCAL
                for kv_h in range(nkvh):
                    total_flops += self.rope_hf_core(
                        N=ahd,
                        input_dram_addr=self.LAYER0_K_NORM_DRAM + kv_h * ahd * bpe,
                        output_dram_addr=self.LAYER0_K_NORM_DRAM + kv_h * ahd * bpe,
                        cos_dram_addr=ROPE_WEIGHT_ADDR,
                        sin_dram_addr=ROPE_WEIGHT_ADDR + ahd * bpe,
                        rope_size_reg=self.ROPE_SIZE_REG,
                        tmp_reg=self.TMP_REG)
                for q_h in range(nkvh * qpkv):
                    total_flops += self.rope_hf_core(
                        N=ahd,
                        input_dram_addr=self.LAYER0_Q_NORM_DRAM + q_h * ahd * bpe,
                        output_dram_addr=self.LAYER0_Q_NORM_DRAM + q_h * ahd * bpe,
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
                    self.accelerator_memory_to_sram(self.LAYER0_K_NORM_DRAM + kv_h * ahd * bpe, 0x10000, ahd)
                    self.generate_instruction_add_imm(self.V_CACHE_SIZE_REG, k_cache_base, self.TMP_REG)
                    self.sram_to_accelerator_memory(0x10000, 0, ahd)
                    self.overwrite_instruction_with_general_register(self.TMP_REG)

                    # Store V_h to KV cache at decode position (via V_CACHE_SIZE_REG)
                    self.accelerator_memory_to_sram(self.LAYER0_V_PROJ_TEMP + kv_h * ahd * bpe, 0x20000, ahd)
                    self.generate_instruction_add_imm(self.V_CACHE_SIZE_REG, v_cache_base, self.TMP_REG)
                    self.sram_to_accelerator_memory(0x20000, 0, ahd)
                    self.overwrite_instruction_with_general_register(self.TMP_REG)

                    # For each Q head in this KV head's group: run decoder_attention
                    for q in range(qpkv):
                        q_src = self.LAYER0_Q_NORM_DRAM + (kv_h * qpkv + q) * ahd * bpe
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

                # o_proj
                total_flops += self.quantized_matmat_core(M=1, K=hd * qpkv, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off, data_type=TYPE.INT4)

                # Qwen3: no post-attention norm; residual direct on o_proj output
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_INPUT_DRAM, sram_address=0x10000, element_size=self.vector_length)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM, sram_address=0x90000, element_size=self.vector_length)
                self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, element_size=self.vector_length)

                # Qwen3: post_attention_layernorm IS the pre-FFN norm
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off)

                # MLP: SwiGLU
                total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=self.mlp_elements,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off, data_type=TYPE.INT4, silu_enable=True)
                total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=self.mlp_elements,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off, data_type=TYPE.INT4)

                # gate x up (M=1: mlp_elements fits in SRAM in one shot)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_GATE_DRAM, sram_address=0x10000, element_size=self.mlp_elements)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_UP_DRAM, sram_address=0x90000, element_size=self.mlp_elements)
                self.eltwise_mul_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.mlp_elements)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_MLP_MULT_DRAM, element_size=self.mlp_elements)

                # down_proj: K=6144 ≤ SCALE_BRAM_ELEMENTS=8192, single call
                total_flops += self.quantized_matmat_core(M=1, K=self.mlp_elements, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off, data_type=TYPE.INT4)

                # Qwen3: no post-FFN norm; residual direct on down_proj output
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, sram_address=0x10000, element_size=self.vector_length)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_DOWN_DRAM, sram_address=0x90000, element_size=self.vector_length)
                self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, element_size=self.vector_length)

            if layer_size == self.LAYER_SIZE:
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_OUTPUT_DRAM,
                    OUTPUT_DRAM_ADDR=self.OUTPUT_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_OUTPUT_NORM_GAMMA)
                total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=self.EMBEDDING_ELEMENTS,
                    A_DRAM_ADDR=self.OUTPUT_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_QUANT, OUTPUT_DRAM_ADDR=self.LOGITS_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_SCALE, data_type=TYPE.INT4)

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

    def run_decoder(self, decoder_program_sizes: list[int], decoder_base_addr: int, token_id: int, gflops_per_token: list[int] | None = None) -> dict:
        """Run decode loop. seq_len capped at MAX_CONTEXT_SIZE. Breaks on Qwen3 EOS/EOT tokens."""
        if token_id is None:
            print("No last token available for decode.")
            return {}

        # Qwen3 stop tokens: <|im_end|>=151645, <|endoftext|>=151643
        _qwen3_stop_tokens = {151643, 151645, self._end_of_turn_token_id}

        ahd = self.actual_head_dim
        bpe = self.bytes_per_element
        # V_CACHE_SIZE_REG: decode_pos × actual_head_dim × bpe (per-head KV cache stride)
        _kv_stride  = ahd * bpe                  # 128 * 2 = 256 bytes per position per head
        # ROPE_SIZE_REG: decode_pos × rope_row_bytes (rope table stride per position)
        _rope_stride = ahd * 2 * bpe             # 128 * 2 * 2 = 512 bytes per position

        global _SILENT_MODE
        max_seq_len = self.MAX_CONTEXT_SIZE
        while self.seq_len < max_seq_len:
            _SILENT_MODE = True
            self.seq_len += 1
            aligned_seq_len = ((self.seq_len + 63) // 64) * 64
            prog_idx = min((self.seq_len - 1) // 256, 7)
            prog_addr = decoder_base_addr + sum(decoder_program_sizes[:prog_idx])
            gflops = gflops_per_token[prog_idx] if gflops_per_token else None

            self.isa_add_set_core(self.V_CACHE_SIZE_REG, (self.seq_len - 1) * _kv_stride)
            self.isa_add_set_core(self.ROPE_SIZE_REG,    (self.seq_len - 1) * _rope_stride)

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
    parser = argparse.ArgumentParser(description="Qwen3-1.7B prefill + decode on accelerator.")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt: tokenizer encodes this to prefill_seq (overrides default)")
    parser.add_argument("--local-weights", action="store_true", help="Use qwen3_1.7b_bin/full_model_weights.bin instead of generated weights_qwen3_1.7b_hf.bin")
    parser.add_argument('--dev', type=str, default='xdma0',
                        help='DMA device name (e.g., xdma0, xdma1). Default: xdma0')
    parser.add_argument('--cycle', type=float, default=1/0.17,
                        help='Clock cycle time in nanoseconds (default: ~5.88ns)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.local_weights:
        weights_bin_rel = "qwen3_1.7b_bin/full_model_weights.bin"
    else:
        weights_bin_rel = "qwen3_1.7b_bin/weights_qwen3_1.7b_hf.bin"
        weights_bin_full = os.path.join(script_dir, weights_bin_rel)
        if not os.path.exists(weights_bin_full):
            weight_bin_generate(script_dir=script_dir, output_path=weights_bin_full)

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

    ue = Qwen3_1_7b_UnifiedEngine(script_dir=script_dir, weights_bin=weights_bin_rel)
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
    prefill_program_addr, gflops_prefill = ue.compile_prefill(seq_len=len(prefill_seq))
    print(f"Prefill compile done in {time.perf_counter() - timer:.2f} seconds, start decoder compile...")
    decoder_bin_path = os.path.join(script_dir, "qwen3_1.7b_bin", "decoder_program.bin")
    decoder_meta_path = os.path.join(script_dir, "qwen3_1.7b_bin", "decoder_program.json")
    if os.path.exists(decoder_bin_path) and os.path.exists(decoder_meta_path):
        with open(decoder_meta_path, "r") as f:
            meta = json.load(f)
        if "instruction_counts" in meta:
            decoder_program_sizes = [c * 32 for c in meta["instruction_counts"]]
        else:
            decoder_program_sizes = meta["program_sizes"]
        gflops_per_token = meta["total_flops"]
        print(f"Decoder bin found, skipped compile ({time.perf_counter() - timer:.2f}s).")
    else:
        timer_dec = time.perf_counter()
        decoder_program_sizes, gflops_per_token = ue.compile_decoder()
        print(f"Decoder compile done in {time.perf_counter() - timer_dec:.2f} seconds.")
    decoder_base_addr, _ = ue.load_instructions(decoder_bin_path)

    print(f"\n--- Starting prefill ---")
    print(f"Prompt tokens ({len(prefill_seq)}): {prefill_seq}")
    timer = time.perf_counter()
    ue.run_prefill(prefill_program_addr, prefill_seq=prefill_seq, gflops=gflops_prefill)
    latency_prefill = time.perf_counter() - timer
    print(f"Prefill execute done in {latency_prefill:.2f} seconds, start decoding...\n")

    print(f"\n--- Starting decoder ---")
    timer = time.perf_counter()
    token_cnt_decoded = ue.run_decoder(decoder_program_sizes, decoder_base_addr, token_id=prefill_seq[-1], gflops_per_token=gflops_per_token)
    latency_decoder = time.perf_counter() - timer
    print(f"\nDecoder done in {latency_prefill + latency_decoder:.2f} seconds, total {token_cnt_decoded} tokens.")
    print("Qwen3-1.7B test ends.")

if __name__ == "__main__":
    main()
