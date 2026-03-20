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
        super().__init__()
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
        self._isa_reg_counter = 1
        fixed = self._cfg.get("fixed_isa_regs", {})
        self.V_CACHE_SIZE_REG = fixed["V_CACHE_SIZE_REG"]
        self.TMP_REG = fixed["TMP_REG"]
        self.ROPE_SIZE_REG = fixed["ROPE_SIZE_REG"]
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

    def allocate_params_dram(self, size_bytes: int) -> int:
        """Allocate memory from the params DRAM region incrementally."""
        params_dram_addr = self._next_params_dram_addr
        self._next_params_dram_addr += size_bytes
        return params_dram_addr

    def clear_inst_id(self) -> None:
        """Reset instruction ID counter for the next capture."""
        self._inst_id = 0

    def get_arg_max_index(self) -> int:
        """Get the arg max index from the Unified Engine."""
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

        if debug_mode:
            assert SM_OUTPUT_DRAM_ADDR is not None, "SM_OUTPUT_DRAM_ADDR is not set for debug mode"

        SCRATCH_DRAM_PARTIAL_SM = SCRATCH_DRAM_ADDR + head_dim * seq_len * bytes_per_element

        M = head_dim
        K = head_dim
        N = seq_len

        identity_tensor = torch.eye(head_dim, dtype=torch.bfloat16)

        self.accelerator_memory_to_sram(accelerator_dram_address=IDENTITY_DRAM_ADDR,
                                        sram_address=0,
                                        element_size=UE_VECTOR_SIZE * UE_VECTOR_SIZE)

        usable_uram_a_start_addr = UE_VECTOR_SIZE * UE_VECTOR_SIZE * bytes_per_element

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
            gflops_program = self.report_flop_rate_gflops(gflops)
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
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off, data_type=TYPE.INT4)
            total_flops += self.quantized_matmat_core(M=seq_len, K=self.vector_length, N=self.head_dim,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off, data_type=TYPE.INT4)
            # v_proj writes to interleaved temp (T, 512); per-head KV cache populated below.
            total_flops += self.quantized_matmat_core(M=seq_len, K=self.vector_length, N=self.head_dim,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_V_PROJ_TEMP,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off, data_type=TYPE.INT4)

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
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off, data_type=TYPE.INT4)

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
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off, data_type=TYPE.INT4, silu_enable=True)
            total_flops += self.quantized_matmat_core(M=seq_len, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off, data_type=TYPE.INT4)
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
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off, data_type=TYPE.INT4)

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

    def compile_decoder(self, layer_size: int | None = None) -> tuple[list[int], list[int]]:
        """Compile decoder programs for seq_len buckets; write decoder_program.bin and decoder_program.json.
        Returns (program_sizes[8], total_flops_list[8])."""
        if layer_size is None:
            layer_size = self.LAYER_SIZE
        paths_cfg = self._cfg.get("paths", {})
        decoder_bin_rel = paths_cfg.get("decoder_program_bin", "llama3.2_1b_bin/decoder_program.bin")
        decoder_meta_rel = paths_cfg.get("decoder_program_meta", "llama3.2_1b_bin/decoder_program.json")
        decoder_bin_path = os.path.join(self.script_dir, decoder_bin_rel)
        decoder_meta_path = os.path.join(self.script_dir, decoder_meta_rel)
        os.makedirs(os.path.dirname(decoder_bin_path), exist_ok=True)
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        segment_instruction_counts = []
        total_flops_list = []

        global _SILENT_MODE
        _SILENT_MODE = True
        self.clear_inst_id()
        self.start_capture()
        for seq_len in self._cfg["model"]["decoder_seq_len_buckets"]:
            count_at_start = self.capture_count
            total_flops = 0
            for layer_idx in range(layer_size):
                layer_off = layer_idx * LAYER_WEIGHT_SIZE
                if layer_idx != 0:
                    self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, sram_address=0x10000, element_size=self.vector_length)
                    self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_INPUT_DRAM, element_size=self.vector_length)
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off)
                total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=self.head_dim * self.group_size,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off, data_type=TYPE.INT4)
                total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=self.head_dim,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off, data_type=TYPE.INT4)
                total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=self.head_dim,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off, data_type=TYPE.INT4)

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
                        self.V_CACHE_SIZE_REG, k_cache_base, self.TMP_REG)
                    self.sram_to_accelerator_memory(0x10000, 0, half_ahd)
                    self.overwrite_instruction_with_general_register(self.TMP_REG)
                    self.generate_instruction_add_imm(
                        self.V_CACHE_SIZE_REG, k_cache_base + half_ahd * bpe, self.TMP_REG)
                    self.sram_to_accelerator_memory(0x10080, 0, half_ahd)
                    self.overwrite_instruction_with_general_register(self.TMP_REG)

                    # Scatter V_h (64-dim, standard layout) → V cache at decode position
                    # v_proj output at LAYER0_FLASH_V_DRAM: [V_KV0(64)..V_KV7(64)] = 512-dim
                    self.accelerator_memory_to_sram(
                        self.LAYER0_FLASH_V_DRAM + kv_h * ahd * bpe, 0x20000, ahd)
                    self.generate_instruction_add_imm(
                        self.V_CACHE_SIZE_REG, v_cache_base, self.TMP_REG)
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
                total_flops += self.quantized_matmat_core(M=1, K=self.head_dim * self.group_size, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off, data_type=TYPE.INT4)

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
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off, data_type=TYPE.INT4, silu_enable=True)
                total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=self.mlp_elements,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off, data_type=TYPE.INT4)

                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_GATE_DRAM, sram_address=0x10000, element_size=self.mlp_elements)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_UP_DRAM, sram_address=0x90000, element_size=self.mlp_elements)
                self.eltwise_mul_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.mlp_elements)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_MLP_MULT_DRAM, element_size=self.mlp_elements)

                total_flops += self.quantized_matmat_core(M=1, K=self.mlp_elements, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off, data_type=TYPE.INT4)

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
        """Run decode loop. seq_len capped at MAX_CONTEXT_SIZE. Breaks on LLaMA EOS/EOT tokens."""
        if token_id is None:
            print("No last token available for decode.")
            return {}

        # LLaMA stop tokens: <|end_of_text|>=128001, <|eom_id|>=128008, <|eot_id|>=128009
        _llama_stop_tokens = {128001, 128008, self._end_of_turn_token_id}

        global _SILENT_MODE
        max_seq_len = self.MAX_CONTEXT_SIZE
        while self.seq_len < max_seq_len:
            _SILENT_MODE = True
            timer_start = time.perf_counter()
            self.seq_len += 1
            aligned_seq_len = ((self.seq_len + 63) // 64) * 64
            prog_idx = min((self.seq_len - 1) // 64, 7)
            prog_addr = decoder_base_addr + sum(decoder_program_sizes[:prog_idx])
            gflops = gflops_per_token[prog_idx] if gflops_per_token else None

            # V_CACHE_SIZE_REG: decode_pos × actual_hd × bpe (per-head KV cache stride = 128B)
            # ROPE_SIZE_REG:    decode_pos × head_dim × 2 × bpe (N=512 rope row = 2048B)
            _kv_stride  = self.actual_head_dim * self.bytes_per_element  # 128 bytes/position
            _rope_row   = self.head_dim * 2 * self.bytes_per_element     # 2048 bytes/position
            self.isa_add_set_core(self.V_CACHE_SIZE_REG, (self.seq_len - 1) * _kv_stride)
            self.isa_add_set_core(self.ROPE_SIZE_REG,    (self.seq_len - 1) * _rope_row)

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
    decoder_bin_path = os.path.join(script_dir, "llama3.2_1b_bin", "decoder_program.bin")
    decoder_meta_path = os.path.join(script_dir, "llama3.2_1b_bin", "decoder_program.json")
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
    timer=time.perf_counter()
    ue.run_prefill(prefill_program_addr, prefill_seq=prefill_seq, gflops=gflops_prefill)
    latency_prefill = time.perf_counter() - timer
    print(f"Prefill execute done in {latency_prefill:.2f} seconds, start decoding...\n")

    print(f"\n--- Starting decoder ---")
    timer=time.perf_counter()
    token_cnt_decoded = ue.run_decoder(decoder_program_sizes, decoder_base_addr, token_id=prefill_seq[-1], gflops_per_token=gflops_per_token)
    latency_decoder = time.perf_counter() - timer
    print(f"\nDecoder done in {latency_prefill + latency_decoder:.2f} seconds, total {token_cnt_decoded} tokens.")
    print("Llama-3.2-1B test ends.")

if __name__ == "__main__":
    main()
