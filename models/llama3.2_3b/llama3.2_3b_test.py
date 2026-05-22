#!/usr/bin/env python3
"""Llama-3.2-3B inference on accelerator: prefill + decode.

First run produces two bin files under ``llama3.2_3b_bin/``:
  * ``weights_llama3.2_3b_hf.bin``      IF4-quantized weights + bf16 embedding (~2.3 GiB)
  * ``llama3.2_3b_instruction.bin``     unified PBI-compressed bin holding all
                                         prefill buckets followed by all decoder
                                         buckets, with a ``.json`` meta listing
                                         per-bucket offsets and total FLOPs.

Subsequent runs reuse both bins from disk — no HuggingFace model load, no recompile.
The tokenizer is loaded from local files only (``local_files_only=True``).

Shape diff vs 1B: 28 layers, hidden=3072, actual_head_dim=128, num_kv_heads=8,
group_size=3, mlp=8192. Because 3B's ``actual_head_dim=128`` satisfies
``rope_hf_core``'s N≥128 per head, RoPE runs PER HEAD (N=128) with no q/k weight
permutation (vs 1B which tiles 8 KV heads in N=512). See notes/notes_llama3.2_3b.md.

Architecture (same as 1B; vs Gemma3): no per-head q/k norm, no post-attn/post-MLP
norm, ``post_attention_layernorm`` is the pre-FFN norm, embedding not sqrt-scaled,
LM head ties to embedding, ``gamma_offset = 0``.

Usage:
  python llama3.2_3b_test.py
  python llama3.2_3b_test.py --prompt "your prompt"
  python llama3.2_3b_test.py --dev xdma0 --cycle 5.88
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
import threading

# This file's folder; user_dma_core.py is two folders up (repo root); that directory is added to sys.path.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

import user_dma_core
from user_dma_core import DMA_DEVICE_H2C, TYPE, UE_FMAX_CONTEXT_SIZE, UE_VECTOR_SIZE, UE_ARGMAX_INDEX, URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS, set_dma_device, ue_35bit_addr_shifter
from user_dma_core import UnifiedEngine

# Toggled inside compile_instructions to suppress the per-instruction prints
# emitted by the user_dma_core kernels during bucket capture.
import builtins
_original_print = builtins.print
_SILENT_MODE = False
def _quiet_print(*args, **kwargs):
    if _SILENT_MODE:
        return
    _original_print(*args, **kwargs)
builtins.print = _quiet_print

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
    """Generate weights_llama3.2_3b_hf.bin from HuggingFace model per llama3.2_3b_config.json layout.
    Returns the path to the written file.

    3B path: per-head RoPE (N=actual_head_dim=128 satisfies rope_hf_core's N>=128
    constraint per head). No q/k row permutation — HF's natural per-head [lo,hi]
    layout is fed directly to rope_hf_core.
    """
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
    head_dim = cfg["file_info"]["head_dim"]                  # 1024 (= num_kv_heads * actual_head_dim)
    vector_length = cfg["file_info"]["hidden_size"]          # 3072
    group_size = cfg["file_info"]["group_size"]              # 3
    actual_head_dim = cfg["file_info"]["actual_head_dim"]    # 128
    blk0_structure = cfg["layers"]["structure"]

    # Per-head RoPE path: no row permutation needed for q_proj/k_proj. The HF
    # weights produce per-head [lo,hi]-layout outputs (each head_dim_actual=128
    # already contains the half-rotate halves), and rope_hf_core(N=128) operates
    # on each head independently.
    num_kv_heads = head_dim // actual_head_dim               # 8
    num_q_heads = vector_length // actual_head_dim           # 24 (= num_kv_heads * group_size)

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
        # 3B per-head RoPE: no row permutation. HF Q/K outputs are
        # (num_q_heads, head_dim_actual=128) / (num_kv_heads, 128) and each head's
        # [lo(64), hi(64)] layout is exactly what rope_hf_core(N=128) expects.
        q_w = attn.q_proj.weight.detach().cpu().to(torch.bfloat16)
        k_w = attn.k_proj.weight.detach().cpu().to(torch.bfloat16)
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

    # ROPE: LLaMA uses a single RoPE table (rope_global_layers is empty, all layers use local/default).
    # 3B per-head: NO tiling across KV heads — each rope_hf_core(N=actual_head_dim=128)
    # consumes a single head's worth of cos/sin (D_per_head=64). Each per-position
    # row = [cos_head(64), cos_head(64), -sin_head(64), sin_head(64)] = 256 elements.
    rope_cfg = cfg["special"]["rope"]
    theta = rope_cfg["theta"]
    local_base = rope_cfg["local_base"]
    num_positions = rope_cfg["num_positions"]
    D_per_head = actual_head_dim // 2           # = 64
    for name, theta_val, off_key, sz_key in [
        ("ROPE_LOCAL", local_base, "ROPE_LOCAL", "ROPE_LOCAL_SIZE"),
        ("ROPE_GLOBAL", theta, "ROPE_GLOBAL", "ROPE_GLOBAL_SIZE"),
    ]:
        inv_freq = 1.0 / (theta_val ** (torch.arange(D_per_head, dtype=torch.float32) / D_per_head))
        pos = torch.arange(num_positions, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)                     # (num_positions, 64)
        cos_head = freqs.cos().to(torch.bfloat16)              # (num_positions, 64)
        sin_head = freqs.sin().to(torch.bfloat16)              # (num_positions, 64)
        # Per-head layout (no KV-head tiling): [cos_head, cos_head, -sin_head, sin_head]
        rope_tensor = torch.cat([cos_head, cos_head, -sin_head, sin_head], dim=1)  # (num_pos, 256)
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
    """Load llama3.2_3b_config.json and build weight_defs (offset/size dict) from regions."""
    config_path = os.path.join(script_dir, "llama3.2_3b_config.json")
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
# Llama-3.2-3B unified engine
# -----------------------------------------------------------------------------
class Llama32_3b_UnifiedEngine(UnifiedEngine):
    """UnifiedEngine for Llama-3.2-3B: loads config + weight bin, compile_prefill/compile_decoder, run_prefill/run_decoder.

    Same architectural family as 1B (GQA, no q/k/post-attn/post-mlp norms, tied lm_head)
    but with different shapes (28 layers, hidden=3072, head_dim_actual=128, 8 KV heads,
    group_size=3) and a per-head RoPE pipeline (N=128 per call) — see module docstring.
    """

    def __init__(self, script_dir: str | None = None, hf_model_dir: str | None = None, weights_bin: str | None = None):
        # Override default DRAM bases (params 768 MB, tensor 512 MB, program 768 MB)
        # because 3B layer weights alone are ~1.5 GB. Mirror Gemma4's full-4-GB
        # layout (see memory notes: gemma4_fixed_dram_layout, fpga_params_dram_budget).
        #   Params : 0x00000000 – 0x80000000   (2 GiB)  — fits 1.7 GiB weight bin with room
        #   Tensor : 0x80000000 – 0xA0000000   (512 MiB) — KV cache (~60 MB) + activations
        #   Program: 0xA0000000 – 0x100000000  (1.5 GiB) — prefill + decoder bin
        super().__init__(params_dram_base=0x00000000,
                         tensor_dram_base=0x80000000,
                         program_dram_base=0xA0000000)
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
        # LLaMA 3.2 3B GQA: 8 KV heads × 128-dim per head = head_dim=1024 combined
        # actual_head_dim must come from config; 1B uses 64, 3B uses 128.
        self.actual_head_dim = fi["actual_head_dim"]
        self.num_kv_heads = self.head_dim // self.actual_head_dim
        self.num_q_heads = self.vector_length // self.actual_head_dim  # = num_kv_heads * group_size
        self.MAX_CONTEXT_SIZE = model["max_context_size"]
        self.LAYER_SIZE = fi["num_layers"]
        self.EMBEDDING_ELEMENTS = fi["embedding_vocab"]
        # PBI loop counters borrow ISA regs via alloc_isa_reg(). Regs 1-3 are
        # reserved for V_CACHE_SIZE_REG / TMP_REG / ROPE_SIZE_REG (see config);
        # reg 4 is left spare. PBI ops allocate counter regs from 5 onward, so
        # loop_start never clobbers V_CACHE_SIZE_REG (which would corrupt the
        # decoder K/V scatter at runtime).
        self._isa_reg_counter = 5
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
        """Reset the ISA register allocation counter to 5 (regs 1-4 reserved)."""
        self._isa_reg_counter = 5

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
            self._isa_reg_counter = 5

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
        """Generate per-head RoPE table (N=actual_head_dim=128 per call) and write to DRAM.

        3B uses per-head RoPE because actual_head_dim=128 already satisfies
        rope_hf_core's N>=128 constraint. Layout per position:
            [cos_head(D), cos_head(D), -sin_head(D), sin_head(D)]
        where D = actual_head_dim // 2 = 64.  Total row = 4*D*bpe = 512 bytes.
        rope_hf_core(N=128) on each Q/K head reads cos at row_offset and sin at
        row_offset + N*bpe = row_offset + 256.
        """
        rope_cfg = self._cfg["special"]["rope"]
        theta = rope_theta if rope_theta is not None else rope_cfg["theta"]
        local_base = rope_local_base if rope_local_base is not None else rope_cfg["local_base"]
        num_rope_positions = rope_cfg["num_positions"]
        D_per_head = self.actual_head_dim // 2  # = 64 frequencies per head
        for name, theta_val, sz_key, attr in [
            ("ROPE_LOCAL", local_base, "ROPE_LOCAL_SIZE", "DRAM_ADDR_ROPE_LOCAL"),
            ("ROPE_GLOBAL", theta, "ROPE_GLOBAL_SIZE", "DRAM_ADDR_ROPE_GLOBAL"),
        ]:
            inv_freq = 1.0 / (theta_val ** (torch.arange(D_per_head, dtype=torch.float32) / D_per_head))
            pos = torch.arange(num_rope_positions, dtype=torch.float32)
            freqs = torch.outer(pos, inv_freq)
            cos_head = freqs.cos().to(torch.bfloat16)  # (num_pos, 64)
            sin_head = freqs.sin().to(torch.bfloat16)  # (num_pos, 64)
            # Per-head layout (no tiling): [cos_head, cos_head, -sin_head, sin_head]
            rope_tensor = torch.cat([cos_head, cos_head, -sin_head, sin_head], dim=1)  # (num_pos, 256)
            sz = self.weight_defs[sz_key]
            raw = rope_tensor.contiguous().view(torch.uint8).numpy().tobytes()
            raw = (raw + b"\x00" * sz)[:sz]
            addr = self.allocate_params_dram(sz)
            self.dma_write(DMA_DEVICE_H2C, addr, raw, sz)
            setattr(self, attr, addr)

    def weight_init(self) -> None:
        """Initialize DRAM from local files only.

        Embedding is read from the weight bin (already serialized there by
        weight_bin_generate). Tokenizer is loaded from the local HF model
        directory if it exists. The HF model itself is NEVER loaded here —
        only weight_bin_generate touches HuggingFace.
        """
        # Embedding from weight bin (bf16 stored as raw bytes at token_embd_offset)
        emb_cfg = self._cfg["special"]["embedding"]
        emb_off = _parse_offset(emb_cfg["token_embd_offset"])
        vocab_size = emb_cfg["vocab_size"]
        emb_dim = emb_cfg["embedding_dim"]
        emb_nbytes = vocab_size * emb_dim * self.bytes_per_element
        emb_raw = self.weight_bin[emb_off : emb_off + emb_nbytes]
        emb_u16 = np.frombuffer(emb_raw, dtype=np.uint16).copy()
        self.embedding_weight = torch.from_numpy(emb_u16).view(torch.bfloat16).reshape(vocab_size, emb_dim)

        # Tokenizer from local files only (no network)
        model_dir = os.path.join(self.script_dir, self._cfg["paths"]["hf_model_dir"])
        if not os.path.exists(os.path.join(model_dir, "tokenizer_config.json")) and \
           not os.path.exists(os.path.join(model_dir, "tokenizer.json")):
            raise RuntimeError(
                f"Tokenizer files not found at {model_dir}. "
                "First-time setup requires the HF model to be downloaded "
                "(run weight_bin_generate to fetch it). After that, the "
                "tokenizer files persist locally and no HF access is needed."
            )
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
        """Initialize hardware DRAM tensors for Llama-3.2-3B (layer-wise overlap except for kv cache)."""
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

    def _emit_prefill_program(self, seq_len: int, layer_size: int) -> int:
        """Emit instructions for ONE prefill bucket (no capture-session boundary).

        Caller must wrap this in start_capture()/stop_capture(). The emitted
        program ends with a halt so a bucket-program can be executed as a unit.
        Returns total_flops for this bucket.

        seq_len here is the actual prefill length (i.e., bucket_size - 1, since
        prefill processes all but the last token of the prompt).

        All PBI-able ops (rms_norm, matmat, rope, flash) emit
        ``loop_start``/``loop_end`` rather than unrolled instruction streams,
        which compresses the captured bin ~3× vs the unrolled path.
        """
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        total_flops = 0
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        for layer_idx in range(layer_size):
            layer_off = layer_idx * LAYER_WEIGHT_SIZE
            if layer_idx != 0:
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, sram_address=0x10000, element_size=seq_len * self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_INPUT_DRAM, element_size=seq_len * self.vector_length)
            total_flops += (self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off,
                              use_pbi=True) or 0)

            total_flops += (self.matmat_mul_core(M=seq_len, K=self.vector_length, N=self.head_dim * self.group_size,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off, use_pbi=True) or 0)
            total_flops += (self.matmat_mul_core(M=seq_len, K=self.vector_length, N=self.head_dim,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off, use_pbi=True) or 0)
            # v_proj writes to interleaved temp (T, hd); per-head KV cache populated below.
            total_flops += (self.matmat_mul_core(M=seq_len, K=self.vector_length, N=self.head_dim,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_V_PROJ_TEMP,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off, use_pbi=True) or 0)

            # LLaMA 3B per-head GQA: rope_hf_core(N=actual_head_dim=128) on each Q/K head
            # independently using HF's native per-head [lo,hi] layout. Then scatter
            # head_dim_actual-sized per-head slices into per-head flash buffers and KV cache,
            # then run flash_attention(head_dim=128) × 8 KV heads.
            #
            # NOTE: no q/k weight row permutation in weight_bin_generate for 3B.
            # Q output per token is laid out as (num_q_heads × actual_head_dim) =
            # (24 × 128) = 3072 in HF head order. K output is (8 × 128) = 1024.
            # Each per-head slice is contiguous in DRAM.
            ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE_GLOBAL if layer_idx in self._rope_global_layers else self.DRAM_ADDR_ROPE_LOCAL
            ahd      = self.actual_head_dim   # 128
            nkvh     = self.num_kv_heads      # 8
            nqh      = self.num_q_heads       # 24
            qpkv     = self.group_size        # 3 (= num_q_heads // num_kv_heads)
            bpe      = self.bytes_per_element
            hd       = self.head_dim          # 1024 (combined KV-head width)
            total_q_dim = self.vector_length  # 3072 (o_proj input width = num_q_heads * ahd)
            rope_row = 2 * ahd * bpe          # 512 bytes per rope table row (N=128)

            # Phase 1: K per-head RoPE in-place on LAYER0_K_DRAM (N=ahd=128 per head).
            # K layout (seq_len, num_kv_heads, ahd) matches rope_hf_core_dram_gqa's
            # [M, group_size, N] expectation. cos/sin shared across heads per token.
            # PBI assertion: sin_dram_addr == cos_dram_addr + N*bpe ✓ (256B = ahd*bpe).
            total_flops += self.rope_hf_core_dram_gqa(
                M=seq_len, group_size=nkvh, N=ahd,
                input_dram_addr=self.LAYER0_K_DRAM,
                output_dram_addr=self.LAYER0_K_DRAM,
                cos_dram_addr=ROPE_WEIGHT_ADDR,
                sin_dram_addr=ROPE_WEIGHT_ADDR + ahd * bpe,
                use_pbi=True)

            # Phase 2: Q per-head RoPE in-place on LAYER0_Q_DRAM (N=ahd=128 per head).
            # Q layout (seq_len, num_q_heads, ahd) → same gqa shape, group_size=24.
            total_flops += self.rope_hf_core_dram_gqa(
                M=seq_len, group_size=nqh, N=ahd,
                input_dram_addr=self.LAYER0_Q_DRAM,
                output_dram_addr=self.LAYER0_Q_DRAM,
                cos_dram_addr=ROPE_WEIGHT_ADDR,
                sin_dram_addr=ROPE_WEIGHT_ADDR + ahd * bpe,
                use_pbi=True)

            # Phase 3: Per-KV-head scatter → KV cache + flash buffers → flash_attention
            # Each kv_h owns qpkv=3 consecutive Q heads in HF order (3*kv_h .. 3*kv_h+2).
            for kv_h in range(nkvh):
                k_cache_base = (self.LAYER0_K_ROPE_DRAM
                                + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)
                v_cache_base = (self.LAYER0_V_DRAM
                                + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)

                # Scatter K head (ahd-dim, already [lo,hi] per head after rope) → KV cache + FLASH_K
                for t in range(seq_len):
                    k_src = self.LAYER0_K_DRAM + t * hd * bpe + kv_h * ahd * bpe
                    self.accelerator_memory_to_sram(k_src, 0x10000, ahd)
                    self.sram_to_accelerator_memory(0x10000, k_cache_base + t * ahd * bpe, ahd)
                    for g in range(qpkv):
                        k_f = self.LAYER0_FLASH_K_DRAM + (t * qpkv + g) * ahd * bpe
                        self.sram_to_accelerator_memory(0x10000, k_f, ahd)

                # Scatter V head (ahd-dim, no rope) from V_PROJ_TEMP → KV cache + FLASH_V
                # V_PROJ_TEMP[t] layout: [V_KV0(ahd), V_KV1(ahd), ..., V_KV7(ahd)] = hd
                for t in range(seq_len):
                    v_src = self.LAYER0_V_PROJ_TEMP + (t * nkvh + kv_h) * ahd * bpe
                    self.accelerator_memory_to_sram(v_src, 0x20000, ahd)
                    self.sram_to_accelerator_memory(0x20000, v_cache_base + t * ahd * bpe, ahd)
                    for g in range(qpkv):
                        self.sram_to_accelerator_memory(
                            0x20000,
                            self.LAYER0_FLASH_V_DRAM + (t * qpkv + g) * ahd * bpe, ahd)

                # Scatter Q heads (qpkv per KV head, each ahd-dim) → FLASH_Q
                # Q heads belonging to KV head kv_h: indices kv_h*qpkv .. kv_h*qpkv + qpkv - 1
                for t in range(seq_len):
                    for q in range(qpkv):
                        q_src = self.LAYER0_Q_DRAM + t * total_q_dim * bpe + (kv_h * qpkv + q) * ahd * bpe
                        q_dst = self.LAYER0_FLASH_Q_DRAM + (t * qpkv + q) * ahd * bpe
                        self.accelerator_memory_to_sram(q_src, 0x30000, ahd)
                        self.sram_to_accelerator_memory(0x30000, q_dst, ahd)

                # Flash attention for this KV head (head_dim=ahd=128)
                total_flops += self.flash_attention_core(
                    head_dim=ahd,
                    seq_len=aligned_seq_len,
                    Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
                    K_DRAM_ADDR=self.LAYER0_FLASH_K_DRAM,
                    V_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                    OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUT_HEAD_DRAM,
                    SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                    BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_DRAM,
                    use_pbi=True,
                )

                # Assemble output into LAYER0_FLASH_OUTPUT_DRAM
                # Standard GQA layout per token: [kv0_q0(ahd), kv0_q1, kv0_q2, kv1_q0, ...]
                out_h_base = kv_h * qpkv * ahd * bpe
                for t in range(seq_len):
                    for g in range(qpkv):
                        src = self.LAYER0_FLASH_OUT_HEAD_DRAM + (t * qpkv + g) * ahd * bpe
                        dst = (self.LAYER0_FLASH_OUTPUT_DRAM
                               + t * total_q_dim * bpe + out_h_base + g * ahd * bpe)
                        self.accelerator_memory_to_sram(src, 0x40000, ahd)
                        self.sram_to_accelerator_memory(0x40000, dst, ahd)

            total_flops += (self.matmat_mul_core(M=seq_len, K=self.head_dim * self.group_size, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off, use_pbi=True) or 0)

            # LLaMA: no post-attention norm; add residual directly to o_proj output
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_INPUT_DRAM, sram_address=0x10000, element_size=seq_len * self.vector_length)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM, sram_address=0x90000, element_size=seq_len * self.vector_length)
            self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=seq_len * self.vector_length)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, element_size=seq_len * self.vector_length)

            # LLaMA: post_attention_layernorm IS the pre-FFN norm
            total_flops += (self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off,
                              use_pbi=True) or 0)
            total_flops += (self.matmat_mul_core(M=seq_len, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off, silu_enable=True,
                use_pbi=True) or 0)
            total_flops += (self.matmat_mul_core(M=seq_len, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off,
                use_pbi=True) or 0)
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
            total_flops += (self.matmat_mul_core(M=seq_len, K=self.mlp_elements, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off,
                use_pbi=True) or 0)

            # LLaMA: no post-FFN norm; add residual directly to down_proj output
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, sram_address=0x10000, element_size=seq_len * self.vector_length)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_DOWN_DRAM, sram_address=0x90000, element_size=seq_len * self.vector_length)
            self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=seq_len * self.vector_length)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, element_size=seq_len * self.vector_length)
        # Per-bucket halt so each bucket-program is independently executable.
        self.generate_instruction_halt()
        return total_flops


    def run_prefill(self, prefill_program_addr: int, prefill_seq, gflops: int = None,
                    bucket_seq_len: int | None = None) -> dict:
        """Run prefill for the given prefill sequence.

        Args:
            prefill_program_addr: The address of the prefill bucket program in DRAM.
            prefill_seq: The full prompt; last token is the decoder's seed and is
                dropped from the prefill input.
            gflops: Total FLOP count for the prefill program (for GFLOPS reporting).
            bucket_seq_len: If set, the prefill program was compiled for this many
                positions and the actual sequence is padded out to that length with
                the last-actual-token embedding. ``self.seq_len`` is set to the
                actual length after execute so the decoder starts at the correct
                position; padded KV slots are masked out by the decoder's bias mask.
        """
        if prefill_seq is None:
            prefill_seq = tuple(self._cfg["default_prefill_tokens"])

        if len(prefill_seq) > 1:
            prefill_seq = prefill_seq[:-1]
            assert len(prefill_seq) == self.seq_len, f"Expected seq_len {self.seq_len}, but got {len(prefill_seq)}"
        else:
            raise ValueError("Prefill sequence must have at least 2 tokens.")

        actual_seq_len = len(prefill_seq)
        if bucket_seq_len is None:
            bucket_seq_len = actual_seq_len
        assert bucket_seq_len >= actual_seq_len, (
            f"bucket_seq_len={bucket_seq_len} must be >= actual_seq_len={actual_seq_len}"
        )

        q_seq_len = bucket_seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64

        embedding_tensor = self.get_embedding_for_tokens(prefill_seq)
        if bucket_seq_len > actual_seq_len:
            pad = embedding_tensor[-1:].repeat(bucket_seq_len - actual_seq_len, 1)
            embedding_tensor = torch.cat([embedding_tensor, pad], dim=0)
        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)

        # Block-causal mask sized for the bucket. Cols beyond q_seq_len stay -inf.
        bias_one_group = torch.full((aligned_seq_len, aligned_seq_len), float("-inf"), dtype=torch.bfloat16)
        rows = torch.arange(aligned_seq_len).unsqueeze(1)
        cols = torch.arange(aligned_seq_len).unsqueeze(0)
        valid_mask = (cols // self.group_size) <= (rows // self.group_size)
        bias_one_group.masked_fill_(valid_mask, 0.0)
        bias_one_group[:, q_seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_one_group)

        self.program_execute(prefill_program_addr, timeout=120.0, gflops=gflops)
        # Decoder reads ctx 0..actual_seq_len-1; padded KV slots are masked out.
        self.seq_len = actual_seq_len

    def _emit_decoder_program(self, seq_len: int, layer_size: int) -> int:
        """Emit instructions for ONE decoder bucket (no capture-session boundary).

        Caller wraps this in start_capture()/stop_capture(). The emitted program
        ends with a halt so a bucket-program can be executed as a unit. Returns
        total_flops for this bucket. The bucket value ``seq_len`` is the kv-context
        length the decoder will run against.

        All PBI-able ops (rms_norm, matmat) emit loop_start/loop_end. Decoder
        RoPE uses register-indirect cos/sin via ROPE_SIZE_REG (legacy path only),
        and decoder_attention_core has no PBI path, so those remain unrolled.
        """
        total_flops = 0
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        for layer_idx in range(layer_size):
            layer_off = layer_idx * LAYER_WEIGHT_SIZE
            if layer_idx != 0:
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, sram_address=0x10000, element_size=self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_INPUT_DRAM, element_size=self.vector_length)
            total_flops += (self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                          OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off,
                          use_pbi=True) or 0)
            total_flops += (self.matmat_mul_core(M=1, K=self.vector_length, N=self.head_dim * self.group_size,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off, use_pbi=True) or 0)
            total_flops += (self.matmat_mul_core(M=1, K=self.vector_length, N=self.head_dim,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off, use_pbi=True) or 0)
            total_flops += (self.matmat_mul_core(M=1, K=self.vector_length, N=self.head_dim,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off, use_pbi=True) or 0)

            # LLaMA 3B per-head GQA decoder: rope_hf_core(N=128) per head on K and Q
            # in-place (using ROPE_SIZE_REG for decode position), then scatter
            # ahd-dim slices to KV cache (via V_CACHE_SIZE_REG for decode position),
            # then decoder_attention(head_dim=128).
            ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE_GLOBAL if layer_idx in self._rope_global_layers else self.DRAM_ADDR_ROPE_LOCAL
            ahd      = self.actual_head_dim   # 128
            nkvh     = self.num_kv_heads      # 8
            nqh      = self.num_q_heads       # 24
            qpkv     = self.group_size        # 3
            bpe      = self.bytes_per_element
            hd       = self.head_dim          # 1024
            total_q_dim = self.vector_length  # 3072

            # Step 1: K per-head rope in-place (N=128 per head, ROPE_SIZE_REG = decode pos × rope_row)
            for kv_h in range(nkvh):
                k_h_addr = self.LAYER0_K_DRAM + kv_h * ahd * bpe
                total_flops += self.rope_hf_core(
                    N=ahd,
                    input_dram_addr=k_h_addr,
                    output_dram_addr=k_h_addr,
                    cos_dram_addr=ROPE_WEIGHT_ADDR,
                    sin_dram_addr=ROPE_WEIGHT_ADDR + ahd * bpe,
                    rope_size_reg=self.ROPE_SIZE_REG,
                    tmp_reg=self.TMP_REG)

            # Step 2: Q per-head rope in-place (N=128 per Q head, ROPE_SIZE_REG)
            for q_h in range(nqh):
                q_h_addr = self.LAYER0_Q_DRAM + q_h * ahd * bpe
                total_flops += self.rope_hf_core(
                    N=ahd,
                    input_dram_addr=q_h_addr,
                    output_dram_addr=q_h_addr,
                    cos_dram_addr=ROPE_WEIGHT_ADDR,
                    sin_dram_addr=ROPE_WEIGHT_ADDR + ahd * bpe,
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

                # Scatter K head (ahd-dim) → KV cache at decode position (V_CACHE_SIZE_REG)
                self.accelerator_memory_to_sram(
                    self.LAYER0_K_DRAM + kv_h * ahd * bpe, 0x10000, ahd)
                self.generate_instruction_add_imm(
                    self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter(k_cache_base), self.TMP_REG)
                self.sram_to_accelerator_memory(0x10000, 0, ahd)
                self.overwrite_instruction_with_general_register(self.TMP_REG)

                # Scatter V head (ahd-dim) → V cache at decode position
                # v_proj output at LAYER0_FLASH_V_DRAM: [V_KV0(ahd)..V_KV7(ahd)] = hd
                self.accelerator_memory_to_sram(
                    self.LAYER0_FLASH_V_DRAM + kv_h * ahd * bpe, 0x20000, ahd)
                self.generate_instruction_add_imm(
                    self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter(v_cache_base), self.TMP_REG)
                self.sram_to_accelerator_memory(0x20000, 0, ahd)
                self.overwrite_instruction_with_general_register(self.TMP_REG)

                # Scatter Q heads belonging to kv_h (qpkv=3 of them) → FLASH_Q → decoder_attn
                for q in range(qpkv):
                    q_src = self.LAYER0_Q_DRAM + (kv_h * qpkv + q) * ahd * bpe
                    flash_q_addr = self.LAYER0_FLASH_Q_DRAM + (kv_h * qpkv + q) * ahd * bpe
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
            total_flops += (self.matmat_mul_core(M=1, K=self.head_dim * self.group_size, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off, use_pbi=True) or 0)

            # LLaMA: no post-attention norm; residual directly on o_proj output
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_INPUT_DRAM, sram_address=0x10000, element_size=self.vector_length)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM, sram_address=0x90000, element_size=self.vector_length)
            self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, element_size=self.vector_length)

            # LLaMA: post_attention_layernorm IS the pre-FFN norm
            total_flops += (self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                          OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off,
                          use_pbi=True) or 0)

            total_flops += (self.matmat_mul_core(M=1, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off, silu_enable=True,
                use_pbi=True) or 0)
            total_flops += (self.matmat_mul_core(M=1, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off,
                use_pbi=True) or 0)

            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_GATE_DRAM, sram_address=0x10000, element_size=self.mlp_elements)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_UP_DRAM, sram_address=0x90000, element_size=self.mlp_elements)
            self.eltwise_mul_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.mlp_elements)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_MLP_MULT_DRAM, element_size=self.mlp_elements)

            total_flops += (self.matmat_mul_core(M=1, K=self.mlp_elements, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off,
                use_pbi=True) or 0)

            # LLaMA: no post-FFN norm; residual directly on down_proj output
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, sram_address=0x10000, element_size=self.vector_length)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_DOWN_DRAM, sram_address=0x90000, element_size=self.vector_length)
            self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, element_size=self.vector_length)

        if layer_size == self.LAYER_SIZE:
            total_flops += (self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_OUTPUT_DRAM,
                OUTPUT_DRAM_ADDR=self.OUTPUT_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_OUTPUT_NORM_GAMMA,
                use_pbi=True) or 0)
            total_flops += (self.matmat_mul_core(M=1, K=self.vector_length, N=self.EMBEDDING_ELEMENTS,
                A_DRAM_ADDR=self.OUTPUT_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_QUANT, OUTPUT_DRAM_ADDR=self.LOGITS_DRAM,
                is_B_quantized=True, data_type=TYPE.IF4,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_SCALE, use_pbi=True) or 0)

        # Per-bucket halt so each bucket-program is independently executable.
        self.generate_instruction_halt()
        return total_flops

    def compile_instructions(self, layer_size: int | None = None) -> dict:
        """Compile a UNIFIED instruction bin holding all prefill buckets followed
        by all decoder buckets in one capture session, then write a single
        ``llama3.2_3b_instruction.bin`` and matching ``.json`` meta to disk.

        If the bin + meta already exist on disk, this is a no-op: it just loads
        the meta and returns it. Subsequent runs need no HF access and no
        recompile — they DMA the bin into program DRAM and run.

        Returns the meta dict with prefill_buckets/program_sizes/total_flops and
        decoder_buckets/program_sizes/total_flops.
        """
        if layer_size is None:
            layer_size = self.LAYER_SIZE

        paths_cfg = self._cfg.get("paths", {})
        bin_rel  = paths_cfg.get("instruction_bin",  "llama3.2_3b_bin/llama3.2_3b_instruction.bin")
        meta_rel = paths_cfg.get("instruction_meta", "llama3.2_3b_bin/llama3.2_3b_instruction.json")
        bin_path  = os.path.join(self.script_dir, bin_rel)
        meta_path = os.path.join(self.script_dir, meta_rel)

        if os.path.exists(bin_path) and os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"Instruction bin found, skipping compile: {bin_path}")
            return meta

        os.makedirs(os.path.dirname(bin_path), exist_ok=True)
        prefill_buckets = self._cfg["model"]["prefill_seq_len_buckets"]
        decoder_buckets = self._cfg["model"]["decoder_seq_len_buckets"]

        global _SILENT_MODE
        _SILENT_MODE = True
        self.clear_inst_id()
        self.start_capture()

        prefill_program_sizes: list[int] = []
        prefill_total_flops:   list[int] = []
        for bucket in prefill_buckets:
            count_at_start = self.capture_count
            flops = self._emit_prefill_program(seq_len=bucket - 1, layer_size=layer_size)
            prefill_program_sizes.append((self.capture_count - count_at_start) * 32)
            prefill_total_flops.append(flops)

        decoder_program_sizes: list[int] = []
        decoder_total_flops:   list[int] = []
        for bucket in decoder_buckets:
            count_at_start = self.capture_count
            flops = self._emit_decoder_program(seq_len=bucket, layer_size=layer_size)
            decoder_program_sizes.append((self.capture_count - count_at_start) * 32)
            decoder_total_flops.append(flops)

        self.stop_capture()
        _SILENT_MODE = False

        # Sanity check: ensure capture didn't hit MAX_DECODER_INSTRUCTIONS cap.
        from user_dma_core import MAX_DECODER_INSTRUCTIONS
        if self.capture_count >= MAX_DECODER_INSTRUCTIONS:
            raise RuntimeError(
                f"Capture hit MAX_DECODER_INSTRUCTIONS cap "
                f"({MAX_DECODER_INSTRUCTIONS} instructions = "
                f"{MAX_DECODER_INSTRUCTIONS * 32 / 2**20:.0f} MiB). "
                f"Captured {self.capture_count} instructions = "
                f"{self.capture_count * 32 / 2**20:.0f} MiB. Shrink bucket lists."
            )

        all_bytes = bytearray()
        for inst in self.capture_buffer:
            all_bytes.extend(inst.get_bytes())
        with open(bin_path, "wb") as f:
            f.write(all_bytes)

        meta = {
            "prefill_buckets":       prefill_buckets,
            "prefill_program_sizes": prefill_program_sizes,
            "prefill_total_flops":   prefill_total_flops,
            "decoder_buckets":       decoder_buckets,
            "decoder_program_sizes": decoder_program_sizes,
            "decoder_total_flops":   decoder_total_flops,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=0)
        self.clear_capture_buffer()
        print(f"Instruction bin written: {bin_path} "
              f"({len(all_bytes)} bytes, {len(prefill_buckets)} prefill + {len(decoder_buckets)} decoder buckets)")
        return meta

    def run_decoder(self, decoder_program_sizes: list[int], decoder_base_addr: int, token_id: int, gflops_per_token: list[int] | None = None) -> dict:
        """Run decode loop. seq_len capped at MAX_CONTEXT_SIZE. Breaks on LLaMA EOS/EOT tokens."""
        if token_id is None:
            print("No last token available for decode.")
            return {}

        # LLaMA stop tokens: <|end_of_text|>=128001, <|eom_id|>=128008, <|eot_id|>=128009
        _llama_stop_tokens = {128001, 128008, self._end_of_turn_token_id}

        global _SILENT_MODE
        max_seq_len = self.MAX_CONTEXT_SIZE
        decoded_count = 0
        per_token_times: list[float] = []
        decode_loop_start = time.perf_counter()
        while self.seq_len < max_seq_len:
            _SILENT_MODE = True
            timer_start = time.perf_counter()
            self.seq_len += 1
            aligned_seq_len = ((self.seq_len + 63) // 64) * 64
            prog_idx = min((self.seq_len - 1) // 64, 7)
            prog_addr = decoder_base_addr + sum(decoder_program_sizes[:prog_idx])
            gflops = gflops_per_token[prog_idx] if gflops_per_token else None

            # V_CACHE_SIZE_REG: decode_pos × actual_hd × bpe (per-head KV cache stride = 256B for 3B)
            # ROPE_SIZE_REG:    decode_pos × 2 × actual_hd × bpe (per-head rope row = 512B for 3B)
            _kv_stride  = self.actual_head_dim * self.bytes_per_element       # 256 bytes/position (3B)
            _rope_row   = 2 * self.actual_head_dim * self.bytes_per_element   # 512 bytes/position (3B)
            self.isa_add_set_core(self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter((self.seq_len - 1) * _kv_stride))
            self.isa_add_set_core(self.ROPE_SIZE_REG,    ue_35bit_addr_shifter((self.seq_len - 1) * _rope_row))

            embedding_tensor = self.get_embedding_for_tokens([token_id])
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)

            bias_host = torch.full((1, aligned_seq_len), -1e36, dtype=torch.bfloat16)
            bias_host[0, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_host)

            self.start_execute_from_dram(prog_addr)
            # Per-token decode for 3B is ~0.5 s; 30 s gives 60× margin and
            # avoids the stale-latency-register pitfall hit on prefill.
            self.wait_queue(30.0)
            token_id = self.get_arg_max_index()
            token_char = self.tokenizer.decode([token_id])
            per_token_times.append(time.perf_counter() - timer_start)
            decoded_count += 1
            _SILENT_MODE = False
            if token_id in _llama_stop_tokens:
                print(f"\nStop token {token_id} reached.")
                break
            print(token_char, end="", flush=True)
        decode_elapsed = time.perf_counter() - decode_loop_start
        tps = decoded_count / decode_elapsed if decode_elapsed > 0 else 0.0
        avg_ms = (sum(per_token_times) / len(per_token_times) * 1000.0) if per_token_times else 0.0
        print(f"\n--- Decode benchmark ---")
        print(f"  Tokens decoded : {decoded_count}")
        print(f"  Elapsed        : {decode_elapsed:.3f} s")
        print(f"  Throughput     : {tps:.3f} tokens/s")
        print(f"  Avg per token  : {avg_ms:.2f} ms")
        return {
            "decoded_count": decoded_count,
            "seq_len": self.seq_len,
            "decode_elapsed_s": decode_elapsed,
            "tokens_per_second": tps,
            "avg_per_token_ms": avg_ms,
            "per_token_times": per_token_times,
        }
        
# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Llama-3.2-3B prefill + decode on accelerator.")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt; tokenized with the model's chat template. "
                             "Overrides the default prompt from the config.")
    parser.add_argument("--dev", type=str, default="xdma0",
                        help="DMA device name (e.g. xdma0). Default: xdma0.")
    parser.add_argument("--cycle", type=float, default=1/0.17,
                        help="Clock cycle time in nanoseconds. Default: ~5.88 ns.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_bin_rel = "llama3.2_3b_bin/weights_llama3.2_3b_hf.bin"
    weights_bin_full = os.path.join(script_dir, weights_bin_rel)
    if not os.path.exists(weights_bin_full):
        # First-time setup: download the HF model and produce the weight bin.
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

    ue = Llama32_3b_UnifiedEngine(script_dir=script_dir, weights_bin=weights_bin_rel)
    cfg = _load_config(script_dir)
    if args.prompt is not None:
        conversation = [{"role": "user", "content": args.prompt}]
        prompt_with_template = ue.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True)
        prefill_seq = tuple(ue.tokenizer.encode(prompt_with_template, add_special_tokens=False))
        print(f"Prefill from prompt ({len(prefill_seq)} tokens): {args.prompt!r}")
    else:
        prefill_seq = tuple(cfg["default_prefill_tokens"])

    # === Unified instruction bin: compile-once cached on disk, no recompile on rerun ===
    print(f"\n--- Compiling instruction bin (if not cached) ---")
    timer = time.perf_counter()
    _compile_done = threading.Event()
    def _compile_heartbeat():
        # Bypass _quiet_print: compile_instructions() flips _SILENT_MODE=True
        # so the regular print() is swallowed during compile.
        while not _compile_done.wait(10.0):
            _original_print(f"  ... still compiling ({time.perf_counter() - timer:.0f}s elapsed)", flush=True)
    _hb_thread = threading.Thread(target=_compile_heartbeat, daemon=True)
    _hb_thread.start()
    try:
        inst_meta = ue.compile_instructions()
    finally:
        _compile_done.set()
        _hb_thread.join()
    print(f"Instruction bin ready in {time.perf_counter() - timer:.2f}s")

    # Load the unified bin into program DRAM at a single base address.
    inst_bin_path = os.path.join(script_dir, cfg["paths"]["instruction_bin"])
    inst_base_addr, _ = ue.load_instructions(inst_bin_path)

    prefill_buckets       = inst_meta["prefill_buckets"]
    prefill_program_sizes = inst_meta["prefill_program_sizes"]
    prefill_total_flops   = inst_meta["prefill_total_flops"]
    decoder_buckets       = inst_meta["decoder_buckets"]
    decoder_program_sizes = inst_meta["decoder_program_sizes"]
    decoder_total_flops   = inst_meta["decoder_total_flops"]

    prefill_base_addr = inst_base_addr
    decoder_base_addr = inst_base_addr + sum(prefill_program_sizes)

    # Pick smallest prefill bucket that fits actual_seq_len = len(prompt) - 1.
    actual_seq_len = len(prefill_seq) - 1
    try:
        bucket_idx = next(i for i, b in enumerate(prefill_buckets) if b - 1 >= actual_seq_len)
    except StopIteration:
        raise RuntimeError(
            f"Prompt too long: actual_seq_len={actual_seq_len} > largest prefill bucket "
            f"({prefill_buckets[-1]}-1={prefill_buckets[-1]-1}). Add a bigger bucket to config."
        )
    bucket_seq_len = prefill_buckets[bucket_idx] - 1
    prefill_program_addr = prefill_base_addr + sum(prefill_program_sizes[:bucket_idx])
    gflops_prefill = prefill_total_flops[bucket_idx]

    print(f"\n--- Starting prefill (actual {actual_seq_len} tokens, bucket {prefill_buckets[bucket_idx]} = seq_len {bucket_seq_len}) ---")
    print(f"Prompt tokens ({len(prefill_seq)}): {prefill_seq}")
    print(f"Prompt text: {ue.tokenizer.decode(prefill_seq, skip_special_tokens=False)!r}")
    ue.seq_len = actual_seq_len  # required by run_prefill's assert
    timer = time.perf_counter()
    ue.run_prefill(prefill_program_addr, prefill_seq=prefill_seq, gflops=gflops_prefill,
                   bucket_seq_len=bucket_seq_len)
    latency_prefill = time.perf_counter() - timer
    print(f"Prefill execute done in {latency_prefill:.2f} seconds, start decoding...\n")

    print(f"\n--- Starting decoder ---")
    timer = time.perf_counter()
    decode_stats = ue.run_decoder(decoder_program_sizes, decoder_base_addr, token_id=prefill_seq[-1], gflops_per_token=decoder_total_flops)
    latency_decoder = time.perf_counter() - timer
    if isinstance(decode_stats, dict):
        decoded_count = decode_stats.get("decoded_count", 0)
        final_seq_len = decode_stats.get("seq_len", 0)
        tps_wall = decoded_count / latency_decoder if latency_decoder > 0 else 0.0
        print(f"\nDecoder done in {latency_prefill + latency_decoder:.2f} seconds "
              f"(prefill {latency_prefill:.2f}s + decode {latency_decoder:.2f}s), "
              f"decoded {decoded_count} tokens, final seq_len {final_seq_len}.")
        print(f"Decode throughput (wall-clock): {tps_wall:.3f} tokens/s")
    else:
        print(f"\nDecoder done in {latency_prefill + latency_decoder:.2f} seconds, total {decode_stats} tokens.")
    print("Llama-3.2-3B test ends.")

if __name__ == "__main__":
    main()
