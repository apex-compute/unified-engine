#!/usr/bin/env python3
"""
Gemma3 inference on accelerator: prefill + decode.

  - Config from gemma3_config.json; weights from a single bin (see below).
  - Prefill: compiled each run. Decoder: if gemma3_bin/decoder_instruction.bin and
    gemma3_bin/decoder_instruction_meta.json exist, skip decoder compile and load
    program sizes from meta; otherwise compile and write the bin + meta.
  - Run prefill then decode loop. For numeric verification use gemma3_numeric.py.

Weights:
  - Default: gemma3_bin/weights_gemma3_hf.bin (generated from HF model if missing).
  - --local-weights: use gemma3_bin/full_model_weights.bin instead.

Usage:
  python gemma3_test.py
  python gemma3_test.py --prompt "your prompt"
  python gemma3_test.py --dev xdma0 [--cycle 5.62]
  python gemma3_test.py --local-weights
  python gemma3_test.py --dual-engine

Fixed layout: gemma3_test.py, gemma3_numeric.py, *.json, and gemma3_bin/ live in the same folder.
  user_dma_core.py is two folders up (repo root); that directory is added to sys.path.
"""

import json
import math
import os
import sys

# This file's folder: gemma3_bin/, *.json, decoder_program.json live here. user_dma_core is two levels up (repo root).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import time
# pcie_utils imports (run from andromeda/pcie_utils or with PYTHONPATH)
import user_dma_core
from user_dma_core import DMA_DEVICE_H2C, DRAM_INSTRUCTION_ADDR, TYPE, UE_FMAX_CONTEXT_SIZE, UE_VECTOR_SIZE, UE_ARGMAX_INDEX, URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS, set_dma_device
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

def weight_bin_generate(output_path: str | None = None, config_path: str | None = None) -> str:
    """Generate full_model_weights.bin from Hugging Face model per gemma3_config.json layout.
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

# -----------------------------------------------------------------------------
# Gemma3 unified engine
# -----------------------------------------------------------------------------
class Gemma3_UnifiedEngine(UnifiedEngine):
    """UnifiedEngine with Gemma3 dims: loads config + weight bin, compile_prefill/compile_decoder, run_prefill/run_decoder. Numeric checks in gemma3_numeric.py."""

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
            # Dummy 510 prefill tokens, all value 107
            # self.prefill_seq = (2,) * 60

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

    def decoder_attention_core(self, head_dim: int, seq_len: int, Q_DRAM_ADDR: int, K_DRAM_ADDR: int, V_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, SCRATCH_DRAM_ADDR: int, IDENTITY_DRAM_ADDR: int =None, BIAS_DRAM_ADDR: int = None,
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
        self.LAYER0_K_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_Q_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_FLASH_OUTPUT_DRAM = self.allocate_tensor_dram(aligned_seq_len * self.head_dim * self.bytes_per_element)
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

    def program_execute(self, program_start_addr: int = user_dma_core.DRAM_INSTRUCTION_ADDR, timeout: float = 50.0, flops: float = None) -> None:
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
                flop_rate_program = self.report_flop_rate_gflops(flops)
                print(f"Report FLOPS for program execution: {flop_rate_program:.2f} GFLOPS")
        return latency, flop_rate_program

    def compile_prefill(self, seq_len: int, layer_size: int = 26) -> dict:
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

        # --- Gemma3 26 layers: compile---
        global _SILENT_MODE
        _SILENT_MODE = True
        self.clear_capture_buffer()
        self.start_capture()
        self.generate_instruction_flag_clear()
        total_flops = 0
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        for layer_idx in range(layer_size):
            layer_off = layer_idx * LAYER_WEIGHT_SIZE
            if layer_idx != 0 and not self.engine_slave:
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, sram_address=0x10000, element_size=seq_len * self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_INPUT_DRAM, element_size=seq_len * self.vector_length)
            if not self.engine_slave:
                total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                                    OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off)
            if not self.engine_slave:
                # Master: set -> clear -> workload -> check | set -> clear -> workload -> check |
                self.generate_instruction_flag_set()
                self.generate_instruction_flag_clear()
                total_flops += self.matmat_mul_core(M=seq_len_engine0, K=self.vector_length, N=self.head_dim * self.group_size,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.INT4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off,
                    )
                total_flops += self.matmat_mul_core(M=seq_len_engine0, K=self.vector_length, N=self.head_dim,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.INT4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off,
                    )
                total_flops += self.matmat_mul_core(M=seq_len_engine0, K=self.vector_length, N=self.head_dim,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_V_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size,
                    is_B_quantized=True,
                    data_type=TYPE.INT4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off,
                    )
                if self.dual_engine:
                    self.generate_instruction_flag_check(target_engine_idx=1)
            else:
                # Slave(run first): check -> clear -> workload -> set | check -> clear -> workload -> set | 
                self.generate_instruction_flag_check(target_engine_idx=0)
                self.generate_instruction_flag_clear()
                total_flops += self.matmat_mul_core(M=seq_len_engine1, K=self.vector_length, N=self.head_dim * self.group_size,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM + seq_len_engine0 * self.vector_length * self.bytes_per_element,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM + seq_len_engine0 * self.head_dim * self.group_size * self.bytes_per_element,
                    is_B_quantized=True,
                    data_type=TYPE.INT4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off,
                    )
                total_flops += self.matmat_mul_core(M=seq_len_engine1, K=self.vector_length, N=self.head_dim,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM + seq_len_engine0 * self.vector_length * self.bytes_per_element,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM + seq_len_engine0 * self.head_dim* self.bytes_per_element,
                    is_B_quantized=True,
                    data_type=TYPE.INT4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off,
                    )
                total_flops += self.matmat_mul_core(M=seq_len_engine1, K=self.vector_length, N=self.head_dim,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM + seq_len_engine0 * self.vector_length * self.bytes_per_element,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_V_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size + seq_len_engine0 * self.head_dim* self.bytes_per_element,
                    is_B_quantized=True,
                    data_type=TYPE.INT4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off,
                    )
                self.generate_instruction_flag_set()  
            if not self.engine_slave:
                total_flops += self.rms_norm_core_dram(M=seq_len, N=self.head_dim, A_DRAM_ADDR=self.LAYER0_K_DRAM,
                                OUTPUT_DRAM_ADDR=self.LAYER0_K_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_NORM_GAMMA + layer_off)
                total_flops += self.rms_norm_core_dram(M=seq_len * self.group_size, N=self.head_dim, A_DRAM_ADDR=self.LAYER0_Q_DRAM,
                                OUTPUT_DRAM_ADDR=self.LAYER0_Q_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_NORM_GAMMA + layer_off)

                # ROPE weights are shared between layers
                ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE_GLOBAL if layer_idx in self._rope_global_layers else self.DRAM_ADDR_ROPE_LOCAL
                for t in range(seq_len):
                    total_flops += self.rope_hf_core(N=self.head_dim, input_dram_addr=self.LAYER0_K_NORM_DRAM + t * self.head_dim * self.bytes_per_element, output_dram_addr=(self.LAYER0_K_ROPE_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size) + t * self.head_dim * self.bytes_per_element,
                                cos_dram_addr=ROPE_WEIGHT_ADDR + t * self.head_dim * 2 * self.bytes_per_element, sin_dram_addr=ROPE_WEIGHT_ADDR + t * self.head_dim * 2 * self.bytes_per_element + self.head_dim * self.bytes_per_element)
                    for g in range(self.group_size):
                        total_flops += self.rope_hf_core(N=self.head_dim, input_dram_addr=self.LAYER0_Q_NORM_DRAM + (t * self.group_size + g) * self.head_dim * self.bytes_per_element, output_dram_addr=self.LAYER0_FLASH_Q_DRAM + (t * self.group_size + g) * self.head_dim * self.bytes_per_element,
                                    cos_dram_addr=ROPE_WEIGHT_ADDR + t * self.head_dim * 2 * self.bytes_per_element, sin_dram_addr=ROPE_WEIGHT_ADDR + t * self.head_dim * 2 * self.bytes_per_element + self.head_dim * self.bytes_per_element)
                
                # Duplicate k rope and v projection to GQA flash attention:
                self.accelerator_memory_to_sram(self.LAYER0_K_ROPE_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size, 0x10000, seq_len * self.head_dim)
                for i in range(seq_len):
                    for g in range(self.group_size):
                        self.sram_to_accelerator_memory(0x10000 + i * self.head_dim * self.bytes_per_element, self.LAYER0_FLASH_K_DRAM + (g + i * self.group_size) * self.head_dim * self.bytes_per_element, self.head_dim)

                self.accelerator_memory_to_sram(self.LAYER0_V_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size, 0x20000, seq_len * self.head_dim)
                for i in range(seq_len):
                    for g in range(self.group_size):
                        self.sram_to_accelerator_memory(0x20000 + i * self.head_dim * self.bytes_per_element, self.LAYER0_FLASH_V_DRAM + (g + i * self.group_size) * self.head_dim * self.bytes_per_element, self.head_dim)

                total_flops += self.flash_attention_core(
                    head_dim=self.head_dim,
                    seq_len=aligned_seq_len,
                    Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
                    K_DRAM_ADDR=self.LAYER0_FLASH_K_DRAM,
                    V_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                    OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                    SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                    BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_DRAM,
                )
            if not self.engine_slave:
                # Master: set -> clear -> workload -> check | set -> clear -> workload -> check |
                self.generate_instruction_flag_set()
                self.generate_instruction_flag_clear()
                total_flops += self.matmat_mul_core(M=seq_len_engine0, K=self.head_dim * self.group_size, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.INT4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off,
                    )
                if self.dual_engine:
                    self.generate_instruction_flag_check(target_engine_idx=1)
            else:
                # Slave(run first): check -> clear -> workload -> set | check -> clear -> workload -> set | 
                self.generate_instruction_flag_check(target_engine_idx=0)
                self.generate_instruction_flag_clear()
                total_flops += self.matmat_mul_core(M=seq_len_engine1, K=self.head_dim * self.group_size, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM + seq_len_engine0 * self.head_dim * self.group_size * self.bytes_per_element,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM + seq_len_engine0 * self.vector_length * self.bytes_per_element,
                    is_B_quantized=True,
                    data_type=TYPE.INT4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off,
                    )
                self.generate_instruction_flag_set()
            if not self.engine_slave:
                total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                                OUTPUT_DRAM_ADDR=self.LAYER0_POST_ATTN_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_NORM_GAMMA + layer_off)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_INPUT_DRAM, sram_address=0x10000, element_size=seq_len * self.vector_length)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_NORM_DRAM, sram_address=0x90000, element_size=seq_len * self.vector_length)
                # ToDo: no FLOPS return for eltwise_add_core and eltwise_mul_core
                self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=seq_len * self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, element_size=seq_len * self.vector_length)
                total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                                OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off)
            if not self.engine_slave:
                # Master: set -> clear -> workload -> check | set -> clear -> workload -> check |
                self.generate_instruction_flag_set()
                self.generate_instruction_flag_clear()
                total_flops += self.matmat_mul_core(M=seq_len_engine0, K=self.vector_length, N=self.mlp_elements,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.INT4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off,
                    gelu_enable=True,
                    )
                total_flops += self.matmat_mul_core(M=seq_len_engine0, K=self.vector_length, N=self.mlp_elements,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.INT4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off,
                    )
                if self.dual_engine:
                    self.generate_instruction_flag_check(target_engine_idx=1)
            else:
                # Slave: check -> clear -> workload -> set | check -> clear -> workload -> set |
                self.generate_instruction_flag_check(target_engine_idx=0)
                self.generate_instruction_flag_clear()
                total_flops += self.matmat_mul_core(M=seq_len_engine1, K=self.vector_length, N=self.mlp_elements,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM + seq_len_engine0 * self.vector_length * self.bytes_per_element,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM + seq_len_engine0 * self.mlp_elements * self.bytes_per_element,
                    is_B_quantized=True,
                    data_type=TYPE.INT4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off,
                    gelu_enable=True,
                    )
                total_flops += self.matmat_mul_core(M=seq_len_engine1, K=self.vector_length, N=self.mlp_elements,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM + seq_len_engine0 * self.vector_length * self.bytes_per_element,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM + seq_len_engine0 * self.mlp_elements * self.bytes_per_element,
                    is_B_quantized=True,
                    data_type=TYPE.INT4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off,
                    )
                self.generate_instruction_flag_set()
            if not self.engine_slave:
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_GATE_DRAM, sram_address=0x10000, element_size=seq_len * self.mlp_elements)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_UP_DRAM, sram_address=0x90000, element_size=seq_len * self.mlp_elements)
                self.eltwise_mul_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=seq_len * self.mlp_elements)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_MLP_MULT_DRAM, element_size=seq_len * self.mlp_elements)
            if not self.engine_slave:    
                # Master: set -> clear -> workload -> check | set -> clear -> workload -> check |
                self.generate_instruction_flag_set()
                self.generate_instruction_flag_clear()
                total_flops += self.matmat_mul_core(M=seq_len_engine0, K=self.mlp_elements, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.INT4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off,
                    )
                if self.dual_engine:
                    self.generate_instruction_flag_check(target_engine_idx=1)
            else:
                # Slave: check -> clear -> workload -> set | check -> clear -> workload -> set |
                self.generate_instruction_flag_check(target_engine_idx=0)
                self.generate_instruction_flag_clear()
                total_flops += self.matmat_mul_core(M=seq_len_engine1, K=self.mlp_elements, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM + seq_len_engine0 * self.mlp_elements * self.bytes_per_element,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM + seq_len_engine0 * self.vector_length * self.bytes_per_element,
                    is_B_quantized=True,
                    data_type=TYPE.INT4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off,
                    )
                self.generate_instruction_flag_set()
            if not self.engine_slave:
                total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                                OUTPUT_DRAM_ADDR=self.LAYER0_POST_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_FFW_NORM_GAMMA + layer_off)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, sram_address=0x10000, element_size=seq_len * self.vector_length)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_MLP_NORM_DRAM, sram_address=0x90000, element_size=seq_len * self.vector_length)
                self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=seq_len * self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, element_size=seq_len * self.vector_length)
        self.stop_capture()
        self.generate_instruction_halt()
        prefill_program_addr = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(prefill_program_addr)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        _SILENT_MODE = False
        print(f"    Prefill program start at 0x{prefill_program_addr:X} end at 0x{self.get_program_dram_addr():X}, usage: {self.get_program_dram_usage()} bytes")

        return prefill_program_addr, total_flops
        
    def run_prefill(self, prefill_program_addr: int, prefill_seq=None, flops: int = None) -> dict:
        """
        Run prefill. Uses prefill_seq if provided, otherwise self.prefill_seq (set by set_prefill_seq()).
        
        Args:
            prefill_program_addr: The address of the prefill program in DRAM.
            prefill_seq: Optional sequence; if None, uses self.prefill_seq.
            gflops: The number of GFLOPS to use for the prefill.
        
        Returns:
            A dictionary containing the results of the prefill.
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
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64

        embedding_tensor = self.get_embedding_for_tokens(prefill_seq)
        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
        bias_one_group = torch.full((aligned_seq_len, aligned_seq_len), float("-inf"), dtype=torch.bfloat16)
        valid_mask = torch.tril(torch.ones(aligned_seq_len, aligned_seq_len, dtype=torch.bool), diagonal=0) if not self.causal_mask_upper else torch.triu(torch.ones(aligned_seq_len, aligned_seq_len, dtype=torch.bool), diagonal=0)
        bias_one_group.masked_fill_(valid_mask, 0.0)
        bias_one_group[:, q_seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_one_group)
        latency, flop_rate_program = self.program_execute(prefill_program_addr, flops=flops)
        return latency, flop_rate_program

    def compile_decoder(self, layer_size: int = 26) -> tuple[str, list[int], list[int]]:
        """Compile decoder programs for seq_len buckets, or load from existing bin/meta.
        Returns (decoder_bin_path, program_sizes[8], total_flops_list[8])."""
        decoder_bin_path = os.path.join(self.script_dir, "gemma3_bin", "decoder_program.bin")
        decoder_meta_path = os.path.join(self.script_dir, "decoder_program.json")
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
                if layer_idx != 0:
                    self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, sram_address=0x10000, element_size=self.vector_length)
                    self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_INPUT_DRAM, element_size=self.vector_length)
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off)
                # total_flops += self.quantized_matmat_core(M=1, K=self.vector_length, N=self.head_dim * self.group_size,
                #     A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                #     SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off, data_type=TYPE.INT4)
                total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=self.head_dim * self.group_size,
                                                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                                                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off,
                                                    OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                                                    is_B_quantized=True,
                                                    data_type=TYPE.INT4,
                                                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off,
                                                    )
                total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=self.head_dim,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.INT4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off,
                    )
                total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=self.head_dim,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.INT4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off,
                    )
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_FLASH_V_DRAM, sram_address=0x10000, element_size=self.head_dim)
                self.generate_instruction_add_imm(self.V_CACHE_SIZE_REG, self.LAYER0_V_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size, self.TMP_REG)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=0, element_size=self.head_dim)
                self.overwrite_instruction_with_general_register(self.TMP_REG)
                total_flops += self.rms_norm_core_dram(M=1, N=self.head_dim, A_DRAM_ADDR=self.LAYER0_K_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_K_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_NORM_GAMMA + layer_off)
                total_flops += self.rms_norm_core_dram(M=self.group_size, N=self.head_dim, A_DRAM_ADDR=self.LAYER0_Q_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_Q_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_NORM_GAMMA + layer_off)

                ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE_GLOBAL if layer_idx in self._rope_global_layers else self.DRAM_ADDR_ROPE_LOCAL
                total_flops += self.rope_hf_core(N=self.head_dim, input_dram_addr=self.LAYER0_K_NORM_DRAM, output_dram_addr=self.LAYER0_K_ROPE_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size,
                        cos_dram_addr=ROPE_WEIGHT_ADDR, sin_dram_addr=ROPE_WEIGHT_ADDR + self.head_dim * self.bytes_per_element,
                        rope_size_reg=self.ROPE_SIZE_REG, output_addr_inc_reg=self.V_CACHE_SIZE_REG, tmp_reg=self.TMP_REG)
                for g in range(self.group_size):
                    total_flops += self.rope_hf_core(N=self.head_dim, input_dram_addr=self.LAYER0_Q_NORM_DRAM + g * self.head_dim * self.bytes_per_element, output_dram_addr=self.LAYER0_FLASH_Q_DRAM + g * self.head_dim * self.bytes_per_element,
                            cos_dram_addr=ROPE_WEIGHT_ADDR, sin_dram_addr=ROPE_WEIGHT_ADDR + self.head_dim * self.bytes_per_element,
                            rope_size_reg=self.ROPE_SIZE_REG, tmp_reg=self.TMP_REG)
                for g in range(self.group_size):
                    total_flops += self.decoder_attention_core(
                        head_dim=self.head_dim,
                        seq_len=seq_len,
                        Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM + g * self.head_dim * self.bytes_per_element,
                        K_DRAM_ADDR=self.LAYER0_K_ROPE_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size,
                        V_DRAM_ADDR=self.LAYER0_V_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size,
                        OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM + g * self.head_dim * self.bytes_per_element,
                        IDENTITY_DRAM_ADDR=self.IDENTITY_DRAM_ADDR,
                        SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                        BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_DRAM,
                    )
                total_flops += self.matmat_mul_core(M=1, K=self.head_dim * self.group_size, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.INT4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off,
                    )
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_POST_ATTN_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_NORM_GAMMA + layer_off)

                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_INPUT_DRAM, sram_address=0x10000, element_size=self.vector_length)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_NORM_DRAM, sram_address=0x90000, element_size=self.vector_length)
                self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, element_size=self.vector_length)

                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off)

                total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=self.mlp_elements,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.INT4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off,
                    gelu_enable=True,
                    )
                total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=self.mlp_elements,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.INT4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off,
                    )

                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_GATE_DRAM, sram_address=0x10000, element_size=self.mlp_elements)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_UP_DRAM, sram_address=0x90000, element_size=self.mlp_elements)
                self.eltwise_mul_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.mlp_elements)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_MLP_MULT_DRAM, element_size=self.mlp_elements)

                total_flops += self.matmat_mul_core(M=1, K=self.mlp_elements, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.INT4,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off,
                    )
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_POST_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_FFW_NORM_GAMMA + layer_off)

                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, sram_address=0x10000, element_size=self.vector_length)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_MLP_NORM_DRAM, sram_address=0x90000, element_size=self.vector_length)
                self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, element_size=self.vector_length)

            if layer_size == self.LAYER_SIZE:
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_OUTPUT_DRAM,
                    OUTPUT_DRAM_ADDR=self.OUTPUT_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_OUTPUT_NORM_GAMMA)
                total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=self.EMBEDDING_ELEMENTS,
                    A_DRAM_ADDR=self.OUTPUT_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_QUANT,
                    OUTPUT_DRAM_ADDR=self.LOGITS_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.INT4,
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
        with open(decoder_bin_path, "wb") as f:
            f.write(all_programs_bytes)
        program_sizes = [c * 32 for c in segment_instruction_counts]
        with open(decoder_meta_path, "w") as f:
            json.dump({"instruction_counts": segment_instruction_counts, "program_sizes": program_sizes, "total_flops": total_flops_list}, f, indent=0)
        self.clear_capture_buffer()
        print(f"Decoder programs: {len(segment_instruction_counts)} segments written to {decoder_bin_path} ({len(all_programs_bytes)} bytes)")
        return decoder_bin_path, program_sizes, total_flops_list

    def run_decoder(self, decoder_program_sizes: list[int], decoder_base_addr: int, token_id: int, flops_per_token: list[int] | None = None) -> dict:
        """Run decode loop. seq_len capped at 512. Breaks on END_OF_TURN_TOKEN_ID."""
        if token_id is None:
            print("No last token available for decode.")
            return {}

        global _SILENT_MODE
        max_seq_len = self.MAX_CONTEXT_SIZE
        total_latency, total_flop_rate = 0, 0
        while self.seq_len < max_seq_len:
            _SILENT_MODE = True
            timer_start = time.perf_counter()
            self.seq_len += 1
            aligned_seq_len = ((self.seq_len + 63) // 64) * 64
            prog_idx = min((self.seq_len - 1) // 64, 7)
            prog_addr = decoder_base_addr + sum(decoder_program_sizes[:prog_idx])
            flops_per_token_idx = flops_per_token[prog_idx] if flops_per_token else None

            self.isa_add_set_core(self.V_CACHE_SIZE_REG, (self.seq_len - 1) * self.k_size)
            self.isa_add_set_core(self.ROPE_SIZE_REG, (self.seq_len - 1) * self.k_size * 2)
            embedding_tensor = self.get_embedding_for_tokens([token_id])
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
            bias_host = torch.full((1, aligned_seq_len), -1e36, dtype=torch.bfloat16)
            bias_host[0, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_host)

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
    parser = argparse.ArgumentParser(description="Gemma3 layer-0 prefill: run on accelerator, verify with torch ref.")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt: tokenizer encodes this to prefill_seq (overrides default)")
    parser.add_argument("--local-weights", action="store_true", help="Use gemma3_bin/full_model_weights.bin instead of generated weights_gemma3_hf.bin")
    parser.add_argument("--dual-engine", action="store_true", help="Use dual engine")
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

    dual_engine = args.dual_engine
    ue = Gemma3_UnifiedEngine(local_weights=args.local_weights, dual_engine=dual_engine)
    ue.set_prefill_seq(args.prompt)

    if dual_engine:
        ue2 = Gemma3_UnifiedEngine(local_weights=args.local_weights, dual_engine=True, engine_slave=True)
        ue2.set_prefill_seq(args.prompt)

    print(f"\n--- Compiling ---")
    timer = time.perf_counter()
    prefill_program_addr, flops_prefill = ue.compile_prefill(seq_len=len(ue.prefill_seq))
    if dual_engine:
        prefill_program_addr2, flops_prefill2 = ue2.compile_prefill(seq_len=len(ue2.prefill_seq))
    print(f"Prefill compile done in {time.perf_counter() - timer:.2f} seconds")
    timer_dec = time.perf_counter()
    decoder_bin_path, decoder_program_sizes, flops_per_token = ue.compile_decoder()
    print(f"Decoder compile done in {time.perf_counter() - timer_dec:.2f} seconds.")
    decoder_base_addr, _ = ue.load_instructions(decoder_bin_path)

    print(f"\n--- Starting prefill ---")
    print(f"Prompt ({len(ue.prefill_seq)}) tokens: {ue.prefill_seq}")
    timer=time.perf_counter()
    if dual_engine:
        ue2.program_execute(prefill_program_addr2, timeout=0)
    latency_hw_prefill, flop_rate_hw_prefill = ue.run_prefill(prefill_program_addr, flops=flops_prefill)
    latency_prefill = time.perf_counter() - timer
    if dual_engine:
        print(f"Dual engine prefill gflops: {((flops_prefill + flops_prefill2) / (latency_hw_prefill * 1e3)):.2f} GFLOPS")
    print(f"Prefill execute done in {latency_prefill:.2f} seconds, start decoding...\n")

    print(f"\n--- Starting decoder ---")
    timer=time.perf_counter()
    token_cnt_decoded, latency_hw_decoder, flop_rate_hw_decoder = ue.run_decoder(decoder_program_sizes, decoder_base_addr, token_id=ue.prefill_seq[-1], flops_per_token=flops_per_token)
    latency_decoder = time.perf_counter() - timer
    print(f"\nDecoder done in {latency_prefill + latency_decoder:.2f} seconds, speed: {(token_cnt_decoded - len(ue.prefill_seq) + 1) / latency_decoder:.2f} tokens/s, total {token_cnt_decoded} tokens.")
    print(f"HW counter: Latency: {(latency_hw_prefill + latency_hw_decoder) / 1e6:.2f} seconds, decoder average Gflops: {flop_rate_hw_decoder / (token_cnt_decoded - len(ue.prefill_seq) + 1):.2f} Gflops")
    print("Gemma3 test ends.")

if __name__ == "__main__":
    main()
