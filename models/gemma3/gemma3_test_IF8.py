#!/usr/bin/env python3
"""
Gemma3 IF8 inference on accelerator: prefill + decode.

This is the IF8 (8-bit adaptive block-scaled) variant of the gemma3 template.
It mirrors gemma3_test.py one-for-one except for two IF8-specific design
points that customers porting other models to IF8 will need to follow:

  1. Layout rewrite (`_if4_to_if8_layout`): every ``*_DATA`` region in
     gemma3_config.json doubles in size (1 byte/elem vs 4 bits/elem) and
     all offsets are repacked contiguously at load time. The on-disk JSON
     stays IF4-shaped so a single config serves both codecs.

  2. Dynamic Params/Tensor/Program DRAM bases (`Gemma3_UnifiedEngine.__init__`).
     IF8 doubles every quantized weight; layers (~687 MB) plus LM_HEAD
     (~297 MB) totals ~990 MB, which overflows the 768 MB Params default
     baked into user_dma_core.py. Total physical DRAM is 4 GB, so the
     fix is just to recompute the boundary integers from the config:
     ``_tensor_base = 0x80000000 + params_estimate``, ``_program_base =
     _tensor_base + tensor_estimate``, both 1 MB aligned. This mirrors
     Gemma4 E2B's pattern. With this in place LM_HEAD lives in Params
     DRAM with all other weights and the on-chip LM_HEAD path runs
     end-to-end -- prefill + decode are fully on chip; per-token host
     work is just the embedding-table row select for the new token id.

Quantization: ``QUANT_PRECISION = "if8"`` selects per-block MixMSE between
INT8 and FP8 E4M3 (scale-sign dispatches the codebook). Forcing pure INT8
(``int_variant=True``) is *not* recommended -- INT8 preserves activation
outliers that compound across all 26 layers and inflate the final-layer
magnitude past the on-chip rms_norm_core's safe range, producing NaN
logits. MixMSE keeps growth bounded.

Compile/run flow is identical to the IF4 reference: one prefill program per
``prefill_seq_len`` bucket plus all decoder buckets are captured into a
single combined instruction image (gemma3_bin/gemma3_instruction.bin) with
metadata sidecar.

Weights bin: gemma3_bin/weights_gemma3_hf_if8.bin (auto-generated from HF
on first run; the ``_if8`` suffix avoids collision with an IF4 bin in the
same directory). ``--local-weights`` uses gemma3_bin/full_model_weights.bin.

Usage:
  python gemma3_test_IF8.py
  python gemma3_test_IF8.py --prompt "your prompt"
  python gemma3_test_IF8.py --dev xdma0 [--cycle 5.62]
  python gemma3_test_IF8.py --local-weights
  python gemma3_test_IF8.py --dual-engine

Layout: this file, gemma3_numeric.py, gemma3_config.json, and gemma3_bin/
live in the same folder. user_dma_core.py is two folders up; that parent
directory is prepended to sys.path at import.
"""

import json
import math
import os
import sys

# This file's folder; user_dma_core.py is two folders up (repo root); that directory is added to sys.path.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import time
# pcie_utils imports (run from andromeda/pcie_utils or with PYTHONPATH)
import user_dma_core
from user_dma_core import DMA_DEVICE_H2C, DRAM_INSTRUCTION_ADDR, TYPE, UE_FMAX_CONTEXT_SIZE, UE_VECTOR_SIZE, UE_ARGMAX_INDEX, URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS, set_dma_device, ue_35bit_addr_shifter
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

# Codec lives in template/quant_schemas.py. Change QUANT_PRECISION to swap
# codebooks (e.g. "if4"); for IF8 the default per-block MixMSE selection
# between INT8 and FP8 is what we want -- do NOT force pure INT8 here, see
# module docstring point 3.
from quant_schemas import quantize

QUANT_PRECISION = "if8"


def _if4_to_if8_layout(cfg: dict) -> dict:
    """Rewrite an IF4 ``gemma3_config.json`` layout in place for IF8.

    IF8 stores 1 byte/element vs IF4's 4 bits/element, so every quantized
    ``*_DATA`` region doubles in size. Offsets are recomputed contiguously
    (preserving the original blk0 base, e.g. 0x24000000 for gemma3); the
    per-layer ``file_info.layer_size`` is updated; non-layer regions are
    repositioned after the (now-larger) layer block; and the weights bin
    filename gets an ``_if8`` suffix so an existing IF4 bin in the same
    directory is not picked up by mistake.
    """
    def _is_data(key: str) -> bool:
        return key.endswith("_DATA")

    regions = cfg.get("regions", {})
    nlr = cfg.get("non_layer_regions", {})

    # Walk blk0 regions in original-offset order, double *_DATA sizes,
    # repack contiguously starting from the original base.
    if regions:
        sorted_keys = sorted(regions.keys(),
                             key=lambda k: _parse_offset(regions[k]["offset"]))
        base = _parse_offset(regions[sorted_keys[0]]["offset"])
        cur = base
        for k in sorted_keys:
            r = regions[k]
            if _is_data(k):
                r["size"] *= 2
            r["offset"] = f"0x{cur:08X}"
            cur += r["size"]
        cfg["file_info"]["layer_size"] = cur - base

    # Non-layer regions sit immediately after the last layer.
    if nlr:
        num_layers = cfg["file_info"]["num_layers"]
        layer_block_end = base + num_layers * cfg["file_info"]["layer_size"]
        sorted_nlr = sorted(nlr.keys(),
                            key=lambda k: _parse_offset(nlr[k]["offset"]))
        cur = layer_block_end
        for k in sorted_nlr:
            r = nlr[k]
            if _is_data(k):
                r["size"] *= 2
            r["offset"] = f"0x{cur:08X}"
            cur += r["size"]

    # Distinct filenames so stale IF4 artifacts don't get reused. Both the
    # weights bin AND the captured instruction image (which embeds literal
    # IF8 DRAM addresses) must be separate from the IF4 reference's caches.
    paths = cfg.setdefault("paths", {})
    for key, default in (
        ("weights_bin",     "gemma3_bin/weights_gemma3_hf.bin"),
        ("instruction_bin", "gemma3_bin/gemma3_instruction.bin"),
        ("instruction_meta","gemma3_bin/gemma3_instruction.json"),
    ):
        rel = paths.get(key, default)
        if "_if8" not in rel:
            stem, sep, ext = rel.rpartition(".")
            paths[key] = f"{stem}_if8{sep}{ext}" if sep else f"{rel}_if8"
    return cfg

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

    # Layers (IF8 MixMSE quantization runs on CPU; ~few seconds per layer)
    print(f"Quantizing {num_layers} transformer layers to {QUANT_PRECISION.upper()} ...")
    layer_t0 = time.perf_counter()
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

        # ``kind`` is "bf16" for raw bf16 bytes, "quant" for codec-driven
        # (scale, data) pairs produced by ``quantize(QUANT_PRECISION, ...)``.
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
        if (layer_idx + 1) % 4 == 0 or layer_idx == num_layers - 1:
            print(f"  layer {layer_idx + 1}/{num_layers} done ({time.perf_counter() - layer_t0:.1f}s elapsed)")

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

    # LM_HEAD (262144 x 1152 -> ~302 M elements; the slowest single step)
    print(f"Quantizing LM_HEAD ({tuple(model.lm_head.weight.shape)}) to {QUANT_PRECISION.upper()} ...")
    lm_t0 = time.perf_counter()
    lm_head_w = model.lm_head.weight.detach().cpu().to(torch.bfloat16)
    scale_sz = weight_defs["LM_HEAD_WEIGHT_SCALE_SIZE"]
    data_sz = weight_defs["LM_HEAD_WEIGHT_DATA_SIZE"]
    data_bytes, scale_bytes = quantize(QUANT_PRECISION, lm_head_w)
    print(f"  LM_HEAD quantized in {time.perf_counter() - lm_t0:.1f}s")
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
        engine_base = user_dma_core.UE_0_BASE_ADDR + 0x00010000 if engine_slave else user_dma_core.UE_0_BASE_ADDR
        # Compute Params/Tensor/Program DRAM bases dynamically so the IF8
        # weight set (~990 MB total: 26 layers + LM_HEAD + ROPE + OUTPUT_NORM)
        # fits in a single contiguous Params region. The 768 MB / 512 MB /
        # 256 MB defaults in user_dma_core.py are sized for small IF4 models;
        # IF8 doubles every quantized tensor and pushes Params past those
        # defaults. Total physical DRAM is 4 GB, so the only thing standing
        # in the way is the boundary integers -- mirror Gemma4 E2B's pattern
        # and recompute them from the config sizes.
        _script_dir = script_dir or SCRIPT_DIR
        _cfg_tmp = self.load_config(script_dir=_script_dir)
        _fi = _cfg_tmp["file_info"]
        _layers_sz = _fi["num_layers"] * _fi["layer_size"]
        _non_layer_sz = sum(r["size"] for r in _cfg_tmp["non_layer_regions"].values())
        _params_estimate = _layers_sz + _non_layer_sz + 0x100000  # +1 MB margin (alignment, identity, etc.)
        _tensor_base = ((0x80000000 + _params_estimate + 0xFFFFF) // 0x100000) * 0x100000  # 1 MB aligned
        _tensor_estimate = 0x12000000  # 288 MB; KV cache + activations + flash buffers fit in ~64 MB on Gemma3-1B with margin
        _program_base = ((_tensor_base + _tensor_estimate + 0xFFFFF) // 0x100000) * 0x100000  # 1 MB aligned
        if engine_slave:
            _program_base += 0x10000000
        super().__init__(BASE_ADDR=engine_base, program_dram_base=_program_base, tensor_dram_base=_tensor_base)
        self.dual_engine = dual_engine
        self.script_dir = _script_dir
        self._cfg = _cfg_tmp
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
        fixed = self._cfg.get("fixed_isa_regs", {})
        self.V_CACHE_SIZE_REG = fixed["V_CACHE_SIZE_REG"]
        self.TMP_REG = fixed["TMP_REG"]
        self.ROPE_SIZE_REG = fixed["ROPE_SIZE_REG"]
        self._isa_reg_counter = 4 # Register 1, 2, 3 are used specifically for V_CACHE_SIZE_REG, TMP_REG, and ROPE_SIZE_REG in Gemma3
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
        """Load gemma3_config.json, apply the IF4->IF8 layout rewrite, and
        build a flat ``weight_defs`` dict of {key: offset, key_SIZE: size}."""
        if config_path is None:
            script_dir = script_dir or SCRIPT_DIR
            config_path = os.path.join(script_dir, "gemma3_config.json")
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

    def set_prefill_seq(self, prompt: str | None = None) -> None:
        """Set self.prefill_seq.

        Prefill is compiled as one program per bucket prefill_seq_len in
        [1, prefill_max_seq_len]. The runtime prompt is processed as
        prompt[:-1] (the last token feeds the decoder), so the tokenized prompt
        length must lie in [2, prefill_max_seq_len + 1]. Out-of-range prompts
        fall back to the default prompt with a warning.
        """
        max_prefill = int(self._cfg["model"]["prefill_max_seq_len"])
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

    def decoder_group_attention_core_legacy(
        self,
        group_size: int,
        head_dim: int,
        seq_len: int,
        Q_DRAM_ADDR: int,
        K_DRAM_ADDR: int,
        V_DRAM_ADDR: int,
        OUTPUT_DRAM_ADDR: int,
        SCRATCH_DRAM_ADDR: int,
        IDENTITY_DRAM_ADDR: int = None,
        BIAS_DRAM_ADDR: int = None,
        debug_mode: bool = False,
        SM_OUTPUT_DRAM_ADDR: int = None,
    ) -> int:
        total_flops = 0
        group_stride_bytes = head_dim * self.bytes_per_element
        bytes_per_element = 2
        bias_enable = True if BIAS_DRAM_ADDR is not None else False

        if debug_mode: # DEBUG only, needs to be allocated in DRAM
            assert SM_OUTPUT_DRAM_ADDR is not None, "SM_OUTPUT_DRAM_ADDR is not set for debug mode"

        identity_tensor = torch.eye(head_dim, dtype=torch.bfloat16)
        for g in range(group_size):
            group_q_dram_addr = Q_DRAM_ADDR + g * group_stride_bytes
            group_output_dram_addr = OUTPUT_DRAM_ADDR + g * group_stride_bytes

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
                self.accelerator_memory_to_sram(accelerator_dram_address=group_q_dram_addr + i * K * bytes_per_element,
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
                                                                accelerator_dram_address=group_output_dram_addr + (i + m_take_chunk_idx) * head_dim * bytes_per_element
                                                                                                            + v_tr_column_idx * bytes_per_element
                                                                                                            + p_row_idx * head_dim * bytes_per_element,
                                                                element_size=v_tr_column_take)


                        if v_tr_row_chunk_size_aligned is None:
                            self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr,
                                                            accelerator_dram_address=group_output_dram_addr + (i + m_take_chunk_idx) * head_dim * bytes_per_element + v_tr_column_idx * bytes_per_element,
                                                            element_size=m_take_chunk_size * v_tr_column_take,
                                                            stride_bytes_per_chunk=v_tr_column_take * bytes_per_element,
                                                            stride_jump_bytes=head_dim * bytes_per_element)

            # Total Theoretical FLOPS
            group_flops = 1 * head_dim # q_scale 
            group_flops += 2 * 1 * head_dim * seq_len # Q @ K^T
            group_flops += 1 * seq_len * 5 # softmax
            group_flops += 2 * 1 * seq_len * head_dim # sm @ v
            print(f"Total Theoretical FLOPS: {group_flops}")
            total_flops += group_flops
        return total_flops

    def decoder_group_attention_core_pbi(
        self,
        group_size: int,
        head_dim: int,
        seq_len: int,
        Q_DRAM_ADDR: int,
        K_DRAM_ADDR: int,
        V_DRAM_ADDR: int,
        OUTPUT_DRAM_ADDR: int,
        SCRATCH_DRAM_ADDR: int,
        IDENTITY_DRAM_ADDR: int = None,
        BIAS_DRAM_ADDR: int = None,
        debug_mode: bool = False,
        SM_OUTPUT_DRAM_ADDR: int = None,
    ) -> int:
        del debug_mode, SM_OUTPUT_DRAM_ADDR, IDENTITY_DRAM_ADDR

        bytes_per_element = self.bytes_per_element
        score_dram_addr = SCRATCH_DRAM_ADDR + head_dim * seq_len * bytes_per_element
        scaled_q_dram_addr = score_dram_addr + group_size * seq_len * bytes_per_element
        total_flops = 0

        # Materialize V^T once since K/V cache is shared across all query groups.
        self.bf16_transpose_core(
            M=seq_len,
            N=head_dim,
            INPUT_DRAM_ADDR=V_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=SCRATCH_DRAM_ADDR,
        )

        self.accelerator_memory_to_sram(
            accelerator_dram_address=Q_DRAM_ADDR,
            sram_address=0x00000,
            element_size=group_size * head_dim,
        )
        self.broadcast_mul(
            scalar=1 / math.sqrt(head_dim),
            sram_start_addr=0x00000,
            sram_wb_addr=0x00000,
            element_size=group_size * head_dim,
        )
        self.sram_to_accelerator_memory(
            sram_address=0x00000,
            accelerator_dram_address=scaled_q_dram_addr,
            element_size=group_size * head_dim,
        )

        self.matmat_mul_core(
            M=group_size,
            K=head_dim,
            N=seq_len,
            A_DRAM_ADDR=scaled_q_dram_addr,
            B_DRAM_ADDR=K_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=score_dram_addr,
            softmax_enable=True,
            C_DRAM_ADDR=BIAS_DRAM_ADDR,
            use_pbi=True,
        )
        self.matmat_mul_core(
            M=group_size,
            K=seq_len,
            N=head_dim,
            A_DRAM_ADDR=score_dram_addr,
            B_DRAM_ADDR=SCRATCH_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
            use_pbi=True,
        )

        # Match legacy decoder_group_attention_core GFLOP accounting exactly.
        group_flops = 1 * head_dim
        group_flops += 2 * 1 * head_dim * seq_len
        group_flops += 1 * seq_len * 5
        group_flops += 2 * 1 * seq_len * head_dim
        total_flops += group_size * group_flops

        return total_flops

    def decoder_group_attention_core(
        self,
        group_size: int,
        head_dim: int,
        seq_len: int,
        Q_DRAM_ADDR: int,
        K_DRAM_ADDR: int,
        V_DRAM_ADDR: int,
        OUTPUT_DRAM_ADDR: int,
        SCRATCH_DRAM_ADDR: int,
        IDENTITY_DRAM_ADDR: int = None,
        BIAS_DRAM_ADDR: int = None,
        debug_mode: bool = False,
        SM_OUTPUT_DRAM_ADDR: int = None,
        use_pbi: bool = False,
    ) -> int:
        if use_pbi:
            return self.decoder_group_attention_core_pbi(
                group_size=group_size,
                head_dim=head_dim,
                seq_len=seq_len,
                Q_DRAM_ADDR=Q_DRAM_ADDR,
                K_DRAM_ADDR=K_DRAM_ADDR,
                V_DRAM_ADDR=V_DRAM_ADDR,
                OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                SCRATCH_DRAM_ADDR=SCRATCH_DRAM_ADDR,
                IDENTITY_DRAM_ADDR=IDENTITY_DRAM_ADDR,
                BIAS_DRAM_ADDR=BIAS_DRAM_ADDR,
                debug_mode=debug_mode,
                SM_OUTPUT_DRAM_ADDR=SM_OUTPUT_DRAM_ADDR,
            )
        return self.decoder_group_attention_core_legacy(
            group_size=group_size,
            head_dim=head_dim,
            seq_len=seq_len,
            Q_DRAM_ADDR=Q_DRAM_ADDR,
            K_DRAM_ADDR=K_DRAM_ADDR,
            V_DRAM_ADDR=V_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
            SCRATCH_DRAM_ADDR=SCRATCH_DRAM_ADDR,
            IDENTITY_DRAM_ADDR=IDENTITY_DRAM_ADDR,
            BIAS_DRAM_ADDR=BIAS_DRAM_ADDR,
            debug_mode=debug_mode,
            SM_OUTPUT_DRAM_ADDR=SM_OUTPUT_DRAM_ADDR,
        )

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

        # All non-layer weights (OUTPUT_NORM + LM_HEAD scale/data) live in
        # Params DRAM. The IF8 LM_HEAD (~297 MB) plus the 26 IF8 layers
        # (~687 MB) fit because __init__ pushed _tensor_base above them.
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
                flop_rate_program, _ = self.report_flop_rate_gflops(flops)
                print(f"Report FLOPS for program execution: {flop_rate_program:.2f} GFLOPS")
        return latency, flop_rate_program

    def _compile_prefill_program(self, prefill_seq_len: int, layer_size: int = 26, use_pbi: bool = False) -> dict:
        """
        Compile a single prefill program for the given prefill seq_len and return
        its metadata (start_addr, size_bytes, flops).

        Args:
            prefill_seq_len: Number of tokens this prefill program processes. This
                is prompt_len - 1 since the last prompt token feeds the decoder.
            layer_size: Number of transformer layers to compile.
            use_pbi: Use pointer-backed instruction (PBI) descriptors.

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
        engine_master = not self.engine_slave
        row_offset = 0 if engine_master else seq_len // 2
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
                data_type=TYPE.IF8,
                SCALE_DRAM_ADDR=scale_dram_addr,
                use_pbi=use_pbi,
                **kwargs,
            )
            end_parallel_stage()
            return flops

        def duplicate_gqa_rows_pbi(src_sram_addr: int, dst_dram_addr: int) -> None:
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
            self.loop_start(seq_len)
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
        for layer_idx in range(layer_size):
            layer_off = layer_idx * LAYER_WEIGHT_SIZE
            if layer_idx != 0 and engine_master:
                # change the first layer input to addr=LAYER0_OUTPUT_DRAM
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, sram_address=0x10000, element_size=seq_len * self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_INPUT_DRAM, element_size=seq_len * self.vector_length)
            if engine_master:
                total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                                    OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off,
                                    use_pbi=use_pbi)
            # TODO: dual engine is not supported in this stage yet.
            total_flops += self.matmat_mul_core(
                M=seq_len,
                K=self.vector_length,
                N=self.head_dim * self.group_size,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                is_B_quantized=True,
                data_type=TYPE.IF8,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off,
                use_pbi=use_pbi,
            )
            total_flops += self.matmat_mul_core(
                M=seq_len,
                K=self.vector_length,
                N=self.head_dim,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                is_B_quantized=True,
                data_type=TYPE.IF8,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off,
                use_pbi=use_pbi,
            )
            total_flops += self.matmat_mul_core(
                M=seq_len,
                K=self.vector_length,
                N=self.head_dim,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_V_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size,
                is_B_quantized=True,
                data_type=TYPE.IF8,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off,
                use_pbi=use_pbi,
            )
            # TODO: OUTPUT_DRAM_ADDR=temp addr. Then memcpy from temp addr to self.LAYER0_V_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size
            if engine_master:
                total_flops += self.rms_norm_core_dram(M=seq_len, N=self.head_dim, A_DRAM_ADDR=self.LAYER0_K_DRAM,
                                OUTPUT_DRAM_ADDR=self.LAYER0_K_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_NORM_GAMMA + layer_off,
                                use_pbi=use_pbi)
                total_flops += self.rms_norm_core_dram(M=seq_len * self.group_size, N=self.head_dim, A_DRAM_ADDR=self.LAYER0_Q_DRAM,
                                OUTPUT_DRAM_ADDR=self.LAYER0_Q_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_NORM_GAMMA + layer_off,
                                use_pbi=use_pbi)

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
                    use_pbi=use_pbi,
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
                    use_pbi=use_pbi,
                )
                
                # Pre-flash-attn layout:
                # Q: [seq_len, group_size, head_dim], [seq_len:max_seq_len, :] has been padded 0
                # K/V cache: [seq_len, head_dim]; duplicate each token row group_size times for GQA. [seq_len, group_size, head_dim], [seq_len:max_seq_len, :] has been padded 0
                # TODO: generate register for dram addr over layer_idx loop.
                self.accelerator_memory_to_sram(self.LAYER0_K_ROPE_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size, 0x10000, seq_len * self.head_dim)
                duplicate_gqa_rows_pbi(0x10000, self.LAYER0_FLASH_K_DRAM)

                self.accelerator_memory_to_sram(self.LAYER0_V_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size, 0x20000, seq_len * self.head_dim)
                duplicate_gqa_rows_pbi(0x20000, self.LAYER0_FLASH_V_DRAM)

                total_flops += self.flash_attention_core(
                    head_dim=self.head_dim,
                    seq_len=aligned_seq_len_q,
                    Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
                    K_DRAM_ADDR=self.LAYER0_FLASH_K_DRAM,
                    V_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                    OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                    SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                    IDENTITY_DRAM_ADDR=self.IDENTITY_FULL_DRAM_ADDR if use_pbi else self.IDENTITY_DRAM_ADDR,
                    BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_DRAM,
                    use_pbi=use_pbi
                )
            total_flops += self.matmat_mul_core(
                M=seq_len,
                K=self.head_dim * self.group_size,
                N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                is_B_quantized=True,
                data_type=TYPE.IF8,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off,
                use_pbi=use_pbi,
            )
            if engine_master:
                total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                                OUTPUT_DRAM_ADDR=self.LAYER0_POST_ATTN_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_NORM_GAMMA + layer_off,
                                use_pbi=use_pbi)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_INPUT_DRAM, sram_address=0x10000, element_size=seq_len * self.vector_length)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_NORM_DRAM, sram_address=0x90000, element_size=seq_len * self.vector_length)
                # ToDo: no FLOPS return for eltwise_add_core and eltwise_mul_core
                self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=seq_len * self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, element_size=seq_len * self.vector_length)
                total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                                OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off,
                                use_pbi=use_pbi)
            total_flops += self.matmat_mul_core(
                M=seq_len,
                K=self.vector_length,
                N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                is_B_quantized=True,
                data_type=TYPE.IF8,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off,
                gelu_enable=True,
                use_pbi=use_pbi,
            )
            total_flops += self.matmat_mul_core(
                M=seq_len,
                K=self.vector_length,
                N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                is_B_quantized=True,
                data_type=TYPE.IF8,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off,
                use_pbi=use_pbi,
            )
            if engine_master:
                # TODO: chunk this MLP eltwise to lift the seq_len cap below.
                assert seq_len * self.mlp_elements * 2 + 0x10000 <= URAM_FULL_ELEMENTS * 2, \
                    f"prefill seq_len={seq_len} overflows URAM-staged MLP eltwise"
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_GATE_DRAM, sram_address=0x10000, element_size=seq_len * self.mlp_elements)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_UP_DRAM, sram_address=0x90000, element_size=seq_len * self.mlp_elements)
                self.eltwise_mul_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=seq_len * self.mlp_elements)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_MLP_MULT_DRAM, element_size=seq_len * self.mlp_elements)
            total_flops += self.matmat_mul_core(
                M=seq_len,
                K=self.mlp_elements,
                N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_MLP_MULT_DRAM,
                B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_QUANT + layer_off,
                OUTPUT_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                is_B_quantized=True,
                data_type=TYPE.IF8,
                SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off,
                use_pbi=use_pbi,
            )
            if engine_master:
                total_flops += self.rms_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                                OUTPUT_DRAM_ADDR=self.LAYER0_POST_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_FFW_NORM_GAMMA + layer_off,
                                use_pbi=use_pbi)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, sram_address=0x10000, element_size=seq_len * self.vector_length)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_MLP_NORM_DRAM, sram_address=0x90000, element_size=seq_len * self.vector_length)
                self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=seq_len * self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, element_size=seq_len * self.vector_length)
        self.generate_instruction_halt()
        prefill_program_addr = self.get_program_dram_addr() + count_at_start * 32
        prefill_program_size = (self.capture_count - count_at_start) * 32
        _SILENT_MODE = False

        return {
            "prefill_seq_len": prefill_seq_len,
            "start_addr": prefill_program_addr,
            "size_bytes": prefill_program_size,
            "flops": total_flops,
        }
        
    def _compile_decoder_programs(self, layer_size: int = 26, use_pbi: bool = False) -> dict:
        """Compile decoder programs for all seq_len buckets."""
        if not getattr(self, "is_capture_on", False):
            raise RuntimeError("_compile_decoder_programs() requires an active capture session")
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        segment_instruction_counts = []
        total_flops_list = []
        decoder_count_at_start = self.capture_count

        global _SILENT_MODE
        _SILENT_MODE = True
        for seq_len in self._cfg["model"]["decoder_seq_len_buckets"]:
            count_at_start = self.capture_count
            total_flops = 0
            for layer_idx in range(layer_size):
                layer_off = layer_idx * LAYER_WEIGHT_SIZE
                if layer_idx != 0:
                    self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, sram_address=0x10000, element_size=self.vector_length)
                    self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_INPUT_DRAM, element_size=self.vector_length)
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_PRE_NORM_GAMMA + layer_off,
                              use_pbi=use_pbi)
                total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=self.head_dim * self.group_size,
                                                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                                                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_QUANT + layer_off,
                                                    OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                                                    is_B_quantized=True,
                                                    data_type=TYPE.IF8,
                                                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ_SCALE + layer_off,
                                                    use_pbi=use_pbi
                                                    )
                total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=self.head_dim,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.IF8,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ_SCALE + layer_off,
                    use_pbi=use_pbi
                    )
                total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=self.head_dim,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.IF8,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ_SCALE + layer_off,
                    use_pbi=use_pbi
                    )
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_FLASH_V_DRAM, sram_address=0x10000, element_size=self.head_dim)
                self.generate_instruction_add_imm(self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter(self.LAYER0_V_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size), self.TMP_REG)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=0, element_size=self.head_dim)
                self.overwrite_instruction_with_general_register(self.TMP_REG)
                total_flops += self.rms_norm_core_dram(M=1, N=self.head_dim, A_DRAM_ADDR=self.LAYER0_K_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_K_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_NORM_GAMMA + layer_off,
                              use_pbi=use_pbi)
                total_flops += self.rms_norm_core_dram(M=self.group_size, N=self.head_dim, A_DRAM_ADDR=self.LAYER0_Q_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_Q_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_NORM_GAMMA + layer_off,
                              use_pbi=use_pbi)

                ROPE_WEIGHT_ADDR = self.DRAM_ADDR_ROPE_GLOBAL if layer_idx in self._rope_global_layers else self.DRAM_ADDR_ROPE_LOCAL
                total_flops += self.rope_hf_core(N=self.head_dim, input_dram_addr=self.LAYER0_K_NORM_DRAM, output_dram_addr=self.LAYER0_K_ROPE_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size,
                        cos_dram_addr=ROPE_WEIGHT_ADDR, sin_dram_addr=ROPE_WEIGHT_ADDR + self.head_dim * self.bytes_per_element,
                        rope_size_reg=self.ROPE_SIZE_REG, output_addr_inc_reg=self.V_CACHE_SIZE_REG, tmp_reg=self.TMP_REG)
                for g in range(self.group_size):
                    total_flops += self.rope_hf_core(N=self.head_dim, input_dram_addr=self.LAYER0_Q_NORM_DRAM + g * self.head_dim * self.bytes_per_element, output_dram_addr=self.LAYER0_FLASH_Q_DRAM + g * self.head_dim * self.bytes_per_element,
                            cos_dram_addr=ROPE_WEIGHT_ADDR, sin_dram_addr=ROPE_WEIGHT_ADDR + self.head_dim * self.bytes_per_element,
                            rope_size_reg=self.ROPE_SIZE_REG, tmp_reg=self.TMP_REG)
                total_flops += self.decoder_group_attention_core(
                    group_size=self.group_size,
                    head_dim=self.head_dim,
                    seq_len=seq_len,
                    Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
                    K_DRAM_ADDR=self.LAYER0_K_ROPE_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size,
                    V_DRAM_ADDR=self.LAYER0_V_DRAM + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size,
                    OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                    IDENTITY_DRAM_ADDR=self.IDENTITY_DRAM_ADDR,
                    SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                    BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_DRAM,
                    use_pbi=use_pbi,
                )
                total_flops += self.matmat_mul_core(M=1, K=self.head_dim * self.group_size, N=self.vector_length,
                    A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.IF8,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_SCALE + layer_off,
                    use_pbi=use_pbi
                    )
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_POST_ATTN_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_NORM_GAMMA + layer_off,
                              use_pbi=use_pbi)

                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_INPUT_DRAM, sram_address=0x10000, element_size=self.vector_length)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_NORM_DRAM, sram_address=0x90000, element_size=self.vector_length)
                self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, element_size=self.vector_length)

                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_FFN_NORM_GAMMA + layer_off,
                              use_pbi=use_pbi)

                total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=self.mlp_elements,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_GATE_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.IF8,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_GATE_SCALE + layer_off,
                    gelu_enable=True,
                    use_pbi=use_pbi
                    )
                total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=self.mlp_elements,
                    A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_QUANT + layer_off,
                    OUTPUT_DRAM_ADDR=self.LAYER0_MLP_UP_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.IF8,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_UP_SCALE + layer_off,
                    use_pbi=use_pbi
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
                    data_type=TYPE.IF8,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_DOWN_SCALE + layer_off,
                    use_pbi=use_pbi
                    )
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_MLP_DOWN_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_POST_MLP_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_POST_FFW_NORM_GAMMA + layer_off,
                              use_pbi=use_pbi)

                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, sram_address=0x10000, element_size=self.vector_length)
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_MLP_NORM_DRAM, sram_address=0x90000, element_size=self.vector_length)
                self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, element_size=self.vector_length)

            if layer_size == self.LAYER_SIZE:
                total_flops += self.rms_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_OUTPUT_DRAM,
                    OUTPUT_DRAM_ADDR=self.OUTPUT_NORM_DRAM, GAMMA_DRAM_ADDR=self.DRAM_ADDR_OUTPUT_NORM_GAMMA,
                    use_pbi=use_pbi)
                total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=self.EMBEDDING_ELEMENTS,
                    A_DRAM_ADDR=self.OUTPUT_NORM_DRAM,
                    B_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_QUANT,
                    OUTPUT_DRAM_ADDR=self.LOGITS_DRAM,
                    is_B_quantized=True,
                    data_type=TYPE.IF8,
                    SCALE_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD_SCALE,
                    use_pbi=use_pbi,
                    write_back_disable=True
                    )

            self.generate_instruction_halt()
            segment_instruction_counts.append(self.capture_count - count_at_start)
            total_flops_list.append(total_flops)
        _SILENT_MODE = False
        decoder_program_addr = self.get_program_dram_addr() + decoder_count_at_start * 32
        program_sizes = [c * 32 for c in segment_instruction_counts]
        print(f"Decoder program compiled at 0x{decoder_program_addr:X}, {len(segment_instruction_counts)} segments, size {sum(program_sizes)} bytes")
        return {
            "instruction_counts": segment_instruction_counts,
            "program_sizes": program_sizes,
            "total_flops": total_flops_list,
        }

    def compile_gemma3(self, layer_size: int = 26, use_pbi: bool = False, slave_engine = None) -> None:
        """Compile a per-bucket prefill family + every decoder seq_len bucket into
        one combined instruction image.

        Layout in program DRAM (after load_instructions):
          [prefill seq_len=1][prefill seq_len=2]...[prefill seq_len=N]
            [decoder bucket 0][decoder bucket 1]...[decoder bucket M-1]

        N = model.prefill_max_seq_len (one prefill program per integer seq_len in
        [1, N]). At runtime the actual prompt length picks the matching prefill
        program by start address.

        Writes:
          - paths.instruction_bin : raw 32B-per-instruction stream, ready for DMA
            into program DRAM.
          - paths.instruction_meta : per-stage start addresses, sizes, FLOPs.

        If the bin and meta already exist, this is a no-op so multiple runs reuse
        the same image. Delete either file (or change config in a way that makes
        the bin stale) to force a recompile.
        """
        instruction_bin_path = os.path.join(self.script_dir, self._cfg["paths"]["instruction_bin"])
        instruction_meta_path = os.path.join(self.script_dir, self._cfg["paths"]["instruction_meta"])
        if os.path.exists(instruction_bin_path) and os.path.exists(instruction_meta_path):
            print(f"Reusing existing instruction image at {instruction_bin_path}")
            print(f"  delete {instruction_bin_path} to force recompile.")
            return

        prefill_max_seq_len = int(self._cfg["model"]["prefill_max_seq_len"])
        prefill_seq_len_buckets = list(range(1, prefill_max_seq_len + 1))

        self.clear_inst_id()
        self.start_capture()
        prefill_programs = []
        print(f"Compiling {len(prefill_seq_len_buckets)} prefill buckets (seq_len 1..{prefill_max_seq_len})...")
        compile_t0 = time.perf_counter()
        for prefill_seq_len in prefill_seq_len_buckets:
            prog = self._compile_prefill_program(
                prefill_seq_len=prefill_seq_len, layer_size=layer_size, use_pbi=True
            )
            prefill_programs.append(prog)
            if prefill_seq_len % 8 == 0 or prefill_seq_len == prefill_max_seq_len:
                elapsed = time.perf_counter() - compile_t0
                print(f"  bucket {prefill_seq_len}/{prefill_max_seq_len} compiled, "
                      f"size={prog['size_bytes']} bytes, elapsed={elapsed:.1f}s")
        decoder_program = self._compile_decoder_programs(layer_size=layer_size, use_pbi=True)
        self.stop_capture()

        os.makedirs(os.path.dirname(instruction_bin_path), exist_ok=True)
        instruction_bytes = bytearray()
        for inst in self.capture_buffer:
            instruction_bytes.extend(inst.get_bytes())
        with open(instruction_bin_path, "wb") as f:
            f.write(instruction_bytes)
        self.clear_capture_buffer()

        # Compute per-program start addresses (cumulative offsets from base).
        instruction_base_addr = self.get_program_dram_addr()
        next_addr = instruction_base_addr
        prefill_program_addrs = []
        for prog in prefill_programs:
            prefill_program_addrs.append(next_addr)
            next_addr += prog["size_bytes"]
        decoder_program_addrs = []
        for program_size in decoder_program["program_sizes"]:
            decoder_program_addrs.append(next_addr)
            next_addr += program_size

        # Buckets themselves (1..prefill_max_seq_len, decoder_seq_len_buckets) live
        # in gemma3_config.json since they are part of the model/instruction
        # design. Only fields that the compiler decides at capture time end up
        # here: per-bucket start addresses, per-bucket sizes, per-bucket FLOPs.
        metadata = {
            "instruction_bin": os.path.relpath(instruction_bin_path, self.script_dir),
            "instruction_base_addr": f"0x{instruction_base_addr:X}",
            "instruction_total_size": len(instruction_bytes),
            "prefill_program_start_addrs": [f"0x{a:X}" for a in prefill_program_addrs],
            "prefill_program_sizes": [p["size_bytes"] for p in prefill_programs],
            "prefill_total_flops": [p["flops"] for p in prefill_programs],
            "decoder_program_start_addrs": [f"0x{addr:X}" for addr in decoder_program_addrs],
            "decoder_program_sizes": decoder_program["program_sizes"],
            "decoder_total_flops": decoder_program["total_flops"],
        }

        if slave_engine is not None:
            # Dual-engine slave compiles the same per-bucket prefill family so
            # any prompt length supported by master is also runnable on slave.
            slave_engine.clear_inst_id()
            slave_engine.start_capture()
            slave_prefill_programs = []
            for prefill_seq_len in prefill_seq_len_buckets:
                slave_prefill_programs.append(
                    slave_engine._compile_prefill_program(
                        prefill_seq_len=prefill_seq_len, layer_size=layer_size, use_pbi=use_pbi
                    )
                )
            slave_engine.stop_capture()
            slave_bin_path = os.path.join(self.script_dir, "gemma3_bin", "prefill_program_slave.bin")
            slave_program_bytes = bytearray()
            for inst in slave_engine.capture_buffer:
                slave_program_bytes.extend(inst.get_bytes())
            with open(slave_bin_path, "wb") as f:
                f.write(slave_program_bytes)
            slave_engine.clear_capture_buffer()
            slave_base = slave_engine.get_program_dram_addr()
            slave_addrs = []
            slave_next = slave_base
            for prog in slave_prefill_programs:
                slave_addrs.append(slave_next)
                slave_next += prog["size_bytes"]
            metadata["dual_engine_slave_prefill"] = {
                "bin_path": os.path.relpath(slave_bin_path, self.script_dir),
                "start_addrs": [f"0x{a:X}" for a in slave_addrs],
                "flops": [p["flops"] for p in slave_prefill_programs],
            }

        with open(instruction_meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Combined gemma3 instruction image written to {instruction_bin_path} ({len(instruction_bytes)} bytes)")
        print(f"  prefill: {len(prefill_seq_len_buckets)} buckets (seq_len 1..{prefill_max_seq_len}), "
              f"total {sum(p['size_bytes'] for p in prefill_programs)} bytes")
        print(f"  decoder: {len(decoder_program_addrs)} buckets, "
              f"total {sum(decoder_program['program_sizes'])} bytes")
        print(f"Metadata written to {instruction_meta_path}")

    def run_gemma3(self, slave_engine = None) -> dict:
        """Load the unified instruction image once, then run prefill + decoder loop.

        The whole gemma3_instruction.bin is DMA'd into program DRAM in a single
        load. Prefill is dispatched by picking the start address of the bucket
        matching the runtime prefill seq_len (= len(prompt) - 1). Per-bucket
        decoder programs are dispatched the same way, all from the metadata.
        """
        meta_path = os.path.join(self.script_dir, self._cfg["paths"]["instruction_meta"])
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.load_instructions(os.path.join(self.script_dir, meta["instruction_bin"]))

        # Bucket set comes from config (design-time); per-bucket runtime info
        # (addresses, sizes, flops) comes from the compile-time meta.
        prefill_max_seq_len = int(self._cfg["model"]["prefill_max_seq_len"])
        prefill_program_addrs = [_parse_offset(a) for a in meta["prefill_program_start_addrs"]]
        prefill_flops_list = meta["prefill_total_flops"]
        assert len(prefill_program_addrs) == prefill_max_seq_len, (
            f"meta has {len(prefill_program_addrs)} prefill programs but "
            f"model.prefill_max_seq_len={prefill_max_seq_len}. "
            f"Delete {self._cfg['paths']['instruction_bin']} to force a recompile."
        )
        decoder_program_addrs = [_parse_offset(addr) for addr in meta["decoder_program_start_addrs"]]
        flops_per_token = meta["decoder_total_flops"]

        prefill_seq = self.prefill_seq
        if prefill_seq is None:
            prefill_seq = tuple(self._cfg["default_prefill_tokens"])
        if len(prefill_seq) <= 1:
            raise ValueError("Prefill sequence must have at least 2 tokens.")
        # Prefill program processes all but the last token (the last one is the
        # first input to the decoder). Buckets are 1..prefill_max_seq_len, so the
        # compiled bucket index for seq_len S is just S - 1.
        prefill_seq = prefill_seq[:-1]
        prefill_seq_len = len(prefill_seq)
        if not 1 <= prefill_seq_len <= prefill_max_seq_len:
            raise ValueError(
                f"prefill_seq_len={prefill_seq_len} is outside compiled buckets "
                f"[1, {prefill_max_seq_len}]. Bump model.prefill_max_seq_len and recompile."
            )
        bucket_idx = prefill_seq_len - 1
        prefill_program_addr = prefill_program_addrs[bucket_idx]
        flops_prefill = prefill_flops_list[bucket_idx]
        self.seq_len = prefill_seq_len

        slave_prefill_flops = None
        slave_prefill_addr = None
        slave_prefill = meta.get("dual_engine_slave_prefill")
        if slave_engine is not None and slave_prefill is not None:
            slave_bin_path = os.path.join(self.script_dir, slave_prefill["bin_path"])
            slave_engine.reset_program_dram_addr()
            slave_engine.load_instructions(slave_bin_path)
            slave_prefill_addr = _parse_offset(slave_prefill["start_addrs"][bucket_idx])
            slave_prefill_flops = slave_prefill["flops"][bucket_idx]

        print(f"\n--- Starting prefill (seq_len={prefill_seq_len}, bucket {bucket_idx + 1}/{prefill_max_seq_len}) ---")
        prompt_text = self.tokenizer.decode(list(self.prefill_seq), skip_special_tokens=False)
        print(f"Prompt ({len(self.prefill_seq)} tokens): {prompt_text!r}")
        timer = time.perf_counter()
        if slave_prefill_addr is not None:
            slave_engine.program_execute(slave_prefill_addr, timeout=0)

        seq_len = prefill_seq_len
        q_seq_len = seq_len * self.group_size
        aligned_seq_len_q = ((q_seq_len + 63) // 64) * 64

        embedding_tensor = self.get_embedding_for_tokens(prefill_seq)
        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
        bias_one_group = torch.full((aligned_seq_len_q, aligned_seq_len_q), float("-inf"), dtype=torch.bfloat16)
        valid_mask = torch.tril(torch.ones(aligned_seq_len_q, aligned_seq_len_q, dtype=torch.bool), diagonal=0) if not self.causal_mask_upper else torch.triu(torch.ones(aligned_seq_len_q, aligned_seq_len_q, dtype=torch.bool), diagonal=0)
        bias_one_group.masked_fill_(valid_mask, 0.0)
        bias_one_group[:, q_seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_one_group)
        latency_hw_prefill, flop_rate_hw_prefill = self.program_execute(prefill_program_addr, flops=flops_prefill)
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
        while self.seq_len < max_seq_len:
            decoder_token_cnt += 1
            _SILENT_MODE = True
            self.seq_len += 1
            aligned_seq_len_q = ((self.seq_len + 63) // 64) * 64
            prog_idx = min((self.seq_len - 1) // 64, 7)
            prog_addr = decoder_program_addrs[prog_idx]
            flops_per_token_idx = flops_per_token[prog_idx] if flops_per_token else None

            self.isa_add_set_core(self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter((self.seq_len - 1) * self.k_size))
            self.isa_add_set_core(self.ROPE_SIZE_REG, ue_35bit_addr_shifter((self.seq_len - 1) * self.k_size * 2))
            embedding_tensor = self.get_embedding_for_tokens([token_id])
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
            bias_host = torch.full((1, aligned_seq_len_q), -1e36, dtype=torch.bfloat16)
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

        token_cnt_decoded = self.seq_len
        latency_hw_decoder = total_latency
        flop_rate_hw_decoder = total_flop_rate
        latency_decoder = time.perf_counter() - timer
        print(f"\nDecoder done in {latency_prefill + latency_decoder:.2f} seconds, speed: {(token_cnt_decoded - len(self.prefill_seq) + 1) / latency_decoder:.2f} tokens/s, total {token_cnt_decoded} tokens.")
        print(f"HW counter: Latency: {(latency_hw_prefill + latency_hw_decoder) / 1e6:.2f} seconds, decoder average Gflops: {flop_rate_hw_decoder / (token_cnt_decoded - len(self.prefill_seq) + 1):.2f} Gflops")
        
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
    assert dual_engine == False, "Dual engine is not supported yet for pbi mode"
    ue = Gemma3_UnifiedEngine(local_weights=args.local_weights, dual_engine=dual_engine)
    ue.set_prefill_seq(args.prompt)

    if dual_engine:
        ue2 = Gemma3_UnifiedEngine(local_weights=args.local_weights, dual_engine=True, engine_slave=True)
        ue2.set_prefill_seq(args.prompt)

    print(f"\n--- Compiling ---")
    timer = time.perf_counter()
    ue.compile_gemma3(slave_engine=ue2 if dual_engine else None)
    print(f"Compile done in {time.perf_counter() - timer:.2f} seconds")

    run_result = ue.run_gemma3(slave_engine=ue2 if dual_engine else None)
    print("Gemma3 test ends.")

    global _SILENT_MODE
    _SILENT_MODE = True
    ue.software_reset()

if __name__ == "__main__":
    main()
