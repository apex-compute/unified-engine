#!/usr/bin/env python3
"""
GPT-2 Base (124M) inference on accelerator: prefill + decode.

  - Config from gpt2_config.json; weights from a single bin (see below).
  - Dynamic-PBI flow (see src/template/core_changes.md): ONE prefill program +
    ONE decoder program compiled into a single instruction bin. Runtime row
    counts and attention bucket selection are driven by fixed-index GPRs
    (gpr_seq_len / gpr_bucket_idx) primed by a tiny per-call preamble program.
  - Shared-subroutine attention (core_changes.md §7): flash_attention_core
    (prefill) and decoder_group_attention_core (decode) are each compiled ONCE
    after the program HALT; every per-head call site jumps in and the body
    returns via JUMP_REG_ABS(gpr_ret_id).
  - If gpt2_bin/gpt2_instruction.bin + .json exist, compile is skipped and the
    cached image is reused (delete the bin to force recompile).

Architecture notes (vs LLaMA/Gemma3):
  - 12 layers, 12 heads (MHA, group_size=1), actual_head_dim=64.
  - LayerNorm with bias (not RMSNorm). eps=1e-5.
  - Learned positional embeddings (not RoPE). Added host-side.
  - Fused QKV projection (c_attn) split into Q, K, V during weight gen.
  - GELU activation in MLP (not SwiGLU). No gate projection.
  - Bias in all linear layers (Q, K, V, output, c_fc, c_proj).
  - LM head weight tied to input embedding.
  - HuggingFace GPT-2 uses Conv1D: weights are (in, out), need transpose.
  - BF16 weights (no quantization) — GPT-2 stays on its original weight scheme.
    Decode projections use matmat_mul_core(M=1), because quantized_matmat_core
    requires quantized weight/scale tensors that this GPT-2 bin does not have.

Weights:
  - Default: gpt2_bin/weights_gpt2_hf.bin (generated from HF model if missing).
  - --local-weights: use gpt2_bin/full_model_weights.bin instead.

Usage:
  python models/gpt2/gpt2_test.py --prompt "The scientists at MIT announced today that they have discovered "
"""

import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import time

GPT2_INSTRUCTION_CACHE_VERSION = 4

# This file's folder; user_dma_core.py is two folders up (repo root); that directory is added to sys.path.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

import user_dma_core
from user_dma_core import DMA_DEVICE_H2C, UE_MODE, UE_FMAX_CONTEXT_SIZE, UE_VECTOR_SIZE, set_dma_device, ue_35bit_addr_shifter, INSTRUCTION_SIZE_BYTES
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
    """Parse offset/size from JSON: int or hex string like '0x499F000'."""
    if isinstance(val, str):
        return int(val, 0)
    return int(val)


def _bf16_bytes(tensor: torch.Tensor) -> bytes:
    """Convert a tensor to bf16 raw bytes."""
    return tensor.detach().cpu().to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()


def weight_bin_generate(script_dir: str | None = None, output_path: str | None = None) -> str:
    """Generate weights_gpt2_hf.bin from HuggingFace model per gpt2_config.json layout.
    Returns the path to the written file.

    Key differences from LLaMA/Qwen3 weight gen:
      - Conv1D transpose: HF GPT-2 stores linear weights as (in, out); we transpose to (out, in).
      - Fused QKV split: c_attn (768, 2304) → transpose → split into Q, K, V each (768, 768).
      - Bias vectors stored as separate bf16 regions.
      - LayerNorm weight + bias (not RMSNorm gamma only).
      - Learned positional embeddings stored as non-layer region.
      - No RoPE table generation.
      - All weights stored as bf16 (no quantization) for best quality on 124M model.
    """
    script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
    cfg = _load_config(script_dir)
    weight_defs = cfg["_weight_defs"]
    paths = cfg["paths"]
    paths_full = os.path.join(script_dir, paths["weights_bin"])
    out_path = output_path or paths_full

    model, model_dir = _ensure_hf_model(script_dir, cfg)
    emb_cfg = cfg["special"]["embedding"]
    token_embd_offset = _parse_offset(emb_cfg["token_embd_offset"])
    token_embd_size = _parse_offset(emb_cfg["token_embd_size"])
    LAYER_WEIGHT_SIZE = weight_defs["LAYER_WEIGHT_SIZE"]
    num_layers = cfg["file_info"]["num_layers"]
    hidden_size = cfg["file_info"]["hidden_size"]
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

    # Embedding: GPT-2 does NOT scale by sqrt(hidden_size)
    # Pad from actual vocab (50257) to padded vocab (50304) for UE_VECTOR_SIZE alignment
    embed_raw = model.get_input_embeddings().weight.detach().cpu().to(torch.bfloat16)
    padded_vocab = cfg["special"]["embedding"]["vocab_size"]  # 50304
    actual_vocab = embed_raw.shape[0]  # 50257
    if actual_vocab < padded_vocab:
        embed = torch.zeros(padded_vocab, embed_raw.shape[1], dtype=torch.bfloat16)
        embed[:actual_vocab] = embed_raw
    else:
        embed = embed_raw
    raw_emb = embed.contiguous().view(torch.uint8).numpy().tobytes()
    write_at(token_embd_offset, raw_emb)

    # Layers
    for layer_idx in range(num_layers):
        layer = model.transformer.h[layer_idx]

        # LayerNorm 1: weight and bias
        ln1_w = layer.ln_1.weight.detach().cpu().to(torch.bfloat16)
        ln1_b = layer.ln_1.bias.detach().cpu().to(torch.bfloat16)

        # Fused QKV: c_attn.weight is Conv1D (in=768, out=2304) → transpose to (2304, 768) → split
        c_attn_w = layer.attn.c_attn.weight.data.T.contiguous().to(torch.bfloat16)  # (2304, 768)
        c_attn_b = layer.attn.c_attn.bias.data.to(torch.bfloat16)  # (2304,)
        q_w, k_w, v_w = c_attn_w.split(hidden_size, dim=0)  # each (768, 768)
        q_b, k_b, v_b = c_attn_b.split(hidden_size, dim=0)  # each (768,)

        # Output projection: Conv1D (in=768, out=768) → transpose
        o_w = layer.attn.c_proj.weight.data.T.contiguous().to(torch.bfloat16)  # (768, 768)
        o_b = layer.attn.c_proj.bias.data.to(torch.bfloat16)  # (768,)

        # LayerNorm 2: weight and bias
        ln2_w = layer.ln_2.weight.detach().cpu().to(torch.bfloat16)
        ln2_b = layer.ln_2.bias.detach().cpu().to(torch.bfloat16)

        # MLP c_fc: Conv1D (in=768, out=3072) → transpose
        fc_w = layer.mlp.c_fc.weight.data.T.contiguous().to(torch.bfloat16)  # (3072, 768)
        fc_b = layer.mlp.c_fc.bias.data.to(torch.bfloat16)  # (3072,)

        # MLP c_proj: Conv1D (in=3072, out=768) → transpose
        proj_w = layer.mlp.c_proj.weight.data.T.contiguous().to(torch.bfloat16)  # (768, 3072)
        proj_b = layer.mlp.c_proj.bias.data.to(torch.bfloat16)  # (768,)

        # Write all weights as bf16 in config structure order
        region_tensors = [
            ln1_w, ln1_b, q_w, q_b, k_w, k_b, v_w, v_b,
            o_w, o_b, ln2_w, ln2_b, fc_w, fc_b, proj_w, proj_b,
        ]
        for i, entry in enumerate(blk0_structure):
            off_key = entry["key"]
            sz = weight_defs[f"{off_key}_SIZE"]
            file_off = weight_defs[off_key] + layer_idx * LAYER_WEIGHT_SIZE
            raw = (_bf16_bytes(region_tensors[i]) + b"\x00" * sz)[:sz]
            write_at(file_off, raw)

    # Positional embeddings (replaces RoPE table generation)
    pos_embed = model.transformer.wpe.weight.detach().cpu().to(torch.bfloat16)  # (1024, 768)
    sz = weight_defs["POS_EMBED_SIZE"]
    raw = (pos_embed.contiguous().view(torch.uint8).numpy().tobytes() + b"\x00" * sz)[:sz]
    write_at(weight_defs["POS_EMBED"], raw)

    # OUTPUT LayerNorm: weight and bias
    ln_f_w = model.transformer.ln_f.weight.detach().cpu().to(torch.bfloat16)
    sz = weight_defs["OUTPUT_LN_WEIGHT_SIZE"]
    raw = (ln_f_w.contiguous().view(torch.uint8).numpy().tobytes() + b"\x00" * sz)[:sz]
    write_at(weight_defs["OUTPUT_LN_WEIGHT"], raw)

    ln_f_b = model.transformer.ln_f.bias.detach().cpu().to(torch.bfloat16)
    sz = weight_defs["OUTPUT_LN_BIAS_SIZE"]
    raw = (ln_f_b.contiguous().view(torch.uint8).numpy().tobytes() + b"\x00" * sz)[:sz]
    write_at(weight_defs["OUTPUT_LN_BIAS"], raw)

    # LM_HEAD: GPT-2 ties lm_head weight to input embedding (use padded embedding).
    # Padded rows (50257..50303) are zero → zero logits for those tokens.
    # Hardware argmax over 50304-dim may pick a padded index; handled in run_gpt2.
    lm_head_w = embed  # already padded to 50304, padded rows are zero
    sz = weight_defs["LM_HEAD_WEIGHT_SIZE"]
    raw = (_bf16_bytes(lm_head_w) + b"\x00" * sz)[:sz]
    write_at(weight_defs["LM_HEAD_WEIGHT"], raw)

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
        snapshot_download(repo_id=hf_repo, local_dir=model_dir, local_dir_use_symlinks=False,
                         allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"])
        _original_print("Download complete.")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map=None, trust_remote_code=True
    )
    return model, model_dir


def _load_config(script_dir: str) -> dict:
    """Load gpt2_config.json and build weight_defs (offset/size dict) from regions."""
    config_path = os.path.join(script_dir, "gpt2_config.json")
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
# GPT-2 unified engine
# -----------------------------------------------------------------------------
class GPT2_UnifiedEngine(UnifiedEngine):
    """UnifiedEngine for GPT-2 Base (124M): loads config + weight bin, compile_gpt2 (one
    prefill + one decoder program in a single bin), run_gpt2 (preamble-primed prefill +
    decode loop).

    Key architectural differences from LLaMA/Gemma3:
      - LayerNorm with bias (not RMSNorm). Uses layer_norm_core_dram with GAMMA + BETA.
      - Learned positional embeddings added host-side (not RoPE on FPGA).
      - MHA (group_size=1): each of 12 Q heads maps 1:1 to a KV head, no duplication.
      - GELU activation in MLP (gelu_enable=True on c_fc matmul). No gate projection.
      - Bias in all linear layers (Q, K, V, output, c_fc, c_proj).
      - No Q/K norm, no post-attention norm, no post-FFN norm.
      - LM head weight tied to input embedding.
      - BF16 weights throughout: prefill matmuls use matmat_mul_core(gpr_M_reg=...),
        decode projection matmuls use matmat_mul_core(M=1).
    """

    def __init__(self, script_dir: str | None = None, hf_model_dir: str | None = None, weights_bin: str | None = None):
        super().__init__()
        self.script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
        self._cfg = _load_config(self.script_dir)
        self.weight_defs = self._cfg["_weight_defs"]

        fi = self._cfg["file_info"]
        model = self._cfg["model"]
        paths = self._cfg["paths"]

        self.vector_length = fi["hidden_size"]         # 768
        self.head_dim = fi["head_dim"]                 # 768 (all heads combined)
        self.bytes_per_element = fi["bytes_per_element"]
        self.group_size = fi["group_size"]             # 1 (MHA)
        self.mlp_elements = fi["mlp_elements"]         # 3072
        self.hf_model_dir = hf_model_dir or os.path.join(self.script_dir, paths["hf_model_dir"])
        self.q_size = self.head_dim * self.group_size * self.bytes_per_element  # 768*1*2 = 1536
        self.k_size = self.head_dim * self.bytes_per_element                   # 768*2 = 1536
        self.actual_head_dim = fi["actual_head_dim"]   # 64
        self.num_kv_heads = fi["num_kv_heads"]         # 12
        self.MAX_CONTEXT_SIZE = model["max_context_size"]  # 1024
        self.PREFILL_CONTEXT_SIZE = int(model.get("prefill_context_size", model.get("prefill_max_seq_len", 256)))
        self.PREFILL_MAX_SEQ_LEN = int(model.get("prefill_max_seq_len", self.PREFILL_CONTEXT_SIZE))
        self.LAYER_SIZE = fi["num_layers"]             # 12
        self.EMBEDDING_ELEMENTS = fi["embedding_vocab"]    # 50304
        fixed = self._cfg.get("fixed_isa_regs", {})
        self.V_CACHE_SIZE_REG = fixed["V_CACHE_SIZE_REG"]
        self.TMP_REG = fixed["TMP_REG"]
        self.gpr_bucket_idx = fixed["GPR_BUCKET_IDX_REG"]
        self.gpr_seq_len = fixed["GPR_SEQ_LEN_REG"]
        self._isa_reg_counter = max(fixed.values()) + 1  # must start past all fixed ISA regs
        self.causal_mask_upper = False
        self._end_of_turn_token_id = model["end_of_turn_token_id"]  # 50256

        bin_path = weights_bin or paths["weights_bin"]
        full_path = os.path.join(self.script_dir, bin_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Bin file not found: {full_path}")
        with open(full_path, "rb") as f:
            self.weight_bin = f.read()
        self.weight_init()
        self.tensor_init()

    def get_embedding_for_tokens(self, token_ids: list[int] | tuple, start_pos: int = 0) -> torch.Tensor:
        """Return (len(token_ids), vector_length) bfloat16 tensor with token + positional embeddings added host-side."""
        tid_t = torch.tensor(token_ids, dtype=torch.long)
        tok_emb = torch.zeros(len(token_ids), self.vector_length, dtype=torch.bfloat16)
        valid = tid_t < self.embedding_weight.shape[0]
        tok_emb[valid] = self.embedding_weight[tid_t[valid]]
        # Add learned positional embeddings
        pos_ids = torch.arange(start_pos, start_pos + len(token_ids), dtype=torch.long)
        pos_ids = pos_ids.clamp(max=self.pos_embedding_weight.shape[0] - 1)
        pos_emb = self.pos_embedding_weight[pos_ids]
        return (tok_emb + pos_emb).to(torch.bfloat16)

    def weight_init(self) -> None:
        """Initialize DRAM: load HF embedding+tokenizer+pos_embed, layer weights from bin, OUTPUT_LN/LM_HEAD from bin."""
        model, model_dir = _ensure_hf_model(self.script_dir, self._cfg)
        # GPT-2 does NOT scale the embedding; pad to 50304 for alignment
        embed_raw = model.get_input_embeddings().weight.detach().cpu().to(torch.bfloat16)
        padded_vocab = self.EMBEDDING_ELEMENTS  # 50304
        if embed_raw.shape[0] < padded_vocab:
            embed = torch.zeros(padded_vocab, embed_raw.shape[1], dtype=torch.bfloat16)
            embed[:embed_raw.shape[0]] = embed_raw
        else:
            embed = embed_raw
        self.embedding_weight = embed
        # Load learned positional embeddings for host-side addition
        self.pos_embedding_weight = model.transformer.wpe.weight.detach().cpu().to(torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        base_layer0 = self.weight_defs["BLK0_LN1_WEIGHT"]
        blk0_regions = [
            (s["key"], f"{s['key']}_SIZE", s["attr"])
            for s in self._cfg["layers"]["structure"]
        ]
        non_layer = [
            (s["key"], f"{s['key']}_SIZE", s["attr"])
            for s in self._cfg["layers"]["non_layer"]
        ]

        # Verify layer 0 fits within LAYER_WEIGHT_SIZE (stride includes 4K alignment padding)
        last_key = self._cfg["layers"]["structure"][-1]["key"]
        layer0_end = (self.weight_defs[last_key] - base_layer0
                      + self.weight_defs[f"{last_key}_SIZE"])
        assert layer0_end <= LAYER_WEIGHT_SIZE, (
            f"Layer 0 size overflow: computed {layer0_end} > LAYER_WEIGHT_SIZE {LAYER_WEIGHT_SIZE}"
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

        print(f"    Allocate weights end at DRAM address: 0x{self.get_params_dram_addr():X}, usage: {self.get_params_dram_usage()} bytes")
        print("Tokenizer loaded successfully.")

    def tensor_init(self) -> None:
        """Initialize hardware DRAM tensors for GPT-2 (layer-wise overlap except for kv cache).

        Simplifications vs LLaMA:
          - No V_PROJ_TEMP lo|hi interleaving to undo; V heads are contiguous 64-dim slices.
          - No MLP_GATE_DRAM or MLP_MULT_DRAM (no SwiGLU gate).
          - group_size=1: flash buffers indexed per head at seq_len * actual_head_dim.

        Decode shared-subroutine marshalling buffers (core_changes.md §7c):
          - FLASH_K / FLASH_V receive the gathered per-head K/V history each step.
          - FLASH_Q receives the current head's 64-dim Q at its base.
          - LAYER0_FLASH_ATTN_P_DRAM is the flash bucket-dispatcher softmax scratch (§3d).
        """
        seq_len = self.MAX_CONTEXT_SIZE
        q_seq_len = seq_len * self.group_size  # = seq_len for MHA
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64

        print(f"Allocate tensor dram start at DRAM address: 0x{self.get_tensor_dram_addr():X}")
        # KV cache: per layer, per head, max_context positions, each actual_head_dim elements
        self.LAYER0_V_DRAM = self.allocate_tensor_dram(self.LAYER_SIZE * self.MAX_CONTEXT_SIZE * self.k_size)
        self.LAYER0_K_DRAM_CACHE = self.allocate_tensor_dram(self.LAYER_SIZE * self.MAX_CONTEXT_SIZE * self.k_size)
        zero_pad = torch.zeros(self.LAYER_SIZE * self.MAX_CONTEXT_SIZE * self.k_size, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_K_DRAM_CACHE, zero_pad)
        # Zero tensor, identity matrix (actual_head_dim == UE_VECTOR_SIZE == 64, so one
        # identity slot serves both the flash and the decoder attention subroutines —
        # seeded once here, never re-DMAed (core_changes.md §1).
        zero_add = torch.zeros(seq_len * self.head_dim * self.bytes_per_element, dtype=torch.bfloat16)
        self.ZERO_DRAM_ADDR = self.allocate_tensor_dram(seq_len * self.head_dim * self.bytes_per_element)
        self.dma_to_accelerator_memory(self.ZERO_DRAM_ADDR, zero_add)
        self.IDENTITY_DRAM_ADDR = self.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * self.bytes_per_element)
        self.dma_to_accelerator_memory(self.IDENTITY_DRAM_ADDR, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))
        # Flash attention buffers. Sized generously (aligned_seq_len * head_dim) so the
        # decode-time history gather (aligned_seq_len * actual_head_dim) always fits.
        self.LAYER0_FLASH_Q_DRAM = self.allocate_tensor_dram(aligned_seq_len * self.head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_K_DRAM = self.allocate_tensor_dram(aligned_seq_len * self.head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_V_DRAM = self.allocate_tensor_dram(aligned_seq_len * self.head_dim * self.bytes_per_element)
        zero_flash = torch.zeros(aligned_seq_len * self.head_dim * self.bytes_per_element, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_Q_DRAM, zero_flash)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_K_DRAM, zero_flash)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_V_DRAM, zero_flash)
        # Layer intermediate tensors
        self.LAYER0_INPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_Q_DRAM = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_K_DRAM = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_V_PROJ_DRAM = self.allocate_tensor_dram(seq_len * self.k_size)
        # Per-head flash output
        self.LAYER0_FLASH_OUT_HEAD_DRAM = self.allocate_tensor_dram(
            aligned_seq_len * self.actual_head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.head_dim * self.group_size * self.bytes_per_element)
        self.LAYER0_FLASH_SCRATCH_DRAM = self.allocate_tensor_dram(max(self.head_dim, UE_FMAX_CONTEXT_SIZE) * aligned_seq_len * 2 + self.head_dim * aligned_seq_len * 2)
        self.LAYER0_FLASH_BIAS_DRAM = self.allocate_tensor_dram(aligned_seq_len * aligned_seq_len * self.bytes_per_element)
        # Flash PBI bucket-dispatcher softmax scratch (core_changes.md §3d)
        self.LAYER0_FLASH_ATTN_P_DRAM = self.allocate_tensor_dram(aligned_seq_len * aligned_seq_len * self.bytes_per_element)
        self.LAYER0_ATTN_PROJ_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_ATTN_RESIDUAL_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_MLP_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        # MLP: no gate/mult (no SwiGLU); just fc output and proj output
        self.LAYER0_MLP_FC_DRAM = self.allocate_tensor_dram(seq_len * self.mlp_elements * 2)
        self.LAYER0_MLP_PROJ_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.OUTPUT_NORM_DRAM = self.allocate_tensor_dram(1 * self.vector_length * self.bytes_per_element)
        self.LOGITS_DRAM = self.allocate_tensor_dram(1 * self.EMBEDDING_ELEMENTS * self.bytes_per_element)

        print(f"    Allocate tensor dram end at DRAM address: 0x{self.get_tensor_dram_addr():X}, usage: {self.get_tensor_dram_usage()} bytes")

    def _compile_prefill_program(self, template_seq_len: int, layer_size: int) -> dict:
        """Compile prefill into the active capture session.

        ``template_seq_len`` is used only for FLOPs accounting and static M= args;
        all runtime loop counts are driven by ``gpr_seq_len`` / ``gpr_bucket_idx``
        primed by the caller's preamble, so a single bin works for any seq_len
        up to PREFILL_CONTEXT_SIZE. Returns dict with ``size_bytes`` and ``flops``.

        GPT-2 differences from LLaMA:
          - LayerNorm with gamma+beta (not RMSNorm).
          - No RoPE steps (positions added host-side).
          - MHA scatter: group_size=1, contiguous 64-dim head slices (no lo|hi split).
          - GELU MLP (no SwiGLU gate × up).
          - Bias on all linear layers via C_DRAM_ADDR.
        """
        if not getattr(self, "is_capture_on", False):
            raise RuntimeError("_compile_prefill_program() requires an active capture session")
        count_at_start = self.capture_count
        seq_len = template_seq_len
        q_seq_len = seq_len * self.group_size  # = seq_len for MHA
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64

        global _SILENT_MODE
        _SILENT_MODE = True
        num_bucket = (self.PREFILL_MAX_SEQ_LEN * self.group_size + 63) // 64
        total_flops = 0
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        ahd = self.actual_head_dim   # 64
        nkvh = self.num_kv_heads     # 12
        bpe = self.bytes_per_element
        hd = self.head_dim           # 768

        # flash_attention_core compiled once as a subroutine after the layer loop.
        # Each call site sets gpr_ret_id to its return word address then jumps to the
        # subroutine; flash_attention returns via JUMP_REG_ABS(gpr_ret_id).
        program_dram_base = self.get_program_dram_addr()
        gpr_ret_id = self.alloc_isa_reg()
        call_site_jump_capture_indices: list[int] = []

        for layer_idx in range(layer_size):
            layer_off = layer_idx * LAYER_WEIGHT_SIZE
            if layer_idx != 0:
                # Inter-layer copy: previous layer's OUTPUT → next layer's INPUT.
                # Per-token PBI loop (one row of vector_length per iter, gpr_seq_len trips).
                self._emit_pbi_scatter_per_token(
                    read_base=self.LAYER0_OUTPUT_DRAM,
                    read_stride_bytes=self.vector_length * bpe,
                    write_specs=[(self.LAYER0_INPUT_DRAM, self.vector_length * bpe)],
                    sram_byte_addr=0x10000,
                    element_count=self.vector_length,
                    gpr_seq_len=self.gpr_seq_len,
                    template_seq_len=seq_len,
                )

            # LayerNorm 1 (gamma + beta) — PBI row loop driven by gpr_seq_len
            total_flops += self.layer_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                              GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_LN1_GAMMA + layer_off,
                              BETA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_LN1_BETA + layer_off,
                              gpr_M_reg=self.gpr_seq_len)

            # Q projection with bias (BF16 GEMM, runtime row count)
            total_flops += self.matmat_mul_core(M=seq_len, K=self.vector_length, N=self.head_dim,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                C_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_BIAS + layer_off, bias_mode="broadcast_N",
                gpr_M_reg=self.gpr_seq_len)
            # K projection with bias
            total_flops += self.matmat_mul_core(M=seq_len, K=self.vector_length, N=self.head_dim,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                C_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_BIAS + layer_off, bias_mode="broadcast_N",
                gpr_M_reg=self.gpr_seq_len)
            # V projection with bias
            total_flops += self.matmat_mul_core(M=seq_len, K=self.vector_length, N=self.head_dim,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_V_PROJ_DRAM,
                C_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_BIAS + layer_off, bias_mode="broadcast_N",
                gpr_M_reg=self.gpr_seq_len)

            # NO RoPE — GPT-2 uses learned positional embeddings (added host-side)

            # Per-head scatter + flash attention call site (MHA: group_size=1, 12 heads).
            # Q/K/V are each (seq_len, 768) with contiguous 64-dim head slices.
            for kv_h in range(nkvh):
                k_cache_base = (self.LAYER0_K_DRAM_CACHE
                                + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)
                v_cache_base = (self.LAYER0_V_DRAM
                                + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)

                # Scatter K_h (contiguous 64-dim slice) → KV cache + FLASH_K
                self._emit_pbi_scatter_per_token(
                    read_base=self.LAYER0_K_DRAM + kv_h * ahd * bpe,
                    read_stride_bytes=hd * bpe,
                    write_specs=[(k_cache_base, ahd * bpe),
                                 (self.LAYER0_FLASH_K_DRAM, ahd * bpe)],
                    sram_byte_addr=0x10000,
                    element_count=ahd,
                    gpr_seq_len=self.gpr_seq_len,
                    template_seq_len=seq_len,
                )
                # Scatter V_h (contiguous 64-dim slice) → KV cache + FLASH_V
                self._emit_pbi_scatter_per_token(
                    read_base=self.LAYER0_V_PROJ_DRAM + kv_h * ahd * bpe,
                    read_stride_bytes=hd * bpe,
                    write_specs=[(v_cache_base, ahd * bpe),
                                 (self.LAYER0_FLASH_V_DRAM, ahd * bpe)],
                    sram_byte_addr=0x20000,
                    element_count=ahd,
                    gpr_seq_len=self.gpr_seq_len,
                    template_seq_len=seq_len,
                )
                # Scatter Q_h (contiguous 64-dim slice) → FLASH_Q
                self._emit_pbi_scatter_per_token(
                    read_base=self.LAYER0_Q_DRAM + kv_h * ahd * bpe,
                    read_stride_bytes=hd * bpe,
                    write_specs=[(self.LAYER0_FLASH_Q_DRAM, ahd * bpe)],
                    sram_byte_addr=0x30000,
                    element_count=ahd,
                    gpr_seq_len=self.gpr_seq_len,
                    template_seq_len=seq_len,
                )

                # Call flash attention subroutine (compiled after the layer loop).
                # Pad so capture_count is even; return address = capture_count + 2
                # (ADD_SET + JUMP_ABS), which is then also even = 512-bit aligned.
                self.pad_capture_to_64b_boundary()
                return_word_addr = ue_35bit_addr_shifter(
                    program_dram_base + (self.capture_count + 2) * INSTRUCTION_SIZE_BYTES)
                self.generate_instruction_add_set(gpr_ret_id, return_word_addr)
                call_site_jump_capture_indices.append(self.capture_count)
                self.generate_instruction_jump_abs(target_instruction_word_addr=0)

                # Assemble output: head_h output rows → FLASH_OUTPUT at head's 64-dim slot
                self._emit_pbi_scatter_per_token(
                    read_base=self.LAYER0_FLASH_OUT_HEAD_DRAM,
                    read_stride_bytes=ahd * bpe,
                    write_specs=[(self.LAYER0_FLASH_OUTPUT_DRAM + kv_h * ahd * bpe, hd * bpe)],
                    sram_byte_addr=0x40000,
                    element_count=ahd,
                    gpr_seq_len=self.gpr_seq_len,
                    template_seq_len=seq_len,
                )

            # Output projection with bias
            total_flops += self.matmat_mul_core(M=seq_len, K=self.head_dim, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                C_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_BIAS + layer_off, bias_mode="broadcast_N",
                gpr_M_reg=self.gpr_seq_len)

            # Residual add: input + attn output — per-row PBI loop (gpr_seq_len trips)
            self.eltwise_core_dram(
                M=seq_len, N=self.vector_length,
                dram_a=self.LAYER0_INPUT_DRAM,
                dram_b=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                dram_out=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                mode=UE_MODE.ELTWISE_ADD,
                gpr_M_reg=self.gpr_seq_len,
            )

            # LayerNorm 2 (gamma + beta)
            total_flops += self.layer_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                              GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_LN2_GAMMA + layer_off,
                              BETA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_LN2_BETA + layer_off,
                              gpr_M_reg=self.gpr_seq_len)

            # MLP c_fc with GELU and bias (no SwiGLU gate)
            total_flops += self.matmat_mul_core(M=seq_len, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_FC + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_FC_DRAM,
                gelu_enable=True, C_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_FC_BIAS + layer_off, bias_mode="broadcast_N",
                gpr_M_reg=self.gpr_seq_len)

            # MLP c_proj with bias
            total_flops += self.matmat_mul_core(M=seq_len, K=self.mlp_elements, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_MLP_FC_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_PROJ + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_PROJ_DRAM,
                C_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_PROJ_BIAS + layer_off, bias_mode="broadcast_N",
                gpr_M_reg=self.gpr_seq_len)

            # Residual add: post_attn_residual + mlp output → layer output
            self.eltwise_core_dram(
                M=seq_len, N=self.vector_length,
                dram_a=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                dram_b=self.LAYER0_MLP_PROJ_DRAM,
                dram_out=self.LAYER0_OUTPUT_DRAM,
                mode=UE_MODE.ELTWISE_ADD,
                gpr_M_reg=self.gpr_seq_len,
            )

        # HALT ends the normal execution path; the flash_attention subroutine follows and
        # is only reachable via the JUMP_ABS call sites within the layer loop above.
        self.generate_instruction_halt()

        # Compile flash_attention subroutine after the HALT; bucket bodies return via
        # JUMP_REG_ABS(gpr_ret_id), which each call site pre-loaded with its return address.
        flash_sub_start_inst_dram_addr, flash_flops = self.flash_attention_core(
            head_dim=ahd,
            seq_len=aligned_seq_len,
            Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
            K_DRAM_ADDR=self.LAYER0_FLASH_K_DRAM,
            V_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
            OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUT_HEAD_DRAM,
            SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
            IDENTITY_DRAM_ADDR=self.IDENTITY_DRAM_ADDR,
            BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_DRAM,
            ATTN_P_DRAM_ADDR=self.LAYER0_FLASH_ATTN_P_DRAM,
            gpr_bucket_idx=self.gpr_bucket_idx,
            num_buckets=num_bucket,
            gpr_ret_id=gpr_ret_id,
        )
        total_flops += flash_flops[num_bucket - 1] * layer_size * nkvh

        # Patch all call-site JUMP_ABS placeholders to point at the flash subroutine.
        for jump_idx in call_site_jump_capture_indices:
            self._patch_jump_immediate(
                jump_idx, ue_35bit_addr_shifter(flash_sub_start_inst_dram_addr))

        self.release_isa_reg()  # gpr_ret_id
        prefill_program_size = (self.capture_count - count_at_start) * INSTRUCTION_SIZE_BYTES
        _SILENT_MODE = False
        return {"size_bytes": prefill_program_size, "flops": total_flops}

    def _compile_decoder_program(self, layer_size: int) -> dict:
        """Compile decoder into the active capture session.
        Returns dict with ``program_size_bytes`` and ``total_flops``.

        M=1 throughout. GPT-2 keeps its BF16 weight scheme, so per-layer
        projections use matmat_mul_core(M=1). The decode-position KV cache writes use the
        host-primed V_CACHE_SIZE_REG (llama3.2-style, core_changes.md §3e); attention runs through
        the shared decoder_group_attention_core subroutine (§7) with the per-head K/V
        history gathered into the fixed FLASH_K / FLASH_V operand buffers (§7c).
        """
        if not getattr(self, "is_capture_on", False):
            raise RuntimeError("_compile_decoder_program() requires an active capture session")
        count_at_start = self.capture_count
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        total_flops = 0
        program_dram_base = self.get_program_dram_addr()
        gpr_ret_id = self.alloc_isa_reg()
        call_site_jump_capture_indices: list[int] = []

        global _SILENT_MODE
        _SILENT_MODE = True
        ahd = self.actual_head_dim   # 64
        nkvh = self.num_kv_heads     # 12
        bpe = self.bytes_per_element
        hd = self.head_dim           # 768

        for layer_idx in range(layer_size):
            layer_off = layer_idx * LAYER_WEIGHT_SIZE
            if layer_idx != 0:
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, sram_address=0x10000, element_size=self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_INPUT_DRAM, element_size=self.vector_length)

            # LayerNorm 1
            total_flops += self.layer_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                          OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                          GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_LN1_GAMMA + layer_off,
                          BETA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_LN1_BETA + layer_off)

            # Q projection with bias
            total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=self.head_dim,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                C_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_BIAS + layer_off, bias_mode="broadcast_N")
            # K projection with bias
            total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=self.head_dim,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                C_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_BIAS + layer_off, bias_mode="broadcast_N")
            # V projection with bias
            total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=self.head_dim,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_V_PROJ_DRAM,
                C_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_BIAS + layer_off, bias_mode="broadcast_N")

            # NO RoPE — GPT-2 uses learned positional embeddings

            # Per-head: K/V → cache at decode position, gather history into the fixed
            # FLASH operand buffers, Q → FLASH_Q base, then jump to the shared
            # decoder attention subroutine (core_changes.md §7c).
            for kv_h in range(nkvh):
                k_cache_base = (self.LAYER0_K_DRAM_CACHE
                                + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)
                v_cache_base = (self.LAYER0_V_DRAM
                                + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)

                # Scatter K_h (contiguous 64-dim) → KV cache at decode position
                self.accelerator_memory_to_sram(
                    self.LAYER0_K_DRAM + kv_h * ahd * bpe, 0x10000, ahd)
                self.generate_instruction_add_imm(
                    self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter(k_cache_base), self.TMP_REG)
                self.sram_to_accelerator_memory(0x10000, 0, ahd)
                self.overwrite_instruction_with_general_register(self.TMP_REG)

                # Copy valid K history → FLASH_K; loop count = gpr_bucket_idx so only
                # aligned_seq_len tokens are copied, not the full MAX_CONTEXT_SIZE.
                self._emit_pbi_scatter_per_token(
                    read_base=k_cache_base,
                    read_stride_bytes=UE_VECTOR_SIZE * ahd * bpe,
                    write_specs=[(self.LAYER0_FLASH_K_DRAM, UE_VECTOR_SIZE * ahd * bpe)],
                    sram_byte_addr=0,
                    element_count=UE_VECTOR_SIZE * ahd,
                    gpr_seq_len=self.gpr_bucket_idx,
                )

                # Scatter V_h (contiguous 64-dim) → V cache at decode position
                self.accelerator_memory_to_sram(
                    self.LAYER0_V_PROJ_DRAM + kv_h * ahd * bpe, 0x20000, ahd)
                self.generate_instruction_add_imm(
                    self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter(v_cache_base), self.TMP_REG)
                self.sram_to_accelerator_memory(0x20000, 0, ahd)
                self.overwrite_instruction_with_general_register(self.TMP_REG)

                # Copy valid V history → FLASH_V; same dynamic size.
                self._emit_pbi_scatter_per_token(
                    read_base=v_cache_base,
                    read_stride_bytes=UE_VECTOR_SIZE * ahd * bpe,
                    write_specs=[(self.LAYER0_FLASH_V_DRAM, UE_VECTOR_SIZE * ahd * bpe)],
                    sram_byte_addr=0,
                    element_count=UE_VECTOR_SIZE * ahd,
                    gpr_seq_len=self.gpr_bucket_idx,
                )

                # Q_h (contiguous 64-dim) → FLASH_Q base
                self.accelerator_memory_to_sram(
                    self.LAYER0_Q_DRAM + kv_h * ahd * bpe, 0x30000, ahd)
                self.sram_to_accelerator_memory(0x30000, self.LAYER0_FLASH_Q_DRAM, ahd)

                # Call decoder attention subroutine (compiled after the HALT below).
                self.pad_capture_to_64b_boundary()
                return_word_addr = ue_35bit_addr_shifter(
                    program_dram_base + (self.capture_count + 2) * INSTRUCTION_SIZE_BYTES)
                self.generate_instruction_add_set(gpr_ret_id, return_word_addr)
                call_site_jump_capture_indices.append(self.capture_count)
                self.generate_instruction_jump_abs(target_instruction_word_addr=0)

                # Copy per-head output to its slot in FLASH_OUTPUT_DRAM.
                self.accelerator_memory_to_sram(
                    self.LAYER0_FLASH_OUT_HEAD_DRAM, 0x40000, self.group_size * ahd)
                self.sram_to_accelerator_memory(
                    0x40000, self.LAYER0_FLASH_OUTPUT_DRAM + kv_h * self.group_size * ahd * bpe, self.group_size * ahd)

            # Output projection with bias
            total_flops += self.matmat_mul_core(M=1, K=self.head_dim, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                C_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_BIAS + layer_off, bias_mode="broadcast_N")

            # Residual add
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_INPUT_DRAM, sram_address=0x10000, element_size=self.vector_length)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM, sram_address=0x90000, element_size=self.vector_length)
            self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, element_size=self.vector_length)

            # LayerNorm 2
            total_flops += self.layer_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                          OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                          GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_LN2_GAMMA + layer_off,
                          BETA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_LN2_BETA + layer_off)

            # MLP c_fc + GELU with bias
            total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_FC + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_FC_DRAM,
                gelu_enable=True, C_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_FC_BIAS + layer_off, bias_mode="broadcast_N")

            # MLP c_proj with bias
            total_flops += self.matmat_mul_core(M=1, K=self.mlp_elements, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_MLP_FC_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_PROJ + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_PROJ_DRAM,
                C_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_PROJ_BIAS + layer_off, bias_mode="broadcast_N")

            # Residual add
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, sram_address=0x10000, element_size=self.vector_length)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_PROJ_DRAM, sram_address=0x90000, element_size=self.vector_length)
            self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=self.vector_length)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, element_size=self.vector_length)

        if layer_size == self.LAYER_SIZE:
            # Final LayerNorm with gamma + beta
            total_flops += self.layer_norm_core_dram(M=1, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_OUTPUT_DRAM,
                OUTPUT_DRAM_ADDR=self.OUTPUT_NORM_DRAM,
                GAMMA_DRAM_ADDR=self.DRAM_ADDR_OUTPUT_LN_GAMMA,
                BETA_DRAM_ADDR=self.DRAM_ADDR_OUTPUT_LN_BETA)
            # LM head (no bias). Stays matmat_mul_core (core_changes.md §3h exception);
            # writeback ON — the host sampling path reads LOGITS_DRAM every step, and the
            # greedy path re-argmaxes from it when the HW argmax lands on a padded index.
            total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=self.EMBEDDING_ELEMENTS,
                A_DRAM_ADDR=self.OUTPUT_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD, OUTPUT_DRAM_ADDR=self.LOGITS_DRAM)

        self.generate_instruction_halt()
        self.pad_capture_to_64b_boundary()

        # Compile decoder_group_attention_core once as a subroutine after HALT.
        num_buckets = (self.MAX_CONTEXT_SIZE + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
        dec_sub_start_addr, dec_attn_flops = self.decoder_group_attention_core(
            group_size=self.group_size,
            head_dim=ahd,
            seq_len=self.MAX_CONTEXT_SIZE,
            gpr_bucket_idx=self.gpr_bucket_idx,
            num_buckets=num_buckets,
            Q_DRAM_ADDR=self.LAYER0_FLASH_Q_DRAM,
            K_DRAM_ADDR=self.LAYER0_FLASH_K_DRAM,
            V_DRAM_ADDR=self.LAYER0_FLASH_V_DRAM,
            OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUT_HEAD_DRAM,
            IDENTITY_DRAM_ADDR=self.IDENTITY_DRAM_ADDR,
            SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
            BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_DRAM,
            gpr_ret_id=gpr_ret_id,
        )
        total_flops += dec_attn_flops[-1] * layer_size * nkvh

        for jump_idx in call_site_jump_capture_indices:
            self._patch_jump_immediate(
                jump_idx, ue_35bit_addr_shifter(dec_sub_start_addr))

        self.release_isa_reg()  # gpr_ret_id
        decoder_program_size = (self.capture_count - count_at_start) * INSTRUCTION_SIZE_BYTES
        _SILENT_MODE = False
        return {"program_size_bytes": decoder_program_size, "total_flops": total_flops}

    def compile_gpt2(self, layer_size: int | None = None) -> None:
        """Compile prefill + decoder into a single combined instruction image.

        Layout in program DRAM:  [prefill][decoder]

        Prefill is compiled with a fixed template — all runtime loop counts are
        driven by GPRs primed by the preamble, so the same bin works for any
        seq_len up to PREFILL_MAX_SEQ_LEN. If both bin and meta already exist,
        this is a no-op (delete the bin to force recompile, e.g. after any
        tensor_init layout change — baked addresses must match).

        Writes:
          - paths.instruction_bin  : combined raw instruction stream
          - paths.instruction_meta : per-stage start addresses, sizes, FLOPs
        """
        if layer_size is None:
            layer_size = self.LAYER_SIZE
        paths_cfg = self._cfg.get("paths", {})
        instruction_bin_path = os.path.join(self.script_dir, paths_cfg.get("instruction_bin", "gpt2_bin/gpt2_instruction.bin"))
        instruction_meta_path = os.path.join(self.script_dir, paths_cfg.get("instruction_meta", "gpt2_bin/gpt2_instruction.json"))
        if os.path.exists(instruction_bin_path) and os.path.exists(instruction_meta_path):
            with open(instruction_meta_path, "r") as f:
                existing_meta = json.load(f)
            if existing_meta.get("gpt2_instruction_cache_version") == GPT2_INSTRUCTION_CACHE_VERSION:
                print(f"Reusing existing instruction image at {instruction_bin_path}")
                print(f"  delete {instruction_bin_path} to force recompile.")
                return
            print("Existing GPT-2 instruction image is stale; rebuilding.")

        # Compile the prefill template at PREFILL_CONTEXT_SIZE (the max prompt length).
        template_seq_len = self.PREFILL_MAX_SEQ_LEN

        self.clear_inst_id()
        self.start_capture()

        print(f"Compiling prefill (template_seq_len={template_seq_len})...")
        t0 = time.perf_counter()
        prefill_prog = self._compile_prefill_program(template_seq_len=template_seq_len, layer_size=layer_size)
        print(f"  prefill compiled: {prefill_prog['size_bytes']} bytes, {time.perf_counter() - t0:.1f}s")

        print("Compiling decoder...")
        t0 = time.perf_counter()
        decoder_prog = self._compile_decoder_program(layer_size=layer_size)
        print(f"  decoder compiled: {decoder_prog['program_size_bytes']} bytes, {time.perf_counter() - t0:.1f}s")

        self.stop_capture()

        os.makedirs(os.path.dirname(instruction_bin_path), exist_ok=True)
        instruction_bytes = bytearray()
        for inst in self.capture_buffer:
            instruction_bytes.extend(inst.get_bytes())
        with open(instruction_bin_path, "wb") as f:
            f.write(instruction_bytes)
        self.clear_capture_buffer()

        instruction_base_addr = self.get_program_dram_addr()
        prefill_program_addr = instruction_base_addr
        decoder_program_addr = instruction_base_addr + prefill_prog["size_bytes"]

        metadata = {
            "gpt2_instruction_cache_version": GPT2_INSTRUCTION_CACHE_VERSION,
            "instruction_bin": os.path.relpath(instruction_bin_path, self.script_dir),
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
        with open(instruction_meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Combined instruction image written to {instruction_bin_path} ({len(instruction_bytes)} bytes)")
        print(f"Metadata written to {instruction_meta_path}")

    @staticmethod
    def _sample_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0,
                      generated_ids: list[int] | None = None, repetition_penalty: float = 1.0,
                      bad_token_ids: set[int] | None = None, fallback_token_id: int | None = None) -> int:
        """Sample a token from logits with temperature, top-k, top-p, and repetition penalty.

        Args:
            logits: 1-D tensor of raw logits over the real vocabulary.
            temperature: Scales logits before softmax. 0 = greedy argmax.
            top_k: If > 0, keep only the top-k highest-probability tokens.
            top_p: If < 1.0, keep the smallest set of tokens with cumulative probability >= top_p.
            generated_ids: List of already-generated token IDs for repetition penalty.
            repetition_penalty: Penalize already-generated tokens. 1.0 = no penalty.
            bad_token_ids: Token IDs to suppress during sampling.
            fallback_token_id: Token returned if the sampled distribution is degenerate.
        """
        if temperature == 0:
            return logits.argmax().item()

        logits = logits.float()
        finite = torch.isfinite(logits)
        if not finite.any():
            return fallback_token_id if fallback_token_id is not None else 0
        logits = logits.masked_fill(~finite, float("-inf"))
        if bad_token_ids:
            bad = torch.tensor([tok for tok in bad_token_ids if 0 <= tok < logits.size(0)], dtype=torch.long)
            if bad.numel() > 0:
                logits[bad] = float("-inf")

        # Repetition penalty: divide positive logits / multiply negative logits for seen tokens
        if repetition_penalty != 1.0 and generated_ids:
            seen = torch.tensor(list(set(generated_ids)), dtype=torch.long)
            seen = seen[seen < logits.size(0)]
            if seen.numel() > 0:
                orig = logits[seen]
                logits[seen] = torch.where(orig > 0, orig / repetition_penalty, orig * repetition_penalty)

        logits = logits / temperature

        # Top-k: zero out everything outside the top k
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            threshold = logits.topk(top_k).values[-1]
            logits[logits < threshold] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        prob_sum = probs.sum()
        if prob_sum <= 0:
            return fallback_token_id if fallback_token_id is not None else logits.argmax().item()
        probs = probs / prob_sum

        # Top-p (nucleus): keep smallest set with cumulative prob >= top_p
        if top_p < 1.0:
            sorted_probs, sorted_indices = probs.sort(descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)
            # Remove tokens with cumulative probability above top_p (keep at least 1)
            mask = cumsum - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_sum = sorted_probs.sum()
            if sorted_sum <= 0:
                return fallback_token_id if fallback_token_id is not None else logits.argmax().item()
            sorted_probs /= sorted_sum
            # Sample from the filtered sorted distribution, then map back
            idx = torch.multinomial(sorted_probs, num_samples=1).item()
            return sorted_indices[idx].item()

        return torch.multinomial(probs, num_samples=1).item()

    def run_gpt2(self, temperature: float = 0.9, top_k: int = 50, top_p: float = 0.9,
                 repetition_penalty: float = 1.25, max_new_tokens: int | None = None) -> None:
        """Load the unified instruction image and run prefill + decoder loop.

        Primes GPRs via a small captured preamble that jumps into the cached
        prefill or decoder program at runtime (core_changes.md §3f).

        Args:
            temperature: Logit temperature. 0 = greedy (HW argmax, no logit readback
                unless the argmax lands on a padded vocab index).
            top_k: Keep only top-k tokens before sampling. 0 = disabled.
            top_p: Nucleus sampling threshold. 1.0 = disabled.
            repetition_penalty: Penalize repeated tokens host-side. 1.0 = no penalty.
            max_new_tokens: Optional decode limit for short-context speed runs.
        """
        paths_cfg = self._cfg.get("paths", {})
        meta_path = os.path.join(self.script_dir, paths_cfg.get("instruction_meta", "gpt2_bin/gpt2_instruction.json"))
        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.load_program_instructions_from_file(os.path.join(self.script_dir, meta["instruction_bin"]))
        preamble_addr = self.get_program_dram_addr()

        prefill_program_addr = int(meta["prefill_program_start_addr"], 16)
        decoder_program_addr = int(meta["decoder_program_start_addr"], 16)
        template_seq_len = int(meta["prefill_template_seq_len"])
        flops_prefill_template = meta["prefill_template_flops"]
        decoder_flops_per_token = meta["decoder_total_flops"]
        _max_gpr_bucket = (self.MAX_CONTEXT_SIZE + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
        _kv_stride = self.actual_head_dim * self.bytes_per_element  # 128 bytes/position

        prefill_seq = self.prefill_seq
        if prefill_seq is None:
            prefill_seq = tuple(self._cfg["default_prefill_tokens"])
        if len(prefill_seq) < 2:
            raise ValueError("Prefill sequence must have at least 2 tokens.")
        if len(prefill_seq) > self.PREFILL_MAX_SEQ_LEN + 1:
            raise ValueError(
                f"Prefill sequence has {len(prefill_seq)} tokens, but this bin supports "
                f"at most {self.PREFILL_MAX_SEQ_LEN + 1} including the first decode token. "
                "Increase model.prefill_max_seq_len and rebuild the instruction bin."
            )
        prefill_seq = prefill_seq[:-1]  # last token starts the decoder
        prefill_seq_len = len(prefill_seq)
        self.seq_len = prefill_seq_len

        q_seq_len = prefill_seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        bucket_idx = aligned_seq_len // UE_VECTOR_SIZE
        flops_prefill = flops_prefill_template * prefill_seq_len // max(template_seq_len, 1)

        # Prefill preamble: prime gpr_seq_len + gpr_bucket_idx, then jump into cached prefill.
        self.clear_inst_id()
        self.start_capture()
        self.generate_instruction_add_set(self.gpr_seq_len, prefill_seq_len)
        self.generate_instruction_add_set(self.gpr_bucket_idx, bucket_idx)
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(prefill_program_addr))
        self.stop_capture()
        self.write_captured_instructions_to_dram(preamble_addr)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        self.clear_capture_buffer()

        # Host-side: token + positional embeddings (positions 0..N-1)
        embedding_tensor = self.get_embedding_for_tokens(prefill_seq, start_pos=0)
        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
        # Causal mask: group_size=1 so standard lower-triangular, sized to the
        # runtime-selected bucket (aligned_seq_len) so row stride matches the body.
        bias_one_group = torch.full((aligned_seq_len, aligned_seq_len), float("-inf"), dtype=torch.bfloat16)
        rows = torch.arange(aligned_seq_len).unsqueeze(1)
        cols = torch.arange(aligned_seq_len).unsqueeze(0)
        valid_mask = cols <= rows
        bias_one_group.masked_fill_(valid_mask, 0.0)
        bias_one_group[:, q_seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_one_group)

        print(f"\n--- Starting prefill (seq_len={prefill_seq_len}) ---")
        print(f"Prompt ({len(self.prefill_seq)}) tokens: {self.prefill_seq}")
        timer = time.perf_counter()
        self.program_execute(preamble_addr, flops=flops_prefill)
        latency_prefill = time.perf_counter() - timer
        print(f"Prefill done in {latency_prefill:.2f}s\n")

        print("--- Starting decoder ---")
        timer = time.perf_counter()
        token_id = self.prefill_seq[-1]
        _stop_tokens = {self._end_of_turn_token_id}  # 50256
        use_sampling = temperature > 0
        generated_ids: list[int] = list(self.prefill_seq)
        decoded_chars: list[str] = []
        bad_token_ids = {self._end_of_turn_token_id}
        bad_token_ids.update(range(50257, self.EMBEDDING_ELEMENTS))
        bad_token_ids.update(self.tokenizer.encode("!"))
        first_decode_speed = None
        peak_decode_speed = None
        global _SILENT_MODE

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
            if max_new_tokens is not None and (self.seq_len - prefill_seq_len) >= max_new_tokens:
                break
            _SILENT_MODE = True
            self.seq_len += 1
            aligned_seq_len = ((self.seq_len + 63) // 64) * 64
            bucket_idx = min((self.seq_len + 63) // 64, _max_gpr_bucket)
            decode_pos = self.seq_len - 1

            # Host-side: token + positional embedding for the current decode position
            embedding_tensor = self.get_embedding_for_tokens([token_id], start_pos=decode_pos)
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
            bias_host = torch.full((1, aligned_seq_len), -1e36, dtype=torch.bfloat16)
            bias_host[0, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_host)

            # Decoder preamble: prime bucket + KV-cache position GPRs, then jump into
            # the cached decoder. The preamble slot is one-shot and reused every step.
            self.clear_inst_id()
            self.start_capture()
            self.generate_instruction_add_set(self.gpr_bucket_idx, bucket_idx)
            self.generate_instruction_add_set(self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter(decode_pos * _kv_stride))
            self.generate_instruction_jump_abs(ue_35bit_addr_shifter(decoder_program_addr))
            self.stop_capture()
            self.write_captured_instructions_to_dram(preamble_addr)
            self.clear_capture_buffer()

            step_timer = time.perf_counter()
            self.program_execute(preamble_addr, flops=decoder_flops_per_token)
            step_latency = time.perf_counter() - step_timer
            step_speed = 1.0 / step_latency if step_latency > 0 else 0.0
            if first_decode_speed is None:
                first_decode_speed = step_speed
                peak_decode_speed = step_speed

            if use_sampling:
                # Read logits from DRAM and sample host-side
                logits = self.dma_from_accelerator_memory(self.LOGITS_DRAM, 50257)
                fallback_token_id = self.get_arg_max_index(rank=1)
                if fallback_token_id >= 50257:
                    fallback_token_id = None
                token_id = self._sample_token(logits, temperature=temperature, top_k=top_k, top_p=top_p,
                                              generated_ids=generated_ids, repetition_penalty=repetition_penalty,
                                              bad_token_ids=bad_token_ids, fallback_token_id=fallback_token_id)
                generated_ids.append(token_id)
            else:
                token_id = self.get_arg_max_index(rank=1)
                # If argmax picks a padded token (>= real vocab), read logits and re-argmax
                if token_id >= 50257:
                    logits = self.dma_from_accelerator_memory(self.LOGITS_DRAM, 50257)
                    token_id = logits.argmax().item()

            token_char = self.tokenizer.decode([token_id])
            _SILENT_MODE = False
            if token_id in _stop_tokens:
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
        avg_decode_speed = tokens_decoded / latency_decoder if latency_decoder > 0 else 0.0
        print(f"\nDecoder done in {latency_prefill + latency_decoder:.2f}s, "
              f"speed: {avg_decode_speed:.2f} tokens/s, "
              f"total {self.seq_len} tokens.")
        print(f"Decode speed (greedy): peak {peak_decode_speed or 0.0:.2f} tok/s "
              f"(1st token {first_decode_speed or 0.0:.2f}), average {avg_decode_speed:.2f} tok/s.")

        return {
            "prefill_tokens": prefill_seq_len,
            "decoded_text": "".join(decoded_chars),
            "decoded_tokens": tokens_decoded,
            "prefill_speed_tok_s": round(prefill_seq_len / latency_prefill, 2) if latency_prefill > 0 else None,
            "decode_speed_tok_s": round(avg_decode_speed, 2),
            "prefill_size_kb": round(meta["prefill_program_size"] / 1024, 1) if "prefill_program_size" in meta else None,
            "decoder_size_kb": round(meta["decoder_program_size"] / 1024, 1) if "decoder_program_size" in meta else None,
        }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPT-2 Base (124M) prefill + decode on accelerator.")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt (GPT-2 is a base model; no chat template applied)")
    parser.add_argument("--local-weights", action="store_true", help="Use gpt2_bin/full_model_weights.bin instead of generated weights_gpt2_hf.bin")
    parser.add_argument('--dev', type=str, default='xdma0',
                        help='DMA device name (e.g., xdma0, xdma1). Default: xdma0')
    parser.add_argument('--cycle', type=float, default=1/0.17,
                        help='Clock cycle time in nanoseconds (default: ~5.88ns)')
    parser.add_argument('--temperature', type=float, default=0.9,
                        help='Sampling temperature. 0 = greedy argmax (default: 0.9)')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k sampling. 0 = disabled (default: 50)')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='Nucleus (top-p) sampling. 1.0 = disabled (default: 0.9)')
    parser.add_argument('--repetition-penalty', type=float, default=1.25,
                        help='Repetition penalty for seen tokens. 1.0 = disabled (default: 1.25)')
    parser.add_argument('--max-new-tokens', type=int, default=None,
                        help='Optional decode token limit; useful for short-context speed measurement')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = _load_config(script_dir)
    if args.local_weights:
        weights_bin_rel = cfg["paths"]["local_weights_bin"]
    else:
        weights_bin_rel = cfg["paths"]["weights_bin"]
        weights_bin_full = os.path.join(script_dir, weights_bin_rel)
        if not os.path.exists(weights_bin_full):
            weight_bin_generate(script_dir=script_dir, output_path=weights_bin_full)

    set_dma_device(args.dev)
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle
    print(f"Using DMA device: {args.dev}, CLOCK_CYCLE_TIME_NS={user_dma_core.CLOCK_CYCLE_TIME_NS}")

    ue = GPT2_UnifiedEngine(script_dir=script_dir, weights_bin=weights_bin_rel)

    if args.prompt is not None:
        # GPT-2 is a base model — encode prompt directly, no chat template
        prefill_seq = tuple(ue.tokenizer.encode(args.prompt))
        print(f"Prefill from prompt ({len(prefill_seq)} tokens): {args.prompt!r}")
        print(f"Sequence ids: {prefill_seq}")
    else:
        prefill_seq = tuple(cfg["default_prefill_tokens"])
        default_prompt_text = ue.tokenizer.decode(list(prefill_seq))
        print(f"Default prompt ({len(prefill_seq)} tokens): {default_prompt_text!r}")
        print(f"Sequence ids: {prefill_seq}")

    max_prefill = ue.PREFILL_MAX_SEQ_LEN
    if len(prefill_seq) < 2 or len(prefill_seq) > max_prefill + 1:
        print(f"WARNING: prompt length {len(prefill_seq)} out of range [2, {max_prefill + 1}], falling back to default.")
        prefill_seq = tuple(cfg["default_prefill_tokens"])

    ue.prefill_seq = prefill_seq

    print("\n--- Compiling ---")
    timer = time.perf_counter()
    ue.compile_gpt2()
    print(f"Compile done in {time.perf_counter() - timer:.2f}s")

    print("\n--- Running ---")
    run_result = ue.run_gpt2(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                             repetition_penalty=args.repetition_penalty, max_new_tokens=args.max_new_tokens)
    ue.clear_dram()
    print("GPT-2 test ends.")
    print(f"TEST_RESULT: {json.dumps(run_result)}")

if __name__ == "__main__":
    main()
