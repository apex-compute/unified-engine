#!/usr/bin/env python3
"""
GPT-2 Base (124M) inference on accelerator: prefill + decode.

  - Config from gpt2_config.json; weights from a single bin (see below).
  - Prefill: compiled each run. Decoder: if gpt2_bin/decoder_program.bin and
    gpt2_bin/decoder_program.json exist, skip decoder compile and load
    program sizes from meta; otherwise compile and write the bin + meta.
  - Run prefill then decode loop.

Architecture notes (vs LLaMA/Gemma3):
  - 12 layers, 12 heads (MHA, group_size=1), actual_head_dim=64.
  - LayerNorm with bias (not RMSNorm). eps=1e-5.
  - Learned positional embeddings (not RoPE). Added host-side.
  - Fused QKV projection (c_attn) split into Q, K, V during weight gen.
  - GELU activation in MLP (not SwiGLU). No gate projection.
  - Bias in all linear layers (Q, K, V, output, c_fc, c_proj).
  - LM head weight tied to input embedding.
  - HuggingFace GPT-2 uses Conv1D: weights are (in, out), need transpose.
  - BF16 weights (no quantization) — 124M model is small enough for full precision.

Weights:
  - Default: gpt2_bin/weights_gpt2_hf.bin (generated from HF model if missing).
  - --local-weights: use gpt2_bin/full_model_weights.bin instead.

Usage:
  python models/gpt2/gpt2_test.py --prompt "The scientists at MIT announced today that they have discovered "
"""

import json
import math
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import time

# This file's folder; user_dma_core.py is two folders up (repo root); that directory is added to sys.path.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

import user_dma_core
from user_dma_core import DMA_DEVICE_H2C, UE_FMAX_CONTEXT_SIZE, UE_VECTOR_SIZE, UE_ARGMAX_INDEX, URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS, set_dma_device
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
    # Hardware argmax over 50304-dim may pick a padded index; handled in run_decoder.
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
        snapshot_download(repo_id=hf_repo, local_dir=model_dir, local_dir_use_symlinks=False)
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
    """UnifiedEngine for GPT-2 Base (124M): loads config + weight bin, compile_prefill/compile_decoder, run_prefill/run_decoder.

    Key architectural differences from LLaMA/Gemma3:
      - LayerNorm with bias (not RMSNorm). Uses layer_norm_core_dram with GAMMA + BETA.
      - Learned positional embeddings added host-side (not RoPE on FPGA).
      - MHA (group_size=1): each of 12 Q heads maps 1:1 to a KV head, no duplication.
      - GELU activation in MLP (gelu_enable=True on c_fc matmul). No gate projection.
      - Bias in all linear layers (Q, K, V, output, c_fc, c_proj).
      - No Q/K norm, no post-attention norm, no post-FFN norm.
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
        self.LAYER_SIZE = fi["num_layers"]             # 12
        self.EMBEDDING_ELEMENTS = fi["embedding_vocab"]    # 50257
        self._isa_reg_counter = 1
        fixed = self._cfg.get("fixed_isa_regs", {})
        self.V_CACHE_SIZE_REG = fixed["V_CACHE_SIZE_REG"]
        self.TMP_REG = fixed["TMP_REG"]
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

    def reset_isa_reg_counter(self) -> None:
        """Reset the ISA register allocation counter to 1 (register 0 is hard-wired zero)."""
        self._isa_reg_counter = 1

    def alloc_isa_reg(self, reset: bool = False) -> int:
        if reset:
            self._isa_reg_counter = 1
        if self._isa_reg_counter > 15:
            raise ValueError("Exceeded available ISA registers (max 15)")
        reg_idx = self._isa_reg_counter
        self._isa_reg_counter += 1
        return reg_idx

    def overwrite_instruction_with_general_register(self, general_register: int) -> None:
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
        w[0] = ((0 & 0xF) << 0) | \
               ((general_register & 0xF) << 4) | \
               ((0 & 0xF) << 8) | \
               ((0 & 0xF) << 12)
        w[7] = (w[7] & 0x1FFFFFFF) | ((user_dma_core.INSTRUCTION_REG_REWRITE & 0x7) << 29)

    def isa_add_set_core(self, dst_reg_idx: int, immediate_value: int, timeout_s: float = 10.0) -> None:
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
        params_dram_addr = self._next_params_dram_addr
        self._next_params_dram_addr += size_bytes
        return params_dram_addr

    def clear_inst_id(self) -> None:
        self._inst_id = 0

    def get_arg_max_index(self) -> int:
        return self.read_reg32(UE_ARGMAX_INDEX)

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

        # Q @ K^T: (1, head_dim) @ (head_dim, seq_len) -> (1, seq_len)
        M = 1
        K = head_dim
        N = seq_len
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


            # SOFTMAX
            max_m_take = min((URAM_FULL_ELEMENTS - UE_VECTOR_SIZE) // N, UE_FMAX_CONTEXT_SIZE)

            for m_take_chunk_idx, m_take_chunk_size in self.chunk_ranges(m_take, max_m_take):
                self.accelerator_memory_to_sram(accelerator_dram_address=SCRATCH_DRAM_PARTIAL_SM + m_take_chunk_idx * N * bytes_per_element,
                                            sram_address=uram_a_start_addr,
                                            element_size=m_take_chunk_size * N)

                for row_idx in range(m_take_chunk_size):
                    self.start_queue_for_bf16_softmax_operation(fmax_context_addr=row_idx + m_take_chunk_idx,
                                                                vector_sram_start_addr=uram_a_start_addr + row_idx * N * bytes_per_element,
                                                                output_sram_wb_addr=uram_a_start_addr + row_idx * N * bytes_per_element,
                                                                N=N)

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

                v_t_sram_start_addr = 0x80000
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

        total_flops = 1 * head_dim
        total_flops += 2 * 1 * head_dim * seq_len
        total_flops += 1 * seq_len * 5
        total_flops += 2 * 1 * seq_len * head_dim
        print(f"Total Theoretical FLOPS: {total_flops}")
        return total_flops

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
          - No V_PROJ_TEMP (no lo|hi interleaving to undo; V contiguous).
          - No MLP_GATE_DRAM or MLP_MULT_DRAM (no SwiGLU gate).
          - group_size=1: flash buffers sized for seq_len * actual_head_dim per head.
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
        # Zero tensor, identity matrix
        zero_add = torch.zeros(seq_len * self.head_dim * self.bytes_per_element, dtype=torch.bfloat16)
        self.ZERO_DRAM_ADDR = self.allocate_tensor_dram(seq_len * self.head_dim * self.bytes_per_element)
        self.dma_to_accelerator_memory(self.ZERO_DRAM_ADDR, zero_add)
        self.IDENTITY_DRAM_ADDR = self.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * self.bytes_per_element)
        self.dma_to_accelerator_memory(self.IDENTITY_DRAM_ADDR, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))
        # Flash attention buffers (group_size=1: Q/K/V sized for seq_len * head_dim)
        self.LAYER0_FLASH_Q_DRAM = self.allocate_tensor_dram(aligned_seq_len * self.actual_head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_K_DRAM = self.allocate_tensor_dram(aligned_seq_len * self.actual_head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_V_DRAM = self.allocate_tensor_dram(aligned_seq_len * self.actual_head_dim * self.bytes_per_element)
        zero_flash = torch.zeros(aligned_seq_len * self.actual_head_dim * self.bytes_per_element, dtype=torch.bfloat16)
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
        self.LAYER0_FLASH_SCRATCH_DRAM = self.allocate_tensor_dram(max(self.actual_head_dim, UE_FMAX_CONTEXT_SIZE) * aligned_seq_len * 2 + self.actual_head_dim * aligned_seq_len * 2)
        self.LAYER0_FLASH_BIAS_DRAM = self.allocate_tensor_dram(aligned_seq_len * aligned_seq_len * self.bytes_per_element)
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

    def program_execute(self, program_start_addr: int = user_dma_core.DRAM_INSTRUCTION_ADDR, timeout: float = 10.0, gflops: float = None) -> None:
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

        GPT-2 differences from LLaMA:
          - LayerNorm with gamma+beta (not RMSNorm).
          - No RoPE steps.
          - MHA scatter: group_size=1, contiguous 64-dim head slices.
          - GELU MLP (no SwiGLU gate × up).
          - Bias on all linear layers via C_DRAM_ADDR.
        """
        if layer_size is None:
            layer_size = self.LAYER_SIZE
        seq_len -= 1
        self.seq_len = seq_len
        q_seq_len = seq_len * self.group_size  # = seq_len for MHA
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64

        global _SILENT_MODE
        _SILENT_MODE = True
        self.start_capture()
        total_flops = 0
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        ahd = self.actual_head_dim   # 64
        nkvh = self.num_kv_heads     # 12
        bpe = self.bytes_per_element

        for layer_idx in range(layer_size):
            layer_off = layer_idx * LAYER_WEIGHT_SIZE
            if layer_idx != 0:
                self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_OUTPUT_DRAM, sram_address=0x10000, element_size=seq_len * self.vector_length)
                self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_INPUT_DRAM, element_size=seq_len * self.vector_length)

            # LayerNorm 1 (gamma + beta)
            total_flops += self.layer_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_INPUT_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM,
                              GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_LN1_GAMMA + layer_off,
                              BETA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_LN1_BETA + layer_off)

            # Q projection with bias
            total_flops += self.matmat_mul_core(M=seq_len, K=self.vector_length, N=self.head_dim,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_PROJ + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_Q_DRAM,
                C_DRAM_ADDR=self.DRAM_ADDR_LAYER0_Q_BIAS + layer_off, bias_mode="broadcast_N")
            # K projection with bias
            total_flops += self.matmat_mul_core(M=seq_len, K=self.vector_length, N=self.head_dim,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_PROJ + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_K_DRAM,
                C_DRAM_ADDR=self.DRAM_ADDR_LAYER0_K_BIAS + layer_off, bias_mode="broadcast_N")
            # V projection with bias
            total_flops += self.matmat_mul_core(M=seq_len, K=self.vector_length, N=self.head_dim,
                A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_V_PROJ_DRAM,
                C_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_BIAS + layer_off, bias_mode="broadcast_N")

            # NO RoPE — GPT-2 uses learned positional embeddings (added host-side)

            # Per-head scatter + flash attention (MHA: group_size=1, 12 heads)
            # Q/K/V are each (seq_len, 768) with contiguous 64-dim head slices
            for kv_h in range(nkvh):
                k_cache_base = (self.LAYER0_K_DRAM_CACHE
                                + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)
                v_cache_base = (self.LAYER0_V_DRAM
                                + layer_idx * self.MAX_CONTEXT_SIZE * self.k_size
                                + kv_h * self.MAX_CONTEXT_SIZE * ahd * bpe)

                # Scatter K_h (contiguous 64-dim slice) → KV cache + FLASH_K
                for t in range(seq_len):
                    k_src = self.LAYER0_K_DRAM + t * self.head_dim * bpe + kv_h * ahd * bpe
                    self.accelerator_memory_to_sram(k_src, 0x10000, ahd)
                    self.sram_to_accelerator_memory(0x10000, k_cache_base + t * ahd * bpe, ahd)
                    self.sram_to_accelerator_memory(0x10000, self.LAYER0_FLASH_K_DRAM + t * ahd * bpe, ahd)

                # Scatter V_h (contiguous 64-dim slice) → KV cache + FLASH_V
                for t in range(seq_len):
                    v_src = self.LAYER0_V_PROJ_DRAM + t * self.head_dim * bpe + kv_h * ahd * bpe
                    self.accelerator_memory_to_sram(v_src, 0x20000, ahd)
                    self.sram_to_accelerator_memory(0x20000, v_cache_base + t * ahd * bpe, ahd)
                    self.sram_to_accelerator_memory(0x20000, self.LAYER0_FLASH_V_DRAM + t * ahd * bpe, ahd)

                # Scatter Q_h (contiguous 64-dim slice) → FLASH_Q
                for t in range(seq_len):
                    q_src = self.LAYER0_Q_DRAM + t * self.head_dim * bpe + kv_h * ahd * bpe
                    self.accelerator_memory_to_sram(q_src, 0x30000, ahd)
                    self.sram_to_accelerator_memory(0x30000, self.LAYER0_FLASH_Q_DRAM + t * ahd * bpe, ahd)

                # Flash attention for this head (head_dim=64)
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

                # Assemble output: head_h output → FLASH_OUTPUT at head's 64-dim slot
                for t in range(seq_len):
                    src = self.LAYER0_FLASH_OUT_HEAD_DRAM + t * ahd * bpe
                    dst = self.LAYER0_FLASH_OUTPUT_DRAM + t * self.head_dim * bpe + kv_h * ahd * bpe
                    self.accelerator_memory_to_sram(src, 0x40000, ahd)
                    self.sram_to_accelerator_memory(0x40000, dst, ahd)

            # Output projection with bias
            total_flops += self.matmat_mul_core(M=seq_len, K=self.head_dim, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM,
                C_DRAM_ADDR=self.DRAM_ADDR_LAYER0_ATTN_PROJ_BIAS + layer_off, bias_mode="broadcast_N")

            # Residual add: input + attn output
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_INPUT_DRAM, sram_address=0x10000, element_size=seq_len * self.vector_length)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_ATTN_PROJ_OUTPUT_DRAM, sram_address=0x90000, element_size=seq_len * self.vector_length)
            self.eltwise_add_core(vector_A_sram_start_addr=0x10000, vector_B_sram_start_addr=0x90000, vector_C_sram_wb_addr=0x10000, element_size=seq_len * self.vector_length)
            self.sram_to_accelerator_memory(sram_address=0x10000, accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, element_size=seq_len * self.vector_length)

            # LayerNorm 2 (gamma + beta)
            total_flops += self.layer_norm_core_dram(M=seq_len, N=self.vector_length, A_DRAM_ADDR=self.LAYER0_POST_ATTN_RESIDUAL_DRAM,
                              OUTPUT_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM,
                              GAMMA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_LN2_GAMMA + layer_off,
                              BETA_DRAM_ADDR=self.DRAM_ADDR_LAYER0_LN2_BETA + layer_off)

            # MLP c_fc with GELU and bias (no SwiGLU gate)
            total_flops += self.matmat_mul_core(M=seq_len, K=self.vector_length, N=self.mlp_elements,
                A_DRAM_ADDR=self.LAYER0_PRE_MLP_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_FC + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_FC_DRAM,
                gelu_enable=True, C_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_FC_BIAS + layer_off, bias_mode="broadcast_N")

            # MLP c_proj with bias
            total_flops += self.matmat_mul_core(M=seq_len, K=self.mlp_elements, N=self.vector_length,
                A_DRAM_ADDR=self.LAYER0_MLP_FC_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_PROJ + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_MLP_PROJ_DRAM,
                C_DRAM_ADDR=self.DRAM_ADDR_LAYER0_MLP_PROJ_BIAS + layer_off, bias_mode="broadcast_N")

            # Residual add: post_attn_residual + mlp output
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_POST_ATTN_RESIDUAL_DRAM, sram_address=0x10000, element_size=seq_len * self.vector_length)
            self.accelerator_memory_to_sram(accelerator_dram_address=self.LAYER0_MLP_PROJ_DRAM, sram_address=0x90000, element_size=seq_len * self.vector_length)
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
        if prefill_seq is None:
            prefill_seq = tuple(self._cfg["default_prefill_tokens"])

        if len(prefill_seq) > 1:
            prefill_seq = prefill_seq[:-1]
            assert len(prefill_seq) == self.seq_len, f"Expected seq_len {self.seq_len}, but got {len(prefill_seq)}"
        else:
            raise ValueError("Prefill sequence must have at least 2 tokens.")

        seq_len = len(prefill_seq)
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64

        # Host-side: token + positional embeddings
        embedding_tensor = self.get_embedding_for_tokens(prefill_seq, start_pos=0)
        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
        # Causal mask: group_size=1 so standard lower-triangular
        bias_one_group = torch.full((aligned_seq_len, aligned_seq_len), float("-inf"), dtype=torch.bfloat16)
        rows = torch.arange(aligned_seq_len).unsqueeze(1)
        cols = torch.arange(aligned_seq_len).unsqueeze(0)
        valid_mask = cols <= rows
        bias_one_group.masked_fill_(valid_mask, 0.0)
        bias_one_group[:, q_seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_one_group)
        self.program_execute(prefill_program_addr, gflops=gflops)

    def compile_decoder(self, layer_size: int | None = None) -> tuple[list[int], list[int]]:
        """Compile decoder programs for seq_len buckets; write decoder_program.bin and decoder_program.json."""
        if layer_size is None:
            layer_size = self.LAYER_SIZE
        paths_cfg = self._cfg.get("paths", {})
        decoder_bin_rel = paths_cfg.get("decoder_program_bin", "gpt2_bin/decoder_program.bin")
        decoder_meta_rel = paths_cfg.get("decoder_program_meta", "gpt2_bin/decoder_program.json")
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

        ahd = self.actual_head_dim   # 64
        nkvh = self.num_kv_heads     # 12
        bpe = self.bytes_per_element

        for seq_len in self._cfg["model"]["decoder_seq_len_buckets"]:
            count_at_start = self.capture_count
            total_flops = 0
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
                # V projection with bias → directly to FLASH_V_DRAM (no temp needed)
                total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=self.head_dim,
                    A_DRAM_ADDR=self.LAYER0_PRE_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_PROJ + layer_off, OUTPUT_DRAM_ADDR=self.LAYER0_V_PROJ_DRAM,
                    C_DRAM_ADDR=self.DRAM_ADDR_LAYER0_V_BIAS + layer_off, bias_mode="broadcast_N")

                # NO RoPE — GPT-2 uses learned positional embeddings

                # Per-head scatter K/V to cache + Q → decoder_attention
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
                        self.V_CACHE_SIZE_REG, k_cache_base, self.TMP_REG)
                    self.sram_to_accelerator_memory(0x10000, 0, ahd)
                    self.overwrite_instruction_with_general_register(self.TMP_REG)

                    # Scatter V_h (contiguous 64-dim) → V cache at decode position
                    self.accelerator_memory_to_sram(
                        self.LAYER0_V_PROJ_DRAM + kv_h * ahd * bpe, 0x20000, ahd)
                    self.generate_instruction_add_imm(
                        self.V_CACHE_SIZE_REG, v_cache_base, self.TMP_REG)
                    self.sram_to_accelerator_memory(0x20000, 0, ahd)
                    self.overwrite_instruction_with_general_register(self.TMP_REG)

                    # Q_h (contiguous 64-dim) → FLASH_Q → decoder_attention
                    flash_q_addr = self.LAYER0_FLASH_Q_DRAM
                    self.accelerator_memory_to_sram(
                        self.LAYER0_Q_DRAM + kv_h * ahd * bpe, 0x30000, ahd)
                    self.sram_to_accelerator_memory(0x30000, flash_q_addr, ahd)

                    total_flops += self.decoder_attention_core(
                        head_dim=ahd,
                        seq_len=seq_len,
                        Q_DRAM_ADDR=flash_q_addr,
                        K_DRAM_ADDR=k_cache_base,
                        V_DRAM_ADDR=v_cache_base,
                        OUTPUT_DRAM_ADDR=self.LAYER0_FLASH_OUTPUT_DRAM + kv_h * ahd * bpe,
                        IDENTITY_DRAM_ADDR=self.IDENTITY_DRAM_ADDR,
                        SCRATCH_DRAM_ADDR=self.LAYER0_FLASH_SCRATCH_DRAM,
                        BIAS_DRAM_ADDR=self.LAYER0_FLASH_BIAS_DRAM,
                    )

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
                # LM head (no bias)
                total_flops += self.matmat_mul_core(M=1, K=self.vector_length, N=self.EMBEDDING_ELEMENTS,
                    A_DRAM_ADDR=self.OUTPUT_NORM_DRAM, B_DRAM_ADDR=self.DRAM_ADDR_LM_HEAD, OUTPUT_DRAM_ADDR=self.LOGITS_DRAM)

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

    @staticmethod
    def _sample_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0,
                      generated_ids: list[int] | None = None, repetition_penalty: float = 1.0) -> int:
        """Sample a token from logits with temperature, top-k, top-p, and repetition penalty.

        Args:
            logits: 1-D tensor of raw logits over the real vocabulary.
            temperature: Scales logits before softmax. 0 = greedy argmax.
            top_k: If > 0, keep only the top-k highest-probability tokens.
            top_p: If < 1.0, keep the smallest set of tokens with cumulative probability >= top_p.
            generated_ids: List of already-generated token IDs for repetition penalty.
            repetition_penalty: Penalize already-generated tokens. 1.0 = no penalty.
        """
        if temperature == 0:
            return logits.argmax().item()

        logits = logits.float()

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

        # Top-p (nucleus): keep smallest set with cumulative prob >= top_p
        if top_p < 1.0:
            sorted_probs, sorted_indices = probs.sort(descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)
            # Remove tokens with cumulative probability above top_p (keep at least 1)
            mask = cumsum - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum()
            # Sample from the filtered sorted distribution, then map back
            idx = torch.multinomial(sorted_probs, num_samples=1).item()
            return sorted_indices[idx].item()

        return torch.multinomial(probs, num_samples=1).item()

    def run_decoder(self, decoder_program_sizes: list[int], decoder_base_addr: int, token_id: int,
                    temperature: float = 0.8, top_k: int = 40, top_p: float = 0.95,
                    repetition_penalty: float = 1.2) -> dict:
        """Run decode loop with sampling. seq_len capped at MAX_CONTEXT_SIZE.

        Args:
            temperature: Logit temperature. 0 = greedy argmax (no sampling).
            top_k: Keep only top-k tokens before sampling. 0 = disabled.
            top_p: Nucleus sampling threshold. 1.0 = disabled.
            repetition_penalty: Penalize repeated tokens. 1.0 = no penalty.
        """
        if token_id is None:
            print("No last token available for decode.")
            return {}

        _stop_tokens = {self._end_of_turn_token_id}  # 50256
        use_sampling = temperature > 0
        generated_ids = []

        global _SILENT_MODE
        max_seq_len = self.MAX_CONTEXT_SIZE
        buckets = self._cfg["model"]["decoder_seq_len_buckets"]
        while self.seq_len < max_seq_len:
            _SILENT_MODE = True
            self.seq_len += 1
            # Find the correct bucket index
            prog_idx = 0
            for bi, b in enumerate(buckets):
                if self.seq_len <= b:
                    prog_idx = bi
                    break
            else:
                prog_idx = len(buckets) - 1
            prog_addr = decoder_base_addr + sum(decoder_program_sizes[:prog_idx])

            # V_CACHE_SIZE_REG: decode_pos × actual_hd × bpe
            _kv_stride = self.actual_head_dim * self.bytes_per_element  # 128 bytes/position
            self.isa_add_set_core(self.V_CACHE_SIZE_REG, (self.seq_len - 1) * _kv_stride)

            # Host-side: token + positional embedding for current decode position
            embedding_tensor = self.get_embedding_for_tokens([token_id], start_pos=self.seq_len - 1)
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)

            # Bias must match the decoder bucket seq_len, not aligned_seq_len
            bucket_seq_len = buckets[prog_idx]
            bias_host = torch.full((1, bucket_seq_len), -1e36, dtype=torch.bfloat16)
            bias_host[0, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_host)

            self.start_execute_from_dram(prog_addr)
            self.wait_queue(10.0)

            if use_sampling:
                # Read logits from DRAM and sample host-side
                logits = self.dma_from_accelerator_memory(self.LOGITS_DRAM, 50257)
                token_id = self._sample_token(logits, temperature=temperature, top_k=top_k, top_p=top_p,
                                              generated_ids=generated_ids, repetition_penalty=repetition_penalty)
                generated_ids.append(token_id)
            else:
                token_id = self.get_arg_max_index()
                # If argmax picks a padded token (>= real vocab), read logits and re-argmax
                if token_id >= 50257:
                    logits = self.dma_from_accelerator_memory(self.LOGITS_DRAM, 50257)
                    token_id = logits.argmax().item()

            token_char = self.tokenizer.decode([token_id])
            _SILENT_MODE = False
            if token_id in _stop_tokens:
                print(f"\nStop token {token_id} reached.")
                break
            print(token_char, end="", flush=True)
        return self.seq_len


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
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature. 0 = greedy argmax (default: 0.8)')
    parser.add_argument('--top-k', type=int, default=40,
                        help='Top-k sampling. 0 = disabled (default: 40)')
    parser.add_argument('--top-p', type=float, default=0.95,
                        help='Nucleus (top-p) sampling. 1.0 = disabled (default: 0.95)')
    parser.add_argument('--repetition-penalty', type=float, default=1.2,
                        help='Repetition penalty for seen tokens. 1.0 = disabled (default: 1.2)')
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
    print(f"Using DMA device: {args.dev}")
    print(f"  H2C: {user_dma_core.DMA_DEVICE_H2C}")
    print(f"  C2H: {user_dma_core.DMA_DEVICE_C2H}")
    print(f"  USER: {user_dma_core.DMA_DEVICE_USER}")
    print(f"Setting CLOCK_CYCLE_TIME_NS = {user_dma_core.CLOCK_CYCLE_TIME_NS}")

    ue = GPT2_UnifiedEngine(script_dir=script_dir, weights_bin=weights_bin_rel)

    if args.prompt is not None:
        # GPT-2 is a base model — encode prompt directly, no chat template
        prefill_seq = tuple(ue.tokenizer.encode(args.prompt))
        print(f"Prefill from prompt ({len(prefill_seq)} tokens): {args.prompt!r}")
        print(f"Sequence ids: {prefill_seq}")
    else:
        prefill_seq = tuple(ue._cfg["default_prefill_tokens"])

    print(f"\n--- Compiling ---")
    timer = time.perf_counter()
    prefill_program_addr, gflops_prefill = ue.compile_prefill(seq_len=len(prefill_seq))
    print(f"Prefill compile done in {time.perf_counter() - timer:.2f} seconds, start decoder compile...")
    decoder_bin_path = os.path.join(script_dir, cfg["paths"]["decoder_program_bin"])
    decoder_meta_path = os.path.join(script_dir, cfg["paths"]["decoder_program_meta"])
    if os.path.exists(decoder_bin_path) and os.path.exists(decoder_meta_path):
        with open(decoder_meta_path, "r") as f:
            meta = json.load(f)
        if "instruction_counts" in meta:
            decoder_program_sizes = [c * 32 for c in meta["instruction_counts"]]
        else:
            decoder_program_sizes = meta["program_sizes"]
        print(f"Decoder bin found, skipped compile ({time.perf_counter() - timer:.2f}s).")
    else:
        timer_dec = time.perf_counter()
        decoder_program_sizes, _ = ue.compile_decoder()
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
    token_cnt_decoded = ue.run_decoder(decoder_program_sizes, decoder_base_addr, token_id=prefill_seq[-1],
                                       temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                                       repetition_penalty=args.repetition_penalty)
    latency_decoder = time.perf_counter() - timer
    print(f"\nDecoder done in {latency_prefill + latency_decoder:.2f} seconds, total {token_cnt_decoded} tokens.")
    print("GPT-2 test ends.")

if __name__ == "__main__":
    main()
