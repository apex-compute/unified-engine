#!/usr/bin/env python3
"""Llama-3.2-1B inference from pre-compiled bin files (customer release).

This script is the runtime-only counterpart to ``llama3.2_1b_test.py``. It
**only loads the pre-built artifacts shipped alongside it** and executes them
on the accelerator. It never contacts HuggingFace, never imports
``AutoModelForCausalLM``, and never falls back to downloading anything. If any
required local file is missing, the script aborts with a clear error.

Required artifacts (must already exist in ``llama3.2_1b_bin/`` next to this
script — they are produced by running ``llama3.2_1b_test.py`` once on a build
machine that has internet/HF access):

  * ``weights_llama3.2_1b_hf.bin``    IF4-quantized layer weights + bf16 embedding
  * ``llama3.2_1b_instruction.bin``   unified PBI instruction bin (prefill + decoder buckets)
  * ``llama3.2_1b_instruction.json``  per-bucket sizes + total-FLOP counts
  * ``Llama-3.2-1B-Instruct/``        the tokenizer files only (tokenizer.json,
                                      tokenizer_config.json, special_tokens_map.json).
                                      The safetensors do NOT need to be shipped.

Usage:
  python llama3.2_1b_run_from_bin.py
  python llama3.2_1b_run_from_bin.py --prompt "your question"
  python llama3.2_1b_run_from_bin.py --dev xdma0 --cycle 5.88
"""

import json
import math
import os
import sys
import threading
import time

import torch
from transformers import AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, TYPE, UE_FMAX_CONTEXT_SIZE, UE_VECTOR_SIZE, UE_ARGMAX_INDEX,
    URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS, set_dma_device, ue_35bit_addr_shifter,
)
from user_dma_core import UnifiedEngine

# Print suppression toggled inside the decode loop to keep the streamed text clean.
import builtins
_original_print = builtins.print
_SILENT_MODE = False

def _quiet_print(*args, **kwargs):
    if _SILENT_MODE:
        return
    _original_print(*args, **kwargs)

builtins.print = _quiet_print


def _parse_offset(val) -> int:
    if isinstance(val, str):
        return int(val, 0)
    return int(val)


def _load_config(script_dir: str) -> dict:
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


class Llama32_1b_UnifiedEngine(UnifiedEngine):
    """Bin-only runtime engine. Does NOT compile anything; only loads
    pre-built weight + instruction bins and reads from them."""

    def __init__(self, script_dir: str | None = None):
        # IMPORTANT: program_dram_base must match the value used at capture
        # time in llama3.2_1b_test.py (default DRAM_INSTRUCTION_ADDR = 0xD0000000).
        # PBI bakes absolute JUMP_ABS targets at capture time, so loading the
        # instruction bin at a different program-DRAM address would produce
        # wrong jump targets and garbage execution. See memory note
        # ``fpga_pbi_jump_target_bake``.
        super().__init__()  # default program_dram_base = 0xD0000000

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
        self.hf_model_dir = os.path.join(self.script_dir, paths["hf_model_dir"])
        self.q_size = self.head_dim * self.group_size * self.bytes_per_element
        self.k_size = self.head_dim * self.bytes_per_element
        self.actual_head_dim = 64
        self.num_kv_heads = self.head_dim // self.actual_head_dim  # 8

        self.MAX_CONTEXT_SIZE = model["max_context_size"]
        self.LAYER_SIZE = fi["num_layers"]
        self.EMBEDDING_ELEMENTS = fi["embedding_vocab"]

        # PBI loop counters borrow ISA regs via alloc_isa_reg(). Regs 1-3 are
        # reserved for V_CACHE_SIZE_REG / TMP_REG / ROPE_SIZE_REG and reg 4 is
        # spare, so PBI loop counters allocate from reg 5 onward. The test
        # script bumps this counter to 5 at capture time so all PBI ops in the
        # instruction bin assume that reservation. This runtime engine merely
        # mirrors the test script's setup so any host-emitted isa_add_set
        # programs (V_CACHE_SIZE_REG / ROPE_SIZE_REG updates) don't clobber the
        # PBI loop counter regs at runtime.
        self._isa_reg_counter = 5
        fixed = self._cfg.get("fixed_isa_regs", {})
        self.V_CACHE_SIZE_REG = fixed["V_CACHE_SIZE_REG"]
        self.TMP_REG = fixed["TMP_REG"]
        self.ROPE_SIZE_REG = fixed["ROPE_SIZE_REG"]
        self.causal_mask_upper = False
        self._rope_global_layers = set(model["rope_global_layers"])
        self._end_of_turn_token_id = model["end_of_turn_token_id"]
        self._gamma_bin_offset = self._cfg["special"]["rms_norm"]["gamma_offset"]

        # LLaMA: these norms do not exist (engine flags carried for parity with test script).
        self._has_q_k_norm = False
        self._has_post_attn_norm = False
        self._has_post_mlp_norm = False

        # Load weight bin (file must already exist; never auto-generate).
        weights_bin = paths["weights_bin"]
        weights_path = os.path.join(self.script_dir, weights_bin)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Weight bin not found: {weights_path}\n"
                f"This is a runtime-only script; it cannot generate the weight bin. "
                f"Ensure the file is shipped alongside this script."
            )
        with open(weights_path, "rb") as f:
            self.weight_bin = f.read()
        self.weight_init()
        self.tensor_init()

    def reset_isa_reg_counter(self) -> None:
        self._isa_reg_counter = 5

    def alloc_isa_reg(self, reset: bool = False) -> int:
        if reset:
            self._isa_reg_counter = 5
        if self._isa_reg_counter > 15:
            raise ValueError("Exceeded available ISA registers (max 15)")
        reg_idx = self._isa_reg_counter
        self._isa_reg_counter += 1
        return reg_idx

    def isa_add_set_core(self, dst_reg_idx: int, immediate_value: int, timeout_s: float = 10.0) -> None:
        """Run a tiny program that sets one ISA register to an immediate value.

        Used by the decode loop to update V_CACHE_SIZE_REG / ROPE_SIZE_REG per
        token. Captured on the host, written to program DRAM after the loaded
        instruction bin, and executed once.
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

    def load_instructions(self, bin_path: str) -> tuple[int, int]:
        with open(bin_path, "rb") as f:
            data = f.read()
        total_size = len(data)
        start_addr = self.allocate_program_dram(total_size)
        self.dma_write(DMA_DEVICE_H2C, start_addr, data, total_size)
        _original_print(f"    Loaded {total_size} bytes from {os.path.basename(bin_path)} to DRAM at 0x{start_addr:x}")
        return start_addr, total_size

    def clear_inst_id(self) -> None:
        self._inst_id = 0

    def get_arg_max_index(self) -> int:
        return self.read_reg32(UE_ARGMAX_INDEX)

    def get_embedding_for_tokens(self, token_ids: list[int] | tuple) -> torch.Tensor:
        tid_t = torch.tensor(token_ids, dtype=torch.long)
        out = torch.zeros(len(token_ids), self.vector_length, dtype=torch.bfloat16)
        valid = tid_t < self.embedding_weight.shape[0]
        out[valid] = self.embedding_weight[tid_t[valid]]
        return out

    def _load_rope_host(self) -> None:
        """Generate the RoPE cos/sin tables on the host and DMA them to params
        DRAM. Tables are deterministic from the rope config — no HF lookup."""
        rope_cfg = self._cfg["special"]["rope"]
        theta = rope_cfg["theta"]
        local_base = rope_cfg["local_base"]
        num_rope_positions = rope_cfg["num_positions"]
        D_per_head = self.actual_head_dim // 2
        for name, theta_val, sz_key, attr in [
            ("ROPE_LOCAL", local_base, "ROPE_LOCAL_SIZE", "DRAM_ADDR_ROPE_LOCAL"),
            ("ROPE_GLOBAL", theta, "ROPE_GLOBAL_SIZE", "DRAM_ADDR_ROPE_GLOBAL"),
        ]:
            inv_freq = 1.0 / (theta_val ** (torch.arange(D_per_head, dtype=torch.float32) / D_per_head))
            pos = torch.arange(num_rope_positions, dtype=torch.float32)
            freqs = torch.outer(pos, inv_freq)
            cos_head = freqs.cos().to(torch.bfloat16)
            sin_head = freqs.sin().to(torch.bfloat16)
            cos_full = cos_head.repeat(1, self.num_kv_heads)
            sin_full = sin_head.repeat(1, self.num_kv_heads)
            rope_tensor = torch.cat([cos_full, cos_full, -sin_full, sin_full], dim=1)
            sz = self.weight_defs[sz_key]
            raw = rope_tensor.contiguous().view(torch.uint8).numpy().tobytes()
            raw = (raw + b"\x00" * sz)[:sz]
            addr = self.allocate_params_dram(sz)
            self.dma_write(DMA_DEVICE_H2C, addr, raw, sz)
            setattr(self, attr, addr)

    def weight_init(self) -> None:
        """Initialize FPGA DRAM from the loaded weight bin and load the local
        tokenizer files. No HuggingFace network access — ``local_files_only=True``
        on the tokenizer fails loudly if tokenizer files are missing."""
        # Tokenizer (local files only — never reaches network).
        if not os.path.exists(os.path.join(self.hf_model_dir, "tokenizer.json")) and \
           not os.path.exists(os.path.join(self.hf_model_dir, "tokenizer_config.json")):
            raise FileNotFoundError(
                f"Tokenizer files not found at {self.hf_model_dir}\n"
                f"Expected at least tokenizer.json or tokenizer_config.json. "
                f"Ship the tokenizer files alongside this script — they are tiny "
                f"(~few MB) and contain no model weights."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_dir, trust_remote_code=True, local_files_only=True,
        )

        # Embedding loaded from the weight bin (no HF model needed).
        emb_cfg = self._cfg["special"]["embedding"]
        token_embd_offset = _parse_offset(emb_cfg["token_embd_offset"])
        vocab_size = emb_cfg["vocab_size"]
        embedding_dim = emb_cfg["embedding_dim"]
        emb_nbytes = vocab_size * embedding_dim * self.bytes_per_element
        raw_emb = self.weight_bin[token_embd_offset:token_embd_offset + emb_nbytes]
        emb_tensor = torch.frombuffer(bytearray(raw_emb), dtype=torch.bfloat16).reshape(vocab_size, embedding_dim)
        self.embedding_weight = emb_tensor.clone()

        # DMA layer weights from the bin to params DRAM.
        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        base_layer0 = self.weight_defs["BLK0_ATTN_NORM_WEIGHT"]
        blk0_regions = [
            (s["key"], f"{s['key']}_SIZE", s["attr"])
            for s in self._cfg["layers"]["structure"]
        ]
        non_layer = [
            (s["key"], f"{s['key']}_SIZE", s["attr"])
            for s in self._cfg["layers"]["non_layer"]
            if s["key"] not in ("ROPE_LOCAL", "ROPE_GLOBAL")
        ]

        _original_print(f"\n--- Weights DRAM allocation, start at DRAM address: 0x{self.get_params_dram_addr():X} ---")
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
        _original_print(f"Layers 0..{self.LAYER_SIZE - 1} loaded: 0x{layers_base_dram:X} size {layers_total}")

        for off_key, sz_key, attr in non_layer:
            off = self.weight_defs[off_key]
            sz = self.weight_defs[sz_key]
            raw = self.weight_bin[off : off + sz]
            addr = self.allocate_params_dram(sz)
            self.dma_write(DMA_DEVICE_H2C, addr, raw, sz)
            setattr(self, attr, addr)

        self._load_rope_host()
        _original_print(f"    Weights end at DRAM address: 0x{self.get_params_dram_addr():X}")
        _original_print("Tokenizer loaded from local files.")

    def tensor_init(self) -> None:
        seq_len = self.MAX_CONTEXT_SIZE
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64

        _original_print(f"Allocate tensor dram start at DRAM address: 0x{self.get_tensor_dram_addr():X}")
        self.LAYER0_V_DRAM = self.allocate_tensor_dram(self.LAYER_SIZE * self.MAX_CONTEXT_SIZE * self.k_size)
        self.LAYER0_K_ROPE_DRAM = self.allocate_tensor_dram(self.LAYER_SIZE * self.MAX_CONTEXT_SIZE * self.k_size)
        zero_pad = torch.zeros(self.LAYER_SIZE * self.MAX_CONTEXT_SIZE * self.k_size, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_K_ROPE_DRAM, zero_pad)
        zero_add = torch.zeros(seq_len * self.head_dim * self.bytes_per_element, dtype=torch.bfloat16)
        self.ZERO_DRAM_ADDR = self.allocate_tensor_dram(seq_len * self.head_dim * self.bytes_per_element)
        self.dma_to_accelerator_memory(self.ZERO_DRAM_ADDR, zero_add)
        self.IDENTITY_DRAM_ADDR = self.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * self.bytes_per_element)
        self.dma_to_accelerator_memory(self.IDENTITY_DRAM_ADDR, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))
        self.LAYER0_FLASH_Q_DRAM = self.allocate_tensor_dram(aligned_seq_len * self.head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_K_DRAM = self.allocate_tensor_dram(aligned_seq_len * self.head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_V_DRAM = self.allocate_tensor_dram(aligned_seq_len * self.head_dim * self.bytes_per_element)
        zero_pad = torch.zeros(aligned_seq_len * self.head_dim * self.bytes_per_element, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_Q_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_K_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_V_DRAM, zero_pad)
        self.LAYER0_INPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_Q_DRAM = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_K_DRAM = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_K_NORM_DRAM = self.LAYER0_K_DRAM
        self.LAYER0_Q_NORM_DRAM = self.LAYER0_Q_DRAM
        self.LAYER0_V_PROJ_TEMP = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_FLASH_OUT_HEAD_DRAM = self.allocate_tensor_dram(aligned_seq_len * self.actual_head_dim * self.bytes_per_element)
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
        _original_print(f"    Tensor dram end: 0x{self.get_tensor_dram_addr():X}, usage: {self.get_tensor_dram_usage()} bytes")

    def program_execute(self, program_start_addr: int, timeout: float = 120.0, gflops: float = None) -> None:
        self.start_execute_from_dram(program_start_addr)
        self.wait_queue(timeout)
        latency = self.report_latency_in_us()
        _original_print(f"    Total program execution latency = {latency} us")
        if gflops is not None:
            gflops_program, _ = self.report_flop_rate_gflops(gflops)
            _original_print(f"    GFLOPS: {gflops_program:.2f}")

    def run_prefill(self, prefill_program_addr: int, prefill_seq, bucket_seq_len: int,
                    gflops: int = None) -> None:
        """Run the prefill bucket-program. Pads ``prefill_seq[:-1]`` to
        ``bucket_seq_len`` with the last-actual-token embedding; the
        bucket-padded positions get computed but their KV cache slots are
        masked from the decoder's attention by a bias mask, so they never
        affect output. ``self.seq_len`` is set to the *actual* prefill length
        after execute so the decoder starts at the right position."""
        if len(prefill_seq) <= 1:
            raise ValueError("Prefill sequence must have at least 2 tokens.")
        prefill_seq = prefill_seq[:-1]  # last token seeds the decoder
        actual_seq_len = len(prefill_seq)
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
        # Decoder reads ctx 0..actual_seq_len-1 only; padded KV slots are
        # masked out by the decoder's own bias mask in run_decoder.
        self.seq_len = actual_seq_len

    def run_decoder(self, decoder_program_sizes: list[int], decoder_base_addr: int, token_id: int,
                    gflops_per_token: list[int] | None = None) -> dict:
        """Decode loop: ~one bucket-program per token, picked by current
        seq_len. Emits a small isa_add_set program per token to update
        V_CACHE_SIZE_REG and ROPE_SIZE_REG for the new decode position."""
        if token_id is None:
            _original_print("No last token available for decode.")
            return {"decoded_count": 0, "seq_len": self.seq_len}

        _llama_stop_tokens = {128001, 128008, self._end_of_turn_token_id}
        global _SILENT_MODE
        max_seq_len = self.MAX_CONTEXT_SIZE
        decoded_count = 0
        per_token_times: list[float] = []
        decode_loop_start = time.perf_counter()

        while self.seq_len < max_seq_len:
            _SILENT_MODE = True
            t0 = time.perf_counter()
            self.seq_len += 1
            aligned_seq_len = ((self.seq_len + 63) // 64) * 64
            prog_idx = min((self.seq_len - 1) // 64, len(decoder_program_sizes) - 1)
            prog_addr = decoder_base_addr + sum(decoder_program_sizes[:prog_idx])

            _kv_stride = self.actual_head_dim * self.bytes_per_element
            _rope_row  = self.head_dim * 2 * self.bytes_per_element
            self.isa_add_set_core(self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter((self.seq_len - 1) * _kv_stride))
            self.isa_add_set_core(self.ROPE_SIZE_REG,    ue_35bit_addr_shifter((self.seq_len - 1) * _rope_row))

            embedding_tensor = self.get_embedding_for_tokens([token_id])
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)

            bias_host = torch.full((1, aligned_seq_len), -1e36, dtype=torch.bfloat16)
            bias_host[0, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_host)

            self.start_execute_from_dram(prog_addr)
            self.wait_queue(30.0)
            token_id = self.get_arg_max_index()
            token_char = self.tokenizer.decode([token_id])
            per_token_times.append(time.perf_counter() - t0)
            decoded_count += 1
            _SILENT_MODE = False
            if token_id in _llama_stop_tokens:
                _original_print(f"\nStop token {token_id} reached.")
                break
            _original_print(token_char, end="", flush=True)

        decode_elapsed = time.perf_counter() - decode_loop_start
        tps = decoded_count / decode_elapsed if decode_elapsed > 0 else 0.0
        avg_ms = (sum(per_token_times) / len(per_token_times) * 1000.0) if per_token_times else 0.0
        _original_print(f"\n--- Decode benchmark ---")
        _original_print(f"  Tokens decoded : {decoded_count}")
        _original_print(f"  Elapsed        : {decode_elapsed:.3f} s")
        _original_print(f"  Throughput     : {tps:.3f} tokens/s")
        _original_print(f"  Avg per token  : {avg_ms:.2f} ms")
        return {
            "decoded_count": decoded_count,
            "seq_len": self.seq_len,
            "decode_elapsed_s": decode_elapsed,
            "tokens_per_second": tps,
            "avg_per_token_ms": avg_ms,
        }


def _progress_timer(label: str, start_time: float, stop_event: threading.Event) -> None:
    while not stop_event.wait(1.0):
        _original_print(f"\r  {label} ({time.perf_counter() - start_time:.0f}s)", end="", flush=True)


def _verify_artifacts(script_dir: str) -> None:
    """Hard fail before any FPGA touch if a required file is missing. This is
    what guarantees the customer doesn't see a half-initialized engine when an
    artifact wasn't shipped."""
    bin_dir = os.path.join(script_dir, "llama3.2_1b_bin")
    required_files = [
        ("llama3.2_1b_config.json",                                       "model config"),
        ("llama3.2_1b_bin/weights_llama3.2_1b_hf.bin",                    "weight bin"),
        ("llama3.2_1b_bin/llama3.2_1b_instruction.bin",                   "unified instruction bin"),
        ("llama3.2_1b_bin/llama3.2_1b_instruction.json",                  "instruction bin meta"),
    ]
    tokenizer_dir = os.path.join(bin_dir, "Llama-3.2-1B-Instruct")
    tokenizer_required = [
        # AutoTokenizer needs at least one of these to identify the format.
        (os.path.join(tokenizer_dir, "tokenizer.json"),        "tokenizer.json"),
        (os.path.join(tokenizer_dir, "tokenizer_config.json"), "tokenizer_config.json"),
    ]

    missing = []
    for rel, label in required_files:
        if not os.path.exists(os.path.join(script_dir, rel)):
            missing.append(f"  {rel}   ({label})")
    # Tokenizer: need at least one of the two key files to exist.
    if not any(os.path.exists(p) for p, _ in tokenizer_required):
        missing.append(f"  llama3.2_1b_bin/Llama-3.2-1B-Instruct/{{tokenizer.json,tokenizer_config.json}}   (tokenizer files)")

    if missing:
        _original_print("ERROR: This is a runtime-only script and requires pre-built artifacts.")
        _original_print("Missing files:")
        for line in missing:
            _original_print(line)
        _original_print("\nGet them from the build step:")
        _original_print("  python llama3.2_1b_test.py")
        _original_print("on a machine with internet/HF access, then ship the bin directory alongside this script.")
        sys.exit(1)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Llama-3.2-1B inference from pre-compiled bins (no HF access).")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt; tokenized via the local chat template. "
                             "Overrides the default prompt in the config.")
    parser.add_argument("--dev", type=str, default="xdma0",
                        help="DMA device name (e.g. xdma0). Default: xdma0.")
    parser.add_argument("--cycle", type=float, default=1/0.17,
                        help="Clock cycle time in nanoseconds. Default: ~5.88 ns.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Hard-fail BEFORE touching the FPGA if any artifact is missing.
    _verify_artifacts(script_dir)

    set_dma_device(args.dev)
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle
    _original_print(f"Llama-3.2-1B on {args.dev} (run from pre-compiled bins)")

    t0 = time.perf_counter()
    _original_print("Loading weights + tokenizer ...")
    ue = Llama32_1b_UnifiedEngine(script_dir=script_dir)
    _original_print(f"  Weights + tensors: {time.perf_counter() - t0:.2f}s")

    cfg = _load_config(script_dir)
    if args.prompt is not None:
        conversation = [{"role": "user", "content": args.prompt}]
        prompt_with_template = ue.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True)
        prefill_seq = tuple(ue.tokenizer.encode(prompt_with_template, add_special_tokens=False))
        _original_print(f"Prompt ({len(prefill_seq)} tokens): {args.prompt!r}")
    else:
        prefill_seq = tuple(cfg["default_prefill_tokens"])
        _original_print(f"Default prompt ({len(prefill_seq)} tokens)")
    _original_print(f"Prompt text: {ue.tokenizer.decode(prefill_seq, skip_special_tokens=False)!r}")

    # Unified instruction bin: contains all prefill buckets followed by all
    # decoder buckets, written under one start_capture/stop_capture session by
    # the test script's compile_instructions().
    inst_bin_path  = os.path.join(script_dir, cfg["paths"]["instruction_bin"])
    inst_meta_path = os.path.join(script_dir, cfg["paths"]["instruction_meta"])
    with open(inst_meta_path) as f:
        meta = json.load(f)

    prefill_buckets       = meta["prefill_buckets"]
    prefill_program_sizes = meta["prefill_program_sizes"]
    prefill_total_flops   = meta["prefill_total_flops"]
    decoder_buckets       = meta["decoder_buckets"]
    decoder_program_sizes = meta["decoder_program_sizes"]
    decoder_total_flops   = meta["decoder_total_flops"]

    # Pick the smallest prefill bucket that fits actual prefill length.
    actual_seq_len = len(prefill_seq) - 1  # last token seeds the decoder
    try:
        bucket_idx = next(i for i, b in enumerate(prefill_buckets) if b - 1 >= actual_seq_len)
    except StopIteration:
        raise RuntimeError(
            f"Prompt too long: actual_seq_len={actual_seq_len} > largest prefill bucket "
            f"({prefill_buckets[-1]}-1={prefill_buckets[-1]-1}). Add a bigger bucket in the build step."
        )
    bucket_seq_len = prefill_buckets[bucket_idx] - 1
    gflops_prefill = prefill_total_flops[bucket_idx]

    _original_print(f"\n--- Loading unified instruction bin ---")
    t_load = time.perf_counter()
    inst_base_addr, _ = ue.load_instructions(inst_bin_path)
    # Decoder buckets follow all prefill buckets in the same bin.
    prefill_base_addr = inst_base_addr
    decoder_base_addr = inst_base_addr + sum(prefill_program_sizes)
    prefill_program_addr = prefill_base_addr + sum(prefill_program_sizes[:bucket_idx])
    _original_print(f"  Programs loaded: {time.perf_counter() - t_load:.3f}s "
                    f"(prefill base 0x{prefill_base_addr:X}, decoder base 0x{decoder_base_addr:X})")

    _original_print(f"\n--- Prefill (actual {actual_seq_len} tokens, bucket {prefill_buckets[bucket_idx]} = seq_len {bucket_seq_len}) ---")
    t_prefill = time.perf_counter()
    stop = threading.Event()
    timer = threading.Thread(target=_progress_timer, args=("Prefill running...", t_prefill, stop), daemon=True)
    timer.start()
    ue.run_prefill(prefill_program_addr, prefill_seq=prefill_seq,
                   bucket_seq_len=bucket_seq_len, gflops=gflops_prefill)
    stop.set(); timer.join()
    latency_prefill = time.perf_counter() - t_prefill
    _original_print(f"\r  Prefill: {latency_prefill:.3f}s")

    _original_print("\n--- Decoding ---")
    t_decode = time.perf_counter()
    decode_stats = ue.run_decoder(
        decoder_program_sizes, decoder_base_addr,
        token_id=prefill_seq[-1], gflops_per_token=decoder_total_flops)
    latency_decode = time.perf_counter() - t_decode

    if isinstance(decode_stats, dict):
        decoded_count = decode_stats.get("decoded_count", 0)
        final_seq_len = decode_stats.get("seq_len", 0)
        tps_wall = decoded_count / latency_decode if latency_decode > 0 else 0.0
        _original_print(f"\nDecoder done in {latency_prefill + latency_decode:.2f}s "
                        f"(prefill {latency_prefill:.2f}s + decode {latency_decode:.2f}s), "
                        f"decoded {decoded_count} tokens, final seq_len {final_seq_len}.")
        _original_print(f"Decode throughput (wall-clock): {tps_wall:.3f} tokens/s")
    _original_print("Llama-3.2-1B run_from_bin ends.")


if __name__ == "__main__":
    main()
