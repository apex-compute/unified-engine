#!/usr/bin/env python3
"""Llama-3.2-1B inference from pre-compiled bins — runtime-only, fully self-contained.

This is the runtime counterpart to ``llama3.2_1b_test.py``. It imports **nothing**
from the test script — only ``user_dma_core`` and a *local* tokenizer — so it can
ship to users with just the bin files and run with no build-time code, no
HuggingFace model, and no internet. If any required local artifact is missing it
aborts with a clear error before touching the FPGA.

Required artifacts (in ``llama3.2_1b_bin/`` next to this script — produced by
running ``llama3.2_1b_test.py`` once on a build machine that has HF access):
  * ``weights_llama3.2_1b_hf.bin``   IF4 layer weights + bf16 embedding
  * ``llama_instruction.bin``        unified dynamic-PBI instruction bin (prefill + decoder)
  * ``llama_instruction.json``       per-stage start addresses / sizes / FLOPs
  * ``Llama-3.2-1B-Instruct/``       tokenizer files only (tokenizer.json /
                                     tokenizer_config.json). Model weights NOT needed.

Design mirrors test.py §A–§D (see notes_llama3.2_1b.md): full-4 GB DRAM layout,
dynamic-PBI single bin (GPR-primed preamble + jump_abs), prefill_context_size=128,
and sampling decode (recency-decayed / windowed / structural-exempt repetition
penalty). DECODE DEFAULT is PENALIZED GREEDY (temp 0 + rep-pen 1.1): argmax of the
penalty-adjusted logits — deterministic, correct math, loop-free; see notes §D. The
shipped bin is compiled with LM-head writeback enabled, so sampling (temp>0),
penalized greedy (temp 0 + rep-pen), and pure greedy (temp 0, rep-pen 1.0) all work.

Usage:
  python llama3.2_1b_run_from_bin.py
  python llama3.2_1b_run_from_bin.py --prompt "your question"
  python llama3.2_1b_run_from_bin.py --temperature 0 --dev xdma0     # greedy
"""

import json
import os
import sys
import time

import torch
from transformers import AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, DMA_DEVICE_C2H, UE_FMAX_CONTEXT_SIZE, UE_VECTOR_SIZE,
    set_dma_device, ue_35bit_addr_shifter,
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


class Llama32_1b_RunFromBin(UnifiedEngine):
    """Self-contained bin-only runtime engine. Loads pre-built weight + instruction
    bins and runs prefill + sampling decode. No compile, no HF model, no test import."""

    def __init__(self, script_dir: str | None = None):
        # Full 4 GB DRAM layout — MUST match llama3.2_1b_test.py at capture time:
        # the instruction bin bakes absolute JUMP_ABS targets AND the tensor-DRAM
        # addresses the decoder program reads, so the layout has to be identical.
        #   params 0x00000000 | tensor 0x58000000 | program 0xE0000000
        super().__init__(
            params_dram_base=0x00000000,
            tensor_dram_base=0x58000000,
            program_dram_base=0xE0000000,
        )
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
        self.num_kv_heads = self.head_dim // self.actual_head_dim   # 8
        self.MAX_CONTEXT_SIZE = model["max_context_size"]
        self.PREFILL_CONTEXT_SIZE = model["prefill_context_size"]
        self.LAYER_SIZE = fi["num_layers"]
        self.EMBEDDING_ELEMENTS = fi["embedding_vocab"]

        # Fixed ISA registers (must match capture-time reservation). The runtime
        # preamble writes these via ADD_SET; PBI loop counters in the bin live at
        # max(fixed)+1 and above.
        fixed = self._cfg.get("fixed_isa_regs", {})
        self.V_CACHE_SIZE_REG = fixed["V_CACHE_SIZE_REG"]
        self.TMP_REG = fixed["TMP_REG"]
        self.ROPE_SIZE_REG = fixed["ROPE_SIZE_REG"]
        self.gpr_bucket_idx = fixed["GPR_BUCKET_IDX_REG"]
        self.gpr_seq_len = fixed["GPR_SEQ_LEN_REG"]
        self._isa_reg_counter = max(fixed.values()) + 1
        self._rope_global_layers = set(model["rope_global_layers"])
        self._end_of_turn_token_id = model["end_of_turn_token_id"]

        weights_path = os.path.join(self.script_dir, paths["weights_bin"])
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Weight bin not found: {weights_path}\n"
                f"This runtime-only script cannot generate it — ship it alongside the script."
            )
        with open(weights_path, "rb") as f:
            self.weight_bin = f.read()
        self.weight_init()
        self.tensor_init()

    # ---- embedding + RoPE (from bin / config — no HF model) -----------------
    def get_embedding_for_tokens(self, token_ids) -> torch.Tensor:
        tid_t = torch.tensor(token_ids, dtype=torch.long)
        out = torch.zeros(len(token_ids), self.vector_length, dtype=torch.bfloat16)
        valid = tid_t < self.embedding_weight.shape[0]
        out[valid] = self.embedding_weight[tid_t[valid]]
        return out

    def _load_rope_host(self) -> None:
        """Host-compute the N=512 tiled RoPE table (8 KV heads tiled) and DMA it to
        params DRAM. Deterministic from rope config — no HF lookup. Layout per row:
        [cos_full(256), cos_full(256), -sin_full(256), sin_full(256)] = 1024 bf16."""
        rope_cfg = self._cfg["special"]["rope"]
        theta = rope_cfg["theta"]
        local_base = rope_cfg["local_base"]
        num_rope_positions = rope_cfg["num_positions"]
        D_per_head = self.actual_head_dim // 2   # 32 freqs per KV head
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
        """Init DRAM from the weight bin + load the local tokenizer. No HF model;
        ``local_files_only=True`` fails loudly if tokenizer files are missing."""
        if not os.path.exists(os.path.join(self.hf_model_dir, "tokenizer.json")) and \
           not os.path.exists(os.path.join(self.hf_model_dir, "tokenizer_config.json")):
            raise FileNotFoundError(
                f"Tokenizer files not found at {self.hf_model_dir}\n"
                f"Ship tokenizer.json / tokenizer_config.json alongside this script (tiny, no weights)."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_dir, trust_remote_code=True, local_files_only=True)

        # Embedding straight from the weight bin (bf16) — no HF model.
        emb_cfg = self._cfg["special"]["embedding"]
        token_embd_offset = _parse_offset(emb_cfg["token_embd_offset"])
        vocab_size = emb_cfg["vocab_size"]
        embedding_dim = emb_cfg["embedding_dim"]
        emb_nbytes = vocab_size * embedding_dim * self.bytes_per_element
        raw_emb = self.weight_bin[token_embd_offset:token_embd_offset + emb_nbytes]
        self.embedding_weight = torch.frombuffer(
            bytearray(raw_emb), dtype=torch.bfloat16).reshape(vocab_size, embedding_dim).clone()

        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        base_layer0 = self.weight_defs["BLK0_ATTN_NORM_WEIGHT"]
        blk0_regions = [(s["key"], f"{s['key']}_SIZE", s["attr"])
                        for s in self._cfg["layers"]["structure"]]
        non_layer = [(s["key"], f"{s['key']}_SIZE", s["attr"])
                     for s in self._cfg["layers"]["non_layer"]
                     if s["key"] not in ("ROPE_LOCAL", "ROPE_GLOBAL")]

        _original_print(f"\n--- Weights DRAM allocation, start 0x{self.get_params_dram_addr():X} ---")
        layers_total = self.LAYER_SIZE * LAYER_WEIGHT_SIZE
        layers_base_dram = self.allocate_params_dram(layers_total)
        for layer_idx in range(self.LAYER_SIZE):
            for off_key, sz_key, attr in blk0_regions:
                off = self.weight_defs[off_key]
                sz = self.weight_defs[sz_key]
                bin_off = off + layer_idx * LAYER_WEIGHT_SIZE
                raw = self.weight_bin[bin_off:bin_off + sz]
                dram_addr = layers_base_dram + layer_idx * LAYER_WEIGHT_SIZE + (off - base_layer0)
                self.dma_write(DMA_DEVICE_H2C, dram_addr, raw, sz)
            if layer_idx == 0:
                for off_key, sz_key, attr in blk0_regions:
                    off = self.weight_defs[off_key]
                    setattr(self, attr, layers_base_dram + (off - base_layer0))
        _original_print(f"Layers 0..{self.LAYER_SIZE - 1} loaded: 0x{layers_base_dram:X} size {layers_total}")

        for off_key, sz_key, attr in non_layer:
            off = self.weight_defs[off_key]
            sz = self.weight_defs[sz_key]
            raw = self.weight_bin[off:off + sz]
            addr = self.allocate_params_dram(sz)
            self.dma_write(DMA_DEVICE_H2C, addr, raw, sz)
            setattr(self, attr, addr)

        self._load_rope_host()
        _original_print(f"    Weights end 0x{self.get_params_dram_addr():X}; tokenizer loaded (local).")

    def tensor_init(self) -> None:
        """Allocate hardware DRAM tensors. The order + sizes MUST match
        llama3.2_1b_test.py.tensor_init exactly, because the decoder program in the
        bin reads baked tensor-DRAM addresses derived from this allocation order."""
        seq_len = self.MAX_CONTEXT_SIZE
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64

        _original_print(f"Allocate tensor dram start 0x{self.get_tensor_dram_addr():X}")
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
        self.LAYER0_FLASH_ATTN_P_DRAM = self.allocate_tensor_dram(aligned_seq_len * aligned_seq_len * self.bytes_per_element)
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
        _original_print(f"    Tensor dram end 0x{self.get_tensor_dram_addr():X}, usage {self.get_tensor_dram_usage()} bytes")

    # ---- sampling (identical mechanism to test.py §D) -----------------------
    def _structural_token_ids(self) -> set:
        """Token ids never repetition-penalized: punctuation/whitespace/newline/special.
        Precomputed once from the local tokenizer vocab and cached. Exempting these
        'glue' tokens is what stops long small-model generations collapsing to word-salad."""
        cached = getattr(self, "_struct_ids_cache", None)
        if cached is not None:
            return cached
        import string
        allowed = set(string.punctuation) | set(string.whitespace) | set("—–’‘“”…·•‹›«»¡¿")
        ids = set(int(i) for i in (getattr(self.tokenizer, "all_special_ids", []) or []))
        for i in range(self.EMBEDDING_ELEMENTS):
            s = self.tokenizer.decode([i]).strip()
            if s == "" or all(ch in allowed for ch in s):
                ids.add(i)
        self._struct_ids_cache = ids
        return ids

    def _structural_ids_tensor(self) -> torch.Tensor:
        t = getattr(self, "_struct_ids_tensor_cache", None)
        if t is None:
            t = torch.tensor(sorted(self._structural_token_ids()), dtype=torch.long)
            self._struct_ids_tensor_cache = t
        return t

    def sample_next_token(self, prev_tokens) -> int:
        """Read the full logits row from LOGITS_DRAM and sample with the configured
        temperature / top_k / top_p / repetition_penalty (recency-decayed, windowed,
        structural-exempt). Greedy (temperature<=0) is handled by the caller via the
        HW argmax register, so this assumes sampling is on."""
        vocab = self.EMBEDDING_ELEMENTS
        bpe = self.bytes_per_element
        buf = torch.empty(vocab, dtype=torch.bfloat16)
        self.dma_read(DMA_DEVICE_C2H, self.LOGITS_DRAM, buf, vocab * bpe)
        logits = buf.float()
        # Recency-decayed frequency repetition penalty over the last rep_window tokens,
        # exempting structural/special tokens. See notes_llama3.2_1b.md §D.
        rep_pen = float(getattr(self, "repetition_penalty", 1.0))
        if rep_pen != 1.0 and prev_tokens:
            rep_window = int(getattr(self, "rep_window", 256))
            rep_decay = float(getattr(self, "rep_decay", 0.97))
            window = torch.tensor(prev_tokens[-rep_window:], dtype=torch.long)
            L = window.numel()
            ages = torch.arange(L - 1, -1, -1, dtype=torch.float32)
            contrib = rep_decay ** ages
            weight = torch.zeros(vocab, dtype=torch.float32)
            weight.index_add_(0, window, contrib)
            weight[self._structural_ids_tensor()] = 0.0
            nz = weight > 0
            if bool(nz.any()):
                factor = rep_pen ** weight[nz]
                v = logits[nz]
                logits[nz] = torch.where(v > 0, v / factor, v * factor)
        temp = float(getattr(self, "temperature", 1.0))
        if temp <= 0:
            return int(logits.argmax().item())
        logits = logits / temp
        top_k = int(getattr(self, "top_k", 0) or 0)
        if top_k > 0 and top_k < vocab:
            kth = torch.topk(logits, top_k).values[-1]
            logits[logits < kth] = float("-inf")
        top_p = float(getattr(self, "top_p", 1.0))
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumprob = torch.cumsum(probs, dim=-1)
            keep = cumprob <= top_p
            keep[0] = True
            drop_idx = sorted_idx[~keep]
            logits[drop_idx] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    # ---- run path (dynamic-PBI single bin: preamble + jump_abs) -------------
    def run_llama(self) -> None:
        """Load the unified instruction image and run prefill + decoder loop. A small
        captured preamble primes the GPRs and jumps into the cached prefill / decoder
        program. ``self.prefill_seq`` must be set by the caller."""
        paths_cfg = self._cfg.get("paths", {})
        meta_path = os.path.join(self.script_dir,
                                 paths_cfg.get("instruction_meta", "llama3.2_1b_bin/llama_instruction.json"))
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
        _kv_stride = self.actual_head_dim * self.bytes_per_element
        _rope_row = self.head_dim * 2 * self.bytes_per_element

        prefill_seq = self.prefill_seq
        if len(prefill_seq) < 2:
            raise ValueError("Prefill sequence must have at least 2 tokens.")
        prefill_seq = prefill_seq[:-1]  # last token seeds the decoder
        prefill_seq_len = len(prefill_seq)
        self.seq_len = prefill_seq_len

        q_seq_len = prefill_seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        bucket_idx = aligned_seq_len // UE_VECTOR_SIZE
        flops_prefill = flops_prefill_template * prefill_seq_len // max(template_seq_len, 1)

        # Prefill preamble: prime gpr_seq_len + gpr_bucket_idx, jump into cached prefill.
        self.clear_inst_id()
        self.start_capture()
        self.generate_instruction_add_set(self.gpr_seq_len, prefill_seq_len)
        self.generate_instruction_add_set(self.gpr_bucket_idx, bucket_idx)
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(prefill_program_addr))
        self.stop_capture()
        self.write_captured_instructions_to_dram(preamble_addr)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        self.clear_capture_buffer()

        embedding_tensor = self.get_embedding_for_tokens(prefill_seq)
        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
        bias_one_group = torch.full((aligned_seq_len, aligned_seq_len), float("-inf"), dtype=torch.bfloat16)
        rows = torch.arange(aligned_seq_len).unsqueeze(1)
        cols = torch.arange(aligned_seq_len).unsqueeze(0)
        valid_mask = (cols // self.group_size) <= (rows // self.group_size)
        bias_one_group.masked_fill_(valid_mask, 0.0)
        bias_one_group[:, q_seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_one_group)

        _original_print(f"\n--- Starting prefill (seq_len={prefill_seq_len}) ---")
        timer = time.perf_counter()
        self.program_execute(preamble_addr, flops=flops_prefill)
        latency_prefill = time.perf_counter() - timer
        _original_print(f"Prefill done in {latency_prefill:.2f}s\n")

        _original_print("--- Starting decoder ---")
        timer = time.perf_counter()
        token_id = self.prefill_seq[-1]
        _llama_stop_tokens = {128001, 128008, self._end_of_turn_token_id}
        global _SILENT_MODE

        if not hasattr(self, "_generated_tokens"):
            self._generated_tokens = list(self.prefill_seq)
        # Penalized greedy (temp 0 + rep_pen!=1) and sampling (temp>0) read the logits back
        # so the repetition penalty applies before token selection; pure HW argmax otherwise.
        _temperature = float(getattr(self, "temperature", 0.0))
        _rep_pen = float(getattr(self, "repetition_penalty", 1.0))
        _use_logit_readback = _temperature > 0 or _rep_pen != 1.0

        import shutil
        _use_status = sys.stdout.isatty()
        def _status_setup():
            r = shutil.get_terminal_size().lines
            sys.stdout.write(f"\033[1;{r - 1}r"); sys.stdout.write(f"\033[{r - 1};1H"); sys.stdout.flush()
        def _status_update():
            r = shutil.get_terminal_size().lines
            n = self.seq_len - prefill_seq_len
            elapsed = time.perf_counter() - timer
            rate = n / elapsed if elapsed > 0 else 0.0
            sys.stdout.write("\0337"); sys.stdout.write(f"\033[{r};1H\033[2K")
            sys.stdout.write(f" decoding… {n} tokens  (pos {self.seq_len}/{self.MAX_CONTEXT_SIZE})  "
                             f"{elapsed:.1f}s  {rate:.1f} tok/s")
            sys.stdout.write("\0338"); sys.stdout.flush()
        def _status_teardown():
            r = shutil.get_terminal_size().lines
            sys.stdout.write("\033[r"); sys.stdout.write(f"\033[{r};1H\033[2K"); sys.stdout.flush()
        if _use_status:
            _status_setup()

        while self.seq_len < self.MAX_CONTEXT_SIZE:
            _SILENT_MODE = True
            self.seq_len += 1
            aligned_seq_len = ((self.seq_len + 63) // 64) * 64
            bucket_idx = min((self.seq_len + 63) // 64, _max_gpr_bucket)
            decode_pos = self.seq_len - 1

            embedding_tensor = self.get_embedding_for_tokens([token_id])
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
            bias_host = torch.full((1, aligned_seq_len), -1e36, dtype=torch.bfloat16)
            bias_host[0, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_host)

            # Decoder preamble: prime bucket + 2 position GPRs, jump into cached decoder.
            self.clear_inst_id()
            self.start_capture()
            self.generate_instruction_add_set(self.gpr_bucket_idx, bucket_idx)
            self.generate_instruction_add_set(self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter(decode_pos * _kv_stride))
            self.generate_instruction_add_set(self.ROPE_SIZE_REG, ue_35bit_addr_shifter(decode_pos * _rope_row))
            self.generate_instruction_jump_abs(ue_35bit_addr_shifter(decoder_program_addr))
            self.stop_capture()
            self.write_captured_instructions_to_dram(preamble_addr)
            self.clear_capture_buffer()

            self.program_execute(preamble_addr, flops=decoder_flops_per_token)
            if _use_logit_readback:
                token_id = self.sample_next_token(self._generated_tokens)
            else:
                token_id = self.get_arg_max_index(rank=1)
            self._generated_tokens.append(token_id)
            token_char = self.tokenizer.decode([token_id])
            _SILENT_MODE = False
            if token_id in _llama_stop_tokens:
                if _use_status:
                    _status_teardown()
                _original_print(f"\nStop token {token_id} reached.")
                break
            _original_print(token_char, end="", flush=True)
            if _use_status:
                _status_update()
        else:
            if _use_status:
                _status_teardown()

        latency_decoder = time.perf_counter() - timer
        tokens_decoded = self.seq_len - prefill_seq_len
        _original_print(f"\nDecoder done in {latency_prefill + latency_decoder:.2f}s, "
                        f"speed: {tokens_decoded / max(latency_decoder, 1e-9):.2f} tokens/s, "
                        f"total {self.seq_len} tokens.")


def _verify_artifacts(script_dir: str, cfg: dict) -> None:
    """Hard-fail before any FPGA touch if a required artifact is missing."""
    paths = cfg["paths"]
    required = [
        ("llama3.2_1b_config.json", "model config"),
        (paths["weights_bin"], "weight bin"),
        (paths["instruction_bin"], "unified instruction bin"),
        (paths["instruction_meta"], "instruction bin meta"),
    ]
    tok_dir = os.path.join(script_dir, paths["hf_model_dir"])
    missing = [f"  {rel}   ({label})" for rel, label in required
               if not os.path.exists(os.path.join(script_dir, rel))]
    if not any(os.path.exists(os.path.join(tok_dir, f)) for f in ("tokenizer.json", "tokenizer_config.json")):
        missing.append(f"  {paths['hf_model_dir']}/{{tokenizer.json,tokenizer_config.json}}   (tokenizer files)")
    if missing:
        _original_print("ERROR: runtime-only script — required pre-built artifacts are missing:")
        for line in missing:
            _original_print(line)
        _original_print("\nGenerate them on a build machine with HF access:\n  python llama3.2_1b_test.py\n"
                        "then ship the llama3.2_1b_bin/ directory alongside this script.")
        sys.exit(1)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Llama-3.2-1B inference from pre-compiled bins (no HF, no internet, no test-script import).")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt (tokenized via the local chat template). Overrides the config default.")
    parser.add_argument("--dev", type=str, default="xdma0", help="DMA device name. Default: xdma0.")
    parser.add_argument("--cycle", type=float, default=1 / 0.17, help="Clock cycle time in ns. Default ~5.88.")
    # Sampling — defaults match llama3.2_1b_test.py (temp 0.4 + top-k 40 fixes FPGA
    # arithmetic errors while keeping long-context clean; see notes §D).
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = penalized greedy (argmax of penalty-adjusted logits — deterministic, correct math, loop-free; the default). >0 = sampling. Default 0.")
    parser.add_argument("--top-k", type=int, default=40,
                        help="Top-k filter (0=off). Caps the tail so wrong tokens can't be sampled. Default 40.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) filter. Default 0.9.")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                        help="Recency-decayed repetition penalty (>1 down-weights repeats). Default 1.1.")
    parser.add_argument("--rep-window", type=int, default=256,
                        help="Repetition penalty look-back window in tokens. Default 256.")
    parser.add_argument("--rep-decay", type=float, default=0.97,
                        help="Recency decay for the repetition penalty (per-token age). Default 0.97.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible sampling.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = _load_config(script_dir)

    # Hard-fail BEFORE touching the FPGA if any artifact is missing.
    _verify_artifacts(script_dir, cfg)

    set_dma_device(args.dev)
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle
    _original_print(f"Llama-3.2-1B on {args.dev} (run from pre-compiled bins)")

    t0 = time.perf_counter()
    _original_print("Loading weights + tokenizer ...")
    ue = Llama32_1b_RunFromBin(script_dir=script_dir)
    _original_print(f"  Weights + tensors: {time.perf_counter() - t0:.2f}s")

    if args.prompt is not None:
        conversation = [{"role": "user", "content": args.prompt}]
        prompt_with_template = ue.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True)
        prefill_seq = tuple(ue.tokenizer.encode(prompt_with_template, add_special_tokens=False))
        _original_print(f"Prompt ({len(prefill_seq)} tokens): {args.prompt!r}")
    else:
        prefill_seq = tuple(cfg["default_prefill_tokens"])
        _original_print(f"Default prompt ({len(prefill_seq)} tokens)")

    max_prefill = ue.PREFILL_CONTEXT_SIZE
    if len(prefill_seq) < 2 or len(prefill_seq) > max_prefill + 1:
        _original_print(f"WARNING: prompt length {len(prefill_seq)} out of range [2, {max_prefill + 1}]; "
                        f"falling back to the default prompt.")
        prefill_seq = tuple(cfg["default_prefill_tokens"])

    ue.prefill_seq = prefill_seq
    ue.temperature = float(args.temperature)
    ue.top_k = int(args.top_k)
    ue.top_p = float(args.top_p)
    ue.repetition_penalty = float(args.repetition_penalty)
    ue.rep_window = int(args.rep_window)
    ue.rep_decay = float(args.rep_decay)
    ue._generated_tokens = list(prefill_seq)
    if args.seed is not None and ue.temperature > 0:
        torch.manual_seed(args.seed)
    if ue.temperature > 0:
        _original_print(f"Sampling: temperature={ue.temperature}  top_k={ue.top_k}  top_p={ue.top_p}  "
                        f"repetition_penalty={ue.repetition_penalty}  rep_window={ue.rep_window}  "
                        f"rep_decay={ue.rep_decay}  seed={args.seed}")
        if ue.repetition_penalty != 1.0:
            _n = len(ue._structural_token_ids())  # precompute upfront (no mid-decode stall)
            _original_print(f"  repetition penalty exempts {_n} structural/special tokens")

    ue.run_llama()
    ue.clear_dram()
    _original_print("Llama-3.2-1B run_from_bin ends.")


if __name__ == "__main__":
    main()
