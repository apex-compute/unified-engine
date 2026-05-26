#!/usr/bin/env python3
"""Qwen3-4B inference from pre-compiled bin files.

Load-and-execute only — no compile, no HF download. The cached single-bin
(``qwen3_4b_instruction.bin`` + matching ``.json`` meta) plus the
weights bin and tokenizer files must already exist on disk. Run
``qwen3_4b_test.py`` once on a build machine with HF access to produce
them; this script then runs entirely offline.

Bin layout (matches ``compile_instructions`` in qwen3_4b_test.py):

    [prefill program] [decoder program]

Each is one seq_len-agnostic program; the runtime preamble in
:meth:`run_prefill` / :meth:`run_decoder` primes ``gf_seq_len`` /
``gf_q_seq_len`` / ``gf_bucket_idx`` GPRs and unconditional-jumps into
the appropriate program.
"""
import builtins
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

_original_print = builtins.print
_SILENT_MODE = False
def _quiet_print(*args, **kwargs):
    if not _SILENT_MODE:
        _original_print(*args, **kwargs)
builtins.print = _quiet_print

import torch
from transformers import AutoTokenizer

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, UE_FMAX_CONTEXT_SIZE, UE_VECTOR_SIZE, UE_ARGMAX_INDEX,
    UnifiedEngine, set_dma_device, ue_35bit_addr_shifter,
)
from model_lib_core import load_config_with_weight_defs


def _parse_offset(val):
    """Parse hex string or int from JSON."""
    return int(val, 0) if isinstance(val, str) else int(val)


BIN_DIR = os.path.join(SCRIPT_DIR, "qwen3_4b_bin")


def check_bins_early(weights_bin_full):
    """Hard-fail BEFORE any FPGA / HF touch if a required local file is missing.

    Runtime-only script — never reaches the network. The customer ships
    these files alongside the script, produced once by running
    ``qwen3_4b_test.py`` on a build machine with HF access.
    """
    missing = []
    if not os.path.exists(weights_bin_full):
        missing.append(os.path.relpath(weights_bin_full, SCRIPT_DIR))
    for name in ("qwen3_4b_instruction.bin", "qwen3_4b_instruction.json"):
        if not os.path.exists(os.path.join(BIN_DIR, name)):
            missing.append(name)
    tokenizer_dir = os.path.join(BIN_DIR, "Qwen3-4B")
    if not (os.path.exists(os.path.join(tokenizer_dir, "tokenizer.json")) or
            os.path.exists(os.path.join(tokenizer_dir, "tokenizer_config.json"))):
        missing.append("Qwen3-4B/{tokenizer.json,tokenizer_config.json}")
    return missing


class Qwen3_4b_UnifiedEngine(UnifiedEngine):

    def __init__(self, script_dir=None, weights_bin=None):
        # See qwen3_4b_test.py for DRAM layout rationale.
        super().__init__(
            params_dram_base=0x00000000,
            tensor_dram_base=0x90000000,
            program_dram_base=0xE0000000,
        )
        self.script_dir = script_dir or SCRIPT_DIR
        self._cfg = load_config_with_weight_defs(os.path.join(self.script_dir, "qwen3_4b_config.json"))
        self.weight_defs = self._cfg["_weight_defs"]

        fi = self._cfg["file_info"]
        model = self._cfg["model"]
        paths = self._cfg["paths"]

        self.vector_length = fi["hidden_size"]
        self.head_dim = fi["head_dim"]
        self.actual_head_dim = fi["actual_head_dim"]
        self.num_kv_heads = fi["num_kv_heads"]
        self.bytes_per_element = fi["bytes_per_element"]
        self.group_size = fi["group_size"]
        self.mlp_elements = fi["mlp_elements"]
        self.hf_model_dir = os.path.join(self.script_dir, paths["hf_model_dir"])
        self.q_size = self.head_dim * self.group_size * self.bytes_per_element
        self.k_size = self.head_dim * self.bytes_per_element
        self.MAX_CONTEXT_SIZE = model["max_context_size"]
        self.PREFILL_MAX_SEQ_LEN = model.get("prefill_max_seq_len", 64)
        self.LAYER_SIZE = fi["num_layers"]
        self.EMBEDDING_ELEMENTS = fi["embedding_vocab"]

        # Fixed dynamic-PBI GPRs (mirror qwen3_4b_test.py / compare_qwen3_4b.py).
        # Primed by the runtime preamble before jumping into the cached programs.
        fixed = self._cfg["fixed_isa_regs"]
        self.TMP_REG          = fixed["TMP_REG"]
        self.gf_seq_len       = fixed["GF_SEQ_LEN_REG"]
        self.gf_q_seq_len     = fixed["GF_Q_SEQ_LEN_REG"]
        self.gf_bucket_idx    = fixed["GF_BUCKET_IDX_REG"]

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

    # ------------------------------------------------------------------
    # DRAM allocator override (matches test.py)
    # ------------------------------------------------------------------
    def allocate_params_dram(self, size_bytes):
        params_dram_addr = self._next_params_dram_addr
        self._next_params_dram_addr += size_bytes
        return params_dram_addr

    def clear_inst_id(self):
        self._inst_id = 0

    # ------------------------------------------------------------------
    # Instruction loader
    # ------------------------------------------------------------------
    def load_instructions(self, bin_path):
        with open(bin_path, "rb") as f:
            data = f.read()
        total_size = len(data)
        start_addr = self.allocate_program_dram(total_size)
        self.dma_write(DMA_DEVICE_H2C, start_addr, data, total_size)
        return start_addr, total_size

    # ------------------------------------------------------------------
    # Embedding lookup
    # ------------------------------------------------------------------
    def get_embedding_for_tokens(self, token_ids):
        tid_t = torch.tensor(token_ids, dtype=torch.long)
        out = torch.zeros(len(token_ids), self.vector_length, dtype=torch.bfloat16)
        valid = tid_t < self.embedding_weight.shape[0]
        out[valid] = self.embedding_weight[tid_t[valid]]
        return out

    # ------------------------------------------------------------------
    # RoPE table (host-computed, DMA'd to params DRAM)
    # ------------------------------------------------------------------
    def _load_rope_host(self, rope_theta=None):
        rope_cfg = self._cfg["special"]["rope"]
        theta = rope_theta if rope_theta is not None else rope_cfg["theta"]
        num_rope_positions = rope_cfg["num_positions"]
        D_per_head = self.actual_head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(D_per_head, dtype=torch.float32) / D_per_head))
        pos = torch.arange(num_rope_positions, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        cos_ = freqs.cos().to(torch.bfloat16)
        sin_ = freqs.sin().to(torch.bfloat16)
        rope_tensor = torch.cat([cos_, cos_, -sin_, sin_], dim=1)
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

    # ------------------------------------------------------------------
    # Weight init — reads embedding directly from weights bin (no HF model load)
    # ------------------------------------------------------------------
    def weight_init(self):
        emb_cfg = self._cfg["special"]["embedding"]
        token_embd_offset = _parse_offset(emb_cfg["token_embd_offset"])
        token_embd_size   = _parse_offset(emb_cfg["token_embd_size"])
        vocab_size   = emb_cfg["vocab_size"]
        emb_dim      = emb_cfg["embedding_dim"]
        emb_bytes = vocab_size * emb_dim * self.bytes_per_element
        raw_emb = bytearray(self.weight_bin[token_embd_offset : token_embd_offset + emb_bytes])
        self.embedding_weight = torch.frombuffer(raw_emb, dtype=torch.bfloat16).reshape(vocab_size, emb_dim).clone()
        # ``local_files_only=True``: never reach HF. check_bins_early() already
        # validated the tokenizer files exist; this is the second line of defense.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_dir, trust_remote_code=True, local_files_only=True,
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
            if s["key"] not in ("ROPE_LOCAL", "ROPE_GLOBAL")
        ]

        layers_total = self.LAYER_SIZE * LAYER_WEIGHT_SIZE
        layers_base_dram = self.allocate_params_dram(layers_total)
        bin_layers_start = base_layer0
        self.dma_write(DMA_DEVICE_H2C, layers_base_dram,
                       self.weight_bin[bin_layers_start : bin_layers_start + layers_total],
                       layers_total)
        for off_key, sz_key, attr in blk0_regions:
            offset_in_layer = self.weight_defs[off_key] - base_layer0
            setattr(self, attr, layers_base_dram + offset_in_layer)

        nl_slices = [self.weight_bin[self.weight_defs[k] : self.weight_defs[k] + self.weight_defs[s]]
                     for k, s, _ in non_layer]
        nl_buf = b"".join(nl_slices)
        nl_base_dram = self.allocate_params_dram(len(nl_buf))
        self.dma_write(DMA_DEVICE_H2C, nl_base_dram, nl_buf, len(nl_buf))
        nl_offset = 0
        for off_key, sz_key, attr in non_layer:
            setattr(self, attr, nl_base_dram + nl_offset)
            nl_offset += self.weight_defs[sz_key]

        self._load_rope_host()

    # ------------------------------------------------------------------
    # Tensor init (mirror test.py, with LAYER0_FLASH_ATTN_P_DRAM for new flash PBI)
    # ------------------------------------------------------------------
    def tensor_init(self):
        seq_len = self.MAX_CONTEXT_SIZE
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        ahd  = self.actual_head_dim
        nkvh = self.num_kv_heads
        bpe  = self.bytes_per_element

        zero_add = torch.zeros(seq_len * self.head_dim * bpe, dtype=torch.bfloat16)
        self.ZERO_DRAM_ADDR = self.allocate_tensor_dram(seq_len * self.head_dim * bpe)
        self.dma_to_accelerator_memory(self.ZERO_DRAM_ADDR, zero_add)
        self.IDENTITY_DRAM_ADDR = self.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe)
        self.dma_to_accelerator_memory(self.IDENTITY_DRAM_ADDR, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))
        # Flash-attention bucket-dispatcher scratch buffer for prefill's
        # PBI flash kernel (ATTN_P_DRAM). Sized for the largest aligned Q
        # seq this engine instance will see (= PREFILL_MAX_SEQ_LEN * group_size).
        aligned_q_max = ((self.PREFILL_MAX_SEQ_LEN * self.group_size + UE_VECTOR_SIZE - 1)
                         // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
        self.LAYER0_FLASH_ATTN_P_DRAM = self.allocate_tensor_dram(aligned_q_max * aligned_q_max * bpe)

        self.LAYER0_FLASH_Q_DRAM = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        self.LAYER0_FLASH_K_DRAM = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        self.LAYER0_FLASH_V_DRAM = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        zero_pad = torch.zeros(aligned_seq_len * ahd * bpe, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_Q_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_K_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_V_DRAM, zero_pad)

        self.LAYER0_FLASH_OUT_HEAD_DRAM    = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        self.LAYER0_FLASH_OUTPUT_DRAM      = self.allocate_tensor_dram(seq_len * self.head_dim * self.group_size * bpe)
        self.LAYER0_FLASH_SCRATCH_DRAM     = self.allocate_tensor_dram(max(ahd, UE_FMAX_CONTEXT_SIZE) * aligned_seq_len * 2 + ahd * aligned_seq_len * 2)
        self.LAYER0_FLASH_BIAS_DRAM        = self.allocate_tensor_dram(aligned_seq_len * aligned_seq_len * bpe)
        self.LAYER0_INPUT_DRAM             = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_NORM_DRAM          = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_Q_DRAM                 = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_K_DRAM                 = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_K_NORM_DRAM            = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_Q_NORM_DRAM            = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_V_PROJ_TEMP            = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_ATTN_PROJ_OUTPUT_DRAM  = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_ATTN_NORM_DRAM    = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_ATTN_RESIDUAL_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_MLP_NORM_DRAM      = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_MLP_GATE_DRAM          = self.allocate_tensor_dram(seq_len * self.mlp_elements * 2)
        self.LAYER0_MLP_UP_DRAM            = self.allocate_tensor_dram(seq_len * self.mlp_elements * 2)
        self.LAYER0_MLP_MULT_DRAM          = self.allocate_tensor_dram(seq_len * self.mlp_elements * 2)
        self.LAYER0_MLP_DOWN_DRAM          = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_MLP_NORM_DRAM     = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_OUTPUT_DRAM            = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.OUTPUT_NORM_DRAM              = self.allocate_tensor_dram(1 * self.vector_length * bpe)
        self.LOGITS_DRAM                   = self.allocate_tensor_dram(1 * self.EMBEDDING_ELEMENTS * bpe)

        kv_cache_total = self.LAYER_SIZE * nkvh * self.MAX_CONTEXT_SIZE * ahd * bpe
        self.LAYER0_V_DRAM       = self.allocate_tensor_dram(kv_cache_total)
        self.LAYER0_K_ROPE_DRAM  = self.allocate_tensor_dram(kv_cache_total)
        zero_kv = torch.zeros(kv_cache_total, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, zero_kv)
        self.dma_to_accelerator_memory(self.LAYER0_K_ROPE_DRAM, zero_kv)

    # ------------------------------------------------------------------
    # Program execute helper
    # ------------------------------------------------------------------
    def program_execute(self, program_start_addr, timeout=10.0, gflops=None):
        """Execute a program from DRAM. If ``gflops`` is supplied, print latency + GFLOPS."""
        self.start_execute_from_dram(program_start_addr)
        self.wait_queue(timeout)
        if gflops is not None:
            latency = self.report_latency_in_us()
            gflops_program, _ = self.report_flop_rate_gflops(gflops)
            # High-precision so we can sanity-check vs peak ( = 128 / clock_ns ).
            _original_print(f"    HW latency = {latency:.1f} us, {gflops_program:.4f} GFLOPS "
                            f"(num_flops={int(gflops)}, clock_ns={self._clock_period_ns})")

    # ------------------------------------------------------------------
    # Run prefill via runtime preamble + jump_abs (matches test.py)
    # ------------------------------------------------------------------
    def run_prefill(self, prefill_program_addr, preamble_addr, prefill_seq, gflops=None):
        if len(prefill_seq) <= 1:
            raise ValueError("Prefill sequence must have at least 2 tokens.")
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

        # Defeat stale-NaN: the prefill program scatters all template_seq_len
        # K/V/Q rows into the cache, but only the first actual_seq_len were
        # computed this run. Zero the leading region so slots
        # [actual_seq_len..template_seq_len) become 0 instead of stale DRAM bits
        # (which can be NaN from previous runs and cascade through attention).
        ahd  = self.actual_head_dim
        nkvh = self.num_kv_heads
        qpkv = self.group_size
        template = self.PREFILL_MAX_SEQ_LEN
        zero_kvq = torch.zeros(template * nkvh * qpkv * ahd, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_K_NORM_DRAM, zero_kvq[: template * nkvh * ahd])
        self.dma_to_accelerator_memory(self.LAYER0_Q_NORM_DRAM, zero_kvq[: template * nkvh * qpkv * ahd])
        self.dma_to_accelerator_memory(self.LAYER0_V_PROJ_TEMP, zero_kvq[: template * nkvh * ahd])

        embedding_tensor = self.get_embedding_for_tokens(prefill_seq)
        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
        bias_one_group = torch.full((aligned_seq_len, aligned_seq_len), float("-inf"), dtype=torch.bfloat16)
        valid_mask = torch.tril(torch.ones(aligned_seq_len, aligned_seq_len, dtype=torch.bool), diagonal=0) if not self.causal_mask_upper else torch.triu(torch.ones(aligned_seq_len, aligned_seq_len, dtype=torch.bool), diagonal=0)
        bias_one_group.masked_fill_(valid_mask, 0.0)
        bias_one_group[:, q_seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_one_group)

        # Runtime preamble: prime the 3 GPRs and JUMP_ABS into cached prefill.
        self.clear_inst_id()
        self.start_capture()
        self.generate_instruction_add_set(self.gf_seq_len,    actual_seq_len)
        self.generate_instruction_add_set(self.gf_q_seq_len,  q_seq_len)
        self.generate_instruction_add_set(self.gf_bucket_idx, bucket_idx)
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(prefill_program_addr))
        self.stop_capture()
        self.write_captured_instructions_to_dram(preamble_addr)
        self.clear_capture_buffer()
        self.program_execute(preamble_addr, gflops=gflops)

    # ------------------------------------------------------------------
    # Run decoder loop via per-step preamble (matches test.py)
    # ------------------------------------------------------------------
    def run_decoder(self, decoder_program_addr, preamble_addr, token_id, gflops_per_token=None):
        if token_id is None:
            return 0

        _qwen3_stop_tokens = {151643, 151645, self._end_of_turn_token_id}
        num_buckets_decoder = (self.MAX_CONTEXT_SIZE + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
        max_bucket_seq_len = num_buckets_decoder * UE_VECTOR_SIZE
        max_seq_len = self.MAX_CONTEXT_SIZE

        while self.seq_len < max_seq_len:
            _set_silent(True)
            decode_pos = self.seq_len
            new_ctx_len = decode_pos + 1
            aligned_ctx = ((new_ctx_len + 63) // 64) * 64
            bucket_idx = max(1, aligned_ctx // UE_VECTOR_SIZE)

            embedding_tensor = self.get_embedding_for_tokens([token_id])
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
            bias_host = torch.full((1, max_bucket_seq_len), -1e36, dtype=torch.bfloat16)
            bias_host[0, :new_ctx_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_host)

            # Decode preamble: prime only gf_bucket_idx; gf_seq_len carries over
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
            token_id = self.read_reg32(UE_ARGMAX_INDEX)
            token_char = self.tokenizer.decode([token_id])
            self.seq_len += 1
            _set_silent(False)
            if token_id in _qwen3_stop_tokens:
                print(f"\nStop token {token_id} reached.")
                break
            print(token_char, end="", flush=True)
        return self.seq_len


def _set_silent(val):
    global _SILENT_MODE
    _SILENT_MODE = val


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3-4B inference from pre-compiled bins (offline)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt (default: from qwen3_4b_config.json default_prompt)")
    parser.add_argument("--local-weights", action="store_true",
                        help="Use qwen3_4b_bin/full_model_weights.bin instead of weights_qwen3_4b_hf.bin")
    parser.add_argument("--dev", type=str, default="xdma0", help="DMA device name (default: xdma0)")
    parser.add_argument("--cycle", type=float, default=5.62,
                        help="Clock cycle time in ns (default: 5.62ns ≈ peak 22.8 GFLOPS)")
    args = parser.parse_args()

    weights_bin_rel = ("qwen3_4b_bin/full_model_weights.bin" if args.local_weights
                       else "qwen3_4b_bin/weights_qwen3_4b_hf.bin")
    weights_bin_full = os.path.join(SCRIPT_DIR, weights_bin_rel)

    missing = check_bins_early(weights_bin_full)
    if missing:
        _original_print("Missing local files (run qwen3_4b_test.py first on a build machine with HF access):")
        for f in missing:
            _original_print(f"  {f}")
        sys.exit(1)

    set_dma_device(args.dev)
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle
    user_dma_core.UE_PEAK_GFLOPS = 0.128 / args.cycle
    _original_print(f"Setting CLOCK_CYCLE_TIME_NS = {user_dma_core.CLOCK_CYCLE_TIME_NS}, "
                    f"UE_PEAK_GFLOPS = {user_dma_core.UE_PEAK_GFLOPS:.4f}")

    _set_silent(True)
    ue = Qwen3_4b_UnifiedEngine(script_dir=SCRIPT_DIR, weights_bin=weights_bin_rel)
    _set_silent(False)

    cfg = ue._cfg
    user_prompt = args.prompt if args.prompt is not None else cfg.get("default_prompt", "What is 3 + 5?")
    system_prompt = cfg.get("default_system_prompt", "You are a helpful assistant.")
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    prompt_with_template = ue.tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True,
    )
    prefill_seq = tuple(ue.tokenizer.encode(prompt_with_template, add_special_tokens=False))
    _original_print(f"User prompt ({len(prefill_seq)} tokens): {user_prompt!r}")

    paths_cfg = cfg.get("paths", {})
    inst_bin_path  = os.path.join(SCRIPT_DIR, paths_cfg.get("instruction_bin",
                                  "qwen3_4b_bin/qwen3_4b_instruction.bin"))
    inst_meta_path = os.path.join(SCRIPT_DIR, paths_cfg.get("instruction_meta",
                                  "qwen3_4b_bin/qwen3_4b_instruction.json"))
    with open(inst_meta_path) as f:
        meta = json.load(f)

    _original_print(f"\n--- Loading unified instruction bin ---")
    timer = time.perf_counter()
    base_addr, total_size = ue.load_instructions(inst_bin_path)
    # Reserve a preamble slot after the loaded bin. Prefill preamble = 4 instructions (128 B);
    # decode preamble = 2 instructions and overwrites the same slot.
    preamble_addr = ue.get_program_dram_addr()
    ue.allocate_program_dram(128)
    _original_print(f"  Loaded {total_size} B at 0x{base_addr:X}; preamble slot at 0x{preamble_addr:X} "
                    f"({time.perf_counter() - timer:.3f}s)")

    prefill_program_addr = _parse_offset(meta["prefill_program_start_addr"])
    decoder_program_addr = _parse_offset(meta["decoder_program_start_addr"])
    decoder_total_flops  = meta["decoder_total_flops"]

    actual_seq_len = len(prefill_seq) - 1
    # Rescale prefill FLOPs from compile-time template to actual seq_len so the
    # GFLOPS report reflects the real work this prompt did (the captured ops
    # run gf_seq_len iterations at runtime, not template_seq_len).
    template_seq_len = int(meta["prefill_template_seq_len"])
    gflops_prefill = meta["prefill_template_flops"] * actual_seq_len // max(template_seq_len, 1)
    _original_print(f"\n--- Starting prefill (actual {actual_seq_len} tokens, dynamic seq_len) ---")
    timer = time.perf_counter()
    ue.run_prefill(prefill_program_addr, preamble_addr, prefill_seq=prefill_seq, gflops=gflops_prefill)
    latency_prefill = time.perf_counter() - timer
    _original_print(f"  Prefill done in {latency_prefill:.2f}s")

    _original_print(f"\n--- Starting decoder ---")
    timer = time.perf_counter()
    token_cnt = ue.run_decoder(decoder_program_addr, preamble_addr,
                               token_id=prefill_seq[-1], gflops_per_token=decoder_total_flops)
    latency_decoder = time.perf_counter() - timer
    decoded_tokens = max(token_cnt - len(prefill_seq) + 1, 1)
    _original_print(f"\nDecoder done in {latency_prefill + latency_decoder:.2f}s, total {token_cnt} tokens, "
                    f"decode speed: {decoded_tokens / latency_decoder:.2f} tokens/s ({decoded_tokens} decoded tokens / {latency_decoder:.2f}s).")


if __name__ == "__main__":
    main()
