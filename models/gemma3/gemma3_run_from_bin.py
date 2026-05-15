#!/usr/bin/env python3
"""Gemma3 inference from pre-compiled bin files. Run gemma3_test.py first to generate bins."""
import builtins
import json
import mmap
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
    DMA_DEVICE_H2C, DRAM_INSTRUCTION_ADDR, TYPE, UE_FMAX_CONTEXT_SIZE, UE_VECTOR_SIZE,
    URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS,
    UnifiedEngine, set_dma_device, ue_35bit_addr_shifter,
)

def _set_silent(val: bool) -> None:
    global _SILENT_MODE
    _SILENT_MODE = val

def _parse_offset(val) -> int:
    if isinstance(val, str):
        return int(val, 0)
    return int(val)

BIN_DIR = os.path.join(SCRIPT_DIR, "gemma3_bin")


def check_bins_early(weights_bin_full, meta_path, instruction_bin_path):
    missing = []
    if not os.path.exists(weights_bin_full):
        missing.append(os.path.relpath(weights_bin_full, SCRIPT_DIR))
    if not os.path.exists(meta_path):
        missing.append(os.path.relpath(meta_path, SCRIPT_DIR))
    if not os.path.exists(instruction_bin_path):
        missing.append(os.path.relpath(instruction_bin_path, SCRIPT_DIR))
    return missing


class Gemma3_UnifiedEngine(UnifiedEngine):

    def __init__(self, script_dir=None, weights_bin=None):
        super().__init__(
            BASE_ADDR=user_dma_core.UE_0_BASE_ADDR,
            program_dram_base=DRAM_INSTRUCTION_ADDR,
        )
        self.script_dir = script_dir or SCRIPT_DIR
        self._cfg = self._load_config()
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
        self.causal_mask_upper = False
        self._rope_global_layers = set(model["rope_global_layers"])
        self._end_of_turn_token_id = model["end_of_turn_token_id"]
        self.seq_len = 0
        self._instruction_program_addr = None

        bin_path = weights_bin or paths["weights_bin"]
        full_path = os.path.join(self.script_dir, bin_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Weights bin not found: {full_path}")
        self._weight_bin_file = open(full_path, "rb")
        self.weight_bin = mmap.mmap(self._weight_bin_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.weight_init()
        self.tensor_init()

    def _load_config(self) -> dict:
        config_path = os.path.join(self.script_dir, "gemma3_config.json")
        with open(config_path) as f:
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

    # ------------------------------------------------------------------
    # ISA register helpers
    # ------------------------------------------------------------------
    def clear_inst_id(self) -> None:
        self._inst_id = 0

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

    # ------------------------------------------------------------------
    # Instruction loader
    # ------------------------------------------------------------------
    def load_instructions(self, bin_path: str) -> tuple[int, int]:
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
    def _load_rope_host(self) -> None:
        rope_cfg = self._cfg["special"]["rope"]
        theta = rope_cfg["theta"]
        local_base = rope_cfg["local_base"]
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

    # ------------------------------------------------------------------
    # Weight init — reads embedding directly from weights bin (no HF load)
    # ------------------------------------------------------------------
    def weight_init(self) -> None:
        emb_cfg = self._cfg["special"]["embedding"]
        token_embd_offset = _parse_offset(emb_cfg["token_embd_offset"])
        token_embd_size = _parse_offset(emb_cfg["token_embd_size"])
        vocab_size = emb_cfg["vocab_size"]
        emb_dim = emb_cfg["embedding_dim"]
        raw_emb = bytes(self.weight_bin[token_embd_offset : token_embd_offset + token_embd_size])
        self.embedding_weight = torch.frombuffer(raw_emb, dtype=torch.bfloat16).reshape(vocab_size, emb_dim).clone()

        hf_model_dir = os.path.join(self.script_dir, self._cfg["paths"]["hf_model_dir"])
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, trust_remote_code=True)

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
        self.dma_write(DMA_DEVICE_H2C, layers_base_dram,
                       self.weight_bin[base_layer0 : base_layer0 + layers_total],
                       layers_total)
        for off_key, sz_key, attr in blk0_regions:
            offset_in_layer = self.weight_defs[off_key] - base_layer0
            setattr(self, attr, layers_base_dram + offset_in_layer)

        for off_key, sz_key, attr in non_layer:
            off = self.weight_defs[off_key]
            sz = self.weight_defs[sz_key]
            raw = self.weight_bin[off : off + sz]
            addr = self.allocate_params_dram(sz)
            self.dma_write(DMA_DEVICE_H2C, addr, raw, sz)
            setattr(self, attr, addr)

        self._load_rope_host()

        # Release mmap and file handle — weights are in DRAM now
        self.weight_bin.close()
        self._weight_bin_file.close()
        del self.weight_bin, self._weight_bin_file

    # ------------------------------------------------------------------
    # Tensor init
    # ------------------------------------------------------------------
    def tensor_init(self) -> None:
        seq_len = self.MAX_CONTEXT_SIZE
        q_seq_len = seq_len * self.group_size
        aligned_seq_len_q = ((q_seq_len + 63) // 64) * 64

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

        self.LAYER0_FLASH_Q_DRAM = self.allocate_tensor_dram(aligned_seq_len_q * self.head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_K_DRAM = self.allocate_tensor_dram(aligned_seq_len_q * self.head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_V_DRAM = self.allocate_tensor_dram(aligned_seq_len_q * self.head_dim * self.bytes_per_element)
        zero_pad = torch.zeros(aligned_seq_len_q * self.head_dim * self.bytes_per_element, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_Q_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_K_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_V_DRAM, zero_pad)

        self.LAYER0_INPUT_DRAM             = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_NORM_DRAM          = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_Q_DRAM                 = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_K_DRAM                 = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_K_NORM_DRAM            = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_Q_NORM_DRAM            = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_FLASH_OUTPUT_DRAM      = self.allocate_tensor_dram(aligned_seq_len_q * self.head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_SCRATCH_DRAM     = self.allocate_tensor_dram(max(self.head_dim, UE_FMAX_CONTEXT_SIZE) * aligned_seq_len_q * 2 + self.head_dim * aligned_seq_len_q * 2)
        self.LAYER0_FLASH_BIAS_DRAM        = self.allocate_tensor_dram(aligned_seq_len_q * aligned_seq_len_q * self.bytes_per_element)
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
        self.OUTPUT_NORM_DRAM              = self.allocate_tensor_dram(1 * self.vector_length * self.bytes_per_element)
        self.LOGITS_DRAM                   = self.allocate_tensor_dram(1 * self.EMBEDDING_ELEMENTS * self.bytes_per_element)

    # ------------------------------------------------------------------
    # Program execute helper
    # ------------------------------------------------------------------
    def program_execute(self, program_start_addr, timeout=50.0, flops=None):
        self.start_execute_from_dram(program_start_addr)
        latency, flop_rate = 0, 0
        if timeout != 0:
            self.wait_queue(timeout)
            latency = self.report_latency_in_us()
            if flops is not None:
                flop_rate = self.report_flop_rate_gflops(flops)
        return latency, flop_rate

    # ------------------------------------------------------------------
    # Run prefill + decoder
    # ------------------------------------------------------------------
    def run_gemma3(self, prefill_seq: tuple) -> None:
        meta_path = os.path.join(self.script_dir, self._cfg["paths"]["instruction_meta"])
        with open(meta_path) as f:
            meta = json.load(f)

        instruction_bin = os.path.join(self.script_dir, meta["instruction_bin"])
        if self._instruction_program_addr is None:
            self._instruction_program_addr, _ = self.load_instructions(instruction_bin)

        kv_size = self.LAYER_SIZE * self.MAX_CONTEXT_SIZE * self.k_size
        zero_kv = torch.zeros(kv_size, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, zero_kv)
        self.dma_to_accelerator_memory(self.LAYER0_K_ROPE_DRAM, zero_kv)

        prefill_buckets = list(self._cfg["model"]["prefill_seq_len_buckets"])
        prefill_program_addrs = [_parse_offset(a) for a in meta["prefill_program_start_addrs"]]
        prefill_flops_list = meta["prefill_total_flops"]
        decoder_program_addrs = [_parse_offset(a) for a in meta["decoder_program_start_addrs"]]
        flops_per_token = meta["decoder_total_flops"]

        if len(prefill_seq) <= 1:
            raise ValueError("Prefill sequence must have at least 2 tokens.")
        prefill_tokens = prefill_seq[:-1]
        prefill_seq_len = len(prefill_tokens)
        max_bucket = prefill_buckets[-1]
        if prefill_seq_len > max_bucket:
            raise ValueError(
                f"prefill_seq_len={prefill_seq_len} exceeds largest compiled bucket ({max_bucket}). Recompile."
            )
        bucket_idx = next(i for i, b in enumerate(prefill_buckets) if b >= prefill_seq_len)
        bucket_seq_len = prefill_buckets[bucket_idx]
        prefill_program_addr = prefill_program_addrs[bucket_idx]
        self.seq_len = prefill_seq_len

        q_seq_len = bucket_seq_len * self.group_size
        aligned_seq_len_q = ((q_seq_len + 63) // 64) * 64

        _original_print(f"\n--- Starting prefill (seq_len={prefill_seq_len}, bucket={bucket_seq_len}) ---")
        timer = time.perf_counter()
        embedding_tensor = self.get_embedding_for_tokens(prefill_tokens)
        if bucket_seq_len > prefill_seq_len:
            pad = embedding_tensor[-1:].repeat(bucket_seq_len - prefill_seq_len, 1)
            embedding_tensor = torch.cat([embedding_tensor, pad], dim=0)
        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
        bias = torch.full((aligned_seq_len_q, aligned_seq_len_q), float("-inf"), dtype=torch.bfloat16)
        valid_mask = torch.tril(torch.ones(aligned_seq_len_q, aligned_seq_len_q, dtype=torch.bool))
        bias.masked_fill_(valid_mask, 0.0)
        bias[:, q_seq_len:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias)
        latency_hw_prefill, _ = self.program_execute(prefill_program_addr, flops=prefill_flops_list[bucket_idx])
        if bucket_seq_len > prefill_seq_len:
            zero_kv = torch.zeros((bucket_seq_len - prefill_seq_len) * self.head_dim, dtype=torch.bfloat16)
            pad_byte_offset = prefill_seq_len * self.k_size
            for layer_idx in range(self.LAYER_SIZE):
                layer_byte_offset = layer_idx * self.MAX_CONTEXT_SIZE * self.k_size
                self.dma_to_accelerator_memory(self.LAYER0_K_ROPE_DRAM + layer_byte_offset + pad_byte_offset, zero_kv)
                self.dma_to_accelerator_memory(self.LAYER0_V_DRAM + layer_byte_offset + pad_byte_offset, zero_kv)
        _original_print(f"  Done in {time.perf_counter() - timer:.2f}s")

        _original_print(f"\n--- Starting decoder ---")
        timer = time.perf_counter()
        token_id = prefill_seq[-1]
        total_latency = 0
        decoder_token_cnt = 0
        max_seq_len = self.MAX_CONTEXT_SIZE

        while self.seq_len < max_seq_len:
            decoder_token_cnt += 1
            _set_silent(True)
            self.seq_len += 1
            aligned_seq_len_q = ((self.seq_len + 63) // 64) * 64
            prog_idx = min((self.seq_len - 1) // 64, 7)
            prog_addr = decoder_program_addrs[prog_idx]
            flops = flops_per_token[prog_idx] if flops_per_token else None

            self.isa_add_set_core(self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter((self.seq_len - 1) * self.k_size))
            self.isa_add_set_core(self.ROPE_SIZE_REG, ue_35bit_addr_shifter((self.seq_len - 1) * self.k_size * 2))

            embedding_tensor = self.get_embedding_for_tokens([token_id])
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)
            bias_host = torch.full((1, aligned_seq_len_q), -1e36, dtype=torch.bfloat16)
            bias_host[0, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_host)

            latency, _ = self.program_execute(prog_addr, flops=flops)
            total_latency += latency
            token_id = self.get_arg_max_index()
            token_char = self.tokenizer.decode([token_id])
            _set_silent(False)
            if token_id in [1, self._end_of_turn_token_id]:
                _original_print(f"\nStop token {token_id} reached.")
                break
            _original_print(token_char, end="", flush=True)

        latency_decoder = time.perf_counter() - timer
        _original_print(
            f"\nDecoder done in {latency_decoder:.2f}s, "
            f"speed: {decoder_token_cnt / latency_decoder:.2f} tokens/s, "
            f"total {self.seq_len} tokens."
        )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Gemma3 inference from pre-compiled bins")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--local-weights", action="store_true")
    parser.add_argument("--dev", type=str, default="xdma0")
    parser.add_argument("--cycle", type=float, default=5.62)
    args = parser.parse_args()

    if args.local_weights:
        weights_bin_rel = "gemma3_bin/full_model_weights.bin"
    else:
        weights_bin_rel = None  # uses config default

    cfg_path = os.path.join(SCRIPT_DIR, "gemma3_config.json")
    with open(cfg_path) as f:
        cfg_raw = json.load(f)
    paths = cfg_raw["paths"]
    weights_bin_full = os.path.join(SCRIPT_DIR, weights_bin_rel or paths["weights_bin"])
    meta_path = os.path.join(SCRIPT_DIR, paths["instruction_meta"])
    instruction_bin_path = os.path.join(SCRIPT_DIR, paths["instruction_bin"])

    missing = check_bins_early(weights_bin_full, meta_path, instruction_bin_path)
    if missing:
        _original_print("Missing bin files (run gemma3_test.py --smax 512 first to compile):")
        for f in missing:
            _original_print(f"  {f}")
        return

    set_dma_device(args.dev)
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle

    _set_silent(True)
    ue = Gemma3_UnifiedEngine(script_dir=SCRIPT_DIR, weights_bin=weights_bin_rel)
    ue.software_reset()
    _set_silent(False)

    if args.prompt is not None:
        conversation = [{"role": "user", "content": args.prompt}]
        prompt_with_template = ue.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        prefill_seq = tuple(ue.tokenizer.encode(prompt_with_template, add_special_tokens=True))
        _original_print(f"Prompt ({len(prefill_seq)} tokens): {args.prompt!r}")
    else:
        prefill_seq = tuple(cfg_raw["default_prefill_tokens"])
        _original_print(f"Using default prefill ({len(prefill_seq)} tokens)")

    ue.run_gemma3(prefill_seq=prefill_seq)


if __name__ == "__main__":
    main()
