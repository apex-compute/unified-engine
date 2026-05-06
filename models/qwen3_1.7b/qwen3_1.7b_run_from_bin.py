#!/usr/bin/env python3
"""Qwen3-1.7B inference from pre-compiled bin files.

Compile first (auto-skipped if already compiled):
    python qwen3_1.7b_test.py

Then run:
    python qwen3_1.7b_run_from_bin.py --prompt "your prompt"
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
    DMA_DEVICE_H2C, TYPE, UE_FMAX_CONTEXT_SIZE, UE_VECTOR_SIZE,
    UE_ARGMAX_INDEX, URAM_NEAR_FULL_ELEMENTS, URAM_FULL_ELEMENTS,
    UnifiedEngine, set_dma_device, ue_35bit_addr_shifter,
)
from model_lib_core import load_config_with_weight_defs


def _set_silent(val: bool) -> None:
    global _SILENT_MODE
    _SILENT_MODE = val


def _parse_offset(val) -> int:
    if isinstance(val, str):
        return int(val, 0)
    return int(val)


def _load_config(script_dir: str) -> dict:
    config_path = os.path.join(script_dir, "qwen3_1.7b_config.json")
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


def check_bins_early(weights_bin_full, cfg):
    missing = []
    if not os.path.exists(weights_bin_full):
        missing.append(os.path.relpath(weights_bin_full, SCRIPT_DIR))
    paths = cfg.get("paths", {})
    for key, default in [("instruction_bin", "qwen3_1.7b_bin/qwen3_instruction.bin"),
                         ("instruction_meta", "qwen3_1.7b_bin/qwen3_instruction.json")]:
        p = os.path.join(SCRIPT_DIR, paths.get(key, default))
        if not os.path.exists(p):
            missing.append(os.path.relpath(p, SCRIPT_DIR))
    return missing


class Qwen3_1_7b_UnifiedEngine(UnifiedEngine):
    """Qwen3-1.7B inference from pre-compiled bins — no HF model at runtime."""

    def __init__(self, script_dir=None, weights_bin=None):
        self._identity_dram_written = False
        self._identity_dram_addr = None
        self._IDENTITY_MAT_BYTES = UE_VECTOR_SIZE * UE_VECTOR_SIZE * 2
        super().__init__(
            params_dram_base=0x00000000,
            tensor_dram_base=0x58000000,
            program_dram_base=0x98000000,
        )
        self.script_dir = script_dir or SCRIPT_DIR
        self._cfg = _load_config(self.script_dir)
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

        bin_path = weights_bin or paths["weights_bin"]
        full_path = os.path.join(self.script_dir, bin_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Weights bin not found: {full_path}")
        with open(full_path, "rb") as f:
            self.weight_bin = f.read()
        self.weight_init()
        self.tensor_init()

    _IDENTITY_MAT_BYTES = UE_VECTOR_SIZE * UE_VECTOR_SIZE * 2

    def _preallocate_identity_matrix(self) -> None:
        if self._identity_dram_addr is not None:
            return
        self._identity_dram_addr = super().allocate_params_dram(self._IDENTITY_MAT_BYTES)
        eye = torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16)
        super().dma_write(DMA_DEVICE_H2C, self._identity_dram_addr, eye, self._IDENTITY_MAT_BYTES)
        self._identity_dram_written = True

    def dma_write(self, device, addr, data, size):
        if (self._identity_dram_written
                and self._identity_dram_addr is not None
                and addr == self._identity_dram_addr
                and size == self._IDENTITY_MAT_BYTES):
            return
        super().dma_write(device, addr, data, size)

    def reset_isa_reg_counter(self) -> None:
        self._isa_reg_counter = 1

    def alloc_isa_reg(self, reset: bool = False) -> int:
        if reset:
            self._isa_reg_counter = 1
        if self._isa_reg_counter > 15:
            raise ValueError("Exceeded available ISA registers (max 15)")
        reg_idx = self._isa_reg_counter
        self._isa_reg_counter += 1
        return reg_idx

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
        with open(bin_path, "rb") as f:
            data = f.read()
        total_size = len(data)
        start_addr = self.allocate_program_dram(total_size)
        self.dma_write(DMA_DEVICE_H2C, start_addr, data, total_size)
        _original_print(f"    Loaded {total_size} bytes from instruction.bin to DRAM at 0x{start_addr:x}")
        return start_addr, total_size

    def allocate_params_dram(self, size_bytes: int) -> int:
        params_dram_addr = self._next_params_dram_addr
        self._next_params_dram_addr += size_bytes
        return params_dram_addr

    def clear_inst_id(self) -> None:
        self._inst_id = 0

    def get_arg_max_index(self) -> int:
        return self.read_reg32(UE_ARGMAX_INDEX)

    def get_embedding_for_tokens(self, token_ids) -> torch.Tensor:
        tid_t = torch.tensor(token_ids, dtype=torch.long)
        out = torch.zeros(len(token_ids), self.vector_length, dtype=torch.bfloat16)
        valid = tid_t < self.embedding_weight.shape[0]
        out[valid] = self.embedding_weight[tid_t[valid]]
        return out

    def _load_rope_host(self) -> None:
        rope_cfg = self._cfg["special"]["rope"]
        theta = rope_cfg["theta"]
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

    def weight_init(self) -> None:
        emb_cfg = self._cfg["special"]["embedding"]
        token_embd_offset = int(emb_cfg["token_embd_offset"], 16)
        vocab_size = emb_cfg["vocab_size"]
        emb_dim    = emb_cfg["embedding_dim"]
        emb_bytes  = vocab_size * emb_dim * self.bytes_per_element
        raw_emb = bytearray(self.weight_bin[token_embd_offset : token_embd_offset + emb_bytes])
        self.embedding_weight = torch.frombuffer(raw_emb, dtype=torch.bfloat16).reshape(vocab_size, emb_dim).clone()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_dir, trust_remote_code=True)

        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        base_layer0 = self.weight_defs["BLK0_ATTN_NORM_WEIGHT"]
        blk0_regions = [(s["key"], f"{s['key']}_SIZE", s["attr"]) for s in self._cfg["layers"]["structure"]]
        non_layer = [(s["key"], f"{s['key']}_SIZE", s["attr"]) for s in self._cfg["layers"]["non_layer"]
                     if s["key"] not in ("ROPE_LOCAL", "ROPE_GLOBAL")]

        layers_total = self.LAYER_SIZE * LAYER_WEIGHT_SIZE
        layers_base_dram = self.allocate_params_dram(layers_total)
        self.dma_write(DMA_DEVICE_H2C, layers_base_dram,
                       self.weight_bin[base_layer0 : base_layer0 + layers_total], layers_total)
        for off_key, sz_key, attr in blk0_regions:
            setattr(self, attr, layers_base_dram + self.weight_defs[off_key] - base_layer0)

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

    def tensor_init(self) -> None:
        seq_len = self.MAX_CONTEXT_SIZE
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64
        ahd = self.actual_head_dim
        nkvh = self.num_kv_heads
        bpe = self.bytes_per_element

        zero_add = torch.zeros(seq_len * self.head_dim * bpe, dtype=torch.bfloat16)
        self.ZERO_DRAM_ADDR = self.allocate_tensor_dram(seq_len * self.head_dim * bpe)
        self.dma_to_accelerator_memory(self.ZERO_DRAM_ADDR, zero_add)
        self.IDENTITY_DRAM_ADDR = self.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe)
        self.dma_to_accelerator_memory(self.IDENTITY_DRAM_ADDR, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))

        self.LAYER0_FLASH_Q_DRAM = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        self.LAYER0_FLASH_K_DRAM = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        self.LAYER0_FLASH_V_DRAM = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        zero_pad = torch.zeros(aligned_seq_len * ahd * bpe, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_Q_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_K_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_V_DRAM, zero_pad)

        self.LAYER0_FLASH_OUT_HEAD_DRAM = self.allocate_tensor_dram(aligned_seq_len * ahd * bpe)
        self.LAYER0_FLASH_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.head_dim * self.group_size * bpe)
        self.LAYER0_FLASH_SCRATCH_DRAM = self.allocate_tensor_dram(max(ahd, UE_FMAX_CONTEXT_SIZE) * aligned_seq_len * 2 + ahd * aligned_seq_len * 2)
        self.LAYER0_FLASH_BIAS_DRAM = self.allocate_tensor_dram(aligned_seq_len * aligned_seq_len * bpe)

        self.LAYER0_INPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_Q_DRAM = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_K_DRAM = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_K_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_Q_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_V_PROJ_TEMP = self.allocate_tensor_dram(seq_len * self.k_size)
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
        self.OUTPUT_NORM_DRAM = self.allocate_tensor_dram(1 * self.vector_length * bpe)
        self.LOGITS_DRAM = self.allocate_tensor_dram(1 * self.EMBEDDING_ELEMENTS * bpe)

        kv_cache_total = self.LAYER_SIZE * nkvh * self.MAX_CONTEXT_SIZE * ahd * bpe
        self.LAYER0_V_DRAM = self.allocate_tensor_dram(kv_cache_total)
        self.LAYER0_K_ROPE_DRAM = self.allocate_tensor_dram(kv_cache_total)
        zero_kv = torch.zeros(kv_cache_total, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, zero_kv)
        self.dma_to_accelerator_memory(self.LAYER0_K_ROPE_DRAM, zero_kv)

        self._preallocate_identity_matrix()

    def run_qwen3(self, prefill_seq: tuple, temperature: float = 0.0) -> None:
        cfg = self._cfg
        paths_cfg = cfg.get("paths", {})
        inst_bin_path  = os.path.join(self.script_dir, paths_cfg.get("instruction_bin",  "qwen3_1.7b_bin/qwen3_instruction.bin"))
        inst_meta_path = os.path.join(self.script_dir, paths_cfg.get("instruction_meta", "qwen3_1.7b_bin/qwen3_instruction.json"))

        _original_print(f"Loading instruction bin ({os.path.getsize(inst_bin_path)} bytes)...")
        self.load_instructions(inst_bin_path)
        with open(inst_meta_path) as f:
            meta = json.load(f)

        prefill_buckets     = meta["prefill_seq_len_buckets"]
        decoder_buckets     = meta["decoder_seq_len_buckets"]
        prefill_start_addrs = [_parse_offset(a) for a in meta["prefill_start_addrs"]]
        prefill_flops_list  = meta["prefill_flops"]
        decoder_start_addrs = [_parse_offset(a) for a in meta["decoder_start_addrs"]]

        if len(prefill_seq) > 1:
            prefill_tokens = prefill_seq[:-1]
        else:
            raise ValueError("Prefill sequence must have at least 2 tokens.")

        prefill_seq_len = len(prefill_tokens)
        max_bucket = max(prefill_buckets)
        if prefill_seq_len > max_bucket:
            raise ValueError(
                f"prefill_seq_len={prefill_seq_len} exceeds max compiled bucket {max_bucket}. "
                "Recompile with python qwen3_1.7b_test.py"
            )

        bucket_idx = next(i for i, b in enumerate(prefill_buckets) if b >= prefill_seq_len)
        bucket_seq_len = prefill_buckets[bucket_idx]
        self.seq_len = prefill_seq_len

        q_seq_len = bucket_seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64

        _original_print(f"\n--- Starting prefill (seq_len={prefill_seq_len}, bucket={bucket_seq_len}) ---")
        timer = time.perf_counter()

        embedding_tensor = self.get_embedding_for_tokens(list(prefill_tokens))
        if bucket_seq_len > prefill_seq_len:
            pad = embedding_tensor[-1:].repeat(bucket_seq_len - prefill_seq_len, 1)
            embedding_tensor = torch.cat([embedding_tensor, pad], dim=0)
        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)

        bias = torch.full((aligned_seq_len, aligned_seq_len), float("-inf"), dtype=torch.bfloat16)
        rows = torch.arange(aligned_seq_len).unsqueeze(1)
        cols = torch.arange(aligned_seq_len).unsqueeze(0)
        bias.masked_fill_(cols <= rows, 0.0)
        bias[:, bucket_seq_len * self.group_size:] = float("-inf")
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias)

        self.start_execute_from_dram(prefill_start_addrs[bucket_idx])
        self.wait_queue(120.0)
        latency_us = self.report_latency_in_us()
        _original_print(f"  Done in {time.perf_counter() - timer:.2f}s "
                        f"({latency_us/1e6:.2f}s hw), "
                        f"{prefill_flops_list[bucket_idx]/latency_us/1e3:.2f} GFLOPS")

        _original_print(f"\n--- Starting decoder ---")
        timer = time.perf_counter()
        token_id = prefill_seq[-1]
        _qwen3_stop_tokens = {151643, 151645, self._end_of_turn_token_id}
        ahd = self.actual_head_dim
        bpe = self.bytes_per_element
        _kv_stride   = ahd * bpe
        _rope_stride = ahd * 2 * bpe

        while self.seq_len < self.MAX_CONTEXT_SIZE:
            _set_silent(True)
            self.seq_len += 1
            prog_idx = next((i for i, b in enumerate(decoder_buckets) if self.seq_len <= b),
                            len(decoder_buckets) - 1)

            self.isa_add_set_core(self.V_CACHE_SIZE_REG, ue_35bit_addr_shifter((self.seq_len - 1) * _kv_stride))
            self.isa_add_set_core(self.ROPE_SIZE_REG,    ue_35bit_addr_shifter((self.seq_len - 1) * _rope_stride))

            emb = self.get_embedding_for_tokens([token_id])
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, emb)

            aligned_seq_len = ((self.seq_len + 63) // 64) * 64
            bias_host = torch.full((1, aligned_seq_len), -1e36, dtype=torch.bfloat16)
            bias_host[0, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_host)

            self.start_execute_from_dram(decoder_start_addrs[prog_idx])
            self.wait_queue(10.0)
            token_id = self.get_arg_max_index()
            _set_silent(False)
            if token_id in _qwen3_stop_tokens:
                _original_print(f"\nStop token {token_id} reached.")
                break
            _original_print(self.tokenizer.decode([token_id]), end="", flush=True)

        _original_print(f"\nDecoder done in {time.perf_counter() - timer:.2f}s, "
                        f"{self.seq_len} total tokens.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3-1.7B inference from pre-compiled bins")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--local-weights", action="store_true")
    parser.add_argument("--dev", type=str, default="xdma0")
    parser.add_argument("--cycle", type=float, default=1/0.17)
    args = parser.parse_args()

    if args.local_weights:
        weights_bin_rel = "qwen3_1.7b_bin/full_model_weights.bin"
    else:
        weights_bin_rel = "qwen3_1.7b_bin/weights_qwen3_1.7b_hf.bin"
    weights_bin_full = os.path.join(SCRIPT_DIR, weights_bin_rel)

    cfg = _load_config(SCRIPT_DIR)
    missing = check_bins_early(weights_bin_full, cfg)
    if missing:
        _original_print("Missing files (run python qwen3_1.7b_test.py first):")
        for f in missing:
            _original_print(f"  {f}")
        _original_print("Run: python qwen3_1.7b_test.py")
        return

    set_dma_device(args.dev)
    # Refresh local bindings shadowed at import time so DMA goes to the right device
    import sys as _sys, user_dma_core as _udc
    _mod = _sys.modules[__name__]
    _mod.DMA_DEVICE_H2C = _udc.DMA_DEVICE_H2C
    _mod.DMA_DEVICE_C2H = _udc.DMA_DEVICE_C2H
    _mod.DMA_DEVICE_USER = _udc.DMA_DEVICE_USER
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle

    _set_silent(True)
    ue = Qwen3_1_7b_UnifiedEngine(script_dir=SCRIPT_DIR, weights_bin=weights_bin_rel)
    _set_silent(False)

    if args.prompt is not None:
        conversation = [{"role": "user", "content": args.prompt}]
        prompt_with_template = ue.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        prefill_seq = tuple(ue.tokenizer.encode(prompt_with_template, add_special_tokens=False))
        _original_print(f"Prompt ({len(prefill_seq)} tokens): {args.prompt!r}")
    else:
        prefill_seq = tuple(cfg["default_prefill_tokens"])
        _original_print(f"Using default prefill ({len(prefill_seq)} tokens): "
                        f"{ue.tokenizer.decode(list(prefill_seq))!r}")

    ue.run_qwen3(prefill_seq=prefill_seq)


if __name__ == "__main__":
    main()
