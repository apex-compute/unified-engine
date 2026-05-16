#!/usr/bin/env python3
"""GPT-2 inference from pre-compiled bin files.

Compile first:
    python gpt2_test.py --compile

Then run:
    python gpt2_run_from_bin.py --prompt "The quick brown fox"
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
    DMA_DEVICE_H2C, DRAM_INSTRUCTION_ADDR, UE_FMAX_CONTEXT_SIZE, UE_VECTOR_SIZE,
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
    config_path = os.path.join(script_dir, "gpt2_config.json")
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


def check_bins_early(script_dir, cfg):
    paths = cfg["paths"]
    weights_bin = os.path.join(script_dir, paths["weights_bin"])
    inst_bin = os.path.join(script_dir, paths.get("instruction_bin", "gpt2_bin/gpt2_instruction.bin"))
    inst_meta = os.path.join(script_dir, paths.get("instruction_meta", "gpt2_bin/gpt2_instruction.json"))
    missing = []
    for p in [weights_bin, inst_bin, inst_meta]:
        if not os.path.exists(p):
            missing.append(os.path.relpath(p, script_dir))
    return missing


class GPT2_UnifiedEngine(UnifiedEngine):
    """GPT-2 Base (124M) inference from pre-compiled bins — no HF model at runtime."""

    def __init__(self, script_dir=None, weights_bin=None):
        super().__init__()
        self.script_dir = script_dir or SCRIPT_DIR
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
        self.actual_head_dim = fi["actual_head_dim"]
        self.num_kv_heads = fi["num_kv_heads"]
        self.MAX_CONTEXT_SIZE = model["max_context_size"]
        self.LAYER_SIZE = fi["num_layers"]
        self.EMBEDDING_ELEMENTS = fi["embedding_vocab"]
        self._isa_reg_counter = 1
        fixed = self._cfg.get("fixed_isa_regs", {})
        self.V_CACHE_SIZE_REG = fixed["V_CACHE_SIZE_REG"]
        self.TMP_REG = fixed["TMP_REG"]
        self.causal_mask_upper = False
        self._end_of_turn_token_id = model["end_of_turn_token_id"]

        bin_path = weights_bin or paths["weights_bin"]
        full_path = os.path.join(self.script_dir, bin_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Weights bin not found: {full_path}")
        with open(full_path, "rb") as f:
            self.weight_bin = f.read()
        self.weight_init()
        self.tensor_init()

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

    def get_embedding_for_tokens(self, token_ids, start_pos: int = 0) -> torch.Tensor:
        tid_t = torch.tensor(token_ids, dtype=torch.long)
        tok_emb = torch.zeros(len(token_ids), self.vector_length, dtype=torch.bfloat16)
        valid = tid_t < self.embedding_weight.shape[0]
        tok_emb[valid] = self.embedding_weight[tid_t[valid]]
        pos_ids = torch.arange(start_pos, start_pos + len(token_ids), dtype=torch.long)
        pos_ids = pos_ids.clamp(max=self.pos_embedding_weight.shape[0] - 1)
        pos_emb = self.pos_embedding_weight[pos_ids]
        return (tok_emb + pos_emb).to(torch.bfloat16)

    def weight_init(self) -> None:
        cfg = self._cfg
        emb_cfg = cfg["special"]["embedding"]

        token_embd_offset = _parse_offset(emb_cfg["token_embd_offset"])
        token_embd_size = _parse_offset(emb_cfg["token_embd_size"])
        vocab_size = emb_cfg["vocab_size"]
        embed_dim = emb_cfg["embedding_dim"]
        raw = self.weight_bin[token_embd_offset : token_embd_offset + token_embd_size]
        self.embedding_weight = (
            torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
            .reshape(vocab_size, embed_dim).clone()
        )

        nr = cfg["non_layer_regions"]
        pos_off = _parse_offset(nr["POS_EMBED"]["offset"])
        pos_sz = nr["POS_EMBED"]["size"]
        pos_raw = self.weight_bin[pos_off : pos_off + pos_sz]
        self.pos_embedding_weight = (
            torch.frombuffer(bytearray(pos_raw), dtype=torch.bfloat16)
            .reshape(1024, embed_dim).clone()
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_dir, trust_remote_code=True)

        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        blk0_structure = cfg["layers"]["structure"]
        base_layer0 = self.weight_defs[blk0_structure[0]["key"]]

        layers_total = self.LAYER_SIZE * LAYER_WEIGHT_SIZE
        layers_base_dram = self.allocate_params_dram(layers_total)
        for layer_idx in range(self.LAYER_SIZE):
            for entry in blk0_structure:
                off_key = entry["key"]
                off = self.weight_defs[off_key]
                sz = self.weight_defs[f"{off_key}_SIZE"]
                bin_off = off + layer_idx * LAYER_WEIGHT_SIZE
                raw = self.weight_bin[bin_off : bin_off + sz]
                offset_in_layer = off - base_layer0
                dram_addr = layers_base_dram + layer_idx * LAYER_WEIGHT_SIZE + offset_in_layer
                self.dma_write(DMA_DEVICE_H2C, dram_addr, raw, sz)
            if layer_idx == 0:
                for entry in blk0_structure:
                    off_key = entry["key"]
                    off = self.weight_defs[off_key]
                    offset_in_layer = off - base_layer0
                    setattr(self, entry["attr"], layers_base_dram + offset_in_layer)

        for entry in cfg["layers"]["non_layer"]:
            off_key = entry["key"]
            off = self.weight_defs[off_key]
            sz = self.weight_defs[f"{off_key}_SIZE"]
            raw = self.weight_bin[off : off + sz]
            addr = self.allocate_params_dram(sz)
            self.dma_write(DMA_DEVICE_H2C, addr, raw, sz)
            setattr(self, entry["attr"], addr)

    def tensor_init(self) -> None:
        seq_len = self.MAX_CONTEXT_SIZE
        q_seq_len = seq_len * self.group_size
        aligned_seq_len = ((q_seq_len + 63) // 64) * 64

        self.LAYER0_V_DRAM = self.allocate_tensor_dram(self.LAYER_SIZE * self.MAX_CONTEXT_SIZE * self.k_size)
        self.LAYER0_K_DRAM_CACHE = self.allocate_tensor_dram(self.LAYER_SIZE * self.MAX_CONTEXT_SIZE * self.k_size)
        zero_pad = torch.zeros(self.LAYER_SIZE * self.MAX_CONTEXT_SIZE * self.k_size, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_V_DRAM, zero_pad)
        self.dma_to_accelerator_memory(self.LAYER0_K_DRAM_CACHE, zero_pad)

        zero_add = torch.zeros(seq_len * self.head_dim * self.bytes_per_element, dtype=torch.bfloat16)
        self.ZERO_DRAM_ADDR = self.allocate_tensor_dram(seq_len * self.head_dim * self.bytes_per_element)
        self.dma_to_accelerator_memory(self.ZERO_DRAM_ADDR, zero_add)
        self.IDENTITY_DRAM_ADDR = self.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * self.bytes_per_element)
        self.dma_to_accelerator_memory(self.IDENTITY_DRAM_ADDR, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))

        self.LAYER0_FLASH_Q_DRAM = self.allocate_tensor_dram(aligned_seq_len * self.actual_head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_K_DRAM = self.allocate_tensor_dram(aligned_seq_len * self.actual_head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_V_DRAM = self.allocate_tensor_dram(aligned_seq_len * self.actual_head_dim * self.bytes_per_element)
        zero_flash = torch.zeros(aligned_seq_len * self.actual_head_dim * self.bytes_per_element, dtype=torch.bfloat16)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_Q_DRAM, zero_flash)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_K_DRAM, zero_flash)
        self.dma_to_accelerator_memory(self.LAYER0_FLASH_V_DRAM, zero_flash)

        self.LAYER0_INPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_Q_DRAM = self.allocate_tensor_dram(seq_len * self.q_size)
        self.LAYER0_K_DRAM = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_V_PROJ_DRAM = self.allocate_tensor_dram(seq_len * self.k_size)
        self.LAYER0_FLASH_OUT_HEAD_DRAM = self.allocate_tensor_dram(
            aligned_seq_len * self.actual_head_dim * self.bytes_per_element)
        self.LAYER0_FLASH_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.head_dim * self.group_size * self.bytes_per_element)
        self.LAYER0_FLASH_SCRATCH_DRAM = self.allocate_tensor_dram(max(self.actual_head_dim, UE_FMAX_CONTEXT_SIZE) * aligned_seq_len * 2 + self.actual_head_dim * aligned_seq_len * 2)
        self.LAYER0_FLASH_BIAS_DRAM = self.allocate_tensor_dram(aligned_seq_len * aligned_seq_len * self.bytes_per_element)
        self.LAYER0_ATTN_PROJ_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_POST_ATTN_RESIDUAL_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_PRE_MLP_NORM_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_MLP_FC_DRAM = self.allocate_tensor_dram(seq_len * self.mlp_elements * 2)
        self.LAYER0_MLP_PROJ_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.LAYER0_OUTPUT_DRAM = self.allocate_tensor_dram(seq_len * self.vector_length * 2)
        self.OUTPUT_NORM_DRAM = self.allocate_tensor_dram(1 * self.vector_length * self.bytes_per_element)
        self.LOGITS_DRAM = self.allocate_tensor_dram(1 * self.EMBEDDING_ELEMENTS * self.bytes_per_element)

    def program_execute(self, program_start_addr, timeout=10.0, gflops=None):
        self.start_execute_from_dram(program_start_addr)
        if timeout != 0:
            self.wait_queue(timeout)
            latency = self.report_latency_in_us()
            if gflops is not None:
                self.report_flop_rate_gflops(gflops)
            return latency
        return 0

    @staticmethod
    def _sample_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0,
                      top_p: float = 1.0, generated_ids=None, repetition_penalty: float = 1.0) -> int:
        if temperature == 0:
            return logits.argmax().item()
        logits = logits.float()
        if repetition_penalty != 1.0 and generated_ids:
            seen = torch.tensor(list(set(generated_ids)), dtype=torch.long)
            seen = seen[seen < logits.size(0)]
            if seen.numel() > 0:
                orig = logits[seen]
                logits[seen] = torch.where(orig > 0, orig / repetition_penalty, orig * repetition_penalty)
        logits = logits / temperature
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            threshold = logits.topk(top_k).values[-1]
            logits[logits < threshold] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_indices = probs.sort(descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)
            mask = cumsum - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum()
            idx = torch.multinomial(sorted_probs, num_samples=1).item()
            return sorted_indices[idx].item()
        return torch.multinomial(probs, num_samples=1).item()

    def run_gpt2(self, prefill_seq: tuple,
                 temperature: float = 0.8, top_k: int = 40,
                 top_p: float = 0.95, repetition_penalty: float = 1.2) -> None:
        cfg = self._cfg
        paths_cfg = cfg.get("paths", {})
        inst_bin_path = os.path.join(self.script_dir, paths_cfg.get("instruction_bin", "gpt2_bin/gpt2_instruction.bin"))
        inst_meta_path = os.path.join(self.script_dir, paths_cfg.get("instruction_meta", "gpt2_bin/gpt2_instruction.json"))
        _original_print(f"Loading instruction bin ({os.path.getsize(inst_bin_path)} bytes)...")
        self.load_instructions(inst_bin_path)
        with open(inst_meta_path) as f:
            meta = json.load(f)

        prefill_buckets = meta["prefill_seq_len_buckets"]
        decoder_buckets = meta["decoder_seq_len_buckets"]
        prefill_start_addrs = [_parse_offset(a) for a in meta["prefill_start_addrs"]]
        prefill_flops_list = meta["prefill_flops"]
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
                "Delete gpt2_bin/gpt2_instruction.bin to recompile."
            )

        bucket_idx = next(i for i, b in enumerate(prefill_buckets) if b >= prefill_seq_len)
        bucket_seq_len = prefill_buckets[bucket_idx]
        self.seq_len = prefill_seq_len
        aligned_seq_len = ((bucket_seq_len + 63) // 64) * 64

        _original_print(f"\n--- Starting prefill (seq_len={prefill_seq_len}, bucket={bucket_seq_len}) ---")
        timer = time.perf_counter()

        embedding_tensor = self.get_embedding_for_tokens(list(prefill_tokens), start_pos=0)
        if bucket_seq_len > prefill_seq_len:
            pad = embedding_tensor[-1:].repeat(bucket_seq_len - prefill_seq_len, 1)
            embedding_tensor = torch.cat([embedding_tensor, pad], dim=0)
        self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, embedding_tensor)

        bias = torch.full((aligned_seq_len, aligned_seq_len), float("-inf"), dtype=torch.bfloat16)
        rows = torch.arange(aligned_seq_len).unsqueeze(1)
        cols = torch.arange(aligned_seq_len).unsqueeze(0)
        bias.masked_fill_(cols <= rows, 0.0)
        bias[:, bucket_seq_len:] = float("-inf")
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
        generated_ids = []
        _kv_stride = self.actual_head_dim * self.bytes_per_element

        while self.seq_len < self.MAX_CONTEXT_SIZE:
            _set_silent(True)
            self.seq_len += 1

            prog_idx = 0
            for bi, b in enumerate(decoder_buckets):
                if self.seq_len <= b:
                    prog_idx = bi
                    break
            else:
                prog_idx = len(decoder_buckets) - 1

            self.isa_add_set_core(self.V_CACHE_SIZE_REG,
                                  ue_35bit_addr_shifter((self.seq_len - 1) * _kv_stride))

            emb = self.get_embedding_for_tokens([token_id], start_pos=self.seq_len - 1)
            self.dma_to_accelerator_memory(self.LAYER0_INPUT_DRAM, emb)

            bucket_dec_len = decoder_buckets[prog_idx]
            bias_host = torch.full((1, bucket_dec_len), -1e36, dtype=torch.bfloat16)
            bias_host[0, :self.seq_len] = 0.0
            self.dma_to_accelerator_memory(self.LAYER0_FLASH_BIAS_DRAM, bias_host)

            self.start_execute_from_dram(decoder_start_addrs[prog_idx])
            self.wait_queue(10.0)
            _set_silent(False)

            if temperature > 0:
                logits = self.dma_from_accelerator_memory(self.LOGITS_DRAM, 50257)
                token_id = GPT2_UnifiedEngine._sample_token(
                    logits, temperature=temperature, top_k=top_k, top_p=top_p,
                    generated_ids=generated_ids, repetition_penalty=repetition_penalty)
                generated_ids.append(token_id)
            else:
                token_id = self.get_arg_max_index()
                if token_id >= 50257:
                    logits = self.dma_from_accelerator_memory(self.LOGITS_DRAM, 50257)
                    token_id = logits.argmax().item()

            if token_id == self._end_of_turn_token_id:
                _original_print("\nStop token reached.")
                break
            _original_print(self.tokenizer.decode([token_id]), end="", flush=True)

        _original_print(f"\nDecoder done in {time.perf_counter() - timer:.2f}s, "
                        f"{self.seq_len} total tokens.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPT-2 inference from pre-compiled bins")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt (raw, no chat template — GPT-2 is a base model)")
    parser.add_argument("--local-weights", action="store_true")
    parser.add_argument("--dev", type=str, default="xdma0")
    parser.add_argument("--cycle", type=float, default=1 / 0.17)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    args = parser.parse_args()

    cfg_raw = _load_config(SCRIPT_DIR)

    if args.local_weights:
        weights_bin_rel = "gpt2_bin/full_model_weights.bin"
        cfg_raw["paths"]["weights_bin"] = weights_bin_rel

    missing = check_bins_early(SCRIPT_DIR, cfg_raw)
    if missing:
        _original_print("Missing bin files (run python gpt2_test.py --compile first):")
        for f in missing:
            _original_print(f"  {f}")
        return

    set_dma_device(args.dev)
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle

    _set_silent(True)
    ue = GPT2_UnifiedEngine(script_dir=SCRIPT_DIR,
                            weights_bin=cfg_raw["paths"]["weights_bin"])
    ue.software_reset()
    _set_silent(False)

    if args.prompt is not None:
        prefill_seq = tuple(ue.tokenizer.encode(args.prompt))
        _original_print(f"Prompt ({len(prefill_seq)} tokens): {args.prompt!r}")
    else:
        prefill_seq = tuple(cfg_raw["default_prefill_tokens"])
        _original_print(f"Using default prefill ({len(prefill_seq)} tokens): "
                        f"{ue.tokenizer.decode(list(prefill_seq))!r}")

    ue.run_gpt2(
        prefill_seq=prefill_seq,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )


if __name__ == "__main__":
    main()
