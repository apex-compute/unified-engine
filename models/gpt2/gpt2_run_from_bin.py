#!/usr/bin/env python3
"""GPT-2 inference from pre-compiled bins — no recompilation at inference time.

Compile first:
    python gpt2_test.py --compile

Then run:
    python gpt2_run_from_bin.py --prompt "The quick brown fox"
"""

import os, sys, time, json
import builtins
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_THIS_DIR)))

from transformers import AutoTokenizer
import user_dma_core
from user_dma_core import (
    set_dma_device, ue_35bit_addr_shifter, DMA_DEVICE_H2C, DRAM_INSTRUCTION_ADDR,
)
from gpt2_test import GPT2_UnifiedEngine, _load_config, _parse_offset

_original_print = builtins.print
_silent = False


def _set_silent(v):
    global _silent
    _silent = v


def _quiet_print(*a, **kw):
    if not _silent:
        _original_print(*a, **kw)


builtins.print = _quiet_print


class GPT2_RunFromBin(GPT2_UnifiedEngine):
    """GPT-2 inference without HF model at runtime — overrides weight_init to load from bin."""

    def weight_init(self) -> None:
        cfg = self._cfg
        emb_cfg = cfg["special"]["embedding"]

        # Token embedding from weights bin
        token_embd_offset = _parse_offset(emb_cfg["token_embd_offset"])
        token_embd_size = _parse_offset(emb_cfg["token_embd_size"])
        vocab_size = emb_cfg["vocab_size"]    # 50304
        embed_dim = emb_cfg["embedding_dim"]  # 768
        raw = self.weight_bin[token_embd_offset:token_embd_offset + token_embd_size]
        self.embedding_weight = (
            torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
            .reshape(vocab_size, embed_dim).clone()
        )

        # Positional embedding from weights bin
        nr = cfg["non_layer_regions"]
        pos_off = _parse_offset(nr["POS_EMBED"]["offset"])
        pos_sz = nr["POS_EMBED"]["size"]
        pos_raw = self.weight_bin[pos_off:pos_off + pos_sz]
        self.pos_embedding_weight = (
            torch.frombuffer(bytearray(pos_raw), dtype=torch.bfloat16)
            .reshape(1024, embed_dim).clone()
        )

        # Tokenizer only — no full model load
        hf_model_dir = os.path.join(self.script_dir, cfg["paths"]["hf_model_dir"])
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, trust_remote_code=True)

        # DMA layer weights to DRAM
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
                raw = self.weight_bin[bin_off:bin_off + sz]
                offset_in_layer = off - base_layer0
                dram_addr = layers_base_dram + layer_idx * LAYER_WEIGHT_SIZE + offset_in_layer
                self.dma_write(DMA_DEVICE_H2C, dram_addr, raw, sz)
            if layer_idx == 0:
                for entry in blk0_structure:
                    off_key = entry["key"]
                    off = self.weight_defs[off_key]
                    offset_in_layer = off - base_layer0
                    setattr(self, entry["attr"], layers_base_dram + offset_in_layer)

        # Non-layer weights (OUTPUT_LN, LM_HEAD, POS_EMBED)
        for entry in cfg["layers"]["non_layer"]:
            off_key = entry["key"]
            off = self.weight_defs[off_key]
            sz = self.weight_defs[f"{off_key}_SIZE"]
            raw = self.weight_bin[off:off + sz]
            addr = self.allocate_params_dram(sz)
            self.dma_write(DMA_DEVICE_H2C, addr, raw, sz)
            setattr(self, entry["attr"], addr)

    def run_gpt2(self, prefill_seq: tuple,
                 temperature: float = 0.8, top_k: int = 40,
                 top_p: float = 0.95, repetition_penalty: float = 1.2) -> None:
        cfg = self._cfg
        paths_cfg = cfg.get("paths", {})
        inst_bin_path = os.path.join(self.script_dir, paths_cfg.get("instruction_bin", "gpt2_bin/gpt2_instruction.bin"))
        meta_path = os.path.join(self.script_dir, paths_cfg.get("instruction_meta", "gpt2_bin/gpt2_instruction.json"))
        _original_print(f"Loading instruction bin ({os.path.getsize(inst_bin_path)} bytes)...")
        self.load_instructions(inst_bin_path)
        with open(meta_path) as f:
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
                "Recompile with python gpt2_test.py --compile"
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
        _kv_stride = self.actual_head_dim * self.bytes_per_element  # 128 bytes/position

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

            _set_silent(False)
            if token_id == self._end_of_turn_token_id:
                _original_print("\nStop token reached.")
                break
            _original_print(self.tokenizer.decode([token_id]), end="", flush=True)

        _original_print(f"\nDecoder done in {time.perf_counter() - timer:.2f}s, "
                        f"{self.seq_len} total tokens.")


def main():
    import argparse
    script_dir = _THIS_DIR
    cfg_raw = _load_config(script_dir)
    paths = cfg_raw["paths"]

    inst_bin = os.path.join(script_dir, paths.get("instruction_bin", "gpt2_bin/gpt2_instruction.bin"))
    inst_meta = os.path.join(script_dir, paths.get("instruction_meta", "gpt2_bin/gpt2_instruction.json"))
    weights_bin = os.path.join(script_dir, paths["weights_bin"])
    missing = [(p, n) for p, n in [(inst_bin, "instruction bin"),
                                    (inst_meta, "instruction meta"),
                                    (weights_bin, "weights bin")] if not os.path.exists(p)]
    if missing:
        for p, n in missing:
            _original_print(f"Missing {n}: {p}")
        _original_print("Run: python gpt2_test.py --compile")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="GPT-2 inference from pre-compiled bins")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt (raw, no chat template — GPT-2 is a base model)")
    parser.add_argument("--dev", type=str, default="xdma0")
    parser.add_argument("--cycle", type=float, default=1 / 0.17)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    args = parser.parse_args()

    set_dma_device(args.dev)
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle

    _set_silent(True)
    ue = GPT2_RunFromBin(script_dir=script_dir)
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
