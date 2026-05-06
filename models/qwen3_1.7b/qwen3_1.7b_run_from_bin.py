#!/usr/bin/env python3
"""Qwen3-1.7B inference from pre-compiled bin files.
Compile first (auto-skipped if already compiled):
    python qwen3_1.7b_test.py

Then run:
    python qwen3_1.7b_run_from_bin.py --prompt "your prompt"
"""
import builtins
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

# Load qwen3_1.7b_test (filename has dot, so use importlib).
# The test module installs its own builtins.print suppression; we capture
# _original_print before that happens so main() can use it for user-visible output.
_original_print = builtins.print

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("qwen3_1_7b_test", os.path.join(SCRIPT_DIR, "qwen3_1.7b_test.py"))
_mod  = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Qwen3_1_7b_UnifiedEngine = _mod.Qwen3_1_7b_UnifiedEngine
_load_config = _mod._load_config
_parse_offset = _mod._parse_offset
# After exec, builtins.print is the test module's quiet_print.
# run_qwen3() sets the test module's _SILENT_MODE to suppress it.

import user_dma_core
from user_dma_core import set_dma_device

BIN_DIR = os.path.join(SCRIPT_DIR, "qwen3_1.7b_bin")


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


class Qwen3_1_7b_RunFromBin(Qwen3_1_7b_UnifiedEngine):
    """Qwen3-1.7B inference without HF model at runtime — weight_init loads from bin."""

    def weight_init(self) -> None:
        from transformers import AutoTokenizer
        emb_cfg = self._cfg["special"]["embedding"]
        token_embd_offset = int(emb_cfg["token_embd_offset"], 16)
        vocab_size = emb_cfg["vocab_size"]
        emb_dim    = emb_cfg["embedding_dim"]
        emb_bytes  = vocab_size * emb_dim * self.bytes_per_element
        raw_emb = bytearray(self.weight_bin[token_embd_offset : token_embd_offset + emb_bytes])
        import torch
        self.embedding_weight = torch.frombuffer(raw_emb, dtype=torch.bfloat16).reshape(vocab_size, emb_dim).clone()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_dir, trust_remote_code=True)

        LAYER_WEIGHT_SIZE = self.weight_defs["LAYER_WEIGHT_SIZE"]
        base_layer0 = self.weight_defs["BLK0_ATTN_NORM_WEIGHT"]
        blk0_regions = [(s["key"], f"{s['key']}_SIZE", s["attr"]) for s in self._cfg["layers"]["structure"]]
        non_layer = [(s["key"], f"{s['key']}_SIZE", s["attr"]) for s in self._cfg["layers"]["non_layer"]
                     if s["key"] not in ("ROPE_LOCAL", "ROPE_GLOBAL")]

        layers_total = self.LAYER_SIZE * LAYER_WEIGHT_SIZE
        layers_base_dram = self.allocate_params_dram(layers_total)
        from user_dma_core import DMA_DEVICE_H2C
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

    from model_lib_core import load_config_with_weight_defs
    cfg = load_config_with_weight_defs(os.path.join(SCRIPT_DIR, "qwen3_1.7b_config.json"))
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

    _mod._SILENT_MODE = True
    ue = Qwen3_1_7b_RunFromBin(script_dir=SCRIPT_DIR, weights_bin=weights_bin_rel)
    _mod._SILENT_MODE = False

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
