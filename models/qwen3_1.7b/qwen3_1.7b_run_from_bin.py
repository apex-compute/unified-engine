#!/usr/bin/env python3
"""Qwen3-1.7B inference from pre-compiled bin files.

Load-and-execute path — no HF download. The cached single bin
(``qwen3_1.7b_instruction.bin`` + matching ``.json`` meta) plus the
weights bin and tokenizer files must already exist on disk. Run
``qwen3_1.7b_test.py`` once on a build machine with HF access to
produce them; this script then runs entirely offline.

Why this script still calls ``compile_instructions()`` even though the
bin is on disk: ``flash_attention_core_pbi`` and
``bf16_transpose_core_pbi`` issue **host-side identity-matrix DMA
writes** during compile, and the captured bin references those exact
DRAM addresses. If any other script (e.g. a different model) ran on
this FPGA between bin-write time and now, those identity addresses
hold whatever that other script wrote there → garbage identity →
NaN attention → all-`!` decode. Re-running compile re-issues the
identity DMAs (the captured instructions are discarded; the on-disk
bin is reused, not re-written).
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

def _set_silent(val):
    global _SILENT_MODE
    _SILENT_MODE = val

import user_dma_core
from user_dma_core import set_dma_device

# Import the engine class from the test script — it has the full
# compile_instructions implementation we need to re-DMA identities.
# weight_init was refactored to read the embedding straight from the
# weight bin and load the tokenizer with ``local_files_only=True``, so
# importing + using it in this offline script is safe.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_qwen3_1_7b_test_for_runbin",
    os.path.join(SCRIPT_DIR, "qwen3_1.7b_test.py"),
)

_test_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_test_mod)
Qwen3_1_7b_UnifiedEngine = _test_mod.Qwen3_1_7b_UnifiedEngine


def _parse_offset(val):
    return int(val, 0) if isinstance(val, str) else int(val)


BIN_DIR = os.path.join(SCRIPT_DIR, "qwen3_1.7b_bin")


def check_bins_early(weights_bin_full):
    """Hard-fail BEFORE any FPGA / HF touch if a required local file is missing.

    The customer ships these files alongside this script, produced once by
    running ``qwen3_1.7b_test.py`` on a build machine with HF access.
    """
    missing = []
    if not os.path.exists(weights_bin_full):
        missing.append(os.path.relpath(weights_bin_full, SCRIPT_DIR))
    for name in ("qwen3_1.7b_instruction.bin", "qwen3_1.7b_instruction.json"):
        if not os.path.exists(os.path.join(BIN_DIR, name)):
            missing.append(name)
    tokenizer_dir = os.path.join(BIN_DIR, "Qwen3-1.7B")
    if not (os.path.exists(os.path.join(tokenizer_dir, "tokenizer.json")) or
            os.path.exists(os.path.join(tokenizer_dir, "tokenizer_config.json"))):
        missing.append("Qwen3-1.7B/{tokenizer.json,tokenizer_config.json}")
    return missing


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3-1.7B inference from pre-compiled bins (offline)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt (default: from qwen3_1.7b_config.json default_prompt)")
    parser.add_argument("--local-weights", action="store_true",
                        help="Use qwen3_1.7b_bin/full_model_weights.bin instead of weights_qwen3_1.7b_hf.bin")
    parser.add_argument("--dev", type=str, default="xdma0", help="DMA device name (default: xdma0)")
    parser.add_argument("--cycle", type=float, default=5.62,
                        help="Clock cycle time in ns (default: 5.62ns ≈ peak 22.8 GFLOPS)")
    args = parser.parse_args()

    weights_bin_rel = ("qwen3_1.7b_bin/full_model_weights.bin" if args.local_weights
                       else "qwen3_1.7b_bin/weights_qwen3_1.7b_hf.bin")
    weights_bin_full = os.path.join(SCRIPT_DIR, weights_bin_rel)

    missing = check_bins_early(weights_bin_full)
    if missing:
        _original_print("Missing local files (run qwen3_1.7b_test.py first on a build machine with HF access):")
        for f in missing:
            _original_print(f"  {f}")
        sys.exit(1)

    set_dma_device(args.dev)
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle
    user_dma_core.UE_PEAK_GFLOPS = 0.128 / args.cycle
    _original_print(f"Setting CLOCK_CYCLE_TIME_NS = {user_dma_core.CLOCK_CYCLE_TIME_NS}, "
                    f"UE_PEAK_GFLOPS = {user_dma_core.UE_PEAK_GFLOPS:.4f}")

    _set_silent(True)
    ue = Qwen3_1_7b_UnifiedEngine(script_dir=SCRIPT_DIR, weights_bin=weights_bin_rel)
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

    # Re-run compile (re-issues host-side identity DMAs; bin on disk is reused, not re-written).
    _original_print(f"\n--- Compile pass (replays identity-matrix DMAs; bin on disk is reused) ---")
    timer = time.perf_counter()
    inst_meta = ue.compile_instructions()
    _original_print(f"  compile_instructions done in {time.perf_counter() - timer:.2f}s")

    paths_cfg = cfg.get("paths", {})
    inst_bin_path = os.path.join(SCRIPT_DIR, paths_cfg.get("instruction_bin",
                                  "qwen3_1.7b_bin/qwen3_1.7b_instruction.bin"))

    _original_print(f"\n--- Loading unified instruction bin ---")
    timer = time.perf_counter()
    base_addr, total_size = ue.load_instructions(inst_bin_path)
    # Reserve a preamble slot after the loaded bin. Prefill preamble = 4 instructions (128 B);
    # decode preamble = 2 instructions and overwrites the same slot.
    preamble_addr = ue.get_program_dram_addr()
    ue.allocate_program_dram(128)
    _original_print(f"  Loaded {total_size} B at 0x{base_addr:X}; preamble slot at 0x{preamble_addr:X} "
                    f"({time.perf_counter() - timer:.3f}s)")

    prefill_program_addr = _parse_offset(inst_meta["prefill_program_start_addr"])
    decoder_program_addr = _parse_offset(inst_meta["decoder_program_start_addr"])
    decoder_total_flops  = inst_meta["decoder_total_flops"]

    actual_seq_len = len(prefill_seq) - 1
    # Rescale prefill FLOPs from compile-time template to actual seq_len so the
    # GFLOPS report reflects the real work this prompt did.
    template_seq_len = int(inst_meta["prefill_template_seq_len"])
    gflops_prefill = inst_meta["prefill_template_flops"] * actual_seq_len // max(template_seq_len, 1)

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
