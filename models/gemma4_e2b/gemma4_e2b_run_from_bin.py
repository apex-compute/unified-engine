#!/usr/bin/env python3
"""Gemma4 E2B inference from the pre-compiled unified instruction bin.

Execute-only. No compilation; no ISA emission; no on-disk cache files
beyond what `gemma4_e2b_test.py` produces. Run `gemma4_e2b_test.py` first
(see README — recommended VLM-first run) to generate the bin, then use
this script for fast subsequent inference.

Usage (from the repo root):

    # Text-only LM
    python models/gemma4_e2b/gemma4_e2b_run_from_bin.py --prompt "Hello"

    # VLM (image + text); requires the unified bin to already contain vision
    python models/gemma4_e2b/gemma4_e2b_run_from_bin.py --vision-enable
    python models/gemma4_e2b/gemma4_e2b_run_from_bin.py --image my.jpg --prompt "Describe."

Compared to `gemma4_e2b_test.py`, this script:
  - Refuses to run if `gemma4_instruction.bin` is missing (no compile path).
  - Refuses VLM mode if the bin lacks the `vision_*` sections.
  - Refuses audio mode (audio is not yet folded into the unified bin).
  - Reuses the `Gemma4_UnifiedEngine` class from the test script for all
    setup (weight upload, tensor init, vision_tensor_init, etc.) — so the
    DRAM layout matches exactly what the bin was compiled against.
"""
import argparse
import builtins
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add the templates root (two levels up) so `user_dma_core` is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))
# Add the model dir itself so `audio_primitives` (used by Gemma4_UnifiedEngine
# at import) is importable.
sys.path.insert(0, SCRIPT_DIR)

import user_dma_core
from user_dma_core import set_dma_device

# Reuse the engine + module constants. Importing gemma4_e2b_test runs its
# import-time side effects (sys.path additions, audio_primitives import, etc.)
# but does not execute main() unless invoked directly.
import gemma4_e2b_test as _ge
from gemma4_e2b_test import (
    Gemma4_UnifiedEngine,
    INSTRUCTION_BIN_COMPILE_VERSION,
    VISION_FIXED_NUM_PATCHES,
    SCRIPT_DIR as _TEST_SCRIPT_DIR,
)

_TEST_SAMPLES = os.path.join(_TEST_SCRIPT_DIR, "..", "..", "test_samples")
DEFAULT_IMAGE = os.path.join(_TEST_SAMPLES, "yosemite.jpg")
DEFAULT_AUDIO = os.path.join(_TEST_SAMPLES, "apex.wav")
BIN_DIR       = os.path.join(SCRIPT_DIR, "gemma4_e2b_bin")
INSTR_BIN     = os.path.join(BIN_DIR, "gemma4_instruction.bin")
INSTR_META    = os.path.join(BIN_DIR, "gemma4_instruction.json")


def _block_compile_paths(ue: Gemma4_UnifiedEngine) -> None:
    """Replace every compile / bin-mutating method on the engine with a
    loud error. If we somehow reach one of them in execute-only mode it
    means the manifest didn't actually have what we asserted, so failing
    fast is better than silently rebuilding."""
    def _refuse(name):
        def _stub(*a, **kw):
            raise RuntimeError(
                f"gemma4_e2b_run_from_bin: {name}() must not run in execute-only mode. "
                f"The unified bin is supposed to already contain the program(s) "
                f"this code path wants to compile. Run gemma4_e2b_test.py first.")
        return _stub
    for _m in ("compile_instruction_bin",
               "extend_instruction_bin_with_vision",
               "compile_prefill",
               "compile_decoder",
               "compile_vision_encoder_bin"):
        if hasattr(ue, _m):
            setattr(ue, _m, _refuse(_m))


def _check_bin_and_manifest() -> dict:
    """Verify the unified bin + manifest exist and the compile version
    matches. Return the manifest dict (with addrs unresolved).
    """
    if not os.path.exists(INSTR_BIN) or not os.path.exists(INSTR_META):
        print("ERROR: Unified instruction bin not found.")
        print(f"  Missing: {INSTR_BIN}")
        print(f"           {INSTR_META}")
        print("")
        print("Run gemma4_e2b_test.py first to generate it. Recommended:")
        print(f"  python {os.path.relpath(os.path.join(SCRIPT_DIR, 'gemma4_e2b_test.py'))} --vision-enable")
        print("(VLM-first run produces the complete bin; subsequent runs in any "
              "mode load it without recompiling — see README.)")
        sys.exit(1)
    with open(INSTR_META, "r") as f:
        manifest = json.load(f)
    if manifest.get("compile_version") != INSTRUCTION_BIN_COMPILE_VERSION:
        print(f"ERROR: instruction bin compile_version mismatch")
        print(f"  Bin on disk:    {manifest.get('compile_version')!r}")
        print(f"  Code expects:   {INSTRUCTION_BIN_COMPILE_VERSION!r}")
        print(f"Delete {INSTR_BIN} and rerun gemma4_e2b_test.py to rebuild.")
        sys.exit(1)
    return manifest


def _check_mode_requirements(manifest: dict, vision_on: bool, audio_on: bool) -> None:
    """Fail fast (before any FPGA work) if the unified bin lacks the
    section the requested mode needs."""
    if vision_on and not manifest.get("contains_vision"):
        print("ERROR: VLM mode requested but the unified bin does not contain")
        print("       the vision encoder section.")
        print(f"  contains_vision={manifest.get('contains_vision')}")
        print("Run gemma4_e2b_test.py --vision-enable first to extend the bin.")
        sys.exit(1)
    if vision_on and "vision_rope_cos_offset" not in manifest:
        print("ERROR: VLM mode requested but the unified bin lacks rope pad sections.")
        print("Run gemma4_e2b_test.py --vision-enable to rebuild with rope pads.")
        sys.exit(1)
    if audio_on:
        print("ERROR: audio mode is not yet supported by run_from_bin.")
        print("       (audio_program_cache_*.bin lives outside the unified bin)")
        print("Use gemma4_e2b_test.py --audio-enable instead.")
        sys.exit(1)


def _fmt_prompt_tokens(seq, min_run: int = 4) -> str:
    """Collapse runs of repeated tokens (e.g. image padding) into '<id>*N'."""
    parts = []
    i = 0
    while i < len(seq):
        j = i
        while j < len(seq) and seq[j] == seq[i]:
            j += 1
        run = j - i
        if run >= min_run:
            parts.append(f"{seq[i]}*{run}")
        else:
            parts.extend(str(t) for t in seq[i:j])
        i = j
    return ", ".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Gemma4 E2B inference from the pre-compiled unified instruction bin.")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt text (LM or VLM). If omitted in LM mode, "
                             "uses the default test prompt from gemma4_e2b_config.json.")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to image file for VLM. Implies --vision-enable.")
    parser.add_argument("--vision-enable", action="store_true",
                        help=f"Enable VLM mode using the shipped image "
                             f"({os.path.relpath(DEFAULT_IMAGE)}). "
                             "Ignored if --image is also given.")
    parser.add_argument("--audio", type=str, default=None,
                        help="Audio file path — currently NOT supported by run_from_bin.")
    parser.add_argument("--audio-enable", action="store_true",
                        help="Audio mode — currently NOT supported by run_from_bin.")
    parser.add_argument("--local-weights", action="store_true",
                        help="Use gemma4_e2b_bin/full_model_weights.bin instead of weights_gemma4_e2b_hf.bin")
    parser.add_argument("--dev", type=str, default="xdma0",
                        help="DMA device name (default: xdma0)")
    parser.add_argument("--cycle", type=float, default=5.62,
                        help="Clock cycle time in ns (default: 5.62)")
    args = parser.parse_args()

    set_dma_device(args.dev)
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle
    print(f"Using DMA device: {args.dev}")
    print(f"Setting CLOCK_CYCLE_TIME_NS = {args.cycle}")

    # 1. Up-front: verify the unified bin + manifest, fail fast on missing.
    manifest = _check_bin_and_manifest()
    vision_on = bool(args.image) or args.vision_enable
    audio_on  = bool(args.audio) or args.audio_enable
    if vision_on and audio_on:
        print("ERROR: --vision-enable / --image and --audio-enable / --audio are mutually exclusive.")
        sys.exit(2)
    _check_mode_requirements(manifest, vision_on, audio_on)

    image_path = args.image or (DEFAULT_IMAGE if args.vision_enable else None)
    if vision_on and image_path and not os.path.exists(image_path):
        print(f"ERROR: image file not found: {image_path}")
        sys.exit(1)

    # 2. Build engine. weight_init + tensor_init mirror exactly what
    # gemma4_e2b_test.py did at compile time — required so DRAM addresses
    # (KV cache, scratch buffers, IDENTITY, etc.) match the baked
    # absolute addrs in the manifest. We block all compile-mutating
    # methods on the engine right after construction so any accidental
    # call surfaces as a loud RuntimeError instead of a silent rebuild.
    ue = Gemma4_UnifiedEngine(local_weights=args.local_weights)
    _block_compile_paths(ue)

    # 3. Set up the prefill sequence. VLM path runs the vision encoder
    # (using the manifest's vision_program_start_addr from the unified
    # bin) and merges image features into the LM prompt. LM-only path
    # just tokenizes.
    if vision_on:
        print(f"[Mode] VLM — image: {image_path}")
        ue.set_prefill_seq_vlm(image_path, prompt=args.prompt)
    elif args.prompt:
        print(f"[Mode] LM — prompt: {args.prompt!r}")
        ue.set_prefill_seq(args.prompt)
    else:
        print(f"[Mode] LM — default prompt from config")
        ue.set_prefill_seq()

    # 4. Load the unified bin. In VLM mode set_prefill_seq_vlm has
    # already done this, but reloading is idempotent (~5 s) and ensures
    # the bin is at base for the LM dispatch that follows.
    manifest = ue.load_instruction_bin()

    # 5. Bucketed prefill + decode.
    # Collapse runs of identical <...>-style special tokens (e.g. 256 ×
    # <|image|>) into "<|image|>*256" form for readability.
    import re as _re_collapse
    _collapse_re = _re_collapse.compile(r'(<[^<>]+>)\1{3,}')
    def _collapse_text_runs(s):
        return _collapse_re.sub(
            lambda m: f"{m.group(1)}*{(m.end() - m.start()) // len(m.group(1))}", s)
    print(f"\n--- Prompt begin ({len(ue.prefill_seq)} tokens) ---")
    print(f"  [{_fmt_prompt_tokens(ue.prefill_seq)}]")
    try:
        _txt = ue.tokenizer.decode(list(ue.prefill_seq), skip_special_tokens=False)
        print(f"  text: {_collapse_text_runs(_txt)!r}")
    except Exception as _e:
        print(f"  text: (decode failed: {_e})")
    print(f"--- Prompt end ---")

    timer = time.perf_counter()
    latency_hw_prefill, flop_rate_hw_prefill = ue.run_prefill_bucketed(manifest)
    latency_prefill = time.perf_counter() - timer
    print(f"Prefill execute done in {latency_prefill:.2f} seconds, start decoding...\n")

    print(f"\n--- Starting decoder ---")
    decoder_base_addr     = manifest["_decoder_addrs_int"][0]
    decoder_program_sizes = manifest["decoder_program_sizes"]
    flops_per_token       = manifest["decoder_total_flops"]
    timer = time.perf_counter()
    token_cnt_decoded, latency_hw_decoder, flop_rate_hw_decoder = ue.run_decoder(
        decoder_program_sizes,
        decoder_base_addr,
        token_id=ue.prefill_seq[-1],
        flops_per_token=flops_per_token,
    )
    latency_decoder = time.perf_counter() - timer

    new_tokens = max(token_cnt_decoded - len(ue.prefill_seq) + 1, 1)
    print(f"\nDecoder done in {latency_prefill + latency_decoder:.2f} seconds, "
          f"speed: {new_tokens / latency_decoder:.2f} tokens/s, "
          f"total {token_cnt_decoded} tokens.")
    print(f"HW counter: latency: {(latency_hw_prefill + latency_hw_decoder) / 1e6:.2f} seconds, "
          f"decoder average GFLOPS: {flop_rate_hw_decoder / new_tokens:.2f}")
    print("Gemma4 E2B run_from_bin ends.")


if __name__ == "__main__":
    main()
