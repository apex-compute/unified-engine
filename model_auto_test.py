#!/usr/bin/env python3
"""
Model auto-test runner.

Runs each registered model test, captures output, parses metrics, and
writes a summary to test_results.txt.

Usage:
    source myvenv/bin/activate && python model_auto_test.py
"""

import json
import os
import re
import subprocess
import sys
import time
import zlib
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# DRAM Prewriting Tests
# ---------------------------------------------------------------------------
# When enabled, the harness DMA-writes test data across the FULL 4 GiB
# DRAM right before launching each model, so the model (loaded from its bin) starts
# on poisoned DRAM. A PASS then proves run_from_bin (re)writes every byte it reads —
# no per-model edits needed. Toggle off with RANDOMIZE_DRAM=0 in the env to run on
# clean DRAM.
RANDOMIZE_DRAM   = os.environ.get("RANDOMIZE_DRAM", "1") != "0"
RANDOM_DRAM_SEED = int(os.environ.get("RANDOM_DRAM_SEED", "0"))
DMA_DEV          = os.environ.get("DMA_DEV", "xdma0")


def randomize_dram(seed: int = 0, dev: str = "xdma0",
                   chunk_bytes: int = 64 * 1024 * 1024,
                   total_bytes: int = 0x100000000) -> bool:
    """DMA-write data across the full 4 GiB DRAM, then release the
    device, so the model launched next starts on poisoned DRAM.

    Covers the ENTIRE 4 GB DMA-mapped DRAM (0x00000000..0xFFFFFFFF): weight / ISA /
    scratch regions all start as garbage, so a model that still decodes correctly
    proves its run-from-bin path (re)writes every byte it reads.

    Returns True on success. On any failure it returns False; the caller must not
    run the model because that would not exercise the required poisoned-DRAM state.
    """
    try:
        if chunk_bytes <= 0 or total_bytes <= 0:
            raise ValueError("chunk_bytes and total_bytes must be positive")
        import random as _random
        import user_dma_core
        user_dma_core.set_dma_device(dev)
        ue = user_dma_core.UnifiedEngine()           # bare engine: opens device, self-tests
        rng = _random.Random(seed)
        fill_ff = b"\xff" * chunk_bytes
        offset = 0
        while offset < total_bytes:
            n = min(chunk_bytes, total_bytes - offset)
            # data = rng.randbytes(n)  #!!! Original random-data test.
            data = fill_ff[:n]        # Requested all-0xFF test.
            wrote = ue.dma_write(user_dma_core.DMA_DEVICE_H2C, offset, data, n)
            if wrote != n:
                raise RuntimeError(
                    f"dma_write wrote {wrote} of {n} bytes at offset {offset:#x}"
                )
            offset += n
        print(f"[randomize_dram] wrote {total_bytes / 1024**3:.2f} GiB of poison data to "
              f"DRAM [0x00000000..{total_bytes - 1:#010x}]", flush=True)
        del ue                                       # release (dma ops already open/close per call)
        return True
    except Exception as e:
        print(f"[randomize_dram] ERROR: could not poison DRAM ({e}); "
              f"model will not run.", flush=True)
        return False


# ---------------------------------------------------------------------------
# Test registry
# Each entry: script (relative to SCRIPT_DIR), optional prompt override,
# and a pass_check callable: (decoded_text: str) -> (passed: bool, reason: str)
# ---------------------------------------------------------------------------
def _check_x_equals_2(text):
    found = bool(re.search(r"x\s*=\s*2", text))
    return found, (
        "found 'x = 2' in decoded output"
        if found
        else "did not find 'x = 2' in decoded output"
    )

def _check_nonempty(text):
    # Lenient: pass if any non-empty generation came back (didn't crash / NaN-out
    # on poisoned DRAM). Used for encoder models that don't emit LM-style answers.
    found = bool(text.strip())
    return found, ("non-empty generation produced" if found else "empty generation")

def _check_locateanything(text):
    n_boxes = len(re.findall(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", text))
    return n_boxes > 0, (
        f"detected {n_boxes} box(es)"
        if n_boxes > 0
        else "no boxes detected"
    )

def _check_parakeet(text):
    # Pass if any non-empty transcription was produced.
    found = bool(text.strip())
    return found, (
        "non-empty transcription produced"
        if found
        else "empty transcription"
    )

def _check_mbv2_224(text):
    # vette.jpg is a sports car — expect a car-related ImageNet label
    found = bool(re.search(r"sports.car|sport.car|racer|race.car|car", text, re.IGNORECASE))
    return found, (
        f"car-class label found: {text!r}"
        if found
        else f"unexpected label: {text!r}"
    )

def _check_mbv2_ssd(text):
    # vette.jpg should produce at least one detection
    found = text.strip() != "(no detections)"
    return found, (
        f"detections found: {text}"
        if found
        else "no detections above threshold"
    )

# Shared algebra prompt: a single-answer math question whose correct result is
# "x = 2" (checked by _check_x_equals_2). Used for all LM/decoder models below.
MATH_PROMPT = "If x + 3 = 5, what is x?"

TESTS = [
    {"name": "gemma4_e2b",  "script": "models/gemma4_e2b/gemma4_e2b_run_from_bin.py",   "prompt": MATH_PROMPT, "pass_check": _check_x_equals_2},
    {"name": "gemma4_e4b",  "script": "models/gemma4_e4b/gemma4_e4b_run_from_bin.py",   "prompt": MATH_PROMPT, "pass_check": _check_x_equals_2},
    {"name": "llama3.2_1b", "script": "models/llama3.2_1b/llama3.2_1b_run_from_bin.py", "prompt": MATH_PROMPT, "pass_check": _check_x_equals_2},
    {"name": "llama3.2_3b", "script": "models/llama3.2_3b/llama3.2_3b_run_from_bin.py", "prompt": MATH_PROMPT, "pass_check": _check_x_equals_2},
    {"name": "qwen3_1.7b",  "script": "models/qwen3_1.7b/qwen3_1.7b_run_from_bin.py",   "prompt": MATH_PROMPT, "pass_check": _check_x_equals_2},
    {"name": "qwen3_4b",    "script": "models/qwen3_4b/qwen3_4b_run_from_bin.py",       "prompt": MATH_PROMPT, "pass_check": _check_x_equals_2},
    {"name": "qwen3.5_2b",  "script": "models/qwen3.5_2b/qwen3.5_2b_run_from_bin.py",   "prompt": MATH_PROMPT, "pass_check": _check_x_equals_2},
    {"name": "qwen2.5_vl_3b", "script": "models/qwen2.5_vl_3b/qwen2.5_vl_3b_run_from_bin.py", "prompt": MATH_PROMPT, "pass_check": _check_x_equals_2},
    # smolvlm2 DEFAULTS to VLM (loads a bundled image + runs the vision encoder).
    # --lm-enable forces pure language-model (text-only) mode so the algebra prompt
    # is answered as text. The other VL models default to LM when --image is omitted.
    {"name": "smolvlm2",    "script": "models/smolvlm2/smolvlm2_run_from_bin.py",       "prompt": MATH_PROMPT, "pass_check": _check_x_equals_2, "extra_args": ["--lm-enable"]},

    # GPT-2 is a base (non-chat) model: text continuation, no single correct answer,
    # so the check is lenient (non-empty generation). No run_from_bin yet — uses the
    # _test.py entry point; swap to gpt2_run_from_bin.py once it exists.
    {"name": "gpt2",        "script": "models/gpt2/gpt2_test.py",                        "prompt": "The scientists at MIT announced today that they have discovered ", "pass_check": _check_nonempty},

    # Encoder models take no --prompt and emit non-LM output (ASR transcription /
    # segmentation); they also run after the harness poisons DRAM.
    # Enable once their *_bin/ artifacts + sample inputs are in place. Checks are lenient
    # (non-empty == the model didn't crash / NaN-out on poisoned DRAM).
    # {"name": "parakeet",  "script": "models/parakeet/parakeet_run_from_bin.py",       "pass_check": _check_parakeet},
    # {"name": "mobilesam", "script": "models/mobilesam/mobilesam_run_from_bin.py",     "pass_check": _check_nonempty},
]


# ---------------------------------------------------------------------------
# Device reset
# ---------------------------------------------------------------------------

def reset_device(dev: str = "xdma0") -> None:
    """Initialize the FPGA once before the suite: instantiate the engine and
    software-reset it. (Optional zero-fill DRAM with 0x00 is left commented —
    0xff would be NaN in bf16; 0x00 is a safe 0.0 — enable if cross-model
    contamination resurfaces.) Runs in this orchestrator process, not during a
    model's own device access.
    """
    try:
        import user_dma_core
        ue = user_dma_core.UnifiedEngine(device_name=dev)
        ue.software_reset()
        # ue.clear_dram(fill_byte=0x00)
    except Exception as e:
        print(f"[reset_device] WARNING: reset failed: {e}", flush=True)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_test(test: dict, verbose: bool = False) -> dict:
    """Run one model test as a subprocess.

    verbose=True  — stream the model's stdout live, as it prints (tee to console).
    verbose=False — show only the banner + verdict; the model's own run log is
                    captured for parsing but not echoed.
    """
    script = os.path.join(SCRIPT_DIR, test["script"])
    # -u: force the child's stdout unbuffered so verbose mode streams live
    # instead of arriving in big blocks (its stdout is a pipe, not a TTY).
    cmd = [sys.executable, "-u", script]
    if test.get("prompt"):
        cmd += ["--prompt", test["prompt"]]
    cmd += test.get("extra_args", [])
    if test.get("dev_arg"):
        cmd += ["--dev", DMA_DEV]

    print(f"\n{'='*60}")
    print(f"Running test : {test['name']}")
    print(f"Script       : {test['script']}")
    if test.get("prompt"):
        print(f"Prompt       : {test['prompt']}")
    print(f"{'='*60}\n", flush=True)

    # Check immediately before poisoning. Model worktrees are sometimes updated
    # while a long suite is running; do not spend time writing 4 GiB when the
    # subprocess entry point is no longer present.
    if not os.path.isfile(script):
        result = _parse_output(test, "", 1, 0.0)
        result["pass_reason"] = f"model script not found: {test['script']}"
        return result

    # Poison the full 4 GiB DRAM BEFORE the model runs, so loading
    # from bin has to (re)write every byte it reads. Done here in the harness — the
    # model scripts are untouched.
    if RANDOMIZE_DRAM:
        model_seed = (
            RANDOM_DRAM_SEED + zlib.crc32(test["name"].encode("utf-8"))
        ) & 0xFFFFFFFF
        print(f"[randomize_dram] poisoning full 4 GiB DRAM "
              f"(seed={model_seed}) before {test['name']} ...", flush=True)
        poison_start = time.perf_counter()
        poisoned = randomize_dram(seed=model_seed, dev=DMA_DEV)
        poison_elapsed = time.perf_counter() - poison_start
        print(f"{'-'*60}\n", flush=True)
        if not poisoned:
            result = _parse_output(test, "", 1, poison_elapsed)
            result["pass_reason"] = "DRAM poisoning failed; model was not run"
            return result

    t_start = time.perf_counter()
    if verbose:
        # Stream live while still capturing for the pass-check / summary.
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=SCRIPT_DIR,
        )
        captured = []
        for line in proc.stdout:
            print(line, end="", flush=True)
            captured.append(line)
        proc.wait()
        stdout, returncode = "".join(captured), proc.returncode
    else:
        # concise: capture but do not echo the model's run log.
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=SCRIPT_DIR,
        )
        stdout, returncode = proc.stdout, proc.returncode
    elapsed = time.perf_counter() - t_start

    return _parse_output(test, stdout, returncode, elapsed)


def _parse_output(test: dict, stdout: str, returncode: int, elapsed: float) -> dict:
    result = {
        "name": test["name"],
        "returncode": returncode,
        "elapsed_s": elapsed,
        "prefill_text": test.get("prompt", "(default)"),
        "prefill_tokens": None,
        "decoded_text": "",
        "decoded_tokens": None,
        "prefill_speed_tok_s": None,
        "decode_speed_tok_s": None,
        "prefill_size_kb": None,
        "decoder_size_kb": None,
        "passed": False,
        "pass_reason": "",
    }

    # Extract structured result from TEST_RESULT: <json> line emitted by the model script
    for line in stdout.splitlines():
        if line.startswith("TEST_RESULT:"):
            try:
                data = json.loads(line[len("TEST_RESULT:"):].strip())
                result["prefill_tokens"]     = data.get("prefill_tokens")
                result["decoded_text"]       = data.get("decoded_text", "")
                result["decoded_tokens"]     = data.get("decoded_tokens")
                result["prefill_speed_tok_s"] = data.get("prefill_speed_tok_s")
                result["decode_speed_tok_s"] = data.get("decode_speed_tok_s")
                result["prefill_size_kb"]    = data.get("prefill_size_kb")
                result["decoder_size_kb"]    = data.get("decoder_size_kb")
            except json.JSONDecodeError as e:
                result["pass_reason"] = f"TEST_RESULT JSON parse error: {e}"
            break

    # Pass check — run against the FULL captured stdout, not just the parsed
    # TEST_RESULT json. Most models don't emit a TEST_RESULT line yet, so stdout
    # (which contains the live-streamed decoded text) is the source of truth.
    check_text = stdout or result["decoded_text"]
    if returncode != 0:
        result["passed"] = False
        result["pass_reason"] = f"process exited with code {returncode}"
    else:
        # passed, reason = test["pass_check"](result["decoded_text"])  # old: TEST_RESULT json only
        passed, reason = test["pass_check"](check_text)
        result["passed"] = passed
        result["pass_reason"] = reason

    return result


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def write_summary(results: list, output_path: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "Model Auto Test Summary",
        f"Generated : {ts}",
        "=" * 60,
        "",
    ]

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        decoded_preview = r["decoded_text"]
        if decoded_preview and len(decoded_preview) > 300:
            decoded_preview = decoded_preview[:300] + "..."

        prefill_spd = (
            f"{r['prefill_speed_tok_s']:.1f} tok/s"
            if r["prefill_speed_tok_s"] is not None
            else "N/A"
        )
        decode_spd = (
            f"{r['decode_speed_tok_s']:.2f} tok/s"
            if r["decode_speed_tok_s"] is not None
            else "N/A"
        )
        prefill_kb = (
            f"{r['prefill_size_kb']} kB"
            if r["prefill_size_kb"] is not None
            else "N/A"
        )
        decoder_kb = (
            f"{r['decoder_size_kb']} kB"
            if r["decoder_size_kb"] is not None
            else "N/A"
        )

        lines += [
            f"Test             : {r['name']}",
            f"  Result         : {status}",
            f"  Pass reason    : {r['pass_reason']}",
            f"  Prefill text   : {r['prefill_text']}",
            f"  Prefill tokens : {r['prefill_tokens']}",
            f"  Decoded tokens : {r['decoded_tokens']}",
            f"  Decoded text   :",
            *[f"    {line}" for line in decoded_preview.splitlines()],
            f"  Prefill speed  : {prefill_spd}",
            f"  Decode speed   : {decode_spd}",
            f"  Program size   : prefill={prefill_kb}  decoder={decoder_kb}",
            f"  Total time     : {r['elapsed_s']:.1f}s",
            "",
        ]

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    lines += [
        "=" * 60,
        f"Overall: {passed}/{total} passed",
    ]

    text = "\n".join(lines) + "\n"
    with open(output_path, "w") as f:
        f.write(text)

    print(f"\nSummary written to: {output_path}")
    print(text)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", metavar="NAME", help="Run only these named tests")
    ap.add_argument("--verbose", action="store_true",
                    help="Stream each model's live stdout as it runs (full run log)")
    ap.add_argument("--list-names", action="store_true",
                    help="Print the registered test names (space-separated) and exit")
    args = ap.parse_args()

    if args.list_names:
        print(" ".join(t["name"] for t in TESTS))
        return

    summary_path = os.path.join(SCRIPT_DIR, "model_auto_test_results.txt")
    results = []

    # Resolve the test subset (and validate --only names) BEFORE touching the
    # FPGA, so an unknown --only NAME fails fast without resetting the device.
    known_names = {test["name"] for test in TESTS}
    if args.only:
        unknown = sorted(set(args.only) - known_names)
        if unknown:
            ap.error(f"unknown test name(s): {', '.join(unknown)}")
        tests = [test for test in TESTS if test["name"] in args.only]
    else:
        tests = TESTS

    # Initialize the FPGA once before running any model (software reset).
    reset_device()

    for test in tests:
        result = run_test(test, verbose=args.verbose)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n>>> {test['name']}: {status} — {result['pass_reason']}\n")

    write_summary(results, summary_path)

    all_passed = all(r["passed"] for r in results)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
