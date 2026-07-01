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

def _check_smolvlm2(text):
    # SmolVLM2-500M is too small to reliably solve the algebra prompt (it tends to
    # ramble into broken LaTeX without reaching "x = 2"). Instead it gets a factual
    # question it can handle ("What is the capital of France?") and we check for "Paris".
    found = bool(re.search(r"paris", text, re.IGNORECASE))
    return found, (
        "found 'Paris' in decoded output"
        if found
        else f"did not find 'Paris' in decoded output: {text[:120]!r}"
    )

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

# Each test now declares TWO entry points:
#   compile_script       — always compiles from scratch (its model_bin dir is wiped
#                           before this runs). Proves the compile path works.
#   run_from_bin_script  — loads the program the compile pass just wrote and executes
#                           it (no recompile). Proves the cached-bin path works.
# If a model has no dedicated run_from_bin script yet, run_from_bin_script is None and
# the harness reruns compile_script a second time — its own bin-cache check (see e.g.
# gpt2_test.py) makes that second run a real load-from-bin pass.
TESTS = [
    {"name": "gemma3",      "compile_script": "models/gemma3/gemma3_test.py",                   "run_from_bin_script": None,                                       "prompt": MATH_PROMPT, "pass_check": _check_x_equals_2},
    {"name": "gemma4_e2b",  "compile_script": "models/gemma4_e2b/gemma4_e2b_test.py",            "run_from_bin_script": "models/gemma4_e2b/gemma4_e2b_run_from_bin.py",            "prompt": MATH_PROMPT, "pass_check": _check_x_equals_2},
    {"name": "gemma4_e4b",  "compile_script": "models/gemma4_e4b/gemma4_e4b_test.py",            "run_from_bin_script": "models/gemma4_e4b/gemma4_e4b_run_from_bin.py",            "prompt": MATH_PROMPT, "pass_check": _check_x_equals_2},
    {"name": "llama3.2_1b", "compile_script": "models/llama3.2_1b/llama3.2_1b_test.py",          "run_from_bin_script": "models/llama3.2_1b/llama3.2_1b_run_from_bin.py",          "prompt": MATH_PROMPT, "pass_check": _check_x_equals_2},
    {"name": "llama3.2_3b", "compile_script": "models/llama3.2_3b/llama3.2_3b_test.py",          "run_from_bin_script": "models/llama3.2_3b/llama3.2_3b_run_from_bin.py",          "prompt": MATH_PROMPT, "pass_check": _check_x_equals_2},
    {"name": "qwen3_1.7b",  "compile_script": "models/qwen3_1.7b/qwen3_1.7b_test.py",            "run_from_bin_script": "models/qwen3_1.7b/qwen3_1.7b_run_from_bin.py",            "prompt": MATH_PROMPT, "pass_check": _check_x_equals_2},
    {"name": "qwen3_4b",    "compile_script": "models/qwen3_4b/qwen3_4b_test.py",                "run_from_bin_script": "models/qwen3_4b/qwen3_4b_run_from_bin.py",                "prompt": MATH_PROMPT, "pass_check": _check_x_equals_2},
    {"name": "qwen3.5_2b",  "compile_script": "models/qwen3.5_2b/qwen3.5_2b_test.py",            "run_from_bin_script": "models/qwen3.5_2b/qwen3.5_2b_run_from_bin.py",            "prompt": MATH_PROMPT, "pass_check": _check_x_equals_2},
    {"name": "qwen2.5_vl_3b", "compile_script": "models/qwen2.5_vl_3b/qwen2.5_vl_3b_test.py",    "run_from_bin_script": "models/qwen2.5_vl_3b/qwen2.5_vl_3b_run_from_bin.py",      "prompt": MATH_PROMPT, "pass_check": _check_x_equals_2},
    # smolvlm2 DEFAULTS to VLM (loads a bundled image + runs the vision encoder), so
    # --lm-enable forces pure language-model (text-only) mode. SmolVLM2-500M is too
    # small to reliably do algebra, so it gets a factual question it can answer
    # ("What is the capital of France?" — its built-in default lm_prompt) and we check
    # for "Paris". The other VL models default to LM when --image is omitted.
    {"name": "smolvlm2",    "compile_script": "models/smolvlm2/smolvlm2_test.py",                "run_from_bin_script": "models/smolvlm2/smolvlm2_run_from_bin.py",                "prompt": "What is the capital of France?", "pass_check": _check_smolvlm2, "extra_args": ["--lm-enable"]},

    # GPT-2 is a base (non-chat) model: text continuation, no single correct answer,
    # so the check is lenient (non-empty generation). No run_from_bin yet.
    {"name": "gpt2",        "compile_script": "models/gpt2/gpt2_test.py",                        "run_from_bin_script": None, "prompt": "The scientists at MIT announced today that they have discovered ", "pass_check": _check_nonempty},

    # Vision / detection models: no --prompt; emit detections / class labels parsed
    # from stdout. No run_from_bin yet.
    {"name": "locateanything_3b", "compile_script": "models/locateanything_3b/locateanything_3b_test.py",            "run_from_bin_script": None, "pass_check": _check_locateanything},
    {"name": "mobilenetv2_224",   "compile_script": "models/mobilenetv2/mobilenetv2_224_test.py",                    "run_from_bin_script": None, "pass_check": _check_mbv2_224},
    {"name": "mobilenetv2_ssd",   "compile_script": "models/mobilenetv2/mobilenetv2_ssd_fpnlite_640_test.py",        "run_from_bin_script": None, "pass_check": _check_mbv2_ssd},

    # Encoder models take no --prompt and emit non-LM output (ASR transcription /
    # segmentation). parakeet and mobilesam have dedicated run_from_bin scripts; swin
    # doesn't yet, so it reruns its own _test.py for the second (cached) pass.
    {"name": "parakeet",  "compile_script": "models/parakeet/parakeet_test.py",                "run_from_bin_script": "models/parakeet/parakeet_run_from_bin.py",   "pass_check": _check_parakeet},
    {"name": "mobilesam", "compile_script": "models/mobilesam/mobilesam_test.py",              "run_from_bin_script": "models/mobilesam/mobilesam_run_from_bin.py", "pass_check": _check_nonempty},
    {"name": "swin",      "compile_script": "models/swin/swin_test.py",                        "run_from_bin_script": None,                                          "pass_check": _check_nonempty},
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
# Per-model bin cache cleanup
# ---------------------------------------------------------------------------

def model_bin_dir(script: str) -> str:
    """models/<model>/<model>_test.py -> models/<model>/<model>_bin"""
    d = os.path.dirname(script)
    return os.path.join(d, os.path.basename(d) + "_bin")


def clean_model_bin(script: str) -> None:
    """Delete cached programs.bin/programs.json under this model's bin dir so
    the next run is forced to compile from scratch. Leaves weights untouched."""
    bin_dir = os.path.join(SCRIPT_DIR, model_bin_dir(script))
    removed = []
    for root, _, files in os.walk(bin_dir):
        for f in files:
            if f in ("programs.bin", "programs.json"):
                path = os.path.join(root, f)
                os.remove(path)
                removed.append(path)
    if removed:
        print(f"[clean_model_bin] removed: {', '.join(removed)}", flush=True)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_test(test: dict, script: str, phase: str, verbose: bool = True) -> dict:
    """Run one model entry point (compile or run-from-bin pass) as a subprocess,
    poisoning DRAM immediately before it.

    verbose=True  — stream the model's stdout live, as it prints (tee to console).
                    Default ON: a CI run that fails fast needs to show *why* it
                    failed, not just a bare exit code.
    verbose=False — show only the banner + verdict; the model's own run log is
                    captured for parsing but not echoed.
    """
    script_abs = os.path.join(SCRIPT_DIR, script)
    # -u: force the child's stdout unbuffered so verbose mode streams live
    # instead of arriving in big blocks (its stdout is a pipe, not a TTY).
    cmd = [sys.executable, "-u", script_abs]
    if test.get("prompt"):
        cmd += ["--prompt", test["prompt"]]
    cmd += test.get("extra_args", [])
    if test.get("dev_arg"):
        cmd += ["--dev", DMA_DEV]

    print(f"\n{'='*60}")
    print(f"Running test : {test['name']} [{phase}]")
    print(f"Script       : {script}")
    if test.get("prompt"):
        print(f"Prompt       : {test['prompt']}")
    print(f"{'='*60}\n", flush=True)

    # Check immediately before poisoning. Model worktrees are sometimes updated
    # while a long suite is running; do not spend time writing 4 GiB when the
    # subprocess entry point is no longer present.
    if not os.path.isfile(script_abs):
        result = _parse_output(test, phase, "", 1, 0.0)
        result["pass_reason"] = f"model script not found: {script}"
        return result

    # Poison the full 4 GiB DRAM BEFORE every single run (compile pass AND
    # run-from-bin pass), so each pass has to (re)write every byte it reads.
    if RANDOMIZE_DRAM:
        model_seed = (
            RANDOM_DRAM_SEED + zlib.crc32(f"{test['name']}:{phase}".encode("utf-8"))
        ) & 0xFFFFFFFF
        print(f"[randomize_dram] poisoning full 4 GiB DRAM "
              f"(seed={model_seed}) before {test['name']} [{phase}] ...", flush=True)
        poison_start = time.perf_counter()
        poisoned = randomize_dram(seed=model_seed, dev=DMA_DEV)
        poison_elapsed = time.perf_counter() - poison_start
        print(f"{'-'*60}\n", flush=True)
        if not poisoned:
            result = _parse_output(test, phase, "", 1, poison_elapsed)
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

    result = _parse_output(test, phase, stdout, returncode, elapsed)
    if not result["passed"] and not verbose:
        # Always show the failing run's full output, even in concise mode —
        # a CI run that stops on first failure must show *why* immediately.
        print(stdout)
    return result


def run_model_two_pass(test: dict, verbose: bool = True) -> list:
    """Compile-from-scratch pass, then run-from-cached-bin pass. DRAM is
    randomized before each. Returns the list of phase results produced before
    either a failure or completion (stops early on first failure)."""
    compile_script = test["compile_script"]
    runfrombin_script = test.get("run_from_bin_script") or compile_script

    clean_model_bin(compile_script)

    results = []
    compile_result = run_test(test, compile_script, "compile", verbose=verbose)
    results.append(compile_result)
    if not compile_result["passed"]:
        return results

    runfrombin_result = run_test(test, runfrombin_script, "run_from_bin", verbose=verbose)
    results.append(runfrombin_result)
    return results


# Best-effort regex extraction of HW compile / execute time from a model
# script's stdout. Model scripts report these inconsistently (no shared
# TEST_RESULT fields exist yet), so this tries the known print formats in
# order and sums all matches found (e.g. separate prefill/decoder compiles,
# or a hardware-reported "execution latency" per program).
_COMPILE_TIME_PATTERNS = [
    r"Compile done in ([\d.]+)\s*s",                 # gpt2, gemma3, qwen3_1.7b style
    r"Compile:\s*([\d.]+)\s*s",                       # swin style
    r"(?:prefill|decoder) compiled:.*?([\d.]+)\s*s",  # gpt2 per-stage compiles
]
_EXEC_TIME_PATTERNS = [
    r"Total program execution latency = ([\d.]+)\s*us",  # HW register readout (program_execute)
    r"Executing:\s*([\d.]+)\s*s",                          # swin style
]


def _extract_timing(stdout: str) -> tuple:
    """Returns (compile_time_s, exec_time_s), either None if not found."""
    compile_s = None
    for pat in _COMPILE_TIME_PATTERNS:
        matches = re.findall(pat, stdout)
        if matches:
            compile_s = (compile_s or 0.0) + sum(float(m) for m in matches)

    exec_s = None
    for pat in _EXEC_TIME_PATTERNS:
        matches = re.findall(pat, stdout)
        if not matches:
            continue
        total = sum(float(m) for m in matches)
        if "latency" in pat:  # hardware readout is in microseconds
            total /= 1_000_000.0
        exec_s = (exec_s or 0.0) + total

    return compile_s, exec_s


def _parse_output(test: dict, phase: str, stdout: str, returncode: int, elapsed: float) -> dict:
    compile_time_s, exec_time_s = _extract_timing(stdout)
    result = {
        "name": test["name"],
        "phase": phase,
        "returncode": returncode,
        "elapsed_s": elapsed,
        "executed": returncode == 0,
        "compile_time_s": compile_time_s,
        "exec_time_s": exec_time_s,
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

        compile_t = (
            f"{r['compile_time_s']:.2f}s" if r.get("compile_time_s") is not None else "N/A"
        )
        exec_t = (
            f"{r['exec_time_s']:.3f}s" if r.get("exec_time_s") is not None else "N/A"
        )

        lines += [
            f"Test             : {r['name']} [{r['phase']}]",
            f"  Result         : {status}",
            f"  Executed       : {'YES' if r['executed'] else 'NO'}",
            f"  HW compile time: {compile_t}",
            f"  HW exec time   : {exec_t}",
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

    lines += ["=" * 60, "Summary table", "-" * 60,
              f"{'Model':<22}{'Phase':<14}{'Executed':<10}{'Compile':<12}{'Exec':<12}{'Result':<6}"]
    for r in results:
        compile_t = f"{r['compile_time_s']:.2f}s" if r.get("compile_time_s") is not None else "N/A"
        exec_t = f"{r['exec_time_s']:.3f}s" if r.get("exec_time_s") is not None else "N/A"
        lines.append(
            f"{r['name']:<22}{r['phase']:<14}{'YES' if r['executed'] else 'NO':<10}{compile_t:<12}{exec_t:<12}"
            f"{'PASS' if r['passed'] else 'FAIL':<6}"
        )

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
    ap.add_argument("--quiet", action="store_true",
                    help="Don't stream each model's live stdout (still shown on failure)")
    ap.add_argument("--list-names", action="store_true",
                    help="Print the registered test names (space-separated) and exit")
    ap.add_argument("--no-fail-fast", action="store_true",
                    help="Keep running remaining models after a failure (default: stop immediately)")
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

    stopped_early = False
    for test in tests:
        phase_results = run_model_two_pass(test, verbose=not args.quiet)
        results.extend(phase_results)
        for r in phase_results:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"\n>>> {test['name']} [{r['phase']}]: {status} — {r['pass_reason']}\n")

        if not all(r["passed"] for r in phase_results) and not args.no_fail_fast:
            print(f"!!! Stopping immediately: {test['name']} failed.\n", flush=True)
            stopped_early = True
            break

    write_summary(results, summary_path)
    if stopped_early:
        skipped = [t["name"] for t in tests if t["name"] not in {r["name"] for r in results}]
        if skipped:
            print(f"Skipped (suite stopped early): {', '.join(skipped)}")

    all_passed = bool(results) and all(r["passed"] for r in results)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
