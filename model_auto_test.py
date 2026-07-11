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
import shlex
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
# on poisoned DRAM. A PASS then proves model setup writes every byte it reads.
# Toggle off with RANDOMIZE_DRAM=0 in the env to run on clean DRAM.
RANDOMIZE_DRAM   = os.environ.get("RANDOMIZE_DRAM", "1") != "0"
RANDOM_DRAM_SEED = int(os.environ.get("RANDOM_DRAM_SEED", "0"))
DMA_DEV          = os.environ.get("DMA_DEV", "xdma0")


def randomize_dram(seed: int = 0, dev: str = "xdma0",
                   chunk_bytes: int = 64 * 1024 * 1024,
                   total_bytes: int = 0x100000000) -> bool:
    """DMA-write random bf16 values across the full 4 GiB DRAM, then release the
    device, so the model launched next starts on poisoned DRAM.

    Covers the ENTIRE 4 GB DMA-mapped DRAM (0x00000000..0xFFFFFFFF): weight / ISA /
    scratch regions all start as garbage, so a model that still decodes correctly
    proves its run-from-bin path (re)writes every byte it reads.

    Uses random bf16 values in [-8, 8] (same method as randomize_dram.py), NOT an
    all-0xFF fill — 0xFF bytes decode to bf16 NaN, which corrupts any DRAM region
    a model's cache-load path doesn't fully rewrite (masking real bugs instead of
    just exercising the rewrite-every-byte guarantee).

    Returns True on success. On any failure it returns False; the caller must not
    run the model because that would not exercise the required poisoned-DRAM state.
    """
    try:
        if chunk_bytes <= 0 or total_bytes <= 0:
            raise ValueError("chunk_bytes and total_bytes must be positive")
        import torch
        import user_dma_core
        user_dma_core.set_dma_device(dev)
        ue = user_dma_core.UnifiedEngine()           # bare engine: opens device, self-tests
        gen = torch.Generator().manual_seed(seed)
        bpe = 2
        chunk_elements = chunk_bytes // bpe
        offset = 0
        while offset < total_bytes:
            n = min(chunk_bytes, total_bytes - offset)
            take = n // bpe
            data = torch.empty(take, dtype=torch.bfloat16).uniform_(-8.0, 8.0, generator=gen)
            wrote = ue.dma_write(user_dma_core.DMA_DEVICE_H2C, offset, data, take * bpe)
            if wrote != take * bpe:
                raise RuntimeError(
                    f"dma_write wrote {wrote} of {take * bpe} bytes at offset {offset:#x}"
                )
            offset += take * bpe
        print(f"[randomize_dram] wrote {total_bytes / 1024**3:.2f} GiB of random bf16 poison data to "
              f"DRAM [0x00000000..{total_bytes - 1:#010x}]", flush=True)
        del ue                                       # release (dma ops already open/close per call)
        return True
    except Exception as e:
        print(f"[randomize_dram] ERROR: could not poison DRAM ({e}); "
              f"model will not run.", flush=True)
        return False


def zero_dram(dev: str = "xdma0",
              chunk_bytes: int = 64 * 1024 * 1024) -> bool:
    """Zero the FPGA model-working DRAM range before a clean-state-only test."""
    try:
        import user_dma_core
        user_dma_core.set_dma_device(dev)
        ue = user_dma_core.UnifiedEngine()
        start = user_dma_core.DRAM_START_ADDR
        total_bytes = 0x100000000 - start
        zeros = b"\x00" * chunk_bytes
        offset = 0
        while offset < total_bytes:
            n = min(chunk_bytes, total_bytes - offset)
            wrote = ue.dma_write(
                user_dma_core.DMA_DEVICE_H2C, start + offset, zeros[:n], n
            )
            if wrote != n:
                raise RuntimeError(
                    f"dma_write wrote {wrote} of {n} bytes at offset {start + offset:#x}"
                )
            offset += n
        print(
            f"[zero_dram] wrote {total_bytes / 1024**3:.2f} GiB of zeros to "
            f"DRAM [{start:#010x}..0xffffffff]",
            flush=True,
        )
        del ue
        return True
    except Exception as e:
        print(f"[zero_dram] ERROR: could not zero DRAM ({e}); model will not run.",
              flush=True)
        return False


# ---------------------------------------------------------------------------
# Test registry
# Each entry: script (relative to SCRIPT_DIR), optional prompt override,
# and a pass_check callable: (decoded_text: str) -> (passed: bool, reason: str)
# ---------------------------------------------------------------------------
def _test_result_field(text, key):
    """Pull `key` out of a `TEST_RESULT: {json}` line in the model's stdout, or None.

    Model scripts that emit this line hand the harness a structured result to check
    against instead of scraping free-form stdout. Returns None when there is no such
    line or it doesn't parse.
    """
    for line in text.splitlines():
        if line.startswith("TEST_RESULT:"):
            try:
                return json.loads(line[len("TEST_RESULT:"):].strip()).get(key)
            except (json.JSONDecodeError, AttributeError):
                return None
    return None


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
    decoded = _extract_decode_text(text)
    coherent, reason = _score_coherence(decoded)
    if not coherent:
        return False, reason
    car_words = ("car", "vehicle", "sports car", "corvette", "automobile")
    hits = [word for word in car_words if re.search(
        rf"\b{re.escape(word)}\b", decoded, re.IGNORECASE
    )]
    return bool(hits), (
        f"{reason}; vehicle keywords: {', '.join(hits)}"
        if hits
        else f"coherent response did not identify the vehicle: {decoded[:120]!r}"
    )

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

def _extract_decode_text(text):
    """Extract streamed generation from the supported VLM output formats."""
    formats = (
        (
            "------------------------------ DECODE START ------------------------------",
            ("\nDecode speed:", "\nStop token ", "\nDecoder done in"),
        ),
        (
            "  Generated:",
            ("\n  Decode summary:", "\nDecode summary", "\n  Decode speed"),
        ),
        (
            "--- Decode run ---",
            ("\nStop token ", "\nDecoder done in"),
        ),
        (
            "Response:",
            ("\nStop token ", "\nStopped:", "\nDecoder done in"),
        ),
    )
    for marker, stops in formats:
        if marker not in text:
            continue
        decoded = text.split(marker, 1)[1]
        stop_positions = [decoded.find(stop) for stop in stops if stop in decoded]
        if stop_positions:
            decoded = decoded[:min(stop_positions)]
        return _ANSI_RE.sub("", decoded).strip()
    return ""

def _score_coherence(decoded):
    """Heuristic coherence score over already-extracted decode text."""
    if not decoded:
        return False, "no decoded text found"
    words = re.findall(r"[A-Za-z][A-Za-z']{1,}", decoded)
    unique_words = {w.lower() for w in words if len(w) >= 3}
    longest_char_run = max((len(m.group(0)) for m in re.finditer(r"(.)\1{3,}", decoded)), default=0)
    pipe_count = decoded.count("|")
    alpha_ratio = sum(ch.isalpha() for ch in decoded) / max(1, len(decoded))
    coherent = (
        len(words) >= 8
        and len(unique_words) >= 5
        and alpha_ratio >= 0.35
        and longest_char_run <= 24
        and pipe_count <= max(3, len(decoded) // 40)
    )
    preview = decoded[:120].replace("\n", " ")
    return coherent, (
        f"coherent decoded text found: {preview!r}"
        if coherent
        else (
            "decoded text did not look coherent "
            f"(words={len(words)}, unique={len(unique_words)}, "
            f"alpha_ratio={alpha_ratio:.2f}, longest_run={longest_char_run}, "
            f"pipes={pipe_count}): {preview!r}"
        )
    )

def _check_vlm(text, model_name, keywords, minimum_hits=4):
    """Require coherent generated text and enough image-scene keyword hits."""
    decoded = _extract_decode_text(text)
    coherent, reason = _score_coherence(decoded)
    if not coherent:
        return False, reason

    hits = [
        kw for kw in keywords
        if re.search(rf"\b{re.escape(kw)}\b", decoded, re.IGNORECASE)
    ]
    if len(hits) < minimum_hits:
        preview = decoded[:120].replace("\n", " ")
        return False, (
            f"{model_name} VLM: only {len(hits)} scene keyword(s) {hits}; "
            f"need >={minimum_hits}; "
            f"decoded preview: {preview!r}"
        )
    return True, f"{reason}; scene keywords: {', '.join(hits)}"


GEMMA4_VLM_KEYWORDS = ("landscape", "lighting", "sun", "horizon")
QWEN_VLM_KEYWORDS = (
    "mountain", "landscape", "valley", "sky", "sun", "sunrise", "sunset",
    "light", "snow", "tree", "forest", "cliff", "rock", "river", "lake",
    "water", "horizon", "fog", "mist", "peak",
)

def _check_gemma4_e2b_vlm(text):
    return _check_vlm(text, "E2B", GEMMA4_VLM_KEYWORDS)

def _check_gemma4_e4b_vlm(text):
    return _check_vlm(text, "E4B", GEMMA4_VLM_KEYWORDS)

def _check_qwen_vlm(text, model_name="qwen"):
    return _check_vlm(text, model_name, QWEN_VLM_KEYWORDS)

def _check_locateanything(text):
    n_boxes = len(re.findall(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", text))
    return n_boxes > 0, (
        f"detected {n_boxes} box(es)"
        if n_boxes > 0
        else "no boxes detected"
    )

def _check_parakeet(text):
    decoded = _test_result_field(text, "decoded_text")
    if not isinstance(decoded, str):
        return False, "missing decoded_text in TEST_RESULT"
    keywords = ("unified engine", "apex computes", "custom ml inference accelerator")
    matched = [keyword for keyword in keywords if keyword in decoded.lower()]
    if not matched:
        return False, "no expected transcription keywords found"
    if len(matched) < len(keywords):
        missing = [keyword for keyword in keywords if keyword not in matched]
        return True, (
            f"WARNING: matched {len(matched)}/{len(keywords)} transcription keywords; "
            f"missing: {', '.join(missing)}"
        )
    return True, f"all expected transcription keywords found: {decoded!r}"

def _check_swin(text):
    decoded = _test_result_field(text, "decoded_text")
    if not isinstance(decoded, str):
        return False, "missing decoded_text in TEST_RESULT"
    found = bool(re.search(r"\bsports?_car\b", decoded, re.IGNORECASE))
    return found, (
        f"car-class keyword found: {decoded!r}"
        if found
        else f"expected sports_car or sport_car, got: {decoded!r}"
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
    # vette.jpg should produce >=1 detection. The model emits a TEST_RESULT line whose
    # n_detections/decoded_text carry the real result; key on that instead of the raw
    # stdout (which is hundreds of lines and never equals "(no detections)" exactly).
    n = _test_result_field(text, "n_detections")
    labels = _test_result_field(text, "decoded_text")
    if n is not None:
        return n > 0, (
            f"{n} detection(s): {labels}"
            if n > 0
            else "no detections above threshold"
        )
    # Fallback (no TEST_RESULT): look for a printed detection row like "  73.2%  car".
    found = bool(re.search(r"\d+\.\d\s*%\s+\S", text))
    return found, (
        "detection row found in output"
        if found
        else "no detections above threshold"
    )

# Shared algebra prompt: a single-answer math question whose correct result is
# "x = 2" (checked by _check_x_equals_2). Used for all LM/decoder models below.
MATH_PROMPT = "If x + 3 = 5, what is x?"

TESTS = [
    # The deprecated gemma3_test_IF8.py is deliberately excluded: IF8 is
    # currently not working.
    {"name": "gemma3",      "script": "models/gemma3/gemma3_test.py",                   "prompt": MATH_PROMPT, "pass_check": _check_x_equals_2},
    # Run Gemma4 in VLM mode with its built-in default image/prompt and check
    # that the generated decode is coherent text.
    {"name": "gemma4_e2b",  "script": "models/gemma4_e2b/gemma4_e2b_test.py",           "pass_check": _check_gemma4_e2b_vlm, "extra_args": ["--vision-enable"], "mode": "VLM", "image": "test_samples/yosemite.jpg", "prompt_desc": "Describe this image in detail. (default)"},
    {"name": "gemma4_e4b",  "script": "models/gemma4_e4b/gemma4_e4b_test.py",           "pass_check": _check_gemma4_e4b_vlm, "extra_args": ["--vision-enable"], "mode": "VLM", "image": "test_samples/yosemite.jpg", "prompt_desc": "Describe this image in detail. (default)"},
    {"name": "llama3.2_1b", "script": "models/llama3.2_1b/llama3.2_1b_test.py", "prompt": MATH_PROMPT, "pass_check": _check_x_equals_2},
    {"name": "llama3.2_3b", "script": "models/llama3.2_3b/llama3.2_3b_test.py", "prompt": MATH_PROMPT, "pass_check": _check_x_equals_2},
    {"name": "qwen3_1.7b",  "script": "models/qwen3_1.7b/qwen3_1.7b_test.py",   "prompt": MATH_PROMPT, "pass_check": _check_x_equals_2},
    {"name": "qwen3_4b",    "script": "models/qwen3_4b/qwen3_4b_test.py",       "prompt": MATH_PROMPT, "pass_check": _check_x_equals_2},
    {"name": "qwen3.5_2b",  "script": "models/qwen3.5_2b/qwen3.5_2b_test.py",   "prompt": MATH_PROMPT, "pass_check": _check_x_equals_2},
    # qwen3.5_2b VLM: FPGA vision encoder (--vision-enable defaults to on-FPGA vision)
    # on yosemite.jpg; gemma4-style criteria (coherent decode + scene keywords).
    {"name": "qwen3.5_2b_vlm", "script": "models/qwen3.5_2b/qwen3.5_2b_test.py",        "pass_check": _check_qwen_vlm, "extra_args": ["--vision-enable", "--vision-on-hardware"], "mode": "VLM", "image": "test_samples/yosemite.jpg", "prompt_desc": "Describe what you see in this image. (default)"},
    {"name": "qwen2.5_vl_3b", "script": "models/qwen2.5_vl_3b/qwen2.5_vl_3b_test.py", "pass_check": _check_qwen_vlm, "extra_args": ["--vision-enable"], "mode": "VLM", "image": "test_samples/yosemite.jpg", "prompt_desc": "Describe the image in detail. (default)"},
    # SmolVLM2 has a read-before-write defect and depends on clean, zero-filled
    # DRAM. Rather than special-casing it in the harness, SmolVLM2 is poisoned
    # before its run like every other model; smolvlm2_test.py zeroes DRAM itself
    # at startup (ue.zero_dram()) before loading weights. Remove that self-zero
    # once the read-before-write gap is fixed.
    {"name": "smolvlm2", "script": "models/smolvlm2/smolvlm2_test.py", "pass_check": _check_smolvlm2, "mode": "VLM", "image": "test_samples/vette.jpg", "prompt_desc": "Describe this image. (default)"},
    # TODO: SMOLVLM2 LM PATH IS BUGGY — restore the smolvlm2_lm auto-test only after it is fixed.

    # Vision / detection models: no --prompt; emit detections / class labels parsed
    # from stdout.
    {"name": "locateanything_3b", "script": "models/locateanything_3b/locateanything_3b_test.py",            "pass_check": _check_locateanything},
    {"name": "mobilenetv2_224",   "script": "models/mobilenetv2/mobilenetv2_224_test.py",                    "pass_check": _check_mbv2_224},
    {"name": "mobilenetv2_ssd",   "script": "models/mobilenetv2/mobilenetv2_ssd_fpnlite_640_test.py",        "pass_check": _check_mbv2_ssd},

    # Encoder models take no --prompt and emit non-LM output (ASR transcription /
    # segmentation).
    {"name": "parakeet",  "script": "models/parakeet/parakeet_test.py",        "pass_check": _check_parakeet},
    {"name": "mobilesam", "script": "models/mobilesam/mobilesam_test.py",      "pass_check": _check_nonempty},
    {"name": "swin",      "script": "models/swin/swin_test.py",                        "pass_check": _check_swin},
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

def _script_supports_flag(script_path: str, flag: str) -> bool:
    """True when the model script declares `flag` in its own argparse.

    Not every entry point accepts --dev / --device yet, and argparse hard-fails
    on unknown flags, so the harness forwards a flag only after sniffing the script source for the quoted
    literal ('--dev' won't false-match '--device' because the closing quote is
    part of the pattern).
    """
    try:
        with open(script_path, "r", encoding="utf-8", errors="ignore") as f:
            src = f.read()
    except OSError:
        return False
    return f"'{flag}'" in src or f'"{flag}"' in src


def run_test(test: dict, verbose: bool = False,
             dev: str = None, device: str = None) -> dict:
    """Run one model test as a subprocess.

    verbose=True  — stream the model's stdout live, as it prints (tee to console).
    verbose=False — show only the banner + verdict; the model's own run log is
                    captured for parsing but not echoed.
    dev / device  — DMA device name / FPGA board profile, forwarded as --dev /
                    --device to model scripts that declare the flag (see
                    _script_supports_flag). None means don't forward.
    """
    rel_script = test["script"]
    script = os.path.join(SCRIPT_DIR, rel_script)
    # -u: force the child's stdout unbuffered so verbose mode streams live
    # instead of arriving in big blocks (its stdout is a pipe, not a TTY).
    cmd = [sys.executable, "-u", script]
    if test.get("prompt"):
        cmd += ["--prompt", test["prompt"]]
    cmd += test.get("extra_args", [])
    unsupported = []
    for flag, value in (("--dev", dev), ("--device", device)):
        if value is None:
            continue
        if _script_supports_flag(script, flag):
            cmd += [flag, value]
        else:
            unsupported.append(f"{flag} {value}")

    print(f"\n{'='*60}")
    print(f"Running test : {test['name']}")
    print(f"Script       : {rel_script}")
    display_cmd = [os.path.basename(sys.executable), "-u", rel_script] + cmd[3:]
    print(f"Command      : {shlex.join(display_cmd)}")
    if test.get("mode"):
        print(f"Mode         : {test['mode']}")
    print("Execution    : FPGA")
    if test.get("image"):
        print(f"Image        : {test['image']}")
    if test.get("prompt_desc"):
        print(f"Prompt       : {test['prompt_desc']}")
    if unsupported:
        print(f"Note         : script does not accept {', '.join(unsupported)} — not forwarded")
    if test.get("prompt") and not test.get("prompt_desc"):
        print(f"Prompt       : {test['prompt']}")
    print(f"{'='*60}\n", flush=True)

    # Check immediately before poisoning. Model worktrees are sometimes updated
    # while a long suite is running; do not spend time writing 4 GiB when the
    # subprocess entry point is no longer present.
    if not os.path.isfile(script):
        result = _parse_output(test, "", 1, 0.0)
        result["pass_reason"] = f"model script not found: {rel_script}"
        return result

    # Dormant per-test escape hatch: a test may set clear_dram_zero to start from
    # zero instead of the suite's normal 0xFF poison (for a model with a known
    # read-before-write defect that cannot self-zero). No test currently uses it —
    # SmolVLM2 now poisons normally and zeroes DRAM inside its own script.
    if test.get("clear_dram_zero"):
        print(f"[zero_dram] {test['name']} requested clean DRAM; "
              "zeroing before run ...", flush=True)
        clear_start = time.perf_counter()
        cleared = zero_dram(dev=DMA_DEV)
        clear_elapsed = time.perf_counter() - clear_start
        print(f"{'-'*60}\n", flush=True)
        if not cleared:
            result = _parse_output(test, "", 1, clear_elapsed)
            result["pass_reason"] = "DRAM zero-fill failed; model was not run"
            return result
    # Normal models start from full 4 GiB of 0xFF poison so setup must rewrite
    # every byte it reads.
    elif RANDOMIZE_DRAM:
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
        # Stream live while still capturing for the pass-check / summary. Read raw
        # bytes: os.read() returns as soon as ANY data is available (it does NOT wait
        # for a newline), so partial lines and \r progress updates appear the instant
        # the child emits them. If the screen stops updating, the child is genuinely
        # stalled — not just buffered behind a missing newline.
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=0, cwd=SCRIPT_DIR,
        )
        captured = []
        fd = proc.stdout.fileno()
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:                       # EOF — child closed stdout
                break
            text = chunk.decode("utf-8", errors="replace")
            sys.stdout.write(text)
            sys.stdout.flush()
            captured.append(text)
        proc.wait()
        stdout, returncode = "".join(captured), proc.returncode
    else:
        # concise: capture but do not echo the model's run log. Still print a
        # heartbeat so a long compile / prefill / decode does not look hung.
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=SCRIPT_DIR,
        )
        heartbeat_s = 10.0
        while True:
            try:
                stdout, _ = proc.communicate(timeout=heartbeat_s)
                break
            except subprocess.TimeoutExpired:
                elapsed_now = time.perf_counter() - t_start
                print(f"[running] {test['name']} still running ... {elapsed_now:.0f}s",
                      flush=True)
        returncode = proc.returncode
    elapsed = time.perf_counter() - t_start

    return _parse_output(test, stdout, returncode, elapsed)


def _parse_output(test: dict, stdout: str, returncode: int, elapsed: float) -> dict:
    result = {
        "name": test["name"],
        "stdout": stdout,
        "returncode": returncode,
        "elapsed_s": elapsed,
        "prefill_text": test.get("prompt") or test.get("prompt_desc", "(default)"),
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
    if not result["decoded_text"]:
        result["decoded_text"] = _extract_decode_text(stdout)
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
    global DMA_DEV
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", metavar="NAME", help="Run only these named tests")
    ap.add_argument("--verbose", action="store_true",
                    help="Stream each model's live stdout as it runs (full run log)")
    ap.add_argument("--list-names", action="store_true",
                    help="Print the registered test names (space-separated) and exit")
    ap.add_argument("--dev", type=str, default=DMA_DEV,
                    help="DMA device name (e.g., xdma0, xdma1). Used for the harness's "
                         "own reset / DRAM poisoning and forwarded to model scripts "
                         f"that accept --dev. Default: {DMA_DEV} (env DMA_DEV)")
    ap.add_argument("--device", type=str, default=None,
                    help="FPGA board / bitstream profile (kintex7, rk, puzhi, bittware, "
                         "bittware_256, alveo): affects UE_AXI_DATA_WIDTH_BITS and default "
                         "--cycle. Forwarded to model scripts that accept --device only when "
                         "explicitly provided.")
    args = ap.parse_args()

    # --dev also selects the device the harness itself resets and poisons.
    DMA_DEV = args.dev

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
    reset_device(DMA_DEV)

    try:
        for test in tests:
            result = run_test(test, verbose=args.verbose,
                              dev=args.dev, device=args.device)
            results.append(result)
            status = "PASS" if result["passed"] else "FAIL"
            print(f"\n>>> {test['name']}: {status} — {result['pass_reason']}\n")
            if not result["passed"]:
                if not args.verbose and result.get("stdout"):
                    print(f"{'='*60}")
                    print(f"Failing run log: {test['name']}")
                    print(f"{'='*60}")
                    print(result["stdout"], end="" if result["stdout"].endswith("\n") else "\n")
                    print(f"{'='*60}")
                    print("Stopping on first failure.")
                elif args.verbose:
                    print("Stopping on first failure.")
                else:
                    print("Stopping on first failure; no model run log was captured.")
                break
    except KeyboardInterrupt:
        print("\n[interrupted] writing summary for completed tests so far ...")
    finally:
        if results:
            write_summary(results, summary_path)

    all_passed = bool(results) and all(r["passed"] for r in results)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
