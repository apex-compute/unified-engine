#!/usr/bin/env python3
"""BigCodec (speech codec round trip) CI entry point for model_auto_test.py.

Andromeda port in three stages: bigcodec_fetch.py downloads the pinned
checkpoint, bigcodec_compile.py builds one whole-utterance bin sized to the
input clip (compiled_samples is baked in), and bigcodec_run_from_bin.py encodes
+ quantizes + decodes that clip on the FPGA. This wrapper gives the harness its
one-script contract:

  1. fetch the checkpoint and compile the bin for test_samples/apex.wav when the
     bin is missing or --force (README's compensated-decoder flag set);
  2. run the round trip on the board;
  3. unless --no-ref, run the CPU round trip (bigcodec_run_cpu.py) and compare
     waveforms + tokens with bigcodec_compare.py: snr_db, relative_l2,
     cosine_similarity, token match_fraction;
  4. re-emit the runner's TEST_RESULT with those metrics and timings added.

Hardware needs the RK AXI-256 CONV/MAXPOOL queue-CONFIG image.

Usage:
  python models/bigcodec/bigcodec_test.py --dev xdma0 [--input test_samples/apex.wav]
                                          [--force] [--no-ref]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
FETCH = SCRIPT_DIR / "bigcodec_fetch.py"
COMPILE = SCRIPT_DIR / "bigcodec_compile.py"
RUNNER = SCRIPT_DIR / "bigcodec_run_from_bin.py"
CPU_REF = SCRIPT_DIR / "bigcodec_run_cpu.py"
COMPARE = SCRIPT_DIR / "bigcodec_compare.py"
BIN_DIR = SCRIPT_DIR / "bigcodec_bin"
CHECKPOINT = BIN_DIR / "bigcodec.pt"
BIN = BIN_DIR / "bigcodec-andromeda.bin"
DEFAULT_INPUT = MODELS_DIR / "test_samples" / "apex.wav"

# README's recommended hardware build: BF16 dense weights + compensated decoder math.
COMPILE_FLAGS = [
    "--conv-precision", "bf16",
    "--lstm-cell-precision", "compensated", "--lstm-tanh-precision", "compensated",
    "--lstm-fused-gates", "--lstm-math-scope", "decoder",
    "--center-quantizer-scores", "--compensated-codebook",
]


def _stream(cmd: list[str], tag: str = "bigcodec_test") -> tuple[int, dict | None, str]:
    """Run cmd, echo its stdout live; return (rc, last TEST_RESULT json, full stdout)."""
    print(f"[{tag}] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=str(MODELS_DIR.parent), env={**os.environ, "PYTHONUNBUFFERED": "1"})
    result, captured = None, []
    assert proc.stdout is not None
    for line in proc.stdout:
        captured.append(line)
        if line.startswith("TEST_RESULT:"):
            try:
                result = json.loads(line[len("TEST_RESULT:"):].strip())
            except json.JSONDecodeError:
                pass
            continue
        print(line, end="", flush=True)
    return proc.wait(), result, "".join(captured)


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _discard_corrupt(path: Path, *, zip_archive: bool = False, sha256: str | None = None, tag: str) -> None:
    """Delete a cached artifact that a box reboot mid-write left truncated.

    The compile scripts treat an existing file as authoritative (and refuse to
    overwrite a checkpoint with a bad hash), so a half-written file would be
    reused forever. torch.save bins and .pt/.pth checkpoints are zip archives;
    ONNX / other files are checked by SHA256 when one is pinned.
    """
    if not path.is_file():
        return
    ok = True
    if zip_archive:
        import zipfile
        try:
            ok = zipfile.ZipFile(path).testzip() is None
        except Exception:
            ok = False
    if ok and sha256 is not None:
        ok = _sha256(path) == sha256
    if not ok:
        print(f"[{tag}] cached {path.name} is truncated/corrupt; deleting so it is rebuilt", flush=True)
        path.unlink()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="BigCodec on the UnifiedEngine FPGA (CI entry)")
    ap.add_argument("--dev", default="xdma0", help="XDMA device name")
    ap.add_argument("--device", default="rk",
                    help="board profile from the harness; BigCodec only supports the rk transport")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                    help="WAV clip; the bin is compiled for exactly this clip's length")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--force", action="store_true", help="rebuild the bin even if present")
    ap.add_argument("--no-ref", action="store_true",
                    help="skip the CPU round trip and the waveform/token comparison")
    args = ap.parse_args(argv)

    py = sys.executable
    clip = args.input.resolve()
    timings = {}
    manifest = json.loads((SCRIPT_DIR / "upstream_manifest.json").read_text())
    _discard_corrupt(BIN, zip_archive=True, tag="bigcodec_test")
    _discard_corrupt(CHECKPOINT, zip_archive=True, sha256=manifest["checkpoint"]["sha256"],
                     tag="bigcodec_test")

    if args.force or not BIN.is_file():
        t0 = time.perf_counter()
        if not CHECKPOINT.is_file():
            rc, _, _ = _stream([py, "-u", str(FETCH), "--output", str(CHECKPOINT)])
            if rc != 0 or not CHECKPOINT.is_file():
                print(f"[bigcodec_test] checkpoint fetch failed (rc={rc})")
                return rc or 1
        cmd = [py, "-u", str(COMPILE), "--input", str(clip), "--checkpoint", str(CHECKPOINT),
               "--output", str(BIN)] + COMPILE_FLAGS
        if args.force:
            cmd.append("--force")
        rc, _, _ = _stream(cmd)
        timings["compile_s"] = round(time.perf_counter() - t0, 3)
        if rc != 0 or not BIN.is_file():
            print(f"[bigcodec_test] compile failed (rc={rc}); bin missing: {BIN}")
            return rc or 1
    else:
        print(f"[bigcodec_test] reusing bin {BIN}")
        timings["compile_s"] = 0.0

    fpga_wav = BIN_DIR / f"ci_{clip.stem}_fpga.wav"
    fpga_tokens = BIN_DIR / f"ci_{clip.stem}_fpga.tokens.npz"
    cmd = [py, "-u", str(RUNNER), "--bin", str(BIN), "--input", str(clip),
           "--output", str(fpga_wav), "--tokens", str(fpga_tokens),
           "--dev", args.dev, "--device", "rk", "--timeout", str(args.timeout)]
    t0 = time.perf_counter()
    rc, result, _ = _stream(cmd)
    timings["run_s"] = round(time.perf_counter() - t0, 3)
    if rc != 0:
        print(f"[bigcodec_test] runner failed (rc={rc})")
        return rc
    if result is None:
        print("[bigcodec_test] runner produced no TEST_RESULT line")
        return 1

    if not args.no_ref:
        cpu_wav = BIN_DIR / f"ci_{clip.stem}_cpu.wav"
        cpu_tokens = BIN_DIR / f"ci_{clip.stem}_cpu.tokens.npz"
        t0 = time.perf_counter()
        rc, _, _ = _stream([py, "-u", str(CPU_REF), "--mode", "roundtrip", "--input", str(clip),
                            "--output", str(cpu_wav), "--tokens", str(cpu_tokens),
                            "--checkpoint", str(CHECKPOINT)], "bigcodec_test/cpu")
        timings["cpu_ref_s"] = round(time.perf_counter() - t0, 3)
        if rc != 0 or not cpu_wav.is_file():
            result["ref_compare_error"] = f"CPU reference failed (rc={rc})"
        else:
            report = BIN_DIR / f"ci_{clip.stem}_compare.json"
            rc, _, out = _stream([py, "-u", str(COMPARE), "--reference", str(cpu_wav),
                                  "--actual", str(fpga_wav), "--reference-tokens", str(cpu_tokens),
                                  "--actual-tokens", str(fpga_tokens), "--report", str(report)],
                                 "bigcodec_test/compare")
            if rc != 0 or not report.is_file():
                result["ref_compare_error"] = f"bigcodec_compare.py failed (rc={rc})"
            else:
                cmp = json.loads(report.read_text())
                result["finite_output"] = bool(cmp.get("finite", False))
                for key in ("snr_db", "relative_l2", "cosine_similarity", "max_abs_error"):
                    result[f"{key}_vs_cpu"] = cmp.get(key)
                tokens = cmp.get("tokens") or {}
                result["token_match_fraction"] = tokens.get("match_fraction")
                result["token_mismatches"] = tokens.get("mismatches")
                result["compare_report"] = str(report)
                snr = result["snr_db_vs_cpu"]
                print(f"[bigcodec_test] FPGA vs CPU: SNR {'n/a' if snr is None else f'{snr:.2f} dB'}, "
                      f"tokens match {result['token_match_fraction']}", flush=True)

    result.update(timings)
    result["ci_entry"] = "bigcodec_test.py"
    result["bin"] = str(BIN)
    print("TEST_RESULT:" + json.dumps(result, separators=(",", ":"), allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
