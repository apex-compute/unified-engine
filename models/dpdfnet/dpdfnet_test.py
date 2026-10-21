#!/usr/bin/env python3
"""DPDFNet2 (16 kHz speech enhancement) CI entry point for model_auto_test.py.

Andromeda two-stage port: dpdfnet_compile.py downloads CEVA's pinned ONNX graph
and builds one resident streaming bin; dpdfnet_run_from_bin.py enhances a WAV on
the FPGA, one START/HALT per 10 ms hop. This wrapper gives the harness its
one-script contract:

  1. compile the bin (with --download) when it is missing or --force;
  2. enhance test_samples/apex.wav on the board;
  3. unless --no-ref, enhance the same clip with the onnxruntime CPU reference
     (dpdfnet_run_cpu.py) and add snr_db_vs_cpu / relative_l2_vs_cpu;
  4. re-emit the runner's TEST_RESULT with those metrics and timings added.

Hardware needs the RK AXI-256 CONV/MAXPOOL queue-CONFIG image.

Usage:
  python models/dpdfnet/dpdfnet_test.py --dev xdma0 [--input test_samples/apex.wav]
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
COMPILE = SCRIPT_DIR / "dpdfnet_compile.py"
RUNNER = SCRIPT_DIR / "dpdfnet_run_from_bin.py"
CPU_REF = SCRIPT_DIR / "dpdfnet_run_cpu.py"
BIN_DIR = SCRIPT_DIR / "dpdfnet_bin"
BIN = BIN_DIR / "dpdfnet2-andromeda.bin"
DEFAULT_INPUT = MODELS_DIR / "test_samples" / "apex.wav"


def _stream(cmd: list[str], tag: str = "dpdfnet_test") -> tuple[int, dict | None]:
    """Run cmd, echo its stdout live, and return (rc, last TEST_RESULT json)."""
    print(f"[{tag}] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=str(MODELS_DIR.parent), env={**os.environ, "PYTHONUNBUFFERED": "1"})
    result = None
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.startswith("TEST_RESULT:"):
            try:
                result = json.loads(line[len("TEST_RESULT:"):].strip())
            except json.JSONDecodeError:
                pass
            continue
        print(line, end="", flush=True)
    return proc.wait(), result


def _wav_metrics(reference: Path, candidate: Path) -> dict:
    """SNR / relative L2 of candidate vs reference on original sample indices."""
    import numpy as np
    import soundfile as sf

    ref, ref_rate = sf.read(reference, dtype="float64", always_2d=True)
    got, got_rate = sf.read(candidate, dtype="float64", always_2d=True)
    if ref_rate != got_rate or ref.shape != got.shape:
        return {"ref_compare_error": f"shape/rate mismatch: ref {ref.shape}@{ref_rate} "
                                     f"vs fpga {got.shape}@{got_rate}"}
    if not np.isfinite(got).all():
        return {"finite_output": False, "snr_db_vs_cpu": None, "relative_l2_vs_cpu": None}
    err = got - ref
    ref_e = float(np.sum(ref ** 2))
    err_e = float(np.sum(err ** 2))
    # JSON has no inf: an exact match reports snr None + exact_match True.
    snr = 10.0 * np.log10(ref_e / err_e) if ref_e > 0 and err_e > 0 else None
    return {
        "finite_output": True,
        "exact_match": bool(err_e == 0),
        "snr_db_vs_cpu": None if snr is None else float(snr),
        "relative_l2_vs_cpu": float(np.sqrt(err_e / ref_e)) if ref_e > 0 else None,
        "cpu_reference_wav": str(reference),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DPDFNet2 16 kHz on the UnifiedEngine FPGA (CI entry)")
    ap.add_argument("--dev", default="xdma0", help="XDMA device name")
    ap.add_argument("--device", default="rk",
                    help="board profile from the harness; only 'efinix' changes the transport")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="WAV/FLAC clip")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--force", action="store_true", help="rebuild the bin even if present")
    ap.add_argument("--no-ref", action="store_true",
                    help="skip the onnxruntime CPU reference and the SNR comparison")
    args = ap.parse_args(argv)

    py = sys.executable
    timings = {}

    if args.force or not BIN.is_file():
        t0 = time.perf_counter()
        cmd = [py, "-u", str(COMPILE), "--download", "--output", str(BIN)]
        if args.force:
            cmd.append("--force")
        rc, _ = _stream(cmd)
        timings["compile_s"] = round(time.perf_counter() - t0, 3)
        if rc != 0 or not BIN.is_file():
            print(f"[dpdfnet_test] compile failed (rc={rc}); bin missing: {BIN}")
            return rc or 1
    else:
        print(f"[dpdfnet_test] reusing bin {BIN}")
        timings["compile_s"] = 0.0

    transport = "efinix" if args.device == "efinix" else "rk"
    stem = args.input.stem
    fpga_wav = BIN_DIR / f"ci_{stem}_16khz_fpga.wav"
    cmd = [py, "-u", str(RUNNER), "--bin", str(BIN), "--input", str(args.input.resolve()),
           "--output", str(fpga_wav), "--dev", args.dev, "--device", transport,
           "--timeout", str(args.timeout)]
    t0 = time.perf_counter()
    rc, result = _stream(cmd)
    timings["run_s"] = round(time.perf_counter() - t0, 3)
    if rc != 0:
        print(f"[dpdfnet_test] runner failed (rc={rc})")
        return rc
    if result is None:
        print("[dpdfnet_test] runner produced no TEST_RESULT line")
        return 1

    if not args.no_ref:
        cpu_wav = BIN_DIR / f"ci_{stem}_16khz_cpu.wav"
        t0 = time.perf_counter()
        rc, _ = _stream([py, "-u", str(CPU_REF), "--download", "--input",
                         str(args.input.resolve()), "--output", str(cpu_wav)], "dpdfnet_test/cpu")
        timings["cpu_ref_s"] = round(time.perf_counter() - t0, 3)
        if rc != 0 or not cpu_wav.is_file():
            result["ref_compare_error"] = f"CPU reference failed (rc={rc})"
        else:
            result.update(_wav_metrics(cpu_wav, fpga_wav))
            m = result.get("snr_db_vs_cpu")
            print(f"[dpdfnet_test] FPGA vs CPU reference: SNR "
                  f"{'n/a' if m is None else f'{m:.2f} dB'}, "
                  f"relative L2 {result.get('relative_l2_vs_cpu')}", flush=True)

    result.update(timings)
    result["ci_entry"] = "dpdfnet_test.py"
    result["bin"] = str(BIN)
    print("TEST_RESULT:" + json.dumps(result, separators=(",", ":"), allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
