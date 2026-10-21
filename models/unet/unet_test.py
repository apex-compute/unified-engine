#!/usr/bin/env python3
"""Carvana U-Net CI entry point for model_auto_test.py.

Andromeda two-stage port: unet_compile.py downloads the pinned milesial
checkpoint and builds one fixed-resolution single-HALT bin; unet_run_from_bin.py
executes it. This wrapper gives the harness its one-script contract:

  1. compile the bin (with --download) when it is missing or --force;
  2. run it on the board with test_samples/vette.jpg and --compare, which adds
     logit_rmse and mask_agreement vs the quantized CPU reference to TEST_RESULT;
  3. re-emit that TEST_RESULT with wrapper timings added.

Hardware needs the CONV2D/MAXPOOL queue-CONFIG image (AXI-256 or AXI-512).

Usage:
  python models/unet/unet_test.py --dev xdma0 [--image test_samples/vette.jpg]
                                  [--resolution 256x256] [--force] [--cpu]
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
COMPILE = SCRIPT_DIR / "unet_compile.py"
RUNNER = SCRIPT_DIR / "unet_run_from_bin.py"
BIN_DIR = SCRIPT_DIR / "unet_bin"
BIN = BIN_DIR / "unet-andromeda.bin"
DEFAULT_IMAGE = MODELS_DIR / "test_samples" / "vette.jpg"


def _stream(cmd: list[str]) -> tuple[int, dict | None]:
    """Run cmd, echo its stdout live, and return (rc, last TEST_RESULT json)."""
    print(f"[unet_test] $ {' '.join(cmd)}", flush=True)
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Carvana U-Net on the UnifiedEngine FPGA (CI entry)")
    ap.add_argument("--dev", default="xdma0", help="XDMA device name")
    ap.add_argument("--device", default="rk",
                    help="board profile from the harness; only 'efinix' changes the transport")
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    ap.add_argument("--resolution", default="256x256", metavar="WIDTHxHEIGHT",
                    help="compiled resolution; the bin holds exactly one")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--force", action="store_true", help="rebuild the bin even if present")
    ap.add_argument("--cpu", action="store_true",
                    help="run the quantized CPU backend instead of the FPGA (host smoke test)")
    args = ap.parse_args(argv)

    py = sys.executable
    timings = {}

    if args.force or not BIN.is_file():
        t0 = time.perf_counter()
        cmd = [py, "-u", str(COMPILE), "--download", "--resolution", args.resolution,
               "--output", str(BIN)]
        if args.force:
            cmd.append("--force")
        rc, _ = _stream(cmd)
        timings["compile_s"] = round(time.perf_counter() - t0, 3)
        if rc != 0 or not BIN.is_file():
            print(f"[unet_test] compile failed (rc={rc}); bin missing: {BIN}")
            return rc or 1
    else:
        print(f"[unet_test] reusing bin {BIN}")
        timings["compile_s"] = 0.0

    transport = "efinix" if args.device == "efinix" else "rk"
    output = BIN_DIR / f"ci_{args.image.stem}_{args.resolution}_mask.png"
    cmd = [py, "-u", str(RUNNER), "--bin", str(BIN),
           "--image", str(args.image.resolve()), "--resolution", args.resolution,
           "--output", str(output), "--timeout", str(args.timeout), "--compare"]
    if args.cpu:
        cmd += ["--backend", "cpu-quantized"]
    else:
        cmd += ["--backend", "hardware", "--dev", args.dev, "--device", transport]

    t0 = time.perf_counter()
    rc, result = _stream(cmd)
    timings["run_s"] = round(time.perf_counter() - t0, 3)
    if rc != 0:
        print(f"[unet_test] runner failed (rc={rc})")
        return rc
    if result is None:
        print("[unet_test] runner produced no TEST_RESULT line")
        return 1

    result.update(timings)
    result["ci_entry"] = "unet_test.py"
    result["bin"] = str(BIN)
    print("TEST_RESULT:" + json.dumps(result, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
