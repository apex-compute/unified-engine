#!/usr/bin/env python3
"""YOLOv5s CI entry point for model_auto_test.py.

The Andromeda port ships as two stages: yolov5s_compile.py builds the single
fixed-address bin from the pinned v7.0 checkpoint, and yolov5s_run_from_bin.py
executes that bin on the FPGA. The harness runs ONE script per model and expects
a TEST_RESULT: JSON line, so this wrapper:

  1. compiles the bin when it is missing (or --force), via the compile script;
  2. runs the bin on the board with a fixed test_samples/ image;
  3. re-emits the runner's TEST_RESULT with wrapper timings added.

Pass 1 of the nightly compiles + runs; pass 2 (shuffled run-from-bin) finds the
bin on disk and only runs. Hardware needs the CONV2D/MAXPOOL queue-CONFIG image.

Usage:
  python models/yolov5s/yolov5s_test.py --dev xdma0 [--image test_samples/vette.jpg]
                                        [--resolution 640x480] [--force] [--cpu]
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
COMPILE = SCRIPT_DIR / "yolov5s_compile.py"
RUNNER = SCRIPT_DIR / "yolov5s_run_from_bin.py"
CONFIG = SCRIPT_DIR / "yolov5s_config.json"
DEFAULT_IMAGE = MODELS_DIR / "test_samples" / "vette.jpg"


def _bin_path() -> Path:
    with CONFIG.open("r", encoding="utf-8") as stream:
        return (SCRIPT_DIR / json.load(stream)["paths"]["artifact"]).resolve()


def _stream(cmd: list[str]) -> tuple[int, dict | None]:
    """Run cmd, echo its stdout live, and return (rc, last TEST_RESULT json)."""
    print(f"[yolov5s_test] $ {' '.join(cmd)}", flush=True)
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
            continue  # re-emitted (merged) by the wrapper below
        print(line, end="", flush=True)
    return proc.wait(), result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="YOLOv5s on the UnifiedEngine FPGA (CI entry)")
    ap.add_argument("--dev", default="xdma0", help="XDMA device name")
    ap.add_argument("--device", default="rk",
                    help="board profile from the harness; only 'efinix' changes the transport")
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    ap.add_argument("--resolution", default="640x480", metavar="WIDTHxHEIGHT")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--force", action="store_true", help="rebuild the bin even if present")
    ap.add_argument("--cpu", action="store_true",
                    help="run the quantized CPU backend instead of the FPGA (host smoke test)")
    args = ap.parse_args(argv)

    py = sys.executable
    bin_path = _bin_path()
    timings = {}

    if args.force or not bin_path.is_file():
        t0 = time.perf_counter()
        rc, _ = _stream([py, "-u", str(COMPILE)] + (["--force"] if args.force else []))
        timings["compile_s"] = round(time.perf_counter() - t0, 3)
        if rc != 0 or not bin_path.is_file():
            print(f"[yolov5s_test] compile failed (rc={rc}); bin missing: {bin_path}")
            return rc or 1
    else:
        print(f"[yolov5s_test] reusing bin {bin_path}")
        timings["compile_s"] = 0.0

    transport = "efinix" if args.device == "efinix" else "rk"
    out_dir = bin_path.parent
    output = out_dir / f"ci_{args.image.stem}_{args.resolution}.jpg"
    cmd = [py, "-u", str(RUNNER), "--bin", str(bin_path),
           "--image", str(args.image.resolve()), "--resolution", args.resolution,
           "--output", str(output), "--timeout", str(args.timeout)]
    if args.cpu:
        cmd += ["--cpu"]
    else:
        cmd += ["--dev", args.dev, "--device", transport]

    t0 = time.perf_counter()
    rc, result = _stream(cmd)
    timings["run_s"] = round(time.perf_counter() - t0, 3)
    if rc != 0:
        print(f"[yolov5s_test] runner failed (rc={rc})")
        return rc
    if result is None:
        print("[yolov5s_test] runner produced no TEST_RESULT line")
        return 1

    result.update(timings)
    result["ci_entry"] = "yolov5s_test.py"
    result["bin"] = str(bin_path)
    print("TEST_RESULT:" + json.dumps(result, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
