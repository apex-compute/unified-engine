#!/usr/bin/env python3
"""YOLOv5n CI entry point for model_auto_test.py.

Same two-stage Andromeda port as YOLOv5s (shared code lives in models/yolov5s/):
yolov5n_compile.py builds the single fixed-address bin from the pinned v7.0
checkpoint and yolov5n_run_from_bin.py executes it on the FPGA. This wrapper
gives the harness the one-script contract it expects:

  1. compile the bin when it is missing (or --force);
  2. run the bin on the board with a fixed test_samples/ image;
  3. re-emit the runner's TEST_RESULT with wrapper timings added.

Hardware needs the CONV2D/MAXPOOL queue-CONFIG image.

Usage:
  python models/yolov5n/yolov5n_test.py --dev xdma0 [--image test_samples/people.jpg]
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
COMPILE = SCRIPT_DIR / "yolov5n_compile.py"
RUNNER = SCRIPT_DIR / "yolov5n_run_from_bin.py"
CONFIG = SCRIPT_DIR / "yolov5n_config.json"
DEFAULT_IMAGE = MODELS_DIR / "test_samples" / "people.jpg"


def _config() -> dict:
    with CONFIG.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _bin_path() -> Path:
    return (SCRIPT_DIR / _config()["paths"]["artifact"]).resolve()


def _stream(cmd: list[str]) -> tuple[int, dict | None]:
    """Run cmd, echo its stdout live, and return (rc, last TEST_RESULT json)."""
    print(f"[yolov5n_test] $ {' '.join(cmd)}", flush=True)
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
    ap = argparse.ArgumentParser(description="YOLOv5n on the UnifiedEngine FPGA (CI entry)")
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
    cfg = _config()
    _discard_corrupt(bin_path, zip_archive=True, tag="yolov5n_test")
    _discard_corrupt((SCRIPT_DIR / cfg["paths"]["weights"]).resolve(), zip_archive=True,
                     sha256=cfg["source"]["weights_sha256"], tag="yolov5n_test")

    if args.force or not bin_path.is_file():
        t0 = time.perf_counter()
        rc, _ = _stream([py, "-u", str(COMPILE)] + (["--force"] if args.force else []))
        timings["compile_s"] = round(time.perf_counter() - t0, 3)
        if rc != 0 or not bin_path.is_file():
            print(f"[yolov5n_test] compile failed (rc={rc}); bin missing: {bin_path}")
            return rc or 1
    else:
        print(f"[yolov5n_test] reusing bin {bin_path}")
        timings["compile_s"] = 0.0

    transport = "efinix" if args.device == "efinix" else "rk"
    output = bin_path.parent / f"ci_{args.image.stem}_{args.resolution}.jpg"
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
        print(f"[yolov5n_test] runner failed (rc={rc})")
        return rc
    if result is None:
        print("[yolov5n_test] runner produced no TEST_RESULT line")
        return 1

    result.update(timings)
    result["ci_entry"] = "yolov5n_test.py"
    result["bin"] = str(bin_path)
    print("TEST_RESULT:" + json.dumps(result, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
