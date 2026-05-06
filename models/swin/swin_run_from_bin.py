#!/usr/bin/env python3
"""Swin-Large-384 inference from pre-compiled bin files.

Compile first (auto-skipped if already compiled):
    python swin_test.py

Then run:
    python swin_run_from_bin.py --image path/to/image.jpg
"""

import builtins
import json
import os
import sys
import threading
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

_original_print = builtins.print
_SILENT_MODE = False

def _quiet_print(*args, **kwargs):
    if not _SILENT_MODE:
        _original_print(*args, **kwargs)

builtins.print = _quiet_print

import numpy as np
import torch
from PIL import Image

import user_dma_core
from user_dma_core import set_dma_device

from swin_test import Swin_UnifiedEngine, preprocess_image


def _set_silent(val: bool) -> None:
    global _SILENT_MODE
    _SILENT_MODE = val


def check_bins_early(script_dir):
    bin_dir = os.path.join(script_dir, "swin_bin")
    missing = []
    for name in ["params.bin", "params.json", "programs.bin", "programs.json", "labels.json"]:
        p = os.path.join(bin_dir, name)
        if not os.path.exists(p):
            missing.append(os.path.join("swin_bin", name))
    return missing


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Swin-Large-384 inference from pre-compiled bins")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--dev", type=str, default="xdma0")
    parser.add_argument("--cycle", type=float, default=5.62)
    args = parser.parse_args()

    missing = check_bins_early(SCRIPT_DIR)
    if missing:
        _original_print("Missing bin files (run python swin_test.py first):")
        for f in missing:
            _original_print(f"  {f}")
        return

    image_path = args.image or os.path.join(SCRIPT_DIR, "../../test_samples/vette.jpg")
    pixel_values = preprocess_image(image_path)

    set_dma_device(args.dev)
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle

    _original_print(f"Swin-Large-384 on {args.dev}")

    t0 = time.perf_counter()
    _set_silent(True)
    ue = Swin_UnifiedEngine(script_dir=SCRIPT_DIR)
    _set_silent(False)
    _original_print(f"  Weights: {time.perf_counter() - t0:.3f}s")

    prog_addr = ue.load_programs()
    if prog_addr is None:
        _original_print("programs.bin not found — run python swin_test.py first.")
        return

    def _progress_timer(start_time, stop_event):
        while not stop_event.wait(1.0):
            _original_print(f"\r  Executing ({time.perf_counter() - start_time:.0f}s)", end="", flush=True)

    t_exec = time.perf_counter()
    stop = threading.Event()
    timer = threading.Thread(target=_progress_timer, args=(t_exec, stop), daemon=True)
    timer.start()

    predicted_idx = ue.run_full_fused(pixel_values, prog_addr)

    stop.set()
    timer.join()
    _original_print(f"\r  Executing: {time.perf_counter() - t_exec:.3f}s")

    labels_path = os.path.join(SCRIPT_DIR, "swin_bin", "labels.json")
    with open(labels_path) as f:
        id2label = json.load(f)
    label = id2label.get(str(predicted_idx), str(predicted_idx))

    _original_print(f"\n  Image: {image_path}")
    _original_print(f"  Prediction: {label!r} (class {predicted_idx})")


if __name__ == "__main__":
    main()
