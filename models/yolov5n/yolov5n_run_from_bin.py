#!/usr/bin/env python3
"""Run YOLOv5n directly from its compiled Andromeda artifact."""

from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parent / "yolov5s"
REPO_ROOT = SCRIPT_DIR.parents[1]
for path in (SHARED_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

CONFIG_PATH = SCRIPT_DIR / "yolov5n_config.json"
# Advertise delegated options to model_auto_test.py's source-level flag probe.
HARNESS_FORWARDED_FLAGS = ("--dev", "--device")

from yolov5s_run_from_bin import main as _shared_main


def main(argv=None) -> None:
    _shared_main(argv, pinned_variant="n", config_path=CONFIG_PATH)


if __name__ == "__main__":
    main()
