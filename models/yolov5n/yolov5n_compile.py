#!/usr/bin/env python3
"""Compile the pinned YOLOv5n model into its direct-run artifact."""

from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parent / "yolov5"
REPO_ROOT = SCRIPT_DIR.parents[1]
for path in (SHARED_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

CONFIG_PATH = SCRIPT_DIR / "yolov5n_config.json"

from yolov5_compile import main as _shared_main


def main(argv=None) -> None:
    _shared_main(argv, pinned_variant="n", config_path=CONFIG_PATH)


if __name__ == "__main__":
    main()
