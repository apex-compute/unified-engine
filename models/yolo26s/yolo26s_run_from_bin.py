#!/usr/bin/env python3
"""YOLO26s entry point; graph/compiler/runtime are shared with YOLO26n."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'yolo26n'))
from yolo26n_run_from_bin import main

if __name__ == '__main__':
    main(pinned_variant='s')
