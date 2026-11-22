#!/usr/bin/env python3
"""YOLO26s entry point; graph/compiler/runtime are shared with YOLO26n."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'yolo26n'))
from yolo26n_test import main

# The harness discovers forwarded flags from quoted literals in each wrapper.
# --run-from-bin is parsed by yolo26n_test.main (sys.argv is forwarded unchanged).
SUPPORTED_RUNTIME_FLAGS = ('--dev', '--device', '--run-from-bin')

if __name__ == '__main__':
    main(pinned_variant='s')
