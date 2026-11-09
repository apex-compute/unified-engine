#!/usr/bin/env python3
"""Compare YOLO26s native, Winograd and optimized resident programs."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'yolo26n'))
from benchmark_yolo26n import main

if __name__ == '__main__':
    main(pinned_variant='s')
