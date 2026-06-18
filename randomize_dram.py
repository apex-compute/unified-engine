#!/usr/bin/env python3
"""DRAM reset with random bf16 values (not zeros, not small numbers).

Populates the full 4 GB DRAM space [0x00000000..0x100000000) with random
bfloat16 data via DMA writes. Useful before running HW tests so that
uninitialised reads hit random data rather than stale NaN/zero patterns.

Usage:
    python dram_reset_test.py [--dev xdma0] [--chunk-mb 64]
"""

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from user_dma_core import (
    DMA_DEVICE_H2C,
    DMA_DEVICE_C2H,
    set_dma_device,
    UnifiedEngine,
)

DRAM_BASE = 0x00000000
DRAM_END  = 0x100000000  # 4 GB
BPE = 2


def dram_random_fill(ue: UnifiedEngine, chunk_elements: int = 32 * 1024 * 1024) -> None:
    dram_total_bytes = DRAM_END - DRAM_BASE
    total_elements = dram_total_bytes // BPE
    chunk_bytes = chunk_elements * BPE
    offset = 0
    bar_width = 40

    print(f"Resetting DRAM [{hex(DRAM_BASE)}..{hex(DRAM_END)}) with random bf16 values "
          f"({dram_total_bytes / 1024**3:.2f} GB, chunk={chunk_bytes / 1024**2:.0f} MB)")

    while offset < total_elements:
        take = min(chunk_elements, total_elements - offset)
        data = torch.empty(take, dtype=torch.bfloat16)
        data.uniform_(-8.0, 8.0)
        ue.dma_write(DMA_DEVICE_H2C, DRAM_BASE + offset * BPE, data, take * BPE)
        offset += take
        pct = offset / total_elements
        filled = int(bar_width * pct)
        bar = '#' * filled + '.' * (bar_width - filled)
        print(f"\r  [{bar}] {pct*100:5.1f}%  {offset/1024**2:.0f}/{total_elements/1024**2:.0f} Melem", end='', flush=True)
    print()

    # Quick verification: read back first and last elements
    v0 = ue.dma_from_accelerator_memory(DRAM_BASE, (2,))
    vN = ue.dma_from_accelerator_memory(DRAM_END - 4, (2,))
    print(f"  First element: {v0[0].item():.4f}  Last element: {vN[0].item():.4f}")
    print("DRAM reset done.")


def main():
    parser = argparse.ArgumentParser(description="DRAM random-fill reset test")
    parser.add_argument("--dev", default="xdma0", help="DMA device (default: xdma0)")
    parser.add_argument("--chunk-mb", type=int, default=64,
                        help="DMA chunk size in MB (default: 64)")
    args = parser.parse_args()

    set_dma_device(args.dev)

    ue = UnifiedEngine()

    chunk_elements = (args.chunk_mb * 1024 * 1024) // BPE
    dram_random_fill(ue, chunk_elements)


if __name__ == "__main__":
    main()
