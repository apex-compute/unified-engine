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


def dram_random_fill(ue: UnifiedEngine, chunk_elements: int = 32 * 1024 * 1024,
                     value: str = "random") -> None:
    """Fill the whole 4 GB DRAM so an uninitialised read downstream is obvious.

    ``value``:
      ``random``  bf16 in [-8, 8]. Every read-before-write yields a
                  wrong-but-FINITE number, so a bug shows up as bad output you
                  can still trace numerically.
      ``ff``      every byte 0xFF, i.e. every element NaN. NaN propagates, so a
                  read-before-write is impossible to miss -- but it destroys the
                  numeric detail, and it also makes any *partly* written buffer
                  fatal rather than merely wrong.
      ``zero``    every byte 0x00. The benign control: pair it with ``ff`` and a
                  model that decodes under ``zero`` but not ``ff`` is reading
                  something it never fully wrote.
    """
    dram_total_bytes = DRAM_END - DRAM_BASE
    total_elements = dram_total_bytes // BPE
    chunk_bytes = chunk_elements * BPE
    offset = 0
    bar_width = 40

    what = {"random": "random bf16 values", "ff": "0xFF bytes (bf16 NaN)",
            "zero": "0x00 bytes"}[value]
    print(f"Resetting DRAM [{hex(DRAM_BASE)}..{hex(DRAM_END)}) with {what} "
          f"({dram_total_bytes / 1024**3:.2f} GB, chunk={chunk_bytes / 1024**2:.0f} MB)")

    while offset < total_elements:
        take = min(chunk_elements, total_elements - offset)
        if value == "random":
            data = torch.empty(take, dtype=torch.bfloat16)
            data.uniform_(-8.0, 8.0)
        else:
            # bf16 has no value spelling for the all-ones pattern (a literal NaN
            # packs as 0x7FC0), so build the bits as int16 and reinterpret;
            # dma_write ships bf16 bits verbatim.
            fill = -1 if value == "ff" else 0
            data = torch.full((take,), fill, dtype=torch.int16).view(torch.bfloat16)
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
    parser = argparse.ArgumentParser(description="DRAM fill / reset test")
    parser.add_argument("--dev", default="xdma0", help="DMA device (default: xdma0)")
    parser.add_argument("--value", choices=("random", "zero", "ff"), default="random",
                        help="Fill value: 'random' bf16 in [-8,8] (default), 'zero' all "
                             "0x00, 'ff' all 0xFF (every element NaN). Run 'zero' and 'ff' "
                             "back to back to tell a read-before-write from a real bug.")
    parser.add_argument("--chunk-mb", type=int, default=64,
                        help="DMA chunk size in MB (default: 64)")
    args = parser.parse_args()

    set_dma_device(args.dev)

    ue = UnifiedEngine()

    chunk_elements = (args.chunk_mb * 1024 * 1024) // BPE
    dram_random_fill(ue, chunk_elements, args.value)


if __name__ == "__main__":
    main()
