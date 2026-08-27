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
                     pattern: str = "random", rng_seed: int = 0) -> None:
    """Fill the full 4 GB DRAM with one of three patterns.

    ``pattern``:
      ``"random"``  random bf16 in [-8, 8]. The useful poison: every
                    uninitialised read yields an obviously wrong, finite value.
                    Seeded by ``rng_seed`` so a failure is REPRODUCIBLE and
                    therefore bisectable.
      ``"0"``       all 0x00 bytes -> bf16 +0.0. The most FORGIVING state: a
                    read-before-write on zeros usually looks benign, so a pass
                    here proves nothing. Use it to confirm a suspected
                    read-before-write disappears, not to validate.
      ``"0xff"``    all 0xFF bytes -> bf16 NaN. Harshest: NaN propagates through
                    any arithmetic, so it flags regions a load path fails to
                    rewrite -- but it also masks which region was at fault, and
                    poisons caches that legitimately rewrite later.
    """
    dram_total_bytes = DRAM_END - DRAM_BASE
    total_elements = dram_total_bytes // BPE
    chunk_bytes = chunk_elements * BPE
    offset = 0
    bar_width = 40

    if pattern == "random":
        torch.manual_seed(rng_seed)
        desc = f"random bf16 values in [-8, 8] (seed={rng_seed})"
    elif pattern == "0":
        desc = "0x00 bytes (bf16 +0.0)"
    elif pattern == "0xff":
        desc = "0xFF bytes (bf16 NaN)"
    else:
        raise ValueError(f"unknown pattern {pattern!r}")
    byte_fill = None if pattern == "random" else (
        b"\x00" if pattern == "0" else b"\xff") * chunk_bytes

    print(f"Resetting DRAM [{hex(DRAM_BASE)}..{hex(DRAM_END)}) with {desc} "
          f"({dram_total_bytes / 1024**3:.2f} GB, chunk={chunk_bytes / 1024**2:.0f} MB)")

    while offset < total_elements:
        take = min(chunk_elements, total_elements - offset)
        if byte_fill is None:
            data = torch.empty(take, dtype=torch.bfloat16)
            data.uniform_(-8.0, 8.0)
        else:
            data = byte_fill[: take * BPE]
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
    parser = argparse.ArgumentParser(description="DRAM fill/reset test")
    parser.add_argument("--dev", default="xdma0", help="DMA device (default: xdma0)")
    parser.add_argument("--chunk-mb", type=int, default=64,
                        help="DMA chunk size in MB (default: 64)")
    parser.add_argument("--seed", choices=("0", "0xff", "random"), default="random",
                        help="fill pattern: '0' = all 0x00 (bf16 +0.0, most "
                             "forgiving); '0xff' = all 0xFF (bf16 NaN, harshest); "
                             "'random' = random bf16 in [-8, 8] (default, the "
                             "useful poison)")
    parser.add_argument("--rng-seed", type=int, default=0,
                        help="RNG seed for --seed random, so a failure reproduces "
                             "(default: 0)")
    args = parser.parse_args()

    set_dma_device(args.dev)

    ue = UnifiedEngine()

    chunk_elements = (args.chunk_mb * 1024 * 1024) // BPE
    dram_random_fill(ue, chunk_elements, pattern=args.seed, rng_seed=args.rng_seed)


if __name__ == "__main__":
    main()
