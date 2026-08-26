#!/usr/bin/env python3
"""Aggregate DRAM bandwidth vs engine count -- the roofline denominator.

user_hw_test's dram_read_speed test measures ONE engine (~5.7 GB/s). That number
is only the right divisor for a 12-engine roofline if the engines' DMA masters
scale independently; if they share one DDR/HBM port, twelve engines get 5.7 GB/s
BETWEEN them and every stage of pi0.5 is far more memory-bound than a per-engine
figure suggests. This measures which it is: N engines each stream their own
private DRAM region concurrently, and we report the aggregate.

    python models/pi05/utility/pi05_bw_scaling.py --max-engines 12
"""
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

import user_dma_core
from user_dma_core import (UnifiedEngine, URAM_NEAR_FULL_ELEMENTS,
                           UE_PIPELINE_COUNTER_CLK_DIV, UE_LATENCY_COUNT_ADDR)

CHUNK_ELEMS = URAM_NEAR_FULL_ELEMENTS      # 262080 bf16 elements
CHUNK_BYTES = CHUNK_ELEMS * 2              # 524160 B, one URAM fill
ENGINE_REG_STRIDE = 0x00010000             # AXI-Lite base stride between engines
ENGINE_DRAM_STRIDE = 0x04000000            # 64 MB of private DRAM per engine


def _build(eng_idxs, region_idxs, n_chunks, repeats, direction):
    """One engine per entry of `eng_idxs`, each streaming from the DRAM window
    named by the matching entry of `region_idxs`. Engine index and memory region
    are decoupled on purpose: with them tied together a fast group could equally
    well be four fast ENGINES or four addresses on a second memory port, and the
    two have completely different consequences for the model."""
    engines, progs = [], []
    for i, r in zip(eng_idxs, region_idxs):
        base = user_dma_core.DRAM_START_ADDR + r * ENGINE_DRAM_STRIDE
        ue = UnifiedEngine(BASE_ADDR=user_dma_core.UE_0_BASE_ADDR + i * ENGINE_REG_STRIDE,
                           params_dram_base=base,
                           tensor_dram_base=base + 0x01000000,
                           program_dram_base=base + 0x03000000)
        buf = ue.allocate_tensor_dram(CHUNK_BYTES * n_chunks)
        ue.start_capture()
        for _ in range(repeats):
            for c in range(n_chunks):
                # Walk distinct blocks so this measures streaming bandwidth and
                # not one DRAM row buffer being hit over and over.
                if direction == "read":
                    ue.accelerator_memory_to_sram(
                        accelerator_dram_address=buf + c * CHUNK_BYTES,
                        sram_address=0x00000, element_size=CHUNK_ELEMS)
                else:
                    ue.sram_to_accelerator_memory(
                        sram_address=0x00000,
                        accelerator_dram_address=buf + c * CHUNK_BYTES,
                        element_size=CHUNK_ELEMS)
        ue.stop_capture()
        ue.generate_instruction_halt()
        addr = ue.get_program_dram_addr()
        ue.write_captured_instructions_to_dram(addr)
        ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())
        ue.clear_capture_buffer()
        engines.append(ue)
        progs.append(addr)
    return engines, progs


def measure(eng_idxs, region_idxs, n_chunks, repeats, direction):
    n_engines = len(eng_idxs)
    engines, progs = _build(eng_idxs, region_idxs, n_chunks, repeats, direction)
    bytes_per_engine = CHUNK_BYTES * n_chunks * repeats

    t0 = time.perf_counter()
    for ue, addr in zip(engines, progs):
        ue.start_execute_from_dram(addr)
    for ue in engines:
        ue.wait_queue(120.0)
    wall = time.perf_counter() - t0

    # Per-engine hardware cycle counter: busy time for that engine alone, free of
    # the host-side launch skew that inflates the wall figure at high N.
    per_engine_gbs = []
    for ue in engines:
        cycles = ue.read_reg32(UE_LATENCY_COUNT_ADDR) * UE_PIPELINE_COUNTER_CLK_DIV
        secs = cycles * ue._clock_period_ns * 1e-9
        per_engine_gbs.append(bytes_per_engine / secs / 1e9 if secs > 0 else 0.0)

    total = bytes_per_engine * n_engines
    return {
        "n": n_engines,
        "eng": list(eng_idxs),
        "reg": list(region_idxs),
        "bytes_total": total,
        "wall_s": wall,
        "agg_gbs_wall": total / wall / 1e9,
        "agg_gbs_hw": sum(per_engine_gbs),
        "per_engine_gbs": per_engine_gbs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="xdma0")
    ap.add_argument("--max-engines", type=int, default=12)
    ap.add_argument("--chunks", type=int, default=32, help="URAM-sized blocks per pass")
    ap.add_argument("--repeats", type=int, default=4, help="passes over those blocks")
    ap.add_argument("--direction", choices=["read", "write", "both"], default="both")
    ap.add_argument("--engine-list", default=None,
                    help="explicit engine indices, e.g. 8,9,10,11 (skips the sweep)")
    ap.add_argument("--region-list", default=None,
                    help="DRAM window indices to pair with --engine-list "
                         "(default: same as the engine indices)")
    args = ap.parse_args()

    user_dma_core.set_dma_device(args.dev)

    counts = [1, 2, 4, 6, 8, 12]
    counts = [c for c in counts if c <= args.max_engines]
    if args.max_engines not in counts:
        counts.append(args.max_engines)

    if args.engine_list:
        eng = [int(x) for x in args.engine_list.split(",")]
        reg = [int(x) for x in args.region_list.split(",")] if args.region_list else list(eng)
        assert len(eng) == len(reg), "--engine-list and --region-list must be the same length"
        cases = [(eng, reg)]
    else:
        cases = [(list(range(n)), list(range(n))) for n in counts]

    dirs = ["read", "write"] if args.direction == "both" else [args.direction]
    for direction in dirs:
        print(f"\n=== DRAM {direction.upper()} bandwidth ===")
        print(f"{'engines':>7} {'MB moved':>10} {'wall s':>8} "
              f"{'agg GB/s (wall)':>16} {'agg GB/s (hw)':>14} {'per-engine GB/s':>16}")
        base = None
        for eng, reg in cases:
            r = measure(eng, reg, args.chunks, args.repeats, direction)
            if base is None:
                base = r["agg_gbs_hw"]
            pe = sum(r["per_engine_gbs"]) / len(r["per_engine_gbs"])
            print(f"{r['n']:>7} {r['bytes_total']/1e6:>10.1f} {r['wall_s']:>8.3f} "
                  f"{r['agg_gbs_wall']:>16.2f} {r['agg_gbs_hw']:>14.2f} {pe:>16.2f}"
                  f"   eng={r['eng']} reg={r['reg']}")


if __name__ == "__main__":
    main()
