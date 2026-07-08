#!/usr/bin/env python3
"""Isolated repro: does compile_encoder() hang/misbehave on its SECOND call
within the same process? Skips prefix/action-expert entirely (no weight_init
for those stacks) to test whether the slot-1 hang is intrinsic to repeated
compile_encoder() invocations, or an artifact of accumulated pipeline state
from prefix/action-expert running first.

Usage:
    python repro_vision_twice.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pi05_libero_test import (
    Pi05Libero_UnifiedEngine, init_hang_prevention, DMA_DEVICE_H2C, DMA_DEVICE_C2H,
)
import numpy as np
import torch


def main():
    ue = Pi05Libero_UnifiedEngine()
    init_hang_prevention(ue)

    print(f"[repro] isa_reg_counter before anything: {ue._isa_reg_counter}")

    # Only vision weights -- skip prefix/action-expert entirely.
    ue._weight_init_vision()
    print(f"[repro] params DRAM after vision weight_init: 0x{ue.get_params_dram_addr():X}")

    # tensor_init() also allocates the (unused-by-vision) LM base buffers before
    # calling _tensor_init_vision() at the end (IS_VLM branch) -- keep this call
    # as-is to match the real pipeline's tensor layout exactly, so this repro is
    # a faithful subset rather than a differently-shaped test.
    ue.tensor_init(1024)
    print(f"[repro] tensor DRAM after tensor_init: 0x{ue.get_tensor_dram_addr():X}")
    print(f"[repro] isa_reg_counter after tensor_init: {ue._isa_reg_counter}")
    print(f"[repro] program DRAM base before any compile_encoder call: 0x{ue.get_program_dram_addr():X}")

    # One real sample patch, reused for both calls (content doesn't matter for
    # this repro -- we're testing execution behavior, not correctness).
    S, PK = ue.VIS_S, ue.VIS_PATCH_K
    patches_pad = np.zeros((S, PK), dtype=np.float32)
    pixel_t = torch.from_numpy(patches_pad).to(torch.bfloat16)

    for i in range(3):
        print(f"\n[repro] ===== call {i} =====")
        print(f"[repro] isa_reg_counter before compile_encoder: {ue._isa_reg_counter}")
        print(f"[repro] program DRAM addr before compile_encoder: 0x{ue.get_program_dram_addr():X}")

        ue.dma_write(DMA_DEVICE_H2C, ue.VIS_PIXEL_IN_DRAM, pixel_t, S * PK * 2)

        t0 = time.perf_counter()
        prog_addr = ue.compile_encoder()
        t_compile = time.perf_counter() - t0
        print(f"[repro] call {i}: compiled in {t_compile:.2f}s, program at 0x{prog_addr:X}")
        print(f"[repro] isa_reg_counter after compile_encoder: {ue._isa_reg_counter}")

        t1 = time.perf_counter()
        ue.start_execute_from_dram(prog_addr)
        ue._wait_with_heartbeat(f"repro call {i}", timeout=180.0)
        t_exec = time.perf_counter() - t1
        print(f"[repro] call {i}: execute+wait took {t_exec:.2f}s")

        buf = bytearray(S * ue.VIS_HEAD_OUT * 2)
        ue.dma_read(DMA_DEVICE_C2H, ue.VIS_HEAD_OUT_DRAM, buf, len(buf))
        out = torch.frombuffer(bytes(buf), dtype=torch.bfloat16)
        print(f"[repro] call {i}: readback ok, nonzero elems = {(out != 0).sum().item()}/{out.numel()}")

    print("\n[repro] all 3 calls completed without hanging.")


if __name__ == "__main__":
    main()
