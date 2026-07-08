#!/usr/bin/env python3
"""Isolate whether the prefix stage's hang lives in flash-attention itself
(bucket-16 dispatch, or the extreme uniform-masking bias pattern) before
re-attempting the full 18-layer compile_prefix.

Two tests, run in sequence, each on its own fresh UnifiedEngine (no state
carried between them) and each with an HONEST heartbeat wrapper (unlike
user_hw_test.flash_attention_test's bare wait_queue(), which does NOT raise
on timeout and would silently "succeed" past a real hang -- see
pi05_libero_test._wait_with_heartbeat for the same fix, replicated here since
this script intentionally does not depend on the pi05_libero model code).

Test 1 (baseline): head_dim=256, seq_len=1024, bucket_idx=16 (the max/last
bucket in a 16-bucket dispatcher -- untested exact value; nearest neighbors
in user_hw_test's sweep are 512 and 4096), MILD random bias (matches
flash_attention_test's own bias_enable=True content). Isolates: does
bucket-16-exactly hang on its own, independent of masking severity?

Test 2 (the real suspect): same shape, but the bias matches compile_prefix's
ACTUAL build_prefix_attn_bias content exactly -- valid_len=16 real columns,
columns [16:1024] (98.4% of the row) masked to -1e36, IDENTICALLY for every
row (not staggered like a causal mask, which is the only masking pattern
proven safe on this hardware so far per locateanything_3b_test.py's
documented -1e36-vs-inf NaN gotcha). Isolates: does the specific "every row
uniformly near-all-masked" arrangement break something the causal case
doesn't exercise?

Usage: python debug_prefix_attn_isolate.py [--test 1|2|both]
"""
import argparse
import math
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from user_dma_core import (
    DMA_DEVICE_H2C, DMA_DEVICE_C2H, UE_FMAX_CONTEXT_SIZE, UE_VECTOR_SIZE,
    UnifiedEngine,
)


def _wait_with_heartbeat(ue, label, timeout=60.0, heartbeat_every=5.0):
    """Same honest-timeout pattern as pi05_libero_test.py's _wait_with_heartbeat:
    wait_queue() does NOT raise on its own timeout, it just prints and returns,
    so a real hang would otherwise look like silent success. This checks
    is_queue_busy() itself afterward and raises for real if still stuck."""
    t0 = time.perf_counter()
    stop = threading.Event()
    def _hb():
        while not stop.wait(heartbeat_every):
            print(f"  [{label}] ... still waiting on FPGA queue ({time.perf_counter()-t0:.0f}s elapsed)", flush=True)
    hb_th = threading.Thread(target=_hb, daemon=True)
    hb_th.start()
    try:
        ue.wait_queue(timeout)
    finally:
        stop.set()
        hb_th.join(timeout=1.0)
    if ue.is_queue_busy():
        raise RuntimeError(
            f"[{label}] FPGA queue still busy after {timeout:.0f}s timeout -- "
            f"real hang, not a silent success.")
    print(f"  [{label}] done in {time.perf_counter()-t0:.1f}s", flush=True)


def _run_flash(label, head_dim, seq_len, bias_matrix, timeout=60.0):
    """Minimal single-shot bucketed-PBI flash attention, mirroring
    flash_attention_test's structure but with an honest wait and a
    caller-supplied bias matrix (so Test 2 can inject the exact
    compile_prefix masking pattern instead of flash_attention_test's mild
    random bias)."""
    print(f"\n=== {label}: head_dim={head_dim} seq_len={seq_len} ===", flush=True)
    ue = UnifiedEngine()
    bpe = 2

    assert seq_len % UE_VECTOR_SIZE == 0
    num_buckets = seq_len // UE_VECTOR_SIZE
    bucket_idx = num_buckets  # the max/last bucket, matching compile_prefix's own S//64 usage

    Q_DRAM_ADDR = ue.allocate_tensor_dram(seq_len * head_dim * bpe)
    K_DRAM_ADDR = ue.allocate_tensor_dram(seq_len * head_dim * bpe)
    V_DRAM_ADDR = ue.allocate_tensor_dram(seq_len * head_dim * bpe)
    BIAS_DRAM_ADDR = ue.allocate_tensor_dram(seq_len * seq_len * bpe)
    SCRATCH_DRAM_ADDR = ue.allocate_tensor_dram(max(head_dim, UE_FMAX_CONTEXT_SIZE) * seq_len * bpe + head_dim * seq_len * bpe)
    ATTN_P_DRAM_ADDR = ue.allocate_tensor_dram(seq_len * seq_len * bpe)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(seq_len * head_dim * bpe)
    IDENTITY_DRAM_ADDR = ue.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe)
    ue.dma_to_accelerator_memory(IDENTITY_DRAM_ADDR, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))

    bucket_reg = ue.alloc_isa_reg()

    ue.start_capture()
    ue.generate_instruction_add_set(bucket_reg, bucket_idx)
    flops_per_bucket = ue.flash_attention_core(
        head_dim=head_dim, seq_len=seq_len,
        Q_DRAM_ADDR=Q_DRAM_ADDR, K_DRAM_ADDR=K_DRAM_ADDR, V_DRAM_ADDR=V_DRAM_ADDR,
        OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR, SCRATCH_DRAM_ADDR=SCRATCH_DRAM_ADDR,
        IDENTITY_DRAM_ADDR=IDENTITY_DRAM_ADDR, BIAS_DRAM_ADDR=BIAS_DRAM_ADDR,
        ATTN_P_DRAM_ADDR=ATTN_P_DRAM_ADDR,
        gpr_bucket_idx=bucket_reg, use_pbi=True, num_buckets=num_buckets,
    )
    ue.release_isa_reg()
    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    q = torch.randn(seq_len, head_dim, dtype=torch.bfloat16)
    k = torch.randn(seq_len, head_dim, dtype=torch.bfloat16)
    v = torch.randn(seq_len, head_dim, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(Q_DRAM_ADDR, q)
    ue.dma_to_accelerator_memory(K_DRAM_ADDR, k)
    ue.dma_to_accelerator_memory(V_DRAM_ADDR, v)
    ue.dma_to_accelerator_memory(BIAS_DRAM_ADDR, bias_matrix)

    ue.start_execute_from_dram(program_dram_addr)
    _wait_with_heartbeat(ue, label, timeout=timeout)

    out = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (seq_len, head_dim))
    has_nan = torch.isnan(out.float()).any().item()
    has_inf = torch.isinf(out.float()).any().item()
    print(f"  output sample: {out.float().flatten()[:6].tolist()}")
    print(f"  has_nan={has_nan} has_inf={has_inf}")
    print(f"  {label}: PASS (no hang, no nan/inf)" if not (has_nan or has_inf) else f"  {label}: RAN but produced NaN/Inf -- numeric bug, not a hang")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["1", "2", "both"], default="both")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    head_dim, seq_len, valid_len = 256, 1024, 16

    if args.test in ("1", "both"):
        mild_bias = torch.randn(seq_len, seq_len, dtype=torch.bfloat16) * 0.1
        _run_flash("test1_bucket16_mild_bias", head_dim, seq_len, mild_bias, timeout=args.timeout)

    if args.test in ("2", "both"):
        # Exact replica of pi05_libero_test.Pi05Libero_UnifiedEngine.build_prefix_attn_bias:
        # every row identically masked past valid_len (uniform, not staggered/causal).
        NEG = -1e36
        real_bias = torch.zeros(seq_len, seq_len, dtype=torch.bfloat16)
        real_bias[:, valid_len:] = NEG
        _run_flash("test2_extreme_uniform_mask", head_dim, seq_len, real_bias, timeout=args.timeout)


if __name__ == "__main__":
    main()
