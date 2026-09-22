#!/usr/bin/env python3
"""Q/K/V projections at 80 lanes/head (1280) instead of 128 (2048), SigLIP layer.

The 128-lane pad is only needed INSIDE the per-head Q.K^T operand (its contraction
must be 64-aligned). So project Q/K/V at 16 x 80 = 1280 (72 real + 8 zero lanes per
head) and pad to 128 during the per-head gather that already exists:

  qkv2048  CURRENT: Q/K/V (256 x 1152) -> 2048; per head, gather 256 B chunks
           (128 lanes) jumping 4096 B into FQ/FK/FV (256 x 128), contiguous write.
  qkv1280  Q/K/V (256 x 1152) -> 1280; per head, gather 160 B chunks (80 lanes)
           jumping 2560 B, then a STRIDED write (160 B chunks, 256 B jump) into
           FQ/FK/FV. Their lanes 80..127 are zeroed once and never written, so the
           Q.K^T operand is still a zero-padded 128. Both sides are whole AXI beats.

Both then run split80 attention (pi05_attn_kernels.attention_split_hd) into the
(256 x 1280) O-proj input. One engine, full S = 256 rows (the whole slot).

Checks: projections of the real lanes bit-identical; FQ/FK/FV bit-identical
(including zero pad lanes); final attention output bit-identical; SNR vs a CPU
reference built from the same dequantized IF4 weights.

Run from the repo root:
    python models/pi05/utility/pi05_qkv80_bench.py [--reps 3]
"""
import argparse
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
sys.path.insert(0, _HERE)

import torch

import user_dma_core
from user_dma_core import UnifiedEngine, TYPE, calculate_snr
from nn_lib import store_weight, store_identity_matrix
from pi05_attn_kernels import attention_split_hd
from pi05_unpad_bench import q4_dequant, store_q4

S, H, NH, D, DP, L = 256, 1152, 16, 72, 128, 80
CLK_NS = 1000.0 / 366.67
bpe = 2


def run(ue, label, emit, results):
    ue.clear_capture_buffer()
    ue.start_capture()
    emit()
    ue.stop_capture()
    ue.generate_instruction_halt()
    prog = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(prog)
    nbytes = ue.get_capture_instruction_size_bytes()
    ue.allocate_program_dram(nbytes)
    ue.start_execute_from_dram(prog)
    ue.wait_queue(120.0)
    cycles, n_inst = ue.report_timing_and_instruction_count()
    results[label] = {"cycles": cycles, "inst": n_inst, "prog_bytes": nbytes}
    ue.clear_capture_buffer()


def pad_heads(w_kn, b, lanes, scale=1.0):
    """(1152, 16*72) weight + (16*72) bias -> (16*lanes, 1152) N x K, zero pad lanes."""
    k = w_kn.reshape(H, NH, D)
    kp = torch.zeros(H, NH, lanes)
    kp[:, :, :D] = k * scale
    bp = torch.zeros(NH, lanes)
    bp[:, :D] = b.reshape(NH, D) * scale
    return kp.reshape(H, NH * lanes).T.contiguous().to(torch.bfloat16), bp.reshape(-1).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    user_dma_core.set_dma_device("xdma0")
    user_dma_core.configure_clock_from_hardware()
    ue = UnifiedEngine(clock_period_ns=CLK_NS)
    ident = store_identity_matrix(ue)

    torch.manual_seed(0)
    x = torch.randn(S, H).to(torch.bfloat16)                       # LayerNorm output stand-in
    raw = {p: ((torch.randn(H, NH * D) / H ** 0.5), torch.randn(NH * D) * 0.1) for p in "qkv"}
    q_scale = math.sqrt(DP / D)                                      # as _weight_init_vision

    W = {}
    for lanes in (DP, L):
        for p in "qkv":
            wk, bb = pad_heads(*raw[p], lanes, q_scale if p == "q" else 1.0)
            W[(lanes, p)] = (store_q4(ue, wk), store_weight(ue, bb), q4_dequant(wk), bb.float())

    buf = {"X": ue.allocate_tensor_dram(S * H * bpe)}
    ue.dma_to_accelerator_memory(buf["X"], x)
    for lanes in (DP, L):
        for p in "qkv":
            # +256 B slack: head 15's 256 B read on the LAST row runs 96 B past the end.
            buf[(lanes, p)] = ue.allocate_tensor_dram(S * NH * lanes * bpe + 256)
        for f in ("FQ", "FK", "FV"):
            buf[(lanes, f)] = ue.allocate_tensor_dram(S * DP * bpe)
            ue.dma_to_accelerator_memory(buf[(lanes, f)], torch.zeros(S, DP, dtype=torch.bfloat16))
        buf[(lanes, "OUT")] = ue.allocate_tensor_dram(S * NH * L * bpe)
        ue.dma_to_accelerator_memory(buf[(lanes, "OUT")], torch.full((S, NH * L), 7.0, dtype=torch.bfloat16))
    buf["SCR"] = ue.allocate_tensor_dram((DP * S + S * S + S * DP) * bpe)
    buf["BIAS"] = ue.allocate_tensor_dram(S * S * bpe)
    ue.dma_to_accelerator_memory(buf["BIAS"], torch.zeros(S, S, dtype=torch.bfloat16))

    def proj(lanes):
        r = ue.alloc_isa_reg()
        ue.generate_instruction_add_set(r, S)
        for p in "qkv":
            (sc, da), bias = W[(lanes, p)][0], W[(lanes, p)][1]
            ue.matmat_mul_core(M=S, K=H, N=NH * lanes, A_DRAM_ADDR=buf["X"], B_DRAM_ADDR=da,
                               SCALE_DRAM_ADDR=sc, C_DRAM_ADDR=bias, bias_mode="broadcast_N",
                               OUTPUT_DRAM_ADDR=buf[(lanes, p)], is_B_quantized=True,
                               data_type=TYPE.IF4, gpr_M_reg=r)
        ue.release_isa_reg()

    def gather(lanes, h):
        pitch = NH * lanes * bpe
        for p, f in (("q", "FQ"), ("k", "FK"), ("v", "FV")):
            src = buf[(lanes, p)] + h * lanes * bpe
            if lanes == DP:          # current: whole 128-lane block, contiguous write
                ue.accelerator_memory_to_sram(src, 0x00000, S * DP,
                                              stride_bytes_per_chunk=DP * bpe, stride_jump_bytes=pitch)
                ue.sram_to_accelerator_memory(0x00000, buf[(lanes, f)], S * DP)
            else:
                # MEASURED: a strided WRITE consumes one whole 256 B SRAM slot per 160 B
                # chunk, while a 160 B strided READ packs chunks at 160 B. So read FULL
                # 256 B chunks (this head's 80 lanes + the next head's first 48 --
                # never written out) to give every row its own 256 B slot, then write
                # only the first 160 B of each slot. FQ/FK/FV lanes 80..127 stay zero.
                ue.accelerator_memory_to_sram(src, 0x00000, S * DP,
                                              stride_bytes_per_chunk=DP * bpe, stride_jump_bytes=pitch)
                ue.sram_to_accelerator_memory(0x00000, buf[(lanes, f)], S * L,
                                              stride_bytes_per_chunk=L * bpe, stride_jump_bytes=DP * bpe)

    def attn(lanes, h):
        attention_split_hd(ue, S, S, DP, L, buf[(lanes, "FQ")], buf[(lanes, "FK")], buf[(lanes, "FV")],
                           buf["BIAS"], buf[(lanes, "OUT")] + h * L * bpe, buf["SCR"], ident,
                           out_row_stride=NH * L)

    res = {}
    for lanes, tag in ((DP, "qkv2048"), (L, "qkv1280")):
        run(ue, f"{tag}: projections", lambda: [proj(lanes) for _ in range(args.reps)], res)
        run(ue, f"{tag}: 16 head gathers",
            lambda: [gather(lanes, h) for _ in range(args.reps) for h in range(NH)], res)
        run(ue, f"{tag}: full (proj+gather+attn)",
            lambda: [(proj(lanes), [(gather(lanes, h), attn(lanes, h)) for h in range(NH)])
                     for _ in range(args.reps)], res)

    rd = lambda a, shape: ue.dma_from_accelerator_memory(a, shape).float()
    checks = {}
    for p in "qkv":
        a = rd(buf[(DP, p)], (S, NH, DP))[:, :, :D]
        b = rd(buf[(L, p)], (S, NH, L))
        checks[f"{p} proj real lanes"] = (a - b[:, :, :D]).abs().max().item()
        checks[f"{p} proj 1280 pad lanes (must be 0)"] = b[:, :, D:].abs().max().item()
    for f in ("FQ", "FK", "FV"):          # last head's operands, incl. the zero pad lanes
        checks[f"{f} (head 15) 2048 vs 1280"] = (rd(buf[(DP, f)], (S, DP)) - rd(buf[(L, f)], (S, DP))).abs().max().item()
    if os.environ.get("QKV_DUMP"):
        import numpy as np
        np.savez(os.environ["QKV_DUMP"],
                 **{f"{f}_{lanes}": rd(buf[(lanes, f)], (S, DP)).numpy()
                    for lanes in (DP, L) for f in ("FQ", "FK", "FV")},
                 **{f"{p}_{lanes}": rd(buf[(lanes, p)], (S, NH * lanes)).numpy()
                    for lanes in (DP, L) for p in "qkv"})
    oa, ob = rd(buf[(DP, "OUT")], (S, NH, L)), rd(buf[(L, "OUT")], (S, NH, L))
    checks["attention output 2048 vs 1280"] = (oa - ob).abs().max().item()

    # CPU reference from the SAME dequantized IF4 weights (80-lane set, pad lanes zero).
    qkv = {p: (x.float() @ W[(L, p)][2].T + W[(L, p)][3]).to(torch.bfloat16).float().reshape(S, NH, L)
           for p in "qkv"}
    snrs = []
    for h in range(NH):
        sc = qkv["q"][:, h] @ qkv["k"][:, h].T / math.sqrt(DP)
        ref = torch.softmax(sc, -1) @ qkv["v"][:, h, :D]
        snrs.append(calculate_snr(ref.flatten(), ob[:, h, :D].flatten()))

    print("\n" + "=" * 90)
    print(f"Q/K/V 2048 vs 1280  (1 engine, {1000 / CLK_NS:.2f} MHz, S={S}, per layer = total / {args.reps})")
    print("=" * 90)
    print(f"  {'program':<36}{'ms/layer':>10}{'inst/layer':>12}{'prog KB':>10}")
    for k, v in res.items():
        print(f"  {k:<36}{v['cycles'] * CLK_NS / 1e6 / args.reps:>10.3f}"
              f"{v['inst'] / args.reps:>12.0f}{v['prog_bytes'] / 1024:>10.1f}")
    # "Exact" tolerates the hardware's known 0*w mantissa leak (~1e-37) in zero lanes.
    print("\n  exactness (max |diff|; 0, or < 1e-30 = the known 0*w denormal leak):")
    for k, v in checks.items():
        print(f"    {k:<44} {v:g}")
    print(f"\n  1280 path vs CPU reference, per-head SNR: min {min(snrs):.1f}  max {max(snrs):.1f} dB")
    ok = all(v < 1e-30 for v in checks.values()) and min(snrs) >= 40
    print(f"\n  {'PASS' if ok else 'FAIL'}: exact vs the 2048 path and >= 40 dB vs CPU on every head")


if __name__ == "__main__":
    main()
