#!/usr/bin/env python3
"""Vision attention with a split head_dim: Q.K^T at 128, P.V at 80 or 72.

One engine, one SigLIP layer's attention (16 heads, S = 256 query rows each, the
head-sharded shape the model runs), three ways:

  stock128  CURRENT MODEL: stock unified_attention_core_dynamic at head_dim 128,
            scatter each head's (S, 128) result into (S, 2048), then the dma80
            gather (160 B chunks) into (S, 1280) -- what VIS_UNPAD=dma80 does now.
  split80   pi05_attn_kernels.attention_split_hd, v_dim 80: P.V writes each
            head's (S, 80) straight into its columns of (S, 1280) via the output
            row stride. No scatter, no gather. Rows are 160 B = 5 whole beats.
  split72   same, v_dim 72, straight into (S, 1152) -- the exact layout. Rows are
            144 B = 4.5 beats: the case that scrambled on the READ side; whether
            the matmul's per-row WRITE survives it is what this measures.

Every variant is checked per head against a CPU attention reference (SNR >= 40 dB)
and its (S, 16*72) real lanes are compared across variants.

Run from the repo root:
    python models/pi05/utility/pi05_attn_hd_bench.py [--reps 3]
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
from user_dma_core import UnifiedEngine, calculate_snr
from nn_lib import store_identity_matrix
from pi05_attn_kernels import attention_split_hd

S, NH, D, DP = 256, 16, 72, 128
HP = NH * DP
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=3, help="repeats of the 16-head layer per program")
    args = ap.parse_args()

    # HW_INFO first: it sets the AXI beat width the strided-DMA asserts check, and the
    # real clock (366.67 MHz on this board).
    user_dma_core.set_dma_device("xdma0")
    user_dma_core.configure_clock_from_hardware()
    ue = UnifiedEngine(clock_period_ns=CLK_NS)
    ident = store_identity_matrix(ue)

    torch.manual_seed(0)
    q = (torch.randn(S, NH, D) * 0.5).to(torch.bfloat16)
    k = (torch.randn(S, NH, D) * 0.5).to(torch.bfloat16)
    v = torch.randn(S, NH, D).to(torch.bfloat16)

    def pad(t):                                   # (S, NH, 72) -> (S, NH*128), zero pad lanes
        p = torch.zeros(S, NH, DP, dtype=torch.bfloat16)
        p[:, :, :D] = t
        return p.reshape(S, HP).contiguous()

    buf = {n: ue.allocate_tensor_dram(S * HP * bpe) for n in ("Q", "K", "V", "RES128")}
    for n, t in (("Q", q), ("K", k), ("V", v)):
        ue.dma_to_accelerator_memory(buf[n], pad(t))
    for n in ("FQ", "FK", "FV", "FO"):
        buf[n] = ue.allocate_tensor_dram(S * DP * bpe)
    buf["SCR"] = ue.allocate_tensor_dram((DP * S + S * S + S * DP) * bpe)
    buf["BIAS"] = ue.allocate_tensor_dram(S * S * bpe)
    ue.dma_to_accelerator_memory(buf["BIAS"], torch.zeros(S, S, dtype=torch.bfloat16))
    buf["U80"] = ue.allocate_tensor_dram(S * NH * 80 * bpe)
    buf["U72"] = ue.allocate_tensor_dram(S * NH * 72 * bpe)
    for n, w in (("U80", 80), ("U72", 72), ("RES128", DP)):   # poison: stale data must not pass
        ue.dma_to_accelerator_memory(buf[n], torch.full((S, NH * w), 7.0, dtype=torch.bfloat16))

    col = DP * bpe                 # one head's column block in the (S, 2048) buffers
    row = HP * bpe

    def gather(h):                 # the model's per-head Q/K/V marshalling, unchanged
        for src, dst in ((buf["Q"], buf["FQ"]), (buf["K"], buf["FK"]), (buf["V"], buf["FV"])):
            ue.accelerator_memory_to_sram(src + h * col, 0x00000, S * DP,
                                          stride_bytes_per_chunk=col, stride_jump_bytes=row)
            ue.sram_to_accelerator_memory(0x00000, dst, S * DP)

    def e_stock():
        for _ in range(args.reps):
            for h in range(NH):
                gather(h)
                ue.unified_attention_core_dynamic(
                    batch=S, aligned_seq_len=S, head_dim=DP,
                    Q_DRAM_ADDR=buf["FQ"], K_DRAM_ADDR=buf["FK"], V_DRAM_ADDR=buf["FV"],
                    BIAS_DRAM_ADDR=buf["BIAS"], OUTPUT_DRAM_ADDR=buf["FO"],
                    SCRATCH_DRAM_ADDR=buf["SCR"], IDENTITY_DRAM_ADDR=ident)
                ue.accelerator_memory_to_sram(buf["FO"], 0x00000, S * DP)
                ue.sram_to_accelerator_memory(0x00000, buf["RES128"] + h * col, S * DP,
                                              stride_bytes_per_chunk=col, stride_jump_bytes=row)
            for r0 in range(0, S, 32):   # dma80 gather, as VIS_UNPAD=dma80 emits it
                ue.accelerator_memory_to_sram(buf["RES128"] + r0 * row, 0x00000, 32 * NH * 80,
                                              stride_bytes_per_chunk=160, stride_jump_bytes=col)
                ue.sram_to_accelerator_memory(0x00000, buf["U80"] + r0 * NH * 80 * bpe, 32 * NH * 80)

    def e_split(v_dim, out):
        def emit():
            for _ in range(args.reps):
                for h in range(NH):
                    gather(h)
                    attention_split_hd(ue, S, S, DP, v_dim,
                                       buf["FQ"], buf["FK"], buf["FV"], buf["BIAS"],
                                       buf[out] + h * v_dim * bpe, buf["SCR"], ident,
                                       out_row_stride=NH * v_dim)
        return emit

    # CPU reference: the model's effective scale is 1/sqrt(128) on the padded Q.
    ref = torch.empty(S, NH, D)
    for h in range(NH):
        sc = (q[:, h].float() @ k[:, h].float().T) / math.sqrt(DP)
        ref[:, h] = torch.softmax(sc, -1) @ v[:, h].float()

    res, outs, errs = {}, {}, {}
    for label, emit, name, w in (("stock128 + dma80 gather", e_stock, "U80", 80),
                                 ("split80 (row stride)", e_split(80, "U80"), "U80", 80),
                                 ("split72 (row stride)", e_split(72, "U72"), "U72", 72)):
        ue.dma_to_accelerator_memory(buf[name], torch.full((S, NH * w), 7.0, dtype=torch.bfloat16))
        try:
            run(ue, label, emit, res)
        except (AssertionError, ValueError) as ex:
            print(f"  [{label}] refused in software: {ex}")
            continue
        got = ue.dma_from_accelerator_memory(buf[name], (S, NH * w)).float().reshape(S, NH, w)
        outs[label] = got[:, :, :D]
        per_head = [calculate_snr(ref[:, h].flatten(), got[:, h, :D].flatten()) for h in range(NH)]
        pad_max = got[:, :, D:].abs().max().item() if w > D else 0.0
        errs[label] = (min(per_head), max(per_head), pad_max, per_head)

    print("\n" + "=" * 92)
    print(f"SPLIT-HEAD-DIM ATTENTION  (1 engine, {1000 / CLK_NS:.2f} MHz, 16 heads x S={S}, "
          f"per layer = total / {args.reps})")
    print("=" * 92)
    print(f"  {'variant':<26}{'ms/layer':>10}{'us/head':>10}{'inst/layer':>12}"
          f"{'SNR min':>9}{'SNR max':>9}{'|pad| max':>11}")
    for label, v_ in res.items():
        ms = v_["cycles"] * CLK_NS / 1e6 / args.reps
        lo, hi, pm, _ = errs.get(label, (float("nan"),) * 3 + ([],))
        print(f"  {label:<26}{ms:>10.3f}{ms * 1e3 / NH:>10.1f}{v_['inst'] / args.reps:>12.0f}"
              f"{lo:>9.1f}{hi:>9.1f}{pm:>11.3g}")
    for label, (lo, hi, pm, ph) in errs.items():
        if lo < 40:
            print(f"  [{label}] per-head SNR: " + " ".join(f"{x:.0f}" for x in ph))
    base = outs.get("stock128 + dma80 gather")
    if base is not None:
        for label, o in outs.items():
            if o is base:
                continue
            d = (o - base).abs().max().item()
            print(f"  {label} vs stock128, real lanes: "
                  + ("bit-identical" if d == 0 else f"max |diff| {d:.4g}"))
    for label, (lo, _, pm, _) in errs.items():
        ok = lo >= 40 and pm < 1e-3
        print(f"  {'PASS' if ok else 'FAIL'}  {label}  (SNR >= 40 dB on every head, pad lanes ~0)")


if __name__ == "__main__":
    main()
