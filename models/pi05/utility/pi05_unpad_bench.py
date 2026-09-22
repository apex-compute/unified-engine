#!/usr/bin/env python3
"""Vision attention head-dim UNPAD micro-benchmark (SigLIP: 16 heads, 128 -> 72).

Before touching pi05_test.py, measure on ONE engine, at the per-engine row counts
the model actually runs, every way of getting the padded attention output
(rows, 16*128) down to the real (rows, 16*72) the O projection wants:

  padO    OLD: no unpad; O proj at K=2048 with zero rows in each head's pad lanes
  sel     CURRENT: one selection matmul 2048 -> 1152, then O proj K=1152
  sel8    two block-diagonal selection matmuls (8 heads each, 1024 -> 576) from two
          (rows, 1024) group buffers, then O proj as two chained K=576 matmuls
  dma144  pure DMA: strided DRAM->SRAM read with 144 B chunks (72 lanes) jumping
          256 B (128 lanes), then one contiguous SRAM->DRAM write. 144 B is 4.5 AXI
          beats; the READ side has no software alignment assert, and whether the
          hardware packs such chunks tightly is exactly what this probes.

The pad lanes of the input are filled with a SENTINEL (not zero), so any variant
that leaks a pad lane into the output shows up as a hard mismatch.

Each timed program repeats its op body --reps times (default 27 = one vision
slot's layers) so the latency counter sees a stable number.

Run from the repo root:
    python models/pi05/utility/pi05_unpad_bench.py                 # rows 24 32 256
    python models/pi05/utility/pi05_unpad_bench.py --rows 32 --reps 27
"""
import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO)

import numpy as np
import torch

import user_dma_core
from user_dma_core import UnifiedEngine, TYPE, calculate_snr
from nn_lib import store_weight, store_quantized_weight
from quant_lib import quantize_q4_64

NH, D, DP, H = 16, 72, 128, 1152
HP, HR = NH * DP, NH * D           # 2048 padded, 1152 real
G = 8                              # heads per unpad group: 8*72 = 576 = 9*64
GP, GR = G * DP, G * D             # 1024, 576
CLK_NS = 1000.0 / 366.67
SENTINEL = 5.0
DMA_ROW_BLOCK = 32                 # rows per SRAM round trip in dma144 (32*2304 B = 72 KB)
L80 = 80                           # dma160: 160 B chunks = 5 whole AXI beats = 72 real + 8 pad lanes
HL = NH * L80                      # 1280 = 20*64: O proj K with 8 zero rows per head


# ---------------------------------------------------------------- host refs ----
def q4_dequant(w_nk):
    """Exact host dequant of quantize_q4_64 (blocks of 64 along the flattened
    (N, K) row-major weight, fp32 scale for rounding, bf16 scale on device)."""
    blocks = w_nk.float().reshape(-1, 64)
    s = blocks.abs().amax(1)
    s[s == 0] = 1.0
    s = s / 7.0
    q = torch.clamp(torch.round(blocks / s[:, None]), -8, 7)
    return (q * s.to(torch.bfloat16).float()[:, None]).reshape(w_nk.shape)


def store_q4(ue, w_nk):
    packed, _ = quantize_q4_64(w_nk.to(torch.bfloat16).contiguous())
    return store_quantized_weight(ue, packed)          # (scale_addr, data_addr)


def sel_matrix(n_heads):
    """(n_heads*D, n_heads*DP) bf16, 1 at [h*D+d, h*DP+d]: A @ sel^T drops pad lanes."""
    s = torch.zeros(n_heads * D, n_heads * DP, dtype=torch.bfloat16)
    for h in range(n_heads):
        for d in range(D):
            s[h * D + d, h * DP + d] = 1.0
    return s


def snr(ref, got):
    return calculate_snr(ref.float().flatten(), got.float().flatten())


# ------------------------------------------------------------- program glue ----
def run(ue, label, emit, results):
    """Capture emit(), halt, execute once, record latency. Same capture/halt/write
    sequence as user_hw_test's single-engine tests."""
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
    ue.wait_queue(60.0)
    cycles, n_inst = ue.report_timing_and_instruction_count()
    results[label] = {"cycles": cycles, "inst": n_inst, "prog_bytes": nbytes}
    ue.clear_capture_buffer()


def mreg(ue, rows):
    r = ue.alloc_isa_reg()
    ue.generate_instruction_add_set(r, rows)
    return r


def bench_rows(ue, rows, reps, W, bufs):
    res = {}
    torch.manual_seed(rows)
    x = torch.randn(rows, NH, D, dtype=torch.bfloat16)
    x_pad = torch.full((rows, NH, DP), SENTINEL, dtype=torch.bfloat16)
    x_pad[:, :, :D] = x
    x_unpad = x.reshape(rows, HR)
    ue.dma_to_accelerator_memory(bufs["X"], x_pad.reshape(rows, HP).contiguous())
    # sel8's two group buffers, laid out exactly as the model's attention scatter
    # would write them (row stride GP instead of HP).
    for g in range(2):
        ue.dma_to_accelerator_memory(bufs["Xg"][g],
                                     x_pad[:, g * G:(g + 1) * G, :].reshape(rows, GP).contiguous())

    # ---- unpad-only programs -------------------------------------------------
    def e_sel():
        r = mreg(ue, rows)
        for _ in range(reps):
            ue.matmat_mul_core(M=rows, K=HP, N=HR, A_DRAM_ADDR=bufs["X"], B_DRAM_ADDR=W["sel"],
                               OUTPUT_DRAM_ADDR=bufs["U1"], is_B_quantized=False, gpr_M_reg=r)
        ue.release_isa_reg()

    def e_sel8():
        r = mreg(ue, rows)
        for _ in range(reps):
            for g in range(2):
                ue.matmat_mul_core(M=rows, K=GP, N=GR, A_DRAM_ADDR=bufs["Xg"][g],
                                   B_DRAM_ADDR=W["sel8"], OUTPUT_DRAM_ADDR=bufs["U2"][g],
                                   is_B_quantized=False, gpr_M_reg=r)
        ue.release_isa_reg()

    def e_dma():
        for _ in range(reps):
            for r0 in range(0, rows, DMA_ROW_BLOCK):
                rb = min(DMA_ROW_BLOCK, rows - r0)
                ue.accelerator_memory_to_sram(bufs["X"] + r0 * HP * 2, 0x00000, rb * HR,
                                              stride_bytes_per_chunk=D * 2,
                                              stride_jump_bytes=DP * 2)
                ue.sram_to_accelerator_memory(0x00000, bufs["U3"] + r0 * HR * 2, rb * HR)

    def e_dma160():
        for _ in range(reps):
            for r0 in range(0, rows, DMA_ROW_BLOCK):
                rb = min(DMA_ROW_BLOCK, rows - r0)
                ue.accelerator_memory_to_sram(bufs["X"] + r0 * HP * 2, 0x00000, rb * HL,
                                              stride_bytes_per_chunk=L80 * 2,
                                              stride_jump_bytes=DP * 2)
                ue.sram_to_accelerator_memory(0x00000, bufs["U4"] + r0 * HL * 2, rb * HL)

    run(ue, "sel (unpad only)", e_sel, res)
    run(ue, "sel8 (unpad only)", e_sel8, res)
    try:
        run(ue, "dma144 (unpad only)", e_dma, res)
        dma_ok = True
    except AssertionError as ex:
        print(f"  [dma144] refused in software: {ex}")
        dma_ok = False

    run(ue, "dma160 (unpad only)", e_dma160, res)
    u1 = ue.dma_from_accelerator_memory(bufs["U1"], (rows, HR))
    u2 = torch.cat([ue.dma_from_accelerator_memory(bufs["U2"][g], (rows, GR)) for g in range(2)], 1)
    checks = {
        "sel": (u1 - x_unpad).abs().max().item(),
        "sel8": (u2 - x_unpad).abs().max().item(),
    }
    u4 = ue.dma_from_accelerator_memory(bufs["U4"], (rows, HL))
    checks["dma160 (vs x[:, :80])"] = (u4 - x_pad[:, :, :L80].reshape(rows, HL)).abs().max().item()
    if dma_ok:
        u3 = ue.dma_from_accelerator_memory(bufs["U3"], (rows, HR))
        checks["dma144"] = (u3 - x_unpad).abs().max().item()
        if checks["dma144"] != 0:
            diagnose_dma(x_pad, u3, rows)

    # ---- unpad + O projection (what one layer actually pays) -----------------
    b = W["o_bias_t"]
    def o_full(A, OUT, r):
        ue.matmat_mul_core(M=rows, K=HR, N=H, A_DRAM_ADDR=A, B_DRAM_ADDR=W["o"][1],
                           SCALE_DRAM_ADDR=W["o"][0], C_DRAM_ADDR=W["o_bias"],
                           bias_mode="broadcast_N", OUTPUT_DRAM_ADDR=OUT,
                           is_B_quantized=True, data_type=TYPE.IF4, gpr_M_reg=r)

    def e_padO():
        r = mreg(ue, rows)
        for _ in range(reps):
            ue.matmat_mul_core(M=rows, K=HP, N=H, A_DRAM_ADDR=bufs["X"], B_DRAM_ADDR=W["o_pad"][1],
                               SCALE_DRAM_ADDR=W["o_pad"][0], C_DRAM_ADDR=W["o_bias"],
                               bias_mode="broadcast_N", OUTPUT_DRAM_ADDR=bufs["O"][0],
                               is_B_quantized=True, data_type=TYPE.IF4, gpr_M_reg=r)
        ue.release_isa_reg()

    def e_selO():
        r = mreg(ue, rows)
        for _ in range(reps):
            ue.matmat_mul_core(M=rows, K=HP, N=HR, A_DRAM_ADDR=bufs["X"], B_DRAM_ADDR=W["sel"],
                               OUTPUT_DRAM_ADDR=bufs["U1"], is_B_quantized=False, gpr_M_reg=r)
            o_full(bufs["U1"], bufs["O"][1], r)
        ue.release_isa_reg()

    def e_sel8O():
        r = mreg(ue, rows)
        for _ in range(reps):
            for g in range(2):
                ue.matmat_mul_core(M=rows, K=GP, N=GR, A_DRAM_ADDR=bufs["Xg"][g],
                                   B_DRAM_ADDR=W["sel8"], OUTPUT_DRAM_ADDR=bufs["U2"][g],
                                   is_B_quantized=False, gpr_M_reg=r)
            # O = U2[0] @ Wo[:, :576]^T + b ; O += U2[1] @ Wo[:, 576:]^T
            ue.matmat_mul_core(M=rows, K=GR, N=H, A_DRAM_ADDR=bufs["U2"][0],
                               B_DRAM_ADDR=W["o_k"][0][1], SCALE_DRAM_ADDR=W["o_k"][0][0],
                               C_DRAM_ADDR=W["o_bias"], bias_mode="broadcast_N",
                               OUTPUT_DRAM_ADDR=bufs["O"][2], is_B_quantized=True,
                               data_type=TYPE.IF4, gpr_M_reg=r)
            ue.matmat_mul_core(M=rows, K=GR, N=H, A_DRAM_ADDR=bufs["U2"][1],
                               B_DRAM_ADDR=W["o_k"][1][1], SCALE_DRAM_ADDR=W["o_k"][1][0],
                               C_DRAM_ADDR=bufs["O"][2], bias_mode="full_matrix",
                               OUTPUT_DRAM_ADDR=bufs["O"][2], is_B_quantized=True,
                               data_type=TYPE.IF4, gpr_M_reg=r)
        ue.release_isa_reg()

    def e_dmaO():
        r = mreg(ue, rows)
        for _ in range(reps):
            for r0 in range(0, rows, DMA_ROW_BLOCK):
                rb = min(DMA_ROW_BLOCK, rows - r0)
                ue.accelerator_memory_to_sram(bufs["X"] + r0 * HP * 2, 0x00000, rb * HR,
                                              stride_bytes_per_chunk=D * 2,
                                              stride_jump_bytes=DP * 2)
                ue.sram_to_accelerator_memory(0x00000, bufs["U3"] + r0 * HR * 2, rb * HR)
            o_full(bufs["U3"], bufs["O"][3], r)
        ue.release_isa_reg()

    def e_dma160O():
        r = mreg(ue, rows)
        for _ in range(reps):
            for r0 in range(0, rows, DMA_ROW_BLOCK):
                rb = min(DMA_ROW_BLOCK, rows - r0)
                ue.accelerator_memory_to_sram(bufs["X"] + r0 * HP * 2, 0x00000, rb * HL,
                                              stride_bytes_per_chunk=L80 * 2,
                                              stride_jump_bytes=DP * 2)
                ue.sram_to_accelerator_memory(0x00000, bufs["U4"] + r0 * HL * 2, rb * HL)
            ue.matmat_mul_core(M=rows, K=HL, N=H, A_DRAM_ADDR=bufs["U4"], B_DRAM_ADDR=W["o80"][1],
                               SCALE_DRAM_ADDR=W["o80"][0], C_DRAM_ADDR=W["o_bias"],
                               bias_mode="broadcast_N", OUTPUT_DRAM_ADDR=bufs["O"][4],
                               is_B_quantized=True, data_type=TYPE.IF4, gpr_M_reg=r)
        ue.release_isa_reg()

    run(ue, "padO (old)", e_padO, res)
    run(ue, "sel + O (current)", e_selO, res)
    run(ue, "sel8 + O", e_sel8O, res)
    run(ue, "dma160 + O(K=1280)", e_dma160O, res)
    if dma_ok:
        run(ue, "dma144 + O", e_dmaO, res)

    # O-proj references: exact IF4 dequant of the SAME blobs each variant reads.
    ref_real = x_unpad.float() @ W["o_deq"].T + b
    ref_pad = x_pad.reshape(rows, HP).float() @ W["o_pad_deq"].T + b
    ref_k = (x_unpad[:, :GR].float() @ W["o_k_deq"][0].T + b) + x_unpad[:, GR:].float() @ W["o_k_deq"][1].T
    o_snr = {
        "padO (old)": snr(ref_pad, ue.dma_from_accelerator_memory(bufs["O"][0], (rows, H))),
        "sel + O (current)": snr(ref_real, ue.dma_from_accelerator_memory(bufs["O"][1], (rows, H))),
        "sel8 + O": snr(ref_k, ue.dma_from_accelerator_memory(bufs["O"][2], (rows, H))),
    }
    ref_80 = x_pad[:, :, :L80].reshape(rows, HL).float() @ W["o80_deq"].T + b
    o_snr["dma160 + O(K=1280)"] = snr(ref_80, ue.dma_from_accelerator_memory(bufs["O"][4], (rows, H)))
    o_snr["dma160 vs real-72 ref"] = None
    o_snr_real80 = snr(ref_real, ue.dma_from_accelerator_memory(bufs["O"][4], (rows, H)))
    print(f"  [dma160] O vs the unpadded-72 reference (different IF4 blocking): {o_snr_real80:.1f} dB")
    if dma_ok:
        o_snr["dma144 + O"] = snr(ref_real, ue.dma_from_accelerator_memory(bufs["O"][3], (rows, H)))
    return res, checks, o_snr


def diagnose_dma(x_pad, got, rows):
    """dma144 came back wrong: say which layout the hardware actually produced."""
    xp = x_pad.reshape(rows * NH, DP)                    # one row per 128-lane chunk
    flat = got.reshape(-1).float()
    n = flat.numel()
    cands = {
        "chunk rounded up to 160 B (5 beats), packed": xp[:, :80].reshape(-1)[:n],
        "chunk rounded down to 128 B (4 beats), packed": xp[:, :64].reshape(-1)[:n],
        "each chunk starts a fresh 128 B SRAM row (72 real + gap)":
            torch.cat([xp[:, :72], torch.zeros(rows * NH, 56, dtype=xp.dtype)], 1).reshape(-1)[:n],
        "stride ignored (plain contiguous read)": x_pad.reshape(-1)[:n],
    }
    print("  [dma144] MISMATCH -- closest layouts (SNR vs readback, higher = better match):")
    for k, v in cands.items():
        m = min(v.numel(), n)
        print(f"      {snr(v[:m], flat[:m]):8.2f} dB   {k}")
    # Raw dump for offline mapping of every readback element to its source lane.
    dump = os.environ.get("UNPAD_DUMP")
    if dump:
        np.savez(dump, x_pad=x_pad.float().numpy(), got=got.float().numpy())
        print(f"      raw readback saved to {dump}")
    print(f"      first 8 got : {flat[:8].tolist()}")
    print(f"      first 8 want: {xp[0, :8].float().tolist()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, nargs="+", default=[24, 32, 256],
                    help="per-engine row counts (24 = 12-engine shard, 32 = 8-engine, 256 = 1 engine)")
    ap.add_argument("--reps", type=int, default=27, help="op-body repeats per program (27 = layers/slot)")
    args = ap.parse_args()
    for r in args.rows:
        assert r % 4 == 0 and r <= 256, f"rows={r}"

    # HW_INFO first: it sets the AXI beat width the strided-DMA asserts check, and the
    # real clock (366.67 MHz on this board).
    user_dma_core.set_dma_device("xdma0")
    user_dma_core.configure_clock_from_hardware()
    ue = UnifiedEngine(clock_period_ns=CLK_NS)           # build BEFORE any weight write

    torch.manual_seed(0)
    Wo = (torch.randn(H, HR) / HR ** 0.5).to(torch.bfloat16)      # (N=H, K=16*72)
    Wo_pad = torch.zeros(H, NH, DP, dtype=torch.bfloat16)
    Wo_pad[:, :, :D] = Wo.reshape(H, NH, D)
    Wo_pad = Wo_pad.reshape(H, HP)
    bias = (torch.randn(H) * 0.1).to(torch.bfloat16)
    Wo80 = torch.zeros(H, NH, L80, dtype=torch.bfloat16)
    Wo80[:, :, :D] = Wo.reshape(H, NH, D)
    Wo80 = Wo80.reshape(H, HL)
    W = {
        "sel": store_weight(ue, sel_matrix(NH)),
        "sel8": store_weight(ue, sel_matrix(G)),
        "o": store_q4(ue, Wo),
        "o_pad": store_q4(ue, Wo_pad),
        "o80": store_q4(ue, Wo80),
        "o80_deq": q4_dequant(Wo80),
        "o_k": [store_q4(ue, Wo[:, :GR].contiguous()), store_q4(ue, Wo[:, GR:].contiguous())],
        "o_bias": store_weight(ue, bias),
        "o_bias_t": bias.float(),
        "o_deq": q4_dequant(Wo),
        "o_pad_deq": q4_dequant(Wo_pad),
        "o_k_deq": [q4_dequant(Wo[:, :GR].contiguous()), q4_dequant(Wo[:, GR:].contiguous())],
    }
    R = max(args.rows)
    bufs = {
        "X": ue.allocate_tensor_dram(R * HP * 2),
        "Xg": [ue.allocate_tensor_dram(R * GP * 2) for _ in range(2)],
        "U1": ue.allocate_tensor_dram(R * HR * 2),
        "U2": [ue.allocate_tensor_dram(R * GR * 2) for _ in range(2)],
        "U3": ue.allocate_tensor_dram(R * HR * 2),
        "U4": ue.allocate_tensor_dram(R * HL * 2),
        "O": [ue.allocate_tensor_dram(R * H * 2) for _ in range(5)],
    }

    summary = []
    for rows in args.rows:
        print(f"\n===== rows={rows}  reps={args.reps} =====")
        res, checks, o_snr = bench_rows(ue, rows, args.reps, W, bufs)
        summary.append((rows, res, checks, o_snr))

    print("\n" + "=" * 86)
    print(f"UNPAD BENCH  (1 engine, {1000 / CLK_NS:.2f} MHz, times are per layer = total / {args.reps})")
    print("=" * 86)
    for rows, res, checks, o_snr in summary:
        print(f"\nrows = {rows}")
        print("  unpad exactness (max |got - x[:, :72]|, must be 0):  "
              + "   ".join(f"{k} {v:g}" for k, v in checks.items()))
        print(f"  {'variant':<22}{'us/layer':>10}{'inst/layer':>12}{'prog KB':>10}{'O SNR dB':>10}")
        for k, v in res.items():
            us = v["cycles"] * CLK_NS / 1e3 / args.reps
            s = o_snr.get(k)
            print(f"  {k:<22}{us:>10.2f}{v['inst'] / args.reps:>12.0f}{v['prog_bytes'] / 1024:>10.1f}"
                  f"{('' if s is None else f'{s:10.1f}')}")
    print("\nPASS criteria: unpad exactness 0 for every variant that ran; O SNR >= 40 dB.")


if __name__ == "__main__":
    main()
