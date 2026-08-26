#!/usr/bin/env python3
"""Causal proof of what actually bounds pi0.5's stages on this device.

A roofline plot is a MODEL. It says the prefix stage sits at 201 FLOP/byte
against a ridge point of 24, therefore it is compute-bound -- but that is an
inference from two ceilings, not a measurement of the thing itself. A stage can
sit anywhere on a roofline chart and still be limited by something the chart
does not draw: DMA latency, barrier waits, PBI loop overhead, idle ALUs.

So this script does not plot anything. It INTERVENES on one axis at a time and
watches which axis the runtime follows:

  starve   take memory bandwidth away and see if the runtime notices.
           A hog engine on the SAME memory port streams DRAM flat out while the
           victim runs. A hog on the OTHER port is the control: it must show no
           effect, which is what distinguishes "bandwidth was removed" from
           "something else got slower when two engines ran at once".

  bytes    double the bytes, hold the FLOPs EXACTLY constant. IF4 costs
           0.5 + 2/64 = 0.531 B/weight, IF8 costs 1.0 + 2/64 = 1.031 B/weight --
           a 1.94x byte increase for a bit-identical FLOP count. Runtime flat =>
           compute-bound. Runtime x1.94 => memory-bound. Swept over M, because M
           is what moves arithmetic intensity: an IF4 matmul does 2*M FLOPs per
           weight element and spends 0.531 bytes on it, so AI = 3.76*M FLOP/byte
           and the ridge point (460.8 GFLOP/s / 19.2 GB/s = 24) is crossed at
           M ~= 6.4 rows. That crossover is a falsifiable prediction with a
           number attached, which is the point.

  ports    same total work, engines packed onto one memory port vs spread over
           both. Memory-bound => spreading is worth up to 2x. Compute-bound =>
           no change. (Measured topology, from pi05_bw_scaling.py: engines 0-7
           share one ~9.6 GB/s port, engines 8-11 share a second, and ONE engine
           already saturates a port.)

Weight VALUES are never quantized here, only the byte counts are honoured. The
MAC array is fixed-latency, so what the nibbles decode to cannot change a cycle
count -- and quantize_weight's IF4 packing is a 2M-iteration Python loop per
matrix, which would dominate the runtime of the experiment rather than the
device. Scales are written as sane positive bf16 so nothing downstream sees a
NaN; the data bytes are random.

    python models/pi05/utility/pi05_bound_proof.py bytes
    python models/pi05/utility/pi05_bound_proof.py starve
    python models/pi05/utility/pi05_bound_proof.py ports
    python models/pi05/utility/pi05_bound_proof.py all
"""
import argparse
import contextlib
import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

import torch
import user_dma_core
from user_dma_core import (UnifiedEngine, TYPE, URAM_NEAR_FULL_ELEMENTS,
                           UE_VECTOR_SIZE, UE_PIPELINE_COUNTER_CLK_DIV,
                           UE_LATENCY_COUNT_ADDR, DMA_DEVICE_H2C)

ENGINE_REG_STRIDE = 0x00010000
ENGINE_DRAM_STRIDE = 0x04000000      # 64 MB private window per engine

# Ceilings measured on this board. MACS_PER_CYCLE*2/CYCLE_NS per engine, and
# pi05_bw_scaling.py's port sweep for the memory side.
PEAK_GFLOPS_PER_ENGINE = 64 * 2 / 3.3333333
PORT_GBS = 9.6
PORT_OF_ENGINE = lambda i: 0 if i < 8 else 1

# Bytes of DRAM per weight element, scales included (q4_64: one bf16 scale per
# 64-element block along K). The whole 'bytes' intervention rests on this ratio.
BYTES_PER_WEIGHT = {TYPE.IF4: 0.5 + 2.0 / 64, TYPE.IF8: 1.0 + 2.0 / 64}


# ------------------------------------------------------------- plumbing ----
def _quiet(fn, *a, **kw):
    """UnifiedEngine narrates every DMA and register poke. Swallow it."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*a, **kw)


def make_engine(idx, region=None):
    region = idx if region is None else region
    base = user_dma_core.DRAM_START_ADDR + region * ENGINE_DRAM_STRIDE
    return _quiet(UnifiedEngine,
                  BASE_ADDR=user_dma_core.UE_0_BASE_ADDR + idx * ENGINE_REG_STRIDE,
                  params_dram_base=base,
                  tensor_dram_base=base + 0x01000000,
                  program_dram_base=base + 0x03000000)


def assert_card_alive(engines):
    """A wedged card reads all-ones on the AXI-Lite BAR. Recovering needs a PCIe
    hot-remove + rescan, i.e. root -- so catch it here and stop rather than
    spend the rest of the run writing into a hole."""
    for i, ue in enumerate(engines):
        v = ue.read_reg32(0)
        if v == 0xFFFFFFFF:
            raise SystemExit(
                f"CARD WEDGED: engine {i} hw register reads 0xFFFFFFFF. "
                f"Stop and run `bash rescan_xilinx.sh` (needs sudo).")


def _alloc_params(ue, nbytes):
    addr = ue.get_params_dram_addr()
    ue.allocate_params_dram(nbytes)
    return addr


def latency_us(ue):
    cycles = ue.read_reg32(UE_LATENCY_COUNT_ADDR) * UE_PIPELINE_COUNTER_CLK_DIV
    return cycles * ue._clock_period_ns / 1e3


def _finish_program(ue):
    ue.generate_instruction_halt()
    addr = ue.get_program_dram_addr()
    _quiet(ue.write_captured_instructions_to_dram, addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())
    ue.clear_capture_buffer()
    return addr


# ---------------------------------------------------------- the workload ----
def build_matmul(ue, M, K, N, data_type, repeats):
    """A quantized matmul repeated `repeats` times, on the DYNAMIC (PBI) path.

    Dynamic, not legacy, for the same reason pi05 uses it everywhere: the legacy
    path unrolls the M/N tiling in Python, so M=832,N=2048 would compile to a
    program measured in hundreds of MB. The dynamic path keeps each call to a
    fixed handful of instructions with the trip counts in GPRs.

    Weight buffer is sized, not quantized -- see the module docstring."""
    nk = N * K
    b_bytes = nk // 2 if data_type == TYPE.IF4 else nk
    n_blocks = nk // UE_VECTOR_SIZE

    B = _alloc_params(ue, b_bytes)
    S = _alloc_params(ue, n_blocks * 2)
    A = ue.allocate_tensor_dram(M * K * 2)
    OUT = ue.allocate_tensor_dram(M * N * 2)

    _quiet(ue.dma_write, DMA_DEVICE_H2C, B,
           torch.randint(0, 255, (b_bytes,), dtype=torch.uint8), b_bytes)
    # Positive, order-1 scales: nothing here should produce a NaN that some
    # downstream op might treat specially.
    _quiet(ue.dma_write, DMA_DEVICE_H2C, S,
           torch.full((n_blocks,), 1.0, dtype=torch.bfloat16).view(torch.uint16),
           n_blocks * 2)
    _quiet(ue.dma_write, DMA_DEVICE_H2C, A,
           torch.randn(M * K, dtype=torch.bfloat16), M * K * 2)

    m_reg, k_reg, n_reg = ue.alloc_isa_reg(), ue.alloc_isa_reg(), ue.alloc_isa_reg()
    ue.start_capture()
    for _ in range(repeats):
        # Re-seed every iteration: the kernel treats gpr_M_reg as a read-only
        # alias but decrements its own counters off it, and re-seeding costs
        # three instructions against a matmul, so there is no reason to gamble.
        ue.generate_instruction_add_set(m_reg, M)
        ue.generate_instruction_add_set(k_reg, K)
        ue.generate_instruction_add_set(n_reg, N)
        _quiet(ue.matmat_mul_core,
               M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=B, OUTPUT_DRAM_ADDR=OUT,
               is_B_quantized=True, data_type=data_type, SCALE_DRAM_ADDR=S,
               gpr_M_reg=m_reg, gpr_K_reg=k_reg, gpr_N_reg=n_reg)
    ue.stop_capture()
    ue.release_isa_reg(); ue.release_isa_reg(); ue.release_isa_reg()
    return _finish_program(ue)


def build_hog(ue, seconds_target):
    """A program that does nothing but pull DRAM as fast as the port allows.

    Sized to outlast the victim: if the hog halts early the victim finishes its
    tail unstarved and the effect washes out. One engine already saturates a
    port (measured), so a single hog is the whole intervention."""
    chunk_elems = URAM_NEAR_FULL_ELEMENTS
    chunk_bytes = chunk_elems * 2
    n_chunks = 32
    buf = ue.allocate_tensor_dram(chunk_bytes * n_chunks)
    # ~9.6 GB/s per port -> bytes needed for the target duration.
    passes = max(1, int(seconds_target * PORT_GBS * 1e9 / (chunk_bytes * n_chunks)))
    ue.start_capture()
    for _ in range(passes):
        for c in range(n_chunks):
            ue.accelerator_memory_to_sram(
                accelerator_dram_address=buf + c * chunk_bytes,
                sram_address=0x00000, element_size=chunk_elems)
    ue.stop_capture()
    return _finish_program(ue), passes * n_chunks * chunk_bytes


def run_and_time(ue, prog, timeout):
    ue.start_execute_from_dram(prog)
    ue.wait_queue(timeout)
    return latency_us(ue)


# ------------------------------------------------- intervention 2: bytes ----
def intervention_bytes(shapes, m_list, target_ms):
    print("\n" + "=" * 96)
    print("INTERVENTION 2 -- double the bytes, hold the FLOPs constant (IF4 vs IF8)")
    print("=" * 96)
    print("IF8 moves 1.94x the weight bytes of IF4 for a bit-identical FLOP count.")
    print("compute-bound => t(IF8)/t(IF4) ~ 1.00      memory-bound => ~1.94")
    print(f"predicted crossover at M ~ {PEAK_GFLOPS_PER_ENGINE / PORT_GBS / 2 * BYTES_PER_WEIGHT[TYPE.IF4] * 2:.1f} rows "
          f"(single engine: {PEAK_GFLOPS_PER_ENGINE:.1f} GFLOP/s over {PORT_GBS} GB/s)")

    results = {}
    for (K, N) in shapes:
        print(f"\n  K={K} N={N}")
        print(f"  {'M':>5} {'reps':>5} {'t(IF4) us':>11} {'t(IF8) us':>11} {'ratio':>7} "
              f"{'IF4 GF/s':>9} {'IF4 GB/s':>9} {'AI':>7}  verdict")
        for M in m_list:
            flops_call = 2 * M * K * N
            reps = max(1, min(200, int(target_ms * 1e-3 * PEAK_GFLOPS_PER_ENGINE * 1e9
                                       / max(flops_call, 1))))
            row = {}
            for dt in (TYPE.IF4, TYPE.IF8):
                ue = make_engine(0)
                assert_card_alive([ue])
                prog = build_matmul(ue, M, K, N, dt, reps)
                # Generous but bounded: 8x the memory-bound floor, never unbounded.
                floor = reps * (M * K * 2 + N * K * BYTES_PER_WEIGHT[dt] + M * N * 2) / (PORT_GBS * 1e9)
                row[dt] = run_and_time(ue, prog, max(20.0, floor * 8))
            t4, t8 = row[TYPE.IF4], row[TYPE.IF8]
            ratio = t8 / t4 if t4 else 0.0
            gf = reps * flops_call / (t4 * 1e-6) / 1e9
            b4 = reps * (M * K * 2 + N * K * BYTES_PER_WEIGHT[TYPE.IF4] + M * N * 2)
            gb = b4 / (t4 * 1e-6) / 1e9
            ai = reps * flops_call / b4
            verdict = ("MEMORY" if ratio > 1.5 else
                       "mixed" if ratio > 1.15 else "COMPUTE")
            print(f"  {M:>5} {reps:>5} {t4:>11.1f} {t8:>11.1f} {ratio:>7.2f} "
                  f"{gf:>9.1f} {gb:>9.2f} {ai:>7.1f}  {verdict}")
            results[(K, N, M)] = (t4, t8, ratio, gf, gb, ai)
    return results


# ------------------------------------------------ intervention 1: starve ----
def intervention_starve(cases):
    print("\n" + "=" * 96)
    print("INTERVENTION 1 -- bandwidth starvation")
    print("=" * 96)
    print("Victim runs on engine 0 (port 0). Hog streams DRAM flat out on engine 4")
    print("(SAME port) or engine 8 (OTHER port, the control).")
    print("compute-bound => same-port hog costs ~nothing;  memory-bound => victim slows")
    print("If the CONTROL also slows, the effect is not bandwidth and the test is void.\n")
    print(f"  {'case':>26} {'M':>5} {'solo us':>10} {'+hog p0 us':>11} {'slow':>6} "
          f"{'+hog p1 us':>11} {'slow':>6}  verdict")

    for label, (M, K, N, dt, reps) in cases:
        # Solo first, to size the hog against a measured duration rather than a guess.
        ue = make_engine(0)
        assert_card_alive([ue])
        prog = build_matmul(ue, M, K, N, dt, reps)
        solo = run_and_time(ue, prog, 60.0)

        slowed = {}
        for hog_eng in (4, 8):
            v = make_engine(0)
            hog = make_engine(hog_eng)
            assert_card_alive([v, hog])
            vprog = build_matmul(v, M, K, N, dt, reps)
            # 3x the victim's measured solo time, so the hog cannot halt early.
            hprog, _ = build_hog(hog, solo * 1e-6 * 3)
            hog.start_execute_from_dram(hprog)
            time.sleep(0.02)              # let the hog reach steady state
            t = run_and_time(v, vprog, 120.0)
            hog.wait_queue(180.0)
            slowed[hog_eng] = t

        s_same = slowed[4] / solo if solo else 0.0
        s_ctrl = slowed[8] / solo if solo else 0.0
        if s_ctrl > 1.15:
            verdict = "VOID (control moved)"
        elif s_same > 1.5:
            verdict = "MEMORY"
        elif s_same > 1.15:
            verdict = "mixed"
        else:
            verdict = "COMPUTE"
        print(f"  {label:>26} {M:>5} {solo:>10.1f} {slowed[4]:>11.1f} {s_same:>6.2f} "
              f"{slowed[8]:>11.1f} {s_ctrl:>6.2f}  {verdict}")


# ------------------------------------------------- intervention 3: ports ----
def intervention_ports(M, K, N, dt, reps):
    print("\n" + "=" * 96)
    print("INTERVENTION 3 -- port placement")
    print("=" * 96)
    print("Four engines, identical work each. Packed onto port 0 vs split across both.")
    print("memory-bound => splitting is worth up to 2x;  compute-bound => no change\n")
    print(f"  {'placement':>22} {'slowest engine us':>19} {'speedup':>9}  verdict")

    base = None
    for label, engs in (("packed  0,1,2,3", [0, 1, 2, 3]),
                        ("split   0,1,8,9", [0, 1, 8, 9])):
        engines = [make_engine(e) for e in engs]
        assert_card_alive(engines)
        progs = [build_matmul(ue, M, K, N, dt, reps) for ue in engines]
        for ue, p in zip(engines, progs):
            ue.start_execute_from_dram(p)
        for ue in engines:
            ue.wait_queue(180.0)
        # The slowest engine is the one a barrier would make everyone wait for.
        worst = max(latency_us(ue) for ue in engines)
        if base is None:
            base = worst
        sp = base / worst
        verdict = "MEMORY" if sp > 1.3 else "COMPUTE"
        print(f"  {label:>22} {worst:>19.1f} {sp:>9.2f}  {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bytes", "starve", "ports", "all"])
    ap.add_argument("--dev", default="xdma0")
    ap.add_argument("--target-ms", type=float, default=40.0,
                    help="aim each timed run at roughly this many ms")
    args = ap.parse_args()
    user_dma_core.set_dma_device(args.dev)

    # (K, N): the prefix LM's attention projection shape and the action expert's
    # gated-MLP shape -- the two that dominate their stages.
    shapes = [(2048, 2048), (1024, 4096)]
    m_list = [1, 2, 4, 6, 8, 16, 64, 256, 832]

    if args.mode in ("bytes", "all"):
        intervention_bytes(shapes, m_list, args.target_ms)
    if args.mode in ("starve", "all"):
        intervention_starve([
            ("prefix-like M=832", (832, 2048, 2048, TYPE.IF4, 2)),
            ("vision-like  M=256", (256, 1152, 1152, TYPE.IF4, 8)),
            ("denoise-like M=10", (10, 1024, 4096, TYPE.IF4, 40)),
            ("thin         M=1", (1, 2048, 2048, TYPE.IF4, 60)),
        ])
    if args.mode in ("ports", "all"):
        intervention_ports(832, 2048, 2048, TYPE.IF4, 2)


if __name__ == "__main__":
    main()
