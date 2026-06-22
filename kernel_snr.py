#!/usr/bin/env python3
"""Kernel-only repro for quantized_matmat_core (IF4) run-to-run nondeterminism.

Same A / packed-B / scales / addresses every run; the program is captured ONCE and
re-executed N times, so any output difference is the hardware kernel itself.

  matmat_mul_core(is_B_quantized=True, IF4)  -> reference, must be bit-identical (control)
  quantized_matmat_core(IF4)                 -> test, expected to wobble ~1-2 bf16 ULP

See overnight_quantized_matmat_evidence/PASSOFF_quantized_matmat_core_nondeterminism.md for the write-up.
Drop this file next to user_dma_core.py and run it from there: `python3 kernel_snr.py`.
"""
import sys, os
from collections import Counter
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # import user_dma_core from this dir
import user_dma_core
from user_dma_core import UnifiedEngine, set_dma_device, TYPE

set_dma_device("xdma0")                               # board name; change if yours differs
os.environ["UE_AXI_DATA_WIDTH_BITS"] = "256"
user_dma_core.UE_AXI_DATA_WIDTH_BITS = 256

# (M, K, N, seed) -- M=1 is the production decode GEMV; bias toward large N/K. Add rows to scale.
SHAPES = [(1, 2048, 2048, 0),
          (1, 2048, 4096, 1),
          (1, 4096, 2048, 2)]
N_REPS = 100      # test executions per shape (use more for small shapes -- wobble is rare there)
REF_REPS = 100    # reference executions (determinism control)


def bf16f32(u16):
    """bf16 bit pattern (uint16) -> float32, by left-shifting into the f32 mantissa."""
    return (u16.astype(np.uint32) << 16).view(np.float32)


def snr_db(ref_f32, x_f32):
    err = ref_f32 - x_f32
    npow = np.mean(err ** 2)
    if npow == 0:
        return float("inf")
    return float(10.0 * np.log10(np.mean(ref_f32 ** 2) / npow))


def read_u16(ue, addr, n):
    buf = bytearray(n * 2)
    ue.dma_read(ue.c2h_device, addr, buf, n * 2)
    return np.frombuffer(bytes(buf), dtype=np.uint16).copy()


def build_program(ue, fn):
    """Capture a single kernel call as a standalone program; return its DRAM address."""
    ue.start_capture()
    fn()
    ue.stop_capture()
    ue.generate_instruction_halt()
    addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())
    ue.clear_capture_buffer()
    return addr


def run_shape(M, K, N, seed):
    torch.manual_seed(seed)
    ue = UnifiedEngine()

    # 1. operands -- one A, one quantized B, fixed addresses, reused by both kernels
    a = (torch.randn(M, K, dtype=torch.bfloat16) / np.sqrt(K)).to(torch.bfloat16)
    w = torch.randn(N, K, dtype=torch.bfloat16)
    A_ADDR = ue.allocate_tensor_dram(M * K * 2)
    OUT_ADDR = ue.allocate_tensor_dram(M * N * 2)
    B_ADDR, SC_ADDR = ue.quantize_weight(w, N, K, data_type=TYPE.IF4)   # writes B+scales to params DRAM
    ue.dma_to_accelerator_memory(A_ADDR, a)

    prog_ref = build_program(ue, lambda: ue.matmat_mul_core(
        M=M, K=K, N=N, A_DRAM_ADDR=A_ADDR, B_DRAM_ADDR=B_ADDR, OUTPUT_DRAM_ADDR=OUT_ADDR,
        is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=SC_ADDR))
    prog_test = build_program(ue, lambda: ue.quantized_matmat_core(
        M=M, K=K, N=N, A_DRAM_ADDR=A_ADDR, B_DRAM_ADDR=B_ADDR, OUTPUT_DRAM_ADDR=OUT_ADDR,
        SCALE_DRAM_ADDR=SC_ADDR, data_type=TYPE.IF4))

    def execute(prog):
        ue.dma_to_accelerator_memory(OUT_ADDR, torch.zeros(M * N, dtype=torch.bfloat16))  # zero-fill 0x00
        ue.start_execute_from_dram(prog)
        ue.wait_queue(30.0)

    # 2. reference control -- matmat_mul_core, run REF_REPS times (expected bit-identical)
    ref_runs = [run_once(ue, execute, prog_ref, OUT_ADDR, M * N) for _ in range(REF_REPS)]

    # 3. test -- quantized_matmat_core, run N_REPS times on the same bytes
    test_runs = [run_once(ue, execute, prog_test, OUT_ADDR, M * N) for _ in range(N_REPS)]

    return dict(M=M, K=K, N=N, ref=metrics(ref_runs), test=metrics(test_runs))


def run_once(ue, execute, prog, addr, n):
    execute(prog)
    return read_u16(ue, addr, n)


def metrics(runs):
    """Divergence stats vs the modal (most-common) output, on decoded bf16 values."""
    modal = np.frombuffer(Counter(r.tobytes() for r in runs).most_common(1)[0][0], dtype=np.uint16)
    modal_f = bf16f32(modal)
    div = [r for r in runs if r.tobytes() != modal.tobytes()]
    return dict(
        runs=len(runs),
        diverged_runs=len(div),
        distinct_of_diverged=len({r.tobytes() for r in div}),
        avg_snr_div=(float(np.mean([snr_db(modal_f, bf16f32(r)) for r in div])) if div else None),
        max_ndiff=(max(int(np.sum(r != modal)) for r in div) if div else None),
        max_abs=(max(float(np.abs(bf16f32(r) - modal_f).max()) for r in div) if div else None),
    )


HDR = ("M", "K", "N", "runs", "diverged_runs", "distinct_of_diverged", "avg_snr_div(dB)", "ndiff", "max|d|")
FMT = "{:>3} {:>6} {:>6} {:>6} {:>14} {:>21} {:>16} {:>7} {:>9}"


LEGEND = """
Columns:
  diverged_runs         how many runs gave a different result
  distinct_of_diverged  how many different results those diverged runs produced
  avg_snr_div           average SNR (dB) of the diverged runs vs the common output; higher = smaller difference
  ndiff                 in the worst run, how many output values (bf16 elements) differed from the common output
  max|d|                the largest gap seen on any single output value (as a bf16 number, not raw bits)"""

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_snr_run.log")


def table_lines(title, rows, key):
    lines = [f"\n=== {title} ===", FMT.format(*HDR)]
    for r in rows:
        m = r[key]
        nodiv = m["diverged_runs"] == 0                              # deterministic -> dash divergence cols
        lines.append(FMT.format(
            r["M"], r["K"], r["N"], m["runs"],
            "-" if nodiv else m["diverged_runs"],
            "-" if nodiv else m["distinct_of_diverged"],
            "-" if m["avg_snr_div"] is None else f"{m['avg_snr_div']:.1f}",
            "-" if m["max_ndiff"] is None else m["max_ndiff"],
            "-" if m["max_abs"] is None else f"{m['max_abs']:.4f}"))
    return lines


def main():
    rows = [run_shape(*s) for s in SHAPES]
    tables = "\n".join(
        table_lines("reference: matmat_mul_core (IF4), must be deterministic", rows, "ref")
        + table_lines("test: quantized_matmat_core (IF4)", rows, "test")
    )
    print(tables)                                   # stdout: tables only (after the library's setup noise)
    with open(LOG_PATH, "w") as f:                  # log file: tables + column legend, no engine/quantize noise
        f.write(tables + "\n" + LEGEND + "\n")
    print(f"\n[results written to {LOG_PATH}]")


if __name__ == "__main__":
    main()
