#!/usr/bin/env python3
"""Standalone op-level benchmark: convolution workarounds vs the CONV hardware.

WHY THIS EXISTS. Every convolution in models/ is currently expressed with
non-convolution primitives, because the CONV/MAXPOOL instruction modes landed
only in andromeda 2abec6afd (2026-10-07). This script measures each of those
workarounds against the real CONV op at the SAME shapes the models use, so the
decision to migrate (or to drop the conv feature) rests on measured numbers
rather than on the assumption that a dedicated op must be faster.

It touches nothing under models/. It builds its own activations and weights,
runs each variant on one engine, and reports:

  true GF      the FLOPs a mathematically minimal convolution needs
  issued GF    the FLOPs the variant actually asks the hardware to perform
  prep MiB     extra DRAM a variant must materialize BEFORE computing
               (im2col windows, Toeplitz matrices, gather matrices). This is the
               cost the FLOP columns hide, and for Toeplitz it dominates.
  HW ms        measured on-chip latency (report_latency_in_us)
  GFLOPS       true GF / HW time -- deliberately TRUE work, not issued work, so
               a variant that computes 100x too much does not get credit for it
  % peak       against ONE engine's peak (clock x 128 FLOP/cycle)
  SNR dB       against the torch reference ('inf' = bit-exact)

WEIGHT PRECISION IS NOT CONTROLLED HERE. The workarounds run BF16 weights
through matmat_mul_core; the CONV op has no BF16 weight path at all
(conv2d_core takes TYPE.IF4 / IF8 / TQ4 X-stream formats only). The timings
below therefore compare BF16-matmul against INT4-conv. That difference is
reported, not corrected: a like-for-like precision comparison would need an
IF4 matmul baseline, and the point here is the cost of the op, not of the
number format. Any migration decision must treat the precision change as a
separate question -- it is a model-quality question for an audio front end.

DEPTHWISE IS NOT EXPRESSIBLE ON THE CONV HARDWARE. There is no groups/
depthwise mode. A depthwise conv run as a dense conv with a block-diagonal
weight costs C x the true FLOPs (C=1024 for the audio encoders), which is worse
than the Toeplitz scheme it would replace. So for depthwise cases this script
compares the two SOFTWARE schemes instead -- Toeplitz matmul (parakeet,
gemma4 audio) against tap-wise eltwise mul+add (mobilenetv2) -- because that,
not the conv op, is the available fix.

Usage:
    myvenv/bin/python conv_walkaround_bench.py --dev xdma1 [--report out.md]
    myvenv/bin/python conv_walkaround_bench.py --dev xdma1 --only omni_conv2
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import user_dma_core
from user_dma_core import (
    UnifiedEngine,
    TYPE,
    UE_VECTOR_SIZE,
    set_dma_device,
)

BPE = 2  # bf16


# ---------------------------------------------------------------------------
# Case table.
#
# Shapes are the ones the models actually run. `stage_share` is sourced, not
# guessed: each entry names where the number comes from so the report can be
# audited. Where a share is computed rather than measured, the arithmetic is
# given in the report generator below.
# ---------------------------------------------------------------------------

@dataclass
class Case:
    name: str
    model: str
    pattern: str            # im2col | toeplitz | per_tap | tapwise_eltwise | folded
    depthwise: bool
    # Conv1d geometry: (C_in, C_out, T_in, kernel, stride, dilation)
    c_in: int
    c_out: int
    length: int
    kernel: int
    stride: int = 1
    dilation: int = 1
    # Conv2d geometry (when in_hw is set, the case is 2D)
    in_hw: Optional[int] = None
    stage_share: str = ""
    share_note: str = ""
    conv_op_possible: bool = True
    occurrences: int = 1          # times this op runs per stage
    notes: str = ""

    @property
    def out_len(self) -> int:
        pad = self.dilation * (self.kernel - 1) // 2
        return (self.length + 2 * pad - self.dilation * (self.kernel - 1) - 1) // self.stride + 1

    @property
    def true_flops(self) -> int:
        """Minimal conv FLOPs: 2 * out_positions * C_out * C_in_eff * taps."""
        if self.in_hw is not None:
            out = self.out_hw ** 2
            taps = self.kernel * self.kernel
        else:
            out = self.out_len
            taps = self.kernel
        c_in_eff = 1 if self.depthwise else self.c_in
        return 2 * out * self.c_out * c_in_eff * taps

    @property
    def out_hw(self) -> int:
        assert self.in_hw is not None
        pad = self.kernel // 2
        return (self.in_hw + 2 * pad - self.kernel) // self.stride + 1


CASES: list[Case] = [
    # ---------------- dense Conv1d: audio front ends ----------------
    Case(
        name="omni_conv1", model="qwen2.5_omni_7b (audio)", pattern="im2col",
        depthwise=False, c_in=128, c_out=1280, length=600, kernel=3, stride=1,
        stage_share="0.15% of audio-stage FLOPs", occurrences=1,
        share_note=("0.590 GF of 395.7 GF total audio FLOPs. Audio stage = conv "
                    "front end (3.54 GF) + 32 encoder layers at H=1280, FFN=5120, "
                    "S=300 (392.2 GF). Stage wall time 166/336/400 ms for "
                    "low/medium/high in models/qwen2.5_omni_7b/benchmark.md."),
        notes="Conv1d(128->1280, k3, s1, pad1). conv1d_core's docstring names this exact layer.",
    ),
    Case(
        name="omni_conv2", model="qwen2.5_omni_7b (audio)", pattern="im2col",
        depthwise=False, c_in=1280, c_out=1280, length=600, kernel=3, stride=2,
        stage_share="0.75% of audio-stage FLOPs", occurrences=1,
        share_note="2.949 GF of 395.7 GF. Same decomposition as omni_conv1.",
        notes="Conv1d(1280->1280, k3, s2, pad1) + GELU. 600 mel frames -> 300 tokens.",
    ),
    # ---------------- dense Conv1d: vocoder ----------------
    Case(
        name="bigvgan_conv1d", model="qwen2.5_omni_7b (token2wav)", pattern="per_tap",
        depthwise=False, c_in=512, c_out=512, length=1000, kernel=3, stride=1,
        stage_share="~95% of token2wav compute (est.; conv IS the vocoder)",
        share_note=("token2wav runs ~4.5 min per 3 s of audio on one engine "
                    "(session-measured, memory note omni-token2wav-fpga-port). The "
                    "AMP blocks are k-tap conv1d plus ConvTranspose1d upsampling, so "
                    "conv IS the stage rather than a fraction of it."),
        notes="k separate matmuls, each accumulating through the previous partial's C term.",
    ),
    # ---------------- dense Conv2d: vision / CNN ----------------
    Case(
        name="mbv2_stem", model="mobilenetv2_224", pattern="im2col",
        depthwise=False, c_in=3, c_out=32, length=0, kernel=3, stride=2,
        in_hw=224,
        stage_share="~3.6% of mobilenetv2-224 FLOPs (est. 22 MF of ~600 MF)",
        share_note=("0.022 GF of the 224x224 network. Small in isolation, but "
                    "mobilenetv2 is conv end to end, so the per-op result "
                    "generalises across every block."),
        notes="conv2d_3x3_stride2_dram: 'row-by-row im2col + matmul', HWC layout.",
    ),
    Case(
        name="mobilesam_3x3", model="mobilesam", pattern="im2col",
        depthwise=False, c_in=64, c_out=64, length=0, kernel=3, stride=1,
        in_hw=64,
        stage_share="~1-2% per block (est.), repeated across the encoder",
        share_note=("One of the repeated 3x3 blocks in the image encoder. "
                    "_conv3x3_im2col_row_pbi materialises ONE output row's "
                    "im2col per row, so cost scales with H."),
        notes="PBI im2col per output row + matmul.",
    ),
    # ---------------- depthwise: conv op CANNOT express these ----------------
    Case(
        name="parakeet_dw", model="parakeet (audio)", pattern="toeplitz",
        depthwise=True, c_in=1024, c_out=1024, length=128, kernel=9, stride=1,
        conv_op_possible=False, occurrences=24,
        stage_share="~1% of encoder FLOPs but ~5% of encoder TIME (est.): x24 layers at 1.2% of peak",
        share_note=("Toeplitz issues 2*D*L^2 against a true 2*D*L*k: 56.9x at "
                    "L=512, 113.8x at L=1024. parakeet_test.py's own FLOP comment "
                    "concedes this ('full Toeplitz matmul, not just 9-tap'). The "
                    "same block also runs SiLU as an identity matmul at 2*D*L^2."),
        notes="D=1024 channels, each a (1,L)@(L,L) matmul. d_model=1024, conv_kernel=9.",
    ),
    Case(
        name="parakeet_dw_long", model="parakeet (audio, 40 s)", pattern="toeplitz",
        depthwise=True, c_in=1024, c_out=1024, length=512, kernel=9, stride=1,
        conv_op_possible=False, occurrences=24,
        stage_share="~14% of encoder TIME at 40 s audio (est.); Toeplitz cost grows as L^2",
        share_note=("The Toeplitz penalty grows as L/k, so the same op costs 4x "
                    "more waste at 40 s than at 10 s. Included to show that the "
                    "scheme degrades with utterance length, which a fixed-shape "
                    "measurement would hide."),
        notes="Same op as parakeet_dw at 4x the sequence length.",
    ),
    Case(
        name="gemma4_audio_dw", model="gemma4_e2b/e4b (audio)", pattern="toeplitz",
        depthwise=True, c_in=1024, c_out=1024, length=128, kernel=5, stride=1,
        conv_op_possible=False, occurrences=12,
        stage_share="~1% of encoder FLOPs but ~4% of encoder TIME (est.): x12 blocks at 0.7% of peak",
        share_note=("102.4x issued-vs-true at L=512, 204.8x at L=1024. "
                    "depthwise_conv1d_core_dram runs the D channels SEQUENTIALLY, "
                    "so latency is D x matmul cost by construction."),
        notes="AUD_H=1024, conv_kernel_size=5. Toeplitz is (D, L, L) bf16 resident in DRAM.",
    ),
]


# ---------------------------------------------------------------------------
# Preparation memory: the cost the FLOP columns hide.
# ---------------------------------------------------------------------------

def prep_bytes(case: Case) -> tuple[int, str]:
    """Extra DRAM a variant must materialize before it can compute."""
    if case.pattern == "toeplitz":
        # One (L, L) bf16 matrix PER CHANNEL, per layer. This is the whole point
        # of the memory column: a 9-tap filter is stored as an L x L matrix.
        n = case.c_in * case.length * case.length * BPE
        return n, (f"{case.c_in} x ({case.length} x {case.length}) bf16 Toeplitz "
                   f"matrices, one per channel, resident per layer")
    if case.pattern == "im2col":
        if case.in_hw is not None:
            rows = case.out_hw ** 2
            k = case.kernel * case.kernel * case.c_in
        else:
            rows = case.out_len
            k = case.kernel * case.c_in
        n = rows * k * BPE
        return n, f"im2col buffer [{rows} x {k}] bf16 ({case.kernel}-tap window materialized)"
    if case.pattern == "per_tap":
        # k weight "taps" reshaped to [Cout, Cin] each, plus 2 accumulators.
        n = case.kernel * case.c_out * case.c_in * BPE + 2 * case.out_len * case.c_out * BPE
        return n, f"{case.kernel} tap matrices [{case.c_out} x {case.c_in}] + 2 accumulators"
    return 0, "none"


def conv_op_prep_bytes(case: Case) -> tuple[int, str]:
    """What the CONV op itself must stage: packed activation map + weight stream."""
    if case.in_hw is not None:
        act = case.c_in * (case.in_hw + 2 * (case.kernel // 2)) ** 2 * BPE
    else:
        pad = case.dilation * (case.kernel - 1) // 2
        act = case.c_in * (case.length + 2 * pad) * BPE
    return act, "padded activation map (no window materialization)"


# ---------------------------------------------------------------------------
# Measurement helpers.
# ---------------------------------------------------------------------------

@dataclass
class Result:
    variant: str
    issued_flops: int
    prep_bytes: int
    prep_desc: str
    hw_ms: float = float("nan")
    snr_db: float = float("nan")
    inst_bytes: int = 0
    error: str = ""
    extra: str = ""


def snr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref = ref.float().flatten()
    got = got.float().flatten()
    n = min(ref.numel(), got.numel())
    ref, got = ref[:n], got[:n]
    err = ref - got
    p_s = float((ref * ref).sum())
    p_n = float((err * err).sum())
    if p_n == 0.0:
        return float("inf")
    if p_s == 0.0:
        return float("-inf")
    return 10.0 * math.log10(p_s / p_n)


def timed_program(ue: UnifiedEngine, emit: Callable[[], None]) -> tuple[float, int]:
    """Capture one program, run it, return (ms on chip, instruction bytes)."""
    ue.start_capture()
    emit()
    ue.generate_instruction_halt()
    ue.stop_capture()
    addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(addr)
    size = ue.get_capture_instruction_size_bytes()
    ue.allocate_program_dram(size)
    ue.start_execute_from_dram(addr)
    ue.wait_queue(600.0)
    return ue.report_latency_in_us() / 1000.0, size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", default="xdma1",
                    help="DMA device. xdma1 = kintex7 (default), xdma0 = alveo.")
    ap.add_argument("--only", nargs="+", metavar="NAME",
                    help="Run only these cases (see CASES).")
    ap.add_argument("--report", default="conv_walkaround_report.md",
                    help="Markdown report path.")
    ap.add_argument("--list", action="store_true", help="List cases and exit.")
    args = ap.parse_args()

    if args.list:
        for c in CASES:
            geo = (f"{c.in_hw}x{c.in_hw}" if c.in_hw else f"T={c.length}")
            print(f"  {c.name:18s} {c.model:32s} {c.pattern:16s} "
                  f"C{c.c_in}->{c.c_out} k{c.kernel} s{c.stride} {geo} "
                  f"{'DEPTHWISE' if c.depthwise else ''}")
        return

    cases = [c for c in CASES if not args.only or c.name in args.only]
    if not cases:
        sys.exit(f"no case matched {args.only}")

    set_dma_device(args.dev)
    user_dma_core.configure_clock_from_hardware()
    clock_mhz = 1000.0 / user_dma_core.CLOCK_CYCLE_TIME_NS
    peak_1e = clock_mhz * 1e6 * 128 / 1e9  # one engine, 128 FLOP/cycle
    print(f"dev={args.dev} clock={clock_mhz:.2f} MHz  "
          f"single-engine peak={peak_1e:.2f} GFLOPS  lanes={UE_VECTOR_SIZE}")

    results: dict[str, list[Result]] = {}
    for case in cases:
        print(f"\n=== {case.name} ({case.model}, {case.pattern}"
              f"{', DEPTHWISE' if case.depthwise else ''}) ===")
        results[case.name] = run_case(case, args.dev, peak_1e)

    write_report(args.report, cases, results, args.dev, clock_mhz, peak_1e)
    print(f"\nreport -> {args.report}")


def run_case(case: Case, dev: str, peak_1e: float) -> list[Result]:
    """Run every applicable variant of one case. Implemented in bench_impl."""
    import conv_walkaround_impl as impl
    return impl.run(case, dev, peak_1e, prep_bytes, conv_op_prep_bytes,
                    timed_program, snr_db, Result)


def write_report(path: str, cases: list[Case], results: dict[str, list[Result]],
                 dev: str, clock_mhz: float, peak_1e: float) -> None:
    import conv_walkaround_impl as impl
    impl.write_report(path, cases, results, dev, clock_mhz, peak_1e,
                      prep_bytes, conv_op_prep_bytes)


if __name__ == "__main__":
    main()
