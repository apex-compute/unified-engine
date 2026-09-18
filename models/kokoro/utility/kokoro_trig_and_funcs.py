#!/usr/bin/env python3
"""
Kokoro transcendental & generator-composition hardware proofs.

WHY THIS FILE EXISTS
--------------------
The accelerator has no sign / floor / sqrt / divide / sin / cos / exp cores.
Kokoro's ISTFTNet generator needs all of them, so each one had to be BUILT out
of the primitives that do exist (clamp ramps, LALU eltwise, matmul, CORDIC
rotations) and then PROVEN on hardware before kokoro_fpga.py could rely on it.

These proofs originally lived in the shared user_hw_test.py, where ~800 lines
of kokoro-only bring-up sat in the middle of the repo-wide regression suite.
They are model-specific, so they live here now. Nothing about them changed in
the move.

WHAT EACH PROOF ESTABLISHES
---------------------------
Transcendentals (the "trig and friends" set):

  exp_via_sigmoid_test
      exp(x) is exactly derivable from the LALU's sigmoid:
      exp(x) = sigmoid(x) / (1 - sigmoid(x)). Needed for ISTFTNet's
      `spec = exp(raw)` -- the softmax epilogue in Section 5d.

  sincos_bounded_poly_test
      sin(x)/cos(x) for BOUNDED-domain x via a fixed-degree odd/even
      polynomial, no range reduction. This is the cheap path, valid only
      once the argument is already small -- which is what magic_round_test
      below guarantees. Used by Snake1D and by the SineGen oscillator.

  magic_round_test
      Range reduction by the "magic number" trick: adding and subtracting a
      large constant (192.0) forces the float to drop its fractional bits, so
      p - round(p) lands the argument back in one period. This is the floor()
      the hardware does not have. Snake1D's cos(2ax) argument reaches ~15
      turns; this folds it to one.

  cordic_atan2_magnitude_test
      atan2 and magnitude of a complex spectrum via iterated CORDIC rotations
      (8 iterations), standing in for torch.abs / torch.angle in
      `har = cat([har_spec, har_phase])`. No divide, no sqrt, no atan needed.

  snake_vs_gelu_timing_test
      Not a correctness proof -- a cost one. Side-by-side latency of a native
      LALU activation (gelu via activation_core) against the composed Snake1D,
      so the price of synthesising a missing transcendental is on the record.

Generator compositions (Section 5b-5d) -- every op below already existed
individually; what had never been assembled on hardware is the COMPOSITION:

  instance_norm_large_T_test
      InstanceNorm1d at T=30601 via chunked reduction, avoiding a large-N
      transpose.

  sinegen_wrapped_phase_test
      SineGen (SourceModuleHnNSF.l_sin_gen) with the phase kept WRAPPED in
      [0,1) turns rather than accumulating -- the accumulating form loses all
      precision in bf16 after a few thousand frames.

  stft_istft_matmul_test
      Kokoro's TorchSTFT (n_fft=20, hop=5, periodic Hann) forward AND inverse
      expressed as block-Toeplitz matmuls.

The _kg_* helpers are the shared scaffolding these proofs run on (program
exec/teardown, row tiling, the clamp-ramp step function, and the sine
polynomial evaluator).

Measured SNR ceilings for these constructions are recorded in the repo notes;
each test asserts its own threshold.

USAGE
-----
    python models/kokoro/utility/kokoro_trig_and_funcs.py [--dev xdma0]

Runs against real hardware. Results append to the same record_test registry
user_hw_test.py uses, so --summary-path output is directly comparable.
"""

import argparse
import math
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F

from user_dma_core import (
    DMA_DEVICE_C2H,
    DMA_DEVICE_H2C,
    DRAM_ACTIVATION_ADDR,
    DRAM_INSTRUCTION_ADDR,
    INSTRUCTION_SIZE_BYTES,
    LALU_MODE,
    TYPE,
    UE_MODE,
    UE_VECTOR_SIZE,
    URAM_SECTION,
    calculate_snr,
    set_dma_device,
    ue_35bit_addr_shifter,
    UnifiedEngine,
)

# The test registry lives in user_hw_test.py so a run of these proofs lands in
# the same summary the repo-wide suite writes. Importing it is side-effect free:
# every device touch in that file sits under its `if __name__ == "__main__"`.
from user_hw_test import (
    record_test,
    write_test_summary,
    _round_up_vec,
)


def exp_via_sigmoid_test(N: int = 64, x_min: float = -3.0, x_max: float = 3.0):
    """Proves exp(x) (needed for Kokoro ISTFTNet's ``spec = exp(raw)``) is exactly derivable from
    the native ``sigmoid`` activation plus elementwise ops:
        sigmoid(x) = 1/(1+e^-x)  =>  e^x = sigmoid(x) / (1 - sigmoid(x))

    Reciprocal honesty note: there is no exposed standalone elementwise reciprocal primitive (see
    header). This uses HONEST OPTION (a) -- a Newton-Raphson reciprocal (r_{n+1} = r_n*(2 - a*r_n))
    built entirely from eltwise_mul_core/eltwise_add_core/broadcast ops (via eltwise_core_dram),
    seeded with a constant initial guess valid over the known bounded range of ``1 - sigmoid(x)``
    for x in [x_min, x_max].
    """
    torch.manual_seed(0)
    N_pad = ((N + 63) // 64) * 64  # round up to UE_VECTOR_SIZE-friendly width
    x = torch.linspace(x_min, x_max, N, dtype=torch.float32).to(torch.bfloat16)
    x_pad = torch.zeros(N_pad, dtype=torch.bfloat16); x_pad[:N] = x
    ref = torch.exp(x.float())

    ue = UnifiedEngine()
    A = ue.allocate_tensor_dram(N_pad * 2)
    IDENT = ue.allocate_tensor_dram(N_pad * N_pad * 2)
    S = ue.allocate_tensor_dram(N_pad * 2)          # sigmoid(x)
    D = ue.allocate_tensor_dram(N_pad * 2)          # 1 - sigmoid(x)
    R = ue.allocate_tensor_dram(N_pad * 2)          # reciprocal(D), Newton-Raphson
    TMP = ue.allocate_tensor_dram(N_pad * 2)
    OUT = ue.allocate_tensor_dram(N_pad * 2)         # exp(x) = S * R

    ue.start_capture()
    ue.activation_core(M=1, N=N_pad, A_DRAM_ADDR=A, OUTPUT_DRAM_ADDR=S,
                        IDENTITY_DRAM_ADDR=IDENT, activation="sigmoid")
    # D = 1 - S  ==  (-1)*S + 1
    ue.eltwise_core_dram(1, N_pad, S, None, D, UE_MODE.MUL_BROADCAST, scalar=-1.0)
    ue.eltwise_core_dram(1, N_pad, D, None, D, UE_MODE.ADD_BROADCAST, scalar=1.0)
    # Newton-Raphson reciprocal of D: r0 = constant seed (1 - sigmoid(x) in [x_min,x_max] is bounded
    # well away from 0 for this range; a flat mid-range seed converges in a few iterations).
    d_lo = float((1.0 - torch.sigmoid(torch.tensor(x_max, dtype=torch.float32))).item())
    d_hi = float((1.0 - torch.sigmoid(torch.tensor(x_min, dtype=torch.float32))).item())
    r0 = 2.0 / (d_lo + d_hi)  # 1 / midpoint(D range)
    ue.eltwise_core_dram(1, N_pad, D, None, R, UE_MODE.MUL_BROADCAST, scalar=0.0)
    ue.eltwise_core_dram(1, N_pad, R, None, R, UE_MODE.ADD_BROADCAST, scalar=r0)
    for _ in range(6):
        # TMP = D * R
        ue.eltwise_core_dram(1, N_pad, D, R, TMP, UE_MODE.ELTWISE_MUL)
        # TMP = 2 - D*R  ==  (-1)*TMP + 2
        ue.eltwise_core_dram(1, N_pad, TMP, None, TMP, UE_MODE.MUL_BROADCAST, scalar=-1.0)
        ue.eltwise_core_dram(1, N_pad, TMP, None, TMP, UE_MODE.ADD_BROADCAST, scalar=2.0)
        # R = R * TMP
        ue.eltwise_core_dram(1, N_pad, R, TMP, R, UE_MODE.ELTWISE_MUL)
    # OUT = S * R  (= exp(x))
    total_flops = ue.eltwise_core_dram(1, N_pad, S, R, OUT, UE_MODE.ELTWISE_MUL)
    ue.stop_capture()
    ue.generate_instruction_halt()
    prog = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(prog)
    inst_bytes = ue.get_capture_instruction_size_bytes()
    ue.allocate_program_dram(inst_bytes)

    ue.dma_to_accelerator_memory(A, x_pad.reshape(-1).contiguous())
    ue.dma_to_accelerator_memory(IDENT, torch.eye(N_pad, dtype=torch.bfloat16).reshape(-1).contiguous())

    ue.start_execute_from_dram(prog)
    ue.wait_queue(10.0)
    ue.report_timing_and_instruction_count()
    gflops, _ = ue.report_flop_rate_gflops(total_flops)

    out = ue.dma_from_accelerator_memory(OUT, (N_pad,))[:N]
    snr_db = calculate_snr(ref, out)
    print(f"[exp_via_sigmoid] N={N} range=[{x_min},{x_max}] SNR={snr_db:.2f} dB GFLOPS={gflops:.2f}")
    assert snr_db >= 25.0 or snr_db == float("inf"), \
        f"exp_via_sigmoid N={N} SNR {snr_db:.2f} dB < 25 dB"
    record_test("exp_via_sigmoid", f"N={N},range=[{x_min},{x_max}]", snr_db=snr_db, gflops=gflops,
                inst_bytes=inst_bytes)
    ue.clear_capture_buffer(); ue.reset_tensor_dram_addr(); ue.reset_program_dram_addr()


def sincos_bounded_poly_test(N: int = 64, x_min: float = -1.0, x_max: float = 1.0):
    """Proves sin(x)/cos(x) for BOUNDED-domain x (as in Kokoro's Snake1D activation and
    iSTFT phase reconstruction, both post-normalization / clamped to roughly [-1, 1]) can be
    computed via a fixed low-degree Taylor polynomial built entirely from eltwise_mul_core /
    broadcast_add -- no transcendental hardware needed. Uses independent odd (sin) and even (cos)
    polynomials (NOT cos(x)=sin(x+pi/2), since x+pi/2 would leave the fit range for x in [-1,1]).
    """
    torch.manual_seed(0)
    # sin(x) ~= x - x^3/6 + x^5/120 - x^7/5040   (degree-7 Taylor, evaluated via Horner in x^2)
    # cos(x) ~= 1 - x^2/2 + x^4/24 - x^6/720     (degree-6 Taylor, evaluated via Horner in x^2)
    x_np = torch.linspace(x_min, x_max, 4001, dtype=torch.float64)
    sin_taylor = x_np - x_np**3/6 + x_np**5/120 - x_np**7/5040
    cos_taylor = 1 - x_np**2/2 + x_np**4/24 - x_np**6/720
    max_sin_err = (sin_taylor - torch.sin(x_np)).abs().max().item()
    max_cos_err = (cos_taylor - torch.cos(x_np)).abs().max().item()
    assert max_sin_err < 1e-3, f"sin taylor fit error too large: {max_sin_err}"
    assert max_cos_err < 1e-3, f"cos taylor fit error too large: {max_cos_err}"

    N_pad = ((N + 63) // 64) * 64
    x = torch.linspace(x_min, x_max, N, dtype=torch.float32).to(torch.bfloat16)
    x_pad = torch.zeros(N_pad, dtype=torch.bfloat16); x_pad[:N] = x
    sin_ref = torch.sin(x.float())
    cos_ref = torch.cos(x.float())

    ue = UnifiedEngine()
    X = ue.allocate_tensor_dram(N_pad * 2)
    X2 = ue.allocate_tensor_dram(N_pad * 2)     # x^2
    SIN = ue.allocate_tensor_dram(N_pad * 2)
    COS = ue.allocate_tensor_dram(N_pad * 2)
    TMP = ue.allocate_tensor_dram(N_pad * 2)

    ue.start_capture()
    # X2 = x * x
    ue.eltwise_core_dram(1, N_pad, X, X, X2, UE_MODE.ELTWISE_MUL)

    # cos(x) ~= 1 - x^2/2 + x^4/24 - x^6/720
    #         = 1 + x^2*(-1/2 + x^2*(1/24 + x^2*(-1/720)))   Horner in x2 = X2
    ue.eltwise_core_dram(1, N_pad, X2, None, TMP, UE_MODE.MUL_BROADCAST, scalar=-1.0 / 720.0)
    ue.eltwise_core_dram(1, N_pad, TMP, None, TMP, UE_MODE.ADD_BROADCAST, scalar=1.0 / 24.0)
    ue.eltwise_core_dram(1, N_pad, TMP, X2, TMP, UE_MODE.ELTWISE_MUL)
    ue.eltwise_core_dram(1, N_pad, TMP, None, TMP, UE_MODE.ADD_BROADCAST, scalar=-1.0 / 2.0)
    ue.eltwise_core_dram(1, N_pad, TMP, X2, TMP, UE_MODE.ELTWISE_MUL)
    total_flops = ue.eltwise_core_dram(1, N_pad, TMP, None, COS, UE_MODE.ADD_BROADCAST, scalar=1.0)

    # sin(x) ~= x - x^3/6 + x^5/120 - x^7/5040
    #         = x * (1 + x^2*(-1/6 + x^2*(1/120 + x^2*(-1/5040))))   Horner in x2 = X2
    ue.eltwise_core_dram(1, N_pad, X2, None, TMP, UE_MODE.MUL_BROADCAST, scalar=-1.0 / 5040.0)
    ue.eltwise_core_dram(1, N_pad, TMP, None, TMP, UE_MODE.ADD_BROADCAST, scalar=1.0 / 120.0)
    ue.eltwise_core_dram(1, N_pad, TMP, X2, TMP, UE_MODE.ELTWISE_MUL)
    ue.eltwise_core_dram(1, N_pad, TMP, None, TMP, UE_MODE.ADD_BROADCAST, scalar=-1.0 / 6.0)
    ue.eltwise_core_dram(1, N_pad, TMP, X2, TMP, UE_MODE.ELTWISE_MUL)
    ue.eltwise_core_dram(1, N_pad, TMP, None, TMP, UE_MODE.ADD_BROADCAST, scalar=1.0)
    total_flops += ue.eltwise_core_dram(1, N_pad, TMP, X, SIN, UE_MODE.ELTWISE_MUL)
    ue.stop_capture()
    ue.generate_instruction_halt()
    prog = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(prog)
    inst_bytes = ue.get_capture_instruction_size_bytes()
    ue.allocate_program_dram(inst_bytes)

    ue.dma_to_accelerator_memory(X, x_pad.reshape(-1).contiguous())

    ue.start_execute_from_dram(prog)
    ue.wait_queue(10.0)
    ue.report_timing_and_instruction_count()
    gflops, _ = ue.report_flop_rate_gflops(total_flops)

    sin_out = ue.dma_from_accelerator_memory(SIN, (N_pad,))[:N]
    cos_out = ue.dma_from_accelerator_memory(COS, (N_pad,))[:N]
    sin_snr_db = calculate_snr(sin_ref, sin_out)
    cos_snr_db = calculate_snr(cos_ref, cos_out)
    print(f"[sincos_bounded_poly] N={N} range=[{x_min},{x_max}] "
          f"sin_SNR={sin_snr_db:.2f} dB cos_SNR={cos_snr_db:.2f} dB GFLOPS={gflops:.2f}")
    assert sin_snr_db >= 35.0 or sin_snr_db == float("inf"), \
        f"sincos_bounded_poly sin N={N} SNR {sin_snr_db:.2f} dB < 35 dB"
    assert cos_snr_db >= 35.0 or cos_snr_db == float("inf"), \
        f"sincos_bounded_poly cos N={N} SNR {cos_snr_db:.2f} dB < 35 dB"
    record_test("sincos_bounded_poly_sin", f"N={N},range=[{x_min},{x_max}]", snr_db=sin_snr_db,
                gflops=gflops, inst_bytes=inst_bytes)
    record_test("sincos_bounded_poly_cos", f"N={N},range=[{x_min},{x_max}]", snr_db=cos_snr_db,
                gflops=gflops, inst_bytes=inst_bytes)
    ue.clear_capture_buffer(); ue.reset_tensor_dram_addr(); ue.reset_program_dram_addr()


def snake_vs_gelu_timing_test(M: int = 1024, N: int = 64, alpha: float = 1.0):
    """Side-by-side latency of a native LALU activation (gelu via activation_core) vs Kokoro's
    Snake1D, which has no core and is built from the cos Taylor polynomial:
        snake(x) = x + sin^2(a*x)/a = x + (1 - cos(2*a*x)) / (2*a)
    cos is the degree-6 series 1 - z^2/2 + z^4/24 - z^6/720 (Horner in z^2), valid for |z|<~2,
    so the input is drawn in [-1, 1] with alpha=1 (|2*a*x| <= 2). Same M x N bf16 buffer for
    both, each in its own program so the reported latency isolates one activation.
    """
    torch.manual_seed(0)
    x = torch.linspace(-1.0, 1.0, M * N, dtype=torch.float32).to(torch.bfloat16)
    xf = x.float()
    gelu_ref = torch.nn.functional.gelu(xf)
    snake_ref = xf + (1.0 / alpha) * torch.sin(alpha * xf) ** 2

    def _finish(ue, total_flops, out_dram, ref, label):
        ue.stop_capture()
        ue.generate_instruction_halt()
        prog = ue.get_program_dram_addr()
        ue.write_captured_instructions_to_dram(prog)
        inst_bytes = ue.get_capture_instruction_size_bytes()
        ue.allocate_program_dram(inst_bytes)
        ue.start_execute_from_dram(prog)
        ue.wait_queue(10.0)
        latency_cycles, inst_count = ue.report_timing_and_instruction_count()
        latency_us = latency_cycles * ue._clock_period_ns / 1e3
        gflops, _ = ue.report_flop_rate_gflops(total_flops)
        out = ue.dma_from_accelerator_memory(out_dram, (M * N,))
        snr_db = calculate_snr(ref, out)
        print(f"[snake_vs_gelu] {label:6s} M={M} N={N} elems={M*N} "
              f"latency={latency_us:.2f} us ({latency_us*1e3/(M*N):.2f} ns/elem) "
              f"insts={inst_count} SNR={snr_db:.2f} dB")
        record_test(f"snake_vs_gelu_{label}", f"M={M},N={N}", snr_db=snr_db, gflops=gflops,
                    inst_bytes=inst_bytes)
        ue.clear_capture_buffer(); ue.reset_tensor_dram_addr(); ue.reset_program_dram_addr()
        return latency_us, snr_db

    # ---- GELU: one activation_core pass (identity-matmul epilogue) ----
    ue = UnifiedEngine()
    X = ue.allocate_tensor_dram(M * N * 2)
    OUT = ue.allocate_tensor_dram(M * N * 2)
    IDENT = ue.allocate_tensor_dram(N * N * 2)
    ue.dma_to_accelerator_memory(X, x.reshape(-1).contiguous())
    ue.dma_to_accelerator_memory(IDENT, torch.eye(N, dtype=torch.bfloat16).reshape(-1).contiguous())
    ue.start_capture()
    flops = ue.activation_core(M=M, N=N, A_DRAM_ADDR=X, OUTPUT_DRAM_ADDR=OUT,
                               IDENTITY_DRAM_ADDR=IDENT, activation="gelu")
    gelu_us, gelu_snr = _finish(ue, flops, OUT, gelu_ref, "gelu")

    # ---- Snake1D: x + (1 - cos(2ax)) / (2a), cos via degree-6 Taylor ----
    ue = UnifiedEngine()
    X = ue.allocate_tensor_dram(M * N * 2)
    Z2 = ue.allocate_tensor_dram(M * N * 2)     # (2ax)^2
    TMP = ue.allocate_tensor_dram(M * N * 2)
    OUT = ue.allocate_tensor_dram(M * N * 2)
    ue.dma_to_accelerator_memory(X, x.reshape(-1).contiguous())
    ue.start_capture()
    flops = 0
    # z = 2a*x ; Z2 = z*z
    flops += ue.eltwise_core_dram(M, N, X, None, TMP, UE_MODE.MUL_BROADCAST, scalar=2.0 * alpha)
    flops += ue.eltwise_core_dram(M, N, TMP, TMP, Z2, UE_MODE.ELTWISE_MUL)
    # cos(z) = 1 + Z2*(-1/2 + Z2*(1/24 + Z2*(-1/720)))
    flops += ue.eltwise_core_dram(M, N, Z2, None, TMP, UE_MODE.MUL_BROADCAST, scalar=-1.0 / 720.0)
    flops += ue.eltwise_core_dram(M, N, TMP, None, TMP, UE_MODE.ADD_BROADCAST, scalar=1.0 / 24.0)
    flops += ue.eltwise_core_dram(M, N, TMP, Z2, TMP, UE_MODE.ELTWISE_MUL)
    flops += ue.eltwise_core_dram(M, N, TMP, None, TMP, UE_MODE.ADD_BROADCAST, scalar=-1.0 / 2.0)
    flops += ue.eltwise_core_dram(M, N, TMP, Z2, TMP, UE_MODE.ELTWISE_MUL)
    flops += ue.eltwise_core_dram(M, N, TMP, None, TMP, UE_MODE.ADD_BROADCAST, scalar=1.0)
    # snake = x + (1 - cos) / (2a)  =  x + (-1/(2a))*cos + 1/(2a)
    flops += ue.eltwise_core_dram(M, N, TMP, None, TMP, UE_MODE.MUL_BROADCAST, scalar=-1.0 / (2.0 * alpha))
    flops += ue.eltwise_core_dram(M, N, TMP, None, TMP, UE_MODE.ADD_BROADCAST, scalar=1.0 / (2.0 * alpha))
    flops += ue.eltwise_core_dram(M, N, TMP, X, OUT, UE_MODE.ELTWISE_ADD)
    snake_us, snake_snr = _finish(ue, flops, OUT, snake_ref, "snake")

    print(f"[snake_vs_gelu] snake/gelu latency ratio = {snake_us / gelu_us:.2f}x "
          f"(gelu {gelu_us:.2f} us, snake {snake_us:.2f} us; 11 eltwise passes vs 1 activation pass)")
    assert gelu_snr >= 35.0 or gelu_snr == float("inf"), f"gelu SNR {gelu_snr:.2f} dB < 35 dB"
    assert snake_snr >= 35.0 or snake_snr == float("inf"), f"snake SNR {snake_snr:.2f} dB < 35 dB"


# ===========================================================================
# Kokoro ISTFTNet generator (Section 5b-5d) composition proofs.
#
# Every op below already exists; what has never been assembled on hardware is the COMPOSITION
# each of these tests builds. Run them before porting the generator into kokoro_fpga.py.
# ===========================================================================

def _kg_exec(ue, timeout=60.0):
    """stop capture -> halt -> write program -> execute -> wait. Returns instruction bytes."""
    ue.stop_capture()
    ue.generate_instruction_halt()
    prog = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(prog)
    inst_bytes = ue.get_capture_instruction_size_bytes()
    ue.allocate_program_dram(inst_bytes)
    ue.start_execute_from_dram(prog)
    ue.wait_queue(timeout)
    ue.report_timing_and_instruction_count()
    return inst_bytes


def _kg_done(ue):
    ue.clear_capture_buffer(); ue.reset_tensor_dram_addr(); ue.reset_program_dram_addr()


def _kg_tile_row(ue, src_row_dram, dst_dram, n_rows, elems):
    """Broadcast ONE row (elems bf16, contiguous) down n_rows rows of dst: stage it in SRAM once,
    then a hardware loop writes it at every row. Same mechanism as kokoro_fpga._pbi_row_loop."""
    row_bytes = elems * 2
    assert row_bytes % 8 == 0 and dst_dram % 8 == 0 and src_row_dram % 8 == 0
    ue.accelerator_memory_to_sram(accelerator_dram_address=src_row_dram, sram_address=0,
                                  element_size=elems)
    i_reg = ue.alloc_isa_reg()
    t_reg = ue.alloc_isa_reg()
    ue.generate_instruction_add_set(i_reg, 0)
    ue.loop_start(loop_cnt=int(n_rows))
    ue.generate_instruction_reg_mul_imm(t_reg, i_reg, ue_35bit_addr_shifter(row_bytes))
    ue.generate_instruction_add_imm(t_reg, ue_35bit_addr_shifter(dst_dram), t_reg)
    ue.sram_to_accelerator_memory(sram_address=0, accelerator_dram_address=0,
                                  element_size=elems, general_reg_src=t_reg)
    ue.generate_instruction_add_inc(i_reg)
    ue.loop_end()
    ue.release_isa_reg(); ue.release_isa_reg()


def _kg_step(ue, M, N, src, dst, ident, ramp=4096.0):
    """dst = clamp(ramp * src, 0, 1): the soft unit step (reference_transcendentals_via_clamp).
    Issued on a FLAT 64-wide view (activation_core multiplies by a 64x64 identity)."""
    rows = (M * N) // UE_VECTOR_SIZE
    ue.eltwise_core_dram(M, N, src, None, dst, UE_MODE.MUL_BROADCAST, scalar=ramp)
    ue.activation_core(M=rows, N=UE_VECTOR_SIZE, A_DRAM_ADDR=dst, OUTPUT_DRAM_ADDR=dst,
                       IDENTITY_DRAM_ADDR=ident, activation="clamp", clamp_min=0.0, clamp_max=1.0)


def _kg_sin_poly(ue, M, N, r, out, x2, tmp, degree=13):
    """out = sin(r) by an odd Taylor polynomial of the given degree (Horner in r^2), pure eltwise.
    degree 7 covers |r|<=1 (sincos_bounded_poly_test); 13 covers |r|<=pi (first dropped term
    pi^15/15! = 2e-5). Coefficients are the exact 1/(2k+1)! constants -- nothing is precomputed."""
    ue.eltwise_core_dram(M, N, r, r, x2, UE_MODE.ELTWISE_MUL)
    ks = list(range(degree, 0, -2))            # 13, 11, ..., 1
    c = [((-1) ** ((k - 1) // 2)) / math.factorial(k) for k in ks]
    ue.eltwise_core_dram(M, N, x2, None, tmp, UE_MODE.MUL_BROADCAST, scalar=c[0])
    for coef in c[1:]:
        ue.eltwise_core_dram(M, N, tmp, None, tmp, UE_MODE.ADD_BROADCAST, scalar=coef)
        if coef != c[-1]:
            ue.eltwise_core_dram(M, N, tmp, x2, tmp, UE_MODE.ELTWISE_MUL)
    ue.eltwise_core_dram(M, N, tmp, r, out, UE_MODE.ELTWISE_MUL)


def instance_norm_large_T_test(C: int = 128, T_real: int = 30601, chunk: int = 4096,
                               newton_iters: int = 40):
    """AdaIN1d's InstanceNorm at the generator's time axis (T=30601 samples, C=128 channels).

    kokoro_fpga._adain1d cannot be reused here: it normalises in the [C, T] domain and transposes
    BACK with N=T, and bf16_transpose_core caps N at 4032. This composition never transposes back:

      1. per-channel SUM over time = masked-ones row @ x_ct, accumulated over 64-aligned T-chunks
         (each chunk transposed [chunk, C] -> [C, chunk] with N=C, always legal), so the reduction
         axis is the matmul's K and the padding rows are masked to zero exactly;
      2. mean row tiled down T rows (one hardware loop), xc = x - mean, sq = xc*xc;
      3. var row the same way;  rstd = Newton rsqrt  y <- y*(1.5 - 0.5*var*y^2)  from a constant
         seed (no rsqrt/log/exp core needed; converges from any seed 0 < y0 < sqrt(3/var));
      4. out = xc * ((1+gamma)*rstd) + beta, both rows tiled down T.
    Everything runs in the [T, C] layout the convs need, so the generator never needs a
    large-N transpose at all.
    """
    torch.manual_seed(0)
    T_pad = ((T_real + chunk - 1) // chunk) * chunk
    n_chunks = T_pad // chunk
    # Per-channel scale spanning two decades so the Newton rsqrt is exercised over var in
    # [1e-2, 1e2], plus a per-channel offset (of the channel's own scale, like a conv output --
    # an offset >> std would make the bf16 rounding of the MEAN itself the whole error budget).
    ch_scale = torch.logspace(-1, 1, C)
    ch_off = torch.randn(C) * ch_scale
    x = (torch.randn(T_pad, C) * ch_scale + ch_off).to(torch.bfloat16)
    x[T_real:] = 0
    gamma = (torch.randn(C) * 0.5).to(torch.bfloat16)
    beta = (torch.randn(C) * 0.5).to(torch.bfloat16)
    xf = x[:T_real].float()
    mean = xf.mean(0, keepdim=True)
    var = ((xf - mean) ** 2).mean(0, keepdim=True)
    ref = (xf - mean) / torch.sqrt(var) * (1.0 + gamma.float()) + beta.float()
    y0, eps = 1e-2, 1e-6
    assert y0 < math.sqrt(3.0 / (var.max().item() + eps)), "Newton seed outside the convergence basin"

    ue = UnifiedEngine()
    n = T_pad * C
    X = ue.allocate_tensor_dram(n * 2)
    XT = ue.allocate_tensor_dram(C * chunk * 2)            # one transposed chunk, reused
    MASKS = [ue.allocate_tensor_dram(64 * chunk * 2) for _ in range(n_chunks)]
    ACC = [ue.allocate_tensor_dram(64 * C * 2) for _ in range(2)]
    MEANT = ue.allocate_tensor_dram(n * 2)
    XC = ue.allocate_tensor_dram(n * 2)
    SQ = ue.allocate_tensor_dram(n * 2)
    VAR = ue.allocate_tensor_dram(64 * C * 2)
    Y = ue.allocate_tensor_dram(64 * C * 2)
    TMP = ue.allocate_tensor_dram(64 * C * 2)
    G1 = ue.allocate_tensor_dram(64 * C * 2)
    BETA = ue.allocate_tensor_dram(64 * C * 2)
    SCALE = ue.allocate_tensor_dram(64 * C * 2)
    SCALET = ue.allocate_tensor_dram(n * 2)
    BETAT = ue.allocate_tensor_dram(n * 2)
    OUT = ue.allocate_tensor_dram(n * 2)

    ue.dma_to_accelerator_memory(X, x.reshape(-1).contiguous())
    for j in range(n_chunks):
        m = torch.zeros(64, chunk, dtype=torch.bfloat16)
        lo, hi = j * chunk, min((j + 1) * chunk, T_real)
        if hi > lo:
            m[0, :hi - lo] = 1.0 / T_real
        ue.dma_to_accelerator_memory(MASKS[j], m.reshape(-1).contiguous())
    g1 = torch.zeros(64, C, dtype=torch.bfloat16); g1[0] = 1.0 + gamma.float()
    bb = torch.zeros(64, C, dtype=torch.bfloat16); bb[0] = beta
    ue.dma_to_accelerator_memory(G1, g1.reshape(-1).contiguous())
    ue.dma_to_accelerator_memory(BETA, bb.reshape(-1).contiguous())

    def reduce_over_T(src, dst_final):
        """dst row 0 = sum_t mask[t] * src[t, :]  (chunked transpose + K=chunk matmuls)."""
        for j in range(n_chunks):
            ue.bf16_transpose_core(M=chunk, N=C, INPUT_DRAM_ADDR=src + j * chunk * C * 2,
                                   OUTPUT_DRAM_ADDR=XT)
            cur, prev = ACC[j % 2], ACC[(j + 1) % 2]
            ue.matmat_mul_core(M=64, K=chunk, N=C, A_DRAM_ADDR=MASKS[j], B_DRAM_ADDR=XT,
                               OUTPUT_DRAM_ADDR=(dst_final if j == n_chunks - 1 else cur),
                               C_DRAM_ADDR=(prev if j > 0 else None), bias_mode="full_matrix")

    ue.start_capture()
    MEAN = ACC[0] if n_chunks % 2 == 0 else ACC[1]      # any free block; reduce writes dst_final
    MEANB = ue.allocate_tensor_dram(64 * C * 2)
    reduce_over_T(X, MEANB)
    _kg_tile_row(ue, MEANB, MEANT, T_pad, C)
    ue.eltwise_core_dram(T_pad, C, X, MEANT, XC, UE_MODE.ELTWISE_SUB)
    ue.eltwise_core_dram(T_pad, C, XC, XC, SQ, UE_MODE.ELTWISE_MUL)
    reduce_over_T(SQ, VAR)
    ue.eltwise_core_dram(64, C, VAR, None, VAR, UE_MODE.ADD_BROADCAST, scalar=eps)
    # Newton rsqrt from a constant seed: Y = y0 (built as 0*VAR + y0 so it has VAR's shape).
    ue.eltwise_core_dram(64, C, VAR, None, Y, UE_MODE.MUL_BROADCAST, scalar=0.0)
    ue.eltwise_core_dram(64, C, Y, None, Y, UE_MODE.ADD_BROADCAST, scalar=y0)
    total_flops = 0
    for _ in range(newton_iters):
        ue.eltwise_core_dram(64, C, Y, Y, TMP, UE_MODE.ELTWISE_MUL)
        ue.eltwise_core_dram(64, C, TMP, VAR, TMP, UE_MODE.ELTWISE_MUL)
        ue.eltwise_core_dram(64, C, TMP, None, TMP, UE_MODE.MUL_BROADCAST, scalar=-0.5)
        ue.eltwise_core_dram(64, C, TMP, None, TMP, UE_MODE.ADD_BROADCAST, scalar=1.5)
        ue.eltwise_core_dram(64, C, Y, TMP, Y, UE_MODE.ELTWISE_MUL)
    ue.eltwise_core_dram(64, C, Y, G1, SCALE, UE_MODE.ELTWISE_MUL)
    _kg_tile_row(ue, SCALE, SCALET, T_pad, C)
    _kg_tile_row(ue, BETA, BETAT, T_pad, C)
    ue.eltwise_core_dram(T_pad, C, XC, SCALET, OUT, UE_MODE.ELTWISE_MUL)
    total_flops += ue.eltwise_core_dram(T_pad, C, OUT, BETAT, OUT, UE_MODE.ELTWISE_ADD)
    inst_bytes = _kg_exec(ue)

    out = ue.dma_from_accelerator_memory(OUT, (T_pad, C)).float()[:T_real]
    rstd_hw = ue.dma_from_accelerator_memory(Y, (64, C)).float()[0]
    rstd_snr = calculate_snr(1.0 / torch.sqrt(var.reshape(-1)), rstd_hw)
    snr_db = calculate_snr(ref.reshape(-1), out.reshape(-1))
    print(f"[instance_norm_large_T] C={C} T={T_real} (pad {T_pad}, {n_chunks} chunks) "
          f"rstd_SNR={rstd_snr:.2f} dB  out_SNR={snr_db:.2f} dB")
    assert snr_db >= 35.0, f"instance_norm_large_T SNR {snr_db:.2f} dB < 35 dB"
    record_test("instance_norm_large_T", f"C={C},T={T_real}", snr_db=snr_db, inst_bytes=inst_bytes)
    _kg_done(ue)


def sinegen_wrapped_phase_test(T_f: int = 64, UP: int = 300, lanes: int = 9, sr: int = 24000):
    """Kokoro SineGen (SourceModuleHnNSF.l_sin_gen) with the phase kept WRAPPED in [0,1) turns.

    What the reference does (kokoro_test.py SineGen._f02sine, f0 already nearest-upsampled x300):
      rad[r,h] = (f0[r]*(h+1)/sr) mod 1
      P_r = cumsum_r(rad)                                                (unwrapped, turns)
      phase(300r+j) = 300*P_r + (j-149.5)*rad[r+1]   j >= 150            (F.interpolate linear,
                    = 300*P_r + (j-149.5)*rad[r]     j <  150             align_corners=False)
      with the slope clamped to 0 at r=0 (j<150) and r=T_f-1 (j>=150), and sine = sin(2*pi*phase).
    (The 1/300 downsample samples position 300r+149.5, inside a frame where f0 is constant, so the
    reference's rand_ini on sample 0 is dead: its sine part is deterministic. Verified 72 dB
    formula-vs-torch in the host check below.)

    Hardware recipe -- everything eltwise + the clamp step, validated in a bf16 host simulation
    before this test was written (50 dB):
      * every increment is DOUBLE-bf16 (hi + lo, split on host): a single bf16 rad repeated 150
        times is a SYSTEMATIC 2^-9 error per step, which alone caps the result at ~20 dB;
      * every add is a compensated two-sum carrying (p, lo); the wrap is p -= step(p-1) when
        counting up. Counting DOWN the phase is kept as q = p+1 in [1,2), wrapped with q += step(1-q)
        THROUGH the two-sum: near 0 bf16 is dense enough to land inside the soft step's band, and
        a +1 on a [0.5,1)-grid value is not exact in [1,2) -- both silently cost ~10 dB;
      * sin(2*pi*p): fold p to |q| <= 0.25 with two steps, degree-9 Taylor (|arg| <= pi/2, no
        cancellation), then a first-order correction  sin(a+b) ~ sin a + b*cos a  for the lo part.
    Frame accumulator runs over T_f frames sequentially on one 64-lane row; the x300 fine expansion
    is 150 steps up and 149 down with ALL frames in parallel ([T_f, 64] rows). Output layout is
    [UP, T_f, 64] (j-major); the model needs it time-major, a bf16_permute_dram_core interleave.
    """
    torch.manual_seed(0)
    V = UE_VECTOR_SIZE
    assert lanes <= V and UP % 2 == 0
    half = UP // 2
    f0 = 200.0 + 120.0 * torch.sin(torch.linspace(0, 6.0, T_f, dtype=torch.float64))
    harm = torch.arange(1, lanes + 1, dtype=torch.float64)
    rad = (f0[:, None] * harm[None, :] / sr) % 1.0                       # [T_f, lanes] fp64
    P = torch.cumsum(rad, 0)
    rad_up = torch.cat([rad[1:], torch.zeros(1, lanes, dtype=torch.float64)], 0)   # slope for j>=150
    rad_dn = rad.clone(); rad_dn[0] = 0                                             # slope for j<150
    j = torch.arange(UP, dtype=torch.float64)
    slope = torch.where((j >= half)[:, None, None], rad_up[None], rad_dn[None])
    phase = UP * P[None] + (j - (half - 0.5))[:, None, None] * slope
    ref = torch.sin(2 * math.pi * phase)                                   # [UP, T_f, lanes]
    f0_fine = f0.float().repeat_interleave(UP)[None, :, None] * harm.float()[None, None, :]
    rv = (f0_fine / sr) % 1
    rv = F.interpolate(rv.transpose(1, 2), scale_factor=1 / UP, mode="linear").transpose(1, 2)
    ph = torch.cumsum(rv, 1) * 2 * math.pi
    ph = F.interpolate(ph.transpose(1, 2) * UP, scale_factor=UP, mode="linear").transpose(1, 2)
    ref_torch = torch.sin(ph)[0].reshape(T_f, UP, lanes).permute(1, 0, 2)
    print(f"[sinegen] host check, closed-form phase vs torch SineGen: "
          f"{calculate_snr(ref.reshape(-1).float(), ref_torch.reshape(-1)):.2f} dB")

    def split(v):
        """double-bf16: v ~= hi + lo, both bf16, padded into a [T_f, 64] row block."""
        hi = v.to(torch.bfloat16)
        lo = (v - hi.double()).to(torch.bfloat16)
        H = torch.zeros(T_f, V, dtype=torch.bfloat16); H[:, :lanes] = hi
        Lo = torch.zeros(T_f, V, dtype=torch.bfloat16); Lo[:, :lanes] = lo
        return H, Lo
    D_h, D_l = split((UP * rad) % 1.0)             # frame increments, < 1
    U_h, U_l = split(rad_up)                        # fine slope, counting up
    N_h, N_l = split(-rad_dn)                       # fine slope, counting down (negated)

    ue = UnifiedEngine()
    row = V * 2
    blk = T_f * row
    up = lambda t: (lambda a: (ue.dma_to_accelerator_memory(a, t.reshape(-1).contiguous()), a)[1])(
        ue.allocate_tensor_dram(t.numel() * 2))
    DH, DL, UH, UL, NH, NL = [up(t) for t in (D_h, D_l, U_h, U_l, N_h, N_l)]
    IDENT = up(torch.eye(V, dtype=torch.bfloat16))
    ONES = up(torch.ones(T_f, V, dtype=torch.bfloat16))
    PF = ue.allocate_tensor_dram(blk)                 # frame phase hi  [T_f, 64]
    PFL = ue.allocate_tensor_dram(blk)                # frame phase lo
    OUT = ue.allocate_tensor_dram(UP * blk)           # [UP, T_f, 64]
    # frame-accumulator scratch (one row) and fine-expansion scratch (T_f rows)
    fr = {k: ue.allocate_tensor_dram(row) for k in ("p", "lo", "s", "bv", "t1", "t2")}
    fn = {k: ue.allocate_tensor_dram(blk) for k in
          ("p", "lo", "s", "bv", "t1", "t2", "q", "x2", "tm", "sn", "cs", "u", "w", "pc", "pm")}

    def twosum(M, b, d):
        """(b.p, b.lo) += d, compensated. Knuth two-sum, 10 eltwise ops."""
        E = lambda a, bb, o, m: ue.eltwise_core_dram(M, V, a, bb, o, m)
        E(b["p"], d, b["s"], UE_MODE.ELTWISE_ADD)           # s  = p + d
        E(b["s"], b["p"], b["bv"], UE_MODE.ELTWISE_SUB)     # bv = s - p
        E(b["s"], b["bv"], b["t1"], UE_MODE.ELTWISE_SUB)    # t1 = s - bv
        E(b["p"], b["t1"], b["t1"], UE_MODE.ELTWISE_SUB)    # t1 = p - (s - bv)
        E(d, b["bv"], b["t2"], UE_MODE.ELTWISE_SUB)         # t2 = d - bv
        E(b["t1"], b["t2"], b["t1"], UE_MODE.ELTWISE_ADD)   # e  = t1 + t2
        E(b["lo"], b["t1"], b["lo"], UE_MODE.ELTWISE_ADD)   # lo += e
        E(b["s"], b["lo"], b["p"], UE_MODE.ELTWISE_ADD)     # p  = s + lo
        E(b["p"], b["s"], b["t1"], UE_MODE.ELTWISE_SUB)     # t1 = p - s
        E(b["lo"], b["t1"], b["lo"], UE_MODE.ELTWISE_SUB)   # lo -= t1

    def add_lo(M, b, dl):
        ue.eltwise_core_dram(M, V, b["lo"], dl, b["lo"], UE_MODE.ELTWISE_ADD)

    def wrap_up(M, b):
        """p -= step(p - 1)  (p in [0, 2) -> [0, 1); the -1 is exact by Sterbenz)."""
        ue.eltwise_core_dram(M, V, b["p"], None, b["t1"], UE_MODE.ADD_BROADCAST, scalar=-1.0)
        _kg_step(ue, M, V, b["t1"], b["t2"], IDENT)
        ue.eltwise_core_dram(M, V, b["p"], b["t2"], b["p"], UE_MODE.ELTWISE_SUB)

    def wrap_dn(M, b):
        """q += step(1 - q) through the two-sum (q in (0, 2) -> [1, 2))."""
        ue.eltwise_core_dram(M, V, b["p"], None, b["t1"], UE_MODE.MUL_BROADCAST, scalar=-1.0)
        ue.eltwise_core_dram(M, V, b["t1"], None, b["t1"], UE_MODE.ADD_BROADCAST, scalar=1.0)
        _kg_step(ue, M, V, b["t1"], b["q"], IDENT)
        twosum(M, b, b["q"])

    def sin2pi_fold(M, b, src, dst, degree=9):
        """dst = sin(2*pi*src) for src in [0, 1): fold to |q| <= 0.25, then odd Taylor."""
        E = lambda a, bb, o, m, sc=None: ue.eltwise_core_dram(M, V, a, bb, o, m, scalar=sc)
        E(src, None, b["t1"], UE_MODE.ADD_BROADCAST, -0.25); _kg_step(ue, M, V, b["t1"], b["u"], IDENT)  # s1
        E(src, None, b["t1"], UE_MODE.ADD_BROADCAST, -0.75); _kg_step(ue, M, V, b["t1"], b["w"], IDENT)  # s2
        E(src, None, b["t2"], UE_MODE.MUL_BROADCAST, 2.0)                       # t2 = 2p
        E(b["t2"], None, b["t1"], UE_MODE.MUL_BROADCAST, -1.0)
        E(b["t1"], None, b["t1"], UE_MODE.ADD_BROADCAST, 0.5)                   # t1 = 0.5 - 2p
        E(b["t1"], b["u"], b["t1"], UE_MODE.ELTWISE_MUL)                        # * s1
        E(src, b["t1"], b["q"], UE_MODE.ELTWISE_ADD)                            # q = p + s1(0.5-2p)
        E(b["t2"], None, b["t2"], UE_MODE.ADD_BROADCAST, -1.5)                  # t2 = 2p - 1.5
        E(b["t2"], b["w"], b["t2"], UE_MODE.ELTWISE_MUL)                        # * s2
        E(b["q"], b["t2"], b["q"], UE_MODE.ELTWISE_ADD)                         # q += s2(2p-1.5)
        E(b["q"], None, b["q"], UE_MODE.MUL_BROADCAST, 2.0 * math.pi)           # r = 2 pi q
        _kg_sin_poly(ue, M, V, b["q"], dst, b["x2"], b["tm"], degree=degree)

    def emit_sine(M, b, src, out):
        """out = sin(2 pi (src + lo)) ~= sin(2 pi src) + 2 pi lo * cos(2 pi src)."""
        # src must not be one of the fold's scratch buffers (t1, t2, u, w, q): it is p or pm.
        sin2pi_fold(M, b, src, b["sn"])
        ue.eltwise_core_dram(M, V, src, None, b["pc"], UE_MODE.ADD_BROADCAST, scalar=0.25)
        wrap_up(M, {"p": b["pc"], "t1": b["t1"], "t2": b["t2"]})              # (p+0.25) mod 1
        sin2pi_fold(M, b, b["pc"], b["cs"])                                    # cos(2 pi p)
        ue.eltwise_core_dram(M, V, b["lo"], None, b["t1"], UE_MODE.MUL_BROADCAST, scalar=2.0 * math.pi)
        ue.eltwise_core_dram(M, V, b["t1"], b["cs"], b["t1"], UE_MODE.ELTWISE_MUL)
        ue.eltwise_core_dram(M, V, b["sn"], b["t1"], out, UE_MODE.ELTWISE_ADD)

    ue.start_capture()
    # ---- 1. frame accumulator (sequential over frames, one 64-lane row) ----
    ue.eltwise_core_dram(1, V, DH, None, fr["p"], UE_MODE.MUL_BROADCAST, scalar=0.0)
    ue.eltwise_core_dram(1, V, DH, None, fr["lo"], UE_MODE.MUL_BROADCAST, scalar=0.0)
    for r in range(T_f):
        twosum(1, fr, DH + r * row); add_lo(1, fr, DL + r * row); wrap_up(1, fr)
        ue.eltwise_core_dram(1, V, fr["p"], None, PF + r * row, UE_MODE.MUL_BROADCAST, scalar=1.0)
        ue.eltwise_core_dram(1, V, fr["lo"], None, PFL + r * row, UE_MODE.MUL_BROADCAST, scalar=1.0)
    # ---- 2a. fine expansion, counting UP from the frame centre: j = 150 .. 299 ----
    ue.eltwise_core_dram(T_f, V, PF, None, fn["p"], UE_MODE.MUL_BROADCAST, scalar=1.0)
    ue.eltwise_core_dram(T_f, V, PFL, None, fn["lo"], UE_MODE.MUL_BROADCAST, scalar=1.0)
    ue.eltwise_core_dram(T_f, V, UH, None, fn["x2"], UE_MODE.MUL_BROADCAST, scalar=0.5)   # first half step
    ue.eltwise_core_dram(T_f, V, UL, None, fn["tm"], UE_MODE.MUL_BROADCAST, scalar=0.5)
    twosum(T_f, fn, fn["x2"]); add_lo(T_f, fn, fn["tm"]); wrap_up(T_f, fn)
    emit_sine(T_f, fn, fn["p"], OUT + half * blk)
    for jj in range(half + 1, UP):
        twosum(T_f, fn, UH); add_lo(T_f, fn, UL); wrap_up(T_f, fn)
        emit_sine(T_f, fn, fn["p"], OUT + jj * blk)
    # ---- 2b. counting DOWN: j = 149 .. 0, phase held as q = p + 1 in [1, 2) ----
    ue.eltwise_core_dram(T_f, V, PF, None, fn["p"], UE_MODE.MUL_BROADCAST, scalar=1.0)
    ue.eltwise_core_dram(T_f, V, PFL, None, fn["lo"], UE_MODE.MUL_BROADCAST, scalar=1.0)
    twosum(T_f, fn, ONES)                                                      # q = p + 1, exactly
    ue.eltwise_core_dram(T_f, V, NH, None, fn["x2"], UE_MODE.MUL_BROADCAST, scalar=0.5)
    ue.eltwise_core_dram(T_f, V, NL, None, fn["tm"], UE_MODE.MUL_BROADCAST, scalar=0.5)
    twosum(T_f, fn, fn["x2"]); add_lo(T_f, fn, fn["tm"]); wrap_dn(T_f, fn)
    ue.eltwise_core_dram(T_f, V, fn["p"], None, fn["pm"], UE_MODE.ADD_BROADCAST, scalar=-1.0)  # p = q - 1
    emit_sine(T_f, fn, fn["pm"], OUT + (half - 1) * blk)
    for jj in range(half - 2, -1, -1):
        twosum(T_f, fn, NH); add_lo(T_f, fn, NL); wrap_dn(T_f, fn)
        ue.eltwise_core_dram(T_f, V, fn["p"], None, fn["pm"], UE_MODE.ADD_BROADCAST, scalar=-1.0)
        emit_sine(T_f, fn, fn["pm"], OUT + jj * blk)
    inst_bytes = _kg_exec(ue, timeout=300.0)

    out = ue.dma_from_accelerator_memory(OUT, (UP, T_f, V)).float()[:, :, :lanes]
    pf = ue.dma_from_accelerator_memory(PF, (T_f, V)).float()[:, :lanes]
    pfl = ue.dma_from_accelerator_memory(PFL, (T_f, V)).float()[:, :lanes]
    pf_ref = torch.cos(2 * math.pi * ((UP * P) % 1.0)).float()
    ph_snr = calculate_snr(pf_ref.reshape(-1), torch.cos(2 * math.pi * (pf + pfl)).reshape(-1))
    snr_db = calculate_snr(ref.reshape(-1).float(), out.reshape(-1))
    snr_torch = calculate_snr(ref_torch.reshape(-1), out.reshape(-1))
    per_lane = [calculate_snr(ref[:, :, h].reshape(-1).float(), out[:, :, h].reshape(-1)) for h in range(lanes)]
    print(f"[sinegen] T_f={T_f} UP={UP} lanes={lanes}: frame-phase SNR={ph_snr:.2f} dB, "
          f"sine SNR vs closed-form={snr_db:.2f} dB, vs torch SineGen={snr_torch:.2f} dB "
          f"(host bf16 sim of this recipe: 50 dB)")
    print("[sinegen] per-harmonic SNR: " + " ".join(f"{s:.1f}" for s in per_lane))
    assert snr_db >= 30.0, f"sinegen SNR {snr_db:.2f} dB < 30 dB"
    record_test("sinegen_wrapped_phase", f"T_f={T_f},UP={UP},lanes={lanes}", snr_db=snr_db,
                inst_bytes=inst_bytes)
    _kg_done(ue)


def cordic_atan2_magnitude_test(N: int = 4096, iters: int = 8):
    """torch.abs / torch.angle of a complex spectrum (Kokoro's `har = cat([har_spec, har_phase])`)
    via CORDIC vectoring mode, built from eltwise mul/add and the clamp step -- no divide, sqrt or
    atan core. Quadrant fold first (vectoring needs x > 0): (x, y) -> (|x|, y*sign(x)), then
    theta += (x<0) * sign(y) * pi. 8 iterations is the bf16 ceiling; gain K is a host constant.
    """
    torch.manual_seed(0)
    V = UE_VECTOR_SIZE
    M = N // V
    re = (torch.randn(N) * 2.0).to(torch.bfloat16)
    im = (torch.randn(N) * 2.0).to(torch.bfloat16)
    z = torch.complex(re.float(), im.float())
    mag_ref, ang_ref = z.abs(), z.angle()
    K = 1.0
    for k in range(iters):
        K *= math.sqrt(1.0 + 2.0 ** (-2 * k))

    ue = UnifiedEngine()
    n2 = N * 2
    X, Y_, SX, SY, T1, T2, Z, MAG, ANG, IDENT = [ue.allocate_tensor_dram(n2) for _ in range(9)] + \
        [ue.allocate_tensor_dram(V * V * 2)]
    ue.dma_to_accelerator_memory(X, re.reshape(-1).contiguous())
    ue.dma_to_accelerator_memory(Y_, im.reshape(-1).contiguous())
    ue.dma_to_accelerator_memory(IDENT, torch.eye(V, dtype=torch.bfloat16).reshape(-1).contiguous())

    def sign(src, dst):
        _kg_step(ue, M, V, src, dst, IDENT)
        ue.eltwise_core_dram(M, V, dst, None, dst, UE_MODE.MUL_BROADCAST, scalar=2.0)
        ue.eltwise_core_dram(M, V, dst, None, dst, UE_MODE.ADD_BROADCAST, scalar=-1.0)

    ue.start_capture()
    # quadrant fold: sx = sign(x); x *= sx; y *= sx; Z = (1 - step(x)) * sign(y) * pi
    _kg_step(ue, M, V, X, T1, IDENT)                                             # step(x)
    ue.eltwise_core_dram(M, V, T1, None, T1, UE_MODE.MUL_BROADCAST, scalar=-1.0)
    ue.eltwise_core_dram(M, V, T1, None, T1, UE_MODE.ADD_BROADCAST, scalar=1.0)   # 1 - step(x)
    sign(Y_, SY)
    ue.eltwise_core_dram(M, V, T1, SY, Z, UE_MODE.ELTWISE_MUL)
    ue.eltwise_core_dram(M, V, Z, None, Z, UE_MODE.MUL_BROADCAST, scalar=math.pi)  # angle offset
    sign(X, SX)
    ue.eltwise_core_dram(M, V, X, SX, X, UE_MODE.ELTWISE_MUL)
    ue.eltwise_core_dram(M, V, Y_, SX, Y_, UE_MODE.ELTWISE_MUL)
    # vectoring: d = -sign(y); x' = x - d*y*2^-k ; y' = y + d*x*2^-k ; z' = z - d*atan(2^-k)
    for k in range(iters):
        sign(Y_, SY)                                        # SY = sign(y) = -d
        ue.eltwise_core_dram(M, V, Y_, SY, T1, UE_MODE.ELTWISE_MUL)       # T1 = -d*y = |y|
        ue.eltwise_core_dram(M, V, T1, None, T1, UE_MODE.MUL_BROADCAST, scalar=2.0 ** (-k))
        ue.eltwise_core_dram(M, V, X, SY, T2, UE_MODE.ELTWISE_MUL)        # T2 = -d*x
        ue.eltwise_core_dram(M, V, T2, None, T2, UE_MODE.MUL_BROADCAST, scalar=2.0 ** (-k))
        ue.eltwise_core_dram(M, V, X, T1, X, UE_MODE.ELTWISE_ADD)         # x' = x + |y|2^-k
        ue.eltwise_core_dram(M, V, Y_, T2, Y_, UE_MODE.ELTWISE_SUB)       # y' = y - sign(y)*x*2^-k
        ue.eltwise_core_dram(M, V, SY, None, T1, UE_MODE.MUL_BROADCAST, scalar=math.atan(2.0 ** (-k)))
        ue.eltwise_core_dram(M, V, Z, T1, Z, UE_MODE.ELTWISE_ADD)         # z' = z + sign(y)*atan
    total_flops = ue.eltwise_core_dram(M, V, X, None, MAG, UE_MODE.MUL_BROADCAST, scalar=1.0 / K)
    inst_bytes = _kg_exec(ue)

    mag = ue.dma_from_accelerator_memory(MAG, (N,)).float()
    ang = ue.dma_from_accelerator_memory(Z, (N,)).float()
    mag_snr = calculate_snr(mag_ref, mag)
    ang_snr = calculate_snr(ang_ref, ang)
    unit_snr = calculate_snr(torch.cat([torch.cos(ang_ref), torch.sin(ang_ref)]),
                             torch.cat([torch.cos(ang), torch.sin(ang)]))
    print(f"[cordic] N={N} iters={iters}: |z| SNR={mag_snr:.2f} dB, angle SNR={ang_snr:.2f} dB, "
          f"(cos,sin) SNR={unit_snr:.2f} dB")
    assert mag_snr >= 35.0 and unit_snr >= 35.0, f"cordic mag {mag_snr:.2f} / unit {unit_snr:.2f} dB < 35"
    record_test("cordic_atan2_magnitude", f"N={N},iters={iters}", snr_db=min(mag_snr, ang_snr),
                inst_bytes=inst_bytes)
    _kg_done(ue)


def stft_istft_matmul_test(n_blocks: int = 8, n_fft: int = 20, hop: int = 5):
    """Kokoro's TorchSTFT (n_fft=20, hop=5, periodic hann) forward AND inverse as block-Toeplitz
    matmuls -- no framing gather (a 5-sample hop is 10 bytes, not word-aligned, so per-frame row
    copies are out).

    Forward: 64 frames per block. spec[64b:64b+64, :64] = x_pad[320b : 320b+384] @ Bf^T, with
    Bf [4096, 384] holding, for frame i and bin k, hann*cos / -hann*sin at the frame's offset
    5i + n. One M=1 matmul per block; the 4096-wide output row IS 64 rows of the [F, 64] layout
    (re in cols 0..10, im in cols 11..21, zero elsewhere).
    Inverse: y[320b : 320b+320] = spec[64b-2 : 64b+66] @ Bi^T with Bi [320, 4352] carrying the
    iDFT, the synthesis window and the overlap-add (frames 64b-2 .. 64b+65 are exactly the ones
    that touch this block; two zero frames precede spec row 0). Interior envelope sum(w^2) = 1.5
    is folded into Bi; torch.istft's edge correction (first/last 2 frames) is not, so the
    comparison excludes 20 samples at each end.
    """
    torch.manual_seed(0)
    V = UE_VECTOR_SIZE
    FB = 64                                  # frames per block
    L = n_blocks * FB * hop                  # samples
    n_bins = n_fft // 2 + 1                  # 11
    win = torch.hann_window(n_fft, periodic=True, dtype=torch.float64)
    x = (torch.randn(L, dtype=torch.float64) * 0.5).to(torch.bfloat16)
    xf = x.double()
    spec_ref = torch.stft(xf, n_fft, hop, n_fft, window=win, center=True, return_complex=True)  # [11, F]
    F_frames = spec_ref.shape[1]             # L/5 + 1
    nbf = -(-F_frames // FB)                 # forward blocks: one more than the inverse needs
    x_pad = torch.cat([xf[1:n_fft // 2 + 1].flip(0), xf, xf[-n_fft // 2 - 1:-1].flip(0)])  # reflect
    # ---- forward basis: Bf[i*64 + c, n_off] ----
    K_f = 384
    Bf = torch.zeros(FB * V, K_f, dtype=torch.float64)
    n = torch.arange(n_fft, dtype=torch.float64)
    for i in range(FB):
        for k in range(n_bins):
            Bf[i * V + k, hop * i: hop * i + n_fft] = win * torch.cos(2 * math.pi * k * n / n_fft)
            Bf[i * V + n_bins + k, hop * i: hop * i + n_fft] = -win * torch.sin(2 * math.pi * k * n / n_fft)
    # ---- inverse basis: y[m] = sum_frames sum_n  w[n] * idft(spec_f)[n] * [5f - 10 + n == m] / 1.5
    # Row m of Bi (0..319 in the block), column (f_rel*64 + c) for f_rel in 0..67 (frame 64b-2+f_rel).
    # idft(spec)[n] = (1/N) * sum_k c_k * (re_k cos(2 pi k n/N) - im_k sin(2 pi k n/N)),
    # c_k = 1 for k in {0, N/2}, 2 otherwise (one-sided spectrum).
    K_i = 68 * V
    Bi = torch.zeros(FB * hop, K_i, dtype=torch.float64)
    ck = torch.full((n_bins,), 2.0, dtype=torch.float64); ck[0] = 1.0; ck[-1] = 1.0
    for f_rel in range(68):
        f_abs_in_block = f_rel - 2                         # frame index relative to 64b
        for nn_ in range(n_fft):
            m = hop * f_abs_in_block - n_fft // 2 + nn_
            if 0 <= m < FB * hop:
                for k in range(n_bins):
                    g = win[nn_] * ck[k] / n_fft / 1.5
                    Bi[m, f_rel * V + k] = g * math.cos(2 * math.pi * k * nn_ / n_fft)
                    Bi[m, f_rel * V + n_bins + k] = -g * math.sin(2 * math.pi * k * nn_ / n_fft)
    # host sanity of the inverse basis on the exact spectrum
    spec_rows = torch.zeros(2 + nbf * FB, V, dtype=torch.float64)
    spec_rows[2:2 + F_frames, :n_bins] = spec_ref.real.T
    spec_rows[2:2 + F_frames, n_bins:2 * n_bins] = spec_ref.imag.T
    y_host = torch.cat([spec_rows[FB * b: FB * b + 68].reshape(-1) @ Bi.T for b in range(n_blocks)])
    print(f"[stft] host sanity, inverse basis on exact spectrum (interior): "
          f"{calculate_snr(xf[20:-20].float(), y_host[20:-20].float()):.2f} dB")

    ue = UnifiedEngine()
    XP = ue.allocate_tensor_dram((len(x_pad) + K_f) * 2)
    BF = ue.allocate_tensor_dram(FB * V * K_f * 2)
    SPEC = ue.allocate_tensor_dram((2 + nbf * FB + 4) * V * 2)     # 2 zero frames in front
    BI = ue.allocate_tensor_dram(FB * hop * K_i * 2)
    SPEC_EXACT = ue.allocate_tensor_dram((2 + nbf * FB + 4) * V * 2)
    Y = ue.allocate_tensor_dram(n_blocks * FB * hop * 2)
    Y2 = ue.allocate_tensor_dram(n_blocks * FB * hop * 2)
    xp = torch.zeros(len(x_pad) + K_f, dtype=torch.bfloat16); xp[:len(x_pad)] = x_pad.to(torch.bfloat16)
    ue.dma_to_accelerator_memory(XP, xp)
    ue.dma_to_accelerator_memory(BF, Bf.to(torch.bfloat16).reshape(-1).contiguous())
    ue.dma_to_accelerator_memory(BI, Bi.to(torch.bfloat16).reshape(-1).contiguous())
    ue.dma_to_accelerator_memory(SPEC, torch.zeros((2 + nbf * FB + 4) * V, dtype=torch.bfloat16))
    se = torch.zeros(2 + nbf * FB + 4, V, dtype=torch.bfloat16); se[:spec_rows.shape[0]] = spec_rows.to(torch.bfloat16)
    ue.dma_to_accelerator_memory(SPEC_EXACT, se.reshape(-1).contiguous())

    ue.start_capture()
    for b in range(nbf):     # forward: one M=1 matmul per 64-frame block
        ue.matmat_mul_core(M=1, K=K_f, N=FB * V, A_DRAM_ADDR=XP + FB * hop * b * 2, B_DRAM_ADDR=BF,
                           OUTPUT_DRAM_ADDR=SPEC + (2 + FB * b) * V * 2)
    for b in range(n_blocks):     # inverse from the exact spectrum, and from the device's own
        ue.matmat_mul_core(M=1, K=K_i, N=FB * hop, A_DRAM_ADDR=SPEC_EXACT + FB * b * V * 2, B_DRAM_ADDR=BI,
                           OUTPUT_DRAM_ADDR=Y + FB * hop * b * 2)
        ue.matmat_mul_core(M=1, K=K_i, N=FB * hop, A_DRAM_ADDR=SPEC + FB * b * V * 2, B_DRAM_ADDR=BI,
                           OUTPUT_DRAM_ADDR=Y2 + FB * hop * b * 2)
    inst_bytes = _kg_exec(ue)

    spec_hw = ue.dma_from_accelerator_memory(SPEC, (2 + nbf * FB + 4, V)).float()[2:2 + F_frames]
    re_snr = calculate_snr(spec_ref.real.T.float().reshape(-1), spec_hw[:, :n_bins].reshape(-1))
    im_snr = calculate_snr(spec_ref.imag.T.float().reshape(-1), spec_hw[:, n_bins:2 * n_bins].reshape(-1))
    y = ue.dma_from_accelerator_memory(Y, (n_blocks * FB * hop,)).float()
    y2 = ue.dma_from_accelerator_memory(Y2, (n_blocks * FB * hop,)).float()
    inv_snr = calculate_snr(xf[20:-20].float(), y[20:-20])
    rt_snr = calculate_snr(xf[20:-20].float(), y2[20:-20])
    print(f"[stft] L={L} frames={F_frames}: fwd re={re_snr:.2f} dB im={im_snr:.2f} dB | "
          f"istft(exact spec)={inv_snr:.2f} dB | fwd->inv round trip={rt_snr:.2f} dB")
    assert min(re_snr, im_snr) >= 35.0 and inv_snr >= 35.0 and rt_snr >= 30.0, "stft/istft below threshold"
    record_test("stft_istft_matmul", f"L={L},n_fft={n_fft},hop={hop}", snr_db=rt_snr, inst_bytes=inst_bytes)
    _kg_done(ue)


def magic_round_test(N: int = 4096, magic: float = 192.0, p_max: float = 60.0):
    """Integer part by magic rounding on the eltwise unit: n = (p + magic) - magic. With magic=192
    and |p| < 64, p+magic sits in [128, 256) where the bf16 grid is exactly 1.0, so n is an integer
    with |p - n| <= 1. (256 was tried first: its positive half lands on the grid-2 region, and the
    unit TRUNCATES rather than rounds, so |p - n| reached 2 for 65/4096 values.)
    Kokoro's Snake1D uses this to reduce cos(2ax) (operand up to ~15 turns) to one period with
    2 ops instead of a 30-step clamp staircase. Holds under round-to-nearest OR truncation; this
    checks the hardware actually delivers one of those (and not, say, a wider internal format).
    """
    torch.manual_seed(0)
    V = UE_VECTOR_SIZE
    p = (torch.rand(N) * 2 * p_max - p_max).to(torch.bfloat16)
    ue = UnifiedEngine()
    P = ue.allocate_tensor_dram(N * 2); NN = ue.allocate_tensor_dram(N * 2); R = ue.allocate_tensor_dram(N * 2)
    ue.dma_to_accelerator_memory(P, p)
    ue.start_capture()
    ue.eltwise_core_dram(N // V, V, P, None, NN, UE_MODE.ADD_BROADCAST, scalar=magic)
    ue.eltwise_core_dram(N // V, V, NN, None, NN, UE_MODE.ADD_BROADCAST, scalar=-magic)
    ue.eltwise_core_dram(N // V, V, P, NN, R, UE_MODE.ELTWISE_SUB)
    inst_bytes = _kg_exec(ue)
    n = ue.dma_from_accelerator_memory(NN, (N,)).float()
    r = ue.dma_from_accelerator_memory(R, (N,)).float()
    non_int = (n != n.round()).sum().item()
    big = (r.abs() > 1.0).sum().item()
    # the residual must equal p - n exactly (both on a fine grid) so sin(2 pi r) == sin(2 pi p)
    exact = ((p.float() - n) == r).sum().item()
    print(f"[magic_round] N={N} |p|<{p_max}: non-integer n: {non_int}, |p-n|>1: {big}, "
          f"residual exact: {exact}/{N}")
    assert non_int == 0 and big == 0, "magic rounding did not isolate an integer part"
    record_test("magic_round", f"N={N},magic={magic}", snr_db=float("inf") if exact == N else 0.0,
                inst_bytes=inst_bytes)
    _kg_done(ue)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kokoro transcendental & generator-composition hardware proofs")
    parser.add_argument('--dev', type=str, default='xdma0',
                        help='DMA device name (e.g., xdma0, xdma1). Default: xdma0')
    parser.add_argument('--summary-path', type=str, default=None,
                        help='Write a per-test summary here (same format as user_hw_test.py).')
    args = parser.parse_args()

    set_dma_device(args.dev)

    # Same fixed seed as user_hw_test.py, so SNR numbers stay comparable.
    torch.manual_seed(0)

    # --- Transcendentals -------------------------------------------------
    exp_via_sigmoid_test()
    sincos_bounded_poly_test()
    magic_round_test()
    cordic_atan2_magnitude_test()
    snake_vs_gelu_timing_test()

    # --- Generator (Section 5b-5d) compositions --------------------------
    instance_norm_large_T_test()
    sinegen_wrapped_phase_test()
    stft_istft_matmul_test()

    if args.summary_path:
        write_test_summary(args.summary_path)
