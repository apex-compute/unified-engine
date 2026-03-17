"""
Hardware test runner for the Unified Engine.

Runs generic_tests() (memcpy, matmat, transpose, broadcast, layer norm, RMS, RoPE, etc.)
and simple_kq_test() (K/Q projection and K@Q^T attention).

Usage:
    python user_hw_test.py [--dev xdma0] [--cycle 3.0]
"""

import argparse
import math
import os
import random
from re import S
import sys
import time
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from read_trace import generate_trace
import torch

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_C2H,
    DMA_DEVICE_H2C,
    DMA_DEVICE_USER,
    DRAM_ACTIVATION_ADDR,
    DRAM_INSTRUCTION_ADDR,
    TYPE,
    URAM_NEAR_FULL_ELEMENTS,
    calculate_snr,
    set_dma_device,
    UnifiedEngine,
    UE_FMAX_CONTEXT_SIZE,
    UE_VECTOR_SIZE,
)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute frequency tensor for complex exponentials (RoPE/attention)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def matmat_mul_two_engine_flag_check_test(
    M: int,
    K: int,
    N: int,
):
    """
    Shard A along the M dimension and run the two halves on engine0/engine1
    in parallel, both multiplied by the same B.
    """
    import user_dma_core

    assert M % 2 == 0, f"M must be even for two-engine sharding, got {M}"
    M_three_fourth = M * 3 // 4
    M_one_fourth = M // 4

    engine0_base = user_dma_core.UE_0_BASE_ADDR
    engine1_base = user_dma_core.UE_0_BASE_ADDR + 0x00010000
    tensor_region_stride = 0x04000000

    ue0 = UnifiedEngine(BASE_ADDR=engine0_base)
    ue1 = UnifiedEngine(BASE_ADDR=engine1_base)
    ue0._tensor_dram_addr = DRAM_ACTIVATION_ADDR
    ue1._tensor_dram_addr = DRAM_ACTIVATION_ADDR + tensor_region_stride
    ue0._next_program_dram_addr = DRAM_INSTRUCTION_ADDR
    ue1._next_program_dram_addr = DRAM_INSTRUCTION_ADDR + 0x01000000


    e0_a_addr = ue0.allocate_tensor_dram(M_three_fourth * K * 2)
    e0_b_addr = ue0.allocate_tensor_dram(N * K * 2)
    e0_out_addr = ue0.allocate_tensor_dram(M_three_fourth * N * 2)

    e1_a_addr = ue1.allocate_tensor_dram(M_one_fourth * K * 2)
    e1_b_addr = ue1.allocate_tensor_dram(N * K * 2)
    e1_out_addr = ue1.allocate_tensor_dram(M_one_fourth * N * 2)

    ue0.start_capture()
    ue0.generate_instruction_flag_clear()
    ue0.matmat_mul_core(
        M=M_three_fourth, K=K, N=N,
        A_DRAM_ADDR=e0_a_addr, B_DRAM_ADDR=e0_b_addr, OUTPUT_DRAM_ADDR=e0_out_addr
    )
    ue0.generate_instruction_flag_set()
    ue0.stop_capture()
    ue0.generate_instruction_halt()

    e0_prog_addr = ue0.get_program_dram_addr()
    ue0.write_captured_instructions_to_dram(e0_prog_addr)
    ue0.allocate_program_dram(ue0.get_capture_instruction_size_bytes())

    ue1.start_capture()
    ue1.matmat_mul_core(
        M=M_one_fourth, K=K, N=N,
        A_DRAM_ADDR=e1_a_addr, B_DRAM_ADDR=e1_b_addr, OUTPUT_DRAM_ADDR=e1_out_addr
    )
    ue1.generate_instruction_flag_check(target_engine_idx=0)
    ue1.generate_instruction_halt()
    ue1.stop_capture()

    e1_prog_addr = ue1.get_program_dram_addr()
    ue1.write_captured_instructions_to_dram(e1_prog_addr)
    ue1.allocate_program_dram(ue1.get_capture_instruction_size_bytes())

    a = torch.randn(M, K, dtype=torch.bfloat16) / math.sqrt(K)
    b = torch.randn(N, K, dtype=torch.bfloat16)
    a_top = a[:M_three_fourth, :]
    a_bot = a[M_three_fourth:(M_three_fourth + M_one_fourth), :]

    ue0.dma_to_accelerator_memory(e0_a_addr, a_top)
    ue0.dma_to_accelerator_memory(e0_b_addr, b)
    ue1.dma_to_accelerator_memory(e1_a_addr, a_bot)
    ue1.dma_to_accelerator_memory(e1_b_addr, b)
    
    # True sequential scheduling for diagnosis.
    ue0.start_execute_from_dram(e0_prog_addr)
    ue1.start_execute_from_dram(e1_prog_addr)

    ue1.wait_queue(10.0)
    generate_trace(ue0, f"matmat_mul_core_trace_0_{M_three_fourth}_{K}_{N}.csv", clock_period_ns=user_dma_core.CLOCK_CYCLE_TIME_NS)
    generate_trace(ue1, f"matmat_mul_core_trace_1_{M_one_fourth}_{K}_{N}.csv", clock_period_ns=user_dma_core.CLOCK_CYCLE_TIME_NS)

    out_top = ue0.dma_from_accelerator_memory(e0_out_addr, (M_three_fourth, N))
    out_bot = ue1.dma_from_accelerator_memory(e1_out_addr, (M_one_fourth, N))
    out_combined = torch.cat([out_top, out_bot], dim=0)

    ref = a @ b.T

    ref_top = ref[:M_three_fourth, :]
    ref_bot = ref[M_three_fourth:(M_three_fourth + M_one_fourth), :]
    snr_top = calculate_snr(ref_top, out_top)
    snr_bot = calculate_snr(ref_bot, out_bot)
    snr_combined = calculate_snr(ref, out_combined)
    print(f"Parallel sharded matmul SNR top-half:    {snr_top:.2f} dB")
    print(f"Parallel sharded matmul SNR bottom-half: {snr_bot:.2f} dB")
    print(f"Parallel sharded matmul SNR combined: {snr_combined:.2f} dB")

    ue0.reset_tensor_dram_addr()
    ue0.clear_capture_buffer()
    ue1.reset_tensor_dram_addr()
    ue1.clear_capture_buffer()

def matmat_mul_two_cores_test(M: int, K: int, N: int, softmax_enable: bool = False, gelu_enable: bool = False, silu_enable: bool = False):
    """
    Run two-engine matmul via UnifiedEngine.matmat_mul_two_cores().
    Uses balanced 1/2 + 1/2 row sharding.
    """
    import user_dma_core

    assert M >= 2, f"M must be at least 2 for two-core test, got {M}"

    engine0_base = user_dma_core.UE_0_BASE_ADDR
    engine1_base = user_dma_core.UE_0_BASE_ADDR + 0x00010000
    tensor_region_stride = 0x04000000

    ue0 = UnifiedEngine(BASE_ADDR=engine0_base)
    ue1 = UnifiedEngine(BASE_ADDR=engine1_base)
    ue0._tensor_dram_addr = DRAM_ACTIVATION_ADDR
    ue1._tensor_dram_addr = DRAM_ACTIVATION_ADDR + tensor_region_stride
    ue0._next_program_dram_addr = DRAM_INSTRUCTION_ADDR
    ue1._next_program_dram_addr = DRAM_INSTRUCTION_ADDR + 0x01000000

    A_DRAM_ADDR = ue0.allocate_tensor_dram(M * K * 2)
    B_DRAM_ADDR = ue0.allocate_tensor_dram(N * K * 2)
    OUTPUT_DRAM_ADDR = ue0.allocate_tensor_dram(M * N * 2)

    a = torch.randn(M, K, dtype=torch.bfloat16) / math.sqrt(K)
    b = torch.randn(N, K, dtype=torch.bfloat16)

    ue0.dma_to_accelerator_memory(A_DRAM_ADDR, a)
    ue0.dma_to_accelerator_memory(B_DRAM_ADDR, b)

    total_flops_from_matmat_mul = UnifiedEngine.matmat_mul_two_cores(ue0=ue0, ue1=ue1, M=M, K=K, N=N, A_DRAM_ADDR=A_DRAM_ADDR, B_DRAM_ADDR=B_DRAM_ADDR, OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR, softmax_enable=softmax_enable, gelu_enable=gelu_enable, silu_enable=silu_enable)
    ue0.report_timing_and_instruction_count()
    ue1.report_timing_and_instruction_count()

    # Parallel completion time is bounded by the slower engine.
    latency_us = max(ue0.report_latency_in_us(), ue1.report_latency_in_us())
    flop_rate_gflops = total_flops_from_matmat_mul / (latency_us * 1e3)
    print(f"Report FLOPS for two-cores MxKxN Matmul: {flop_rate_gflops:.2f} GFLOPS for M={M}, K={K}, N={N}, softmax_enable={softmax_enable}, gelu_enable={gelu_enable}, silu_enable={silu_enable}")

    generate_trace(ue0, f"matmat_mul_two_cores_trace_engine0_{M // 2}_{K}_{N}_{'softmax_enabled' if softmax_enable else 'softmax_disabled'}_{'gelu_enabled' if gelu_enable else 'gelu_disabled'}_{'silu_enabled' if silu_enable else 'silu_disabled'}.csv", clock_period_ns=user_dma_core.CLOCK_CYCLE_TIME_NS)
    generate_trace(ue1, f"matmat_mul_two_cores_trace_engine1_{M - (M // 2)}_{K}_{N}_{'softmax_enabled' if softmax_enable else 'softmax_disabled'}_{'gelu_enabled' if gelu_enable else 'gelu_disabled'}_{'silu_enabled' if silu_enable else 'silu_disabled'}.csv", clock_period_ns=user_dma_core.CLOCK_CYCLE_TIME_NS)

    output = ue0.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M, N))
    ref = a @ b.T

    def apply_gelu(x):
        return x * torch.sigmoid(1.702 * x)

    def apply_silu(x):
        return x * torch.sigmoid(x)

    if gelu_enable:
        ref = apply_gelu(ref)
    elif silu_enable:
        ref = apply_silu(ref)

    if softmax_enable:
        ref = torch.softmax(ref, dim=-1).to(torch.bfloat16)

    snr_combined = calculate_snr(ref, output)
    print(f"Two-cores matmul SNR combined: {snr_combined:.2f} dB")
    assert snr_combined >= 40 or snr_combined == float('inf'), f"SNR {snr_combined:.2f} dB must be at least 40 dB"

    ue0.reset_tensor_dram_addr()
    ue0.clear_capture_buffer()
    ue1.reset_tensor_dram_addr()
    ue1.clear_capture_buffer()

def flash_attention_test(head_dim: int, seq_len: int, bias_enable: bool = False):
    """
    Tests flash attention core.
    """
    ue = UnifiedEngine()
    debug_mode = False

    Q_DRAM_ADDR = ue.allocate_tensor_dram(seq_len * head_dim * 2)
    K_DRAM_ADDR = ue.allocate_tensor_dram(seq_len * head_dim * 2)
    V_DRAM_ADDR = ue.allocate_tensor_dram(seq_len * head_dim * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(seq_len * head_dim * 2)
    SCRATCH_DRAM_ADDR = ue.allocate_tensor_dram(max(head_dim, UE_FMAX_CONTEXT_SIZE) * seq_len * 2 + head_dim * seq_len * 2) # V_trans + partial softmax output
    BIAS_DRAM_ADDR = ue.allocate_tensor_dram(seq_len * seq_len * 2)
    SM_OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(seq_len * seq_len * 2) if debug_mode else None

    ue.start_capture() # -------------------------------------------------------------

    total_flops_from_flash_attention = ue.flash_attention_core(head_dim=head_dim, seq_len=seq_len,
                                                            Q_DRAM_ADDR=Q_DRAM_ADDR,
                                                            K_DRAM_ADDR=K_DRAM_ADDR,
                                                            V_DRAM_ADDR=V_DRAM_ADDR,
                                                            OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                                                            SCRATCH_DRAM_ADDR=SCRATCH_DRAM_ADDR,
                                                            BIAS_DRAM_ADDR=BIAS_DRAM_ADDR if bias_enable else None,
                                                            debug_mode=debug_mode,
                                                            SM_OUTPUT_DRAM_ADDR=SM_OUTPUT_DRAM_ADDR)

    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    # Test Time
    q = torch.randn(seq_len, head_dim, dtype=torch.bfloat16)
    k = torch.randn(seq_len, head_dim, dtype=torch.bfloat16)
    v = torch.randn(seq_len, head_dim, dtype=torch.bfloat16)
    if bias_enable:
        bias = torch.randn(seq_len, seq_len, dtype=torch.bfloat16)

    # DMA to accelerator memory -------------------------------------------------------------
    ue.dma_to_accelerator_memory(Q_DRAM_ADDR, q)
    ue.dma_to_accelerator_memory(K_DRAM_ADDR, k)
    ue.dma_to_accelerator_memory(V_DRAM_ADDR, v)
    if bias_enable:
        ue.dma_to_accelerator_memory(BIAS_DRAM_ADDR, bias)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(50.0) # 30 seconds timeout
    ue.report_timing_and_instruction_count()

    generate_trace(ue, f"flash_attention_core_trace_{head_dim}_{seq_len}_{'bias_enabled' if bias_enable else 'bias_disabled'}.csv", clock_period_ns=user_dma_core.CLOCK_CYCLE_TIME_NS)

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (seq_len, head_dim))

    if debug_mode:
        sm_output = ue.dma_from_accelerator_memory(SM_OUTPUT_DRAM_ADDR, (seq_len, seq_len))
        v_trans = ue.dma_from_accelerator_memory(SCRATCH_DRAM_ADDR, (head_dim, seq_len))

    report_flop_rate_gflops = ue.report_flop_rate_gflops(total_flops_from_flash_attention)
    print(f"Report FLOPS for Flash Attention: {report_flop_rate_gflops:.2f} GFLOPS for head_dim={head_dim} and seq_len={seq_len} and bias_enable={bias_enable}")

    # Reference calculation (CPU)
    start_time = time.time()
    q_scaled = q * (1 / math.sqrt(head_dim))
    q = q_scaled
    qkt = q_scaled @ k.t()
    if bias_enable:
        qkt = qkt + bias
    sm = torch.softmax(qkt, dim=-1).to(torch.bfloat16)
    ref = sm @ v
    end_time = time.time()
    print(f"Reference Time taken: {(end_time - start_time) * 1000} milliseconds")
    
    if debug_mode:
        snr_db_sm_output = calculate_snr(sm, sm_output)
        print(f"SM Output SNR Analysis: {snr_db_sm_output:.2f} dB")
        assert snr_db_sm_output >= 40 or snr_db_sm_output == float('inf'), f"SNR {snr_db_sm_output:.2f} dB must be at least 40 dB"

        snr_db_v_trans = calculate_snr(v.t(), v_trans)
        print(f"V Trans SNR Analysis: {snr_db_v_trans:.2f} dB")
        assert snr_db_v_trans >= 40 or snr_db_v_trans == float('inf'), f"SNR {snr_db_v_trans:.2f} dB must be at least 40 dB"

    snr_db_ref = calculate_snr(ref, output)
    print(f"Reference SNR Analysis for Flash Attention: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 40 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 40 dB"

    # Before running matmat_mul_core, let's clear the capture buffer and reset the tensor DRAM address
    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

def matmat_mul_test(M: int, K: int, N: int, bias_enable: bool = False, softmax_enable: bool = False, bias_mode: str = "broadcast_N", gelu_enable: bool = False, silu_enable: bool = False):
    """
    Tests matmat_mul core.
    """
    ue = UnifiedEngine()

    A_DRAM_ADDR = ue.allocate_tensor_dram(M * K * 2)
    B_DRAM_ADDR = ue.allocate_tensor_dram(N * K * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)

    C_DRAM_ADDR = None
    if bias_enable and bias_mode == "full_matrix":
        C_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    elif bias_enable and bias_mode == "broadcast_N":
        C_DRAM_ADDR = ue.allocate_tensor_dram(N * 2)

    ue.start_capture()

    total_flops_from_matmat_mul = ue.matmat_mul_core(M=M, K=K, N=N,
                                                    A_DRAM_ADDR=A_DRAM_ADDR,
                                                    B_DRAM_ADDR=B_DRAM_ADDR,
                                                    OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                                                    softmax_enable=softmax_enable,
                                                    C_DRAM_ADDR=C_DRAM_ADDR,
                                                    bias_mode=bias_mode,
                                                    gelu_enable=gelu_enable,
                                                    silu_enable=silu_enable)

    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    # Test Time
    a = torch.randn(M, K, dtype=torch.bfloat16) / math.sqrt(K) # normalizing input helps with numerical stability of softmax
    b = torch.randn(N, K, dtype=torch.bfloat16)

    c = None
    if bias_enable:
        if bias_mode == "full_matrix":
            c = torch.randn(M, N, dtype=torch.bfloat16)
        elif bias_mode == "broadcast_N":
            c = torch.randn(N, dtype=torch.bfloat16)
        ue.dma_to_accelerator_memory(C_DRAM_ADDR, c)

    # DMA to accelerator memory -------------------------------------------------------------
    ue.dma_to_accelerator_memory(A_DRAM_ADDR, a)
    ue.dma_to_accelerator_memory(B_DRAM_ADDR, b)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0) # 10 seconds timeout
    ue.report_timing_and_instruction_count()

    report_flop_rate_gflops = ue.report_flop_rate_gflops(total_flops_from_matmat_mul)
    print(f"Report FLOPS for MxKxN Matmul: {report_flop_rate_gflops:.2f} GFLOPS for M={M}, K={K}, N={N}, bias_enable={bias_enable}, softmax_enable={softmax_enable}, bias_mode={bias_mode}, gelu_enable={gelu_enable}, silu_enable={silu_enable}")

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M, N))

    ref = (a @ b.T + c) if bias_enable else (a @ b.T)

    def apply_gelu(x):
        return x * torch.sigmoid(1.702 * x)

    def apply_silu(x):
        return x * torch.sigmoid(x)

    if gelu_enable:
        ref = apply_gelu(ref)
    elif silu_enable:
        ref = apply_silu(ref)

    if softmax_enable:
        ref = torch.softmax(ref, dim=-1).to(torch.bfloat16)

    snr_db_ref = calculate_snr(ref, output)
    print(f"Reference SNR Analysis for MxKxN Matmul: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 40 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 40 dB"

    generate_trace(ue, f"matmat_mul_core_trace_{M}_{K}_{N}_{'bias_enabled' if bias_enable else 'bias_disabled'}_{'softmax_enabled' if softmax_enable else 'softmax_disabled'}_{'bias_mode_{bias_mode}' if bias_mode else 'bias_mode_none'}_{'gelu_enabled' if gelu_enable else 'gelu_disabled'}_{'silu_enabled' if silu_enable else 'silu_disabled'}.csv", clock_period_ns=user_dma_core.CLOCK_CYCLE_TIME_NS)

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

def rms_norm_test(shape: tuple):
    """
    Tests rms norm core.
    """
    ue = UnifiedEngine()

    assert len(shape) == 2, f"shape must be a tuple of length 2, got {shape}"

    M = shape[0]
    N = shape[1]

    A_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    GAMMA_DRAM_ADDR = ue.allocate_tensor_dram(N * 2)

    ue.start_capture()

    total_flops_from_rms_norm = ue.rms_norm_core_dram(M=M, N=N,
                                                      A_DRAM_ADDR=A_DRAM_ADDR,
                                                      OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                                                      GAMMA_DRAM_ADDR=GAMMA_DRAM_ADDR)

    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())
    ue.clear_capture_buffer()

    # Test Time
    x = torch.randn(M, N, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(A_DRAM_ADDR, x)
    gamma = torch.randn(N, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(GAMMA_DRAM_ADDR, gamma)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0) # 10 seconds timeout
    ue.report_timing_and_instruction_count()

    flop_rate_gflops = ue.report_flop_rate_gflops(total_flops_from_rms_norm)
    print(f"Report FLOPS for RMS Norm: {flop_rate_gflops:.2f} GFLOPS for M={M}, N={N}")

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M, N))

    rms_norm = torch.nn.RMSNorm(N)
    rms_norm.weight.data = gamma
    ref = rms_norm(x)
    snr_db_ref = calculate_snr(ref, output)
    print(f"Reference SNR Analysis for RMS Norm: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 40 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 40 dB"

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

def layer_norm_test(shape: tuple, gamma_enable: bool = False, beta_enable: bool = False):
    """
    Tests layer norm core.
    """
    ue = UnifiedEngine()

    assert len(shape) == 2, f"shape must be a tuple of length 2, got {shape}"

    M = shape[0]
    N = shape[1]

    A_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    GAMMA_DRAM_ADDR = ue.allocate_tensor_dram(N * 2) if gamma_enable else None
    BETA_DRAM_ADDR = ue.allocate_tensor_dram(N * 2) if beta_enable else None

    ue.start_capture()

    total_flops_from_layer_norm = ue.layer_norm_core_dram(M=M, N=N,
                                                          A_DRAM_ADDR=A_DRAM_ADDR,
                                                          OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                                                          GAMMA_DRAM_ADDR=GAMMA_DRAM_ADDR,
                                                          BETA_DRAM_ADDR=BETA_DRAM_ADDR)

    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    # Test Time
    x = torch.randn(M, N, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(A_DRAM_ADDR, x)
    if gamma_enable:
        gamma = torch.randn(N, dtype=torch.bfloat16)
        ue.dma_to_accelerator_memory(GAMMA_DRAM_ADDR, gamma)
    if beta_enable:
        beta = torch.randn(N, dtype=torch.bfloat16)
        ue.dma_to_accelerator_memory(BETA_DRAM_ADDR, beta)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0) # 10 seconds timeout
    ue.report_timing_and_instruction_count()

    generate_trace(ue, f"layer_norm_core_trace_{M}_{N}_{'gamma_enabled' if gamma_enable else 'gamma_disabled'}_{'beta_enabled' if beta_enable else 'beta_disabled'}.csv", clock_period_ns=user_dma_core.CLOCK_CYCLE_TIME_NS)

    flop_rate_gflops = ue.report_flop_rate_gflops(total_flops_from_layer_norm)
    print(f"Report FLOPS for Layer Norm: {flop_rate_gflops:.2f} GFLOPS for M={M}, N={N}, gamma_enable={gamma_enable}, beta_enable={beta_enable}")

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M, N))

    layer_norm = torch.nn.LayerNorm(N)
    if gamma_enable:
        layer_norm.weight.data = gamma
    else:
        layer_norm.weight.data = torch.ones(N, dtype=torch.bfloat16)

    if beta_enable:
        layer_norm.bias.data = beta
    else:
        layer_norm.bias.data = torch.zeros(N, dtype=torch.bfloat16)

    ref = layer_norm(x)
    snr_db_ref = calculate_snr(ref, output)
    print(f"Reference SNR Analysis for Layer Norm: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 40 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 40 dB"

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

def rope_core_dram_test(M: int, N: int):
    """
    Tests rope_core_dram with M rows of N elements using HF-style RoPE.
    """
    ue = UnifiedEngine()

    X_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    COS_DRAM_ADDR = ue.allocate_tensor_dram(N * 2)
    SIN_DRAM_ADDR = ue.allocate_tensor_dram(N * 2)

    ue.start_capture()

    total_flops = ue.rope_core_dram(M=M, N=N,
                                    X_DRAM_ADDR=X_DRAM_ADDR,
                                    OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                                    COS_DRAM_ADDR=COS_DRAM_ADDR,
                                    SIN_DRAM_ADDR=SIN_DRAM_ADDR)

    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    head_dim = N
    MAX_SEQ_LEN = 32768
    freqs_cis = precompute_freqs_cis(head_dim, MAX_SEQ_LEN * 2)
    random_seq_index = random.randint(0, MAX_SEQ_LEN - 1)
    one_rope_seq_params = freqs_cis[random_seq_index, :]
    one_rope_seq = torch.view_as_real(one_rope_seq_params).to(torch.bfloat16).reshape(-1)
    cos = torch.cat((one_rope_seq[0::2], one_rope_seq[0::2]), dim=-1)
    sin = torch.cat((one_rope_seq[1::2], one_rope_seq[1::2]), dim=-1)

    # Negate sin's first half (rope_core_dram expects pre-negated sin)
    sin_negated = sin.clone()
    sin_negated[:N // 2] = -sin_negated[:N // 2]

    x_hf = torch.randn(M, N, dtype=torch.bfloat16)

    ue.dma_to_accelerator_memory(X_DRAM_ADDR, x_hf)
    ue.dma_to_accelerator_memory(COS_DRAM_ADDR, cos)
    ue.dma_to_accelerator_memory(SIN_DRAM_ADDR, sin_negated)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0)
    ue.report_timing_and_instruction_count()

    flop_rate_gflops = ue.report_flop_rate_gflops(total_flops)
    print(f"Report FLOPS for RoPE core DRAM: {flop_rate_gflops:.2f} GFLOPS for M={M}, N={N}")

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M, N))

    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    ref = x_hf * cos + rotate_half(x_hf) * sin

    snr_db = calculate_snr(ref, output)
    print(f"RoPE core DRAM SNR Analysis: {snr_db:.2f} dB for M={M}, N={N}")
    assert snr_db >= 40 or snr_db == float('inf'), f"SNR {snr_db:.2f} dB must be at least 40 dB"

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

def bf16_permute_test(dim_0: int, dim_1: int, dim_2: int):
    """
    Tests bf16_permute_core: permutes (dim_0, dim_1, dim_2) -> (dim_1, dim_0, dim_2).
    """
    ue = UnifiedEngine()

    INPUT_DRAM_ADDR = ue.allocate_tensor_dram(dim_0 * dim_1 * dim_2 * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(dim_1 * dim_0 * dim_2 * 2)

    ue.start_capture()
    ue.bf16_permute_core(dim_0=dim_0, dim_1=dim_1, dim_2=dim_2,
                                       INPUT_DRAM_ADDR=INPUT_DRAM_ADDR,
                                       OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR)
    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    a = torch.randn(dim_0, dim_1, dim_2, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(INPUT_DRAM_ADDR, a)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0)
    ue.report_timing_and_instruction_count()

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (dim_1, dim_0, dim_2))
    ref = a.permute(1, 0, 2)

    snr_db = calculate_snr(ref.flatten(), output.flatten())
    print(f"BF16 Permute core SNR Analysis: {snr_db:.2f} dB")
    assert snr_db >= 40 or snr_db == float('inf'), f"SNR {snr_db:.2f} dB must be at least 40 dB"

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

def patching_test():
    """
    Tests patching_core: extracts 4x4x3 patches from a 3x384x384 image and
    projects through quantized identity-like weight matrices.
    """
    ue = UnifiedEngine()

    C, H, W = 3, 384, 384
    patch_h, patch_w = 4, 4
    K, N = 1024, 64
    block_size = 64
    data_type = TYPE.INT4
    patches_per_group = 16

    # Build 16 identity-like weight matrices (same as user_dma_ops.patching)
    matrix_dram_addrs = []
    scale_dram_addrs = []
    for matrix_idx in range(patches_per_group):
        weight = torch.zeros(N, K, dtype=torch.bfloat16)
        for i in range(48):
            weight[i, matrix_idx * patch_w + i % patch_w + (i // patch_w) * UE_VECTOR_SIZE] = 1.0
        matrix_addr, scale_addr = ue.quantize_weight(weight, N, K, data_type=data_type)
        matrix_dram_addrs.append(matrix_addr)
        scale_dram_addrs.append(scale_addr)

    INPUT_DRAM_ADDR = ue.allocate_tensor_dram(C * H * W * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(96 * 96 * N * 2)

    ue.start_capture()
    ue.patching_core(INPUT_DRAM_ADDR=INPUT_DRAM_ADDR,
                                   OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                                   matrix_dram_addrs=matrix_dram_addrs,
                                   scale_dram_addrs=scale_dram_addrs,
                                   C=C, H=H, W=W,
                                   patch_h=patch_h, patch_w=patch_w,
                                   K=K, N=N, data_type=data_type)
    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    a = torch.randn(C, H, W, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(INPUT_DRAM_ADDR, a)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0)
    ue.report_timing_and_instruction_count()

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (96 * 96, N))

    # Reference: extract patches in (Ph, Pw, C, ph, pw) order, flatten to 48
    ref = a.reshape(C, 96, patch_h, 96, patch_w) \
           .permute(0, 1, 3, 2, 4) \
           .permute(2, 1, 0, 3, 4) \
           .permute(1, 0, 2, 3, 4) \
           .reshape(-1, 48)

    snr_db = calculate_snr(ref[:, :48].flatten(), output[:, :48].flatten())
    print(f"Patching core SNR Analysis: {snr_db:.2f} dB")
    assert snr_db >= 40 or snr_db == float('inf'), f"SNR {snr_db:.2f} dB must be at least 40 dB"

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

def dequantize_test():
    """
    Tests dequantize core.
    """
    ue = UnifiedEngine()

    M = 64
    N = 64
    x = torch.rand(M, N, dtype=torch.bfloat16) * 2 - 1
    QUANTIZED_MATRIX_DRAM_ADDR, SCALE_DRAM_ADDR = ue.quantize_weight(weight=x, N=M, K=N, data_type=TYPE.INT4)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)

    ue.start_capture()
    
    vector_sram_start_addr = 0x00000
    total_flops_from_dequantize = ue.start_queue_for_bf16_dequantize_operation(VECTOR_INPUT_DRAM_ADDR=QUANTIZED_MATRIX_DRAM_ADDR,
                                                SCALE_INPUT_DRAM_ADDR=SCALE_DRAM_ADDR,
                                                data_type=TYPE.INT4,
                                                output_sram_wb_addr=vector_sram_start_addr,
                                                element_size=M * N)

    ue.sram_to_accelerator_memory(sram_address=vector_sram_start_addr, accelerator_dram_address=OUTPUT_DRAM_ADDR, element_size=M * N)

    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0) # 10 seconds timeout
    ue.report_timing_and_instruction_count()

    generate_trace(ue, f"dequantize_core_trace_{M}_{N}.csv", clock_period_ns=user_dma_core.CLOCK_CYCLE_TIME_NS)

    report_flop_rate_gflops = ue.report_flop_rate_gflops(total_flops_from_dequantize)
    print(f"Report FLOPS for Dequantize: {report_flop_rate_gflops:.2f} GFLOPS for M={M}, N={N}")

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M, N))
    # fake quantized matrix is x with block size of 64 and INT4 or FP4
    fake_quantized_matrix = x.reshape(-1, UE_VECTOR_SIZE)
    scales = 7.0 / fake_quantized_matrix.abs().max(dim=-1).values
    scales = scales.unsqueeze(-1)
    quantized_matrix = (fake_quantized_matrix * scales).round().to(torch.int8)
    dequantized_matrix = quantized_matrix / scales
    dequantized_matrix = dequantized_matrix.reshape(M, N)

    snr_db_ref = calculate_snr(dequantized_matrix, output)
    print(f"Reference SNR Analysis for Dequantize: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 30 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 30 dB"

    snr_db_ref = calculate_snr(x, output)
    print(f"Reference SNR Analysis for Reference: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 19 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 19 dB"

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

def matmat_mul_quantized_weights_test(M: int, K: int, N: int, bias_enable: bool = False, bias_mode: str = "broadcast_N", data_type: TYPE = TYPE.INT4, gelu_enable: bool = False, silu_enable: bool = False):
    """
    Tests matrix-matrix multiplication with quantized weights.
    Args:
        M: batch dimension (number of input vectors)
        K: inner dimension (vector length), must be a multiple of UE_VECTOR_SIZE
        N: outer dimension (matrix height), must be a multiple of UE_VECTOR_SIZE
        bias_enable: enable bias
        bias_mode: bias mode
        data_type: data type of the quantized weights, must be one of TYPE.INT4, TYPE.FP4
        gelu_enable: enable gelu activation
        silu_enable: enable silu activation
    """
    ue = UnifiedEngine()

    x = torch.randn(N, K, dtype=torch.bfloat16)
    x = x.reshape(-1, UE_VECTOR_SIZE) 

    out_dim = x.shape[1]

    for i in range(out_dim):
        x[i, :] = torch.randn(UE_VECTOR_SIZE, dtype=torch.bfloat16) * ( i - (out_dim // 2))

    x = x.reshape(N, K)

    QUANTIZED_MATRIX_DRAM_ADDR, SCALE_DRAM_ADDR = ue.quantize_weight(weight=x, N=N, K=K, data_type=data_type)
    A_DRAM_ADDR = ue.allocate_tensor_dram(M * K * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(N * K * 2)

    BIAS_DRAM_ADDR = None
    bias = None
    if bias_enable:
        if bias_mode == "broadcast_N":
            BIAS_DRAM_ADDR = ue.allocate_tensor_dram(N * 2)
            bias = torch.randn(N, dtype=torch.bfloat16)
            ue.dma_to_accelerator_memory(BIAS_DRAM_ADDR, bias)
        elif bias_mode == "full_matrix":
            BIAS_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
            bias = torch.randn(M, N, dtype=torch.bfloat16)
            ue.dma_to_accelerator_memory(BIAS_DRAM_ADDR, bias)
        else:
            assert False, f"bias_mode={bias_mode} is not supported"

    ue.start_capture()
    total_flops_from_dequantize = ue.matmat_mul_core(M=M, K=K, N=N,
                                                    A_DRAM_ADDR=A_DRAM_ADDR,
                                                    B_DRAM_ADDR=QUANTIZED_MATRIX_DRAM_ADDR,
                                                    OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                                                    C_DRAM_ADDR=BIAS_DRAM_ADDR,
                                                    bias_mode=bias_mode,
                                                    is_B_quantized=True,
                                                    data_type=data_type,
                                                    SCALE_DRAM_ADDR=SCALE_DRAM_ADDR,
                                                    gelu_enable=gelu_enable,
                                                    silu_enable=silu_enable)

    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    a = torch.randn(M, K, dtype=torch.bfloat16) # normalizing input helps with numerical stability of softmax
    ue.dma_to_accelerator_memory(A_DRAM_ADDR, a)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0) # 10 seconds timeout
    ue.report_timing_and_instruction_count()

    generate_trace(ue, f"matmat_mul_quantized_weights_core_trace_{M}_{K}_{N}_{data_type}.csv", clock_period_ns=user_dma_core.CLOCK_CYCLE_TIME_NS)

    report_flop_rate_gflops = ue.report_flop_rate_gflops(total_flops_from_dequantize)
    print(f"Report FLOPS for Quantize Matrix-Matrix Multiply: {report_flop_rate_gflops:.2f} GFLOPS for M={M}, N={N}")

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M, N))

    def apply_gelu(x):
        return x * torch.sigmoid(1.702 * x)

    def apply_silu(x):
        return x * torch.sigmoid(x)

    ref = a @ x.T

    if bias_enable:
        ref = ref + bias

    if gelu_enable:
        ref = apply_gelu(ref)
    elif silu_enable:
        ref = apply_silu(ref)

    snr_db_ref = calculate_snr(ref, output)

    print(f"Reference SNR Analysis for Dequantize: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 19 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 40 dB"

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

# TODO: Not very efficient for larger M 
def quantized_matmat_mul_test(M: int, K: int, N: int, data_type: TYPE = TYPE.INT4, bias_enable: bool = False, bias_mode: str = "broadcast_N", gelu_enable: bool = False, silu_enable: bool = False):
    """
    Tests quantized matrix-matrix multiplication core.
    Args:
        M: batch dimension (number of input vectors)
        K: inner dimension (vector length), must be a multiple of UE_VECTOR_SIZE
        N: outer dimension (matrix height), must be a multiple of UE_VECTOR_SIZE
        bias_enable: enable bias
        bias_mode: bias mode
        gelu_enable: enable gelu
        silu_enable: enable silu
    """
    ue = UnifiedEngine()

    x = torch.rand(N, K, dtype=torch.bfloat16) * 2 - 1

    QUANTIZED_MATRIX_DRAM_ADDR, SCALE_DRAM_ADDR = ue.quantize_weight(weight=x, N=N, K=K, data_type=data_type)
    A_DRAM_ADDR = ue.allocate_tensor_dram(M * K * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(N * K * 2)

    C_DRAM_ADDR = None
    if bias_enable and bias_mode == "full_matrix":
        C_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    elif bias_enable and bias_mode == "broadcast_N":
        C_DRAM_ADDR = ue.allocate_tensor_dram(N * 2)

    print(f"Quantized Matrix-Matrix Multiply Test for M={M}, K={K}, N={N}, bias_enable={bias_enable}, bias_mode={bias_mode}, gelu_enable={gelu_enable}, silu_enable={silu_enable}")

    ue.start_capture()
    
    total_flops_from_dequantize = ue.quantized_matmat_core(M=M, K=K, N=N,
                                                    A_DRAM_ADDR=A_DRAM_ADDR,
                                                    B_DRAM_ADDR=QUANTIZED_MATRIX_DRAM_ADDR,
                                                    OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                                                    SCALE_DRAM_ADDR=SCALE_DRAM_ADDR,
                                                    C_DRAM_ADDR=C_DRAM_ADDR,
                                                    bias_mode=bias_mode,
                                                    data_type=data_type,
                                                    gelu_enable=gelu_enable,
                                                    silu_enable=silu_enable)

    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    a = torch.randn(M, K, dtype=torch.bfloat16) # normalizing input helps with numerical stability of softmax
    ue.dma_to_accelerator_memory(A_DRAM_ADDR, a)
    
    if bias_enable:
        if bias_mode == "full_matrix":
            c = torch.randn(M, N, dtype=torch.bfloat16)
        elif bias_mode == "broadcast_N":
            c = torch.randn(N, dtype=torch.bfloat16)
        ue.dma_to_accelerator_memory(C_DRAM_ADDR, c)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0) # 10 seconds timeout
    ue.report_timing_and_instruction_count()

    generate_trace(ue, f"quantized_matmat_mul_core_trace_{M}_{K}_{N}_{'bias_enabled' if bias_enable else 'bias_disabled'}_{'bias_mode_{bias_mode}' if bias_mode else 'bias_mode_none'}_{'gelu_enabled' if gelu_enable else 'gelu_disabled'}_{'silu_enabled' if silu_enable else 'silu_disabled'}.csv", clock_period_ns=user_dma_core.CLOCK_CYCLE_TIME_NS)

    report_flop_rate_gflops = ue.report_flop_rate_gflops(total_flops_from_dequantize)
    print(f"Report FLOPS for Quantize Matrix-Matrix Multiply: {report_flop_rate_gflops:.2f} GFLOPS for M={M}, N={N}")

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M, N))

    def apply_gelu(x):
        return x * torch.sigmoid(1.702 * x)

    def apply_silu(x):
        return x * torch.sigmoid(x)

    ref = (a @ x.T + c) if bias_enable else (a @ x.T)

    if gelu_enable:
        ref = apply_gelu(ref)
    elif silu_enable:
        ref = apply_silu(ref)

    snr_db_ref = calculate_snr(ref, output)

    print(f"Reference SNR Analysis for Dequantize: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 22 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 40 dB"

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

def matmat_mul_non_aligned_writeback_test():
    """
    Tests matmat mul non aligned writeback core.
    """
    ue = UnifiedEngine()

    M = 2
    K = 256
    N = 32

    A_DRAM_ADDR = ue.allocate_tensor_dram(M * K * 2)
    B_DRAM_ADDR = ue.allocate_tensor_dram(N * K * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    
    ue.start_capture()

    ue.accelerator_memory_to_sram(accelerator_dram_address=A_DRAM_ADDR,
                                  sram_address=0x00000,
                                  element_size=M * K)
    ue.accelerator_memory_to_sram(accelerator_dram_address=B_DRAM_ADDR,
                                  sram_address=0x80000,
                                  element_size=N * K)

    # bf16 dot product
    N_aligned = (N + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE * UE_VECTOR_SIZE
    if N < UE_VECTOR_SIZE:
        print(f"Warning: N={N} is less than UE_VECTOR_SIZE={UE_VECTOR_SIZE}, padding to the nearest multiple of UE_VECTOR_SIZE")

    for i in range(M):
        ue.start_queue_for_bf16_matvec_operation(max_clear_en=0,
                                            fmax_context_addr=0,
                                            vector_sram_start_addr=0x00000 + i * K * 2,
                                            matrix_sram_start_addr=0x80000,
                                            output_sram_wb_addr=0xC0000 + i * N_aligned * 2,
                                            K=K,
                                            N=N)

        ue.sram_to_accelerator_memory(sram_address=0xC0000 + i * N_aligned * 2,
                                    accelerator_dram_address=OUTPUT_DRAM_ADDR + i * N * 2,
                                    element_size=N)


    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    a = torch.randn(M, K, dtype=torch.bfloat16) # normalizing input helps with numerical stability of softmax
    ue.dma_to_accelerator_memory(A_DRAM_ADDR, a)
    b = torch.randn(N, K, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(B_DRAM_ADDR, b)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0) # 10 seconds timeout
    ue.report_timing_and_instruction_count()

    generate_trace(ue, f"matmat_mul_non_aligned_writeback_core_trace_{M}_{K}_{N}.csv", clock_period_ns=user_dma_core.CLOCK_CYCLE_TIME_NS)

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M, N))

    snr_db_ref = calculate_snr(a @ b.T, output)
    print(f"Reference SNR Analysis for Matmat Mul Non Aligned Writeback: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 40 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 40 dB"

def mix_of_broadcast_eltwise_add_eltwise_mul_core_test():
    """
    Mix of broadcast, eltwise add, and eltwise mul core.
    """
    ue = UnifiedEngine()

    dim = 8192
    A_DRAM_ADDR = ue.allocate_tensor_dram(dim * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(dim * 2)

    scalar_a = 4.34
    scalar_b = -5.67

    ue.start_capture()

    ue.accelerator_memory_to_sram(accelerator_dram_address=A_DRAM_ADDR,
                                  sram_address=0x10000,
                                  element_size=dim)
    # x + a
    ue.broadcast_add(
        scalar=scalar_a,
        sram_start_addr=0x10000,
        sram_wb_addr=0x20000,
        element_size=dim
    )

    # x * b
    ue.broadcast_mul(
        scalar=scalar_b,
        sram_start_addr=0x10000,
        sram_wb_addr=0x80000,
        element_size=dim
    )

    # (x + a) + (x * b)
    ue.eltwise_add_core(
        vector_A_sram_start_addr=0x20000,
        vector_B_sram_start_addr=0x80000,
        vector_C_sram_wb_addr=0x30000,
        element_size=dim
    )

    # ((x + a) + (x * b)) * (x * b)
    ue.eltwise_mul_core( 
        vector_A_sram_start_addr=0x30000,
        vector_B_sram_start_addr=0x80000,
        vector_C_sram_wb_addr=0x10000,
        element_size=dim
    )

    ue.sram_to_accelerator_memory(sram_address=0x10000,
                                accelerator_dram_address=OUTPUT_DRAM_ADDR,
                                element_size=dim)

    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    # Test Time
    x = torch.randn(dim, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(A_DRAM_ADDR, x)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0) # 10 seconds timeout
    ue.report_timing_and_instruction_count()

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (dim,))
    snr_db_ref = calculate_snr(((x + scalar_a) + (x * scalar_b)) * (x * scalar_b), output)
    print(f"Reference SNR Analysis for Custom Kernel: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 40 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 40 dB"

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

def dram_read_write_speed_test():
    """
    Tests DRAM read speed.
    """
    ue = UnifiedEngine()
    A_DRAM_ADDR = ue.allocate_tensor_dram(URAM_NEAR_FULL_ELEMENTS * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(URAM_NEAR_FULL_ELEMENTS * 2)
    
    ue.start_capture()
    ue.accelerator_memory_to_sram(accelerator_dram_address=A_DRAM_ADDR,
                                  sram_address=0x00000,
                                  element_size=URAM_NEAR_FULL_ELEMENTS)

    ue.stop_capture()
    ue.generate_instruction_halt()

    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    x = torch.randn(URAM_NEAR_FULL_ELEMENTS, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(A_DRAM_ADDR, x)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0) # 10 seconds timeout
    ue.report_timing_and_instruction_count()
    latency_us = ue.report_latency_in_us()
    speed_mbps = URAM_NEAR_FULL_ELEMENTS * 2 / latency_us
    print(f"Read Speed: {speed_mbps:.2f} MB/s")

    ue.clear_capture_buffer()

    # Writeback
    ue.start_capture()
    ue.sram_to_accelerator_memory(sram_address=0x00000,
                                  accelerator_dram_address=OUTPUT_DRAM_ADDR,
                                  element_size=URAM_NEAR_FULL_ELEMENTS)

    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0) # 10 seconds timeout
    ue.report_timing_and_instruction_count()
    latency_us = ue.report_latency_in_us()
    speed_mbps = URAM_NEAR_FULL_ELEMENTS * 2 / latency_us
    print(f"Write Speed: {speed_mbps:.2f} MB/s")

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (URAM_NEAR_FULL_ELEMENTS,))
    snr_db_ref = calculate_snr(x, output)
    print(f"Reference SNR Analysis for DRAM Read Write Speed Test: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 40 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 40 dB"

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

def padding_zero_test():
    """
    Padding zero test.
    """
    ue = UnifiedEngine()
    M = 128
    N = 48
    A_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    
    # capture instructions
    ue.start_capture()

    N_ALIGNED = ((N - 1) // UE_VECTOR_SIZE + 1) * UE_VECTOR_SIZE

    for i in range(M):
        ue.accelerator_memory_to_sram(accelerator_dram_address=A_DRAM_ADDR + i * N * 2,
                                      sram_address=0x00000 + i * N_ALIGNED * 2,
                                      element_size=N)


    ue.sram_to_accelerator_memory(sram_address=0x00000,
                                  accelerator_dram_address=OUTPUT_DRAM_ADDR,
                                  element_size=M * N_ALIGNED)

    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    x = torch.randn(M, N, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(A_DRAM_ADDR, x)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0) # 10 seconds timeout
    ue.report_timing_and_instruction_count()

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M, N_ALIGNED))

    x_padded = torch.zeros(M, N_ALIGNED, dtype=torch.bfloat16)
    x_padded[:, :N] = x
    snr_db_ref = calculate_snr(x_padded, output)
    print(f"Reference SNR Analysis for Padding Zero Test: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 40 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 40 dB"

def slicing_test():
    """
    Slicing test.
    """
    ue = UnifiedEngine()
    M = 5
    N = 64
    A_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    
    # capture instructions
    ue.start_capture()
    ue.accelerator_memory_to_sram(accelerator_dram_address=A_DRAM_ADDR,
                                  sram_address=0x00000,
                                  element_size=M * N)

    aligned_uram_row = ((N - 1) // UE_VECTOR_SIZE + 1) * UE_VECTOR_SIZE
    for i in range(M):
        ue.sram_to_accelerator_memory(sram_address=0x00000 + i * aligned_uram_row * 2,
                                      accelerator_dram_address=OUTPUT_DRAM_ADDR + i * N // 4 * 2,
                                      element_size=N // 4)

    ue.stop_capture()
    ue.generate_instruction_halt()

    x = torch.arange(M * N, dtype=torch.bfloat16).reshape(M, N) + 1
    ue.dma_to_accelerator_memory(A_DRAM_ADDR, x)

    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0) # 10 seconds timeout
    ue.report_timing_and_instruction_count()

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M, N // 4))

    snr_db_ref = calculate_snr(x[:, :N // 4], output)
    print(f"Reference SNR Analysis for Slicing Test: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 40 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 40 dB"

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

def packing_test(packing_mode: int):
    """
    Packing test.
    """
    ue = UnifiedEngine()
    M = 1024
    A_DRAM_ADDR = ue.allocate_tensor_dram(M * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * 2 * packing_mode // UE_VECTOR_SIZE)

    # capture instructions
    ue.start_capture()
    ue.accelerator_memory_to_sram(accelerator_dram_address=A_DRAM_ADDR,
                                  sram_address=0x00000,
                                  element_size=M)

    ue.sram_to_accelerator_memory(sram_address=0x00000,
                                  accelerator_dram_address=OUTPUT_DRAM_ADDR,
                                  element_size=M * packing_mode // UE_VECTOR_SIZE,
                                  stride_bytes_per_chunk=packing_mode * 2,
                                  stride_jump_bytes=packing_mode * 2)

    ue.stop_capture()
    ue.generate_instruction_halt()

    x = torch.arange(M, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(A_DRAM_ADDR, x)

    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0) # 10 seconds timeout
    ue.report_timing_and_instruction_count()

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M * packing_mode // UE_VECTOR_SIZE,))
    ref = x.reshape(-1, UE_VECTOR_SIZE)[:, :packing_mode].flatten()
    snr_db_ref = calculate_snr(ref, output)
    print(f"Reference SNR Analysis for Packing Test: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 40 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 40 dB"

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

def bf16_transpose_test(M: int, N: int):
    """
    Tests bf16 transpose core.
    """
    ue = UnifiedEngine()
    INPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(N * M * 2)

    ue.start_capture()

    ue.bf16_transpose_core(M=M, N=N, INPUT_DRAM_ADDR=INPUT_DRAM_ADDR, OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR)

    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)


    x = torch.randn(M, N, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(INPUT_DRAM_ADDR, x)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0) # 10 seconds timeout
    ue.report_timing_and_instruction_count()

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (N, M))
    snr_db_ref = calculate_snr(x.T, output)

    print(f"Reference SNR Analysis for BF16 Transpose: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 40 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 40 dB"

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

def quantized_fp4_test():
    """
    Tests quantized matrix-matrix multiplication core.
    """
    ue = UnifiedEngine()
    N = 64
    K = 2048
    
    x = torch.randn(N, K, dtype=torch.bfloat16)

    QUANTIZED_MATRIX_DRAM_ADDR, SCALE_DRAM_ADDR = ue.quantize_weight(weight=x, N=N, K=K, data_type=TYPE.FP4)
    A_DRAM_ADDR = ue.allocate_tensor_dram(N * K * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(N * K // 2)

    ue.start_capture()

    ue.accelerator_memory_to_sram(accelerator_dram_address=A_DRAM_ADDR,
                                  sram_address=0x00000,
                                  element_size=N * K)

    ue.start_queue_for_quantize_operation(input_sram_addr=0x00000, output_sram_addr=0x80000, data_type=TYPE.FP4, element_size=N * K)

    ue.sram_to_accelerator_memory(sram_address=0x80000,
                                accelerator_dram_address=OUTPUT_DRAM_ADDR,
                                element_size=N * K // 4)

    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    ue.dma_to_accelerator_memory(A_DRAM_ADDR, x)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0) # 10 seconds timeout
    ue.report_timing_and_instruction_count()
    flop_rate_gflops = ue.report_flop_rate_gflops(N * K * 2)
    print(f"Report FLOPS for Quantized FP4: {flop_rate_gflops:.2f} GFLOPS for N={N}, K={K}")

    generate_trace(ue, f"quantized_fp4_core_trace_{N}_{K}.csv", clock_period_ns=user_dma_core.CLOCK_CYCLE_TIME_NS)

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (1, N * K // 4))
    ref = ue.dma_from_accelerator_memory(QUANTIZED_MATRIX_DRAM_ADDR, (1, N * K// 4))

    out_bytes = output.view(dtype=torch.uint8).flatten()
    ref_bytes = ref.view(dtype=torch.uint8).flatten()
    assert out_bytes.numel() == ref_bytes.numel(), \
        f"Size mismatch: output {out_bytes.numel()} vs ref {ref_bytes.numel()}"

    fp4_to_float = {
        0x0:  0.0, 0x1:  0.5, 0x2:  1.0, 0x3:  1.5,
        0x4:  2.0, 0x5:  3.0, 0x6:  4.0, 0x7:  6.0,
        0x8: -0.0, 0x9: -0.5, 0xA: -1.0, 0xB: -1.5,
        0xC: -2.0, 0xD: -3.0, 0xE: -4.0, 0xF: -6.0,
    }

    num_bytes = out_bytes.numel()
    num_elements = num_bytes * 2
    mismatch_count = 0
    first_mismatches = []
    for i in range(num_bytes):
        ob = int(out_bytes[i].item())
        rb = int(ref_bytes[i].item())
        lo_out, hi_out = ob & 0xF, (ob >> 4) & 0xF
        lo_ref, hi_ref = rb & 0xF, (rb >> 4) & 0xF
        if abs(fp4_to_float[lo_ref] - fp4_to_float[lo_out]) >= 0.5 and abs(lo_ref - lo_out) > 1:
            mismatch_count += 1
            first_mismatches.append(
                f"  elem[{i*2}]: hw=0x{lo_out:X}({fp4_to_float[lo_out]:+g}) "
                f"ref=0x{lo_ref:X}({fp4_to_float[lo_ref]:+g})")
        if abs(fp4_to_float[hi_ref] - fp4_to_float[hi_out]) >= 0.5 and abs(hi_ref - hi_out) > 1:
            mismatch_count += 1
            first_mismatches.append(
                f"  elem[{i*2+1}]: hw=0x{hi_out:X}({fp4_to_float[hi_out]:+g}) "
                f"ref=0x{hi_ref:X}({fp4_to_float[hi_ref]:+g})")

    if mismatch_count == 0:
        print(f"FP4 quantization PASS: all {num_elements} nibbles match")
    else:
        print(f"FP4 quantization FAIL: {mismatch_count}/{num_elements} nibbles differ")
        for m in first_mismatches:
            print(m)

    ue.clear_capture_buffer()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='User DMA Operations for Unified Engine')
    parser.add_argument('--dev', type=str, default='xdma0',
                        help='DMA device name (e.g., xdma0, xdma1). Default: xdma0')
    parser.add_argument('--cycle', type=float, default=5.62,
                        help='Clock cycle time in nanoseconds')
    parser.add_argument('--base-addr', type=lambda x: int(x, 0), default=0x02000000,
                        help='AXI-Lite register base address (default: 0x02000000). '
                             'Try 0x0 if register read hangs after bitstream update.')
    args = parser.parse_args()
    
    # Set DMA device paths based on device name and force-sync every module
    # that may hold a copied device-path global.
    set_dma_device(args.dev)
    import user_dma_core
    DMA_DEVICE_H2C = user_dma_core.DMA_DEVICE_H2C
    DMA_DEVICE_C2H = user_dma_core.DMA_DEVICE_C2H
    DMA_DEVICE_USER = user_dma_core.DMA_DEVICE_USER

    print(f"Using DMA device: {args.dev}")
    print(f"  H2C: {DMA_DEVICE_H2C}")
    print(f"  C2H: {DMA_DEVICE_C2H}")
    print(f"  USER: {DMA_DEVICE_USER}")
    print(f"Using AXI-Lite base addr: 0x{args.base_addr:08x}")
    
    # kintex7 operates at 178 Mhz
    # alveo operates at 300 Mhz
    # kintex ultrascale+ operates at 333 Mhz
    # bittware board operates at 400 Mhz
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle
    print(f"Setting CLOCK_CYCLE_TIME_NS = {user_dma_core.CLOCK_CYCLE_TIME_NS}")
    
    for packing_mode in [16, 32, 48, 64]:
        packing_test(packing_mode=packing_mode)

    padding_zero_test()
    slicing_test()
    quantized_fp4_test()
    dequantize_test()
    matmat_mul_non_aligned_writeback_test()
    rope_core_dram_test(M=4, N=256)
    bf16_permute_test(dim_0=144, dim_1=48, dim_2=64)
    patching_test()
    dram_read_write_speed_test()
    mix_of_broadcast_eltwise_add_eltwise_mul_core_test()
    bf16_transpose_test(M=1024, N=512)
    matmat_mul_test(M=1920, K=768, N=2048)
    matmat_mul_test(M=1920, K=768, N=2048, softmax_enable=True)
    quantized_matmat_mul_test(M=640, K=1280, N=1408, bias_enable=True, bias_mode="broadcast_N", silu_enable=True)
    matmat_mul_quantized_weights_test(M=4032, K=1152, N=640, bias_enable=True, bias_mode="full_matrix")
    flash_attention_test(head_dim=256, seq_len=2048, bias_enable=True)
    rms_norm_test(shape=(768, 1024))
    layer_norm_test(shape=(192, 6912), gamma_enable=True, beta_enable=True)
    matmat_mul_two_engine_flag_check_test(M=256, K=2048, N=1024)
    matmat_mul_two_cores_test(M=1920, K=768, N=2048)
    matmat_mul_two_cores_test(M=1920, K=768, N=2048, softmax_enable=True)
    matmat_mul_two_cores_test(M=1920, K=768, N=2048, gelu_enable=True)
    matmat_mul_two_cores_test(M=1920, K=768, N=2048, silu_enable=True)

    # for M in [64, 192, 4096]:
    #     for N in [64, 576, 1024]:
    #         bf16_transpose_test(M=M, N=N)

    # for M in [64, 384, 1024]:
    #     for N in [64, 576, 1024]:
    #         for K in [64, 192, 1024]:
    #             for bias_enable in [False, True]:
    #                 for bias_mode in ["broadcast_N", "full_matrix"]:
    #                     matmat_mul_quantized_weights_test(M=M, K=K, N=N, bias_enable=bias_enable, bias_mode=bias_mode)
    #                     matmat_mul_quantized_weights_test(M=M, K=K, N=N, bias_enable=bias_enable, bias_mode=bias_mode, gelu_enable=True)
    #                     matmat_mul_quantized_weights_test(M=M, K=K, N=N, bias_enable=bias_enable, bias_mode=bias_mode, silu_enable=True)

    # for M in [64, 384, 1024]:
    #     for N in [64, 576, 1024]:
    #         for K in [64, 192, 1024]:
    #             for bias_enable in [False, True]:
    #                 for bias_mode in ["broadcast_N", "full_matrix"]:
    #                     for softmax_enable in [True, False]:
    #                         matmat_mul_test(M=M, K=K, N=N, bias_enable=bias_enable, bias_mode=bias_mode, softmax_enable=softmax_enable)
    #                         matmat_mul_test(M=M, K=K, N=N, bias_enable=bias_enable, bias_mode=bias_mode, gelu_enable=True)
    #                         matmat_mul_test(M=M, K=K, N=N, bias_enable=bias_enable, bias_mode=bias_mode, silu_enable=True)

    # for bias_enable in [False, True]:
    #     for bias_mode in ["full_matrix", "broadcast_N"]:
    #             quantized_matmat_mul_test(M=1024, K=1024, N=1024, bias_enable=bias_enable, bias_mode=bias_mode)
    #             quantized_matmat_mul_test(M=1024, K=1024, N=1024, bias_enable=bias_enable, bias_mode=bias_mode, gelu_enable=True)
    #             quantized_matmat_mul_test(M=1024, K=1024, N=1024, bias_enable=bias_enable, bias_mode=bias_mode, silu_enable=True)

    # for head_dim in [64, 128, 256]:
    #     for seq_len in [64, 128, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192]:
    #         for bias_enable in [True, False]:
    #             flash_attention_test(head_dim=head_dim, seq_len=seq_len, bias_enable=bias_enable)

    # for M in [64, 128, 256, 512, 1024, 2048, 4096]:
    #     for N in [64, 128, 256, 512, 1024, 2048, 4096]:
    #         for gamma_enable in [True, False]:
    #             for beta_enable in [True, False]:   
    #                 layer_norm_test(shape=(M, N), gamma_enable=gamma_enable, beta_enable=beta_enable)

    # for M in [64, 128, 256, 512, 1024, 2048, 4096]:
    #     for N in [64, 128, 256, 512, 1024, 2048, 4096]:
    #         rms_norm_test(shape=(M, N))

    # for M in [64, 128, 256, 512, 1024, 2048, 4096]:
    #     for K in [64, 128, 256, 512, 1024, 2048, 4032]:
    #         for N in [64, 128, 256, 512, 1024, 2048, 4096]:
    #             for bias_enable in [True, False]:
    #                 for softmax_enable in [True, False]:
    #                     for bias_mode in ["broadcast_N", "full_matrix"]:
    #                         matmat_mul_test(M=M, K=K, N=N, bias_enable=bias_enable, softmax_enable=softmax_enable, bias_mode=bias_mode)
