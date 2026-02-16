"""
Hardware test runner for the Unified Engine.

Runs generic_tests() (memcpy, matmat, transpose, broadcast, layer norm, RMS, RoPE, etc.)
and simple_kq_test() (K/Q projection and K@Q^T attention).

Usage:
    python user_hw_test.py [--dev xdma0] [--cycle 3.0]
"""

import argparse
import math
import time
import torch

from user_dma_core import (
    DMA_DEVICE_C2H,
    DMA_DEVICE_H2C,
    DMA_DEVICE_USER,
    DRAM_ACTIVATION_ADDR,
    DeviceTensor,
    TYPE,
    UE_MODE,
    URAM_NEAR_FULL_ELEMENTS,
    calculate_snr,
    set_dma_device,
    UnifiedEngine,
    UE_FMAX_CONTEXT_SIZE,
)
from user_dma_ops import UnifiedEngine


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute frequency tensor for complex exponentials (RoPE/attention)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def generic_tests():
    """Tests functions"""
    # Initialize Unified Engine
    ue = UnifiedEngine()

    # Memcpy DRAM Readback and Writeback Test =======================================================
    print("\n=== Running Memcpy DRAM Readback and Writeback Test ===")
    v_size = URAM_NEAR_FULL_ELEMENTS
    x = torch.randn(v_size).to(torch.bfloat16)


    memcpy_readback_handler = ue.memcpy_dram_readback_benchmark(input_dram_addr=DRAM_ACTIVATION_ADDR,
                                                                 uram_dst_addr=0x000, # 12-bit each address is 64 elements
                                                                 vector_size=v_size,
                                                                 program_dram_addr=None)
    memcpy_readback_handler(x)

    memcpy_writeback_handler = ue.memcpy_dram_writeback_benchmark(uram_src_addr=0x000, # 12-bit each address is 64 elements
                                                                 output_dram_addr=DRAM_ACTIVATION_ADDR + v_size * 2,
                                                                 vector_size=v_size,
                                                                 program_dram_addr=None)
    result = memcpy_writeback_handler()

    snr_db_memcpy_readback = calculate_snr(x, result.data)
    print(f"Memcpy Readback FPGA SNR Analysis: {snr_db_memcpy_readback:.2f} dB")
    assert snr_db_memcpy_readback >= 40 or snr_db_memcpy_readback == float('inf'), f"SNR {snr_db_memcpy_readback:.2f} dB must be at least 40 dB"
    # MATRIX-MATRIX MULTIPLY EXAMPLE =======================================================
    print("\n=== Running Partitioned Matrix-Matrix Multiply Test ===")
    M_part, K_part, N_part = 144, 192, 192
    batch_size = 384
    x_part = torch.randn(batch_size, M_part, K_part).to(torch.bfloat16)
    w_part = torch.rand(batch_size, N_part, K_part).to(torch.bfloat16)
    bias_part = torch.randn(batch_size, M_part, N_part).to(torch.bfloat16)
    bf16_matmat_part_handler = ue.batched_matmat_mul(M=M_part, K=K_part, N=N_part,
        input_dram_addr_batch=DRAM_ACTIVATION_ADDR,
        weight_dram_addr_batch=DRAM_ACTIVATION_ADDR + batch_size * M_part * K_part * 2,
        output_dram_addr_batch=DRAM_ACTIVATION_ADDR + batch_size * M_part * K_part * 2 + batch_size * K_part * N_part * 2,
        batch_size=batch_size,
        program_dram_addr=None,
        bias_enable=True,
        softmax_enable=True,
        bias_matrix_enable=True,
        bias_dram_addr_batch=DRAM_ACTIVATION_ADDR + batch_size * M_part * K_part * 2 + batch_size * K_part * N_part * 2 + batch_size * M_part * N_part * 2)
    bf16_matmat_part_result = bf16_matmat_part_handler(x_part, w_part, bias=bias_part)
    # Reference: (B, M, K) @ (B, N, K).mT -> (B, M, N) + (B, M, N)
    snr_db_bf16_part = calculate_snr(torch.softmax(x_part @ w_part.transpose(-2, -1) + bias_part, dim=-1), bf16_matmat_part_result)
    print(f"BF16 MatMat (Partitioned) SNR Analysis: {snr_db_bf16_part:.2f} dB")
    assert snr_db_bf16_part >= 34 or snr_db_bf16_part == float('inf'), f"SNR {snr_db_bf16_part:.2f} dB must be at least 40 dB"
    
    # TRANSPOSE EXAMPLE =======================================================
    print("\n=== Running Transpose Test with MatMat Multiply ===")
    M, N = 1024, 1024
    x = torch.randn(M, N).to(torch.bfloat16)

    # Create transpose handler
    transpose_handler = ue.bf16_transpose(M=M, N=N,
                                          input_dram_addr=DRAM_ACTIVATION_ADDR,
                                          output_dram_addr=DRAM_ACTIVATION_ADDR + M * N * 2)

    # Execute transpose
    transposed = transpose_handler(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {transposed.shape}")

    # Compare with reference (accounting for padding)
    ref_transposed = x.T
    snr_db_transpose = calculate_snr(ref_transposed, transposed[:, :M])
    print(f"Transpose SNR Analysis: {snr_db_transpose:.2f} dB")
    assert snr_db_transpose >= 100 or snr_db_transpose == float('inf'), f"SNR {snr_db_transpose:.2f} dB must be at least 100 dB"

    # BROADCAST EXAMPLE =======================================================
    print("\n=== Running Broadcast Test ===")
    BROADCAST_SIZE = 1024 * 1024
    x = torch.randn(BROADCAST_SIZE).to(torch.bfloat16)
    scalar = -0.1235487654321

    broadcast_op = ue.broadcast_op(BROADCAST_SIZE,
                                broadcast_type=UE_MODE.MUL_BROADCAST,
                                scalar=scalar,
                                input_dram_addr=DRAM_ACTIVATION_ADDR,
                                output_dram_addr=DRAM_ACTIVATION_ADDR + BROADCAST_SIZE * 2)

    broadcast_result_fpga = broadcast_op(x)
    broadcast_result_pytorch = x * scalar
    snr_db_broadcast = calculate_snr(broadcast_result_pytorch, broadcast_result_fpga)
    print(f"Broadcast SNR Analysis: {snr_db_broadcast:.2f} dB")
    assert snr_db_broadcast >= 40 or snr_db_broadcast == float('inf'), f"SNR {snr_db_broadcast:.2f} dB must be at least 40 dB"

    # BROADCAST ADD EXAMPLE =======================================================
    print("\n=== Running Broadcast Add Test ===")
    BROADCAST_SIZE = 1024 * 1024
    x = torch.randn(BROADCAST_SIZE).to(torch.bfloat16)
    scalar = 4.1234

    broadcast_add_op = ue.broadcast_op(BROADCAST_SIZE,
                                broadcast_type=UE_MODE.ADD_BROADCAST,
                                scalar=scalar,
                                input_dram_addr=DRAM_ACTIVATION_ADDR,
                                output_dram_addr=DRAM_ACTIVATION_ADDR + BROADCAST_SIZE * 2)

    broadcast_add_result_fpga = broadcast_add_op(x)
    broadcast_add_result_pytorch = x + scalar
    snr_db_broadcast_add = calculate_snr(broadcast_add_result_pytorch, broadcast_add_result_fpga)
    print(f"Broadcast Add SNR Analysis: {snr_db_broadcast_add:.2f} dB")
    assert snr_db_broadcast_add >= 40 or snr_db_broadcast_add == float('inf'), f"SNR {snr_db_broadcast_add:.2f} dB must be at least 40 dB"

    # # MATRIX-MATRIX MULTIPLY EXAMPLE =======================================================
    print("\n=== Running Partitioned Matrix-Matrix Multiply Test ===")
    M_part, K_part, N_part = 1024, 1024, 1024
    x_part = torch.randn(M_part, K_part).to(torch.bfloat16)
    w_part = torch.rand(N_part, K_part).to(torch.bfloat16)

    # init time
    bf16_matmat_part_handler = ue.matmat_mul(M=M_part, K=K_part, N=N_part,
                                                   input_dram_addr=DRAM_ACTIVATION_ADDR,
                                                   weight_dram_addr=DRAM_ACTIVATION_ADDR + M_part * K_part * 2,
                                                   output_dram_addr=DRAM_ACTIVATION_ADDR + M_part * K_part * 2 + K_part * N_part * 2,
                                                   program_dram_addr=None)

    bf16_matmat_part_result = bf16_matmat_part_handler(x_part, w_part)
    snr_db_bf16_part = calculate_snr(x_part @ w_part.t(), bf16_matmat_part_result)
    print(f"BF16 MatMat (Partitioned) SNR Analysis: {snr_db_bf16_part:.2f} dB")
    assert snr_db_bf16_part >= 40 or snr_db_bf16_part == float('inf'), f"SNR {snr_db_bf16_part:.2f} dB must be at least 40 dB"

    print("\n=== Running BF16 Matrix-Matrix Multiply with Bias ===")
    bias_part = torch.randn(N_part).to(torch.bfloat16) * 21.345
    bf16_matmat_part_handler = ue.matmat_mul(M=M_part, K=K_part, N=N_part,
                                                   input_dram_addr=DRAM_ACTIVATION_ADDR,
                                                   weight_dram_addr=DRAM_ACTIVATION_ADDR + M_part * K_part * 2,
                                                   output_dram_addr=DRAM_ACTIVATION_ADDR + M_part * K_part * 2 + K_part * N_part * 2,
                                                   program_dram_addr=None,
                                                   bias_enable=True,
                                                   bias_dram_addr=DRAM_ACTIVATION_ADDR + M_part * K_part * 2 + K_part * N_part * 2 + M_part * N_part * 2)
    bf16_matmat_part_result = bf16_matmat_part_handler(x_part, w_part, bias_part)
    snr_db_bf16_part = calculate_snr(x_part @ w_part.transpose(-2, -1) + bias_part, bf16_matmat_part_result)
    print(f"BF16 MatMat (Partitioned) SNR Analysis: {snr_db_bf16_part:.2f} dB")
    assert snr_db_bf16_part >= 40 or snr_db_bf16_part == float('inf'), f"SNR {snr_db_bf16_part:.2f} dB must be at least 40 dB"

    print("\n=== Running BF16 Matrix-Matrix Multiply with Softmax ===")
    bf16_matmat_softmax_part_handler = ue.matmat_mul(M=M_part, K=K_part, N=N_part,
                                                   input_dram_addr=DRAM_ACTIVATION_ADDR,
                                                   weight_dram_addr=DRAM_ACTIVATION_ADDR + M_part * K_part * 2,
                                                   output_dram_addr=DRAM_ACTIVATION_ADDR + M_part * K_part * 2 + K_part * N_part * 2,
                                                   program_dram_addr=None,
                                                   softmax_enable=True,)

    bf16_matmat_softmax_part_result = bf16_matmat_softmax_part_handler(x_part, w_part)
    result_ref = torch.softmax((x_part @ w_part.t()).bfloat16(), dim=-1).to(torch.bfloat16)
    snr_db_bf16_softmax_part = calculate_snr(result_ref, bf16_matmat_softmax_part_result)
    print(f"BF16 MatMat (Partitioned) SNR Analysis: {snr_db_bf16_softmax_part:.2f} dB")
    print(f"Sum of each row of reference softmax: {result_ref.sum(dim=-1)}")
    print(f"Sum of each row of FPGA softmax: {bf16_matmat_softmax_part_result.sum(dim=-1)}")
    assert snr_db_bf16_softmax_part >= 29 or snr_db_bf16_softmax_part == float('inf'), f"SNR {snr_db_bf16_softmax_part:.2f} dB must be at least 29 dB"

    print("\n=== Running BF16 Matrix-Matrix Multiply with Softmax and Bias ===")
    bf16_matmat_softmax_part_handler = ue.matmat_mul(M=M_part, K=K_part, N=N_part,
                                                   input_dram_addr=DRAM_ACTIVATION_ADDR,
                                                   weight_dram_addr=DRAM_ACTIVATION_ADDR + M_part * K_part * 2,
                                                   output_dram_addr=DRAM_ACTIVATION_ADDR + M_part * K_part * 2 + K_part * N_part * 2,
                                                   program_dram_addr=None,
                                                   softmax_enable=True,
                                                   bias_enable=True,
                                                   bias_dram_addr=DRAM_ACTIVATION_ADDR + M_part * K_part * 2 + K_part * N_part * 2 + M_part * N_part * 2)

    bf16_matmat_softmax_part_result = bf16_matmat_softmax_part_handler(x_part, w_part, bias_part)
    result_ref = torch.softmax((x_part @ w_part.t() + bias_part).bfloat16(), dim=-1).to(torch.bfloat16)
    snr_db_bf16_softmax_part = calculate_snr(result_ref, bf16_matmat_softmax_part_result)
    print(f"BF16 MatMat (Partitioned) SNR Analysis: {snr_db_bf16_softmax_part:.2f} dB")
    print(f"Sum of each row of reference softmax: {result_ref.sum(dim=-1)}")
    print(f"Sum of each row of FPGA softmax: {bf16_matmat_softmax_part_result.sum(dim=-1)}")
    assert snr_db_bf16_softmax_part >= 30 or snr_db_bf16_softmax_part == float('inf'), f"SNR {snr_db_bf16_softmax_part:.2f} dB must be at least 30 dB"

    print("\n=== Running BF16 Matrix-Matrix Multiply with Bias matrix (M x N)===")
    bf16_matmat_bias_matrix_part_handler = ue.matmat_mul(M=M_part, K=K_part, N=N_part,
                                                   input_dram_addr=DRAM_ACTIVATION_ADDR,
                                                   weight_dram_addr=DRAM_ACTIVATION_ADDR + M_part * K_part * 2,
                                                   output_dram_addr=DRAM_ACTIVATION_ADDR + M_part * K_part * 2 + K_part * N_part * 2,
                                                   bias_enable=True,
                                                   bias_matrix_enable=True,
                                                   bias_dram_addr=DRAM_ACTIVATION_ADDR + M_part * K_part * 2 + K_part * N_part * 2 + M_part * N_part * 2)

    # bias matrix is a random matrix
    bias_matrix_part = torch.randn(M_part, N_part).to(torch.bfloat16)
    bf16_matmat_bias_matrix_part_result = bf16_matmat_bias_matrix_part_handler(x_part, w_part, bias_matrix_part)
    result_ref = (x_part @ w_part.t() + bias_matrix_part).bfloat16()
    snr_db_bf16_bias_matrix_part = calculate_snr(result_ref, bf16_matmat_bias_matrix_part_result)
    print(f"BF16 MatMat (Partitioned) + Bias matrix SNR Analysis: {snr_db_bf16_bias_matrix_part:.2f} dB")
    assert snr_db_bf16_bias_matrix_part >= 40 or snr_db_bf16_bias_matrix_part == float('inf'), f"SNR {snr_db_bf16_bias_matrix_part:.2f} dB must be at least 30 dB"

    print("\n=== Running BF16 Matrix-Matrix Multiply with softmax and Bias matrix (M x N)===")
    bf16_matmat_softmax_bias_matrix_part_handler = ue.matmat_mul(M=M_part, K=K_part, N=N_part,
                                                   input_dram_addr=DRAM_ACTIVATION_ADDR,
                                                   weight_dram_addr=DRAM_ACTIVATION_ADDR + M_part * K_part * 2,
                                                   output_dram_addr=DRAM_ACTIVATION_ADDR + M_part * K_part * 2 + K_part * N_part * 2,
                                                   softmax_enable=True,
                                                   bias_enable=True,
                                                   bias_matrix_enable=True,
                                                   bias_dram_addr=DRAM_ACTIVATION_ADDR + M_part * K_part * 2 + K_part * N_part * 2 + M_part * N_part * 2)


    bf16_matmat_softmax_bias_matrix_part_result = bf16_matmat_softmax_bias_matrix_part_handler(x_part, w_part, bias_matrix_part)
    result_ref = torch.softmax((x_part @ w_part.t() + bias_matrix_part).bfloat16(), dim=-1).to(torch.bfloat16)
    snr_db_bf16_softmax_bias_matrix_part = calculate_snr(result_ref, bf16_matmat_softmax_bias_matrix_part_result)
    print(f"BF16 MatMat (Partitioned) + Softmax + Bias matrix SNR Analysis: {snr_db_bf16_softmax_bias_matrix_part:.2f} dB")
    print(f"Sum of each row of reference softmax: {result_ref.sum(dim=-1)}")
    print(f"Sum of each row of FPGA softmax: {bf16_matmat_softmax_bias_matrix_part_result.sum(dim=-1)}")
    assert snr_db_bf16_softmax_bias_matrix_part >= 29 or snr_db_bf16_softmax_bias_matrix_part == float('inf'), f"SNR {snr_db_bf16_softmax_bias_matrix_part:.2f} dB must be at least 29 dB"

    print("\n=== Running BF16 Matrix-Matrix Multiply with GELU ===")
    bf16_matmat_gelu_part_handler = ue.matmat_mul(M=M_part, K=K_part, N=N_part,
                                                   input_dram_addr=DRAM_ACTIVATION_ADDR,
                                                   weight_dram_addr=DRAM_ACTIVATION_ADDR + M_part * K_part * 2,
                                                   output_dram_addr=DRAM_ACTIVATION_ADDR + M_part * K_part * 2 + K_part * N_part * 2,
                                                   program_dram_addr=None,
                                                   gelu_enable=True)

    bf16_matmat_gelu_part_result = bf16_matmat_gelu_part_handler(x_part, w_part)
    result_ref = torch.nn.functional.gelu((x_part @ w_part.t()).bfloat16())
    snr_db_bf16_gelu_part = calculate_snr(result_ref, bf16_matmat_gelu_part_result)
    print(f"BF16 MatMat (Partitioned) SNR Analysis: {snr_db_bf16_gelu_part:.2f} dB")
    assert snr_db_bf16_gelu_part >= 40 or snr_db_bf16_gelu_part == float('inf'), f"SNR {snr_db_bf16_gelu_part:.2f} dB must be at least 40 dB"

    print("\n=== Running BF16 Matrix-Matrix Multiply with SILU ===")
    bf16_matmat_silu_part_handler = ue.matmat_mul(M=M_part, K=K_part, N=N_part,
                                                   input_dram_addr=DRAM_ACTIVATION_ADDR,
                                                   weight_dram_addr=DRAM_ACTIVATION_ADDR + M_part * K_part * 2,
                                                   output_dram_addr=DRAM_ACTIVATION_ADDR + M_part * K_part * 2 + K_part * N_part * 2,
                                                   program_dram_addr=None,
                                                   silu_enable=True)

    bf16_matmat_silu_part_result = bf16_matmat_silu_part_handler(x_part, w_part)
    result_ref = torch.nn.functional.silu((x_part @ w_part.t()).bfloat16())
    snr_db_bf16_silu_part = calculate_snr(result_ref, bf16_matmat_silu_part_result)
    print(f"BF16 MatMat (Partitioned) SNR Analysis: {snr_db_bf16_silu_part:.2f} dB")
    assert snr_db_bf16_silu_part >= 40 or snr_db_bf16_silu_part == float('inf'), f"SNR {snr_db_bf16_silu_part:.2f} dB must be at least 40 dB"

    print("\n=== Running BF16 Matrix-Matrix Multiply with GELU and Bias ===")
    bias_part = torch.randn(N_part).to(torch.bfloat16) * 21.345
    bf16_matmat_gelu_part_handler = ue.matmat_mul(M=M_part, K=K_part, N=N_part,
                                                   input_dram_addr=DRAM_ACTIVATION_ADDR,
                                                   weight_dram_addr=DRAM_ACTIVATION_ADDR + M_part * K_part * 2,
                                                   output_dram_addr=DRAM_ACTIVATION_ADDR + M_part * K_part * 2 + K_part * N_part * 2,
                                                   program_dram_addr=None,
                                                   bias_enable=True,
                                                   bias_dram_addr=DRAM_ACTIVATION_ADDR + M_part * K_part * 2 + K_part * N_part * 2 + M_part * N_part * 2,
                                                   gelu_enable=True)

    bf16_matmat_gelu_part_result = bf16_matmat_gelu_part_handler(x_part, w_part, bias_part)
    result_ref = torch.nn.functional.gelu((x_part @ w_part.t() + bias_part).bfloat16())
    snr_db_bf16_gelu_part = calculate_snr(result_ref, bf16_matmat_gelu_part_result)
    print(f"BF16 MatMat (Partitioned) SNR Analysis: {snr_db_bf16_gelu_part:.2f} dB")
    assert snr_db_bf16_gelu_part >= 40 or snr_db_bf16_gelu_part == float('inf'), f"SNR {snr_db_bf16_gelu_part:.2f} dB must be at least 40 dB"

    print("\n=== Running BF16 Matrix-Matrix Multiply with SILU and Bias ===")
    bias_part = torch.randn(N_part).to(torch.bfloat16) * 21.345
    bf16_matmat_silu_part_handler = ue.matmat_mul(M=M_part, K=K_part, N=N_part,
                                                   input_dram_addr=DRAM_ACTIVATION_ADDR,
                                                   weight_dram_addr=DRAM_ACTIVATION_ADDR + M_part * K_part * 2,
                                                   output_dram_addr=DRAM_ACTIVATION_ADDR + M_part * K_part * 2 + K_part * N_part * 2,
                                                   program_dram_addr=None,
                                                   bias_enable=True,
                                                   bias_dram_addr=DRAM_ACTIVATION_ADDR + M_part * K_part * 2 + K_part * N_part * 2 + M_part * N_part * 2,
                                                   silu_enable=True)

    bf16_matmat_silu_part_result = bf16_matmat_silu_part_handler(x_part, w_part, bias_part)
    result_ref = torch.nn.functional.silu((x_part @ w_part.t() + bias_part).bfloat16())
    snr_db_bf16_silu_part = calculate_snr(result_ref, bf16_matmat_silu_part_result)
    print(f"BF16 MatMat (Partitioned) SNR Analysis: {snr_db_bf16_silu_part:.2f} dB")
    assert snr_db_bf16_silu_part >= 40 or snr_db_bf16_silu_part == float('inf'), f"SNR {snr_db_bf16_silu_part:.2f} dB must be at least 40 dB"

    # ELTWISE ADD EXAMPLE =======================================================
    print("\n=== Running Eltwise Add ===")
    dim = 1024 * 1024
    eltwise_add_input_1 = (torch.randn(dim) * 1.25 - 3.0).to(torch.bfloat16)
    eltwise_add_input_2 = (torch.randn(dim) * 0.75 - 2.1).to(torch.bfloat16)
    eltwise_add_1 = ue.eltwise_add(dim, DRAM_ACTIVATION_ADDR, DRAM_ACTIVATION_ADDR + dim * 2, DRAM_ACTIVATION_ADDR + dim * 2 * 2)
    eltwise_result_device = eltwise_add_1(eltwise_add_input_1, eltwise_add_input_2)
    eltwise_result_ref = eltwise_add_input_1 + eltwise_add_input_2
    snr_db_eltwise_add = calculate_snr(eltwise_result_ref, eltwise_result_device)
    print(f"Eltwise Add SNR Analysis: {snr_db_eltwise_add:.2f} dB")
    assert snr_db_eltwise_add >= 40 or snr_db_eltwise_add == float('inf'), f"SNR {snr_db_eltwise_add:.2f} dB must be at least 40 dB"

    # ELTWISE MUL EXAMPLE =======================================================
    print("\n=== Running Eltwise Mul ===")
    dim = 1024 * 1024
    eltwise_mul_input_1 = (torch.randn(dim) * 22.0 - 7.0).to(torch.bfloat16)
    eltwise_mul_input_2 = (torch.randn(dim) * 0.75 - 2.1).to(torch.bfloat16)
    eltwise_mul_1 = ue.eltwise_mul(dim, DRAM_ACTIVATION_ADDR, DRAM_ACTIVATION_ADDR + dim * 2, DRAM_ACTIVATION_ADDR + dim * 2 * 2)
    eltwise_result_device = eltwise_mul_1(eltwise_mul_input_1, eltwise_mul_input_2)
    eltwise_result_ref = eltwise_mul_input_1 * eltwise_mul_input_2
    snr_db_eltwise_mul = calculate_snr(eltwise_result_ref, eltwise_result_device)
    print(f"Eltwise Mul SNR Analysis: {snr_db_eltwise_mul:.2f} dB")
    assert snr_db_eltwise_mul >= 40 or snr_db_eltwise_mul == float('inf'), f"SNR {snr_db_eltwise_mul:.2f} dB must be at least 40 dB"

    # LAYER NORMALIZATION EXAMPLE =======================================================
    print("\n=== Running Layer Normalization ===")
    norm_dim = 1024
    norm_dim_2 = 1024
    batch_size = 1024
    batch_size_2 = 1024
    layer_norm_input = (torch.randn(batch_size, norm_dim) * 1.5 - 5.0).to(torch.bfloat16)
    layer_norm_input_2 = (torch.randn(batch_size_2, norm_dim_2) * 1.5 - 5.0).to(torch.bfloat16)

    gamma = torch.randn(norm_dim, dtype=torch.bfloat16)
    beta = torch.randn(norm_dim, dtype=torch.bfloat16)

    gamma_2 = torch.randn(norm_dim_2, dtype=torch.bfloat16)
    beta_2 = torch.randn(norm_dim_2, dtype=torch.bfloat16)

    layer_norm1 = ue.layer_norm([batch_size, norm_dim], input_dram_addr=DRAM_ACTIVATION_ADDR,
                                                        output_dram_addr=DRAM_ACTIVATION_ADDR + batch_size * norm_dim * 2,
                                                        program_dram_addr=None,
                                                        params_dram_addr=None,
                                                        gamma=gamma,
                                                        beta=beta)

    layer_norm2 = ue.layer_norm([batch_size_2, norm_dim_2], input_dram_addr=DRAM_ACTIVATION_ADDR,
                                                            output_dram_addr=DRAM_ACTIVATION_ADDR + batch_size_2 * norm_dim_2 * 2,
                                                            program_dram_addr=None,
                                                            params_dram_addr=None,
                                                            gamma=gamma_2,
                                                            beta=beta_2)

    layer_norm_result_device = layer_norm1(layer_norm_input)
    layer_norm_result_device_2 = layer_norm2(layer_norm_input_2)
    layer_norm_result_device_3 = layer_norm1(layer_norm_input*0.1 + 1.0)
    layer_norm_result_device_4 = layer_norm2(layer_norm_input_2*1.1 + 1.0)

    layer_norm = torch.nn.LayerNorm(norm_dim)
    layer_norm.weight.data = gamma
    layer_norm.bias.data = beta
    layer_norm_2 = torch.nn.LayerNorm(norm_dim_2)
    layer_norm_2.weight.data = gamma_2
    layer_norm_2.bias.data = beta_2

    start_time = time.perf_counter()
    ref_layer_norm = layer_norm(layer_norm_input)
    ref_layer_norm_2 = layer_norm_2(layer_norm_input_2)
    ref_layer_norm_3 = layer_norm(layer_norm_input*0.1 + 1.0)
    ref_layer_norm_4 = layer_norm_2(layer_norm_input_2*1.1 + 1.0)
    end_time = time.perf_counter()

    execution_time = (end_time - start_time) * 1e6  # Convert to microseconds
    print(f"Reference layer_norm execution time: {execution_time:.3f} us")

    snr_db_layer_norm = calculate_snr(ref_layer_norm, layer_norm_result_device)
    snr_db_layer_norm_2 = calculate_snr(ref_layer_norm_2, layer_norm_result_device_2)
    snr_db_layer_norm_3 = calculate_snr(ref_layer_norm_3, layer_norm_result_device_3)
    snr_db_layer_norm_4 = calculate_snr(ref_layer_norm_4, layer_norm_result_device_4)
    print(f"Layer Normalization SNR Analysis 1: {snr_db_layer_norm:.2f} dB")
    print(f"Layer Normalization SNR Analysis 2: {snr_db_layer_norm_2:.2f} dB")
    print(f"Layer Normalization SNR Analysis 3: {snr_db_layer_norm_3:.2f} dB")
    print(f"Layer Normalization SNR Analysis 4: {snr_db_layer_norm_4:.2f} dB")
    assert snr_db_layer_norm >= 40 or snr_db_layer_norm == float('inf'), f"SNR {snr_db_layer_norm:.2f} dB must be at least 40 dB"
    assert snr_db_layer_norm_2 >= 40 or snr_db_layer_norm_2 == float('inf'), f"SNR {snr_db_layer_norm_2:.2f} dB must be at least 40 dB"
    assert snr_db_layer_norm_3 >= 40 or snr_db_layer_norm_3 == float('inf'), f"SNR {snr_db_layer_norm_3:.2f} dB must be at least 40 dB"
    assert snr_db_layer_norm_4 >= 40 or snr_db_layer_norm_4 == float('inf'), f"SNR {snr_db_layer_norm_4:.2f} dB must be at least 40 dB"

    # RMS NORM EXAMPLE =======================================================
    print("\n=== Running RMS Norm ===")
    rms_norm_input = (torch.randn(batch_size, norm_dim) * 1.5 - 5.0).to(torch.bfloat16)
    rms_norm_input_2 = (torch.randn(batch_size_2, norm_dim_2) * 1.5 - 5.0).to(torch.bfloat16)
    rms_norm1 = ue.rms_norm([batch_size, norm_dim], input_dram_addr=DRAM_ACTIVATION_ADDR,
                                                        output_dram_addr=DRAM_ACTIVATION_ADDR + batch_size * norm_dim * 2,
                                                        program_dram_addr=None,
                                                        params_dram_addr=None,
                                                        gamma=gamma)
    rms_norm2 = ue.rms_norm([batch_size_2, norm_dim_2], input_dram_addr=DRAM_ACTIVATION_ADDR,
                                                        output_dram_addr=DRAM_ACTIVATION_ADDR + batch_size_2 * norm_dim_2 * 2,
                                                        program_dram_addr=None,
                                                        params_dram_addr=None,
                                                        gamma=gamma_2)
    rms_norm_result_device = rms_norm1(rms_norm_input)
    rms_norm_result_device_2 = rms_norm2(rms_norm_input_2)
    rms_norm_result_device_3 = rms_norm1(rms_norm_input*0.1 + 1.0)
    rms_norm_result_device_4 = rms_norm2(rms_norm_input_2*1.1 + 1.0)

    rms_norm = torch.nn.RMSNorm(norm_dim)
    rms_norm.weight.data = gamma
    rms_norm_2 = torch.nn.RMSNorm(norm_dim_2)
    rms_norm_2.weight.data = gamma_2
    rms_norm_result_ref = rms_norm(rms_norm_input)
    rms_norm_result_ref_2 = rms_norm_2(rms_norm_input_2)
    rms_norm_result_ref_3 = rms_norm(rms_norm_input*0.1 + 1.0)
    rms_norm_result_ref_4 = rms_norm_2(rms_norm_input_2*1.1 + 1.0)

    snr_db_rms_norm_1 = calculate_snr(rms_norm_result_ref, rms_norm_result_device)
    snr_db_rms_norm_2 = calculate_snr(rms_norm_result_ref_2, rms_norm_result_device_2)
    snr_db_rms_norm_3 = calculate_snr(rms_norm_result_ref_3, rms_norm_result_device_3)
    snr_db_rms_norm_4 = calculate_snr(rms_norm_result_ref_4, rms_norm_result_device_4)
    print(f"RMS Norm SNR Analysis 1: {snr_db_rms_norm_1:.2f} dB")
    print(f"RMS Norm SNR Analysis 2: {snr_db_rms_norm_2:.2f} dB")
    print(f"RMS Norm SNR Analysis 3: {snr_db_rms_norm_3:.2f} dB")
    print(f"RMS Norm SNR Analysis 4: {snr_db_rms_norm_4:.2f} dB")
    assert snr_db_rms_norm_1 >= 40 or snr_db_rms_norm_1 == float('inf'), f"SNR {snr_db_rms_norm_1:.2f} dB must be at least 40 dB"
    assert snr_db_rms_norm_2 >= 40 or snr_db_rms_norm_2 == float('inf'), f"SNR {snr_db_rms_norm_2:.2f} dB must be at least 40 dB"
    assert snr_db_rms_norm_3 >= 40 or snr_db_rms_norm_3 == float('inf'), f"SNR {snr_db_rms_norm_3:.2f} dB must be at least 40 dB"
    assert snr_db_rms_norm_4 >= 40 or snr_db_rms_norm_4 == float('inf'), f"SNR {snr_db_rms_norm_4:.2f} dB must be at least 40 dB"

    # MATRIX-VECTOR MULTIPLY EXAMPLE =======================================================
    print("\n=== Running Matrix-Vector Multiply ===")
    M = 1024  # Number of rows
    N = 1024  # Number of columns (and vector size)
    matrix_bf16 = (torch.rand(M, N) * 2 - 1).to(torch.bfloat16)
    vector_bf16 = (torch.randn(N) * 2 - 1).to(torch.bfloat16)
    quantized_matvec = ue.quantized_matvec(matrix_bf16, M, N,
                               vector_dram_addr=DRAM_ACTIVATION_ADDR,
                               output_dram_addr= DRAM_ACTIVATION_ADDR + N * 2,
                               program_dram_addr=None,
                               params_dram_addr=None,
                               gelu_enable=False,
                               silu_enable=False,
                               data_type=TYPE.INT4)
    result_device = quantized_matvec(vector_bf16)
    ref = matrix_bf16 @ vector_bf16
    snr_db = calculate_snr(ref, result_device)
    print(f"Matrix-Vector Multiply SNR Analysis: {snr_db:.2f} dB")
    assert snr_db >= 22 or snr_db == float('inf'), f"SNR {snr_db:.2f} dB must be at least 40 dB"

    # MATRIX-MATRIX MULTIPLY EXAMPLE =======================================================
    print("\n=== Running Matrix-Matrix Multiply ===")
    M = 1  # Number of rows
    K = 1024  # Number of columns (and vector size)
    N = 1024  # Number of columns (and vector size)
    matrix_bf16 = (torch.rand(M, K) * 2 - 1).to(torch.bfloat16)
    matrix_bf16_2 = (torch.rand(N, K) * 2 - 1).to(torch.bfloat16)
    bias_vec = torch.randn(N).to(torch.bfloat16) * 21.345


    matmat_instance = ue.matmat_mul_quantized_weights(matrix_bf16_2, M=M, K=K, N=N,
                               input_dram_addr=DRAM_ACTIVATION_ADDR,
                               output_dram_addr=DRAM_ACTIVATION_ADDR + M * K * 2,
                               program_dram_addr=None,
                               params_dram_addr=None,
                               data_type=TYPE.INT4)
    result_device = matmat_instance(matrix_bf16)
    ref = matrix_bf16 @ matrix_bf16_2.t()
    snr_db = calculate_snr(ref, result_device)

    print(f"Quantized Matrix-Matrix Multiply SNR Analysis: {snr_db:.2f} dB")
    assert snr_db >= 22 or snr_db == float('inf'), f"SNR {snr_db:.2f} dB must be at least 25 dB"

    matmat_instance = ue.matmat_mul_quantized_weights(matrix_bf16_2, M=M, K=K, N=N,
                               input_dram_addr=DRAM_ACTIVATION_ADDR,
                               output_dram_addr=DRAM_ACTIVATION_ADDR + M * K * 2,
                               program_dram_addr=None,
                               params_dram_addr=None,
                               bias=bias_vec,
                               data_type=TYPE.INT4)
    result_device = matmat_instance(matrix_bf16)
    ref = matrix_bf16 @ matrix_bf16_2.t() + bias_vec
    snr_db = calculate_snr(ref, result_device)

    print(f"Quantized Matrix-Matrix Multiply with Bias SNR Analysis: {snr_db:.2f} dB")
    assert snr_db >= 25 or snr_db == float('inf'), f"SNR {snr_db:.2f} dB must be at least 25 dB"


    matmat_instance = ue.matmat_mul_quantized_weights(matrix_bf16_2, M=M, K=K, N=N,
                               input_dram_addr=DRAM_ACTIVATION_ADDR,
                               output_dram_addr=DRAM_ACTIVATION_ADDR + M * K * 2,
                               program_dram_addr=None,
                               params_dram_addr=None,
                               gelu_enable=True,
                               bias=bias_vec,
                               data_type=TYPE.INT4)
    result_device = matmat_instance(matrix_bf16)
    ref = torch.nn.GELU()(matrix_bf16 @ matrix_bf16_2.t() + bias_vec)
    snr_db = calculate_snr(ref, result_device)

    print(f"Quantized Matrix-Matrix Multiply with Bias GELU SNR Analysis: {snr_db:.2f} dB")
    assert snr_db >= 25 or snr_db == float('inf'), f"SNR {snr_db:.2f} dB must be at least 25 dB"

    matmat_instance = ue.matmat_mul_quantized_weights(matrix_bf16_2, M=M, K=K, N=N,
                               input_dram_addr=DRAM_ACTIVATION_ADDR,
                               output_dram_addr=DRAM_ACTIVATION_ADDR + M * K * 2,
                               program_dram_addr=None,
                               params_dram_addr=None,
                               silu_enable=True,
                               bias=bias_vec,
                               data_type=TYPE.INT4)
    result_device = matmat_instance(matrix_bf16)
    ref = torch.nn.SiLU()(matrix_bf16 @ matrix_bf16_2.t() + bias_vec)
    snr_db = calculate_snr(ref, result_device)

    print(f"Quantized Matrix-Matrix Multiply with Bias SiLU SNR Analysis: {snr_db:.2f} dB")
    assert snr_db >= 25 or snr_db == float('inf'), f"SNR {snr_db:.2f} dB must be at least 25 dB"

    # BF16 WEIGHT MATRIX-MATRIX MULTIPLY EXAMPLE =======================================================
    matmat_instance = ue.matmat_mul_quantized_weights(matrix_bf16_2, M=M, K=K, N=N,
                               input_dram_addr=DRAM_ACTIVATION_ADDR,
                               output_dram_addr=DRAM_ACTIVATION_ADDR + M * K * 2,
                               program_dram_addr=None,
                               params_dram_addr=None,
                               silu_enable=True,
                               bias=bias_vec,
                               data_type=TYPE.BF16)
    result_device = matmat_instance(matrix_bf16)
    ref = torch.nn.SiLU()(matrix_bf16 @ matrix_bf16_2.t() + bias_vec)
    snr_db = calculate_snr(ref, result_device)

    print(f"BF16 Weight Matrix-Matrix Multiply with Bias SiLU SNR Analysis: {snr_db:.2f} dB")
    assert snr_db >= 40 or snr_db == float('inf'), f"SNR {snr_db:.2f} dB must be at least 40 dB"

    matmat_instance = ue.matmat_mul_quantized_weights(matrix_bf16_2, M=M, K=K, N=N,
                               input_dram_addr=DRAM_ACTIVATION_ADDR,
                               output_dram_addr=DRAM_ACTIVATION_ADDR + M * K * 2,
                               program_dram_addr=None,
                               params_dram_addr=None,
                               bias=bias_vec,
                               data_type=TYPE.BF16)
    result_device = matmat_instance(matrix_bf16)
    ref = matrix_bf16 @ matrix_bf16_2.t() + bias_vec
    snr_db = calculate_snr(ref, result_device)

    print(f"BF16 Weight Matrix-Matrix Multiply with Bias SNR Analysis: {snr_db:.2f} dB")
    assert snr_db >= 40 or snr_db == float('inf'), f"SNR {snr_db:.2f} dB must be at least 40 dB"

    # BF16 MATRIX-VECTOR MULTIPLY EXAMPLE - bf16_matvec_activation =======================================================
    print("\n=== Running BF16 Matrix-Vector Multiply (Activation) ===")
    M_bf16 = 256  # Number of rows
    N_bf16 = 1024  # Number of columns (and vector size)
    M_bf16_2 = 2048
    N_bf16_2 = 768

    matrix_bf16 = (torch.rand(M_bf16, N_bf16) * 2 - 1).to(torch.bfloat16)
    vector_bf16 = (torch.randn(N_bf16) * 2 - 1).to(torch.bfloat16)
    vector_bf16_2 = (torch.randn(N_bf16) * 2 - 1).to(torch.bfloat16)

    matrix_bf16_2 = (torch.rand(M_bf16_2, N_bf16_2) * 2 - 1).to(torch.bfloat16)
    vector_bf16_3 = (torch.randn(N_bf16_2) * 2 - 1).to(torch.bfloat16)
    vector_bf16_4 = (torch.randn(N_bf16_2) * 2 - 1).to(torch.bfloat16)

    bf16_matvec_activation = ue.matmat_mul(M=1, K=N_bf16, N=M_bf16,
                                                                input_dram_addr=DRAM_ACTIVATION_ADDR,
                                                                weight_dram_addr=DRAM_ACTIVATION_ADDR + M_bf16 * N_bf16 * 2,
                                                                output_dram_addr=DRAM_ACTIVATION_ADDR + M_bf16 * N_bf16 * 2 + M_bf16 * 1 * 2,
                                                                program_dram_addr=None)

    bf16_matvec_activation_2 = ue.matmat_mul(M=1, K=N_bf16_2, N=M_bf16_2,
                                                   input_dram_addr=DRAM_ACTIVATION_ADDR,
                                                   weight_dram_addr=DRAM_ACTIVATION_ADDR + M_bf16_2 * N_bf16 * 2,
                                                   output_dram_addr=DRAM_ACTIVATION_ADDR + M_bf16_2 * N_bf16_2 * 2 + M_bf16_2 * 1 * 2,
                                                   program_dram_addr=None)

    result_bf16_device_activation = bf16_matvec_activation(vector_bf16.reshape(1, N_bf16), matrix_bf16)
    result_bf16_device_activation_3 = bf16_matvec_activation_2(vector_bf16_3.reshape(1, N_bf16_2), matrix_bf16_2)
    result_bf16_device_activation_2 = bf16_matvec_activation(vector_bf16_2.reshape(1, N_bf16), matrix_bf16)
    result_bf16_device_activation_4 = bf16_matvec_activation_2(vector_bf16_4.reshape(1, N_bf16_2), matrix_bf16_2)

    ref_bf16_activation = matrix_bf16 @ vector_bf16
    ref_bf16_activation_3 = matrix_bf16_2 @ vector_bf16_3
    ref_bf16_activation_2 = matrix_bf16 @ vector_bf16_2
    ref_bf16_activation_4 = matrix_bf16_2 @ vector_bf16_4

    snr_db_bf16_activation = calculate_snr(ref_bf16_activation, result_bf16_device_activation)
    snr_db_bf16_activation_3 = calculate_snr(ref_bf16_activation_3, result_bf16_device_activation_3)
    snr_db_bf16_activation_2 = calculate_snr(ref_bf16_activation_2, result_bf16_device_activation_2)
    snr_db_bf16_activation_4 = calculate_snr(ref_bf16_activation_4, result_bf16_device_activation_4)
    print(f"BF16 Matrix-Vector Multiply (Activation) SNR Analysis: {snr_db_bf16_activation:.2f} dB")
    print(f"BF16 Matrix-Vector Multiply (Activation) SNR Analysis 2: {snr_db_bf16_activation_2:.2f} dB")
    print(f"BF16 Matrix-Vector Multiply (Activation) SNR Analysis 3: {snr_db_bf16_activation_3:.2f} dB")
    print(f"BF16 Matrix-Vector Multiply (Activation) SNR Analysis 4: {snr_db_bf16_activation_4:.2f} dB")
    assert snr_db_bf16_activation >= 40 or snr_db_bf16_activation == float('inf'), f"SNR {snr_db_bf16_activation:.2f} dB must be at least 40 dB"
    assert snr_db_bf16_activation_2 >= 40 or snr_db_bf16_activation_2 == float('inf'), f"SNR {snr_db_bf16_activation_2:.2f} dB must be at least 40 dB"
    assert snr_db_bf16_activation_3 >= 40 or snr_db_bf16_activation_3 == float('inf'), f"SNR {snr_db_bf16_activation_3:.2f} dB must be at least 40 dB"
    assert snr_db_bf16_activation_4 >= 40 or snr_db_bf16_activation_4 == float('inf'), f"SNR {snr_db_bf16_activation_4:.2f} dB must be at least 40 dB"

    # BF16 Padding Zero Test =======================================================
    print("\n=== Running BF16 Padding Zero Test ===")
    M = 144
    N = 32
    a = torch.randn(M, N).to(torch.bfloat16)
    padding_zero_handler = ue.bf16_padding_zero(input_dram_addr=DRAM_ACTIVATION_ADDR,
                                                output_dram_addr=DRAM_ACTIVATION_ADDR + M * N * 2,
                                                M=M, N=N, program_dram_addr=None)
    padded_a = padding_zero_handler(a)
    snr_db_padding_zero = calculate_snr(a.flatten(), padded_a.data[:,:N].flatten())
    print(f"BF16 Padding Zero FPGA SNR Analysis: {snr_db_padding_zero:.2f} dB")
    assert snr_db_padding_zero >= 40 or snr_db_padding_zero == float('inf'), f"SNR {snr_db_padding_zero:.2f} dB must be at least 40 dB"

    # RoPE EXAMPLE =======================================================
    DIM = 1024
    N_HEADS = 4
    MAX_SEQ_LEN = 32768
    freqs_cis = precompute_freqs_cis(DIM // N_HEADS, MAX_SEQ_LEN * 2)
    x = torch.randn(DIM // N_HEADS, dtype=torch.bfloat16)
    one_rope_seq_params = freqs_cis[128, :]
    print(one_rope_seq_params.shape)
    one_rope_seq = torch.view_as_real(one_rope_seq_params).to(torch.bfloat16).reshape(-1)
    x_rotated_ref = torch.view_as_real(torch.view_as_complex(x.to(torch.float).reshape(-1, 2)) * one_rope_seq_params).to(torch.bfloat16).reshape(-1)

    # TODO: Native RoPE implementation is removed for now, as it is not supported by the hardware.
    # rope_size = x.numel()
    # rope_op = ue.rope_operation(rope_size,
    #                             x_vector_dram_addr=DRAM_ACTIVATION_ADDR,
    #                             output_dram_addr=DRAM_ACTIVATION_ADDR + rope_size * 2,
    #                             rope_params=one_rope_seq,
    #                             program_dram_addr=None,
    #                             params_dram_addr=None)
    # x_rotated_device = rope_op(x)
    # snr_db_rope = calculate_snr(x_rotated_ref, x_rotated_device)
    # print(f"RoPE SNR Analysis: {snr_db_rope:.2f} dB")
    # assert snr_db_rope >= 40 or snr_db_rope == float('inf'), f"SNR {snr_db_rope:.2f} dB must be at least 40 dB"

    # Hugging Face RoPE example
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    x_hf = x.reshape(-1, 2).permute(1, 0).reshape(-1) # weight permutation
    cos = torch.zeros(DIM // N_HEADS, dtype=torch.bfloat16)
    sin = torch.zeros(DIM // N_HEADS, dtype=torch.bfloat16)
    cos = torch.cat((one_rope_seq[0::2], one_rope_seq[0::2]), dim=-1)
    sin = torch.cat((one_rope_seq[1::2], one_rope_seq[1::2]), dim=-1)

    x_rotated_hf = ((x_hf * cos) + (rotate_half(x_hf) * sin)).reshape(2, -1).permute(1, 0).reshape(-1)
    snr_db_rope_hf = calculate_snr(x_rotated_ref, x_rotated_hf)
    print(f"Hugging Face RoPE SNR Analysis: {snr_db_rope_hf:.2f} dB")
    assert snr_db_rope_hf >= 40 or snr_db_rope_hf == float('inf'), f"SNR {snr_db_rope_hf:.2f} dB must be at least 40 dB"

    # sin, cos is input. x_hf is input and output is d.
    # sin[:sin.shape[0]//2] = -sin[:sin.shape[0]//2] # negate sin for the second half (critical step!!! Hugging Face is not doing this, it is doing as a part of the op)
    # a, a_latency = ue.element_wise_operation(x_hf, cos, UE_MODE.ELTWISE_MUL)
    # b, b_latency = ue.element_wise_operation(x_hf[x_hf.shape[0]//2:], sin[:sin.shape[0]//2], UE_MODE.ELTWISE_MUL)
    # c, c_latency = ue.element_wise_operation(x_hf[:x_hf.shape[0]//2], sin[sin.shape[0]//2:], UE_MODE.ELTWISE_MUL)
    # d, d_latency = ue.element_wise_operation(a, torch.cat((b, c), dim=-1), UE_MODE.ELTWISE_ADD)
    # output is d.
    rope_op_hf = ue.rope_operation_hf( x.numel(),
                                    x_vector_dram_addr=DRAM_ACTIVATION_ADDR,
                                    output_dram_addr=DRAM_ACTIVATION_ADDR + x.numel() * 2,
                                    cos=cos,
                                    sin=sin,
                                    program_dram_addr=None,
                                    params_dram_addr=None)
    d = rope_op_hf(x_hf)

    snr_db_rope_hf = calculate_snr(x_rotated_ref, d.reshape(2, -1).permute(1, 0).reshape(-1))
    print(f"Hugging Face RoPE SNR Analysis: {snr_db_rope_hf:.2f} dB")
    assert snr_db_rope_hf >= 40 or snr_db_rope_hf == float('inf'), f"SNR {snr_db_rope_hf:.2f} dB must be at least 40 dB"

    # EXP APPROXIMATE EXAMPLE =======================================================
    print("\n=== Running Exp Approximate Test ===")
    exp_size = 8192
    input_vector = torch.rand(exp_size).to(torch.bfloat16) * 2

    # Create exp handler
    exp_handler = ue.unary_op_exp(size=exp_size,
                                  input_dram_addr=DRAM_ACTIVATION_ADDR,
                                  output_dram_addr=DRAM_ACTIVATION_ADDR + exp_size * 2,
                                  program_dram_addr=None,
                                  params_dram_addr=None)

    # Execute exp approximation
    exp_result_device = exp_handler(input_vector)
    ref = 1 + input_vector + input_vector**2 / 2 + input_vector**3 / 6

    snr_db_exp = calculate_snr(ref, exp_result_device)
    print(f"Exp approximate SNR Analysis: {snr_db_exp:.2f} dB")
    assert snr_db_exp >= 40 or snr_db_exp == float('inf'), f"SNR {snr_db_exp:.2f} dB must be at least 40 dB"

    # Total parameters in the model
    print("\n=== Memory Usage Summary ===")
    total_params_usage = ue.get_params_dram_usage()
    print(f"Total parameters usage: {total_params_usage/(1024*1024)} MB")

    # Total program size in bytes
    total_program_size = ue.get_program_dram_usage()
    print(f"Total program size: {total_program_size/(1024*1024)} MB")


    print("\n=== Running BF16 Permute Test ===")
    dim_0 = 144
    dim_1 = 48
    dim_2 = 64

    a = torch.randn(dim_0, dim_1, dim_2).to(torch.bfloat16)
    reference_result = a.permute(1, 0, 2)

    c = torch.zeros(dim_0*dim_1, dim_2, dtype=torch.bfloat16)

    permute_a = torch.arange(0, dim_0*dim_1, dtype=torch.int32).reshape(dim_0, dim_1, 1)
    permute_a = permute_a.permute(1, 0, 2)
    permute_a = permute_a.flatten()

    for i in range(dim_0*dim_1):
        c[i, :] = a.reshape(-1, dim_2)[permute_a[i], :]

    snr_db_permute = calculate_snr(reference_result.flatten(), c.flatten())
    print(f"BF16 Permute Software Emulation SNR Analysis: {snr_db_permute:.2f} dB")
    assert snr_db_permute >= 40 or snr_db_permute == float('inf'), f"SNR {snr_db_permute:.2f} dB must be at least 40 dB"


    bf16_permute_handler = ue.bf16_permute(input_dram_addr=DRAM_ACTIVATION_ADDR,
                                           output_dram_addr=DRAM_ACTIVATION_ADDR + dim_0*dim_1*dim_2*2,
                                           dim_0=dim_0,
                                           dim_1=dim_1,
                                           dim_2=dim_2,
                                           program_dram_addr=None)
    from_device = bf16_permute_handler(a)

    snr_db_permute = calculate_snr(reference_result.flatten(), from_device.data.flatten())
    print(f"BF16 Permute FPGA SNR Analysis: {snr_db_permute:.2f} dB")
    assert snr_db_permute >= 40 or snr_db_permute == float('inf'), f"SNR {snr_db_permute:.2f} dB must be at least 40 dB"

    print("\n=== Running Stride Mode Memcpy Benchmark ===")
    M = 27648
    N = 64
    a = torch.randn(M, N).to(torch.bfloat16)
    some_slice = a[:, :N//2]
    bytes_per_element = 2
    stride_mode_memcpy_handler = ue.stride_mode_memcpy_benchmark(input_dram_addr=DRAM_ACTIVATION_ADDR,
                                                                 output_dram_addr=DRAM_ACTIVATION_ADDR + M * N * bytes_per_element,
                                                                 stride_bytes_per_chunk= N//2 * bytes_per_element,
                                                                 stride_jump_bytes=N * bytes_per_element,
                                                                 memcpy_length_bytes= M * N//2 * bytes_per_element,
                                                                 program_dram_addr=None)
    result = stride_mode_memcpy_handler(a)
    snr_db_stride_mode_memcpy = calculate_snr(some_slice.flatten(), result.data.flatten())
    print(f"Stride Mode Memcpy FPGA SNR Analysis: {snr_db_stride_mode_memcpy:.2f} dB")
    assert snr_db_stride_mode_memcpy >= 40 or snr_db_stride_mode_memcpy == float('inf'), f"SNR {snr_db_stride_mode_memcpy:.2f} dB must be at least 40 dB"

    print("\n=== Running Patching Test ===")
    a = torch.randn(3, 384, 384).to(torch.bfloat16)
    # Original permutation
    #  ref = a.reshape(3,96,4,96,4).permute(0, 1, 3, 2, 4).permute(2, 1, 0, 3, 4).reshape(-1, 48)    
    ref = a.reshape(3,96,4,96,4).permute(0, 1, 3, 2, 4).permute(2, 1, 0, 3, 4).permute(1, 0, 2, 3, 4).reshape(-1, 48)

    proj = ue.patching(input_dram_addr=DRAM_ACTIVATION_ADDR, output_dram_addr=DRAM_ACTIVATION_ADDR + 3 * 384 * 384 * 2)
    result = proj(a)

    snr_db_permute = calculate_snr(ref[:,:48].flatten(), result.data[:,:48].flatten())
    print(f"BF16 Permute FPGA SNR Analysis: {snr_db_permute:.2f} dB")
    assert snr_db_permute >= 40 or snr_db_permute == float('inf'), f"SNR {snr_db_permute:.2f} dB must be at least 40 dB"

    print("\n=== Running Stride Mode Memcpy Benchmark ===")
    M = 27648
    N = 64
    a = torch.randn(M, N).to(torch.bfloat16)
    some_slice = a[:, :N//2]
    stride_mode_memcpy_handler = ue.stride_mode_memcpy_benchmark(input_dram_addr=DRAM_ACTIVATION_ADDR,
                                                                 output_dram_addr=DRAM_ACTIVATION_ADDR + M * N * 2,
                                                                 stride_bytes_per_chunk= N//2 * 2,
                                                                 stride_jump_bytes=N * 2,
                                                                 memcpy_length_bytes= M * N//2 * 2,
                                                                 program_dram_addr=None)
    result = stride_mode_memcpy_handler(a)
    snr_db_stride_mode_memcpy = calculate_snr(some_slice.flatten(), result.data.flatten())
    print(f"Stride Mode Memcpy FPGA SNR Analysis: {snr_db_stride_mode_memcpy:.2f} dB")
    assert snr_db_stride_mode_memcpy >= 40 or snr_db_stride_mode_memcpy == float('inf'), f"SNR {snr_db_stride_mode_memcpy:.2f} dB must be at least 40 dB"

    print("\n=== All tests passed ===")

def simple_kq_test():
    """
    Tests K/Q projection and attention score computation.

    Tensor Memory Map (16MB regions):
    ┌─────────────┬───────────────────────────────────┬─────────────────────────┬─────────────────────────────────┐
    │ Offset      │ Address                           │ Tensor                  │ Description                     │
    ├─────────────┼───────────────────────────────────┼─────────────────────────┼─────────────────────────────────┤
    │ +0x0000000  │ DRAM_ACTIVATION_ADDR              │ Input X [seq, x_size]   │ Original input activation       │
    │ +0x1000000  │ DRAM_ACTIVATION_ADDR + 0x1000000  │ K Proj  [seq, k_size]   │ Key projection (X @ k_weight)   │
    │ +0x2000000  │ DRAM_ACTIVATION_ADDR + 0x2000000  │ Q Proj  [seq, q_size]   │ Query projection (X @ q_weight) │
    │ +0x3000000  │ DRAM_ACTIVATION_ADDR + 0x3000000  │ Scaled Q [seq, head_dim]│ Q * (1/√head_dim)               │
    │ +0x4000000  │ DRAM_ACTIVATION_ADDR + 0x4000000  │ Attn Scores [seq, seq]  │ softmax(K @ Q_scaled^T)         │
    └─────────────┴───────────────────────────────────┴─────────────────────────┴─────────────────────────────────┘

    Data Flow:
        Input X ─┬──► K Projection ────────────────────┐
                 │                                      ▼
                 └──► Q Projection ──► Scale (1/√d) ──► K @ Q^T ──► Softmax ──► Attention Scores
    """
    # let's do x and x projection example with cache hit
    print("\n=== Running K and Q Projection Example ===")


    ue = UnifiedEngine()

    head_dim = 1024
    x_size = head_dim
    k_size = head_dim
    q_size = head_dim
    seq_len = 1024
    x = torch.randn(seq_len, x_size).to(torch.bfloat16)
    q_weight = torch.randn(x_size, k_size).to(torch.bfloat16) * 0.1
    k_weight = torch.randn(k_size, q_size).to(torch.bfloat16) * 0.01

    k_projection = ue.matmat_mul_quantized_weights(k_weight, M=seq_len, K=x_size, N=k_size,
                                   input_dram_addr=DRAM_ACTIVATION_ADDR,
                                   output_dram_addr=DRAM_ACTIVATION_ADDR + 0x1000000,
                                   data_type=TYPE.BF16)

    q_projection = ue.matmat_mul_quantized_weights(q_weight, M=seq_len, K=x_size, N=q_size,
                                   input_dram_addr=DRAM_ACTIVATION_ADDR,
                                   output_dram_addr=DRAM_ACTIVATION_ADDR + 0x2000000,
                                   data_type=TYPE.BF16)

    broadcast_mult = ue.broadcast_op(head_dim * seq_len,
                                    broadcast_type=UE_MODE.MUL_BROADCAST,
                                    scalar = 1/math.sqrt(head_dim),
                                    input_dram_addr=DRAM_ACTIVATION_ADDR + 0x2000000,
                                    output_dram_addr=DRAM_ACTIVATION_ADDR + 0x3000000)

    k_qt_calc = ue.matmat_mul(M=seq_len, K=head_dim, N=seq_len,
                                   input_dram_addr=DRAM_ACTIVATION_ADDR + 0x1000000,
                                   weight_dram_addr=DRAM_ACTIVATION_ADDR + 0x3000000,
                                   output_dram_addr=DRAM_ACTIVATION_ADDR + 0x4000000,
                                   softmax_enable=True)

    x_device = DeviceTensor((seq_len, x_size), ue, DRAM_ACTIVATION_ADDR, data=x)
    x_device.sync(DRAM_ACTIVATION_ADDR)

    # FPGA execution
    q_result_device = q_projection(x_device)
    k_result_device = k_projection(x_device)
    q_scale_result_device = broadcast_mult(q_result_device)
    k_qt_result_device = k_qt_calc(k_result_device, q_scale_result_device)

    # Reference calculation
    scalar = torch.ones(1, dtype=torch.bfloat16) * (1 / math.sqrt(head_dim))
    scalar = scalar.to(torch.bfloat16)
    q_result_ref = (x @ q_weight.t()) * scalar
    k_result_ref = x @ k_weight.t()
    k_qt_ref = k_result_ref @ q_result_ref.t()
    sm_ref = torch.softmax(k_qt_ref, dim=-1).to(torch.bfloat16)

    # SNR calculation
    snr_db_k_qt = calculate_snr(sm_ref, k_qt_result_device.data)

    print(f"K @ Q^T SNR Analysis: {snr_db_k_qt:.2f} dB")

    assert snr_db_k_qt >= 40 or snr_db_k_qt == float('inf'), f"SNR {snr_db_k_qt:.2f} dB must be at least 40 dB"

def flash_attention_test(head_dim: int, seq_len: int, bias_enable: bool = False):
    """
    Tests flash attention core.
    """
    ue = UnifiedEngine()

    Q_DRAM_ADDR = ue.allocate_tensor_dram(seq_len * head_dim * 2)
    K_DRAM_ADDR = ue.allocate_tensor_dram(seq_len * head_dim * 2)
    V_DRAM_ADDR = ue.allocate_tensor_dram(seq_len * head_dim * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(seq_len * head_dim * 2)
    SCRATCH_DRAM_ADDR = ue.allocate_tensor_dram(max(head_dim, UE_FMAX_CONTEXT_SIZE) * seq_len * 2) # it is max of head_dim * seq_len and UE_FMAX_CONTEXT_SIZE * seq_len
    BIAS_DRAM_ADDR = ue.allocate_tensor_dram(seq_len * seq_len * 2)

    ue.start_capture() # -------------------------------------------------------------

    total_flops_from_flash_attention = ue.flash_attention_core(head_dim=head_dim, seq_len=seq_len,
                                                            Q_DRAM_ADDR=Q_DRAM_ADDR,
                                                            K_DRAM_ADDR=K_DRAM_ADDR,
                                                            V_DRAM_ADDR=V_DRAM_ADDR,
                                                            OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                                                            SCRATCH_DRAM_ADDR=SCRATCH_DRAM_ADDR,
                                                            BIAS_DRAM_ADDR=BIAS_DRAM_ADDR if bias_enable else None)

    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())
    ue.clear_capture_buffer()

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
    ue.wait_queue(20.0) # 20 seconds timeout
    ue.report_timing_and_instruction_count()

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (seq_len, head_dim))
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

    snr_db_ref = calculate_snr(ref, output)
    print(f"Reference SNR Analysis for Flash Attention: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 40 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 40 dB"

    # Before running matmat_mul_core, let's clear the capture buffer and reset the tensor DRAM address
    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

def matmat_mul_test(M: int, K: int, N: int, bias_enable: bool = False, softmax_enable: bool = False, bias_mode: str = "broadcast_N"):
    """
    Tests matmat_mul core.
    """
    ue = UnifiedEngine()

    A_DRAM_ADDR = ue.allocate_tensor_dram(M * K * 2)
    B_DRAM_ADDR = ue.allocate_tensor_dram(N * K * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    C_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2) if bias_enable else None

    ue.start_capture()

    total_flops_from_matmat_mul = ue.matmat_mul_core(M=M, K=K, N=N,
                                                    A_DRAM_ADDR=A_DRAM_ADDR,
                                                    B_DRAM_ADDR=B_DRAM_ADDR,
                                                    OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                                                    softmax_enable=softmax_enable,
                                                    C_DRAM_ADDR=C_DRAM_ADDR,
                                                    bias_mode=bias_mode)

    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())
    ue.clear_capture_buffer()

    # Test Time
    a = torch.randn(M, K, dtype=torch.bfloat16) / math.sqrt(K) # normalizing input helps with numerical stability of softmax
    b = torch.randn(N, K, dtype=torch.bfloat16)
    c = torch.zeros(M, N, dtype=torch.bfloat16) if bias_enable else None

    # DMA to accelerator memory -------------------------------------------------------------
    ue.dma_to_accelerator_memory(A_DRAM_ADDR, a)
    ue.dma_to_accelerator_memory(B_DRAM_ADDR, b)

    if bias_enable:
        ue.dma_to_accelerator_memory(C_DRAM_ADDR, c)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0) # 10 seconds timeout
    ue.report_timing_and_instruction_count()

    report_flop_rate_gflops = ue.report_flop_rate_gflops(total_flops_from_matmat_mul)
    print(f"Report FLOPS for MxKxN Matmul: {report_flop_rate_gflops:.2f} GFLOPS for M={M}, K={K}, N={N}, bias_enable={bias_enable}, softmax_enable={softmax_enable}, bias_mode={bias_mode}")

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M, N))

    ref = (a @ b.T + c) if bias_enable else (a @ b.T)
    if softmax_enable:
        ref = torch.softmax(ref, dim=-1).to(torch.bfloat16)

    snr_db_ref = calculate_snr(ref, output)
    print(f"Reference SNR Analysis for MxKxN Matmul: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 40 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 40 dB"

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
    B_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    ADDOUTPUT_DRAM_ADDR_1 = ue.allocate_tensor_dram(M * N * 2)
    NORMOUTPUT_DRAM_ADDR_1 = ue.allocate_tensor_dram(M * N * 2)
    ADDOUTPUT_DRAM_ADDR_2 = ue.allocate_tensor_dram(M * N * 2)
    NORMOUTPUT_DRAM_ADDR_2 = ue.allocate_tensor_dram(M * N * 2)
    GAMMA_DRAM_ADDR = ue.allocate_tensor_dram(N * 2) if gamma_enable else None
    BETA_DRAM_ADDR = ue.allocate_tensor_dram(N * 2) if beta_enable else None

    ue.start_capture()

    total_flops_from_layer_norm = ue.layer_norm_core_dram(M=M, N=N,
                                                          A_DRAM_ADDR=A_DRAM_ADDR,
                                                          OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                                                          GAMMA_DRAM_ADDR=GAMMA_DRAM_ADDR,
                                                          BETA_DRAM_ADDR=BETA_DRAM_ADDR)
    total_flops_from_layer_norm_pre_add = ue.layer_norm_core_dram_pre_add(M=M, N=N,
                                                          A_DRAM_ADDR=A_DRAM_ADDR,
                                                          B_DRAM_ADDR=B_DRAM_ADDR,
                                                          ADDOUTPUT_DRAM_ADDR=ADDOUTPUT_DRAM_ADDR_1,
                                                          NORMOUTPUT_DRAM_ADDR=NORMOUTPUT_DRAM_ADDR_1,
                                                          GAMMA_DRAM_ADDR=GAMMA_DRAM_ADDR,
                                                          BETA_DRAM_ADDR=BETA_DRAM_ADDR)
    total_flops_from_layer_norm_post_add = ue.layer_norm_core_dram_post_add(M=M, N=N,
                                                          A_DRAM_ADDR=A_DRAM_ADDR,
                                                          B_DRAM_ADDR=B_DRAM_ADDR,
                                                          ADDOUTPUT_DRAM_ADDR=ADDOUTPUT_DRAM_ADDR_2,
                                                          NORMOUTPUT_DRAM_ADDR=NORMOUTPUT_DRAM_ADDR_2,
                                                          GAMMA_DRAM_ADDR=GAMMA_DRAM_ADDR,
                                                          BETA_DRAM_ADDR=BETA_DRAM_ADDR)
    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())
    ue.clear_capture_buffer()

    # Test Time
    x = torch.randn(M, N, dtype=torch.bfloat16)
    y = torch.randn(M, N, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(A_DRAM_ADDR, x)
    ue.dma_to_accelerator_memory(B_DRAM_ADDR, y)
    if gamma_enable:
        gamma = torch.randn(N, dtype=torch.bfloat16)
        ue.dma_to_accelerator_memory(GAMMA_DRAM_ADDR, gamma)
    if beta_enable:
        beta = torch.randn(N, dtype=torch.bfloat16)
        ue.dma_to_accelerator_memory(BETA_DRAM_ADDR, beta)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0) # 10 seconds timeout
    ue.report_timing_and_instruction_count()

    flop_rate_gflops = ue.report_flop_rate_gflops(total_flops_from_layer_norm)
    print(f"Report FLOPS for Layer Norm: {flop_rate_gflops:.2f} GFLOPS for M={M}, N={N}, gamma_enable={gamma_enable}, beta_enable={beta_enable}")
    flop_rate_gflops_pre_add = ue.report_flop_rate_gflops(total_flops_from_layer_norm_pre_add)
    print(f"Report FLOPS for Layer Norm Pre-Add: {flop_rate_gflops_pre_add:.2f} GFLOPS for M={M}, N={N}, gamma_enable={gamma_enable}, beta_enable={beta_enable}")
    flop_rate_gflops_post_add = ue.report_flop_rate_gflops(total_flops_from_layer_norm_post_add)
    print(f"Report FLOPS for Layer Norm Post-Add: {flop_rate_gflops_post_add:.2f} GFLOPS for M={M}, N={N}, gamma_enable={gamma_enable}, beta_enable={beta_enable}")

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M, N))
    pre_normoutput = ue.dma_from_accelerator_memory(NORMOUTPUT_DRAM_ADDR_1, (M, N))
    pre_addoutput = ue.dma_from_accelerator_memory(ADDOUTPUT_DRAM_ADDR_1, (M, N))
    post_normoutput = ue.dma_from_accelerator_memory(NORMOUTPUT_DRAM_ADDR_2, (M, N))
    post_addoutput = ue.dma_from_accelerator_memory(ADDOUTPUT_DRAM_ADDR_2, (M, N))

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

    pre_ref_norm = layer_norm(x)
    pre_ref_add = layer_norm(x) + y
    pre_norm_snr = calculate_snr(pre_ref_norm, pre_normoutput)
    pre_add_snr = calculate_snr(pre_ref_add, pre_addoutput)

    print(f"Reference SNR Analysis for Layer Norm Pre-Add: Addition: {pre_add_snr:.2f} dB , Norm: {pre_norm_snr:.2f}")
    assert pre_norm_snr >= 40 or pre_norm_snr == float('inf'), f"Pre-Add Norm SNR {pre_norm_snr:.2f} dB must be at least 40 dB"
    assert pre_add_snr >= 40 or pre_add_snr == float('inf'), f"Pre-Add Addition SNR {pre_add_snr:.2f} dB must be at least 40 dB"

    post_ref_add = x + y
    post_ref_norm = layer_norm(post_ref_add)
    post_add_snr = calculate_snr(post_ref_add, post_addoutput)
    post_norm_snr = calculate_snr(post_ref_norm, post_normoutput)
    print(f"Reference SNR Analysis for Layer Norm Post-Add: Addition: {post_add_snr:.2f} dB , Norm: {post_norm_snr:.2f}")
    assert post_add_snr >= 40 or post_add_snr == float('inf'), f"Post-Add Addition SNR {post_add_snr:.2f} dB must be at least 40 dB"
    assert post_norm_snr >= 40 or post_norm_snr == float('inf'), f"Post-Add Norm SNR {post_norm_snr:.2f} dB must be at least 40 dB"


    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

def custom_kernel_test():
    """
    Custom kernel test.
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

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='User DMA Operations for Unified Engine')
    parser.add_argument('--dev', type=str, default='xdma0',
                        help='DMA device name (e.g., xdma0, xdma1). Default: xdma0')
    parser.add_argument('--cycle', type=float, default=3.0,
                        help='Clock cycle time in nanoseconds (default: 3.0, use 2.5 for alveo)')
    args = parser.parse_args()
    
    # Set DMA device paths based on device name
    set_dma_device(args.dev)
    print(f"Using DMA device: {args.dev}")
    print(f"  H2C: {DMA_DEVICE_H2C}")
    print(f"  C2H: {DMA_DEVICE_C2H}")
    print(f"  USER: {DMA_DEVICE_USER}")
    
    # Set CLOCK_CYCLE_TIME_NS in core module so engine code sees it
    import user_dma_core
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle
    print(f"Setting CLOCK_CYCLE_TIME_NS = {user_dma_core.CLOCK_CYCLE_TIME_NS}")
    
    #generic_tests()
    #simple_kq_test()

    custom_kernel_test()
    flash_attention_test(head_dim=256, seq_len=3840, bias_enable=True)
    matmat_mul_test(M=1024, K=1024, N=1024, bias_enable=True, softmax_enable=True, bias_mode="full_matrix")
    layer_norm_test(shape=(1024, 1024), gamma_enable=True, beta_enable=True)
    #rms_norm_test(shape=(1024, 1024))

    # for M in [64, 128, 256, 512, 1024, 2048, 4096]:
    #     for N in [64, 128, 256, 512, 1024, 2048, 4096]:
    #         for gamma_enable in [True, False]:
    #             for beta_enable in [True, False]:   
    #                 layer_norm_test(shape=(M, N), gamma_enable=gamma_enable, beta_enable=beta_enable)
    #                 print(f"Layer Norm Test for M={M}, N={N}, gamma_enable={gamma_enable}, beta_enable={beta_enable}")

    # for M in [64, 128, 256, 512, 1024, 2048, 4096]:
    #     for N in [64, 128, 256, 512, 1024, 2048, 4096]:
    #         rms_norm_test(shape=(M, N))
    #         print(f"RMS Norm Test for M={M}, N={N}")

    # for head_dim in [64, 128, 256, 512, 1024, 2048]:
    #     for seq_len in [64, 128, 256, 512, 1024, 2048, 4032]:
    #         for bias_enable in [True, False]:
    #             flash_attention_test(head_dim=head_dim, seq_len=seq_len, bias_enable=bias_enable)

    # for M in [64, 128, 256, 512, 1024, 2048, 4096]:
    #     for K in [64, 128, 256, 512, 1024, 2048, 4032]:
    #         for N in [64, 128, 256, 512, 1024, 2048, 4096]:
    #             for bias_enable in [True, False]:
    #                 for softmax_enable in [True, False]:
    #                     for bias_mode in ["broadcast_N", "full_matrix"]:
    #                         matmat_mul_test(M=M, K=K, N=N, bias_enable=bias_enable, softmax_enable=softmax_enable, bias_mode=bias_mode)

