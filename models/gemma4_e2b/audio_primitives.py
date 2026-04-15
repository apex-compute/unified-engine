"""
FPGA primitives for Gemma4 E2B audio (Conformer) encoder.

Ported from /home/davew/apex-compute-ML/unified-engine/models/parakeet/parakeet_test.py.
These are pure instruction-emit helpers; they assume the caller has
``start_capture()`` open and will ``stop_capture()`` later. Each helper takes
a UnifiedEngine and DRAM addresses, no hidden state.

Helpers:
    glu_core_dram        — output = a * sigmoid(b), GLU
    silu_core_dram       — output = x * sigmoid(x), SiLU
    half_step_residual_core_dram — output = residual + 0.5 * ff_out
    softcap_core_dram    — output = soft_cap * tanh(input / soft_cap)
    build_toeplitz_for_depthwise — host-side Toeplitz matrix builder for depthwise 1D conv
    depthwise_conv1d_core_dram — per-channel matmul executing the conv
"""

from __future__ import annotations

import torch

from user_dma_core import UE_VECTOR_SIZE, URAM_FULL_ELEMENTS


# Standard URAM working addresses used by all helpers below.
URAM_A_BASE = 0x00000
URAM_B_BASE = 0x80000


def _row_chunk(M: int, N: int, divisor: int = 2) -> int:
    """Pick a row chunk size so that ``M_chunk * N`` fits in one URAM bank
    with headroom (``URAM_FULL_ELEMENTS // divisor``). Always at least 1.

    divisor=2 leaves room for both an A-bank load and a B-bank load running
    concurrently in the eltwise step.
    """
    cap = URAM_FULL_ELEMENTS // max(divisor, 1)
    m = max(1, cap // N)
    return min(M, m)


def silu_core_dram(ue, M: int, N: int,
                   A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                   IDENTITY_DRAM_ADDR: int) -> None:
    """Standalone SiLU on a (M, N) bf16 tensor: output = x * sigmoid(x).

    Chunks rows so each sub-batch's bulk SRAM ops fit in URAM_A/B.

    Per chunk:
      1. Copy x_chunk → OUTPUT_chunk.
      2. matmat(M_chunk, N, N) @ I(N, N) sigmoid_enable=True → sigmoid(x) at OUTPUT_chunk.
      3. eltwise_mul x_chunk * sigmoid(x_chunk) → OUTPUT_chunk.
    """
    assert N % UE_VECTOR_SIZE == 0, f"N={N} must be a multiple of {UE_VECTOR_SIZE}"
    bpe = 2
    M_chunk = _row_chunk(M, N, divisor=2)
    row_bytes = N * bpe
    for m_start in range(0, M, M_chunk):
        m_take = min(M_chunk, M - m_start)
        chunk_elems = m_take * N
        chunk_in = A_DRAM_ADDR + m_start * row_bytes
        chunk_out = OUTPUT_DRAM_ADDR + m_start * row_bytes
        # Step 1: copy chunk to OUTPUT (so we can sigmoid in place there).
        ue.accelerator_memory_to_sram(chunk_in, URAM_A_BASE, chunk_elems)
        ue.sram_to_accelerator_memory(URAM_A_BASE, chunk_out, chunk_elems)
        # Step 2: identity matmul with sigmoid → sigmoid(x) at OUTPUT.
        # matmat_mul_core handles its own internal URAM chunking.
        ue.matmat_mul_core(M=m_take, K=N, N=N,
                           A_DRAM_ADDR=chunk_out,
                           B_DRAM_ADDR=IDENTITY_DRAM_ADDR,
                           OUTPUT_DRAM_ADDR=chunk_out,
                           sigmoid_enable=True)
        # Step 3: bulk x * sigmoid(x) into OUTPUT.
        ue.accelerator_memory_to_sram(chunk_in, URAM_A_BASE, chunk_elems)
        ue.accelerator_memory_to_sram(chunk_out, URAM_B_BASE, chunk_elems)
        ue.eltwise_mul_core(vector_A_sram_start_addr=URAM_A_BASE,
                            vector_B_sram_start_addr=URAM_B_BASE,
                            vector_C_sram_wb_addr=URAM_A_BASE,
                            element_size=chunk_elems)
        ue.sram_to_accelerator_memory(URAM_A_BASE, chunk_out, chunk_elems)


def glu_core_dram(ue, M: int, C: int,
                  GATE_DRAM_ADDR: int, VALUE_DRAM_ADDR: int,
                  OUTPUT_DRAM_ADDR: int, IDENTITY_DRAM_ADDR: int) -> None:
    """GLU on (M, 2C) → (M, C). Caller must split the linear projection's
    output into the first half (``GATE``) and second half (``VALUE``) BEFORE
    calling, OR pass the same buffer with appropriate strides.

    Convention matches torch.nn.functional.glu(x, dim=-1):
        a, b = x.chunk(2, dim=-1); out = a * sigmoid(b)

    So GATE = a (first half), VALUE = b (second half, sigmoid'd).

    Chunked over rows so eltwise_mul fits in URAM A+B.
    """
    assert C % UE_VECTOR_SIZE == 0, f"C={C} must be a multiple of {UE_VECTOR_SIZE}"
    bpe = 2
    # Step 1: in-place sigmoid(VALUE) — matmat handles its own URAM chunking
    ue.matmat_mul_core(M=M, K=C, N=C,
                       A_DRAM_ADDR=VALUE_DRAM_ADDR,
                       B_DRAM_ADDR=IDENTITY_DRAM_ADDR,
                       OUTPUT_DRAM_ADDR=VALUE_DRAM_ADDR,
                       sigmoid_enable=True)
    # Step 2: eltwise_mul GATE * sigmoid(VALUE), chunked over rows
    M_chunk = _row_chunk(M, C, divisor=2)
    row_bytes = C * bpe
    for m_start in range(0, M, M_chunk):
        m_take = min(M_chunk, M - m_start)
        chunk_elems = m_take * C
        ue.accelerator_memory_to_sram(GATE_DRAM_ADDR + m_start * row_bytes,
                                       URAM_A_BASE, chunk_elems)
        ue.accelerator_memory_to_sram(VALUE_DRAM_ADDR + m_start * row_bytes,
                                       URAM_B_BASE, chunk_elems)
        ue.eltwise_mul_core(vector_A_sram_start_addr=URAM_A_BASE,
                            vector_B_sram_start_addr=URAM_B_BASE,
                            vector_C_sram_wb_addr=URAM_A_BASE,
                            element_size=chunk_elems)
        ue.sram_to_accelerator_memory(URAM_A_BASE,
                                       OUTPUT_DRAM_ADDR + m_start * row_bytes,
                                       chunk_elems)


def eltwise_add_core_dram(ue, M: int, N: int,
                          A_DRAM_ADDR: int, B_DRAM_ADDR: int,
                          OUTPUT_DRAM_ADDR: int) -> None:
    """output = A + B for two (M, N) bf16 tensors. Chunked over rows.

    OUTPUT can overlap A or B safely (we read both into SRAM before writing).
    """
    assert N % UE_VECTOR_SIZE == 0, f"N={N} must be a multiple of {UE_VECTOR_SIZE}"
    bpe = 2
    M_chunk = _row_chunk(M, N, divisor=2)
    row_bytes = N * bpe
    for m_start in range(0, M, M_chunk):
        m_take = min(M_chunk, M - m_start)
        chunk_elems = m_take * N
        ue.accelerator_memory_to_sram(A_DRAM_ADDR + m_start * row_bytes,
                                       URAM_A_BASE, chunk_elems)
        ue.accelerator_memory_to_sram(B_DRAM_ADDR + m_start * row_bytes,
                                       URAM_B_BASE, chunk_elems)
        ue.eltwise_add_core(vector_A_sram_start_addr=URAM_A_BASE,
                            vector_B_sram_start_addr=URAM_B_BASE,
                            vector_C_sram_wb_addr=URAM_A_BASE,
                            element_size=chunk_elems)
        ue.sram_to_accelerator_memory(URAM_A_BASE,
                                       OUTPUT_DRAM_ADDR + m_start * row_bytes,
                                       chunk_elems)


def half_step_residual_core_dram(ue, M: int, N: int,
                                 RESIDUAL_DRAM_ADDR: int, FF_DRAM_ADDR: int,
                                 OUTPUT_DRAM_ADDR: int) -> None:
    """output = residual + 0.5 * ff_output. Chunked over rows."""
    assert N % UE_VECTOR_SIZE == 0, f"N={N} must be a multiple of {UE_VECTOR_SIZE}"
    bpe = 2
    M_chunk = _row_chunk(M, N, divisor=2)
    row_bytes = N * bpe
    for m_start in range(0, M, M_chunk):
        m_take = min(M_chunk, M - m_start)
        chunk_elems = m_take * N
        chunk_ff = FF_DRAM_ADDR + m_start * row_bytes
        chunk_res = RESIDUAL_DRAM_ADDR + m_start * row_bytes
        chunk_out = OUTPUT_DRAM_ADDR + m_start * row_bytes
        ue.accelerator_memory_to_sram(chunk_ff, URAM_A_BASE, chunk_elems)
        ue.broadcast_mul(scalar=0.5, sram_start_addr=URAM_A_BASE,
                         sram_wb_addr=URAM_A_BASE, element_size=chunk_elems)
        ue.accelerator_memory_to_sram(chunk_res, URAM_B_BASE, chunk_elems)
        ue.eltwise_add_core(vector_A_sram_start_addr=URAM_A_BASE,
                            vector_B_sram_start_addr=URAM_B_BASE,
                            vector_C_sram_wb_addr=URAM_A_BASE,
                            element_size=chunk_elems)
        ue.sram_to_accelerator_memory(URAM_A_BASE, chunk_out, chunk_elems)


def copy_dram_to_dram_chunked(ue, src: int, dst: int, total_elems: int,
                              row_n: int = None) -> None:
    """Bulk DRAM-to-DRAM copy via SRAM, chunked so it fits URAM_A.

    If ``row_n`` is given, the copy is split into row-aligned chunks of
    ``row_n`` columns; otherwise the buffer is treated as flat and split
    every URAM_A_capacity elements.
    """
    bpe = 2
    if row_n is not None:
        rows = total_elems // row_n
        M_chunk = _row_chunk(rows, row_n, divisor=1)
        row_bytes = row_n * bpe
        for m_start in range(0, rows, M_chunk):
            m_take = min(M_chunk, rows - m_start)
            n = m_take * row_n
            ue.accelerator_memory_to_sram(src + m_start * row_bytes,
                                          URAM_A_BASE, n)
            ue.sram_to_accelerator_memory(URAM_A_BASE,
                                          dst + m_start * row_bytes, n)
    else:
        cap = URAM_FULL_ELEMENTS  # whole bank
        offset = 0
        while offset < total_elems:
            n = min(cap, total_elems - offset)
            ue.accelerator_memory_to_sram(src + offset * bpe,
                                          URAM_A_BASE, n)
            ue.sram_to_accelerator_memory(URAM_A_BASE,
                                          dst + offset * bpe, n)
            offset += n


def build_toeplitz_for_depthwise(kernel: torch.Tensor, L_pad: int,
                                 causal: bool = True) -> torch.Tensor:
    """Build a depthwise-1D-conv equivalent Toeplitz matrix per channel.

    For each channel ``c``, the resulting (L_pad, L_pad) matrix ``T[c]`` has
    the property that ``output_row = input_row @ T[c]`` is the convolution of
    ``input_row`` (length L_pad) with ``kernel[c]`` (length k).

    Args:
        kernel: (D, k) bf16 — per-channel 1D kernel of length k.
        L_pad:  output / input length (must be ≥ k).
        causal: if True, the kernel is left-padded so output position t reads
                input positions [t-(k-1) .. t]. This matches Gemma4's
                ``Gemma4AudioCausalConv1d`` (``F.pad(x, (left_pad, 0))``).
                If False, the kernel is centered (Parakeet style), giving
                output position t reads input positions [t-(k//2) .. t+(k//2)].

    Returns:
        (D, L_pad, L_pad) bf16 Toeplitz tensor ready to DMA to FPGA.
    """
    assert kernel.dim() == 2, f"kernel must be (D, k); got {kernel.shape}"
    D, k = kernel.shape
    toeplitz = torch.zeros(D, L_pad, L_pad, dtype=torch.bfloat16)
    for tap in range(k):
        # `tap` indexes into the kernel; `offset` is the input column relative
        # to the output row. For causal (left padding only), output t reads
        # input (t - (k-1) + tap), so offset = tap - (k-1).
        if causal:
            offset = tap - (k - 1)
        else:
            offset = tap - (k // 2)
        # row indices where this tap contributes (input column must be in
        # range [0, L_pad)).
        row_lo = max(0, -offset)
        row_hi = min(L_pad, L_pad - offset)
        if row_lo >= row_hi:
            continue
        rows = torch.arange(row_lo, row_hi)
        cols = rows + offset
        toeplitz[:, rows, cols] = kernel[:, tap].unsqueeze(-1).expand(-1, len(rows))
    return toeplitz


def depthwise_conv1d_core_dram(ue, D: int, L_pad: int,
                               INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                               TOEPLITZ_DRAM_ADDR: int) -> None:
    """Depthwise 1D conv via per-channel Toeplitz matmul.

    Inputs at INPUT_DRAM_ADDR are (D, L_pad) bf16 — channel-first layout.
    Toeplitz at TOEPLITZ_DRAM_ADDR is (D, L_pad, L_pad) bf16, built by
    build_toeplitz_for_depthwise().

    Output at OUTPUT_DRAM_ADDR is (D, L_pad) bf16, same layout as input.

    Each channel runs an independent (1, L_pad) @ (L_pad, L_pad) matmul,
    sequentially. Latency = D × matmul cost.

    Caller must transpose to channel-first BEFORE calling this and back to
    sequence-first AFTER, since Conformer's outer code keeps tensors as
    (L_pad, D).
    """
    assert L_pad % UE_VECTOR_SIZE == 0, (
        f"L_pad={L_pad} must be a multiple of {UE_VECTOR_SIZE}")
    bpe = 2  # bf16
    row_bytes = L_pad * bpe
    toeplitz_bytes_per_channel = L_pad * L_pad * bpe
    for ch in range(D):
        ue.matmat_mul_core(
            M=1, K=L_pad, N=L_pad,
            A_DRAM_ADDR=INPUT_DRAM_ADDR + ch * row_bytes,
            B_DRAM_ADDR=TOEPLITZ_DRAM_ADDR + ch * toeplitz_bytes_per_channel,
            OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR + ch * row_bytes,
        )
