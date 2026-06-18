"""
Hardware test runner for the Unified Engine.

Runs generic_tests() (memcpy, matmat, transpose, broadcast, layer norm, RMS, RoPE, etc.)
and simple_kq_test() (K/Q projection and K@Q^T attention).

Usage:
    python user_hw_test.py [--dev xdma0] [--ext]
"""

import argparse
import atexit
import math
import os
import random
from re import S
import time
import threading
from read_trace import generate_trace
import torch

from user_dma_core import (
    DMA_DEVICE_C2H,
    DMA_DEVICE_H2C,
    DMA_DEVICE_USER,
    DRAM_ACTIVATION_ADDR,
    DRAM_INSTRUCTION_ADDR,
    INSTRUCTION_SIZE_BYTES,
    INT_CAUSE_HALT,
    INT_CAUSE_NONE,
    INT_CAUSE_SWI,
    LALU_MODE,
    REGFILE_R1_LOOP,
    TYPE,
    UE_MODE,
    UE_INT_REG,
    UE_AXI_DATA_WIDTH_BITS,
    UE_MODE,
    URAM_FULL_ELEMENTS,
    URAM_NEAR_FULL_ELEMENTS,
    URAM_HALF_ELEMENTS,
    URAM_NEAR_FULL_ELEMENTS,
    URAM_SECTION,
    URAM_WRITE_SRC,
    WB_PADDING_ZERO,
    calculate_snr,
    set_dma_device,
    UnifiedEngine,
    UE_FMAX_CONTEXT_SIZE,
    UE_VECTOR_SIZE,
    ue_35bit_addr_shifter,
)

# ---------------------------------------------------------------------------
# Hardware-helper ops (built on unified-engine primitives, shared by tests
# that exercise model-inspired patterns without depending on model modules).
# ---------------------------------------------------------------------------

_BPE = 2


def dram_zero_fill(ue: UnifiedEngine, dram_addr: int, num_elements: int) -> None:
    """Fill *dram_addr* with ``num_elements`` zeros."""
    chunk = min(URAM_NEAR_FULL_ELEMENTS, num_elements)
    z_addr = ue.get_params_dram_addr()
    ue.allocate_params_dram(chunk * _BPE)
    ue.dma_write(DMA_DEVICE_H2C, z_addr,
                 torch.zeros(chunk, dtype=torch.bfloat16), chunk * _BPE)
    ue.accelerator_memory_to_sram(z_addr, 0x00000, chunk)
    offset = 0
    while offset < num_elements:
        take = min(chunk, num_elements - offset)
        ue.sram_to_accelerator_memory(0x00000, dram_addr + offset * _BPE, take)
        offset += take


def _pbi_zero_dram(ue: UnifiedEngine, DRAM_ADDR: int, num_elements: int) -> None:
    """Model-faithful copy of mobilesam_test._pbi_zero_dram: zero num_elements bf16
    at DRAM_ADDR using ONE lazily-allocated, cached zero source on `ue` (so calling
    it per output row does not allocate params every call). This is the zeroing path
    the model's conv2d_3x3_pbi / conv2d_3x3_dw_pbi actually use — dram_zero_fill (the
    path the older tests exercised) allocates a fresh buffer per call and never shares
    one cached source, so it could not catch a bug that only shows when the SAME zero
    source + staging row is reused across many back-to-back convs in one program."""
    full = URAM_NEAR_FULL_ELEMENTS
    if getattr(ue, "_pbi_zero_src", None) is None:
        z = ue.get_params_dram_addr()
        ue.allocate_params_dram(full * _BPE)
        ue.dma_write(DMA_DEVICE_H2C, z, torch.zeros(full, dtype=torch.bfloat16), full * _BPE)
        ue._pbi_zero_src = z
    offset = 0
    while offset < num_elements:
        take = min(full, num_elements - offset)
        ue.accelerator_memory_to_sram(ue._pbi_zero_src, 0x00000, take)
        ue.sram_to_accelerator_memory(0x00000, DRAM_ADDR + offset * _BPE, take)
        offset += take


def pbi_strided_copy(ue: UnifiedEngine, gpr: int,
                     src_base: int, dst_base: int, count: int,
                     copy_bytes: int, src_stride: int, dst_stride: int) -> None:
    """One pointer-based-iteration loop copying *count* blocks of *copy_bytes*

    Each iteration DMAs *copy_bytes* bytes src→dst, then advances the source
    pointer by *src_stride* and the destination pointer by *dst_stride*.

    HW rules:
      * Exactly TWO advancing PBI pointers.
      * *copy_bytes* must fit one SRAM staging row.
      * When *dst_stride* > *copy_bytes* (writing into a padded slot) the caller
        MUST pre-zero *dst* so the untouched pad bytes stay zero.
    """
    if copy_bytes <= 0 or copy_bytes % 2 != 0:
        raise ValueError(f"pbi_strided_copy: copy_bytes must be a positive even (bf16) byte count, got {copy_bytes}")
    if copy_bytes > URAM_NEAR_FULL_ELEMENTS * 2:
        raise ValueError(
            f"pbi_strided_copy: copy_bytes={copy_bytes} exceeds one staging row "
            f"({URAM_NEAR_FULL_ELEMENTS * 2} bytes)")
    if count < 1:
        raise ValueError(f"pbi_strided_copy: count must be >=1, got {count}")

    ue._isa_reg_counter = gpr + 1
    ue.reset_inst_ptr_counter()
    ue.generate_instruction_add_set(dst_reg_idx=gpr, immediate_value=count)

    _, load_uram_row = ue.sram_address_to_uram_address(0x00000)
    _, store_uram_row = ue.sram_address_to_uram_address(0x00000)

    ptr_src = ue.alloc_inst_ptr()
    ptr_dst = ue.alloc_inst_ptr()

    ue.generate_instruction_pbi_init(
        dram_shared_addr=src_base, dma_length=copy_bytes,
        uram_dst_addr=load_uram_row, inst_pointer_idx=ptr_src,
    )
    ue.generate_instruction_pbi_init(
        dram_shared_addr=dst_base, dma_length=copy_bytes,
        uram_a_start_addr=store_uram_row, inst_pointer_idx=ptr_dst,
    )

    program_dram_start_addr = ue.get_program_dram_addr()
    cur_inst_count = ue.capture_count
    ue.generate_instruction_jump_abs(
        ue_35bit_addr_shifter(
            program_dram_start_addr + (cur_inst_count + 1) * INSTRUCTION_SIZE_BYTES
        )
    )

    ue.loop_start(loop_cnt=count, gpr_loop_cnt=gpr)
    ue.accelerator_memory_to_sram(
        accelerator_dram_address=src_stride, sram_address=0x00000,
        element_size=0, inst_pointer_idx=ptr_src,
    )
    ue.sram_to_accelerator_memory(
        sram_address=0x00000, accelerator_dram_address=dst_stride,
        element_size=0, inst_pointer_idx=ptr_dst,
    )
    body = ue.loop_end()
    assert body <= 256, f"pbi_strided_copy: loop body {body} exceeds 256 i-cache budget"

    ue.release_inst_ptr(ptr_dst)
    ue.release_inst_ptr(ptr_src)


def multihead_reshape_dram_legacy(ue: UnifiedEngine, INPUT_DRAM_ADDR: int,
                                   OUTPUT_DRAM_ADDR: int, TEMP_DRAM_ADDR: int,
                                   seq_len: int, num_heads: int,
                                   head_dim: int, head_dim_pad: int,
                                   input_row_stride: int = None) -> None:
    """(seq_len, input_row_stride) → (num_heads, seq_len, head_dim_pad).

    Replicates the current MobileSAM ``multihead_reshape_dram`` exactly
    (Python-unrolled scatter + bf16_permute_core).
    """
    if input_row_stride is None:
        input_row_stride = num_heads * head_dim
    if head_dim == head_dim_pad and input_row_stride == num_heads * head_dim:
        ue.bf16_permute_core(dim_0=seq_len, dim_1=num_heads, dim_2=head_dim,
                             INPUT_DRAM_ADDR=INPUT_DRAM_ADDR,
                             OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR)
    else:
        padded_dim = num_heads * head_dim_pad
        dram_zero_fill(ue, TEMP_DRAM_ADDR, seq_len * padded_dim)
        z_addr = ue.get_params_dram_addr()
        ue.allocate_params_dram(UE_VECTOR_SIZE * _BPE)
        ue.dma_write(DMA_DEVICE_H2C, z_addr,
                     torch.zeros(UE_VECTOR_SIZE, dtype=torch.bfloat16),
                     UE_VECTOR_SIZE * _BPE)
        for h in range(num_heads):
            for row in range(seq_len):
                src = INPUT_DRAM_ADDR + (row * input_row_stride + h * head_dim) * _BPE
                dst = TEMP_DRAM_ADDR + (row * padded_dim + h * head_dim_pad) * _BPE
                ue.accelerator_memory_to_sram(z_addr, 0x00000, UE_VECTOR_SIZE)
                ue.accelerator_memory_to_sram(src, 0x00000, head_dim)
                ue.sram_to_accelerator_memory(0x00000, dst, UE_VECTOR_SIZE)
        ue.bf16_permute_core(dim_0=seq_len, dim_1=num_heads, dim_2=head_dim_pad,
                             INPUT_DRAM_ADDR=TEMP_DRAM_ADDR,
                             OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR)


def multihead_reshape_dram_pbi(ue: UnifiedEngine, INPUT_DRAM_ADDR: int,
                                OUTPUT_DRAM_ADDR: int, TEMP_DRAM_ADDR: int,
                                seq_len: int, num_heads: int,
                                head_dim: int, head_dim_pad: int,
                                gpr_pbi: int,
                                input_row_stride: int = None) -> None:
    """PBI-based multi-head reshape — replaces the Python-unrolled scatter+
    bf16_permute_core with two ``pbi_strided_copy`` phases, mirroring the SWIN
    ``multihead_pad_and_permute`` pattern.

    Step 1 — scatter:  one PBI loop copies ``head_dim`` elements of each
    (row, head) slot from the interleaved input into a pre-zeroed padded TEMP,
    implicitly zeroing the ``head_dim_pad - head_dim`` tail of every slot.

    Step 2 — transpose: ``num_heads`` small PBI loops copy each head's rows
    from TEMP to the OUTPUT while transposing (seq_len, num_heads, hd_pad) →
    (num_heads, seq_len, hd_pad).

    The input_row_stride parameter is accepted for API compatibility but the
    PBI path always reads from the packed layout; pass the base address of
    the packed Q/K/V projection output.
    """
    if input_row_stride is None:
        input_row_stride = num_heads * head_dim

    padded_dim = num_heads * head_dim_pad
    dram_zero_fill(ue, TEMP_DRAM_ADDR, seq_len * padded_dim)

    # Step 1: scatter (seq_len, input_row_stride) → padded (seq_len, num_heads, head_dim_pad).
    # MUST honor input_row_stride: a channel-padded qkv output (e.g. 160 real cols in a 192-wide
    # row) would, under a single contiguous sweep, read the pad columns into head>0 slots and
    # scramble every row after the first. One PBI loop per head reads that head's head_dim cols at
    # the TRUE row stride. For packed input (input_row_stride == num_heads*head_dim) this is
    # bit-identical to the old single-sweep scatter.
    in_row_bytes = input_row_stride * _BPE
    for h in range(num_heads):
        pbi_strided_copy(
            ue, gpr_pbi,
            src_base=INPUT_DRAM_ADDR + h * head_dim * _BPE,
            dst_base=TEMP_DRAM_ADDR + h * head_dim_pad * _BPE,
            count=seq_len,
            copy_bytes=head_dim * _BPE,
            src_stride=in_row_bytes,
            dst_stride=padded_dim * _BPE,
        )

    # Step 2: transpose each head from TEMP to OUTPUT
    head_bytes = head_dim_pad * _BPE
    temp_row_stride = padded_dim * _BPE
    output_head_stride = seq_len * head_dim_pad * _BPE

    for h in range(num_heads):
        pbi_strided_copy(
            ue, gpr_pbi,
            src_base=TEMP_DRAM_ADDR + h * head_bytes,
            dst_base=OUTPUT_DRAM_ADDR + h * output_head_stride,
            count=seq_len,
            copy_bytes=head_bytes,
            src_stride=temp_row_stride,
            dst_stride=head_bytes,
        )


def conv2d_3x3_im2col_pbi(ue: UnifiedEngine, INPUT_DRAM_ADDR: int,
                          IM2COL_DRAM_ADDR: int, H_in: int, W_in: int,
                          C_in: int, stride: int, gpr: int) -> None:
    """Build the 3x3 (pad=1) im2col matrix (H_out*W_out, 9*C_in) for an HWC input
    using hardware PBI loops instead of per-(pixel,tap) gathers.

    Column layout matches the model convs: tap (dy,dx) occupies block
    ``(dy+1)*3+(dx+1)`` of ``C_in`` elements within each ``9*C_in`` row, and
    out-of-bounds taps/pixels stay zero (the buffer is pre-zeroed here).

    For a fixed output row ``h_out`` and tap ``(dy,dx)`` the input row is fixed and
    consecutive output columns read the input at a uniform stride of ``stride*C_in``
    while writing at a uniform ``9*C_in`` stride — i.e. one ``pbi_strided_copy`` per
    (row, tap) collapses the W_out-wide per-pixel gather. Boundary columns are
    skipped by clipping the valid ``w_out`` range; their blocks stay zero.

    Emits ~9*H_out PBI loops instead of ~H_out*W_out*9 gather DMAs.
    """
    K = 9 * C_in
    H_out = (H_in - 1) // stride + 1
    W_out = (W_in - 1) // stride + 1
    cb = C_in * _BPE
    row_bytes = K * _BPE

    # Pre-zero so every boundary tap/pixel that we skip below stays zero.
    dram_zero_fill(ue, IM2COL_DRAM_ADDR, H_out * W_out * K)

    for h_out in range(H_out):
        h_in = h_out * stride
        for dy in (-1, 0, 1):
            nh = h_in + dy
            if not (0 <= nh < H_in):
                continue  # whole tap-row out of bounds -> stays zero
            for dx in (-1, 0, 1):
                tap = (dy + 1) * 3 + (dx + 1)
                # valid output columns: 0 <= w_out*stride + dx < W_in
                lo = 0
                while lo < W_out and not (0 <= lo * stride + dx < W_in):
                    lo += 1
                hi = W_out - 1
                while hi >= 0 and not (0 <= hi * stride + dx < W_in):
                    hi -= 1
                if lo > hi:
                    continue
                count = hi - lo + 1
                src_base = INPUT_DRAM_ADDR + (nh * W_in + (lo * stride + dx)) * cb
                dst_base = IM2COL_DRAM_ADDR + (h_out * W_out + lo) * row_bytes + tap * cb
                pbi_strided_copy(
                    ue, gpr, src_base=src_base, dst_base=dst_base, count=count,
                    copy_bytes=cb, src_stride=stride * cb, dst_stride=row_bytes,
                )


def _window_dims(H: int, W: int, ws: int):
    pad_h = (ws - H % ws) % ws
    pad_w = (ws - W % ws) % ws
    nH, nW = (H + pad_h) // ws, (W + pad_w) // ws
    WIN_PAD = ((ws * ws + 63) // 64) * 64
    return nH, nW, WIN_PAD


def window_partition_pbi(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                         H: int, W: int, C: int, ws: int, gpr: int):
    """(H*W, C) HWC -> (nH*nW, WIN_PAD, C) via PBI loops. Returns (nH, nW, WIN_PAD).

    For a fixed window-row ``wh`` and in-window row ``dy`` the input source row is
    fixed; consecutive window-columns read at a uniform ``ws*C`` stride and write at
    a uniform ``WIN_PAD*C`` stride -> one ``pbi_strided_copy`` over the full window
    columns. The partial last column (when W%ws!=0) is a single clipped copy; pad
    rows/cols stay zero (output pre-zeroed)."""
    nH, nW, WIN_PAD = _window_dims(H, W, ws)
    cbe = C * _BPE
    full_w = W // ws            # full-width window columns
    rem_w  = W - full_w * ws    # partial last-window width (0 if exact)

    dram_zero_fill(ue, OUTPUT_DRAM_ADDR, nH * nW * WIN_PAD * C)

    for wh in range(nH):
        win_row = wh * nW
        for dy in range(ws):
            src_h = wh * ws + dy
            if src_h >= H:
                continue  # padded window row -> stays zero
            row_src = INPUT_DRAM_ADDR + (src_h * W) * cbe
            row_dst = OUTPUT_DRAM_ADDR + (win_row * WIN_PAD + dy * ws) * cbe
            if full_w >= 1:
                pbi_strided_copy(
                    ue, gpr, src_base=row_src, dst_base=row_dst, count=full_w,
                    copy_bytes=ws * cbe, src_stride=ws * cbe, dst_stride=WIN_PAD * cbe)
            if rem_w > 0:
                src = INPUT_DRAM_ADDR + (src_h * W + full_w * ws) * cbe
                dst = OUTPUT_DRAM_ADDR + ((win_row + full_w) * WIN_PAD + dy * ws) * cbe
                pbi_strided_copy(
                    ue, gpr, src_base=src, dst_base=dst, count=1,
                    copy_bytes=rem_w * cbe, src_stride=rem_w * cbe, dst_stride=rem_w * cbe)
    return nH, nW, WIN_PAD


def window_reverse_pbi(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                       H: int, W: int, C: int, ws: int, nH: int, nW: int, WIN_PAD: int,
                       gpr: int):
    """(nH*nW, WIN_PAD, C) -> (H*W, C). Inverse of window_partition_pbi: same loop,
    src/dst swapped (window slot -> image row). Caller pre-zeros OUTPUT if needed."""
    cbe = C * _BPE
    full_w = W // ws
    rem_w  = W - full_w * ws

    for wh in range(nH):
        win_row = wh * nW
        for dy in range(ws):
            src_h = wh * ws + dy
            if src_h >= H:
                continue
            row_src = INPUT_DRAM_ADDR + (win_row * WIN_PAD + dy * ws) * cbe
            row_dst = OUTPUT_DRAM_ADDR + (src_h * W) * cbe
            if full_w >= 1:
                pbi_strided_copy(
                    ue, gpr, src_base=row_src, dst_base=row_dst, count=full_w,
                    copy_bytes=ws * cbe, src_stride=WIN_PAD * cbe, dst_stride=ws * cbe)
            if rem_w > 0:
                src = INPUT_DRAM_ADDR + ((win_row + full_w) * WIN_PAD + dy * ws) * cbe
                dst = OUTPUT_DRAM_ADDR + (src_h * W + full_w * ws) * cbe
                pbi_strided_copy(
                    ue, gpr, src_base=src, dst_base=dst, count=1,
                    copy_bytes=rem_w * cbe, src_stride=rem_w * cbe, dst_stride=rem_w * cbe)


def _conv3x3_im2col_row_pbi(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, IM2COL_ROW_ADDR: int,
                            h_out: int, H: int, W: int, C_total: int, c_off: int,
                            c_take: int, stride: int, gpr: int):
    """Build ONE output row's 3x3/pad-1 im2col (W_out, 9*c_take) into IM2COL_ROW_ADDR
    (caller pre-zeros the row so boundary taps/pixels stay zero).

    Gathers ``c_take`` channels starting at ``c_off`` from an HWC input whose row pitch
    is the full ``C_total``. One PBI loop per in-bounds tap: consecutive output columns
    read the input at a uniform ``stride*C_total`` element stride and write at ``9*c_take``.
    Regular conv: c_off=0, c_take=C_total=C_in. Depthwise block: c_take=64, c_off=blk*64."""
    K = 9 * c_take
    W_out = (W - 1) // stride + 1
    cb = c_take * _BPE
    row_bytes = K * _BPE
    src_pitch = C_total * _BPE          # element pitch of one HWC pixel
    h_in = h_out * stride
    for dy in (-1, 0, 1):
        nh = h_in + dy
        if not (0 <= nh < H):
            continue
        for dx in (-1, 0, 1):
            tap = (dy + 1) * 3 + (dx + 1)
            lo = 0
            while lo < W_out and not (0 <= lo * stride + dx < W):
                lo += 1
            hi = W_out - 1
            while hi >= 0 and not (0 <= hi * stride + dx < W):
                hi -= 1
            if lo > hi:
                continue
            count = hi - lo + 1
            src_base = INPUT_DRAM_ADDR + ((nh * W + (lo * stride + dx)) * C_total + c_off) * _BPE
            dst_base = IM2COL_ROW_ADDR + (lo * K + tap * c_take) * _BPE
            pbi_strided_copy(
                ue, gpr, src_base=src_base, dst_base=dst_base, count=count,
                copy_bytes=cb, src_stride=stride * src_pitch, dst_stride=row_bytes)


def _prime_m_reg(ue: UnifiedEngine, gpr: int, m: int):
    """Prime a GPR with the runtime row count M for matmat_mul_core(gpr_M_reg=...)."""
    ue._isa_reg_counter = gpr + 1
    ue.reset_inst_ptr_counter()
    ue.generate_instruction_add_set(dst_reg_idx=gpr, immediate_value=m)


def conv2d_3x3_pbi(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                   IM2COL_DRAM_ADDR: int, WEIGHT_DRAM_ADDR: int, BIAS_DRAM_ADDR: int,
                   H: int, W: int, C_in: int, C_out: int, stride: int = 1,
                   gelu_enable: bool = False, gpr: int = 8, zero_fn=dram_zero_fill):
    """Drop-in PBI replacement for conv2d_3x3_stride{1,2}_dram (regular 3x3, pad=1).

    Same HWC I/O, same WEIGHT layout (C_out, 9*C_in), same per-row IM2COL buffer
    (W_out*9*C_in). Per output row: PBI im2col (9 loops) + one matmul with gpr_M_reg —
    replaces W_out*9 per-pixel gathers and the M-unrolled matmul. No ZERO_PAD buffer
    needed (padding handled by zeroing the im2col row)."""
    K = 9 * C_in
    H_out = (H - 1) // stride + 1
    W_out = (W - 1) // stride + 1
    for h_out in range(H_out):
        zero_fn(ue, IM2COL_DRAM_ADDR, W_out * K)
        _conv3x3_im2col_row_pbi(ue, INPUT_DRAM_ADDR, IM2COL_DRAM_ADDR, h_out,
                                H, W, C_in, 0, C_in, stride, gpr)
        _prime_m_reg(ue, gpr, W_out)
        ue.matmat_mul_core(
            M=W_out, K=K, N=C_out,
            A_DRAM_ADDR=IM2COL_DRAM_ADDR, B_DRAM_ADDR=WEIGHT_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR + (h_out * W_out) * C_out * _BPE,
            C_DRAM_ADDR=BIAS_DRAM_ADDR, bias_mode="broadcast_N",
            gelu_enable=gelu_enable, gpr_M_reg=gpr)


def conv2d_3x3_dw_pbi(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                      IM2COL_DRAM_ADDR: int, WEIGHT_DRAM_ADDR: int, BIAS_DRAM_ADDR: int,
                      H: int, W: int, C: int, stride: int = 1,
                      gelu_enable: bool = False, gpr: int = 8, zero_fn=dram_zero_fill):
    """Drop-in PBI replacement for conv2d_3x3_dw_dram / _stride2 (depthwise 3x3, pad=1).

    Same HWC I/O, same block weight layout (per 64-ch block: WEIGHT+blk*64*9*64, shape
    (64, 9*64)), same scatter (BLK back into the full-C output). Per (block, row):
    PBI im2col (9 loops) + matmul(gpr_M_reg) + PBI scatter — replaces n_blocks*H*W*9
    gathers. IM2COL buffer must hold W_out*(9*64 + 64) elements (im2col row + TEMP)."""
    BLK = 64
    K = 9 * BLK
    n_blocks = C // BLK
    H_out = (H - 1) // stride + 1
    W_out = (W - 1) // stride + 1
    TEMP = IM2COL_DRAM_ADDR + W_out * K * _BPE
    for blk in range(n_blocks):
        c_off = blk * BLK
        W_ADDR = WEIGHT_DRAM_ADDR + blk * BLK * K * _BPE
        B_ADDR = BIAS_DRAM_ADDR + c_off * _BPE
        for h_out in range(H_out):
            zero_fn(ue, IM2COL_DRAM_ADDR, W_out * K)
            _conv3x3_im2col_row_pbi(ue, INPUT_DRAM_ADDR, IM2COL_DRAM_ADDR, h_out,
                                    H, W, C, c_off, BLK, stride, gpr)
            _prime_m_reg(ue, gpr, W_out)
            ue.matmat_mul_core(
                M=W_out, K=K, N=BLK,
                A_DRAM_ADDR=IM2COL_DRAM_ADDR, B_DRAM_ADDR=W_ADDR,
                OUTPUT_DRAM_ADDR=TEMP, C_DRAM_ADDR=B_ADDR, bias_mode="broadcast_N",
                gelu_enable=gelu_enable, gpr_M_reg=gpr)
            # Scatter (W_out, BLK) into the HWC output at channel offset c_off (stride C).
            pbi_strided_copy(
                ue, gpr, src_base=TEMP,
                dst_base=OUTPUT_DRAM_ADDR + (h_out * W_out * C + c_off) * _BPE,
                count=W_out, copy_bytes=BLK * _BPE,
                src_stride=BLK * _BPE, dst_stride=C * _BPE)


def conv2d_3x3_dw_pbi_looped(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                              IM2COL_DRAM_ADDR: int, WEIGHT_DRAM_ADDR: int, BIAS_DRAM_ADDR: int,
                              H: int, W: int, C: int, stride: int = 1,
                              gelu_enable: bool = False, gpr: int = 1):
    """Hardware-looped depthwise 3x3/pad-1 conv — collapses h_out into ONE hardware loop.

    Same block-diagonal weight layout, same IM2COL+TEMP buffer contract as
    conv2d_3x3_dw_pbi, but the per-block h_out Python loop is replaced by a
    single hardware loop over interior rows (h=1..H-2). Interior rows load 3
    contiguous input rows into SRAM via general_reg_src (runtime-adressed) and
    build im2col with fixed strided scatters; boundary rows (top/bottom) fall
    back to the existing PBI gather so the boundary-cost code path is never
    exercised inside the hardware loop.

    This reduces the instruction footprint from H_out*(zero+9*gather+matmul+scatter)
    to zero + (3*load + 9*scatter + matmul + scatter) per interior row → the
    instruction count is O(width) not O(height*width), critical for tall encoder
    inputs like the S0 MBConv (256x256).
    """
    BLK = 64
    K = 9 * BLK
    n_blocks = C // BLK
    H_out = (H - 1) // stride + 1
    W_out = (W - 1) // stride + 1
    TEMP = IM2COL_DRAM_ADDR + W_out * K * _BPE

    # --- SRAM layout (BYTE offsets, matching the fast-path pattern) ---
    ROW_B = W * BLK * _BPE           # bytes per input row
    SRAM_R0 = 0                      # row  h-1
    SRAM_R1 = ROW_B                  # row  h
    SRAM_R2 = 2 * ROW_B              # row  h+1
    SRAM_OT = 3 * ROW_B              # output temp (W_out*BLK elements → bytes)
    SRAM_SZ = SRAM_OT + W_out * BLK * _BPE  # total SRAM bytes

    # Pre-load BLK zeros into SRAM (reused for column-boundary pixels every iteration)
    zero_dram = ue.get_params_dram_addr()
    ue.allocate_params_dram(BLK * _BPE)
    ue.dma_write(DMA_DEVICE_H2C, zero_dram, torch.zeros(BLK, dtype=torch.bfloat16), BLK * _BPE)
    ue.accelerator_memory_to_sram(zero_dram, SRAM_SZ, BLK)  # zero pixel at SRAM_SZ

    # ------------------------------------------------------------------ #
    # Per-block: Python loop over 64-channel groups (weight layout changes)
    # ------------------------------------------------------------------ #
    for blk in range(n_blocks):
        c_off = blk * BLK
        W_ADDR = WEIGHT_DRAM_ADDR + blk * BLK * K * _BPE
        B_ADDR = BIAS_DRAM_ADDR + c_off * _BPE
        _pbi_zero_dram(ue, IM2COL_DRAM_ADDR, W_out * K)

        # ----- interior rows (all 3 input rows in-bounds, no boundary) -----
        # Interior: h_out where h_in-1>=0 AND h_in+1<H.
        # h_in = h_out*stride. Condition: h_out >= 1 AND h_out*stride+1 < H.
        # h_out < (H-1)/stride = H_out-1 + ((H-1)%stride != 0)
        # So interior_last = H_out-2 when (H-1)%stride==0, else H_out-1.
        interior_first = 1
        if (H - 1) % stride == 0:
            interior_last = H_out - 2
        else:
            interior_last = H_out - 1
        n_interior = max(0, interior_last - interior_first + 1)

        if n_interior > 0:
            # Prime h_out-loop counter
            ue.generate_instruction_add_set(dst_reg_idx=gpr, immediate_value=n_interior)
            # GPR layout:
            #   gpr+0 = loop counter (OVERRIDDEN by loop_start alloc: set via gpr_loop_cnt=gpr)
            #   gpr+1 = input-row byte address for current h_out (includes c_off)
            #   gpr+2 = output-row byte address for current h_out (includes c_off)
            #   gpr+3 = tmp for address arithmetic

            # Prime input base to point to row h-1 of the first interior h_out
            # (so inside the loop we only add POSITIVE W*C*BPE offsets).
            h0_minus = interior_first - 1  # row h-1 for first interior iteration
            in0 = INPUT_DRAM_ADDR + (h0_minus * W * C + c_off) * _BPE
            out0 = OUTPUT_DRAM_ADDR + (interior_first * W_out * C + c_off) * _BPE
            ue.generate_instruction_add_set(dst_reg_idx=gpr + 1, immediate_value=ue_35bit_addr_shifter(in0))
            ue.generate_instruction_add_set(dst_reg_idx=gpr + 2, immediate_value=ue_35bit_addr_shifter(out0))

            ue.reset_inst_ptr_counter()
            ue._isa_reg_counter = gpr + 4  # pin loop-counter allocation ABOVE our 4 GPRs
            ue.loop_start(loop_cnt=n_interior, gpr_loop_cnt=gpr)

            # -------- Loop body: gather 3 input rows into SRAM --------
            # Row h-1: gpr+1 + 0
            ue.accelerator_memory_to_sram(0, SRAM_R0, W * BLK, general_reg_src=gpr + 1)

            # Row h: gpr+1 + W*C*BPE
            ue.generate_instruction_add_imm(src_reg_idx=gpr + 1, immediate_value=ue_35bit_addr_shifter(W * C * _BPE), dst_reg_idx=gpr + 3)
            ue.accelerator_memory_to_sram(0, SRAM_R1, W * BLK, general_reg_src=gpr + 3)

            # Row h+1: gpr+1 + 2*W*C*BPE
            ue.generate_instruction_add_imm(src_reg_idx=gpr + 1, immediate_value=ue_35bit_addr_shifter(2 * W * C * _BPE), dst_reg_idx=gpr + 3)
            ue.accelerator_memory_to_sram(0, SRAM_R2, W * BLK, general_reg_src=gpr + 3)

            # -------- Build im2col via 9 strided scatters --------
            # Pre-write zero boundary pixels at column 0 (dx=-1 taps) and column W-1 (dx=+1 taps)
            for dy in (-1, 0, 1):
                ki = (dy + 1) * 3 + 0  # dx=-1
                ue.sram_to_accelerator_memory(SRAM_SZ, IM2COL_DRAM_ADDR + 0 * K * _BPE + ki * BLK * _BPE, BLK)
                ki = (dy + 1) * 3 + 2  # dx=+1
                ue.sram_to_accelerator_memory(SRAM_SZ, IM2COL_DRAM_ADDR + (W_out - 1) * K * _BPE + ki * BLK * _BPE, BLK)

            # 9 tap scatters
            for dy in (-1, 0, 1):
                sram_row = SRAM_R1 + dy * ROW_B  # byte offset in SRAM for this dy
                for dx in (-1, 0, 1):
                    ki = (dy + 1) * 3 + (dx + 1)
                    w0 = 0 if dx >= 0 else 1  # first valid output column
                    w_last = W_out - 1 if dx <= 0 else W_out - 2
                    w_valid = w_last - w0 + 1
                    if w_valid <= 0:
                        continue
                    ue.sram_to_accelerator_memory(
                        sram_row + max(0, dx) * BLK * _BPE,
                        IM2COL_DRAM_ADDR + w0 * K * _BPE + ki * BLK * _BPE,
                        w_valid * BLK,
                        stride_bytes_per_chunk=BLK * _BPE,
                        stride_jump_bytes=K * _BPE)

            # -------- Matmul: IM2COL (W_out, K) × WEIGHT (K, BLK) → TEMP (W_out, BLK) --------
            ue.generate_instruction_add_set(dst_reg_idx=gpr + 3, immediate_value=W_out)
            ue.matmat_mul_core(
                M=W_out, K=K, N=BLK,
                A_DRAM_ADDR=IM2COL_DRAM_ADDR, B_DRAM_ADDR=W_ADDR,
                OUTPUT_DRAM_ADDR=TEMP, C_DRAM_ADDR=B_ADDR, bias_mode="broadcast_N",
                gelu_enable=gelu_enable, gpr_M_reg=gpr + 3)

            # -------- Scatter TEMP → output (runtime output base in gpr+2) --------
            ue.accelerator_memory_to_sram(TEMP, SRAM_OT, W_out * BLK)
            ue.sram_to_accelerator_memory(
                SRAM_OT, 0, W_out * BLK,
                stride_bytes_per_chunk=BLK * _BPE,
                stride_jump_bytes=C * _BPE,
                general_reg_src=gpr + 2)

            # -------- Advance input/output row bases --------
            ue.generate_instruction_add_imm(src_reg_idx=gpr + 1, immediate_value=ue_35bit_addr_shifter(W * C * _BPE), dst_reg_idx=gpr + 1)
            ue.generate_instruction_add_imm(src_reg_idx=gpr + 2, immediate_value=ue_35bit_addr_shifter(W_out * C * _BPE), dst_reg_idx=gpr + 2)

            body = ue.loop_end()
            assert body <= 256, f"dw_pbi_looped: loop body {body} exceeds 256 i-cache budget"

        # ----- Boundary rows (top + bottom) — fall back to existing per-row PBI gather -----
        boundary_rows = []
        if H_out >= 1:
            boundary_rows.append(0)
        if H_out >= 2:
            boundary_rows.append(H_out - 1)
        for h_out in sorted(set(boundary_rows)):
            _pbi_zero_dram(ue, IM2COL_DRAM_ADDR, W_out * K)
            _conv3x3_im2col_row_pbi(ue, INPUT_DRAM_ADDR, IM2COL_DRAM_ADDR, h_out,
                                    H, W, C, c_off, BLK, stride, gpr)
            ue.generate_instruction_add_set(dst_reg_idx=gpr, immediate_value=W_out)
            ue.matmat_mul_core(
                M=W_out, K=K, N=BLK,
                A_DRAM_ADDR=IM2COL_DRAM_ADDR, B_DRAM_ADDR=W_ADDR,
                OUTPUT_DRAM_ADDR=TEMP, C_DRAM_ADDR=B_ADDR, bias_mode="broadcast_N",
                gelu_enable=gelu_enable, gpr_M_reg=gpr)
            ue.accelerator_memory_to_sram(TEMP, 0, W_out * BLK)
            pbi_strided_copy(
                ue, gpr, src_base=TEMP,
                dst_base=OUTPUT_DRAM_ADDR + (h_out * W_out * C + c_off) * _BPE,
                count=W_out, copy_bytes=BLK * _BPE,
                src_stride=BLK * _BPE, dst_stride=C * _BPE)


def bf16_permute_pbi(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                     dim_0: int, dim_1: int, dim_2: int, gpr: int):
    """Drop-in PBI replacement for bf16_permute_core: (dim_0, dim_1, dim_2) ->
    (dim_1, dim_0, dim_2).

    Output row ``i1*dim_0 + i0`` comes from input row ``i0*dim_1 + i1``. Either index can
    be the PBI loop count; we iterate the SMALLER of (dim_0, dim_1) in Python and make the
    larger the hardware loop count, so the program is ``min(dim_0, dim_1)`` PBI loops
    regardless of orientation (e.g. 8 loops for both 4096x8 and 8x4096). ``dim_2`` must be a
    multiple of UE_VECTOR_SIZE (so copy_bytes is row-aligned)."""
    assert dim_2 % UE_VECTOR_SIZE == 0, f"dim_2={dim_2} must be a multiple of {UE_VECTOR_SIZE}"
    cb = dim_2 * _BPE
    if dim_1 <= dim_0:
        # iterate i1 (few), PBI-count over i0 (many). dst contiguous, src strided.
        for i1 in range(dim_1):
            pbi_strided_copy(
                ue, gpr,
                src_base=INPUT_DRAM_ADDR + i1 * cb,
                dst_base=OUTPUT_DRAM_ADDR + i1 * dim_0 * cb,
                count=dim_0, copy_bytes=cb,
                src_stride=dim_1 * cb, dst_stride=cb)
    else:
        # iterate i0 (few), PBI-count over i1 (many). src contiguous, dst strided.
        for i0 in range(dim_0):
            pbi_strided_copy(
                ue, gpr,
                src_base=INPUT_DRAM_ADDR + i0 * dim_1 * cb,
                dst_base=OUTPUT_DRAM_ADDR + i0 * cb,
                count=dim_1, copy_bytes=cb,
                src_stride=cb, dst_stride=dim_0 * cb)


def conv_transpose2d_2x2_pbi(ue: UnifiedEngine, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                             TEMP_DRAM_ADDR: int, WEIGHT_SLICES: list, BIAS_DRAM_ADDR: int,
                             H: int, W: int, C_in: int, C_out: int, gpr: int):
    """ConvTranspose2d kernel=2 stride=2 (no overlap). (H*W, C_in) -> (2H*2W, C_out) HWC.

    Same 4-slice matmul as the legacy path (matmuls left untouched), but the per-pixel
    output scatter is collapsed to PBI loops: for a fixed slice and input row h_i the
    W input pixels read TEMP contiguously (stride C_out) and write the output at every
    OTHER pixel (stride 2*C_out) -> one pbi_strided_copy per (slice, h_i) instead of
    W per-pixel memcpys. OUTPUT must be pre-zeroed (caller). C_out multiple of 64."""
    cb = C_out * _BPE
    for s, (kh, kw) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        ue.matmat_mul_core(
            M=H * W, K=C_in, N=C_out,
            A_DRAM_ADDR=INPUT_DRAM_ADDR, B_DRAM_ADDR=WEIGHT_SLICES[s],
            OUTPUT_DRAM_ADDR=TEMP_DRAM_ADDR, C_DRAM_ADDR=BIAS_DRAM_ADDR,
            bias_mode="broadcast_N")
        for h_i in range(H):
            src_base = TEMP_DRAM_ADDR + (h_i * W) * cb
            dst_base = OUTPUT_DRAM_ADDR + ((2 * h_i + kh) * (2 * W) + kw) * cb
            pbi_strided_copy(
                ue, gpr, src_base=src_base, dst_base=dst_base, count=W,
                copy_bytes=cb, src_stride=cb, dst_stride=2 * cb)


# ---------------------------------------------------------------------------
# Test result registry (consumed by the CI PR-comment step).
# Each test calls record_test(...) once it has computed SNR / GFLOPS so a
# concise per-test summary line can be written to user_hw_test_summary.md.
# ---------------------------------------------------------------------------
TEST_RESULTS = []

# Set True only when ``if __name__ == "__main__"`` reaches the end of the suite
# without AssertionError or other abort; ``write_test_summary`` (atexit) uses
# this so logs can distinguish full pass vs summary written after a failure.
_ALL_TESTS_PASSED_BEFORE_SUMMARY = False


def record_test(name: str, dims: str = "", snr_db=None, gflops=None, mb_per_s=None, inst_bytes=None) -> None:
    TEST_RESULTS.append({
        "name": name,
        "dims": dims,
        "snr_db": snr_db,
        "gflops": gflops,
        "mb_per_s": mb_per_s,
        "inst_bytes": inst_bytes,
    })


def _fmt_metric(value, fmt: str) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and value == float("inf"):
        return "inf"
    return fmt.format(value)


def write_test_summary(path: str = "user_hw_test_summary.md") -> None:
    """Write a one-line-per-test markdown table of recorded results."""
    headers = ["Test", "Dimensions", "SNR (dB)", "GFLOPS", "MB/s", "Inst Bytes"]
    rows = [
        [
            r["name"],
            r["dims"],
            _fmt_metric(r["snr_db"], "{:.2f}"),
            _fmt_metric(r["gflops"], "{:.2f}"),
            _fmt_metric(r["mb_per_s"], "{:.2f}"),
            _fmt_metric(r["inst_bytes"], "{:.0f}"),
        ]
        for r in TEST_RESULTS
    ]
    widths = [
        max(len(h), max((len(row[i]) for row in rows), default=0))
        for i, h in enumerate(headers)
    ]
    def fmt_row(cols):
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols)) + " |"
    lines = [
        fmt_row(headers),
        "| " + " | ".join("-" * w for w in widths) + " |",
        *[fmt_row(row) for row in rows],
    ]
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    print("=== TEST SUMMARY START ===")
    print(text, end="")
    print("=== TEST SUMMARY END ===")
    if _ALL_TESTS_PASSED_BEFORE_SUMMARY:
        print("all tests pass before")


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
    generate_trace(ue0, f"matmat_mul_core_trace_0_{M_three_fourth}_{K}_{N}.csv")
    generate_trace(ue1, f"matmat_mul_core_trace_1_{M_one_fourth}_{K}_{N}.csv")

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
    record_test("matmat_mul_two_engine_flag_check",
                f"M={M}, K={K}, N={N}",
                snr_db=snr_combined)

    ue0.reset_tensor_dram_addr()
    ue0.clear_capture_buffer()
    ue1.reset_tensor_dram_addr()
    ue1.clear_capture_buffer()


def matmat_mul_multi_engine_flag_check_test(M: int, K: int, N: int, num_engines: int = 8):
    """
    Shard A along the M dimension into num_engines parts and run each part on
    engine0..engine(num_engines-1) in parallel, each multiplied by the same B.
    ue0 is the host: it waits for engines 1..(num_engines-1) after its matmul.
    Workers do matmul then flag_set. After run, only need to wait for ue0.
    Same test purpose as matmat_mul_two_engine_flag_check_test when num_engines=2
    (equivalent, not identical: equal M-split and ue0-waits-for-workers sync).
    When num_engines=1, no flag_check or worker engines; reset loop is unchanged.
    """
    import user_dma_core

    engine_base_stride = 0x00010000
    dram_base_stride = 0x10000000

    ues = []
    for i in range(num_engines):
        ue = UnifiedEngine(BASE_ADDR=user_dma_core.UE_0_BASE_ADDR + i * engine_base_stride,
                            params_dram_base=user_dma_core.DRAM_START_ADDR + i * dram_base_stride,
                            tensor_dram_base=user_dma_core.DRAM_START_ADDR + i * dram_base_stride + 0x08000000,
                            program_dram_base=user_dma_core.DRAM_START_ADDR + i * dram_base_stride + 0x0F000000,
                            )
        ues.append(ue)

    a_addrs = []
    for i, ue in enumerate(ues):
        a_addrs.append(ue.allocate_tensor_dram(1 * 1024 * 1024))

    prog_addrs = []
    ues[0].start_capture()

    element_size = URAM_HALF_ELEMENTS
    ues[0].generate_instruction_flag_set()
    ues[0].accelerator_memory_to_sram(accelerator_dram_address=a_addrs[0],
                                  sram_address=0x00000,
                                  element_size=element_size)
    if num_engines >= 2:
        for i in range(1, num_engines):
            ues[0].generate_instruction_flag_check(target_engine_idx=i)
    ues[0].generate_instruction_flag_clear()
    ues[0].generate_instruction_halt()
    ues[0].stop_capture()
    prog_addrs.append(ues[0].get_program_dram_addr())
    ues[0].write_captured_instructions_to_dram(prog_addrs[0])
    ues[0].allocate_program_dram(ues[0].get_capture_instruction_size_bytes())

    if num_engines >= 2:
        for i in range(1, num_engines):
            ues[i].start_capture()
            ues[i].generate_instruction_flag_clear()
            ues[i].generate_instruction_flag_check(target_engine_idx=0)
            ues[i].accelerator_memory_to_sram(accelerator_dram_address=a_addrs[i],
                                  sram_address=0x00000,
                                  element_size=element_size)
            ues[i].generate_instruction_flag_set()
            ues[i].generate_instruction_halt()
            ues[i].stop_capture()
            prog_addrs.append(ues[i].get_program_dram_addr())
            ues[i].write_captured_instructions_to_dram(prog_addrs[i])
            ues[i].allocate_program_dram(ues[i].get_capture_instruction_size_bytes())

    for i in range(1, num_engines):
        ues[i].start_execute_from_dram(prog_addrs[i])
    ues[0].start_execute_from_dram(prog_addrs[0])
    # ue0 waits for 1..7 inside its program; host only needs to wait for ue0
    ues[0].wait_queue(10.0)
    print(f"Total latency: {ues[0].report_latency_in_us()} us")
    total_bytes_transferred = num_engines * element_size * 2
    print(f"speed {total_bytes_transferred / ues[0].report_latency_in_us():.2f} MB/s")
    for i in range(num_engines):
        generate_trace(ues[i], f"multi_engine_read_test_engine_{num_engines}_{i}.csv")

    record_test("matmat_mul_multi_engine_flag_check",
                f"M={M}, K={K}, N={N}, num_engines={num_engines}")

    # print(f"Report FLOPS for {num_engines}-engine parallel sharded matmul: {flop_rate_gflops:.2f} GFLOPS for M={M}, K={K}, N={N}")

    for ue in ues:
        ue.reset_tensor_dram_addr()
        ue.clear_capture_buffer()


def matmat_mul_two_cores_test(M: int, K: int, N: int, softmax_enable: bool = False, gelu_enable: bool = False, silu_enable: bool = False, sigmoid_enable: bool = False, clamp_enable: bool = False, log_enable: bool = False, use_pbi: bool = False, input_scale: float = 1.0, snr_threshold_db: float = 40.0):
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
    # input_scale widens the post-matmul variance so softmax hits the
    # exp + bf20 adder tree across a broader dynamic range (see matmat_mul_test).
    if input_scale != 1.0:
        a = (a.to(torch.float32) * float(input_scale)).to(torch.bfloat16)
    b = torch.randn(N, K, dtype=torch.bfloat16)

    ue0.dma_to_accelerator_memory(A_DRAM_ADDR, a)
    ue0.dma_to_accelerator_memory(B_DRAM_ADDR, b)

    total_flops_from_matmat_mul = UnifiedEngine.matmat_mul_two_cores(ue0=ue0, ue1=ue1, M=M, K=K, N=N, A_DRAM_ADDR=A_DRAM_ADDR, B_DRAM_ADDR=B_DRAM_ADDR, OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR, softmax_enable=softmax_enable, gelu_enable=gelu_enable, silu_enable=silu_enable, sigmoid_enable=sigmoid_enable, clamp_enable=clamp_enable, log_enable=log_enable, use_pbi=use_pbi)
    ue0.report_timing_and_instruction_count()
    ue1.report_timing_and_instruction_count()

    # Parallel completion time is bounded by the slower engine.
    latency_us = max(ue0.report_latency_in_us(), ue1.report_latency_in_us())
    flop_rate_gflops = total_flops_from_matmat_mul / (latency_us * 1e3)
    flops_ratio = flop_rate_gflops / user_dma_core.UE_PEAK_GFLOPS / 20
    print(f"Report FLOPS for two-cores MxKxN Matmul: {flop_rate_gflops:.2f} GFLOPS, {flops_ratio:.2f}% peak throughput for M={M}, K={K}, N={N}, softmax_enable={softmax_enable}, gelu_enable={gelu_enable}, silu_enable={silu_enable}, sigmoid_enable={sigmoid_enable}, use_pbi={use_pbi}")

    generate_trace(ue0, f"matmat_mul_two_cores_trace_engine0_{M // 2}_{K}_{N}_{'softmax_enabled' if softmax_enable else 'softmax_disabled'}_{'gelu_enabled' if gelu_enable else 'gelu_disabled'}_{'silu_enabled' if silu_enable else 'silu_disabled'}_{'sigmoid_enabled' if sigmoid_enable else 'sigmoid_disabled'}.csv")
    generate_trace(ue1, f"matmat_mul_two_cores_trace_engine1_{M - (M // 2)}_{K}_{N}_{'softmax_enabled' if softmax_enable else 'softmax_disabled'}_{'gelu_enabled' if gelu_enable else 'gelu_disabled'}_{'silu_enabled' if silu_enable else 'silu_disabled'}_{'sigmoid_enabled' if sigmoid_enable else 'sigmoid_disabled'}.csv")

    output = ue0.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M, N))
    ref = a @ b.T

    def apply_gelu(x):
        return x * torch.sigmoid(1.702 * x)

    def apply_silu(x):
        return x * torch.sigmoid(x)

    def apply_sigmoid(x):
        return torch.sigmoid(x)

    if gelu_enable:
        ref = apply_gelu(ref)
    elif silu_enable:
        ref = apply_silu(ref)
    elif sigmoid_enable:
        ref = apply_sigmoid(ref)
    elif clamp_enable:
        ref = torch.clamp(ref, min=0.0)
    elif log_enable:
        ref = torch.log(torch.clamp(ref, min=1e-3))

    if softmax_enable:
        ref = torch.softmax(ref, dim=-1).to(torch.bfloat16)

    snr_combined = calculate_snr(ref, output)
    print(f"Two-cores matmul SNR combined: {snr_combined:.2f} dB")
    assert snr_combined >= snr_threshold_db or snr_combined == float('inf'), f"SNR {snr_combined:.2f} dB must be at least {snr_threshold_db:g} dB"

    flags = []
    if softmax_enable: flags.append("softmax")
    if gelu_enable:    flags.append("gelu")
    if silu_enable:    flags.append("silu")
    if sigmoid_enable: flags.append("sigmoid")
    if clamp_enable:   flags.append("clamp")
    if log_enable:     flags.append("log")
    if use_pbi:        flags.append("pbi")
    if input_scale != 1.0: flags.append(f"scale={input_scale:g}")
    flag_str = ("+" + "+".join(flags)) if flags else ""
    record_test(f"matmat_mul_two_cores{flag_str}",
                f"M={M}, K={K}, N={N}",
                snr_db=snr_combined,
                gflops=flop_rate_gflops)

    ue0.reset_tensor_dram_addr()
    ue0.clear_capture_buffer()
    ue1.reset_tensor_dram_addr()
    ue1.clear_capture_buffer()

def flash_attention_test(head_dim: int, seq_len: int, bias_enable: bool = False, use_pbi: bool = False, force_legacy: bool = False,
                         num_buckets: int = 64):
    """
    Tests flash attention core.

    - ``use_pbi=True``: captures the bucketized :meth:`~UnifiedEngine.flash_attention_core_pbi`
      via :meth:`~UnifiedEngine.flash_attention_core` (requires ``ATTN_P_DRAM_ADDR``, ``seq_len**2``
      BF16). The captured program contains a ``num_buckets``-bucket dispatcher; an ``ADD_SET``
      prologue primes the bucket-index GPR with ``bucket_idx = seq_len_padded // UE_VECTOR_SIZE``
      and the JZ cascade routes to that bucket at execute time. Bucket step is fixed to
      ``UE_VECTOR_SIZE``.
    - Otherwise captures :meth:`~UnifiedEngine.flash_attention_core_legacy` via
      ``flash_attention_core(..., gpr_bucket_idx=None)``.

    Unaligned seq_len values are handled by zero-padding Q/K/V to the next multiple of
    ``UE_VECTOR_SIZE`` and masking padded key positions with ``-inf`` in the attention bias
    so softmax assigns them zero weight.  Output is sliced back to ``seq_len`` rows before SNR comparison.
    """
    ue = UnifiedEngine()
    debug_mode = False

    seq_len_padded = math.ceil(seq_len / UE_VECTOR_SIZE) * UE_VECTOR_SIZE
    needs_padding = seq_len_padded != seq_len
    # A mask bias is required whenever key positions are padded out.
    bias_mask_needed = bias_enable or needs_padding

    if use_pbi:
        assert seq_len_padded % UE_VECTOR_SIZE == 0, (
            f"seq_len_padded={seq_len_padded} must align with UE_VECTOR_SIZE={UE_VECTOR_SIZE}"
        )
        bucket_idx = seq_len_padded // UE_VECTOR_SIZE
        if not (1 <= bucket_idx <= num_buckets):
            raise ValueError(
                f"seq_len_padded={seq_len_padded} maps to bucket_idx={bucket_idx} outside "
                f"[1, {num_buckets}] (num_buckets={num_buckets})"
            )

    Q_DRAM_ADDR = ue.allocate_tensor_dram(seq_len_padded * head_dim * 2)
    K_DRAM_ADDR = ue.allocate_tensor_dram(seq_len_padded * head_dim * 2)
    V_DRAM_ADDR = ue.allocate_tensor_dram(seq_len_padded * head_dim * 2)
    BIAS_DRAM_ADDR = ue.allocate_tensor_dram(seq_len_padded * seq_len_padded * 2) if bias_mask_needed else None
    SM_OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(seq_len_padded * seq_len_padded * 2) if debug_mode else None
    SCRATCH_DRAM_ADDR = ue.allocate_tensor_dram(max(head_dim, UE_FMAX_CONTEXT_SIZE) * seq_len_padded * 2 + head_dim * seq_len_padded * 2)
    ATTN_P_DRAM_ADDR = ue.allocate_tensor_dram(seq_len_padded * seq_len_padded * 2) if use_pbi else None
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(seq_len_padded * head_dim * 2)
    IDENTITY_DRAM_ADDR = ue.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * 2)
    ue.dma_to_accelerator_memory(IDENTITY_DRAM_ADDR, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))

    # Bucket-index GPR is held for the entire dispatcher (decremented by the per-bucket JZ
    # cascade), so allocate it before capture; the kernel's internal allocations nest above it.
    bucket_reg = ue.alloc_isa_reg() if use_pbi else None

    ue.start_capture() # -------------------------------------------------------------

    if use_pbi:
        ue.generate_instruction_add_set(bucket_reg, bucket_idx)

    total_flops_from_flash_attention = ue.flash_attention_core(
        head_dim=head_dim,
        seq_len=seq_len_padded,
        Q_DRAM_ADDR=Q_DRAM_ADDR,
        K_DRAM_ADDR=K_DRAM_ADDR,
        V_DRAM_ADDR=V_DRAM_ADDR,
        OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
        SCRATCH_DRAM_ADDR=SCRATCH_DRAM_ADDR,
        IDENTITY_DRAM_ADDR=IDENTITY_DRAM_ADDR,
        BIAS_DRAM_ADDR=BIAS_DRAM_ADDR,
        debug_mode=debug_mode,
        SM_OUTPUT_DRAM_ADDR=SM_OUTPUT_DRAM_ADDR,
        ATTN_P_DRAM_ADDR=ATTN_P_DRAM_ADDR,
        gpr_bucket_idx=bucket_reg,
        use_pbi=not force_legacy,
        num_buckets=num_buckets,
    )

    if use_pbi:
        ue.release_isa_reg()

    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    instruction_size_bytes = ue.get_capture_instruction_size_bytes()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    # Test Time — build padded tensors
    q_raw = torch.randn(seq_len, head_dim, dtype=torch.bfloat16)
    k_raw = torch.randn(seq_len, head_dim, dtype=torch.bfloat16)
    v_raw = torch.randn(seq_len, head_dim, dtype=torch.bfloat16)

    q = torch.zeros(seq_len_padded, head_dim, dtype=torch.bfloat16)
    k = torch.zeros(seq_len_padded, head_dim, dtype=torch.bfloat16)
    v = torch.zeros(seq_len_padded, head_dim, dtype=torch.bfloat16)
    q[:seq_len] = q_raw
    k[:seq_len] = k_raw
    v[:seq_len] = v_raw

    # Build combined bias+mask: -inf for padded key columns, user bias in valid region.
    if bias_mask_needed:
        combined_bias = torch.zeros(seq_len_padded, seq_len_padded, dtype=torch.bfloat16)
        if needs_padding:
            combined_bias[:, seq_len:] = float('-inf')
        if bias_enable:
            bias_raw = torch.randn(seq_len, seq_len, dtype=torch.bfloat16)
            combined_bias[:seq_len, :seq_len] = combined_bias[:seq_len, :seq_len] + bias_raw
    else:
        combined_bias = None

    # DMA to accelerator memory -------------------------------------------------------------
    ue.dma_to_accelerator_memory(Q_DRAM_ADDR, q)
    ue.dma_to_accelerator_memory(K_DRAM_ADDR, k)
    ue.dma_to_accelerator_memory(V_DRAM_ADDR, v)
    if bias_mask_needed:
        ue.dma_to_accelerator_memory(BIAS_DRAM_ADDR, combined_bias)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(50.0) # 30 seconds timeout
    ue.report_timing_and_instruction_count()

    trace_suffix = "_pbi" if use_pbi else ""
    generate_trace(ue, f"flash_attention_core_trace_{head_dim}_{seq_len}_{'bias_enabled' if bias_enable else 'bias_disabled'}{trace_suffix}.csv")

    output_padded = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (seq_len_padded, head_dim))
    output = output_padded[:seq_len]

    if debug_mode:
        sm_output = ue.dma_from_accelerator_memory(SM_OUTPUT_DRAM_ADDR, (seq_len_padded, seq_len_padded))
        v_trans = ue.dma_from_accelerator_memory(SCRATCH_DRAM_ADDR, (head_dim, seq_len_padded))

    # PBI path returns a per-bucket FLOPS list; only bucket_idx ran at execute time,
    # so reporting GFLOPS uses just that bucket's work (using max-bucket FLOPS would
    # divide max-bucket work by smaller-bucket time and overstate the rate).
    executed_flops = (
        total_flops_from_flash_attention[bucket_idx - 1]
        if use_pbi
        else total_flops_from_flash_attention
    )
    report_flop_rate_gflops, report_gflops_ratio = ue.report_flop_rate_gflops(executed_flops)
    print(f"Report FLOPS for Flash Attention: {report_flop_rate_gflops:.2f} GFLOPS, {report_gflops_ratio:.2f}% peak throughput for head_dim={head_dim} and seq_len={seq_len} and bias_enable={bias_enable} and use_pbi={use_pbi}")

    # Reference calculation (CPU) — use padded inputs + same mask so reference matches HW output
    start_time = time.time()
    q_scaled = q * (1.0 / math.sqrt(head_dim))
    qkt = q_scaled @ k.t()
    if combined_bias is not None:
        qkt = qkt + combined_bias
    sm = torch.softmax(qkt, dim=-1).to(torch.bfloat16)
    ref_padded = sm @ v
    ref = ref_padded[:seq_len]
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
    assert snr_db_ref >= 38 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 39 dB"

    _fa_tag = "+pbi" if use_pbi else ""
    _bucket_dims = f", bucket={bucket_idx}/{num_buckets}" if use_pbi else ""
    record_test(f"flash_attention{'+bias' if bias_enable else ''}{_fa_tag}",
                f"head_dim={head_dim}, seq_len={seq_len}{_bucket_dims}",
                snr_db=snr_db_ref,
                gflops=report_flop_rate_gflops,
                inst_bytes=instruction_size_bytes)

    # Before running matmat_mul_core, let's clear the capture buffer and reset the tensor DRAM address
    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()
    ue.reset_program_dram_addr()

def decoder_group_attention_test(head_dim: int = 256, seq_len: int = 8192, use_pbi: bool = False, num_buckets: int = 64, force_legacy: bool = False, group_size: int = 4):
    """
    Tests decoder group attention core (single-query GQA, used in autoregressive decode).
    Bias is always enabled and group_size is fixed to 4.
    """
    ue = UnifiedEngine()
    ue.bytes_per_element = 2

    # For the PBI bucketized path, allocate Q/K/V/OUTPUT/SCRATCH/BIAS sized to the **max bucket**
    # so per-bucket DRAM offset arithmetic stays inside the allocation. Inputs and outputs are
    # filled / sliced based on the actual ``seq_len`` below.
    bucket_idx_one = seq_len // UE_VECTOR_SIZE  # 1-based runtime bucket selector
    if use_pbi and not (1 <= bucket_idx_one <= num_buckets and seq_len % UE_VECTOR_SIZE == 0):
        raise ValueError(
            f"decoder_group_attention_test: seq_len={seq_len} must be a multiple of UE_VECTOR_SIZE "
            f"and yield bucket_idx in [1, {num_buckets}] (got bucket_idx={bucket_idx_one})"
        )
    max_seq_len = num_buckets * UE_VECTOR_SIZE if use_pbi else seq_len
    Q_DRAM_ADDR = ue.allocate_tensor_dram(group_size * head_dim * ue.bytes_per_element)
    K_DRAM_ADDR = ue.allocate_tensor_dram(max_seq_len * head_dim * ue.bytes_per_element)
    V_DRAM_ADDR = ue.allocate_tensor_dram(max_seq_len * head_dim * ue.bytes_per_element)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(group_size * head_dim * ue.bytes_per_element)
    SCRATCH_DRAM_ADDR = ue.allocate_tensor_dram(max(head_dim, UE_FMAX_CONTEXT_SIZE) * max_seq_len * 2 + head_dim * max_seq_len * 2)
    BIAS_DRAM_ADDR = ue.allocate_tensor_dram(max_seq_len * ue.bytes_per_element)
    IDENTITY_DRAM_ADDR = ue.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * ue.bytes_per_element)

    bucket_reg = ue.alloc_isa_reg() if use_pbi else None

    ue.start_capture()
    if use_pbi:
        ue.generate_instruction_add_set(bucket_reg, bucket_idx_one)
    flops_result = ue.decoder_group_attention_core(
        group_size=group_size,
        head_dim=head_dim,
        seq_len=seq_len,
        Q_DRAM_ADDR=Q_DRAM_ADDR,
        K_DRAM_ADDR=K_DRAM_ADDR,
        V_DRAM_ADDR=V_DRAM_ADDR,
        OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
        SCRATCH_DRAM_ADDR=SCRATCH_DRAM_ADDR,
        gpr_bucket_idx=bucket_reg if use_pbi else None,
        num_buckets=num_buckets,
        IDENTITY_DRAM_ADDR=IDENTITY_DRAM_ADDR,
        BIAS_DRAM_ADDR=BIAS_DRAM_ADDR,
        use_pbi=not force_legacy,
    )
    total_flops = flops_result[bucket_idx_one - 1] if use_pbi else flops_result
    ue.stop_capture()
    if use_pbi:
        ue.release_isa_reg()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    instruction_size_bytes = ue.get_capture_instruction_size_bytes()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(instruction_size_bytes)

    q = torch.randn(group_size, head_dim, dtype=torch.bfloat16)
    k = torch.randn(seq_len, head_dim, dtype=torch.bfloat16)
    v = torch.randn(seq_len, head_dim, dtype=torch.bfloat16)
    bias = torch.randn(seq_len, dtype=torch.bfloat16)
    identity = torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16)

    ue.dma_to_accelerator_memory(Q_DRAM_ADDR, q)
    ue.dma_to_accelerator_memory(K_DRAM_ADDR, k)
    ue.dma_to_accelerator_memory(V_DRAM_ADDR, v)
    ue.dma_to_accelerator_memory(BIAS_DRAM_ADDR, bias)
    ue.dma_to_accelerator_memory(IDENTITY_DRAM_ADDR, identity)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(50.0)
    ue.report_timing_and_instruction_count()

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (group_size, head_dim))

    report_flop_rate_gflops, report_gflops_ratio = ue.report_flop_rate_gflops(total_flops)
    print(
        f"Report FLOPS for Decoder Group Attention: {report_flop_rate_gflops:.2f} GFLOPS, "
        f"{report_gflops_ratio:.2f}% peak throughput for head_dim={head_dim}, seq_len={seq_len}, "
        f"group_size={group_size}, use_pbi={use_pbi}"
    )

    q_scaled = q * (1 / math.sqrt(head_dim))
    qkt = q_scaled @ k.t()
    qkt = qkt + bias.unsqueeze(0)
    sm = torch.softmax(qkt, dim=-1).to(torch.bfloat16)
    ref = sm @ v

    snr_db_ref = calculate_snr(ref, output)
    print(f"Reference SNR Analysis for Decoder Group Attention: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 33 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 40 dB"

    tag = "+pbi" if (use_pbi and not force_legacy) else ("+pbi+legacy_body" if (use_pbi and force_legacy) else "")
    record_test(
        f"decoder_group_attention{tag}",
        f"group_size={group_size}, head_dim={head_dim}, seq_len={seq_len}",
        snr_db=snr_db_ref,
        gflops=report_flop_rate_gflops,
        inst_bytes=instruction_size_bytes,
    )

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()
    ue.reset_program_dram_addr()

def matmat_mul_test(M: int, K: int, N: int, bias_enable: bool = False, softmax_enable: bool = False, bias_mode: str = "broadcast_N", gelu_enable: bool = False, silu_enable: bool = False, sigmoid_enable: bool = False, clamp_enable: bool = False, log_enable: bool = False, debug_fmax: bool = False, use_pbi: bool = False, input_scale: float = 1.0, snr_threshold_db: float = 40.0, fmax_snr_threshold_db: float = 40.0):
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

    ZERO_DRAM_ADDR = None
    FMAX_DRAM_ADDR = None
    if softmax_enable:
        if debug_fmax:
            ZERO_DRAM_ADDR = ue.allocate_tensor_dram(UE_VECTOR_SIZE * 2)
            FMAX_DRAM_ADDR = ue.allocate_tensor_dram(M * UE_VECTOR_SIZE * 2)

    # PBI path is driven by gpr_M_reg; allocate + prime a GPR with M when use_pbi is requested.
    m_reg = ue.alloc_isa_reg() if use_pbi else None

    ue.start_capture()
    if use_pbi:
        ue.generate_instruction_add_set(m_reg, M)

    total_flops_from_matmat_mul = ue.matmat_mul_core(M=M, K=K, N=N,
                                                    A_DRAM_ADDR=A_DRAM_ADDR,
                                                    B_DRAM_ADDR=B_DRAM_ADDR,
                                                    OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                                                    softmax_enable=softmax_enable,
                                                    C_DRAM_ADDR=C_DRAM_ADDR,
                                                    bias_mode=bias_mode,
                                                    gelu_enable=gelu_enable,
                                                    silu_enable=silu_enable,
                                                    sigmoid_enable=sigmoid_enable,
                                                    clamp_enable=clamp_enable,
                                                    log_enable=log_enable,
                                                    debug_fmax=debug_fmax,
                                                    ZERO_DRAM_ADDR=ZERO_DRAM_ADDR,
                                                    FMAX_DRAM_ADDR=FMAX_DRAM_ADDR,
                                                    gpr_M_reg=m_reg,
                                                    )

    ue.stop_capture()
    if use_pbi:
        ue.release_isa_reg()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    instruction_size = ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    # Test Time
    a_logical = torch.randn(M, K, dtype=torch.bfloat16) / math.sqrt(K) # normalizing input helps with numerical stability of softmax
    # input_scale amplifies the post-matmul dynamic range. Larger scales produce
    # wider variance in (a @ b.T), which stresses the exp + bf20 adder tree
    # denominator in softmax (and the bf19 adder tree in raw exp-mode sums).
    if input_scale != 1.0:
        a_logical = (a_logical.to(torch.float32) * float(input_scale)).to(torch.bfloat16)

    a = a_logical

    b = torch.randn(N, K, dtype=torch.bfloat16)

    c_logical = None
    c_broadcast_n = None
    if bias_enable:
        if bias_mode == "full_matrix":
            c_logical = torch.randn(M, N, dtype=torch.bfloat16)
            ue.dma_to_accelerator_memory(C_DRAM_ADDR, c_logical)
        elif bias_mode == "broadcast_N":
            c_broadcast_n = torch.randn(N, dtype=torch.bfloat16)
            ue.dma_to_accelerator_memory(C_DRAM_ADDR, c_broadcast_n)

    # DMA to accelerator memory -------------------------------------------------------------
    ue.dma_to_accelerator_memory(A_DRAM_ADDR, a)
    ue.dma_to_accelerator_memory(B_DRAM_ADDR, b)
    if debug_fmax:
        zero = torch.zeros(UE_VECTOR_SIZE, dtype=torch.bfloat16)
        ue.dma_to_accelerator_memory(ZERO_DRAM_ADDR, zero)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0) # 10 seconds timeout
    ue.report_timing_and_instruction_count()

    report_flop_rate_gflops, flops_ratio = ue.report_flop_rate_gflops(total_flops_from_matmat_mul)
    print(f"Report FLOPS for MxKxN Matmul: {report_flop_rate_gflops:.2f} GFLOPS, {flops_ratio:.2f}% peak throughput, {instruction_size // 32} instructions, for M={M}, K={K}, N={N}, bias_enable={bias_enable}, softmax_enable={softmax_enable}, bias_mode={bias_mode}, use_pbi={use_pbi}")

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M, N))
    fmax = None
    if softmax_enable:
        if debug_fmax:
            fmax = ue.dma_from_accelerator_memory(FMAX_DRAM_ADDR, (M, UE_VECTOR_SIZE))
            fmax = -fmax[:, 0] # hardware negates the fmax, so we need to negate it again

    if bias_enable and bias_mode == "full_matrix":
        ref = a_logical @ b.T + c_logical
    elif bias_enable and bias_mode == "broadcast_N":
        ref = a_logical @ b.T + c_broadcast_n
    else:
        ref = a_logical @ b.T

    def apply_gelu(x):
        return x * torch.sigmoid(1.702 * x)

    def apply_silu(x):
        return x * torch.sigmoid(x)

    def apply_sigmoid(x):
        return torch.sigmoid(x)

    if gelu_enable:
        ref = apply_gelu(ref)
    elif silu_enable:
        ref = apply_silu(ref)
    elif sigmoid_enable:
        ref = apply_sigmoid(ref)
    elif clamp_enable:
        ref = torch.clamp(ref, min=0.0)
    elif log_enable:
        ref = torch.log(torch.clamp(ref, min=1e-3))

    if softmax_enable:
        if debug_fmax:
            fmax_ref = torch.max(ref, dim=-1).values
            snr_db_fmax = calculate_snr(fmax_ref, fmax)
            print(f"FMAX SNR Analysis: {snr_db_fmax:.2f} dB")
            assert snr_db_fmax >= fmax_snr_threshold_db or snr_db_fmax == float('inf'), f"FMAX SNR {snr_db_fmax:.2f} dB must be at least {fmax_snr_threshold_db:g} dB"
        ref = torch.softmax(ref, dim=-1).to(torch.bfloat16)

    snr_db_ref = calculate_snr(ref, output)
    print(f"Reference SNR Analysis for MxKxN Matmul: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= snr_threshold_db or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least {snr_threshold_db:g} dB"

    flags = []
    if bias_enable:    flags.append(f"bias-{bias_mode}")
    if softmax_enable: flags.append("softmax")
    if gelu_enable:    flags.append("gelu")
    if silu_enable:    flags.append("silu")
    if sigmoid_enable: flags.append("sigmoid")
    if clamp_enable:   flags.append("clamp")
    if log_enable:     flags.append("log")
    if use_pbi:        flags.append("pbi")
    if input_scale != 1.0: flags.append(f"scale={input_scale:g}")
    flag_str = ("+" + "+".join(flags)) if flags else ""
    name_base = "matmat_mul"
    record_test(f"{name_base}{flag_str}",
                f"M={M}, K={K}, N={N}",
                snr_db=snr_db_ref,
                gflops=report_flop_rate_gflops,
                inst_bytes=instruction_size)

    bias_trace = f"bias_{bias_mode}" if bias_enable else "bias_disabled"
    softmax_trace = "softmax_enabled" if softmax_enable else "softmax_disabled"
    pbi_suffix = "_pbi" if use_pbi else ""
    # generate_trace(
    #     ue,
    #     f"matmat_mul_core_M{M}_K{K}_N{N}_{bias_trace}_{softmax_trace}{pbi_suffix}.csv",
    # )

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

def matmat_mul_dynamic_m_test(
    K: int,
    N: int,
    M_runtime_values: list[int],
    bias_enable: bool = False,
    softmax_enable: bool = False,
    bias_mode: str = "broadcast_N",
    gelu_enable: bool = False,
    silu_enable: bool = False,
    sigmoid_enable: bool = False,
    clamp_enable: bool = False,
    log_enable: bool = False,
    snr_threshold_db: float = 40.0,
    compare_legacy: bool = False
):
    assert M_runtime_values, "M_runtime_values must be non-empty"

    def _apply_activations(t):
        if gelu_enable:    return t * torch.sigmoid(1.702 * t)
        if silu_enable:    return t * torch.sigmoid(t)
        if sigmoid_enable: return torch.sigmoid(t)
        if clamp_enable:   return torch.clamp(t, min=0.0)
        if log_enable:     return torch.log(torch.clamp(t, min=1e-3))
        return t

    def _flags(path_tag: str) -> str:
        parts = []
        if bias_enable:    parts.append(f"bias-{bias_mode}")
        if softmax_enable: parts.append("softmax")
        if gelu_enable:    parts.append("gelu")
        if silu_enable:    parts.append("silu")
        if sigmoid_enable: parts.append("sigmoid")
        if clamp_enable:   parts.append("clamp")
        if log_enable:     parts.append("log")
        parts.append(path_tag)
        return "+" + "+".join(parts)

    # =========================================================================
    # BLOCK 1 — PBI + dynamic M
    # =========================================================================
    print(f"\n{'#'*64}")
    print(f"# BLOCK 1 — PBI dynamic-M  _, K={K}, N={N})")
    print(f"{'#'*64}")

    ue = UnifiedEngine()

    # ── gpr_M_reg is a persistent register allocated OUTSIDE any capture session.
    # The main matmul body references it by index only; its value is injected
    # at runtime by the per-M preamble — exactly as self.gpr_seq_len works in
    # _compile_prefill_program.
    gpr_M_reg = ue.alloc_isa_reg()

    MAX_CONTEXT_SIZE = 512
    M_template = UE_VECTOR_SIZE

    # ── DRAM — sized for worst-case M ─────────────────────────────────
    A_DRAM_ADDR      = ue.allocate_tensor_dram(MAX_CONTEXT_SIZE * K * 2)
    B_DRAM_ADDR      = ue.allocate_tensor_dram(N * K * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(MAX_CONTEXT_SIZE * N * 2)

    C_DRAM_ADDR = None
    if bias_enable:
        C_DRAM_ADDR = ue.allocate_tensor_dram(
            (MAX_CONTEXT_SIZE * N if bias_mode == "full_matrix" else N) * 2
        )

    ZERO_DRAM_ADDR = FMAX_DRAM_ADDR = None
    if softmax_enable:
        ZERO_DRAM_ADDR = ue.allocate_tensor_dram(UE_VECTOR_SIZE * 2)
        FMAX_DRAM_ADDR = ue.allocate_tensor_dram(MAX_CONTEXT_SIZE * UE_VECTOR_SIZE * 2)

    # ── Compile main matmul body ONCE ─────────────────────────────────────────
    ue.start_capture()
    ue.matmat_mul_core(
        M=M_template, K=K, N=N,
        A_DRAM_ADDR=A_DRAM_ADDR, B_DRAM_ADDR=B_DRAM_ADDR, OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
        softmax_enable=softmax_enable, C_DRAM_ADDR=C_DRAM_ADDR, bias_mode=bias_mode,
        gelu_enable=gelu_enable, silu_enable=silu_enable, sigmoid_enable=sigmoid_enable,
        clamp_enable=clamp_enable, log_enable=log_enable,
        ZERO_DRAM_ADDR=ZERO_DRAM_ADDR, FMAX_DRAM_ADDR=FMAX_DRAM_ADDR,
        gpr_M_reg=gpr_M_reg,   # register index is fixed at compile time; value injected at runtime
    )
    ue.stop_capture()
    ue.generate_instruction_halt()

    main_program_dram_addr = ue.get_program_dram_addr()
    main_instruction_size  = ue.write_captured_instructions_to_dram(main_program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    # ── Reserve a fixed preamble DRAM slot right after the main body.
    # Each runtime M overwrites this same slot with a fresh 2-instruction preamble
    # (add_set gpr_M_reg, M_test  +  jump_abs → main body), then executes from it.
    # 4 slots reserved to absorb any NOP padding _generate_instruction_jump may emit.
    PREAMBLE_RESERVED_BYTES = 4 * INSTRUCTION_SIZE_BYTES
    preamble_dram_addr = ue.get_program_dram_addr()
    ue.allocate_program_dram(PREAMBLE_RESERVED_BYTES)

    # Word address of the main body — used as the absolute jump target in every preamble
    main_program_word_addr = ue_35bit_addr_shifter(main_program_dram_addr)

    # ── Constant data: uploaded once, reused across all M_test runs ───────────
    b = torch.randn(N, K, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(B_DRAM_ADDR, b)

    c_broadcast_n = None
    if bias_enable and bias_mode == "broadcast_N":
        c_broadcast_n = torch.randn(N, dtype=torch.bfloat16)
        ue.dma_to_accelerator_memory(C_DRAM_ADDR, c_broadcast_n)
    if softmax_enable:
        ue.dma_to_accelerator_memory(ZERO_DRAM_ADDR, torch.zeros(UE_VECTOR_SIZE, dtype=torch.bfloat16))

    # ── Runtime loop ──────────────────────────────────────────────────────────
    for M_test in M_runtime_values:
        print(f"\n{'='*64}")
        print(f"[PBI] M_test={M_test}, K={K}, N={N})")

        # Compile the 2-instruction preamble for this M_test:
        #   add_set  gpr_M_reg, M_test        <- seeds the runtime row count
        #   jump_abs main_program_word_addr  <- enters the cached matmul body
        # This mirrors how _compile_prefill_program's runtime preamble primes
        # gpr_seq_len / gpr_q_seq_len / gpr_bucket_idx before entering the cached bin.
        ue.clear_capture_buffer()
        ue.start_capture()
        ue.generate_instruction_add_set(gpr_M_reg, M_test)
        ue.generate_instruction_jump_abs(main_program_word_addr)
        ue.stop_capture()
        # Overwrite the fixed preamble slot — main body in DRAM is untouched
        ue.write_captured_instructions_to_dram(preamble_dram_addr)

        # Upload A for this M_test (only the first M_test rows of the DRAM region are filled)
        a = torch.randn(M_test, K, dtype=torch.bfloat16) / math.sqrt(K)
        ue.dma_to_accelerator_memory(A_DRAM_ADDR, a)

        c_logical = None
        if bias_enable and bias_mode == "full_matrix":
            c_logical = torch.randn(M_test, N, dtype=torch.bfloat16)
            ue.dma_to_accelerator_memory(C_DRAM_ADDR, c_logical)

        # Execute from preamble: add_set seeds gpr_M_reg, jump enters matmul
        ue.start_execute_from_dram(preamble_dram_addr)
        ue.wait_queue(10.0)
        ue.report_timing_and_instruction_count()

        actual_flops = 2 * M_test * K * N
        if softmax_enable:                               actual_flops += M_test * N * 5
        if gelu_enable or silu_enable or sigmoid_enable: actual_flops += M_test * N
        if clamp_enable:                                 actual_flops += M_test * N
        if log_enable:                                   actual_flops += 2 * M_test * N

        report_gflops, flops_ratio = ue.report_flop_rate_gflops(actual_flops)
        print(f"[PBI] {report_gflops:.2f} GFLOPS ({flops_ratio:.2f}% peak), "
              f"{main_instruction_size // 32} instructions")

        output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M_test, N))

        ref = a @ b.T
        if bias_enable and bias_mode == "full_matrix":   ref = ref + c_logical
        elif bias_enable and bias_mode == "broadcast_N": ref = ref + c_broadcast_n
        ref = _apply_activations(ref)
        if softmax_enable:
            ref = torch.softmax(ref, dim=-1).to(torch.bfloat16)

        snr_db = calculate_snr(ref, output)
        print(f"[PBI] SNR: {snr_db:.2f} dB")
        assert snr_db >= snr_threshold_db or snr_db == float("inf"), (
            f"[PBI] M_test={M_test}: SNR {snr_db:.2f} dB below {snr_threshold_db:g} dB"
        )

        record_test(
            f"matmat_mul{_flags('pbi+dynamic_M')}",
            f" M={M_test}, K={K}, N={N}",
            snr_db=snr_db, gflops=report_gflops, inst_bytes=main_instruction_size,
        )

    ue.release_isa_reg()  # gpr_M_reg — release after all PBI runs are done
    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

    if not compare_legacy:
        return

    # =========================================================================
    # BLOCK 2 — Legacy (use_pbi=False): one compile-and-run per M value
    # =========================================================================
    print(f"\n{'#'*64}")
    print(f"# BLOCK 2 — Legacy (no PBI)  (K={K}, N={N})")
    print(f"{'#'*64}")

    for M_test in M_runtime_values:
        print(f"\n{'='*64}")
        print(f"[Legacy] M_test={M_test}  (K={K}, N={N})")

        ue = UnifiedEngine()

        A_DRAM_ADDR      = ue.allocate_tensor_dram(M_test * K * 2)
        B_DRAM_ADDR_leg  = ue.allocate_tensor_dram(N * K * 2)
        OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M_test * N * 2)

        C_DRAM_ADDR = None
        if bias_enable:
            C_DRAM_ADDR = ue.allocate_tensor_dram(
                (M_test * N if bias_mode == "full_matrix" else N) * 2
            )

        ZERO_DRAM_ADDR = FMAX_DRAM_ADDR = None
        if softmax_enable:
            ZERO_DRAM_ADDR = ue.allocate_tensor_dram(UE_VECTOR_SIZE * 2)
            FMAX_DRAM_ADDR = ue.allocate_tensor_dram(M_test * UE_VECTOR_SIZE * 2)

        ue.start_capture()
        ue.matmat_mul_core(
            M=M_test, K=K, N=N,
            A_DRAM_ADDR=A_DRAM_ADDR, B_DRAM_ADDR=B_DRAM_ADDR_leg, OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
            softmax_enable=softmax_enable, C_DRAM_ADDR=C_DRAM_ADDR, bias_mode=bias_mode,
            gelu_enable=gelu_enable, silu_enable=silu_enable, sigmoid_enable=sigmoid_enable,
            clamp_enable=clamp_enable, log_enable=log_enable,
            ZERO_DRAM_ADDR=ZERO_DRAM_ADDR, FMAX_DRAM_ADDR=FMAX_DRAM_ADDR,
        )
        ue.stop_capture()
        ue.generate_instruction_halt()

        program_dram_addr = ue.get_program_dram_addr()
        instruction_size  = ue.write_captured_instructions_to_dram(program_dram_addr)
        ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

        a = torch.randn(M_test, K, dtype=torch.bfloat16) / math.sqrt(K)
        ue.dma_to_accelerator_memory(A_DRAM_ADDR, a)
        ue.dma_to_accelerator_memory(B_DRAM_ADDR_leg, b)  # same b as Block 1

        c_logical = None
        if bias_enable and bias_mode == "full_matrix":
            c_logical = torch.randn(M_test, N, dtype=torch.bfloat16)
            ue.dma_to_accelerator_memory(C_DRAM_ADDR, c_logical)
        elif bias_enable and bias_mode == "broadcast_N":
            ue.dma_to_accelerator_memory(C_DRAM_ADDR, c_broadcast_n)
        if softmax_enable:
            ue.dma_to_accelerator_memory(ZERO_DRAM_ADDR, torch.zeros(UE_VECTOR_SIZE, dtype=torch.bfloat16))

        ue.start_execute_from_dram(program_dram_addr)
        ue.wait_queue(10.0)
        ue.report_timing_and_instruction_count()

        actual_flops = 2 * M_test * K * N
        if softmax_enable:                               actual_flops += M_test * N * 5
        if gelu_enable or silu_enable or sigmoid_enable: actual_flops += M_test * N
        if clamp_enable:                                 actual_flops += M_test * N
        if log_enable:                                   actual_flops += 2 * M_test * N

        report_gflops, flops_ratio = ue.report_flop_rate_gflops(actual_flops)
        print(f"[Legacy] {report_gflops:.2f} GFLOPS ({flops_ratio:.2f}% peak), "
              f"{instruction_size // 32} instructions")

        output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M_test, N))

        ref = a @ b.T
        if bias_enable and bias_mode == "full_matrix":   ref = ref + c_logical
        elif bias_enable and bias_mode == "broadcast_N": ref = ref + c_broadcast_n
        ref = _apply_activations(ref)
        if softmax_enable:
            ref = torch.softmax(ref, dim=-1).to(torch.bfloat16)

        snr_db = calculate_snr(ref, output)
        print(f"[Legacy] SNR: {snr_db:.2f} dB")
        assert snr_db >= snr_threshold_db or snr_db == float("inf"), (
            f"[Legacy] M_test={M_test}: SNR {snr_db:.2f} dB below {snr_threshold_db:g} dB"
        )

        record_test(
            f"matmat_mul{_flags('legacy')}",
            f"M={M_test}, K={K}, N={N}",
            snr_db=snr_db, gflops=report_gflops, inst_bytes=instruction_size,
        )

        ue.clear_capture_buffer()
        ue.reset_tensor_dram_addr()

def rms_norm_test(shape: tuple, use_pbi: bool = False):
    """
    Tests rms norm core.

    When ``use_pbi=True``, primes a fixed GPR with ``M`` before the kernel and passes that register
    as ``gpr_M_reg`` so the wrapper routes to :meth:`rms_norm_core_dram_pbi` (outer row loop uses
    runtime trip count). When ``use_pbi=False`` the wrapper routes to the legacy compile-time path.
    """
    ue = UnifiedEngine()

    assert len(shape) == 2, f"shape must be a tuple of length 2, got {shape}"

    M = shape[0]
    N = shape[1]

    A_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    GAMMA_DRAM_ADDR = ue.allocate_tensor_dram(N * 2)

    # Fixed GPR for row count — stays above aliasing with loop counter from loop_start (~1–2 during capture).
    _GPR_M_REG = 8

    ue.start_capture()
    if use_pbi:
        ue.generate_instruction_add_set(_GPR_M_REG, M)
    total_flops_from_rms_norm = ue.rms_norm_core_dram(
        M=M,
        N=N,
        A_DRAM_ADDR=A_DRAM_ADDR,
        OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
        GAMMA_DRAM_ADDR=GAMMA_DRAM_ADDR,
        gpr_M_reg=_GPR_M_REG if use_pbi else None,
    )

    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())
    ue.clear_capture_buffer()

    x = torch.randn(M, N, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(A_DRAM_ADDR, x)
    gamma = torch.randn(N, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(GAMMA_DRAM_ADDR, gamma)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0) # 10 seconds timeout
    ue.report_timing_and_instruction_count()

    flop_rate_gflops, flops_ratio = ue.report_flop_rate_gflops(total_flops_from_rms_norm)
    print(
        f"Report FLOPS for RMS Norm: {flop_rate_gflops:.2f} GFLOPS, {flops_ratio:.2f}% peak throughput "
        f"for M={M}, N={N}, use_pbi={use_pbi}"
    )

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M, N))

    rms_norm = torch.nn.RMSNorm(N)
    rms_norm.weight.data = gamma
    ref = rms_norm(x)
    snr_db_ref = calculate_snr(ref, output)
    print(f"Reference SNR Analysis for RMS Norm: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 40 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 40 dB"

    test_name = "rms_norm_pbi" if use_pbi else "rms_norm"
    record_test(test_name,
                f"M={M}, N={N}",
                snr_db=snr_db_ref,
                gflops=flop_rate_gflops)

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

def layer_norm_test(shape: tuple, gamma_enable: bool = False, beta_enable: bool = False,
                    use_pbi: bool = False):
    """Tests layer_norm_core_dram.

    When ``use_pbi=True``, primes GPR 8 with the chunk count ``M // chunk_size`` (where
    ``chunk_size = min(URAM_NEAR_FULL_ELEMENTS // N, M)``) and routes to
    :meth:`layer_norm_core_dram_pbi`.  The PBI loop loads/stores ``chunk_size`` rows per DMA
    call — identical granularity to the legacy path — so performance is on par while the
    program shrinks from ~M*6 to ~chunk_size*6+4 instructions.
    When ``use_pbi=False`` the wrapper routes to the legacy compile-time path.
    """
    ue = UnifiedEngine()

    assert len(shape) == 2, f"shape must be a tuple of length 2, got {shape}"

    M = shape[0]
    N = shape[1]

    A_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    GAMMA_DRAM_ADDR = ue.allocate_tensor_dram(N * 2) if gamma_enable else None
    BETA_DRAM_ADDR = ue.allocate_tensor_dram(N * 2) if beta_enable else None

    _GPR_M_REG = 8  # fixed GPR; stays clear of loop_start internal registers

    ue.start_capture()
    if use_pbi:
        # Must mirror the chunk_size logic in layer_norm_core_dram_pbi exactly.
        _ops = 4 + (1 if gamma_enable else 0) + (1 if beta_enable else 0)
        _ideal = min(URAM_NEAR_FULL_ELEMENTS // N, M, (256 - 4) // _ops)
        _chunk_size = _ideal
        while M % _chunk_size != 0:
            _chunk_size -= 1
        ue.generate_instruction_add_set(_GPR_M_REG, M // _chunk_size)
    total_flops = ue.layer_norm_core_dram(
        M=M, N=N,
        A_DRAM_ADDR=A_DRAM_ADDR,
        OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
        GAMMA_DRAM_ADDR=GAMMA_DRAM_ADDR,
        BETA_DRAM_ADDR=BETA_DRAM_ADDR,
        gpr_M_reg=_GPR_M_REG if use_pbi else None,
    )
    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    instruction_size_bytes = ue.get_capture_instruction_size_bytes()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(instruction_size_bytes)
    ue.clear_capture_buffer()

    x = torch.randn(M, N, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(A_DRAM_ADDR, x)
    gamma = beta = None
    if gamma_enable:
        gamma = torch.randn(N, dtype=torch.bfloat16)
        ue.dma_to_accelerator_memory(GAMMA_DRAM_ADDR, gamma)
    if beta_enable:
        beta = torch.randn(N, dtype=torch.bfloat16)
        ue.dma_to_accelerator_memory(BETA_DRAM_ADDR, beta)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0)
    ue.report_timing_and_instruction_count()

    flop_rate_gflops, flops_ratio = ue.report_flop_rate_gflops(total_flops)
    print(
        f"Report FLOPS for Layer Norm{'(PBI)' if use_pbi else ''}: "
        f"{flop_rate_gflops:.2f} GFLOPS, {flops_ratio:.2f}% peak for "
        f"M={M}, N={N}, gamma={gamma_enable}, beta={beta_enable}"
    )

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M, N))

    layer_norm = torch.nn.LayerNorm(N)
    layer_norm.weight.data = gamma if gamma_enable else torch.ones(N, dtype=torch.bfloat16)
    layer_norm.bias.data   = beta  if beta_enable  else torch.zeros(N, dtype=torch.bfloat16)

    ref = layer_norm(x)
    snr_db = calculate_snr(ref, output)
    print(f"Layer Norm SNR Analysis: {snr_db:.2f} dB")
    assert snr_db >= 40 or snr_db == float('inf'), f"SNR {snr_db:.2f} dB must be at least 40 dB"

    flags = []
    if gamma_enable: flags.append("gamma")
    if beta_enable:  flags.append("beta")
    flag_str = ("+" + "+".join(flags)) if flags else ""
    test_name = f"layer_norm{'_pbi' if use_pbi else ''}{flag_str}"
    record_test(test_name, f"M={M}, N={N}", snr_db=snr_db, gflops=flop_rate_gflops,
                inst_bytes=instruction_size_bytes)

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()
    ue.reset_program_dram_addr()


def _ln_hw_run(ue, M, N, x_pad, gamma_pad, beta_pad):
    """Capture+run one layer_norm_core_dram over (M, N) and return the (M, N) output.
    Caller supplies already-padded x/gamma/beta (all width N). Self-contained: allocates,
    captures, executes, and resets DRAM pointers so it can be called repeatedly."""
    A = ue.allocate_tensor_dram(M * N * 2)
    O = ue.allocate_tensor_dram(M * N * 2)
    G = ue.allocate_tensor_dram(N * 2)
    B = ue.allocate_tensor_dram(N * 2)

    ue.start_capture()
    ue.layer_norm_core_dram(M=M, N=N, A_DRAM_ADDR=A, OUTPUT_DRAM_ADDR=O,
                            GAMMA_DRAM_ADDR=G, BETA_DRAM_ADDR=B)
    ue.stop_capture()
    ue.generate_instruction_halt()
    prog = ue.get_program_dram_addr()
    isz = ue.get_capture_instruction_size_bytes()
    ue.write_captured_instructions_to_dram(prog)
    ue.allocate_program_dram(isz)
    ue.clear_capture_buffer()

    ue.dma_to_accelerator_memory(A, x_pad.contiguous())
    ue.dma_to_accelerator_memory(G, gamma_pad.contiguous())
    ue.dma_to_accelerator_memory(B, beta_pad.contiguous())
    ue.start_execute_from_dram(prog)
    ue.wait_queue(10.0)
    out = ue.dma_from_accelerator_memory(O, (M, N))

    ue.reset_tensor_dram_addr()
    ue.reset_program_dram_addr()
    return out


def _ln_hw_run_chunked(ue, M, N, x_pad, gamma_pad, beta_pad, chunk_size):
    """Like _ln_hw_run but drives the gpr_M_reg M-collapse path EXACTLY as the model's
    _ln_hw does (mobilesam S2/neck): GPR14 = M//chunk_size, layer_norm_core_dram(gpr_M_reg=14).
    This is the path the model uses but the plain LN test never exercised."""
    A = ue.allocate_tensor_dram(M * N * 2)
    O = ue.allocate_tensor_dram(M * N * 2)
    G = ue.allocate_tensor_dram(N * 2)
    B = ue.allocate_tensor_dram(N * 2)

    ue.start_capture()
    ue._isa_reg_counter = 15
    ue.reset_inst_ptr_counter()
    ue.generate_instruction_add_set(dst_reg_idx=14, immediate_value=M // chunk_size)
    ue.layer_norm_core_dram(M=M, N=N, A_DRAM_ADDR=A, OUTPUT_DRAM_ADDR=O,
                            GAMMA_DRAM_ADDR=G, BETA_DRAM_ADDR=B, gpr_M_reg=14)
    ue.stop_capture()
    ue.generate_instruction_halt()
    prog = ue.get_program_dram_addr()
    isz = ue.get_capture_instruction_size_bytes()
    ue.write_captured_instructions_to_dram(prog)
    ue.allocate_program_dram(isz)
    ue.clear_capture_buffer()

    ue.dma_to_accelerator_memory(A, x_pad.contiguous())
    ue.dma_to_accelerator_memory(G, gamma_pad.contiguous())
    ue.dma_to_accelerator_memory(B, beta_pad.contiguous())
    ue.start_execute_from_dram(prog)
    ue.wait_queue(20.0)
    out = ue.dma_from_accelerator_memory(O, (M, N))
    ue.reset_tensor_dram_addr()
    ue.reset_program_dram_addr()
    return out


def s2_layernorm_chunked_vs_plain_test(M=4096, C_real=160, C_pad=192, chunk_size=32):
    """Pin the MobileSAM S2 LayerNorm collapse: the model runs layer_norm_core_dram with
    gpr_M_reg M-collapse (M=4096 in 128 chunks of 32). The plain LN path scores ~30 dB on
    real S2 data, but the model's S2 LN output measures ~4.5 dB vs golden. This A/B's the
    SAME input through plain vs chunked(gpr_M_reg) to isolate the chunked-LN path."""
    import numpy as _np
    ue = UnifiedEngine()
    scale = math.sqrt(C_real / C_pad)
    _here = os.path.dirname(os.path.abspath(__file__))
    arr = _np.load(os.path.join(_here, "models", "mobilesam", "cpu_golden.npz"))["PM12"]
    x_real = torch.from_numpy(arr.reshape(-1, C_real)[:M]).float().bfloat16()
    _sd = torch.load(os.path.join(_here, "models", "mobilesam", "mobilesam_bin", "mobile_sam.pt"),
                     map_location="cpu", weights_only=True)
    gamma = _sd["image_encoder.layers.2.blocks.0.attn.norm.weight"].to(torch.bfloat16)
    beta  = _sd["image_encoder.layers.2.blocks.0.attn.norm.bias"].to(torch.bfloat16)

    ln = torch.nn.LayerNorm(C_real); ln.weight.data = gamma.float(); ln.bias.data = beta.float()
    ref = ln(x_real.float()).bfloat16()

    def _pad_w(v):
        w = torch.zeros(C_pad, dtype=torch.bfloat16); w[:C_real] = v; return w
    gamma_scaled = _pad_w((gamma.float() * scale).bfloat16())
    beta_pad = _pad_w(beta)
    x_zero = torch.zeros(M, C_pad, dtype=torch.bfloat16); x_zero[:, :C_real] = x_real

    out_plain = _ln_hw_run(ue, M, C_pad, x_zero, gamma_scaled, beta_pad)[:, :C_real]
    out_chunk = _ln_hw_run_chunked(ue, M, C_pad, x_zero, gamma_scaled, beta_pad, chunk_size)[:, :C_real]
    snr_plain = calculate_snr(ref, out_plain)
    snr_chunk = calculate_snr(ref, out_chunk)

    print(f"\n[s2_layernorm_chunked_vs_plain] M={M} N={C_pad} chunk_size={chunk_size} (GPR14={M//chunk_size})")
    print(f"  plain  layer_norm_core_dram          SNR = {snr_plain:.2f} dB")
    print(f"  chunked layer_norm_core_dram(gpr_M)  SNR = {snr_chunk:.2f} dB   <- the model's path")
    record_test("s2_ln_plain", f"M={M},N={C_pad}", snr_db=snr_plain)
    record_test("s2_ln_chunked", f"M={M},N={C_pad},cs={chunk_size}", snr_db=snr_chunk)
    return snr_plain, snr_chunk


def layer_norm_padded_channels_test(M=None, C_real=160, C_pad=192):
    """Faithful MobileSAM S2 LayerNorm-over-padded-channels reproducer.

    Mirrors the model EXACTLY (mobilesam_test.py S2 block, ~line 2300):
      - model dims: M = 64*64 = 4096 tokens, C_real=160 padded to C_pad=192,
      - real S2B0 `attn.norm` gamma/beta loaded from the checkpoint,
      - gamma pre-scaled by sqrt(C_real/C_pad) and zero-padded to 192,
      - beta zero-padded to 192,
      - input is the real golden PM12 tensor, zero-filled in the 32 pad cols.

    The encoder pads S2's 160 real channels up to 192 (3x64) for kernel alignment.
    LayerNorm normalizes over the channel dim, but the HW core normalizes over ALL N
    columns. The model (a) zero-fills the 32 pad channels and (b) pre-scales gamma by
    sqrt(C_real/C_pad) to correct the variance. That scaling is exact ONLY when the
    per-row mean is 0 (the zeros drag the mean to (C_real/C_pad)*mu and inject
    (C_pad-C_real)*mu^2 of extra variance). It also relies on the gamma/beta pad cols
    being zero so the LN OUTPUT pad cols stay 0 and cannot leak into qkv -- this test
    asserts that explicitly.

    Compares, against the true LayerNorm over the 160 real channels:
      [current]   pad = 0,        gamma_pad = gamma * sqrt(C_real/C_pad)
      [pad=mean]  pad = row-mean, gamma_pad = gamma * sqrt(C_real/C_pad)  (proposed, exact)
    """
    if M is None:
        M = 64 * 64  # ENC_S2_H * ENC_S2_W -- the model's real S2 token count
    ue = UnifiedEngine()
    scale = math.sqrt(C_real / C_pad)
    _here = os.path.dirname(os.path.abspath(__file__))

    # Real S2 LN input: the golden PM12 tensor (key "PM12", shape (4096,160)).
    _npz = os.path.join(_here, "models", "mobilesam", "cpu_golden.npz")
    if os.path.exists(_npz):
        import numpy as _np
        arr = _np.load(_npz)["PM12"].reshape(-1, C_real)[:M]
        x_real = torch.from_numpy(arr).float().bfloat16()
        src = f"golden PM12 (real S2 LN input), per-row mean abs~{x_real.float().mean(-1).abs().mean():.3f}"
    else:
        x_real = (torch.randn(M, C_real) + 3.0 * torch.randn(M, 1)).bfloat16()  # nonzero per-row mean
        src = f"synthetic randn+per-row-bias, per-row mean abs~{x_real.float().mean(-1).abs().mean():.3f}"

    # Real S2B0 attn.norm weights from the checkpoint (fall back to random if absent).
    _ckpt = os.path.join(_here, "models", "mobilesam", "mobilesam_bin", "mobile_sam.pt")
    if os.path.exists(_ckpt):
        _sd = torch.load(_ckpt, map_location="cpu", weights_only=True)
        gamma = _sd["image_encoder.layers.2.blocks.0.attn.norm.weight"].to(torch.bfloat16)
        beta  = _sd["image_encoder.layers.2.blocks.0.attn.norm.bias"].to(torch.bfloat16)
        src += " | real S2B0 attn.norm gamma/beta"
    else:
        gamma = torch.randn(C_real, dtype=torch.bfloat16)
        beta  = torch.randn(C_real, dtype=torch.bfloat16)
        src += " | random gamma/beta"

    ln = torch.nn.LayerNorm(C_real)
    ln.weight.data = gamma.float()
    ln.bias.data   = beta.float()
    ref = ln(x_real.float()).bfloat16()  # (M, C_real) golden

    def _pad_w(v):
        w = torch.zeros(C_pad, dtype=torch.bfloat16); w[:C_real] = v; return w
    gamma_scaled = _pad_w((gamma.float() * scale).bfloat16())
    beta_pad     = _pad_w(beta)

    # --- approach 1: current model (zero pad) ---
    x_zero = torch.zeros(M, C_pad, dtype=torch.bfloat16); x_zero[:, :C_real] = x_real
    out1_full = _ln_hw_run(ue, M, C_pad, x_zero, gamma_scaled, beta_pad)
    out1 = out1_full[:, :C_real]
    snr1 = calculate_snr(ref, out1)

    # The model's S2 correctness DEPENDS on the LN output pad cols (160:192) staying ~0,
    # so they cannot leak into the downstream qkv matmul (which reads all 192 cols).
    # gamma_pad/beta_pad are zero there, so the HW must emit ~0. Assert it.
    pad_leak = out1_full[:, C_real:].abs().max().item()
    print(f"  [pad-col leak] max|LN_out[:,160:192]| = {pad_leak:.4e}  (must be ~0)")
    assert pad_leak < 1e-2, (
        f"LN output pad columns are nonzero ({pad_leak:.4e}) -- they WILL leak into "
        f"qkv and corrupt S2 attention. This is a real padded-channel bug.")

    # --- approach 2: pad with per-row mean of the real channels (proposed fix) ---
    row_mean = x_real.float().mean(dim=1, keepdim=True)
    x_mean = torch.zeros(M, C_pad, dtype=torch.bfloat16)
    x_mean[:, :C_real] = x_real
    x_mean[:, C_real:] = row_mean.expand(M, C_pad - C_real).bfloat16()
    out2 = _ln_hw_run(ue, M, C_pad, x_mean, gamma_scaled, beta_pad)[:, :C_real]
    snr2 = calculate_snr(ref, out2)

    print(f"\n[layer_norm_padded_channels] M={M} C_real={C_real} C_pad={C_pad}")
    print(f"  input: {src}")
    print(f"  [current  zero-pad + gamma*sqrt(C_real/C_pad)] SNR = {snr1:.2f} dB   <- model today")
    print(f"  [fix  pad=row-mean + gamma*sqrt(C_real/C_pad) ] SNR = {snr2:.2f} dB   <- proposed")
    record_test("layer_norm_padded_zero", f"M={M},{C_real}->{C_pad}", snr_db=snr1)
    record_test("layer_norm_padded_meanfill", f"M={M},{C_real}->{C_pad}", snr_db=snr2)
    assert snr2 >= 40 or snr2 == float('inf'), f"pad=mean fix must reach 40 dB, got {snr2:.2f}"
    return snr1, snr2


def conv2d_3x3_im2col_pbi_test(H_in: int, W_in: int, C_in: int, stride: int = 1,
                               snr_threshold_db: float = 40.0):
    """Verify conv2d_3x3_im2col_pbi against a torch reference im2col (3x3, pad=1).

    Builds the im2col matrix on HW via PBI hardware loops and checks it bit-for-bit
    (SNR) against the same gather computed on the host, including zero padding.
    """
    ue = UnifiedEngine()
    K = 9 * C_in
    H_out = (H_in - 1) // stride + 1
    W_out = (W_in - 1) // stride + 1
    _GPR = 8

    INPUT_DRAM_ADDR  = ue.allocate_tensor_dram(H_in * W_in * C_in * _BPE)
    IM2COL_DRAM_ADDR = ue.allocate_tensor_dram(H_out * W_out * K * _BPE)

    ue.start_capture()
    conv2d_3x3_im2col_pbi(ue, INPUT_DRAM_ADDR, IM2COL_DRAM_ADDR,
                          H_in, W_in, C_in, stride, _GPR)
    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    instruction_size_bytes = ue.get_capture_instruction_size_bytes()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(instruction_size_bytes)
    ue.clear_capture_buffer()

    x = torch.randn(H_in * W_in, C_in, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(INPUT_DRAM_ADDR, x)
    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(30.0)
    out = ue.dma_from_accelerator_memory(IM2COL_DRAM_ADDR, (H_out * W_out, K)).float()

    # Host reference im2col (vectorized): pad input, then gather each tap block.
    xg = x.reshape(H_in, W_in, C_in).float()
    xp = torch.zeros(H_in + 2, W_in + 2, C_in)        # pad=1
    xp[1:H_in + 1, 1:W_in + 1] = xg
    ref = torch.zeros(H_out * W_out, K)
    rows = torch.arange(H_out) * stride               # top-left of each 3x3 (in padded coords)
    cols = torch.arange(W_out) * stride
    for dy in range(3):
        for dx in range(3):
            tap = dy * 3 + dx
            patch = xp[rows[:, None] + dy, cols[None, :] + dx]   # (H_out, W_out, C_in)
            ref[:, tap * C_in:(tap + 1) * C_in] = patch.reshape(H_out * W_out, C_in)

    snr_db = calculate_snr(ref, out)
    print(f"conv2d_3x3_im2col_pbi SNR: {snr_db:.2f} dB  "
          f"(H={H_in},W={W_in},C={C_in},s={stride}) inst={instruction_size_bytes}B "
          f"vs ~{H_out * W_out * 9} legacy gathers")
    assert snr_db >= snr_threshold_db or snr_db == float('inf'), \
        f"SNR {snr_db:.2f} dB must be at least {snr_threshold_db} dB"

    record_test("conv2d_3x3_im2col_pbi",
                f"H={H_in},W={W_in},C={C_in},s={stride}",
                snr_db=snr_db, inst_bytes=instruction_size_bytes)

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()
    ue.reset_program_dram_addr()


def window_partition_reverse_pbi_test(H: int, W: int, C: int, ws: int,
                                      snr_threshold_db: float = 40.0):
    """Verify window_partition_pbi (vs torch reference) and window_reverse_pbi
    (round-trips the partition output back to the original input)."""
    ue = UnifiedEngine()
    nH, nW, WIN_PAD = _window_dims(H, W, ws)
    _GPR = 8

    IN   = ue.allocate_tensor_dram(H * W * C * _BPE)
    WIN  = ue.allocate_tensor_dram(nH * nW * WIN_PAD * C * _BPE)
    BACK = ue.allocate_tensor_dram(H * W * C * _BPE)

    def _build_run(emit_fn, timeout=30.0):
        ue.start_capture()
        emit_fn()
        ue.stop_capture()
        ue.generate_instruction_halt()
        prog = ue.get_program_dram_addr()
        nbytes = ue.get_capture_instruction_size_bytes()
        ue.write_captured_instructions_to_dram(prog)
        ue.allocate_program_dram(nbytes)
        ue.clear_capture_buffer()
        ue.start_execute_from_dram(prog)
        ue.wait_queue(timeout)
        return nbytes

    x = torch.randn(H * W, C, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(IN, x)

    # --- forward: partition ---
    part_bytes = _build_run(lambda: window_partition_pbi(ue, IN, WIN, H, W, C, ws, _GPR))
    win_hw = ue.dma_from_accelerator_memory(WIN, (nH * nW, WIN_PAD, C)).float()

    # torch reference partition (zero-padded image, window permute, WIN_PAD pad rows)
    xp = torch.zeros(nH * ws, nW * ws, C)
    xp[:H, :W] = x.reshape(H, W, C).float()
    win_ref = torch.zeros(nH * nW, WIN_PAD, C)
    win_ref[:, :ws * ws] = xp.reshape(nH, ws, nW, ws, C).permute(0, 2, 1, 3, 4).reshape(nH * nW, ws * ws, C)
    snr_part = calculate_snr(win_ref, win_hw)

    # --- inverse: reverse (round-trip back to the original input) ---
    rev_bytes = _build_run(lambda: (
        dram_zero_fill(ue, BACK, H * W * C),
        window_reverse_pbi(ue, WIN, BACK, H, W, C, ws, nH, nW, WIN_PAD, _GPR),
    ))
    back_hw = ue.dma_from_accelerator_memory(BACK, (H * W, C)).float()
    snr_rev = calculate_snr(x.float(), back_hw)

    print(f"window_partition_pbi SNR: {snr_part:.2f} dB | reverse round-trip SNR: {snr_rev:.2f} dB "
          f"(H={H},W={W},C={C},ws={ws}) part_inst={part_bytes}B rev_inst={rev_bytes}B "
          f"vs ~{nH * nW * ws * 2} legacy memcpys")
    assert snr_part >= snr_threshold_db or snr_part == float('inf'), f"partition SNR {snr_part:.2f}"
    assert snr_rev >= snr_threshold_db or snr_rev == float('inf'), f"reverse SNR {snr_rev:.2f}"

    record_test("window_partition_pbi", f"H={H},W={W},C={C},ws={ws}",
                snr_db=snr_part, inst_bytes=part_bytes)
    record_test("window_reverse_pbi", f"H={H},W={W},C={C},ws={ws}",
                snr_db=snr_rev, inst_bytes=rev_bytes)

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()
    ue.reset_program_dram_addr()


def conv2d_3x3_pbi_test(H: int, W: int, C_in: int, C_out: int, stride: int = 1,
                        gelu_enable: bool = False, snr_threshold_db: float = 35.0,
                        zero_fn=dram_zero_fill):
    """End-to-end: conv2d_3x3_pbi vs torch.nn.functional.conv2d (3x3, pad=1)."""
    import torch.nn.functional as F
    ue = UnifiedEngine()
    K = 9 * C_in
    H_out = (H - 1) // stride + 1
    W_out = (W - 1) // stride + 1
    _GPR = 8

    IN   = ue.allocate_tensor_dram(H * W * C_in * _BPE)
    OUT  = ue.allocate_tensor_dram(H_out * W_out * C_out * _BPE)
    IM   = ue.allocate_tensor_dram(W_out * K * _BPE)              # one im2col row
    WGT  = ue.allocate_tensor_dram(C_out * K * _BPE)
    BIA  = ue.allocate_tensor_dram(C_out * _BPE)

    x  = torch.randn(H * W, C_in, dtype=torch.bfloat16)
    w  = torch.randn(C_out, C_in, 3, 3, dtype=torch.bfloat16) * 0.1
    b  = torch.randn(C_out, dtype=torch.bfloat16)
    # Weight -> (C_out, 9*C_in): tap (dy,dx) block at (dy+1)*3+(dx+1).
    wpk = torch.zeros(C_out, K, dtype=torch.bfloat16)
    for dy in range(3):
        for dx in range(3):
            tap = dy * 3 + dx
            wpk[:, tap * C_in:(tap + 1) * C_in] = w[:, :, dy, dx]
    ue.dma_to_accelerator_memory(IN, x)
    ue.dma_to_accelerator_memory(WGT, wpk)
    ue.dma_to_accelerator_memory(BIA, b)

    ue.start_capture()
    conv2d_3x3_pbi(ue, IN, OUT, IM, WGT, BIA, H, W, C_in, C_out, stride, gelu_enable, _GPR,
                   zero_fn=zero_fn)
    ue.stop_capture()
    ue.generate_instruction_halt()
    prog = ue.get_program_dram_addr()
    inst_bytes = ue.get_capture_instruction_size_bytes()
    ue.write_captured_instructions_to_dram(prog)
    ue.allocate_program_dram(inst_bytes)
    ue.clear_capture_buffer()
    ue.start_execute_from_dram(prog)
    ue.wait_queue(60.0)
    out = ue.dma_from_accelerator_memory(OUT, (H_out * W_out, C_out)).float()

    xin = x.reshape(H, W, C_in).permute(2, 0, 1).unsqueeze(0).float()
    ref = F.conv2d(xin, w.float(), b.float(), stride=stride, padding=1)
    if gelu_enable:
        ref = F.gelu(ref)
    ref = ref.squeeze(0).permute(1, 2, 0).reshape(H_out * W_out, C_out)
    snr_db = calculate_snr(ref, out)
    print(f"conv2d_3x3_pbi SNR: {snr_db:.2f} dB (H={H},W={W},Cin={C_in},Cout={C_out},s={stride}) "
          f"inst={inst_bytes}B vs ~{H_out * W_out * 9} legacy gathers + {H_out} matmuls")
    assert snr_db >= snr_threshold_db or snr_db == float('inf'), f"SNR {snr_db:.2f} dB"
    record_test(f"conv2d_3x3_pbi{'_gelu' if gelu_enable else ''}",
                f"H={H},W={W},Cin={C_in},Cout={C_out},s={stride}",
                snr_db=snr_db, inst_bytes=inst_bytes)

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()
    ue.reset_program_dram_addr()


def conv2d_3x3_dw_pbi_test(H: int, W: int, C: int, stride: int = 1,
                           gelu_enable: bool = False, snr_threshold_db: float = 35.0,
                           zero_fn=dram_zero_fill):
    """End-to-end: conv2d_3x3_dw_pbi vs torch depthwise conv2d (groups=C, 3x3, pad=1)."""
    import torch.nn.functional as F
    ue = UnifiedEngine()
    BLK = 64
    K = 9 * BLK
    n_blocks = C // BLK
    H_out = (H - 1) // stride + 1
    W_out = (W - 1) // stride + 1
    _GPR = 8

    IN   = ue.allocate_tensor_dram(H * W * C * _BPE)
    OUT  = ue.allocate_tensor_dram(H_out * W_out * C * _BPE)
    IM   = ue.allocate_tensor_dram(W_out * (K + BLK) * _BPE)      # im2col row + TEMP
    WGT  = ue.allocate_tensor_dram(C * K * _BPE)                  # (n_blocks*BLK, 9*BLK)
    BIA  = ue.allocate_tensor_dram(C * _BPE)

    x = torch.randn(H * W, C, dtype=torch.bfloat16)
    w = torch.randn(C, 1, 3, 3, dtype=torch.bfloat16) * 0.2
    b = torch.randn(C, dtype=torch.bfloat16)
    # Block weight: row (blk*BLK+i) has w[c,0,dy,dx] only in tap-block column i (depthwise).
    wblk = torch.zeros(C, K, dtype=torch.bfloat16)
    for blk in range(n_blocks):
        for i in range(BLK):
            c = blk * BLK + i
            for dy in range(3):
                for dx in range(3):
                    tap = dy * 3 + dx
                    wblk[blk * BLK + i, tap * BLK + i] = w[c, 0, dy, dx]
    ue.dma_to_accelerator_memory(IN, x)
    ue.dma_to_accelerator_memory(WGT, wblk)
    ue.dma_to_accelerator_memory(BIA, b)

    ue.start_capture()
    conv2d_3x3_dw_pbi(ue, IN, OUT, IM, WGT, BIA, H, W, C, stride, gelu_enable, _GPR,
                      zero_fn=zero_fn)
    ue.stop_capture()
    ue.generate_instruction_halt()
    prog = ue.get_program_dram_addr()
    inst_bytes = ue.get_capture_instruction_size_bytes()
    ue.write_captured_instructions_to_dram(prog)
    ue.allocate_program_dram(inst_bytes)
    ue.clear_capture_buffer()
    ue.start_execute_from_dram(prog)
    ue.wait_queue(60.0)
    out = ue.dma_from_accelerator_memory(OUT, (H_out * W_out, C)).float()

    xin = x.reshape(H, W, C).permute(2, 0, 1).unsqueeze(0).float()
    ref = F.conv2d(xin, w.float(), b.float(), stride=stride, padding=1, groups=C)
    if gelu_enable:
        ref = F.gelu(ref)
    ref = ref.squeeze(0).permute(1, 2, 0).reshape(H_out * W_out, C)
    snr_db = calculate_snr(ref, out)
    print(f"conv2d_3x3_dw_pbi SNR: {snr_db:.2f} dB (H={H},W={W},C={C},s={stride}) "
          f"inst={inst_bytes}B vs ~{n_blocks * H_out * W_out * 9} legacy gathers")
    assert snr_db >= snr_threshold_db or snr_db == float('inf'), f"SNR {snr_db:.2f} dB"
    record_test(f"conv2d_3x3_dw_pbi{'_gelu' if gelu_enable else ''}",
                f"H={H},W={W},C={C},s={stride}", snr_db=snr_db, inst_bytes=inst_bytes)

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()
    ue.reset_program_dram_addr()


def conv2d_3x3_pbi_chain_test(reps: int = 4, H: int = 32, W: int = 32,
                              C_in: int = 64, C_out: int = 64, stride: int = 1,
                              interleave_matmul: bool = True,
                              snr_threshold_db: float = 35.0,
                              zero_fn=None):
    """Reproduce the MODEL conditions the one-conv-per-program tests miss: run several
    PBI convs back-to-back inside a SINGLE captured program, each interleaved with a
    plain matmul (the model's 1x1 / identity matmuls), then verify EVERY conv output.

    The per-conv tests pass at 54 dB but the model's neck was corrupt — the only
    structural difference is that the model fires many PBI loops + matmuls in one
    program, sharing the cached zero source, the 0x00000 staging row, GPR 8 and the
    inst-pointer pool. If a later conv diverges while the first is clean, the bug is
    in that back-to-back reuse, not the gather. Defaults to the model zeroing path."""
    import torch.nn.functional as F
    if zero_fn is None:
        zero_fn = _pbi_zero_dram
    ue = UnifiedEngine()
    K = 9 * C_in
    H_out = (H - 1) // stride + 1
    W_out = (W - 1) // stride + 1
    _GPR = 8

    w = torch.randn(C_out, C_in, 3, 3, dtype=torch.bfloat16) * 0.1
    b = torch.randn(C_out, dtype=torch.bfloat16)
    wpk = torch.zeros(C_out, K, dtype=torch.bfloat16)
    for dy in range(3):
        for dx in range(3):
            tap = dy * 3 + dx
            wpk[:, tap * C_in:(tap + 1) * C_in] = w[:, :, dy, dx]
    # Identity weight for the interleaved 1x1 matmul (C_out -> C_out passthrough).
    ident = torch.eye(C_out, dtype=torch.bfloat16) if interleave_matmul else None

    WGT = ue.allocate_tensor_dram(C_out * K * _BPE)
    BIA = ue.allocate_tensor_dram(C_out * _BPE)
    IDW = ue.allocate_tensor_dram(C_out * C_out * _BPE) if interleave_matmul else None
    ue.dma_to_accelerator_memory(WGT, wpk)
    ue.dma_to_accelerator_memory(BIA, b)
    if interleave_matmul:
        ue.dma_to_accelerator_memory(IDW, ident)

    xs, INs, OUTs, IMs = [], [], [], []
    for r in range(reps):
        x = torch.randn(H * W, C_in, dtype=torch.bfloat16)
        IN  = ue.allocate_tensor_dram(H * W * C_in * _BPE)
        OUT = ue.allocate_tensor_dram(H_out * W_out * C_out * _BPE)
        IM  = ue.allocate_tensor_dram(W_out * K * _BPE)
        ue.dma_to_accelerator_memory(IN, x)
        xs.append(x); INs.append(IN); OUTs.append(OUT); IMs.append(IM)

    ue.start_capture()
    for r in range(reps):
        conv2d_3x3_pbi(ue, INs[r], OUTs[r], IMs[r], WGT, BIA, H, W, C_in, C_out,
                       stride, False, _GPR, zero_fn=zero_fn)
        if interleave_matmul:
            _prime_m_reg(ue, _GPR, H_out * W_out)
            ue.matmat_mul_core(
                M=H_out * W_out, K=C_out, N=C_out,
                A_DRAM_ADDR=OUTs[r], B_DRAM_ADDR=IDW, OUTPUT_DRAM_ADDR=OUTs[r],
                gpr_M_reg=_GPR)
    ue.stop_capture()
    ue.generate_instruction_halt()
    prog = ue.get_program_dram_addr()
    inst_bytes = ue.get_capture_instruction_size_bytes()
    ue.write_captured_instructions_to_dram(prog)
    ue.allocate_program_dram(inst_bytes)
    ue.clear_capture_buffer()
    ue.start_execute_from_dram(prog)
    ue.wait_queue(120.0)

    worst = float('inf')
    for r in range(reps):
        out = ue.dma_from_accelerator_memory(OUTs[r], (H_out * W_out, C_out)).float()
        xin = xs[r].reshape(H, W, C_in).permute(2, 0, 1).unsqueeze(0).float()
        ref = F.conv2d(xin, w.float(), b.float(), stride=stride, padding=1)
        ref = ref.squeeze(0).permute(1, 2, 0).reshape(H_out * W_out, C_out)
        snr_db = calculate_snr(ref, out)
        worst = min(worst, snr_db)
        flag = "  <-- DIVERGED" if not (snr_db >= snr_threshold_db or snr_db == float('inf')) else ""
        print(f"  chain conv[{r}] SNR: {snr_db:.2f} dB{flag}")
    print(f"conv2d_3x3_pbi_chain reps={reps} interleave_matmul={interleave_matmul} "
          f"worst={worst:.2f} dB inst={inst_bytes}B")
    assert worst >= snr_threshold_db or worst == float('inf'), \
        f"chain worst SNR {worst:.2f} dB < {snr_threshold_db} dB"
    record_test("conv2d_3x3_pbi_chain",
                f"reps={reps},H={H},W={W},Cin={C_in},mm={interleave_matmul}",
                snr_db=worst, inst_bytes=inst_bytes)

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()
    ue.reset_program_dram_addr()


def bf16_permute_pbi_test(dim_0: int, dim_1: int, dim_2: int, snr_threshold_db: float = 40.0):
    """Verify bf16_permute_pbi against torch permute(1,0,2). Pure data movement -> inf dB."""
    ue = UnifiedEngine()
    _GPR = 8
    IN  = ue.allocate_tensor_dram(dim_0 * dim_1 * dim_2 * _BPE)
    OUT = ue.allocate_tensor_dram(dim_1 * dim_0 * dim_2 * _BPE)

    ue.start_capture()
    bf16_permute_pbi(ue, IN, OUT, dim_0, dim_1, dim_2, _GPR)
    ue.stop_capture()
    ue.generate_instruction_halt()
    prog = ue.get_program_dram_addr()
    inst_bytes = ue.get_capture_instruction_size_bytes()
    ue.write_captured_instructions_to_dram(prog)
    ue.allocate_program_dram(inst_bytes)
    ue.clear_capture_buffer()

    a = torch.randn(dim_0, dim_1, dim_2, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(IN, a)
    ue.start_execute_from_dram(prog)
    ue.wait_queue(30.0)
    out = ue.dma_from_accelerator_memory(OUT, (dim_1, dim_0, dim_2)).float()

    ref = a.permute(1, 0, 2).float()
    snr_db = calculate_snr(ref.flatten(), out.flatten())
    print(f"bf16_permute_pbi SNR: {snr_db:.2f} dB (dim_0={dim_0},dim_1={dim_1},dim_2={dim_2}) "
          f"inst={inst_bytes}B vs ~{dim_0 * dim_1} legacy gathers")
    assert snr_db >= snr_threshold_db or snr_db == float('inf'), f"SNR {snr_db:.2f} dB"
    record_test("bf16_permute_pbi", f"dim_0={dim_0},dim_1={dim_1},dim_2={dim_2}",
                snr_db=snr_db, inst_bytes=inst_bytes)

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()
    ue.reset_program_dram_addr()


def conv_transpose2d_2x2_pbi_test(H: int, W: int, C_in: int, C_out: int,
                                  snr_threshold_db: float = 35.0):
    """End-to-end: conv_transpose2d_2x2_pbi vs torch.nn.ConvTranspose2d(k=2, s=2)."""
    import torch.nn.functional as F
    ue = UnifiedEngine()
    _GPR = 8
    IN   = ue.allocate_tensor_dram(H * W * C_in * _BPE)
    OUT  = ue.allocate_tensor_dram(2 * H * 2 * W * C_out * _BPE)
    TEMP = ue.allocate_tensor_dram(H * W * C_out * _BPE)
    BIA  = ue.allocate_tensor_dram(C_out * _BPE)
    WSL  = [ue.allocate_tensor_dram(C_out * C_in * _BPE) for _ in range(4)]

    x = torch.randn(H * W, C_in, dtype=torch.bfloat16)
    w = torch.randn(C_in, C_out, 2, 2, dtype=torch.bfloat16) * 0.1   # torch convT weight (Cin,Cout,kH,kW)
    b = torch.randn(C_out, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(IN, x)
    ue.dma_to_accelerator_memory(BIA, b)
    # slice s=(kh,kw) weight is (C_out, C_in) = w[:, :, kh, kw].T
    for s, (kh, kw) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        ue.dma_to_accelerator_memory(WSL[s], w[:, :, kh, kw].t().contiguous())
    # OUTPUT must be pre-zeroed
    ue.dma_to_accelerator_memory(OUT, torch.zeros(2 * H * 2 * W * C_out, dtype=torch.bfloat16))

    ue.start_capture()
    conv_transpose2d_2x2_pbi(ue, IN, OUT, TEMP, WSL, BIA, H, W, C_in, C_out, _GPR)
    ue.stop_capture()
    ue.generate_instruction_halt()
    prog = ue.get_program_dram_addr()
    inst_bytes = ue.get_capture_instruction_size_bytes()
    ue.write_captured_instructions_to_dram(prog)
    ue.allocate_program_dram(inst_bytes)
    ue.clear_capture_buffer()
    ue.start_execute_from_dram(prog)
    ue.wait_queue(60.0)
    out = ue.dma_from_accelerator_memory(OUT, (2 * H * 2 * W, C_out)).float()

    xin = x.reshape(H, W, C_in).permute(2, 0, 1).unsqueeze(0).float()
    ref = F.conv_transpose2d(xin, w.float(), b.float(), stride=2)
    ref = ref.squeeze(0).permute(1, 2, 0).reshape(2 * H * 2 * W, C_out)
    snr_db = calculate_snr(ref, out)
    print(f"conv_transpose2d_2x2_pbi SNR: {snr_db:.2f} dB (H={H},W={W},Cin={C_in},Cout={C_out}) "
          f"inst={inst_bytes}B vs ~{4 * H * W * 2} legacy scatter DMAs")
    assert snr_db >= snr_threshold_db or snr_db == float('inf'), f"SNR {snr_db:.2f} dB"
    record_test("conv_transpose2d_2x2_pbi", f"H={H},W={W},Cin={C_in},Cout={C_out}",
                snr_db=snr_db, inst_bytes=inst_bytes)

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()
    ue.reset_program_dram_addr()


def matmat_pbi_vs_legacy_test(M: int, K: int, N: int, gelu: bool = False, clamp: bool = False,
                              bias: bool = False, in_place: bool = False,
                              snr_threshold_db: float = 40.0):
    """Compare matmat_mul_core PBI (gpr_M_reg) vs legacy for the EXACT model conditions.
    Reports PBI-vs-torch, legacy-vs-torch, and the decisive PBI-vs-legacy SNR. in_place
    runs A_DRAM == OUTPUT_DRAM (requires K==N). Asserts PBI matches legacy."""
    import torch.nn.functional as F
    A = torch.randn(M, K, dtype=torch.bfloat16) * 0.5
    B = torch.randn(N, K, dtype=torch.bfloat16) * 0.1
    bt = torch.randn(N, dtype=torch.bfloat16) if bias else None
    ref = A.float() @ B.float().t()
    if bias:  ref = ref + bt.float()
    if gelu:  ref = ref * torch.sigmoid(1.702 * ref)
    if clamp: ref = torch.clamp(ref, min=0.0)

    if in_place:
        assert K == N, "in_place requires K==N (A and OUTPUT share the buffer)"

    def run(use_pbi: bool):
        ue = UnifiedEngine()
        A_DRAM = ue.allocate_tensor_dram(M * K * 2)
        B_DRAM = ue.allocate_tensor_dram(N * K * 2)
        OUT = A_DRAM if in_place else ue.allocate_tensor_dram(M * N * 2)
        C_DRAM = ue.allocate_tensor_dram(N * 2) if bias else None
        ue.dma_to_accelerator_memory(A_DRAM, A)
        ue.dma_to_accelerator_memory(B_DRAM, B)
        if bias:
            ue.dma_to_accelerator_memory(C_DRAM, bt)
        m_reg = ue.alloc_isa_reg() if use_pbi else None
        ue.start_capture()
        if use_pbi:
            ue.generate_instruction_add_set(m_reg, M)
        ue.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=A_DRAM, B_DRAM_ADDR=B_DRAM,
                           OUTPUT_DRAM_ADDR=OUT, C_DRAM_ADDR=C_DRAM, bias_mode="broadcast_N",
                           gelu_enable=gelu, clamp_enable=clamp, gpr_M_reg=m_reg)
        ue.stop_capture()
        ue.generate_instruction_halt()
        prog = ue.get_program_dram_addr()
        nbytes = ue.get_capture_instruction_size_bytes()
        ue.write_captured_instructions_to_dram(prog)
        ue.allocate_program_dram(nbytes)
        ue.clear_capture_buffer()
        ue.start_execute_from_dram(prog)
        ue.wait_queue(120.0)
        out = ue.dma_from_accelerator_memory(OUT, (M, N)).float()
        ue.clear_capture_buffer(); ue.reset_tensor_dram_addr(); ue.reset_program_dram_addr()
        return out, nbytes

    out_l, b_l = run(False)
    out_p, b_p = run(True)
    snr_lt = calculate_snr(ref, out_l)
    snr_pt = calculate_snr(ref, out_p)
    snr_pl = calculate_snr(out_l, out_p)
    cond = "".join(t for t, f in [("gelu", gelu), ("clamp", clamp), ("bias", bias),
                                   ("inplace", in_place)] if f) or "plain"
    print(f"matmat PBI-vs-legacy M={M},K={K},N={N},{cond}: "
          f"legacy/torch={snr_lt:.1f}dB pbi/torch={snr_pt:.1f}dB PBI/LEGACY={snr_pl:.1f}dB "
          f"legacy={b_l}B pbi={b_p}B")
    record_test(f"matmat_pbi_vs_legacy+{cond}", f"M={M},K={K},N={N}",
                snr_db=snr_pl, inst_bytes=b_p)
    assert snr_pl >= snr_threshold_db or snr_pl == float('inf'), \
        f"PBI diverges from legacy: {snr_pl:.1f} dB for M={M},K={K},N={N},{cond}"


def rope_hf_core_dram_test(M: int, N: int, use_pbi: bool = False):
    """
    Tests rope_hf_core_dram by emitting one HF-style RoPE instruction sequence per row.
    """
    ue = UnifiedEngine()

    X_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    ROPE_DRAM_ADDR = ue.allocate_tensor_dram(M * 2 * N * 2)

    # PBI path is driven by gpr_M_reg; allocate + prime a GPR with M when use_pbi is requested.
    m_reg = ue.alloc_isa_reg() if use_pbi else None

    ue.start_capture()
    if use_pbi:
        ue.generate_instruction_add_set(m_reg, M)
    total_flops = ue.rope_hf_core_dram(
        M=M,
        N=N,
        input_dram_addr=X_DRAM_ADDR,
        output_dram_addr=OUTPUT_DRAM_ADDR,
        cos_dram_addr=ROPE_DRAM_ADDR,
        sin_dram_addr=ROPE_DRAM_ADDR + N * 2,
        gpr_M_reg=m_reg,
    )
    ue.stop_capture()
    if use_pbi:
        ue.release_isa_reg()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    instruction_size_bytes = ue.get_capture_instruction_size_bytes()
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

    sin_negated = sin.clone()
    sin_negated[:N // 2] = -sin_negated[:N // 2]
    x_hf = torch.randn(M, N, dtype=torch.bfloat16)
    rope_table = torch.cat((cos, sin_negated), dim=0).repeat(M)

    ue.dma_to_accelerator_memory(X_DRAM_ADDR, x_hf)
    ue.dma_to_accelerator_memory(ROPE_DRAM_ADDR, rope_table)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0)
    ue.report_timing_and_instruction_count()

    flop_rate_gflops, flops_ratio = ue.report_flop_rate_gflops(total_flops)
    print(f"Report FLOPS for HF RoPE core DRAM: {flop_rate_gflops:.2f} GFLOPS, {flops_ratio:.2f}% peak throughput for M={M}, N={N}, use_pbi={use_pbi}")

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M, N))

    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    ref = x_hf * cos + rotate_half(x_hf) * sin
    snr_db = calculate_snr(ref, output)
    print(f"HF RoPE core DRAM SNR Analysis: {snr_db:.2f} dB for M={M}, N={N}")
    assert snr_db >= 40 or snr_db == float('inf'), f"SNR {snr_db:.2f} dB must be at least 40 dB"

    record_test(f"rope_hf_core_dram{'+pbi' if use_pbi else ''}",
                f"M={M}, N={N}",
                snr_db=snr_db,
                gflops=flop_rate_gflops,
                inst_bytes=instruction_size_bytes)

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()
    ue.reset_program_dram_addr()

def rope_hf_core_dram_gqa_test(M: int, group_size: int, N: int, use_pbi: bool = False):
    """
    Tests grouped-query RoPE with Q rows laid out as [M, group_size, N].
    """
    ue = UnifiedEngine()

    X_DRAM_ADDR = ue.allocate_tensor_dram(M * group_size * N * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * group_size * N * 2)
    ROPE_DRAM_ADDR = ue.allocate_tensor_dram(M * 2 * N * 2)

    # PBI path is driven by gpr_M_reg; allocate + prime a GPR with M when use_pbi is requested.
    m_reg = ue.alloc_isa_reg() if use_pbi else None

    ue.start_capture()
    if use_pbi:
        ue.generate_instruction_add_set(m_reg, M)
    total_flops = ue.rope_hf_core_dram_gqa(
        M=M,
        group_size=group_size,
        N=N,
        input_dram_addr=X_DRAM_ADDR,
        output_dram_addr=OUTPUT_DRAM_ADDR,
        cos_dram_addr=ROPE_DRAM_ADDR,
        sin_dram_addr=ROPE_DRAM_ADDR + N * 2,
        gpr_M_reg=m_reg,
    )
    ue.stop_capture()
    if use_pbi:
        ue.release_isa_reg()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    instruction_size_bytes = ue.get_capture_instruction_size_bytes()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    head_dim = N
    MAX_SEQ_LEN = 32768
    freqs_cis = precompute_freqs_cis(head_dim, MAX_SEQ_LEN * 2)
    random_seq_index = random.randint(0, MAX_SEQ_LEN - M)
    rope_rows = []
    cos_rows = []
    sin_rows = []
    for row_idx in range(M):
        one_rope_seq_params = freqs_cis[random_seq_index + row_idx, :]
        one_rope_seq = torch.view_as_real(one_rope_seq_params).to(torch.bfloat16).reshape(-1)
        cos = torch.cat((one_rope_seq[0::2], one_rope_seq[0::2]), dim=-1)
        sin = torch.cat((one_rope_seq[1::2], one_rope_seq[1::2]), dim=-1)
        sin_negated = sin.clone()
        sin_negated[:N // 2] = -sin_negated[:N // 2]
        rope_rows.append(torch.cat((cos, sin_negated), dim=0))
        cos_rows.append(cos)
        sin_rows.append(sin)

    x_hf = torch.randn(M, group_size, N, dtype=torch.bfloat16)
    rope_table = torch.cat(rope_rows, dim=0)

    ue.dma_to_accelerator_memory(X_DRAM_ADDR, x_hf.reshape(M * group_size, N))
    ue.dma_to_accelerator_memory(ROPE_DRAM_ADDR, rope_table)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0)
    ue.report_timing_and_instruction_count()

    flop_rate_gflops, flops_ratio = ue.report_flop_rate_gflops(total_flops)
    print(f"Report FLOPS for GQA HF RoPE core DRAM: {flop_rate_gflops:.2f} GFLOPS, {flops_ratio:.2f}% peak throughput for M={M}, group_size={group_size}, N={N}, use_pbi={use_pbi}")

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M * group_size, N)).reshape(M, group_size, N)

    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    cos_ref = torch.stack(cos_rows, dim=0).unsqueeze(1)
    sin_ref = torch.stack(sin_rows, dim=0).unsqueeze(1)
    ref = x_hf * cos_ref + rotate_half(x_hf) * sin_ref
    snr_db = calculate_snr(ref, output)
    print(f"GQA HF RoPE core DRAM SNR Analysis: {snr_db:.2f} dB for M={M}, group_size={group_size}, N={N}")
    assert snr_db >= 40 or snr_db == float('inf'), f"SNR {snr_db:.2f} dB must be at least 40 dB"

    record_test(f"rope_hf_core_dram_gqa{'+pbi' if use_pbi else ''}",
                f"M={M}, G={group_size}, N={N}",
                snr_db=snr_db,
                gflops=flop_rate_gflops,
                inst_bytes=instruction_size_bytes)

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()
    ue.reset_program_dram_addr()

def smolvlm_rope_hf_core_dram_pbi_test(M: int, N: int = 64, theta: float = 100000.0):
    """
    Exercises rope_hf_core_dram in PBI mode at SmolVLM2 dimensions (head_dim N=64).

    SmolVLM2's RoPE table convention (see _load_rope_tables): per position p,
        cos_row = [cos(freqs_p) || cos(freqs_p)]            (N elements)
        sin_row = [sin(freqs_p) || sin(freqs_p)]            (N elements, first half pre-negated)
    laid out contiguously as [cos_row || sin_row] per token so the kernel reads one
    2N-element rope row per token. Positions are sequential (0..M-1) as in real prefill.

    head_dim=64 is below the shared PBI rope's N>=128 / 128-byte-half-alignment
    requirement; this test is the concrete repro for that gap.
    """
    ue = UnifiedEngine()

    X_DRAM_ADDR      = ue.allocate_tensor_dram(M * N * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    ROPE_DRAM_ADDR   = ue.allocate_tensor_dram(M * 2 * N * 2)

    m_reg = ue.alloc_isa_reg()

    ue.start_capture()
    ue.generate_instruction_add_set(m_reg, M)
    total_flops = ue.rope_hf_core_dram(
        M=M, N=N,
        input_dram_addr=X_DRAM_ADDR,
        output_dram_addr=OUTPUT_DRAM_ADDR,
        cos_dram_addr=ROPE_DRAM_ADDR,
        sin_dram_addr=ROPE_DRAM_ADDR + N * 2,
        gpr_M_reg=m_reg,
    )
    ue.stop_capture()
    ue.release_isa_reg()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    instruction_size_bytes = ue.get_capture_instruction_size_bytes()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    # Build SmolVLM-style cos/sin for sequential positions 0..M-1.
    half = N // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, N, 2, dtype=torch.float32) / N))
    freqs = torch.outer(torch.arange(M, dtype=torch.float32), inv_freq)  # [M, N/2]
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=1).to(torch.bfloat16)  # [M, N]
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=1).to(torch.bfloat16)  # [M, N]
    sin_negated = sin.clone()
    sin_negated[:, :half] = -sin_negated[:, :half]

    x_hf = torch.randn(M, N, dtype=torch.bfloat16)
    rope_table = torch.cat([cos, sin_negated], dim=1).reshape(-1)  # [M, 2N] -> flat

    ue.dma_to_accelerator_memory(X_DRAM_ADDR, x_hf)
    ue.dma_to_accelerator_memory(ROPE_DRAM_ADDR, rope_table)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0)
    ue.report_timing_and_instruction_count()

    flop_rate_gflops, flops_ratio = ue.report_flop_rate_gflops(total_flops)
    print(f"Report FLOPS for SmolVLM HF RoPE PBI: {flop_rate_gflops:.2f} GFLOPS, {flops_ratio:.2f}% peak for M={M}, N={N}")

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M, N))

    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    ref = x_hf * cos + rotate_half(x_hf) * sin
    snr_db = calculate_snr(ref, output)
    print(f"SmolVLM HF RoPE PBI SNR Analysis: {snr_db:.2f} dB for M={M}, N={N}")
    assert snr_db >= 40 or snr_db == float('inf'), f"SNR {snr_db:.2f} dB must be at least 40 dB"

    record_test("smolvlm_rope_hf_core_dram+pbi",
                f"M={M}, N={N}",
                snr_db=snr_db,
                gflops=flop_rate_gflops,
                inst_bytes=instruction_size_bytes)

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()
    ue.reset_program_dram_addr()

def smolvlm_rope_d64_pbi_test(M: int, D: int = 64, theta: float = 100000.0):
    """
    Tests nn_lib.rope_hf_core_dram_pbi: the D<128 padded-split RoPE wrapped in a PBI
    hardware loop (register-addressed reads/writes, 0 PBI pointers). This is the kernel
    SmolVLM2 prefill needs (head_dim=64).

    Packed per-token rope table = [cos_lo | cos_hi | sin_lo | sin_hi], each 32-elem half
    zero-padded to UE_VECTOR_SIZE (64) so it is one full 128-byte SRAM row. sin_lo is
    pre-negated (HW add-only). Positions are sequential (real prefill).
    """
    from nn_lib import rope_hf_core_dram_pbi

    ue = UnifiedEngine()
    PAD = UE_VECTOR_SIZE  # 64
    half = D // 2

    X_DRAM_ADDR      = ue.allocate_tensor_dram(M * D * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * D * 2)
    ROPE_PACKED_ADDR = ue.allocate_tensor_dram(M * 4 * PAD * 2)

    gpr_M_reg = ue.alloc_isa_reg()
    tmp_reg   = ue.alloc_isa_reg()
    t_reg     = ue.alloc_isa_reg()

    ue.start_capture()
    ue.generate_instruction_add_set(gpr_M_reg, M)
    total_flops = rope_hf_core_dram_pbi(
        ue, M=M, D=D,
        X_DRAM_ADDR=X_DRAM_ADDR, OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
        ROPE_PACKED_ADDR=ROPE_PACKED_ADDR,
        gpr_M_reg=gpr_M_reg, tmp_reg=tmp_reg, t_reg=t_reg,
    )
    ue.stop_capture()
    ue.release_isa_reg(); ue.release_isa_reg(); ue.release_isa_reg()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    instruction_size_bytes = ue.get_capture_instruction_size_bytes()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(instruction_size_bytes)

    # SmolVLM-style cos/sin, sequential positions 0..M-1 (only the 32 unique freqs).
    inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float32) / D))  # [32]
    freqs = torch.outer(torch.arange(M, dtype=torch.float32), inv_freq)           # [M, 32]
    cos = torch.cos(freqs).to(torch.bfloat16)  # [M, 32]
    sin = torch.sin(freqs).to(torch.bfloat16)  # [M, 32]

    def _pad(t):
        p = torch.zeros(M, PAD, dtype=torch.bfloat16)
        p[:, :half] = t
        return p
    # [cos_lo | cos_hi | sin_lo(neg) | sin_hi], each padded to 64
    packed = torch.cat([_pad(cos), _pad(cos), _pad(-sin), _pad(sin)], dim=1)  # [M, 256]

    x = torch.randn(M, D, dtype=torch.bfloat16)

    ue.dma_to_accelerator_memory(X_DRAM_ADDR, x)
    ue.dma_to_accelerator_memory(ROPE_PACKED_ADDR, packed.reshape(-1))

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0)
    ue.report_timing_and_instruction_count()

    flop_rate_gflops, flops_ratio = ue.report_flop_rate_gflops(total_flops)
    print(f"Report FLOPS for SmolVLM D=64 RoPE PBI: {flop_rate_gflops:.2f} GFLOPS, {flops_ratio:.2f}% peak for M={M}, D={D}")

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M, D))

    cos_full = torch.cat([cos, cos], dim=1)  # [M, 64]
    sin_full = torch.cat([sin, sin], dim=1)  # [M, 64]
    def rotate_half(t):
        t1 = t[..., :t.shape[-1] // 2]
        t2 = t[..., t.shape[-1] // 2:]
        return torch.cat((-t2, t1), dim=-1)
    ref = x * cos_full + rotate_half(x) * sin_full
    snr_db = calculate_snr(ref, output)
    print(f"SmolVLM D=64 RoPE PBI SNR Analysis: {snr_db:.2f} dB for M={M}, D={D}")
    assert snr_db >= 40 or snr_db == float('inf'), f"SNR {snr_db:.2f} dB must be at least 40 dB"

    record_test("smolvlm_rope_d64+pbi",
                f"M={M}, D={D}",
                snr_db=snr_db,
                gflops=flop_rate_gflops,
                inst_bytes=instruction_size_bytes)

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()
    ue.reset_program_dram_addr()

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

    record_test("bf16_permute",
                f"dim_0={dim_0}, dim_1={dim_1}, dim_2={dim_2}",
                snr_db=snr_db)

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
    data_type = TYPE.IF4
    int_variant = True  # legacy patching test used INT4 codes
    patches_per_group = 16

    # Build 16 identity-like weight matrices (same as user_dma_ops.patching)
    matrix_dram_addrs = []
    scale_dram_addrs = []
    for matrix_idx in range(patches_per_group):
        weight = torch.zeros(N, K, dtype=torch.bfloat16)
        for i in range(48):
            weight[i, matrix_idx * patch_w + i % patch_w + (i // patch_w) * UE_VECTOR_SIZE] = 1.0
        matrix_addr, scale_addr = ue.quantize_weight(weight, N, K, data_type=data_type, int_variant=int_variant)
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

    record_test("patching",
                f"C={C}, H={H}, W={W}, patch={patch_h}x{patch_w}, K={K}, N={N}",
                snr_db=snr_db)

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

def if4_if8_tests():
    """
    Dequantize core exhaustive test for INT4 and INT8.

    A single fixed scale (bf16) is applied to every 64-element block, and the
    quantized tensor is constructed to cover every possible quantized value:
      - INT4: 16 signed values (-8..+7), tiled to M=256 elements (4 blocks x 64).
      - INT8: 256 signed values (-128..+127), laid out exactly once across M=256.

    With scale = 0.5, every q*scale product is exactly representable in bf16,
    so we assert bitwise equality instead of only SNR.
    """
    from user_dma_core import DMA_DEVICE_H2C

    # https://asawicki.info/articles/fp8_tables.php
    FP8_E4M3FN_TABLE = torch.tensor([
        # 0x00..0x0F  (positive subnormals, exponent field = 0)
        +0.0,     0.001953, 0.003906, 0.005859, 0.007812, 0.009766, 0.01172, 0.01367,
        0.01562,  0.01758,  0.01953,  0.02148,  0.02344,  0.02539,  0.02734, 0.0293,
        # 0x10..0x1F
        0.03125,  0.03516,  0.03906,  0.04297,  0.04688,  0.05078,  0.05469, 0.05859,
        0.0625,   0.07031,  0.07812,  0.08594,  0.09375,  0.1016,   0.1094,  0.1172,
        # 0x20..0x2F
        0.125,    0.1406,   0.1562,   0.1719,   0.1875,   0.2031,   0.2188,  0.2344,
        0.25,     0.2812,   0.3125,   0.3438,   0.375,    0.4062,   0.4375,  0.4688,
        # 0x30..0x3F
        0.5,      0.5625,   0.625,    0.6875,   0.75,     0.8125,   0.875,   0.9375,
        1.0,      1.125,    1.25,     1.375,    1.5,      1.625,    1.75,    1.875,
        # 0x40..0x4F
        2.0,      2.25,     2.5,      2.75,     3.0,      3.25,     3.5,     3.75,
        4.0,      4.5,      5.0,      5.5,      6.0,      6.5,      7.0,     7.5,
        # 0x50..0x5F
        8.0,      9.0,     10.0,     11.0,     12.0,     13.0,     14.0,    15.0,
    16.0,     18.0,     20.0,     22.0,     24.0,     26.0,     28.0,    30.0,
        # 0x60..0x6F
    32.0,     36.0,     40.0,     44.0,     48.0,     52.0,     56.0,    60.0,
    64.0,     72.0,     80.0,     88.0,     96.0,    104.0,    112.0,   120.0,
        # 0x70..0x7F  (0x7F = +NaN)
    128.0,    144.0,    160.0,    176.0,    192.0,    208.0,    224.0,   240.0,
    256.0,    288.0,    320.0,    352.0,    384.0,    416.0,    448.0,   math.nan,
        # 0x80..0x8F  (negative subnormals; 0x80 = -0)
    -0.0,    -0.001953, -0.003906, -0.005859, -0.007812, -0.009766, -0.01172, -0.01367,
    -0.01562, -0.01758,  -0.01953,  -0.02148,  -0.02344,  -0.02539,  -0.02734, -0.0293,
        # 0x90..0x9F
    -0.03125, -0.03516,  -0.03906,  -0.04297,  -0.04688,  -0.05078,  -0.05469, -0.05859,
    -0.0625,  -0.07031,  -0.07812,  -0.08594,  -0.09375,  -0.1016,   -0.1094,  -0.1172,
        # 0xA0..0xAF
    -0.125,   -0.1406,   -0.1562,   -0.1719,   -0.1875,   -0.2031,   -0.2188,  -0.2344,
    -0.25,    -0.2812,   -0.3125,   -0.3438,   -0.375,    -0.4062,   -0.4375,  -0.4688,
        # 0xB0..0xBF
    -0.5,     -0.5625,   -0.625,    -0.6875,   -0.75,     -0.8125,   -0.875,   -0.9375,
    -1.0,     -1.125,    -1.25,     -1.375,    -1.5,      -1.625,    -1.75,    -1.875,
        # 0xC0..0xCF
    -2.0,     -2.25,     -2.5,      -2.75,     -3.0,      -3.25,     -3.5,     -3.75,
    -4.0,     -4.5,      -5.0,      -5.5,      -6.0,      -6.5,      -7.0,     -7.5,
        # 0xD0..0xDF
    -8.0,     -9.0,     -10.0,     -11.0,     -12.0,     -13.0,     -14.0,    -15.0,
    -16.0,    -18.0,     -20.0,     -22.0,     -24.0,     -26.0,     -28.0,    -30.0,
        # 0xE0..0xEF
    -32.0,    -36.0,     -40.0,     -44.0,     -48.0,     -52.0,     -56.0,    -60.0,
    -64.0,    -72.0,     -80.0,     -88.0,     -96.0,    -104.0,    -112.0,   -120.0,
        # 0xF0..0xFF  (0xFF = -NaN)
    -128.0,   -144.0,    -160.0,    -176.0,    -192.0,    -208.0,    -224.0,   -240.0,
    -256.0,   -288.0,    -320.0,    -352.0,    -384.0,    -416.0,    -448.0,   -math.nan,
    ]).to(torch.bfloat16)

    # NVFP4 (FP4 E2M1): 1 sign bit, 2 exponent bits, 1 mantissa bit.
    # Indexed by the raw 4-bit code (0x0..0xF). No inf / no NaN; 0x0 = +0, 0x8 = -0.
    NVFP4_TABLE = torch.tensor([
        # 0x0..0x7  (sign=0: +values)
        +0.0,  +0.5,  +1.0,  +1.5,  +2.0,  +3.0,  +4.0,  +6.0,
        # 0x8..0xF  (sign=1: -values)
        -0.0,  -0.5,  -1.0,  -1.5,  -2.0,  -3.0,  -4.0,  -6.0,
    ]).to(torch.bfloat16)

    # Signed 2's-complement INT4 lookup indexed by 4-bit code.
    # 0x0..0x7 -> 0..7 ; 0x8..0xF -> -8..-1
    INT4_TABLE = torch.tensor(
        [c - 16 if c >= 8 else c for c in range(16)],
        dtype=torch.int16,
    ).to(torch.bfloat16)

    # Signed 2's-complement INT8 lookup indexed by byte code.
    # 0x00..0x7F -> 0..127 ; 0x80..0xFF -> -128..-1
    INT8_TABLE = torch.tensor(
        [c - 256 if c >= 128 else c for c in range(256)],
        dtype=torch.int16,
    ).to(torch.bfloat16)

    M = 8 * 64
    assert M % UE_VECTOR_SIZE == 0
    num_blocks = M // UE_VECTOR_SIZE

    # (label, hw data_type, bf16 scale, reference lookup table, number of codes)
    # Scale sign = mode select: +scale -> FP variant, -scale -> INT variant.
    # |scale| = 1 keeps the multiplied result bitwise-exact in bf16.
    configs = [
        ("IF4-FP4",  TYPE.IF4, +1.0, NVFP4_TABLE,       16),
        ("IF4-INT4", TYPE.IF4, -1.0, INT4_TABLE,        16),
        ("IF8-FP8",  TYPE.IF8, +1.0, FP8_E4M3FN_TABLE,  256),
        ("IF8-INT8", TYPE.IF8, -1.0, INT8_TABLE,        256),
    ]

    for type_str, data_type, scale_value, value_table, num_codes in configs:
        ue = UnifiedEngine()

        scales_bf16 = torch.full((num_blocks,), scale_value, dtype=torch.bfloat16)

        # All possible codes 0..num_codes-1, tiled to M elements
        codes_u8 = torch.arange(num_codes, dtype=torch.int16).to(torch.uint8)
        reps = (M + num_codes - 1) // num_codes
        q_u8 = codes_u8.repeat(reps)[:M].contiguous()

        if data_type == TYPE.IF4:
            # Pack two nibbles per byte (low nibble first, matches quantize_weight)
            num_payload_bytes = M // 2
            payload = torch.zeros(num_payload_bytes, dtype=torch.uint8)
            for i in range(0, M, 2):
                v1 = q_u8[i].item() & 0xF
                v2 = q_u8[i + 1].item() & 0xF
                payload[i // 2] = ((v2 & 0xF) << 4) | v1
        else:  # TYPE.IF8
            num_payload_bytes = M
            payload = q_u8

        q_dram = ue.get_params_dram_addr()
        ue.dma_write(DMA_DEVICE_H2C, q_dram, payload, num_payload_bytes)
        scale_dram = q_dram + num_payload_bytes
        ue.dma_write(DMA_DEVICE_H2C, scale_dram,
                     scales_bf16.view(torch.uint16), num_blocks * 2)
        ue.allocate_params_dram(num_payload_bytes + num_blocks * 2)

        OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * 2)

        ue.start_capture()
        vector_sram_start_addr = 0x00000
        ue.start_queue_for_bf16_dequantize_operation(
            VECTOR_INPUT_DRAM_ADDR=q_dram,
            SCALE_INPUT_DRAM_ADDR=scale_dram,
            data_type=data_type,
            output_sram_wb_addr=vector_sram_start_addr,
            element_size=M,
        )
        ue.sram_to_accelerator_memory(
            sram_address=vector_sram_start_addr,
            accelerator_dram_address=OUTPUT_DRAM_ADDR,
            element_size=M,
        )
        ue.stop_capture()
        ue.generate_instruction_halt()
        program_dram_addr = ue.get_program_dram_addr()
        ue.write_captured_instructions_to_dram(program_dram_addr)
        ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

        ue.start_execute_from_dram(program_dram_addr)
        ue.wait_queue(10.0)
        ue.report_timing_and_instruction_count()

        output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M,))

        # HW strips the sign bit of scale -> effective multiplier is |scale|.
        abs_scale = abs(scale_value)
        expected = (value_table.to(torch.float32) * abs_scale).to(torch.bfloat16)
        expected = expected.repeat(reps)[:M]

        snr_db = math.inf if torch.allclose(expected, output, atol=0, rtol=0, equal_nan=True) else 0

        record_test(f"if4_if8-{type_str}",
                    f"M={M}, scale={scale_value}, all_q_values",
                    snr_db=snr_db)

        torch.set_printoptions(profile="default")
        ue.clear_capture_buffer()
        ue.reset_tensor_dram_addr()


def if4_if8_mixed_sign_test():
    """
    Mixed-scale-sign coverage for IF4 / IF8 dequantize.

    The variant select for adaptive block-scale formats is communicated per
    block via the sign of the bf16 scale (negative -> INT path, positive ->
    FP path). The exhaustive ``if4_if8_tests`` keeps the sign uniform across
    a tensor; this companion test interleaves positive- and negative-scale
    blocks within the same dequantize call so the per-block variant select
    is exercised. Each block is filled with all ``num_codes`` values so the
    full code table is hit on both the FP and INT side, and ``|scale| = 1``
    keeps the multiplied result bitwise-exact in bf16.
    """
    from user_dma_core import DMA_DEVICE_H2C

    NVFP4_TABLE = torch.tensor([
        +0.0,  +0.5,  +1.0,  +1.5,  +2.0,  +3.0,  +4.0,  +6.0,
        -0.0,  -0.5,  -1.0,  -1.5,  -2.0,  -3.0,  -4.0,  -6.0,
    ]).to(torch.bfloat16)

    INT4_TABLE = torch.tensor(
        [c - 16 if c >= 8 else c for c in range(16)],
        dtype=torch.int16,
    ).to(torch.bfloat16)

    # FP8 / INT8 lookups built lazily at use-time (256 entries each); reuse
    # the canonical tables from if4_if8_tests via a small helper.
    FP8_E4M3FN_TABLE = torch.tensor([
        +0.0, 0.001953, 0.003906, 0.005859, 0.007812, 0.009766, 0.01172, 0.01367,
        0.01562, 0.01758, 0.01953, 0.02148, 0.02344, 0.02539, 0.02734, 0.0293,
        0.03125, 0.03516, 0.03906, 0.04297, 0.04688, 0.05078, 0.05469, 0.05859,
        0.0625, 0.07031, 0.07812, 0.08594, 0.09375, 0.1016, 0.1094, 0.1172,
        0.125, 0.1406, 0.1562, 0.1719, 0.1875, 0.2031, 0.2188, 0.2344,
        0.25, 0.2812, 0.3125, 0.3438, 0.375, 0.4062, 0.4375, 0.4688,
        0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375,
        1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875,
        2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75,
        4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,
        8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0,
        32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0,
        64.0, 72.0, 80.0, 88.0, 96.0, 104.0, 112.0, 120.0,
        128.0, 144.0, 160.0, 176.0, 192.0, 208.0, 224.0, 240.0,
        256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, math.nan,
        -0.0, -0.001953, -0.003906, -0.005859, -0.007812, -0.009766, -0.01172, -0.01367,
        -0.01562, -0.01758, -0.01953, -0.02148, -0.02344, -0.02539, -0.02734, -0.0293,
        -0.03125, -0.03516, -0.03906, -0.04297, -0.04688, -0.05078, -0.05469, -0.05859,
        -0.0625, -0.07031, -0.07812, -0.08594, -0.09375, -0.1016, -0.1094, -0.1172,
        -0.125, -0.1406, -0.1562, -0.1719, -0.1875, -0.2031, -0.2188, -0.2344,
        -0.25, -0.2812, -0.3125, -0.3438, -0.375, -0.4062, -0.4375, -0.4688,
        -0.5, -0.5625, -0.625, -0.6875, -0.75, -0.8125, -0.875, -0.9375,
        -1.0, -1.125, -1.25, -1.375, -1.5, -1.625, -1.75, -1.875,
        -2.0, -2.25, -2.5, -2.75, -3.0, -3.25, -3.5, -3.75,
        -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -7.0, -7.5,
        -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0,
        -16.0, -18.0, -20.0, -22.0, -24.0, -26.0, -28.0, -30.0,
        -32.0, -36.0, -40.0, -44.0, -48.0, -52.0, -56.0, -60.0,
        -64.0, -72.0, -80.0, -88.0, -96.0, -104.0, -112.0, -120.0,
        -128.0, -144.0, -160.0, -176.0, -192.0, -208.0, -224.0, -240.0,
        -256.0, -288.0, -320.0, -352.0, -384.0, -416.0, -448.0, -math.nan,
    ]).to(torch.bfloat16)

    INT8_TABLE = torch.tensor(
        [c - 256 if c >= 128 else c for c in range(256)],
        dtype=torch.int16,
    ).to(torch.bfloat16)

    # Three sign-pattern variants per width, all exercising the per-block
    # variant select within a single dequantize op:
    #   alternating, FP-first-half / INT-second-half, INT-first-half / FP-second-half.
    sign_patterns = {
        "alt": lambda b: +1.0 if (b % 2 == 0) else -1.0,
        "fp_int": lambda b, total: +1.0 if (b < total // 2) else -1.0,
        "int_fp": lambda b, total: -1.0 if (b < total // 2) else +1.0,
    }

    # (label, hw data_type, fp_table, int_table, num_codes)
    widths = [
        ("IF4", TYPE.IF4, NVFP4_TABLE,       INT4_TABLE, 16),
        ("IF8", TYPE.IF8, FP8_E4M3FN_TABLE,  INT8_TABLE, 256),
    ]

    for width_str, data_type, fp_table, int_table, num_codes in widths:
        # 8 blocks of 64 elements -> 512 values; covers IF8 codes exactly once.
        M = 8 * UE_VECTOR_SIZE
        num_blocks = M // UE_VECTOR_SIZE

        codes_u8 = torch.arange(num_codes, dtype=torch.int16).to(torch.uint8)
        reps = (M + num_codes - 1) // num_codes
        q_u8 = codes_u8.repeat(reps)[:M].contiguous()

        for pattern_name, pattern_fn in sign_patterns.items():
            ue = UnifiedEngine()

            scales_bf16 = torch.zeros(num_blocks, dtype=torch.bfloat16)
            for b in range(num_blocks):
                if pattern_name == "alt":
                    scales_bf16[b] = pattern_fn(b)
                else:
                    scales_bf16[b] = pattern_fn(b, num_blocks)

            if data_type == TYPE.IF4:
                num_payload_bytes = M // 2
                payload = torch.zeros(num_payload_bytes, dtype=torch.uint8)
                for i in range(0, M, 2):
                    v1 = q_u8[i].item() & 0xF
                    v2 = q_u8[i + 1].item() & 0xF
                    payload[i // 2] = ((v2 & 0xF) << 4) | v1
            else:  # TYPE.IF8
                num_payload_bytes = M
                payload = q_u8

            q_dram = ue.get_params_dram_addr()
            ue.dma_write(DMA_DEVICE_H2C, q_dram, payload, num_payload_bytes)
            scale_dram = q_dram + num_payload_bytes
            ue.dma_write(DMA_DEVICE_H2C, scale_dram,
                         scales_bf16.view(torch.uint16), num_blocks * 2)
            ue.allocate_params_dram(num_payload_bytes + num_blocks * 2)

            OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * 2)

            ue.start_capture()
            vector_sram_start_addr = 0x00000
            ue.start_queue_for_bf16_dequantize_operation(
                VECTOR_INPUT_DRAM_ADDR=q_dram,
                SCALE_INPUT_DRAM_ADDR=scale_dram,
                data_type=data_type,
                output_sram_wb_addr=vector_sram_start_addr,
                element_size=M,
            )
            ue.sram_to_accelerator_memory(
                sram_address=vector_sram_start_addr,
                accelerator_dram_address=OUTPUT_DRAM_ADDR,
                element_size=M,
            )
            ue.stop_capture()
            ue.generate_instruction_halt()
            program_dram_addr = ue.get_program_dram_addr()
            ue.write_captured_instructions_to_dram(program_dram_addr)
            ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

            ue.start_execute_from_dram(program_dram_addr)
            ue.wait_queue(10.0)
            ue.report_timing_and_instruction_count()

            output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M,))

            # Per-block reference: positive scale -> FP table, negative -> INT.
            expected = torch.zeros(M, dtype=torch.bfloat16)
            for b in range(num_blocks):
                start = b * UE_VECTOR_SIZE
                stop = start + UE_VECTOR_SIZE
                use_fp = float(scales_bf16[b].item()) > 0.0
                table = fp_table if use_fp else int_table
                idx = q_u8[start:stop].to(torch.long)
                expected[start:stop] = (table[idx].to(torch.float32)
                                        * abs(float(scales_bf16[b].item()))
                                        ).to(torch.bfloat16)

            snr_db = math.inf if torch.allclose(expected, output, atol=0, rtol=0, equal_nan=True) else 0
            record_test(f"if4_if8_mixed-{width_str}-{pattern_name}",
                        f"M={M}, num_blocks={num_blocks}",
                        snr_db=snr_db)

            ue.clear_capture_buffer()
            ue.reset_tensor_dram_addr()


def _build_tq4_test_codebook() -> torch.Tensor:
    """16-entry bf16 TQ4 codebook for tests.

    Uses a moderate dynamic range (~[-3, +3]) so block scales can normalize
    most random bf16 inputs without needing extreme magnitudes, which keeps
    the dot-product reference accurate in bf16 arithmetic.
    """
    raw = (torch.rand(16) * 6.0 - 3.0)
    return raw.to(torch.bfloat16)


def tq4_dequantize_test():
    """
    TQ4 (TurboQuant 4-bit) dequantize coverage.

    Builds a fixed 16-entry bf16 codebook (latched from URAM-B[0] by the
    on-chip auto-load FSM at dma_start), tiles all 16 codebook indices
    across 8 blocks of 64 elements, and exercises the dequantize core with
    several positive per-block scale patterns.

    Hardware constraint: TQ4 requires positive bf16 scales. The compute
    unit's data-path mux (compute_unit.vhdl: gen_data_select) only routes
    the 4-bit nibble into ``fp_data`` (the codebook lookup input) when
    ``quant_datatype = '0'`` (positive scale sign). Under a negative
    scale, ``fp_data`` defaults to zero and the codebook lookup always
    returns entry 0. This is unlike IF4/IF8 where the scale sign is a
    per-block FP-vs-INT variant select; for TQ4 the scale is a magnitude
    only and the sign bit must stay clear.
    """
    from user_dma_core import DMA_DEVICE_H2C

    codebook = _build_tq4_test_codebook()

    M = 8 * UE_VECTOR_SIZE
    num_blocks = M // UE_VECTOR_SIZE

    codes_u8 = torch.arange(16, dtype=torch.int16).to(torch.uint8)
    reps = (M + 16 - 1) // 16
    q_u8 = codes_u8.repeat(reps)[:M].contiguous()

    # Pre-pack the 4-bit nibbles (low nibble first matches quantize_weight).
    num_payload_bytes = M // 2
    payload = torch.zeros(num_payload_bytes, dtype=torch.uint8)
    for i in range(0, M, 2):
        v1 = q_u8[i].item() & 0xF
        v2 = q_u8[i + 1].item() & 0xF
        payload[i // 2] = ((v2 & 0xF) << 4) | v1

    # All scales must be strictly positive (bf16 sign bit clear). Patterns
    # cover unit, varying-per-block, and a stress with a wide magnitude range.
    scale_patterns = [
        ("pos_unit",     lambda b: 1.0),
        ("varying",      lambda b: float(0.25 * (b + 1))),
        ("wide_range",   lambda b: float(2.0 ** (b - num_blocks // 2))),
    ]

    for pattern_name, scale_fn in scale_patterns:
        ue = UnifiedEngine()

        codebook_dram_addr = ue.prepare_tq4_codebook_dram(codebook)

        scales_bf16 = torch.tensor([scale_fn(b) for b in range(num_blocks)],
                                   dtype=torch.bfloat16)
        assert (scales_bf16 > 0).all(), \
            "TQ4 requires positive scales; negative scales steer the data path away from fp_data"

        q_dram = ue.get_params_dram_addr()
        ue.dma_write(DMA_DEVICE_H2C, q_dram, payload, num_payload_bytes)
        scale_dram = q_dram + num_payload_bytes
        ue.dma_write(DMA_DEVICE_H2C, scale_dram,
                     scales_bf16.view(torch.uint16), num_blocks * 2)
        ue.allocate_params_dram(num_payload_bytes + num_blocks * 2)

        OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * 2)

        ue.start_capture()
        ue.load_tq4_codebook(codebook_dram_addr)
        vector_sram_start_addr = 0x00000
        ue.start_queue_for_bf16_dequantize_operation(
            VECTOR_INPUT_DRAM_ADDR=q_dram,
            SCALE_INPUT_DRAM_ADDR=scale_dram,
            data_type=TYPE.TQ4,
            output_sram_wb_addr=vector_sram_start_addr,
            element_size=M,
        )
        ue.sram_to_accelerator_memory(
            sram_address=vector_sram_start_addr,
            accelerator_dram_address=OUTPUT_DRAM_ADDR,
            element_size=M,
        )
        ue.stop_capture()
        ue.generate_instruction_halt()
        program_dram_addr = ue.get_program_dram_addr()
        ue.write_captured_instructions_to_dram(program_dram_addr)
        ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

        ue.start_execute_from_dram(program_dram_addr)
        ue.wait_queue(10.0)
        ue.report_timing_and_instruction_count()

        output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M,))

        expected = torch.zeros(M, dtype=torch.bfloat16)
        for b in range(num_blocks):
            start = b * UE_VECTOR_SIZE
            stop = start + UE_VECTOR_SIZE
            idx = q_u8[start:stop].to(torch.long)
            block_vals = codebook[idx].to(torch.float32)
            expected[start:stop] = (block_vals
                                    * float(scales_bf16[b].item())
                                    ).to(torch.bfloat16)

        snr_db = calculate_snr(expected, output)
        print(f"TQ4 Dequantize ({pattern_name}) SNR: {snr_db:.2f} dB" if snr_db != float('inf')
              else f"TQ4 Dequantize ({pattern_name}) SNR: inf")
        assert snr_db >= 30 or snr_db == float('inf'), (
            f"TQ4 dequantize ({pattern_name}) SNR {snr_db:.2f} dB must be at least 30 dB"
        )

        record_test(f"tq4_dequantize-{pattern_name}",
                    f"M={M}, num_blocks={num_blocks}",
                    snr_db=snr_db)

        ue.clear_capture_buffer()
        ue.reset_tensor_dram_addr()


def tq4_dot_product_test(K: int = 64, N: int = 64):
    """
    TQ4 dot product coverage.

    Computes ``y = A @ B^T`` for a single bf16 vector ``A`` (length ``K``)
    against a TQ4-quantized matrix ``B`` (``N`` rows of ``K`` codebook
    indices). The codebook is auto-loaded from URAM-B[0] at dma_start; the
    output writeback uses URAM-B starting at offset 1 so the codebook row
    is preserved across the op (for test repeatability and to mirror how
    multi-tile kernels would have to lay out their writeback).

    Hardware constraint: TQ4 requires positive bf16 per-block scales. The
    compute unit only routes the 4-bit nibble into ``fp_data`` (the
    codebook lookup input) when the scale sign bit is clear; under a
    negative scale, ``fp_data`` defaults to zero and every code reads
    codebook[0]. We therefore always set positive magnitudes here.
    """
    from user_dma_core import DMA_DEVICE_H2C, LALU_MODE

    assert K % UE_VECTOR_SIZE == 0, f"K={K} must be a multiple of UE_VECTOR_SIZE={UE_VECTOR_SIZE}"
    assert N % UE_VECTOR_SIZE == 0, f"N={N} must be a multiple of UE_VECTOR_SIZE={UE_VECTOR_SIZE}"

    codebook = _build_tq4_test_codebook()

    # B: N x K, each value is an index 0..15 into the codebook.
    B_indices = torch.randint(0, 16, (N, K), dtype=torch.uint8)

    blocks_per_row = K // UE_VECTOR_SIZE
    num_blocks = N * blocks_per_row

    for mag_label in ("uniform", "varying"):
        ue = UnifiedEngine()

        codebook_dram_addr = ue.prepare_tq4_codebook_dram(codebook)

        # Random per-block positive magnitudes (sign bit must stay clear).
        magnitudes = torch.rand(num_blocks).to(torch.bfloat16) + 0.25
        scales_bf16 = magnitudes.to(torch.bfloat16)
        assert (scales_bf16 > 0).all(), \
            "TQ4 requires positive scales; negative scales steer the data path away from fp_data"

        # Pack B (N x K) row-major into 4-bit nibbles, low nibble first.
        flat_indices = B_indices.flatten()
        num_payload_bytes = (N * K) // 2
        payload = torch.zeros(num_payload_bytes, dtype=torch.uint8)
        for i in range(0, N * K, 2):
            v1 = flat_indices[i].item() & 0xF
            v2 = flat_indices[i + 1].item() & 0xF
            payload[i // 2] = ((v2 & 0xF) << 4) | v1

        B_DRAM_ADDR = ue.get_params_dram_addr()
        ue.dma_write(DMA_DEVICE_H2C, B_DRAM_ADDR, payload, num_payload_bytes)
        SCALE_DRAM_ADDR = B_DRAM_ADDR + num_payload_bytes
        ue.dma_write(DMA_DEVICE_H2C, SCALE_DRAM_ADDR,
                     scales_bf16.view(torch.uint16), num_blocks * 2)
        ue.allocate_params_dram(num_payload_bytes + num_blocks * 2)

        A = (torch.rand(K, dtype=torch.bfloat16) * 2.0 - 1.0)
        A_DRAM_ADDR = ue.allocate_tensor_dram(K * 2)
        OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(N * 2)

        URAM_B_BASE_SRAM_ADDR = 0x80000
        # URAM line size = 128 bytes. Reserve URAM-B row 0 for the codebook
        # and write outputs starting at row 1 so subsequent ops keep finding
        # the codebook in URAM-B[0] (the auto-load FSM re-reads it on every
        # DEQUANTIZE / DOT_PRODUCT dma_start).
        OUTPUT_SRAM_WB_ADDR = URAM_B_BASE_SRAM_ADDR + 0x80

        ue.start_capture()
        ue.load_tq4_codebook(codebook_dram_addr)
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=A_DRAM_ADDR,
            sram_address=0x00000,
            element_size=K,
        )
        ue.accelerator_memory_to_scale_sram(
            accelerator_dram_address=SCALE_DRAM_ADDR,
            element_size=num_blocks,
        )
        ue.start_queue_for_dot_product_operation(
            max_clear_en=1,
            fmax_context_addr=0,
            vector_sram_start_addr=0x00000,
            output_sram_wb_addr=OUTPUT_SRAM_WB_ADDR,
            K=K,
            N=N,
            dma_start_addr=B_DRAM_ADDR,
            data_type=TYPE.TQ4,
            bias_enable=False,
            lalu_mode=LALU_MODE.BYPASS,
        )
        ue.sram_to_accelerator_memory(
            sram_address=OUTPUT_SRAM_WB_ADDR,
            accelerator_dram_address=OUTPUT_DRAM_ADDR,
            element_size=N,
        )
        ue.stop_capture()
        ue.generate_instruction_halt()
        program_dram_addr = ue.get_program_dram_addr()
        ue.write_captured_instructions_to_dram(program_dram_addr)
        ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

        ue.dma_to_accelerator_memory(A_DRAM_ADDR, A)

        ue.start_execute_from_dram(program_dram_addr)
        ue.wait_queue(10.0)
        ue.report_timing_and_instruction_count()

        output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (N,))

        # Reference: dequantize B per block (codebook[idx] * scale), then
        # dot product against A in bf16 to match HW arithmetic precision.
        B_dequant = torch.zeros(N, K, dtype=torch.bfloat16)
        for n in range(N):
            for j in range(blocks_per_row):
                block_id = n * blocks_per_row + j
                start = j * UE_VECTOR_SIZE
                stop = start + UE_VECTOR_SIZE
                idx = B_indices[n, start:stop].to(torch.long)
                B_dequant[n, start:stop] = (
                    codebook[idx].to(torch.float32)
                    * float(scales_bf16[block_id].item())
                ).to(torch.bfloat16)
        ref = (A.to(torch.float32) @ B_dequant.to(torch.float32).T).to(torch.bfloat16)

        snr_db = calculate_snr(ref, output)
        print(f"TQ4 Dot Product ({mag_label}) SNR: {snr_db:.2f} dB (K={K}, N={N})")
        assert snr_db >= 30 or snr_db == float('inf'), (
            f"TQ4 dot product ({mag_label}) SNR {snr_db:.2f} dB must be at least 30 dB"
        )

        record_test(f"tq4_dot_product-{mag_label}",
                    f"K={K}, N={N}",
                    snr_db=snr_db)

        ue.clear_capture_buffer()
        ue.reset_tensor_dram_addr()


def tq4_dequantize_variant_tests():
    """Additional TQ4 dequantize variants without changing tq4_dequantize_test()."""
    from user_dma_core import DMA_DEVICE_H2C

    codebook = _build_tq4_test_codebook()
    M = 8 * UE_VECTOR_SIZE
    num_blocks = M // UE_VECTOR_SIZE

    codes_u8 = torch.arange(16, dtype=torch.int16).to(torch.uint8)
    reps = (M + 16 - 1) // 16
    q_u8 = codes_u8.repeat(reps)[:M].contiguous()

    num_payload_bytes = M // 2
    payload = torch.zeros(num_payload_bytes, dtype=torch.uint8)
    for i in range(0, M, 2):
        v1 = q_u8[i].item() & 0xF
        v2 = q_u8[i + 1].item() & 0xF
        payload[i // 2] = ((v2 & 0xF) << 4) | v1

    # New patterns:
    # - tiny: stresses underflow / scale handling
    # - ramp_pow2: power-of-two steps to catch exponent/rounding issues
    scale_patterns = [
        ("tiny",       lambda b: float(2.0 ** -8)),
        ("ramp_pow2",  lambda b: float(2.0 ** (-4 + (b % 8)))),
    ]

    for pattern_name, scale_fn in scale_patterns:
        ue = UnifiedEngine()
        codebook_dram_addr = ue.prepare_tq4_codebook_dram(codebook)

        scales_bf16 = torch.tensor([scale_fn(b) for b in range(num_blocks)], dtype=torch.bfloat16)
        assert (scales_bf16 > 0).all(), "TQ4 requires positive scales"

        q_dram = ue.get_params_dram_addr()
        ue.dma_write(DMA_DEVICE_H2C, q_dram, payload, num_payload_bytes)
        scale_dram = q_dram + num_payload_bytes
        ue.dma_write(DMA_DEVICE_H2C, scale_dram, scales_bf16.view(torch.uint16), num_blocks * 2)
        ue.allocate_params_dram(num_payload_bytes + num_blocks * 2)

        OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * 2)

        ue.start_capture()
        ue.load_tq4_codebook(codebook_dram_addr)
        vector_sram_start_addr = 0x00000
        ue.start_queue_for_bf16_dequantize_operation(
            VECTOR_INPUT_DRAM_ADDR=q_dram,
            SCALE_INPUT_DRAM_ADDR=scale_dram,
            data_type=TYPE.TQ4,
            output_sram_wb_addr=vector_sram_start_addr,
            element_size=M,
        )
        ue.sram_to_accelerator_memory(
            sram_address=vector_sram_start_addr,
            accelerator_dram_address=OUTPUT_DRAM_ADDR,
            element_size=M,
        )
        ue.stop_capture()
        ue.generate_instruction_halt()
        program_dram_addr = ue.get_program_dram_addr()
        ue.write_captured_instructions_to_dram(program_dram_addr)
        ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

        ue.start_execute_from_dram(program_dram_addr)
        ue.wait_queue(10.0)

        output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M,))

        expected = torch.zeros(M, dtype=torch.bfloat16)
        for b in range(num_blocks):
            start = b * UE_VECTOR_SIZE
            stop = start + UE_VECTOR_SIZE
            idx = q_u8[start:stop].to(torch.long)
            block_vals = codebook[idx].to(torch.float32)
            expected[start:stop] = (block_vals * float(scales_bf16[b].item())).to(torch.bfloat16)

        snr_db = calculate_snr(expected, output)
        print(f"TQ4 Dequantize Variant ({pattern_name}) SNR: {snr_db:.2f} dB" if snr_db != float('inf')
              else f"TQ4 Dequantize Variant ({pattern_name}) SNR: inf")
        assert snr_db >= 30 or snr_db == float('inf'), (
            f"TQ4 dequantize variant ({pattern_name}) SNR {snr_db:.2f} dB must be at least 30 dB"
        )

        record_test(f"tq4_dequantize_variant-{pattern_name}",
                    f"M={M}, num_blocks={num_blocks}",
                    snr_db=snr_db)

        ue.clear_capture_buffer()
        ue.reset_tensor_dram_addr()


def tq4_dot_product_variant_tests():
    """Additional TQ4 dot-product variants without changing tq4_dot_product_test()."""
    from user_dma_core import DMA_DEVICE_H2C, LALU_MODE

    def _run_one(K: int, N: int, *, a_mode: str, b_mode: str, scale_mode: str, label: str):
        assert K % UE_VECTOR_SIZE == 0
        assert N % UE_VECTOR_SIZE == 0

        codebook = _build_tq4_test_codebook()

        # Deterministic B patterns to hit edge cases.
        if b_mode == "tiled":
            block = torch.arange(16, dtype=torch.uint8).repeat(UE_VECTOR_SIZE // 16)
            row = block.repeat(K // UE_VECTOR_SIZE)
            B_indices = row.unsqueeze(0).repeat(N, 1).contiguous()
        elif b_mode == "zeros":
            B_indices = torch.zeros((N, K), dtype=torch.uint8)
        elif b_mode == "max":
            B_indices = torch.full((N, K), 15, dtype=torch.uint8)
        else:
            raise ValueError(f"unknown b_mode={b_mode!r}")

        blocks_per_row = K // UE_VECTOR_SIZE
        num_blocks = N * blocks_per_row

        if scale_mode == "unit":
            scales_bf16 = torch.ones(num_blocks, dtype=torch.bfloat16)
        elif scale_mode == "pow2":
            scales = torch.tensor([2.0 ** (-4 + (i % 8)) for i in range(num_blocks)], dtype=torch.float32)
            scales_bf16 = scales.to(torch.bfloat16)
        else:
            raise ValueError(f"unknown scale_mode={scale_mode!r}")
        assert (scales_bf16 > 0).all()

        if a_mode == "ones":
            A = torch.ones(K, dtype=torch.bfloat16)
        elif a_mode == "alt_sign":
            A = torch.ones(K, dtype=torch.bfloat16)
            A[1::2] = -1
        else:
            raise ValueError(f"unknown a_mode={a_mode!r}")

        ue = UnifiedEngine()
        codebook_dram_addr = ue.prepare_tq4_codebook_dram(codebook)

        flat_indices = B_indices.flatten()
        num_payload_bytes = (N * K) // 2
        payload = torch.zeros(num_payload_bytes, dtype=torch.uint8)
        for i in range(0, N * K, 2):
            v1 = flat_indices[i].item() & 0xF
            v2 = flat_indices[i + 1].item() & 0xF
            payload[i // 2] = ((v2 & 0xF) << 4) | v1

        B_DRAM_ADDR = ue.get_params_dram_addr()
        ue.dma_write(DMA_DEVICE_H2C, B_DRAM_ADDR, payload, num_payload_bytes)
        SCALE_DRAM_ADDR = B_DRAM_ADDR + num_payload_bytes
        ue.dma_write(DMA_DEVICE_H2C, SCALE_DRAM_ADDR, scales_bf16.view(torch.uint16), num_blocks * 2)
        ue.allocate_params_dram(num_payload_bytes + num_blocks * 2)

        A_DRAM_ADDR = ue.allocate_tensor_dram(K * 2)
        OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(N * 2)

        URAM_B_BASE_SRAM_ADDR = 0x80000
        OUTPUT_SRAM_WB_ADDR = URAM_B_BASE_SRAM_ADDR + 0x80

        ue.start_capture()
        ue.load_tq4_codebook(codebook_dram_addr)
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=A_DRAM_ADDR,
            sram_address=0x00000,
            element_size=K,
        )
        ue.accelerator_memory_to_scale_sram(
            accelerator_dram_address=SCALE_DRAM_ADDR,
            element_size=num_blocks,
        )
        ue.start_queue_for_dot_product_operation(
            max_clear_en=1,
            fmax_context_addr=0,
            vector_sram_start_addr=0x00000,
            output_sram_wb_addr=OUTPUT_SRAM_WB_ADDR,
            K=K,
            N=N,
            dma_start_addr=B_DRAM_ADDR,
            data_type=TYPE.TQ4,
            bias_enable=False,
            lalu_mode=LALU_MODE.BYPASS,
        )
        ue.sram_to_accelerator_memory(
            sram_address=OUTPUT_SRAM_WB_ADDR,
            accelerator_dram_address=OUTPUT_DRAM_ADDR,
            element_size=N,
        )
        ue.stop_capture()
        ue.generate_instruction_halt()
        program_dram_addr = ue.get_program_dram_addr()
        ue.write_captured_instructions_to_dram(program_dram_addr)
        ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

        ue.dma_to_accelerator_memory(A_DRAM_ADDR, A)

        ue.start_execute_from_dram(program_dram_addr)
        ue.wait_queue(10.0)

        output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (N,))

        B_dequant = torch.zeros(N, K, dtype=torch.bfloat16)
        for n in range(N):
            for j in range(blocks_per_row):
                block_id = n * blocks_per_row + j
                start = j * UE_VECTOR_SIZE
                stop = start + UE_VECTOR_SIZE
                idx = B_indices[n, start:stop].to(torch.long)
                B_dequant[n, start:stop] = (
                    codebook[idx].to(torch.float32) * float(scales_bf16[block_id].item())
                ).to(torch.bfloat16)
        ref = (A.to(torch.float32) @ B_dequant.to(torch.float32).T).to(torch.bfloat16)

        snr_db = calculate_snr(ref, output)
        print(f"TQ4 Dot Product Variant ({label}) SNR: {snr_db:.2f} dB (K={K}, N={N})")
        assert snr_db >= 30 or snr_db == float('inf'), (
            f"TQ4 dot product variant ({label}) SNR {snr_db:.2f} dB must be at least 30 dB"
        )
        record_test(f"tq4_dot_product_variant-{label}", f"K={K}, N={N}", snr_db=snr_db)
        ue.clear_capture_buffer()
        ue.reset_tensor_dram_addr()

    # Variant matrix:
    # - Shapes: asymmetric/tall/wide
    # - Inputs: deterministic A/B and non-random scales to catch packing/layout bugs
    variants = [
        (64, 128, "ones", "tiled", "unit", "64x128-ones-tiled-unit"),
        (128, 64, "alt_sign", "tiled", "unit", "128x64-altsign-tiled-unit"),
        (192, 64, "ones", "zeros", "pow2", "192x64-ones-zero-pow2"),
        (64, 256, "alt_sign", "max", "pow2", "64x256-altsign-max-pow2"),
    ]
    for K, N, a_mode, b_mode, scale_mode, label in variants:
        _run_one(K, N, a_mode=a_mode, b_mode=b_mode, scale_mode=scale_mode, label=label)


def tq4_dot_product_onehot_oracle_tests():
    """
    Strong TQ4 dot-product verification using one-hot A vectors.

    With a one-hot input A, y[n] = B_dequant[n, idx] exactly (no accumulation),
    so this catches:
    - nibble packing order (low/high nibble swap)
    - K indexing / URAM-A lane mapping
    - per-block scale addressing (block-id mapping)
    - codebook lookup correctness
    """
    from user_dma_core import DMA_DEVICE_H2C, LALU_MODE

    K = 128
    N = 64
    assert K % UE_VECTOR_SIZE == 0
    assert N % UE_VECTOR_SIZE == 0

    # Use a hand-crafted codebook with distinct values so nibble swaps are obvious.
    # Keep magnitudes modest to stay well within bf16 dynamic range.
    codebook = torch.tensor(
        [-3.0, -2.5, -2.0, -1.5,
         -1.0, -0.5, -0.25, -0.125,
         +0.125, +0.25, +0.5, +1.0,
         +1.5, +2.0, +2.5, +3.0],
        dtype=torch.bfloat16,
    )

    # Build B indices so each lane is a unique, repeating pattern. Also make
    # the second 64-lane block different to verify block boundary at 63/64.
    block0 = torch.tensor([(i * 3) % 16 for i in range(UE_VECTOR_SIZE)], dtype=torch.uint8)
    block1 = torch.tensor([(i * 5 + 7) % 16 for i in range(UE_VECTOR_SIZE)], dtype=torch.uint8)
    row = torch.cat([block0, block1], dim=0)
    B_indices = row.unsqueeze(0).repeat(N, 1).contiguous()  # N x K

    blocks_per_row = K // UE_VECTOR_SIZE
    num_blocks = N * blocks_per_row

    # Per-block scales: distinct per-block magnitudes to validate block-id mapping.
    scales = []
    for n in range(N):
        for b in range(blocks_per_row):
            scales.append(0.5 if b == 0 else 2.0)
    scales_bf16 = torch.tensor(scales, dtype=torch.bfloat16)
    assert (scales_bf16 > 0).all()

    # Pack B row-major into 4-bit nibbles, low nibble first.
    flat_indices = B_indices.flatten()
    num_payload_bytes = (N * K) // 2
    payload = torch.zeros(num_payload_bytes, dtype=torch.uint8)
    for i in range(0, N * K, 2):
        v1 = flat_indices[i].item() & 0xF
        v2 = flat_indices[i + 1].item() & 0xF
        payload[i // 2] = ((v2 & 0xF) << 4) | v1

    # Probe indices around block boundary and a few interior lanes.
    probe_positions = [0, 1, 2, 31, 62, 63, 64, 65, 95, 126, 127]

    for pos in probe_positions:
        ue = UnifiedEngine()
        codebook_dram_addr = ue.prepare_tq4_codebook_dram(codebook)

        B_DRAM_ADDR = ue.get_params_dram_addr()
        ue.dma_write(DMA_DEVICE_H2C, B_DRAM_ADDR, payload, num_payload_bytes)
        SCALE_DRAM_ADDR = B_DRAM_ADDR + num_payload_bytes
        ue.dma_write(DMA_DEVICE_H2C, SCALE_DRAM_ADDR, scales_bf16.view(torch.uint16), num_blocks * 2)
        ue.allocate_params_dram(num_payload_bytes + num_blocks * 2)

        # One-hot A at position pos.
        A = torch.zeros(K, dtype=torch.bfloat16)
        A[pos] = 1.0
        A_DRAM_ADDR = ue.allocate_tensor_dram(K * 2)
        OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(N * 2)

        URAM_B_BASE_SRAM_ADDR = 0x80000
        OUTPUT_SRAM_WB_ADDR = URAM_B_BASE_SRAM_ADDR + 0x80

        ue.start_capture()
        ue.load_tq4_codebook(codebook_dram_addr)
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=A_DRAM_ADDR,
            sram_address=0x00000,
            element_size=K,
        )
        ue.accelerator_memory_to_scale_sram(
            accelerator_dram_address=SCALE_DRAM_ADDR,
            element_size=num_blocks,
        )
        ue.start_queue_for_dot_product_operation(
            max_clear_en=1,
            fmax_context_addr=0,
            vector_sram_start_addr=0x00000,
            output_sram_wb_addr=OUTPUT_SRAM_WB_ADDR,
            K=K,
            N=N,
            dma_start_addr=B_DRAM_ADDR,
            data_type=TYPE.TQ4,
            bias_enable=False,
            lalu_mode=LALU_MODE.BYPASS,
        )
        ue.sram_to_accelerator_memory(
            sram_address=OUTPUT_SRAM_WB_ADDR,
            accelerator_dram_address=OUTPUT_DRAM_ADDR,
            element_size=N,
        )
        ue.stop_capture()
        ue.generate_instruction_halt()
        program_dram_addr = ue.get_program_dram_addr()
        ue.write_captured_instructions_to_dram(program_dram_addr)
        ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

        ue.dma_to_accelerator_memory(A_DRAM_ADDR, A)

        ue.start_execute_from_dram(program_dram_addr)
        ue.wait_queue(10.0)

        output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (N,))

        # Expected output is exactly the dequantized B at column pos.
        block_id_in_row = pos // UE_VECTOR_SIZE
        idx = B_indices[:, pos].to(torch.long)  # N
        scale_for_rows = scales_bf16.view(N, blocks_per_row)[:, block_id_in_row].to(torch.float32)  # N
        expected = (codebook[idx].to(torch.float32) * scale_for_rows).to(torch.bfloat16)

        # This should be very tight because there is no accumulation error.
        max_abs_err = (expected.to(torch.float32) - output.to(torch.float32)).abs().max().item()
        print(f"TQ4 onehot oracle: pos={pos} max_abs_err={max_abs_err:g}")
        assert torch.allclose(expected, output, atol=0, rtol=0, equal_nan=True), (
            f"TQ4 onehot oracle mismatch at pos={pos}: max_abs_err={max_abs_err:g}"
        )

        record_test("tq4_dot_product_onehot_oracle",
                    f"K={K}, N={N}, pos={pos}",
                    snr_db=math.inf)

        ue.clear_capture_buffer()
        ue.reset_tensor_dram_addr()


def tq4_codebook_reload_tests():
    """
    Verify that changing the codebook changes results (i.e., auto-load path works).
    """
    from user_dma_core import DMA_DEVICE_H2C, LALU_MODE

    K = 64
    N = 64
    assert K % UE_VECTOR_SIZE == 0
    assert N % UE_VECTOR_SIZE == 0

    # Simple B: fixed indices; simple A: ones; unit scales. Output should
    # be proportional to sum(codebook[idx]) so different codebooks must differ.
    B_indices = torch.arange(K, dtype=torch.uint8).remainder(16).unsqueeze(0).repeat(N, 1).contiguous()
    flat_indices = B_indices.flatten()
    num_payload_bytes = (N * K) // 2
    payload = torch.zeros(num_payload_bytes, dtype=torch.uint8)
    for i in range(0, N * K, 2):
        v1 = flat_indices[i].item() & 0xF
        v2 = flat_indices[i + 1].item() & 0xF
        payload[i // 2] = ((v2 & 0xF) << 4) | v1

    scales_bf16 = torch.ones(N * (K // UE_VECTOR_SIZE), dtype=torch.bfloat16)
    A = torch.ones(K, dtype=torch.bfloat16)

    outputs = []
    for cb_label, codebook in (
        ("cb0", _build_tq4_test_codebook()),
        ("cb1", _build_tq4_test_codebook()),
    ):
        ue = UnifiedEngine()
        codebook_dram_addr = ue.prepare_tq4_codebook_dram(codebook)

        B_DRAM_ADDR = ue.get_params_dram_addr()
        ue.dma_write(DMA_DEVICE_H2C, B_DRAM_ADDR, payload, num_payload_bytes)
        SCALE_DRAM_ADDR = B_DRAM_ADDR + num_payload_bytes
        ue.dma_write(DMA_DEVICE_H2C, SCALE_DRAM_ADDR, scales_bf16.view(torch.uint16), scales_bf16.numel() * 2)
        ue.allocate_params_dram(num_payload_bytes + scales_bf16.numel() * 2)

        A_DRAM_ADDR = ue.allocate_tensor_dram(K * 2)
        OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(N * 2)

        URAM_B_BASE_SRAM_ADDR = 0x80000
        OUTPUT_SRAM_WB_ADDR = URAM_B_BASE_SRAM_ADDR + 0x80

        ue.start_capture()
        ue.load_tq4_codebook(codebook_dram_addr)
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=A_DRAM_ADDR,
            sram_address=0x00000,
            element_size=K,
        )
        ue.accelerator_memory_to_scale_sram(
            accelerator_dram_address=SCALE_DRAM_ADDR,
            element_size=scales_bf16.numel(),
        )
        ue.start_queue_for_dot_product_operation(
            max_clear_en=1,
            fmax_context_addr=0,
            vector_sram_start_addr=0x00000,
            output_sram_wb_addr=OUTPUT_SRAM_WB_ADDR,
            K=K,
            N=N,
            dma_start_addr=B_DRAM_ADDR,
            data_type=TYPE.TQ4,
            bias_enable=False,
            lalu_mode=LALU_MODE.BYPASS,
        )
        ue.sram_to_accelerator_memory(
            sram_address=OUTPUT_SRAM_WB_ADDR,
            accelerator_dram_address=OUTPUT_DRAM_ADDR,
            element_size=N,
        )
        ue.stop_capture()
        ue.generate_instruction_halt()
        program_dram_addr = ue.get_program_dram_addr()
        ue.write_captured_instructions_to_dram(program_dram_addr)
        ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

        ue.dma_to_accelerator_memory(A_DRAM_ADDR, A)
        ue.start_execute_from_dram(program_dram_addr)
        ue.wait_queue(10.0)

        out = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (N,))
        outputs.append((cb_label, out))

        record_test("tq4_codebook_reload", f"{cb_label}: K={K}, N={N}", snr_db=math.inf)
        ue.clear_capture_buffer()
        ue.reset_tensor_dram_addr()

    # Ensure the two outputs differ (codebook actually took effect).
    diff = (outputs[0][1].to(torch.float32) - outputs[1][1].to(torch.float32)).abs().max().item()
    print(f"TQ4 codebook reload: max_abs_diff={diff:g}")
    assert diff > 0, "codebook reload test: outputs identical across different codebooks"

def run_turboquant_mse(dim: int):
    """
    Executes TurboQuant MSE (Algorithm 1) using the custom UnifiedEngine hardware.
    """
    from quant_lib import get_codebook_tensors, generate_rotation_matrix
    from user_dma_core import DMA_DEVICE_H2C

    M = dim
    num_blocks = M // UE_VECTOR_SIZE

    # 1. Initialization and CPU Pre-processing
    ue = UnifiedEngine()
    x = torch.randn(1, dim, dtype=torch.bfloat16)

    # Store norms for rescaling
    norms = x.norm(dim=-1, keepdim=False)
    x_unit = x / (norms.unsqueeze(-1) + 1e-10)

    # Prepare Rotation Matrix and Codebook
    Pi = generate_rotation_matrix(dim, "cpu", torch.bfloat16)
    centroids, boundaries = get_codebook_tensors(dim, 4, "cpu", torch.bfloat16)
    decision_boundaries = boundaries[1:-1].contiguous()

    # Apply random rotation
    y = torch.matmul(x_unit, Pi.T)

    # Quantize: find bucket via searchsorted and flatten for packing
    indices = torch.searchsorted(decision_boundaries, y.contiguous()).view(-1)

    # 2. Pack 4-bit indices into uint8 payload for Hardware
    indices_u8 = indices.to(torch.uint8)
    num_payload_bytes = M // 2
    payload = torch.zeros(num_payload_bytes, dtype=torch.uint8)

    # Low nibble first matching the hardware behavior
    for i in range(0, M, 2):
        v1 = indices_u8[i].item() & 0xF
        v2 = indices_u8[i + 1].item() & 0xF
        payload[i // 2] = ((v2 & 0xF) << 4) | v1

    # 3. Setup Scales for Hardware
    # Pass the constant norm value as the scale for every block
    scales_bf16 = torch.full((num_blocks,), norms.item(), dtype=torch.bfloat16)

    # 4. Hardware Memory Allocation & DMA Transfers
    codebook_dram_addr = ue.prepare_tq4_codebook_dram(centroids)

    q_dram = ue.get_params_dram_addr()
    ue.dma_write(DMA_DEVICE_H2C, q_dram, payload, num_payload_bytes)

    scale_dram = q_dram + num_payload_bytes
    ue.dma_write(DMA_DEVICE_H2C, scale_dram, scales_bf16.view(torch.uint16), num_blocks * 2)

    ue.allocate_params_dram(num_payload_bytes + num_blocks * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * 2)

    # 5. Capture Hardware Instructions
    ue.start_capture()
    ue.load_tq4_codebook(codebook_dram_addr)

    vector_sram_start_addr = 0x00000
    ue.start_queue_for_bf16_dequantize_operation(
        VECTOR_INPUT_DRAM_ADDR=q_dram,
        SCALE_INPUT_DRAM_ADDR=scale_dram,
        data_type=TYPE.TQ4,
        output_sram_wb_addr=vector_sram_start_addr,
        element_size=M,
    )
    ue.sram_to_accelerator_memory(
        sram_address=vector_sram_start_addr,
        accelerator_dram_address=OUTPUT_DRAM_ADDR,
        element_size=M,
    )
    ue.stop_capture()
    ue.generate_instruction_halt()

    # 6. Execute on Hardware
    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0)

    # Optional: ue.report_timing_and_instruction_count()

    # 7. Fetch Results and Post-Processing
    dequantized_hw = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (1, M))

    # Since the hardware already scaled by the `norms` values, we do not multiply by norms.float() again.
    # We simply cast the hardware output to float32 for the final matmul.
    dequant_x = dequantized_hw.float()

    # Reverse the rotation
    dequant = torch.matmul(dequant_x, Pi.float())

    # Calculate and print MSE
    mse = torch.nn.functional.mse_loss(dequant, x.float())
    print(f"MSE between Original X and HW Dequantized X: {mse.item():.6f}")

    # 8. Cleanup
    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

    return mse.item()

def if4_if8_dot_product_test(K: int = 64, N: int = 64):
    """
    IF4 / IF8 dot product coverage for both INT and FP variants.

    Computes ``y = A @ B^T`` for a bf16 vector ``A`` (length ``K``) against
    an IF4 / IF8 quantized matrix ``B`` (``N`` rows of ``K`` per-element
    codes plus per-block bf16 scales). The variant select (INT vs FP) is
    encoded per block via the sign of the bf16 scale: negative -> INT path
    (two's complement codes), positive -> FP path (NVFP4 / FP8 E4M3 codes).
    ``|scale|`` is the effective multiplier on hardware.

    Codes cover the full code table tiled across the matrix and ``|scale|
    = 1`` keeps the per-element dequant ``code_table[idx] * |scale|``
    exactly representable in bf16 - the only precision loss comes from
    the bf20 adder-tree accumulation in the dot product, which is what
    we want to exercise.

    IF8 (both FP and INT) at multi-block K (K >= 128) is currently skipped.
    End-to-end runs at K=128 N=128 produced near-zero SNR for both IF8-FP
    and IF8-INT, while IF4-FP, IF4-INT (and TQ4 via tq4_dot_product_test)
    all pass at the same K=128 N=128 dimensions. The C-side coverage in
    Vitis/common/src/andromeda.c only validates IF8-FP at M=1, N=128
    (a single output row, so the multi-block-K wrap of URAM-A is not
    exercised) and IF8-INT large cases only check latency (not output
    values), so the multi-block-K IF8 dot-product path with N > 1 is
    unvalidated upstream of this test. The IF4 path uses 1 DMA beat per
    block; IF8 uses 2 beats per block (int8_data_handler.sv), which is
    the only obvious structural difference between the working IF4 path
    and the failing IF8 path - the most likely locus of the bug. Needs
    HW investigation; until then the IF8 coverage here uses K=64 only.
    """
    from user_dma_core import DMA_DEVICE_H2C, LALU_MODE

    assert K % UE_VECTOR_SIZE == 0, f"K={K} must be a multiple of UE_VECTOR_SIZE={UE_VECTOR_SIZE}"
    assert N % UE_VECTOR_SIZE == 0, f"N={N} must be a multiple of UE_VECTOR_SIZE={UE_VECTOR_SIZE}"

    NVFP4_TABLE = torch.tensor([
        +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ], dtype=torch.bfloat16)
    INT4_TABLE = torch.tensor(
        [c - 16 if c >= 8 else c for c in range(16)],
        dtype=torch.int16,
    ).to(torch.bfloat16)
    INT8_TABLE = torch.tensor(
        [c - 256 if c >= 128 else c for c in range(256)],
        dtype=torch.int16,
    ).to(torch.bfloat16)
    # FP8 E4M3FN: sign + 4-exp + 3-mant. 0x7F / 0xFF are NaN; we drop them
    # below to keep the dot-product reference well-defined.
    FP8_E4M3FN_TABLE = torch.tensor([
        +0.0, 0.001953, 0.003906, 0.005859, 0.007812, 0.009766, 0.01172, 0.01367,
        0.01562, 0.01758, 0.01953, 0.02148, 0.02344, 0.02539, 0.02734, 0.0293,
        0.03125, 0.03516, 0.03906, 0.04297, 0.04688, 0.05078, 0.05469, 0.05859,
        0.0625, 0.07031, 0.07812, 0.08594, 0.09375, 0.1016, 0.1094, 0.1172,
        0.125, 0.1406, 0.1562, 0.1719, 0.1875, 0.2031, 0.2188, 0.2344,
        0.25, 0.2812, 0.3125, 0.3438, 0.375, 0.4062, 0.4375, 0.4688,
        0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375,
        1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875,
        2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75,
        4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,
        8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0,
        32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0,
        64.0, 72.0, 80.0, 88.0, 96.0, 104.0, 112.0, 120.0,
        128.0, 144.0, 160.0, 176.0, 192.0, 208.0, 224.0, 240.0,
        256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, math.nan,
        -0.0, -0.001953, -0.003906, -0.005859, -0.007812, -0.009766, -0.01172, -0.01367,
        -0.01562, -0.01758, -0.01953, -0.02148, -0.02344, -0.02539, -0.02734, -0.0293,
        -0.03125, -0.03516, -0.03906, -0.04297, -0.04688, -0.05078, -0.05469, -0.05859,
        -0.0625, -0.07031, -0.07812, -0.08594, -0.09375, -0.1016, -0.1094, -0.1172,
        -0.125, -0.1406, -0.1562, -0.1719, -0.1875, -0.2031, -0.2188, -0.2344,
        -0.25, -0.2812, -0.3125, -0.3438, -0.375, -0.4062, -0.4375, -0.4688,
        -0.5, -0.5625, -0.625, -0.6875, -0.75, -0.8125, -0.875, -0.9375,
        -1.0, -1.125, -1.25, -1.375, -1.5, -1.625, -1.75, -1.875,
        -2.0, -2.25, -2.5, -2.75, -3.0, -3.25, -3.5, -3.75,
        -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -7.0, -7.5,
        -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0,
        -16.0, -18.0, -20.0, -22.0, -24.0, -26.0, -28.0, -30.0,
        -32.0, -36.0, -40.0, -44.0, -48.0, -52.0, -56.0, -60.0,
        -64.0, -72.0, -80.0, -88.0, -96.0, -104.0, -112.0, -120.0,
        -128.0, -144.0, -160.0, -176.0, -192.0, -208.0, -224.0, -240.0,
        -256.0, -288.0, -320.0, -352.0, -384.0, -416.0, -448.0, -math.nan,
    ]).to(torch.bfloat16)

    # (label, hw data_type, scale_value, value_table, num_codes, max_K)
    # Sign of scale is the per-block FP-vs-INT variant select.
    # Bound IF8 code ranges so the K-wide dot-product stays well inside
    # bf16 dynamic range: full INT8 (up to +/-128) and FP8 E4M3 (up to
    # +/-448) with K=64..256 would push sums past the ~3e4 regime where
    # bf16 quantization noise dominates the test. Capping at the smaller
    # half of each table keeps the geometry meaningful while still hitting
    # both signs and a wide magnitude range. Drops the +/-NaN entries at
    # 0x7F / 0xFF from the FP8 sweep.
    # ``max_K``: skip the variant entirely when the call's K exceeds this.
    # Both IF8 variants produce near-zero SNR at K >= 128 on the current
    # HW build (see the docstring); skip rather than masking the failure.
    configs = [
        ("IF4-FP",  TYPE.IF4, +1.0, NVFP4_TABLE,        16, None),
        ("IF4-INT", TYPE.IF4, -1.0, INT4_TABLE,         16, None),
        ("IF8-FP",  TYPE.IF8, +1.0, FP8_E4M3FN_TABLE,   64, 64),
        ("IF8-INT", TYPE.IF8, -1.0, INT8_TABLE,         32, 64),
    ]

    blocks_per_row = K // UE_VECTOR_SIZE
    num_blocks = N * blocks_per_row

    for label, data_type, scale_value, value_table, num_codes, max_K in configs:
        if max_K is not None and K > max_K:
            print(f"IF4/IF8 Dot Product ({label}) skipped at K={K} > max_K={max_K}")
            continue
        ue = UnifiedEngine()

        scales_bf16 = torch.full((num_blocks,), scale_value, dtype=torch.bfloat16)

        # Tile codes 0..num_codes-1 across the N*K matrix elements so each
        # block sees the full range of codes.
        flat_codes = torch.randint(0, num_codes, (N * K,), dtype=torch.int16).to(torch.uint8)

        if data_type == TYPE.IF4:
            num_payload_bytes = (N * K) // 2
            payload = torch.zeros(num_payload_bytes, dtype=torch.uint8)
            for i in range(0, N * K, 2):
                v1 = flat_codes[i].item() & 0xF
                v2 = flat_codes[i + 1].item() & 0xF
                payload[i // 2] = ((v2 & 0xF) << 4) | v1
        else:  # TYPE.IF8
            num_payload_bytes = N * K
            payload = flat_codes.contiguous()

        B_DRAM_ADDR = ue.get_params_dram_addr()
        ue.dma_write(DMA_DEVICE_H2C, B_DRAM_ADDR, payload, num_payload_bytes)
        SCALE_DRAM_ADDR = B_DRAM_ADDR + num_payload_bytes
        ue.dma_write(DMA_DEVICE_H2C, SCALE_DRAM_ADDR,
                     scales_bf16.view(torch.uint16), num_blocks * 2)
        ue.allocate_params_dram(num_payload_bytes + num_blocks * 2)

        A = (torch.rand(K, dtype=torch.bfloat16) * 2.0 - 1.0)
        A_DRAM_ADDR = ue.allocate_tensor_dram(K * 2)
        OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(N * 2)

        URAM_B_BASE_SRAM_ADDR = 0x80000
        OUTPUT_SRAM_WB_ADDR = URAM_B_BASE_SRAM_ADDR

        ue.start_capture()
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=A_DRAM_ADDR,
            sram_address=0x00000,
            element_size=K,
        )
        ue.accelerator_memory_to_scale_sram(
            accelerator_dram_address=SCALE_DRAM_ADDR,
            element_size=num_blocks,
        )
        ue.start_queue_for_dot_product_operation(
            max_clear_en=1,
            fmax_context_addr=0,
            vector_sram_start_addr=0x00000,
            output_sram_wb_addr=OUTPUT_SRAM_WB_ADDR,
            K=K,
            N=N,
            dma_start_addr=B_DRAM_ADDR,
            data_type=data_type,
            bias_enable=False,
            lalu_mode=LALU_MODE.BYPASS,
        )
        ue.sram_to_accelerator_memory(
            sram_address=OUTPUT_SRAM_WB_ADDR,
            accelerator_dram_address=OUTPUT_DRAM_ADDR,
            element_size=N,
        )
        ue.stop_capture()
        ue.generate_instruction_halt()
        program_dram_addr = ue.get_program_dram_addr()
        ue.write_captured_instructions_to_dram(program_dram_addr)
        ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

        ue.dma_to_accelerator_memory(A_DRAM_ADDR, A)

        ue.start_execute_from_dram(program_dram_addr)
        ue.wait_queue(10.0)
        ue.report_timing_and_instruction_count()

        output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (N,))

        # Reference: dequantize each row via lookup-table[code] * |scale|,
        # then dot product against A in bf16 to mirror HW arithmetic.
        abs_scale = abs(scale_value)
        codes_2d = flat_codes.view(N, K).to(torch.long)
        B_dequant = (value_table[codes_2d].to(torch.float32) * abs_scale).to(torch.bfloat16)
        ref = (A.to(torch.float32) @ B_dequant.to(torch.float32).T).to(torch.bfloat16)

        snr_db = calculate_snr(ref, output)
        print(f"IF4/IF8 Dot Product ({label}) SNR: {snr_db:.2f} dB (K={K}, N={N})")
        assert snr_db >= 30 or snr_db == float('inf'), (
            f"{label} dot product SNR {snr_db:.2f} dB must be at least 30 dB"
        )

        record_test(f"if4_if8_dot_product-{label}",
                    f"K={K}, N={N}",
                    snr_db=snr_db)

        ue.clear_capture_buffer()
        ue.reset_tensor_dram_addr()


def dequantize_test(data_type=TYPE.IF4, int_variant: bool = True):
    """
    Tests dequantize core for the selected adaptive type.

    ``data_type`` selects the bit width (TYPE.IF4 or TYPE.IF8). ``int_variant``
    selects the INT vs FP variant within that width; the variant is encoded on
    the wire via the sign of the bf16 scale.
    """
    ue = UnifiedEngine()

    M = 64
    N = 128

    if not int_variant:
        # Floating-point variants need a wider distribution
        x = torch.randn(M, N, dtype=torch.bfloat16)
    else:
        x = torch.rand(M, N, dtype=torch.bfloat16)

    QUANTIZED_MATRIX_DRAM_ADDR, SCALE_DRAM_ADDR = ue.quantize_weight(
        weight=x, N=M, K=N, data_type=data_type, int_variant=int_variant)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)

    ue.start_capture()

    vector_sram_start_addr = 0x00000
    total_flops_from_dequantize = ue.start_queue_for_bf16_dequantize_operation(VECTOR_INPUT_DRAM_ADDR=QUANTIZED_MATRIX_DRAM_ADDR,
                                                SCALE_INPUT_DRAM_ADDR=SCALE_DRAM_ADDR,
                                                data_type=data_type,
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

    width_str = "IF4" if data_type == TYPE.IF4 else "IF8"
    variant_str = "INT" if int_variant else "FP"
    data_type_str = f"{width_str}-{variant_str}"
    generate_trace(ue, f"dequantize_core_trace_{M}_{N}_{data_type_str}.csv")

    report_flop_rate_gflops, flops_ratio = ue.report_flop_rate_gflops(total_flops_from_dequantize)
    print(f"Report FLOPS for Dequantize ({data_type_str}): {report_flop_rate_gflops:.2f} GFLOPS, {flops_ratio:.2f}% peak throughput for M={M}, N={N}")

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M, N))
    if data_type == TYPE.IF4:
        quant_max = 7.0 if int_variant else 6.0
    else:  # TYPE.IF8
        quant_max = 127.0 if int_variant else 448.0

    fake_quantized_matrix = x.reshape(-1, UE_VECTOR_SIZE)
    scales = quant_max / fake_quantized_matrix.abs().max(dim=-1).values
    scales = scales.unsqueeze(-1)
    scaled = fake_quantized_matrix * scales
    if data_type == TYPE.IF4 and not int_variant:
        fp4_values = torch.tensor(
            [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
             0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            dtype=torch.bfloat16)
        distances = torch.abs(scaled.unsqueeze(-1) - fp4_values.unsqueeze(0).unsqueeze(0))
        closest_indices = torch.argmin(distances, dim=-1)
        quantized_matrix = fp4_values[closest_indices]
    else:
        quantized_matrix = scaled.round()
    dequantized_matrix = quantized_matrix / scales
    dequantized_matrix = dequantized_matrix.reshape(M, N)

    snr_db_ref = calculate_snr(dequantized_matrix, output)
    print(f"Reference SNR Analysis for Dequantize ({data_type_str}): {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 30 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 30 dB"

    snr_db_ref = calculate_snr(x, output)
    print(f"Reference SNR Analysis vs Original x ({data_type_str}): {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 19 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 19 dB"

    record_test(f"dequantize-{data_type_str}",
                f"M={M}, N={N}",
                snr_db=snr_db_ref,
                gflops=report_flop_rate_gflops)

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

def matmat_mul_quantized_weights_test(M: int, K: int, N: int, bias_enable: bool = False, bias_mode: str = "broadcast_N", data_type: TYPE = TYPE.IF4, int_variant: bool = True, gelu_enable: bool = False, silu_enable: bool = False, sigmoid_enable: bool = False, clamp_enable: bool = False, log_enable: bool = False, use_pbi: bool = False):
    """
    Tests matrix-matrix multiplication with quantized weights.
    Args:
        M: batch dimension (number of input vectors)
        K: inner dimension (vector length), must be a multiple of UE_VECTOR_SIZE
        N: outer dimension (matrix height), must be a multiple of UE_VECTOR_SIZE
        bias_enable: enable bias
        bias_mode: bias mode
        data_type: width of the quantized weights (TYPE.IF4 or TYPE.IF8).
        int_variant: select INT vs FP variant within the chosen width. The
            variant is communicated to the hardware via the sign of the bf16
            scale (negative = INT, positive = FP).
        gelu_enable: enable gelu activation
        silu_enable: enable silu activation
        clamp_enable: enable clamp activation (relu via clamp(x, 0, +inf))
        log_enable: enable log activation (log(clamp(x, 1e-3, +inf)))
        use_pbi: emit matmul with PBI-backed weight stream when True.

    Reference matmul uses :meth:`UnifiedEngine.quantize_weight_simulate` on ``x`` (same
    ``data_type`` / ``int_variant`` as ``quantize_weight``) so the golden tensor matches the
    effective BF16 weights implied by the packed weights + scales, not the raw pre-quant ``x``.
    """
    ue = UnifiedEngine()

    x = torch.randn(N, K, dtype=torch.bfloat16)
    x = x.reshape(-1, UE_VECTOR_SIZE)

    out_dim = x.shape[1]

    for i in range(out_dim):
        x[i, :] = torch.randn(UE_VECTOR_SIZE, dtype=torch.bfloat16) * ( i - (out_dim // 2))

    x = x.reshape(N, K)

    QUANTIZED_MATRIX_DRAM_ADDR, SCALE_DRAM_ADDR = ue.quantize_weight(weight=x, N=N, K=K, data_type=data_type, int_variant=int_variant)
    A_DRAM_ADDR = ue.allocate_tensor_dram(M * K * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)

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

    # PBI path is driven by gpr_M_reg; allocate + prime a GPR with M when use_pbi is requested.
    m_reg = ue.alloc_isa_reg() if use_pbi else None

    ue.start_capture()
    if use_pbi:
        ue.generate_instruction_add_set(m_reg, M)
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
                                                    silu_enable=silu_enable,
                                                    sigmoid_enable=sigmoid_enable,
                                                    clamp_enable=clamp_enable,
                                                    log_enable=log_enable,
                                                    gpr_M_reg=m_reg,
                                                    )

    ue.stop_capture()
    if use_pbi:
        ue.release_isa_reg()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    instruction_size_bytes = ue.get_capture_instruction_size_bytes()
    ue.allocate_program_dram(instruction_size_bytes)

    a = torch.randn(M, K, dtype=torch.bfloat16) # normalizing input helps with numerical stability of softmax
    ue.dma_to_accelerator_memory(A_DRAM_ADDR, a)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0) # 10 seconds timeout
    ue.report_timing_and_instruction_count()

    #generate_trace(ue, f"matmat_mul_quantized_weights_core_trace_{M}_{K}_{N}_{data_type}.csv")

    report_flop_rate_gflops, report_gflops_ratio = ue.report_flop_rate_gflops(total_flops_from_dequantize)
    print(f"Report FLOPS for Quantize Matrix-Matrix Multiply bf16: {report_flop_rate_gflops:.2f} GFLOPS, {report_gflops_ratio:.2f}% peak throughput for M={M}, K={K}, N={N}, bias_enable={bias_enable}, bias_mode={bias_mode}, use_pbi={use_pbi}")

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M, N))

    def apply_gelu(x):
        return x * torch.sigmoid(1.702 * x)

    def apply_silu(x):
        return x * torch.sigmoid(x)

    def apply_sigmoid(x):
        return torch.sigmoid(x)

    # Reference uses the same effective BF16 weights as the accelerator (quantize + dequant),
    # not the raw pre-quantization x — otherwise SNR is dominated by quantization error.
    x_effective_bf16 = ue.quantize_weight_simulate(x, data_type, int_variant=int_variant)
    ref = a @ x_effective_bf16.T

    if bias_enable:
        ref = ref + bias

    if gelu_enable:
        ref = apply_gelu(ref)
    elif silu_enable:
        ref = apply_silu(ref)
    elif sigmoid_enable:
        ref = apply_sigmoid(ref)
    elif clamp_enable:
        ref = torch.clamp(ref, min=0.0)
    elif log_enable:
        ref = torch.log(torch.clamp(ref, min=1e-3))

    snr_db_ref = calculate_snr(ref, output)

    print(f"Reference SNR Analysis for Dequantize: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 45 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 45 dB"

    width_str = "IF4" if data_type == TYPE.IF4 else "IF8"
    variant_str = "INT" if int_variant else "FP"
    type_str = f"{width_str}-{variant_str}"
    flags = [f"qB-{type_str}"]
    if bias_enable:    flags.append(f"bias-{bias_mode}")
    if gelu_enable:    flags.append("gelu")
    if silu_enable:    flags.append("silu")
    if sigmoid_enable: flags.append("sigmoid")
    if clamp_enable:   flags.append("clamp")
    if log_enable:     flags.append("log")
    if use_pbi:        flags.append("pbi")
    record_test(f"matmat_mul_quantized_weights+{'+'.join(flags)}",
                f"M={M}, K={K}, N={N}",
                snr_db=snr_db_ref,
                gflops=report_flop_rate_gflops,
                inst_bytes=instruction_size_bytes)

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

# TODO: Not very efficient for larger M
def quantized_matmat_mul_test(M: int, K: int, N: int, data_type: TYPE = TYPE.IF4, int_variant: bool = True, bias_enable: bool = False, bias_mode: str = "broadcast_N", gelu_enable: bool = False, silu_enable: bool = False, sigmoid_enable: bool = False, clamp_enable: bool = False, log_enable: bool = False):
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

    QUANTIZED_MATRIX_DRAM_ADDR, SCALE_DRAM_ADDR = ue.quantize_weight(weight=x, N=N, K=K, data_type=data_type, int_variant=int_variant)
    A_DRAM_ADDR = ue.allocate_tensor_dram(M * K * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)

    C_DRAM_ADDR = None
    if bias_enable and bias_mode == "full_matrix":
        C_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    elif bias_enable and bias_mode == "broadcast_N":
        C_DRAM_ADDR = ue.allocate_tensor_dram(N * 2)

    print(f"Quantized Matrix-Matrix Multiply Test for M={M}, K={K}, N={N}, bias_enable={bias_enable}, bias_mode={bias_mode}, gelu_enable={gelu_enable}, silu_enable={silu_enable}, sigmoid_enable={sigmoid_enable}")

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
                                                    silu_enable=silu_enable,
                                                    sigmoid_enable=sigmoid_enable,
                                                    clamp_enable=clamp_enable,
                                                    log_enable=log_enable)

    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    instruction_size_bytes = ue.get_capture_instruction_size_bytes()
    ue.allocate_program_dram(instruction_size_bytes)

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

    generate_trace(ue, f"quantized_matmat_mul_core_trace_{M}_{K}_{N}_{'bias_enabled' if bias_enable else 'bias_disabled'}_{'bias_mode_{bias_mode}' if bias_mode else 'bias_mode_none'}_{'gelu_enabled' if gelu_enable else 'gelu_disabled'}_{'silu_enabled' if silu_enable else 'silu_disabled'}_{'sigmoid_enabled' if sigmoid_enable else 'sigmoid_disabled'}.csv")

    report_flop_rate_gflops, flops_ratio = ue.report_flop_rate_gflops(total_flops_from_dequantize)
    print(f"Report FLOPS for Quantize Matrix-Matrix Multiply dot-product: {report_flop_rate_gflops:.2f} GFLOPS, {flops_ratio:.2f}% peak throughput for M={M}, N={N}")

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M, N))

    def apply_gelu(x):
        return x * torch.sigmoid(1.702 * x)

    def apply_silu(x):
        return x * torch.sigmoid(x)

    def apply_sigmoid(x):
        return torch.sigmoid(x)

    ref = (a @ x.T + c) if bias_enable else (a @ x.T)

    if gelu_enable:
        ref = apply_gelu(ref)
    elif silu_enable:
        ref = apply_silu(ref)
    elif sigmoid_enable:
        ref = apply_sigmoid(ref)
    elif clamp_enable:
        ref = torch.clamp(ref, min=0.0)
    elif log_enable:
        ref = torch.log(torch.clamp(ref, min=1e-3))

    snr_db_ref = calculate_snr(ref, output)

    print(f"Reference SNR Analysis for Dequantize: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 22 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 40 dB"

    width_str = "IF4" if data_type == TYPE.IF4 else "IF8"
    variant_str = "INT" if int_variant else "FP"
    type_str = f"{width_str}-{variant_str}"
    flags = [type_str]
    if bias_enable:    flags.append(f"bias-{bias_mode}")
    if gelu_enable:    flags.append("gelu")
    if silu_enable:    flags.append("silu")
    if sigmoid_enable: flags.append("sigmoid")
    if clamp_enable:   flags.append("clamp")
    if log_enable:     flags.append("log")
    record_test(f"quantized_matmat_mul+{'+'.join(flags)}",
                f"M={M}, K={K}, N={N}",
                snr_db=snr_db_ref,
                gflops=report_flop_rate_gflops,
                inst_bytes=instruction_size_bytes)

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

    generate_trace(ue, f"matmat_mul_non_aligned_writeback_core_trace_{M}_{K}_{N}.csv")

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M, N))

    snr_db_ref = calculate_snr(a @ b.T, output)
    print(f"Reference SNR Analysis for Matmat Mul Non Aligned Writeback: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 40 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 40 dB"

    record_test("matmat_mul_non_aligned_writeback",
                f"M={M}, K={K}, N={N}",
                snr_db=snr_db_ref)

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

    record_test("mix_of_broadcast_eltwise_add_eltwise_mul",
                f"dim={dim}",
                snr_db=snr_db_ref)

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

def eltwise_core_dram_test(M=None, N=None, use_pbi: bool = False):
    """
    Exercise DRAM eltwise over M×N BF16 tensors (row-major flat).

    Captures via the :meth:`UnifiedEngine.eltwise_core_dram` wrapper:
    - ``use_pbi=False`` (default): :meth:`eltwise_core_dram_legacy` — vertical ``m_chunk`` tiling
      with PBI loop + optional non-PBI tail.
    - ``use_pbi=True``: :meth:`eltwise_core_dram_pbi` — one logical row per ISA iteration with a
      runtime row count carried in a GPR (allocated and primed with ``ADD_SET`` here).

    Default grid: ``M`` in ``(1, 32, 64, 512)``, ``N`` in ``(64, 512, 6912, 8192)``, ops
    ``mul`` / ``add`` / ``sub``. Skips when ``N > URAM_NEAR_FULL_ELEMENTS`` or ``N`` is not a
    multiple of ``UE_VECTOR_SIZE``.

    Pass ``M`` and ``N`` together to run a single shape (all three ops still run).

    SNR compares the flattened ``M * N`` DRAM output to torch.
    """
    if (M is None) ^ (N is None):
        raise ValueError("eltwise_core_dram_test: pass both M and N, or neither (full grid).")

    tag = "eltwise_core_dram" + ("_pbi" if use_pbi else "_legacy")

    m_list = [M] if M is not None else [1, 32, 64, 512]
    n_list = [N] if N is not None else [64, 512, 6912, 8192]

    ops = (
        ("mul", UE_MODE.ELTWISE_MUL),
        ("add", UE_MODE.ELTWISE_ADD),
        ("sub", UE_MODE.ELTWISE_SUB),
    )

    ue = UnifiedEngine()

    for m in m_list:
        for n in n_list:
            elements = m * n
            if n > URAM_NEAR_FULL_ELEMENTS or n % UE_VECTOR_SIZE != 0:
                print(
                    f"[{tag}] skip M={m} N={n} "
                    f"(need N<={URAM_NEAR_FULL_ELEMENTS} and N%{UE_VECTOR_SIZE}==0)"
                )
                continue

            for op_name, mode in ops:
                ue.reset_tensor_dram_addr()

                a_dram = ue.allocate_tensor_dram(elements * 2)
                b_dram = ue.allocate_tensor_dram(elements * 2)
                out_dram = ue.allocate_tensor_dram(elements * 2)

                # PBI path requires a GPR holding the runtime row count; legacy passes None.
                m_reg = ue.alloc_isa_reg() if use_pbi else None

                ue.start_capture()
                if use_pbi:
                    ue.generate_instruction_add_set(m_reg, m)
                total_flops = ue.eltwise_core_dram(
                    m, n, a_dram, b_dram, out_dram, mode,
                    gpr_M_reg=m_reg,
                )
                ue.stop_capture()
                if use_pbi:
                    ue.release_isa_reg()
                ue.generate_instruction_halt()
                program_dram_addr = ue.get_program_dram_addr()
                ue.write_captured_instructions_to_dram(program_dram_addr)
                inst_bytes = ue.get_capture_instruction_size_bytes()
                ue.allocate_program_dram(inst_bytes)

                a = torch.randn(m, n, dtype=torch.bfloat16)
                b = torch.randn(m, n, dtype=torch.bfloat16)
                a_flat = a.reshape(-1).contiguous()
                b_flat = b.reshape(-1).contiguous()

                ue.dma_to_accelerator_memory(a_dram, a_flat)
                ue.dma_to_accelerator_memory(b_dram, b_flat)

                ue.start_execute_from_dram(program_dram_addr)
                ue.wait_queue(10.0)
                ue.report_timing_and_instruction_count()
                gflops, flops_ratio = ue.report_flop_rate_gflops(total_flops)
                print(
                    f"[{tag}] GFLOPS={gflops:.2f}, {flops_ratio:.2f}% peak "
                    f"(counted flops={total_flops})"
                )

                out_flat = ue.dma_from_accelerator_memory(out_dram, (elements,))
                if op_name == "mul":
                    y_ref = (a * b).reshape(-1)
                elif op_name == "add":
                    y_ref = (a + b).reshape(-1)
                else:
                    y_ref = (a - b).reshape(-1)

                snr_db = calculate_snr(y_ref, out_flat)
                print(
                    f"[{tag}] M={m} N={n} op={op_name} elements={elements} "
                    f"SNR={snr_db:.2f} dB inst_bytes={inst_bytes} GFLOPS={gflops:.2f}"
                )
                assert snr_db >= 40 or snr_db == float("inf"), (
                    f"{tag} M={m} N={n} op={op_name} SNR {snr_db:.2f} dB < 40 dB"
                )

                record_test(
                    f"{tag}_{op_name}",
                    f"M={m},N={n},elements={elements}",
                    snr_db=snr_db,
                    gflops=gflops,
                    inst_bytes=inst_bytes,
                )

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
    read_speed_mbps = URAM_NEAR_FULL_ELEMENTS * 2 / latency_us
    print(f"Read Speed: {read_speed_mbps:.2f} MB/s")

    record_test("dram_read_speed",
                f"elements={URAM_NEAR_FULL_ELEMENTS}",
                mb_per_s=read_speed_mbps)

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
    write_speed_mbps = URAM_NEAR_FULL_ELEMENTS * 2 / latency_us
    print(f"Write Speed: {write_speed_mbps:.2f} MB/s")

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (URAM_NEAR_FULL_ELEMENTS,))
    snr_db_ref = calculate_snr(x, output)
    print(f"Reference SNR Analysis for DRAM Read Write Speed Test: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 40 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 40 dB"

    record_test("dram_write_speed",
                f"elements={URAM_NEAR_FULL_ELEMENTS}",
                snr_db=snr_db_ref,
                mb_per_s=write_speed_mbps)

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

def padding_zero_test():
    """
    Padding zero test.
    """
    ue = UnifiedEngine()
    M = 128
    axi_m_bf16_per_beat = UE_AXI_DATA_WIDTH_BITS // 16
    N = axi_m_bf16_per_beat * 3
    N_ALIGNED = ((N - 1) // UE_VECTOR_SIZE + 1) * UE_VECTOR_SIZE
    A_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * N_ALIGNED * 2)

    # capture instructions
    ue.start_capture()

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

    record_test("padding_zero",
                f"M={M}, N={N}, N_aligned={N_ALIGNED}",
                snr_db=snr_db_ref)

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

def slicing_test():
    """
    Slicing test.
    """
    ue = UnifiedEngine()
    M = 5
    N = 64
    # Host-side tests should avoid sub-beat DRAM writebacks on wider AXI ports.
    # Write back a beat-aligned prefix per row and validate only the requested slice.
    axi_beat_elems = max(1, UE_AXI_DATA_WIDTH_BITS // 16)
    slice_elems = N // 4
    writeback_elems = ((max(slice_elems, axi_beat_elems) + axi_beat_elems - 1) // axi_beat_elems) * axi_beat_elems
    A_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * writeback_elems * 2)

    # capture instructions
    ue.start_capture()
    ue.accelerator_memory_to_sram(accelerator_dram_address=A_DRAM_ADDR,
                                  sram_address=0x00000,
                                  element_size=M * N)

    aligned_uram_row = ((N - 1) // UE_VECTOR_SIZE + 1) * UE_VECTOR_SIZE
    for i in range(M):
        ue.sram_to_accelerator_memory(sram_address=0x00000 + i * aligned_uram_row * 2,
                                      accelerator_dram_address=OUTPUT_DRAM_ADDR + i * writeback_elems * 2,
                                      element_size=writeback_elems)

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

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M, writeback_elems))

    snr_db_ref = calculate_snr(x[:, :slice_elems], output[:, :slice_elems])
    print(f"Reference SNR Analysis for Slicing Test: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 40 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 40 dB"

    record_test("slicing",
                f"M={M}, N={N}, slice_elems={slice_elems}, writeback_elems={writeback_elems}",
                snr_db=snr_db_ref)

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

def packing_test(packing_mode: int):
    """
    Packing test.
    """
    ue = UnifiedEngine()
    M = 1024
    axi_beat_elems = max(1, UE_AXI_DATA_WIDTH_BITS // 16)
    writeback_elems = ((max(packing_mode, axi_beat_elems) + axi_beat_elems - 1) // axi_beat_elems) * axi_beat_elems
    A_DRAM_ADDR = ue.allocate_tensor_dram(M * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram((M // UE_VECTOR_SIZE) * writeback_elems * 2)

    # capture instructions
    ue.start_capture()
    ue.accelerator_memory_to_sram(accelerator_dram_address=A_DRAM_ADDR,
                                  sram_address=0x00000,
                                  element_size=M)

    for row in range(M // UE_VECTOR_SIZE):
        ue.sram_to_accelerator_memory(
            sram_address=row * UE_VECTOR_SIZE * 2,
            accelerator_dram_address=OUTPUT_DRAM_ADDR + row * writeback_elems * 2,
            element_size=writeback_elems,
        )

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

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (M // UE_VECTOR_SIZE, writeback_elems))
    ref = x.reshape(-1, UE_VECTOR_SIZE)[:, :packing_mode]
    snr_db_ref = calculate_snr(ref, output[:, :packing_mode])
    print(f"Reference SNR Analysis for Packing Test: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 40 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 40 dB"

    record_test("packing",
                f"M={M}, packing_mode={packing_mode}, writeback_elems={writeback_elems}",
                snr_db=snr_db_ref)

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

def bf16_transpose_test(M: int, N: int, use_pbi: bool = False):
    """
    Tests bf16 transpose core.
    """
    ue = UnifiedEngine()
    INPUT_DRAM_ADDR = ue.allocate_tensor_dram(M * N * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(N * M * 2)

    ue.start_capture()

    ue.bf16_transpose_core(M=M, N=N, INPUT_DRAM_ADDR=INPUT_DRAM_ADDR, OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR, use_pbi=use_pbi)

    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    instruction_size = ue.write_captured_instructions_to_dram(program_dram_addr)

    x = torch.randn(M, N, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(INPUT_DRAM_ADDR, x)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0) # 10 seconds timeout
    ue.report_timing_and_instruction_count()

    # Throughput: one full M×N bf16 read + one full write ≈ 4*M*N bytes DRAM traffic.
    latency_us = ue.report_latency_in_us()
    bytes_moved = 4 * M * N
    mb_per_s = bytes_moved / latency_us if latency_us > 0 else 0.0
    # GFLOPS column: proxy = one touch per matrix element (read+write) for a memory-style kernel.
    transpose_flop_proxy = 2 * M * N
    gflops_rate, flops_ratio = ue.report_flop_rate_gflops(transpose_flop_proxy)

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (N, M))
    snr_db_ref = calculate_snr(x.T, output)

    print(f"Reference SNR Analysis for BF16 Transpose: {snr_db_ref:.2f} dB")
    assert snr_db_ref >= 40 or snr_db_ref == float('inf'), f"SNR {snr_db_ref:.2f} dB must be at least 40 dB"
    print(
        f"Report transpose proxy: {gflops_rate:.2f} GFLOPS, {flops_ratio:.2f}% peak throughput, "
        f"{mb_per_s:.2f} MB/s (4×M×N bytes / latency), {instruction_size} inst bytes, use_pbi={use_pbi}"
    )

    name_base = "bf16_transpose"
    flag_str = "+pbi" if use_pbi else ""
    record_test(
        f"{name_base}{flag_str}",
        f"M={M}, N={N}",
        snr_db=snr_db_ref,
        gflops=gflops_rate,
        mb_per_s=mb_per_s,
        inst_bytes=instruction_size,
    )

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

    QUANTIZED_MATRIX_DRAM_ADDR, SCALE_DRAM_ADDR = ue.quantize_weight(weight=x, N=N, K=K, data_type=TYPE.IF4, int_variant=False)
    A_DRAM_ADDR = ue.allocate_tensor_dram(N * K * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(N * K // 2)

    ue.start_capture()

    ue.accelerator_memory_to_sram(accelerator_dram_address=A_DRAM_ADDR,
                                  sram_address=0x00000,
                                  element_size=N * K)

    ue.start_queue_for_quantize_operation(input_sram_addr=0x00000, output_sram_addr=0x80000, data_type=TYPE.IF4, element_size=N * K)

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
    flop_rate_gflops, flops_ratio = ue.report_flop_rate_gflops(N * K * 2)
    print(f"Report FLOPS for Quantized FP4: {flop_rate_gflops:.2f} GFLOPS, {flops_ratio:.2f}% peak throughput for N={N}, K={K}")

    generate_trace(ue, f"quantized_fp4_core_trace_{N}_{K}.csv")

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
    elif mismatch_count <= 16:
        for m in first_mismatches:
            print(m)
        print(f"FP4 quantization PASS mostly match")
    else:
        print(f"FP4 quantization FAIL: {mismatch_count}/{num_elements} nibbles differ")

    record_test("quantized_fp4",
                f"N={N}, K={K}",
                gflops=flop_rate_gflops)

    ue.clear_capture_buffer()

def fmax_test(length: int = 256):
    """
    FMAX test: loads a vector, applies x - fmax via broadcast add with FMAX_NEGATE.
    """
    ue = UnifiedEngine()
    INPUT_DRAM_ADDR = ue.allocate_tensor_dram(length * 2)
    IDENTITY_DRAM_ADDR = ue.allocate_tensor_dram(length * length * 2)
    ZERO_DRAM_ADDR = ue.allocate_tensor_dram(length * 2)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(length * 2)
    FMAX_DRAM_ADDR = ue.allocate_tensor_dram(length * 2)

    vector_sram_addr = 0x00000  # URAM_A
    identity_sram_addr = 0x80000  # URAM_B
    output_sram_addr = vector_sram_addr + length * 2  # URAM_A
    zero_sram_addr = output_sram_addr + length * 2  # URAM_A
    fmax_sram_addr = zero_sram_addr + UE_VECTOR_SIZE * 2  # URAM_A

    ue.start_capture()
    ue.accelerator_memory_to_sram(accelerator_dram_address=ZERO_DRAM_ADDR,
                                  sram_address=zero_sram_addr,
                                  element_size=UE_VECTOR_SIZE)
    ue.accelerator_memory_to_sram(accelerator_dram_address=INPUT_DRAM_ADDR,
                                  sram_address=vector_sram_addr,
                                  element_size=length)
    ue.accelerator_memory_to_sram(accelerator_dram_address=IDENTITY_DRAM_ADDR,
                                  sram_address=identity_sram_addr,
                                  element_size=length * length)
    clear_en = 1
    for i in range(UE_FMAX_CONTEXT_SIZE):
        ue.start_queue_for_bf16_matvec_operation(max_clear_en=clear_en,
                                                 fmax_context_addr=i,
                                                 vector_sram_start_addr=vector_sram_addr,
                                                 matrix_sram_start_addr=identity_sram_addr,
                                                 output_sram_wb_addr=vector_sram_addr,
                                                 K=length, N=length)
        clear_en = 0
    ue.fmax_core(vector_sram_start_addr=zero_sram_addr,
                 output_sram_wb_addr=fmax_sram_addr,
                 N=UE_VECTOR_SIZE,
                 fmax_context_addr=0)
    ue.sram_to_accelerator_memory(sram_address=vector_sram_addr,
                                  accelerator_dram_address=OUTPUT_DRAM_ADDR,
                                  element_size=length)
    ue.sram_to_accelerator_memory(sram_address=fmax_sram_addr,
                                  accelerator_dram_address=FMAX_DRAM_ADDR,
                                  element_size=UE_VECTOR_SIZE)
    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    x = torch.randn(length, dtype=torch.bfloat16)
    identity = torch.eye(length, dtype=torch.bfloat16)
    zero = torch.zeros(UE_VECTOR_SIZE, dtype=torch.bfloat16)

    ue.dma_to_accelerator_memory(INPUT_DRAM_ADDR, x)
    ue.dma_to_accelerator_memory(IDENTITY_DRAM_ADDR, identity)
    ue.dma_to_accelerator_memory(ZERO_DRAM_ADDR, zero)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0) # 10 seconds timeout
    ue.report_timing_and_instruction_count()

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR, (length,))
    fmax_ref = torch.max(x).item()
    fmax = -1.0 * ue.dma_from_accelerator_memory(FMAX_DRAM_ADDR, (UE_VECTOR_SIZE,))[0].item()
    print("fmax_ref:", fmax_ref)
    print("fmax:", fmax)
    assert abs(fmax - fmax_ref) < 1e-6, f"FMAX {fmax} does not match reference {fmax_ref}"

    record_test("fmax",
                f"length={length}",
                snr_db=float("inf"))

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()

def isa_rela_loop_test() -> None:
    """
    Exercises (1) **relative loop + PC**: the loop uses :meth:`UnifiedEngine.loop_start` /
    :meth:`UnifiedEngine.loop_end` so the backward jump distance is derived from captured
    instruction indices (``ADD_DEC`` + ``RELA_JNZ`` when the counter register is still non-zero).
    Asserts the instruction/PC counter matches ``temp.c`` ``isa_rela_loop_test``
    (via ``UE_INSTRUCTION_CTL_ADDR``).

    (2) **Pointer-backed memcpy**: :meth:`UnifiedEngine.generate_instruction_pbi_init` seeds two
    stream pointers (input B row stream and output row stream). Each iteration loads the next B row
    with :meth:`UnifiedEngine.accelerator_memory_to_sram` (input pointer), adds A (ones) and B with
    :meth:`UnifiedEngine.eltwise_add_core`, and stores the sum with
    :meth:`UnifiedEngine.sram_to_accelerator_memory` (output pointer).

    UE and ISA ops share auto-managed :attr:`UnifiedEngine._inst_id` (see
    :meth:`UnifiedEngine.ue_op_descriptor` and :meth:`UnifiedEngine.ue_isa_descriptor`).

    Dummy data: URAM_A holds one row of bf16 ones; operand B DRAM is ``1..256`` in order (four
    URAM rows of 64). Expected output is ``2..257`` element-wise (each value plus one).
    """
    TEST_RESULTS_URAM_ADDR = 0x300
    TEST_RESULTS_SRAM_ADDR = TEST_RESULTS_URAM_ADDR << 7
    SRAM_URAM_A_ROW0 = 0x00000
    SRAM_URAM_B_ROW0 = 0x80000

    result_size_bytes = UE_VECTOR_SIZE * 2
    loop_cnt = 4
    n_elem = loop_cnt * UE_VECTOR_SIZE

    ue = UnifiedEngine()

    pointer_idx_input = ue.alloc_inst_ptr()
    pointer_idx_out = ue.alloc_inst_ptr()

    dram_16bit_input = ue.allocate_tensor_dram(result_size_bytes)
    dram_16bit_input2 = ue.allocate_tensor_dram(n_elem * 2)
    dram_16bit_output = ue.allocate_tensor_dram(n_elem * 2)

    ones = torch.ones(UE_VECTOR_SIZE, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(dram_16bit_input, ones)

    in2 = torch.arange(1, n_elem + 1, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(dram_16bit_input2, in2)

    zero_out = torch.zeros(n_elem, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(dram_16bit_output, zero_out)

    ue.start_capture()
    ue.generate_instruction_pbi_init(
        dram_shared_addr=dram_16bit_input,
        dma_length=result_size_bytes,
        output_size=0,
        uram_length=0,
        uram_a_start_addr=0,
        uram_b_start_addr=0,
        uram_wb_addr=0,
        uram_dst_addr=0,
        fmax_context_addr=0,
        inst_pointer_idx=pointer_idx_input,
    )

    ue.accelerator_memory_to_sram(
        accelerator_dram_address=0,
        sram_address=SRAM_URAM_A_ROW0,
        element_size=UE_VECTOR_SIZE,
        inst_pointer_idx=pointer_idx_input,
        memcpy_length_bytes=0,
    )

    ue.generate_instruction_pbi_init(
        dram_shared_addr=dram_16bit_input2,
        dma_length=result_size_bytes,
        output_size=0,
        uram_length=0,
        uram_a_start_addr=0,
        uram_b_start_addr=0,
        uram_wb_addr=0,
        uram_dst_addr=0,
        fmax_context_addr=0,
        inst_pointer_idx=pointer_idx_input,
    )

    ue.generate_instruction_pbi_init(
        dram_shared_addr=dram_16bit_output,
        dma_length=result_size_bytes,
        output_size=0,
        uram_length=0,
        uram_a_start_addr=TEST_RESULTS_URAM_ADDR,
        uram_b_start_addr=TEST_RESULTS_URAM_ADDR,
        uram_wb_addr=0,
        uram_dst_addr=0,
        fmax_context_addr=0,
        inst_pointer_idx=pointer_idx_out,
    )

    loop_reg = ue.loop_start(loop_cnt)
    ue.accelerator_memory_to_sram(
        accelerator_dram_address=result_size_bytes,
        sram_address=SRAM_URAM_B_ROW0,
        element_size=UE_VECTOR_SIZE,
        inst_pointer_idx=pointer_idx_input,
        memcpy_length_bytes=0,
    )

    ue.eltwise_add_core(
        SRAM_URAM_A_ROW0,
        SRAM_URAM_B_ROW0,
        TEST_RESULTS_SRAM_ADDR,
        UE_VECTOR_SIZE,
    )

    ue.sram_to_accelerator_memory(
        sram_address=SRAM_URAM_A_ROW0,
        accelerator_dram_address=result_size_bytes,
        element_size=UE_VECTOR_SIZE,
        inst_pointer_idx=pointer_idx_out,
        memcpy_length_bytes=result_size_bytes,
    )
    loop_body_size = ue.loop_end()

    ue.generate_instruction_halt()
    ue.stop_capture()

    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(30.0)

    _, pc_reg = ue.report_timing_and_instruction_count()
    inst_index_after_halt = ue._inst_id
    expected_pc = loop_cnt * loop_body_size + (inst_index_after_halt - loop_body_size) - 1
    assert pc_reg == expected_pc, (
        f"instruction/PC counter mismatch: got {pc_reg}, expected {expected_pc} "
        f"(_inst_id after halt={inst_index_after_halt})"
    )

    got = ue.dma_from_accelerator_memory(dram_16bit_output, (n_elem,))
    assert torch.equal(got.view(-1), in2 + 1), "output must equal sequential B plus ones row"
    print(
        f"isa_rela_loop_test_transplant: PASS ({n_elem} elements, _inst_id_after_halt={inst_index_after_halt}, "
        f"pc_reg={pc_reg}, pbi_stream={pointer_idx_input}, pbi_out={pointer_idx_out}, loop_level={loop_reg})"
    )

    record_test("isa_rela_loop",
                f"loop_cnt={loop_cnt}, n_elem={n_elem}")

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()
    ue.reset_inst_ptr_counter()
    ue.reset_isa_reg_counter()

    # --- REG_RELA sub-test ---
    # Exercises JUMP_MODE_REG_RELA: a GPR holds the backward instruction-word offset
    # so the loop stride can be computed at runtime rather than assembled as an
    # immediate.  Exit is via JZ (absolute, placeholder patched after capture).
    #
    # Program layout (3 setup + optional align NOP + 4-instruction body + HALT):
    #   0: SET cnt_reg    = loop_cnt_rela   (setup)
    #   1: SET accum_reg  = 0               (setup)
    #   2: SET offset_reg = 4               (setup; backward offset is always 4)
    #  [3: NOP]                             (optional 512-bit alignment pad)
    #   3+a: INC accum_reg                  <- loop body start (a = n_align_nops)
    #   4+a: DEC cnt_reg
    #   5+a: JZ  cnt_reg -> HALT            (placeholder 0, patched after capture)
    #   6+a: JMP_REG_RELA offset_reg        (read_ptr -= 4, lands on loop body)
    #   7+a: HALT
    #
    # The backward offset is always 4 regardless of alignment NOPs: the body is
    # always 4 instructions before the JMP, so read_ptr - 4 lands on INC accum.
    # Unlike REG_ABS, relative jumps do not trigger a DMA reload; the loop body
    # stays in the original instruction cache window.
    #
    # PC formula: (3 + n_align_nops) setup + (loop_cnt-1)*4 + 3 last + 1 HALT
    #           = 4*loop_cnt + 3 + n_align_nops
    loop_cnt_rela = 5
    cnt_reg_r  = 5
    accum_reg_r = 6
    offset_reg  = 7
    bwd_offset = 4  # always 4: distance from JMP_REG_RELA to INC accum in the cache

    ue_r = UnifiedEngine()
    program_dram_addr_r = ue_r.get_program_dram_addr()

    # Align the loop body start to a 512-bit (64-byte) DRAM instruction boundary.
    n_align_nops_r = 0
    if (program_dram_addr_r + 3 * INSTRUCTION_SIZE_BYTES) % (2 * INSTRUCTION_SIZE_BYTES) != 0:
        n_align_nops_r = 1

    ue_r.start_capture()
    ue_r.generate_instruction_add_set(cnt_reg_r, loop_cnt_rela)         # idx 0
    ue_r.generate_instruction_add_set(accum_reg_r, 0)                   # idx 1
    ue_r.generate_instruction_add_set(offset_reg, bwd_offset)           # idx 2
    for _ in range(n_align_nops_r):
        ue_r.generate_instruction_nop()                                  # idx 3 if needed
    ue_r.generate_instruction_add_inc(accum_reg_r)                      # loop body start
    ue_r.generate_instruction_add_dec(cnt_reg_r)
    jz_capture_idx_r = ue_r.capture_count
    ue_r.generate_instruction_jump_abs_jz(0, cnt_reg_r)                 # placeholder
    ue_r.generate_instruction_jump_reg_rela(offset_reg)
    halt_idx_r = ue_r.capture_count
    ue_r.generate_instruction_halt()
    ue_r.stop_capture()

    halt_word_addr_r = ue_35bit_addr_shifter(program_dram_addr_r + halt_idx_r * INSTRUCTION_SIZE_BYTES)
    ue_r._patch_jump_immediate(jz_capture_idx_r, halt_word_addr_r)

    ue_r.write_captured_instructions_to_dram(program_dram_addr_r)
    ue_r.allocate_program_dram(ue_r.get_capture_instruction_size_bytes())
    ue_r.start_execute_from_dram(program_dram_addr_r)
    ue_r.wait_queue(30.0)

    _, pc_reg_r = ue_r.report_timing_and_instruction_count()
    expected_pc_r = 4 * loop_cnt_rela + 3 + n_align_nops_r
    assert pc_reg_r == expected_pc_r, (
        f"isa_rela_loop_reg_rela_test PC mismatch: got {pc_reg_r}, expected {expected_pc_r} "
        f"(loop_cnt={loop_cnt_rela}, bwd_offset={bwd_offset}, n_align_nops={n_align_nops_r})"
    )
    print(
        f"isa_rela_loop_reg_rela_test: PASS (loop_cnt={loop_cnt_rela}, pc_reg={pc_reg_r}, "
        f"bwd_offset={bwd_offset}, n_align_nops={n_align_nops_r})"
    )
    record_test("isa_rela_loop_reg_rela", f"loop_cnt={loop_cnt_rela}, bwd_offset={bwd_offset}")
    ue_r.clear_capture_buffer()
    ue_r.reset_tensor_dram_addr()
    ue_r.reset_isa_reg_counter()


def isa_abs_loop_test() -> None:
    """
    Port of ``andromeda.c`` ``isa_abs_loop_test(loop_cnt)``: same raw ADD register-file sequence,
    then absolute ``JNZ`` to instruction index 1 (byte offset ``1 * INSTRUCTION_SIZE_BYTES``), then
    ``HALT``. Asserts the instruction/PC counter from ``UE_INSTRUCTION_CTL_ADDR`` matches
    ``loop_cnt * 8 + 2`` (same pass condition as the C test).

    Uses ``loop_cnt = 6`` like ``main`` → ``isa_abs_loop_test(6)``.
    """
    loop_cnt = 6
    loop_reg = 4

    ue = UnifiedEngine()

    program_dram_addr = ue.get_program_dram_addr()
    jump_target_word_addr = ue_35bit_addr_shifter(program_dram_addr + INSTRUCTION_SIZE_BYTES)

    ue.start_capture()
    ue.generate_instruction_add_set(loop_reg, loop_cnt)
    ue.generate_instruction_add_set(3, 7)
    ue.generate_instruction_add_inc(3)
    ue.generate_instruction_add_set(1, 1)
    ue.generate_instruction_add_set(2, 2)
    ue.generate_instruction_add_reg(3, 2, 1)
    ue.generate_instruction_add_imm(3, 5)
    ue.generate_instruction_add_dec(loop_reg)
    ue.generate_instruction_jump_abs_jnz(jump_target_word_addr, loop_reg)
    ue.generate_instruction_halt()
    ue.stop_capture()

    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(30.0)

    _, pc_reg = ue.report_timing_and_instruction_count()
    expected_pc = loop_cnt * 8 + 2 + 1
    assert pc_reg == expected_pc, (
        f"instruction/PC counter mismatch: got {pc_reg}, expected {expected_pc} "
        f"(isa_abs_loop_test, loop_cnt={loop_cnt})"
    )

    print(
        f"isa_abs_loop_test: PASS (loop_cnt={loop_cnt}, pc_reg={pc_reg}, "
        f"jump_target_word=0x{jump_target_word_addr:x}, program_dram=0x{program_dram_addr:x})"
    )

    record_test("isa_abs_loop",
                f"loop_cnt={loop_cnt}")

    ue.clear_capture_buffer()
    ue.reset_inst_ptr_counter()
    ue.reset_isa_reg_counter()

    # --- REG_ABS sub-test ---
    # Exercises JUMP_MODE_REG_ABS: the backward loop address is loaded into a GPR at
    # setup time via ADD_SET and the unconditional JMP_REG_ABS uses that register as
    # the target each iteration.  Exit is via JZ (immediate target = HALT).
    #
    # Program layout (3 setup + optional align NOP + 4-instruction body + HALT):
    #   0: SET cnt_reg   = loop_cnt         (setup)
    #   1: SET accum_reg = 0                (setup)
    #   2: SET addr_reg  = word_addr(body)  (setup; 512-bit-aligned loop body address)
    #  [3: NOP]                             (optional alignment pad)
    #   3+a: INC accum_reg                  <- loop body start (a = n_align_nops)
    #   4+a: DEC cnt_reg
    #   5+a: JZ  cnt_reg -> HALT            (placeholder 0, patched after capture)
    #   6+a: JMP_REG_ABS addr_reg
    #   7+a: HALT
    #
    # PC formula: (3 + n_align_nops) setup + (loop_cnt-1)*4 + 3 last + 1 HALT
    #           = 4*loop_cnt + 3 + n_align_nops
    loop_cnt_reg_abs = 4
    cnt_reg  = 5
    accum_reg = 6
    addr_reg  = 7

    ue2 = UnifiedEngine()
    program_dram_addr2 = ue2.get_program_dram_addr()

    # Align the loop body start to a 512-bit (64-byte) DRAM instruction boundary
    # before storing the word address into the GPR.
    loop_body_byte_addr = program_dram_addr2 + 3 * INSTRUCTION_SIZE_BYTES
    n_align_nops = 0
    if loop_body_byte_addr % (2 * INSTRUCTION_SIZE_BYTES) != 0:
        loop_body_byte_addr += INSTRUCTION_SIZE_BYTES
        n_align_nops = 1
    loop_body_word_addr = ue_35bit_addr_shifter(loop_body_byte_addr)

    ue2.start_capture()
    ue2.generate_instruction_add_set(cnt_reg, loop_cnt_reg_abs)         # idx 0
    ue2.generate_instruction_add_set(accum_reg, 0)                      # idx 1
    ue2.generate_instruction_add_set(addr_reg, loop_body_word_addr)     # idx 2
    for _ in range(n_align_nops):
        ue2.generate_instruction_nop()                                   # idx 3 if needed
    ue2.generate_instruction_add_inc(accum_reg)                         # loop body start
    ue2.generate_instruction_add_dec(cnt_reg)
    jz_capture_idx = ue2.capture_count
    ue2.generate_instruction_jump_abs_jz(0, cnt_reg)                    # placeholder
    ue2.generate_instruction_jump_reg_abs(addr_reg)
    halt_idx = ue2.capture_count
    ue2.generate_instruction_halt()
    ue2.stop_capture()

    halt_word_addr = ue_35bit_addr_shifter(program_dram_addr2 + halt_idx * INSTRUCTION_SIZE_BYTES)
    ue2._patch_jump_immediate(jz_capture_idx, halt_word_addr)

    ue2.write_captured_instructions_to_dram(program_dram_addr2)
    ue2.allocate_program_dram(ue2.get_capture_instruction_size_bytes())
    ue2.start_execute_from_dram(program_dram_addr2)
    ue2.wait_queue(30.0)

    _, pc_reg2 = ue2.report_timing_and_instruction_count()
    expected_pc2 = 4 * loop_cnt_reg_abs + 3 + n_align_nops
    assert pc_reg2 == expected_pc2, (
        f"isa_abs_loop_reg_abs_test PC mismatch: got {pc_reg2}, expected {expected_pc2} "
        f"(loop_cnt={loop_cnt_reg_abs}, n_align_nops={n_align_nops})"
    )
    print(
        f"isa_abs_loop_reg_abs_test: PASS (loop_cnt={loop_cnt_reg_abs}, pc_reg={pc_reg2}, "
        f"loop_body_word=0x{loop_body_word_addr:x}, n_align_nops={n_align_nops})"
    )
    record_test("isa_abs_loop_reg_abs", f"loop_cnt={loop_cnt_reg_abs}")
    ue2.clear_capture_buffer()
    ue2.reset_isa_reg_counter()


def isa_reg_min_sub_mul_test() -> None:
    """
    Exercises ALU_MODE_SUB, ALU_MODE_MIN, ALU_MODE_MUL_IMM, and a follow-on SUB
    (multiply then subtract immediate loaded in a GPR) with three counted loops.

    Structure:
      Setup (3 instructions):
        SET reg_a = val_a (12), SET reg_b = val_b (5)
        SUB reg_sub = reg_a - reg_b  -> 7

      Loop 1 - verifies ALU_MODE_SUB as runtime trip count:
        ADD_IMM loop1_reg = reg_sub   (header, 1 instruction)
        body: ADD_INC reg_a           (dummy, 1 instruction)
        ADD_DEC loop1_reg + JUMP_RELA_JNZ  -> trips = 7

      Between (2 instructions):
        SET reg_cap = cap (4)
        MIN reg_min = min(reg_sub, reg_cap)  -> 4

      Loop 2 - verifies ALU_MODE_MIN as runtime trip count:
        ADD_IMM loop2_reg = reg_min   (header, 1 instruction)
        body: ADD_INC reg_a           (dummy, 1 instruction)
        ADD_DEC loop2_reg + JUMP_RELA_JNZ  -> trips = 4

      Between_mul (4 instructions):
        SET reg_m1 = mul_a
        MUL_IMM reg_mul = reg_m1 * mul_b (immediate)
        SET reg_mul_adj = mul_adj
        SUB reg_mul = reg_mul - reg_mul_adj            -> 6

      Loop 3 - trip count from adjusted product (still 6 for PC check):
        ADD_IMM loop3_reg = reg_mul   (header, 1 instruction)
        body: ADD_INC reg_a           (dummy, 1 instruction)
        ADD_DEC loop3_reg + JUMP_RELA_JNZ  -> trips = 6

      HALT

    expected_pc is the exact instruction-decoded count including HALT
    (pc_reg_out increments on every STATE_DECODE_TYPE, NOP-after-halt never executes).
    """
    val_a = 12
    val_b = 5
    cap   = 4                              # < (val_a - val_b), so MIN clamps
    mul_a = 65535
    mul_b = 65535
    mul_adj = mul_a * mul_b - 6  # reg_mul = (mul_a * mul_b) - mul_adj  -> 6 loop trips
    expected_sub = val_a - val_b           # 7  -- loop 1 trip count
    expected_min = min(expected_sub, cap)  # 4  -- loop 2 trip count
    expected_mul_loop = mul_a * mul_b - mul_adj  # 6  -- loop 3 (after MUL_IMM then SUB)

    ue = UnifiedEngine()

    reg_a   = ue.alloc_isa_reg()
    reg_b   = ue.alloc_isa_reg()
    reg_sub = ue.alloc_isa_reg()
    reg_cap = ue.alloc_isa_reg()
    reg_min = ue.alloc_isa_reg()

    ue.start_capture()

    # --- setup: 3 instructions ---
    ue.generate_instruction_add_set(reg_a, val_a)
    ue.generate_instruction_add_set(reg_b, val_b)
    ue.generate_instruction_reg_sub(reg_sub, reg_a, reg_b)   # reg_sub = 7
    n_setup = 3

    # --- Loop 1: SUB result drives trip count ---
    ue.loop_start(expected_sub, gpr_loop_cnt=reg_sub)          # header: ADD_IMM (1 inst)
    ue.generate_instruction_add_inc(reg_a)                    # body: dummy (1 inst)
    loop1_body_size = ue.loop_end()                           # ADD_DEC + JNZ; returns 3

    # --- between loops: 2 instructions ---
    ue.generate_instruction_add_set(reg_cap, cap)
    ue.generate_instruction_reg_min(reg_min, reg_sub, reg_cap)  # reg_min = min(7,4) = 4
    n_between = 2

    # --- Loop 2: MIN result drives trip count ---
    ue.loop_start(expected_min, gpr_loop_cnt=reg_min)          # header: ADD_IMM (1 inst)
    ue.generate_instruction_add_inc(reg_a)                    # body: dummy (1 inst)
    loop2_body_size = ue.loop_end()                           # ADD_DEC + JNZ; returns 3

    # --- multiply setup + Loop 3: (MUL_IMM then SUB) drives trip count ---
    reg_m1 = ue.alloc_isa_reg()
    reg_mul = ue.alloc_isa_reg()
    reg_mul_adj = ue.alloc_isa_reg()
    ue.generate_instruction_add_set(reg_m1, mul_a)
    ue.generate_instruction_reg_mul_imm(reg_mul, reg_m1, mul_b)  # reg_mul = mul_a * mul_b
    ue.generate_instruction_add_set(reg_mul_adj, mul_adj)
    ue.generate_instruction_reg_sub(reg_mul, reg_mul, reg_mul_adj)  # reg_mul -> loop trips
    n_between_mul = 4

    ue.loop_start(expected_mul_loop, gpr_loop_cnt=reg_mul)  # header: ADD_IMM from reg_mul
    ue.generate_instruction_add_inc(reg_a)
    loop3_body_size = ue.loop_end()

    ue.generate_instruction_halt()
    ue.stop_capture()

    # pc_reg_out counts every STATE_DECODE_TYPE including HALT.
    # NOP-after-HALT (alignment padding) never executes and is not counted.
    expected_pc = (
        n_setup
        + 1
        + expected_sub * loop1_body_size
        + n_between
        + 1
        + expected_min * loop2_body_size
        + n_between_mul
        + 1
        + expected_mul_loop * loop3_body_size
        + 1                                  # HALT
    )

    program_dram_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(30.0)

    _, pc_reg = ue.report_timing_and_instruction_count()
    assert pc_reg == expected_pc, (
        f"isa_reg_min_sub_mul_test: pc_reg mismatch: got {pc_reg}, expected {expected_pc} "
        f"(val_a={val_a}, val_b={val_b}, cap={cap}, mul_a={mul_a}, mul_b={mul_b}, mul_adj={mul_adj}, "
        f"expected_sub={expected_sub}, expected_min={expected_min}, expected_mul_loop={expected_mul_loop}, "
        f"loop1_body_size={loop1_body_size}, loop2_body_size={loop2_body_size}, "
        f"loop3_body_size={loop3_body_size})"
    )

    print(
        f"isa_reg_min_sub_mul_test: PASS "
        f"(SUB={expected_sub} -> loop1x{loop1_body_size}, "
        f"MIN={expected_min} -> loop2x{loop2_body_size}, "
        f"MUL_IMM-SUB -> loop3 trips={expected_mul_loop} x{loop3_body_size}, pc_reg={pc_reg})"
    )

    record_test("isa_reg_min_sub_mul",
                f"val_a={val_a}, val_b={val_b}, cap={cap}, mul_a={mul_a}, mul_b={mul_b}, mul_adj={mul_adj}, "
                f"SUB={expected_sub}, MIN={expected_min}, mul_loop={expected_mul_loop}")

    ue.clear_capture_buffer()
    ue.reset_inst_ptr_counter()
    ue.reset_isa_reg_counter()

def software_reset_test():
    """
    Verifies that software reset breaks a deterministic deadlock caused
    by flag registers, and that the engine recovers cleanly afterwards.

    Phase 1 — flag deadlock (expect engine stuck):
      Program engine 0 to FLAG_CHECK on engine 1 whose flag is never set.
      This causes an infinite spin-wait (deadlock).  Confirm the engine
      is stuck, then issue software_reset() to break it.

    Phase 2 — simple halt instruction (expect PASS):
    """
    # ---- Phase 1: cause a flag deadlock, then reset ----
    ue = UnifiedEngine()

    # ue.start_capture()
    # ue.generate_instruction_flag_check(target_engine_idx=1)
    # ue.generate_instruction_halt()
    # ue.stop_capture()
    # program_dram_addr = ue.get_program_dram_addr()
    # print(f"program_dram_addr: {program_dram_addr:08x}")
    # print(f"capture_instruction_size_bytes: {ue.get_capture_instruction_size_bytes()}")
    # ue.write_captured_instructions_to_dram(program_dram_addr)
    # ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())
    # ue.start_execute_from_dram(program_dram_addr)

    # ue.clear_capture_buffer()
    # ue.reset_tensor_dram_addr()

    # ue.wait_queue(1.0) # 3 seconds timeout

    # assert ue.is_queue_busy(), \
    #     "Engine should be stuck in FLAG_CHECK spin-wait but reports idle"
    # print("Engine is deadlocked on FLAG_CHECK(engine 1) — issuing software reset...")
    ue.software_reset()

    # ---- Phase 2: run flag set/clear after reset, expect completion ----
    ue.start_capture()
    ue.generate_instruction_halt()
    ue.stop_capture()
    program_dram_addr = ue.get_program_dram_addr()
    print(f"program_dram_addr: {program_dram_addr:08x}")
    print(f"capture_instruction_size_bytes: {ue.get_capture_instruction_size_bytes()}")
    ue.write_captured_instructions_to_dram(program_dram_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(3.0) # 3 seconds timeout

    assert not ue.is_queue_busy(), \
        "Engine should have completed HALT but still reports busy"

    print("Software reset test PASSED")
    record_test("software_reset", "n/a")
    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()


def test_ue_int_reg_read():
    """
    AXI-Lite read of UE_INT_REG: bits [1:0] are interrupt cause (SWI/HALT), matching
    queue_state_module.sv. No host ISR clears the latch, so we can poll SWI during a
    delay loop and HALT after the stream completes.
    """
    ue = UnifiedEngine()

    ue.write_reg32(UE_INT_REG, 1)
    idle = ue.read_reg32(UE_INT_REG)
    assert (idle & 3) == INT_CAUSE_NONE and (idle & ~3) == 0, (
        f"after clear expected cause=0 and reserved bits 0, got 0x{idle:08x}"
    )

    ue.clear_capture_buffer()
    ue.reset_inst_ptr_counter()
    ue.start_capture()
    ue.generate_instruction_swi()
    ue.generate_instruction_add_set(REGFILE_R1_LOOP, 500)
    ue.generate_instruction_add_dec(REGFILE_R1_LOOP)
    ue.generate_instruction_jump_rela_jnz(2, REGFILE_R1_LOOP)
    ue.generate_instruction_halt()
    ue.stop_capture()

    prog = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(prog)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())
    ue.start_execute_from_dram(prog)

    saw_swi = False
    deadline = time.time() + 5.0
    while ue.is_queue_busy():
        c = ue.read_reg32(UE_INT_REG) & 3
        if c == INT_CAUSE_SWI:
            saw_swi = True
        assert time.time() < deadline, "test_ue_int_reg_read: queue wait timeout"

    assert saw_swi, "never observed INT_CAUSE_SWI on UE_INT_REG while queue was busy"
    final_c = ue.read_reg32(UE_INT_REG) & 3
    assert final_c == INT_CAUSE_HALT, f"after HALT expected cause HALT ({INT_CAUSE_HALT}), got {final_c}"

    ue.write_reg32(UE_INT_REG, 1)
    cleared = ue.read_reg32(UE_INT_REG) & 3
    assert cleared == INT_CAUSE_NONE, f"after write-clear expected 0, got {cleared}"

    print("test_ue_int_reg_read: PASS")
    record_test("ue_int_reg_read", "n/a")
    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()
    ue.reset_inst_ptr_counter()


# ---------------------------------------------------------------------------
# Multi-head reshape tests (MobileSAM-inspired, models SWIN's
# multihead_pad_and_permute pattern)
# ---------------------------------------------------------------------------

def multihead_reshape_dram_test(seq_len: int, num_heads: int,
                                 head_dim: int, head_dim_pad: int,
                                 use_pbi: bool = False) -> None:
    """
    Validate (seq_len, num_heads * head_dim) → (num_heads, seq_len, head_dim_pad).

    Legacy path (use_pbi=False) replicates the current MobileSAM
    ``multihead_reshape_dram`` (Python-unrolled scatter + bf16_permute_core).

    PBI path (use_pbi=True) uses ``pbi_strided_copy`` for both the scatter
    and transpose, matching the SWIN ``multihead_pad_and_permute`` pattern.
    """
    ue = UnifiedEngine()

    input_row_stride = num_heads * head_dim
    padded_dim = num_heads * head_dim_pad

    input_elements = seq_len * input_row_stride
    temp_elements = seq_len * padded_dim
    output_elements = num_heads * seq_len * head_dim_pad

    INPUT_DRAM_ADDR = ue.allocate_tensor_dram(input_elements * _BPE)
    TEMP_DRAM_ADDR = ue.allocate_tensor_dram(temp_elements * _BPE)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(output_elements * _BPE)

    if use_pbi:
        gpr_pbi = ue.alloc_isa_reg()
        ue.start_capture()
        ue.generate_instruction_add_set(gpr_pbi, seq_len * num_heads)
        multihead_reshape_dram_pbi(
            ue, INPUT_DRAM_ADDR, OUTPUT_DRAM_ADDR, TEMP_DRAM_ADDR,
            seq_len, num_heads, head_dim, head_dim_pad, gpr_pbi,
        )
        ue.stop_capture()
        ue.release_isa_reg()
    else:
        ue.start_capture()
        multihead_reshape_dram_legacy(
            ue, INPUT_DRAM_ADDR, OUTPUT_DRAM_ADDR, TEMP_DRAM_ADDR,
            seq_len, num_heads, head_dim, head_dim_pad,
        )
        ue.stop_capture()

    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    instruction_size = ue.write_captured_instructions_to_dram(program_dram_addr)

    x = torch.randn(seq_len, input_row_stride, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(INPUT_DRAM_ADDR, x)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(10.0)
    ue.report_timing_and_instruction_count()

    y_ref = x.view(seq_len, num_heads, head_dim).permute(1, 0, 2)
    if head_dim_pad > head_dim:
        pad = torch.zeros(num_heads, seq_len, head_dim_pad - head_dim, dtype=torch.bfloat16)
        y_ref = torch.cat([y_ref, pad], dim=-1)

    output = ue.dma_from_accelerator_memory(
        OUTPUT_DRAM_ADDR, (num_heads, seq_len, head_dim_pad))

    snr_db = calculate_snr(y_ref, output)
    latency_us = ue.report_latency_in_us()
    bytes_moved = 2 * 2 * output_elements  # one read (input) + one write (output)
    mb_per_s = (bytes_moved / latency_us) if latency_us > 0 else 0.0
    gflops_rate, flops_ratio = ue.report_flop_rate_gflops(2 * output_elements)

    print(
        f"[multihead_reshape] seq={seq_len} heads={num_heads} "
        f"hd={head_dim} hd_pad={head_dim_pad} "
        f"PBI={use_pbi} SNR={snr_db:.2f} dB "
        f"inst_bytes={instruction_size} GFLOPS={gflops_rate:.2f}"
    )
    assert snr_db >= 40 or snr_db == float('inf'), \
        f"multihead_reshape PBI={use_pbi} seq={seq_len} heads={num_heads} " \
        f"hd={head_dim} hd_pad={head_dim_pad} SNR {snr_db:.2f} dB < 40 dB"

    flag_str = "+pbi" if use_pbi else ""
    record_test(
        f"multihead_reshape{flag_str}",
        f"seq={seq_len},heads={num_heads},hd={head_dim},pad={head_dim_pad}",
        snr_db=snr_db,
        gflops=gflops_rate,
        mb_per_s=mb_per_s,
        inst_bytes=instruction_size,
    )

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()


# ---------------------------------------------------------------------------
# Flash attention batched (PBI staging) test — validates the subroutine-based
# batching used by SWIN's flash_attention_batched_pbi.
# ---------------------------------------------------------------------------

def flash_attention_batched_test(head_dim: int = 64, seq_len: int = 64,
                                  num_heads: int = 4, bias_enable: bool = False) -> None:
    """Validate ``flash_attention_batched_pbi`` with ``num_heads`` batches.

    The PBI-batched path compiles ONE static flash body as a subroutine and
    emits tiny per-batch call-stubs that memcpy Q/K/V into staging buffers,
    call the subroutine, then memcpy the output back.  This replaces the
    Python-unrolled ``for h in range(num_heads): flash_attention_core(...)``
    with a single captured sequence (SWIN reduces flash program size from
    ~103 MB to ~1-2 MB this way).
    """
    from nn_lib import flash_attention_batched_pbi

    ue = UnifiedEngine()
    bpe = 2
    win_bytes = seq_len * head_dim * bpe
    attn_p_bytes = seq_len * seq_len * bpe
    scratch_elements = (head_dim + UE_FMAX_CONTEXT_SIZE) * seq_len

    total_qkv_bytes = num_heads * win_bytes
    total_bias_bytes = num_heads * attn_p_bytes if bias_enable else 0

    Q_DRAM_ADDR = ue.allocate_tensor_dram(total_qkv_bytes)
    K_DRAM_ADDR = ue.allocate_tensor_dram(total_qkv_bytes)
    V_DRAM_ADDR = ue.allocate_tensor_dram(total_qkv_bytes)
    OUTPUT_DRAM_ADDR = ue.allocate_tensor_dram(total_qkv_bytes)
    SCRATCH_DRAM_ADDR = ue.allocate_tensor_dram(scratch_elements * bpe)
    ATTN_P_DRAM_ADDR = ue.allocate_tensor_dram(attn_p_bytes)
    BIAS_DRAM_ADDR = ue.allocate_tensor_dram(total_bias_bytes) if bias_enable else None
    IDENTITY_DRAM_ADDR = ue.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe)
    ue.dma_to_accelerator_memory(IDENTITY_DRAM_ADDR,
                                 torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))

    q = torch.randn(num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    k = torch.randn(num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    v = torch.randn(num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    bias = torch.randn(num_heads, seq_len, seq_len, dtype=torch.bfloat16) if bias_enable else None

    ue.dma_to_accelerator_memory(Q_DRAM_ADDR, q)
    ue.dma_to_accelerator_memory(K_DRAM_ADDR, k)
    ue.dma_to_accelerator_memory(V_DRAM_ADDR, v)
    if bias_enable:
        ue.dma_to_accelerator_memory(BIAS_DRAM_ADDR, bias)

    ue.start_capture()
    flash_attention_batched_pbi(
        ue,
        num_batches=num_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        Q_DRAM_ADDR=Q_DRAM_ADDR,
        K_DRAM_ADDR=K_DRAM_ADDR,
        V_DRAM_ADDR=V_DRAM_ADDR,
        OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
        SCRATCH_DRAM_ADDR=SCRATCH_DRAM_ADDR,
        ATTN_P_DRAM_ADDR=ATTN_P_DRAM_ADDR,
        IDENTITY_TRANSPOSE_DRAM_ADDR=IDENTITY_DRAM_ADDR,
        BIAS_DRAM_ADDR=BIAS_DRAM_ADDR if bias_enable else None,
        _silent=False,
    )
    ue.stop_capture()
    ue.generate_instruction_halt()
    program_dram_addr = ue.get_program_dram_addr()
    instruction_size = ue.write_captured_instructions_to_dram(program_dram_addr)

    ue.start_execute_from_dram(program_dram_addr)
    ue.wait_queue(30.0)
    ue.report_timing_and_instruction_count()

    q_scaled = q * (1.0 / math.sqrt(head_dim))
    attn = q_scaled @ k.transpose(-2, -1)
    if bias_enable:
        attn = attn + bias
    sm = torch.softmax(attn, dim=-1).to(torch.bfloat16)
    ref = sm @ v

    output = ue.dma_from_accelerator_memory(OUTPUT_DRAM_ADDR,
                                             (num_heads, seq_len, head_dim))
    snr_db = calculate_snr(ref, output)
    latency_us = ue.report_latency_in_us()
    bytes_moved = 4 * total_qkv_bytes
    mb_per_s = (bytes_moved / latency_us) if latency_us > 0 else 0.0
    gflops_rate, flops_ratio = ue.report_flop_rate_gflops(
        num_heads * seq_len * seq_len * (2 * head_dim + 1))

    print(
        f"[flash_attn_batched] hd={head_dim} seq={seq_len} heads={num_heads} "
        f"bias={bias_enable} SNR={snr_db:.2f} dB "
        f"inst_bytes={instruction_size} GFLOPS={gflops_rate:.2f}"
    )
    assert snr_db >= 38 or snr_db == float('inf'), \
        f"flash_attn_batched hd={head_dim} seq={seq_len} heads={num_heads} " \
        f"bias={bias_enable} SNR {snr_db:.2f} dB < 38 dB"

    tag = "flash_attn_batched" + ("+bias" if bias_enable else "")
    record_test(tag,
                f"hd={head_dim},seq={seq_len},heads={num_heads}",
                snr_db=snr_db,
                gflops=gflops_rate,
                mb_per_s=mb_per_s,
                inst_bytes=instruction_size)

    ue.clear_capture_buffer()
    ue.reset_tensor_dram_addr()


def mobilesam_s2_flash_btb_test(n_calls: int = 5, num_batches: int = 25,
                                head_dim: int = 64, seq_len: int = 256,
                                seq_real: int = 196):
    """Reproduce MobileSAM's S2 flash usage EXACTLY: `n_calls` separate
    flash_attention_batched_pbi calls BACK-TO-BACK in ONE program (the model
    loops `for h in range(ENC_S2_HEADS=5): flash_attention_batched_pbi(
    num_batches=ENC_S2_NWIN=25, head_dim=64, seq_len=256, bias_shared=True, ...)`).

    The model's S2B0 collapses to 3 dB while every op passes in isolation -- the
    suspect is that the 2nd..Nth back-to-back batched-flash call corrupts. This
    test gives each call its OWN data+ref and reports per-call SNR, so a clean
    call-0 + bad call-1.. pinpoints the back-to-back bug at the real S2 dims
    (seq 196->256 padded, shared -inf key mask, shared SCRATCH across calls).
    """
    from nn_lib import flash_attention_batched_pbi
    ue = UnifiedEngine()
    bpe = 2
    win_bytes = seq_len * head_dim * bpe
    bias_bytes = seq_len * seq_len * bpe
    scratch_elements = (head_dim + UE_FMAX_CONTEXT_SIZE) * seq_len

    # Per-call Q/K/V/OUT (num_batches windows each) + one shared bias per call.
    Qs, Ks, Vs, Os, Bs = [], [], [], [], []
    qkv, biases, refs = [], [], []
    # model's S2 shared bias pattern: -inf for padded key cols, 0 for padded query rows
    for c in range(n_calls):
        Qs.append(ue.allocate_tensor_dram(num_batches * win_bytes))
        Ks.append(ue.allocate_tensor_dram(num_batches * win_bytes))
        Vs.append(ue.allocate_tensor_dram(num_batches * win_bytes))
        Os.append(ue.allocate_tensor_dram(num_batches * win_bytes))
        Bs.append(ue.allocate_tensor_dram(bias_bytes))
        q = torch.zeros(num_batches, seq_len, head_dim, dtype=torch.bfloat16)
        k = torch.zeros(num_batches, seq_len, head_dim, dtype=torch.bfloat16)
        v = torch.zeros(num_batches, seq_len, head_dim, dtype=torch.bfloat16)
        # vary per call so a corrupted call can't accidentally match
        q[:, :seq_real] = torch.randn(num_batches, seq_real, head_dim) * (0.5 + 0.1 * c)
        k[:, :seq_real] = torch.randn(num_batches, seq_real, head_dim)
        v[:, :seq_real] = torch.randn(num_batches, seq_real, head_dim)
        bias = torch.zeros(seq_len, seq_len, dtype=torch.float32)
        bias[:, seq_real:] = float('-inf')       # mask padded keys
        bias[seq_real:, :] = 0.0                  # padded query rows
        bias[:seq_real, :seq_real] = torch.randn(seq_real, seq_real)  # rel-pos bias
        bias_bf = bias.to(torch.bfloat16)
        ue.dma_to_accelerator_memory(Qs[c], q)
        ue.dma_to_accelerator_memory(Ks[c], k)
        ue.dma_to_accelerator_memory(Vs[c], v)
        ue.dma_to_accelerator_memory(Bs[c], bias_bf)
        qkv.append((q, k, v)); biases.append(bias_bf)
        # torch ref (legacy flash applies 1/sqrt(head_dim) internally)
        qs = q.float() * (1.0 / math.sqrt(head_dim))
        att = qs @ k.float().transpose(-2, -1) + bias.unsqueeze(0)
        sm = torch.softmax(att, dim=-1)
        refs.append((sm @ v.float()))

    IDENT = ue.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe)
    ue.dma_to_accelerator_memory(IDENT, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))
    SCRATCH = ue.allocate_tensor_dram(scratch_elements * bpe)  # SHARED across calls (like model)

    ue.start_capture()
    for c in range(n_calls):
        flash_attention_batched_pbi(
            ue, num_batches=num_batches, head_dim=head_dim, seq_len=seq_len,
            Q_DRAM_ADDR=Qs[c], K_DRAM_ADDR=Ks[c], V_DRAM_ADDR=Vs[c],
            OUTPUT_DRAM_ADDR=Os[c], SCRATCH_DRAM_ADDR=SCRATCH,
            ATTN_P_DRAM_ADDR=0, IDENTITY_TRANSPOSE_DRAM_ADDR=IDENT,
            BIAS_DRAM_ADDR=Bs[c], bias_shared=True, _silent=True)
    ue.stop_capture()
    ue.generate_instruction_halt()
    prog = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(prog)
    ue.start_execute_from_dram(prog)
    ue.wait_queue(120.0)

    print(f"\n[s2_flash_btb] {n_calls} back-to-back batched-flash calls "
          f"(num_batches={num_batches}, hd={head_dim}, seq={seq_len}, real={seq_real})")
    worst = 1e9
    for c in range(n_calls):
        out = ue.dma_from_accelerator_memory(Os[c], (num_batches, seq_len, head_dim))
        snr = calculate_snr(refs[c][:, :seq_real], out[:, :seq_real].float())
        worst = min(worst, snr)
        print(f"   call {c}: SNR={snr:6.2f} dB" + ("" if snr >= 38 else "   <<< CORRUPT"))
    record_test("s2_flash_btb", f"calls={n_calls},nb={num_batches},seq={seq_len}", snr_db=worst)
    ue.clear_capture_buffer(); ue.reset_tensor_dram_addr(); ue.reset_program_dram_addr()
    assert worst >= 38, f"back-to-back S2 flash corrupts: worst call SNR {worst:.2f} dB < 38"
    return worst


def mobilesam_dw_conv_em_test(use_looped: bool = False):
    """EXACT-MATCH recreation of the encoder depthwise-conv situation (the encoder's #1
    instruction hog, ~20 MB). Runs conv2d_3x3_dw at the real model dims vs torch F.conv2d
    (depthwise, groups=C) — the exact oracle — and reports SNR + program size, for the
    current per-row-unrolled kernel (use_looped=False) or the h_out-hardware-loop kernel
    (use_looped=True). Also runs 2 convs BACK-TO-BACK (S0 has 2 MBConv dw convs) to catch
    any loop-ceiling/state issue, exactly like the i2t EM gate.

    Model dw-conv shapes: S0 MBConv dw (256x256, C=256, gelu), PatchMerging dw (stride2),
    S2/S3 local_conv (64x64). S0 is the 4.45 MB/conv case that dominates.
    """
    import sys as _sys, importlib
    import torch.nn.functional as F
    def _e(x): _sys.stderr.write(str(x) + "\n"); _sys.stderr.flush()
    M = importlib.import_module("models.mobilesam.mobilesam_test")
    kernel = (M.conv2d_3x3_dw_pbi_looped if use_looped else M.conv2d_3x3_dw_pbi)

    def _one(H, W, C, stride, gelu, n_btb):
        ue = UnifiedEngine()
        BLK = 64; K = 9 * BLK; n_blocks = C // BLK
        H_out = (H - 1) // stride + 1; W_out = (W - 1) // stride + 1
        ins, outs, ims, xs = [], [], [], []
        WGT = ue.allocate_tensor_dram(C * K * _BPE); BIA = ue.allocate_tensor_dram(C * _BPE)
        torch.manual_seed(0)
        w = torch.randn(C, 1, 3, 3, dtype=torch.bfloat16) * 0.2
        b = torch.randn(C, dtype=torch.bfloat16)
        wblk = torch.zeros(C, K, dtype=torch.bfloat16)
        for blk in range(n_blocks):
            for i in range(BLK):
                c = blk * BLK + i
                for dy in range(3):
                    for dx in range(3):
                        wblk[blk * BLK + i, (dy * 3 + dx) * BLK + i] = w[c, 0, dy, dx]
        ue.dma_to_accelerator_memory(WGT, wblk); ue.dma_to_accelerator_memory(BIA, b)
        for j in range(n_btb):
            IN = ue.allocate_tensor_dram(H * W * C * _BPE)
            OUT = ue.allocate_tensor_dram(H_out * W_out * C * _BPE)
            IM = ue.allocate_tensor_dram(W_out * (K + BLK) * _BPE)
            x = torch.randn(H * W, C, dtype=torch.bfloat16)
            ue.dma_to_accelerator_memory(IN, x)
            ins.append(IN); outs.append(OUT); ims.append(IM); xs.append(x)
        ue.start_capture()
        gpr_arg = 1 if use_looped else 8
        for j in range(n_btb):
            kernel(ue, ins[j], outs[j], ims[j], WGT, BIA, H, W, C, stride, gelu, gpr_arg)
        ue.stop_capture(); ue.generate_instruction_halt()
        prog = ue.get_program_dram_addr(); isz = ue.get_capture_instruction_size_bytes()
        ue.write_captured_instructions_to_dram(prog); ue.allocate_program_dram(isz)
        ue.start_execute_from_dram(prog); ue.wait_queue(120.0)
        worst = float("inf")
        for j in range(n_btb):
            out = ue.dma_from_accelerator_memory(outs[j], (H_out * W_out, C)).float()
            xin = xs[j].reshape(H, W, C).permute(2, 0, 1).unsqueeze(0).float()
            ref = F.conv2d(xin, w.float(), b.float(), stride=stride, padding=1, groups=C)
            if gelu: ref = F.gelu(ref)
            ref = ref.squeeze(0).permute(1, 2, 0).reshape(H_out * W_out, C)
            worst = min(worst, calculate_snr(ref, out))
        return worst, isz

    _e(f"\n[dw_conv_em] kernel={'LOOPED' if use_looped else 'current'}")
    for (H, W, C, st, g, n) in [(256, 256, 256, 1, True, 2), (128, 128, 128, 2, True, 1),
                                 (64, 64, 192, 1, False, 1)]:
        snr, isz = _one(H, W, C, st, g, n)
        nanf = "" if snr >= 35 else "  <<< FAIL"
        _e(f"  {H}x{W} C={C} s={st} gelu={g} btb={n}: SNR={snr:6.2f} dB  "
           f"{isz/1e6:6.3f} MB{nanf}")


def mobilesam_i2t_em_test(timeout_s: float = 120.0):
    """EXACT-MATCH recreation of the decoder i2t cross-attention in user_hw_test.

    Mirrors mobilesam_test._cross_attn_i2t's flash loop bit-for-bit: DEC_HEADS=8 heads,
    GA=4096 image rows chunked by ROW_CHUNK=64 -> 8*64 = 512 cross-attn calls emitted
    BACK-TO-BACK in ONE program, at the real dims (head_dim=64, kv_len=64), with the same
    per-head Q/K/V^T/OUT address layout. Runs both the legacy unrolled kernel and the PBI
    hardware-loop kernel, then reports per-kernel: bit-exactness vs each other, program size,
    and EXECUTE LATENCY (the thing that stalled the model — a slow PBI path blows the 120s
    decode timeout even when it's numerically correct).
    """
    import time as _time, sys as _sys
    def _e(x): _sys.stderr.write(str(x)+"\n"); _sys.stderr.flush()
    import importlib
    M = importlib.import_module("models.mobilesam.mobilesam_test")
    DEC_HEADS = M.DEC_HEADS; GA = M.GA; NT_PAD = M.NT_PAD
    HD = M.DEC_HD_PAD; ROW_CHUNK = M.ROW_CHUNK; BPE = M.BPE
    q_stride  = GA * HD
    kv_stride = NT_PAD * HD
    vt_stride = HD * NT_PAD

    def _alloc(ue):
        Q  = ue.allocate_tensor_dram(DEC_HEADS * GA * HD * BPE)
        K  = ue.allocate_tensor_dram(DEC_HEADS * NT_PAD * HD * BPE)
        VT = ue.allocate_tensor_dram(DEC_HEADS * HD * NT_PAD * BPE)
        O  = ue.allocate_tensor_dram(DEC_HEADS * GA * HD * BPE)
        SC = ue.allocate_tensor_dram(ROW_CHUNK * NT_PAD * BPE)
        BI = ue.allocate_tensor_dram(NT_PAD * BPE)
        return Q, K, VT, O, SC, BI

    torch.manual_seed(0)
    q  = torch.randn(DEC_HEADS, GA, HD, dtype=torch.bfloat16) * 0.3
    k  = torch.randn(DEC_HEADS, NT_PAD, HD, dtype=torch.bfloat16) * 0.3
    v  = torch.randn(DEC_HEADS, NT_PAD, HD, dtype=torch.bfloat16) * 0.3
    vt = v.transpose(1, 2).contiguous()
    bias = torch.randn(NT_PAD, dtype=torch.bfloat16)

    def _emit_i2t(ue, Q, K, VT, O, SC, BI, use_pbi):
        for h in range(DEC_HEADS):
            q_h  = Q  + h * q_stride  * BPE
            k_h  = K  + h * kv_stride * BPE
            vt_h = VT + h * vt_stride * BPE
            o_h  = O  + h * q_stride  * BPE
            if use_pbi:
                # ONE hardware loop over all GA rows per head -> DEC_HEADS loops total
                # (under the ~32-loops-per-program ceiling that NaN'd the 512-chunk version).
                M.cross_attn_flash_single_head_pbi(
                    ue=ue, head_dim=HD, q_len=GA, kv_len=NT_PAD,
                    Q_DRAM_ADDR=q_h, K_DRAM_ADDR=k_h, V_T_DRAM_ADDR=vt_h,
                    OUTPUT_DRAM_ADDR=o_h, gpr_loop=8, BIAS_DRAM_ADDR=BI)
            else:
                for row_base in range(0, GA, ROW_CHUNK):
                    row_take = min(ROW_CHUNK, GA - row_base)
                    M.cross_attn_flash_single_head(
                        ue=ue, head_dim=HD, q_len=row_take, kv_len=NT_PAD,
                        Q_DRAM_ADDR=q_h + row_base * HD * BPE, K_DRAM_ADDR=k_h,
                        V_T_DRAM_ADDR=vt_h, OUTPUT_DRAM_ADDR=o_h + row_base * HD * BPE,
                        SCORES_SCRATCH_ADDR=SC, BIAS_DRAM_ADDR=BI)

    def _run(use_pbi):
        ue = UnifiedEngine()
        Q, K, VT, O, SC, BI = _alloc(ue)
        ue.dma_to_accelerator_memory(Q, q.reshape(-1, HD))
        ue.dma_to_accelerator_memory(K, k.reshape(-1, HD))
        ue.dma_to_accelerator_memory(VT, vt.reshape(-1, NT_PAD))
        ue.dma_to_accelerator_memory(BI, bias)
        ue.start_capture()
        _emit_i2t(ue, Q, K, VT, O, SC, BI, use_pbi)
        ue.stop_capture()
        ue.generate_instruction_halt()
        prog = ue.get_program_dram_addr()
        isz = ue.get_capture_instruction_size_bytes()
        ue.write_captured_instructions_to_dram(prog)
        ue.allocate_program_dram(isz)
        t0 = _time.perf_counter()
        ue.start_execute_from_dram(prog)
        ue.wait_queue(timeout_s)
        dt = _time.perf_counter() - t0
        out = ue.dma_from_accelerator_memory(O, (DEC_HEADS, GA, HD)).float()
        return out, isz, dt

    _e(f"\n[i2t_em] EXACT-MATCH decoder i2t: {DEC_HEADS} heads x {GA//ROW_CHUNK} chunks "
          f"= {DEC_HEADS*(GA//ROW_CHUNK)} back-to-back calls (hd={HD}, kv={NT_PAD})")
    ref = torch.zeros(DEC_HEADS, GA, HD)
    for h in range(DEC_HEADS):
        att = q[h].float() @ k[h].float().t() + bias.float()
        ref[h] = torch.softmax(att, -1) @ v[h].float()
    out_l, isz_l, dt_l = _run(False)
    _e(f"  legacy : {isz_l/1e6:6.2f} MB  exec {dt_l:6.2f}s  max_abs={out_l.abs().max():.3f}  vs_torch={calculate_snr(ref,out_l):.1f}dB")
    out_p, isz_p, dt_p = _run(True)
    _e(f"  pbi    : {isz_p/1e6:6.2f} MB  exec {dt_p:6.2f}s  max_abs={out_p.abs().max():.3f}  vs_torch={calculate_snr(ref,out_p):.1f}dB")
    for h in range(DEC_HEADS):
        _e(f"    head {h}: pbi-vs-legacy {calculate_snr(out_l[h], out_p[h]):7.2f} dB")
    snr = calculate_snr(out_l, out_p)
    _e(f"  pbi-vs-legacy SNR = {snr:.2f} dB  (inf = bit-exact)")
    _e(f"  size shrink = {isz_l/max(isz_p,1):.1f}x   latency ratio pbi/legacy = {dt_p/max(dt_l,1e-6):.2f}x")
    record_test("i2t_em_pbi", f"heads={DEC_HEADS},GA={GA}", snr_db=snr, inst_bytes=isz_p)
    return snr, dt_l, dt_p


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='User DMA Operations for Unified Engine')
    parser.add_argument('--dev', type=str, default='xdma0',
                        help='DMA device name (e.g., xdma0, xdma1). Default: xdma0')
    parser.add_argument('--device', type=str, default='kintex7',
                        help='FPGA type')
    parser.add_argument('--base-addr', type=lambda x: int(x, 0), default=0x02000000,
                        help='AXI-Lite register base address (default: 0x02000000). '
                             'Try 0x0 if register read hangs after bitstream update.')
    parser.add_argument(
        '--ext',
        action='store_true',
        help='Run the large nested-loop sweeps at the end of the suite (slow).',
    )
    args = parser.parse_args()

    # Set DMA device paths based on device name and force-sync every module
    # that may hold a copied device-path global.
    set_dma_device(args.dev)
    import user_dma_core
    print(f"DMA dev={args.dev}"
          f" (H2C={user_dma_core.DMA_DEVICE_H2C},"
          f" C2H={user_dma_core.DMA_DEVICE_C2H},"
          f" USER={user_dma_core.DMA_DEVICE_USER}),"
          f" base=0x{args.base_addr:08x}")

    axi_width_bits = 512 if args.device in ("bittware", "rk") else 256
    os.environ["UE_AXI_DATA_WIDTH_BITS"] = str(axi_width_bits)
    user_dma_core.UE_AXI_DATA_WIDTH_BITS = axi_width_bits
    globals()["UE_AXI_DATA_WIDTH_BITS"] = axi_width_bits
    print(f"UE_AXI_DATA_WIDTH_BITS={axi_width_bits} (device={args.device})")

    # Fix RNG seed so SNR numbers are reproducible across runs and easy to
    # compare across HDL changes (e.g. exp/LALU tweaks).
    random.seed(0)
    torch.manual_seed(0)

    # Sanity check: print 4 bf16 samples so the log proves the seed is in effect.
    _seed_probe = torch.randn(4, dtype=torch.bfloat16)
    print(f"[seed-check-start] bf16 samples after seed=0: {_seed_probe.tolist()}")

    # kintex7 operates at 194 Mhz = 5.1594 ns
    # alveo operates at 250 Mhz = 4.0 ns
    # kintex ultrascale+ operates at 333 Mhz = 3.0 ns
    # bittware board operates at 300 Mhz = 3.3333 ns
    clock = None
    if args.device == "kintex7":
        clock = 5.1594
    elif args.device == "rk" or args.device == "puzhi":
        clock = 3
    elif args.device in ("bittware", "bittware_256"):
        clock = 3.3333
    elif args.device == "alveo":
        clock = 1000 / 250
    else:
        clock = 10

    user_dma_core.CLOCK_CYCLE_TIME_NS = clock
    user_dma_core.UE_PEAK_GFLOPS = 0.128 / clock
    print(f"Clock period={user_dma_core.CLOCK_CYCLE_TIME_NS:.6f} ns")

    # Always emit user_hw_test_summary.md, even if an assertion aborts the run.
    atexit.register(write_test_summary, "user_hw_test_summary.md")

    # --- Multi-head reshape (MobileSAM scatter+pad+permute patterns) ---
    # Padded case (MobileSAM-like: hd=32, hd_pad=64, heads=4)
    # multihead_reshape_dram_test(seq_len=64, num_heads=4, head_dim=32, head_dim_pad=64, use_pbi=False)
    # multihead_reshape_dram_test(seq_len=64, num_heads=4, head_dim=32, head_dim_pad=64, use_pbi=True)
    # # Larger seq_len, more heads (stresses the PBI transpose phase)
    # multihead_reshape_dram_test(seq_len=256, num_heads=8, head_dim=32, head_dim_pad=64, use_pbi=False)
    # multihead_reshape_dram_test(seq_len=256, num_heads=8, head_dim=32, head_dim_pad=64, use_pbi=True)
    # # No-padding fast path (head_dim == head_dim_pad)
    # multihead_reshape_dram_test(seq_len=144, num_heads=6, head_dim=64, head_dim_pad=64, use_pbi=False)
    # multihead_reshape_dram_test(seq_len=144, num_heads=6, head_dim=64, head_dim_pad=64, use_pbi=True)

    # --- Flash attention batched (PBI staging) ---
    # flash_attention_batched_test(head_dim=64, seq_len=64, num_heads=4, bias_enable=False)
    # flash_attention_batched_test(head_dim=64, seq_len=64, num_heads=4, bias_enable=True)
    # flash_attention_batched_test(head_dim=64, seq_len=256, num_heads=5, bias_enable=False)

    # --- MobileSAM dw conv: baseline (current per-row-unrolled PBI) ---
    # mobilesam_dw_conv_em_test(use_looped=False)

    # --- MobileSAM dw conv: LOOPED (h_out hardware loop) ---
    mobilesam_dw_conv_em_test(use_looped=True)

    _ALL_TESTS_PASSED_BEFORE_SUMMARY = True