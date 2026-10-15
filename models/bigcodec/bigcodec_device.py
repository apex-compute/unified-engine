"""Offline instruction helpers for packed time/channel BigCodec tensors."""
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / 'models/yolov5s', ROOT / 'models/dpdfnet'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import user_dma_core as udc
import yolov5_precompiled as shared


def aligned(value, alignment=128):
    return (int(value) + alignment - 1) // alignment * alignment


def channels(value):
    return aligned(value, 64)


def copy(engine, source, destination, size):
    if source == destination:
        return
    for offset in range(0, size, udc.URAM_NEAR_FULL_SIZE):
        take = min(size - offset, udc.URAM_NEAR_FULL_SIZE)
        engine.accelerator_memory_to_sram(source + offset, 0, 0, memcpy_length_bytes=take)
        engine.sram_to_accelerator_memory(0, destination + offset, 0, memcpy_length_bytes=take)


def zero(engine, destination, size, zero_address):
    for offset in range(0, size, udc.URAM_NEAR_FULL_SIZE):
        copy(engine, zero_address, destination + offset, min(size - offset, udc.URAM_NEAR_FULL_SIZE))


def elementwise(engine, mode, source, other, destination, elements, scalar=None):
    assert elements > 0 and elements % 64 == 0
    if scalar is not None:
        # The driver encodes broadcast immediates by discarding low FP32 bits.
        # Supply an already rounded BF16 value so constants use the same RNE
        # precision contract as tensors instead of a systematic truncation.
        scalar = float(torch.tensor(scalar, dtype=torch.float32).bfloat16())
    engine.eltwise_core_dram(M=elements // 64, N=64, dram_a=source,
                           dram_b=other, dram_out=destination,
                           mode=mode, scalar=scalar)


def scale(engine, source, destination, elements, value):
    elementwise(engine, udc.UE_MODE.MUL_BROADCAST, source, None, destination, elements, value)


def shift(engine, source, destination, elements, value):
    elementwise(engine, udc.UE_MODE.ADD_BROADCAST, source, None, destination, elements, value)


def sram_scale(engine, source, destination, elements, value):
    """Wide SRAM broadcast with the same BF16 immediate rounding as DRAM ops."""
    engine.broadcast_mul(float(torch.tensor(value, dtype=torch.float32).bfloat16()),
                         source, destination, elements)


def sram_shift(engine, source, destination, elements, value):
    engine.broadcast_add(float(torch.tensor(value, dtype=torch.float32).bfloat16()),
                         source, destination, elements)


def sram_copy(engine, source, destination, elements):
    """Copy finite BF16 lanes from URAM_A using multiplication by exact one.

    Native 1x1 MAXPOOL shifts the source by one 64-lane row on RK 0x40519e0a.
    Broadcast multiplication preserves all normal values and signed zeros;
    exhaustive hardware tests found magnitudes below 2^-127 flush to signed
    zero. The upper half of the BF16 subnormal range remains exact.
    """
    size = elements * 2
    if (elements <= 0 or elements % 64 or elements // 64 > 0xFFF
            or any(address % 128 for address in (source, destination))
            or not 0 <= source < 0x80000
            or source + size > udc.URAM_NEAR_FULL_SIZE
            or not 0 <= destination < 0x100000
            or destination % 0x80000 + size > udc.URAM_NEAR_FULL_SIZE):
        raise ValueError('Invalid SRAM copy geometry')
    if source == destination:
        return
    if destination < source + size and source < destination + size:
        raise ValueError('SRAM copy ranges partially overlap')
    engine.broadcast_mul(1.0, source, destination, elements)


def sram_maximum(engine, left, right, destination, elements):
    """Compare two finite BF16 vectors using the native 64-lane MAXPOOL unit.

    Both inputs occupy disjoint, contiguous URAM_A ranges. Treat them as two
    image rows: a vertical 2x1 window compares corresponding lanes without
    interleaving or an identity matmul. Output may alias either input. NaNs
    follow MAXPOOL's filtering policy; callers must supply finite activations.
    """
    if left > right:
        left, right = right, left
    size = elements * 2
    if (elements <= 0 or elements % 64 or 2 * (elements // 64) > 0xFFF
            or any(address % 128 for address in (left, right, destination))
            or not 0 <= left < right < 0x80000
            or left + size > right or right + size > udc.URAM_NEAR_FULL_SIZE
            or not 0 <= destination < 0x100000
            or destination % 0x80000 + size > udc.URAM_NEAR_FULL_SIZE):
        raise ValueError('Invalid SRAM maximum geometry or overlapping inputs')
    for source in (left, right):
        if destination != source and destination < source + size and source < destination + size:
            raise ValueError('SRAM maximum output partially overlaps an input')
    engine.start_queue_for_maxpool2d_operation(
        act_sram_start_addr=left, output_sram_wb_addr=destination,
        kernel_w=1, kernel_h=2, out_w=elements // 64, out_h=1,
        w_pad=(right - left) // 128, stride_s=1)


def sram_clamp(engine, source, destination, elements, *, scratch_address,
               lo=-float('inf'), hi=float('inf')):
    """Finite-input clamp, preserving other SRAM and every BF16 value boundary.

    Source must be in URAM_A; destination can be in either bank or equal source.
    Workspace is two contiguous vectors in URAM_A, disjoint from input/output.
    MAXPOOL selects BF16 values directly. Upper bounds also use arithmetic
    negation; subnormal flushing therefore depends on the active datapath and
    needs hardware verification. No polynomial approximation is introduced.
    """
    size = elements * 2
    temporary, constant = scratch_address, scratch_address + size
    if (lo != lo or hi != hi or lo > hi or elements <= 0 or elements % 64
            or any(address % 128 for address in (source, destination, scratch_address))
            or not 0 <= source < 0x80000
            or source + size > udc.URAM_NEAR_FULL_SIZE
            or not 0 <= scratch_address < 0x80000
            or scratch_address + 2 * size > udc.URAM_NEAR_FULL_SIZE):
        raise ValueError('Invalid SRAM clamp geometry or bounds')
    for address in (source, destination):
        if scratch_address < address + size and address < scratch_address + 2 * size:
            raise ValueError('SRAM clamp workspace overlaps input/output')
    sram_scale(engine, source, constant, elements, 0)
    current = source
    if lo != -float('inf'):
        if lo != 0:
            sram_shift(engine, constant, constant, elements, lo)
        sram_maximum(engine, source, constant,
                     destination if hi == float('inf') else temporary, elements)
        current = destination if hi == float('inf') else temporary
    if hi != float('inf'):
        sram_scale(engine, current, temporary, elements, -1)
        sram_scale(engine, source, constant, elements, 0)
        if hi != 0:
            sram_shift(engine, constant, constant, elements, -hi)
        sram_maximum(engine, temporary, constant, temporary, elements)
        sram_scale(engine, temporary, destination, elements, -1)
    elif current == source and source != destination:
        sram_copy(engine, source, destination, elements)


def gather_rows_to_sram(engine, source, destination, *, source_rows, rows,
                        width, start=0, stride=1):
    """Gather an endpoint-padded time tile directly into a bounded SRAM span."""
    row_bytes = width * 2
    if (source_rows <= 0 or rows <= 0 or width <= 0 or width % 64 or stride <= 0
            or destination % 128 or destination % 0x80000 + rows * row_bytes > udc.URAM_NEAR_FULL_SIZE):
        raise ValueError('Invalid SRAM row gather geometry')
    first = min(rows, max(0, (-start + stride - 1) // stride))
    stop = max(first, min(rows, (source_rows - 1 - start) // stride + 1))
    for index in range(first):
        engine.accelerator_memory_to_sram(source, destination + index * row_bytes,
                                          0, memcpy_length_bytes=row_bytes)
    if stop > first:
        shared._copy_contiguous_or_strided_read(
            engine, source=source + (start + first * stride) * row_bytes,
            sram=destination + first * row_bytes, total=(stop - first) * row_bytes,
            chunk=row_bytes, jump=stride * row_bytes)
    for index in range(stop, rows):
        engine.accelerator_memory_to_sram(source + (source_rows - 1) * row_bytes,
                                          destination + index * row_bytes, 0,
                                          memcpy_length_bytes=row_bytes)


def clamp_wide(engine, source, destination, elements, lo=0.0, hi=float('inf')):
    """DRAM wrapper for the finite-input, native wide SRAM clamp."""
    if elements <= 0 or elements % 64:
        raise ValueError('Wide clamp requires a positive multiple of 64 elements')
    for first in range(0, elements, 32768):
        count = min(32768, elements - first)
        engine.accelerator_memory_to_sram(source + first * 2, 0, count)
        sram_clamp(engine, 0, 0, count, scratch_address=0x20000, lo=lo, hi=hi)
        engine.sram_to_accelerator_memory(0, destination + first * 2, count)


def clamp(engine, source, destination, elements, identity_address, lo=0.0, hi=float('inf')):
    assert elements > 0 and elements % 64 == 0
    rows = elements // 64
    register = None
    # The static kernel emits one dot instruction per row. Above this cutoff
    # the existing dynamic-M kernel captures112 instructions independent of
    # utterance length, while preserving the same identity matmul and clamp.
    if rows >= 128:
        register = engine.alloc_isa_reg()
        engine.generate_instruction_add_set(register, rows)
    try:
        engine.matmat_mul_core(M=rows, K=64, N=64,
                              A_DRAM_ADDR=source, B_DRAM_ADDR=identity_address,
                              OUTPUT_DRAM_ADDR=destination,
                              clamp_enable=True, clamp_min=lo, clamp_max=hi,
                              gpr_M_reg=register)
    finally:
        if register is not None:
            engine.release_isa_reg()


def channel_mul(engine, source, destination, rows, width, constant_address, constant_rows):
    for first in range(0, rows, constant_rows):
        count = min(constant_rows, rows - first)
        elementwise(engine, udc.UE_MODE.ELTWISE_MUL,
                    source + first * width * 2, constant_address,
                    destination + first * width * 2, count * width)


def gather_rows(engine, source, destination, *, source_rows, rows, width, start=0, stride=1):
    """Static strided row gather, replicating source endpoints outside bounds."""
    assert source_rows > 0 and rows > 0 and stride > 0 and width % 64 == 0
    row_bytes = width * 2
    first = min(rows, max(0, (-start + stride - 1) // stride))
    stop = max(first, min(rows, (source_rows - 1 - start) // stride + 1))
    for index in range(first):
        copy(engine, source, destination + index * row_bytes, row_bytes)
    block = max(1, udc.URAM_NEAR_FULL_SIZE // row_bytes)
    for index in range(first, stop, block):
        count = min(block, stop - index)
        shared._copy_contiguous_or_strided_read(
            engine, source=source + (start + index * stride) * row_bytes,
            sram=0, total=count * row_bytes, chunk=row_bytes, jump=stride * row_bytes)
        engine.sram_to_accelerator_memory(0, destination + index * row_bytes, 0,
                                          memcpy_length_bytes=count * row_bytes)
    for index in range(stop, rows):
        copy(engine, source + (source_rows - 1) * row_bytes,
             destination + index * row_bytes, row_bytes)


def scatter_rows(engine, source, destination, *, rows, width, start=0, stride=1):
    row_bytes = width * 2
    block = max(1, udc.URAM_NEAR_FULL_SIZE // row_bytes)
    for index in range(0, rows, block):
        count = min(block, rows - index)
        engine.accelerator_memory_to_sram(source + index * row_bytes, 0, 0,
                                          memcpy_length_bytes=count * row_bytes)
        shared._copy_contiguous_or_strided_write(
            engine, sram=0, destination=destination + (start + index * stride) * row_bytes,
            total=count * row_bytes, chunk=row_bytes, jump=stride * row_bytes)
