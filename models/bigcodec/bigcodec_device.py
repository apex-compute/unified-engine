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
