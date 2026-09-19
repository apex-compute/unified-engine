"""Factor-two alias filters using native matrix dots and compact device loops.

Tensors retain the [time, pad64(channels)] BF16 layout. Split coefficients store
BF16 high values followed by their BF16 residuals in one dot. The native BF19
reduction tree/BF20 accumulator rounds once to BF16 at each filter output; this
is not an FP32 filter. Callers own the shared identity matrix and START/HALT.
"""
from __future__ import annotations

from dataclasses import dataclass
import operator

import torch

from bigcodec_device import channels, gather_rows, shared, udc


@dataclass
class FIRPlan:
    kind: str
    coefficient_precision: str
    input_rows: int
    output_rows: int
    width: int
    source: int
    destination: int
    scratch: int
    scratch_bytes: int
    identity: int
    weight: int
    base_k: int
    dot_k: int
    buffers: dict
    coefficient_error: dict


def _geometry(input_shape, kind, coefficient_precision):
    if kind not in ('up', 'down'):
        raise ValueError("FIR kind must be 'up' or 'down'")
    if coefficient_precision not in ('dc', 'bf16', 'split'):
        raise ValueError("FIR coefficient precision must be 'dc', 'bf16' or 'split'")
    try:
        shape = tuple(input_shape)
        if len(shape) != 2 or any(isinstance(value, bool) for value in shape):
            raise ValueError
        rows, logical = map(operator.index, shape)
    except (TypeError, ValueError):
        raise ValueError('FIR input shape must contain two positive integers') from None
    if rows < 1 or logical < 1:
        raise ValueError('FIR input shape must contain two positive integers')
    width = channels(logical)
    if 64 * width * 2 > udc.URAM_NEAR_FULL_SIZE:
        raise ValueError('A 64-row FIR output tile exceeds SRAM capacity')
    base_k = 64 if kind == 'up' else 192
    dot_k = base_k * (2 if coefficient_precision == 'split' else 1)
    output_rows = 2 * rows if kind == 'up' else (rows + 1) // 2
    return rows, width, output_rows, base_k, dot_k


def scratch_bytes(input_shape, *, kind='up', coefficient_precision='split'):
    """Bytes for source staging, transpose, optional duplication and output."""
    _, width, _, base_k, dot_k = _geometry(input_shape, kind, coefficient_precision)
    duplicated = dot_k if coefficient_precision == 'split' else 0
    # Preserve the validated layout's two 64-row output slots. The direct
    # emitter uses the last slot for partial tiles.
    return (2 * base_k + duplicated + 128) * width * 2


def _validate_regions(regions):
    for name, address, size in regions:
        if (isinstance(address, bool) or not isinstance(address, int)
                or address < 0 or address % 128 or address + size > 1 << 35):
            raise ValueError(f'{name} must be aligned and fit a 32-bit word address')
    for index, (name, address, size) in enumerate(regions):
        for other_name, other, length in regions[index + 1:]:
            if address < other + length and other < address + size:
                raise ValueError(f'FIR regions overlap: {name} and {other_name}')


def prepare_fir(taps, image, *, input_shape, source, destination, scratch,
                identity, kind='up', coefficient_precision='split'):
    """Pack twelve FP32 taps; identity must address a BF16 64-by-64 eye matrix.

    ``split`` preserves the original taps as high/residual BF16 values. ``bf16``
    stores only the high values. ``dc`` retains the existing symmetric unit-DC
    coefficient control. Preparation never modifies the supplied taps.
    """
    rows, width, output_rows, base_k, dot_k = _geometry(input_shape, kind, coefficient_precision)
    workspace = scratch_bytes(input_shape, kind=kind, coefficient_precision=coefficient_precision)
    regions = [('source', source, rows * width * 2),
               ('destination', destination, output_rows * width * 2),
               ('scratch', scratch, workspace), ('identity', identity, 64 * 64 * 2)]
    _validate_regions(regions)
    original = torch.as_tensor(taps).detach().cpu().float().flatten()
    if original.numel() != 12 or not bool(torch.isfinite(original).all()):
        raise ValueError('FIR requires twelve finite FP32 taps')
    if coefficient_precision == 'dc':
        # The existing activation module may import this helper; defer the
        # control-only dependency until both modules have initialized.
        from bigcodec_activation import quantize_filter_dc
        high = torch.tensor(quantize_filter_dc(tuple(original.tolist())))
    else:
        high = original.bfloat16().float()
    low = ((original - high).bfloat16().float() if coefficient_precision == 'split'
           else torch.zeros_like(high))
    matrix = torch.zeros(64, dot_k, dtype=torch.bfloat16)
    for output in range(64):
        if kind == 'up':
            phase = output % 2
            terms = [(output // 2 + (phase + 5 - tap) // 2 + 3, tap, 2.)
                     for tap in range(12) if (phase + 15 - tap) % 2 == 0]
        else:
            terms = [(2 * output + tap, tap, 1.) for tap in range(12)]
        for column, tap, factor in terms:
            matrix[output, column] = factor * high[tap]
            if coefficient_precision == 'split':
                matrix[output, base_k + column] = factor * low[tap]
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError('FIR taps overflow packed BF16 coefficients')
    buffers, cursor = {}, scratch
    for name, elements in (('staged', base_k * width), ('transposed', base_k * width),
                          ('duplicated', dot_k * width if coefficient_precision == 'split' else 0),
                          ('filtered', 64 * width), ('output', 64 * width)):
        buffers[name] = cursor
        cursor += elements * 2
    weight = image.allocate(matrix, alignment=128)
    _validate_regions([*regions, ('weight', weight, matrix.numel() * 2)])
    error = high + low - original
    return FIRPlan(kind, coefficient_precision, rows, output_rows, width, source, destination,
                   scratch, workspace, identity, weight, base_k, dot_k, buffers,
                   dict(max_abs=float(error.abs().max()), l2=float(error.norm()),
                        sum=float((high + low).double().sum())))


def _transpose_and_duplicate(engine, plan):
    b, c, k = plan.buffers, plan.width, plan.base_k
    engine.bf16_transpose_core(M=k, N=c, INPUT_DRAM_ADDR=b['staged'],
        OUTPUT_DRAM_ADDR=b['transposed'], IDENTITY_DRAM_ADDR=plan.identity)
    if plan.coefficient_precision != 'split':
        return b['transposed']
    chunk_rows = max(1, udc.URAM_NEAR_FULL_SIZE // (k * 2))
    for channel in range(0, c, chunk_rows):
        count = min(chunk_rows, c - channel)
        engine.accelerator_memory_to_sram(b['transposed'] + channel * k * 2, 0, count * k)
        for half in (0, 1):
            shared._copy_contiguous_or_strided_write(engine, sram=0,
                destination=b['duplicated'] + (channel * 2 * k + half * k) * 2,
                total=count * k * 2, chunk=k * 2, jump=2 * k * 2)
    return b['duplicated']


def _emit_tile(engine, plan, first):
    b, c, k = plan.buffers, plan.width, plan.base_k
    take = min(64, plan.output_rows - first)
    start = first // 2 - 3 if plan.kind == 'up' else first * 2 - 5
    gather_rows(engine, plan.source, b['staged'], source_rows=plan.input_rows,
                rows=k, width=c, start=start)
    dot_input = _transpose_and_duplicate(engine, plan)
    output = plan.destination + first * c * 2 if take == 64 else b['output']
    engine.matmat_mul_core(M=64, K=plan.dot_k, N=c,
        A_DRAM_ADDR=plan.weight, B_DRAM_ADDR=dot_input, OUTPUT_DRAM_ADDR=output)
    if take != 64:
        shared._copy_contiguous_or_strided_read(engine, source=output, sram=0,
            total=take * c * 2, chunk=c * 2, jump=c * 2)
        engine.sram_to_accelerator_memory(0, plan.destination + first * c * 2, take * c)


def _interior_tiles(plan):
    """Complete output tiles with an entirely in-bounds staged source span."""
    last_full = (plan.output_rows // 64 - 1) * 64
    last_source = (2 * (plan.input_rows - plan.base_k + 3) if plan.kind == 'up'
                   else (plan.input_rows - plan.base_k + 5) // 2)
    last = min(last_full, last_source) // 64 * 64
    # Tile zero always contains the left halo. All subsequent eligible tiles
    # form one interval; range avoids Python work proportional to clip length.
    return range(64, max(64, last + 64), 64)


def _emit_interior(engine, plan, *, source_reg, output_reg, rows_reg, chunk_reg):
    b, c, k = plan.buffers, plan.width, plan.base_k
    block = max(1, udc.URAM_NEAR_FULL_SIZE // (c * 2))
    for first in range(0, k, block):
        count = min(block, k - first)
        engine.generate_instruction_add_imm(src_reg_idx=source_reg,
            immediate_value=(first * c * 2) >> 3, dst_reg_idx=chunk_reg)
        engine.accelerator_memory_to_sram(0, 0, count * c, general_reg_src=chunk_reg)
        engine.sram_to_accelerator_memory(0, b['staged'] + first * c * 2, count * c)
    dot_input = _transpose_and_duplicate(engine, plan)
    engine.matmat_mul_core(M=64, K=plan.dot_k, N=c,
        A_DRAM_ADDR=plan.weight, B_DRAM_ADDR=dot_input, OUTPUT_DRAM_ADDR=plan.destination,
        gpr_M_reg=rows_reg, gpr_out_addr=output_reg)


def emit_fir(engine, plan):
    """Emit direct border tiles and a counted loop over repeated interior tiles."""
    interior = _interior_tiles(plan)
    if len(interior) < 2:
        for first in range(0, plan.output_rows, 64):
            _emit_tile(engine, plan, first)
        return
    first, end = interior[0], interior[-1] + 64
    for border in range(0, first, 64):
        _emit_tile(engine, plan, border)
    registers = [engine.alloc_isa_reg() for _ in range(4)]
    source_reg, output_reg, rows_reg, chunk_reg = registers
    try:
        start = first // 2 - 3 if plan.kind == 'up' else first * 2 - 5
        engine.generate_instruction_add_set(source_reg, (plan.source + start * plan.width * 2) >> 3)
        engine.generate_instruction_add_set(output_reg, (plan.destination + first * plan.width * 2) >> 3)
        engine.generate_instruction_add_set(rows_reg, 64)
        # Refetch the body from DRAM: nested transpose/matmul programs exceed
        # the relative-jump cache window and contain their own jump anchors.
        engine.loop_start(loop_cnt=len(interior), relative=False)
        _emit_interior(engine, plan, source_reg=source_reg, output_reg=output_reg,
                       rows_reg=rows_reg, chunk_reg=chunk_reg)
        input_step = 32 if plan.kind == 'up' else 128
        engine.generate_instruction_add_imm(source_reg, (input_step * plan.width * 2) >> 3)
        engine.generate_instruction_add_imm(output_reg, (64 * plan.width * 2) >> 3)
        engine.loop_end()
    finally:
        for _ in registers:
            engine.release_isa_reg()
    for border in range(end, plan.output_rows, 64):
        _emit_tile(engine, plan, border)
