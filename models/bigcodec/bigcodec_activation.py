"""Alias-free SnakeBeta lowering with an explicit BF16 sine approximation.

The filters and endpoint handling follow upstream. Sine-squared is evaluated
on a folded interval with the legacy degree-ten polynomial. This backend is approximate:
its argument is clamped to +/-32*pi before folding, and FPGA accuracy must be
measured against the official reference before deployment.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import math
from numbers import Integral
import torch

from bigcodec_device import (channels, channel_mul, clamp, copy, elementwise,
                             gather_rows, gather_rows_to_sram, scale, scatter_rows,
                             shift, sram_copy, sram_maximum, sram_scale, sram_shift,
                             shared, udc)

SNAKE_ARGUMENT_LIMIT = 32 * math.pi
ACTIVATION_TILE_ELEMENTS = 32768


def _snake_coefficients(polynomial):
    """Retain only the validated legacy Snake approximation."""
    if polynomial != 'legacy':
        raise ValueError("snake_polynomial must be 'legacy'")
    return (1., -1 / 3, 2 / 45, -1 / 315, 2 / 14175)


def _ordered_filter_taps(taps, accumulation):
    """Preserve tap offsets while selecting the BF16 addition order.

    Sorting is stable for equal absolute coefficients. It changes rounding of
    partial sums, without changing the FIR equation or adding device operations.
    """
    if accumulation == 'serial':
        return list(taps)
    if accumulation == 'sorted':
        return sorted(taps, key=lambda tap: abs(tap[1]))
    raise ValueError("filter_accumulation must be 'serial' or 'sorted'")


def sine_squared(value, *, bf16=False, snake_polynomial='legacy'):
    """Torch reference with direct BF16 rounding at each storage boundary.

    The native ALU additionally rounds through BF19; its arithmetic is tested
    separately. Keep this reference's legacy rounding behavior unchanged.
    """
    coefficients = _snake_coefficients(snake_polynomial)
    q = lambda x: x.to(torch.bfloat16).float() if bf16 else x
    scalar = lambda x: float(torch.tensor(x).bfloat16()) if bf16 else x
    x = q(value).abs().clamp(max=SNAKE_ARGUMENT_LIMIT)
    x = q(x)
    for exponent in range(5, -1, -1):
        center = scalar((math.pi / 2) * 2 ** exponent)
        delta = q(x - center).clamp_min(0)
        x = q(x - q(2 * delta))
    z = q(x * x)
    result = q(z * scalar(coefficients[-1]))
    for coefficient in reversed(coefficients[:-1]):
        result = q(result + scalar(coefficient))
        result = q(result * z)
    return result.clamp(0, 1)


@lru_cache(maxsize=8)
def quantize_filter_dc(coefficients):
    """Nearest symmetric BF16 filter with exact unit DC within two ULPs.

    The official12-tap low-pass filter has six unique coefficients. Search
    their neighboring BF16 values subject to an exact half-sum of0.5, then
    mirror them. This preserves symmetry and both factor-two phase gains
    without multiplying the filter output by a fitted correction factor.
    """
    values = tuple(float(x) for x in coefficients)
    if (len(values) != 12 or any(not math.isfinite(x) for x in values)
            or values != values[::-1] or abs(math.fsum(values) - 1) > 1e-6):
        raise ValueError("Expected a finite symmetric unit-DC12-tap filter")
    candidates = []
    for value in values[:6]:
        bits = int(torch.tensor(value).bfloat16().view(torch.uint16))
        neighbors = torch.tensor([bits + step for step in range(-2, 3)],
                                 dtype=torch.uint16).view(torch.bfloat16).float()
        candidates.append(tuple(float(x) for x in neighbors))
    first = {}
    for chosen in product(*candidates[:3]):
        total = math.fsum(chosen)
        error = math.fsum((x - y) ** 2 for x, y in zip(chosen, values[:3]))
        item = (error, chosen)
        if total not in first or item < first[total]:
            first[total] = item
    best = None
    for chosen in product(*candidates[3:]):
        match = first.get(0.5 - math.fsum(chosen))
        if match is None:
            continue
        error = match[0] + math.fsum((x - y) ** 2 for x, y in zip(chosen, values[3:6]))
        item = (error, match[1] + chosen)
        if best is None or item < best:
            best = item
    if best is None:
        raise ValueError("No unit-DC BF16 filter exists within two ULPs of RNE")
    result = best[1] + best[1][::-1]
    assert math.fsum(result) == 1.0
    return result


def _activation_geometry(input_shape):
    try:
        rows, logical_channels = input_shape
    except (TypeError, ValueError):
        raise ValueError('Activation shape must contain two positive integer dimensions') from None
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 1
           for value in (rows, logical_channels)):
        raise ValueError('Activation shape must contain positive integer dimensions')
    rows, logical_channels = int(rows), int(logical_channels)
    width = channels(logical_channels)
    if width > ACTIVATION_TILE_ELEMENTS:
        raise ValueError('Activation channel width exceeds the SRAM tile budget')
    return rows, logical_channels, width


def _filter_options(accumulation, stage):
    if accumulation not in ('serial', 'sorted', 'matrix'):
        raise ValueError("filter_accumulation must be 'serial', 'sorted' or 'matrix'")
    if stage not in ('up', 'down', 'both'):
        raise ValueError("filter_stage must be 'up', 'down' or 'both'")


def activation_scratch_bytes(input_shape, *, filter_accumulation='serial',
                             filter_stage='both'):
    """Exact scratch requirement; stage selects matrix FIRs only.

    Legacy serial/sorted retain their original nine-row-tensor allocation.
    Matrix mode retains one complete upsampled/Snake tensor and shares one
    workspace between its selected FIRs; the other FIR uses legacy SRAM math.
    """
    _filter_options(filter_accumulation, filter_stage)
    rows, logical_channels, width = _activation_geometry(input_shape)
    if filter_accumulation != 'matrix':
        return 9 * rows * width * 2
    from bigcodec_filter import scratch_bytes as filter_scratch_bytes
    work = []
    if filter_stage in ('up', 'both'):
        work.append(filter_scratch_bytes((rows, logical_channels), kind='up'))
    if filter_stage in ('down', 'both'):
        work.append(filter_scratch_bytes((2 * rows, logical_channels), kind='down'))
    return 2 * rows * width * 2 + max(work)


@dataclass
class ActivationPlan:
    rows: int
    width: int
    source: int
    destination: int
    scratch: int
    identity: int
    alpha: int
    inverse_beta: int
    constant_rows: int
    up_filter: tuple
    down_filter: tuple
    filter_accumulation: str = 'serial'
    snake_polynomial: str = 'legacy'
    filter_stage: str = 'both'
    up_fir: object | None = None
    down_fir: object | None = None


def prepare_activation(module, image, *, input_shape, input_address,
                       output_address, scratch_address, identity_address,
                       filter_accumulation='serial', filter_stage='both',
                       workspace_bytes=None, snake_polynomial='legacy'):
    """Pack legacy Snake and selectable FIR arithmetic.

    Serial/sorted preserve their existing coefficients and addition order.
    Matrix mode packs original FP32 taps as BF16 high/residual coefficients,
    accumulating a complete FIR in one native dot. The unselected FIR stays
    exactly legacy serial. ``workspace_bytes`` bounds the caller's allocation;
    omit it when allocating exactly ``activation_scratch_bytes(...)``.
    The legacy Snake keyword remains for diagnostic-helper compatibility.
    """
    _filter_options(filter_accumulation, filter_stage)
    _snake_coefficients(snake_polynomial)
    rows, logical_channels, width = _activation_geometry(input_shape)
    needed = activation_scratch_bytes(input_shape,
        filter_accumulation=filter_accumulation, filter_stage=filter_stage)
    if workspace_bytes is not None and (isinstance(workspace_bytes, bool)
            or not isinstance(workspace_bytes, Integral) or workspace_bytes < needed):
        raise ValueError(f'Activation needs {needed} scratch bytes; available {workspace_bytes}')
    for address in (input_address, output_address, scratch_address, identity_address):
        if isinstance(address, bool) or not isinstance(address, Integral) or address < 0 or address % 128:
            raise ValueError('Activation addresses must be nonnegative and 128-byte aligned')
    if (module.up_ratio, module.down_ratio) != (2, 2):
        raise ValueError('Only official factor-two alias-free activation is supported')
    if not module.act.alpha_logscale or module.act.alpha.numel() != logical_channels:
        raise ValueError('Expected the official log-scale SnakeBeta activation')
    count = max(1, min(2 * rows, ACTIVATION_TILE_ELEMENTS // width))
    alpha = torch.zeros(count, width, dtype=torch.bfloat16)
    beta = torch.zeros_like(alpha)
    alpha[:, :logical_channels] = module.act.alpha.detach().exp()
    beta[:, :logical_channels] = 1 / (module.act.beta.detach().exp() + 1e-9)
    up = quantize_filter_dc(tuple(float(x) for x in module.upsample.filter.flatten()))
    down = quantize_filter_dc(tuple(float(x) for x in module.downsample.lowpass.filter.flatten()))
    plan = ActivationPlan(rows, width, input_address, output_address, scratch_address,
                          identity_address, image.allocate(alpha), image.allocate(beta),
                          count, up, down, filter_accumulation, snake_polynomial, filter_stage)
    if filter_accumulation == 'matrix':
        from bigcodec_filter import prepare_fir
        workspace = scratch_address + 2 * rows * width * 2
        if filter_stage in ('up', 'both'):
            plan.up_fir = prepare_fir(module.upsample.filter, image,
                input_shape=input_shape, source=input_address, destination=scratch_address,
                scratch=workspace, identity=identity_address, kind='up')
        if filter_stage in ('down', 'both'):
            plan.down_fir = prepare_fir(module.downsample.lowpass.filter, image,
                input_shape=(2 * rows, logical_channels), source=scratch_address,
                destination=output_address, scratch=workspace, identity=identity_address, kind='down')
    return plan


def emit_activation_dram(engine, plan):
    """DRAM reference lowering with the same FIR order as the SRAM path."""
    if plan.filter_accumulation == 'matrix':
        raise ValueError('Matrix FIR requires the SRAM activation emitter')
    t, c = plan.rows, plan.width
    coefficients = _snake_coefficients(plan.snake_polynomial)
    row_tensor_bytes = t * c * 2
    y = plan.scratch
    a = y + 2 * row_tensor_bytes
    b = a + 2 * row_tensor_bytes
    p = b + 2 * row_tensor_bytes
    # Polyphase filtered upsampling, retaining the exact replicated endpoints.
    for phase in range(2):
        first = True
        taps = ((tap, 2 * plan.up_filter[tap]) for tap in range(12)
                if (phase + 15 - tap) % 2 == 0)
        for tap, coefficient in _ordered_filter_taps(taps, plan.filter_accumulation):
            gather_rows(engine, plan.source, b, source_rows=t, rows=t, width=c,
                        start=(phase + 5 - tap) // 2)
            scale(engine, b, b, t * c, coefficient)
            if first:
                copy(engine, b, a, row_tensor_bytes)
                first = False
            else:
                elementwise(engine, udc.UE_MODE.ELTWISE_ADD, a, b, a, t * c)
        scatter_rows(engine, a, y, rows=t, width=c, start=phase, stride=2)

    n = 2 * t * c
    channel_mul(engine, y, a, 2 * t, c, plan.alpha, plan.constant_rows)
    scale(engine, a, b, n, -1)
    clamp(engine, b, b, n, plan.identity)
    clamp(engine, a, a, n, plan.identity)
    elementwise(engine, udc.UE_MODE.ELTWISE_ADD, a, b, a, n)
    clamp(engine, a, a, n, plan.identity, 0, SNAKE_ARGUMENT_LIMIT)
    for exponent in range(5, -1, -1):
        shift(engine, a, b, n, -(math.pi / 2) * 2 ** exponent)
        clamp(engine, b, b, n, plan.identity)
        scale(engine, b, b, n, 2)
        elementwise(engine, udc.UE_MODE.ELTWISE_SUB, a, b, a, n)
    elementwise(engine, udc.UE_MODE.ELTWISE_MUL, a, a, a, n)
    scale(engine, a, p, n, coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        shift(engine, p, p, n, coefficient)
        elementwise(engine, udc.UE_MODE.ELTWISE_MUL, p, a, p, n)
    clamp(engine, p, p, n, plan.identity, 0, 1)
    channel_mul(engine, p, p, 2 * t, c, plan.inverse_beta, plan.constant_rows)
    elementwise(engine, udc.UE_MODE.ELTWISE_ADD, y, p, y, n)
    # Low-pass filter followed by factor-two decimation.
    for index, (tap, coefficient) in enumerate(_ordered_filter_taps(
            enumerate(plan.down_filter), plan.filter_accumulation)):
        gather_rows(engine, y, b, source_rows=2 * t, rows=t, width=c,
                    start=tap - 5, stride=2)
        scale(engine, b, b, t * c, coefficient)
        if index == 0:
            copy(engine, b, plan.destination, row_tensor_bytes)
        else:
            elementwise(engine, udc.UE_MODE.ELTWISE_ADD,
                        plan.destination, b, plan.destination, t * c)


def _sine_polynomial_sram(engine, elements, snake_polynomial='legacy'):
    """Reduce folded A0 values to sine-squared in A10000, before clamping.

    At most 32768 elements occupy each 64 KiB slot. Preserve the original
    signal in B80000 and zeros in A30000.
    """
    coefficients = _snake_coefficients(snake_polynomial)
    if (not isinstance(elements, int) or isinstance(elements, bool)
            or elements < 64 or elements > ACTIVATION_TILE_ELEMENTS or elements % 64):
        raise ValueError('Snake polynomial needs 64-aligned elements within one SRAM tile')
    a, p, square = 0, 0x10000, 0x90000
    sram_copy(engine, a, square, elements)
    engine.eltwise_mul_core(a, square, a, elements)
    sram_copy(engine, a, square, elements)
    sram_scale(engine, a, p, elements, coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        sram_shift(engine, p, p, elements, coefficient)
        engine.eltwise_mul_core(p, square, p, elements)


def emit_activation(engine, plan):
    """Keep FIR accumulators and the complete folded Snake polynomial in SRAM.

    Legacy FIR multiplications/additions write BF16 after each operation;
    selected matrix FIRs write once after the complete dot product.
    Wide MAXPOOL compare/select implements finite-value clamps, avoiding the
    scalar identity-dot path. Full tensors cross DRAM only between upsample,
    Snake and downsample stages; FIR taps load directly into the tile buffer.
    """
    t, c = plan.rows, plan.width
    _snake_coefficients(plan.snake_polynomial)
    _filter_options(plan.filter_accumulation, plan.filter_stage)
    if plan.filter_accumulation == 'matrix':
        from bigcodec_filter import emit_fir
        if ((plan.filter_stage in ('up', 'both')) != (plan.up_fir is not None)
                or (plan.filter_stage in ('down', 'both')) != (plan.down_fir is not None)):
            raise ValueError('Matrix FIR plans do not match the selected stage')
    elif plan.up_fir is not None or plan.down_fir is not None:
        raise ValueError('Legacy activation cannot contain matrix FIR plans')
    legacy_order = 'serial' if plan.filter_accumulation == 'matrix' else plan.filter_accumulation
    tile_rows = min(plan.constant_rows, ACTIVATION_TILE_ELEMENTS // c)
    if tile_rows <= 0:
        raise ValueError('Activation channel width exceeds the SRAM tile budget')
    # Fixed disjoint vector slots permit up to32768 BF16 elements each.
    a, p, temporary, zeros, limit = 0, 0x10000, 0x20000, 0x30000, 0x40000
    original, square, constant = 0x80000, 0x90000, 0xA0000

    def filter_phase(source, destination, *, source_rows, start, stride,
                     taps, output_start=0, output_stride=1):
        taps = _ordered_filter_taps(taps, legacy_order)
        for first in range(0, t, tile_rows):
            count = min(tile_rows, t - first)
            elements = count * c
            for index, (tap_offset, coefficient) in enumerate(taps):
                gather_rows_to_sram(engine, source, a, source_rows=source_rows,
                    rows=count, width=c, start=start + first * stride + tap_offset,
                    stride=stride)
                sram_scale(engine, a, a, elements, coefficient)
                if index == 0:
                    sram_copy(engine, a, original, elements)
                else:
                    engine.eltwise_add_core(a, original, original, elements)
            shared._copy_contiguous_or_strided_write(engine, sram=original,
                destination=destination + (output_start + first * output_stride) * c * 2,
                total=elements * 2, chunk=c * 2, jump=output_stride * c * 2)

    if plan.up_fir is not None:
        emit_fir(engine, plan.up_fir)
    else:
        for phase in range(2):
            taps = [((phase + 5 - tap) // 2, 2 * plan.up_filter[tap])
                    for tap in range(12) if (phase + 15 - tap) % 2 == 0]
            filter_phase(plan.source, plan.scratch, source_rows=t, start=0,
                         stride=1, taps=taps, output_start=phase, output_stride=2)

    for first in range(0, 2 * t, tile_rows):
        count = min(tile_rows, 2 * t - first)
        elements = count * c
        address = plan.scratch + first * c * 2
        engine.accelerator_memory_to_sram(address, a, elements)
        sram_copy(engine, a, original, elements)
        engine.accelerator_memory_to_sram(plan.alpha, constant, elements)
        engine.eltwise_mul_core(a, constant, a, elements)
        sram_scale(engine, a, zeros, elements, 0)
        # abs(x) is exact for finite BF16; max selects without a dot reduction.
        sram_scale(engine, a, temporary, elements, -1)
        sram_maximum(engine, a, temporary, a, elements)
        # min(abs(x), argument_limit) via -max(-abs(x), -argument_limit).
        sram_scale(engine, a, temporary, elements, -1)
        sram_shift(engine, zeros, limit, elements, -SNAKE_ARGUMENT_LIMIT)
        sram_maximum(engine, temporary, limit, temporary, elements)
        sram_scale(engine, temporary, a, elements, -1)
        for exponent in range(5, -1, -1):
            sram_shift(engine, a, temporary, elements, -(math.pi / 2) * 2 ** exponent)
            sram_maximum(engine, temporary, zeros, temporary, elements)
            sram_scale(engine, temporary, square, elements, 2)
            engine.eltwise_sub_core(a, square, a, elements)
        _sine_polynomial_sram(engine, elements, plan.snake_polynomial)
        sram_maximum(engine, p, zeros, p, elements)
        sram_scale(engine, p, temporary, elements, -1)
        sram_shift(engine, zeros, limit, elements, -1)
        sram_maximum(engine, temporary, limit, temporary, elements)
        sram_scale(engine, temporary, p, elements, -1)
        engine.accelerator_memory_to_sram(plan.inverse_beta, constant, elements)
        engine.eltwise_mul_core(p, constant, p, elements)
        engine.eltwise_add_core(p, original, a, elements)
        engine.sram_to_accelerator_memory(a, address, elements)

    if plan.down_fir is not None:
        emit_fir(engine, plan.down_fir)
    else:
        filter_phase(plan.scratch, plan.destination, source_rows=2 * t, start=-5,
                     stride=2, taps=list(enumerate(plan.down_filter)))
