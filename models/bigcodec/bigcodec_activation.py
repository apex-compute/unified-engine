"""Alias-free SnakeBeta lowering with an explicit BF16 sine approximation.

The filters and endpoint handling follow upstream. Sine-squared is evaluated
on a folded interval with a degree-ten polynomial. This backend is approximate:
its argument is clamped to +/-32*pi before folding, and FPGA accuracy must be
measured against the official reference before deployment.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import math
import torch

from bigcodec_device import (channels, channel_mul, clamp, copy, elementwise,
                             gather_rows, scale, scatter_rows, shift, udc)

SNAKE_ARGUMENT_LIMIT = 32 * math.pi


def sine_squared(value, *, bf16=False):
    """Torch model of the lowering, including every BF16 storage boundary."""
    q = lambda x: x.to(torch.bfloat16).float() if bf16 else x
    scalar = lambda x: float(torch.tensor(x).bfloat16()) if bf16 else x
    x = q(value).abs().clamp(max=SNAKE_ARGUMENT_LIMIT)
    x = q(x)
    for exponent in range(5, -1, -1):
        center = scalar((math.pi / 2) * 2 ** exponent)
        delta = q(x - center).clamp_min(0)
        x = q(x - q(2 * delta))
    z = q(x * x)
    result = q(z * scalar(2 / 14175))
    for coefficient in (-1 / 315, 2 / 45, -1 / 3, 1):
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


def activation_scratch_bytes(input_shape):
    rows, logical_channels = map(int, input_shape)
    return 9 * rows * channels(logical_channels) * 2


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


def prepare_activation(module, image, *, input_shape, input_address,
                       output_address, scratch_address, identity_address):
    rows, logical_channels = map(int, input_shape)
    width = channels(logical_channels)
    if (module.up_ratio, module.down_ratio) != (2, 2):
        raise ValueError('Only official factor-two alias-free activation is supported')
    if not module.act.alpha_logscale or module.act.alpha.numel() != logical_channels:
        raise ValueError('Expected the official log-scale SnakeBeta activation')
    count = max(1, min(2 * rows, 8192 // width))
    alpha = torch.zeros(count, width, dtype=torch.bfloat16)
    beta = torch.zeros_like(alpha)
    alpha[:, :logical_channels] = module.act.alpha.detach().exp()
    beta[:, :logical_channels] = 1 / (module.act.beta.detach().exp() + 1e-9)
    up = quantize_filter_dc(tuple(float(x) for x in module.upsample.filter.flatten()))
    down = quantize_filter_dc(tuple(float(x) for x in module.downsample.lowpass.filter.flatten()))
    return ActivationPlan(rows, width, input_address, output_address, scratch_address,
                          identity_address, image.allocate(alpha), image.allocate(beta),
                          count, up, down)


def emit_activation(engine, plan):
    t, c = plan.rows, plan.width
    row_tensor_bytes = t * c * 2
    y = plan.scratch
    a = y + 2 * row_tensor_bytes
    b = a + 2 * row_tensor_bytes
    p = b + 2 * row_tensor_bytes
    # Polyphase filtered upsampling, retaining the exact replicated endpoints.
    for phase in range(2):
        first = True
        for tap in range(12):
            if (phase + 15 - tap) % 2:
                continue
            gather_rows(engine, plan.source, b, source_rows=t, rows=t, width=c,
                        start=(phase + 5 - tap) // 2)
            scale(engine, b, b, t * c, 2 * plan.up_filter[tap])
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
    scale(engine, a, p, n, 2 / 14175)
    for coefficient in (-1 / 315, 2 / 45, -1 / 3, 1):
        shift(engine, p, p, n, coefficient)
        elementwise(engine, udc.UE_MODE.ELTWISE_MUL, p, a, p, n)
    clamp(engine, p, p, n, plan.identity, 0, 1)
    channel_mul(engine, p, p, 2 * t, c, plan.inverse_beta, plan.constant_rows)
    elementwise(engine, udc.UE_MODE.ELTWISE_ADD, y, p, y, n)
    # Low-pass filter followed by factor-two decimation.
    for tap, coefficient in enumerate(plan.down_filter):
        gather_rows(engine, y, b, source_rows=2 * t, rows=t, width=c,
                    start=tap - 5, stride=2)
        scale(engine, b, b, t * c, coefficient)
        if tap == 0:
            copy(engine, b, plan.destination, row_tensor_bytes)
        else:
            elementwise(engine, udc.UE_MODE.ELTWISE_ADD,
                        plan.destination, b, plan.destination, t * c)
