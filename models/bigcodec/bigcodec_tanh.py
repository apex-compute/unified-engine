"""Compensated BF16 Padé tanh with a BF16 high and low part per value.

The native arithmetic rounds through BF19 before BF16 writeback. Product
residuals and an approximate TwoSum retain the smaller terms across Horner
evaluation and one reciprocal Newton correction. This is finite arithmetic
for LSTM arguments clamped to [-4,4], not a general IEEE double-double API.
"""

from __future__ import annotations

import torch

from bigcodec_lstm import _binary_a, _identity_sram, _two_product_sram, _two_sum_sram, udc


_NUMERATOR = (1 / 135135, 378 / 135135, 17325 / 135135, 1.)
_DENOMINATOR = (28 / 135135, 3150 / 135135, 62370 / 135135, 1.)

# Fixed 4096-element slots; arithmetic helpers own A[0x50000:0x66000].
_ZERO = 0x4E000
_X = (0x10000, _ZERO)
_SQUARE = (0x12000, 0x14000)
_NUM = (0x16000, 0x18000)
_DEN = (0x1A000, 0x1C000)
_COEFF = (0x1E000, 0x20000)
_INVERSE = (0x22000, 0x24000)
_PRODUCT = (0x26000, 0x28000)
_NEGATIVE = (0x2A000, 0x2C000)
_ERROR = (0x2E000, 0x30000)
_RESULT = (0x32000, 0x34000)
_HI, _LO, _CROSS_A, _CROSS_B, _CORRECTION = (0x36000, 0x38000, 0x3A000, 0x3C000, 0x3E000)


def _constant_pair(engine, value, output, count):
    value = torch.tensor(value, dtype=torch.float32)
    high = value.bfloat16().float()
    low = (value - high).bfloat16().float()
    for scalar, address in zip((high, low), output):
        engine.broadcast_add(float(scalar), _ZERO, address, count)


def _add_pair(engine, left, right, output, count):
    """Allow output to alias either input pair; common work is disjoint."""
    _two_sum_sram(engine, left[0], right[0], _HI, _LO, count)
    _binary_a(engine, "add", left[1], right[1], _CORRECTION, count)
    _binary_a(engine, "add", _CORRECTION, _LO, _CORRECTION, count)
    _two_sum_sram(engine, _HI, _CORRECTION, output[0], output[1], count)


def _multiply_pair(engine, left, right, output, count):
    _two_product_sram(engine, left[0], right[0], _HI, _LO, count)
    _binary_a(engine, "mul", left[0], right[1], _CROSS_A, count)
    _binary_a(engine, "mul", left[1], right[0], _CROSS_B, count)
    _binary_a(engine, "add", _CROSS_A, _CROSS_B, _CORRECTION, count)
    _binary_a(engine, "add", _CORRECTION, _LO, _CORRECTION, count)
    _two_sum_sram(engine, _HI, _CORRECTION, output[0], output[1], count)


def _polynomial(engine, coefficients, output, count):
    _constant_pair(engine, coefficients[0], output, count)
    for coefficient in coefficients[1:]:
        _multiply_pair(engine, output, _SQUARE, output, count)
        _constant_pair(engine, coefficient, _COEFF, count)
        _add_pair(engine, output, _COEFF, output, count)


def compensated_tanh_sram(engine, source, output, count, *, identity=0x80000):
    """Emit tanh for 64..4096 pad64 values in low URAM_A, allowing aliasing.

    Only the requested output in A[0:0x10000] changes. Scratch is
    A[0x10000:0x66000] and B[0xF0000:0xF2000]; all other B, including the
    64x64 identity, is preserved. A[0x30000:0x40000] overlaps cell-product
    temporaries, which must be dead before this function is called. No DRAM
    traffic is emitted. Low SRAM gate/cell vectors and input are preserved
    except where they intentionally overlap output.
    """
    if not isinstance(count, int) or count <= 0 or count > 4096 or count % 64:
        raise ValueError("Compensated tanh requires 64..4096 pad64 elements")
    if any(not isinstance(address, int) or address < 0 or address % 128
           or address + count * 2 > 0x10000 for address in (source, output)):
        raise ValueError("Compensated tanh input/output must be aligned below A0x10000")
    if (not isinstance(identity, int) or identity % 128 or identity < 0x80000
            or identity + 64 * 64 * 2 > 0xF0000):
        raise ValueError("Compensated tanh identity must fit in B below the operand mirror")

    # Clamp before populating polynomial vectors: the wide clamp uses A0x20000.
    _identity_sram(engine, source, _X[0], count, identity, udc.LALU_MODE.CLAMP,
                   lower=-4., upper=4.)
    engine.broadcast_mul(0., _X[0], _ZERO, count)
    _two_product_sram(engine, _X[0], _X[0], _SQUARE[0], _SQUARE[1], count)
    _polynomial(engine, _NUMERATOR, _NUM, count)
    _polynomial(engine, _DENOMINATOR, _DEN, count)

    # y <- y + y*(1 - denominator*y), retaining denominator/product residuals.
    _identity_sram(engine, _DEN[0], _INVERSE[0], count, identity,
                   udc.LALU_MODE.MODE_RECIP)
    engine.broadcast_mul(0., _ZERO, _INVERSE[1], count)
    _multiply_pair(engine, _DEN, _INVERSE, _PRODUCT, count)
    for source_part, negative_part in zip(_PRODUCT, _NEGATIVE):
        engine.broadcast_mul(-1., source_part, negative_part, count)
    _constant_pair(engine, 1., _COEFF, count)
    _add_pair(engine, _COEFF, _NEGATIVE, _ERROR, count)
    _binary_a(engine, "add", _ERROR[0], _ERROR[1], _RESULT[0], count)
    _binary_a(engine, "mul", _INVERSE[0], _RESULT[0], _RESULT[0], count)
    engine.broadcast_mul(0., _ZERO, _RESULT[1], count)
    _add_pair(engine, _INVERSE, _RESULT, _INVERSE, count)

    _multiply_pair(engine, _NUM, _INVERSE, _PRODUCT, count)
    _multiply_pair(engine, _PRODUCT, _X, _RESULT, count)
    _binary_a(engine, "add", _RESULT[0], _RESULT[1], output, count)
    _identity_sram(engine, output, output, count, identity, udc.LALU_MODE.CLAMP,
                   lower=-1., upper=1.)
