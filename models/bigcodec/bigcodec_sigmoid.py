"""Compensated sigmoid gate pairs for the BF16 decoder LSTM.

Logits enter as BF16 after the recurrent dot product. Sigmoid uses the existing
compensated Padé tanh at x/2, retaining BF16 high/low gate values through the
cell and hidden products. Native arithmetic still rounds through BF19 and the
final hidden output is BF16; this is not FP32 emulation.
"""
from __future__ import annotations

import bigcodec_lstm as lstm
import bigcodec_tanh as paired
from bigcodec_device import udc


def sigmoid_pair_sram(engine, source, output, count):
    """Emit a sigmoid pair in low A SRAM, allowing source/output aliasing.

    The two outputs must be disjoint. Scratch is A[0x10000:0x66000] and
    B[0xF0000:0xF2000]. A BF16 64x64 identity must already be at B0x80000.
    The argument is clamped to [-8,8]. No DRAM transfers are emitted.
    """
    if type(count) is not int or count not in (64, 1536):
        raise ValueError('Paired sigmoid supports 64 or 1536 elements')
    if not isinstance(output, (tuple, list)) or len(output) != 2:
        raise ValueError('Paired sigmoid requires two output addresses')
    if any(type(address) is not int or address < 0 or address % 128
           or address + count * 2 > 0x10000 for address in (source, *output)):
        raise ValueError('Paired sigmoid vectors must be aligned below A0x10000')
    if abs(output[0] - output[1]) < count * 2:
        raise ValueError('Paired sigmoid high and low outputs overlap')

    engine.broadcast_mul(.5, source, paired._X[0], count)
    lstm._identity_sram(engine, paired._X[0], paired._X[0], count, 0x80000,
                        udc.LALU_MODE.CLAMP, lower=-4., upper=4.)
    engine.broadcast_mul(0., paired._X[0], paired._ZERO, count)
    lstm._two_product_sram(engine, paired._X[0], paired._X[0], *paired._SQUARE, count)
    paired._polynomial(engine, paired._NUMERATOR, paired._NUM, count)
    paired._polynomial(engine, paired._DENOMINATOR, paired._DEN, count)
    lstm._identity_sram(engine, paired._DEN[0], paired._INVERSE[0], count,
                        0x80000, udc.LALU_MODE.MODE_RECIP)
    engine.broadcast_mul(0., paired._ZERO, paired._INVERSE[1], count)
    paired._multiply_pair(engine, paired._DEN, paired._INVERSE, paired._PRODUCT, count)
    for source_part, negative_part in zip(paired._PRODUCT, paired._NEGATIVE):
        engine.broadcast_mul(-1., source_part, negative_part, count)
    paired._constant_pair(engine, 1., paired._COEFF, count)
    paired._add_pair(engine, paired._COEFF, paired._NEGATIVE, paired._ERROR, count)
    lstm._binary_a(engine, 'add', *paired._ERROR, paired._RESULT[0], count)
    lstm._binary_a(engine, 'mul', paired._INVERSE[0], paired._RESULT[0],
                    paired._RESULT[0], count)
    engine.broadcast_mul(0., paired._ZERO, paired._RESULT[1], count)
    paired._add_pair(engine, paired._INVERSE, paired._RESULT, paired._INVERSE, count)
    paired._multiply_pair(engine, paired._NUM, paired._INVERSE, paired._PRODUCT, count)
    paired._multiply_pair(engine, paired._PRODUCT, paired._X, paired._RESULT, count)
    paired._constant_pair(engine, 1., paired._COEFF, count)
    paired._add_pair(engine, paired._COEFF, paired._RESULT, paired._RESULT, count)
    paired._constant_pair(engine, .5, paired._COEFF, count)
    paired._multiply_pair(engine, paired._RESULT, paired._COEFF, paired._RESULT, count)
    for source_part, destination in zip(paired._RESULT, output):
        engine.broadcast_mul(1., source_part, destination, count)


def paired_lstm_step_sram(engine, plan, projection, previous_cell, output):
    """Consume four staged BF16 logit vectors and emit one decoder timestep.

    The recurrent projection has placed logits at A0x8000. Gate low parts are
    retained at A0x8000/0x9000/0xA000 after copying logits to low A. The fixed
    layout is supported only at width1536. Both persistent cell parts are saved
    before tanh overwrites the SRAM high part. Hidden output rounds to BF16.
    """
    width = plan.padded_width
    if (width != 1536 or plan.recurrent_precision != 'bf16'
            or not plan.compensated_cell or not plan.preserve_cell_residual
            or not plan.compensated_tanh or not plan.fused_projection):
        raise ValueError('Paired LSTM step requires width1536 and compensated fused BF16 math')
    low_cell = plan.regions.get('cell_low')
    if low_cell is None or len(low_cell) != 2 or low_cell[1] != width:
        raise ValueError('Paired LSTM step requires the complete low-cell region')
    regions = ((plan.identity_address, 8192), (previous_cell, width * 2),
               (low_cell[0], width * 2), (output, width * 2))
    if any(type(address) is not int or address < 0 or address % 128
           or address + size > 1 << 35 for address, size in regions):
        raise ValueError('Paired LSTM step requires aligned, bounded DRAM regions')
    ordered = sorted(regions)
    if any(left[0] + left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError('Paired LSTM step DRAM regions overlap')

    engine.broadcast_mul(1., 0x8000, 0, 4 * width)
    engine.accelerator_memory_to_sram(plan.identity_address, 0x80000, 4096)
    input_gate, forget_gate, candidate, output_gate = (index * width * 2 for index in range(4))
    input_pair = (input_gate, 0x8000)
    forget_pair = (forget_gate, 0x9000)
    output_pair = (output_gate, 0xA000)
    for high, low in (input_pair, forget_pair, output_pair):
        sigmoid_pair_sram(engine, high, (high, low), width)
    lstm._lstm_tanh_sram(engine, plan, candidate, candidate, width)
    cell = (0xC000, 0xE000)
    engine.accelerator_memory_to_sram(previous_cell, cell[0], width)
    engine.accelerator_memory_to_sram(low_cell[0], cell[1], width)
    paired._multiply_pair(engine, forget_pair, cell, paired._PRODUCT, width)
    paired._multiply_pair(engine, input_pair, (candidate, paired._ZERO), paired._NEGATIVE, width)
    paired._add_pair(engine, paired._PRODUCT, paired._NEGATIVE, cell, width)
    engine.sram_to_accelerator_memory(cell[0], previous_cell, width)
    engine.sram_to_accelerator_memory(cell[1], low_cell[0], width)
    lstm._lstm_tanh_sram(engine, plan, cell[0], cell[0], width)
    paired._multiply_pair(engine, output_pair, (cell[0], paired._ZERO), paired._RESULT, width)
    lstm._binary_a(engine, 'add', *paired._RESULT, 0, width)
    engine.sram_to_accelerator_memory(0, output, width)
