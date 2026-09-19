#!/usr/bin/env python3
"""Explicitly gated native probe of32/64/128-byte strided DMA and final Tanh."""
from pathlib import Path
import argparse
import json
import sys

import torch

import native_sram_smoke as native
from bigcodec_device import shared, zero, udc
from bigcodec_compile import emit_waveform_tanh, WAVEFORM_ROW_LANES
from bigcodec_lstm import tanh_identity, TANH_CHUNK_ELEMENTS

HERE = Path(__file__).resolve().parent / 'waveform_tanh_smoke'


def cases(include_tanh):
    results, expected = [], {}
    rows = 256
    full = (torch.arange(rows * 64) + 0x2000).to(torch.uint16).view(torch.bfloat16).reshape(rows, 64)
    for chunk in (32, 64, 128):
        lanes = chunk // 2
        constants = native.image()
        zeros = constants.allocate(torch.zeros(udc.URAM_NEAR_FULL_SIZE // 2, dtype=torch.bfloat16), alignment=128)
        def gather(engine, chunk=chunk, lanes=lanes, zeros=zeros, rows=rows):
            engine.accelerator_memory_to_sram(zeros, 0, rows * 64)
            shared._copy_contiguous_or_strided_read(engine,
                source=native.SOURCE, sram=0, total=rows * chunk, chunk=chunk, jump=128)
            engine.sram_to_accelerator_memory(0, native.OUTPUT, rows * lanes)
        name = f'gather_{chunk}'
        results.append(native.Case(name, constants, gather, [(native.SOURCE, full)], rows * lanes))
        expected[name] = full[:, :lanes].contiguous().flatten()
        compact = full[:, :lanes].contiguous()
        def scatter(engine, chunk=chunk, lanes=lanes, zeros=zeros, rows=rows):
            zero(engine, native.OUTPUT, rows * 128, zeros)
            engine.accelerator_memory_to_sram(native.SOURCE, 0, rows * lanes)
            shared._copy_contiguous_or_strided_write(engine, sram=0,
                destination=native.OUTPUT, total=rows * chunk, chunk=chunk, jump=128)
        name = f'scatter_{chunk}'
        results.append(native.Case(name, constants, scatter, [(native.SOURCE, compact)], rows * 64))
        target = torch.zeros_like(full)
        target[:, :lanes] = compact
        expected[name] = target.flatten()
    if include_tanh:
        for rows in (200, 260):
            constants = native.image()
            zeros = constants.allocate(torch.zeros(udc.URAM_NEAR_FULL_SIZE // 2, dtype=torch.bfloat16), alignment=128)
            identity = constants.allocate(torch.eye(64, dtype=torch.bfloat16), alignment=128)
            mask = torch.zeros(TANH_CHUNK_ELEMENTS, dtype=torch.bfloat16)
            mask[::WAVEFORM_ROW_LANES] = 1
            mask_address = constants.allocate(mask, alignment=128)
            values = torch.zeros(rows, 64, dtype=torch.bfloat16)
            values[:, 0] = torch.linspace(-.75, .75, rows)
            for compact in (False, True):
                name = f'tanh_{rows}_' + ('sram' if compact else 'full')
                def callback(engine, rows=rows, compact=compact, identity=identity,
                             zeros=zeros, mask_address=mask_address):
                    if compact:
                        emit_waveform_tanh(engine, native.SOURCE, native.OUTPUT, rows,
                                           identity, zeros, mask_address)
                    else:
                        tanh_identity(engine, native.SOURCE, native.OUTPUT, rows * 64,
                                      identity, scratch_address=native.SCRATCH)
                results.append(native.Case(name, constants, callback,
                    [(native.SOURCE, values)], rows * 64))
    return results, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--capture-only', action='store_true')
    parser.add_argument('--tanh', action='store_true')
    parser.add_argument('--expected-version', default='0x40519e0a')
    args = parser.parse_args()
    prepared, expected = cases(args.tanh)
    HERE.mkdir(exist_ok=True)
    native.HERE = HERE
    native.primitives = lambda: iter(prepared)
    sys.argv = [sys.argv[0], '--execute' if args.execute else '--capture-only',
                '--selection', 'primitives', '--expected-version', args.expected_version]
    native.main()
    if args.capture_only:
        return
    records = json.loads((HERE / 'native_sram_smoke.json').read_text())
    for record in records:
        name = record['name']
        actual = torch.load(HERE / (name + '.pt'), weights_only=True).flatten()
        if name in expected:
            target = expected[name]
        elif name.endswith('_sram'):
            target = torch.load(HERE / (name.removesuffix('_sram') + '_full.pt'), weights_only=True).flatten()
        else:
            continue
        mismatch = actual.view(torch.int16) != target.view(torch.int16)
        record.update(value_mismatches=int((actual != target).sum()), bit_mismatches=int(mismatch.sum()),
                      expected_first_bits=target[:16].view(torch.int16).tolist(),
                      actual_first_bits=actual[:16].view(torch.int16).tolist())
    (HERE / 'comparison.json').write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
    print(json.dumps([{key: item.get(key) for key in ('name', 'value_mismatches', 'bit_mismatches', 'fpga_s')}
                      for item in records], indent=2))


if __name__ == '__main__':
    main()
