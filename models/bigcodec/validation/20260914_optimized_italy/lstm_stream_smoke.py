#!/usr/bin/env python3
"""Capture official BigCodec LSTM smoke programs without touching hardware.

Add --execute to run on Italy. Execution locks the board and requires version
0xdf0749de by default, or an explicit --expected-version, with AXI256 before
resetting or uploading anything. Each selected
stack/precision uses one START/HALT and compares against independent BF16 gate
math, using dequantized recurrent weights for the IF8 reference.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'models/bigcodec'))
from bigcodec_common import CHECKPOINT_SHA256, load_models
from bigcodec_device import shared, udc
from bigcodec_lstm import prepare_lstm, emit_lstm

HERE = Path(__file__).resolve().parent
INPUT_BASE, TENSOR_BASE = 0x80000000, 0xB0000000
GUARD_ELEMENTS = 64
SENTINEL = -3.25
EXPECTED_VERSION = 0xDF0749DE


def rounded(value):
    return torch.as_tensor(value).bfloat16().float()


def reference_if8_weights(weight):
    """Independent reproduction of the driver's INT8 block scale/dequantizer."""
    blocks = rounded(weight).reshape(-1, 64)
    maximum = blocks.abs().amax(-1)
    scale = rounded(maximum / 127.)
    scale[maximum == 0] = 1.
    codes = rounded(blocks / scale[:, None]).round().clamp(-128, 127)
    return rounded(codes * scale[:, None]).reshape_as(weight)


def reference_tanh(value):
    x = rounded(value.clamp(-4., 4.))
    x2 = rounded(x * x)
    numerator = rounded(x2 * rounded(1. / 135135.))
    for c in (378. / 135135., 17325. / 135135.):
        numerator = rounded(rounded(numerator + rounded(c)) * x2)
    numerator = rounded(numerator + 1.)
    denominator = rounded(x2 * rounded(28. / 135135.))
    for c in (3150. / 135135., 62370. / 135135.):
        denominator = rounded(rounded(denominator + rounded(c)) * x2)
    denominator = rounded(denominator + 1.)
    return rounded(x * rounded(numerator * rounded(1. / denominator))).clamp(-1., 1.)


def reference_lstm(module, source, precision):
    """Explicit i/f/g/o equations, with the residual after both LSTM layers."""
    original = source.float()
    current = original
    with torch.inference_mode():
        for layer in range(2):
            wi = rounded(getattr(module, f'weight_ih_l{layer}'))
            wh = getattr(module, f'weight_hh_l{layer}').detach()
            wh = reference_if8_weights(wh) if precision == 'if8' else rounded(wh)
            bi = rounded(getattr(module, f'bias_ih_l{layer}'))
            bh = rounded(getattr(module, f'bias_hh_l{layer}'))
            projections = rounded(current @ wi.T + bi)
            hidden = torch.zeros(module.hidden_size)
            cell = torch.zeros_like(hidden)
            output = []
            for projection in projections:
                i, f, g, o = rounded(projection + rounded(hidden @ wh.T + bh)).chunk(4)
                i, f, o = (rounded(value.sigmoid()) for value in (i, f, o))
                g = reference_tanh(g)
                cell = rounded(rounded(f * cell) + rounded(i * g))
                hidden = rounded(o * reference_tanh(cell))
                output.append(hidden)
            current = torch.stack(output)
    return rounded(current + original)


def prepare_case(name, module, precision, steps, seed, expected_version=EXPECTED_VERSION):
    if module.input_size != 1536 or module.hidden_size != 1536:
        raise ValueError('Smoke requires the official 1536-wide LSTM')
    generator = torch.Generator().manual_seed(seed)
    source = (torch.randn(steps, 1536, generator=generator) * .2).bfloat16()
    expected = reference_lstm(module, source, precision)
    image = shared._ImageBuilder(shared.MODEL_BASE, shared.MODEL_LIMIT)
    zero_address = image.allocate(torch.zeros(64, dtype=torch.bfloat16), alignment=128)
    identity_address = image.allocate(torch.eye(64, dtype=torch.bfloat16), alignment=128)
    output_address = TENSOR_BASE + GUARD_ELEMENTS * 2
    output_bytes = source.numel() * 2
    scratch_address = shared._align_up(output_address + output_bytes + GUARD_ELEMENTS * 2, 128)
    plan = prepare_lstm(module, image, input_shape=source.shape,
        input_address=INPUT_BASE + GUARD_ELEMENTS * 2, output_address=output_address,
        scratch_address=scratch_address, zero_address=zero_address,
        identity_address=identity_address, recurrent_precision=precision)
    program_address = image.align(128)
    capture = shared._WholeGraphEngine(program_address)
    capture.start_capture()
    with contextlib.redirect_stdout(io.StringIO()):
        emit_lstm(capture, plan)
    halt_index = capture.capture_count
    capture.generate_instruction_halt()
    capture.stop_capture()
    issues = udc.check_isa_jumps(capture.capture_buffer, program_address, name=name)
    if issues:
        raise RuntimeError('\n'.join(issues))
    kinds = [(instruction.words[0] >> 8) & 15 for instruction in capture.capture_buffer]
    if (kinds.count(udc.INSTRUCTION_HALT) != 1 or udc.INSTRUCTION_SWI in kinds
            or any(kind != udc.INSTRUCTION_NOP for kind in kinds[halt_index + 1:])):
        raise RuntimeError('Capture must contain one terminal HALT and no SWI')
    if capture._isa_reg_counter != 1 or capture._inst_ptr_counter != 1:
        raise RuntimeError('Capture leaked an ISA register or pointer')
    program = b''.join(instruction.get_bytes() for instruction in capture.capture_buffer)
    if not program or len(program) % 64:
        raise RuntimeError('Program is not aligned for instruction fetch')
    image.write(program_address, program)
    raw = torch.frombuffer(bytearray(image.data), dtype=torch.uint8)
    tensor_end = scratch_address + plan.scratch_bytes + GUARD_ELEMENTS * 2
    tensor_init = torch.full(((tensor_end - TENSOR_BASE) // 2,), SENTINEL, dtype=torch.bfloat16)
    output_slice = slice((output_address - TENSOR_BASE) // 2,
                         (output_address - TENSOR_BASE + output_bytes) // 2)
    scratch_slice = slice((scratch_address - TENSOR_BASE) // 2,
                          (scratch_address - TENSOR_BASE + plan.scratch_bytes) // 2)
    guard_mask = torch.ones(tensor_init.numel(), dtype=torch.bool)
    guard_mask[output_slice] = False
    guard_mask[scratch_slice] = False
    tensor_init[output_slice] = float('nan')
    tensor_init[scratch_slice] = float('nan')
    input_init = torch.full((source.numel() + 2 * GUARD_ELEMENTS,), SENTINEL, dtype=torch.bfloat16)
    input_init[GUARD_ELEMENTS:-GUARD_ELEMENTS] = source.flatten()
    record = dict(case=f'{name}_{precision}', stack=name, recurrent_precision=precision,
        checkpoint_sha256=CHECKPOINT_SHA256, shape=list(source.shape), padded_width=plan.padded_width,
        padded_channels=plan.padded_width - module.hidden_size,
        source='seeded random BF16 features: torch.randn * 0.2', seed=seed,
        input_sha256=hashlib.sha256(source.view(torch.uint8).numpy().tobytes()).hexdigest(),
        reference_sha256=hashlib.sha256(expected.numpy().tobytes()).hexdigest(),
        reference_finite=bool(torch.isfinite(expected).all()),
        reference_note='Independent BF16 gate equations; IF8 weights first dequantized to BF16. Streaming hardware scale/accumulator rounding can differ.',
        instructions=capture.capture_count, halt_index=halt_index, terminal_halts=1,
        resident_bytes=raw.numel(), program_bytes=len(program),
        program_sha256=hashlib.sha256(program).hexdigest(),
        model_sha256=hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
        axi_data_width_bits=256, expected_hardware_version=f'0x{expected_version:08x}',
        input_address=plan.input_address, output_address=plan.output_address,
        output_bytes=output_bytes, scratch_address=scratch_address, scratch_bytes=plan.scratch_bytes,
        tensor_guard_elements=int(guard_mask.sum()), hardware_executed=False)
    return dict(raw=raw, source=source, expected=expected, plan=plan, program_address=program_address,
                tensor_init=tensor_init, output_slice=output_slice, guard_mask=guard_mask,
                input_init=input_init, record=record)


def execute_case(engine, clock, case, timeout, max_error, expected_version=EXPECTED_VERSION):
    observed_version = engine.get_hardware_version()
    if observed_version != expected_version or engine.is_queue_busy():
        raise RuntimeError('Wrong hardware version or FPGA queue already busy')
    engine.software_reset(run_dram_self_test=False)
    def write(address, value):
        size = value.numel() * value.element_size()
        if engine.dma_write(engine.h2c_device, address, value, size) != size:
            raise RuntimeError('Short DMA write')
    write(shared.MODEL_BASE, case['raw'])
    write(INPUT_BASE, case['input_init'])
    write(TENSOR_BASE, case['tensor_init'])
    engine.write_reg32(udc.UE_INT_REG, 1)
    began = time.perf_counter()
    engine.start_execute_from_dram(case['program_address'])
    while engine.read_reg32(udc.UE_INT_REG) & 3 != udc.INT_CAUSE_HALT or engine.is_queue_busy():
        if time.perf_counter() - began > timeout:
            raise TimeoutError(case['record']['case'])
        time.sleep(.0001)
    host_execute_s = time.perf_counter() - began
    cycles = engine.read_latency_cycles()
    tensor = torch.empty_like(case['tensor_init'])
    size = tensor.numel() * 2
    if engine.dma_read(engine.c2h_device, TENSOR_BASE, tensor, size) != size:
        raise RuntimeError('Short output/guard DMA read')
    input_after = torch.empty_like(case['input_init'])
    if engine.dma_read(engine.c2h_device, INPUT_BASE, input_after, input_after.numel() * 2) != input_after.numel() * 2:
        raise RuntimeError('Short input/guard DMA read')
    actual = tensor[case['output_slice']].reshape_as(case['source']).float()
    delta = actual - case['expected']
    finite = bool(torch.isfinite(actual).all())
    record = case['record']
    record.update(hardware_executed=True, hardware_version=f'0x{engine.get_hardware_version():08x}',
        observed_hardware_version_before_upload=f'0x{observed_version:08x}',
        host_execute_s=host_execute_s, fpga_cycles=cycles, fpga_s=cycles * clock / 1e9,
        host_start_commands=1, terminal_halt_observed=True,
        output_finite=finite,
        output_guards_intact=bool((tensor[case['guard_mask']] == SENTINEL).all()),
        input_and_guards_unchanged=bool(torch.equal(input_after, case['input_init'])),
        padded_channels_zero=bool((actual[:, 1536:] == 0).all()),
        relative_l2=float(delta.norm() / case['expected'].norm()) if finite else None,
        max_abs_error=float(delta.abs().max()) if finite else None,
        max_relative_l2_allowed=max_error)
    torch.save({'actual':actual, 'expected':case['expected'], 'input':case['source'], 'report':record},
               HERE / (record['case'] + '_lstm_stream_smoke.pt'))
    record['passed'] = (record['output_finite'] and record['output_guards_intact']
        and record['input_and_guards_unchanged'] and record['padded_channels_zero']
        and record['relative_l2'] <= max_error)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Explicitly enable Italy FPGA execution')
    parser.add_argument('--expected-version', type=lambda value: int(value, 0), default=EXPECTED_VERSION,
                        help='Required hardware version before upload (default: 0xdf0749de)')
    parser.add_argument('--stack', choices=('encoder','decoder','both'), default='both')
    parser.add_argument('--precision', choices=('bf16','if8','both'), default='both')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--seed', type=int, default=850)
    parser.add_argument('--cpu-core', type=int, default=7)
    parser.add_argument('--timeout', type=float, default=60.)
    parser.add_argument('--max-relative-error', type=float, default=.05)
    args = parser.parse_args()
    if not 0 <= args.expected_version < 2**32:
        parser.error('--expected-version must be an unsigned 32-bit value')
    if not 1 <= args.steps <= 317:
        parser.error('--steps must be between 1 and 317')
    if args.cpu_core not in os.sched_getaffinity(0):
        parser.error('--cpu-core is unavailable')
    if not 0 < args.timeout < float('inf') or not 0 < args.max_relative_error < float('inf'):
        parser.error('timeout and max-relative-error must be positive and finite')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.sched_setaffinity(0,{args.cpu_core})
    udc.UE_AXI_DATA_WIDTH_BITS = 256
    encoder, decoder = load_models(remove_weight_norm=True)
    prepared = []
    for name, model in (('encoder',encoder),('decoder',decoder)):
        if args.stack not in ('both',name):
            continue
        module = next(module for module in model.modules() if isinstance(module,torch.nn.LSTM))
        for precision in ('bf16','if8'):
            if args.precision in ('both',precision):
                prepared.append(prepare_case(name,module,precision,args.steps,args.seed,args.expected_version))
    records = [case['record'] for case in prepared]
    if not args.execute:
        path = HERE / 'lstm_stream_smoke_capture.json'
        path.write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'mode':'offline_capture','report':str(path),'cases':records},indent=2),flush=True)
        return
    # Hardware-specific imports and all device operations remain behind --execute.
    if socket.gethostname().split('.')[0].lower() != 'italy':
        raise RuntimeError('--execute is restricted to Italy')
    from bigcodec_precompiled import StreamingEngine
    from yolov5_common import configure_hardware_runtime
    result_path = HERE / 'lstm_stream_smoke_execute.json'
    try:
        lock = open('/tmp/pcie_ci_hw_italy.lock','a')
    except PermissionError:
        # flock does not require write access. Reuse the same existing inode.
        lock = open('/tmp/pcie_ci_hw_italy.lock','r')
    with lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        clock, info, _ = configure_hardware_runtime(device='rk',dev='xdma0',cycle_override_ns=None)
        if info.axi_data_width_bits != 256:
            raise RuntimeError('LSTM smoke requires AXI256')
        with StreamingEngine(clock_period_ns=clock,conv_geometry_mode=udc.CONV_GEOMETRY_QUEUE_CONFIG) as engine:
            if engine.get_hardware_version() != args.expected_version:
                raise RuntimeError(f'LSTM smoke requires version 0x{args.expected_version:08x}; no upload performed')
            for case in prepared:
                record = execute_case(engine,clock,case,args.timeout,args.max_relative_error,args.expected_version)
                result_path.write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
                print(json.dumps(record,allow_nan=False),flush=True)
                if not record['passed']:
                    raise AssertionError(record)


if __name__ == '__main__':
    main()
