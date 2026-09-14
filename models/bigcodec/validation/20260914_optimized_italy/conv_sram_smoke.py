"""Root-only HW recipe: compare old/new BF16 convolutions on real weights.

Without --execute this only compiles and checks instructions on the CPU.
With --execute it takes Italy's hardware lock and tests one START/HALT per
case/backend, requiring exact BF16 equality between old and new outputs.
"""
from pathlib import Path
import argparse
import contextlib
import fcntl
import hashlib
import io
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'models/bigcodec'))
import bigcodec_conv_bf16 as conv
from bigcodec_common import load_models
from bigcodec_precompiled import StreamingEngine
from yolov5_common import configure_hardware_runtime

HERE = Path(__file__).resolve().parent
INPUT, OUTPUT, SCRATCH = 0x80000000, 0xB0000000, 0xB1000000


def cases(encoder, decoder):
    modules = list(encoder.named_modules()) + [('decoder.' + n, m) for n, m in decoder.named_modules()]
    def pick(channels, kernel, dilation):
        return next((n, m) for n, m in modules if isinstance(m, torch.nn.Conv1d)
                    and m.in_channels == m.out_channels == channels
                    and m.kernel_size == (kernel,) and m.dilation == (dilation,))
    yield 'stem', 'encoder.block.0', encoder.block[0], 400
    for label, length, channels, dilation in (
            ('small_tile_cross', 2051, 48, 1),
            ('deep_n32_tile_cross', 635, 768, 1),
            ('deep_dilation9', 95, 768, 9)):
        name, module = pick(channels, 7, dilation)
        yield label, name, module, length
    name, module = next((n, m) for n, m in modules if isinstance(m, torch.nn.ConvTranspose1d)
                        and m.in_channels == 1536 and m.out_channels == 768)
    yield 'transpose_tile_cross', name, module, 173


def prepare(label, name, module, length):
    image = conv.shared._ImageBuilder(conv.shared.MODEL_BASE, conv.shared.MODEL_LIMIT)
    zero = image.allocate(torch.zeros(conv.ZERO_BYTES // 2, dtype=torch.bfloat16), alignment=128)
    shape = (length, module.in_channels)
    kwargs = dict(input_shape=shape, input_address=INPUT, output_address=OUTPUT,
                  scratch_address=SCRATCH, image=image, stride=module.stride[0], padding=module.padding[0])
    transpose = isinstance(module, torch.nn.ConvTranspose1d)
    if transpose:
        plan = conv.prepare_conv_transpose1d(label, module.weight, module.bias,
                                             output_padding=module.output_padding[0], **kwargs)
    else:
        plan = conv.prepare_conv1d(label, module.weight, module.bias,
                                   dilation=module.dilation[0], **kwargs)
    generator = torch.Generator().manual_seed(1847)
    packed = torch.zeros(length, conv.shared._align_up(module.in_channels, 64), dtype=torch.bfloat16)
    packed[:, :module.in_channels] = torch.randn(shape, generator=generator) * .2
    x = packed[:, :module.in_channels].T[None].float()
    weight = module.weight.detach().bfloat16().float()
    bias = None if module.bias is None else module.bias.detach().bfloat16().float()
    with torch.inference_mode():
        if transpose:
            expected = F.conv_transpose1d(x, weight, bias, stride=module.stride[0],
                                           padding=module.padding[0], output_padding=module.output_padding[0])
        else:
            expected = F.conv1d(x, weight, bias, stride=module.stride[0],
                                padding=module.padding[0], dilation=module.dilation[0])
    expected = expected[0].T.bfloat16().float()
    program_address = image.align(128)
    prefix = bytes(image.data) + bytes(program_address - image.base - len(image.data))
    payloads = []
    for fast in (False, True):
        plan['use_sram'] = fast
        capture = conv.shared._WholeGraphEngine(program_address)
        capture.start_capture()
        with contextlib.redirect_stdout(io.StringIO()):
            conv.emit_conv(capture, plan, zero_address=zero)
        capture.generate_instruction_halt()
        capture.stop_capture()
        issues = conv.udc.check_isa_jumps(capture.capture_buffer, program_address, name=label)
        assert not issues, issues
        assert capture._isa_reg_counter == capture._inst_ptr_counter == 1
        kinds = [(inst.words[0] >> 8) & 15 for inst in capture.capture_buffer]
        assert kinds.count(conv.udc.INSTRUCTION_HALT) == 1 and conv.udc.INSTRUCTION_SWI not in kinds
        program = b''.join(inst.get_bytes() for inst in capture.capture_buffer)
        raw = torch.frombuffer(bytearray(prefix + program), dtype=torch.uint8)
        record = dict(case=label, module=name, backend='sram' if fast else 'im2col',
                      shape=list(shape), output_shape=list(plan['output_shape']),
                      instructions=capture.capture_count, resident_bytes=raw.numel(),
                      program_sha256=hashlib.sha256(program).hexdigest())
        payloads.append((raw, record))
    return plan, packed, expected, program_address, payloads


def execute(engine, clock, plan, packed, address, payloads, expected, timeout, records):
    outputs = []
    def write(destination, tensor):
        size = tensor.numel() * tensor.element_size()
        assert engine.dma_write(engine.h2c_device, destination, tensor, size) == size
    for raw, record in payloads:
        assert not engine.is_queue_busy()
        engine.software_reset(run_dram_self_test=False)
        write(conv.shared.MODEL_BASE, raw)
        write(INPUT, packed)
        size = conv.packed_bytes(plan['output_shape'])
        write(OUTPUT, torch.full((size // 2,), float('nan'), dtype=torch.bfloat16))
        write(SCRATCH, torch.full((plan['scratch_bytes'] // 2,), float('nan'), dtype=torch.bfloat16))
        engine.write_reg32(conv.udc.UE_INT_REG, 1)
        began = time.perf_counter()
        engine.start_execute_from_dram(address)
        while engine.read_reg32(conv.udc.UE_INT_REG) & 3 != conv.udc.INT_CAUSE_HALT or engine.is_queue_busy():
            if time.perf_counter() - began > timeout:
                raise TimeoutError(record['case'] + '/' + record['backend'])
            time.sleep(.0001)
        cycles = engine.read_latency_cycles()
        actual = torch.empty(size // 2, dtype=torch.bfloat16)
        assert engine.dma_read(engine.c2h_device, OUTPUT, actual, size) == size
        actual = actual.reshape(plan['output_shape'][0], -1)
        logical = actual[:, :plan['output_shape'][1]].float()
        record.update(fpga_cycles=cycles, fpga_s=cycles * clock / 1e9,
                      finite=bool(torch.isfinite(actual).all()),
                      padded_channels_zero=bool((actual[:, plan['output_shape'][1]:] == 0).all()),
                      cpu_bf16_relative_l2=float(torch.linalg.norm(logical - expected) / torch.linalg.norm(expected)))
        outputs.append(actual.clone())
        torch.save(actual, HERE / (record['case'] + '_' + record['backend'] + '.pt'))
        records.append(record)
        (HERE / 'conv_sram_smoke.json').write_text(json.dumps(records, indent=2) + '\n')
        print(json.dumps(record), flush=True)
        assert record['finite'] and record['padded_channels_zero'], record
    mismatch = int((outputs[0].view(torch.int16) != outputs[1].view(torch.int16)).sum())
    payloads[1][1].update(bit_mismatches=mismatch,
                          speedup=payloads[0][1]['fpga_s'] / payloads[1][1]['fpga_s'])
    (HERE / 'conv_sram_smoke.json').write_text(json.dumps(records, indent=2) + '\n')
    assert mismatch == 0, payloads[1][1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--case', choices=('stem', 'small_tile_cross', 'deep_n32_tile_cross',
                                           'deep_dilation9', 'transpose_tile_cross'))
    parser.add_argument('--expected-version', type=lambda value: int(value, 0), default=0xDF0749DE)
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if 6 in os.sched_getaffinity(0):
        os.sched_setaffinity(0, {6})
    conv.udc.UE_AXI_DATA_WIDTH_BITS = 256
    encoder, decoder = load_models(remove_weight_norm=True)
    prepared = [prepare(*case) for case in cases(encoder, decoder) if args.case is None or case[0] == args.case]
    if not args.execute:
        print(json.dumps([record for item in prepared for _, record in item[-1]], indent=2))
        return
    records = []
    with open('/tmp/pcie_ci_hw_italy.lock', 'r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        clock, info, _ = configure_hardware_runtime(device='rk', dev='xdma0', cycle_override_ns=None)
        assert info.axi_data_width_bits == 256
        with StreamingEngine(clock_period_ns=clock, conv_geometry_mode=conv.udc.CONV_GEOMETRY_QUEUE_CONFIG) as engine:
            assert engine.get_hardware_version() == args.expected_version
            for plan, packed, expected, address, payloads in prepared:
                execute(engine, clock, plan, packed, address, payloads, expected, args.timeout, records)


if __name__ == '__main__':
    main()
