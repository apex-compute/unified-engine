"""Exhaustive native SRAM tests and real BigCodec operator comparisons.

--capture-only compiles/checks every case without opening a device.
--execute is required for hardware access; the script obtains Italy's lock
and verifies the requested FPGA version and AXI256 interface before upload.
"""
from dataclasses import dataclass, replace
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

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'models/bigcodec'))
from bigcodec_device import shared, udc, sram_copy, sram_maximum, clamp_wide
from bigcodec_activation import prepare_activation, emit_activation, emit_activation_dram, activation_scratch_bytes
from bigcodec_quantizer import prepare_quantizer, emit_quantizer, emit_positive_mask, decode_split_tokens
from bigcodec_common import load_models
from bigcodec_precompiled import StreamingEngine
from yolov5_common import configure_hardware_runtime

HERE = Path(__file__).resolve().parent
SOURCE, SECOND, OUTPUT, SCRATCH = 0x80000000, 0x80040000, 0xB0000000, 0xB1000000


@dataclass
class Case:
    name: str
    image: object
    callback: object
    inputs: list
    count: int
    expected: torch.Tensor | None = None
    classification: torch.Tensor | None = None
    exact_bits: bool = False
    pair: str | None = None
    shape: tuple | None = None
    logical_channels: int | None = None
    token_offset: int | None = None
    scratch_bytes: int = 0


def image():
    return shared._ImageBuilder(shared.MODEL_BASE, shared.MODEL_LIMIT)


def finite_encodings():
    values = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
    values[~torch.isfinite(values)] = 0
    return values


def primitives():
    values = finite_encodings()
    other = torch.roll(values, 17311)
    for label, destination in (('a', 0), ('b', 0x80000)):
        def maximum(engine, destination=destination):
            engine.accelerator_memory_to_sram(SOURCE, 0, 65536)
            engine.accelerator_memory_to_sram(SECOND, 0x30000, 65536)
            sram_maximum(engine, 0, 0x30000, destination, 65536)
            engine.sram_to_accelerator_memory(destination, OUTPUT, 65536)
        expected = torch.maximum(values, other)
        # Binary max must be classified by its selected result: a normal
        # left operand may lose to a subnormal right operand.
        yield Case('maximum_' + label, image(), maximum,
                   [(SOURCE, values), (SECOND, other)], 65536,
                   expected, expected)
    for label, destination in (('a', 0x30000), ('b', 0x80000)):
        def copy(engine, destination=destination):
            engine.accelerator_memory_to_sram(SOURCE, 0, 65536)
            sram_copy(engine, 0, destination, 65536)
            engine.sram_to_accelerator_memory(destination, OUTPUT, 65536)
        yield Case('copy_all_finite_bits_' + label, image(), copy,
                   [(SOURCE, values)], 65536, values, values, exact_bits=True)
    for label, lo, hi in (('zero_one', 0., 1.), ('plus_minus_four', -4., 4.)):
        def clamp(engine, lo=lo, hi=hi):
            clamp_wide(engine, SOURCE, OUTPUT, 65536, lo, hi)
        yield Case('clamp_' + label, image(), clamp, [(SOURCE, values)],
                   65536, values.clamp(lo, hi), values)
    constants = image()
    identity = constants.allocate(torch.eye(64, dtype=torch.bfloat16), alignment=128)
    def mask(engine):
        emit_positive_mask(engine, source=SOURCE, destination=OUTPUT, elements=65536,
                           identity_address=identity)
    yield Case('positive_mask_exhaustive', constants, mask, [(SOURCE, values)],
               65536, (values > 0).bfloat16(), values)


def model_cases(encoder, decoder, selection):
    generator = torch.Generator().manual_seed(1974)
    if selection in ('all', 'activation'):
        module = next(m for m in encoder.modules() if type(m).__name__ == 'Activation1d')
        logical = module.act.alpha.numel()
        for rows in (17, 259):
            generator.manual_seed(1974 + rows)
            constants = image()
            identity = constants.allocate(torch.eye(64, dtype=torch.bfloat16), alignment=128)
            plan = prepare_activation(module, constants, input_shape=(rows, logical),
                input_address=SOURCE, output_address=OUTPUT, scratch_address=SCRATCH,
                identity_address=identity)
            packed = torch.zeros(rows, plan.width, dtype=torch.bfloat16)
            packed[:, :logical] = torch.randn(rows, logical, generator=generator) * .2
            pair = f'activation_{rows}'
            for label, emitter in (('dram', emit_activation_dram), ('sram', emit_activation)):
                def callback(engine, plan=plan, emitter=emitter):
                    emitter(engine, plan)
                yield Case(pair + '_' + label, constants, callback, [(SOURCE, packed)],
                           packed.numel(), pair=pair, shape=tuple(packed.shape),
                           logical_channels=logical,
                           scratch_bytes=activation_scratch_bytes((rows, logical)))
    if selection in ('all', 'quantizer'):
        frames = 3
        constants = image()
        offset = frames * 1024
        plan = prepare_quantizer(decoder.quantizer, constants, source_address=SOURCE,
            destination_address=OUTPUT, token_address=OUTPUT + offset * 2,
            scratch_address=SCRATCH, frames=frames)
        generator.manual_seed(3003)
        features = (torch.randn(frames, 1024, generator=generator) * .2).bfloat16()
        for fast in (False, True):
            selected = replace(plan, use_sram_tournament=fast)
            def callback(engine, selected=selected):
                emit_quantizer(engine, selected)
            yield Case('quantizer_' + ('sram' if fast else 'dram'), constants,
                       callback, [(SOURCE, features)], offset + frames * 64,
                       pair='quantizer', token_offset=offset, scratch_bytes=plan.scratch_bytes)


def compile_case(case):
    address = case.image.align(128)
    engine = shared._WholeGraphEngine(address)
    engine.start_capture()
    with contextlib.redirect_stdout(io.StringIO()):
        case.callback(engine)
    engine.generate_instruction_halt()
    engine.stop_capture()
    issues = udc.check_isa_jumps(engine.capture_buffer, address, name=case.name)
    assert not issues, issues
    assert engine._isa_reg_counter == engine._inst_ptr_counter == 1
    kinds = [(inst.words[0] >> 8) & 15 for inst in engine.capture_buffer]
    assert kinds.count(udc.INSTRUCTION_HALT) == 1 and udc.INSTRUCTION_SWI not in kinds
    program = b''.join(inst.get_bytes() for inst in engine.capture_buffer)
    prefix = bytes(case.image.data) + bytes(address - case.image.base - len(case.image.data))
    raw = torch.frombuffer(bytearray(prefix + program), dtype=torch.uint8)
    record = dict(name=case.name, instructions=engine.capture_count,
                  resident_bytes=raw.numel(), program_sha256=hashlib.sha256(program).hexdigest(),
                  image_sha256=hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
                  input_sha256=[hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                                for _, value in case.inputs])
    return address, raw, record


def mismatches(actual, expected, source):
    actual, expected, source = actual.flatten(), expected.flatten(), source.flatten()
    normal = source.abs() >= torch.finfo(torch.bfloat16).tiny
    subnormal = (source != 0) & ~normal
    zero_input = source == 0
    unequal = actual != expected
    unequal_bits = actual.view(torch.int16) != expected.view(torch.int16)
    both_zero = (actual == 0) & (expected == 0)
    return dict(value_mismatches=int(unequal.sum()), bit_mismatches=int(unequal_bits.sum()),
                normal_value_mismatches=int((unequal & normal).sum()),
                normal_nonzero_bit_mismatches=int((unequal_bits & normal & ~both_zero).sum()),
                subnormal_value_mismatches=int((unequal & subnormal).sum()),
                subnormal_bit_mismatches=int((unequal_bits & subnormal).sum()),
                zero_input_value_mismatches=int((unequal & zero_input).sum()),
                signed_zero_bit_mismatches=int((unequal_bits & both_zero).sum()),
                input_positive_subnormals=int((subnormal & (source > 0)).sum()),
                input_negative_subnormals=int((subnormal & (source < 0)).sum()))


def maximum_diagnostics(actual, expected, *, copy=False):
    """Recognize only the measured 0x40519e0a tiny-result flush exception.

    Exhaustive maximum_a selected every positive/negative BF16 subnormal.
    Mantissas1..63 became matching-sign zero; mantissas64..127 and every
    normal result remained bit-exact. Independent multiply-one copy tests
    on both SRAM banks reproduced exactly the same exception.
    """
    actual, expected = actual.flatten(), expected.flatten()
    actual_bits = actual.view(torch.int16).to(torch.int32) & 0xFFFF
    expected_bits = expected.view(torch.int16).to(torch.int32) & 0xFFFF
    magnitude = expected_bits & 0x7FFF
    tiny_result = (magnitude > 0) & (magnitude < 0x40)
    allowed_flush = tiny_result & (actual_bits == (expected_bits & 0x8000))
    unequal = actual != expected
    unequal_bits = actual_bits != expected_bits
    return dict(
        mismatch_classification=('copy source' if copy else
                                 'selected maximum result, not left operand'),
        ieee_value_comparison_pass=not bool(unequal.any()),
        ieee_bit_comparison_pass=not bool(unequal_bits.any()),
        observed_tiny_subnormal_flushes=int((unequal & allowed_flush).sum()),
        unexplained_value_mismatches=int((unequal & ~allowed_flush).sum()),
        unexplained_bit_mismatches=int((unequal_bits & ~allowed_flush).sum()),
        selected_normal_value_mismatches=int((unequal & (magnitude >= 0x80)).sum()),
        selected_upper_subnormal_value_mismatches=int(
            (unequal & (magnitude >= 0x40) & (magnitude < 0x80)).sum()),
        subnormal_exception='Only selected BF16 magnitudes bits0x0001..0x003f '
            'may flush to matching-sign zero; all other results stay exact.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--capture-only', action='store_true')
    parser.add_argument('--selection', choices=('all', 'primitives', 'activation', 'quantizer'), default='all')
    parser.add_argument('--expected-version', type=lambda text: int(text, 0), default=0xDF0749DE)
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if 6 in os.sched_getaffinity(0):
        os.sched_setaffinity(0, {6})
    udc.UE_AXI_DATA_WIDTH_BITS = 256
    cases = list(primitives()) if args.selection in ('all', 'primitives') else []
    if args.selection != 'primitives':
        encoder, decoder = load_models(remove_weight_norm=True)
        cases.extend(model_cases(encoder, decoder, args.selection))
    prepared = [(case, *compile_case(case)) for case in cases]
    if args.capture_only:
        print(json.dumps([record for _, _, _, record in prepared], indent=2))
        return
    records, pairs = [], {}
    def save():
        (HERE / 'native_sram_smoke.json').write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
    with open('/tmp/pcie_ci_hw_italy.lock', 'r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        clock, info, _ = configure_hardware_runtime(device='rk', dev='xdma0', cycle_override_ns=None)
        assert info.axi_data_width_bits == 256
        with StreamingEngine(clock_period_ns=clock, conv_geometry_mode=udc.CONV_GEOMETRY_QUEUE_CONFIG) as engine:
            version = engine.get_hardware_version()
            assert version == args.expected_version
            def write(address, tensor):
                tensor = tensor.contiguous()
                size = tensor.numel() * tensor.element_size()
                assert engine.dma_write(engine.h2c_device, address, tensor, size) == size
            for case, address, raw, record in prepared:
                assert not engine.is_queue_busy()
                engine.software_reset(run_dram_self_test=False)
                write(shared.MODEL_BASE, raw)
                for destination, values in case.inputs:
                    write(destination, values)
                write(OUTPUT, torch.full((case.count,), float('nan'), dtype=torch.bfloat16))
                if case.scratch_bytes:
                    write(SCRATCH, torch.full((case.scratch_bytes // 2,), float('nan'), dtype=torch.bfloat16))
                engine.write_reg32(udc.UE_INT_REG, 1)
                began = time.perf_counter()
                engine.start_execute_from_dram(address)
                while engine.read_reg32(udc.UE_INT_REG) & 3 != udc.INT_CAUSE_HALT or engine.is_queue_busy():
                    if time.perf_counter() - began > args.timeout:
                        raise TimeoutError(case.name)
                    time.sleep(.0001)
                cycles = engine.read_latency_cycles()
                actual = torch.empty(case.count, dtype=torch.bfloat16)
                assert engine.dma_read(engine.c2h_device, OUTPUT, actual, case.count * 2) == case.count * 2
                record.update(hardware_version=f'0x{version:08x}', clock_ns=clock,
                              axi_data_width_bits=256, fpga_cycles=cycles,
                              fpga_s=cycles * clock / 1e9, finite=bool(torch.isfinite(actual).all()))
                torch.save(actual, HERE / (case.name + '.pt'))
                records.append(record)
                save()
                assert record['finite'], record
                if case.expected is not None:
                    record.update(mismatches(actual, case.expected, case.classification))
                    record['nonfinite_input_patterns_replaced_with_zero'] = 256
                    save()
                    if case.exact_bits:
                        record.update(maximum_diagnostics(actual, case.expected, copy=True))
                        save()
                        assert record['unexplained_bit_mismatches'] == 0, record
                    elif case.name.startswith('maximum_'):
                        record.update(maximum_diagnostics(actual, case.expected))
                        save()
                        assert record['unexplained_value_mismatches'] == 0, record
                        assert record['unexplained_bit_mismatches'] == 0, record
                    else:
                        assert record['normal_value_mismatches'] == record['normal_nonzero_bit_mismatches'] == 0, record
                        assert record['zero_input_value_mismatches'] == 0, record
                if case.shape is not None:
                    physical = actual.reshape(case.shape)
                    record['padded_channels_zero'] = bool((physical[:, case.logical_channels:] == 0).all())
                    assert record['padded_channels_zero'], record
                if case.token_offset is not None:
                    record['tokens'] = decode_split_tokens(actual[case.token_offset:].reshape(-1, 64)).tolist()
                if case.pair is not None:
                    if case.pair not in pairs:
                        pairs[case.pair] = (actual.clone(), record)
                    else:
                        baseline, baseline_record = pairs[case.pair]
                        record['values_identical_to_dram'] = bool(torch.equal(actual, baseline))
                        record['bit_mismatches_to_dram'] = int((actual.view(torch.int16) != baseline.view(torch.int16)).sum())
                        record['speedup'] = baseline_record['fpga_s'] / record['fpga_s']
                        if case.token_offset is not None:
                            record['output_features_identical_to_dram'] = bool(torch.equal(actual[:case.token_offset], baseline[:case.token_offset]))
                            record['tokens_identical_to_dram'] = record['tokens'] == baseline_record['tokens']
                        save()
                        assert record['values_identical_to_dram'], record
                save()
                print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
