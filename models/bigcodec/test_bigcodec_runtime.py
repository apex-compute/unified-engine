"""Malformed-bin, file-protection and single-execution BigCodec runtime tests."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bigcodec_precompiled as runtime
import bigcodec_run_from_bin as runner
from bigcodec_common import CHECKPOINT_SHA256, SAMPLE_RATE, HOP_LENGTH, sha256_file
from bigcodec_device import shared, udc
from bigcodec_layout import LEGACY_LAYOUT, EXTENDED_LAYOUT


def fixture_payload(layout=LEGACY_LAYOUT):
    """Small real captured queue with a synthetic but valid tensor inventory."""
    parameters = udc.URAM_NEAR_FULL_SIZE + 64 * 64 * 2
    offset = (parameters + 127) // 128 * 128
    engine = shared._WholeGraphEngine(layout.model_base + offset)
    engine.start_capture()
    for _ in range(3):
        engine.generate_instruction_nop()
    engine.generate_instruction_halt()
    engine.stop_capture()
    program = b''.join(instruction.get_bytes() for instruction in engine.capture_buffer)
    raw = bytearray(offset) + program
    samples, frames = 200, 1
    wave_bytes = samples * 128
    output_bytes = wave_bytes + frames * 128
    encoded = layout.tensor_base + output_bytes
    quantized = encoded + 1024 * 2
    scratch = quantized + 1024 * 2
    operations = [
        dict(name='encoder', op='conv1d', start=0, stop=1, inputs=['input'], output='encoded'),
        dict(name='quantizer', op='quantizer', start=1, stop=2, inputs=['encoded'], output='quantized'),
        dict(name='decoder', op='conv_transpose1d', start=2, stop=3, inputs=['quantized'], output='waveform'),
    ]

    def tensor(shape, address, first, last):
        width = (shape[1] + 63) // 64 * 64
        return dict(shape=shape, padded_channels=width, address=address,
                    bytes=shape[0] * width * 2, first=first, last=last)

    return dict(format=runtime.FORMAT, model='bigcodec', checkpoint_sha256=CHECKPOINT_SHA256,
                sample_rate=SAMPLE_RATE, hop_length=HOP_LENGTH, native_samples=199,
                full_utterance=True, all_neural_operations_on_device=True,
                hardware=dict(
                    model_base=layout.model_base, model_limit=layout.model_limit,
                    model_image=torch.frombuffer(raw, dtype=torch.uint8).clone(),
                    model_sha256=hashlib.sha256(raw).hexdigest(),
                    program_address=layout.model_base + offset, program_offset=offset,
                    program_size=len(program), program_sha256=hashlib.sha256(program).hexdigest(),
                    parameter_bytes=parameters, instructions=len(engine.capture_buffer), halt_index=3,
                    axi_data_width_bits=256, compiled_samples=samples, code_frames=frames,
                    input_address=layout.input_base, input_bytes=wave_bytes,
                    output_address=layout.tensor_base, output_bytes=output_bytes,
                    tokens_address=layout.tensor_base + wave_bytes, tokens_offset=wave_bytes,
                    tensor_base=layout.tensor_base, tensor_limit=layout.tensor_limit,
                    scratch_address=scratch, scratch_bytes=128, tensor_end=scratch + 128,
                    zero_address=layout.model_base,
                    identity_address=layout.model_base + udc.URAM_NEAR_FULL_SIZE,
                    operations=operations, tensors={
                        'input': tensor([samples, 1], layout.input_base, -1, 0),
                        'encoded': tensor([frames, 1024], encoded, 0, 1),
                        'quantized': tensor([frames, 1024], quantized, 1, 2),
                        'waveform': tensor([samples, 1], layout.tensor_base, 2, 3)}))


def replace_program(payload, kinds):
    h = payload['hardware']
    raw = bytearray(h['model_image'].tolist()[:h['program_offset']])
    program = b''.join(struct.pack('<8I', kind << 8, 0, 0, 0, 0, 0, 0, 0) for kind in kinds)
    raw.extend(program)
    h.update(model_image=torch.frombuffer(raw, dtype=torch.uint8).clone(),
             model_sha256=hashlib.sha256(raw).hexdigest(), program_size=len(program),
             program_sha256=hashlib.sha256(program).hexdigest(), instructions=len(kinds))


class FakeEngine:
    conv_geometry_mode = udc.CONV_GEOMETRY_QUEUE_CONFIG
    h2c_device, c2h_device = 'fake-h2c', 'fake-c2h'

    def __init__(self, payload, failure=None):
        self.h = payload['hardware']
        self.failure = failure
        self.events = []
        self.started = False
        self.status = udc.INT_CAUSE_HALT  # stale completion must be cleared
        self.output = torch.zeros(self.h['output_bytes'] // 2, dtype=torch.bfloat16)
        wave = self.output[:self.h['tokens_offset'] // 2].reshape(-1, 64)
        wave[:, 0] = 0.25
        tokens = self.output[self.h['tokens_offset'] // 2:].reshape(-1, 64)
        tokens[:, 0], tokens[:, 1] = 255, 31

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.events.append('close')

    def is_queue_busy(self):
        self.events.append('busy')
        return self.failure == 'initial_busy' or (self.failure == 'halt_busy' and self.started)

    def dma_write(self, device, address, value, size):
        assert device == self.h2c_device
        name = 'model' if address == self.h['model_base'] else 'input'
        self.events.append(('write', name, address, size))
        if name == 'input':
            self.packed_input = value.clone()
        return size - 1 if self.failure == f'{name}_short' else size

    def write_reg32(self, address, value):
        assert address == udc.UE_INT_REG and value == 1
        self.events.append('clear')
        self.status = 0

    def start_execute_from_dram(self, address):
        assert address == self.h['program_address'] and self.status == 0
        self.events.append('start')
        self.started = True
        if self.failure != 'no_halt':
            self.status = udc.INT_CAUSE_HALT

    def read_reg32(self, address):
        assert address == udc.UE_INT_REG
        self.events.append('irq')
        return self.status

    def read_latency_cycles(self):
        self.events.append('cycles')
        return 1600

    def dma_read(self, device, address, output, size):
        assert device == self.c2h_device and address == self.h['output_address']
        self.events.append(('read', address, size))
        output.copy_(self.output)
        return size - 1 if self.failure == 'output_short' else size

    def software_reset(self, *, run_dram_self_test):
        assert run_dram_self_test is False
        self.events.append('reset')

    def get_hardware_version(self):
        return 0xdf0749de


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_valid_small_artifact_loads_without_model_checkpoint(self):
        payload = fixture_payload()
        self.assertIs(runtime.validate_artifact(payload), payload['hardware'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.bin'
            torch.save(payload, path)
            actual = runtime.load_artifact(path)
            self.assertEqual(actual['hardware']['model_sha256'], payload['hardware']['model_sha256'])

    def test_extended_layout_loads_and_executes_with_the_existing_waveform_abi(self):
        payload = fixture_payload(EXTENDED_LAYOUT)
        h = runtime.validate_artifact(payload)
        self.assertEqual(h['model_limit'], 0xD0000000)
        self.assertEqual(h['output_address'], 0xD0000000)
        self.assertEqual(h['tensor_limit'], 0x100000000)
        self.assertNotIn('memory_layout', payload)  # Bounds alone identify both formats.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'extended.bin'
            torch.save(payload, path)
            loaded = runtime.load_artifact(path)
        engine = FakeEngine(loaded)
        backend = runtime.WholeGraphBackend(engine, loaded, axi_data_width_bits=256)
        wave, tokens = backend.execute(np.zeros(200, dtype=np.float32))
        np.testing.assert_array_equal(wave, np.full(200, .25, dtype=np.float32))
        np.testing.assert_array_equal(tokens, [8191])
        self.assertIn(('read', 0xD0000000, 201 * 128), engine.events)
        self.assertEqual(backend.model_upload_writes, 1)
        self.assertEqual(backend.program_kicks, 1)
        self.assertEqual(backend.halts, 1)
        self.assertEqual(backend.output_reads, 1)

    def test_layouts_accept_their_last_byte_and_reject_any_tensor_overrun(self):
        for layout in (LEGACY_LAYOUT, EXTENDED_LAYOUT):
            with self.subTest(layout=layout.name):
                payload = fixture_payload(layout)
                h = payload['hardware']
                h.update(scratch_address=layout.tensor_limit - 128,
                         scratch_bytes=128, tensor_end=layout.tensor_limit)
                h['tensors']['encoded']['address'] = h['scratch_address'] - 4096
                h['tensors']['quantized']['address'] = h['scratch_address'] - 2048
                runtime.validate_artifact(payload)
                h['scratch_bytes'] += 128
                h['tensor_end'] += 128
                with self.assertRaisesRegex(ValueError, 'DRAM bounds'):
                    runtime.validate_artifact(payload)
                h['scratch_bytes'] -= 128
                h['tensor_end'] -= 128
                h['tensors']['quantized']['address'] = layout.tensor_limit - 128
                with self.assertRaisesRegex(ValueError, 'workspace'):
                    runtime.validate_artifact(payload)

    def test_mixed_layouts_and_bounds_outside_visible_two_gib_are_rejected(self):
        for changes in (
                {'model_limit': LEGACY_LAYOUT.model_limit},
                {'tensor_base': LEGACY_LAYOUT.tensor_base},
                {'tensor_limit': 0x100000080},
                {'model_base': 0x7FFFFF80},
                {'model_limit': 0x100000000},
                {'input_address': 0x7FFFFF80},
                {'program_address': 0x100000000},
                {'output_address': 0x100000000}):
            with self.subTest(changes=changes):
                payload = fixture_payload(EXTENDED_LAYOUT)
                payload['hardware'].update(changes)
                with self.assertRaises(ValueError):
                    runtime.validate_artifact(payload)

    def test_dma_encodes_upper_visible_addresses_without_signed_truncation(self):
        engine = shared._WholeGraphEngine(0xCFFFFF00)
        with patch.object(udc, 'UE_AXI_DATA_WIDTH_BITS', 256), contextlib.redirect_stdout(io.StringIO()):
            engine.start_capture()
            engine.accelerator_memory_to_sram(0xFFFFFF80, 0, 64)
            engine.sram_to_accelerator_memory(0, 0xFFFFFF00, 64)
            engine.generate_instruction_halt()
            engine.stop_capture()
        for instruction, address in zip(engine.capture_buffer, (0xFFFFFF80, 0xFFFFFF00)):
            self.assertEqual(instruction.words[1] << 3, address)
            self.assertEqual(instruction.words[2], 128)
            self.assertEqual(len(instruction.get_bytes()), 32)
        # Exercise the host offset plumbing without opening a device.
        engine = object.__new__(udc.UnifiedEngine)
        buffer = torch.zeros(64, dtype=torch.bfloat16)
        with patch.object(udc.os, 'open', return_value=101), \
             patch.object(udc.os, 'close'), \
             patch.object(udc.os, 'lseek') as seek, \
             patch.object(udc.os, 'write', return_value=128), \
             patch.object(udc.os, 'read', return_value=bytes(128)):
            self.assertEqual(engine.dma_write('fake-h2c', 0xFFFFFF80, buffer, 128), 128)
            self.assertEqual(engine.dma_read('fake-c2h', 0xFFFFFF80, buffer, 128), 128)
        self.assertEqual(seek.call_args_list[0].args, (101, 0xFFFFFF80, os.SEEK_SET))
        self.assertEqual(seek.call_args_list[1].args, (101, 0xFFFFFF80, os.SEEK_SET))

    def test_detected_capacity_uses_actual_extents_and_preserves_legacy_bins(self):
        legacy = fixture_payload()['hardware']
        # The reserved legacy arena ends above 1 GiB, but this bin uses less.
        self.assertGreater(legacy['tensor_limit'] - udc.DRAM_START_ADDR, 2**30)
        report = runner.validate_dram_capacity(legacy, 1)
        self.assertEqual(report['memory_layout'], 'legacy')
        self.assertEqual(report['dram_addressable_bytes'], 2**30)
        self.assertEqual(report['dram_required_bytes'], legacy['tensor_end'] - udc.DRAM_START_ADDR)
        extended = fixture_payload(EXTENDED_LAYOUT)['hardware']
        with self.assertRaisesRegex(ValueError, 'requires.*reported 1 GiB'):
            runner.validate_dram_capacity(extended, 1)
        report = runner.validate_dram_capacity(extended, 2)
        self.assertEqual(report['memory_layout'], 'extended')
        self.assertEqual(report['detected_dram_size_gib'], 2)
        self.assertEqual(report['dram_required_bytes'], extended['tensor_end'] - udc.DRAM_START_ADDR)

    def test_detected_capacity_checks_all_used_extents_and_caps_visible_addresses(self):
        h = fixture_payload(EXTENDED_LAYOUT)['hardware']
        for key in ('tensor_end', 'model_image', 'input_bytes'):
            with self.subTest(extent=key):
                changed = dict(h)
                if key == 'tensor_end':
                    changed[key] = 0x100000000
                elif key == 'model_image':
                    changed[key] = SimpleNamespace(numel=lambda: 0x100000000 - h['model_base'])
                else:
                    changed[key] = 0x100000000 - h['input_address']
                self.assertEqual(runner.validate_dram_capacity(changed, 4)['dram_required_bytes'], 2**31)
                if key == 'model_image':
                    changed[key] = SimpleNamespace(numel=lambda: 0x100000080 - h['model_base'])
                else:
                    changed[key] += 128
                with self.assertRaisesRegex(ValueError, 'requires'):
                    runner.validate_dram_capacity(changed, 4)
        for bad_report in (0, -1, True, 2.0, None, '2'):
            with self.subTest(reported_capacity=bad_report):
                with self.assertRaisesRegex(ValueError, 'positive integer'):
                    runner.validate_dram_capacity(h, bad_report)

    def test_cli_rejects_insufficient_ddr_before_engine_reset_or_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, binary, output = root / 'source.wav', root / 'codec.bin', root / 'output.wav'
            sf.write(source, np.zeros(199, dtype=np.float32), SAMPLE_RATE, subtype='FLOAT')
            torch.save(fixture_payload(EXTENDED_LAYOUT), binary)
            source_hash, bin_hash = sha256_file(source), sha256_file(binary)
            hostname = f'bigcodec-capacity-test-{os.getpid()}'
            lockfile = Path(f'/tmp/pcie_ci_hw_{hostname}.lock')
            error_output = io.StringIO()
            try:
                with patch.object(sys, 'argv', ['bigcodec', '--bin', str(binary), '--input', str(source),
                                                '--output', str(output), '--cpu-core', '6']), \
                     patch.object(runner, 'StreamingEngine') as engine, \
                     patch.object(runner, 'WholeGraphBackend') as backend, \
                     patch.object(runner, 'configure_hardware_runtime', return_value=(3.0,
                         SimpleNamespace(axi_data_width_bits=256, dram_size_gb=1), 3.0)), \
                     patch.object(runner.socket, 'gethostname', return_value=hostname), \
                     patch.object(runner.os, 'sched_getaffinity', return_value={6}), \
                     patch.object(runner.os, 'sched_setaffinity'), \
                     contextlib.redirect_stderr(error_output):
                    with self.assertRaises(SystemExit) as error:
                        runner.main()
                    self.assertEqual(error.exception.code, 2)
                    engine.assert_not_called()
                    backend.assert_not_called()
            finally:
                lockfile.unlink(missing_ok=True)
            self.assertIn('reported 1 GiB', error_output.getvalue())
            self.assertEqual(source_hash, sha256_file(source))
            self.assertEqual(bin_hash, sha256_file(binary))
            for path in (output, output.with_suffix('.tokens.npz'), output.with_suffix('.metrics.json')):
                self.assertFalse(path.exists())

    def test_reject_malformed_identity_dimensions_and_memory_metadata(self):
        changes = [
            ('checkpoint_sha256', 'wrong', False), ('sample_rate', 8000, False),
            ('hop_length', 160, False), ('native_samples', 200, False),
            ('full_utterance', False, False), ('all_neural_operations_on_device', False, False),
            ('axi_data_width_bits', 512, True), ('model_base', shared.MODEL_BASE + 128, True),
            ('model_limit', shared.MODEL_LIMIT + 128, True), ('model_sha256', 'wrong', True),
            ('program_sha256', 'wrong', True), ('compiled_samples', True, True),
            ('code_frames', 2, True), ('program_offset', 1, True), ('program_size', 96, True),
            ('parameter_bytes', -1, True), ('instructions', 5, True), ('halt_index', 2, True),
            ('input_bytes', 128, True), ('output_bytes', 128, True),
            ('input_bytes', 25600.0, True), ('code_frames', True, True),
            ('tokens_offset', 0, True), ('tokens_address', shared.TENSOR_BASE, True),
            ('tensor_end', shared.TENSOR_LIMIT + 128, True), ('scratch_bytes', -128, True),
            ('scratch_address', shared.TENSOR_BASE, True), ('identity_address', shared.INPUT_BASE, True),
        ]
        for key, value, hardware in changes:
            with self.subTest(key=key):
                payload = fixture_payload()
                (payload['hardware'] if hardware else payload)[key] = value
                with self.assertRaises(ValueError):
                    runtime.validate_artifact(payload)
        for payload in (None, [], {'format': runtime.FORMAT}):
            with self.assertRaises(ValueError):
                runtime.validate_artifact(payload)

    def test_reject_incomplete_queue_and_operation_ranges(self):
        for kinds in ([udc.INSTRUCTION_NOP] * 4,
                      [udc.INSTRUCTION_NOP, udc.INSTRUCTION_HALT, udc.INSTRUCTION_NOP, udc.INSTRUCTION_HALT],
                      [udc.INSTRUCTION_SWI, udc.INSTRUCTION_NOP, udc.INSTRUCTION_NOP, udc.INSTRUCTION_HALT],
                      [udc.INSTRUCTION_NOP, udc.INSTRUCTION_HALT, udc.INSTRUCTION_UE_OP, udc.INSTRUCTION_NOP],
                      [udc.INSTRUCTION_NOP, udc.INSTRUCTION_NOP, udc.INSTRUCTION_HALT],
                      [3, udc.INSTRUCTION_NOP, udc.INSTRUCTION_NOP, udc.INSTRUCTION_HALT]):
            payload = fixture_payload()
            replace_program(payload, kinds)
            with self.assertRaises(ValueError):
                runtime.validate_artifact(payload)
        payload = fixture_payload()
        payload['hardware']['operations'][1]['start'] = 0
        with self.assertRaises(ValueError):
            runtime.validate_artifact(payload)

    def test_reject_live_tensor_alias_and_wrong_output_shape(self):
        for mutate in ('alias', 'shape', 'padding', 'lifetime', 'forward_reference'):
            payload = fixture_payload()
            h = payload['hardware']
            if mutate == 'alias':
                h['tensors']['quantized']['address'] = h['tensors']['encoded']['address']
            elif mutate == 'shape':
                h['tensors']['waveform']['shape'] = [100, 2]
            elif mutate == 'padding':
                h['tensors']['encoded']['padded_channels'] = 64
            elif mutate == 'lifetime':
                h['tensors']['encoded']['last'] = 0
            else:
                h['operations'][0]['inputs'] = ['quantized']
            with self.assertRaises(ValueError, msg=mutate):
                runtime.validate_artifact(payload)

    def test_terminal_halt_may_be_followed_by_fetch_padding_nops(self):
        payload = fixture_payload()
        replace_program(payload, [udc.INSTRUCTION_NOP] * 3 + [udc.INSTRUCTION_HALT,
                                                             udc.INSTRUCTION_NOP, udc.INSTRUCTION_NOP])
        runtime.validate_artifact(payload)

    def test_reject_instruction_corruption_and_out_of_program_absolute_jump(self):
        payload = fixture_payload()
        payload['hardware']['model_image'][0] ^= 1
        with self.assertRaisesRegex(ValueError, 'checksum'):
            runtime.validate_artifact(payload)
        payload = fixture_payload()
        h = payload['hardware']
        engine = shared._WholeGraphEngine(h['program_address'])
        engine.start_capture()
        engine.generate_instruction_jump_abs((h['program_address'] - 64) >> 3)
        raw = bytearray(h['model_image'].numpy().tobytes())
        start = h['program_offset']
        raw[start:start + 32] = engine.capture_buffer[0].get_bytes()
        h['model_image'] = torch.frombuffer(raw, dtype=torch.uint8).clone()
        h['model_sha256'] = hashlib.sha256(raw).hexdigest()
        h['program_sha256'] = hashlib.sha256(raw[start:]).hexdigest()
        with self.assertRaisesRegex(ValueError, 'leaves its resident program'):
            runtime.validate_artifact(payload)

    def test_one_resident_upload_and_one_start_halt_read_per_execution(self):
        payload = fixture_payload()
        engine = FakeEngine(payload)
        backend = runtime.WholeGraphBackend(engine, payload, axi_data_width_bits=256)
        source = np.linspace(-1, 1, 200, dtype=np.float32)
        for _ in range(2):
            waveform, tokens = backend.execute(source)
            np.testing.assert_array_equal(waveform, np.full(200, 0.25, dtype=np.float32))
            np.testing.assert_array_equal(tokens, [8191])
        self.assertEqual(engine.events[:2], ['busy', ('write', 'model', shared.MODEL_BASE,
                                                      payload['hardware']['model_image'].numel())])
        per_execution = ['busy', ('write', 'input', shared.INPUT_BASE, 200 * 128),
                         'clear', 'start', 'irq', 'busy', 'cycles',
                         ('read', shared.TENSOR_BASE, 201 * 128)]
        self.assertEqual(engine.events[2:], per_execution * 2)
        torch.testing.assert_close(engine.packed_input[:, 0], torch.from_numpy(source).bfloat16(), rtol=0, atol=0)
        self.assertEqual(int(torch.count_nonzero(engine.packed_input[:, 1:])), 0)
        self.assertEqual(backend.last_metrics['model_upload_writes'], 1)
        for key in ('input_upload_writes', 'program_kicks', 'halts', 'output_reads'):
            self.assertEqual(backend.last_metrics[key], 2)
        self.assertEqual(backend.last_metrics['cpu_neural_ops'], 0)

    def test_invalid_inputs_do_not_issue_device_commands(self):
        payload = fixture_payload()
        engine = FakeEngine(payload)
        backend = runtime.WholeGraphBackend(engine, payload, axi_data_width_bits=256)
        before = list(engine.events)
        for source in (np.zeros(199), np.zeros((200, 1)), np.full(200, np.nan),
                       np.full(200, np.inf), np.full(200, np.finfo(np.float32).max)):
            with self.assertRaises(ValueError):
                backend.execute(source)
            self.assertEqual(engine.events, before)
        self.assertFalse(backend.failed)

    def test_bad_hardware_and_initial_queue_state_do_not_upload(self):
        for failure in ('width', 'config', 'timeout', 'initial_busy', 'model_short'):
            payload = fixture_payload()
            engine = FakeEngine(payload, failure)
            if failure == 'config':
                engine.conv_geometry_mode = udc.CONV_GEOMETRY_LIVE_CSR
            with self.assertRaises((ValueError, RuntimeError)):
                runtime.WholeGraphBackend(engine, payload,
                    axi_data_width_bits=512 if failure == 'width' else 256,
                    timeout_s=0 if failure == 'timeout' else 1)
            if failure != 'model_short':
                self.assertFalse(any(isinstance(event, tuple) and event[0] == 'write' for event in engine.events))

    def test_any_failed_execution_stops_backend_and_never_reads_early(self):
        for failure in ('input_short', 'output_short', 'no_halt', 'halt_busy', 'nan_audio', 'bad_token'):
            with self.subTest(failure=failure):
                payload = fixture_payload()
                engine = FakeEngine(payload, failure)
                if failure == 'nan_audio':
                    engine.output[0] = float('nan')
                if failure == 'bad_token':
                    engine.output[-64] = 256
                backend = runtime.WholeGraphBackend(engine, payload, axi_data_width_bits=256, timeout_s=.5)
                with patch.object(runtime.time, 'monotonic', side_effect=[0, 1]), patch.object(runtime.time, 'sleep'):
                    with self.assertRaises((RuntimeError, ValueError, FloatingPointError, TimeoutError)):
                        backend.execute(np.zeros(200, dtype=np.float32))
                self.assertTrue(backend.failed)
                before = list(engine.events)
                with self.assertRaisesRegex(RuntimeError, 'stopped'):
                    backend.execute(np.zeros(200, dtype=np.float32))
                self.assertEqual(engine.events, before)
                if failure in ('input_short', 'no_halt', 'halt_busy'):
                    self.assertFalse(any(isinstance(event, tuple) and event[0] == 'read' for event in engine.events))

    def test_padding_is_reported_without_altering_valid_audio(self):
        payload = fixture_payload()
        engine = FakeEngine(payload)
        engine.output[1] = float('nan')
        engine.output[2] = 0.001
        backend = runtime.WholeGraphBackend(engine, payload, axi_data_width_bits=256)
        waveform, _ = backend.execute(np.zeros(200, dtype=np.float32))
        self.assertTrue(np.isfinite(waveform).all())
        self.assertFalse(backend.last_metrics['waveform_padding_finite'])
        self.assertEqual(backend.last_metrics['waveform_padding_nonzero'], 2)

    def test_resident_metadata_is_snapshotted_after_upload(self):
        payload = fixture_payload()
        engine = FakeEngine(payload)
        backend = runtime.WholeGraphBackend(engine, payload, axi_data_width_bits=256)
        original = payload['hardware']['program_address']
        payload['hardware']['program_address'] += 128
        self.assertEqual(backend.h['program_address'], original)

    def test_cli_restores_source_count_and_writes_bound_report_without_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, binary, output = root / 'source.wav', root / 'codec.bin', root / 'output.wav'
            sf.write(source, np.full(199, .5, dtype=np.float32), SAMPLE_RATE, subtype='FLOAT')
            payload = fixture_payload()
            torch.save(payload, binary)
            engine = FakeEngine(payload)
            hostname = f'bigcodec-unit-test-{os.getpid()}'
            lockfile = Path(f'/tmp/pcie_ci_hw_{hostname}.lock')
            try:
                with patch.object(sys, 'argv', ['bigcodec', '--bin', str(binary), '--input', str(source),
                                                '--output', str(output), '--cpu-core', '6']), \
                     patch.object(runner, 'StreamingEngine', return_value=engine), \
                     patch.object(runner, 'configure_hardware_runtime', return_value=(3.0,
                         SimpleNamespace(axi_data_width_bits=256, dram_size_gb=2), 3.0)), \
                     patch.object(runner.socket, 'gethostname', return_value=hostname), \
                     patch.object(runner.os, 'sched_getaffinity', return_value={6}), \
                     patch.object(runner.os, 'sched_setaffinity'), \
                     contextlib.redirect_stdout(io.StringIO()):
                    runner.main()
            finally:
                lockfile.unlink(missing_ok=True)
            audio, rate = sf.read(output, dtype='float32')
            self.assertEqual((audio.size, rate), (199, SAMPLE_RATE))
            np.testing.assert_array_equal(audio, np.full(199, .25, dtype=np.float32))
            report = json.loads(output.with_suffix('.metrics.json').read_text())
            self.assertEqual(report['compiled_samples'], 200)
            self.assertEqual(report['source_samples'], report['output_samples'])
            self.assertEqual(report['tokens'], 1)
            self.assertEqual(report['bin_sha256'], sha256_file(binary))
            self.assertEqual(report['input_sha256'], sha256_file(source))
            self.assertEqual(report['output_sha256'], sha256_file(output))
            self.assertEqual(report['hardware_version'], '0xdf0749de')
            self.assertEqual(report['cpu_neural_ops'], 0)
            self.assertEqual(report['memory_layout'], 'legacy')
            self.assertEqual(report['detected_dram_size_gib'], 2)
            self.assertEqual(report['dram_addressable_bytes'], 2**31)
            self.assertEqual(report['dram_required_bytes'], payload['hardware']['tensor_end'] - udc.DRAM_START_ADDR)

    def test_cli_rejects_wrong_or_changing_build_without_publishing_audio(self):
        for expected, versions, message in (
                ('0x90f1f464', [0xdf0749de], 'differs from required'),
                ('0xdf0749de', [0xdf0749de, 0x90f1f464], 'changed during execution')):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, binary, output = root/'source.wav', root/'codec.bin', root/'output.wav'
                sf.write(source, np.full(199, .5, dtype=np.float32), SAMPLE_RATE, subtype='FLOAT')
                payload = fixture_payload()
                torch.save(payload, binary)
                engine = FakeEngine(payload)
                hostname = f'bigcodec-build-test-{os.getpid()}'
                lockfile = Path(f'/tmp/pcie_ci_hw_{hostname}.lock')
                try:
                    with patch.object(sys, 'argv', ['bigcodec', '--bin', str(binary), '--input', str(source),
                            '--output', str(output), '--cpu-core', '6', '--expected-version', expected]), \
                         patch.object(runner, 'StreamingEngine', return_value=engine), \
                         patch.object(engine, 'get_hardware_version', side_effect=versions), \
                         patch.object(runner, 'configure_hardware_runtime', return_value=(3.0,
                             SimpleNamespace(axi_data_width_bits=256, dram_size_gb=2), 3.0)), \
                         patch.object(runner.socket, 'gethostname', return_value=hostname), \
                         patch.object(runner.os, 'sched_getaffinity', return_value={6}), \
                         patch.object(runner.os, 'sched_setaffinity'), \
                         contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaisesRegex(RuntimeError, message):
                            runner.main()
                finally:
                    lockfile.unlink(missing_ok=True)
                self.assertFalse(output.exists())
                self.assertFalse(output.with_suffix('.tokens.npz').exists())
                self.assertFalse(output.with_suffix('.metrics.json').exists())
                if len(versions) == 1:
                    self.assertNotIn('reset', engine.events)
                    self.assertFalse(any(isinstance(event, tuple) and event[0] == 'write'
                                         for event in engine.events))


class FileProtectionTests(unittest.TestCase):
    def test_resolved_symlink_hardlink_and_checkpoint_aliases_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, binary, checkpoint = root / 'input.wav', root / 'codec.bin', root / 'checkpoint.pt'
            for path in (source, binary, checkpoint):
                path.write_bytes(b'preserve')
            defaults = [source, binary, root / 'output.wav', root / 'output.npz', root / 'report.json']
            with patch.object(runner, 'DEFAULT_CHECKPOINT', checkpoint):
                runner.validate_output_paths(*defaults)
                for protected in (source, binary, checkpoint):
                    for kind in ('same', 'symlink', 'hardlink'):
                        alias = root / f'{protected.name}.{kind}'
                        if kind == 'symlink':
                            alias.symlink_to(protected)
                        elif kind == 'hardlink':
                            os.link(protected, alias)
                        else:
                            alias = protected
                        paths = list(defaults)
                        paths[3] = alias
                        with self.assertRaises(ValueError, msg=f'{protected.name} {kind}'):
                            runner.validate_output_paths(*paths)
                        self.assertEqual(protected.read_bytes(), b'preserve')


if __name__ == '__main__':
    unittest.main()
