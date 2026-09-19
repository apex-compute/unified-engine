"""Independent alias-filter geometry and BF16 activation memory execution."""

import contextlib
from dataclasses import replace
import io
import hashlib
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bigcodec_activation import (SNAKE_ARGUMENT_LIMIT, activation_scratch_bytes,
                                _ordered_filter_taps, _sine_polynomial_sram,
                                emit_activation, emit_activation_dram, prepare_activation,
                                quantize_filter_dc, sine_squared)
from bigcodec_device import (channels, clamp, clamp_wide, gather_rows, scale,
                            scatter_rows, shift, sram_clamp, sram_copy,
                            sram_maximum, udc)
from bigcodec_vq.activations import SnakeBeta
from bigcodec_vq.alias_free_torch.act import Activation1d
from test_bigcodec_quantizer import MemoryEngine as QuantizerMemoryEngine
from test_bigcodec_lstm import NativeArithmeticMemoryEngine, native_round
from yolov5_precompiled import _WholeGraphEngine, _instruction_types


class MemoryEngine(QuantizerMemoryEngine):
    """Extend the independent BF16 interpreter with byte/strided DMAs."""

    def __init__(self):
        super().__init__()
        self.transfers = []
        self.watch = None
        self.snapshots = {}
        self._isa_reg_counter = 1
        self.registers = {}
        self.dynamic_clamps = 0

    def alloc_isa_reg(self):
        value = self._isa_reg_counter
        self._isa_reg_counter += 1
        return value

    def generate_instruction_add_set(self, register, value):
        self.registers[register] = value

    def release_isa_reg(self):
        self._isa_reg_counter -= 1
        self.registers.pop(self._isa_reg_counter)

    def matmat_mul_core(self, *, gpr_M_reg=None, **kwargs):
        if gpr_M_reg is not None:
            assert self.registers[gpr_M_reg] == kwargs["M"]
            self.dynamic_clamps += 1
        return super().matmat_mul_core(**kwargs)

    def _dma(self, source, destination, elements, *, memcpy_length_bytes,
             stride_bytes_per_chunk, stride_jump_bytes, read):
        size = elements * 2 if memcpy_length_bytes is None else memcpy_length_bytes
        sram = destination if read else source
        assert size > 0 and size % 128 == sram % 128 == 0
        assert sram % 0x80000 + size <= udc.URAM_NEAR_FULL_SIZE
        chunk = stride_bytes_per_chunk or size
        jump = stride_jump_bytes or size
        assert chunk % 128 == jump % 128 == 0 and size % chunk == 0
        if self.watch is not None and read and source == self.watch.alpha and 'upsampled' not in self.snapshots:
            self.snapshots['upsampled'] = self.view(self.watch.scratch, 2 * self.watch.rows * self.watch.width).float().clone()
        for i in range(size // chunk):
            if read:
                value = self.view(source + i * jump, chunk // 2)
                self.sram[(destination + i * chunk) // 2:][:chunk // 2].copy_(value)
            else:
                value = self.sram[(source + i * chunk) // 2:][:chunk // 2]
                self.view(destination + i * jump, chunk // 2).copy_(value)
        self.transfers.append((read, source, destination, size, chunk, jump))
        if (self.watch is not None and not read and source == 0
                and self.watch.scratch <= destination < self.watch.scratch + 4 * self.watch.rows * self.watch.width):
            self.snapshots['snake'] = self.view(self.watch.scratch, 2 * self.watch.rows * self.watch.width).float().clone()

    def accelerator_memory_to_sram(self, source, destination, elements, *,
                                   memcpy_length_bytes=None,
                                   stride_bytes_per_chunk=None,
                                   stride_jump_bytes=None):
        self._dma(source, destination, elements, memcpy_length_bytes=memcpy_length_bytes,
                  stride_bytes_per_chunk=stride_bytes_per_chunk,
                  stride_jump_bytes=stride_jump_bytes, read=True)

    def sram_to_accelerator_memory(self, source, destination, elements, *,
                                   memcpy_length_bytes=None,
                                   stride_bytes_per_chunk=None,
                                   stride_jump_bytes=None):
        self._dma(source, destination, elements, memcpy_length_bytes=memcpy_length_bytes,
                  stride_bytes_per_chunk=stride_bytes_per_chunk,
                  stride_jump_bytes=stride_jump_bytes, read=False)

    def eltwise_core_dram(self, *, M, N, dram_a, dram_b, dram_out, mode, scalar=None):
        if self.watch is not None:
            plan = self.watch
            if dram_b == plan.alpha and "upsampled" not in self.snapshots:
                self.snapshots["upsampled"] = self.view(plan.scratch, 2 * plan.rows * plan.width).float().clone()
        if mode == udc.UE_MODE.ADD_BROADCAST:
            assert dram_b is None
            result = self.view(dram_a, M * N).float() + scalar
            self.view(dram_out, M * N).copy_(result)
        else:
            super().eltwise_core_dram(M=M, N=N, dram_a=dram_a, dram_b=dram_b,
                                     dram_out=dram_out, mode=mode, scalar=scalar)
        if (self.watch is not None and dram_out == self.watch.scratch
                and mode == udc.UE_MODE.ELTWISE_ADD):
            self.snapshots["snake"] = self.view(dram_out, 2 * self.watch.rows * self.watch.width).float().clone()


class NativeMemoryEngine(MemoryEngine):
    """Use the separately validated native ALU model for both memory paths."""
    broadcast_mul = NativeArithmeticMemoryEngine.broadcast_mul
    broadcast_add = NativeArithmeticMemoryEngine.broadcast_add
    _binary = NativeArithmeticMemoryEngine._binary
    eltwise_mul_core = NativeArithmeticMemoryEngine.eltwise_mul_core
    eltwise_add_core = NativeArithmeticMemoryEngine.eltwise_add_core
    eltwise_sub_core = NativeArithmeticMemoryEngine.eltwise_sub_core

    def eltwise_core_dram(self, *, M, N, dram_a, dram_b, dram_out, mode, scalar=None):
        left = self.view(dram_a, M * N).float()
        if mode in (udc.UE_MODE.MUL_BROADCAST, udc.UE_MODE.ADD_BROADCAST):
            assert dram_b is None
            result = left * scalar if mode == udc.UE_MODE.MUL_BROADCAST else left + scalar
        else:
            right = self.view(dram_b, M * N).float()
            operation = {udc.UE_MODE.ELTWISE_MUL: torch.mul,
                         udc.UE_MODE.ELTWISE_ADD: torch.add,
                         udc.UE_MODE.ELTWISE_SUB: torch.sub}[mode]
            result = operation(left, right)
        self.view(dram_out, M * N).copy_(native_round(result))


def fixture(length, logical_channels, *, filter_accumulation=None,
            snake_polynomial=None, filter_stage=None, engine_type=MemoryEngine):
    module = Activation1d(SnakeBeta(logical_channels, alpha_logscale=True))
    generator = torch.Generator().manual_seed(728)
    with torch.no_grad():
        module.act.alpha.copy_(torch.randn(logical_channels, generator=generator) * .15)
        module.act.beta.copy_(torch.randn(logical_channels, generator=generator) * .15)
    module.eval().requires_grad_(False)
    engine = engine_type()
    identity = engine.allocate(torch.eye(64, dtype=torch.bfloat16))
    width = channels(logical_channels)
    source, destination, scratch = 0xB0000000, 0xB1000000, 0xB2000000
    options = {}
    if filter_accumulation is not None:
        options['filter_accumulation'] = filter_accumulation
    if filter_stage is not None:
        options['filter_stage'] = filter_stage
    scratch_bytes = activation_scratch_bytes((length, logical_channels), **options)
    plan = prepare_activation(module, engine, input_shape=(length, logical_channels),
                              input_address=source, output_address=destination,
                              scratch_address=scratch, identity_address=identity,
                              workspace_bytes=scratch_bytes, **options,
                              **({} if snake_polynomial is None else
                                 {'snake_polynomial': snake_polynomial}))
    for address, count in ((source, length * width), (destination, length * width),
                           (scratch, scratch_bytes // 2)):
        engine.regions[address] = torch.full((count,), float("nan"), dtype=torch.bfloat16)
    packed = torch.zeros(length, width, dtype=torch.bfloat16)
    packed[:, :logical_channels] = torch.randn(length, logical_channels, generator=generator) * .35
    engine.view(source, packed.numel()).copy_(packed.flatten())
    engine.watch = plan
    return engine, plan, module, packed.float()


def matrix_engine_type():
    # Import after this module initializes: the independent FIR test engine
    # extends the shared memory interpreter above. No experimental imports.
    from test_bigcodec_filter import FilterEngine

    class MatrixActivationEngine(FilterEngine, NativeMemoryEngine):
        """FP64 dot geometry with the measured native pointwise rounding."""

    return MatrixActivationEngine


def filter_reference(source, taps, kind, *, matrix):
    """Independent grouped-convolution equation with explicit store rounding."""
    from test_bigcodec_filter import reference
    if matrix:
        high = taps.bfloat16().float()
        effective = high.double() + (taps - high).bfloat16().double()
        return reference(source, effective, kind).bfloat16().float()
    coefficients = quantize_filter_dc(tuple(taps.tolist()))
    terms = []
    for index, coefficient in enumerate(coefficients):
        kernel = torch.zeros(12, dtype=torch.float64)
        kernel[index] = coefficient
        terms.append(native_round(reference(source, kernel, kind).float()))
    if kind == 'down':
        result = terms[0]
        for term in terms[1:]:
            result = native_round(result + term)
        return result
    result = torch.zeros_like(terms[0])
    for phase in (0, 1):
        nonzero = [terms[tap][phase::2] for tap in range(12)
                   if (phase + 15 - tap) % 2 == 0]
        value = nonzero[0]
        for term in nonzero[1:]:
            value = native_round(value + term)
        result[phase::2] = value
    return result


def native_snake_reference(source, module):
    """Scalar formula, independent of the emitted SRAM allocation/schedule."""
    q = native_round
    scalar = lambda x: float(torch.tensor(x).bfloat16())
    alpha = torch.zeros(source.shape[1])
    inverse = torch.zeros_like(alpha)
    logical = module.act.alpha.numel()
    alpha[:logical] = module.act.alpha.detach().exp().bfloat16().float()
    inverse[:logical] = (1 / (module.act.beta.detach().exp() + 1e-9)).bfloat16().float()
    theta = q(source * alpha).abs().clamp(max=scalar(SNAKE_ARGUMENT_LIMIT))
    for exponent in range(5, -1, -1):
        delta = q(theta - scalar((math.pi / 2) * 2 ** exponent)).clamp_min(0)
        theta = q(theta - q(2 * delta))
    square = q(theta * theta)
    value = q(square * scalar(2 / 14175))
    for coefficient in (-1 / 315, 2 / 45, -1 / 3, 1.):
        value = q(q(value + scalar(coefficient)) * square)
    return q(q(value.clamp(0, 1) * inverse) + source)


class ActivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gather_clamps_endpoints_and_crosses_sram_chunks(self):
        for source_rows, rows, width, start, stride in (
                (1, 17, 64, -8, 1), (2, 17, 128, -11, 2),
                (17, 19, 64, -3, 1), (17, 19, 128, 30, 3),
                (17, 19, 64, -100, 2), (9000, 4200, 64, -3, 2)):
            with self.subTest(source_rows=source_rows, rows=rows, start=start, stride=stride):
                engine = MemoryEngine()
                # Distinct finite BF16 encodings expose row or lane movement.
                bits = (torch.arange(source_rows * width) % 0x7000 + 0x0800).to(torch.uint16)
                value = bits.view(torch.bfloat16).reshape(source_rows, width)
                source = engine.allocate(value)
                output = engine.allocate(torch.full((rows, width), float("nan")))
                gather_rows(engine, source, output, source_rows=source_rows,
                            rows=rows, width=width, start=start, stride=stride)
                indices = (start + torch.arange(rows) * stride).clamp(0, source_rows - 1)
                actual = engine.view(output, rows * width).reshape(rows, width)
                torch.testing.assert_close(actual, value[indices], rtol=0, atol=0)

    def test_scatter_preserves_unwritten_rows_and_crosses_sram_chunks(self):
        for rows, width, start, stride in ((1, 128, 1, 2), (17, 64, 0, 2), (4200, 64, 2, 3)):
            engine = MemoryEngine()
            value = (torch.arange(rows * width) % 0x7000 + 0x0800).to(torch.uint16).view(torch.bfloat16).reshape(rows, width)
            source = engine.allocate(value)
            expected = torch.full((start + rows * stride + 3, width), -3.5, dtype=torch.bfloat16)
            output = engine.allocate(expected)
            expected[start + torch.arange(rows) * stride] = value
            scatter_rows(engine, source, output, rows=rows, width=width, start=start, stride=stride)
            actual = engine.view(output, expected.numel()).reshape_as(expected)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_polyphase_filters_and_padding_match_upstream(self):
        for length in (1, 2, 17, 65):
            for logical in (48, 65):
                with self.subTest(length=length, channels=logical):
                    engine, plan, module, packed = fixture(length, logical)
                    emit_activation(engine, plan)
                    actual = engine.view(plan.destination, length * plan.width).float().reshape(length, plan.width)
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual[:, logical:], torch.zeros_like(actual[:, logical:]), rtol=0, atol=0)
                    # Ordinary grouped transpose/forward conv are independent
                    # of the emitter's phase indexing and strided DMA logic.
                    x = packed[:, :logical].T[None]
                    expected_up = 2 * F.conv_transpose1d(
                        F.pad(x, (5, 5), mode="replicate"),
                        module.upsample.filter.expand(logical, 1, 12),
                        stride=2, groups=logical)[..., 15:-15]
                    up = engine.snapshots["upsampled"].reshape(2 * length, plan.width)[:, :logical].T[None]
                    self.assertLess(float(torch.linalg.vector_norm(up - expected_up) / torch.linalg.vector_norm(expected_up)), .01)
                    snake = engine.snapshots["snake"].reshape(2 * length, plan.width)[:, :logical].T[None]
                    expected_down = F.conv1d(
                        F.pad(snake, (5, 6), mode="replicate"),
                        module.downsample.lowpass.filter.expand(logical, 1, 12),
                        stride=2, groups=logical)
                    out = actual[:, :logical].T[None]
                    self.assertLess(float(torch.linalg.vector_norm(out - expected_down) / torch.linalg.vector_norm(expected_down)), .012)
                    # End-to-end against the official activation also binds
                    # alpha/beta broadcasting and excludes a missing filter.
                    expected = module(x)
                    self.assertLess(float(torch.linalg.vector_norm(out - expected) / torch.linalg.vector_norm(expected)), .025)
                    self.assertEqual(engine._isa_reg_counter, 1)
                    self.assertEqual(engine.dynamic_clamps, 0)

    def test_sram_activation_is_bf16_exact_against_original_lowering_across_tiles(self):
        for length, logical in ((1, 48), (17, 65), (259, 65), (45, 1536)):
            with self.subTest(length=length, channels=logical):
                fast, fast_plan, _, _ = fixture(length, logical)
                reference, reference_plan, _, _ = fixture(length, logical)
                emit_activation(fast, fast_plan)
                emit_activation_dram(reference, reference_plan)
                count = length * fast_plan.width
                torch.testing.assert_close(fast.view(fast_plan.destination, count),
                    reference.view(reference_plan.destination, count), rtol=0, atol=0)
                self.assertEqual(fast.matmul_shapes, [])

    def test_sorted_filters_match_independent_tap_convolutions_at_endpoints(self):
        for length, logical in ((1, 1), (2, 65), (17, 48), (65, 65)):
            with self.subTest(length=length, logical=logical):
                engine, plan, _, packed = fixture(length, logical, filter_accumulation='sorted')
                emit_activation(engine, plan)
                x = F.pad(packed[:, :logical].T[None], (5, 5), mode='replicate')
                # Evaluate each tap with ordinary grouped convolution. Its
                # indexing is independent of the emitter's polyphase offsets.
                up = None
                for tap in sorted(range(12), key=lambda i: abs(plan.up_filter[i])):
                    weight = torch.zeros(1, 1, 12); weight[..., tap] = plan.up_filter[tap]
                    term = (2 * F.conv_transpose1d(x, weight.expand(logical, 1, 12),
                            stride=2, groups=logical)[..., 15:-15]).bfloat16().float()
                    up = term if up is None else (up + term).bfloat16().float()
                actual_up = engine.snapshots['upsampled'].reshape(2 * length, plan.width)[:, :logical].T[None]
                torch.testing.assert_close(actual_up, up, rtol=0, atol=0)
                snake = engine.snapshots['snake'].reshape(2 * length, plan.width)[:, :logical].T[None]
                padded_snake = F.pad(snake, (5, 6), mode='replicate')
                down = None
                for tap in sorted(range(12), key=lambda i: abs(plan.down_filter[i])):
                    weight = torch.zeros(1, 1, 12); weight[..., tap] = plan.down_filter[tap]
                    term = F.conv1d(padded_snake, weight.expand(logical, 1, 12),
                                    stride=2, groups=logical).bfloat16().float()
                    down = term if down is None else (down + term).bfloat16().float()
                actual = engine.view(plan.destination, length * plan.width).float().reshape(length, plan.width)
                torch.testing.assert_close(actual[:, :logical].T[None], down, rtol=0, atol=0)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue((actual[:, logical:] == 0).all())

    def test_sorted_sram_and_dram_paths_agree_across_tiles(self):
        for length, logical in ((1, 48), (17, 65), (259, 65), (45, 1536)):
            with self.subTest(length=length, logical=logical):
                fast, plan, _, _ = fixture(length, logical, filter_accumulation='sorted')
                slow, reference, _, _ = fixture(length, logical, filter_accumulation='sorted')
                emit_activation(fast, plan); emit_activation_dram(slow, reference)
                torch.testing.assert_close(fast.view(plan.destination, length * plan.width),
                                           slow.view(reference.destination, length * plan.width), rtol=0, atol=0)

    def test_filter_order_is_stable_and_reduces_a_bf16_rounding_case(self):
        taps = [(20, .5), (-3, -.125), (7, .125), (1, -.5)]
        self.assertEqual(_ordered_filter_taps(iter(taps), 'serial'), taps)
        self.assertEqual(_ordered_filter_taps(iter(taps), 'sorted'), [taps[1], taps[2], taps[0], taps[3]])
        errors = []
        for mode in ('serial', 'sorted'):
            engine, plan, _, packed = fixture(65, 48, filter_accumulation=mode)
            emit_activation(engine, plan)
            source = F.pad(packed[:, :48].T[None], (5, 5), mode='replicate')
            weights = torch.tensor(plan.up_filter).reshape(1, 1, 12)
            reference = 2 * F.conv_transpose1d(source, weights.expand(48, 1, 12),
                                              stride=2, groups=48)[..., 15:-15]
            actual = engine.snapshots['upsampled'].reshape(130, plan.width)[:, :48].T[None]
            errors.append(float(torch.linalg.vector_norm(actual - reference)))
        self.assertLess(errors[1], .8 * errors[0])

    def test_filter_accumulation_default_image_and_instruction_count(self):
        image, default, module, _ = fixture(17, 65)
        explicit_image, serial, _, _ = fixture(17, 65, filter_accumulation='serial')
        sorted_image, ordered, _, _ = fixture(17, 65, filter_accumulation='sorted')
        self.assertEqual(default.filter_accumulation, 'serial')
        self.assertEqual(default, serial)
        self.assertEqual(replace(ordered, filter_accumulation='serial'), default)
        for other in (explicit_image, sorted_image):
            self.assertEqual(image.regions.keys(), other.regions.keys())
            for address in image.regions:
                torch.testing.assert_close(image.regions[address], other.regions[address],
                                           rtol=0, atol=0, equal_nan=True)
        for operation in (emit_activation, emit_activation_dram):
            programs = []
            for plan in (default, serial, ordered):
                engine = _WholeGraphEngine(0x98000000)
                with patch.object(udc, 'UE_AXI_DATA_WIDTH_BITS', 256), contextlib.redirect_stdout(io.StringIO()):
                    engine.start_capture(); operation(engine, plan)
                    engine.generate_instruction_halt(); engine.stop_capture()
                self.assertEqual(udc.check_isa_jumps(engine.capture_buffer, 0x98000000, name='FIR ordering'), [])
                raw = b''.join(inst.get_bytes() for inst in engine.capture_buffer)
                self.assertEqual(_instruction_types(raw).count(udc.INSTRUCTION_HALT), 1)
                programs.append(raw)
            self.assertEqual(programs[0], programs[1])
            self.assertEqual(len(programs[0]), len(programs[2]))
        before = image.cursor
        for mode in (None, True, 1, 'ascending'):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'filter_accumulation'):
                prepare_activation(module, image, input_shape=(17, 65), input_address=default.source,
                                   output_address=default.destination, scratch_address=default.scratch,
                                   identity_address=default.identity, filter_accumulation=mode)
        self.assertEqual(image.cursor, before)

    def test_legacy_snake_complete_bf16_grid_and_preserved_sram(self):
        encodings = torch.arange(65536).to(torch.uint16).view(torch.bfloat16)
        source = encodings[torch.isfinite(encodings) & (encodings.float().abs() <= math.pi / 2)]
        self.assertEqual(source.numel(), 32660)  # Both signs, including signed zero.
        count = (source.numel() + 63) // 64 * 64
        reference = source.double().sin().square()
        engine = NativeArithmeticMemoryEngine()
        engine.sram.fill_(-13.)
        engine.sram_view(0, count).zero_()
        engine.sram_view(0, count)[:source.numel()].copy_(source)
        before = engine.sram.clone()
        _sine_polynomial_sram(engine, count, 'legacy')
        output = engine.sram_view(0x10000, count)[:source.numel()].double().clamp(0, 1)
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output[:len(source) // 2], output[len(source) // 2:], rtol=0, atol=0)
        self.assertLess(float((output - reference).abs().max()), .0081)
        writable = torch.zeros(engine.sram.numel(), dtype=torch.bool)
        for address in (0, 0x10000, 0x90000):
            writable[address // 2:address // 2 + count] = True
        torch.testing.assert_close(engine.sram[~writable], before[~writable], rtol=0, atol=0)
        for invalid_count in (0, 63, 32769, True):
            with self.assertRaisesRegex(ValueError, 'SRAM tile'):
                _sine_polynomial_sram(NativeArithmeticMemoryEngine(), invalid_count, 'legacy')

    def test_legacy_snake_matches_dram_sram_with_native_rounding_across_tiles(self):
        for order in ('serial', 'sorted'):
            for length, logical in ((1, 1), (259, 65), (45, 1536)):
                with self.subTest(order=order, length=length, channels=logical):
                    kwargs = dict(filter_accumulation=order, engine_type=NativeMemoryEngine)
                    fast, plan, _, packed = fixture(length, logical, **kwargs)
                    slow, reference, _, _ = fixture(length, logical, **kwargs)
                    guard = 0xBF000000
                    for engine in (fast, slow):
                        engine.regions[guard] = torch.full((64,), -9.5, dtype=torch.bfloat16)
                    emit_activation(fast, plan)
                    emit_activation_dram(slow, reference)
                    count = length * plan.width
                    actual = fast.view(plan.destination, count)
                    torch.testing.assert_close(actual, slow.view(reference.destination, count), rtol=0, atol=0)
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertTrue((actual.reshape(length, plan.width)[:, logical:] == 0).all())
                    for engine in (fast, slow):
                        torch.testing.assert_close(engine.view(plan.source, count).float(),
                                                   packed.flatten(), rtol=0, atol=0)
                        self.assertTrue((engine.regions[guard] == -9.5).all())

    def test_snake_default_preserves_recorded_legacy_programs_and_parameter_image(self):
        original, default, module, _ = fixture(17, 65)
        captures = {
            emit_activation: '9138f5f12d47b29c3ac4e4ec8f2240405954c6c98ac9689bbe9fedd0d84232f5',
            emit_activation_dram: '7db7e3248bb5ee629895f3bec4f6547b3a5d75727a9cbcb13d93f476c6fadd12',
        }  # Captured on AXI256 before optional polynomials were introduced.
        for emit, legacy_sha256 in captures.items():
            instructions = {}
            for mode in (None, 'legacy'):
                image, plan, _, _ = fixture(17, 65, snake_polynomial=mode)
                self.assertEqual(replace(plan, snake_polynomial='legacy'), default)
                for address in original.regions:
                    torch.testing.assert_close(original.regions[address], image.regions[address],
                                               rtol=0, atol=0, equal_nan=True)
                engine = _WholeGraphEngine(0x98000000)
                with patch.object(udc, 'UE_AXI_DATA_WIDTH_BITS', 256), contextlib.redirect_stdout(io.StringIO()):
                    engine.start_capture(); emit(engine, plan)
                    engine.generate_instruction_halt(); engine.stop_capture()
                raw = b''.join(inst.get_bytes() for inst in engine.capture_buffer)
                self.assertEqual(udc.check_isa_jumps(engine.capture_buffer, 0x98000000, name='Snake polynomial'), [])
                if mode in (None, 'legacy'):
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), legacy_sha256)
                instructions[mode] = _instruction_types(raw).count(udc.INSTRUCTION_UE_OP)
        before = original.cursor
        for mode in (None, True, 1, 'tuned', 'estrin', 'minimax', []):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'snake_polynomial'):
                prepare_activation(module, original, input_shape=(17, 65), input_address=default.source,
                    output_address=default.destination, scratch_address=default.scratch,
                    identity_address=default.identity, snake_polynomial=mode)
            with self.assertRaisesRegex(ValueError, 'snake_polynomial'):
                sine_squared(torch.zeros(64), snake_polynomial=mode)
        self.assertEqual(original.cursor, before)

    def test_wide_clamp_finite_bf16_values_and_inplace_workspace(self):
        bits = torch.arange(65536).to(torch.uint16)
        values = bits.view(torch.bfloat16)
        values[~torch.isfinite(values)] = 0
        for lower, upper in ((0, float('inf')), (-.25, .5), (0, 1)):
            for inplace in (False, True):
                engine = MemoryEngine()
                source = engine.allocate(values)
                output = source if inplace else engine.allocate(torch.full_like(values, float('nan')))
                clamp_wide(engine, source, output, values.numel(), lower, upper)
                torch.testing.assert_close(engine.view(output, values.numel()),
                    values.clamp(lower, upper), rtol=0, atol=0)
                self.assertEqual(engine.matmul_shapes, [])
        # A bounded SRAM clamp must preserve unrelated gate/state workspaces.
        engine = MemoryEngine()
        engine.sram.fill_(3.5)
        engine.sram[:4096].copy_(torch.linspace(-4, 4, 4096).bfloat16())
        before = engine.sram.clone()
        sram_clamp(engine, 0, 0, 4096, scratch_address=0x20000, lo=-1, hi=1)
        torch.testing.assert_close(engine.sram[:4096], before[:4096].clamp(-1, 1), rtol=0, atol=0)
        untouched = torch.ones_like(before, dtype=torch.bool)
        untouched[:4096] = False
        untouched[0x20000 // 2:0x20000 // 2 + 8192] = False
        torch.testing.assert_close(engine.sram[untouched], before[untouched], rtol=0, atol=0)

    def test_sram_maximum_and_copy_validate_geometry_and_preserve_finite_bits(self):
        engine = MemoryEngine()
        bits = torch.arange(65536).to(torch.uint16)
        values = bits.view(torch.bfloat16)
        values[~torch.isfinite(values)] = 0
        engine.sram[:values.numel()].copy_(values)
        sram_copy(engine, 0, 0x80000, values.numel())
        torch.testing.assert_close(engine.sram[0x80000 // 2:][:values.numel()].view(torch.uint16),
                                   values.view(torch.uint16), rtol=0, atol=0)
        for left, right, destination, count in ((0, 128, 0, 128),
                (0, 0x80000, 0, 64), (0, 0x10000, 128, 128),
                (0, 0x10000, 0, 131072), (0, 0x7FF80, 0, 128)):
            with self.assertRaises(ValueError):
                sram_maximum(engine, left, right, destination, count)

    def test_sram_activation_capture_reduces_instructions_without_identity_dots(self):
        _, plan, _, _ = fixture(1024, 64)
        counts = []
        for operation in (emit_activation_dram, emit_activation):
            engine = _WholeGraphEngine(0x98000000)
            with patch.object(udc, 'UE_AXI_DATA_WIDTH_BITS', 256), contextlib.redirect_stdout(io.StringIO()):
                engine.start_capture()
                operation(engine, plan)
                engine.generate_instruction_halt()
                engine.stop_capture()
            self.assertEqual(udc.check_isa_jumps(engine.capture_buffer, 0x98000000, name='activation SRAM'), [])
            counts.append(engine.capture_count)
            if operation is emit_activation:
                modes = [udc._inst_desc_bits(inst.words, 172, 175) for inst in engine.capture_buffer
                         if (inst.words[0] >> 8) & 15 == udc.INSTRUCTION_UE_OP]
                self.assertNotIn(udc.UE_MODE.BF16_DOT_PRODUCT, modes)
                self.assertIn(udc.UE_MODE.MAXPOOL, modes)
        self.assertLess(counts[1], counts[0] // 2)

    def test_clamp_dynamic_rows_keep_values_inplace_and_release_register(self):
        for rows in (1, 127, 128, 800, 4097):
            for inplace in (False, True):
                engine = MemoryEngine()
                values = torch.linspace(-3, 3, rows * 64).bfloat16().reshape(rows, 64)
                source = engine.allocate(values)
                output = source if inplace else engine.allocate(torch.full_like(values, float("nan")))
                identity = engine.allocate(torch.eye(64))
                clamp(engine, source, output, values.numel(), identity, -.25, .5)
                actual = engine.view(output, values.numel()).reshape_as(values)
                torch.testing.assert_close(actual, values.clamp(-.25, .5), rtol=0, atol=0)
                self.assertEqual(engine._isa_reg_counter, 1)
                self.assertEqual(engine.registers, {})
                self.assertEqual(engine.dynamic_clamps, int(rows >= 128))

    def test_large_clamp_capture_is_bounded_and_releases_registers(self):
        for rows in (128, 800, 13200, 26400):
            engine = _WholeGraphEngine(0x98000000)
            with patch.object(udc, "UE_AXI_DATA_WIDTH_BITS", 256), contextlib.redirect_stdout(io.StringIO()):
                engine.start_capture()
                clamp(engine, 0xB0000000, 0xB1000000, rows * 64, 0x90000000, 0, 1)
            self.assertLess(engine.capture_count, 120)
            self.assertEqual(engine._isa_reg_counter, 1)
            self.assertEqual(engine._inst_ptr_counter, 1)
            self.assertEqual(udc.check_isa_jumps(engine.capture_buffer, 0x98000000, name="dynamic clamp"), [])

    def test_sine_polynomial_accuracy_and_explicit_argument_limit(self):
        x = torch.linspace(-SNAKE_ARGUMENT_LIMIT, SNAKE_ARGUMENT_LIMIT, 100001)
        self.assertLess(float((sine_squared(x) - x.sin().square()).abs().max()), .001)
        x = torch.linspace(-13, 13, 65537).bfloat16().float()
        self.assertLess(float((sine_squared(x, bf16=True) - x.sin().square()).abs().max()), .012)
        values = torch.tensor([-2 * SNAKE_ARGUMENT_LIMIT, -SNAKE_ARGUMENT_LIMIT,
                               SNAKE_ARGUMENT_LIMIT, 2 * SNAKE_ARGUMENT_LIMIT])
        actual = sine_squared(values, bf16=True)
        torch.testing.assert_close(actual, actual[1].expand_as(actual), rtol=0, atol=0)
        self.assertTrue(((actual >= 0) & (actual <= 1)).all())

    def test_broadcast_immediates_encode_rne_bf16_bits(self):
        for scalar in (.44320979714393616, -.0255434587597847, -1 / 3, 2 / 45):
            for operation, mode in ((scale, udc.UE_MODE.MUL_BROADCAST),
                                    (shift, udc.UE_MODE.ADD_BROADCAST)):
                engine = _WholeGraphEngine(0x98000000)
                with patch.object(udc, "UE_AXI_DATA_WIDTH_BITS", 256):
                    engine.start_capture()
                    operation(engine, 0xB0000000, 0xB1000000, 64, scalar)
                broadcasts = [inst.words for inst in engine.capture_buffer
                              if udc._inst_desc_bits(inst.words, 172, 175) == mode]
                self.assertEqual(len(broadcasts), 1)
                encoded = udc._inst_desc_bits(broadcasts[0], 186, 206)
                expected = int(torch.tensor(scalar).bfloat16().view(torch.uint16))
                self.assertEqual(encoded, expected)

    def test_filter_quantization_is_symmetric_unit_dc_and_nearest_candidate(self):
        module = Activation1d(SnakeBeta(1, alpha_logscale=True))
        original = module.upsample.filter.flatten().double()
        quantized = torch.tensor(quantize_filter_dc(tuple(original.tolist())), dtype=torch.float64)
        torch.testing.assert_close(quantized, quantized.flip(0), rtol=0, atol=0)
        self.assertEqual(float(quantized.sum()), 1.0)
        self.assertEqual(float(2 * quantized[::2].sum()), 1.0)
        self.assertEqual(float(2 * quantized[1::2].sum()), 1.0)
        torch.testing.assert_close(quantized, quantized.bfloat16().double(), rtol=0, atol=0)
        # Independent exhaustive enumeration checks the constrained optimum.
        bits = original[:6].bfloat16().view(torch.uint16).int()
        combinations = torch.cartesian_prod(*[torch.arange(-2, 3)] * 6)
        candidates = (bits[None] + combinations).to(torch.uint16).view(torch.bfloat16).double()
        valid = candidates.sum(1) == .5
        self.assertTrue(bool(valid.any()))
        error = (candidates[valid] - original[:6]).square().sum(1).min()
        self.assertEqual(float((quantized[:6] - original[:6]).square().sum()), float(error))
        self.assertLess(float((quantized - original).abs().max()), .0005)

    def test_activation_offline_capture_has_no_host_boundary(self):
        _, plan, _, _ = fixture(2, 65)
        engine = _WholeGraphEngine(0x98000000)
        with patch.object(udc, "UE_AXI_DATA_WIDTH_BITS", 256), contextlib.redirect_stdout(io.StringIO()):
            engine.start_capture()
            emit_activation(engine, plan)
            engine.generate_instruction_halt()
            engine.stop_capture()
        self.assertEqual(udc.check_isa_jumps(engine.capture_buffer, 0x98000000, name="BigCodec activation"), [])
        raw = b"".join(inst.get_bytes() for inst in engine.capture_buffer)
        types = _instruction_types(raw)
        self.assertEqual(types.count(udc.INSTRUCTION_HALT), 1)
        self.assertNotIn(udc.INSTRUCTION_SWI, types)
        self.assertGreater(len(types), 100)

    def test_matrix_stages_match_independent_filter_and_snake_equations(self):
        # Single samples, both FIR tile tails, an interior ISA loop, and the
        # official 1536-channel width cross the distinct workspace boundaries.
        for stage in ('up', 'down', 'both'):
            for rows, logical in ((1, 1), (79, 65), (317, 3), (45, 1536)):
                with self.subTest(stage=stage, rows=rows, logical=logical):
                    engine, plan, module, packed = fixture(rows, logical,
                        filter_accumulation='matrix', filter_stage=stage,
                        engine_type=matrix_engine_type())
                    parameters = {address: value.clone() for address, value in engine.regions.items()
                                  if address not in (plan.source, plan.destination, plan.scratch)}
                    # Surround exactly the advertised tensor/workspace extents.
                    for address in (plan.source, plan.destination, plan.scratch):
                        value = engine.regions.pop(address)
                        guarded = torch.full((value.numel() + 128,), -9.5, dtype=torch.bfloat16)
                        guarded[64:-64] = value
                        engine.regions[address - 128] = guarded
                    emit_activation(engine, plan)
                    up = filter_reference(packed, module.upsample.filter.flatten(), 'up',
                                          matrix=stage in ('up', 'both'))
                    snake = native_snake_reference(up, module)
                    expected = filter_reference(snake, module.downsample.lowpass.filter.flatten(),
                                                'down', matrix=stage in ('down', 'both'))
                    actual = engine.view(plan.destination, rows * plan.width).reshape(rows, plan.width)
                    torch.testing.assert_close(engine.snapshots['upsampled'].reshape_as(up), up,
                                               rtol=0, atol=0)
                    torch.testing.assert_close(engine.view(plan.scratch, snake.numel()).float().reshape_as(snake),
                                               snake, rtol=0, atol=0)
                    torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0)
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertFalse(actual[:, logical:].count_nonzero())
                    torch.testing.assert_close(engine.view(plan.source, packed.numel()).float(),
                                               packed.flatten(), rtol=0, atol=0)
                    for address in (plan.source, plan.destination, plan.scratch):
                        guarded = engine.regions[address - 128]
                        self.assertTrue((guarded[:64] == -9.5).all() and (guarded[-64:] == -9.5).all())
                    for address, value in parameters.items():
                        torch.testing.assert_close(engine.regions[address], value, rtol=0, atol=0)
                    self.assertEqual(engine._isa_reg_counter, 1)

    def test_matrix_workspace_and_stage_plan_contract(self):
        from bigcodec_filter import scratch_bytes as fir_scratch_bytes
        shape = (79, 65)
        for stage in ('up', 'down', 'both'):
            image, plan, module, _ = fixture(*shape, filter_accumulation='matrix', filter_stage=stage)
            selected = []
            if stage in ('up', 'both'):
                selected.append(fir_scratch_bytes(shape, kind='up'))
            if stage in ('down', 'both'):
                selected.append(fir_scratch_bytes((2 * shape[0], shape[1]), kind='down'))
            needed = 2 * shape[0] * channels(shape[1]) * 2 + max(selected)
            self.assertEqual(activation_scratch_bytes(shape, filter_accumulation='matrix', filter_stage=stage), needed)
            self.assertEqual(plan.up_fir is not None, stage in ('up', 'both'))
            self.assertEqual(plan.down_fir is not None, stage in ('down', 'both'))
            for fir in (plan.up_fir, plan.down_fir):
                if fir is not None:
                    self.assertEqual(fir.coefficient_precision, 'split')
                    self.assertEqual(fir.scratch, plan.scratch + 4 * plan.rows * plan.width)
                    self.assertLessEqual(fir.scratch + fir.scratch_bytes, plan.scratch + needed)
            kwargs = dict(input_shape=shape, input_address=plan.source, output_address=plan.destination,
                          scratch_address=plan.scratch, identity_address=plan.identity,
                          filter_accumulation='matrix', filter_stage=stage)
            before = image.cursor
            for size in (needed - 1, 0, -1, True, float(needed)):
                with self.assertRaisesRegex(ValueError, 'scratch bytes'):
                    prepare_activation(module, image, workspace_bytes=size, **kwargs)
            self.assertEqual(image.cursor, before)
            # A compiler may provide its larger graph-wide shared workspace.
            prepare_activation(module, image, workspace_bytes=needed + 4096, **kwargs)
            with self.assertRaisesRegex(ValueError, 'SRAM activation emitter'):
                emit_activation_dram(image, plan)
            malformed = replace(plan, up_fir=None) if plan.up_fir is not None else replace(plan, down_fir=None)
            with self.assertRaisesRegex(ValueError, 'selected stage'):
                emit_activation(image, malformed)
        for order in ('serial', 'sorted'):
            for stage in ('up', 'down', 'both'):
                self.assertEqual(activation_scratch_bytes(shape, filter_accumulation=order, filter_stage=stage),
                                 9 * shape[0] * channels(shape[1]) * 2)
        for stage in (None, True, 1, 'encoder', []):
            with self.assertRaisesRegex(ValueError, 'filter_stage'):
                activation_scratch_bytes(shape, filter_stage=stage)
        for shape in ((0, 1), (-1, 1), (1, 0), (True, 1), (1., 1), (1,), None):
            with self.assertRaisesRegex(ValueError, 'shape'):
                activation_scratch_bytes(shape)
        with self.assertRaisesRegex(ValueError, 'SRAM capacity'):
            activation_scratch_bytes((1, 4096), filter_accumulation='matrix')

    def test_matrix_activation_capture_is_one_closed_program_for_each_stage(self):
        for stage in ('up', 'down', 'both'):
            _, plan, _, _ = fixture(317, 65, filter_accumulation='matrix', filter_stage=stage)
            engine = _WholeGraphEngine(0x98000000)
            with patch.object(udc, 'UE_AXI_DATA_WIDTH_BITS', 256), contextlib.redirect_stdout(io.StringIO()):
                engine.start_capture()
                emit_activation(engine, plan)
                halt = engine.capture_count
                engine.generate_instruction_halt()
                engine.stop_capture()
            self.assertEqual(engine._isa_reg_counter, 1)
            self.assertFalse(engine._capture_loop_stack)
            self.assertEqual(udc.check_isa_jumps(engine.capture_buffer, 0x98000000, name='Matrix activation'), [])
            raw = b''.join(inst.get_bytes() for inst in engine.capture_buffer)
            types = _instruction_types(raw)
            self.assertEqual(types.count(udc.INSTRUCTION_HALT), 1)
            self.assertNotIn(udc.INSTRUCTION_SWI, types)
            self.assertTrue(all(kind == udc.INSTRUCTION_NOP for kind in types[halt + 1:]))


if __name__ == "__main__":
    unittest.main()
