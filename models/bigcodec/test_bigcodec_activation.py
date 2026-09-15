"""Independent alias-filter geometry and BF16 activation memory execution."""

import contextlib
import io
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
                                emit_activation, emit_activation_dram, prepare_activation,
                                quantize_filter_dc, sine_squared)
from bigcodec_device import (channels, clamp, clamp_wide, gather_rows, scale,
                            scatter_rows, shift, sram_clamp, sram_copy,
                            sram_maximum, udc)
from bigcodec_vq.activations import SnakeBeta
from bigcodec_vq.alias_free_torch.act import Activation1d
from test_bigcodec_quantizer import MemoryEngine as QuantizerMemoryEngine
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


def fixture(length, logical_channels):
    module = Activation1d(SnakeBeta(logical_channels, alpha_logscale=True))
    generator = torch.Generator().manual_seed(728)
    with torch.no_grad():
        module.act.alpha.copy_(torch.randn(logical_channels, generator=generator) * .15)
        module.act.beta.copy_(torch.randn(logical_channels, generator=generator) * .15)
    module.eval().requires_grad_(False)
    engine = MemoryEngine()
    identity = engine.allocate(torch.eye(64))
    width = channels(logical_channels)
    source, destination, scratch = 0xB0000000, 0xB1000000, 0xB2000000
    plan = prepare_activation(module, engine, input_shape=(length, logical_channels),
                              input_address=source, output_address=destination,
                              scratch_address=scratch, identity_address=identity)
    for address, count in ((source, length * width), (destination, length * width),
                           (scratch, activation_scratch_bytes((length, logical_channels)) // 2)):
        engine.regions[address] = torch.full((count,), float("nan"), dtype=torch.bfloat16)
    packed = torch.zeros(length, width, dtype=torch.bfloat16)
    packed[:, :logical_channels] = torch.randn(length, logical_channels, generator=generator) * .35
    engine.view(source, packed.numel()).copy_(packed.flatten())
    engine.watch = plan
    return engine, plan, module, packed.float()


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


if __name__ == "__main__":
    unittest.main()
