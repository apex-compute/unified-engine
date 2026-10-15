"""Independent FIR geometry, coefficient packing, outer-loop and ISA checks.

The memory interpreter uses FP64 dots followed by BF16 storage to isolate
geometry. Native BF19/BF20 numerical parity is covered by recorded FPGA runs;
this test does not claim to emulate the driver's complete matrix ISA.
"""
import contextlib
import hashlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bigcodec_filter import prepare_fir, emit_fir, scratch_bytes
from bigcodec_device import shared, udc
from bigcodec_activation import quantize_filter_dc
from bigcodec_vq.alias_free_torch.act import Activation1d
from bigcodec_vq.activations import SnakeBeta
from test_bigcodec_activation import MemoryEngine


class FilterEngine(MemoryEngine):
    """Replay the emitted outer loop's memory calls and integer addresses."""
    def __init__(self):
        super().__init__()
        self.recording = None
        self.loop_count = self.loop_replays = 0
        self.dynamic_reads, self.dynamic_writes = [], []

    def _record(self, method, args, kwargs):
        if self.recording is not None:
            self.recording.append((method, args, dict(kwargs)))
            return True
        return False

    def generate_instruction_add_imm(self, src_reg_idx, immediate_value, dst_reg_idx=None):
        args = (src_reg_idx, immediate_value, dst_reg_idx)
        if self._record('generate_instruction_add_imm', args, {}):
            return
        destination = src_reg_idx if dst_reg_idx is None else dst_reg_idx
        self.registers[destination] = (self.registers[src_reg_idx] + immediate_value) & 0xffffffff

    def loop_start(self, loop_cnt, relative):
        assert not relative and self.recording is None and loop_cnt >= 2
        self.registers[self.alloc_isa_reg()] = loop_cnt
        self.loop_count, self.recording = loop_cnt, []

    def loop_end(self):
        operations, self.recording = self.recording, None
        assert operations is not None
        for _ in range(self.loop_count):
            for method, args, kwargs in operations:
                getattr(self, method)(*args, **kwargs)
            self.loop_replays += 1
        self.release_isa_reg()

    def accelerator_memory_to_sram(self, source, destination, elements, **kwargs):
        if self._record('accelerator_memory_to_sram', (source, destination, elements), kwargs):
            return
        register = kwargs.pop('general_reg_src', None)
        if register is not None:
            source = self.registers[register] << 3
            self.dynamic_reads.append((source, elements * 2))
        return super().accelerator_memory_to_sram(source, destination, elements, **kwargs)

    def sram_to_accelerator_memory(self, source, destination, elements, **kwargs):
        if self._record('sram_to_accelerator_memory', (source, destination, elements), kwargs):
            return
        return super().sram_to_accelerator_memory(source, destination, elements, **kwargs)

    def bf16_transpose_core(self, **kwargs):
        if self._record('bf16_transpose_core', (), kwargs):
            return
        # Read the actual packed identity. A previous harness wrote FP32 eye64;
        # the inherited transpose interpreter had ignored those bytes entirely.
        identity = self.view(kwargs['IDENTITY_DRAM_ADDR'], 4096).reshape(64, 64)
        torch.testing.assert_close(identity, torch.eye(64, dtype=torch.bfloat16), rtol=0, atol=0)
        return super().bf16_transpose_core(**kwargs)

    def matmat_mul_core(self, **kwargs):
        if self._record('matmat_mul_core', (), kwargs):
            return
        rows = kwargs.pop('gpr_M_reg', None)
        output = kwargs.pop('gpr_out_addr', None)
        if rows is not None:
            assert kwargs['M'] == self.registers[rows] == 64
        if output is not None:
            kwargs['OUTPUT_DRAM_ADDR'] = self.registers[output] << 3
            self.dynamic_writes.append(kwargs['OUTPUT_DRAM_ADDR'])
        m, k, n = (kwargs[key] for key in ('M', 'K', 'N'))
        assert all(kwargs[key] % 128 == 0 for key in
                   ('A_DRAM_ADDR', 'B_DRAM_ADDR', 'OUTPUT_DRAM_ADDR'))
        left = self.view(kwargs['A_DRAM_ADDR'], m * k).double().reshape(m, k)
        right = self.view(kwargs['B_DRAM_ADDR'], n * k).double().reshape(n, k)
        self.view(kwargs['OUTPUT_DRAM_ADDR'], m * n).copy_((left @ right.T).flatten())
        self.matmul_shapes.append((m, k, n))


def official_taps():
    return Activation1d(SnakeBeta(1, alpha_logscale=True)).upsample.filter.flatten()


def reference(source, taps, kind):
    """Upstream grouped-convolution equation; no packed-matrix indexing."""
    source = source.double().T.unsqueeze(0)
    channels = source.shape[1]
    weight = taps.double().reshape(1, 1, 12).expand(channels, 1, 12)
    if kind == 'up':
        output = 2 * F.conv_transpose1d(F.pad(source, (5, 5), mode='replicate'),
            weight, stride=2, groups=channels)[..., 15:-15]
    else:
        output = F.conv1d(F.pad(source, (5, 6), mode='replicate'), weight, stride=2, groups=channels)
    return output[0].T


def fixture(rows, logical, kind, precision, *, taps=None, identity_dtype=torch.bfloat16):
    taps = official_taps() if taps is None else taps
    image = shared._ImageBuilder(0x90000000, 0xA0000000)
    identity = image.allocate(torch.eye(64, dtype=identity_dtype), alignment=128)
    plan = prepare_fir(taps, image, input_shape=(rows, logical), source=0xB0000080,
        destination=0xB1000080, scratch=0xB2000080, identity=identity,
        kind=kind, coefficient_precision=precision)
    engine = FilterEngine()
    engine.regions[image.base] = torch.frombuffer(bytearray(image.data), dtype=torch.bfloat16)
    for address, count in ((plan.source, rows * plan.width),
                           (plan.destination, plan.output_rows * plan.width),
                           (plan.scratch, plan.scratch_bytes // 2)):
        guarded = torch.full((count + 128,), -3.25, dtype=torch.bfloat16)
        guarded[64:-64] = float('nan')
        engine.regions[address - 128] = guarded
    generator = torch.Generator().manual_seed(271828 + rows)
    source = torch.randn(rows, plan.width, generator=generator).bfloat16()
    source[:, logical:] = 0
    source[0, 0] = -.75
    source[-1, 0] += .3125
    engine.view(plan.source, source.numel()).copy_(source.flatten())
    return engine, image, plan, source


class CaptureEngine(shared._WholeGraphEngine):
    def __init__(self):
        super().__init__(0x98000000)
        self.max_register, self.outer_loops = 0, []

    def alloc_isa_reg(self):
        register = super().alloc_isa_reg()
        self.max_register = max(self.max_register, register)
        return register

    def loop_start(self, loop_cnt=0, gpr_loop_cnt=None, relative=True):
        register = super().loop_start(loop_cnt, gpr_loop_cnt, relative)
        self.outer_loops.append((loop_cnt, self.capture_count, register, relative))
        return register


def capture(plan):
    engine = CaptureEngine()
    with contextlib.redirect_stdout(io.StringIO()):
        engine.start_capture()
        emit_fir(engine, plan)
        halt = engine.capture_count
        engine.generate_instruction_halt()
        engine.stop_capture()
    assert engine._isa_reg_counter == 1 and not engine._capture_loop_stack
    assert not udc.check_isa_jumps(engine.capture_buffer, engine.get_program_dram_addr(), name='FIR')
    types = [(instruction.words[0] >> 8) & 15 for instruction in engine.capture_buffer]
    assert types.count(udc.INSTRUCTION_HALT) == 1 and udc.INSTRUCTION_SWI not in types
    assert all(kind == udc.INSTRUCTION_NOP for kind in types[halt + 1:])
    for count, head, register, relative in engine.outer_loops:
        assert count >= 2 and not relative
        target = engine.get_program_dram_addr() + head * 32
        assert target % 64 == 0
        matches = []
        for index, instruction in enumerate(engine.capture_buffer):
            words = instruction.words
            if (words[0] >> 8) & 15 != udc.INSTRUCTION_JUMP:
                continue
            immediate = ((words[1] >> 22) & 1023) | ((words[2] & ((1 << 22) - 1)) << 10)
            if (words[1] & 15 == udc.JUMP_MODE_JNZ and (words[1] >> 4) & 63 == register
                    and immediate << 3 == target):
                matches.append(index)
        assert len(matches) == 1 and matches[0] > head
    raw = b''.join(instruction.get_bytes() for instruction in engine.capture_buffer)
    assert len(raw) % 64 == 0
    return raw, engine


class FilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.old_width = udc.UE_AXI_DATA_WIDTH_BITS
        udc.UE_AXI_DATA_WIDTH_BITS = 256

    @classmethod
    def tearDownClass(cls):
        udc.UE_AXI_DATA_WIDTH_BITS = cls.old_width

    def test_filters_match_independent_geometry_at_edges_loops_and_padding(self):
        for kind, rows, logical in (('up', 1, 1), ('up', 79, 3), ('up', 317, 65),
                                    ('down', 2, 1), ('down', 159, 65), ('down', 634, 65),
                                    ('down', 768, 1536)):
            for precision in ('dc', 'bf16', 'split'):
                with self.subTest(kind=kind, rows=rows, logical=logical, precision=precision):
                    engine, image, plan, source = fixture(rows, logical, kind, precision)
                    parameters = bytes(image.data)
                    emit_fir(engine, plan)
                    output = engine.view(plan.destination, plan.output_rows * plan.width).reshape(plan.output_rows, plan.width)
                    taps = official_taps()
                    if precision == 'dc':
                        effective = torch.tensor(quantize_filter_dc(tuple(taps.tolist()))).double()
                    else:
                        effective = taps.bfloat16().double()
                        if precision == 'split':
                            effective += (taps - taps.bfloat16().float()).bfloat16().double()
                    expected = reference(source[:, :logical], effective, kind).bfloat16()
                    torch.testing.assert_close(output[:, :logical], expected, rtol=0, atol=0)
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertFalse(output[:, logical:].count_nonzero())
                    self.assertTrue(torch.equal(engine.view(plan.source, source.numel()), source.flatten()))
                    for base in (plan.source - 128, plan.destination - 128, plan.scratch - 128):
                        guarded = engine.regions[base]
                        self.assertTrue((guarded[:64] == -3.25).all() and (guarded[-64:] == -3.25).all())
                    self.assertEqual(engine.regions[image.base].view(torch.uint8).numpy().tobytes(), parameters)
                    self.assertEqual(engine._isa_reg_counter, 1)
                    if engine.loop_replays:
                        writes = engine.dynamic_writes
                        self.assertEqual(len(writes), engine.loop_replays)
                        self.assertEqual(writes[0], plan.destination + 64 * plan.width * 2)
                        self.assertTrue(all(b - a == 64 * plan.width * 2 for a, b in zip(writes, writes[1:])))
                        for address, count in engine.dynamic_reads:
                            self.assertLessEqual(plan.source, address)
                            self.assertLessEqual(address + count, plan.source + source.numel() * 2)

    def test_original_asymmetric_taps_and_split_coefficients_use_fp32_residuals(self):
        taps = torch.tensor([.01193, -.0417, .1083, .29371, -.32983, .120031,
                             .17113, -.009931, .07771, -.08039, .010013, .030117])
        original = taps.clone()
        for kind in ('up', 'down'):
            engine, image, plan, source = fixture(317, 3, kind, 'split', taps=taps)
            emit_fir(engine, plan)
            high = taps.bfloat16().double()
            effective = high + (taps - taps.bfloat16().float()).bfloat16().double()
            expected = reference(source[:, :3], effective, kind).bfloat16()
            actual = engine.view(plan.destination, plan.output_rows * plan.width).reshape(plan.output_rows, plan.width)
            torch.testing.assert_close(actual[:, :3], expected, rtol=0, atol=0)
            self.assertTrue(torch.equal(taps, original))
            # Derive an ideal matrix with one-hot time inputs through the
            # independent convolution, rather than repeating matrix packing.
            start = 29 if kind == 'up' else 123
            basis = torch.zeros(384, plan.base_k, dtype=torch.float64)
            basis[start:start + plan.base_k] = torch.eye(plan.base_k)
            ideal = reference(basis, taps.double(), kind)[64:128]
            matrix = engine.view(plan.weight, 64 * plan.dot_k).double().reshape(64, plan.dot_k)
            high_error = (matrix[:, :plan.base_k] - ideal).norm()
            pair_error = (matrix[:, :plan.base_k] + matrix[:, plan.base_k:] - ideal).norm()
            self.assertLess(pair_error, high_error / 100)
            self.assertEqual(matrix.numel() * 2, 64 * plan.dot_k * 2)
            self.assertIn((plan.weight, plan.weight + matrix.numel() * 2), image.spans)

    def test_identity_contract_checks_actual_image_bytes(self):
        for dtype in (torch.bfloat16, torch.float32):
            engine, image, plan, _ = fixture(79, 3, 'up', 'split', identity_dtype=dtype)
            if dtype == torch.bfloat16:
                self.assertIn((plan.identity, plan.identity + 8192), image.spans)
                packed = torch.frombuffer(bytearray(image.data[:8192]), dtype=torch.bfloat16).reshape(64, 64)
                torch.testing.assert_close(packed, torch.eye(64, dtype=torch.bfloat16), rtol=0, atol=0)
                emit_fir(engine, plan)
                engine.view(plan.identity, 4096)[1] = 1
            with self.assertRaises(AssertionError):
                emit_fir(engine, plan)

    def test_validation_rejects_unsupported_geometry_ranges_and_taps(self):
        defaults = dict(input_shape=(317, 65), source=0xB0000000, destination=0xB1000000,
                        scratch=0xB2000000, identity=0x90000000)
        invalid = [dict(input_shape=shape) for shape in
                   ((0, 1), (1, 0), (-1, 1), (1, 4033), (1, 4096), (1, True), (1.5, 2), (1,))]
        invalid += [dict(kind='sideways'), dict(coefficient_precision='unknown'),
                    dict(source=-128), dict(destination=0xB1000002), dict(identity=True),
                    dict(source=(1 << 35) - 128), dict(destination=0xB0000080),
                    dict(scratch=0xB0000000), dict(identity=0xB0000000)]
        for changes in invalid:
            with self.subTest(changes=changes):
                image = Mock()
                with self.assertRaises(ValueError):
                    prepare_fir(official_taps(), image, **(defaults | changes))
                image.allocate.assert_not_called()
        for taps in (torch.zeros(11), torch.full((12,), float('nan')),
                     torch.full((12,), float('inf')), torch.full((12,), 3e38)):
            with self.assertRaises(ValueError):
                prepare_fir(taps, Mock(), **defaults)
        for collision in (defaults['source'], defaults['destination'], defaults['scratch'], defaults['identity']):
            with self.assertRaises(ValueError):
                prepare_fir(official_taps(), Mock(allocate=Mock(return_value=collision)), **defaults)
        with self.assertRaises(ValueError):
            prepare_fir(torch.arange(12).float(), Mock(), **(defaults | dict(coefficient_precision='dc')))
        self.assertGreater(scratch_bytes((1, 4032)), 0)

    def test_native_capture_matches_validated_loop_program_and_has_bounded_registers(self):
        # Captured from the independently validated prototype used in the
        # op087 native direct-versus-loop parity run. These protect the port's
        # instruction sequence; all geometry assertions above are independent.
        goldens = [
            ('up', 317, 65, 'split', '073415067c2443816fb2ca69083d77984d351059751988c72970e54082b56fa8'),
            ('down', 634, 65, 'dc', 'd208db368c682f98f5bf1ec213f71aefecd366a97bed39bb3b3b8397a00537b4'),
            ('down', 768, 1536, 'split', 'b74659ecb0252b243c1158c444e4fc5d10dae8df306745ffe0680e9082202a50'),
            ('up', 1, 1, 'split', 'b6f0c8c25ce8120932b584b74e53f6b12e94eaaff48baf8e988fb3094fb744d0')]
        for kind, rows, logical, precision, digest in goldens:
            with self.subTest(kind=kind, rows=rows, logical=logical):
                _, _, plan, _ = fixture(rows, logical, kind, precision)
                raw, engine = capture(plan)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
                self.assertLessEqual(engine.max_register, 25)
        _, _, plan, _ = fixture(1, 4032, 'up', 'split')
        capture(plan)

    def test_fir_program_size_does_not_grow_with_interior_clip_length(self):
        for kind, rows in (('up', 317), ('down', 634)):
            sizes = []
            for length in (rows, rows + 64000):
                image = shared._ImageBuilder(0x90000000, 0xA0000000)
                identity = image.allocate(torch.eye(64, dtype=torch.bfloat16), alignment=128)
                plan = prepare_fir(official_taps(), image, input_shape=(length, 1536),
                    source=0x80000000, destination=0xA0000000, scratch=0xD0000000,
                    identity=identity, kind=kind)
                self.assertEqual(plan.scratch_bytes, scratch_bytes((length, 1536), kind=kind))
                raw, engine = capture(plan)
                sizes.append(len(raw))
                self.assertEqual(len(engine.outer_loops), 1)
            self.assertEqual(sizes[0], sizes[1])


if __name__ == '__main__':
    unittest.main()
