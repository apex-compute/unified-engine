"""Native rounding, accuracy, aliasing, and SRAM preservation for Padé tanh."""

import contextlib
import io
import unittest

import torch

from bigcodec_tanh import compensated_tanh_sram
from bigcodec_lstm import _tanh_sram, udc
from bigcodec_device import shared
from test_bigcodec_lstm import NativeArithmeticMemoryEngine, native_round


class TanhEngine(NativeArithmeticMemoryEngine):
    def start_queue_for_bf16_matvec_operation(self, *, max_clear_en, fmax_context_addr,
            vector_sram_start_addr, matrix_sram_start_addr, output_sram_wb_addr,
            K, N, lalu_mode, lalu_scalar=0, lalu_a=0, lalu_b=0):
        self.native_calls.append(lalu_mode)
        self.assert_identity(matrix_sram_start_addr, K, N)
        value = self.sram_view(vector_sram_start_addr, K).float()
        if lalu_mode == udc.LALU_MODE.MODE_RECIP:
            assert lalu_scalar == 1.
            result = native_round(1 / value)
        elif lalu_mode == udc.LALU_MODE.CLAMP:
            result = value.clamp(lalu_a, lalu_b)
        else:
            raise AssertionError(lalu_mode)
        self.sram_view(output_sram_wb_addr, N).copy_(result)

    def assert_identity(self, address, K, N):
        assert K == N == 64 and address >= 0x80000
        assert torch.equal(self.sram_view(address, K * N).reshape(N, K),
                           torch.eye(64, dtype=torch.bfloat16))


class CompensatedTanhTest(unittest.TestCase):
    def make_engine(self, value, source, identity=0x80000):
        engine = TanhEngine()
        engine.sram.fill_(-9.25)
        engine.sram_view(identity, 4096).copy_(torch.eye(64, dtype=torch.bfloat16).flatten())
        engine.sram_view(source, value.numel()).copy_(value)
        return engine

    def test_accuracy_and_preserved_sram_at_lstm_offsets(self):
        for count, source, output, identity in (
                (64, 0x1800, 0x1800, 0x80000),
                (1536, 0x1800, 0x8000, 0x80000),
                (4096, 0xC000, 0xC000, 0x90000)):
            with self.subTest(count=count, source=source, output=output):
                value = torch.linspace(-8, 8, count).bfloat16()
                engine = self.make_engine(value, source, identity)
                before = engine.sram.clone()
                compensated_tanh_sram(engine, source, output, count, identity=identity)
                actual = engine.sram_view(output, count).float()
                reference = value.float().tanh()
                ideal = reference.bfloat16().float()
                error = (actual - reference).double().norm()
                rounding_floor = (ideal - reference).double().norm()
                # Native double rounding can select the adjacent BF16 value
                # around a midpoint; require proximity to the rounding floor.
                self.assertLessEqual(float(error), float(rounding_floor) * 1.2)
                self.assertLess(float((actual - reference).abs().max()), .0023)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue(torch.equal(actual, -actual.flip(0)))
                preserved = torch.ones_like(engine.sram, dtype=torch.bool)
                for start, end in ((output, output + count * 2),
                                   (0x10000, 0x66000), (0xF0000, 0xF2000)):
                    preserved[start // 2:end // 2] = False
                self.assertTrue(torch.equal(engine.sram[preserved], before[preserved]))
                self.assertEqual(engine.dma_calls, [])

    def test_all_bf16_arguments_improve_over_uncompensated_expression(self):
        positive = torch.arange(0x3A80, 0x4201, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
        value = torch.cat((positive, -positive, torch.zeros(62, dtype=torch.bfloat16)))
        outputs = []
        for emitter in (_tanh_sram, compensated_tanh_sram):
            engine = self.make_engine(value, 0)
            emitter(engine, 0, 0, value.numel())
            outputs.append(engine.sram_view(0, value.numel()).float())
        reference = value.float().tanh()
        old_error = (outputs[0] - reference).double().norm()
        new_error = (outputs[1] - reference).double().norm()
        self.assertLess(float(new_error), float(old_error) * .35)
        self.assertLess(float((outputs[1] - reference).abs().max()), .0023)

    def test_invalid_geometry_rejected_before_instructions(self):
        for source, output, count, identity in (
                (0, 0, 0, 0x80000), (0, 0, 65, 0x80000),
                (0, 0, 4160, 0x80000), (2, 0, 64, 0x80000),
                (0xF000, 0, 4096, 0x80000), (0, 0x10000, 64, 0x80000),
                (0, 0, 64, 0xF0000), (0, 0, 64, 0x80001)):
            with self.subTest(source=source, output=output, count=count, identity=identity):
                engine = TanhEngine()
                before = engine.sram.clone()
                with self.assertRaises(ValueError):
                    compensated_tanh_sram(engine, source, output, count, identity=identity)
                self.assertTrue(torch.equal(engine.sram, before))
                self.assertEqual(engine.native_calls, [])

    def test_instruction_capture_is_finite_and_contains_no_dma(self):
        engine = shared._WholeGraphEngine(shared.MODEL_BASE)
        def unexpected_dma(*args, **kwargs):
            self.fail("SRAM tanh unexpectedly emitted a DRAM transfer")
        engine.accelerator_memory_to_sram = unexpected_dma
        engine.sram_to_accelerator_memory = unexpected_dma
        with contextlib.redirect_stdout(io.StringIO()):
            engine.start_capture()
            compensated_tanh_sram(engine, 0x1800, 0x1800, 1536)
            engine.generate_instruction_halt()
            engine.stop_capture()
        self.assertGreater(engine.capture_count, 100)
        self.assertLess(engine.capture_count, 10000)
        self.assertEqual(udc.check_isa_jumps(engine.capture_buffer, shared.MODEL_BASE,
                                           name="compensated_tanh"), [])
        kinds = [(instruction.words[0] >> 8) & 15 for instruction in engine.capture_buffer]
        self.assertEqual(kinds.count(udc.INSTRUCTION_HALT), 1)
        self.assertNotIn(udc.INSTRUCTION_SWI, kinds)


if __name__ == "__main__":
    unittest.main()
