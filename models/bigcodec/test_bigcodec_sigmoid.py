"""Independent gate/cell equations, aliasing, guards and captured instructions."""
from __future__ import annotations

import contextlib
import hashlib
import io
from types import SimpleNamespace
import unittest

import torch

from bigcodec_sigmoid import sigmoid_pair_sram, paired_lstm_step_sram
from bigcodec_device import shared, udc
from test_bigcodec_lstm import native_round as q
from test_bigcodec_tanh import TanhEngine


def value(x):
    x = torch.as_tensor(x).float()
    high = x.bfloat16().float()
    return high, (x - high).bfloat16().float()


def two_sum(a, b):
    high = q(a + b)
    virtual = q(high - a)
    low = q(q(a - q(high - virtual)) + q(b - virtual))
    return high, low


def two_product(a, b):
    def split(x):
        z = q(17 * x)
        high = q(z - q(z - x))
        return high, q(x - high)
    ah, al = split(a)
    bh, bl = split(b)
    high = q(a * b)
    error = q(q(q(high - q(ah * bh)) - q(al * bh)) - q(ah * bl))
    return high, q(q(al * bl) - error)


def add(a, b):
    high, error = two_sum(a[0], b[0])
    return two_sum(high, q(q(a[1] + b[1]) + error))


def multiply(a, b):
    high, error = two_product(a[0], b[0])
    correction = q(q(q(a[0] * b[1]) + q(a[1] * b[0])) + error)
    return two_sum(high, correction)


def tanh_pair(x):
    x = x.clamp(-4, 4)
    square = two_product(x, x)
    def polynomial(coefficients):
        result = value(coefficients[0])
        for coefficient in coefficients[1:]:
            result = add(multiply(result, square), value(coefficient))
        return result
    numerator = polynomial((1 / 135135, 378 / 135135, 17325 / 135135, 1.))
    denominator = polynomial((28 / 135135, 3150 / 135135, 62370 / 135135, 1.))
    inverse = q(1 / denominator[0])
    product = multiply(denominator, (inverse, torch.zeros_like(inverse)))
    error = add(value(1.), (-product[0], -product[1]))
    inverse_pair = add((inverse, torch.zeros_like(inverse)),
                       (q(inverse * q(error[0] + error[1])), torch.zeros_like(inverse)))
    return multiply(multiply(numerator, inverse_pair), (x, torch.zeros_like(x)))


def sigmoid_pair(x):
    return multiply(add(value(1.), tanh_pair(q(x * .5))), value(.5))


def tanh(x):
    high, low = tanh_pair(x)
    return q(high + low).clamp(-1, 1)


class SigmoidTests(unittest.TestCase):
    def engine(self, source, x):
        engine = TanhEngine()
        engine.sram.fill_(-3.25)
        engine.sram_view(0x80000, 4096).copy_(torch.eye(64, dtype=torch.bfloat16).flatten())
        engine.sram_view(source, len(x)).copy_(x)
        return engine

    def test_components_match_native_rounding_equations_and_preserve_other_sram(self):
        for count in (64, 1536):
            for source, high, low in ((0, 0, 0x8000), (0xC00, 0xC00, 0x9000),
                                       (0x2400, 0x2400, 0xA000), (0, 0x8000, 0)):
                x = torch.linspace(-12, 12, count).bfloat16().float()
                engine = self.engine(source, x)
                before = engine.sram.clone()
                sigmoid_pair_sram(engine, source, (high, low), count)
                for address, expected in zip((high, low), sigmoid_pair(x)):
                    torch.testing.assert_close(engine.sram_view(address, count).float(), expected, rtol=0, atol=0)
                preserve = torch.ones_like(engine.sram, dtype=torch.bool)
                for start, end in ((high, high + count * 2), (low, low + count * 2),
                                   (0x10000, 0x66000), (0xF0000, 0xF2000)):
                    preserve[start // 2:end // 2] = False
                self.assertTrue(torch.equal(engine.sram[preserve], before[preserve]))
                self.assertEqual(engine.dma_calls, [])

    def test_retained_low_improves_gate_accuracy_over_high_alone(self):
        x = torch.linspace(-8, 8, 1536).bfloat16().float()
        engine = self.engine(0, x)
        sigmoid_pair_sram(engine, 0, (0, 0x8000), len(x))
        high, low = (engine.sram_view(address, len(x)).float() for address in (0, 0x8000))
        exact = x.sigmoid()
        self.assertGreater(int(torch.count_nonzero(low)), 1000)
        self.assertLess(float((high + low - exact).norm()), .6 * float((high - exact).norm()))
        self.assertLess(float((high + low - exact).abs().max()), 4e-4)

    def test_invalid_geometry_rejected_before_sram_changes(self):
        engine = self.engine(0, torch.zeros(1536))
        before = engine.sram.clone()
        for source, output, count in ((0, (0, 0), 64), (0, (0, 128), 1536),
                (0, (0,), 64), (2, (0, 0x8000), 64), (-128, (0, 0x8000), 64),
                (0, (0, 0x10000), 64), (0, (0, 0x8000), 128),
                (0, (0, 0x8000), 0), (0, (0, 0x8000), 4096)):
            with self.assertRaises(ValueError):
                sigmoid_pair_sram(engine, source, output, count)
            self.assertTrue(torch.equal(engine.sram, before))
            self.assertEqual(engine.native_calls, [])
            self.assertEqual(engine.dma_calls, [])

    def make_step(self):
        torch.manual_seed(91015)
        width = 1536
        logits = (torch.randn(4, width) * 2).bfloat16().float()
        cell = (torch.randn(width) * 3).bfloat16().float()
        low = (torch.randn(width) / 1000).bfloat16().float()
        identity, previous, low_address, output = 0x90000000, 0xB0000000, 0xB0010000, 0xB0020000
        engine = TanhEngine()
        engine.sram.fill_(-3.25)
        engine.sram_view(0x8000, width * 4).copy_(logits.flatten())
        engine.regions[identity] = torch.eye(64, dtype=torch.bfloat16).flatten()
        engine.regions[previous] = cell.bfloat16().clone()
        engine.regions[low_address] = low.bfloat16().clone()
        engine.regions[output] = torch.full((width,), float('nan'), dtype=torch.bfloat16)
        plan = SimpleNamespace(padded_width=width, recurrent_precision='bf16',
            compensated_cell=True, preserve_cell_residual=True, compensated_tanh=True,
            fused_projection=True, identity_address=identity, regions={'cell_low': (low_address, width)})
        return engine, plan, previous, output, logits, cell, low

    def test_joint_step_preserves_cell_pair_and_rounds_final_hidden_to_bf16(self):
        engine, plan, previous, output, logits, cell, low = self.make_step()
        before = engine.sram.clone()
        paired_lstm_step_sram(engine, plan, 0, previous, output)
        i, f, o = (sigmoid_pair(logits[index]) for index in (0, 1, 3))
        g = tanh(logits[2])
        state = add(multiply(f, (cell, low)), multiply(i, (g, torch.zeros_like(g))))
        h = multiply(o, (tanh(state[0]), torch.zeros_like(g)))
        hidden = q(h[0] + h[1])
        for address, expected in ((previous, state[0]), (plan.regions['cell_low'][0], state[1]),
                                  (output, hidden)):
            torch.testing.assert_close(engine.view(address, 1536).float(), expected, rtol=0, atol=0)
        preserve = torch.ones_like(engine.sram, dtype=torch.bool)
        for start, end in ((0, 0x3000), (0x8000, 0x8C00), (0x9000, 0x9C00),
                (0xA000, 0xAC00), (0xC000, 0xCC00), (0xE000, 0xEC00),
                (0x10000, 0x66000), (0x80000, 0x82000), (0xF0000, 0xF2000)):
            preserve[start // 2:end // 2] = False
        self.assertTrue(torch.equal(engine.sram[preserve], before[preserve]))
        self.assertEqual(engine.dma_calls, [('read', plan.identity_address, 8192),
            ('read', previous, 3072), ('read', plan.regions['cell_low'][0], 3072),
            ('write', previous, 3072), ('write', plan.regions['cell_low'][0], 3072),
            ('write', output, 3072)])

    def test_step_rejects_unsupported_plans_and_dram_overlap_before_mutation(self):
        for attribute, value in (('padded_width', 4096), ('recurrent_precision', 'if8'),
                ('compensated_cell', False), ('preserve_cell_residual', False),
                ('compensated_tanh', False), ('fused_projection', False),
                ('identity_address', 1)):
            engine, plan, previous, output, *_ = self.make_step()
            before = engine.sram.clone()
            setattr(plan, attribute, value)
            with self.assertRaises(ValueError):
                paired_lstm_step_sram(engine, plan, 0, previous, output)
            self.assertTrue(torch.equal(engine.sram, before))
            self.assertEqual(engine.dma_calls, [])
        engine, plan, previous, _, *_ = self.make_step()
        with self.assertRaisesRegex(ValueError, 'overlap'):
            paired_lstm_step_sram(engine, plan, 0, previous, previous)
        self.assertEqual(engine.dma_calls, [])

    def test_capture_matches_native_validated_prototype_and_never_calls_host_io(self):
        # Frozen from the independently executed prototype, with byte-for-byte
        # parity checked before porting these golden instruction digests.
        goldens = {'primitive64': '33087b2bbc7f6b44c5d4d457f09db44066c32478ef86af141571755c53fd0d06',
                   'primitive1536': '2c2075cb36de2a285b8f5d85d71e26aae58fa9cb08910bb2fa1f2d65a0547449',
                   'step': '84efefdf06b33726cd1e9f150ad8db88687038c760149eed4822c3396fc1bd09'}
        previous_width = udc.UE_AXI_DATA_WIDTH_BITS
        try:
            udc.UE_AXI_DATA_WIDTH_BITS = 256
            _, plan, previous, output, *_ = self.make_step()
            for name, digest in goldens.items():
                engine = shared._WholeGraphEngine(0xA0000000)
                def unexpected_host_call(*args, **kwargs):
                    self.fail('Paired gate capture attempted host I/O or execution')
                for method in ('dma_read', 'dma_write', 'read_reg32', 'write_reg32',
                               'start_execute_from_dram', 'software_reset'):
                    setattr(engine, method, unexpected_host_call)
                if name != 'step':
                    engine.accelerator_memory_to_sram = unexpected_host_call
                    engine.sram_to_accelerator_memory = unexpected_host_call
                with contextlib.redirect_stdout(io.StringIO()):
                    engine.start_capture()
                    if name == 'step':
                        paired_lstm_step_sram(engine, plan, 0, previous, output)
                    else:
                        sigmoid_pair_sram(engine, 0, (0, 0x8000),
                                          64 if name == 'primitive64' else 1536)
                    engine.generate_instruction_halt()
                    engine.stop_capture()
                raw = b''.join(instruction.get_bytes() for instruction in engine.capture_buffer)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
                self.assertEqual(udc.check_isa_jumps(engine.capture_buffer, 0xA0000000), [])
                kinds = [(instruction.words[0] >> 8) & 15 for instruction in engine.capture_buffer]
                self.assertEqual(kinds.count(udc.INSTRUCTION_HALT), 1)
                self.assertNotIn(udc.INSTRUCTION_SWI, kinds)
                self.assertEqual(engine._isa_reg_counter, 1)
                self.assertEqual(engine._capture_loop_stack, [])
        finally:
            udc.UE_AXI_DATA_WIDTH_BITS = previous_width


if __name__ == '__main__':
    unittest.main()
