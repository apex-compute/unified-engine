"""Independent gate equations, BF16 boundaries, state reset and memory bounds."""

import unittest

import torch

from bigcodec_lstm import emit_lstm, prepare_lstm, scratch_bytes, tanh_identity, tanh_scratch_bytes, pade_tanh, udc


class MemoryImage:
    def __init__(self, engine):
        self.engine = engine
        self.cursor = 0x10000000

    def allocate(self, value, *, alignment=128):
        self.cursor = (self.cursor + alignment - 1) // alignment * alignment
        address = self.cursor
        self.engine.regions[address] = value.to(torch.bfloat16).flatten().clone()
        self.cursor += value.numel() * 2
        return address


class MemoryEngine:
    def __init__(self):
        self.regions = {}
        self.matmul_calls = []
        self.sram = torch.zeros(0x100000 // 2, dtype=torch.bfloat16)

    def view(self, address, count):
        assert address % 128 == 0 and count % 64 == 0
        for start, buffer in self.regions.items():
            offset = (address - start) // 2
            if address >= start and offset + count <= buffer.numel():
                return buffer[offset:offset + count]
        raise AssertionError(f"Out-of-bounds memory access {address:#x}, {count}")

    def matmat_mul_core(self, *, M, K, N, A_DRAM_ADDR, B_DRAM_ADDR,
                        OUTPUT_DRAM_ADDR, C_DRAM_ADDR=None,
                        bias_mode="broadcast_N", sigmoid_enable=False,
                        clamp_enable=False, clamp_min=0., clamp_max=float("inf")):
        self.matmul_calls.append(dict(M=M, K=K, N=N, weight=B_DRAM_ADDR,
                                      sigmoid=sigmoid_enable))
        a = self.view(A_DRAM_ADDR, M * K).float().reshape(M, K)
        b = self.view(B_DRAM_ADDR, N * K).float().reshape(N, K)
        result = a @ b.T
        if C_DRAM_ADDR is not None:
            assert bias_mode == "broadcast_N"
            result += self.view(C_DRAM_ADDR, N).float()
        if sigmoid_enable:
            result = result.sigmoid()
        if clamp_enable:
            result = result.clamp(clamp_min, clamp_max)
        self.view(OUTPUT_DRAM_ADDR, M * N).copy_(result.flatten())

    def eltwise_core_dram(self, *, M, N, dram_a, dram_b, dram_out, mode, scalar=None):
        a = self.view(dram_a, M * N).float()
        if dram_b is not None:
            b = self.view(dram_b, M * N).float()
            op = {udc.UE_MODE.ELTWISE_ADD: torch.add,
                  udc.UE_MODE.ELTWISE_MUL: torch.mul,
                  udc.UE_MODE.ELTWISE_SUB: torch.sub}[mode]
            result = op(a, b)
        elif mode == udc.UE_MODE.MUL_BROADCAST:
            result = a * scalar
        elif mode == udc.UE_MODE.ADD_BROADCAST:
            result = a + scalar
        else:
            raise AssertionError(mode)
        self.view(dram_out, M * N).copy_(result)

    def accelerator_memory_to_sram(self, address, sram_address, count):
        assert sram_address % 128 == 0 and sram_address // 2 + count <= self.sram.numel()
        self.sram[sram_address // 2:sram_address // 2 + count].copy_(self.view(address, count))

    def sram_to_accelerator_memory(self, sram_address, address, count):
        assert sram_address % 128 == 0 and sram_address // 2 + count <= self.sram.numel()
        self.view(address, count).copy_(self.sram[sram_address // 2:sram_address // 2 + count])

    def float_to_bf19(self, value):
        return value

    def start_queue_for_bf16_matvec_operation(self, *, max_clear_en, fmax_context_addr,
            vector_sram_start_addr, matrix_sram_start_addr, output_sram_wb_addr,
            K, N, lalu_mode, lalu_scalar):
        assert lalu_mode == udc.LALU_MODE.MODE_RECIP and lalu_scalar == 1.0
        assert vector_sram_start_addr == output_sram_wb_addr == 0
        assert matrix_sram_start_addr == 0x80000 and K == N == 64
        matrix = self.sram[0x80000 // 2:0x80000 // 2 + K * N].float().reshape(N, K)
        result = matrix @ self.sram[:K].float()
        self.sram[:N].copy_(1 / result)


def manual_lstm(module, source, *, bf16=False, skip=True):
    """Independent PyTorch gate unroll, optionally rounding at device boundaries."""
    rounded = (lambda x: x.to(torch.bfloat16).float()) if bf16 else (lambda x: x)
    def tanh(value):
        if bf16:
            value = rounded(value.clamp(-4, 4))
            square = rounded(value * value)
            numerator = rounded(square * rounded(torch.tensor(1 / 135135)))
            for coefficient in (378 / 135135, 17325 / 135135):
                numerator = rounded(rounded(numerator + rounded(torch.tensor(coefficient))) * square)
            numerator = rounded(numerator + 1)
            denominator = rounded(square * rounded(torch.tensor(28 / 135135)))
            for coefficient in (3150 / 135135, 62370 / 135135):
                denominator = rounded(rounded(denominator + rounded(torch.tensor(coefficient))) * square)
            denominator = rounded(denominator + 1)
            return rounded(value * rounded(numerator * rounded(1 / denominator))).clamp(-1, 1)
        return torch.tanh(value)
    original = source
    for layer in range(2):
        w_i = rounded(getattr(module, f"weight_ih_l{layer}"))
        w_h = rounded(getattr(module, f"weight_hh_l{layer}"))
        b_i = rounded(getattr(module, f"bias_ih_l{layer}"))
        b_h = rounded(getattr(module, f"bias_hh_l{layer}"))
        projections = rounded(source @ w_i.T + b_i)
        hidden = torch.zeros(source.shape[-1])
        cell = torch.zeros_like(hidden)
        rows = []
        for projection in projections:
            recurrent = rounded(hidden @ w_h.T + b_h)
            i, f, g, o = rounded(projection + recurrent).chunk(4)
            i, f, o = (rounded(torch.sigmoid(v)) for v in (i, f, o))
            g = rounded(tanh(g))
            cell = rounded(rounded(f * cell) + rounded(i * g))
            hidden = rounded(o * rounded(tanh(cell)))
            rows.append(hidden)
        source = torch.stack(rows)
    return rounded(source + original) if skip else source


class BigCodecLSTMTest(unittest.TestCase):
    def model(self, width):
        torch.manual_seed(width)
        module = torch.nn.LSTM(width, width, num_layers=2, batch_first=True).eval()
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.mul_(0.3)
        return module

    def prepare(self, module, source, skip=True):
        sequence, width = source.shape
        padded = (width + 63) // 64 * 64
        engine = MemoryEngine()
        image = MemoryImage(engine)
        input_address, output_address = 0x80000000, 0x81000000
        scratch_address, identity_address, zero_address = 0x82000000, 0x83000000, 0x83010000
        packed = torch.zeros((sequence, padded), dtype=torch.bfloat16)
        packed[:, :width] = source
        engine.regions[input_address] = packed.flatten().clone()
        engine.regions[output_address] = torch.full((sequence * padded,), float("nan"), dtype=torch.bfloat16)
        engine.regions[scratch_address] = torch.full((scratch_bytes(source.shape) // 2,), float("nan"), dtype=torch.bfloat16)
        engine.regions[identity_address] = torch.eye(64, dtype=torch.bfloat16).flatten()
        engine.regions[zero_address] = torch.zeros(64, dtype=torch.bfloat16)
        plan = prepare_lstm(module, image, input_shape=source.shape,
                            input_address=input_address, output_address=output_address,
                            scratch_address=scratch_address, identity_address=identity_address,
                            zero_address=zero_address, skip=skip)
        return engine, image, plan

    def test_manual_gate_order_and_whole_stack_skip_match_torch(self):
        for width in (7, 64):
            module = self.model(width)
            source = torch.randn(5, width) * 0.2
            with torch.no_grad():
                official, _ = module(source[None])
                actual = manual_lstm(module, source)
            torch.testing.assert_close(actual, official[0] + source, rtol=2e-6, atol=2e-7)

    def test_device_recurrence_matches_independent_bf16_equations(self):
        for width, sequence, skip in ((64, 1, False), (64, 3, True), (65, 4, True)):
            with self.subTest(width=width, sequence=sequence, skip=skip):
                module = self.model(width)
                source = (torch.randn(sequence, width) * 0.2).to(torch.bfloat16).float()
                engine, _image, plan = self.prepare(module, source, skip)
                emit_lstm(engine, plan)
                actual = engine.view(plan.output_address, sequence * plan.padded_width).reshape(sequence, plan.padded_width).float()
                expected = manual_lstm(module, source, bf16=True, skip=skip)
                torch.testing.assert_close(actual[:, :width], expected, rtol=0, atol=0)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertEqual(float(actual[:, width:].abs().sum()), 0)
                projection_calls = [call for call in engine.matmul_calls if call["weight"] in {layer.input_weights for layer in plan.layers}]
                self.assertEqual([call["M"] for call in projection_calls], [sequence, sequence])
                self.assertTrue(all(call["K"] == call["N"] == 64 for call in engine.matmul_calls if call["sigmoid"]))

    def test_each_execution_resets_both_layers_state(self):
        module = self.model(64)
        engine, _image, plan = self.prepare(module, torch.randn(3, 64).to(torch.bfloat16).float())
        emit_lstm(engine, plan)
        first = engine.regions[plan.output_address].clone()
        emit_lstm(engine, plan)
        torch.testing.assert_close(engine.regions[plan.output_address], first, rtol=0, atol=0)

    def test_reject_overlaps_and_unsupported_lstm(self):
        module = self.model(64)
        engine = MemoryEngine()
        kwargs = dict(input_shape=(2, 64), input_address=0x80000000,
                      output_address=0x80001000, scratch_address=0x80002000,
                      identity_address=0x80010000, zero_address=0x80020000)
        for override in ({"output_address": 0x80000000}, {"scratch_address": 0x80002001}):
            with self.assertRaises(ValueError):
                prepare_lstm(module, MemoryImage(engine), **(kwargs | override))
        with self.assertRaises(ValueError):
            prepare_lstm(module.train(), MemoryImage(engine), **kwargs)
        with self.assertRaises(ValueError):
            prepare_lstm(torch.nn.LSTM(64, 64, bidirectional=True, batch_first=True).eval(), MemoryImage(engine), **kwargs)

    def test_pade_tanh_preserves_tiny_values_and_bounds_full_range(self):
        tiny = torch.tensor([-1e-5, -1e-3, 0, 1e-3, 1e-5]).bfloat16().float()
        actual = pade_tanh(tiny, bf16=True)
        torch.testing.assert_close(actual, tiny, rtol=0, atol=0)
        self.assertEqual(float(actual[2]), 0.)
        for bound, max_error, relative_error in ((.01, 4e-7, 3e-5), (.1, .0008, .004), (4., .015, .005)):
            positive = torch.linspace(0, bound, 10001).bfloat16().float()
            x = torch.cat((-positive.flip(0), positive))
            actual, expected = pade_tanh(x, bf16=True), x.tanh()
            error = actual - expected
            self.assertLess(float(error.abs().max()), max_error)
            self.assertLess(float(error.norm() / expected.norm()), relative_error)
            self.assertTrue(torch.equal(actual, -actual.flip(0)))
            self.assertTrue(torch.isfinite(actual).all())
        outside = pade_tanh(torch.tensor([-1e30, -20, 20, 1e30]), bf16=True)
        self.assertTrue(torch.isfinite(outside).all())
        self.assertTrue(bool((outside.abs() <= 1).all()))
        x = torch.linspace(-4, 4, 10001, dtype=torch.float64)
        self.assertLess(float((pade_tanh(x) - x.tanh()).abs().max()), 2e-5)

    def test_emitted_pade_tanh_matches_bf16_contract_across_chunks(self):
        count = 4160
        engine = MemoryEngine()
        source, output, scratch, identity = 0x80000000, 0x81000000, 0x82000000, 0x83000000
        value = torch.linspace(-6, 6, count).bfloat16().float()
        value[:5] = torch.tensor([-1e-5, -1e-3, 0, 1e-3, 1e-5]).bfloat16().float()
        engine.regions[source] = value.bfloat16().clone()
        engine.regions[output] = torch.full((count,), float("nan"), dtype=torch.bfloat16)
        engine.regions[scratch] = torch.full((tanh_scratch_bytes(count) // 2,), float("nan"), dtype=torch.bfloat16)
        engine.regions[identity] = torch.eye(64, dtype=torch.bfloat16).flatten()
        tanh_identity(engine, source, output, count, identity, scratch_address=scratch)
        torch.testing.assert_close(engine.regions[output].float(), pade_tanh(value, bf16=True), rtol=0, atol=0)
        self.assertEqual(tanh_scratch_bytes(count), 24576)
        tanh_identity(engine, source, source, count, identity, scratch_address=scratch)
        torch.testing.assert_close(engine.regions[source], engine.regions[output], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
