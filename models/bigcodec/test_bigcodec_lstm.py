"""Independent gate equations, BF16 boundaries, state reset and memory bounds."""

import contextlib
import io
import unittest
from unittest import mock
import sys

import torch

from bigcodec_lstm import emit_lstm, prepare_lstm, scratch_bytes, tanh_identity, tanh_scratch_bytes, pade_tanh, quantize_recurrent_if8, udc
from bigcodec_lstm import _two_product_sram, _two_sum_sram, _compensated_cell_sram
from bigcodec_lstm import _tanh_sram


class MemoryImage:
    def __init__(self, engine):
        self.engine = engine
        self.cursor = 0x10000000

    def allocate(self, value, *, alignment=128):
        self.cursor = (self.cursor + alignment - 1) // alignment * alignment
        address = self.cursor
        stored = value if value.dtype == torch.uint8 else value.to(torch.bfloat16)
        self.engine.regions[address] = stored.flatten().clone()
        self.cursor += stored.numel() * stored.element_size()
        return address


class MemoryEngine:
    """Direct BF16 boundary interpreter, not the native BF19 arithmetic model."""
    def __init__(self):
        self.regions = {}
        self.matmul_calls = []
        self.native_calls = []
        self.quantized_calls = []
        self.projection_calls = []
        self.dma_calls = []
        self.sram = torch.zeros(0x100000 // 2, dtype=torch.bfloat16)

    def view(self, address, count):
        assert address % 128 == 0 and count % 64 == 0
        for start, buffer in self.regions.items():
            if buffer.dtype != torch.bfloat16:
                continue
            offset = (address - start) // 2
            if address >= start and offset + count <= buffer.numel():
                return buffer[offset:offset + count]
        raise AssertionError(f"Out-of-bounds memory access {address:#x}, {count}")

    def view_bytes(self, address, count):
        for start, buffer in self.regions.items():
            raw = buffer.view(torch.uint8)
            if address >= start and address - start + count <= raw.numel():
                return raw[address - start:address - start + count]
        raise AssertionError(f"Out-of-bounds byte access {address:#x}, {count}")

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
        self.dma_calls.append(("read", address, count * 2))
        self.sram[sram_address // 2:sram_address // 2 + count].copy_(self.view(address, count))

    def accelerator_memory_to_bias_sram(self, address, count):
        self.dma_calls.append(("bias", address, count * 2))
        self.bias = self.view(address, count).float().clone()
        self.bias_address = address

    def accelerator_memory_to_scale_sram(self, address, count):
        assert count <= udc.SCALE_BRAM_ELEMENTS
        self.dma_calls.append(("scale", address, count * 2))
        self.scales = self.view(address, count).clone()

    def start_queue_for_dot_product_operation(self, *, max_clear_en, fmax_context_addr,
            vector_sram_start_addr, output_sram_wb_addr, K, N, dma_start_addr,
            data_type, bias_enable=False, lalu_mode=udc.LALU_MODE.BYPASS, lalu_a=0, lalu_b=0):
        assert data_type == udc.TYPE.IF8 and bool((self.scales < 0).all())
        assert K % 64 == N % 64 == 0 and self.scales.numel() == N * K // 64
        self.quantized_calls.append((K, N, dma_start_addr, output_sram_wb_addr))
        self.dma_calls.append(("stream", dma_start_addr, N * K))
        codes = self.view_bytes(dma_start_addr, N * K).view(torch.int8).reshape(-1, 64)
        # Quantized-weight sensitivity reference: driver dequantization to BF16.
        weights = (codes.float() * self.scales.float().abs()[:, None]).bfloat16().float().reshape(N, K)
        result = (self.sram_view(vector_sram_start_addr, K).float()[None] @ weights.T)[0]
        if bias_enable:
            result += self.bias[:N]
            self.projection_calls.append(dict(K=K, N=N, output=output_sram_wb_addr,
                bias=self.bias_address, mode=lalu_mode, lalu_a=lalu_a, lalu_b=lalu_b))
        if lalu_mode == udc.LALU_MODE.ACT_NO_X:
            assert lalu_a == udc.LALU_ACT_SIGMOID_A and lalu_b == udc.LALU_ACT_SIGMOID_B
            result = result.sigmoid()
        else:
            assert lalu_mode == udc.LALU_MODE.BYPASS
        self.sram_view(output_sram_wb_addr, N).copy_(result)

    def sram_to_accelerator_memory(self, sram_address, address, count):
        assert sram_address % 128 == 0 and sram_address // 2 + count <= self.sram.numel()
        self.dma_calls.append(("write", address, count * 2))
        self.view(address, count).copy_(self.sram[sram_address // 2:sram_address // 2 + count])

    def sram_view(self, address, count):
        assert address >= 0 and address % 128 == 0 and count % 64 == 0
        assert address // 0x80000 == (address + count * 2 - 1) // 0x80000
        assert address + count * 2 <= 0x100000
        return self.sram[address // 2:address // 2 + count]

    def broadcast_mul(self, scalar, sram_start_addr, sram_wb_addr, element_size):
        assert sram_start_addr < 0x80000
        self.sram_view(sram_wb_addr, element_size).copy_(
            self.sram_view(sram_start_addr, element_size).float() * scalar)

    def broadcast_add(self, scalar, sram_start_addr, sram_wb_addr, element_size):
        assert sram_start_addr < 0x80000
        self.sram_view(sram_wb_addr, element_size).copy_(
            self.sram_view(sram_start_addr, element_size).float() + scalar)

    def eltwise_mul_core(self, a, b, output, count):
        assert a // 0x80000 != b // 0x80000
        self.sram_view(output, count).copy_(self.sram_view(a, count).float() * self.sram_view(b, count).float())

    def eltwise_add_core(self, a, b, output, count):
        assert a // 0x80000 != b // 0x80000
        self.sram_view(output, count).copy_(self.sram_view(a, count).float() + self.sram_view(b, count).float())

    def eltwise_sub_core(self, a, b, output, count):
        assert a < 0x80000 <= b
        self.sram_view(output, count).copy_(self.sram_view(a, count).float() - self.sram_view(b, count).float())

    def start_queue_for_maxpool2d_operation(self, *, act_sram_start_addr,
            output_sram_wb_addr, kernel_w, kernel_h, out_w, out_h, w_pad, stride_s):
        assert kernel_w == out_h == stride_s == 1 and kernel_h == 2
        count = out_w * 64
        first = self.sram_view(act_sram_start_addr, count).clone()
        second = self.sram_view(act_sram_start_addr + w_pad * 128, count)
        self.sram_view(output_sram_wb_addr, count).copy_(torch.maximum(first, second))

    def float_to_bf19(self, value):
        return value

    def float_to_bf16(self, value):
        return float(torch.tensor(value).bfloat16())

    def start_queue_for_bf16_matvec_operation(self, *, max_clear_en, fmax_context_addr,
            vector_sram_start_addr, matrix_sram_start_addr, output_sram_wb_addr,
            K, N, lalu_mode=udc.LALU_MODE.BYPASS, lalu_scalar=0, lalu_a=0, lalu_b=0, bias_enable=False):
        assert vector_sram_start_addr < 0x80000 <= matrix_sram_start_addr
        self.native_calls.append(lalu_mode)
        matrix = self.sram_view(matrix_sram_start_addr, K * N).float().reshape(N, K)
        result = (self.sram_view(vector_sram_start_addr, K).float()[None] @ matrix.T)[0]
        if bias_enable:
            result += self.bias[:N]
            self.projection_calls.append(dict(K=K, N=N, output=output_sram_wb_addr,
                bias=self.bias_address, mode=lalu_mode, lalu_a=lalu_a, lalu_b=lalu_b))
        if lalu_mode == udc.LALU_MODE.MODE_RECIP:
            assert lalu_scalar == 1.0
            result = 1 / result
        elif lalu_mode == udc.LALU_MODE.CLAMP:
            result = result.clamp(lalu_a, lalu_b)
        elif lalu_mode == udc.LALU_MODE.ACT_NO_X:
            assert lalu_a == udc.LALU_ACT_SIGMOID_A and lalu_b == udc.LALU_ACT_SIGMOID_B
            result = result.sigmoid()
        elif lalu_mode != udc.LALU_MODE.BYPASS:
            raise AssertionError(lalu_mode)
        self.sram_view(output_sram_wb_addr, N).copy_(result)


def native_round(value):
    """Native finite normal arithmetic: FP32 ->10 fraction bits ->BF16, RNE.

    This models the elementwise operations used by the compensated cell. It
    does not model matrix accumulation, LALU activation tables, or underflow.
    """
    value = torch.as_tensor(value, dtype=torch.float32).contiguous()
    bits = value.view(torch.int32)
    rounded = (bits + 0xFFF + ((bits >> 13) & 1)) & ~0x1FFF
    return rounded.view(torch.float32).bfloat16().float()


class NativeArithmeticMemoryEngine(MemoryEngine):
    """Model the measured BF19-to-BF16 double rounding for scalar/vector ALU."""
    def broadcast_mul(self, scalar, sram_start_addr, sram_wb_addr, element_size):
        assert sram_start_addr < 0x80000
        self.sram_view(sram_wb_addr, element_size).copy_(native_round(
            self.sram_view(sram_start_addr, element_size).float() * scalar))

    def broadcast_add(self, scalar, sram_start_addr, sram_wb_addr, element_size):
        assert sram_start_addr < 0x80000
        self.sram_view(sram_wb_addr, element_size).copy_(native_round(
            self.sram_view(sram_start_addr, element_size).float() + scalar))

    def _binary(self, a, b, output, count, operation):
        assert a < 0x80000 <= b
        self.sram_view(output, count).copy_(native_round(operation(
            self.sram_view(a, count).float(), self.sram_view(b, count).float())))

    def eltwise_mul_core(self, a, b, output, count):
        self._binary(a, b, output, count, torch.mul)

    def eltwise_add_core(self, a, b, output, count):
        self._binary(a, b, output, count, torch.add)

    def eltwise_sub_core(self, a, b, output, count):
        self._binary(a, b, output, count, torch.sub)


def manual_lstm(module, source, *, bf16=False, skip=True, compensated_cell=False,
                preserve_cell_residual=True, fused_projection=False):
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
        if fused_projection:
            b_i = rounded(getattr(module, f"bias_ih_l{layer}").float()
                          + getattr(module, f"bias_hh_l{layer}").float())
        projections = rounded(source @ w_i.T + b_i)
        hidden = torch.zeros(source.shape[-1])
        cell = torch.zeros_like(hidden)
        cell_low = torch.zeros_like(hidden)
        rows = []
        for projection in projections:
            if fused_projection:
                # Independent fused epilogue reference: no gate BF16 boundary
                # between the recurrent dot, projected-input bias, and sigmoid.
                i, f, g, o = (hidden @ w_h.T + projection).chunk(4)
                g = rounded(g)
            else:
                recurrent = rounded(hidden @ w_h.T + b_h)
                i, f, g, o = rounded(projection + recurrent).chunk(4)
            i, f, o = (rounded(torch.sigmoid(v)) for v in (i, f, o))
            g = rounded(tanh(g))
            if compensated_cell:
                # Independent oracle: obtain exact residuals with FP32 rather
                # than reproducing the emitted Dekker/TwoSum instruction chain.
                left_exact, right_exact = f * cell, i * g
                left, right = rounded(left_exact), rounded(right_exact)
                left_error, right_error = left_exact - left, right_exact - right
                total = rounded(left + right)
                total_error = (left + right) - total
                correction = rounded(rounded(left_error + right_error) + total_error)
                if preserve_cell_residual:
                    correction = rounded(correction + rounded(f * cell_low))
                cell = rounded(total + correction)
                if preserve_cell_residual:
                    cell_low = rounded((total + correction) - cell)
            else:
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

    def prepare(self, module, source, skip=True, recurrent_precision="bf16", compensated_cell=False,
                preserve_cell_residual=True, compensated_tanh=False, fused_projection=False):
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
        engine.regions[scratch_address] = torch.full((scratch_bytes(source.shape,
            compensated_cell=compensated_cell, preserve_cell_residual=preserve_cell_residual) // 2,), float("nan"), dtype=torch.bfloat16)
        engine.regions[identity_address] = torch.eye(64, dtype=torch.bfloat16).flatten()
        engine.regions[zero_address] = torch.zeros(64, dtype=torch.bfloat16)
        plan = prepare_lstm(module, image, input_shape=source.shape,
                            input_address=input_address, output_address=output_address,
                            scratch_address=scratch_address, identity_address=identity_address,
                            zero_address=zero_address, skip=skip,
                            recurrent_precision=recurrent_precision, compensated_cell=compensated_cell,
                            preserve_cell_residual=preserve_cell_residual, compensated_tanh=compensated_tanh,
                            fused_projection=fused_projection)
        return engine, image, plan

    def test_compensated_products_recover_exact_bf16_products_and_preserve_sram(self):
        gen = torch.Generator().manual_seed(15092026)
        for count in (64, 1536, 4096):
            engine = MemoryEngine()
            engine.sram.fill_(-13.)
            left = (torch.randn(count, generator=gen) * 20.).bfloat16()
            right = (torch.rand(count, generator=gen) * 2. - 1.).bfloat16()
            left[:4] = torch.tensor([1.0078125, -1.0078125, 16., 0.])
            right[:4] = torch.tensor([.99609375, .99609375, .00390625, -1.])
            engine.sram_view(0, count).copy_(left)
            engine.sram_view(0x2000, count).copy_(right)
            before = engine.sram.clone()
            _two_product_sram(engine, 0, 0x2000, 0x4000, 0x6000, count)
            high, low = (engine.sram_view(address, count).float() for address in (0x4000, 0x6000))
            torch.testing.assert_close(high + low, left.float() * right.float(), rtol=0, atol=0)
            allowed = torch.zeros(engine.sram.numel(), dtype=torch.bool)
            for address in (0x4000, 0x6000, *(0x50000 + n * 0x2000 for n in range(7)), 0xF0000):
                allowed[address // 2:address // 2 + count] = True
            torch.testing.assert_close(engine.sram[~allowed], before[~allowed], rtol=0, atol=0)

    def test_compensated_sum_preserves_cancellation_and_both_operand_orders(self):
        engine = MemoryEngine()
        left = torch.tensor([16., 1., 1.0078125, .00390625, -16., -1., 0., -0.]).repeat(8).bfloat16()
        right = torch.tensor([.00390625, -1., -1., 16., -.00390625, 1., 0., 0.]).repeat(8).bfloat16()
        for a, b in ((left, right), (right, left)):
            engine.sram_view(0, 64).copy_(a)
            engine.sram_view(0x2000, 64).copy_(b)
            _two_sum_sram(engine, 0, 0x2000, 0x4000, 0x6000, 64)
            high = engine.sram_view(0x4000, 64).float()
            low = engine.sram_view(0x6000, 64).float()
            torch.testing.assert_close(high + low, a.float() + b.float(), rtol=0, atol=0)

    def test_native_double_rounding_product_residuals_recover_exact_products(self):
        engine = NativeArithmeticMemoryEngine()
        values = torch.arange(128, 256).float() / 128.
        left = values[:, None].expand(128, 128).flatten()
        right = (values[None, :] / 2.).expand(128, 128).flatten()
        different_from_direct_rounding = 0
        for start in range(0, left.numel(), 4096):
            a, b = left[start:start + 4096], right[start:start + 4096]
            engine.sram_view(0, 4096).copy_(a)
            engine.sram_view(0x2000, 4096).copy_(b)
            _two_product_sram(engine, 0, 0x2000, 0x4000, 0x6000, 4096)
            high = engine.sram_view(0x4000, 4096).float()
            low = engine.sram_view(0x6000, 4096).float()
            torch.testing.assert_close(high, native_round(a * b), rtol=0, atol=0)
            torch.testing.assert_close(high + low, a * b, rtol=0, atol=0)
            different_from_direct_rounding += int((high != (a * b).bfloat16().float()).sum())
        self.assertGreater(different_from_direct_rounding, 0)

    def test_native_double_rounding_sum_residual_is_approximate_and_bounded(self):
        engine = NativeArithmeticMemoryEngine()
        gen = torch.Generator().manual_seed(15092026)
        left = (torch.randn(4096, generator=gen) * 20.).bfloat16().float()
        right = (torch.rand(4096, generator=gen) * 2. - 1.).bfloat16().float()
        left[0], right[0] = 1.0078125, -.0034332275390625
        engine.sram_view(0, 4096).copy_(left)
        engine.sram_view(0x2000, 4096).copy_(right)
        _two_sum_sram(engine, 0, 0x2000, 0x4000, 0x6000, 4096)
        high, low = (engine.sram_view(address, 4096).float() for address in (0x4000, 0x6000))
        # This counterexample fails the ordinary exact-TwoSum assumption:
        # native double rounding picks1.0; the remaining error needs9bits.
        self.assertEqual(float(high[0]), 1.)
        self.assertEqual(float(low[0]), .00439453125)
        self.assertNotEqual(float(high[0] + low[0]), float(left[0] + right[0]))
        expected = left.double() + right.double()
        recovered = high.double() + low.double()
        self.assertLess(float((recovered - expected).norm() / expected.norm()), 2e-6)
        self.assertLess(float((recovered - expected).abs().max()), .002)
        self.assertLess(float((recovered - expected).norm()), float((high.double() - expected).norm()) / 100)

    def test_compensated_cell_keeps_updates_smaller_than_cell_ulp(self):
        engine = MemoryEngine()
        for address, value in ((0, 1.), (0x2000, 1.), (0x4000, 1. / 256), (0x6000, 16.), (0x8000, 0.)):
            engine.sram_view(address, 64).fill_(value)
        for _ in range(64):
            _compensated_cell_sram(engine, 0, 0x2000, 0x4000, 0x6000, 0x8000, 64)
        torch.testing.assert_close(engine.sram_view(0x6000, 64).float()
            + engine.sram_view(0x8000, 64).float(), torch.full((64,), 16.25), rtol=0, atol=0)
        self.assertEqual(float((torch.tensor(16.) + 1. / 256).bfloat16()), 16.)
        for address, value in ((0, 1.), (0x2000, 1.), (0x4000, 1. / 256)):
            self.assertTrue(bool((engine.sram_view(address, 64) == value).all()))

    def test_compensated_recurrence_matches_independent_residual_oracle_and_resets(self):
        for width, sequence in ((65, 4), (1536, 2)):
            module = self.model(width)
            source = (torch.randn(sequence, width) * .2).bfloat16().float()
            engine, _, plan = self.prepare(module, source, compensated_cell=True)
            self.assertTrue(plan.compensated_cell)
            self.assertEqual(plan.scratch_bytes - scratch_bytes(source.shape), plan.padded_width * 2)
            self.assertIn("cell_low", plan.regions)
            emit_lstm(engine, plan)
            actual = engine.regions[plan.output_address].reshape(sequence, plan.padded_width).float().clone()
            expected = manual_lstm(module, source, bf16=True, compensated_cell=True)
            torch.testing.assert_close(actual[:, :width], expected, rtol=0, atol=0)
            self.assertTrue(bool((actual[:, width:] == 0).all()))
            engine.view(plan.regions["cell_low"][0], plan.padded_width).fill_(123.)
            emit_lstm(engine, plan)
            torch.testing.assert_close(engine.regions[plan.output_address].float(), actual.flatten(), rtol=0, atol=0)

    def test_product_compensation_has_no_persistent_low_cell_and_matches_oracle(self):
        source = (torch.randn(4, 65) * .2).bfloat16().float()
        module = self.model(65)
        engine, _, plan = self.prepare(module, source, compensated_cell=True, preserve_cell_residual=False)
        self.assertEqual(plan.scratch_bytes, scratch_bytes(source.shape))
        self.assertNotIn("cell_low", plan.regions)
        emit_lstm(engine, plan)
        actual = engine.regions[plan.output_address].reshape(4, 128).float()
        expected = manual_lstm(module, source, bf16=True, compensated_cell=True, preserve_cell_residual=False)
        torch.testing.assert_close(actual[:, :65], expected, rtol=0, atol=0)
        self.assertTrue(bool((actual[:, 65:] == 0).all()))

    def test_tanh_precision_dispatch_applies_to_candidate_and_cell_of_both_layers(self):
        source = torch.zeros(2, 64)
        engine, _, plan = self.prepare(self.model(64), source, compensated_tanh=True)
        replacement = mock.Mock(side_effect=_tanh_sram)
        module = mock.Mock(compensated_tanh_sram=replacement)
        with mock.patch.dict(sys.modules, {"bigcodec_tanh": module}):
            emit_lstm(engine, plan)
        self.assertEqual(replacement.call_count, 2 * 2 * 2)
        self.assertTrue(all(call.args[4 - 1] == 64 for call in replacement.call_args_list))

    def test_fused_projection_folds_biases_before_bf16_packing_without_mutation(self):
        module = self.model(65)
        with torch.no_grad():
            for layer in range(2):
                getattr(module, f"bias_ih_l{layer}").fill_(1.00390625)
                getattr(module, f"bias_hh_l{layer}").fill_(.00390625)
        before = {name: value.detach().clone() for name, value in module.named_parameters()}
        engine, _, plan = self.prepare(module, torch.zeros(2, 65), fused_projection=True)
        for layer in plan.layers:
            packed = engine.regions[layer.input_bias].reshape(4, 128)
            self.assertTrue(bool((packed[:, :65] == 1.0078125).all()))
            self.assertTrue(bool((packed[:, 65:] == 0).all()))
            self.assertTrue(bool((engine.regions[layer.recurrent_bias] == 0).all()))
        separately_rounded = (torch.tensor(1.00390625).bfloat16().float()
                              + torch.tensor(.00390625).bfloat16().float()).bfloat16().float()
        self.assertNotEqual(float(separately_rounded), 1.0078125)
        for name, value in module.named_parameters():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_fused_projection_matches_oracle_and_keeps_gate_tiles_separate(self):
        for width, precision, compensated in ((65, "bf16", True), (1536, "if8", False)):
            with self.subTest(width=width, precision=precision):
                module = self.model(width)
                source = (torch.randn(2, width) * .2).bfloat16().float()
                engine, _, plan = self.prepare(module, source, recurrent_precision=precision,
                    fused_projection=True, compensated_cell=compensated)
                if precision == "if8":
                    for index, layer in enumerate(plan.layers):
                        codes = engine.regions[layer.recurrent_weights].view(torch.int8).reshape(-1, 64)
                        scales = engine.regions[layer.recurrent_scales]
                        effective = (codes.float() * scales.float().abs()[:, None]).bfloat16()
                        effective = effective.reshape(4, plan.padded_width, plan.padded_width)[:, :width, :width]
                        with torch.no_grad():
                            getattr(module, f"weight_hh_l{index}").copy_(effective.reshape(4 * width, width))
                emit_lstm(engine, plan)
                actual = engine.regions[plan.output_address].reshape(2, plan.padded_width).float()
                expected = manual_lstm(module, source, bf16=True, fused_projection=True,
                                        compensated_cell=compensated)
                torch.testing.assert_close(actual[:, :width], expected, rtol=0, atol=0)
                self.assertTrue(bool((actual[:, width:] == 0).all()))
                row_bytes = 4 * plan.padded_width * 2
                rows_seen = set()
                for call in engine.projection_calls:
                    first = (call["output"] - 0x8000) // 2
                    gate = first // plan.padded_width
                    self.assertEqual((first + call["N"] - 1) // plan.padded_width, gate)
                    row_address = call["bias"] - first * 2
                    self.assertIn(row_address, {plan.regions["input_gates"][0],
                                               plan.regions["input_gates"][0] + row_bytes})
                    rows_seen.add(row_address)
                    mode = udc.LALU_MODE.BYPASS if gate == 2 else udc.LALU_MODE.ACT_NO_X
                    self.assertEqual(call["mode"], mode)
                    self.assertEqual(call["lalu_a"], 0 if gate == 2 else udc.LALU_ACT_SIGMOID_A)
                    self.assertEqual(call["lalu_b"], 0 if gate == 2 else udc.LALU_ACT_SIGMOID_B)
                self.assertEqual(len(rows_seen), 2)
                # There are no separate64x64 identity sigmoid passes after fusion.
                expected_native_sigmoids = sum(call["mode"] == udc.LALU_MODE.ACT_NO_X
                                              for call in engine.projection_calls) if precision == "bf16" else 0
                self.assertEqual(engine.native_calls.count(udc.LALU_MODE.ACT_NO_X), expected_native_sigmoids)
                if precision == "if8":
                    self.assertIn(256, [call["N"] for call in engine.projection_calls])

    def test_explicit_disabled_projection_fusion_preserves_legacy_capture(self):
        from bigcodec_device import shared
        source = torch.zeros(2, 64)
        module = self.model(64)
        first_engine, _, first = self.prepare(module, source)
        second_engine, _, second = self.prepare(module, source, fused_projection=False)
        self.assertEqual(first, second)
        for address in first_engine.regions:
            a, b = first_engine.regions[address], second_engine.regions[address]
            torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
        previous_width = udc.UE_AXI_DATA_WIDTH_BITS
        try:
            udc.UE_AXI_DATA_WIDTH_BITS = 256
            programs = []
            for plan in (first, second):
                capture = shared._WholeGraphEngine(0xA0000000)
                capture.start_capture()
                with contextlib.redirect_stdout(io.StringIO()):
                    emit_lstm(capture, plan)
                capture.generate_instruction_halt()
                capture.stop_capture()
                programs.append(b''.join(instruction.get_bytes() for instruction in capture.capture_buffer))
            self.assertEqual(programs[0], programs[1])
        finally:
            udc.UE_AXI_DATA_WIDTH_BITS = previous_width

    def test_compensated_workspace_rejects_overlap_and_unsupported_width(self):
        engine = MemoryEngine()
        for helper in (_two_product_sram, _two_sum_sram):
            for addresses in ((0, 0x2000, 0, 0x6000), (0, 0x2000, 0x4000, 0x4000),
                              (0x50000, 0x2000, 0x4000, 0x6000), (1, 0x2000, 0x4000, 0x6000)):
                with self.assertRaises(ValueError):
                    helper(engine, *addresses, 64)
        with self.assertRaises(ValueError):
            scratch_bytes((2, 4097), compensated_cell=True)
        with self.assertRaises(ValueError):
            scratch_bytes((2, 64), compensated_cell=1)
        with self.assertRaises(ValueError):
            scratch_bytes((2, 64), preserve_cell_residual=1)
        # At4096 columns a64-row BF16 recurrent tile is one URAM row too big.
        # Meta tensors let us validate dispatch without allocating1GiB weights.
        module = torch.nn.LSTM(4096, 4096, num_layers=2, batch_first=True, device="meta").eval()
        with self.assertRaisesRegex(ValueError, "SRAM recurrent projection"):
            self.prepare(module, torch.zeros(1, 4096), compensated_cell=True)
        with self.assertRaisesRegex(ValueError, "SRAM recurrent projection"):
            self.prepare(module, torch.zeros(1, 4096), compensated_tanh=True)
        with self.assertRaisesRegex(ValueError, "SRAM recurrent projection"):
            self.prepare(module, torch.zeros(1, 4096), fused_projection=True)
        with self.assertRaisesRegex(ValueError, "fused_projection must be a bool"):
            self.prepare(self.model(64), torch.zeros(1, 64), fused_projection=1)
        _, _, plan = self.prepare(self.model(64), torch.zeros(2, 64))
        self.assertFalse(plan.compensated_cell)
        self.assertNotIn("cell_low", plan.regions)

    def test_manual_gate_order_and_whole_stack_skip_match_torch(self):
        for width in (7, 64):
            module = self.model(width)
            source = torch.randn(5, width) * 0.2
            with torch.no_grad():
                official, _ = module(source[None])
                actual = manual_lstm(module, source)
            torch.testing.assert_close(actual, official[0] + source, rtol=2e-6, atol=2e-7)

    def test_device_recurrence_matches_independent_bf16_equations(self):
        for width, sequence, skip in ((64, 1, False), (64, 3, True), (65, 4, True), (1536, 2, False)):
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
                self.assertEqual(engine.native_calls.count(udc.LALU_MODE.ACT_NO_X),
                                 2 * sequence * 3 * plan.padded_width // 64)
                # Initial h/c reset, weight+bias tiles, hidden/projection/identity/
                # cell reads, and cell/output writes. Gates never spill to DRAM.
                tile = min(4 * plan.padded_width,
                           udc.URAM_NEAR_FULL_ELEMENTS // plan.padded_width // 64 * 64)
                tiles = (4 * plan.padded_width + tile - 1) // tile
                self.assertEqual(len(engine.dma_calls),
                                 2 * (4 * plan.padded_width // 64 + (6 + 2 * tiles) * sequence))

    def test_each_execution_resets_both_layers_state(self):
        module = self.model(64)
        engine, _image, plan = self.prepare(module, torch.randn(3, 64).to(torch.bfloat16).float())
        emit_lstm(engine, plan)
        first = engine.regions[plan.output_address].clone()
        emit_lstm(engine, plan)
        torch.testing.assert_close(engine.regions[plan.output_address], first, rtol=0, atol=0)

    def test_if8_packing_matches_driver_and_tags_zero_blocks(self):
        from bigcodec_device import shared
        torch.manual_seed(845)
        weight = torch.randn(64, 128).bfloat16()
        weight[:2] = 0
        codes, scales = quantize_recurrent_if8(weight)
        self.assertEqual(codes.dtype, torch.uint8)
        self.assertTrue(bool((scales < 0).all()))
        self.assertTrue(torch.equal(scales[:4], -torch.ones(4, dtype=torch.bfloat16)))
        dequantized = (codes.view(torch.int8).reshape(-1, 64).float()
            * scales.float().abs()[:, None]).bfloat16().reshape_as(weight)
        engine = shared._WholeGraphEngine(0xA0000000)
        expected = engine.quantize_weight_simulate(weight, udc.TYPE.IF8, int_variant=True)
        torch.testing.assert_close(dequantized, expected, rtol=0, atol=0)
        writes = []
        engine.dma_write = lambda device, address, value, size: writes.append(
            value.contiguous().view(torch.uint8).flatten()[:size].clone())
        with contextlib.redirect_stdout(io.StringIO()):
            engine.quantize_weight(weight, N=64, K=128, data_type=udc.TYPE.IF8, int_variant=True)
        self.assertEqual(len(writes), 2)
        torch.testing.assert_close(codes.flatten(), writes[0], rtol=0, atol=0)
        torch.testing.assert_close(scales.view(torch.uint8), writes[1], rtol=0, atol=0)
        for invalid in (torch.zeros(3, 64), torch.full((64, 64), float("nan"))):
            with self.assertRaises(ValueError):
                quantize_recurrent_if8(invalid)

    def test_if8_streaming_matches_dequantized_recurrent_reference(self):
        for width, compensated in ((65, False), (1536, False), (65, True)):
            module = self.model(width)
            source = (torch.randn(2, width) * .2).bfloat16().float()
            engine, _, plan = self.prepare(module, source, recurrent_precision="if8", compensated_cell=compensated)
            self.assertEqual(plan.recurrent_precision, "if8")
            for index, layer in enumerate(plan.layers):
                self.assertEqual(engine.regions[layer.input_weights].dtype, torch.bfloat16)
                self.assertEqual(engine.regions[layer.recurrent_weights].dtype, torch.uint8)
                codes = engine.regions[layer.recurrent_weights].view(torch.int8).reshape(-1, 64)
                scales = engine.regions[layer.recurrent_scales]
                effective = (codes.float() * scales.float().abs()[:, None]).bfloat16()
                effective = effective.reshape(4, plan.padded_width, plan.padded_width)[:, :width, :width]
                with torch.no_grad():
                    getattr(module, f"weight_hh_l{index}").copy_(effective.reshape(4 * width, width))
            emit_lstm(engine, plan)
            actual = engine.view(plan.output_address, 2 * plan.padded_width).reshape(2, plan.padded_width)
            torch.testing.assert_close(actual[:, :width].float(), manual_lstm(module, source,
                bf16=True, compensated_cell=compensated), rtol=0, atol=0)
            self.assertTrue(bool((actual[:, width:] == 0).all()))
            self.assertTrue(engine.quantized_calls)
            # Every layer streams exactly its entire packed matrix per timestep;
            # scale and code reads are checked against actual byte allocations.
            stream_bytes = sum(size for kind, _, size in engine.dma_calls if kind == "stream")
            self.assertEqual(stream_bytes, 2 * 2 * 4 * plan.padded_width ** 2)
            if width == 1536:
                self.assertEqual(max(call[1] for call in engine.quantized_calls), 320)

    def test_native_capture_has_valid_addresses_and_single_terminal_halt(self):
        from bigcodec_device import shared
        _, _, plan = self.prepare(self.model(64), torch.zeros(3, 64))
        _, _, quantized_plan = self.prepare(self.model(64), torch.zeros(3, 64), recurrent_precision="if8")
        previous_width = udc.UE_AXI_DATA_WIDTH_BITS
        try:
            udc.UE_AXI_DATA_WIDTH_BITS = 256
            engine = shared._WholeGraphEngine(0xA0000000)
            engine.start_capture()
            with contextlib.redirect_stdout(io.StringIO()):
                emit_lstm(engine, plan)
                emit_lstm(engine, quantized_plan)
                tanh_identity(engine, 0xB0000000, 0xB1000000, 4160,
                    plan.identity_address, scratch_address=0xB2000000)
            engine.generate_instruction_halt()
            engine.stop_capture()
            self.assertEqual(udc.check_isa_jumps(engine.capture_buffer, 0xA0000000), [])
            kinds = [(instruction.words[0] >> 8) & 15 for instruction in engine.capture_buffer]
            self.assertEqual(kinds.count(udc.INSTRUCTION_HALT), 1)
            self.assertNotIn(udc.INSTRUCTION_SWI, kinds)
            self.assertLess(engine.capture_count, 1000)
        finally:
            udc.UE_AXI_DATA_WIDTH_BITS = previous_width

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
        with self.assertRaisesRegex(ValueError, "precision"):
            prepare_lstm(module.eval(), MemoryImage(engine), recurrent_precision="fp8", **kwargs)

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
        self.assertEqual(len(engine.dma_calls), 5)  # One identity + input/output per chunk.
        self.assertTrue(torch.isnan(engine.regions[scratch].float()).all())
        self.assertEqual(tanh_scratch_bytes(count), 24576)
        tanh_identity(engine, source, source, count, identity, scratch_address=scratch)
        torch.testing.assert_close(engine.regions[source], engine.regions[output], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
