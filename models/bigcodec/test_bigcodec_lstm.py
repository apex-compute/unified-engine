"""Independent gate equations, BF16 boundaries, state reset and memory bounds."""

import contextlib
import io
import unittest

import torch

from bigcodec_lstm import emit_lstm, prepare_lstm, scratch_bytes, tanh_identity, tanh_scratch_bytes, pade_tanh, quantize_recurrent_if8, udc


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
    def __init__(self):
        self.regions = {}
        self.matmul_calls = []
        self.native_calls = []
        self.quantized_calls = []
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

    def accelerator_memory_to_scale_sram(self, address, count):
        assert count <= udc.SCALE_BRAM_ELEMENTS
        self.dma_calls.append(("scale", address, count * 2))
        self.scales = self.view(address, count).clone()

    def start_queue_for_dot_product_operation(self, *, max_clear_en, fmax_context_addr,
            vector_sram_start_addr, output_sram_wb_addr, K, N, dma_start_addr,
            data_type, bias_enable=False):
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

    def prepare(self, module, source, skip=True, recurrent_precision="bf16"):
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
                            zero_address=zero_address, skip=skip,
                            recurrent_precision=recurrent_precision)
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
        for width in (65, 1536):
            module = self.model(width)
            source = (torch.randn(2, width) * .2).bfloat16().float()
            engine, _, plan = self.prepare(module, source, recurrent_precision="if8")
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
            torch.testing.assert_close(actual[:, :width].float(), manual_lstm(module, source, bf16=True), rtol=0, atol=0)
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
