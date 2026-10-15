"""Independent tensor semantics and native capture for BF16 convolutions."""

import contextlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bigcodec_conv_bf16 as conv
from test_bigcodec_conv import MemoryEngine as DmaMemoryEngine, bf16_bytes


class MemoryEngine(DmaMemoryEngine):
    def __init__(self):
        super().__init__()
        self._isa_reg_counter = 1
        self.registers = {}
        self.dynamic_launches = 0
        self._inst_ptr_counter = 1
        self.pointers = {}
        self.loop = None
        self.capture_count = 0

    def alloc_inst_ptr(self):
        value = self._inst_ptr_counter
        self._inst_ptr_counter += 1
        return value

    def release_inst_ptr(self, pointer):
        self._inst_ptr_counter -= 1
        self.pointers.pop(pointer)

    def generate_instruction_pbi_init(self, inst_pointer_idx, **kwargs):
        self.pointers[inst_pointer_idx] = kwargs.copy()

    def get_program_dram_addr(self):
        return 0xA0000000

    def generate_instruction_jump_abs(self, address):
        pass

    def loop_start(self, loop_cnt):
        assert self.loop is None
        self.loop = (loop_cnt, [])

    def loop_end(self):
        count, operations = self.loop
        self.loop = None
        for operation, kwargs in operations:
            if operation == self._dot:
                pointer = self.pointers[kwargs["inst_pointer_idx"]]
                self.launches.append((count, pointer["uram_length"] * 64, pointer["output_size"]))
        for _ in range(count):
            for operation, kwargs in operations:
                operation(**kwargs)

    def _submit(self, operation, kwargs):
        if self.loop is not None:
            self.loop[1].append((operation, kwargs.copy()))
        else:
            operation(**kwargs)

    def start_queue_for_bf16_matvec_operation(self, **kwargs):
        self._submit(self._dot, kwargs)

    def _dot(self, *, vector_sram_start_addr, matrix_sram_start_addr,
             output_sram_wb_addr, K, N, bias_enable=False,
             inst_pointer_idx=None, **kwargs):
        if inst_pointer_idx is None:
            source, matrix, destination = vector_sram_start_addr, matrix_sram_start_addr, output_sram_wb_addr
            self.launches.append((1, K, N))
        else:
            pointer = self.pointers[inst_pointer_idx]
            source = pointer.get("uram_a_start_addr", 0) * 128
            matrix = 0x80000 + pointer.get("uram_b_start_addr", 0) * 128
            destination = pointer.get("uram_wb_addr", 0) * 128
            K, N = pointer["uram_length"] * 64, pointer["output_size"]
            pointer["uram_a_start_addr"] = source // 128 + vector_sram_start_addr // 128
            pointer["uram_wb_addr"] = destination // 128 + output_sram_wb_addr // 128
        assert source + K * 2 <= destination
        assert destination + conv.shared._align_up(N, 64) * 2 <= conv.udc.URAM_NEAR_FULL_SIZE
        vector = self.floats(source, K, sram=True)
        weights = self.floats(matrix, N * K, sram=True).reshape(N, K)
        output = weights @ vector
        if bias_enable:
            output += self.bias
        packed = np.zeros(conv.shared._align_up(N, 64), dtype=np.float32)
        packed[:N] = output
        data = np.frombuffer(bf16_bytes(packed), dtype=np.uint8)
        self.sram[destination:destination + data.size] = data

    def sram_to_accelerator_memory(self, source, destination, elements,
                                   *, inst_pointer_idx=None, **kwargs):
        self._submit(self._write, dict(source=source, destination=destination,
                                      elements=elements, inst_pointer_idx=inst_pointer_idx, **kwargs))

    def _write(self, *, source, destination, elements, inst_pointer_idx=None, **kwargs):
        if inst_pointer_idx is not None:
            pointer = self.pointers[inst_pointer_idx]
            actual_source = pointer.get("uram_a_start_addr", 0) * 128
            actual_destination = pointer["dram_shared_addr"]
            kwargs["memcpy_length_bytes"] = pointer["dma_length"]
            pointer["uram_a_start_addr"] = (actual_source + source) // 128
            pointer["dram_shared_addr"] += destination
            source, destination = actual_source, actual_destination
        super().sram_to_accelerator_memory(source, destination, elements, **kwargs)

    def alloc_isa_reg(self):
        register = self._isa_reg_counter
        self._isa_reg_counter += 1
        return register

    def generate_instruction_add_set(self, register, value):
        self.registers[register] = value

    def release_isa_reg(self):
        self._isa_reg_counter -= 1
        self.registers.pop(self._isa_reg_counter)

    def matmat_mul_core(self, *, M, K, N, A_DRAM_ADDR, B_DRAM_ADDR,
                        OUTPUT_DRAM_ADDR, C_DRAM_ADDR=None, gpr_M_reg=None):
        # Interpret the public matmul ABI independently of convolution plans.
        # DMA reads/writes retain the byte-addressed, initially dirty memory.
        if gpr_M_reg is not None:
            assert self.registers[gpr_M_reg] == M
            self.dynamic_launches += 1
        a = torch.from_numpy(self.floats(A_DRAM_ADDR, M * K).copy()).reshape(M, K)
        b = torch.from_numpy(self.floats(B_DRAM_ADDR, N * K).copy()).reshape(N, K)
        output = a @ b.T
        if C_DRAM_ADDR is not None:
            output += torch.from_numpy(self.floats(C_DRAM_ADDR, N).copy())
        self.view(OUTPUT_DRAM_ADDR, M * N * 2)[:] = np.frombuffer(bf16_bytes(output), dtype=np.uint8)
        self.launches.append((M, K, N))


class Bf16ConvTests(unittest.TestCase):
    INPUT = 0x80000000
    OUTPUT = 0xB0000000
    SCRATCH = 0xB1000000

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1847)
        self.image = conv.shared._ImageBuilder(conv.shared.MODEL_BASE, conv.shared.MODEL_LIMIT)
        self.zero = self.image.allocate(torch.zeros(conv.ZERO_BYTES // 2, dtype=torch.bfloat16), alignment=128)

    def args(self, shape):
        return dict(input_shape=shape, input_address=self.INPUT,
                    output_address=self.OUTPUT, scratch_address=self.SCRATCH,
                    image=self.image)

    def execute(self, plan, values):
        engine = MemoryEngine()
        engine.allocate(self.INPUT, conv.packed_bytes(plan["input_shape"]), fill=0)
        engine.allocate(self.OUTPUT, conv.packed_bytes(plan["output_shape"]))
        engine.allocate(self.SCRATCH, plan["scratch_bytes"])
        engine.regions[self.image.base] = np.frombuffer(self.image.data, dtype=np.uint8).copy()
        cpad = conv.shared._align_up(values.shape[1], 64)
        packed = torch.zeros(values.shape[0], cpad)
        packed[:, :values.shape[1]] = values
        input_bytes = np.frombuffer(bf16_bytes(packed), dtype=np.uint8)
        engine.view(self.INPUT, input_bytes.size)[:] = input_bytes
        conv.emit_conv(engine, plan, zero_address=self.zero)
        np.testing.assert_array_equal(engine.view(self.INPUT, input_bytes.size), input_bytes)
        length, channels = plan["output_shape"]
        output = engine.floats(self.OUTPUT, conv.packed_bytes(plan["output_shape"]) // 2).reshape(length, -1)
        np.testing.assert_array_equal(output[:, channels:], 0)
        self.assertTrue(np.isfinite(output).all())
        self.assertTrue(all(1 <= row[0] <= 1024 for row in engine.launches))
        self.assertEqual(engine._isa_reg_counter, 1)
        self.assertEqual(engine.registers, {})
        self.assertEqual(engine._inst_ptr_counter, 1)
        self.assertEqual(engine.pointers, {})
        return torch.from_numpy(output[:, :channels].copy()), engine

    def test_regular_im2col_padding_stride_dilation_and_tail(self):
        for length, inputs, outputs, kernel, stride, dilation, padding, bias_enabled in (
                (1, 1, 48, 7, 1, 1, 3, True),
                (2, 65, 48, 3, 1, 1, (1, 2), False),
                (131, 48, 65, 3, 1, 1, 1, True),
                (131, 65, 96, 4, 2, 1, 1, True),
                (17, 96, 65, 10, 5, 1, 3, False),
                (10, 768, 65, 7, 1, 9, 27, True),
                (19, 65, 48, 3, 2, 3, (2, 4), True)):
            with self.subTest(length=length, inputs=inputs, stride=stride, dilation=dilation):
                values = torch.randn(length, inputs).bfloat16().float()
                weight = (torch.randn(outputs, inputs, kernel) * .03).bfloat16().float()
                bias = (torch.randn(outputs) * .01).bfloat16().float() if bias_enabled else None
                plan = conv.prepare_conv1d("conv", weight, bias, **self.args(values.shape),
                                           stride=stride, dilation=dilation, padding=padding)
                expected_scratch = conv.conv_scratch_bytes(values.shape, weight.shape,
                                                           stride=stride, dilation=dilation, padding=padding)
                self.assertEqual(plan["scratch_bytes"], expected_scratch)
                actual, engine = self.execute(plan, values)
                padded = F.pad(values.T[None], conv._pads(padding))
                expected = F.conv1d(padded, weight, bias, stride=stride, dilation=dilation)[0].T.bfloat16().float()
                torch.testing.assert_close(actual, expected, rtol=.008, atol=1e-6)
                if length == 131 and stride == 1:
                    self.assertEqual([item[0] for item in engine.launches], [131])

    def test_patch_dma_is_byte_exact_with_compact_native_capture(self):
        generator = np.random.default_rng(9741)
        old = conv.udc.UE_AXI_DATA_WIDTH_BITS
        conv.udc.UE_AXI_DATA_WIDTH_BITS = 256
        try:
            for take, kernel, width, stride, dilation in (
                    (64, 1, 64, 1, 1), (64, 7, 64, 1, 1),
                    (64, 7, 768, 1, 9), (64, 2, 1536, 1, 1),
                    (64, 10, 128, 5, 1), (17, 3, 128, 2, 3),
                    (2, 7, 768, 1, 9), (1, 10, 128, 5, 1)):
                with self.subTest(take=take, kernel=kernel, width=width, stride=stride, dilation=dilation):
                    row_bytes, first = width * 2, 3
                    rows = (first + take - 1) * stride + (kernel - 1) * dilation + 1
                    source = generator.integers(0, 256, size=(rows, row_bytes), dtype=np.uint8)
                    plan = {"padded_input_address": self.SCRATCH, "patch_address": self.SCRATCH + 0x1000000}
                    phase = {"kernel": kernel, "stride": stride, "dilation": dilation}
                    engine = MemoryEngine()
                    engine.regions[self.SCRATCH] = source.reshape(-1).copy()
                    engine.allocate(plan["patch_address"], take * kernel * row_bytes)
                    conv._stage_patches(engine, plan, phase, first=first, take=take, row_bytes=row_bytes)
                    indices = ((first + np.arange(take))[:, None] * stride
                               + np.arange(kernel)[None, :] * dilation)
                    expected = source[indices].reshape(-1)
                    np.testing.assert_array_equal(engine.view(plan["patch_address"], expected.size), expected)
                    self.assertEqual(len(engine.transfers), 2 * min(take, kernel))
                    captured = conv.shared._WholeGraphEngine(0xA0000000)
                    captured.start_capture()
                    conv._stage_patches(captured, plan, phase, first=first, take=take, row_bytes=row_bytes)
                    self.assertEqual(captured.capture_count, 2 * min(take, kernel))
        finally:
            conv.udc.UE_AXI_DATA_WIDTH_BITS = old

    def test_dynamic_deep_convolution_preserves_arithmetic_and_releases_registers(self):
        values = torch.randn(67, 768).bfloat16().float()
        weight = (torch.randn(65, 768, 7) * .01).bfloat16().float()
        bias = (torch.randn(65) * .01).bfloat16().float()
        plan = conv.prepare_conv1d("dynamic", weight, bias,
                                   **self.args(values.shape), padding=27, dilation=9)
        plan["use_sram"] = False
        actual, engine = self.execute(plan, values)
        expected = F.conv1d(values.T[None], weight, bias, padding=27, dilation=9)[0].T.bfloat16().float()
        torch.testing.assert_close(actual, expected, rtol=.008, atol=1e-6)
        self.assertEqual(engine.dynamic_launches, 1)
        self.assertEqual([row[0] for row in engine.launches], [64, 3])

    def test_sram_windows_cross_tile_limits_and_skip_patch_dram(self):
        for length, inputs, outputs, kernel, stride, dilation, padding in (
                (2051, 48, 65, 7, 1, 1, 3),
                (635, 768, 65, 7, 1, 1, 3),
                (151, 65, 48, 7, 6, 9, 27),
                (95, 768, 65, 7, 1, 9, 27)):
            with self.subTest(length=length, inputs=inputs, dilation=dilation):
                values = torch.randn(length, inputs).bfloat16().float()
                weight = (torch.randn(outputs, inputs, kernel) * .01).bfloat16().float()
                bias = (torch.randn(outputs) * .01).bfloat16().float()
                plan = conv.prepare_conv1d("sram", weight, bias, **self.args(values.shape),
                                           stride=stride, dilation=dilation, padding=padding)
                actual, engine = self.execute(plan, values)
                expected = F.conv1d(values.T[None], weight, bias, stride=stride,
                                    dilation=dilation, padding=padding)[0].T.bfloat16().float()
                torch.testing.assert_close(actual, expected, rtol=.008, atol=1e-6)
                self.assertFalse(any(kind == "write" and plan["patch_address"] <= destination
                                     < plan["scratch_address"] + plan["scratch_bytes"]
                                     for kind, _, destination, _ in engine.transfers))
                if length > 600:
                    self.assertGreater(max(item[0] for item in engine.launches), 64)

    def test_sram_capture_reuses_weights_and_bounds_all_jump_targets(self):
        old = conv.udc.UE_AXI_DATA_WIDTH_BITS
        conv.udc.UE_AXI_DATA_WIDTH_BITS = 256
        try:
            for shape, weight_shape, options in (
                    ((2051, 48), (48, 48, 7), dict(padding=3)),
                    ((635, 768), (768, 768, 7), dict(padding=3)),
                    ((635, 768), (768, 768, 7), dict(padding=27, dilation=9))):
                with self.subTest(shape=shape, options=options):
                    plan = conv.prepare_conv1d("capture_sram", torch.zeros(weight_shape),
                                               **self.args(shape), **options)
                    counts = {}
                    for fast in (False, True):
                        plan["use_sram"] = fast
                        engine = conv.shared._WholeGraphEngine(self.image.align(128))
                        engine.start_capture()
                        with contextlib.redirect_stdout(io.StringIO()):
                            conv.emit_conv(engine, plan, zero_address=self.zero)
                        counts[fast] = engine.capture_count
                        self.assertEqual(conv.udc.check_isa_jumps(engine.capture_buffer, engine._program_dram_base), [])
                        self.assertEqual(engine._isa_reg_counter, 1)
                        self.assertEqual(engine._inst_ptr_counter, 1)
                    if options.get("dilation", 1) == 1:
                        self.assertLess(counts[True], counts[False])
                    else:
                        # Dilation residues trade a few more static loop
                        # headers for fewer weight loads and no patch DRAM.
                        self.assertLess(counts[True], 4096)
        finally:
            conv.udc.UE_AXI_DATA_WIDTH_BITS = old

    def test_transpose_polyphases_include_padding_and_last_samples(self):
        for length, inputs, outputs, stride, padding, extra in (
                (1, 48, 65, 2, 1, 0), (2, 65, 48, 5, 3, 1),
                (67, 65, 48, 2, 1, 0), (67, 48, 65, 5, 3, 1),
                (7, 65, 48, 5, 0, 4), (2, 48, 65, 1, 1, 0)):
            with self.subTest(length=length, stride=stride, padding=padding, extra=extra):
                values = torch.randn(length, inputs).bfloat16().float()
                weight = (torch.randn(inputs, outputs, 2 * stride) * .03).bfloat16().float()
                bias = (torch.randn(outputs) * .01).bfloat16().float()
                plan = conv.prepare_conv_transpose1d("transpose", weight, bias, **self.args(values.shape),
                                                     stride=stride, padding=padding, output_padding=extra)
                self.assertEqual(plan["scratch_bytes"], conv.conv_scratch_bytes(
                    values.shape, weight.shape, transpose=True,
                    stride=stride, padding=padding, output_padding=extra))
                actual, _ = self.execute(plan, values)
                expected = F.conv_transpose1d(values.T[None], weight, bias, stride=stride,
                                              padding=padding, output_padding=extra)[0].T.bfloat16().float()
                torch.testing.assert_close(actual, expected, rtol=.008, atol=1e-6)

    def test_weight_layout_is_bf16_and_independent_of_audio_length(self):
        weight = torch.randn(65, 48, 7)
        before = len(self.image.data)
        first = conv.prepare_conv1d("short", weight, **self.args((3, 48)), padding=3)
        first_bytes = len(self.image.data) - before
        before = len(self.image.data)
        conv.prepare_conv1d("long", weight, **self.args((1001, 48)), padding=3)
        self.assertEqual(len(self.image.data) - before, first_bytes)
        self.assertEqual(first_bytes, 128 * 64 * 7 * 2)
        phase = first["phases"][0]
        offset = phase["weight_address"] - self.image.base
        actual = torch.frombuffer(bytearray(self.image.data[offset:offset + phase["weight_bytes"]]),
                                  dtype=torch.bfloat16).reshape(128, 7, 64)
        torch.testing.assert_close(actual[:65, :, :48], weight.bfloat16().permute(0, 2, 1), rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(actual[65:]).item(), 0)
        self.assertEqual(torch.count_nonzero(actual[:, :, 48:]).item(), 0)

    def test_transpose_weights_partition_kernel_without_repetition(self):
        weight = torch.randn(64, 128, 10)
        before = len(self.image.data)
        plan = conv.prepare_conv_transpose1d("transpose", weight, **self.args((3, 64)),
                                             stride=5, padding=3, output_padding=1)
        self.assertEqual(len(self.image.data) - before, weight.numel() * 2)
        self.assertEqual(sum(phase["weight_bytes"] for phase in plan["phases"]), weight.numel() * 2)

    def test_offline_capture_deep_k_and_transpose_without_host_io(self):
        old = conv.udc.UE_AXI_DATA_WIDTH_BITS
        conv.udc.UE_AXI_DATA_WIDTH_BITS = 256
        try:
            cases = [(False, (10, 768), (768, 768, 7), dict(padding=27, dilation=9)),
                     (False, (3, 1024), (1024, 1024, 7), dict(padding=3)),
                     (True, (2, 1536), (1536, 768, 10), dict(stride=5, padding=3, output_padding=1))]
            for transpose, shape, weight_shape, kwargs in cases:
                with self.subTest(shape=shape, transpose=transpose):
                    prepare = conv.prepare_conv_transpose1d if transpose else conv.prepare_conv1d
                    plan = prepare("capture", torch.zeros(weight_shape), torch.zeros(weight_shape[1 if transpose else 0]),
                                   **self.args(shape), **kwargs)
                    engine = conv.shared._WholeGraphEngine(self.image.align(128))
                    engine.start_capture()
                    with contextlib.redirect_stdout(io.StringIO()):
                        conv.emit_conv(engine, plan, zero_address=self.zero)
                    kinds = [(inst.words[0] >> 8) & 15 for inst in engine.capture_buffer]
                    self.assertGreater(engine.capture_count, 30)
                    self.assertNotIn(conv.udc.INSTRUCTION_HALT, kinds)
                    self.assertNotIn(conv.udc.INSTRUCTION_SWI, kinds)
                    self.assertEqual(conv.udc.check_isa_jumps(engine.capture_buffer, engine._program_dram_base), [])
                    self.assertEqual(engine._isa_reg_counter, 1)
        finally:
            conv.udc.UE_AXI_DATA_WIDTH_BITS = old

    def test_dynamic_capture_reduces_deep_matrices_and_retains_cheap_static_calls(self):
        old = conv.udc.UE_AXI_DATA_WIDTH_BITS
        conv.udc.UE_AXI_DATA_WIDTH_BITS = 256
        addresses = dict(A_DRAM_ADDR=self.SCRATCH, B_DRAM_ADDR=0x90000000,
                         OUTPUT_DRAM_ADDR=self.OUTPUT, C_DRAM_ADDR=0x92000000)
        try:
            for rows, kernel, outputs, expect_dynamic in (
                    (1, 448, 64, False), (64, 448, 64, False),
                    (2, 3072, 768, False), (64, 1344, 192, False),
                    (64, 5376, 768, True), (64, 5376, 1536, True),
                    (64, 7168, 768, True), (64, 7168, 1536, True),
                    (3, 5376, 768, True), (64, 3072, 1536, True)):
                with self.subTest(M=rows, K=kernel, N=outputs):
                    baseline = conv.shared._WholeGraphEngine(0xA0000000)
                    baseline.start_capture()
                    captured = conv.shared._WholeGraphEngine(0xA0000000)
                    captured.start_capture()
                    with contextlib.redirect_stdout(io.StringIO()):
                        baseline.matmat_mul_core(M=rows, K=kernel, N=outputs, **addresses)
                        conv._matmul(captured, M=rows, K=kernel, N=outputs, **addresses)
                    if expect_dynamic:
                        self.assertLess(captured.capture_count, baseline.capture_count)
                        self.assertLessEqual(captured.capture_count, 120)
                    else:
                        self.assertEqual(captured.capture_count, baseline.capture_count)
                    self.assertEqual(conv.udc.check_isa_jumps(captured.capture_buffer, 0xA0000000), [])
                    self.assertEqual(captured._isa_reg_counter, 1)
                    self.assertEqual(captured._inst_ptr_counter, 1)
        finally:
            conv.udc.UE_AXI_DATA_WIDTH_BITS = old

    def test_dynamic_matmul_releases_caller_register_on_failure(self):
        class FailingEngine(MemoryEngine):
            def matmat_mul_core(self, **kwargs):
                raise RuntimeError("capture rejected")

        engine = FailingEngine()
        with self.assertRaisesRegex(RuntimeError, "capture rejected"):
            conv._matmul(engine, M=64, K=5376, N=768)
        self.assertEqual(engine._isa_reg_counter, 1)
        self.assertEqual(engine.registers, {})

    def test_rejects_unsafe_or_invalid_plans(self):
        args = self.args((3, 48))
        weight = torch.ones(65, 48, 3)
        for extra in (dict(chunk_rows=65), dict(chunk_rows=0),
                      dict(output_address=self.INPUT), dict(scratch_address=self.OUTPUT),
                      dict(scratch_address=self.SCRATCH + 1), dict(dilation=0)):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                conv.prepare_conv1d("bad", weight, **(args | extra), padding=1)
        with self.assertRaises(ValueError):
            conv.prepare_conv1d("bad", weight * float("nan"), **args, padding=1)
        with self.assertRaises(ValueError):
            conv.prepare_conv1d("bad", weight, torch.ones(64), **args, padding=1)


if __name__ == "__main__":
    unittest.main()
