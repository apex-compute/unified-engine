"""Convolution algebra, packed stream interpretation and offline ISA capture."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bigcodec_conv as conv


def bf16_bytes(value):
    return torch.as_tensor(value).bfloat16().contiguous().view(torch.uint8).numpy().tobytes()


class MemoryEngine:
    """Interpret actual DMA/IF8 streams against separate byte-addressed memory.

    Arithmetic uses an FP32 reference dot and rounds its output to BF16; this
    verifies placement, padding, taps and weight encoding, not RTL numerics.
    """

    def __init__(self):
        self.regions = {}
        self.sram = np.full(0x100000, 0xCD, dtype=np.uint8)
        self.launches = []
        self.transfers = []

    def allocate(self, address, size, fill=0xCD):
        self.regions[address] = np.full(size, fill, dtype=np.uint8)

    def view(self, address, size):
        for base, value in self.regions.items():
            if base <= address and address + size <= base + value.size:
                return value[address - base:address - base + size]
        raise AssertionError(f"unallocated DRAM {address:#x}+{size}")

    def floats(self, address, size, *, sram=False):
        value = self.sram[address:address + size * 2] if sram else self.view(address, size * 2)
        return torch.frombuffer(bytearray(value), dtype=torch.bfloat16).float().numpy()

    def _dma(self, source, destination, size, chunk, jump, *, read):
        sr = destination if read else source
        if sr % 128:
            raise AssertionError(f"unaligned SRAM {sr:#x}")
        assert sr % 0x80000 + size <= conv.udc.URAM_NEAR_FULL_SIZE
        if not chunk:
            chunk = size
            jump = size
        assert size % chunk == 0
        for i in range(size // chunk):
            if read:
                self.sram[destination + i * chunk:destination + (i + 1) * chunk] = self.view(source + i * jump, chunk)
            else:
                self.view(destination + i * jump, chunk)[:] = self.sram[source + i * chunk:source + (i + 1) * chunk]
        self.transfers.append(("read" if read else "write", source, destination, size))

    def accelerator_memory_to_sram(self, source, destination, elements,
                                   *, memcpy_length_bytes=None,
                                   stride_bytes_per_chunk=None, stride_jump_bytes=None):
        self._dma(source, destination, elements * 2 if memcpy_length_bytes is None else memcpy_length_bytes,
                  stride_bytes_per_chunk, stride_jump_bytes, read=True)

    def sram_to_accelerator_memory(self, source, destination, elements,
                                   *, memcpy_length_bytes=None,
                                   stride_bytes_per_chunk=None, stride_jump_bytes=None):
        self._dma(source, destination, elements * 2 if memcpy_length_bytes is None else memcpy_length_bytes,
                  stride_bytes_per_chunk, stride_jump_bytes, read=False)

    def accelerator_memory_to_scale_sram(self, address, elements):
        self.scales = self.floats(address, elements)

    def accelerator_memory_to_bias_sram(self, address, elements):
        self.bias = self.floats(address, elements)

    def start_queue_for_conv2d_operation(self, **kw):
        self.launches.append(kw)
        assert kw["kernel_h"] == kw["out_h"] == 1
        assert kw["data_type"] == conv.udc.TYPE.IF8
        count, channels, kernel = kw["oc_count"], kw["c_in"], kw["kernel_w"]
        cpad = kw["ct"] * 64
        blocks = (kernel * channels + 63) // 64 if kw["gather"] else kernel * kw["ct"]
        code_bytes = kw["out_w"] * count * blocks * 64
        weights = self.view(kw["weights_dram_addr"], code_bytes).view(np.int8).reshape(kw["out_w"], count, blocks * 64)
        scales = np.repeat(np.abs(self.scales.reshape(count, blocks)), 64, axis=1)
        assert np.all(self.scales < 0)
        inputs = self.floats(kw["act_sram_start_addr"], kw["w_pad"] * cpad, sram=True).reshape(kw["w_pad"], cpad)
        outputs = []
        for t in range(kw["out_w"]):
            patch = inputs[t * kw["stride_s"] + np.arange(kernel) * kw["dilation"]]
            values = np.zeros(blocks * 64, dtype=np.float32)
            flattened = patch[:, :channels].reshape(-1) if kw["gather"] else patch.reshape(-1)
            values[:flattened.size] = flattened
            result = (weights[t].astype(np.float32) * scales) @ values
            if kw["bias_enable"]:
                result += self.bias
            outputs.append(result)
        data = np.frombuffer(bf16_bytes(np.stack(outputs)), dtype=np.uint8)
        address = kw["output_sram_wb_addr"]
        self.sram[address:address + data.size] = data


class ConvTests(unittest.TestCase):
    INPUT = 0x80000000
    OUTPUT = 0xB0000000
    PAD = 0xB1000000
    SCRATCH = 0xB2000000

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(481)
        self.image = conv.shared._ImageBuilder(conv.shared.MODEL_BASE, conv.shared.MODEL_LIMIT)
        self.zero = self.image.allocate(torch.zeros(conv.ZERO_BYTES // 2, dtype=torch.bfloat16), alignment=128)

    def args(self, shape):
        return dict(input_shape=shape, input_address=self.INPUT,
                    output_address=self.OUTPUT, padded_input_address=self.PAD,
                    image=self.image)

    def execute(self, plan, values):
        engine = MemoryEngine()
        engine.allocate(self.INPUT, conv.packed_bytes(plan["input_shape"]), fill=0)
        engine.allocate(self.OUTPUT, conv.packed_bytes(plan["output_shape"]))
        if plan["padding_bytes"]:
            engine.allocate(self.PAD, plan["padding_bytes"])
        engine.allocate(self.SCRATCH, max(128, plan["scatter_scratch_bytes"]))
        engine.regions[self.image.base] = np.frombuffer(self.image.data, dtype=np.uint8).copy()
        cpad = (values.shape[1] + 63) // 64 * 64
        packed = torch.zeros(values.shape[0], cpad)
        packed[:, :values.shape[1]] = values
        engine.view(self.INPUT, packed.numel() * 2)[:] = np.frombuffer(bf16_bytes(packed), dtype=np.uint8)
        emitter = conv.emit_conv1d if plan["kind"] == "conv1d" else conv.emit_conv_transpose1d
        emitter(engine, plan, zero_address=self.zero, scratch_address=self.SCRATCH)
        out_t, out_c = plan["output_shape"]
        physical = engine.floats(self.OUTPUT, conv.packed_bytes(plan["output_shape"]) // 2).reshape(out_t, -1)
        np.testing.assert_array_equal(physical[:, out_c:], 0)
        self.assertTrue(np.isfinite(physical).all())
        return torch.from_numpy(physical[:, :out_c].copy()), engine

    @staticmethod
    def dequantize(weight):
        encoded = conv.quantize_conv1d_if8(weight)
        oc, channels, _, kernel = encoded["codes_shape"]
        codes = encoded["codes_packed"].view(torch.int8).reshape(oc, channels, kernel).float()
        scale = encoded["block_scales"].float().abs()
        if encoded["layout"] == "gather":
            expanded = scale.repeat_interleave(64, dim=1)[:, :kernel * channels]
            expanded = expanded.reshape(oc, kernel, channels)
        else:
            expanded = scale.reshape(oc, kernel, -1).repeat_interleave(64, dim=2)[:, :, :channels]
        return codes * expanded.permute(0, 2, 1)

    def test_transpose_polyphase_matches_pytorch_including_boundaries(self):
        for stride in (2, 5):
            for length in (1, 2, 7):
                for padding in (0, (stride + 1) // 2):
                    for extra in (0, stride - 1):
                        with self.subTest(stride=stride, length=length, padding=padding, extra=extra):
                            x, weight, bias = torch.randn(1, 2, length, dtype=torch.float64), torch.randn(2, 3, 2 * stride, dtype=torch.float64), torch.randn(3, dtype=torch.float64)
                            expected = F.conv_transpose1d(x, weight, bias, stride=stride, padding=padding, output_padding=extra)
                            actual = torch.empty_like(expected)
                            phases = conv.transpose_polyphase(weight, input_length=length, stride=stride, padding=padding, output_padding=extra)
                            for phase in phases:
                                indices = torch.arange(phase["input_start"], phase["input_start"] + phase["input_length"])
                                source = torch.zeros(1, 2, len(indices), dtype=x.dtype)
                                valid = (indices >= 0) & (indices < length)
                                source[:, :, valid] = x[:, :, indices[valid]]
                                actual[:, :, phase["phase"]::stride] = F.conv1d(source, phase["weight"], bias)
                            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_emitted_conv_layouts_match_independent_pytorch(self):
        for channels, outputs, kernel, stride, dilation, padding in (
                (1, 48, 7, 1, 1, 3), (48, 96, 4, 2, 1, 1),
                (96, 192, 10, 5, 1, 3), (64, 64, 7, 1, 9, 27)):
            with self.subTest(channels=channels, outputs=outputs, dilation=dilation):
                values = torch.randn(17, channels).bfloat16().float()
                weight, bias = torch.randn(outputs, channels, kernel) * .025, torch.randn(outputs) * .01
                plan = conv.prepare_conv1d("test", weight, bias, **self.args(values.shape), stride=stride, dilation=dilation, padding=padding)
                actual, engine = self.execute(plan, values)
                expected = F.conv1d(values.T[None], self.dequantize(weight), bias.bfloat16().float(), stride=stride, dilation=dilation, padding=padding)[0].T.bfloat16().float()
                torch.testing.assert_close(actual, expected, rtol=.008, atol=1e-5)
                self.assertTrue(engine.launches)
                self.assertTrue(all(p["groups"][0][3] == 1 for p in plan["subplans"]))

    def test_emitted_transpose_phases_bias_padding_and_scatter(self):
        for stride in (2, 5):
            values = torch.randn(7, 96).bfloat16().float()
            weight, bias = torch.randn(96, 48, 2 * stride) * .025, torch.randn(48) * .01
            padding, extra = (stride + 1) // 2, stride % 2
            plan = conv.prepare_conv_transpose1d("transpose", weight, bias, **self.args(values.shape), stride=stride, padding=padding, output_padding=extra, weight_reuse_pixels=3)
            actual, engine = self.execute(plan, values)
            decoded = torch.empty_like(weight)
            for phase in conv.transpose_polyphase(weight, input_length=7, stride=stride, padding=padding, output_padding=extra):
                indices = list(range((phase["phase"] + padding) % stride, 2 * stride, stride))[::-1]
                decoded[:, :, indices] = self.dequantize(phase["weight"]).permute(1, 0, 2)
            expected = F.conv_transpose1d(values.T[None], decoded, bias.bfloat16().float(), stride=stride, padding=padding, output_padding=extra)[0].T.bfloat16().float()
            torch.testing.assert_close(actual, expected, rtol=.008, atol=1e-5)
            self.assertEqual(actual.shape[0], 7 * stride)
            self.assertTrue(all(launch["kernel_w"] == 2 for launch in engine.launches))

    def test_deep_channels_use_aligned_subplans_without_weight_repetition(self):
        weight = torch.randn(768, 768, 7) * .01
        plan = conv.prepare_conv1d("deep", weight, **self.args((3, 768)), padding=3)
        self.assertEqual(len(plan["subplans"]), 12)
        for subplan in plan["subplans"]:
            self.assertEqual(subplan["oc_chunk"], 64)
            self.assertTrue(all(tile[2:4] == (1, 1) for tile in subplan["tiles"]))
        # One int8 copy of each weight plus scale bytes/alignment, never the
        # activation length times those weights.
        self.assertLess(len(self.image.data), conv.ZERO_BYTES + weight.numel() * 1.04)

    def test_official_c768_dilation9_uses_exact_phases_and_shared_weights(self):
        for length, outputs, stride, reuse in ((10, 768, 1, 1), (30, 64, 6, 3)):
            with self.subTest(length=length, stride=stride, reuse=reuse):
                values = torch.randn(length, 768).bfloat16().float()
                weight, bias = torch.randn(outputs, 768, 7) * .01, torch.randn(outputs) * .01
                before = len(self.image.data)
                plan = conv.prepare_conv1d("official_d9", weight, bias,
                                           **self.args(values.shape), padding=27,
                                           dilation=9, stride=stride,
                                           weight_reuse_pixels=reuse)
                actual, engine = self.execute(plan, values)
                expected = F.conv1d(values.T[None], self.dequantize(weight),
                                    bias.bfloat16().float(), stride=stride,
                                    padding=27, dilation=9)[0].T.bfloat16().float()
                torch.testing.assert_close(actual, expected, rtol=.008, atol=1e-5)
                self.assertEqual(plan["padding_bytes"], conv.conv1d_padding_bytes(values.shape, padding=27))
                self.assertEqual(plan["output_step"], 9 if stride == 1 else 3)
                self.assertTrue(all(launch["dilation"] == 1 for launch in engine.launches))
                # Every phase references the same resident weight streams.
                addresses = [{chunk["weight_address"] for subplan in phase["subplans"]
                              for chunk in subplan["chunks"]} for phase in plan["dilation_phases"]]
                self.assertTrue(all(item == addresses[0] for item in addresses))
                self.assertLess(len(self.image.data) - before, weight.numel() * reuse * 1.04 + 8192)

                old = conv.udc.UE_AXI_DATA_WIDTH_BITS
                conv.udc.UE_AXI_DATA_WIDTH_BITS = 256
                try:
                    captured = conv.shared._WholeGraphEngine(self.image.align(128))
                    captured.start_capture()
                    conv.emit_conv1d(captured, plan, zero_address=self.zero,
                                     scratch_address=self.SCRATCH)
                    self.assertGreaterEqual(captured.capture_count, 30)
                finally:
                    conv.udc.UE_AXI_DATA_WIDTH_BITS = old

    def test_if8_quantization_matches_independent_library(self):
        import quant_lib
        weight = torch.randn(3, 97, 3)
        encoded = conv.quantize_conv1d_if8(weight)
        blocks = torch.zeros(3, 3, 128)
        blocks[:, :, :97] = weight.permute(0, 2, 1)
        code, scales = quant_lib.quantize_int8(blocks.reshape(3, -1))
        full = torch.frombuffer(bytearray(code), dtype=torch.uint8).reshape(3, 3, 128)
        expected = full[:, :, :97].permute(0, 2, 1).contiguous().flatten()
        torch.testing.assert_close(encoded["codes_packed"], expected, rtol=0, atol=0)
        expected_scales = torch.frombuffer(bytearray(scales), dtype=torch.bfloat16).reshape(3, -1)
        torch.testing.assert_close(encoded["block_scales"], -expected_scales.abs(), rtol=0, atol=0)

    def test_offline_capture_emits_conv_configs_without_live_io(self):
        old = conv.udc.UE_AXI_DATA_WIDTH_BITS
        conv.udc.UE_AXI_DATA_WIDTH_BITS = 256
        try:
            plan = conv.prepare_conv_transpose1d("capture", torch.randn(96, 48, 10),
                                                **self.args((3, 96)), stride=5, padding=3, output_padding=1)
            engine = conv.shared._WholeGraphEngine(self.image.align(128))
            engine.start_capture()
            conv.emit_conv_transpose1d(engine, plan, zero_address=self.zero, scratch_address=self.SCRATCH)
            # Operators append to the caller's capture, with no HALT/SWI or
            # nested capture reset between phases.
            kinds = [(inst.words[0] >> 8) & 15 for inst in engine.capture_buffer]
            self.assertIn(conv.udc.INSTRUCTION_CONFIG, kinds)
            self.assertNotIn(conv.udc.INSTRUCTION_HALT, kinds)
            self.assertNotIn(conv.udc.INSTRUCTION_SWI, kinds)
            self.assertGreater(engine.capture_count, 30)
        finally:
            conv.udc.UE_AXI_DATA_WIDTH_BITS = old

    def test_rejects_invalid_channels_nonfinite_and_workspace_aliasing(self):
        with self.assertRaisesRegex(ValueError, "channel mismatch"):
            conv.conv1d_output_shape((5, 3), (4, 2, 3))
        with self.assertRaisesRegex(ValueError, "output_padding"):
            conv.conv_transpose1d_output_shape((5, 3), (3, 4, 10), stride=5, output_padding=5)
        with self.assertRaisesRegex(ValueError, "finite"):
            conv.quantize_conv1d_if8(torch.full((2, 3, 1), float("nan")))
        args = self.args((5, 3))
        args["padded_input_address"] = self.OUTPUT
        with self.assertRaisesRegex(ValueError, "overlaps"):
            conv.prepare_conv1d("alias", torch.ones(4, 3, 3), **args, padding=1)


if __name__ == "__main__":
    unittest.main()
