"""Independent CPU tensor semantics and offline ISA checks for BF16 Conv2d.

The memory interpreter checks the public DMA/matmul ABI using FP32 reference
arithmetic and BF16 output rounding. It does not simulate RTL accumulation.
"""

import contextlib
from dataclasses import replace
import io
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dpdfnet_bf16_conv as conv
from dpdfnet_precompiled import make_layout, pack_tensor


def bf16_bytes(value):
    return torch.as_tensor(value).bfloat16().contiguous().view(torch.uint8).numpy().tobytes()


class ConstantEmitter:
    def __init__(self):
        self.next_address = 0x90000000
        self.constants = {}

    def allocate_constant(self, value):
        assert value.dtype == torch.bfloat16
        address = self.next_address
        self.constants[address] = value.detach().clone()
        self.next_address += (value.numel() * 2 + 127) // 128 * 128
        return address


class MemoryEngine:
    """Byte-addressed DMA and the independent A @ B.T matmul contract."""

    def __init__(self):
        self.regions = {}
        self.sram = np.full(0x100000, 0xCD, dtype=np.uint8)
        self.transfers = []
        self.launches = []

    def allocate(self, address, size):
        self.regions[address] = np.full(size, 0xCD, dtype=np.uint8)

    def view(self, address, size):
        for base, value in self.regions.items():
            if base <= address and address + size <= base + value.size:
                return value[address - base:address - base + size]
        raise AssertionError(f"unallocated DRAM {address:#x}+{size}")

    def floats(self, address, count):
        return torch.frombuffer(bytearray(self.view(address, count * 2)),
                                dtype=torch.bfloat16).float()

    def _dma(self, source, destination, size, chunk, jump, *, read):
        sram_address = destination if read else source
        assert sram_address % 128 == 0
        assert sram_address + size <= conv.udc.URAM_NEAR_FULL_SIZE
        chunk, jump = (size, size) if chunk is None else (chunk, jump)
        assert chunk > 0 and size % chunk == 0
        for index in range(size // chunk):
            if read:
                self.sram[destination + index * chunk:destination + (index + 1) * chunk] = self.view(
                    source + index * jump, chunk)
            else:
                self.view(destination + index * jump, chunk)[:] = self.sram[
                    source + index * chunk:source + (index + 1) * chunk]
        self.transfers.append((read, source, destination, size))

    def accelerator_memory_to_sram(self, source, destination, elements, *,
                                   memcpy_length_bytes=None,
                                   stride_bytes_per_chunk=None, stride_jump_bytes=None):
        self._dma(source, destination,
                  elements * 2 if memcpy_length_bytes is None else memcpy_length_bytes,
                  stride_bytes_per_chunk, stride_jump_bytes, read=True)

    def sram_to_accelerator_memory(self, source, destination, elements, *,
                                   memcpy_length_bytes=None,
                                   stride_bytes_per_chunk=None, stride_jump_bytes=None):
        self._dma(source, destination,
                  elements * 2 if memcpy_length_bytes is None else memcpy_length_bytes,
                  stride_bytes_per_chunk, stride_jump_bytes, read=False)

    def matmat_mul_core(self, *, M, K, N, A_DRAM_ADDR, B_DRAM_ADDR,
                        OUTPUT_DRAM_ADDR, C_DRAM_ADDR=None,
                        is_B_quantized=False, bias_mode="broadcast_N"):
        assert is_B_quantized is False
        assert bias_mode == "broadcast_N"
        a = self.floats(A_DRAM_ADDR, M * K).reshape(M, K)
        b = self.floats(B_DRAM_ADDR, N * K).reshape(N, K)
        result = a @ b.T
        if C_DRAM_ADDR is not None:
            result += self.floats(C_DRAM_ADDR, N)
        self.view(OUTPUT_DRAM_ADDR, M * N * 2)[:] = np.frombuffer(bf16_bytes(result), dtype=np.uint8)
        self.launches.append((M, K, N, A_DRAM_ADDR))


# Distinct dense geometries in the pinned official 8 and 16 kHz graphs.
# Store the geometry itself so these tests need neither ONNX nor a download.
PINNED_GEOMETRIES = (
    ((1, 3, 32), (64, 1, 3, 3), (0, 1, 0, 1)),
    ((1, 3, 96), (32, 1, 3, 3), (0, 1, 0, 1)),
    ((1, 3, 80), (32, 1, 3, 3), (0, 1, 0, 1)),
    ((1, 3, 80), (64, 1, 3, 3), (0, 1, 0, 1)),
    *(((10, 1, width), (10, 10, 1, 1), (0, 0, 0, 0)) for width in (80, 96)),
    *(((32, 5, width), (5, 32, 5, 1), (0, 0, 0, 0)) for width in (80, 96)),
    *(((64, 1, width), (64, 64, 1, 1), (0, 0, 0, 0))
      for width in (8, 10, 16, 20, 32, 40, 48, 80, 96)),
    *(((64, 1, width), (1, 64, 1, 3), (0, 1, 0, 1)) for width in (32, 80)),
)


class Bf16ConvTests(unittest.TestCase):
    INPUT = 0x80000000
    OUTPUT = 0xB0000000
    WORKSPACE = 0xB1000000

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3917)

    def prepare(self, shape, weight_shape, pads=(0, 0, 0, 0), *,
                stride=(1, 1), dilation=(1, 1), bias_enabled=True):
        channels, height, width = shape
        values = torch.randn(1, channels, height, width).bfloat16().float()
        weight = torch.randn(weight_shape) * .03
        bias = torch.randn(weight_shape[0]) * .02 if bias_enabled else None
        padded = F.pad(values, (pads[1], pads[3], pads[0], pads[2]))
        expected = F.conv2d(padded, weight.bfloat16().float(),
                            None if bias is None else bias.bfloat16().float(),
                            stride=stride, dilation=dilation).bfloat16().float()
        source = make_layout("input", (*padded.shape[2:], channels), self.INPUT)
        output = make_layout("output", (*expected.shape[2:], weight_shape[0]), self.OUTPUT)
        workspace = make_layout("workspace", conv.workspace_shape(source, output, weight_shape), self.WORKSPACE)
        attrs = {"pads": pads, "strides": stride, "dilations": dilation,
                 "kernel_shape": weight_shape[2:], "group": 1}
        emitter = ConstantEmitter()
        resource = conv.prepare_conv(emitter, source, output, attrs, weight, bias, workspace)
        return resource, emitter, source, output, workspace, padded, weight, bias, expected

    def execute(self, prepared):
        resource, emitter, source, output, workspace, padded, _, _, expected = prepared
        engine = MemoryEngine()
        for layout in (source, output, workspace):
            engine.allocate(layout.address, layout.size_bytes)
        for address, value in emitter.constants.items():
            engine.regions[address] = np.frombuffer(bf16_bytes(value), dtype=np.uint8).copy()
        source_bytes = np.frombuffer(bf16_bytes(pack_tensor(padded[0].permute(1, 2, 0), source)), dtype=np.uint8)
        engine.view(source.address, source.size_bytes)[:] = source_bytes
        conv.emit_conv(engine, resource)
        np.testing.assert_array_equal(engine.view(source.address, source.size_bytes), source_bytes)
        actual = engine.floats(output.address, output.physical_elements).reshape(*output.shape[:2], output.padded_last)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(torch.count_nonzero(actual[..., output.logical_last:]).item(), 0)
        torch.testing.assert_close(actual[..., :output.logical_last].permute(2, 0, 1)[None],
                                   expected, rtol=.008, atol=1e-5)
        self.assertEqual(engine.launches, [(resource["M"], resource["K"], resource["N"],
                                           source.address if resource["direct"] else workspace.address)])
        if resource["direct"]:
            self.assertEqual(engine.transfers, [])
            self.assertTrue(np.all(engine.view(workspace.address, workspace.size_bytes) == 0xCD))
        else:
            # Independent unfold expresses each logical channel's complete
            # spatial kernel, then changes only the explicit matrix ordering.
            kh, kw = resource["weight_shape"][2:]
            logical = F.unfold(padded, (kh, kw), dilation=resource["dilation"],
                               stride=resource["stride"]).reshape(source.logical_last, kh * kw, -1)
            patches = torch.zeros(resource["M"], kh * kw, source.padded_last, dtype=torch.bfloat16)
            patches[:, :, :source.logical_last] = logical.permute(2, 1, 0).bfloat16()
            np.testing.assert_array_equal(engine.view(workspace.address, workspace.size_bytes),
                                          np.frombuffer(bf16_bytes(patches), dtype=np.uint8))
        return engine

    def test_all_pinned_dense_geometries_with_and_without_bias(self):
        for shape, weight_shape, pads in PINNED_GEOMETRIES:
            for bias_enabled in (True, False):
                with self.subTest(shape=shape, weight=weight_shape, bias=bias_enabled):
                    self.execute(self.prepare(shape, weight_shape, pads, bias_enabled=bias_enabled))

    def test_asymmetric_stride_dilation_spatial_padding_and_partial_rows(self):
        for shape, weight_shape, pads, stride, dilation in (
                ((3, 7, 11), (5, 3, 2, 3), (1, 2, 3, 1), (2, 3), (2, 1)),
                ((65, 3, 7), (67, 65, 2, 2), (1, 0, 0, 1), (1, 2), (1, 2)),
                ((3, 2, 33), (1, 3, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1)),
                ((1, 1, 1), (1, 1, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))):
            with self.subTest(shape=shape, stride=stride, dilation=dilation):
                self.execute(self.prepare(shape, weight_shape, pads, stride=stride, dilation=dilation))

    def test_original_weight_and_bias_bits_with_zero_padding_and_no_scales(self):
        prepared = self.prepare((3, 5, 7), (5, 3, 2, 3))
        resource, emitter, source, output, _, _, weight, bias, _ = prepared
        self.assertEqual(len(emitter.constants), 2)
        actual = emitter.constants[resource["weight_address"]].reshape(output.padded_last, 2, 3, source.padded_last)
        expected = torch.zeros_like(actual)
        expected[:5, :, :, :3] = weight.bfloat16().permute(0, 2, 3, 1)
        np.testing.assert_array_equal(actual.view(torch.uint16).numpy(), expected.view(torch.uint16).numpy())
        actual_bias = emitter.constants[resource["bias_address"]]
        expected_bias = torch.zeros_like(actual_bias)
        expected_bias[:5] = bias.bfloat16()
        np.testing.assert_array_equal(actual_bias.view(torch.uint16).numpy(), expected_bias.view(torch.uint16).numpy())
        self.assertEqual(sum(value.numel() * 2 for value in emitter.constants.values()),
                         resource["N"] * resource["K"] * 2 + resource["N"] * 2)
        small = self.prepare((3, 1, 1), (5, 3, 1, 1), bias_enabled=False)
        large = self.prepare((3, 3, 97), (5, 3, 1, 1), bias_enabled=False)
        self.assertEqual(small[0]["weight_size_bytes"], large[0]["weight_size_bytes"])
        for item in (small, large):
            self.assertIsNone(item[0]["bias_address"])
            self.assertEqual(len(item[1].constants), 1)

    def test_reject_invalid_resources_before_allocating_constants(self):
        prepared = self.prepare((3, 5, 7), (5, 3, 2, 3))
        _, _, source, output, workspace, _, weight, bias, _ = prepared
        attrs = {"strides": (1, 1), "dilations": (1, 1)}
        cases = [
            (replace(source, address=source.address + 1), output, attrs, weight, bias, workspace),
            (source, replace(output, address=source.address), attrs, weight, bias, workspace),
            (source, output, attrs, weight, bias, replace(workspace, address=output.address)),
            (source, output, attrs, weight, bias, make_layout("bad", (1, 64), workspace.address)),
            (source, output, attrs, weight, bias, replace(workspace, padded_last=workspace.padded_last + 64)),
            (replace(source, padded_last=32), output, attrs, weight, bias, workspace),
            (source, output, {"group": 3}, weight, bias, workspace),
            (source, output, {"auto_pad": b"SAME_UPPER"}, weight, bias, workspace),
            (source, output, {"strides": (0, 1)}, weight, bias, workspace),
            (source, output, {"strides": (1,)}, weight, bias, workspace),
            (source, output, {"dilations": (2, 1)}, weight, bias, workspace),
            (source, output, {"kernel_shape": (1, 1)}, weight, bias, workspace),
            (source, output, attrs, weight, torch.zeros(6), workspace),
        ]
        for nonfinite in (float("nan"), float("inf"), torch.finfo(torch.float32).max):
            bad_weight = weight.clone()
            bad_weight[0, 0, 0, 0] = nonfinite
            cases.append((source, output, attrs, bad_weight, bias, workspace))
            bad_bias = bias.clone()
            bad_bias[0] = nonfinite
            cases.append((source, output, attrs, weight, bad_bias, workspace))
        for index, args in enumerate(cases):
            with self.subTest(case=index):
                emitter = ConstantEmitter()
                with self.assertRaises(ValueError):
                    conv.prepare_conv(emitter, *args)
                self.assertEqual(emitter.constants, {})

    def test_workspace_rejects_channel_rank_and_sram_capacity_mismatches(self):
        for source_shape, output_shape, weight_shape in (
                ((3, 5), (1, 3, 5), (5, 3, 1, 1)),
                ((1, 5, 3), (1, 5, 5), (5, 4, 1, 1)),
                ((1, 5, 3), (1, 5, 5), (6, 3, 1, 1)),
                ((1, 5, 3), (1, 5, 5), (5, 3, 0, 1)),
                ((1, 5, 3), (1, 5, 5), (5, 3, 1)),
                ((1, 2100, 1), (1, 2100, 1), (1, 1, 1, 1)),
                ((1, 5, 1024), (1, 1, 256), (256, 1024, 1, 5)),
                ((1, 4096, 64), (1, 1, 64), (64, 64, 1, 4096))):
            with self.subTest(source=source_shape, weight=weight_shape):
                with self.assertRaises(ValueError):
                    conv.workspace_shape(make_layout("input", source_shape, self.INPUT),
                                         make_layout("output", output_shape, self.OUTPUT), weight_shape)

    def test_offline_capture_contains_bf16_dot_and_no_quantized_arithmetic(self):
        old = conv.udc.UE_AXI_DATA_WIDTH_BITS
        conv.udc.UE_AXI_DATA_WIDTH_BITS = 256
        try:
            cases = list(PINNED_GEOMETRIES)
            for shape, weight_shape, pads in cases:
                with self.subTest(shape=shape, weight=weight_shape):
                    resource = self.prepare(shape, weight_shape, pads)[0]
                    engine = conv.shared._WholeGraphEngine(0xA0000000)
                    engine.start_capture()
                    with contextlib.redirect_stdout(io.StringIO()):
                        conv.emit_conv(engine, resource)
                    kinds = [(int(inst.words[0]) >> 8) & 15 for inst in engine.capture_buffer]
                    modes = [conv.udc._inst_desc_bits(inst.words, 172, 175)
                             for inst, kind in zip(engine.capture_buffer, kinds)
                             if kind in (conv.udc.INSTRUCTION_UE_OP, conv.udc.INSTRUCTION_UE_PBI)]
                    self.assertIn(conv.udc.UE_MODE.BF16_DOT_PRODUCT, modes)
                    for forbidden in (conv.udc.UE_MODE.DOT_PRODUCT, conv.udc.UE_MODE.CONV2D,
                                      conv.udc.UE_MODE.QUANTIZE, conv.udc.UE_MODE.DEQUANTIZE):
                        self.assertNotIn(forbidden, modes)
                    self.assertNotIn(conv.udc.INSTRUCTION_HALT, kinds)
                    self.assertNotIn(conv.udc.INSTRUCTION_SWI, kinds)
                    self.assertEqual(conv.udc.check_isa_jumps(engine.capture_buffer, engine._program_dram_base), [])
                    self.assertEqual(engine._isa_reg_counter, 1)
        finally:
            conv.udc.UE_AXI_DATA_WIDTH_BITS = old


if __name__ == "__main__":
    unittest.main()
