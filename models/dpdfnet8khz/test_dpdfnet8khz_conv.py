"""Execute native 8 kHz convolution layout DMAs on independent memory maps."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "dpdfnet"))
from dpdfnet8khz_conv import ConvTransposeOptimizationMixin
from dpdfnet_precompiled import DeviceEmitter, make_layout
import user_dma_core as udc


class MemoryEngine:
    """Interpret emitted DMA addresses and one-hot matvec selected indices.

    Unique finite BF16 bit patterns identify individual source elements. This
    checks tensor permutations independently of compiler mapping helpers; it
    does not claim to emulate the hardware's floating-point accumulation.
    """

    def __init__(self):
        self.regions = {}
        self.sram = np.full(0x100000 // 2, 0x7FC0, dtype=np.uint16)
        self.matvecs = []
        self.dmas = []

    def view(self, address, size):
        assert address % 2 == 0 and size % 2 == 0
        for base, values in self.regions.items():
            if base <= address and address + size <= base + values.nbytes:
                offset = (address - base) // 2
                return values[offset:offset + size // 2]
        raise AssertionError(f"DMA outside allocation: {address:#x} + {size}")

    def sram_view(self, address, size):
        assert address % 128 == 0
        offset = address % 0x80000
        assert offset + size <= udc.URAM_NEAR_FULL_SIZE
        return self.sram[address // 2:(address + size) // 2]

    def accelerator_memory_to_sram(self, source, destination, elements,
                                   *, memcpy_length_bytes=None):
        size = elements * 2 if memcpy_length_bytes is None else memcpy_length_bytes
        self.sram_view(destination, size)[:] = self.view(source, size)
        self.dmas.append(("read", source, size))

    def sram_to_accelerator_memory(self, source, destination, elements,
                                   *, memcpy_length_bytes=None):
        size = elements * 2 if memcpy_length_bytes is None else memcpy_length_bytes
        self.view(destination, size)[:] = self.sram_view(source, size)
        self.dmas.append(("write", destination, size))

    def start_queue_for_bf16_matvec_operation(self, **kwargs):
        self.matvecs.append(kwargs)
        assert kwargs["K"] == 64
        vector_address = kwargs["vector_sram_start_addr"]
        matrix_address = kwargs["matrix_sram_start_addr"]
        output_address = kwargs["output_sram_wb_addr"]
        assert vector_address < 0x80000 <= matrix_address
        vector = self.sram_view(vector_address, 128)
        selected = np.flatnonzero(vector)
        assert len(selected) == 1 and vector[selected[0]] == 0x3F80
        values = []
        for row in range(kwargs["N"]):
            address = matrix_address + row * kwargs["stride_z"] * 2
            values.append(self.sram_view(address, 128)[selected[0]])
        self.sram_view(output_address, kwargs["N"] * 2)[:] = values


class Baseline:
    def _stage_conv_input(self, index, node):
        self.fallbacks.append("stage")

    def _unstage_conv_output(self, index, node):
        self.fallbacks.append("unstage")


class Compiler(ConvTransposeOptimizationMixin, Baseline):
    def layout(self, name):
        return self.emitter.layouts[name]


class ConvLayoutTest(unittest.TestCase):
    def setUp(self):
        self.engine = MemoryEngine()
        self.compiler = Compiler()
        self.compiler.fallbacks = []
        emitter = DeviceEmitter.__new__(DeviceEmitter)
        emitter.engine = self.engine
        emitter.layouts = {}
        emitter._zero_padding = set()
        emitter.zero_address = 0x1000000
        self.engine.regions[emitter.zero_address] = np.zeros(
            udc.URAM_NEAR_FULL_SIZE // 2, dtype=np.uint16)
        self.compiler.emitter = emitter
        self.compiler.identity_address = 0x2000000
        self.engine.regions[self.compiler.identity_address] = (
            np.eye(64, dtype=np.uint16) * 0x3F80).reshape(-1)
        self.cursor = 0x3000000
        self.node = SimpleNamespace(input=["source"], output=["output"])

    def layout(self, name, shape):
        layout = make_layout(name, shape, self.cursor)
        self.cursor += 0x1000000
        self.compiler.emitter.layouts[name] = layout
        self.engine.regions[layout.address] = np.full(
            layout.physical_elements, 0x7FC0, dtype=np.uint16)
        return layout

    def seed(self, layout):
        values = (0x1000 + np.arange(layout.logical_elements)).astype(np.uint16)
        physical = self.engine.regions[layout.address].reshape(
            layout.rows, layout.padded_last)
        # Nonzero finite padding exposes accidental copying of padded columns.
        physical[:] = 0x4300
        physical[:, :layout.logical_last] = values.reshape(
            layout.rows, layout.logical_last)
        return values.reshape(layout.shape)

    def assert_tensor(self, layout, expected):
        physical = self.engine.regions[layout.address].reshape(
            layout.rows, layout.padded_last)
        np.testing.assert_array_equal(
            physical[:, :layout.logical_last].reshape(layout.shape), expected)
        np.testing.assert_array_equal(physical[:, layout.logical_last:], 0)

    def stage(self, channels, height, width, pads):
        source = self.layout("source", (1, channels, height, width))
        packed = self.layout("packed", (
            height + pads[0] + pads[2], width + pads[1] + pads[3], channels))
        self.compiler.conv_aux = {0: {"input": packed, "attrs": {"pads": pads}}}
        expected = np.pad(self.seed(source)[0].transpose(1, 2, 0),
                          ((pads[0], pads[2]), (pads[1], pads[3]), (0, 0)))
        self.compiler._stage_conv_input(0, self.node)
        self.assert_tensor(packed, expected)
        self.assertEqual(self.compiler.fallbacks, [])
        self.assertEqual(len(self.engine.matvecs), height * width)
        self.assertTrue(all(call["N"] == channels for call in self.engine.matvecs))
        self.assertEqual(sum(address == source.address for kind, address, size
                             in self.engine.dmas if kind == "read"), 1)

    def test_all_native_input_shapes_and_padding(self):
        # Covers every distinct input layout across the 29 native 8 kHz Convs.
        cases = [(1, 3, 80, (0, 1, 0, 1)),
                 (32, 5, 80, (0, 0, 0, 0)),
                 (10, 1, 80, (0, 0, 0, 0))]
        cases += [(64, 1, width, pads)
                  for width in (10, 20, 40, 80)
                  for pads in ((0, 0, 0, 0), (0, 1, 0, 1))]
        for case in cases:
            with self.subTest(case=case):
                self.setUp()
                self.stage(*case)

    def test_input_spatial_padding_and_three_column_blocks(self):
        self.stage(5, 3, 129, (1, 2, 2, 3))

    def test_all_native_output_shapes_and_padding(self):
        for channels, width in ([(64, width) for width in (10, 20, 40, 80)]
                                + [(channels, 80) for channels in (1, 5, 10, 32)]
                                + [(5, 64), (10, 129)]):
            with self.subTest(channels=channels, width=width):
                self.setUp()
                packed = self.layout("packed", (1, width, channels))
                output = self.layout("output", (1, channels, 1, width))
                self.compiler.conv_aux = {0: {"output": packed}}
                expected = self.seed(packed).transpose(2, 0, 1)[None]
                self.compiler._unstage_conv_output(0, self.node)
                self.assert_tensor(output, expected)
                self.assertEqual(self.compiler.fallbacks, [])
                self.assertEqual(len(self.engine.matvecs), channels)
                self.assertTrue(all(call["N"] == width for call in self.engine.matvecs))
                self.assertIn(output.name, self.compiler.emitter._zero_padding)

    def test_multiline_output_and_large_buffers_retain_baseline(self):
        packed = self.layout("packed", (3, 80, 5))
        output = self.layout("output", (1, 5, 3, 80))
        self.compiler.conv_aux = {0: {"output": packed}}
        self.compiler._unstage_conv_output(0, self.node)
        self.assertEqual(self.compiler.fallbacks, ["unstage"])
        self.assertEqual(self.engine.dmas, [])

        source = self.layout("source", (1, 64, 1, 80))
        self.compiler.conv_aux = {0: {
            "input": self.layout("input", (1, 80, 64)),
            "attrs": {"pads": (0, 0, 0, 0)}}}
        with patch.object(udc, "URAM_NEAR_FULL_SIZE", 1024):
            self.compiler._stage_conv_input(0, self.node)
        self.assertEqual(self.compiler.fallbacks, ["unstage", "stage"])
        self.assertEqual(self.engine.dmas, [])

    def test_channel_mismatch_is_rejected_before_emission(self):
        self.layout("source", (1, 32, 1, 80))
        self.compiler.conv_aux = {0: {
            "input": self.layout("packed", (1, 80, 64)), "attrs": {}}}
        with self.assertRaisesRegex(RuntimeError, "channel mismatch"):
            self.compiler._stage_conv_input(0, self.node)
        self.assertEqual(self.engine.dmas, [])


if __name__ == "__main__":
    unittest.main()
