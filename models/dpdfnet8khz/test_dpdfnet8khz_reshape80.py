"""Descriptor and bounds tests for the opt-in RK256 80-lane unpadding path."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np


HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parent / "dpdfnet"):
    sys.path.insert(0, str(path))
from dpdfnet8khz_reshape80 import Reshape80OptimizationMixin
from dpdfnet_precompiled import make_layout
import user_dma_core as udc


class Engine:
    def __init__(self, source, output):
        self.source, self.output = source, output
        self.values = np.full((source.rows, source.padded_last), 0x7FC0, dtype=np.uint16)
        self.values[:, :source.logical_last] = np.arange(source.logical_elements, dtype=np.uint16).reshape(
            source.rows, source.logical_last)
        self.destination = np.full(output.physical_elements, 0x7F80, dtype=np.uint16)
        self.sram = np.full(udc.URAM_NEAR_FULL_SIZE // 2, 0x7FC0, dtype=np.uint16)
        self.calls = []

    def accelerator_memory_to_sram(self, address, sram, elements):
        assert address == self.source.address and sram == 0
        assert elements == self.source.physical_elements and elements * 2 <= udc.URAM_NEAR_FULL_SIZE
        self.calls.append(("read", address, sram, elements))
        self.sram[:elements] = self.values.reshape(-1)

    def sram_to_accelerator_memory(self, sram, address, elements, **stride):
        assert address == self.output.address and sram == 0
        assert elements == self.output.physical_elements
        assert stride == {"stride_bytes_per_chunk": 160, "stride_jump_bytes": 160}
        self.calls.append(("write", sram, address, elements, stride))
        # Explicit model of WRITE_RESP resetting its 128-byte SRAM slice.
        for row in range(self.source.rows):
            self.destination[row * 80:(row + 1) * 80] = self.sram[row * 128:row * 128 + 80]


class Delegate:
    def emit_view(self, index, node):
        self.fallbacks.append(index)


class Compiler(Reshape80OptimizationMixin, Delegate):
    def __init__(self, source, output):
        self.layouts = {"source": source, "output": output}
        self.layout = self.layouts.__getitem__
        self.engine = Engine(source, output)
        self.zero_padding = []
        self.emitter = SimpleNamespace(engine=self.engine, mark_padding_zero=self.zero_padding.append)
        self.fallbacks = []


class Reshape80Test(unittest.TestCase):
    def compile_case(self, rows=320, *, output_shape=None, source_address=0x1000000,
                     output_address=0x2000000):
        source = make_layout("source", (rows, 80), source_address)
        output = make_layout("output", output_shape or (rows * 80,), output_address)
        compiler = Compiler(source, output)
        node = SimpleNamespace(input=["source"], output=["output"])
        compiler.emit_view(422, node)
        return compiler

    def test_all_logical_words_survive_dirty_source_padding_and_sram(self):
        with patch.object(udc, "UE_AXI_DATA_WIDTH_BITS", 256):
            for rows in (4, 8, 20, 320):
                with self.subTest(rows=rows):
                    compiler = self.compile_case(rows)
                    np.testing.assert_array_equal(compiler.engine.destination,
                                                  compiler.engine.values[:, :80].reshape(-1))
                    self.assertEqual(len(compiler.engine.calls), 2)
                    self.assertEqual(compiler.fallbacks, [])

    def test_partial_flat_tail_and_padded_output_delegate(self):
        with patch.object(udc, "UE_AXI_DATA_WIDTH_BITS", 256):
            for rows in (1, 2, 3, 5, 319):
                compiler = self.compile_case(rows)
                self.assertEqual(compiler.fallbacks, [422])
                self.assertEqual(compiler.engine.calls, [])

    def test_multiline_destination_does_not_match_flatten(self):
        with patch.object(udc, "UE_AXI_DATA_WIDTH_BITS", 256):
            compiler = self.compile_case(320, output_shape=(400, 64))
            self.assertEqual(compiler.fallbacks, [422])

    def test_other_axi_width_or_insufficient_sram_delegates_before_dma(self):
        for width, available in ((512, 0x70000), (256, 81920 - 128)):
            with self.subTest(width=width, available=available):
                with patch.object(udc, "UE_AXI_DATA_WIDTH_BITS", width), patch.object(
                        udc, "URAM_NEAR_FULL_SIZE", available):
                    compiler = self.compile_case()
                    self.assertEqual(compiler.fallbacks, [422])
                    self.assertEqual(compiler.engine.calls, [])

    def test_unaligned_or_overlapping_allocations_delegate(self):
        with patch.object(udc, "UE_AXI_DATA_WIDTH_BITS", 256):
            for source, output in ((0x1000020, 0x2000000), (0x1000000, 0x2000020),
                                   (0x1000000, 0x1000080)):
                compiler = self.compile_case(source_address=source, output_address=output)
                self.assertEqual(compiler.fallbacks, [422])
                self.assertEqual(compiler.engine.calls, [])


if __name__ == "__main__":
    unittest.main()
