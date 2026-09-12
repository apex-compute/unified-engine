"""Host regressions for padded-row DMA copies, independent of FPGA access."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
from dpdfnet_precompiled import DeviceEmitter, make_layout, padded_row_patterns
import user_dma_core as udc


class MemoryEngine:
    """Execute DMA byte contracts on raw BF16 bits without compiler helpers."""

    def __init__(self):
        self.regions = {}
        self.sram = np.zeros(udc.URAM_NEAR_FULL_SIZE // 2, dtype=np.uint16)
        self.calls = []

    def view(self, address, nbytes):
        assert address % 2 == nbytes % 2 == 0
        for base, values in self.regions.items():
            if base <= address and address + nbytes <= base + values.nbytes:
                offset = (address - base) // 2
                return values[offset:offset + nbytes // 2]
        raise AssertionError(f"DMA outside allocated memory: {address:#x}, {nbytes}")

    def accelerator_memory_to_sram(
            self, source, sram, elements, *, memcpy_length_bytes=None,
            stride_bytes_per_chunk=0, stride_jump_bytes=0):
        nbytes = elements * 2 if memcpy_length_bytes is None else memcpy_length_bytes
        assert sram % 128 == 0
        assert nbytes <= udc.URAM_NEAR_FULL_SIZE
        assert sram + nbytes <= self.sram.nbytes
        self.calls.append(("read", source, nbytes,
                           stride_bytes_per_chunk, stride_jump_bytes))
        target = self.sram[sram // 2:(sram + nbytes) // 2]
        if stride_jump_bytes:
            assert 0 < stride_bytes_per_chunk <= udc.UE_STRIDE_CHUNK_MAX_BYTES
            assert stride_jump_bytes <= udc.UE_STRIDE_JUMP_MAX_BYTES
            assert nbytes % stride_bytes_per_chunk == 0
            for offset in range(0, nbytes, stride_bytes_per_chunk):
                source_offset = offset // stride_bytes_per_chunk * stride_jump_bytes
                target[offset // 2:(offset + stride_bytes_per_chunk) // 2] = (
                    self.view(source + source_offset, stride_bytes_per_chunk))
        else:
            target[:] = self.view(source, nbytes)

    def sram_to_accelerator_memory(
            self, sram, destination, elements, *, memcpy_length_bytes=None):
        nbytes = elements * 2 if memcpy_length_bytes is None else memcpy_length_bytes
        assert sram % 128 == 0
        assert nbytes <= udc.URAM_NEAR_FULL_SIZE
        self.calls.append(("write", destination, nbytes, 0, 0))
        self.view(destination, nbytes)[:] = self.sram[sram // 2:(sram + nbytes) // 2]


class DPDFNetRowCopyTest(unittest.TestCase):
    def setUp(self):
        self.engine = MemoryEngine()
        self.emitter = DeviceEmitter.__new__(DeviceEmitter)
        self.emitter.engine = self.engine
        self.emitter.layouts = {}
        self.emitter._zero_padding = set()
        self.emitter.zero_address = 0x100000
        self.engine.regions[self.emitter.zero_address] = np.zeros(
            udc.URAM_NEAR_FULL_SIZE // 2, dtype=np.uint16)
        self.cursor = 0x1000000

    def layout(self, name, shape, *, padding=0):
        layout = make_layout(name, shape, self.cursor)
        self.cursor += 0x1000000
        self.emitter.layouts[name] = layout
        self.engine.regions[layout.address] = np.full(
            layout.physical_elements, padding, dtype=np.uint16)
        return layout

    def seed(self, layout):
        # Distinct finite BF16 bit patterns; no floating-point arithmetic is
        # involved in either the device copy or this reference calculation.
        logical = (0x3E00 + np.arange(layout.logical_elements) % 256).astype(np.uint16)
        self.engine.regions[layout.address].reshape(
            layout.rows, layout.padded_last)[:, :layout.logical_last] = logical.reshape(
                layout.rows, layout.logical_last)
        return logical.reshape(layout.shape)

    @staticmethod
    def indices(layout):
        return (np.arange(layout.rows)[:, None] * layout.padded_last
                + np.arange(layout.logical_last)).reshape(-1)

    def assert_output(self, layout, expected):
        physical = self.engine.regions[layout.address].reshape(
            layout.rows, layout.padded_last)
        np.testing.assert_array_equal(
            physical[:, :layout.logical_last].reshape(layout.shape), expected)
        np.testing.assert_array_equal(physical[:, layout.logical_last:], 0)

    def test_complex_transpose_uses_five_gathers_and_preserves_bits(self):
        source = self.layout("source", (96, 5, 2))
        destination = self.layout("destination", (5, 96, 2))
        logical = self.seed(source)
        self.emitter.mark_padding_zero(source)
        mapping = self.indices(source).reshape(source.shape).transpose(1, 0, 2).reshape(-1)
        self.emitter.emit_scatter(source, destination, mapping, self.indices(destination))
        self.assert_output(destination, logical.transpose(1, 0, 2))
        self.assertEqual(len(self.engine.calls), 10)
        for call in self.engine.calls[::2]:
            self.assertEqual(call[0], "read")
            self.assertEqual(call[2:], (96 * 128, 128, 5 * 128))

    def test_identity_and_slice_preserve_zero_padding(self):
        for selected in (slice(None), slice(2, 6)):
            with self.subTest(selected=selected):
                self.setUp()
                source = self.layout("source", (8, 2))
                logical = self.seed(source)
                expected = logical[selected]
                destination = self.layout("destination", expected.shape, padding=0xFFFF)
                self.emitter.mark_padding_zero(source)
                mapping = self.indices(source).reshape(source.shape)[selected].reshape(-1)
                self.emitter.emit_mapping(source, destination, mapping)
                self.assert_output(destination, expected)
                self.assertEqual(len(self.engine.calls), 4)  # Zero fill plus bulk copy.
                self.assertIn(destination.name, self.emitter._zero_padding)

    def test_unknown_nan_padding_uses_logical_copies_and_becomes_safe_source(self):
        source = self.layout("source", (4, 2), padding=0x7FC0)
        destination = self.layout("destination", source.shape, padding=0xFFFF)
        logical = self.seed(source)
        self.emitter.emit_mapping(source, destination, self.indices(source))
        self.assert_output(destination, logical)
        self.assertEqual(len(self.engine.calls), 2 + source.rows * 2)
        self.assertTrue(all(call[2] == 4 for call in self.engine.calls[2:]))

        copied = self.layout("copied", source.shape)
        self.engine.calls.clear()
        self.emitter.emit_scatter(
            destination, copied, self.indices(destination), self.indices(copied))
        self.assert_output(copied, logical)
        self.assertEqual(len(self.engine.calls), 2)

    def test_concat_scatter_preserves_unselected_rows(self):
        source = self.layout("source", (3, 2))
        destination = self.layout("destination", (8, 2), padding=0x3F00)
        self.seed(source)
        self.emitter.mark_padding_zero(source)
        expected = self.engine.regions[destination.address].copy().reshape(8, 64)
        expected[[1, 2, 6]] = self.engine.regions[source.address].reshape(3, 64)
        selected = self.indices(destination).reshape(8, 2)[[1, 2, 6]].reshape(-1)
        self.emitter.emit_scatter(source, destination, self.indices(source), selected)
        np.testing.assert_array_equal(
            self.engine.regions[destination.address].reshape(8, 64), expected)
        self.assertEqual(len(self.engine.calls), 4)

    def test_reversed_and_repeated_rows_copy_in_requested_order(self):
        source = self.layout("source", (5, 2))
        destination = self.layout("destination", (6, 2))
        logical = self.seed(source)
        self.emitter.mark_padding_zero(source)
        selected = np.array([4, 3, 2, 2, 0, 1])
        mapping = self.indices(source).reshape(5, 2)[selected].reshape(-1)
        self.emitter.emit_scatter(source, destination, mapping, self.indices(destination))
        self.assert_output(destination, logical[selected])

    def test_partial_row_scatter_does_not_overwrite_neighbors(self):
        source = self.layout("source", (2, 2))
        destination = self.layout("destination", (2, 2), padding=0x3F00)
        logical = self.seed(source)
        self.emitter.mark_padding_zero(source)
        expected = self.engine.regions[destination.address].copy()
        expected[1] = logical[0, 0]
        self.emitter.emit_scatter(source, destination, [0], [1])
        np.testing.assert_array_equal(self.engine.regions[destination.address], expected)
        self.assertEqual([call[2] for call in self.engine.calls], [2, 2])

    def test_column_permutation_stays_logical_and_rejects_row_fastpath(self):
        source = self.layout("source", (4, 2))
        destination = self.layout("destination", source.shape)
        logical = self.seed(source)
        self.emitter.mark_padding_zero(source)
        mapping = self.indices(source).reshape(4, 2)[:, ::-1].reshape(-1)
        self.assertIsNone(padded_row_patterns(
            source, destination, mapping, self.indices(destination)))
        self.emitter.emit_mapping(source, destination, mapping)
        self.assert_output(destination, logical[:, ::-1])

    def test_uram_limit_splits_gather_without_skipping_rows(self):
        source = self.layout("source", (20, 2))
        destination = self.layout("destination", (10, 2))
        logical = self.seed(source)
        self.emitter.mark_padding_zero(source)
        mapping = self.indices(source).reshape(20, 2)[::2].reshape(-1)
        with patch.object(udc, "URAM_NEAR_FULL_SIZE", 3 * 128):
            self.emitter.emit_scatter(
                source, destination, mapping, self.indices(destination))
        self.assert_output(destination, logical[::2])
        self.assertEqual([call[2] for call in self.engine.calls[::2]],
                         [384, 384, 384, 128])

    def test_stride_limits_fall_back_to_logical_row_copies(self):
        for field, limit in (("UE_STRIDE_JUMP_MAX_BYTES", 200),
                             ("UE_STRIDE_CHUNK_MAX_BYTES", 64)):
            with self.subTest(field=field):
                self.setUp()
                source = self.layout("source", (8, 2))
                destination = self.layout("destination", (4, 2))
                logical = self.seed(source)
                self.emitter.mark_padding_zero(source)
                mapping = self.indices(source).reshape(8, 2)[::2].reshape(-1)
                with patch.object(udc, field, limit):
                    self.emitter.emit_scatter(
                        source, destination, mapping, self.indices(destination))
                self.assert_output(destination, logical[::2])
                self.assertEqual(len(self.engine.calls), 8)
                self.assertTrue(all(call[2] == 4 for call in self.engine.calls))

    def test_unpadded_rows_need_no_padding_fact(self):
        source = self.layout("source", (3, 64))
        destination = self.layout("destination", source.shape)
        logical = self.seed(source)
        self.emitter.emit_scatter(
            source, destination, self.indices(source), self.indices(destination))
        self.assert_output(destination, logical)
        self.assertEqual(len(self.engine.calls), 2)

    def test_padding_facts_exclude_scratch_and_unregistered_layouts(self):
        scratch = self.layout("@scratch", (2, 2))
        self.emitter.mark_padding_zero(scratch)
        self.emitter.mark_padding_zero(make_layout("unregistered", (2, 2), 0x400000))
        self.assertEqual(self.emitter._zero_padding, set())


if __name__ == "__main__":
    unittest.main()
