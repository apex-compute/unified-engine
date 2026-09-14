"""Independent memory-contract tests for overlapping 128-input state copies."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "dpdfnet"))
from dpdfnet8khz_state_shift import StateShiftOptimizationMixin
from dpdfnet_compile import GraphCompiler as SharedCompiler
from dpdfnet_precompiled import make_layout
import user_dma_core as udc


class MemoryEngine:
    def __init__(self):
        self.regions = {}
        self.sram = np.full(0x100000 // 2, 0x7FC0, dtype=np.uint16)
        self.matvecs = []
        self.reads = []

    def view(self, address, size):
        assert address % 2 == size % 2 == 0
        for base, values in self.regions.items():
            if base <= address and address + size <= base + values.nbytes:
                offset = (address - base) // 2
                return values[offset:offset + size // 2]
        raise AssertionError(f"DMA outside allocation: {address:#x} + {size}")

    def sram_view(self, address, size):
        assert address % 128 == 0
        assert address % 0x80000 + size <= udc.URAM_NEAR_FULL_SIZE
        return self.sram[address // 2:(address + size) // 2]

    def accelerator_memory_to_sram(self, source, destination, elements,
                                   *, memcpy_length_bytes=None):
        size = elements * 2 if memcpy_length_bytes is None else memcpy_length_bytes
        self.reads.append((source, destination, size))
        self.sram_view(destination, size)[:] = self.view(source, size)

    def sram_to_accelerator_memory(self, source, destination, elements,
                                   *, memcpy_length_bytes=None):
        size = elements * 2 if memcpy_length_bytes is None else memcpy_length_bytes
        self.view(destination, size)[:] = self.sram_view(source, size)

    def start_queue_for_bf16_matvec_operation(self, **operation):
        self.matvecs.append(operation)
        assert operation["K"] == 128 and operation["N"] == 64
        assert operation["stride_z"] == 64
        vector = self.sram_view(operation["vector_sram_start_addr"], 256)
        # A NaN in an unselected lane still poisons one-hot BF16 arithmetic.
        assert not np.any((vector & 0x7F80) == 0x7F80)
        # RTL advances B by stride_z per 64-lane K block. A full matrix
        # row spans K/64 such blocks, rather than one stride_z step.
        blocks = operation["K"] // 64
        matrix = np.stack([
            np.concatenate([
                self.sram_view(operation["matrix_sram_start_addr"]
                               + (row * blocks + block) * operation["stride_z"] * 2, 128)
                for block in range(blocks)])
            for row in range(operation["N"])])
        assert np.all(np.count_nonzero(matrix, axis=1) == 1)
        selected = np.argmax(matrix, axis=1)
        assert np.all(matrix[np.arange(64), selected] == 0x3F80)
        self.sram_view(operation["output_sram_wb_addr"], 128)[:] = vector[selected]


class Emitter:
    def __init__(self, engine):
        self.engine = engine
        self.cursor = 0x5000000

    def allocate_constant(self, tensor):
        address = self.cursor
        values = tensor.contiguous().view(torch.uint16).numpy().reshape(-1).copy()
        self.engine.regions[address] = values
        self.cursor += (values.nbytes + 127) // 128 * 128
        return address


class LegacyCopy:
    def _prepare_state_copies(self):
        return self.resources

    def emit_state_copy(self, resource):
        self.fallbacks.append(resource)
        source = self.engine.regions[resource["source"].address]
        output = self.engine.regions[resource["destination"].address]
        source_start, destination_start = resource["source_start"], resource["destination_start"]
        if resource["prefix"]:
            count = resource["prefix"]
            output[destination_start:destination_start + count] = source[source_start:source_start + count]
        for row in resource["rows"]:
            begin = max(row["destination_row"] * 64, destination_start)
            end = min((row["destination_row"] + 1) * 64,
                      destination_start + resource["_test_count"])
            offset = source_start + begin - destination_start
            output[begin:end] = source[offset:offset + end - begin]


class Compiler(StateShiftOptimizationMixin, LegacyCopy):
    pass


class StateShiftTest(unittest.TestCase):
    def setUp(self):
        self.engine = MemoryEngine()
        self.compiler = Compiler()
        self.compiler.engine = self.engine
        self.compiler.emitter = Emitter(self.engine)
        self.compiler.fallbacks = []

    def copy(self, source_size, source_start, destination_start, count, *, destination_extra=29):
        source = make_layout("source", (source_size,), 0x1000000)
        output = make_layout("output", (destination_start + count + destination_extra,), 0x2000000)
        values = np.full(source.physical_elements, 0x7FC0, dtype=np.uint16)
        values[:source_size] = 0x1000 + np.arange(source_size, dtype=np.uint16) % 0x6000
        self.engine.regions[source.address] = values
        self.engine.regions[output.address] = np.full(output.physical_elements, 0x3F01, dtype=np.uint16)
        expected = self.engine.regions[output.address].copy()
        expected[destination_start:destination_start + count] = values[source_start:source_start + count]
        resource = SharedCompiler.prepare_state_copy(
            self.compiler, source, output, source_start, destination_start, count)
        resource["_test_count"] = count
        self.compiler.resources = {"case": [resource]}
        self.compiler._prepare_state_copies()
        self.compiler.emit_state_copy(resource)
        np.testing.assert_array_equal(self.engine.regions[output.address], expected)
        return resource

    def test_native_large_slice_preserves_overlapping_windows_across_batches(self):
        self.copy(37860, 8564, 0, 25600)
        self.assertEqual(len(self.engine.matvecs), 400)
        self.assertEqual(self.compiler.fallbacks, [])
        input_reads = [entry for entry in self.engine.reads if entry[1] == 0]
        self.assertEqual([entry[2] for entry in input_reads], [129 * 128] * 3 + [17 * 128])
        for index, operation in enumerate(self.engine.matvecs):
            self.assertEqual(operation["vector_sram_start_addr"], index % 128 * 128)

    def test_concat_preserves_partial_boundary_neighbors(self):
        self.copy(25600, 0, 8564, 25600)
        self.assertEqual(len(self.engine.matvecs), 399)
        self.assertEqual(len(self.compiler.fallbacks), 2)
        self.assertTrue(all(item["rows"][0]["keep"] is not None
                            for item in self.compiler.fallbacks))

    def test_unknown_nan_source_padding_retains_last_window_fallback(self):
        resource = self.copy(810, 0, 58, 810)
        self.assertTrue(self.engine.matvecs)
        unsafe_full_rows = [row for item in self.compiler.fallbacks for row in item["rows"]
                            if row["keep"] is None]
        self.assertEqual(len(unsafe_full_rows), 1)
        self.assertIsNone(self.compiler._full_shift_row(resource, unsafe_full_rows[0]))

    def test_all_unaligned_phases_preserve_requested_subrange(self):
        for phase in range(1, 64):
            with self.subTest(phase=phase):
                self.setUp()
                self.copy(512, phase, 7, 320)

    def test_aligned_prefix_uses_existing_dma_path(self):
        self.copy(1024, 32, 64, 480)
        self.assertEqual(self.engine.matvecs, [])
        self.assertEqual(len(self.compiler.fallbacks), 1)
        self.assertEqual(self.compiler.fallbacks[0]["prefix"], 480)

    def test_unfused_aligned_rows_retain_delegate_batching(self):
        # Equal unaligned offsets produce many phase-zero full rows. Passing
        # each row to the old copier separately defeats its matrix batching.
        resource = self.copy(19300, 17, 81, 19223)
        self.assertEqual(self.engine.matvecs, [])
        self.assertEqual(len(self.compiler.fallbacks), 1)
        self.assertEqual(self.compiler.fallbacks[0]["rows"], resource["rows"])
        self.assertGreater(len(resource["rows"]), 300)

    def test_small_buffers_fall_back_without_partial_emission(self):
        with patch.object(udc, "URAM_NEAR_FULL_SIZE", 8192):
            self.copy(512, 52, 0, 320)
        self.assertEqual(self.engine.matvecs, [])
        self.assertEqual(self.engine.reads, [])
        self.assertEqual(len(self.compiler.fallbacks), 1)


if __name__ == "__main__":
    unittest.main()
