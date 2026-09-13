"""Independent memory-model checks for native 8-kHz compiler layout operators."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import onnx
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))
from dpdfnet8khz_compile import AlignedMappingEmitter, GraphCompiler, OptimizedGraphCompiler
from dpdfnet_precompiled import make_layout, physical_indices
import user_dma_core as udc


class MemoryEngine:
    def __init__(self):
        self.regions = {}
        self.sram = np.zeros(262144, dtype=np.float32)
        self.constant_cursor = 0x10000000
        self.matmul_shapes = []

    def view(self, address, elements):
        for base, data in self.regions.items():
            offset = (address - base) // 2
            if address >= base and address + elements * 2 <= base + data.size * 2:
                return data[offset:offset + elements]
        raise AssertionError(f"unallocated memory {address:#x}, {elements} BF16 elements")

    def allocate_constant(self, value):
        address = self.constant_cursor
        data = torch.as_tensor(value).to(torch.bfloat16).float().numpy().reshape(-1).copy()
        self.regions[address] = data
        self.constant_cursor += (data.size * 2 + 127) // 128 * 128
        return address

    def accelerator_memory_to_sram(self, address, sram, elements, *, memcpy_length_bytes=None,
                                   stride_bytes_per_chunk=0, stride_jump_bytes=0):
        assert address % 32 == 0 and sram % 128 == 0
        count = elements if memcpy_length_bytes is None else memcpy_length_bytes // 2
        if stride_jump_bytes:
            chunk = stride_bytes_per_chunk // 2
            for offset in range(0, count, chunk):
                self.sram[sram // 2 + offset:sram // 2 + offset + chunk] = self.view(
                    address + offset // chunk * stride_jump_bytes, chunk)
        else:
            self.sram[sram // 2:sram // 2 + count] = self.view(address, count)

    def sram_to_accelerator_memory(self, sram, address, elements, *, memcpy_length_bytes=None,
                                   stride_bytes_per_chunk=0, stride_jump_bytes=0):
        assert address % 32 == 0 and sram % 128 == 0
        count = elements if memcpy_length_bytes is None else memcpy_length_bytes // 2
        if stride_jump_bytes:
            chunk = stride_bytes_per_chunk // 2
            for offset in range(0, count, chunk):
                self.view(address + offset // chunk * stride_jump_bytes, chunk)[:] = (
                    self.sram[sram // 2 + offset:sram // 2 + offset + chunk])
        else:
            self.view(address, count)[:] = self.sram[sram // 2:sram // 2 + count]

    def matmat_mul_core(self, *, M, K, N, A_DRAM_ADDR, B_DRAM_ADDR,
                        OUTPUT_DRAM_ADDR, is_B_quantized=False):
        self.matmul_shapes.append((M, K, N))
        assert not is_B_quantized
        assert A_DRAM_ADDR % 128 == B_DRAM_ADDR % 128 == OUTPUT_DRAM_ADDR % 128 == 0
        source = self.view(A_DRAM_ADDR, M * K).reshape(M, K).copy()
        weight = self.view(B_DRAM_ADDR, N * K).reshape(N, K)
        result = source @ weight.T
        self.view(OUTPUT_DRAM_ADDR, M * N)[:] = torch.from_numpy(result).bfloat16().float().numpy().reshape(-1)

    def eltwise_core_dram(self, *, M, N, dram_a, dram_b, dram_out, mode):
        assert mode == udc.UE_MODE.ELTWISE_ADD
        result = self.view(dram_a, M * N) + self.view(dram_b, M * N)
        self.view(dram_out, M * N)[:] = torch.from_numpy(result).bfloat16().float().numpy()

    def accelerator_memcpy(self, source, destination, size):
        assert source % 32 == destination % 32 == 0
        self.view(destination, size // 2)[:] = self.view(source, size // 2).copy()


class NativeCompilerLayoutTest(unittest.TestCase):
    def setUp(self):
        axi = patch.object(udc, "UE_AXI_DATA_WIDTH_BITS", 256)
        axi.start()
        self.addCleanup(axi.stop)
        self.engine = MemoryEngine()
        self.emitter = AlignedMappingEmitter.__new__(AlignedMappingEmitter)
        self.emitter.engine = self.engine
        self.emitter.layouts = {}
        self.emitter._zero_padding = set()
        self.emitter.mapping_resources = {}
        self.emitter.planning = True
        self.emitter.allocate_constant = self.engine.allocate_constant
        self.emitter.zero_address = 0x100000
        self.engine.regions[self.emitter.zero_address] = np.zeros(262144, dtype=np.float32)
        self.emitter.copy_scratch = self.layout("@copy", (64,), 0x200000)
        self.emitter.source_scratch = self.layout("@source", (64,), 0x300000)
        self.compiler = GraphCompiler.__new__(GraphCompiler)
        self.compiler.emitter = self.emitter
        self.compiler.layouts = self.emitter.layouts
        self.compiler.onnx = onnx
        self.compiler.initializers = {}

    def layout(self, name, shape, address, padding=0):
        layout = make_layout(name, shape, address)
        self.emitter.layouts[name] = layout
        self.engine.regions[address] = np.full(layout.physical_elements, padding, dtype=np.float32)
        return layout

    def seed(self, layout):
        data = ((np.arange(layout.logical_elements) % 97) - 48).astype(np.float32).reshape(layout.shape)
        self.engine.regions[layout.address].reshape(layout.rows, layout.padded_last)[
            :, :layout.logical_last] = data.reshape(layout.rows, layout.logical_last)
        return data

    def assert_logical(self, layout, value):
        actual = self.engine.regions[layout.address].reshape(layout.rows, layout.padded_last)
        np.testing.assert_array_equal(actual[:, :layout.logical_last].reshape(layout.shape), value)
        np.testing.assert_array_equal(actual[:, layout.logical_last:], 0)

    def test_unaligned_81_bin_state_reshape_preserves_all_values(self):
        source = self.layout("source", (3, 1, 1, 81), 0x400000, padding=np.nan)
        destination = self.layout("destination", (243,), 0x500000)
        logical = self.seed(source)
        self.emitter.emit_mapping(source, destination, physical_indices(source))
        self.assert_logical(destination, logical.reshape(-1))

    def test_fifty_wide_projection_pack_and_ten_wide_unpack(self):
        source = self.layout("source", (16, 50), 0x400000, padding=np.nan)
        packed = self.layout("packed", (1, 800), 0x500000)
        unpacked = self.layout("unpacked", (80, 10), 0x600000)
        logical = self.seed(source)
        self.emitter.emit_mapping(source, packed, physical_indices(source))
        self.emitter.emit_mapping(packed, unpacked, physical_indices(packed))
        self.assert_logical(packed, logical.reshape(1, 800))
        self.assert_logical(unpacked, logical.reshape(80, 10))

    def test_reflect_pad_repeats_penultimate_bin_and_ignores_padding(self):
        source = self.layout("source", (1, 1, 1, 80), 0x400000, padding=np.nan)
        destination = self.layout("destination", (1, 1, 1, 81), 0x500000)
        logical = self.seed(source)
        self.compiler.initializers["pads"] = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int64)
        node = onnx.helper.make_node("Pad", ["source", "pads"], ["destination"], mode="reflect")
        self.compiler.emit_pad(node)
        self.assert_logical(destination, np.pad(logical, [(0, 0)] * 3 + [(0, 1)], mode="reflect"))

    def test_partial_scatter_preserves_previously_written_logical_lanes(self):
        source = self.layout("source", (80,), 0x400000)
        destination = self.layout("destination", (243,), 0x500000)
        logical = self.seed(source)
        expected = self.engine.regions[destination.address].copy()
        expected[:243] = 12
        self.engine.regions[destination.address][:243] = 12
        self.emitter.emit_scatter(source, destination, np.arange(80), np.arange(81, 161))
        expected[81:161] = logical
        np.testing.assert_array_equal(self.engine.regions[destination.address], expected)

    def test_clear_81_lane_padding_removes_nonfinite_values_without_changing_data(self):
        destination = self.layout("destination", (3, 81), 0x400000, padding=np.inf)
        logical = self.seed(destination)
        self.emitter.emit_clear_padding(destination)
        self.assert_logical(destination, logical)

    def test_selectors_must_exist_before_final_capture(self):
        source = self.layout("source", (16, 50), 0x400000)
        destination = self.layout("destination", (800,), 0x500000)
        self.seed(source)
        self.emitter.planning = False
        with self.assertRaisesRegex(RuntimeError, "before program capture"):
            self.emitter.emit_mapping(source, destination, physical_indices(source))

    def test_subbeat_stride_reverse_and_repeated_lanes_use_safe_selectors(self):
        source = self.layout("source", (64,), 0x400000)
        destination = self.layout("destination", (16,), 0x500000)
        logical = self.seed(source)
        for indices in (np.arange(0, 32, 2), np.arange(15, -1, -1), np.zeros(16, dtype=np.int64)):
            with self.subTest(indices=indices.tolist()):
                self.emitter.emit_mapping(source, destination, indices)
                self.assert_logical(destination, logical[indices])

    def test_stride_exceeding_descriptor_field_uses_selectors(self):
        jump = (udc.UE_STRIDE_JUMP_MAX_BYTES + 1) // 2
        source = self.layout("source", (jump + 64,), 0x400000)
        destination = self.layout("destination", (2,), 0x900000)
        self.engine.regions[source.address][0] = 3
        self.engine.regions[source.address][jump] = 5
        self.emitter.emit_mapping(source, destination, [0, jump])
        self.assert_logical(destination, np.array([3, 5], dtype=np.float32))

    def test_pixel_shuffle_40_bins_crosses_the_64_lane_boundary(self):
        source = self.layout("source", (1, 2, 64, 1, 40), 0x400000)
        destination = self.layout("destination", (1, 64, 1, 80), 0x500000)
        odd = self.layout("@odd", destination.shape, 0x600000)
        logical = self.seed(source)
        self.compiler.pixel8_aux = {0: {"source": source, "width": 40, "odd": odd}}
        selectors = []
        for parity in range(2):
            selector = torch.zeros(128, 64, dtype=torch.bfloat16)
            for column in range(40):
                selector[2 * column + parity, column] = 1
            selectors.append(self.engine.allocate_constant(selector))
        self.compiler.pixel8_selectors = {0: selectors}
        self.compiler.emit_view(0, SimpleNamespace(output=["destination"]))
        self.assert_logical(destination, logical.transpose(0, 2, 3, 4, 1).reshape(destination.shape))

    def optimized_compiler(self):
        compiler = OptimizedGraphCompiler.__new__(OptimizedGraphCompiler)
        compiler.__dict__.update(self.compiler.__dict__)
        compiler.state_copy_scratch = self.layout("@state_copy", (64,), 0x700000)
        compiler.state_batch_scratch = self.layout("@state_batch", (128, 64), 0x800000)
        return compiler

    def test_batched_state_copy_handles_long_shift_and_partial_boundaries(self):
        compiler = self.optimized_compiler()
        source = self.layout("source", (21000,), 0x400000)
        destination = self.layout("destination", (21000,), 0x500000)
        logical = self.seed(source)
        for source_start, destination_start, count in (
                (17, 81, 64 * 300 + 23), (81, 0, 64 * 300),
                (0, 161, 64 * 300), (31, 63, 301), (32, 64, 81)):
            with self.subTest(source=source_start, destination=destination_start, count=count):
                self.engine.regions[destination.address][:] = 12
                expected = self.engine.regions[destination.address].copy()
                expected[destination_start:destination_start + count] = logical[source_start:source_start + count]
                resource = compiler.prepare_state_copy(
                    source, destination, source_start, destination_start, count)
                self.engine.matmul_shapes.clear()
                compiler.emit_state_copy(resource)
                np.testing.assert_array_equal(self.engine.regions[destination.address], expected)
                self.assertTrue(all(shape[0] <= 128 for shape in self.engine.matmul_shapes))
                self.assertLess(len(self.engine.matmul_shapes), 30)

    def test_native_split10_preserves_all_pairs_and_zero_padding(self):
        compiler = self.optimized_compiler()
        source = self.layout("source", (1, 1, 80, 10), 0x400000)
        destination = self.layout("destination", (1, 1, 80, 5, 2), 0x500000)
        temporary = self.layout("@split10", source.shape, 0x600000)
        logical = self.seed(source)
        compiler.pixel8_aux = {}
        compiler.pack60_aux = {}
        compiler.unpack10_aux = {}
        compiler.split10_aux = {0: temporary}
        compiler.split10_addresses = []
        for pair in range(5):
            selector = torch.zeros(64, 64, dtype=torch.bfloat16)
            selector[0, 2 * pair] = selector[1, 2 * pair + 1] = 1
            compiler.split10_addresses.append(self.engine.allocate_constant(selector))
        compiler.emit_view(0, SimpleNamespace(input=["source"], output=["destination"]))
        self.assert_logical(destination, logical.reshape(destination.shape))
        self.assertEqual(self.engine.matmul_shapes, [(80, 64, 64)] * 5)
        self.assertIn(destination.name, self.emitter._zero_padding)


if __name__ == "__main__":
    unittest.main()
