"""Packed-state copying regressions without a device or model download."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))
from dpdfnet_compile import GraphCompiler
from dpdfnet_precompiled import make_layout, pack_tensor
import user_dma_core as udc


class MemoryEngine:
    """Independent BF16 memory model for emitted DMA, matrix and add operations."""

    def __init__(self):
        self.regions = {}
        self.constant_cursor = 0x1000000

    def view(self, address, count):
        for base, data in self.regions.items():
            offset = (address - base) // 2
            if address >= base and offset + count <= data.numel():
                return data[offset:offset + count]
        raise AssertionError(f"out-of-bounds memory access: {address:#x}, {count} elements")

    def allocate_constant(self, value):
        address = self.constant_cursor
        value = value.flatten().clone()
        self.regions[address] = value
        self.constant_cursor += value.numel() * 2
        return address

    def accelerator_memcpy(self, source, destination, nbytes):
        assert source % 64 == destination % 64 == nbytes % 64 == 0
        self.view(destination, nbytes // 2).copy_(self.view(source, nbytes // 2).clone())

    def matmat_mul_core(self, *, M, K, N, A_DRAM_ADDR, B_DRAM_ADDR,
                        OUTPUT_DRAM_ADDR, is_B_quantized):
        assert M == 1 and K == N == 64 and not is_B_quantized
        assert A_DRAM_ADDR % 128 == B_DRAM_ADDR % 128 == OUTPUT_DRAM_ADDR % 128 == 0
        source = self.view(A_DRAM_ADDR, K).float()
        matrix = self.view(B_DRAM_ADDR, N * K).float().reshape(N, K)
        self.view(OUTPUT_DRAM_ADDR, N).copy_((matrix @ source).to(torch.bfloat16))

    def eltwise_core_dram(self, *, M, N, dram_a, dram_b, dram_out, mode):
        assert M == 1 and N == 64 and mode == udc.UE_MODE.ELTWISE_ADD
        result = self.view(dram_a, N).float() + self.view(dram_b, N).float()
        self.view(dram_out, N).copy_(result.to(torch.bfloat16))


class DPDFNetStateCopyTest(unittest.TestCase):
    def setUp(self):
        self.engine = MemoryEngine()
        self.compiler = GraphCompiler.__new__(GraphCompiler)
        self.compiler.emitter = SimpleNamespace(
            engine=self.engine, allocate_constant=self.engine.allocate_constant)
        self.compiler.state_copy_scratch = make_layout("scratch", (64,), 0x50000)
        self.engine.regions[0x50000] = torch.zeros(64, dtype=torch.bfloat16)

    def test_canonical_state_boundaries_preserve_neighbors_and_slice_back(self):
        state = make_layout("state", (45424,), 0x20000)
        expected = pack_tensor((torch.arange(45424) % 127 - 63), state)
        self.engine.regions[state.address] = expected.clone()
        for index, (start, count) in enumerate(((39968, 966), (40934, 2880), (43814, 1610))):
            with self.subTest(start=start):
                source = make_layout("segment", (count,), 0x100000 + index * 0x10000)
                value = ((torch.arange(count) % 61 - 30).float() / 8 + index * 10)
                packed = pack_tensor(value, source)
                self.engine.regions[source.address] = packed.clone()
                plan = self.compiler.prepare_state_copy(source, state, 0, start, count)
                self.compiler.emit_state_copy(plan)
                expected[start:start + count] = packed[:count]
                torch.testing.assert_close(
                    self.engine.regions[state.address], expected, rtol=0, atol=0)

        for index, (start, count) in enumerate(((39968, 966), (40934, 2880), (43814, 1610))):
            with self.subTest(slice_start=start):
                destination = make_layout("slice", (count,), 0x200000 + index * 0x10000)
                self.engine.regions[destination.address] = torch.zeros(
                    destination.physical_elements, dtype=torch.bfloat16)
                plan = self.compiler.prepare_state_copy(state, destination, start, 0, count)
                self.compiler.emit_state_copy(plan)
                torch.testing.assert_close(
                    self.engine.regions[destination.address],
                    pack_tensor(expected[start:start + count], destination), rtol=0, atol=0)

    def test_partial_rows_and_cross_row_shifts(self):
        for source_start, destination_start, count in (
                (0, 0, 1), (1, 17, 64), (31, 32, 33),
                (38, 0, 64), (38, 63, 65), (63, 38, 66), (0, 32, 966)):
            with self.subTest(source=source_start, destination=destination_start, count=count):
                source = make_layout("source", (1024,), 0x100000)
                destination = make_layout("destination", (1024,), 0x20000)
                values = (torch.arange(1024) % 127 - 63).to(torch.bfloat16)
                self.engine.regions[source.address] = values.clone()
                expected = torch.full((1024,), -100, dtype=torch.bfloat16)
                self.engine.regions[destination.address] = expected.clone()
                plan = self.compiler.prepare_state_copy(
                    source, destination, source_start, destination_start, count)
                self.compiler.emit_state_copy(plan)
                expected[destination_start:destination_start + count] = values[
                    source_start:source_start + count]
                torch.testing.assert_close(
                    self.engine.regions[destination.address], expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
