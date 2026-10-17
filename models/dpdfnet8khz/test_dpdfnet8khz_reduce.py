"""Independent logical and SRAM-bound checks for complex pair reduction."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_dpdfnet8khz_compile import MemoryEngine
from dpdfnet8khz_reduce import ReductionOptimizationMixin
from dpdfnet_precompiled import make_layout
import user_dma_core as udc


class ReduceEngine(MemoryEngine):
    def start_queue_for_bf16_matvec_operation(self, **call):
        self.matvec_calls.append(call)
        self.assert_call_bounds(call)
        vector = self.sram[call["vector_sram_start_addr"] // 2:][:64]
        matrix = np.stack([
            self.sram[call["matrix_sram_start_addr"] // 2 + row * call["stride_z"]:][:64]
            for row in range(call["N"])])
        value = torch.from_numpy(matrix @ vector).bfloat16().float().numpy()
        self.sram[call["output_sram_wb_addr"] // 2:][:call["N"]] = value

    @staticmethod
    def assert_call_bounds(call):
        assert call["K"] == 64 and call["N"] in (80, 81) and call["stride_z"] == 64
        vector, matrix, output = (call[name] for name in (
            "vector_sram_start_addr", "matrix_sram_start_addr", "output_sram_wb_addr"))
        assert all(address % 128 == 0 for address in (vector, matrix, output))
        assert 0 <= vector and vector + 128 <= output
        assert output + 256 <= udc.URAM_NEAR_FULL_SIZE
        assert 0x80000 <= matrix and matrix - 0x80000 + call["N"] * 128 <= udc.URAM_NEAR_FULL_SIZE


class Baseline:
    def emit_reduce_sum(self, index, node):
        self.fallback = True


class Compiler(ReductionOptimizationMixin, Baseline):
    def layout(self, name):
        return self.layouts[name]

    def _integer_initializer(self, name):
        return self.axes


class ComplexReductionTest(unittest.TestCase):
    def prepare(self, bins, *, known=False, padding=np.nan):
        compiler, engine = Compiler(), ReduceEngine()
        source = make_layout("source", (1, 1, bins, 2), 0x1000000)
        output = make_layout("output", (1, 1, bins), 0x2000000)
        values = np.column_stack(((np.arange(bins) % 17 - 8) / 8,
                                  (np.arange(bins) % 13 - 6) / 4)).astype(np.float32)
        physical = np.full((bins, 64), padding, dtype=np.float32)
        physical[:, :2] = values
        engine.regions[source.address] = physical.reshape(-1)
        engine.regions[output.address] = np.full(output.physical_elements, np.nan, dtype=np.float32)
        zeros = engine.allocate_constant(torch.zeros(81 * 64))
        vector = torch.zeros(64)
        vector[:2] = 1
        compiler.reduce_resources = {2: engine.allocate_constant(vector)}
        compiler.layouts = {"source": source, "output": output}
        compiler.axes = np.array([-1])
        compiler.fallback = False
        facts = {"source"} if known else set()
        compiler.emitter = SimpleNamespace(engine=engine, zero_address=zeros,
                                          _zero_padding=facts,
                                          mark_padding_zero=lambda layout: facts.add(layout.name))
        engine.sram[:] = np.nan
        return compiler, engine, source, output, values

    def test_both_native_sizes_sum_signed_values_and_clear_output_padding(self):
        for bins in (80, 81):
            for known, padding in ((True, 0), (False, np.nan), (False, np.inf), (False, 17)):
                with self.subTest(bins=bins, known=known, padding=padding):
                    compiler, engine, source, output, values = self.prepare(bins, known=known, padding=padding)
                    compiler.emit_reduce_sum(2, SimpleNamespace(input=["source", "axes"], output=["output"]))
                    self.assertFalse(compiler.fallback)
                    result = engine.regions[output.address]
                    expected = torch.from_numpy(values.sum(axis=1)).bfloat16().float().numpy()
                    np.testing.assert_array_equal(result[:bins], expected)
                    np.testing.assert_array_equal(result[bins:], 0)
                    source_reads = [entry for entry in engine.dma_reads
                                    if source.address <= entry[0] < source.address + source.size_bytes]
                    self.assertEqual(len(source_reads), 1 if known else bins)
                    self.assertEqual(len(engine.matvec_calls), 1)
                    self.assertIn("output", compiler.emitter._zero_padding)

    def test_other_axis_or_size_keeps_original_lowering(self):
        for bins, axis in ((79, -1), (81, 1)):
            compiler, engine, _, _, _ = self.prepare(bins)
            compiler.axes = np.array([axis])
            compiler.emit_reduce_sum(2, SimpleNamespace(input=["source", "axes"], output=["output"]))
            self.assertTrue(compiler.fallback)
            self.assertFalse(engine.dma_reads)

    def test_memory_bounds_and_alignment_fail_before_emission(self):
        for attribute, value in (("_REDUCE_OUTPUT", 64), ("_REDUCE_OUTPUT", 0),
                                 ("_REDUCE_MATRIX", 0xFFFF80), ("_REDUCE_MATRIX", 0x80001)):
            with self.subTest(attribute=attribute):
                compiler, engine, _, _, _ = self.prepare(81)
                setattr(compiler, attribute, value)
                compiler.emit_reduce_sum(2, SimpleNamespace(input=["source", "axes"], output=["output"]))
                self.assertTrue(compiler.fallback)
                self.assertFalse(engine.dma_reads)
        compiler, engine, _, _, _ = self.prepare(81)
        with patch.object(udc, "URAM_NEAR_FULL_SIZE", 8192):
            compiler.emit_reduce_sum(2, SimpleNamespace(input=["source", "axes"], output=["output"]))
        self.assertTrue(compiler.fallback)
        self.assertFalse(engine.dma_reads)


if __name__ == "__main__":
    unittest.main()
