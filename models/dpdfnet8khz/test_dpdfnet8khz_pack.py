"""Independent SRAM interpreter for complex pair packing."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from test_dpdfnet8khz_compile import MemoryEngine
from dpdfnet8khz_pack import PairPackOptimizationMixin
from dpdfnet_precompiled import make_layout
import user_dma_core as udc


class PackEngine(MemoryEngine):
    def start_queue_for_bf16_matvec_operation(self, **call):
        self.matvec_calls.append(call)
        self.assert_sram_ranges(call)
        source = self.sram[call['vector_sram_start_addr'] // 2:][:call['K']].copy()
        # Hardware stride advances each 64-lane block of K. This differs
        # from an ordinary matrix leading dimension when K exceeds 64.
        stride = call['K'] // 64 * call['stride_z']
        weight = np.stack([
            self.sram[call['matrix_sram_start_addr'] // 2 + row * stride:][:call['K']]
            for row in range(call['N'])])
        value = torch.from_numpy(source @ weight.T).bfloat16().float().numpy()
        output = self.sram[call['output_sram_wb_addr'] // 2:][:64]
        output[:call['N']] = value

    @staticmethod
    def assert_sram_ranges(call):
        assert call['K'] in (64, 128) and 2 <= call['N'] <= 64
        vector, matrix, output = (call[key] for key in (
            'vector_sram_start_addr', 'matrix_sram_start_addr', 'output_sram_wb_addr'))
        assert all(address % 128 == 0 for address in (vector, matrix, output))
        assert 0 <= vector and vector + call['K'] * 2 <= udc.URAM_NEAR_FULL_SIZE
        assert 0x80000 <= matrix
        stride = call['K'] // 64 * call['stride_z']
        assert matrix - 0x80000 + ((call['N'] - 1) * stride + call['K']) * 2 <= udc.URAM_NEAR_FULL_SIZE
        assert output >= vector + call['K'] * 2
        assert output + 128 <= udc.URAM_NEAR_FULL_SIZE


class Baseline:
    def emit_view(self, index, node):
        self.fallback = True


class Compiler(PairPackOptimizationMixin, Baseline):
    def layout(self, name):
        return self.layouts[name]


class PairPackTest(unittest.TestCase):
    def compiler(self, rows, *, padding_known=True):
        compiler = Compiler.__new__(Compiler)
        engine = PackEngine()
        engine.sram[:] = np.nan
        source = make_layout('source', (rows, 2), 0x1000000)
        output = make_layout('output', (rows * 2,), 0x2000000)
        engine.regions[source.address] = np.zeros(source.physical_elements, dtype=np.float32)
        logical = ((np.arange(rows * 2) % 127) - 63).astype(np.float32).reshape(rows, 2)
        engine.regions[source.address].reshape(rows, 64)[:, :2] = logical
        engine.regions[output.address] = np.full(output.physical_elements, np.nan, dtype=np.float32)
        zeros = engine.allocate_constant(torch.zeros(128))
        compiler.emitter = SimpleNamespace(
            engine=engine, zero_address=zeros,
            _zero_padding={'source'} if padding_known else set(),
            mark_padding_zero=lambda layout: None)
        compiler.identity_address = engine.allocate_constant(torch.eye(64))
        selector = torch.zeros(64, 128)
        for pair in range(32):
            selector[2 * pair, pair] = selector[2 * pair + 1, 64 + pair] = 1
        compiler.pair_pack_selector = engine.allocate_constant(selector)
        compiler.layouts = {'source': source, 'output': output}
        compiler.pair_pack_aux = {0: {}}
        compiler.fallback = False
        return compiler, engine, source, output, logical

    def test_pack_chunks_preserves_pairs_and_clears_all_padding(self):
        for rows in (1, 2, 31, 32, 33, 243, 405, 1200):
            with self.subTest(rows=rows):
                compiler, engine, _, output, logical = self.compiler(rows)
                compiler.emit_view(0, SimpleNamespace(input=['source'], output=['output']))
                self.assertFalse(compiler.fallback)
                actual = engine.regions[output.address]
                np.testing.assert_array_equal(actual[:rows * 2], logical.reshape(-1))
                np.testing.assert_array_equal(actual[rows * 2:], 0)
                self.assertEqual(len(engine.matvec_calls), 3 * ((rows + 31) // 32))

    def test_unknown_nan_padding_retains_previous_lowering(self):
        compiler, engine, source, _, _ = self.compiler(33, padding_known=False)
        engine.regions[source.address].reshape(33, 64)[:, 2:] = np.nan
        compiler.emit_view(0, SimpleNamespace(input=['source'], output=['output']))
        self.assertTrue(compiler.fallback)
        self.assertFalse(engine.matvec_calls)

    def test_oversized_source_keeps_fallback_without_dma(self):
        compiler, engine, _, _, _ = self.compiler(3072)
        compiler.emit_view(0, SimpleNamespace(input=['source'], output=['output']))
        self.assertTrue(compiler.fallback)
        self.assertFalse(engine.matvec_calls)
        self.assertTrue(np.isnan(engine.sram).all())


if __name__ == '__main__':
    unittest.main()
