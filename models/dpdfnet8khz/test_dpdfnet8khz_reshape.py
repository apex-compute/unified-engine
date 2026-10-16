"""Independent SRAM checks for native 800-to-80x10 coefficient unpacking."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_dpdfnet8khz_pack import PackEngine
from dpdfnet8khz_reshape import CoefficientReshapeOptimizationMixin
from dpdfnet_precompiled import make_layout
import user_dma_core as udc


class Baseline:
    def __init__(self, source, output, engine):
        self.layouts = {source.name: source, output.name: output}
        facts = set()
        self.emitter = SimpleNamespace(
            engine=engine, allocate_constant=engine.allocate_constant,
            zero_address=engine.allocate_constant(torch.zeros(80 * 64)),
            _zero_padding=facts,
            mark_padding_zero=lambda layout: facts.add(layout.name))
        self.fallback = False

    def emit_view(self, index, node):
        self.fallback = True


class Compiler(CoefficientReshapeOptimizationMixin, Baseline):
    def layout(self, name):
        return self.layouts[name]


class CoefficientReshapeTest(unittest.TestCase):
    def prepare(self, *, source_shape=(1, 1, 800), output_shape=(1, 1, 80, 10),
                padding=np.nan):
        engine = PackEngine()
        source = make_layout('source', source_shape, 0x1000000)
        output = make_layout('output', output_shape, 0x2000000)
        # Each position has an independently generated BF16 value, exposing
        # dropped rows, repeated phases, wrong K strides and boundary shifts.
        rng = np.random.default_rng(318)
        logical = torch.from_numpy(rng.normal(size=source.logical_elements).astype(np.float32))
        logical = logical.bfloat16().float().numpy()
        physical = np.full(source.physical_elements, padding, dtype=np.float32)
        physical.reshape(source.rows, source.padded_last)[:, :source.logical_last] = (
            logical.reshape(source.rows, source.logical_last))
        engine.regions[source.address] = physical
        engine.regions[output.address] = np.full(output.physical_elements, np.nan, dtype=np.float32)
        engine.sram[:] = np.nan
        compiler = Compiler(source, output, engine)
        return compiler, engine, source, output, logical

    def test_all_phases_crossings_and_final_row_preserve_bf16_with_dirty_padding(self):
        for padding in (np.nan, np.inf, -np.inf, 17):
            with self.subTest(padding=padding):
                compiler, engine, source, output, logical = self.prepare(padding=padding)
                compiler.emit_view(439, SimpleNamespace(input=['source'], output=['output']))
                self.assertFalse(compiler.fallback)
                actual = engine.regions[output.address].reshape(80, 64)
                np.testing.assert_array_equal(actual[:, :10], logical.reshape(80, 10))
                np.testing.assert_array_equal(actual[:, 10:], 0)
                self.assertEqual(len(engine.matvec_calls), 80)
                self.assertTrue(all(call['K'] == 128 and call['N'] == 10
                                    and call['stride_z'] == 64 for call in engine.matvec_calls))
                self.assertEqual([entry[2] for entry in engine.dma_reads if entry[0] == source.address], [800])
                self.assertEqual(len(engine.dma_writes), 1)
                self.assertIn('output', compiler.emitter._zero_padding)

    def test_other_tensor_shapes_keep_existing_lowering_without_dma(self):
        for source_shape, output_shape in (((1, 800), (1, 10, 80)),
                                          ((16, 50), (1, 800)),
                                          ((1, 960), (96, 10))):
            with self.subTest(source=source_shape, output=output_shape):
                compiler, engine, _, _, _ = self.prepare(
                    source_shape=source_shape, output_shape=output_shape)
                compiler.emit_view(0, SimpleNamespace(input=['source'], output=['output']))
                self.assertTrue(compiler.fallback)
                self.assertFalse(engine.dma_reads)
                self.assertFalse(engine.matvec_calls)

    def test_sram_boundaries_and_alignment_fall_back_before_dma(self):
        for attribute, value in (('_RESHAPE_INPUT', 64), ('_RESHAPE_INPUT', -128),
                                 ('_RESHAPE_OUTPUT', 128), ('_RESHAPE_OUTPUT', 0xFFFF80),
                                 ('_RESHAPE_MATRIX', 0x80001), ('_RESHAPE_MATRIX', 0xFFFF80)):
            with self.subTest(attribute=attribute, value=value):
                compiler, engine, _, _, _ = self.prepare()
                setattr(compiler, attribute, value)
                compiler.emit_view(439, SimpleNamespace(input=['source'], output=['output']))
                self.assertTrue(compiler.fallback)
                self.assertFalse(engine.dma_reads)
        compiler, engine, _, _, _ = self.prepare()
        with patch.object(udc, 'URAM_NEAR_FULL_SIZE', 65536):
            compiler.emit_view(439, SimpleNamespace(input=['source'], output=['output']))
        self.assertTrue(compiler.fallback)
        self.assertFalse(engine.dma_reads)


if __name__ == '__main__':
    unittest.main()
