"""Gemma4 E4B on the shared Gemma4 (E2B) implementation: variant wiring and layout, without DMA.

gemma4_e4b_test.py is a launcher that runs models/gemma4_e2b with E4B's config
and artifacts. These checks cover what differs from E2B -- the head counts
(8 query heads over 2 KV heads), 42 layers, the per-engine private-weight
reserve derived from the larger weights -- on the same tiled U50 map.
"""

import builtins
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import user_dma_core as core


MODEL_DIR = Path(__file__).resolve().parents[1] / "models/gemma4_e4b"
_print = builtins.print
try:
    with patch.object(core, "configure_clock_from_hardware",
                      side_effect=AssertionError("unexpected hardware access")), \
            patch.object(core.UnifiedEngine, "dma_read",
                         side_effect=AssertionError("unexpected hardware access")), \
            patch.object(core.UnifiedEngine, "dma_write",
                         side_effect=AssertionError("unexpected hardware access")):
        spec = importlib.util.spec_from_file_location("_gemma4_e4b_subject",
                                                      MODEL_DIR / "gemma4_e4b_test.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        gemma = launcher._gemma4
finally:
    builtins.print = _print


class Gemma4E4BVariant(unittest.TestCase):
    def setUp(self):
        self.stack = self.enterContext(ExitStack())
        for name in ("dma_read", "dma_write", "read_reg32", "write_reg32",
                     "user_read_reg32", "user_write_reg32"):
            self.stack.enter_context(patch.object(core.UnifiedEngine, name,
                side_effect=AssertionError(f"unexpected hardware access: {name}")))
        self.stack.enter_context(patch.object(core, "configure_clock_from_hardware",
            side_effect=AssertionError("unexpected hardware access")))

    def engine(self, cores=8, gib=8, active=8):
        with patch.multiple(core, ANDROMEDA_CORE_COUNT=cores,
                            AVAILABLE_DRAM_SIZE_GB=gib, CLOCK_CYCLE_TIME_NS=4.0), \
                patch.object(gemma.Gemma4_UnifiedEngine, "weight_init"), \
                patch.object(gemma.Gemma4_UnifiedEngine, "tensor_init"), \
                patch.object(core.UnifiedEngine, "dma_to_accelerator_memory"), \
                patch("builtins.print"):
            return gemma.Gemma4_UnifiedEngine(multi_core=active)

    def test_variant_points_at_the_e4b_directory(self):
        self.assertEqual(gemma.MODEL_NAME, "gemma4_e4b")
        self.assertEqual(Path(gemma.SCRIPT_DIR), MODEL_DIR)
        engine = self.engine()
        self.assertEqual(Path(engine.script_dir), MODEL_DIR)
        self.assertEqual(engine._cfg["paths"]["hf_model_repo"], "google/gemma-4-E4B-it")
        self.assertEqual(Path(engine._program_image_paths()[0]).parent.name, "gemma4_e4b_bin")

    def test_head_counts_and_layer_dims(self):
        engine = self.engine()
        self.assertEqual((engine.num_q_heads, engine.num_kv_heads, engine.group_size), (8, 2, 4))
        self.assertEqual(engine.LAYER_SIZE, 42)
        self.assertEqual(engine.vector_length, 2560)
        self.assertEqual(engine.MAX_CONTEXT_SIZE, 8192)
        # Sliding layer 0: 8 x 256 query, 2 x 256 KV; full layer 5: 8 x 512, 2 x 512.
        self.assertEqual(engine._get_layer_attention_dims(0), (256, 2048, 512))
        self.assertEqual(engine._get_layer_attention_dims(5), (512, 4096, 1024))
        self.assertEqual(engine.k_size, 512 * 2 * 2)   # widest KV row, bytes

    def test_u50_tiled_map_and_private_reserve(self):
        engine = self.engine()
        self.assertTrue(engine._tiled_map)
        self.assertEqual(engine._tile_bases, [i << 30 for i in range(8)])
        self.assertEqual(engine.mc_arena.tensor_bytes, 64 << 20)
        self.assertEqual(engine.TENSOR_LIMIT - engine._tensor_dram_base, 480 << 20)
        # E4B's private slices need ~337 MiB/engine, more than E2B's 256 MiB floor.
        bound = gemma.Gemma4_UnifiedEngine._private_weight_upper_bound(None, 8)
        self.assertGreater(bound, 256 << 20)
        self.assertEqual(engine.mc_arena._private_reserve, [bound] * 8)


if __name__ == "__main__":
    unittest.main()
