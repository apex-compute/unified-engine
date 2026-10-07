"""Gemma4 E2B controller placement and packaged-cache isolation, without DMA."""

import builtins
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import user_dma_core as core
import multi_engine_shard as shard


MODEL_DIR = Path(__file__).resolve().parents[1] / "models/gemma4_e2b"
_print = builtins.print
try:
    with patch.object(core, "configure_clock_from_hardware",
                      side_effect=AssertionError("unexpected hardware access")), \
            patch.object(core.UnifiedEngine, "dma_read",
                         side_effect=AssertionError("unexpected hardware access")), \
            patch.object(core.UnifiedEngine, "dma_write",
                         side_effect=AssertionError("unexpected hardware access")), \
            patch.object(sys, "path", [str(MODEL_DIR), *sys.path]):
        spec = importlib.util.spec_from_file_location("_gemma4_layout_subject", MODEL_DIR / "gemma4_e2b_test.py")
        gemma = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gemma)
finally:
    builtins.print = _print


class Gemma4MulticoreLayout(unittest.TestCase):
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

    def test_u50_full_eight_engine_shared_pool_map_is_preserved(self):
        engine = self.engine()
        self.assertTrue(engine._tiled_map)
        self.assertEqual(engine._tile_bases, [i << 30 for i in range(8)])
        self.assertEqual(engine.mc_arena.stride, 1 << 30)
        self.assertEqual(engine.mc_arena.tensor_bytes, 32 << 20)
        self.assertEqual(engine.TENSOR_LIMIT - engine._tensor_dram_base, 480 << 20)
        self.assertEqual(engine.VISION_ISA_BASE, engine.mc_arena.isa_base(0))
        self.assertLess(engine.LM_ISA_BASE, engine.MASTER_ISA_LIMIT)
        self.assertLessEqual(engine.mc_arena.usage()[0], 256 << 20)

    def test_u55c_tiled_windows_follow_native_stack_controller_map(self):
        engine = self.engine(12, 16, 12)
        self.assertTrue(engine._tiled_map)
        self.assertEqual(engine._tile_bases, shard.alveo_u55c_core_bases(12, 16))
        self.assertEqual([r.base for r in engine.mc_arena.regions], engine._tile_bases)

    def test_non_tiled_windows_follow_board_controllers_and_reserve_model(self):
        for cores, gib, active in ((8, 8, 4), (12, 16, 4), (12, 8, 12)):
            with self.subTest(cores=cores, gib=gib, active=active):
                engine = self.engine(cores, gib, active)
                self.assertFalse(engine._tiled_map)
                with patch.multiple(core, ANDROMEDA_CORE_COUNT=cores, AVAILABLE_DRAM_SIZE_GB=gib):
                    expected, model_base = shard.model_multicore_layout(active)
                self.assertEqual(engine._board_windows, expected)
                self.assertEqual(engine._params_dram_base, model_base)
                self.assertEqual(engine.DRAM_END, model_base + (2 << 30))
                for base, size in expected:
                    self.assertTrue(base + size <= model_base or base >= engine.DRAM_END)
        self.assertEqual(self.engine(active=4)._board_windows,
                         [(base, 512 << 20) for base in shard.alveo_core_bases(4)])

    def test_layout_identity_distinguishes_same_count_on_different_boards(self):
        u50 = self.engine()
        u55 = self.engine(12, 16, 8)
        self.assertNotEqual(u50.dram_layout, u55.dram_layout)
        self.assertEqual(u50.dram_layout, self.engine().dram_layout)
        self.assertTrue(u50.dram_layout.startswith("tile8_"))
        self.assertNotEqual(self.engine(active=2).dram_layout,
                            self.engine(active=4).dram_layout)
        self.assertEqual(self.engine(active=1).dram_layout, "legacy")

    def test_stale_program_sections_are_rejected_before_dma(self):
        engine = self.engine()
        stale = {"dram_layout": "tile8", "dram_base": "0x1", "file_offset": 0, "size": 4}
        with patch.object(engine, "_read_program_sections", return_value=({"lm": stale}, b"test")):
            self.assertEqual(engine._get_program_section("lm"), (None, None))
            with self.assertRaises(FileNotFoundError):
                engine._load_program_section("lm")
        current = {**stale, "dram_layout": engine.dram_layout}
        with patch.object(engine, "_read_program_sections", return_value=({"lm": current}, b"test")):
            self.assertEqual(engine._get_program_section("lm"), (current, b"test"))

    def test_every_stored_section_includes_the_actual_layout(self):
        engine = self.engine()
        with tempfile.TemporaryDirectory() as directory:
            engine.script_dir = directory
            # Audio metadata did not previously include dram_layout. The common
            # writer must stamp it for every stage, including external callers.
            engine._store_program_section("audio", engine.LM_ISA_BASE, b"test", {})
            metadata, blob = engine._get_program_section("audio")
            self.assertEqual(metadata["dram_layout"], engine.dram_layout)
            self.assertEqual(blob, b"test")
            self.assertEqual(Path(engine._program_image_paths()[0]).name, "programs.bin")


if __name__ == "__main__":
    unittest.main()
