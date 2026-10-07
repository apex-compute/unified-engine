"""Gemma3 model integration tests without loading weights or opening an FPGA."""

import builtins
import json
import unittest
from unittest.mock import mock_open, patch

import user_dma_core as core
import multi_engine_shard as shard

# The CLI installs its print filter at import time; keep test-runner output intact.
_print = builtins.print
try:
    from models.gemma3 import gemma3_test as gemma
finally:
    builtins.print = _print


class GemmaMulticoreLayout(unittest.TestCase):
    def setUp(self):
        for method in ("dma_read", "dma_write", "configure_clock_from_hardware"):
            owner = core if method == "configure_clock_from_hardware" else core.UnifiedEngine
            guard = patch.object(owner, method,
                                 side_effect=AssertionError("unexpected hardware access"))
            guard.start()
            self.addCleanup(guard.stop)

    def engine(self, cores=2, gib=4, active=2):
        with patch.multiple(core, ANDROMEDA_CORE_COUNT=cores,
                            AVAILABLE_DRAM_SIZE_GB=gib, CLOCK_CYCLE_TIME_NS=4.0), \
                patch.object(gemma.Gemma3_UnifiedEngine, "weight_init"), \
                patch.object(gemma.Gemma3_UnifiedEngine, "tensor_init"):
            return gemma.Gemma3_UnifiedEngine(multi_core=active)

    def test_kintex_shards_use_both_ddr_controllers_and_clear_model(self):
        engine = self.engine()
        self.assertEqual(engine._board_windows, [(0, 512 << 20), (3 << 30, 512 << 20)])
        self.assertEqual(engine._params_dram_base, 1 << 30)
        self.assertEqual(engine._tensor_dram_base, 0x70000000)
        self.assertEqual(engine._program_dram_base, 0x90000000)
        model_end = engine._model_base + gemma.MULTI_CORE_MODEL_SPAN
        self.assertEqual(model_end, 3 << 30)
        for base, size in engine._board_windows:
            self.assertTrue(base + size <= engine._model_base or base >= model_end)
        self.assertEqual({base // (2 << 30) for base, _ in engine._board_windows}, {0, 1})

    def test_alveo_uses_board_map_even_with_fewer_than_eight_active_engines(self):
        for cores, gib, counts in ((8, 8, (2, 4, 8)), (12, 16, (2, 4, 8, 12)),
                                   (12, 8, (2, 4, 8, 12))):
            for active in counts:
                with self.subTest(cores=cores, gib=gib, active=active):
                    engine = self.engine(cores, gib, active)
                    self.assertTrue(engine._use_multicore_dram_layout)
                    self.assertEqual(len(engine._board_windows), active)
                    self.assertEqual(engine._model_base, 6 << 30)
                    # The scheduler takes explicit hardware-order windows.
                    arena = shard.PrivateArena(active, windows=engine._board_windows)
                    self.assertIsNotNone(arena)
        u50 = self.engine(8, 8, 4)
        self.assertEqual([base for base, _ in u50._board_windows],
                         shard.alveo_core_bases(4))

    def test_single_engine_retains_original_addresses_and_cache_name(self):
        engine = self.engine(active=1)
        self.assertEqual(engine._params_dram_base, core.DRAM_START_ADDR)
        self.assertEqual(engine._tensor_dram_base, core.DRAM_ACTIVATION_ADDR)
        self.assertEqual(engine._program_dram_base, core.DRAM_INSTRUCTION_ADDR)
        self.assertIsNone(engine._board_windows)
        self.assertEqual(engine._mc_tag(), "")

    def test_too_many_engines_fail_before_weights_are_loaded(self):
        with patch.multiple(core, ANDROMEDA_CORE_COUNT=2, AVAILABLE_DRAM_SIZE_GB=4), \
                patch.object(gemma.Gemma3_UnifiedEngine, "weight_init") as load_weights:
            with self.assertRaisesRegex(ValueError, "exceeds.*2 engines"):
                gemma.Gemma3_UnifiedEngine(multi_core=4)
            load_weights.assert_not_called()

    def test_cache_key_changes_with_model_base_window_base_and_engine_order(self):
        engine = self.engine()
        original = engine._mc_tag()
        self.assertNotEqual(original, "_mc2")
        self.assertNotEqual(original, "_mc2_mcmap")
        self.assertEqual(original, self.engine()._mc_tag())
        engine._model_base += 512 << 20
        self.assertNotEqual(engine._mc_tag(), original)
        engine = self.engine()
        engine._board_windows = list(reversed(engine._board_windows))
        self.assertNotEqual(engine._mc_tag(), original)
        engine._board_windows = [(0, 512 << 20), (2 << 30, 512 << 20)]
        self.assertNotEqual(engine._mc_tag(), original)
        self.assertNotEqual(self.engine(8, 8, 2)._mc_tag(), original)

    def test_allocators_fail_before_crossing_region_or_advancing_cursor(self):
        engine = self.engine()
        cases = ((engine.allocate_params_dram, "_next_params_dram_addr",
                  engine._tensor_dram_base),
                 (engine.allocate_tensor_dram, "_tensor_dram_addr",
                  engine._program_dram_base),
                 (engine.allocate_program_dram, "_next_program_dram_addr",
                  engine._model_base + gemma.MULTI_CORE_MODEL_SPAN))
        for allocate, cursor, end in cases:
            with self.subTest(allocator=allocate.__name__):
                setattr(engine, cursor, end - 64)
                self.assertEqual(allocate(64), end - 64)
                with self.assertRaises(MemoryError):
                    allocate(1)
                self.assertEqual(getattr(engine, cursor), end)

    def test_program_preamble_is_checked_before_dma_including_padding(self):
        engine = self.engine()
        engine.capture_count = 1
        end = engine._model_base + gemma.MULTI_CORE_MODEL_SPAN
        with patch.object(core.UnifiedEngine, "write_captured_instructions_to_dram") as write:
            with self.assertRaises(MemoryError):
                engine.write_captured_instructions_to_dram(end - 32)
            write.assert_not_called()
            engine.write_captured_instructions_to_dram(end - 64)
            write.assert_called_once_with(end - 64)

    def test_binary_reuse_cannot_skip_rebuilding_worker_programs(self):
        single = self.engine(active=1)
        multi = self.engine(active=2)
        cached_geometry = json.dumps({"vector_size": gemma.UE_VECTOR_SIZE,
                                      "axi_width": core.UE_AXI_DATA_WIDTH_BITS})
        with patch.object(gemma.os.path, "exists", return_value=True), \
                patch("builtins.open", mock_open(read_data=cached_geometry)), \
                patch.object(single, "start_capture") as single_capture, \
                patch.object(multi, "start_capture",
                             side_effect=RuntimeError("compile started")) as multi_capture:
            single.compile_gemma3(bin_reuse=True)
            single_capture.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "compile started"):
                multi.compile_gemma3(bin_reuse=True)
            multi_capture.assert_called_once()

    def test_shipped_model_params_and_tensor_allocations_fit_rebased_regions(self):
        engine = self.engine()
        engine.allocate_params_dram(engine.LAYER_SIZE * engine.weight_defs["LAYER_WEIGHT_SIZE"])
        for region in engine._cfg["layers"]["non_layer"]:
            key = region["key"]
            if key not in ("ROPE_LOCAL", "ROPE_GLOBAL"):
                engine.allocate_params_dram(engine.weight_defs[key + "_SIZE"])
        # RoPE allocation follows the model's real loader, with DMA stubbed.
        with patch.object(engine, "dma_to_accelerator_memory"), \
                patch.object(engine, "dma_write"), \
                patch.object(engine, "_zero_kv_cache"), \
                patch.object(engine, "_zero_flash_attention_inputs"):
            engine._load_rope_host()
            engine.tensor_init()
        self.assertLess(engine.get_params_dram_addr(), engine._tensor_dram_base)
        self.assertLess(engine.get_tensor_dram_addr(), engine._program_dram_base)


if __name__ == "__main__":
    unittest.main()
