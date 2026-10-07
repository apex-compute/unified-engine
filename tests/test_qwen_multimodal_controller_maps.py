"""Existing eight-engine Qwen model maps and cache contracts, without FPGA I/O."""

import builtins
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import user_dma_core as core
import multi_engine_shard as shard

ROOT = Path(__file__).resolve().parents[1]


def load_model(folder, filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "models" / folder / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    original_print = builtins.print
    try:
        spec.loader.exec_module(module)
    finally:
        builtins.print = original_print
    return module


VL = load_model("qwen2.5_vl_3b", "qwen2.5_vl_3b_test.py", "qwen_vl_controller_tests")
OMNI = load_model("qwen2.5_omni_7b", "qwen2.5_omni_7b_test.py", "qwen_omni_controller_tests")


class MultimodalControllerMaps(unittest.TestCase):
    def setUp(self):
        for owner, method in ((core, "configure_clock_from_hardware"),
                              (core.UnifiedEngine, "dma_read"),
                              (core.UnifiedEngine, "dma_write"),
                              (core.UnifiedEngine, "read_reg32"),
                              (core.UnifiedEngine, "write_reg32")):
            guard = patch.object(owner, method, side_effect=AssertionError("unexpected FPGA I/O"))
            guard.start()
            self.addCleanup(guard.stop)

    def engine(self, module, cores=8, gib=8, active=8):
        cls = module.Qwen25VL_UnifiedEngine if module is VL else module.Qwen25OmniUnifiedEngine
        with patch.multiple(core, ANDROMEDA_CORE_COUNT=cores, AVAILABLE_DRAM_SIZE_GB=gib,
                            CLOCK_CYCLE_TIME_NS=3.0), contextlib.redirect_stdout(io.StringIO()):
            return cls(multi_core=active)

    def test_vl_u50_keeps_engine_saxi_order_for_all_active_counts(self):
        for active in (2, 4, 8):
            with self.subTest(active=active):
                engine = self.engine(VL, active=active)
                self.assertEqual(engine._board_windows,
                                 [(base, 512 << 20) for base in shard.alveo_core_bases(active)])
                self.assertEqual(engine.PARAMS_BASE, 6 << 30)
                self.assertEqual(engine.DRAM_END, 8 << 30)
                self.assertEqual(engine.mc_arena.windows, engine._board_windows)
                self.assertEqual(engine.mc_arena.weight_bytes(), 496 << 20)
                for index in range(1, active):
                    self.assertGreaterEqual(engine.mc_arena.isa_base(index), engine.WORKER_ISA_BASE)
                    self.assertLessEqual(engine.mc_arena.isa_limit(index), engine.DRAM_END)

    def test_vl_u55c_windows_avoid_shared_model_for_8_and_16_gib_images(self):
        for gib in (8, 16):
            for active in (4, 8, 12):
                with self.subTest(gib=gib, active=active):
                    engine = self.engine(VL, cores=12, gib=gib, active=active)
                    self.assertEqual(len(engine._board_windows), active)
                    for base, size in engine._board_windows:
                        self.assertTrue(base + size <= 6 << 30 or base >= 8 << 30)
                        self.assertLessEqual(base + size, gib << 30)
                    size = (512 << 20) if gib == 8 and active > 6 else (1 << 30)
                    self.assertEqual({span for _, span in engine._board_windows}, {size})

    def test_vl_does_not_fallback_on_unknown_hardware_or_policy_failure(self):
        with self.assertRaisesRegex(ValueError, "no controller-aware model map"):
            self.engine(VL, cores=4, gib=8, active=4)
        with patch.object(VL, "model_multicore_layout", side_effect=ValueError("bad controller map")), \
                self.assertRaisesRegex(ValueError, "bad controller map"):
            self.engine(VL)
        with self.assertRaisesRegex(ValueError, "needs the 8 GB DRAM map"):
            self.engine(VL, cores=2, gib=4, active=2)

    def test_vl_single_core_retains_original_map_without_private_arena(self):
        engine = self.engine(VL, active=1)
        self.assertEqual(engine.PARAMS_BASE, 2 << 30)
        self.assertEqual(engine.DRAM_END, 4 << 30)
        self.assertIsNone(engine.mc_arena)

    def test_omni_preserves_whole_device_capacity_and_guarded_windows(self):
        for cores, gib in ((8, 8), (12, 8), (12, 16)):
            with self.subTest(cores=cores, gib=gib), \
                    patch.dict(os.environ, {}, clear=True):
                engine = self.engine(OMNI, cores=cores, gib=gib)
                with patch.multiple(core, ANDROMEDA_CORE_COUNT=cores,
                                    AVAILABLE_DRAM_SIZE_GB=gib):
                    expected = shard.tiled_window_bases(8, 1 << 30, "Qwen Omni test")
                self.assertEqual(engine.mc_arena.windows, [(base, 1 << 30) for base in expected])
                self.assertEqual(engine.mc_arena.arena_bytes, 8 << 30)
                self.assertGreater(engine.mc_arena.weight_bytes(), 512 << 20)
                for index, base in enumerate(expected):
                    region = engine.mc_arena.regions[index]
                    self.assertEqual(engine.mc_arena.isa_limit(index), base + (1 << 30))
                    self.assertEqual(region.isa_base - region.tensor_base - engine.mc_arena.tensor_bytes,
                                     OMNI.OMNI_ISA_GUARD_BYTES)

    def test_omni_rejects_forced_capacity_before_any_allocation(self):
        with patch.dict(os.environ, {"UE_FORCE_DRAM_SIZE_GB": "8"}), \
                self.assertRaisesRegex(ValueError, "must be backed by DRAM"):
            self.engine(OMNI)

    def test_omni_program_cache_rejects_changed_controller_order(self):
        with patch.dict(os.environ, {}, clear=True):
            engine = self.engine(OMNI)
        manifest = dict(schema_version=1, config_sha256="config", generation_id="generation",
                        params_size=64, model_revision="revision", generation_trailer_bytes=0,
                        regions={})
        with tempfile.TemporaryDirectory() as tmp:
            engine.script_dir = tmp
            engine._cfg["paths"]["params"] = "params.bin"
            engine.fpga_build = 0xE6703022
            params = Path(tmp) / "params.bin"
            params.with_suffix(".json").write_text(json.dumps(manifest))
            with patch.multiple(core, HW_INFO_RAW=0x89074D55, ANDROMEDA_CORE_COUNT=8,
                                UE_AXI_DATA_WIDTH_BITS=256), \
                    patch.object(OMNI, "_PROGRAM_CODE_FILES", []):
                engine.configure_runtime_artifacts(str(params), tmp)
                first_bundle = engine._program_bundle
                first_bundle.store_stage("decode", [dict(engine_index=0,
                    dram_base=engine.ISA_BASE, bytes=bytes(64))])
                first_bases = first_bundle.identity["hardware"]["dram_map"]["worker_isa_bases"]
                with patch.object(engine.mc_arena, "isa_base", side_effect=reversed(first_bases)):
                    engine.configure_runtime_artifacts(str(params), tmp)
                self.assertNotEqual(first_bundle.identity, engine._program_bundle.identity)
                with self.assertRaisesRegex(OMNI._program_mod.ProgramBundleError, "identity"):
                    engine._program_bundle.load()


if __name__ == "__main__":
    unittest.main()
