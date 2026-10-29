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

    # Engine clock per board: it is what tells the interleaved kintex7 (213 MHz
    # release image) from kintex7_systolic (150 MHz), see dram_interleaved.
    CLOCK_HZ = {2: 213.2e6, 4: 300e6, 8: 333.3e6, 12: 290e6}

    def engine(self, module, cores=8, gib=8, active=8, clock_hz=None, **kwargs):
        cls = module.Qwen25VL_UnifiedEngine if module is VL else module.Qwen25OmniUnifiedEngine
        with patch.multiple(core, ANDROMEDA_CORE_COUNT=cores, AVAILABLE_DRAM_SIZE_GB=gib,
                            CLOCK_CYCLE_TIME_NS=3.0,
                            CORE_CLOCK_FREQ_HZ=clock_hz or self.CLOCK_HZ[cores]), \
                contextlib.redirect_stdout(io.StringIO()):
            return cls(multi_core=active, **kwargs)

    def assert_vl_consecutive(self, engine, active):
        """Sequential 512 MiB windows, the 2 GiB model map directly above them."""
        self.assertEqual(engine._board_windows,
                         [(i << 29, 512 << 20) for i in range(active)])
        self.assertEqual(engine.PARAMS_BASE, active << 29)
        self.assertEqual(engine.DRAM_END, engine.PARAMS_BASE + (2 << 30))
        self.assertEqual(engine.mc_arena.windows, engine._board_windows)
        self.assertEqual(engine.mc_arena.weight_bytes(), 496 << 20)
        for index in range(1, active):
            self.assertGreaterEqual(engine.mc_arena.isa_base(index), engine.WORKER_ISA_BASE)
            self.assertLessEqual(engine.mc_arena.isa_limit(index), engine.DRAM_END)

    def test_vl_u50_model_map_follows_the_sequential_windows(self):
        for active in (2, 4, 8):
            with self.subTest(active=active):
                self.assert_vl_consecutive(self.engine(VL, active=active), active)

    def test_vl_kintex7_two_engines_fit_4_gib(self):
        # 2 x 512 MiB windows + the 2 GiB model map = 3 GiB of the 4 GiB DDR3.
        engine = self.engine(VL, cores=2, gib=4, active=2)
        self.assert_vl_consecutive(engine, 2)
        self.assertEqual(engine.DRAM_END, 3 << 30)

    def test_vl_u55c_windows_avoid_shared_model_for_8_and_16_gib_images(self):
        for gib in (8, 16):
            for active in (4, 8, 12):
                with self.subTest(gib=gib, active=active):
                    engine = self.engine(VL, cores=12, gib=gib, active=active)
                    self.assertEqual(len(engine._board_windows), active)
                    for base, size in engine._board_windows:
                        self.assertTrue(base + size <= engine.PARAMS_BASE
                                        or base >= engine.DRAM_END)
                        self.assertLessEqual(base + size, gib << 30)
                    size = (512 << 20) if gib == 8 and active > 6 else (1 << 30)
                    self.assertEqual({span for _, span in engine._board_windows}, {size})

    def test_vl_does_not_fallback_on_unknown_hardware_or_policy_failure(self):
        with self.assertRaisesRegex(ValueError, "no controller-aware model map"):
            self.engine(VL, cores=4, gib=8, active=4)
        with patch.object(VL, "model_multicore_layout", side_effect=ValueError("bad controller map")), \
                self.assertRaisesRegex(ValueError, "bad controller map"):
            self.engine(VL)
        # kintex7_systolic places windows per DDR3 channel and keeps the 8 GB rule.
        with self.assertRaisesRegex(ValueError, "needs the 8 GB DRAM map"):
            self.engine(VL, cores=2, gib=4, active=2, clock_hz=150e6)

    def test_vl_single_core_retains_original_map_without_private_arena(self):
        engine = self.engine(VL, active=1)
        self.assertEqual(engine.PARAMS_BASE, 2 << 30)
        self.assertEqual(engine.DRAM_END, 4 << 30)
        self.assertIsNone(engine.mc_arena)

    def test_omni_windowed_preserves_whole_device_capacity_and_guarded_windows(self):
        for cores, gib in ((8, 8), (12, 8), (12, 16)):
            with self.subTest(cores=cores, gib=gib), \
                    patch.dict(os.environ, {}, clear=True):
                engine = self.engine(OMNI, cores=cores, gib=gib, layout="windowed")
                with patch.multiple(core, ANDROMEDA_CORE_COUNT=cores,
                                    AVAILABLE_DRAM_SIZE_GB=gib):
                    expected = shard.tiled_window_bases(8, 1 << 30, "Qwen Omni test")
                arena = engine.layout
                self.assertEqual(arena.kind, "windowed")
                self.assertEqual(arena.windows, [(base, 1 << 30) for base in expected])
                self.assertEqual(arena.arena_bytes, 8 << 30)
                self.assertGreater(arena.weight_bytes(), 512 << 20)
                for index, base in enumerate(expected):
                    region = arena.regions[index]
                    self.assertEqual(arena.isa_limit(index), base + (1 << 30))
                    self.assertEqual(region.isa_base - region.tensor_base - arena.tensor_bytes,
                                     OMNI._layout_mod.OMNI_ISA_GUARD_BYTES)

    def test_omni_flat_is_one_consecutive_map_inside_the_device(self):
        # The U50 map (flat on interleaved DRAM): private weights,
        # shared pools, tensor zones and ISA back to back from 0.
        with patch.dict(os.environ, {}, clear=True):
            plan = OMNI._layout_mod.plan_flat({"lm", "vision", "audio"})
            engine = self.engine(OMNI, layout="flat", flat_plan=plan)
        arena = engine.layout
        self.assertEqual(arena.kind, "flat")
        self.assertEqual(arena.regions[0].weight_base, 0)
        bases = [r.weight_base for r in arena.regions]
        self.assertEqual(bases, [i * arena.weight_bytes() for i in range(8)])
        self.assertLessEqual(arena.map_end, 8 << 30)
        self.assertGreaterEqual(arena.max_context, 4096)

    def test_omni_rejects_forced_capacity_before_any_allocation(self):
        with patch.dict(os.environ, {"UE_FORCE_DRAM_SIZE_GB": "8"}), \
                self.assertRaisesRegex(ValueError, "must be backed by DRAM"):
            self.engine(OMNI, layout="windowed")

    def test_omni_program_cache_rejects_changed_controller_order(self):
        with patch.dict(os.environ, {}, clear=True):
            engine = self.engine(OMNI, layout="windowed")
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
                reordered = first_bases[::-1]
                with patch.object(engine.layout, "isa_base", side_effect=lambda i: reordered[i]):
                    engine.configure_runtime_artifacts(str(params), tmp)
                self.assertNotEqual(first_bundle.identity, engine._program_bundle.identity)
                with self.assertRaisesRegex(OMNI._program_mod.ProgramBundleError, "identity"):
                    engine._program_bundle.load()


if __name__ == "__main__":
    unittest.main()
