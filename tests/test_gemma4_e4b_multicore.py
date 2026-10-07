"""E4B controller geometry, replay guards, and decode orchestration without DMA."""

import builtins
from contextlib import ExitStack, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import user_dma_core as core
from multi_engine_shard import alveo_core_bases


MODEL_DIR = Path(__file__).resolve().parents[1] / "models/gemma4_e4b"
_print = builtins.print
try:
    with patch.object(sys, "path", [str(MODEL_DIR), *sys.path]), \
            patch.object(core, "configure_clock_from_hardware",
                         side_effect=AssertionError("unexpected hardware probe")):
        spec = importlib.util.spec_from_file_location("_gemma4_e4b_multicore_subject",
                                                      MODEL_DIR / "gemma4_e4b_test.py")
        gemma = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gemma)
finally:
    builtins.print = _print


class Gemma4E4BMulticoreTests(unittest.TestCase):
    def setUp(self):
        self.stack = self.enterContext(ExitStack())
        for name in ("dma_read", "dma_write", "read_reg32", "write_reg32",
                     "user_read_reg32", "user_write_reg32"):
            self.stack.enter_context(patch.object(core.UnifiedEngine, name,
                side_effect=AssertionError(f"unexpected hardware access: {name}")))
        self.stack.enter_context(patch.object(core, "configure_clock_from_hardware",
            side_effect=AssertionError("unexpected hardware probe")))

    def engine(self, active=8, cores=8, gib=8):
        with patch.multiple(core, ANDROMEDA_CORE_COUNT=cores,
                            AVAILABLE_DRAM_SIZE_GB=gib, CLOCK_CYCLE_TIME_NS=4.0), \
                patch.object(gemma.Gemma4_UnifiedEngine, "weight_init"), \
                patch.object(gemma.Gemma4_UnifiedEngine, "tensor_init"), \
                patch.object(gemma.Gemma4_UnifiedEngine, "_preallocate_identity_matrix"), \
                redirect_stdout(io.StringIO()):
            return gemma.Gemma4_UnifiedEngine(multi_core=active)

    def test_u50_uses_upper_owned_controllers_without_rebasing_shared_model(self):
        engine = self.engine()
        self.assertEqual(engine._decode_windows,
                         [(base, 512 << 20) for base in alveo_core_bases(8, stack=1)])
        self.assertEqual(engine._params_dram_base, 0)
        self.assertEqual(engine._tensor_dram_base, 0xC0000000)
        self.assertEqual(engine._program_dram_base, 0xDA000000)
        self.assertEqual(engine._tensor_dram_top, 0xD4000000)
        self.assertTrue(engine.dram_layout.startswith("mc8_"))

    def test_u55_windows_stay_outside_complete_shared_image(self):
        for gib, active in ((16, 12), (16, 8), (8, 8)):
            with self.subTest(gib=gib, active=active):
                engine = self.engine(active, 12, gib)
                windows = engine._decode_windows
                self.assertEqual(len(windows), active)
                self.assertTrue(all((4 << 30) <= base < base + size <= gib << 30
                                    for base, size in windows))
                ordered = sorted(windows)
                self.assertTrue(all(base + size <= following[0]
                                    for (base, size), following in zip(ordered, ordered[1:])))
                if gib == 16:
                    self.assertEqual(len({base // (1 << 30) for base, _ in windows}), active)
                else:
                    self.assertEqual([base // (1 << 30) for base, _ in windows[:4]],
                                     [4, 5, 6, 7])
        self.assertNotEqual(self.engine().dram_layout, self.engine(8, 12, 16).dram_layout)

    def test_single_engine_keeps_legacy_layout_and_no_private_sharder(self):
        engine = self.engine(active=1, cores=2, gib=4)
        self.assertIsNone(engine._decode_windows)
        self.assertIsNone(engine._ensure_decode_sharder())
        self.assertEqual(engine.dram_layout, "legacy")
        self.assertEqual((engine._params_dram_base, engine._tensor_dram_base,
                          engine._program_dram_base), (0, 0xC0000000, 0xDA000000))

    def test_insufficient_private_capacity_rejects_before_loading_weights(self):
        with patch.multiple(core, ANDROMEDA_CORE_COUNT=8, AVAILABLE_DRAM_SIZE_GB=8,
                            CLOCK_CYCLE_TIME_NS=4.0), \
                patch.object(gemma.Gemma4_UnifiedEngine, "weight_init") as load, \
                redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(MemoryError, "decode shards need"):
                gemma.Gemma4_UnifiedEngine(multi_core=2)
            load.assert_not_called()
        with patch.multiple(core, ANDROMEDA_CORE_COUNT=2, AVAILABLE_DRAM_SIZE_GB=4):
            with self.assertRaisesRegex(ValueError, "shared model already occupies"):
                gemma._decode_private_windows(2)
        with patch.multiple(core, ANDROMEDA_CORE_COUNT=12, AVAILABLE_DRAM_SIZE_GB=8):
            with self.assertRaises(ValueError):
                gemma._decode_private_windows(12)

    def test_projection_shapes_follow_full_sliding_and_wide_layers(self):
        engine = self.engine()
        specs = {(name, layer): (K, N) for name, _, K, N, layer
                 in engine._decode_projection_specs()}
        self.assertEqual(len(specs), engine.LAYER_SIZE * 6)
        for layer in range(engine.LAYER_SIZE):
            dim = engine.head_dim if layer in engine._full_attention_layers else engine.head_dim_sliding
            self.assertEqual(specs["q", layer], (engine.vector_length, dim * engine.num_attention_heads))
            self.assertEqual(specs["k", layer], (engine.vector_length, dim * engine.num_key_value_heads))
            self.assertEqual(specs["o", layer], (dim * engine.num_attention_heads, engine.vector_length))
            width = engine.mlp_elements_wide if layer >= engine._double_wide_mlp_first else engine.mlp_elements
            self.assertEqual(specs["gate", layer], (engine.vector_length, width))
        self.assertFalse(any(name in ("down", "lm_head") for name, _ in specs))

    def test_projection_dispatch_preserves_primary_fallback_and_zero_flops(self):
        engine = self.engine()
        kwargs = dict(M=1, K=2560, N=512, A_DRAM_ADDR=64, B_DRAM_ADDR=128,
                      OUTPUT_DRAM_ADDR=192, SCALE_DRAM_ADDR=256)
        engine.quantized_matmat_core = Mock(return_value=123)
        engine._decode_sharder = Mock()
        engine._decode_sharder.projection.return_value = 0
        self.assertEqual(engine._decode_projection_core(**kwargs), 0)
        engine.quantized_matmat_core.assert_not_called()
        engine._decode_sharder.projection.return_value = None
        self.assertEqual(engine._decode_projection_core(**kwargs), 123)
        engine.quantized_matmat_core.assert_called_once_with(**kwargs)

    def manifest(self, engine):
        return {"dram_layout": engine.dram_layout, "multi_core": engine.multi_core,
                "instruction_base_addr": "0xda000000", "instruction_total_size": 64,
                "tensor_layout_sig": {"input": 123, "tensor_high_water": "ignored"}}

    def test_cached_layout_and_unrestored_workers_rejected_before_program_dma(self):
        engine = self.engine()
        engine._tensor_layout_signature = lambda: {"input": 123, "tensor_high_water": "other"}
        with tempfile.TemporaryDirectory() as directory:
            engine.script_dir = directory
            bins = Path(directory) / "gemma4_e4b_bin"
            bins.mkdir()
            (bins / "programs.bin").write_bytes(bytes(64))
            for changed, message in (({"dram_layout": "legacy"}, "different engine/memory layout"),
                                     ({}, "workers must be compiled")):
                with self.subTest(changed=changed):
                    (bins / "programs.json").write_text(json.dumps({**self.manifest(engine), **changed}))
                    with self.assertRaisesRegex(ValueError, message):
                        engine.load_instruction_bin()

    def test_manifest_tensor_and_program_extent_guards(self):
        engine = self.engine()
        engine._decode_sharder = SimpleNamespace(_worker_programs=[0x110000000])
        engine._tensor_layout_signature = lambda: {"input": 123, "tensor_high_water": "other"}
        manifest = self.manifest(engine)
        engine._validate_instruction_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "stale tensor layout"):
            engine._validate_instruction_manifest({**manifest, "tensor_layout_sig": {"input": 456}})
        with self.assertRaises(MemoryError):
            engine._validate_instruction_manifest({**manifest, "instruction_total_size": 1 << 30})
        engine._validate_program_bounds(0xFFFFFFC0, 64)
        for base, size in ((0xD4000000, 64), (0xFFFFFFC0, 65), (0xDA000000, -1)):
            with self.assertRaises(MemoryError):
                engine._validate_program_bounds(base, size)

    def test_standalone_loader_rejects_multicore_artifacts_before_dma(self):
        # The execute-only consumer sets HF offline flags and replaces print
        # during import. Neither side effect may escape into other tests.
        original_print = builtins.print
        try:
            with patch.dict("os.environ"), \
                    patch.object(sys, "path", [str(MODEL_DIR), *sys.path]):
                spec = importlib.util.spec_from_file_location(
                    "_gemma4_e4b_standalone_subject", MODEL_DIR / "gemma4_e4b_run_from_bin.py")
                standalone = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(standalone)
        finally:
            builtins.print = original_print
        engine = object.__new__(standalone.Gemma4_UnifiedEngine)
        engine.dma_write = Mock(side_effect=AssertionError("unexpected program DMA"))
        standalone._validate_single_engine_manifest({})
        standalone._validate_single_engine_manifest({"multi_core": 1, "dram_layout": "legacy"})
        with tempfile.TemporaryDirectory() as directory:
            engine.script_dir = directory
            bins = Path(directory) / "gemma4_e4b_bin"
            bins.mkdir()
            (bins / "programs.bin").write_bytes(bytes(64))
            for manifest in ({"multi_core": 8, "dram_layout": "mc8_test"},
                             {"multi_core": 8, "dram_layout": "legacy"},
                             {"multi_core": 1, "dram_layout": "mc8_test"}):
                with self.subTest(manifest=manifest):
                    (bins / "programs.json").write_text(json.dumps(manifest))
                    with self.assertRaisesRegex(ValueError, "single.engine"):
                        engine.load_instruction_bin()
                    engine.dma_write.assert_not_called()

    def test_decode_limit_and_worker_order_preserve_context_capacity(self):
        engine = self.engine()
        original_capacity = engine.MAX_CONTEXT_SIZE
        engine.seq_len = 10
        engine.EMBEDDING_ELEMENTS = 16
        for name in ("PENALTY_BIAS_DRAM", "LAYER0_INPUT_DRAM", "PER_LAYER_INPUTS_DRAM",
                     "LAYER0_FLASH_BIAS_FULL_DRAM", "LAYER0_FLASH_BIAS_SLIDING_DRAM"):
            setattr(engine, name, 64)
        engine.tokenizer = SimpleNamespace(decode=lambda *a, **k: "test")
        engine.dma_to_accelerator_memory = Mock()
        engine._isa_add_set_core = Mock()
        engine.get_embedding_for_tokens = Mock(return_value=torch.zeros(1, 1))
        engine._compute_per_layer_inputs = Mock(return_value=torch.zeros(1, 1, 1))
        engine.get_arg_max_index = Mock(side_effect=[2, 3, 4])
        events = []
        engine._decode_sharder = SimpleNamespace(start=lambda: events.append("start"),
                                                 wait=lambda: events.append("wait"))
        def execute(*args, **kwargs):
            events.append("primary")
            return 1000, 2
        engine.program_execute = execute
        with patch.dict("os.environ", {"GEMMA4_PENALTY": "0", "GEMMA4_NAN_TRIPWIRE": "0",
                                       "GEMMA4_DECODE_TRACE": "0"}), \
                redirect_stdout(io.StringIO()):
            result = engine.run_decoder([64], 0xDA000000, 9, max_new_tokens=3)
        self.assertEqual(result, (13, 3000, 6))
        self.assertEqual(engine._decoded_token_ids, [2, 3, 4])
        self.assertEqual(engine._decode_step_us, [1000, 1000, 1000])
        self.assertEqual(events, ["start", "primary", "wait"] * 3)
        self.assertEqual(engine.MAX_CONTEXT_SIZE, original_capacity)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            engine.run_decoder([64], 0xDA000000, 9, max_new_tokens=0)


if __name__ == "__main__":
    unittest.main()
