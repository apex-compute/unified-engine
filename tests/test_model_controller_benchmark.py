"""Output checks must reject matching garbage without accessing an FPGA."""
import unittest
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import user_dma_core as core
with patch.object(core, "configure_clock_from_hardware", side_effect=AssertionError("hardware access")):
    import model_controller_benchmark as benchmark


class TokenOutputValidation(unittest.TestCase):
    def test_matching_padding_is_not_a_success(self):
        output = dict(token_ids=[0, 0], decoded_text_without_special_tokens="", pad_token_id=0)
        self.assertFalse(benchmark.valid_token_output(output))
        output["decoded_text_without_special_tokens"] = "<pad><pad>"
        self.assertFalse(benchmark.valid_token_output(output))

    def test_special_token_only_or_empty_decode_is_rejected(self):
        for ids, text in (([], ""), ([1, 106], ""), ([42], " \n")):
            with self.subTest(ids=ids):
                self.assertFalse(benchmark.valid_token_output(dict(
                    token_ids=ids, decoded_text_without_special_tokens=text, pad_token_id=0)))

    def test_meaningful_generation_and_stop_token_pass(self):
        self.assertTrue(benchmark.valid_token_output(dict(
            token_ids=[87, 284, 17, 106], decoded_text_without_special_tokens="x = 2", pad_token_id=0)))

    def test_runner_decodes_without_special_tokens(self):
        engine = Mock()
        engine.tokenizer.pad_token_id = 0
        engine.tokenizer.decode.return_value = "x = 2"
        output = benchmark.token_output(engine, [87, 284, 17, 106])
        engine.tokenizer.decode.assert_called_once_with([87, 284, 17, 106], skip_special_tokens=True)
        self.assertTrue(benchmark.valid_token_output(output))

    def test_e4b_short_initialization_fails_before_loading_weights(self):
        module = Mock()
        raw = Mock()
        raw.dma_write.return_value = 1
        with patch.object(core, "AVAILABLE_DRAM_SIZE_GB", 8), \
                patch.object(core, "UnifiedEngine", return_value=raw), \
                self.assertRaisesRegex(IOError, "DRAM initialization"):
            benchmark.run_e4b(module, 8, "x+3=5", 32)
        module.Gemma4_UnifiedEngine.assert_not_called()


class Qwen2BRunnerValidation(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config_path = self.root / "config.json"
        self.weights_path = self.root / "bin" / "params.bin"
        self.config_path.write_text(json.dumps({
            "paths": {"weights_bin": "bin/params.bin"},
            "model": {"max_context_size": 128, "linear_attn_layer_indices": [0, 1]},
        }))
        self.tokenizer = Mock()
        self.tokenizer.pad_token_id = 0
        self.tokenizer.decode.return_value = "x = 2"
        self.engine = Mock()
        self.engine.tokenizer = self.tokenizer
        self.engine.max_context = 128
        self.engine._params_dram_base = 1 << 30
        self.engine._decode_windows = None
        self.engine._decode_step_us = [11000.0, 19000.0]
        self.engine.controller_decoder = None
        self.engine.compile_decoder.return_value = ("decoder.bin", [64], [100])
        self.engine.run_decoder.return_value = {
            "token_ids": [10, 20, 106], "generated_text": "x = 2",
        }
        self.module = SimpleNamespace(
            CONFIG_PATH=self.config_path,
            _ensure_hf_model=Mock(return_value=str(self.root / "hf")),
            _load_hf_model=Mock(return_value=("hf", "text-model")),
            _extract_all_weights=Mock(return_value={"fresh": "weights"}),
            Qwen3_5_2b_UnifiedEngine=Mock(return_value=self.engine),
            _tokenize_with_chat_template=Mock(return_value=([1, 2, 3], "prompt")),
            prefill_via_decode=Mock(return_value=10),
        )
        fake_transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=self.tokenizer)))
        for guard in (
                patch.dict(sys.modules, {"transformers": fake_transformers}),
                patch.object(core, "configure_clock_from_hardware",
                             side_effect=AssertionError("unexpected hardware access")),
                patch.object(core.UnifiedEngine, "dma_read",
                             side_effect=AssertionError("unexpected hardware access")),
                patch.object(core.UnifiedEngine, "dma_write",
                             side_effect=AssertionError("unexpected hardware access"))):
            guard.start()
            self.addCleanup(guard.stop)

    def cached_run(self, engines=1, max_tokens=3):
        self.weights_path.parent.mkdir(exist_ok=True)
        self.weights_path.touch()
        with patch.object(torch, "load", return_value={"cached": "weights"}) as load:
            result = benchmark.run_qwen_2b(self.module, engines, "x+3=5", max_tokens)
        load.assert_called_once_with(self.weights_path, weights_only=False, map_location="cpu")
        return result

    def test_cached_weights_reupload_into_a_fresh_engine_and_program(self):
        result = self.cached_run(engines=2)
        self.module._load_hf_model.assert_not_called()
        self.module._extract_all_weights.assert_not_called()
        self.module.Qwen3_5_2b_UnifiedEngine.assert_called_once_with(
            device="cpu", init_unified_engine=False, multi_core=2)
        self.engine.prepare_inference.assert_called_once_with(
            {"cached": "weights"}, max_context=128)
        self.engine.compile_decoder.assert_called_once_with()
        self.engine.load_instructions.assert_called_once_with("decoder.bin")
        self.module.prefill_via_decode.assert_called_once_with(self.engine, [1, 2, 3])
        self.assertFalse(self.engine.fpga_penalty)
        # The first generated token comes from prefill. Only the later two
        # token executions contribute to this runner's hardware decode timing.
        self.assertEqual(result["token_ids"], [10, 20, 106])
        self.assertEqual(result["timed_decode_steps"], 2)
        self.assertEqual(result["decode_first_hw_ms"], 11.0)
        self.assertEqual(result["decode_avg_hw_ms"], 15.0)

    def test_missing_weight_cache_extracts_and_uploads_fresh_weights(self):
        with patch.object(torch, "load") as load, patch.object(torch, "save") as save:
            benchmark.run_qwen_2b(self.module, 1, "x+3=5", 3)
        load.assert_not_called()
        self.module._extract_all_weights.assert_called_once_with("text-model", {0, 1})
        save.assert_called_once_with({"fresh": "weights"}, self.weights_path)
        self.engine.prepare_inference.assert_called_once_with({"fresh": "weights"}, max_context=128)

    def test_no_decode_timing_cannot_report_a_speedup(self):
        self.engine._decode_step_us = []
        with self.assertRaisesRegex(RuntimeError, "no decode timings"):
            self.cached_run()

    def test_token_timing_mismatch_cannot_report_a_speedup(self):
        self.engine._decode_step_us = [11000.0]
        with self.assertRaisesRegex(RuntimeError, "tokens and decode timings do not match"):
            self.cached_run()

    def test_nonpositive_or_nonfinite_hardware_timing_is_rejected(self):
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.engine._decode_step_us = [11000.0, value]
                with self.assertRaisesRegex(RuntimeError, "invalid hardware timing"):
                    self.cached_run()

    def test_context_overrun_fails_before_prefill_or_decode(self):
        with self.assertRaisesRegex(ValueError, "exceeds.*context"):
            self.cached_run(max_tokens=126)
        self.module.prefill_via_decode.assert_not_called()
        self.engine.run_decoder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
