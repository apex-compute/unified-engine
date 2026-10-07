"""Output checks must reject matching garbage without accessing an FPGA."""
import unittest
from unittest.mock import Mock, patch

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


if __name__ == "__main__":
    unittest.main()
