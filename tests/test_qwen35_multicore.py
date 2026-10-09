"""Qwen3.5 hybrid-state controller placement and paired decoder execution."""
import importlib.util
import io
import json
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import user_dma_core as core
import multi_engine_decode

MODEL = Path(__file__).resolve().parents[1] / "models/qwen3.5_2b/qwen3.5_2b_test.py"
spec = importlib.util.spec_from_file_location("_qwen35_multicore_subject", MODEL)
qwen = importlib.util.module_from_spec(spec)
_clock_before_import = core.CLOCK_CYCLE_TIME_NS
spec.loader.exec_module(qwen)
_clock_after_import = core.CLOCK_CYCLE_TIME_NS


class Qwen35MulticoreTests(unittest.TestCase):
    def setUp(self):
        stack = self.enterContext(ExitStack())
        for name in ("dma_read", "dma_write", "read_reg32", "write_reg32",
                     "user_read_reg32", "user_write_reg32"):
            stack.enter_context(patch.object(core.UnifiedEngine, name,
                side_effect=AssertionError("unexpected hardware access: " + name)))
        stack.enter_context(patch.object(core, "configure_clock_from_hardware",
                                        side_effect=AssertionError("unexpected hardware probe")))

    def engine(self, active=2, cores=2, gib=4):
        with patch.multiple(core, ANDROMEDA_CORE_COUNT=cores, AVAILABLE_DRAM_SIZE_GB=gib,
                            CLOCK_CYCLE_TIME_NS=5.0), redirect_stdout(io.StringIO()):
            return qwen.Qwen3_5_2b_UnifiedEngine(multi_core=active, device="cpu")

    def test_import_preserves_measured_clock(self):
        self.assertEqual(_clock_before_import, _clock_after_import)

    def test_kintex_complete_shared_model_and_private_shards_fit(self):
        engine = self.engine()
        self.assertEqual(engine._decode_windows, [(0, 512 << 20), (3 << 30, 512 << 20)])
        self.assertEqual((engine._params_dram_base, engine._params_limit,
                          engine._program_dram_base, engine._model_end),
                         (1 << 30, 2 << 30, 2560 << 20, 3 << 30))
        self.assertEqual(engine._decode_private_weight_bytes, [499400704, 499400704])
        self.assertTrue(all(used < 480 << 20 for used in engine._decode_private_weight_bytes))

    def test_u50_eight_engine_layout_and_legacy_layout_remain_disjoint(self):
        engine = self.engine(8, 8, 8)
        self.assertEqual(engine._params_dram_base, 6 << 30)
        self.assertEqual(len(engine._decode_windows), 8)
        self.assertTrue(all(base + size <= 6 << 30 for base, size in engine._decode_windows))
        legacy = self.engine(active=1)
        self.assertEqual((legacy._params_dram_base, legacy._tensor_dram_base,
                          legacy._program_dram_base), (0, 0xB0000000, 0xD0000000))
        self.assertIsNone(legacy._ensure_decode_sharder())

    def test_capacity_and_allocator_guards_reject_before_dma(self):
        with patch("multi_engine_shard.model_multicore_layout",
                   return_value=([(0, 256 << 20), (3 << 30, 256 << 20)], 1 << 30)):
            with self.assertRaises(MemoryError):
                self.engine()
        engine = self.engine()
        # prepare_inference moves the live tensor base past persistent caches.
        # That must not silently grow the immutable params allocation limit.
        engine._tensor_dram_base += 128 << 20
        for name, cursor, limit in (("allocate_params_dram", "_next_params_dram_addr", engine._params_limit),
                                    ("allocate_tensor_dram", "_tensor_dram_addr", engine._program_dram_base),
                                    ("allocate_program_dram", "_next_program_dram_addr", engine._model_end)):
            setattr(engine, cursor, limit - 64)
            self.assertEqual(getattr(engine, name)(64), limit - 64)
            with self.assertRaises(MemoryError):
                getattr(engine, name)(1)
            self.assertEqual(getattr(engine, cursor), limit)

    def test_projection_plan_includes_doubled_q_fused_kv_and_complete_lm_head(self):
        engine = self.engine()
        plan = list(engine._decode_projection_specs())
        self.assertEqual(len(plan), 145)
        full = {(name, K, N) for layer, name, K, N in plan if layer == 3}
        self.assertIn(("Q_Q", 2048, 4096), full)
        self.assertIn(("Q_KV", 2048, 1024), full)
        self.assertNotIn("Q_K", {name for name, _, _ in full})
        self.assertIn((None, "lm_head", 2048, 248320), plan)
        self.assertTrue(all(K <= core.SCALE_BRAM_ELEMENTS for _, _, K, _ in plan))

    def test_sharder_registers_each_uploaded_operand_once_and_dispatches_only_in_compile(self):
        engine = self.engine()
        cursor = engine._params_dram_base
        engine._layer_weights = {layer: {} for layer in range(engine.num_layers)}
        for layer, name, K, N in engine._decode_projection_specs():
            scale, weight = cursor, cursor + K * N // 64 * 2
            cursor = weight + K * N // 2
            if layer is None:
                engine._lm_head_scale_dram, engine._lm_head_data_dram = scale, weight
            else:
                engine._layer_weights[layer][name] = (scale, weight)
        decoder = Mock()
        with patch.object(multi_engine_decode, "ControllerShardedDecoder", return_value=decoder) as factory:
            self.assertIs(engine._ensure_decode_sharder(), decoder)
            self.assertIs(engine._ensure_decode_sharder(), decoder)
            factory.assert_called_once()
        self.assertEqual(decoder.add_weight.call_count, 145)
        sources = [call.args[1] for call in decoder.add_weight.call_args_list]
        self.assertEqual(len(sources), len(set(sources)))
        engine.quantized_matmat_core = Mock(return_value=17)
        decoder.projection.return_value = 23
        kwargs = dict(M=1, K=2048, N=6144)
        self.assertEqual(engine._decode_projection_core(**kwargs), 17)
        decoder.projection.assert_not_called()
        engine._compile_mode = True
        self.assertEqual(engine._decode_projection_core(**kwargs), 23)
        decoder.projection.return_value = None
        self.assertEqual(engine._decode_projection_core(**kwargs), 17)

    def test_primary_only_emission_cache_cannot_skip_worker_rounds(self):
        engine = self.engine()
        engine._compile_mode = True
        emit = Mock()
        engine._primitive_cache["same"] = ["stale primary instructions"]
        qwen._cached_emit(engine, "same", emit)
        qwen._cached_emit(engine, "same", emit)
        self.assertEqual(emit.call_count, 2)

    def test_execution_starts_workers_before_primary_and_merges_argmax(self):
        engine = self.engine()
        events = []
        engine._decode_sharder = SimpleNamespace(
            start=lambda: events.append("workers"), wait=lambda timeout: events.append("wait"),
            global_argmax=Mock(return_value=123))
        engine.start_execute_from_dram = lambda addr: events.append("primary")
        engine.is_queue_busy = lambda: False
        engine.report_latency_in_us = lambda: 1000
        engine.LOGITS_DRAM = 0x81000000
        self.assertEqual(engine._execute_decoder_program(), 1000)
        self.assertEqual(events, ["workers", "primary", "wait"])
        self.assertEqual(engine._decoder_argmax(), 123)
        engine._decode_sharder.global_argmax.assert_called_once_with("lm_head", engine.LOGITS_DRAM)

    def test_recurrent_prefill_and_decode_launch_each_token_and_report_timings(self):
        engine = self.engine()
        engine.max_context = engine.max_context_aligned = 64
        engine._embed_weight = torch.zeros(12, engine.hidden_size, dtype=torch.bfloat16)
        engine._bias_dram = engine.PENALTY_BIAS_DRAM = 0x80000000
        engine.VOCAB = 12
        engine.dma_write = Mock()
        engine.isa_add_set_core = Mock()
        engine.reset_state = lambda: setattr(engine, "_cache_pos", 0)
        engine._execute_decoder_program = Mock(return_value=1000)
        engine._decoder_argmax = Mock(side_effect=[7, 8, 9])
        tokenizer = SimpleNamespace(decode=lambda ids, **kw: str(ids))
        first = qwen.prefill_via_decode(engine, torch.tensor([2, 3]), verbose=False)
        self.assertEqual((first, engine._cache_pos), (7, 2))
        result = engine.run_decoder(tokenizer, first, max_new_tokens=3, verbose=False)
        self.assertEqual(result["token_ids"], [7, 8, 9])
        self.assertEqual(result["hw_step_us"], [1000, 1000])
        self.assertEqual(result["decode_first_hw_ms"], 1)
        self.assertEqual(result["decode_avg_hw_ms"], 1)
        self.assertEqual(engine._execute_decoder_program.call_count, 4)
        self.assertEqual(engine._cache_pos, 4)

    def test_cache_key_covers_context_and_mc_cache_is_rejected_before_dma(self):
        engine = self.engine()
        engine.max_context = 256
        first = engine._decoder_cache_key()
        engine.max_context = 512
        self.assertNotEqual(first, engine._decoder_cache_key())
        with patch.object(core, "UE_AXI_DATA_WIDTH_BITS", 256):
            axi256 = engine._decoder_cache_key()
        with patch.object(core, "UE_AXI_DATA_WIDTH_BITS", 512):
            self.assertNotEqual(axi256, engine._decoder_cache_key())
        unchanged = engine._decoder_cache_key()
        engine._cfg = {**engine._cfg, "model": {**engine._cfg["model"], "end_of_turn_token_id": 42}}
        self.assertNotEqual(unchanged, engine._decoder_cache_key())
        engine._decoder_meta_path = "unused.json"
        with self.assertRaisesRegex(ValueError, "compiled with its workers"):
            engine.load_instructions("unused.bin")
        with self.assertRaisesRegex(ValueError, "fresh compilation"):
            qwen.load_decoder_from_bin(engine, b"", {})
        legacy = self.engine(active=1)
        legacy.max_context = 256
        with self.assertRaisesRegex(ValueError, "stale"):
            qwen.load_decoder_from_bin(legacy, b"", {"controller_layout": legacy._controller_layout()})
        with tempfile.TemporaryDirectory() as directory:
            legacy._decoder_meta_path = str(Path(directory) / "program.json")
            Path(legacy._decoder_meta_path).write_text(json.dumps({"cache_key": {"max_context": 512}}))
            with self.assertRaisesRegex(ValueError, "stale"):
                legacy.load_instructions("unused.bin")

    def test_identity_addresses_are_scoped_to_each_memory_layout(self):
        first, second = self.engine(active=1), self.engine()
        first._identity_matrix_cache[64] = 0x1000
        with patch.object(qwen, "_upload", return_value=0x40001000) as upload:
            self.assertEqual(qwen._get_identity_dram(first, 64), 0x1000)
            self.assertEqual(qwen._get_identity_dram(second, 64), 0x40001000)
            upload.assert_called_once()


if __name__ == "__main__":
    unittest.main()
