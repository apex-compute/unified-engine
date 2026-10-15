"""Decode integration contracts without opening an FPGA device."""

import builtins
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import multi_engine_decode as decode
from multi_engine_shard import MultiEngineScheduler, ShardedWeight, WeightShard
from user_dma_core import TYPE


class DecodeShards(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.engines = [Mock(), Mock()]
        self.group = Mock(num_engines=2, engines=self.engines,
                          primary=self.engines[0], workers=self.engines[1:])
        for index, engine in enumerate(self.engines):
            engine.quantized_matmat_core.side_effect = (
                lambda index=index, **kw: self.events.append(("matmul", index, kw)))
            engine.is_queue_busy.return_value = False
        for name in ("release", "join", "begin_worker_round", "end_worker_round"):
            getattr(self.group, name).side_effect = (
                lambda *args, name=name: self.events.append((name, *args)))
        self.group.shard_quantized_weight.side_effect = self.make_weight
        self.group.finalize.return_value = [0xDF000000]
        with patch.object(decode, "MultiEngineScheduler", return_value=self.group), \
                patch.object(decode, "PrivateArena"):
            self.decoder = decode.ControllerShardedDecoder(Mock(), 2,
                                                           [(0, 2**29), (3 * 2**30, 2**29)])

    @staticmethod
    def make_weight(**kw):
        n, k = kw["N"], kw["K"]
        cols = n // 2
        return ShardedWeight(
            name=kw["name"], K=k, N=n, layers=kw["layers"], data_type=kw["data_type"],
            shards=[WeightShard(i, i * cols, cols, i * 3 * 2**30 + 0x100000,
                                i * 3 * 2**30 + 0x200000, cols * k // 2, cols * k // 32)
                    for i in range(2)])

    def register(self, **kw):
        defaults = dict(name="gate", main_weight_addr=0x40000000,
                        main_scale_addr=0x41000000, K=128, N=256,
                        layers=3, main_layer_stride=0x10000)
        defaults.update(kw)
        return self.decoder.add_weight(**defaults)

    def projection_args(self, **kw):
        defaults = dict(M=1, K=128, N=256, A_DRAM_ADDR=0x60000000,
                        B_DRAM_ADDR=0x40010000, SCALE_DRAM_ADDR=0x41010000,
                        OUTPUT_DRAM_ADDR=0x60010000, data_type=TYPE.IF4)
        defaults.update(kw)
        return defaults

    def test_every_round_uses_matching_layer_shards_and_preserves_silu(self):
        sw = self.register()
        flops = self.decoder.projection(**self.projection_args(silu_enable=True, gpr_M_reg=5))
        self.assertEqual(flops, 2 * 128 * 256)
        self.assertEqual([event[:2] for event in self.events],
                         [("release",), ("matmul", 0), ("begin_worker_round", 1),
                          ("matmul", 1), ("end_worker_round", 1), ("join",)])
        for i, engine in enumerate(self.engines):
            call = engine.quantized_matmat_core.call_args.kwargs
            shard = sw.shard(i)
            self.assertEqual(call["B_DRAM_ADDR"], shard.weight_addr + shard.layer_stride)
            self.assertEqual(call["SCALE_DRAM_ADDR"], shard.scale_addr + shard.scale_layer_stride)
            self.assertEqual(call["OUTPUT_DRAM_ADDR"], 0x60010000 + shard.col_offset * 2)
            self.assertEqual(call["N"], shard.cols)
            self.assertTrue(call["silu_enable"])
            self.assertNotIn("gpr_M_reg", call)

    def test_lm_head_slices_penalty_bias_and_enables_candidate_writeback(self):
        self.register(name="lm_head")
        self.decoder.projection(**self.projection_args(
            C_DRAM_ADDR=0x60020000, bias_mode="broadcast_N", write_back_disable=True))
        for i, engine in enumerate(self.engines):
            call = engine.quantized_matmat_core.call_args.kwargs
            self.assertEqual(call["C_DRAM_ADDR"], 0x60020000 + i * 128 * 2)
            self.assertFalse(call["write_back_disable"])
        self.group.global_argmax.return_value = 193
        self.assertEqual(self.decoder.global_argmax("lm_head", 0x60010000), 193)
        self.group.global_argmax.assert_called_once_with(
            self.decoder.weights["lm_head"], 0x60010000)

    def test_unregistered_bf16_or_narrow_projection_stays_on_primary(self):
        self.assertIsNone(self.register(N=64))
        self.group.shard_quantized_weight.assert_not_called()
        self.assertIsNone(self.decoder.projection(**self.projection_args()))
        self.assertFalse(self.events)

    def test_shape_scale_runtime_address_and_bias_mismatches_fail_before_release(self):
        self.register()
        for changes in (dict(M=2), dict(K=256), dict(N=128), dict(data_type=TYPE.IF8),
                        dict(SCALE_DRAM_ADDR=0x41000000), dict(gpr_a_addr=7),
                        dict(C_DRAM_ADDR=0x60020000, bias_mode="full")):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.decoder.projection(**self.projection_args(**changes))
        self.assertFalse(self.events)

    def test_workers_must_be_compiled_and_relaunched_for_every_token(self):
        with self.assertRaises(RuntimeError):
            self.decoder.start()
        self.decoder.begin()
        self.group.begin_program.assert_called_once()
        for engine in self.engines:
            engine.generate_instruction_flag_clear.assert_called_once()
        self.assertEqual(self.decoder.finalize(), [0xDF000000])
        self.decoder.start()
        self.decoder.wait()
        self.decoder.start()
        self.assertEqual(self.group.start_workers.call_count, 2)
        self.engines[1].is_queue_busy.return_value = True
        with self.assertRaisesRegex(TimeoutError, "engine 1"):
            self.decoder.wait(timeout=0.1)

    def test_global_argmax_rejects_nan_in_any_engine_candidate(self):
        sw = self.register(name="lm_head")
        self.group._read_bf16 = Mock()
        for engine in self.engines:
            engine.get_arg_max_index.return_value = 3
        for values in ((float("nan"), 1.0), (1.0, float("nan"))):
            self.group._read_bf16.side_effect = values
            with self.subTest(values=values), self.assertRaises(FloatingPointError):
                MultiEngineScheduler.global_argmax(self.group, sw, 0x60010000)

    def test_global_argmax_keeps_masked_shards_and_stable_finite_ties(self):
        sw = self.register(name="lm_head")
        for engine in self.engines:
            engine.get_arg_max_index.return_value = 3
        for values, winner in (((float("-inf"), 2.0), 131), ((2.0, 2.0), 3)):
            self.group._read_bf16 = Mock(side_effect=values)
            self.assertEqual(MultiEngineScheduler.global_argmax(
                self.group, sw, 0x60010000), winner)


class QwenLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(__file__).resolve().parents[1] / "models/qwen3_0.6b/qwen3_0.6b_test.py"
        spec = importlib.util.spec_from_file_location("qwen06_test_contracts", cls.path)
        cls.module = importlib.util.module_from_spec(spec)
        original_print = builtins.print
        try:
            spec.loader.exec_module(cls.module)
        finally:
            builtins.print = original_print
        cls.cfg = cls.module._load_config(str(cls.path.parent))

    def test_exact_embedding_and_program_bounds_rebase_together(self):
        for base in (1 << 30, 2 << 30, 6 << 30):
            with self.subTest(base=base):
                embedding, size = self.module._device_bf16_embedding_layout(self.cfg, base)
                self.assertEqual(embedding + size, base + 2 * 2**30)
                program = base + 0x60000000
                self.module._validate_program_embedding_separation(2**20, embedding,
                                                                    10 * 2**20, program)
                with self.assertRaisesRegex(ValueError, "overlapping"):
                    self.module._validate_program_embedding_separation(
                        embedding - program, embedding, 64, program)

    def test_multicore_cache_paths_include_count_and_controller_layout(self):
        model = object.__new__(self.module.Qwen3_0_6b_UnifiedEngine)
        model.script_dir = str(self.path.parent)
        model._cfg = self.cfg
        model._model_base = 1 << 30
        model._board_windows = [(0, 2**29), (3 << 30, 2**29)]
        model.multi_core = 1
        single = model.instruction_paths()
        model.multi_core = 2
        dual = model.instruction_paths()
        model._board_windows = [(0, 2**29), (2 << 30, 2**29)]
        different_map = model.instruction_paths()
        self.assertNotEqual(single, dual)
        self.assertNotEqual(dual, different_map)
        self.assertIn("_mc2_", dual[0])

    def test_decode_dispatch_keeps_quantized_and_bf16_paths_distinct(self):
        model = object.__new__(self.module.Qwen3_0_6b_UnifiedEngine)
        model.controller_decoder = Mock()
        model.controller_decoder.projection.return_value = 65536
        model.quantized_matmat_core = Mock()
        model.matmat_mul_core = Mock(return_value=131072)
        model._matmul_b_kwargs = Mock(return_value={
            "is_B_quantized": True, "B_DRAM_ADDR": 0x44000000,
            "SCALE_DRAM_ADDR": 0x41000000, "data_type": TYPE.IF4})
        args = dict(M=1, K=128, N=256, A_DRAM_ADDR=0x60000000,
                    OUTPUT_DRAM_ADDR=0x60010000, silu_enable=True)
        self.assertEqual(model._decode_matmul(0, "gate", **args), 65536)
        self.assertTrue(model.controller_decoder.projection.call_args.kwargs["silu_enable"])
        model.quantized_matmat_core.assert_not_called()
        model.controller_decoder.projection.reset_mock()
        model._matmul_b_kwargs.return_value = {"is_B_quantized": False, "B_DRAM_ADDR": 0x45000000}
        self.assertEqual(model._decode_matmul(27, "gate", **args), 131072)
        model.controller_decoder.projection.assert_not_called()


class QueueReset(unittest.TestCase):
    def test_reset_uses_reported_engines_without_selftest(self):
        engines = [Mock(), Mock()]
        for engine in engines:
            engine.is_queue_busy.return_value = False
        with patch.object(decode.user_dma_core, "ANDROMEDA_CORE_COUNT", 2), \
                patch.object(decode.user_dma_core, "UnifiedEngine", side_effect=engines) as factory:
            decode.reset_engine_queues(2)
        self.assertEqual(factory.call_count, 2)
        for i, call in enumerate(factory.call_args_list):
            self.assertEqual(call.kwargs, {
                "BASE_ADDR": decode.user_dma_core.UE_0_BASE_ADDR + i * 0x10000,
                "init_unified_engine": False})
        for engine in engines:
            engine.write_reg32.assert_called_once_with(decode.user_dma_core.UE_QUEUE_CTRL_ADDR,
                                                       0x80008000)
            engine.software_reset.assert_not_called()
            engine.init_unified_engine.assert_not_called()

    def test_invalid_count_and_busy_after_reset_fail(self):
        with patch.object(decode.user_dma_core, "ANDROMEDA_CORE_COUNT", 2), \
                patch.object(decode.user_dma_core, "UnifiedEngine") as factory:
            for count in (0, 3):
                with self.assertRaises(ValueError):
                    decode.reset_engine_queues(count)
            factory.assert_not_called()
            factory.return_value.is_queue_busy.return_value = True
            with patch.object(decode.time, "monotonic", side_effect=[0.0, 2.0]), \
                    self.assertRaisesRegex(TimeoutError, "engine 0"):
                decode.reset_engine_queues(1, timeout=1.0)


if __name__ == "__main__":
    unittest.main()
