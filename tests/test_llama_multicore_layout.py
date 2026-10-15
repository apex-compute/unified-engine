"""Llama allocation, dispatch, and address encoding without FPGA or model downloads."""

import builtins
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
import user_dma_core as core


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/llama3.2_1b/llama3.2_1b_test.py"


def forbid_hardware(stack):
    """Guard the class, so newly constructed worker engines are also covered."""
    for method in ("dma_read", "dma_write", "read_reg32", "write_reg32",
                   "user_read_reg32", "user_write_reg32", "_get_user_fd"):
        stack.enter_context(patch.object(
            core.UnifiedEngine, method,
            side_effect=AssertionError(f"unexpected hardware access: {method}")))
    for method in ("configure_clock_from_hardware", "read_hardware_info",
                   "read_core_clock_frequency_hz"):
        stack.enter_context(patch.object(
            core, method,
            side_effect=AssertionError(f"unexpected hardware access: {method}")))


with ExitStack() as _import_guards:
    forbid_hardware(_import_guards)
    _print = builtins.print
    try:
        _spec = importlib.util.spec_from_file_location("_llama_multicore_subject", MODEL_PATH)
        llama = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(llama)
    finally:
        builtins.print = _print


class LlamaMulticoreLayout(unittest.TestCase):
    def setUp(self):
        self.stack = self.enterContext(ExitStack())
        forbid_hardware(self.stack)
        self.stack.enter_context(patch.multiple(
            core, CLOCK_CYCLE_TIME_NS=4.0, QUEUE_MODE_ENABLED=True,
            UE_AXI_DATA_WIDTH_BITS=256))
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        self.weights = Path(self.directory) / "empty-weights.bin"
        self.weights.write_bytes(b"")
        self.addCleanup(setattr, llama, "_SILENT_MODE", False)

    def engine(self, cores=2, gib=4, active=2, **kwargs):
        with patch.multiple(core, ANDROMEDA_CORE_COUNT=cores,
                            AVAILABLE_DRAM_SIZE_GB=gib), \
                patch.object(llama.Llama32_1b_UnifiedEngine, "weight_init"), \
                patch.object(llama.Llama32_1b_UnifiedEngine, "tensor_init"):
            engine = llama.Llama32_1b_UnifiedEngine(
                weights_bin=str(self.weights), multi_core=active, **kwargs)
        engine.prefill_seq = None
        return engine

    def initialize_allocations(self, engine):
        # Execute the real allocation paths using a tiny host embedding. The
        # empty bin is deliberate: no weight data is required to test addresses.
        hf_model = Mock()
        hf_model.get_input_embeddings.return_value.weight = torch.zeros(
            (1, engine.vector_length), dtype=torch.bfloat16)
        with patch.object(llama, "_ensure_hf_model", return_value=(hf_model, self.directory)), \
                patch.object(llama.AutoTokenizer, "from_pretrained"), \
                patch.object(core.UnifiedEngine, "dma_write", return_value=0), \
                patch.object(core.UnifiedEngine, "dma_to_accelerator_memory"), \
                patch("builtins.print"):
            engine.weight_init()
            engine.tensor_init()

    def test_whole_model_rebases_without_changing_internal_offsets(self):
        for cores, gib, active, base in ((2, 4, 2, 1 << 30), (8, 8, 2, 6 << 30),
                                         (8, 8, 8, 6 << 30), (12, 16, 12, 6 << 30)):
            with self.subTest(cores=cores, gib=gib, active=active):
                engine = self.engine(cores, gib, active)
                self.assertEqual(engine._params_dram_base, base)
                self.assertEqual(engine._tensor_dram_base, base + 0x30000000)
                self.assertEqual(engine.WORKER_ISA_BASE, base + 0x7E000000)
                self.assertEqual(engine._program_dram_base, base + 0x7F600000)
                self.assertEqual(engine._model_dram_limit, base + (2 << 30))
                self.assertLessEqual(engine.WORKER_ISA_BASE + (active - 1) *
                                     engine.WORKER_ISA_STRIDE, engine._program_dram_base)
                for private_base, size in engine._decode_windows:
                    self.assertTrue(private_base + size <= base or
                                    private_base >= engine._model_dram_limit)

    def test_single_engine_preserves_original_model_map(self):
        engine = self.engine(active=1)
        self.assertEqual(engine._params_dram_base, 0x80000000)
        self.assertEqual(engine._tensor_dram_base, 0xB0000000)
        self.assertEqual(engine._program_dram_base, 0xFF600000)
        self.assertEqual(engine._model_dram_limit, 1 << 32)
        self.assertIsNone(engine._decode_windows)
        self.assertIsNone(engine._ensure_decode_sharder())
        self.assertIsNone(engine._ensure_prefill_scheduler())

    def test_library_initialization_needs_no_hardware_info_or_weights(self):
        with patch.object(core, "ANDROMEDA_CORE_COUNT", None), \
                patch.object(llama.Llama32_1b_UnifiedEngine, "weight_init") as weights, \
                patch.object(llama.Llama32_1b_UnifiedEngine, "tensor_init") as tensors:
            engine = llama.Llama32_1b_UnifiedEngine(initialize_model=False)
        self.assertEqual(engine.multi_core, 1)
        self.assertIsNone(engine._decode_windows)
        self.assertEqual(engine._params_dram_base, 0x80000000)
        weights.assert_not_called()
        tensors.assert_not_called()

    def test_allocators_fail_before_advancing_into_another_region(self):
        for cores, gib in ((2, 4), (8, 8)):
            engine = self.engine(cores, gib)
            for allocate, cursor, limit in (
                    (engine.allocate_params_dram, "_next_params_dram_addr", engine._tensor_dram_base),
                    (engine.allocate_tensor_dram, "_tensor_dram_addr", engine.WORKER_ISA_BASE),
                    (engine.allocate_program_dram, "_next_program_dram_addr", engine._model_dram_limit)):
                with self.subTest(cores=cores, allocator=allocate.__name__):
                    setattr(engine, cursor, limit - 64)
                    self.assertEqual(allocate(64), limit - 64)
                    with self.assertRaises(MemoryError):
                        allocate(1)
                    self.assertEqual(getattr(engine, cursor), limit)

    def test_capture_write_rejects_padded_overflow_before_dma(self):
        engine = self.engine()
        engine.capture_count = 1
        with patch.object(core.UnifiedEngine, "write_captured_instructions_to_dram") as write:
            with self.assertRaises(MemoryError):
                engine.write_captured_instructions_to_dram(engine._model_dram_limit - 32)
            write.assert_not_called()
            engine.write_captured_instructions_to_dram(engine._model_dram_limit - 64)
            write.assert_called_once_with(engine._model_dram_limit - 64)

    def test_cache_paths_include_core_count_and_actual_placement(self):
        engine = self.engine()
        original = engine._instruction_paths()
        self.assertEqual(original, self.engine()._instruction_paths())
        self.assertIn("_mc2_", original[0])
        engine._params_dram_base += 512 << 20
        self.assertNotEqual(original, engine._instruction_paths())
        engine = self.engine()
        engine._decode_windows = list(reversed(engine._decode_windows))
        self.assertNotEqual(original, engine._instruction_paths())
        self.assertNotEqual(original, self.engine(8, 8, 2)._instruction_paths())
        self.assertNotEqual(self.engine(8, 8, 2)._instruction_paths(),
                            self.engine(8, 8, 4)._instruction_paths())
        self.assertNotIn("_mc", self.engine(active=1)._instruction_paths()[0])

    def test_shipped_weight_and_tensor_layouts_fit_and_register_all_projections(self):
        engine = self.engine(8, 8, 2)
        self.initialize_allocations(engine)
        self.assertLess(engine.get_params_dram_addr(), engine._tensor_dram_base)
        self.assertLess(engine.get_tensor_dram_addr(), engine.WORKER_ISA_BASE)
        with patch("multi_engine_decode.ControllerShardedDecoder") as decoder_type:
            decoder = engine._ensure_decode_sharder()
            self.assertIs(engine._ensure_decode_sharder(), decoder)
            decoder_type.assert_called_once_with(engine, 2, engine._decode_windows)
            calls = decoder.add_weight.call_args_list
            self.assertEqual([call.args[0] for call in calls],
                             ["q_proj", "k_proj", "v_proj", "attn_proj",
                              "mlp_gate", "mlp_up", "mlp_down", "lm_head"])
            for call in calls:
                _, weight_addr, scale_addr, K, N, layers, stride = call.args
                self.assertGreaterEqual(weight_addr, 6 << 30)
                self.assertGreaterEqual(scale_addr, 6 << 30)
                self.assertGreater(K, 0)
                self.assertGreater(N, 0)
                if layers > 1:
                    self.assertEqual(stride, engine.weight_defs["LAYER_WEIGHT_SIZE"])

    def test_decoder_routes_supported_projections_and_keeps_primary_fallback(self):
        engine = self.engine(12, 16, 12)
        self.initialize_allocations(engine)
        engine.LAYER_SIZE = 1
        engine._decode_sharder = Mock()
        # Twelve cores cannot split K/V's eight 64-column blocks. The adapter
        # returns None for those weights and the original kernel must run.
        engine._decode_sharder.projection.side_effect = (
            lambda **kwargs: None if kwargs["N"] == engine.head_dim else 1)
        engine.fpga_penalty = True
        with ExitStack() as kernels:
            for name in ("rms_norm_core_dram", "rope_hf_core_decode", "unified_attention_core",
                         "eltwise_core_dram", "eltwise_add_core", "eltwise_mul_core"):
                kernels.enter_context(patch.object(engine, name, return_value=0))
            primary = kernels.enter_context(patch.object(engine, "quantized_matmat_core", return_value=1))
            engine.start_capture()
            engine._compile_decoder_program(layer_size=1)
            engine.stop_capture()
        calls = engine._decode_sharder.projection.call_args_list
        self.assertEqual(len(calls), 8)
        self.assertEqual(primary.call_count, 2)
        self.assertTrue(all(call.kwargs["N"] == engine.head_dim for call in primary.call_args_list))
        self.assertEqual(calls[-1].kwargs["B_DRAM_ADDR"], engine.DRAM_ADDR_LM_HEAD_QUANT)
        self.assertEqual(calls[-1].kwargs["C_DRAM_ADDR"], engine.PENALTY_BIAS_DRAM)
        self.assertEqual(calls[-1].kwargs["bias_mode"], "broadcast_N")

    def test_high_hbm_addresses_survive_kernel_and_jump_serialization(self):
        engine = self.engine(12, 16, 12)
        addresses = dict(A_DRAM_ADDR=6 * (1 << 30) + 0x30000000,
                         B_DRAM_ADDR=14 * (1 << 30),
                         SCALE_DRAM_ADDR=14 * (1 << 30) + 0x10000,
                         OUTPUT_DRAM_ADDR=7 * (1 << 30))
        for dynamic in (False, True):
            with self.subTest(dynamic=dynamic), patch("builtins.print"):
                engine.start_capture()
                extra = {"gpr_M_reg": engine.gpr_seq_len} if dynamic else {}
                engine.quantized_matmat_core(M=1, K=64, N=64, data_type=core.TYPE.IF4,
                                             **addresses, **extra)
                target = engine._program_dram_base + 64
                engine.generate_instruction_jump_abs(core.ue_35bit_addr_shifter(target))
                engine.stop_capture()
                values = set()
                for instruction in engine.capture_buffer:
                    words = struct.unpack("<8I", instruction.get_bytes())
                    kind = (words[0] >> 8) & 0xF
                    if kind in (core.INSTRUCTION_REG_ALU, core.INSTRUCTION_JUMP):
                        values.add((words[1] >> 22) | ((words[2] & 0x3FFFFF) << 10))
                    else:
                        values.add(words[1])
                for address in [*addresses.values(), target]:
                    self.assertIn(address >> 3, values)
                engine.clear_capture_buffer()

    def test_prefill_worker_row_register_stays_reserved_across_all_layers(self):
        from multi_engine_shard import MultiEngineScheduler
        engine = self.engine()
        self.initialize_allocations(engine)
        with patch.object(MultiEngineScheduler, "_save_dram_selftest_region", return_value=None):
            scheduler = engine._ensure_prefill_scheduler()
        worker = scheduler.workers[0]
        row_registers = []
        original = worker.rms_norm_core_dram

        def checked_norm(**kwargs):
            row_register = kwargs["gpr_M_reg"]
            # The next temporary register must be above the live loop bound.
            self.assertLess(row_register, worker._isa_reg_counter)
            self.assertLessEqual(row_register, 15)
            row_registers.append(row_register)
            return original(**kwargs)

        with ExitStack() as kernels:
            # Keep real worker kernels; primary attention emission is unrelated.
            for name in ("rms_norm_core_dram", "quantized_matmat_core",
                         "rope_hf_core_dram", "rope_hf_core_dram_gqa",
                         "unified_attention_core", "eltwise_core_dram",
                         "_emit_pbi_scatter_per_token"):
                kernels.enter_context(patch.object(engine, name, return_value=0))
            kernels.enter_context(patch.object(worker, "rms_norm_core_dram", side_effect=checked_norm))
            kernels.enter_context(patch("builtins.print"))
            engine.start_capture()
            scheduler.begin_program()
            engine._compile_prefill_program(29, engine.LAYER_SIZE, prefill_scheduler=scheduler)
            engine.stop_capture()
            worker.stop_capture()
        self.assertEqual(row_registers, [1] * engine.LAYER_SIZE)
        self.assertEqual(worker._isa_reg_counter, 1)

    def test_invalid_core_count_and_decode_kernel_are_rejected(self):
        with self.assertRaises(ValueError):
            self.engine(active=3)
        with self.assertRaisesRegex(ValueError, "streaming"):
            self.engine(decode_kernel="matmatmul")


if __name__ == "__main__":
    unittest.main()
