"""Gemma4 E2B controller placement and packaged-cache isolation, without DMA."""

import builtins
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

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

    def test_kintex_two_engine_tiles_follow_separate_ddr_controllers(self):
        engine = self.engine(2, 4, 2)
        self.assertTrue(engine._kintex_tiled_map)
        self.assertTrue(engine._tiled_map)
        self.assertEqual(engine._tile_bases, [0, 2 << 30])
        self.assertEqual(engine.mc_arena.windows, [(0, 2 << 30), (2 << 30, 2 << 30)])
        self.assertEqual(engine.mc_arena.tensor_bytes, 64 << 20)
        self.assertEqual(engine.mc_arena._private_reserve, [1 << 30, 1 << 30])
        self.assertEqual(engine.DRAM_END, 4 << 30)
        self.assertEqual(engine.TENSOR_LIMIT - engine._tensor_dram_base, 480 << 20)
        self.assertLessEqual(engine.TENSOR_LIMIT, engine.VISION_ISA_BASE)
        self.assertEqual(engine.mc_arena.isa_base(1), 0xFB000000)
        self.assertLess(engine.LM_ISA_BASE, engine.MASTER_ISA_LIMIT)
        self.assertEqual(engine.MASTER_ISA_LIMIT, engine.mc_arena.regions[0].tensor_base)
        self.assertNotEqual(engine.dram_layout, self.engine(active=2).dram_layout)

    def test_kintex_budget_covers_private_down_copies_shared_layers_and_scratch(self):
        engine = self.engine(2, 4, 2)
        plan = engine._kintex_capacity_plan
        # Conservative K/V on every layer plus two different down layouts.
        self.assertGreater(plan["private_weights_upper_bound"], 700 << 20)
        self.assertLess(plan["private_weights_upper_bound"], 1 << 30)
        self.assertGreater(plan["shared_weight_bytes"], 530 << 20)
        self.assertLess(plan["shared_weight_bytes"], 550 << 20)
        self.assertGreater(sum(plan["shared_free_bytes"]), 800 << 20)
        # A dry-run capacity check must not consume the live private cursors.
        self.assertEqual(engine.mc_arena.usage(), [0, 0])
        self.assertEqual(engine._ensure_prefill_mlp_lane_buffers()["gate"],
                         [engine.mc_arena.regions[i].tensor_base for i in range(2)])
        for index in range(2):
            used = engine.mc_arena._tensor_cursor[index] - engine.mc_arena.regions[index].tensor_base
            self.assertEqual(used, (19 << 20) + (512 << 10))

    def test_kintex_insufficient_reservation_fails_before_weight_loading(self):
        with patch.multiple(core, ANDROMEDA_CORE_COUNT=2,
                            AVAILABLE_DRAM_SIZE_GB=4, CLOCK_CYCLE_TIME_NS=4.0), \
                patch.object(gemma, "KINTEX_PRIVATE_RESERVE_BYTES", 512 << 20), \
                patch.object(gemma.Gemma4_UnifiedEngine, "weight_init") as load, \
                patch("builtins.print"):
            with self.assertRaisesRegex(MemoryError, "Kintex E2B private projections need"):
                gemma.Gemma4_UnifiedEngine(multi_core=2)
            load.assert_not_called()

    def test_kintex_mlp_stages_from_host_without_a_duplicate_shared_copy(self):
        engine = self.engine(2, 4, 2)
        engine._layer_dram_base = [0x90000000 + i * (16 << 20) for i in range(engine.LAYER_SIZE)]
        for prefix in ("Q_PROJ", "K_PROJ", "V_PROJ", "ATTN_PROJ"):
            setattr(engine, f"DRAM_ADDR_LAYER0_{prefix}_QUANT", 0x90000000)
            setattr(engine, f"DRAM_ADDR_LAYER0_{prefix}_SCALE", 0x91000000)
        for prefix in ("MLP_GATE", "MLP_UP", "MLP_DOWN"):
            setattr(engine, f"DRAM_ADDR_LAYER0_{prefix}_QUANT", None)
            setattr(engine, f"DRAM_ADDR_LAYER0_{prefix}_SCALE", None)
        engine.DRAM_ADDR_LM_HEAD_QUANT = 0xA0000000
        engine.DRAM_ADDR_LM_HEAD_SCALE = 0xA1000000
        scheduler = Mock(num_engines=2)
        scheduler.private_usage.return_value = [0, 0]
        engine._stage_private_n_shard = Mock()
        with patch("builtins.print"):
            engine._ensure_decode_qkv_shards(scheduler, engine.LAYER_SIZE)
        self.assertEqual(engine._stage_private_n_shard.call_count, 3 * engine.LAYER_SIZE)
        self.assertEqual({call.args[0] for call in engine._stage_private_n_shard.call_args_list},
                         {"gate", "up", "down"})
        # Poisoned shared MLP pointers must never reach the device-copy path.
        for call in scheduler.shard_quantized_weight.call_args_list:
            self.assertIsNotNone(call.kwargs["main_weight_addr"])
            self.assertIsNotNone(call.kwargs["main_scale_addr"])
            self.assertFalse(call.kwargs["name"].startswith(("gate", "up", "down")))

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
        current = {**stale, "dram_layout": engine.dram_layout,
                   "program_format_version": 2}
        with patch.object(engine, "_read_program_sections", return_value=({"lm": current}, b"test")):
            self.assertEqual(engine._get_program_section("lm"), (current, b"test"))

    def test_old_program_format_is_rejected_even_when_memory_layout_matches(self):
        engine = self.engine(active=1)
        for version in (None, 1):
            with self.subTest(version=version):
                metadata = {"dram_layout": engine.dram_layout,
                            "dram_base": hex(engine.LM_ISA_BASE),
                            "file_offset": 0, "size": 4}
                if version is not None:
                    metadata["program_format_version"] = version
                with patch.object(engine, "_read_program_sections",
                                  return_value=({"lm": metadata}, b"test")):
                    self.assertEqual(engine._get_program_section("lm"), (None, None))
                    with self.assertRaises(FileNotFoundError):
                        engine._load_program_section("lm")

    def test_dispatch_jumps_to_the_first_instruction_of_both_compiled_entries(self):
        # Keep the real image packer, ISA encoding, and dispatch builder. Small
        # bodies make a skipped first DMA visible without weights or hardware.
        engine = self.engine(active=1)
        engine.prefill_seq = (1, 2, 3)
        engine._preamble_addr = engine.LM_ISA_BASE + 4096
        first_input = {"prefill": 0x90000000, "decoder": 0x90001000}

        def emit_body(name):
            engine.accelerator_memory_to_sram(first_input[name], 0, 64)
            engine.generate_instruction_halt()

        def emit_prefill(**kwargs):
            emit_body("prefill")
            return None, 1

        def emit_decoder(**kwargs):
            start = engine.capture_count
            emit_body("decoder")
            engine._decoder_non_attention_flops = 1
            return None, [(engine.capture_count - start) * 32], [1]

        with patch.object(engine, "_get_program_section", return_value=(None, None)), \
                patch.object(engine, "_store_program_section") as store, \
                patch.object(engine, "compile_prefill", side_effect=emit_prefill), \
                patch.object(engine, "compile_decoder", side_effect=emit_decoder), \
                patch("builtins.print"):
            engine.compile_gemma4(layer_size=0)
        _, base, image_bytes, metadata = store.call_args.args
        for name in ("prefill", "decoder"):
            with self.subTest(entry=name):
                target = int(metadata[f"{name}_program_start_addr"], 16)
                first = struct.unpack_from("<8I", image_bytes, target - base)
                self.assertEqual((first[5] >> 12) & 15, core.UE_MODE.MEMCPY_FROM_DRAM)
                self.assertEqual(first[1] << 3, first_input[name])
                emitted = []

                def capture_preamble(address):
                    emitted.extend(tuple(inst.words) for inst in engine.capture_buffer)
                    return len(emitted) * 32

                with patch.object(engine, "write_captured_instructions_to_dram",
                                  side_effect=capture_preamble), \
                        patch.object(engine, "program_execute", return_value=(1, 1)), \
                        patch("builtins.print"):
                    engine._dispatch_program([(engine.gpr_seq_len, 2)], target)
                jumps = [w for w in emitted
                         if ((w[0] >> 8) & 15) == core.INSTRUCTION_JUMP]
                self.assertEqual(len(jumps), 1)
                encoded_target = (((jumps[0][1] >> 22) & 1023)
                                  | ((jumps[0][2] & 0x3FFFFF) << 10)) << 3
                # Checking the encoded PC catches the old +32 -> +64 rewrite,
                # which silently skipped the first GEMM input load.
                self.assertEqual(encoded_target, target)

    def test_unaligned_cached_entry_is_rejected_before_dispatch_or_profile_dma(self):
        engine = self.engine(active=1)
        stale_target = engine.LM_ISA_BASE + 32
        with self.assertRaisesRegex(ValueError, "64-byte aligned"):
            engine._dispatch_program([], stale_target)
        with self.assertRaisesRegex(ValueError, "64-byte aligned"):
            engine._profile_execute([], stale_target, [], "tail")

    def test_actual_prefill_and_decoder_profile_checkpoints_resume_after_halt_padding(self):
        engine = self.engine(active=1)
        for index, name in enumerate((
                "LAYER0_INPUT_DRAM", "DRAM_ADDR_PER_LAYER_MODEL_PROJ",
                "PER_LAYER_MODEL_PROJ_OUTPUT_DRAM", "PER_LAYER_INPUTS_DRAM",
                "DRAM_ADDR_PER_LAYER_PROJ_NORM", "PER_LAYER_EMBED_DRAM")):
            setattr(engine, name, 0x90000000 + index * 0x10000)

        def emit_kernel(**kwargs):
            # Exercise the real model checkpoint closures while keeping each
            # arithmetic kernel small; no weights or DMA are needed.
            engine.generate_instruction_nop()
            return 1

        with patch.object(engine, "matmat_mul_core_legacy", side_effect=emit_kernel), \
                patch.object(engine, "rms_norm_core_dram", side_effect=emit_kernel), \
                patch.object(engine, "eltwise_core_dram", side_effect=emit_kernel), \
                patch("builtins.print"):
            engine.start_capture()
            engine.generate_instruction_flag_clear()
            engine._align_dispatch_entry()
            engine.compile_prefill(seq_len=2, layer_size=0, profile=True)
            engine._align_dispatch_entry()
            engine.compile_decoder(layer_size=0, profile=True, accounting_seq_len=64)
            engine.stop_capture()
        checkpoints = engine._prefill_checkpoints + engine._decoder_checkpoints
        self.assertEqual(len(checkpoints), 2)
        for name, resume_hex, _ in checkpoints:
            with self.subTest(checkpoint=name, address=resume_hex):
                resume = int(resume_hex, 16)
                self.assertEqual(resume % 64, 0)
                index = (resume - engine.get_program_dram_addr()) // 32
                previous = [(inst.words[0] >> 8) & 15
                            for inst in engine.capture_buffer[max(index - 2, 0):index]]
                self.assertIn(core.INSTRUCTION_HALT, previous)
                # Metadata must skip any post-HALT padding and point at the
                # next executable instruction of the model program.
                following = (engine.capture_buffer[index].words[0] >> 8) & 15
                self.assertNotEqual(following, core.INSTRUCTION_NOP)

    def test_every_stored_section_includes_the_actual_layout(self):
        engine = self.engine()
        with tempfile.TemporaryDirectory() as directory:
            engine.script_dir = directory
            # Audio metadata did not previously include dram_layout. The common
            # writer must stamp it for every stage, including external callers.
            engine._store_program_section("audio", engine.LM_ISA_BASE, b"test", {})
            metadata, blob = engine._get_program_section("audio")
            self.assertEqual(metadata["dram_layout"], engine.dram_layout)
            self.assertEqual(metadata["program_format_version"], 2)
            self.assertEqual(blob, b"test")
            self.assertEqual(Path(engine._program_image_paths()[0]).name, "programs.bin")


if __name__ == "__main__":
    unittest.main()
