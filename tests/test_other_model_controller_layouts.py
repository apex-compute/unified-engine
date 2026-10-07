"""Controller placement and cache guards for existing SmolVLM2/pi05 paths."""
import builtins
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import user_dma_core as core
from multi_engine_shard import alveo_core_bases

_original_print = builtins.print
try:
    from models.smolvlm2 import smolvlm2_test as smol
    from models.pi05 import pi05_test as pi05
finally:
    builtins.print = _original_print


class ControllerModelTests(unittest.TestCase):
    def setUp(self):
        for name in ('dma_read', 'dma_write'):
            guard = patch.object(core.UnifiedEngine, name,
                                 side_effect=AssertionError('unexpected hardware access'))
            guard.start()
            self.addCleanup(guard.stop)
        guard = patch.object(core, 'configure_clock_from_hardware',
                             side_effect=AssertionError('unexpected hardware probe'))
        guard.start()
        self.addCleanup(guard.stop)

    def smol_engine(self, cores=2, gib=4, active=2):
        with patch.multiple(core, ANDROMEDA_CORE_COUNT=cores,
                            AVAILABLE_DRAM_SIZE_GB=gib, CLOCK_CYCLE_TIME_NS=4), \
                patch.object(smol.SmolVLM2_UnifiedEngine, 'DECODE_NUM_ENGINES', active):
            engine = smol.SmolVLM2_UnifiedEngine(num_engines=max(2, active))
            engine.DECODE_NUM_ENGINES = active
            return engine

    def test_smol_kintex_separates_weight_controllers_and_model(self):
        engine = self.smol_engine()
        self.assertEqual(engine._decode_windows, [(0, 512 << 20), (3 << 30, 512 << 20)])
        self.assertEqual(engine._params_dram_base, 1 << 30)
        self.assertEqual(engine._tensor_dram_base, 2 << 30)
        self.assertEqual(engine._model_end, 3 << 30)

    def test_smol_u50_and_u55_preserve_private_windows_outside_model(self):
        for cores, gib, active in ((8, 8, 8), (12, 16, 8), (12, 8, 8)):
            engine = self.smol_engine(cores, gib, active)
            self.assertEqual(engine._model_base, 6 << 30)
            for base, size in engine._decode_windows:
                self.assertTrue(base + size <= engine._model_base or base >= engine._model_end)
        self.assertEqual([b for b, _ in self.smol_engine(8, 8, 8)._decode_windows],
                         alveo_core_bases(8))

    def test_smol_single_engine_retains_original_snapshot_names(self):
        engine = self.smol_engine(active=1)
        self.assertEqual(engine._params_dram_base, 2 << 30)
        self.assertEqual(engine._artifact_mode_suffix(), '')
        self.assertNotIn('controller_layout', engine._artifact_mode_meta())

    def test_smol_cache_checks_full_geometry_not_truthiness(self):
        engine = self.smol_engine()
        meta = engine._artifact_mode_meta()
        engine._validate_artifact_mode(meta, 'test')
        changed = copy.deepcopy(meta)
        changed['controller_layout']['private_windows'].reverse()
        with self.assertRaisesRegex(RuntimeError, 'metadata mismatch'):
            engine._validate_artifact_mode(changed, 'test')
        suffix = engine._artifact_mode_suffix()
        engine._decode_windows.reverse()
        self.assertNotEqual(suffix, engine._artifact_mode_suffix())

    def test_smol_allocators_reject_crossing_before_moving_cursor(self):
        engine = self.smol_engine()
        for allocate, cursor, limit in (
                (engine.allocate_params_dram, '_next_params_dram_addr', engine._tensor_dram_base),
                (engine.allocate_tensor_dram, '_tensor_dram_addr', engine._program_dram_base),
                (engine.allocate_program_dram, '_next_program_dram_addr', engine._model_end)):
            setattr(engine, cursor, limit - 64)
            self.assertEqual(allocate(64), limit - 64)
            with self.assertRaises(MemoryError):
                allocate(1)
            self.assertEqual(getattr(engine, cursor), limit)
        with patch.object(core.UnifiedEngine, 'write_captured_instructions_to_dram') as write:
            engine.capture_count = 1
            with self.assertRaises(MemoryError):
                engine.write_captured_instructions_to_dram(engine._model_end - 32)
            write.assert_not_called()

    def test_smol_private_copy_slices_packed_weights_and_scales_exactly(self):
        engine = self.smol_engine()
        K, N = engine.HIDDEN_SIZE, engine.INTERMEDIATE_SIZE
        base = engine._params_dram_base
        engine.lm_layer_addrs = [{'gate_data': base, 'gate_scale': base + (4 << 20),
                                 'up_data': base + (8 << 20), 'up_scale': base + (12 << 20)}]
        copy_dma = Mock()
        engine._lm_sched = SimpleNamespace(
            split_cols=lambda n: [(0, n // 2), (n // 2, n // 2)],
            _copy_dram_bytes=copy_dma)
        engine._materialize_decode_weights()
        self.assertEqual(copy_dma.call_count, 8)
        calls = [call.args for call in copy_dma.call_args_list]
        for index, (offset, width) in enumerate(((0, N // 2), (N // 2, N // 2))):
            dst_weight, dst_scale = engine._decode_private_weights[base][index]
            self.assertIn((base + offset * K // 2, dst_weight, width * K // 2), calls)
            self.assertIn((base + (4 << 20) + offset * K // 64 * 2,
                           dst_scale, width * K // 64 * 2), calls)
            region, length = engine._decode_windows[index]
            self.assertTrue(region <= dst_weight < dst_scale < region + length)
        engine._materialize_decode_weights()
        self.assertEqual(copy_dma.call_count, 8)

    def test_smol_cached_scheduler_retries_failed_materialization(self):
        engine = self.smol_engine()
        scheduler = engine._lm_sched = Mock()
        engine._materialize_decode_weights = Mock(side_effect=[IOError("copy failed"), None])
        with self.assertRaises(IOError):
            engine._lm_make_scheduler(2)
        self.assertIs(engine._lm_make_scheduler(2), scheduler)
        self.assertEqual(engine._materialize_decode_weights.call_count, 2)

    def test_smol_invalid_source_fails_before_any_copy(self):
        engine = self.smol_engine()
        engine.lm_layer_addrs = [{'gate_data': engine._tensor_dram_base - 64,
                                 'gate_scale': engine._params_dram_base,
                                 'up_data': engine._params_dram_base,
                                 'up_scale': engine._params_dram_base}]
        copy_dma = Mock()
        engine._lm_sched = SimpleNamespace(split_cols=lambda n: [(0, n // 2), (n // 2, n // 2)],
                                           _copy_dram_bytes=copy_dma)
        with self.assertRaisesRegex(ValueError, 'source escapes'):
            engine._materialize_decode_weights()
        copy_dma.assert_not_called()

    def pi_engine(self):
        return object.__new__(pi05.Pi05Libero_UnifiedEngine)

    def test_pi05_u50_replica_for_each_engine_uses_upper_owned_segment(self):
        with patch.multiple(core, ANDROMEDA_CORE_COUNT=8, AVAILABLE_DRAM_SIZE_GB=8):
            windows = self.pi_engine()._vis_controller_copy_windows(280 << 20, 8)
            self.assertEqual([base for base, _ in windows], alveo_core_bases(8, stack=1))
            self.assertEqual(len({base // (512 << 20) for base, _ in windows}), 8)
            with self.assertRaises(MemoryError):
                self.pi_engine()._vis_controller_copy_windows(513 << 20, 8)

    def test_pi05_u55_16gb_avoids_model_and_keeps_distinct_controllers(self):
        with patch.multiple(core, ANDROMEDA_CORE_COUNT=12, AVAILABLE_DRAM_SIZE_GB=16):
            windows = self.pi_engine()._vis_controller_copy_windows(280 << 20, 12)
            self.assertEqual(len(windows), 12)
            self.assertTrue(all(base >= 4 << 30 for base, _ in windows))
            self.assertEqual(len({base // (1 << 30) for base, _ in windows}), 12)

    def test_pi05_capacity_limit_is_explicit_on_4gb_and_8gb_u55(self):
        with patch.multiple(core, ANDROMEDA_CORE_COUNT=2, AVAILABLE_DRAM_SIZE_GB=4):
            self.assertEqual(self.pi_engine()._vis_controller_copy_windows(280 << 20, 2), [])
            with self.assertRaisesRegex(ValueError, 'HW_INFO'):
                self.pi_engine()._vis_controller_copy_windows(280 << 20, 8)
        with patch.multiple(core, ANDROMEDA_CORE_COUNT=12, AVAILABLE_DRAM_SIZE_GB=8):
            self.assertEqual(len(self.pi_engine()._vis_controller_copy_windows(280 << 20, 8)), 4)
            self.assertEqual(len(self.pi_engine()._vis_controller_copy_windows(280 << 20, 2)), 2)

    def test_smol_emits_private_weight_addresses_without_changing_outputs(self):
        engine = self.smol_engine()
        engine.DECODE_NUM_ENGINES = 2
        engine.LAYER0_PRE_NORM_DRAM = 0x80000000
        engine.LAYER0_MLP_GATE_DRAM = 0x80010000
        engine.LAYER0_MLP_UP_DRAM = 0x80020000
        engine.LAYER0_MLP_MULT_DRAM = 0x80030000
        la = {'gate_data': 100, 'gate_scale': 200, 'up_data': 300, 'up_scale': 400}
        private = {100: [(0x1000, 0x2000), (0xC0001000, 0xC0002000)],
                   300: [(0x3000, 0x4000), (0xC0003000, 0xC0004000)]}
        engine._decode_private_weights = private
        engines = [Mock(), Mock()]
        width = engine.INTERMEDIATE_SIZE // 2
        def emit(total, body, join):
            for index, ue in enumerate(engines):
                body(SimpleNamespace(engine_idx=index, unsafe_ue=ue, cols=width,
                                     col_offset=index * width, b_addr=lambda *a, **k: -1,
                                     scale_addr=lambda *a, **k: -1))
        engine._lm_sched = SimpleNamespace(col_sharded_region=emit)
        with patch.dict('os.environ', {'SMOLVLM2_DECODE_BARRIER_ONLY': '0'}):
            engine._emit_decode_gate_up(la, Mock(), engine.HIDDEN_SIZE,
                                        engine.INTERMEDIATE_SIZE, 2)
        for index, ue in enumerate(engines):
            for call, source, output in zip(ue.quantized_matmat_core.call_args_list,
                                           (100, 300), (engine.LAYER0_MLP_GATE_DRAM,
                                                        engine.LAYER0_MLP_UP_DRAM)):
                self.assertEqual(call.kwargs['B_DRAM_ADDR'], private[source][index][0])
                self.assertEqual(call.kwargs['SCALE_DRAM_ADDR'], private[source][index][1])
                self.assertEqual(call.kwargs['OUTPUT_DRAM_ADDR'], output + index * width * 2)
                self.assertEqual(call.kwargs['N'], width)
            self.assertEqual(ue.quantized_matmat_core.call_count, 2)

    def test_pi05_private_upload_preserves_scale_data_bytes_and_replay_records(self):
        engine = self.pi_engine()
        # One 64x64 IF4 projection: 128 scale bytes + 2048 packed data bytes.
        raw = bytes(range(128)) + bytes(range(256)) * 8
        engine.vis_layer_addrs = [{f'{key}_{part}': 0x100000 + index * 0x10000 + offset
                                  for index, key in enumerate(engine._VIS_COPY_KEYS)
                                  for part, offset in (('scale', 0), ('data', 128))}]
        engine._vis_copy_record = {'blobs': [], 'tensor_before': 2 << 30,
                                  'tensor_after': 2 << 30}
        writes = []
        engine._dma_write_retry = lambda dev, addr, tensor, size: writes.append(
            (addr, bytes(tensor.tolist()), size))
        packed = [{key: raw for key in engine._VIS_COPY_KEYS}]
        with patch.multiple(core, ANDROMEDA_CORE_COUNT=8, AVAILABLE_DRAM_SIZE_GB=8):
            self.assertTrue(engine._vis_upload_controller_copies(packed, len(raw) * 6, 2, 2))
        self.assertEqual(len(writes), 24)
        for scale, data in zip(writes[::2], writes[1::2]):
            self.assertEqual(scale[1], raw[:128])
            self.assertEqual(data[1], raw[128:])
        self.assertEqual(len(engine._vis_copy_record['blobs']), 24)
        self.assertEqual(engine._vis_copy_record['tensor_after'], 2 << 30)
        for dst, src, size in engine._vis_copy_record['blobs']:
            self.assertGreaterEqual(dst, 4 << 30)
            self.assertLess(src + size, 4 << 30)
        self.assertEqual(len(engine._vis_weight_sets), 2)

    def test_pi05_replay_rejects_changed_controller_map_before_copy(self):
        engine = object.__new__(pi05.Pi05Libero_Run)
        engine._num_engines = lambda stage: 8
        engine._manifest = {'vis_weight_copies': {
            'controller_layout_version': 1, 'copy_bytes': 280 << 20,
            'requested_sets': 8, 'board': [8, 8], 'controller_windows': [[0, 512 << 20]],
            'blobs': [], 'tensor_before': 2 << 30, 'tensor_after': 2 << 30}}
        engine._tensor_dram_addr = 2 << 30
        engine._dma_write_retry = Mock()
        with patch.multiple(core, ANDROMEDA_CORE_COUNT=8, AVAILABLE_DRAM_SIZE_GB=8):
            for attempt in range(2):
                with self.assertRaisesRegex(RuntimeError, 'controller map changed'):
                    engine._vis_alloc_weight_copies(7)
                self.assertIsNone(getattr(engine, '_vis_weight_sets', None))
        engine._dma_write_retry.assert_not_called()


if __name__ == '__main__':
    unittest.main()
