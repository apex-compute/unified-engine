"""Exact-copy, cache replay and emitter coverage without FPGA access."""
import builtins
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import user_dma_core as core
from models.controller_replicas import ControllerReplicas, free_controller_windows

_saved_print = builtins.print
try:
    from models.act import act_test as act
    from models.kokoro import kokoro_fpga as kokoro
    from models.verapulse import verapulse_test as vera
finally:
    builtins.print = _saved_print


class ControllerReplicaTests(unittest.TestCase):
    def setUp(self):
        for owner, name in ((core.UnifiedEngine, 'dma_read'), (core.UnifiedEngine, 'dma_write'),
                            (core, 'configure_clock_from_hardware')):
            guard = patch.object(owner, name, side_effect=AssertionError('unexpected FPGA access'))
            guard.start()
            self.addCleanup(guard.stop)
        board = patch.multiple(core, ANDROMEDA_CORE_COUNT=8, AVAILABLE_DRAM_SIZE_GB=8)
        board.start()
        self.addCleanup(board.stop)

    def primary(self):
        writes = {}
        def read(device, address, data, count):
            data[:] = bytes((address + offset) % 256 for offset in range(count))
            return count
        def write(device, address, data, count):
            writes[address] = bytes(data)
            return count
        return SimpleNamespace(dma_read=Mock(side_effect=read), dma_write=Mock(side_effect=write),
                               writes=writes)

    def test_exact_copy_uses_eight_free_regions_and_memoizes(self):
        primary = self.primary()
        pool = ControllerReplicas(primary, 8)
        addresses = pool.copy(0x1234, 70)
        expected = bytes((0x1234 + offset) % 256 for offset in range(70))
        self.assertEqual(len(set(addresses)), 8)
        self.assertTrue(all(address >= 4 << 30 for address in addresses))
        self.assertEqual({primary.writes[address] for address in addresses}, {expected})
        self.assertEqual(pool.copy(0x1234, 70), addresses)
        primary.dma_read.assert_called_once()
        next_addresses = pool.copy(0x2345, 128)
        self.assertEqual([b - a for a, b in zip(addresses, next_addresses)], [128] * 8)

    def test_u55_old_image_shares_only_four_free_controllers(self):
        with patch.multiple(core, ANDROMEDA_CORE_COUNT=12, AVAILABLE_DRAM_SIZE_GB=8):
            primary = self.primary()
            pool = ControllerReplicas(primary, 12)
            addresses = pool.copy(0, 128)
            self.assertEqual(len(pool.windows), 4)
            self.assertEqual(addresses[:4], addresses[4:8])
            self.assertEqual(addresses[:4], addresses[8:12])
            self.assertEqual(primary.dma_write.call_count, 4)

    def test_kintex_retains_shared_addresses_without_replica_dma(self):
        with patch.multiple(core, ANDROMEDA_CORE_COUNT=2, AVAILABLE_DRAM_SIZE_GB=4):
            primary = self.primary()
            pool = ControllerReplicas(primary, 2)
            self.assertEqual(pool.copy(128, 256), [128, 128])
            primary.dma_read.assert_not_called()
            primary.dma_write.assert_not_called()

    def test_capacity_and_source_guard_precede_dma(self):
        primary = self.primary()
        pool = ControllerReplicas(primary, 8)
        for source, size, error in ((0, (512 << 20) + 1, MemoryError),
                                    ((4 << 30) - 64, 128, ValueError), (0, 0, ValueError)):
            with self.assertRaises(error):
                pool.copy(source, size)
        primary.dma_read.assert_not_called()
        primary.dma_write.assert_not_called()

    def test_short_source_read_and_replica_write_fail(self):
        primary = self.primary()
        primary.dma_read = Mock(return_value=63)
        pool = ControllerReplicas(primary, 2)
        with self.assertRaisesRegex(IOError, 'reading'):
            pool.copy(0, 64)
        primary.dma_write.assert_not_called()
        primary = self.primary()
        primary.dma_write = Mock(return_value=63)
        pool = ControllerReplicas(primary, 2)
        with self.assertRaisesRegex(IOError, 'writing'):
            pool.copy(0, 64)
        self.assertEqual(pool.records, [])

    def test_cached_replicas_restore_exact_addresses_and_contents(self):
        primary = self.primary()
        original = ControllerReplicas(primary, 8)
        original.copy(0, 64)
        original.copy(0x100, 256)
        replay_primary = self.primary()
        replay = ControllerReplicas(replay_primary, 8)
        replay.restore(copy.deepcopy(original.manifest()))
        self.assertEqual(replay.manifest(), original.manifest())
        self.assertEqual(replay_primary.writes, primary.writes)

    def test_cached_layout_and_destination_changes_fail_before_dma(self):
        original = ControllerReplicas(self.primary(), 8)
        original.copy(0, 64)
        for field in ('layout', 'destination'):
            manifest = copy.deepcopy(original.manifest())
            if field == 'layout':
                manifest['layout']['windows'].reverse()
            else:
                manifest['copies'][0][2][0] += 64
            primary = self.primary()
            with self.assertRaises(ValueError):
                ControllerReplicas(primary, 8).restore(manifest)
            primary.dma_read.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'lack controller replicas'):
            ControllerReplicas(self.primary(), 8).restore(None)

    def test_act_mm_maps_only_immutable_weight_operands(self):
        engine = object.__new__(act.ACT_UnifiedEngine)
        engine._base_addr = 0x2000000
        engine._params_dram_base = 0
        engine._controller_weight_end = 0x1000000
        engine._flops = 0
        engine._controller_replicas = Mock()
        engine._controller_replicas.address.return_value = 0x160000000
        worker = Mock(_base_addr=engine._base_addr + 3 * 0x10000)
        engine.mm(worker, 1, 64, 64, 0x40000000, 0x100000, 0x40010000)
        engine._controller_replicas.address.assert_called_once_with(3, 0x100000, 8192)
        self.assertEqual(worker.matmat_mul_core.call_args.kwargs['B_DRAM_ADDR'], 0x160000000)
        self.assertEqual(worker.matmat_mul_core.call_args.kwargs['OUTPUT_DRAM_ADDR'], 0x40010000)
        engine.mm(worker, 1, 64, 64, 0x40000000, 0x40020000, 0x40010000)
        self.assertEqual(worker.matmat_mul_core.call_args.kwargs['B_DRAM_ADDR'], 0x40020000)
        self.assertEqual(engine._controller_replicas.address.call_count, 1)

    def test_act_sram_convolution_streams_private_weight_strip(self):
        engine = object.__new__(act.ACT_UnifiedEngine)
        engine._base_addr, engine._params_dram_base = 0x2000000, 0
        engine._controller_weight_end = 1 << 24
        engine._controller_replicas = Mock()
        engine._controller_replicas.address.return_value = 0x180000000
        engine.SNE, engine.EOFF, engine._flops = 1, 0, 0
        engine.KSPLIT_SPILL = [0x40010000]
        engine.ZERO_ROW, engine.ONES_BLOCK = 0x100, 0x200
        worker = Mock(_base_addr=engine._base_addr)
        grid = act.Grid(1, 1, 64)
        conv = {'groups': [([4], 0x10000, 128)], 'N_s': 64}
        engine.conv3x3_rows(worker, 0, 0x40000000, 0x40020000, grid, grid, conv, False)
        engine._controller_replicas.address.assert_called_once_with(0, 0x10000, 16384)
        self.assertIn(((0x180000000, 0x80000, 8192), {}),
                      [(call.args, call.kwargs) for call in worker.accelerator_memory_to_sram.call_args_list])

    def test_verapulse_copies_six_projections_but_keeps_norms_shared(self):
        engine = object.__new__(vera.VeraPulse_UnifiedEngine)
        engine._cfg = {'vision': {'hidden_size': 64, 'intermediate_size': 128}}
        engine._num_engines = lambda stage: 8
        layer = {f'{name}_weight': index * 0x10000 for index, name in enumerate(
            ('q', 'k', 'v', 'o', 'fc1', 'fc2', 'ln1', 'ln2'))}
        engine.vis_layer_addrs = [layer]
        primary = self.primary()
        engine.dma_read, engine.dma_write = primary.dma_read, primary.dma_write
        engine._materialize_vision_controller_weights()
        self.assertEqual(len(engine._vision_replicas.records), 6)
        self.assertEqual(primary.dma_read.call_count, 6)
        for index, layers in enumerate(engine._vision_private_layers):
            self.assertEqual(layers[0]['ln1_weight'], layer['ln1_weight'])
            self.assertEqual(layers[0]['ln2_weight'], layer['ln2_weight'])
            for name in ('q', 'k', 'v', 'o', 'fc1', 'fc2'):
                self.assertGreaterEqual(layers[0][f'{name}_weight'], 4 << 30)
        engine._materialize_vision_controller_weights()
        self.assertEqual(primary.dma_read.call_count, 6)

    def test_kokoro_tap_emission_uses_engine_private_weights(self):
        generator = object.__new__(kokoro.GeneratorFPGA)
        engines = [Mock(), Mock()]
        generator.ue = engines[0]
        generator._level_of = lambda rows: 0
        def region(rows, body):
            for index, ue in enumerate(engines):
                body(SimpleNamespace(engine_idx=index, unsafe_ue=ue))
        generator.sched = SimpleNamespace(num_engines=2, sharded_region=region)
        pool = Mock()
        pool.copy.return_value = [0x100000000, 0x140000000]
        with patch.object(kokoro, '_CONTROLLER_REPLICAS', [pool]), \
                patch.object(kokoro, '_dyn_matmul') as matmul, \
                patch.dict('os.environ', {'KOKORO_SHARD_KINDS': 'conv'}):
            generator._tap_matmuls(64, 64, 64, [(0x40000000, 0x1000, 0x40010000, None, None)])
        pool.copy.assert_called_once_with(0x1000, 8192)
        self.assertEqual([call.kwargs['B_DRAM_ADDR'] for call in matmul.call_args_list],
                         [0x100000000, 0x140000000])
        self.assertTrue(all(call.kwargs['OUTPUT_DRAM_ADDR'] == 0x40010000
                            for call in matmul.call_args_list))


if __name__ == '__main__':
    unittest.main()
