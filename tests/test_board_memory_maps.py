"""Board/controller allocation contracts without opening an FPGA device."""

import unittest
from unittest.mock import patch

import multi_engine_shard as shard


GIB = 1 << 30
MIB = 1 << 20


class BoardMemoryMaps(unittest.TestCase):
    def board(self, cores, gib):
        return patch.multiple(shard.user_dma_core,
                              ANDROMEDA_CORE_COUNT=cores,
                              AVAILABLE_DRAM_SIZE_GB=gib,
                              DRAM_START_ADDR=2 * GIB)

    def assert_disjoint(self, windows, model_base=None, model_bytes=2 * GIB):
        spans = [(base, base + size) for base, size in windows]
        if model_base is not None:
            spans.append((model_base, model_base + model_bytes))
        spans.sort()
        for left, right in zip(spans, spans[1:]):
            self.assertLessEqual(left[1], right[0])

    def test_kintex_benchmark_windows_select_different_ddr_channels(self):
        with self.board(2, 4):
            windows = shard.board_private_windows(2)
        self.assertEqual([w.base for w in windows], [0, 2 * GIB])
        for i, window in enumerate(windows):
            self.assertEqual(window.primary_bytes, 512 * MIB)
            self.assertEqual(window.base >> 31, i)
            self.assertEqual((window.base + window.primary_bytes - 1) >> 31, i)

    def test_kintex_legacy_aperture_is_preserved(self):
        with self.board(2, 2):
            windows = shard.board_private_windows(2)
        self.assertEqual([w.base for w in windows], [0x80000000, 0x90000000])

    def test_kintex_model_and_allocator_tails_share_no_addresses(self):
        with self.board(2, 4):
            windows, model_base = shard.model_multicore_layout(2)
            arena = shard.PrivateArena(2, windows=windows)
        self.assertEqual(windows, [(0, 512 * MIB), (3 * GIB, 512 * MIB)])
        self.assertEqual(model_base, GIB)
        self.assert_disjoint(windows, model_base)
        for i, region in enumerate(arena.regions):
            self.assertEqual(region.weight_base >> 31, i)
            self.assertEqual((region.tensor_base + arena.tensor_bytes - 1) >> 31, i)
            self.assertLessEqual(region.weight_limit, region.isa_base)
            self.assertLessEqual(region.isa_base + arena.isa_bytes, region.tensor_base)
        # Merely enabling this board must not move a generic low-arena caller
        # whose model still occupies [2,4) GiB into that live shared model.
        with self.board(2, 4):
            legacy = shard.PrivateArena(2)
        self.assertEqual([r.base for r in legacy.regions], [0, GIB])

    def test_kintex_reservation_cannot_silently_alias_the_model(self):
        with self.board(2, 4):
            with self.assertRaisesRegex(ValueError, "reserved region"):
                shard.board_private_windows(2, reserve=(2 * GIB, 2 * GIB))
            windows, base = shard.model_multicore_layout(
                2, preferred_model_base=2 * GIB)
        self.assertEqual(base, GIB)
        self.assert_disjoint(windows, base)

    def test_u50_ports_keep_primary_and_secondary_stack_ownership(self):
        # build_alveo.tcl: engine SAXI ports 00,04,08,06,02,14,12,10.
        ports = [0, 4, 8, 6, 2, 14, 12, 10]
        with self.board(8, 8):
            for count in range(1, 9):
                board = shard.board_private_windows(count)
                windows, model_base = shard.model_multicore_layout(count)
                self.assertEqual(model_base, 6 * GIB)
                for i, window in enumerate(board):
                    expected = ports[i] // 2 * 512 * MIB
                    self.assertEqual(window.segments,
                                     ((expected, 512 * MIB),
                                      (expected + 4 * GIB, 512 * MIB)))
                    self.assertEqual(windows[i], (expected, 512 * MIB))
                self.assert_disjoint(windows, model_base)

    def test_u55c_native_stack_and_controller_match_each_engine_port(self):
        # build_alveo_u55c.tcl: paired SAXI ports map to the same MC.
        ports = [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7]
        with self.board(12, 16):
            for count in range(1, 13):
                board = shard.board_private_windows(count)
                windows, model_base = shard.model_multicore_layout(count)
                self.assertEqual(model_base, 6 * GIB)
                self.assert_disjoint(windows, model_base)
                for i, ((base, size), physical) in enumerate(zip(windows, board)):
                    self.assertEqual(size, GIB)
                    self.assertEqual((base >> 30) & 7, ports[i] // 2)
                    self.assertEqual((physical.base >> 30) & 7, ports[i] // 2)
                    self.assertLessEqual(base + size, 16 * GIB)
                self.assertEqual(len({base >> 30 for base, _ in windows}), count)

    def test_u55c_model_reserve_only_moves_the_colliding_engine(self):
        with self.board(12, 16):
            before = shard.board_private_windows(12)
            after = shard.board_private_windows(12, reserve=(6 * GIB, 2 * GIB))
        for i in range(12):
            self.assertEqual(after[i].base, 14 * GIB if i == 6 else before[i].base)

    def test_u55c_larger_reserve_keeps_unique_controllers_even_if_mc_must_move(self):
        with self.board(12, 16):
            windows = shard.board_private_windows(12, reserve=(0, 4 * GIB))
        self.assert_disjoint([(w.base, w.primary_bytes) for w in windows], 0, 4 * GIB)
        self.assertEqual(len({w.base >> 30 for w in windows}), 12)

    def test_u55c_8gib_preserves_capacity_without_claiming_extra_controllers(self):
        with self.board(12, 8):
            for count in range(1, 13):
                windows, model_base = shard.model_multicore_layout(count)
                self.assertEqual(model_base, 6 * GIB)
                self.assertEqual(len(windows), count)
                self.assert_disjoint(windows, model_base)
                expected_size = GIB if count <= 6 else 512 * MIB
                self.assertTrue(all(size == expected_size for _, size in windows))
                controllers = {base >> 30 for base, _ in windows}
                self.assertEqual(len(controllers), min(count, 6))
                self.assertTrue(all(base + size <= 6 * GIB for base, size in windows))
            windows, _ = shard.model_multicore_layout(12)
        for core in list(range(6)) + list(range(8, 12)):
            self.assertEqual(windows[core][0] >> 30, core % 8)

    def test_invalid_layouts_fail_before_allocating(self):
        with self.board(2, 4):
            for count in (0, -1, 3):
                with self.subTest(count=count), self.assertRaises(ValueError):
                    shard.model_multicore_layout(count)
            for size in (0, -MIB, MIB, 5 * GIB, 4 * GIB):
                with self.subTest(size=size), self.assertRaises(ValueError):
                    shard.model_multicore_layout(2, model_bytes=size)
            for base in (-GIB, MIB):
                with self.subTest(base=base), self.assertRaises(ValueError):
                    shard.model_multicore_layout(2, preferred_model_base=base)
        for cores, gib in ((2, 2), (8, 4), (12, 4), (4, 8)):
            with self.board(cores, gib), self.assertRaises(ValueError):
                shard.model_multicore_layout(2)


if __name__ == "__main__":
    unittest.main()
