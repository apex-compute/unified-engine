"""Memory benchmark contracts without opening any FPGA device."""

import unittest
from unittest.mock import Mock, patch

import torch

import multi_engine_memory_test as benchmark


class MemoryRegions(unittest.TestCase):
    def windows(self, gib=4):
        with patch.multiple(benchmark.core, AVAILABLE_DRAM_SIZE_GB=gib,
                            ANDROMEDA_CORE_COUNT=2, DRAM_START_ADDR=0x80000000):
            return benchmark.shard.board_private_windows(2)

    def assert_disjoint(self, regions, size):
        spans = sorted((address, address + size)
                       for region in regions
                       for address in (region.source, region.destination))
        for previous, following in zip(spans, spans[1:]):
            self.assertLessEqual(previous[1], following[0])

    def test_private_and_split_have_equal_work_and_disjoint_buffers(self):
        size = 512 * 1024
        windows = self.windows()
        private = benchmark.memory_regions(windows, [0, 1], size, "private")
        split = benchmark.memory_regions(windows, [0, 1], size, "split")
        self.assertEqual([r.source for r in private], [0x08000000, 0x88000000])
        self.assertEqual([r.destination for r in private], [0x08080000, 0x88080000])
        self.assertEqual([r.source for r in split], [0x08000000, 0x08080000])
        self.assertEqual([r.destination for r in split], [0x08100000, 0x08180000])
        for regions in (private, split):
            self.assertEqual([r.engine for r in regions], [0, 1])
            self.assert_disjoint(regions, size)
            for region in regions:
                owner = windows[region.engine] if regions is private else windows[0]
                self.assertGreaterEqual(region.source, owner.base)
                self.assertLessEqual(region.destination + size,
                                     owner.base + benchmark.shard.ENGINE_PROGRAM_OFFSET)

    def test_legacy_two_gib_windows_keep_the_high_dram_origin(self):
        windows = self.windows(gib=2)
        private = benchmark.memory_regions(windows, [0, 1], 65536, "private")
        self.assertEqual([w.base for w in windows], [0x80000000, 0x90000000])
        self.assertEqual([r.source for r in private], [0x88000000, 0x98000000])
        split = benchmark.memory_regions(windows, [0, 1], 65536, "split")
        self.assertEqual([r.source for r in split], [0x88000000, 0x88010000])
        self.assert_disjoint(private, 65536)
        self.assert_disjoint(split, 65536)

    def test_two_gib_spacing_preserves_engine_identity_and_split_layout(self):
        windows = self.windows()
        size = 512 * 1024
        spacing = 2 << 30
        with patch.object(benchmark.core, "AVAILABLE_DRAM_SIZE_GB", 4):
            regions = benchmark.memory_regions(windows, [0, 1], size, "private", spacing)
            self.assertEqual([r.source for r in regions], [0x08000000, 0x88000000])
            self.assertEqual([r.destination for r in regions], [0x08080000, 0x88080000])
            self.assert_disjoint(regions, size)
            self.assertEqual(benchmark.memory_regions(windows, [1], size, "private", spacing),
                             [regions[1]])
            self.assertEqual(benchmark.memory_regions(windows, [0, 1], size, "split", spacing),
                             benchmark.memory_regions(windows, [0, 1], size, "split"))

    def test_custom_spacing_rejects_aliasing_isa_overlap_and_capacity_overflow(self):
        windows = self.windows()
        with patch.object(benchmark.core, "AVAILABLE_DRAM_SIZE_GB", 4):
            for spacing in (0, -128, 512 * 1024, (2 << 30) + 1, 4 << 30,
                            benchmark.shard.ENGINE_PROGRAM_OFFSET - benchmark.shard.ENGINE_TENSOR_OFFSET):
                with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                    benchmark.memory_regions(windows, [0, 1], 512 * 1024, "private", spacing)
        with patch.object(benchmark.core, "AVAILABLE_DRAM_SIZE_GB", 2), \
                self.assertRaisesRegex(ValueError, "capacity"):
            benchmark.memory_regions(self.windows(gib=2), [0, 1], 512 * 1024, "private", 2 << 30)

    def test_individual_engine_and_reordered_subset_use_selected_owner(self):
        windows = self.windows()
        individual = benchmark.memory_regions(windows, [1], 128, "private")
        self.assertEqual(individual, [benchmark.MemoryRegion(1, 0x88000000, 0x88000080)])
        split = benchmark.memory_regions(windows, [1, 0], 128, "split")
        self.assertEqual([r.engine for r in split], [1, 0])
        self.assertEqual([r.source for r in split], [0x88000000, 0x88000080])
        self.assert_disjoint(split, 128)

    def test_invalid_size_engine_selection_and_layout_are_rejected(self):
        windows = self.windows()
        cases = [([0, 1], size, "private")
                 for size in (0, -128, 127, benchmark.core.URAM_FULL_ELEMENTS * 2 + 128)]
        cases += [(engines, 128, "private") for engines in ([], [0, 0], [-1], [2])]
        cases += [([0, 1], 128, "shared")]
        for engine_ids, size, layout in cases:
            with self.subTest(engines=engine_ids, size=size, layout=layout):
                with self.assertRaises(ValueError):
                    benchmark.memory_regions(windows, engine_ids, size, layout)

    def test_small_window_cannot_spill_into_program_storage_or_out_of_window(self):
        windows = [benchmark.shard.EngineWindow(
            0, ((0, benchmark.shard.ENGINE_TENSOR_OFFSET + 128),))]
        with self.assertRaisesRegex(ValueError, "overlap ISA storage or leave its window"):
            benchmark.memory_regions(windows, [0], 128, "private")


class HaltAndTiming(unittest.TestCase):
    def test_idle_without_fresh_halt_and_busy_with_halt_both_time_out(self):
        for cause, busy in ((0, False), (benchmark.core.INT_CAUSE_HALT, True)):
            with self.subTest(cause=cause, busy=busy):
                ue = Mock(_base_addr=0x2000000)
                ue.read_reg32.return_value = cause
                ue.is_queue_busy.return_value = busy
                with patch.object(benchmark.time, "monotonic", side_effect=[0.0, 2.0]):
                    with self.assertRaises(TimeoutError):
                        benchmark._wait_for_halt(ue, timeout=1.0)

    def test_halt_and_idle_are_both_required_for_success(self):
        ue = Mock(_base_addr=0x2000000)
        ue.read_reg32.return_value = benchmark.core.INT_CAUSE_HALT
        ue.is_queue_busy.return_value = False
        benchmark._wait_for_halt(ue, timeout=0)

    def measured_engines(self, latency=23.5):
        events = []
        engines = []

        class Engine:
            def __init__(self, name, index):
                self.name = name
                self._base_addr = 0x2000000 + index * 0x10000
                self.flag = True  # Simulate sticky completion from the previous run.
                self.program = None
                self.clear_halted = False

            def write_reg32(self, register, value):
                if register != benchmark.core.UE_INT_REG or value != 0:
                    raise AssertionError("launch must acknowledge the previous interrupt")
                events.append(("ack", self.name))

            def start_execute_from_dram(self, address):
                self.program = address
                events.append(("start", self.name, address))
                if address == "clear":
                    self.flag = False
                elif not all(not e.flag and e.clear_halted for e in engines):
                    raise AssertionError("measurement launched before all flags were cleared")

            def read_reg32(self, register):
                return benchmark.core.INT_CAUSE_HALT

            def is_queue_busy(self):
                events.append(("idle", self.name, self.program))
                if self.program == "clear":
                    self.clear_halted = True
                return False

            def report_latency_in_us(self):
                if self.name != "master":
                    raise AssertionError("worker latency includes arbitrary host launch skew")
                events.append(("latency", self.name))
                return latency

        engines.extend([Engine("master", 0), Engine("worker", 1)])
        return engines, events

    def test_measure_clears_sticky_flags_launches_workers_first_and_times_master(self):
        engines, events = self.measured_engines()
        self.assertEqual(benchmark._measure(engines, ["transfer", "transfer"],
                                            ["clear", "clear"]), 23.5)
        launches = [event[1] for event in events
                    if event[0] == "start" and event[2] == "transfer"]
        self.assertEqual(launches, ["worker", "master"])
        self.assertEqual(len([event for event in events if event[0] == "ack"]), 4)
        self.assertEqual(events[-1], ("latency", "master"))
        for name in ("master", "worker"):
            self.assertIn(("idle", name, "transfer"), events)

    def test_invalid_hardware_timer_values_are_rejected(self):
        for latency in (0, -1, float("inf"), float("nan")):
            with self.subTest(latency=latency):
                engines, _ = self.measured_engines(latency)
                with self.assertRaisesRegex(RuntimeError, "invalid hardware latency"):
                    benchmark._measure(engines, ["transfer", "transfer"], ["clear", "clear"])


class TransferFailures(unittest.TestCase):
    def test_default_kintex_cases_include_both_controllers_and_legacy_spacing(self):
        with patch.multiple(benchmark.core, HW_INFO_RAW=1, QUEUE_MODE_ENABLED=True,
                            ANDROMEDA_CORE_COUNT=2, AVAILABLE_DRAM_SIZE_GB=4), \
                patch.object(benchmark, "_run_case", return_value=[]) as run_case:
            benchmark.run_memory_comparison(sizes_kib=(64,), samples=1)
            layouts = [(call.args[1], call.args[2], call.args[-1])
                       for call in run_case.call_args_list]
            self.assertIn(([0, 1], "private", None), layouts)
            self.assertIn(([0, 1], "private", 512 << 20), layouts)
            self.assertIn(([0, 1], "split", None), layouts)

    def test_u55c_eight_gib_can_benchmark_all_twelve_disjoint_ports(self):
        with patch.multiple(benchmark.core, ANDROMEDA_CORE_COUNT=12, AVAILABLE_DRAM_SIZE_GB=8):
            windows = benchmark.benchmark_windows(12)
            self.assertEqual(len(windows), 12)
            regions = benchmark.memory_regions(windows, list(range(12)), 512 * 1024, "private")
            self.assertEqual(len({r.source for r in regions}), 12)
            self.assertTrue(all(r.destination + 512 * 1024 < 8 << 30 for r in regions))

    def test_short_data_upload_is_rejected(self):
        ue = Mock(h2c_device="mock-h2c")
        data = torch.zeros(64, dtype=torch.bfloat16)
        ue.dma_write.return_value = 126
        with self.assertRaisesRegex(IOError, "126/128 bytes"):
            benchmark._write_data(ue, 0x08000000, data)

    def test_short_program_upload_does_not_advance_allocator(self):
        ue = Mock()
        ue.get_program_dram_addr.return_value = 0x0F000000
        ue.get_capture_instruction_size_bytes.return_value = 64
        ue.write_captured_instructions_to_dram.return_value = 32
        with self.assertRaisesRegex(IOError, "32/64 bytes"):
            benchmark._upload_program(ue, lambda engine: None)
        ue.allocate_program_dram.assert_not_called()
        ue.start_execute_from_dram.assert_not_called()

    def test_short_or_corrupt_readback_cannot_be_reported_as_verified(self):
        windows = [benchmark.shard.EngineWindow(0, ((0, 0x20000000),))]
        for short in (True, False):
            with self.subTest(short=short):
                ue = Mock(_base_addr=0x2000000, h2c_device="mock-h2c", c2h_device="mock-c2h")
                ue.is_queue_busy.return_value = False
                ue.dma_write.side_effect = lambda device, address, data, size: size

                def readback(device, address, data, size):
                    data.zero_()
                    return size - 2 if short else size

                ue.dma_read.side_effect = readback
                with patch.object(benchmark.core, "UnifiedEngine", return_value=ue), \
                        patch.object(benchmark, "_upload_program", return_value=0), \
                        patch.object(benchmark, "_measure", return_value=1.0):
                    with self.assertRaises(IOError if short else AssertionError):
                        benchmark._run_case(windows, [0], "private", 1, 1, 1)

    def test_all_sizes_are_validated_before_running_any_case(self):
        with patch.multiple(benchmark.core, HW_INFO_RAW=1, QUEUE_MODE_ENABLED=True,
                            ANDROMEDA_CORE_COUNT=2, AVAILABLE_DRAM_SIZE_GB=4), \
                patch.object(benchmark, "_run_case") as run_case:
            with self.assertRaises(ValueError):
                benchmark.run_memory_comparison(sizes_kib=(64, 513))
            run_case.assert_not_called()

    def test_unbounded_or_empty_runs_are_rejected_before_running_any_case(self):
        with patch.multiple(benchmark.core, HW_INFO_RAW=1, QUEUE_MODE_ENABLED=True,
                            ANDROMEDA_CORE_COUNT=2, AVAILABLE_DRAM_SIZE_GB=4), \
                patch.object(benchmark, "_run_case") as run_case:
            for options in ({"iterations": 0}, {"iterations": 1025},
                            {"samples": 0}, {"sizes_kib": ()}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    benchmark.run_memory_comparison(**options)
            run_case.assert_not_called()

    def test_custom_spacing_validated_before_any_hardware_case(self):
        with patch.multiple(benchmark.core, HW_INFO_RAW=1, QUEUE_MODE_ENABLED=True,
                            ANDROMEDA_CORE_COUNT=2, AVAILABLE_DRAM_SIZE_GB=4), \
                patch.object(benchmark, "_run_case") as run_case:
            with self.assertRaisesRegex(ValueError, "capacity"):
                benchmark.run_memory_comparison(private_spacing_bytes=4 << 30)
            run_case.assert_not_called()


if __name__ == "__main__":
    unittest.main()
