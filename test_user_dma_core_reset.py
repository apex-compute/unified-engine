"""Focused regressions for reset-time hardware compatibility and DRAM probing."""

import contextlib
import io
import unittest
from unittest import mock

import torch

import user_dma_core


class UnifiedEngineResetTests(unittest.TestCase):
    @staticmethod
    def _initialization_fixture(*, queued=False, version=0xF9303071,
                                hw_info=0x80214D40):
        engine = object.__new__(user_dma_core.UnifiedEngine)
        engine.device = "cpu"
        engine.hw_version = None
        engine.conv_geometry_mode = (
            user_dma_core.CONV_GEOMETRY_QUEUE_CONFIG if queued
            else user_dma_core.CONV_GEOMETRY_LIVE_CSR)
        engine.get_hardware_version = mock.Mock(return_value=version)
        engine.user_read_reg32 = mock.Mock(
            side_effect=lambda address: (
                hw_info if address == user_dma_core.UE_HW_INFO_ADDR else 0xDEADBEEF))
        engine.dma_write = mock.Mock()
        engine.dma_read = mock.Mock()
        return engine

    def test_queued_convolution_accepts_stamped_256_and_512_bit_builds(self):
        for version in (0xB97C477A, 0x12345678):
            for hw_info in (0x80214D40, 0xA0214D40):
                with self.subTest(version=hex(version), hw_info=hex(hw_info)):
                    engine = self._initialization_fixture(
                        queued=True, version=version, hw_info=hw_info)
                    with contextlib.redirect_stdout(io.StringIO()):
                        engine.init_unified_engine(run_dram_self_test=False)
                    engine.user_read_reg32.assert_any_call(user_dma_core.UE_HW_INFO_ADDR)
                    engine.dma_write.assert_not_called()
                    engine.dma_read.assert_not_called()

    def test_queued_convolution_rejects_unresponsive_build_register(self):
        for version in (0, 0xFFFFFFFF):
            with self.subTest(version=hex(version)):
                engine = self._initialization_fixture(queued=True, version=version)
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "Invalid FPGA build stamp"):
                        engine.init_unified_engine()
                engine.dma_write.assert_not_called()
                engine.dma_read.assert_not_called()

    def test_queued_convolution_rejects_invalid_hardware_info_before_dma(self):
        invalid_info = {
            "unresponsive": 0xFFFFFFFF,
            "zero": 0,
            "unsupported_axi_width": 0xC0214D40,
            "zero_cores": 0x80014D40,
            "zero_clock": 0x80200000,
            "queue_disabled": 0x00214D40,
        }
        for description, hw_info in invalid_info.items():
            with self.subTest(description=description):
                engine = self._initialization_fixture(
                    queued=True, version=0xB97C477A, hw_info=hw_info)
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(RuntimeError):
                        engine.init_unified_engine()
                engine.dma_write.assert_not_called()
                engine.dma_read.assert_not_called()

    def test_legacy_mode_keeps_release_build_check(self):
        engine = self._initialization_fixture(version=0xB97C477A)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "expected 0xf9303071"):
                engine.init_unified_engine()
        engine.dma_write.assert_not_called()
        engine.dma_read.assert_not_called()

    def test_initialization_can_skip_dram_dma_probe(self):
        engine = self._initialization_fixture()
        engine.dma_write = mock.Mock()
        engine.dma_read = mock.Mock()

        with contextlib.redirect_stdout(io.StringIO()):
            engine.init_unified_engine(run_dram_self_test=False)

        engine.dma_write.assert_not_called()
        engine.dma_read.assert_not_called()
        engine.get_hardware_version.assert_called_once_with()
        self.assertGreater(engine.user_read_reg32.call_count, 0)

    def test_initialization_keeps_dram_dma_probe_by_default(self):
        engine = self._initialization_fixture()
        written = {}

        def dma_write(_device, address, value, size):
            written["address"] = address
            written["value"] = value.clone()
            written["size"] = size
            return size

        def dma_read(_device, address, value, size):
            self.assertEqual(address, written["address"])
            self.assertEqual(size, written["size"])
            value.copy_(written["value"])
            return size

        engine.dma_write = mock.Mock(side_effect=dma_write)
        engine.dma_read = mock.Mock(side_effect=dma_read)

        with contextlib.redirect_stdout(io.StringIO()):
            engine.init_unified_engine()

        engine.dma_write.assert_called_once()
        engine.dma_read.assert_called_once()
        self.assertEqual(written["address"], user_dma_core.DRAM_START_ADDR)
        self.assertEqual(written["size"], 8192 * 2)
        self.assertEqual(written["value"].dtype, torch.uint16)

    def test_software_reset_forwards_probe_choice(self):
        engine = object.__new__(user_dma_core.UnifiedEngine)
        engine._inst_id = 17
        engine.write_reg32 = mock.Mock()
        engine.wait_queue = mock.Mock()
        engine.init_unified_engine = mock.Mock()

        with contextlib.redirect_stdout(io.StringIO()):
            engine.software_reset(run_dram_self_test=False)

        engine.write_reg32.assert_called_once_with(
            user_dma_core.UE_QUEUE_CTRL_ADDR, 0x80008000)
        engine.wait_queue.assert_called_once_with(1.0)
        engine.init_unified_engine.assert_called_once_with(
            run_dram_self_test=False)
        self.assertEqual(engine._inst_id, 0)

    def test_software_reset_keeps_probe_enabled_by_default(self):
        engine = object.__new__(user_dma_core.UnifiedEngine)
        engine._inst_id = 0
        engine.write_reg32 = mock.Mock()
        engine.wait_queue = mock.Mock()
        engine.init_unified_engine = mock.Mock()

        with contextlib.redirect_stdout(io.StringIO()):
            engine.software_reset()

        engine.init_unified_engine.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
