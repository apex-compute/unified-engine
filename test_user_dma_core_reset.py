"""Focused regressions for UnifiedEngine reset-time DRAM probing."""

import contextlib
import io
import unittest
from unittest import mock

import torch

import user_dma_core


class UnifiedEngineResetTests(unittest.TestCase):
    @staticmethod
    def _initialization_fixture():
        engine = object.__new__(user_dma_core.UnifiedEngine)
        engine.device = "cpu"
        engine.hw_version = None
        engine.get_hardware_version = mock.Mock(return_value=0x12345678)
        engine.user_read_reg32 = mock.Mock(return_value=0xDEADBEEF)
        return engine

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
