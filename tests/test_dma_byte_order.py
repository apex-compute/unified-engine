"""DMA data stays little-endian on the wire; register words stay CPU-native.

The DMA "device" is a temporary file, so this runs without hardware. Patching
HOST_IS_LITTLE_ENDIAN to the opposite of sys.byteorder exercises the swap path
on whichever host runs the test.
"""

import contextlib
import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import user_dma_core as udc


class OfflineEngine(udc.UnifiedEngine):
    pass


class DMAByteOrderTests(unittest.TestCase):
    def setUp(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.engine = OfflineEngine(clock_period_ns=3.0, init_unified_engine=False)
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()
        self.device = tmp.name
        self.addCleanup(Path(self.device).unlink)

    def _device_bytes(self, size, address=0):
        with open(self.device, "rb") as f:
            f.seek(address)
            return f.read(size)

    def _check_round_trip(self, host_little):
        values = {
            torch.bfloat16: torch.tensor([1.0, -2.5, 3.0e8, -0.0], dtype=torch.bfloat16),
            torch.int16: torch.tensor([0x1234, -2, 0x7F00, 1], dtype=torch.int16),
            torch.int32: torch.tensor([0x12345678, -3], dtype=torch.int32),
            torch.float32: torch.tensor([1.5, -7.25], dtype=torch.float32),
            torch.uint8: torch.tensor([1, 2, 3, 250], dtype=torch.uint8),
        }
        with mock.patch.object(udc, "HOST_IS_LITTLE_ENDIAN", host_little):
            for dtype, tensor in values.items():
                with self.subTest(dtype=dtype, host_little=host_little):
                    size = tensor.numel() * tensor.element_size()
                    self.assertEqual(
                        self.engine.dma_write(self.device, 64, tensor, size), size)
                    # A host that believes it is little-endian sends memory
                    # as is; one that believes it is big-endian swaps each
                    # element.
                    native = tensor.contiguous().view(torch.uint8).numpy().tobytes()
                    elem = tensor.element_size()
                    expected = native if host_little else b"".join(
                        native[i:i + elem][::-1] for i in range(0, size, elem))
                    self.assertEqual(self._device_bytes(size, 64), expected)

                    back = torch.zeros_like(tensor)
                    self.assertEqual(
                        self.engine.dma_read(self.device, 64, back, size), size)
                    self.assertTrue(torch.equal(back.view(torch.uint8),
                                                tensor.view(torch.uint8)))

                    # Non-contiguous destination takes the staged path.
                    wide = torch.zeros((tensor.numel(), 2), dtype=dtype)
                    column = wide[:, 0]
                    self.assertEqual(
                        self.engine.dma_read(self.device, 64, column, size), size)
                    self.assertTrue(torch.equal(column.contiguous().view(torch.uint8),
                                                tensor.view(torch.uint8)))

    def test_round_trip_native_order(self):
        self._check_round_trip(sys.byteorder == "little")

    def test_round_trip_swapped_order(self):
        self._check_round_trip(sys.byteorder != "little")

    def test_real_host_writes_little_endian(self):
        tensor = torch.tensor([0x1234, 0x5678], dtype=torch.int16)
        self.engine.dma_write(self.device, 0, tensor, 4)
        self.assertEqual(self._device_bytes(4), bytes([0x34, 0x12, 0x78, 0x56]))
        self.engine.dma_write(self.device, 8, 0x11223344, 4)
        self.assertEqual(self._device_bytes(4, 8), bytes([0x44, 0x33, 0x22, 0x11]))
        self.engine.dma_write(self.device, 16, -2, 2)
        self.assertEqual(self._device_bytes(2, 16), bytes([0xFE, 0xFF]))
        array = np.array([0x0102, 0x0304], dtype=np.uint16)
        self.engine.dma_write(self.device, 24, array, 4)
        self.assertEqual(self._device_bytes(4, 24), bytes([0x02, 0x01, 0x04, 0x03]))
        array_be = array.astype(">u2")
        self.engine.dma_write(self.device, 32, array_be, 4)
        self.assertEqual(self._device_bytes(4, 32), bytes([0x02, 0x01, 0x04, 0x03]))

    def test_instruction_words_are_little_endian(self):
        inst = udc.Instructions()
        inst.words[0] = 0x11223344
        self.assertEqual(inst.get_bytes()[:4], bytes([0x44, 0x33, 0x22, 0x11]))

    def test_le_bytes_helpers(self):
        tensor = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)  # 0x3F80, 0xC000
        self.assertEqual(udc.tensor_le_bytes(tensor), bytes([0x80, 0x3F, 0x00, 0xC0]))
        decoded = udc.tensor_from_le_bytes(bytes([0x80, 0x3F, 0x00, 0xC0]), torch.bfloat16)
        self.assertTrue(torch.equal(decoded, tensor))
        self.assertEqual(udc.tensor_from_le_bytes(b"", torch.bfloat16).numel(), 0)
        for host_little in (True, False):
            with self.subTest(host_little=host_little), \
                 mock.patch.object(udc, "HOST_IS_LITTLE_ENDIAN", host_little):
                self.assertTrue(torch.equal(
                    udc.tensor_from_le_bytes(udc.tensor_le_bytes(tensor), torch.bfloat16),
                    tensor))

    def test_quant_lib_scales_are_little_endian(self):
        import quant_lib
        scales = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
        raw = quant_lib._scale_bytes(scales)
        self.assertEqual(raw, bytes([0x80, 0x3F, 0x00, 0xC0]))
        self.assertTrue(torch.equal(quant_lib._read_scale(raw, 1, 2), scales))

    def test_register_words_are_native(self):
        # The XDMA user device copies ioread32() results as CPU-native u32s.
        self.assertEqual(struct.pack(udc.REG32_FORMAT, 0x11223344),
                         (0x11223344).to_bytes(4, sys.byteorder))


if __name__ == "__main__":
    unittest.main()
