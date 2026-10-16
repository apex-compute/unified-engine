"""Regression checks for native SRAM copies and their address bounds."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bigcodec_device import sram_copy, shared, udc
from test_bigcodec_quantizer import MemoryEngine


class SRAMCopyTest(unittest.TestCase):
    def test_every_finite_encoding_keeps_its_lane_in_either_bank(self):
        values = torch.arange(65536).to(torch.uint16).view(torch.bfloat16)
        values[~torch.isfinite(values)] = 0
        for source, destination in ((0, 0x30000), (0, 0x80000),
                                    (0x10000, 0x90000)):
            with self.subTest(source=source, destination=destination):
                engine = MemoryEngine()
                engine.sram.fill_(float('nan'))
                engine.sram_view(source, values.numel()).copy_(values)
                before = engine.sram.clone()
                sram_copy(engine, source, destination, values.numel())
                torch.testing.assert_close(
                    engine.sram_view(destination, values.numel()).view(torch.int16),
                    values.view(torch.int16), rtol=0, atol=0)
                # A copy may change only its destination, including when the
                # source has a nonzero SRAM base. Hardware tiny-subnormal
                # flushing is checked separately by native_sram_smoke.py.
                mask = torch.ones_like(engine.sram, dtype=torch.bool)
                mask[destination // 2:destination // 2 + values.numel()] = False
                torch.testing.assert_close(engine.sram[mask], before[mask],
                                           rtol=0, atol=0, equal_nan=True)

    def test_native_copy_avoids_broken_one_tap_maxpool(self):
        for destination in (0x30000, 0x80000):
            engine = shared._WholeGraphEngine(0x98000000)
            with contextlib.redirect_stdout(io.StringIO()):
                engine.start_capture()
                sram_copy(engine, 0, destination, 65536)
                engine.stop_capture()
            self.assertEqual(engine.capture_count, 1)
            descriptor = engine.capture_buffer[0]
            self.assertEqual((descriptor.words[0] >> 8) & 15,
                             udc.INSTRUCTION_UE_OP)
            self.assertEqual(udc._inst_desc_bits(descriptor.words, 172, 175),
                             udc.UE_MODE.MUL_BROADCAST)
            self.assertEqual(engine._isa_reg_counter, 1)
            self.assertEqual(engine._inst_ptr_counter, 1)

    def test_partial_overlap_and_bank_boundaries_remain_rejected(self):
        for source, destination, elements in ((0, 128, 128),
                (0x80000, 0, 64), (0, 0xFFF80, 128),
                (0x7FF00, 0x80000, 128), (0, 0x80000, 262144)):
            with self.subTest(source=source, destination=destination, elements=elements):
                with self.assertRaises(ValueError):
                    sram_copy(MemoryEngine(), source, destination, elements)


if __name__ == '__main__':
    unittest.main()
