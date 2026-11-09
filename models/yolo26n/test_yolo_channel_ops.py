"""Numerical and poisoned-memory checks for resident channel optimization."""
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yolo26_common import dense_conv
from yolo26_artifact import CompileBackend
from yolov5_artifact import TensorSpec, _unpack_codes
from yolov5_common import QuantizedConv, dequantize_conv
from yolov5_precompiled import _PackedMap
import user_dma_core as core
import yolo_channel_ops as ops


class MemoryEngine:
    def __init__(self):
        self.dram = bytearray([0xff]) * (1 << 18)
        self.sram = bytearray(1 << 18)

    def accelerator_memory_to_sram(self, address, sram, unused, *, memcpy_length_bytes,
                                   stride_bytes_per_chunk=0, stride_jump_bytes=0):
        chunk = stride_bytes_per_chunk or memcpy_length_bytes
        jump = stride_jump_bytes or chunk
        for i in range(memcpy_length_bytes // chunk):
            self.sram[sram + i * chunk:sram + (i + 1) * chunk] = self.dram[address + i * jump:address + i * jump + chunk]

    def sram_to_accelerator_memory(self, sram, address, unused, *, memcpy_length_bytes,
                                   stride_bytes_per_chunk=0, stride_jump_bytes=0):
        chunk = stride_bytes_per_chunk or memcpy_length_bytes
        jump = stride_jump_bytes or chunk
        for i in range(memcpy_length_bytes // chunk):
            self.dram[address + i * jump:address + i * jump + chunk] = self.sram[sram + i * chunk:sram + (i + 1) * chunk]


class ChannelOptimizationTests(unittest.TestCase):
    def test_channel_dma_copies_exact_values_and_clears_poisoned_padding(self):
        for start, count, channels in [(32, 32, 64), (64, 64, 64), (0, 128, 128)]:
            with self.subTest(start=start, count=count):
                engine = MemoryEngine()
                source = _PackedMap((128, 3, 7), 128, tuple(range(128)), 4096)
                destination = _PackedMap((channels, 3, 7), channels, tuple(range(channels)), 32768)
                values = torch.arange(3 * 7 * 128, dtype=torch.int16).view(3, 7, 128)
                engine.dram[source.address:source.address + source.size_bytes] = values.numpy().tobytes()
                engine.dram[:512] = bytes(512)
                ops.copy_channels(engine, source, destination, start=start, count=count,
                                  zero_address=0, zero_bytes=512)
                actual = torch.frombuffer(engine.dram, dtype=torch.int16,
                    count=destination.size_bytes // 2, offset=destination.address).view(3, 7, channels)
                expected = torch.zeros_like(actual)
                expected[:, :, :count] = values[:, :, start:start + count]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_sub_beat_copy_rejected(self):
        source = _PackedMap((64, 2, 2), 64, tuple(range(64)))
        with self.assertRaisesRegex(ValueError, '32-channel'):
            ops.copy_channels(MemoryEngine(), source, source, start=0, count=16,
                              zero_address=0, zero_bytes=512)

    def test_depthwise_blocks_keep_original_quantized_kernel(self):
        torch.manual_seed(17)
        for channels in (80, 128):
            conv = dense_conv(torch.nn.Conv2d(channels, channels, 3, padding=1, groups=channels))
            backend = CompileBackend(optimize=True)
            backend.conv_tensor('dw', TensorSpec('input', (channels, 5, 7)), conv, None, activate=False)
            weight = backend.weights['dw']
            source = _PackedMap((channels, 5, 7), (channels + 63) // 64 * 64, tuple(range(channels)))
            groups = []
            def capture(operation, entry, source, destination, image):
                groups.append(entry)
                return {}
            ops.prepare_depthwise(backend.operations[0], weight, source, source, None, 0,
                                  unpack_codes=_unpack_codes, prepare_conv=capture)
            def prepared(entry):
                return QuantizedConv(codes=_unpack_codes(entry['codes_packed'], entry['codes_shape'], 'if8'),
                    block_scales=entry['block_scales'], bias=entry['bias'], data_type=core.TYPE.IF8, gather=False)
            original = prepared(weight)
            x = torch.randn(1, channels, 5, 7)
            expected = F.conv2d(x, dequantize_conv(original).float(), original.bias.float(), padding=1)
            pieces = []
            for i, entry in enumerate(groups):
                p = prepared(entry)
                pieces.append(F.conv2d(x[:, i * 64:i * 64 + p.codes.shape[1]],
                                      dequantize_conv(p).float(), p.bias.float(), padding=1))
            torch.testing.assert_close(torch.cat(pieces, dim=1), expected, atol=1e-6, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
