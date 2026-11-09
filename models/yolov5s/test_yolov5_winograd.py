"""Winograd numerical, memory-boundary, and artifact regression tests; no FPGA."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parents[1]):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import user_dma_core as core
import yolov5_winograd as wino
from yolov5_artifact import _pack_codes
from yolov5_common import quantize_conv_if4, dequantize_conv
from yolov5_precompiled import (_ImageBuilder, _PackedMap, _unpack_codes,
                                compile_precompiled_hardware, validate_precompiled_hardware)


class TensorEngine:
    """Execute the emitter's DMA/SRAM arithmetic against bounded BF16 memory.

    This exercises addressing and transform signs without relying on FPGA
    instruction encoding; it is not a cycle or hardware-precision simulator.
    """
    def __init__(self, regions):
        self.regions = regions
        self.sram = torch.full((0x100000 // 2,), float('nan'), dtype=torch.bfloat16)

    def memory(self, address, count):
        for base, tensor in self.regions:
            offset = (address - base) // 2
            if address >= base and address % 2 == 0 and offset + count <= tensor.numel():
                return tensor[offset:offset + count]
        raise AssertionError(f'Out-of-bounds memory access: {address:#x}, {count} elements')

    def accelerator_memory_to_sram(self, address, sram, count,
                                    stride_bytes_per_chunk=0, stride_jump_bytes=0):
        chunk = stride_bytes_per_chunk // 2 or count
        self.assert_sram(sram, count)
        for i in range(0, count, chunk):
            n = min(chunk, count - i)
            src = address + (i // chunk) * (stride_jump_bytes or chunk * 2)
            self.sram[sram // 2 + i:sram // 2 + i + n] = self.memory(src, n)

    def sram_to_accelerator_memory(self, sram, address, count,
                                    stride_bytes_per_chunk=0, stride_jump_bytes=0):
        chunk = stride_bytes_per_chunk // 2 or count
        self.assert_sram(sram, count)
        for i in range(0, count, chunk):
            n = min(chunk, count - i)
            dst = address + (i // chunk) * (stride_jump_bytes or chunk * 2)
            self.memory(dst, n).copy_(self.sram[sram // 2 + i:sram // 2 + i + n])

    def assert_sram(self, address, count):
        assert address % 128 == 0
        assert (address % 0x80000) + count * 2 <= 0x80000

    def arithmetic(self, a, b, dst, count, subtract):
        assert a // 0x80000 != b // 0x80000
        if subtract:
            assert a < 0x80000 <= b
        self.assert_sram(a, count)
        self.assert_sram(b, count)
        self.assert_sram(dst, count)
        av, bv = self.sram[a // 2:a // 2 + count].float(), self.sram[b // 2:b // 2 + count].float()
        self.sram[dst // 2:dst // 2 + count] = av - bv if subtract else av + bv

    def eltwise_add_core(self, a, b, dst, count):
        self.arithmetic(a, b, dst, count, False)

    def eltwise_sub_core(self, a, b, dst, count):
        self.arithmetic(a, b, dst, count, True)

    def matmat_mul_core(self, *, M, K, N, A_DRAM_ADDR, B_DRAM_ADDR, OUTPUT_DRAM_ADDR,
                        C_DRAM_ADDR=None, bias_mode=None, **_):
        a = self.memory(A_DRAM_ADDR, M * K).view(M, K).float()
        b = self.memory(B_DRAM_ADDR, N * K).view(N, K).float()
        result = a @ b.T
        if C_DRAM_ADDR is not None:
            c = self.memory(C_DRAM_ADDR, M * N if bias_mode == 'full_matrix' else N).float()
            result += c.view(M, N) if bias_mode == 'full_matrix' else c
        self.memory(OUTPUT_DRAM_ADDR, M * N).copy_(result.flatten())

    def activation_core(self, *, M, N, A_DRAM_ADDR, OUTPUT_DRAM_ADDR, activation, **_):
        assert activation == 'silu'
        self.memory(OUTPUT_DRAM_ADDR, M * N).copy_(F.silu(self.memory(A_DRAM_ADDR, M * N).float()))

    def alloc_isa_reg(self):
        return 1

    def release_isa_reg(self):
        pass

    def generate_instruction_add_set(self, *_):
        pass


def fixture(height=12, width=12, channels=128, physical=128, activate=True, bias=True):
    torch.manual_seed(17)
    conv = torch.nn.Conv2d(channels, channels, 3, padding=1, bias=bias)
    quantized = quantize_conv_if4(conv, None, include_dequant=True)
    entry = dict(precision='if4', layout='channel', codes_shape=list(quantized.codes.shape),
                 codes_packed=_pack_codes(quantized.codes, 'if4')[0], block_scales=quantized.block_scales,
                 bias=quantized.bias if bias else torch.empty(0, dtype=torch.bfloat16))
    mapping = tuple(range(64)) + tuple(range(physical - channels + 64, physical))
    source = _PackedMap((channels, height, width), physical, mapping, address=0x80000000)
    dest = _PackedMap((channels, height, width), channels, tuple(range(channels)), address=0xb0000000)
    op = dict(op='conv', name='conv', inputs=['input'], output='output',
              output_shape=[channels, height, width], stride=1, pad=1, dilation=1, activate=activate)
    return quantized, entry, source, dest, op


class WinogradTests(unittest.TestCase):
    def test_matches_direct_convolution_with_halo_bias_activation_and_channel_gaps(self):
        for physical, activate, bias, h, w, c in [
                (128, False, True, 8, 10, 128), (192, True, True, 7, 9, 128),
                (128, True, False, 8, 10, 128), (256, True, True, 4, 128, 256)]:
            with self.subTest(physical=physical, activate=activate, bias=bias):
                quantized, entry, src, dst, op = fixture(height=h, width=w, channels=c, physical=physical, activate=activate, bias=bias)
                image = _ImageBuilder(0x90000000, 0xa0000000)
                zero = image.allocate(torch.zeros(1024, dtype=torch.bfloat16), alignment=128)
                plan = wino.prepare(op, entry, src, dst, image, _unpack_codes)
                x = torch.randn(src.channels, src.height, src.width).to(torch.bfloat16)
                packed = torch.zeros(src.height, src.width, physical, dtype=torch.bfloat16)
                packed[:, :, list(src.channel_map)] = x.permute(1, 2, 0)
                # One tile row per group exercises reuse/overwrite of the plane buffers.
                budget = 16 * ((src.width + 1) // 2) * (physical + dst.channels) * 2
                scratch = torch.full(((wino.padded_bytes(src) + budget) // 2,), float('nan'), dtype=torch.bfloat16)
                output = torch.full((dst.size_bytes // 2,), float('nan'), dtype=torch.bfloat16)
                engine = TensorEngine([(src.address, packed.flatten()), (dst.address, output),
                    (0xc0000000, scratch), (image.base, core.tensor_from_le_bytes(bytes(image.data), torch.bfloat16))])
                with patch.object(wino, 'SCRATCH_BYTES', budget):
                    wino.emit(engine, plan, 0xc0000000, zero, 2048)
                actual = output.view(src.height, src.width, dst.channels).permute(2, 0, 1).float()
                expected = F.conv2d(x.float()[None], dequantize_conv(quantized).float(),
                                     quantized.bias.float() if bias else None, padding=1)[0]
                if activate:
                    expected = F.silu(expected)
                self.assertTrue(torch.isfinite(actual).all())
                relative_error = (actual - expected).norm() / expected.norm()
                self.assertLess(relative_error.item(), .02)

    def test_selection_rejects_incompatible_shapes(self):
        _, entry, src, dst, op = fixture()
        self.assertTrue(wino.eligible(op, entry, src, dst))
        for key, value in [('stride', 2), ('dilation', 2), ('pad', 0)]:
            self.assertFalse(wino.eligible(dict(op, **{key: value}), entry, src, dst))
        self.assertFalse(wino.eligible(op, dict(entry, codes_shape=[128, 128, 1, 1]), src, dst))
        small = _PackedMap((128, 8, 8), 128, tuple(range(128)))
        self.assertFalse(wino.eligible(op, entry, small, small))
        odd = _PackedMap((128, 17, 17), 128, tuple(range(128)))
        self.assertTrue(wino.eligible(op, entry, odd, odd))
        self.assertIsNotNone(wino._conv_wino_chunk(64, 256, 256))
        self.assertLess(wino._conv_wino_chunk(64, 256, 256), 64)

    def test_compiled_manifest_and_scratch_are_validated(self):
        _, entry, src, _, op = fixture()
        payload = dict(model=dict(input_shape=list(src.logical_shape)), operations=[op],
                       weights={'conv': entry}, head_outputs=['output'])
        with contextlib.redirect_stdout(io.StringIO()):
            hardware = compile_precompiled_hardware(payload, winograd=True)
        self.assertEqual(hardware['winograd_layers'], ['conv'])
        self.assertGreaterEqual(hardware['scratch_bytes'], wino.SCRATCH_BYTES + wino.padded_bytes(src))
        validate_precompiled_hardware(payload, hardware)
        with self.assertRaisesRegex(RuntimeError, 'layer manifest'):
            validate_precompiled_hardware(payload, dict(hardware, winograd_layers=['missing']))
        with self.assertRaisesRegex(RuntimeError, 'scratch'):
            validate_precompiled_hardware(payload, dict(hardware, scratch_bytes=128))


if __name__ == '__main__':
    unittest.main()
