"""Mono SRAM Tanh, BF16 math, and measured native DMA row stepping."""

import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bigcodec_compile import (Graph, Operation, Tensor, WAVEFORM_ROW_LANES,
    _prepare_operation, emit_operation, emit_waveform_tanh, plan_memory, shared)
from bigcodec_lstm import TANH_CHUNK_ELEMENTS, pade_tanh, tanh_identity, tanh_scratch_bytes, udc
from test_bigcodec_lstm import MemoryEngine as LSTMMemoryEngine, MemoryImage


class MemoryEngine(LSTMMemoryEngine):
    """Native reads pack beats; strided writes step by at least one SRAM row.

    RK0x40519e0a probes with row-coded data found32/64/128-byte gathers exact,
    but sub-row scatters skipped source samples. Modeling that asymmetry makes
    the original compact-Tanh implementation fail the waveform tests below.
    """
    def __init__(self):
        super().__init__()
        self.transfers = []

    def view(self, address, count):
        assert address % 2 == 0 and count > 0
        for start, buffer in self.regions.items():
            offset = (address - start) // 2
            if address >= start and offset + count <= buffer.numel():
                return buffer[offset:offset + count]
        raise AssertionError(f'Out-of-bounds DRAM range {address:#x}, {count}')

    def _dma(self, source, destination, count, *, read, memcpy_length_bytes,
             stride_bytes_per_chunk, stride_jump_bytes):
        size = count * 2 if memcpy_length_bytes is None else memcpy_length_bytes
        chunk = stride_bytes_per_chunk or size
        jump = stride_jump_bytes or size
        sram = destination if read else source
        assert size > 0 and size % 128 == sram % 128 == 0
        assert chunk % 32 == jump % 32 == 0 and size % chunk == 0
        assert sram % 0x80000 + size <= udc.URAM_NEAR_FULL_SIZE
        for index in range(size // chunk):
            if read:
                value = self.view(source + index * jump, chunk // 2)
                self.sram[(destination + index * chunk) // 2:][:chunk // 2].copy_(value)
            else:
                step = max(128, chunk) if stride_bytes_per_chunk else chunk
                value = self.sram[(source + index * step) // 2:][:chunk // 2]
                self.view(destination + index * jump, chunk // 2).copy_(value)
        self.transfers.append((read, source, destination, size, chunk, jump))

    def accelerator_memory_to_sram(self, source, destination, count, *,
            memcpy_length_bytes=None, stride_bytes_per_chunk=None, stride_jump_bytes=None):
        self._dma(source, destination, count, read=True,
            memcpy_length_bytes=memcpy_length_bytes,
            stride_bytes_per_chunk=stride_bytes_per_chunk, stride_jump_bytes=stride_jump_bytes)

    def sram_to_accelerator_memory(self, source, destination, count, *,
            memcpy_length_bytes=None, stride_bytes_per_chunk=None, stride_jump_bytes=None):
        self._dma(source, destination, count, read=False,
            memcpy_length_bytes=memcpy_length_bytes,
            stride_bytes_per_chunk=stride_bytes_per_chunk, stride_jump_bytes=stride_jump_bytes)


def fixture(rows):
    engine = MemoryEngine()
    image = MemoryImage(engine)
    generator = torch.Generator().manual_seed(614)
    # Nonzero finite padding catches accidental publication of gathered lanes.
    values = (torch.randn(rows, 64, generator=generator) * 3).bfloat16()
    source = image.allocate(values)
    destination = image.allocate(torch.full_like(values, float('nan')))
    identity = image.allocate(torch.eye(64))
    zero = image.allocate(torch.zeros(udc.URAM_NEAR_FULL_SIZE // 2))
    mask = torch.zeros(TANH_CHUNK_ELEMENTS)
    mask[::WAVEFORM_ROW_LANES] = 1
    mask_address = image.allocate(mask)
    return engine, values, (source, destination, rows, identity, zero, mask_address)


class WaveformTanhTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_waveform_output_matches_bf16_math_and_zeros_all_padding(self):
        for rows in (1, 4, 63, 64, 65, 200, 256, 260, 516):
            with self.subTest(rows=rows):
                engine, values, args = fixture(rows)
                emit_waveform_tanh(engine, *args)
                output = engine.view(args[1], rows * 64).reshape(rows, 64)
                expected = pade_tanh(values[:, 0].float(), bf16=True).bfloat16()
                torch.testing.assert_close(output[:, 0], expected, rtol=0, atol=0)
                self.assertTrue(torch.isfinite(output).all())
                self.assertEqual(int(torch.count_nonzero(output[:, 1:])), 0)
                torch.testing.assert_close(engine.view(args[0], values.numel()).reshape_as(values),
                                           values, rtol=0, atol=0)
                reads = [item for item in engine.transfers
                         if item[0] and args[0] <= item[1] < args[0] + rows * 128]
                self.assertEqual(sum(item[3] for item in reads), rows * 128)
                self.assertTrue(all(item[4] == item[5] for item in engine.transfers))
                self.assertEqual(engine.native_calls.count(udc.LALU_MODE.MODE_RECIP), rows)

    def test_compiler_reserves_no_dram_scratch_and_prepares_padding_mask(self):
        operation = Operation('final', 'tanh', ('input',), 'wave', torch.nn.Tanh())
        graph = Graph(200, 1, {
            'input': Tensor('input', (200, 1), shared.INPUT_BASE),
            'wave': Tensor('wave', (200, 1))}, [operation], 'wave')
        plan_memory(graph)
        self.assertEqual(operation.scratch_bytes, 0)
        self.assertEqual(graph.scratch_bytes, 0)
        engine = MemoryEngine()
        image = MemoryImage(engine)
        _prepare_operation(graph, image, 0x90000000, 0x90010000, operation)
        mask = engine.view(operation.plan['mask_address'], TANH_CHUNK_ELEMENTS).reshape(-1, WAVEFORM_ROW_LANES)
        torch.testing.assert_close(mask[:, 0], torch.ones_like(mask[:, 0]), rtol=0, atol=0)
        self.assertEqual(int(torch.count_nonzero(mask[:, 1:])), 0)
        with patch('bigcodec_compile.emit_waveform_tanh') as emit:
            emit_operation(engine, graph, operation, 0x90000000, 0x90010000)
        self.assertEqual(emit.call_args.args[1:4],
            (shared.INPUT_BASE, shared.TENSOR_BASE, 200))

    def test_capture_has_one_halt_and_same_reciprocals_as_existing_sram_tanh(self):
        _, _, args = fixture(1024)
        reciprocals, instructions = [], []
        for waveform in (False, True):
            engine = shared._WholeGraphEngine(0x98000000)
            with patch.object(udc, 'UE_AXI_DATA_WIDTH_BITS', 256), contextlib.redirect_stdout(io.StringIO()):
                engine.start_capture()
                if waveform:
                    emit_waveform_tanh(engine, *args)
                else:
                    tanh_identity(engine, args[0], args[1], args[2] * 64, args[3],
                                  scratch_address=0xB0000000)
                engine.generate_instruction_halt()
                engine.stop_capture()
            self.assertEqual(udc.check_isa_jumps(engine.capture_buffer, 0x98000000, name='waveform tanh'), [])
            types = [(inst.words[0] >> 8) & 15 for inst in engine.capture_buffer]
            self.assertEqual(types.count(udc.INSTRUCTION_HALT), 1)
            self.assertNotIn(udc.INSTRUCTION_SWI, types)
            reciprocals.append(sum(
                udc._inst_desc_bits(inst.words, 172, 175) == udc.UE_MODE.BF16_DOT_PRODUCT
                and udc._inst_desc_bits(inst.words, 183, 185) == udc.LALU_MODE.MODE_RECIP.value
                for inst in engine.capture_buffer if (inst.words[0] >> 8) & 15 == udc.INSTRUCTION_UE_OP))
            instructions.append(engine.capture_count)
        self.assertEqual(reciprocals[0], reciprocals[1])
        self.assertGreater(reciprocals[1], 0)
        self.assertLess(instructions[1], 1.1 * instructions[0])

    def test_native_subrow_scatter_reproduces_the_rejected_layout(self):
        for chunk in (32, 64):
            rows = 64
            lanes = chunk // 2
            engine = MemoryEngine()
            image = MemoryImage(engine)
            source = torch.zeros(rows, lanes, dtype=torch.bfloat16)
            source[:, 0] = torch.arange(1, rows + 1)
            engine.sram[:source.numel()].copy_(source.flatten())
            destination = image.allocate(torch.zeros(rows, 64))
            shared._copy_contiguous_or_strided_write(engine, sram=0,
                destination=destination, total=rows * chunk, chunk=chunk, jump=128)
            actual = engine.view(destination, rows * 64).reshape(rows, 64)[:, 0]
            step = 128 // chunk
            self.assertEqual(int(torch.count_nonzero(actual)), rows // step)
            torch.testing.assert_close(actual[:rows // step], source[::step, 0], rtol=0, atol=0)
            self.assertFalse(torch.equal(actual, source[:, 0]))

    def test_invalid_shapes_and_aliases_fail_before_dma(self):
        engine, _, args = fixture(200)
        for index, replacement in ((2, 0), (2, -1), (2, True), (2, 1.5),
                                   (1, args[0]), (1, args[0] + 128), (0, args[0] + 32)):
            invalid = list(args)
            invalid[index] = replacement
            with self.assertRaises(ValueError):
                emit_waveform_tanh(engine, *invalid)
            self.assertEqual(engine.transfers, [])


if __name__ == '__main__':
    unittest.main()
