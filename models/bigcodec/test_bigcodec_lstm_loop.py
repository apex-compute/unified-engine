"""Counted-loop arithmetic/address parity, guarded state, and ISA contracts."""
import contextlib
from dataclasses import replace
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bigcodec_lstm as lstm
import bigcodec_lstm_loop as loop
from bigcodec_device import shared, udc
import test_bigcodec_lstm as lstm_tests
from test_bigcodec_lstm import NativeArithmeticMemoryEngine


class LoopMemoryEngine(NativeArithmeticMemoryEngine):
    """Replay only the outer integer loop; reuse existing arithmetic models."""
    OPERATIONS = frozenset((
        'accelerator_memory_to_sram', 'sram_to_accelerator_memory',
        'accelerator_memory_to_bias_sram', 'accelerator_memory_to_scale_sram',
        'matmat_mul_core', 'eltwise_core_dram', 'broadcast_mul', 'broadcast_add',
        'eltwise_mul_core', 'eltwise_add_core', 'eltwise_sub_core',
        'start_queue_for_bf16_matvec_operation', 'start_queue_for_dot_product_operation',
        'start_queue_for_maxpool2d_operation'))

    def __init__(self):
        super().__init__()
        self.recording = None
        self.registers, self.pointers = {}, {}
        self._isa_reg_counter = self.next_pointer = 1
        self.loop_counts, self.arithmetic = [], []

    def __getattribute__(self, name):
        method = super().__getattribute__(name)
        if name not in LoopMemoryEngine.OPERATIONS:
            return method
        def invoke(*args, **kwargs):
            if self.recording is not None:
                self.recording.append((name, args, dict(kwargs)))
                return
            if name not in ('accelerator_memory_to_sram', 'sram_to_accelerator_memory',
                            'accelerator_memory_to_bias_sram'):
                self.arithmetic.append((name, args, dict(kwargs)))
            return method(*args, **kwargs)
        return invoke

    def alloc_isa_reg(self):
        index = self._isa_reg_counter
        self._isa_reg_counter += 1
        return index

    def release_isa_reg(self):
        self._isa_reg_counter -= 1
        self.registers.pop(self._isa_reg_counter, None)

    def alloc_inst_ptr(self):
        index = self.next_pointer
        self.next_pointer += 1
        return index

    def release_inst_ptr(self, index):
        assert index == self.next_pointer - 1
        self.next_pointer -= 1

    def _record(self, name, args, kwargs):
        if self.recording is None:
            return False
        self.recording.append((name, args, dict(kwargs)))
        return True

    def generate_instruction_add_set(self, index, value):
        self.registers[index] = value

    def generate_instruction_add_imm(self, src_reg_idx, immediate_value, dst_reg_idx=None):
        if self._record('generate_instruction_add_imm', (src_reg_idx, immediate_value, dst_reg_idx), {}):
            return
        destination = src_reg_idx if dst_reg_idx is None else dst_reg_idx
        self.registers[destination] = (self.registers[src_reg_idx] + immediate_value) & 0xffffffff

    def generate_instruction_pbi_init(self, *, dma_length, inst_pointer_idx):
        if not self._record('generate_instruction_pbi_init', (), dict(dma_length=dma_length, inst_pointer_idx=inst_pointer_idx)):
            self.pointers[inst_pointer_idx] = {'length': dma_length}

    def generate_instruction_pbi_inc(self, *, inst_pointer_idx, pbi_field_select, general_reg_src):
        kwargs = dict(inst_pointer_idx=inst_pointer_idx, pbi_field_select=pbi_field_select, general_reg_src=general_reg_src)
        if not self._record('generate_instruction_pbi_inc', (), kwargs):
            assert pbi_field_select == udc.PBI_FIELD.DRAM_ADDR
            self.pointers[inst_pointer_idx]['address'] = self.registers[general_reg_src] << 3

    def accelerator_memory_to_sram(self, address, sram, count, general_reg_src=None):
        if general_reg_src is not None:
            address = self.registers[general_reg_src] << 3
        return super().accelerator_memory_to_sram(address, sram, count)

    def sram_to_accelerator_memory(self, sram, address, count, general_reg_src=None):
        if general_reg_src is not None:
            address = self.registers[general_reg_src] << 3
        return super().sram_to_accelerator_memory(sram, address, count)

    def accelerator_memory_to_bias_sram(self, address, count, inst_pointer_idx=None):
        if inst_pointer_idx is not None:
            assert address == count == 0  # Both operands are PBI deltas.
            pointer = self.pointers[inst_pointer_idx]
            address, count = pointer['address'], pointer['length'] // 2
        return super().accelerator_memory_to_bias_sram(address, count)

    def loop_start(self, loop_cnt, relative):
        assert self.recording is None and loop_cnt >= 1 and not relative
        self.alloc_isa_reg()
        self.recording, self.count = [], loop_cnt

    def loop_end(self):
        operations, self.recording = self.recording, None
        self.loop_counts.append(self.count)
        for _ in range(self.count):
            for name, args, kwargs in operations:
                getattr(self, name)(*args, **kwargs)
        self.release_isa_reg()


class CaptureEngine(shared._WholeGraphEngine):
    def __init__(self):
        super().__init__(0x98000000)
        self.maximum_register = 0
        self.outer_loops = []

    def alloc_isa_reg(self):
        index = super().alloc_isa_reg()
        self.maximum_register = max(self.maximum_register, index)
        return index

    def loop_start(self, loop_cnt=0, gpr_loop_cnt=None, relative=True):
        index = super().loop_start(loop_cnt, gpr_loop_cnt, relative)
        self.outer_loops.append((loop_cnt, self.capture_count, index, relative))
        return index


def fixture(width, sequence, *, compensated=False, precision='bf16', paired=False):
    helper = lstm_tests.BigCodecLSTMTest()
    source = torch.randn(sequence, width, generator=torch.Generator().manual_seed(71)) * .2
    seed, _, plan = helper.prepare(helper.model(width), source,
        recurrent_precision=precision, compensated_cell=compensated,
        compensated_tanh=compensated, fused_projection=compensated)
    guarded = (plan.input_address, plan.output_address, plan.scratch_address)
    for address in guarded:
        value = seed.regions.pop(address)
        sentinel = torch.full((64,), -3.25, dtype=torch.bfloat16)
        seed.regions[address - 128] = torch.cat((sentinel, value, sentinel))
    return seed.regions, replace(plan, paired_sigmoid=paired)


class LSTMLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.axi_width = udc.UE_AXI_DATA_WIDTH_BITS
        udc.UE_AXI_DATA_WIDTH_BITS = 256

    @classmethod
    def tearDownClass(cls):
        udc.UE_AXI_DATA_WIDTH_BITS = cls.axi_width

    def test_expanded_arithmetic_and_all_dram_addresses_match_unrolled(self):
        configurations = ((64, 1, False, 'bf16', False), (65, 2, False, 'bf16', False),
                          (65, 5, False, 'bf16', False), (64, 5, True, 'bf16', False),
                          (64, 4, True, 'if8', False), (1536, 3, True, 'bf16', False),
                          (1536, 3, True, 'bf16', True))
        for width, sequence, compensated, precision, paired in configurations:
            with self.subTest(width=width, sequence=sequence, compensated=compensated,
                              precision=precision, paired=paired):
                regions, plan = fixture(width, sequence, compensated=compensated,
                                        precision=precision, paired=paired)
                old, new = LoopMemoryEngine(), LoopMemoryEngine()
                old.regions = {address: value.clone() for address, value in regions.items()}
                new.regions = {address: value.clone() for address, value in regions.items()}
                lstm._emit_lstm_unrolled(old, plan)
                loop.emit_lstm(new, plan)
                self.assertEqual(old.arithmetic, new.arithmetic)
                self.assertEqual(old.dma_calls, new.dma_calls)
                for address in regions:
                    torch.testing.assert_close(old.regions[address], new.regions[address], rtol=0, atol=0, equal_nan=True)
                output = new.view(plan.output_address, sequence * plan.padded_width).reshape(sequence, plan.padded_width)
                self.assertTrue(torch.isfinite(output).all())
                self.assertFalse(output[:, width:].count_nonzero())
                for address in (plan.input_address, plan.output_address, plan.scratch_address):
                    value = new.regions[address - 128]
                    self.assertTrue((value[:64] == -3.25).all() and (value[-64:] == -3.25).all())
                self.assertTrue(torch.equal(new.regions[plan.input_address - 128], regions[plan.input_address - 128]))
                for address in (key for key in regions if key < plan.input_address - 128):
                    self.assertTrue(torch.equal(new.regions[address], regions[address]))
                self.assertEqual(new.loop_counts, [sequence - 1] * 2 if sequence > 1 else [])
                self.assertEqual((new._isa_reg_counter, new.next_pointer), (1, 1))
                for destination in (plan.regions['layer0_output'][0], plan.output_address):
                    writes = [address for kind, address, size in new.dma_calls
                              if kind == 'write' and destination <= address < destination + sequence * plan.padded_width * 2]
                    self.assertEqual(writes, [destination + time * plan.padded_width * 2 for time in range(sequence)])

    def test_public_dispatch_and_dynamic_arithmetic_callbacks(self):
        _, plan = fixture(64, 3, compensated=True)
        with patch.object(loop, 'emit_lstm') as counted, patch.object(lstm, '_emit_lstm_unrolled') as unrolled:
            lstm.emit_lstm(object(), plan)
            counted.assert_called_once()
            unrolled.assert_not_called()
            counted.reset_mock()
            lstm.emit_lstm(object(), replace(plan, padded_width=8192))
            unrolled.assert_called_once()
            counted.assert_not_called()
        regions, plan = fixture(64, 3, compensated=True)
        engine = LoopMemoryEngine()
        engine.regions = regions
        with patch.object(lstm, '_recurrent_projection_sram', wraps=lstm._recurrent_projection_sram) as projection, \
             patch.object(lstm, '_lstm_step_sram', wraps=lstm._lstm_step_sram) as step:
            loop.emit_lstm(engine, plan)
        # Each layer emits timestep zero and one repeated body, independently
        # of the number of iterations executed by the recorded loop.
        self.assertEqual(projection.call_count, 4)
        self.assertEqual(step.call_count, 4)

    def test_bias_pointer_has_zero_address_and_length_descriptor_deltas(self):
        engine = CaptureEngine()
        engine.start_capture()
        registers = [engine.alloc_isa_reg() for _ in range(4)]
        wrapper = loop._TimeAddresses(engine, width=1536, hidden=0x90000000,
            projection=0xA0003000, output=0x90000C00, hidden_reg=registers[0],
            projection_reg=registers[1], output_reg=registers[2], temporary_reg=registers[3])
        for offset in (0, 256, 3072, 11776):
            begin = engine.capture_count
            wrapper.accelerator_memory_to_bias_sram(0xA0003000 + offset, 128)
            words = [item.words for item in engine.capture_buffer[begin:]]
            sets = [word for word in words if (word[0] >> 8) & 15 == udc.INSTRUCTION_PBI_SET]
            copies = [word for word in words if (word[0] >> 8) & 15 == udc.INSTRUCTION_UE_PBI]
            self.assertEqual((len(sets), len(copies)), (2, 1))
            initial, update = sets
            transfer = copies[0]
            self.assertEqual((initial[0] >> 16) & 15, udc.PBI_MODE_INIT)
            self.assertEqual(initial[2], 256)
            self.assertEqual((update[0] >> 16) & 15, udc.PBI_MODE_REG)
            self.assertEqual((update[0] >> 20) & 15, udc.PBI_FIELD.DRAM_ADDR)
            self.assertEqual((update[0] >> 24) & 63, registers[1] if offset == 0 else registers[3])
            self.assertEqual((update[1], update[2], transfer[1], transfer[2]), (0, 0, 0, 0))
            self.assertEqual((transfer[7] >> 8) & 3, udc.MEMCPY_TYPE.BIAS_BRAM)
            self.assertEqual((transfer[5] >> 12) & 15, udc.UE_MODE.MEMCPY_FROM_DRAM)
            self.assertEqual((initial[0] >> 12) & 15, (update[0] >> 12) & 15)
            self.assertEqual((initial[0] >> 12) & 15, (transfer[0] >> 12) & 15)
        for _ in registers:
            engine.release_isa_reg()
        self.assertEqual(engine._isa_reg_counter, 1)

    def test_capture_has_valid_absolute_loops_one_halt_and_no_register_leaks(self):
        _, plan = fixture(64, 5, compensated=True)
        engine = CaptureEngine()
        with contextlib.redirect_stdout(io.StringIO()):
            engine.start_capture()
            loop.emit_lstm(engine, plan)
            halt = engine.capture_count
            engine.generate_instruction_halt()
            engine.stop_capture()
        self.assertEqual(engine._isa_reg_counter, 1)
        self.assertFalse(engine._capture_loop_stack)
        self.assertLessEqual(engine.maximum_register, 5)
        self.assertEqual(udc.check_isa_jumps(engine.capture_buffer, engine.get_program_dram_addr()), [])
        kinds = [(item.words[0] >> 8) & 15 for item in engine.capture_buffer]
        self.assertEqual(kinds.count(udc.INSTRUCTION_HALT), 1)
        self.assertNotIn(udc.INSTRUCTION_SWI, kinds)
        self.assertTrue(all(kind == udc.INSTRUCTION_NOP for kind in kinds[halt + 1:]))
        self.assertEqual(len(engine.outer_loops), 2)
        for count, head, register, relative in engine.outer_loops:
            self.assertEqual(count, plan.sequence - 1)
            self.assertFalse(relative)
            target = engine.get_program_dram_addr() + head * 32
            self.assertEqual(target % 64, 0)
            back_edges = []
            for index, instruction in enumerate(engine.capture_buffer):
                word = instruction.words
                if (word[0] >> 8) & 15 != udc.INSTRUCTION_JUMP:
                    continue
                immediate = ((word[1] >> 22) & 1023) | ((word[2] & ((1 << 22) - 1)) << 10)
                if word[1] & 15 == udc.JUMP_MODE_JNZ and (word[1] >> 4) & 63 == register and immediate << 3 == target:
                    back_edges.append(index)
            self.assertEqual(len(back_edges), 1)
            self.assertGreater(back_edges[0], head)


if __name__ == '__main__':
    unittest.main()
