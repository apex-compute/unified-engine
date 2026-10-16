import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import read_trace
import user_dma_core


def _instruction(instruction_type):
    value = user_dma_core.Instructions()
    value.words[0] = int(instruction_type) << 8
    return value


class FakeTraceEngine:
    def __init__(self, pointer, memory):
        self.pointer = int(pointer)
        self.memory = list(memory)
        self.read_address = 0
        self._clock_period_ns = 4.0

    def write_reg32(self, address, value):
        if address == user_dma_core.UE_TRACE_BRAM_ADDR:
            self.read_address = int(value)

    def read_reg32(self, address):
        if address == user_dma_core.UE_TRACE_BRAM_ADDR:
            return self.pointer
        if address == user_dma_core.UE_TRACE_BRAM_DATA:
            return self.memory[self.read_address]
        raise AssertionError(f"unexpected register 0x{address:x}")


class CircularTailTraceTests(unittest.TestCase):
    def test_prefetch_windows_skip_only_types_two_and_six(self):
        insts = [_instruction(kind) for kind in (0, 2, 6, 5, 7, 9)]
        self.assertEqual(read_trace.decode_windows(insts, list(range(6)),
                                                  [0, 1, 2, 100, 200, 300]),
                         [100, 100, 100, 200, 300, None])

    def test_names_and_compute_wait_are_not_attributed_to_pbi(self):
        engine = FakeTraceEngine(4, [0, 1, 2, 100])
        insts = [_instruction(kind) for kind in (0, 2, 6, 9)]
        labels = ['UE_COMPUTE (CONV2D)', 'ISA_REG_ALU (SET)',
                  'PBI_SET (REG) inst_pointer_idx=1', 'UE_HALT_INST']
        def parse(inst, index, address):
            print(f'  [{index:4d}] @ 0x00000000 = 00000000')
            print('        ' + labels[index])
        engine.parse_instruction = parse
        packets = []
        def packet(**kw):
            packets.append(kw)
            return b'p'
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
                read_trace, '_pf_packet', side_effect=packet), mock.patch.object(
                read_trace, '_pf_track_event_begin', side_effect=lambda uuid, name, annotations: name.encode()), \
                mock.patch.object(read_trace, '_pf_track_event_end', return_value=b'end'):
            self.assertTrue(read_trace.build_perfetto([0, 1, 2, 100], engine,
                            str(Path(directory)/'test.pftrace'), instructions=insts))
        compute = next(i for i,p in enumerate(packets)
                       if p.get('track_event') == b'UE_COMPUTE (CONV2D)')
        self.assertEqual(packets[compute+1]['timestamp_ns'], 100*16*4)
        self.assertTrue(any(p.get('track_event') == b'PBI_SET (REG) [decode gap]'
                            for p in packets))

    def test_full_buffer_is_rotated_into_chronological_order(self):
        total = user_dma_core.UE_TRACE_SIZE + 3
        instructions = [
            _instruction(user_dma_core.INSTRUCTION_NOP)
            for _ in range(total)]
        instructions[-1] = _instruction(user_dma_core.INSTRUCTION_HALT)
        memory = [0] * user_dma_core.UE_TRACE_SIZE
        for event in range(total):
            memory[event % user_dma_core.UE_TRACE_SIZE] = 1000 + event
        engine = FakeTraceEngine(total % user_dma_core.UE_TRACE_SIZE, memory)

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
                read_trace, "build_perfetto", return_value=True) as perfetto:
            path = Path(directory) / "nano_tail.csv"
            result = read_trace.generate_circular_tail_trace(
                engine, path, instructions=instructions,
                program_dram_addr=0x90000000)
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(len(rows), user_dma_core.UE_TRACE_SIZE)
        self.assertEqual(int(rows[0]["physical_address"]), 3)
        self.assertEqual(int(rows[0]["instruction_index"]), 3)
        self.assertEqual(int(rows[0]["counter"]), 1003)
        self.assertEqual(int(rows[-1]["physical_address"]), 2)
        self.assertEqual(int(rows[-1]["instruction_index"]), total - 1)
        self.assertEqual(int(rows[-1]["counter"]), 1000 + total - 1)
        self.assertEqual(result["retained_events"], user_dma_core.UE_TRACE_SIZE)
        self.assertEqual(result["first_instruction_index"], 3)
        kwargs = perfetto.call_args.kwargs
        self.assertEqual(kwargs["instruction_indices"][0], 3)
        self.assertEqual(kwargs["instruction_indices"][-1], total - 1)

    def test_pointer_mismatch_fails_before_reading_trace_rows(self):
        instructions = [
            _instruction(user_dma_core.INSTRUCTION_NOP),
            _instruction(user_dma_core.INSTRUCTION_HALT),
        ]
        engine = FakeTraceEngine(pointer=1, memory=[0] * 2)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "write pointer"):
                read_trace.generate_circular_tail_trace(
                    engine, Path(directory) / "tail.csv",
                    instructions=instructions, program_dram_addr=0)

    def test_jump_bearing_program_is_rejected(self):
        instructions = [
            _instruction(user_dma_core.INSTRUCTION_JUMP),
            _instruction(user_dma_core.INSTRUCTION_HALT),
        ]
        engine = FakeTraceEngine(pointer=2, memory=[0, 1])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "control-flow"):
                read_trace.generate_circular_tail_trace(
                    engine, Path(directory) / "tail.csv",
                    instructions=instructions, program_dram_addr=0)


if __name__ == "__main__":
    unittest.main()
