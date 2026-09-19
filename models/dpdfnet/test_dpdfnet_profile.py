"""Prefix profiling regressions using an independent circular-trace device."""

import contextlib
import io
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest

import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))
import dpdfnet_profile as profile
from dpdfnet_precompiled import make_layout


def operation(index, start, stop):
    return {"node_index": index, "name": f"op_{index}", "start": start, "stop": stop}


def instruction(index, kind=0):
    return struct.pack("<8I", (index & 255) | (kind << 8), *([0] * 7))


class FakeEngine:
    """Execute through the first HALT and expose a wrapping timestamp ring."""

    h2c_device = "offline"

    def __init__(self, hardware, *, timeout=False):
        self.hardware = hardware
        self.memory = bytearray(hardware["model_image"].numpy().tobytes())
        self.writes = []
        self.busy = False
        self.timeout = timeout
        self.reset_count = 0
        self.stop = 0
        self.read_address = 0

    def is_queue_busy(self):
        return self.busy

    def dma_write(self, device, address, data, size):
        raw = data.view(torch.uint8).numpy().tobytes() if isinstance(data, torch.Tensor) else bytes(data)
        self.writes.append((address, raw))
        offset = address - self.hardware["model_base"]
        if 0 <= offset and offset + size <= len(self.memory):
            self.memory[offset:offset + size] = raw
        return size

    def start_execute_from_dram(self, address):
        assert address == self.hardware["program_address"]
        start = self.hardware["program_offset"]
        for index in range(self.hardware["program_size"] // 32):
            if (self.memory[start + index * 32 + 1] & 15) == profile.udc.INSTRUCTION_HALT:
                self.stop = index
                break
        else:
            raise AssertionError("fake program did not contain HALT")
        self.busy = True
        # Adversarial state mutation proves each replay and cleanup restore it.
        self.memory[:128] = bytes([255]) * 128

    def wait(self):
        if self.timeout:
            raise TimeoutError("injected timeout")
        self.busy = False

    def software_reset(self, *, run_dram_self_test):
        assert not run_dram_self_test
        self.reset_count += 1
        self.busy = False

    def read_latency_cycles(self):
        return (3 * self.stop + 8) * 16

    def write_reg32(self, register, value):
        if register == profile.udc.UE_TRACE_BRAM_ADDR:
            self.read_address = value

    def read_reg32(self, register):
        if register == profile.udc.UE_TRACE_BRAM_ADDR:
            return (self.stop + 1) % profile.udc.UE_TRACE_SIZE
        assert register == profile.udc.UE_TRACE_BRAM_DATA
        index = self.read_address
        while index < max(0, self.stop + 1 - profile.udc.UE_TRACE_SIZE):
            index += profile.udc.UE_TRACE_SIZE
        return 3 * index + 7


class DPDFNetProfileTest(unittest.TestCase):
    def hardware(self):
        operations = [operation(0, 0, 4096), operation(1, 4096, 8191),
                      operation(2, 8191, 8194)]
        # A lone final HALT is followed by image data resembling an unsupported
        # opcode. Neither that data nor bytes beyond program_size are code.
        program = b"".join(instruction(index) for index in range(8194))
        program += instruction(8194, profile.udc.INSTRUCTION_HALT)
        image = bytes(range(128)) + program + instruction(8195, profile.udc.INSTRUCTION_JUMP)
        base = 0x10000
        return {
            "model_base": base,
            "model_image": torch.frombuffer(bytearray(image), dtype=torch.uint8),
            "program_address": base + 128,
            "program_offset": 128,
            "program_size": len(program),
            "program_sha256": "test",
            "operations": operations,
            "tensors": {
                "state_in": make_layout("state_in", (64,), base).manifest(),
                "spec": make_layout("spec", (1, 1, 161, 2), 0x1000000).manifest(),
            },
        }

    def test_group_capacity_includes_halt_endpoint_and_covers_all_operations(self):
        ops = [operation(0, 0, 4), operation(1, 4, 7), operation(2, 7, 8)]
        self.assertEqual(profile.operation_groups(ops, trace_size=8), [ops[:2], ops[2:]])
        for invalid in ([], [operation(0, 0, 8)], [operation(1, 0, 1)],
                        [operation(0, 0, 1), operation(1, 2, 3)]):
            with self.subTest(operations=invalid), self.assertRaises(ValueError):
                profile.operation_groups(invalid, trace_size=8)

    def test_halt_patch_preserves_neighbor_and_accepts_unpaired_final_halt(self):
        original = b"".join(instruction(index) for index in range(4))
        original += instruction(4, profile.udc.INSTRUCTION_HALT)
        for stop in range(4):
            with self.subTest(stop=stop):
                offset, pair, patched = profile.halt_patch(original, stop)
                expected = bytearray(original)
                expected[stop * 32:(stop + 1) * 32] = instruction(stop, profile.udc.INSTRUCTION_HALT)
                actual = bytearray(original)
                actual[offset:offset + len(pair)] = patched
                self.assertEqual(actual, expected)
                self.assertEqual(offset % 64, 0)
        offset, pair, patched = profile.halt_patch(original, 4)
        self.assertEqual((offset, len(pair), patched), (128, 32, pair))
        for stop in (-1, 5):
            with self.assertRaises(ValueError):
                profile.halt_patch(original, stop)

    def test_wrapped_trace_and_missing_endpoints(self):
        engine = FakeEngine(self.hardware())
        engine.stop = 8194
        group = [operation(2, 8191, 8194)]
        self.assertEqual(profile.read_boundaries(engine, group),
                         {8191: 3 * 8191 + 7, 8194: 3 * 8194 + 7})
        with self.assertRaisesRegex(RuntimeError, "missing"):
            profile.read_boundaries(engine, [operation(0, 2, 8194)])
        with self.assertRaisesRegex(RuntimeError, "write pointer"):
            profile.read_boundaries(engine, [operation(0, 8191, 8193)])

    def test_complete_profile_preserves_program_and_restores_state_every_replay(self):
        hardware = self.hardware()
        engine = FakeEngine(hardware)
        original = bytes(engine.memory)
        backend = SimpleNamespace(hardware=hardware, ue=engine, _wait=engine.wait)
        with contextlib.redirect_stdout(io.StringIO()):
            result = profile.profile(backend, torch.zeros(1, 1, 161, 2), 3.0)
        self.assertEqual(len(result["replays"]), 2)
        self.assertEqual([entry["node_index"] for entry in result["operations"]], [0, 1, 2])
        self.assertEqual([entry["cycles"] for entry in result["operations"]],
                         [4096 * 48, 4095 * 48, 3 * 48])
        self.assertEqual(bytes(engine.memory), original)
        state_writes = [raw for address, raw in engine.writes if address == hardware["model_base"]]
        self.assertEqual(state_writes, [original[:128]] * 3)
        input_writes = [raw for address, raw in engine.writes
                        if address == hardware["tensors"]["spec"]["address"]]
        self.assertEqual(len(input_writes), 2)
        self.assertEqual(input_writes[0], input_writes[1])
        patches = [raw for address, raw in engine.writes
                   if hardware["program_address"] <= address < hardware["program_address"] + hardware["program_size"]]
        self.assertEqual([len(raw) for raw in patches], [64, 64])

    def test_timeout_resets_queue_and_restores_original_instruction_and_state(self):
        hardware = self.hardware()
        engine = FakeEngine(hardware, timeout=True)
        original = bytes(engine.memory)
        backend = SimpleNamespace(hardware=hardware, ue=engine, _wait=engine.wait)
        with self.assertRaisesRegex(TimeoutError, "injected timeout"):
            profile.profile(backend, torch.zeros(1, 1, 161, 2), 3.0)
        self.assertEqual(engine.reset_count, 1)
        self.assertFalse(engine.busy)
        self.assertEqual(bytes(engine.memory), original)


if __name__ == "__main__":
    unittest.main()
