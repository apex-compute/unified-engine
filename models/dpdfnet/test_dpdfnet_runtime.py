"""Host-side streaming output validation without FPGA access."""

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))
import dpdfnet_precompiled as dp


class FakeEngine:
    """Return supplied physical BF16 outputs and record all device access."""

    conv_geometry_mode = dp.udc.CONV_GEOMETRY_QUEUE_CONFIG
    h2c_device = "host-to-device"
    c2h_device = "device-to-host"

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def dma_write(self, device, address, value, size):
        self.calls.append(("write", address, size))
        return size

    def dma_read(self, device, address, value, size):
        self.calls.append(("read", address, size))
        value.copy_(next(self.outputs))
        return size

    def is_queue_busy(self):
        self.calls.append(("busy",))
        return False

    def write_reg32(self, address, value):
        self.calls.append(("write_register", address, value))

    def read_reg32(self, address):
        self.calls.append(("read_register", address))
        return dp.udc.INT_CAUSE_HALT

    def start_execute_from_dram(self, address):
        self.calls.append(("kick", address))

    def read_latency_cycles(self):
        self.calls.append(("cycles",))
        return 123


class DPDFNetRuntimeTest(unittest.TestCase):
    def test_old_artifact_requires_recompilation(self):
        with self.assertRaisesRegex(RuntimeError, "rebuild.*--force"):
            dp.validate_hardware({
                "format": "andromeda.dpdfnet2.streaming-v1",
                "model": "dpdfnet2",
            })

    def setUp(self):
        self.input_layout = dp.make_layout(
            "spec", (1, 1, 161, 2), dp.INPUT_BASE)
        self.output_layout = dp.make_layout(
            "spec_e", (1, 1, 161, 2), dp.TENSOR_BASE)
        self.input = torch.zeros(self.input_layout.shape)
        self.expected = torch.linspace(-4, 4, 322).reshape(
            self.output_layout.shape).to(torch.bfloat16)

    def backend(self, outputs):
        engine = FakeEngine(outputs)
        payload = {"hardware": {
            "model_base": dp.MODEL_BASE,
            "model_image": torch.zeros(64, dtype=torch.uint8),
            "program_address": dp.MODEL_BASE + 128,
            "tensors": {
                "spec": self.input_layout.manifest(),
                "spec_e": self.output_layout.manifest(),
            },
        }}
        # Use a minimal artifact while retaining the real backend
        # initialization, upload, execution, readback and output validation.
        with mock.patch.object(dp, "validate_hardware"):
            backend = dp.WholeGraphBackend(
                engine, payload, axi_data_width_bits=256)
        return backend, engine

    def test_finite_output_preserves_values_shape_dtype_and_streaming(self):
        packed = dp.pack_tensor(self.expected, self.output_layout)
        backend, engine = self.backend([packed, packed])
        for _ in range(2):
            actual = backend.execute(self.input)
            torch.testing.assert_close(actual, self.expected, rtol=0, atol=0)
        self.assertEqual(backend.program_kicks, 2)
        self.assertEqual(backend.input_upload_writes, 2)
        self.assertEqual(backend.output_reads, 2)
        self.assertEqual(backend.cycles, 246)
        self.assertEqual(sum(call[0] == "write" for call in engine.calls), 3)

    def test_nonfinite_logical_output_reports_counts_and_prevents_reuse(self):
        finite = dp.pack_tensor(self.expected, self.output_layout)
        corrupted = self.expected.clone()
        corrupted.reshape(-1)[:5] = torch.tensor(
            [float("nan"), float("inf"), -float("inf"),
             float("nan"), -float("inf")], dtype=torch.bfloat16)
        backend, engine = self.backend([
            finite, dp.pack_tensor(corrupted, self.output_layout)])
        backend.execute(self.input)
        with self.assertRaisesRegex(
                FloatingPointError,
                r"frame 2 \(1-based\): NaN=2, \+Inf=1, -Inf=2"):
            backend.execute(self.input)
        device_calls_after_failure = list(engine.calls)
        with self.assertRaisesRegex(
                FloatingPointError, "backend is stopped.*frame 2"):
            backend.execute(self.input)
        self.assertEqual(engine.calls, device_calls_after_failure)
        self.assertEqual(backend.program_kicks, 2)
        self.assertEqual(backend.input_upload_writes, 2)

    def test_each_nonfinite_kind_is_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                corrupted = self.expected.clone()
                corrupted.reshape(-1)[-1] = value
                backend, _ = self.backend([
                    dp.pack_tensor(corrupted, self.output_layout)])
                with self.assertRaisesRegex(FloatingPointError, "frame 1"):
                    backend.execute(self.input)

    def test_nonfinite_padding_is_excluded(self):
        packed = dp.pack_tensor(self.expected, self.output_layout)
        rows = packed.reshape(
            self.output_layout.rows, self.output_layout.padded_last)
        rows[:, 2:] = float("nan")
        rows[:, 3] = float("inf")
        rows[:, 4] = -float("inf")
        backend, _ = self.backend([packed])
        torch.testing.assert_close(
            backend.execute(self.input), self.expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
