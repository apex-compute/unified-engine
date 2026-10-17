"""Native spectrum ABI and artifact identity checks without FPGA access."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
import numpy as np

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parent / "dpdfnet"):
    sys.path.insert(0, str(path))
import dpdfnet8khz_precompiled as dp
from test_dpdfnet_runtime import FakeEngine
from dpdfnet8khz_run_from_bin import _load_frames


class NativeRuntimeTest(unittest.TestCase):
    def test_spectrum_loader_rejects_complex_and_overflowing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.npy"
            np.save(path, np.ones((1, 1, 81, 2), dtype=np.complex64))
            with self.assertRaisesRegex(TypeError, "real numeric"):
                _load_frames(path)
            np.save(path, np.full((1, 1, 81, 2), 1e100, dtype=np.float64))
            with self.assertRaisesRegex(ValueError, "float32 range"):
                _load_frames(path)

    def test_sixteen_kilohertz_artifact_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "native 8 kHz artifact"):
            dp.validate_hardware({"format": "andromeda.dpdfnet2.streaming-v2",
                                  "model": "dpdfnet2"})

    def test_old_native_artifact_requires_corrected_compiler(self):
        with self.assertRaisesRegex(RuntimeError, "rebuild"):
            dp.validate_hardware({"format": "andromeda.dpdfnet2_8khz.streaming-v1",
                                  "model": "dpdfnet2_8khz"})

    def test_wrong_source_model_is_rejected_even_with_native_format(self):
        with self.assertRaisesRegex(RuntimeError, "source-model checksum"):
            dp.validate_hardware({"format": dp.FORMAT, "model": "dpdfnet2_8khz",
                                  "onnx_sha256": "0" * 64})

    def test_wrong_axi_width_is_rejected_before_upload(self):
        engine = FakeEngine([])
        payload = {"hardware": {"axi_data_width_bits": 256}}
        with mock.patch.object(dp, "validate_hardware"):
            with self.assertRaisesRegex(RuntimeError, "live hardware reports AXI-512"):
                dp.WholeGraphBackend(engine, payload, axi_data_width_bits=512)
        self.assertEqual(engine.calls, [])

    def test_eighty_one_bins_preserve_bits_and_stop_after_nonfinite_output(self):
        spec = dp.make_layout("spec", (1, 1, 81, 2), dp.INPUT_BASE)
        output = dp.make_layout("spec_e", spec.shape, dp.TENSOR_BASE)
        expected = torch.linspace(-2, 2, 162).reshape(spec.shape).bfloat16()
        packed = dp.pack_tensor(expected, output)
        # Nonfinite padding is outside the ABI; the last logical bin is inside it.
        packed.reshape(-1, 64)[:, 2:] = float("nan")
        corrupt = expected.clone()
        corrupt.reshape(-1)[-1] = float("nan")
        engine = FakeEngine([packed, dp.pack_tensor(corrupt, output)])
        payload = {"hardware": {
            "axi_data_width_bits": 256,
            "model_base": dp.MODEL_BASE, "model_image": torch.zeros(64, dtype=torch.uint8),
            "program_address": dp.MODEL_BASE + 128,
            "tensors": {"spec": spec.manifest(), "spec_e": output.manifest()},
        }}
        with mock.patch.object(dp, "validate_hardware"):
            backend = dp.WholeGraphBackend(engine, payload, axi_data_width_bits=256)
        actual = backend.execute(torch.zeros(spec.shape))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(FloatingPointError, "frame 2"):
            backend.execute(torch.zeros(spec.shape))
        calls = list(engine.calls)
        with self.assertRaisesRegex(FloatingPointError, "backend is stopped"):
            backend.execute(torch.zeros(spec.shape))
        self.assertEqual(calls, engine.calls)
        self.assertEqual((backend.program_kicks, backend.input_upload_writes,
                          backend.output_reads), (2, 2, 2))


if __name__ == "__main__":
    unittest.main()
