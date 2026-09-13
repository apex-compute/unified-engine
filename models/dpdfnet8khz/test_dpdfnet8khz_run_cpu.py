"""CPU runner boundary tests with a delayed-identity streaming session."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dpdfnet8khz_run_cpu as runner
from test_dpdfnet8khz_common import native_metadata


class FakeSession:
    def __init__(self):
        self.frames = []
        self.calls = 0
        self.invalid_state = False

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=native_metadata())

    def get_inputs(self):
        return [SimpleNamespace(name="spec", shape=[1, 1, 81, 2]),
                SimpleNamespace(name="state_in", shape=[37860])]

    def get_outputs(self):
        return [SimpleNamespace(name="spec_e", shape=[1, 1, 81, 2]),
                SimpleNamespace(name="state_out", shape=[37860])]

    def run(self, names, inputs):
        assert names == ["spec_e", "state_out"]
        assert inputs["state_in"][-1] == self.calls
        self.calls += 1
        self.frames.append(inputs["spec"].copy())
        state = inputs["state_in"].copy()
        state[-1] = np.nan if self.invalid_state else self.calls
        output = self.frames[-5] if len(self.frames) >= 5 else np.zeros_like(inputs["spec"])
        return output, state


class CpuRunnerTest(unittest.TestCase):
    def test_file_roundtrip_preserves_original_rate_length_and_state(self):
        for rate in (8000, 16000):
            with self.subTest(rate=rate), tempfile.TemporaryDirectory() as directory:
                source, output = Path(directory) / "input.wav", Path(directory) / "output.wav"
                waveform = (0.1 * np.sin(np.arange(1003) * 2 * np.pi * 400 / rate)).astype(np.float32)
                sf.write(source, waveform, rate, subtype="FLOAT")
                session = FakeSession()
                with mock.patch.object(runner, "validate_digest", return_value="verified"), \
                        mock.patch.object(runner, "create_session", return_value=session):
                    result = runner.run_cpu(source, output, Path(directory) / "model.onnx")
                enhanced, output_rate = sf.read(output)
                self.assertEqual(output_rate, rate)
                self.assertEqual(enhanced.shape, waveform.shape)
                self.assertEqual(sf.info(output).subtype, "FLOAT")
                np.testing.assert_allclose(enhanced[100:-100], waveform[100:-100], atol=3e-4)
                self.assertEqual(result["model_sample_rate"], 8000)
                self.assertEqual(result["state_size"], 37860)
                self.assertEqual(result["output_samples"], 1003)
                self.assertEqual(result["frames"], session.calls)
                self.assertGreater(result["neural_inference_s"], 0)

    def test_nonfinite_state_stops_before_writing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.wav", Path(directory) / "output.wav"
            sf.write(source, np.zeros(161), 8000, subtype="FLOAT")
            session = FakeSession()
            session.invalid_state = True
            with mock.patch.object(runner, "validate_digest", return_value="verified"), \
                    mock.patch.object(runner, "create_session", return_value=session):
                with self.assertRaisesRegex(RuntimeError, "output/state at frame 0"):
                    runner.run_cpu(source, output)
            self.assertEqual(session.calls, 1)
            self.assertFalse(output.exists())

    def test_session_uses_one_cpu_thread_and_checks_streaming_abi(self):
        session, options = FakeSession(), SimpleNamespace()
        fake_ort = SimpleNamespace(
            SessionOptions=lambda: options,
            GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=99),
            InferenceSession=mock.Mock(return_value=session))
        with mock.patch.dict(sys.modules, {"onnxruntime": fake_ort}):
            self.assertIs(runner.create_session(Path("model.onnx")), session)
            fake_ort.InferenceSession.assert_called_once_with(
                "model.onnx", sess_options=options, providers=["CPUExecutionProvider"])
            self.assertEqual(options.intra_op_num_threads, 1)
            self.assertEqual(options.inter_op_num_threads, 1)
            session.get_inputs = lambda: [SimpleNamespace(name="spec", shape=[1, 1, 161, 2])]
            with self.assertRaisesRegex(RuntimeError, "streaming ABI"):
                runner.create_session(Path("model.onnx"))

    def test_invalid_output_and_attenuation_rejected_before_model_load(self):
        for source, output, limit in (("input.wav", "input.wav", None),
                                       ("input.wav", "output.npy", None),
                                       ("input.wav", "output.wav", -1),
                                       ("input.wav", "output.wav", float("nan"))):
            with self.subTest(output=output, limit=limit), \
                    mock.patch.object(runner, "validate_digest", side_effect=AssertionError("model loaded")):
                with self.assertRaises(ValueError):
                    runner.run_cpu(Path(source), Path(output), attn_limit_db=limit)


if __name__ == "__main__":
    unittest.main()
