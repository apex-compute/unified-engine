"""Reference-corpus boundaries and recurrent-state lifetime without a model."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dpdfnet8khz_prepare_validation as prepare
from test_dpdfnet8khz_common import native_metadata


class StateCounterSession:
    def __init__(self):
        self.counters = []

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=native_metadata())

    def run(self, names, inputs):
        count = float(inputs["state_in"][-1])
        self.counters.append(count)
        state = inputs["state_in"].copy()
        state[-1] = count + 1
        return inputs["spec"] + count, state


class ValidationPreparationTest(unittest.TestCase):
    def test_short_transition_padding_and_one_state_sequence(self):
        first = np.ones((3, 1, 1, 81, 2), dtype=np.float32)
        last = np.full((2, 1, 1, 81, 2), 2, dtype=np.float32)
        frames = prepare.transition_frames(first, last)
        np.testing.assert_array_equal(frames[:3], first)
        self.assertEqual(np.count_nonzero(frames[3:256]), 0)
        np.testing.assert_array_equal(frames[256:258], last)
        self.assertEqual(np.count_nonzero(frames[258:]), 0)
        session = StateCounterSession()
        reset = np.zeros(4, dtype=np.float32)
        enhanced, seconds = prepare.infer_frames(session, frames, reset)
        self.assertEqual(session.counters, list(range(384)))
        np.testing.assert_array_equal(enhanced[256], frames[256] + 256)
        self.assertEqual(np.count_nonzero(reset), 0)
        self.assertGreater(seconds, 0)
        prepare.infer_frames(session, frames[:2], reset)
        self.assertEqual(session.counters[-2:], [0, 1])

    def test_manifest_hashes_and_fresh_state_per_case(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            inputs = [directory / "first.wav", directory / "last.wav"]
            for i, path in enumerate(inputs):
                sf.write(path, np.full((83, 2), 0.01 * (i + 1)), 16000, subtype="FLOAT")
            session = StateCounterSession()
            with mock.patch.object(prepare, "validate_digest", return_value="verified"), \
                    mock.patch.object(prepare, "create_session", return_value=session):
                report = prepare.prepare_validation(inputs, directory / "corpus", directory / "model.onnx")
            self.assertEqual([case["id"] for case in report["cases"]],
                             ["first", "last", "silence_128", "audio_silence_audio_384"])
            start = 0
            for case in report["cases"]:
                self.assertEqual(session.counters[start], 0)
                self.assertEqual(session.counters[start + case["frames"] - 1], case["frames"] - 1)
                start += case["frames"]
                for kind in ("input", "cpu"):
                    self.assertEqual(prepare.sha256(directory / "corpus" / case[kind]), case[kind + "_sha256"])
            info = sf.info(directory / "corpus" / report["cases"][0]["cpu_wav"])
            self.assertEqual((info.samplerate, info.frames, info.channels), (8000, 42, 1))
            self.assertTrue((directory / "corpus/cpu_reference_manifest.json").is_file())

    def test_duplicate_reserved_and_occupied_output_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            audio = directory / "audio.wav"
            sf.write(audio, np.zeros(81), 8000)
            reserved = directory / "silence_128.wav"
            sf.write(reserved, np.zeros(81), 8000)
            invalid_stem = directory / "not safe.wav"
            sf.write(invalid_stem, np.zeros(81), 8000)
            occupied = directory / "occupied"
            occupied.mkdir()
            (occupied / "keep.txt").write_text("preserve me")
            for inputs, output in (([audio, audio], directory / "new"),
                                    ([reserved], directory / "new"), ([audio], occupied),
                                    ([invalid_stem], directory / "new"), ([audio], directory)):
                with self.subTest(inputs=inputs, output=output), \
                        mock.patch.object(prepare, "validate_digest", side_effect=AssertionError("loaded")):
                    with self.assertRaises(ValueError):
                        prepare.prepare_validation(inputs, output, directory / "model.onnx")
            self.assertEqual((occupied / "keep.txt").read_text(), "preserve me")


if __name__ == "__main__":
    unittest.main()
