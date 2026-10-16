"""Audio framing and synthesis checks without running the neural network."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf


sys.path.insert(0, str(Path(__file__).resolve().parent))
import dpdfnet_audio as audio


def delayed_identity(frames):
    """Independent four-frame delay standing in for the model's buffers."""
    delayed = np.zeros_like(frames)
    delayed[4:] = frames[:-4]
    return delayed


class DPDFNetAudioTest(unittest.TestCase):
    def test_delayed_identity_preserves_first_and_last_impulses(self):
        # Include recordings shorter than a window and each hop/window edge.
        for length in (1, 2, 159, 160, 161, 319, 320, 321, 1001):
            with self.subTest(length=length):
                waveform = np.zeros(length, dtype=np.float32)
                waveform[0] = 0.75
                waveform[-1] = -0.375
                analyzed = audio.analyze_audio(waveform, 16000)
                result = audio.synthesize_audio(delayed_identity(analyzed.frames), analyzed)
                self.assertEqual(result.shape, waveform.shape)
                self.assertTrue(np.isfinite(result).all())
                np.testing.assert_allclose(result, waveform, atol=2e-6, rtol=1e-5)
                self.assertAlmostEqual(float(result[-1]), float(waveform[-1]), places=6)

    def test_delayed_identity_preserves_non_hop_aligned_waveform(self):
        generator = np.random.default_rng(42)
        waveform = generator.uniform(-0.5, 0.5, 1237).astype(np.float32)
        analyzed = audio.analyze_audio(waveform, 16000)
        result = audio.synthesize_audio(delayed_identity(analyzed.frames), analyzed)
        self.assertEqual(analyzed.source_sample_rate, 16000)
        self.assertEqual(analyzed.source_samples, waveform.size)
        self.assertEqual(analyzed.model_samples, waveform.size)
        self.assertEqual(analyzed.frames.shape[1:], (1, 1, 161, 2))
        np.testing.assert_allclose(result, waveform, atol=2e-6, rtol=1e-5)

    def test_silence_remains_exact_zero(self):
        for length in (1, 321, 1000):
            with self.subTest(length=length):
                analyzed = audio.analyze_audio(np.zeros(length, dtype=np.float32), 16000)
                self.assertEqual(np.count_nonzero(analyzed.frames), 0)
                result = audio.synthesize_audio(delayed_identity(analyzed.frames), analyzed)
                self.assertEqual(result.shape, (length,))
                self.assertTrue(np.isfinite(result).all())
                self.assertEqual(np.count_nonzero(result), 0)

    def test_stereo_48khz_downmix_and_original_duration(self):
        length = 1003
        signal = np.sin(np.arange(length, dtype=np.float32) * (2 * np.pi * 400 / 48000))
        stereo = np.stack((0.1 * signal, 0.3 * signal), axis=1)
        analyzed = audio.analyze_audio(stereo, 48000)
        reference = audio.analyze_audio(0.2 * signal, 48000)
        self.assertEqual(analyzed.source_sample_rate, 48000)
        self.assertEqual(analyzed.source_samples, length)
        self.assertEqual(analyzed.model_samples, 335)
        np.testing.assert_allclose(analyzed.frames, reference.frames, atol=2e-6, rtol=1e-5)
        result = audio.synthesize_audio(delayed_identity(analyzed.frames), analyzed)
        self.assertEqual(result.shape, (length,))
        self.assertTrue(np.isfinite(result).all())
        # Interior samples of this low-frequency signal survive resampling.
        np.testing.assert_allclose(result[100:-100], (0.2 * signal)[100:-100],
                                   atol=2e-4, rtol=2e-3)

    def test_read_audio_retains_source_metadata(self):
        waveform = np.zeros((481, 2), dtype=np.float32)
        waveform[100, :] = (0.25, -0.25)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            sf.write(path, waveform, 48000, subtype="FLOAT")
            analyzed = audio.read_audio(path)
        self.assertEqual(analyzed.source_sample_rate, 48000)
        self.assertEqual(analyzed.source_samples, 481)
        self.assertEqual(analyzed.model_samples, 161)
        self.assertEqual(np.count_nonzero(analyzed.frames), 0)

    def test_zero_db_attenuation_limit_restores_aligned_input(self):
        waveform = np.zeros(997, dtype=np.float32)
        waveform[0], waveform[450], waveform[-1] = 0.25, -0.5, 0.75
        analyzed = audio.analyze_audio(waveform, 16000)
        result = audio.synthesize_audio(np.zeros_like(analyzed.frames), analyzed,
                                        attn_limit_db=0.0)
        np.testing.assert_allclose(result, waveform, atol=2e-6, rtol=1e-5)

    def test_invalid_audio_is_rejected(self):
        invalid = (np.empty(0, dtype=np.float32), np.zeros((2, 2, 2)),
                   np.zeros((4, 0)), np.array([0.0, np.nan]),
                   np.array([0.0, np.inf]), np.array([0.0, -np.inf]))
        for waveform in invalid:
            with self.subTest(shape=waveform.shape), self.assertRaises((TypeError, ValueError)):
                audio.analyze_audio(waveform, 16000)
        for sample_rate in (0, -16000):
            with self.subTest(sample_rate=sample_rate), self.assertRaises((TypeError, ValueError)):
                audio.analyze_audio(np.ones(321, dtype=np.float32), sample_rate)

    def test_invalid_output_shape_and_nonfinite_values_are_rejected(self):
        analyzed = audio.analyze_audio(np.zeros(321, dtype=np.float32), 16000)
        for malformed in (analyzed.frames[:-1], analyzed.frames[..., :-1],
                          analyzed.frames[:, :, :, :-1, :], analyzed.frames.reshape(-1)):
            with self.subTest(shape=malformed.shape), self.assertRaises((TypeError, ValueError)):
                audio.synthesize_audio(malformed, analyzed)
        for value in (np.nan, np.inf, -np.inf):
            malformed = analyzed.frames.copy()
            malformed.reshape(-1)[0] = value
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                audio.synthesize_audio(malformed, analyzed)


if __name__ == "__main__":
    unittest.main()
