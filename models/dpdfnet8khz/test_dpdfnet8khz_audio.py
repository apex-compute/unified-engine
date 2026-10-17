"""Native 8-kHz framing checks without running the neural network."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf


sys.path.insert(0, str(Path(__file__).resolve().parent))
import dpdfnet8khz_audio as audio


def delayed_identity(frames):
    """Model the four streaming buffers independently of the audio helper."""
    delayed = np.zeros_like(frames)
    delayed[4:] = frames[:-4]
    return delayed


class DPDFNet8kHzAudioTest(unittest.TestCase):
    def test_delayed_identity_preserves_endpoints_at_hop_and_window_edges(self):
        for length in (1, 2, 79, 80, 81, 159, 160, 161, 1001):
            with self.subTest(length=length):
                waveform = np.zeros(length, dtype=np.float32)
                waveform[0] = 0.75
                waveform[-1] = -0.375
                analyzed = audio.analyze_audio(waveform, 8000)
                result = audio.synthesize_audio(delayed_identity(analyzed.frames), analyzed)
                self.assertEqual(result.shape, waveform.shape)
                self.assertTrue(np.isfinite(result).all())
                np.testing.assert_allclose(result, waveform, atol=2e-6, rtol=1e-5)

    def test_native_8khz_frames_and_odd_length_identity(self):
        waveform = np.random.default_rng(42).uniform(-0.5, 0.5, 1237).astype(np.float32)
        analyzed = audio.analyze_audio(waveform, 8000)
        self.assertEqual(analyzed.source_sample_rate, 8000)
        self.assertEqual(analyzed.source_samples, waveform.size)
        self.assertEqual(analyzed.model_samples, waveform.size)
        # 160-point FFT, 80-sample hops, 480 samples to flush delayed output.
        self.assertEqual(analyzed.frames.shape, ((1237 + 480) // 80 + 1, 1, 1, 81, 2))
        result = audio.synthesize_audio(delayed_identity(analyzed.frames), analyzed)
        np.testing.assert_allclose(result, waveform, atol=2e-6, rtol=1e-5)

    def test_silence_remains_exact_zero(self):
        for length in (1, 161, 1000):
            with self.subTest(length=length):
                analyzed = audio.analyze_audio(np.zeros(length, dtype=np.float32), 8000)
                self.assertEqual(np.count_nonzero(analyzed.frames), 0)
                result = audio.synthesize_audio(delayed_identity(analyzed.frames), analyzed)
                self.assertEqual(result.shape, (length,))
                self.assertTrue(np.isfinite(result).all())
                self.assertEqual(np.count_nonzero(result), 0)

    def test_stereo_resampling_retains_source_rate_and_exact_sample_count(self):
        length = 2003
        for source_rate, model_samples in ((16000, 1002), (48000, 334)):
            with self.subTest(source_rate=source_rate):
                signal = np.sin(np.arange(length, dtype=np.float32)
                                * (2 * np.pi * 400 / source_rate))
                stereo = np.stack((0.1 * signal, 0.3 * signal), axis=1)
                analyzed = audio.analyze_audio(stereo, source_rate)
                reference = audio.analyze_audio(0.2 * signal, source_rate)
                self.assertEqual(analyzed.source_sample_rate, source_rate)
                self.assertEqual(analyzed.source_samples, length)
                self.assertEqual(analyzed.model_samples, model_samples)
                self.assertEqual(analyzed.frames.shape[1:], (1, 1, 81, 2))
                np.testing.assert_allclose(analyzed.frames, reference.frames,
                                           atol=2e-6, rtol=1e-5)
                result = audio.synthesize_audio(delayed_identity(analyzed.frames), analyzed)
                self.assertEqual(result.shape, (length,))
                self.assertTrue(np.isfinite(result).all())
                # Compare a pass-band sinusoid away from resampling boundaries.
                np.testing.assert_allclose(result[120:-120], (0.2 * signal)[120:-120],
                                           atol=3e-4, rtol=3e-3)

    def test_read_audio_retains_metadata_and_downmixes(self):
        waveform = np.zeros((481, 2), dtype=np.float32)
        waveform[100, :] = (0.25, -0.25)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            sf.write(path, waveform, 48000, subtype="FLOAT")
            analyzed = audio.read_audio(path)
        self.assertEqual(analyzed.source_sample_rate, 48000)
        self.assertEqual(analyzed.source_samples, 481)
        self.assertEqual(analyzed.model_samples, 81)
        self.assertEqual(np.count_nonzero(analyzed.frames), 0)

    def test_zero_db_attenuation_limit_restores_aligned_input(self):
        waveform = np.zeros(997, dtype=np.float32)
        waveform[0], waveform[450], waveform[-1] = 0.25, -0.5, 0.75
        analyzed = audio.analyze_audio(waveform, 8000)
        result = audio.synthesize_audio(np.zeros_like(analyzed.frames), analyzed,
                                        attn_limit_db=0.0)
        np.testing.assert_allclose(result, waveform, atol=2e-6, rtol=1e-5)

    def test_invalid_audio_is_rejected(self):
        invalid = (np.empty(0, dtype=np.float32), np.zeros((2, 2, 2)),
                   np.zeros((4, 0)), np.array([0.0, np.nan]),
                   np.array([0.0, np.inf]), np.array([0.0, -np.inf]))
        for waveform in invalid:
            with self.subTest(shape=waveform.shape), self.assertRaises((TypeError, ValueError)):
                audio.analyze_audio(waveform, 8000)
        for sample_rate in (0, -8000, 8000.0, True):
            with self.subTest(sample_rate=sample_rate), self.assertRaises((TypeError, ValueError)):
                audio.analyze_audio(np.ones(161, dtype=np.float32), sample_rate)

    def test_invalid_output_shape_and_nonfinite_values_are_rejected(self):
        analyzed = audio.analyze_audio(np.zeros(161, dtype=np.float32), 8000)
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
