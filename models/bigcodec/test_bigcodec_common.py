"""Audio length and portable token-container contracts, without model weights."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from bigcodec_common import (
    CHECKPOINT_SHA256, HOP_LENGTH, SAMPLE_RATE, TOKEN_FORMAT,
    load_tokens, pad_audio, read_audio, restore_audio, save_tokens, validate_tokens,
)


class AudioTokenContractTest(unittest.TestCase):
    def metadata(self, n=200):
        return dict(format=TOKEN_FORMAT, sample_rate=SAMPLE_RATE, hop_length=HOP_LENGTH,
                    source_rate=SAMPLE_RATE, source_samples=n, source_channels=1,
                    native_samples=n, padded_samples=n + 200 - n % 200,
                    checkpoint_sha256=CHECKPOINT_SHA256)

    def test_official_padding_includes_full_hop_on_exact_multiple(self):
        for n in (1, 199, 200, 201, 16000):
            source = np.linspace(-1, 1, n, dtype=np.float32)
            padded = pad_audio(source)
            self.assertEqual(padded.size, n + 200 - n % 200)
            np.testing.assert_array_equal(padded[:n], source)
            np.testing.assert_array_equal(padded[n:], 0)

    def test_invalid_audio_rejected(self):
        for bad in (np.array([]), np.array([np.nan]), np.zeros((2, 3))):
            with self.assertRaises(ValueError):
                pad_audio(bad)

    def test_source_rate_stereo_and_odd_sample_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.wav"
            source = np.full((11027, 2), 0.25, dtype=np.float32)
            source[:, 1] = 0.75
            sf.write(path, source, 44100, subtype="FLOAT")
            audio, metadata = read_audio(path)
            self.assertEqual(metadata["source_samples"], 11027)
            self.assertEqual(metadata["source_channels"], 2)
            self.assertEqual(audio.size, (11027 * 16000 + 44099) // 44100)
            restored = restore_audio(pad_audio(audio), metadata)
            self.assertEqual(restored.shape, (11027,))
            self.assertTrue(np.isfinite(restored).all())
            self.assertAlmostEqual(float(restored[100:-100].mean()), 0.5, places=4)

    def test_tokens_roundtrip_without_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.npz"
            tokens = np.array([0, 8191], dtype=np.int64)
            metadata = self.metadata()
            save_tokens(path, tokens, metadata)
            actual, actual_metadata = load_tokens(path)
            np.testing.assert_array_equal(actual, tokens)
            self.assertEqual(actual_metadata, metadata)
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(archive["tokens"].dtype, np.uint16)

    def test_invalid_token_indices_rejected(self):
        for tokens in (np.array([-1, 0]), np.array([0, 8192]), np.array([0., 1.]), np.zeros((1, 2), dtype=int)):
            with self.assertRaises(ValueError):
                validate_tokens(tokens, self.metadata())

    def test_mismatched_metadata_rejected(self):
        for key, value in (("checkpoint_sha256", "bad"), ("format", "unknown"),
                           ("source_samples", 199), ("padded_samples", 200),
                           ("sample_rate", 8000), ("native_samples", -1)):
            metadata = self.metadata()
            metadata[key] = value
            with self.assertRaises(ValueError, msg=key):
                validate_tokens(np.array([0, 1]), metadata)

    def test_truncated_or_nonfinite_decoder_output_rejected(self):
        for output in (np.zeros(199), np.full(400, np.inf)):
            with self.assertRaises(ValueError):
                restore_audio(output, self.metadata())


if __name__ == "__main__":
    unittest.main()
