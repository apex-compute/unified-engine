"""Model identity and metadata validation without network or accelerator access."""

import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dpdfnet8khz_common as common


def native_metadata():
    return {
        "profile": "dpdfnet2_8khz", "sample_rate": "8000", "n_fft": "160",
        "window_length": "160", "hop_length": "80", "freq_bins": "81",
        "state_size": "37860", "erb_norm_state_size": "81", "spec_norm_state_size": "80",
        "erb_norm_init": ",".join(str(-30 - index) for index in range(81)),
        "spec_norm_init": ",".join(str((index + 1) / 10000) for index in range(80)),
        "window_type": "vorbis", "normalized": "0", "center": "1", "pad_mode": "reflect",
    }


class CommonTest(unittest.TestCase):
    def test_metadata_initializes_all_normalizers_at_correct_boundaries(self):
        metadata = native_metadata()
        common.validate_metadata(metadata)
        state = common.initial_state(metadata)
        self.assertEqual(state.shape, (37860,))
        self.assertEqual(state.dtype, np.float32)
        np.testing.assert_array_equal(state[:81], -30 - np.arange(81))
        np.testing.assert_allclose(state[81:161], np.arange(1, 81) / 10000)
        self.assertEqual(np.count_nonzero(state[161:]), 0)
        state[0] = 999
        self.assertEqual(common.initial_state(metadata)[0], -30)

    def test_wrong_profile_framing_and_malformed_state_rejected(self):
        for key, value in (("profile", "dpdfnet2"), ("sample_rate", "16000"),
                           ("window_length", "320"), ("freq_bins", "161"),
                           ("pad_mode", "constant"), ("erb_norm_state_size", "44")):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                common.validate_metadata({**native_metadata(), key: value})
        for key, value in (("state_size", "2"), ("erb_norm_init", "1,2"),
                           ("spec_norm_init", "nan," * 79 + "nan"),
                           ("erb_norm_init", "1,bad")):
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                common.initial_state({**native_metadata(), key: value})
        with self.assertRaises(RuntimeError):
            common.initial_state({})

    def test_download_verifies_bytes_and_cached_file_without_network(self):
        data = b"pinned official model bytes"
        config = {"onnx_sha256": hashlib.sha256(data).hexdigest(), "onnx_url": "https://example.invalid/model"}
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(common, "load_config", return_value=config):
            path = Path(directory) / "model.onnx"
            with mock.patch.object(common.urllib.request, "urlopen", return_value=io.BytesIO(data)) as request:
                self.assertEqual(common.download_model(path), path)
                request.assert_called_once_with(config["onnx_url"], timeout=60)
            with mock.patch.object(common.urllib.request, "urlopen", side_effect=AssertionError("network used")):
                self.assertEqual(common.download_model(path), path)
                path.write_bytes(b"corrupted")
                with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                    common.download_model(path)

    def test_bad_download_preserves_existing_model_and_cleans_temporary(self):
        original = b"original model"
        config = {"onnx_sha256": hashlib.sha256(original).hexdigest(), "onnx_url": "https://example.invalid/model"}
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(common, "load_config", return_value=config):
            path = Path(directory) / "model.onnx"
            path.write_bytes(original)
            with mock.patch.object(common.urllib.request, "urlopen", return_value=io.BytesIO(b"bad download")):
                with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                    common.download_model(path, force=True)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
