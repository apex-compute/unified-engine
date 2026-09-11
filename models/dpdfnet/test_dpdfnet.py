import unittest

import numpy as np
import torch

from dpdfnet_common import load_config
from dpdfnet_precompiled import (
    copy_patterns, copy_runs, make_layout, pack_tensor, physical_indices,
    transform_source_indices, unpack_tensor,
)
from dpdfnet_run_cpu import (
    attenuation_limit, fit_length, initial_state, resample, vorbis_window,
)


class DPDFNetHelpersTest(unittest.TestCase):
    def test_pinned_streaming_abi(self):
        config = load_config()
        self.assertEqual(config["onnx_inputs"]["spec"], [1, 1, 161, 2])
        self.assertEqual(config["onnx_inputs"]["state_in"], [45424])
        self.assertEqual(config["hop_length"], 160)

    def test_vorbis_window(self):
        window = vorbis_window(320)
        self.assertEqual(tuple(window.shape), (320,))
        self.assertTrue(torch.isfinite(window).all())
        self.assertGreater(float(window.min()), 0.0)
        self.assertLessEqual(float(window.max()), 1.0)
        self.assertTrue(torch.allclose(window, window.flip(0), atol=1e-6))

    def test_initial_state_metadata(self):
        metadata = {
            "state_size": "7", "erb_norm_state_size": "2",
            "spec_norm_state_size": "3", "erb_norm_init": "1,2",
            "spec_norm_init": "3,4,5",
        }
        self.assertEqual(initial_state(metadata).tolist(), [1, 2, 3, 4, 5, 0, 0])

    def test_attenuation_limit_alignment(self):
        noisy = np.ones((1, 6, 2, 2), dtype=np.float32)
        enhanced = np.zeros_like(noisy)
        result = attenuation_limit(noisy, enhanced, 6.020599913)
        self.assertTrue(np.allclose(result[:, :4], 0.0, atol=1e-6))
        self.assertTrue(np.allclose(result[:, 4:], 0.5, atol=1e-6))

    def test_fit_length(self):
        self.assertEqual(fit_length(np.arange(3), 5).tolist(), [0, 1, 2, 0, 0])
        self.assertEqual(fit_length(np.arange(5), 3).tolist(), [0, 1, 2])

    def test_resample_preserves_duration(self):
        value = np.zeros(4800, dtype=np.float32)
        self.assertEqual(resample(value, 48000, 16000).size, 1600)

    def test_padded_tensor_round_trip(self):
        layout = make_layout("value", (2, 3), 0x1000)
        logical = torch.arange(6).reshape(2, 3)
        packed = pack_tensor(logical, layout)
        self.assertEqual(packed.numel(), 128)
        self.assertTrue(torch.equal(
            unpack_tensor(packed, layout).float(), logical.float()))
        self.assertTrue(torch.equal(
            packed.reshape(2, 64)[:, 3:], torch.zeros(2, 61,
                                                       dtype=torch.bfloat16)))

    def test_physical_indices_skip_row_padding(self):
        layout = make_layout("value", (2, 3), 0x1000)
        self.assertEqual(physical_indices(layout).tolist(), [0, 1, 2, 64, 65, 66])
        transformed = transform_source_indices(
            layout, (3, 2), permutation=(1, 0))
        self.assertEqual(transformed.tolist(), [0, 64, 1, 65, 2, 66])

    def test_copy_run_coalescing(self):
        self.assertEqual(
            copy_runs([0, 1, 8, 9, 10], [64, 65, 128, 129, 130]),
            [(0, 64, 2), (8, 128, 3)])

    def test_copy_pattern_strides(self):
        self.assertEqual(
            copy_patterns([0, 8, 16, 1, 9, 17], [0, 1, 2, 64, 65, 66]),
            [(0, 0, 3, 8), (1, 64, 3, 8)])


if __name__ == "__main__":
    unittest.main()
