"""Metric definitions and invalid-utterance handling without optional packages."""

import json
import sys
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
import dpdfnet_audio_metrics as metrics


class DPDFNetAudioMetricsTest(unittest.TestCase):
    def setUp(self):
        self.stoi = Mock(return_value=0.81)
        self.pesq = Mock(return_value=2.75)
        modules = {"pystoi": SimpleNamespace(stoi=self.stoi),
                   "pesq": SimpleNamespace(pesq=self.pesq)}
        patcher = patch.object(metrics.importlib, "import_module", side_effect=modules.__getitem__)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.clean = np.tile(np.array([-1.0, -1.0, 1.0, 1.0]), 4000)
        self.noise = np.tile(np.array([-1.0, 1.0, -1.0, 1.0]), 4000)

    def test_orthogonal_noise_matches_exact_energy_ratio(self):
        for noise_gain in (0.1, 0.5, 1.0, 2.0):
            with self.subTest(noise_gain=noise_gain):
                result = metrics.evaluate_audio(self.clean, self.clean + noise_gain * self.noise)
                self.assertAlmostEqual(result["si_sdr_db"], -20 * np.log10(noise_gain), places=12)
                self.assertEqual(result["stoi"], 0.81)
                self.assertEqual(result["pesq_wb"], 2.75)
                self.assertEqual(result["reasons"], {})

    def test_si_sdr_removes_dc_and_is_gain_invariant(self):
        estimate = self.clean + 0.25 * self.noise
        expected = metrics.evaluate_audio(self.clean, estimate)["si_sdr_db"]
        for clean_gain, estimate_gain in ((0.125, 9.0), (1e-200, 1e200), (-0.5, -4.0)):
            with self.subTest(clean_gain=clean_gain, estimate_gain=estimate_gain):
                value = metrics.evaluate_audio(
                    (self.clean + 0.3) * clean_gain,
                    (estimate - 0.7) * estimate_gain)["si_sdr_db"]
                self.assertAlmostEqual(value, expected, places=12)

    def test_identity_and_exact_gain_have_explicit_infinite_limit(self):
        for gain in (1.0, 0.5, -2.0, 1.7):
            with self.subTest(gain=gain):
                result = metrics.evaluate_audio(self.clean, gain * self.clean)
                self.assertIsNone(result["si_sdr_db"])
                self.assertIn("positive infinity", result["reasons"]["si_sdr_db"])
                self.assertEqual(result["stoi"], 0.81)
                json.dumps(result, allow_nan=False)

    def test_orthogonal_estimate_has_explicit_negative_infinite_limit(self):
        result = metrics.evaluate_audio(self.clean, self.noise)
        self.assertIsNone(result["si_sdr_db"])
        self.assertIn("negative infinity", result["reasons"]["si_sdr_db"])
        json.dumps(result, allow_nan=False)

    def test_silent_or_constant_audio_is_reported_without_scoring_packages(self):
        for clean, estimate in ((np.zeros(16000), self.clean),
                                (self.clean, np.zeros(16000)),
                                (np.ones(16000), self.clean),
                                (self.clean, np.ones(16000))):
            with self.subTest(clean_constant=np.ptp(clean) == 0):
                result = metrics.evaluate_audio(clean, estimate)
                for name in metrics.METRICS:
                    self.assertIsNone(result[name])
                    self.assertIn("zero energy", result["reasons"][name])
                json.dumps(result, allow_nan=False)
        self.stoi.assert_not_called()
        self.pesq.assert_not_called()

    def test_invalid_audio_and_sample_rates_raise(self):
        for clean, estimate in (([], []), ([1, 2], [1]),
                                ([[1, 2]], [[1, 2]]),
                                ([1, np.nan], [1, 2]),
                                ([1, 2], [1, np.inf]),
                                ([1 + 2j, 3], [1, 2])):
            with self.subTest(clean=clean, estimate=estimate):
                with self.assertRaises(ValueError):
                    metrics.evaluate_audio(clean, estimate)
        for rate in (0, -16000, 16000.0, True, np.nan, "16000"):
            with self.subTest(rate=rate):
                with self.assertRaises(ValueError):
                    metrics.evaluate_audio(self.clean, self.noise, rate)
        self.stoi.assert_not_called()
        self.pesq.assert_not_called()

    def test_package_calls_use_standard_modes_without_alignment_or_mutation(self):
        clean = self.clean.copy()
        estimate = np.roll(self.clean, 1) + 0.1 * self.noise
        before = estimate.copy()
        metrics.evaluate_audio(clean, estimate)
        stoi_args, stoi_kwargs = self.stoi.call_args
        pesq_args, pesq_kwargs = self.pesq.call_args
        np.testing.assert_array_equal(stoi_args[0], clean)
        np.testing.assert_array_equal(stoi_args[1], estimate)
        np.testing.assert_array_equal(pesq_args[1], clean)
        np.testing.assert_array_equal(pesq_args[2], estimate)
        self.assertEqual(stoi_args[2], 16000)
        self.assertEqual(stoi_kwargs, {"extended": False})
        self.assertEqual(pesq_args[0], 16000)
        self.assertEqual(pesq_kwargs, {"mode": "wb"})
        self.assertFalse(np.shares_memory(stoi_args[0], clean))
        self.assertFalse(np.shares_memory(pesq_args[2], estimate))
        np.testing.assert_array_equal(estimate, before)

    def test_missing_packages_preserve_si_sdr_and_report_reasons(self):
        with patch.object(metrics.importlib, "import_module", side_effect=ImportError("not installed")):
            result = metrics.evaluate_audio(self.clean, self.clean + self.noise)
        self.assertAlmostEqual(result["si_sdr_db"], 0.0)
        for name in ("stoi", "pesq_wb"):
            self.assertIsNone(result[name])
            self.assertIn("unavailable", result["reasons"][name])

    def test_stoi_short_utterance_placeholder_is_not_a_score(self):
        def short_utterance(*args, **kwargs):
            warnings.warn("Not enough STFT frames; returning 1e-5", RuntimeWarning)
            return 1e-5

        self.stoi.side_effect = short_utterance
        result = metrics.evaluate_audio(self.clean[:100], (self.clean + self.noise)[:100])
        self.assertIsNone(result["stoi"])
        self.assertIn("Not enough STFT frames", result["reasons"]["stoi"])

    def test_library_exceptions_and_nonfinite_results_are_explicit(self):
        self.stoi.return_value = float("nan")
        self.pesq.side_effect = RuntimeError("no utterances detected")
        result = metrics.evaluate_audio(self.clean, self.clean + self.noise)
        self.assertIsNone(result["stoi"])
        self.assertIn("nonfinite", result["reasons"]["stoi"])
        self.assertIsNone(result["pesq_wb"])
        self.assertIn("no utterances detected", result["reasons"]["pesq_wb"])
        json.dumps(result, allow_nan=False)

    def test_wideband_pesq_requires_sixteen_kilohertz(self):
        result = metrics.evaluate_audio(self.clean, self.clean + self.noise, sample_rate=8000)
        self.assertIsNone(result["pesq_wb"])
        self.assertIn("16000 Hz", result["reasons"]["pesq_wb"])
        self.pesq.assert_not_called()
        self.assertEqual(result["stoi"], 0.81)


if __name__ == "__main__":
    unittest.main()
