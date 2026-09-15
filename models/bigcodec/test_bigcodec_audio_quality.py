"""Quality metrics must retain raw errors and expose undefined speech scores."""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bigcodec_audio_quality as quality
from bigcodec_compare import compare_audio


class QualityTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.signal = np.random.default_rng(71).normal(0, .08, 16000).astype(np.float32)

    def wav(self, name, data, rate=16000):
        path = self.root / name
        sf.write(path, data, rate, subtype='FLOAT')
        return path

    def test_identity_has_raw_and_spectral_zero_without_json_infinity(self):
        path = self.wav('same.wav', self.signal)
        report = quality.compare_quality(path, path)
        self.assertEqual(report['waveform'], compare_audio(path, path))
        self.assertEqual(report['quality']['mr_spectral_convergence'], 0)
        self.assertEqual(report['quality']['mr_log_spectral_distance_db'], 0)
        self.assertAlmostEqual(report['quality']['stoi'], 1)
        self.assertAlmostEqual(report['quality']['estoi'], 1)
        self.assertGreater(report['quality']['pesq_wb'], 4.6)
        self.assertEqual(set(report['quality']['metric_status'].values()), {'finite'})
        json.dumps(report, allow_nan=False)

    def test_gain_is_not_fitted_out_of_raw_or_spectral_error(self):
        reference = self.wav('reference.wav', self.signal)
        actual = self.wav('half.wav', self.signal / 2)
        report = quality.compare_quality(reference, actual)
        self.assertAlmostEqual(report['waveform']['relative_l2'], .5)
        self.assertAlmostEqual(report['quality']['mr_spectral_convergence'], .5)
        for row in report['quality']['spectral']:
            self.assertAlmostEqual(row['spectral_convergence'], .5)

    def test_phase_geometry_is_explicit_not_disguised_as_waveform_agreement(self):
        reference = self.wav('reference.wav', self.signal)
        actual = self.wav('inverted.wav', -self.signal)
        report = quality.compare_quality(reference, actual)
        self.assertEqual(report['waveform']['relative_l2'], 2)
        self.assertEqual(report['quality']['mr_spectral_convergence'], 0)
        self.assertIn('no phase', report['analysis']['spectral']['convergence'])
        self.assertIn('no delay, gain or polarity', report['analysis']['external_alignment'])

    def test_48khz_raw_metrics_preserved_and_quality_uses_shared_16khz(self):
        x = np.random.default_rng(81).normal(0, .05, 48000).astype(np.float32)
        reference = self.wav('reference48.wav', x, 48000)
        actual = self.wav('actual48.wav', x / 2, 48000)
        report = quality.compare_quality(reference, actual)
        self.assertEqual(report['waveform'], compare_audio(reference, actual))
        self.assertEqual(report['waveform']['samples'], 48000)
        self.assertEqual(report['waveform']['sample_rate'], 48000)
        self.assertEqual(report['quality']['samples'], 16000)
        self.assertEqual(report['quality']['sample_rate'], 16000)
        self.assertTrue(report['analysis']['resampling']['applied'])
        self.assertEqual(report['analysis']['resampling']['down'], 3)
        self.assertAlmostEqual(report['quality']['mr_spectral_convergence'], .5)

    def test_short_and_silent_audio_has_explicit_status_not_success_sentinel(self):
        short = self.wav('short.wav', self.signal[:1000])
        report = quality.compare_quality(short, short)
        self.assertEqual(report['quality']['metric_status']['pesq_wb'], 'insufficient_duration')
        self.assertEqual(report['quality']['metric_status']['stoi'], 'insufficient_active_frames')
        self.assertIsNone(report['quality']['pesq_wb'])
        self.assertIsNone(report['quality']['stoi'])
        zero = self.wav('zero.wav', np.zeros(16000))
        signal = self.wav('signal.wav', self.signal)
        for ref, act, status in ((zero, zero, 'undefined_zero_reference'),
                                 (zero, signal, 'undefined_zero_reference'),
                                 (signal, zero, 'undefined_zero_actual')):
            report = quality.compare_quality(ref, act)
            self.assertEqual(set(report['quality']['metric_status'].values()), {status})
            self.assertIsNone(report['quality']['stoi'])
            self.assertIsNone(report['quality']['pesq_wb'])
            json.dumps(report, allow_nan=False)
        one = self.wav('one.wav', np.asarray([.2]))
        self.assertEqual(quality.compare_quality(one, one)['quality']['mr_spectral_convergence'], 0)

    def test_invalid_inputs_fail_without_implicit_rate_or_channel_conversion(self):
        reference = self.wav('reference.wav', self.signal)
        for name, data, rate in (('length', self.signal[:-1], 16000),
                ('rate', self.signal, 8000), ('nan', np.full(16000, np.nan), 16000),
                ('infinity', np.full(16000, np.inf), 16000)):
            with self.assertRaises(ValueError):
                quality.compare_quality(reference, self.wav(name + '.wav', data, rate))
        stereo = self.wav('stereo.wav', np.repeat(self.signal[:, None], 2, axis=1))
        with self.assertRaisesRegex(ValueError, 'mono'):
            quality.compare_quality(stereo, stereo)

    def test_original_delay_stays_in_raw_comparison(self):
        reference = self.wav('reference.wav', self.signal)
        actual = self.wav('delayed.wav', np.concatenate(([0], self.signal[:-1])))
        result = quality.compare_quality(reference, actual)
        expected = np.linalg.norm(self.signal.astype(float) - np.concatenate(([0], self.signal[:-1]))) / np.linalg.norm(self.signal.astype(float))
        self.assertAlmostEqual(result['waveform']['relative_l2'], expected)
        self.assertGreater(result['waveform']['relative_l2'], 1)

    def test_cli_report_is_reproducible_and_cannot_overwrite_aliases(self):
        path = self.wav('input.wav', self.signal[:1000])
        before = path.read_bytes()
        destination = self.root / 'report.json'
        args = ['quality', '--reference', str(path), '--actual', str(path), '--report', str(destination)]
        output = io.StringIO()
        with patch.object(sys, 'argv', args), contextlib.redirect_stdout(output):
            quality.main()
        self.assertEqual(json.loads(output.getvalue()), json.loads(destination.read_text()))
        self.assertEqual(path.read_bytes(), before)
        symlink, hardlink = self.root / 'symlink.wav', self.root / 'hardlink.wav'
        symlink.symlink_to(path)
        os.link(path, hardlink)
        for alias in (path, symlink, hardlink):
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                quality.write_report({}, alias, (path,))
        self.assertEqual(path.read_bytes(), before)

    def test_spectral_windows_match_independent_numpy_dft(self):
        signal = quality._dependencies()[0]
        x, y = self.signal[:5000].astype(float), self.signal[:5000].astype(float).copy()
        y[100:110] *= 2
        result = quality.spectral_metrics(x, y, signal)
        for row, n in zip(result['spectral'], quality.WINDOWS):
            hop, window = n // 4, np.hanning(n + 1)[:-1]
            def dft(v):
                padded = np.pad(v, (n // 2, n // 2))
                padded = np.pad(padded, (0, (-(len(padded) - n)) % hop))
                frames = np.lib.stride_tricks.sliding_window_view(padded, n)[::hop]
                return np.abs(np.fft.rfft(frames * window, axis=1) / window.sum())
            a, b = dft(x), dft(y)
            self.assertAlmostEqual(row['spectral_convergence'], np.linalg.norm(a - b) / np.linalg.norm(a))
            self.assertEqual(row['frames'], a.shape[0])


if __name__ == '__main__':
    unittest.main()
