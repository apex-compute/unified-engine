"""Raw waveform and token comparison semantics, including degenerate audio."""
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
import bigcodec_compare as compare
from bigcodec_common import CHECKPOINT_SHA256, TOKEN_FORMAT, save_tokens, sha256_file


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def wav(self, name, samples, rate=16000, subtype='FLOAT'):
        path = self.root / name
        sf.write(path, np.asarray(samples), rate, subtype=subtype)
        return path

    def test_gain_delay_and_polarity_errors_remain_unfitted(self):
        reference = np.asarray([1, -1, 0, 0], dtype=np.float32)
        path = self.wav('reference.wav', reference)
        cases = [
            ('gain', reference * 2, 1.0, 0.0, 1.0),
            ('polarity', -reference, 2.0, -20 * math.log10(2), -1.0),
            ('delay', np.roll(reference, 1), math.sqrt(3), -10 * math.log10(3), -0.5),
        ]
        for name, actual, relative, snr, cosine in cases:
            result = compare.compare_audio(path, self.wav(name + '.wav', actual))
            self.assertAlmostEqual(result['relative_l2'], relative)
            self.assertAlmostEqual(result['snr_db'], snr)
            self.assertAlmostEqual(result['cosine_similarity'], cosine)
            self.assertAlmostEqual(result['rmse'], np.sqrt(np.mean((actual-reference).astype(np.float64)**2)))
            self.assertIn('no gain, delay or polarity fitting', result['alignment'])
            self.assertEqual(result['metric_status']['snr_db'], 'finite')

    def test_exact_nonzero_match_reports_positive_infinite_snr_without_json_nan(self):
        path = self.wav('same.wav', [0.25, -0.5, 1.0, 0.0])
        result = compare.compare_audio(path, path)
        self.assertEqual(result['relative_l2'], 0)
        self.assertEqual(result['rmse'], 0)
        self.assertEqual(result['max_abs_error'], 0)
        self.assertEqual(result['cosine_similarity'], 1)
        self.assertIsNone(result['snr_db'])
        self.assertEqual(result['metric_status']['snr_db'], 'positive_infinity_exact_match')
        json.dumps(result, allow_nan=False)

    def test_silence_and_zero_energy_do_not_claim_zero_relative_error(self):
        silence = self.wav('silence.wav', np.zeros(4))
        signal = self.wav('signal.wav', np.ones(4))
        same = compare.compare_audio(silence, silence)
        self.assertEqual(same['rmse'], 0)
        self.assertIsNone(same['relative_l2'])
        self.assertEqual(same['metric_status']['snr_db'], 'undefined_zero_reference_and_error')
        false_sound = compare.compare_audio(silence, signal)
        self.assertEqual(false_sound['rmse'], 1)
        self.assertIsNone(false_sound['relative_l2'])
        self.assertIsNone(false_sound['cosine_similarity'])
        self.assertEqual(false_sound['metric_status']['snr_db'], 'negative_infinity_zero_reference')
        lost_sound = compare.compare_audio(signal, silence)
        self.assertEqual(lost_sound['relative_l2'], 1)
        self.assertEqual(lost_sound['snr_db'], 0)
        self.assertIsNone(lost_sound['cosine_similarity'])
        for report in (same, false_sound, lost_sound):
            json.dumps(report, allow_nan=False)

    def test_sample_and_channel_counts_have_correct_rmse_denominator(self):
        reference = self.wav('reference.wav', [[1.0, 0], [0, 1.0]])
        actual = self.wav('actual.wav', [[2.0, 0], [0, 2.0]])
        result = compare.compare_audio(reference, actual)
        self.assertEqual((result['samples'], result['channels']), (2, 2))
        self.assertEqual(result['duration_s'], 2 / 16000)
        self.assertAlmostEqual(result['rmse'], math.sqrt(.5))
        self.assertAlmostEqual(result['reference_rms'], math.sqrt(.5))

    def test_reject_shape_rate_nonfinite_and_overflowing_audio(self):
        reference = self.wav('reference.wav', np.ones(4))
        for name, samples, rate in [('length', np.ones(3), 16000),
                                    ('channels', np.ones((4, 2)), 16000),
                                    ('rate', np.ones(4), 8000),
                                    ('nan', np.full(4, np.nan), 16000),
                                    ('inf', np.full(4, np.inf), 16000)]:
            with self.assertRaises(ValueError, msg=name):
                compare.compare_audio(reference, self.wav(name + '.wav', samples, rate))
        empty = self.wav('empty.wav', np.array([], dtype=np.float32))
        with self.assertRaises(ValueError):
            compare.compare_audio(empty, empty)
        huge = self.wav('huge.wav', np.full(4, 1e200), subtype='DOUBLE')
        with self.assertRaisesRegex(ValueError, 'finite comparison range'):
            compare.compare_audio(huge, huge)
        tiny = self.wav('tiny.wav', np.full(4, 1e-200), subtype='DOUBLE')
        with self.assertRaisesRegex(ValueError, 'below the finite comparison range'):
            compare.compare_audio(tiny, tiny)

    def test_metric_ratios_do_not_overflow_before_the_final_value(self):
        small = self.wav('small.wav', np.full(4, 1e-150), subtype='DOUBLE')
        large = self.wav('large.wav', np.full(4, 1e150), subtype='DOUBLE')
        same = compare.compare_audio(large, large)
        self.assertEqual(same['cosine_similarity'], 1)
        report = compare.compare_audio(small, large)
        self.assertAlmostEqual(report['relative_l2'] / 1e300, 1)
        self.assertAlmostEqual(report['snr_db'], -6000)
        self.assertAlmostEqual(report['cosine_similarity'], 1)
        json.dumps(report, allow_nan=False)

    def token_pair(self, *, metadata_change=None):
        metadata = dict(format=TOKEN_FORMAT, sample_rate=16000, hop_length=200,
                        source_rate=16000, source_samples=200, source_channels=1,
                        native_samples=200, padded_samples=400,
                        checkpoint_sha256=CHECKPOINT_SHA256, source_sha256='a' * 64)
        reference, actual = self.root / 'reference.npz', self.root / 'actual.npz'
        save_tokens(reference, np.asarray([0, 8191]), metadata)
        actual_metadata = dict(metadata)
        if metadata_change:
            actual_metadata.update(metadata_change)
        save_tokens(actual, np.asarray([8191, 8191]), actual_metadata)
        return reference, actual, metadata

    def test_tokens_report_exact_id_agreement_with_source_and_checkpoint_binding(self):
        reference, actual, metadata = self.token_pair()
        report = dict(sample_rate=16000, samples=200, channels=1)
        result = compare.compare_tokens(reference, actual, report)
        self.assertEqual(result['count'], 2)
        self.assertEqual(result['equal'], 1)
        self.assertEqual(result['mismatches'], 1)
        self.assertEqual(result['match_fraction'], .5)
        self.assertEqual(result['source_sha256'], metadata['source_sha256'])
        self.assertEqual(result['checkpoint_sha256'], CHECKPOINT_SHA256)
        self.assertEqual(result['reference_sha256'], sha256_file(reference))
        self.assertEqual(result['actual_sha256'], sha256_file(actual))
        # Codebook ID numeric distance does not change token agreement.
        save_tokens(actual, np.asarray([1, 8191]), metadata)
        self.assertEqual(compare.compare_tokens(reference, actual, report)['match_fraction'], .5)

    def test_token_provenance_and_wav_dimensions_must_match(self):
        report = dict(sample_rate=16000, samples=200, channels=1)
        reference, actual, _ = self.token_pair(metadata_change={'source_sha256': 'b' * 64})
        with self.assertRaisesRegex(ValueError, 'same input audio'):
            compare.compare_tokens(reference, actual, report)
        reference, actual, metadata = self.token_pair()
        for key, value in (('sample_rate', 8000), ('samples', 199), ('channels', 2)):
            mismatched = dict(report, **{key: value})
            with self.assertRaisesRegex(ValueError, 'WAV dimensions'):
                compare.compare_tokens(reference, actual, mismatched)
        del metadata['source_sha256']
        save_tokens(reference, np.asarray([0, 8191]), metadata)
        save_tokens(actual, np.asarray([0, 8191]), metadata)
        with self.assertRaisesRegex(ValueError, 'SHA256 provenance'):
            compare.compare_tokens(reference, actual, report)

    def test_cli_preserves_all_source_bytes_and_emits_reproducible_json(self):
        reference = self.wav('reference.wav', np.linspace(-1, 1, 200))
        actual = self.wav('actual.wav', np.linspace(-.5, .5, 200))
        reference_tokens, actual_tokens, _ = self.token_pair()
        inputs = [reference, actual, reference_tokens, actual_tokens]
        before = {path: path.read_bytes() for path in inputs}
        destination = self.root / 'report.json'
        arguments = ['compare', '--reference', str(reference), '--actual', str(actual),
                     '--reference-tokens', str(reference_tokens), '--actual-tokens', str(actual_tokens),
                     '--report', str(destination)]
        output = io.StringIO()
        with patch.object(sys, 'argv', arguments), contextlib.redirect_stdout(output):
            compare.main()
        result = json.loads(destination.read_text())
        self.assertEqual(json.loads(output.getvalue()), result)
        self.assertEqual(result['tokens']['match_fraction'], .5)
        self.assertAlmostEqual(result['relative_l2'], .5)
        for path, data in before.items():
            self.assertEqual(path.read_bytes(), data)

    def test_report_cannot_overwrite_wav_or_token_aliases(self):
        reference = self.wav('reference.wav', np.ones(200))
        actual = self.wav('actual.wav', np.ones(200))
        reference_tokens, actual_tokens, _ = self.token_pair()
        for original in (reference, actual, reference_tokens, actual_tokens):
            data = original.read_bytes()
            for kind in ('same', 'symlink', 'hardlink'):
                alias = self.root / (original.name + '.' + kind)
                if kind == 'same':
                    alias = original
                elif kind == 'symlink':
                    alias.symlink_to(original)
                else:
                    os.link(original, alias)
                args = ['compare', '--reference', str(reference), '--actual', str(actual),
                        '--reference-tokens', str(reference_tokens), '--actual-tokens', str(actual_tokens),
                        '--report', str(alias)]
                with patch.object(sys, 'argv', args), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        compare.main()
                self.assertEqual(original.read_bytes(), data)

    def test_cli_requires_both_token_files(self):
        with patch.object(sys, 'argv', ['compare', '--reference', 'reference.wav',
                                       '--actual', 'actual.wav', '--reference-tokens', 'one.npz']), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                compare.main()


if __name__ == '__main__':
    unittest.main()
