"""Independent energy/time pooling, frozen provenance and output protection."""
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
import bigcodec_accuracy_report as report
from bigcodec_common import CHECKPOINT_SHA256, TOKEN_FORMAT, save_tokens, sha256_file


class AccuracyReportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.manifest = dict(format='bigcodec-accuracy-inputs-v1', checkpoint_sha256=CHECKPOINT_SHA256,
            baseline='before', profiles=[dict(name=name, hardware_version='0x90f1f464',
            precision={'convolutions': 'BF16'}) for name in ('before', 'after')], cases=[])
        self.path = self.root / 'manifest.json'

    def artifact(self, path):
        return dict(path=path.name, sha256=sha256_file(path))

    def json_file(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value, allow_nan=False))
        return self.artifact(path)

    def wave(self, name, values):
        path = self.root / name
        sf.write(path, np.asarray(values), 16000, subtype='FLOAT')
        return self.artifact(path)

    def tokens(self, name, values, metadata):
        path = self.root / name
        save_tokens(path, np.asarray(values, dtype=np.uint16), metadata)
        return self.artifact(path)

    def add_case(self, name, count=200, amplitude=1., *, silence=False):
        duration = count / 16000
        reference = np.full(count, 0. if silence else amplitude, dtype=np.float32)
        source = self.wave(name + '_input.wav', reference)
        metadata = dict(format=TOKEN_FORMAT, sample_rate=16000, hop_length=200,
            source_rate=16000, source_samples=count, source_channels=1, native_samples=count,
            padded_samples=count + 200 - count % 200, source_sha256=source['sha256'],
            checkpoint_sha256=CHECKPOINT_SHA256)
        ids = np.arange(metadata['padded_samples'] // 200, dtype=np.uint16)
        wave = self.wave(name + '_cpu.wav', reference)
        tokens = self.tokens(name + '_cpu.npz', ids, metadata)
        cpu = dict(backend='pytorch-cpu-fp32', mode='roundtrip', checkpoint_sha256=CHECKPOINT_SHA256,
            audio=metadata, output_finite=True, output_samples=count, token_count=len(ids),
            output_sha256=wave['sha256'], tokens_sha256=tokens['sha256'], duration_s=duration,
            processing_s=duration * 2, processing_rtf=2., threads=1, cpu_affinity=[7])
        case = dict(name=name, input=source, reference=dict(wave=wave, tokens=tokens,
            metrics=self.json_file(name + '_cpu.json', cpu)), profiles={})
        for profile in ('before', 'after'):
            error = amplitude if profile == 'before' else 2. if count == 200 else 1.
            actual = self.wave(name + '_' + profile + '.wav', reference + error)
            matches = (profile == 'before') == (count == 200)
            actual_tokens = self.tokens(name + '_' + profile + '.npz', ids if matches else ids + 10, metadata)
            processing = duration * (2 if count == 200 else 6)
            if profile == 'after':
                processing *= 1.1 if count == 200 else .9
            execution = processing - .002
            native = execution - .001
            metrics = dict(backend='hardware', execution_scope='whole utterance',
                **{key: 1 for key in report.COUNTS}, cpu_neural_ops=0,
                finite=True, waveform_padding_finite=True, waveform_padding_nonzero=0,
                checkpoint_sha256=CHECKPOINT_SHA256, input_sha256=source['sha256'],
                output_sha256=actual['sha256'], tokens_sha256=actual_tokens['sha256'],
                source_samples=count, output_samples=count, source_sample_rate=16000,
                compiled_samples=metadata['padded_samples'], tokens=len(ids),
                hardware_version='0x90f1f464', axi_data_width_bits=256,
                precision={'convolutions': 'BF16'}, fpga_counter_wrap_possible=False,
                detected_clock_ns=1., fpga_cycles=round(native * 1e9), fpga_execution_s=native,
                audio_duration_s=duration, audio_processing_s=processing, audio_rtf=processing / duration,
                audio_preprocess_s=.001, execution_s=execution, audio_postprocess_s=.001,
                artifact_load_s=.5, model_upload_s=.1, total_elapsed_s=processing + .75,
                parameter_bytes=128, program_bytes=64, instructions=2, model_upload_bytes=192,
                input_upload_bytes=metadata['padded_samples'] * 128,
                output_read_bytes=(metadata['padded_samples'] + len(ids)) * 128, bin_sha256='b' * 64)
            case['profiles'][profile] = dict(wave=actual, tokens=actual_tokens,
                metrics=self.json_file(name + '_' + profile + '.json', metrics))
        self.manifest['cases'].append(case)
        return case

    def build(self):
        self.path.write_text(json.dumps(self.manifest, allow_nan=False))
        return report.build_report(self.path)

    def mutate_metrics(self, case, profile, **changes):
        spec = case['profiles'][profile]['metrics']
        value = json.loads((self.root / spec['path']).read_text())
        value.update(changes)
        case['profiles'][profile]['metrics'] = self.json_file(spec['path'], value)

    def test_energy_and_duration_weighted_pool_preserves_individual_regressions(self):
        self.add_case('short', 200, 1)
        self.add_case('long', 600, 10)
        result = self.build()
        before, after = result['pooled']['before'], result['pooled']['after']
        self.assertEqual(before['reference_energy'], 200 + 60000)
        self.assertEqual(before['error_energy'], 200 + 60000)
        self.assertEqual(after['error_energy'], 800 + 600)
        self.assertAlmostEqual(after['relative_l2'], math.sqrt(1400 / 60200))
        self.assertNotAlmostEqual(after['relative_l2'], (2. + .1) / 2)
        self.assertAlmostEqual(before['processing_rtf'], 5.)
        self.assertAlmostEqual(after['processing_rtf'], 4.6)
        self.assertAlmostEqual(after['startup_inclusive_rtf'], (4.6 * .05 + 1.5) / .05)
        self.assertEqual((before['token_matches'], after['token_matches'], after['token_count']), (2, 4, 6))
        self.assertEqual(result['regressions']['after'], dict(waveform=['short'], tokens=['short'], processing=['short']))
        self.assertEqual(result['pooled_changes']['after']['waveform'], 'improved')
        self.assertEqual(result['cpu']['processing_rtf'], 2.)
        self.assertIn('regressed', report.markdown(result))
        json.dumps(result, allow_nan=False)

    def test_silent_reference_has_undefined_relative_error_but_honest_sse_regression(self):
        self.add_case('silence', silence=True)
        result = self.build()
        for row in result['pooled'].values():
            self.assertIsNone(row['relative_l2'])
            self.assertEqual(row['relative_l2_status'], 'undefined_zero_reference')
        self.assertEqual(result['regressions']['after']['waveform'], ['silence'])
        self.assertIn('undefined (zero reference)', report.markdown(result))
        json.dumps(result, allow_nan=False)

    def test_execution_counts_hash_bindings_build_and_time_must_match(self):
        case = self.add_case('clip')
        original = dict(case['profiles']['after']['metrics'])
        old_bytes = (self.root / original['path']).read_bytes()
        for field, value in [('program_kicks', 2), ('halts', True), ('cpu_neural_ops', 1),
                             ('output_sha256', 'c' * 64), ('input_sha256', 'c' * 64),
                             ('hardware_version', '0x00000001'), ('fpga_counter_wrap_possible', True),
                             ('audio_rtf', 99), ('fpga_execution_s', 5), ('total_elapsed_s', .01),
                             ('input_upload_bytes', 1), ('model_upload_bytes', 999)]:
            with self.subTest(field=field):
                (self.root / original['path']).write_bytes(old_bytes)
                case['profiles']['after']['metrics'] = dict(original)
                self.mutate_metrics(case, 'after', **{field: value})
                with self.assertRaises(ValueError):
                    self.build()

    def test_complete_profiles_and_frozen_files_are_required(self):
        case = self.add_case('clip')
        after = case['profiles'].pop('after')
        with self.assertRaisesRegex(ValueError, 'Every profile'):
            self.build()
        case['profiles']['after'] = after
        actual = self.root / after['wave']['path']
        actual.write_bytes(actual.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.build()

    def test_cpu_provenance_and_token_metadata_cannot_be_substituted(self):
        case = self.add_case('clip')
        cpu_spec = case['reference']['metrics']
        cpu = json.loads((self.root / cpu_spec['path']).read_text())
        cpu['output_sha256'] = 'e' * 64
        case['reference']['metrics'] = self.json_file(cpu_spec['path'], cpu)
        with self.assertRaisesRegex(ValueError, 'CPU reference provenance'):
            self.build()
        cpu['output_sha256'] = case['reference']['wave']['sha256']
        case['reference']['metrics'] = self.json_file(cpu_spec['path'], cpu)
        spec = case['profiles']['after']['tokens']
        tokens, metadata = report.load_tokens(self.root / spec['path'])
        metadata['source_sha256'] = 'e' * 64
        case['profiles']['after']['tokens'] = self.tokens(spec['path'], tokens, metadata)
        with self.assertRaisesRegex(ValueError, 'same input audio'):
            self.build()

    def test_optional_bin_hash_is_verified_and_omitted_binary_is_labeled(self):
        case = self.add_case('clip')
        self.assertEqual(self.build()['cases'][0]['profiles']['after']['bin_hash_verification'], 'runner-recorded only')
        binary = self.root / 'small.bin'
        binary.write_bytes(b'synthetic artifact for hash checking')
        spec = self.artifact(binary)
        case['profiles']['after']['bin'] = spec
        self.mutate_metrics(case, 'after', bin_sha256=spec['sha256'])
        value = self.build()['cases'][0]['profiles']['after']
        self.assertEqual(value['bin_hash_verification'], 'file rehashed')
        binary.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.build()

    def test_cli_default_is_read_only_and_outputs_cannot_alias_sources_or_each_other(self):
        case = self.add_case('clip')
        result = self.build()
        sources = {path: path.read_bytes() for path in self.root.iterdir()}
        with patch.object(sys, 'argv', ['accuracy_report', str(self.path)]), contextlib.redirect_stdout(io.StringIO()) as output:
            report.main()
        self.assertEqual(json.loads(output.getvalue()), result)
        self.assertEqual({path: path.read_bytes() for path in self.root.iterdir()}, sources)
        output_json, output_md = self.root / 'result.json', self.root / 'README.md'
        report.write_outputs(result, output_json, output_md)
        self.assertEqual(json.loads(output_json.read_text()), result)
        self.assertIn('Startup-inclusive RTF', output_md.read_text())
        for source in (self.path, self.root / case['input']['path'], self.root / case['reference']['tokens']['path']):
            alias = self.root / 'alias'
            os.link(source, alias)
            try:
                with self.assertRaisesRegex(ValueError, 'overwrite an input'):
                    report.write_outputs(result, alias)
            finally:
                alias.unlink()
        with self.assertRaises(ValueError):
            report.write_outputs(result, output_json, output_json)
        for path, value in sources.items():
            self.assertEqual(path.read_bytes(), value)


if __name__ == '__main__':
    unittest.main()
