"""Offline checks for the portable evidence utilities; no model or device use."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_validation as batch
import build_report as report


class ReproductionUtilitiesTest(unittest.TestCase):
    def test_nested_location_and_preview_defaults(self):
        args = batch.parse_args([])
        self.assertTrue((batch.ROOT / 'user_dma_core.py').is_file())
        self.assertEqual(batch.repository_root(Path(__file__)), batch.ROOT)
        self.assertFalse(args.execute or args.compile_only)
        self.assertEqual(args.lstm_math_scope, 'decoder')
        self.assertIn(batch.OUTPUT_PARENT, args.output_root.parents)
        self.assertIn(batch.REPRODUCE_ROOT / 'work', batch.work_root(args).parents)
        self.assertFalse(args.output_root.exists())
        self.assertEqual(batch.PYTHON, sys.executable)

    def test_measured_output_rejected_before_execute(self):
        for directory in (batch.OLD, batch.HERE, batch.HERE.parent):
            with self.subTest(directory=directory), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    batch.parse_args(['--execute', '--output-root', str(directory)])
                self.assertEqual(error.exception.code, 2)

    def test_symlink_cannot_escape_reproduction_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results, protected = root / 'results', root / 'protected'
            results.mkdir(); protected.mkdir()
            (results / 'alias').symlink_to(protected, target_is_directory=True)
            with patch.object(batch, 'OUTPUT_PARENT', results):
                with self.assertRaisesRegex(RuntimeError, 'published evidence is protected'):
                    batch.validate_output_root(results / 'alias' / 'new')

    def test_hardlink_destination_cannot_overwrite_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); reference = root / 'original.wav'
            reference.write_bytes(b'original audio')
            fields = ('bin', 'compile_log', 'run_log', 'comparison_log', 'output', 'tokens', 'metrics', 'comparison')
            job = {field: str(root / field) for field in fields}
            Path(job['output']).hardlink_to(reference)
            with self.assertRaisesRegex(RuntimeError, 'aliases a frozen file'):
                batch.validate_destination_paths({'example': job}, {str(reference): batch.sha(reference)})
            self.assertEqual(reference.read_bytes(), b'original audio')

    def test_new_freeze_binds_both_portable_utilities(self):
        hashes = batch.source_hashes()
        for name in ('run_validation.py', 'build_report.py'):
            key = str((batch.HERE / name).relative_to(batch.ROOT))
            self.assertEqual(hashes[key], batch.sha(batch.HERE / name))
        self.assertNotIn('models/bigcodec/bigcodec_bin/accuracy_20260915/run_accuracy_batch.py', hashes)

    def test_published_measurement_hardlink_is_protected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            published = root / 'models/bigcodec/validation/selected/audio.wav'
            published.parent.mkdir(parents=True); published.write_bytes(b'published measurement')
            work = root / 'work'; work.mkdir()
            fields = ('bin', 'compile_log', 'run_log', 'comparison_log', 'output', 'tokens', 'metrics', 'comparison')
            job = {field: str(work / field) for field in fields}
            Path(job['output']).hardlink_to(published)
            with patch.object(batch, 'ROOT', root):
                with self.assertRaisesRegex(RuntimeError, 'aliases a frozen file'):
                    batch.validate_destination_paths({'example': job}, {})
            self.assertEqual(published.read_bytes(), b'published measurement')

    def test_native_entry_requires_explicit_execute_before_other_work(self):
        with patch.object(batch.socket, 'gethostname', side_effect=AssertionError('guard was bypassed')):
            with self.assertRaisesRegex(RuntimeError, 'requires --execute'):
                batch.guarded_runner(SimpleNamespace(execute=False))

    def test_default_report_preview_is_read_only_and_write_requires_data(self):
        output = io.StringIO()
        with patch.object(sys, 'argv', ['build_report.py']), redirect_stdout(output):
            report.main()
        preview = json.loads(output.getvalue())
        self.assertEqual(preview['status'], 'preview_no_reproduction_runs')
        self.assertEqual(Path(preview['output_root']), batch.DEFAULT_OUT)
        self.assertFalse(Path(preview['output_root']).exists())
        with patch.object(sys, 'argv', ['build_report.py', '--write']):
            with self.assertRaisesRegex(RuntimeError, 'No reproduction batch exists'):
                report.main()
        self.assertFalse(Path(preview['output_root']).exists())

    def test_incomplete_batch_cannot_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary); output = parent / 'fresh'; output.mkdir()
            (output / 'hardware_runs.json').write_text(json.dumps(dict(format='bigcodec-accuracy-hardware-runs-v1', runs={})))
            (output / 'batch_freeze.json').write_text('{}')
            with patch.object(batch, 'OUTPUT_PARENT', parent):
                with self.assertRaisesRegex(RuntimeError, 'all 16 completed runs'):
                    report.build(output, final=True)
            self.assertFalse((output / 'README.md').exists())

    def test_empty_report_preview_keeps_scope_and_resolves_evidence_links(self):
        fixture = dict(status='partial_preview', completed_runs=0,
                       precision_summaries={p: dict(before=None, after=None) for p in ('bf16', 'if8')},
                       cpu_fp32=dict(processing_rtf=2, processing_s=8, duration_s=4, core=7),
                       settings=dict(lstm_math_scope='decoder', center_quantizer_scores=True, compensated_codebook=True),
                       hardware=dict(version='0x40519e0a'),
                       evidence=dict(batch_manifest=dict(path=str(batch.DEFAULT_OUT / 'hardware_runs.json'))),
                       runs=[], pending=[])
        text = report.readme(fixture)
        for phrase in ('all 16 runs', 'one thread on core 7', 'RTF 1', '333.25 MHz', '2 GiB', 'the decoder LSTM stack'):
            self.assertIn(phrase, text)
        self.assertNotIn('all16', text)
        self.assertNotIn('build0x', text)
        import re
        links = re.findall(r'\]\(([^)]+)\)', text)
        for target in links:
            if target not in ('metrics_table.csv', 'metrics_summary.json'):
                self.assertTrue((batch.DEFAULT_OUT / target).resolve().is_file(), target)

    def test_new_run_version_uses_freeze_but_old_baseline_keeps_original(self):
        reference = json.loads((batch.OLD / 'cpu_reference_manifest.json').read_text())['cases'][0]
        metrics = json.loads((batch.OLD / 'fpga_bf16' / (reference['id'] + '.metrics.json')).read_text())
        metrics['dram_required_bytes'] = metrics.get('dram_required_bytes', 1)
        metrics['dram_addressable_bytes'] = metrics.get('dram_addressable_bytes', 2**31)
        report.validate_execution(metrics, reference, legacy=True)
        metrics['hardware_version'] = '0x12345678'
        report.validate_execution(metrics, reference, expected_version=0x12345678)
        with self.assertRaisesRegex(RuntimeError, 'Hardware version/width differs'):
            report.validate_execution(metrics, reference)
        with self.assertRaisesRegex(RuntimeError, 'Hardware version/width differs'):
            report.validate_execution(metrics, reference, legacy=True, expected_version=0x12345678)

    def test_version_check_runs_in_existing_runner_before_configuration(self):
        output = batch.DEFAULT_OUT
        job = {key: str(output / key) for key in ('output', 'tokens', 'metrics', 'comparison')}
        job.update(bin='fake.bin', input='fake.wav')
        freeze_path = output / 'batch_freeze.json'
        request = dict(freeze=str(freeze_path), job=job)
        freeze = dict(source_hashes={str(Path(batch.__file__).relative_to(batch.ROOT)): 'unused'},
                      reference_hashes={}, settings=dict(expected_version=0x40519e0a))
        for measured_version, success in ((0x40519e0a, True), (0xdeadbeef, False)):
            events = []
            fake = SimpleNamespace(configure_hardware_runtime=lambda **kwargs: events.append('configured'))
            def runner_main():
                # Simulate entry to the existing runner after its lock acquisition.
                events.append('runner_lock_acquired')
                fake.configure_hardware_runtime(dev='fake')
            fake.main = runner_main
            args = SimpleNamespace(execute=True, run_job=Path('request.json'), expected_version=0x40519e0a, timeout=300)
            def read_version(*unused):
                events.append('version_read')
                return struct.pack('<I', measured_version)
            with (patch.object(batch.socket, 'gethostname', return_value='italy'),
                  patch.object(batch, 'read', side_effect=lambda path: freeze if str(path) == str(freeze_path) else request),
                  patch.object(batch, 'check_hashes'), patch.dict(sys.modules, bigcodec_run_from_bin=fake),
                  patch.object(batch.os, 'open', return_value=99), patch.object(batch.os, 'pread', side_effect=read_version),
                  patch.object(batch.os, 'close'), patch.object(sys, 'argv', []), patch.object(sys, 'path', list(sys.path))):
                if success:
                    batch.guarded_runner(args)
                    self.assertEqual(events, ['runner_lock_acquired', 'version_read', 'configured'])
                else:
                    with self.assertRaisesRegex(RuntimeError, 'Board version changed'):
                        batch.guarded_runner(args)
                    self.assertEqual(events, ['runner_lock_acquired', 'version_read'])


if __name__ == '__main__':
    unittest.main()
