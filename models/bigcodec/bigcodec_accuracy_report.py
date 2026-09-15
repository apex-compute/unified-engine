#!/usr/bin/env python3
"""Audit frozen CPU/FPGA artifacts and compare complete paired profiles offline.

Usage: python bigcodec_accuracy_report.py manifest.json [--output results.json]
       [--markdown README.md]

The manifest uses format "bigcodec-accuracy-inputs-v1", checkpoint_sha256,
baseline (a profile name), profiles [{name, hardware_version, precision}], and
cases [{name, input, reference: {wave, tokens, metrics}, profiles: {name:
{wave, tokens, metrics, bin?}}}]. Each artifact is {path, sha256}; paths resolve
relative to the manifest. Profile precision is an expected subset of the saved
runner settings. Optional top-level evidence is a list of frozen artifacts.

All profiles must contain the same complete case set. Small artifacts are read
and hashed; supplying an optional bin also rehashes that binary. Otherwise bin
hashes are explicitly identified as runner-recorded. No inference or hardware
access occurs. By default the report is printed and no files are written.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import soundfile as sf

from bigcodec_common import CHECKPOINT_SHA256, load_tokens, sha256_file
from bigcodec_compare import compare_audio, compare_tokens


COUNTS = ('model_upload_writes', 'input_upload_writes', 'program_kicks', 'halts', 'output_reads')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def reject(value):
        raise ValueError(f'Nonfinite JSON value: {value}')
    return json.loads(Path(path).read_text(), parse_constant=reject)


def digest(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Invalid SHA256')
    return value


def number(value, name, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), f'Invalid {name}')
    return value


def close(actual, expected, name):
    number(actual, name)
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), f'Metric mismatch: {name}')


class Artifacts:
    def __init__(self, root):
        self.root, self.entries = root, {}

    def add(self, spec):
        require(isinstance(spec, dict) and isinstance(spec.get('path'), str), 'Invalid artifact entry')
        path = (self.root / spec['path']).resolve()
        expected = digest(spec.get('sha256'))
        require(path.is_file() and sha256_file(path) == expected, f'Artifact hash mismatch: {path}')
        if path in self.entries:
            require(self.entries[path]['sha256'] == expected, f'Conflicting artifact hash: {path}')
        self.entries[path] = dict(path=str(path), sha256=expected, bytes=path.stat().st_size)
        return path

    def recheck(self):
        for path, entry in self.entries.items():
            require(sha256_file(path) == entry['sha256'], f'Artifact changed during report: {path}')


def reference_case(case, artifacts, checkpoint):
    source = artifacts.add(case['input'])
    files = {key: artifacts.add(case['reference'][key]) for key in ('wave', 'tokens', 'metrics')}
    cpu, metadata = read_json(files['metrics']), load_tokens(files['tokens'])[1]
    info, source_info = sf.info(files['wave']), sf.info(source)
    require(info.frames > 0 and info.channels == 1 and info.frames == source_info.frames
            and info.samplerate == source_info.samplerate, 'CPU WAV does not preserve source rate/count')
    require(metadata['source_samples'] == info.frames and metadata['source_rate'] == info.samplerate
            and metadata['source_channels'] == source_info.channels
            and metadata.get('source_sha256') == case['input']['sha256'], 'CPU token source mismatch')
    require(cpu.get('backend') == 'pytorch-cpu-fp32' and cpu.get('mode') == 'roundtrip'
            and cpu.get('checkpoint_sha256') == checkpoint == metadata['checkpoint_sha256']
            and cpu.get('audio') == metadata and cpu.get('output_finite') is True
            and cpu.get('output_samples') == info.frames
            and cpu.get('token_count') == metadata['padded_samples'] // 200
            and cpu.get('output_sha256') == case['reference']['wave']['sha256']
            and cpu.get('tokens_sha256') == case['reference']['tokens']['sha256'],
            'CPU reference provenance mismatch')
    duration = info.frames / info.samplerate
    processing = number(cpu.get('processing_s'), 'CPU processing seconds')
    close(cpu.get('duration_s'), duration, 'CPU duration')
    close(cpu.get('processing_rtf'), processing / duration, 'CPU processing RTF')
    return dict(files=files, source=source, metadata=metadata, source_sha256=case['input']['sha256'],
                duration_s=duration, cpu_processing_s=processing,
                cpu_threads=cpu.get('threads'), cpu_affinity=cpu.get('cpu_affinity'))


def execution(metrics, reference, profile, spec):
    meta = reference['metadata']
    require(metrics.get('backend') == 'hardware' and metrics.get('execution_scope') == 'whole utterance'
            and all(type(metrics.get(key)) is int and metrics[key] == 1 for key in COUNTS)
            and type(metrics.get('cpu_neural_ops')) is int and metrics['cpu_neural_ops'] == 0
            and metrics.get('finite') is True and metrics.get('waveform_padding_finite') is True
            and type(metrics.get('waveform_padding_nonzero')) is int and metrics['waveform_padding_nonzero'] == 0,
            'FPGA execution contract failed')
    require(metrics.get('checkpoint_sha256') == meta['checkpoint_sha256']
            and metrics.get('input_sha256') == reference['source_sha256']
            and metrics.get('output_sha256') == spec['wave']['sha256']
            and metrics.get('tokens_sha256') == spec['tokens']['sha256'], 'FPGA artifact provenance mismatch')
    require(metrics.get('source_samples') == metrics.get('output_samples') == meta['source_samples']
            and metrics.get('source_sample_rate') == meta['source_rate']
            and metrics.get('compiled_samples') == meta['padded_samples']
            and metrics.get('tokens') == meta['padded_samples'] // 200, 'FPGA dimensions mismatch')
    require(metrics.get('hardware_version') == profile['hardware_version']
            and metrics.get('axi_data_width_bits') == 256, 'FPGA build or AXI width mismatch')
    require(isinstance(metrics.get('precision'), dict)
            and all(metrics['precision'].get(key) == value for key, value in profile['precision'].items()),
            'FPGA precision profile mismatch')
    require(metrics.get('fpga_counter_wrap_possible') is False, 'Native cycle counter may have wrapped')
    clock = number(metrics.get('detected_clock_ns'), 'clock period', positive=True)
    require(type(metrics.get('fpga_cycles')) is int and metrics['fpga_cycles'] >= 0, 'Invalid FPGA cycles')
    native = number(metrics.get('fpga_execution_s'), 'native seconds')
    close(native, metrics['fpga_cycles'] * clock / 1e9, 'native cycle time')
    duration = reference['duration_s']
    processing = number(metrics.get('audio_processing_s'), 'processing seconds')
    phases = [number(metrics.get(key), key) for key in ('audio_preprocess_s', 'execution_s', 'audio_postprocess_s')]
    close(processing, sum(phases), 'processing phase sum')
    close(metrics.get('audio_duration_s'), duration, 'audio duration')
    close(metrics.get('audio_rtf'), processing / duration, 'processing RTF')
    elapsed = number(metrics.get('total_elapsed_s'), 'runner elapsed seconds')
    startup = sum(number(metrics.get(key), key) for key in ('artifact_load_s', 'model_upload_s'))
    require(elapsed + 1e-8 >= processing + startup, 'Runner elapsed time omits measured startup')
    for key in ('parameter_bytes', 'program_bytes', 'instructions', 'model_upload_bytes'):
        require(type(metrics.get(key)) is int and metrics[key] > 0, f'Invalid {key}')
    require(metrics['program_bytes'] == 32 * metrics['instructions'] and metrics['program_bytes'] % 64 == 0
            and 0 <= metrics['model_upload_bytes'] - metrics['parameter_bytes'] - metrics['program_bytes'] < 128,
            'Program/resident sizes mismatch')
    for key, expected in (('input_upload_bytes', meta['padded_samples'] * 128),
                          ('output_read_bytes', (meta['padded_samples'] + metrics['tokens']) * 128)):
        require(type(metrics.get(key)) is int and metrics[key] == expected, f'Invalid {key}')
    return dict(processing_s=processing, processing_rtf=processing / duration,
                startup_inclusive_s=elapsed, startup_inclusive_rtf=elapsed / duration,
                native_s=native, native_rtf=native / duration, clock_ns=clock,
                hardware_version=metrics['hardware_version'], precision=metrics['precision'],
                parameter_bytes=metrics['parameter_bytes'], program_bytes=metrics['program_bytes'],
                instructions=metrics['instructions'], resident_bytes=metrics['model_upload_bytes'],
                bin_sha256=digest(metrics.get('bin_sha256')),
                execution_counts={key: metrics[key] for key in (*COUNTS, 'cpu_neural_ops')})


def evaluate(spec, reference, profile, artifacts):
    files = {key: artifacts.add(spec[key]) for key in ('wave', 'tokens', 'metrics')}
    audio = compare_audio(reference['files']['wave'], files['wave'])
    tokens = compare_tokens(reference['files']['tokens'], files['tokens'], audio)
    timing = execution(read_json(files['metrics']), reference, profile, spec)
    if 'bin' in spec:
        require(spec['bin'].get('sha256') == timing['bin_sha256'], 'Bin hash differs from runner')
        artifacts.add(spec['bin'])
    # Use actual float64 energies, not an average of percentages or a value
    # recovered from rounded published RMSE/L2 figures.
    x = sf.read(reference['files']['wave'], dtype='float64', always_2d=True)[0]
    y = sf.read(files['wave'], dtype='float64', always_2d=True)[0]
    return dict(**timing, duration_s=audio['duration_s'], samples=audio['samples'],
                sample_rate=audio['sample_rate'], error_energy=float(np.sum((y - x) ** 2)),
                reference_energy=float(np.sum(x ** 2)), relative_l2=audio['relative_l2'],
                relative_l2_status=audio['metric_status']['relative_l2'],
                token_matches=tokens['equal'], token_count=tokens['count'],
                source_sha256=tokens['source_sha256'], checkpoint_sha256=tokens['checkpoint_sha256'],
                bin_hash_verification='file rehashed' if 'bin' in spec else 'runner-recorded only',
                artifacts={key: artifacts.entries[path] for key, path in files.items()})


def pool(rows):
    result = {key: math.fsum(row[key] for row in rows) for key in
              ('duration_s', 'error_energy', 'reference_energy', 'processing_s', 'startup_inclusive_s', 'native_s')}
    for prefix in ('processing', 'startup_inclusive', 'native'):
        result[prefix + '_rtf'] = result[prefix + '_s'] / result['duration_s']
    result.update(cases=len(rows), samples=sum(row['samples'] for row in rows),
                  token_matches=sum(row['token_matches'] for row in rows),
                  token_count=sum(row['token_count'] for row in rows))
    result['token_match_fraction'] = result['token_matches'] / result['token_count']
    energy = result['reference_energy']
    result['relative_l2'] = math.sqrt(result['error_energy']) / math.sqrt(energy) if energy else None
    result['relative_l2_status'] = 'finite' if energy else 'undefined_zero_reference'
    return result


def change(before, after):
    # Both profiles have the same reference energy. Comparing SSE also gives
    # meaningful regression labels when that reference is entirely silent.
    def label(old, new):
        return 'regressed' if new > old else 'improved' if new < old else 'unchanged'
    return dict(waveform=label(before['error_energy'], after['error_energy']),
                tokens=label(-before['token_matches'], -after['token_matches']),
                processing=label(before['processing_s'], after['processing_s']),
                waveform_sse_delta=after['error_energy'] - before['error_energy'],
                token_matches_delta=after['token_matches'] - before['token_matches'],
                processing_rtf_delta=after['processing_rtf'] - before['processing_rtf'])


def build_report(manifest_path):
    path = Path(manifest_path).resolve()
    artifacts = Artifacts(path.parent)
    artifacts.add(dict(path=str(path), sha256=sha256_file(path)))
    generators = []
    for name in ('bigcodec_accuracy_report.py', 'bigcodec_compare.py', 'bigcodec_common.py'):
        source = Path(__file__).with_name(name).resolve()
        artifacts.add(dict(path=str(source), sha256=sha256_file(source)))
        generators.append(artifacts.entries[source])
    manifest = read_json(path)
    require(manifest.get('format') == 'bigcodec-accuracy-inputs-v1'
            and manifest.get('checkpoint_sha256') == CHECKPOINT_SHA256, 'Unknown manifest format/checkpoint')
    profiles = manifest.get('profiles')
    require(isinstance(profiles, list) and profiles, 'Expected a nonempty profile list')
    names = [profile.get('name') for profile in profiles]
    require(all(isinstance(name, str) and name for name in names) and len(names) == len(set(names))
            and manifest.get('baseline') in names, 'Invalid profile names/baseline')
    for profile in profiles:
        require(isinstance(profile.get('hardware_version'), str)
                and re.fullmatch('0x[0-9a-f]{8}', profile['hardware_version'])
                and isinstance(profile.get('precision'), dict), 'Each profile needs a build ID and precision subset')
    cases = manifest.get('cases')
    require(isinstance(cases, list) and cases, 'Expected a nonempty case list')
    case_names = [case.get('name') for case in cases]
    require(all(isinstance(name, str) and name for name in case_names)
            and len(case_names) == len(set(case_names)), 'Invalid or duplicate case names')
    for item in manifest.get('evidence', []):
        artifacts.add(item)
    rows, references = [], []
    for case in cases:
        require(isinstance(case.get('profiles'), dict) and set(case['profiles']) == set(names),
                f"Every profile must contain case {case['name']}")
        reference = reference_case(case, artifacts, manifest['checkpoint_sha256'])
        references.append(dict(case=case['name'], duration_s=reference['duration_s'],
                               processing_s=reference['cpu_processing_s'], threads=reference['cpu_threads'],
                               affinity=reference['cpu_affinity']))
        values = {p['name']: evaluate(case['profiles'][p['name']], reference, p, artifacts) for p in profiles}
        baseline = values[manifest['baseline']]
        rows.append(dict(case=case['name'], input=artifacts.entries[reference['source']],
                         reference={key: artifacts.entries[file] for key, file in reference['files'].items()},
                         profiles=values,
                         changes={name: change(baseline, value) for name, value in values.items()}))
    pooled = {name: pool([row['profiles'][name] for row in rows]) for name in names}
    regressions = {name: {metric: [row['case'] for row in rows if row['changes'][name][metric] == 'regressed']
                         for metric in ('waveform', 'tokens', 'processing')} for name in names}
    duration = math.fsum(row['duration_s'] for row in references)
    cpu_time = math.fsum(row['processing_s'] for row in references)
    artifacts.recheck()
    return dict(format='bigcodec-accuracy-report-v1', status='complete', baseline=manifest['baseline'],
                checkpoint_sha256=manifest['checkpoint_sha256'], profiles=profiles, cases=rows, pooled=pooled,
                pooled_changes={name: change(pooled[manifest['baseline']], value) for name, value in pooled.items()},
                regressions=regressions, cpu=dict(cases=references, duration_s=duration,
                    processing_s=cpu_time, processing_rtf=cpu_time / duration),
                definitions=dict(waveform='sqrt(sum waveform SSE / sum CPU reference energy); original sample indices, no gain/delay/polarity fitting',
                    tokens='sum identical token IDs / sum token counts',
                    rtf='sum measured seconds / sum source duration; RTF 1 is real time',
                    processing='Audio preprocessing, input upload, execution, output read and output serialization; excludes model load/upload',
                    startup_inclusive='Runner total_elapsed_s, including artifact load/validation and model upload; excludes compilation, Python startup and final metrics serialization',
                    scope='Whole-utterance codec reconstruction agreement, not denoising quality; no profile is selected automatically',
                    build='hardware_version is the runner-recorded 32-bit FPGA build ID, not a full bitstream SHA256'),
                report_source=generators,
                artifact_inventory=list(artifacts.entries.values()))


def markdown(report):
    def percent(value):
        return 'undefined (zero reference)' if value is None else f'{100 * value:.3f}%'
    lines = ['# BigCodec accuracy comparison', '',
             'FPGA reconstruction versus the frozen official FP32 CPU codec; no gain, delay or polarity fitting.', '',
             '| Profile | Pooled waveform L2 | Matching tokens | Processing RTF | Startup-inclusive RTF |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name, value in report['pooled'].items():
        lines.append(f"| {name} | {percent(value['relative_l2'])} | {value['token_matches']}/{value['token_count']} | {value['processing_rtf']:.5f} | {value['startup_inclusive_rtf']:.5f} |")
    lines += ['', f"Baseline: {report['baseline']}. CPU processing RTF: {report['cpu']['processing_rtf']:.5f}. RTF 1 means real time. Pooling uses summed energies and duration-weighted measured time.", '',
              '| Case | Profile | Waveform L2 | Tokens | Processing RTF | Startup-inclusive RTF | Waveform change |',
              '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for row in report['cases']:
        for name, value in row['profiles'].items():
            lines.append(f"| {row['case']} | {name} | {percent(value['relative_l2'])} | {value['token_matches']}/{value['token_count']} | {value['processing_rtf']:.5f} | {value['startup_inclusive_rtf']:.5f} | {row['changes'][name]['waveform']} |")
    lines += ['', 'Regressions against the baseline:', '']
    for name, metrics in report['regressions'].items():
        lines.append(f"- {name}: " + '; '.join(f"{key}: {', '.join(cases) or 'none'}" for key, cases in metrics.items()) + '.')
    lines += ['', 'Each FPGA run has one resident upload, one input upload, one START, one HALT, one output read, and zero CPU neural operations.', '',
              'Startup-inclusive RTF includes runner artifact validation/loading and model upload. It excludes compilation, Python startup and final metrics serialization. Bin hashes are runner-recorded unless the JSON explicitly says the file was rehashed. No profile is selected automatically.', '']
    return '\n'.join(lines)


def write_outputs(report, output=None, markdown_path=None):
    destinations = [Path(path) for path in (output, markdown_path) if path is not None]
    sources = [Path(entry['path']) for entry in report['artifact_inventory']] + [Path(__file__)]
    def aliases(left, right):
        return left.resolve() == right.resolve() or left.exists() and right.exists() and left.samefile(right)
    for index, destination in enumerate(destinations):
        require(not any(aliases(destination, source) for source in sources + destinations[index + 1:]),
                'Report destination must not overwrite an input or another report')
    serialized = json.dumps(report, indent=2, allow_nan=False) + '\n'
    for destination, text in ((output, serialized), (markdown_path, markdown(report))):
        if destination is None:
            continue
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile('w', dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(text)
                stream.close()
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
    return serialized


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--markdown', type=Path)
    args = parser.parse_args()
    try:
        report = build_report(args.manifest)
        print(write_outputs(report, args.output, args.markdown), end='')
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
