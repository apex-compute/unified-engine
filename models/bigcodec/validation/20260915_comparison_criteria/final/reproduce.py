#!/usr/bin/env python3
"""Audit and recompute the complete paired-decoder comparison without inference.

The saved manifest pins every input, including source code and the historical
quality results. No checkpoint, deployment binary, or FPGA is opened. Normal
use writes only explicitly requested report destinations. --freeze creates the
three report files once, after all nine native records have passed the audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[2]
ROOT = MODEL.parents[1]
sys.path.insert(0, str(MODEL))
from bigcodec_accuracy_report import Artifacts, evaluate, execution, pool, read_json, reference_case, require
from bigcodec_audio_quality import compare_quality
from bigcodec_common import CHECKPOINT_SHA256, load_tokens, sha256_file
from bigcodec_compare import compare_audio, compare_tokens

ENC = MODEL / 'validation/20260915_encoder_accuracy'
PARENT = HERE.parent
CASES = ('bus', 'cafe', 'office', 'psquare', 'bus_low_snr', 'cafe_low_snr',
         'office_low_snr', 'psquare_low_snr')
QUALITY_KEYS = ('pesq_wb', 'stoi')
PRECISION = dict(convolutions='BF16', activations='BF16', lstm_input_weights='BF16',
    lstm_cell='compensated', lstm_tanh='compensated', lstm_math_scope='decoder',
    lstm_fused_gates=True, center_quantizer_scores=True, compensated_codebook=True,
    filter_accumulation='serial', filter_math_scope='both', filter_stage='both',
    lstm_recurrent_weights={'encoder': 'BF16', 'decoder': 'BF16'},
    lstm_sigmoid={'encoder.block.6': 'native-bf16',
                  'decoder.model.1': 'compensated-pade-high-low'})


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def stable(value):
    """Historical display paths do not make a metric dependent on its checkout."""
    if isinstance(value, dict):
        return {key: stable(item) for key, item in value.items()
                if key not in ('reference', 'actual')}
    if isinstance(value, list):
        return [stable(item) for item in value]
    return value


def same(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), label)
        for key in expected:
            same(actual[key], expected[key], f'{label}.{key}')
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label)
        for index, (a, b) in enumerate(zip(actual, expected)):
            same(a, b, f'{label}[{index}]')
    elif type(expected) is float:
        require(type(actual) in (float, int) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), label)
    else:
        require(actual == expected, label)


def quality_content(value):
    # Supporting-library version changes are acceptable only when the metrics,
    # analysis settings, and implementation hashes still reproduce. Both the
    # recorded and current versions remain present in the report.
    result = stable(value)
    result.pop('dependencies')
    return result


class Audit(Artifacts):
    def add_path(self, path, expected=None):
        path = Path(path).resolve()
        require(path.is_relative_to(ROOT) and 'bigcodec_bin' not in path.parts,
                f'Nonportable dependency: {path}')
        return self.add(dict(path=relative(path), sha256=expected or sha256_file(path)))

    def spec(self, path, expected=None):
        path = self.add_path(path, expected)
        return dict(path=relative(path), sha256=self.entries[path]['sha256'])

    def inventory(self):
        return [dict(path=relative(path), sha256=entry['sha256'], bytes=entry['bytes'])
                for path, entry in sorted(self.entries.items())]


def imported_spec(audit, spec, base):
    return audit.spec(base / spec['path'], spec['sha256'])


def energies(reference, actual):
    x, xr = sf.read(reference, dtype='float64')
    y, yr = sf.read(actual, dtype='float64')
    require(xr == yr and x.shape == y.shape and x.ndim == 1
            and np.all(np.isfinite(x)) and np.all(np.isfinite(y)), 'Invalid waveform geometry')
    return dict(error_energy=float(np.sum((y - x) ** 2)),
                reference_energy=float(np.sum(x ** 2)))


def error_components(reference, same_token_cpu, actual):
    x = sf.read(reference, dtype='float64')[0]
    s = sf.read(same_token_cpu, dtype='float64')[0]
    y = sf.read(actual, dtype='float64')[0]
    token, decoder = s - x, y - s
    te, de = float(np.sum(token ** 2)), float(np.sum(decoder ** 2))
    cross = float(2 * np.sum(token * decoder))
    full = float(np.sum((y - x) ** 2))
    same(full, te + de + cross, 'Error decomposition identity')
    return dict(token_error_energy=te, decoder_error_energy=de,
        two_error_cross_term=cross, full_error_energy=full,
        token_decoder_error_cosine=cross / (2 * math.sqrt(te * de)),
        identity='full_error_energy = token_error_energy + decoder_error_energy + two_error_cross_term')


def conditional(audit, case, reference, old_tokens):
    short = case == 'p232_007'
    if short:
        base = MODEL / 'validation/20260915_accuracy/short_clip'
        path = base / 'audio/split_decoder_same_tokens_cpu.wav'
        receipt = read_json(audit.add_path(base / 'records/split_decoder_same_tokens_cpu.metrics.json'))
        require(receipt['mode'] == 'decode' and receipt['backend'] == 'pytorch-cpu-fp32'
                and receipt['output_finite'] is True and receipt['audio'] == reference['metadata']
                and receipt['tokens_sha256'] == old_tokens['sha256'], 'Short conditional provenance')
        expected = receipt['output_sha256']
    else:
        base = PARENT / 'decomposition'
        path = base / f'audio/{case}_same_tokens_cpu.wav'
        receipt = read_json(audit.add_path(base / f'records/{case}_same_tokens_cpu.json'))
        require(receipt['case'] == case and receipt['variant'] == 'selected'
                and receipt['recurrent_precision'] == 'bf16'
                and receipt['files']['actual_tokens']['sha256'] == old_tokens['sha256']
                and receipt['files']['reference_wav']['sha256']
                    == sha256_file(reference['files']['wave']), 'Conditional CPU provenance')
        expected = receipt['files']['same_token_cpu_wav']['sha256']
        for name, value in receipt['source_files_sha256'].items():
            audit.add_path(ROOT / name, value)
    require(receipt['checkpoint_sha256'] == CHECKPOINT_SHA256, 'Conditional checkpoint mismatch')
    return audit.add_path(path, expected)


def quality_baselines(audit, label, case, source, cpu, old_wave):
    directory = PARENT / 'perceptual'
    manifest = read_json(audit.add_path(directory / 'manifest.json'))
    spec = manifest['results'][label]
    dataset = read_json(audit.add_path(directory / spec['path'], spec['sha256']))
    expected = {'input': sha256_file(source), 'official_cpu': sha256_file(cpu),
                'serial_baseline': sha256_file(old_wave)}
    for role, value in expected.items():
        items = [item for item in dataset['source_artifacts']
                 if item['case'] == case and item['role'] == role]
        require(len(items) == 1 and items[0]['sha256'] == value, 'Frozen quality source mismatch')
        audit.add_path(ROOT / items[0]['path'], value)
    selected = {}
    for ref, actual in (('input', 'official_cpu'), ('input', 'serial_baseline'),
                        ('official_cpu', 'serial_baseline')):
        rows = [row for row in dataset['rows'] if row['case'] == case
                and row['reference'] == ref and row['actual'] == actual]
        require(len(rows) == 1, 'Missing historical quality row')
        result = rows[0]['comparison']
        require(result['waveform']['reference_sha256'] == expected[ref]
                and result['waveform']['actual_sha256'] == expected[actual], 'Quality row hashes')
        selected[(ref, actual)] = result
    return selected


def evaluate_case(audit, case):
    short = case == 'p232_007'
    source_manifest = ENC / ('restored_short_manifest.json' if short else 'matrix_manifest.json')
    historical = read_json(audit.add_path(source_manifest))
    original = next(item for item in historical['cases'] if item['name'] == case)
    old_name = 'baseline_restored' if short else 'serial_baseline'
    old_profile = next(item for item in historical['profiles'] if item['name'] == old_name)
    source_spec = imported_spec(audit, original['input'], ENC)
    cpu_specs = {key: imported_spec(audit, value, ENC) for key, value in original['reference'].items()}
    old_specs = {key: imported_spec(audit, original['profiles'][old_name][key], ENC)
                 for key in ('wave', 'tokens', 'metrics')}
    reference = reference_case(dict(input=source_spec, reference=cpu_specs), audit, CHECKPOINT_SHA256)
    wave = HERE / f'fpga/{case}.wav'
    specs = {key: audit.spec(wave.with_suffix(suffix)) for key, suffix in
             (('wave', '.wav'), ('tokens', '.tokens.npz'), ('metrics', '.metrics.json'))}
    record = read_json(audit.add_path(wave.with_suffix('.comparison.json')))
    require(record['case'] == case and record['encoder_tokens_unchanged'] is True,
            'Incorrect comparison record')
    require(record['short_matches_unrolled_native'] is (True if short else None),
            'Short unrolled parity receipt mismatch')
    same_cpu = conditional(audit, case, reference, old_specs['tokens'])
    # The historical short input has a byte-identical, tracked portable copy.
    aliases = {'test_samples/p232_007.wav': source_spec['path']} if short else {}
    expected_frozen = {spec['path']: spec['sha256'] for spec in
                       (source_spec, cpu_specs['wave'], cpu_specs['tokens'],
                        audit.spec(same_cpu), old_specs['wave'], old_specs['tokens'])}
    same({aliases.get(name, name): value for name, value in record['frozen_inputs'].items()},
         expected_frozen, 'Frozen native inputs')
    require(record['metrics_path'] == specs['metrics']['path']
            and record['metrics_sha256'] == specs['metrics']['sha256'], 'Native metrics binding')
    a, am = load_tokens(ROOT / old_specs['tokens']['path'])
    b, bm = load_tokens(ROOT / specs['tokens']['path'])
    require(am == bm and np.array_equal(a, b), 'Encoder tokens or metadata changed')
    old = evaluate(old_specs, reference, old_profile, audit)
    new = evaluate(specs, reference, dict(hardware_version='0x90f1f464', precision=PRECISION), audit)
    require(record['bin']['sha256'] == new['bin_sha256']
            and type(record['bin']['bytes']) is int and record['bin']['bytes'] > new['resident_bytes'],
            'Binary receipt mismatch')
    current_metrics = read_json(ROOT / specs['metrics']['path'])
    same(current_metrics['precision'], PRECISION, 'Full precision metadata')
    raw = compare_audio(reference['files']['wave'], wave)
    raw['tokens'] = compare_tokens(reference['files']['tokens'], ROOT / specs['tokens']['path'], raw)
    same(stable(raw), stable(record['comparison']), 'Recomputed raw comparison')
    for name, value, output in (('old', old, ROOT / old_specs['wave']['path']), ('new', new, wave)):
        value['conditional'] = dict(**energies(same_cpu, output),
                                    comparison=stable(compare_audio(same_cpu, output)))
        value['artifacts'] = {key: dict(item, path=relative(item['path']))
                              for key, item in value['artifacts'].items()}
        if name == 'new':
            same(value['conditional']['comparison'], stable(record['conditional']), 'Conditional comparison')
        if case == 'psquare':
            value['error_components'] = error_components(reference['files']['wave'], same_cpu, output)
    baselines = quality_baselines(audit, 'short_clip' if short else 'eight_noisy', case,
                                 reference['source'], reference['files']['wave'], ROOT / old_specs['wave']['path'])
    quality = {}
    for name, ref in (('cpu', reference['files']['wave']), ('input', reference['source'])):
        value = compare_quality(ref, wave)
        require(set(value['quality']['metric_status'].values()) == {'finite'}, 'Undefined quality metric')
        same(quality_content(value), quality_content(record['quality_vs_' + name]),
             f'Recomputed quality vs {name}')
        for dependency, version in (('pesq', '0.0.4'), ('pystoi', '0.4.1')):
            require(value['dependencies'][dependency] == version
                    == record['quality_vs_' + name]['dependencies'][dependency],
                    f'Pinned quality dependency: {dependency}')
        quality[name] = stable(value)
    for key in baselines.values():
        for dependency in ('pesq', 'pystoi'):
            same(key['dependencies'][dependency], quality['cpu']['dependencies'][dependency],
                 f'Historical quality dependency: {dependency}')
        same(key['source_sha256'], quality['cpu']['source_sha256'], 'Quality source versions')
    cpu_quality = baselines[('input', 'official_cpu')]['quality']
    old_quality = baselines[('input', 'serial_baseline')]['quality']
    return dict(case=case, duration_s=reference['duration_s'], old=old, new=new,
        encoder_tokens_unchanged=True, encoder_token_count=int(a.size),
        official_cpu_token_matches=raw['tokens']['equal'], official_cpu_token_count=raw['tokens']['count'],
        same_token_cpu=dict(path=relative(same_cpu), sha256=sha256_file(same_cpu)),
        official_cpu_processing_s=reference['cpu_processing_s'],
        new_bin_receipt=record['bin'], short_matches_unrolled_native_recorded=record['short_matches_unrolled_native'],
        input_path_aliases=aliases, quality=quality,
        native_quality_record_dependencies=record['quality_vs_cpu']['dependencies'],
        historical_quality=dict(cpu_vs_input=stable(baselines[('input', 'official_cpu')]),
            old_vs_cpu=stable(baselines[('official_cpu', 'serial_baseline')]),
            old_vs_input=stable(baselines[('input', 'serial_baseline')])),
        codec_quality_gap_vs_cpu={key: dict(old=old_quality[key] - cpu_quality[key],
            new=quality['input']['quality'][key] - cpu_quality[key]) for key in QUALITY_KEYS})


def image_control(audit, bus):
    directory = PARENT / 'loop'
    manifest = read_json(audit.add_path(directory / 'manifest.json'))
    name = 'bus_baseline_image_parity.json'
    path = audit.add_path(directory / name, manifest['files'][name]['sha256'])
    record = read_json(path)
    require(record['status'] == 'PASS' and record['waveform_sample_bits_identical'] is True
            and record['tokens_identical'] is True
            and record['previous_hardware'] == '0x40519e0a'
            and record['current_hardware'] == '0x90f1f464'
            and record['old_wav_sha256'] == bus['old']['artifacts']['wave']['sha256']
            and record['samples'] == bus['old']['samples'], 'Current-image baseline control receipt')
    historical = read_json(ENC / 'matrix_manifest.json')
    original = next(item for item in historical['cases'] if item['name'] == 'bus')
    cpu_specs = {key: imported_spec(audit, value, ENC) for key, value in original['reference'].items()}
    reference = reference_case(dict(input=imported_spec(audit, original['input'], ENC),
                                    reference=cpu_specs), audit, CHECKPOINT_SHA256)
    profile = next(item for item in historical['profiles'] if item['name'] == 'serial_baseline')
    timing = execution(record['metrics'], reference, dict(profile, hardware_version='0x90f1f464'),
        dict(wave={'sha256': record['current_wav_sha256']},
             tokens={'sha256': bus['old']['artifacts']['tokens']['sha256']}))
    require(timing['bin_sha256'] == bus['old']['bin_sha256'], 'Baseline control binary changed')
    return dict(path=relative(path), sha256=sha256_file(path), status='recorded PASS',
        waveform_sample_bits_identical_recorded=True, tokens_identical_recorded=True,
        current_image_timing=timing,
        verification='Tracked native parity receipt and its execution contract audited; current-image control WAV is not a reproduction input.')


def aggregate(rows):
    result = {name: pool([row[name] for row in rows]) for name in ('old', 'new')}
    for name, value in result.items():
        e = math.fsum(row[name]['conditional']['error_energy'] for row in rows)
        r = math.fsum(row[name]['conditional']['reference_energy'] for row in rows)
        value['conditional'] = dict(error_energy=e, reference_energy=r, relative_l2=math.sqrt(e / r))
        value['mean_quality_vs_cpu'] = {key: math.fsum(
            (row['quality']['cpu'] if name == 'new' else row['historical_quality']['old_vs_cpu'])
            ['quality'][key] for row in rows) / len(rows) for key in QUALITY_KEYS}
        value['mean_quality_vs_input'] = {key: math.fsum(
            (row['quality']['input'] if name == 'new' else row['historical_quality']['old_vs_input'])
            ['quality'][key] for row in rows) / len(rows) for key in QUALITY_KEYS}
        value['mean_codec_quality_gap_vs_cpu'] = {key: math.fsum(
            row['codec_quality_gap_vs_cpu'][key][name] for row in rows) / len(rows) for key in QUALITY_KEYS}
    result['official_cpu_mean_quality_vs_input'] = {key: math.fsum(
        row['historical_quality']['cpu_vs_input']['quality'][key] for row in rows) / len(rows)
        for key in QUALITY_KEYS}
    result['official_cpu_processing_rtf'] = math.fsum(row['official_cpu_processing_s'] for row in rows) / result['new']['duration_s']
    result['new_raw_at_least_10_percent'] = [row['case'] for row in rows if row['new']['relative_l2'] >= .1]
    result['new_conditional_at_least_10_percent'] = [row['case'] for row in rows
        if row['new']['conditional']['comparison']['relative_l2'] >= .1]
    result['regressions'] = {metric: [row['case'] for row in rows if fn(row['new']) > fn(row['old'])]
        for metric, fn in (('raw_l2', lambda p: p['error_energy']),
                           ('conditional_l2', lambda p: p['conditional']['error_energy']),
                           ('processing_rtf', lambda p: p['processing_rtf']))}
    return result


def markdown(report):
    values = report['eight_noisy']['aggregate']
    old, new = values['old'], values['new']
    lines = ['# Paired decoder validation', '',
        f"Eight noisy files: pooled full-codec waveform error **{old['relative_l2']:.2%} → {new['relative_l2']:.2%}**; "
        f"decoder error with identical tokens **{old['conditional']['relative_l2']:.2%} → {new['conditional']['relative_l2']:.2%}**. "
        f"Processing RTF **{old['processing_rtf']:.4f} → {new['processing_rtf']:.4f}**. RTF is processing time/audio duration; below 1 is realtime.", '',
        'The full-codec comparison uses the official CPU encoder and decoder. The conditional comparison decodes the unchanged FPGA tokens on the CPU, isolating decoder arithmetic. Conditional error is not a replacement for the full-codec target.', '',
        'Case links open the new FPGA output. Audio references link the noisy input and official CPU reconstruction.', '',
        '| Case (FPGA output) | Audio references | Full-codec L2 old → new | Same-token decoder L2 old → new | RTF old → new | CPU token matches |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in report['eight_noisy']['rows']:
        a, b = row['old'], row['new']
        case = row['case']
        source = os.path.relpath(ROOT / f'models/dpdfnet/validation/20260914_noisy20s/noisy/{case}_noisy.wav', HERE)
        cpu = os.path.relpath(MODEL / f'validation/20260914_noisy20s/cpu/{case}.wav', HERE)
        lines.append(f"| [{case}](fpga/{case}.wav) | [Input]({source}) · [CPU]({cpu}) | "
            f"{a['relative_l2']:.3%} → {b['relative_l2']:.3%} | "
            f"{a['conditional']['comparison']['relative_l2']:.3%} → {b['conditional']['comparison']['relative_l2']:.3%} | "
            f"{a['processing_rtf']:.4f} → {b['processing_rtf']:.4f} | {row['official_cpu_token_matches']}/{row['official_cpu_token_count']} |")
    require(len(values['new_raw_at_least_10_percent']) == 8
            and not values['new_conditional_at_least_10_percent']
            and not report['short_clip']['aggregate']['new_conditional_at_least_10_percent'],
            'Published threshold summary no longer matches results')
    lines += ['', 'All eight full errors remain above 10%; all nine conditional errors are below 10% (including the separate short clip).',
        f"Raw-error regressions: {', '.join(values['regressions']['raw_l2']) or 'none'}. "
        f"Conditional-error regressions: {', '.join(values['regressions']['conditional_l2']) or 'none'}.", '',
        'On psquare, lower decoder error removes some accidental cancellation between encoder-token and decoder errors: '
        'the negative cross term becomes smaller in magnitude, so full-codec error rises. '
        'The exact energy identity is recomputed in the JSON.', '',
        '| Eight-file arithmetic mean | Old FPGA | New FPGA |', '| --- | ---: | ---: |']
    for key, label in (('pesq_wb', 'PESQ-WB vs CPU'), ('stoi', 'STOI vs CPU')):
        lines.append(f"| {label} | {old['mean_quality_vs_cpu'][key]:.6f} | {new['mean_quality_vs_cpu'][key]:.6f} |")
    for key, label in (('pesq_wb', 'PESQ-WB codec-quality gap'), ('stoi', 'STOI codec-quality gap')):
        lines.append(f"| {label} | {old['mean_codec_quality_gap_vs_cpu'][key]:+.6f} | {new['mean_codec_quality_gap_vs_cpu'][key]:+.6f} |")
    cpu = values['official_cpu_mean_quality_vs_input']
    lines += ['', f"Codec-quality gap is FPGA minus CPU score against the same original noisy input. "
        f"CPU baseline: PESQ-WB {cpu['pesq_wb']:.6f}, STOI {cpu['stoi']:.6f}. "
        'These noisy-reference scores describe codec fidelity, not denoising or word accuracy. PESQ retains its standard internal alignment/level normalization; no external delay, gain, or polarity fitting is applied. '
        'Raw L2 uses original sample indices and rate. Speech/spectral metrics use shared 16 kHz resampling. '
        '[Metric definitions and primary sources](../perceptual/sources.md).', '']
    row = report['short_clip']['row']; a, b = row['old'], row['new']
    lines += [f"Independent [short-clip FPGA output](fpga/p232_007.wav) (3.96 s, 48 kHz, excluded from the eight-file pool): full L2 "
        f"{a['relative_l2']:.3%} → {b['relative_l2']:.3%}; same-token decoder L2 "
        f"{a['conditional']['comparison']['relative_l2']:.3%} → {b['conditional']['comparison']['relative_l2']:.3%}; "
        f"RTF {a['processing_rtf']:.4f} → {b['processing_rtf']:.4f}. "
        f"{row['official_cpu_token_matches']}/{row['official_cpu_token_count']} tokens match the CPU; all {row['encoder_token_count']} are unchanged from the old FPGA.", '',
        '| Compiled input | Old program bytes | New program bytes | New parameter bytes | New resident bytes |',
        '| --- | ---: | ---: | ---: | ---: |']
    for row in [report['short_clip']['row'], *report['eight_noisy']['rows']]:
        a, b = row['old'], row['new']
        lines.append(f"| {row['case']} | {a['program_bytes']:,} | {b['program_bytes']:,} | {b['parameter_bytes']:,} | {b['resident_bytes']:,} |")
    lines += ['', f"Eight-file native RTF: {old['native_rtf']:.4f} → {new['native_rtf']:.4f}; "
        f"startup-inclusive RTF: {old['startup_inclusive_rtf']:.4f} → {new['startup_inclusive_rtf']:.4f}. "
        'RTFs are total measured time divided by total audio duration. Compiled programs cover whole files, not 10 ms streaming chunks.', '',
        'All nine new runs use build `0x90f1f464`, AXI256, BF16 convolution/recurrent weights, compensated decoder cell/tanh and paired sigmoid. '
        'The encoder retains native BF16 sigmoid. Each saved receipt has one model upload, one input upload, one START, one HALT and one output read, zero CPU neural operations, finite output and zero-valued padding. '
        'Encoder token IDs and metadata are unchanged. Long-file historical baseline receipts use build `0x40519e0a`; the short baseline uses the current build. '
        'A [current-image bus control](../loop/bus_baseline_image_parity.json) records identical decoded sample bits and tokens for the old binary on both builds; the reproducer audits that receipt.', '',
        'The reproducer rehashes tracked WAVs, tokens, CPU provenance, native records and frozen source files; recomputes raw/conditional errors, new perceptual scores and timing aggregates; and verifies the frozen historical CPU-quality baseline. '
        'Bin SHA256 and exact program/parameter sizes are runner-recorded in [results.json](results.json). Reproduction does not require cached binaries, model weights, or hardware and does not rerun inference.', '',
        '```bash', 'pip install -r models/bigcodec/requirements-quality.txt',
        'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\',
        '  python models/bigcodec/validation/20260915_comparison_criteria/final/reproduce.py \\',
        '  --output /tmp/bigcodec-final-results.json --markdown /tmp/bigcodec-final-report.md', '```', '']
    return '\n'.join(lines)


def build():
    missing = [f'{case}{suffix}' for case in ('p232_007', *CASES)
               for suffix in ('.wav', '.tokens.npz', '.metrics.json', '.comparison.json')
               if not (HERE / 'fpga' / f'{case}{suffix}').is_file()]
    require(not missing, f'Native suite incomplete: {", ".join(missing)}')
    audit = Audit(ROOT)
    for path in (Path(__file__), MODEL / 'bigcodec_accuracy_report.py'):
        audit.add_path(path)
    sources = read_json(audit.add_path(HERE / 'production_sources.json'))
    for path, value in sources.items():
        audit.add_path(ROOT / path, value)
    rows = []
    for case in ('p232_007', *CASES):
        print(f'Auditing {case}', file=sys.stderr, flush=True)
        rows.append(evaluate_case(audit, case))
    control = image_control(audit, rows[1])
    audit.recheck()
    manifest = dict(format='bigcodec-paired-final-manifest-v1', path_base='repository root',
        cases=['p232_007', *CASES], checkpoint_sha256=CHECKPOINT_SHA256, files=audit.inventory(),
        bin_policy='Runner-recorded SHA256/bytes only; deployment binaries are not reproduction inputs.')
    result = dict(format='bigcodec-paired-final-results-v1', status='complete',
        eight_noisy=dict(rows=rows[1:], aggregate=aggregate(rows[1:])),
        short_clip=dict(row=rows[0], aggregate=aggregate(rows[:1])),
        image_control=control,
        audit=dict(input_files=len(manifest['files']), all_hashes_verified=True,
            new_quality_pairs_recomputed=18, historical_quality='Frozen tracked paired results and their WAV hashes verified',
            dependencies=rows[0]['quality']['cpu']['dependencies'],
            source_code_frozen=True, original_alignment=True, no_inference=True),
        aggregation='Raw and conditional L2 pool error/reference energies independently. RTF pools seconds/audio duration. Quality scores and gaps are per-file arithmetic means. Short clip is separate.')
    return manifest, result


def encoded(value):
    return json.dumps(value, indent=2, allow_nan=False) + '\n'


def alias(left, right):
    return (left.resolve() == right.resolve()
            or (left.exists() and right.exists() and left.samefile(right)))


def write_reports(destinations, inputs, *, freeze=False):
    paths = [path for path, _ in destinations if path is not None]
    for index, path in enumerate(paths):
        require(not any(alias(path, other) for other in paths[:index]),
                'Report destinations alias each other')
        require(freeze or not any(alias(path, source) for source in inputs),
                'Refusing to overwrite an input')
    for path, contents in destinations:
        if path is None:
            continue
        # Replacing a completed temporary file prevents truncating an existing
        # hardlink, including a link introduced after the validation above.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                    dir=path.parent, prefix='.' + path.name + '.', delete=False) as output:
                temporary = Path(output.name)
                output.write(contents)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--markdown', type=Path)
    parser.add_argument('--cpu-core', type=int)
    parser.add_argument('--freeze', action='store_true', help='Create manifest/results/README once after complete validation')
    args = parser.parse_args()
    if args.cpu_core is not None:
        os.sched_setaffinity(0, {args.cpu_core})
    manifest_path = HERE / 'manifest.json'
    expected = None if args.freeze else read_json(manifest_path)
    if expected is not None:
        for entry in expected['files']:
            path = (ROOT / entry['path']).resolve()
            require(path.is_relative_to(ROOT) and sha256_file(path) == entry['sha256']
                    and path.stat().st_size == entry['bytes'], f'Frozen input changed: {entry["path"]}')
    manifest, result = build()
    if expected is not None:
        same(manifest, expected, 'Manifest inventory')
    result['manifest_sha256'] = hashlib.sha256(encoded(manifest).encode()).hexdigest()
    destinations = [(args.output, encoded(result)), (args.markdown, markdown(result))]
    if args.freeze:
        require(args.output is None and args.markdown is None, '--freeze cannot redirect reports')
        destinations = [(manifest_path, encoded(manifest)), (HERE / 'results.json', encoded(result)),
                        (HERE / 'README.md', markdown(result))]
        require(all(not path.exists() for path, _ in destinations), 'Frozen reports already exist')
    inputs = {(ROOT / entry['path']).resolve() for entry in manifest['files']} | {manifest_path.resolve()}
    write_reports(destinations, inputs, freeze=args.freeze)
    if not args.output and not args.freeze:
        print(encoded(result), end='')
    else:
        print(json.dumps(dict(status='complete', input_files=result['audit']['input_files'],
                              eight_noisy=result['eight_noisy']['aggregate'])), file=sys.stderr)


if __name__ == '__main__':
    main()
