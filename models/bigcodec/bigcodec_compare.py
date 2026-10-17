#!/usr/bin/env python3
"""Compare CPU and FPGA reconstructed WAVs without gain or timing alignment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from bigcodec_common import load_tokens, sha256_file


def compare_audio(reference_path, actual_path):
    reference, reference_rate = sf.read(reference_path, dtype='float64', always_2d=True)
    actual, actual_rate = sf.read(actual_path, dtype='float64', always_2d=True)
    if reference_rate != actual_rate or reference.shape != actual.shape or not reference.size:
        raise ValueError('Reference and actual WAVs must have the same nonempty shape and sample rate')
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError('Cannot compare nonfinite audio')
    with np.errstate(over='ignore', invalid='ignore'):
        error = actual - reference
        reference_energy = float(np.sum(reference ** 2))
        actual_energy = float(np.sum(actual ** 2))
        error_energy = float(np.sum(error ** 2))
    if not np.isfinite([reference_energy, actual_energy, error_energy]).all():
        raise ValueError('Audio energy exceeds the finite comparison range')
    for values, energy in ((reference, reference_energy), (actual, actual_energy), (error, error_energy)):
        if energy == 0 and np.any(values != 0):
            raise ValueError('Audio energy is below the finite comparison range')
    reference_norm, actual_norm, error_norm = np.sqrt([reference_energy, actual_energy, error_energy])
    if reference_energy and error_energy:
        snr_status = 'finite'
    elif reference_energy:
        snr_status = 'positive_infinity_exact_match'
    elif error_energy:
        snr_status = 'negative_infinity_zero_reference'
    else:
        snr_status = 'undefined_zero_reference_and_error'
    return {
        'reference': str(Path(reference_path).resolve()),
        'actual': str(Path(actual_path).resolve()),
        'reference_sha256': sha256_file(reference_path),
        'actual_sha256': sha256_file(actual_path),
        'sample_rate': reference_rate, 'samples': reference.shape[0],
        'channels': reference.shape[1], 'duration_s': reference.shape[0] / reference_rate,
        'finite': True, 'alignment': 'original sample indices; no gain, delay or polarity fitting',
        'metric_status': {
            'relative_l2': 'finite' if reference_energy else 'undefined_zero_reference',
            'snr_db': snr_status,
            'cosine_similarity': 'finite' if reference_energy and actual_energy else 'undefined_zero_energy',
        },
        'relative_l2': float(error_norm / reference_norm) if reference_energy else None,
        'snr_db': float(10 * (np.log10(reference_energy) - np.log10(error_energy)))
            if reference_energy and error_energy else None,
        'max_abs_error': float(np.max(np.abs(error))),
        'rmse': float(error_norm / np.sqrt(reference.size)),
        'reference_rms': float(reference_norm / np.sqrt(reference.size)),
        'actual_rms': float(actual_norm / np.sqrt(actual.size)),
        'cosine_similarity': float(np.clip(np.sum(reference * actual) / (reference_norm * actual_norm), -1, 1))
            if reference_energy and actual_energy else None,
    }


def compare_tokens(reference_path, actual_path, audio_report):
    reference, reference_meta = load_tokens(reference_path)
    actual, actual_meta = load_tokens(actual_path)
    if reference.shape != actual.shape or reference_meta != actual_meta:
        raise ValueError('Token files must describe the same input audio and checkpoint')
    if (reference_meta['source_rate'] != audio_report['sample_rate']
            or reference_meta['source_samples'] != audio_report['samples']
            or audio_report['channels'] != 1):
        raise ValueError('Token metadata does not match the compared mono WAV dimensions')
    source_sha256 = reference_meta.get('source_sha256')
    if (not isinstance(source_sha256, str) or len(source_sha256) != 64
            or any(character not in '0123456789abcdef' for character in source_sha256)):
        raise ValueError('Token comparison requires the original input SHA256 provenance')
    equal = int(np.sum(reference == actual))
    return {
        'count': int(reference.size), 'equal': equal,
        'mismatches': int(reference.size) - equal,
        'match_fraction': float(equal / reference.size),
        'reference': str(Path(reference_path).resolve()),
        'actual': str(Path(actual_path).resolve()),
        'reference_sha256': sha256_file(reference_path),
        'actual_sha256': sha256_file(actual_path),
        'source_sha256': source_sha256,
        'checkpoint_sha256': reference_meta['checkpoint_sha256'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--actual', type=Path, required=True)
    parser.add_argument('--reference-tokens', type=Path)
    parser.add_argument('--actual-tokens', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if bool(args.reference_tokens) != bool(args.actual_tokens):
        parser.error('Provide both token files or neither')
    inputs = [p for p in (args.reference, args.actual, args.reference_tokens, args.actual_tokens) if p]
    if args.report is not None and any(args.report.resolve() == p.resolve()
            or args.report.exists() and p.exists() and args.report.samefile(p) for p in inputs):
        parser.error('--report must not overwrite an input')
    report = compare_audio(args.reference, args.actual)
    if args.reference_tokens is not None:
        report['tokens'] = compare_tokens(args.reference_tokens, args.actual_tokens, report)
    serialized = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized)
    print(serialized, end='')


if __name__ == '__main__':
    main()
