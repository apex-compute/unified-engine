#!/usr/bin/env python3
"""Separate STFT magnitude and phase error without aligning or changing audio.

This diagnostic uses the frozen eight-file comparison manifest. It does not
interpret phase error as inaudible, establish a perceptual threshold, or replace
the original time-domain L2 comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import signal
import soundfile as sf

HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent.parent / '20260915_encoder_accuracy/matrix_manifest.json'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(spec):
    path = (MANIFEST.parent / spec['path']).resolve()
    if digest(path) != spec['sha256']:
        raise ValueError(f'Changed frozen waveform: {path}')
    values, rate = sf.read(path, dtype='float64')
    if rate != 16000 or values.ndim != 1 or len(values) < 2048 or not np.isfinite(values).all():
        raise ValueError('Diagnostic requires finite mono 16 kHz recordings longer than 2048 samples')
    return values


def spectrum(values):
    return signal.stft(values, fs=16000, window='hann', nperseg=1024,
        noverlap=768, nfft=1024, boundary='zeros', padded=True,
        return_onesided=True, scaling='spectrum')[2]


def metrics(reference, actual):
    if reference.shape != actual.shape:
        raise ValueError('Waveforms must have identical sample counts')
    x, y = spectrum(reference), spectrum(actual)
    a, b = np.abs(x), np.abs(y)
    reference_energy = float(np.sum(a * a))
    complex_error = float(np.sum(np.abs(y - x) ** 2))
    magnitude_error = float(np.sum((b - a) ** 2))
    # |X-Y|² = (|X|-|Y|)² + 2(|X||Y| - Re(X conj(Y))).
    phase_error = float(2 * np.sum(a * b - np.real(x * np.conj(y))))
    if not math.isclose(complex_error, magnitude_error + phase_error, rel_tol=1e-11, abs_tol=1e-12):
        raise ValueError('Magnitude/phase energy identity failed')
    if reference_energy <= 0 or complex_error <= 0 or phase_error < 0:
        raise ValueError('This nonzero-error diagnostic requires positive finite reference/error energies')
    # A global lag is checked only as a diagnostic; no alignment is applied to
    # any waveform or score. Positive lag means actual lags the reference.
    correlation = signal.correlate(actual, reference, mode='full', method='fft')
    center = len(reference) - 1
    lags = np.arange(-160, 161)
    peak_lag = int(lags[np.argmax(correlation[center - 160:center + 161])])
    return dict(samples=len(reference), duration_s=len(reference) / 16000,
        stft_reference_energy=reference_energy, stft_complex_error_energy=complex_error,
        stft_magnitude_error_energy=magnitude_error, stft_phase_error_energy=phase_error,
        stft_complex_relative_l2=math.sqrt(complex_error / reference_energy),
        stft_magnitude_relative_l2=math.sqrt(magnitude_error / reference_energy),
        phase_share_of_stft_error_energy=phase_error / complex_error,
        magnitude_share_of_stft_error_energy=magnitude_error / complex_error,
        diagnostic_global_cross_correlation_lag_samples=peak_lag,
        lag_search_samples=[-160, 160], applied_lag_samples=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    protected = {MANIFEST.resolve(), Path(__file__).resolve()}
    for case in manifest['cases']:
        protected.add((MANIFEST.parent / case['reference']['wave']['path']).resolve())
        protected.update((MANIFEST.parent / spec['wave']['path']).resolve()
                         for spec in case['profiles'].values())
    if args.output and (args.output.resolve() in protected or args.output.exists()
                       and any(args.output.samefile(path) for path in protected)):
        parser.error('Output would overwrite a frozen input or generator')
    rows = []
    for case in manifest['cases']:
        reference = load(case['reference']['wave'])
        for name, profile in case['profiles'].items():
            rows.append(dict(case=case['name'], profile=name,
                reference=case['reference']['wave'], actual=profile['wave'],
                **metrics(reference, load(profile['wave']))))
    pooled = {}
    for name in (p['name'] for p in manifest['profiles']):
        selected = [row for row in rows if row['profile'] == name]
        energies = {key: math.fsum(row[key] for row in selected) for key in
            ('stft_reference_energy', 'stft_complex_error_energy',
             'stft_magnitude_error_energy', 'stft_phase_error_energy')}
        pooled[name] = dict(cases=len(selected), **energies,
            phase_share_of_stft_error_energy=energies['stft_phase_error_energy'] / energies['stft_complex_error_energy'],
            diagnostic_global_lags={row['case']: row['diagnostic_global_cross_correlation_lag_samples'] for row in selected})
    report = dict(format='bigcodec-stft-error-decomposition-v1',
        manifest='../../20260915_encoder_accuracy/matrix_manifest.json', manifest_sha256=digest(MANIFEST),
        generator_sha256=digest(Path(__file__)),
        wave_paths_relative_to='../../20260915_encoder_accuracy',
        scope='Eight long files; original sample positions, no resampling, gain/polarity fitting or applied delay.',
        definition='One-sided scipy STFT, periodic Hann, 1024-sample FFT/window, 256-sample hop, zero boundaries and padded tail; magnitude and phase error energies sum to complex STFT error.',
        limitation='A phase error contribution is not proof of inaudibility or a structural time shift. These percentages divide error energy, not waveform amplitude.',
        rows=rows, pooled=pooled)
    serialized = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    else:
        print(serialized, end='')


if __name__ == '__main__':
    main()
