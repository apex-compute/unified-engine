#!/usr/bin/env python3
"""Compare original WAV samples and complementary speech/spectral metrics.

Raw waveform comparison is unchanged and uses the original sample rate.
PESQ, STOI and spectra use a shared 16 kHz resampling operation, without an
external gain, delay or polarity fit. PESQ's internal normalization/alignment
and STOI's standard envelope normalization/silence removal are retained.

Install optional dependencies with: pip install -r requirements-quality.txt
Scores using noisy or codec-reconstructed references describe agreement, not
validated clean-reference intelligibility or denoising quality. No score is
equivalent to a percentage of words understood or a waveform-error threshold.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import tempfile
import warnings

import numpy as np
import soundfile as sf

from bigcodec_common import sha256_file
from bigcodec_compare import compare_audio

RATE = 16000
WINDOWS = (512, 1024, 2048)


def _dependencies():
    try:
        from scipy import signal
        from pesq import pesq, BufferTooShortError, NoUtterancesError
        from pystoi import stoi
    except ImportError as error:
        raise ValueError('Install optional metrics: pip install -r '
                         'models/bigcodec/requirements-quality.txt') from error
    return signal, pesq, stoi, BufferTooShortError, NoUtterancesError


def spectral_metrics(reference, actual, signal):
    """Fixed geometry, magnitude-only diagnostics; phase is deliberately absent."""
    rows = []
    for window in WINDOWS:
        # Explicit extension handles even a one-sample file without silently
        # changing scipy's requested window or overlap.
        size = max(window, len(reference))
        x, y = (np.pad(v, (0, size - len(v))) for v in (reference, actual))
        def transform(v):
            return np.abs(signal.stft(v, fs=RATE, window='hann', nperseg=window,
                noverlap=3 * window // 4, nfft=window, boundary='zeros', padded=True,
                return_onesided=True, scaling='spectrum')[2])
        a, b = transform(x), transform(y)
        error, energy = float(np.sum((a - b) ** 2)), float(np.sum(a ** 2))
        db = 20 * np.log10(np.maximum(a, 1e-5)) - 20 * np.log10(np.maximum(b, 1e-5))
        values = (error, energy, float(np.mean(np.abs(db))),
                  float(np.mean(np.sqrt(np.mean(db ** 2, axis=0)))))
        if not np.isfinite(values).all():
            raise ValueError('Spectral energy exceeds the finite comparison range')
        rows.append(dict(fft_size=window, frames=a.shape[1], spectral_error_energy=error,
            spectral_reference_energy=energy,
            spectral_convergence=math.sqrt(error) / math.sqrt(energy) if energy else None,
            spectral_convergence_status='finite' if energy else 'undefined_zero_reference',
            log_magnitude_mae_db=values[2], log_spectral_distance_db=values[3]))
    return dict(spectral=rows,
        mr_spectral_convergence=float(np.mean([r['spectral_convergence'] for r in rows]))
            if all(r['spectral_convergence'] is not None for r in rows) else None,
        mr_log_magnitude_mae_db=float(np.mean([r['log_magnitude_mae_db'] for r in rows])),
        mr_log_spectral_distance_db=float(np.mean([r['log_spectral_distance_db'] for r in rows])))


def compare_quality(reference_path, actual_path):
    waveform = compare_audio(reference_path, actual_path)
    if waveform['channels'] != 1:
        raise ValueError('Speech quality comparison requires mono WAVs; no implicit downmix is applied')
    signal, pesq, stoi, short_error, utterance_error = _dependencies()
    reference, rate = sf.read(reference_path, dtype='float64')
    actual = sf.read(actual_path, dtype='float64')[0]
    divisor = math.gcd(rate, RATE)
    up, down = RATE // divisor, rate // divisor
    if rate != RATE:
        reference, actual = (signal.resample_poly(v, up, down,
            window=('kaiser', 5.0), padtype='constant') for v in (reference, actual))
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError('Resampled audio is nonfinite')
    result = spectral_metrics(reference, actual, signal)
    statuses = {}
    if not np.any(reference):
        degenerate = 'undefined_zero_reference'
    elif not np.any(actual):
        degenerate = 'undefined_zero_actual'
    else:
        degenerate = None
    for name, mode in (('pesq_wb', 'wb'), ('pesq_nb', 'nb')):
        value, status = None, degenerate
        if status is None:
            try:
                value = float(pesq(RATE, reference, actual, mode=mode))
                status = 'finite'
            except short_error:
                status = 'insufficient_duration'
            except utterance_error:
                status = 'no_utterances_detected'
        result[name], statuses[name] = value, status
    for name, extended in (('stoi', False), ('estoi', True)):
        value, status = None, degenerate
        if status is None and len(reference) < 512:
            status = 'insufficient_active_frames'
        if status is None:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                value = float(stoi(reference, actual, RATE, extended=extended))
            if caught:
                messages = [str(item.message) for item in caught]
                if not all('Not enough STFT frames' in message for message in messages):
                    raise ValueError('STOI warning: ' + '; '.join(messages))
                value, status = None, 'insufficient_active_frames'
            else:
                status = 'finite'
        result[name], statuses[name] = value, status
    if any(v is not None and not math.isfinite(v) for key, v in result.items()
           if key in statuses):
        raise ValueError('A speech metric returned a nonfinite score')
    for path, expected in ((reference_path, waveform['reference_sha256']),
                           (actual_path, waveform['actual_sha256'])):
        if sha256_file(path) != expected:
            raise ValueError('Input WAV changed during comparison')
    versions = {name: importlib.metadata.version(name)
                for name in ('numpy', 'soundfile', 'scipy', 'pesq', 'pystoi')}
    return dict(format='bigcodec-audio-quality-v1', waveform=waveform,
        quality=dict(sample_rate=RATE, samples=len(reference), metric_status=statuses, **result),
        analysis=dict(original_sample_rate=rate,
            resampling=dict(applied=rate != RATE, up=up, down=down,
                method='scipy.signal.resample_poly', window=['kaiser', 5.0], padtype='constant',
                scope='Both signals identically, for speech and spectral metrics only'),
            external_alignment='none; no delay, gain or polarity fitting',
            pesq='MOS-LQO; standard internal level and time processing retained',
            stoi='Standard silence removal, resampling and envelope normalization retained; not percent words understood',
            spectral=dict(windows=list(WINDOWS), hops=[n // 4 for n in WINDOWS],
                window='periodic Hann', scaling='spectrum', onesided=True,
                boundary='zeros', padded=True, minimum_length_padding='zero-pad to window length',
                magnitude_floor=1e-5,
                convergence='Frobenius magnitude difference / reference magnitude norm; no phase',
                log_magnitude='20 log10(max(magnitude,1e-5))',
                log_spectral_distance='Mean frame RMS dB difference across bins',
                multi_resolution='Arithmetic mean of the three per-resolution values'),
            reference_scope='Noisy or codec references measure agreement, not validated clean-reference quality',
            threshold='No equivalence to a waveform-error threshold is asserted'),
        dependencies=versions,
        source_sha256={name: sha256_file(Path(__file__).with_name(name))
            for name in ('bigcodec_audio_quality.py', 'bigcodec_compare.py', 'bigcodec_common.py')})


def write_report(report, destination, inputs):
    destination = Path(destination)
    if any(destination.resolve() == Path(path).resolve() or
           destination.exists() and Path(path).exists() and destination.samefile(path)
           for path in inputs):
        raise ValueError('--report must not overwrite an input WAV')
    serialized = json.dumps(report, indent=2, allow_nan=False) + '\n'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(serialized)
            stream.close()
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return serialized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--actual', required=True, type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    try:
        report = compare_quality(args.reference, args.actual)
        if args.report is not None:
            text = write_report(report, args.report, (args.reference, args.actual))
        else:
            text = json.dumps(report, indent=2, allow_nan=False) + '\n'
        print(text, end='')
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
