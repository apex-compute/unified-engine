"""Intrusive quality metrics for aligned, mono speech at a known sample rate.

SI-SDR follows the scalar projection in Le Roux et al., equations (4)-(5),
with mean removal: https://arxiv.org/abs/1811.02508 . Standard STOI is provided
by https://github.com/mpariente/pystoi and wideband PESQ by
https://github.com/ludlows/PESQ . Those packages retain their standard internal
processing; this wrapper performs no delay search, trimming, or resampling.
"""

from __future__ import annotations

import importlib
import numbers
import warnings

import numpy as np


METRICS = ("si_sdr_db", "stoi", "pesq_wb")


def _signal(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real audio samples")
    try:
        signal = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real audio samples") from exc
    if signal.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional mono audio")
    if not signal.size:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(signal).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return signal


def _centered_unit_scale(signal):
    # SI-SDR is independently invariant to both signal scales. Scaling before
    # centering keeps float64 projection energies safe for tiny or large input.
    peak = float(np.max(np.abs(signal)))
    scaled = signal / peak if peak else signal.copy()
    return scaled - np.mean(scaled, dtype=np.float64)


def _si_sdr(clean, estimate):
    target = clean * (np.dot(estimate, clean) / np.dot(clean, clean))
    residual = estimate - target
    target_energy = float(np.dot(target, target))
    residual_energy = float(np.dot(residual, residual))
    if target_energy == 0.0:
        return None, "zero projection onto reference; SI-SDR is negative infinity"
    # Do not turn an exact gain match into a made-up finite score by adding an
    # epsilon to the denominator. Residuals at float64 roundoff are unresolved.
    precision_floor = (8 * np.finfo(np.float64).eps) ** 2
    if residual_energy <= precision_floor * float(np.dot(estimate, estimate)):
        return None, "zero residual within float64 precision; SI-SDR is positive infinity"
    score = 10.0 * (np.log10(target_energy) - np.log10(residual_energy))
    if not np.isfinite(score):
        return None, "SI-SDR calculation produced a nonfinite result"
    return float(score), None


def _package_metric(package, function, arguments, keywords):
    try:
        module = importlib.import_module(package)
    except Exception as exc:
        return None, f"{package} unavailable: {type(exc).__name__}: {exc}"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            score = float(getattr(module, function)(*arguments, **keywords))
        if caught:
            # pystoi returns the finite placeholder 1e-5 when speech is too
            # short after silence removal. Its warning makes that invalidity
            # explicit, so the placeholder must never enter benchmark averages.
            messages = "; ".join(
                f"{warning.category.__name__}: {warning.message}" for warning in caught)
            return None, f"{package} warning: {messages}"
        if not np.isfinite(score):
            return None, f"{package} returned a nonfinite score"
        if package == "pesq" and score < 0:
            return None, f"pesq returned an error code: {score}"
        return score, None
    except Exception as exc:
        return None, f"{package} failed: {type(exc).__name__}: {exc}"


def evaluate_audio(clean, estimate, sample_rate=16000):
    """Return JSON-safe SI-SDR, standard STOI, and wideband PESQ scores.

    Inputs must be equally long, finite, nonempty 1-D real arrays. They must
    already be aligned, including compensation for the model's fixed delay.
    Invalid input shapes, lengths, samples, or sample rate raise ``ValueError``.

    The result has ``si_sdr_db``, ``stoi``, and ``pesq_wb`` values, each a float
    or ``None``, and a ``reasons`` dictionary explaining every unavailable
    value. Silent/DC-only inputs, infinite SI-SDR limits, missing packages,
    library warnings, and rejected utterances never become fake finite scores.
    Wideband PESQ is evaluated only at 16 kHz. Input arrays are never modified.
    """
    if (isinstance(sample_rate, (bool, np.bool_))
            or not isinstance(sample_rate, numbers.Integral) or sample_rate <= 0):
        raise ValueError("sample_rate must be a positive integer in Hz")
    sample_rate = int(sample_rate)
    clean = _signal(clean, "clean")
    estimate = _signal(estimate, "estimate")
    if clean.shape != estimate.shape:
        raise ValueError(
            f"clean and estimate must have equal lengths; got {clean.size} and {estimate.size}")

    result = {metric: None for metric in METRICS}
    result["reasons"] = {}
    clean_centered = _centered_unit_scale(clean)
    estimate_centered = _centered_unit_scale(estimate)
    for name, signal in (("clean reference", clean_centered),
                         ("estimate", estimate_centered)):
        if not np.any(signal):
            reason = f"{name} has zero energy after mean removal (silent or constant audio)"
            result["reasons"] = {metric: reason for metric in METRICS}
            return result

    score, reason = _si_sdr(clean_centered, estimate_centered)
    result["si_sdr_db"] = score
    if reason:
        result["reasons"]["si_sdr_db"] = reason

    score, reason = _package_metric(
        "pystoi", "stoi", (clean.copy(), estimate.copy(), sample_rate), {"extended": False})
    result["stoi"] = score
    if reason:
        result["reasons"]["stoi"] = reason

    if sample_rate != 16000:
        result["reasons"]["pesq_wb"] = "wideband PESQ requires 16000 Hz audio"
    else:
        score, reason = _package_metric(
            "pesq", "pesq", (sample_rate, clean.copy(), estimate.copy()), {"mode": "wb"})
        result["pesq_wb"] = score
        if reason:
            result["reasons"]["pesq_wb"] = reason
    return result
