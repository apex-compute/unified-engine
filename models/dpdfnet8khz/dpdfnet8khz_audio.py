"""Audio framing for the pinned native 8-kHz DPDFNet2 streaming model.

The neural graph has a four-hop delay. Flush it before compensating that
delay so the final samples survive, including files shorter than a window.
These host audio operations are shared by the FPGA and CPU runners.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


SAMPLE_RATE = 8000
WINDOW_LENGTH = 160
HOP_LENGTH = 80
DELAY_FRAMES = 4
DELAY_SAMPLES = DELAY_FRAMES * HOP_LENGTH
# One extra window provides synthesis overlap after the delayed final sample.
FLUSH_SAMPLES = DELAY_SAMPLES + WINDOW_LENGTH


@dataclass(frozen=True)
class AudioInput:
    frames: np.ndarray
    source_sample_rate: int
    source_samples: int
    model_samples: int


def vorbis_window(length: int) -> torch.Tensor:
    index = torch.arange(length, dtype=torch.float32)
    inner = torch.sin(torch.pi * (index + 0.5) / length)
    return torch.sin(0.5 * torch.pi * inner.square())


def fit_length(value: np.ndarray, length: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    if value.size >= length:
        return value[:length]
    return np.pad(value, (0, length - value.size))


def resample(value: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(value, dtype=np.float32)
    import torchaudio.functional as AF

    tensor = torch.from_numpy(np.asarray(value, dtype=np.float32)).reshape(1, -1)
    return AF.resample(tensor, source_rate, target_rate)[0].numpy()


def attenuation_limit(noisy: np.ndarray, enhanced: np.ndarray,
                      limit_db: float | None) -> np.ndarray:
    if limit_db is None:
        return enhanced
    if not np.isfinite(limit_db) or limit_db < 0:
        raise ValueError("--attn-limit-db must be finite and non-negative")
    aligned = np.zeros_like(noisy)
    if noisy.shape[1] > DELAY_FRAMES:
        aligned[:, DELAY_FRAMES:] = noisy[:, :-DELAY_FRAMES]
    alpha = 10.0 ** (-limit_db / 20.0)
    return np.ascontiguousarray(alpha * aligned + (1.0 - alpha) * enhanced)


def analyze_audio(waveform: np.ndarray, sample_rate: int) -> AudioInput:
    waveform = np.asarray(waveform, dtype=np.float32)
    if (waveform.ndim not in (1, 2) or waveform.size == 0
            or not np.isfinite(waveform).all()):
        raise ValueError("audio must contain finite, nonempty mono or multichannel samples")
    if (not isinstance(sample_rate, (int, np.integer))
            or isinstance(sample_rate, (bool, np.bool_)) or sample_rate <= 0):
        raise ValueError("audio sample rate must be a positive integer")
    source_samples = waveform.shape[0]
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1, dtype=np.float32)
    mono = resample(waveform, sample_rate, SAMPLE_RATE)
    padded = np.pad(mono, (0, FLUSH_SAMPLES))
    spectrum = torch.stft(
        torch.from_numpy(padded), n_fft=WINDOW_LENGTH,
        hop_length=HOP_LENGTH, win_length=WINDOW_LENGTH,
        window=vorbis_window(WINDOW_LENGTH), center=True,
        pad_mode="reflect", normalized=False, return_complex=True)
    real_imag = torch.view_as_real(spectrum.transpose(0, 1).contiguous()).numpy()
    frames = np.ascontiguousarray(real_imag[:, None, None], dtype=np.float32)
    return AudioInput(frames, int(sample_rate), source_samples, mono.size)


def read_audio(path: Path) -> AudioInput:
    import soundfile as sf

    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return analyze_audio(waveform, sample_rate)


def synthesize_audio(enhanced_frames: np.ndarray, audio: AudioInput,
                     attn_limit_db: float | None = None) -> np.ndarray:
    frames = np.asarray(enhanced_frames, dtype=np.float32)
    if frames.shape != audio.frames.shape:
        raise ValueError(
            f"enhanced spectrum shape {frames.shape} does not match input {audio.frames.shape}")
    if not np.isfinite(frames).all():
        raise ValueError("enhanced spectrum contains NaN or infinity")
    enhanced = attenuation_limit(
        audio.frames[:, 0, 0][None], frames[:, 0, 0][None], attn_limit_db)[0]
    spectrum = torch.view_as_complex(
        torch.from_numpy(np.ascontiguousarray(enhanced))).transpose(0, 1)
    reconstructed = torch.istft(
        spectrum, n_fft=WINDOW_LENGTH, hop_length=HOP_LENGTH,
        win_length=WINDOW_LENGTH, window=vorbis_window(WINDOW_LENGTH),
        center=True, normalized=False,
        length=audio.model_samples + DELAY_SAMPLES).numpy()
    aligned = reconstructed[DELAY_SAMPLES:DELAY_SAMPLES + audio.model_samples]
    result = fit_length(
        resample(aligned, SAMPLE_RATE, audio.source_sample_rate), audio.source_samples)
    if not np.isfinite(result).all():
        raise ValueError("reconstructed audio contains NaN or infinity")
    return result
