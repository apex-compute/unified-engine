#!/usr/bin/env python3
"""Run the pinned DPDFNet2 streaming ONNX reference on one 16-kHz WAV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from dpdfnet_common import DEFAULT_MODEL_PATH, download_model, validate_digest


def vorbis_window(length: int) -> torch.Tensor:
    index = torch.arange(length, dtype=torch.float32)
    inner = torch.sin(torch.pi * (index + 0.5) / length)
    return torch.sin(0.5 * torch.pi * inner.square())


def initial_state(metadata: dict[str, str]) -> np.ndarray:
    required = ("state_size", "erb_norm_state_size", "spec_norm_state_size",
                "erb_norm_init", "spec_norm_init")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise RuntimeError(f"ONNX metadata is missing {missing}")
    state = np.zeros(int(metadata["state_size"]), dtype=np.float32)
    erb = np.fromstring(metadata["erb_norm_init"], sep=",", dtype=np.float32)
    spec = np.fromstring(metadata["spec_norm_init"], sep=",", dtype=np.float32)
    ne, ns = int(metadata["erb_norm_state_size"]), int(metadata["spec_norm_state_size"])
    if erb.size != ne or spec.size != ns:
        raise RuntimeError("ONNX normalization metadata has inconsistent lengths")
    state[:ne] = erb
    state[ne:ne + ns] = spec
    return state


def fit_length(value: np.ndarray, length: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    if value.size >= length:
        return value[:length]
    return np.pad(value, (0, length - value.size))


def resample(value: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(value, dtype=np.float32)
    tensor = torch.from_numpy(np.asarray(value, dtype=np.float32)).reshape(1, -1)
    return AF.resample(tensor, source_rate, target_rate)[0].numpy()


def attenuation_limit(noisy: np.ndarray, enhanced: np.ndarray,
                      limit_db: float | None) -> np.ndarray:
    if limit_db is None:
        return enhanced
    if not np.isfinite(limit_db) or limit_db < 0:
        raise ValueError("--attn-limit-db must be finite and non-negative")
    aligned = np.zeros_like(noisy)
    if noisy.shape[1] > 4:
        aligned[:, 4:] = noisy[:, :-4]
    alpha = 10.0 ** (-limit_db / 20.0)
    return np.ascontiguousarray(alpha * aligned + (1.0 - alpha) * enhanced)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attn-limit-db", type=float)
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError as exc:
        parser.error("onnxruntime is required; install models/dpdfnet/requirements.txt")

    model = args.model.expanduser().resolve()
    if args.download:
        model = download_model(model)
    elif not model.is_file():
        parser.error(f"model not found: {model}; pass --download")
    digest = validate_digest(model)

    waveform, sample_rate = sf.read(args.input, dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1, dtype=np.float32)
    if waveform.ndim != 1:
        parser.error(f"expected mono/stereo WAV, got shape {waveform.shape}")
    source_samples = waveform.size
    waveform_model_rate = resample(waveform, sample_rate, 16000)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map
    state = initial_state(metadata)
    window_length, hop = int(metadata["window_length"]), int(metadata["hop_length"])
    window = vorbis_window(window_length)

    padded = np.pad(waveform_model_rate, (0, window_length))
    spectrum = torch.stft(
        torch.from_numpy(padded), n_fft=window_length, hop_length=hop,
        win_length=window_length, window=window, center=True,
        pad_mode="reflect", normalized=False, return_complex=True)
    spectrum = spectrum.transpose(0, 1).contiguous()
    spec_ri = torch.view_as_real(spectrum).numpy()[None].astype(np.float32)

    input_spec, input_state = (item.name for item in session.get_inputs())
    output_spec, output_state = (item.name for item in session.get_outputs())
    enhanced_frames = []
    inference_s = 0.0
    for frame in range(spec_ri.shape[1]):
        value = np.ascontiguousarray(spec_ri[:, frame:frame + 1])
        start = time.perf_counter()
        enhanced, state = session.run(
            [output_spec, output_state], {input_spec: value, input_state: state})
        inference_s += time.perf_counter() - start
        enhanced_frames.append(enhanced)
    enhanced = np.concatenate(enhanced_frames, axis=1)
    enhanced = attenuation_limit(spec_ri, enhanced, args.attn_limit_db)

    complex_spectrum = torch.view_as_complex(
        torch.from_numpy(enhanced[0]).contiguous()).transpose(0, 1)
    reconstructed = torch.istft(
        complex_spectrum, n_fft=window_length, hop_length=hop,
        win_length=window_length, window=window, center=True,
        normalized=False).numpy()
    delayed = np.concatenate((reconstructed[2 * window_length:],
                              np.zeros(2 * window_length, dtype=np.float32)))
    result = resample(delayed, 16000, sample_rate)
    result = fit_length(result, source_samples)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, result, sample_rate)

    frames = spec_ri.shape[1]
    duration_s = source_samples / sample_rate
    summary = {
        "model": "dpdfnet2",
        "backend": "onnxruntime-cpu",
        "onnx_sha256": digest,
        "input": str(args.input.resolve()),
        "output": str(output),
        "sample_rate": sample_rate,
        "model_sample_rate": 16000,
        "audio_duration_s": duration_s,
        "frames": frames,
        "state_size": int(state.size),
        "neural_inference_s": inference_s,
        "average_frame_ms": inference_s * 1000.0 / frames,
        "neural_rtf": inference_s / (frames * hop / sample_rate),
    }
    print("TEST_RESULT:" + json.dumps(summary))


if __name__ == "__main__":
    main()
