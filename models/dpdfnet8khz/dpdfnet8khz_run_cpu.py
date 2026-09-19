#!/usr/bin/env python3
"""Enhance a WAV using the pinned native 8-kHz DPDFNet2 ONNX CPU model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf

from dpdfnet8khz_audio import HOP_LENGTH, SAMPLE_RATE, read_audio, synthesize_audio
from dpdfnet8khz_common import (
    DEFAULT_MODEL_PATH, download_model, initial_state, load_config,
    validate_digest, validate_metadata,
)


def create_session(model: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"])
    config = load_config()
    inputs = {item.name: list(item.shape) for item in session.get_inputs()}
    outputs = {item.name: list(item.shape) for item in session.get_outputs()}
    if inputs != config["onnx_inputs"] or outputs != config["onnx_outputs"]:
        raise RuntimeError(f"unexpected 8-kHz streaming ABI: inputs={inputs}, outputs={outputs}")
    validate_metadata(session.get_modelmeta().custom_metadata_map)
    return session


def run_cpu(input_path: Path, output_path: Path,
            model_path: Path = DEFAULT_MODEL_PATH,
            attn_limit_db: float | None = None) -> dict:
    """Run one file with fresh recurrent state, preserving source rate and length."""
    input_path, output_path, model_path = (
        Path(path).expanduser().resolve() for path in (input_path, output_path, model_path))
    if input_path == output_path:
        raise ValueError("--output must differ from --input")
    if output_path == model_path:
        raise ValueError("--output must differ from --model")
    if output_path.suffix.lower() != ".wav":
        raise ValueError("--output must be a .wav file")
    if attn_limit_db is not None and (not np.isfinite(attn_limit_db) or attn_limit_db < 0):
        raise ValueError("--attn-limit-db must be finite and non-negative")
    total_started = time.perf_counter()
    digest = validate_digest(model_path)
    session_started = time.perf_counter()
    session = create_session(model_path)
    session_s = time.perf_counter() - session_started
    state = initial_state(session.get_modelmeta().custom_metadata_map)
    state_shape = state.shape
    preprocess_started = time.perf_counter()
    audio = read_audio(input_path)
    preprocess_s = time.perf_counter() - preprocess_started
    enhanced_frames, inference_s = [], 0.0
    for index, frame in enumerate(audio.frames):
        started = time.perf_counter()
        enhanced, state = session.run(
            ["spec_e", "state_out"], {"spec": frame, "state_in": state})
        inference_s += time.perf_counter() - started
        if (enhanced.shape != frame.shape or state.shape != state_shape
                or not np.isfinite(enhanced).all() or not np.isfinite(state).all()):
            raise RuntimeError(f"invalid or nonfinite CPU output/state at frame {index}")
        enhanced_frames.append(enhanced)
    postprocess_started = time.perf_counter()
    result = synthesize_audio(np.stack(enhanced_frames), audio, attn_limit_db)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, result, audio.source_sample_rate, subtype="FLOAT")
    postprocess_s = time.perf_counter() - postprocess_started
    frames = len(audio.frames)
    return {
        "model": "dpdfnet2_8khz", "backend": "onnxruntime-cpu",
        "onnx_sha256": digest, "input": str(input_path), "output": str(output_path),
        "sample_rate": audio.source_sample_rate, "model_sample_rate": SAMPLE_RATE,
        "audio_duration_s": audio.source_samples / audio.source_sample_rate,
        "output_samples": len(result), "output_channels": 1, "frames": frames,
        "state_size": int(state.size), "cpu_intra_op_threads": 1, "cpu_inter_op_threads": 1,
        "neural_inference_s": inference_s, "average_frame_ms": inference_s * 1000 / frames,
        "neural_rtf": inference_s / (frames * HOP_LENGTH / SAMPLE_RATE),
        "session_load_s": session_s, "audio_preprocess_s": preprocess_s,
        "audio_postprocess_s": postprocess_s,
        "total_elapsed_s": time.perf_counter() - total_started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attn-limit-db", type=float)
    args = parser.parse_args()
    model = args.model.expanduser().resolve()
    if args.download:
        model = download_model(model)
    elif not model.is_file():
        parser.error(f"model not found: {model}; pass --download")
    try:
        result = run_cpu(args.input, args.output, model, args.attn_limit_db)
    except (ValueError, ImportError) as exc:
        parser.error(str(exc))
    print("TEST_RESULT:" + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
