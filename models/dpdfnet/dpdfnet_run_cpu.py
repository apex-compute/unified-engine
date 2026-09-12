#!/usr/bin/env python3
"""Run the pinned DPDFNet2 streaming ONNX reference on one 16-kHz WAV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf

from dpdfnet_audio import (
    HOP_LENGTH, SAMPLE_RATE, attenuation_limit, fit_length, read_audio,
    resample, synthesize_audio, vorbis_window,
)
from dpdfnet_common import (
    DEFAULT_MODEL_PATH, download_model, initial_state, validate_digest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attn-limit-db", type=float)
    args = parser.parse_args()
    if args.input.expanduser().resolve() == args.output.expanduser().resolve():
        parser.error("--output must differ from --input")
    if args.attn_limit_db is not None and (
            not np.isfinite(args.attn_limit_db) or args.attn_limit_db < 0):
        parser.error("--attn-limit-db must be finite and non-negative")

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

    audio_input = read_audio(args.input)
    sample_rate = audio_input.source_sample_rate
    source_samples = audio_input.source_samples

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map
    state = initial_state(metadata)
    input_spec, input_state = (item.name for item in session.get_inputs())
    output_spec, output_state = (item.name for item in session.get_outputs())
    enhanced_frames = []
    inference_s = 0.0
    for value in audio_input.frames:
        start = time.perf_counter()
        enhanced, state = session.run(
            [output_spec, output_state], {input_spec: value, input_state: state})
        inference_s += time.perf_counter() - start
        enhanced_frames.append(enhanced)
    result = synthesize_audio(np.stack(enhanced_frames), audio_input, args.attn_limit_db)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, result, sample_rate, subtype="FLOAT")

    frames = len(audio_input.frames)
    duration_s = source_samples / sample_rate
    summary = {
        "model": "dpdfnet2",
        "backend": "onnxruntime-cpu",
        "onnx_sha256": digest,
        "input": str(args.input.resolve()),
        "output": str(output),
        "sample_rate": sample_rate,
        "model_sample_rate": SAMPLE_RATE,
        "audio_duration_s": duration_s,
        "frames": frames,
        "state_size": int(state.size),
        "neural_inference_s": inference_s,
        "average_frame_ms": inference_s * 1000.0 / frames,
        "neural_rtf": inference_s / (frames * HOP_LENGTH / SAMPLE_RATE),
    }
    print("TEST_RESULT:" + json.dumps(summary))


if __name__ == "__main__":
    main()
