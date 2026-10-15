#!/usr/bin/env python3
"""Save reproducible native 8-kHz CPU references and recurrent-state stress cases.

Each input becomes one mono 8-kHz WAV pair and spectrum input/reference pair.
The output directory must be new or empty. This tool never uses an FPGA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import numpy as np
import soundfile as sf

from dpdfnet8khz_audio import SAMPLE_RATE, analyze_audio, resample, synthesize_audio
from dpdfnet8khz_common import (
    DEFAULT_MODEL_PATH, download_model, initial_state, sha256, validate_digest,
)
from dpdfnet8khz_run_cpu import create_session


SILENCE_ID = "silence_128"
TRANSITION_ID = "audio_silence_audio_384"


def transition_frames(first: np.ndarray, last: np.ndarray) -> np.ndarray:
    """Keep 128 frames from each endpoint, padding short clips with zeros."""
    def segment(frames):
        result = np.zeros((128, 1, 1, 81, 2), dtype=np.float32)
        result[:min(128, len(frames))] = frames[:128]
        return result

    return np.concatenate((segment(first), np.zeros((128, 1, 1, 81, 2), dtype=np.float32),
                           segment(last)))


def infer_frames(session, frames: np.ndarray, reset_state: np.ndarray):
    """Reset once per case and carry each returned state into the next frame."""
    state = reset_state.copy()
    outputs, seconds = [], 0.0
    for index, frame in enumerate(frames):
        started = time.perf_counter()
        enhanced, state = session.run(
            ["spec_e", "state_out"], {"spec": frame, "state_in": state})
        seconds += time.perf_counter() - started
        if (enhanced.shape != frame.shape or state.shape != reset_state.shape
                or not np.isfinite(enhanced).all() or not np.isfinite(state).all()):
            raise RuntimeError(f"invalid or nonfinite CPU output/state at frame {index}")
        outputs.append(enhanced)
    return np.stack(outputs), seconds


def prepare_validation(inputs, output: Path, model: Path = DEFAULT_MODEL_PATH,
                       *, download: bool = False) -> dict:
    inputs = [Path(path).expanduser().resolve() for path in inputs]
    output, model = (Path(path).expanduser().resolve() for path in (output, model))
    identifiers = [path.stem for path in inputs]
    if any(re.fullmatch(r"[A-Za-z0-9_-]+", name) is None for name in identifiers):
        raise ValueError("input filenames must use only letters, digits, underscores or hyphens before the extension")
    if (not inputs or len(set(identifiers)) != len(identifiers)
            or any(name in ("", ".", "..", SILENCE_ID, TRANSITION_ID) for name in identifiers)):
        raise ValueError("input stems must be unique and cannot use reserved stress-case IDs")
    if any(not path.is_file() for path in inputs):
        raise ValueError("every --input must name an existing audio file")
    if any(path.is_relative_to(output) for path in [*inputs, model]):
        raise ValueError("the output directory must not contain an input or model")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("--output must be a new or empty directory")
    if download:
        model = download_model(model)
    digest = validate_digest(model)
    session = create_session(model)
    reset_state = initial_state(session.get_modelmeta().custom_metadata_map)
    output.mkdir(parents=True, exist_ok=True)
    cases, endpoints = [], []

    def save_case(identifier, frames, extra):
        enhanced, seconds = infer_frames(session, frames, reset_state)
        input_path, cpu_path = (output / f"{identifier}_{kind}.npy" for kind in ("input", "cpu"))
        np.save(input_path, frames, allow_pickle=False)
        np.save(cpu_path, enhanced, allow_pickle=False)
        cases.append({
            "id": identifier, "frames": len(frames),
            "input": input_path.name, "cpu": cpu_path.name,
            "input_sha256": sha256(input_path), "cpu_sha256": sha256(cpu_path),
            "neural_inference_s": seconds, "cpu_ms_per_frame": seconds * 1000 / len(frames),
            "finite_spec_and_state": True, "state_reset": True, **extra,
        })
        return enhanced

    for identifier, path in zip(identifiers, inputs):
        waveform, rate = sf.read(path, dtype="float32", always_2d=True)
        if not waveform.size or not np.isfinite(waveform).all():
            raise ValueError(f"audio must contain finite, nonempty samples: {path}")
        mono = waveform.mean(axis=1, dtype=np.float32)
        native = resample(mono, rate, SAMPLE_RATE)
        audio = analyze_audio(native, SAMPLE_RATE)
        if not endpoints:
            endpoints.append(audio.frames[:128].copy())
        if len(endpoints) == 1:
            endpoints.append(audio.frames[:128].copy())
        else:
            endpoints[1] = audio.frames[:128].copy()
        noisy_path = output / f"{identifier}_noisy8k.wav"
        cpu_path = output / f"{identifier}_cpu8k.wav"
        enhanced = save_case(identifier, audio.frames, {
            "source": str(path), "source_sha256": sha256(path),
            "source_sample_rate": rate, "source_samples": len(waveform),
            "model_sample_rate": SAMPLE_RATE, "model_samples": len(native),
            "noisy_wav": noisy_path.name, "cpu_wav": cpu_path.name,
            "audio_duration_s": len(native) / SAMPLE_RATE,
        })
        sf.write(noisy_path, native, SAMPLE_RATE, subtype="FLOAT")
        sf.write(cpu_path, synthesize_audio(enhanced, audio), SAMPLE_RATE, subtype="FLOAT")
        cases[-1].update(noisy_wav_sha256=sha256(noisy_path), cpu_wav_sha256=sha256(cpu_path))
    save_case(SILENCE_ID, np.zeros((128, 1, 1, 81, 2), dtype=np.float32), {
        "description": "128 exact-zero spectrum frames from fresh model state"})
    save_case(TRANSITION_ID, transition_frames(*endpoints), {
        "description": f"{identifiers[0]} first128 frames +128 zero frames +{identifiers[-1]} first128 frames; recurrent state retained across all384frames; short clips zero-padded to128frames",
    })
    report = {"model": "dpdfnet2_8khz", "onnx_sha256": digest,
              "model_sample_rate": SAMPLE_RATE, "cpu_threads": 1, "cases": cases}
    (output / "cpu_reference_manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True,
                        help="audio file; repeat for additional cases, in corpus order")
    parser.add_argument("--output", type=Path, required=True, help="new or empty corpus directory")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    try:
        report = prepare_validation(args.input, args.output, args.model, download=args.download)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps({"manifest": str((args.output / "cpu_reference_manifest.json").resolve()),
                      "cases": len(report["cases"]),
                      "frames": sum(case["frames"] for case in report["cases"])}, indent=2))


if __name__ == "__main__":
    main()
