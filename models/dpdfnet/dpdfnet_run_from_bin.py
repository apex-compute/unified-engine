#!/usr/bin/env python3
"""Enhance audio files or spectrum frames with one resident DPDFNet2 bin.

The host converts audio to/from spectra; the entire neural graph runs on
the FPGA with one START/HALT per frame and resident recurrent state.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
YOLO_HELPERS = ROOT / "models" / "yolov5s"
for search_path in (HERE, ROOT, YOLO_HELPERS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import user_dma_core as udc
from dpdfnet_precompiled import WholeGraphBackend, load_artifact
from yolov5_common import configure_hardware_runtime


def _load_frames(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.shape == (1, 1, 161, 2):
        value = value[None]
    if value.ndim != 5 or tuple(value.shape[1:]) != (1, 1, 161, 2):
        raise ValueError(
            "spectrum input must have shape [1,1,161,2] or "
            "[frames,1,1,161,2]")
    if value.shape[0] == 0:
        raise ValueError("spectrum input contains no frames")
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError("spectrum input must be numeric")
    if not np.isfinite(value).all():
        raise ValueError("spectrum input contains NaN or infinity")
    return np.asarray(value, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bin", type=Path,
        default=HERE / "dpdfnet_bin" / "dpdfnet2-andromeda.bin")
    parser.add_argument(
        "--input", type=Path, required=True,
        help="audio file (WAV/FLAC), or NumPy spectra [T,1,1,161,2]")
    parser.add_argument(
        "--output", type=Path,
        help="enhanced mono WAV for audio input, or .npy for spectrum input")
    parser.add_argument("--device", default="rk")
    parser.add_argument("--dev", default="xdma0")
    parser.add_argument("--cycle", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--trace-tail", type=Path, metavar="DIR",
        help="export the final frame's last 8192 hardware events")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    audio_mode = args.input.suffix.lower() != ".npy"
    if args.output is None:
        args.output = HERE / "dpdfnet_bin" / (
            "enhanced.wav" if audio_mode else "enhanced_spec.npy")
    if audio_mode and args.output.suffix.lower() != ".wav":
        parser.error("audio input requires a .wav output")
    if not audio_mode and args.output.suffix.lower() == ".wav":
        parser.error("WAV output requires the original audio file as --input")
    if args.input.expanduser().resolve() == args.output.expanduser().resolve():
        parser.error("--output must differ from --input")

    total_started = time.perf_counter()
    load_started = time.perf_counter()
    payload = load_artifact(args.bin)
    artifact_load_s = time.perf_counter() - load_started
    preprocess_started = time.perf_counter()
    audio_input = None
    if audio_mode:
        from dpdfnet_audio import read_audio, synthesize_audio

        audio_input = read_audio(args.input)
        frames = audio_input.frames
    else:
        frames = _load_frames(args.input)
    preprocess_s = time.perf_counter() - preprocess_started
    clock, hw_info, detected_clock = configure_hardware_runtime(
        device=args.device, dev=args.dev, cycle_override_ns=args.cycle)
    print(
        f"DPDFNet2 FPGA neural inference, frames={frames.shape[0]}, "
        f"AXI={hw_info.axi_data_width_bits}, clock={detected_clock:.4f} ns")
    print(f"Single bin: {args.bin.expanduser().resolve()}")
    print("Contract: resident recurrent state, one input write/kick/HALT/output read per frame")
    if audio_input is not None:
        print(f"Audio: {audio_input.source_samples / audio_input.source_sample_rate:.3f} s, "
              f"{audio_input.source_sample_rate} Hz; host STFT/iSTFT, mono output")
    engine = udc.UnifiedEngine(
        clock_period_ns=clock,
        conv_geometry_mode=udc.CONV_GEOMETRY_QUEUE_CONFIG)
    engine.software_reset(run_dram_self_test=False)
    backend = WholeGraphBackend(
        engine, payload, axi_data_width_bits=hw_info.axi_data_width_bits,
        timeout_s=args.timeout)

    outputs = []
    started = time.perf_counter()
    with torch.inference_mode():
        for frame_index, frame in enumerate(frames):
            # Trace export is deliberately armed only for the final kick.
            if args.trace_tail is not None and frame_index + 1 == len(frames):
                backend.trace_tail_path = args.trace_tail
            outputs.append(backend.execute(frame).float().numpy())
    execution_s = time.perf_counter() - started
    output = np.stack(outputs)
    postprocess_started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if audio_input is not None:
        import soundfile as sf

        waveform = synthesize_audio(output, audio_input)
        # Float WAV preserves output amplitudes without clipping or rescaling.
        sf.write(args.output, waveform, audio_input.source_sample_rate, subtype="FLOAT")
    else:
        if frames.shape[0] == 1:
            output = output[0]
        np.save(args.output, output, allow_pickle=False)
    postprocess_s = time.perf_counter() - postprocess_started

    result = {
        "model": "dpdfnet2",
        "backend": "hardware",
        "cpu_neural_ops": 0,
        "frames": int(frames.shape[0]),
        "full_graph": True,
        "state_resident": True,
        "artifact_load_s": artifact_load_s,
        "model_upload_s": backend.model_upload_seconds,
        "model_upload_bytes": backend.model_upload_bytes,
        "model_upload_writes": 1,
        "input_upload_writes": backend.input_upload_writes,
        "program_kicks": backend.program_kicks,
        "halts": backend.program_kicks,
        "output_reads": backend.output_reads,
        "intermediate_upload_writes": 0,
        "intermediate_output_reads": 0,
        "execution_elapsed_s": execution_s,
        "fpga_cycles_sum": backend.cycles,
        "fpga_execution_s_sum": backend.cycles * clock * 1e-9,
        "output": str(args.output.expanduser().resolve()),
    }
    if audio_input is not None:
        result.update({
            "input": str(args.input.expanduser().resolve()),
            "audio_duration_s": audio_input.source_samples / audio_input.source_sample_rate,
            "sample_rate": audio_input.source_sample_rate,
            "model_sample_rate": 16000,
            "output_samples": len(waveform),
            "output_channels": 1,
            "audio_preprocess_s": preprocess_s,
            "audio_postprocess_s": postprocess_s,
            "total_elapsed_s": time.perf_counter() - total_started,
        })
    if backend.trace_tail_result is not None:
        result["trace_tail"] = backend.trace_tail_result
        result["trace_export_s"] = backend.trace_export_seconds
        print(f"Perfetto trace: {backend.trace_tail_result['perfetto']}")
    print("TEST_RESULT:" + json.dumps(result))


if __name__ == "__main__":
    main()
