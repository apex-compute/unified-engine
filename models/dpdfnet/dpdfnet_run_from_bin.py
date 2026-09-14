#!/usr/bin/env python3
"""Enhance audio files or spectrum frames with one resident DPDFNet2 16 kHz bin.

The host converts audio to/from spectra; the entire neural graph runs on
the FPGA with one START/HALT per frame and resident recurrent state.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
from dpdfnet_common import sha256
# The scalar DMA helper is model-independent and shared with native 8 kHz.
from models.dpdfnet8khz.dpdfnet8khz_engine import StreamingEngine
from yolov5_common import configure_hardware_runtime


def validate_runtime_hardware(payload: dict, live_width: int) -> tuple[int, str]:
    """Enforce the AXI-256 target before resetting or uploading the model.

    Historical streaming-v2 bins omit an explicit target. Their compiler
    fixes UE_AXI_DATA_WIDTH_BITS to 256 during capture, so preserve those
    artifacts while refusing to run them on an incompatible transport.
    """
    hardware = payload["hardware"]
    explicit = "axi_data_width_bits" in hardware
    compiled_width = hardware.get("axi_data_width_bits", 256)
    if compiled_width != 256 or live_width != compiled_width:
        raise RuntimeError(
            "DPDFNet2 bin targets RK AXI-256; live hardware reports "
            f"AXI-{live_width}. Use the compatible RK-256 queue-CONFIG build")
    return compiled_width, "artifact-metadata" if explicit else "legacy-v2-compiler-contract"


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
    if value.dtype.kind not in "fiu":
        raise TypeError("spectrum input must contain real numeric values")
    if not np.isfinite(value).all():
        raise ValueError("spectrum input contains NaN or infinity")
    with np.errstate(over="ignore"):
        value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("spectrum input exceeds the finite float32 range")
    return value


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
    parser.add_argument("--cpu-core", type=int,
                        help="pin the host thread to an available CPU core for stable timing")
    parser.add_argument("--report", type=Path,
                        help="summary and per-frame timings JSON; default: OUTPUT.metrics.json")
    parser.add_argument(
        "--trace-tail", type=Path, metavar="DIR",
        help="export the final frame's last 8192 hardware events")
    args = parser.parse_args()
    if args.cpu_core is not None:
        if not hasattr(os, "sched_setaffinity"):
            parser.error("--cpu-core requires CPU affinity support")
        if args.cpu_core not in os.sched_getaffinity(0):
            parser.error("--cpu-core is not available to this process")
        os.sched_setaffinity(0, {args.cpu_core})
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    audio_mode = args.input.suffix.lower() != ".npy"
    if args.output is None:
        args.output = HERE / "dpdfnet_bin" / (
            "enhanced.wav" if audio_mode else "enhanced_spec.npy")
    if audio_mode and args.output.suffix.lower() != ".wav":
        parser.error("audio input requires a .wav output")
    if not audio_mode and args.output.suffix.lower() != ".npy":
        parser.error("spectrum input requires a .npy output")
    if args.input.expanduser().resolve() == args.output.expanduser().resolve():
        parser.error("--output must differ from --input")
    if args.report is None:
        args.report = args.output.with_suffix(".metrics.json")
    protected = {path.expanduser().resolve() for path in (args.input, args.bin)}
    if (args.output.expanduser().resolve() in protected
            or args.report.expanduser().resolve() in protected
            or args.report.expanduser().resolve() == args.output.expanduser().resolve()):
        parser.error("--output and --report must be distinct from each other, input and bin")

    total_started = time.perf_counter()
    load_started = time.perf_counter()
    bin_digest = sha256(args.bin)
    payload = load_artifact(args.bin)
    artifact_load_s = time.perf_counter() - load_started
    preprocess_started = time.perf_counter()
    input_digest = sha256(args.input)
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
    compiled_width, target_source = validate_runtime_hardware(payload, hw_info.axi_data_width_bits)
    print(
        f"DPDFNet2 16 kHz FPGA neural inference, frames={frames.shape[0]}, "
        f"AXI={hw_info.axi_data_width_bits}, clock={detected_clock:.4f} ns")
    print(f"Single bin: {args.bin.expanduser().resolve()}")
    print("Contract: resident recurrent state, one input write/kick/HALT/output read per frame")
    if audio_input is not None:
        print(f"Audio: {audio_input.source_samples / audio_input.source_sample_rate:.3f} s, "
              f"{audio_input.source_sample_rate} Hz; host STFT/iSTFT, mono output")
    engine = StreamingEngine(
        clock_period_ns=clock,
        conv_geometry_mode=udc.CONV_GEOMETRY_QUEUE_CONFIG)
    try:
        engine.software_reset(run_dram_self_test=False)
        backend = WholeGraphBackend(
            engine, payload, axi_data_width_bits=hw_info.axi_data_width_bits,
            timeout_s=args.timeout)
        outputs, frame_seconds, frame_cycles = [], [], []
        started = time.perf_counter()
        with torch.inference_mode():
            for frame_index, frame in enumerate(frames):
                # Trace export is deliberately armed only for the final kick.
                if args.trace_tail is not None and frame_index + 1 == len(frames):
                    backend.trace_tail_path = args.trace_tail
                frame_started = time.perf_counter()
                cycles_before = backend.cycles
                outputs.append(backend.execute(frame).float().numpy())
                frame_seconds.append(time.perf_counter() - frame_started)
                frame_cycles.append(backend.cycles - cycles_before)
        execution_s = time.perf_counter() - started
        hardware_version = f"0x{engine.get_hardware_version():08x}"
    finally:
        engine.close()
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
        with args.output.open("wb") as stream:
            np.save(stream, output, allow_pickle=False)
    postprocess_s = time.perf_counter() - postprocess_started
    if sha256(args.bin) != bin_digest or sha256(args.input) != input_digest:
        raise RuntimeError("bin or input changed during execution; discard this run")

    result = {
        "model": "dpdfnet2",
        "onnx_sha256": payload["onnx_sha256"],
        "bin_sha256": bin_digest,
        "input_sha256": input_digest,
        "output_sha256": sha256(args.output),
        "hardware_version": hardware_version,
        "dma_io": "cached-scalar-read-write",
        "host_cpu_affinity": (sorted(os.sched_getaffinity(0))
                              if hasattr(os, "sched_getaffinity") else None),
        "axi_data_width_bits": hw_info.axi_data_width_bits,
        "compiled_axi_data_width_bits": compiled_width,
        "compiled_target_source": target_source,
        "hw_info_raw": f"0x{hw_info.raw:08x}",
        "detected_clock_ns": detected_clock,
        "effective_clock_ns": clock,
        "trace_enabled": args.trace_tail is not None,
        "clock_override_enabled": args.cycle is not None,
        "cycle_override_ns": args.cycle,
        "frame_hop_ms": 10.0,
        "deadline_ms": 10.0,
        "fpga_frame_ms_mean": float(np.mean(frame_cycles)) * clock * 1e-6,
        "fpga_frame_ms_max": float(np.max(frame_cycles)) * clock * 1e-6,
        "fpga_frame_ms_p95": float(np.percentile(frame_cycles, 95)) * clock * 1e-6,
        "fpga_frame_ms_p99": float(np.percentile(frame_cycles, 99)) * clock * 1e-6,
        "fpga_frame_deadline_misses": int(np.count_nonzero(
            np.asarray(frame_cycles) * clock * 1e-9 > 0.010)),
        "host_frame_ms_mean": float(np.mean(frame_seconds)) * 1000,
        "host_frame_ms_p95": float(np.percentile(frame_seconds, 95)) * 1000,
        "host_frame_ms_p99": float(np.percentile(frame_seconds, 99)) * 1000,
        "host_frame_ms_max": float(np.max(frame_seconds)) * 1000,
        "host_frame_deadline_misses": int(np.count_nonzero(np.asarray(frame_seconds) > 0.010)),
        "host_neural_rtf": execution_s / (len(frames) * 0.010),
        "frame_timing_excludes": "one-time load/upload, host STFT/iSTFT and file I/O",
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
            "steady_audio_processing_s": preprocess_s + execution_s + postprocess_s,
            "steady_audio_rtf": (preprocess_s + execution_s + postprocess_s) / (
                audio_input.source_samples / audio_input.source_sample_rate),
        })
    if backend.trace_tail_result is not None:
        result["trace_tail"] = backend.trace_tail_result
        result["trace_export_s"] = backend.trace_export_seconds
        print(f"Perfetto trace: {backend.trace_tail_result['perfetto']}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        **result,
        "host_frame_ms": (np.asarray(frame_seconds) * 1000).tolist(),
        "fpga_frame_ms": (np.asarray(frame_cycles) * clock * 1e-6).tolist(),
    }, indent=2, allow_nan=False) + "\n")
    result["metrics_file"] = str(args.report.expanduser().resolve())
    result["metrics_sha256"] = sha256(args.report)
    print("TEST_RESULT:" + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
