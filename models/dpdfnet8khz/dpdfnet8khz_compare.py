#!/usr/bin/env python3
"""Verify native 8 kHz CPU agreement before qualifying measured FPGA timing.

The reference manifest names hashed input/CPU spectrum .npy files. Each FPGA
case supplies <id>_fpga.npy and <id>_fpga.log, whose TEST_RESULT binds a hashed
metrics JSON containing raw per-frame timing arrays. No model is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from dpdfnet8khz_common import load_config


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _positive(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def _load_spectrum(path, frames):
    values = np.load(path, allow_pickle=False)
    if frames == 1 and values.shape == (1, 1, 81, 2):
        values = values[None]
    if values.shape != (frames, 1, 1, 81, 2) or values.dtype.kind != "f":
        raise ValueError(f"{path.name}: expected floating [frames,1,1,81,2] spectrum")
    return np.asarray(values, dtype=np.float64)


def _verified_file(directory, name, expected):
    if not isinstance(name, str) or not name:
        raise ValueError("missing file path")
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("reference path escapes the manifest directory")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"{path.name}: missing/invalid SHA256")
    if _sha256(path) != expected:
        raise ValueError(f"{path.name}: SHA256 mismatch")
    return path


def _accuracy(cpu, fpga, relative_limit, silence_limit):
    finite = bool(np.isfinite(cpu).all() and np.isfinite(fpga).all())
    if not finite:
        return {"passed": False, "finite": False, "reason": "nonfinite spectrum"}
    delta = fpga - cpu
    error_energy = float(np.sum(delta * delta))
    cpu_energy = float(np.sum(cpu * cpu))
    maximum = float(np.max(np.abs(delta)))
    relative = math.sqrt(error_energy / cpu_energy) if cpu_energy else None
    return {
        "passed": relative <= relative_limit if relative is not None else maximum <= silence_limit,
        "finite": True, "relative_l2": relative,
        "relative_l2_reason": "zero-energy CPU reference" if relative is None else None,
        "rmse": math.sqrt(error_energy / cpu.size), "max_abs_error": maximum,
        "sum_squared_error": error_energy, "cpu_energy": cpu_energy,
        "elements": int(cpu.size),
    }


def timing_statistics(milliseconds, deadline_ms):
    raw = np.asarray(milliseconds)
    if raw.dtype.kind not in "ifu":
        raise ValueError("timings must contain numeric milliseconds")
    values = raw.astype(np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("timings must be a nonempty list of finite positive milliseconds")
    return {
        "frames": int(values.size), "mean_ms": float(np.mean(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(np.max(values)),
        "deadline_misses": int(np.count_nonzero(values > deadline_ms)),
        "deadline_miss_fraction": float(np.mean(values > deadline_ms)),
        "rtf_at_10ms_hop": float(np.mean(values)) / 10.0,
    }


def _run_record(directory, case_id):
    log_path = directory / f"{case_id}_fpga.log"
    lines = [line[len("TEST_RESULT:"):] for line in log_path.read_text().splitlines()
             if line.startswith("TEST_RESULT:")]
    if len(lines) != 1:
        raise ValueError("FPGA log must contain exactly one TEST_RESULT")
    summary = json.loads(lines[0])
    if not isinstance(summary, dict):
        raise ValueError("TEST_RESULT must be a JSON object")
    named = summary.get("metrics_file")
    if not isinstance(named, str) or not named:
        raise ValueError("FPGA log does not bind a metrics_file")
    reported = Path(named).expanduser()
    candidates = [directory / reported.name, directory / f"{case_id}_fpga.metrics.json"]
    candidates.append(reported if reported.is_absolute() else directory / reported)
    metrics_path = next((path for path in candidates if path.is_file()), None)
    if metrics_path is None:
        raise ValueError("bound metrics file is missing")
    metrics_digest = _sha256(metrics_path)
    if summary.get("metrics_sha256") != metrics_digest:
        raise ValueError("metrics JSON SHA256 mismatch")
    metrics = json.loads(metrics_path.read_text())
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a JSON object")
    # The summary and raw timing record must describe the same execution.
    for key in ("model", "frames", "input_sha256", "output_sha256", "bin_sha256",
                "onnx_sha256", "axi_data_width_bits", "hardware_version",
                "cycle_override_ns", "trace_enabled"):
        if key not in summary or key not in metrics or summary[key] != metrics[key]:
            raise ValueError(f"summary/metrics mismatch or missing field: {key}")
    return metrics, {"log_sha256": _sha256(log_path), "metrics_sha256": metrics_digest}


def compare(reference_manifest, fpga_directory, *, deadline_ms=10.0,
            max_relative_l2=0.25, max_silence_abs=1e-6, expected_axi_bits=256):
    """Return a fail-closed corpus report; invalid cases remain in the report."""
    if not all(_positive(value) for value in (deadline_ms, max_relative_l2, max_silence_abs)):
        raise ValueError("deadline and accuracy limits must be finite and positive")
    if deadline_ms > 10.0:
        raise ValueError("real-time deadline cannot exceed the model's 10 ms hop")
    if expected_axi_bits not in (256, 512):
        raise ValueError("expected AXI width must be 256 or 512")
    reference_manifest = Path(reference_manifest).expanduser().resolve()
    directory = Path(fpga_directory).expanduser().resolve()
    manifest = json.loads(reference_manifest.read_text())
    config = load_config()
    if (manifest.get("model") != "dpdfnet2_8khz"
            or manifest.get("onnx_sha256") != config["onnx_sha256"]
            or manifest.get("model_sample_rate") != 8000):
        raise ValueError("reference manifest is not the pinned native 8 kHz model")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("reference manifest contains no cases")
    if any(not isinstance(case, dict) for case in cases):
        raise ValueError("reference cases must be JSON objects")
    ids = [case.get("id") for case in cases]
    if (any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("reference case IDs must be unique safe names")

    records, all_host, all_fpga = [], [], []
    for case in cases:
        record = {"id": case["id"], "validation_errors": [], "accuracy": None,
                  "timing": None, "realtime_smoke_pass": False}
        records.append(record)
        frames = case.get("frames")
        try:
            if not isinstance(frames, int) or isinstance(frames, bool) or frames <= 0:
                raise ValueError("invalid reference frame count")
            if case.get("finite_spec_and_state") is not True or case.get("state_reset") is not True:
                raise ValueError("CPU manifest does not attest finite output/state and fresh state")
            input_path = _verified_file(reference_manifest.parent, case.get("input"), case.get("input_sha256"))
            cpu_path = _verified_file(reference_manifest.parent, case.get("cpu"), case.get("cpu_sha256"))
            source = _load_spectrum(input_path, frames)
            cpu = _load_spectrum(cpu_path, frames)
            if not np.isfinite(source).all() or not np.isfinite(cpu).all():
                raise ValueError("nonfinite input or CPU reference")
            fpga_path = directory / f"{case['id']}_fpga.npy"
            fpga = _load_spectrum(fpga_path, frames)
            record["input_sha256"] = case["input_sha256"]
            record["cpu_sha256"] = case["cpu_sha256"]
            record["fpga_sha256"] = _sha256(fpga_path)
            record["accuracy"] = _accuracy(cpu, fpga, max_relative_l2, max_silence_abs)
            if _positive(case.get("neural_inference_s")):
                record["cpu_neural_ms_per_frame"] = case["neural_inference_s"] * 1000 / frames
        except (OSError, ValueError, TypeError, KeyError) as exc:
            record["validation_errors"].append(str(exc))

        try:
            metrics, provenance = _run_record(directory, case["id"])
            record.update(provenance)
            if metrics.get("frames") != frames:
                raise ValueError("FPGA/reference frame count mismatch")
            for key, expected in (("model", "dpdfnet2_8khz"),
                                  ("onnx_sha256", config["onnx_sha256"]),
                                  ("input_sha256", case.get("input_sha256")),
                                  ("output_sha256", record.get("fpga_sha256"))):
                if metrics.get(key) != expected or expected is None:
                    raise ValueError(f"FPGA run does not bind the compared data/model: {key}")
            if not re.fullmatch(r"[0-9a-f]{64}", str(metrics.get("bin_sha256", ""))):
                raise ValueError("missing/invalid bin SHA256")
            host = np.asarray(metrics.get("host_frame_ms"))
            fpga_times = np.asarray(metrics.get("fpga_frame_ms"))
            if host.shape != (frames,) or fpga_times.shape != (frames,):
                raise ValueError("raw timing arrays do not cover every compared frame")
            timing = {"host": timing_statistics(host, deadline_ms),
                      "fpga": timing_statistics(fpga_times, deadline_ms),
                      "qualification_exclusions": []}
            record["timing"] = timing
            record["hardware"] = {key: metrics.get(key) for key in (
                "axi_data_width_bits", "hardware_version", "bin_sha256")}
            exclusions = timing["qualification_exclusions"]
            if metrics.get("axi_data_width_bits") != expected_axi_bits:
                exclusions.append("detected AXI width differs from the requested target")
            if not isinstance(metrics.get("hardware_version"), str) or not metrics["hardware_version"]:
                exclusions.append("missing hardware version")
            if metrics.get("cycle_override_ns") is not None:
                exclusions.append("clock override was used")
            if (not _positive(metrics.get("detected_clock_ns"))
                    or metrics.get("effective_clock_ns") != metrics.get("detected_clock_ns")):
                exclusions.append("timing does not use the detected physical clock")
            if metrics.get("trace_enabled") is not False:
                exclusions.append("trace instrumentation was enabled or unspecified")
            for key, expected in (("backend", "hardware"), ("full_graph", True),
                                  ("state_resident", True), ("cpu_neural_ops", 0),
                                  ("program_kicks", frames), ("halts", frames),
                                  ("input_upload_writes", frames), ("output_reads", frames),
                                  ("intermediate_upload_writes", 0), ("intermediate_output_reads", 0)):
                if metrics.get(key) != expected:
                    exclusions.append(f"resident whole-graph execution contract mismatch: {key}")
            timing["observed_deadline_pass"] = (
                timing["host"]["deadline_misses"] == 0 and timing["fpga"]["deadline_misses"] == 0)
            if not exclusions:
                all_host.extend(host.tolist())
                all_fpga.extend(fpga_times.tolist())
            record["realtime_smoke_pass"] = bool(
                not record["validation_errors"] and record["accuracy"]
                and record["accuracy"]["passed"] and not exclusions
                and timing["observed_deadline_pass"])
        except (OSError, ValueError, TypeError, KeyError) as exc:
            record["validation_errors"].append(str(exc))

    hardware = {(record["hardware"]["axi_data_width_bits"], record["hardware"]["hardware_version"],
                 record["hardware"]["bin_sha256"]) for record in records if record.get("hardware")}
    same_deployment = len(hardware) == 1
    return {
        "model": "dpdfnet2_8khz", "reference_manifest": str(reference_manifest),
        "reference_manifest_sha256": _sha256(reference_manifest),
        "fpga_directory": str(directory), "deadline_ms": deadline_ms,
        "expected_axi_data_width_bits": expected_axi_bits,
        "accuracy_screen": {"max_relative_l2": max_relative_l2, "max_silence_abs": max_silence_abs,
                            "description": "CPU spectrum agreement without gain or delay fitting; gross-error screening, not perceptual quality certification"},
        "timing_scope": "every measured host neural frame, including DMA/wait; excludes upload, audio STFT/iSTFT, file I/O; no warmup frames discarded",
        "qualification_scope": "observed corpus only; does not establish hard real-time guarantees or end-to-end audio latency",
        "speech_quality_qualified": False,
        "same_deployment": same_deployment,
        "realtime_smoke_pass": same_deployment and all(record["realtime_smoke_pass"] for record in records),
        "cases_total": len(records), "cases_smoke_pass": sum(record["realtime_smoke_pass"] for record in records),
        "host_timing": timing_statistics(all_host, deadline_ms) if all_host else None,
        "fpga_timing": timing_statistics(all_fpga, deadline_ms) if all_fpga else None,
        "cases": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--fpga-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-ms", type=float, default=10.0)
    parser.add_argument("--max-relative-l2", type=float, default=0.25)
    parser.add_argument("--max-silence-abs", type=float, default=1e-6)
    parser.add_argument("--expected-axi-bits", type=int, choices=(256, 512), default=256)
    args = parser.parse_args()
    if args.output.resolve() == args.reference_manifest.resolve():
        parser.error("--output must differ from --reference-manifest")
    try:
        report = compare(args.reference_manifest, args.fpga_dir, deadline_ms=args.deadline_ms,
                         max_relative_l2=args.max_relative_l2, max_silence_abs=args.max_silence_abs,
                         expected_axi_bits=args.expected_axi_bits)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in (
        "realtime_smoke_pass", "cases_total", "cases_smoke_pass", "host_timing", "fpga_timing")}, indent=2))
    print(f"Report: {args.output.resolve()}")
    raise SystemExit(0 if report["realtime_smoke_pass"] else 1)


if __name__ == "__main__":
    main()
