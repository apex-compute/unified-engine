#!/usr/bin/env python3
"""Compare saved FPGA and CPU noisy-audio evaluations without rerunning models.

Quality deltas are FPGA minus CPU, using recorded scores against the same
clean reference. Waveform errors use the verified saved WAVs directly, with
no alignment or gain adjustment. Clean controls stay in per-case records
but are excluded from all aggregates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
import soundfile as sf


METRICS = ("si_sdr_db", "stoi", "pesq_wb")
DELTA_TOLERANCE = 1e-7


def _number(value):
    return (float(value) if isinstance(value, (int, float))
            and not isinstance(value, bool) and math.isfinite(value) else None)


def _verified_wave(directory, case, backend, expected_rate):
    output = case.get("outputs", {}).get(backend)
    if output is None:
        return None, f"{backend} output missing"
    if output.get("error"):
        return None, f"{backend} failed: {output['error']}"
    if not output.get("sha256"):
        return None, f"{backend} WAV SHA256 missing"
    path = directory / case["id"] / f"{backend}.wav"
    if not path.resolve().is_relative_to(directory):
        raise ValueError(f"WAV path is outside the results directory: {path}")
    if not path.is_file():
        return None, f"{backend} WAV missing: {path}"
    if hashlib.sha256(path.read_bytes()).hexdigest() != output["sha256"]:
        raise ValueError(f"WAV SHA256 mismatch: {path}")
    value, rate = sf.read(path, dtype="float64")
    if rate != expected_rate:
        raise ValueError(f"WAV sample rate differs from results.json: {path}")
    if value.ndim != 1 or not value.size or not np.isfinite(value).all():
        raise ValueError(f"WAV must contain finite, nonempty mono audio: {path}")
    if case.get("samples") is not None and value.size != case["samples"]:
        raise ValueError(f"WAV sample count differs from results.json: {path}")
    return value, None


def _timing(case, backend_reason):
    if backend_reason:
        return {"reason": backend_reason}
    outputs = case["outputs"]
    cpu = outputs["cpu"].get("runtime", {})
    fpga = outputs["fpga"].get("runtime", {})
    frames = cpu.get("frames")
    if (not isinstance(frames, int) or isinstance(frames, bool) or frames <= 0
            or not isinstance(fpga.get("frames"), int)
            or isinstance(fpga.get("frames"), bool) or frames != fpga["frames"]):
        return {"reason": "CPU/FPGA frame counts are missing, invalid or different"}
    values = {
        "cpu_neural_s": _number(cpu.get("neural_inference_s")),
        "fpga_execution_s": _number(fpga.get("fpga_execution_s_sum")),
        "fpga_host_s": _number(fpga.get("execution_elapsed_s")),
    }
    missing = [name for name, value in values.items() if value is None or value <= 0]
    if missing:
        return {"reason": "missing or invalid timing: " + ", ".join(missing)}
    return {"frames": frames, **values}


def _exclusions(cases, key, metric=None):
    return [{"id": case["id"], "reason": record["reason"]}
            for case in cases
            if "reason" in (record := case[key][metric] if metric else case[key])]


def _aggregate(cases):
    quality = {}
    for metric in METRICS:
        pairs = [case["quality"][metric] for case in cases
                 if "reason" not in case["quality"][metric]]
        deltas = [pair["delta"] for pair in pairs]
        quality[metric] = {
            "paired_cases": len(pairs),
            "excluded_cases": len(cases) - len(pairs),
            "cpu_mean": statistics.fmean(pair["cpu"] for pair in pairs) if pairs else None,
            "fpga_mean": statistics.fmean(pair["fpga"] for pair in pairs) if pairs else None,
            "mean_delta": statistics.fmean(deltas) if deltas else None,
            "median_delta": statistics.median(deltas) if deltas else None,
            "fpga_better_cases": sum(value > DELTA_TOLERANCE for value in deltas),
            "fpga_worse_cases": sum(value < -DELTA_TOLERANCE for value in deltas),
            "tied_cases": sum(abs(value) <= DELTA_TOLERANCE for value in deltas),
            "exclusions": _exclusions(cases, "quality", metric),
        }
    waves = [case["waveform"] for case in cases if "reason" not in case["waveform"]]
    samples = sum(wave["samples"] for wave in waves)
    sse = sum(wave["sum_squared_error"] for wave in waves)
    energy = sum(wave["cpu_energy"] for wave in waves)
    waveform = {
        "paired_cases": len(waves), "excluded_cases": len(cases) - len(waves), "samples": samples,
        "sum_squared_error": sse, "cpu_energy": energy,
        "relative_l2": math.sqrt(sse / energy) if energy else None,
        "relative_l2_reason": None if energy else "no nonzero CPU reference energy",
        "rmse": math.sqrt(sse / samples) if samples else None,
        "max_abs_error": max((wave["max_abs_error"] for wave in waves), default=None),
        "exclusions": _exclusions(cases, "waveform"),
    }
    paired_timing = [case["timing"] for case in cases if "reason" not in case["timing"]]
    frames = sum(case["frames"] for case in paired_timing)
    timing = {"paired_cases": len(paired_timing), "excluded_cases": len(cases) - len(paired_timing), "frames": frames,
              "exclusions": _exclusions(cases, "timing")}
    for key in ("cpu_neural", "fpga_execution", "fpga_host"):
        total = sum(case[key + "_s"] for case in paired_timing)
        timing[key + "_s_sum"] = total
        timing[key + "_ms_per_frame"] = 1000 * total / frames if frames else None
    cpu_seconds = timing["cpu_neural_s_sum"]
    timing["fpga_execution_over_cpu"] = timing["fpga_execution_s_sum"] / cpu_seconds if cpu_seconds else None
    timing["fpga_host_over_cpu"] = timing["fpga_host_s_sum"] / cpu_seconds if cpu_seconds else None
    return {"quality": quality, "waveform": waveform, "timing": timing}


def compare_results(directory) -> dict:
    """Read and verify saved results, returning JSON-safe paired comparisons."""
    directory = Path(directory).expanduser().resolve()
    raw = (directory / "results.json").read_bytes()
    report = json.loads(raw)
    rate = report.get("sample_rate")
    if not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0:
        raise ValueError("results.json requires a positive integer sample_rate")
    cases, identifiers = [], set()
    for source in report["cases"]:
        identifier = source["id"]
        if (not isinstance(identifier, str) or identifier in ("", ".", "..")
                or Path(identifier).name != identifier or identifier in identifiers):
            raise ValueError("case IDs must be unique directory names")
        identifiers.add(identifier)
        cpu, cpu_reason = _verified_wave(directory, source, "cpu", rate)
        fpga, fpga_reason = _verified_wave(directory, source, "fpga", rate)
        backend_reason = "; ".join(reason for reason in (cpu_reason, fpga_reason) if reason)
        record = {key: source.get(key) for key in ("id", "noise", "snr_db", "speaker", "group")}
        record["excluded_clean_control"] = source.get("group") == "clean_control"
        record["output_sha256"] = {backend: source.get("outputs", {}).get(backend, {}).get("sha256")
                                    for backend in ("cpu", "fpga")}
        record["waveform"] = {"reason": backend_reason}
        if not backend_reason:
            if cpu.shape != fpga.shape:
                raise ValueError(f"CPU/FPGA WAV shapes differ for {identifier}")
            delta = fpga - cpu
            sse, energy = float(np.dot(delta, delta)), float(np.dot(cpu, cpu))
            record["waveform"] = {
                "samples": cpu.size, "sample_rate": rate,
                "sum_squared_error": sse, "cpu_energy": energy,
                "relative_l2": math.sqrt(sse / energy) if energy else None,
                "relative_l2_reason": None if energy else "zero CPU reference energy",
                "rmse": math.sqrt(sse / cpu.size), "max_abs_error": float(np.max(np.abs(delta))),
            }
        record["quality"] = {}
        for metric in METRICS:
            pair, reasons = {}, [backend_reason] if backend_reason else []
            for backend in ("cpu", "fpga"):
                metrics = source.get("outputs", {}).get(backend, {}).get("metrics", {})
                pair[backend] = _number(metrics.get(metric))
                if pair[backend] is None:
                    reasons.append(f"{backend}: " + metrics.get("reasons", {}).get(metric, "metric missing or nonfinite"))
            pair["delta"] = None if reasons else pair["fpga"] - pair["cpu"]
            if reasons:
                pair["reason"] = "; ".join(reasons)
            record["quality"][metric] = pair
        record["timing"] = _timing(source, backend_reason)
        cases.append(record)
    noisy = [case for case in cases if not case["excluded_clean_control"]]
    return {
        "results_directory": str(directory), "results_sha256": hashlib.sha256(raw).hexdigest(),
        "source_complete": report.get("complete"), "source_expected_cases": report.get("expected_cases"),
        "method": "Paired recorded quality scores against clean; FPGA minus CPU. Verified WAV errors use no alignment or gain adjustment. Timing is summed seconds divided by matched frame counts.",
        "delta_tolerance": DELTA_TOLERANCE,
        "cases_total": len(cases), "noisy_cases": len(noisy),
        "clean_controls_excluded": len(cases) - len(noisy),
        "summary": _aggregate(noisy), "cases": cases,
    }


def write_comparison(result, prefix):
    prefix = Path(prefix).expanduser().resolve()
    json_path, csv_path = Path(str(prefix) + ".json"), Path(str(prefix) + ".csv")
    directory = Path(result["results_directory"])
    protected = {(directory / name).resolve() for name in ("results.json", "results.csv")}
    if any(path.resolve() in protected for path in (json_path, csv_path)):
        raise ValueError("comparison output must not overwrite the original results")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    fields = ["id", "noise", "snr_db", "speaker", "excluded_clean_control"]
    fields += [f"{metric}_{backend}" for metric in METRICS for backend in ("cpu", "fpga", "delta")]
    fields += ["relative_l2", "rmse", "max_abs_error", "frames", "cpu_neural_s", "fpga_execution_s", "fpga_host_s", "reasons"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in result["cases"]:
            row = {key: case.get(key) for key in fields[:5]}
            row.update({f"{metric}_{backend}": case["quality"][metric][backend]
                        for metric in METRICS for backend in ("cpu", "fpga", "delta")})
            row.update({key: case["waveform"].get(key) for key in ("relative_l2", "rmse", "max_abs_error")})
            row.update({key: case["timing"].get(key) for key in ("frames", "cpu_neural_s", "fpga_execution_s", "fpga_host_s")})
            reasons = {metric: values["reason"] for metric, values in case["quality"].items() if "reason" in values}
            reasons.update({key: case[key]["reason"] for key in ("waveform", "timing") if "reason" in case[key]})
            row["reasons"] = json.dumps(reasons)
            writer.writerow(row)
    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    result = compare_results(args.results)
    prefix = args.output_prefix or args.results / "cpu_comparison"
    for path in write_comparison(result, prefix):
        print(path)
    print(json.dumps(result["summary"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
