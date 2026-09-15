#!/usr/bin/env python3
"""Evaluate paired noisy/clean speech through the WAV-to-WAV bin runner.

Every clip starts a fresh model state. Both recordings are resampled once
to 16 kHz, with no gain normalization or additional time alignment. Saved
WAVs allow listening to the same signals used by the objective metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys

import numpy as np
import soundfile as sf

from dpdfnet_audio import SAMPLE_RATE, resample
from dpdfnet_audio_metrics import evaluate_audio


HERE = Path(__file__).resolve().parent
METRICS = ("si_sdr_db", "stoi", "pesq_wb")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_mono(path):
    waveform, rate = sf.read(path, dtype="float32", always_2d=True)
    if not waveform.size or not np.isfinite(waveform).all():
        raise ValueError(f"empty/nonfinite audio: {path}")
    return resample(waveform.mean(axis=1, dtype=np.float32), rate, SAMPLE_RATE)


def aggregate(cases, backend):
    result = {}
    for metric in METRICS:
        values, deltas = [], []
        for case in cases:
            noisy = case["outputs"]["noisy"]["metrics"][metric]
            enhanced = case["outputs"].get(backend, {}).get("metrics", {}).get(metric)
            if enhanced is not None:
                values.append(enhanced)
                if noisy is not None:
                    deltas.append(enhanced - noisy)
        result[metric] = {
            "mean": statistics.fmean(values) if values else None,
            "valid_cases": len(values),
            "mean_improvement": statistics.fmean(deltas) if deltas else None,
            "paired_cases": len(deltas),
            "improved": sum(value > 1e-7 for value in deltas),
            "regressed": sum(value < -1e-7 for value in deltas),
        }
    return result


def write_results(directory, report):
    cases = report["cases"]
    noisy_cases = [case for case in cases if case.get("group") != "clean_control"]
    report["runner_errors"] = [
        {"id": case["id"], "backend": backend, "error": output["error"]}
        for case in cases for backend, output in case["outputs"].items() if "error" in output
    ]
    report["completed_cases"] = len(cases)
    report["complete"] = len(cases) == report.get("expected_cases", len(cases))
    report["summary"] = {
        backend: aggregate(noisy_cases, backend) for backend in ("noisy", *report["backends"])
    }
    report["by_noise"] = {
        noise: {backend: aggregate([case for case in noisy_cases if case["noise"] == noise], backend)
                for backend in ("noisy", *report["backends"])}
        for noise in sorted({case["noise"] for case in noisy_cases})
    }
    (directory / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (directory / "results.csv").open("w", newline="") as stream:
        fields = ("id", "noise", "snr_db", "speaker", "backend", *METRICS, "error", "metric_reasons")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            for backend, output in case["outputs"].items():
                writer.writerow({
                    **{key: case.get(key) for key in fields[:4]}, "backend": backend,
                    **{key: output.get("metrics", {}).get(key) for key in METRICS},
                    "error": output.get("error", ""),
                    "metric_reasons": json.dumps(output.get("metrics", {}).get("reasons", {})),
                })
    lines = [
        "# DPDFNet2 noisy speech evaluation", "",
        "Paired clean/noisy recordings; higher SI-SDR, STOI and PESQ-WB are better.",
        "Scores use the full 16-kHz waveform after the runner's model-delay compensation.",
        "No additional alignment or gain normalization is applied. These are objective",
        "scores on a small selected subset, not a human listening panel or full-corpus result.", "",
        f"Completed cases: {len(cases)}/{report.get('expected_cases', len(cases))}; runner failures: {len(report['runner_errors'])}.",
        "Each metric cell includes its valid clip count. Paired improvements and exclusions are in results.json.", "",
        "| Backend | SI-SDR (dB) | STOI | PESQ-WB |",
        "|---|---:|---:|---:|",
    ]
    def number(value):
        return "n/a" if value is None else f"{value:.3f}"
    for backend, summary in report["summary"].items():
        lines.append(f"| {backend} | " + " | ".join(
            f"{number(summary[key]['mean'])} (n={summary[key]['valid_cases']})" for key in METRICS) + " |")
    lines += ["", "## Listening samples", "",
              "All links play the actual scored files. Clean controls are excluded from the table above.", "",
              "| Case | Noise / SNR | Clean | Noisy | CPU | FPGA |",
              "|---|---|---|---|---|---|"]
    for case in cases:
        links = [f"[clean]({case['id']}/clean.wav)"]
        for backend in ("noisy", "cpu", "fpga"):
            output = case["outputs"].get(backend, {})
            links.append(f"[{backend}]({case['id']}/{backend}.wav)" if "metrics" in output else "unavailable")
        lines.append(f"| {case['id']} | {case['noise']} / {case.get('snr_db')} dB | " + " | ".join(links) + " |")
    lines += ["", "Per-case scores, metric failure reasons, input/artifact hashes and runner logs",
              "are recorded in `results.json`, `results.csv` and the case directories.", ""]
    (directory / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="JSON cases with id, clean, noisy, noise, snr_db, speaker")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bin", type=Path, default=HERE / "dpdfnet_bin/dpdfnet2-andromeda.bin")
    parser.add_argument("--backends", choices=("cpu", "fpga"), nargs="+", default=["cpu", "fpga"])
    parser.add_argument("--device", default="rk")
    parser.add_argument("--dev", default="xdma0")
    parser.add_argument("--limit", type=int, help="evaluate the first N selected clips")
    parser.add_argument("--clean-controls", action="store_true",
                        help="also process one clean-only clip per selected speaker")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    selections = list(manifest["cases"][:args.limit])
    if not selections:
        parser.error("manifest contains no selected cases")
    if args.clean_controls:
        speakers = set()
        for case in list(selections):
            if case["speaker"] not in speakers:
                speakers.add(case["speaker"])
                selections.append({**case, "id": case["id"] + "_clean", "noisy": case["clean"],
                                   "noisy_sha256": case.get("clean_sha256"),
                                   "noise": "clean_control", "snr_db": None, "group": "clean_control"})
    identifiers = [case["id"] for case in selections]
    if (len(set(identifiers)) != len(identifiers)
            or any(Path(name).name != name or name in (".", "..") for name in identifiers)):
        parser.error("case IDs must be unique directory names")
    directory = args.output_dir.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
        "dataset": {key: value for key, value in manifest.items() if key != "cases"},
        "bin_sha256": sha256(args.bin) if "fpga" in args.backends else None,
        "host": socket.gethostname(), "device": args.device, "dev": args.dev,
        "sample_rate": SAMPLE_RATE, "backends": list(dict.fromkeys(args.backends)),
        "expected_cases": len(selections),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "soundfile", "pystoi", "pesq")},
        "cases": [],
    }
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    errors = 0
    for index, selection in enumerate(selections):
        print(f"[{index + 1}/{len(selections)}] {selection['id']}: {selection['noise']} {selection.get('snr_db')} dB", flush=True)
        case_dir = directory / selection["id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        case = {**selection, "outputs": {}, "source_sha256": {}}
        signals = {}
        for kind in ("clean", "noisy"):
            source = manifest_path.parent / selection[kind]
            digest = sha256(source)
            expected = selection.get(f"{kind}_sha256")
            if expected is not None and digest != expected:
                raise ValueError(f"{source}: SHA-256 mismatch")
            case["source_sha256"][kind] = digest
            signals[kind] = read_mono(source)
            destination = case_dir / f"{kind}.wav"
            if source.resolve() == destination.resolve():
                raise ValueError(f"output directory would overwrite source audio: {source}")
            sf.write(destination, signals[kind], SAMPLE_RATE, subtype="FLOAT")
        if signals["clean"].shape != signals["noisy"].shape:
            raise ValueError(f"{selection['id']}: paired audio lengths differ")
        case["samples"] = len(signals["clean"])
        case["duration_s"] = case["samples"] / SAMPLE_RATE
        case["outputs"]["noisy"] = {"metrics": evaluate_audio(signals["clean"], signals["noisy"], SAMPLE_RATE)}
        for backend in report["backends"]:
            command = [sys.executable, str(HERE / ("dpdfnet_run_cpu.py" if backend == "cpu" else "dpdfnet_run_from_bin.py")),
                       "--input", str(case_dir / "noisy.wav"), "--output", str(case_dir / f"{backend}.wav")]
            if backend == "fpga":
                command += ["--bin", str(args.bin.resolve()), "--device", args.device, "--dev", args.dev]
            try:
                with (case_dir / f"{backend}.log").open("w") as log:
                    subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
                log = (case_dir / f"{backend}.log").read_text()
                runtime = json.loads(next(line[len("TEST_RESULT:"):] for line in log.splitlines() if line.startswith("TEST_RESULT:")))
                enhanced, rate = sf.read(case_dir / f"{backend}.wav", dtype="float32")
                if rate != SAMPLE_RATE or enhanced.shape != signals["clean"].shape:
                    raise ValueError("runner changed sample rate, channels or duration")
                metrics = evaluate_audio(signals["clean"], enhanced, SAMPLE_RATE)
                case["outputs"][backend] = {"metrics": metrics, "runtime": runtime,
                                            "sha256": sha256(case_dir / f"{backend}.wav")}
                print(f"  {backend}: " + ", ".join(f"{key}={metrics[key]}" for key in METRICS), flush=True)
            except (subprocess.CalledProcessError, ValueError, RuntimeError, StopIteration) as exc:
                case["outputs"][backend] = {"error": f"{type(exc).__name__}: {exc}"}
                errors += 1
                print(f"  {backend}: ERROR {exc}; see {case_dir / (backend + '.log')}", flush=True)
        report["cases"].append(case)
        write_results(directory, report)
    print(f"Saved {len(report['cases'])} cases to {directory / 'report.md'}; runner errors={errors}")
    return int(errors > 0)


if __name__ == "__main__":
    raise SystemExit(main())
