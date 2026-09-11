#!/usr/bin/env python3
"""Benchmark the pinned DPDFNet2 ONNX on CPU using FPGA benchmark frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from dpdfnet_common import DEFAULT_MODEL_PATH, initial_state, validate_digest


def _load_frames(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.shape == (1, 1, 161, 2):
        value = value[None]
    if value.ndim != 5 or tuple(value.shape[1:]) != (1, 1, 161, 2):
        raise ValueError(
            "spectrum input must have shape [1,1,161,2] or "
            "[frames,1,1,161,2]")
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise ValueError("spectrum input must contain finite numeric values")
    return np.ascontiguousarray(value, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--threads", type=int, default=1,
        help="ONNX Runtime intra-op CPU threads (default: 1)")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")

    try:
        import onnxruntime as ort
    except ImportError:
        parser.error(
            "onnxruntime is required; install models/dpdfnet/requirements.txt")

    model = args.model.expanduser().resolve()
    if not model.is_file():
        parser.error(f"model not found: {model}; run prepare_benchmark.sh first")
    digest = validate_digest(model)
    frames = _load_frames(args.input.expanduser().resolve())

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    load_started = time.perf_counter()
    session = ort.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"])
    session_load_s = time.perf_counter() - load_started
    state = initial_state(session.get_modelmeta().custom_metadata_map)
    input_spec, input_state = (item.name for item in session.get_inputs())
    output_spec, output_state = (item.name for item in session.get_outputs())

    outputs = []
    started = time.perf_counter()
    for frame in frames:
        enhanced, state = session.run(
            [output_spec, output_state],
            {input_spec: frame, input_state: state},
        )
        outputs.append(enhanced)
    execution_s = time.perf_counter() - started

    output = np.stack(outputs)
    if frames.shape[0] == 1:
        output = output[0]
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, output, allow_pickle=False)

    count = int(frames.shape[0])
    result = {
        "model": "dpdfnet2",
        "backend": "onnxruntime-cpu",
        "onnx_sha256": digest,
        "threads": args.threads,
        "frames": count,
        "session_load_s": session_load_s,
        "execution_elapsed_s": execution_s,
        "mean_latency_ms": execution_s * 1000.0 / count,
        "frames_per_second": count / execution_s,
        "real_time_factor_10ms_hop": execution_s / (count * 0.010),
        "output": str(output_path),
        "finite_output": bool(np.isfinite(output).all()),
    }
    print("TEST_RESULT:" + json.dumps(result))


if __name__ == "__main__":
    main()
