#!/usr/bin/env python3
"""BigCodec CPU reference: full-utterance WAV reconstruction or token encode/decode."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from bigcodec_common import (
    CHECKPOINT_SHA256, DEFAULT_CHECKPOINT, HOP_LENGTH, SAMPLE_RATE,
    decode_tokens, encode_audio, load_models, load_tokens, read_audio,
    restore_audio, save_tokens, sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("roundtrip", "encode", "decode"), default="roundtrip")
    parser.add_argument("--input", type=Path, help="Input WAV for roundtrip/encode")
    parser.add_argument("--output", type=Path, help="Output WAV for roundtrip/decode")
    parser.add_argument("--tokens", type=Path, help="Output NPZ for encode; input NPZ for decode; optional for roundtrip")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metrics", type=Path, help="JSON report (default: output/tokens.metrics.json)")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cpu-core", type=int)
    args = parser.parse_args()
    if args.mode != "decode" and args.input is None:
        parser.error("--input is required for roundtrip/encode")
    if args.mode != "encode" and args.output is None:
        parser.error("--output is required for roundtrip/decode")
    if args.mode in ("encode", "decode") and args.tokens is None:
        parser.error("--tokens is required for encode/decode")
    if args.mode == "decode" and args.input is not None:
        parser.error("decode reads --tokens; omit --input")
    if args.mode == "encode" and args.output is not None:
        parser.error("encode writes --tokens; omit --output")
    if args.threads < 1:
        parser.error("--threads must be positive")
    args.metrics = args.metrics or (args.output or args.tokens).with_suffix(".metrics.json")
    output_paths = [p.resolve() for p in (args.output, args.metrics, args.tokens if args.mode != "decode" else None) if p is not None]
    input_paths = [p.resolve() for p in (args.input, args.checkpoint, args.tokens if args.mode == "decode" else None) if p is not None]
    if len(set(output_paths)) != len(output_paths) or set(output_paths) & set(input_paths):
        parser.error("Input, checkpoint, token and output files must have distinct paths")
    if args.cpu_core is not None:
        if args.cpu_core not in os.sched_getaffinity(0):
            parser.error(f"CPU core {args.cpu_core} is not available")
        os.sched_setaffinity(0, {args.cpu_core})
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    import numpy as np
    import soundfile as sf
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    start = time.perf_counter()
    encoder, decoder = load_models(args.checkpoint)
    load_s = time.perf_counter() - start
    processing_start = time.perf_counter()
    encode_s = decode_s = 0.0
    if args.mode == "decode":
        tokens, metadata = load_tokens(args.tokens)
    else:
        audio, metadata = read_audio(args.input)
        start = time.perf_counter()
        tokens, quantized = encode_audio(encoder, decoder, audio)
        encode_s = time.perf_counter() - start
        del quantized
        if args.tokens is not None:
            save_tokens(args.tokens, tokens, metadata)
    output = None
    if args.mode != "encode":
        start = time.perf_counter()
        reconstruction = decode_tokens(decoder, tokens)
        decode_s = time.perf_counter() - start
        output = restore_audio(reconstruction, metadata)
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite reconstructed audio")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.output, output, metadata["source_rate"], subtype="FLOAT")
    processing_s = time.perf_counter() - processing_start
    duration_s = metadata["source_samples"] / metadata["source_rate"]
    metrics = {
        "format": "bigcodec-cpu-run-v1", "mode": args.mode,
        "backend": "pytorch-cpu-fp32", "checkpoint_sha256": CHECKPOINT_SHA256,
        "torch_version": torch.__version__, "threads": args.threads,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "audio": metadata, "duration_s": duration_s,
        "token_count": int(tokens.size), "nominal_token_hop_ms": 1000 * HOP_LENGTH / SAMPLE_RATE,
        "nominal_codec_bits_per_second": SAMPLE_RATE / HOP_LENGTH * 13,
        "token_storage": "NPZ uint16 with metadata; not a packed 13-bit transport bitstream",
        "model_load_s": load_s, "encode_s": encode_s, "decode_s": decode_s,
        "neural_s": encode_s + decode_s,
        "neural_rtf": (encode_s + decode_s) / duration_s,
        "processing_s": processing_s, "processing_rtf": processing_s / duration_s,
        "processing_timing_scope": "audio/token I/O, preprocessing, neural inference and output write; excludes model loading",
        "output_samples": int(output.size) if output is not None else None,
        "output_finite": bool(np.isfinite(output).all()) if output is not None else None,
        "output_sha256": sha256_file(args.output) if output is not None else None,
        "tokens_sha256": sha256_file(args.tokens) if args.tokens is not None else None,
        "input_file": str(args.input) if args.input is not None else None,
        "output_file": str(args.output) if args.output is not None else None,
        "tokens_file": str(args.tokens) if args.tokens is not None else None,
        "execution": "full utterance; centered convolutions and recurrent state reset for each file",
    }
    metrics_path = args.metrics
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print("TEST_RESULT " + json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
