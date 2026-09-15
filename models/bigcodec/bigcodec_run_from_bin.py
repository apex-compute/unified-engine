#!/usr/bin/env python3
"""Reconstruct a complete audio file with one resident BigCodec FPGA bin."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import soundfile as sf
import torch

from bigcodec_common import (DEFAULT_CHECKPOINT, pad_audio, read_audio, restore_audio,
                             sha256_file, save_tokens)
from bigcodec_precompiled import (DEFAULT_BIN, StreamingEngine, WholeGraphBackend,
                                  load_artifact, udc)
from bigcodec_layout import layout_for_hardware
from yolov5_common import configure_hardware_runtime


def validate_output_paths(input_path, bin_path, output_path, tokens_path, report_path):
    """Protect the source, deployment image and checkpoint from output aliases."""
    paths = [Path(p) for p in (input_path, bin_path, output_path, tokens_path, report_path)]

    def aliases(left, right):
        return (left.resolve() == right.resolve()
                or left.exists() and right.exists() and left.samefile(right))

    for index, path in enumerate(paths):
        if any(aliases(path, other) for other in paths[index + 1:]):
            raise ValueError('Input, bin, output, tokens and report paths must be distinct files')
    if any(aliases(path, DEFAULT_CHECKPOINT) for path in paths[2:]):
        raise ValueError('Output files must not overwrite the pinned BigCodec checkpoint')


def validate_dram_capacity(hardware, dram_size_gb):
    """Check a validated artifact's used addresses against detected, visible DDR.

    The controller exposes DDR starting at 0x80000000, within a 32-bit byte
    address window. Reserved arena limits are not allocation sizes: legacy
    artifacts can fit a smaller board when their actual tensor extent fits.
    """
    if (not isinstance(dram_size_gb, int) or isinstance(dram_size_gb, bool)
            or dram_size_gb <= 0):
        raise ValueError('FPGA must report a positive integer DRAM capacity in GiB')
    base = udc.DRAM_START_ADDR
    addressable_bytes = min(dram_size_gb * 2**30, 0x100000000 - base)
    required_end = max(hardware['model_base'] + hardware['model_image'].numel(),
                       hardware['tensor_end'],
                       hardware['input_address'] + hardware['input_bytes'])
    required_bytes = required_end - base
    if required_bytes <= 0 or required_bytes > addressable_bytes:
        raise ValueError(
            f'BigCodec bin requires {required_bytes} bytes of addressable DRAM '
            f'(end 0x{required_end:x}); FPGA provides {addressable_bytes} visible bytes '
            f'from its reported {dram_size_gb} GiB')
    return dict(memory_layout=layout_for_hardware(hardware).name,
                detected_dram_size_gib=dram_size_gb,
                dram_addressable_bytes=addressable_bytes,
                dram_required_bytes=required_bytes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin', type=Path, default=DEFAULT_BIN)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--device', choices=['rk'], default='rk')
    parser.add_argument('--dev', default='xdma0')
    parser.add_argument('--cpu-core', type=int, default=6)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--expected-version', type=lambda value: int(value, 0),
                        help='Require this FPGA build before reset/upload (for controlled comparisons)')
    args = parser.parse_args()
    if args.output.suffix.lower() != '.wav':
        parser.error('--output must be a WAV file')
    if args.tokens is None:
        args.tokens = args.output.with_suffix('.tokens.npz')
    if args.report is None:
        args.report = args.output.with_suffix('.metrics.json')
    try:
        validate_output_paths(args.input, args.bin, args.output, args.tokens, args.report)
    except ValueError as error:
        parser.error(str(error))
    if args.cpu_core not in os.sched_getaffinity(0):
        parser.error('--cpu-core is not available')
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be positive and finite')
    if args.expected_version is not None and not 0 <= args.expected_version <= 0xFFFFFFFF:
        parser.error('--expected-version must be a 32-bit unsigned build value')
    torch.set_num_threads(1)
    os.sched_setaffinity(0, {args.cpu_core})
    began = time.perf_counter()
    bin_sha256 = sha256_file(args.bin)
    payload = load_artifact(args.bin)
    if sha256_file(args.bin) != bin_sha256:
        raise RuntimeError('BigCodec bin changed while it was being loaded')
    artifact_load_s = time.perf_counter() - began
    preprocessing = time.perf_counter()
    audio, metadata = read_audio(args.input)
    padded = pad_audio(audio)
    if len(padded) != payload['hardware']['compiled_samples']:
        parser.error(f"This bin accepts {payload['hardware']['compiled_samples']} padded samples; "
                     f"input needs {len(padded)}. Compile with bigcodec_compile.py --input {args.input}")
    preprocess_s = time.perf_counter() - preprocessing
    lock_path = Path(f'/tmp/pcie_ci_hw_{socket.gethostname()}.lock')
    try:
        lock = lock_path.open('a')
    except PermissionError:
        # An existing shared lock may be owned by the CI account. flock
        # works on a read-only descriptor and still locks the same inode.
        lock = lock_path.open('r')
    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f'FPGA is reserved by another job: {lock_path}')
        clock, info, detected_clock = configure_hardware_runtime(
            device=args.device, dev=args.dev, cycle_override_ns=None)
        if info.axi_data_width_bits != 256:
            parser.error('BigCodec bin requires an AXI256 FPGA')
        try:
            capacity = validate_dram_capacity(payload['hardware'], info.dram_size_gb)
        except ValueError as error:
            parser.error(str(error))
        with StreamingEngine(clock_period_ns=clock,
                             conv_geometry_mode=udc.CONV_GEOMETRY_QUEUE_CONFIG) as engine:
            if engine.is_queue_busy():
                raise RuntimeError('FPGA queue is already busy')
            version_before = engine.get_hardware_version()
            if args.expected_version is not None and version_before != args.expected_version:
                raise RuntimeError(f'FPGA build 0x{version_before:08x} differs from required '
                                   f'0x{args.expected_version:08x}; no reset/upload performed')
            engine.software_reset(run_dram_self_test=False)
            backend = WholeGraphBackend(engine, payload, axi_data_width_bits=256,
                                        timeout_s=args.timeout)
            execute_started = time.perf_counter()
            waveform, tokens = backend.execute(padded)
            execute_s = time.perf_counter() - execute_started
            version_after = engine.get_hardware_version()
            if version_after != version_before:
                raise RuntimeError('FPGA build changed during execution; output is not a valid comparison')
            hardware_version = f'0x{version_after:08x}'
    postprocessing = time.perf_counter()
    reconstructed = restore_audio(waveform, metadata)
    for path in (args.output, args.tokens, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, reconstructed, metadata['source_rate'], subtype='FLOAT')
    save_tokens(args.tokens, tokens, metadata)
    postprocess_s = time.perf_counter() - postprocessing
    duration = metadata['source_samples'] / metadata['source_rate']
    audio_s = preprocess_s + execute_s + postprocess_s
    report = dict(backend.last_metrics)
    report.update(capacity)
    report.update(model='bigcodec', backend='hardware', execution_scope='whole utterance',
                  checkpoint_sha256=payload['checkpoint_sha256'], bin_sha256=bin_sha256,
                  precision=payload.get('precision'),
                  parameter_bytes=payload['hardware']['parameter_bytes'],
                  program_bytes=payload['hardware']['program_size'],
                  instructions=payload['hardware']['instructions'],
                  input_sha256=metadata['source_sha256'], output_sha256=sha256_file(args.output),
                  tokens_sha256=sha256_file(args.tokens), hardware_version=hardware_version,
                  axi_data_width_bits=256, detected_clock_ns=detected_clock,
                  host_cpu_affinity=sorted(os.sched_getaffinity(0)), host_math_threads=1,
                  compiled_samples=len(padded), source_samples=metadata['source_samples'],
                  output_samples=len(reconstructed), source_sample_rate=metadata['source_rate'],
                  model_sample_rate=16000, tokens=len(tokens), token_hop_ms=12.5,
                  audio_duration_s=duration, artifact_load_s=artifact_load_s,
                  model_upload_s=backend.model_upload_s,
                  model_upload_bytes=payload['hardware']['model_image'].numel(),
                  input_upload_bytes=payload['hardware']['input_bytes'],
                  output_read_bytes=payload['hardware']['output_bytes'],
                  audio_preprocess_s=preprocess_s, execution_s=execute_s,
                  audio_postprocess_s=postprocess_s,
                  audio_processing_s=audio_s, audio_rtf=audio_s / duration,
                  total_elapsed_s=time.perf_counter() - began,
                  approximation=payload.get('approximation'),
                  input=str(args.input.resolve()), output=str(args.output.resolve()))
    # The hardware counter increments once per16cycles and is32bits wide.
    counter_period_s = 2**32 * udc.UE_PIPELINE_COUNTER_CLK_DIV * clock / 1e9
    report['fpga_execution_s'] = (report['fpga_cycles'] * clock / 1e9
                                  if execute_s < counter_period_s else None)
    report['fpga_counter_wrap_possible'] = execute_s >= counter_period_s
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('TEST_RESULT:' + json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    main()
