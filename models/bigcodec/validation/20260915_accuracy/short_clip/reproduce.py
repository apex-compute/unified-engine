"""Verify packaged hashes and recompute measured waveform/token decomposition.

Run with Python, NumPy and soundfile. No Torch, model checkpoint, device, or
repository imports are used. This script only reads files and prints JSON.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local_path(name):
    path = (HERE / name).resolve()
    require(path.is_relative_to(HERE), f'Path escapes package: {name}')
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(name):
    return json.loads(local_path(name).read_text())


def verify_file(record):
    path = local_path(record['path'])
    require(digest(path) == record['sha256'], f'Hash mismatch: {path.name}')
    if 'bytes' in record:
        require(path.stat().st_size == record['bytes'], f'Size mismatch: {path.name}')
    return path


def read_wave(record, metadata):
    data, rate = sf.read(verify_file(record), dtype='float32', always_2d=False)
    require(rate == metadata['source_rate'], 'Waveform sample rate differs')
    require(data.shape == (metadata['source_samples'],), 'Waveform length/channels differ')
    require(np.isfinite(data).all(), 'Nonfinite waveform sample')
    return data.astype(np.float64)


def read_tokens(record, metadata):
    with np.load(verify_file(record), allow_pickle=False) as archive:
        tokens = archive['tokens']
        actual_metadata = json.loads(str(archive['metadata'].item()))
    require(actual_metadata == metadata, 'Token metadata differs from the reference')
    require(tokens.ndim == 1 and tokens.dtype.kind in 'iu', 'Invalid token array')
    require(tokens.size == metadata['padded_samples'] // metadata['hop_length'], 'Wrong token count')
    require(tokens.min() >= 0 and tokens.max() < 8192, 'Token ID outside codebook')
    return tokens


def relative(error_energy, reference_energy):
    # This evidence has nonzero reference energies. Make the zero-energy policy
    # explicit instead of silently hiding it behind a denominator epsilon.
    if reference_energy == 0:
        return 0.0 if error_energy == 0 else None
    return math.sqrt(error_energy / reference_energy)


def decompose(fpga, same, reference):
    token = same - reference
    decoder = fpga - same
    full = fpga - reference
    er, es = float(reference @ reference), float(same @ same)
    et, ed, ef = float(token @ token), float(decoder @ decoder), float(full @ full)
    cross = float(2 * (token @ decoder))
    require(math.isclose(ef, et + ed + cross, rel_tol=1e-12, abs_tol=1e-10), 'Energy identity failed')
    return dict(
        full_fpga_vs_official_l2=relative(ef, er),
        token_only_cpu_vs_official_l2=relative(et, er),
        same_token_fpga_decoder_vs_cpu_l2=relative(ed, es),
        decoder_error_relative_to_official_energy=relative(ed, er),
        reference_energy=er, same_token_cpu_energy=es,
        full_error_energy=ef, token_error_energy=et, decoder_error_energy=ed,
        two_error_cross_term=cross,
        token_decoder_error_cosine=cross / (2 * math.sqrt(et * ed)) if et * ed else None,
        identity='full_error_energy = token_error_energy + decoder_error_energy + two_error_cross_term')


def main():
    manifest = read_json('manifest.json')
    for file in manifest['files']:
        verify_file(file)
    data = read_json('results.json')
    metadata = data['audio']
    read_wave(data['input'], metadata)
    reference = read_wave(data['reference']['wav'], metadata)
    official = read_tokens(data['reference']['tokens'], metadata)
    ref_run = read_json(data['reference']['run_record']['path'])
    require(ref_run['tokens_sha256'] == data['reference']['tokens']['sha256'], 'Reference tokens binding failed')
    require(ref_run['output_sha256'] == data['reference']['wav']['sha256'], 'Reference waveform binding failed')
    require(ref_run['checkpoint_sha256'] == data['checkpoint']['sha256'], 'Reference checkpoint differs')
    require(metadata['source_sha256'] == data['input']['sha256'], 'Input metadata binding failed')
    rows = []
    for variant in data['variants']:
        files = variant['files']
        fpga = read_wave(files['fpga_wav'], metadata)
        same = read_wave(files['same_token_cpu_wav'], metadata)
        tokens = read_tokens(files['actual_tokens'], metadata)
        run = read_json(variant['fpga_run_record']['path'])
        cpu = read_json(variant['same_token_cpu_run_record']['path'])
        compile_record = read_json(variant['compile_record']['path'])
        require(run['input_sha256'] == data['input']['sha256'], 'FPGA input binding failed')
        require(run['output_sha256'] == files['fpga_wav']['sha256'], 'FPGA waveform binding failed')
        require(run['tokens_sha256'] == cpu['tokens_sha256'] == files['actual_tokens']['sha256'], 'Same-token decode binding failed')
        require(cpu['output_sha256'] == files['same_token_cpu_wav']['sha256'], 'CPU waveform binding failed')
        require(run['checkpoint_sha256'] == cpu['checkpoint_sha256'] == data['checkpoint']['sha256'], 'Checkpoint differs')
        require(run['bin_sha256'] == compile_record['bin_sha256'] == variant['artifact']['sha256'], 'Bin binding failed')
        for key in ('parameter_bytes', 'program_bytes', 'instructions'):
            require(run[key] == compile_record[key] == variant['artifact'][key], f'Bin count differs: {key}')
        require(run['model_upload_bytes'] == compile_record['resident_bytes'] == variant['artifact']['resident_bytes'], 'Resident size differs')
        require(run['model_upload_bytes'] == run['parameter_bytes'] + run['program_bytes'], 'Resident extent differs')
        for key in ('model_upload_writes', 'input_upload_writes', 'program_kicks', 'halts', 'output_reads'):
            require(run[key] == 1, f'Unexpected execution count: {key}')
        require(run['cpu_neural_ops'] == 0 and run['finite'], 'Unexpected backend or invalid output')
        metrics = decompose(fpga, same, reference)
        for key, expected in variant['metrics'].items():
            actual = metrics[key]
            if isinstance(expected, (int, float)):
                require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), f'Metric differs: {variant["name"]}/{key}')
            else:
                require(actual == expected, f'Metric differs: {key}')
        matches = int(np.count_nonzero(tokens == official))
        require(matches == variant['tokens_matching_official'] and tokens.size == variant['frames'], 'Token count differs')
        rows.append(dict(name=variant['name'], tokens_matching_official=matches,
                         frames=int(tokens.size), metrics=metrics))
    print(json.dumps(dict(status='PASS', verified_files=len(manifest['files']),
                          all_samples_finite=True, variants=rows), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
