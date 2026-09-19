#!/usr/bin/env python3
"""Reproduce decoder-scope BigCodec validation in a fresh ignored work area.

Default: print commands only. --compile-only never opens a device. --execute uses
one hardware runner at a time, with its existing flock and a version check inside
that lock. Frozen official CPU references are reused; no CPU model is recomputed.
Pass the printed --output-root to resume your reproduction. Published evidence
is protected; this utility cannot resume or overwrite the original measured batch.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, concurrent.futures, fcntl, hashlib, json, math, os, socket, struct
import subprocess, sys, threading, time
sys.dont_write_bytecode = True

def repository_root(script):
    for directory in Path(script).resolve().parents:
        if ((directory / 'models/bigcodec/bigcodec_compile.py').is_file()
                and (directory / 'user_dma_core.py').is_file()):
            return directory
    raise RuntimeError('Place these utilities inside the unified-engine repository')

ROOT = repository_root(__file__)
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'models/bigcodec/validation/20260914_noisy20s'
REPRODUCE_ROOT = ROOT / 'models/bigcodec/bigcodec_bin/accuracy_20260915/reproduce_runs'
OUTPUT_PARENT = REPRODUCE_ROOT / 'results'
DEFAULT_OUT = OUTPUT_PARENT / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '-' + str(os.getpid()))
PYTHON = sys.executable
ENV = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
CASES = ['bus', 'cafe', 'office', 'psquare', 'bus_low_snr', 'cafe_low_snr', 'office_low_snr', 'psquare_low_snr']
CHECKPOINT = '1fba3806e87cc01c1a65bea22fa1becefbbf46881e4219593c4d9f3cf56206b9'

def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()
def read(path): return json.loads(Path(path).read_text())
def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(path)
def require(condition, message):
    if not condition: raise RuntimeError(message)
def validate_output_root(path):
    resolved = Path(path).resolve()
    require(OUTPUT_PARENT.resolve() in resolved.parents,
            f'Reproduction output must be a child of {OUTPUT_PARENT}; published evidence is protected')
    return resolved
def check_hashes(hashes):
    for path, expected in hashes.items(): require(sha(ROOT / path) == expected, f'Frozen file changed: {path}')
def source_hashes():
    paths = list((ROOT / 'models/bigcodec').glob('*.py'))
    paths += list((ROOT / 'models/bigcodec/bigcodec_vq').rglob('*.py'))
    paths += [ROOT / p for p in ['user_dma_core.py', 'andromeda_hw_info.py', 'quant_lib.py',
        'models/yolov5s/yolov5_precompiled.py', 'models/yolov5s/yolov5_common.py',
        'models/dpdfnet8khz/dpdfnet8khz_engine.py']]
    paths += [Path(__file__).resolve(), HERE / 'build_report.py']
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(set(paths))}

def load_references():
    manifest_path = OLD / 'cpu_reference_manifest.json'; manifest = read(manifest_path)
    require(manifest['status'] == 'complete' and manifest['checkpoint_sha256'] == CHECKPOINT, 'Wrong CPU reference manifest')
    rows = {row['id']: row for row in manifest['cases']}
    require(set(rows) == set(CASES), 'Expected all eight original noisy cases')
    hashes = {str(manifest_path.relative_to(ROOT)): sha(manifest_path)}
    for case, row in rows.items():
        for path_key, hash_key in [('input', 'input_sha256'), ('output', 'output_sha256'),
                                   ('tokens_file', 'tokens_sha256'), ('metrics', 'metrics_sha256')]:
            path = ROOT / row[path_key]; require(sha(path) == row[hash_key], f'CPU/input hash mismatch: {path}')
            hashes[str(path.relative_to(ROOT))] = row[hash_key]
        for precision in ('bf16', 'if8'):
            for suffix in ('.wav', '.tokens.npz', '.metrics.json', '.comparison.json'):
                path = OLD / ('fpga_' + precision) / (case + suffix)
                hashes[str(path.relative_to(ROOT))] = sha(path)
    return rows, hashes

def baseline(row, precision):
    directory = OLD / ('fpga_' + precision); case = row['id']
    metrics_path = directory / (case + '.metrics.json'); comparison_path = directory / (case + '.comparison.json')
    metrics, comparison = read(metrics_path), read(comparison_path)
    require(metrics['input_sha256'] == row['input_sha256'], 'Old baseline input differs')
    require(metrics['output_sha256'] == sha(directory / (case + '.wav')), 'Old baseline output changed')
    require(metrics['tokens_sha256'] == sha(directory / (case + '.tokens.npz')), 'Old baseline tokens changed')
    require(comparison['actual_sha256'] == metrics['output_sha256'] and comparison['reference_sha256'] == row['output_sha256'], 'Old comparison wave hashes differ')
    require(comparison['tokens']['reference_sha256'] == row['tokens_sha256'] and comparison['tokens']['actual_sha256'] == metrics['tokens_sha256'], 'Old comparison token hashes differ')
    return dict(metrics_file=str(metrics_path), metrics_sha256=sha(metrics_path),
                comparison_file=str(comparison_path), comparison_sha256=sha(comparison_path),
                audio_processing_s=metrics['audio_processing_s'], rtf=metrics['audio_rtf'],
                waveform_relative_l2=comparison['relative_l2'], tokens_equal=comparison['tokens']['equal'],
                tokens_total=comparison['tokens']['count'], comparison=comparison)

def work_root(args):
    suffix = hashlib.sha256(str(args.output_root).encode()).hexdigest()[:12]
    return REPRODUCE_ROOT / 'work' / (args.output_root.name + '-' + suffix)

def make_job(args, row, precision):
    case = row['id']; output = args.output_root / ('fpga_' + precision)
    directory = work_root(args) / (case + '_' + precision)
    return dict(case=case, recurrent_precision=precision, lstm_math_scope=args.lstm_math_scope,
        status='queued', memory_layout=args.memory_layout, center_quantizer_scores=args.center_quantizer_scores, compensated_codebook=args.compensated_codebook, input=str(ROOT / row['input']), input_sha256=row['input_sha256'],
        bin=str(directory / 'model.bin'), compile_log=str(directory / 'compile.log'),
        run_log=str(directory / 'run.log'), comparison_log=str(directory / 'comparison.log'),
        output=str(output / (case + '.wav')), tokens=str(output / (case + '.tokens.npz')),
        metrics=str(output / (case + '.metrics.json')), comparison=str(output / (case + '.comparison.json')),
        reference=row, old_baseline=baseline(row, precision))

def compile_command(job, core):
    return [PYTHON, str(ROOT / 'models/bigcodec/bigcodec_compile.py'), '--input', job['input'],
        '--conv-precision', 'bf16', '--lstm-precision', job['recurrent_precision'],
        '--lstm-cell-precision', 'compensated', '--lstm-tanh-precision', 'compensated',
        '--lstm-fused-gates', '--lstm-math-scope', job['lstm_math_scope'], '--memory-layout', job['memory_layout'],
        '--cpu-core', str(core), '--output', job['bin'],
        '--center-quantizer-scores' if job['center_quantizer_scores'] else '--no-center-quantizer-scores',
        '--compensated-codebook' if job['compensated_codebook'] else '--no-compensated-codebook']

def comparison_command(job):
    row = job['reference']
    return ['taskset', '-c', '10', PYTHON, str(ROOT / 'models/bigcodec/bigcodec_compare.py'),
        '--reference', str(ROOT / row['output']), '--actual', job['output'],
        '--reference-tokens', str(ROOT / row['tokens_file']), '--actual-tokens', job['tokens'],
        '--report', job['comparison']]

def compile_record(job):
    for line in reversed(Path(job['compile_log']).read_text().splitlines()):
        if line.startswith('TEST_RESULT '):
            record = json.loads(line[len('TEST_RESULT '):]); break
    else: raise RuntimeError(f"No successful compile record: {job['compile_log']}")
    expected = dict(convolution_precision='bf16', lstm_recurrent_precision=job['recurrent_precision'],
        lstm_cell_precision='compensated', lstm_tanh_precision='compensated',
        lstm_math_scope=job['lstm_math_scope'], lstm_fused_gates=True, memory_layout=job['memory_layout'], center_quantizer_scores=job['center_quantizer_scores'], compensated_codebook=job['compensated_codebook'],
        native_samples=job['reference']['native_samples'], compiled_samples=job['reference']['padded_samples'],
        code_frames=job['reference']['tokens'])
    for key, value in expected.items(): require(record.get(key) == value, f'Compile setting differs: {key}')
    require(Path(job['bin']).stat().st_size == record['bin_bytes'] and sha(job['bin']) == record['bin_sha256'], 'Bin differs from compile log')
    require(record['program_bytes'] % 64 == 0 and record['instructions'] * 32 == record['program_bytes'], 'Invalid recorded instruction extent')
    return record

def validate_metrics(job, expected_version):
    record = read(job['metrics']); row = job['reference']; compiled = job['compile']
    pairs = [('bin_sha256', compiled['bin_sha256']), ('input_sha256', row['input_sha256']),
        ('output_sha256', sha(job['output'])), ('tokens_sha256', sha(job['tokens'])),
        ('checkpoint_sha256', CHECKPOINT), ('hardware_version', f'0x{expected_version:08x}'),
        ('compiled_samples', row['padded_samples']), ('tokens', row['tokens']),
        ('source_samples', row['input_samples']), ('output_samples', row['input_samples']),
        ('source_sample_rate', 16000), ('model_sample_rate', 16000), ('axi_data_width_bits', 256),
        ('memory_layout', job['memory_layout']), ('host_cpu_affinity', [6]), ('host_math_threads', 1), ('cpu_neural_ops', 0),
        ('waveform_padding_nonzero', 0), ('backend', 'hardware'), ('execution_scope', 'whole utterance')]
    for key, value in pairs: require(record.get(key) == value, f'Measured contract differs: {key}')
    for key in ('model_upload_writes', 'input_upload_writes', 'program_kicks', 'halts', 'output_reads'):
        require(record.get(key) == 1, f'Expected one {key}')
    require(record.get('finite') is True and record.get('waveform_padding_finite') is True, 'Nonfinite output/padding')
    require(0 < record['dram_required_bytes'] <= record['dram_addressable_bytes'] <= 2**31, 'DRAM extent invalid')
    precision = record['precision']; weights = 'BF16' if job['recurrent_precision'] == 'bf16' else 'IF8-INT'
    expected = dict(convolutions='BF16', lstm_input_weights='BF16', lstm_recurrent_weights=dict(encoder=weights, decoder=weights),
        lstm_cell='compensated', lstm_tanh='compensated', lstm_math_scope=job['lstm_math_scope'], lstm_fused_gates=True, center_quantizer_scores=job['center_quantizer_scores'], compensated_codebook=job['compensated_codebook'])
    for key, value in expected.items(): require(precision.get(key) == value, f'Measured precision differs: {key}')
    for key in ('audio_processing_s', 'audio_duration_s', 'audio_rtf', 'detected_clock_ns'):
        require(math.isfinite(record[key]) and record[key] > 0, f'Invalid timing: {key}')
    require(math.isclose(record['audio_duration_s'], row['input_samples']/16000, rel_tol=1e-12), 'Audio duration differs')
    require(math.isclose(record['audio_rtf'], record['audio_processing_s']/record['audio_duration_s'], rel_tol=1e-12), 'RTF calculation differs')
    return record

def validate_comparison(job):
    result = read(job['comparison']); row = job['reference']
    for key, value in [('reference_sha256', row['output_sha256']), ('actual_sha256', sha(job['output'])),
                       ('samples', row['input_samples']), ('sample_rate', 16000), ('channels', 1), ('finite', True)]:
        require(result.get(key) == value, f'Comparison differs: {key}')
    tokens = result['tokens']
    for key, value in [('reference_sha256', row['tokens_sha256']), ('actual_sha256', sha(job['tokens'])), ('count', row['tokens']),
                       ('source_sha256', row['input_sha256']), ('checkpoint_sha256', CHECKPOINT)]:
        require(tokens.get(key) == value, f'Token comparison differs: {key}')
    return result

def validate_destination_paths(jobs, protected_hashes):
    protected = [ROOT / name for name in protected_hashes]
    # Preserve all published measurements too, including hardlink aliases placed
    # inside an otherwise valid ignored reproduction directory.
    protected += [path for path in (ROOT / 'models/bigcodec/validation').rglob('*') if path.is_file()]
    destinations = []
    for job in jobs.values():
        for key in ('bin', 'compile_log', 'run_log', 'comparison_log', 'output', 'tokens', 'metrics', 'comparison'):
            path = Path(job[key])
            for other in protected + destinations:
                aliases = path.resolve() == other.resolve() or (path.exists() and other.exists() and path.samefile(other))
                require(not aliases, f'Output aliases a frozen file or another output: {path}')
            destinations.append(path)

def run_command(command, log):
    log = Path(log); log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists(): log.rename(log.with_name(log.name + '.' + str(time.time_ns()) + '.previous'))
    with log.open('w') as handle: subprocess.run(command, cwd=ROOT, env=ENV, stdout=handle, stderr=subprocess.STDOUT, check=True)

def guarded_runner(args):
    """Only called by an explicit --execute child; version check runs under runner flock."""
    require(args.execute, 'Hardware requires --execute')
    require(socket.gethostname().split('.')[0].lower() == 'italy', 'Hardware batch requires Italy')
    request = read(args.run_job); freeze = read(request['freeze']); check_hashes(freeze['source_hashes']); check_hashes(freeze['reference_hashes'])
    output_root = validate_output_root(Path(request['freeze']).parent)
    require(str(Path(__file__).resolve().relative_to(ROOT)) in freeze['source_hashes'],
            'This reproduction utility is not bound by the freeze; start a fresh output root')
    for key in ('output', 'tokens', 'metrics', 'comparison'):
        require(output_root in Path(request['job'][key]).resolve().parents, 'Requested output leaves its reproduction root')
    require(freeze['settings']['expected_version'] == args.expected_version, 'Requested board version differs from frozen batch')
    job = request['job']; sys.path.insert(0, str(ROOT / 'models/bigcodec'))
    import bigcodec_run_from_bin as runner
    original = runner.configure_hardware_runtime
    def checked_configure(**kwargs):
        # Called after runner acquires its flock, before any engine/reset/upload.
        fd = os.open('/dev/' + kwargs['dev'] + '_user', os.O_RDONLY)
        try: version = struct.unpack('<I', os.pread(fd, 4, 0x02000000))[0]
        finally: os.close(fd)
        require(version == args.expected_version, f'Board version changed: {version:#x}')
        return original(**kwargs)
    runner.configure_hardware_runtime = checked_configure
    sys.argv = ['bigcodec_run_from_bin.py', '--bin', job['bin'], '--input', job['input'], '--output', job['output'],
        '--tokens', job['tokens'], '--report', job['metrics'], '--cpu-core', '6', '--timeout', str(args.timeout)]
    runner.main()

class Batch:
    def __init__(self, args, jobs, freeze):
        self.args, self.jobs, self.freeze = args, jobs, freeze
        self.lock = threading.Lock(); self.stop = threading.Event()
        self.path = args.output_root / 'hardware_runs.json'
        self.state = read(self.path) if self.path.exists() else dict(format='bigcodec-accuracy-hardware-runs-v1', runs={})
        require(isinstance(self.state.get('runs'), dict), 'Unexpected prior batch manifest')
        for key, job in jobs.items():
            prior = self.state['runs'].get(key)
            if prior:
                for field in ('case', 'recurrent_precision', 'lstm_math_scope', 'input_sha256', 'bin', 'output', 'memory_layout', 'center_quantizer_scores', 'compensated_codebook'):
                    require(prior[field] == job[field], f'Prior job changed: {key}/{field}')
                for field, hash_field in [('output', 'output_sha256'), ('tokens', 'tokens_sha256'), ('metrics', 'metrics_sha256'), ('comparison', 'comparison_sha256')]:
                    if hash_field in prior: require(sha(prior[field]) == prior[hash_field], f'Completed artifact changed: {key}/{field}')
                job.update(prior)
            self.state['runs'][key] = job
    def check(self):
        require(not self.stop.is_set(), 'Batch stopped'); check_hashes(self.freeze['source_hashes']); check_hashes(self.freeze['reference_hashes'])
    def update(self, key, **fields):
        with self.lock:
            self.jobs[key].update(fields); self.state['updated_utc'] = now(); atomic_json(self.path, self.state)
        print(now(), key, json.dumps(fields), flush=True)
    def compile_precision(self, precision, core, events):
        os.sched_setaffinity(0, {core})
        try:
            for case in self.args.cases:
                key = case + '_' + precision; self.check(); job = self.jobs[key]
                if not Path(job['bin']).exists():
                    require('compile' not in job, 'Previously recorded bin is missing; restore it or choose another output root')
                    self.update(key, status='compiling', compile_started_utc=now())
                    run_command(compile_command(job, core), job['compile_log'])
                self.check()
                if 'compile_log_sha256' in job: require(sha(job['compile_log']) == job['compile_log_sha256'], 'Prior compile log changed')
                record = compile_record(job)
                self.update(key, status='compiled', compile=record, compile_log_sha256=sha(job['compile_log']), compile_finished_utc=now())
                events[key].set()
        except BaseException as error:
            self.stop.set(); self.update(key, status='compile_failed', error=str(error))
            for event in events.values(): event.set()
            raise
    def consume(self, key):
        self.check(); job = self.jobs[key]; metrics_path = Path(job['metrics'])
        if not metrics_path.exists():
            require(not Path(job['output']).exists() and not Path(job['tokens']).exists(),
                    'Partial hardware outputs exist without metrics; preserve/move them before retrying')
            self.update(key, status='running', run_started_utc=now())
            request = Path(job['bin']).parent / 'run_request.json'
            atomic_json(request, dict(freeze=str(self.args.output_root / 'batch_freeze.json'), job=job))
            run_command([PYTHON, str(Path(__file__).resolve()), '--execute', '--run-job', str(request),
                '--expected-version', hex(self.args.expected_version), '--timeout', str(self.args.timeout)], job['run_log'])
        self.check(); measured = validate_metrics(job, self.args.expected_version)
        if not Path(job['comparison']).exists(): run_command(comparison_command(job), job['comparison_log'])
        comparison = validate_comparison(job); old = job['old_baseline']
        self.check(); self.update(key, status='complete', finished_utc=now(),
            audio_processing_s=measured['audio_processing_s'], rtf=measured['audio_rtf'], duration_s=measured['audio_duration_s'],
            waveform_relative_l2=comparison['relative_l2'], tokens_equal=comparison['tokens']['equal'], tokens_total=comparison['tokens']['count'],
            old_baseline_rtf=old['rtf'], old_baseline_waveform_relative_l2=old['waveform_relative_l2'],
            waveform_error_change=comparison['relative_l2']-old['waveform_relative_l2'], rtf_change=measured['audio_rtf']-old['rtf'],
            output_sha256=sha(job['output']), tokens_sha256=sha(job['tokens']), metrics_sha256=sha(job['metrics']), comparison_sha256=sha(job['comparison']))

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(); mode.add_argument('--execute', action='store_true'); mode.add_argument('--compile-only', action='store_true')
    parser.add_argument('--cases', nargs='+', choices=CASES, default=CASES)
    parser.add_argument('--precisions', nargs='+', choices=['bf16', 'if8'], default=['bf16', 'if8'])
    parser.add_argument('--lstm-math-scope', choices=['both', 'decoder'], default='decoder')
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUT,
                        help='Fresh ignored results directory; use the same path to resume your own frozen run')
    parser.add_argument('--compensated-codebook', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--center-quantizer-scores', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--memory-layout', choices=['legacy', 'extended', 'large-program'], default='large-program')
    parser.add_argument('--expected-version', type=lambda value: int(value, 0), default=0x40519e0a,
                        help='Expected build; the validation profile remains Italy, AXI 256 and 333.25 MHz')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--run-job', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv); args.output_root = args.output_root.resolve()
    if len(set(args.cases)) != len(args.cases) or len(set(args.precisions)) != len(args.precisions): parser.error('Duplicate cases/precisions')
    if not math.isfinite(args.timeout) or args.timeout <= 0: parser.error('--timeout must be positive and finite')
    if not 0 <= args.expected_version <= 0xffffffff: parser.error('--expected-version must fit 32 bits')
    try: validate_output_root(args.output_root)
    except RuntimeError as error: parser.error(str(error))
    return args

def main():
    args = parse_args()
    if args.run_job: return guarded_runner(args)
    rows, references = load_references()
    jobs = {case+'_'+precision: make_job(args, rows[case], precision) for case in args.cases for precision in args.precisions}
    if not (args.execute or args.compile_only):
        print(json.dumps(dict(mode='plan_only_no_compilation_or_device_access', output_root=str(args.output_root),
            work_root=str(work_root(args)), python=PYTHON, lstm_math_scope=args.lstm_math_scope,
            execute_command=[PYTHON, str(Path(__file__).resolve()), '--execute', '--output-root', str(args.output_root),
                             '--cases', *args.cases, '--precisions', *args.precisions, '--lstm-math-scope', args.lstm_math_scope,
                             '--memory-layout', args.memory_layout, '--expected-version', hex(args.expected_version),
                             '--timeout', str(args.timeout),
                             '--center-quantizer-scores' if args.center_quantizer_scores else '--no-center-quantizer-scores',
                             '--compensated-codebook' if args.compensated_codebook else '--no-compensated-codebook'],
            report_preview_command=[PYTHON, str(HERE / 'build_report.py'), '--output-root', str(args.output_root)],
            jobs=[dict(key=key,
            compile_command=compile_command(job, 8+(['bf16','if8'].index(job['recurrent_precision']))),
            output=job['output'], compare_command=comparison_command(job), old_baseline_rtf=job['old_baseline']['rtf'],
            old_baseline_waveform_relative_l2=job['old_baseline']['waveform_relative_l2']) for key,job in jobs.items()]), indent=2)); return
    require({8,9}.issubset(os.sched_getaffinity(0)), 'Compiler cores 8/9 unavailable')
    if args.execute: require({6,10}.issubset(os.sched_getaffinity(0)), 'Hardware/compare cores 6/10 unavailable')
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock_path = work_root(args) / 'batch.lock'; lock_path.parent.mkdir(parents=True, exist_ok=True)
    try: batch_lock = lock_path.open('a')
    except PermissionError: batch_lock = lock_path.open('r')
    try: fcntl.flock(batch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('Another batch owns this output root')
    freeze = dict(format='bigcodec-accuracy-batch-freeze-v1', settings=dict(lstm_math_scope=args.lstm_math_scope,
        lstm_cell_precision='compensated', lstm_tanh_precision='compensated', lstm_fused_gates=True,
        convolution_precision='bf16', memory_layout=args.memory_layout, center_quantizer_scores=args.center_quantizer_scores, compensated_codebook=args.compensated_codebook, expected_version=args.expected_version), source_hashes=source_hashes(), reference_hashes=references)
    freeze_path = args.output_root / 'batch_freeze.json'
    if freeze_path.exists(): require(read(freeze_path) == freeze, 'Source/settings/references changed since batch start; choose a separate --output-root')
    else: atomic_json(freeze_path, freeze)
    validate_destination_paths(jobs, {**freeze['source_hashes'], **freeze['reference_hashes']})
    batch = Batch(args, jobs, freeze); events = {key: threading.Event() for key in jobs}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(batch.compile_precision, precision, 8+(['bf16','if8'].index(precision)), events) for precision in args.precisions]
        try:
            if args.execute:
                for key in jobs:
                    events[key].wait(); batch.consume(key)
            for future in futures: future.result()
        except BaseException as error:
            batch.stop.set()
            if 'key' in locals(): batch.update(key, status='failed', failed_phase=jobs[key].get('status'), error=str(error))
            raise
    print(json.dumps(dict(status='complete' if args.execute else 'compiled_only', runs=len(jobs), manifest=str(batch.path))), flush=True)
if __name__ == '__main__': main()
