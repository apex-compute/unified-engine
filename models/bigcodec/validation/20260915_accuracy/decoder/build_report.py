#!/usr/bin/env python3
"""Preview a fresh reproduction; --write publishes only its complete 16-run report.

Reads saved WAVs/NPZs and immutable CPU timing records. No model inference, compiler,
hardware access, source edits, or batch-manifest mutations occur in this script.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, io, json, math, os, sys, tempfile
sys.dont_write_bytecode = True
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import soundfile as sf
import run_validation as batch

ROOT, HERE, OLD = batch.ROOT, batch.HERE, batch.OLD
sys.path.insert(0, str(ROOT / 'models/bigcodec'))
from bigcodec_common import load_tokens
EXPECTED = {case+'_'+precision for case in batch.CASES for precision in ('bf16','if8')}
CSV_FIELDS = ['case','precision','duration_s','samples','cpu_processing_s','cpu_rtf',
 'before_waveform_relative_l2','after_waveform_relative_l2','waveform_relative_l2_change','waveform_result',
 'before_tokens_equal','after_tokens_equal','tokens_total','tokens_equal_change',
 'before_processing_s','after_processing_s','before_rtf','after_rtf','rtf_change',
 'before_fpga_execution_s','after_fpga_execution_s','before_total_elapsed_s','after_total_elapsed_s',
 'before_parameter_bytes','after_parameter_bytes','before_program_bytes','after_program_bytes',
 'before_instructions','after_instructions','before_resident_bytes','after_resident_bytes',
 'after_bin_bytes','after_bin_sha256','after_scratch_bytes','memory_layout','lstm_math_scope',
 'center_quantizer_scores','compensated_codebook']

def digest(path): return batch.sha(path)
def require(ok, reason): batch.require(ok, reason)
def read(path): return batch.read(path)
def close(actual, expected, name):
    require(actual is not None and expected is not None and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-13), f'Metric mismatch: {name}')
def entry(path):
    path=Path(path)
    try: name=str(path.relative_to(ROOT))
    except ValueError: name=str(path)
    return dict(path=name,sha256=digest(path),bytes=path.stat().st_size)
def load_wave(path, count):
    values, rate=sf.read(path,dtype='float64',always_2d=True)
    require(rate==16000 and values.shape==(count,1) and bool(np.isfinite(values).all()), f'Invalid WAV samples/rate/values: {path}')
    return values

def exact_comparison(reference_wave, reference_tokens, reference_metadata, actual_wave_path, actual_tokens_path, comparison, row):
    actual=load_wave(actual_wave_path,row['input_samples']);error=actual-reference_wave
    reference_energy=float(np.sum(reference_wave**2));error_energy=float(np.sum(error**2))
    require(reference_energy>0 and math.isfinite(reference_energy+error_energy), 'Invalid waveform energy')
    relative=math.sqrt(error_energy/reference_energy)
    close(relative,comparison['relative_l2'],'exact WAV relative L2')
    close(math.sqrt(reference_energy/reference_wave.size),comparison['reference_rms'],'CPU reference RMS')
    close(math.sqrt(error_energy/reference_wave.size),comparison['rmse'],'waveform RMSE')
    tokens, metadata=load_tokens(actual_tokens_path)
    require(metadata==reference_metadata and tokens.shape==reference_tokens.shape, 'Token metadata/shape differs from frozen CPU reference')
    equal=int(np.sum(tokens==reference_tokens))
    require(equal==comparison['tokens']['equal'] and tokens.size==comparison['tokens']['count']==row['tokens'], 'Saved token agreement differs')
    return dict(reference_energy=reference_energy,error_energy=error_energy,relative_l2=relative,
                tokens_equal=equal,tokens_total=int(tokens.size),finite=True)

def validate_execution(metrics,row,*,legacy=False,expected_version=0x40519e0a):
    for field in ('model_upload_writes','input_upload_writes','program_kicks','halts','output_reads'):
        require(metrics.get(field)==1, f'Execution contract: expected one {field}')
    require(metrics.get('cpu_neural_ops')==0 and metrics.get('finite') is True and metrics.get('waveform_padding_finite') is True,
            'Nonfinite output or CPU neural fallback')
    require(metrics.get('waveform_padding_nonzero')==0, 'Nonzero final padding')
    require(metrics.get('source_samples')==metrics.get('output_samples')==row['input_samples'], 'Input/output sample count differs')
    require(metrics.get('compiled_samples')==row['padded_samples'] and metrics.get('tokens')==row['tokens'], 'Compiled dimensions differ')
    require(metrics.get('checkpoint_sha256')==batch.CHECKPOINT and metrics.get('input_sha256')==row['input_sha256'], 'Source/model provenance differs')
    version=0x40519e0a if legacy else expected_version
    require(metrics.get('hardware_version')==f'0x{version:08x}' and metrics.get('axi_data_width_bits')==256, 'Hardware version/width differs')
    require(metrics.get('backend')=='hardware' and metrics.get('execution_scope')=='whole utterance', 'Wrong runtime backend/scope')
    duration=row['input_samples']/16000
    close(metrics['audio_duration_s'],duration,'audio duration')
    close(metrics['audio_rtf'],metrics['audio_processing_s']/duration,'RTF')
    if not legacy:
        require(0<metrics['dram_required_bytes']<=metrics['dram_addressable_bytes']<=2**31,'DRAM capacity/ranges invalid')
    clock=metrics['detected_clock_ns'];close(clock,1e9/333250000,'native clock')
    native=metrics.get('fpga_execution_s')
    if native is not None:
        close(native,metrics['fpga_cycles']*clock/1e9,'native execution time')
    return native

def pooled(rows):
    if not rows:return None
    duration=sum(r['duration_s'] for r in rows);processing=sum(r['processing_s'] for r in rows)
    reference=sum(r['reference_energy'] for r in rows);error=sum(r['error_energy'] for r in rows)
    tokens=sum(r['tokens_total'] for r in rows);equal=sum(r['tokens_equal'] for r in rows)
    native=[r['fpga_execution_s'] for r in rows]
    return dict(cases=len(rows),duration_s=duration,processing_s=processing,processing_rtf=processing/duration,
        reference_energy=reference,error_energy=error,waveform_relative_l2=math.sqrt(error/reference),
        tokens_equal=equal,tokens_total=tokens,token_match_fraction=equal/tokens,
        fpga_execution_s=sum(native) if all(v is not None for v in native) else None,
        native_timing_complete=all(v is not None for v in native),
        total_elapsed_s=sum(r['total_elapsed_s'] for r in rows),
        total_elapsed_rtf=sum(r['total_elapsed_s'] for r in rows)/duration,
        bin_ranges={field:dict(min=min(r[field] for r in rows),max=max(r[field] for r in rows))
          for field in ('parameter_bytes','program_bytes','instructions','resident_bytes')})

def result_label(before, after):
    if math.isclose(before,after,rel_tol=1e-12,abs_tol=1e-15):return 'unchanged'
    return 'improved' if after<before else 'regressed'

def run_values(metrics, exact):
    return dict(**exact,duration_s=metrics['audio_duration_s'],processing_s=metrics['audio_processing_s'],
        rtf=metrics['audio_rtf'],fpga_execution_s=metrics.get('fpga_execution_s'),
        total_elapsed_s=metrics['total_elapsed_s'],parameter_bytes=metrics['parameter_bytes'],
        program_bytes=metrics['program_bytes'],instructions=metrics['instructions'],resident_bytes=metrics['model_upload_bytes'])

def build(output_root,*,final=False):
    output_root=batch.validate_output_root(output_root)
    manifest_path=output_root/'hardware_runs.json';freeze_path=output_root/'batch_freeze.json'
    manifest_bytes=manifest_path.read_bytes();manifest=json.loads(manifest_bytes);freeze=read(freeze_path)
    manifest_digest=hashlib.sha256(manifest_bytes).hexdigest()
    require(manifest.get('format')=='bigcodec-accuracy-hardware-runs-v1' and isinstance(manifest.get('runs'),dict),'Unknown batch format')
    jobs=manifest['runs'];require(set(jobs).issubset(EXPECTED),'Unexpected case/precision in batch')
    complete={key for key,job in jobs.items() if job.get('status')=='complete'}
    missing=sorted(EXPECTED-complete)
    if final:require(not missing,'Final report requires all 16 completed runs; pending: '+', '.join(missing))
    # Read-only: a changed snapshot or source rejects the report instead of altering it.
    require(str(Path(batch.__file__).resolve().relative_to(ROOT)) in freeze['source_hashes'],
            'This reproduction utility is not bound by the freeze; the measured batch cannot be resumed here')
    batch.check_hashes(freeze['source_hashes']);batch.check_hashes(freeze['reference_hashes'])
    reference_rows,reference_hashes=batch.load_references()
    require(reference_hashes==freeze['reference_hashes'],'Frozen reference inventory differs')
    cpu_manifest_path=OLD/'cpu_reference_manifest.json';cpu_manifest=read(cpu_manifest_path)
    evidence=dict(batch_manifest=dict(path=str(manifest_path),sha256=manifest_digest,bytes=len(manifest_bytes)),batch_freeze=entry(freeze_path),cpu_manifest=entry(cpu_manifest_path),
                  generator=entry(Path(__file__).resolve()),run_evidence={})
    references={};baselines={};baseline_details=[];cpu_processing=0.;duration=0.
    for case in batch.CASES:
        row=reference_rows[case];cpu_wave=load_wave(ROOT/row['output'],row['input_samples'])
        cpu_tokens,cpu_meta=load_tokens(ROOT/row['tokens_file'])
        require(cpu_tokens.size==row['tokens'] and cpu_meta['source_sha256']==row['input_sha256'],'CPU token metadata differs')
        require(cpu_meta['checkpoint_sha256']==batch.CHECKPOINT,'CPU checkpoint differs')
        cpu_metrics=read(ROOT/row['metrics']);close(cpu_metrics['processing_s'],row['processing_s'],'frozen CPU processing')
        cpu_processing+=row['processing_s'];duration+=row['input_samples']/16000
        references[case]=(cpu_wave,cpu_tokens,cpu_meta)
        for precision in ('bf16','if8'):
            directory=OLD/('fpga_'+precision);metrics_path=directory/(case+'.metrics.json');comparison_path=directory/(case+'.comparison.json')
            metrics,comparison=read(metrics_path),read(comparison_path);batch.baseline(row,precision)
            validate_execution(metrics,row,legacy=True)
            exact=exact_comparison(cpu_wave,cpu_tokens,cpu_meta,directory/(case+'.wav'),directory/(case+'.tokens.npz'),comparison,row)
            values=run_values(metrics,exact);baselines[case+'_'+precision]=values
            baseline_details.append(dict(case=case,precision=precision,**values))
    cpu=dict(cases=8,duration_s=duration,processing_s=cpu_processing,processing_rtf=cpu_processing/duration,
             threads=cpu_manifest['torch_threads'],core=cpu_manifest['cpu_core'],precision='FP32',
             tokens_total=sum(row['tokens'] for row in reference_rows.values()))
    published=read(OLD/'summary.json');close(cpu['processing_rtf'],published['cpu_fp32']['processing_rtf'],'published CPU pooled RTF')
    baseline_all={p:pooled([v for v in baseline_details if v['precision']==p]) for p in ('bf16','if8')}
    for precision in ('bf16','if8'):
        close(baseline_all[precision]['processing_rtf'],published[precision]['processing_rtf'],'published baseline pooled RTF')
        close(baseline_all[precision]['waveform_relative_l2'],published[precision]['waveform_relative_l2'],'published baseline pooled waveform L2')
    rows=[]
    for case in batch.CASES:
        for precision in ('bf16','if8'):
            key=case+'_'+precision
            if key not in complete:continue
            job=jobs[key];ref=reference_rows[case]
            require(job['case']==case and job['recurrent_precision']==precision and job['reference']==ref,'Job reference/identity differs')
            for field in ('lstm_math_scope','memory_layout','center_quantizer_scores','compensated_codebook'):
                require(job[field]==freeze['settings'][field],f'Job differs from frozen setting: {field}')
            for field,hash_field in [('output','output_sha256'),('tokens','tokens_sha256'),('metrics','metrics_sha256'),('comparison','comparison_sha256')]:
                require(digest(job[field])==job[hash_field],f'Completed artifact changed: {key}/{field}')
            compiled=batch.compile_record(job);require(compiled==job['compile'],'Compile record differs from batch manifest')
            require(digest(job['compile_log'])==job['compile_log_sha256'],'Compile log hash differs')
            metrics=batch.validate_metrics(job,freeze['settings']['expected_version'])
            validate_execution(metrics,ref,expected_version=freeze['settings']['expected_version'])
            comparison=batch.validate_comparison(job);old_record=batch.baseline(ref,precision)
            require(old_record==job['old_baseline'],'Old baseline snapshot differs')
            exact=exact_comparison(*references[case],job['output'],job['tokens'],comparison,ref)
            before,after=baselines[key],run_values(metrics,exact)
            row=dict(case=case,precision=precision,environment=ref['environment'],cohort=ref['cohort'],
                nominal_snr_db=ref['nominal_snr_db'],duration_s=ref['input_samples']/16000,samples=ref['input_samples'],
                cpu_processing_s=ref['processing_s'],cpu_rtf=ref['processing_s']/(ref['input_samples']/16000),
                before=before,after=after,waveform_result=result_label(before['relative_l2'],after['relative_l2']),
                waveform_relative_l2_change=after['relative_l2']-before['relative_l2'],
                tokens_equal_change=after['tokens_equal']-before['tokens_equal'],rtf_change=after['rtf']-before['rtf'],
                memory_layout=job['memory_layout'],lstm_math_scope=job['lstm_math_scope'],
                center_quantizer_scores=job['center_quantizer_scores'],compensated_codebook=job['compensated_codebook'],
                after_bin_bytes=compiled['bin_bytes'],after_bin_sha256=compiled['bin_sha256'],after_scratch_bytes=compiled['scratch_bytes'])
            rows.append(row)
            evidence['run_evidence'][key]={field:entry(job[field]) for field in ('metrics','comparison','compile_log','output','tokens')}
            evidence['run_evidence'][key]['bin']=dict(path=job['bin'],sha256=compiled['bin_sha256'],bytes=compiled['bin_bytes'])
    summaries={}
    for precision in ('bf16','if8'):
        subset=[r for r in rows if r['precision']==precision]
        counts={name:sum(r['waveform_result']==name for r in subset) for name in ('improved','regressed','unchanged')}
        summaries[precision]=dict(cases=[r['case'] for r in subset],before=pooled([r['before'] for r in subset]),
            after=pooled([r['after'] for r in subset]),waveform_case_counts=counts,
            regressed_cases=[r['case'] for r in subset if r['waveform_result']=='regressed'],
            improved_cases=[r['case'] for r in subset if r['waveform_result']=='improved'])
    if final:require(digest(manifest_path)==manifest_digest,'Batch manifest changed while generating final report')
    return dict(format='bigcodec-accuracy-comparison-v1',status='complete' if not missing else 'partial_preview',
        generated_utc=datetime.now(timezone.utc).isoformat(),expected_cases=8,expected_runs=16,completed_runs=len(rows),
        pending=[dict(key=k,status=jobs.get(k,{}).get('status','not_started')) for k in missing],
        checkpoint_sha256=batch.CHECKPOINT,settings=freeze['settings'],cpu_fp32=cpu,baseline_all_cases=baseline_all,
        precision_summaries=summaries,runs=rows,all_completed_runs_pass_execution_contract=True,
        aggregation='RTF = sum processing_s / sum audio_s; waveform relative L2 = sqrt(sum error_energy / sum CPU reference_energy); tokens = sum matching IDs / sum token IDs.',
        waveform_scope='Official CPU codec reconstruction at original sample indices; no gain/delay/polarity fitting. This is not denoising quality or error against clean speech.',
        processing_scope='Preprocessing, one input upload, native execution, one output read, postprocessing and output WAV/token writes; excludes compilation, artifact load and resident program/parameter upload.',
        total_elapsed_scope='Runner total also includes artifact load/validation and resident upload; excludes compilation, Python startup and final report serialization.',
        execution='Whole utterance per bin; one resident upload, one input upload, one START, one HALT and one output bundle read; zero CPU neural operations. States reset between files.',
        hardware=dict(platform='Italy / Kintex UltraScale+ KU5P',version=f"0x{freeze['settings']['expected_version']:08x}",axi_data_width_bits=256,clock_mhz=333.25,visible_dram_bytes=2**31),
        evidence=evidence)

def csv_text(report):
    output=io.StringIO();writer=csv.DictWriter(output,fieldnames=CSV_FIELDS);writer.writeheader()
    for row in report['runs']:
        flat={k:row[k] for k in CSV_FIELDS if k in row}
        for side in ('before','after'):
            value=row[side]
            for key in ('processing_s','rtf','fpga_execution_s','total_elapsed_s','parameter_bytes','program_bytes','instructions','resident_bytes','tokens_equal'):
                flat[side+'_'+key]=value[key]
            flat[side+'_waveform_relative_l2']=value['relative_l2']
        flat['tokens_total']=row['after']['tokens_total'];writer.writerow(flat)
    return output.getvalue()

def readme(report):
    complete=report['status']=='complete';title='BigCodec noisy-audio accuracy comparison' if complete else 'BigCodec accuracy preview — incomplete'
    lines=['# '+title,'',
      'BigCodec reconstructs audio; it is not trained for background-noise suppression. These errors measure FPGA agreement with the CPU codec, not noise removal. See the [background-noise implementation check](../filtering_check/README.md).','',
      f"{report['completed_runs']}/16 runs completed across eight noisy test cases. Results compare FPGA reconstructions with the frozen official FP32 CPU reconstructions.",'']
    if not complete:lines+=['This preview is provisional. Final report files require all 16 runs.','']
    lines += ['| Recurrent weights | Processing RTF, before → after | Pooled waveform error, before → after | Token agreement, before → after | Cases improved / regressed / unchanged |',
              '| --- | ---: | ---: | ---: | ---: |']
    for p in ('bf16','if8'):
        group=report['precision_summaries'][p];old,new=group['before'],group['after']
        if new is None:lines.append(f'| {p.upper()} | pending | pending | pending | 0 / 0 / 0 |');continue
        count=group['waveform_case_counts']
        lines.append(f"| {p.upper()} | {old['processing_rtf']:.4f} → {new['processing_rtf']:.4f} | {old['waveform_relative_l2']*100:.3f}% → {new['waveform_relative_l2']*100:.3f}% | {old['tokens_equal']}/{old['tokens_total']} → {new['tokens_equal']}/{new['tokens_total']} | {count['improved']} / {count['regressed']} / {count['unchanged']} |")
    cpu=report['cpu_fp32'];scope=report['settings']['lstm_math_scope']
    scope_label='both encoder and decoder LSTM stacks' if scope=='both' else 'the '+scope+' LSTM stack'
    output_root=Path(report['evidence']['batch_manifest']['path']).parent
    attribution_link=os.path.relpath(OLD/'README.md',output_root)
    bus_link=os.path.relpath(ROOT/'models/bigcodec/validation/20260915_accuracy/bus_error_decomposition.md',output_root)
    lines += ['',f"Frozen FP32 CPU processing RTF with one thread on core {cpu['core']}: {cpu['processing_rtf']:.5f} ({cpu['processing_s']:.3f} s / {cpu['duration_s']:.5f} s audio). RTF 1 means real time; lower is faster. Each before/after aggregate uses the same cases. Waveform pooling sums error and reference energies.",'',
      f"Candidate: BF16 convolutions and activations; the table distinguishes BF16 versus IF8 recurrent weights. Compensated cell state/tanh and fused gates apply to {scope_label}. Centered scores: {report['settings']['center_quantizer_scores']}; compensated codebook: {report['settings']['compensated_codebook']}.",'',
      '| Case | Weights | Waveform error, before → after | Tokens equal, before → after | Processing seconds, before → after | RTF, before → after |',
      '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in report['runs']:
        old,new=row['before'],row['after']
        lines.append(f"| {row['case']} | {row['precision'].upper()} | {100*old['relative_l2']:.3f}% → {100*new['relative_l2']:.3f}% | {old['tokens_equal']} → {new['tokens_equal']} / {new['tokens_total']} | {old['processing_s']:.3f} → {new['processing_s']:.3f} | {old['rtf']:.4f} → {new['rtf']:.4f} |")
    lines += ['', 'Each complete WAV uses one program/parameter upload, one input upload, one START, one HALT and one output bundle read. Completed runs have finite output, zero padding and zero CPU neural operations. The audio-processing RTF excludes compilation, artifact loading and the resident upload; JSON/CSV also include total runner latency.', '',
      f"Platform: Italy, Kintex UltraScale+ KU5P; build {report['hardware']['version']}, AXI 256, 333.25 MHz, 2 GiB visible DRAM. These files contain noisy speech, but BigCodec is an audio codec. Waveform error measures agreement with CPU reconstruction, not noise removal or perceptual quality.", '',
      f'[Per-case metrics and deployment sizes](metrics_table.csv) · [Machine-readable summary and evidence hashes](metrics_summary.json) · [Frozen CPU references and source attribution]({attribution_link}) · [Bus pilot error decomposition]({bus_link})', '']
    if not complete:lines+=['Pending: '+', '.join(p['key']+' ('+p['status']+')' for p in report['pending'])+'.','']
    return '\n'.join(lines)

def write_report(output_root,report):
    require(report['status']=='complete' and report['completed_runs']==16,'Refusing incomplete final report')
    batch.validate_output_root(output_root)
    freeze=read(output_root/'batch_freeze.json')
    require(digest(output_root/'hardware_runs.json')==report['evidence']['batch_manifest']['sha256'],'Batch changed before final write')
    require(digest(output_root/'batch_freeze.json')==report['evidence']['batch_freeze']['sha256'],'Freeze changed before final write')
    protected=[ROOT/p for p in {**freeze['reference_hashes'],**freeze['source_hashes']}]
    protected += [output_root/'hardware_runs.json',output_root/'batch_freeze.json']
    protected += [Path(v['path']) if Path(v['path']).is_absolute() else ROOT/v['path'] for group in report['evidence']['run_evidence'].values() for v in group.values()]
    contents={'metrics_summary.json':json.dumps(report,indent=2,allow_nan=False)+'\n','metrics_table.csv':csv_text(report),'README.md':readme(report)}
    for name in contents:
        path=output_root/name
        for source in protected:
            require(path.resolve()!=source.resolve() and not(path.exists() and source.exists() and path.samefile(source)),'Report output aliases evidence')
    for name,content in contents.items():
        path=output_root/name
        fd,temporary_name=tempfile.mkstemp(prefix=name+'.',suffix='.tmp',dir=output_root);os.close(fd)
        temporary=Path(temporary_name)
        try:temporary.write_text(content);temporary.replace(path)
        finally:temporary.unlink(missing_ok=True)
    return {name:digest(output_root/name) for name in contents}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-root',type=Path,default=batch.DEFAULT_OUT)
    parser.add_argument('--write',action='store_true');parser.add_argument('--cpu-core',type=int);parser.add_argument('--json',action='store_true',help='Print machine-readable preview')
    args=parser.parse_args();args.output_root=batch.validate_output_root(args.output_root)
    if args.cpu_core is not None:
        require(args.cpu_core in os.sched_getaffinity(0),'Requested report CPU core is unavailable');os.sched_setaffinity(0,{args.cpu_core})
    if not (args.output_root/'hardware_runs.json').exists():
        require(not args.write,'No reproduction batch exists; run run_validation.py with --execute first')
        print(json.dumps(dict(status='preview_no_reproduction_runs',output_root=str(args.output_root),
            explanation='No files were written. Plan with run_validation.py, then pass its output_root here to preview results.'),indent=2));return
    report=build(args.output_root.resolve(),final=args.write)
    if args.write:print(json.dumps(dict(status='written',files=write_report(args.output_root.resolve(),report)),indent=2))
    else:print(json.dumps(report,indent=2,allow_nan=False) if args.json else readme(report))
if __name__=='__main__':main()
