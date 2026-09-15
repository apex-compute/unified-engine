#!/usr/bin/env python3
"""Recompute both frozen codec-quality datasets using tracked WAVs only.

Install models/bigcodec/requirements-quality.txt, then run this script with
--output-dir pointing to an empty comparison directory. No inference, FPGA,
compiled binary, model weights, or external alignment is used.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
import soundfile as sf

HERE=Path(__file__).resolve().parent
MODEL=HERE.parents[2]
ROOT=MODEL.parents[1]
sys.path.insert(0,str(MODEL))
from bigcodec_accuracy_report import build_report
from bigcodec_audio_quality import compare_quality
from bigcodec_common import sha256_file

BASE=MODEL/'validation/20260915_encoder_accuracy'
METRICS=('pesq_wb','pesq_nb','stoi','estoi','mr_spectral_convergence',
         'mr_log_magnitude_mae_db','mr_log_spectral_distance_db')


def read(path):return json.loads(path.read_text())
def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def frozen(spec):
 path=(BASE/spec['path']).resolve()
 assert path.is_relative_to(ROOT) and sha256_file(path)==spec['sha256'],path
 return path


def datasets():
 matrix_path=BASE/'matrix_manifest.json'
 build_report(matrix_path)
 matrix=read(matrix_path)
 eight=[]
 for case in matrix['cases']:
  waves={'input':case['input'],'official_cpu':case['reference']['wave']}
  waves.update({name:value['wave'] for name,value in case['profiles'].items()})
  eight.append({'case':case['name'],'waves':waves})
 restored_path=BASE/'restored_short_manifest.json';short_path=BASE/'short_manifest.json'
 build_report(restored_path);build_report(short_path)
 restored,short=read(restored_path)['cases'][0],read(short_path)['cases'][0]
 waves={'input':restored['input'],'official_cpu':restored['reference']['wave'],
        'serial_baseline':restored['profiles']['baseline_restored']['wave'],
        'sorted_encoder':short['profiles']['sorted']['wave'],
        'matrix_encoder':restored['profiles']['matrix_production']['wave']}
 return {'eight_noisy':[eight,'Original noisy codec input; all eight cases'],
         'short_clip':[[{'case':'p232_007','waves':waves}],'Separate speech clip; excluded from eight-case aggregation']},[matrix_path,restored_path,short_path]


def pool(rows):
 duration=math.fsum(row['comparison']['waveform']['duration_s'] for row in rows)
 names=METRICS
 out={'cases':len(rows),'duration_s':duration,
      'original_samples':sum(row['comparison']['waveform']['samples'] for row in rows),
      'pooled_relative_l2':math.sqrt(math.fsum(r['original_error_energy'] for r in rows)/
                                      math.fsum(r['original_reference_energy'] for r in rows)),
      'mean_per_file':{key:math.fsum(r['comparison']['quality'][key] for r in rows)/len(rows) for key in names},
      'duration_weighted_mean':{key:math.fsum(r['comparison']['quality'][key]*r['comparison']['waveform']['duration_s'] for r in rows)/duration for key in names},
      'worst_file':{key:min(r['comparison']['quality'][key] for r in rows) for key in ('pesq_wb','pesq_nb','stoi','estoi')}}
 out['spectral_energy_pooled']=[{'fft_size':n,'spectral_convergence':math.sqrt(
    math.fsum(r['comparison']['quality']['spectral'][i]['spectral_error_energy'] for r in rows)/
    math.fsum(r['comparison']['quality']['spectral'][i]['spectral_reference_energy'] for r in rows))}
    for i,n in enumerate((512,1024,2048))]
 out['pooled_mr_spectral_convergence']=math.fsum(s['spectral_convergence'] for s in out['spectral_energy_pooled'])/3
 return out


def evaluate(label,cases,description):
 rows=[];inventory=[]
 for case in cases:
  paths={name:frozen(spec) for name,spec in case['waves'].items()}
  arrays={name:sf.read(path,dtype='float64')[0] for name,path in paths.items()}
  inventory.extend({'case':case['case'],'role':name,'path':str(path.relative_to(ROOT)),
                    'sha256':sha256_file(path)} for name,path in paths.items())
  for reference,roles in (('official_cpu',('official_cpu','serial_baseline','sorted_encoder','matrix_encoder')),
                         ('input',('input','official_cpu','serial_baseline','sorted_encoder','matrix_encoder'))):
   for role in roles:
    comparison=compare_quality(paths[reference],paths[role])
    assert set(comparison['quality']['metric_status'].values())=={'finite'},(case['case'],reference,role)
    x,y=arrays[reference],arrays[role]
    rows.append({'case':case['case'],'reference':reference,'actual':role,
       'original_error_energy':float(np.sum((x-y)**2)),
       'original_reference_energy':float(np.sum(x*x)),'comparison':comparison})
    print(f'{label}: {case["case"]} {reference} -> {role}',flush=True)
 for artifact in inventory:assert sha256_file(ROOT/artifact['path'])==artifact['sha256']
 aggregates={ref:{role:pool([r for r in rows if r['reference']==ref and r['actual']==role])
    for role in dict.fromkeys(r['actual'] for r in rows if r['reference']==ref)} for ref in ('official_cpu','input')}
 return {'format':'bigcodec-quality-dataset-v1','status':'complete','dataset':label,'description':description,
  'source_artifacts':inventory,'pairs':len(rows),'aggregates':aggregates,'rows':rows,
  'aggregation':'PESQ/STOI/log-spectrum metrics: per-file arithmetic mean and separate duration-weighted mean. Waveform L2: pooled original-rate energies. Spectral convergence: per-resolution pooled energies, then three-resolution arithmetic mean.',
  'scope':'Reference CPU measures implementation agreement; reference input measures codec fidelity. Noisy/codec references are not validated clean-reference intelligibility or denoising quality.',
  'threshold':'No metric is substituted for the requested waveform L2 threshold.',
  'generator_sha256':sha256_file(__file__)}


def markdown(results):
 lines=['# Complementary codec metrics','','Waveform L2 remains the numerical agreement measure. PESQ/STOI and magnitude spectra provide additional views; their scores do not certify waveform error below10%. No external gain, delay or polarity fitting is used.','']
 for label,result in results.items():
  lines += [f'## {label}', '',result['description']+'.','']
  for reference,profiles in result['aggregates'].items():
   lines += [f'Reference: **{reference}**.','','| Output | Pooled waveform L2 | Mean PESQ-WB | Mean PESQ-NB | Mean STOI | Pooled MR magnitude convergence |','| --- | ---: | ---: | ---: | ---: | ---: |']
   for name,v in profiles.items():
    m=v['mean_per_file'];lines.append(f"| {name} | {100*v['pooled_relative_l2']:.5f}% | {m['pesq_wb']:.5f} | {m['pesq_nb']:.5f} | {m['stoi']:.6f} | {100*v['pooled_mr_spectral_convergence']:.5f}% |")
   lines.append('')
 lines += ['Eight-case data are16kHz. The independent short clip is48kHz: raw waveform L2 stays at48kHz, while each signal is identically resampled to16kHz for the speech/spectral metrics. Short results are never pooled into the eight-case results.','','PESQ/STOI are arithmetic means across files, not percentages of words understood. PESQ retains its internal level/time processing; STOI retains silence removal, resampling and envelope normalization. Magnitude convergence ignores phase and has no equivalence to the10% waveform target. Detailed settings, versions, individual results and source hashes are in the JSON.','','The matrix psquare waveform error is37.0452%, while PESQ-WB is4.1834 and STOI is0.98585; these measures capture different properties. The CPU codec also changes the original noisy input substantially. Agreement with its output does not establish noise suppression.','','[Primary sources and interpretation](sources.md), [eight-case results](eight_noisy_results.json), [separate short results](short_clip_results.json), [input manifest](manifest.json).','','Reproduce without model weights, deployment bins or an FPGA:','','```bash','pip install -r models/bigcodec/requirements-quality.txt','OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\','  python models/bigcodec/validation/20260915_comparison_criteria/perceptual/generate_report.py \\','  --output-dir /tmp/bigcodec-quality-recomputed','```','']
 return '\n'.join(lines)


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output-dir',type=Path,required=True)
 parser.add_argument('--cpu-core',type=int)
 args=parser.parse_args()
 if args.cpu_core is not None:os.sched_setaffinity(0,{args.cpu_core})
 data,manifests=datasets();results={}
 args.output_dir.mkdir(parents=True,exist_ok=True)
 for label,(cases,description) in data.items():
  results[label]=evaluate(label,cases,description)
  dump(args.output_dir/(label+'_results.json'),results[label])
 manifest={'format':'bigcodec-quality-manifest-v1','datasets':{label:result['source_artifacts'] for label,result in results.items()},
   'source_manifests':[{'path':str(path.relative_to(ROOT)),'sha256':sha256_file(path)} for path in manifests],
   'generator_sha256':sha256_file(__file__),
   'analysis_source_sha256':results['eight_noisy']['rows'][0]['comparison']['source_sha256'],
   'dependencies':results['eight_noisy']['rows'][0]['comparison']['dependencies'],
   'results':{label:{'path':label+'_results.json','sha256':sha256_file(args.output_dir/(label+'_results.json'))} for label in results}}
 dump(args.output_dir/'manifest.json',manifest)
 (args.output_dir/'comparison.md').write_text(markdown(results))
 print(json.dumps({'status':'complete','datasets':{label:{'pairs':result['pairs'],'aggregates':result['aggregates']} for label,result in results.items()}},indent=2))


if __name__=='__main__':main()
