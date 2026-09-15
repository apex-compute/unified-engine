"""Verify repository waveform/token evidence and recompute the decomposition.

Requires only NumPy and soundfile. No model, Torch, device or repository imports.
Reads hashed validation files and prints JSON; no artifact is modified. Existing tracked reference artifacts are reused; ignored evaluation caches are never loaded.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import soundfile as sf

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
PACKAGE=HERE.relative_to(ROOT)
ALLOWED_DIRECTORIES=(PACKAGE,
    Path('models/bigcodec/validation/20260914_noisy20s/cpu'),
    Path('models/bigcodec/validation/20260915_accuracy/decoder/fpga_bf16'),
    Path('models/bigcodec/validation/20260915_accuracy/short_clip'))
ALLOWED_FILES={
    Path('models/bigcodec/validation/20260914_noisy20s/cpu_reference_manifest.json'),
    Path('models/bigcodec/validation/20260915_accuracy/decoder/conditional_error_decomposition.json')}
LONG_CASES={'bus','cafe','office','psquare','bus_low_snr','cafe_low_snr','office_low_snr','psquare_low_snr'}
ENERGIES=('reference_energy','same_token_cpu_energy','token_error_energy','decoder_error_energy','full_error_energy','two_error_cross_term')

def require(condition,message):
    if not condition:raise ValueError(message)

def local(name):
    relative=Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts,'Path must be repository-relative: '+name)
    require(relative in ALLOWED_FILES or any(relative.is_relative_to(base) for base in ALLOWED_DIRECTORIES),
            'Path is outside approved validation artifacts: '+name)
    path=(ROOT/relative).resolve()
    require(path.is_relative_to(ROOT),'Path escapes repository: '+name)
    resolved=path.relative_to(ROOT)
    require(resolved in ALLOWED_FILES or any(resolved.is_relative_to(base) for base in ALLOWED_DIRECTORIES),
            'Resolved path is outside approved validation artifacts: '+name)
    return path

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def verify(record):
    path=local(record['path'])
    require(digest(path)==record['sha256'],'SHA256 mismatch: '+record['path'])
    require(path.stat().st_size==record['bytes'],'Size mismatch: '+record['path'])
    return path

def read_json(record):return json.loads(verify(record).read_text())

def close(actual,expected,key):
    require(math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-10),'Value differs: '+key)

def read_wave(record,metadata):
    x,rate=sf.read(verify(record),dtype='float64')
    require(rate==metadata['source_rate'] and x.shape==(metadata['source_samples'],),'Waveform geometry differs')
    require(np.isfinite(x).all(),'Nonfinite waveform')
    return x

def read_tokens(record,metadata):
    with np.load(verify(record),allow_pickle=False) as archive:
        ids=archive['tokens'];actual=json.loads(str(archive['metadata'].item()))
    require(actual==metadata,'Token metadata differs')
    require(ids.ndim==1 and ids.dtype.kind in 'iu' and ids.size>0,'Invalid code IDs')
    require(ids.min()>=0 and ids.max()<8192,'Code ID outside codebook')
    require(metadata['sample_rate']==16000 and metadata['hop_length']==200,'Invalid model rate/hop')
    require(ids.size*metadata['hop_length']==metadata['padded_samples'],'Token count mismatch')
    require(metadata['padded_samples']==metadata['native_samples']+200-metadata['native_samples']%200,'Padding mismatch')
    return ids

def dot(a,b):return float(np.dot(a,b))

def measure(y,z,f):
    t=z-y;d=f-z;full=f-y
    ey,ez,et,ed,ef=(dot(v,v) for v in (y,z,t,d,full));cross=2*dot(t,d)
    require(min(ey,ez,et,ed)>0,'Unexpected zero-energy evidence')
    close(ef,et+ed+cross,'vector energy identity')
    ct=t-t.mean();cd=d-d.mean()
    return dict(full_fpga_vs_official_l2=math.sqrt(ef/ey),token_only_cpu_vs_official_l2=math.sqrt(et/ey),
        same_token_fpga_decoder_vs_cpu_l2=math.sqrt(ed/ez),decoder_error_relative_to_official_energy=math.sqrt(ed/ey),
        reference_energy=ey,same_token_cpu_energy=ez,token_error_energy=et,decoder_error_energy=ed,full_error_energy=ef,
        two_error_cross_term=cross,token_decoder_error_cosine=cross/(2*math.sqrt(et*ed)),
        token_decoder_error_pearson=dot(ct,cd)/math.sqrt(dot(ct,ct)*dot(cd,cd)),
        error_interaction='cancellation' if cross<0 else 'reinforcement' if cross>0 else 'orthogonal',
        component_energy_cancelled_fraction=max(0,-cross)/(et+ed),component_energy_reinforced_fraction=max(0,cross)/(et+ed),
        scalar_identity_residual=ef-et-ed-cross,vector_identity_max_abs_residual=float(np.abs(full-(t+d)).max()))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-core',type=int,help='Optional CPU affinity for this read-only audit')
    args=parser.parse_args()
    if args.cpu_core is not None:os.sched_setaffinity(0,{args.cpu_core})
    manifest=json.loads((HERE/'manifest.json').read_text())
    listed={entry['path'] for entry in manifest['files']}
    require(len(listed)==len(manifest['files']),'Duplicate manifest paths')
    for entry in manifest['files']:verify(entry)
    actual_files={str(path.relative_to(ROOT)) for path in HERE.rglob('*') if path.is_file() and path.name!='manifest.json' and '__pycache__' not in path.parts}
    local_listed={name for name in listed if Path(name).is_relative_to(PACKAGE)}
    require(actual_files==local_listed,'Unlisted or missing decomposition artifacts')
    report=json.loads((HERE/'results.json').read_text())
    required={record['path'] for row in report['cases'] for record in row['files'].values()}
    required.update(record['path'] for record in report['provenance'].values())
    require(required.issubset(listed),'A referenced frozen artifact is absent from the hash inventory')
    require(report['status']=='complete','Evidence is incomplete')
    rows=report['cases'];require(len(rows)==9 and {row['case'] for row in rows}==LONG_CASES|{'p232_007'},'Wrong case coverage')
    collected=[]
    for row in rows:
        case=row['case'];meta=row['metadata'];files=row['files']
        require(meta['checkpoint_sha256']==report['checkpoint_sha256'],'Token checkpoint differs')
        y,z,f=(read_wave(files[key],meta) for key in ('reference_wav','same_token_cpu_wav','fpga_wav'))
        official=read_tokens(files['official_tokens'],meta);actual=read_tokens(files['actual_tokens'],meta)
        equal=int(np.count_nonzero(official==actual))
        require(equal==row['tokens_matching_official'] and actual.size==row['tokens'],'Token equality differs')
        fpga=read_json(files['fpga_metrics']);same=read_json(files['cached_decode_record']);ref=read_json(files['reference_metrics'])
        require(fpga['output_sha256']==files['fpga_wav']['sha256'],'FPGA waveform binding differs')
        require(fpga['tokens_sha256']==files['actual_tokens']['sha256'],'FPGA token binding differs')
        require(fpga['input_sha256']==meta['source_sha256'],'FPGA source binding differs')
        require(ref['output_sha256']==files['reference_wav']['sha256'] and ref['tokens_sha256']==files['official_tokens']['sha256'],'Frozen CPU binding differs')
        require(fpga['checkpoint_sha256']==ref['checkpoint_sha256']==same['checkpoint_sha256']==report['checkpoint_sha256'],'Run checkpoint differs')
        for key in ('model_upload_writes','input_upload_writes','program_kicks','halts','output_reads'):
            require(fpga[key]==1,'Unexpected FPGA execution count: '+key)
        require(fpga['cpu_neural_ops']==0 and fpga['finite'],'Invalid backend/output')
        if row['decode_record_format']=='long_cached_decomposition':
            require(same['files']['actual_tokens']['sha256']==files['actual_tokens']['sha256'],'Cached decode token binding differs')
            require(same['files']['same_token_cpu_wav']['sha256']==files['same_token_cpu_wav']['sha256'],'Cached decode waveform binding differs')
        elif row['decode_record_format']=='short_cpu_run':
            require(same['tokens_sha256']==files['actual_tokens']['sha256'],'Short decode token binding differs')
            require(same['output_sha256']==files['same_token_cpu_wav']['sha256'],'Short decode waveform binding differs')
        else:raise ValueError('Unsupported CPU decode evidence')
        calculated=measure(y,z,f)
        for key,expected in row['metrics'].items():
            if isinstance(expected,(int,float)):close(calculated[key],expected,case+'/'+key)
            else:require(calculated[key]==expected,'Metric differs: '+case+'/'+key)
        for field,key in (('conditional_decoder_below_10pct','same_token_fpga_decoder_vs_cpu_l2'),
                          ('token_induced_below_10pct','token_only_cpu_vs_official_l2'),
                          ('end_to_end_below_10pct','full_fpga_vs_official_l2')):
            require(row[field]==(calculated[key]<.1),'Threshold claim differs: '+case+'/'+field)
        require(row['included_in_eight_file_pool']==(case in LONG_CASES),'Incorrect pooling membership')
        collected.append(dict(case=case,tokens_matching_official=equal,tokens=int(actual.size),metrics=calculated))
    long=[row for row in collected if row['case'] in LONG_CASES]
    sums={key:math.fsum(row['metrics'][key] for row in long) for key in ENERGIES}
    ey,ez,et,ed,ef,cross=(sums[key] for key in ENERGIES)
    pooled=dict(full_fpga_vs_official_l2=math.sqrt(ef/ey),token_only_cpu_vs_official_l2=math.sqrt(et/ey),
        same_token_fpga_decoder_vs_cpu_l2=math.sqrt(ed/ez),decoder_error_relative_to_official_energy=math.sqrt(ed/ey),
        token_decoder_error_cosine=cross/(2*math.sqrt(et*ed)),**sums)
    for key,value in pooled.items():close(value,report['eight_noisy_files_pooled'][key],'pooled/'+key)
    for field in report['eight_noisy_files_threshold_counts']:
        total=sum(row[field] for row in rows if row['case'] in LONG_CASES)
        require(total==report['eight_noisy_files_threshold_counts'][field],'Pooled threshold count differs')
    print(json.dumps(dict(status='PASS',verified_files=len(listed),new_decomposition_files=len(local_listed),existing_frozen_files=len(listed-local_listed),cases=9,pooled_cases=8,
        short_clip_excluded_from_pool=True,neural_or_hardware_execution=False,
        eight_noisy_files_pooled=pooled,results=collected),indent=2,allow_nan=False))

if __name__=='__main__':main()
