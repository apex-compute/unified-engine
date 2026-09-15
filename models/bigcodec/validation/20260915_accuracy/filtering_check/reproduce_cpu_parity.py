#!/usr/bin/env python3
"""Execute pinned upstream inference with only CUDA placements removed, then compare our CPU wrapper."""
from pathlib import Path
import argparse, ast, hashlib, json, math, os, sys, time, urllib.request
from datetime import datetime, timezone
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
sys.dont_write_bytecode=True
import numpy as np
import soundfile as sf
import torch
def repository_root(script):
 for parent in Path(script).resolve().parents:
  if (parent/'models/bigcodec/bigcodec_common.py').is_file(): return parent
 raise RuntimeError('Place this script inside the unified-engine repository')
ROOT=repository_root(__file__);MODEL=ROOT/'models/bigcodec'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--execute',action='store_true',help='Run the approximately two-minute CPU-only comparison')
parser.add_argument('--output-root',type=Path,default=MODEL/'bigcodec_bin/filtering_check_20260915'/('reproduce-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')))
args=parser.parse_args();HERE=args.output_root.resolve()
if not HERE.is_relative_to((MODEL/'bigcodec_bin').resolve()) or HERE==(MODEL/'bigcodec_bin').resolve(): parser.error('Outputs must remain in an ignored bigcodec_bin subdirectory')
UPSTREAM=HERE/'upstream'
INPUT=ROOT/'models/dpdfnet/validation/20260914_noisy20s/noisy/bus_noisy.wav'
BASE=MODEL/'validation/20260914_noisy20s';CHECKPOINT=MODEL/'bigcodec_bin/bigcodec.pt'
REV='09845ab1f5bc7a3d1589c16de820ef15bc2afefe'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def file(p):
 p=Path(p).resolve();return dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)
def metrics(reference,actual):
 r=np.asarray(reference,dtype=np.float64).reshape(-1);a=np.asarray(actual,dtype=np.float64).reshape(-1)
 assert r.shape==a.shape and np.isfinite(r).all() and np.isfinite(a).all()
 d=a-r;er=float(r@r);ed=float(d@d)
 return dict(samples=r.size,array_equal=bool(np.array_equal(r,a)),max_abs_error=float(np.max(abs(d))),relative_l2=math.sqrt(ed/er) if er else 0.0,error_energy=ed,reference_energy=er)
def save(): (HERE/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if not args.execute:
 print(json.dumps(dict(mode='preview_cpu_only_no_execution_or_writes',input=str(INPUT),checkpoint=str(CHECKPOINT),revision=REV,output_root=str(HERE),
  command=[sys.executable,str(Path(__file__).resolve()),'--execute','--output-root',str(HERE)]),indent=2));raise SystemExit(0)
if HERE.exists() and any(HERE.iterdir()): raise RuntimeError('Choose an empty output root; prior evidence is preserved')
HERE.mkdir(parents=True,exist_ok=True)
os.sched_setaffinity(0,{7});torch.set_num_threads(1);torch.set_num_interop_threads(1)
manifest=json.loads((MODEL/'upstream_manifest.json').read_text())
expected_sources={'inference.py':'f2158d28c30b9d00b61ea553f1860ef84988e7b1293dfff2fb205f79a3edf207',
 'README.md':'6c0ef81f6484e3ab2c7d5844f99bcfbf3be4b7e8b63103fe3cdbaaf838148451',
 **{'vq/'+name:digest for name,digest in manifest['code']['upstream_sha256'].items()}}
verified={}
for name,digest in expected_sources.items():
 cached=MODEL/'bigcodec_bin/upstream'/name
 url=f'https://raw.githubusercontent.com/Aria-K-Alethia/BigCodec/{REV}/{name}'
 data=cached.read_bytes() if cached.is_file() else urllib.request.urlopen(url,timeout=30).read()
 if hashlib.sha256(data).hexdigest()!=digest: raise RuntimeError('Pinned upstream source hash mismatch: '+name)
 target=UPSTREAM/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
 verified[name]=dict(url=url,sha256=digest,bytes=len(data),source='verified_local_cache' if cached.is_file() else 'pinned_https_download')
(HERE/'verified_source.json').write_text(json.dumps(verified,indent=2)+'\n')
assert manifest['code']['revision']==REV
assert sha(CHECKPOINT)==manifest['checkpoint']['sha256']=='1fba3806e87cc01c1a65bea22fa1becefbbf46881e4219593c4d9f3cf56206b9'
source_checks={}
for name,digest in manifest['code']['upstream_sha256'].items():
 assert sha(UPSTREAM/'vq'/name)==digest
 assert sha(MODEL/'bigcodec_vq'/name)==manifest['code']['vendored_sha256'][name]
 if name!='__init__.py':assert (UPSTREAM/'vq'/name).read_bytes()==(MODEL/'bigcodec_vq'/name).read_bytes()
 source_checks[name]=dict(upstream_sha256=digest,vendored_sha256=manifest['code']['vendored_sha256'][name],identical=name!='__init__.py')
source=(UPSTREAM/'inference.py').read_text();assert sha(UPSTREAM/'inference.py')==verified['inference.py']['sha256']
class CPUPlacement(ast.NodeTransformer):
 def __init__(self):self.lines=[]
 def visit_Call(self,node):
  self.generic_visit(node)
  if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda':
   assert not node.args and not node.keywords;self.lines.append(node.lineno);return node.func.value
  return node
transform=CPUPlacement();tree=transform.visit(ast.parse(source));ast.fix_missing_locations(tree);assert transform.lines==[28,31,42]
(HERE/'upstream_cpu_placement_only.py').write_text(ast.unparse(tree)+'\n')
report=dict(status='running',scope='One full 23.68225-second bus utterance; CPU core 7, one thread, FP32; no hardware.',upstream_revision=REV,
 upstream_inference_url=f'https://github.com/Aria-K-Alethia/BigCodec/blob/{REV}/inference.py#L24-L48',
 upstream_readme_url=f'https://github.com/Aria-K-Alethia/BigCodec/blob/{REV}/README.md#L20-L26',
 source_checks=source_checks,upstream_source_verification=verified,checkpoint=file(CHECKPOINT),input=file(INPUT),generator=file(Path(__file__)),
 placement_adaptation=dict(removed_cuda_call_lines=transform.lines,description='Only three .cuda() device placements removed from the parsed upstream AST; all neural, audio-loading, padding and WAV-writing statements executed unchanged.'),
 runtime=dict(torch=torch.__version__,numpy=np.__version__,soundfile=sf.__version__,cpu_affinity=sorted(os.sched_getaffinity(0)),torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads()))
input_dir=HERE/'input';input_dir.mkdir(exist_ok=True);input_link=input_dir/'bus_noisy.wav'
if input_link.exists():assert input_link.resolve()==INPUT.resolve()
else:input_link.symlink_to(INPUT)
output_dir=HERE/'literal_pcm16';output_dir.mkdir(exist_ok=True)
sys.path.insert(0,str(UPSTREAM));sys.argv=['inference.py','--input-dir',str(input_dir),'--output-dir',str(output_dir),'--ckpt',str(CHECKPOINT)]
namespace={'__name__':'__main__','__file__':str(UPSTREAM/'inference.py')};tick=time.perf_counter()
exec(compile(tree,str(UPSTREAM/'inference.py'),'exec'),namespace)
report['literal_upstream_wall_s']=time.perf_counter()-tick
literal=np.asarray(namespace['recon']);literal_tokens=namespace['vq_code'].reshape(-1).cpu().numpy();native_count=sf.info(INPUT).frames
np.save(HERE/'literal_upstream_full.npy',literal);np.save(HERE/'literal_upstream_tokens.npy',literal_tokens)
sf.write(HERE/'literal_upstream_cropped_float.wav',literal[:native_count],16000,subtype='FLOAT')
report['literal_output']=dict(full_samples=int(literal.size),source_samples=native_count,padding_samples=int(literal.size-native_count),tokens=int(literal_tokens.size),finite=bool(np.isfinite(literal).all()),wav_format=sf.info(output_dir/'bus_noisy.wav').subtype)
save();print('LITERAL_UPSTREAM_COMPLETE '+json.dumps(report['literal_output']),flush=True)
# Load our namespace and checkpoint separately to check the current wrapper implementation.
sys.path.insert(0,str(MODEL))
from bigcodec_common import load_models,read_audio,pad_audio,encode_audio,decode_tokens,restore_audio,save_tokens,load_tokens
report['wrapper_source']={name:file(MODEL/name) for name in ['bigcodec_common.py','bigcodec_run_cpu.py']}
tick=time.perf_counter();encoder,decoder=load_models();report['wrapper_model_load_s']=time.perf_counter()-tick
for upstream,current in [(namespace['encoder'],encoder),(namespace['decoder'],decoder)]:
 assert upstream.state_dict().keys()==current.state_dict().keys()
 assert all(torch.equal(v,current.state_dict()[key]) for key,v in upstream.state_dict().items())
report['all_encoder_decoder_state_tensors_equal']=True
audio,meta=read_audio(INPUT);padded=pad_audio(audio)
report['input_and_padding']=metrics(namespace['wav'].reshape(-1).cpu().numpy(),padded)
assert report['input_and_padding']['array_equal']
captured={}
hook=encoder.register_forward_hook(lambda module,inp,out:captured.update(features=out.detach()))
tick=time.perf_counter();tokens,quantized=encode_audio(encoder,decoder,audio);report['wrapper_encode_s']=time.perf_counter()-tick;hook.remove()
report['encoder_features']=metrics(namespace['vq_emb'].cpu().numpy(),captured['features'].cpu().numpy())
report['quantizer_direct_output']=metrics(namespace['vq_post_emb'].cpu().numpy(),quantized.cpu().numpy())
report['tokens']=dict(count=int(tokens.size),equal=int(np.count_nonzero(tokens==literal_tokens)),array_equal=bool(np.array_equal(tokens,literal_tokens)))
assert report['tokens']['array_equal']
with torch.inference_mode():from_tokens=decoder.vq2emb(torch.from_numpy(tokens.astype(np.int64)).reshape(1,-1,1)).transpose(1,2)
report['token_reembedding_vs_direct_quantizer']=metrics(quantized.cpu().numpy(),from_tokens.cpu().numpy())
tick=time.perf_counter();reconstruction=decode_tokens(decoder,tokens);report['wrapper_decode_s']=time.perf_counter()-tick
restored=restore_audio(reconstruction,meta);sf.write(HERE/'wrapper_float.wav',restored,16000,subtype='FLOAT');save_tokens(HERE/'wrapper.tokens.npz',tokens,meta)
np.save(HERE/'wrapper_full.npy',reconstruction)
report['wrapper_full_vs_literal_neural']=metrics(literal,reconstruction)
report['wrapper_cropped_vs_literal_neural']=metrics(literal[:native_count],restored)
report['wrapper_output']=dict(samples=int(restored.size),full_decoder_samples=int(reconstruction.size),wav_format=sf.info(HERE/'wrapper_float.wav').subtype,finite=bool(np.isfinite(restored).all()))
frozen_manifest=json.loads((BASE/'cpu_reference_manifest.json').read_text());frozen=next(r for r in frozen_manifest['cases'] if r['id']=='bus')
assert sha(ROOT/frozen['output'])==frozen['output_sha256'];assert sha(ROOT/frozen['tokens_file'])==frozen['tokens_sha256'];assert sha(INPUT)==frozen['input_sha256']
frozen_wave,rate=sf.read(ROOT/frozen['output'],dtype='float32');frozen_tokens,frozen_meta=load_tokens(ROOT/frozen['tokens_file'])
assert frozen_meta==meta
report['wrapper_vs_frozen_cpu_wave']=metrics(frozen_wave,restored)
report['wrapper_vs_frozen_cpu_tokens']=dict(equal=int(np.count_nonzero(tokens==frozen_tokens)),count=int(tokens.size),array_equal=bool(np.array_equal(tokens,frozen_tokens)))
literal_pcm,rate=sf.read(output_dir/'bus_noisy.wav',dtype='float32')
report['upstream_pcm16_quantization_only']=metrics(literal,literal_pcm)
report['literal_pcm16_cropped_vs_wrapper_float']=metrics(restored,literal_pcm[:native_count])
report['literal_vs_input']=metrics(audio,literal[:native_count])
report['wrapper_vs_input']=metrics(audio,restored)
report['padding_boundary_checks']=[dict(samples=n,padded_samples=int(pad_audio(np.zeros(n,dtype=np.float32)).size),matches_literal_formula=pad_audio(np.zeros(n,dtype=np.float32)).size==n+200-n%200) for n in [1,199,200,201,400,native_count]]
report['artifacts']={name:file(HERE/name) for name in ['literal_upstream_full.npy','literal_upstream_tokens.npy','literal_upstream_cropped_float.wav','literal_pcm16/bus_noisy.wav','wrapper_float.wav','wrapper_full.npy','wrapper.tokens.npz','upstream_cpu_placement_only.py']}
report['frozen_reference_artifacts']={name:file(ROOT/frozen[key]) for name,key in [('wave','output'),('tokens','tokens_file')]}
report['status']='complete';save();print('COMPLETE '+str(HERE/'results.json')+' sha256='+sha(HERE/'results.json'),flush=True)
print(json.dumps({k:report[k] for k in ['input_and_padding','encoder_features','quantizer_direct_output','tokens','token_reembedding_vs_direct_quantizer','wrapper_full_vs_literal_neural','wrapper_vs_frozen_cpu_wave','wrapper_vs_frozen_cpu_tokens','upstream_pcm16_quantization_only']},indent=2),flush=True)
