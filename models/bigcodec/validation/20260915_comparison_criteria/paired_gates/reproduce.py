#!/usr/bin/env python3
"""Offline recheck of packaged paired-gate evidence; no hardware or ignored files.

Default checks frozen files, waveform decomposition, tokens, execution receipts,
and the native sigmoid tensor. --quality adds optional PESQ/STOI/spectral metrics.
--lstm-tail regenerates the official fixed-token reference from the checkpoint
and evaluates the packaged native LSTM outputs through its FP32 decoder tail.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,os,sys,tempfile
from pathlib import Path
import numpy as np
import soundfile as sf

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'models/bigcodec/bigcodec_common.py').is_file())
sys.path.insert(0,str(ROOT/'models/bigcodec'))
from bigcodec_common import load_tokens,sha256_file,CHECKPOINT_SHA256
from bigcodec_compare import compare_audio,compare_tokens


def require(condition,message):
 if not condition:raise ValueError(message)

def read_json(path):return json.loads(Path(path).read_text())
def raw_metric(actual,reference):
 d=np.asarray(actual,dtype=np.float64)-np.asarray(reference,dtype=np.float64);r=np.asarray(reference,dtype=np.float64)
 return dict(relative_l2=float(np.linalg.norm(d)/np.linalg.norm(r)),max_abs=float(np.max(np.abs(d))),rmse=float(np.sqrt(np.mean(d*d))))

def build(*,quality=False,lstm_tail=False,checkpoint=None):
 manifest=read_json(HERE/'manifest.json');paths={}
 for name,spec in manifest['inputs'].items():
  p=(ROOT/spec['path']).resolve();require(sha256_file(p)==spec['sha256'],f'Changed {name}: {p}');paths[name]=p
 snapshots=read_json(paths['source_manifest'])
 for row in snapshots['copies']:
  require(sha256_file(ROOT/row['packaged_path'])==row['sha256'],f"Changed packaged artifact: {row['packaged_path']}")
 report=dict(format='bigcodec-paired-gates-recheck-v1',checkpoint_sha256=CHECKPOINT_SHA256,
  scope='Separate 3.956-second clip; no gain, delay or polarity fitting. Full-model reconstruction agreement, not denoising quality.',
  input_inventory=manifest['inputs'],variants={},primitive={},all_packaged_source_and_evidence_hashes_match=True,
  conclusions={'promoted':False,'encoder_paired_gate_candidate':'Rejected: offline official-tail token agreement 309/317 to 307/317'},
  definitions={'raw_error':'L2 against official FP32 end-to-end reconstruction at original 48 kHz sample indices',
   'token_error':'FP32 decoding of the same recorded FPGA tokens versus official FP32 end-to-end reconstruction',
   'decoder_error':'FPGA waveform versus official FP32 decoding of those same recorded tokens',
   'rtf':'Processing seconds/audio seconds; excludes model loading/upload. RTF <= 1 is real time.',
   'metrics':'PESQ/STOI/spectral comparisons are complementary; no equivalence to a 10% waveform threshold is asserted.'})
 official=sf.read(paths['official_wave'],dtype='float64')[0];conditional=sf.read(paths['same_token_wave'],dtype='float64')[0]
 tokens,meta=load_tokens(paths['baseline_tokens']);other,othermeta=load_tokens(paths['candidate_tokens'])
 require(np.array_equal(tokens,other) and meta==othermeta,'Candidate tokens differ; saved conditional CPU waveform cannot be reused')
 cpu_receipt=read_json(paths['same_token_receipt'])
 require(cpu_receipt['checkpoint_sha256']==CHECKPOINT_SHA256 and cpu_receipt['audio']==meta and cpu_receipt['output_sha256']==sha256_file(paths['same_token_wave']),'Same-token CPU receipt mismatch')
 require(meta['source_sha256']==sha256_file(paths['input_wave']),'Input source hash differs')
 for name in ('baseline','candidate'):
  wave=paths[name+'_wave'];metrics=read_json(paths[name+'_metrics']);actual=sf.read(wave,dtype='float64')[0]
  require(actual.shape==official.shape==conditional.shape and np.isfinite(actual).all(),'Waveform dimensions/finite status invalid')
  require(metrics['output_sha256']==sha256_file(wave) and metrics['tokens_sha256']==sha256_file(paths[name+'_tokens']),'FPGA receipt artifact mismatch')
  require(metrics['hardware_version']=='0x90f1f464' and metrics['axi_data_width_bits']==256 and metrics['finite'] and metrics['waveform_padding_finite'] and metrics['waveform_padding_nonzero']==0,'Hardware/padding contract mismatch')
  require(all(metrics[k]==1 for k in ('model_upload_writes','input_upload_writes','program_kicks','halts','output_reads')) and metrics['cpu_neural_ops']==0,'One-shot execution contract mismatch')
  compared=compare_audio(paths['official_wave'],wave);token_result=compare_tokens(paths['official_tokens'],paths[name+'_tokens'],compared)
  t=conditional-official;d=actual-conditional;e=actual-official
  et,ed,ee=(float(np.dot(v,v)) for v in (t,d,e));cross=2*float(np.dot(t,d));refenergy=float(np.dot(official,official));condenergy=float(np.dot(conditional,conditional))
  require(math.isclose(ee,et+ed+cross,rel_tol=1e-12,abs_tol=1e-10),'Error-vector energy identity failed')
  row=dict(official_comparison=compared,tokens=token_result,processing_rtf=metrics['audio_rtf'],startup_inclusive_rtf=metrics['total_elapsed_s']/metrics['audio_duration_s'],
   conditional_decoder_relative_l2=math.sqrt(ed/condenergy),token_only_relative_l2=math.sqrt(et/refenergy),
   raw_error_below_10_percent=compared['relative_l2']<.1,conditional_decoder_below_10_percent=math.sqrt(ed/condenergy)<.1,
   decomposition=dict(reference_energy=refenergy,same_token_reference_energy=condenergy,token_error_energy=et,decoder_error_energy=ed,full_error_energy=ee,two_error_cross_term=cross,token_decoder_error_cosine=cross/(2*math.sqrt(et*ed)),scalar_identity_residual=ee-et-ed-cross),
   execution_counts={k:metrics[k] for k in ('model_upload_writes','input_upload_writes','program_kicks','halts','output_reads','cpu_neural_ops')})
  if quality:
   from bigcodec_audio_quality import compare_quality
   row['quality_vs_official']=compare_quality(paths['official_wave'],wave)
   row['quality_vs_same_tokens']=compare_quality(paths['same_token_wave'],wave)
  report['variants'][name]=row
 # No ignored model trace is needed to verify recovered primitive values.
 import torch
 torch.set_num_threads(1)
 native=torch.load(paths['primitive_tensor'],map_location='cpu',weights_only=True);high,low=native['actual'].float()
 x=torch.linspace(-12,12,3072).bfloat16().float();reference=x.sigmoid()
 require(hashlib.sha256(x.bfloat16().view(torch.uint8).numpy().tobytes()).hexdigest()==native['record']['input_sha256'],'Primitive source mismatch')
 require(native['record']['components_vs_native_rounding_model']['exact_mismatches']==0 and native['record']['guards_intact'] and native['record']['input_unchanged'],'Primitive native contract failed')
 report['primitive']=dict(recovered_vs_fp32=raw_metric((high+low).numpy(),reference.numpy()),high_only_vs_fp32=raw_metric(high.numpy(),reference.numpy()),inrange_recovered_vs_fp32=raw_metric((high+low)[x.abs()<=8].numpy(),reference[x.abs()<=8].numpy()),components_vs_native_model=native['record']['components_vs_native_rounding_model'])
 if lstm_tail:
  probe_path=ROOT/'models/bigcodec/validation/20260915_accuracy/lstm_probe/reproduce.py'
  spec=importlib.util.spec_from_file_location('paired_gates_frozen_reference',probe_path);probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
  decoder,source,reference,reference_wave,provenance=probe.references(checkpoint or probe.DEFAULT_CHECKPOINT)
  report['isolated_lstm']=dict(provenance=provenance,variants={})
  for name in ('baseline','candidate'):
   saved=torch.load(paths[name+'_tensor'],map_location='cpu',weights_only=True)
   require(saved['record']['input_sha256']==probe.proof.digest(source),'LSTM source differs from regenerated reference')
   report['isolated_lstm']['variants'][name]=probe.comparison(saved['actual'].float(),reference,reference_wave,decoder)
 report['status']='complete'
 for name,p in paths.items():require(sha256_file(p)==manifest['inputs'][name]['sha256'],f'Input changed during recheck: {name}')
 return report

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--quality',action='store_true');p.add_argument('--lstm-tail',action='store_true');p.add_argument('--checkpoint',type=Path);p.add_argument('--cpu-core',type=int);p.add_argument('--output',type=Path);args=p.parse_args()
 if args.cpu_core is not None:os.sched_setaffinity(0,{args.cpu_core})
 report=build(quality=args.quality,lstm_tail=args.lstm_tail,checkpoint=args.checkpoint)
 text=json.dumps(report,indent=2,allow_nan=False)+'\n'
 if args.output:
  protected=[ROOT/x['path'] for x in report['input_inventory'].values()]+[Path(__file__),HERE/'manifest.json']
  protected += [ROOT/x['packaged_path'] for x in read_json(HERE/'source_manifest.json')['copies']]
  require(not any(args.output.resolve()==x.resolve() or args.output.exists() and x.exists() and args.output.samefile(x) for x in protected),'Output must not overwrite frozen evidence or source')
  args.output.parent.mkdir(parents=True,exist_ok=True)
  with tempfile.NamedTemporaryFile('w',dir=args.output.parent,delete=False) as out:out.write(text);temporary=Path(out.name)
  os.replace(temporary,args.output)
 print(text,end='')
if __name__=='__main__':main()
