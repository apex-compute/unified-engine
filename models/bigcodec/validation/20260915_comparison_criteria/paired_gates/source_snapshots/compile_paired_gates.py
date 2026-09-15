"""Ignored whole-utterance compiler for decoder-only paired sigmoid diagnostic.

No production files change. No hardware access. Refuses an existing destination.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path[:0]=[str(HERE),str(ROOT/'models/bigcodec')]
import decoder_gate_pair_native as candidate
import bigcodec_lstm as lstm
import bigcodec_compile as compiler

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def sources():
 paths=[*sorted((ROOT/'models/bigcodec').glob('*.py')),ROOT/'user_dma_core.py',ROOT/'quant_lib.py',ROOT/'models/yolov5s/yolov5_precompiled.py',ROOT/'models/yolov5s/yolov5_common.py',Path(__file__),HERE/'decoder_gate_pair_native.py',HERE/'decoder_gate_pair_probe.py']
 return {str(p.relative_to(ROOT)):sha(p) for p in paths}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=ROOT/'test_samples/p232_007.wav');p.add_argument('--output',type=Path,default=HERE/'p232_007_paired_gates.bin');p.add_argument('--cpu-core',type=int,default=11);args=p.parse_args()
 if args.output.exists():p.error('Output already exists; this diagnostic never overwrites a bin')
 provenance=args.output.with_suffix('.paired_compile.json')
 if provenance.exists():p.error('Provenance destination already exists')
 before=sources();input_sha=sha(args.input);began=time.perf_counter()
 original_projection=lstm._recurrent_projection_sram;original_step=lstm._lstm_step_sram;original_compile=compiler.compile_models
 seen={'decoder_projection':0,'decoder_step':0,'legacy_projection':0,'legacy_step':0}
 def selected(plan):return plan.compensated_cell and plan.preserve_cell_residual and plan.compensated_tanh and plan.fused_projection
 def project(e,plan,*a,**kw):
  yes=selected(plan);seen['decoder_projection' if yes else 'legacy_projection']+=1
  return (candidate.projection if yes else original_projection)(e,plan,*a,**kw)
 def step(e,plan,*a,**kw):
  yes=selected(plan);seen['decoder_step' if yes else 'legacy_step']+=1
  return (candidate.step if yes else original_step)(e,plan,*a,**kw)
 lstm._recurrent_projection_sram=project;lstm._lstm_step_sram=step
 def annotated_compile(*a,**kw):
  payload=original_compile(*a,**kw)
  payload['precision']['lstm_sigmoid']='compensated-pade-high-low-decoder'
  payload['approximation']['lstm_sigmoid']='Decoder logits store BF16 before paired sigmoid via compensated tanh(x/2); gate high/low retained in cell and hidden products; sigmoid input clamps to [-8,8]'
  payload['diagnostic']={'name':'decoder_paired_sigmoid','sources_sha256':before,'production_source_files_modified':False}
  return payload
 compiler.compile_models=annotated_compile
 argv=['bigcodec_compile.py','--input',str(args.input),'--output',str(args.output),'--cpu-core',str(args.cpu_core),'--conv-precision','bf16','--lstm-cell-precision','compensated','--lstm-tanh-precision','compensated','--lstm-fused-gates','--lstm-math-scope','decoder','--center-quantizer-scores','--compensated-codebook','--filter-accumulation','serial','--memory-layout','large-program']
 old_argv=sys.argv;sys.argv=argv
 try:compiler.main()
 finally:sys.argv=old_argv;lstm._recurrent_projection_sram=original_projection;lstm._lstm_step_sram=original_step;compiler.compile_models=original_compile
 after=sources()
 if before!=after or sha(args.input)!=input_sha:raise RuntimeError('Sources or input changed during compilation; do not execute the generated bin')
 assert seen['decoder_projection']==seen['decoder_step']>0 and seen['legacy_projection']==seen['legacy_step']>0
 report=dict(status='compiled diagnostic; native acceptance pending',compiler_argv=argv,sources_before=before,sources_after=after,input=str(args.input.resolve()),input_sha256=input_sha,bin=str(args.output.resolve()),bin_sha256=sha(args.output),bin_bytes=args.output.stat().st_size,dispatch_counts=seen,elapsed_s=time.perf_counter()-began,scope='Paired sigmoid high/low only in compensated fused decoder LSTM; encoder arithmetic and all other operators unchanged',metadata='Bin precision.lstm_sigmoid and approximation.lstm_sigmoid identify the paired-gate diagnostic; source hashes embedded in diagnostic field')
 provenance.write_text(json.dumps(report,indent=2)+'\n');print('PAIRED_GATE_COMPILE:'+json.dumps(report),flush=True)
if __name__=='__main__':main()
