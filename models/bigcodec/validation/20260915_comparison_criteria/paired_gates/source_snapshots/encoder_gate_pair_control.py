"""One offline paired-gate encoder LSTM control; no hardware or production edits.

Saved native op085 is injected. Candidate matmul uses FP32 dot accumulation,
not native BF19/BF20 reduction. Native op086 and this CPU candidate then use the
same unchanged official FP32 encoder tail/VQ. These are not whole-FPGA tokens.
"""
import hashlib,json,os,sys,time
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
OLDER=HERE.parent/'accuracy_encoder_20260915'
sys.path[:0]=[str(HERE),str(OLDER)]
import encoder_lstm_ablation as control
import decoder_gate_pair_probe as candidate

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 os.sched_setaffinity(0,{12});torch.set_num_threads(1);torch.set_num_interop_threads(1)
 output=HERE/'encoder_gate_pair_control.json';tensorout=HERE/'encoder_gate_pair_control.pt'
 if output.exists() or tensorout.exists():raise FileExistsError('Preserve existing control; choose a new report name')
 start=time.perf_counter()
 encoder,decoder,source,official,conditional,official_tail,conditional_tail,provenance=control.references(control.DEFAULT_CHECKPOINT)
 assert encoder.block[6].skip
 baselinepath=OLDER/'native_cumulative/op_086.pt';baseline=control.tensor(baselinepath)
 manifest=json.loads((OLDER/'native_cumulative/results.json').read_text())
 row=next(r for r in manifest['layers'] if r['index']==86)
 assert row['output_sha256']==sha(baselinepath)
 baseline_metrics=control.compare(baseline,official,conditional,official_tail,conditional_tail,encoder,decoder)
 with torch.inference_mode():
  actual,stats=candidate.run(encoder.block[6].lstm,source.float(),'pade_rounded_pair')
 metrics=control.compare(actual,official,conditional,official_tail,conditional_tail,encoder,decoder)
 torch.save(dict(output=actual,stats=stats,baseline=baseline),tensorout)
 report=dict(scope=__doc__,status='complete',cpu_core=12,threads=1,hardware_executed=False,script_sha256=sha(Path(__file__)),candidate_source_sha256=sha(HERE/'decoder_gate_pair_probe.py'),provenance=provenance,baseline_source=str(baselinepath),baseline_sha256=sha(baselinepath),baseline_native_comparison=baseline_metrics,candidate_offline_comparison=metrics,candidate_gate_statistics=stats,candidate_tensor=str(tensorout),candidate_tensor_sha256=sha(tensorout),elapsed_s=time.perf_counter()-start)
 report['ranking']=dict(local_lstm_improves=metrics['lstm_vs_fp32_same_input']['relative_l2']<baseline_metrics['lstm_vs_fp32_same_input']['relative_l2'],official_tail_tokens_improve=metrics['fp32_tail_vs_official']['tokens']['matches']>baseline_metrics['fp32_tail_vs_official']['tokens']['matches'],official_tail_features_improve=metrics['fp32_tail_vs_official']['features']['relative_l2']<baseline_metrics['fp32_tail_vs_official']['features']['relative_l2'])
 output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 for name,values in [('baseline_native',baseline_metrics),('paired_gates_offline',metrics)]:
  print(json.dumps(dict(name=name,lstm_official=values['lstm_vs_official_fp32']['relative_l2'],lstm_same_input=values['lstm_vs_fp32_same_input']['relative_l2'],features=values['fp32_tail_vs_official']['features']['relative_l2'],tokens=values['fp32_tail_vs_official']['tokens'])),flush=True)
 print(json.dumps(report['ranking']),flush=True)
if __name__=='__main__':main()
