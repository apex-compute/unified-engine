"""Read saved native paired-gate probes; compare recovered values and frozen FP32 decoder tail. No hardware."""
import hashlib,json,os,sys,time
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import decoder_gate_pair_probe as cpu

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 os.sched_setaffinity(0,{12});torch.set_num_threads(1);torch.set_num_interop_threads(1)
 report=dict(scope=__doc__,script_sha256=sha(Path(__file__)),cpu_core=12,threads=1,native_outputs={},limitations=['Fixed historical FPGA token sequence, same input for both native decoder LSTMs; encoder/token selection is not rerun.','Native LSTM outputs feed the unchanged official FP32 decoder tail; this is not full-model FPGA acceptance.','Primitive sigmoid logits clamp to [-8,8]; recovered pair error includes the resulting tail saturation approximation.'])
 for name in ('primitive','baseline','candidate'):
  p=HERE/f'decoder_gate_pair_{name}_native.pt';value=torch.load(p,weights_only=True);r=value['record']
  assert r['hardware_version']=='0x90f1f464' and r['starts']==r['halts']==1 and r['finite'] and r['guards_intact'] and r['input_unchanged']
  report['native_outputs'][name]=dict(path=str(p),sha256=sha(p),record=r)
 primitive=torch.load(HERE/'decoder_gate_pair_primitive_native.pt',weights_only=True)['actual'].float()
 x=torch.linspace(-12,12,3072).bfloat16().float();expected=torch.stack(cpu.sigmoid_pair(x))
 assert torch.equal(primitive,expected)
 high,low=primitive;reference=x.sigmoid()
 report['primitive']=dict(component_mismatches=int((primitive!=expected).sum()),recovered_vs_fp32=cpu.base.metric(high+low,reference),high_only_vs_fp32=cpu.base.metric(high,reference),inrange_recovered_vs_fp32=cpu.base.metric((high+low)[x.abs()<=8],reference[x.abs()<=8]),input_abs_max=float(x.abs().max()),input_count=len(x))
 _,decoder=cpu.base.load_models(remove_weight_norm=True);tail=decoder.model[2:]
 ref=cpu.base.tensor(cpu.OLD/'p232_007_decoder_fp32/op_091.pt');report['reference_sha256']=sha(cpu.OLD/'p232_007_decoder_fp32/op_091.pt');report['variants']=[]
 with torch.inference_mode():
  refwave=tail(ref.T.unsqueeze(0)).flatten()[:63295]
  for name in ('baseline','candidate'):
   p=HERE/f'decoder_gate_pair_{name}_native.pt';actual=torch.load(p,weights_only=True)['actual'].float();started=time.perf_counter();wave=tail(actual.T.unsqueeze(0)).flatten()[:63295]
   if name=='baseline':
    golden=torch.load(cpu.OLD/'fused_compensated_compensated_native.pt',weights_only=True)['actual'].float();assert torch.equal(actual.view(torch.int32),golden.view(torch.int32));report['baseline_matches_frozen_native_golden']=True
   row=dict(name=name,lstm_vs_official=cpu.base.metric(actual,ref),fp32_tail_vs_official=cpu.base.metric(wave,refwave),native_lstm_s=report['native_outputs'][name]['record']['fpga_s'],cpu_tail_s=time.perf_counter()-started)
   report['variants'].append(row);torch.save(dict(output=actual,wave=wave,reference_wave=refwave),HERE/f'decoder_gate_pair_{name}_tail.pt');print(json.dumps(row),flush=True)
 cap=0xD0000000-0x84000000;report['size_projection']=dict(arena_bytes=cap,additional_instructions_per_layer_timestep=2704,additional_program_bytes_per_frame=2*2704*32,method='Exact isolated 317-frame capture difference; whole-graph size is projected, not compiled',cases=[])
 for name in ('bus','cafe','office','psquare','bus_low_snr','cafe_low_snr','office_low_snr','psquare_low_snr'):
  p=cpu.ROOT/'models/bigcodec/validation/20260915_accuracy/decoder/fpga_bf16'/f'{name}.metrics.json';m=json.loads(p.read_text());extra=m['compiled_samples']//200*2*2704*32
  report['size_projection']['cases'].append(dict(case=name,serial_resident_bytes=m['model_upload_bytes'],additional_bytes=extra,predicted_resident_bytes=m['model_upload_bytes']+extra,arena_remaining_bytes=cap-m['model_upload_bytes']-extra))
 report['status']='complete';(HERE/'decoder_gate_pair_native_evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report['primitive']),flush=True)
if __name__=='__main__':main()
