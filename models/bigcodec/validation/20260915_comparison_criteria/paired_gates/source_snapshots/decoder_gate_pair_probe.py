"""Offline control for retaining sigmoid gate residuals. Native BF19/BF20 matrix reduction is NOT modeled."""
import hashlib,json,os,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[4]
OLD=ROOT/'models/bigcodec/bigcodec_bin/accuracy_20260915'
OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(OLD),str(ROOT/'models/bigcodec')]
import lstm_native_rounding_ablation as base
from lstm_fused_sigmoid_ablation import sigmoid_sfu19
from probe_tanh_compensated import ddvalue,ddadd,ddmul,ddneg,ddrecip,ddpoly,twoproduct,ddpade,NUM,DEN
from probe_bf19_rounding import q19
q=base.double_rne

def tanh_pair(x):
 x=x.clamp(-4,4)
 z=twoproduct(x,x)
 return ddmul(ddmul(ddpoly(z,NUM),ddrecip(ddpoly(z,DEN))),(x,torch.zeros_like(x)))

def sigmoid_pair(x):
 # The positive and negative branches both retain the near-one gate residual.
 return ddmul(ddadd(ddvalue(1.),tanh_pair(q(x*.5))),ddvalue(.5))

def run(module,source,mode):
 original=base.q16(source);current=original;stats=[]
 for layer in range(2):
  wi,wh=(base.q16(getattr(module,f'{prefix}_l{layer}').detach()) for prefix in ('weight_ih','weight_hh'))
  bias=base.q16(getattr(module,f'bias_ih_l{layer}').detach()+getattr(module,f'bias_hh_l{layer}').detach())
  projections=base.q16(current@wi.T+bias)
  h=torch.zeros(module.hidden_size);cell=torch.zeros_like(h);low=torch.zeros_like(h);output=[]
  gate_sse=gate_energy=0.;above8=0;gcount=0
  for projection in projections:
   raw=q19(projection+h@wh.T)
   i,f,g,o=raw.chunk(4)
   values=[]
   for x in (i,f,o):
    exact=x.sigmoid();above8+=int((x.abs()>8).sum());gcount+=x.numel()
    if mode=='native_like_current':pair=(sigmoid_sfu19(x),torch.zeros_like(x))
    elif mode=='ideal_fused_bf16':pair=(base.q16(exact),torch.zeros_like(x))
    elif mode=='ideal_rounded_bf16':pair=(base.q16(base.q16(x).sigmoid()),torch.zeros_like(x))
    elif mode=='pade_rounded_bf16':
     p=sigmoid_pair(base.q16(x));pair=(q(p[0]+p[1]),torch.zeros_like(x))
    elif mode=='pade_rounded_pair':pair=sigmoid_pair(base.q16(x))
    elif mode=='ideal_rounded_pair':pair=ddvalue(base.q16(x).sigmoid())
    else:raise ValueError(mode)
    gate_sse+=float(((pair[0]+pair[1])-exact).double().square().sum());gate_energy+=float(exact.double().square().sum())
    values.append(pair)
   ip,fp,op=values;g=ddpade(base.q16(g),'ddpoly_recipcorrection')
   if mode.endswith('_pair'):
    cell,low=ddadd(ddmul(fp,(cell,low)),ddmul(ip,(g,torch.zeros_like(g))))
   else:
    i,f=ip[0],fp[0]
    p0,e0=base.two_product(f,cell,q);p1,e1=base.two_product(i,g,q)
    high,error=base.two_sum(p0,p1,q);correction=q(q(e0+e1)+error)
    correction=q(correction+q(f*low));cell,low=base.two_sum(high,correction,q)
   ht=ddpade(cell,'ddpoly_recipcorrection')
   if mode.endswith('_pair'):
    hp=ddmul(op,(ht,torch.zeros_like(ht)));h=q(hp[0]+hp[1])
   else:h=q(op[0]*ht)
   output.append(h)
  current=torch.stack(output);stats.append(dict(layer=layer,gate_relative_l2=(gate_sse/gate_energy)**.5,gate_abs_over8=above8,gate_count=gcount))
 return q(current+original),stats

def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);os.sched_setaffinity(0,{12})
 trace=OLD/'p232_007_decoder_fp32';source=base.tensor(trace/'op_090.pt');reference=base.tensor(trace/'op_091.pt')
 _,decoder=base.load_models(remove_weight_norm=True);tail=decoder.model[2:]
 report=dict(scope=__doc__,source_sha256=hashlib.sha256((trace/'op_090.pt').read_bytes()).hexdigest(),reference_sha256=hashlib.sha256((trace/'op_091.pt').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),native_samples=63295,cpu_core=12,threads=1,variants=[])
 with torch.inference_mode():
  refwave=tail(reference.T.unsqueeze(0)).flatten()[:63295]
  for mode in ('native_like_current','ideal_fused_bf16','ideal_rounded_bf16','pade_rounded_bf16','pade_rounded_pair','ideal_rounded_pair'):
   start=time.perf_counter();actual,stats=run(decoder.model[1].lstm,source,mode);wave=tail(actual.T.unsqueeze(0)).flatten()[:63295]
   row=dict(name=mode,lstm_vs_official=base.metric(actual,reference),downstream_wave_vs_official=base.metric(wave,refwave),stats=stats,elapsed_s=time.perf_counter()-start)
   report['variants'].append(row);torch.save(dict(output=actual,wave=wave),OUT/f'decoder_gate_{mode}.pt');(OUT/'decoder_gate_pair_probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
 report['status']='complete';(OUT/'decoder_gate_pair_probe.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
