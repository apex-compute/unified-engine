"""Diagnostic only: paired sigmoid gates; no production source is modified.

Offline self-test/capture is default. Root alone may explicitly pass --execute.
Dots store BF16 logits before paired sigmoid, unlike the current fused SFU path.
"""
import argparse,contextlib,fcntl,hashlib,io,json,os,sys
from pathlib import Path
import torch
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3];OLD=HERE.parent/'accuracy_20260915'
sys.path[:0]=[str(HERE),str(OLD),str(ROOT/'models/bigcodec')]
import bigcodec_lstm as lstm
import bigcodec_tanh as dd
from bigcodec_device import shared,udc
from test_bigcodec_tanh import TanhEngine
import native_compensated_smoke as smoke
import decoder_gate_pair_probe as cpu

def sigmoid_pair_sram(engine,source,output,count):
 if count not in (64,1536):raise ValueError('Diagnostic supports width64/1536')
 if any(a<0 or a%128 or a+count*2>0x10000 for a in (source,*output)):raise ValueError('Low SRAM vectors required')
 engine.broadcast_mul(.5,source,dd._X[0],count)
 lstm._identity_sram(engine,dd._X[0],dd._X[0],count,0x80000,udc.LALU_MODE.CLAMP,lower=-4.,upper=4.)
 engine.broadcast_mul(0.,dd._X[0],dd._ZERO,count)
 lstm._two_product_sram(engine,dd._X[0],dd._X[0],*dd._SQUARE,count)
 dd._polynomial(engine,dd._NUMERATOR,dd._NUM,count)
 dd._polynomial(engine,dd._DENOMINATOR,dd._DEN,count)
 lstm._identity_sram(engine,dd._DEN[0],dd._INVERSE[0],count,0x80000,udc.LALU_MODE.MODE_RECIP)
 engine.broadcast_mul(0.,dd._ZERO,dd._INVERSE[1],count)
 dd._multiply_pair(engine,dd._DEN,dd._INVERSE,dd._PRODUCT,count)
 for a,b in zip(dd._PRODUCT,dd._NEGATIVE):engine.broadcast_mul(-1.,a,b,count)
 dd._constant_pair(engine,1.,dd._COEFF,count)
 dd._add_pair(engine,dd._COEFF,dd._NEGATIVE,dd._ERROR,count)
 lstm._binary_a(engine,'add',*dd._ERROR,dd._RESULT[0],count)
 lstm._binary_a(engine,'mul',dd._INVERSE[0],dd._RESULT[0],dd._RESULT[0],count)
 engine.broadcast_mul(0.,dd._ZERO,dd._RESULT[1],count)
 dd._add_pair(engine,dd._INVERSE,dd._RESULT,dd._INVERSE,count)
 dd._multiply_pair(engine,dd._NUM,dd._INVERSE,dd._PRODUCT,count)
 dd._multiply_pair(engine,dd._PRODUCT,dd._X,dd._RESULT,count)
 dd._constant_pair(engine,1.,dd._COEFF,count)
 dd._add_pair(engine,dd._COEFF,dd._RESULT,dd._RESULT,count)
 dd._constant_pair(engine,.5,dd._COEFF,count)
 dd._multiply_pair(engine,dd._RESULT,dd._COEFF,dd._RESULT,count)
 for a,b in zip(dd._RESULT,output):engine.broadcast_mul(1.,a,b,count)

def projection(engine,plan,layer,hidden,projection):
 width=plan.padded_width
 assert width==1536 and plan.recurrent_precision=='bf16' and plan.fused_projection
 tile=min(width,(udc.URAM_NEAR_FULL_ELEMENTS//width)//64*64)
 engine.accelerator_memory_to_sram(hidden,0,width)
 for gate in range(4):
  for offset in range(0,width,tile):
   first=gate*width+offset;take=min(tile,width-offset)
   engine.accelerator_memory_to_sram(layer.recurrent_weights+first*width*2,0x80000,take*width)
   engine.accelerator_memory_to_bias_sram(projection+first*2,take)
   engine.start_queue_for_bf16_matvec_operation(max_clear_en=0,fmax_context_addr=0,
    vector_sram_start_addr=0,matrix_sram_start_addr=0x80000,output_sram_wb_addr=0x8000+first*2,
    K=width,N=take,bias_enable=True,lalu_mode=udc.LALU_MODE.BYPASS,lalu_a=0,lalu_b=0)

def step(engine,plan,projection,previous_cell,output):
 width=plan.padded_width
 assert width==1536 and plan.compensated_cell and plan.preserve_cell_residual and plan.compensated_tanh
 engine.broadcast_mul(1.,0x8000,0,4*width)
 engine.accelerator_memory_to_sram(plan.identity_address,0x80000,4096)
 ig,fg,candidate,og=(i*width*2 for i in range(4))
 ip,fp,op=(ig,0x8000),(fg,0x9000),(og,0xA000)
 for high,low in (ip,fp,op):sigmoid_pair_sram(engine,high,(high,low),width)
 lstm._lstm_tanh_sram(engine,plan,candidate,candidate,width)
 cell=(0xC000,0xE000)
 engine.accelerator_memory_to_sram(previous_cell,cell[0],width)
 engine.accelerator_memory_to_sram(plan.regions['cell_low'][0],cell[1],width)
 dd._multiply_pair(engine,fp,cell,dd._PRODUCT,width)
 dd._multiply_pair(engine,ip,(candidate,dd._ZERO),dd._NEGATIVE,width)
 dd._add_pair(engine,dd._PRODUCT,dd._NEGATIVE,cell,width)
 engine.sram_to_accelerator_memory(cell[0],previous_cell,width)
 engine.sram_to_accelerator_memory(cell[1],plan.regions['cell_low'][0],width)
 lstm._lstm_tanh_sram(engine,plan,cell[0],cell[0],width)
 dd._multiply_pair(engine,op,(cell[0],dd._ZERO),dd._RESULT,width)
 lstm._binary_a(engine,'add',*dd._RESULT,0,width)
 engine.sram_to_accelerator_memory(0,output,width)

def install():
 lstm._recurrent_projection_sram=projection;lstm._lstm_step_sram=step

def self_test():
 records=[]
 torch.manual_seed(91015)
 for width in (64,1536):
  x=torch.linspace(-12,12,width).bfloat16().float()
  for source,high,low in ((0,0,0x8000),(0xC00,0xC00,0x9000),(0x2400,0x2400,0xA000)):
   e=TanhEngine();e.sram.fill_(-3.25);e.sram_view(0x80000,4096).copy_(torch.eye(64,dtype=torch.bfloat16).flatten());e.sram_view(source,width).copy_(x)
   before=e.sram.clone();sigmoid_pair_sram(e,source,(high,low),width)
   actual=(e.sram_view(high,width).float(),e.sram_view(low,width).float());expected=cpu.sigmoid_pair(x)
   for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
   preserve=torch.ones_like(e.sram,dtype=torch.bool)
   for a,b in ((high,high+width*2),(low,low+width*2),(0x10000,0x66000),(0xF0000,0xF2000)):preserve[a//2:b//2]=False
   assert torch.equal(e.sram[preserve],before[preserve]) and not e.dma_calls
   records.append(dict(width=width,source=source,exact_components=True,preserved_other_sram=True,max_abs=float((actual[0]+actual[1]-x.sigmoid()).abs().max())))
 width=1536;e=TanhEngine();e.sram.fill_(-3.25)
 logits=(torch.randn(4,width)*2).bfloat16().float();ch=(torch.randn(width)*3).bfloat16().float();cl=(torch.randn(width)/1000).bfloat16().float()
 identity,previous,lowaddr,out=0x90000000,0xB0000000,0xB0010000,0xB0020000
 e.regions[identity]=torch.eye(64,dtype=torch.bfloat16).flatten();e.regions[previous]=ch.bfloat16().clone();e.regions[lowaddr]=cl.bfloat16().clone();e.regions[out]=torch.full((width,),float('nan'),dtype=torch.bfloat16)
 e.sram_view(0x8000,4*width).copy_(logits.flatten())
 plan=SimpleNamespace(padded_width=width,compensated_cell=True,preserve_cell_residual=True,compensated_tanh=True,identity_address=identity,regions={'cell_low':(lowaddr,width)})
 step(e,plan,0,previous,out)
 ip,fp,op=(cpu.sigmoid_pair(logits[i]) for i in (0,1,3));g=cpu.ddpade(logits[2],'ddpoly_recipcorrection')
 c=cpu.ddadd(cpu.ddmul(fp,(ch,cl)),cpu.ddmul(ip,(g,torch.zeros_like(g))))
 ht=cpu.ddpade(c[0],'ddpoly_recipcorrection');h=cpu.ddmul(op,(ht,torch.zeros_like(ht)));expected=cpu.q(h[0]+h[1])
 for actual,target in ((e.view(previous,width).float(),c[0]),(e.view(lowaddr,width).float(),c[1]),(e.view(out,width).float(),expected)):torch.testing.assert_close(actual,target,rtol=0,atol=0)
 records.append(dict(width=width,joint_step_exact=True,cell_high_low_and_hidden_verified=True))
 return dict(status='PASS',cases=records,scope='Paired sigmoid components and joint gate/cell/hidden step versus independent BF19-to-BF16 CPU equations; actual identity bytes and SRAM preservation checked')

def make_case(mode):
 source=cpu.base.tensor(OLD/'p232_007_decoder_fp32/op_090.pt').bfloat16();official=cpu.base.tensor(OLD/'p232_007_decoder_fp32/op_091.pt')
 _,decoder=cpu.base.load_models(remove_weight_norm=True);module=decoder.model[1].lstm
 workspace=lstm.scratch_bytes(source.shape,compensated_cell=True,preserve_cell_residual=True)
 name='decoder_gate_pair_'+mode
 case=smoke.allocate_case(name,source,source.numel(),workspace//2)
 image=shared._ImageBuilder(shared.MODEL_BASE,shared.MODEL_LIMIT)
 zero=image.allocate(torch.zeros(64,dtype=torch.bfloat16),alignment=128);identity=image.allocate(torch.eye(64,dtype=torch.bfloat16),alignment=128)
 plan=lstm.prepare_lstm(module,image,input_shape=source.shape,input_address=case['input_address'],output_address=case['output_address'],scratch_address=case['scratch_address'],identity_address=identity,zero_address=zero,recurrent_precision='bf16',compensated_cell=True,preserve_cell_residual=True,compensated_tanh=True,fused_projection=True)
 if mode=='candidate':install()
 expected=cpu.base.tensor(HERE/f'decoder_gate_{"pade_rounded_pair" if mode=="candidate" else "native_like_current"}.pt')
 case.update(expected=expected,official=official,shape=list(source.shape),expected_label='vs_offline_fp32_dot_control')
 smoke.capture(case,image,lambda e:lstm.emit_lstm(e,plan))
 case['record'].update(scope=__doc__,mode=mode,source_fp32_sha256=hashlib.sha256((OLD/'p232_007_decoder_fp32/op_090.pt').read_bytes()).hexdigest(),expected_model='FP32 accumulation approximation, not native dot parity',checkpoint_sha256=cpu.base.CHECKPOINT_SHA256)
 return case

def primitive_case():
 count=3072;x=torch.linspace(-12,12,count).bfloat16().float()
 case=smoke.allocate_case('decoder_gate_pair_primitive',x,count*2)
 image=shared._ImageBuilder(shared.MODEL_BASE,shared.MODEL_LIMIT)
 identity=image.allocate(torch.eye(64,dtype=torch.bfloat16),alignment=128)
 def emit(e):
  e.accelerator_memory_to_sram(identity,0x80000,4096)
  for first in range(0,count,1536):
   e.accelerator_memory_to_sram(case['input_address']+first*2,0,1536)
   sigmoid_pair_sram(e,0,(0,0x8000),1536)
   e.sram_to_accelerator_memory(0,case['output_address']+first*2,1536)
   e.sram_to_accelerator_memory(0x8000,case['output_address']+(count+first)*2,1536)
 case.update(expected=torch.stack(cpu.sigmoid_pair(x)),official=torch.stack(cpu.ddvalue(x.sigmoid())),shape=[2,count],expected_label='components_vs_native_rounding_model')
 smoke.capture(case,image,emit);return case

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--capture',action='store_true');p.add_argument('--primitive',action='store_true');p.add_argument('--mode',choices=('baseline','candidate'),default='candidate');p.add_argument('--cpu-core',type=int,default=12);p.add_argument('--expected-version',type=lambda x:int(x,0),default=0x90f1f464);p.add_argument('--timeout',type=float,default=60);args=p.parse_args()
 os.sched_setaffinity(0,{args.cpu_core});torch.set_num_threads(1);udc.UE_AXI_DATA_WIDTH_BITS=256
 test=self_test();(HERE/'decoder_gate_pair_self_test.json').write_text(json.dumps(test,indent=2)+'\n');print(json.dumps(test),flush=True)
 if not (args.capture or args.execute):return
 case=primitive_case() if args.primitive else make_case(args.mode);report=case['record'];print(json.dumps(report),flush=True)
 if args.execute:
  from bigcodec_precompiled import StreamingEngine
  from yolov5_common import configure_hardware_runtime
  smoke.HERE=HERE;args.max_relative_error=0. if args.primitive else 1.
  with open('/tmp/pcie_ci_hw_italy.lock','r') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   clock,info,_=configure_hardware_runtime(device='rk',dev='xdma0',cycle_override_ns=None)
   assert info.axi_data_width_bits==256 and info.dram_size_gb>=2
   with StreamingEngine(clock_period_ns=clock,conv_geometry_mode=udc.CONV_GEOMETRY_QUEUE_CONFIG) as engine:
    report=smoke.execute_case(engine,clock,case,args)
    if engine.get_hardware_version()!=args.expected_version:raise RuntimeError('Image changed during execution')
  assert report['passed']
  report['guarded_execution_passed']=bool(report['finite'] and report['guards_intact'] and report['input_unchanged'])
  report['accuracy_accepted']=False
  report['pass_scope']='Exact primitive components' if args.primitive else 'Finite guarded execution and broad offline-control sanity bound only; not an accuracy acceptance'
  saved_path=HERE/(case['name']+'_native.pt');saved=torch.load(saved_path,weights_only=True)
  if args.primitive:
   report['native_arithmetic_contract_passed']=report['components_vs_native_rounding_model']['exact_mismatches']==0
   assert report['native_arithmetic_contract_passed']
  elif args.mode=='baseline':
   golden_path=OLD/'fused_compensated_compensated_native.pt'
   golden_hash=hashlib.sha256(golden_path.read_bytes()).hexdigest()
   assert golden_hash=='2a4fa842a9cc56ccd4343d7a31628e884dfdb1c9224ed3ec515480f2c0091d94'
   golden=torch.load(golden_path,weights_only=True)['actual'].float()
   report['baseline_matches_corrected_native_golden']=torch.equal(saved['actual'].view(torch.int32),golden.view(torch.int32))
   report['baseline_golden_sha256']=golden_hash
   assert report['baseline_matches_corrected_native_golden']
  saved['record']=report;torch.save(saved,saved_path)
  print(json.dumps(report),flush=True)
 (HERE/f'decoder_gate_pair_{"primitive" if args.primitive else args.mode}_{"native" if args.execute else "capture"}.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
