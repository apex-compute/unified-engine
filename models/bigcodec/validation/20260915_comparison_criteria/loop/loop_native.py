"""Guarded native parity harness for counted decoder LSTM loops.

Offline capture by default. Root alone may pass --execute. Reuses the original
paired-gate guarded harness and recorded unrolled native tensors without editing
any existing script, reference, parameter layout or arithmetic implementation.
"""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,socket,sys
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path[:0]=[str(HERE),str(HERE.parent/'accuracy_20260915'),str(ROOT/'models/bigcodec')]
import decoder_gate_pair_native as original
import bigcodec_lstm as lstm
import loop_lstm
from bigcodec_device import udc
smoke=original.smoke
EXPECTED_LOOP_SHA256='51fb7ef22981a95e4df3d3e16dd6e41d586a490408ed3213a029712421af8d56'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sources():
    paths=[Path(__file__),HERE/'loop_lstm.py',HERE/'decoder_gate_pair_native.py',HERE/'decoder_gate_pair_probe.py',
        HERE.parent/'accuracy_20260915/native_compensated_smoke.py',ROOT/'models/bigcodec/bigcodec_lstm.py',
        ROOT/'models/bigcodec/bigcodec_tanh.py',ROOT/'user_dma_core.py',ROOT/'models/yolov5s/yolov5_precompiled.py']
    return {str(path.relative_to(ROOT)):sha(path) for path in paths}

def make_case(mode,expected_version,name):
    if sha(HERE/'loop_lstm.py')!=EXPECTED_LOOP_SHA256:raise ValueError('Loop helper differs from the audited version')
    before=sources();golden_path=HERE/f'decoder_gate_pair_{mode}_native.pt';golden_sha=sha(golden_path)
    golden=torch.load(golden_path,map_location='cpu',weights_only=True)
    record=golden['record'];expected=golden['actual'].float().contiguous()
    if (record['hardware_version']!=f'0x{expected_version:08x}' or record['starts']!=1 or record['halts']!=1
            or not all(record[key] for key in ('hardware_executed','finite','guards_intact','input_unchanged'))
            or record['output_sha256']!=smoke.digest(expected)):
        raise ValueError('Unrolled native reference has invalid provenance')
    emit,project,step=lstm.emit_lstm,lstm._recurrent_projection_sram,lstm._lstm_step_sram
    try:
        lstm.emit_lstm=loop_lstm.emit_lstm
        case=original.make_case(mode)
    finally:
        lstm.emit_lstm=emit;lstm._recurrent_projection_sram=project;lstm._lstm_step_sram=step
    if before!=sources() or golden_sha!=sha(golden_path):raise RuntimeError('Sources or native reference changed during capture')
    if case['record']['input_sha256']!=record['input_sha256'] or tuple(case['shape'])!=tuple(expected.shape):
        raise ValueError('Loop and unrolled native input/shape mismatch')
    case['name']=name;case['expected']=expected;case['expected_label']='vs_unrolled_native'
    case['record'].update(name=name,mode=mode,scope=__doc__,source_sha256=before,
        unrolled_native_reference=dict(path=str(golden_path.relative_to(ROOT)),sha256=golden_sha,
            output_sha256=record['output_sha256'],input_sha256=record['input_sha256'],hardware_version=record['hardware_version']),
        expected_label='Exact bit parity with the recorded unrolled native LSTM, not an FP32 approximation',
        expected_model='Recorded unrolled native FPGA output tensor',
        loop_helper_sha256=EXPECTED_LOOP_SHA256,native_parity_accepted=False)
    return case

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--mode',choices=('baseline','candidate'),default='baseline')
    parser.add_argument('--cpu-core',type=int,default=13)
    parser.add_argument('--expected-version',type=lambda value:int(value,0),default=0x90f1f464)
    parser.add_argument('--timeout',type=float,default=120.)
    parser.add_argument('--name',help='Fresh output basename beginning loop_; defaults to loop_decoder_MODE')
    args=parser.parse_args();args.max_relative_error=0.
    name=args.name or f'loop_decoder_{args.mode}'
    if not name.startswith('loop_') or Path(name).name!=name:parser.error('Name must be a loop_ basename')
    native_path=HERE/(name+'_native.pt');report_path=HERE/(name+('_native.json' if args.execute else '_capture.json'))
    if native_path.exists() or report_path.exists():parser.error('Output exists; choose a fresh --name')
    if not 0<=args.expected_version<2**32:parser.error('Expected version must be uint32')
    os.sched_setaffinity(0,{args.cpu_core});torch.set_num_threads(1);torch.set_num_interop_threads(1);udc.UE_AXI_DATA_WIDTH_BITS=256
    case=make_case(args.mode,args.expected_version,name);report=case['record']
    if args.execute:
        if socket.gethostname().split('.')[0].lower()!='italy':raise RuntimeError('Native execution is restricted to Italy')
        from bigcodec_precompiled import StreamingEngine
        from yolov5_common import configure_hardware_runtime
        smoke.HERE=HERE
        with open('/tmp/pcie_ci_hw_italy.lock','r') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            clock,info,_=configure_hardware_runtime(device='rk',dev='xdma0',cycle_override_ns=None)
            if info.axi_data_width_bits!=256 or info.dram_size_gb<2:raise RuntimeError('AXI256/2GiB hardware required')
            with StreamingEngine(clock_period_ns=clock,conv_geometry_mode=udc.CONV_GEOMETRY_QUEUE_CONFIG) as engine:
                report=smoke.execute_case(engine,clock,case,args)
                report['version_unchanged']=engine.get_hardware_version()==args.expected_version
        saved=torch.load(native_path,weights_only=True)
        bit_mismatches=int((saved['actual'].float().contiguous().view(torch.int32)!=case['expected'].view(torch.int32)).sum())
        report['unrolled_native_bit_mismatches']=bit_mismatches
        report['sources_unchanged']=report['source_sha256']==sources()
        golden=report['unrolled_native_reference']
        report['unrolled_reference_unchanged']=sha(ROOT/golden['path'])==golden['sha256']
        report['native_parity_accepted']=bool(report['passed'] and bit_mismatches==0 and report['version_unchanged']
            and report['sources_unchanged'] and report['unrolled_reference_unchanged'])
        report['passed']=report['native_parity_accepted'];saved['record']=report;torch.save(saved,native_path)
        report['saved_native_artifact_sha256']=sha(native_path)
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report),flush=True)
    if args.execute and not report['native_parity_accepted']:raise RuntimeError('Counted-loop native bit parity failed; see saved record')
if __name__=='__main__':main()
