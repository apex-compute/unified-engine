"""Run Carvana U-Net from a precompiled single-HALT deployment image."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from unet_common import Backend, execute, load_artifact, udc

HERE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin', type=Path, default=HERE/'unet_bin/unet-andromeda.bin')
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=HERE/'unet_bin/mask.png')
    parser.add_argument('--resolution', default='256x256')
    parser.add_argument('--backend', choices=['cpu-fp32','cpu-quantized','hardware'], default='cpu-quantized')
    parser.add_argument('--device', choices=['bittware','bittware_512','efinix'], default='bittware_512')
    parser.add_argument('--dev', default='xdma0')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--compare', action='store_true', help='compare final logits and mask against quantized CPU reference')
    parser.add_argument('--progress', action='store_true')
    parser.add_argument('--trace-tail', type=Path, metavar='DIR',
                        help='Export the final whole-graph BRAM tail after the single HALT')
    args = parser.parse_args()
    if args.trace_tail is not None and args.backend != 'hardware':
        parser.error('--trace-tail requires --backend hardware')
    try:
        width, height = map(int, args.resolution.lower().split('x'))
        if min(width,height) < 16:
            raise ValueError()
    except ValueError:
        parser.error('resolution must be WIDTHxHEIGHT, each >=16')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    payload, layers = load_artifact(args.bin)
    if args.backend == 'hardware':
        if not payload.get('full_graph'):
            parser.error('legacy layer bin: rebuild with unet_compile.py --force')
        if payload['hardware']['resolution'] != [width,height]:
            parser.error('resolution differs from bin; rebuild with --resolution '+args.resolution)
    original = Image.open(args.image).convert('RGB')
    # Explicit fixed profile: this may differ from upstream's image-relative scale=0.5.
    resized = original.resize((width,height), Image.Resampling.BICUBIC)
    x = torch.from_numpy(np.asarray(resized).copy()).permute(2,0,1).float()/255
    engine = None
    clock = None
    if args.backend == 'hardware':
        udc.set_dma_device('efinix' if args.device == 'efinix' else args.dev)
        clock = udc.configure_clock_from_hardware()
        info = udc.configured_hardware_info()
        if info.axi_data_width_bits not in (256,512):
            raise RuntimeError('U-Net requires AXI-256 or AXI-512')
        engine = udc.UnifiedEngine(clock_period_ns=clock,
                     conv_geometry_mode=udc.CONV_GEOMETRY_QUEUE_CONFIG)
        engine.software_reset(run_dram_self_test=False)
    if engine is not None:
        from unet_precompiled import WholeGraphBackend
        backend = WholeGraphBackend(engine,payload['hardware'],args.timeout,args.trace_tail)
    else:
        backend = Backend(args.backend)
    print(f'U-Net {args.backend}, input={width}x{height}, full_graph={engine is not None}', flush=True)
    if args.trace_tail is not None:
        print('Trace: final 8192 decode events after one HALT; host time includes export overhead.')
    def observer(name, value):
        if not torch.isfinite(value).all():
            raise RuntimeError(f'{name}: nonfinite output')
        if args.progress:
            print(f'{name}: {tuple(value.shape)}', flush=True)
    with torch.inference_mode():
        start = time.perf_counter()
        logits = backend.execute(x) if engine is not None else execute(layers,x,backend,observer)
        if engine is not None:
            observer('outc',logits)
        elapsed = time.perf_counter()-start
        result = dict(model='unet_carvana', backend=args.backend,
                      input_resolution=args.resolution, full_graph=engine is not None,
                      execution_elapsed_s=elapsed, program_kicks=backend.kicks,
                      checkpoint_sha256=payload['checkpoint_sha256'])
        if args.compare:
            ref = execute(layers,x,Backend('cpu-quantized'))
            error = (logits.float()-ref.float()).square().mean()
            result['logit_rmse'] = float(error.sqrt())
            result['mask_agreement'] = float((logits.argmax(0)==ref.argmax(0)).float().mean())
        output = F.interpolate(logits.float()[None], size=(original.height,original.width),
                               mode='bilinear', align_corners=False)[0].argmax(0)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    values = payload.get('mask_values',payload.get('state_dict',{}).get('mask_values',[0,1]))
    values = np.asarray(values)
    if values.shape != (2,):
        raise ValueError('expected scalar two-class mask values')
    # Save a visible binary mask; class IDs remain represented by black/white.
    Image.fromarray((output.numpy()*255).astype(np.uint8)).save(args.output)
    result['mask_path'] = str(args.output.resolve())
    if clock is not None:
        result['fpga_cycles_sum'] = backend.cycles
        result['fpga_execution_s_sum'] = backend.cycles*clock*1e-9
    if engine is not None:
        result.update(model_upload_bytes=backend.model_upload_bytes,
                      model_upload_s=backend.model_upload_s, model_upload_writes=1,
                      input_upload_writes=1,output_reads=1,
                      intermediate_upload_writes=0,intermediate_output_reads=0)
        if backend.trace_result is not None:
            result['trace_tail'] = backend.trace_result
            result['trace_export_s'] = backend.trace_export_s
            print(f'Perfetto trace: {backend.trace_result["perfetto"]}')
    print('TEST_RESULT:'+json.dumps(result))

if __name__ == '__main__':
    main()
