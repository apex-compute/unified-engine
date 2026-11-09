#!/usr/bin/env python3
"""Run the complete YOLO26n graph from a compiled, pinned artifact."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from yolo26_common import (HERE, add_strategy_arguments, decode_heads,
                           strategy_artifact_path, variant_paths)
from yolo26_artifact import artifact_variant, execute_cpu, load_artifact, select_profile
from yolov5_common import (configure_hardware_runtime, draw_detections,
                           letterbox_image, restore_boxes)
from yolov5_precompiled import WholeGraphAndromedaBackend
import user_dma_core


def main(argv=None, *, pinned_variant='n'):
    config, _, artifact = variant_paths(pinned_variant)
    model_name = config['model']
    parser = argparse.ArgumentParser(description=__doc__)
    strategy = add_strategy_arguments(parser)
    strategy.add_argument('--bin', type=Path,
                         help='Run this exact artifact instead of selecting the default')
    parser.add_argument('--image', type=Path, default=HERE.parent / 'test_samples/people.jpg')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resolution', choices=config['resolutions'], default=config['default_resolution'])
    parser.add_argument('--list-resolutions', action='store_true')
    parser.add_argument('--cpu', action='store_true', help='Use the quantized CPU reference')
    parser.add_argument('--compare', action='store_true', help='Also compare hardware heads with the CPU reference')
    parser.add_argument('--conf-thres', type=float, default=.25)
    parser.add_argument('--max-det', type=int, default=300)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--dev', default='xdma0')
    parser.add_argument('--device', default='kintex7')
    parser.add_argument('--timeout', type=float, default=300)
    args = parser.parse_args(argv)
    args.bin = args.bin or strategy_artifact_path(pinned_variant, args.strategy)
    if args.warmup < 0 or args.repeat < 1:
        parser.error('--warmup must be nonnegative and --repeat must be positive')
    if not 0 <= args.conf_thres <= 1 or not 1 <= args.max_det <= 300:
        parser.error('--conf-thres must be in [0,1] and --max-det in [1,300]')
    if args.compare and args.cpu:
        parser.error('--compare requires the hardware backend')
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    payload = load_artifact(args.bin)
    if artifact_variant(payload) != pinned_variant:
        raise RuntimeError(f'This runner requires {model_name}, but the artifact contains another variant')
    if args.list_resolutions:
        print('\n'.join(payload['profiles']))
        return
    selected = select_profile(payload, args.resolution)
    original, image, letterbox = letterbox_image(args.image, tuple(selected['model']['input_shape'][1:]))
    backend = None
    if not args.cpu:
        clock, info, _ = configure_hardware_runtime(device=args.device, dev=args.dev, cycle_override_ns=None)
        engine = user_dma_core.UnifiedEngine(clock_period_ns=clock,
            conv_geometry_mode=user_dma_core.CONV_GEOMETRY_QUEUE_CONFIG)
        engine.software_reset(run_dram_self_test=False)
        backend = WholeGraphAndromedaBackend(engine, selected,
            axi_data_width_bits=info.axi_data_width_bits, timeout_s=args.timeout)
    elapsed, cycles = [], []
    with torch.inference_mode():
        for iteration in range(args.warmup + args.repeat):
            before_cycles = sum(backend.cycles.values()) if backend else 0
            started = time.perf_counter()
            heads = backend.execute(image) if backend else execute_cpu(selected, image)
            duration = time.perf_counter() - started
            if iteration >= args.warmup:
                elapsed.append(duration)
                if backend:
                    cycles.append(sum(backend.cycles.values()) - before_cycles)
        detections = restore_boxes(decode_heads(heads, payload['names'],
            confidence=args.conf_thres, max_det=args.max_det), letterbox)
        comparison = None
        if args.compare:
            reference = execute_cpu(selected, image)
            comparison = []
            for actual, expected in zip(heads, reference):
                difference = (actual.float() - expected.float()).abs()
                comparison.append(dict(max_abs=float(difference.max()), mean_abs=float(difference.mean()),
                    cosine=float(torch.nn.functional.cosine_similarity(
                        actual.float().flatten(), expected.float().flatten(), dim=0))))
    tag = 'cpu' if args.cpu else 'hw'
    strategy = 'optimized' if payload.get('optimized') else ('winograd' if payload['winograd'] else 'native')
    output = args.output or artifact.parent / f'{args.image.stem}_{args.resolution}_{strategy}_{tag}.jpg'
    draw_detections(original, detections, output)
    result = dict(model=model_name, head='one2one', nms=False, backend=tag,
        convolution_strategy=strategy, input_resolution=args.resolution,
        artifact=str(args.bin.resolve()), artifact_version=payload['artifact_version'],
        bundle_sha256=payload['bundle_sha256'], output=str(output.resolve()),
        decoded_text=', '.join(d.label for d in detections), n_detections=len(detections),
        detections=[dict(label=d.label, class_id=d.class_id, confidence=round(d.score, 6),
                         box_xyxy=[round(x, 2) for x in d.box]) for d in detections],
        host_execution_s=elapsed, median_host_execution_s=statistics.median(elapsed),
        warmup=args.warmup, repeat=args.repeat)
    if backend:
        result.update(full_graph=True, hardware_version=f'0x{backend.hw_version:08x}',
            clock_period_ns=clock, cycles=cycles,
            fpga_execution_ms=[c * clock / 1e6 for c in cycles],
            median_fpga_execution_ms=statistics.median(cycles) * clock / 1e6,
            model_upload_bytes=backend.model_upload_bytes, model_upload_s=backend.model_upload_seconds,
            **{k: getattr(backend, k) for k in ('model_upload_writes', 'input_upload_writes',
                'program_kicks', 'output_reads', 'intermediate_upload_writes', 'intermediate_output_reads')})
    if comparison is not None:
        result['cpu_comparison'] = comparison
    print(f'{model_name.upper()} {args.resolution}, {strategy}, {tag}: {len(detections)} detections')
    for d in detections:
        print(f'  {d.score:.3f} {d.label}: {d.box}')
    print(f'Annotated image: {output}')
    print('TEST_RESULT: ' + json.dumps(result, sort_keys=True))
    return result


if __name__ == '__main__':
    main()
