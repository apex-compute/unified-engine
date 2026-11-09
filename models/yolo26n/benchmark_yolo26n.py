#!/usr/bin/env python3
"""Measure native/Winograd FPGA latency and drift from quantized/float references."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from yolo26_common import (HERE, FloatBackend,
                           decode_heads, execute_model, load_checkpoint, variant_paths)
from yolo26_artifact import artifact_variant, execute_cpu, load_artifact, select_profile
from yolov5_common import box_iou_one_to_many, configure_hardware_runtime, letterbox_image
from yolov5_precompiled import WholeGraphAndromedaBackend
from benchmark_winograd import compare_heads
import user_dma_core as core


def compare(reference, actual, names):
    a, b = [decode_heads(heads, names) for heads in (reference, actual)]
    unmatched = list(b)
    matches = []
    for detection in a:
        candidates = [i for i, other in enumerate(unmatched) if other.class_id == detection.class_id]
        if not candidates:
            continue
        overlaps = box_iou_one_to_many(torch.tensor(detection.box),
            torch.tensor([unmatched[i].box for i in candidates]))
        best = int(overlaps.argmax())
        if overlaps[best] >= .5:
            other = unmatched.pop(candidates[best])
            matches.append((float(overlaps[best]), abs(detection.score - other.score)))
    return dict(**compare_heads(reference, actual), reference_count=len(a), actual_count=len(b),
        matched_iou50=len(matches), min_matched_iou=min((v[0] for v in matches), default=None),
        max_confidence_delta=max((v[1] for v in matches), default=None))


def main(argv=None, *, pinned_variant='n'):
    config, checkpoint, artifact = variant_paths(pinned_variant)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, default=artifact)
    parser.add_argument('--winograd', type=Path, default=artifact.with_name(artifact.stem + '_winograd.bin'))
    parser.add_argument('--optimized', type=Path, default=artifact.with_name(artifact.stem + '_optimized.bin'))
    parser.add_argument('--checkpoint', type=Path, default=checkpoint)
    parser.add_argument('--resolutions', nargs='+', choices=config['resolutions'], default=config['resolutions'])
    parser.add_argument('--images', type=Path, nargs='+', default=[HERE.parent / 'test_samples/people.jpg',
                                                               HERE.parent / 'test_samples/vette.jpg'])
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--dev', default='xdma0')
    parser.add_argument('--device', default='kintex7')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats < 1:
        parser.error('warmup must be nonnegative and repeats must be positive')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    native, winograd, optimized = [load_artifact(p) for p in (args.native, args.winograd, args.optimized)]
    if (native['winograd'] or not winograd['winograd'] or winograd.get('optimized')
            or not optimized.get('optimized') or any(artifact_variant(p) != pinned_variant for p in (native, winograd, optimized))):
        parser.error('Artifacts must be native, Winograd and optimized versions of the selected model')
    model = load_checkpoint(args.checkpoint, variant=pinned_variant)
    clock, info, _ = configure_hardware_runtime(device=args.device, dev=args.dev)
    engine = core.UnifiedEngine(clock_period_ns=clock, conv_geometry_mode=core.CONV_GEOMETRY_QUEUE_CONFIG)
    engine.software_reset(run_dram_self_test=False)
    report = dict(model=config['model'], timestamp_utc=datetime.now(timezone.utc).isoformat(),
        hardware_version=f'0x{engine.get_hardware_version():08x}', clock_ns=clock,
        axi_width_bits=info.axi_data_width_bits, available_cores=info.core_count, used_cores=1,
        runtime_driver_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        torch_version=torch.__version__, source_commit=config['source_commit'],
        checkpoint_sha256=config['checkpoint_sha256'], precision=config['precision'],
        native_bundle_sha256=native['bundle_sha256'], winograd_bundle_sha256=winograd['bundle_sha256'],
        optimized_bundle_sha256=optimized['bundle_sha256'], checkpoint_parameters=sum(p.numel() for p in model.parameters()),
        warmup=args.warmup, repeats=args.repeats, confidence_threshold=.25, results=[])
    with torch.inference_mode():
        for resolution in args.resolutions:
            selected = select_profile(native, resolution)
            inputs = [letterbox_image(path, tuple(selected['model']['input_shape'][1:]))[1] for path in args.images]
            quantized = [execute_cpu(selected, x) for x in inputs]
            floating = [execute_model(model, x, FloatBackend(quantized=False)) for x in inputs]
            result = dict(resolution=resolution, timing={}, quality=[])
            outputs = []
            for mode, payload in [('native', native), ('winograd', winograd), ('optimized', optimized)]:
                program = select_profile(payload, resolution)
                backend = WholeGraphAndromedaBackend(engine, program,
                    axi_data_width_bits=info.axi_data_width_bits, timeout_s=60)
                accelerator, host = [], []
                for i in range(args.warmup + args.repeats):
                    before = sum(backend.cycles.values())
                    start = time.perf_counter()
                    heads = backend.execute(inputs[0])
                    elapsed = (time.perf_counter() - start) * 1000
                    if i >= args.warmup:
                        accelerator.append((sum(backend.cycles.values()) - before) * clock / 1e6)
                        host.append(elapsed)
                outputs.append([heads] + [backend.execute(x) for x in inputs[1:]])
                result['timing'][mode] = dict(accelerator_ms=accelerator, host_execution_ms=host,
                    median_accelerator_ms=statistics.median(accelerator),
                    median_host_execution_ms=statistics.median(host),
                    model_image_bytes=backend.model_upload_bytes, program_bytes=program['hardware']['program_size'],
                    scratch_bytes=program['hardware']['scratch_bytes'],
                    tensor_arena_bytes=program['hardware']['scratch_address'] + program['hardware']['scratch_bytes'] - program['hardware']['tensor_base'],
                    select_layers=[op['name'] for op in program['operations'] if op['op'] == 'select'],
                    depthwise_layers=[op['name'] for op in program['operations'] if op['op'] == 'depthwise'],
                    winograd_layers=program['hardware'].get('winograd_layers', []),
                    model_upload_writes=backend.model_upload_writes,
                    input_upload_writes=backend.input_upload_writes, program_kicks=backend.program_kicks,
                    output_reads=backend.output_reads, intermediate_upload_writes=backend.intermediate_upload_writes,
                    intermediate_output_reads=backend.intermediate_output_reads)
            baseline = result['timing']['native']['median_accelerator_ms']
            best = result['timing']['optimized']['median_accelerator_ms']
            result.update(speedup=baseline / best, latency_reduction_percent=100 * (1 - best / baseline),
                          accelerator_fps=1000 / best)
            for path, fp, cpu, hw, wino, fast in zip(args.images, floating, quantized, *outputs):
                result['quality'].append(dict(image=path.name,
                    float_to_cpu=compare(fp, cpu, native['names']),
                    cpu_to_native=compare(cpu, hw, native['names']),
                    float_to_native=compare(fp, hw, native['names']),
                    native_to_winograd=compare(hw, wino, native['names']),
                    native_to_optimized=compare(hw, fast, native['names']),
                    float_to_optimized=compare(fp, fast, native['names'])))
            report['results'].append(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(f'{resolution}: {baseline:.3f} -> {best:.3f} ms '
                  f'({result["latency_reduction_percent"]:.2f}% reduction)', flush=True)
    return report


if __name__ == '__main__':
    main()
