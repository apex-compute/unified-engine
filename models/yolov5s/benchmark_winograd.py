#!/usr/bin/env python3
"""Compare native and Winograd whole-graph YOLO programs on the same board.

Build the native bins first with yolov5{s,n}_compile.py --native, then pass --artifact-dir
containing yolov5{s,n}-native.bin. Compilation and model uploads are excluded
from accelerator timing. Execution timing includes input/output transfers.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import user_dma_core as core
from yolov5_artifact import (SUPPORTED_INPUT_RESOLUTIONS, artifact_model_view,
                             load_single_bin, select_single_bin_profile)
from yolov5_common import (configure_hardware_runtime, decode_yolov5, letterbox_image,
                           non_max_suppression, box_iou_one_to_many)
from yolov5_precompiled import WholeGraphAndromedaBackend, compile_precompiled_hardware


def compare_heads(reference, actual):
    a = torch.cat([x.float().flatten() for x in reference])
    b = torch.cat([x.float().flatten() for x in actual])
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise RuntimeError("Nonfinite detection head in benchmark")
    error = (a - b).square().sum().item()
    return dict(snr_db=10 * math.log10(a.square().sum().item() / error) if error else None,
                max_abs_error=(a - b).abs().max().item(),
                relative_l2=(error / max(a.square().sum().item(), 1e-30)) ** .5)


def compare_detections(reference, actual, model):
    a, b = [non_max_suppression(decode_yolov5(heads, model), model,
                                conf_threshold=.25, iou_threshold=.45,
                                max_det=300) for heads in (reference, actual)]
    unmatched = list(b)
    matches = []
    for detection in sorted(a, key=lambda d: d.score, reverse=True):
        candidates = [i for i, d in enumerate(unmatched) if d.class_id == detection.class_id]
        if not candidates:
            continue
        ious = box_iou_one_to_many(torch.tensor(detection.box),
                                  torch.tensor([unmatched[i].box for i in candidates]))
        best = int(ious.argmax())
        if ious[best] >= .5:
            other = unmatched.pop(candidates[best])
            matches.append((float(ious[best]), abs(detection.score - other.score)))
    return dict(native_count=len(a), winograd_count=len(b), matched_iou50=len(matches),
                min_matched_iou=min((m[0] for m in matches), default=None),
                max_confidence_delta=max((m[1] for m in matches), default=None))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', choices=('s', 'n'), default=['s', 'n'])
    parser.add_argument('--resolutions', nargs='+', choices=SUPPORTED_INPUT_RESOLUTIONS,
                        default=list(SUPPORTED_INPUT_RESOLUTIONS))
    parser.add_argument('--images', nargs='+', type=Path,
                        default=[ROOT / 'models/test_samples/people.jpg',
                                 ROOT / 'models/test_samples/vette.jpg'])
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--dev', default='xdma0')
    parser.add_argument('--device', default='rk')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats < 1:
        parser.error('warmup must be >= 0 and repeats must be >= 1')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    clock, info, _ = configure_hardware_runtime(device=args.device, dev=args.dev)
    engine = core.UnifiedEngine(clock_period_ns=clock,
                                conv_geometry_mode=core.CONV_GEOMETRY_QUEUE_CONFIG)
    engine.software_reset(run_dram_self_test=False)
    report = dict(board_version=f'0x{engine.get_hardware_version():08x}',
                  timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  runtime_driver_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
                  winograd_source_commit='7c7ff3289d9b2c4bf1dbca25233cad6af569ad57',
                  clock_ns=clock, axi_width_bits=info.axi_data_width_bits,
                  available_cores=info.core_count, used_cores=1,
                  torch_version=torch.__version__, warmup=args.warmup,
                  repeats=args.repeats, results=[])
    for variant in args.variants:
        payload = load_single_bin(args.artifact_dir / f'yolov5{variant}-native.bin')
        if 'winograd_layers' in payload['hardware']:
            raise RuntimeError('The baseline artifact must be compiled with --native')
        for resolution in args.resolutions:
            _, selected = select_single_bin_profile(payload, resolution)
            model = artifact_model_view(selected)
            inputs = [letterbox_image(path, tuple(selected['model']['input_shape'][1:]))[1]
                      for path in args.images]
            with contextlib.redirect_stdout(io.StringIO()):
                hardware = compile_precompiled_hardware(selected, winograd=True)
            optimized = dict(selected, hardware=hardware)
            result = dict(variant=variant, resolution=resolution,
                          winograd_layers=hardware['winograd_layers'], timing={}, quality=[])
            outputs = []
            for mode, program in [('native', selected), ('winograd', optimized)]:
                backend = WholeGraphAndromedaBackend(engine, program,
                            axi_data_width_bits=info.axi_data_width_bits, timeout_s=60)
                accelerator, execution = [], []
                for i in range(args.warmup + args.repeats):
                    before = backend.cycles['whole_graph']
                    start = time.perf_counter()
                    heads = backend.execute(inputs[0])
                    wall_ms = (time.perf_counter() - start) * 1000
                    ms = (backend.cycles['whole_graph'] - before) * clock / 1e6
                    if i >= args.warmup:
                        accelerator.append(ms)
                        execution.append(wall_ms)
                outputs.append([heads] + [backend.execute(x) for x in inputs[1:]])
                result['timing'][mode] = dict(accelerator_ms=accelerator, execution_ms=execution,
                    median_accelerator_ms=statistics.median(accelerator),
                    median_execution_ms=statistics.median(execution),
                    model_image_bytes=program['hardware']['model_image'].numel(),
                    program_bytes=program['hardware']['program_size'],
                    model_sha256=program['hardware']['model_sha256'],
                    program_sha256=program['hardware']['program_sha256'])
            native = result['timing']['native']['median_accelerator_ms']
            wino = result['timing']['winograd']['median_accelerator_ms']
            result.update(speedup=native / wino, latency_reduction_percent=100 * (1 - wino / native))
            for path, a, b in zip(args.images, *outputs):
                path = path.resolve()
                image_name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
                result['quality'].append(dict(image=image_name, **compare_heads(a, b),
                                             detections=compare_detections(a, b, model)))
            report['results'].append(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            snr = result['quality'][0]['snr_db']
            snr_text = f'{snr:.2f} dB' if snr is not None else 'identical'
            print(f'YOLOv5{variant} {resolution}: {native:.3f} -> {wino:.3f} ms, '
                  f'{native / wino:.3f}x, SNR {snr_text}', flush=True)


if __name__ == '__main__':
    main()
