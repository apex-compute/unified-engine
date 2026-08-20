#!/usr/bin/env python3
"""Direct YOLOv5 execution from one compiled Andromeda bin."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import user_dma_core
from yolov5_artifact import (
    artifact_variant,
    artifact_model_view,
    execute_single_bin,
    load_single_bin,
)
from yolov5_common import (
    TorchBackend,
    decode_yolov5,
    draw_detections,
    get_yolov5_variant,
    load_yolov5_config,
    letterbox_image,
    non_max_suppression,
    restore_boxes,
    sha256_file,
)
from yolov5_precompiled import PrecompiledAndromedaBackend


def _clock_ns_default_for_device(device: str) -> float:
    if device == "efinix":
        return 4.0
    if device == "kintex7":
        return 1000 / (1066 / 5.375)
    if device in ("rk", "rk_256", "puzhi"):
        return 3.0
    if device in ("bittware", "bittware_256", "alveo", "alveo_u55c"):
        return 3.3333333
    return 10.0


def _format_detection(detection) -> str:
    x1, y1, x2, y2 = detection.box
    return (f"{detection.score * 100:6.2f}%  {detection.label:<18} "
            f"[{x1:6.1f}, {y1:6.1f}, {x2:6.1f}, {y2:6.1f}]")


def main(argv=None, *, pinned_variant: str = "s",
         config_path: Path | None = None) -> None:
    model_name = get_yolov5_variant(pinned_variant).model_name
    parser = argparse.ArgumentParser(
        description=f"Run {model_name} from one compiled Andromeda bin")
    parser.set_defaults(variant=pinned_variant)
    parser.add_argument("--bin", type=Path, default=None)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--backend", choices=("hardware", "cpu-quantized"),
                        default="hardware")
    parser.add_argument("--cpu", action="store_true",
                        help="Alias for --backend cpu-quantized")
    parser.add_argument("--conf-thres", type=float, default=None)
    parser.add_argument("--iou-thres", type=float, default=None)
    parser.add_argument("--max-det", type=int, default=None)
    parser.add_argument("--dev", default="xdma0")
    parser.add_argument("--device", default="kintex7")
    parser.add_argument("--cycle", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--allow-unknown-hardware", action="store_true",
        help="Bypass native-CONV, queue-CONFIG, and gather-IF8 FPGA hash gates")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    if args.cpu:
        args.backend = "cpu-quantized"

    profile, config, resource_dir = load_yolov5_config(
        args.variant, config_path)
    if args.bin is None:
        args.bin = resource_dir / config["paths"]["artifact"]
    if args.image is None:
        args.image = (resource_dir / config["paths"]["default_image"]).resolve()

    load_started = time.perf_counter()
    payload = load_single_bin(args.bin)
    payload_variant = artifact_variant(payload)
    if payload_variant != profile.key:
        raise RuntimeError(
            f"--variant {profile.key} selected {profile.model_name}, but "
            f"{args.bin} contains YOLOv5{payload_variant}")
    model = artifact_model_view(payload)
    runtime = payload["runtime"]
    defaults = runtime["postprocessing"]
    conf_threshold = (float(defaults["confidence_threshold"])
                      if args.conf_thres is None else args.conf_thres)
    iou_threshold = (float(defaults["iou_threshold"])
                     if args.iou_thres is None else args.iou_thres)
    max_detections = (int(defaults["max_detections"])
                      if args.max_det is None else args.max_det)
    configured_hw_versions = set(runtime["compatible_fpga_hashes"])
    if configured_hw_versions != set(user_dma_core.UE_GATHER_IF8_HW_VERSIONS):
        raise RuntimeError("single-bin gather-IF8 metadata disagrees with the driver")
    load_elapsed = time.perf_counter() - load_started
    print(f"{profile.model_name} direct-bin backend={args.backend} image={args.image}")
    print(f"Single bin: {args.bin.resolve()}")
    print(f"  sha256={sha256_file(args.bin.resolve())} load={load_elapsed:.3f}s")
    original, image_tensor, letterbox = letterbox_image(
        args.image, int(payload["model"]["input_shape"][1]))

    if args.backend == "hardware":
        user_dma_core.set_dma_device(
            "efinix" if args.device == "efinix" else args.dev)
        axi_width = 512 if args.device in ("bittware", "rk") else 256
        os.environ["UE_AXI_DATA_WIDTH_BITS"] = str(axi_width)
        user_dma_core.UE_AXI_DATA_WIDTH_BITS = axi_width
        clock = (args.cycle if args.cycle is not None
                 else _clock_ns_default_for_device(args.device))
        user_dma_core.CLOCK_CYCLE_TIME_NS = clock
        user_dma_core.UE_PEAK_GFLOPS = 0.128 / clock
        ue = user_dma_core.UnifiedEngine(
            clock_period_ns=clock,
            allow_unknown_conv_hardware=args.allow_unknown_hardware,
            conv_geometry_mode=user_dma_core.CONV_GEOMETRY_QUEUE_CONFIG,
            allow_unknown_queue_config_hardware=args.allow_unknown_hardware,
            allow_unknown_gather_if8_hardware=args.allow_unknown_hardware)
        ue.software_reset()
        # The backend uploads the artifact's immutable params/program images
        # once. Per-node execution then moves only image-dependent operands,
        # and starts the resident program. CONV/MAXPOOL geometry is carried by
        # ordered CONFIG instructions; no live geometry CSR is written.
        backend = PrecompiledAndromedaBackend(
            ue, payload, timeout_s=args.timeout)
    else:
        backend = TorchBackend(quantized=True)

    with torch.inference_mode():
        started = time.perf_counter()
        raw_heads = execute_single_bin(
            payload, image_tensor, backend, progress=args.progress,
            # load_single_bin() already performed the complete schema, digest,
            # and canonical-model validation before any hardware setup.
            validate_payload=False)
        execution_elapsed = time.perf_counter() - started
        decoded = decode_yolov5(raw_heads, model)
        detections = non_max_suppression(
            decoded, model,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_det=max_detections)
        detections = restore_boxes(detections, letterbox)

    if detections:
        print("\nDetections:")
        for detection in detections:
            print("  " + _format_detection(detection))
    else:
        print("\n(no detections)")

    backend_tag = "hw" if args.backend == "hardware" else "cpu-quantized"
    if args.backend == "hardware":
        suffix = runtime["output_suffix"]
    elif profile.key == "s":
        suffix = f"_detections_{backend_tag}.jpg"
    else:
        suffix = f"_yolov5{profile.key}_detections_{backend_tag}.jpg"
    output = (args.output or resource_dir / config["paths"]["bin_dir"] /
              f"{args.image.stem}{suffix}")
    draw_detections(original, detections, output)
    print(f"Annotated image: {output}")
    print(f"Execution time: {execution_elapsed:.6f}s")
    if isinstance(backend, PrecompiledAndromedaBackend):
        total_cycles = sum(backend.cycles.values())
        fpga_execution_s = total_cycles * backend.ue._clock_period_ns / 1e9
        print(
            f"Static DRAM load: {backend.static_dram_load_seconds:.6f}s, "
            f"{backend.static_dram_load_bytes} bytes in "
            f"{backend.static_dram_load_writes} writes")
        print(
            f"FPGA execution time: {fpga_execution_s:.6f}s, "
            f"cycles={total_cycles}")

    result = {
        "model": profile.model_name.lower(),
        "decoded_text": ", ".join(detection.label for detection in detections),
        "n_detections": len(detections),
        "detections": [
            {
                "label": detection.label,
                "class_id": detection.class_id,
                "confidence": round(detection.score, 6),
                "box_xyxy": [round(value, 2) for value in detection.box],
            }
            for detection in detections
        ],
        "backend": args.backend,
        "precompiled": True,
        "artifact_version": payload["artifact_version"],
        "geometry_abi": payload["runtime"]["geometry_abi"],
        "artifact": str(args.bin.resolve()),
        "elapsed_s": round(execution_elapsed, 6),
        "execution_elapsed_s": round(execution_elapsed, 6),
    }
    if isinstance(backend, PrecompiledAndromedaBackend):
        result["hardware_version"] = f"0x{backend.hw_version:08x}"
        result["static_dram_load_s"] = round(
            backend.static_dram_load_seconds, 6)
        result["static_dram_load_bytes"] = backend.static_dram_load_bytes
        result["static_dram_load_writes"] = backend.static_dram_load_writes
        result["fpga_cycles"] = total_cycles
        result["fpga_execution_s"] = round(fpga_execution_s, 6)
    print("TEST_RESULT:" + json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
