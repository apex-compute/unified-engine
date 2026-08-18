#!/usr/bin/env python3
"""Direct YOLOv5s execution from one compiled Andromeda bin."""

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
    artifact_model_view,
    execute_single_bin,
    load_single_bin,
)
from yolov5_common import (
    TorchBackend,
    decode_yolov5,
    draw_detections,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run YOLOv5s directly from one compiled Andromeda bin")
    parser.add_argument("--bin", type=Path,
                        default=SCRIPT_DIR / "yolov5_bin/yolov5s-andromeda.bin")
    parser.add_argument("--image", type=Path,
                        default=REPO_ROOT / "test_samples/vette.jpg")
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
    parser.add_argument("--allow-unknown-hardware", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if args.cpu:
        args.backend = "cpu-quantized"

    load_started = time.perf_counter()
    payload = load_single_bin(args.bin)
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
    if configured_hw_versions != set(user_dma_core.UE_QUEUE_CONFIG_HW_VERSIONS):
        raise RuntimeError("single-bin queued-CONFIG metadata disagrees with the driver")
    load_elapsed = time.perf_counter() - load_started
    print(f"YOLOv5s direct-bin backend={args.backend} image={args.image}")
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
            allow_unknown_queue_config_hardware=args.allow_unknown_hardware)
        ue.software_reset()
        # The backend uploads the artifact's immutable params/program images
        # once. Per-node execution then moves only image-dependent operands,
        # and starts the resident program. CONV/MAXPOOL geometry is carried by
        # ordered CONFIG instructions; no live geometry CSR is written.
        backend = PrecompiledAndromedaBackend(
            ue, payload, timeout_s=args.timeout)
    else:
        backend = TorchBackend(quantized=True)

    started = time.perf_counter()
    with torch.inference_mode():
        raw_heads = execute_single_bin(
            payload, image_tensor, backend, progress=args.progress)
        decoded = decode_yolov5(raw_heads, model)
        detections = non_max_suppression(
            decoded, model,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_det=max_detections)
        detections = restore_boxes(detections, letterbox)
    elapsed = time.perf_counter() - started

    if detections:
        print("\nDetections:")
        for detection in detections:
            print("  " + _format_detection(detection))
    else:
        print("\n(no detections)")

    backend_tag = "hw" if args.backend == "hardware" else "cpu-quantized"
    suffix = (runtime["output_suffix"]
              if args.backend == "hardware"
              else f"_detections_{backend_tag}.jpg")
    output = (args.output or SCRIPT_DIR / "yolov5_bin" /
              f"{args.image.stem}{suffix}")
    draw_detections(original, detections, output)
    print(f"Annotated image: {output}")
    print(f"Inference time: {elapsed:.3f}s")
    if isinstance(backend, PrecompiledAndromedaBackend):
        print(f"Hardware: 0x{backend.hw_version:08x}, "
              f"cycles={sum(backend.cycles.values())}")

    result = {
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
        "elapsed_s": round(elapsed, 6),
    }
    print("TEST_RESULT:" + json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
