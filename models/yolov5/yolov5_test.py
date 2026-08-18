#!/usr/bin/env python3
"""YOLOv5s v7.0 inference using Andromeda's native convolution primitives."""

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import user_dma_core
from yolov5_common import (
    AndromedaBackend,
    TorchBackend,
    decode_yolov5,
    draw_detections,
    ensure_checkpoint,
    execute_yolov5s,
    letterbox_image,
    load_official_yolov5s,
    non_max_suppression,
    restore_boxes,
)


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


def _parse_hw_versions(values) -> list[int]:
    parsed = []
    for value in values:
        text = str(value)
        parsed.append(int(text, 0 if text.lower().startswith("0x") else 16))
    return parsed


def _format_detection(detection) -> str:
    x1, y1, x2, y2 = detection.box
    return (f"{detection.score * 100:6.2f}%  {detection.label:<18} "
            f"[{x1:6.1f}, {y1:6.1f}, {x2:6.1f}, {y2:6.1f}]")


def main() -> None:
    with (SCRIPT_DIR / "yolov5_config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    defaults = config["postprocessing"]
    configured_hw_versions = set(_parse_hw_versions(
        config["hardware"]["compatible_fpga_hashes"]))
    if configured_hw_versions != set(user_dma_core.UE_NATIVE_CONV_HW_VERSIONS):
        raise RuntimeError(
            "yolov5_config.json compatible_fpga_hashes is out of sync with "
            "user_dma_core.UE_NATIVE_CONV_HW_VERSIONS")

    parser = argparse.ArgumentParser(
        description="YOLOv5s v7.0 inference on Andromeda CONV2D/MAXPOOL")
    parser.add_argument("--image", type=Path,
                        default=(SCRIPT_DIR / config["paths"]["default_image"]).resolve())
    parser.add_argument("--output", type=Path, default=None,
                        help="Annotated image path (default: yolov5_bin/<stem>_detections_<backend>.jpg)")
    parser.add_argument("--backend", choices=("hardware", "cpu", "cpu-quantized"),
                        default="hardware")
    parser.add_argument("--cpu", action="store_true",
                        help="Alias for --backend cpu")
    parser.add_argument("--image-size", type=int,
                        default=int(config["model"]["input_size"]))
    parser.add_argument("--conf-thres", type=float,
                        default=float(defaults["confidence_threshold"]))
    parser.add_argument("--iou-thres", type=float,
                        default=float(defaults["iou_threshold"]))
    parser.add_argument("--max-det", type=int,
                        default=int(defaults["max_detections"]))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dev", default="xdma0")
    parser.add_argument("--device", default="kintex7")
    parser.add_argument("--cycle", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--allow-unknown-hardware", action="store_true",
                        help="Bypass the CONV2D-capable FPGA hash allow-list")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if args.cpu:
        args.backend = "cpu"

    checkpoint = (args.checkpoint or
                  SCRIPT_DIR / config["paths"]["weights"])
    print(f"YOLOv5s v7.0 backend={args.backend} image={args.image}")
    print(f"Checkpoint: {checkpoint}")
    checkpoint = ensure_checkpoint(
        checkpoint,
        url=config["source"]["weights_url"],
        sha256=config["source"]["weights_sha256"])
    model = load_official_yolov5s(checkpoint)
    original, image_tensor, letterbox = letterbox_image(
        args.image, args.image_size)

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
        print(f"FPGA profile: {args.device}, {clock:.4f} ns, AXI={axi_width} bits")
        ue = user_dma_core.UnifiedEngine(
            clock_period_ns=clock,
            allow_unknown_conv_hardware=args.allow_unknown_hardware)
        ue.software_reset()
        backend = AndromedaBackend(
            ue,
            known_hw_versions=configured_hw_versions,
            allow_unknown_hardware=args.allow_unknown_hardware,
            timeout_s=args.timeout)
    else:
        backend = TorchBackend(quantized=args.backend == "cpu-quantized")

    started = time.perf_counter()
    with torch.inference_mode():
        raw_heads = execute_yolov5s(
            model, image_tensor, backend, progress=args.progress)
        decoded = decode_yolov5(raw_heads, model)
        detections = non_max_suppression(
            decoded, model,
            conf_threshold=args.conf_thres,
            iou_threshold=args.iou_thres,
            max_det=args.max_det)
        detections = restore_boxes(detections, letterbox)
    elapsed = time.perf_counter() - started

    if detections:
        print("\nDetections:")
        for detection in detections:
            print("  " + _format_detection(detection))
    else:
        print("\n(no detections)")

    backend_tag = "hw" if args.backend == "hardware" else args.backend
    output_suffix = (config["paths"]["output_suffix"]
                     if args.backend == "hardware"
                     else f"_detections_{backend_tag}.jpg")
    output = (args.output or
              SCRIPT_DIR / config["paths"]["bin_dir"] /
              f"{args.image.stem}{output_suffix}")
    draw_detections(original, detections, output)
    print(f"Annotated image: {output}")
    print(f"Inference time: {elapsed:.3f}s")
    if isinstance(backend, AndromedaBackend):
        total_cycles = sum(backend.cycles.values())
        print(f"Hardware: 0x{backend.hw_version:08x}, cycles={total_cycles}, "
              f"max scratch params={backend.max_params_scratch_bytes / 2**20:.1f} MiB, "
              f"tensor={backend.max_tensor_scratch_bytes / 2**20:.1f} MiB")

    labels = [detection.label for detection in detections]
    result = {
        "decoded_text": ", ".join(labels),
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
        "elapsed_s": round(elapsed, 6),
    }
    print("TEST_RESULT:" + json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
