#!/usr/bin/env python3
"""YOLOv5 v7.0 inference using Andromeda's native convolution primitives."""

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
    execute_yolov5,
    get_yolov5_variant,
    load_yolov5_config,
    letterbox_image,
    load_official_yolov5,
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


def _create_hardware_backend(*, clock: float, known_hw_versions: set[int],
                             allow_unknown_hardware: bool,
                             timeout_s: float) -> AndromedaBackend:
    """Create the checkpoint runner's ordered-CONFIG hardware backend."""
    ue = user_dma_core.UnifiedEngine(
        clock_period_ns=clock,
        allow_unknown_conv_hardware=allow_unknown_hardware,
        conv_geometry_mode=user_dma_core.CONV_GEOMETRY_QUEUE_CONFIG,
        allow_unknown_queue_config_hardware=allow_unknown_hardware,
        allow_unknown_gather_if8_hardware=allow_unknown_hardware)
    ue.software_reset()
    return AndromedaBackend(
        ue,
        known_hw_versions=known_hw_versions,
        allow_unknown_hardware=allow_unknown_hardware,
        timeout_s=timeout_s)


def main(argv=None, *, pinned_variant: str = "s",
         config_path: Path | None = None) -> None:
    model_name = get_yolov5_variant(pinned_variant).model_name
    default_bin_dir = "yolov5n_bin" if pinned_variant == "n" else "yolov5_bin"
    parser = argparse.ArgumentParser(
        description=f"{model_name} v7.0 inference on Andromeda")
    parser.set_defaults(variant=pinned_variant)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None,
                        help=("Annotated image path (default: "
                              f"{default_bin_dir}/<stem>_detections_<backend>.jpg)"))
    parser.add_argument("--backend", choices=("hardware", "cpu", "cpu-quantized"),
                        default="hardware")
    parser.add_argument("--cpu", action="store_true",
                        help="Alias for --backend cpu")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--conf-thres", type=float, default=None)
    parser.add_argument("--iou-thres", type=float, default=None)
    parser.add_argument("--max-det", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dev", default="xdma0")
    parser.add_argument("--device", default="kintex7")
    parser.add_argument("--cycle", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--allow-unknown-hardware", action="store_true",
                        help="Bypass native-CONV, queue-CONFIG, and gather-IF8 FPGA hash gates")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    if args.cpu:
        args.backend = "cpu"

    profile, config, resource_dir = load_yolov5_config(
        args.variant, config_path)
    defaults = config["postprocessing"]
    if args.image is None:
        args.image = (resource_dir / config["paths"]["default_image"]).resolve()
    if args.image_size is None:
        args.image_size = int(config["model"]["input_size"])
    if args.conf_thres is None:
        args.conf_thres = float(defaults["confidence_threshold"])
    if args.iou_thres is None:
        args.iou_thres = float(defaults["iou_threshold"])
    if args.max_det is None:
        args.max_det = int(defaults["max_detections"])

    configured_native_hw_versions = set(_parse_hw_versions(
        config["hardware"]["compatible_fpga_hashes"]))
    if configured_native_hw_versions != set(user_dma_core.UE_NATIVE_CONV_HW_VERSIONS):
        raise RuntimeError(
            f"{profile.model_name} compatible_fpga_hashes is out of sync with "
            "user_dma_core.UE_NATIVE_CONV_HW_VERSIONS")
    configured_queue_hw_versions = set(_parse_hw_versions(
        config["hardware"]["queued_config_fpga_hashes"]))
    if configured_queue_hw_versions != set(user_dma_core.UE_QUEUE_CONFIG_HW_VERSIONS):
        raise RuntimeError(
            f"{profile.model_name} queued_config_fpga_hashes is out of sync with "
            "user_dma_core.UE_QUEUE_CONFIG_HW_VERSIONS")
    configured_gather_if8_hw_versions = set(_parse_hw_versions(
        config["hardware"]["gather_if8_fpga_hashes"]))
    if configured_gather_if8_hw_versions != set(
            user_dma_core.UE_GATHER_IF8_HW_VERSIONS):
        raise RuntimeError(
            f"{profile.model_name} gather_if8_fpga_hashes is out of sync with "
            "user_dma_core.UE_GATHER_IF8_HW_VERSIONS")

    checkpoint = (args.checkpoint or
                  resource_dir / config["paths"]["weights"])
    print(f"{profile.model_name} v7.0 backend={args.backend} image={args.image}")
    print(f"Checkpoint: {checkpoint}")
    checkpoint = ensure_checkpoint(
        checkpoint,
        url=config["source"]["weights_url"],
        sha256=config["source"]["weights_sha256"])
    model = load_official_yolov5(checkpoint, variant=profile.key)
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
        backend = _create_hardware_backend(
            clock=clock,
            known_hw_versions=configured_gather_if8_hw_versions,
            allow_unknown_hardware=args.allow_unknown_hardware,
            timeout_s=args.timeout)
    else:
        backend = TorchBackend(quantized=args.backend == "cpu-quantized")

    started = time.perf_counter()
    with torch.inference_mode():
        raw_heads = execute_yolov5(
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
    if args.backend == "hardware":
        output_suffix = config["paths"]["output_suffix"]
    elif profile.key == "s":
        output_suffix = f"_detections_{backend_tag}.jpg"
    else:
        output_suffix = f"_yolov5{profile.key}_detections_{backend_tag}.jpg"
    output = (args.output or
              resource_dir / config["paths"]["bin_dir"] /
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
        "model": profile.model_name.lower(),
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
    if isinstance(backend, AndromedaBackend):
        result.update({
            "geometry_abi": "conv-config-inst-v1",
            "hardware_version": f"0x{backend.hw_version:08x}",
        })
    print("TEST_RESULT:" + json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
