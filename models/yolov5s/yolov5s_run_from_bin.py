#!/usr/bin/env python3
"""Direct YOLOv5 execution from one compiled Andromeda bin."""

from __future__ import annotations

import argparse
import json
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
    available_input_resolutions,
    artifact_variant,
    artifact_model_view,
    execute_single_bin,
    load_single_bin,
    select_single_bin_profile,
)
from yolov5_common import (
    TorchBackend,
    configure_hardware_runtime,
    decode_yolov5,
    draw_detections,
    get_yolov5_variant,
    load_yolov5_config,
    letterbox_image,
    non_max_suppression,
    restore_boxes,
    sha256_file,
)
from yolov5_precompiled import WholeGraphAndromedaBackend


def _format_detection(detection) -> str:
    x1, y1, x2, y2 = detection.box
    return (f"{detection.score * 100:6.2f}%  {detection.label:<18} "
            f"[{x1:6.1f}, {y1:6.1f}, {x2:6.1f}, {y2:6.1f}]")


def _cap_hardware_host_threads() -> None:
    """Bound intra-op parallelism before direct hardware tensor work."""
    current = torch.get_num_threads()
    target = min(current, 4)
    if target != current:
        torch.set_num_threads(target)


def main(argv=None, *, pinned_variant: str = "s",
         config_path: Path | None = None) -> None:
    model_name = get_yolov5_variant(pinned_variant).model_name
    parser = argparse.ArgumentParser(
        description=f"Run {model_name} from one compiled Andromeda bin")
    parser.set_defaults(variant=pinned_variant)
    parser.add_argument("--bin", type=Path, default=None)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--resolution", default=None, metavar="WIDTHxHEIGHT",
        help="Select an exact precompiled input profile embedded in the bin")
    parser.add_argument(
        "--list-resolutions", action="store_true",
        help="List the precompiled input profiles and exit")
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
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    if args.cpu:
        args.backend = "cpu-quantized"
    if args.backend == "hardware":
        # Bound the host work used to pack the single input and unpack the
        # bundled final output. Preserve smaller user settings.
        _cap_hardware_host_threads()

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
    resolutions = available_input_resolutions(payload)
    if args.list_resolutions:
        print("\n".join(resolutions))
        return
    resolution, selected_payload = select_single_bin_profile(
        payload, args.resolution)
    model = artifact_model_view(payload)
    runtime = payload["runtime"]
    defaults = runtime["postprocessing"]
    conf_threshold = (float(defaults["confidence_threshold"])
                      if args.conf_thres is None else args.conf_thres)
    iou_threshold = (float(defaults["iou_threshold"])
                     if args.iou_thres is None else args.iou_thres)
    max_detections = (int(defaults["max_detections"])
                      if args.max_det is None else args.max_det)
    load_elapsed = time.perf_counter() - load_started
    print(f"{profile.model_name} direct-bin backend={args.backend} "
          f"resolution={resolution} image={args.image}")
    print(f"Single bin: {args.bin.resolve()}")
    print(f"  sha256={sha256_file(args.bin.resolve())} load={load_elapsed:.3f}s")
    original, image_tensor, letterbox = letterbox_image(
        args.image, tuple(selected_payload["model"]["input_shape"][1:]))

    if args.backend == "hardware":
        clock, hw_info, detected_clock = configure_hardware_runtime(
            device=args.device, dev=args.dev,
            cycle_override_ns=args.cycle)
        print(f"FPGA profile: {args.device}, {detected_clock:.4f} ns, "
              f"AXI={hw_info.axi_data_width_bits} bits")
        if args.cycle is not None:
            print(f"FPGA clock override: {clock:.4f} ns "
                  f"(HW_INFO={detected_clock:.4f} ns)")
        ue = user_dma_core.UnifiedEngine(
            clock_period_ns=clock,
            conv_geometry_mode=user_dma_core.CONV_GEOMETRY_QUEUE_CONFIG)
        ue.software_reset(run_dram_self_test=False)
        # Backend initialization uploads the immutable model/program image
        # once. Each execute() call then performs one packed-input upload, one
        # whole-program kick, and one bundled final-output read.
        backend = WholeGraphAndromedaBackend(
            ue, selected_payload,
            axi_data_width_bits=hw_info.axi_data_width_bits,
            timeout_s=args.timeout)
    else:
        backend = TorchBackend(quantized=True)

    with torch.inference_mode():
        started = time.perf_counter()
        if args.backend == "hardware":
            raw_heads = backend.execute(image_tensor)
        else:
            raw_heads = execute_single_bin(
                selected_payload, image_tensor, backend,
                progress=args.progress,
                # load_single_bin() already performed the complete schema,
                # digest, and canonical-model validation before setup.
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
              f"{args.image.stem}_{resolution}{suffix}")
    draw_detections(original, detections, output)
    print(f"Annotated image: {output}")
    print(f"Execution time: {execution_elapsed:.6f}s")
    hardware_accounting = None
    if args.backend == "hardware":
        # The compatibility fallbacks keep lightweight mocked backends useful;
        # the production whole-graph backend exposes every counter directly.
        hardware_accounting = {
            "model_upload_writes": int(getattr(
                backend, "model_upload_writes",
                getattr(backend, "static_dram_load_writes", 1))),
            "input_upload_writes": int(getattr(
                backend, "input_upload_writes", 1)),
            "program_kicks": int(getattr(backend, "program_kicks", 1)),
            "output_reads": int(getattr(backend, "output_reads", 1)),
            "intermediate_upload_writes": int(getattr(
                backend, "intermediate_upload_writes", 0)),
            "intermediate_output_reads": int(getattr(
                backend, "intermediate_output_reads", 0)),
        }
        expected_accounting = {
            "model_upload_writes": 1,
            "input_upload_writes": 1,
            "program_kicks": 1,
            "output_reads": 1,
            "intermediate_upload_writes": 0,
            "intermediate_output_reads": 0,
        }
        if hardware_accounting != expected_accounting:
            raise RuntimeError(
                "whole-graph hardware transaction contract was violated: "
                f"{hardware_accounting}")
        total_cycles = sum(backend.cycles.values())
        fpga_execution_s = total_cycles * backend.ue._clock_period_ns / 1e9
        model_upload_seconds = float(getattr(
            backend, "model_upload_seconds",
            getattr(backend, "static_dram_load_seconds", 0.0)))
        model_upload_bytes = int(getattr(
            backend, "model_upload_bytes",
            getattr(backend, "static_dram_load_bytes", 0)))
        print(
            f"Model/program image load: {model_upload_seconds:.6f}s, "
            f"{model_upload_bytes} bytes in "
            f"{hardware_accounting['model_upload_writes']} write")
        print(
            f"FPGA execution time: {fpga_execution_s:.6f}s, "
            f"cycles={total_cycles}")
        print(
            "Hardware accounting: full_graph=true, "
            + ", ".join(
                f"{key}={value}" for key, value
                in hardware_accounting.items()))

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
        "input_resolution": resolution,
        "input_shape": list(selected_payload["model"]["input_shape"]),
        "artifact": str(args.bin.resolve()),
        "elapsed_s": round(execution_elapsed, 6),
        "execution_elapsed_s": round(execution_elapsed, 6),
        "full_graph": args.backend == "hardware",
    }
    if args.backend == "hardware":
        result["hardware_version"] = f"0x{backend.hw_version:08x}"
        result["model_upload_s"] = round(model_upload_seconds, 6)
        result["model_upload_bytes"] = model_upload_bytes
        result.update(hardware_accounting)
        # Preserve the previous static-load names for existing consumers.
        result["static_dram_load_s"] = round(model_upload_seconds, 6)
        result["static_dram_load_bytes"] = model_upload_bytes
        result["static_dram_load_writes"] = hardware_accounting[
            "model_upload_writes"]
        result["fpga_cycles"] = total_cycles
        result["fpga_execution_s"] = round(fpga_execution_s, 6)
    print("TEST_RESULT:" + json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
