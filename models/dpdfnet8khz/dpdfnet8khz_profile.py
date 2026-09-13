#!/usr/bin/env python3
"""Profile native 8 kHz operation prefixes; this is not a streaming benchmark."""
from pathlib import Path
import argparse
import json
import math
import sys
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for search_path in (HERE, ROOT / "models" / "dpdfnet", ROOT, ROOT / "models" / "yolov5s"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import user_dma_core as udc
from dpdfnet_profile import profile
from dpdfnet8khz_precompiled import WholeGraphBackend, load_artifact, validate_runtime_hardware
from dpdfnet8khz_run_from_bin import _load_frames
from yolov5_common import configure_hardware_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", type=Path,
                        default=HERE / "dpdfnet8khz_bin" / "dpdfnet2_8khz-andromeda.bin")
    parser.add_argument("--input", type=Path, required=True,
                        help="NumPy spectrum frame or frame sequence")
    parser.add_argument("--frame", type=int, default=0,
                        help="input frame index; recurrent state always starts from the bin")
    parser.add_argument("--output", type=Path,
                        default=HERE / "dpdfnet8khz_bin" / "profile.json")
    parser.add_argument("--device", default="rk")
    parser.add_argument("--dev", default="xdma0")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    payload, frames = load_artifact(args.bin), _load_frames(args.input)
    if not 0 <= args.frame < len(frames):
        parser.error("--frame is outside the input sequence")
    clock, hardware, _ = configure_hardware_runtime(
        device=args.device, dev=args.dev, cycle_override_ns=None)
    validate_runtime_hardware(payload, hardware.axi_data_width_bits)
    engine = udc.UnifiedEngine(
        clock_period_ns=clock, conv_geometry_mode=udc.CONV_GEOMETRY_QUEUE_CONFIG)
    engine.software_reset(run_dram_self_test=False)
    backend = WholeGraphBackend(
        engine, payload, axi_data_width_bits=hardware.axi_data_width_bits,
        timeout_s=args.timeout)
    with torch.inference_mode():
        result = profile(backend, frames[args.frame], clock)
    result.update({"bin": str(args.bin.resolve()), "input": str(args.input.resolve()),
                   "input_frame_index": args.frame, "device": args.device,
                   "dev": args.dev, "axi_data_width_bits": hardware.axi_data_width_bits,
                   "hardware_version": f"0x{engine.get_hardware_version():08x}"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for entry in sorted(result["operations"], key=lambda entry: entry["cycles"], reverse=True)[:20]:
        print(f"{entry['elapsed_us']:9.2f} us {entry['percent_of_operation_time']:5.1f}% "
              f"{entry['node_index']:3d} {entry['name']}")
    print(f"Profile saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
