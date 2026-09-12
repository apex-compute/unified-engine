#!/usr/bin/env python3
"""Profile DPDFNet operations using repeated resident-program prefixes.

This is diagnostic timing, not a streaming benchmark. Each replay restores
the initial recurrent state and the same input, then temporarily replaces an
operation boundary with HALT. The circular trace supplies elapsed operation
times, including instruction dispatch and DMA, at 16-clock resolution. The
original program and initial state are restored when profiling finishes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import sys

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for search_path in (HERE, ROOT, ROOT / "models" / "yolov5s"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import user_dma_core as udc
from dpdfnet_precompiled import TensorLayout, WholeGraphBackend, load_artifact, pack_tensor
from dpdfnet_run_from_bin import _load_frames
from read_trace import read_trace_addresses
from yolov5_common import configure_hardware_runtime


def operation_groups(operations, trace_size=udc.UE_TRACE_SIZE):
    """Keep both endpoints of every operation inside one retained trace."""
    groups, group, cursor = [], [], 0
    for index, operation in enumerate(operations):
        start, stop = operation["start"], operation["stop"]
        if operation["node_index"] != index or start != cursor or stop <= start:
            raise ValueError("operation ranges are not contiguous and complete")
        if stop - start >= trace_size:
            raise ValueError(f"operation {index} exceeds the trace capacity")
        if group and stop - group[0]["start"] >= trace_size:
            groups.append(group)
            group = []
        group.append(operation)
        cursor = stop
    if group:
        groups.append(group)
    if not groups:
        raise ValueError("operation manifest is empty")
    return groups


def halt_patch(program, stop):
    """Replace one descriptor while preserving its aligned 64-byte pair."""
    if stop < 0 or (stop + 1) * 32 > len(program):
        raise ValueError("HALT patch extends past the resident program")
    offset = (stop // 2) * 64
    original = program[offset:offset + 64]
    within_pair = (stop % 2) * 32
    if (original[within_pair + 1] & 15) == udc.INSTRUCTION_HALT:
        # An odd instruction count can end with a lone 32-byte HALT.
        # The final replay runs the original program without patching it.
        return offset, original, original
    if len(original) != 64:
        raise ValueError("HALT patch extends past the resident program")
    patched = bytearray(original)
    patched[within_pair:within_pair + 32] = struct.pack(
        "<8I", (stop & 255) | (udc.INSTRUCTION_HALT << 8), *([0] * 7))
    return offset, original, bytes(patched)


def read_boundaries(engine, group):
    stop = group[-1]["stop"]
    expected_pointer = (stop + 1) % udc.UE_TRACE_SIZE
    pointer = int(engine.read_reg32(udc.UE_TRACE_BRAM_ADDR))
    if pointer != expected_pointer:
        raise RuntimeError(
            f"trace write pointer {pointer}, expected {expected_pointer}; "
            "firmware trace format or executed program differs")
    boundaries = [group[0]["start"], *[entry["stop"] for entry in group]]
    oldest = max(0, stop + 1 - udc.UE_TRACE_SIZE)
    if boundaries[0] < oldest or len(set(boundaries)) != len(boundaries):
        raise RuntimeError("operation endpoints are missing from the trace")
    values = read_trace_addresses(
        engine, [index % udc.UE_TRACE_SIZE for index in boundaries])
    if any(b < a for a, b in zip(values, values[1:])):
        raise RuntimeError("trace timestamps are not monotonic (wrap or format mismatch)")
    return dict(zip(boundaries, values))


def profile(backend, spectrum, clock_ns):
    hardware, engine = backend.hardware, backend.ue
    image = hardware["model_image"]
    program_offset = hardware["program_offset"]
    program = image[program_offset:program_offset + hardware["program_size"]].numpy().tobytes()
    types = [(program[index + 1] & 15) for index in range(0, len(program), 32)]
    # PBI/control-flow instructions need additional retirement interpretation.
    allowed = {udc.INSTRUCTION_UE_OP, udc.INSTRUCTION_CONFIG,
               udc.INSTRUCTION_NOP, udc.INSTRUCTION_HALT}
    if set(types) - allowed:
        raise RuntimeError("prefix profiler requires a linear, non-prefetchable program")
    groups = operation_groups(hardware["operations"])
    source = TensorLayout.from_manifest(hardware["tensors"]["spec"])
    state = TensorLayout.from_manifest(hardware["tensors"]["state_in"])
    state_offset = state.address - hardware["model_base"]
    initial_state = image[state_offset:state_offset + state.size_bytes]
    packed = pack_tensor(spectrum, source)
    records, replays = [], []

    def write(address, data, size):
        if engine.dma_write(engine.h2c_device, address, data, size) != size:
            raise RuntimeError(f"short profiling DMA write at {address:#x}")

    try:
        for group_index, group in enumerate(groups):
            if engine.is_queue_busy():
                raise RuntimeError("cannot patch a running resident program")
            write(state.address, initial_state, state.size_bytes)
            write(source.address, packed, source.size_bytes)
            stop = group[-1]["stop"]
            offset, original, patched = halt_patch(program, stop)
            address = hardware["program_address"] + offset
            try:
                if patched != original:
                    write(address, patched, len(patched))
                engine.write_reg32(udc.UE_INT_REG, 1)
                engine.start_execute_from_dram(hardware["program_address"])
                backend._wait()
                cycles = int(engine.read_latency_cycles())
                ticks = read_boundaries(engine, group)
                for entry in group:
                    elapsed_cycles = (
                        ticks[entry["stop"]] - ticks[entry["start"]]) \
                        * udc.UE_PIPELINE_COUNTER_CLK_DIV
                    records.append({**entry, "cycles": elapsed_cycles,
                                    "elapsed_us": elapsed_cycles * clock_ns / 1000})
                replays.append({
                    "first_operation": group[0]["node_index"],
                    "last_operation": group[-1]["node_index"],
                    "halt_instruction": stop,
                    "prefix_cycles": cycles,
                    "first_boundary_ticks": ticks[group[0]["start"]],
                    "halt_ticks": ticks[stop],
                })
                print(f"Profile {group_index + 1}/{len(groups)}: operations "
                      f"{group[0]['node_index']}..{group[-1]['node_index']}", flush=True)
            finally:
                # A timeout must not leave an early HALT in the resident bin.
                if engine.is_queue_busy():
                    engine.software_reset(run_dram_self_test=False)
                if patched != original:
                    write(address, original, len(original))
    finally:
        if engine.is_queue_busy():
            engine.software_reset(run_dram_self_test=False)
        write(state.address, initial_state, state.size_bytes)

    if [entry["node_index"] for entry in records] != list(range(len(hardware["operations"]))):
        raise RuntimeError("profiling did not cover every operation exactly once")
    operation_us = sum(entry["elapsed_us"] for entry in records)
    full_us = replays[-1]["prefix_cycles"] * clock_ns / 1000
    for entry in records:
        entry["percent_of_operation_time"] = 100 * entry["elapsed_us"] / operation_us
    return {
        "method": "initial-state prefix replays; decode-boundary elapsed times include dispatch and DMA",
        "benchmark": False,
        "program_sha256": hardware["program_sha256"],
        "clock_period_ns": clock_ns,
        "trace_size": udc.UE_TRACE_SIZE,
        "counter_clock_divider": udc.UE_PIPELINE_COUNTER_CLK_DIV,
        "operation_time_us": operation_us,
        "full_program_time_us": full_us,
        "full_program_minus_operation_time_us": full_us - operation_us,
        "replays": replays,
        "operations": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", type=Path,
                        default=HERE / "dpdfnet_bin" / "dpdfnet2-andromeda.bin")
    parser.add_argument("--input", type=Path, required=True,
                        help="NumPy spectrum frame or frame sequence")
    parser.add_argument("--frame", type=int, default=0,
                        help="input frame index; recurrent state always starts from the bin")
    parser.add_argument("--output", type=Path,
                        default=HERE / "dpdfnet_bin" / "profile.json")
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
