#!/usr/bin/env python3
"""Check compiled DPDFNet packed-state boundaries and their neighboring values.

Uploads one resident image, copies the actual 966/2880/1610-element state
segments at offsets 39968/40934/43814, then slices them back using the
production state-copy lowering. Every BF16 value, sentinel, and padding lane
must be preserved exactly.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for search_path in (HERE, ROOT, ROOT / "models" / "yolov5s"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import user_dma_core as udc
from dpdfnet_compile import GraphCompiler
from dpdfnet_precompiled import (
    DeviceEmitter, MODEL_BASE, TENSOR_BASE, make_layout, pack_tensor,
)
from dpdfnet_sqrt_smoke import wait_for_halt
from yolov5_common import configure_hardware_runtime


def compile_state_copy_program():
    state = make_layout("state_in", (45424,), MODEL_BASE)
    initial = (torch.arange(state.logical_elements) % 127 - 63).to(torch.bfloat16)
    expected = initial.clone()
    layouts = {"state_in": state}
    segments = []
    cursor = TENSOR_BASE
    for index, (start, count) in enumerate(((39968, 966), (40934, 2880), (43814, 1610))):
        name = f"segment_{index}"
        value = ((torch.arange(count) % 61 - 30).float() / 8 + index * 10).to(
            torch.bfloat16)
        output = make_layout(f"slice_{index}", (count,), cursor)
        layouts[output.name] = output
        cursor += output.size_bytes
        segments.append({"name": name, "start": start, "value": value, "output": output})
        expected[start:start + count] = value
    scratch = make_layout("scratch", (64,), cursor)
    layouts[scratch.name] = scratch
    emitter = DeviceEmitter(layouts, cursor + scratch.size_bytes, initial)
    compiler = GraphCompiler.__new__(GraphCompiler)
    compiler.emitter = emitter
    compiler.state_copy_scratch = scratch
    copies, slices = [], []
    for segment in segments:
        probe = make_layout(segment["name"], segment["value"].shape, 0)
        address = emitter.allocate_constant(segment["value"], probe)
        source = make_layout(segment["name"], segment["value"].shape, address)
        copies.append(compiler.prepare_state_copy(
            source, state, 0, segment["start"], source.logical_elements))
        slices.append(compiler.prepare_state_copy(
            state, segment["output"], segment["start"], 0, source.logical_elements))
    program_address = emitter.begin_program()
    for copy in copies:
        compiler.emit_state_copy(copy)
    for segment, copy in zip(segments, slices):
        emitter.emit_zero(segment["output"])
        compiler.emit_state_copy(copy)
    program = emitter.finish_program(program_address)
    emitter.image.align(64)
    image = torch.frombuffer(emitter.image.data, dtype=torch.uint8).clone()
    expected_tensors = [(state, pack_tensor(expected, state))] + [
        (segment["output"], pack_tensor(segment["value"], segment["output"]))
        for segment in segments
    ]
    return image, program_address, len(program), expected_tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", default="xdma0")
    parser.add_argument("--device", default="rk")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    clock_ns, hardware, _ = configure_hardware_runtime(
        device=args.device, dev=args.dev, cycle_override_ns=None)
    image, program_address, program_size, expected_tensors = compile_state_copy_program()
    engine = udc.UnifiedEngine(
        clock_period_ns=clock_ns,
        conv_geometry_mode=udc.CONV_GEOMETRY_QUEUE_CONFIG)
    engine.software_reset(run_dram_self_test=False)
    if engine.dma_write(
            engine.h2c_device, MODEL_BASE, image, image.numel()) != image.numel():
        raise RuntimeError("short state-copy model/program upload")
    engine.write_reg32(udc.UE_INT_REG, 1)
    engine.start_execute_from_dram(program_address)
    wait_for_halt(engine, args.timeout)
    cycles = int(engine.read_latency_cycles())
    reports = []
    for layout, expected in expected_tensors:
        actual = torch.empty(layout.physical_elements, dtype=torch.bfloat16)
        if engine.dma_read(
                engine.c2h_device, layout.address, actual,
                layout.size_bytes) != layout.size_bytes:
            raise RuntimeError(f"{layout.name}: short result read")
        if not torch.equal(actual, expected):
            indices = torch.nonzero(actual != expected).flatten()
            first = int(indices[0])
            raise AssertionError(
                f"{layout.name}: {indices.numel()} mismatches; first at {first}: "
                f"actual={float(actual[first])}, expected={float(expected[first])}")
        report = {"tensor": layout.name, "verified_bf16_elements": layout.physical_elements}
        reports.append(report)
        print("STATE_COPY_RESULT:" + json.dumps(report), flush=True)
    print("TEST_RESULT:" + json.dumps({
        "test": "dpdfnet_packed_state_copy",
        "passed": len(reports),
        "axi_data_width_bits": hardware.axi_data_width_bits,
        "program_size_bytes": program_size,
        "model_upload_writes": 1,
        "fpga_cycles": cycles,
        "fpga_execution_s": cycles * clock_ns * 1e-9,
        "tensors": reports,
    }))


if __name__ == "__main__":
    main()
