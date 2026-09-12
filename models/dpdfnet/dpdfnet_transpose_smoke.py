#!/usr/bin/env python3
"""Check the production partial transpose, including untouched output rows."""

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
from dpdfnet_precompiled import DeviceEmitter, MODEL_BASE, TENSOR_BASE, make_layout
from dpdfnet_sqrt_smoke import wait_for_halt
from yolov5_common import configure_hardware_runtime


def compile_transpose_program():
    dimensions = ((2, 64), (17, 64), (32, 64), (64, 64),
                  (2, 1), (2, 2), (17, 2), (32, 2), (2, 17), (32, 31), (32, 32))
    layouts = {"state_in": make_layout("state_in", (64,), MODEL_BASE)}
    outputs = []
    for index, (columns, rows) in enumerate(dimensions):
        output = make_layout(f"transpose_{columns}_{rows}", (64, 64),
                             TENSOR_BASE + index * 8192)
        layouts[output.name] = output
        outputs.append(output)
    emitter = DeviceEmitter(layouts, TENSOR_BASE + len(outputs) * 8192,
                            torch.zeros(64, dtype=torch.bfloat16))
    value = ((torch.arange(4096).reshape(64, 64) * 31) % 251 - 125).to(
        torch.bfloat16)
    value[::5, ::3] = 0
    source_address = emitter.allocate_constant(value)
    sentinel = torch.full((64, 64), -255, dtype=torch.bfloat16)
    sentinel_address = emitter.allocate_constant(sentinel)
    compiler = GraphCompiler.__new__(GraphCompiler)
    compiler.emitter = emitter
    compiler.identity_address = emitter.allocate_constant(
        torch.eye(64, dtype=torch.bfloat16))
    program_address = emitter.begin_program()
    expected_tensors = []
    for (columns, rows), output in zip(dimensions, outputs):
        emitter.engine.accelerator_memcpy(sentinel_address, output.address, 8192)
        compiler._emit_transpose64_columns(source_address, output.address, columns, rows)
        expected = sentinel.clone()
        expected[:columns] = 0
        expected[:columns, :rows] = value[:rows, :columns].T
        expected_tensors.append((output, expected.flatten()))
    program = emitter.finish_program(program_address)
    emitter.image.align(64)
    image = torch.frombuffer(emitter.image.data, dtype=torch.uint8).clone()
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
    image, program_address, program_size, expected_tensors = compile_transpose_program()
    engine = udc.UnifiedEngine(
        clock_period_ns=clock_ns,
        conv_geometry_mode=udc.CONV_GEOMETRY_QUEUE_CONFIG)
    engine.software_reset(run_dram_self_test=False)
    if engine.dma_write(
            engine.h2c_device, MODEL_BASE, image, image.numel()) != image.numel():
        raise RuntimeError("short transpose model/program upload")
    engine.write_reg32(udc.UE_INT_REG, 1)
    engine.start_execute_from_dram(program_address)
    wait_for_halt(engine, args.timeout)
    cycles = int(engine.read_latency_cycles())
    for layout, expected in expected_tensors:
        actual = torch.empty(layout.physical_elements, dtype=torch.bfloat16)
        if engine.dma_read(engine.c2h_device, layout.address, actual,
                           layout.size_bytes) != layout.size_bytes:
            raise RuntimeError(f"{layout.name}: short result read")
        if not torch.equal(actual, expected):
            indices = torch.nonzero(actual != expected).flatten()
            first = int(indices[0])
            raise AssertionError(
                f"{layout.name}: {indices.numel()} mismatches; first at {first}: "
                f"actual={float(actual[first])}, expected={float(expected[first])}")
    print("TEST_RESULT:" + json.dumps({
        "test": "dpdfnet_partial_transpose",
        "passed": len(expected_tensors),
        "axi_data_width_bits": hardware.axi_data_width_bits,
        "program_instructions": program_size // 32,
        "verified_bf16_elements": len(expected_tensors) * 4096,
        "fpga_cycles": cycles,
        "fpga_execution_s": cycles * clock_ns * 1e-9,
    }))


if __name__ == "__main__":
    main()
