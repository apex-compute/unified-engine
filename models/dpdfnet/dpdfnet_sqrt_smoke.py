#!/usr/bin/env python3
"""Check DPDFNet's compiled square root on zero and positive BF16 energies.

Compiles the production GraphCompiler.emit_sqrt lowering into one resident
program, uploads it once, and replays it on three (1, 1, 96) test tensors.
The physical layout includes the same 32 padding lanes as DPDFNet's energy
tensor. CPU square roots are used only as test references.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for search_path in (HERE, ROOT, ROOT / "models" / "yolov5s"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import user_dma_core as udc
from dpdfnet_compile import GraphCompiler
from dpdfnet_precompiled import (
    DeviceEmitter, INPUT_BASE, MODEL_BASE, TENSOR_BASE,
    make_layout, pack_tensor, unpack_tensor,
)
from yolov5_common import configure_hardware_runtime


def compile_sqrt_program():
    """Build the production lowering without opening a device or an ONNX file."""
    shape = (1, 1, 96)
    source = make_layout("input", shape, INPUT_BASE)
    output = make_layout("output", shape, TENSOR_BASE)
    inverse = make_layout("inverse", shape, TENSOR_BASE + output.size_bytes)
    layouts = {
        "state_in": make_layout("state_in", (64,), MODEL_BASE),
        "input": source,
        "output": output,
        "inverse": inverse,
    }
    emitter = DeviceEmitter(
        layouts, inverse.address + inverse.size_bytes,
        torch.zeros(64, dtype=torch.bfloat16))
    identity_address = emitter.allocate_constant(
        torch.eye(udc.UE_VECTOR_SIZE, dtype=torch.bfloat16))

    # Only the node's tensor layouts and identity constant are needed by the
    # actual lowering; the full constructor plans the entire DPDFNet graph.
    compiler = GraphCompiler.__new__(GraphCompiler)
    compiler.layouts = layouts
    compiler.initializer_layouts = {}
    compiler.unary_aux = {0: inverse}
    compiler.emitter = emitter
    compiler.identity_address = identity_address
    node = SimpleNamespace(input=["input"], output=["output"])

    program_address = emitter.begin_program()
    compiler.emit_sqrt(0, node)
    program = emitter.finish_program(program_address)
    emitter.image.align(64)
    image = torch.frombuffer(emitter.image.data, dtype=torch.uint8).clone()
    return image, program_address, len(program), source, output


def test_cases():
    tiny = float(torch.finfo(torch.bfloat16).tiny)
    values = torch.tensor([
        tiny, 2 * tiny, 1e-30, 1e-20, 1e-12, 1e-6, 0.01, 0.25,
        1.0, 2.0, 4.0, 9.0, 16.0, 1e6, 1e12, 1e30,
    ], dtype=torch.bfloat16).repeat(6).reshape(1, 1, 96)
    alternating = values.clone()
    alternating[..., 1::2] = 0
    return (
        ("all_zero", torch.zeros_like(values)),
        ("alternating_zero_positive", alternating),
        ("positive_normal_range", values),
    )


def wait_for_halt(engine, timeout_s):
    deadline = time.monotonic() + timeout_s
    while (engine.read_reg32(udc.UE_INT_REG) & 3) != udc.INT_CAUSE_HALT:
        if time.monotonic() >= deadline:
            raise TimeoutError("DPDFNet square-root program did not reach HALT")
        time.sleep(0.0001)
    while engine.is_queue_busy():
        if time.monotonic() >= deadline:
            raise TimeoutError("DPDFNet square-root queue stayed busy after HALT")
        time.sleep(0.001)


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
    image, program_address, program_size, source, output = compile_sqrt_program()
    engine = udc.UnifiedEngine(
        clock_period_ns=clock_ns,
        conv_geometry_mode=udc.CONV_GEOMETRY_QUEUE_CONFIG)
    engine.software_reset(run_dram_self_test=False)
    if engine.dma_write(
            engine.h2c_device, MODEL_BASE, image, image.numel()) != image.numel():
        raise RuntimeError("short square-root model/program upload")

    reports = []
    for name, value in test_cases():
        packed = pack_tensor(value, source)
        if engine.dma_write(
                engine.h2c_device, source.address,
                packed, source.size_bytes) != source.size_bytes:
            raise RuntimeError(f"{name}: short input upload")
        engine.write_reg32(udc.UE_INT_REG, 1)
        engine.start_execute_from_dram(program_address)
        wait_for_halt(engine, args.timeout)
        cycles = int(engine.read_latency_cycles())
        flat = torch.empty(output.physical_elements, dtype=torch.bfloat16)
        if engine.dma_read(
                engine.c2h_device, output.address,
                flat, output.size_bytes) != output.size_bytes:
            raise RuntimeError(f"{name}: short result read")

        actual = unpack_tensor(flat, output).float()
        reference = value.float().sqrt()
        if not torch.isfinite(flat).all():
            raise AssertionError(f"{name}: nonfinite output or padding")
        zero = value == 0
        if not (actual[zero] == 0).all():
            raise AssertionError(f"{name}: sqrt(0) is not exactly zero")
        padding = flat.reshape(output.rows, output.padded_last)[
            :, output.logical_last:]
        if not (padding == 0).all():
            raise AssertionError(f"{name}: output padding is not zero")
        positive = ~zero
        relative_error = (
            (actual[positive] - reference[positive]).abs() / reference[positive])
        max_relative_error = (
            float(relative_error.max()) if relative_error.numel() else 0.0)
        if max_relative_error > 0.02:
            raise AssertionError(
                f"{name}: maximum relative sqrt error {max_relative_error:.6f} "
                "exceeds 0.02")
        report = {
            "case": name,
            "zero_elements": int(zero.sum()),
            "positive_elements": int(positive.sum()),
            "padding_elements": int(padding.numel()),
            "max_relative_error": max_relative_error,
            "fpga_cycles": cycles,
            "fpga_execution_s": cycles * clock_ns * 1e-9,
        }
        reports.append(report)
        print("SQRT_RESULT:" + json.dumps(report), flush=True)

    print("TEST_RESULT:" + json.dumps({
        "test": "dpdfnet_compiled_sqrt",
        "passed": len(reports),
        "axi_data_width_bits": hardware.axi_data_width_bits,
        "logical_shape": list(source.shape),
        "program_size_bytes": program_size,
        "model_upload_writes": 1,
        "model_upload_bytes": image.numel(),
        "cases": reports,
    }))


if __name__ == "__main__":
    main()
