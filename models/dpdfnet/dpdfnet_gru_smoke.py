#!/usr/bin/env python3
"""Compare compiled SRAM GRUs with the original DRAM lowering on the FPGA."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import sys

import onnx
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


def emit_dram_reference(compiler, index, node, output, output_h):
    """Original per-timestep implementation, with every gate stored in DRAM."""
    source, weights, recurrent, biases, initial = (
        compiler.layout(node.input[position]) for position in (0, 1, 2, 3, 5))
    sequence = source.shape[0]
    scratch = compiler.gru_aux[index]
    xg, hg, z, r, candidate, temporary, hidden = (
        scratch[name].address for name in ("xg", "hg", "z", "r", "candidate", "tmp", "h"))
    for direction in range(2):
        current_h = initial.address + direction * 128
        order = range(sequence) if direction == 0 else range(sequence - 1, -1, -1)
        for timestep in order:
            compiler._emit_gru_matmul(
                source.address + timestep * 128,
                weights.address + direction * 24576,
                biases.address + direction * 768, xg)
            compiler._emit_gru_matmul(
                current_h, recurrent.address + direction * 24576,
                biases.address + direction * 768 + 384, hg)
            compiler._emit_gru_eltwise(xg, hg, temporary, udc.UE_MODE.ELTWISE_ADD)
            compiler._emit_gru_activation(temporary, z, "sigmoid")
            compiler._emit_gru_eltwise(xg + 128, hg + 128, temporary,
                                       udc.UE_MODE.ELTWISE_ADD)
            compiler._emit_gru_activation(temporary, r, "sigmoid")
            compiler._emit_gru_eltwise(hg + 256, r, temporary,
                                       udc.UE_MODE.ELTWISE_MUL)
            compiler._emit_gru_eltwise(xg + 256, temporary, candidate,
                                       udc.UE_MODE.ELTWISE_ADD)
            compiler._emit_gru_activation(candidate, candidate, "tanh")
            compiler._emit_gru_eltwise(current_h, candidate, temporary,
                                       udc.UE_MODE.ELTWISE_SUB)
            compiler._emit_gru_eltwise(temporary, z, temporary, udc.UE_MODE.ELTWISE_MUL)
            compiler._emit_gru_eltwise(temporary, candidate, hidden, udc.UE_MODE.ELTWISE_ADD)
            current_h = hidden
            compiler._emit_row_copy(hidden, output.address + (timestep * 2 + direction) * 128)
        compiler._emit_row_copy(current_h, output_h.address + direction * 128)


def compile_gru_program():
    layouts = {"state_in": make_layout("state_in", (64,), MODEL_BASE)}
    compiler = GraphCompiler.__new__(GraphCompiler)
    compiler.onnx = onnx
    compiler.layouts = layouts
    compiler.initializer_layouts = {}
    compiler.gru_aux = {}
    cursor = TENSOR_BASE

    def allocate(name, shape):
        nonlocal cursor
        layout = make_layout(name, shape, cursor)
        cursor += layout.size_bytes
        layouts[name] = layout
        return layout

    cases = []
    for index, sequence in enumerate((8, 17, 48)):
        prefix = f"gru_{sequence}"
        output = allocate(f"{prefix}/Y", (sequence, 2, 1, 64))
        final = allocate(f"{prefix}/Y_h", (2, 1, 64))
        reference = allocate(f"{prefix}/reference", output.shape)
        reference_h = allocate(f"{prefix}/reference_h", final.shape)
        compiler.gru_aux[index] = {
            name: allocate(f"{prefix}/{name}", shape)
            for name, shape in (
                ("xg", (sequence, 192)), ("hg", (1, 192)),
                *((name, (1, 64)) for name in ("z", "r", "candidate", "tmp", "h")),
            )
        }
        node = onnx.helper.make_node(
            "GRU", [f"{prefix}/{name}" for name in ("X", "W", "R", "B", "length", "H")],
            [output.name, final.name], direction="bidirectional", hidden_size=64,
            linear_before_reset=1)
        cases.append((sequence, node, output, final, reference, reference_h))
    emitter = DeviceEmitter(layouts, cursor, torch.zeros(64, dtype=torch.bfloat16))
    compiler.emitter = emitter
    compiler.identity_address = emitter.allocate_constant(torch.eye(64, dtype=torch.bfloat16))
    for sequence, node, *_ in cases:
        generator = torch.Generator().manual_seed(sequence)
        for position, shape, scale in (
                (0, (sequence, 1, 64), 0.2), (1, (2, 192, 64), 0.04),
                (2, (2, 192, 64), 0.04), (3, (2, 384), 0.1), (5, (2, 1, 64), 0.2)):
            value = (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)
            address = emitter.allocate_constant(value)
            layouts[node.input[position]] = make_layout(node.input[position], shape, address)
    program_address = emitter.begin_program()
    with contextlib.redirect_stdout(io.StringIO()):
        for index, (_, node, _, _, reference, reference_h) in enumerate(cases):
            compiler.emit_gru(index, node)
            emit_dram_reference(compiler, index, node, reference, reference_h)
    program = emitter.finish_program(program_address)
    emitter.image.align(64)
    image = torch.frombuffer(emitter.image.data, dtype=torch.uint8).clone()
    return image, program_address, len(program), cases


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
    image, program_address, program_size, cases = compile_gru_program()
    engine = udc.UnifiedEngine(clock_period_ns=clock_ns,
                               conv_geometry_mode=udc.CONV_GEOMETRY_QUEUE_CONFIG)
    engine.software_reset(run_dram_self_test=False)
    if engine.dma_write(engine.h2c_device, MODEL_BASE, image, image.numel()) != image.numel():
        raise RuntimeError("short GRU model/program upload")
    engine.write_reg32(udc.UE_INT_REG, 1)
    engine.start_execute_from_dram(program_address)
    wait_for_halt(engine, args.timeout)
    verified = 0
    for sequence, _, output, final, reference, reference_h in cases:
        for actual_layout, expected_layout in ((output, reference), (final, reference_h)):
            tensors = []
            for layout in (actual_layout, expected_layout):
                value = torch.empty(layout.physical_elements, dtype=torch.bfloat16)
                if engine.dma_read(engine.c2h_device, layout.address, value,
                                   layout.size_bytes) != layout.size_bytes:
                    raise RuntimeError(f"{layout.name}: short GRU result read")
                tensors.append(value)
            actual, expected = tensors
            if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
                raise AssertionError(f"{actual_layout.name}: nonfinite GRU values")
            if not torch.equal(actual.view(torch.uint16), expected.view(torch.uint16)):
                differences = actual.view(torch.uint16) != expected.view(torch.uint16)
                first = int(torch.nonzero(differences)[0])
                raise AssertionError(
                    f"{actual_layout.name}: {int(differences.sum())} mismatches, "
                    f"first {first}: SRAM={float(actual[first])}, DRAM={float(expected[first])}")
            verified += actual.numel()
    print("TEST_RESULT:" + json.dumps({
        "test": "dpdfnet_sram_gru", "passed": len(cases),
        "axi_data_width_bits": hardware.axi_data_width_bits,
        "program_instructions": program_size // 32,
        "verified_bf16_elements": verified,
        "model_upload_bytes": image.numel(),
    }))


if __name__ == "__main__":
    main()
