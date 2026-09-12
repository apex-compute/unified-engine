#!/usr/bin/env python3
"""Validate every DPDFNet2 Add/Mul/Sub node on Andromeda hardware."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import user_dma_core as udc
from dpdfnet_common import DEFAULT_MODEL_PATH, download_model, validate_digest
from dpdfnet_conv_smoke import _metric, _shape, _value_info, expose_intermediates
from dpdfnet_common import initial_state


MODE = {
    "Add": udc.UE_MODE.ELTWISE_ADD,
    "Mul": udc.UE_MODE.ELTWISE_MUL,
    "Sub": udc.UE_MODE.ELTWISE_SUB,
}


def select_nodes(model):
    result = [
        (index, node) for index, node in enumerate(model.graph.node)
        if node.op_type in MODE
    ]
    if not result:
        raise RuntimeError("no Add/Mul/Sub nodes found")
    return result


def run_eltwise(engine, a, b, op_type, *, timeout_s):
    a, b = torch.broadcast_tensors(a.float(), b.float())
    shape = tuple(int(value) for value in a.shape)
    elements = a.numel()
    padded_elements = (
        (elements + udc.UE_VECTOR_SIZE - 1) // udc.UE_VECTOR_SIZE
        * udc.UE_VECTOR_SIZE)
    a_pad = torch.zeros(padded_elements, dtype=torch.bfloat16)
    b_pad = torch.zeros(padded_elements, dtype=torch.bfloat16)
    a_pad[:elements] = a.flatten().to(torch.bfloat16)
    b_pad[:elements] = b.flatten().to(torch.bfloat16)

    def upload(value):
        nbytes = value.numel() * 2
        address = engine.allocate_tensor_dram(nbytes)
        if engine.dma_write(
                engine.h2c_device, address, value, nbytes) != nbytes:
            raise RuntimeError("short DPDFNet eltwise DMA upload")
        return address

    a_addr = upload(a_pad)
    b_addr = upload(b_pad)
    output_addr = engine.allocate_tensor_dram(padded_elements * 2)
    engine.start_capture()
    engine.eltwise_core_dram(
        M=padded_elements // udc.UE_VECTOR_SIZE, N=udc.UE_VECTOR_SIZE,
        dram_a=a_addr, dram_b=b_addr, dram_out=output_addr,
        mode=MODE[op_type])
    engine.generate_instruction_halt()
    engine.stop_capture()
    program_addr = engine.get_program_dram_addr()
    engine.write_captured_instructions_to_dram(program_addr)
    engine.allocate_program_dram(engine.get_capture_instruction_size_bytes())

    started = time.perf_counter()
    engine.start_execute_from_dram(program_addr)
    engine.wait_queue(timeout_s)
    if engine.is_queue_busy():
        raise TimeoutError(f"DPDFNet {op_type} program did not finish")
    elapsed = time.perf_counter() - started
    cycles = int(engine.read_latency_cycles())
    engine.clear_capture_buffer()

    result = torch.empty_like(a_pad)
    nbytes = result.numel() * 2
    if engine.dma_read(
            engine.c2h_device, output_addr, result, nbytes) != nbytes:
        raise RuntimeError("short DPDFNet eltwise result read")
    return result[:elements].reshape(shape), a_pad[:elements].reshape(shape), \
        b_pad[:elements].reshape(shape), cycles, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--device", choices=("bittware", "bittware_512", "efinix"),
        default="bittware_512")
    parser.add_argument("--dev", default="xdma0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-relative-rmse", type=float, default=0.02)
    args = parser.parse_args()

    try:
        import onnx
        import onnxruntime as ort
        from onnx import numpy_helper
    except ImportError:
        parser.error(
            "onnx and onnxruntime are required; install "
            "models/dpdfnet/requirements.txt")

    model_path = args.model.expanduser().resolve()
    if args.download:
        model_path = download_model(model_path)
    elif not model_path.is_file():
        parser.error(f"model not found: {model_path}; pass --download")
    digest = validate_digest(model_path)
    model = onnx.load(str(model_path), load_external_data=False)
    inferred = onnx.shape_inference.infer_shapes(model)
    selected = select_nodes(model)
    initializers = {
        value.name: torch.from_numpy(np.array(
            numpy_helper.to_array(value), dtype=np.float32, copy=True))
        for value in model.graph.initializer
    }
    runtime_names = tuple(dict.fromkeys(
        name for _, node in selected for name in (*node.input, node.output[0])
        if name not in initializers))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        expose_intermediates(onnx, model, runtime_names), sess_options=options,
        providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(args.seed)
    spec = rng.normal(0.0, 0.25, (1, 1, 161, 2)).astype(np.float32)
    state = initial_state(session.get_modelmeta().custom_metadata_map)
    values = {
        name: torch.from_numpy(np.array(value, dtype=np.float32, copy=True))
        for name, value in zip(runtime_names, session.run(
            list(runtime_names), {"spec": spec, "state_in": state}))
    }
    values.update(initializers)

    udc.set_dma_device("efinix" if args.device == "efinix" else args.dev)
    clock_ns = udc.configure_clock_from_hardware()
    hardware = udc.configured_hardware_info()
    if hardware.axi_data_width_bits not in (256, 512):
        raise RuntimeError("DPDFNet requires AXI-256 or AXI-512")
    engine = udc.UnifiedEngine(clock_period_ns=clock_ns)
    engine.software_reset(run_dram_self_test=False)

    reports = []
    for node_index, node in selected:
        actual, a_bf16, b_bf16, cycles, host_elapsed = run_eltwise(
            engine, values[node.input[0]], values[node.input[1]],
            node.op_type, timeout_s=args.timeout)
        if node.op_type == "Add":
            reference = a_bf16 + b_bf16
        elif node.op_type == "Mul":
            reference = a_bf16 * b_bf16
        else:
            reference = a_bf16 - b_bf16
        reference = reference.to(torch.bfloat16)
        error = _metric(actual, reference)
        report = {
            "node_index": node_index,
            "node_type": node.op_type,
            "node_name": node.name,
            "input_shapes": [list(values[name].shape) for name in node.input],
            "output_shape": list(_shape(_value_info(inferred, node.output[0]))),
            "fpga_cycles": cycles,
            "fpga_execution_s": cycles * clock_ns * 1e-9,
            "host_elapsed_s": host_elapsed,
            "hardware_vs_bf16_reference": error,
            "bf16_reference_vs_onnx_fp32": _metric(
                reference, values[node.output[0]]),
        }
        reports.append(report)
        print("ELTWISE_RESULT:" + json.dumps(report), flush=True)
        if not torch.isfinite(actual).all():
            raise RuntimeError(f"node {node_index} produced non-finite values")
        if error["relative_rmse"] > args.max_relative_rmse:
            raise RuntimeError(
                f"node {node_index} relative RMSE "
                f"{error['relative_rmse']:.6f} exceeds "
                f"{args.max_relative_rmse:.6f}")

    summary = {
        "model": "dpdfnet2",
        "test": "all_add_mul_sub_nodes",
        "onnx_sha256": digest,
        "axi_data_width_bits": hardware.axi_data_width_bits,
        "passed": len(reports),
        "add": sum(item["node_type"] == "Add" for item in reports),
        "mul": sum(item["node_type"] == "Mul" for item in reports),
        "sub": sum(item["node_type"] == "Sub" for item in reports),
        "total_fpga_cycles": sum(item["fpga_cycles"] for item in reports),
        "max_hardware_relative_rmse": max(
            item["hardware_vs_bf16_reference"]["relative_rmse"]
            for item in reports),
        "nodes": reports,
    }
    print("TEST_RESULT:" + json.dumps(summary))


if __name__ == "__main__":
    main()
