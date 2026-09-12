#!/usr/bin/env python3
"""Validate all DPDFNet2 Relu/Sigmoid/Tanh nodes on Andromeda hardware."""

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
from nn_lib import tanh_core_dram
from dpdfnet_common import DEFAULT_MODEL_PATH, download_model, validate_digest
from dpdfnet_conv_smoke import _metric, _shape, _value_info, expose_intermediates
from dpdfnet_common import initial_state


SUPPORTED = ("Relu", "Sigmoid", "Tanh")


def select_activation_nodes(model):
    result = [
        (index, node) for index, node in enumerate(model.graph.node)
        if node.op_type in SUPPORTED
    ]
    if not result:
        raise RuntimeError("no supported activation nodes found")
    return result


def run_activation(engine, x, op_type, identity_addr, *, timeout_s):
    original_shape = tuple(int(value) for value in x.shape)
    elements = x.numel()
    padded_elements = (
        (elements + udc.UE_VECTOR_SIZE - 1) // udc.UE_VECTOR_SIZE
        * udc.UE_VECTOR_SIZE)
    rows = padded_elements // udc.UE_VECTOR_SIZE
    padded = torch.zeros(padded_elements, dtype=torch.bfloat16)
    padded[:elements] = x.flatten().to(torch.bfloat16)

    nbytes = padded.numel() * 2
    input_addr = engine.allocate_tensor_dram(nbytes)
    output_addr = engine.allocate_tensor_dram(nbytes)
    if engine.dma_write(
            engine.h2c_device, input_addr, padded, nbytes) != nbytes:
        raise RuntimeError("short DPDFNet activation DMA upload")

    engine.start_capture()
    if op_type == "Relu":
        engine.activation_core(
            M=rows, N=udc.UE_VECTOR_SIZE,
            A_DRAM_ADDR=input_addr, OUTPUT_DRAM_ADDR=output_addr,
            IDENTITY_DRAM_ADDR=identity_addr, activation="clamp",
            clamp_min=0.0)
    elif op_type == "Sigmoid":
        engine.activation_core(
            M=rows, N=udc.UE_VECTOR_SIZE,
            A_DRAM_ADDR=input_addr, OUTPUT_DRAM_ADDR=output_addr,
            IDENTITY_DRAM_ADDR=identity_addr, activation="sigmoid")
    elif op_type == "Tanh":
        tanh_core_dram(
            engine, M=rows, N=udc.UE_VECTOR_SIZE,
            A_DRAM_ADDR=input_addr, OUTPUT_DRAM_ADDR=output_addr,
            IDENTITY_DRAM_ADDR=identity_addr)
    else:
        raise ValueError(f"unsupported activation {op_type}")
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

    result = torch.empty_like(padded)
    if engine.dma_read(
            engine.c2h_device, output_addr, result, nbytes) != nbytes:
        raise RuntimeError("short DPDFNet activation result read")
    return result[:elements].reshape(original_shape), cycles, elapsed


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
    parser.add_argument("--max-relative-rmse", type=float, default=0.05)
    args = parser.parse_args()

    try:
        import onnx
        import onnxruntime as ort
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
    selected = select_activation_nodes(model)
    names = tuple(dict.fromkeys(
        name for _, node in selected
        for name in (node.input[0], node.output[0])))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        expose_intermediates(onnx, model, names), sess_options=options,
        providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(args.seed)
    spec = rng.normal(0.0, 0.25, (1, 1, 161, 2)).astype(np.float32)
    state = initial_state(session.get_modelmeta().custom_metadata_map)
    values = dict(zip(names, session.run(
        list(names), {"spec": spec, "state_in": state})))

    udc.set_dma_device("efinix" if args.device == "efinix" else args.dev)
    clock_ns = udc.configure_clock_from_hardware()
    hardware = udc.configured_hardware_info()
    if hardware.axi_data_width_bits not in (256, 512):
        raise RuntimeError("DPDFNet requires AXI-256 or AXI-512")
    engine = udc.UnifiedEngine(clock_period_ns=clock_ns)
    engine.software_reset(run_dram_self_test=False)
    identity = torch.eye(udc.UE_VECTOR_SIZE, dtype=torch.bfloat16)
    identity_bytes = identity.numel() * 2
    identity_addr = engine.allocate_params_dram(identity_bytes)
    if engine.dma_write(
            engine.h2c_device, identity_addr,
            identity.contiguous(), identity_bytes) != identity_bytes:
        raise RuntimeError("short DPDFNet identity DMA upload")

    reports = []
    for node_index, node in selected:
        x = torch.from_numpy(np.array(
            values[node.input[0]], dtype=np.float32, copy=True))
        actual, cycles, host_elapsed = run_activation(
            engine, x, node.op_type, identity_addr, timeout_s=args.timeout)
        original = torch.from_numpy(np.array(
            values[node.output[0]], dtype=np.float32, copy=True))
        error = _metric(actual, original)
        report = {
            "node_index": node_index,
            "node_type": node.op_type,
            "node_name": node.name,
            "input_shape": list(_shape(_value_info(inferred, node.input[0]))),
            "output_shape": list(_shape(_value_info(inferred, node.output[0]))),
            "fpga_cycles": cycles,
            "fpga_execution_s": cycles * clock_ns * 1e-9,
            "host_elapsed_s": host_elapsed,
            "hardware_bf16_vs_onnx_fp32": error,
        }
        reports.append(report)
        print("ACTIVATION_RESULT:" + json.dumps(report), flush=True)
        if not torch.isfinite(actual).all():
            raise RuntimeError(f"node {node_index} produced non-finite values")
        if error["relative_rmse"] > args.max_relative_rmse:
            raise RuntimeError(
                f"node {node_index} relative RMSE "
                f"{error['relative_rmse']:.6f} exceeds "
                f"{args.max_relative_rmse:.6f}")

    summary = {
        "model": "dpdfnet2",
        "test": "all_relu_sigmoid_tanh_nodes",
        "onnx_sha256": digest,
        "axi_data_width_bits": hardware.axi_data_width_bits,
        "passed": len(reports),
        "relu": sum(item["node_type"] == "Relu" for item in reports),
        "sigmoid": sum(item["node_type"] == "Sigmoid" for item in reports),
        "tanh": sum(item["node_type"] == "Tanh" for item in reports),
        "total_fpga_cycles": sum(item["fpga_cycles"] for item in reports),
        "max_hardware_relative_rmse": max(
            item["hardware_bf16_vs_onnx_fp32"]["relative_rmse"]
            for item in reports),
        "nodes": reports,
    }
    print("TEST_RESULT:" + json.dumps(summary))


if __name__ == "__main__":
    main()
