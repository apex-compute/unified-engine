#!/usr/bin/env python3
"""Validate every DPDFNet2 LayerNormalization node on Andromeda hardware."""

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
from dpdfnet_conv_smoke import (
    _attributes, _metric, _shape, _value_info, expose_intermediates,
)
from dpdfnet_run_cpu import initial_state


def select_layer_norm_nodes(onnx, model):
    initializers = {value.name for value in model.graph.initializer}
    result = []
    for index, node in enumerate(model.graph.node):
        if node.op_type != "LayerNormalization":
            continue
        attrs = _attributes(onnx, node)
        if int(attrs.get("axis", -1)) != -1:
            raise RuntimeError(
                f"LayerNormalization node {index} does not use the last axis")
        if len(node.input) != 3 or not all(
                name in initializers for name in node.input[1:]):
            raise RuntimeError(
                f"LayerNormalization node {index} needs constant gamma/beta")
        result.append((index, node, attrs))
    if not result:
        raise RuntimeError("no LayerNormalization nodes found")
    return result


def staged_reference(x, gamma, beta, *, epsilon):
    """Approximate the hardware's BF16 writeback points.

    The current LayerNorm primitive computes mean/subtract/RMS/affine without
    an epsilon input. DPDFNet's epsilon is reported separately against ONNX;
    it is negligible for normal model activations but is not silently claimed
    as an exact implementation of the ONNX operator.
    """
    x = x.to(torch.bfloat16)
    centered = (x.float() - x.float().mean(dim=-1, keepdim=True)).to(
        torch.bfloat16)
    variance = centered.float().square().mean(dim=-1, keepdim=True)
    normalized = (centered.float() * torch.rsqrt(variance)).to(torch.bfloat16)
    scaled = (normalized.float() * gamma.float()).to(torch.bfloat16)
    result = (scaled.float() + beta.float()).to(torch.bfloat16)

    onnx_semantics = torch.nn.functional.layer_norm(
        x.float(), (x.shape[-1],), gamma.float(), beta.float(), epsilon)
    return result, onnx_semantics.to(torch.bfloat16)


def run_layer_norm(engine, x, gamma, beta, *, timeout_s):
    shape = tuple(int(value) for value in x.shape)
    n = shape[-1]
    m = x.numel() // n
    if n % udc.UE_VECTOR_SIZE:
        raise ValueError(
            f"LayerNorm width must be a multiple of {udc.UE_VECTOR_SIZE}, "
            f"got N={n}")
    x = x.reshape(m, n).to(torch.bfloat16).contiguous()
    gamma = gamma.reshape(n).to(torch.bfloat16).contiguous()
    beta = beta.reshape(n).to(torch.bfloat16).contiguous()

    def upload(value, *, params):
        nbytes = value.numel() * 2
        address = (engine.allocate_params_dram(nbytes) if params
                   else engine.allocate_tensor_dram(nbytes))
        written = engine.dma_write(engine.h2c_device, address, value, nbytes)
        if written != nbytes:
            raise RuntimeError("short DPDFNet LayerNorm DMA upload")
        return address

    input_addr = upload(x, params=False)
    gamma_addr = upload(gamma, params=True)
    beta_addr = upload(beta, params=True)
    output_addr = engine.allocate_tensor_dram(x.numel() * 2)

    engine.start_capture()
    engine.layer_norm_core_dram(
        M=m, N=n, A_DRAM_ADDR=input_addr, OUTPUT_DRAM_ADDR=output_addr,
        GAMMA_DRAM_ADDR=gamma_addr, BETA_DRAM_ADDR=beta_addr)
    engine.generate_instruction_halt()
    engine.stop_capture()
    program_addr = engine.get_program_dram_addr()
    engine.write_captured_instructions_to_dram(program_addr)
    engine.allocate_program_dram(engine.get_capture_instruction_size_bytes())

    started = time.perf_counter()
    engine.start_execute_from_dram(program_addr)
    engine.wait_queue(timeout_s)
    if engine.is_queue_busy():
        raise TimeoutError("DPDFNet LayerNorm program did not finish")
    elapsed = time.perf_counter() - started
    cycles = int(engine.read_latency_cycles())
    engine.clear_capture_buffer()

    result = torch.empty_like(x)
    read = engine.dma_read(
        engine.c2h_device, output_addr, result, result.numel() * 2)
    if read != result.numel() * 2:
        raise RuntimeError("short DPDFNet LayerNorm result read")
    return result.reshape(shape), cycles, elapsed


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
    parser.add_argument("--max-relative-rmse", type=float, default=0.03)
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
    selected = select_layer_norm_nodes(onnx, model)
    names = tuple(dict.fromkeys(
        name for _, node, _ in selected
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
    initializers = {value.name: value for value in model.graph.initializer}

    udc.set_dma_device("efinix" if args.device == "efinix" else args.dev)
    clock_ns = udc.configure_clock_from_hardware()
    hardware = udc.configured_hardware_info()
    if hardware.axi_data_width_bits not in (256, 512):
        raise RuntimeError("DPDFNet requires AXI-256 or AXI-512")
    engine = udc.UnifiedEngine(clock_period_ns=clock_ns)
    engine.software_reset(run_dram_self_test=False)

    reports = []
    for node_index, node, attrs in selected:
        x = torch.from_numpy(np.array(
            values[node.input[0]], dtype=np.float32, copy=True))
        gamma = torch.from_numpy(np.array(
            numpy_helper.to_array(initializers[node.input[1]]),
            dtype=np.float32, copy=True)).to(torch.bfloat16)
        beta = torch.from_numpy(np.array(
            numpy_helper.to_array(initializers[node.input[2]]),
            dtype=np.float32, copy=True)).to(torch.bfloat16)
        epsilon = float(attrs.get("epsilon", 1.0e-5))
        reference, onnx_semantics = staged_reference(
            x, gamma, beta, epsilon=epsilon)
        actual, cycles, host_elapsed = run_layer_norm(
            engine, x, gamma, beta, timeout_s=args.timeout)
        error = _metric(actual, reference)
        original = torch.from_numpy(np.array(
            values[node.output[0]], dtype=np.float32, copy=True))
        report = {
            "node_index": node_index,
            "node_name": node.name,
            "input_shape": list(_shape(_value_info(inferred, node.input[0]))),
            "output_shape": list(_shape(_value_info(inferred, node.output[0]))),
            "epsilon": epsilon,
            "fpga_cycles": cycles,
            "fpga_execution_s": cycles * clock_ns * 1e-9,
            "host_elapsed_s": host_elapsed,
            "hardware_vs_staged_bf16_reference": error,
            "no_epsilon_vs_onnx_epsilon_bf16": _metric(
                reference, onnx_semantics),
            "onnx_epsilon_bf16_vs_onnx_fp32": _metric(
                onnx_semantics, original),
        }
        reports.append(report)
        print("LAYERNORM_RESULT:" + json.dumps(report), flush=True)
        if not torch.isfinite(actual).all():
            raise RuntimeError(f"node {node_index} produced non-finite values")
        if error["relative_rmse"] > args.max_relative_rmse:
            raise RuntimeError(
                f"node {node_index} relative RMSE "
                f"{error['relative_rmse']:.6f} exceeds "
                f"{args.max_relative_rmse:.6f}")

    summary = {
        "model": "dpdfnet2",
        "test": "all_layer_normalization_nodes",
        "onnx_sha256": digest,
        "axi_data_width_bits": hardware.axi_data_width_bits,
        "passed": len(reports),
        "total_fpga_cycles": sum(item["fpga_cycles"] for item in reports),
        "max_hardware_relative_rmse": max(
            item["hardware_vs_staged_bf16_reference"]["relative_rmse"]
            for item in reports),
        "max_epsilon_omission_relative_rmse": max(
            item["no_epsilon_vs_onnx_epsilon_bf16"]["relative_rmse"]
            for item in reports),
        "nodes": reports,
    }
    print("TEST_RESULT:" + json.dumps(summary))


if __name__ == "__main__":
    main()
