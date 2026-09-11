#!/usr/bin/env python3
"""Validate all DPDFNet2 Gemm/MatMul nodes on Andromeda hardware."""

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


def _align64(value: int) -> int:
    return (int(value) + udc.UE_VECTOR_SIZE - 1) // udc.UE_VECTOR_SIZE \
        * udc.UE_VECTOR_SIZE


def select_linear_nodes(model):
    initializers = {value.name for value in model.graph.initializer}
    result = []
    for index, node in enumerate(model.graph.node):
        if (node.op_type in ("Gemm", "MatMul") and len(node.input) >= 2
                and node.input[1] in initializers):
            result.append((index, node))
    if not result:
        raise RuntimeError("no constant-weight Gemm/MatMul nodes found")
    return result


def prepare_linear(onnx, numpy_helper, node, initializers, input_value):
    """Convert ONNX linear semantics into batches of A @ B.T + bias."""
    attrs = _attributes(onnx, node)
    a = torch.from_numpy(np.array(
        input_value, dtype=np.float32, copy=True)).to(torch.bfloat16)
    raw_b = torch.from_numpy(np.array(
        numpy_helper.to_array(initializers[node.input[1]]),
        dtype=np.float32, copy=True)).to(torch.bfloat16)
    bias = None
    if node.op_type == "Gemm":
        if int(attrs.get("transA", 0)) != 0:
            raise ValueError("transA Gemm is not supported")
        if float(attrs.get("alpha", 1.0)) != 1.0 \
                or float(attrs.get("beta", 1.0)) != 1.0:
            raise ValueError("scaled Gemm is not supported")
        b_nk = raw_b if int(attrs.get("transB", 0)) else raw_b.transpose(-1, -2)
        if len(node.input) >= 3:
            bias = torch.from_numpy(np.array(
                numpy_helper.to_array(initializers[node.input[2]]),
                dtype=np.float32, copy=True)).to(torch.bfloat16)
        batches = 1
        k, n = int(a.shape[-1]), int(b_nk.shape[-2])
        matrices_a = a.reshape(1, -1, k)
        matrices_b = b_nk.reshape(1, n, k)
    else:
        b_nk = raw_b.transpose(-1, -2)
        k, n = int(a.shape[-1]), int(b_nk.shape[-2])
        if raw_b.dim() == 2:
            batches = 1
            matrices_a = a.reshape(1, -1, k)
            matrices_b = b_nk.reshape(1, n, k)
        elif raw_b.dim() == 3:
            batches = int(raw_b.shape[0])
            if a.numel() % (batches * k):
                raise ValueError("batched MatMul input cannot be flattened evenly")
            matrices_a = a.reshape(batches, -1, k)
            matrices_b = b_nk.reshape(batches, n, k)
        else:
            raise ValueError(f"unsupported MatMul weight rank {raw_b.dim()}")
    if int(matrices_b.shape[-1]) != k:
        raise ValueError("linear K dimensions do not match")
    return matrices_a.contiguous(), matrices_b.contiguous(), bias


def run_linear(engine, matrices_a, matrices_b, bias, *, timeout_s):
    batches, m, k = (int(value) for value in matrices_a.shape)
    n = int(matrices_b.shape[1])
    kp, npad = _align64(k), _align64(n)
    a_pad = torch.zeros(batches, m, kp, dtype=torch.bfloat16)
    b_pad = torch.zeros(batches, npad, kp, dtype=torch.bfloat16)
    a_pad[..., :k] = matrices_a
    b_pad[:, :n, :k] = matrices_b
    bias_pad = None
    if bias is not None:
        bias_pad = torch.zeros(npad, dtype=torch.bfloat16)
        bias_pad[:n] = bias.reshape(-1)

    def upload(value, *, params):
        value = value.contiguous()
        nbytes = value.numel() * 2
        address = (engine.allocate_params_dram(nbytes) if params
                   else engine.allocate_tensor_dram(nbytes))
        if engine.dma_write(engine.h2c_device, address, value, nbytes) != nbytes:
            raise RuntimeError("short DPDFNet linear DMA upload")
        return address

    a_addr = upload(a_pad, params=False)
    b_addr = upload(b_pad, params=True)
    c_addr = upload(bias_pad, params=True) if bias_pad is not None else None
    output_bytes = batches * m * npad * 2
    output_addr = engine.allocate_tensor_dram(output_bytes)
    engine.start_capture()
    for batch in range(batches):
        engine.matmat_mul_core(
            M=m, K=kp, N=npad,
            A_DRAM_ADDR=a_addr + batch * m * kp * 2,
            B_DRAM_ADDR=b_addr + batch * npad * kp * 2,
            OUTPUT_DRAM_ADDR=output_addr + batch * m * npad * 2,
            C_DRAM_ADDR=c_addr, bias_mode="broadcast_N")
    engine.generate_instruction_halt()
    engine.stop_capture()
    program_addr = engine.get_program_dram_addr()
    engine.write_captured_instructions_to_dram(program_addr)
    engine.allocate_program_dram(engine.get_capture_instruction_size_bytes())
    started = time.perf_counter()
    engine.start_execute_from_dram(program_addr)
    engine.wait_queue(timeout_s)
    if engine.is_queue_busy():
        raise TimeoutError("DPDFNet linear program did not finish")
    elapsed = time.perf_counter() - started
    cycles = int(engine.read_latency_cycles())
    engine.clear_capture_buffer()
    result = torch.empty(batches, m, npad, dtype=torch.bfloat16)
    if engine.dma_read(
            engine.c2h_device, output_addr, result,
            result.numel() * 2) != result.numel() * 2:
        raise RuntimeError("short DPDFNet linear result read")
    return result[..., :n].contiguous(), cycles, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", choices=("bittware", "bittware_512", "efinix"),
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
    selected = select_linear_nodes(model)
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
    initializers = {value.name: value for value in model.graph.initializer}

    udc.set_dma_device("efinix" if args.device == "efinix" else args.dev)
    clock_ns = udc.configure_clock_from_hardware()
    hardware = udc.configured_hardware_info()
    if hardware.axi_data_width_bits not in (256, 512):
        raise RuntimeError("DPDFNet requires AXI-256 or AXI-512")
    engine = udc.UnifiedEngine(clock_period_ns=clock_ns)
    engine.software_reset(run_dram_self_test=False)

    reports = []
    for node_index, node in selected:
        matrices_a, matrices_b, bias = prepare_linear(
            onnx, numpy_helper, node, initializers, values[node.input[0]])
        reference = torch.matmul(
            matrices_a.float(), matrices_b.float().transpose(-1, -2))
        if bias is not None:
            reference += bias.float()
        reference = reference.to(torch.bfloat16)
        actual, cycles, host_elapsed = run_linear(
            engine, matrices_a, matrices_b, bias, timeout_s=args.timeout)
        error = _metric(actual, reference)
        output_shape = _shape(_value_info(inferred, node.output[0]))
        original = torch.from_numpy(np.array(
            values[node.output[0]], dtype=np.float32, copy=True))
        original = original.reshape(actual.shape)
        report = {
            "node_index": node_index,
            "node_type": node.op_type,
            "node_name": node.name,
            "output_shape": list(output_shape),
            "hardware_shape": list(actual.shape),
            "fpga_cycles": cycles,
            "fpga_execution_s": cycles * clock_ns * 1e-9,
            "host_elapsed_s": host_elapsed,
            "hardware_vs_bf16_reference": error,
            "bf16_reference_vs_onnx_fp32": _metric(reference, original),
        }
        reports.append(report)
        print("LINEAR_RESULT:" + json.dumps(report), flush=True)
        if not torch.isfinite(actual).all():
            raise RuntimeError(f"node {node_index} produced non-finite values")
        if error["relative_rmse"] > args.max_relative_rmse:
            raise RuntimeError(
                f"node {node_index} relative RMSE "
                f"{error['relative_rmse']:.6f} exceeds "
                f"{args.max_relative_rmse:.6f}")

    summary = {
        "model": "dpdfnet2",
        "test": "all_linear_nodes",
        "onnx_sha256": digest,
        "axi_data_width_bits": hardware.axi_data_width_bits,
        "passed": len(reports),
        "gemm": sum(item["node_type"] == "Gemm" for item in reports),
        "matmul": sum(item["node_type"] == "MatMul" for item in reports),
        "total_fpga_cycles": sum(item["fpga_cycles"] for item in reports),
        "max_hardware_relative_rmse": max(
            item["hardware_vs_bf16_reference"]["relative_rmse"]
            for item in reports),
        "nodes": reports,
    }
    print("TEST_RESULT:" + json.dumps(summary))


if __name__ == "__main__":
    main()
