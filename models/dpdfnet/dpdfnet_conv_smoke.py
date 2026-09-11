#!/usr/bin/env python3
"""Run DPDFNet2 dense ONNX Conv nodes on the Andromeda CONV2D engine."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
YOLO_HELPERS = REPO_ROOT / "models" / "yolov5s"
for search_path in (REPO_ROOT, YOLO_HELPERS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import user_dma_core as udc
from dpdfnet_common import DEFAULT_MODEL_PATH, download_model, validate_digest
from dpdfnet_common import initial_state
from yolov5_common import quantize_conv_for_andromeda


def _attributes(onnx, node) -> dict:
    return {
        value.name: onnx.helper.get_attribute_value(value)
        for value in node.attribute
    }


def select_native_convs(onnx, model):
    """Return dense Conv nodes representable by the current CONV2D API."""
    initializers = {value.name: value for value in model.graph.initializer}
    selected = []
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Conv" or len(node.input) < 2:
            continue
        weight = initializers.get(node.input[1])
        if weight is None or len(weight.dims) != 4:
            continue
        attrs = _attributes(onnx, node)
        if int(attrs.get("group", 1)) != 1:
            continue
        strides = tuple(int(v) for v in attrs.get("strides", (1, 1)))
        dilations = tuple(int(v) for v in attrs.get("dilations", (1, 1)))
        pads = tuple(int(v) for v in attrs.get("pads", (0, 0, 0, 0)))
        if (len(strides) == 2 and strides[0] == strides[1]
                and len(dilations) == 2 and dilations[0] == dilations[1]
                and len(pads) == 4 and pads[0] == pads[2]
                and pads[1] == pads[3]):
            selected.append((index, node, weight, attrs))
    if not selected:
        raise RuntimeError("no dense Conv compatible with Andromeda CONV2D was found")
    return selected


def select_first_native_conv(onnx, model):
    """Backward-compatible helper for the smallest hardware smoke test."""
    return select_native_convs(onnx, model)[0]


def _value_info(model, name: str):
    values = list(model.graph.value_info) + list(model.graph.input) \
        + list(model.graph.output)
    for value in values:
        if value.name == name:
            return value
    raise RuntimeError(f"shape inference did not describe tensor {name!r}")


def expose_intermediates(onnx, model, names: tuple[str, ...]) -> bytes:
    """Return an in-memory ONNX graph with selected tensors as graph outputs."""
    inferred = onnx.shape_inference.infer_shapes(model)
    exposed = copy.deepcopy(inferred)
    existing = {value.name for value in exposed.graph.output}
    for name in names:
        if name not in existing:
            exposed.graph.output.add().CopyFrom(_value_info(inferred, name))
    onnx.checker.check_model(exposed)
    return exposed.SerializeToString()


def _shape(value_info) -> tuple[int, ...]:
    dims = value_info.type.tensor_type.shape.dim
    if any(not dim.HasField("dim_value") for dim in dims):
        raise RuntimeError(f"dynamic intermediate shape for {value_info.name!r}")
    return tuple(int(dim.dim_value) for dim in dims)


def _metric(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = actual.float() - expected.float()
    reference_rms = expected.float().square().mean().sqrt()
    rmse = delta.square().mean().sqrt()
    return {
        "rmse": float(rmse),
        "relative_rmse": float(rmse / reference_rms.clamp_min(1e-12)),
        "max_abs": float(delta.abs().max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", choices=("bittware", "bittware_512", "efinix"),
                        default="bittware_512")
    parser.add_argument("--dev", default="xdma0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-relative-rmse", type=float, default=0.08)
    parser.add_argument(
        "--all", action="store_true",
        help="exercise every dense Conv directly representable by CONV2D")
    args = parser.parse_args()

    try:
        import onnx
        import onnxruntime as ort
        from onnx import numpy_helper
    except ImportError as exc:
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
    selected = select_native_convs(onnx, model)
    if not args.all:
        selected = selected[:1]
    inferred = onnx.shape_inference.infer_shapes(model)
    for _, node, _, _ in selected:
        input_shape = _shape(_value_info(inferred, node.input[0]))
        if len(input_shape) != 4 or input_shape[0] != 1:
            raise RuntimeError(
                f"expected static NCHW batch-one input, got {input_shape}")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    exposed_names = tuple(dict.fromkeys(
        name for _, node, _, _ in selected
        for name in (node.input[0], node.output[0])))
    session = ort.InferenceSession(
        expose_intermediates(onnx, model, exposed_names),
        sess_options=options, providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map
    rng = np.random.default_rng(args.seed)
    spec = rng.normal(0.0, 0.25, (1, 1, 161, 2)).astype(np.float32)
    state = initial_state(metadata)
    values = dict(zip(exposed_names, session.run(
        list(exposed_names), {"spec": spec, "state_in": state})))

    initializers = {value.name: value for value in model.graph.initializer}
    udc.set_dma_device("efinix" if args.device == "efinix" else args.dev)
    clock_ns = udc.configure_clock_from_hardware()
    hardware = udc.configured_hardware_info()
    if hardware.axi_data_width_bits not in (256, 512):
        raise RuntimeError(
            f"DPDFNet Conv smoke requires AXI-256 or AXI-512, got "
            f"{hardware.axi_data_width_bits}")
    engine = udc.UnifiedEngine(
        clock_period_ns=clock_ns,
        conv_geometry_mode=udc.CONV_GEOMETRY_QUEUE_CONFIG)
    engine.software_reset(run_dram_self_test=False)
    reports = []
    for node_index, node, weight_proto, attrs in selected:
        input_shape = _shape(_value_info(inferred, node.input[0]))
        output_shape = _shape(_value_info(inferred, node.output[0]))
        weights = torch.from_numpy(np.array(
            numpy_helper.to_array(weight_proto), dtype=np.float32, copy=True))
        bias = None
        if len(node.input) >= 3 and node.input[2] in initializers:
            bias = torch.from_numpy(np.array(
                numpy_helper.to_array(initializers[node.input[2]]),
                dtype=np.float32, copy=True))
        strides = tuple(int(v) for v in attrs.get("strides", (1, 1)))
        dilations = tuple(int(v) for v in attrs.get("dilations", (1, 1)))
        pads = tuple(int(v) for v in attrs.get("pads", (0, 0, 0, 0)))
        conv = torch.nn.Conv2d(
            weights.shape[1], weights.shape[0], tuple(weights.shape[2:]),
            stride=strides, padding=(pads[0], pads[1]), dilation=dilations,
            bias=bias is not None)
        with torch.no_grad():
            conv.weight.copy_(weights)
            if bias is not None:
                conv.bias.copy_(bias)
        prepared = quantize_conv_for_andromeda(
            conv, None, include_dequant=True)
        x = torch.from_numpy(np.array(
            values[node.input[0]], dtype=np.float32, copy=True))[0]
        quantized_reference = F.conv2d(
            x.to(torch.bfloat16).float().unsqueeze(0),
            prepared.dequant_weight.float(),
            None if prepared.bias is None else prepared.bias.float(),
            stride=strides, padding=(pads[0], pads[1]),
            dilation=dilations)[0].to(torch.bfloat16)

        started = time.perf_counter()
        actual = engine.run_conv2d_layer(
            x.to(torch.bfloat16), prepared.codes,
            stride_s=strides[0], pad=pads[1], pad_h=pads[0],
            dilation=dilations[0], block_scales=prepared.block_scales,
            bias=prepared.bias, data_type=prepared.data_type,
            gather=prepared.gather, timeout_s=args.timeout)
        host_elapsed = time.perf_counter() - started
        quant_error = _metric(actual, quantized_reference)
        original = torch.from_numpy(np.array(
            values[node.output[0]], dtype=np.float32, copy=True))[0]
        report = {
            "node_index": node_index,
            "node_name": node.name,
            "input_shape": list(input_shape),
            "output_shape": list(output_shape),
            "kernel_shape": list(weights.shape),
            "stride": list(strides),
            "padding_tlbr": list(pads),
            "dilation": list(dilations),
            "data_type": prepared.data_type.name,
            "gather": prepared.gather,
            "host_elapsed_s": host_elapsed,
            "fpga_cycles": int(engine.last_conv_cycles),
            "fpga_execution_s": float(
                engine.last_conv_cycles * clock_ns * 1e-9),
            "hardware_vs_quantized": quant_error,
            "quantized_vs_onnx_fp32": _metric(
                quantized_reference, original),
        }
        reports.append(report)
        print("CONV_RESULT:" + json.dumps(report), flush=True)
        if not torch.isfinite(actual).all():
            raise RuntimeError(
                f"node {node_index} hardware output contains non-finite values")
        if quant_error["relative_rmse"] > args.max_relative_rmse:
            raise RuntimeError(
                f"node {node_index} hardware/quantized-reference relative RMSE "
                f"{quant_error['relative_rmse']:.6f} exceeds "
                f"{args.max_relative_rmse:.6f}")

    summary = {
        "model": "dpdfnet2",
        "test": "all_native_convs" if args.all else "first_native_conv",
        "onnx_sha256": digest,
        "axi_data_width_bits": hardware.axi_data_width_bits,
        "passed": len(reports),
        "total_fpga_cycles": sum(item["fpga_cycles"] for item in reports),
        "max_hardware_relative_rmse": max(
            item["hardware_vs_quantized"]["relative_rmse"]
            for item in reports),
        "nodes": reports,
    }
    print("TEST_RESULT:" + json.dumps(summary))


if __name__ == "__main__":
    main()
