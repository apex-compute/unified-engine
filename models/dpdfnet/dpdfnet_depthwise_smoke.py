#!/usr/bin/env python3
"""Validate all DPDFNet2 depthwise Conv nodes on Andromeda hardware."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


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


def select_depthwise_convs(onnx, model):
    initializers = {value.name: value for value in model.graph.initializer}
    selected = []
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Conv" or len(node.input) < 2:
            continue
        weight = initializers.get(node.input[1])
        if weight is None or len(weight.dims) != 4:
            continue
        attrs = _attributes(onnx, node)
        group = int(attrs.get("group", 1))
        if group == int(weight.dims[0]) and int(weight.dims[1]) == 1:
            selected.append((index, node, weight, attrs))
    if not selected:
        raise RuntimeError("no depthwise Conv nodes found")
    return selected


def run_depthwise_conv(
        engine, x: torch.Tensor, weight: torch.Tensor,
        bias: torch.Tensor | None, *, stride: int, pad: int,
        dilation: int, timeout_s: float) -> tuple[torch.Tensor, int, float]:
    """Execute a C-group, H=1 depthwise Conv entirely on the FPGA."""
    channels, height, width = (int(value) for value in x.shape)
    out_channels, one, kernel_h, kernel_w = weight.shape
    if (height != 1 or out_channels != channels or one != 1
            or kernel_h != 1 or channels % udc.UE_VECTOR_SIZE):
        raise ValueError(
            "DPDFNet depthwise emitter needs (C,1,W), C=OC multiple of 64, "
            f"and (C,1,1,K) weights; got x={tuple(x.shape)}, "
            f"w={tuple(weight.shape)}")
    output_width = (width + 2 * pad
                    - dilation * (kernel_w - 1) - 1) // stride + 1
    chunk_width = min(32, output_width)
    bpe = 2

    packed_input = udc.conv2d_pack_activation_map(
        x.to(torch.bfloat16), 0, pad_h=0).flatten()
    input_addr = engine.allocate_tensor_dram(packed_input.numel() * bpe)
    output_addr = engine.allocate_tensor_dram(output_width * channels * bpe)
    engine.dma_write(
        engine.h2c_device, input_addr, packed_input,
        packed_input.numel() * bpe)

    zero = torch.zeros(channels, dtype=torch.bfloat16)
    zero_addr = engine.allocate_params_dram(zero.numel() * bpe)
    engine.dma_write(engine.h2c_device, zero_addr, zero, zero.numel() * bpe)
    bias_bf16 = (zero if bias is None else
                 bias.detach().cpu().to(torch.bfloat16).contiguous())
    bias_tile = bias_bf16.unsqueeze(0).expand(
        chunk_width, channels).contiguous()
    bias_addr = engine.allocate_params_dram(bias_tile.numel() * bpe)
    engine.dma_write(
        engine.h2c_device, bias_addr, bias_tile,
        bias_tile.numel() * bpe)
    tap_addresses = []
    for tap in range(kernel_w):
        values = weight[:, 0, 0, tap].detach().cpu().to(torch.bfloat16)
        tile = values.unsqueeze(0).expand(chunk_width, channels).contiguous()
        address = engine.allocate_params_dram(tile.numel() * bpe)
        engine.dma_write(engine.h2c_device, address, tile, tile.numel() * bpe)
        tap_addresses.append(address)

    a_sram = 0x10000
    temporary_sram = 0x40000
    weight_sram = 0x80000
    accumulator_sram = 0xC0000
    engine.start_capture()
    for output_start in range(0, output_width, chunk_width):
        take = min(chunk_width, output_width - output_start)
        elements = take * channels
        engine.accelerator_memory_to_sram(
            bias_addr, accumulator_sram, elements)
        for tap, weight_addr in enumerate(tap_addresses):
            for offset in range(take):
                output_index = output_start + offset
                input_index = output_index * stride - pad + tap * dilation
                source = (input_addr + input_index * channels * bpe
                          if 0 <= input_index < width else zero_addr)
                engine.accelerator_memory_to_sram(
                    source, a_sram + offset * channels * bpe, channels)
            engine.accelerator_memory_to_sram(
                weight_addr, weight_sram, elements)
            engine.eltwise_mul_core(
                a_sram, weight_sram, temporary_sram, elements)
            engine.eltwise_add_core(
                temporary_sram, accumulator_sram,
                accumulator_sram, elements)
        engine.sram_to_accelerator_memory(
            accumulator_sram,
            output_addr + output_start * channels * bpe, elements)
    engine.generate_instruction_halt()
    engine.stop_capture()
    program_addr = engine.get_program_dram_addr()
    engine.write_captured_instructions_to_dram(program_addr)
    engine.allocate_program_dram(engine.get_capture_instruction_size_bytes())
    started = time.perf_counter()
    engine.start_execute_from_dram(program_addr)
    engine.wait_queue(timeout_s)
    if engine.is_queue_busy():
        raise TimeoutError("DPDFNet depthwise program did not finish")
    elapsed = time.perf_counter() - started
    cycles = int(engine.read_latency_cycles())
    engine.clear_capture_buffer()
    flat = torch.empty(output_width * channels, dtype=torch.bfloat16)
    read = engine.dma_read(
        engine.c2h_device, output_addr, flat, flat.numel() * bpe)
    if read != flat.numel() * bpe:
        raise RuntimeError("short DPDFNet depthwise output read")
    output = flat.reshape(output_width, channels).permute(1, 0)
    return output[:, None, :].contiguous(), cycles, elapsed


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
    selected = select_depthwise_convs(onnx, model)
    names = tuple(dict.fromkeys(
        name for _, node, _, _ in selected
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
    for node_index, node, weight_proto, attrs in selected:
        input_shape = _shape(_value_info(inferred, node.input[0]))
        output_shape = _shape(_value_info(inferred, node.output[0]))
        x = torch.from_numpy(np.array(
            values[node.input[0]], dtype=np.float32, copy=True))[0]
        weight = torch.from_numpy(np.array(
            numpy_helper.to_array(weight_proto), dtype=np.float32, copy=True))
        bias = None
        if len(node.input) >= 3:
            bias = torch.from_numpy(np.array(
                numpy_helper.to_array(initializers[node.input[2]]),
                dtype=np.float32, copy=True))
        strides = tuple(int(v) for v in attrs.get("strides", (1, 1)))
        dilations = tuple(int(v) for v in attrs.get("dilations", (1, 1)))
        pads = tuple(int(v) for v in attrs.get("pads", (0, 0, 0, 0)))
        if (strides[0] != 1 or dilations[0] != 1
                or pads[0] != 0 or pads[2] != 0
                or pads[1] != pads[3]):
            raise RuntimeError(f"unsupported depthwise geometry at node {node_index}")
        reference = F.conv2d(
            x.to(torch.bfloat16).float().unsqueeze(0),
            weight.to(torch.bfloat16).float(),
            None if bias is None else bias.to(torch.bfloat16).float(),
            stride=strides, padding=(0, pads[1]), dilation=dilations,
            groups=x.shape[0])[0].to(torch.bfloat16)
        actual, cycles, host_elapsed = run_depthwise_conv(
            engine, x, weight, bias, stride=strides[1], pad=pads[1],
            dilation=dilations[1], timeout_s=args.timeout)
        error = _metric(actual, reference)
        original = torch.from_numpy(np.array(
            values[node.output[0]], dtype=np.float32, copy=True))[0]
        report = {
            "node_index": node_index,
            "node_name": node.name,
            "input_shape": list(input_shape),
            "output_shape": list(output_shape),
            "kernel_shape": list(weight.shape),
            "stride": list(strides),
            "padding_tlbr": list(pads),
            "fpga_cycles": cycles,
            "fpga_execution_s": cycles * clock_ns * 1e-9,
            "host_elapsed_s": host_elapsed,
            "hardware_vs_bf16_reference": error,
            "bf16_reference_vs_onnx_fp32": _metric(reference, original),
        }
        reports.append(report)
        print("DEPTHWISE_RESULT:" + json.dumps(report), flush=True)
        if not torch.isfinite(actual).all():
            raise RuntimeError(f"node {node_index} produced non-finite values")
        if error["relative_rmse"] > args.max_relative_rmse:
            raise RuntimeError(
                f"node {node_index} relative RMSE "
                f"{error['relative_rmse']:.6f} exceeds "
                f"{args.max_relative_rmse:.6f}")

    summary = {
        "model": "dpdfnet2",
        "test": "all_depthwise_convs",
        "onnx_sha256": digest,
        "axi_data_width_bits": hardware.axi_data_width_bits,
        "passed": len(reports),
        "total_fpga_cycles": sum(item["fpga_cycles"] for item in reports),
        "max_hardware_relative_rmse": max(
            item["hardware_vs_bf16_reference"]["relative_rmse"]
            for item in reports),
        "nodes": reports,
    }
    print("TEST_RESULT:" + json.dumps(summary))


if __name__ == "__main__":
    main()
