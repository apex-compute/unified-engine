#!/usr/bin/env python3
"""Audit a DPDFNet BF16 bin against its pinned ONNX model without hardware."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (HERE, HERE.parent / "dpdfnet8khz", ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import user_dma_core as udc


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _counts(values):
    return {str(key): count for key, count in sorted(Counter(map(int, values)).items())}


def _bf16_bytes(value):
    return value.contiguous().view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()


def _image_bytes(raw, hardware, address, size):
    offset = address - hardware["model_base"]
    _require(address % 64 == 0 and size > 0 and offset >= 0
             and offset + size <= hardware["program_offset"],
             "Convolution constant is outside the aligned resident data section")
    return raw[offset:offset + size]


def _read_matches(words, address, size, bank=0, memory_type=0):
    return (((words[0] >> 8) & 15) == udc.INSTRUCTION_UE_OP
            and ((words[5] >> 12) & 15) == udc.UE_MODE.MEMCPY_FROM_DRAM
            and int(words[1]) * 8 == address and words[2] == size
            and ((words[5] >> 16) & 1) == bank
            and ((words[7] >> 8) & 3) == memory_type
            and ((words[7] >> 10) & 4095) == 0
            and ((words[7] >> 7) & 1) == 1
            and ((words[6] >> 30) & 1) == 0)


def _check_matmul(words, resource, start):
    """Bind the current single-tile BF16 lowering to its actual descriptors.

    This deliberately fails if the compiler switches to dynamic or tiled
    matmul; an audit must understand that new sequence before certifying it.
    """
    m, k, n = (resource[key] for key in ("M", "K", "N"))
    a_address = resource["input_address"] if resource["direct"] else resource["workspace_address"]
    matches = [i for i in range(len(words) - 1)
               if _read_matches(words[i], a_address, m * k * 2)
               and _read_matches(words[i + 1], resource["weight_address"], n * k * 2, bank=1)]
    _require(len(matches) == 1, "Expected one static A/B load pair for dense convolution")
    first = matches[0]
    cursor = first + 2
    bias = resource["bias_address"] is not None
    if bias:
        _require(cursor < len(words) and _read_matches(
            words[cursor], resource["bias_address"], n * 2, memory_type=2),
            "Dense convolution bias address/size does not match its descriptor")
        cursor += 1
    first_dot = cursor
    _require(cursor + m < len(words), "Truncated dense convolution matmul sequence")
    for row, instruction in enumerate(words[cursor:cursor + m]):
        w = list(map(int, instruction))
        vector_row = (w[3] >> 24) | ((w[4] & 15) << 8)
        matrix_row = (w[4] >> 4) & 4095
        output_row = (w[4] >> 16) & 4095
        output_size = (w[4] >> 28) | ((w[5] & 4095) << 4)
        _require(((w[0] >> 8) & 15) == udc.INSTRUCTION_UE_OP
                 and ((w[5] >> 12) & 15) == udc.UE_MODE.BF16_DOT_PRODUCT
                 and ((w[5] >> 21) & 3) == 0 and (w[3] & 4095) * 64 == k
                 and output_size == n and w[2] == k * n
                 and vector_row == row * k // 64 and matrix_row == 0
                 and output_row == (m * k + row * n) // 64
                 and ((w[5] >> 16) & 1) == 0
                 and ((w[5] >> 17) & 3) == udc.URAM_WRITE_SRC.URAM_WRITE_BACK
                 and ((w[5] >> 19) & 3) == 2
                 and ((w[6] >> 18) & 4095) == 1
                 and ((w[6] >> 17) & 1) == int(bias)
                 and ((w[5] >> 23) & 7) == udc.LALU_MODE.BYPASS,
                 f"Dense BF16 matvec row {row} differs from its declared geometry")
    output = list(map(int, words[cursor + m]))
    source_y = (output[3] >> 24) | ((output[4] & 15) << 8)
    source_z = (output[4] >> 4) & 4095
    chunk = (output[4] >> 28) | ((output[5] & 4095) << 4) | (((output[7] >> 23) & 1) << 16)
    jump = (output[5] >> 26) | ((output[6] & 32767) << 6)
    _require(((output[0] >> 8) & 15) == udc.INSTRUCTION_UE_OP
             and ((output[5] >> 12) & 15) == udc.UE_MODE.URAM_DRAM_WRITEBACK
             and output[1] * 8 == resource["output_address"]
             and output[2] == m * n * 2 and source_y == source_z == m * k // 64
             and ((output[5] >> 16) & 1) == 0 and ((output[7] >> 8) & 3) == 0
             and ((output[6] >> 30) & 1) == 1 and chunk == jump == n * 2,
             "Dense convolution output DMA address/size/SRAM/stride mismatch")
    return {"input_weight_load_start": start + first,
            "matvec_start": start + first_dot, "matvec_stop": start + first_dot + m,
            "matvec_descriptors": m, "output_write_index": start + first_dot + m}


def audit_payload(payload, model_path):
    """Validate one in-memory payload and return an offline precision report."""
    names = {"dpdfnet2": "dpdfnet", "dpdfnet2_8khz": "dpdfnet8khz"}
    _require(payload.get("model") in names, "Unsupported DPDFNet model identity")
    module = names[payload["model"]]
    source_digest = importlib.import_module(f"{module}_common").validate_digest(Path(model_path))
    importlib.import_module(f"{module}_precompiled").validate_hardware(payload)
    hardware = payload["hardware"]
    _require(payload.get("onnx_sha256") == hardware.get("onnx_sha256") == source_digest,
             "ONNX source and artifact hashes disagree")
    _require(hardware.get("precision") == hardware.get("dense_convolution_precision") == "BF16",
             "Artifact does not declare BF16 precision")
    raw = hardware["model_image"].numpy().tobytes()
    program = raw[hardware["program_offset"]:hardware["program_offset"] + hardware["program_size"]]
    words = np.frombuffer(program, dtype="<u4").reshape(-1, 8)
    kinds = (words[:, 0] >> 8) & 15
    modes = (words[:, 5] >> 12) & 15
    types = (words[:, 5] >> 21) & 3
    ue = np.isin(kinds, [udc.INSTRUCTION_UE_OP, udc.INSTRUCTION_UE_PBI])
    forbidden = ue & np.isin(modes, [udc.UE_MODE.DOT_PRODUCT, udc.UE_MODE.QUANTIZE,
                                   udc.UE_MODE.DEQUANTIZE, udc.UE_MODE.CONV2D])
    _require(not np.any(forbidden), "Quantized DOT/QUANTIZE/DEQUANTIZE/CONV instruction remains")
    _require(np.all(types[ue & (modes == udc.UE_MODE.BF16_DOT_PRODUCT)] == 0),
             "BF16 matvec has a nonzero data_type")
    halt = int(np.flatnonzero(kinds == udc.INSTRUCTION_HALT)[0])
    model = onnx.shape_inference.infer_shapes(onnx.load(model_path))
    shapes = {v.name: tuple(d.dim_value for d in v.type.tensor_type.shape.dim)
              for v in [*model.graph.input, *model.graph.value_info, *model.graph.output]}
    initializers = {v.name: onnx.numpy_helper.to_array(v) for v in model.graph.initializer}
    dense = {}
    for index, node in enumerate(model.graph.node):
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type == "Conv" and attrs.get("group", 1) == 1:
            dense[index] = node, attrs
    resources = hardware.get("dense_convolutions", [])
    _require(len(dense) == len(resources) == 15 and {r["node_index"] for r in resources} == set(dense),
             "Expected exactly the fifteen ONNX dense convolutions")
    _require(hardware["graph_operations"] == len(model.graph.node), "ONNX graph node count mismatch")
    reports = []
    for resource in resources:
        index = resource["node_index"]
        node, attrs = dense[index]
        source_weight = torch.from_numpy(np.array(initializers[node.input[1]], dtype=np.float32, copy=True))
        oc, ic, kh, kw = source_weight.shape
        source_shape, output_shape = shapes[node.input[0]], shapes[node.output[0]]
        pads = attrs.get("pads", (0, 0, 0, 0))
        packed_input = (source_shape[2] + pads[0] + pads[2], source_shape[3] + pads[1] + pads[3], ic)
        packed_output = (output_shape[2], output_shape[3], oc)
        cpad, n = ((ic + 63) // 64) * 64, ((oc + 63) // 64) * 64
        m, k = output_shape[2] * output_shape[3], kh * kw * cpad
        _require(resource["kind"] == "dense_bf16"
                 and tuple(resource["weight_shape"]) == tuple(source_weight.shape)
                 and tuple(resource["input_shape"]) == packed_input
                 and tuple(resource["output_shape"]) == packed_output
                 and resource["input_channels_padded"] == cpad
                 and (resource["M"], resource["K"], resource["N"]) == (m, k, n)
                 and tuple(resource["stride"]) == tuple(attrs.get("strides", (1, 1)))
                 and tuple(resource["dilation"]) == tuple(attrs.get("dilations", (1, 1))),
                 f"Node {index}: resource geometry differs from ONNX")
        direct = kh == kw == 1 and tuple(resource["stride"]) == (1, 1) and packed_input[:2] == packed_output[:2]
        _require(resource["direct"] == direct, f"Node {index}: invalid direct-input selection")
        for suffix, address_key, shape, padded in (
                ("input", "input_address", packed_input, cpad),
                ("output", "output_address", packed_output, n),
                ("bf16_patches", "workspace_address", (m, k), k)):
            layout = hardware["tensors"][f"@conv/{index}/{suffix}"]
            _require(layout["address"] == resource[address_key] and tuple(layout["shape"]) == shape
                     and layout["padded_last"] == padded,
                     f"Node {index}: tensor/resource binding mismatch")
        rounded = source_weight.bfloat16()
        _require(bool(torch.isfinite(rounded).all()), "Nonfinite BF16 source weights")
        # Independent lane placement: each input channel occupies its own
        # position inside every spatial tap, then each output row is flattened.
        expected = torch.zeros(n, kh, kw, cpad, dtype=torch.bfloat16)
        for channel in range(ic):
            expected[:oc, :, :, channel] = rounded[:, channel]
        expected_bytes = _bf16_bytes(expected)
        _require(resource["weight_size_bytes"] == len(expected_bytes), "Weight byte count mismatch")
        actual = _image_bytes(raw, hardware, resource["weight_address"], len(expected_bytes))
        _require(actual == expected_bytes, f"Node {index}: stored weights differ from padded BF16 ONNX weights")
        bias_bytes = b""
        if len(node.input) > 2 and node.input[2]:
            bias = torch.from_numpy(np.array(initializers[node.input[2]], dtype=np.float32, copy=True)).bfloat16()
            _require(tuple(bias.shape) == (oc,) and bool(torch.isfinite(bias).all()), "Invalid ONNX bias")
            expected_bias = torch.zeros(n, dtype=torch.bfloat16)
            expected_bias[:oc] = bias
            bias_bytes = _bf16_bytes(expected_bias)
            _require(resource["bias_size_bytes"] == len(bias_bytes), "Bias byte count mismatch")
            _require(_image_bytes(raw, hardware, resource["bias_address"], len(bias_bytes)) == bias_bytes,
                     f"Node {index}: stored bias differs from padded BF16 ONNX bias")
        else:
            _require(resource["bias_address"] is None and resource["bias_size_bytes"] == 0,
                     "Unexpected bias on a bias-free ONNX convolution")
        entry = hardware["operations"][index]
        _require(entry["node_index"] == index and entry["name"] == (node.name or f"Conv_{index}"),
                 "Convolution operation range has the wrong identity")
        first, stop = entry["start"], entry["stop"]
        sequence = _check_matmul(words[first:stop], resource, first)
        reports.append({"node_index": index, "name": node.name,
                        "instruction_range": [first, stop], "M": m, "K": k, "N": n,
                        "source_weight_shape": list(source_weight.shape),
                        "weight_address": resource["weight_address"], "weight_bytes": len(expected_bytes),
                        "weight_sha256": _sha256(actual), "bias_address": resource["bias_address"],
                        "bias_bytes": len(bias_bytes), "bias_sha256": _sha256(bias_bytes) if bias_bytes else None,
                        "weight_and_bias_match_rounded_onnx_including_padding": True,
                        "ue_mode_counts": _counts(modes[first:stop][ue[first:stop]]),
                        "static_matmul_sequence": sequence})
    return {"format": "dpdfnet.bf16-precision-audit-v1", "passed": True,
            "hardware_accessed": False, "model": payload["model"],
            "onnx_sha256": source_digest, "resident_image_sha256": _sha256(raw),
            "resident_image_bytes": len(raw), "program_sha256": _sha256(program),
            "program_bytes": len(program), "program_instructions": len(words),
            "graph_operations": len(model.graph.node), "halt_instruction_index": halt,
            "halt_count": int(np.count_nonzero(kinds == udc.INSTRUCTION_HALT)), "swi_count": 0,
            "instruction_type_counts": _counts(kinds), "ue_mode_counts": _counts(modes[ue]),
            "quantized_compute_descriptors": 0, "bf16_matvec_data_type": 0,
            "dense_convolutions_audited": len(reports), "dense_convolutions": reports,
            "limitations": [
                "This is an offline storage and instruction audit, not hardware execution, accuracy or RTF measurement.",
                "BF16 describes stored weights/activations; the hardware's internal accumulation precision is unchanged.",
                "Per-node UE mode counts include layout selectors; static_matmul_sequence identifies the dense arithmetic.",
                "Weight/bias byte counts include zero padding and may reference deduplicated constants; they are not unique learned parameter bytes."]}


def audit(bin_path, model_path):
    bin_path, model_path = Path(bin_path).resolve(), Path(model_path).resolve()
    payload = torch.load(bin_path, map_location="cpu", weights_only=True)
    report = audit_payload(payload, model_path)
    report.update(bin_file=_path(bin_path), bin_sha256=_sha256(bin_path.read_bytes()),
                  model_file=_path(model_path), auditor_sha256=_sha256(Path(__file__).read_bytes()))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve() in (args.bin.resolve(), args.model.resolve()):
        parser.error("--output must differ from --bin and --model")
    report = audit(args.bin, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "model": report["model"],
                      "dense_convolutions_audited": report["dense_convolutions_audited"],
                      "program_instructions": report["program_instructions"],
                      "bin_sha256": report["bin_sha256"], "output": _path(args.output)}, indent=2))


if __name__ == "__main__":
    main()
