#!/usr/bin/env python3
"""Compile official DPDFNet2 into one stateful Andromeda deployment image."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
YOLO_HELPERS = ROOT / "models" / "yolov5s"
for search_path in (ROOT, YOLO_HELPERS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import user_dma_core as udc
import yolov5_precompiled as shared
from yolov5_common import quantize_conv_for_andromeda
from nn_lib import tanh_core_dram
from dpdfnet_common import DEFAULT_MODEL_PATH, download_model, validate_digest
from dpdfnet_precompiled import (
    DeviceEmitter, FORMAT, MODEL_BASE, MODEL_LIMIT, TENSOR_ALIGNMENT,
    TENSOR_LIMIT, TensorLayout, align_up, build_layout_plan, make_layout,
    manifest_sha256, physical_indices, tensor_manifest,
    transform_source_indices,
)
from dpdfnet_run_cpu import initial_state


class GraphCompiler:
    """Static-shape ONNX-to-Andromeda compiler for the pinned DPDFNet2 graph."""

    def __init__(self, onnx, model, digest: str):
        self.onnx = onnx
        self.model = onnx.shape_inference.infer_shapes(model)
        self.digest = digest
        self.layouts, self.tensor_cursor = build_layout_plan(self.model)
        self.shapes = {
            value.name: tuple(
                int(dim.dim_value) for dim in value.type.tensor_type.shape.dim)
            for value in [*self.model.graph.input, *self.model.graph.value_info,
                          *self.model.graph.output]
        }
        from onnx import numpy_helper
        self.initializers = {
            value.name: np.array(numpy_helper.to_array(value), copy=True)
            for value in self.model.graph.initializer
        }
        metadata = {item.key: item.value for item in self.model.metadata_props}
        state = torch.from_numpy(initial_state(metadata).copy())
        self.broadcast_aux = {}
        self.reduce_aux = {}
        self.direct_reduce_aux = set()
        self.unary_aux = {}
        self.gather_aux = {}
        self.complex_gather_aux = {}
        self.view_aux = {}
        self.compact_aux = {}
        self.expand_aux = {}
        self.pixel_shuffle_aux = {}
        self.pixel_shuffle_skip = set()
        self.pack60_aux = {}
        self.unpack10_aux = {}
        self.split10_aux = {}
        self.pair_pack_aux = {}
        self.pair_unpack_aux = {}
        self.concat_aux = {}
        self.transpose_aux = {}
        self.conv_aux = {}
        self.gru_aux = {}
        self._plan_auxiliary_layouts()
        self.emitter = DeviceEmitter(self.layouts, self.tensor_cursor, state)
        self.initializer_layouts = {}
        self._allocate_initializers()
        self.broadcast_constants = self._prepare_broadcast_constants()
        self.reduce_resources = self._prepare_reduce_resources()
        self.concat_resources = self._prepare_concat_resources()
        self.identity_address = self.emitter.allocate_constant(
            torch.eye(udc.UE_VECTOR_SIZE, dtype=torch.bfloat16))
        complex_broadcast = torch.zeros(64, 64, dtype=torch.bfloat16)
        complex_broadcast[0, 0] = 1
        complex_broadcast[1, 0] = 1
        self.complex_broadcast_address = self.emitter.allocate_constant(
            complex_broadcast)
        compact_high = torch.zeros(32, 64, dtype=torch.bfloat16)
        compact_high[16:32, :16] = torch.eye(16, dtype=torch.bfloat16)
        self.compact_high_address = self.emitter.allocate_constant(compact_high)
        self.expand_addresses = []
        for quarter in range(4):
            selector = torch.zeros(64, 64, dtype=torch.bfloat16)
            selector[:16, quarter * 16:(quarter + 1) * 16] = torch.eye(
                16, dtype=torch.bfloat16)
            self.expand_addresses.append(self.emitter.allocate_constant(selector))
        self.pixel_shuffle_addresses = {}
        selector_widths = {resource["width"]
                           for resource in self.pixel_shuffle_aux.values()}
        if self.pair_pack_aux or self.pair_unpack_aux:
            selector_widths.add(32)
        for width in sorted(selector_widths):
            even = torch.zeros(64, 64, dtype=torch.bfloat16)
            odd = torch.zeros(64, 64, dtype=torch.bfloat16)
            for column in range(width):
                even[2 * column, column] = 1
                odd[2 * column + 1, column] = 1
            self.pixel_shuffle_addresses[width] = (
                self.emitter.allocate_constant(even),
                self.emitter.allocate_constant(odd),
            )
        self.pair_extract_addresses = None
        if self.pair_unpack_aux:
            even_extract = torch.zeros(64, 64, dtype=torch.bfloat16)
            odd_extract = torch.zeros(64, 64, dtype=torch.bfloat16)
            for column in range(32):
                even_extract[column, 2 * column] = 1
                odd_extract[column, 2 * column + 1] = 1
            self.pair_extract_addresses = (
                self.emitter.allocate_constant(even_extract),
                self.emitter.allocate_constant(odd_extract),
            )
        self.pack60_addresses = {}
        for index in self.pack60_aux:
            selectors = []
            for chunk in range(15):
                logical_start = chunk * 64
                column = logical_start % 60
                first_count = min(64, 60 - column)
                first = torch.zeros(64, 64, dtype=torch.bfloat16)
                second = torch.zeros(64, 64, dtype=torch.bfloat16)
                for offset in range(first_count):
                    first[offset, column + offset] = 1
                for offset in range(64 - first_count):
                    second[first_count + offset, offset] = 1
                selectors.append((
                    self.emitter.allocate_constant(first),
                    self.emitter.allocate_constant(second),
                ))
            self.pack60_addresses[index] = selectors
        self.unpack10_addresses = {}
        for index in self.unpack10_aux:
            selectors = []
            for row in range(96):
                column = (row * 10) % 64
                first_count = min(10, 64 - column)
                first = torch.zeros(64, 64, dtype=torch.bfloat16)
                second = torch.zeros(64, 64, dtype=torch.bfloat16)
                for offset in range(first_count):
                    first[offset, column + offset] = 1
                for offset in range(10 - first_count):
                    second[first_count + offset, offset] = 1
                selectors.append((
                    self.emitter.allocate_constant(first),
                    self.emitter.allocate_constant(second),
                ))
            self.unpack10_addresses[index] = selectors
        self.split10_addresses = []
        if self.split10_aux:
            for pair in range(5):
                selector = torch.zeros(64, 64, dtype=torch.bfloat16)
                selector[0, 2 * pair] = 1
                selector[1, 2 * pair + 1] = 1
                self.split10_addresses.append(
                    self.emitter.allocate_constant(selector))
        self.linear_resources = self._prepare_linear_resources()
        self.conv_resources = self._prepare_conv_resources()

    def _allocate_aux(self, name: str, shape) -> TensorLayout:
        self.tensor_cursor = align_up(self.tensor_cursor, TENSOR_ALIGNMENT)
        layout = make_layout(name, shape, self.tensor_cursor)
        if layout.address + layout.size_bytes > TENSOR_LIMIT:
            raise RuntimeError("DPDFNet auxiliary tensor arena overflow")
        self.layouts[name] = layout
        self.tensor_cursor += align_up(layout.size_bytes, TENSOR_ALIGNMENT)
        return layout

    def _plan_auxiliary_layouts(self):
        binary = {"Add", "Mul", "Sub", "Div"}
        nodes = self.model.graph.node
        for index, node in enumerate(self.model.graph.node):
            output = self.layouts[node.output[0]]
            if node.op_type in binary:
                for position, name in enumerate(node.input[:2]):
                    shape = tuple(self.initializers[name].shape) \
                        if name in self.initializers else self.shapes[name]
                    if shape != output.shape:
                        self.broadcast_aux[index, position] = self._allocate_aux(
                            f"@broadcast/{index}/{position}", output.shape)
            if node.op_type == "ReduceSum":
                axes_name = node.input[1] if len(node.input) > 1 else None
                axes = (np.asarray(self.initializers[axes_name]).reshape(-1)
                        if axes_name else np.arange(len(self.shapes[node.input[0]])))
                source_shape = self.shapes[node.input[0]]
                reduced = math.prod(source_shape[int(axis) % len(source_shape)]
                                    for axis in axes)
                normalized_axes = [int(axis) % len(source_shape) for axis in axes]
                if (source_shape == (1, 1, 5, 96)
                        and normalized_axes == [2]
                        and output.shape == (1, 1, 96)):
                    self.direct_reduce_aux.add(index)
                    continue
                if reduced > udc.UE_VECTOR_SIZE:
                    raise RuntimeError(f"node {index}: ReduceSum depth exceeds 64")
                elements = output.logical_elements
                matrix_rows = align_up(elements, udc.UE_VECTOR_SIZE)
                self.reduce_aux[index] = {
                    "packed": self._allocate_aux(
                        f"@reduce/{index}/packed", (elements, reduced)),
                    "projected": self._allocate_aux(
                        f"@reduce/{index}/projected",
                        (matrix_rows, udc.UE_VECTOR_SIZE)),
                    # bf16_transpose_core writes a dense 64 x elements matrix.
                    # A padded layout reserves at least that much storage; only
                    # its address is used for this raw transpose scratch.
                    "transposed": self._allocate_aux(
                        f"@reduce/{index}/transposed",
                        (udc.UE_VECTOR_SIZE, matrix_rows)),
                    "depth": int(reduced),
                    "matrix_rows": matrix_rows,
                }
            if node.op_type == "Gather":
                attrs = {
                    value.name: self.onnx.helper.get_attribute_value(value)
                    for value in node.attribute
                }
                source_shape = self.shapes[node.input[0]]
                axis = int(attrs.get("axis", 0)) % len(source_shape)
                indexes = np.asarray(self.initializers[node.input[1]])
                if axis == len(source_shape) - 1 and indexes.shape == ():
                    rows = math.prod(source_shape[:-1])
                    matrix_rows = align_up(rows, udc.UE_VECTOR_SIZE)
                    if (source_shape[-1] == 2 and output.rows == 5
                            and output.logical_last == 96):
                        self.complex_gather_aux[index] = {
                            "input": self._allocate_aux(
                                f"@complex_gather/{index}/input", (64, 64)),
                            "transposed": self._allocate_aux(
                                f"@complex_gather/{index}/transposed", (64, 64)),
                        }
                        continue
                    self.gather_aux[index] = {
                        "packed": self._allocate_aux(
                            f"@gather/{index}/packed",
                            (matrix_rows, udc.UE_VECTOR_SIZE)),
                        "transposed": self._allocate_aux(
                            f"@gather/{index}/transposed",
                            (udc.UE_VECTOR_SIZE, matrix_rows)),
                        "matrix_rows": matrix_rows,
                    }
            if node.op_type in ("Reshape", "Unsqueeze", "Squeeze", "Flatten"):
                source = self.layouts.get(node.input[0])
                # Fuse the ONNX PixelShuffle Reshape-Transpose-Reshape.
                if index >= 1 and nodes[index - 1].op_type == "Transpose":
                    transpose = nodes[index - 1]
                    attrs = {
                        value.name: self.onnx.helper.get_attribute_value(value)
                        for value in transpose.attribute
                    }
                    transposed_shape = self.shapes[transpose.output[0]]
                    packed_shape = self.shapes[transpose.input[0]]
                    if (node.input[0] == transpose.output[0]
                            and tuple(attrs.get("perm", ())) == (0, 2, 3, 4, 1)
                            and len(packed_shape) == 5
                            and packed_shape[:4] == (1, 2, 64, 1)
                            and packed_shape[4] in (8, 16)
                            and transposed_shape == (1, 64, 1, packed_shape[4], 2)
                            and output.shape == (1, 64, 1, 2 * packed_shape[4])):
                        self.pixel_shuffle_skip.add(index - 1)
                        self.pixel_shuffle_aux[index] = {
                            "source": self.layouts[transpose.input[0]],
                            "odd": self._allocate_aux(
                                f"@pixel_shuffle/{index}/odd", output.shape),
                            "width": packed_shape[4],
                        }
                if (source is not None and source.rows == 16
                        and source.logical_last == 60
                        and source.padded_last == 64 and output.rows == 1
                        and output.logical_elements == 960):
                    self.pack60_aux[index] = {
                        "temporary": self._allocate_aux(
                            f"@pack60/{index}/temporary", output.shape),
                    }
                if (source is not None and source.rows == 1
                        and source.logical_elements == 960
                        and output.rows == 96 and output.logical_last == 10
                        and output.padded_last == 64):
                    self.unpack10_aux[index] = {
                        "temporary": self._allocate_aux(
                            f"@unpack10/{index}/temporary", output.shape),
                    }
                if (source is not None and source.rows == 96
                        and source.logical_last == 10
                        and output.rows == 480 and output.logical_last == 2
                        and source.logical_elements == output.logical_elements):
                    self.split10_aux[index] = self._allocate_aux(
                        f"@split10/{index}/temporary", source.shape)
                if (source is not None and source.rows > 1
                        and source.logical_last == 2
                        and source.padded_last == 64 and output.rows == 1
                        and source.logical_elements == output.logical_elements):
                    self.pair_pack_aux[index] = {
                        "input": self._allocate_aux(
                            f"@pair_pack/{index}/input", (64, 64)),
                        "transposed": self._allocate_aux(
                            f"@pair_pack/{index}/transposed", (64, 64)),
                        "odd": self._allocate_aux(
                            f"@pair_pack/{index}/odd", output.shape),
                    }
                if (source is not None and source.rows == 1
                        and output.rows > 1 and output.logical_last == 2
                        and output.padded_last == 64
                        and source.logical_elements == output.logical_elements):
                    self.pair_unpack_aux[index] = {
                        "input": self._allocate_aux(
                            f"@pair_unpack/{index}/input", (64, 64)),
                        "transposed": self._allocate_aux(
                            f"@pair_unpack/{index}/transposed", (64, 64)),
                    }
                if (source is not None and source.rows in (16, 32)
                        and source.padded_last == udc.UE_VECTOR_SIZE
                        and source.logical_last == 16 and output.rows == 1
                        and source.logical_elements == output.logical_elements):
                    pair_rows = source.rows // 2
                    self.compact_aux[index] = {
                        "even": self._allocate_aux(
                            f"@compact/{index}/even", (pair_rows, 64)),
                        "odd": self._allocate_aux(
                            f"@compact/{index}/odd", (pair_rows, 64)),
                        "high": self._allocate_aux(
                            f"@compact/{index}/high", output.shape),
                        "pair_rows": pair_rows,
                    }
                if (source is not None and source.rows == 1
                        and source.logical_elements == 256
                        and output.rows == 16 and output.logical_last == 16):
                    self.expand_aux[index] = [
                        self._allocate_aux(
                            f"@expand/{index}/{quarter}", (4, 64))
                        for quarter in range(4)
                    ]
                if (source is not None and source.rows == 1
                        and output.logical_last == 1
                        and source.logical_elements == output.logical_elements):
                    self.view_aux[index] = {
                        "matrix": self._allocate_aux(
                            f"@view/{index}/matrix",
                            (udc.UE_VECTOR_SIZE, source.padded_last)),
                        "column": self._allocate_aux(
                            f"@view/{index}/column",
                            (source.padded_last, udc.UE_VECTOR_SIZE)),
                    }
            if node.op_type == "Concat":
                attrs = {
                    value.name: self.onnx.helper.get_attribute_value(value)
                    for value in node.attribute
                }
                axis = int(attrs.get("axis", 0)) % len(output.shape)
                sources = [self.layouts[name] for name in node.input]
                if (axis == len(output.shape) - 1
                        and output.padded_last == udc.UE_VECTOR_SIZE
                        and all(item.rows == output.rows
                                and item.padded_last == udc.UE_VECTOR_SIZE
                                for item in sources)):
                    self.concat_aux[index] = self._allocate_aux(
                        f"@concat/{index}/temporary", output.shape)
            if node.op_type == "Transpose":
                attrs = {
                    value.name: self.onnx.helper.get_attribute_value(value)
                    for value in node.attribute
                }
                source_shape = self.shapes[node.input[0]]
                permutation = tuple(attrs.get(
                    "perm", reversed(range(len(source_shape)))))
                # NCHW (C=64,H=1) -> NHWC is a physical 64x64
                # transpose followed by a contiguous prefix copy.  This avoids
                # thousands of sub-beat scatter DMAs on AXI-512.
                if (len(source_shape) == 4 and source_shape[0] == 1
                        and source_shape[1] <= 64 and source_shape[2] == 1
                        and 64 < source_shape[3] <= 128
                        and permutation == (0, 2, 3, 1)):
                    self.transpose_aux[index] = {
                        "direction": "wide_channels",
                        "input": self._allocate_aux(
                            f"@transpose/{index}/input", (64, 64)),
                        "matrix": self._allocate_aux(
                            f"@transpose/{index}/matrix", (64, 64)),
                    }
                elif (len(source_shape) == 4 and source_shape[0] == 1
                        and source_shape[1] == udc.UE_VECTOR_SIZE
                        and source_shape[2] == 1
                        and source_shape[3] <= udc.UE_VECTOR_SIZE
                        and permutation in ((0, 2, 3, 1),
                                            (0, 3, 2, 1))):
                    self.transpose_aux[index] = {
                        "direction": "from_nchw",
                        "matrix": self._allocate_aux(
                            f"@transpose/{index}/matrix",
                            (udc.UE_VECTOR_SIZE, udc.UE_VECTOR_SIZE)),
                    }
                elif (len(source_shape) == 4 and source_shape[0] == 1
                        and source_shape[1] * source_shape[2]
                        <= udc.UE_VECTOR_SIZE
                        and source_shape[3] == udc.UE_VECTOR_SIZE
                        and permutation in ((0, 3, 1, 2),
                                            (0, 3, 2, 1))):
                    self.transpose_aux[index] = {
                        "direction": "to_nchw",
                        "matrix": self._allocate_aux(
                            f"@transpose/{index}/matrix",
                            (udc.UE_VECTOR_SIZE, udc.UE_VECTOR_SIZE)),
                    }
                elif (len(source_shape) == 4
                        and source_shape[:2] == (1, 1)
                        and source_shape[3] == 2
                        and permutation == (1, 0, 3, 2)):
                    self.transpose_aux[index] = {
                        "direction": "complex_columns",
                        "input": self._allocate_aux(
                            f"@transpose/{index}/input",
                            (align_up(source_shape[2], udc.UE_VECTOR_SIZE),
                             udc.UE_VECTOR_SIZE)),
                        "matrix": self._allocate_aux(
                            f"@transpose/{index}/matrix",
                            (udc.UE_VECTOR_SIZE,
                             align_up(source_shape[2], udc.UE_VECTOR_SIZE))),
                    }
            if node.op_type in ("Sqrt", "Div"):
                self.unary_aux[index] = self._allocate_aux(
                    f"@unary/{index}", output.shape)
            if node.op_type == "Conv":
                attrs = {
                    value.name: self.onnx.helper.get_attribute_value(value)
                    for value in node.attribute
                }
                source_shape = self.shapes[node.input[0]]
                output_shape = self.shapes[node.output[0]]
                if (len(source_shape) != 4 or source_shape[0] != 1
                        or len(output_shape) != 4 or output_shape[0] != 1):
                    raise RuntimeError(f"node {index}: Conv must be batch-one NCHW")
                pads = tuple(int(value) for value in attrs.get(
                    "pads", (0, 0, 0, 0)))
                c, h, w = source_shape[1:]
                oc, oh, ow = output_shape[1:]
                packed_source = self._allocate_aux(
                    f"@conv/{index}/input",
                    (h + pads[0] + pads[2], w + pads[1] + pads[3], c))
                packed_output = self._allocate_aux(
                    f"@conv/{index}/output", (oh, ow, oc))
                source_layout = self.layouts[node.input[0]]
                input_columns = h * source_layout.padded_last
                input_matrix = self._allocate_aux(
                    f"@conv/{index}/input_matrix",
                    (udc.UE_VECTOR_SIZE, input_columns))
                input_transposed = self._allocate_aux(
                    f"@conv/{index}/input_transposed",
                    (input_columns, udc.UE_VECTOR_SIZE))
                output_rows = align_up(oh * ow, udc.UE_VECTOR_SIZE)
                output_matrix = self._allocate_aux(
                    f"@conv/{index}/output_matrix",
                    (output_rows, udc.UE_VECTOR_SIZE))
                output_transposed = self._allocate_aux(
                    f"@conv/{index}/output_transposed",
                    (udc.UE_VECTOR_SIZE, output_rows))
                self.conv_aux[index] = {
                    "input": packed_source, "output": packed_output,
                    "attrs": attrs,
                    "input_matrix": input_matrix,
                    "input_transposed": input_transposed,
                    "input_columns": input_columns,
                    "output_matrix": output_matrix,
                    "output_transposed": output_transposed,
                    "output_rows": output_rows,
                }
            if node.op_type == "GRU":
                attrs = {
                    value.name: self.onnx.helper.get_attribute_value(value)
                    for value in node.attribute
                }
                hidden = int(attrs.get("hidden_size", 0))
                if (hidden != 64
                        or attrs.get("direction", b"forward") != b"bidirectional"
                        or int(attrs.get("linear_before_reset", 0)) != 1):
                    raise RuntimeError(f"node {index}: unsupported GRU contract")
                names = ("xg", "hg", "z", "r", "candidate", "tmp", "h")
                shapes = ((1, 192), (1, 192)) + ((1, hidden),) * 5
                self.gru_aux[index] = {
                    name: self._allocate_aux(f"@gru/{index}/{name}", shape)
                    for name, shape in zip(names, shapes)
                }

    def _allocate_initializers(self):
        for name, value in self.initializers.items():
            if value.dtype.kind not in "fc":
                continue
            probe = make_layout(name, value.shape, 0)
            address = self.emitter.allocate_constant(
                torch.from_numpy(np.asarray(value, dtype=np.float32)), probe)
            self.initializer_layouts[name] = make_layout(name, value.shape, address)

    def layout(self, name: str) -> TensorLayout:
        try:
            return self.layouts[name]
        except KeyError:
            try:
                return self.initializer_layouts[name]
            except KeyError as exc:
                raise RuntimeError(f"no device tensor for {name!r}") from exc

    def _prepare_linear_resources(self):
        resources = {}
        for index, node in enumerate(self.model.graph.node):
            if node.op_type not in ("Gemm", "MatMul"):
                continue
            if node.input[1] not in self.initializers:
                raise RuntimeError(f"node {index}: runtime MatMul weights unsupported")
            attrs = {
                value.name: self.onnx.helper.get_attribute_value(value)
                for value in node.attribute
            }
            raw = torch.from_numpy(np.asarray(
                self.initializers[node.input[1]], dtype=np.float32)).to(
                    torch.bfloat16)
            if node.op_type == "Gemm":
                if (int(attrs.get("transA", 0)) != 0
                        or float(attrs.get("alpha", 1.0)) != 1.0
                        or float(attrs.get("beta", 1.0)) != 1.0):
                    raise RuntimeError(f"node {index}: scaled/transposed-A Gemm")
                b_nk = raw if int(attrs.get("transB", 0)) else raw.T
            else:
                b_nk = raw.transpose(-1, -2)
            source, output = self.layout(node.input[0]), self.layout(node.output[0])
            k, n = source.logical_last, output.logical_last
            batches = 1 if b_nk.dim() == 2 else int(b_nk.shape[0])
            b_nk = b_nk.reshape(batches, n, k)
            packed = torch.zeros(
                batches, output.padded_last, source.padded_last,
                dtype=torch.bfloat16)
            packed[:, :n, :k] = b_nk
            weight_address = self.emitter.allocate_constant(packed)
            bias_address = None
            if node.op_type == "Gemm" and len(node.input) >= 3:
                bias = torch.from_numpy(np.asarray(
                    self.initializers[node.input[2]], dtype=np.float32)).reshape(-1)
                packed_bias = torch.zeros(
                    output.padded_last, dtype=torch.bfloat16)
                packed_bias[:n] = bias.to(torch.bfloat16)
                bias_address = self.emitter.allocate_constant(packed_bias)
            if source.rows % batches:
                raise RuntimeError(f"node {index}: batched MatMul rows do not divide")
            resources[index] = {
                "batches": batches,
                "rows_per_batch": source.rows // batches,
                "weight_address": weight_address,
                "weight_batch_bytes": output.padded_last * source.padded_last * 2,
                "bias_address": bias_address,
            }
        return resources

    def _prepare_broadcast_constants(self):
        resources = {}
        for index, node in enumerate(self.model.graph.node):
            if node.op_type not in ("Add", "Mul", "Sub", "Div"):
                continue
            output = self.layout(node.output[0])
            for position, name in enumerate(node.input[:2]):
                if name not in self.initializers:
                    continue
                source = self.layout(name)
                if source.shape == output.shape:
                    continue
                temporary = self.broadcast_aux[index, position]
                expanded = np.broadcast_to(
                    np.asarray(self.initializers[name], dtype=np.float32),
                    output.shape).copy()
                address = self.emitter.allocate_constant(
                    torch.from_numpy(expanded), temporary)
                resources[index, position] = make_layout(
                    temporary.name, temporary.shape, address)
        return resources

    def _prepare_reduce_resources(self):
        resources = {}
        for index, item in self.reduce_aux.items():
            matrix = torch.zeros(
                udc.UE_VECTOR_SIZE, udc.UE_VECTOR_SIZE,
                dtype=torch.bfloat16)
            matrix[0, :item["depth"]] = 1
            resources[index] = self.emitter.allocate_constant(matrix)
        return resources

    def _prepare_concat_resources(self):
        resources = {}
        for index in self.concat_aux:
            node = self.model.graph.node[index]
            offset = 0
            matrices = []
            for name in node.input:
                source = self.layout(name)
                matrix = torch.zeros(
                    udc.UE_VECTOR_SIZE, udc.UE_VECTOR_SIZE,
                    dtype=torch.bfloat16)
                for column in range(source.logical_last):
                    matrix[offset + column, column] = 1
                matrices.append(self.emitter.allocate_constant(matrix))
                offset += source.logical_last
            resources[index] = matrices
        return resources

    @staticmethod
    def _pack_quantized_codes(codes: torch.Tensor, precision: str):
        flat = codes.detach().cpu().to(torch.uint8).contiguous().flatten()
        if precision == "if8":
            return flat
        if precision != "if4":
            raise RuntimeError(f"unsupported convolution precision {precision}")
        if flat.numel() % 2:
            flat = torch.cat((flat, torch.zeros(1, dtype=torch.uint8)))
        return (flat[0::2] | (flat[1::2] << 4)).contiguous()

    def _prepare_conv_resources(self):
        resources = {}
        for index, node in enumerate(self.model.graph.node):
            if node.op_type != "Conv":
                continue
            aux = self.conv_aux[index]
            packed_source_layout, packed_output_layout, attrs = (
                aux["input"], aux["output"], aux["attrs"])
            weight = torch.from_numpy(np.asarray(
                self.initializers[node.input[1]], dtype=np.float32))
            bias = None
            if len(node.input) >= 3:
                bias = torch.from_numpy(np.asarray(
                    self.initializers[node.input[2]], dtype=np.float32))
            group = int(attrs.get("group", 1))
            if group == 1:
                conv = torch.nn.Conv2d(
                    weight.shape[1], weight.shape[0], tuple(weight.shape[2:]),
                    stride=tuple(attrs.get("strides", (1, 1))), padding=0,
                    dilation=tuple(attrs.get("dilations", (1, 1))),
                    bias=bias is not None)
                with torch.no_grad():
                    conv.weight.copy_(weight)
                    if bias is not None:
                        conv.bias.copy_(bias)
                prepared = quantize_conv_for_andromeda(conv, None)
                precision = prepared.data_type.name.lower()
                packed_source = shared._PackedMap(
                    (packed_source_layout.logical_last,
                     packed_source_layout.shape[0], packed_source_layout.shape[1]),
                    packed_source_layout.padded_last,
                    tuple(range(packed_source_layout.logical_last)),
                    packed_source_layout.address)
                packed_output = shared._PackedMap(
                    (packed_output_layout.logical_last,
                     packed_output_layout.shape[0], packed_output_layout.shape[1]),
                    packed_output_layout.padded_last,
                    tuple(range(packed_output_layout.logical_last)),
                    packed_output_layout.address)
                strides = tuple(int(value) for value in attrs.get(
                    "strides", (1, 1)))
                dilations = tuple(int(value) for value in attrs.get(
                    "dilations", (1, 1)))
                if strides[0] != strides[1] or dilations[0] != dilations[1]:
                    raise RuntimeError(
                        f"node {index}: dense Conv geometry is asymmetric")
                operation = {
                    "name": node.name or f"Conv_{index}",
                    "stride": strides[0], "pad": 0,
                    "dilation": dilations[0], "activate": False,
                }
                encoded = {
                    "precision": precision,
                    "layout": "gather" if prepared.gather else "channels",
                    "codes_packed": self._pack_quantized_codes(
                        prepared.codes, precision),
                    "codes_shape": list(prepared.codes.shape),
                    "block_scales": prepared.block_scales.to(
                        torch.bfloat16).contiguous(),
                    "bias": (torch.empty(0, dtype=torch.bfloat16)
                             if prepared.bias is None else
                             prepared.bias.to(torch.bfloat16).contiguous()),
                }
                resources[index] = {
                    "kind": "dense",
                    "plan": shared._prepare_conv_plan(
                        operation, encoded, packed_source, packed_output,
                        self.emitter.image, allow_half_vector_output=True),
                }
                continue

            channels = packed_source_layout.logical_last
            if (group != channels or weight.shape[0] != channels
                    or weight.shape[1] != 1 or weight.shape[2] != 1):
                raise RuntimeError(f"node {index}: unsupported grouped Conv")
            chunk_width = min(32, packed_output_layout.shape[1])
            cpad = packed_source_layout.padded_last
            padded_bias = torch.zeros(cpad, dtype=torch.bfloat16)
            if bias is not None:
                padded_bias[:channels] = bias.to(torch.bfloat16)
            bias_tile = padded_bias.unsqueeze(0).expand(
                chunk_width, cpad).contiguous()
            bias_address = self.emitter.allocate_constant(bias_tile)
            tap_addresses = []
            for tap in range(weight.shape[3]):
                padded_weight = torch.zeros(cpad, dtype=torch.bfloat16)
                padded_weight[:channels] = weight[:, 0, 0, tap].to(
                    torch.bfloat16)
                tile = padded_weight.unsqueeze(0).expand(
                    chunk_width, cpad).contiguous()
                tap_addresses.append(self.emitter.allocate_constant(tile))
            resources[index] = {
                "kind": "depthwise", "chunk_width": chunk_width,
                "bias_address": bias_address,
                "tap_addresses": tap_addresses,
            }
        return resources

    @staticmethod
    def _logical_to_physical(layout: TensorLayout, logical):
        logical = np.asarray(logical, dtype=np.int64)
        return ((logical // layout.logical_last) * layout.padded_last
                + logical % layout.logical_last)

    def _broadcast_indices(self, source: TensorLayout, output: TensorLayout):
        logical = np.arange(source.logical_elements, dtype=np.int64).reshape(
            source.shape)
        try:
            expanded = np.broadcast_to(logical, output.shape).reshape(-1)
        except ValueError as exc:
            raise RuntimeError(
                f"cannot broadcast {source.shape} to {output.shape}") from exc
        return self._logical_to_physical(source, expanded)

    def _binary_operand(self, index, position, output):
        name = self.model.graph.node[index].input[position]
        source = self.layout(name)
        if source.shape == output.shape:
            return source
        temporary = self.broadcast_aux[index, position]
        if name in self.initializers:
            return self.broadcast_constants[index, position]
        if (source.rows == output.rows and source.logical_last == 1
                and output.logical_last == 2
                and source.padded_last == 64 and output.padded_last == 64):
            self.emitter.engine.matmat_mul_core(
                M=output.rows, K=64, N=64,
                A_DRAM_ADDR=source.address,
                B_DRAM_ADDR=self.complex_broadcast_address,
                OUTPUT_DRAM_ADDR=temporary.address,
                is_B_quantized=False)
            return temporary
        self.emitter.emit_mapping(
            source, temporary, self._broadcast_indices(source, output))
        return temporary

    def emit_binary(self, index, node, mode):
        output = self.layout(node.output[0])
        left = self._binary_operand(index, 0, output)
        right = self._binary_operand(index, 1, output)
        self.emitter.engine.eltwise_core_dram(
            M=output.rows, N=output.padded_last,
            dram_a=left.address, dram_b=right.address,
            dram_out=output.address, mode=mode)

    def emit_linear(self, index, node):
        source, output = self.layout(node.input[0]), self.layout(node.output[0])
        resource = self.linear_resources[index]
        rows = resource["rows_per_batch"]
        for batch in range(resource["batches"]):
            self.emitter.engine.matmat_mul_core(
                M=rows, K=source.padded_last, N=output.padded_last,
                A_DRAM_ADDR=(source.address
                             + batch * rows * source.padded_last * 2),
                B_DRAM_ADDR=(resource["weight_address"]
                             + batch * resource["weight_batch_bytes"]),
                OUTPUT_DRAM_ADDR=(output.address
                                  + batch * rows * output.padded_last * 2),
                C_DRAM_ADDR=resource["bias_address"],
                bias_mode="broadcast_N")

    def emit_activation(self, node):
        source, output = self.layout(node.input[0]), self.layout(node.output[0])
        lines = output.physical_elements // udc.UE_VECTOR_SIZE
        line_bytes = udc.UE_VECTOR_SIZE * 2
        # Keep each standalone activation within one legacy matmul M tile.
        # On the current hardware path a larger M is internally split into
        # 16-row tiles whose reused writeback descriptor retains only the last
        # tile.  Issuing explicit <=16-row calls gives every tile an immutable
        # source/destination address in the resident program.
        for line_start in range(0, lines, 16):
            take = min(16, lines - line_start)
            source_address = source.address + line_start * line_bytes
            output_address = output.address + line_start * line_bytes
            if node.op_type == "Tanh":
                tanh_core_dram(
                    self.emitter.engine, M=take, N=udc.UE_VECTOR_SIZE,
                    A_DRAM_ADDR=source_address,
                    OUTPUT_DRAM_ADDR=output_address,
                    IDENTITY_DRAM_ADDR=self.identity_address)
            else:
                activation = {
                    "Relu": "clamp", "Sigmoid": "sigmoid", "Log": "log",
                }[node.op_type]
                self.emitter.engine.activation_core(
                    M=take, N=udc.UE_VECTOR_SIZE,
                    A_DRAM_ADDR=source_address,
                    OUTPUT_DRAM_ADDR=output_address,
                    IDENTITY_DRAM_ADDR=self.identity_address,
                    activation=activation,
                    # The ONNX frontend adds 1e-10 before Log. The shared
                    # transformer default of 1e-3 is too high for log power.
                    clamp_min=1e-10 if node.op_type == "Log" else 0.0)
        # Do not issue a separate strided padding clear here.  The identity
        # matmul writes every physical lane, and logical consumers ignore the
        # padded lanes.  More importantly, a 48-BF16 (96-byte) padding span is
        # AXI-256 aligned but not AXI-512 aligned; replaying that descriptor on
        # AXI-512 can overwrite the next row's 16 logical values.

    def emit_layer_norm(self, node):
        source, output = self.layout(node.input[0]), self.layout(node.output[0])
        if source.logical_last != 64 or output.shape != source.shape:
            raise RuntimeError("DPDFNet LayerNorm contract changed")
        self.emitter.engine.layer_norm_core_dram(
            M=source.rows, N=64,
            A_DRAM_ADDR=source.address, OUTPUT_DRAM_ADDR=output.address,
            GAMMA_DRAM_ADDR=self.layout(node.input[1]).address,
            BETA_DRAM_ADDR=self.layout(node.input[2]).address,
            ZEROS_DRAM_ADDR=self.emitter.zero_address)

    def emit_view(self, node_index, node):
        source, output = self.layout(node.input[0]), self.layout(node.output[0])
        if node_index in self.pack60_aux:
            temporary = self.pack60_aux[node_index]["temporary"]
            for chunk, (first_selector, second_selector) in enumerate(
                    self.pack60_addresses[node_index]):
                logical_start = chunk * 64
                source_row = logical_start // 60
                destination = output.address + chunk * 128
                self.emitter.engine.matmat_mul_core(
                    M=1, K=64, N=64,
                    A_DRAM_ADDR=source.address + source_row * 128,
                    B_DRAM_ADDR=first_selector,
                    OUTPUT_DRAM_ADDR=destination, is_B_quantized=False)
                self.emitter.engine.matmat_mul_core(
                    M=1, K=64, N=64,
                    A_DRAM_ADDR=source.address + (source_row + 1) * 128,
                    B_DRAM_ADDR=second_selector,
                    OUTPUT_DRAM_ADDR=temporary.address + chunk * 128,
                    is_B_quantized=False)
                self.emitter.engine.eltwise_core_dram(
                    M=1, N=64, dram_a=destination,
                    dram_b=temporary.address + chunk * 128,
                    dram_out=destination, mode=udc.UE_MODE.ELTWISE_ADD)
            return
        if node_index in self.unpack10_aux:
            temporary = self.unpack10_aux[node_index]["temporary"]
            for row, (first_selector, second_selector) in enumerate(
                    self.unpack10_addresses[node_index]):
                source_chunk, source_column = divmod(row * 10, 64)
                destination = output.address + row * 128
                self.emitter.engine.matmat_mul_core(
                    M=1, K=64, N=64,
                    A_DRAM_ADDR=source.address + source_chunk * 128,
                    B_DRAM_ADDR=first_selector,
                    OUTPUT_DRAM_ADDR=destination, is_B_quantized=False)
                if source_column > 54:
                    self.emitter.engine.matmat_mul_core(
                        M=1, K=64, N=64,
                        A_DRAM_ADDR=source.address + (source_chunk + 1) * 128,
                        B_DRAM_ADDR=second_selector,
                        OUTPUT_DRAM_ADDR=temporary.address + row * 128,
                        is_B_quantized=False)
                    self.emitter.engine.eltwise_core_dram(
                        M=1, N=64, dram_a=destination,
                        dram_b=temporary.address + row * 128,
                        dram_out=destination, mode=udc.UE_MODE.ELTWISE_ADD)
            return
        if node_index in self.split10_aux:
            temporary = self.split10_aux[node_index]
            for pair, selector in enumerate(self.split10_addresses):
                self.emitter.engine.matmat_mul_core(
                    M=source.rows, K=64, N=64,
                    A_DRAM_ADDR=source.address, B_DRAM_ADDR=selector,
                    OUTPUT_DRAM_ADDR=temporary.address,
                    is_B_quantized=False)
                self.emitter.engine.accelerator_memory_to_sram(
                    temporary.address, 0, 0,
                    memcpy_length_bytes=temporary.size_bytes)
                self.emitter.engine.sram_to_accelerator_memory(
                    0, output.address + pair * 128, 0,
                    memcpy_length_bytes=temporary.size_bytes,
                    stride_bytes_per_chunk=128,
                    stride_jump_bytes=5 * 128)
            return
        if node_index in self.pair_pack_aux:
            resource = self.pair_pack_aux[node_index]
            even_selector, odd_selector = self.pixel_shuffle_addresses[32]
            for chunk, row_start in enumerate(range(0, source.rows, 32)):
                take = min(32, source.rows - row_start)
                self.emitter.emit_zero(resource["input"])
                self.emitter.engine.accelerator_memory_to_sram(
                    source.address + row_start * 128, 0, 0,
                    memcpy_length_bytes=take * 128)
                self.emitter.engine.sram_to_accelerator_memory(
                    0, resource["input"].address, 0,
                    memcpy_length_bytes=take * 128)
                self.emitter.engine.bf16_transpose_core(
                    M=64, N=64, INPUT_DRAM_ADDR=resource["input"].address,
                    OUTPUT_DRAM_ADDR=resource["transposed"].address,
                    IDENTITY_DRAM_ADDR=self.identity_address)
                destination = output.address + chunk * 128
                self.emitter.engine.matmat_mul_core(
                    M=1, K=64, N=64,
                    A_DRAM_ADDR=resource["transposed"].address,
                    B_DRAM_ADDR=even_selector, OUTPUT_DRAM_ADDR=destination,
                    is_B_quantized=False)
                self.emitter.engine.matmat_mul_core(
                    M=1, K=64, N=64,
                    A_DRAM_ADDR=resource["transposed"].address + 128,
                    B_DRAM_ADDR=odd_selector,
                    OUTPUT_DRAM_ADDR=resource["odd"].address + chunk * 128,
                    is_B_quantized=False)
                self.emitter.engine.eltwise_core_dram(
                    M=1, N=64, dram_a=destination,
                    dram_b=resource["odd"].address + chunk * 128,
                    dram_out=destination, mode=udc.UE_MODE.ELTWISE_ADD)
            return
        if node_index in self.pair_unpack_aux:
            resource = self.pair_unpack_aux[node_index]
            even_extract, odd_extract = self.pair_extract_addresses
            for chunk, row_start in enumerate(range(0, output.rows, 32)):
                take = min(32, output.rows - row_start)
                self.emitter.emit_zero(resource["input"])
                self.emitter.engine.matmat_mul_core(
                    M=1, K=64, N=64,
                    A_DRAM_ADDR=source.address + chunk * 128,
                    B_DRAM_ADDR=even_extract,
                    OUTPUT_DRAM_ADDR=resource["input"].address,
                    is_B_quantized=False)
                self.emitter.engine.matmat_mul_core(
                    M=1, K=64, N=64,
                    A_DRAM_ADDR=source.address + chunk * 128,
                    B_DRAM_ADDR=odd_extract,
                    OUTPUT_DRAM_ADDR=resource["input"].address + 128,
                    is_B_quantized=False)
                self.emitter.engine.bf16_transpose_core(
                    M=64, N=64, INPUT_DRAM_ADDR=resource["input"].address,
                    OUTPUT_DRAM_ADDR=resource["transposed"].address,
                    IDENTITY_DRAM_ADDR=self.identity_address)
                self.emitter.engine.accelerator_memory_to_sram(
                    resource["transposed"].address, 0, 0,
                    memcpy_length_bytes=take * 128)
                self.emitter.engine.sram_to_accelerator_memory(
                    0, output.address + row_start * 128, 0,
                    memcpy_length_bytes=take * 128)
            return
        if node_index in self.pixel_shuffle_aux:
            resource = self.pixel_shuffle_aux[node_index]
            packed = resource["source"]
            width = resource["width"]
            even_selector, odd_selector = self.pixel_shuffle_addresses[width]
            branch_bytes = 64 * packed.padded_last * 2
            self.emitter.engine.matmat_mul_core(
                M=64, K=64, N=64, A_DRAM_ADDR=packed.address,
                B_DRAM_ADDR=even_selector, OUTPUT_DRAM_ADDR=output.address,
                is_B_quantized=False)
            self.emitter.engine.matmat_mul_core(
                M=64, K=64, N=64,
                A_DRAM_ADDR=packed.address + branch_bytes,
                B_DRAM_ADDR=odd_selector,
                OUTPUT_DRAM_ADDR=resource["odd"].address,
                is_B_quantized=False)
            self.emitter.engine.eltwise_core_dram(
                M=output.rows, N=output.padded_last,
                dram_a=output.address, dram_b=resource["odd"].address,
                dram_out=output.address, mode=udc.UE_MODE.ELTWISE_ADD)
            return
        if node_index in self.expand_aux:
            engine = self.emitter.engine
            row_bytes = output.padded_last * 2
            for quarter, staged in enumerate(self.expand_aux[node_index]):
                engine.matmat_mul_core(
                    M=4, K=64, N=64,
                    A_DRAM_ADDR=source.address,
                    B_DRAM_ADDR=self.expand_addresses[quarter],
                    OUTPUT_DRAM_ADDR=staged.address,
                    is_B_quantized=False)
                engine.accelerator_memory_to_sram(
                    staged.address, 0, 0,
                    memcpy_length_bytes=staged.size_bytes)
                engine.sram_to_accelerator_memory(
                    0, output.address + quarter * row_bytes, 0,
                    memcpy_length_bytes=staged.size_bytes,
                    stride_bytes_per_chunk=row_bytes,
                    stride_jump_bytes=4 * row_bytes)
            return
        if node_index in self.compact_aux:
            engine = self.emitter.engine
            aux = self.compact_aux[node_index]
            row_bytes = source.padded_last * 2
            for parity, key in ((0, "even"), (1, "odd")):
                staged = aux[key]
                engine.accelerator_memory_to_sram(
                    source.address + parity * row_bytes, 0, 0,
                    memcpy_length_bytes=staged.size_bytes,
                    stride_bytes_per_chunk=row_bytes,
                    stride_jump_bytes=2 * row_bytes)
                engine.sram_to_accelerator_memory(
                    0, staged.address, 0,
                    memcpy_length_bytes=staged.size_bytes)
            engine.matmat_mul_core(
                M=aux["pair_rows"], K=64, N=32,
                A_DRAM_ADDR=aux["even"].address,
                B_DRAM_ADDR=self.identity_address,
                OUTPUT_DRAM_ADDR=output.address,
                is_B_quantized=False)
            engine.matmat_mul_core(
                M=aux["pair_rows"], K=64, N=32,
                A_DRAM_ADDR=aux["odd"].address,
                B_DRAM_ADDR=self.compact_high_address,
                OUTPUT_DRAM_ADDR=aux["high"].address,
                is_B_quantized=False)
            engine.eltwise_core_dram(
                M=1, N=output.padded_last,
                dram_a=output.address, dram_b=aux["high"].address,
                dram_out=output.address, mode=udc.UE_MODE.ELTWISE_ADD)
            return
        if node_index in self.view_aux:
            scratch = self.view_aux[node_index]
            engine = self.emitter.engine
            self.emitter.emit_zero(scratch["matrix"])
            engine.accelerator_memory_to_sram(
                source.address, 0, 0, memcpy_length_bytes=source.size_bytes)
            engine.sram_to_accelerator_memory(
                0, scratch["matrix"].address, 0,
                memcpy_length_bytes=source.size_bytes)
            engine.bf16_transpose_core(
                M=udc.UE_VECTOR_SIZE, N=source.padded_last,
                INPUT_DRAM_ADDR=scratch["matrix"].address,
                OUTPUT_DRAM_ADDR=scratch["column"].address,
                IDENTITY_DRAM_ADDR=self.identity_address)
            engine.accelerator_memory_to_sram(
                scratch["column"].address, 0, 0,
                memcpy_length_bytes=output.size_bytes)
            engine.sram_to_accelerator_memory(
                0, output.address, 0,
                memcpy_length_bytes=output.size_bytes)
            return
        indices = transform_source_indices(source, output.shape)
        self.emitter.emit_mapping(source, output, indices)

    def emit_transpose(self, node_index, node):
        attrs = {
            value.name: self.onnx.helper.get_attribute_value(value)
            for value in node.attribute
        }
        source, output = self.layout(node.input[0]), self.layout(node.output[0])
        if node_index in self.pixel_shuffle_skip:
            self.emitter.engine.generate_instruction_nop()
            return
        if node_index in self.transpose_aux:
            resource = self.transpose_aux[node_index]
            scratch = resource["matrix"]
            if resource["direction"] == "wide_channels":
                staged = resource["input"]
                source_row_bytes = source.padded_last * 2
                for block_start in range(0, source.logical_last, 64):
                    take = min(64, source.logical_last - block_start)
                    self.emitter.emit_zero(staged)
                    self.emitter.engine.accelerator_memory_to_sram(
                        source.address + block_start * 2, 0, 0,
                        memcpy_length_bytes=source.rows * 128,
                        stride_bytes_per_chunk=128,
                        stride_jump_bytes=source_row_bytes)
                    self.emitter.engine.sram_to_accelerator_memory(
                        0, staged.address, 0,
                        memcpy_length_bytes=source.rows * 128)
                    self.emitter.engine.bf16_transpose_core(
                        M=64, N=64, INPUT_DRAM_ADDR=staged.address,
                        OUTPUT_DRAM_ADDR=scratch.address,
                        IDENTITY_DRAM_ADDR=self.identity_address)
                    self.emitter.engine.accelerator_memory_to_sram(
                        scratch.address, 0, 0,
                        memcpy_length_bytes=take * 128)
                    self.emitter.engine.sram_to_accelerator_memory(
                        0, output.address + block_start * 128, 0,
                        memcpy_length_bytes=take * 128)
                return
            if resource["direction"] == "complex_columns":
                staged = resource["input"]
                self.emitter.emit_zero(staged)
                self.emitter.engine.accelerator_memory_to_sram(
                    source.address, 0, 0,
                    memcpy_length_bytes=source.size_bytes)
                self.emitter.engine.sram_to_accelerator_memory(
                    0, staged.address, 0,
                    memcpy_length_bytes=source.size_bytes)
                self.emitter.engine.bf16_transpose_core(
                    M=staged.rows, N=staged.padded_last,
                    INPUT_DRAM_ADDR=staged.address,
                    OUTPUT_DRAM_ADDR=scratch.address,
                    IDENTITY_DRAM_ADDR=self.identity_address)
                row_bytes = output.logical_last * 2
                for row in range(output.rows):
                    self.emitter.engine.accelerator_memory_to_sram(
                        scratch.address + row * scratch.padded_last * 2, 0, 0,
                        memcpy_length_bytes=row_bytes)
                    self.emitter.engine.sram_to_accelerator_memory(
                        0, output.address + row * output.padded_last * 2, 0,
                        memcpy_length_bytes=row_bytes)
                return
            if resource["direction"] == "to_nchw":
                self.emitter.emit_zero(scratch)
                self.emitter.engine.accelerator_memory_to_sram(
                    source.address, 0, 0,
                    memcpy_length_bytes=source.size_bytes)
                self.emitter.engine.sram_to_accelerator_memory(
                    0, scratch.address, 0,
                    memcpy_length_bytes=source.size_bytes)
                transpose_input = scratch.address
                transpose_output = output.address
            else:
                transpose_input = source.address
                transpose_output = scratch.address
            self.emitter.engine.bf16_transpose_core(
                M=udc.UE_VECTOR_SIZE, N=udc.UE_VECTOR_SIZE,
                INPUT_DRAM_ADDR=transpose_input,
                OUTPUT_DRAM_ADDR=transpose_output,
                IDENTITY_DRAM_ADDR=self.identity_address)
            if resource["direction"] == "from_nchw":
                self.emitter.engine.accelerator_memory_to_sram(
                    scratch.address, 0, 0,
                    memcpy_length_bytes=output.size_bytes)
                self.emitter.engine.sram_to_accelerator_memory(
                    0, output.address, 0,
                    memcpy_length_bytes=output.size_bytes)
            return
        permutation = attrs.get("perm", tuple(reversed(range(len(source.shape)))))
        indices = transform_source_indices(
            source, output.shape, permutation=permutation)
        self.emitter.emit_mapping(source, output, indices)

    def _integer_initializer(self, name):
        if name not in self.initializers:
            raise RuntimeError(f"expected constant integer input {name!r}")
        return np.asarray(self.initializers[name], dtype=np.int64).reshape(-1)

    def emit_slice(self, node):
        source, output = self.layout(node.input[0]), self.layout(node.output[0])
        starts = self._integer_initializer(node.input[1])
        ends = self._integer_initializer(node.input[2])
        axes = (self._integer_initializer(node.input[3]) if len(node.input) > 3
                else np.arange(starts.size))
        steps = (self._integer_initializer(node.input[4]) if len(node.input) > 4
                 else np.ones(starts.size, dtype=np.int64))
        slices = [slice(None)] * len(source.shape)
        for start, end, axis, step in zip(starts, ends, axes, steps):
            slices[int(axis)] = slice(int(start), int(end), int(step))
        indices = transform_source_indices(
            source, output.shape, slices=slices)
        self.emitter.emit_mapping(source, output, indices)

    def emit_gather(self, node_index, node):
        attrs = {
            value.name: self.onnx.helper.get_attribute_value(value)
            for value in node.attribute
        }
        source, output = self.layout(node.input[0]), self.layout(node.output[0])
        axis = int(attrs.get("axis", 0))
        index = self._integer_initializer(node.input[1])
        if self.initializers[node.input[1]].shape == ():
            index = int(index[0])
        if node_index in self.complex_gather_aux:
            resource = self.complex_gather_aux[node_index]
            component = int(index) % 2
            self.emitter.emit_zero(output)
            for row in range(output.rows):
                for block_start in range(0, output.logical_last, 64):
                    take = min(64, output.logical_last - block_start)
                    self.emitter.emit_zero(resource["input"])
                    source_row = row * output.logical_last + block_start
                    self.emitter.engine.accelerator_memory_to_sram(
                        source.address + source_row * 128, 0, 0,
                        memcpy_length_bytes=take * 128)
                    self.emitter.engine.sram_to_accelerator_memory(
                        0, resource["input"].address, 0,
                        memcpy_length_bytes=take * 128)
                    self.emitter.engine.bf16_transpose_core(
                        M=64, N=64, INPUT_DRAM_ADDR=resource["input"].address,
                        OUTPUT_DRAM_ADDR=resource["transposed"].address,
                        IDENTITY_DRAM_ADDR=self.identity_address)
                    self.emitter.engine.accelerator_memory_to_sram(
                        resource["transposed"].address + component * 128,
                        0, 0, memcpy_length_bytes=take * 2)
                    self.emitter.engine.sram_to_accelerator_memory(
                        0, output.address + row * output.padded_last * 2
                        + block_start * 2, 0, memcpy_length_bytes=take * 2)
            return
        if node_index in self.gather_aux:
            scratch = self.gather_aux[node_index]
            if source.padded_last != udc.UE_VECTOR_SIZE:
                raise RuntimeError("scalar Gather expects 64-wide physical rows")
            self.emitter.emit_zero(scratch["packed"])
            self.emitter.engine.accelerator_memory_to_sram(
                source.address, 0, 0, memcpy_length_bytes=source.size_bytes)
            self.emitter.engine.sram_to_accelerator_memory(
                0, scratch["packed"].address, 0,
                memcpy_length_bytes=source.size_bytes)
            self.emitter.engine.bf16_transpose_core(
                M=scratch["matrix_rows"], N=udc.UE_VECTOR_SIZE,
                INPUT_DRAM_ADDR=scratch["packed"].address,
                OUTPUT_DRAM_ADDR=scratch["transposed"].address,
                IDENTITY_DRAM_ADDR=self.identity_address)
            self.emitter.emit_zero(output)
            row_address = (scratch["transposed"].address
                           + int(index) * scratch["matrix_rows"] * 2)
            self.emitter.engine.accelerator_memory_to_sram(
                row_address, 0, 0,
                memcpy_length_bytes=output.logical_elements * 2)
            self.emitter.engine.sram_to_accelerator_memory(
                0, output.address, 0,
                memcpy_length_bytes=output.logical_elements * 2)
            return
        logical = np.arange(source.logical_elements, dtype=np.int64).reshape(
            source.shape)
        selected = np.take(logical, index, axis=axis).reshape(-1)
        self.emitter.emit_mapping(
            source, output, self._logical_to_physical(source, selected))

    def emit_concat(self, node_index, node):
        attrs = {
            value.name: self.onnx.helper.get_attribute_value(value)
            for value in node.attribute
        }
        output = self.layout(node.output[0])
        axis = int(attrs.get("axis", 0)) % len(output.shape)
        if node_index in self.concat_aux:
            temporary = self.concat_aux[node_index]
            for position, name in enumerate(node.input):
                source = self.layout(name)
                destination = output if position == 0 else temporary
                self.emitter.engine.matmat_mul_core(
                    M=output.rows, K=udc.UE_VECTOR_SIZE,
                    N=udc.UE_VECTOR_SIZE,
                    A_DRAM_ADDR=source.address,
                    B_DRAM_ADDR=self.concat_resources[node_index][position],
                    OUTPUT_DRAM_ADDR=destination.address)
                if position:
                    self.emitter.engine.eltwise_core_dram(
                        M=output.rows, N=udc.UE_VECTOR_SIZE,
                        dram_a=output.address, dram_b=temporary.address,
                        dram_out=output.address,
                        mode=udc.UE_MODE.ELTWISE_ADD)
            return
        destination_logical = np.arange(
            output.logical_elements, dtype=np.int64).reshape(output.shape)
        offset = 0
        self.emitter.emit_zero(output)
        for name in node.input:
            source = self.layout(name)
            selection = [slice(None)] * len(output.shape)
            selection[axis] = slice(offset, offset + source.shape[axis])
            destination = destination_logical[tuple(selection)].reshape(-1)
            self.emitter.emit_scatter(
                source, output, physical_indices(source),
                self._logical_to_physical(output, destination))
            offset += source.shape[axis]
        if offset != output.shape[axis]:
            raise RuntimeError("Concat axis coverage mismatch")

    def emit_split(self, node):
        attrs = {
            value.name: self.onnx.helper.get_attribute_value(value)
            for value in node.attribute
        }
        source = self.layout(node.input[0])
        axis = int(attrs.get("axis", 0)) % len(source.shape)
        if len(node.input) > 1:
            sizes = self._integer_initializer(node.input[1]).tolist()
        else:
            sizes = [source.shape[axis] // len(node.output)] * len(node.output)
        logical = np.arange(source.logical_elements, dtype=np.int64).reshape(
            source.shape)
        offset = 0
        for name, size in zip(node.output, sizes):
            output = self.layout(name)
            selection = [slice(None)] * len(source.shape)
            selection[axis] = slice(offset, offset + int(size))
            selected = logical[tuple(selection)].reshape(-1)
            self.emitter.emit_mapping(
                source, output, self._logical_to_physical(source, selected))
            offset += int(size)
        if offset != source.shape[axis]:
            raise RuntimeError("Split axis coverage mismatch")

    def emit_reduce_sum(self, index, node):
        source, output = self.layout(node.input[0]), self.layout(node.output[0])
        attrs = {
            value.name: self.onnx.helper.get_attribute_value(value)
            for value in node.attribute
        }
        axes = (self._integer_initializer(node.input[1]).tolist()
                if len(node.input) > 1 else list(range(len(source.shape))))
        axes = sorted(int(axis) % len(source.shape) for axis in axes)
        if len(axes) != 1:
            raise RuntimeError("DPDFNet ReduceSum expects one static axis")
        if index in self.direct_reduce_aux:
            self.emitter.engine.accelerator_memcpy(
                source.address, output.address, output.size_bytes)
            for row in range(1, 5):
                self.emitter.engine.eltwise_core_dram(
                    M=1, N=output.padded_last, dram_a=output.address,
                    dram_b=source.address + row * source.padded_last * 2,
                    dram_out=output.address, mode=udc.UE_MODE.ELTWISE_ADD)
            return
        temporary = self.reduce_aux[index]
        logical = np.arange(source.logical_elements, dtype=np.int64).reshape(
            source.shape)
        selected = np.moveaxis(logical, axes[0], -1).reshape(-1)
        self.emitter.emit_mapping(
            source, temporary["packed"],
            self._logical_to_physical(source, selected))
        self.emitter.emit_zero(temporary["projected"])
        self.emitter.engine.matmat_mul_core(
            M=output.logical_elements, K=udc.UE_VECTOR_SIZE,
            N=udc.UE_VECTOR_SIZE,
            A_DRAM_ADDR=temporary["packed"].address,
            B_DRAM_ADDR=self.reduce_resources[index],
            OUTPUT_DRAM_ADDR=temporary["projected"].address)
        self.emitter.engine.bf16_transpose_core(
            M=temporary["matrix_rows"], N=udc.UE_VECTOR_SIZE,
            INPUT_DRAM_ADDR=temporary["projected"].address,
            OUTPUT_DRAM_ADDR=temporary["transposed"].address,
            IDENTITY_DRAM_ADDR=self.identity_address)
        self.emitter.emit_zero(output)
        self.emitter.engine.accelerator_memory_to_sram(
            temporary["transposed"].address, 0, 0,
            memcpy_length_bytes=output.logical_elements * 2)
        self.emitter.engine.sram_to_accelerator_memory(
            0, output.address, 0,
            memcpy_length_bytes=output.logical_elements * 2)

    def emit_pow(self, node):
        exponent = float(np.asarray(self.initializers[node.input[1]]))
        if exponent != 2.0:
            raise RuntimeError(f"only square Pow is supported, got {exponent}")
        source, output = self.layout(node.input[0]), self.layout(node.output[0])
        self.emitter.engine.eltwise_core_dram(
            M=output.rows, N=output.padded_last,
            dram_a=source.address, dram_b=source.address,
            dram_out=output.address, mode=udc.UE_MODE.ELTWISE_MUL)

    def emit_sqrt(self, index, node):
        source, output = self.layout(node.input[0]), self.layout(node.output[0])
        inverse = self.unary_aux[index]
        self.emitter.emit_identity_lalu(
            source, inverse, self.identity_address,
            udc.LALU_MODE.MODE_RSQRT, scalar=1.0)
        self.emitter.engine.eltwise_core_dram(
            M=output.rows, N=output.padded_last,
            dram_a=source.address, dram_b=inverse.address,
            dram_out=output.address, mode=udc.UE_MODE.ELTWISE_MUL)

    def emit_div(self, index, node):
        output = self.layout(node.output[0])
        numerator = self._binary_operand(index, 0, output)
        denominator = self._binary_operand(index, 1, output)
        inverse = self.unary_aux[index]
        self.emitter.emit_identity_lalu(
            denominator, inverse, self.identity_address,
            udc.LALU_MODE.MODE_RECIP, scalar=1.0)
        self.emitter.engine.eltwise_core_dram(
            M=output.rows, N=output.padded_last,
            dram_a=numerator.address, dram_b=inverse.address,
            dram_out=output.address, mode=udc.UE_MODE.ELTWISE_MUL)

    def _stage_conv_input(self, index, node):
        source = self.layout(node.input[0])
        aux = self.conv_aux[index]
        packed, attrs = aux["input"], aux["attrs"]
        pads = tuple(int(value) for value in attrs.get(
            "pads", (0, 0, 0, 0)))
        _batch, channels, height, width = source.shape
        padded_height, padded_width, packed_channels = packed.shape
        if packed_channels != channels:
            raise RuntimeError("Conv packed-input channel mismatch")
        engine = self.emitter.engine
        matrix = aux["input_matrix"]
        transposed = aux["input_transposed"]
        self.emitter.emit_zero(matrix)
        engine.accelerator_memory_to_sram(
            source.address, 0, 0, memcpy_length_bytes=source.size_bytes)
        engine.sram_to_accelerator_memory(
            0, matrix.address, 0, memcpy_length_bytes=source.size_bytes)
        engine.bf16_transpose_core(
            M=udc.UE_VECTOR_SIZE, N=aux["input_columns"],
            INPUT_DRAM_ADDR=matrix.address,
            OUTPUT_DRAM_ADDR=transposed.address,
            IDENTITY_DRAM_ADDR=self.identity_address)
        self.emitter.emit_zero(packed)
        row_bytes = packed.padded_last * 2
        for row in range(height):
            source_address = transposed.address + (
                row * source.padded_last) * row_bytes
            destination_address = packed.address + (
                (row + pads[0]) * padded_width + pads[1]) * row_bytes
            size = width * row_bytes
            engine.accelerator_memory_to_sram(
                source_address, 0, 0, memcpy_length_bytes=size)
            engine.sram_to_accelerator_memory(
                0, destination_address, 0, memcpy_length_bytes=size)

    def _unstage_conv_output(self, index, node):
        aux = self.conv_aux[index]
        packed = aux["output"]
        output = self.layout(node.output[0])
        engine = self.emitter.engine
        matrix, transposed = aux["output_matrix"], aux["output_transposed"]
        self.emitter.emit_zero(matrix)
        engine.accelerator_memory_to_sram(
            packed.address, 0, 0, memcpy_length_bytes=packed.size_bytes)
        engine.sram_to_accelerator_memory(
            0, matrix.address, 0, memcpy_length_bytes=packed.size_bytes)
        engine.bf16_transpose_core(
            M=aux["output_rows"], N=udc.UE_VECTOR_SIZE,
            INPUT_DRAM_ADDR=matrix.address,
            OUTPUT_DRAM_ADDR=transposed.address,
            IDENTITY_DRAM_ADDR=self.identity_address)
        self.emitter.emit_zero(output)
        _batch, channels, height, width = output.shape
        for channel in range(channels):
            for row in range(height):
                source_address = transposed.address + (
                    channel * aux["output_rows"] + row * width) * 2
                destination_address = output.address + (
                    (channel * height + row) * output.padded_last) * 2
                engine.accelerator_memory_to_sram(
                    source_address, 0, 0, memcpy_length_bytes=width * 2)
                engine.sram_to_accelerator_memory(
                    0, destination_address, 0, memcpy_length_bytes=width * 2)

    def _emit_depthwise_conv(self, index, node, resource):
        aux = self.conv_aux[index]
        packed_input, packed_output, attrs = (
            aux["input"], aux["output"], aux["attrs"])
        channels = packed_input.logical_last
        cpad = packed_input.padded_last
        input_width = packed_input.shape[1]
        output_width = packed_output.shape[1]
        strides = tuple(int(value) for value in attrs.get("strides", (1, 1)))
        dilations = tuple(int(value) for value in attrs.get(
            "dilations", (1, 1)))
        if packed_input.shape[0] != 1 or packed_output.shape[0] != 1:
            raise RuntimeError(f"node {index}: depthwise H must be one")
        chunk_width = resource["chunk_width"]
        a_sram = 0x10000
        temporary_sram = 0x40000
        weight_sram = 0x80000
        accumulator_sram = 0xC0000
        for output_start in range(0, output_width, chunk_width):
            take = min(chunk_width, output_width - output_start)
            elements = take * cpad
            self.emitter.engine.accelerator_memory_to_sram(
                resource["bias_address"], accumulator_sram, elements)
            for tap, weight_address in enumerate(resource["tap_addresses"]):
                for offset in range(take):
                    input_x = ((output_start + offset) * strides[1]
                               + tap * dilations[1])
                    if not 0 <= input_x < input_width:
                        raise RuntimeError("pre-padded depthwise index is invalid")
                    self.emitter.engine.accelerator_memory_to_sram(
                        packed_input.address + input_x * cpad * 2,
                        a_sram + offset * cpad * 2, cpad)
                self.emitter.engine.accelerator_memory_to_sram(
                    weight_address, weight_sram, elements)
                self.emitter.engine.eltwise_mul_core(
                    a_sram, weight_sram, temporary_sram, elements)
                self.emitter.engine.eltwise_add_core(
                    temporary_sram, accumulator_sram,
                    accumulator_sram, elements)
            self.emitter.engine.sram_to_accelerator_memory(
                accumulator_sram,
                packed_output.address + output_start * cpad * 2,
                elements)

    def emit_conv(self, index, node):
        self._stage_conv_input(index, node)
        resource = self.conv_resources[index]
        if resource["kind"] == "dense":
            shared._emit_conv(
                self.emitter.engine, resource["plan"],
                self.emitter.zero_address)
        else:
            self._emit_depthwise_conv(index, node, resource)
        self._unstage_conv_output(index, node)

    def _emit_row_copy(self, source_address: int, destination_address: int,
                       elements: int = 64):
        self.emitter.engine.accelerator_memory_to_sram(
            source_address, 0, elements)
        self.emitter.engine.sram_to_accelerator_memory(
            0, destination_address, elements)

    def _emit_gru_matmul(self, source_address, weight_address, bias_address,
                         output_address):
        self.emitter.engine.matmat_mul_core(
            M=1, K=64, N=192,
            A_DRAM_ADDR=source_address,
            B_DRAM_ADDR=weight_address,
            OUTPUT_DRAM_ADDR=output_address,
            C_DRAM_ADDR=bias_address, bias_mode="broadcast_N")

    def _emit_gru_activation(self, source_address, output_address, kind):
        if kind == "sigmoid":
            self.emitter.engine.activation_core(
                M=1, N=64, A_DRAM_ADDR=source_address,
                OUTPUT_DRAM_ADDR=output_address,
                IDENTITY_DRAM_ADDR=self.identity_address,
                activation="sigmoid")
        elif kind == "tanh":
            tanh_core_dram(
                self.emitter.engine, M=1, N=64,
                A_DRAM_ADDR=source_address,
                OUTPUT_DRAM_ADDR=output_address,
                IDENTITY_DRAM_ADDR=self.identity_address)
        else:
            raise ValueError(kind)

    def _emit_gru_eltwise(self, left, right, output, mode):
        self.emitter.engine.eltwise_core_dram(
            M=1, N=64, dram_a=left, dram_b=right,
            dram_out=output, mode=mode)

    def emit_gru(self, index, node):
        attrs = {
            value.name: self.onnx.helper.get_attribute_value(value)
            for value in node.attribute
        }
        if (attrs.get("direction") != b"bidirectional"
                or int(attrs.get("hidden_size", 0)) != 64
                or int(attrs.get("linear_before_reset", 0)) != 1):
            raise RuntimeError(f"node {index}: GRU attributes changed")
        source = self.layout(node.input[0])
        output = self.layout(node.output[0])
        output_h = self.layout(node.output[1])
        initial_h = self.layout(node.input[5])
        weights = self.layout(node.input[1])
        recurrent = self.layout(node.input[2])
        biases = self.layout(node.input[3])
        sequence = source.shape[0]
        if (source.shape != (sequence, 1, 64)
                or output.shape != (sequence, 2, 1, 64)
                or output_h.shape != (2, 1, 64)):
            raise RuntimeError(f"node {index}: GRU tensor shapes changed")
        scratch = self.gru_aux[index]
        xg, hg = scratch["xg"], scratch["hg"]
        z, r = scratch["z"], scratch["r"]
        candidate, temporary, hidden = (
            scratch["candidate"], scratch["tmp"], scratch["h"])
        gate_bytes = 64 * 2
        matrix_bytes = 192 * 64 * 2
        bias_row_bytes = 384 * 2

        for direction in range(2):
            current_h = initial_h.address + direction * gate_bytes
            order = range(sequence) if direction == 0 \
                else range(sequence - 1, -1, -1)
            for timestep in order:
                x_address = source.address + timestep * gate_bytes
                self._emit_gru_matmul(
                    x_address,
                    weights.address + direction * matrix_bytes,
                    biases.address + direction * bias_row_bytes,
                    xg.address)
                self._emit_gru_matmul(
                    current_h,
                    recurrent.address + direction * matrix_bytes,
                    biases.address + direction * bias_row_bytes + 192 * 2,
                    hg.address)

                # ONNX gates are ordered z, r, h. For linear_before_reset=1:
                # z=sigmoid(XWz+HRz), r=sigmoid(XWr+HRr),
                # n=tanh(XWh + r*(HRh)), h'=(h-n)*z+n.
                self._emit_gru_eltwise(
                    xg.address, hg.address, temporary.address,
                    udc.UE_MODE.ELTWISE_ADD)
                self._emit_gru_activation(
                    temporary.address, z.address, "sigmoid")
                self._emit_gru_eltwise(
                    xg.address + gate_bytes, hg.address + gate_bytes,
                    temporary.address, udc.UE_MODE.ELTWISE_ADD)
                self._emit_gru_activation(
                    temporary.address, r.address, "sigmoid")
                self._emit_gru_eltwise(
                    hg.address + 2 * gate_bytes, r.address,
                    temporary.address, udc.UE_MODE.ELTWISE_MUL)
                self._emit_gru_eltwise(
                    xg.address + 2 * gate_bytes, temporary.address,
                    candidate.address, udc.UE_MODE.ELTWISE_ADD)
                self._emit_gru_activation(
                    candidate.address, candidate.address, "tanh")
                self._emit_gru_eltwise(
                    current_h, candidate.address, temporary.address,
                    udc.UE_MODE.ELTWISE_SUB)
                self._emit_gru_eltwise(
                    temporary.address, z.address, temporary.address,
                    udc.UE_MODE.ELTWISE_MUL)
                self._emit_gru_eltwise(
                    temporary.address, candidate.address, hidden.address,
                    udc.UE_MODE.ELTWISE_ADD)
                current_h = hidden.address
                y_row = (timestep * 2 + direction) * gate_bytes
                self._emit_row_copy(current_h, output.address + y_row)
            self._emit_row_copy(
                current_h, output_h.address + direction * gate_bytes)

    def emit_node(self, index, node):
        kind = node.op_type
        if kind in ("Reshape", "Unsqueeze", "Squeeze", "Flatten"):
            return self.emit_view(index, node)
        if kind == "Transpose":
            return self.emit_transpose(index, node)
        if kind == "Slice":
            return self.emit_slice(node)
        if kind == "Gather":
            return self.emit_gather(index, node)
        if kind == "Concat":
            return self.emit_concat(index, node)
        if kind == "Split":
            return self.emit_split(node)
        if kind in ("Gemm", "MatMul"):
            return self.emit_linear(index, node)
        if kind == "Add":
            return self.emit_binary(index, node, udc.UE_MODE.ELTWISE_ADD)
        if kind == "Mul":
            return self.emit_binary(index, node, udc.UE_MODE.ELTWISE_MUL)
        if kind == "Sub":
            return self.emit_binary(index, node, udc.UE_MODE.ELTWISE_SUB)
        if kind == "Div":
            return self.emit_div(index, node)
        if kind == "Pow":
            return self.emit_pow(node)
        if kind == "Sqrt":
            return self.emit_sqrt(index, node)
        if kind == "ReduceSum":
            return self.emit_reduce_sum(index, node)
        if kind in ("Relu", "Sigmoid", "Tanh", "Log"):
            return self.emit_activation(node)
        if kind == "LayerNormalization":
            return self.emit_layer_norm(node)
        if kind == "Conv":
            return self.emit_conv(index, node)
        if kind == "GRU":
            return self.emit_gru(index, node)
        raise RuntimeError(f"node {index}: unsupported operator {kind}")

    def compile(self):
        previous_axi_width = udc.UE_AXI_DATA_WIDTH_BITS
        udc.UE_AXI_DATA_WIDTH_BITS = 256
        try:
            program_address = self.emitter.begin_program()
            for index, node in enumerate(self.model.graph.node):
                name = node.name or f"{node.op_type}_{index}"
                self.emitter.mark_operation(
                    name, index, lambda i=index, n=node: self.emit_node(i, n))
            # state_out is committed only after all consumers read state_in.
            state_out = self.layout("state_out")
            state_in = self.layout("state_in")
            self.emitter.mark_operation(
                "@state_commit", len(self.model.graph.node),
                lambda: self.emitter.emit_identity_copy(state_out, state_in))
            program = self.emitter.finish_program(program_address)
        finally:
            udc.UE_AXI_DATA_WIDTH_BITS = previous_axi_width
        image = torch.frombuffer(
            bytearray(self.emitter.image.data), dtype=torch.uint8).clone()
        if image.numel() > MODEL_LIMIT - MODEL_BASE:
            raise RuntimeError("DPDFNet deployment image exceeds model arena")
        return {
            "format": FORMAT,
            "onnx_sha256": self.digest,
            "precision": "BF16/IF8-INT",
            "full_graph": True,
            "one_halt": True,
            "stateful": True,
            "model_base": MODEL_BASE,
            "model_image": image,
            "model_sha256": manifest_sha256(image),
            "program_address": program_address,
            "program_offset": program_address - MODEL_BASE,
            "program_size": len(program),
            "program_sha256": hashlib.sha256(program).hexdigest(),
            "tensor_end": self.tensor_cursor,
            "tensors": tensor_manifest(self.layouts),
            "operations": self.emitter.operation_ranges,
            "graph_operations": len(self.model.graph.node),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "dpdfnet_bin" / "dpdfnet2-andromeda.bin")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        import onnx
    except ImportError:
        parser.error("onnx is required to compile DPDFNet2")
    model_path = args.model.expanduser().resolve()
    if args.download:
        model_path = download_model(model_path)
    elif not model_path.is_file():
        parser.error(f"model not found: {model_path}; pass --download")
    if args.output.exists() and not args.force:
        parser.error(f"output exists: {args.output}; pass --force")
    digest = validate_digest(model_path)
    model = onnx.load(str(model_path), load_external_data=False)
    # The shared matrix/CONV planners retain useful interactive diagnostics,
    # but a 472-node production compile would otherwise print thousands of
    # lines.  Keep the CLI deterministic and concise; exceptions still escape.
    diagnostics = io.StringIO()
    with contextlib.redirect_stdout(diagnostics):
        hardware = GraphCompiler(onnx, model, digest).compile()
    payload = {
        "format": FORMAT,
        "model": "dpdfnet2",
        "onnx_sha256": digest,
        "streaming_abi": {
            "inputs": {"spec": [1, 1, 161, 2], "state_in": [45424]},
            "outputs": {"spec_e": [1, 1, 161, 2], "state_out": [45424]},
        },
        "hardware": hardware,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    print(json.dumps({
        "artifact": str(args.output.resolve()),
        "onnx_sha256": digest,
        "model_image_bytes": int(hardware["model_image"].numel()),
        "program_instructions": hardware["program_size"] // 32,
        "operations": len(hardware["operations"]),
    }, indent=2))


if __name__ == "__main__":
    main()
