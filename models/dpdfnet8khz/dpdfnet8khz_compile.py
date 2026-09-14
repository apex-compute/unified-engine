#!/usr/bin/env python3
"""Compile the pinned native 8-kHz DPDFNet2 graph into one resident program."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "dpdfnet"
for path in (HERE, SHARED, HERE.parents[1]):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import user_dma_core as udc
from dpdfnet_compile import GraphCompiler as SharedGraphCompiler
import yolov5_precompiled as shared
from yolov5_common import quantize_conv_gather_if8
from dpdfnet_precompiled import (
    DeviceEmitter, copy_patterns, padded_row_patterns, physical_indices,
    transform_source_indices,
)
from dpdfnet8khz_common import DEFAULT_MODEL_PATH, download_model, validate_digest
from dpdfnet8khz_precompiled import FORMAT


class AlignedMappingEmitter(DeviceEmitter):
    """Keep native 80/81/50-wide layout copies on encodable DMA boundaries."""

    @classmethod
    def adopt(cls, emitter, scratch, source_scratch):
        result = cls.__new__(cls)
        result.__dict__.update(emitter.__dict__)
        result.copy_scratch = scratch
        result.source_scratch = source_scratch
        result.mapping_resources = {}
        result.planning = True
        return result

    def _dma_mapping_safe(self, source, destination, source_indices, destination_indices):
        if (source.logical_last == source.padded_last or source.name in self._zero_padding):
            if padded_row_patterns(source, destination, source_indices, destination_indices) is not None:
                return True
        beat = udc.ue_axi_beat_bytes()
        return all((source.address + start * 2) % beat == 0
                   and (destination.address + target * 2) % beat == 0
                   and (stride == 1 or (stride * 2 % beat == 0
                                        and stride * 2 <= udc.UE_STRIDE_JUMP_MAX_BYTES))
                   for start, target, _, stride in copy_patterns(source_indices, destination_indices))

    def _resource(self, source, destination, source_indices, destination_indices, preserve):
        source_indices = np.asarray(source_indices, dtype=np.int64).reshape(-1)
        destination_indices = np.asarray(destination_indices, dtype=np.int64).reshape(-1)
        if source_indices.size != destination_indices.size:
            raise ValueError("mapping lengths differ")
        if (np.any(source_indices < 0) or np.any(source_indices >= source.physical_elements)
                or np.any(destination_indices < 0)
                or np.any(destination_indices >= destination.physical_elements)):
            raise ValueError("mapping exceeds its tensor allocation")
        if np.unique(destination_indices).size != destination_indices.size:
            raise ValueError("mapping writes a destination lane more than once")
        key = (source, destination, preserve, source_indices.tobytes(), destination_indices.tobytes())
        if key in self.mapping_resources:
            return self.mapping_resources[key]
        if not self.planning:
            raise RuntimeError("mapping selector was not allocated before program capture")
        groups = {}
        for origin, target in zip(source_indices, destination_indices):
            destination_row, destination_lane = divmod(int(target), 64)
            source_row, source_lane = divmod(int(origin), 64)
            groups.setdefault(destination_row, {}).setdefault(source_row, []).append(
                (destination_lane, source_lane))
        resources = []
        for destination_row, pieces in sorted(groups.items()):
            touched = {lane for entries in pieces.values() for lane, _ in entries}
            keep = None
            if preserve and len(touched) < 64:
                selector = torch.eye(64, dtype=torch.bfloat16)
                selector[list(touched)] = 0
                keep = self.allocate_constant(selector)
            rows = []
            for source_row, entries in sorted(pieces.items()):
                selector = torch.zeros(64, 64, dtype=torch.bfloat16)
                for destination_lane, source_lane in entries:
                    selector[destination_lane, source_lane] = 1
                rows.append((source_row, self.allocate_constant(selector)))
            resources.append((destination_row, keep, rows))
        self.mapping_resources[key] = resources
        return resources

    def _emit_selectors(self, source, destination, source_indices, destination_indices, preserve):
        resources = self._resource(source, destination, source_indices, destination_indices, preserve)
        for row, keep, pieces in resources:
            target = destination.address + row * 128
            accumulated = keep is not None
            if keep is not None:
                self.engine.matmat_mul_core(
                    M=1, K=64, N=64, A_DRAM_ADDR=target, B_DRAM_ADDR=keep,
                    OUTPUT_DRAM_ADDR=target, is_B_quantized=False)
            for source_row, selector in pieces:
                source_address = source.address + source_row * 128
                valid = min(64, source.logical_last - (source_row * 64 % source.padded_last))
                if valid < 64 and source.name not in self._zero_padding:
                    # Unused activation lanes may contain NaN/Inf. A selector
                    # matrix alone cannot remove them because zero*NaN is NaN.
                    self.emit_zero(self.source_scratch)
                    self.engine.accelerator_memory_to_sram(
                        source_address, 0, 0, memcpy_length_bytes=valid * 2)
                    self.engine.sram_to_accelerator_memory(
                        0, self.source_scratch.address, 0, memcpy_length_bytes=valid * 2)
                    source_address = self.source_scratch.address
                output = self.copy_scratch.address if accumulated else target
                self.engine.matmat_mul_core(
                    M=1, K=64, N=64, A_DRAM_ADDR=source_address,
                    B_DRAM_ADDR=selector, OUTPUT_DRAM_ADDR=output, is_B_quantized=False)
                if accumulated:
                    self.engine.eltwise_core_dram(
                        M=1, N=64, dram_a=target, dram_b=output,
                        dram_out=target, mode=udc.UE_MODE.ELTWISE_ADD)
                accumulated = True

    def emit_mapping(self, source, destination, source_physical, *, zero=True):
        source_physical = np.asarray(source_physical, dtype=np.int64).reshape(-1)
        if source_physical.size != destination.logical_elements:
            raise ValueError("mapping does not cover its logical output")
        destination_physical = physical_indices(destination)
        if self._dma_mapping_safe(source, destination, source_physical, destination_physical):
            super().emit_mapping(source, destination, source_physical, zero=zero)
        else:
            self._emit_selectors(source, destination, source_physical, destination_physical,
                                 preserve=not zero)
            if zero:
                self.mark_padding_zero(destination)

    def emit_scatter(self, source, destination, source_physical, destination_physical):
        if self._dma_mapping_safe(source, destination, source_physical, destination_physical):
            return super().emit_scatter(source, destination, source_physical, destination_physical)
        self._emit_selectors(source, destination, source_physical, destination_physical, preserve=True)

    def emit_clear_padding(self, destination):
        padding = destination.padded_last - destination.logical_last
        if not padding:
            return
        if destination.logical_last * 2 % udc.ue_axi_beat_bytes() == 0:
            return super().emit_clear_padding(destination)
        # Clear a partial beat without rounding its destination address down.
        # The short write into a zeroed aligned tile excludes NaN/Inf padding;
        # then a complete 128-byte row replaces the original final row.
        last_block = destination.logical_last // 64 * 64
        valid_bytes = (destination.logical_last - last_block) * 2
        for row in range(destination.rows):
            address = destination.address + (row * destination.padded_last + last_block) * 2
            self.emit_zero(self.copy_scratch)
            self.engine.accelerator_memory_to_sram(address, 0, 0, memcpy_length_bytes=valid_bytes)
            self.engine.sram_to_accelerator_memory(
                0, self.copy_scratch.address, 0, memcpy_length_bytes=valid_bytes)
            self.engine.accelerator_memory_to_sram(self.copy_scratch.address, 0, 64)
            self.engine.sram_to_accelerator_memory(0, address, 64)
        self.mark_padding_zero(destination)


class GraphCompiler(SharedGraphCompiler):
    """Reuse the convolution/recurrent compiler with native 8-kHz geometry."""

    def __init__(self, onnx, model, digest):
        super().__init__(onnx, model, digest)
        self.emitter = AlignedMappingEmitter.adopt(
            self.emitter, self.mapping_scratch, self.mapping_source_scratch)
        self.pixel8_selectors = {}
        for index, resource in self.pixel8_aux.items():
            width = resource["width"]
            matrices = []
            for parity in range(2):
                selector = torch.zeros(128 if width > 32 else 64, 64, dtype=torch.bfloat16)
                for column in range(width):
                    selector[2 * column + parity, column] = 1
                matrices.append(self.emitter.allocate_constant(selector))
            self.pixel8_selectors[index] = matrices

    def _prepare_conv_resources(self):
        resources = super()._prepare_conv_resources()
        for index, resource in resources.items():
            if resource["kind"] != "dense" or resource["plan"]["data_type"] != udc.TYPE.IF4:
                continue
            # The shared YOLO policy favors IF4 for pointwise/channel layouts.
            # Native speech enhancement is sensitive to that weight error.
            # Every pinned 8-kHz dense kernel fits the existing IF8 gather
            # format; reuse its quantizer and precision-aware tile planner.
            node = self.model.graph.node[index]
            weight = torch.from_numpy(np.asarray(self.initializers[node.input[1]], dtype=np.float32))
            bias = (torch.from_numpy(np.asarray(self.initializers[node.input[2]], dtype=np.float32))
                    if len(node.input) > 2 else None)
            conv = torch.nn.Conv2d(weight.shape[1], weight.shape[0], tuple(weight.shape[2:]),
                                   bias=bias is not None)
            with torch.no_grad():
                conv.weight.copy_(weight)
                if bias is not None:
                    conv.bias.copy_(bias)
            prepared = quantize_conv_gather_if8(conv, None)
            encoded = {
                "precision": "if8", "layout": "gather",
                "codes_packed": self._pack_quantized_codes(prepared.codes, "if8"),
                "codes_shape": list(prepared.codes.shape),
                "block_scales": prepared.block_scales,
                "bias": (torch.empty(0, dtype=torch.bfloat16)
                         if prepared.bias is None else prepared.bias),
            }
            original = resource["plan"]
            resource["plan"] = shared._prepare_conv_plan(
                original["operation"], encoded, original["source"], original["destination"],
                self.emitter.image, allow_half_vector_output=True)
        return resources

    def _plan_auxiliary_layouts(self):
        super()._plan_auxiliary_layouts()
        self.mapping_scratch = self._allocate_aux("@native8/copy", (64,))
        self.mapping_source_scratch = self._allocate_aux("@native8/copy_source", (64,))
        self.pixel8_aux = {}
        for index, node in enumerate(self.model.graph.node):
            output = self.layouts[node.output[0]]
            if node.op_type == "ReduceSum":
                source = self.layouts[node.input[0]]
                axes = self._integer_initializer(node.input[1]).tolist()
                if source.shape == (1, 1, 5, 80) and axes == [2] and output.shape == (1, 1, 80):
                    self.direct_reduce_aux.add(index)
            if node.op_type == "Gather":
                source = self.layouts[node.input[0]]
                if source.logical_last == 2 and output.rows == 5 and output.logical_last == 80:
                    self.complex_gather_aux[index] = {
                        "input": self._allocate_aux(f"@native8/gather/{index}/input", (64, 64)),
                        "transposed": self._allocate_aux(f"@native8/gather/{index}/transpose", (64, 64)),
                    }
            if node.op_type == "Reshape" and index and self.model.graph.node[index - 1].op_type == "Transpose":
                transpose = self.model.graph.node[index - 1]
                attrs = {a.name: self.onnx.helper.get_attribute_value(a) for a in transpose.attribute}
                source = self.layouts[transpose.input[0]]
                if (node.input[0] == transpose.output[0] and source.shape[:4] == (1, 2, 64, 1)
                        and len(source.shape) == 5 and source.shape[4] in (10, 20, 40)
                        and attrs.get("perm") == [0, 2, 3, 4, 1]
                        and output.shape == (1, 64, 1, 2 * source.shape[4])):
                    self.pixel_shuffle_skip.add(index - 1)
                    self.pixel8_aux[index] = {
                        "source": source, "width": source.shape[4],
                        "odd": self._allocate_aux(f"@native8/pixel/{index}/odd", output.shape),
                    }

    def emit_view(self, index, node):
        if index not in self.pixel8_aux:
            return super().emit_view(index, node)
        resource = self.pixel8_aux[index]
        source, output = resource["source"], self.layout(node.output[0])
        engine = self.emitter.engine
        branch_bytes = 64 * source.padded_last * 2
        for parity, selector in enumerate(self.pixel8_selectors[index]):
            engine.matmat_mul_core(
                M=64, K=64, N=output.padded_last,
                A_DRAM_ADDR=source.address + parity * branch_bytes,
                B_DRAM_ADDR=selector,
                OUTPUT_DRAM_ADDR=output.address if parity == 0 else resource["odd"].address,
                is_B_quantized=False)
        engine.eltwise_core_dram(
            M=output.rows, N=output.padded_last, dram_a=output.address,
            dram_b=resource["odd"].address, dram_out=output.address, mode=udc.UE_MODE.ELTWISE_ADD)
        self.emitter.mark_padding_zero(output)

    def emit_pad(self, node):
        source, output = self.layout(node.input[0]), self.layout(node.output[0])
        attrs = {a.name: self.onnx.helper.get_attribute_value(a) for a in node.attribute}
        pads = self._integer_initializer(node.input[1])
        if attrs.get("mode") != b"reflect" or pads.size != len(source.shape) * 2:
            raise RuntimeError("native8 Pad requires static full-rank reflect padding")
        mapping = physical_indices(source).reshape(source.shape)
        pairs = list(zip(pads[:len(source.shape)], pads[len(source.shape):]))
        mapping = np.pad(mapping, pairs, mode="reflect")
        if tuple(mapping.shape) != output.shape:
            raise RuntimeError("reflect Pad inferred shape does not match its output")
        self.emitter.emit_mapping(source, output, mapping.reshape(-1))

    def emit_node(self, index, node):
        if node.op_type == "Pad":
            return self.emit_pad(node)
        return super().emit_node(index, node)

    def compile(self):
        # Discover arbitrary layout selectors during an offline dry capture.
        # Allocate every selector before fixing the resident program address.
        previous_width = udc.UE_AXI_DATA_WIDTH_BITS
        udc.UE_AXI_DATA_WIDTH_BITS = 256
        try:
            initial_padding_facts = set(self.emitter._zero_padding)
            self.emitter.begin_program()
            for index, node in enumerate(self.model.graph.node):
                self.emit_node(index, node)
            self.emitter.emit_identity_copy(self.layout("state_out"), self.layout("state_in"))
            self.emitter.engine.stop_capture()
            self.emitter.engine = None
            self.emitter.planning = False
            self.emitter._zero_padding = initial_padding_facts
            hardware = super().compile()
        finally:
            udc.UE_AXI_DATA_WIDTH_BITS = previous_width
        hardware["format"] = FORMAT
        hardware["axi_data_width_bits"] = 256
        hardware["dense_convolution_precision"] = "IF8"
        hardware["optimizations"] = list(getattr(self, "optimization_names", ()))
        return hardware


class CopyOptimizationMixin:
    """Batch repeated layout work while retaining the same BF16 operations."""

    def _plan_auxiliary_layouts(self):
        super()._plan_auxiliary_layouts()
        self.state_batch_scratch = self._allocate_aux("@native8/state_batch", (128, 64))
        for index, node in enumerate(self.model.graph.node):
            if node.op_type != "Reshape":
                continue
            source, output = self.layouts[node.input[0]], self.layouts[node.output[0]]
            if (source.rows == 80 and source.logical_last == 10
                    and output.rows == 400 and output.logical_last == 2):
                self.split10_aux[index] = self._allocate_aux(
                    f"@native8/split10/{index}", source.shape)

    def emit_binary(self, index, node, mode):
        super().emit_binary(index, node, mode)
        if mode in (udc.UE_MODE.ELTWISE_ADD, udc.UE_MODE.ELTWISE_SUB):
            output = self.layout(node.output[0])
            inputs = [self.layout(name) for name in node.input]
            if all(value.shape == output.shape and (
                    value.logical_last == value.padded_last
                    or value.name in self.emitter._zero_padding) for value in inputs):
                self.emitter.mark_padding_zero(output)

    def emit_transpose(self, index, node):
        super().emit_transpose(index, node)
        resource = self.transpose_aux.get(index)
        if resource is not None and resource["direction"] == "wide_channels":
            # The shared lowering clears its 64-row input tile before loading
            # the logical channels. Unused output channels are exactly zero.
            self.emitter.mark_padding_zero(self.layout(node.output[0]))

    def _load_pair_vector(self, address, valid_lanes, *, zero_padding_known):
        engine = self.emitter.engine
        if valid_lanes == 64 or zero_padding_known:
            engine.accelerator_memory_to_sram(address, 0, 64)
            return
        # A one-hot product still propagates NaN from unselected lanes. Copy
        # only the logical prefix into a zeroed, aligned DRAM row first.
        scratch = self.emitter.source_scratch
        self.emitter.emit_zero(scratch)
        engine.accelerator_memory_to_sram(address, 0, 0, memcpy_length_bytes=valid_lanes * 2)
        engine.sram_to_accelerator_memory(
            0, scratch.address, 0, memcpy_length_bytes=valid_lanes * 2)
        engine.accelerator_memory_to_sram(scratch.address, 0, 64)

    def _emit_pair_rows(self, source, output, *, split10):
        engine = self.emitter.engine
        identity_sram, output_sram = 0x80000, 0x2000
        engine.accelerator_memory_to_sram(self.identity_address, identity_sram, 64 * 64)
        zero_known = source.name in self.emitter._zero_padding
        bulk_bytes = source.size_bytes + output.size_bytes
        if (bulk_bytes <= udc.URAM_NEAR_FULL_SIZE
                and (not split10 or zero_known)):
            # Keep the complete input before the complete output in URAM_A.
            # The identity occupies only URAM_B. This retains identical N=2
            # matvecs while removing a DMA pair per input row/tile.
            tail = source.logical_elements % 64
            patch_tail = not split10 and tail != 0 and not zero_known
            if patch_tail:
                self._load_pair_vector(
                    source.address + source.size_bytes - 128, tail,
                    zero_padding_known=False)
            engine.accelerator_memory_to_sram(
                source.address, 0, 0, memcpy_length_bytes=source.size_bytes)
            if patch_tail:
                engine.accelerator_memory_to_sram(
                    self.emitter.source_scratch.address, source.size_bytes - 128, 64)
            pairs_per_source = 5 if split10 else 32
            for row in range(output.rows):
                source_row, pair = divmod(row, pairs_per_source)
                engine.start_queue_for_bf16_matvec_operation(
                    max_clear_en=0, fmax_context_addr=0,
                    vector_sram_start_addr=source_row * 128,
                    matrix_sram_start_addr=identity_sram + pair * 256,
                    output_sram_wb_addr=source.size_bytes + row * 128,
                    K=64, N=2, stride_z=64)
            engine.sram_to_accelerator_memory(
                source.size_bytes, output.address, 0,
                memcpy_length_bytes=output.size_bytes)
            self.emitter.mark_padding_zero(output)
            return
        if split10:
            groups = ((row, row * 5, 5, 10) for row in range(source.rows))
        else:
            groups = ((chunk, row, min(32, output.rows - row),
                       min(32, output.rows - row) * 2)
                      for chunk, row in enumerate(range(0, output.rows, 32)))
        for source_row, destination_row, pairs, valid_lanes in groups:
            self._load_pair_vector(
                source.address + source_row * 128, valid_lanes,
                zero_padding_known=zero_known)
            for pair in range(pairs):
                engine.start_queue_for_bf16_matvec_operation(
                    max_clear_en=0, fmax_context_addr=0,
                    vector_sram_start_addr=0,
                    matrix_sram_start_addr=identity_sram + pair * 256,
                    output_sram_wb_addr=output_sram + pair * 128,
                    K=64, N=2, stride_z=64)
            engine.sram_to_accelerator_memory(
                output_sram, output.address + destination_row * 128, 0,
                memcpy_length_bytes=pairs * 128)
        self.emitter.mark_padding_zero(output)

    def emit_view(self, index, node):
        if index in self.pair_unpack_aux or index in self.split10_aux:
            return self._emit_pair_rows(
                self.layout(node.input[0]), self.layout(node.output[0]),
                split10=index in self.split10_aux)
        super().emit_view(index, node)

    def emit_state_copy(self, resource):
        # prepare_state_copy advances complete destination rows by 64 source
        # lanes. Their selectors therefore repeat exactly. Keep partial rows
        # on the original path, including its neighbor-preserving selector.
        if resource["prefix"]:
            super().emit_state_copy({**resource, "rows": []})
        rows = resource["rows"]
        position = 0
        while position < len(rows):
            first = rows[position]
            end = position + 1
            if first["keep"] is None:
                while end < min(len(rows), position + 128):
                    candidate = rows[end]
                    delta = end - position
                    if (candidate["keep"] is not None
                            or candidate["destination_row"] != first["destination_row"] + delta
                            or len(candidate["pieces"]) != len(first["pieces"])
                            or any(current[0] != original[0] + delta
                                   for current, original in zip(candidate["pieces"], first["pieces"]))):
                        break
                    end += 1
            if end == position + 1:
                super().emit_state_copy({**resource, "prefix": 0, "rows": [first]})
                position = end
                continue
            count = end - position
            target = resource["destination"].address + first["destination_row"] * 128
            engine = self.emitter.engine
            for part, (source_row, selector) in enumerate(first["pieces"]):
                output = self.state_batch_scratch.address if part else target
                engine.matmat_mul_core(
                    M=count, K=64, N=64,
                    A_DRAM_ADDR=resource["source"].address + source_row * 128,
                    B_DRAM_ADDR=selector, OUTPUT_DRAM_ADDR=output, is_B_quantized=False)
                if part:
                    engine.eltwise_core_dram(
                        M=count, N=64, dram_a=target, dram_b=output,
                        dram_out=target, mode=udc.UE_MODE.ELTWISE_ADD)
            position = end


from dpdfnet8khz_conv import ConvTransposeOptimizationMixin
from dpdfnet8khz_pack import PairPackOptimizationMixin
from dpdfnet8khz_reduce import ReductionOptimizationMixin
from dpdfnet8khz_state_shift import StateShiftOptimizationMixin
from dpdfnet8khz_reshape import CoefficientReshapeOptimizationMixin
from dpdfnet8khz_reshape80 import Reshape80OptimizationMixin


class OptimizedGraphCompiler(CoefficientReshapeOptimizationMixin, Reshape80OptimizationMixin,
                             StateShiftOptimizationMixin, PairPackOptimizationMixin,
                             ReductionOptimizationMixin, ConvTransposeOptimizationMixin,
                             CopyOptimizationMixin, GraphCompiler):
    """Native IF8 compiler with fused arithmetic and SRAM layout operations."""

    optimization_names = ("conv-transpose-n2", "conv-relu", "state-copy-batching",
                          "state-shift128", "complex-pair-unpack", "complex-pair-pack",
                          "complex-reduce", "coefficient-reshape128", "row80-unpadding")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=HERE / "dpdfnet8khz_bin/dpdfnet2_8khz-andromeda.bin")
    parser.add_argument("--force", action="store_true")
    lowering = parser.add_mutually_exclusive_group()
    lowering.add_argument("--optimize", dest="optimize", action="store_true", default=True,
                          help="use fused native layout and convolution lowering (default)")
    lowering.add_argument("--baseline", dest="optimize", action="store_false",
                          help="build the unoptimized IF8 reference for compiler comparisons")
    args = parser.parse_args()
    if args.output.expanduser().resolve() == args.model.expanduser().resolve():
        parser.error("--output must differ from the source ONNX model")
    if args.output.exists() and not args.force:
        parser.error("output exists; pass --force")
    model_path = args.model.expanduser().resolve()
    if args.download:
        model_path = download_model(model_path)
    if not model_path.is_file():
        parser.error("model does not exist; pass --download")
    import onnx

    digest = validate_digest(model_path)
    with contextlib.redirect_stdout(io.StringIO()):
        compiler = OptimizedGraphCompiler if args.optimize else GraphCompiler
        hardware = compiler(onnx, onnx.load(model_path), digest).compile()
    payload = {
        "format": FORMAT, "model": "dpdfnet2_8khz", "onnx_sha256": digest,
        "streaming_abi": {"inputs": {"spec": [1, 1, 81, 2], "state_in": [37860]},
                          "outputs": {"spec_e": [1, 1, 81, 2], "state_out": [37860]}},
        "hardware": hardware,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    print(json.dumps({"artifact": str(args.output.resolve()), "onnx_sha256": digest,
                      "model_image_bytes": hardware["model_image"].numel(),
                      "program_instructions": hardware["program_size"] // 32,
                      "operations": len(hardware["operations"])}, indent=2))


if __name__ == "__main__":
    main()
