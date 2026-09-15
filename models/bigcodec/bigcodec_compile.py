#!/usr/bin/env python3
"""Compile official BigCodec into one full-utterance Andromeda deployment bin."""

from __future__ import annotations

import argparse
from collections import Counter
import contextlib
from dataclasses import dataclass, field
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time

import torch

from bigcodec_common import CHECKPOINT_SHA256, DEFAULT_CHECKPOINT, HOP_LENGTH, SAMPLE_RATE, load_models, read_audio, sha256_file
from bigcodec_device import aligned, channels, elementwise, shared, udc, zero
from bigcodec_activation import SNAKE_ARGUMENT_LIMIT, activation_scratch_bytes, prepare_activation, emit_activation
from bigcodec_conv import (
    conv1d_output_shape, conv1d_padding_bytes, conv_transpose1d_output_shape,
    conv_transpose1d_padding_bytes, packed_bytes, prepare_conv1d,
    prepare_conv_transpose1d, emit_conv1d, emit_conv_transpose1d,
)
from bigcodec_lstm import scratch_bytes as lstm_scratch_bytes, prepare_lstm, emit_lstm, tanh_identity, tanh_scratch_bytes, TANH_ARGUMENT_LIMIT, TANH_CHUNK_ELEMENTS, _tanh_sram
from bigcodec_quantizer import quantizer_scratch_bytes, prepare_quantizer, emit_quantizer
from bigcodec_layout import MemoryLayout, LEGACY_LAYOUT, EXTENDED_LAYOUT, LARGE_PROGRAM_LAYOUT, LAYOUTS

FORMAT = "andromeda.bigcodec.whole-utterance-v1"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "bigcodec_bin" / "bigcodec-andromeda.bin"
CONV_SCATTER_BYTES = 512 * 1024
CONV_STREAM_BUDGET_PER_64_OUTPUTS = 256 * 1024
CONV_MAX_REUSE_PIXELS = 32
WAVEFORM_ROW_LANES = 64  # SRAM DMA advances in complete 128-byte rows.
WAVEFORM_ROW_BYTES = WAVEFORM_ROW_LANES * 2
WAVEFORM_TILE_ROWS = TANH_CHUNK_ELEMENTS // WAVEFORM_ROW_LANES


@dataclass
class Tensor:
    name: str
    shape: tuple[int, int]
    address: int = 0
    first: int = -1
    last: int = -1

    @property
    def size_bytes(self):
        return packed_bytes(self.shape)


@dataclass
class Operation:
    name: str
    op: str
    inputs: tuple[str, ...]
    output: str
    module: object = field(repr=False)
    plan: object = field(default=None, repr=False)
    scratch_bytes: int = 0
    scatter_address: int = 0
    weight_reuse_pixels: int = 1


@dataclass
class Graph:
    compiled_samples: int
    code_frames: int
    tensors: dict[str, Tensor]
    operations: list[Operation]
    output: str
    conv_precision: str = "if8"
    lstm_precision: str = "bf16"
    lstm_cell_precision: str = "bf16"
    lstm_tanh_precision: str = "bf16"
    lstm_math_scope: str = "both"
    lstm_fused_gates: bool = False
    center_quantizer_scores: bool = False
    compensated_codebook: bool = False
    filter_accumulation: str = "serial"
    filter_math_scope: str = "both"
    filter_stage: str = "both"
    output_address: int = shared.TENSOR_BASE
    output_bytes: int = 0
    tokens_address: int = 0
    tokens_offset: int = 0
    scratch_address: int = 0
    scratch_bytes: int = 0
    tensor_end: int = 0
    layout: MemoryLayout = LEGACY_LAYOUT


def compiled_sample_count(samples: int) -> int:
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive native 16 kHz sample count")
    return samples + HOP_LENGTH - samples % HOP_LENGTH


def convolution_reuse_pixels(module) -> int:
    """Spend bounded weight replication on the early, longest convolutions."""
    inputs = int(module.in_channels)
    kernel = int(module.kernel_size[0])
    if isinstance(module, torch.nn.ConvTranspose1d):
        # The helper emits ordinary convolutions for temporal polyphases.
        kernel = (kernel + module.stride[0] - 1) // module.stride[0]
    padded_blocks = (inputs + 63) // 64
    gather_blocks = (kernel * inputs + 63) // 64
    gather = inputs <= 255 and gather_blocks <= 4 and gather_blocks < kernel * padded_blocks
    blocks = gather_blocks if gather else kernel * padded_blocks
    bytes_per_pixel = 64 * blocks * 64  # 64 OC × IF8 blocks × 64 one-byte codes.
    return max(1, min(CONV_MAX_REUSE_PIXELS, CONV_STREAM_BUDGET_PER_64_OUTPUTS // bytes_per_pixel))


def build_graph(encoder, decoder, compiled_samples: int, *, conv_precision="if8", lstm_precision="bf16",
                lstm_cell_precision="bf16", lstm_tanh_precision="bf16", lstm_math_scope="both",
                lstm_fused_gates=False, center_quantizer_scores=False, compensated_codebook=False,
                filter_accumulation="serial", filter_math_scope="both", filter_stage="both") -> Graph:
    """Trace module structure and exact lengths without executing neural layers."""
    if compiled_samples < HOP_LENGTH or compiled_samples % HOP_LENGTH:
        raise ValueError("compiled_samples must be a positive multiple of 200")
    if conv_precision not in ("if8", "bf16"):
        raise ValueError("convolution precision must be if8 or bf16")
    if lstm_precision not in ("bf16", "encoder-if8", "if8"):
        raise ValueError("LSTM precision must be bf16, encoder-if8 or if8")
    if lstm_cell_precision not in ("bf16", "products", "compensated"):
        raise ValueError("LSTM cell precision must be bf16, products or compensated")
    if lstm_tanh_precision not in ("bf16", "compensated"):
        raise ValueError("LSTM tanh precision must be bf16 or compensated")
    if lstm_math_scope not in ("encoder", "decoder", "both"):
        raise ValueError("LSTM math scope must be encoder, decoder or both")
    if not isinstance(lstm_fused_gates, bool):
        raise ValueError("LSTM fused gates must be a bool")
    if not isinstance(center_quantizer_scores, bool):
        raise ValueError("Centered quantizer scores must be a bool")
    if not isinstance(compensated_codebook, bool):
        raise ValueError("Compensated codebook must be a bool")
    if filter_accumulation not in ("serial", "sorted", "matrix"):
        raise ValueError("Filter accumulation must be serial, sorted or matrix")
    if filter_math_scope not in ("encoder", "decoder", "both"):
        raise ValueError("Filter math scope must be encoder, decoder or both")
    if filter_stage not in ("up", "down", "both"):
        raise ValueError("Filter stage must be up, down or both")
    tensors = {"input": Tensor("input", (compiled_samples, 1), shared.INPUT_BASE)}
    operations = []

    def append(name, op, inputs, shape, module=None):
        output = name + ":out"
        if output in tensors:
            raise ValueError(f"Duplicate operation name {name}")
        tensors[output] = Tensor(output, tuple(shape))
        operations.append(Operation(name, op, tuple(inputs), output, module))
        return output

    def walk(module, source, name):
        shape = tensors[source].shape
        kind = type(module).__name__
        if isinstance(module, torch.nn.Sequential):
            for index, child in enumerate(module):
                source = walk(child, source, f"{name}.{index}")
            return source
        if kind in ("EncoderBlock", "DecoderBlock"):
            return walk(module.block, source, name + ".block")
        if kind == "ResidualUnit":
            result = walk(module.block, source, name + ".block")
            if tensors[result].shape != shape:
                raise ValueError(f"{name}: residual branch changes shape")
            return append(name + ".add", "add", (source, result), shape)
        if kind == "ResLSTM":
            if module.lstm.input_size != shape[1] or module.lstm.hidden_size != shape[1]:
                raise ValueError(f"{name}: LSTM width does not match input")
            return append(name, "lstm", (source,), shape, module)
        if kind == "Activation1d":
            return append(name, "activation", (source,), shape, module)
        if isinstance(module, torch.nn.Conv1d):
            if module.groups != 1 or module.padding_mode != "zeros":
                raise ValueError(f"{name}: unsupported Conv1d groups or padding mode")
            output_shape = conv1d_output_shape(shape, module.weight.shape,
                stride=module.stride[0], padding=module.padding[0], dilation=module.dilation[0])
            return append(name, "conv1d", (source,), output_shape, module)
        if isinstance(module, torch.nn.ConvTranspose1d):
            if module.groups != 1 or module.dilation != (1,):
                raise ValueError(f"{name}: unsupported transposed-convolution groups or dilation")
            output_shape = conv_transpose1d_output_shape(shape, module.weight.shape,
                stride=module.stride[0], padding=module.padding[0], output_padding=module.output_padding[0])
            return append(name, "conv_transpose1d", (source,), output_shape, module)
        if isinstance(module, torch.nn.Tanh):
            return append(name, "tanh", (source,), shape, module)
        raise ValueError(f"{name}: unsupported BigCodec module {kind}")

    encoded = walk(encoder.block, "input", "encoder.block")
    expected_codes = (compiled_samples // HOP_LENGTH, 1024)
    if tensors[encoded].shape != expected_codes:
        raise ValueError(f"Encoder output is {tensors[encoded].shape}, expected {expected_codes}")
    quantized = append("quantizer", "quantizer", (encoded,), expected_codes, decoder.quantizer)
    output = walk(decoder.model, quantized, "decoder.model")
    if tensors[output].shape != (compiled_samples, 1):
        raise ValueError(f"Decoder output is {tensors[output].shape}, expected {(compiled_samples, 1)}")
    return Graph(compiled_samples, expected_codes[0], tensors, operations, output,
                 conv_precision, lstm_precision, lstm_cell_precision, lstm_tanh_precision, lstm_math_scope,
                 lstm_fused_gates, center_quantizer_scores, compensated_codebook,
                 filter_accumulation, filter_math_scope, filter_stage)


def _filter_numerics(graph, operation):
    selected = graph.filter_math_scope == "both" or operation.name.startswith(graph.filter_math_scope + ".")
    return dict(filter_accumulation=graph.filter_accumulation if selected else "serial",
                filter_stage=graph.filter_stage)


def _lstm_numerics(graph, operation):
    selected = graph.lstm_math_scope == "both" or operation.name.startswith(graph.lstm_math_scope + ".")
    return dict(compensated_cell=selected and graph.lstm_cell_precision != "bf16",
                preserve_cell_residual=graph.lstm_cell_precision != "products",
                compensated_tanh=selected and graph.lstm_tanh_precision == "compensated",
                fused_projection=selected and graph.lstm_fused_gates)


def _bf16_convolutions():
    import bigcodec_conv_bf16
    return bigcodec_conv_bf16


def _waveform_tanh(graph, operation):
    return (operation.op == 'tanh' and operation.output == graph.output
            and graph.tensors[operation.output].shape[1] == 1)


def emit_waveform_tanh(engine, source, destination, rows, identity_address,
                       zero_address, mask_address):
    """Evaluate mono Tanh in SRAM while preserving complete pad64 sample rows.

    Native strided writes with 32/64-byte chunks advance by a full 128-byte SRAM
    row on RK 0x40519e0a, so sub-row scattering corrupts the sample sequence.
    Full-row contiguous transfers preserve the layout. A mask retains each
    row's first lane and zeros its padding; no DRAM scratch is needed.
    """
    if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
        raise ValueError('Waveform tanh needs a positive row count')
    size = rows * 128
    if (any(address < 0 or address % 128 for address in
                   (source, destination, identity_address, zero_address, mask_address))
            or source < destination + size and destination < source + size):
        raise ValueError('Waveform tanh needs aligned, disjoint pad64 tensors')
    zero(engine, destination, size, zero_address)
    engine.accelerator_memory_to_sram(identity_address, 0x80000, 64 * 64)
    # _tanh_sram uses B[0x80000:0x82000] and B[0x90000:0x96000].
    mask_sram = 0xB0000
    engine.accelerator_memory_to_sram(mask_address, mask_sram, TANH_CHUNK_ELEMENTS)
    for first in range(0, rows, WAVEFORM_TILE_ROWS):
        take = min(WAVEFORM_TILE_ROWS, rows - first)
        elements = take * WAVEFORM_ROW_LANES
        shared._copy_contiguous_or_strided_read(engine,
            source=source + first * 128, sram=0, total=elements * 2,
            chunk=WAVEFORM_ROW_BYTES, jump=WAVEFORM_ROW_BYTES)
        _tanh_sram(engine, 0, 0, elements)
        engine.eltwise_mul_core(0, mask_sram, 0, elements)
        shared._copy_contiguous_or_strided_write(engine, sram=0,
            destination=destination + first * 128, total=elements * 2,
            chunk=WAVEFORM_ROW_BYTES, jump=WAVEFORM_ROW_BYTES)


class _TensorCapacityError(ValueError):
    pass


class _ModelCapacityError(ValueError):
    def __init__(self, message, layout):
        super().__init__(message)
        self.layout = layout


class Arena:
    """Best-fit free blocks; source buffers remain live until their op completes."""
    def __init__(self, start, limit):
        self.cursor, self.limit = int(start), int(limit)
        self.free = []

    def allocate(self, size):
        size = aligned(size)
        candidates = [(length, index) for index, (_start, length) in enumerate(self.free) if length >= size]
        if candidates:
            _length, index = min(candidates)
            start, length = self.free.pop(index)
            if length > size:
                self.free.append((start + size, length - size))
            return start
        start = aligned(self.cursor)
        self.cursor = start + size
        if self.cursor > self.limit:
            raise _TensorCapacityError("BigCodec intermediate tensors exceed the FPGA tensor arena")
        return start

    def release(self, start, size):
        self.free.append((start, aligned(size)))
        merged = []
        for address, length in sorted(self.free):
            if merged and merged[-1][0] + merged[-1][1] == address:
                merged[-1] = (merged[-1][0], merged[-1][1] + length)
            else:
                merged.append((address, length))
        self.free = merged


def plan_memory(graph: Graph, *, layout: MemoryLayout | None = None) -> Graph:
    """Preserve legacy addresses when possible; use the larger RK arenas if needed."""
    if layout is not None:
        if layout not in LAYOUTS:
            raise ValueError("Unsupported BigCodec memory layout")
        return _plan_memory(graph, layout)
    for candidate in (LEGACY_LAYOUT, EXTENDED_LAYOUT):
        try:
            return _plan_memory(graph, candidate)
        except _TensorCapacityError:
            if candidate is EXTENDED_LAYOUT:
                raise
    raise AssertionError("No BigCodec layouts configured")


def _plan_memory(graph: Graph, layout: MemoryLayout) -> Graph:
    graph.layout = layout
    graph.output_address = layout.tensor_base
    graph.tensors["input"].address = layout.input_base
    if graph.tensors["input"].size_bytes > layout.input_limit - layout.input_base:
        raise ValueError("BigCodec packed input exceeds the FPGA input arena")
    graph.tokens_offset = packed_bytes((graph.compiled_samples, 1))
    graph.tokens_address = graph.output_address + graph.tokens_offset
    graph.output_bytes = graph.tokens_offset + packed_bytes((graph.code_frames, 2))
    graph.tensors[graph.output].address = graph.output_address
    arena = Arena(aligned(graph.output_address + graph.output_bytes), layout.tensor_limit)
    uses = Counter(name for operation in graph.operations for name in operation.inputs)
    scratch = 0
    for index, operation in enumerate(graph.operations):
        output = graph.tensors[operation.output]
        output.first = index
        if operation.output != graph.output:
            output.address = arena.allocate(output.size_bytes)
        for name in operation.inputs:
            source = graph.tensors[name]
            source.last = index
            if (source.address < output.address + output.size_bytes
                    and output.address < source.address + source.size_bytes):
                raise AssertionError(f"{operation.name}: input/output alias")
            uses[name] -= 1
            if uses[name] == 0 and name not in ("input", graph.output):
                arena.release(source.address, source.size_bytes)
        shape = graph.tensors[operation.inputs[0]].shape
        module = operation.module
        if operation.op == "activation":
            operation.scratch_bytes = activation_scratch_bytes(shape, **_filter_numerics(graph, operation))
        elif operation.op == "lstm":
            numerics = _lstm_numerics(graph, operation)
            operation.scratch_bytes = lstm_scratch_bytes(shape,
                compensated_cell=numerics["compensated_cell"],
                preserve_cell_residual=numerics["preserve_cell_residual"])
        elif operation.op == "quantizer":
            operation.scratch_bytes = quantizer_scratch_bytes(shape[0])
        elif operation.op == "tanh":
            operation.scratch_bytes = (0 if _waveform_tanh(graph, operation)
                else tanh_scratch_bytes(output.size_bytes // 2))
        elif operation.op == "conv1d":
            if graph.conv_precision == "bf16":
                operation.scratch_bytes = _bf16_convolutions().conv_scratch_bytes(shape, module.weight.shape,
                    padding=module.padding[0], stride=module.stride[0], dilation=module.dilation[0])
            else:
                operation.weight_reuse_pixels = convolution_reuse_pixels(module)
                operation.scratch_bytes = aligned(conv1d_padding_bytes(shape, padding=module.padding[0])) + CONV_SCATTER_BYTES
        elif operation.op == "conv_transpose1d":
            if graph.conv_precision == "bf16":
                operation.scratch_bytes = _bf16_convolutions().conv_scratch_bytes(shape, module.weight.shape,
                    padding=module.padding[0], stride=module.stride[0], transpose=True,
                    output_padding=module.output_padding[0])
            else:
                operation.weight_reuse_pixels = convolution_reuse_pixels(module)
                operation.scratch_bytes = aligned(conv_transpose1d_padding_bytes(shape, module.weight.shape,
                    stride=module.stride[0], padding=module.padding[0], output_padding=module.output_padding[0])) + CONV_SCATTER_BYTES
        scratch = max(scratch, operation.scratch_bytes)
    graph.tensors[graph.output].last = len(graph.operations)
    graph.scratch_address = aligned(arena.cursor)
    graph.scratch_bytes = aligned(scratch)
    graph.tensor_end = graph.scratch_address + graph.scratch_bytes
    if graph.tensor_end > layout.tensor_limit:
        raise _TensorCapacityError(f"BigCodec tensors/workspace need {graph.tensor_end - layout.tensor_base} bytes; arena holds {layout.tensor_limit - layout.tensor_base}")
    return graph


def _prepare_operation(graph, image, identity_address, zero_address, operation):
    source = graph.tensors[operation.inputs[0]]
    output = graph.tensors[operation.output]
    module = operation.module
    if operation.op in ("conv1d", "conv_transpose1d") and graph.conv_precision == "bf16":
        kwargs = dict(input_shape=source.shape, input_address=source.address,
            output_address=output.address, scratch_address=graph.scratch_address,
            image=image, stride=module.stride[0], padding=module.padding[0])
        if operation.op == "conv1d":
            operation.plan = _bf16_convolutions().prepare_conv1d(operation.name,
                module.weight, module.bias, dilation=module.dilation[0], **kwargs)
        else:
            operation.plan = _bf16_convolutions().prepare_conv_transpose1d(operation.name,
                module.weight, module.bias, output_padding=module.output_padding[0], **kwargs)
    elif operation.op == "conv1d":
        operation.plan = prepare_conv1d(operation.name, module.weight, module.bias,
            input_shape=source.shape, input_address=source.address, output_address=output.address,
            padded_input_address=graph.scratch_address, image=image, stride=module.stride[0],
            padding=module.padding[0], dilation=module.dilation[0], weight_reuse_pixels=operation.weight_reuse_pixels)
    elif operation.op == "conv_transpose1d":
        operation.plan = prepare_conv_transpose1d(operation.name, module.weight, module.bias,
            input_shape=source.shape, input_address=source.address, output_address=output.address,
            padded_input_address=graph.scratch_address, image=image, stride=module.stride[0],
            padding=module.padding[0], output_padding=module.output_padding[0], weight_reuse_pixels=operation.weight_reuse_pixels)
    elif operation.op == "activation":
        operation.plan = prepare_activation(module, image, input_shape=source.shape,
            input_address=source.address, output_address=output.address,
            scratch_address=graph.scratch_address, identity_address=identity_address,
            workspace_bytes=graph.scratch_bytes, **_filter_numerics(graph, operation))
    elif operation.op == "lstm":
        recurrent_precision = ('if8' if graph.lstm_precision == 'if8'
            or (graph.lstm_precision == 'encoder-if8' and operation.name.startswith('encoder.'))
            else 'bf16')
        operation.plan = prepare_lstm(module.lstm, image, input_shape=source.shape,
            input_address=source.address, output_address=output.address,
            scratch_address=graph.scratch_address, identity_address=identity_address,
            zero_address=zero_address, skip=module.skip,
            recurrent_precision=recurrent_precision, **_lstm_numerics(graph, operation))
    elif operation.op == "quantizer":
        operation.plan = prepare_quantizer(module, image, frames=source.shape[0],
            source_address=source.address, destination_address=output.address,
            scratch_address=graph.scratch_address, token_address=graph.tokens_address,
            center_scores=graph.center_quantizer_scores, compensated_codebook=graph.compensated_codebook)
    elif _waveform_tanh(graph, operation):
        mask = torch.zeros(TANH_CHUNK_ELEMENTS, dtype=torch.bfloat16)
        mask[::WAVEFORM_ROW_LANES] = 1
        operation.plan = {'mask_address': image.allocate(mask, alignment=128)}
    elif operation.op not in ("add", "tanh"):
        raise AssertionError(operation.op)
    if operation.op in ("conv1d", "conv_transpose1d") and graph.conv_precision == "if8":
        operation.scatter_address = graph.scratch_address + aligned(operation.plan["padding_bytes"])
        if operation.plan["scatter_scratch_bytes"] > CONV_SCATTER_BYTES:
            raise ValueError(f"{operation.name}: convolution scatter exceeds reserved workspace")


def prepare_operations(graph: Graph, image, identity_address, zero_address):
    for operation in graph.operations:
        try:
            _prepare_operation(graph, image, identity_address, zero_address, operation)
        except (ValueError, AssertionError, RuntimeError) as error:
            shape = graph.tensors[operation.inputs[0]].shape
            raise RuntimeError(f"{operation.name} ({operation.op}, input {shape}): {error}") from error


def emit_operation(engine, graph, operation, identity_address, zero_address):
    source = graph.tensors[operation.inputs[0]]
    output = graph.tensors[operation.output]
    if operation.op in ("conv1d", "conv_transpose1d"):
        # OC32 fragments need explicit zeros in any unwritten pad64 lanes.
        zero(engine, output.address, output.size_bytes, zero_address)
        if graph.conv_precision == "bf16":
            _bf16_convolutions().emit_conv(engine, operation.plan, zero_address=zero_address)
        else:
            emit = emit_conv1d if operation.op == "conv1d" else emit_conv_transpose1d
            emit(engine, operation.plan, zero_address=zero_address, scratch_address=operation.scatter_address)
    elif operation.op == "activation":
        emit_activation(engine, operation.plan)
    elif operation.op == "lstm":
        emit_lstm(engine, operation.plan)
    elif operation.op == "quantizer":
        emit_quantizer(engine, operation.plan)
    elif operation.op == "add":
        other = graph.tensors[operation.inputs[1]]
        elementwise(engine, udc.UE_MODE.ELTWISE_ADD, source.address,
                    other.address, output.address, output.size_bytes // 2)
    elif operation.op == "tanh":
        if _waveform_tanh(graph, operation):
            emit_waveform_tanh(engine, source.address, output.address, source.shape[0],
                identity_address, zero_address, operation.plan['mask_address'])
        else:
            tanh_identity(engine, source.address, output.address, output.size_bytes // 2, identity_address,
                           scratch_address=graph.scratch_address)
    else:
        raise AssertionError(operation.op)


def compile_models(encoder, decoder, *, samples: int, conv_precision="if8", lstm_precision="bf16",
                   lstm_cell_precision="bf16", lstm_tanh_precision="bf16", lstm_math_scope="both",
                   lstm_fused_gates=False, center_quantizer_scores=False, compensated_codebook=False,
                   filter_accumulation="serial", filter_math_scope="both", filter_stage="both",
                   memory_layout=None) -> dict:
    if memory_layout is not None and memory_layout not in LAYOUTS:
        raise ValueError("Unsupported BigCodec memory layout")
    layout = memory_layout
    while True:
        try:
            return _compile_models(encoder, decoder, samples=samples,
                conv_precision=conv_precision, lstm_precision=lstm_precision,
                lstm_cell_precision=lstm_cell_precision, lstm_tanh_precision=lstm_tanh_precision,
                lstm_math_scope=lstm_math_scope, lstm_fused_gates=lstm_fused_gates,
                center_quantizer_scores=center_quantizer_scores,
                compensated_codebook=compensated_codebook,
                filter_accumulation=filter_accumulation, filter_math_scope=filter_math_scope,
                filter_stage=filter_stage,
                memory_layout=layout)
        except _ModelCapacityError as error:
            if memory_layout is not None:
                raise
            if layout is not None and error.layout != layout:
                raise
            if error.layout == LEGACY_LAYOUT:
                layout = EXTENDED_LAYOUT
            elif error.layout == EXTENDED_LAYOUT:
                layout = LARGE_PROGRAM_LAYOUT
            else:
                raise
        # Replan every address before recapturing into the next accepted arena.


def _compile_models(encoder, decoder, *, samples: int, conv_precision="if8",
                    lstm_precision="bf16", lstm_cell_precision="bf16", lstm_tanh_precision="bf16",
                    lstm_math_scope="both", lstm_fused_gates=False, center_quantizer_scores=False,
                    compensated_codebook=False,
                    filter_accumulation="serial", filter_math_scope="both", filter_stage="both",
                    memory_layout=None) -> dict:
    if encoder.training or decoder.training:
        raise ValueError("Compile eval models loaded with remove_weight_norm=True")
    if any(name.endswith(("weight_g", "weight_v"))
           for model in (encoder, decoder) for name, _ in model.named_parameters()):
        raise ValueError("Remove all weight normalization before reading compile-time weights")
    compiled_samples = compiled_sample_count(samples)
    graph = plan_memory(build_graph(encoder, decoder, compiled_samples,
                                   conv_precision=conv_precision, lstm_precision=lstm_precision,
                                   lstm_cell_precision=lstm_cell_precision,
                                   lstm_tanh_precision=lstm_tanh_precision, lstm_math_scope=lstm_math_scope,
                                   lstm_fused_gates=lstm_fused_gates,
                                   center_quantizer_scores=center_quantizer_scores,
                                   compensated_codebook=compensated_codebook,
                                   filter_accumulation=filter_accumulation, filter_math_scope=filter_math_scope,
                                   filter_stage=filter_stage),
                        layout=memory_layout)
    layout = graph.layout
    image = shared._ImageBuilder(layout.model_base, layout.model_limit)
    zero_address = image.allocate(torch.zeros(udc.URAM_NEAR_FULL_SIZE // 2, dtype=torch.bfloat16), alignment=128)
    identity_address = image.allocate(torch.eye(64, dtype=torch.bfloat16), alignment=128)
    previous_axi_width = udc.UE_AXI_DATA_WIDTH_BITS
    previous_capture_limit = udc.MAX_DECODER_INSTRUCTIONS
    try:
        udc.UE_AXI_DATA_WIDTH_BITS = 256
        # The shared emitter's 768 MiB default predates these long programs.
        # Keep an offline upper bound for the visible DRAM span; the exact
        # model-arena check below still rejects overflow after each operation.
        udc.MAX_DECODER_INSTRUCTIONS = max(previous_capture_limit,
            (layout.tensor_limit - layout.input_base) // udc.INSTRUCTION_SIZE_BYTES)
        prepare_operations(graph, image, identity_address, zero_address)
        parameter_bytes = len(image.data)
        program_address = image.align(128)
        engine = shared._WholeGraphEngine(program_address)
        engine.start_capture()
        operations = []
        for operation in graph.operations:
            start = engine.capture_count
            emit_operation(engine, graph, operation, identity_address, zero_address)
            stop = engine.capture_count
            if stop <= start:
                raise RuntimeError(f"{operation.name}: no instructions were emitted")
            operations.append({"name": operation.name, "op": operation.op,
                "start": start, "stop": stop, "inputs": list(operation.inputs),
                "output": operation.output, "weight_reuse_pixels": operation.weight_reuse_pixels})
            if program_address + aligned((stop + 1) * udc.INSTRUCTION_SIZE_BYTES, 64) > layout.model_limit:
                counts = Counter()
                for entry in operations:
                    counts[entry["op"]] += entry["stop"] - entry["start"]
                raise _ModelCapacityError(f"BigCodec model/program image exceeds {layout.model_limit - layout.model_base} bytes at {operation.name}; "
                    f"parameters={parameter_bytes}, captured_instructions={stop}, "
                    f"instructions_by_type={dict(counts)}", layout)
        halt_index = engine.capture_count
        engine.generate_instruction_halt()
        engine.stop_capture()
        issues = udc.check_isa_jumps(engine.capture_buffer, program_address, name="BigCodec")
        if issues:
            raise RuntimeError("Invalid BigCodec instruction jumps:\n" + "\n".join(issues))
        program = b"".join(instruction.get_bytes() for instruction in engine.capture_buffer)
        kinds = [(instruction.words[0] >> 8) & 15 for instruction in engine.capture_buffer]
        if (kinds.count(udc.INSTRUCTION_HALT) != 1 or udc.INSTRUCTION_SWI in kinds
                or any(kind != udc.INSTRUCTION_NOP for kind in kinds[halt_index + 1:])):
            raise RuntimeError("BigCodec program must have one terminal HALT and no SWI")
        if not program or len(program) % 64:
            raise RuntimeError("BigCodec program is not aligned for instruction fetch")
        image.write(program_address, program)
    finally:
        udc.UE_AXI_DATA_WIDTH_BITS = previous_axi_width
        udc.MAX_DECODER_INSTRUCTIONS = previous_capture_limit
    model_image = torch.frombuffer(image.data, dtype=torch.uint8).clone()
    hardware = {
        "memory_layout": layout.name,
        "model_base": layout.model_base, "model_limit": layout.model_limit,
        "model_image": model_image,
        "model_sha256": hashlib.sha256(image.data).hexdigest(),
        "program_address": program_address, "program_offset": program_address - layout.model_base,
        "program_size": len(program), "program_sha256": hashlib.sha256(program).hexdigest(),
        "parameter_bytes": parameter_bytes, "instructions": len(engine.capture_buffer),
        "halt_index": halt_index, "axi_data_width_bits": 256,
        "compiled_samples": compiled_samples, "code_frames": graph.code_frames,
        "input_address": layout.input_base, "input_bytes": graph.tensors["input"].size_bytes,
        "output_address": graph.output_address, "output_bytes": graph.output_bytes,
        "tokens_address": graph.tokens_address, "tokens_offset": graph.tokens_offset,
        "tensor_base": layout.tensor_base, "tensor_limit": layout.tensor_limit,
        "tensor_end": graph.tensor_end, "scratch_address": graph.scratch_address,
        "scratch_bytes": graph.scratch_bytes, "identity_address": identity_address,
        "zero_address": zero_address, "operations": operations,
        "tensors": {name: {"shape": list(tensor.shape), "padded_channels": channels(tensor.shape[1]),
            "address": tensor.address, "bytes": tensor.size_bytes,
            "first": tensor.first, "last": tensor.last} for name, tensor in graph.tensors.items()},
    }
    return {"format": FORMAT, "model": "bigcodec", "checkpoint_sha256": CHECKPOINT_SHA256,
        "sample_rate": SAMPLE_RATE, "hop_length": HOP_LENGTH, "native_samples": samples,
        "full_utterance": True, "all_neural_operations_on_device": True,
        "precision": {"convolutions": "IF8-INT" if conv_precision == "if8" else "BF16",
            "activations": "BF16", "lstm_input_weights": "BF16",
            "lstm_cell": lstm_cell_precision,
            "lstm_tanh": lstm_tanh_precision, "lstm_math_scope": lstm_math_scope,
            "lstm_fused_gates": lstm_fused_gates,
            "center_quantizer_scores": center_quantizer_scores,
            "compensated_codebook": compensated_codebook,
            "filter_accumulation": filter_accumulation, "filter_math_scope": filter_math_scope,
            "filter_stage": filter_stage,
            "lstm_recurrent_weights": {
                "encoder": "BF16" if lstm_precision == "bf16" else "IF8-INT",
                "decoder": "IF8-INT" if lstm_precision == "if8" else "BF16"}},
        "convolution_tiling": ({"target_weight_stream_bytes_per_64_outputs": CONV_STREAM_BUDGET_PER_64_OUTPUTS,
            "maximum_reuse_pixels": CONV_MAX_REUSE_PIXELS,
            "minimum_one_pixel_even_when_weights_exceed_target": True} if conv_precision == "if8"
            else {"method": "SRAM overlapping windows with dilation residue tiling and BF16 matrix multiplication",
                  "maximum_output_rows_per_tile": 1024}),
        "execution_optimizations": {
            "activation": "SRAM FIR and Snake tiles; native wide MAXPOOL clamp",
            "lstm": {
                "math_scope": lstm_math_scope,
                "cell_arithmetic": lstm_cell_precision,
                "tanh_arithmetic": lstm_tanh_precision,
                "fused_projection_bias_sigmoid": lstm_fused_gates,
            },
            "quantizer": "SRAM comparison masks and first seven tournament rounds",
            "waveform_tanh": "SRAM Pade evaluation with full 128-byte sample rows and explicit zero padding",
        },
        "approximation": {"snake_argument_clamp": SNAKE_ARGUMENT_LIMIT,
            "snake_sine_squared_polynomial_degree": 10,
            "snake_storage": "BF16",
            "alias_free_filters": (
                "Original taps split into BF16 high/residual lanes; native BF19/BF20 dot reduction, one BF16 output store"
                if filter_accumulation == "matrix" else
                "Symmetric BF16, exact unit DC, nearest L2 within +/-2 ULP"),
            "filter_math_scope": filter_math_scope, "filter_stage": filter_stage,
            "remaining_filter_stages": "Legacy serial BF16 FIR",
            "broadcast_scalars": "BF16 round-to-nearest-even before device encoding",
            "tanh": "BF16 odd Pade [7/6] with native reciprocal",
            "lstm_tanh": ("BF16 high/low Pade coefficients, products and sums with reciprocal refinement"
                if lstm_tanh_precision == "compensated" else "BF16 Pade"),
            "tanh_argument_clamp": TANH_ARGUMENT_LIMIT, "tanh_output_clamp": [-1, 1],
            "quantizer": "BF16 nearest normalized-codebook tournament",
            "quantizer_score_offset_before_writeback": -1 if center_quantizer_scores else 0,
            "quantizer_codebook_residual_lanes": 8 if compensated_codebook else 0,
            "accuracy_status": "requires comparison with the official CPU model"},
        "hardware": hardware}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    length = parser.add_mutually_exclusive_group()
    length.add_argument("--samples", type=int, help="Native 16 kHz input samples (default: 3200)")
    length.add_argument("--input", type=Path, help="Infer native sample count from a WAV")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--conv-precision", choices=("if8", "bf16"), default="if8")
    parser.add_argument("--lstm-precision", choices=("bf16", "encoder-if8", "if8"), default="bf16",
                        help="Recurrent weight precision; input projections and gate/state math stay BF16")
    parser.add_argument("--lstm-cell-precision", choices=("bf16", "products", "compensated"), default="bf16",
                        help="Cell update arithmetic; products corrects product rounding, compensated also carries a cell residual")
    parser.add_argument("--lstm-tanh-precision", choices=("bf16", "compensated"), default="bf16",
                        help="LSTM tanh arithmetic; compensated preserves intermediate residuals")
    parser.add_argument("--lstm-math-scope", choices=("encoder", "decoder", "both"), default="both",
                        help="LSTM stacks that use the selected cell/tanh arithmetic")
    parser.add_argument("--lstm-fused-gates", action=argparse.BooleanOptionalAction, default=False,
                        help="Fuse recurrent dot, input projection bias and sigmoid before BF16 writeback")
    parser.add_argument("--center-quantizer-scores", action=argparse.BooleanOptionalAction, default=False,
                        help="Subtract one inside the codebook dot product to preserve close score differences")
    parser.add_argument("--compensated-codebook", action=argparse.BooleanOptionalAction, default=False,
                        help="Use spare dot-product lanes for BF16 codebook residuals")
    parser.add_argument("--filter-accumulation", choices=("serial", "sorted", "matrix"), default="serial",
                        help="FIR taps: legacy order, ascending magnitude, or original high/residual matrix dot")
    parser.add_argument("--filter-math-scope", choices=("encoder", "decoder", "both"), default="both",
                        help="Select the stack that uses the requested FIR accumulation")
    parser.add_argument("--filter-stage", choices=("up", "down", "both"), default="both",
                        help="Select resampling stages for matrix FIR; remaining stages use legacy FIR")
    parser.add_argument("--memory-layout", choices=("auto", *(layout.name for layout in LAYOUTS)),
                        default="auto", help="Auto replans larger programs; an explicit arena avoids retrying compilation")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cpu-core", type=int)
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        parser.error(f"output exists: {args.output}; pass --force to replace it")
    inputs = [p for p in (args.checkpoint, args.input) if p is not None]
    if (args.output.resolve() in {p.resolve() for p in inputs}
            or any(args.output.exists() and p.exists() and args.output.samefile(p) for p in inputs)):
        parser.error("output must differ from input and checkpoint")
    if args.cpu_core is not None:
        if args.cpu_core not in os.sched_getaffinity(0):
            parser.error(f"CPU core {args.cpu_core} is unavailable")
        os.sched_setaffinity(0, {args.cpu_core})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    samples = len(read_audio(args.input)[0]) if args.input else (args.samples if args.samples is not None else 3200)
    try:
        compiled_sample_count(samples)
    except ValueError as error:
        parser.error(str(error))
    started = time.perf_counter()
    encoder, decoder = load_models(args.checkpoint, remove_weight_norm=True)
    # Shared planners print per-tile diagnostics; retain a quiet compile command.
    with contextlib.redirect_stdout(io.StringIO()):
        payload = compile_models(encoder, decoder, samples=samples,
                                 conv_precision=args.conv_precision, lstm_precision=args.lstm_precision,
                                 lstm_cell_precision=args.lstm_cell_precision,
                                 lstm_tanh_precision=args.lstm_tanh_precision,
                                 lstm_math_scope=args.lstm_math_scope, lstm_fused_gates=args.lstm_fused_gates,
                                 center_quantizer_scores=args.center_quantizer_scores,
                                 compensated_codebook=args.compensated_codebook,
                                 filter_accumulation=args.filter_accumulation, filter_math_scope=args.filter_math_scope,
                                 filter_stage=args.filter_stage,
                                 memory_layout=next((layout for layout in LAYOUTS if layout.name == args.memory_layout), None))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix="bigcodec-", suffix=".tmp", dir=args.output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    hardware = payload["hardware"]
    report = {"bin": str(args.output), "bin_sha256": sha256_file(args.output),
        "convolution_precision": args.conv_precision,
        "lstm_recurrent_precision": args.lstm_precision,
        "lstm_cell_precision": args.lstm_cell_precision,
        "lstm_tanh_precision": args.lstm_tanh_precision, "lstm_math_scope": args.lstm_math_scope,
        "lstm_fused_gates": args.lstm_fused_gates,
        "center_quantizer_scores": args.center_quantizer_scores,
        "compensated_codebook": args.compensated_codebook,
        "filter_accumulation": args.filter_accumulation, "filter_math_scope": args.filter_math_scope,
        "filter_stage": args.filter_stage,
        "memory_layout": payload["hardware"]["memory_layout"],
        "bin_bytes": args.output.stat().st_size, "native_samples": samples,
        "compiled_samples": hardware["compiled_samples"], "code_frames": hardware["code_frames"],
        "operations": len(hardware["operations"]), "instructions": hardware["instructions"],
        "program_bytes": hardware["program_size"], "parameter_bytes": hardware["parameter_bytes"],
        "resident_bytes": hardware["model_image"].numel(), "scratch_bytes": hardware["scratch_bytes"],
        "compile_s": time.perf_counter() - started}
    print("TEST_RESULT " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
