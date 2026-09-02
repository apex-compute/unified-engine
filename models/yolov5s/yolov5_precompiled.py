"""Compile and run a complete YOLOv5 graph as one FPGA program.

The hardware path has four explicit transport phases: upload one immutable
model/program image when the backend is created, upload one packed input,
start one resident program, and read one contiguous bundle of detection heads.
Every intermediate tensor stays in accelerator DRAM and every graph primitive
is emitted into the same HALT-terminated instruction stream.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
import hashlib
import json
import math
import struct
import time
from typing import Sequence

import torch

import user_dma_core


PRECOMPILED_ABI = "andromeda-yolov5-whole-graph-v1"
GEOMETRY_ABI = "conv-config-inst-v1"
PRECOMPILED_AXI_DATA_WIDTH_BITS = 256

# In strided-write mode the checked-in RTL stores the per-stride AXI beat
# count in eight bits.  A zero value represents the legal 256-beat maximum;
# larger chunks wrap and no longer agree with the byte count.  Non-strided
# writes are split into legal AXI bursts by the RTL and do not need this cap.
_MAX_STRIDED_WRITE_CHUNK_BYTES = (
    256 * (PRECOMPILED_AXI_DATA_WIDTH_BITS // 8))

# Image-dependent input, immutable deployment image, and persistent graph
# tensors occupy disjoint fixed-address arenas.  The deployment image includes
# both packed parameters and the program, so backend construction needs one
# H2C transfer rather than separate parameter/program writes.
INPUT_BASE = user_dma_core.DRAM_START_ADDR                # 0x80000000
INPUT_LIMIT = 0x90000000
MODEL_BASE = INPUT_LIMIT                                  # 0x90000000
MODEL_LIMIT = user_dma_core.DRAM_ACTIVATION_ADDR          # 0xB0000000
TENSOR_BASE = user_dma_core.DRAM_ACTIVATION_ADDR          # 0xB0000000
TENSOR_LIMIT = user_dma_core.DRAM_INSTRUCTION_ADDR        # 0xD0000000

# Compatibility aliases for callers which only inspect fixed arena bounds.
DYNAMIC_PARAMS_BASE = INPUT_BASE
STATIC_PARAMS_BASE = MODEL_BASE
STATIC_PARAMS_LIMIT = MODEL_LIMIT
PROGRAM_BASE = MODEL_BASE
PROGRAM_LIMIT = MODEL_LIMIT

_LINE_BYTES = user_dma_core.UE_VECTOR_SIZE * 2
_ACT_URAM_LINES = 0x300
_ACT_TEMPLATE_BYTES = _ACT_URAM_LINES * _LINE_BYTES
_WB_SRAM_ADDRESS = _ACT_URAM_LINES << 7


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _tensor_bytes(value: torch.Tensor, size: int | None = None) -> bytes:
    value = value.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        raw = value.view(torch.uint16).numpy().tobytes()
    else:
        raw = value.numpy().tobytes()
    return raw if size is None else raw[:int(size)]


def _bytes_tensor(value: bytes | bytearray) -> torch.Tensor:
    if not value:
        return torch.empty(0, dtype=torch.uint8)
    return torch.frombuffer(bytearray(value), dtype=torch.uint8).clone()


def _image_sha256(value: torch.Tensor | bytes | bytearray) -> str:
    raw = _tensor_bytes(value) if isinstance(value, torch.Tensor) else bytes(value)
    return hashlib.sha256(raw).hexdigest()


def _unpack_codes(packed: torch.Tensor, shape: Sequence[int],
                  precision: str) -> torch.Tensor:
    count = math.prod(int(value) for value in shape)
    if (not isinstance(packed, torch.Tensor) or packed.dtype != torch.uint8
            or packed.dim() != 1):
        raise RuntimeError("quantized code payload must be a flat uint8 tensor")
    if precision == "if8":
        if packed.numel() != count:
            raise RuntimeError("IF8 code payload size does not match its shape")
        return packed.view(tuple(int(value) for value in shape)).contiguous()
    if precision != "if4":
        raise RuntimeError(f"unsupported convolution precision {precision!r}")
    if packed.numel() != (count + 1) // 2:
        raise RuntimeError("IF4 code payload size does not match its shape")
    raw = torch.empty(packed.numel() * 2, dtype=torch.uint8)
    raw[0::2] = packed & 0x0F
    raw[1::2] = packed >> 4
    return raw[:count].view(tuple(int(value) for value in shape)).contiguous()


def _scan_queue_configs(
        image, offset: int, size: int) -> list[tuple[int, int, int, int]]:
    """Return and validate queued convolution/max-pool CONFIG payloads."""
    raw = _tensor_bytes(image) if isinstance(image, torch.Tensor) else bytes(image)
    offset, size = int(offset), int(size)
    if offset < 0 or size <= 0 or size % 32 or offset + size > len(raw):
        raise RuntimeError("invalid instruction range while scanning queued CONFIG")
    result = []
    for position in range(offset, offset + size, 32):
        words = struct.unpack_from("<8I", raw, position)
        instruction_type = (words[0] >> 8) & 0xF
        if instruction_type != user_dma_core.INSTRUCTION_CONFIG:
            continue
        subtype = (words[0] >> 12) & 0xF
        if ((words[0] >> 16) != 0
                or subtype != user_dma_core.CONFIG_SUBTYPE_CONV
                or any(words[5:])):
            raise RuntimeError("queued CONFIG has non-zero reserved fields")
        try:
            geometry = user_dma_core.validate_conv2d_geometry_words(
                tuple(int(value) for value in words[1:5]))
        except ValueError as exc:
            raise RuntimeError(f"queued CONFIG geometry is invalid: {exc}") from exc
        result.append(geometry)
    return result


class _ImageBuilder:
    def __init__(self, base: int, limit: int):
        self.base = int(base)
        self.limit = int(limit)
        self.data = bytearray()
        self.spans: list[tuple[int, int]] = []

    @property
    def cursor(self) -> int:
        return self.base + len(self.data)

    def align(self, alignment: int = 64) -> int:
        address = _align_up(self.cursor, alignment)
        if address > self.cursor:
            self.data.extend(b"\0" * (address - self.cursor))
        return address

    def write(self, address: int, value) -> None:
        raw = _tensor_bytes(value) if isinstance(value, torch.Tensor) else bytes(value)
        address, end = int(address), int(address) + len(raw)
        if address < self.base or end > self.limit:
            raise RuntimeError(
                f"deployment write [{address:#x}, {end:#x}) is outside "
                f"[{self.base:#x}, {self.limit:#x})")
        if any(address < old_end and old_start < end
               for old_start, old_end in self.spans):
            raise RuntimeError("overlapping writes in the deployment image")
        offset = address - self.base
        if len(self.data) < offset:
            self.data.extend(b"\0" * (offset - len(self.data)))
        if len(self.data) < offset + len(raw):
            self.data.extend(b"\0" * (offset + len(raw) - len(self.data)))
        self.data[offset:offset + len(raw)] = raw
        self.spans.append((address, end))

    def allocate(self, value, *, alignment: int = 64) -> int:
        address = self.align(alignment)
        self.write(address, value)
        return address


@dataclass
class _PackedMap:
    logical_shape: tuple[int, int, int]
    physical_channels: int
    channel_map: tuple[int, ...]
    address: int = 0

    @property
    def channels(self) -> int:
        return self.logical_shape[0]

    @property
    def height(self) -> int:
        return self.logical_shape[1]

    @property
    def width(self) -> int:
        return self.logical_shape[2]

    @property
    def channel_tiles(self) -> int:
        return self.physical_channels // user_dma_core.UE_VECTOR_SIZE

    @property
    def size_bytes(self) -> int:
        return self.height * self.width * self.physical_channels * 2

    def manifest(self) -> dict:
        return {
            "address": int(self.address),
            "size_bytes": int(self.size_bytes),
            "logical_shape": list(self.logical_shape),
            "physical_channels": int(self.physical_channels),
            "channel_map": list(self.channel_map),
        }


class _WholeGraphEngine(user_dma_core.UnifiedEngine):
    """State-only descriptor emitter; construction never opens a device."""

    def __init__(self, program_address: int):
        self.device = "cpu"
        self.h2c_device = "offline"
        self.c2h_device = "offline"
        self.user_device = "offline"
        self._user_fd = None
        self.latency_count = 0
        self.capture_buffer = []
        self.capture_count = 0
        self.is_capture_on = False
        self._params_dram_base = MODEL_BASE
        self._program_dram_base = int(program_address)
        self._tensor_dram_base = TENSOR_BASE
        self._next_params_dram_addr = MODEL_BASE
        self._next_program_dram_addr = int(program_address)
        self._tensor_dram_addr = TENSOR_BASE
        self._isa_reg_counter = 1
        self._inst_ptr_counter = 1
        self._dram_addresses = {}
        self._inst_id = 0
        self._capture_loop_stack = []
        self._capture_conv_geometry = None
        self._clock_period_ns = 10.0
        self.conv_geometry_mode = user_dma_core.CONV_GEOMETRY_QUEUE_CONFIG
        self._base_addr = 0
        self.hw_version = 0

    def start_capture(self):
        self._inst_id = 0
        self.capture_count = 0
        self.capture_buffer = []
        self._capture_loop_stack = []
        self._capture_conv_geometry = None
        self.is_capture_on = True

    def stop_capture(self):
        self.is_capture_on = False

    def write_reg32(self, address: int, _value: int) -> None:
        raise RuntimeError(
            f"whole-graph compiler attempted a live CSR write at {address:#x}")

    def read_reg32(self, _address: int) -> int:
        return 0


def _layout_for_conv(shape: Sequence[int]) -> _PackedMap:
    logical = tuple(int(value) for value in shape)
    physical = _align_up(logical[0], user_dma_core.UE_VECTOR_SIZE)
    return _PackedMap(logical, physical, tuple(range(logical[0])))


def _build_memory_plan(payload: dict):
    input_shape = tuple(int(value) for value in payload["model"]["input_shape"])
    layouts: dict[str, _PackedMap] = {
        "input": _layout_for_conv(input_shape),
    }
    layouts["input"].address = INPUT_BASE
    if layouts["input"].size_bytes > INPUT_LIMIT - INPUT_BASE:
        raise RuntimeError("packed YOLO input exceeds its fixed DRAM arena")

    for operation in payload["operations"]:
        sources = [layouts[name] for name in operation["inputs"]]
        shape = tuple(int(value) for value in operation["output_shape"])
        kind = operation["op"]
        if kind == "conv":
            result = _layout_for_conv(shape)
        elif kind in ("maxpool", "upsample2x"):
            source = sources[0]
            result = _PackedMap(
                shape, source.physical_channels, tuple(source.channel_map))
        elif kind == "add":
            first, second = sources
            if (first.logical_shape != second.logical_shape
                    or first.physical_channels != second.physical_channels
                    or first.channel_map != second.channel_map):
                raise RuntimeError(
                    f"{operation['name']}: residual layouts are incompatible")
            result = _PackedMap(
                shape, first.physical_channels, tuple(first.channel_map))
        elif kind == "concat":
            if any(source.logical_shape[1:] != sources[0].logical_shape[1:]
                   for source in sources[1:]):
                raise RuntimeError(
                    f"{operation['name']}: concat spatial layouts differ")
            channel_map = []
            physical_offset = 0
            for source in sources:
                channel_map.extend(
                    physical_offset + value for value in source.channel_map)
                physical_offset += source.physical_channels
            result = _PackedMap(shape, physical_offset, tuple(channel_map))
        else:
            raise RuntimeError(f"unsupported YOLO graph primitive {kind!r}")
        if len(result.channel_map) != shape[0]:
            raise RuntimeError(
                f"{operation['name']}: logical/physical channel plan disagrees")
        if (result.physical_channels <= 0
                or result.physical_channels % user_dma_core.UE_VECTOR_SIZE):
            raise RuntimeError(f"{operation['name']}: invalid physical channel count")
        layouts[operation["output"]] = result

    heads = list(payload.get("head_outputs", ()))
    if not heads or any(name not in layouts for name in heads):
        raise RuntimeError("whole-graph artifact has invalid detection heads")
    head_set = set(heads)
    if len(head_set) != len(heads):
        raise RuntimeError("whole-graph artifact repeats a detection head")

    cursor = TENSOR_BASE
    outputs = [operation["output"] for operation in payload["operations"]]
    for name in [value for value in outputs if value not in head_set] + heads:
        layout = layouts[name]
        cursor = _align_up(cursor, _LINE_BYTES)
        layout.address = cursor
        cursor += layout.size_bytes

    head_bundle_address = layouts[heads[0]].address
    expected = head_bundle_address
    head_manifest = []
    for name in heads:
        layout = layouts[name]
        if layout.address != expected:
            raise RuntimeError("detection heads are not one contiguous result bundle")
        item = {"name": name, "offset": layout.address - head_bundle_address}
        item.update(layout.manifest())
        head_manifest.append(item)
        expected += layout.size_bytes
    head_bundle_bytes = expected - head_bundle_address

    scratch_bytes = 0
    for operation in payload["operations"]:
        if operation["op"] != "upsample2x":
            continue
        source = layouts[operation["inputs"][0]]
        scratch_bytes = max(
            scratch_bytes,
            2 * source.height * source.width * source.channel_tiles * _LINE_BYTES)
    scratch_address = _align_up(cursor, _LINE_BYTES)
    cursor = scratch_address + scratch_bytes
    if cursor > TENSOR_LIMIT:
        raise RuntimeError(
            f"whole YOLO graph needs {cursor - TENSOR_BASE} tensor bytes, "
            f"arena holds {TENSOR_LIMIT - TENSOR_BASE}")
    return (layouts, head_manifest, head_bundle_address, head_bundle_bytes,
            scratch_address, scratch_bytes)


def _expanded_channel_weights(
        codes: torch.Tensor, scales: torch.Tensor, source: _PackedMap):
    oc, logical_c, kh, kw = (int(value) for value in codes.shape)
    if logical_c != source.channels:
        raise RuntimeError("convolution weight/input channel count differs")
    physical_c = source.physical_channels
    expanded = torch.zeros(oc, physical_c, kh, kw, dtype=codes.dtype)
    for logical_channel, physical_channel in enumerate(source.channel_map):
        expanded[:, physical_channel, :, :] = codes[:, logical_channel, :, :]

    old_ct = (logical_c + user_dma_core.UE_VECTOR_SIZE - 1) \
        // user_dma_core.UE_VECTOR_SIZE
    new_ct = physical_c // user_dma_core.UE_VECTOR_SIZE
    expected = (oc, kh * kw * old_ct)
    if tuple(scales.shape) != expected:
        raise RuntimeError(
            f"channel convolution scales have {tuple(scales.shape)}, "
            f"expected {expected}")
    expanded_scales = torch.full(
        (oc, kh * kw * new_ct), -1.0, dtype=torch.bfloat16)
    block_sources: list[set[int]] = [set() for _ in range(new_ct)]
    for logical_channel, physical_channel in enumerate(source.channel_map):
        block_sources[physical_channel // user_dma_core.UE_VECTOR_SIZE].add(
            logical_channel // user_dma_core.UE_VECTOR_SIZE)
    for physical_block, logical_blocks in enumerate(block_sources):
        if not logical_blocks:
            continue
        if len(logical_blocks) != 1:
            raise RuntimeError(
                "concat layout mixes two independently scaled logical blocks "
                "inside one physical block")
        logical_block = next(iter(logical_blocks))
        for tap in range(kh * kw):
            expanded_scales[:, tap * new_ct + physical_block] = \
                scales[:, tap * old_ct + logical_block]
    return expanded, expanded_scales


def _prepare_conv_plan(operation: dict, weight: dict, source: _PackedMap,
                       destination: _PackedMap, image: _ImageBuilder) -> dict:
    precision = weight["precision"]
    data_type = user_dma_core.TYPE[precision.upper()]
    codes = _unpack_codes(
        weight["codes_packed"], weight["codes_shape"], precision)
    logical_oc, logical_c, kh, kw = (int(value) for value in codes.shape)
    if logical_oc != destination.channels or logical_c != source.channels:
        raise RuntimeError(f"{operation['name']}: convolution shape mismatch")

    use_gather = weight["layout"] == "gather"
    if use_gather and source.channel_map != tuple(range(source.channels)):
        raise RuntimeError(
            f"{operation['name']}: gather convolution requires contiguous channels")
    if use_gather:
        expanded_codes = codes
        expanded_scales = weight["block_scales"].to(torch.bfloat16)
        convolution_c = source.channels
        chunks = (kh * kw * convolution_c + user_dma_core.UE_VECTOR_SIZE - 1) \
            // user_dma_core.UE_VECTOR_SIZE
        if tuple(expanded_scales.shape) != (logical_oc, chunks):
            raise RuntimeError(f"{operation['name']}: invalid gather scale shape")
    else:
        expanded_codes, expanded_scales = _expanded_channel_weights(
            codes, weight["block_scales"], source)
        convolution_c = source.physical_channels
        chunks = 0

    physical_oc = destination.physical_channels
    padded_codes = torch.zeros(
        physical_oc, expanded_codes.shape[1], kh, kw,
        dtype=expanded_codes.dtype)
    padded_codes[:logical_oc] = expanded_codes
    padded_scales = torch.full(
        (physical_oc, expanded_scales.shape[1]), -1.0,
        dtype=torch.bfloat16)
    padded_scales[:logical_oc] = expanded_scales

    bias = weight["bias"].to(torch.bfloat16)
    if bias.numel() not in (0, logical_oc):
        raise RuntimeError(f"{operation['name']}: invalid convolution bias")
    padded_bias = None
    if bias.numel():
        padded_bias = torch.zeros(physical_oc, dtype=torch.bfloat16)
        padded_bias[:logical_oc] = bias

    out_h, out_w, oc_chunk, tiles = user_dma_core.plan_conv2d_layer_tiles(
        c_in=convolution_c, oc_count=physical_oc,
        in_h=source.height, in_w=source.width,
        kernel_h=kh, kernel_w=kw,
        stride_s=int(operation["stride"]), pad=int(operation["pad"]),
        dilation=int(operation["dilation"]), gather=use_gather,
        bias_enabled=padded_bias is not None,
        wb_uram_addr=_ACT_URAM_LINES)
    if (out_h, out_w) != destination.logical_shape[1:]:
        raise RuntimeError(f"{operation['name']}: convolution planner shape differs")
    if oc_chunk % user_dma_core.UE_VECTOR_SIZE:
        raise RuntimeError(
            f"{operation['name']}: OC chunk {oc_chunk} cannot be scattered "
            "into the persistent packed map without host repacking")
    groups = user_dma_core.conv2d_tile_geometry_groups(tiles)
    stream_group = max(groups, key=lambda group: group[2] * group[3])
    stream_h, stream_w = stream_group[2], stream_group[3]

    chunks_manifest = []
    for oc0 in range(0, physical_oc, oc_chunk):
        chunk_codes = padded_codes[oc0:oc0 + oc_chunk]
        chunk_scales = padded_scales[oc0:oc0 + oc_chunk]
        if use_gather:
            weight_stream = user_dma_core.conv2d_pack_weight_stream_gather(
                chunk_codes, stream_h, stream_w, data_type)
            scale_stream = user_dma_core.conv2d_pack_scale_stream_gather(
                chunk_scales, oc_chunk, chunks)
        else:
            taps = kh * kw * (convolution_c // user_dma_core.UE_VECTOR_SIZE)
            weight_stream = user_dma_core.conv2d_pack_weight_stream(
                chunk_codes, stream_h, stream_w, data_type)
            scale_stream = user_dma_core.conv2d_pack_scale_stream(
                chunk_scales, oc_chunk, taps, stream_h, stream_w)
        weight_address = image.allocate(weight_stream, alignment=64)
        scale_address = image.allocate(scale_stream, alignment=64)
        bias_address = None
        if padded_bias is not None:
            bias_stream = user_dma_core.conv2d_pack_bias_stream(
                padded_bias[oc0:oc0 + oc_chunk], stream_h, stream_w)
            bias_address = image.allocate(bias_stream, alignment=64)
        chunks_manifest.append({
            "oc0": oc0,
            "weight_address": weight_address,
            "scale_address": scale_address,
            "bias_address": bias_address,
        })
    return {
        "operation": operation,
        "source": source,
        "destination": destination,
        "data_type": data_type,
        "use_gather": use_gather,
        "convolution_c": convolution_c,
        "kernel_h": kh,
        "kernel_w": kw,
        "oc_chunk": oc_chunk,
        "tiles": tiles,
        "groups": groups,
        "chunks": chunks_manifest,
        "gather_chunks": chunks,
        "bias_enabled": padded_bias is not None,
    }


def _copy_contiguous_or_strided_read(engine: _WholeGraphEngine, *,
                                     source: int, sram: int,
                                     total: int, chunk: int, jump: int) -> None:
    if total == chunk or chunk == jump:
        engine.accelerator_memory_to_sram(
            source, sram, 0, memcpy_length_bytes=total)
    else:
        engine.accelerator_memory_to_sram(
            source, sram, 0, memcpy_length_bytes=total,
            stride_bytes_per_chunk=chunk, stride_jump_bytes=jump)


def _copy_contiguous_or_strided_write(engine: _WholeGraphEngine, *,
                                      sram: int, destination: int,
                                      total: int, chunk: int, jump: int) -> None:
    if total == chunk or chunk == jump:
        engine.sram_to_accelerator_memory(
            sram, destination, 0, memcpy_length_bytes=total)
    elif (chunk <= user_dma_core.UE_STRIDE_CHUNK_MAX_BYTES
          and jump <= user_dma_core.UE_STRIDE_JUMP_MAX_BYTES
          and chunk <= _MAX_STRIDED_WRITE_CHUNK_BYTES):
        engine.sram_to_accelerator_memory(
            sram, destination, 0, memcpy_length_bytes=total,
            stride_bytes_per_chunk=chunk, stride_jump_bytes=jump)
    else:
        if chunk <= 0 or total <= 0 or total % chunk:
            raise RuntimeError("strided write cannot be split into whole chunks")
        for index in range(total // chunk):
            engine.sram_to_accelerator_memory(
                sram + index * chunk, destination + index * jump, 0,
                memcpy_length_bytes=chunk)


def _stage_conv_window(engine: _WholeGraphEngine, source: _PackedMap,
                       tile, pad: int, zero_address: int) -> None:
    _oy, _ox, _th, _tw, y0, x0, win_h, win_w = (
        int(value) for value in tile)
    ct = source.channel_tiles
    act_bytes = win_h * win_w * ct * _LINE_BYTES
    padded_y0, padded_x0 = y0, x0
    padded_y1, padded_x1 = y0 + win_h, x0 + win_w
    valid_y0 = max(padded_y0, pad)
    valid_x0 = max(padded_x0, pad)
    valid_y1 = min(padded_y1, pad + source.height)
    valid_x1 = min(padded_x1, pad + source.width)
    valid_h = max(0, valid_y1 - valid_y0)
    valid_w = max(0, valid_x1 - valid_x0)
    if valid_h == win_h and valid_w == win_w:
        source_address = source.address + (
            ((valid_y0 - pad) * source.width + (valid_x0 - pad))
            * ct * _LINE_BYTES)
        row_bytes = win_w * ct * _LINE_BYTES
        _copy_contiguous_or_strided_read(
            engine, source=source_address, sram=0,
            total=win_h * row_bytes, chunk=row_bytes,
            jump=source.width * ct * _LINE_BYTES)
        return

    engine.accelerator_memory_to_sram(
        zero_address, 0, 0, memcpy_length_bytes=act_bytes)
    if not valid_h or not valid_w:
        return
    source_x = valid_x0 - pad
    destination_x = valid_x0 - padded_x0
    destination_y = valid_y0 - padded_y0
    row_bytes = valid_w * ct * _LINE_BYTES
    for row in range(valid_h):
        source_address = source.address + (
            ((valid_y0 - pad + row) * source.width + source_x)
            * ct * _LINE_BYTES)
        sram_address = (
            ((destination_y + row) * win_w + destination_x)
            * ct * _LINE_BYTES)
        engine.accelerator_memory_to_sram(
            source_address, sram_address, 0,
            memcpy_length_bytes=row_bytes)


def _scatter_conv_tile(engine: _WholeGraphEngine, destination: _PackedMap,
                       tile, oc0: int, oc_chunk: int) -> None:
    oy, ox, th, tw = (int(value) for value in tile[:4])
    source_row_bytes = tw * oc_chunk * 2
    destination_pixel_bytes = destination.physical_channels * 2
    if oc_chunk == destination.physical_channels:
        destination_address = destination.address + (
            (oy * destination.width + ox) * destination_pixel_bytes)
        _copy_contiguous_or_strided_write(
            engine, sram=_WB_SRAM_ADDRESS,
            destination=destination_address,
            total=th * source_row_bytes, chunk=source_row_bytes,
            jump=destination.width * destination_pixel_bytes)
        return
    for row in range(th):
        source_sram = _WB_SRAM_ADDRESS + row * source_row_bytes
        destination_address = destination.address + (
            ((oy + row) * destination.width + ox)
            * destination_pixel_bytes + oc0 * 2)
        _copy_contiguous_or_strided_write(
            engine, sram=source_sram, destination=destination_address,
            total=tw * oc_chunk * 2, chunk=oc_chunk * 2,
            jump=destination_pixel_bytes)


def _emit_conv(engine: _WholeGraphEngine, plan: dict,
               zero_address: int) -> None:
    operation = plan["operation"]
    source, destination = plan["source"], plan["destination"]
    stride, pad, dilation = (int(operation[key])
                             for key in ("stride", "pad", "dilation"))
    lalu_mode, lalu_a, lalu_b = user_dma_core._conv_fused_lalu(
        False, bool(operation["activate"]), False)
    ct = (plan["convolution_c"] + user_dma_core.UE_VECTOR_SIZE - 1) \
        // user_dma_core.UE_VECTOR_SIZE
    taps = plan["kernel_h"] * plan["kernel_w"] * ct
    for chunk in plan["chunks"]:
        oc0 = chunk["oc0"]
        for start, stop, th, tw, _win_h, win_w in plan["groups"]:
            if plan["use_gather"]:
                scale_count = plan["oc_chunk"] * plan["gather_chunks"]
            else:
                scale_count = th * tw * plan["oc_chunk"] * taps
            engine.accelerator_memory_to_scale_sram(
                chunk["scale_address"], scale_count)
            if chunk["bias_address"] is not None:
                engine.accelerator_memory_to_bias_sram(
                    chunk["bias_address"], th * tw * plan["oc_chunk"])
            for tile in plan["tiles"][start:stop]:
                _stage_conv_window(engine, source, tile, pad, zero_address)
                engine.start_queue_for_conv2d_operation(
                    act_sram_start_addr=0,
                    output_sram_wb_addr=_WB_SRAM_ADDRESS,
                    weights_dram_addr=chunk["weight_address"],
                    kernel_w=plan["kernel_w"], kernel_h=plan["kernel_h"],
                    ct=ct, oc_count=plan["oc_chunk"],
                    out_w=tw, out_h=th, w_pad=win_w,
                    stride_s=stride, data_type=plan["data_type"],
                    bias_enable=plan["bias_enabled"],
                    lalu_mode=lalu_mode, lalu_a=lalu_a, lalu_b=lalu_b,
                    dilation=dilation, gather=plan["use_gather"],
                    c_in=plan["convolution_c"])
                _scatter_conv_tile(
                    engine, destination, tile, oc0, plan["oc_chunk"])


def _maxpool_rows(operation: dict, source: _PackedMap):
    kernel = int(operation["kernel"])
    stride = int(operation["stride"])
    pad = int(operation["pad"])
    h_pad, w_pad = source.height + 2 * pad, source.width + 2 * pad
    out_h = (h_pad - kernel) // stride + 1
    out_w = (w_pad - kernel) // stride + 1

    def fits(rows: int) -> bool:
        window_h = (rows - 1) * stride + kernel
        return (kernel * kernel * rows * out_w <= 0xFFF
                and window_h * w_pad <= _ACT_URAM_LINES
                and _ACT_URAM_LINES + rows * out_w <= 4096
                and rows * out_w <= 0xFFFF)

    rows = 1
    while rows < out_h and fits(rows + 1):
        rows += 1
    if not fits(rows):
        raise RuntimeError(f"{operation['name']}: one max-pool row does not fit")
    starts = list(range(0, max(out_h - rows, 0) + 1, rows))
    if starts[-1] != out_h - rows:
        starts.append(out_h - rows)
    return kernel, stride, pad, w_pad, out_h, out_w, rows, starts


def _emit_maxpool(engine: _WholeGraphEngine, operation: dict,
                  source: _PackedMap, destination: _PackedMap,
                  neg_inf_address: int) -> None:
    (kernel, stride, pad, w_pad, out_h, out_w,
     rows, starts) = _maxpool_rows(operation, source)
    if (out_h, out_w) != destination.logical_shape[1:]:
        raise RuntimeError(f"{operation['name']}: max-pool shape differs")
    window_h = (rows - 1) * stride + kernel
    window_bytes = window_h * w_pad * _LINE_BYTES
    for channel_tile in range(source.channel_tiles):
        for output_y in starts:
            engine.accelerator_memory_to_sram(
                neg_inf_address, 0, 0, memcpy_length_bytes=window_bytes)
            padded_y0 = output_y * stride
            valid_y0 = max(padded_y0, pad)
            valid_y1 = min(padded_y0 + window_h, pad + source.height)
            for padded_y in range(valid_y0, valid_y1):
                source_y = padded_y - pad
                source_address = source.address + (
                    (source_y * source.width * source.channel_tiles
                     + channel_tile) * _LINE_BYTES)
                sram_address = (
                    ((padded_y - padded_y0) * w_pad + pad) * _LINE_BYTES)
                _copy_contiguous_or_strided_read(
                    engine, source=source_address, sram=sram_address,
                    total=source.width * _LINE_BYTES, chunk=_LINE_BYTES,
                    jump=source.channel_tiles * _LINE_BYTES)
            engine.start_queue_for_maxpool2d_operation(
                act_sram_start_addr=0,
                output_sram_wb_addr=_WB_SRAM_ADDRESS,
                kernel_w=kernel, kernel_h=kernel,
                out_w=out_w, out_h=rows, w_pad=w_pad, stride_s=stride)
            destination_address = destination.address + (
                (output_y * destination.width * destination.channel_tiles
                 + channel_tile) * _LINE_BYTES)
            _copy_contiguous_or_strided_write(
                engine, sram=_WB_SRAM_ADDRESS,
                destination=destination_address,
                total=rows * out_w * _LINE_BYTES, chunk=_LINE_BYTES,
                jump=destination.channel_tiles * _LINE_BYTES)


def _emit_upsample(engine: _WholeGraphEngine, source: _PackedMap,
                   destination: _PackedMap, scratch_address: int) -> None:
    out_h, out_w, _ct, passes = user_dma_core.plan_nn_upsample_2x(
        source.height, source.width, source.physical_channels)
    if (out_h, out_w) != destination.logical_shape[1:]:
        raise RuntimeError("upsample planner/output shape differs")
    source_of = {"in": source.address, "tmp": scratch_address}
    for item in passes:
        src = source_of[item["src"]]
        dst = scratch_address if item["kind"] == "v" else destination.address
        chunk, jump = int(item["chunk_bytes"]), int(item["jump_bytes"])
        if chunk > user_dma_core.URAM_NEAR_FULL_SIZE:
            raise RuntimeError("upsample row exceeds the URAM staging budget")
        round_bytes = (user_dma_core.URAM_NEAR_FULL_SIZE // chunk) * chunk
        done_bytes = 0
        chunks_done = 0
        while done_bytes < item["total_bytes"]:
            take = min(round_bytes, item["total_bytes"] - done_bytes)
            engine.accelerator_memory_to_sram(
                src + done_bytes, 0, 0, memcpy_length_bytes=take)
            base = dst + item["dst_off"] + chunks_done * jump
            _copy_contiguous_or_strided_write(
                engine, sram=0, destination=base, total=take,
                chunk=chunk, jump=jump)
            done_bytes += take
            chunks_done += take // chunk


def _emit_add(engine: _WholeGraphEngine, left: _PackedMap,
              right: _PackedMap, destination: _PackedMap) -> None:
    if (left.physical_channels != right.physical_channels
            or left.logical_shape != right.logical_shape
            or left.channel_map != right.channel_map):
        raise RuntimeError("residual add operands use different packed layouts")
    total_lines = left.size_bytes // _LINE_BYTES
    round_lines = min(user_dma_core.URAM_NEAR_FULL_SIZE // _LINE_BYTES, 4096)
    done = 0
    while done < total_lines:
        take = min(round_lines, total_lines - done)
        offset = done * _LINE_BYTES
        size = take * _LINE_BYTES
        engine.accelerator_memory_to_sram(
            left.address + offset, 0, 0, memcpy_length_bytes=size)
        engine.accelerator_memory_to_sram(
            right.address + offset, 0x80000, 0, memcpy_length_bytes=size)
        engine.eltwise_add_core(
            0, 0x80000, 0, take * user_dma_core.UE_VECTOR_SIZE)
        engine.sram_to_accelerator_memory(
            0, destination.address + offset, 0, memcpy_length_bytes=size)
        done += take


def _emit_concat(engine: _WholeGraphEngine, sources: Sequence[_PackedMap],
                 destination: _PackedMap) -> None:
    pixels = destination.height * destination.width
    destination_pixel_bytes = destination.channel_tiles * _LINE_BYTES
    destination_tile_offset = 0
    for source in sources:
        source_pixel_bytes = source.channel_tiles * _LINE_BYTES
        pixels_per_round = max(
            1, user_dma_core.URAM_NEAR_FULL_SIZE // source_pixel_bytes)
        done = 0
        while done < pixels:
            take = min(pixels_per_round, pixels - done)
            total = take * source_pixel_bytes
            engine.accelerator_memory_to_sram(
                source.address + done * source_pixel_bytes,
                0, 0, memcpy_length_bytes=total)
            destination_address = destination.address + (
                done * destination_pixel_bytes
                + destination_tile_offset * _LINE_BYTES)
            _copy_contiguous_or_strided_write(
                engine, sram=0, destination=destination_address,
                total=total, chunk=source_pixel_bytes,
                jump=destination_pixel_bytes)
            done += take
        destination_tile_offset += source.channel_tiles


def compile_precompiled_hardware(payload: dict) -> dict:
    """Compile one fixed-address, HALT-terminated program without an FPGA."""
    previous_axi_width = user_dma_core.UE_AXI_DATA_WIDTH_BITS
    user_dma_core.UE_AXI_DATA_WIDTH_BITS = PRECOMPILED_AXI_DATA_WIDTH_BITS
    try:
        return _compile_precompiled_hardware(payload)
    finally:
        user_dma_core.UE_AXI_DATA_WIDTH_BITS = previous_axi_width


def _compile_precompiled_hardware(payload: dict) -> dict:
    (layouts, heads, head_bundle_address, head_bundle_bytes,
     scratch_address, scratch_bytes) = _build_memory_plan(payload)
    image = _ImageBuilder(MODEL_BASE, MODEL_LIMIT)
    zero_address = image.allocate(
        torch.zeros(_ACT_TEMPLATE_BYTES // 2, dtype=torch.bfloat16),
        alignment=64)
    neg_inf_address = image.allocate(
        torch.full((_ACT_TEMPLATE_BYTES // 2,), float("-inf"),
                   dtype=torch.bfloat16), alignment=64)

    conv_plans = {}
    for operation in payload["operations"]:
        if operation["op"] == "conv":
            conv_plans[operation["name"]] = _prepare_conv_plan(
                operation, payload["weights"][operation["name"]],
                layouts[operation["inputs"][0]], layouts[operation["output"]],
                image)

    program_address = image.align(64)
    engine = _WholeGraphEngine(program_address)
    entries = []
    engine.start_capture()
    for graph_index, operation in enumerate(payload["operations"]):
        start = engine.get_capture_instruction_size_bytes()
        sources = [layouts[name] for name in operation["inputs"]]
        destination = layouts[operation["output"]]
        if operation["op"] == "conv":
            _emit_conv(engine, conv_plans[operation["name"]], zero_address)
        elif operation["op"] == "maxpool":
            _emit_maxpool(
                engine, operation, sources[0], destination, neg_inf_address)
        elif operation["op"] == "upsample2x":
            _emit_upsample(engine, sources[0], destination, scratch_address)
        elif operation["op"] == "add":
            _emit_add(engine, sources[0], sources[1], destination)
        elif operation["op"] == "concat":
            _emit_concat(engine, sources, destination)
        else:
            raise AssertionError(operation["op"])
        stop = engine.get_capture_instruction_size_bytes()
        if stop <= start:
            raise RuntimeError(f"{operation['name']}: emitted no instructions")
        entries.append({
            "graph_index": graph_index,
            "name": operation["name"],
            "op": operation["op"],
            "program_offset": start,
            "program_size": stop - start,
        })
    engine.generate_instruction_halt()
    engine.stop_capture()
    issues = user_dma_core.check_isa_jumps(
        engine.capture_buffer, program_address, name="YOLOv5 whole graph")
    if issues:
        raise RuntimeError("invalid whole-graph program jumps:\n" + "\n".join(issues))
    program = b"".join(
        instruction.get_bytes() for instruction in engine.capture_buffer)
    if not program or len(program) % 64:
        raise RuntimeError("whole-graph program is empty or not 64-byte aligned")
    image.write(program_address, program)
    model_image = _bytes_tensor(image.data)
    program_offset = program_address - MODEL_BASE
    hardware = {
        "abi": PRECOMPILED_ABI,
        "geometry_abi": GEOMETRY_ABI,
        "input_base": INPUT_BASE,
        "input_bytes": layouts["input"].size_bytes,
        "model_base": MODEL_BASE,
        "model_limit": MODEL_LIMIT,
        "tensor_base": TENSOR_BASE,
        "tensor_limit": TENSOR_LIMIT,
        "program_address": program_address,
        "program_offset": program_offset,
        "program_size": len(program),
        "model_image": model_image,
        "model_sha256": _image_sha256(model_image),
        "program_sha256": hashlib.sha256(program).hexdigest(),
        "tensors": {name: layout.manifest()
                    for name, layout in layouts.items()},
        "heads": heads,
        "head_bundle_address": head_bundle_address,
        "head_bundle_bytes": head_bundle_bytes,
        "scratch_address": scratch_address,
        "scratch_bytes": scratch_bytes,
        "operations": entries,
    }
    validate_precompiled_hardware(payload, hardware)
    return hardware


def precompiled_manifest_sha256(hardware: dict) -> str:
    """Digest every fixed-address field except the deployment bytes/hashes."""
    omitted = {"model_image", "model_sha256", "program_sha256"}
    descriptor = {
        key: value for key, value in hardware.items() if key not in omitted}
    encoded = json.dumps(
        descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _instruction_types(program: bytes) -> list[int]:
    return [
        (struct.unpack_from("<I", program, offset)[0] >> 8) & 0xF
        for offset in range(0, len(program), 32)
    ]


def validate_precompiled_hardware(payload: dict,
                                  hardware: dict | None = None) -> None:
    """Validate the closed, non-relocatable whole-graph hardware section."""
    hardware = payload.get("hardware") if hardware is None else hardware
    if not isinstance(hardware, dict):
        raise RuntimeError("YOLO artifact is missing its whole-graph image")
    expected_keys = {
        "abi", "geometry_abi", "input_base", "input_bytes", "model_base",
        "model_limit", "tensor_base", "tensor_limit", "program_address",
        "program_offset", "program_size", "model_image", "model_sha256",
        "program_sha256", "tensors", "heads", "head_bundle_address",
        "head_bundle_bytes", "scratch_address", "scratch_bytes", "operations",
    }
    if set(hardware) != expected_keys:
        raise RuntimeError("YOLO whole-graph hardware schema has unknown/missing fields")
    expected_scalars = {
        "abi": PRECOMPILED_ABI,
        "geometry_abi": GEOMETRY_ABI,
        "input_base": INPUT_BASE,
        "model_base": MODEL_BASE,
        "model_limit": MODEL_LIMIT,
        "tensor_base": TENSOR_BASE,
        "tensor_limit": TENSOR_LIMIT,
    }
    if any(hardware[key] != value for key, value in expected_scalars.items()):
        raise RuntimeError("YOLO whole-graph image uses an incompatible fixed-address ABI")

    image = hardware["model_image"]
    if (not isinstance(image, torch.Tensor) or image.dtype != torch.uint8
            or image.dim() != 1 or image.numel() == 0
            or image.numel() > MODEL_LIMIT - MODEL_BASE):
        raise RuntimeError("YOLO deployment image is invalid")
    if hardware["model_sha256"] != _image_sha256(image):
        raise RuntimeError("YOLO deployment-image digest does not match")
    program_offset = hardware["program_offset"]
    program_size = hardware["program_size"]
    program_address = hardware["program_address"]
    if (not isinstance(program_offset, int) or not isinstance(program_size, int)
            or program_address != MODEL_BASE + program_offset
            or program_address % 64 or program_size <= 0 or program_size % 64
            or program_offset < 0 or program_offset + program_size > image.numel()):
        raise RuntimeError("YOLO whole-graph program range is invalid")
    program = _tensor_bytes(image)[program_offset:program_offset + program_size]
    if hardware["program_sha256"] != hashlib.sha256(program).hexdigest():
        raise RuntimeError("YOLO whole-graph program digest does not match")
    instruction_types = _instruction_types(program)
    halt_positions = [index for index, value in enumerate(instruction_types)
                      if value == user_dma_core.INSTRUCTION_HALT]
    if len(halt_positions) != 1:
        raise RuntimeError("YOLO whole-graph program must contain exactly one HALT")
    if user_dma_core.INSTRUCTION_SWI in instruction_types:
        raise RuntimeError("YOLO whole-graph program must not contain SWI boundaries")
    halt = halt_positions[0]
    if any(value != user_dma_core.INSTRUCTION_NOP
           for value in instruction_types[halt + 1:]):
        raise RuntimeError("YOLO whole-graph HALT is not terminal")
    _scan_queue_configs(program, 0, len(program))
    decoded = []
    for offset in range(0, len(program), 32):
        instruction = user_dma_core.Instructions()
        instruction.words = list(struct.unpack_from("<8I", program, offset))
        decoded.append(instruction)
    jump_issues = user_dma_core.check_isa_jumps(
        decoded, program_address, name="YOLOv5 whole graph")
    if jump_issues:
        raise RuntimeError(
            "invalid whole-graph program jumps:\n" + "\n".join(jump_issues))

    (layouts, heads, head_bundle_address, head_bundle_bytes,
     scratch_address, scratch_bytes) = _build_memory_plan(payload)
    expected_tensors = {
        name: layout.manifest() for name, layout in layouts.items()}
    if hardware["tensors"] != expected_tensors:
        raise RuntimeError("YOLO whole-graph tensor memory plan differs")
    if hardware["input_bytes"] != layouts["input"].size_bytes:
        raise RuntimeError("YOLO packed-input size differs from the graph")
    if (hardware["heads"] != heads
            or hardware["head_bundle_address"] != head_bundle_address
            or hardware["head_bundle_bytes"] != head_bundle_bytes):
        raise RuntimeError("YOLO bundled detection-head layout differs")
    if (hardware["scratch_address"] != scratch_address
            or hardware["scratch_bytes"] != scratch_bytes):
        raise RuntimeError("YOLO whole-graph scratch layout differs")

    entries = hardware["operations"]
    operations = payload["operations"]
    if not isinstance(entries, list) or len(entries) != len(operations):
        raise RuntimeError("YOLO whole-graph operation manifest is incomplete")
    cursor = 0
    entry_keys = {"graph_index", "name", "op", "program_offset", "program_size"}
    for graph_index, (entry, operation) in enumerate(zip(entries, operations)):
        if not isinstance(entry, dict) or set(entry) != entry_keys:
            raise RuntimeError("YOLO whole-graph operation entry is invalid")
        if (entry["graph_index"] != graph_index
                or entry["name"] != operation["name"]
                or entry["op"] != operation["op"]
                or entry["program_offset"] != cursor
                or not isinstance(entry["program_size"], int)
                or entry["program_size"] <= 0
                or entry["program_size"] % 32):
            raise RuntimeError("YOLO whole-graph operation order/range differs")
        cursor += entry["program_size"]
    if cursor != halt * 32:
        raise RuntimeError("YOLO graph entries do not end immediately before HALT")


class WholeGraphAndromedaBackend:
    """One model upload, one input upload, one kick, and one result read."""

    uses_precompiled_hardware = True
    executes_whole_graph = True

    def __init__(self, ue: user_dma_core.UnifiedEngine, payload: dict, *,
                 axi_data_width_bits: int,
                 timeout_s: float = 300.0):
        validate_precompiled_hardware(payload)
        if int(axi_data_width_bits) != PRECOMPILED_AXI_DATA_WIDTH_BITS:
            raise RuntimeError(
                f"whole-graph YOLO was compiled for AXI-"
                f"{PRECOMPILED_AXI_DATA_WIDTH_BITS}, live hardware reports "
                f"AXI-{axi_data_width_bits}")
        if getattr(ue, "conv_geometry_mode", None) != \
                user_dma_core.CONV_GEOMETRY_QUEUE_CONFIG:
            raise RuntimeError(
                "whole-graph YOLO requires queue-config-v1 engine mode")
        self.ue = ue
        self.payload = payload
        self.hardware = payload["hardware"]
        self.timeout_s = float(timeout_s)
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be finite and > 0, got {timeout_s!r}")
        cached_version = getattr(ue, "hw_version", None)
        self.hw_version = (int(cached_version) if cached_version is not None
                           else user_dma_core.UnifiedEngine.get_hardware_version(ue))
        self.cycles: dict[str, int] = collections.defaultdict(int)
        self.instruction_bytes: dict[str, int] = collections.defaultdict(int)
        self.model_upload_bytes = 0
        self.model_upload_seconds = 0.0
        self.model_upload_writes = 0
        self.input_upload_writes = 0
        self.program_kicks = 0
        self.output_reads = 0
        self.intermediate_upload_writes = 0
        self.intermediate_output_reads = 0
        self._load_model_image()

    @property
    def static_dram_load_bytes(self) -> int:
        return self.model_upload_bytes

    @property
    def static_dram_load_seconds(self) -> float:
        return self.model_upload_seconds

    @property
    def static_dram_load_writes(self) -> int:
        return self.model_upload_writes

    def _load_model_image(self) -> None:
        image = self.hardware["model_image"]
        started = time.perf_counter_ns()
        written = self.ue.dma_write(
            self.ue.h2c_device, self.hardware["model_base"],
            image, image.numel())
        elapsed = (time.perf_counter_ns() - started) / 1e9
        if written != image.numel():
            raise RuntimeError(
                f"loading whole-graph model image wrote {written} of "
                f"{image.numel()} bytes")
        self.model_upload_bytes = int(written)
        self.model_upload_seconds = elapsed
        self.model_upload_writes = 1

    def _wait_strict(self) -> None:
        deadline = time.monotonic() + self.timeout_s
        while True:
            cause = self.ue.read_reg32(user_dma_core.UE_INT_REG) & 0x3
            if cause == user_dma_core.INT_CAUSE_HALT:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"whole-graph YOLO never reported HALT within "
                    f"{self.timeout_s:.1f}s")
            time.sleep(0.0001)
        while self.ue.is_queue_busy():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"whole-graph YOLO remained busy beyond {self.timeout_s:.1f}s")
            time.sleep(0.001)

    @staticmethod
    def _unpack_head(flat: torch.Tensor, head: dict) -> torch.Tensor:
        channels, height, width = (int(value)
                                   for value in head["logical_shape"])
        physical = int(head["physical_channels"])
        values = flat.view(height, width, physical)
        indexes = torch.tensor(head["channel_map"], dtype=torch.long)
        if indexes.numel() != channels:
            raise RuntimeError("detection-head channel map is invalid")
        return values.index_select(2, indexes).permute(2, 0, 1).contiguous()

    def execute(self, image_chw: torch.Tensor) -> list[torch.Tensor]:
        expected = tuple(int(value) for value in self.payload["model"]["input_shape"])
        if tuple(image_chw.shape) != expected:
            raise ValueError(
                f"artifact expects CHW input {expected}, got {tuple(image_chw.shape)}")
        before = (self.input_upload_writes, self.program_kicks, self.output_reads)
        packed = user_dma_core.conv2d_pack_activation_map(
            image_chw.to(torch.bfloat16), 0)
        input_bytes = packed.numel() * packed.element_size()
        if input_bytes != self.hardware["input_bytes"]:
            raise RuntimeError(
                f"packed input is {input_bytes} bytes, artifact expects "
                f"{self.hardware['input_bytes']}")
        written = self.ue.dma_write(
            self.ue.h2c_device, self.hardware["input_base"],
            packed, input_bytes)
        if written != input_bytes:
            raise RuntimeError(
                f"packed-input DMA wrote {written} of {input_bytes} bytes")
        self.input_upload_writes += 1

        if self.ue.is_queue_busy():
            raise RuntimeError("cannot start whole-graph YOLO while queue is busy")
        self.ue.write_reg32(user_dma_core.UE_INT_REG, 1)
        self.ue.start_execute_from_dram(self.hardware["program_address"])
        self.program_kicks += 1
        self._wait_strict()
        self.cycles["whole_graph"] += self.ue.read_latency_cycles()
        self.instruction_bytes["whole_graph"] += self.hardware["program_size"]

        bundle = torch.empty(
            self.hardware["head_bundle_bytes"] // 2, dtype=torch.bfloat16)
        read = self.ue.dma_read(
            self.ue.c2h_device, self.hardware["head_bundle_address"],
            bundle, self.hardware["head_bundle_bytes"])
        if read != self.hardware["head_bundle_bytes"]:
            raise RuntimeError(
                f"bundled result DMA read {read} of "
                f"{self.hardware['head_bundle_bytes']} bytes")
        self.output_reads += 1
        result = []
        for head in self.hardware["heads"]:
            start = int(head["offset"]) // 2
            count = int(head["size_bytes"]) // 2
            result.append(self._unpack_head(bundle[start:start + count], head))
        after = (self.input_upload_writes, self.program_kicks, self.output_reads)
        if tuple(new - old for old, new in zip(before, after)) != (1, 1, 1):
            raise RuntimeError("whole-graph transport contract was violated")
        if self.intermediate_upload_writes or self.intermediate_output_reads:
            raise RuntimeError("whole-graph execution performed intermediate transport")
        return result


# Preserve the old import name while changing its semantics to graph-level
# execution.  No per-primitive methods are implemented intentionally.
PrecompiledAndromedaBackend = WholeGraphAndromedaBackend
