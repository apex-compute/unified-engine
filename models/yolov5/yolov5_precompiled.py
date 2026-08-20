"""Offline FPGA image compiler and direct runner for the YOLOv5 artifact.

Queued CONFIG instructions carry CONV2D/MAXPOOL geometry in program order, so
the direct runner never writes live geometry CSRs.  The graph still needs one
host dispatch per primitive.  This module moves build-time work out of inference:
weights/scales/biases are
prepacked at fixed DRAM addresses and every primitive program is captured
offline into a resident instruction image.
"""

from __future__ import annotations

import collections
import contextlib
import hashlib
import io
import json
import math
import struct
import time
from typing import Sequence

import torch
import torch.nn.functional as F

import user_dma_core


PRECOMPILED_ABI = "andromeda-yolov5-precompiled-v3"
GEOMETRY_ABI = "conv-config-inst-v1"

# Keep the image-dependent staging arena separate from the compact immutable
# parameter image. The optimized 256 gather stem is well below this 256 MiB
# staging arena, leaving a useful guard before the static section. Programs
# embed all of these values.
DYNAMIC_PARAMS_BASE = user_dma_core.DRAM_START_ADDR       # 0x80000000
STATIC_PARAMS_BASE = 0x90000000
STATIC_PARAMS_LIMIT = user_dma_core.DRAM_ACTIVATION_ADDR  # 0xB0000000
TENSOR_BASE = user_dma_core.DRAM_ACTIVATION_ADDR          # 0xB0000000
TENSOR_LIMIT = user_dma_core.DRAM_INSTRUCTION_ADDR        # 0xD0000000
PROGRAM_BASE = user_dma_core.DRAM_INSTRUCTION_ADDR        # 0xD0000000
PROGRAM_LIMIT = 1 << 32

_GEOMETRY_REGISTERS = (
    user_dma_core.UE_CONV_GEOM_ADDR,
    user_dma_core.UE_CONV_CTRL_ADDR,
    user_dma_core.UE_CONV_STRIDE_ADDR,
    user_dma_core.UE_CONV_PIXSTEP_ADDR,
)


def _tensor_bytes(value: torch.Tensor, size: int | None = None) -> bytes:
    value = value.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        raw = value.view(torch.uint16).numpy().tobytes()
    else:
        raw = value.numpy().tobytes()
    return raw if size is None else raw[:size]


def _scan_queue_configs(
        image, offset: int, size: int) -> list[tuple[int, int, int, int]]:
    """Return and validate CONFIG subtype-0 payloads in one program range."""
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


def _image_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_bytes(value)).hexdigest()


def precompiled_manifest_sha256(hardware: dict) -> str:
    """Digest every dispatch/address field, excluding the byte images/hashes."""
    omitted = {"params_image", "program_image", "params_sha256", "program_sha256"}
    descriptor = {key: value for key, value in hardware.items()
                  if key not in omitted}
    encoded = json.dumps(
        descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bytes_tensor(value: bytearray) -> torch.Tensor:
    if not value:
        return torch.empty(0, dtype=torch.uint8)
    return torch.frombuffer(value, dtype=torch.uint8).clone()


def _unpack_codes(packed: torch.Tensor, shape: Sequence[int],
                  precision: str) -> torch.Tensor:
    count = 1
    for value in shape:
        count *= int(value)
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


class _ImageBuilder:
    def __init__(self, base: int, limit: int):
        self.base = int(base)
        self.limit = int(limit)
        self.data = bytearray()
        self.spans: list[tuple[int, int]] = []

    def write(self, address: int, raw: bytes) -> None:
        address, end = int(address), int(address) + len(raw)
        if address < self.base or end > self.limit:
            raise RuntimeError(
                f"compiled image write [{address:#x}, {end:#x}) is outside "
                f"[{self.base:#x}, {self.limit:#x})")
        for old_start, old_end in self.spans:
            if address < old_end and old_start < end:
                raise RuntimeError(
                    f"overlapping compiled image writes [{address:#x}, {end:#x}) "
                    f"and [{old_start:#x}, {old_end:#x})")
        offset = address - self.base
        if len(self.data) < offset:
            self.data.extend(b"\0" * (offset - len(self.data)))
        if len(self.data) < offset + len(raw):
            self.data.extend(b"\0" * (offset + len(raw) - len(self.data)))
        self.data[offset:offset + len(raw)] = raw
        self.spans.append((address, end))


class _OfflineEngine(user_dma_core.UnifiedEngine):
    """Instruction emitter that deliberately has no device side effects."""

    def __init__(self):
        # This mirrors the state-only portion of UnifiedEngine.__init__.  Calling
        # the real constructor would touch BAR registers and run a DRAM test.
        self.device = "cpu"
        self.h2c_device = "offline"
        self.c2h_device = "offline"
        self.user_device = "offline"
        self._user_fd = None
        self.latency_count = 0
        self.capture_buffer = []
        self.capture_count = 0
        self.is_capture_on = False
        self._params_dram_base = DYNAMIC_PARAMS_BASE
        self._program_dram_base = PROGRAM_BASE
        self._tensor_dram_base = TENSOR_BASE
        self._next_params_dram_addr = DYNAMIC_PARAMS_BASE
        self._next_program_dram_addr = PROGRAM_BASE
        self._tensor_dram_addr = TENSOR_BASE
        self._isa_reg_counter = 1
        self._inst_ptr_counter = 1
        self._dram_addresses = {}
        self._inst_id = 0
        self._capture_loop_stack = []
        self._capture_conv_geometry = None
        self._clock_period_ns = 10.0
        self._allow_unknown_conv_hardware = True
        self._allow_unknown_queue_config_hardware = True
        self._allow_unknown_gather_if8_hardware = True
        self.conv_geometry_mode = user_dma_core.CONV_GEOMETRY_QUEUE_CONFIG
        self._base_addr = 0
        self.hw_version = 0

        self._params_image = _ImageBuilder(STATIC_PARAMS_BASE, STATIC_PARAMS_LIMIT)
        self._program_image = _ImageBuilder(PROGRAM_BASE, PROGRAM_LIMIT)
        self._static_next = STATIC_PARAMS_BASE
        self._dynamic_next = DYNAMIC_PARAMS_BASE
        self._dynamic_remaining = 0
        self._current_params: list[list[int]] = []
        self._current_tensors: list[list[int]] = []
        self._current_program: tuple[int, int] | None = None
        self._current_name = ""

    def begin_operation(self, dynamic_params: int, name: str = "") -> None:
        if self.is_capture_on or self.capture_buffer:
            raise RuntimeError("offline compiler started an operation with a live capture")
        self._dynamic_next = DYNAMIC_PARAMS_BASE
        self._dynamic_remaining = int(dynamic_params)
        self._tensor_dram_addr = TENSOR_BASE
        self._next_params_dram_addr = DYNAMIC_PARAMS_BASE
        self._current_params = []
        self._current_tensors = []
        self._current_program = None
        self._current_name = str(name)
        self.reset_isa_reg_counter()
        self.reset_inst_ptr_counter()

    def finish_operation(self, graph_index: int, operation: dict) -> dict:
        if self._dynamic_remaining:
            raise RuntimeError(
                f"{operation['name']}: expected {self._dynamic_remaining} more "
                "dynamic parameter allocations")
        if self._current_program is None:
            raise RuntimeError(f"{operation['name']}: no program was captured")
        program_address, program_size = self._current_program
        program_offset = program_address - PROGRAM_BASE
        configs = _scan_queue_configs(
            self._program_image.data, program_offset, program_size)
        expects_config = operation["op"] in ("conv", "maxpool")
        if expects_config and not configs:
            raise RuntimeError(
                f"{operation['name']}: capture omitted queued geometry CONFIG")
        if not expects_config and configs:
            raise RuntimeError(
                f"{operation['name']}: unexpected queued geometry CONFIG")
        return {
            "graph_index": int(graph_index),
            "name": operation["name"],
            "op": operation["op"],
            "program_offset": program_offset,
            "program_size": program_size,
            "queue_config_count": len(configs),
            "params_allocations": self._current_params,
            "tensor_allocations": self._current_tensors,
        }

    def allocate_params_dram(self, size_bytes: int, label=None,
                             align_bytes: int = 64) -> int:
        size_bytes = int(size_bytes)
        if self._dynamic_remaining:
            address = self._align_up(self._dynamic_next, align_bytes)
            end = address + size_bytes
            if end > STATIC_PARAMS_BASE:
                raise RuntimeError(
                    f"dynamic parameter workspace reaches {end:#x}, static image "
                    f"starts at {STATIC_PARAMS_BASE:#x}")
            self._dynamic_next = end
            self._dynamic_remaining -= 1
            self._current_params.append([address, size_bytes])
        else:
            address = self._align_up(self._static_next, align_bytes)
            end = address + size_bytes
            if end > STATIC_PARAMS_LIMIT:
                raise RuntimeError("static YOLO parameters exceed their DRAM arena")
            self._static_next = end
        self._next_params_dram_addr = end
        if label is not None:
            self._dram_addresses[label] = address
        return address

    def allocate_tensor_dram(self, size_bytes: int, label=None,
                             align_bytes: int = 64) -> int:
        size_bytes = int(size_bytes)
        address = self._align_up(self._tensor_dram_addr, align_bytes)
        end = address + size_bytes
        if end > TENSOR_LIMIT:
            raise RuntimeError("YOLO tensor workspace exceeds its DRAM arena")
        self._tensor_dram_addr = end
        self._current_tensors.append([address, size_bytes])
        if label is not None:
            self._dram_addresses[label] = address
        return address

    def dma_write(self, _device: str, address: int, buffer, size: int) -> int:
        raw = (_tensor_bytes(buffer, size) if isinstance(buffer, torch.Tensor)
               else bytes(buffer)[:size])
        if len(raw) != size:
            raise RuntimeError(f"offline DMA serialization made {len(raw)} of {size} bytes")
        if STATIC_PARAMS_BASE <= address < STATIC_PARAMS_LIMIT:
            self._params_image.write(address, raw)
        elif PROGRAM_BASE <= address < PROGRAM_LIMIT:
            self._program_image.write(address, raw)
            self._current_program = (int(address), int(size))
        elif DYNAMIC_PARAMS_BASE <= address < STATIC_PARAMS_BASE:
            # Image-dependent activation operands are intentionally absent.
            pass
        else:
            raise RuntimeError(f"offline compiler saw an unexpected DMA address {address:#x}")
        return int(size)

    def dma_from_accelerator_memory(self, _address: int, shape) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.bfloat16)

    def write_captured_instructions_to_dram(
            self, start_addr: int = PROGRAM_BASE) -> int:
        if not self.capture_buffer or not self.capture_count:
            raise RuntimeError(f"{self._current_name}: empty offline instruction capture")
        self.pad_capture_to_64b_boundary()
        issues = user_dma_core.check_isa_jumps(
            self.capture_buffer, int(start_addr), name=self._current_name)
        if issues:
            raise RuntimeError("invalid offline program jumps:\n" + "\n".join(issues))
        raw = b"".join(instruction.get_bytes()
                       for instruction in self.capture_buffer)
        return self.dma_write("offline", start_addr, raw, len(raw))

    def write_reg32(self, address: int, value: int) -> None:
        if address in _GEOMETRY_REGISTERS:
            raise RuntimeError(
                f"offline queued compiler attempted live geometry write {address:#x}")

    def read_reg32(self, _address: int) -> int:
        return 0

    def start_execute_from_dram(self, _instruction_addr: int = PROGRAM_BASE):
        return None

    def wait_queue(self, *_args, **_kwargs):
        return None


def compile_precompiled_hardware(payload: dict) -> dict:
    """Create fixed-address params/program images without opening an FPGA."""
    engine = _OfflineEngine()
    shapes: dict[str, tuple[int, int, int]] = {
        "input": tuple(int(v) for v in payload["model"]["input_shape"])
    }
    entries = []

    # The live helpers are the authoritative instruction emitters.  Feeding
    # zero tensors through the side-effect-free engine also exercises their
    # planners and keeps the compiled ABI byte-identical to normal capture.
    with contextlib.redirect_stdout(io.StringIO()), torch.inference_mode():
        for graph_index, operation in enumerate(payload["operations"]):
            kind = operation["op"]
            input_shapes = [shapes[key] for key in operation["inputs"]]
            if kind == "concat":
                shapes[operation["output"]] = tuple(operation["output_shape"])
                continue

            dynamic_count = {"conv": 1, "maxpool": 1,
                             "upsample2x": 1, "add": 2}[kind]
            engine.begin_operation(dynamic_count, operation["name"])
            inputs = [torch.zeros(shape, dtype=torch.bfloat16)
                      for shape in input_shapes]
            if kind == "conv":
                weight = payload["weights"][operation["name"]]
                precision = weight["precision"]
                data_type = user_dma_core.TYPE[precision.upper()]
                gather = weight["layout"] == "gather"
                codes = _unpack_codes(
                    weight["codes_packed"], weight["codes_shape"], precision)
                bias = weight["bias"]
                result = engine.run_conv2d_layer(
                    inputs[0], codes,
                    stride_s=operation["stride"], pad=operation["pad"],
                    dilation=operation["dilation"],
                    block_scales=weight["block_scales"],
                    bias=None if bias.numel() == 0 else bias,
                    silu_enable=operation["activate"],
                    data_type=data_type,
                    gather=gather)
            elif kind == "maxpool":
                result = engine.run_maxpool2d_layer(
                    inputs[0], kernel=operation["kernel"],
                    stride_s=operation["stride"], pad=operation["pad"])
            elif kind == "upsample2x":
                result = engine.run_nn_upsample_2x(inputs[0])
            elif kind == "add":
                result = engine.run_eltwise_add_layer(inputs[0], inputs[1])
            else:
                raise AssertionError(kind)
            if tuple(result.shape) != tuple(operation["output_shape"]):
                raise RuntimeError(
                    f"offline {operation['name']} produced {tuple(result.shape)}, "
                    f"expected {tuple(operation['output_shape'])}")
            entries.append(engine.finish_operation(graph_index, operation))
            shapes[operation["output"]] = tuple(operation["output_shape"])
            del inputs, result

    params_image = _bytes_tensor(engine._params_image.data)
    program_image = _bytes_tensor(engine._program_image.data)
    hardware = {
        "abi": PRECOMPILED_ABI,
        "geometry_abi": GEOMETRY_ABI,
        "dynamic_params_base": DYNAMIC_PARAMS_BASE,
        "dynamic_params_limit": STATIC_PARAMS_BASE,
        "static_params_base": STATIC_PARAMS_BASE,
        "static_params_limit": STATIC_PARAMS_LIMIT,
        "tensor_base": TENSOR_BASE,
        "tensor_limit": TENSOR_LIMIT,
        "program_base": PROGRAM_BASE,
        "program_limit": PROGRAM_LIMIT,
        "params_image": params_image,
        "program_image": program_image,
        "params_sha256": _image_sha256(params_image),
        "program_sha256": _image_sha256(program_image),
        "entries": entries,
    }
    validate_precompiled_hardware(payload, hardware)
    return hardware


def _validate_allocations(value, *, low: int, high: int, label: str) -> None:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} allocations are not a list")
    spans = []
    for item in value:
        if (not isinstance(item, list) or len(item) != 2
                or any(not isinstance(v, int) for v in item)):
            raise RuntimeError(f"{label} has an invalid allocation {item!r}")
        address, size = item
        if address % 64 or size <= 0 or address < low or address + size > high:
            raise RuntimeError(f"{label} allocation {item!r} is outside its arena")
        spans.append((address, address + size))
    for index, (start, end) in enumerate(spans):
        if any(start < other_end and other_start < end
               for other_start, other_end in spans[:index]):
            raise RuntimeError(f"{label} allocations overlap")


def validate_precompiled_hardware(payload: dict, hardware: dict | None = None) -> None:
    """Validate the non-relocatable hardware sections and dispatch table."""
    hardware = payload.get("hardware") if hardware is None else hardware
    if not isinstance(hardware, dict):
        raise RuntimeError("YOLO artifact is missing its precompiled hardware image")
    expected_keys = {
        "abi", "geometry_abi", "dynamic_params_base", "dynamic_params_limit",
        "static_params_base", "static_params_limit", "tensor_base",
        "tensor_limit", "program_base", "program_limit", "params_image",
        "program_image", "params_sha256", "program_sha256", "entries",
    }
    if set(hardware) != expected_keys:
        raise RuntimeError("YOLO precompiled hardware schema has unknown/missing fields")
    expected_scalars = {
        "abi": PRECOMPILED_ABI,
        "geometry_abi": GEOMETRY_ABI,
        "dynamic_params_base": DYNAMIC_PARAMS_BASE,
        "dynamic_params_limit": STATIC_PARAMS_BASE,
        "static_params_base": STATIC_PARAMS_BASE,
        "static_params_limit": STATIC_PARAMS_LIMIT,
        "tensor_base": TENSOR_BASE,
        "tensor_limit": TENSOR_LIMIT,
        "program_base": PROGRAM_BASE,
        "program_limit": PROGRAM_LIMIT,
    }
    if any(hardware[key] != value for key, value in expected_scalars.items()):
        raise RuntimeError("YOLO precompiled image uses an incompatible fixed-address ABI")

    params = hardware["params_image"]
    programs = hardware["program_image"]
    if (not isinstance(params, torch.Tensor) or params.dtype != torch.uint8
            or params.dim() != 1 or params.numel() == 0
            or params.numel() > STATIC_PARAMS_LIMIT - STATIC_PARAMS_BASE):
        raise RuntimeError("YOLO precompiled params image is invalid")
    if (not isinstance(programs, torch.Tensor) or programs.dtype != torch.uint8
            or programs.dim() != 1 or programs.numel() == 0
            or programs.numel() > PROGRAM_LIMIT - PROGRAM_BASE):
        raise RuntimeError("YOLO precompiled program image is invalid")
    if hardware["params_sha256"] != _image_sha256(params):
        raise RuntimeError("YOLO precompiled params digest does not match")
    if hardware["program_sha256"] != _image_sha256(programs):
        raise RuntimeError("YOLO precompiled program digest does not match")

    expected_ops = [(i, op) for i, op in enumerate(payload["operations"])
                    if op["op"] != "concat"]
    entries = hardware["entries"]
    if not isinstance(entries, list) or len(entries) != len(expected_ops):
        raise RuntimeError(
            f"YOLO precompiled table needs {len(expected_ops)} FPGA entries")
    entry_keys = {
        "graph_index", "name", "op", "program_offset", "program_size",
        "queue_config_count", "params_allocations", "tensor_allocations",
    }
    expected_program_offset = 0
    for entry, (graph_index, operation) in zip(entries, expected_ops):
        if not isinstance(entry, dict) or set(entry) != entry_keys:
            raise RuntimeError("YOLO precompiled dispatch entry has invalid fields")
        if (entry["graph_index"] != graph_index
                or entry["name"] != operation["name"]
                or entry["op"] != operation["op"]):
            raise RuntimeError("YOLO precompiled dispatch order differs from the graph")
        offset, size = entry["program_offset"], entry["program_size"]
        if (not isinstance(offset, int) or not isinstance(size, int)
                or offset % 64 or size <= 0 or size % 64
                or offset + size > programs.numel()):
            raise RuntimeError(f"{entry['name']}: invalid precompiled program range")
        if offset != expected_program_offset:
            raise RuntimeError(
                f"{entry['name']}: program range aliases or leaves an unexpected gap")
        expected_program_offset += size
        configs = _scan_queue_configs(programs, offset, size)
        expected_config = entry["op"] in ("conv", "maxpool")
        count = entry["queue_config_count"]
        if (not isinstance(count, int) or isinstance(count, bool)
                or count != len(configs)):
            raise RuntimeError(f"{entry['name']}: queued CONFIG count is invalid")
        if expected_config and count <= 0:
            raise RuntimeError(f"{entry['name']}: queued geometry CONFIG is missing")
        if not expected_config and count != 0:
            raise RuntimeError(f"{entry['name']}: unexpected queued geometry CONFIG")
        _validate_allocations(
            entry["params_allocations"], low=DYNAMIC_PARAMS_BASE,
            high=STATIC_PARAMS_BASE, label=f"{entry['name']} params")
        _validate_allocations(
            entry["tensor_allocations"], low=TENSOR_BASE,
            high=TENSOR_LIMIT, label=f"{entry['name']} tensors")
        expected_counts = {
            "conv": (1, 1), "maxpool": (1, 1),
            "upsample2x": (1, 2), "add": (2, 1),
        }[entry["op"]]
        if (len(entry["params_allocations"]), len(entry["tensor_allocations"])) != expected_counts:
            raise RuntimeError(f"{entry['name']}: unexpected workspace allocation count")
    if expected_program_offset != programs.numel():
        raise RuntimeError("YOLO program image has unreferenced trailing bytes")


class PrecompiledAndromedaBackend:
    """Execute resident artifact programs; only dynamic operands move per op."""

    uses_precompiled_hardware = True

    def __init__(self, ue: user_dma_core.UnifiedEngine, payload: dict, *,
                 timeout_s: float = 300.0):
        validate_precompiled_hardware(payload)
        ue.require_native_conv_hardware()
        ue.require_queue_conv_config_hardware()
        if any(weight.get("layout") == "gather"
               and weight.get("precision") == "if8"
               for weight in payload.get("weights", {}).values()):
            ue.require_gather_if8_hardware()
        if getattr(ue, "conv_geometry_mode", None) != \
                user_dma_core.CONV_GEOMETRY_QUEUE_CONFIG:
            raise RuntimeError(
                "precompiled YOLO v4 requires queue-config-v1 engine mode")
        expected_bases = {
            "_params_dram_base": DYNAMIC_PARAMS_BASE,
            "_tensor_dram_base": TENSOR_BASE,
            "_program_dram_base": PROGRAM_BASE,
        }
        for attribute, expected in expected_bases.items():
            actual = getattr(ue, attribute, expected)
            if actual != expected:
                raise RuntimeError(
                    f"precompiled YOLO programs require {attribute}={expected:#x}, "
                    f"engine uses {actual:#x}; this artifact is not relocatable")
        self.ue = ue
        self.payload = payload
        self.hardware = payload["hardware"]
        self.timeout_s = float(timeout_s)
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be finite and > 0, got {timeout_s!r}")
        # ``cycles`` is expressed in FPGA aclk cycles. UnifiedEngine owns the
        # latency-counter prescaler conversion; model backends never interpret
        # raw register units.
        self.cycles: dict[str, int] = collections.defaultdict(int)
        self.instruction_bytes: dict[str, int] = collections.defaultdict(int)
        self.static_dram_load_bytes = 0
        self.static_dram_load_seconds = 0.0
        self.static_dram_load_writes = 0
        self.hw_version = int(ue.hw_version) & 0xFFFFFFFF
        self._entries = {entry["name"]: entry for entry in self.hardware["entries"]}
        self._operations = {operation["name"]: operation
                            for operation in payload["operations"]}
        self._load_images()

    def _load_images(self) -> None:
        for label, image_key, base_key in (
                ("params", "params_image", "static_params_base"),
                ("program", "program_image", "program_base")):
            image = self.hardware[image_key]
            base = self.hardware[base_key]
            started = time.perf_counter_ns()
            written = self.ue.dma_write(
                self.ue.h2c_device, base, image, image.numel())
            elapsed = (time.perf_counter_ns() - started) / 1e9
            if written != image.numel():
                raise RuntimeError(
                    f"loading precompiled {label} image wrote {written} of "
                    f"{image.numel()} bytes")
            self.static_dram_load_bytes += int(written)
            self.static_dram_load_seconds += elapsed
            self.static_dram_load_writes += 1

    @staticmethod
    def _require_allocation(entry: dict, kind: str, index: int,
                            size: int) -> int:
        address, compiled_size = entry[kind][index]
        if int(size) != compiled_size:
            raise RuntimeError(
                f"{entry['name']}: runtime {kind}[{index}] is {size} bytes, "
                f"compiled for {compiled_size}")
        return address

    def _write_dynamic(self, address: int, value: torch.Tensor) -> None:
        size = value.numel() * value.element_size()
        written = self.ue.dma_write(self.ue.h2c_device, address, value, size)
        if written != size:
            raise RuntimeError(
                f"dynamic operand DMA wrote {written} of {size} bytes at {address:#x}")

    def _read_bf16(self, address: int, count: int) -> torch.Tensor:
        # A successful XDMA read overwrites every byte and the exact byte count
        # is checked below, so zero-filling large result buffers is wasted work.
        value = torch.empty(int(count), dtype=torch.bfloat16)
        size = value.numel() * 2
        read = self.ue.dma_read(self.ue.c2h_device, address, value, size)
        if read != size:
            raise RuntimeError(
                f"dynamic result DMA read {read} of {size} bytes at {address:#x}")
        return value

    def _wait_strict(self) -> None:
        # queue_busy can both assert after the posted kick returns and disappear
        # between two PCIe BAR reads.  HALT's interrupt cause is latched, so it
        # is the reliable completion handshake for these HALT-terminated bins.
        deadline = time.monotonic() + self.timeout_s
        while True:
            cause = self.ue.read_reg32(user_dma_core.UE_INT_REG) & 0x3
            if cause == user_dma_core.INT_CAUSE_HALT:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"precompiled YOLO dispatch never reported HALT "
                    f"within {self.timeout_s:.1f}s")
            time.sleep(0.0001)
        # HALT is terminal; wait for the exposed busy bit to settle before the
        # next resident program is kicked.
        while self.ue.is_queue_busy():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"precompiled YOLO dispatch timed out after {self.timeout_s:.1f}s")
            time.sleep(0.001)

    def _dispatch(self, entry: dict) -> None:
        if self.ue.is_queue_busy():
            raise RuntimeError("cannot start a program while queue is busy")
        program_address = self.hardware["program_base"] + entry["program_offset"]
        # Any write clears the prior latched cause; a failed kick can therefore
        # no longer masquerade as completion from the preceding node.
        self.ue.write_reg32(user_dma_core.UE_INT_REG, 1)
        self.ue.start_execute_from_dram(program_address)
        self._wait_strict()
        self.instruction_bytes[entry["op"]] += entry["program_size"]
        self.cycles[entry["op"]] += self.ue.read_latency_cycles()

    def conv_prepared(self, name: str, x: torch.Tensor, _prepared, *,
                      stride: int, pad: int, dilation: int,
                      activate: bool) -> torch.Tensor:
        entry = self._entries[name]
        operation = self._operations[name]
        if (operation["op"] != "conv" or operation["stride"] != stride
                or operation["pad"] != pad or operation["dilation"] != dilation
                or operation["activate"] != bool(activate)):
            raise RuntimeError(f"{name}: invocation differs from its compiled convolution")
        C, H, W = (int(v) for v in x.shape)
        weight = self.payload["weights"][name]
        OC, weight_c, kh, kw = (int(v) for v in weight["codes_shape"])
        if weight_c != C:
            raise RuntimeError(f"{name}: runtime input channels differ from the bin")
        gather = weight["layout"] == "gather"
        out_h, out_w, oc_chunk, tiles = user_dma_core.plan_conv2d_layer_tiles(
            c_in=C, oc_count=OC, in_h=H, in_w=W,
            kernel_h=kh, kernel_w=kw, stride_s=stride,
            pad=pad, pad_h=pad, dilation=dilation, gather=gather,
            bias_enabled=weight["bias"].numel() != 0)
        ct = (C + user_dma_core.UE_VECTOR_SIZE - 1) // user_dma_core.UE_VECTOR_SIZE
        th, tw = tiles[0][2], tiles[0][3]
        win_h, win_w = tiles[0][6], tiles[0][7]
        win_lines = win_h * win_w * ct
        act_bytes = win_lines * 128
        n_tiles = len(tiles)
        n_chunks = OC // oc_chunk
        result_lines = (th * tw * oc_chunk + user_dma_core.UE_VECTOR_SIZE - 1) // user_dma_core.UE_VECTOR_SIZE
        out_bytes = result_lines * 128

        staged = user_dma_core.conv2d_pack_activation_tiles(
            x, tiles, pad=pad)
        if staged.numel() * staged.element_size() != n_tiles * act_bytes:
            raise RuntimeError(f"{name}: staged activation size differs from bin")
        act_address = self._require_allocation(
            entry, "params_allocations", 0, n_tiles * act_bytes)
        self._write_dynamic(act_address, staged)
        out_address = self._require_allocation(
            entry, "tensor_allocations", 0, n_chunks * n_tiles * out_bytes)
        self._dispatch(entry)

        big = self._read_bf16(
            out_address,
            n_chunks * n_tiles * result_lines * user_dma_core.UE_VECTOR_SIZE)
        return user_dma_core.conv2d_unpack_tiled_result(
            big, tiles, out_h, out_w, OC, oc_chunk)

    def maxpool(self, name: str, x: torch.Tensor, *, kernel: int,
                stride: int, pad: int) -> torch.Tensor:
        entry = self._entries[name]
        operation = self._operations[name]
        if (operation["op"] != "maxpool" or operation["kernel"] != kernel
                or operation["stride"] != stride or operation["pad"] != pad):
            raise RuntimeError(f"{name}: invocation differs from its compiled maxpool")
        C, H, W = (int(v) for v in x.shape)
        h_pad, w_pad = H + 2 * pad, W + 2 * pad
        out_h = (h_pad - kernel) // stride + 1
        out_w = (w_pad - kernel) // stride + 1

        def fits(rows: int) -> bool:
            win = (rows - 1) * stride + kernel
            return (kernel * kernel * rows * out_w <= 0xFFF
                    and win * w_pad <= 0x300
                    and 0x300 + rows * out_w <= 4096
                    and rows * out_w <= 0xFFFF)

        rows = 1
        while rows < out_h and fits(rows + 1):
            rows += 1
        oy0s = list(range(0, max(out_h - rows, 0) + 1, rows))
        if oy0s[-1] != out_h - rows:
            oy0s.append(out_h - rows)
        win_h = (rows - 1) * stride + kernel
        win_lines = win_h * w_pad
        act_bytes = win_lines * 128
        out_bytes = rows * out_w * 128
        slots = [(c0, min(user_dma_core.UE_VECTOR_SIZE, C - c0), oy0)
                 for c0 in range(0, C, user_dma_core.UE_VECTOR_SIZE)
                 for oy0 in oy0s]
        x_padded = F.pad(
            x.to(torch.bfloat16), (pad, pad, pad, pad), value=float("-inf"))
        staged = torch.empty(
            len(slots) * win_lines, user_dma_core.UE_VECTOR_SIZE,
            dtype=torch.bfloat16)
        for index, (c0, n_ch, oy0) in enumerate(slots):
            y0 = oy0 * stride
            window = x_padded[c0:c0 + n_ch, y0:y0 + win_h, :]
            staged[index * win_lines:(index + 1) * win_lines] = \
                user_dma_core.conv2d_pack_activation_map(
                    window, 0, pad_value=float("-inf"))
        act_address = self._require_allocation(
            entry, "params_allocations", 0, len(slots) * act_bytes)
        out_address = self._require_allocation(
            entry, "tensor_allocations", 0, len(slots) * out_bytes)
        self._write_dynamic(act_address, staged)
        self._dispatch(entry)
        big = self._read_bf16(
            out_address,
            len(slots) * rows * out_w * user_dma_core.UE_VECTOR_SIZE)
        big = big.view(len(slots), rows * out_w * user_dma_core.UE_VECTOR_SIZE)
        out = torch.zeros(C, out_h, out_w, dtype=torch.bfloat16)
        for index, (c0, n_ch, oy0) in enumerate(slots):
            out[c0:c0 + n_ch, oy0:oy0 + rows, :] = \
                user_dma_core.maxpool2d_unpack_result(
                    big[index], rows, out_w)[:n_ch]
        return out

    def upsample2x(self, name: str, x: torch.Tensor) -> torch.Tensor:
        entry = self._entries[name]
        C, H, W = (int(v) for v in x.shape)
        out_h, out_w, ct, _passes = user_dma_core.plan_nn_upsample_2x(H, W, C)
        packed = user_dma_core.conv2d_pack_activation_map(
            x.to(torch.bfloat16), 0)
        in_size = H * W * ct * 128
        in_address = self._require_allocation(
            entry, "params_allocations", 0, in_size)
        self._require_allocation(
            entry, "tensor_allocations", 0, 2 * H * W * ct * 128)
        out_address = self._require_allocation(
            entry, "tensor_allocations", 1, out_h * out_w * ct * 128)
        self._write_dynamic(in_address, packed)
        self._dispatch(entry)
        flat = self._read_bf16(
            out_address, out_h * out_w * ct * user_dma_core.UE_VECTOR_SIZE)
        return user_dma_core.nn_upsample_2x_unpack_result(
            flat, out_h, out_w, C)

    def add(self, name: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        entry = self._entries[name]
        if x.shape != y.shape:
            raise RuntimeError(f"{name}: residual operands have different shapes")
        C, H, W = (int(v) for v in x.shape)
        ct = (C + user_dma_core.UE_VECTOR_SIZE - 1) // user_dma_core.UE_VECTOR_SIZE
        total_lines = H * W * ct
        size = total_lines * 128
        xs = user_dma_core.conv2d_pack_activation_map(x.to(torch.bfloat16), 0)
        ys = user_dma_core.conv2d_pack_activation_map(y.to(torch.bfloat16), 0)
        x_address = self._require_allocation(
            entry, "params_allocations", 0, size)
        y_address = self._require_allocation(
            entry, "params_allocations", 1, size)
        out_address = self._require_allocation(
            entry, "tensor_allocations", 0, size)
        self._write_dynamic(x_address, xs)
        self._write_dynamic(y_address, ys)
        self._dispatch(entry)
        flat = self._read_bf16(
            out_address, total_lines * user_dma_core.UE_VECTOR_SIZE)
        return user_dma_core.packed_map_to_chw(flat, H, W, C)

    def concat(self, _name: str, values: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat(tuple(values), dim=0)
