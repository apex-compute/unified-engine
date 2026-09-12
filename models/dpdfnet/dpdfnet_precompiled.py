"""Fixed-shape, stateful, single-HALT DPDFNet2 deployment support.

The physical tensor ABI pads only the final logical dimension to the
accelerator's 64-lane width.  Every graph value consequently has a simple
row-major description, while layout operators are lowered to device DMA
copies rather than host transformations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import struct
import sys
import time
from typing import Iterable, Sequence

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


FORMAT = "andromeda.dpdfnet2.streaming-v1"
MODEL_BASE = shared.MODEL_BASE
MODEL_LIMIT = shared.MODEL_LIMIT
INPUT_BASE = shared.INPUT_BASE
INPUT_LIMIT = shared.INPUT_LIMIT
TENSOR_BASE = shared.TENSOR_BASE
TENSOR_LIMIT = shared.TENSOR_LIMIT
STATE_ALIGNMENT = 128
TENSOR_ALIGNMENT = 128


def align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


@dataclass(frozen=True)
class TensorLayout:
    name: str
    shape: tuple[int, ...]
    address: int
    padded_last: int

    @property
    def rows(self) -> int:
        return math.prod(self.shape[:-1]) if self.shape else 1

    @property
    def logical_last(self) -> int:
        return self.shape[-1] if self.shape else 1

    @property
    def logical_elements(self) -> int:
        return math.prod(self.shape) if self.shape else 1

    @property
    def physical_elements(self) -> int:
        return self.rows * self.padded_last

    @property
    def size_bytes(self) -> int:
        return self.physical_elements * 2

    def physical_index(self, logical_flat_index: int) -> int:
        logical_flat_index = int(logical_flat_index)
        if not 0 <= logical_flat_index < self.logical_elements:
            raise IndexError(logical_flat_index)
        row, column = divmod(logical_flat_index, self.logical_last)
        return row * self.padded_last + column

    def manifest(self) -> dict:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "address": int(self.address),
            "padded_last": int(self.padded_last),
            "size_bytes": int(self.size_bytes),
        }

    @classmethod
    def from_manifest(cls, value: dict) -> "TensorLayout":
        result = cls(
            name=str(value["name"]),
            shape=tuple(int(item) for item in value["shape"]),
            address=int(value["address"]),
            padded_last=int(value["padded_last"]),
        )
        if result.size_bytes != int(value["size_bytes"]):
            raise RuntimeError(f"tensor layout size mismatch for {result.name}")
        return result


def make_layout(name: str, shape: Sequence[int], address: int) -> TensorLayout:
    shape = tuple(int(value) for value in shape)
    if any(value <= 0 for value in shape):
        raise ValueError(f"{name}: tensor dimensions must be positive, got {shape}")
    logical_last = shape[-1] if shape else 1
    padded_last = align_up(logical_last, udc.UE_VECTOR_SIZE)
    return TensorLayout(str(name), shape, int(address), padded_last)


def pack_tensor(value: torch.Tensor | np.ndarray, layout: TensorLayout) -> torch.Tensor:
    """Pack a logical tensor into the deployment's padded BF16 row layout."""
    tensor = torch.as_tensor(value)
    if tuple(tensor.shape) != layout.shape:
        raise ValueError(
            f"{layout.name}: expected shape {layout.shape}, got {tuple(tensor.shape)}")
    logical = tensor.reshape(layout.rows, layout.logical_last).to(torch.bfloat16)
    result = torch.zeros(
        layout.rows, layout.padded_last, dtype=torch.bfloat16)
    result[:, :layout.logical_last] = logical
    return result.flatten().contiguous()


def unpack_tensor(value: torch.Tensor, layout: TensorLayout) -> torch.Tensor:
    """Remove physical padding from a BF16 deployment tensor."""
    if value.numel() != layout.physical_elements:
        raise ValueError(
            f"{layout.name}: expected {layout.physical_elements} physical "
            f"elements, got {value.numel()}")
    return value.reshape(layout.rows, layout.padded_last)[
        :, :layout.logical_last].reshape(layout.shape).contiguous()


def physical_indices(layout: TensorLayout) -> np.ndarray:
    """Map each logical-flat element to its physical padded-flat slot."""
    result = np.arange(layout.logical_elements, dtype=np.int64)
    rows = result // layout.logical_last
    columns = result % layout.logical_last
    return rows * layout.padded_last + columns


def copy_runs(source_indices: Sequence[int], destination_indices: Sequence[int]):
    """Coalesce element mappings into source+destination-contiguous runs."""
    source = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    destination = np.asarray(destination_indices, dtype=np.int64).reshape(-1)
    if source.shape != destination.shape:
        raise ValueError("source/destination index maps differ in length")
    if not source.size:
        return []
    starts = np.concatenate((
        np.array([0], dtype=np.int64),
        np.nonzero((np.diff(source) != 1) | (np.diff(destination) != 1))[0] + 1,
    ))
    stops = np.concatenate((starts[1:], np.array([source.size], dtype=np.int64)))
    return [
        (int(source[start]), int(destination[start]), int(stop - start))
        for start, stop in zip(starts, stops)
    ]


def copy_patterns(source_indices: Sequence[int], destination_indices: Sequence[int]):
    """Coalesce mappings with contiguous destinations and fixed positive source stride.

    Returns ``(source, destination, count, source_stride)`` in BF16 elements.
    A source stride of one is a normal contiguous copy; larger strides are
    represented by the read DMA's gather mode.
    """
    source = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    destination = np.asarray(destination_indices, dtype=np.int64).reshape(-1)
    if source.shape != destination.shape:
        raise ValueError("source/destination index maps differ in length")
    result = []
    start = 0
    while start < source.size:
        if start + 1 < source.size and destination[start + 1] == destination[start] + 1:
            stride = int(source[start + 1] - source[start])
            if stride <= 0:
                stride = None
        else:
            stride = None
        stop = start + 1
        if stride is not None:
            stop += 1
            while (stop < source.size
                   and destination[stop] == destination[stop - 1] + 1
                   and source[stop] == source[stop - 1] + stride):
                stop += 1
        result.append((
            int(source[start]), int(destination[start]), int(stop - start),
            1 if stride is None else stride,
        ))
        start = stop
    return result


def transform_source_indices(
        source: TensorLayout, output_shape: Sequence[int],
        *, permutation: Sequence[int] | None = None,
        slices: Sequence[slice | int] | None = None) -> np.ndarray:
    """Return source physical indices in output logical-flat order."""
    logical = np.arange(source.logical_elements, dtype=np.int64).reshape(source.shape)
    if slices is not None:
        logical = logical[tuple(slices)]
    if permutation is not None:
        logical = logical.transpose(tuple(int(value) for value in permutation))
    output_shape = tuple(int(value) for value in output_shape)
    if tuple(logical.shape) != output_shape:
        try:
            logical = logical.reshape(output_shape)
        except ValueError as exc:
            raise ValueError(
                f"transform produces {logical.shape}, expected {output_shape}") from exc
    logical = logical.reshape(-1)
    rows = logical // source.logical_last
    columns = logical % source.logical_last
    return rows * source.padded_last + columns


def build_layout_plan(model) -> tuple[dict[str, TensorLayout], int]:
    """Assign deterministic non-overlapping addresses to every graph value."""
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("building DPDFNet layouts requires onnx") from exc
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {}
    for value in [*inferred.graph.input, *inferred.graph.value_info,
                  *inferred.graph.output]:
        dims = value.type.tensor_type.shape.dim
        if any(not dim.HasField("dim_value") or int(dim.dim_value) <= 0
               for dim in dims):
            raise RuntimeError(f"{value.name}: dynamic/invalid graph shape")
        shapes[value.name] = tuple(int(dim.dim_value) for dim in dims)

    layouts = {}
    # The spectrum is the only per-frame H2C transfer. State lives in the
    # writable model image and is initialized by that image's one-time upload.
    layouts["spec"] = make_layout("spec", shapes["spec"], INPUT_BASE)
    state_probe = make_layout("state_in", shapes["state_in"], MODEL_BASE)
    state_address = align_up(MODEL_BASE, STATE_ALIGNMENT)
    layouts["state_in"] = make_layout("state_in", shapes["state_in"], state_address)
    cursor = TENSOR_BASE
    for node in inferred.graph.node:
        for name in node.output:
            if name not in shapes:
                raise RuntimeError(f"shape inference omitted graph value {name!r}")
            cursor = align_up(cursor, TENSOR_ALIGNMENT)
            layout = make_layout(name, shapes[name], cursor)
            if layout.address + layout.size_bytes > TENSOR_LIMIT:
                raise RuntimeError("DPDFNet tensor arena exceeds its DRAM range")
            layouts[name] = layout
            cursor += align_up(layout.size_bytes, TENSOR_ALIGNMENT)
    if state_probe.size_bytes != layouts["state_in"].size_bytes:
        raise AssertionError("state layout changed unexpectedly")
    if layouts["spec"].address + layouts["spec"].size_bytes > INPUT_LIMIT:
        raise RuntimeError("DPDFNet spectrum input exceeds input arena")
    return layouts, align_up(cursor, TENSOR_ALIGNMENT)


def tensor_manifest(layouts: dict[str, TensorLayout]) -> dict[str, dict]:
    return {name: layout.manifest() for name, layout in layouts.items()}


def manifest_sha256(value: bytes | bytearray | torch.Tensor) -> str:
    raw = shared._tensor_bytes(value) if isinstance(value, torch.Tensor) else bytes(value)
    return hashlib.sha256(raw).hexdigest()


class DeviceEmitter:
    """Shared offline image builder and fixed-address instruction emitter."""

    def __init__(self, layouts: dict[str, TensorLayout], tensor_end: int,
                 initial_state: torch.Tensor):
        self.layouts = layouts
        self.tensor_end = int(tensor_end)
        self.image = shared._ImageBuilder(MODEL_BASE, MODEL_LIMIT)
        state_layout = layouts["state_in"]
        if state_layout.address != MODEL_BASE:
            raise RuntimeError("state must begin the one-shot deployment image")
        self.image.write(MODEL_BASE, pack_tensor(initial_state, state_layout))
        self.image.align(64)
        largest = max(layout.size_bytes for layout in layouts.values())
        if largest > udc.URAM_NEAR_FULL_SIZE:
            raise RuntimeError("largest DPDFNet value exceeds one-URAM DMA staging")
        self.zero_address = self.image.allocate(
            torch.zeros(largest // 2, dtype=torch.bfloat16))
        self.engine = None
        self.operation_ranges = []
        self._constant_cache: dict[tuple, int] = {}

    def allocate_constant(self, value, layout: TensorLayout | None = None) -> int:
        tensor = torch.as_tensor(value)
        if layout is not None:
            tensor = pack_tensor(tensor, layout)
        elif tensor.dtype != torch.bfloat16:
            tensor = tensor.to(torch.bfloat16)
        tensor = tensor.contiguous().flatten()
        raw = shared._tensor_bytes(tensor)
        key = (str(tensor.dtype), tuple(tensor.shape), hashlib.sha256(raw).digest())
        if key not in self._constant_cache:
            self._constant_cache[key] = self.image.allocate(tensor)
        return self._constant_cache[key]

    def begin_program(self) -> int:
        address = self.image.align(64)
        self.engine = shared._WholeGraphEngine(address)
        self.engine.start_capture()
        return address

    def mark_operation(self, name: str, node_index: int, callback) -> None:
        if self.engine is None:
            raise RuntimeError("program capture has not started")
        start = self.engine.capture_count
        callback()
        stop = self.engine.capture_count
        self.operation_ranges.append({
            "name": str(name), "node_index": int(node_index),
            "start": int(start), "stop": int(stop),
        })

    def finish_program(self, program_address: int) -> bytes:
        if self.engine is None:
            raise RuntimeError("program capture has not started")
        self.engine.generate_instruction_halt()
        self.engine.stop_capture()
        program = b"".join(
            instruction.get_bytes() for instruction in self.engine.capture_buffer)
        self.image.write(program_address, program)
        return program

    def emit_zero(self, destination: TensorLayout) -> None:
        self.engine.accelerator_memory_to_sram(
            self.zero_address, 0, 0,
            memcpy_length_bytes=destination.size_bytes)
        self.engine.sram_to_accelerator_memory(
            0, destination.address, 0,
            memcpy_length_bytes=destination.size_bytes)

    def emit_clear_padding(self, destination: TensorLayout) -> None:
        padding = destination.padded_last - destination.logical_last
        if not padding:
            return
        padding_bytes = padding * 2
        total_bytes = destination.rows * padding_bytes
        self.engine.accelerator_memory_to_sram(
            self.zero_address, 0, 0, memcpy_length_bytes=total_bytes)
        if padding_bytes % udc.ue_axi_beat_bytes() == 0:
            self.engine.sram_to_accelerator_memory(
                0,
                destination.address + destination.logical_last * 2,
                0,
                memcpy_length_bytes=total_bytes,
                stride_bytes_per_chunk=padding_bytes,
                stride_jump_bytes=destination.padded_last * 2,
            )
        else:
            for row in range(destination.rows):
                self.engine.sram_to_accelerator_memory(
                    0,
                    destination.address
                    + (row * destination.padded_last
                       + destination.logical_last) * 2,
                    0, memcpy_length_bytes=padding_bytes)

    def emit_identity_lalu(self, source: TensorLayout,
                           destination: TensorLayout, identity_address: int,
                           mode: udc.LALU_MODE, *, scalar: float = 1.0) -> None:
        """Apply a LALU function lane-wise through a 64x64 identity matvec."""
        if source.shape != destination.shape \
                or source.padded_last != destination.padded_last:
            raise ValueError("identity-LALU source/destination layouts differ")
        line_bytes = udc.UE_VECTOR_SIZE * 2
        self.engine.accelerator_memory_to_sram(
            identity_address, 0x80000, udc.UE_VECTOR_SIZE ** 2)
        for line in range(source.physical_elements // udc.UE_VECTOR_SIZE):
            offset = line * line_bytes
            self.engine.accelerator_memory_to_sram(
                source.address + offset, 0, udc.UE_VECTOR_SIZE)
            self.engine.start_queue_for_bf16_matvec_operation(
                max_clear_en=0, fmax_context_addr=0,
                vector_sram_start_addr=0,
                matrix_sram_start_addr=0x80000,
                output_sram_wb_addr=0,
                K=udc.UE_VECTOR_SIZE, N=udc.UE_VECTOR_SIZE,
                lalu_mode=mode,
                lalu_scalar=self.engine.float_to_bf19(float(scalar)),
            )
            self.engine.sram_to_accelerator_memory(
                0, destination.address + offset, udc.UE_VECTOR_SIZE)
        self.emit_clear_padding(destination)

    def emit_mapping(self, source: TensorLayout, destination: TensorLayout,
                     source_physical: Sequence[int], *, zero=True) -> None:
        """Materialize a static reshape/transpose/slice/broadcast on-device."""
        source_physical = np.asarray(source_physical, dtype=np.int64).reshape(-1)
        if source_physical.size != destination.logical_elements:
            raise ValueError(
                f"{source.name}->{destination.name}: mapping has "
                f"{source_physical.size} elements, expected "
                f"{destination.logical_elements}")
        destination_physical = physical_indices(destination)
        if zero and destination.physical_elements != destination.logical_elements:
            self.emit_zero(destination)
        self.emit_scatter(
            source, destination, source_physical, destination_physical)

    def emit_scatter(self, source: TensorLayout, destination: TensorLayout,
                     source_physical: Sequence[int],
                     destination_physical: Sequence[int]) -> None:
        source_physical = np.asarray(source_physical, dtype=np.int64).reshape(-1)
        destination_physical = np.asarray(
            destination_physical, dtype=np.int64).reshape(-1)
        if source_physical.size != destination_physical.size:
            raise ValueError("scatter source/destination maps differ in length")
        for source_start, destination_start, count, stride in copy_patterns(
                source_physical, destination_physical):
            total = count * 2
            if stride == 1:
                self.engine.accelerator_memory_to_sram(
                    source.address + source_start * 2, 0, 0,
                    memcpy_length_bytes=total)
            else:
                jump = stride * 2
                if jump <= udc.UE_STRIDE_JUMP_MAX_BYTES:
                    self.engine.accelerator_memory_to_sram(
                        source.address + source_start * 2, 0, 0,
                        memcpy_length_bytes=total,
                        stride_bytes_per_chunk=2,
                        stride_jump_bytes=jump)
                else:
                    for offset in range(count):
                        self.engine.accelerator_memory_to_sram(
                            source.address + (source_start + offset * stride) * 2,
                            offset * 2, 0, memcpy_length_bytes=2)
            self.engine.sram_to_accelerator_memory(
                0, destination.address + destination_start * 2, 0,
                memcpy_length_bytes=total)

    def emit_identity_copy(self, source: TensorLayout,
                           destination: TensorLayout) -> None:
        indices = transform_source_indices(source, destination.shape)
        self.emit_mapping(source, destination, indices)


def validate_hardware(payload: dict) -> None:
    """Validate the closed, fixed-address, single-HALT deployment image."""
    if payload.get("format") != FORMAT or payload.get("model") != "dpdfnet2":
        raise RuntimeError("not a DPDFNet2 Andromeda streaming artifact")
    hardware = payload.get("hardware")
    if not isinstance(hardware, dict):
        raise RuntimeError("DPDFNet artifact has no hardware image")
    if not hardware.get("full_graph") or not hardware.get("one_halt") \
            or not hardware.get("stateful"):
        raise RuntimeError("DPDFNet artifact is not a stateful whole graph")
    if hardware.get("model_base") != MODEL_BASE:
        raise RuntimeError("DPDFNet model base is incompatible")
    image = hardware.get("model_image")
    if (not isinstance(image, torch.Tensor) or image.dtype != torch.uint8
            or image.ndim != 1 or not 0 < image.numel() <= MODEL_LIMIT - MODEL_BASE):
        raise RuntimeError("invalid DPDFNet deployment image")
    raw = shared._tensor_bytes(image)
    if manifest_sha256(raw) != hardware.get("model_sha256"):
        raise RuntimeError("DPDFNet deployment image checksum mismatch")
    offset = hardware.get("program_offset")
    size = hardware.get("program_size")
    address = hardware.get("program_address")
    if (not isinstance(offset, int) or not isinstance(size, int)
            or offset < 0 or offset % 64 or size <= 0 or size % 32
            or offset + size != len(raw) or address != MODEL_BASE + offset):
        raise RuntimeError("invalid DPDFNet resident-program bounds")
    program = raw[offset:offset + size]
    if hashlib.sha256(program).hexdigest() != hardware.get("program_sha256"):
        raise RuntimeError("DPDFNet resident-program checksum mismatch")
    types = shared._instruction_types(program)
    halts = [index for index, kind in enumerate(types)
             if kind == udc.INSTRUCTION_HALT]
    if len(halts) != 1 or udc.INSTRUCTION_SWI in types:
        raise RuntimeError("DPDFNet program requires one HALT and no SWI")
    halt = halts[0]
    if any(kind != udc.INSTRUCTION_NOP for kind in types[halt + 1:]):
        raise RuntimeError("DPDFNet HALT is not terminal")
    entries = hardware.get("operations")
    graph_operations = hardware.get("graph_operations")
    if (not isinstance(entries, list) or graph_operations != 472
            or len(entries) != graph_operations + 1):
        raise RuntimeError("DPDFNet operation manifest is incomplete")
    cursor = 0
    for index, entry in enumerate(entries):
        expected_name = "@state_commit" if index == graph_operations else None
        if (not isinstance(entry, dict)
                or set(entry) != {"name", "node_index", "start", "stop"}
                or entry["node_index"] != index or entry["start"] != cursor
                or not isinstance(entry["stop"], int) or entry["stop"] <= cursor
                or (expected_name is not None and entry["name"] != expected_name)):
            raise RuntimeError("DPDFNet operation order/range mismatch")
        cursor = entry["stop"]
    if cursor != halt:
        raise RuntimeError("DPDFNet operations do not end immediately before HALT")
    tensors = hardware.get("tensors")
    if not isinstance(tensors, dict):
        raise RuntimeError("DPDFNet tensor manifest is missing")
    for required, shape, base in (
            ("spec", (1, 1, 161, 2), INPUT_BASE),
            ("state_in", (45424,), MODEL_BASE),
            ("spec_e", (1, 1, 161, 2), None),
            ("state_out", (45424,), None)):
        if required not in tensors:
            raise RuntimeError(f"DPDFNet tensor {required!r} is missing")
        layout = TensorLayout.from_manifest(tensors[required])
        if layout.shape != shape or (base is not None and layout.address != base):
            raise RuntimeError(f"DPDFNet tensor {required!r} has incompatible layout")
        if not TENSOR_BASE <= layout.address < TENSOR_LIMIT \
                and required not in ("spec", "state_in"):
            raise RuntimeError(f"DPDFNet tensor {required!r} is outside tensor DRAM")
    shared._scan_queue_configs(program, 0, len(program))
    issues = udc.check_isa_jumps(
        shared.decode_precompiled_program(hardware), address,
        name="DPDFNet2 whole graph")
    if issues:
        raise RuntimeError("invalid DPDFNet program jumps:\n" + "\n".join(issues))


def load_artifact(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"DPDFNet single bin not found: {path}. Build it with "
            "dpdfnet_compile.py --force")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    validate_hardware(payload)
    return payload


class WholeGraphBackend:
    """One resident image, one spectrum write/kick/result read per frame."""

    def __init__(self, engine: udc.UnifiedEngine, payload: dict, *,
                 axi_data_width_bits: int, timeout_s: float = 300.0,
                 trace_tail_path: str | Path | None = None):
        validate_hardware(payload)
        if int(axi_data_width_bits) not in (256, 512):
            raise RuntimeError(
                "DPDFNet runtime supports AXI-256 or AXI-512, live hardware "
                f"reports AXI-{axi_data_width_bits}")
        if getattr(engine, "conv_geometry_mode", None) != \
                udc.CONV_GEOMETRY_QUEUE_CONFIG:
            raise RuntimeError("DPDFNet requires queue-config-v1 CONV geometry")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout must be finite and positive")
        self.ue = engine
        self.payload = payload
        self.hardware = payload["hardware"]
        self.timeout_s = float(timeout_s)
        self.trace_tail_path = (None if trace_tail_path is None
                                else Path(trace_tail_path).expanduser())
        self.trace_tail_result = None
        self.trace_export_seconds = 0.0
        self.model_upload_bytes = int(self.hardware["model_image"].numel())
        started = time.perf_counter()
        written = engine.dma_write(
            engine.h2c_device, self.hardware["model_base"],
            self.hardware["model_image"], self.model_upload_bytes)
        if written != self.model_upload_bytes:
            raise RuntimeError("short DPDFNet model upload")
        self.model_upload_seconds = time.perf_counter() - started
        self.input_upload_writes = 0
        self.program_kicks = 0
        self.output_reads = 0
        self.cycles = 0

    def _wait(self) -> None:
        deadline = time.monotonic() + self.timeout_s
        while True:
            cause = self.ue.read_reg32(udc.UE_INT_REG) & 3
            if cause == udc.INT_CAUSE_HALT:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("DPDFNet whole graph did not reach HALT")
            time.sleep(0.0001)
        while self.ue.is_queue_busy():
            if time.monotonic() >= deadline:
                raise TimeoutError("DPDFNet queue remained busy after HALT")
            time.sleep(0.001)

    def execute(self, spectrum: torch.Tensor | np.ndarray) -> torch.Tensor:
        spec = TensorLayout.from_manifest(self.hardware["tensors"]["spec"])
        output = TensorLayout.from_manifest(self.hardware["tensors"]["spec_e"])
        packed = pack_tensor(spectrum, spec)
        input_bytes = packed.numel() * packed.element_size()
        if self.ue.is_queue_busy():
            raise RuntimeError("cannot start DPDFNet while the queue is busy")
        written = self.ue.dma_write(
            self.ue.h2c_device, spec.address, packed, input_bytes)
        if written != input_bytes:
            raise RuntimeError("short DPDFNet spectrum upload")
        self.input_upload_writes += 1
        self.ue.write_reg32(udc.UE_INT_REG, 1)
        self.ue.start_execute_from_dram(self.hardware["program_address"])
        self.program_kicks += 1
        self._wait()
        self.cycles += self.ue.read_latency_cycles()
        if self.trace_tail_path is not None:
            from read_trace import generate_circular_tail_trace
            trace_started = time.perf_counter()
            labels = [""] * (self.hardware["program_size"] // 32)
            for entry in self.hardware["operations"]:
                labels[entry["start"]:entry["stop"]] = \
                    [entry["name"]] * (entry["stop"] - entry["start"])
            self.trace_tail_path.mkdir(parents=True, exist_ok=True)
            csv_path = self.trace_tail_path / "dpdfnet2_tail.csv"
            self.trace_tail_result = generate_circular_tail_trace(
                self.ue, csv_path,
                instructions=shared.decode_precompiled_program(self.hardware),
                instruction_labels=labels,
                program_dram_addr=self.hardware["program_address"])
            self.trace_export_seconds = time.perf_counter() - trace_started
        flat = torch.empty(output.physical_elements, dtype=torch.bfloat16)
        read = self.ue.dma_read(
            self.ue.c2h_device, output.address, flat, output.size_bytes)
        if read != output.size_bytes:
            raise RuntimeError("short DPDFNet spectrum-result read")
        self.output_reads += 1
        return unpack_tensor(flat, output)
