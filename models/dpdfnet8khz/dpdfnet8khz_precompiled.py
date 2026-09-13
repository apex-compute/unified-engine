"""Native 8 kHz artifact validation and one-START/HALT resident execution.

Tensor packing, instruction decoding and the frame execution mechanism reuse
the established DPDFNet helpers. The model identity, graph and streaming ABI
are validated independently so a 16 kHz artifact cannot enter this runner.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for search_path in (HERE, ROOT / "models" / "dpdfnet", ROOT, ROOT / "models" / "yolov5s"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from dpdfnet_precompiled import (
    MODEL_BASE, MODEL_LIMIT, INPUT_BASE, INPUT_LIMIT, TENSOR_BASE, TENSOR_LIMIT,
    STATE_ALIGNMENT, TENSOR_ALIGNMENT, TensorLayout, DeviceEmitter, align_up,
    build_layout_plan, tensor_manifest, physical_indices, make_layout,
    pack_tensor, unpack_tensor, manifest_sha256, transform_source_indices,
    WholeGraphBackend as _FrameBackend, shared, udc,
)
from dpdfnet8khz_common import load_config

FORMAT = "andromeda.dpdfnet2_8khz.streaming-v1"


def validate_hardware(payload: dict) -> None:
    """Validate the closed, fixed-address, single-HALT deployment image."""
    if payload.get("format") != FORMAT or payload.get("model") != "dpdfnet2_8khz":
        raise RuntimeError(
            "incompatible native 8 kHz artifact; rebuild with dpdfnet8khz_compile.py --force")
    expected_digest = load_config()["onnx_sha256"]
    if payload.get("onnx_sha256") != expected_digest:
        raise RuntimeError("DPDFNet8K source-model checksum is not the pinned 8 kHz model")
    hardware = payload.get("hardware")
    if not isinstance(hardware, dict):
        raise RuntimeError("DPDFNet8K artifact has no hardware image")
    if hardware.get("axi_data_width_bits") != 256:
        raise RuntimeError("DPDFNet8K artifact must declare its compiled AXI-256 target; rebuild it")
    if not hardware.get("full_graph") or not hardware.get("one_halt") \
            or not hardware.get("stateful"):
        raise RuntimeError("DPDFNet8K artifact is not a stateful whole graph")
    if hardware.get("model_base") != MODEL_BASE:
        raise RuntimeError("DPDFNet8K model base is incompatible")
    image = hardware.get("model_image")
    if (not isinstance(image, torch.Tensor) or image.dtype != torch.uint8
            or image.ndim != 1 or not 0 < image.numel() <= MODEL_LIMIT - MODEL_BASE):
        raise RuntimeError("invalid DPDFNet8K deployment image")
    raw = shared._tensor_bytes(image)
    if manifest_sha256(raw) != hardware.get("model_sha256"):
        raise RuntimeError("DPDFNet8K deployment image checksum mismatch")
    offset = hardware.get("program_offset")
    size = hardware.get("program_size")
    address = hardware.get("program_address")
    if (not isinstance(offset, int) or not isinstance(size, int)
            or offset < 0 or offset % 64 or size <= 0 or size % 32
            or offset + size != len(raw) or address != MODEL_BASE + offset):
        raise RuntimeError("invalid DPDFNet8K resident-program bounds")
    program = raw[offset:offset + size]
    if hashlib.sha256(program).hexdigest() != hardware.get("program_sha256"):
        raise RuntimeError("DPDFNet8K resident-program checksum mismatch")
    types = shared._instruction_types(program)
    halts = [index for index, kind in enumerate(types)
             if kind == udc.INSTRUCTION_HALT]
    if len(halts) != 1 or udc.INSTRUCTION_SWI in types:
        raise RuntimeError("DPDFNet8K program requires one HALT and no SWI")
    halt = halts[0]
    if any(kind != udc.INSTRUCTION_NOP for kind in types[halt + 1:]):
        raise RuntimeError("DPDFNet8K HALT is not terminal")
    entries = hardware.get("operations")
    graph_operations = hardware.get("graph_operations")
    if (not isinstance(entries, list) or graph_operations != 492
            or len(entries) != graph_operations + 1):
        raise RuntimeError("DPDFNet8K operation manifest is incomplete")
    cursor = 0
    for index, entry in enumerate(entries):
        expected_name = "@state_commit" if index == graph_operations else None
        if (not isinstance(entry, dict)
                or set(entry) != {"name", "node_index", "start", "stop"}
                or entry["node_index"] != index or entry["start"] != cursor
                or not isinstance(entry["stop"], int) or entry["stop"] <= cursor
                or (expected_name is not None and entry["name"] != expected_name)):
            raise RuntimeError("DPDFNet8K operation order/range mismatch")
        cursor = entry["stop"]
    if cursor != halt:
        raise RuntimeError("DPDFNet8K operations do not end immediately before HALT")
    tensors = hardware.get("tensors")
    if not isinstance(tensors, dict):
        raise RuntimeError("DPDFNet8K tensor manifest is missing")
    for required, shape, base in (
            ("spec", (1, 1, 81, 2), INPUT_BASE),
            ("state_in", (37860,), MODEL_BASE),
            ("spec_e", (1, 1, 81, 2), None),
            ("state_out", (37860,), None)):
        if required not in tensors:
            raise RuntimeError(f"DPDFNet8K tensor {required!r} is missing")
        layout = TensorLayout.from_manifest(tensors[required])
        if layout.shape != shape or (base is not None and layout.address != base):
            raise RuntimeError(f"DPDFNet8K tensor {required!r} has incompatible layout")
        if not TENSOR_BASE <= layout.address < TENSOR_LIMIT \
                and required not in ("spec", "state_in"):
            raise RuntimeError(f"DPDFNet8K tensor {required!r} is outside tensor DRAM")
    shared._scan_queue_configs(program, 0, len(program))
    issues = udc.check_isa_jumps(
        shared.decode_precompiled_program(hardware), address,
        name="DPDFNet2 8 kHz whole graph")
    if issues:
        raise RuntimeError("invalid DPDFNet8K program jumps:\n" + "\n".join(issues))


def load_artifact(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"DPDFNet8K single bin not found: {path}. Build it with "
            "dpdfnet8khz_compile.py --force")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    validate_hardware(payload)
    return payload


def validate_runtime_hardware(payload: dict, axi_data_width_bits: int) -> None:
    """Reject a transport mismatch before resetting or writing to the device."""
    compiled_width = payload["hardware"].get("axi_data_width_bits")
    if compiled_width != 256 or axi_data_width_bits != compiled_width:
        raise RuntimeError(
            "DPDFNet8K bin targets RK AXI-256; live hardware reports "
            f"AXI-{axi_data_width_bits}. Use the compatible RK-256 queue-CONFIG "
            "build. A --device label does not change the FPGA interface.")


class WholeGraphBackend(_FrameBackend):
    """Use the shared frame executor with the independently validated 8 kHz ABI."""

    def __init__(self, engine: udc.UnifiedEngine, payload: dict, *,
                 axi_data_width_bits: int, timeout_s: float = 300.0,
                 trace_tail_path: str | Path | None = None):
        validate_hardware(payload)
        validate_runtime_hardware(payload, axi_data_width_bits)
        if getattr(engine, "conv_geometry_mode", None) != \
                udc.CONV_GEOMETRY_QUEUE_CONFIG:
            raise RuntimeError("DPDFNet8K requires queue-config-v1 CONV geometry")
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
            raise RuntimeError("short DPDFNet8K model upload")
        self.model_upload_seconds = time.perf_counter() - started
        self.input_upload_writes = 0
        self.program_kicks = 0
        self.output_reads = 0
        self.cycles = 0
        self._failure_reason: str | None = None
