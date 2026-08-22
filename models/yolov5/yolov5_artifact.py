"""Single-file YOLOv5 compile artifacts and checkpoint-free executor.

The artifact is a restricted ``torch.save`` container holding only primitive
Python values and tensors: an explicit flattened graph, native mixed IF4/IF8 tensors,
anchors, metadata, a shared fixed-address prepacked parameter image, and
resident instruction images for a closed family of input profiles. The runtime
loads it with ``weights_only=True`` and never
imports/downloads the upstream checkpoint or captures hardware programs.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import types
from typing import Sequence

import torch
import user_dma_core

from yolov5_precompiled import (
    GEOMETRY_ABI,
    PRECOMPILED_ABI,
    compile_precompiled_hardware,
    precompiled_manifest_sha256,
    validate_precompiled_hardware,
)

from yolov5_common import (
    QuantizedConv,
    _class_names,
    _pair,
    execute_yolov5,
    gather_if8_is_profitable,
    get_yolov5_variant,
    quantize_conv_for_andromeda,
    sha256_file,
)


ARTIFACT_VERSION = 5

# Public resolution strings use the conventional WIDTHxHEIGHT spelling while
# tensor shapes remain CHW. Every profile is compiled offline into the bin;
# unsupported sizes fail closed instead of capturing a program at runtime.
SUPPORTED_INPUT_RESOLUTIONS = (
    "256x256",
    "320x320",
    "416x416",
    "512x512",
    "640x480",
    "640x640",
)
DEFAULT_INPUT_RESOLUTION = SUPPORTED_INPUT_RESOLUTIONS[0]


@dataclass(frozen=True)
class CanonicalArtifact:
    variant: str
    format: str
    graph_sha256: str
    weights_sha256: str
    params_sha256: str
    program_sha256: str
    dispatch_sha256: str
    bundle_sha256: str
    params_bytes: int
    program_bytes: int


CANONICAL_ARTIFACTS = {
    "s": CanonicalArtifact(
        variant="s",
        format="andromeda.yolov5s.single-bin",
        graph_sha256=(
            "0a3f6dbd9e3950fe5b13e829effeb5827c06321daee9d75f3fd690965939fc22"),
        weights_sha256=(
            "273634fad085fa82ed76b5c5f03d80abebf3c8edcf4c8dbda95697c8c8b7d4d3"),
        params_sha256=(
            "9b75c0f41b7fc4ae725b47355a6c5fee8f0f8254893d4225da1b5ad0f2ae29b2"),
        program_sha256=(
            "32561feb335f1030f94f536a12cb50852e2ab44d90c998eb2c1ae7fb1a10e08e"),
        dispatch_sha256=(
            "f81641aecb8e7c785be53c16601572a8a8bc747e911ac06bb6c754a06abd0cf9"),
        bundle_sha256=(
            "47474a0c006fa0576a9286109f97b0c28577fd7b6dbce59850413ae7c54881fd"),
        params_bytes=16_862_520,
        program_bytes=77_952,
    ),
    "n": CanonicalArtifact(
        variant="n",
        format="andromeda.yolov5n.single-bin",
        graph_sha256=(
            "69dbe201b089756886973e98b9f24e8b58c27baf195f7b31c97f7d9c9c2e4272"),
        weights_sha256=(
            "b8ca688e49a44c87b657de6b5c4c9d7b0193718c40ccc8f860733cd89865f666"),
        params_sha256=(
            "5901879827c9e9e10525d6879ce9f852941af9be2a4fd7c4d01deaa1101ca931"),
        program_sha256=(
            "e69b09076c9c6628960a1096142cea3493e3cd2b7c10c656f2075c87b4fc86a7"),
        dispatch_sha256=(
            "00b0af7c3b61ccd25c0a9726840664362ddfb700226ed3f2baa8887f01fb808b"),
        bundle_sha256=(
            "d90c1e2c4ac36ec7c3f8b9ee1649c35977767bf8d490ad8abed672849faf98b8"),
        params_bytes=18_219_952,
        program_bytes=93_504,
    ),
}

# Compatibility aliases retained for existing YOLOv5s callers/tests.
ARTIFACT_FORMAT = CANONICAL_ARTIFACTS["s"].format
CANONICAL_GRAPH_SHA256 = CANONICAL_ARTIFACTS["s"].graph_sha256
CANONICAL_WEIGHTS_SHA256 = CANONICAL_ARTIFACTS["s"].weights_sha256
CANONICAL_PARAMS_SHA256 = CANONICAL_ARTIFACTS["s"].params_sha256
CANONICAL_PROGRAM_SHA256 = CANONICAL_ARTIFACTS["s"].program_sha256
CANONICAL_DISPATCH_SHA256 = CANONICAL_ARTIFACTS["s"].dispatch_sha256
CANONICAL_PARAMS_BYTES = CANONICAL_ARTIFACTS["s"].params_bytes
CANONICAL_PROGRAM_BYTES = CANONICAL_ARTIFACTS["s"].program_bytes


def get_canonical_artifact(variant: str = "s") -> CanonicalArtifact:
    profile = get_yolov5_variant(variant)
    return CANONICAL_ARTIFACTS[profile.key]


def artifact_variant(payload: dict) -> str:
    """Return the canonical variant selected by a payload's format."""
    format_name = payload.get("format") if isinstance(payload, dict) else None
    for key, spec in CANONICAL_ARTIFACTS.items():
        if format_name == spec.format:
            return key
    raise RuntimeError(f"unsupported YOLO artifact format {format_name!r}")


def parse_input_resolution(value: str | int | Sequence[int]) -> tuple[int, int]:
    """Return ``(height, width)`` for a WIDTHxHEIGHT public resolution."""
    if isinstance(value, bool):
        raise ValueError(f"invalid input resolution {value!r}")
    if isinstance(value, int):
        height = width = int(value)
    elif isinstance(value, str):
        parts = value.lower().replace(" ", "").split("x")
        if len(parts) != 2:
            raise ValueError(
                f"resolution must use WIDTHxHEIGHT, got {value!r}")
        try:
            width, height = (int(part) for part in parts)
        except ValueError as exc:
            raise ValueError(
                f"resolution must use integer WIDTHxHEIGHT, got {value!r}") from exc
    else:
        values = tuple(int(item) for item in value)
        if len(values) != 2:
            raise ValueError(f"resolution must contain height and width, got {value!r}")
        height, width = values
    if height <= 0 or width <= 0 or height % 32 or width % 32:
        raise ValueError(
            f"input resolution must use positive multiples of 32, got "
            f"{width}x{height}")
    return height, width


def input_resolution_key(height: int, width: int) -> str:
    """Return the canonical public WIDTHxHEIGHT profile key."""
    height, width = parse_input_resolution((height, width))
    return f"{width}x{height}"


def available_input_resolutions(payload: dict) -> tuple[str, ...]:
    """List the default and additional profiles embedded in one artifact."""
    model = payload.get("model", {}) if isinstance(payload, dict) else {}
    default_shape = tuple(model.get("input_shape", ()))
    if len(default_shape) != 3:
        return ()
    default = input_resolution_key(default_shape[1], default_shape[2])
    profiles = payload.get("profiles", {})
    if not isinstance(profiles, dict):
        return ()
    return (default, *tuple(profiles))


def select_single_bin_profile(payload: dict, resolution=None) -> tuple[str, dict]:
    """Materialize the selected fixed-shape graph/program view.

    The returned dictionary shares logical tensors and the common parameter
    image with the outer artifact. Only its operation shapes and resident
    program table are resolution-specific.
    """
    default_shape = tuple(int(value) for value in payload["model"]["input_shape"])
    if resolution is None:
        height, width = default_shape[1:]
    else:
        height, width = parse_input_resolution(resolution)
    key = input_resolution_key(height, width)
    available = available_input_resolutions(payload)
    if key not in available:
        raise ValueError(
            f"resolution {key} is not embedded in this bin; available: "
            f"{', '.join(available)}")
    if key == available[0]:
        return key, payload

    profile = payload["profiles"][key]
    selected = dict(payload)
    selected["model"] = dict(payload["model"])
    selected["model"]["input_shape"] = list(profile["input_shape"])
    selected["graph_sha256"] = profile["graph_sha256"]
    selected["operations"] = profile["operations"]
    selected["head_outputs"] = profile["head_outputs"]
    selected["hardware"] = profile["hardware"]
    selected.pop("profiles", None)
    selected.pop("bundle_sha256", None)
    return key, selected


def _tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    raw = (value.view(torch.uint16).numpy().tobytes()
           if value.dtype == torch.bfloat16 else value.numpy().tobytes())
    return hashlib.sha256(raw).hexdigest()


def _weights_sha256(weights: dict) -> str:
    descriptor = []
    for name in sorted(weights):
        entry = weights[name]
        descriptor.append({
            "name": name,
            "precision": entry["precision"],
            "layout": entry["layout"],
            "codes_shape": entry["codes_shape"],
            "codes_packed_shape": list(entry["codes_packed"].shape),
            "codes_packed_sha256": _tensor_sha256(entry["codes_packed"]),
            "block_scales_shape": list(entry["block_scales"].shape),
            "block_scales_sha256": _tensor_sha256(entry["block_scales"]),
            "bias_shape": list(entry["bias"].shape),
            "bias_sha256": _tensor_sha256(entry["bias"]),
        })
    encoded = json.dumps(
        descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pack_codes(codes: torch.Tensor, precision: str) -> tuple[torch.Tensor, list[int]]:
    raw = codes.detach().cpu().to(torch.uint8).contiguous().view(-1)
    if precision == "if8":
        return raw, list(codes.shape)
    if precision != "if4":
        raise ValueError(f"unsupported convolution precision {precision!r}")
    if raw.numel() and int(raw.max()) > 15:
        raise ValueError("IF4 codes must fit one nibble")
    if raw.numel() & 1:
        raw = torch.cat((raw, torch.zeros(1, dtype=torch.uint8)))
    return (raw[0::2] | (raw[1::2] << 4)).contiguous(), list(codes.shape)


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


def _graph_descriptor(payload: dict) -> dict:
    model = payload["model"]
    return {
        "input_shape": model["input_shape"],
        "names": model["names"],
        "strides": model["strides"].tolist(),
        "anchors": model["anchors"].tolist(),
        "operations": payload["operations"],
        "head_outputs": payload["head_outputs"],
    }


def _graph_sha256(payload: dict) -> str:
    encoded = json.dumps(
        _graph_descriptor(payload), sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _profile_digest_descriptor(key: str, payload: dict) -> dict:
    hardware = payload["hardware"]
    return {
        "resolution": key,
        "input_shape": payload["model"]["input_shape"],
        "graph_sha256": payload["graph_sha256"],
        "params_bytes": int(hardware["params_image"].numel()),
        "params_sha256": hardware["params_sha256"],
        "program_bytes": int(hardware["program_image"].numel()),
        "program_sha256": hardware["program_sha256"],
        "dispatch_sha256": precompiled_manifest_sha256(hardware),
    }


def _bundle_sha256(payload: dict) -> str:
    """Bind every embedded shape, resident program, and dispatch manifest."""
    profiles = []
    for key in available_input_resolutions(payload):
        _selected_key, selected = select_single_bin_profile(payload, key)
        profiles.append(_profile_digest_descriptor(key, selected))
    descriptor = {
        "weights_sha256": _weights_sha256(payload["weights"]),
        "profiles": profiles,
    }
    encoded = json.dumps(
        descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _specialize_operations(operations: Sequence[dict], weights: dict,
                           input_shape: Sequence[int]) -> list[dict]:
    """Recompute graph tensor shapes without changing its operation topology."""
    specialized = copy.deepcopy(list(operations))
    shapes = {"input": tuple(int(value) for value in input_shape)}
    for operation in specialized:
        inputs = [shapes[key] for key in operation["inputs"]]
        kind = operation["op"]
        if kind == "conv":
            oc, _channels, kh, kw = (
                int(value) for value in weights[operation["name"]]["codes_shape"])
            effective_h = operation["dilation"] * (kh - 1) + 1
            effective_w = operation["dilation"] * (kw - 1) + 1
            output = (
                oc,
                (inputs[0][1] + 2 * operation["pad"] - effective_h)
                // operation["stride"] + 1,
                (inputs[0][2] + 2 * operation["pad"] - effective_w)
                // operation["stride"] + 1,
            )
        elif kind == "maxpool":
            channels, height, width = inputs[0]
            output = (
                channels,
                (height + 2 * operation["pad"] - operation["kernel"])
                // operation["stride"] + 1,
                (width + 2 * operation["pad"] - operation["kernel"])
                // operation["stride"] + 1,
            )
        elif kind == "upsample2x":
            channels, height, width = inputs[0]
            output = (channels, 2 * height, 2 * width)
        elif kind == "add":
            if inputs[0] != inputs[1]:
                raise ValueError(
                    f"{operation['name']}: residual shapes differ: "
                    f"{inputs[0]} vs {inputs[1]}")
            output = inputs[0]
        elif kind == "concat":
            if any(value[1:] != inputs[0][1:] for value in inputs[1:]):
                raise ValueError(
                    f"{operation['name']}: concat spatial shapes do not match")
            output = (sum(value[0] for value in inputs), *inputs[0][1:])
        else:
            raise ValueError(f"unsupported graph operation {kind!r}")
        if any(value <= 0 for value in output):
            raise ValueError(
                f"{operation['name']}: resolution produced invalid shape {output}")
        operation["output_shape"] = list(output)
        shapes[operation["output"]] = output
    return specialized


@dataclass(frozen=True)
class TensorSpec:
    key: str
    shape: tuple[int, int, int]

    def dim(self) -> int:
        return len(self.shape)


class ArtifactCompileBackend:
    """Flatten canonical YOLO modules into native primitive operations/data."""

    def __init__(self):
        self.operations: list[dict] = []
        self.weights: dict[str, dict] = {}

    @staticmethod
    def _output_hw(x: TensorSpec, kernel: int, stride: int,
                   pad: int, dilation: int) -> tuple[int, int]:
        _, height, width = x.shape
        effective = dilation * (kernel - 1) + 1
        return (
            (height + 2 * pad - effective) // stride + 1,
            (width + 2 * pad - effective) // stride + 1,
        )

    def conv_tensor(self, name: str, x: TensorSpec, conv: torch.nn.Conv2d,
                    bn: torch.nn.BatchNorm2d | None, *, activate: bool) -> TensorSpec:
        kh, kw = _pair(conv.kernel_size, "kernel_size")
        sh, sw = _pair(conv.stride, "stride")
        ph, pw = _pair(conv.padding, "padding")
        dh, dw = _pair(conv.dilation, "dilation")
        if kh != kw or sh != sw or ph != pw or dh != dw or conv.groups != 1:
            raise ValueError(
                f"{name}: single-bin path requires square, symmetric, dense Conv2d")
        if x.shape[0] != conv.in_channels:
            raise ValueError(
                f"{name}: input has {x.shape[0]} channels, expected {conv.in_channels}")

        prepared = quantize_conv_for_andromeda(conv, bn)
        precision = prepared.data_type.name.lower()
        layout = "gather" if prepared.gather else "channels"
        packed_codes, codes_shape = _pack_codes(prepared.codes, precision)
        self.weights[name] = {
            "precision": precision,
            "layout": layout,
            "codes_packed": packed_codes,
            "codes_shape": codes_shape,
            "block_scales": prepared.block_scales.to(torch.bfloat16).contiguous(),
            "bias": (torch.empty(0, dtype=torch.bfloat16)
                     if prepared.bias is None else prepared.bias.contiguous()),
        }
        out_h, out_w = self._output_hw(x, kh, sh, ph, dh)
        result = TensorSpec(name, (int(conv.out_channels), out_h, out_w))
        self.operations.append({
            "op": "conv",
            "name": name,
            "inputs": [x.key],
            "output": result.key,
            "output_shape": list(result.shape),
            "stride": sh,
            "pad": ph,
            "dilation": dh,
            "activate": bool(activate),
        })
        return result

    def maxpool(self, name: str, x: TensorSpec, *, kernel: int,
                stride: int, pad: int) -> TensorSpec:
        out_h, out_w = self._output_hw(x, kernel, stride, pad, 1)
        result = TensorSpec(name, (x.shape[0], out_h, out_w))
        self.operations.append({
            "op": "maxpool",
            "name": name,
            "inputs": [x.key],
            "output": result.key,
            "output_shape": list(result.shape),
            "kernel": int(kernel),
            "stride": int(stride),
            "pad": int(pad),
        })
        return result

    def upsample2x(self, name: str, x: TensorSpec) -> TensorSpec:
        result = TensorSpec(name, (x.shape[0], 2 * x.shape[1], 2 * x.shape[2]))
        self.operations.append({
            "op": "upsample2x",
            "name": name,
            "inputs": [x.key],
            "output": result.key,
            "output_shape": list(result.shape),
        })
        return result

    def add(self, name: str, x: TensorSpec, y: TensorSpec) -> TensorSpec:
        if x.shape != y.shape:
            raise ValueError(f"{name}: residual shapes differ: {x.shape} vs {y.shape}")
        result = TensorSpec(name, x.shape)
        self.operations.append({
            "op": "add",
            "name": name,
            "inputs": [x.key, y.key],
            "output": result.key,
            "output_shape": list(result.shape),
        })
        return result

    def concat(self, name: str, values: Sequence[TensorSpec]) -> TensorSpec:
        if not values:
            raise ValueError(f"{name}: empty concat")
        height, width = values[0].shape[1:]
        if any(value.shape[1:] != (height, width) for value in values):
            raise ValueError(f"{name}: concat spatial shapes do not match")
        result = TensorSpec(
            name, (sum(value.shape[0] for value in values), height, width))
        self.operations.append({
            "op": "concat",
            "name": name,
            "inputs": [value.key for value in values],
            "output": result.key,
            "output_shape": list(result.shape),
        })
        return result


def compile_single_bin(model: torch.nn.Module, output_path: Path, *,
                       image_size: int = 256,
                       input_resolutions: Sequence[str] | None = None,
                       checkpoint_sha256: str,
                       variant: str = "s") -> dict:
    """Compile canonical weights and several fixed shapes into one bin."""
    profile = get_yolov5_variant(variant)
    artifact_spec = get_canonical_artifact(profile.key)
    resolutions = tuple(input_resolutions or SUPPORTED_INPUT_RESOLUTIONS)
    if image_size != 256 or resolutions != SUPPORTED_INPUT_RESOLUTIONS:
        raise ValueError(
            f"canonical {profile.model_name} single-bin format requires default "
            f"image_size=256 and profiles {SUPPORTED_INPUT_RESOLUTIONS}")
    if str(checkpoint_sha256) != profile.checkpoint_sha256:
        raise ValueError(
            f"{profile.model_name} compile requested checkpoint SHA-256 "
            f"{checkpoint_sha256}, expected {profile.checkpoint_sha256}")
    backend = ArtifactCompileBackend()
    heads = execute_yolov5(
        model, TensorSpec("input", (3, image_size, image_size)), backend)
    detect = model.model[-1]
    payload = {
        "format": artifact_spec.format,
        "artifact_version": ARTIFACT_VERSION,
        "checkpoint_sha256": str(checkpoint_sha256),
        "model": {
            "name": profile.model_name,
            "upstream_version": "v7.0",
            "input_shape": [3, image_size, image_size],
            "names": _class_names(model),
            "strides": model.stride.detach().cpu().float().contiguous(),
            "anchors": detect.anchors.detach().cpu().float().contiguous(),
        },
        "operations": backend.operations,
        "head_outputs": [head.key for head in heads],
        "weights": backend.weights,
        "runtime": {
            "engine_abi": PRECOMPILED_ABI,
            "geometry_abi": GEOMETRY_ABI,
            "precision": "CHANNEL-IF4/GATHER-IF8-MIXED-BLOCK64",
            "compatible_fpga_hashes": sorted(
                int(value) for value in user_dma_core.UE_GATHER_IF8_HW_VERSIONS),
            "preprocessing": {
                "letterbox": True,
                "pad_value": 114,
                "scale": "0_to_1",
            },
            "postprocessing": {
                "confidence_threshold": 0.25,
                "iou_threshold": 0.45,
                "max_detections": 300,
            },
            "output_suffix": ("_detections_hw.jpg" if profile.key == "s"
                              else f"_yolov5{profile.key}_detections_hw.jpg"),
        },
        "profiles": {},
    }
    payload["graph_sha256"] = _graph_sha256(payload)
    payload["hardware"] = compile_precompiled_hardware(payload)
    shared_params = payload["hardware"]["params_image"]
    shared_params_sha256 = payload["hardware"]["params_sha256"]
    shared_params_bytes = shared_params.numel()
    for resolution in resolutions[1:]:
        height, width = parse_input_resolution(resolution)
        input_shape = [3, height, width]
        operations = _specialize_operations(
            payload["operations"], payload["weights"], input_shape)
        selected = {
            "model": dict(payload["model"], input_shape=input_shape),
            "operations": operations,
            "head_outputs": list(payload["head_outputs"]),
            "weights": payload["weights"],
        }
        selected["graph_sha256"] = _graph_sha256(selected)
        hardware = compile_precompiled_hardware(selected)
        if (hardware["params_sha256"] != shared_params_sha256
                or hardware["params_image"].numel() != shared_params_bytes
                or not torch.equal(hardware["params_image"], shared_params)):
            raise RuntimeError(
                f"{resolution}: static parameter image differs across profiles")
        # Reference one tensor object so torch.save stores the immutable image
        # once even though every selected hardware view remains self-contained.
        hardware["params_image"] = shared_params
        payload["profiles"][resolution] = {
            "input_shape": input_shape,
            "graph_sha256": selected["graph_sha256"],
            "operations": operations,
            "head_outputs": list(payload["head_outputs"]),
            "hardware": hardware,
        }
    payload["bundle_sha256"] = _bundle_sha256(payload)
    validate_single_bin(payload)

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(output_path),
        "size": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "operations": len(backend.operations),
        "convolutions": len(backend.weights),
        "static_params_bytes": payload["hardware"]["params_image"].numel(),
        "program_bytes": payload["hardware"]["program_image"].numel(),
        "profile_program_bytes": {
            key: selected["hardware"]["program_image"].numel()
            for key, selected in ((
                DEFAULT_INPUT_RESOLUTION, payload), *payload["profiles"].items())
        },
    }


def validate_single_bin(payload: dict) -> None:
    """Validate the closed artifact schema before any graph execution."""
    if not isinstance(payload, dict):
        raise RuntimeError("YOLO single-bin payload is not a dictionary")
    variant = artifact_variant(payload)
    profile = get_yolov5_variant(variant)
    artifact_spec = get_canonical_artifact(variant)
    if payload.get("artifact_version") != ARTIFACT_VERSION:
        raise RuntimeError(
            f"unsupported YOLO artifact version {payload.get('artifact_version')!r}")
    expected_keys = {
        "format", "artifact_version", "checkpoint_sha256", "graph_sha256",
        "model", "operations", "head_outputs", "weights", "runtime",
        "hardware", "profiles", "bundle_sha256",
    }
    if set(payload) != expected_keys:
        raise RuntimeError(
            f"YOLO artifact keys differ from the closed schema: {sorted(payload)}")
    if payload.get("checkpoint_sha256") != profile.checkpoint_sha256:
        raise RuntimeError(
            f"{profile.model_name} artifact checkpoint hash is not the pinned "
            "v7.0 release")
    if payload.get("graph_sha256") != _graph_sha256(payload):
        raise RuntimeError("YOLO artifact graph digest does not match its contents")
    if payload["graph_sha256"] != artifact_spec.graph_sha256:
        raise RuntimeError(
            f"YOLO artifact graph is not canonical {profile.model_name} v7.0")

    runtime = payload.get("runtime")
    expected_runtime = {
        "engine_abi": PRECOMPILED_ABI,
        "geometry_abi": GEOMETRY_ABI,
        "precision": "CHANNEL-IF4/GATHER-IF8-MIXED-BLOCK64",
        "compatible_fpga_hashes": sorted(
            int(value) for value in user_dma_core.UE_GATHER_IF8_HW_VERSIONS),
        "preprocessing": {
            "letterbox": True, "pad_value": 114, "scale": "0_to_1"},
        "postprocessing": {
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "max_detections": 300,
        },
        "output_suffix": ("_detections_hw.jpg" if variant == "s"
                          else f"_yolov5{variant}_detections_hw.jpg"),
    }
    if runtime != expected_runtime:
        raise RuntimeError(
            "YOLO artifact runtime ABI/defaults are not canonical; rebuild it "
            f"with `make yolov5{variant}_bin FORCE=1` after updating the driver")

    model = payload.get("model")
    operations = payload.get("operations")
    weights = payload.get("weights")
    heads = payload.get("head_outputs")
    if not isinstance(model, dict) or not isinstance(operations, list):
        raise RuntimeError("YOLO artifact is missing model/operations metadata")
    if not isinstance(weights, dict) or not isinstance(heads, list):
        raise RuntimeError("YOLO artifact is missing weights/head outputs")
    if tuple(model.get("input_shape", ())) != (3, 256, 256):
        raise RuntimeError(f"unexpected artifact input shape {model.get('input_shape')}")
    if set(model) != {"name", "upstream_version", "input_shape", "names",
                      "strides", "anchors"}:
        raise RuntimeError("YOLO artifact model metadata differs from the closed schema")
    if (model.get("name") != profile.model_name
            or model.get("upstream_version") != "v7.0"):
        raise RuntimeError(
            f"YOLO artifact model identity is not canonical {profile.model_name} v7.0")
    if len(model.get("names", ())) != 80:
        raise RuntimeError("YOLO artifact must contain 80 COCO class names")
    strides = model.get("strides")
    anchors = model.get("anchors")
    if not isinstance(strides, torch.Tensor) or tuple(strides.shape) != (3,):
        raise RuntimeError("YOLO artifact has invalid strides")
    if not isinstance(anchors, torch.Tensor) or tuple(anchors.shape) != (3, 3, 2):
        raise RuntimeError("YOLO artifact has invalid anchors")
    if not torch.equal(strides.float(), torch.tensor((8.0, 16.0, 32.0))):
        raise RuntimeError("YOLO artifact has noncanonical strides")
    expected_anchors = torch.tensor((
        ((10 / 8, 13 / 8), (16 / 8, 30 / 8), (33 / 8, 23 / 8)),
        ((30 / 16, 61 / 16), (62 / 16, 45 / 16), (59 / 16, 119 / 16)),
        ((116 / 32, 90 / 32), (156 / 32, 198 / 32), (373 / 32, 326 / 32)),
    ), dtype=torch.float32)
    if not torch.equal(anchors.float(), expected_anchors):
        raise RuntimeError("YOLO artifact has noncanonical anchors")
    if len(operations) != 85 or len(heads) != 3 or len(weights) != 60:
        raise RuntimeError(
            f"canonical {profile.model_name} needs 85 ops/3 heads/60 convs, got "
            f"{len(operations)}/{len(heads)}/{len(weights)}")

    produced = {"input"}
    shapes = {"input": (3, 256, 256)}
    conv_names = set()
    allowed = {"conv", "maxpool", "upsample2x", "add", "concat"}
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict) or operation.get("op") not in allowed:
            raise RuntimeError(f"invalid artifact operation {index}: {operation!r}")
        inputs = operation.get("inputs")
        output = operation.get("output")
        shape = operation.get("output_shape")
        if (not isinstance(inputs, list) or not inputs
                or any(key not in produced for key in inputs)):
            raise RuntimeError(f"operation {index} has unavailable inputs {inputs!r}")
        if not isinstance(output, str) or output in produced:
            raise RuntimeError(f"operation {index} has invalid output {output!r}")
        if (not isinstance(shape, list) or len(shape) != 3
                or any(not isinstance(value, int) or value <= 0 for value in shape)):
            raise RuntimeError(f"operation {index} has invalid shape {shape!r}")
        if operation["op"] == "conv":
            entry = weights.get(operation.get("name"))
            if not isinstance(entry, dict):
                raise RuntimeError(f"operation {index} has no convolution weights")
            if set(entry) != {
                    "precision", "layout", "codes_packed", "codes_shape",
                    "block_scales", "bias"}:
                raise RuntimeError(f"operation {index} convolution entry has unknown fields")
            precision = entry.get("precision")
            layout = entry.get("layout")
            packed = entry.get("codes_packed")
            codes_shape = entry.get("codes_shape")
            scales = entry.get("block_scales")
            bias = entry.get("bias")
            if (not isinstance(codes_shape, list) or len(codes_shape) != 4
                    or any(not isinstance(value, int) or value <= 0
                           for value in codes_shape)):
                raise RuntimeError(f"operation {index} has invalid code shape")
            code_count = 1
            for value in codes_shape:
                code_count *= value
            oc, channels, kh, kw = codes_shape
            if channels != shapes[inputs[0]][0]:
                raise RuntimeError(f"operation {index} weight/input channels disagree")
            ct = (channels + 63) // 64
            patch_taps = kh * kw * channels
            gather_blocks = (patch_taps + 63) // 64
            channel_blocks = kh * kw * ct
            expected_gather = gather_if8_is_profitable(
                channels=channels, out_channels=oc,
                kernel_h=kh, kernel_w=kw)
            expected_precision = "if8" if expected_gather else "if4"
            expected_layout = "gather" if expected_gather else "channels"
            if precision != expected_precision or layout != expected_layout:
                raise RuntimeError(
                    f"operation {index} uses {layout}/{precision}, expected "
                    f"{expected_layout}/{expected_precision}")
            expected_code_bytes = (code_count if precision == "if8"
                                   else (code_count + 1) // 2)
            if (not isinstance(packed, torch.Tensor) or packed.dtype != torch.uint8
                    or packed.dim() != 1
                    or packed.numel() != expected_code_bytes):
                raise RuntimeError(f"operation {index} has invalid {precision} codes")
            expected_scales = (oc, gather_blocks if expected_gather
                               else channel_blocks)
            if (not isinstance(scales, torch.Tensor)
                    or scales.dtype != torch.bfloat16
                    or tuple(scales.shape) != expected_scales):
                raise RuntimeError(f"operation {index} has invalid block scales")
            if (not torch.isfinite(scales.float()).all() or (scales == 0).any()):
                raise RuntimeError(f"operation {index} has nonfinite/zero block scales")
            if (not isinstance(bias, torch.Tensor) or bias.dtype != torch.bfloat16
                    or bias.dim() != 1 or bias.numel() != oc
                    or not torch.isfinite(bias.float()).all()):
                raise RuntimeError(f"operation {index} has invalid folded bias")
            in_h, in_w = shapes[inputs[0]][1:]
            effective_h = operation["dilation"] * (kh - 1) + 1
            effective_w = operation["dilation"] * (kw - 1) + 1
            expected_shape = (
                oc,
                (in_h + 2 * operation["pad"] - effective_h) // operation["stride"] + 1,
                (in_w + 2 * operation["pad"] - effective_w) // operation["stride"] + 1,
            )
            if tuple(shape) != expected_shape:
                raise RuntimeError(f"operation {index} output channels disagree with weights")
            conv_names.add(operation["name"])
        produced.add(output)
        shapes[output] = tuple(shape)
    if set(weights) != conv_names:
        raise RuntimeError("YOLO artifact contains missing or unused convolution weights")
    if _weights_sha256(weights) != artifact_spec.weights_sha256:
        raise RuntimeError("YOLO artifact logical quantized tensors are not canonical")
    if any(head not in produced for head in heads):
        raise RuntimeError("YOLO artifact head output is not produced by the graph")
    if [shapes[head] for head in heads] != [
            (255, 32, 32), (255, 16, 16), (255, 8, 8)]:
        raise RuntimeError("YOLO artifact detection head shapes are not canonical")
    validate_precompiled_hardware(payload)
    hardware = payload["hardware"]
    if (hardware["params_image"].numel() != artifact_spec.params_bytes
            or hardware["params_sha256"] != artifact_spec.params_sha256):
        raise RuntimeError("YOLO artifact prepacked parameter image is not canonical")
    if (hardware["program_image"].numel() != artifact_spec.program_bytes
            or hardware["program_sha256"] != artifact_spec.program_sha256):
        raise RuntimeError("YOLO artifact precompiled program image is not canonical")
    if precompiled_manifest_sha256(hardware) != artifact_spec.dispatch_sha256:
        raise RuntimeError("YOLO artifact precompiled dispatch manifest is not canonical")

    profiles = payload.get("profiles")
    expected_profile_keys = set(SUPPORTED_INPUT_RESOLUTIONS[1:])
    if not isinstance(profiles, dict) or set(profiles) != expected_profile_keys:
        raise RuntimeError(
            "YOLO artifact embedded-resolution set is not canonical: "
            f"expected {list(SUPPORTED_INPUT_RESOLUTIONS)}, got "
            f"{list(available_input_resolutions(payload))}")
    shared_params = hardware["params_image"]
    profile_keys = {
        "input_shape", "graph_sha256", "operations", "head_outputs", "hardware"}
    for key in SUPPORTED_INPUT_RESOLUTIONS[1:]:
        embedded = profiles[key]
        if not isinstance(embedded, dict) or set(embedded) != profile_keys:
            raise RuntimeError(f"YOLO profile {key} has unknown/missing fields")
        height, width = parse_input_resolution(key)
        input_shape = [3, height, width]
        if embedded["input_shape"] != input_shape:
            raise RuntimeError(f"YOLO profile {key} has the wrong input shape")
        expected_operations = _specialize_operations(
            operations, weights, input_shape)
        if embedded["operations"] != expected_operations:
            raise RuntimeError(
                f"YOLO profile {key} graph differs from the canonical topology")
        if embedded["head_outputs"] != heads:
            raise RuntimeError(f"YOLO profile {key} has noncanonical head outputs")
        _selected_key, selected = select_single_bin_profile(payload, key)
        if (embedded["graph_sha256"] != _graph_sha256(selected)
                or selected["graph_sha256"] != embedded["graph_sha256"]):
            raise RuntimeError(f"YOLO profile {key} graph digest does not match")
        selected_shapes = {
            operation["output"]: tuple(operation["output_shape"])
            for operation in embedded["operations"]}
        expected_heads = [
            (255, height // 8, width // 8),
            (255, height // 16, width // 16),
            (255, height // 32, width // 32),
        ]
        if [selected_shapes[name] for name in heads] != expected_heads:
            raise RuntimeError(f"YOLO profile {key} detection heads are invalid")
        validate_precompiled_hardware(selected)
        selected_hardware = embedded["hardware"]
        if (selected_hardware["params_sha256"] != artifact_spec.params_sha256
                or selected_hardware["params_image"].numel()
                != artifact_spec.params_bytes
                or not torch.equal(selected_hardware["params_image"], shared_params)):
            raise RuntimeError(
                f"YOLO profile {key} does not share the canonical parameter image")

    computed_bundle = _bundle_sha256(payload)
    if payload.get("bundle_sha256") != computed_bundle:
        raise RuntimeError("YOLO artifact profile-bundle digest does not match")
    if computed_bundle != artifact_spec.bundle_sha256:
        raise RuntimeError(
            f"YOLO artifact profile bundle is not canonical {profile.model_name} v7.0")


def load_single_bin(path: Path) -> dict:
    """Load one artifact with PyTorch's restricted tensor-only loader."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"YOLO single bin not found: {path}. Build it with yolov5_compile.py")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_single_bin(payload)
    return payload


def artifact_model_view(payload: dict):
    """Expose only the metadata interface needed by decode/NMS helpers."""
    meta = payload["model"]
    detect = types.SimpleNamespace(anchors=meta["anchors"])
    return types.SimpleNamespace(
        names=list(meta["names"]), stride=meta["strides"], model=[detect])


def _prepared_convs(payload: dict) -> dict[str, QuantizedConv]:
    result = {}
    for name, entry in payload["weights"].items():
        bias = entry["bias"]
        data_type = user_dma_core.TYPE[entry["precision"].upper()]
        result[name] = QuantizedConv(
            codes=_unpack_codes(
                entry["codes_packed"], entry["codes_shape"],
                entry["precision"]),
            block_scales=entry["block_scales"],
            bias=None if bias.numel() == 0 else bias,
            data_type=data_type,
            gather=entry["layout"] == "gather",
        )
    return result


def execute_single_bin(payload: dict, image_chw: torch.Tensor, backend, *,
                       progress: bool = False,
                       validate_payload: bool = True) -> list[torch.Tensor]:
    """Dispatch the embedded primitive graph directly from a loaded artifact."""
    if "profiles" in payload:
        if validate_payload:
            validate_single_bin(payload)
        _key, payload = select_single_bin_profile(
            payload, tuple(int(value) for value in image_chw.shape[1:]))
        validate_payload = False
    elif validate_payload:
        validate_single_bin(payload)
    expected_input = tuple(payload["model"]["input_shape"])
    if tuple(image_chw.shape) != expected_input:
        raise ValueError(
            f"artifact expects CHW input {expected_input}, got {tuple(image_chw.shape)}")

    precompiled = bool(getattr(backend, "uses_precompiled_hardware", False))
    prepared = {} if precompiled else _prepared_convs(payload)
    values: dict[str, torch.Tensor] = {"input": image_chw}
    remaining = collections.Counter(
        key for operation in payload["operations"] for key in operation["inputs"])
    remaining.update(payload["head_outputs"])

    for index, operation in enumerate(payload["operations"]):
        inputs = [values[key] for key in operation["inputs"]]
        kind = operation["op"]
        name = operation["name"]
        if kind == "conv":
            result = backend.conv_prepared(
                name, inputs[0], prepared.get(name),
                stride=operation["stride"], pad=operation["pad"],
                dilation=operation["dilation"],
                activate=operation["activate"])
        elif kind == "maxpool":
            result = backend.maxpool(
                name, inputs[0], kernel=operation["kernel"],
                stride=operation["stride"], pad=operation["pad"])
        elif kind == "upsample2x":
            result = backend.upsample2x(name, inputs[0])
        elif kind == "add":
            result = backend.add(name, inputs[0], inputs[1])
        elif kind == "concat":
            result = backend.concat(name, inputs)
        else:  # validate_single_bin closes this set.
            raise AssertionError(kind)

        expected = tuple(operation["output_shape"])
        if tuple(result.shape) != expected:
            raise RuntimeError(
                f"{name} produced {tuple(result.shape)}, artifact expects {expected}")
        values[operation["output"]] = result
        for key in operation["inputs"]:
            remaining[key] -= 1
            if remaining[key] == 0:
                values.pop(key, None)
        if progress:
            print(f"  [{index + 1:03d}/{len(payload['operations'])}] "
                  f"{name:<24} {kind:<10} -> {expected}")

    return [values[key] for key in payload["head_outputs"]]
