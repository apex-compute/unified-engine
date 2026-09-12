"""Trusted-source helpers and hardware-lowering audit for DPDFNet2."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import urllib.request

import numpy as np


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "dpdfnet_config.json"
DEFAULT_MODEL_PATH = HERE / "dpdfnet_bin" / "dpdfnet2.onnx"


def initial_state(metadata: dict[str, str]) -> np.ndarray:
    """Build the recurrent state from the pinned ONNX metadata."""
    required = (
        "state_size", "erb_norm_state_size", "spec_norm_state_size",
        "erb_norm_init", "spec_norm_init",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise RuntimeError(f"ONNX metadata is missing {missing}")
    state = np.zeros(int(metadata["state_size"]), dtype=np.float32)
    erb = np.fromstring(metadata["erb_norm_init"], sep=",", dtype=np.float32)
    spec = np.fromstring(metadata["spec_norm_init"], sep=",", dtype=np.float32)
    ne = int(metadata["erb_norm_state_size"])
    ns = int(metadata["spec_norm_state_size"])
    if erb.size != ne or spec.size != ns:
        raise RuntimeError("ONNX normalization metadata has inconsistent lengths")
    state[:ne] = erb
    state[ne:ne + ns] = spec
    return state


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_digest(path: Path, config: dict | None = None) -> str:
    config = config or load_config()
    actual = sha256(path)
    expected = config["onnx_sha256"]
    if actual != expected:
        raise RuntimeError(
            f"DPDFNet ONNX SHA256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def download_model(path: Path = DEFAULT_MODEL_PATH, *, force: bool = False) -> Path:
    """Atomically download the pinned official ONNX model."""
    config = load_config()
    path = Path(path).expanduser().resolve()
    if path.exists() and not force:
        validate_digest(path, config)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    try:
        urllib.request.urlretrieve(config["onnx_url"], temporary)
        validate_digest(temporary, config)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _onnx_module():
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "ONNX graph inspection requires `pip install -r "
            "models/dpdfnet/requirements.txt`"
        ) from exc
    return onnx


def _value_shape(value) -> list[int | str]:
    result = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        else:
            result.append(dim.dim_param or "?")
    return result


def _attributes(onnx, node) -> dict:
    return {
        item.name: onnx.helper.get_attribute_value(item)
        for item in node.attribute
    }


def inspect_model(path: Path = DEFAULT_MODEL_PATH) -> dict:
    """Validate the fixed streaming ABI and return a lowering-oriented report."""
    path = Path(path).expanduser().resolve()
    config = load_config()
    digest = validate_digest(path, config)
    onnx = _onnx_module()
    model = onnx.load(str(path), load_external_data=False)
    inputs = {item.name: _value_shape(item) for item in model.graph.input}
    outputs = {item.name: _value_shape(item) for item in model.graph.output}
    if inputs != config["onnx_inputs"] or outputs != config["onnx_outputs"]:
        raise RuntimeError(
            f"unexpected DPDFNet streaming ABI: inputs={inputs}, outputs={outputs}"
        )
    metadata = {item.key: item.value for item in model.metadata_props}
    expected_metadata = {
        "sample_rate": str(config["sample_rate"]),
        "window_length": str(config["window_length"]),
        "hop_length": str(config["hop_length"]),
        "freq_bins": str(config["frequency_bins"]),
        "state_size": str(config["state_size"]),
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"unexpected ONNX metadata {key}={metadata.get(key)!r}; "
                f"expected {expected!r}"
            )

    initializers = {item.name: list(item.dims) for item in model.graph.initializer}
    operators = Counter(node.op_type for node in model.graph.node)
    dense_conv = depthwise_conv = grouped_conv = 0
    conv_geometries = Counter()
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        attrs = _attributes(onnx, node)
        group = int(attrs.get("group", 1))
        weight_shape = initializers.get(node.input[1], [])
        kernel = tuple(int(value) for value in attrs.get("kernel_shape", ()))
        strides = tuple(int(value) for value in attrs.get("strides", (1, 1)))
        conv_geometries[(kernel, strides, group)] += 1
        if group == 1:
            dense_conv += 1
        elif len(weight_shape) == 4 and weight_shape[1] == 1 \
                and group == weight_shape[0]:
            depthwise_conv += 1
        else:
            grouped_conv += 1

    native_ops = {
        "Conv", "Gemm", "MatMul", "Add", "Mul", "Relu", "Sigmoid", "Tanh"
    }
    layout_ops = {
        "Reshape", "Transpose", "Slice", "Split", "Unsqueeze", "Concat",
        "Squeeze", "Flatten", "Gather",
    }
    composite_ops = {
        "GRU", "LayerNormalization", "ReduceSum", "Sub", "Div", "Pow",
        "Sqrt", "Log",
    }
    known = native_ops | layout_ops | composite_ops
    unsupported = sorted(set(operators) - known)
    return {
        "model": config["model"],
        "path": str(path),
        "sha256": digest,
        "opset": [{"domain": item.domain, "version": item.version}
                  for item in model.opset_import],
        "inputs": inputs,
        "outputs": outputs,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "operators": dict(sorted(operators.items())),
        "convolution": {
            "dense": dense_conv,
            "depthwise": depthwise_conv,
            "other_grouped": grouped_conv,
            "geometries": [
                {"kernel": list(kernel), "strides": list(strides),
                 "group": group, "count": count}
                for (kernel, strides, group), count in sorted(
                    conv_geometries.items(), key=lambda item: str(item[0]))
            ],
        },
        "lowering": {
            "native_candidates": sorted(set(operators) & native_ops),
            "layout_dma": sorted(set(operators) & layout_ops),
            "composite": sorted(set(operators) & composite_ops),
            "unknown": unsupported,
            "blockers": [
                "13 depthwise Conv nodes need the existing MobileSAM/Parakeet "
                "software lowering or a native depthwise RTL mode",
                "4 GRU nodes need stateful gate lowering based on the Parakeet LSTM path",
                "8 LayerNormalization nodes and recurrent state layout must be fused",
                "the first milestone keeps STFT/iSTFT on the host; they are outside this ONNX graph",
            ],
        },
    }
