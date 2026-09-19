"""Pinned official native 8-kHz model, metadata and recurrent-state helpers."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

import numpy as np


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "dpdfnet8khz_config.json"
DEFAULT_MODEL_PATH = HERE / "dpdfnet8khz_bin" / "dpdfnet2_8khz.onnx"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_digest(path: Path, config: dict | None = None) -> str:
    config = load_config() if config is None else config
    actual = sha256(path)
    if actual != config["onnx_sha256"]:
        raise RuntimeError(
            f"DPDFNet 8-kHz ONNX SHA256 mismatch: expected "
            f"{config['onnx_sha256']}, got {actual}")
    return actual


def download_model(path: Path = DEFAULT_MODEL_PATH, *, force: bool = False) -> Path:
    """Verify an existing file or atomically download the pinned official model."""
    path = Path(path).expanduser().resolve()
    config = load_config()
    if path.exists() and not force:
        validate_digest(path, config)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".download",
                                       dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            with urllib.request.urlopen(config["onnx_url"], timeout=60) as response:
                for chunk in iter(lambda: response.read(1 << 20), b""):
                    stream.write(chunk)
        validate_digest(temporary, config)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def initial_state(metadata: dict[str, str]) -> np.ndarray:
    """Use the model's normalization values, then zero its remaining state."""
    try:
        size = int(metadata["state_size"])
        ne = int(metadata["erb_norm_state_size"])
        ns = int(metadata["spec_norm_state_size"])
        erb = np.array([float(item) for item in metadata["erb_norm_init"].split(",")],
                       dtype=np.float32)
        spec = np.array([float(item) for item in metadata["spec_norm_init"].split(",")],
                        dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("ONNX recurrent-state metadata is missing or malformed") from exc
    if (size <= 0 or ne <= 0 or ns <= 0 or ne + ns > size
            or erb.size != ne or spec.size != ns
            or not np.isfinite(erb).all() or not np.isfinite(spec).all()):
        raise RuntimeError("ONNX normalization metadata has inconsistent lengths or values")
    state = np.zeros(size, dtype=np.float32)
    state[:ne], state[ne:ne + ns] = erb, spec
    return state


def validate_metadata(metadata: dict[str, str]) -> None:
    config = load_config()
    expected = {
        "profile": "dpdfnet2_8khz", "sample_rate": str(config["sample_rate"]),
        "n_fft": str(config["window_length"]),
        "window_length": str(config["window_length"]),
        "hop_length": str(config["hop_length"]),
        "freq_bins": str(config["frequency_bins"]),
        "state_size": str(config["state_size"]),
        "erb_norm_state_size": "81", "spec_norm_state_size": "80",
        "window_type": "vorbis", "normalized": "0", "center": "1", "pad_mode": "reflect",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"unexpected 8-kHz ONNX metadata {key}={metadata.get(key)!r}; expected {value!r}")
    initial_state(metadata)


def inspect_model(path: Path = DEFAULT_MODEL_PATH) -> dict:
    """Check model identity, fixed streaming ABI and native audio metadata."""
    import onnx

    path = Path(path).expanduser().resolve()
    digest = validate_digest(path)
    model = onnx.load(str(path), load_external_data=False)
    inputs = {item.name: [int(dim.dim_value) for dim in item.type.tensor_type.shape.dim]
              for item in model.graph.input}
    outputs = {item.name: [int(dim.dim_value) for dim in item.type.tensor_type.shape.dim]
               for item in model.graph.output}
    config = load_config()
    if inputs != config["onnx_inputs"] or outputs != config["onnx_outputs"]:
        raise RuntimeError(f"unexpected 8-kHz streaming ABI: inputs={inputs}, outputs={outputs}")
    metadata = {item.key: item.value for item in model.metadata_props}
    validate_metadata(metadata)
    return {"model": config["model"], "path": str(path), "sha256": digest,
            "inputs": inputs, "outputs": outputs, "metadata": metadata,
            "nodes": len(model.graph.node),
            "operators": dict(sorted(Counter(node.op_type for node in model.graph.node).items()))}
