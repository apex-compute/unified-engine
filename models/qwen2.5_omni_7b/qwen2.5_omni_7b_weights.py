#!/usr/bin/env python3
"""Streaming Qwen2.5-Omni-7B Thinker checkpoint converter.

The converter reads safetensors directly and writes the accelerator's sectioned
wire format without constructing Talker, Token2Wav, or even the complete
Thinker in host RAM.  ``params.bin`` contains four time-shared regions: LM,
vision, audio, and the BF16 attention-output matrices used only by decode.
Transformer matrices follow the configured mixed IF4/BF16 policy; norms,
biases, and FPGA media-front-end matrices remain BF16. The BF16 embedding
table stays on the host; only selected rows are transferred to accelerator
tensor DRAM.
"""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Callable

import torch


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import quant_lib


PRECISION = "if4"
BLOCK = 64
SCALE_BYTES = 2
DATA_BYTES = 32
WIRE_BYTES = SCALE_BYTES + DATA_BYTES
SCHEMA_VERSION = 10  # BF16 host embedding; vision attn.qk/attn.v remain compact
GENERATION_TAG_BYTES = 32

# Keep the host-only tokenizer/media preprocessing assets beside params.bin so
# a deployed runtime does not need the multi-gigabyte Hugging Face checkpoint.
# This is deliberately an allowlist: weight files and their index must never be
# copied into the stripped runtime bundle.
PROCESSOR_BUNDLE_SCHEMA_VERSION = 1
PROCESSOR_BUNDLE_MANIFEST = ".processor_bundle.json"
PROCESSOR_BUNDLE_FILES = (
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _load_config(script_dir: str) -> dict:
    with open(os.path.join(script_dir, "qwen2.5_omni_7b_config.json")) as f:
        return json.load(f)


# What params.bin's CONTENT actually depends on. This used to hash the whole
# config, which made the artifact's identity include `hardware` -- the DRAM map.
# Nothing in this module reads `hardware`: the map decides WHERE weights are
# loaded, never WHAT they are, so a map change was invalidating 5.9 GB of
# bit-identical weights and forcing a re-quantization that could not alter a
# single byte. The fingerprint is therefore an explicit allowlist: adding a
# section here is a deliberate statement that it changes the payload.
_PARAMS_IDENTITY_KEYS = ("file_info", "model", "precision", "paths",
                         "vision", "audio")

# INPUT GEOMETRY IS NOT WEIGHT IDENTITY. How big an image is, how many patches
# it makes and how many soft tokens they merge into are properties of the RUN,
# not of the tensors: the encoder is a transformer over a patch sequence and
# nothing in this module reads these. Only patch_embed.proj is patch-shaped and
# it is per-patch ([1280, 1216]), while position information is computed mRoPE,
# not a learned table sized by patch count. Leaving them in the fingerprint
# meant raising the image size invalidated 5.9 GB of bit-identical weights.
_VISION_RUNTIME_KEYS = ("image_size", "num_patches", "num_merged_tokens")


def _config_fingerprint(cfg: dict) -> str:
    """Identify the exact geometry, precision policy, and artifact paths.

    Deliberately EXCLUDES `hardware` and the vision input geometry: see
    :data:`_PARAMS_IDENTITY_KEYS` and :data:`_VISION_RUNTIME_KEYS`.
    """
    identity = {k: cfg[k] for k in _PARAMS_IDENTITY_KEYS if k in cfg}
    if "vision" in identity:
        identity["vision"] = {k: v for k, v in identity["vision"].items()
                              if k not in _VISION_RUNTIME_KEYS}
    canonical = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _checkpoint_revision(cfg: dict) -> str:
    """Return the required immutable Hugging Face checkpoint revision."""
    revision = cfg.get("paths", {}).get("hf_model_revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(char not in "0123456789abcdef" for char in revision)
    ):
        raise ValueError(
            "paths.hf_model_revision must be a lowercase, 40-character "
            "Hugging Face commit SHA"
        )
    return revision


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        while chunk := file_obj.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _processor_bundle_path(script_dir: str, cfg: dict) -> str:
    params_path = os.path.join(script_dir, cfg["paths"]["params"])
    return os.path.join(os.path.dirname(os.path.abspath(params_path)), "processor")


def _validate_processor_files(directory: str, cfg: dict) -> list[str]:
    """Validate files that materially define Omni preprocessing/token IDs."""
    errors: list[str] = []
    for name in PROCESSOR_BUNDLE_FILES:
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            errors.append(f"missing {name}")
        elif os.path.getsize(path) == 0:
            errors.append(f"empty {name}")
    if errors:
        return errors

    parsed: dict[str, dict] = {}
    for name in (
        "chat_template.json",
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
    ):
        try:
            with open(os.path.join(directory, name), encoding="utf-8") as file_obj:
                value = json.load(file_obj)
            if not isinstance(value, dict):
                raise ValueError("top level is not an object")
            parsed[name] = value
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid {name}: {exc}")
    if errors:
        return errors

    if parsed["config.json"].get("model_type") != "qwen2_5_omni":
        errors.append("config.json is not Qwen2.5-Omni")
    expected_processor = "Qwen2_5OmniProcessor"
    for name in ("preprocessor_config.json", "tokenizer_config.json"):
        if parsed[name].get("processor_class") != expected_processor:
            errors.append(f"{name} does not declare {expected_processor}")
    if not isinstance(parsed["chat_template.json"].get("chat_template"), str):
        errors.append("chat_template.json has no chat_template string")

    token_config = parsed["tokenizer_config.json"]
    decoder = token_config.get("added_tokens_decoder")
    if not isinstance(decoder, dict):
        errors.append("tokenizer_config.json has no added_tokens_decoder")
    else:
        expected_tokens = {
            "eos_token_id": "<|im_end|>",
            "audio_token_id": "<|AUDIO|>",
            "image_token_id": "<|IMAGE|>",
        }
        for config_key, content in expected_tokens.items():
            token_id = cfg["tokens"][config_key]
            entry = decoder.get(str(token_id))
            if not isinstance(entry, dict) or entry.get("content") != content:
                errors.append(
                    f"tokenizer token {token_id} does not match {content!r}"
                )
    return errors


def _processor_bundle_errors(directory: str, cfg: dict) -> list[str]:
    errors = _validate_processor_files(directory, cfg)
    manifest_path = os.path.join(directory, PROCESSOR_BUNDLE_MANIFEST)
    try:
        with open(manifest_path, encoding="utf-8") as file_obj:
            manifest = json.load(file_obj)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid {PROCESSOR_BUNDLE_MANIFEST}: {exc}")
        return errors

    if manifest.get("schema_version") != PROCESSOR_BUNDLE_SCHEMA_VERSION:
        errors.append("processor bundle schema differs")
    if manifest.get("model_repo") != cfg["paths"]["hf_model_repo"]:
        errors.append("processor bundle model repository differs")
    if manifest.get("model_revision") != _checkpoint_revision(cfg):
        errors.append("processor bundle model revision differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(PROCESSOR_BUNDLE_FILES):
        errors.append("processor bundle file manifest differs")
        return errors
    if errors:
        return errors
    for name in PROCESSOR_BUNDLE_FILES:
        path = os.path.join(directory, name)
        metadata = files[name]
        if not isinstance(metadata, dict):
            errors.append(f"invalid processor file metadata for {name}")
            continue
        if metadata.get("size") != os.path.getsize(path):
            errors.append(f"processor file size differs for {name}")
        elif metadata.get("sha256") != _sha256_file(path):
            errors.append(f"processor file digest differs for {name}")
    for root, _, filenames in os.walk(directory):
        for name in filenames:
            lower = name.lower()
            if lower.endswith(".safetensors") or lower.endswith(".safetensors.index.json"):
                errors.append(
                    f"weight artifact is forbidden in processor bundle: "
                    f"{os.path.relpath(os.path.join(root, name), directory)}"
                )
    return errors


def _atomic_copy_file(source: str, destination: str) -> None:
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.",
        suffix=".building",
        dir=os.path.dirname(destination),
    )
    try:
        with open(source, "rb") as src, os.fdopen(fd, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        shutil.copystat(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def ensure_processor_bundle(
    script_dir: str | None = None, verbose: bool = True
) -> str:
    """Return a validated, local-only AutoProcessor directory.

    Existing bundles are reused without inspecting checkpoint indexes or
    safetensors. Missing/corrupt bundles are rebuilt from the already-local HF
    metadata using atomic per-file replacements, with the manifest committed
    last.
    """
    script_dir = os.path.abspath(script_dir or _THIS_DIR)
    cfg = _load_config(script_dir)
    destination = _processor_bundle_path(script_dir, cfg)
    if os.path.isdir(destination) and not _processor_bundle_errors(destination, cfg):
        return destination

    source = os.path.join(script_dir, cfg["paths"]["hf_model_dir"])
    source_errors = _validate_processor_files(source, cfg)
    if source_errors:
        raise RuntimeError(
            "cannot build the Qwen2.5-Omni processor bundle from local files "
            f"in {source} ({'; '.join(source_errors)})"
        )
    os.makedirs(destination, exist_ok=True)
    file_manifest: dict[str, dict[str, object]] = {}
    for name in PROCESSOR_BUNDLE_FILES:
        src = os.path.join(source, name)
        dst = os.path.join(destination, name)
        _atomic_copy_file(src, dst)
        file_manifest[name] = {
            "size": os.path.getsize(dst),
            "sha256": _sha256_file(dst),
        }

    manifest = {
        "schema_version": PROCESSOR_BUNDLE_SCHEMA_VERSION,
        "model_repo": cfg["paths"]["hf_model_repo"],
        "model_revision": _checkpoint_revision(cfg),
        "files": file_manifest,
    }
    manifest_path = os.path.join(destination, PROCESSOR_BUNDLE_MANIFEST)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{PROCESSOR_BUNDLE_MANIFEST}.",
        suffix=".building",
        dir=destination,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file_obj:
            json.dump(manifest, file_obj, indent=2, sort_keys=True)
            file_obj.write("\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary, manifest_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)

    errors = _processor_bundle_errors(destination, cfg)
    if errors:
        raise RuntimeError(
            f"generated Qwen2.5-Omni processor bundle is invalid "
            f"({'; '.join(errors)})"
        )
    if verbose:
        total = sum(item["size"] for item in file_manifest.values())
        print(
            f"Bundled local Omni processor metadata: "
            f"{len(file_manifest)} files, {total / 2**20:.1f} MiB -> "
            f"{destination}"
        )
    return destination


def _decode_bf16_projections(cfg: dict) -> tuple[str, ...]:
    """Return the artifact's explicitly configured decode-only projections."""
    precision = cfg.get("precision", {})
    projections = precision.get("decode_bf16_projections")
    if projections != ["o"]:
        raise ValueError(
            "precision.decode_bf16_projections must be exactly ['o'] for "
            "the Qwen2.5-Omni decode_o artifact region"
        )
    if "o" not in precision.get("lm_quantized_projections", []):
        raise ValueError(
            "precision.lm_quantized_projections must retain 'o' so LM prefill "
            "uses its IF4 O projection"
        )
    return tuple(projections)


def _ensure_checkpoint(script_dir: str, cfg: dict) -> tuple[str, str]:
    """Fetch metadata and only checkpoint shards containing Thinker tensors."""
    from huggingface_hub import hf_hub_download, snapshot_download

    model_dir = os.path.join(script_dir, cfg["paths"]["hf_model_dir"])
    repo = cfg["paths"]["hf_model_repo"]
    revision = _checkpoint_revision(cfg)
    os.makedirs(model_dir, exist_ok=True)
    index_name = "model.safetensors.index.json"
    # Resolve the index at the pinned commit before deriving the shard allowlist.
    # Reading an arbitrary pre-existing local index could otherwise select files
    # belonging to a different revision.
    index_path = hf_hub_download(
        repo_id=repo,
        filename=index_name,
        local_dir=model_dir,
        revision=revision,
    )
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})
    thinker_keys = [key for key in weight_map if key.startswith("thinker.")]
    if not thinker_keys:
        raise RuntimeError(f"{index_path} contains no thinker.* tensors")
    shards = sorted({weight_map[key] for key in thinker_keys})
    metadata_patterns = [
        "*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken",
        "tokenizer*", "merges.txt", "vocab.json", "preprocessor_config.json",
    ]
    missing = [name for name in shards if not os.path.exists(os.path.join(model_dir, name))]
    if missing:
        print(
            f"Downloading {len(shards)} Thinker checkpoint shard(s) from {repo}; "
            "Talker/Token2Wav-only shards are skipped ..."
        )
    snapshot_download(
        repo_id=repo,
        local_dir=model_dir,
        allow_patterns=metadata_patterns + shards,
        revision=revision,
    )
    for name in shards:
        if not os.path.exists(os.path.join(model_dir, name)):
            raise FileNotFoundError(f"checkpoint shard was not downloaded: {name}")
    return model_dir, revision


class _Checkpoint:
    """Lazily mmap all required safetensor shards, with indexed lookup."""

    def __init__(self, model_dir: str):
        from safetensors import safe_open

        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            self.weight_map = json.load(f)["weight_map"]
        self._stack = ExitStack()
        self._handles = {}
        for filename in sorted({v for k, v in self.weight_map.items()
                                if k.startswith("thinker.")}):
            path = os.path.join(model_dir, filename)
            self._handles[filename] = self._stack.enter_context(
                safe_open(path, framework="pt", device="cpu")
            )

    def close(self) -> None:
        self._stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _handle(self, name: str):
        try:
            filename = self.weight_map[name]
        except KeyError:
            raise KeyError(f"checkpoint tensor {name!r} is missing") from None
        return self._handles[filename]

    def tensor(self, name: str) -> torch.Tensor:
        return self._handle(name).get_tensor(name)

    def slice(self, name: str):
        return self._handle(name).get_slice(name)

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(int(x) for x in self.slice(name).get_shape())


def _bf16_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().to(torch.bfloat16).contiguous()
    return value.view(torch.uint8).numpy().tobytes()


class _RegionWriter:
    """Write one region and track a relative tensor manifest."""

    def __init__(self, file_obj, name: str):
        self.file = file_obj
        self.name = name
        self.start = file_obj.tell()
        self.manifest: dict[str, dict] = {}

    def _record(self, key: str, start: int, size: int, shape) -> None:
        if key in self.manifest:
            raise KeyError(f"duplicate {self.name} output tensor {key!r}")
        self.manifest[key] = {
            "offset": start - self.start,
            "size": size,
            "shape": [int(x) for x in shape],
        }

    def add_bf16(self, key: str, tensor: torch.Tensor) -> None:
        start = self.file.tell()
        raw = _bf16_bytes(tensor)
        self.file.write(raw)
        self._record(key, start, len(raw), tensor.shape)

    def add_bf16_source(self, key: str, source, shape, rows_per_chunk: int = 1024) -> None:
        """Stream a large rank-2 source (notably the 1-GiB embedding)."""
        shape = tuple(int(x) for x in shape)
        if len(shape) != 2:
            raise ValueError(f"streamed BF16 source {key} must be rank 2, got {shape}")
        start = self.file.tell()
        for row in range(0, shape[0], rows_per_chunk):
            self.file.write(_bf16_bytes(source[row:row + rows_per_chunk, :]))
        size = self.file.tell() - start
        expected = math.prod(shape) * 2
        if size != expected:
            raise AssertionError(f"{key}: wrote {size} BF16 bytes, expected {expected}")
        self._record(key, start, size, shape)

    def add_if4_source(
        self,
        key: str,
        source: Callable[[int, int], torch.Tensor] | object,
        shape,
        *,
        rows_per_chunk: int | None = None,
    ) -> None:
        """Stream rank-2 rows into ``[all scales | all nibble data]``."""
        shape = tuple(int(x) for x in shape)
        if len(shape) != 2 or shape[1] % BLOCK:
            raise ValueError(f"IF4 source {key} must be [N,K] with K%64=0, got {shape}")
        n, k = shape
        if rows_per_chunk is None:
            rows_per_chunk = max(1, (1024 * 1024) // k)
        blocks = n * k // BLOCK
        scale_size, data_size = blocks * SCALE_BYTES, blocks * DATA_BYTES
        start = self.file.tell()
        total = scale_size + data_size
        if total:
            self.file.seek(start + total - 1)
            self.file.write(b"\0")
        scale_cursor = data_cursor = 0
        for row in range(0, n, rows_per_chunk):
            end = min(n, row + rows_per_chunk)
            chunk = source(row, end) if callable(source) else source[row:end, :]
            chunk = torch.as_tensor(chunk).to(torch.bfloat16).contiguous().reshape(end - row, k)
            data, scales = quant_lib.quantize(PRECISION, chunk, block_size=BLOCK)
            self.file.seek(start + scale_cursor)
            self.file.write(scales)
            self.file.seek(start + scale_size + data_cursor)
            self.file.write(data)
            scale_cursor += len(scales)
            data_cursor += len(data)
        if scale_cursor != scale_size or data_cursor != data_size:
            raise AssertionError(
                f"{key}: IF4 payload mismatch scales {scale_cursor}/{scale_size}, "
                f"data {data_cursor}/{data_size}")
        self.file.seek(start + total)
        self._record(f"{key}.{PRECISION}", start, total, shape)

    def add_quantized_embedding_source(
        self,
        key: str,
        source: Callable[[int, int], torch.Tensor] | object,
        shape,
        *,
        precision: str,
        scale_row_bytes: int = 128,
        rows_per_chunk: int | None = None,
    ) -> None:
        """Stream an embedding as ``[padded row scales | row-major data]``.

        A single 3584-wide IF4/IF8 row has 56 bf16 scales (112 bytes), aligned
        to neither supported 32- nor 64-byte AXI beat. Padding every scale row
        to 128 bytes gives both widths a common multiple; both the IF4 nibble
        row and the IF8 byte row are already beat-aligned.
        """
        if precision not in ("if4", "if8"):
            raise ValueError(
                f"embedding precision must be 'if4' or 'if8', got {precision!r}"
            )
        shape = tuple(int(x) for x in shape)
        if len(shape) != 2 or shape[1] % BLOCK:
            raise ValueError(
                f"{precision.upper()} embedding {key} must be [N,K] with "
                f"K%64=0, got {shape}"
            )
        n, k = shape
        compact_scale_row = (k // BLOCK) * SCALE_BYTES
        data_row = k // 2 if precision == "if4" else k
        if scale_row_bytes < compact_scale_row or scale_row_bytes % 64:
            raise ValueError(
                f"scale_row_bytes={scale_row_bytes} must be a 64-byte multiple "
                f"at least {compact_scale_row}"
            )
        if data_row % 64:
            raise ValueError(
                f"packed embedding row is {data_row} bytes, not AXI-beat aligned"
            )
        if rows_per_chunk is None:
            rows_per_chunk = max(1, (1024 * 1024) // k)

        scale_size, data_size = n * scale_row_bytes, n * data_row
        start = self.file.tell()
        total = scale_size + data_size
        if total:
            self.file.seek(start + total - 1)
            self.file.write(b"\0")
        scale_cursor = data_cursor = 0
        for row in range(0, n, rows_per_chunk):
            end = min(n, row + rows_per_chunk)
            rows = end - row
            chunk = source(row, end) if callable(source) else source[row:end, :]
            chunk = torch.as_tensor(chunk).to(torch.bfloat16).contiguous().reshape(rows, k)
            data, scales = quant_lib.quantize(
                precision, chunk, block_size=BLOCK
            )
            if len(scales) != rows * compact_scale_row or len(data) != rows * data_row:
                raise AssertionError(f"{key}: quantizer returned an unexpected row layout")
            padded_scales = bytearray(rows * scale_row_bytes)
            for local_row in range(rows):
                src = local_row * compact_scale_row
                dst = local_row * scale_row_bytes
                padded_scales[dst:dst + compact_scale_row] = scales[
                    src:src + compact_scale_row
                ]
            self.file.seek(start + scale_cursor)
            self.file.write(padded_scales)
            self.file.seek(start + scale_size + data_cursor)
            self.file.write(data)
            scale_cursor += len(padded_scales)
            data_cursor += len(data)
        if scale_cursor != scale_size or data_cursor != data_size:
            raise AssertionError(
                f"{key}: {precision.upper()} embedding payload mismatch scales "
                f"{scale_cursor}/{scale_size}, data {data_cursor}/{data_size}"
            )
        self.file.seek(start + total)
        manifest_key = f"{key}.{precision}"
        self._record(manifest_key, start, total, shape)
        self.manifest[manifest_key].update(
            {
                "layout": "padded_row_scales_then_data",
                "precision": precision,
                "block_size": BLOCK,
                "scale_values_per_row": k // BLOCK,
                "scale_row_bytes": scale_row_bytes,
                "data_row_bytes": data_row,
            }
        )

    def add_if4(self, key: str, tensor: torch.Tensor) -> None:
        logical_shape = tuple(int(x) for x in tensor.shape)
        flat = tensor.detach().cpu().to(torch.bfloat16).contiguous().flatten()
        if flat.numel() % BLOCK:
            flat = torch.nn.functional.pad(flat, (0, BLOCK - flat.numel() % BLOCK))
        matrix = flat.reshape(-1, BLOCK)
        self.add_if4_source(key, matrix, matrix.shape)
        self.manifest[f"{key}.{PRECISION}"]["shape"] = list(logical_shape)

    def finish(self) -> dict:
        return {
            "offset": self.start,
            "size": self.file.tell() - self.start,
            "manifest": self.manifest,
        }


def _expect(reader: _Checkpoint, name: str, shape) -> None:
    got = reader.shape(name)
    want = tuple(shape)
    if got != want:
        raise ValueError(f"{name} has shape {got}, expected {want}")


def _write_lm(out, reader: _Checkpoint, cfg: dict) -> dict:
    w = _RegionWriter(out, "lm")
    fi = cfg["file_info"]
    h, kv, group = fi["hidden_size"], fi["num_kv_heads"], fi["group_size"]
    q, mlp = kv * group * fi["actual_head_dim"], fi["mlp_elements"]
    layers, vocab = fi["num_layers"], fi["embedding_vocab"]
    projections = {
        "q": ("self_attn.q_proj", (q, h)),
        "k": ("self_attn.k_proj", (kv * fi["actual_head_dim"], h)),
        "v": ("self_attn.v_proj", (kv * fi["actual_head_dim"], h)),
        "o": ("self_attn.o_proj", (h, q)),
        "gate": ("mlp.gate_proj", (mlp, h)),
        "up": ("mlp.up_proj", (mlp, h)),
        "down": ("mlp.down_proj", (h, mlp)),
    }
    quantized = set(cfg["precision"]["lm_quantized_projections"])
    for li in range(layers):
        src = f"thinker.model.layers.{li}"
        dst = f"language_model.layers.{li}"
        for tag, (component, shape) in projections.items():
            name = f"{src}.{component}.weight"
            _expect(reader, name, shape)
            view = reader.slice(name)
            if tag in quantized:
                w.add_if4_source(f"{dst}.{component}.weight", view, shape)
            else:
                w.add_bf16_source(f"{dst}.{component}.weight", view, shape)
        for component, width in (
            ("self_attn.q_proj", q),
            ("self_attn.k_proj", kv * fi["actual_head_dim"]),
            ("self_attn.v_proj", kv * fi["actual_head_dim"]),
        ):
            value = reader.tensor(f"{src}.{component}.bias")
            _expect(reader, f"{src}.{component}.bias", (width,))
            w.add_bf16(f"{dst}.{component}.bias", value)
        for norm in ("input_layernorm", "post_attention_layernorm"):
            value = reader.tensor(f"{src}.{norm}.weight")
            _expect(reader, f"{src}.{norm}.weight", (h,))
            w.add_bf16(f"{dst}.{norm}.weight", value)
        if (li + 1) % 4 == 0 or li + 1 == layers:
            print(f"  LM layer {li + 1}/{layers}")

    norm_name = "thinker.model.norm.weight"
    _expect(reader, norm_name, (h,))
    w.add_bf16("language_model.norm.weight", reader.tensor(norm_name))
    head_name = "thinker.lm_head.weight"
    _expect(reader, head_name, (vocab, h))
    head = reader.slice(head_name)
    w.add_if4_source("lm_head.weight", lambda a, b: head[a:b, :], (vocab, h))
    embed_name = "thinker.model.embed_tokens.weight"
    _expect(reader, embed_name, (vocab, h))
    if cfg["precision"].get("embedding") != "bf16":
        raise ValueError("Qwen2.5-Omni requires a host BF16 embedding")
    w.add_bf16_source(
        "language_model.embed_tokens.weight",
        reader.slice(embed_name), (vocab, h),
    )
    return w.finish()


def _write_vision(out, reader: _Checkpoint, cfg: dict) -> dict:
    """Q/K/V are stored fully COMPACT -- ``attn.qk`` ([2h, h], Q's real
    1280 rows then K's), ``attn.v`` ([h, h]) -- no padding baked into the
    weight at all, for either projection.

    Replaces the older ``qk_padded``/``v_padded`` layout (16 heads x 80 real
    dims padded all the way to 128 each, entangled with the RoPE
    rotate-half swap-matrix layout -- see gemma4_e2b_vision.py's history for
    why that padding existed at all: the per-head 64-alignment the
    attention/transpose kernels genuinely need). The runtime
    (qwen2.5_omni_7b_vision.py's qkv_proj emission) now inserts ALL of that
    padding itself, with NO extra data movement: one small matmul per head
    (V) or per RoPE head-half (QK) writes its real-width output directly to
    its final padded position via the matmul's own gpr_out_row_stride_reg,
    addressing its weight rows by pure offset arithmetic into this same
    compact blob (IF4 blocks are along K, not N, so any N-row slice needs no
    re-quantization). No scatter DMA, no intermediate compact buffer -- the
    padding is inserted natively, by the same op that already computes the
    projection, so there is no separate cost for it at all.

    vision_weight_init() detects which key is present and falls back to the
    old padded layout if this one is absent, so an old, not-yet-reconverted
    params.bin for this model keeps working during the transition.
    """
    w = _RegionWriter(out, "vision")
    v = cfg["vision"]
    h, heads, hd = v["hidden_size"], v["num_heads"], v["head_dim"]
    vi, vi_pad, layers = v["intermediate_size"], 3456, v["depth"]
    for li in range(layers):
        src = f"thinker.visual.blocks.{li}"
        dst = f"visual.blocks.{li}"
        qn, qbn = f"{src}.attn.q.weight", f"{src}.attn.q.bias"
        kn, kbn = f"{src}.attn.k.weight", f"{src}.attn.k.bias"
        _expect(reader, qn, (h, h)); _expect(reader, qbn, (h,))
        _expect(reader, kn, (h, h)); _expect(reader, kbn, (h,))
        qk_w = torch.cat([reader.tensor(qn), reader.tensor(kn)], dim=0)
        qk_b = torch.cat([reader.tensor(qbn), reader.tensor(kbn)], dim=0)
        w.add_if4(f"{dst}.attn.qk.weight", qk_w)
        w.add_bf16(f"{dst}.attn.qk.bias", qk_b)

        vn, vbn = f"{src}.attn.v.weight", f"{src}.attn.v.bias"
        _expect(reader, vn, (h, h)); _expect(reader, vbn, (h,))
        w.add_if4(f"{dst}.attn.v.weight", reader.tensor(vn))
        w.add_bf16(f"{dst}.attn.v.bias", reader.tensor(vbn))

        on, obn = f"{src}.attn.proj.weight", f"{src}.attn.proj.bias"
        _expect(reader, on, (h, h)); _expect(reader, obn, (h,))
        w.add_if4(f"{dst}.attn.proj.weight", reader.tensor(on))
        w.add_bf16(f"{dst}.attn.proj.bias", reader.tensor(obn))

        for tag in ("gate", "up"):
            wn, bn = f"{src}.mlp.{tag}_proj.weight", f"{src}.mlp.{tag}_proj.bias"
            _expect(reader, wn, (vi, h)); _expect(reader, bn, (vi,))
            padded = torch.zeros(vi_pad, h, dtype=torch.bfloat16)
            padded[:vi] = reader.tensor(wn)
            bias = torch.zeros(vi_pad, dtype=torch.bfloat16)
            bias[:vi] = reader.tensor(bn)
            w.add_if4(f"{dst}.mlp.{tag}_proj.weight", padded)
            w.add_bf16(f"{dst}.mlp.{tag}_proj.bias", bias)
        wn, bn = f"{src}.mlp.down_proj.weight", f"{src}.mlp.down_proj.bias"
        _expect(reader, wn, (h, vi)); _expect(reader, bn, (h,))
        down = torch.zeros(h, vi_pad, dtype=torch.bfloat16)
        down[:, :vi] = reader.tensor(wn)
        w.add_if4(f"{dst}.mlp.down_proj.weight", down)
        w.add_bf16(f"{dst}.mlp.down_proj.bias", reader.tensor(bn))
        for norm in ("norm1", "norm2"):
            name = f"{src}.{norm}.weight"
            _expect(reader, name, (h,))
            w.add_bf16(f"{dst}.{norm}.weight", reader.tensor(name))
        if (li + 1) % 4 == 0 or li + 1 == layers:
            print(f"  vision layer {li + 1}/{layers}")

    patch_name = "thinker.visual.patch_embed.proj.weight"
    patch_shape = (h, 3, v["temporal_patch_size"], v["patch_size"], v["patch_size"])
    _expect(reader, patch_name, patch_shape)
    patch = reader.tensor(patch_name).reshape(h, -1)
    patch_k_padded = ((patch.shape[1] + BLOCK - 1) // BLOCK) * BLOCK
    patch_padded = torch.zeros(h, patch_k_padded, dtype=torch.bfloat16)
    patch_padded[:, : patch.shape[1]] = patch
    w.add_if4("visual.patch_embed.proj.weight", patch_padded)
    # Keep the first learned image operation in BF16 on the device. Its extra
    # ~3 MiB is negligible inside the phase-shared params window.
    w.add_bf16("visual.patch_embed.proj.weight.bf16", patch_padded)

    ln = "thinker.visual.merger.ln_q.weight"
    _expect(reader, ln, (h,))
    w.add_bf16("visual.merger.ln_q.weight", reader.tensor(ln))
    merge = h * v["spatial_merge_size"] ** 2
    for idx, shape in ((0, (merge, merge)), (2, (v["out_hidden_size"], merge))):
        wn, bn = (f"thinker.visual.merger.mlp.{idx}.weight",
                  f"thinker.visual.merger.mlp.{idx}.bias")
        _expect(reader, wn, shape); _expect(reader, bn, (shape[0],))
        w.add_if4(f"visual.merger.mlp.{idx}.weight", reader.tensor(wn))
        w.add_bf16(f"visual.merger.mlp.{idx}.bias", reader.tensor(bn))
    return w.finish()


def _write_audio(out, reader: _Checkpoint, cfg: dict) -> dict:
    w = _RegionWriter(out, "audio")
    a = cfg["audio"]
    h, mel, ff = a["hidden_size"], a["num_mel_bins"], a["intermediate_size"]
    layers, out_h = a["depth"], a["out_hidden_size"]
    for name, shape in (
        ("conv1.weight", (h, mel, 3)), ("conv1.bias", (h,)),
        ("conv2.weight", (h, h, 3)), ("conv2.bias", (h,)),
    ):
        src = f"thinker.audio_tower.{name}"
        _expect(reader, src, shape)
        value = reader.tensor(src)
        if name.endswith("weight"):
            # FPGA im2col rows are kernel-major: [left channels | center
            # channels | right channels]. Match that layout once at export.
            value = value.permute(0, 2, 1).reshape(shape[0], -1).contiguous()
        w.add_bf16(f"audio.{name}", value)
    # Generate the released tower's fixed sinusoidal table once so runtime
    # performs no host trigonometry or other model arithmetic.
    position_count = int(a.get("n_window", 100))
    half = h // 2
    theta = float(a.get("position_theta", 10000.0))
    log_step = math.log(theta) / (half - 1)
    inv_timescales = torch.exp(
        -log_step * torch.arange(half, dtype=torch.float32)
    )
    positions = torch.arange(position_count, dtype=torch.float32).unsqueeze(1)
    scaled = positions * inv_timescales.unsqueeze(0)
    positional = torch.cat((torch.sin(scaled), torch.cos(scaled)), dim=1)
    w.add_bf16("audio.positional_embedding", positional)
    for li in range(layers):
        src = f"thinker.audio_tower.layers.{li}"
        dst = f"audio.layers.{li}"
        for out_name, in_name in (
            ("ln1", "self_attn_layer_norm"), ("ln2", "final_layer_norm")
        ):
            for suffix in ("weight", "bias"):
                name = f"{src}.{in_name}.{suffix}"
                _expect(reader, name, (h,))
                w.add_bf16(f"{dst}.{out_name}.{suffix}", reader.tensor(name))
        for out_name, in_name in (("q", "q_proj"), ("k", "k_proj"),
                                  ("v", "v_proj"), ("o", "out_proj")):
            name = f"{src}.self_attn.{in_name}.weight"
            _expect(reader, name, (h, h))
            w.add_if4(f"{dst}.{out_name}.weight", reader.tensor(name))
            if out_name != "k":
                bias = f"{src}.self_attn.{in_name}.bias"
                _expect(reader, bias, (h,))
                w.add_bf16(f"{dst}.{out_name}.bias", reader.tensor(bias))
        for fc, shape in (("fc1", (ff, h)), ("fc2", (h, ff))):
            name = f"{src}.{fc}.weight"
            _expect(reader, name, shape)
            w.add_if4(f"{dst}.{fc}.weight", reader.tensor(name))
            bias = f"{src}.{fc}.bias"
            _expect(reader, bias, (shape[0],))
            w.add_bf16(f"{dst}.{fc}.bias", reader.tensor(bias))
        if (li + 1) % 4 == 0 or li + 1 == layers:
            print(f"  audio layer {li + 1}/{layers}")
    for name, shape, quant in (
        ("ln_post.weight", (h,), False), ("ln_post.bias", (h,), False),
        ("proj.weight", (out_h, h), True), ("proj.bias", (out_h,), False),
    ):
        src = f"thinker.audio_tower.{name}"
        _expect(reader, src, shape)
        if quant:
            w.add_if4(f"audio.{name}", reader.tensor(src))
        else:
            w.add_bf16(f"audio.{name}", reader.tensor(src))
    return w.finish()


def _write_decode_o(out, reader: _Checkpoint, cfg: dict) -> dict:
    """Stream decode-only attention output projections as raw BF16."""
    _decode_bf16_projections(cfg)
    w = _RegionWriter(out, "decode_o")
    fi = cfg["file_info"]
    hidden = int(fi["hidden_size"])
    layers = int(fi["num_layers"])
    q_width = (
        int(fi["num_kv_heads"])
        * int(fi["group_size"])
        * int(fi["actual_head_dim"])
    )
    if q_width != hidden:
        raise ValueError(
            f"decode_o requires square [{hidden},{hidden}] O projections, "
            f"but configured Q width is {q_width}"
        )
    shape = (hidden, hidden)
    for li in range(layers):
        src = f"thinker.model.layers.{li}.self_attn.o_proj.weight"
        dst = f"language_model.layers.{li}.self_attn.o_proj.weight"
        _expect(reader, src, shape)
        w.add_bf16_source(dst, reader.slice(src), shape)
        if (li + 1) % 4 == 0 or li + 1 == layers:
            print(f"  decode BF16 O layer {li + 1}/{layers}")
    return w.finish()


def _validate_decode_o_region(region: dict, cfg: dict) -> list[str]:
    """Validate the exact decode_o manifest implied by the model config."""
    errors: list[str] = []
    try:
        _decode_bf16_projections(cfg)
    except ValueError as exc:
        return [str(exc)]

    fi = cfg["file_info"]
    hidden = int(fi["hidden_size"])
    layers = int(fi["num_layers"])
    layer_bytes = hidden * hidden * 2
    expected_size = layers * layer_bytes
    if not isinstance(region, dict):
        return ["decode_o region metadata is invalid"]
    if region.get("size") != expected_size:
        errors.append(
            f"decode_o size {region.get('size')!r} != {expected_size}"
        )

    manifest = region.get("manifest")
    if not isinstance(manifest, dict):
        errors.append("decode_o tensor manifest is invalid")
        return errors
    expected_keys = {
        f"language_model.layers.{li}.self_attn.o_proj.weight"
        for li in range(layers)
    }
    if set(manifest) != expected_keys:
        errors.append("decode_o tensor set differs from configured layers")
        return errors
    for li in range(layers):
        key = f"language_model.layers.{li}.self_attn.o_proj.weight"
        expected = {
            "offset": li * layer_bytes,
            "size": layer_bytes,
            "shape": [hidden, hidden],
        }
        if manifest[key] != expected:
            errors.append(f"decode_o tensor metadata differs for layer {li}")
    return errors


def _validate_embedding_region(region: dict, cfg: dict) -> list[str]:
    """Validate the host BF16 embedding in the LM region."""
    errors: list[str] = []
    if not isinstance(region, dict):
        return ["LM region metadata is invalid"]

    fi = cfg["file_info"]
    vocab = int(fi["embedding_vocab"])
    hidden = int(fi["hidden_size"])
    precision = cfg.get("precision", {}).get("embedding")
    if precision != "bf16":
        return [f"schema-{SCHEMA_VERSION} embedding precision must be 'bf16'"]

    manifest = region.get("manifest")
    if not isinstance(manifest, dict):
        return ["LM tensor manifest is invalid"]
    base_key = "language_model.embed_tokens.weight"
    expected_key = base_key
    embedding_keys = {
        key for key in manifest
        if key == base_key or key.startswith(f"{base_key}.")
    }
    if embedding_keys != {expected_key}:
        errors.append("LM embedding tensor set differs from host BF16 layout")
        return errors

    section = manifest[expected_key]
    if not isinstance(section, dict):
        return ["LM embedding tensor metadata is invalid"]

    section_size = vocab * hidden * 2
    expected_fields = {
        "size": section_size,
        "shape": [vocab, hidden],
    }
    if set(section) != {"offset", *expected_fields}:
        errors.append("LM embedding has unexpected metadata fields")
    for field, expected in expected_fields.items():
        if section.get(field) != expected:
            errors.append(
                f"LM embedding {field} {section.get(field)!r} != {expected!r}"
            )

    offset = section.get("offset")
    region_size = region.get("size")
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        errors.append("LM embedding offset is invalid")
    elif (
        isinstance(region_size, int)
        and not isinstance(region_size, bool)
        and region_size >= 0
        and offset + section_size != region_size
    ):
        errors.append("LM embedding is not the final, complete LM region section")
    return errors


def weight_bin_generate(script_dir: str | None = None, output_params: str | None = None) -> str:
    script_dir = script_dir or _THIS_DIR
    cfg = _load_config(script_dir)
    _decode_bf16_projections(cfg)
    params_path = output_params or os.path.join(script_dir, cfg["paths"]["params"])
    output_dir = os.path.dirname(os.path.abspath(params_path))
    os.makedirs(output_dir, exist_ok=True)
    model_dir, revision = _ensure_checkpoint(script_dir, cfg)
    tmp_path = params_path + ".building"
    json_tmp = None
    try:
        with _Checkpoint(model_dir) as reader, open(tmp_path, "wb") as out:
            print("Converting Thinker LM (streaming mixed BF16/IF4) ...")
            lm = _write_lm(out, reader, cfg)
            print("Converting vision tower ...")
            vision = _write_vision(out, reader, cfg)
            print("Converting audio tower ...")
            audio = _write_audio(out, reader, cfg)
            print("Converting decode-only BF16 O projections ...")
            decode_o = _write_decode_o(out, reader, cfg)
            # Bind the payload to its sidecar. The two files cannot be renamed
            # atomically together, so a random trailer lets the loader detect
            # a crash between the two replacements instead of trusting stale
            # offsets from another multi-gigabyte conversion.
            generation_tag = os.urandom(GENERATION_TAG_BYTES)
            out.write(generation_tag)
            out.flush()
            os.fsync(out.fileno())
        regions = {
            "lm": lm,
            "vision": vision,
            "audio": audio,
            "decode_o": decode_o,
        }
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "model_repo": cfg["paths"]["hf_model_repo"],
            "model_revision": revision,
            "precision": cfg["precision"],
            "config_sha256": _config_fingerprint(cfg),
            "generation_id": generation_tag.hex(),
            "generation_trailer_bytes": GENERATION_TAG_BYTES,
            "params_size": os.path.getsize(tmp_path),
            "regions": regions,
        }
        json_path = params_path.rsplit(".", 1)[0] + ".json"
        json_tmp = json_path + ".building"
        with open(json_tmp, "w") as f:
            json.dump(metadata, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, params_path)
        os.replace(json_tmp, json_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if json_tmp is not None and os.path.exists(json_tmp):
            os.remove(json_tmp)
    print(
        "Unified Thinker params: "
        + ", ".join(f"{name}={region['size']/2**20:.1f} MiB"
                    for name, region in regions.items())
        + f" -> {params_path}"
    )
    ensure_processor_bundle(script_dir, verbose=True)
    return params_path


def ensure_params_bin(script_dir: str, verbose: bool = True) -> str:
    cfg = _load_config(script_dir)
    _decode_bf16_projections(cfg)
    params_path = os.path.join(script_dir, cfg["paths"]["params"])
    json_path = params_path.rsplit(".", 1)[0] + ".json"
    if os.path.exists(params_path) and os.path.exists(json_path):
        with open(json_path) as f:
            metadata = json.load(f)
        errors = []
        if metadata.get("schema_version") != SCHEMA_VERSION:
            errors.append(
                f"schema {metadata.get('schema_version')!r} != {SCHEMA_VERSION}"
            )
        if metadata.get("model_repo") != cfg["paths"]["hf_model_repo"]:
            errors.append("model repository differs")
        expected_revision = _checkpoint_revision(cfg)
        if metadata.get("model_revision") != expected_revision:
            errors.append(
                f"model revision {metadata.get('model_revision')!r} != "
                f"{expected_revision}"
            )
        if metadata.get("precision") != cfg["precision"]:
            errors.append("precision policy differs")
        if metadata.get("config_sha256") != _config_fingerprint(cfg):
            errors.append("model config differs")
        regions = metadata.get("regions") or {}
        if not isinstance(regions, dict):
            errors.append("regions metadata is invalid")
            regions = {}
        required_region_names = {"lm", "vision", "audio", "decode_o"}
        if set(regions) != required_region_names:
            errors.append("required lm/vision/audio/decode_o regions differ")

        actual_size = os.path.getsize(params_path)
        if metadata.get("params_size") != actual_size:
            errors.append(
                f"payload size {actual_size} != manifest "
                f"{metadata.get('params_size')!r}"
            )
        if set(regions) == required_region_names:
            cursor = 0
            for name in ("lm", "vision", "audio", "decode_o"):
                region = regions[name]
                if not isinstance(region, dict):
                    errors.append(f"{name} region metadata is invalid")
                    continue
                offset, size = region.get("offset"), region.get("size")
                if offset != cursor:
                    errors.append(
                        f"{name} region offset {offset!r} != expected {cursor}"
                    )
                if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                    errors.append(f"{name} region size is invalid")
                    continue
                cursor += size
            if cursor + GENERATION_TAG_BYTES != actual_size:
                errors.append(
                    f"region payload end {cursor} plus trailer does not match "
                    f"file size {actual_size}"
                )
            errors.extend(_validate_embedding_region(regions["lm"], cfg))
            errors.extend(_validate_decode_o_region(regions["decode_o"], cfg))
        generation_hex = metadata.get("generation_id")
        trailer_bytes = metadata.get("generation_trailer_bytes")
        try:
            expected_tag = bytes.fromhex(generation_hex)
        except (TypeError, ValueError):
            expected_tag = b""
        if trailer_bytes != GENERATION_TAG_BYTES or len(expected_tag) != GENERATION_TAG_BYTES:
            errors.append("generation tag metadata is invalid")
        elif actual_size < GENERATION_TAG_BYTES:
            errors.append("payload is shorter than its generation tag")
        else:
            with open(params_path, "rb") as f:
                f.seek(-GENERATION_TAG_BYTES, os.SEEK_END)
                actual_tag = f.read(GENERATION_TAG_BYTES)
            if actual_tag != expected_tag:
                errors.append("payload and manifest are from different generations")

        if not errors:
            ensure_processor_bundle(script_dir, verbose=verbose)
            return params_path
        raise RuntimeError(
            f"cached Qwen2.5-Omni params are incompatible ({'; '.join(errors)}). "
            f"Remove {params_path} and {json_path}, then regenerate schema "
            f"{SCHEMA_VERSION}."
        )
    if os.path.exists(params_path) != os.path.exists(json_path):
        remaining = params_path if os.path.exists(params_path) else json_path
        raise RuntimeError(
            "cached Qwen2.5-Omni params are incomplete; remove the remaining "
            f"artifact ({remaining}) and regenerate"
        )
    if verbose:
        print("Qwen2.5-Omni Thinker params are missing; generating once from safetensors.")
    return weight_bin_generate(script_dir)


if __name__ == "__main__":
    weight_bin_generate()
