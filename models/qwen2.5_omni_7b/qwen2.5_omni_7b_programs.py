#!/usr/bin/env python3
"""Atomic, identity-bound program bundles for Qwen2.5-Omni-7B.

The compiler stores each FPGA stage in one ``programs.bin`` and records its
engine, baked DRAM address, compile metadata, byte range, and SHA-256 in
``programs.json``.  The final 32 bytes of the binary are a random generation
tag also held by the sidecar.  This makes a crash between the two atomic
renames detectable on the next load.

``ProgramBundle.store_stage`` publishes every engine section of one stage in a
single rewrite.  It deliberately starts with an empty image on its first call
for a path in each process.  Later calls for that path preserve only complete
stages produced by the current run.  This is the Gemma-style fresh-compile
hygiene: stages left by a previous invocation cannot silently enter a newly
compiled image.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any


SCHEMA_VERSION = 1
GENERATION_TRAILER_BYTES = 32
_ARTIFACT_KIND = "qwen2.5-omni-7b-programs"
_TOP_LEVEL_KEYS = {
    "artifact",
    "schema_version",
    "identity",
    "identity_sha256",
    "generation_id",
    "generation_trailer_bytes",
    "payload_size",
    "programs_size",
    "section_count",
    "sections",
}
_SECTION_KEYS = {
    "name",
    "engine_index",
    "dram_base",
    "file_offset",
    "size",
    "sha256",
    "metadata",
}


class ProgramBundleError(RuntimeError):
    """A program bundle is absent, stale, corrupt, or internally inconsistent."""


@dataclass(frozen=True)
class _Stage:
    name: str
    engine_index: int
    dram_base: int
    program: bytes
    metadata: dict[str, Any]


# Path -> canonical identity digest.  This state is intentionally process-wide
# rather than per object: independent compiler components may each construct a
# ProgramBundle for the same output directory and must still preserve stages
# emitted earlier in this run.
_STARTED_BUNDLES: dict[tuple[str, str], str] = {}
_BUNDLE_LOCK = threading.RLock()


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _canonical_json(value: object, label: str) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite JSON data: {exc}") from exc


def _normalise_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} keys must be strings")
    return json.loads(_canonical_json(dict(value), label).decode("ascii"))


def _validate_stage_key(name: str, engine_index: int) -> tuple[str, int]:
    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError("stage name must be a non-empty, trimmed string")
    if not _is_plain_int(engine_index) or engine_index < 0:
        raise ValueError("engine_index must be a non-negative integer")
    return name, engine_index


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProgramBundleError(f"program manifest repeats JSON key {key!r}")
        result[key] = value
    return result


def _write_temp(path: Path, data: bytes) -> str:
    fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".building", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise
    return tmp_path


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(directory, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class ProgramBundle:
    """Build and load one strict ``programs.bin``/``programs.json`` pair.

    ``identity`` is compared as canonical JSON, including its exact keys and
    values.  It must include non-null ``params`` and ``code`` entries supplied
    by the caller (normally the params generation/config fingerprints and a
    compiler/source fingerprint).  Model geometry, engine count, FPGA build,
    or compile schema may be included as additional identity keys.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        identity: Mapping[str, Any],
        *,
        stem: str = "programs",
    ) -> None:
        if not isinstance(stem, str) or not stem or Path(stem).name != stem:
            raise ValueError("stem must be one non-empty filename component")
        self.directory = Path(directory).expanduser().resolve()
        self.bin_path = self.directory / f"{stem}.bin"
        self.json_path = self.directory / f"{stem}.json"
        self.identity = _normalise_mapping(identity, "program identity")
        missing = [key for key in ("params", "code") if self.identity.get(key) is None]
        if missing:
            raise ValueError(
                "program identity must contain non-null caller-supplied "
                + " and ".join(repr(key) for key in missing)
                + " identity"
            )
        self._identity_bytes = _canonical_json(self.identity, "program identity")
        self._identity_sha256 = _sha256(self._identity_bytes)
        self._run_key = (str(self.bin_path), str(self.json_path))

    def store_stage(
        self,
        name: str,
        sections: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
        stage_metadata: Mapping[str, Any] | None = None,
    ) -> dict[int, bytes]:
        """Backward-compatible wrapper for one-stage atomic publication."""
        return self.store_stages(
            {
                name: {
                    "sections": sections,
                    "metadata": stage_metadata or {},
                }
            }
        )[name]

    def store_stages(
        self,
        stages: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, dict[int, bytes]]:
        """Atomically store or replace one or more complete program stages.

        ``stages`` maps each stage name to ``{"sections": [...],
        "metadata": {...}}``.  Each section must contain exactly
        ``engine_index``, ``dram_base``, and ``bytes``.  Every stage and every
        engine section is normalised before the lock or any filesystem write;
        consequently a bad decode group cannot publish a valid prefill group.
        All named groups then replace their previous versions in one binary
        generation while unnamed current-run stages are preserved.

        Duplicate DRAM bases are valid because media, prefill, and decode
        phases time-share program addresses.  Returned bytes come only from
        the newly reopened and validated artifact.
        """
        if not isinstance(stages, Mapping) or not stages:
            raise ValueError("stages must be a non-empty mapping")
        replacements: OrderedDict[str, list[_Stage]] = OrderedDict()
        for name, raw_stage in stages.items():
            if not isinstance(name, str) or not name or name != name.strip():
                raise ValueError("stage name must be a non-empty, trimmed string")
            if not isinstance(raw_stage, Mapping):
                raise TypeError(f"stage {name!r} must be a mapping")
            if set(raw_stage) != {"sections", "metadata"}:
                raise ValueError(
                    f"stage {name!r} must contain exactly sections and metadata"
                )
            sections = raw_stage["sections"]
            if not isinstance(sections, (list, tuple)) or not sections:
                raise ValueError(
                    f"stage {name!r} sections must be a non-empty list or tuple"
                )
            raw_metadata = raw_stage["metadata"]
            metadata = _normalise_mapping(
                {} if raw_metadata is None else raw_metadata,
                f"stage {name!r} metadata",
            )
            replacement: list[_Stage] = []
            replacement_engines = set()
            for index, raw_section in enumerate(sections):
                if not isinstance(raw_section, Mapping):
                    raise TypeError(f"stage {name!r} sections[{index}] must be a mapping")
                if set(raw_section) != {"engine_index", "dram_base", "bytes"}:
                    raise ValueError(
                        f"stage {name!r} sections[{index}] must contain exactly "
                        "engine_index, dram_base, and bytes"
                    )
                engine_index = raw_section["engine_index"]
                _validate_stage_key(name, engine_index)
                if engine_index in replacement_engines:
                    raise ValueError(f"stage {name!r} repeats engine {engine_index}")
                replacement_engines.add(engine_index)
                dram_base = raw_section["dram_base"]
                if not _is_plain_int(dram_base) or not 0 <= dram_base < (1 << 64):
                    raise ValueError(
                        f"stage {name!r} sections[{index}].dram_base must be an "
                        "unsigned 64-bit integer"
                    )
                try:
                    program = memoryview(raw_section["bytes"]).tobytes()
                except TypeError as exc:
                    raise TypeError(
                        f"stage {name!r} sections[{index}].bytes must support "
                        "the buffer protocol"
                    ) from exc
                if not program:
                    raise ValueError(
                        f"stage {name!r} sections[{index}].bytes must not be empty"
                    )
                replacement.append(
                    _Stage(name, engine_index, dram_base, program, metadata)
                )
            replacements[name] = replacement

        with _BUNDLE_LOCK:
            started_identity = _STARTED_BUNDLES.get(self._run_key)
            current: OrderedDict[tuple[str, int], _Stage] = OrderedDict()
            if started_identity is not None:
                if started_identity != self._identity_sha256:
                    raise ProgramBundleError(
                        "this programs path already started in the current process "
                        "with a different params/code identity"
                    )
                manifest, payload = self._load_validated()
                for section in manifest["sections"]:
                    section_key = (section["name"], section["engine_index"])
                    start = section["file_offset"]
                    end = start + section["size"]
                    current[section_key] = _Stage(
                        name=section["name"],
                        engine_index=section["engine_index"],
                        dram_base=int(section["dram_base"], 16),
                        program=payload[start:end],
                        metadata=section["metadata"],
                    )
            replaced_names = set(replacements)
            current = OrderedDict(
                (key, stage)
                for key, stage in current.items()
                if stage.name not in replaced_names
            )
            for replacement in replacements.values():
                for stage in replacement:
                    current[(stage.name, stage.engine_index)] = stage

            payload_out = bytearray()
            sections_out = []
            for stage in current.values():
                offset = len(payload_out)
                payload_out.extend(stage.program)
                sections_out.append(
                    {
                        "name": stage.name,
                        "engine_index": stage.engine_index,
                        "dram_base": f"0x{stage.dram_base:X}",
                        "file_offset": offset,
                        "size": len(stage.program),
                        "sha256": _sha256(stage.program),
                        "metadata": stage.metadata,
                    }
                )

            generation = os.urandom(GENERATION_TRAILER_BYTES)
            binary = bytes(payload_out) + generation
            manifest_out = {
                "artifact": _ARTIFACT_KIND,
                "schema_version": SCHEMA_VERSION,
                "identity": self.identity,
                "identity_sha256": self._identity_sha256,
                "generation_id": generation.hex(),
                "generation_trailer_bytes": GENERATION_TRAILER_BYTES,
                "payload_size": len(payload_out),
                "programs_size": len(binary),
                "section_count": len(sections_out),
                "sections": sections_out,
            }
            manifest_bytes = json.dumps(
                manifest_out, indent=2, ensure_ascii=True, allow_nan=False
            ).encode("ascii") + b"\n"
            self._replace_pair(binary, manifest_bytes)

            disk_manifest, disk_payload = self._load_validated()
            disk_stages: dict[str, dict[int, bytes]] = {
                name: {} for name in replacements
            }
            for section in disk_manifest["sections"]:
                if section["name"] in disk_stages:
                    start = section["file_offset"]
                    disk_stages[section["name"]][section["engine_index"]] = disk_payload[
                        start:start + section["size"]
                    ]
            expected_stages = {
                name: {stage.engine_index: stage.program for stage in replacement}
                for name, replacement in replacements.items()
            }
            if disk_stages != expected_stages:
                changed = sorted(
                    name
                    for name in replacements
                    if disk_stages[name] != expected_stages[name]
                )
                raise ProgramBundleError(
                    "stages changed or became partial during store: "
                    + ", ".join(changed)
                )
            _STARTED_BUNDLES[self._run_key] = self._identity_sha256
            return disk_stages

    def load(self) -> tuple[dict[str, Any], bytes]:
        """Return the validated manifest and payload bytes (trailer excluded)."""
        with _BUNDLE_LOCK:
            return self._load_validated()

    def read_stage(self, name: str) -> dict[int, bytes]:
        """Return a complete stage's engine-to-bytes map from disk."""
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError("stage name must be a non-empty, trimmed string")
        manifest, payload = self.load()
        result = {}
        for section in manifest["sections"]:
            if section["name"] == name:
                start = section["file_offset"]
                result[section["engine_index"]] = payload[
                    start:start + section["size"]
                ]
        if not result:
            raise KeyError(f"program stage {name!r} is absent")
        return result

    def _replace_pair(self, binary: bytes, manifest: bytes) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        bin_tmp = _write_temp(self.bin_path, binary)
        json_tmp = None
        try:
            json_tmp = _write_temp(self.json_path, manifest)
            os.replace(bin_tmp, self.bin_path)
            bin_tmp = None
            _fsync_directory(self.directory)
            os.replace(json_tmp, self.json_path)
            json_tmp = None
            _fsync_directory(self.directory)
        finally:
            for tmp in (bin_tmp, json_tmp):
                if tmp is not None:
                    try:
                        os.unlink(tmp)
                    except FileNotFoundError:
                        pass

    def _load_validated(self) -> tuple[dict[str, Any], bytes]:
        if not self.bin_path.is_file() or not self.json_path.is_file():
            missing = [
                str(path)
                for path in (self.bin_path, self.json_path)
                if not path.is_file()
            ]
            raise FileNotFoundError("program bundle is incomplete: " + ", ".join(missing))

        try:
            with self.json_path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream, object_pairs_hook=_reject_duplicate_json_keys)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ProgramBundleError(f"invalid program manifest: {exc}") from exc
        with self.bin_path.open("rb") as stream:
            binary = stream.read()

        if len(binary) < GENERATION_TRAILER_BYTES:
            raise ProgramBundleError(
                f"programs.bin is shorter than its {GENERATION_TRAILER_BYTES}-byte "
                "generation trailer"
            )
        if not isinstance(manifest, dict):
            raise ProgramBundleError("program manifest must be a JSON object")
        if set(manifest) != _TOP_LEVEL_KEYS:
            raise ProgramBundleError("program manifest top-level schema differs")
        if manifest["artifact"] != _ARTIFACT_KIND:
            raise ProgramBundleError("program artifact kind differs")
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise ProgramBundleError("program schema version differs")

        try:
            manifest_identity = _canonical_json(manifest["identity"], "manifest identity")
        except ValueError as exc:
            raise ProgramBundleError(str(exc)) from exc
        if manifest_identity != self._identity_bytes:
            raise ProgramBundleError("program params/code identity is stale")
        if manifest["identity_sha256"] != self._identity_sha256:
            raise ProgramBundleError("program identity SHA-256 differs")

        generation_hex = manifest["generation_id"]
        if (
            manifest["generation_trailer_bytes"] != GENERATION_TRAILER_BYTES
            or not _is_lower_sha256(generation_hex)
        ):
            raise ProgramBundleError("generation trailer metadata is invalid")
        generation = bytes.fromhex(generation_hex)
        if binary[-GENERATION_TRAILER_BYTES:] != generation:
            raise ProgramBundleError(
                "programs.bin and programs.json are from different generations"
            )

        payload = binary[:-GENERATION_TRAILER_BYTES]
        if manifest["programs_size"] != len(binary):
            raise ProgramBundleError("programs.bin size differs from its manifest")
        if manifest["payload_size"] != len(payload):
            raise ProgramBundleError("program payload size differs from its manifest")
        sections = manifest["sections"]
        if not isinstance(sections, list):
            raise ProgramBundleError("program sections must be a list")
        if manifest["section_count"] != len(sections):
            raise ProgramBundleError("program section count differs")

        cursor = 0
        seen = set()
        for index, section in enumerate(sections):
            if not isinstance(section, dict) or set(section) != _SECTION_KEYS:
                raise ProgramBundleError(f"program section {index} schema differs")
            try:
                key = _validate_stage_key(section["name"], section["engine_index"])
            except ValueError as exc:
                raise ProgramBundleError(f"program section {index}: {exc}") from exc
            if key in seen:
                raise ProgramBundleError(f"duplicate program section {key!r}")
            seen.add(key)

            base_text = section["dram_base"]
            try:
                base = int(base_text, 16)
            except (TypeError, ValueError):
                raise ProgramBundleError(
                    f"program section {index} DRAM base is invalid"
                ) from None
            if base < 0 or base >= (1 << 64) or base_text != f"0x{base:X}":
                raise ProgramBundleError(
                    f"program section {index} DRAM base is not canonical"
                )
            offset, size = section["file_offset"], section["size"]
            if not _is_plain_int(offset) or not _is_plain_int(size) or size <= 0:
                raise ProgramBundleError(f"program section {index} bounds are invalid")
            if offset != cursor:
                raise ProgramBundleError(
                    f"program section {index} offset {offset!r} is not contiguous "
                    f"at {cursor}"
                )
            end = offset + size
            if end > len(payload):
                raise ProgramBundleError(f"program section {index} exceeds the payload")
            digest = section["sha256"]
            if not _is_lower_sha256(digest) or _sha256(payload[offset:end]) != digest:
                raise ProgramBundleError(f"program section {index} SHA-256 differs")
            if not isinstance(section["metadata"], dict):
                raise ProgramBundleError(f"program section {index} metadata is invalid")
            cursor = end
        if cursor != len(payload):
            raise ProgramBundleError("program sections do not cover the complete payload")
        return manifest, payload


def _expect_bundle_error(action, text: str) -> None:
    try:
        action()
    except ProgramBundleError as exc:
        if text not in str(exc):
            raise AssertionError(f"expected {text!r} in {str(exc)!r}") from exc
    else:
        raise AssertionError(f"expected ProgramBundleError containing {text!r}")


def self_test() -> None:
    """Exercise bundle semantics using temporary files and no hardware."""
    import shutil

    identity = {
        "params": {"generation_id": "11" * 32, "size": 1234},
        "code": {"sha256": "22" * 32, "schema": 9},
        "model": "Qwen2.5-Omni-7B-Thinker",
        "engines": 8,
    }
    with tempfile.TemporaryDirectory(prefix="qwen-omni-programs-") as root:
        root_path = Path(root)
        happy = ProgramBundle(root_path / "happy", identity)
        first = bytes(range(64))
        second = bytes(reversed(range(64)))
        assert happy.store_stage(
            "prefill",
            [
                {"engine_index": 0, "dram_base": 0x1FA000000, "bytes": first},
                {"engine_index": 1, "dram_base": 0x1FE000000, "bytes": second},
            ],
            {"rows": 64},
        ) == {0: first, 1: second}
        assert happy.store_stage(
            "decode",
            [{"engine_index": 0, "dram_base": 0x1FA000000, "bytes": second}],
            {"rows": 1},
        ) == {0: second}
        manifest, payload = happy.load()
        assert payload == first + second + second
        assert [section["file_offset"] for section in manifest["sections"]] == [0, 64, 128]
        assert happy.read_stage("prefill") == {0: first, 1: second}

        # Prefill and decode publish in one generation.  A malformed member of
        # a later paired replacement is rejected during normalisation, before
        # either final file or either previously valid stage can change.
        paired = ProgramBundle(root_path / "paired", identity)
        original = paired.store_stages(
            {
                "prefill": {
                    "sections": [
                        {"engine_index": 0, "dram_base": 0x1000, "bytes": first},
                        {"engine_index": 1, "dram_base": 0x2000, "bytes": second},
                    ],
                    "metadata": {"phase": "prefill", "revision": 1},
                },
                "decode": {
                    "sections": [
                        {"engine_index": 0, "dram_base": 0x1000, "bytes": second},
                        {"engine_index": 1, "dram_base": 0x2000, "bytes": first},
                    ],
                    "metadata": {"phase": "decode", "revision": 1},
                },
            }
        )
        assert original == {
            "prefill": {0: first, 1: second},
            "decode": {0: second, 1: first},
        }
        original_manifest, _ = paired.load()
        original_bin = paired.bin_path.read_bytes()
        original_json = paired.json_path.read_bytes()
        try:
            paired.store_stages(
                {
                    "prefill": {
                        "sections": [
                            {"engine_index": 0, "dram_base": 0x1000, "bytes": second},
                            {"engine_index": 1, "dram_base": 0x2000, "bytes": first},
                        ],
                        "metadata": {"phase": "prefill", "revision": 2},
                    },
                    "decode": {
                        "sections": [
                            {"engine_index": 0, "dram_base": 0x1000, "bytes": first},
                            {"engine_index": 0, "dram_base": 0x2000, "bytes": second},
                        ],
                        "metadata": {"phase": "decode", "revision": 2},
                    },
                }
            )
        except ValueError as exc:
            assert "repeats engine 0" in str(exc)
        else:
            raise AssertionError("malformed paired replacement unexpectedly succeeded")
        assert paired.bin_path.read_bytes() == original_bin
        assert paired.json_path.read_bytes() == original_json
        assert paired.read_stage("prefill") == original["prefill"]
        assert paired.read_stage("decode") == original["decode"]

        replaced = paired.store_stages(
            {
                "prefill": {
                    "sections": [
                        {"engine_index": 0, "dram_base": 0x1000, "bytes": second},
                        {"engine_index": 1, "dram_base": 0x2000, "bytes": first},
                    ],
                    "metadata": {"phase": "prefill", "revision": 2},
                },
                "decode": {
                    "sections": [
                        {"engine_index": 0, "dram_base": 0x1000, "bytes": first},
                        {"engine_index": 1, "dram_base": 0x2000, "bytes": second},
                    ],
                    "metadata": {"phase": "decode", "revision": 2},
                },
            }
        )
        assert replaced == {
            "prefill": {0: second, 1: first},
            "decode": {0: first, 1: second},
        }
        replaced_manifest, _ = paired.load()
        assert replaced_manifest["generation_id"] != original_manifest["generation_id"]
        assert paired.read_stage("prefill") == replaced["prefill"]
        assert paired.read_stage("decode") == replaced["decode"]

        fresh_dir = root_path / "fresh"
        fresh_dir.mkdir()
        shutil.copy2(happy.bin_path, fresh_dir / "programs.bin")
        shutil.copy2(happy.json_path, fresh_dir / "programs.json")
        fresh = ProgramBundle(fresh_dir, identity)
        fresh.store_stage(
            "audio",
            [{"engine_index": 3, "dram_base": 0x1FA000000, "bytes": first}],
        )
        assert [(s["name"], s["engine_index"]) for s in fresh.load()[0]["sections"]] == [
            ("audio", 3)
        ]

        corrupt = ProgramBundle(root_path / "corrupt", identity)
        corrupt.store_stage(
            "lm", [{"engine_index": 0, "dram_base": 0x1FA000000, "bytes": first}]
        )
        with corrupt.bin_path.open("r+b") as stream:
            byte = stream.read(1)
            stream.seek(0)
            stream.write(bytes([byte[0] ^ 0xFF]))
            stream.flush()
            os.fsync(stream.fileno())
        _expect_bundle_error(corrupt.load, "SHA-256")

        stale = ProgramBundle(root_path / "stale", identity)
        stale.store_stage(
            "lm", [{"engine_index": 0, "dram_base": 0x1FA000000, "bytes": first}]
        )
        stale_identity = {**identity, "code": {"sha256": "33" * 32, "schema": 9}}
        _expect_bundle_error(
            ProgramBundle(root_path / "stale", stale_identity).load,
            "identity is stale",
        )

        short = ProgramBundle(root_path / "short", identity)
        short.store_stage(
            "lm", [{"engine_index": 0, "dram_base": 0x1FA000000, "bytes": first}]
        )
        with short.bin_path.open("r+b") as stream:
            stream.truncate(7)
            stream.flush()
            os.fsync(stream.fileno())
        _expect_bundle_error(short.load, "shorter than its 32-byte")

    print("Qwen2.5-Omni program bundle self-test: PASS")


if __name__ == "__main__":
    self_test()
