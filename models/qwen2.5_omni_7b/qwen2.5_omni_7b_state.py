#!/usr/bin/env python3
"""What a compiled programs.bin needs beside its images, so a run can skip compiling.

programs.bin / programs.json hold the instruction images and where each one lives
(DRAM start address and size, per stage and engine). A compiler also leaves facts
in Python objects that the *runtime* uses -- the addresses of tensors it allocated,
the registers it reserved, FLOP counts, which worker belongs to which image. This
module records exactly those attributes around each compile call and stores them in
a sidecar tied to the bin's generation id and to a signature of the request the
programs were built for, so ``--run_from_bin`` can restore them instead of
recompiling (and a bin built for a different request is not reused).
"""

from __future__ import annotations

import io
import json
import os
import pickle
from typing import Any, Iterable

STATE_VERSION = 1


class Recorder:
    """Attributes an object gained or rebound between construction and ``delta``."""

    def __init__(self, obj: Any) -> None:
        self.obj = obj
        self.before = dict(vars(obj))

    def delta(self, skip: Iterable[str] = ()) -> dict[str, Any]:
        skip = set(skip)
        return {k: v for k, v in vars(self.obj).items()
                if k not in skip and (k not in self.before or self.before[k] is not v)}


class _Pickler(pickle.Pickler):
    """Engine objects are stored as ("engine", name) and rebound on load."""

    def __init__(self, stream, engines: dict[str, Any]):
        super().__init__(stream, protocol=pickle.HIGHEST_PROTOCOL)
        self._by_id = {id(obj): name for name, obj in engines.items()}

    def persistent_id(self, obj):
        name = self._by_id.get(id(obj))
        if name is not None:
            return ("engine", name)
        return None


class _Unpickler(pickle.Unpickler):
    def __init__(self, stream, engines: dict[str, Any]):
        super().__init__(stream)
        self._engines = engines

    def persistent_load(self, pid):
        kind, name = pid
        if kind != "engine" or name not in self._engines:
            raise pickle.UnpicklingError(f"unknown persistent object {pid!r}")
        return self._engines[name]


def dumps(value: Any, engines: dict[str, Any]) -> bytes:
    stream = io.BytesIO()
    _Pickler(stream, engines).dump(value)
    return stream.getvalue()


def loads(blob: bytes, engines: dict[str, Any]) -> Any:
    return _Unpickler(io.BytesIO(blob), engines).load()


def state_path(bin_path: str) -> str:
    return os.path.splitext(bin_path)[0] + ".state"


def check_picklable(payload: dict, engines: dict[str, Any]) -> None:
    """Name the first attribute that cannot be stored, instead of a bare pickle error."""
    for stage, attrs in payload.items():
        if not isinstance(attrs, dict):
            continue
        for name, value in attrs.items():
            try:
                dumps(value, engines)
            except Exception as exc:  # noqa: BLE001
                raise TypeError(
                    f"compile state {stage}.{name} ({type(value).__name__}) cannot be "
                    f"stored for --run_from_bin: {exc}") from exc


def save(bin_path: str, generation_id: str, signature: dict, payload: dict,
         engines: dict[str, Any]) -> None:
    """Write the sidecar atomically; it names the bin generation it belongs to."""
    blob = dumps({"version": STATE_VERSION, "generation_id": generation_id,
                  "signature": signature, "payload": payload}, engines)
    path = state_path(bin_path)
    tmp = path + ".tmp"
    with open(tmp, "wb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def load(bin_path: str, generation_id: str, signature: dict,
         engines: dict[str, Any]) -> dict:
    """Return the payload if the sidecar matches this bin and this request."""
    path = state_path(bin_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} is missing: run once without --run_from_bin to build the "
            "programs and their state")
    with open(path, "rb") as stream:
        blob = stream.read()
    state = loads(blob, engines)
    if state.get("version") != STATE_VERSION:
        raise RuntimeError(f"{path}: state version differs; rebuild without --run_from_bin")
    if state["generation_id"] != generation_id:
        raise RuntimeError(
            f"{path} belongs to another programs.bin generation; rebuild without "
            "--run_from_bin")
    if state["signature"] != signature:
        diff = {k: (state["signature"].get(k), signature.get(k))
                for k in set(state["signature"]) | set(signature)
                if state["signature"].get(k) != signature.get(k)}
        raise RuntimeError(
            "programs.bin was built for a different request; rebuild without "
            f"--run_from_bin. Differences (built, requested): {json.dumps(diff, default=str)}")
    return state["payload"]


class _Peek(pickle.Unpickler):
    def persistent_load(self, pid):
        return pid                      # engines are not needed to read the header


def peek(bin_path: str) -> dict:
    """The sidecar's header (version, generation id, request signature)."""
    path = state_path(bin_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} does not exist")
    with open(path, "rb") as stream:
        state = _Peek(stream).load()
    return {"version": state.get("version"), "generation_id": state.get("generation_id"),
            "signature": state.get("signature")}


def reuse_problem(bin_path: str, manifest_generation: str | None,
                  signature: dict) -> str | None:
    """Why a stored bin cannot serve this request, or None when it can."""
    try:
        header = peek(bin_path)
    except FileNotFoundError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001 - an unreadable sidecar is a mismatch
        return f"the state file cannot be read ({type(exc).__name__}: {exc})"
    if header["version"] != STATE_VERSION:
        return "the state file is from another version"
    if manifest_generation is None or header["generation_id"] != manifest_generation:
        return "the state file belongs to another programs.bin generation"
    built = header["signature"] or {}
    diff = {k: (built.get(k), signature.get(k))
            for k in sorted(set(built) | set(signature)) if built.get(k) != signature.get(k)}
    if diff:
        return ("it was built for a different request: "
                + "; ".join(f"{k}: built {a!r}, now {b!r}" for k, (a, b) in diff.items()))
    return None
