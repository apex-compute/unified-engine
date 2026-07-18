"""Standalone pi05_libero weight-export materializer.

This is the SAME lazy weights_export/ materialization the FPGA test.py uses
(ensure_weights_export -> download openpi checkpoint -> dump 51 tensors as .npy +
manifest.json), but with NO unified-engine / hardware dependency, so the pure-torch
reference (pi05_torch_ref.py) can reuse it on a GPU-only box.

Kept intentionally in lockstep with pi05_libero_test.py's export logic: same config,
same _canonical_name, same manifest format ({name: {shape,dtype,file}}), so the export
this module writes is byte-identical to the one test.py would write. If you change the
export format in one place, change it in the other.
"""

import json
import os
from pathlib import Path

import numpy as np


def _load_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "pi05_libero_config.json")
    with open(path) as f:
        return json.load(f)


_CFG = _load_config()
_BIN_SUBDIR = _CFG["paths"].get("bin_dir", "pi05_libero_bin")


def weights_export_dir(script_dir=None):
    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, _BIN_SUBDIR, "weights_export")


def _weights_export_complete(wdir):
    """True iff wdir has a manifest.json and every .npy file it lists. Read-only."""
    manifest_path = os.path.join(wdir, "manifest.json")
    if not os.path.exists(manifest_path):
        return False, None
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (ValueError, OSError):
        return False, None
    missing = [e["file"] for e in manifest.values()
               if not os.path.exists(os.path.join(wdir, e["file"]))]
    return (not missing), manifest


_OPENPI_MISSING_MSG = """\
pi05_libero needs the upstream openpi checkpoint to export its weights, but
`openpi` is not importable in this environment.

Install this model's dependencies:
    cd models/pi05_libero && pip install -r pi_requirements.txt

openpi is NOT on PyPI under that name -- the PyPI `openpi` package is an empty 0.0.0
placeholder, so pi_requirements.txt pulls the real one from git.

Then re-run: the export runs once (~13 GB, from the openpi checkpoint cached under
~/.cache/openpi) and every later run detects it and skips the step.

Underlying import error: {err}
"""


def _import_openpi():
    try:
        from openpi.models.model import restore_params
        from openpi.shared.download import maybe_download
    except ImportError as err:
        raise ImportError(str(err)) from err
    return maybe_download, restore_params


def _flatten_params(tree, prefix=""):
    """Flatten a params PyTree (nested dicts) to {dotted_name: np.ndarray}."""
    flat = {}
    for key, value in tree.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten_params(value, prefix=f"{name}."))
        else:
            flat[name] = np.asarray(value)
    return flat


def _canonical_name(name):
    """Checkpoint leaf name -> manifest key: strip trailing '.value', leading 'params.'."""
    if name.endswith(".value"):
        name = name[: -len(".value")]
    if name.startswith("params."):
        name = name[len("params.") :]
    return name


def _restore_flat_params(cfg):
    """Resolve the checkpoint (downloading if needed) and return {name: np.ndarray}."""
    maybe_download, restore_params = _import_openpi()
    ckpt_url = cfg["paths"]["openpi_checkpoint"]
    print(f"[export] resolving checkpoint: {ckpt_url}")
    ckpt_dir = maybe_download(ckpt_url)          # cached under ~/.cache/openpi, idempotent
    params_path = Path(ckpt_dir) / "params"
    print(f"[export] restoring params from: {params_path}")
    params = restore_params(params_path, restore_type=np.ndarray)

    flat = {}
    for raw_name, arr in _flatten_params(params).items():
        name = _canonical_name(raw_name)
        if name in flat:
            raise RuntimeError(f"[export] duplicate name after canonicalization: {name}")
        flat[name] = np.asarray(arr)
    print(f"[export] {len(flat)} leaves restored")
    return flat


def _do_export(cfg, out_dir):
    """Write every checkpoint tensor to out_dir as .npy + manifest.json.

    Tensors are stored EXACTLY as the checkpoint has them (JAX-style (in, out)
    layout); consumers transpose at load time. Do NOT transpose here.
    """
    flat = _restore_flat_params(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    total_bytes = 0
    for name in sorted(flat):
        arr = flat[name]
        np.save(out_dir / f"{name}.npy", arr, allow_pickle=False)
        manifest[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype),
                          "file": f"{name}.npy"}
        total_bytes += arr.nbytes
        print(f"[export]   {name}  {tuple(arr.shape)}  {arr.dtype}")

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[export] wrote manifest.json ({len(manifest)} entries)")
    print(f"[export] done -> {out_dir}  ({total_bytes / 1e9:.2f} GB)")
    return manifest


def ensure_weights_export(script_dir=None):
    """Lazily materialize weights_export/ on first run. Read-only no-op once complete."""
    wdir = weights_export_dir(script_dir)
    complete, _ = _weights_export_complete(wdir)
    if complete:
        return wdir

    print(
        "\n" + "=" * 78 +
        "\n[pi05_libero] weights_export/ is missing or incomplete -- running the"
        "\n              one-time weight download + export step."
        f"\n              target : {os.path.abspath(wdir)}"
        "\n              size   : ~13 GB on disk (plus the openpi checkpoint cached"
        "\n                       under ~/.cache/openpi)"
        "\n              This happens ONCE; later runs detect it and skip it."
        "\n" + "=" * 78 + "\n")

    try:
        _import_openpi()
    except ImportError as err:
        raise RuntimeError(_OPENPI_MISSING_MSG.format(err=err)) from err

    _do_export(_CFG, wdir)

    complete, _ = _weights_export_complete(wdir)
    if not complete:
        raise RuntimeError(
            f"[export] weights_export still incomplete after export: {wdir}")
    return wdir


if __name__ == "__main__":
    print(ensure_weights_export())
