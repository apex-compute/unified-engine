#!/usr/bin/env python3
"""Export pi05_libero weights from the upstream openpi JAX checkpoint to .npy + manifest.json.

This makes pi05_libero_bin/weights_export/ a reproducible prep-phase artifact instead of
an un-regenerable blob.

Requires the `pi05` conda env (openpi installed):
    /home/rohit/miniconda3/envs/pi05/bin/python pi05_libero_export_weights.py \
        --out pi05_libero_bin/weights_export_new

Modes:
    (default)   export the checkpoint to --out
    --verify    compare --out against --reference (read-only) tensor by tensor

Paths: the export lives under the bin dir (pi05_libero_bin/weights_export) alongside
params.bin/programs.bin -- everything generated is consolidated there.

Weights are stored exactly as the checkpoint has them (JAX-style (in, out) layout);
the consumer transposes at load time. Do NOT transpose here.
"""
import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

import numpy as np

CONFIG_PATH = SCRIPT_DIR / "pi05_libero_config.json"
# Everything generated lives under the bin dir (repo convention; *_bin/ is gitignored).
BIN_DIR = SCRIPT_DIR / json.loads(CONFIG_PATH.read_text())["paths"].get(
    "bin_dir", "pi05_libero_bin")
DEFAULT_OUT = BIN_DIR / "weights_export_new"
DEFAULT_REFERENCE = BIN_DIR / "weights_export"

_OPENPI_HELP = """\
openpi is not importable -- this model's dependencies are not installed.

Install them:
    cd models/pi05_libero && pip install -r requirements.txt

That requirements.txt is the merged env spec for pi05_libero (engine + openpi +
libero/robosuite, numpy<2 pinned for all three). openpi is NOT installable from PyPI
under that name -- the PyPI `openpi` package is a 0.0.0 placeholder stub with no code,
so requirements.txt pulls the real one from git.

Underlying import error: {err}
"""


def _import_openpi():
    """Import openpi, or raise ImportError carrying ONLY the raw cause.

    Callers render the guidance: the CLI (main) prints _OPENPI_HELP; pi05_libero_test.py
    prints its own equivalent. Raising the full help text here would get interpolated
    into the caller's message and print two competing sets of instructions.
    """
    try:
        from openpi.models.model import restore_params
        from openpi.shared.download import maybe_download
    except ImportError as err:
        raise ImportError(str(err)) from err
    return maybe_download, restore_params


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def flatten_params(tree, prefix=""):
    """Flatten a params PyTree (nested dicts) to {dotted_name: np.ndarray}."""
    flat = {}
    for key, value in tree.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_params(value, prefix=f"{name}."))
        else:
            flat[name] = np.asarray(value)
    return flat


def canonical_name(name):
    """Checkpoint leaf name -> manifest key.

    Strips a trailing `.value` and a leading `params.`:
      params.PaliGemma.img.Transformer.encoder_norm.bias.value
        -> PaliGemma.img.Transformer.encoder_norm.bias
    """
    if name.endswith(".value"):
        name = name[: -len(".value")]
    if name.startswith("params."):
        name = name[len("params.") :]
    return name


def restore_flat_params(cfg):
    """Resolve the checkpoint (downloading if needed) and return {name: np.ndarray}."""
    maybe_download, restore_params = _import_openpi()
    ckpt_url = cfg["paths"]["openpi_checkpoint"]
    print(f"[export] resolving checkpoint: {ckpt_url}")
    ckpt_dir = maybe_download(ckpt_url)
    params_path = Path(ckpt_dir) / "params"
    print(f"[export] restoring params from: {params_path}")
    params = restore_params(params_path, restore_type=np.ndarray)

    flat = {}
    for raw_name, arr in flatten_params(params).items():
        name = canonical_name(raw_name)
        if name in flat:
            sys.exit(f"[export] ERROR: duplicate name after canonicalization: {name}")
        flat[name] = np.asarray(arr)
    print(f"[export] {len(flat)} leaves restored")
    return flat


def manifest_entry(name, arr):
    return {"shape": list(arr.shape), "dtype": str(arr.dtype), "file": f"{name}.npy"}


def do_export(cfg, out_dir, only=None):
    flat = restore_flat_params(cfg)
    if only is not None:
        missing = [n for n in only if n not in flat]
        if missing:
            sys.exit(f"[export] ERROR: requested names not in checkpoint: {missing}")
        flat = {n: flat[n] for n in only}
        print(f"[export] subset mode: exporting {len(flat)} of the restored leaves")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    total_bytes = 0
    for name in sorted(flat):
        arr = flat[name]
        np.save(out_dir / f"{name}.npy", arr, allow_pickle=False)
        manifest[name] = manifest_entry(name, arr)
        total_bytes += arr.nbytes
        print(f"[export]   {name}  {tuple(arr.shape)}  {arr.dtype}")

    if only is None:
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[export] wrote manifest.json ({len(manifest)} entries)")
    else:
        print("[export] subset mode: manifest.json NOT written (would be incomplete)")

    print(f"[export] done -> {out_dir}  ({total_bytes / 1e9:.2f} GB)")
    return manifest


def do_verify(cfg, ref_dir, only=None):
    """Compare freshly-restored checkpoint tensors against an existing export (read-only)."""
    ref_manifest_path = ref_dir / "manifest.json"
    if not ref_manifest_path.exists():
        sys.exit(f"[verify] ERROR: no manifest at {ref_manifest_path}")
    with open(ref_manifest_path) as f:
        ref_manifest = json.load(f)

    flat = restore_flat_params(cfg)

    gen_names = set(flat)
    ref_names = set(ref_manifest)
    only_gen = sorted(gen_names - ref_names)
    only_ref = sorted(ref_names - gen_names)
    print(f"\n[verify] name sets: generated={len(gen_names)} reference={len(ref_names)}")
    if only_gen:
        print(f"[verify] ONLY IN GENERATED ({len(only_gen)}): {only_gen}")
    if only_ref:
        print(f"[verify] ONLY IN REFERENCE ({len(only_ref)}): {only_ref}")
    if not only_gen and not only_ref:
        print("[verify] name sets match 1:1 OK")

    names = sorted(gen_names & ref_names)
    if only is not None:
        names = [n for n in names if n in set(only)]
        print(f"[verify] subset mode: checking {len(names)} tensors")

    n_exact = 0
    failures = []
    print(f"\n[verify] {'name':<70} {'shape':<18} exact  max|diff|")
    for name in names:
        gen = flat[name]
        ref = np.load(ref_dir / ref_manifest[name]["file"], allow_pickle=False, mmap_mode="r")
        ref = np.asarray(ref)

        if gen.shape != ref.shape or gen.dtype != ref.dtype:
            failures.append(f"{name}: shape/dtype mismatch gen={gen.shape}/{gen.dtype} ref={ref.shape}/{ref.dtype}")
            print(f"[verify] {name:<70} MISMATCH gen={gen.shape}/{gen.dtype} ref={ref.shape}/{ref.dtype}")
            continue

        exact = bool(np.array_equal(gen, ref))
        maxdiff = float(np.max(np.abs(gen.astype(np.float64) - ref.astype(np.float64)))) if gen.size else 0.0
        n_exact += exact
        if not exact:
            failures.append(f"{name}: not bit-identical, max|diff|={maxdiff:.3e}")
        print(f"[verify] {name:<70} {str(tuple(gen.shape)):<18} {'YES' if exact else 'NO ':<6} {maxdiff:.3e}")

        # also cross-check the manifest metadata itself
        ent = ref_manifest[name]
        if list(gen.shape) != list(ent["shape"]) or str(gen.dtype) != ent["dtype"]:
            failures.append(f"{name}: manifest metadata disagrees with restored tensor")

        del ref

    print(f"\n[verify] {n_exact}/{len(names)} tensors bit-identical")
    if failures or only_gen or only_ref:
        print(f"[verify] FAILURES ({len(failures)}):")
        for f_ in failures:
            print(f"[verify]   {f_}")
        return 1
    print("[verify] PASS: export is reproducible")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Export pi05_libero weights from the upstream openpi JAX checkpoint.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"output directory for .npy + manifest.json (default: {DEFAULT_OUT})")
    ap.add_argument("--verify", action="store_true",
                    help="compare the checkpoint against --reference instead of exporting (read-only)")
    ap.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE,
                    help=f"existing export to verify against, read-only (default: {DEFAULT_REFERENCE})")
    ap.add_argument("--only", nargs="+", default=None, metavar="NAME",
                    help="restrict export/verify to these tensor names (subset check)")
    ap.add_argument("--list", action="store_true",
                    help="list checkpoint leaf names + shapes and exit (no writes)")
    args = ap.parse_args()

    cfg = load_config()

    if args.list:
        for name, arr in sorted(restore_flat_params(cfg).items()):
            print(f"{name:<70} {str(tuple(arr.shape)):<18} {arr.dtype}")
        return 0

    if args.verify:
        return do_verify(cfg, args.reference.resolve(), only=args.only)

    out_dir = args.out.resolve()
    # Protect an EXISTING export from being overwritten. Keyed on the manifest actually
    # being there, not on the path alone: if the canonical dir is missing/empty there is
    # nothing to protect and exporting into it is exactly what the user wants.
    if out_dir == DEFAULT_REFERENCE.resolve() and (out_dir / "manifest.json").exists():
        sys.exit(f"[export] REFUSING to write to {out_dir}: a complete export already lives\n"
                 f"         there and it is the validated reference. Export elsewhere\n"
                 f"         (--out {BIN_DIR.name}/weights_export_new) and compare with --verify.")
    do_export(cfg, out_dir, only=args.only)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ImportError as err:  # openpi missing -- render the install guidance once, here
        sys.exit(_OPENPI_HELP.format(err=err))
