"""
Export pi05_libero weights from their JAX/orbax checkpoint into flat,
framework-free numpy files (one .npy per param, plus a manifest) for
hardware bring-up.

Requires the `pi05` conda env (openpi installed):
    conda activate pi05
    python export_weights.py [--out-dir ./weights_export]
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "weights_export"))
    args = ap.parse_args()

    from openpi.shared import download
    import openpi.models.model as _model
    import flax.traverse_util as traverse_util

    checkpoint_dir = download.maybe_download(args.checkpoint)
    params_dir = Path(checkpoint_dir) / "params"

    # Same helper openpi's own CheckpointWeightLoader uses internally.
    params = _model.restore_params(params_dir, restore_type=np.ndarray)
    flat = traverse_util.flatten_dict(params, sep=".")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for name, arr in flat.items():
        arr = np.asarray(arr)
        fname = name.replace("/", "_") + ".npy"
        np.save(out_dir / fname, arr)
        manifest[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "file": fname}
        print(f"{name:80s} {arr.shape} {arr.dtype}")

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total_bytes = sum(np.prod(m["shape"]) * np.dtype(m["dtype"]).itemsize for m in manifest.values())
    print(f"\nExported {len(manifest)} tensors, {total_bytes / 1e9:.2f} GB total, to {out_dir}")


if __name__ == "__main__":
    main()
