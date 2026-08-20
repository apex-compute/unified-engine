#!/usr/bin/env python3
"""Compile a YOLOv5 variant into one direct-run params/program artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from yolov5_artifact import compile_single_bin, load_single_bin
from yolov5_common import (
    ensure_checkpoint,
    get_yolov5_variant,
    load_yolov5_config,
    load_official_yolov5,
    sha256_file,
)


def build_artifact(*, variant: str = "s", checkpoint: Path | None = None,
                   config_path: Path | None = None, output: Path | None = None,
                   force: bool = False) -> Path:
    profile, config, resource_dir = load_yolov5_config(variant, config_path)
    output = (output or resource_dir / config["paths"]["artifact"]).resolve()
    if output.is_file() and not force:
        payload = load_single_bin(output)
        expected = config["source"]["weights_sha256"]
        if payload.get("checkpoint_sha256") != expected:
            raise RuntimeError(
                f"existing artifact was built from {payload.get('checkpoint_sha256')}, "
                f"expected {expected}; pass --force to rebuild")
        print(f"Single bin already valid: {output}")
        print(f"  size={output.stat().st_size / 2**20:.2f} MiB "
              f"sha256={sha256_file(output)}")
        return output

    checkpoint = (checkpoint or resource_dir / config["paths"]["weights"]).resolve()
    checkpoint = ensure_checkpoint(
        checkpoint,
        url=config["source"]["weights_url"],
        sha256=config["source"]["weights_sha256"])
    model = load_official_yolov5(checkpoint, variant=profile.key)
    report = compile_single_bin(
        model, output,
        image_size=int(config["model"]["input_size"]),
        checkpoint_sha256=config["source"]["weights_sha256"],
        variant=profile.key)
    print(f"Single bin written: {report['path']}")
    print(f"  size={report['size'] / 2**20:.2f} MiB sha256={report['sha256']}")
    print(f"  operations={report['operations']} convs={report['convolutions']}")
    print(f"  prepacked_params={report['static_params_bytes'] / 2**20:.2f} MiB "
          f"precompiled_programs={report['program_bytes']} bytes")
    return output


def main(argv=None, *, pinned_variant: str = "s",
         config_path: Path | None = None) -> None:
    model_name = get_yolov5_variant(pinned_variant).model_name
    parser = argparse.ArgumentParser(
        description=f"Compile {model_name} into one Andromeda bin")
    parser.set_defaults(variant=pinned_variant)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    build_artifact(
        variant=args.variant, config_path=config_path, checkpoint=args.checkpoint,
        output=args.output, force=args.force)


if __name__ == "__main__":
    main()
