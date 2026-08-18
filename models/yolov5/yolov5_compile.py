#!/usr/bin/env python3
"""Compile YOLOv5s into one direct-run params/program/model artifact."""

from __future__ import annotations

import argparse
import json
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
    load_official_yolov5s,
    sha256_file,
)


def build_artifact(*, checkpoint: Path | None = None,
                   output: Path | None = None, force: bool = False) -> Path:
    with (SCRIPT_DIR / "yolov5_config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    output = (output or SCRIPT_DIR / config["paths"]["artifact"]).resolve()
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

    checkpoint = (checkpoint or SCRIPT_DIR / config["paths"]["weights"]).resolve()
    checkpoint = ensure_checkpoint(
        checkpoint,
        url=config["source"]["weights_url"],
        sha256=config["source"]["weights_sha256"])
    model = load_official_yolov5s(checkpoint)
    report = compile_single_bin(
        model, output,
        image_size=int(config["model"]["input_size"]),
        checkpoint_sha256=config["source"]["weights_sha256"])
    print(f"Single bin written: {report['path']}")
    print(f"  size={report['size'] / 2**20:.2f} MiB sha256={report['sha256']}")
    print(f"  operations={report['operations']} convs={report['convolutions']}")
    print(f"  prepacked_params={report['static_params_bytes'] / 2**20:.2f} MiB "
          f"precompiled_programs={report['program_bytes']} bytes")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile YOLOv5s into one checkpoint-free Andromeda bin")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_artifact(
        checkpoint=args.checkpoint, output=args.output, force=args.force)


if __name__ == "__main__":
    main()
