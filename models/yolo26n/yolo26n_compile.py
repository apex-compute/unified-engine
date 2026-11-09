#!/usr/bin/env python3
"""Compile the verified official YOLO26n checkpoint into a resident FPGA bin."""
import argparse
from pathlib import Path

from yolo26_common import (add_strategy_arguments, ensure_checkpoint, load_checkpoint,
                           strategy_artifact_path, variant_paths)
from yolo26_artifact import compile_artifact, load_artifact


def build_artifact(*, checkpoint=None, output=None, strategy='optimized', force=False, variant='n'):
    artifact = strategy_artifact_path(variant, strategy)
    config, default_checkpoint, _ = variant_paths(variant)
    checkpoint = checkpoint or default_checkpoint
    winograd, optimize = strategy != 'native', strategy == 'optimized'
    output = Path(output or artifact)
    if output.exists() and not force:
        payload = load_artifact(output)
        if (payload['winograd'] != winograd or bool(payload.get('optimized')) != optimize
                or payload['format'] != f'andromeda.yolo26{variant}.single-bin'):
            raise RuntimeError('Existing bin uses another convolution strategy; pass --force')
        print(f'Validated existing artifact: {output}')
        return output
    checkpoint = ensure_checkpoint(Path(checkpoint), url=config['checkpoint_url'],
                                    sha256=config['checkpoint_sha256'])
    payload = compile_artifact(load_checkpoint(checkpoint, variant=variant), output, winograd=winograd, optimize=optimize)
    print(f'Wrote {output.resolve()} ({output.stat().st_size / 2**20:.2f} MiB)')
    print(f'Bundle SHA-256: {payload["bundle_sha256"]}')
    return output


def main(argv=None, *, pinned_variant='n'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    add_strategy_arguments(parser)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    build_artifact(**vars(args), variant=pinned_variant)


if __name__ == '__main__':
    main()
