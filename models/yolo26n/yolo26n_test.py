#!/usr/bin/env python3
"""Compile (or, with --run-from-bin, reuse the bin) and run YOLO26n for the model_auto_test.py harness."""
import argparse
from pathlib import Path

from yolo26_common import HERE, add_strategy_arguments, strategy_artifact_path, variant_paths
from yolo26n_compile import build_artifact
from yolo26n_run_from_bin import main as run


def main(argv=None, *, pinned_variant='n'):
    config, _, _ = variant_paths(pinned_variant)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dev', default='xdma0')
    parser.add_argument('--device', default='rk')
    parser.add_argument('--image', type=Path, default=HERE.parent / 'test_samples/people.jpg')
    parser.add_argument('--resolution', choices=config['resolutions'], default='640x480')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--force', action='store_true',
                        help='rebuild the bin (now the default; kept for compatibility)')
    parser.add_argument('--run-from-bin', dest='run_from_bin', action='store_true',
                        help='skip compiling and run the existing bin (error if missing or mismatched)')
    parser.add_argument('--cpu', action='store_true')
    add_strategy_arguments(parser)
    args = parser.parse_args(argv)
    if not args.run_from_bin:
        artifact = build_artifact(force=True, strategy=args.strategy, variant=pinned_variant)
    else:
        artifact = strategy_artifact_path(pinned_variant, args.strategy)
        if not Path(artifact).exists():
            raise SystemExit(f'--run-from-bin: bin missing: {artifact} '
                             f'(run once without --run-from-bin to compile it)')
        try:  # existing bin + force=False only validates it (strategy/format), never compiles
            artifact = build_artifact(force=False, strategy=args.strategy, variant=pinned_variant)
        except RuntimeError as exc:
            raise SystemExit(f'--run-from-bin: {artifact}: {exc}')
    # Harness board names include PCIe boards that share the same transport.
    transport = 'efinix' if args.device == 'efinix' else 'rk'
    command = ['--bin', str(artifact), '--dev', args.dev, '--device', transport,
               '--image', str(args.image), '--resolution', args.resolution, '--timeout', str(args.timeout)]
    if args.cpu:
        command.append('--cpu')
    return run(command, pinned_variant=pinned_variant)


if __name__ == '__main__':
    main()
