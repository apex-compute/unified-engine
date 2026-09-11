#!/usr/bin/env python3
"""Compare a DPDFNet hardware spectrum output with the CPU reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()

    reference = np.asarray(np.load(args.reference, allow_pickle=False), dtype=np.float64)
    candidate = np.asarray(np.load(args.candidate, allow_pickle=False), dtype=np.float64)
    if candidate.shape != reference.shape:
        parser.error(
            f"shape mismatch: candidate={candidate.shape}, reference={reference.shape}")
    delta = candidate - reference
    reference_norm = float(np.linalg.norm(reference.reshape(-1)))
    result = {
        "reference": str(args.reference.expanduser().resolve()),
        "candidate": str(args.candidate.expanduser().resolve()),
        "values": int(reference.size),
        "relative_l2": float(np.linalg.norm(delta.reshape(-1)) / max(reference_norm, 1e-12)),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "finite_output": bool(np.isfinite(candidate).all()),
    }
    print("TEST_RESULT:" + json.dumps(result))


if __name__ == "__main__":
    main()
