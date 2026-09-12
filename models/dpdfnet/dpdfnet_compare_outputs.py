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

    with np.errstate(invalid="ignore", over="ignore"):
        reference = np.asarray(
            np.load(args.reference, allow_pickle=False), dtype=np.float64)
        candidate = np.asarray(
            np.load(args.candidate, allow_pickle=False), dtype=np.float64)
    if candidate.shape != reference.shape:
        parser.error(
            f"shape mismatch: candidate={candidate.shape}, reference={reference.shape}")
    reference_finite = np.isfinite(reference)
    candidate_finite = np.isfinite(candidate)
    bad = np.argwhere(~candidate_finite)
    if bad.size:
        result = {
            "reference": str(args.reference.expanduser().resolve()),
            "candidate": str(args.candidate.expanduser().resolve()),
            "values": int(reference.size),
            "reference_finite": bool(reference_finite.all()),
            "finite_output": False,
            "nonfinite_values": int((~candidate_finite).sum()),
            "nan_values": int(np.isnan(candidate).sum()),
            "positive_inf_values": int(np.isposinf(candidate).sum()),
            "negative_inf_values": int(np.isneginf(candidate).sum()),
            "first_nonfinite_index": bad[0].tolist(),
            "relative_l2": None,
            "rmse": None,
        }
        print("TEST_RESULT:" + json.dumps(result))
        return

    delta = candidate - reference
    reference_norm = float(np.linalg.norm(reference.reshape(-1)))
    result = {
        "reference": str(args.reference.expanduser().resolve()),
        "candidate": str(args.candidate.expanduser().resolve()),
        "values": int(reference.size),
        "relative_l2": float(np.linalg.norm(delta.reshape(-1)) / max(reference_norm, 1e-12)),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "reference_finite": bool(reference_finite.all()),
        "finite_output": True,
    }
    print("TEST_RESULT:" + json.dumps(result))


if __name__ == "__main__":
    main()
