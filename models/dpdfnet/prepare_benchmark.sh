#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UE_REPO="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DPDFNET_PYTHON="${DPDFNET_PYTHON:-python3}"
BENCH_DIR="${DPDFNET_BENCHMARK_DIR:-${UE_REPO}/perf_logs/dpdfnet_bittware_256}"
ANDROMEDA_REPO="${ANDROMEDA_REPO:-$(cd -- "${UE_REPO}/.." && pwd)/andromeda}"

echo "Unified Engine: ${UE_REPO}"
echo "Python: ${DPDFNET_PYTHON}"
echo "Benchmark output: ${BENCH_DIR}"

if [[ "${1:-}" == "--check" ]]; then
    test -f "${SCRIPT_DIR}/dpdfnet_prepare.py"
    test -f "${SCRIPT_DIR}/dpdfnet_compile.py"
    "${DPDFNET_PYTHON}" --version
    echo "Preparation script check passed."
    exit 0
fi

mkdir -p "${BENCH_DIR}"
echo "Unified Engine revision: $(git -C "${UE_REPO}" rev-parse HEAD)"
if git -C "${ANDROMEDA_REPO}" rev-parse HEAD >/dev/null 2>&1; then
    echo "Andromeda revision: $(git -C "${ANDROMEDA_REPO}" rev-parse HEAD)"
else
    echo "Andromeda revision unavailable at ${ANDROMEDA_REPO}"
fi

"${DPDFNET_PYTHON}" -m pip install -r "${SCRIPT_DIR}/requirements.txt"
"${DPDFNET_PYTHON}" "${SCRIPT_DIR}/dpdfnet_prepare.py" --download
"${DPDFNET_PYTHON}" "${SCRIPT_DIR}/dpdfnet_compile.py" --force

DPDFNET_BENCHMARK_DIR="${BENCH_DIR}" "${DPDFNET_PYTHON}" - <<'PY'
import hashlib
import os
from pathlib import Path

import numpy as np

output = Path(os.environ["DPDFNET_BENCHMARK_DIR"]) / "input_100_frames.npy"
rng = np.random.default_rng(20260911)
frames = rng.normal(
    0.0, 0.1, size=(100, 1, 1, 161, 2)).astype(np.float32)
np.save(output, frames, allow_pickle=False)
print(f"Benchmark input: {output}")
print(f"SHA256: {hashlib.sha256(output.read_bytes()).hexdigest()}")
PY

echo "Preparation complete."
echo "Run models/dpdfnet/dpdfnet_run_from_bin.py with --device bittware."
echo "CPU comparison: models/dpdfnet/dpdfnet_benchmark_cpu.py"
