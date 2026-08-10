#!/usr/bin/env bash
# Local CI runner — mirrors what CI runs: generic HW op tests, then every
# model compiled from scratch + run from its cached bin, with DRAM randomized
# (poisoned) before every single pass. Stops immediately on the first failure
# — a bad run must not be allowed to keep going and corrupt/mask later results.
#
# Usage:
#   ./run_ci.sh                   # full suite (stops on first failure)
#   ./run_ci.sh --clean-bins      # also wipe cached programs.bin/json first
#                                  # (model_auto_test.py already wipes per-model
#                                  # bins before its own compile pass, so this is
#                                  # mostly redundant — kept for an explicit
#                                  # "start completely clean" run)
#   ./run_ci.sh --only gpt2 swin  # restrict the model round to these names
#   ./run_ci.sh --clean-bins --only gpt2 swin

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CLEAN_BINS=0
if [[ "${1:-}" == "--clean-bins" ]]; then
    CLEAN_BINS=1
    shift
fi

ONLY_ARGS=()
if [[ "${1:-}" == "--only" ]]; then
    shift
    ONLY_ARGS=(--only "$@")
fi

STEP_TOTAL=2
[[ $CLEAN_BINS -eq 1 ]] && STEP_TOTAL=3
STEP=1

if [[ $CLEAN_BINS -eq 1 ]]; then
    echo "############################################################"
    echo "# $STEP/$STEP_TOTAL  clean_program_bins.sh — wipe cached compiled programs"
    echo "############################################################"
    ./clean_program_bins.sh
    if [[ $? -ne 0 ]]; then
        echo "!!! clean_program_bins.sh failed — stopping."
        exit 1
    fi
    STEP=$((STEP + 1))
    echo
fi

echo "############################################################"
echo "# $STEP/$STEP_TOTAL  user_hw_test.py — generic hardware op tests"
echo "############################################################"
python user_hw_test.py
if [[ $? -ne 0 ]]; then
    echo "!!! user_hw_test.py failed — stopping before any model is run."
    exit 1
fi
STEP=$((STEP + 1))

echo
echo "############################################################"
echo "# $STEP/$STEP_TOTAL  model_auto_test.py — compile + run-from-bin per model"
echo "#       (DRAM randomized before every pass; stops on first model failure)"
echo "############################################################"
python model_auto_test.py "${ONLY_ARGS[@]}"
MODEL_STATUS=$?

echo
echo "############################################################"
echo "# CI SUMMARY"
echo "############################################################"
sed -n '/Summary table/,/^Overall/p' model_auto_test_results.txt

exit $MODEL_STATUS
