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
#   ./run_ci.sh --skip-hw-test --only swin
#                                  # skip the generic HW op round and go straight
#                                  # to the model(s) — for fast single-model
#                                  # iteration only, never for a merge gate
#
# --only must come last (it consumes the rest of the argv).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CLEAN_BINS=0
SKIP_HW_TEST=0
ONLY_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean-bins)   CLEAN_BINS=1; shift ;;
        --skip-hw-test) SKIP_HW_TEST=1; shift ;;
        --only)         shift; ONLY_ARGS=(--only "$@"); break ;;
        *) echo "!!! unknown argument: $1" >&2; exit 2 ;;
    esac
done

STEP_TOTAL=1
[[ $CLEAN_BINS -eq 1 ]] && STEP_TOTAL=$((STEP_TOTAL + 1))
[[ $SKIP_HW_TEST -eq 0 ]] && STEP_TOTAL=$((STEP_TOTAL + 1))
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

if [[ $SKIP_HW_TEST -eq 1 ]]; then
    echo "### --skip-hw-test: skipping user_hw_test.py (generic hardware op tests)"
else
    echo "############################################################"
    echo "# $STEP/$STEP_TOTAL  user_hw_test.py — generic hardware op tests"
    echo "############################################################"
    python user_hw_test.py
    if [[ $? -ne 0 ]]; then
        echo "!!! user_hw_test.py failed — stopping before any model is run."
        exit 1
    fi
    STEP=$((STEP + 1))
fi

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
