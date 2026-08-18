#!/usr/bin/env bash
# Local CI runner — mirrors what CI runs: host-only model/driver regressions,
# generic HW op tests, then every model compiled from scratch + run from its
# cached bin, with DRAM randomized (poisoned) before every single pass. Stops
# immediately on the first failure
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
#   ./run_ci.sh --pi05-first      # pi05 first, and the model round BEFORE the HW
#                                  # op tests (used by the nightly full run)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CLEAN_BINS=0
if [[ "${1:-}" == "--clean-bins" ]]; then
    CLEAN_BINS=1
    shift
fi

# --pi05-first: hoist pi05 to the front of the model round AND run that round
# BEFORE user_hw_test.py, so a night run gets its pi05 verdict in ~2 minutes
# instead of after the generic HW ops plus 18 other models. The harness stops on
# the first failure, so an early pi05 failure also skips the rest.
PI05_FIRST=0
if [[ "${1:-}" == "--pi05-first" ]]; then
    PI05_FIRST=1
    shift
fi

ONLY_ARGS=()
if [[ "${1:-}" == "--only" ]]; then
    shift
    ONLY_ARGS=(--only "$@")
fi

FIRST_ARGS=()
[[ $PI05_FIRST -eq 1 ]] && FIRST_ARGS=(--first pi05)

STEP_TOTAL=3
[[ $CLEAN_BINS -eq 1 ]] && STEP_TOTAL=4
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

run_yolov5_host_tests() {
    echo "############################################################"
    echo "# $STEP/$STEP_TOTAL  YOLOv5 host regressions — graph helpers + conv planner"
    echo "############################################################"
    python models/yolov5/test_yolov5_helpers.py
    if [[ $? -ne 0 ]]; then
        echo "!!! YOLOv5 host regressions failed — stopping before hardware tests."
        exit 1
    fi
    STEP=$((STEP + 1))
}

run_hw_tests() {
    echo
    echo "############################################################"
    echo "# $STEP/$STEP_TOTAL  user_hw_test.py — generic hardware op tests"
    echo "############################################################"
    python user_hw_test.py
    if [[ $? -ne 0 ]]; then
        echo "!!! user_hw_test.py failed — stopping."
        exit 1
    fi
    STEP=$((STEP + 1))
}

run_model_tests() {
    echo
    echo "############################################################"
    echo "# $STEP/$STEP_TOTAL  model_auto_test.py — compile + run-from-bin per model"
    echo "#       (DRAM randomized before every pass; stops on first model failure)"
    echo "############################################################"
    python model_auto_test.py "${ONLY_ARGS[@]}" "${FIRST_ARGS[@]}"
    MODEL_STATUS=$?
    STEP=$((STEP + 1))
}

run_yolov5_host_tests

if [[ $PI05_FIRST -eq 1 ]]; then
    # Models FIRST (pi05 hoisted to the head of the round), HW ops after. The
    # model round still runs even if user_hw_test.py would have failed -- that is
    # the point: get the pi05 verdict before spending time on anything else.
    run_model_tests
    run_hw_tests
else
    run_hw_tests
    run_model_tests
fi

echo
echo "############################################################"
echo "# CI SUMMARY"
echo "############################################################"
sed -n '/Summary table/,/^Overall/p' model_auto_test_results.txt

exit $MODEL_STATUS
