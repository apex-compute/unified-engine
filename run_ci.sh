#!/usr/bin/env bash
# Local CI runner — mirrors what CI runs: generic HW op tests, then every
# model, each compiling its program image fresh (every model script's default;
# the run_from_bin entries replay the image their compile entry just wrote),
# with DRAM randomized (poisoned) before every single pass. Stops immediately on
# the first failure — a bad run must not be allowed to keep going and
# corrupt/mask later results.
#
# Usage:
#   ./run_ci.sh                   # full suite (stops on first failure)
#   ./run_ci.sh --only gpt2 swin  # restrict the model round to these names
#   ./run_ci.sh --pi05-first      # pi05 first, and the model round BEFORE the HW
#                                  # op tests (used by the nightly full run)
#   ./run_ci.sh --shuffle         # one round, models in a seeded random order
#                                  # (seed = SHUFFLE_SEED env or today's YYYYMMDD),
#                                  # every model run even after a failure; used by
#                                  # the nightly full run
#   ./run_ci.sh --shuffle-pass    # after the compile round, re-run every model
#                                  # FROM ITS CACHED BIN (--run-from-bin) in a seeded
#                                  # random order
#                                  # (seed = SHUFFLE_SEED env or today's YYYYMMDD),
#                                  # not stopping on failure, with auto-triage that
#                                  # classifies each failure as BIN-RELOAD /
#                                  # CONTAMINATION (prev->model) / FLAKY

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --pi05-first: hoist pi05 to the front of the model round AND run that round
# BEFORE user_hw_test.py, so a night run gets its pi05 verdict in ~2 minutes
# instead of after the generic HW ops plus 18 other models. The harness stops on
# the first failure, so an early pi05 failure also skips the rest.
PI05_FIRST=0
if [[ "${1:-}" == "--pi05-first" ]]; then
    PI05_FIRST=1
    shift
fi

SHUFFLE_PASS=0
if [[ "${1:-}" == "--shuffle-pass" ]]; then
    SHUFFLE_PASS=1
    shift
fi

# --shuffle: the single model round runs in a seeded random order, so each night
# covers different (previous -> model) pairs without a second round, and runs every
# model even after one fails (the exit status still reports the failure).
SHUFFLE=0
if [[ "${1:-}" == "--shuffle" ]]; then
    SHUFFLE=1
    shift
fi

ONLY_ARGS=()
if [[ "${1:-}" == "--only" ]]; then
    shift
    ONLY_ARGS=(--only "$@")
fi

FIRST_ARGS=()
[[ $PI05_FIRST -eq 1 ]] && FIRST_ARGS=(--first pi05)
[[ $SHUFFLE -eq 1 ]] && FIRST_ARGS+=(--shuffle-seed "${SHUFFLE_SEED:-$(date +%Y%m%d)}" --continue-on-fail)

STEP_TOTAL=2
[[ $SHUFFLE_PASS -eq 1 ]] && STEP_TOTAL=$((STEP_TOTAL + 1))
STEP=1

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
    echo "# $STEP/$STEP_TOTAL  model_auto_test.py — every model compiles its program fresh"
    echo "#       (DRAM randomized before every pass; stops on first model failure)"
    echo "############################################################"
    python model_auto_test.py "${ONLY_ARGS[@]}" "${FIRST_ARGS[@]}"
    MODEL_STATUS=$?
    STEP=$((STEP + 1))
}

# Pass 2: every bin is now on disk, so re-run the whole round from cached bins in
# a seeded random order. Cross-model NaN/corruption is state leaking from the
# PREDECESSOR, so a different order every night covers new (prev -> model) pairs
# for free. Runs to completion (no stop-on-first-fail) and triages each failure.
run_shuffle_pass() {
    local seed="${SHUFFLE_SEED:-$(date +%Y%m%d)}"
    echo
    echo "############################################################"
    echo "# $STEP/$STEP_TOTAL  model_auto_test.py — run-from-bin, shuffled order (seed $seed)"
    echo "#       (reproduce with: SHUFFLE_SEED=$seed ./run_ci.sh --shuffle-pass)"
    echo "############################################################"
    cp -f model_auto_test_results.txt model_auto_test_results_pass1.txt 2>/dev/null || true
    python model_auto_test.py "${ONLY_ARGS[@]}" --run-from-bin --shuffle-seed "$seed" --continue-on-fail --triage
    SHUFFLE_STATUS=$?
    cp -f model_auto_test_results.txt model_auto_test_results_shuffle.txt 2>/dev/null || true
    STEP=$((STEP + 1))
}

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

SHUFFLE_STATUS=0
if [[ $SHUFFLE_PASS -eq 1 && $MODEL_STATUS -eq 0 ]]; then
    run_shuffle_pass
elif [[ $SHUFFLE_PASS -eq 1 ]]; then
    echo "!!! compile round failed — skipping the shuffled run-from-bin pass (bins incomplete)."
fi

echo
echo "############################################################"
echo "# CI SUMMARY"
echo "############################################################"
if [[ $SHUFFLE_PASS -eq 1 && -f model_auto_test_results_pass1.txt ]]; then
    echo "--- pass 1: compile (registry order) ---"
    sed -n '/Summary table/,/^Overall/p' model_auto_test_results_pass1.txt
    echo "--- pass 2: run-from-bin, shuffled (seed ${SHUFFLE_SEED:-$(date +%Y%m%d)}) ---"
    sed -n '/Summary table/,/^Overall/p' model_auto_test_results_shuffle.txt
    grep -E "^  (Predecessor|Triage) " model_auto_test_results_shuffle.txt | grep -B1 "Triage" || true
else
    sed -n '/Summary table/,/^Overall/p' model_auto_test_results.txt
fi

[[ $MODEL_STATUS -ne 0 ]] && exit $MODEL_STATUS
exit $SHUFFLE_STATUS
