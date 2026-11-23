#!/usr/bin/env bash
# Alveo U50 leg of the nightly full model CI. Runs on p2 as github, fed over SSH by the
# p620 runner (ci.yml "Start the Alveo U50 leg on p2"):
#
#   ssh github@<p2> "bash -s -- <sha> [run_ci.sh args...]" < .github/scripts/alveo_u50_remote.sh
#
# Same suite as the kintex7 job (the same run_ci.sh args) on p2's U50 (xdma0, 8 engines),
# in the persistent ~/unified-engine checkout that andromeda's p2 CI also uses, so the
# model weights already cached there are reused. Results are copied to
# ~/ue_ci_alveo_u50/ for the runner to collect.
set -uo pipefail

SHA="${1:?usage: alveo_u50_remote.sh <commit sha> [run_ci.sh args...]}"
shift
RUN_CI_ARGS=("$@")
UE_DIR="$HOME/unified-engine"
OUT="$HOME/ue_ci_alveo_u50"
LOCK=/tmp/pcie_ci_hw_p2.lock   # the lock andromeda's p2 hardware steps take

echo "[alveo_u50] $(hostname) as $(whoami), $(date -Is), commit $SHA"
rm -rf "$OUT" && mkdir -p "$OUT"

# Never share the board with andromeda's p2 jobs: wait for its lock (2 h at most).
exec 9>"$LOCK"
if ! flock -w 7200 9; then
  echo "::error::p2 hardware lock $LOCK still held after 2 h"
  exit 1
fi
echo "[alveo_u50] hardware lock acquired"

cd "$UE_DIR" || { echo "::error::$UE_DIR not found on $(hostname)"; exit 1; }
git fetch -q --prune origin || { echo "::error::git fetch failed in $UE_DIR"; exit 1; }
# Same cleaning as the kintex7 job: keep models/ (weights, HF snapshots). Every
# model recompiles its program image itself.
git reset -q --hard HEAD
git clean -ffdxq -e models
git checkout -q --detach "$SHA" || { echo "::error::commit $SHA not in origin"; exit 1; }
echo "[alveo_u50] unified-engine at $(git log -1 --oneline)"

# The U50 image is maintained outside the release, so its HW version is not the one
# user_dma_core.py asserts. Comment the assert out in this checkout only; the next
# reset --hard restores it. The source line itself stays as andromeda's sync expects.
sed -i -E 's/^([[:space:]]*)(assert hw_version ==)/\1# CI alveo_u50: \2/' user_dma_core.py
if ! grep -q '# CI alveo_u50: assert hw_version ==' user_dma_core.py; then
  echo "::error::HW version assert not found in user_dma_core.py"
  exit 1
fi
echo "[alveo_u50] HW version check skipped for this run"

if [ -f "$HOME/venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$HOME/venv/bin/activate"
else
  echo "::error::$HOME/venv/bin/activate not found on $(hostname)"
  exit 1
fi

export DMA_DEV=xdma0          # p2: U50 on xdma0, kintex7 on xdma1
export UE_CI_CONV_IMAGE=1     # the U50 image carries the conv blocks

echo "[alveo_u50] ./run_ci.sh ${RUN_CI_ARGS[*]}"
./run_ci.sh "${RUN_CI_ARGS[@]}"
RC=$?

for f in model_auto_test_results.txt model_auto_test_results_pass1.txt \
         model_auto_test_results_shuffle.txt user_hw_test_summary.md; do
  [ -f "$f" ] && cp -f "$f" "$OUT/"
done

# Leave the shared checkout as andromeda does: clean, detached at origin/main.
git checkout -q -- user_dma_core.py
git checkout -q --detach origin/main
echo "[alveo_u50] done, exit $RC"
exit $RC
