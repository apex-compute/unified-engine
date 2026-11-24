#!/usr/bin/env bash
# Kintex-7 leg of Model HW CI. Runs on p620 as github, fed over SSH by the bootcamp
# runner (ci.yml "Run CI suite on p620"):
#
#   ssh ue-p620 "bash -s -- <sha> [run_ci.sh args...]" < .github/scripts/kintex7_remote.sh
#
# Runs in the checkout the p620 runner used, ~/actions-runner/_work/unified-engine/
# unified-engine (UE_P620_DIR overrides), so its models/ caches (weights, HF snapshots,
# ~155 GB) are reused, in the runner's conda env "ue". The board lock is taken by the
# suite itself (pcie_lock.py / claim_dma_device()), as before. Results are copied to
# ~/ue_ci_kintex7/ for the runner to collect.
set -uo pipefail

SHA="${1:?usage: kintex7_remote.sh <commit sha> [run_ci.sh args...]}"
shift
RUN_CI_ARGS=("$@")
UE_DIR="${UE_P620_DIR:-$HOME/actions-runner/_work/unified-engine/unified-engine}"
OUT="$HOME/ue_ci_kintex7"
CONDA_ENV="$HOME/miniconda3/envs/ue"

echo "[kintex7] $(hostname) as $(whoami), $(date -Is), commit $SHA"
rm -rf "$OUT" && mkdir -p "$OUT"
# While this runs, the CI monitor (ci.apexcompute.com) names the suite's processes
# unified-engine-<run#>: it reads this file and checks the pid is alive.
echo "$$ ${UE_CI_RUN_NUMBER:-}" > "$OUT/.ci_run"
[ -d "$UE_DIR/.git" ] || { echo "::error::$UE_DIR not found on $(hostname)"; exit 1; }

# A run whose job was lost (bootcamp rebooted, the runner restarted) leaves its suite
# running here, holding the board. Stop every process of this user working in this
# checkout before starting again: SIGTERM, then SIGKILL after 30 s. This runs before
# the cd below, so this script (cwd $HOME) is not one of them.
mine() {
  local pid cwd
  for pid in $(pgrep -u "$(id -u)"); do
    [ "$pid" = $$ ] && continue
    cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null) || continue
    case "$cwd/" in "$UE_DIR"/*) echo "$pid" ;; esac
  done
}
left=$(mine)
if [ -n "$left" ]; then
  echo "[kintex7] stopping a leftover run: $(echo $left)"
  kill -TERM $left 2>/dev/null
  for _ in $(seq 30); do [ -z "$(mine)" ] && break; sleep 1; done
  left=$(mine); [ -z "$left" ] || kill -KILL $left 2>/dev/null
fi

cd "$UE_DIR" || { echo "::error::$UE_DIR not found on $(hostname)"; exit 1; }
# Fetch by SHA over SSH: origin is the runner's HTTPS URL, which needed its job token.
# A pull request's merge commit is fetchable by SHA too.
git fetch -q git@github.com:apex-compute/unified-engine.git "$SHA" ||
  { echo "::error::git fetch of $SHA failed in $UE_DIR"; exit 1; }
# Keep models/ (weights, HF snapshots); every model recompiles its program image itself.
git reset -q --hard HEAD
git clean -ffdxq -e models
git checkout -q --detach "$SHA" || { echo "::error::cannot check out $SHA"; exit 1; }
echo "[kintex7] unified-engine at $(git log -1 --oneline)"

[ -x "$CONDA_ENV/bin/python" ] || { echo "::error::conda env $CONDA_ENV not found on $(hostname)"; exit 1; }
# The PATH the p620 runner ran jobs with (its .path).
export PATH="$CONDA_ENV/bin:$HOME/miniconda3/condabin:$PATH"

echo "[kintex7] ./run_ci.sh ${RUN_CI_ARGS[*]}"
./run_ci.sh "${RUN_CI_ARGS[@]}" 2>&1 | tee "$OUT/run_ci.log"
RC=${PIPESTATUS[0]}

for f in model_auto_test_results.txt model_auto_test_results_pass1.txt \
         model_auto_test_results_shuffle.txt user_hw_test_summary.md; do
  [ -f "$f" ] && cp -f "$f" "$OUT/"
done
echo "[kintex7] done, exit $RC"
exit $RC
