#!/usr/bin/env bash
# Bisect harness for the Qwen2.5-VL-3B vision-encoder PBI regression.
#
# Each vision PBI stage can be independently flipped to its legacy (fully-unrolled)
# path via an env var (default ON; "0" = legacy):
#   QWENVL_VIS_PBI_MATMUL   matmul / rms_norm  gpr_M_reg
#   QWENVL_VIS_PBI_POSTADD  rms_norm_core_dram_post_add
#   QWENVL_VIS_PBI_FLASH    flash_attention_core_pbi_fixed
#   QWENVL_VIS_PBI_ROPE     per-token RoPE PBI loop
#
# The encoder bin cache filename is keyed on the active combo (suffix "_no<m|p|f|r>"),
# so each configuration compiles its own bin and never reuses a stale one. That means
# you do NOT need to delete encoder_program*.bin between steps.
#
# Usage:
#   ./bisect_pbi.sh all-legacy     # everything off  -> sanity baseline (should be correct)
#   ./bisect_pbi.sh all-pbi        # everything on   -> reproduces the bug
#   ./bisect_pbi.sh only matmul    # ONLY that stage on, rest legacy
#   ./bisect_pbi.sh drop flash     # everything on EXCEPT that stage (isolates one suspect)
#   ./bisect_pbi.sh raw            # honor whatever QWENVL_VIS_PBI_* are already exported
#
# Recommended order: all-legacy (confirm baseline), then `only <stage>` for each of
# matmul/postadd/flash/rope. The first `only` that breaks is the culprit.

set -euo pipefail
cd "$(dirname "$0")"

STAGES=(MATMUL POSTADD FLASH ROPE)
declare -A FLAG=([MATMUL]=1 [POSTADD]=1 [FLASH]=1 [ROPE]=1)

mode="${1:-all-pbi}"
case "$mode" in
  all-legacy) for s in "${STAGES[@]}"; do FLAG[$s]=0; done ;;
  all-pbi)    : ;;  # all default 1
  only)
    target="${2:?usage: only <matmul|postadd|flash|rope>}"
    for s in "${STAGES[@]}"; do FLAG[$s]=0; done
    FLAG[${target^^}]=1 ;;
  drop)
    target="${2:?usage: drop <matmul|postadd|flash|rope>}"
    FLAG[${target^^}]=0 ;;
  raw) : ;;  # leave to environment below
  *) echo "unknown mode: $mode"; sed -n '2,30p' "$0"; exit 1 ;;
esac

if [[ "$mode" != "raw" ]]; then
  export QWENVL_VIS_PBI_MATMUL=${FLAG[MATMUL]}
  export QWENVL_VIS_PBI_POSTADD=${FLAG[POSTADD]}
  export QWENVL_VIS_PBI_FLASH=${FLAG[FLASH]}
  export QWENVL_VIS_PBI_ROPE=${FLAG[ROPE]}
fi

echo "=================================================================="
echo " PBI bisect config:"
echo "   MATMUL=${QWENVL_VIS_PBI_MATMUL:-1}  POSTADD=${QWENVL_VIS_PBI_POSTADD:-1}" \
     " FLASH=${QWENVL_VIS_PBI_FLASH:-1}  ROPE=${QWENVL_VIS_PBI_ROPE:-1}"
echo "=================================================================="

shift || true
[[ "$mode" == "only" || "$mode" == "drop" ]] && shift || true

python qwen2.5_vl_3b_test.py "$@"
