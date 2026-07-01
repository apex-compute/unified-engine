#!/usr/bin/env bash
# Delete entire models/*/*_bin directories (weights, HF cache, programs.bin/json
# — everything), forcing the next test run to redownload weights from scratch
# AND recompile. More aggressive than clean_program_bins.sh, which only clears
# the compiled programs and leaves weights untouched.
#
# Usage:
#   ./clean_model_bin_dirs.sh           # delete
#   ./clean_model_bin_dirs.sh --dry-run # list what would be deleted

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

mapfile -t DIRS < <(find models -maxdepth 2 -type d -name "*_bin")

if [[ ${#DIRS[@]} -eq 0 ]]; then
    echo "No */_bin directories found."
    exit 0
fi

for d in "${DIRS[@]}"; do
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "would delete: $d"
    else
        rm -rf -- "$d"
        echo "deleted: $d"
    fi
done

echo "${#DIRS[@]} dir(s) $([[ $DRY_RUN -eq 1 ]] && echo "would be deleted" || echo "deleted")."
