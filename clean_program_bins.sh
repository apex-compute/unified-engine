#!/usr/bin/env bash
# Delete cached compiled-program files (programs.bin / programs.json) from
# every models/*/*_bin directory, forcing the next test run to recompile from
# scratch instead of reusing a cached image. Leaves weights (params.bin/json,
# *.pt) and HF cache dirs untouched.
#
# Usage:
#   ./clean_program_bins.sh           # delete
#   ./clean_program_bins.sh --dry-run # list what would be deleted

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

mapfile -t FILES < <(find models -type d -name "*_bin" -prune -exec find {} -type f \( -name "programs.bin" -o -name "programs.json" \) \;)

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No cached programs.bin/programs.json files found."
    exit 0
fi

for f in "${FILES[@]}"; do
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "would delete: $f"
    else
        rm -f -- "$f"
        echo "deleted: $f"
    fi
done

echo "${#FILES[@]} file(s) $([[ $DRY_RUN -eq 1 ]] && echo "would be deleted" || echo "deleted")."
