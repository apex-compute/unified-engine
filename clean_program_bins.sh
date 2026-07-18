#!/usr/bin/env bash
# Canonical `clean` for unified-engine — the single source of truth for wiping
# generated build artifacts so the next test run recompiles from scratch. The
# Makefile `clean` target and CI (ci.yml / run_ci.sh) both call this script.
#
# This is a faithful, line-for-line port of what the unified-engine Makefile
# `clean` target used to do (the accurate, hardware-validated reference for THIS
# repo — not andromeda's make clean, whose models/ only covers gemma3). The
# per-model handling is deliberately NON-uniform and must stay that way:
#   * Most LM models keep params.bin AND params.json (only programs*/instruction
#     artifacts are removed) — regenerating weights is expensive.
#   * mobilenetv2 / mobilenetv2_ssd / parakeet ALSO delete params*.bin/json —
#     those models fail if a stale params survives.
#   * gpt2 / mobilesam / swin delete the ENTIRE <model>_bin/ dir — same reason.
# Keep this in sync with the model scripts' output filenames when models change.
#
# Usage:
#   ./clean_program_bins.sh           # delete
#   ./clean_program_bins.sh --dry-run # list what would be deleted

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# del <paths/globs...> — remove each (rm -rf), or in dry-run just list the ones
# that actually exist. Globs are expanded by the caller; unmatched patterns are
# passed through literally and harmlessly skipped.
del() {
    local p
    for p in "$@"; do
        if [[ -e "$p" || -L "$p" ]]; then
            if [[ $DRY_RUN -eq 1 ]]; then
                echo "would delete: $p"
            else
                rm -rf -- "$p" && echo "deleted: $p"
            fi
        fi
    done
}

# # --- repo-root build artifacts ---------------------------------------------
# del user_dma_core                       # compiled C binary from `make all` (NOT user_dma_core.py)
# del *.mcs *.prm *.jou *.log *.json *.csv
# del mask_point.png
# del model_auto_test_results.txt
# del andromeda_IP-* andromeda_wrapper-*
# del codebooks/

# --- gemma3 (IF4 + IF8): keep params.* ; drop programs/instruction ----------
del models/gemma3/gemma3_bin/programs*.bin
del models/gemma3/gemma3_bin/programs*.json
del models/gemma3/gemma3_bin/*instruction.bin
del models/gemma3/gemma3_bin/*instruction.json
del models/gemma3/gemma3_if8_bin/programs*.bin
del models/gemma3/gemma3_if8_bin/programs*.json
del models/gemma3/gemma3_if8_bin/*instruction.bin
del models/gemma3/gemma3_if8_bin/*instruction.json

# --- gemma4 e2b / e4b: keep params.* ; drop programs/instruction ------------
del models/gemma4_e2b/gemma4_e2b_bin/programs*.bin
del models/gemma4_e2b/gemma4_e2b_bin/programs*.json
del models/gemma4_e2b/gemma4_e2b_bin/*instruction*.bin
del models/gemma4_e2b/gemma4_e2b_bin/*instruction*.json
del models/gemma4_e4b/gemma4_e4b_bin/programs*.bin
del models/gemma4_e4b/gemma4_e4b_bin/programs*.json
del models/gemma4_e4b/gemma4_e4b_bin/*instruction*.bin
del models/gemma4_e4b/gemma4_e4b_bin/*instruction*.json

# --- gpt2: full bin dir wipe ------------------------------------------------
del models/gpt2/gpt2_bin/

# --- llama3.2 1b / 3b: keep params.* ; drop programs/instruction ------------
del models/llama3.2_1b/llama3.2_1b_bin/programs*.bin
del models/llama3.2_1b/llama3.2_1b_bin/programs*.json
del models/llama3.2_1b/llama3.2_1b_bin/llama_instruction*
del models/llama3.2_1b/llama3.2_1b_bin/llama_profile_instruction*
del models/llama3.2_3b/llama3.2_3b_bin/programs*.bin
del models/llama3.2_3b/llama3.2_3b_bin/programs*.json
del models/llama3.2_3b/llama3.2_3b_bin/llama3.2_3b_instruction_fpgapenalty.bin

# --- locateanything_3b: keep params.* ; drop programs + output boxes --------
del models/locateanything_3b/locateanything_3b_bin/programs*.bin
del models/locateanything_3b/locateanything_3b_bin/programs*.json
del models/locateanything_3b/*_boxes_fpga.jpg

# --- mobilenetv2 (+ ssd_fpnlite): ALSO drop params.* + labels + outputs -----
del models/mobilenetv2/mobilenetv2_bin/programs*.bin
del models/mobilenetv2/mobilenetv2_bin/programs*.json
del models/mobilenetv2/mobilenetv2_bin/params*.bin
del models/mobilenetv2/mobilenetv2_bin/params*.json
del models/mobilenetv2/mobilenetv2_bin/labels.json
del models/mobilenetv2/mobilenetv2_ssd_fpnlite_bin/programs*.bin
del models/mobilenetv2/mobilenetv2_ssd_fpnlite_bin/programs*.json
del models/mobilenetv2/mobilenetv2_ssd_fpnlite_bin/params*.bin
del models/mobilenetv2/mobilenetv2_ssd_fpnlite_bin/params*.json
del models/mobilenetv2/vette_detections_hw.jpg

# --- mobilesam: full bin dir wipe -------------------------------------------
del models/mobilesam/mobilesam_bin/

# --- parakeet: ALSO drop params.* -------------------------------------------
del models/parakeet/parakeet_bin/params.bin
del models/parakeet/parakeet_bin/params.json
del models/parakeet/parakeet_bin/programs*.bin
del models/parakeet/parakeet_bin/programs*.json

# --- qwen2.5_vl_3b: keep params.* ; drop programs ---------------------------
del models/qwen2.5_vl_3b/qwen2.5_vl_3b_bin/programs.bin
del models/qwen2.5_vl_3b/qwen2.5_vl_3b_bin/programs.json

# --- qwen3 1.7b / 4b: keep params.* ; drop programs/instruction -------------
del models/qwen3_1.7b/qwen3_1.7b_bin/qwen3_1.7b_instruction*
del models/qwen3_1.7b/qwen3_1.7b_bin/programs.bin
del models/qwen3_1.7b/qwen3_1.7b_bin/programs.json
del models/qwen3_4b/qwen3_4b_bin/qwen3_4b_instruction*
del models/qwen3_4b/qwen3_4b_bin/programs.bin
del models/qwen3_4b/qwen3_4b_bin/programs.json

# --- qwen3.5_2b: keep params.* ; drop programs ------------------------------
del models/qwen3.5_2b/qwen3.5_2b_bin/programs.bin
del models/qwen3.5_2b/qwen3.5_2b_bin/programs.json

# --- smolvlm2: keep params.* ; drop programs + decoder_program --------------
del models/smolvlm2/smolvlm2_bin/programs*.bin
del models/smolvlm2/smolvlm2_bin/programs*.json
del models/smolvlm2/smolvlm2_bin/decoder_program.bin
del models/smolvlm2/smolvlm2_bin/decoder_program.json

# --- swin: full bin dir wipe ------------------------------------------------
del models/swin/swin_bin/

echo "clean done."
