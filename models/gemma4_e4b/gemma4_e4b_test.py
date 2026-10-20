"""Gemma4 E4B on the Apex accelerator: the Gemma4 E2B implementation, run as E4B.

E4B is the same Gemma 4 architecture as E2B with larger dimensions (hidden
2560, 42 layers, 2 KV heads over 8 query heads, MLP 10240), so it runs on the
E2B code in ../gemma4_e2b (LM, vision, audio, multi-core prefill/decode, the
tiled memory map and chunked prefill) with this directory's config and
artifacts: gemma4_e4b_config.json, gemma4_e4b_bin/ (params.bin, programs.bin,
the HF model) and gemma4_e4b_test_*.md run summaries. Every option of
gemma4_e2b_test.py applies; see ../gemma4_e2b/README.md.

    python models/gemma4_e4b/gemma4_e4b_test.py --dev xdma0 --multi-core 8 --image
"""
import os
import sys

E4B_DIR = os.path.dirname(os.path.abspath(__file__))
E2B_DIR = os.path.join(os.path.dirname(E4B_DIR), "gemma4_e2b")
sys.path.insert(0, E2B_DIR)

import gemma4_e2b_test as _gemma4  # noqa: E402

_gemma4.set_model_variant("gemma4_e4b", E4B_DIR, "Gemma4 E4B")

# The options are gemma4_e2b_test.py's. model_auto_test.py forwards a flag only
# to scripts whose source names it, so the ones it forwards are named here.
FORWARDED_OPTIONS = ("--dev", "--multi-core", "--max-new-tokens", "--prompt",
                     "--image", "--audio", "--bin-reuse")

if __name__ == "__main__":
    _gemma4.main()
