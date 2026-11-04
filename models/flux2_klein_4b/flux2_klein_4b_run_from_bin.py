#!/usr/bin/env python3
"""Run-only entry point: FLUX.2 [klein] 4B from the pre-built bins.

Loads ``flux2_klein_4b_bin/{params_te.bin, params_dit.bin, vae_weights.pt,
programs.bin, programs.json}`` and runs TE -> DiT -> VAE on the board. Never
compiles and never quantizes; fails if a bin is missing or the programs were
built for a different geometry (recompile with ``flux2_klein_4b_test.py``).

    python models/flux2_klein_4b/flux2_klein_4b_run_from_bin.py --prompt "..." [--height 256 --width 256 --max-seq-len 128]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flux2_klein_4b_test as T  # noqa: E402

if __name__ == "__main__":
    sys.exit(T.main(sys.argv[1:] + ["--no-compile"]))
