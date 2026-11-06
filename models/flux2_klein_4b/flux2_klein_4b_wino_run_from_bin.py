#!/usr/bin/env python3
"""flux2_klein_4b_run_from_bin.py with the Winograd VAE: loads the *_wino program sets that
flux2_klein_4b_wino_test.py --compile-only builds. Same CLI (and --serve) as the original runner.

Adopting the Winograd VAE: rename flux2_klein_4b_vae_wino.py over flux2_klein_4b_vae.py and delete the
*_wino_* wrappers.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import flux2_klein_4b_wino_test  # noqa: E402,F401  (swaps the VAE module and the program paths into the test module)
import flux2_klein_4b_run_from_bin as R  # noqa: E402

if __name__ == "__main__":
    sys.exit(R.main())
