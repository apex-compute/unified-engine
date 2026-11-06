#!/usr/bin/env python3
"""flux2_klein_4b_test.py with the Winograd VAE (flux2_klein_4b_vae_wino.py) swapped in.

Same CLI, same text encoder / DiT / sampler; only the VAE phase's 3x3 convs change (routed through
UnifiedEngine.conv2d_auto). Programs go to flux2_klein_4b_bin/programs/<config>_e<N>_wino.{bin,json}
so they never replace the baseline's.

    python models/flux2_klein_4b/flux2_klein_4b_wino_test.py --height 256 --width 256 --max-seq-len 64 --engines 8

Adopting it: rename flux2_klein_4b_vae_wino.py over flux2_klein_4b_vae.py and delete this file.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import flux2_klein_4b_test as T  # noqa: E402
import flux2_klein_4b_vae_wino  # noqa: E402

T.flux2_klein_4b_vae = flux2_klein_4b_vae_wino
_baseline_program_paths = T.program_paths


def program_paths(cfg, geo, engines):
    bin_p, meta_p = _baseline_program_paths(cfg, geo, engines)
    return os.path.splitext(bin_p)[0] + "_wino.bin", os.path.splitext(meta_p)[0] + "_wino.json"


T.program_paths = program_paths

if __name__ == "__main__":
    sys.exit(T.main())
