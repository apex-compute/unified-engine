#!/usr/bin/env python3
"""flux2_klein_4b_server.py serving the Winograd VAE: lists only the *_wino program sets and runs them
through flux2_klein_4b_wino_run_from_bin.py. Same CLI as the original server.

    python models/flux2_klein_4b/flux2_klein_4b_wino_server.py --port 8080
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import flux2_klein_4b_server as S  # noqa: E402

S.RUNNER = os.path.join(HERE, "flux2_klein_4b_wino_run_from_bin.py")
_all_configs = S.scan_configs


def scan_configs() -> list[dict]:
    return [c for c in _all_configs() if c["id"].endswith("_wino")]


S.scan_configs = scan_configs

if __name__ == "__main__":
    sys.exit(S.main())
