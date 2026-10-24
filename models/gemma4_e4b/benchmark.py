"""Fixed-shape benchmark tiers for gemma4_e4b: the gemma4_e2b tiers, run on E4B.

The same workload shapes as ../gemma4_e2b/benchmark.py (0 / 1 / 4 camera
frames, 2.55 / 4.5 / 6 s of audio, 850 / 2048 / 6144-token prompts, 32 / 64 /
128 decoded tokens), driving gemma4_e4b_test.py.

    python models/gemma4_e4b/benchmark.py --low | --medium | --high | --all
"""
import os
import sys

E4B_DIR = os.path.dirname(os.path.abspath(__file__))
E2B_DIR = os.path.join(os.path.dirname(E4B_DIR), "gemma4_e2b")
sys.path.insert(0, E2B_DIR)

import benchmark as _bench  # noqa: E402  (gemma4_e2b/benchmark.py)

_bench.MODEL_NAME = "gemma4_e4b"
_bench.SCRIPT_DIR = E4B_DIR
_bench.MAIN_SCRIPT = os.path.join(E4B_DIR, "gemma4_e4b_test.py")

if __name__ == "__main__":
    _bench.main()
