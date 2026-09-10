"""Diff two --dump-programs fingerprint files to see which captured programs are prompt-independent.

Stage 1 of the single-bin work: every captured program must be byte-identical regardless of
sequence length. Only then can the whole set be dumped once as a bin and replayed for any prompt
with a small GPR-priming preamble (the pattern in models/llama3.2_1b/llama3.2_1b_test.py:1633).

Instruction COUNT being equal is necessary but not sufficient -- a baked loop trip count, an
ADD_SET immediate, or a DRAM base that shifted because buffers are sized by T_pad all change the
digest while leaving the count alone. This reports both so they can be told apart.

    python models/kokoro/compare_programs.py short.json long.json
"""
import json
import sys
from collections import defaultdict


def main(path_a: str, path_b: str) -> int:
    a = json.load(open(path_a))
    b = json.load(open(path_b))

    if len(a) != len(b):
        print(f"program COUNT differs: {len(a)} vs {len(b)} -- the two runs did not emit the "
              f"same set of programs, so per-index comparison is meaningless.\n")

    per_section = defaultdict(lambda: {"same": 0, "count_same_bytes_differ": 0, "count_differs": 0})
    rows = []
    for pa, pb in zip(a, b):
        sec = pa["section"] or "(none)"
        if pa["sha1"] == pb["sha1"]:
            per_section[sec]["same"] += 1
            continue
        if pa["n_inst"] == pb["n_inst"]:
            per_section[sec]["count_same_bytes_differ"] += 1
            why = "same count, operands differ"
        else:
            per_section[sec]["count_differs"] += 1
            why = f"count {pa['n_inst']} -> {pb['n_inst']}"
        rows.append((pa["idx"], sec, why))

    print(f"{'section':<30}{'identical':>11}{'operands':>11}{'count':>9}")
    print("-" * 61)
    tot = defaultdict(int)
    for sec, d in per_section.items():
        print(f"{sec:<30}{d['same']:>11}{d['count_same_bytes_differ']:>11}{d['count_differs']:>9}")
        for k, v in d.items():
            tot[k] += v
    print("-" * 61)
    print(f"{'TOTAL':<30}{tot['same']:>11}{tot['count_same_bytes_differ']:>11}{tot['count_differs']:>9}")
    print("\nidentical = already cacheable as a bin")
    print("operands  = same shape, but a prompt-dependent value is baked into an operand")
    print("count     = still emitting a different number of instructions")

    if rows:
        print(f"\nfirst differing programs:")
        for idx, sec, why in rows[:15]:
            print(f"  #{idx:<4} {sec:<28} {why}")
        if len(rows) > 15:
            print(f"  ... and {len(rows) - 15} more")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
