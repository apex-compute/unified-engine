"""Localise prompt-dependent operands: which instruction, which 32-bit word, what changed.

compare_programs.py says WHICH programs differ; this says WHERE inside them, which is what you
need to trace a baked immediate back to the line that emitted it.
"""
import json, sys
from collections import Counter


def words(hexstr):
    b = bytes.fromhex(hexstr)
    return [int.from_bytes(b[i:i + 4], "little") for i in range(0, len(b), 4)]


def main(pa, pb, limit=12):
    A, B = json.load(open(pa)), json.load(open(pb))
    sites = Counter()
    examples = {}
    for x, y in zip(A, B):
        if x["sha1"] == y["sha1"] or not x.get("bytes"):
            continue
        n = len(x["inst"])
        for i in range(n):
            if x["inst"][i] == y["inst"][i]:
                continue
            wa = words(x["bytes"])[i * 8:(i + 1) * 8]
            wb = words(y["bytes"])[i * 8:(i + 1) * 8]
            for w, (va, vb) in enumerate(zip(wa, wb)):
                if va != vb:
                    key = (x["section"], i - n, w)      # index from END: stable across programs
                    sites[key] += 1
                    examples.setdefault(key, (va, vb, x["idx"]))
    print(f"{'section':22} {'inst(from end)':>15} {'word':>5} {'count':>6}   short -> long")
    print("-" * 78)
    for (sec, i, w), c in sites.most_common(limit):
        va, vb, idx = examples[(sec, i, w)]
        d = vb - va
        print(f"{sec:22} {i:>15} {w:>5} {c:>6}   0x{va:X} -> 0x{vb:X}  (delta {d:+d})")
    print(f"\ntotal differing (instruction, word) sites: {len(sites)}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
