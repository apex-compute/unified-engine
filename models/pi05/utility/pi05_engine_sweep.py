#!/usr/bin/env python3
"""Per-stage execution time vs engine count -- the scaling curve.

A stage that is memory-bound stops improving once the engines on a memory port
have saturated it; a stage that is serial-bound stops improving because the extra
engines have nothing to do. Both look like "low device utilization" in a single
12-engine run and only the scaling curve tells them apart, so this runs the same
inference at several engine counts and tabulates the per-stage times.

    python models/pi05/utility/pi05_engine_sweep.py --counts 1,2,4,8,12
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

STAGE_RE = re.compile(r"^\s+⚡ (\w+)[^\n]*?\s([\d.]+)s\s*$")
RATE_RE = re.compile(r"hw-issued\s*:\s*([\d.]+) GFLOP\s+([\d.]+) GFLOP/s\s+\[\s*(\d+)% peak\]")


def run_one(n, python, extra):
    tag = "max" if n == "max" else str(n)
    cmd = [python, "-u", os.path.join(ROOT, "models/pi05/pi05_test.py"),
           "--engines", tag] + extra
    print(f"  running --engines {tag} ...", flush=True)
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=3600)
    out = p.stdout + p.stderr
    stages, cur = {}, None
    for line in out.splitlines():
        m = STAGE_RE.match(line)
        if m:
            cur = m.group(1)
            stages[cur] = {"seconds": float(m.group(2))}
            continue
        m = RATE_RE.search(line)
        if m and cur:
            stages[cur]["gflop"] = float(m.group(1))
            stages[cur]["gflops"] = float(m.group(2))
            stages[cur]["pct_peak"] = int(m.group(3))
    ok = "nan=False inf=False" in out
    if not stages:
        print("    !! no stage timings parsed; tail of output:")
        print("\n".join(out.splitlines()[-15:]))
    return {"engines": tag, "ok": ok, "stages": stages, "raw_tail": out.splitlines()[-5:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default="1,2,4,8,max")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default="pi05_engine_sweep.json")
    args, extra = ap.parse_known_args()

    counts = [c.strip() for c in args.counts.split(",")]
    results = []
    for c in counts:
        n = c if c == "max" else int(c)
        try:
            results.append(run_one(n, args.python, extra))
        except subprocess.TimeoutExpired:
            print(f"    !! --engines {c} timed out")
            results.append({"engines": c, "ok": False, "stages": {}})

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    stages = ["vision", "prefix", "denoise"]
    print("\n" + "=" * 78)
    print("pi0.5 per-stage execution time (seconds) vs engine count")
    print("=" * 78)
    print(f"{'engines':>8}" + "".join(f"{s:>14}" for s in stages) + f"{'total':>10}{'ok':>5}")
    base = {}
    for r in results:
        row = f"{r['engines']:>8}"
        tot = 0.0
        for s in stages:
            v = r["stages"].get(s, {}).get("seconds")
            row += f"{v:>14.2f}" if v is not None else f"{'-':>14}"
            tot += v or 0.0
        print(row + f"{tot:>10.2f}{('y' if r['ok'] else 'n'):>5}")
        if not base:
            base = {s: r["stages"].get(s, {}).get("seconds") for s in stages}

    print("\nspeedup vs 1 engine (higher is better; 12 would be perfect scaling)")
    print(f"{'engines':>8}" + "".join(f"{s:>14}" for s in stages))
    for r in results:
        row = f"{r['engines']:>8}"
        for s in stages:
            v = r["stages"].get(s, {}).get("seconds")
            b = base.get(s)
            row += f"{b/v:>13.2f}x" if (v and b) else f"{'-':>14}"
        print(row)

    print("\n% of the 12-engine compute ceiling (pi05_test's own hw-issued figure)")
    print(f"{'engines':>8}" + "".join(f"{s:>14}" for s in stages))
    for r in results:
        row = f"{r['engines']:>8}"
        for s in stages:
            v = r["stages"].get(s, {}).get("pct_peak")
            row += f"{v:>13}%" if v is not None else f"{'-':>14}"
        print(row)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
