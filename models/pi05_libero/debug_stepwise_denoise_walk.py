#!/usr/bin/env python3
"""Auto-walk debug_stepwise_denoise.py across a range of checkpoints, one
subprocess per checkpoint (isolated state). Mirrors debug_stepwise_prefix_walk.py.

Each checkpoint costs ~190s fixed (vision + prefix rerun to populate a real K/V
cache -- the denoise NaN doesn't reproduce with placeholder K/V), so default to
bisecting a known-bad range rather than walking every checkpoint linearly.

Usage:
    python debug_stepwise_denoise_walk.py --bisect --lo 0 --hi 22   # step0 only, ~5 probes
    python debug_stepwise_denoise_walk.py --start 0 --end 22        # linear, op by op
"""
import argparse
import os
import re
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DONE_RE = re.compile(r"nan=(\d+) inf=(\d+)")
NAME_RE = re.compile(r"\[\d+/\d+\] (\S+):")


def run_one(idx, timeout, heartbeat_every=10.0):
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, os.path.join(SCRIPT_DIR, "debug_stepwise_denoise.py"),
         "--stop-after", str(idx), "--timeout", str(timeout)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        cwd=SCRIPT_DIR,
    )
    lines = []
    last_hb = t0
    while True:
        line = proc.stdout.readline()
        if line:
            lines.append(line.rstrip("\n"))
        elif proc.poll() is not None:
            break
        now = time.perf_counter()
        if now - last_hb >= heartbeat_every:
            print(f"  ... checkpoint {idx} still running ({now - t0:.0f}s elapsed)", flush=True)
            last_hb = now
    wall_s = time.perf_counter() - t0
    out = "\n".join(lines)

    n_nan = None
    m = DONE_RE.search(out)
    if m:
        n_nan = int(m.group(1))
    name_m = NAME_RE.search(out)
    name = name_m.group(1) if name_m else "?"
    return proc.returncode, name, n_nan, wall_s, out


def bisect_range(lo, hi, timeout, heartbeat):
    """Binary search checkpoints [lo, hi] for the first one with nan>0.
    Assumes lo-1 is known-clean and hi is known-bad."""
    good, bad = lo - 1, hi
    print(f"Bisecting [{lo}, {hi}] for first NaN checkpoint...")
    while good + 1 < bad:
        mid = (good + bad) // 2
        print(f"  probing checkpoint {mid} (remaining [{good+1}, {bad}])...")
        rc, name, n_nan, wall_s, out = run_one(mid, timeout, heartbeat)
        if rc != 0:
            print(f"    [{mid}] {name}: HANG/ERROR (exit {rc}, wall={wall_s:.1f}s) -- treating as bad")
            print("\n".join(out.splitlines()[-10:]))
            bad = mid
            continue
        is_bad = n_nan is not None and n_nan > 0
        print(f"    [{mid}] {name}: nan={n_nan} ({'BAD' if is_bad else 'clean'}, wall={wall_s:.1f}s)")
        if is_bad:
            bad = mid
        else:
            good = mid
    rc, name, n_nan, wall_s, out = run_one(bad, timeout, heartbeat)
    print(f"\nBisection result: checkpoint {good} is clean; checkpoint {bad} ({name}, nan={n_nan}) "
          f"is the FIRST checkpoint with NaN.")
    return bad, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bisect", action="store_true")
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=22)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=22)
    ap.add_argument("--timeout", type=float, default=250.0)
    ap.add_argument("--heartbeat", type=float, default=10.0)
    args = ap.parse_args()

    if args.bisect:
        bisect_range(args.lo, args.hi, args.timeout, args.heartbeat)
        return

    for idx in range(args.start, args.end + 1):
        rc, name, n_nan, wall_s, out = run_one(idx, args.timeout, args.heartbeat)
        if rc != 0:
            print(f"[{idx}] {name}: HANG/ERROR (exit {rc}, wall={wall_s:.1f}s)")
            print("--- last 15 lines ---")
            print("\n".join(out.splitlines()[-15:]))
            sys.exit(1)
        is_bad = n_nan is not None and n_nan > 0
        print(f"[{idx}] {name}: nan={n_nan} {'<-- FIRST BAD' if is_bad else ''} wall={wall_s:.1f}s")
        if is_bad:
            print(f"\nSTOPPED at checkpoint {idx} ({name}). Bug is at or before this op.")
            sys.exit(1)

    print(f"\nAll checkpoints [{args.start}, {args.end}] clean.")


if __name__ == "__main__":
    main()
