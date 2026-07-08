#!/usr/bin/env python3
"""Auto-walk debug_stepwise_prefix.py across a range of checkpoints, one
subprocess per checkpoint (keeps every run isolated -- no shared hardware/
DRAM state carried between checkpoints, so a bug at checkpoint N can't
corrupt the measurement at N+1). Prints a heartbeat every ~10s while a
checkpoint is still running (so a real hang is visible live, not just after
the fact), stops immediately on the first real hang/error (non-zero exit),
and flags any run that's suspiciously slower than the running trend well
before it would hit the hard timeout.

Usage:
    python debug_stepwise_prefix_walk.py                      # 0..305, op by op
    python debug_stepwise_prefix_walk.py --start 4 --end 20    # just this op range
    python debug_stepwise_prefix_walk.py --timeout 60 --warn-multiplier 4
    python debug_stepwise_prefix_walk.py --per-layer           # 1 subprocess per LAYER
                                                                 # instead of per op -- same
                                                                 # ~48s fixed weight-load cost
                                                                 # but tests 17 ops per call
                                                                 # instead of 1, so 18 calls
                                                                 # (~15min) instead of 306
                                                                 # (~4hr). Tells you WHICH
                                                                 # LAYER breaks, not which op
                                                                 # within it -- re-walk that
                                                                 # one layer op-by-op after.
    python debug_stepwise_prefix_walk.py --per-layer --layer-start 3 --layer-end 8
"""
import argparse
import os
import re
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pi05_libero_test import _CFG

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DONE_RE = re.compile(r"\] done in ([\d.]+)s")
NAME_RE = re.compile(r"\[\d+/\d+\] (\S+):")


def run_one(idx, timeout, heartbeat_every=10.0):
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, os.path.join(SCRIPT_DIR, "debug_stepwise_prefix.py"),
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
            print(f"  ... checkpoint {idx} still running ({now - t0:.0f}s elapsed, "
                  f"no hang detected yet)", flush=True)
            last_hb = now
    wall_s = time.perf_counter() - t0
    out = "\n".join(lines)

    hw_s = None
    m = DONE_RE.search(out)
    if m:
        hw_s = float(m.group(1))
    name_m = NAME_RE.search(out)
    name = name_m.group(1) if name_m else "?"
    return proc.returncode, name, hw_s, wall_s, out


def checkpoints_per_layer():
    """4 (ln1/qkv_proj/q_permute/kv_cache) + NUM_HEADS (per-head jumps) + 7
    (attn_permute/o_proj_residual/ln2/mlp_lo/mlp_hi/mlp_down/residual2) -- must
    match debug_stepwise_prefix.checkpoint_names()'s per-layer count exactly."""
    return 4 + _CFG["lm"]["num_heads"] + 7


def bisect_range(lo, hi, timeout, heartbeat):
    """Binary search checkpoints [lo, hi] for the first one that hangs.
    Assumes lo-1 (or "start") is known-good and hi is known-bad (the caller
    already confirmed this before calling). Each probe halves the remaining
    window -- ~5 probes to pin an exact op within a 17-checkpoint layer."""
    good, bad = lo - 1, hi
    print(f"Bisecting failing range [{lo}, {hi}] (17-wide layer -> ~{max(1, (hi-lo+1)).bit_length()} probes)...")
    while good + 1 < bad:
        mid = (good + bad) // 2
        half = "first half" if mid < (lo + hi) // 2 else "second half"
        print(f"  probing checkpoint {mid} ({half} of remaining [{good+1}, {bad}])...")
        rc, name, hw_s, wall_s, out = run_one(mid, timeout, heartbeat)
        if rc == 0:
            print(f"    [{mid}] {name}: PASS (hw_exec={hw_s}s) -> hang is after this")
            good = mid
        else:
            print(f"    [{mid}] {name}: HANG -> hang is at or before this")
            print("    --- last 10 lines ---")
            print("\n".join(out.splitlines()[-10:]))
            bad = mid
    rc, name, hw_s, wall_s, out = run_one(bad, timeout, heartbeat)
    print(f"\nBisection result: checkpoint {good} ({'start' if good < lo else 'last known-good op'}) "
          f"is clean; checkpoint {bad} ({name}) is the FIRST op that hangs.")
    return bad, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0, help="op index (ignored with --per-layer)")
    ap.add_argument("--end", type=int, default=305, help="op index (ignored with --per-layer)")
    ap.add_argument("--per-layer", action="store_true",
                     help="1 subprocess per LAYER (its last checkpoint) instead of per op -- "
                          "~18 calls instead of ~306, same fixed weight-load cost per call but "
                          "tests a whole layer's worth of ops each time")
    ap.add_argument("--layer", type=int, default=None,
                     help="test ONE full layer (its last checkpoint); if it hangs, "
                          "automatically binary-searches within that layer's checkpoints "
                          "(~5 extra probes) to pinpoint the exact failing op")
    ap.add_argument("--layer-start", type=int, default=0)
    ap.add_argument("--layer-end", type=int, default=None,
                     help="default: NUM_LAYERS-1 (all layers)")
    ap.add_argument("--timeout", type=float, default=250.0)
    ap.add_argument("--heartbeat", type=float, default=10.0)
    ap.add_argument("--warn-multiplier", type=float, default=5.0,
                     help="flag a checkpoint if its hw exec time exceeds "
                          "this multiple of the running median (once >=5 samples exist)")
    args = ap.parse_args()

    if args.layer is not None:
        per_layer = checkpoints_per_layer()
        lo = args.layer * per_layer
        hi = lo + per_layer - 1
        print(f"Testing layer {args.layer} fully (checkpoint {hi}, covers ops 0..{hi} "
              f"i.e. layers 0..{args.layer} combined)...")
        rc, name, hw_s, wall_s, out = run_one(hi, args.timeout, args.heartbeat)
        if rc == 0:
            print(f"[layer {args.layer}] {name}: CLEAN (hw_exec={hw_s}s, wall={wall_s:.1f}s). "
                  f"Layers 0..{args.layer} are all hang-free.")
            return
        print(f"[layer {args.layer}] {name}: HANG (exit {rc}, wall={wall_s:.1f}s). Bisecting...\n")
        bisect_range(lo, hi, args.timeout, args.heartbeat)
        sys.exit(1)

    if args.per_layer:
        per_layer = checkpoints_per_layer()
        num_layers = _CFG["lm"]["num_layers"]
        layer_end = args.layer_end if args.layer_end is not None else num_layers - 1
        # Each layer's LAST checkpoint index = (L+1)*per_layer - 1; running through
        # it compiles+executes every op in layers [0..L] in one shot (checkpoints
        # are cumulative from op 0), so passing layer L's checkpoint proves every
        # earlier layer AND this one are hang-free together.
        indices = [(l + 1) * per_layer - 1 for l in range(args.layer_start, layer_end + 1)]
        labels = [f"layer{l}_end" for l in range(args.layer_start, layer_end + 1)]
    else:
        indices = list(range(args.start, args.end + 1))
        labels = [str(i) for i in indices]

    hw_times = []
    mode = f"per-layer (layers {args.layer_start}..{layer_end})" if args.per_layer \
        else f"op-by-op ({args.start}..{args.end})"
    print(f"Walking {mode} (timeout={args.timeout}s/call, heartbeat every {args.heartbeat}s)")
    for label, idx in zip(labels, indices):
        rc, name, hw_s, wall_s, out = run_one(idx, args.timeout, args.heartbeat)

        if rc != 0:
            print(f"[{label} -> op {idx}] {name}: HANG or ERROR (exit {rc}, wall={wall_s:.1f}s)")
            print("--- last 15 lines of output ---")
            print("\n".join(out.splitlines()[-15:]))
            print(f"\nSTOPPED at {label} (op {idx}, {name}). "
                  f"Bug is at or before this point" +
                  (f"; re-walk this layer op-by-op to pin the exact op." if args.per_layer else "."))
            sys.exit(1)

        flag = ""
        if hw_s is not None:
            hw_times.append(hw_s)
            if len(hw_times) >= 5:
                med = statistics.median(hw_times[:-1])
                if med > 0 and hw_s > med * args.warn_multiplier:
                    flag = f"  <-- SUSPICIOUS: {hw_s:.1f}s vs running median {med:.2f}s"

        print(f"[{label} -> op {idx}] {name}: clean, hw_exec={hw_s if hw_s is not None else '?'}s, "
              f"wall={wall_s:.1f}s{flag}")

    print(f"\nAll {mode} clean. "
          f"hw_exec times: min={min(hw_times):.2f}s max={max(hw_times):.2f}s "
          f"median={statistics.median(hw_times):.2f}s")


if __name__ == "__main__":
    main()
