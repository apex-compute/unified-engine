#!/usr/bin/env bash
# Closed-loop AlpaSim drive of Alpamayo 2 Edge on the FPGA. Default: the turn scene only.
#
#   models/alpamayo2_edge/sim/run_closed_loop.sh                 # turn on the FPGA
#   models/alpamayo2_edge/sim/run_closed_loop.sh straight        # another scene
#   SEED=1000 COOLDOWN=30 ALPASIM=~/alpasim models/alpamayo2_edge/sim/run_closed_loop.sh turn
#   BACKENDS="device host" ...                                   # also the GPU device-equivalent drive
#   BACKENDS=stock ... straight turn                             # NVIDIA's published model on the GPU (no board)
#   BACKENDS=cached ...                                          # FPGA with the frame cache: only frames not seen in an
#                                                                # earlier plan go through vision (edge_device_cached.py)
#
# If the board dies mid-drive the FPGA driver says so and waits: reset the board, then
# `touch ~/a2e_resume` in another shell. Do not stop this script -- the drive resumes at the same plan.
set -u
SIM_DIR="$(cd "$(dirname "$0")" && pwd)"
ALPASIM="${ALPASIM:-$HOME/alpasim}"
SEED="${SEED:-1000}"
COOLDOWN="${COOLDOWN:-30}"          # idle seconds between FPGA plans (the board overheats back to back)
OUT="${OUT:-$ALPASIM/a2e_closed_loop}"

declare -A SCENES=(
    [straight]="clipgt-02eadd92-02f1-46d8-86fe-a9e338fed0b6"     # 94 m straight
    [turn]="clipgt-07981e6a-22dd-4796-ad2f-1252037ecd28"         # 90 degree right turn, 107 m
)
NAMES=("$@")
[ ${#NAMES[@]} -eq 0 ] && NAMES=(turn)
for n in "${NAMES[@]}"; do
    [ -n "${SCENES[$n]:-}" ] || { echo "unknown scene '$n' (have: ${!SCENES[*]})"; exit 1; }
done
if pgrep -f "alpasim_[d]river.main" >/dev/null; then
    echo "a driver is already running (it holds port 6789); stop it first"; exit 1
fi

mkdir -p "$OUT"
echo "$COOLDOWN" > "$HOME/a2e_cooldown"
rm -f "$HOME/a2e_resume"
DRIVER_PID=""
stop_driver() {
    [ -n "$DRIVER_PID" ] && kill -- -"$DRIVER_PID" 2>/dev/null
    [ -n "${TAIL_PID:-}" ] && kill "$TAIL_PID" 2>/dev/null
    DRIVER_PID=""; TAIL_PID=""
    sleep 3
}
trap 'stop_driver; exit 130' INT TERM

# drive <backend: device|cached|host|stock> <scene name>
drive() {
    local backend=$1 name=$2 tag="$2_$1" ready="engines ready"
    local run="$OUT/$tag" log="$OUT/$tag.driver.log"
    local a2e_backend=$backend equiv=1 cache=0
    [ "$backend" = stock ] && { a2e_backend=host; equiv=0; }      # stock: NVIDIA's published model on the GPU
    [ "$backend" = cached ] && { a2e_backend=device; cache=1; }   # cached: the FPGA, reusing frames seen in earlier plans
    [ "$a2e_backend" = host ] && ready="Starting alpamayo2_edge driver"
    rm -rf "$run" "$OUT/$tag.plans"
    echo; echo "=== $name on $backend: starting the driver ($(date +%T)) ==="
    cd "$ALPASIM" || exit 1
    A2E_BACKEND=$a2e_backend A2E_HOST_EQUIV=$equiv A2E_VISION_CACHE=$cache A2E_SEED=$SEED A2E_DUMP_DIR="$OUT/$tag.plans" \
        setsid uv run --no-sync --project src/driver python -u -m alpasim_driver.main \
        --config-path="$SIM_DIR/host_driver_config" --config-name=alpamayo2_edge > "$log" 2>&1 &
    DRIVER_PID=$!
    tail -n +1 -F "$log" 2>/dev/null | grep --line-buffered -E "^\[device\] (inference|reset|resuming|8 engines|start-up)|^\[host\]|Traceback" &
    TAIL_PID=$!
    until grep -q "$ready" "$log" 2>/dev/null; do
        kill -0 "$DRIVER_PID" 2>/dev/null || { echo "driver exited; see $log"; tail -5 "$log"; stop_driver; return 1; }
        sleep 2
    done
    [ "$a2e_backend" = host ] && sleep 5
    echo "=== $name on $backend: driving ($(date +%T)) ==="
    uv run --no-sync --project src/wizard alpasim_wizard deploy=local topology=1gpu driver=alpamayo2_edge \
        driver_source=external_static wizard.log_dir="$run" "scenes.scene_ids=[\"${SCENES[$name]}\"]" > "$OUT/$tag.sim.log" 2>&1
    local rc=$?
    stop_driver
    echo "=== $name on $backend: simulator exit $rc ($(date +%T)); video: $(ls "$run"/aggregate/videos/all/*.mp4 2>/dev/null | head -1) ==="
    return $rc
}

for name in "${NAMES[@]}"; do
    for backend in ${BACKENDS:-device}; do
        drive "$backend" "$name"
    done
done

echo; echo "=== results (seed $SEED) ==="
"$ALPASIM/.venv/bin/python" - "$OUT" "${NAMES[@]}" <<'EOF'
import sys
from pathlib import Path
import polars as pl

out, names = Path(sys.argv[1]), sys.argv[2:]
keys = ["collision_any", "offroad", "dist_traveled_m", "gt_dist_traveled_m", "dist_to_gt_trajectory", "min_distance_to_obstacle_m"]
for name in names:
    series = {}
    for backend in ("device", "cached", "host", "stock"):
        f = out / f"{name}_{backend}" / "aggregate" / "metrics_unprocessed.parquet"
        if f.exists():
            df = pl.read_parquet(f)
            series[backend] = {k: df.filter(pl.col("name") == k).sort("timestamps_us")["values"].to_list() for k in keys}
    if "device" not in series or len(series) < 2:
        for backend, v in series.items():
            print(f"\n{name} on {backend}: collisions {max(v['collision_any']):.0f}, off-road {max(v['offroad']):.0f}, "
                  f"closest obstacle {min(v['min_distance_to_obstacle_m']):.2f} m, travelled {v['dist_traveled_m'][-1]:.1f} m "
                  f"(recorded {v['gt_dist_traveled_m'][-1]:.1f} m), max sideways {max(v['dist_to_gt_trajectory']):.2f} m")
        if not series:
            print(f"{name}: no finished run")
        continue
    other = next(b for b in ("stock", "host", "cached") if b in series)
    d, h = series["device"], series[other]
    what = {"stock": "GPU (published model)", "host": "GPU (device-equivalent model)", "cached": "FPGA with the frame cache"}[other]
    col = "cache" if other == "cached" else "  GPU"
    print(f"\n{name}: FPGA vs {what}")
    print(f"  collisions {max(d['collision_any']):.0f} vs {max(h['collision_any']):.0f}, off-road {max(d['offroad']):.0f} vs {max(h['offroad']):.0f}, "
          f"closest obstacle {min(d['min_distance_to_obstacle_m']):.2f} vs {min(h['min_distance_to_obstacle_m']):.2f} m, "
          f"recorded drive {d['gt_dist_traveled_m'][-1]:.1f} m")
    print(f"   t (s) | sideways from recorded path (m): FPGA {col} | travelled (m): FPGA  {col}")
    n = min(len(d["dist_traveled_m"]), len(h["dist_traveled_m"]))
    for i in range(0, n, 4):
        print(f"   {i * 0.5:5.1f} | {d['dist_to_gt_trajectory'][i]:38.2f} {h['dist_to_gt_trajectory'][i]:5.2f} | "
              f"{d['dist_traveled_m'][i]:19.1f} {h['dist_traveled_m'][i]:6.1f}")
EOF
echo; echo "=== videos ==="
for name in "${NAMES[@]}"; do
    for backend in ${BACKENDS:-device}; do
        v=$(ls "$OUT/${name}_$backend"/aggregate/videos/all/*.mp4 2>/dev/null | head -1)
        echo "  $name on $backend: ${v:-MISSING}"
    done
done
echo; echo "runs, logs and saved plans are under $OUT"
