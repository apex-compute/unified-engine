#!/usr/bin/env bash
# update_fpga.sh - one-command FPGA flash update, no PC reboot or power cycle.
#
# Called with no arguments it picks up the update_*.bin next to this script and
# runs every step:
#
#   0. python3 update_flash.py --is-current <bin>    skip everything if the FPGA
#                                                    already runs that image
#   1. python3 update_flash.py <bin> --no-rescan     program + verify flash,
#                                                    then ICAPE2 IPROG warm boot
#   2. sleep $BOOT_WAIT; sudo ./rescan_xilinx.sh     PCIe hot remove + rescan
#   3. python3 update_flash.py --verify-image <bin>  confirm the FPGA now runs
#                                                    the flashed image
#
# The rescan is the one step that needs root, so it runs as its own explicit
# sudo step between the two python invocations instead of from inside
# update_flash.py.
#
# If the running image predates the icap_warmboot block there is nothing to
# warm boot into: the flash is written, the user is told to cold reboot once,
# and we stop before the rescan (step 2/3 would only probe a dead PCIe link).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEV="${DEV:-xdma0}"
BOOT_WAIT="${BOOT_WAIT:-8}"   # BPI16 reconfigure is ~3-4 s; add margin

EXIT_NO_WARMBOOT=3            # must match update_flash.py

usage() {
    cat <<EOF
usage: ./update_fpga.sh [options] [update_<hash>.bin]

With no options: check the version, and if the FPGA is out of date program +
verify the flash, ICAPE2 warm boot, PCIe rescan (sudo), then confirm the
running version. The bin defaults to the newest update_*.bin in the repo root.

  --bin PATH      update image to flash (same as the positional argument)
  --check         report device ID, flash status and version, then exit
  --boot          trigger a warm reboot + rescan only; no flashing
  --force         flash even if the version check says already up to date
  -h, --help      this message

Environment overrides: DEV=xdma1 (default xdma0), BOOT_WAIT=8.
EOF
}

MODE="update"
FORCE=0
BIN=""
while [ $# -gt 0 ]; do
    case "$1" in
        --bin)       BIN="${2:-}"; [ -n "$BIN" ] || { echo "ERROR: --bin needs a path" >&2; exit 1; }; shift ;;
        --check)     MODE="check" ;;
        --boot|--boot-only|--warm-boot) MODE="boot" ;;
        --force)     FORCE=1 ;;
        -h|--help)   usage; exit 0 ;;
        -*)          echo "ERROR: unknown option: $1" >&2; usage >&2; exit 1 ;;
        *)           BIN="$1" ;;
    esac
    shift
done

# --- warm reboot trigger only -------------------------------------------------
if [ "$MODE" = "boot" ]; then
    rc=0
    python3 update_flash.py --boot-only --no-rescan --dev "$DEV" || rc=$?
    if [ "$rc" -eq "$EXIT_NO_WARMBOOT" ]; then
        exit "$EXIT_NO_WARMBOOT"
    elif [ "$rc" -ne 0 ]; then
        exit "$rc"
    fi
    echo ">> waiting ${BOOT_WAIT}s for the FPGA to reconfigure from flash..."
    sleep "$BOOT_WAIT"
    sudo ./rescan_xilinx.sh
    exit 0
fi

# --- version / status check ---------------------------------------------------
if [ "$MODE" = "check" ]; then
    exec python3 update_flash.py --check --dev "$DEV"
fi

# --- full update --------------------------------------------------------------
if [ -z "$BIN" ]; then
    BIN="$(ls -t update_*.bin 2>/dev/null | head -n1)"
fi
if [ -z "$BIN" ]; then
    echo "ERROR: no update_*.bin found in $SCRIPT_DIR and none given" >&2
    usage >&2
    exit 1
fi
if [ ! -f "$BIN" ]; then
    echo "ERROR: $BIN not found" >&2
    exit 1
fi
echo ">> update image: $BIN   device: /dev/${DEV}_user"

if [ "$FORCE" -eq 0 ]; then
    if python3 update_flash.py --is-current "$BIN" --dev "$DEV"; then
        echo ">> nothing to do (use --force to reflash anyway)"
        exit 0
    fi
fi

rc=0
python3 update_flash.py "$BIN" --no-rescan --dev "$DEV" || rc=$?
if [ "$rc" -eq "$EXIT_NO_WARMBOOT" ]; then
    # Flash is written and the new image carries the block; the user owes us one
    # cold boot. Skipping the rescan is deliberate - see the header.
    exit "$EXIT_NO_WARMBOOT"
elif [ "$rc" -ne 0 ]; then
    exit "$rc"
fi

echo ">> waiting ${BOOT_WAIT}s for the FPGA to reconfigure from flash..."
sleep "$BOOT_WAIT"
sudo ./rescan_xilinx.sh

python3 update_flash.py --verify-image "$BIN" --dev "$DEV"
