#!/usr/bin/env bash
#
# rescan_xilinx.sh
# Hot-remove and re-add all Xilinx PCIe devices found by lspci.
# Requires root privileges.

set -euo pipefail

# Find all Xilinx devices' BDFs (e.g. 0000:82:00.0)
BDFS=$(lspci -D | grep -i xilinx | awk '{print $1}')

if [[ -z "$BDFS" ]]; then
    echo "No Xilinx PCI device found."
    exit 1
fi

echo "Found Xilinx PCI device(s):"
echo "$BDFS" | while read -r BDF; do
    echo "  - $BDF"
done

# Hot-remove all devices
echo "Removing all Xilinx devices..."
echo "$BDFS" | while read -r BDF; do
    if [[ -e "/sys/bus/pci/devices/$BDF/remove" ]]; then
        echo "  Removing device $BDF..."
        echo 1 | sudo tee "/sys/bus/pci/devices/$BDF/remove" >/dev/null
    else
        echo "  Warning: Device $BDF already removed or not found"
    fi
done

# Wait a moment for devices to be fully removed
sleep 1

# Rescan the PCI bus
echo "Rescanning PCI bus..."
echo 1 | sudo tee /sys/bus/pci/rescan >/dev/null

# Wait a moment for devices to be re-detected
sleep 1

sudo chmod 666 /dev/xdma* 2>/dev/null || true

echo "Done. Current Xilinx devices:"
lspci -nn | grep -i xilinx || echo "None found after rescan."

