# XDMA Driver Setup and Usage Guide

This guide covers installation and usage of the Xilinx XDMA driver for PCIe-based FPGA communication.

## Prerequisites

- Kernel headers installed: `sudo apt install linux-headers-$(uname -r)`

## Installation

### 1. Install XDMA Driver from Xilinx Repository

Clone the official Xilinx DMA driver repository:
```bash
git clone https://github.com/Xilinx/dma_ip_drivers.git
cd dma_ip_drivers/XDMA/linux-kernel/xdma
sudo make install
```

### 2. Load the Driver

Load the XDMA driver with interrupt mode 0 (auto-detect):
```bash
sudo insmod /lib/modules/$(uname -r)/xdma/xdma.ko interrupt_mode=0
```

### 3. Load the Driver Every Boot Automatically

Apply the following script
```bash
# 1. Remove any conflicting configs
sudo rm -f /etc/modprobe.d/blacklist-xdma.conf \
           /etc/modprobe.d/xdma.conf \
           /etc/modules-load.d/xdma.conf

# 2. Create systemd service
sudo tee /etc/systemd/system/xdma.service << 'EOF'
[Unit]
Description=Xilinx XDMA Driver
After=local-fs.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c '/sbin/insmod /lib/modules/$(uname -r)/xdma/xdma.ko || true'
ExecStartPost=/bin/sh -c 'chmod 666 /dev/xdma*'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

# 3. Enable and start
sudo systemctl daemon-reload
sudo systemctl enable xdma
sudo systemctl restart xdma

# 4. Verify
sudo systemctl status xdma
ls -la /dev/xdma* | head -5
```

### 4. Run Tests

```bash
python -m venv my_env
source my_venv/bin/activate
pip install torch
python3 user_hw_test.py
```

## References

- [Xilinx XDMA GitHub](https://github.com/Xilinx/dma_ip_drivers)

## License

The XDMA driver is licensed under BSD/GPL. See the Xilinx repository for details.
