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

### 4. Verify Installation

Check that the driver loaded successfully:
```bash
# Check module is loaded
lsmod | grep xdma

# Check for XDMA devices
ls -la /dev/xdma*

# Verify PCIe binding
lspci -k | grep -A 3 xdma
```

Expected devices:
- `/dev/xdma0_h2c_0` - Host to Card (write)
- `/dev/xdma0_c2h_0` - Card to Host (read)

## Usage

### Read from FPGA (Card to Host)
```bash
sudo ./dma_from_device --verbose \
    --device /dev/xdma0_c2h_0 \
    --address 0x30900004 \
    --size 4 \
    --file RECV
```

**Parameters:**
- `--device`: XDMA device node (c2h = card to host)
- `--address`: Memory address on the AXI bus
- `--size`: Number of bytes to read
- `--file`: Output file to save data

### Write to FPGA (Host to Card)
```bash
sudo ./dma_to_device --verbose \
    --device /dev/xdma0_h2c_0 \
    --address 0x80000000 \
    --size 4 \
    --file TEST
```

**Parameters:**
- `--device`: XDMA device node (h2c = host to card)
- `--address`: Memory address on the AXI bus
- `--size`: Number of bytes to write
- `--file`: Input file containing data to write

## Interrupt Modes

The `interrupt_mode` parameter controls how the driver handles interrupts:

- `0` - Auto-detect (MSI-X → MSI → Legacy)
- `1` - MSI (Message Signaled Interrupts)
- `2` - Legacy interrupts
- `3` - MSI-X (Extended MSI)
- `poll_mode=1` - Polling only (no interrupts)

## Simple C API

For programmatic access, see `simple_dma.c` which provides:
```c
// Read from FPGA
int dma_read(const char *device, uint64_t address, void *buffer, size_t size);

// Write to FPGA  
int dma_write(const char *device, uint64_t address, const void *buffer, size_t size);
```

Compile: `gcc -o simple_dma simple_dma.c`

## References

- [Xilinx XDMA GitHub](https://github.com/Xilinx/dma_ip_drivers)

## License

The XDMA driver is licensed under BSD/GPL. See the Xilinx repository for details.
