"""Encoding helpers for the Andromeda 0x00A0 hardware-info register."""

from dataclasses import dataclass


UE_HW_INFO_ADDR = 0x000000A0
MAX_FREQUENCY_MHZ_Q8 = 0x1FFFFF


@dataclass(frozen=True)
class HardwareInfo:
    raw: int
    queue_mode: bool
    axi_data_width_bits: int
    dram_size_code: int
    dram_size_gb: int
    core_count: int
    frequency_mhz_q8: int
    frequency_mhz: float
    frequency_hz: float


def format_hardware_info(info: HardwareInfo) -> str:
    """Format fields decoded from one HW_INFO register read."""
    return (
        f"HW_INFO[0x{UE_HW_INFO_ADDR:04x}]=0x{info.raw:08x}: "
        f"queue={int(info.queue_mode)}, "
        f"AXI={info.axi_data_width_bits} bits, "
        f"DRAM={info.dram_size_gb} GiB, "
        f"cores={info.core_count}, "
        f"clock_q8=0x{info.frequency_mhz_q8:06x}, "
        f"clock={info.frequency_mhz:.6f} MHz, "
        f"period={1000.0 / info.frequency_mhz:.9f} ns"
    )


def decode_hardware_info(raw_value: int) -> HardwareInfo:
    """Decode and validate queue, AXI, DRAM, topology, and Q13.8 MHz fields."""
    raw = raw_value & 0xFFFFFFFF
    queue_mode = bool((raw >> 31) & 0x1)
    axi_width_code = (raw >> 29) & 0x3
    if axi_width_code not in (0, 1):
        raise RuntimeError(f"Invalid HW_INFO AXI-width code {axi_width_code:02b} in 0x{raw:08x}")
    axi_data_width_bits = 256 if axi_width_code == 0 else 512
    dram_size_code = (raw >> 26) & 0x7
    # Encoding contract: 0 -> 2 GiB, 1 -> 4 GiB, 2 -> 8 GiB, ...
    dram_size_gb = 2 << dram_size_code
    core_count = (raw >> 21) & 0x1F
    if core_count == 0:
        raise RuntimeError(f"Invalid HW_INFO core count 0 in 0x{raw:08x}")
    frequency_mhz_q8 = raw & MAX_FREQUENCY_MHZ_Q8
    frequency_mhz = frequency_mhz_q8 / 256.0
    frequency_hz = frequency_mhz * 1_000_000.0
    if frequency_mhz_q8 == 0:
        raise RuntimeError(
            f"Invalid HW_INFO clock 0 MHz in 0x{raw:08x}"
        )
    return HardwareInfo(
        raw=raw,
        queue_mode=queue_mode,
        axi_data_width_bits=axi_data_width_bits,
        dram_size_code=dram_size_code,
        dram_size_gb=dram_size_gb,
        core_count=core_count,
        frequency_mhz_q8=frequency_mhz_q8,
        frequency_mhz=frequency_mhz,
        frequency_hz=frequency_hz,
    )
