"""
Core DMA/engine ops: constants, register definitions, instruction-level APIs.

This module provides:
- Register address and constant definitions (USER_BASE_ADDR, UE_*_ADDR, etc.)
- Enums: UE_MODE, TYPE, URAM_SECTION, LALU_MODE, etc.
- Instructions, DeviceTensor, UnifiedEngine
- Low-level ops: ue_arithmetic_op, wait_queue, dma_read/dma_write, user_read_reg32, etc.
- High-level engine APIs built on them (eltwise_add, matmat_mul, ...)
- ``ue_35bit_addr_shifter()``: byte DRAM address to 35-bit word address for UE registers / captured instructions

For tests and bf16 helpers (precompute_freqs_cis, calculate_snr, generic_tests),
use user_dma_tests_bf16 or the combined user_dma_ops.

Usage:
    from user_dma_core import UnifiedEngine, UE_MODE, DRAM_ACTIVATION_ADDR, ue_35bit_addr_shifter
    ue = UnifiedEngine(device='cpu')
    handler = ue.eltwise_add(size, input_a_dram_addr, input_b_dram_addr, output_dram_addr)
    result = handler(vec1, vec2)
"""

from re import U
import struct
import os
import sys
import time
from typing import Callable, Optional, Tuple
from enum import IntEnum
import torch
import math

# Register address definitions (matches andromeda.c)
# User device base address (AXI-Lite BAR mapping)
AXI_LITE_TRANSLATION_OFFSET = 0x00000000
UE_0_BASE_ADDR = 0x02000000
# Register address offsets
UE_START_ADDR = 0x00000000
UE_FPGA_VERSION_ADDR = 0x00000000
UE_ISA_CTRL = 0x00000004  # Reserved for ISA debug
UE_DRAM_ADDR = 0x00000008  # Unified DMA address (bits [63:32])
UE_DMA_LENGTH_ADDR = 0x0000000C
UE_CONTROL_ADDR = 0x00000010
UE_STATUS_ADDR = 0x00000014
UE_OUTPUT_VALID_DELAY_ADDR = 0x0000001C
UE_URAM_LENGTH_ADDR = 0x00000020
UE_URAM_WRITEBACK_ADDR = 0x00000024
UE_LATENCY_COUNT_ADDR = 0x00000030
UE_DRAM_URAM_CTRL_ADDR = 0x00000034
UE_LALU_HYPERPARAMETERS_ADDR = 0x00000038  # bf16_lalu_a [15:0], bf16_lalu_b [31:16] (axi_reg_map_pkg.sv)
UE_URAM_ROW_SIZE_ADDR = 0x00000040
UE_VALID_DELAY_EXTRA_ADDR = 0x00000044
UE_LALU_INST_ADDR = 0x00000048
UE_LALU_DELAY_ADDR = 0x0000004C
UE_SCALAR_ADDR = 0x00000050
UE_QUEUE_CTRL_ADDR = 0x00000054
UE_URAM_LENGTH_ADDR_Z = 0x0000005C
UE_BIAS_ADDER_EN_ADDR = 0x00000060
UE_URAM_WB_PADDING_ADDR = 0x00000064
UE_BROADCAST_MODE_ADDR = 0x00000068
UE_INSTRUCTION_CTL_ADDR = 0x00000070
UE_TOTAL_BYTES_PER_STRIDE = 0x00000074
UE_INSTRUCTION_ADDR = 0x00000078
UE_TOTAL_STRIDE_BYTES = 0x0000007C
UE_FMAX_CONTEXT_ADDR = 0x00000080
UE_TRACE_BRAM_ADDR = 0x00000084  # read pointer or write pointer depending on access
UE_TRACE_BRAM_DATA = 0x00000088  # read data from trace BRAM
UE_ARGMAX1_INDEX = 0x0000008C
UE_ARGMAX2_INDEX = 0x00000090
UE_ARGMAX3_INDEX = 0x00000094
UE_ARGMAX4_INDEX = 0x00000098
UE_INT_REG = 0x0000009C  # bits [1:0] interrupt cause; write clears latch (axi_lite_*_module.sv)
# Backward compatibility: primary argmax readout (same as andromeda.c UE_ARGMAX1_INDEX)
UE_ARGMAX_INDEX = UE_ARGMAX1_INDEX
UE_LAST_REG_ADDR = UE_INT_REG  # last AXI-Lite offset for init scan

# queue_state_module.sv int_cause / andromeda.c INT_CAUSE_*
INT_CAUSE_NONE = 0
INT_CAUSE_SWI = 1
INT_CAUSE_HALT = 2

# CLOCK_CYCLE_TIME_NS will be set based on target argument (default: 3 ns)
CLOCK_CYCLE_TIME_NS = 3
# HW prescales PIPELINE_COUNTER (0x30): one count = 16 aclk cycles
# (pipeline_counter_module.sv CLK_DIV). Multiply readouts by this to get cycles.
UE_PIPELINE_COUNTER_CLK_DIV = 16
UE_PEAK_GFLOPS = 333.3 * 0.128
UE_TRACE_SIZE = 8192
UE_AXI_DATA_WIDTH_BITS = int(os.environ.get("UE_AXI_DATA_WIDTH_BITS", "256"))

# ---------------------------------------------------------------------------
# AXI beat width: the ONE number that differs between the two board profiles
# (256-bit: puzhi/alinx/alveo/kintex7; 512-bit: bittware/rk — see the
# ``args.device in ("bittware", "rk")`` check in user_hw_test.py's __main__).
# The DMA engine requires DRAM-side transfer addresses/lengths to land on
# beat boundaries, so every alignment/padding rule elsewhere in this file and
# in user_hw_test.py is a direct consequence of this one doubled granularity:
#
#   width_bits | beat_bytes | beat_bf16_elems
#   -----------+------------+----------------
#      256     |     32     |       16
#      512     |     64     |       32
#
# Use these helpers instead of recomputing ``UE_AXI_DATA_WIDTH_BITS // 8`` (or
# ``// 16``) at each call site so the 256-vs-512 behavior lives in one place.
# ---------------------------------------------------------------------------

def ue_axi_beat_bytes_for(width_bits: int) -> int:
    """AXI beat size in bytes for an explicit width (32 @256-bit, 64 @512-bit)."""
    return width_bits // 8


def ue_axi_beat_bf16_elems_for(width_bits: int) -> int:
    """AXI beat size in bf16 elements for an explicit width (16 @256-bit, 32 @512-bit)."""
    return max(1, width_bits // 16)


def ue_axi_beat_bytes() -> int:
    """AXI beat size in bytes for the live ``UE_AXI_DATA_WIDTH_BITS``."""
    return ue_axi_beat_bytes_for(UE_AXI_DATA_WIDTH_BITS)


def ue_axi_beat_bf16_elems() -> int:
    """AXI beat size in bf16 elements for the live ``UE_AXI_DATA_WIDTH_BITS``."""
    return ue_axi_beat_bf16_elems_for(UE_AXI_DATA_WIDTH_BITS)


def ue_round_up_to_axi_beat_bytes(nbytes: int) -> int:
    """Round ``nbytes`` up to the next AXI-beat multiple (beat = :func:`ue_axi_beat_bytes`)."""
    beat = ue_axi_beat_bytes()
    return ((nbytes + beat - 1) // beat) * beat


def ue_round_up_to_axi_beat_elems(nelems: int) -> int:
    """Round ``nelems`` (bf16 elements) up to the next AXI-beat multiple (beat = :func:`ue_axi_beat_bf16_elems`)."""
    beat = ue_axi_beat_bf16_elems()
    return ((nelems + beat - 1) // beat) * beat


def ue_assert_axi_beat_aligned_bytes(nbytes: int, what: str, hint: str = "") -> None:
    """Assert ``nbytes`` lands on an AXI-beat boundary; ``what`` names the caller for the message."""
    beat = ue_axi_beat_bytes()
    assert nbytes % beat == 0, (
        f"{what}: {nbytes} bytes is not a multiple of the {beat}-byte AXI beat "
        f"(UE_AXI_DATA_WIDTH_BITS={UE_AXI_DATA_WIDTH_BITS})" + (f"; {hint}" if hint else "")
    )


# DMA device paths (can be overridden by command-line argument)
DMA_DEVICE_H2C = "/dev/xdma0_h2c_0"
DMA_DEVICE_C2H = "/dev/xdma0_c2h_0"
DMA_DEVICE_USER = "/dev/xdma0_user"  # AXI-Lite user interface for register access
CURRENT_DEVICE = "default"

def _sync_imported_dma_device_paths(old_h2c: str, old_c2h: str, old_user: str):
    """Rebind modules that imported DMA path constants by value."""
    import sys as _sys
    for _mod in list(_sys.modules.values()):
        if _mod is None or _mod is _sys.modules[__name__]:
            continue
        # Probe via __dict__ to avoid triggering lazy ``__getattr__`` hooks (e.g.
        # transformers.models.* prints a deprecation warning for any name access).
        _md = getattr(_mod, "__dict__", None)
        if _md is None:
            continue
        for _name, _old, _new in (("DMA_DEVICE_H2C", old_h2c, DMA_DEVICE_H2C),
                                  ("DMA_DEVICE_C2H", old_c2h, DMA_DEVICE_C2H),
                                  ("DMA_DEVICE_USER", old_user, DMA_DEVICE_USER)):
            if _md.get(_name) == _old:
                try:
                    setattr(_mod, _name, _new)
                except Exception:
                    pass

def set_dma_device(device_name: str):
    """Set DMA device paths based on device name (e.g., 'xdma0' -> '/dev/xdma0_*').

    Also rebinds the names in any module that imported them by value
    (``from user_dma_core import DMA_DEVICE_H2C``), so callers that took the
    pre-set_dma_device snapshot still see the right device. Without this,
    models split across xdma0/xdma1 — methods on UnifiedEngine read the live
    module global (correct), while bare references in the model file read the
    stale import snapshot (wrong).
    """
    global DMA_DEVICE_H2C, DMA_DEVICE_C2H, DMA_DEVICE_USER
    old_h2c, old_c2h, old_user = DMA_DEVICE_H2C, DMA_DEVICE_C2H, DMA_DEVICE_USER
    DMA_DEVICE_H2C = f"/dev/{device_name}_h2c_0"
    DMA_DEVICE_C2H = f"/dev/{device_name}_c2h_0"
    DMA_DEVICE_USER = f"/dev/{device_name}_user"
    _sync_imported_dma_device_paths(old_h2c, old_c2h, old_user)

# Constants
UE_VECTOR_SIZE = 64
# Captured decoder instruction line size (matches Vitis ``decoder_inst_t`` / ``INSTRUCTION_SIZE_BYTES``).
INSTRUCTION_SIZE_BYTES = 32
UE_FMAX_CONTEXT_SIZE = 64
SCALE_BRAM_ELEMENTS = 8192
SCALE_BRAM_SIZE_BYTES = SCALE_BRAM_ELEMENTS * 2
BIAS_BRAM_ELEMENTS = 8192
BIAS_BRAM_SIZE_BYTES = BIAS_BRAM_ELEMENTS * 2
DRAM_START_ADDR = 0x80000000 # 0 GB
DRAM_ACTIVATION_ADDR = 0xB0000000 # 512 MB reserved for intermediate results
DRAM_INSTRUCTION_ADDR = 0xD0000000  # 256*3 MB reserved for instructions
DRAM_END_ADDR = 0xFFFFFFFF

def configure_device(device_name: str, dma_device: Optional[str] = None, base_addr: Optional[int] = None) -> dict:
    """Apply board-specific DMA, AXI-Lite base, and DRAM layout constants."""
    global CURRENT_DEVICE, UE_0_BASE_ADDR, DRAM_START_ADDR, DRAM_ACTIVATION_ADDR, DRAM_INSTRUCTION_ADDR, DRAM_END_ADDR
    global DMA_DEVICE_H2C, DMA_DEVICE_C2H, DMA_DEVICE_USER

    CURRENT_DEVICE = device_name
    if device_name == "efinix":
        UE_0_BASE_ADDR = 0x00000000 if base_addr is None else int(base_addr)
        DRAM_START_ADDR = 0x00000000
        DRAM_ACTIVATION_ADDR = 0x30000000
        DRAM_INSTRUCTION_ADDR = 0x3F000000
        DRAM_END_ADDR = 0x7FFFFFFF
        old_h2c, old_c2h, old_user = DMA_DEVICE_H2C, DMA_DEVICE_C2H, DMA_DEVICE_USER
        DMA_DEVICE_H2C = "/dev/pcie_dma0_htc_0"
        DMA_DEVICE_C2H = "/dev/pcie_dma0_cth_0"
        DMA_DEVICE_USER = "/dev/pcie_dma0_user"
        _sync_imported_dma_device_paths(old_h2c, old_c2h, old_user)
    else:
        UE_0_BASE_ADDR = 0x02000000 if base_addr is None else int(base_addr)
        DRAM_START_ADDR = 0x80000000
        DRAM_ACTIVATION_ADDR = 0xB0000000
        DRAM_INSTRUCTION_ADDR = 0xD0000000
        DRAM_END_ADDR = 0xFFFFFFFF
        set_dma_device(dma_device or "xdma0")

    for name, addr in (
        ("DRAM_START_ADDR", DRAM_START_ADDR),
        ("DRAM_ACTIVATION_ADDR", DRAM_ACTIVATION_ADDR),
        ("DRAM_INSTRUCTION_ADDR", DRAM_INSTRUCTION_ADDR),
    ):
        assert addr % 64 == 0, f"{name}=0x{addr:x} must be 64-byte aligned"

    return {
        "device": CURRENT_DEVICE,
        "ue_0_base_addr": UE_0_BASE_ADDR,
        "dram_start_addr": DRAM_START_ADDR,
        "dram_activation_addr": DRAM_ACTIVATION_ADDR,
        "dram_instruction_addr": DRAM_INSTRUCTION_ADDR,
        "dram_end_addr": DRAM_END_ADDR,
        "dma_h2c": DMA_DEVICE_H2C,
        "dma_c2h": DMA_DEVICE_C2H,
        "dma_user": DMA_DEVICE_USER,
        "clock_period_ns": 4.0 if device_name == "efinix" else None,
        "axi_data_width_bits": 256 if device_name == "efinix" else None,
    }

# UE DRAM / instruction fields use a 35-bit word address = byte address >> 3 (see andromeda.c, decoder inst words).
UE_WORD_ADDR_BITS = 35
UE_WORD_ADDR_MASK = (1 << UE_WORD_ADDR_BITS) - 1


def ue_35bit_addr_shifter(byte_addr: int) -> int:
    """Byte address to 35-bit UE word address (same as ``byte_addr >> 3`` in hardware)."""
    return (int(byte_addr) >> 3) & UE_WORD_ADDR_MASK


# URAM size in bytes
URAM_START_ADDR = 0x000 # 0th row
URAM_HALFWAY_ADDR = 0x800 # 2048th row
URAM_END_ADDR = 0xFFF # 4095th row
URAM_HALF_WAY_SIZE = (2**11)*UE_VECTOR_SIZE*2  # 262144 bytes
URAM_0XF00_SIZE = (0xF00)*UE_VECTOR_SIZE*2 # 491520 bytes
URAM_NEAR_FULL_ADDR = 0xFFF # 1 row less than full
URAM_NEAR_FULL_SIZE = URAM_NEAR_FULL_ADDR * UE_VECTOR_SIZE * 2
URAM_HALF_ELEMENTS = URAM_HALFWAY_ADDR * UE_VECTOR_SIZE
URAM_NEAR_FULL_ELEMENTS = URAM_NEAR_FULL_ADDR * UE_VECTOR_SIZE
URAM_FULL_ELEMENTS = URAM_HALF_ELEMENTS * 2
# Mode definitions
class UE_MODE(IntEnum):
    DOT_PRODUCT = 0
    EXP = 1
    ELTWISE_MUL = 2
    ROPE = 3
    ELTWISE_ADD = 4
    RMS = 5
    MUL_BROADCAST = 6
    BF16_DOT_PRODUCT = 7
    URAM_DRAM_WRITEBACK = 8
    QUANTIZE = 9
    ADD_BROADCAST = 10
    ADD_REDUCE = 11
    DEQUANTIZE = 12
    ELTWISE_SUB = 13
    MEMCPY_FROM_DRAM = 15

class TYPE(IntEnum):
    # Adaptive block-scale formats. For IF4/IF8 the INT vs FP variant is
    # encoded in the sign bit of each block's bf16 scale: negative scale ->
    # INT path, positive scale -> FP path. The hardware always uses |scale|
    # as the effective multiplier.
    IF8 = 1   # 8-bit: sign=1 -> INT8, sign=0 -> FP8 E4M3
    # TQ4 (TurboQuant 4-bit): 4-bit index into a 16-entry programmable bf16
    # codebook latched from URAM-B[0] at dma_start.
    #
    # Hardware constraint: per-block bf16 scales must be POSITIVE. Unlike
    # IF4/IF8 where the scale sign bit selects the FP-vs-INT variant, for
    # TQ4 the same sign bit (``quant_datatype``) gates which data path the
    # 4-bit nibble takes inside compute_unit.vhdl. Only the
    # ``quant_datatype = '0'`` branch routes the nibble into ``fp_data``,
    # which is what feeds ``tq4_to_bf19``. Under a negative scale the
    # nibble is steered into the INT4 path and the codebook lookup reads
    # entry 0 for every element, so callers must keep the scale sign
    # bit clear.
    TQ4 = 2   # 4-bit: codebook[idx] * scale (scale must be positive)
    IF4 = 3   # 4-bit: sign=1 -> INT4, sign=0 -> FP4 E2M1

class URAM_SECTION(IntEnum):
    URAM_A = 0
    URAM_B = 1

class URAM_WRITE_SRC(IntEnum):
    URAM_WRITE_BACK = 0
    URAM_DRAM = 1
    URAM_MICROBLAZE = 2
    URAM_WB_DISABLE = 3

class LALU_MODE(IntEnum):
    BYPASS = 0
    ACT = 1
    ACT_NO_X = 2
    MODE_RECIP = 3
    MODE_RSQRT = 4
    CLAMP = 5       # max(a, min(x, b))
    LOG = 6         # log(max(a, min(x, b)))

# LALU activation hyperparameters (bf16 values for UE_LALU_HYPERPARAMETERS register)
# Activation function (a+x)*sigmoid(-bx)
# gelu: x*sigmoid(1.702x)
# Set ACT
LALU_ACT_GELU_A = 0x0000
LALU_ACT_GELU_B = 0xbfd9   # -1.702
# silu: x*sigmoid(x)
# Set ACT
LALU_ACT_SILU_A = 0x0000
LALU_ACT_SILU_B = 0xbf80   # -1.0 in bf16
# sigmoid: sigmoid(x)
# Set ACT_NO_X
LALU_ACT_SIGMOID_A = 0x3f80  # 1.0 in bf16
LALU_ACT_SIGMOID_B = 0xbf80  # -1.0 in bf16
# clamp(x, 0, +inf) — equivalent to ReLU
LALU_CLAMP_RELU_A = 0x0000   # 0.0 in bf16
LALU_CLAMP_RELU_B = 0x7f80   # +inf in bf16
# log(clamp(x, 1e-3, +inf)) — keeps argument strictly positive
LALU_LOG_A = 0x3a83          # ~1e-3 in bf16
LALU_LOG_B = 0x7f80          # +inf in bf16

class MEMCPY_TYPE(IntEnum):
    URAM = 0
    BRAM = 1
    BIAS_BRAM = 2
    SCALE_BRAM = 3
class BROADCAST_MODE(IntEnum):
    LALU_RESULT = 0
    LALU_RESULT_NEGATE = 1
    FMAX_NEGATE = 2
    SCALAR_IN_REG = 3

# Writeback padding constants
WB_PADDING_ZERO = 0     # 0x0000
WB_PADDING_NEG_INF = 1  # 0xFF80

# Legacy delay-register timing constants required by the current Efinix
# 0x12345678 bitstream. Newer Andromeda RTL derives these internally; keep the
# writes Efinix-only in init_unified_engine().
UE_PIPELINE_BF19_MULT = 2
UE_PIPELINE_BF19_ADD = 3
UE_PIPELINE_CUSTOM_EXP = 5
UE_PIPELINE_ADDER_TREE = 12
BF20_ADDER_3_CYCLE = True
UE_LATENCY_BF20_ITR2, UE_LATENCY_BF20_ITR3, UE_LATENCY_BF20_ITRGT3 = (
    (3, 11, 11) if BF20_ADDER_3_CYCLE else (2, 5, 7)
)
UE_PIPELINE_STAGES_INPUT_REG = 1
UE_PIPELINE_STAGES_DOT_P = 4
UE_PIPELINE_STAGES_RMS = 4
UE_PIPELINE_STAGES_EXP = 4
UE_PIPELINE_STAGES_MULT = 4
UE_PIPELINE_STAGES_ADD = 3
UE_LATENCY_DOT_PRODUCT = UE_PIPELINE_STAGES_INPUT_REG + UE_PIPELINE_STAGES_DOT_P + UE_PIPELINE_BF19_MULT + UE_PIPELINE_ADDER_TREE - 1
UE_LATENCY_RMS = UE_PIPELINE_STAGES_RMS + UE_PIPELINE_BF19_MULT + UE_PIPELINE_ADDER_TREE - 1
UE_LATENCY_EXP = UE_PIPELINE_STAGES_EXP + UE_PIPELINE_BF19_ADD + UE_PIPELINE_CUSTOM_EXP + UE_PIPELINE_ADDER_TREE - 1
UE_LATENCY_ADD_REDUCE = UE_PIPELINE_STAGES_ADD + UE_PIPELINE_BF19_ADD + UE_PIPELINE_ADDER_TREE - 1
UE_LATENCY_ELTWISE_MUL = UE_PIPELINE_STAGES_MULT + UE_PIPELINE_BF19_MULT - 1
UE_LATENCY_ELTWISE_ADD = UE_PIPELINE_STAGES_ADD + UE_PIPELINE_BF19_ADD - 1
UE_LATENCY_ADD_EXP = UE_PIPELINE_STAGES_EXP + UE_PIPELINE_BF19_ADD + UE_PIPELINE_CUSTOM_EXP - 1
UE_LALU_PIPELINE_FPDIV = 3
UE_LALU_PIPELINE_FPSQRT = 3
UE_LALU_PIPELINE_FACT = 8
UE_LALU_LATENCY_SOFTMAX = 1 + UE_LALU_PIPELINE_FPDIV
UE_LALU_LATENCY_RMS = 1 + UE_LALU_PIPELINE_FPSQRT + 1 + UE_LALU_PIPELINE_FPDIV
UE_LALU_LATENCY_ACT = 1 + UE_LALU_PIPELINE_FACT + 1 + UE_LALU_PIPELINE_FPDIV
UE_QUANTIZE_FMAX_PIPELINE = 8
UE_LATENCY_QSCALE = UE_LALU_PIPELINE_FPDIV + UE_QUANTIZE_FMAX_PIPELINE + 2
UE_QINPUT_DELAY = UE_LATENCY_QSCALE + 1
UE_LATENCY_QUANTIZATION = UE_LATENCY_QSCALE + UE_PIPELINE_BF19_MULT + 5

# ISA instruction type constants (matching queue_state_module.sv / andromeda.c)
INSTRUCTION_UE_OP = 0x0
INSTRUCTION_JUMP = 0x1
INSTRUCTION_REG_ALU_PREFETCH = 0x2
# Instruction type 3 is reserved.
INSTRUCTION_FLAG = 0x4
INSTRUCTION_UE_PBI = 0x5
INSTRUCTION_PBI_SET_PREFETCH = 0x6
INSTRUCTION_REG_ALU_NONPREFETCH = 0x7
INSTRUCTION_SWI = 0x8
INSTRUCTION_HALT = 0x9
INSTRUCTION_NOP = 0xA
INSTRUCTION_PBI_SET_NONPREFETCH = 0xB

# Compatibility aliases for existing compiler call sites.
INSTRUCTION_REG_ALU = INSTRUCTION_REG_ALU_PREFETCH
INSTRUCTION_PBI_SET = INSTRUCTION_PBI_SET_PREFETCH

# PBI mode constants (match queue_state_module.sv)
PBI_MODE_INIT = 0
PBI_MODE_INC  = 1
PBI_MODE_REG  = 2

class PBI_FIELD(IntEnum):
    """Field selector for PBI_MODE_REG (pbi_field_select, inst_descriptor[23:20]).
    Matches the next_* assigns in queue_state_module.sv."""
    DRAM_ADDR            = 0  # next_dram_addr              [31:0]  — bits [31:0]   of pointer row
    DMA_LENGTH           = 1  # next_dma_length             [31:0]  — bits [63:32]
    OUTPUT_SIZE          = 2  # next_output_size            [15:0]  — bits [79:64]
    URAM_ROW_SIZE        = 3  # next_uram_row_size          [11:0]  — bits [91:80]
    URAM_START_ADDR_Y    = 4  # next_uram_start_addr_y      [11:0]  — bits [103:92]
    URAM_START_ADDR_Z    = 5  # next_uram_start_addr_z      [11:0]  — bits [115:104]
    URAM_WRITEB_ADDR     = 6  # next_uram_writeb_addr       [11:0]  — bits [127:116]
    URAM_ROW_SIZE_Z      = 7  # next_uram_row_size_z        [11:0]  — bits [139:128]
    URAM_MEMCPY_DST_ADDR = 8  # next_uram_memcpy_dst_addr   [11:0]  — bits [151:140]
    FMX_CONTEXT          = 9  # next_fmx_context            [5:0]   — bits [157:152]
    URAM_ROW_STRIDE_Z    = 10 # next_uram_row_stride_z_pbi  [11:0]  — bits [169:158]; delta=inst_uram_row_stride_z
    LALU_SCALAR_OR_STRIDE_JUMP = 11 # shared descriptor/PBI field; interpretation follows UE mode
    LALU_SCALAR          = 11 # arithmetic interpretation; delta=inst_lalu_scalar
    STRIDE_JUMP          = 11 # memcpy/writeback interpretation

# FLAG instruction mode constants
FLAG_MODE_SET   = 0  # Assert this engine's flag (signal busy)
FLAG_MODE_CLEAR = 1  # De-assert this engine's flag (signal done)
FLAG_MODE_CHECK = 2  # Spin-wait until target engine's flag is 1

# JUMP mode constants (match queue_state_module.sv / andromeda.c)
JUMP_MODE_ABSOLUTE = 0
JUMP_MODE_REG_ABS  = 1  # unconditional absolute jump to regfile_rdata2 (rst_reg_idx)
JUMP_MODE_JNZ = 2  # absolute if reg != 0
JUMP_MODE_JZ = 3   # absolute if reg == 0
JUMP_MODE_RELATIVE = 4  # inst cache read ptr -= immediate[9:0]
JUMP_MODE_RELA_JNZ = 5
JUMP_MODE_RELA_JZ = 6
JUMP_MODE_REG_RELA = 7  # unconditional relative jump: read ptr -= regfile_rdata2[9:0]

# Register ALU sub-modes (isa_mode[3:0]; queue_state_module.sv ALU_MODE_*)
ALU_MODE_INC     = 0   # dst = src + 1
ALU_MODE_DEC     = 1   # dst = src - 1
ALU_MODE_ADD_REG = 2   # dst = src1 + src2
ALU_MODE_ADD_IMM = 3   # dst = src + immediate
ALU_MODE_SET     = 4   # dst = immediate
ALU_MODE_MIN     = 5   # dst = min(src1, src2), unsigned
ALU_MODE_SUB     = 6   # dst = src1 - src2
# 7, 8: reserved (former single-cycle MUL16_REG / MUL16_IMM; removed from RTL)
ALU_MODE_SHR       = 9   # dst = src >> imm[4:0], logical right shift
ALU_MODE_SHL       = 10  # dst = src << imm[4:0], logical left shift
ALU_MODE_MUL32_REG = 11  # dst = (src * rst)[31:0], pipelined 3-cycle (int_mult_pipe)
ALU_MODE_MUL32_IMM = 12  # dst = (src * imm[15:0])[31:0], pipelined 3-cycle; imm zero-extended from 16 bits
ALU_MODE_DIV_REG   = 13  # dst = src / rst, sequential 32-cycle (int_divider)
ALU_MODE_MUL_SHL   = 14  # dst = ((src * rst)[31:0]) << imm[4:0], fused mul+shift (int_mult_pipe)
ALU_MODE_MUL_SHR   = 15  # dst = ((src * rst)[31:0]) >> imm[4:0], fused mul+shift (int_mult_pipe)
ALU_MODE_MUL_IMM   = ALU_MODE_MUL32_IMM  # alias: reg×imm multiply targets MUL32_IMM (matches queue_state_module.sv)

# Register file indices
REGFILE_R0_ZERO = 0       # Zero register (always 0)
REGFILE_R1_LOOP = 1       # Loop counter
REGFILE_R2_DRAM_ADDR = 2  # DRAM address register
REGFILE_R3_RES = 3        # Result register

# Decoder instruction capture constants
MAX_DECODER_INSTRUCTIONS = 768 * 1024 * 1024 // 32

def _inst_desc_bits(w: list, lo: int, hi: int) -> int:
    """inst_descriptor[hi:lo] inclusive (Verilog indices), from eight LE words (queue_state_module.sv)."""
    val = 0
    for b in range(lo, hi + 1):
        val |= ((w[b // 32] >> (b % 32)) & 1) << (b - lo)
    return val


class Instructions:
    """
    256-bit instruction descriptor (32 bytes); layout matches Vivado/hdl/queue_state_module.sv
    inst_descriptor[255:0] (w[k] = [32k+31 : 32k], LE words).
    """
    def __init__(self):
        # 8 words of 32 bits each = 256 bits = 32 bytes
        self.words = [0] * 8

    def get_bytes(self) -> bytes:
        """Convert instruction to bytes (32 bytes total)"""
        result = bytearray(32)
        for i in range(8):
            # Pack each 32-bit word as little-endian
            word_bytes = struct.pack('<I', self.words[i] & 0xFFFFFFFF)
            result[i*4:(i+1)*4] = word_bytes
        return bytes(result)

    def __repr__(self):
        return f"Instructions(words=[{', '.join(f'0x{w:08X}' for w in self.words)}])"

class DeviceTensor:
    """
    Tensor wrapper with FPGA DRAM address tracking for DMA cache optimization.

    For inputs: Skip H2C DMA if data is already synced at target address.
    For outputs: Lazy C2H DMA - only fetch when to_cpu() called or data accessed.
    """

    def __init__(self, shape: torch.Size, ue: 'UnifiedEngine', dram_addr: Optional[int] = None, data: Optional[torch.Tensor] = None):
        self._ue = ue
        self._data = data
        self._shape = shape
        self.dram_addr = dram_addr
        self.on_fpga_dram = dram_addr is not None
        self.on_host_dram = data is not None
        self.synced = False

    @property
    def data(self) -> torch.Tensor:
        """Get tensor data. For outputs, triggers DMA fetch if not yet fetched."""
        if self.on_host_dram:
            print(f"Data is on host DRAM and is synced" if self.synced else "Data is on host DRAM and is not synced")
            return self._data
        else:
            if self.on_fpga_dram:
                if self.dram_addr is not None:
                    print(f"Data is on FPGA DRAM at {hex(self.dram_addr)} and size is {self.size_bytes()} bytes")
                    self._data = torch.zeros(self.get_num_elements(), dtype=torch.bfloat16)
                    self._ue.dma_read(DMA_DEVICE_C2H, self.dram_addr, self._data, self.size_bytes())
                    self.synced = True
                    return self._data.reshape(self._shape)
                else:
                    print(f"Data is on FPGA DRAM but no DRAM address set")
                    raise ValueError("Data is on FPGA DRAM but no DRAM address set")
            else:
                print(f"Data is neither on FPGA DRAM nor on host DRAM")
                raise ValueError("Data is neither on FPGA DRAM nor on host DRAM")

    @data.setter
    def data(self, value: torch.Tensor):
        self._data = value
        self.on_host_dram = True
        self.on_fpga_dram = False
        self.synced = False

    def sync(self, dram_addr: Optional[int] = None) -> None:
        """Mark as synced at address (call after DMA write)."""
        if dram_addr is not None:
            if self._data is None:
                self.data # fetch data from FPGA DRAM
            self.dram_addr = dram_addr # new address
            self._ue.dma_write(DMA_DEVICE_H2C, dram_addr, self._data, self.size_bytes())
        else:
            print(f"Data is on host DRAM")
            self.on_fpga_dram = False
            self.on_host_dram = True
            self.synced = True

        self.synced = True

    def invalidate(self) -> None:
        """Mark as stale (call after host modification)."""
        self.synced = False

    def needs_dma(self, dram_addr: int) -> bool:
        """Check if H2C DMA is needed to reach the given address."""
        return (self.dram_addr != dram_addr)

    def get_num_elements(self) -> int:
        """Get number of elements in a tensor."""
        if len(self._shape) == 1:
            return self._shape[0]
        elif len(self._shape) == 2:
            return self._shape[0] * self._shape[1]
        elif len(self._shape) == 3:
            return self._shape[0] * self._shape[1] * self._shape[2]
        else:
            raise ValueError(f"Invalid shape: {self._shape}")

    def size_bytes(self) -> int:
        """Get size in bytes (assumes bf16 = 2 bytes per element)."""
        if len(self._shape) == 1:
            return self._shape[0] * 2
        elif len(self._shape) == 2:
            return self._shape[0] * self._shape[1] * 2
        elif len(self._shape) == 3:
            return self._shape[0] * self._shape[1] * self._shape[2] * 2
        else:
            raise ValueError(f"Invalid shape: {self._shape}")

    def where_is_data(self):
        """Check if data is on device (synced) and not yet fetched to host."""
        if not self.synced:
            print(f"Data is not synced")
            if self.on_fpga_dram:
                print(f"Data is on FPGA DRAM at {hex(self.dram_addr)}")
            if self.on_host_dram:
                print(f"Data is on host DRAM")
        else:
            print(f"Data is on FPGA DRAM at {hex(self.dram_addr)} and host DRAM")

    def to_cpu(self) -> torch.Tensor:
        """Return tensor on host (triggers DMA fetch if on FPGA DRAM)."""
        return self.data

class UnifiedEngine:
    """
    PyTorch-based Unified Engine that simulates hardware operations
    """
    def __init__(self,
                 BASE_ADDR: Optional[int] = None,
                 device: str = 'cpu',
                 params_dram_base: Optional[int] = None,
                 program_dram_base: Optional[int] = None,
                 tensor_dram_base: Optional[int] = None,
                 clock_period_ns: float = None,
                 device_name: Optional[str] = None):
        self.device = device
        if device_name is not None:
            set_dma_device(device_name)
        # Snapshot the live DMA device paths so the engine is bound to one FPGA
        # for its lifetime; prefer self.h2c_device/self.c2h_device/self.user_device
        # over module-level constants in new code.
        self.h2c_device = DMA_DEVICE_H2C
        self.c2h_device = DMA_DEVICE_C2H
        self.user_device = DMA_DEVICE_USER
        if BASE_ADDR is None:
            BASE_ADDR = UE_0_BASE_ADDR
        if params_dram_base is None:
            params_dram_base = DRAM_START_ADDR
        if program_dram_base is None:
            program_dram_base = DRAM_INSTRUCTION_ADDR
        if tensor_dram_base is None:
            tensor_dram_base = DRAM_ACTIVATION_ADDR

        # Hardware state (no simulation - uses actual DMA)
        # Note: DRAM/URAM/BRAM are accessed via DMA, not stored locally

        # Latency tracking
        self.latency_count = 0

        # Decoder instruction capture state
        self.capture_buffer: list[Instructions] = []
        self.capture_count = 0
        self.is_capture_on = False

        # Incremental memory allocation tracking for auto-allocation.
        self._params_dram_base = params_dram_base
        self._program_dram_base = program_dram_base
        self._tensor_dram_base = tensor_dram_base
        self._next_params_dram_addr = self._params_dram_base
        self._next_program_dram_addr = self._program_dram_base
        self._tensor_dram_addr = self._tensor_dram_base
        # ISA register and instruction pointer allocation tracking
        self._isa_reg_counter = 1      # Register 0 is hard-wired zero
        self._inst_ptr_counter = 1     # Allocation starts from 1; 15 pointers (1-15) available

        # Optional key-value address registry for tests/utilities.
        self._dram_addresses: dict[str, int] = {}

        self._inst_id = 0
        # During capture: LIFO stack of (counter_reg, body_start_inst_id) per nested loop_start/loop_end.
        self._capture_loop_stack: list[tuple[int, int]] = []
        self._clock_period_ns = clock_period_ns if clock_period_ns is not None else CLOCK_CYCLE_TIME_NS

        self._base_addr = BASE_ADDR

        # Initialize
        self.init_unified_engine()

    @staticmethod
    def _align_up(value: int, align_bytes: int) -> int:
        if align_bytes <= 1:
            return value
        return ((value + align_bytes - 1) // align_bytes) * align_bytes

    def define_dram_address(self, key: str, address: int):
        """Define or override a named DRAM address."""
        self._dram_addresses[key] = address

    def get_dram_address(self, key: str) -> int:
        """Fetch a named DRAM address."""
        assert key in self._dram_addresses, f"Unknown DRAM address key: {key}"
        return self._dram_addresses[key]

    def allocate_params_dram(self, size_bytes: int, label: Optional[str] = None, align_bytes: int = 64) -> int:
        """
        Allocate memory from the params DRAM region incrementally.

        Args:
            size_bytes: Number of bytes to allocate
        """
        self._next_params_dram_addr = self._align_up(self._next_params_dram_addr, align_bytes)
        params_dram_addr = self._next_params_dram_addr
        self._next_params_dram_addr += size_bytes
        if label is not None:
            self._dram_addresses[label] = params_dram_addr
        return params_dram_addr

    def allocate_program_dram(self, size_bytes: int, label: Optional[str] = None, align_bytes: int = 64) -> int:
        """
        Allocate memory from the program/instruction DRAM region incrementally.

        Both the start address and the allocation length are rounded up to
        ``align_bytes`` (default 64 B = 512 bits) so that consecutive program
        regions keep every instruction PC aligned for the RTL fetch lane.

        Args:
            size_bytes: Number of bytes to allocate
        """
        self._next_program_dram_addr = self._align_up(self._next_program_dram_addr, align_bytes)
        program_dram_addr = self._next_program_dram_addr
        self._next_program_dram_addr += self._align_up(size_bytes, align_bytes)
        if label is not None:
            self._dram_addresses[label] = program_dram_addr
        return program_dram_addr

    def get_params_dram_addr(self) -> int:
        return self._next_params_dram_addr

    def get_program_dram_addr(self) -> int:
        return self._next_program_dram_addr

    def get_params_dram_usage(self) -> int:
        """Get the total params DRAM memory used so far."""
        return self._next_params_dram_addr - self._params_dram_base

    def get_program_dram_usage(self) -> int:
        """Get the total program DRAM memory used so far."""
        return self._next_program_dram_addr - self._program_dram_base

    def reset_params_dram_addr(self):
        self._next_params_dram_addr = self._params_dram_base

    def reset_program_dram_addr(self):
        self._next_program_dram_addr = self._program_dram_base

    def load_program_instructions_from_file(self, bin_path: str) -> tuple[int, int]:
        """Load a raw instruction bitstream into program DRAM.

        Uses :meth:`allocate_program_dram` (64 B–aligned reservation) and host-to-card DMA.
        On success, :meth:`get_program_dram_addr` is the next free aligned byte after the image,
        consistent with further ``allocate_program_dram`` / capture flush helpers.

        Returns:
            ``(start_addr, file_size_bytes)`` where ``file_size_bytes`` is the file length in bytes.

        Raises:
            RuntimeError: If the DMA write does not transfer the full file.
        """
        with open(bin_path, "rb") as f:
            data = f.read()
        size_bytes = len(data)
        start_addr = self.allocate_program_dram(size_bytes)
        written = self.dma_write(DMA_DEVICE_H2C, start_addr, data, size_bytes)
        if written != size_bytes:
            raise RuntimeError(
                f"load_program_instructions_from_file({bin_path!r}): DMA wrote {written} of "
                f"{size_bytes} bytes to program DRAM {start_addr:#x}"
            )
        print(f"    Loaded {size_bytes} bytes program instructions to DRAM at 0x{start_addr:x}")
        return start_addr, size_bytes

    def allocate_tensor_dram(self, size_bytes: int, label: Optional[str] = None, align_bytes: int = 64) -> int:
        """Allocate memory from the tensor DRAM region incrementally."""
        self._tensor_dram_addr = self._align_up(self._tensor_dram_addr, align_bytes)
        tensor_dram_addr = self._tensor_dram_addr
        self._tensor_dram_addr += size_bytes
        if label is not None:
            self._dram_addresses[label] = tensor_dram_addr
        return tensor_dram_addr

    def get_tensor_dram_usage(self) -> int:
        """Get the total tensor DRAM memory used so far."""
        return self._tensor_dram_addr - self._tensor_dram_base

    def reset_tensor_dram_addr(self):
        self._tensor_dram_addr = self._tensor_dram_base

    def get_tensor_dram_addr(self) -> int:
        return self._tensor_dram_addr

    def reset_isa_reg_counter(self) -> None:
        """Reset the ISA register allocation counter (register 0 is reserved)."""
        self._isa_reg_counter = 1

    def release_isa_reg(self) -> None:
        """Release the last allocated ISA register by decrementing the counter."""
        if self._isa_reg_counter > 1:
            self._isa_reg_counter -= 1

    def alloc_isa_reg(self) -> int:
        """
        Allocate the next available general-purpose ISA register (1-63).
        Register 0 is hard-wired zero.

        Returns:
            The allocated register index (1-63).
        """
        if self._isa_reg_counter > 63:
            raise ValueError("Exceeded maximum number of general registers (63)")

        reg_idx = self._isa_reg_counter
        self._isa_reg_counter += 1
        return reg_idx

    def reset_inst_ptr_counter(self) -> None:
        """Reset the instruction pointer allocation counter."""
        self._inst_ptr_counter = 1

    def release_inst_ptr(self, idx: int) -> None:
        """Release the last allocated instruction pointer by decrementing the counter."""
        if self._inst_ptr_counter > 1:
            self._inst_ptr_counter -= 1

    def alloc_inst_ptr(self) -> int:
        """
        Allocate the next available instruction pointer (1-15).

        Returns:
            The allocated instruction pointer index (1-15).
        """
        if self._inst_ptr_counter > 15:
            raise ValueError("Exceeded maximum number of instruction pointers (15)")

        idx = self._inst_ptr_counter
        self._inst_ptr_counter += 1
        return idx

    def chunk_ranges(self, total: int, chunk_size: int):
        """Yield (start_index, chunk_size) for each chunk. Last chunk may be smaller."""
        start = 0
        while start < total:
            take = min(chunk_size, total - start)
            yield start, take
            start += take

    def get_arg_max_index(self, rank: int = 1) -> int:
        """Get the rank-th argmax index tracked by the Unified Engine (1..4)."""
        argmax_reg_by_rank = {
            1: UE_ARGMAX1_INDEX,
            2: UE_ARGMAX2_INDEX,
            3: UE_ARGMAX3_INDEX,
            4: UE_ARGMAX4_INDEX,
        }
        if rank not in argmax_reg_by_rank:
            raise ValueError(f"rank={rank} must be in [1, 4]")
        return self.read_reg32(argmax_reg_by_rank[rank])

    def clear_inst_id(self) -> None:
        """Reset the instruction ID counter to 0 for the next capture session."""
        self._inst_id = 0

    def isa_add_set_core(self, dst_reg_idx: int, immediate_value: int, timeout_s: float = 10.0) -> None:
        """Emit and execute a one-instruction program that sets an ISA register to an immediate value.

        Sequence: clear_inst_id → start_capture → ADD SET → stop_capture → HALT →
        write to program DRAM → execute → wait_queue.
        """
        self.clear_inst_id()
        self.start_capture()
        self.generate_instruction_add_set(dst_reg_idx, immediate_value)
        self.stop_capture()
        self.generate_instruction_halt()
        program_addr = self.get_program_dram_addr()
        self.write_captured_instructions_to_dram(program_addr)
        self.allocate_program_dram(self.get_capture_instruction_size_bytes())
        self.clear_capture_buffer()
        self.start_execute_from_dram(program_addr)
        self.wait_queue(timeout_s)

    def write_captured_instructions_to_file(self, start_addr: int, filename: str = "captured_instructions.bin") -> None:
        """Serialize the current capture buffer to a binary file on disk.

        ``start_addr`` is unused (kept for call-site compatibility); ``filename`` is the output path.
        Prints a summary line on success; warns and returns early if the buffer is empty.
        """
        if not hasattr(self, 'capture_buffer') or not self.capture_buffer:
            print("Warning: No captured instructions to write to file.")
            return
        all_instructions_bytes = bytearray()
        for inst in self.capture_buffer:
            all_instructions_bytes.extend(inst.get_bytes())
        with open(filename, "wb") as f:
            f.write(all_instructions_bytes)
        print(f"Successfully wrote {len(self.capture_buffer)} captured instructions ({len(all_instructions_bytes)} bytes) to {filename}")

    def init_unified_engine(self):
        # Test user device register access first
        print(f"{DMA_DEVICE_USER} register access...")
        hw_version = self.user_read_reg32(UE_FPGA_VERSION_ADDR)
        print(f"HW version via user device: 0x{hw_version & 0xFFFFFFFF:08x}")
        expected_hw_version = 0xf2e8d12b if CURRENT_DEVICE == "efinix" else 0x93ffa0c8
        assert hw_version == expected_hw_version, (
            f"HW version mismatch: got 0x{hw_version & 0xFFFFFFFF:08x}, "
            f"expected 0x{expected_hw_version:08x}."
        )

        addr = UE_START_ADDR # first reg address offset
        while addr <= UE_LAST_REG_ADDR: # last reg address
            value = self.user_read_reg32(addr)
            if value != 0xdeadbeef:
                print(f"address 0x{addr:08x} = 0x{value & 0xFFFFFFFF:08x}")
            addr += 4

        # Dram read/write test
        print("FPGA DRAM read/write test")
        number_of_elements = 8192
        test_data = torch.randint(0x0000, 0xFFFF, (number_of_elements,), dtype=torch.uint16, device=self.device)
        self.dma_write(DMA_DEVICE_H2C, DRAM_START_ADDR, test_data, number_of_elements * 2)
        test_read_data = torch.zeros((number_of_elements,), dtype=torch.uint16, device=self.device)
        self.dma_read(DMA_DEVICE_C2H, DRAM_START_ADDR, test_read_data, number_of_elements * 2)
        if torch.allclose(test_read_data, test_data, atol=0):
            print("Dram read/write test passed")
        else:
            print("Dram read/write test failed")

        # Initialize Unified Engine hardware.
        # Newer Andromeda RTL derives delay controls internally. The current
        # Efinix 0x12345678 bitstream still exposes these registers.
        if CURRENT_DEVICE == "efinix":
            ue_output_valid_delay = (
                (UE_LATENCY_ADD_EXP << 27) +
                (UE_LATENCY_RMS << 22) +
                (UE_LATENCY_DOT_PRODUCT << 17) +
                (UE_LATENCY_ELTWISE_ADD << 13) +
                (UE_LATENCY_ELTWISE_MUL << 5) +
                (UE_LATENCY_EXP << 0)
            )
            self.write_reg32(UE_OUTPUT_VALID_DELAY_ADDR, ue_output_valid_delay)
            ue_valid_delay_extra = (
                (UE_LATENCY_BF20_ITRGT3 << 23) +
                (UE_LATENCY_BF20_ITR3 << 19) +
                (UE_LATENCY_BF20_ITR2 << 15) +
                ((UE_PIPELINE_BF19_MULT - 1) << 10) +
                ((UE_PIPELINE_BF19_ADD - 1) << 5) +
                (UE_LATENCY_ADD_REDUCE << 0)
            )
            self.write_reg32(UE_VALID_DELAY_EXTRA_ADDR, ue_valid_delay_extra)
            ue_lalu_delay = (
                ((UE_QINPUT_DELAY & 0x1F) << 21) +
                (UE_LATENCY_QUANTIZATION << 16) +
                (UE_LATENCY_QSCALE << 12) +
                (UE_LALU_LATENCY_ACT << 8) +
                (UE_LALU_LATENCY_RMS << 4) +
                (UE_LALU_LATENCY_SOFTMAX << 0)
            )
            self.write_reg32(UE_LALU_DELAY_ADDR, ue_lalu_delay)
        print("init_unified_engine()")
        print("Unified Engine initialization completed successfully!")

    def write_reg32(self, address: int, value: int):
        """Write 32-bit register using AXI-Lite user interface"""
        self.user_write_reg32(address, value)

    def read_reg32(self, address: int) -> int:
        """Read 32-bit register using AXI-Lite user interface"""
        return self.user_read_reg32(address)

    def user_write_reg32(self, address: int, value: int):
        """
        Write 32-bit register via /dev/xdma0_user (AXI-Lite interface)

        Args:
            address: Register address (absolute, will subtract USER_BASE_ADDR)
            value: 32-bit value to write
        """
        try:
            fd = os.open(DMA_DEVICE_USER, os.O_RDWR)
            try:
                # Subtract base address to get offset within the BAR
                offset = address + self._base_addr - AXI_LITE_TRANSLATION_OFFSET
                data_bytes = struct.pack('I', value & 0xFFFFFFFF)
                # Use pwrite to write at offset (device doesn't support lseek)
                os.pwrite(fd, data_bytes, offset)
            finally:
                os.close(fd)
        except PermissionError as e:
            print(f"Permission denied: {DMA_DEVICE_USER}")
            print(f"  Error: {e}")
            print(f"  Run with sudo or: sudo chmod 666 {DMA_DEVICE_USER}")
        except FileNotFoundError as e:
            print(f"Device not found: {DMA_DEVICE_USER}")
            print(f"  Error: {e}")
            print(f"  Ensure XDMA driver is loaded")
        except OSError as e:
            print(f"user_write_reg32 error: {e}")

    def user_read_reg32(self, address: int) -> int:
        """
        Read 32-bit register via /dev/xdma0_user (AXI-Lite interface)

        Args:
            address: Register address (absolute, will subtract USER_BASE_ADDR)

        Returns:
            32-bit register value, or 0 on error
        """
        try:
            fd = os.open(DMA_DEVICE_USER, os.O_RDONLY)
            try:
                # Subtract base address to get offset within the BAR
                offset = address + self._base_addr - AXI_LITE_TRANSLATION_OFFSET
                # Use pread to read at offset (device doesn't support lseek)
                data_bytes = os.pread(fd, 4, offset)
                if len(data_bytes) == 4:
                    return struct.unpack('I', data_bytes)[0]
                return 0
            finally:
                os.close(fd)
        except PermissionError as e:
            print(f"Permission denied: {DMA_DEVICE_USER}")
            print(f"  Error: {e}")
            print(f"  Run with sudo or: sudo chmod 666 {DMA_DEVICE_USER}")
            return 0
        except FileNotFoundError as e:
            print(f"Device not found: {DMA_DEVICE_USER}")
            print(f"  Error: {e}")
            print(f"  Ensure XDMA driver is loaded")
            return 0
        except OSError as e:
            print(f"user_read_reg32 error: {e}")
            return 0

    def is_queue_busy(self) -> bool:
        """Check if queue is busy by reading hardware register"""
        if self.is_capture_on:
            return False
        queue_reg = self.read_reg32(UE_QUEUE_CTRL_ADDR)
        # axi_top.sv UE_QUEUE_FIFO_DATA: queue_busy is bit [8].
        return ((queue_reg >> 8) & 0x1) == 1

    def wait_queue(self, timeout_seconds: float = 5.0):
        """
        Wait for queue to finish

        Args:
            timeout_seconds: Maximum time to wait in seconds (default: 10.0)
        """
        start_time = time.time()
        iteration = 0

        while self.is_queue_busy():
            time.sleep(0.01)  # 0.01s sleep
            iteration += 1

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                print(f"Error: wait_queue() timed out after {timeout_seconds:.1f} seconds ({iteration} iterations)")
                return

    def dma_write(self, device: str, address: int, buffer, size: int) -> int:
        """
        DMA write function (Host to Card)

        Args:
            device: Device path (e.g., DMA_DEVICE_H2C)
            address: Target address
            buffer: Data to write (torch.Tensor, int, or bytes-like)
            size: Size in bytes to write

        Returns:
            Number of bytes written, or -1 on error
        """
        try:
            # Convert buffer to bytes
            if isinstance(buffer, torch.Tensor):
                # Convert tensor to bytes
                # Handle bfloat16 specially by reinterpreting as uint16 and serializing
                if buffer.dtype == torch.bfloat16:
                    buffer_uint16 = buffer.view(torch.uint16).cpu().contiguous()
                    # Use PyTorch's numpy bridge (no direct numpy dependency/import)
                    data_bytes = buffer_uint16.numpy().tobytes()[:size]
                else:
                    # For other dtypes, serialize contiguous CPU tensor via numpy bridge
                    tensor_cpu = buffer.cpu().contiguous()
                    data_bytes = tensor_cpu.numpy().tobytes()[:size]
            elif isinstance(buffer, int):
                # Convert integer to bytes (for register writes)
                if size == 4:
                    data_bytes = struct.pack('I', buffer & 0xFFFFFFFF)
                elif size == 2:
                    data_bytes = struct.pack('H', buffer & 0xFFFF)
                else:
                    data_bytes = buffer.to_bytes(size, byteorder='little', signed=False)
            elif isinstance(buffer, (bytes, bytearray)):
                data_bytes = bytes(buffer[:size])
            else:
                # Try to convert to bytes
                data_bytes = bytes(buffer)[:size]

            # Open device file
            try:
                fd = os.open(device, os.O_RDWR)
            except PermissionError as e:
                print(f"Permission denied: {device}")
                print(f"  Error: {e}")
                print(f"  Device is owned by root. Solutions:")
                print(f"  1. Run with sudo: sudo python script.py")
                print(f"  2. Change device permissions: sudo chmod 666 {device}")
                print(f"  3. Create udev rule to set permissions automatically")
                return -1
            except FileNotFoundError as e:
                print(f"Device not found: {device}")
                print(f"  Error: {e}")
                print(f"  Solution: Ensure the DMA device driver is loaded")
                return -1
            except OSError as e:
                print(f"Failed to open device {device}: {e}")
                return -1

            try:
                # Seek to address
                os.lseek(fd, address, os.SEEK_SET)

                # Write data
                bytes_written = os.write(fd, data_bytes)

                return bytes_written
            finally:
                os.close(fd)

        except Exception as e:
            print(f"dma_write error: {e}")
            return -1

    def dma_read(self, device: str, address: int, buffer, size: int) -> int:
        """
        DMA read function (Card to Host)

        Args:
            device: Device path (e.g., DMA_DEVICE_C2H)
            address: Source address
            buffer: Buffer to read into (torch.Tensor or mutable buffer)
            size: Size in bytes to read

        Returns:
            Number of bytes read, or -1 on error
        """
        try:
            # Open device file
            try:
                fd = os.open(device, os.O_RDWR)
            except PermissionError as e:
                print(f"Permission denied: {device}")
                print(f"  Error: {e}")
                print(f"  Device is owned by root. Solutions:")
                print(f"  1. Run with sudo: sudo python script.py")
                print(f"  2. Change device permissions: sudo chmod 666 {device}")
                print(f"  3. Create udev rule to set permissions automatically")
                return -1
            except FileNotFoundError as e:
                print(f"Device not found: {device}")
                print(f"  Error: {e}")
                print(f"  Solution: Ensure the DMA device driver is loaded")
                return -1
            except OSError as e:
                print(f"Failed to open device {device}: {e}")
                return -1

            try:
                # Seek to address
                os.lseek(fd, address, os.SEEK_SET)

                # Read raw bytes from device
                data_bytes = os.read(fd, size)

                if len(data_bytes) == 0:
                    print(f"read failed: no data read")
                    return -1

                # Copy data into buffer
                if isinstance(buffer, torch.Tensor):
                    # Convert bytes to tensor using struct unpack and view
                    if buffer.dtype == torch.int32 and size == 4:
                        # 32-bit register read
                        # Unpack as signed int32
                        value = struct.unpack('i', data_bytes[:4])[0]
                        buffer[0] = value
                    else:
                        # Regular data read (bf16/uint16)
                        # Unpack as uint16 (generic type)
                        num_elements = size // 2
                        # Use struct.unpack to convert bytes to uint16 values
                        format_str = f'{num_elements}H'  # 'H' = unsigned short (uint16)
                        uint16_values = struct.unpack(format_str, data_bytes[:size])

                        # Create tensor from uint16 values
                        tensor_uint16 = torch.tensor(uint16_values, dtype=torch.uint16, device=buffer.device)

                        # Convert to target dtype using view
                        if buffer.dtype == torch.bfloat16:
                            # View uint16 as bfloat16
                            tensor_data = tensor_uint16.view(torch.bfloat16)
                        else:
                            # Use as-is or convert to target dtype
                            tensor_data = tensor_uint16.to(buffer.dtype)

                        # Copy to buffer
                        if buffer.numel() >= num_elements:
                            buffer[:num_elements].copy_(tensor_data[:num_elements])
                        else:
                            buffer.copy_(tensor_data[:buffer.numel()])
                else:
                    # For other buffer types, copy bytes directly
                    if hasattr(buffer, '__setitem__'):
                        buffer[:len(data_bytes)] = data_bytes

                return len(data_bytes)
            finally:
                os.close(fd)

        except Exception as e:
            print(f"dma_read error: {e}")
            return -1

    def ue_op_descriptor(
        self,
        inst_type: int = INSTRUCTION_UE_OP,
        inst_pointer_idx: Optional[int] = None,
        pbi_mode: int = PBI_MODE_INIT,
        pbi_field_select: PBI_FIELD = PBI_FIELD.DRAM_ADDR,
        broadcast_mode: int = 0,
        max_clear_en: int = 0,
        stride_z: int = 1,
        lalu_a: int = 0,
        lalu_b: int = 0,
        lalu_mode: int = LALU_MODE.BYPASS.value,
        scalar: int = 0,
        uram_bram: int = 0,
        uram_section: int = URAM_SECTION.URAM_A.value,
        uram_dst_addr: int = 0,
        dram_to_uram_cpy_start: int = 0,
        uram_wb_addr: int = 0,
        uram_write_src: int = URAM_WRITE_SRC.URAM_WRITE_BACK.value,
        mode: UE_MODE = UE_MODE.ELTWISE_ADD,
        data_type: int = TYPE.IF4.value,
        uram_a_start_addr: int = 0,
        uram_b_start_addr: int = 0,
        uram_length: int = 0,
        dma_start_addr: int = 0,
        dma_length: int = 0,
        output_size: int = 0,
        bias_adder_en: int = 0,
        stride_bytes_per_chunk: int = 0,
        stride_jump_bytes: int = 0,
        wb_padding_select: int = WB_PADDING_ZERO,
        general_reg_src: Optional[int] = None,
        fmax_context_addr: int = 0,
        pbi_stride_en: bool = False,
    ) -> None:
        """
        Shared 256b instruction descriptor compiler (queue replay / capture).

        Callers: :meth:`ue_memcpy_from_dram`, :meth:`ue_memcpy_to_dram`, and :meth:`ue_arithmetic_op`
        (compute modes only; ``ue_arithmetic_op`` rejects memcpy modes).

        Instruction index ``w[0][7:0]`` is taken from :attr:`_inst_id`, then :attr:`_inst_id` is
        incremented (same pattern as :meth:`ue_isa_descriptor` for ISA ops).

        Mode groups:
        - memcpy modes: ``UE_MODE.MEMCPY_FROM_DRAM`` and ``UE_MODE.URAM_DRAM_WRITEBACK``.
        - arithmetic modes: all remaining compute-centric queue ops.

        ``inst_type`` (``w[0][11:8]``): default ``INSTRUCTION_UE_OP`` (0); use ``INSTRUCTION_PBI_SET`` for pointer-row updates.
        For ``INSTRUCTION_PBI_SET``, ``inst_pointer_idx`` fills ``w[0][15:12]``, ``pbi_mode`` fills
        ``w[0][19:16]``, and ``pbi_field_select`` fills ``w[0][23:20]``.
        ``pbi_mode=PBI_MODE_REG`` is only valid with ``INSTRUCTION_PBI_SET`` (used by
        :meth:`generate_instruction_pbi_inc` when a GPR override is requested): the field
        selected by ``pbi_field_select`` is written **directly** from general register
        ``general_reg_src`` (``w[0][29:24]``) — its descriptor delta is discarded — while every
        other field still takes the increment path (row + descriptor delta).

        ``general_reg_src`` is valid only for a ``PBI_MODE_REG`` pointer-field update.
        """

        uram_start = int((mode != UE_MODE.DOT_PRODUCT) and
                        (mode != UE_MODE.URAM_DRAM_WRITEBACK) and
                        (mode != UE_MODE.DEQUANTIZE) and
                        (mode != UE_MODE.MEMCPY_FROM_DRAM))
        uram_bram_wb_start = int(mode == UE_MODE.URAM_DRAM_WRITEBACK)
        dma_start = int((mode == UE_MODE.DOT_PRODUCT) or (mode == UE_MODE.DEQUANTIZE))
        uram_length_z = dma_length >> 6  # in 64 element units

        wb_padding_control = 0
        if mode == UE_MODE.BF16_DOT_PRODUCT or mode == UE_MODE.DOT_PRODUCT:
            if max_clear_en == 1:
                wb_padding_control |= (1 << 1)
            if mode == UE_MODE.BF16_DOT_PRODUCT:
                wb_padding_control |= ((wb_padding_select & 1) << 0)

        if self.capture_buffer is not None and self.capture_count < MAX_DECODER_INSTRUCTIONS:
            tid = self._inst_id
            inst = Instructions()
            w = inst.words
            for i in range(8):
                w[i] = 0

            # PBI_SET descriptors seed the pointer row, including field 8
            # (URAM_MEMCPY_DST_ADDR), regardless of the dummy UE mode used by the
            # set instruction itself. Encode their tail with the memcpy layout so
            # generate_instruction_pbi_init(..., uram_dst_addr=...) is not lost.
            is_pbi_set = int(inst_type) == int(INSTRUCTION_PBI_SET)
            is_memcpy_mode = is_pbi_set or mode in (
                UE_MODE.MEMCPY_FROM_DRAM, UE_MODE.URAM_DRAM_WRITEBACK
            )
            # For UE_PBI the stride-z delta must be 0; the actual uram_row_stride_z comes from
            # pointer-row field 10. The descriptor stride_z for UE_OP is used as-is.
            if int(inst_type) == int(INSTRUCTION_UE_PBI):
                stride_z = 0
            if pbi_stride_en:
                assert int(inst_type) == int(INSTRUCTION_UE_PBI), \
                    "pbi_stride_en is only valid for UE_PBI instructions"
            stride_en = 1 if (stride_jump_bytes > 0 or pbi_stride_en) else 0
            output_size = stride_bytes_per_chunk if (stride_en and stride_bytes_per_chunk > 0) else output_size
            scalar = stride_jump_bytes if stride_en else scalar

            # [7:0] inst_id; [11:8] inst_type; [15:12] inst_ptr; [19:16] pbi_mode for PBI_SET; [31:16] lalu_a otherwise
            w[0] = ((tid & 0xFF) |
                    ((inst_type & 0xF) << 8) |
                    (((inst_pointer_idx or 0) & 0xF) << 12))
            # PBI_SET: [19:16] pointer_mode, [23:20] field_select; [29:24] pbi_general_reg_idx when PBI_MODE_REG.
            # Other types: [31:16] = lalu_a.
            if int(inst_type) == int(INSTRUCTION_PBI_SET):
                w[0] |= (int(pbi_mode) & 0xF) << 16
                w[0] |= (int(pbi_field_select) & 0xF) << 20
                if int(pbi_mode) == PBI_MODE_REG:
                    assert general_reg_src is not None, "general_reg_src is required for PBI_MODE_REG"
                    w[0] |= (int(general_reg_src) & 0x3F) << 24
            else:
                w[0] |= ((lalu_a & 0xFFFF) << 16)
            w[1] = ue_35bit_addr_shifter(dma_start_addr)
            w[2] = dma_length
            w[3] = ((uram_length & 0xFFF) |
                        ((uram_length_z & 0xFFF) << 12) |
                        (((uram_a_start_addr & 0xFFF) << 24)))
            w[4] = ((((uram_a_start_addr >> 8) & 0xF)) |
                        (((uram_b_start_addr & 0xFFF) << 4)) |
                        (((uram_wb_addr & 0xFFF) << 16)) |
                        (((output_size & 0xF) << 28)))
            w[5] = ((((output_size >> 4) & 0xFFF)) |
                        (((mode.value & 0xF) << 12)) |
                        (((uram_section & 1) << 16)) |
                        (((uram_write_src & 3) << 17)) |
                        (((dma_start & 1) << 19)) |
                        (((uram_start & 1) << 20)) |
                        (((data_type & 3) << 21)) |
                        (((lalu_mode & 7) << 23)) |
                        (((scalar & 0x3F) << 26)))
            fmx = fmax_context_addr & 0x3F
            w[6] = ((((scalar >> 6) & 0x7FFF)) |
                    (((wb_padding_control & 1) << 15)) |
                    (((max_clear_en & 1) << 16)) |
                    (((bias_adder_en & 1) << 17)) |
                    (((stride_z & 0xFFF) << 18)) |
                    (((stride_en & 1) << 30)) |
                    (((fmx & 1) << 31)))

            if is_memcpy_mode:
                w[7] = ((((dram_to_uram_cpy_start & 1) << 7)) |
                            (((uram_bram & 3) << 8)) |
                            (((uram_dst_addr & 0xFFF) << 10)) |
                            (((uram_bram_wb_start & 1) << 22)))
            else:
                w[7] = ((((fmx >> 1) & 0x1F)) |
                            (((broadcast_mode & 3) << 5)) |
                            (((lalu_b & 0xFFFF) << 7)))

            self.capture_buffer.append(inst)
            self.capture_count += 1
            self._inst_id = tid + 1

    def ue_memcpy_from_dram(self, dram_src_addr: int, memcpy_length_bytes: int,
                            memcpy_type: int, uram_dst_addr: int,
                            uram_type: int,
                            stride_bytes_per_chunk: int = 0,
                            stride_jump_bytes: int = 0,
                            inst_pointer_idx: Optional[int] = None,
                            ):
        """
        Memory copy from DRAM to URAM/BRAM. Emits the UE descriptor via :meth:`ue_op_descriptor` only.

        Direct AXI register programming for immediate execution has been removed; use instruction
        capture and replay from DRAM (or another path that programs the UE) to run the transfer.

        Args:
            dram_src_addr: Source address in DRAM
            memcpy_length_bytes: Number of bytes to copy
            memcpy_type: Type of memory (MEMCPY_TYPE.URAM, BRAM, BIAS_BRAM, SCALE_BRAM)
            uram_dst_addr: Destination address in URAM (only meaningful for URAM type)
            uram_type: URAM section (URAM_SECTION.URAM_A or URAM_B, only meaningful for URAM type)
            stride_bytes_per_chunk: Bytes to copy per stride (0 = no stride mode)
            stride_jump_bytes: Distance in bytes between start of consecutive copies in DRAM
            inst_pointer_idx: when nonzero, emit PBI-style memcpy (pointer-backed registers are incremented by the immediate value specified in the instruction).
        """
        if inst_pointer_idx is not None:
            inst_type = INSTRUCTION_UE_PBI
            encoded_dram_addr = dram_src_addr
        else:
            inst_type = INSTRUCTION_UE_OP
            encoded_dram_addr = dram_src_addr
        self.ue_op_descriptor(
            inst_type=inst_type,
            inst_pointer_idx=inst_pointer_idx,
            broadcast_mode=0,
            max_clear_en=0,
            stride_z=0,
            lalu_a=0,
            lalu_b=0,
            lalu_mode=0,
            scalar=0,
            uram_bram=int(memcpy_type),
            uram_section=int(uram_type),
            uram_dst_addr=uram_dst_addr,
            dram_to_uram_cpy_start=1,
            uram_wb_addr=0,
            uram_write_src=URAM_WRITE_SRC.URAM_DRAM.value,
            mode=UE_MODE.MEMCPY_FROM_DRAM,
            data_type=0,
            uram_a_start_addr=0,
            uram_b_start_addr=0,
            uram_length=0,
            dma_start_addr=encoded_dram_addr,
            dma_length=memcpy_length_bytes,
            output_size=0,
            bias_adder_en=0,
            stride_bytes_per_chunk=stride_bytes_per_chunk,
            stride_jump_bytes=stride_jump_bytes,
            fmax_context_addr=0,
        )

    def ue_memcpy_to_dram(self, memcpy_type: int, uram_type: int,
                         uram_src_addr: int, dram_dst_addr: int,
                         memcpy_length_bytes: int,
                         stride_bytes_per_chunk: int = 0,
                         stride_jump_bytes: int = 0,
                         inst_pointer_idx: Optional[int] = None,
                         pbi_stride_en: bool = False,
                         ):
        """
        Memory copy from URAM to DRAM. Emits the UE descriptor via :meth:`ue_op_descriptor` only.

        Supports stride mode: When stride parameters are provided, the DMA will write
        data in strides to non-contiguous DRAM locations.

        Args:
            memcpy_type: Type of memory (MEMCPY_TYPE.URAM, BRAM, etc.)
            uram_type: URAM section (URAM_SECTION.URAM_A or URAM_B)
            uram_src_addr: Source address in URAM
            dram_dst_addr: Destination address in DRAM
            memcpy_length_bytes: Total number of bytes to copy (should be multiple of 128)
            stride_bytes_per_chunk: Bytes to write per stride (0 = no stride mode)
            stride_jump_bytes: Distance in bytes between start of consecutive writes in DRAM
                              (0 = contiguous, use stride_bytes_per_chunk for the jump)
            inst_pointer_idx: when nonzero, emit PBI-style memcpy (pointer-backed registers are incremented by the immediate value specified in the instruction).

        Example (stride writeback):
            # Write 2 rows (128 bytes each) with 256 byte gaps between them in DRAM
            # Row 0 -> DRAM[0:128], Row 1 -> DRAM[256:384]
            ue.set_stride_mode(128, 256, enable=True)  # Enable stride mode first
            ue.ue_memcpy_to_dram(MEMCPY_TYPE.URAM, URAM_A, 0, dst_addr, 256,
                                 stride_bytes_per_chunk=128, stride_jump_bytes=256)
            ue.wait_queue()
            ue.clear_stride_mode()
        """
        if inst_pointer_idx is not None:
            inst_type = INSTRUCTION_UE_PBI
            encoded_dram_addr = dram_dst_addr
        else:
            inst_type = INSTRUCTION_UE_OP
            encoded_dram_addr = dram_dst_addr
        self.ue_op_descriptor(
            inst_type=inst_type,
            inst_pointer_idx=inst_pointer_idx,
            broadcast_mode=0,
            max_clear_en=0,
            stride_z=1,
            lalu_a=0,
            lalu_b=0,
            lalu_mode=0,
            scalar=0,
            uram_bram=int(memcpy_type),
            uram_section=int(uram_type),
            uram_dst_addr=0,
            dram_to_uram_cpy_start=0,
            uram_wb_addr=0,
            uram_write_src=0,
            mode=UE_MODE.URAM_DRAM_WRITEBACK,
            data_type=0,
            uram_a_start_addr=uram_src_addr,
            uram_b_start_addr=uram_src_addr,
            uram_length=0,
            dma_start_addr=encoded_dram_addr,
            dma_length=memcpy_length_bytes,
            output_size=0,
            bias_adder_en=0,
            stride_bytes_per_chunk=stride_bytes_per_chunk,
            stride_jump_bytes=stride_jump_bytes,
            pbi_stride_en=pbi_stride_en,
        )

    def stride_mode_memcpy_benchmark(self,
                     input_dram_addr: int,
                     output_dram_addr: int,
                     stride_bytes_per_chunk: int,
                     stride_jump_bytes: int,
                     memcpy_length_bytes: int,
                     program_dram_addr: Optional[int] = None) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Run stride mode memcpy operation using the captured instruction stream.
        """
        bytes_per_element = 2
        total_size = stride_jump_bytes * memcpy_length_bytes // stride_bytes_per_chunk
        print(f"total_size: {total_size}")

        # Start instruction capture
        self.start_capture()

        remaining_bytes = memcpy_length_bytes
        aligned_max_chunk_bytes = (URAM_NEAR_FULL_SIZE // stride_bytes_per_chunk) * stride_bytes_per_chunk
        input_dram_addr_offset = input_dram_addr
        output_dram_addr_offset = output_dram_addr
        while remaining_bytes > 0:
            chunk_bytes = min(remaining_bytes, aligned_max_chunk_bytes)

            stride_enabled_way = True
            if stride_enabled_way:
                self.ue_memcpy_from_dram(
                    input_dram_addr_offset,
                    chunk_bytes,
                    MEMCPY_TYPE.URAM,
                    URAM_START_ADDR,
                    URAM_SECTION.URAM_A.value,
                    stride_bytes_per_chunk=stride_bytes_per_chunk,
                    stride_jump_bytes=stride_jump_bytes,
                )
                self.wait_queue()

                self.ue_memcpy_to_dram(MEMCPY_TYPE.URAM.value, URAM_SECTION.URAM_A.value, URAM_START_ADDR, output_dram_addr_offset, chunk_bytes)
                self.wait_queue()
            else:
                print(f"memcpy_length_bytes: {memcpy_length_bytes}")
                print(f"stride_bytes_per_chunk: {stride_bytes_per_chunk}")
                print(f"stride_jump_bytes: {stride_jump_bytes}")
                for i in range(memcpy_length_bytes // stride_bytes_per_chunk):
                    self.ue_memcpy_from_dram(input_dram_addr_offset + i * stride_jump_bytes, stride_bytes_per_chunk, MEMCPY_TYPE.URAM, URAM_START_ADDR + (i * stride_bytes_per_chunk) // UE_VECTOR_SIZE, URAM_SECTION.URAM_A.value)
                    self.wait_queue()

                self.ue_memcpy_to_dram(MEMCPY_TYPE.URAM.value, URAM_SECTION.URAM_A.value, URAM_START_ADDR, output_dram_addr_offset, memcpy_length_bytes)
                self.wait_queue()

            remaining_bytes -= chunk_bytes
            input_dram_addr_offset += chunk_bytes * stride_jump_bytes // stride_bytes_per_chunk
            output_dram_addr_offset += chunk_bytes
        # Finish capture and write instruction stream to DRAM
        self.stop_capture()
        self.generate_instruction_halt()

        if program_dram_addr is None:
            program_dram_addr = self.get_program_dram_addr()
            print(f"Using program DRAM address: {program_dram_addr}")
            program_size_bytes = self.write_captured_instructions_to_dram(program_dram_addr)
            self.allocate_program_dram(program_size_bytes)
        else:
            print(f"Using provided program DRAM address: {program_dram_addr}")
            self.write_captured_instructions_to_dram(program_dram_addr)

        def handler(matrix):
            """
            Run permute operation using the captured instruction stream.
            takes matrix dimension (dim_0, dim_1, dim_2) and return (dim_1, dim_0, dim_2)
            """
            # Extract tensor data and check if DMA can be skipped
            if isinstance(matrix, DeviceTensor):
                matrix_data = matrix._data  # Use _data to avoid triggering fetch
                skip_dma = not matrix.needs_dma(input_dram_addr)
                if skip_dma:
                    print(f"[DMA cache hit] Skipping input DMA to {hex(input_dram_addr)}")
                else:
                    print(f"[DMA write] Writing input matrix to {hex(input_dram_addr)}")
            else:
                matrix_data = matrix
                skip_dma = False
                print(f"Please consider using DeviceTensor for DMA cache optimization")

            # Validate input
            assert matrix_data.dtype == torch.bfloat16, "Input matrix must be in bf16 format"

            # Write input matrix to DRAM (skip if DeviceTensor already synced)
            if skip_dma:
                print(f"[DMA cache hit] Skipping input DMA to {hex(input_dram_addr)}")
            else:
                if isinstance(matrix, DeviceTensor):
                    matrix.sync(input_dram_addr) # sync the data to the FPGA DRAM
                else:
                    self.dma_write(DMA_DEVICE_H2C, input_dram_addr, matrix_data.flatten(), total_size)

            # Start executing from DRAM and wait for completion
            self.start_execute_from_dram(program_dram_addr)
            self.wait_queue()
            self.report_timing_and_instruction_count()
            # tranpose is elemental access for each element 1 ops per element
            _gflops, _ = self.report_flop_rate_gflops(memcpy_length_bytes / bytes_per_element)
            print(f"flops: {_gflops:.3f} GFLOPS")

            if isinstance(matrix, DeviceTensor):
                return DeviceTensor((memcpy_length_bytes // stride_bytes_per_chunk, stride_bytes_per_chunk // bytes_per_element), ue=self, dram_addr=output_dram_addr)
            else:
                return DeviceTensor((memcpy_length_bytes // stride_bytes_per_chunk, stride_bytes_per_chunk // bytes_per_element), ue=self, dram_addr=output_dram_addr).data

        return handler

    def ue_arithmetic_op(self, broadcast_mode: int, max_clear_en: int, stride_z: int,
                   lalu_a: int, lalu_b: int, lalu_mode: int, scalar: int,
                   uram_section: int, uram_dst_addr: int,
                   uram_wb_addr: int,
                   uram_write_src: int, mode: UE_MODE, data_type: int,
                   uram_a_start_addr: int, uram_b_start_addr: int,
                   uram_length: int, dma_start_addr: int, dma_length: int,
                   output_size: int, bias_adder_en: int = 0,
                   stride_bytes_per_chunk: int = 0, stride_jump_bytes: int = 0,
                   wb_padding_select: int = WB_PADDING_ZERO,
                   fmax_context_addr: int = 0,
                   inst_pointer_idx: Optional[int] = None,
                   ) -> Optional[torch.Tensor]:
        """
        Emit a non-memcpy UE operation as a 256-bit descriptor via :meth:`ue_op_descriptor`.
        Pairs with :meth:`ue_memcpy_from_dram` / :meth:`ue_memcpy_to_dram` for memcpy modes.

        inst_pointer_idx: when nonzero, emit PBI-style UE op (pointer-backed registers are incremented by the immediate value specified in the instruction).

        Returns:
            None (legacy API; no tensor result)
        """
        if mode == UE_MODE.URAM_DRAM_WRITEBACK or mode == UE_MODE.MEMCPY_FROM_DRAM:
            raise ValueError(
                "ue_arithmetic_op does not handle memcpy modes; use ue_memcpy_from_dram() or ue_memcpy_to_dram()"
            )
        inst_type = INSTRUCTION_UE_PBI if inst_pointer_idx is not None else INSTRUCTION_UE_OP
        self.ue_op_descriptor(
            inst_type=inst_type,
            inst_pointer_idx=inst_pointer_idx,
            broadcast_mode=broadcast_mode,
            max_clear_en=max_clear_en,
            stride_z=stride_z,
            lalu_a=lalu_a,
            lalu_b=lalu_b,
            lalu_mode=lalu_mode,
            scalar=scalar,
            uram_bram=MEMCPY_TYPE.URAM.value,
            uram_section=uram_section,
            uram_dst_addr=uram_dst_addr,
            dram_to_uram_cpy_start=0,
            uram_wb_addr=uram_wb_addr,
            uram_write_src=uram_write_src,
            mode=mode,
            data_type=data_type,
            uram_a_start_addr=uram_a_start_addr,
            uram_b_start_addr=uram_b_start_addr,
            uram_length=uram_length,
            dma_start_addr=dma_start_addr,
            dma_length=dma_length,
            output_size=output_size,
            bias_adder_en=bias_adder_en,
            stride_bytes_per_chunk=stride_bytes_per_chunk,
            stride_jump_bytes=stride_jump_bytes,
            wb_padding_select=wb_padding_select,
            fmax_context_addr=fmax_context_addr,
        )
        return None

    def dma_to_accelerator_memory(self, dma_address: int, data: torch.Tensor) -> None:
        """DMA data from host memory to accelerator memory"""
        assert data.dtype == torch.bfloat16, "Data must be in bf16 format"
        self.dma_write(DMA_DEVICE_H2C, dma_address, data, data.numel() * 2) # 2 bytes per element

    def dma_from_accelerator_memory(self, dma_address: int, shape: torch.Size) -> torch.Tensor:
        """DMA data from accelerator memory to host memory"""
        data = torch.zeros(shape, dtype=torch.bfloat16).flatten()
        self.dma_read(DMA_DEVICE_C2H, dma_address, data, data.numel() * 2) # 2 bytes per element
        return data.reshape(shape)

    # Describe URAM A and URAM B as follows:
    # URAM A is from 0x00000 to 0x7FFFF (2^19 * 64 * 2 = 524288 bytes)
    # URAM B is from 0x80000 to 0xFFFFF (2^19 * 64 * 2 = 524288 bytes)
    def sram_address_to_uram_address(self, sram_address: int) -> Tuple[int, int]:
        """Convert SRAM address to URAM address"""
        assert sram_address & 0x7F == 0, "SRAM address must be aligned to 128 bytes"
        if ((sram_address >> 19) & 0x1):
            uram_bank = URAM_SECTION.URAM_B
            assert sram_address <= 0xFFFFF and sram_address >= 0x80000, f"sram_address={hex(sram_address)} is not in the range of URAM_B"
        else:
            uram_bank = URAM_SECTION.URAM_A
            assert sram_address <= 0x7FFFF and sram_address >= 0x00000, f"sram_address={hex(sram_address)} is not in the range of URAM_A"
        uram_address = (sram_address >> 7) & 0xFFF
        return uram_bank, uram_address

    # -------------------------------------------------------------------------
    # Wrappers for UE eltwise and broadcast ops (wrap ue_arithmetic_op with fixed mode)
    # Used by user_dma_ops.py eltwise_op, eltwise_add, eltwise_mul, broadcast_op.
    # -------------------------------------------------------------------------

    def accelerator_memory_to_scale_sram(self, accelerator_dram_address: int, element_size: int) -> None:
        """DMA data from accelerator memory to scale SRAM"""
        element_size_bytes = element_size * 2
        self.ue_memcpy_from_dram(accelerator_dram_address, element_size_bytes, MEMCPY_TYPE.BRAM.value, 0, 0)

    def accelerator_memory_to_sram(
        self,
        accelerator_dram_address: int,
        sram_address: int,
        element_size: int,
        stride_bytes_per_chunk: int = 0,
        stride_jump_bytes: int = 0,
        inst_pointer_idx: Optional[int] = None,
        memcpy_length_bytes: Optional[int] = None,
        general_reg_src: Optional[int] = None,
    ) -> None:
        """
        DMA data from accelerator memory to SRAM (URAM).

        Default (``inst_pointer_idx == 0``): ``accelerator_dram_address`` is the absolute DRAM
        source address; transfer size is ``memcpy_length_bytes`` if given, else ``element_size * 2``
        bytes (bf16 elements).

        Pointer mode (``inst_pointer_idx != 0``): emit a PBI-backed memcpy. The first length field
        in the descriptor is ``accelerator_dram_address`` (DRAM pointer **increment** / delta in
        bytes, not an absolute address). The second is ``memcpy_length_bytes`` if given, else
        ``element_size * 2``. **You must issue** :meth:`generate_instruction_pbi_init` first for
        the same pointer index so base address and length live in the pointer row.

        """
        uram_type, uram_start_addr = self.sram_address_to_uram_address(sram_address)
        nbytes = element_size * 2 if memcpy_length_bytes is None else memcpy_length_bytes
        if general_reg_src is not None:
            if inst_pointer_idx is not None:
                raise ValueError("general_reg_src and inst_pointer_idx are mutually exclusive")
            ptr = self.alloc_inst_ptr()
            try:
                self.generate_instruction_pbi_init(
                    dma_length=nbytes, uram_dst_addr=uram_start_addr,
                    inst_pointer_idx=ptr)
                self.generate_instruction_pbi_inc(
                    inst_pointer_idx=ptr, pbi_field_select=PBI_FIELD.DRAM_ADDR,
                    general_reg_src=general_reg_src)
                self.ue_memcpy_from_dram(
                    0, 0, MEMCPY_TYPE.URAM.value, 0, uram_type.value,
                    inst_pointer_idx=ptr)
            finally:
                self.release_inst_ptr(ptr)
            return
        self.ue_memcpy_from_dram(
            accelerator_dram_address,
            nbytes,
            MEMCPY_TYPE.URAM.value,
            uram_start_addr,
            uram_type.value,
            stride_bytes_per_chunk=stride_bytes_per_chunk,
            stride_jump_bytes=stride_jump_bytes,
            inst_pointer_idx=inst_pointer_idx,
        )

    def sram_to_accelerator_memory(
        self,
        sram_address: int,
        accelerator_dram_address: int,
        element_size: int,
        stride_bytes_per_chunk: int = 0,
        stride_jump_bytes: int = 0,
        inst_pointer_idx: Optional[int] = None,
        memcpy_length_bytes: Optional[int] = None,
        general_reg_src: Optional[int] = None,
    ) -> None:
        """
        DMA data from SRAM (URAM) to accelerator memory.

        Default (``inst_pointer_idx == 0``): ``accelerator_dram_address`` is the absolute DRAM
        destination; transfer size is ``memcpy_length_bytes`` if given, else ``element_size * 2``.

        Pointer mode (``inst_pointer_idx != 0``): ``accelerator_dram_address`` is the DRAM pointer
        **increment** for the writeback descriptor field; length is ``memcpy_length_bytes`` if
        given, else ``element_size * 2``. **Requires** a prior :meth:`generate_instruction_pbi_init`
        for that pointer index.

        """
        uram_type, uram_start_addr = self.sram_address_to_uram_address(sram_address)
        if stride_bytes_per_chunk != 0:
            ue_assert_axi_beat_aligned_bytes(
                stride_bytes_per_chunk, "sram_to_accelerator_memory: stride_bytes_per_chunk")
        nbytes = element_size * 2 if memcpy_length_bytes is None else memcpy_length_bytes
        if general_reg_src is not None:
            if inst_pointer_idx is not None:
                raise ValueError("general_reg_src and inst_pointer_idx are mutually exclusive")
            ptr = self.alloc_inst_ptr()
            try:
                self.generate_instruction_pbi_init(
                    dma_length=nbytes, uram_a_start_addr=uram_start_addr,
                    uram_b_start_addr=uram_start_addr, inst_pointer_idx=ptr)
                self.generate_instruction_pbi_inc(
                    inst_pointer_idx=ptr, pbi_field_select=PBI_FIELD.DRAM_ADDR,
                    general_reg_src=general_reg_src)
                self.ue_memcpy_to_dram(
                    MEMCPY_TYPE.URAM.value, uram_type.value, 0, 0, 0,
                    inst_pointer_idx=ptr)
            finally:
                self.release_inst_ptr(ptr)
            return
        self.ue_memcpy_to_dram(
            MEMCPY_TYPE.URAM.value,
            uram_type.value,
            uram_start_addr,
            accelerator_dram_address,
            nbytes,
            stride_bytes_per_chunk=stride_bytes_per_chunk,
            stride_jump_bytes=stride_jump_bytes,
            inst_pointer_idx=inst_pointer_idx,
        )

    def accelerator_memcpy(
        self,
        dram_src_addr: int,
        dram_dst_addr: int,
        size_in_bytes: int,
        gr_src_addr: Optional[int] = None,
        gr_dst_addr: Optional[int] = None,
    ) -> None:
        """DRAM-to-DRAM copy via URAM[0] as a temporary staging buffer.

        If ``gr_src_addr`` is provided the source address is taken from ISA register
        ``gr_src_addr`` at runtime and ``dram_src_addr`` is ignored (pass 0).
        If ``gr_dst_addr`` is provided the destination address is taken from ISA register
        ``gr_dst_addr`` at runtime and ``dram_dst_addr`` is ignored (pass 0).
        """
        self.accelerator_memory_to_sram(
            accelerator_dram_address=0 if gr_src_addr is not None else dram_src_addr,
            sram_address=0,
            element_size=0,
            memcpy_length_bytes=size_in_bytes,
            general_reg_src=gr_src_addr,
        )
        self.sram_to_accelerator_memory(
            sram_address=0,
            accelerator_dram_address=0 if gr_dst_addr is not None else dram_dst_addr,
            element_size=0,
            memcpy_length_bytes=size_in_bytes,
            general_reg_src=gr_dst_addr,
        )

    def accelerator_memory_to_bias_sram(
        self,
        accelerator_dram_address: int,
        element_size: int,
        inst_pointer_idx: Optional[int] = None,
    ) -> None:
        """Copy from accelerator DRAM to bias BRAM. element_size is in elements (bf16 = 2 bytes per element).

        ``inst_pointer_idx``: when provided, emit PBI-style memcpy (pointer-backed increment).
        """
        size_bytes = element_size * 2
        assert size_bytes <= BIAS_BRAM_SIZE_BYTES, f"size_bytes={size_bytes} must be less than or equal to BIAS_BRAM_SIZE_BYTES={BIAS_BRAM_SIZE_BYTES}"
        self.ue_memcpy_from_dram(
            accelerator_dram_address,
            size_bytes,
            MEMCPY_TYPE.BIAS_BRAM.value,
            0,
            0,
            inst_pointer_idx=inst_pointer_idx,
        )

    # Element-wise operations ---------------------------------------------------
    def start_queue_eltwise(
        self,
        mode: UE_MODE,
        uram_a_start_addr: int,
        uram_b_start_addr: int,
        uram_wb_addr: int,
        uram_section: int,
        row_size: int,
        inst_pointer_idx: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """
        Start queue for element-wise op: ELTWISE_ADD, ELTWISE_MUL, or ELTWISE_SUB.
        Wraps :meth:`ue_arithmetic_op` with standard eltwise parameters.

        ``inst_pointer_idx`` is forwarded when provided (PBI / UE_OP_REG path); default ``None`` keeps
        prior behavior.
        """
        if mode not in (UE_MODE.ELTWISE_ADD, UE_MODE.ELTWISE_MUL, UE_MODE.ELTWISE_SUB):
            raise ValueError(f"Invalid eltwise mode: {mode}. Use ELTWISE_ADD, ELTWISE_MUL, or ELTWISE_SUB.")
        self.ue_arithmetic_op(
            0,  # broadcast_mode
            0,  # max_clear_en
            1,  # stride_z
            0,  # lalu_a
            0,  # lalu_b
            LALU_MODE.BYPASS.value,  # lalu_mode
            0,  # scalar (not used)
            uram_section,
            0,  # uram_dst_addr
            uram_wb_addr,
            URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
            mode,
            0,  # data_type
            uram_a_start_addr,
            uram_b_start_addr,
            row_size,
            0,  # dma_start_addr
            0,  # dma_length
            0,  # output_size
            inst_pointer_idx=inst_pointer_idx,
        )

    def eltwise_add_core(self, vector_A_sram_start_addr: int, vector_B_sram_start_addr: int, vector_C_sram_wb_addr: int,
                                element_size: int) -> Optional[torch.Tensor]:
        """Start queue for element-wise add. Wraps ue_arithmetic_op with UE_MODE.ELTWISE_ADD."""
        row_size = (element_size + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
        vector_A_uram_type, vector_A_uram_start_addr = self.sram_address_to_uram_address(vector_A_sram_start_addr)
        vector_B_uram_type, vector_B_uram_start_addr = self.sram_address_to_uram_address(vector_B_sram_start_addr)
        assert vector_A_uram_type != vector_B_uram_type, "vector_A_sram_start_addr and vector_B_sram_start_addr must be in the different URAMs"
        if vector_A_uram_type == URAM_SECTION.URAM_A:
            uram_a_start_addr = vector_A_uram_start_addr
            uram_b_start_addr = vector_B_uram_start_addr
        else:
            uram_a_start_addr = vector_B_uram_start_addr
            uram_b_start_addr = vector_A_uram_start_addr
        uram_wb_type, uram_wb_addr = self.sram_address_to_uram_address(vector_C_sram_wb_addr)
        self.start_queue_eltwise(
            UE_MODE.ELTWISE_ADD,
            uram_a_start_addr, uram_b_start_addr,
            uram_wb_addr, uram_wb_type.value, row_size
        )

    def eltwise_mul_core(self, vector_A_sram_start_addr: int, vector_B_sram_start_addr: int, vector_C_sram_wb_addr: int,
                                element_size: int) -> Optional[torch.Tensor]:
        """Start queue for element-wise multiply. Wraps ue_arithmetic_op with UE_MODE.ELTWISE_MUL."""
        row_size = (element_size + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
        vector_A_uram_type, vector_A_uram_start_addr = self.sram_address_to_uram_address(vector_A_sram_start_addr)
        vector_B_uram_type, vector_B_uram_start_addr = self.sram_address_to_uram_address(vector_B_sram_start_addr)
        assert vector_A_uram_type != vector_B_uram_type, "vector_A_sram_start_addr and vector_B_sram_start_addr must be in the different URAMs"
        if vector_A_uram_type == URAM_SECTION.URAM_A:
            uram_a_start_addr = vector_A_uram_start_addr
            uram_b_start_addr = vector_B_uram_start_addr
        else:
            uram_a_start_addr = vector_B_uram_start_addr
            uram_b_start_addr = vector_A_uram_start_addr
        uram_wb_type, uram_wb_addr = self.sram_address_to_uram_address(vector_C_sram_wb_addr)
        self.start_queue_eltwise(
            UE_MODE.ELTWISE_MUL,
            uram_a_start_addr, uram_b_start_addr,
            uram_wb_addr, uram_wb_type.value, row_size
        )

    def eltwise_sub_core(self, vector_A_sram_start_addr: int, vector_B_sram_start_addr: int, vector_C_sram_wb_addr: int,
                                element_size: int) -> Optional[torch.Tensor]:
        """Start queue for element-wise subtract (A - B). Wraps ue_arithmetic_op with UE_MODE.ELTWISE_SUB.

        A must live in URAM_A and B in URAM_B (order matters: SUB is non-commutative).
        """
        row_size = (element_size + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
        vector_A_uram_type, vector_A_uram_start_addr = self.sram_address_to_uram_address(vector_A_sram_start_addr)
        vector_B_uram_type, vector_B_uram_start_addr = self.sram_address_to_uram_address(vector_B_sram_start_addr)
        assert vector_A_uram_type == URAM_SECTION.URAM_A, \
            "eltwise_sub_core: minuend (vector_A) must be in URAM_A"
        assert vector_B_uram_type == URAM_SECTION.URAM_B, \
            "eltwise_sub_core: subtrahend (vector_B) must be in URAM_B"
        uram_wb_type, uram_wb_addr = self.sram_address_to_uram_address(vector_C_sram_wb_addr)
        self.start_queue_eltwise(
            UE_MODE.ELTWISE_SUB,
            vector_A_uram_start_addr, vector_B_uram_start_addr,
            uram_wb_addr, uram_wb_type.value, row_size
        )

    def _validate_addr_gprs(self, fn: str, gpr_M_reg: Optional[int], addr_gprs: dict) -> None:
        """Shared validation for optional gpr_*_addr params: each must be a GPR index 1..63,
        distinct from ``gpr_M_reg`` (the row-count register) and from each other. ``None`` entries
        (literal-address fallback) are skipped. ``addr_gprs`` maps param-name -> reg-or-None."""
        for _name, _reg in addr_gprs.items():
            if _reg is None:
                continue
            if not (1 <= _reg <= 63):
                raise ValueError(f"{fn}: {_name} must be a GPR index 1..63, got {_reg}")
            if gpr_M_reg is not None and _reg == gpr_M_reg:
                raise ValueError(
                    f"{fn}: {_name}={_reg} collides with gpr_M_reg={gpr_M_reg}; address GPRs "
                    "must be distinct from the row-count GPR"
                )
        _provided = [_r for _r in addr_gprs.values() if _r is not None]
        if len(set(_provided)) != len(_provided):
            raise ValueError(f"{fn}: address GPRs must be distinct from each other, got {addr_gprs}")

    def _pbi_override_dram_base_from_gpr(self, inst_pointer_idx: int, gpr_addr_reg: int) -> None:
        """Emit one PBI_MODE_REG override that replaces PBI pointer ``inst_pointer_idx``'s
        DRAM_ADDR base with the **word** address (``byte_addr >> 3``) held in ISA GPR
        ``gpr_addr_reg``.

        Must be emitted once, AFTER the pointer's ``generate_instruction_pbi_init`` and BEFORE
        the hardware loop, so the in-loop ``PBI_INC`` row strides still advance from the
        GPR-sourced base. This is the shared mechanism that lets a single captured program serve
        any DRAM placement (caller re-primes the GPR before each replay). The +0 INC deltas leave
        every other pointer field untouched; only DRAM_ADDR is overridden.
        """
        self.generate_instruction_pbi_inc(
            dram_shared_addr=0,
            dma_length=0,
            output_size=0,
            uram_length=0,
            uram_a_start_addr=0,
            uram_b_start_addr=0,
            uram_wb_addr=0,
            uram_dst_addr=0,
            fmax_context_addr=0,
            inst_pointer_idx=inst_pointer_idx,
            general_reg_src=gpr_addr_reg,
            pbi_field_select=PBI_FIELD.DRAM_ADDR,
        )

    def eltwise_core_dram(
        self,
        M: int,
        N: int,
        dram_a: int,
        dram_b: Optional[int],
        dram_out: int,
        mode: UE_MODE,
        gpr_M_reg: Optional[int] = None,
        scalar: float = None,
        gpr_a_addr: Optional[int] = None,
        gpr_b_addr: Optional[int] = None,
        gpr_out_addr: Optional[int] = None,
        gpr_N_reg: Optional[int] = None,
        gpr_scalar_reg: Optional[int] = None,
    ) -> int:
        """Dispatches to the dynamic core when any runtime register is supplied.

        - Any runtime dimension, address, or scalar GPR: :meth:`eltwise_core_dram_dynamic` —
          runtime **M and N**, all modes (vector ELTWISE_ADD/MUL/SUB and broadcast MUL/ADD_BROADCAST).
          Missing M/N registers are allocated and seeded from the compile-time values.
        - No runtime GPRs (default): :meth:`eltwise_core_dram_legacy` — compile-time ``m_chunk``
          tiling, no GPR needed.

        Vector modes (ELTWISE_ADD/MUL/SUB): ``dram_b`` required, ``scalar`` ignored.
        Broadcast modes (MUL_BROADCAST/ADD_BROADCAST): ``scalar`` required, ``dram_b`` must be None.

        ``gpr_a_addr`` / ``gpr_b_addr`` / ``gpr_out_addr`` (dynamic path only) optionally source each
        DRAM base from a GPR (word address, ``byte >> 3``) so one captured program serves any
        placement. They require ``gpr_M_reg`` to be set. ``gpr_scalar_reg`` (MUL_BROADCAST only)
        writes shared PBI field 11, interpreted as ``lalu_scalar`` by arithmetic UE_PBI.

        Returns ``M * N`` (one flop per output BF16 element).
        """
        if any(r is not None for r in (gpr_M_reg, gpr_N_reg, gpr_a_addr, gpr_b_addr,
                                       gpr_out_addr, gpr_scalar_reg)):
            seeded_regs = []
            if gpr_M_reg is None:
                gpr_M_reg = self.alloc_isa_reg(); seeded_regs.append(gpr_M_reg)
                self.generate_instruction_add_set(gpr_M_reg, M)
            if gpr_N_reg is None:
                gpr_N_reg = self.alloc_isa_reg(); seeded_regs.append(gpr_N_reg)
                self.generate_instruction_add_set(gpr_N_reg, N)
            result = self.eltwise_core_dram_dynamic(
                M=M, N=N,
                dram_a=dram_a, dram_b=dram_b, dram_out=dram_out,
                mode=mode, scalar=scalar,
                gpr_M_reg=gpr_M_reg, gpr_N_reg=gpr_N_reg,
                gpr_a_addr=gpr_a_addr,
                gpr_b_addr=gpr_b_addr,
                gpr_out_addr=gpr_out_addr,
                gpr_scalar_reg=gpr_scalar_reg,
            )
            for _ in seeded_regs:
                self.release_isa_reg()
            return result
        return self.eltwise_core_dram_legacy(
            M=M, N=N,
            dram_a=dram_a, dram_b=dram_b, dram_out=dram_out,
            mode=mode, scalar=scalar,
        )


    def eltwise_core_dram_legacy(
        self,
        M: int,
        N: int,
        dram_a: int,
        dram_b: Optional[int],
        dram_out: int,
        mode: UE_MODE,
        scalar: float = None,
    ) -> int:
        """
        Legacy BF16 element-wise op on ``[M, N]`` tensors in DRAM with compile-time vertical tiling.

        Vector modes (ELTWISE_ADD/MUL/SUB): ``dram_b`` required. Tiles are Python-unrolled
        (no ISA loop, no jumps).
        Broadcast modes (MUL_BROADCAST/ADD_BROADCAST): ``scalar`` required, ``dram_b`` None.
        Each row is loaded to URAM_A and the scalar broadcast is applied in-place.

        **Staging (fixed):** A at ``0x00000`` (URAM_A), B at ``0x80000`` (URAM_B) for vector modes.
        """
        _VECTOR_MODES = (UE_MODE.ELTWISE_ADD, UE_MODE.ELTWISE_MUL, UE_MODE.ELTWISE_SUB)
        _BROADCAST_MODES = (UE_MODE.MUL_BROADCAST, UE_MODE.ADD_BROADCAST)
        fn = "eltwise_core_dram_legacy"
        if mode not in _VECTOR_MODES + _BROADCAST_MODES:
            raise ValueError(f"{fn}: unsupported mode {mode!r}")
        is_broadcast = mode in _BROADCAST_MODES
        if is_broadcast:
            if scalar is None:
                raise ValueError(f"{fn}: scalar required for {mode!r}")
            if dram_b is not None:
                raise ValueError(f"{fn}: dram_b must be None for broadcast modes")
        else:
            if dram_b is None:
                raise ValueError(f"{fn}: dram_b required for {mode!r}")
        if M < 1 or N < 1:
            raise ValueError(f"{fn}: require M>=1 and N>=1, got M={M}, N={N}")
        if N % UE_VECTOR_SIZE != 0:
            raise ValueError(f"{fn}: N must be a multiple of UE_VECTOR_SIZE={UE_VECTOR_SIZE}, got N={N}")
        if N > URAM_NEAR_FULL_ELEMENTS:
            raise ValueError(
                f"{fn}: N={N} exceeds near-full URAM row ({URAM_NEAR_FULL_ELEMENTS} BF16 slots)."
            )

        def _emit_tile(elements: int) -> None:
            if mode == UE_MODE.ELTWISE_ADD:
                self.eltwise_add_core(0x00000, 0x80000, 0x00000, elements)
            elif mode == UE_MODE.ELTWISE_MUL:
                self.eltwise_mul_core(0x00000, 0x80000, 0x00000, elements)
            elif mode == UE_MODE.ELTWISE_SUB:
                self.eltwise_sub_core(0x00000, 0x80000, 0x00000, elements)
            elif mode == UE_MODE.MUL_BROADCAST:
                self.broadcast_mul(scalar=scalar, sram_start_addr=0x00000, sram_wb_addr=0x00000, element_size=elements)
            else:  # ADD_BROADCAST
                self.broadcast_add(scalar=scalar, sram_start_addr=0x00000, sram_wb_addr=0x00000, element_size=elements)

        m_chunk = URAM_NEAR_FULL_ELEMENTS // N
        if m_chunk < 1:
            raise ValueError(
                f"{fn}: N={N} too large; need at least one row within URAM_NEAR_FULL_ELEMENTS={URAM_NEAR_FULL_ELEMENTS}."
            )

        for i, m_take in self.chunk_ranges(M, m_chunk):
            byte_off = i * N * 2
            elements = m_take * N
            self.accelerator_memory_to_sram(
                accelerator_dram_address=dram_a + byte_off,
                sram_address=0x00000,
                element_size=elements,
            )
            if not is_broadcast:
                self.accelerator_memory_to_sram(
                    accelerator_dram_address=dram_b + byte_off,
                    sram_address=0x80000,
                    element_size=elements,
                )
            _emit_tile(elements)
            self.sram_to_accelerator_memory(
                sram_address=0x00000,
                accelerator_dram_address=dram_out + byte_off,
                element_size=elements,
            )

        return M * N

    def eltwise_core_dram_pbi(
        self,
        M: int,
        N: int,
        dram_a: int,
        dram_b: Optional[int],
        dram_out: int,
        mode: UE_MODE,
        gpr_M_reg: int,
        scalar: float = None,
        gpr_a_addr: Optional[int] = None,
        gpr_b_addr: Optional[int] = None,
        gpr_out_addr: Optional[int] = None,
        gpr_scalar_reg: Optional[int] = None,
    ) -> int:
        # DEPRECATED REFERENCE ONLY: deliberately unreachable, even under ``python -O``.
        raise RuntimeError("eltwise_core_dram_pbi is deprecated; use eltwise_core_dram_dynamic")
        """
        PBI BF16 element-wise op on ``[M, N]`` DRAM tensors (**no vertical tiling**).

        ``gpr_scalar_reg`` (broadcast modes only): GPR holding the bf16 bit-pattern of a **runtime**
        broadcast scalar. Software initializes a complete compute pointer row and writes this GPR
        to shared field 11 using ``PBI_FIELD.LALU_SCALAR``; the broadcast is a normal UE_PBI.
        ``None`` uses the baked immediate ``scalar``.

        Vector modes (ELTWISE_ADD/MUL/SUB): ``dram_b`` required. Loads A→URAM_A, B→URAM_B,
        applies op, stores result.
        Broadcast modes (MUL_BROADCAST/ADD_BROADCAST): ``scalar`` required, ``dram_b`` must be
        None. Loads A→URAM_A, applies scalar broadcast in-place, stores result. No ptr_b allocated.

        Loads/processes/stores **one row per ISA iteration**: ``dma_length = N * 2`` bytes per PBI op,
        DRAM strides ``row_bytes`` each iteration via hardware loop counters (:meth:`loop_start` /
        :meth:`loop_end`). An abs-jump anchor is emitted before the loop so the rel-JNZ from
        ``loop_end`` always lands within the 512-instruction i-cache window.
        No ``m_chunk`` / remnant tail path unlike :meth:`eltwise_core_dram`.

        Requires ``N <= URAM_NEAR_FULL_ELEMENTS`` so one row fits staging.

        **Staging (fixed):** A at ``0x00000``, B at ``0x80000``, output reuses A.

        **Batch dimension / M (dynamic, required):**
        ``gpr_M_reg`` is a **required** GPR index 1..15 holding the runtime row count. Caller
        must prime that register beforehand (typically with ``ADD_SET``). The hardware loop
        runs for whatever value that register holds at execute time; ``M`` is still required
        as a compile-time argument purely for FLOPs accounting / asserts — the captured
        program contains no static reference to ``M``.

        **DRAM base addresses (optional dynamic):**
        ``gpr_a_addr`` / ``gpr_b_addr`` / ``gpr_out_addr`` are optional GPR indices (1..63). When
        a GPR is provided, the corresponding pointer's **base** is sourced from that register at
        execute time instead of the ``dram_a`` / ``dram_b`` / ``dram_out`` literal, so a single
        captured program can be replayed against any DRAM placement by re-priming the GPR before
        each replay. The GPR must hold the **word** address (``byte_addr >> 3``, the PBI
        DRAM_ADDR format — same convention as :meth:`matmat_mul_core_dynamic`). Mechanism: a
        one-shot ``PBI_MODE_REG`` override (``PBI_FIELD.DRAM_ADDR``) emitted once after the
        pointer init replaces the literal base with the GPR value; the in-loop ``PBI_INC`` row
        strides still advance it. When a GPR is ``None`` the literal base is used and **no extra
        instruction is emitted** (byte-for-byte identical to the legacy captured program). The
        per-iteration row stride (``N*2``) stays a compile-time immediate; only the base moves.

        Returns:
            ``M * N`` (one flop per output BF16 element). Since ``M`` is FLOPs-accounting only,
            callers feeding this to :meth:`report_flop_rate_gflops` should pass the realized
            row count separately if it differs from the compile-time ``M``.
        """
        fn = "eltwise_core_dram_pbi"
        _VECTOR_MODES = (UE_MODE.ELTWISE_ADD, UE_MODE.ELTWISE_MUL, UE_MODE.ELTWISE_SUB)
        _BROADCAST_MODES = (UE_MODE.MUL_BROADCAST, UE_MODE.ADD_BROADCAST)
        if mode not in _VECTOR_MODES + _BROADCAST_MODES:
            raise ValueError(
                f"{fn}: mode must be ELTWISE_ADD/MUL/SUB or MUL_BROADCAST/ADD_BROADCAST, got {mode!r}"
            )
        is_broadcast = mode in _BROADCAST_MODES
        if is_broadcast:
            if scalar is None:
                raise ValueError(f"{fn}: scalar required for {mode!r}")
            if dram_b is not None:
                raise ValueError(f"{fn}: dram_b must be None for broadcast modes, got {dram_b:#x}")
        else:
            if dram_b is None:
                raise ValueError(f"{fn}: dram_b required for {mode!r}")
        if M < 1 or N < 1:
            raise ValueError(f"{fn}: require M>=1 and N>=1, got M={M}, N={N}")
        if N % UE_VECTOR_SIZE != 0:
            raise ValueError(
                f"{fn}: N must be a multiple of UE_VECTOR_SIZE={UE_VECTOR_SIZE}, got N={N}"
            )
        if N > URAM_NEAR_FULL_ELEMENTS:
            raise ValueError(
                f"{fn}: N={N} exceeds URAM_NEAR_FULL_ELEMENTS={URAM_NEAR_FULL_ELEMENTS}; "
                "one row must fit staging."
            )
        if not (1 <= gpr_M_reg <= 15):
            raise ValueError(
                f"{fn}: gpr_M_reg must be a GPR index 1..15, got {gpr_M_reg}"
            )
        self._validate_addr_gprs(fn, gpr_M_reg, {"gpr_a_addr": gpr_a_addr})
        self._validate_addr_gprs(fn, gpr_M_reg, {"gpr_b_addr": gpr_b_addr})
        self._validate_addr_gprs(fn, gpr_M_reg, {"gpr_out_addr": gpr_out_addr})
        if gpr_scalar_reg is not None:
            if mode != UE_MODE.MUL_BROADCAST:
                raise ValueError(f"{fn}: gpr_scalar_reg is only supported for MUL_BROADCAST, got {mode!r}")
            if not (1 <= gpr_scalar_reg <= 63):
                raise ValueError(f"{fn}: gpr_scalar_reg must be a GPR index 1..63, got {gpr_scalar_reg}")

        row_bytes = N * 2
        ptr_scalar = None  # complete UE_PBI compute row for a runtime broadcast scale

        def _emit_row() -> None:
            if mode == UE_MODE.ELTWISE_ADD:
                self.eltwise_add_core(0x00000, 0x80000, 0x00000, N)
            elif mode == UE_MODE.ELTWISE_MUL:
                self.eltwise_mul_core(0x00000, 0x80000, 0x00000, N)
            elif mode == UE_MODE.ELTWISE_SUB:
                self.eltwise_sub_core(0x00000, 0x80000, 0x00000, N)
            elif mode == UE_MODE.MUL_BROADCAST:
                if ptr_scalar is None:
                    self.broadcast_mul(scalar=scalar, sram_start_addr=0x00000,
                                       sram_wb_addr=0x00000, element_size=N)
                else:
                    # All runtime state is already in ptr_scalar. Zero descriptor fields are PBI
                    # increments, so the pointer row remains unchanged across loop iterations.
                    self.broadcast_mul(scalar=0.0, sram_start_addr=0x00000,
                                       sram_wb_addr=0x00000, element_size=0,
                                       inst_pointer_idx=ptr_scalar)
            else:  # ADD_BROADCAST
                self.broadcast_add(scalar=scalar, sram_start_addr=0x00000, sram_wb_addr=0x00000, element_size=N)

        _, a_uram_row = self.sram_address_to_uram_address(0x00000)
        _, b_uram_row = self.sram_address_to_uram_address(0x80000)
        _, out_uram_row = self.sram_address_to_uram_address(0x00000)

        ptr_a = self.alloc_inst_ptr()
        ptr_b = self.alloc_inst_ptr() if not is_broadcast else None
        ptr_out = self.alloc_inst_ptr()

        self.generate_instruction_pbi_init(
            dram_shared_addr=dram_a,
            dma_length=row_bytes,
            output_size=0,
            uram_length=0,
            uram_a_start_addr=0,
            uram_b_start_addr=0,
            uram_wb_addr=0,
            uram_dst_addr=a_uram_row,
            fmax_context_addr=0,
            inst_pointer_idx=ptr_a,
        )
        if ptr_b is not None:
            self.generate_instruction_pbi_init(
                dram_shared_addr=dram_b,
                dma_length=row_bytes,
                output_size=0,
                uram_length=0,
                uram_a_start_addr=0,
                uram_b_start_addr=0,
                uram_wb_addr=0,
                uram_dst_addr=b_uram_row,
                fmax_context_addr=0,
                inst_pointer_idx=ptr_b,
            )
        self.generate_instruction_pbi_init(
            dram_shared_addr=dram_out,
            dma_length=row_bytes,
            output_size=0,
            uram_length=0,
            uram_a_start_addr=out_uram_row,
            uram_b_start_addr=out_uram_row,
            uram_wb_addr=0,
            uram_dst_addr=0,
            fmax_context_addr=0,
            inst_pointer_idx=ptr_out,
        )

        # Optional: override each pointer's BASE from an ISA GPR (word address = byte >> 3) so
        # one captured program serves any DRAM placement. PBI_MODE_REG replaces the DRAM_ADDR
        # field with the GPR value once here (before the loop); the in-loop PBI_INC row strides
        # still advance the pointer. Emitted only for the GPRs the caller supplied, so the
        # all-None case captures a byte-for-byte identical program to before.
        _gpr_pairs = [(ptr_a, gpr_a_addr), (ptr_out, gpr_out_addr)]
        if ptr_b is not None:
            _gpr_pairs.append((ptr_b, gpr_b_addr))
        for _ptr_idx, _gpr_addr in _gpr_pairs:
            if _gpr_addr is not None:
                self._pbi_override_dram_base_from_gpr(_ptr_idx, _gpr_addr)

        # Runtime broadcast scale: initialize a complete arithmetic pointer row. Field 11 is the
        # LALU scalar for UE_PBI arithmetic modes; field 10 supplies the Y-row stride.
        if ptr_scalar is None and gpr_scalar_reg is not None:
            ptr_scalar = self.alloc_inst_ptr()
            self.generate_instruction_pbi_init(
                uram_length=N // UE_VECTOR_SIZE,
                uram_a_start_addr=a_uram_row,
                uram_wb_addr=out_uram_row,
                uram_row_stride_z=1,
                inst_pointer_idx=ptr_scalar,
            )
            self.generate_instruction_pbi_inc(
                general_reg_src=gpr_scalar_reg,
                pbi_field_select=PBI_FIELD.LALU_SCALAR,
                inst_pointer_idx=ptr_scalar,
            )

        # Abs-jump anchor: reloads i-cache so the RELA_JNZ from loop_end() lands within
        # the 512-instruction window.
        program_dram_start_addr = self.get_program_dram_addr()
        cur_inst_count = self.capture_count
        self.generate_instruction_jump_abs(
            ue_35bit_addr_shifter(
                program_dram_start_addr + (cur_inst_count + 1) * INSTRUCTION_SIZE_BYTES
            )
        )
        self.loop_start(loop_cnt=M, gpr_loop_cnt=gpr_M_reg)

        self.accelerator_memory_to_sram(
            accelerator_dram_address=row_bytes,
            sram_address=0x00000,
            element_size=0,
            inst_pointer_idx=ptr_a,
        )
        if ptr_b is not None:
            self.accelerator_memory_to_sram(
                accelerator_dram_address=row_bytes,
                sram_address=0x80000,
                element_size=0,
                inst_pointer_idx=ptr_b,
            )
        _emit_row()
        self.sram_to_accelerator_memory(
            sram_address=0x00000,
            accelerator_dram_address=row_bytes,
            element_size=0,
            inst_pointer_idx=ptr_out,
        )

        outer_loop_size = self.loop_end()
        assert outer_loop_size <= 256, (
            f"{fn}: outer loop body {outer_loop_size} instructions exceeds i-cache budget 256"
        )

        if ptr_scalar is not None:
            self.release_inst_ptr(ptr_scalar)
        self.release_inst_ptr(ptr_out)
        if ptr_b is not None:
            self.release_inst_ptr(ptr_b)
        self.release_inst_ptr(ptr_a)

        return M * N

    # Flat-chunk geometry for eltwise_core_dram_dynamic: an [M, N] element-wise op on contiguous
    # row-major tensors is a 1-D op on M*N elements, so it is tiled into compile-time-sized chunks
    # (fast plain compute + batched DMA) with a runtime-sized tail. Power of two so the runtime
    # chunk count is a shift of M*N, not a divide.
    ELTWISE_DYN_CHUNK_LOG2 = 16
    ELTWISE_DYN_CHUNK_ELEMS = 1 << ELTWISE_DYN_CHUNK_LOG2          # 65536 elements
    ELTWISE_DYN_CHUNK_ROWS = ELTWISE_DYN_CHUNK_ELEMS // UE_VECTOR_SIZE  # 1024 URAM rows
    ELTWISE_DYN_CHUNK_BYTES = ELTWISE_DYN_CHUNK_ELEMS * 2

    def eltwise_core_dram_dynamic(
        self,
        M: int,
        N: int,
        dram_a: int,
        dram_b: Optional[int],
        dram_out: int,
        mode: UE_MODE,
        gpr_M_reg: int,
        gpr_N_reg: int,
        scalar: float = None,
        gpr_a_addr: Optional[int] = None,
        gpr_b_addr: Optional[int] = None,
        gpr_out_addr: Optional[int] = None,
        gpr_scalar_reg: Optional[int] = None,
    ) -> int:
        """BF16 element-wise op on ``[M, N]`` with **both M and N runtime**, chunk-tiled for speed.

        Because the tensors are contiguous row-major and element-wise ops have no row structure,
        the ``[M, N]`` problem is executed as a flat ``M*N``-element stream: ``total = M*N`` full
        :attr:`ELTWISE_DYN_CHUNK_ELEMS`-element chunks run in a runtime hardware loop
        (``total >> CHUNK_LOG2`` trips) where each chunk is ONE batched DMA in, ONE **plain**
        compute op with compile-time chunk dims (identical descriptor to the legacy fast path),
        and ONE batched DMA out — so full chunks run at legacy speed. The sub-chunk tail
        (``total & (CHUNK-1)``, always a multiple of 64 since N is) runs once after the loop with
        its DMA lengths and compute row-size sourced from registers via PBI field overrides.

        The vector-mode tail compute is emitted **UE_PBI** via a URAM-only compute pointer
        (element-wise modes are PBI-safe; only reductions are not). The one bitstream-gated feature is
        ``gpr_scalar_reg`` (runtime MUL_BROADCAST scale), which rides shared PBI field 11
        (a set-once scalar pointer); without that bitstream it falls back to the baked ``scalar``. The
        runtime-scalar broadcast is always a plain UE_OP — full chunks and the tail both over-compute a
        bounded chunk of URAM rows (only rows_take are stored), so no UE_PBI is needed.

        Vector modes (ELTWISE_ADD/MUL/SUB): ``dram_b`` required. Broadcast modes
        (MUL_BROADCAST/ADD_BROADCAST): ``scalar`` required, ``dram_b`` must be ``None``.
        ``gpr_M_reg`` 1..15, ``gpr_N_reg`` 1..63 (distinct), caller-primed via ADD_SET; runtime N
        must be a multiple of 64. ``gpr_a/b/out_addr`` optionally source the DRAM bases from GPRs
        (word addr = ``byte >> 3``). ``M``/``N`` literals are template/FLOPs only.

        Live PBI pointers: A-load, B-load (vector only), writeback, compute — ≤4, with ≤3 used
        inside the loop body (the compute pointer is only dispatched in the loop for the
        runtime-scalar broadcast variant).
        """
        fn = "eltwise_core_dram_dynamic"
        _VECTOR_MODES = (UE_MODE.ELTWISE_ADD, UE_MODE.ELTWISE_MUL, UE_MODE.ELTWISE_SUB)
        _BROADCAST_MODES = (UE_MODE.MUL_BROADCAST, UE_MODE.ADD_BROADCAST)
        if mode not in _VECTOR_MODES + _BROADCAST_MODES:
            raise ValueError(f"{fn}: mode must be ELTWISE_ADD/MUL/SUB or MUL_BROADCAST/ADD_BROADCAST, got {mode!r}")
        is_broadcast = mode in _BROADCAST_MODES
        if is_broadcast:
            if scalar is None:
                raise ValueError(f"{fn}: scalar required for {mode!r}")
            if dram_b is not None:
                raise ValueError(f"{fn}: dram_b must be None for broadcast modes")
        else:
            if dram_b is None:
                raise ValueError(f"{fn}: dram_b required for {mode!r}")
        if M < 1 or N < 1:
            raise ValueError(f"{fn}: require M>=1 and N>=1, got M={M}, N={N}")
        if N % UE_VECTOR_SIZE != 0:
            raise ValueError(f"{fn}: N must be a multiple of UE_VECTOR_SIZE={UE_VECTOR_SIZE}, got N={N}")
        if gpr_M_reg is None or not (1 <= gpr_M_reg <= 15):
            raise ValueError(f"{fn}: gpr_M_reg must be a GPR index 1..15, got {gpr_M_reg}")
        if gpr_N_reg is None or not (1 <= gpr_N_reg <= 63):
            raise ValueError(f"{fn}: gpr_N_reg must be a GPR index 1..63, got {gpr_N_reg}")
        if gpr_M_reg == gpr_N_reg:
            raise ValueError(f"{fn}: gpr_M_reg/gpr_N_reg must be distinct")
        for _nm, _rg in (("gpr_a_addr", gpr_a_addr), ("gpr_b_addr", gpr_b_addr), ("gpr_out_addr", gpr_out_addr)):
            self._validate_addr_gprs(fn, gpr_M_reg, {_nm: _rg})
            if _rg is not None and _rg == gpr_N_reg:
                raise ValueError(f"{fn}: {_nm}={_rg} collides with gpr_N_reg={gpr_N_reg}")
        if gpr_scalar_reg is not None:
            if mode != UE_MODE.MUL_BROADCAST:
                raise ValueError(f"{fn}: gpr_scalar_reg is only supported for MUL_BROADCAST, got {mode!r}")
            if not (1 <= gpr_scalar_reg <= 63):
                raise ValueError(f"{fn}: gpr_scalar_reg must be a GPR index 1..63, got {gpr_scalar_reg}")
        assert self.is_capture_on, f"{fn}() requires active capture"

        CHUNK_LOG2 = self.ELTWISE_DYN_CHUNK_LOG2
        CHUNK_ELEMS = self.ELTWISE_DYN_CHUNK_ELEMS
        CHUNK_ROWS = self.ELTWISE_DYN_CHUNK_ROWS
        CHUNK_BYTES = self.ELTWISE_DYN_CHUNK_BYTES

        SRAM_A = 0x00000
        SRAM_B = 0x80000
        _, a_uram_row = self.sram_address_to_uram_address(SRAM_A)
        _, b_uram_row = self.sram_address_to_uram_address(SRAM_B)
        # Template tail (descriptor fallback values only; runtime values come from registers).
        template_total = M * N
        template_tail = template_total & (CHUNK_ELEMS - 1)
        template_chunks = template_total >> CHUNK_LOG2

        _alloc_list = []
        def _alloc():
            r = self.alloc_isa_reg(); _alloc_list.append(r); return r

        total_reg      = _alloc()  # M*N elements
        n_chunks_reg   = _alloc()  # total >> CHUNK_LOG2 (full-chunk trips)
        tail_rows_reg  = _alloc()  # (total mod CHUNK)/64 (tail compute URAM_ROW_SIZE; 0 iff no tail)
        tail_bytes_reg = _alloc()  # (total mod CHUNK)*2 (tail DMA length)
        chunk_rows_reg = _alloc()  # constant CHUNK_ROWS (populates the compute pointer's row size)
        zb_reg         = _alloc() if not is_broadcast else None  # constant b-operand URAM row
        # UE_PBI broadcasts source their Y-row stride from pointer field 10.
        one_reg        = _alloc() if is_broadcast else None

        self.generate_instruction_mul32_reg(total_reg, gpr_M_reg, gpr_N_reg)
        self.generate_instruction_shr(n_chunks_reg, total_reg, CHUNK_LOG2)
        self.generate_instruction_shl(tail_rows_reg, n_chunks_reg, CHUNK_LOG2)       # n_chunks*CHUNK
        self.generate_instruction_reg_sub(tail_rows_reg, total_reg, tail_rows_reg)   # tail elements
        self.generate_instruction_shl(tail_bytes_reg, tail_rows_reg, 1)              # tail*2 bytes
        self.generate_instruction_shr(tail_rows_reg, tail_rows_reg, 6)               # tail/64 rows
        self.generate_instruction_add_set(chunk_rows_reg, CHUNK_ROWS)
        if zb_reg is not None:
            self.generate_instruction_add_set(zb_reg, b_uram_row)
        if one_reg is not None:
            self.generate_instruction_add_set(one_reg, 1)

        def _set(ptr, field, reg):
            self.generate_instruction_pbi_inc(general_reg_src=reg, pbi_field_select=field, inst_pointer_idx=ptr)

        # DMA pointers: base + baked chunk auto-advance (pointer-mode DMA advances its DRAM cursor
        # by the in-op increment after each transfer — same HW-validated pattern as
        # eltwise_core_dram_pbi). Optional GPR base override replaces the literal base one-shot.
        ptr_a = self.alloc_inst_ptr()
        ptr_b = self.alloc_inst_ptr() if not is_broadcast else None
        ptr_out = self.alloc_inst_ptr()
        ptr_comp = self.alloc_inst_ptr()
        self.generate_instruction_pbi_init(dram_shared_addr=dram_a, dma_length=CHUNK_BYTES,
                                           uram_dst_addr=a_uram_row, inst_pointer_idx=ptr_a)
        if gpr_a_addr is not None:
            self._pbi_override_dram_base_from_gpr(ptr_a, gpr_a_addr)
        if ptr_b is not None:
            self.generate_instruction_pbi_init(dram_shared_addr=dram_b, dma_length=CHUNK_BYTES,
                                               uram_dst_addr=b_uram_row, inst_pointer_idx=ptr_b)
            if gpr_b_addr is not None:
                self._pbi_override_dram_base_from_gpr(ptr_b, gpr_b_addr)
        self.generate_instruction_pbi_init(dram_shared_addr=dram_out, dma_length=CHUNK_BYTES,
                                           uram_a_start_addr=a_uram_row, uram_b_start_addr=a_uram_row,
                                           inst_pointer_idx=ptr_out)
        if gpr_out_addr is not None:
            self._pbi_override_dram_base_from_gpr(ptr_out, gpr_out_addr)

        # URAM-only compute pointer: read A row 0 (Y), B band (Z, vector modes), write back to A
        # row 0; row size = CHUNK_ROWS for the loop, re-set to the tail rows after it. Used by the
        # tail compute (runtime row size) and by vector full chunks; broadcast full chunks avoid it.
        self.generate_instruction_pbi_init(
            inst_pointer_idx=ptr_comp,
            lalu_scalar=(self.float_to_bf16(scalar) if is_broadcast else 0),
        )
        _set(ptr_comp, PBI_FIELD.URAM_START_ADDR_Y, 0)   # reg 0 = hardwired zero (A row 0)
        if zb_reg is not None:
            _set(ptr_comp, PBI_FIELD.URAM_START_ADDR_Z, zb_reg)
        _set(ptr_comp, PBI_FIELD.URAM_WRITEB_ADDR, 0)    # A row 0
        _set(ptr_comp, PBI_FIELD.URAM_ROW_SIZE, chunk_rows_reg)
        if one_reg is not None:
            # A multi-row UE_PBI broadcast must advance Y rather than re-read row 0.
            _set(ptr_comp, PBI_FIELD.URAM_ROW_STRIDE_Z, one_reg)
        if gpr_scalar_reg is not None:
            # For arithmetic UE_PBI, shared pointer field 11 is interpreted as lalu_scalar.
            _set(ptr_comp, PBI_FIELD.LALU_SCALAR, gpr_scalar_reg)

        def _emit_compute(rows: int, elems: int, pbi_tail: bool) -> None:
            """One chunk/tail compute. Full chunks are plain baked ops (fast path); the vector-mode
            and all broadcast operations that need runtime pointer state use UE_PBI via ptr_comp."""
            if not is_broadcast:
                if pbi_tail:
                    self.start_queue_eltwise(mode, a_uram_row, b_uram_row, a_uram_row,
                                             URAM_SECTION.URAM_A.value, rows, inst_pointer_idx=ptr_comp)
                else:
                    self.start_queue_eltwise(mode, a_uram_row, b_uram_row, a_uram_row,
                                             URAM_SECTION.URAM_A.value, rows)
            elif mode == UE_MODE.MUL_BROADCAST:
                if gpr_scalar_reg is not None:
                    # ptr_comp supplies row size, Y/WB addresses, stride and runtime scalar. The
                    # descriptor contributes zero increments so the pointer remains set-once.
                    self.broadcast_mul(scalar=0.0, sram_start_addr=SRAM_A, sram_wb_addr=SRAM_A,
                                       element_size=0, inst_pointer_idx=ptr_comp)
                elif pbi_tail:
                    # Baked scale, runtime tail size: UE_PBI via ptr_comp (stride_z=1 via field-10).
                    self.broadcast_mul(scalar=0.0, sram_start_addr=SRAM_A, sram_wb_addr=SRAM_A,
                                       element_size=0, inst_pointer_idx=ptr_comp)
                else:
                    # Full chunk, baked scale: plain baked broadcast (fast path).
                    self.broadcast_mul(scalar=scalar, sram_start_addr=SRAM_A, sram_wb_addr=SRAM_A,
                                       element_size=elems)
            else:  # ADD_BROADCAST
                self.broadcast_add(scalar=scalar, sram_start_addr=SRAM_A, sram_wb_addr=SRAM_A,
                                   element_size=elems,
                                   inst_pointer_idx=ptr_comp if pbi_tail else None)

        program_dram_start_addr = self.get_program_dram_addr()

        # ---- full chunks (skipped entirely when total < CHUNK) ----
        patch_loop = self._emit_forward_skip_jz(n_chunks_reg, program_dram_start_addr)
        cur_inst_count = self.capture_count
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(
            program_dram_start_addr + (cur_inst_count + 1) * INSTRUCTION_SIZE_BYTES))
        self.loop_start(loop_cnt=template_chunks, gpr_loop_cnt=n_chunks_reg)
        self.accelerator_memory_to_sram(accelerator_dram_address=CHUNK_BYTES, sram_address=SRAM_A,
                                        element_size=0, inst_pointer_idx=ptr_a)
        if ptr_b is not None:
            self.accelerator_memory_to_sram(accelerator_dram_address=CHUNK_BYTES, sram_address=SRAM_B,
                                            element_size=0, inst_pointer_idx=ptr_b)
        _emit_compute(CHUNK_ROWS, CHUNK_ELEMS, pbi_tail=False)
        self.sram_to_accelerator_memory(sram_address=SRAM_A, accelerator_dram_address=CHUNK_BYTES,
                                        element_size=0, inst_pointer_idx=ptr_out)
        loop_size = self.loop_end()
        assert loop_size <= 256, f"{fn}: chunk loop body {loop_size} exceeds i-cache budget 256"
        patch_loop()

        # ---- runtime tail (skipped when total is a multiple of CHUNK) ----
        patch_tail = self._emit_forward_skip_jz(tail_rows_reg, program_dram_start_addr)
        _set(ptr_a, PBI_FIELD.DMA_LENGTH, tail_bytes_reg)
        self.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=SRAM_A,
                                        element_size=0, inst_pointer_idx=ptr_a)
        if ptr_b is not None:
            _set(ptr_b, PBI_FIELD.DMA_LENGTH, tail_bytes_reg)
            self.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=SRAM_B,
                                            element_size=0, inst_pointer_idx=ptr_b)
        _set(ptr_comp, PBI_FIELD.URAM_ROW_SIZE, tail_rows_reg)
        _emit_compute(max(template_tail // UE_VECTOR_SIZE, 1), max(template_tail, UE_VECTOR_SIZE), pbi_tail=True)
        _set(ptr_out, PBI_FIELD.DMA_LENGTH, tail_bytes_reg)
        self.sram_to_accelerator_memory(sram_address=SRAM_A, accelerator_dram_address=0,
                                        element_size=0, inst_pointer_idx=ptr_out)
        patch_tail()

        self.release_inst_ptr(ptr_comp)
        self.release_inst_ptr(ptr_out)
        if ptr_b is not None:
            self.release_inst_ptr(ptr_b)
        self.release_inst_ptr(ptr_a)
        for _ in _alloc_list:
            self.release_isa_reg()
        return M * N

    # Broadcast operations ------------------------------------------------------
    def start_queue_broadcast(self, mode: UE_MODE, broadcast_mode: BROADCAST_MODE,
                              uram_src_start_addr: int, uram_wb_start_addr: int,
                              element_size: int, scalar: float = None,
                              inst_pointer_idx: Optional[int] = None) -> Optional[torch.Tensor]:
        """
        Start queue for broadcast op: MUL_BROADCAST or ADD_BROADCAST.
        scalar: encoded scalar (e.g. bf16 from float_to_bf16 for MUL/ADD broadcast).
        Wraps ue_arithmetic_op with BROADCAST_MODE.SCALAR_IN_REG.

        ``inst_pointer_idx`` emits a normal UE_PBI operation. The pointer row supplies the complete
        compute state, including field 11 interpreted as ``lalu_scalar``; descriptor values are PBI
        increments, so callers normally pass zero-valued fields for a set-once compute pointer.
        """
        if mode not in (UE_MODE.MUL_BROADCAST, UE_MODE.ADD_BROADCAST):
            raise ValueError(f"Invalid broadcast mode: {mode}. Use MUL_BROADCAST or ADD_BROADCAST.")

        if broadcast_mode == BROADCAST_MODE.SCALAR_IN_REG:
            assert scalar is not None, "scalar must be provided for SCALAR_IN_REG broadcast mode"

        uram_src_type, uram_src_start_addr = self.sram_address_to_uram_address(uram_src_start_addr)
        assert uram_src_type ==  URAM_SECTION.URAM_A, "uram_src_start_addr must be in URAM_A"

        uram_wb_type, uram_wb_addr = self.sram_address_to_uram_address(uram_wb_start_addr)

        row_size = (element_size + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE # round up to the nearest multiple of UE_VECTOR_SIZE

        self.ue_arithmetic_op(
            broadcast_mode.value,  # broadcast_mode
            0,  # max_clear_en
            1,  # stride_z
            0,  # lalu_a
            0,  # lalu_b
            LALU_MODE.BYPASS.value,  # lalu_mode
            self.float_to_bf16(scalar) if scalar is not None else self.float_to_bf19(1.0),
            uram_wb_type.value,
            0,  # uram_dst_addr
            uram_wb_addr,
            URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
            mode,
            0,  # data_type
            uram_src_start_addr,
            0,  # uram_b_start_addr (not used for broadcast)
            row_size,
            0,  # dma_start_addr
            0,  # dma_length
            0,  # output_size
            inst_pointer_idx=inst_pointer_idx,
        )

    def broadcast_mul(self, scalar: float,
                            sram_start_addr: int, sram_wb_addr: int,
                            element_size: int, inst_pointer_idx: Optional[int] = None) -> Optional[torch.Tensor]:
        """Start queue for broadcast multiply. Wraps ue_arithmetic_op with UE_MODE.MUL_BROADCAST.
        ``inst_pointer_idx`` emits UE_PBI; pointer field 11 is interpreted as ``lalu_scalar``."""
        self.start_queue_broadcast(
            UE_MODE.MUL_BROADCAST, BROADCAST_MODE.SCALAR_IN_REG,
            sram_start_addr, sram_wb_addr, element_size, scalar, inst_pointer_idx=inst_pointer_idx,
        )

    def broadcast_add(self, scalar: float,
                            sram_start_addr: int, sram_wb_addr: int,
                            element_size: int, inst_pointer_idx: Optional[int] = None) -> Optional[torch.Tensor]:
        """Start queue for broadcast add. Wraps ue_arithmetic_op with UE_MODE.ADD_BROADCAST.
        ``inst_pointer_idx``: emit UE_PBI via that pointer (runtime URAM dims; scalar stays the
        baked descriptor immediate)."""
        self.start_queue_broadcast(
            UE_MODE.ADD_BROADCAST, BROADCAST_MODE.SCALAR_IN_REG,
            sram_start_addr, sram_wb_addr, element_size, scalar, inst_pointer_idx=inst_pointer_idx
        )

    def start_queue_for_dot_product_operation(self, max_clear_en: int, fmax_context_addr: int, vector_sram_start_addr: int, output_sram_wb_addr: int,
                                            K: int, N: int, dma_start_addr: int,
                                            data_type: TYPE = TYPE.IF4, bias_enable: bool = False, lalu_mode: LALU_MODE = LALU_MODE.BYPASS,
                                            lalu_a: int = 0, lalu_b: int = 0) -> None:
        """Start queue for dot product operation. Matrix data streams from DRAM via DMA.
        Args:
            max_clear_en: clear max accumulator (1 on first chunk, 0 on subsequent)
            vector_sram_start_addr: SRAM address of the vector in URAM_A
            output_sram_wb_addr: SRAM address for output writeback
            K: inner dimension (vector length), must be a multiple of UE_VECTOR_SIZE
            N: outer dimension (matrix height), must be a multiple of UE_VECTOR_SIZE
            dma_start_addr: DRAM address where matrix data starts
            data_type: quantization data type (e.g. TYPE.IF4.value)
            lalu_mode: LALU mode value (e.g. LALU_MODE.ACT)
            lalu_scalar: scalar for LALU operation
        """
        vector_uram_type, vector_uram_start_addr = self.sram_address_to_uram_address(vector_sram_start_addr)
        assert vector_uram_type == URAM_SECTION.URAM_A, f"vector_sram_start_addr must be in URAM_A hex(vector_uram_start_addr)={hex(vector_uram_start_addr)}"
        output_uram_type, output_uram_start_addr = self.sram_address_to_uram_address(output_sram_wb_addr)

        assert K % UE_VECTOR_SIZE == 0, f"K={K} must be a multiple of UE_VECTOR_SIZE={UE_VECTOR_SIZE}"

        if data_type == TYPE.IF4 or data_type == TYPE.TQ4:
            dma_length = (N * K) // 2
        elif data_type == TYPE.IF8:
            dma_length = N * K
        else:
            assert False, f"data_type={data_type} is not supported"

        self.ue_arithmetic_op(
            0,  # broadcast_mode
            max_clear_en,  # clear_max_en
            1,  # stride_z
            lalu_a,  # lalu_a
            lalu_b,  # lalu_b
            lalu_mode.value,  # lalu_mode
            0,  # lalu_scalar
            output_uram_type.value,  # uram_section
            0,  # uram_dst_addr
            output_uram_start_addr,  # uram_wb_addr
            URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
            UE_MODE.DOT_PRODUCT,  # mode
            data_type,  # data_type
            vector_uram_start_addr,  # uram_a_start_addr
            0,  # uram_b_start_addr (not used, matrix streams from DMA)
            K // UE_VECTOR_SIZE,  # bram_length
            dma_start_addr,  # dma_start_addr
            dma_length,  # dma_length
            N,  # output_size
            bias_adder_en=bias_enable,
            fmax_context_addr=fmax_context_addr
        )


    # Compute engine operations --------------------------------------------------
    def start_queue_for_bf16_matvec_operation(
        self,
        max_clear_en: int,
        fmax_context_addr: int,
        vector_sram_start_addr: int,
        matrix_sram_start_addr: int,
        output_sram_wb_addr: int,
        K: int,
        N: int,
        bias_enable: bool = False,
        lalu_mode: LALU_MODE = LALU_MODE.BYPASS,
        stride_z: int = UE_VECTOR_SIZE,
        lalu_a: int = 0,
        lalu_b: int = 0,
        inst_pointer_idx: Optional[int] = None,
    ) -> None:
        """Start queue for bf16 matvec operation. Wraps ue_arithmetic_op with UE_MODE.BF16_DOT_PRODUCT.
        Args:
            max_clear_en: max_clear_en
            fmax_context_addr: fmax_context_addr
            vector_sram_start_addr: vector_sram_start_addr
            matrix_sram_start_addr: matrix_sram_start_addr
            output_sram_wb_addr: output_sram_wb_addr
            K: K
            N: N
            inst_pointer_idx: when nonzero, emit PBI-style UE op (pointer-backed registers are incremented by the immediate value specified in the instruction).
        """
        vector_uram_type, vector_uram_start_addr = self.sram_address_to_uram_address(vector_sram_start_addr)
        matrix_uram_type, matrix_uram_start_addr = self.sram_address_to_uram_address(matrix_sram_start_addr)
        assert vector_uram_type ==  URAM_SECTION.URAM_A, f"vector_sram_start_addr must be in URAM_A hex(vector_uram_start_addr)={hex(vector_uram_start_addr)}"
        assert matrix_uram_type ==  URAM_SECTION.URAM_B, f"matrix_sram_start_addr must be in URAM_B hex(matrix_uram_start_addr)={hex(matrix_uram_start_addr)}"
        output_uram_type, output_uram_start_addr = self.sram_address_to_uram_address(output_sram_wb_addr)

        assert K % UE_VECTOR_SIZE == 0, f"K={K} must be a multiple of UE_VECTOR_SIZE={UE_VECTOR_SIZE}"

        # TODO: support non-aligned writes
        #assert N % UE_VECTOR_SIZE == 0, f"N={N} must be a multiple of UE_VECTOR_SIZE={UE_VECTOR_SIZE}"

        stride_in_rows = stride_z // UE_VECTOR_SIZE
        self.ue_arithmetic_op(
            0,  # broadcast_mode (not used for bf16 matvec operation)
            max_clear_en,  # max_clear_en
            stride_in_rows,  # stride_z
            lalu_a,  # lalu_a
            lalu_b,  # lalu_b
            lalu_mode.value,  # lalu_mode
            0,  # scalar
            output_uram_type.value,
            0,  # uram_dst_addr
            output_uram_start_addr,
            URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
            UE_MODE.BF16_DOT_PRODUCT,
            0,  # data_type
            vector_uram_start_addr,
            matrix_uram_start_addr,  # uram_b_start_addr (not used for compute)
            K // UE_VECTOR_SIZE,
            0,  # dma_start_addr
            K * N if stride_in_rows == 1 else N * stride_z,  # dma_length
            N,  # output_size
            bias_adder_en=bias_enable,
            fmax_context_addr=fmax_context_addr,
            inst_pointer_idx=inst_pointer_idx,
        )

    # Compute engine operations --------------------------------------------------
    def start_queue_for_bf16_softmax_operation(self, fmax_context_addr: int, vector_sram_start_addr: int, output_sram_wb_addr: int, N: int) -> None:
        """Start queue for compute engine. Wraps ue_arithmetic_op with UE_MODE.
        Args:
            fmax_context_addr: fmax_context_addr
            vector_sram_start_addr: vector_sram_start_addr
            output_sram_wb_addr: output_sram_wb_addr
            N: N
        """
        vector_uram_type, vector_uram_start_addr = self.sram_address_to_uram_address(vector_sram_start_addr)
        assert vector_uram_type ==  URAM_SECTION.URAM_A, f"vector_sram_start_addr must be in URAM_A hex(vector_uram_start_addr)={hex(vector_sram_start_addr)}"

        if N % UE_VECTOR_SIZE != 0:
            print(f"Warning: N must be a multiple of UE_VECTOR_SIZE, got {N}. Rounding up to the nearest multiple of UE_VECTOR_SIZE")

        row_size = (N + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE # round up to the nearest multiple of UE_VECTOR_SIZE

        #$ this writes back exp(x - max(x)) to URAM_A at the same time
        self.ue_arithmetic_op(
            BROADCAST_MODE.FMAX_NEGATE.value,  # broadcast_mode
            0,  # max_clear_en
            1,  # stride_z
            0,  # lalu_a
            0,  # lalu_b
            LALU_MODE.MODE_RECIP.value,  # lalu_mode: 1/sum
            self.float_to_bf19(1.0),  # scalar: 1.0 in bf19 format
            vector_uram_type.value,
            0,  # uram_dst_addr
            vector_uram_start_addr,
            URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
            UE_MODE.EXP,
            0,  # data_type
            vector_uram_start_addr,
            0,  # uram_b_start_addr (not used for this compute)
            row_size,
            0,  # dma_start_addr
            0,  # dma_length
            0,  # output_size
            fmax_context_addr=fmax_context_addr
        )

        self.start_queue_broadcast(UE_MODE.MUL_BROADCAST, BROADCAST_MODE.LALU_RESULT, # use 1/sum result from LALU
                            vector_sram_start_addr, output_sram_wb_addr, N)

    def start_queue_for_bf16_rms_mean(self, vector_sram_start_addr: int, N: int) -> None:
        """
        Start queue for RMS mean operation. Wraps ue_arithmetic_op with UE_MODE.
        Args:
            vector_uram_start_addr: vector_uram_start_addr
            N: number of elements in the vector
        """
        row_size = (N + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE

        vector_uram_type, vector_uram_start_addr = self.sram_address_to_uram_address(vector_sram_start_addr)
        assert vector_uram_type == URAM_SECTION.URAM_A, f"vector_uram_start_addr must be in URAM_A, got {hex(vector_uram_start_addr)}"

        self.ue_arithmetic_op(
            0,  # broadcast_mode
            0,  # max_clear_en
            1,  # stride_z
            0,  # lalu_a
            0,  # lalu_b
            LALU_MODE.MODE_RSQRT.value,  # lalu_mode gets sqrt(N) / sqrt(sum(x_i^2))
            self.float_to_bf19(float(math.sqrt(N))),  # BF19 scalar (sqrt(N))
            vector_uram_type.value,  # uram_section
            0,  # uram_dst_addr
            0,  # uram_wb_addr (no writeback, result in LALU)
            URAM_WRITE_SRC.URAM_WB_DISABLE.value,  # uram_write_src
            UE_MODE.RMS,  # mode
            0,  # data_type
            vector_uram_start_addr,  # uram_a_start_addr (x_i)
            0,  # uram_b_start_addr (not used)
            row_size,  # uram_length
            0,  # dma_start_addr
            0,  # dma_length
            0,  # output_size
        )

    def start_queue_for_bf16_layer_norm_mean(self, vector_sram_start_addr: int, zeros_sram_start_addr: int, N: int) -> None:
        """
        Start queue for layer norm mean operation. Wraps ue_arithmetic_op with UE_MODE.
        Args:
            vector_sram_start_addr: vector_sram_start_addr
            zeros_sram_start_addr: zeros_sram_start_addr
            N: number of elements in the vector
        """
        row_size = (N + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE

        vector_uram_type, vector_uram_start_addr = self.sram_address_to_uram_address(vector_sram_start_addr)
        zeros_uram_type, zeros_uram_start_addr = self.sram_address_to_uram_address(zeros_sram_start_addr)
        assert vector_uram_type == URAM_SECTION.URAM_A, f"vector_uram_start_addr must be in URAM_A, got {hex(vector_uram_start_addr)}"
        assert zeros_uram_type == URAM_SECTION.URAM_B, f"zeros_uram_start_addr must be in URAM_B, got {hex(zeros_uram_start_addr)}"

        self.ue_arithmetic_op(
            0,  # broadcast_mode
            0,  # clear_max_en
            1,  # stride_z
            0,  # lalu_a
            0,  # lalu_b
            LALU_MODE.MODE_RECIP.value,  # lalu_mode (computes 1/sum)
            self.float_to_bf19(float(N)),  # BF19 scalar (n)
            vector_uram_type.value,  # uram_section
            0,  # uram_dst_addr
            0,  # uram_wb_addr (no writeback, result in LALU)
            URAM_WRITE_SRC.URAM_WB_DISABLE.value,  # NO WRITEBACK
            UE_MODE.ADD_REDUCE,  # mode
            0,  # data_type not used
            vector_uram_start_addr,  # uram_a_start_addr
            zeros_uram_start_addr,  # uram_b_start_addr, zeros vector
            row_size,  # uram_length (row_size)
            0,  # dma_start_addr
            0,  # dma_length
            0,  # output_size
        )

    # SRAM only version of rms norm core
    def rms_norm_core(self, vector_sram_start_addr: int, output_sram_wb_addr: int, N: int, gamma_sram_start_addr: int = None) -> None:
        """ Core RMS norm: normalizes vector x -> x / rms(x).
        Args:
            vector_sram_start_addr: SRAM address of input vector (must be URAM_A: 0x00000-0x7FFFF)
            output_sram_wb_addr: SRAM address for normalized output
            gamma_sram_start_addr: SRAM address for gamma
            beta_sram_start_addr: SRAM address for beta
            N: number of elements
        """

        self.start_queue_for_bf16_rms_mean(vector_sram_start_addr, N)
        self.start_queue_broadcast(UE_MODE.MUL_BROADCAST, BROADCAST_MODE.LALU_RESULT, vector_sram_start_addr, output_sram_wb_addr, N) # use 1/rms(x) result from LALU
        if gamma_sram_start_addr is not None:
            self.eltwise_mul_core(output_sram_wb_addr, gamma_sram_start_addr, output_sram_wb_addr, N)


    def rms_norm_core_dram_pbi(self, M: int, N: int, A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, GAMMA_DRAM_ADDR: int, gpr_M_reg: int,
                               gpr_a_addr: Optional[int] = None, gpr_out_addr: Optional[int] = None, gpr_gamma_addr: Optional[int] = None) -> int:
        # DEPRECATED REFERENCE ONLY: deliberately unreachable, even under ``python -O``.
        raise RuntimeError("rms_norm_core_dram_pbi is deprecated; use rms_norm_core_dram_dynamic")
        """
        Core RMS norm: normalizes vector x -> x / rms(x).

        **No M-direction tiling:** Outer ``loop_start`` uses **two PBI pointer rows** only—input row
        DMA and output row writeback—with consecutive DRAM stride ``N * 2``. RMS, broadcast scale,
        and gamma multiply reuse the same fixed-SRAM sequence as :meth:`rms_norm_core` (no UE PBI
        pointers).

        **Batch dimension / M (dynamic, required):**
        ``gpr_M_reg`` is a **required** GPR index 1..15 holding the runtime row count. Caller must
        prime that register beforehand (typically with ``ADD_SET``). The hardware loop runs for
        whatever value that register holds at execute time; ``M`` is still required as a
        compile-time argument purely for FLOPs accounting / asserts — the captured program contains
        no static reference to ``M``.

        **DRAM base addresses (optional dynamic):** ``gpr_a_addr`` (input rows), ``gpr_out_addr``
        (output rows) and ``gpr_gamma_addr`` (gamma vector) are optional GPR indices (1..63). When
        given, that base is sourced from the GPR (word address = ``byte >> 3``) instead of the
        literal, so one captured program serves any placement. Input/output are looped PBI pointers
        (PBI_MODE_REG override); the legacy dynamic gamma preload is no longer supported. ``None`` → literal,
        no extra instruction. See :meth:`eltwise_core_dram_pbi`.

        Caller must have start_capture() active.
        """
        assert M >= 1, "rms_norm_core_dram_pbi() requires M >= 1"
        if not (1 <= gpr_M_reg <= 15):
            raise ValueError(f"rms_norm_core_dram_pbi: gpr_M_reg must be a GPR index 1..15, got {gpr_M_reg}")
        self._validate_addr_gprs("rms_norm_core_dram_pbi", gpr_M_reg, {
            "gpr_a_addr": gpr_a_addr, "gpr_out_addr": gpr_out_addr, "gpr_gamma_addr": gpr_gamma_addr,
        })

        vector_sram_addr = 0x00000
        gamma_sram_addr = 0x80000

        self.accelerator_memory_to_sram(accelerator_dram_address=GAMMA_DRAM_ADDR,
                                        sram_address=gamma_sram_addr,
                                        element_size=N,
                                        general_reg_src=gpr_gamma_addr)

        assert self.is_capture_on, "rms_norm_core_dram_pbi() requires active capture"
        assert N % UE_VECTOR_SIZE == 0, f"rms_norm_core_dram_pbi() requires N to be a multiple of UE_VECTOR_SIZE, got N={N}"
        if N > URAM_NEAR_FULL_ELEMENTS:
            raise ValueError(
                f"rms_norm_core_dram_pbi: N={N} exceeds URAM_NEAR_FULL_ELEMENTS={URAM_NEAR_FULL_ELEMENTS} "
                "(one row must fit staging)."
            )

        bytes_per_element = 2
        row_bytes = N * bytes_per_element

        vector_uram_type, vector_uram_start_addr = self.sram_address_to_uram_address(vector_sram_addr)
        gamma_uram_type, _ = self.sram_address_to_uram_address(gamma_sram_addr)
        assert vector_uram_type == URAM_SECTION.URAM_A, f"vector_sram_addr must be in URAM_A, got {hex(vector_sram_addr)}"
        assert gamma_uram_type == URAM_SECTION.URAM_B, f"gamma_sram_addr must be in URAM_B, got {hex(gamma_sram_addr)}"

        row_load_ptr = self.alloc_inst_ptr()
        row_store_ptr = self.alloc_inst_ptr()

        self.generate_instruction_pbi_init(
            dram_shared_addr=A_DRAM_ADDR,
            dma_length=row_bytes,
            output_size=0,
            uram_length=0,
            uram_a_start_addr=0,
            uram_b_start_addr=0,
            uram_wb_addr=0,
            uram_dst_addr=vector_uram_start_addr,
            fmax_context_addr=0,
            inst_pointer_idx=row_load_ptr,
        )
        self.generate_instruction_pbi_init(
            dram_shared_addr=OUTPUT_DRAM_ADDR,
            dma_length=row_bytes,
            output_size=0,
            uram_length=0,
            uram_a_start_addr=vector_uram_start_addr,
            uram_b_start_addr=vector_uram_start_addr,
            uram_wb_addr=0,
            uram_dst_addr=0,
            fmax_context_addr=0,
            inst_pointer_idx=row_store_ptr,
        )

        # Optional: source the input/output pointer bases from GPRs (word addr) once before the loop.
        if gpr_a_addr is not None:
            self._pbi_override_dram_base_from_gpr(row_load_ptr, gpr_a_addr)
        if gpr_out_addr is not None:
            self._pbi_override_dram_base_from_gpr(row_store_ptr, gpr_out_addr)

        program_dram_start_addr = self.get_program_dram_addr()
        cur_inst_count = self.capture_count
        jump_target_word_addr = ue_35bit_addr_shifter(
            program_dram_start_addr + (cur_inst_count + 1) * INSTRUCTION_SIZE_BYTES
        )
        self.generate_instruction_jump_abs(jump_target_word_addr)
        self.loop_start(loop_cnt=M, gpr_loop_cnt=gpr_M_reg)

        self.accelerator_memory_to_sram(
            accelerator_dram_address=row_bytes,
            sram_address=vector_sram_addr,
            element_size=0,
            inst_pointer_idx=row_load_ptr,
        )

        self.start_queue_for_bf16_rms_mean(vector_sram_addr, N)
        self.start_queue_broadcast(
            UE_MODE.MUL_BROADCAST,
            BROADCAST_MODE.LALU_RESULT,
            vector_sram_addr,
            vector_sram_addr,
            N,
        )
        self.eltwise_mul_core(vector_sram_addr, gamma_sram_addr, vector_sram_addr, N)

        self.sram_to_accelerator_memory(
            sram_address=vector_sram_addr,
            accelerator_dram_address=row_bytes,
            element_size=0,
            inst_pointer_idx=row_store_ptr,
        )

        outer_loop_size = self.loop_end()
        m_src = f"GPR[{gpr_M_reg}]" if gpr_M_reg is not None else str(M)
        print(f"RMS norm PBI outer loop body size: {outer_loop_size} (one row × trips={m_src}, N={N})")
        assert outer_loop_size <= 256, (
            f"Outer loop body size {outer_loop_size} is greater than i-cache size of 256 instructions"
        )

        self.release_inst_ptr(row_store_ptr)
        self.release_inst_ptr(row_load_ptr)

        total_flops = 3 * M * N + M * N  # RMS + gamma mul
        return total_flops

    def rms_norm_core_dram_dynamic(self, M: int, N: int, A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, GAMMA_DRAM_ADDR: int,
                                   gpr_M_reg: int, gpr_N_reg: int, gpr_sqrt_n_reg: int, gpr_rms_scale_reg: Optional[int] = None,
                                   gpr_a_addr: Optional[int] = None, gpr_out_addr: Optional[int] = None,
                                   gpr_gamma_addr: Optional[int] = None) -> int:
        """RMS norm — runtime M (batch) and runtime ``sqrt(N)`` scalar, N baked at capture (chunk-tiled).

        Batches up to ``U = min(URAM_A capacity, 16)`` rows per DMA transfer and runs the per-row
        compute (RMS-mean -> broadcast scale -> gamma multiply) as PLAIN (non-PBI) software-unrolled
        ops at compile-time SRAM offsets; only the two DMA pointers (``row_load_ptr`` /
        ``row_store_ptr``) are PBI. The outer loop is a ``reg_min``-clamped while(remaining) over
        chunks of ``U`` rows; the last partial chunk computes all ``U`` rows but only stores
        ``rows_take``, so tail waste is at most ``U-1`` rows once per call. Batching the DMA and
        keeping compute plain (no per-row PBI overrides) are both required to match legacy throughput.

        **N is baked at capture, not runtime:** ``gpr_N_reg`` is accepted and validated for signature
        compatibility, but the runtime hidden_size at execute time must equal the capture-time ``N``.
        Callers needing a different N must re-capture (norms are captured once per model config, so
        this is free in practice). Non-64-aligned hidden_size is served by host-side zero padding (see
        below); the padded length is what gets baked.

        **``sqrt(N)`` is a runtime scalar sourced from a GPR (fully dynamic N):**
        RMS = ``x * sqrt(N) / sqrt(sum(x^2)) * gamma``. The RSQRT scalar is read at execute time from
        ``gpr_sqrt_n_reg`` (caller-primed with ``float_to_bf19(sqrt(real_N))`` — the ISA has no
        on-device sqrt) through a complete RMS PBI row whose shared field 11 is initialized from
        ``gpr_sqrt_n_reg`` using ``PBI_FIELD.LALU_SCALAR``. Per-row RSQRT operations are normal
        UE_PBI operations whose descriptor deltas advance and wrap the pointer's Y address. sqrt(N)
        rides the RSQRT scalar
        (like legacy :meth:`start_queue_for_bf16_rms_mean`), so the caller uploads a **plain** gamma —
        no ``gamma*sqrt(N)`` re-rounding. ``gpr_rms_scale_reg`` is ignored (signature compat).

        ``gpr_M_reg`` (1..15, loop count), ``gpr_N_reg`` (1..63, signature compat), and
        ``gpr_sqrt_n_reg`` (1..63, = ``float_to_bf19(sqrt(real_N))``) are required and caller-primed.
        ``M`` / ``N`` are template / FLOPs-accounting only. ``gpr_a_addr`` / ``gpr_out_addr`` /
        ``gpr_gamma_addr`` optionally source the input / output / gamma DRAM bases from GPRs
        (word addr = ``byte >> 3``). Requires ``N`` a multiple of ``UE_VECTOR_SIZE`` and one row to
        fit URAM staging.

        **Non-64-aligned hidden_size** is served by host-side zero padding: the reduce is
        ``sum(x^2)``, which doesn't depend on the reduce length, so the caller pads the row to a
        multiple of 64 with zeros, captures with the padded length as ``N``, primes ``gpr_sqrt_n_reg``
        with the *unpadded* ``sqrt(real_N)`` scalar, and slices the real columns out of the output.
        (LayerNorm can't fold the mean this way — its mean-centering makes pad lanes contaminate the
        variance — so it keeps the ``inv_n`` mask-vector path.) See ``rms_norm_core_dram_dynamic_test``
        in user_hw_test.py.
        """
        fn = "rms_norm_core_dram_dynamic"
        assert M >= 1 and N >= 1, f"{fn}: require M>=1 and N>=1"
        if N % UE_VECTOR_SIZE != 0:
            raise ValueError(f"{fn}: N must be a multiple of UE_VECTOR_SIZE={UE_VECTOR_SIZE}, got N={N}")
        if N > URAM_NEAR_FULL_ELEMENTS:
            raise ValueError(f"{fn}: N={N} exceeds URAM_NEAR_FULL_ELEMENTS={URAM_NEAR_FULL_ELEMENTS} (one row must fit staging)")
        # gpr_rms_scale_reg is deprecated/ignored; gpr_M_reg / gpr_N_reg / gpr_sqrt_n_reg are the
        # required runtime registers (sqrt(N) is now a runtime scalar read from gpr_sqrt_n_reg).
        for _nm, _rg, _hi in (("gpr_M_reg", gpr_M_reg, 15), ("gpr_N_reg", gpr_N_reg, 63),
                              ("gpr_sqrt_n_reg", gpr_sqrt_n_reg, 63)):
            if _rg is None or not (1 <= _rg <= _hi):
                raise ValueError(f"{fn}: {_nm} must be a GPR index 1..{_hi}, got {_rg}")
        if len({gpr_M_reg, gpr_N_reg, gpr_sqrt_n_reg}) != 3:
            raise ValueError(f"{fn}: gpr_M_reg/gpr_N_reg/gpr_sqrt_n_reg must be distinct")
        self._validate_addr_gprs(fn, gpr_M_reg, {
            "gpr_a_addr": gpr_a_addr, "gpr_out_addr": gpr_out_addr, "gpr_gamma_addr": gpr_gamma_addr,
        })
        for _nm, _rg in (("gpr_a_addr", gpr_a_addr), ("gpr_out_addr", gpr_out_addr), ("gpr_gamma_addr", gpr_gamma_addr)):
            if _rg is not None and _rg in (gpr_N_reg, gpr_sqrt_n_reg):
                raise ValueError(f"{fn}: {_nm}={_rg} collides with gpr_N_reg/gpr_sqrt_n_reg")
        assert self.is_capture_on, f"{fn}() requires active capture"

        vector_sram_addr = 0x00000   # URAM_A
        gamma_sram_addr = 0x80000    # URAM_B
        vector_uram_type, vector_uram_start_addr = self.sram_address_to_uram_address(vector_sram_addr)
        gamma_uram_type, _gamma_uram_row = self.sram_address_to_uram_address(gamma_sram_addr)
        assert vector_uram_type == URAM_SECTION.URAM_A, f"vector_sram_addr must be in URAM_A, got {hex(vector_sram_addr)}"
        assert gamma_uram_type == URAM_SECTION.URAM_B, f"gamma_sram_addr must be in URAM_B, got {hex(gamma_sram_addr)}"

        # N baked at capture (compile-time strides); M is runtime via the reg_min while-loop below.
        bytes_per_element = 2
        row_bytes = N * bytes_per_element            # compile-time
        row_size = N // UE_VECTOR_SIZE               # compile-time URAM rows per token
        # sqrt(N) is a RUNTIME scalar read from gpr_sqrt_n_reg via a set-once scalar pointer (below),
        # not baked here — so the same host value serves any (padded/real) N without re-rounding.

        # Unroll factor U: rows processed per outer iteration, bounded by URAM_A staging capacity,
        # capped at 16 so the unrolled body stays under the i-cache window, AND capped at the template
        # M so a small-M caller doesn't over-compute. The compute loop software-unrolls U rows and the
        # last chunk always computes U rows but stores only rows_take, so if U > (real M) the surplus
        # rows are wasted compute. Decode calls this with M=1 (single token); without the M cap U=16
        # would compute 16 rows per RMS for 1 real row (~16x waste, 6 RMS/layer x 26 layers = the
        # decode perf regression). M is the template row count; the runtime count still rides gpr_M_reg,
        # so this only sets the unroll granularity and never changes correctness for a larger runtime M
        # (the reg_min chunk loop just runs more chunks).
        U = max(1, min(URAM_NEAR_FULL_ADDR // row_size, 16, M))

        self.accelerator_memory_to_sram(accelerator_dram_address=GAMMA_DRAM_ADDR,
                                        sram_address=gamma_sram_addr, element_size=N, general_reg_src=gpr_gamma_addr)

        _alloc_list = []
        def _alloc():
            r = self.alloc_isa_reg(); _alloc_list.append(r); return r
        remaining_reg = _alloc()   # rows not yet processed
        rows_take_reg = _alloc()   # min(remaining, U) -- this chunk's row count
        dma_len_reg   = _alloc()   # rows_take*row_bytes (chunk DMA length, load and store)
        u_reg         = _alloc()   # U literal (reg_min clamp)
        # dma_len = rows_take * row_bytes: a 1-cycle SHL when row_bytes is a power of two (the common
        # case), else a mul32 (rare non-pow2 host-padded N).
        row_bytes_is_pow2 = (row_bytes & (row_bytes - 1)) == 0
        row_bytes_shift = (row_bytes.bit_length() - 1) if row_bytes_is_pow2 else 0
        rowb_reg = None if row_bytes_is_pow2 else _alloc()
        if rowb_reg is not None:
            self.generate_instruction_add_set(rowb_reg, row_bytes)
        self.generate_instruction_add_set(u_reg, U)

        row_load_ptr = self.alloc_inst_ptr()
        row_store_ptr = self.alloc_inst_ptr()
        rms_ptr = self.alloc_inst_ptr()   # complete UE_PBI state for the runtime-scalar RMS op
        # Both DMA pointers auto-increment their DRAM cursor by the full-chunk stride U*row_bytes each
        # transfer, so no per-chunk DRAM_ADDR set is needed (the last partial chunk over-advances by
        # the unused tail, harmlessly, since the loop exits right after). DMA_LENGTH is set per chunk.
        chunk_stride = U * row_bytes
        self.generate_instruction_pbi_init(dram_shared_addr=A_DRAM_ADDR, dma_length=chunk_stride,
                                           uram_dst_addr=vector_uram_start_addr, inst_pointer_idx=row_load_ptr)
        self.generate_instruction_pbi_init(dram_shared_addr=OUTPUT_DRAM_ADDR, dma_length=chunk_stride,
                                           uram_a_start_addr=vector_uram_start_addr,
                                           uram_b_start_addr=vector_uram_start_addr, inst_pointer_idx=row_store_ptr)
        self.generate_instruction_pbi_init(
            uram_length=row_size,
            uram_a_start_addr=vector_uram_start_addr,
            inst_pointer_idx=rms_ptr,
        )
        # Optional dynamic DRAM bases: override the pointer bases once (auto-inc then walks from there).
        if gpr_a_addr is not None:
            self._pbi_override_dram_base_from_gpr(row_load_ptr, gpr_a_addr)
        if gpr_out_addr is not None:
            self._pbi_override_dram_base_from_gpr(row_store_ptr, gpr_out_addr)

        def _set(ptr, field, reg):
            self.generate_instruction_pbi_inc(general_reg_src=reg, pbi_field_select=field, inst_pointer_idx=ptr)

        # Arithmetic UE_PBI interprets shared field 11 as lalu_scalar.
        _set(rms_ptr, PBI_FIELD.LALU_SCALAR, gpr_sqrt_n_reg)

        def _dma_len():
            if row_bytes_is_pow2:
                self.generate_instruction_shl(dma_len_reg, rows_take_reg, row_bytes_shift)
            else:
                self.generate_instruction_mul32_reg(dma_len_reg, rows_take_reg, rowb_reg)

        def _rms_row(sram, row_idx):
            # RMS-mean is UE_PBI so its complete pointer row supplies the runtime scalar and URAM
            # state. The descriptor's Y field is the post-op pointer increment; the final row wraps
            # the pointer to row zero for the next hardware-loop iteration.
            # 3 ops per row: RMS-mean (RSQRT scalar=sqrt(N) ->
            # sqrt(N)/sqrt(sum(x^2)) = 1/rms), broadcast scale by that LALU result, gamma multiply.
            # Caller uploads a plain gamma -- no gamma*sqrt(N) re-rounding.
            y_delta = row_size if row_idx < U - 1 else -((U - 1) * row_size)
            self.ue_arithmetic_op(
                0, 0, 0, 0, 0, LALU_MODE.MODE_RSQRT.value, 0,
                URAM_SECTION.URAM_A.value, 0, 0, URAM_WRITE_SRC.URAM_WB_DISABLE.value,
                UE_MODE.RMS, 0, y_delta, 0, 0, 0, 0, 0,
                inst_pointer_idx=rms_ptr)
            self.start_queue_broadcast(UE_MODE.MUL_BROADCAST, BROADCAST_MODE.LALU_RESULT, sram, sram, N)
            self.eltwise_mul_core(sram, gamma_sram_addr, sram, N)

        # ===== outer chunk loop (2 live PBI pointers: row_load_ptr, row_store_ptr) =====
        program_dram_start_addr = self.get_program_dram_addr()
        cur_inst_count = self.capture_count
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(
            program_dram_start_addr + (cur_inst_count + 1) * INSTRUCTION_SIZE_BYTES))

        self.generate_instruction_add_imm(src_reg_idx=gpr_M_reg, immediate_value=0, dst_reg_idx=remaining_reg)
        self.generate_instruction_reg_min(rows_take_reg, remaining_reg, u_reg)
        body_start_inst_cnt = self.capture_count

        # Batched DMA load: rows_take rows in ONE transfer into SRAM rows [0, rows_take). The pointer
        # auto-advances by chunk_stride (=U*row_bytes) for the next chunk.
        _dma_len()
        _set(row_load_ptr, PBI_FIELD.DMA_LENGTH, dma_len_reg)
        self.accelerator_memory_to_sram(accelerator_dram_address=chunk_stride, sram_address=vector_sram_addr, element_size=0, inst_pointer_idx=row_load_ptr)

        # Unrolled plain per-row compute over all U slots. Only rows [0, rows_take) hold real data
        # and only they are stored below; slots [rows_take, U) compute on stale SRAM and are dropped.
        for j in range(U):
            _rms_row(vector_sram_addr + j * row_bytes, j)

        # Batched DMA store: rows_take rows in ONE transfer (dma_len_reg unchanged since the load).
        _set(row_store_ptr, PBI_FIELD.DMA_LENGTH, dma_len_reg)
        self.sram_to_accelerator_memory(sram_address=vector_sram_addr, accelerator_dram_address=chunk_stride, element_size=0, inst_pointer_idx=row_store_ptr)

        # Recompute the next chunk's row count (DRAM cursors auto-advanced via the DMA pointers).
        self.generate_instruction_reg_sub(remaining_reg, remaining_reg, rows_take_reg)
        self.generate_instruction_reg_min(rows_take_reg, remaining_reg, u_reg)

        outer_loop_size = self.capture_count - body_start_inst_cnt + 2
        self.generate_instruction_jump_rela_jnz(outer_loop_size, remaining_reg)

        print(f"RMS norm dynamic (chunk-tiled) outer loop body size: {outer_loop_size} "
              f"(U={U} rows/chunk × ceil(M/U) chunks, M=GPR[{gpr_M_reg}], N={N} compile-time addr, "
              f"sqrt(N)=GPR[{gpr_sqrt_n_reg}] runtime, 3 PBI pointers)")
        assert outer_loop_size <= 256, f"{fn}: outer loop body {outer_loop_size} exceeds i-cache budget 256"

        self.release_inst_ptr(rms_ptr)
        self.release_inst_ptr(row_store_ptr)
        self.release_inst_ptr(row_load_ptr)
        for _ in range(len(_alloc_list)):
            self.release_isa_reg()

        return 3 * M * N + M * N  # RMS + gamma mul

    def rms_norm_core_dram_legacy(self, M: int, N: int, A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, GAMMA_DRAM_ADDR: int) -> None:
        """
        Legacy RMS-norm DRAM path that emits one software-unrolled row op sequence per row.
        """
        vector_sram_addr = 0x00000
        gamma_sram_addr = 0x80000

        self.accelerator_memory_to_sram(accelerator_dram_address=GAMMA_DRAM_ADDR,
                                        sram_address=gamma_sram_addr,
                                        element_size=N)

        chunk_size = min(URAM_NEAR_FULL_ELEMENTS // N, M)
        print(f"Chunk size: {chunk_size} for M={M}, N={N}")
        assert chunk_size >= 1 and chunk_size <= M, f"chunk_size={chunk_size} must be greater than 0 and less than M={M}"

        for i, m_take in self.chunk_ranges(M, chunk_size):
            self.accelerator_memory_to_sram(accelerator_dram_address=A_DRAM_ADDR + i * N * 2,
                                            sram_address=vector_sram_addr,
                                            element_size=m_take * N)
            for j in range(m_take):
                self.rms_norm_core(vector_sram_addr + j * N * 2, vector_sram_addr + j * N * 2, N, gamma_sram_addr)

            self.sram_to_accelerator_memory(sram_address=vector_sram_addr,
                                            accelerator_dram_address=OUTPUT_DRAM_ADDR + i * N * 2,
                                            element_size=m_take * N)

        # Total Theoretical FLOPS: 3 * M * N + M * N (gamma)
        total_flops = 3 * M * N + M * N # exp(x), sum(x), broadcast_mul, eltwise_mul(gamma)
        return total_flops

    # rms_norm dram version of rms norm core
    def rms_norm_core_dram(self, M: int, N: int, A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, GAMMA_DRAM_ADDR: int,
                           gpr_M_reg: Optional[int] = None, gpr_N_reg: Optional[int] = None,
                           gpr_rms_scale_reg: Optional[int] = None,
                           gpr_a_addr: Optional[int] = None, gpr_out_addr: Optional[int] = None,
                           gpr_gamma_addr: Optional[int] = None,
                           gpr_sqrt_n_reg: Optional[int] = None) -> int:
        """RMS norm DRAM entrypoint; dispatches on whether any runtime register is supplied:

        - **Any** of ``gpr_M_reg`` / ``gpr_N_reg`` / ``gpr_sqrt_n_reg`` (or a GPR DRAM base):
          :meth:`rms_norm_core_dram_dynamic` — runtime **M** and a runtime ``sqrt(N)`` RSQRT scalar.
          Any dimension register not supplied is auto-allocated and seeded from the compile-time
          ``M`` / ``N`` (mirrors :meth:`matmat_mul_core`), so a ``gpr_M_reg``-only call — the former
          PBI decode contract — transparently runs the dynamic core with ``N`` / ``sqrt(N)`` baked
          into registers. ``gpr_rms_scale_reg`` is deprecated/ignored. (The PBI tier is retired:
          the dynamic core subsumes runtime-M with baked N.)
        - **No** GPRs (default): :meth:`rms_norm_core_dram_legacy` — compile-time ``chunk_size`` tiling.

        ``gpr_a_addr`` / ``gpr_out_addr`` / ``gpr_gamma_addr`` (dynamic path) optionally source the
        input / output / gamma DRAM bases from GPRs (word addr).
        """
        if any(r is not None for r in (gpr_M_reg, gpr_N_reg, gpr_sqrt_n_reg,
                                       gpr_a_addr, gpr_out_addr, gpr_gamma_addr)):
            # Any runtime register -> the dynamic core (it subsumes the retired PBI path). The missing
            # dimension registers are auto-allocated and seeded from the compile-time M / N, so a
            # caller supplying only gpr_M_reg still runs dynamic. gpr_rms_scale_reg: compat, ignored.
            _seeded = []
            if gpr_N_reg is None:
                gpr_N_reg = self.alloc_isa_reg(); _seeded.append(gpr_N_reg)
                self.generate_instruction_add_set(gpr_N_reg, N)
            if gpr_sqrt_n_reg is None:
                gpr_sqrt_n_reg = self.alloc_isa_reg(); _seeded.append(gpr_sqrt_n_reg)
                self.generate_instruction_add_set(gpr_sqrt_n_reg, self.float_to_bf19(float(N ** 0.5)))
            if gpr_M_reg is None:
                gpr_M_reg = self.alloc_isa_reg(); _seeded.append(gpr_M_reg)
                self.generate_instruction_add_set(gpr_M_reg, M)
            result = self.rms_norm_core_dram_dynamic(
                M=M, N=N, A_DRAM_ADDR=A_DRAM_ADDR, OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR, GAMMA_DRAM_ADDR=GAMMA_DRAM_ADDR,
                gpr_M_reg=gpr_M_reg, gpr_N_reg=gpr_N_reg, gpr_sqrt_n_reg=gpr_sqrt_n_reg,
                gpr_rms_scale_reg=gpr_rms_scale_reg,
                gpr_a_addr=gpr_a_addr, gpr_out_addr=gpr_out_addr, gpr_gamma_addr=gpr_gamma_addr,
            )
            for _ in _seeded:
                self.release_isa_reg()
            return result

        return self.rms_norm_core_dram_legacy(
            M=M,
            N=N,
            A_DRAM_ADDR=A_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
            GAMMA_DRAM_ADDR=GAMMA_DRAM_ADDR,
        )

    # we need a layer norm core that takes in a vector and outputs a vector
    def layer_norm_core(self, vector_sram_start_addr: int, output_sram_wb_addr: int, N: int, zeros_sram_start_addr: int = None, input_gamma_sram_start_addr: int = None, input_beta_sram_start_addr: int = None) -> None:
        """
        Core layer norm: normalizes vector x -> x / rms(x).
        Args:
            M: number of rows in the input matrix
            N: number of columns in the input matrix
            vector_sram_start_addr: vector_sram_start_addr
            output_sram_wb_addr: output_sram_wb_addr
            N: number of elements
            input_gamma_sram_start_addr: input_gamma_sram_start_addr
            input_beta_sram_start_addr: input_beta_sram_start_addr
        """

        self.start_queue_for_bf16_layer_norm_mean(vector_sram_start_addr, zeros_sram_start_addr, N)
        self.start_queue_broadcast(UE_MODE.ADD_BROADCAST, BROADCAST_MODE.LALU_RESULT_NEGATE, vector_sram_start_addr, output_sram_wb_addr, N) # use 1/rms(x) result from LALU
        self.start_queue_for_bf16_rms_mean(output_sram_wb_addr, N)
        self.start_queue_broadcast(UE_MODE.MUL_BROADCAST, BROADCAST_MODE.LALU_RESULT, output_sram_wb_addr, output_sram_wb_addr, N) # use 1/rms(x) result from LALU
        if input_gamma_sram_start_addr is not None:
            self.eltwise_mul_core(output_sram_wb_addr, input_gamma_sram_start_addr, output_sram_wb_addr, N)
        if input_beta_sram_start_addr is not None:
            self.eltwise_add_core(output_sram_wb_addr, input_beta_sram_start_addr, output_sram_wb_addr, N)


    def layer_norm_core_dram_pbi(self, M: int, N: int, A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                                 GAMMA_DRAM_ADDR: int = None, BETA_DRAM_ADDR: int = None,
                                 gpr_M_reg: int = None, ZEROS_DRAM_ADDR: int = None,
                                 gpr_a_addr: Optional[int] = None, gpr_out_addr: Optional[int] = None,
                                 gpr_gamma_addr: Optional[int] = None, gpr_beta_addr: Optional[int] = None) -> int:
        # DEPRECATED REFERENCE ONLY: deliberately unreachable, even under ``python -O``.
        raise RuntimeError("layer_norm_core_dram_pbi is deprecated; use layer_norm_core_dram_dynamic")
        """PBI-backed layer norm.  Matches the legacy's chunk-tiled DMA granularity so performance
        is on par, while the captured program shrinks from ~M*6 instructions to ~chunk_size*6+4.

        **DMA chunk tiling (key to performance):**
        The legacy path loads ``chunk_size = min(URAM_NEAR_FULL_ELEMENTS // N, M)`` rows per DMA
        call to amortise per-transfer overhead.  This PBI path does the same: each loop iteration
        loads ``chunk_size`` rows in a single DMA, Python-unrolls ``chunk_size`` ``layer_norm_core``
        calls at different URAM_A offsets, then stores ``chunk_size`` rows in a single DMA.

        **Batch dimension (gpr_M_reg):**
        ``gpr_M_reg`` is a GPR index 1..15.  The caller must prime it with the number of **chunks**
        ``M // chunk_size``, *not* with ``M`` directly.  ``chunk_size`` is
        ``min(URAM_NEAR_FULL_ELEMENTS // N, M)``; both the caller and this function compute it from
        the same formula so they agree.  ``M`` itself is only used for FLOPs accounting and the
        ``M % chunk_size == 0`` assertion.

        Caller must have start_capture() active.
        """
        assert M >= 1, "layer_norm_core_dram_pbi() requires M >= 1"
        if not (1 <= gpr_M_reg <= 15):
            raise ValueError(f"layer_norm_core_dram_pbi: gpr_M_reg must be a GPR index 1..15, got {gpr_M_reg}")
        self._validate_addr_gprs("layer_norm_core_dram_pbi", gpr_M_reg, {
            "gpr_a_addr": gpr_a_addr, "gpr_out_addr": gpr_out_addr,
            "gpr_gamma_addr": gpr_gamma_addr, "gpr_beta_addr": gpr_beta_addr,
        })
        if gpr_gamma_addr is not None and GAMMA_DRAM_ADDR is None:
            raise ValueError("layer_norm_core_dram_pbi: gpr_gamma_addr given but GAMMA_DRAM_ADDR is None")
        if gpr_beta_addr is not None and BETA_DRAM_ADDR is None:
            raise ValueError("layer_norm_core_dram_pbi: gpr_beta_addr given but BETA_DRAM_ADDR is None")
        assert self.is_capture_on, "layer_norm_core_dram_pbi() requires active capture"
        assert N % UE_VECTOR_SIZE == 0, f"layer_norm_core_dram_pbi() requires N to be a multiple of {UE_VECTOR_SIZE}, got N={N}"
        if N > URAM_NEAR_FULL_ELEMENTS:
            raise ValueError(
                f"layer_norm_core_dram_pbi: N={N} exceeds URAM_NEAR_FULL_ELEMENTS={URAM_NEAR_FULL_ELEMENTS} "
                "(one row must fit in URAM_A staging)."
            )

        bpe = 2
        row_bytes = N * bpe

        # ops_per_row: number of ISA instructions layer_norm_core emits for one row.
        # 4 queue ops (mean, subtract, rms, scale) + 1 per optional param.
        ops_per_row = 4 + (1 if GAMMA_DRAM_ADDR is not None else 0) + (1 if BETA_DRAM_ADDR is not None else 0)
        # I-cache is 256 instructions.  Loop body = 1 (pbi_load) + chunk_size*ops_per_row
        # + 1 (pbi_store) + 2 (loop_end add_dec + jump) = chunk_size*ops_per_row + 4.
        max_chunk_from_icache = (256 - 4) // ops_per_row

        # Largest divisor of M that satisfies both the URAM capacity and the i-cache limit.
        ideal = min(URAM_NEAR_FULL_ELEMENTS // N, M, max_chunk_from_icache)
        chunk_size = ideal
        while M % chunk_size != 0:
            chunk_size -= 1  # guaranteed to terminate; M % 1 == 0 always
        n_chunks    = M // chunk_size
        chunk_bytes = chunk_size * row_bytes

        # URAM_A: chunk_size input/output rows per iteration
        # URAM_B: zeros | gamma (opt) | beta (opt) — static, loaded once before the loop
        vector_sram_addr = 0x00000
        zeros_sram_addr  = 0x80000
        params_sram_addr = zeros_sram_addr + N * bpe

        gamma_sram_addr = None
        beta_sram_addr  = None

        if ZEROS_DRAM_ADDR is not None:
            zeros_dram_addr = ZEROS_DRAM_ADDR          # caller-supplied shared zeros buffer
        else:
            zeros_dram_addr = self.get_params_dram_addr()
            self.allocate_params_dram(N * bpe)
            self.dma_write(DMA_DEVICE_H2C, zeros_dram_addr, torch.zeros(N, dtype=torch.bfloat16), N * bpe)

        self.accelerator_memory_to_sram(accelerator_dram_address=zeros_dram_addr,
                                        sram_address=zeros_sram_addr, element_size=N)

        if GAMMA_DRAM_ADDR is not None:
            gamma_sram_addr = params_sram_addr
            self.accelerator_memory_to_sram(accelerator_dram_address=GAMMA_DRAM_ADDR,
                                            sram_address=gamma_sram_addr, element_size=N,
                                            general_reg_src=gpr_gamma_addr)
            params_sram_addr += N * bpe

        if BETA_DRAM_ADDR is not None:
            beta_sram_addr = params_sram_addr
            self.accelerator_memory_to_sram(accelerator_dram_address=BETA_DRAM_ADDR,
                                            sram_address=beta_sram_addr, element_size=N,
                                            general_reg_src=gpr_beta_addr)
            params_sram_addr += N * bpe

        _, vector_uram_start_addr = self.sram_address_to_uram_address(vector_sram_addr)

        row_load_ptr  = self.alloc_inst_ptr()
        row_store_ptr = self.alloc_inst_ptr()

        # PBI load: one call advances by chunk_bytes, loading chunk_size rows into URAM_A.
        self.generate_instruction_pbi_init(
            dram_shared_addr=A_DRAM_ADDR,
            dma_length=chunk_bytes,
            output_size=0, uram_length=0,
            uram_a_start_addr=0, uram_b_start_addr=0, uram_wb_addr=0,
            uram_dst_addr=vector_uram_start_addr,
            fmax_context_addr=0,
            inst_pointer_idx=row_load_ptr,
        )
        # PBI store: reads chunk_size rows from URAM_A (lines 0..chunk_size*N/64-1).
        self.generate_instruction_pbi_init(
            dram_shared_addr=OUTPUT_DRAM_ADDR,
            dma_length=chunk_bytes,
            output_size=0, uram_length=0,
            uram_a_start_addr=vector_uram_start_addr,
            uram_b_start_addr=vector_uram_start_addr,
            uram_wb_addr=0, uram_dst_addr=0, fmax_context_addr=0,
            inst_pointer_idx=row_store_ptr,
        )

        # Optional: source the input/output pointer bases from GPRs (word addr) once before the loop.
        if gpr_a_addr is not None:
            self._pbi_override_dram_base_from_gpr(row_load_ptr, gpr_a_addr)
        if gpr_out_addr is not None:
            self._pbi_override_dram_base_from_gpr(row_store_ptr, gpr_out_addr)

        program_dram_start_addr = self.get_program_dram_addr()
        cur_inst_count = self.capture_count
        jump_target_word_addr = ue_35bit_addr_shifter(
            program_dram_start_addr + (cur_inst_count + 1) * INSTRUCTION_SIZE_BYTES
        )
        self.generate_instruction_jump_abs(jump_target_word_addr)
        # gpr_M_reg holds n_chunks (primed by caller with M // chunk_size).
        self.loop_start(loop_cnt=n_chunks, gpr_loop_cnt=gpr_M_reg)

        # Load chunk_size rows in one DMA.
        self.accelerator_memory_to_sram(
            accelerator_dram_address=chunk_bytes,
            sram_address=vector_sram_addr,
            element_size=0,
            inst_pointer_idx=row_load_ptr,
        )

        # Python-unroll chunk_size layer_norm_core calls at consecutive URAM_A offsets.
        # This mirrors what the legacy inner loop does after loading the chunk.
        for j in range(chunk_size):
            row_sram = vector_sram_addr + j * row_bytes
            self.layer_norm_core(
                vector_sram_start_addr=row_sram,
                output_sram_wb_addr=row_sram,
                N=N,
                zeros_sram_start_addr=zeros_sram_addr,
                input_gamma_sram_start_addr=gamma_sram_addr,
                input_beta_sram_start_addr=beta_sram_addr,
            )

        # Store chunk_size rows in one DMA.
        self.sram_to_accelerator_memory(
            sram_address=vector_sram_addr,
            accelerator_dram_address=chunk_bytes,
            element_size=0,
            inst_pointer_idx=row_store_ptr,
        )

        outer_loop_size = self.loop_end()
        print(f"Layer norm PBI outer loop body size: {outer_loop_size} "
            f"({chunk_size} rows/chunk × {n_chunks} chunks=GPR[{gpr_M_reg}], N={N})"
        )
        assert outer_loop_size <= 256, (
            f"Outer loop body {outer_loop_size} instructions exceeds i-cache limit of 256"
        )

        self.release_inst_ptr(row_store_ptr)
        self.release_inst_ptr(row_load_ptr)

        total_flops = 5 * M * N
        if gamma_sram_addr is not None:
            total_flops += M * N
        if beta_sram_addr is not None:
            total_flops += M * N
        return total_flops

    def layer_norm_core_dram_dynamic(self, M: int, N: int, A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                                     gpr_M_reg: int, gpr_N_reg: int, gpr_sqrt_n_reg: int,
                                     gpr_n_scale_reg: Optional[int] = None,
                                     gpr_rms_scale_reg: Optional[int] = None,
                                     GAMMA_DRAM_ADDR: int = None, BETA_DRAM_ADDR: int = None, ZEROS_DRAM_ADDR: int = None,
                                     INV_N_DRAM_ADDR: int = None, MASK_DRAM_ADDR: Optional[int] = None,
                                     gpr_a_addr: Optional[int] = None, gpr_out_addr: Optional[int] = None,
                                     gpr_gamma_addr: Optional[int] = None, gpr_beta_addr: Optional[int] = None,
                                     gpr_invn_addr: Optional[int] = None, gpr_mask_addr: Optional[int] = None) -> int:
        """Layer norm with runtime M, runtime ``sqrt(N)`` scalar, N baked at capture.

        Chunk-tiled, like :meth:`rms_norm_core_dram_dynamic`: batches up to
        ``U = min(URAM_A capacity, 16)`` rows per DMA and runs the full per-row LayerNorm sequence
        (mean prescale+reduce -> subtract -> optional mask -> rms -> scale -> optional gamma/beta) as
        PLAIN (non-PBI) software-unrolled ops at compile-time SRAM offsets; only the two DMA pointers
        (``load_ptr``/``store_ptr``) are PBI. Each unrolled row gets its own scratch row-group (past
        the U staged vector rows) so the prescale/reduce pairs stay independent across rows.

        **No runtime LALU scalars — both N-dependent scalars are folded away:**

          * **mean** ``= sum(x)/N``: prescale ``xn = x*inv_n`` into a per-row scratch region (so
            ``x`` survives at its slot for the subtract with no copy), then ``ADD_REDUCE(xn)`` +
            ``RECIP(scalar=1.0)`` (``ADD_REDUCE`` swaps the LALU operands so reduce=numerator,
            scalar=denominator -> mean/1). ``inv_n`` (every element = ``1/N``) is uploaded by the
            caller (``INV_N_DRAM_ADDR``), like gamma. **Both instructions must be plain (non-PBI):**
            ``ADD_REDUCE_MODE``'s LALU operand-swap mux in ``accelerator_wrapper.sv`` only works
            through a plain ``INSTRUCTION_UE_OP`` — driving it through a PBI pointer produces garbage
            (confirmed on HW). Correctness when a replay's runtime N is smaller than the compile-time
            N depends on ``inv_n`` being zero out to the compile-time N, not just the nearest 64.
          * **1/std** ``= sqrt(N)/sqrt(sum(centered^2))``: the ``RSQRT`` scalar is a **runtime**
            value read from ``gpr_sqrt_n_reg`` (caller-primed with ``float_to_bf19(sqrt(real_N))``) via
            a complete RMS PBI row whose shared field 11 is initialized using
            ``PBI_FIELD.LALU_SCALAR``. Every per-row RSQRT is a normal UE_PBI operation. The caller
            uploads a **plain**
            gamma, exactly as in :meth:`rms_norm_core_dram_dynamic`. The **mean** ``1/N`` is still
            delivered as the ``inv_n`` vector (already runtime — the host uploads the right ``1/N``;
            its bf16 rounding is second-order).

        ``gpr_sqrt_n_reg`` (1..63) = ``float_to_bf19(sqrt(real_N))`` is required. ``gpr_n_scale_reg`` /
        ``gpr_rms_scale_reg`` are ignored (kept only for signature compatibility).

        Preloads (zeros/inv_n/gamma/beta/mask into URAM_B) are issued as plain non-PBI DMAs, not
        chained ``pbi_init`` reconfigures of one pointer: ``pbi_init``'s ``PBI_SET`` is prefetchable
        (gated only on compute-engine busy, not on a DMA still draining), so a later preload's
        ``pbi_init`` can race ahead of an earlier preload's still-in-flight DMA and silently corrupt
        it (confirmed on HW). Dynamic preloads use the serialized one-shot PBI helpers.

        Only two PBI pointers (load_ptr, store_ptr) are live — all compute is plain — well under the
        hardware's live-PBI-pointer limit, see [[dynamic-core-pbi-pointer-limit]].

        ``gpr_M_reg`` (1..15) and ``gpr_N_reg`` (1..63) are required and caller-primed.
        ``GAMMA_DRAM_ADDR`` / ``BETA_DRAM_ADDR`` are optional; ``INV_N_DRAM_ADDR`` is **required**
        (the ``1/N`` vector). ``gpr_*_addr`` optionally source DRAM bases from GPRs. ``N`` a multiple
        of ``UE_VECTOR_SIZE`` and ``<= 65536`` (so the inv_n/zeros/gamma/beta bands fit URAM_B and the
        vector fits URAM_A); ``<= 51200`` when ``MASK_DRAM_ADDR`` is given (the mask band needs
        URAM_B room too).

        **Non-64-aligned hidden_size** is served by host-side zero padding to ``padded_N`` (a
        multiple of 64), same idea as :meth:`rms_norm_core_dram_dynamic`, plus one extra wrinkle:
        mean-centering means plain zero-padding alone contaminates both the mean (pad lanes still get
        counted by a naive ``1/padded_N``) and the variance (subtracting the mean turns pad lanes'
        value from 0 into ``-mean``, leaking into the RMS reduce). Both are neutralized without any
        RTL change:

          * The mean contamination is fixed by also zero-padding ``inv_n`` itself: real lanes carry
            ``1/N`` (the true, un-padded N), pad lanes carry ``0``, so ``xn = x*inv_n`` is already
            exactly 0 in the pad lanes.
          * The variance contamination is fixed by ``MASK_DRAM_ADDR``, a caller-uploaded 0/1 vector
            (``1`` for the ``N`` real lanes, ``0`` for the pad lanes). After subtracting the mean
            (which reintroduces ``-mean`` in the pad lanes), the body multiplies by this mask before
            the RMS reduce, re-zeroing the pad lanes.

        ``MASK_DRAM_ADDR`` is optional; omitting it reproduces the exact instruction sequence for
        64-aligned callers. Gamma/beta pad lanes don't need special handling: after masking they're
        exactly 0 through the gamma multiply, and the caller slices ``[:N]`` off the output anyway.
        """
        fn = "layer_norm_core_dram_dynamic"
        assert M >= 1 and N >= 1, f"{fn}: require M>=1 and N>=1"
        if N % UE_VECTOR_SIZE != 0:
            raise ValueError(f"{fn}: N must be a multiple of UE_VECTOR_SIZE={UE_VECTOR_SIZE}, got N={N}")
        BAND_ROWS = 800 if MASK_DRAM_ADDR is not None else 1024
        max_N_for_bands = BAND_ROWS * UE_VECTOR_SIZE
        if N > max_N_for_bands:
            raise ValueError(f"{fn}: N={N} exceeds {max_N_for_bands} (inv_n/zeros/gamma/beta{'/mask' if MASK_DRAM_ADDR is not None else ''} URAM_B bands must fit)")
        if N > URAM_NEAR_FULL_ELEMENTS:
            raise ValueError(f"{fn}: N={N} exceeds URAM_NEAR_FULL_ELEMENTS={URAM_NEAR_FULL_ELEMENTS} (vector must fit URAM_A staging)")
        if INV_N_DRAM_ADDR is None:
            raise ValueError(f"{fn}: INV_N_DRAM_ADDR is required (the 1/N vector; caller uploads a length-N tensor of value 1/N)")
        if GAMMA_DRAM_ADDR is None:
            raise ValueError(f"{fn}: GAMMA_DRAM_ADDR is required (the per-row gamma multiply is always "
                             "emitted; pass a plain ones vector for a no-gamma layer norm)")
        # gpr_n_scale_reg / gpr_rms_scale_reg are deprecated/ignored. gpr_M_reg / gpr_N_reg /
        # gpr_sqrt_n_reg are the required runtime registers (sqrt(N) is a runtime RSQRT scalar).
        for _nm, _rg, _hi in (("gpr_M_reg", gpr_M_reg, 15), ("gpr_N_reg", gpr_N_reg, 63),
                              ("gpr_sqrt_n_reg", gpr_sqrt_n_reg, 63)):
            if _rg is None or not (1 <= _rg <= _hi):
                raise ValueError(f"{fn}: {_nm} must be a GPR index 1..{_hi}, got {_rg}")
        if len({gpr_M_reg, gpr_N_reg, gpr_sqrt_n_reg}) != 3:
            raise ValueError(f"{fn}: gpr_M_reg/gpr_N_reg/gpr_sqrt_n_reg must be distinct")
        if gpr_gamma_addr is not None and GAMMA_DRAM_ADDR is None:
            raise ValueError(f"{fn}: gpr_gamma_addr given but GAMMA_DRAM_ADDR is None")
        if gpr_beta_addr is not None and BETA_DRAM_ADDR is None:
            raise ValueError(f"{fn}: gpr_beta_addr given but BETA_DRAM_ADDR is None")
        if gpr_mask_addr is not None and MASK_DRAM_ADDR is None:
            raise ValueError(f"{fn}: gpr_mask_addr given but MASK_DRAM_ADDR is None")
        self._validate_addr_gprs(fn, gpr_M_reg, {
            "gpr_a_addr": gpr_a_addr, "gpr_out_addr": gpr_out_addr,
            "gpr_gamma_addr": gpr_gamma_addr, "gpr_beta_addr": gpr_beta_addr, "gpr_invn_addr": gpr_invn_addr,
            "gpr_mask_addr": gpr_mask_addr,
        })
        for _nm, _rg in (("gpr_a_addr", gpr_a_addr), ("gpr_out_addr", gpr_out_addr),
                         ("gpr_gamma_addr", gpr_gamma_addr), ("gpr_beta_addr", gpr_beta_addr),
                         ("gpr_invn_addr", gpr_invn_addr), ("gpr_mask_addr", gpr_mask_addr)):
            if _rg is not None and _rg in (gpr_N_reg, gpr_sqrt_n_reg):
                raise ValueError(f"{fn}: {_nm}={_rg} collides with gpr_N_reg/gpr_sqrt_n_reg")
        assert self.is_capture_on, f"{fn}() requires active capture"

        # N baked at capture, M runtime (see class docstring). The mean stays a plain ADD_REDUCE
        # (broken via PBI — see [[pbi-hazards-add-reduce-and-preload-race]]).
        #
        # URAM_A: U vector row-groups [0, U*N/64) + one shared SCRATCH row-group (prescale output, so x
        # survives for the subtract). URAM_B bands: zeros@0, gamma@1*BAND_ROWS, beta@2*BAND_ROWS,
        # inv_n@3*BAND_ROWS, mask(optional)@4*BAND_ROWS.
        Z_ZEROS, Z_GAMMA, Z_BETA, Z_INV_N = 0, BAND_ROWS, 2 * BAND_ROWS, 3 * BAND_ROWS
        Z_MASK = 4 * BAND_ROWS if MASK_DRAM_ADDR is not None else None
        vector_sram_addr = 0x00000
        zeros_sram_addr  = 0x80000
        gamma_sram_addr  = 0x80000 + Z_GAMMA * UE_VECTOR_SIZE * 2
        beta_sram_addr   = 0x80000 + Z_BETA * UE_VECTOR_SIZE * 2
        invn_sram_addr   = 0x80000 + Z_INV_N * UE_VECTOR_SIZE * 2
        mask_sram_addr   = 0x80000 + Z_MASK * UE_VECTOR_SIZE * 2 if Z_MASK is not None else None
        _, vector_uram_row = self.sram_address_to_uram_address(vector_sram_addr)

        bytes_per_element = 2
        row_bytes = N * bytes_per_element            # compile-time
        MAX_ROWS = N // UE_VECTOR_SIZE               # compile-time URAM rows per token / mean row_size
        # std's sqrt(N) is a RUNTIME RSQRT scalar read from gpr_sqrt_n_reg via a set-once scalar
        # pointer (below), not baked here. The caller primes it with float_to_bf19(sqrt(real_N)) — the
        # UNPADDED size (the RMS reduce sums centered^2 over the padded row but the mask zeroes pad
        # lanes, so the sqrt factor must use real_N), and uploads a plain gamma.

        # Unroll factor U: U vector row-groups + U per-row scratch row-groups must fit URAM_A; capped
        # at 16 so the unrolled body stays under the i-cache window, AND capped at the template M so a
        # small-M caller doesn't over-compute (the last chunk always computes U rows but stores only
        # rows_take, so U > real M wastes compute — see rms_norm_core_dram_dynamic). M is the template
        # row count; the runtime count rides gpr_M_reg, so the cap only sets the unroll granularity and
        # never changes correctness for a larger runtime M. Each unrolled row gets its own scratch
        # (SCRATCH_BASE + j*MAX_ROWS) rather than sharing one, since a shared scratch would be a WAR
        # hazard between row j's reduce and row j+1's prescale (URAM addresses aren't hazard-tracked
        # the way the single LALU register is).
        U = max(1, min(URAM_NEAR_FULL_ADDR // MAX_ROWS // 2, 16, M))
        SCRATCH_BASE = vector_uram_row + U * MAX_ROWS  # per-row prescale scratch, past all U vector rows

        _alloc_list = []
        def _alloc():
            r = self.alloc_isa_reg(); _alloc_list.append(r); return r

        remaining_reg = _alloc()   # rows not yet processed
        rows_take_reg = _alloc()   # min(remaining, U) -- this chunk's row count
        dma_len_reg   = _alloc()   # rows_take*row_bytes (chunk DMA length, load and store)
        u_reg         = _alloc()   # U literal (reg_min clamp)
        # dma_len = rows_take * row_bytes: a 1-cycle SHL when row_bytes is a power of two (the common
        # case), else a mul32 (rare non-pow2 host-padded N).
        row_bytes_is_pow2 = (row_bytes & (row_bytes - 1)) == 0
        row_bytes_shift = (row_bytes.bit_length() - 1) if row_bytes_is_pow2 else 0
        rowb_reg = None if row_bytes_is_pow2 else _alloc()
        if rowb_reg is not None:
            self.generate_instruction_add_set(rowb_reg, row_bytes)
        self.generate_instruction_add_set(u_reg, U)

        def _set(ptr, field, reg):
            self.generate_instruction_pbi_inc(general_reg_src=reg, pbi_field_select=field, inst_pointer_idx=ptr)

        def _dma_len():
            if row_bytes_is_pow2:
                self.generate_instruction_shl(dma_len_reg, rows_take_reg, row_bytes_shift)
            else:
                self.generate_instruction_mul32_reg(dma_len_reg, rows_take_reg, rowb_reg)

        load_ptr  = self.alloc_inst_ptr()
        store_ptr = self.alloc_inst_ptr()
        rms_ptr = self.alloc_inst_ptr()   # complete UE_PBI state for the runtime-scalar RMS op

        # --- one-shot pre-loop loads of zeros / inv_n / gamma / beta / mask into URAM_B ---
        if ZEROS_DRAM_ADDR is not None:
            zeros_dram_addr = ZEROS_DRAM_ADDR
        else:
            zeros_dram_addr = self.get_params_dram_addr()
            self.allocate_params_dram(N * 2)
            self.dma_write(DMA_DEVICE_H2C, zeros_dram_addr, torch.zeros(N, dtype=torch.bfloat16), N * 2)

        # Plain (non-PBI) DMA preloads — see [[pbi-hazards-add-reduce-and-preload-race]] (chained
        # pbi_init reconfigures can race ahead of a still-draining DMA).
        def _preload(dram_addr, sram_addr, gpr_base):
            self.accelerator_memory_to_sram(
                accelerator_dram_address=(0 if gpr_base is not None else dram_addr),
                sram_address=sram_addr, element_size=N, general_reg_src=gpr_base)

        _preload(zeros_dram_addr, zeros_sram_addr, None)
        _preload(INV_N_DRAM_ADDR, invn_sram_addr, gpr_invn_addr)
        if GAMMA_DRAM_ADDR is not None:
            _preload(GAMMA_DRAM_ADDR, gamma_sram_addr, gpr_gamma_addr)
        if BETA_DRAM_ADDR is not None:
            _preload(BETA_DRAM_ADDR, beta_sram_addr, gpr_beta_addr)
        if MASK_DRAM_ADDR is not None:
            _preload(MASK_DRAM_ADDR, mask_sram_addr, gpr_mask_addr)

        # load_ptr / store_ptr: ONE batched transfer of rows_take rows per chunk. Both auto-increment
        # their DRAM cursor by the full-chunk stride U*row_bytes each transfer, so no per-chunk
        # DRAM_ADDR set is needed. DMA_LENGTH (actual transfer size) is set per chunk.
        chunk_stride = U * row_bytes
        self.generate_instruction_pbi_init(dram_shared_addr=A_DRAM_ADDR, dma_length=chunk_stride,
                                           uram_dst_addr=vector_uram_row, inst_pointer_idx=load_ptr)
        self.generate_instruction_pbi_init(dram_shared_addr=OUTPUT_DRAM_ADDR, dma_length=chunk_stride,
                                           uram_a_start_addr=vector_uram_row,
                                           uram_b_start_addr=vector_uram_row, inst_pointer_idx=store_ptr)
        # Optional dynamic DRAM bases: override the pointer bases once (auto-inc then walks from there).
        if gpr_a_addr is not None:
            self._pbi_override_dram_base_from_gpr(load_ptr, gpr_a_addr)
        if gpr_out_addr is not None:
            self._pbi_override_dram_base_from_gpr(store_ptr, gpr_out_addr)

        self.generate_instruction_pbi_init(
            uram_length=MAX_ROWS,
            uram_a_start_addr=vector_uram_row,
            inst_pointer_idx=rms_ptr,
        )
        # Arithmetic UE_PBI interprets shared field 11 as lalu_scalar.
        _set(rms_ptr, PBI_FIELD.LALU_SCALAR, gpr_sqrt_n_reg)

        def _ln_row(sram, scratch_row, row_idx):
            # Full per-row LayerNorm as PLAIN ops at a compile-time SRAM offset. mean via inv_n
            # prescale into this row's SCRATCH (x survives at `sram`) + ADD_REDUCE(scratch) +
            # RECIP(1.0); then subtract mean, optional mask, rms-of-centered (RSQRT scalar=sqrt(real_N),
            # so sqrt(N) rides the scalar like legacy and gamma is plain), scale, optional gamma/beta.
            _, urow = self.sram_address_to_uram_address(sram)
            self.ue_arithmetic_op(
                0, 0, 1, 0, 0, LALU_MODE.BYPASS.value, 0, URAM_SECTION.URAM_A.value,
                0, scratch_row, URAM_WRITE_SRC.URAM_WRITE_BACK.value,
                UE_MODE.ELTWISE_MUL, 0, urow, Z_INV_N, MAX_ROWS, 0, 0, 0)
            self.ue_arithmetic_op(
                0, 0, 1, 0, 0, LALU_MODE.MODE_RECIP.value, self.float_to_bf19(1.0), URAM_SECTION.URAM_A.value,
                0, 0, URAM_WRITE_SRC.URAM_WB_DISABLE.value,
                UE_MODE.ADD_REDUCE, 0, scratch_row, Z_ZEROS, MAX_ROWS, 0, 0, 0)
            self.start_queue_broadcast(UE_MODE.ADD_BROADCAST, BROADCAST_MODE.LALU_RESULT_NEGATE, sram, sram, N)
            if MASK_DRAM_ADDR is not None:
                self.eltwise_mul_core(sram, mask_sram_addr, sram, N)
            # RSQRT is a normal UE_PBI arithmetic operation. Its pointer supplies the complete RMS
            # state and runtime scalar; the descriptor advances Y, wrapping after the final row.
            y_delta = MAX_ROWS if row_idx < U - 1 else -((U - 1) * MAX_ROWS)
            self.ue_arithmetic_op(
                0, 0, 0, 0, 0, LALU_MODE.MODE_RSQRT.value, 0,
                URAM_SECTION.URAM_A.value,
                0, 0, URAM_WRITE_SRC.URAM_WB_DISABLE.value,
                UE_MODE.RMS, 0, y_delta, 0, 0, 0, 0, 0,
                inst_pointer_idx=rms_ptr)
            self.start_queue_broadcast(UE_MODE.MUL_BROADCAST, BROADCAST_MODE.LALU_RESULT, sram, sram, N)
            if GAMMA_DRAM_ADDR is not None:
                self.eltwise_mul_core(sram, gamma_sram_addr, sram, N)
            if BETA_DRAM_ADDR is not None:
                self.eltwise_add_core(sram, beta_sram_addr, sram, N)

        # ===== outer chunk loop (2 live PBI pointers: load_ptr, store_ptr) =====
        program_dram_start_addr = self.get_program_dram_addr()
        cur_inst_count = self.capture_count
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(
            program_dram_start_addr + (cur_inst_count + 1) * INSTRUCTION_SIZE_BYTES))

        self.generate_instruction_add_imm(src_reg_idx=gpr_M_reg, immediate_value=0, dst_reg_idx=remaining_reg)
        self.generate_instruction_reg_min(rows_take_reg, remaining_reg, u_reg)
        body_start_inst_cnt = self.capture_count

        # Batched DMA load: rows_take rows in ONE transfer into SRAM rows [0, rows_take). The pointer
        # auto-advances by chunk_stride (=U*row_bytes) for the next chunk.
        _dma_len()
        _set(load_ptr, PBI_FIELD.DMA_LENGTH, dma_len_reg)
        self.accelerator_memory_to_sram(accelerator_dram_address=chunk_stride, sram_address=vector_sram_addr, element_size=0, inst_pointer_idx=load_ptr)

        # Unrolled plain per-row LayerNorm over all U slots; only rows [0, rows_take) are stored.
        for j in range(U):
            _ln_row(vector_sram_addr + j * row_bytes, SCRATCH_BASE + j * MAX_ROWS, j)

        # Batched DMA store: rows_take rows in ONE transfer.
        _set(store_ptr, PBI_FIELD.DMA_LENGTH, dma_len_reg)
        self.sram_to_accelerator_memory(sram_address=vector_sram_addr, accelerator_dram_address=chunk_stride, element_size=0, inst_pointer_idx=store_ptr)

        # Recompute the next chunk's row count (DRAM cursors auto-advanced via the DMA pointers).
        self.generate_instruction_reg_sub(remaining_reg, remaining_reg, rows_take_reg)
        self.generate_instruction_reg_min(rows_take_reg, remaining_reg, u_reg)

        outer_loop_size = self.capture_count - body_start_inst_cnt + 2
        self.generate_instruction_jump_rela_jnz(outer_loop_size, remaining_reg)

        print(f"Layer norm dynamic (chunk-tiled) outer loop body size: {outer_loop_size} "
              f"(U={U} rows/chunk × ceil(M/U) chunks, M=GPR[{gpr_M_reg}], N={N} compile-time addr, "
              f"sqrt(N)=GPR[{gpr_sqrt_n_reg}] runtime, 3 PBI pointers)")
        assert outer_loop_size <= 256, f"{fn}: outer loop body {outer_loop_size} exceeds i-cache budget 256"

        self.release_inst_ptr(rms_ptr)
        self.release_inst_ptr(store_ptr)
        self.release_inst_ptr(load_ptr)
        for _ in range(len(_alloc_list)):
            self.release_isa_reg()

        total_flops = 5 * M * N
        if GAMMA_DRAM_ADDR is not None:
            total_flops += M * N
        if BETA_DRAM_ADDR is not None:
            total_flops += M * N
        return total_flops

    def layer_norm_core_dram_legacy(self, M: int, N: int, A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                                     GAMMA_DRAM_ADDR: int = None, BETA_DRAM_ADDR: int = None,
                                     ZEROS_DRAM_ADDR: int = None) -> int:
        """
        Legacy layer norm DRAM path: compile-time chunk-tiled, M-unrolled.
        """
        zeros_sram_addr = 0x80000

        if ZEROS_DRAM_ADDR is not None:
            zeros_dram_addr = ZEROS_DRAM_ADDR          # caller-supplied shared zeros buffer
        else:
            zeros_dram_addr = self.get_params_dram_addr()
            self.allocate_params_dram(N * 2)
            self.dma_write(DMA_DEVICE_H2C, zeros_dram_addr, torch.zeros(N, dtype=torch.bfloat16), N * 2)

        # These are small enough to fit in URAM_B, the layout zeros + gamma + beta
        self.accelerator_memory_to_sram(accelerator_dram_address=zeros_dram_addr,
                                        sram_address=zeros_sram_addr,
                                        element_size=N)
        params_sram_addr = zeros_sram_addr + N * 2

        gamma_sram_addr = None
        beta_sram_addr = None

        if GAMMA_DRAM_ADDR is not None:
            gamma_sram_addr = params_sram_addr
            self.accelerator_memory_to_sram(accelerator_dram_address=GAMMA_DRAM_ADDR,
                                        sram_address=gamma_sram_addr,
                                        element_size=N)
            params_sram_addr += N * 2

        if BETA_DRAM_ADDR is not None:
            beta_sram_addr = params_sram_addr
            self.accelerator_memory_to_sram(accelerator_dram_address=BETA_DRAM_ADDR,
                                        sram_address=beta_sram_addr,
                                        element_size=N)
            params_sram_addr += N * 2

        chunk_size = min(URAM_NEAR_FULL_ELEMENTS // N, M)
        print(f"Chunk size: {chunk_size} for M={M}, N={N}")
        assert chunk_size >= 1 and chunk_size <= M, f"chunk_size={chunk_size} must be greater than 0 and less than M={M}"

        vector_sram_addr = 0x00000

        for i, m_take in self.chunk_ranges(M, chunk_size):
            self.accelerator_memory_to_sram(accelerator_dram_address=A_DRAM_ADDR + i * N * 2,
                                        sram_address=vector_sram_addr,
                                        element_size=m_take * N)

            for j in range(m_take):
                self.layer_norm_core(vector_sram_addr + j * N * 2, vector_sram_addr + j * N * 2, N, zeros_sram_addr, gamma_sram_addr, beta_sram_addr)

            self.sram_to_accelerator_memory(sram_address=vector_sram_addr,
                                            accelerator_dram_address=OUTPUT_DRAM_ADDR + i * N * 2,
                                            element_size=m_take * N)

        # Total Theoretical FLOPS:
        total_flops = 5 * M * N # mean(N), subtract(N), variance(N), sum(N), rsqrt(1), scale(N)
        if gamma_sram_addr is not None:
            total_flops += M * N # mul(gamma) N times
        if beta_sram_addr is not None:
            total_flops += M * N # add(beta) N times
        return total_flops

    def layer_norm_core_dram(self, M: int, N: int, A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                              GAMMA_DRAM_ADDR: int = None, BETA_DRAM_ADDR: int = None,
                              gpr_M_reg: Optional[int] = None, ZEROS_DRAM_ADDR: int = None,
                              gpr_N_reg: Optional[int] = None, gpr_n_scale_reg: Optional[int] = None,
                              gpr_rms_scale_reg: Optional[int] = None, INV_N_DRAM_ADDR: int = None,
                              MASK_DRAM_ADDR: Optional[int] = None,
                              gpr_a_addr: Optional[int] = None, gpr_out_addr: Optional[int] = None,
                              gpr_gamma_addr: Optional[int] = None, gpr_beta_addr: Optional[int] = None,
                              gpr_invn_addr: Optional[int] = None, gpr_mask_addr: Optional[int] = None,
                              gpr_sqrt_n_reg: Optional[int] = None) -> int:
        """Layer norm DRAM entrypoint; dispatches to dynamic for any runtime argument.

        - Any runtime dimension/address GPR, ``INV_N_DRAM_ADDR``, or ``MASK_DRAM_ADDR``:
          :meth:`layer_norm_core_dram_dynamic`. Missing M/N/sqrt(N) registers are allocated and seeded
          from compile-time values, so the former runtime-M-only contract is handled by this path.
          ``gpr_n_scale_reg`` / ``gpr_rms_scale_reg`` are deprecated/ignored. ``MASK_DRAM_ADDR``
          optionally enables non-64-aligned N via host-side zero padding.
        - No runtime arguments (default): :meth:`layer_norm_core_dram_legacy` — compile-time
          chunk-tiled, M-unrolled path.

        ``gpr_a_addr`` / ``gpr_out_addr`` / ``gpr_gamma_addr`` / ``gpr_beta_addr`` (dynamic path)
        optionally source the input/output/gamma/beta DRAM bases from GPRs (word addr). The constant
        zeros buffer stays a literal because it does not vary per layer.
        """
        runtime_args = (gpr_M_reg, gpr_N_reg, gpr_sqrt_n_reg, gpr_a_addr, gpr_out_addr,
                        gpr_gamma_addr, gpr_beta_addr, gpr_invn_addr, gpr_mask_addr)
        if any(r is not None for r in runtime_args) or INV_N_DRAM_ADDR is not None or MASK_DRAM_ADDR is not None:
            seeded_regs = []
            if gpr_M_reg is None:
                gpr_M_reg = self.alloc_isa_reg(); seeded_regs.append(gpr_M_reg)
                self.generate_instruction_add_set(gpr_M_reg, M)
            if gpr_N_reg is None:
                gpr_N_reg = self.alloc_isa_reg(); seeded_regs.append(gpr_N_reg)
                self.generate_instruction_add_set(gpr_N_reg, N)
            if gpr_sqrt_n_reg is None:
                gpr_sqrt_n_reg = self.alloc_isa_reg(); seeded_regs.append(gpr_sqrt_n_reg)
                self.generate_instruction_add_set(gpr_sqrt_n_reg, self.float_to_bf19(float(N ** 0.5)))

            # The retired PBI path did not need these vectors. Materialize static compatibility
            # parameters when a runtime-M-only caller is transparently promoted to dynamic.
            if GAMMA_DRAM_ADDR is None:
                if gpr_gamma_addr is not None:
                    raise ValueError("layer_norm_core_dram: gpr_gamma_addr requires GAMMA_DRAM_ADDR")
                GAMMA_DRAM_ADDR = self.get_params_dram_addr()
                self.allocate_params_dram(N * 2)
                self.dma_write(DMA_DEVICE_H2C, GAMMA_DRAM_ADDR,
                               torch.ones(N, dtype=torch.bfloat16), N * 2)
            if INV_N_DRAM_ADDR is None:
                if gpr_invn_addr is not None:
                    raise ValueError("layer_norm_core_dram: gpr_invn_addr requires INV_N_DRAM_ADDR")
                INV_N_DRAM_ADDR = self.get_params_dram_addr()
                self.allocate_params_dram(N * 2)
                self.dma_write(DMA_DEVICE_H2C, INV_N_DRAM_ADDR,
                               torch.full((N,), 1.0 / N, dtype=torch.bfloat16), N * 2)
            # gpr_n_scale_reg / gpr_rms_scale_reg are accepted for backward compat but ignored (1/N via
            # INV_N_DRAM_ADDR vector; sqrt(N) via gpr_sqrt_n_reg).
            result = self.layer_norm_core_dram_dynamic(
                M=M, N=N, A_DRAM_ADDR=A_DRAM_ADDR, OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                gpr_M_reg=gpr_M_reg, gpr_N_reg=gpr_N_reg, gpr_sqrt_n_reg=gpr_sqrt_n_reg,
                GAMMA_DRAM_ADDR=GAMMA_DRAM_ADDR, BETA_DRAM_ADDR=BETA_DRAM_ADDR, ZEROS_DRAM_ADDR=ZEROS_DRAM_ADDR,
                INV_N_DRAM_ADDR=INV_N_DRAM_ADDR, MASK_DRAM_ADDR=MASK_DRAM_ADDR,
                gpr_a_addr=gpr_a_addr, gpr_out_addr=gpr_out_addr,
                gpr_gamma_addr=gpr_gamma_addr, gpr_beta_addr=gpr_beta_addr, gpr_invn_addr=gpr_invn_addr,
                gpr_mask_addr=gpr_mask_addr,
            )
            for _ in seeded_regs:
                self.release_isa_reg()
            return result
        return self.layer_norm_core_dram_legacy(
            M=M, N=N,
            A_DRAM_ADDR=A_DRAM_ADDR, OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
            GAMMA_DRAM_ADDR=GAMMA_DRAM_ADDR, BETA_DRAM_ADDR=BETA_DRAM_ADDR,
            ZEROS_DRAM_ADDR=ZEROS_DRAM_ADDR,
        )


    def rope_hf_core_decode(self, N: int, input_dram_addr: int, output_dram_addr: int,
                     cos_dram_addr: int = 0, sin_dram_addr: int = 0,
                     rope_size_reg: int = None, gr_weight_dram: int = 0,
                     output_addr_inc_reg: int = None, tmp_reg: int = None) -> int:
        """HuggingFace-style RoPE on N bf16 elements. Caller must bracket with start/stop_capture.

        Compute flow (single token, head_dim=N, half=N//2):
          1. Load x[N]          → sram_x
          2. Load cos[N], sin[N] → sram_cos, sram_sin
             Table layout per position: [cos(half) | cos(half) | -sin(half) | sin(half)]
             so sram_cos holds the full duplicated cos and sram_sin holds the signed sin.
          3. a[N]    = x[N]      * cos[N]           (eltwise_mul, full N)
          4. bc_lo[half] = x[half+:] * sin[:half]   (eltwise_mul, rotate_half lo: x_hi * (-sin_lo))
          5. bc_hi[half] = x[:half]  * sin[half:]   (eltwise_mul, rotate_half hi: x_lo * sin_hi)
          6. d[N]    = a[N] + bc[N]                 (eltwise_add, full N)
          7. Write d[N] → output_dram_addr

        Three mutually exclusive dynamic-address modes for cos/sin:
        - Static (default): load from cos_dram_addr / sin_dram_addr as-is.
        - rope_size_reg: ISA register holding a per-position byte offset added to cos/sin_dram_addr
          at runtime. tmp_reg required.
        - gr_weight_dram: ISA register holding the absolute cos base address at runtime; sin derived
          as cos_base + N*2 bytes. Allocates an internal scratch register.

        output_addr_inc_reg: ISA register holding a per-position byte offset applied to output_dram_addr.
        Returns: approximate FLOPs (4*N).
        """
        assert N % UE_VECTOR_SIZE == 0 and N >= 64, f"N must be a multiple of {UE_VECTOR_SIZE} and >= 64"
        assert N % 2 == 0, "N must be even for RoPE half layout"
        assert N >= 128, "N must be >= 128 so half-vector SRAM offsets are 128-byte aligned"
        half = N // 2
        bytes_per_elem = 2
        sram_x   = 0x00000
        sram_a   = 0x20000
        sram_d   = 0x40000
        sram_cos = 0x80000
        sram_sin = 0x80000 + N * bytes_per_elem
        sram_bc  = 0x80000 + N * bytes_per_elem * 2
        self.accelerator_memory_to_sram(accelerator_dram_address=input_dram_addr, sram_address=sram_x, element_size=N)
        if rope_size_reg is not None:
            self.generate_instruction_add_imm(rope_size_reg, ue_35bit_addr_shifter(cos_dram_addr), tmp_reg)
            self.accelerator_memory_to_sram(accelerator_dram_address=cos_dram_addr, sram_address=sram_cos, element_size=N, general_reg_src=tmp_reg)
            self.generate_instruction_add_imm(rope_size_reg, ue_35bit_addr_shifter(sin_dram_addr), tmp_reg)
            self.accelerator_memory_to_sram(accelerator_dram_address=sin_dram_addr, sram_address=sram_sin, element_size=N, general_reg_src=tmp_reg)
        elif gr_weight_dram != 0:
            tmp_gr = self.alloc_isa_reg()
            self.generate_instruction_add_imm(gr_weight_dram, ue_35bit_addr_shifter(N * bytes_per_elem), tmp_gr)
            self.accelerator_memory_to_sram(accelerator_dram_address=cos_dram_addr, sram_address=sram_cos, element_size=N, general_reg_src=gr_weight_dram)
            self.accelerator_memory_to_sram(accelerator_dram_address=cos_dram_addr + N * bytes_per_elem, sram_address=sram_sin, element_size=N, general_reg_src=tmp_gr)
        else:
            self.accelerator_memory_to_sram(accelerator_dram_address=cos_dram_addr, sram_address=sram_cos, element_size=N)
            self.accelerator_memory_to_sram(accelerator_dram_address=sin_dram_addr, sram_address=sram_sin, element_size=N)
        self.eltwise_mul_core(vector_A_sram_start_addr=sram_x, vector_B_sram_start_addr=sram_cos, vector_C_sram_wb_addr=sram_a, element_size=N)
        self.eltwise_mul_core(vector_A_sram_start_addr=sram_x + half * bytes_per_elem, vector_B_sram_start_addr=sram_sin, vector_C_sram_wb_addr=sram_bc, element_size=half)
        self.eltwise_mul_core(vector_A_sram_start_addr=sram_x, vector_B_sram_start_addr=sram_sin + half * bytes_per_elem, vector_C_sram_wb_addr=sram_bc + half * bytes_per_elem, element_size=half)
        self.eltwise_add_core(vector_A_sram_start_addr=sram_a, vector_B_sram_start_addr=sram_bc, vector_C_sram_wb_addr=sram_d, element_size=N)
        if output_addr_inc_reg is not None:
            self.generate_instruction_add_imm(output_addr_inc_reg, ue_35bit_addr_shifter(output_dram_addr), tmp_reg)
            self.sram_to_accelerator_memory(sram_address=sram_d, accelerator_dram_address=output_dram_addr, element_size=N, general_reg_src=tmp_reg)
        else:
            self.sram_to_accelerator_memory(sram_address=sram_d, accelerator_dram_address=output_dram_addr, element_size=N)
        if gr_weight_dram != 0:
            self.release_isa_reg()
        return 4 * N

    def rope_hf_core_decode_d64(self, N: int, input_dram_addr: int, output_dram_addr: int,
                                packed_table_addr: int, pos_reg: int, tmp_reg: int,
                                output_pos_strided: bool = False) -> int:
        """Single-token decode HF RoPE for head_dim ``N < 128`` (the decode analog of
        :meth:`rope_hf_core_dram_d64_pbi`). :meth:`rope_hf_core_decode` asserts ``N >= 128``;
        this is the N<128 path. Caller must bracket with start_capture()/stop_capture().

        The token at runtime position ``pos_reg`` (= gpr_seq_len) is roped against the contiguous
        per-token packed table ``[cos(N) || sin(N)]`` (sin first half pre-negated on the host); the
        token's row lives at ``packed_table_addr + pos * 2*N*bpe``.

        Alignment discipline (why this path exists): each N/2-element half is < 128 bytes, i.e. half
        a URAM row, and SRAM cannot be addressed mid-row. Each slice is loaded into the START of its
        own 128-byte URAM row via register-addressed DMAs (DRAM is byte-addressable; SRAM targets
        stay row-aligned). No PBI pointers are used.

        input_dram_addr    : single freshly-projected row (static — no position offset).
        output_pos_strided : True  -> write to ``output_dram_addr + pos*N*bpe`` (KV-cache slot, K).
                             False -> write to ``output_dram_addr`` (fixed slot, e.g. Q_PERM).
        """
        assert 0 < N < 128 and N % 2 == 0, "rope_hf_core_decode_d64 is the N<128 even path"
        bpe = 2
        half = N // 2
        half_bytes = half * bpe
        row_bytes = N * bpe
        rope_row_bytes = 2 * row_bytes          # [cos(N) | sin(N)] per token
        shift = ue_35bit_addr_shifter

        SLOT = 128
        uram_x_lo      = 0x00000                # URAM_A (x + cos-products + results)
        uram_x_hi      = 0x00080
        uram_a_lo      = 0x00100
        uram_a_hi      = 0x00180
        uram_result_lo = 0x00200
        uram_result_hi = 0x00280
        uram_cos_lo    = 0x80000                # URAM_B (cos/sin slices + sin-products)
        uram_cos_hi    = uram_cos_lo + SLOT
        uram_sin_lo    = uram_cos_hi + SLOT
        uram_sin_hi    = uram_sin_lo + SLOT
        uram_b_lo      = uram_sin_hi + SLOT
        uram_b_hi      = uram_b_lo   + SLOT

        cos_base = packed_table_addr
        sin_base = packed_table_addr + row_bytes

        def _read_pos(base, sram_addr):
            # tmp_reg = shift(base) + pos_reg * shift(rope_row_bytes) == shift(base + pos*rope_row_bytes)
            self.generate_instruction_reg_mul_imm(tmp_reg, pos_reg, shift(rope_row_bytes))
            self.generate_instruction_add_imm(tmp_reg, shift(base), tmp_reg)
            self.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=sram_addr,
                                            element_size=half, general_reg_src=tmp_reg)

        # Input halves: static reads (single row); URAM targets are row-aligned.
        self.accelerator_memory_to_sram(input_dram_addr, uram_x_lo, half)
        self.accelerator_memory_to_sram(input_dram_addr + half_bytes, uram_x_hi, half)
        # Table halves: position-addressed reads from this token's packed [cos||sin] row.
        _read_pos(cos_base,              uram_cos_lo)   # cos[0:half]
        _read_pos(cos_base + half_bytes, uram_cos_hi)   # cos[half:N]
        _read_pos(sin_base,              uram_sin_lo)   # sin[0:half] (pre-negated)
        _read_pos(sin_base + half_bytes, uram_sin_hi)   # sin[half:N]

        # result = x*cos + rotate_half(x)*sin  (sin_lo pre-negated, so b_lo = x_hi*(-sin))
        self.eltwise_mul_core(uram_x_lo, uram_cos_lo, uram_a_lo, half)
        self.eltwise_mul_core(uram_x_hi, uram_cos_hi, uram_a_hi, half)
        self.eltwise_mul_core(uram_x_hi, uram_sin_lo, uram_b_lo, half)
        self.eltwise_mul_core(uram_x_lo, uram_sin_hi, uram_b_hi, half)
        self.eltwise_add_core(uram_a_lo, uram_b_lo, uram_result_lo, half)
        self.eltwise_add_core(uram_a_hi, uram_b_hi, uram_result_hi, half)

        if output_pos_strided:
            def _write_pos(base, sram_addr):
                self.generate_instruction_reg_mul_imm(tmp_reg, pos_reg, shift(row_bytes))
                self.generate_instruction_add_imm(tmp_reg, shift(base), tmp_reg)
                self.sram_to_accelerator_memory(sram_address=sram_addr, accelerator_dram_address=0,
                                                element_size=half, general_reg_src=tmp_reg)
            _write_pos(output_dram_addr,              uram_result_lo)
            _write_pos(output_dram_addr + half_bytes, uram_result_hi)
        else:
            self.sram_to_accelerator_memory(sram_address=uram_result_lo,
                                            accelerator_dram_address=output_dram_addr, element_size=half)
            self.sram_to_accelerator_memory(sram_address=uram_result_hi,
                                            accelerator_dram_address=output_dram_addr + half_bytes, element_size=half)
        return 4 * N

    def rope_hf_core_dram(self, M: int, N: int, input_dram_addr: int, output_dram_addr: int, cos_dram_addr: int, sin_dram_addr: int, gpr_M_reg: Optional[int] = None, rope_size_reg: int = None, output_addr_inc_reg: int = None, tmp_reg: int = None,
                          gpr_input_addr: Optional[int] = None, gpr_out_addr: Optional[int] = None, gpr_cos_addr: Optional[int] = None,
                          gpr_N_reg: Optional[int] = None) -> int:
        """HF RoPE DRAM entrypoint; dispatches based on ``N``, ``gpr_M_reg`` and ``gpr_N_reg``:

        - Any runtime dimension/address GPR: :meth:`rope_hf_core_dram_dynamic`. Missing M/N registers
          are allocated and seeded from compile-time values. The dynamic core includes the native
          padded-split branch needed when ``N < 128``.
        - No runtime GPRs (default): :meth:`rope_hf_core_dram_legacy` — Python-unrolled rows.
        """
        if any(r is not None for r in (gpr_M_reg, gpr_N_reg, gpr_input_addr, gpr_out_addr, gpr_cos_addr)):
            seeded_regs = []
            if gpr_M_reg is None:
                gpr_M_reg = self.alloc_isa_reg(); seeded_regs.append(gpr_M_reg)
                self.generate_instruction_add_set(gpr_M_reg, M)
            if gpr_N_reg is None:
                gpr_N_reg = self.alloc_isa_reg(); seeded_regs.append(gpr_N_reg)
                self.generate_instruction_add_set(gpr_N_reg, N)
            result = self.rope_hf_core_dram_dynamic(
                M, N, input_dram_addr, output_dram_addr, cos_dram_addr, sin_dram_addr,
                gpr_M_reg=gpr_M_reg, gpr_N_reg=gpr_N_reg,
                gpr_input_addr=gpr_input_addr, gpr_out_addr=gpr_out_addr, gpr_cos_addr=gpr_cos_addr)
            for _ in seeded_regs:
                self.release_isa_reg()
            return result
        return self.rope_hf_core_dram_legacy(M, N, input_dram_addr, output_dram_addr, cos_dram_addr, sin_dram_addr)

    def rope_hf_core_dram_legacy(self, M: int, N: int, input_dram_addr: int, output_dram_addr: int, cos_dram_addr: int, sin_dram_addr: int) -> int:
        """Statically compiled, chunk-batched HuggingFace RoPE.

        Uses the same batch geometry and SRAM reuse as the dynamic core, but all chunk sizes,
        addresses and row offsets are capture-time constants. This keeps legacy as an independent
        static baseline while making DMA granularity comparable to dynamic.
        """
        assert M >= 1, "M must be at least 1"
        assert N % UE_VECTOR_SIZE == 0 and N >= 128, (
            f"N must be a multiple of {UE_VECTOR_SIZE} and >= 128")
        assert N % 2 == 0, "N must be even for RoPE half layout"
        assert sin_dram_addr == cos_dram_addr + N * 2, (
            "RoPE expects contiguous [cos, sin] rows")

        half = N // 2
        row_bytes = N * 2
        rope_row_bytes = 2 * row_bytes
        n_rows = N // UE_VECTOR_SIZE
        chunk_rows = min(M, URAM_NEAR_FULL_ADDR // (3 * n_rows))
        assert chunk_rows >= 1, f"rope_hf_core_dram_legacy: N={N} does not fit URAM"

        # Match dynamic's bank layout: URAM_A=[x/out batch | a batch],
        # URAM_B=[cos+sin batch | b batch].
        sram_x_base = 0x00000
        sram_a_base = sram_x_base + chunk_rows * row_bytes
        sram_rope_base = 0x80000
        sram_b_base = sram_rope_base + chunk_rows * rope_row_bytes

        for row_start, rows_take in self.chunk_ranges(M, chunk_rows):
            self.accelerator_memory_to_sram(
                accelerator_dram_address=input_dram_addr + row_start * row_bytes,
                sram_address=sram_x_base,
                element_size=rows_take * N)
            self.accelerator_memory_to_sram(
                accelerator_dram_address=cos_dram_addr + row_start * rope_row_bytes,
                sram_address=sram_rope_base,
                element_size=rows_take * 2 * N)

            for j in range(rows_take):
                x = sram_x_base + j * row_bytes
                a = sram_a_base + j * row_bytes
                cos = sram_rope_base + j * rope_row_bytes
                sin = cos + row_bytes
                b = sram_b_base + j * row_bytes

                self.eltwise_mul_core(x, cos, a, N)
                self.eltwise_mul_core(x + half * 2, sin, b, half)
                self.eltwise_mul_core(x, sin + half * 2, b + half * 2, half)
                self.eltwise_add_core(a, b, x, N)

            self.sram_to_accelerator_memory(
                sram_address=sram_x_base,
                accelerator_dram_address=output_dram_addr + row_start * row_bytes,
                element_size=rows_take * N)

        return 4 * M * N

    def rope_hf_core_dram_pbi(self, M: int, N: int, input_dram_addr: int, output_dram_addr: int,
                              cos_dram_addr: int, sin_dram_addr: int, gpr_M_reg: int = None,
                              gpr_input_addr: Optional[int] = None, gpr_out_addr: Optional[int] = None, gpr_cos_addr: Optional[int] = None) -> int:
        # DEPRECATED REFERENCE ONLY: deliberately unreachable, even under ``python -O``.
        raise RuntimeError("rope_hf_core_dram_pbi is deprecated; use rope_hf_core_dram_dynamic")

    def _rope_hf_core_dram_dynamic_small_n(self, M: int, N: int, input_dram_addr: int, output_dram_addr: int, cos_dram_addr: int, sin_dram_addr: int, gpr_M_reg: int = None,
                              gpr_input_addr: Optional[int] = None, gpr_out_addr: Optional[int] = None, gpr_cos_addr: Optional[int] = None) -> int:
        """PBI-backed HF RoPE over M rows. Caller must have start_capture() before and stop_capture() after.

        **Batch dimension / M (dynamic, required):**
        ``gpr_M_reg`` is a **required** GPR index 1..15 holding the runtime row count. Caller must
        prime that register beforehand (typically with ``ADD_SET``). The hardware loop runs for
        whatever value that register holds at execute time; ``M`` is FLOPs-accounting / template
        only — the captured program has no static reference to ``M``.

        **DRAM base addresses (optional dynamic):** ``gpr_input_addr`` / ``gpr_out_addr`` /
        ``gpr_cos_addr`` are optional GPR indices (1..63) holding the input / output / cos-table
        base as a **word** address (``byte >> 3``); ``sin`` stays contiguous after ``cos`` (the
        ``sin == cos + row`` invariant holds at runtime too). When given, that base is sourced from
        the GPR so one captured program serves any placement. In the N<128 branch the bases seed the
        internal per-row cursor registers (copied so the caller's GPR is preserved across the loop);
        in the N>=128 branch they override the PBI pointer bases. ``None`` → literal. See
        :meth:`eltwise_core_dram_pbi`.

        When N//2 * 2 < 128 (e.g. N=64, half=32), each half is placed in a padded 128-byte SRAM
        slot. Eight PBI pointers are used (2×x, 4×rope, 2×output), each transferring half_bytes
        per iteration with row-stride advances.
        """
        assert M >= 1, "M must be at least 1"
        assert N % 2 == 0, "N must be even for RoPE half layout"
        self._validate_addr_gprs("rope_hf_core_dram_pbi", gpr_M_reg, {
            "gpr_input_addr": gpr_input_addr, "gpr_out_addr": gpr_out_addr, "gpr_cos_addr": gpr_cos_addr,
        })
        # The padded-split branch (N<128) places each half in its own 128-byte SRAM slot, so on
        # the SRAM side it only needs even N. Its per-half DMAs, however, start at multiples of
        # half_bytes (= N bytes) in DRAM, and the DMA engine requires AXI-beat-aligned addresses
        # — so N bytes must also be a beat multiple (32 B at 256-bit, 64 B at 512-bit; asserted
        # below). The N>=128 branch slices SRAM mid-row, which requires 64-aligned halves (N a
        # multiple of 64) and is thereby always beat-aligned.
        if N < 128:
            assert 0 < N < 128, "padded-split RoPE expects 0 < N < 128"
        else:
            assert N % UE_VECTOR_SIZE == 0, f"N>=128 RoPE expects N a multiple of {UE_VECTOR_SIZE}"
        if gpr_M_reg is None or not (1 <= gpr_M_reg <= 15):
            raise ValueError(f"rope_hf_core_dram_pbi: gpr_M_reg must be a GPR index 1..15, got {gpr_M_reg}")

        half = N // 2
        bytes_per_elem = 2
        half_bytes = half * bytes_per_elem
        row_bytes = N * bytes_per_elem
        rope_row_bytes = 2 * row_bytes
        SRAM_SLOT = 128
        needs_padding = half_bytes < SRAM_SLOT  # True when N < 128 (e.g. N=64, half=32)

        assert sin_dram_addr == cos_dram_addr + row_bytes, "PBI RoPE expects contiguous [cos, sin] rows"

        if needs_padding:
            ue_assert_axi_beat_aligned_bytes(
                half_bytes,
                f"rope_hf_core_dram_pbi (padded-split): half_bytes (N={N})",
                hint="slice DMAs must start on an AXI-beat boundary; "
                     "host-pad each rotate-half up to a beat multiple")
            # Padded split layout: each half occupies a full 128-byte SRAM slot.
            # No PBI pointers — ISA registers (x_reg, cos_reg, out_reg) track the current-row
            # DRAM addresses and are incremented by the relevant stride at the end of each
            # loop body via ADD_IMM.  Per-halfrow loads/stores use
            # runtime-addressed DMA operations lowered through PBI_MODE_REG.
            sram_x_lo      = 0x00000
            sram_x_hi      = sram_x_lo + SRAM_SLOT
            sram_a_lo      = sram_x_hi + SRAM_SLOT
            sram_a_hi      = sram_a_lo + SRAM_SLOT
            sram_result_lo = sram_a_hi + SRAM_SLOT
            sram_result_hi = sram_result_lo + SRAM_SLOT
            sram_cos_lo    = 0x80000
            sram_cos_hi    = sram_cos_lo + SRAM_SLOT
            sram_sin_lo    = sram_cos_hi + SRAM_SLOT
            sram_sin_hi    = sram_sin_lo + SRAM_SLOT
            sram_bc_lo     = sram_sin_hi + SRAM_SLOT
            sram_bc_hi     = sram_bc_lo + SRAM_SLOT

            x_reg   = self.alloc_isa_reg()
            cos_reg = self.alloc_isa_reg()
            out_reg = self.alloc_isa_reg()
            tmp_r   = self.alloc_isa_reg()

            # Prime per-row cursor registers with the starting DRAM addresses (word/shifted format).
            # When a base GPR is supplied, COPY it into the cursor (cursor += imm 0) so the caller's
            # GPR is left intact for the next replay while the cursor is free to advance each row.
            if gpr_input_addr is not None:
                self.generate_instruction_add_imm(gpr_input_addr, 0, x_reg)
            else:
                self.generate_instruction_add_set(x_reg,   ue_35bit_addr_shifter(input_dram_addr))
            if gpr_cos_addr is not None:
                self.generate_instruction_add_imm(gpr_cos_addr, 0, cos_reg)
            else:
                self.generate_instruction_add_set(cos_reg,  ue_35bit_addr_shifter(cos_dram_addr))
            if gpr_out_addr is not None:
                self.generate_instruction_add_imm(gpr_out_addr, 0, out_reg)
            else:
                self.generate_instruction_add_set(out_reg,  ue_35bit_addr_shifter(output_dram_addr))

            program_dram_start_addr = self.get_program_dram_addr()
            cur_inst_count = self.capture_count
            jump_target_word_addr = ue_35bit_addr_shifter(
                program_dram_start_addr + (cur_inst_count + 1) * INSTRUCTION_SIZE_BYTES
            )
            self.generate_instruction_jump_abs(jump_target_word_addr)
            self.loop_start(loop_cnt=M, gpr_loop_cnt=gpr_M_reg)

            # x_lo from [x_reg], x_hi from [x_reg + half_bytes]
            self.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=sram_x_lo, element_size=half, general_reg_src=x_reg)
            self.generate_instruction_add_imm(x_reg, ue_35bit_addr_shifter(half_bytes), tmp_r)
            self.accelerator_memory_to_sram(accelerator_dram_address=half_bytes, sram_address=sram_x_hi, element_size=half, general_reg_src=tmp_r)

            # cos_lo/cos_hi from [cos_reg+0/half_bytes], sin_lo/sin_hi from [cos_reg+N*bytes_per_elem / +N*bytes_per_elem+half_bytes]
            self.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=sram_cos_lo, element_size=half, general_reg_src=cos_reg)
            self.generate_instruction_add_imm(cos_reg, ue_35bit_addr_shifter(half_bytes),             tmp_r)
            self.accelerator_memory_to_sram(accelerator_dram_address=half_bytes, sram_address=sram_cos_hi, element_size=half, general_reg_src=tmp_r)
            self.generate_instruction_add_imm(cos_reg, ue_35bit_addr_shifter(N * bytes_per_elem),                tmp_r)
            self.accelerator_memory_to_sram(accelerator_dram_address=N * bytes_per_elem, sram_address=sram_sin_lo, element_size=half, general_reg_src=tmp_r)
            self.generate_instruction_add_imm(cos_reg, ue_35bit_addr_shifter(N * bytes_per_elem + half_bytes),   tmp_r)
            self.accelerator_memory_to_sram(accelerator_dram_address=N * bytes_per_elem + half_bytes, sram_address=sram_sin_hi, element_size=half, general_reg_src=tmp_r)

            self.eltwise_mul_core(sram_x_lo, sram_cos_lo, sram_a_lo,  half)
            self.eltwise_mul_core(sram_x_hi, sram_cos_hi, sram_a_hi,  half)
            self.eltwise_mul_core(sram_x_hi, sram_sin_lo, sram_bc_lo, half)
            self.eltwise_mul_core(sram_x_lo, sram_sin_hi, sram_bc_hi, half)
            self.eltwise_add_core(sram_a_lo, sram_bc_lo, sram_result_lo, half)
            self.eltwise_add_core(sram_a_hi, sram_bc_hi, sram_result_hi, half)

            # result_lo to [out_reg], result_hi to [out_reg + half_bytes]
            self.sram_to_accelerator_memory(sram_address=sram_result_lo, accelerator_dram_address=0, element_size=half, general_reg_src=out_reg)
            self.generate_instruction_add_imm(out_reg, ue_35bit_addr_shifter(half_bytes), tmp_r)
            self.sram_to_accelerator_memory(sram_address=sram_result_hi, accelerator_dram_address=half_bytes, element_size=half, general_reg_src=tmp_r)

            # Advance address registers for next row
            self.generate_instruction_add_imm(x_reg,   ue_35bit_addr_shifter(row_bytes),      x_reg)
            self.generate_instruction_add_imm(cos_reg,  ue_35bit_addr_shifter(rope_row_bytes), cos_reg)
            self.generate_instruction_add_imm(out_reg,  ue_35bit_addr_shifter(row_bytes),      out_reg)

            self.loop_end()

            self.release_isa_reg()  # tmp_r
            self.release_isa_reg()  # out_reg
            self.release_isa_reg()  # cos_reg
            self.release_isa_reg()  # x_reg
        else:
            # N >= 128: half >= 64 so half_bytes >= 128 — naturally SRAM-slot-aligned
            sram_x = 0x00000
            sram_a = 0x20000
            sram_d = 0x40000
            sram_cos = 0x80000
            sram_sin = 0x80000 + N * bytes_per_elem
            sram_bc = 0x80000 + N * bytes_per_elem * 2

            assert sin_dram_addr == cos_dram_addr + row_bytes, "PBI RoPE expects contiguous [cos, sin] rows"
            x_uram_type, x_uram_addr = self.sram_address_to_uram_address(sram_x)
            rope_uram_type, rope_uram_addr = self.sram_address_to_uram_address(sram_cos)
            d_uram_type, d_uram_addr = self.sram_address_to_uram_address(sram_d)

            x_ptr = self.alloc_inst_ptr()
            rope_ptr = self.alloc_inst_ptr()
            out_ptr = self.alloc_inst_ptr()

            self.generate_instruction_pbi_init(
                dram_shared_addr=input_dram_addr,
                dma_length=row_bytes,
                output_size=0,
                uram_length=0,
                uram_a_start_addr=0,
                uram_b_start_addr=0,
                uram_wb_addr=0,
                uram_dst_addr=x_uram_addr,
                fmax_context_addr=0,
                inst_pointer_idx=x_ptr,
            )
            self.generate_instruction_pbi_init(
                dram_shared_addr=cos_dram_addr,
                dma_length=rope_row_bytes,
                output_size=0,
                uram_length=0,
                uram_a_start_addr=0,
                uram_b_start_addr=0,
                uram_wb_addr=0,
                uram_dst_addr=rope_uram_addr,
                fmax_context_addr=0,
                inst_pointer_idx=rope_ptr,
            )
            self.generate_instruction_pbi_init(
                dram_shared_addr=output_dram_addr,
                dma_length=row_bytes,
                output_size=0,
                uram_length=0,
                uram_a_start_addr=d_uram_addr,
                uram_b_start_addr=d_uram_addr,
                uram_wb_addr=0,
                uram_dst_addr=0,
                fmax_context_addr=0,
                inst_pointer_idx=out_ptr,
            )

            # Optional: source pointer bases from GPRs (word addr) once before the loop. The rope
            # pointer (cos base) reads [cos, sin] contiguously, so only the cos base is overridden.
            if gpr_input_addr is not None:
                self._pbi_override_dram_base_from_gpr(x_ptr, gpr_input_addr)
            if gpr_cos_addr is not None:
                self._pbi_override_dram_base_from_gpr(rope_ptr, gpr_cos_addr)
            if gpr_out_addr is not None:
                self._pbi_override_dram_base_from_gpr(out_ptr, gpr_out_addr)

            # Absolute jump keeps the ISA loop anchored at the current I-cache window.
            program_dram_start_addr = self.get_program_dram_addr()
            cur_inst_count = self.capture_count
            jump_target_word_addr = ue_35bit_addr_shifter(
                program_dram_start_addr + (cur_inst_count + 1) * INSTRUCTION_SIZE_BYTES
            )
            self.generate_instruction_jump_abs(jump_target_word_addr)
            self.loop_start(loop_cnt=M, gpr_loop_cnt=gpr_M_reg)
            self.accelerator_memory_to_sram(
                accelerator_dram_address=row_bytes,
                sram_address=sram_x,
                element_size=0,
                inst_pointer_idx=x_ptr,
            )
            self.accelerator_memory_to_sram(
                accelerator_dram_address=rope_row_bytes,
                sram_address=sram_cos,
                element_size=0,
                inst_pointer_idx=rope_ptr,
            )
            self.eltwise_mul_core(vector_A_sram_start_addr=sram_x, vector_B_sram_start_addr=sram_cos,              vector_C_sram_wb_addr=sram_a,                   element_size=N)
            self.eltwise_mul_core(vector_A_sram_start_addr=sram_x + half * bytes_per_elem, vector_B_sram_start_addr=sram_sin,              vector_C_sram_wb_addr=sram_bc,                  element_size=half)
            self.eltwise_mul_core(vector_A_sram_start_addr=sram_x, vector_B_sram_start_addr=sram_sin + half*bytes_per_elem,   vector_C_sram_wb_addr=sram_bc + half*bytes_per_elem,       element_size=half)
            self.eltwise_add_core(vector_A_sram_start_addr=sram_a, vector_B_sram_start_addr=sram_bc,               vector_C_sram_wb_addr=sram_d,                   element_size=N)
            self.sram_to_accelerator_memory(
                sram_address=0,
                accelerator_dram_address=row_bytes,
                element_size=0,
                inst_pointer_idx=out_ptr,
            )
            self.loop_end()

            self.release_inst_ptr(out_ptr)
            self.release_inst_ptr(rope_ptr)
            self.release_inst_ptr(x_ptr)
        return 4 * M * N

    def rope_hf_core_dram_dynamic(self, M: int, N: int, input_dram_addr: int, output_dram_addr: int,
                                         cos_dram_addr: int, sin_dram_addr: int,
                                         gpr_M_reg: int = None, gpr_N_reg: int = None,
                                         gpr_input_addr: Optional[int] = None, gpr_out_addr: Optional[int] = None,
                                         gpr_cos_addr: Optional[int] = None) -> int:
        """Fast batched, 4-PBI-pointer HF RoPE with runtime M and N (head_dim>=128).

        Follows the shape of the HW-confirmed :meth:`rms_norm_core_dram_dynamic` and
        :meth:`matmat_mul_core_dynamic`: each eltwise op spans the whole head_dim (or half) in one
        hardware op rather than 64 elements at a time, which requires a PBI pointer (only PBI field
        overrides can set ``URAM_ROW_SIZE`` / ``DMA_LENGTH`` at execute time — the register-addressed
        path can move only the DRAM address, not the length).

        **Four PBI pointers** (matmat-style: A-load, B-load, compute, writeback):

          * ``ptr_x``      — input x DMA (batched: ``rows_take`` rows in one transfer -> URAM_A row 0)
          * ``ptr_rope``   — cos+sin table DMA (batched -> URAM_B row 0); table row is the contiguous
                             ``[cos(N) | sin(N)]`` layout with sin's lower half pre-negated (HW add-only)
          * ``ptr_wb``     — output DMA (batched, URAM_A row 0 -> DRAM)
          * ``ptr_compute``— shared by the four row-spanning eltwise ops per row. The writeback bank
                             (``uram_section``) and mode are per-instruction descriptor fields, not
                             pointer-row fields, so one pointer serves ops writing to different banks;
                             only the URAM addresses / row_size come from the pointer via ``_set``.

        **Per-row compute** (mirrors :meth:`rope_hf_core_dram_legacy`'s 4 ops; sin pre-negated so the
        rotate-half sign is folded into the table):

            a         = x * cos                 (full row  -> URAM_A a-region)
            b[:half]  = x[half:] * sin[:half]    (half row  -> URAM_B b-region)   # = x2 * (-s)
            b[half:]  = x[:half] * sin[half:]    (half row  -> URAM_B b-region)   # = x1 *   s
            out       = a + b                    (full row  -> URAM_A, in place over x)

        Operand A is always read from URAM_A (``URAM_START_ADDR_Y``), operand B from URAM_B
        (``URAM_START_ADDR_Z``), writeback row from ``URAM_WRITEB_ADDR``, span from ``URAM_ROW_SIZE`` —
        all four overridden per op from the register file (RTL: queue_state_module.sv fields 4/5/6/3).

        **Row batching.** DMAing one row at a time pays a fixed per-DMA setup cost on every row;
        batching ``rows_take`` rows into one transfer amortizes it. ``chunk_rows`` is computed at
        runtime from the URAM budget: URAM_A holds x/out + a = ``2*(N/64)`` rows/token, URAM_B holds
        cos+sin + b = ``3*(N/64)`` rows/token, so ``chunk_rows = URAM_NEAR_FULL_ADDR // (3*N/64)``
        (URAM_B binds). The outer while-loop runs ``ceil(M / chunk_rows)`` batches
        (``rows_take = min(remaining, chunk_rows)``), matmat's M-tile idiom.

        ``gpr_M_reg`` (1..15, batch row count) and ``gpr_N_reg`` (1..63, derives N/64, N/128, strides)
        are required and caller-primed. ``M`` / ``N`` are template / FLOPs-accounting only. Optional
        ``gpr_input_addr`` / ``gpr_out_addr`` / ``gpr_cos_addr`` source the input / output / cos DRAM
        bases from GPRs (word addr = ``byte >> 3``); ``None`` -> literal. ``N`` must be a multiple of
        128 (each rotate-half is 64-row aligned); non-64-aligned head_dim is served by host padding
        (see ``_rope_padded_layout`` in user_hw_test.py). Consumes the identical table layout as
        legacy so existing tests pass unchanged.

        **Status: offline-validated only (HW-UNVALIDATED).**
        """
        fn = "rope_hf_core_dram_dynamic"
        if N < 128:
            return self._rope_hf_core_dram_dynamic_small_n(
                M, N, input_dram_addr, output_dram_addr, cos_dram_addr, sin_dram_addr,
                gpr_M_reg=gpr_M_reg, gpr_input_addr=gpr_input_addr,
                gpr_out_addr=gpr_out_addr, gpr_cos_addr=gpr_cos_addr)
        assert M >= 1, f"{fn}: M must be at least 1"
        if gpr_M_reg is None or not (1 <= gpr_M_reg <= 15):
            raise ValueError(f"{fn}: gpr_M_reg must be a GPR index 1..15, got {gpr_M_reg}")
        if gpr_N_reg is None or not (1 <= gpr_N_reg <= 63):
            raise ValueError(f"{fn}: gpr_N_reg must be a GPR index 1..63, got {gpr_N_reg}")
        if gpr_N_reg == gpr_M_reg:
            raise ValueError(f"{fn}: gpr_N_reg={gpr_N_reg} collides with gpr_M_reg={gpr_M_reg}")
        self._validate_addr_gprs(fn, gpr_M_reg, {
            "gpr_input_addr": gpr_input_addr, "gpr_out_addr": gpr_out_addr, "gpr_cos_addr": gpr_cos_addr,
        })
        for _nm, _rg in (("gpr_input_addr", gpr_input_addr), ("gpr_out_addr", gpr_out_addr), ("gpr_cos_addr", gpr_cos_addr)):
            if _rg is not None and _rg == gpr_N_reg:
                raise ValueError(f"{fn}: {_nm}={_rg} collides with gpr_N_reg={gpr_N_reg}")
        assert N >= 128 and N % 128 == 0, (
            f"{fn} handles the N>=128 path with 64-row-aligned halves; N must be a multiple of 128, "
            f"got N={N}. Use the d64 path for head_dim < 128.")
        assert sin_dram_addr == cos_dram_addr + N * 2, "RoPE expects contiguous [cos, sin] rows"
        assert self.is_capture_on, f"{fn}() requires active capture"
        # One token needs 3*(N/64) URAM_B rows (cos+sin + b); a single batch of >=1 row must fit the
        # 12-bit (4095) URAM row-address ceiling. (URAM_A needs only 2*(N/64), so URAM_B always binds.)
        assert 3 * (N // UE_VECTOR_SIZE) <= URAM_NEAR_FULL_ADDR, (
            f"{fn}: N={N} too large — 3*(N/64)={3*(N//UE_VECTOR_SIZE)} rows exceeds "
            f"URAM_NEAR_FULL_ADDR={URAM_NEAR_FULL_ADDR}; one token must fit URAM_B staging.")

        # Fixed SRAM bank anchors (row 0 of each bank). Region *bases* within a bank are runtime
        # (chunk_rows-scaled) so they live in registers, not here.
        SRAM_A = 0x00000   # URAM_A row 0: x / out region base
        SRAM_B = 0x80000   # URAM_B row 0: cos+sin region base
        IN_BASE_W  = ue_35bit_addr_shifter(input_dram_addr)
        COS_BASE_W = ue_35bit_addr_shifter(cos_dram_addr)
        OUT_BASE_W = ue_35bit_addr_shifter(output_dram_addr)

        _alloc_list = []
        def _alloc():
            r = self.alloc_isa_reg(); _alloc_list.append(r); return r

        # --- N-derived counts / strides (words = byte >> 3) ---
        n_rows_reg     = _alloc()  # N/64   (full-row URAM span & row stride)
        half_rows_reg  = _alloc()  # N/128  (rotate-half URAM span & half offset)
        rope_rows_reg  = _alloc()  # N/32   (cos+sin URAM rows/token = 2*n_rows)
        row_words_reg  = _alloc()  # N/4    (x/out per-row DRAM word stride: N*2 bytes >> 3)
        rope_row_words_reg = _alloc()  # N/2 (cos+sin per-row DRAM word stride: 2N*2 bytes >> 3)
        # --- batch bookkeeping ---
        chunk_rows_reg = _alloc()  # rows per DMA batch = URAM_NEAR_FULL_ADDR // (3*n_rows)
        m_counter_reg  = _alloc()  # remaining M rows
        rows_take_reg  = _alloc()  # min(remaining, chunk_rows)
        dma_len_x_reg  = _alloc()  # rows_take*N*2   bytes (x / out batch DMA length)
        dma_len_rope_reg = _alloc()  # rows_take*2N*2 bytes (cos+sin batch DMA length)
        # --- URAM region bases (chunk_rows-scaled; fixed for whole program) ---
        a_base_reg     = _alloc()  # URAM_A a-region base = chunk_rows * n_rows
        b_base_reg     = _alloc()  # URAM_B b-region base = chunk_rows * rope_rows
        # --- running DRAM word cursors (per batch) ---
        x_cur_reg      = _alloc()
        rope_cur_reg   = _alloc()
        out_cur_reg    = _alloc()
        # --- running URAM row cursors (per row, within a batch) ---
        xr_reg         = _alloc()  # x/out row  (URAM_A)
        ar_reg         = _alloc()  # a row      (URAM_A)
        cosr_reg       = _alloc()  # cos row    (URAM_B); sin row = cosr + n_rows
        br_reg         = _alloc()  # b row      (URAM_B)
        # --- scratch ---
        s1 = _alloc()
        s2 = _alloc()

        def _set(ptr, field, reg):
            self.generate_instruction_pbi_inc(general_reg_src=reg, pbi_field_select=field, inst_pointer_idx=ptr)

        def _seed(dst_reg, lit_base_w, gpr_base):
            if gpr_base is not None:
                self.generate_instruction_add_imm(src_reg_idx=gpr_base, immediate_value=0, dst_reg_idx=dst_reg)
            else:
                self.generate_instruction_add_set(dst_reg, lit_base_w)

        # ===== Phase 1: derive N-dependent counts / strides =====
        self.generate_instruction_shr(n_rows_reg, gpr_N_reg, 6)          # N/64
        self.generate_instruction_shr(half_rows_reg, gpr_N_reg, 7)       # N/128
        self.generate_instruction_shr(rope_rows_reg, gpr_N_reg, 5)       # N/32  (= 2*n_rows)
        self.generate_instruction_shr(row_words_reg, gpr_N_reg, 2)       # N*2 bytes >> 3 = N/4 words
        self.generate_instruction_shr(rope_row_words_reg, gpr_N_reg, 1)  # 2N*2 bytes >> 3 = N/2 words

        # chunk_rows = floor(URAM_NEAR_FULL_ADDR / (3*n_rows)). 3*n_rows = n_rows + rope_rows.
        self.generate_instruction_add_reg(s1, n_rows_reg, rope_rows_reg)   # 3*n_rows
        self.generate_instruction_add_set(chunk_rows_reg, URAM_NEAR_FULL_ADDR)
        self.generate_instruction_div_reg(chunk_rows_reg, chunk_rows_reg, s1)

        # URAM region bases (max-batch scaled, like matmat's M_chunk*K_rows writeback base).
        self.generate_instruction_mul32_reg(a_base_reg, chunk_rows_reg, n_rows_reg)
        self.generate_instruction_mul32_reg(b_base_reg, chunk_rows_reg, rope_rows_reg)

        # ===== Phase 2: PBI pointer inits (outside all loops) =====
        ptr_x       = self.alloc_inst_ptr()
        ptr_rope    = self.alloc_inst_ptr()
        ptr_wb      = self.alloc_inst_ptr()
        ptr_compute = self.alloc_inst_ptr()
        # Loads write to bank row 0 (uram_dst); DMA_LENGTH set per batch. Store reads URAM_A row 0.
        self.generate_instruction_pbi_init(dram_shared_addr=input_dram_addr, uram_dst_addr=0, inst_pointer_idx=ptr_x)
        self.generate_instruction_pbi_init(dram_shared_addr=cos_dram_addr, uram_dst_addr=0, inst_pointer_idx=ptr_rope)
        self.generate_instruction_pbi_init(dram_shared_addr=output_dram_addr, uram_a_start_addr=0,
                                           uram_b_start_addr=0, inst_pointer_idx=ptr_wb)
        self.generate_instruction_pbi_init(inst_pointer_idx=ptr_compute)

        # ===== Phase 3: seed per-batch DRAM word cursors (from GPR base when supplied) =====
        _seed(x_cur_reg,   IN_BASE_W,  gpr_input_addr)
        _seed(rope_cur_reg, COS_BASE_W, gpr_cos_addr)
        _seed(out_cur_reg, OUT_BASE_W, gpr_out_addr)

        # ===== Phase 4: outer batch while-loop (4 live PBI pointers) =====
        # Hand-rolled while(m_counter != 0): trip count is a reg_min-clamped remaining-rows counter
        # (matmat / bf16_transpose M-tile idiom), not a monotonic decrement. Abs-jump anchor keeps the
        # backward rel-JNZ within the i-cache window.
        program_dram_start_addr = self.get_program_dram_addr()
        cur_inst_count = self.capture_count
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(
            program_dram_start_addr + (cur_inst_count + 1) * INSTRUCTION_SIZE_BYTES))

        self.generate_instruction_add_imm(src_reg_idx=gpr_M_reg, immediate_value=0, dst_reg_idx=m_counter_reg)
        self.generate_instruction_reg_min(rows_take_reg, m_counter_reg, chunk_rows_reg)
        body_start_inst_cnt = self.capture_count

        # --- batch DMA loads: rows_take rows of x -> URAM_A, rows_take rows of [cos|sin] -> URAM_B ---
        self.generate_instruction_mul32_shl_reg(dma_len_x_reg, rows_take_reg, n_rows_reg, 7)     # rows_take*N*2 bytes
        self.generate_instruction_mul32_shl_reg(dma_len_rope_reg, rows_take_reg, rope_rows_reg, 7)  # rows_take*2N*2 bytes
        _set(ptr_x, PBI_FIELD.DRAM_ADDR, x_cur_reg)
        _set(ptr_x, PBI_FIELD.DMA_LENGTH, dma_len_x_reg)
        self.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=SRAM_A, element_size=0, inst_pointer_idx=ptr_x)
        _set(ptr_rope, PBI_FIELD.DRAM_ADDR, rope_cur_reg)
        _set(ptr_rope, PBI_FIELD.DMA_LENGTH, dma_len_rope_reg)
        self.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=SRAM_B, element_size=0, inst_pointer_idx=ptr_rope)

        # --- inner per-row compute loop (fixed trip count = rows_take for this batch) ---
        self.generate_instruction_add_set(xr_reg, 0)                                  # x/out row 0
        self.generate_instruction_add_imm(src_reg_idx=a_base_reg, immediate_value=0, dst_reg_idx=ar_reg)  # a-region base
        self.generate_instruction_add_set(cosr_reg, 0)                                # cos row 0
        self.generate_instruction_add_imm(src_reg_idx=b_base_reg, immediate_value=0, dst_reg_idx=br_reg)  # b-region base
        self.loop_start(gpr_loop_cnt=rows_take_reg)

        # op1: a = x * cos  (full row -> URAM_A a-region)
        _set(ptr_compute, PBI_FIELD.URAM_ROW_SIZE, n_rows_reg)
        _set(ptr_compute, PBI_FIELD.URAM_START_ADDR_Y, xr_reg)      # A = x
        _set(ptr_compute, PBI_FIELD.URAM_START_ADDR_Z, cosr_reg)    # B = cos
        _set(ptr_compute, PBI_FIELD.URAM_WRITEB_ADDR, ar_reg)       # -> a
        self.start_queue_eltwise(UE_MODE.ELTWISE_MUL, 0, 0, 0, URAM_SECTION.URAM_A.value, 0, inst_pointer_idx=ptr_compute)

        # op2: b[:half] = x[half:] * sin[:half]  (half row -> URAM_B b-region).  sin row = cos + n_rows.
        _set(ptr_compute, PBI_FIELD.URAM_ROW_SIZE, half_rows_reg)
        self.generate_instruction_add_reg(s1, xr_reg, half_rows_reg)   # x2 row = x + half_rows
        _set(ptr_compute, PBI_FIELD.URAM_START_ADDR_Y, s1)
        self.generate_instruction_add_reg(s2, cosr_reg, n_rows_reg)    # sin row = cos + n_rows
        _set(ptr_compute, PBI_FIELD.URAM_START_ADDR_Z, s2)
        _set(ptr_compute, PBI_FIELD.URAM_WRITEB_ADDR, br_reg)          # -> b[:half]
        self.start_queue_eltwise(UE_MODE.ELTWISE_MUL, 0, 0, 0, URAM_SECTION.URAM_B.value, 0, inst_pointer_idx=ptr_compute)

        # op3: b[half:] = x[:half] * sin[half:]  (half row -> URAM_B b-region).  row_size still half.
        _set(ptr_compute, PBI_FIELD.URAM_START_ADDR_Y, xr_reg)         # A = x1 = x
        self.generate_instruction_add_reg(s1, s2, half_rows_reg)       # sin[half:] row = sin + half_rows
        _set(ptr_compute, PBI_FIELD.URAM_START_ADDR_Z, s1)
        self.generate_instruction_add_reg(s2, br_reg, half_rows_reg)   # b[half:] row = b + half_rows
        _set(ptr_compute, PBI_FIELD.URAM_WRITEB_ADDR, s2)
        self.start_queue_eltwise(UE_MODE.ELTWISE_MUL, 0, 0, 0, URAM_SECTION.URAM_B.value, 0, inst_pointer_idx=ptr_compute)

        # op4: out = a + b  (full row -> URAM_A, in place over x).  A = a (URAM_A), B = b (URAM_B).
        _set(ptr_compute, PBI_FIELD.URAM_ROW_SIZE, n_rows_reg)
        _set(ptr_compute, PBI_FIELD.URAM_START_ADDR_Y, ar_reg)         # A = a
        _set(ptr_compute, PBI_FIELD.URAM_START_ADDR_Z, br_reg)         # B = b
        _set(ptr_compute, PBI_FIELD.URAM_WRITEB_ADDR, xr_reg)          # -> out (over x)
        self.start_queue_eltwise(UE_MODE.ELTWISE_ADD, 0, 0, 0, URAM_SECTION.URAM_A.value, 0, inst_pointer_idx=ptr_compute)

        # advance per-row URAM cursors
        self.generate_instruction_add_reg(xr_reg, xr_reg, n_rows_reg)
        self.generate_instruction_add_reg(ar_reg, ar_reg, n_rows_reg)
        self.generate_instruction_add_reg(cosr_reg, cosr_reg, rope_rows_reg)
        self.generate_instruction_add_reg(br_reg, br_reg, n_rows_reg)
        self.loop_end()

        # --- batch DMA store: rows_take rows URAM_A row 0 -> DRAM (dma_len_x unchanged since load) ---
        _set(ptr_wb, PBI_FIELD.DRAM_ADDR, out_cur_reg)
        _set(ptr_wb, PBI_FIELD.DMA_LENGTH, dma_len_x_reg)
        self.sram_to_accelerator_memory(sram_address=SRAM_A, accelerator_dram_address=0, element_size=0, inst_pointer_idx=ptr_wb)

        # advance per-batch DRAM cursors, recompute next batch's row count
        self.generate_instruction_mul32_reg(s1, rows_take_reg, row_words_reg)          # x/out advance words
        self.generate_instruction_add_reg(x_cur_reg, x_cur_reg, s1)
        self.generate_instruction_add_reg(out_cur_reg, out_cur_reg, s1)
        self.generate_instruction_mul32_reg(s2, rows_take_reg, rope_row_words_reg)      # cos+sin advance words
        self.generate_instruction_add_reg(rope_cur_reg, rope_cur_reg, s2)
        self.generate_instruction_reg_sub(m_counter_reg, m_counter_reg, rows_take_reg)
        self.generate_instruction_reg_min(rows_take_reg, m_counter_reg, chunk_rows_reg)

        outer_loop_size = self.capture_count - body_start_inst_cnt + 2
        self.generate_instruction_jump_rela_jnz(outer_loop_size, m_counter_reg)
        print(f"RoPE phased outer loop body size: {outer_loop_size} (M=GPR[{gpr_M_reg}], N=GPR[{gpr_N_reg}], batched, 4 PBI pointers)")
        assert outer_loop_size <= 256, (
            f"{fn}: outer loop body {outer_loop_size} exceeds i-cache budget 256")

        self.release_inst_ptr(ptr_compute)
        self.release_inst_ptr(ptr_wb)
        self.release_inst_ptr(ptr_rope)
        self.release_inst_ptr(ptr_x)
        for _ in range(len(_alloc_list)):
            self.release_isa_reg()

        return 4 * M * N

    def rope_hf_core_dram_d64_pbi(self, M: int, N: int, input_dram_addr: int, output_dram_addr: int, cos_dram_addr: int, sin_dram_addr: int, gpr_M_reg: int = None,
                                  gpr_input_addr: Optional[int] = None, gpr_out_addr: Optional[int] = None, gpr_cos_addr: Optional[int] = None) -> int:
        # DEPRECATED REFERENCE ONLY: deliberately unreachable, even under ``python -O``.
        raise RuntimeError("rope_hf_core_dram_d64_pbi is deprecated; use rope_hf_core_dram_dynamic")
        """PBI HF RoPE for head_dim ``N < 128`` (padded-split). Caller brackets with
        start_capture()/stop_capture().

        For ``N < 128`` each rotate-half operand is N/2 elems = N bytes < 128, i.e. half a
        URAM row, and SRAM cannot be addressed mid-row. We sidestep this by issuing
        **PBI register-addressed DMAs** so each
        N/2-elem slice lands at the start of its own 128-byte-aligned URAM row. Reads AND
        writes are register-addressed, so **no PBI pointers** are used (clear of the
        >=4-advancing-pointer failure mode).

        Table layout is identical to the N>=128 path: contiguous ``[cos(N) || sin(N)]`` rows,
        ``sin_dram_addr == cos_dram_addr + N*2``, sin first half pre-negated on the host. The
        four slices cos_lo/cos_hi/sin_lo/sin_hi are read from byte offsets 0 / N / 2N / 3N of
        each token's rope row.

        **AXI beat alignment:** every slice DMA starts at a multiple of ``half_bytes`` (= N
        bytes), and the DMA engine requires beat-aligned DRAM addresses (the read-side packer
        has no sub-beat lane realignment), so ``N`` bytes must be a multiple of the AXI beat —
        32 B at 256-bit, 64 B at 512-bit. Callers host-pad each rotate-half up to a beat
        multiple when needed (see ``rope_hf_core_dram_d64_test``).

        **Batch dimension / M (dynamic, required):** ``gpr_M_reg`` is a GPR index 1..15 holding
        the runtime token count (caller primes via ``ADD_SET``). ``M`` is FLOPs-accounting /
        loop-template only. Scratch GPRs (address calc + token counter) are allocated internally.

        This is the *native* (no head_dim padding) sub-128 path, exercised compile-time-N by
        ``rope_hf_core_dram_d64_test(..., dynamic=False)``. The dynamic sub-128 case is instead
        served by padding head_dim up to 128 and running :meth:`rope_hf_core_dram_dynamic`
        (``rope_hf_core_dram_d64_test(..., dynamic=True)``).
        """
        assert N % 2 == 0, "N must be even for RoPE half layout"
        assert 0 < N < 128, "rope_hf_core_dram_d64_pbi is the N<128 path; use rope_hf_core_dram_pbi for N>=128"
        if gpr_M_reg is None or not (1 <= gpr_M_reg <= 15):
            raise ValueError(f"rope_hf_core_dram_d64_pbi: gpr_M_reg must be a GPR index 1..15, got {gpr_M_reg}")
        assert sin_dram_addr == cos_dram_addr + N * 2, "d64 RoPE expects contiguous [cos, sin] rows"
        self._validate_addr_gprs("rope_hf_core_dram_d64_pbi", gpr_M_reg, {
            "gpr_input_addr": gpr_input_addr, "gpr_out_addr": gpr_out_addr, "gpr_cos_addr": gpr_cos_addr,
        })

        bpe = 2
        half = N // 2
        half_bytes = half * bpe            # offset of x_hi / sin_hi within their N-elem source
        row_bytes = N * bpe                # x / output row stride
        rope_row_bytes = 2 * row_bytes     # [cos(N) | sin(N)] per token

        ue_assert_axi_beat_aligned_bytes(
            half_bytes,
            f"rope_hf_core_dram_d64_pbi: half_bytes (N={N})",
            hint="slice DMAs must start on an AXI-beat boundary; host-pad each rotate-half "
                 "up to a beat multiple as rope_hf_core_dram_d64_test does")

        # eltwise requires its two operands in DIFFERENT URAM banks.
        # URAM_A: x + cos-products + results; URAM_B: cos/sin slices + sin-products.
        SLOT = 128
        uram_x_lo      = 0x00000           # URAM_A
        uram_x_hi      = 0x00080
        uram_a_lo      = 0x00100
        uram_a_hi      = 0x00180
        uram_result_lo = 0x00200
        uram_result_hi = 0x00280

        uram_cos_lo    = 0x80000           # URAM_B
        uram_cos_hi    = uram_cos_lo + SLOT
        uram_sin_lo    = uram_cos_hi + SLOT
        uram_sin_hi    = uram_sin_lo + SLOT
        uram_b_lo      = uram_sin_hi + SLOT
        uram_b_hi      = uram_b_lo   + SLOT

        shift = ue_35bit_addr_shifter
        tmp_reg = self.alloc_isa_reg()
        t_reg   = self.alloc_isa_reg()

        # Address = (gpr_base or lit_base) + offset_bytes + t*stride. When a base GPR is supplied
        # the base comes from the register (word addr) so one captured program serves any placement;
        # otherwise the literal base+offset is folded into a single add_imm (None-path byte-identical).
        # sin is addressed as cos_base + row_bytes (the sin == cos + row invariant), so no separate
        # sin GPR is needed.
        def _emit_addr(lit_base, offset_bytes, gpr_base, stride):
            self.generate_instruction_reg_mul_imm(tmp_reg, t_reg, shift(stride))
            if gpr_base is not None:
                self.generate_instruction_add_reg(tmp_reg, tmp_reg, gpr_base)
                if offset_bytes:
                    self.generate_instruction_add_imm(tmp_reg, shift(offset_bytes), tmp_reg)
            else:
                self.generate_instruction_add_imm(tmp_reg, shift(lit_base + offset_bytes), tmp_reg)

        def _read(lit_base, offset_bytes, gpr_base, stride, sram_addr):
            _emit_addr(lit_base, offset_bytes, gpr_base, stride)
            self.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=sram_addr,
                                            element_size=half, general_reg_src=tmp_reg)

        def _write(lit_base, offset_bytes, gpr_base, stride, sram_addr):
            _emit_addr(lit_base, offset_bytes, gpr_base, stride)
            self.sram_to_accelerator_memory(sram_address=sram_addr, accelerator_dram_address=0,
                                            element_size=half, general_reg_src=tmp_reg)

        self.generate_instruction_add_set(t_reg, 0)
        self.loop_start(loop_cnt=M, gpr_loop_cnt=gpr_M_reg)

        # Reads — each N/2-elem slice into its own aligned row (DRAM byte offsets are free).
        _read(input_dram_addr, 0,                       gpr_input_addr, row_bytes,      uram_x_lo)   # x_lo
        _read(input_dram_addr, half_bytes,              gpr_input_addr, row_bytes,      uram_x_hi)   # x_hi
        _read(cos_dram_addr,   0,                       gpr_cos_addr,   rope_row_bytes, uram_cos_lo) # cos[0:half]
        _read(cos_dram_addr,   half_bytes,              gpr_cos_addr,   rope_row_bytes, uram_cos_hi) # cos[half:N]
        _read(cos_dram_addr,   row_bytes,               gpr_cos_addr,   rope_row_bytes, uram_sin_lo) # sin[0:half] (pre-negated)
        _read(cos_dram_addr,   row_bytes + half_bytes,  gpr_cos_addr,   rope_row_bytes, uram_sin_hi) # sin[half:N]

        # Compute (operands span URAM_A/URAM_B; element_size=half from row starts).
        self.eltwise_mul_core(uram_x_lo, uram_cos_lo, uram_a_lo, half)      # a_lo = x_lo * cos_lo
        self.eltwise_mul_core(uram_x_hi, uram_cos_hi, uram_a_hi, half)      # a_hi = x_hi * cos_hi
        self.eltwise_mul_core(uram_x_hi, uram_sin_lo, uram_b_lo, half)      # b_lo = x_hi * (-sin)
        self.eltwise_mul_core(uram_x_lo, uram_sin_hi, uram_b_hi, half)      # b_hi = x_lo * sin
        self.eltwise_add_core(uram_a_lo, uram_b_lo, uram_result_lo, half)
        self.eltwise_add_core(uram_a_hi, uram_b_hi, uram_result_hi, half)

        # Writes — two aligned halves back to the output row.
        _write(output_dram_addr, 0,          gpr_out_addr, row_bytes, uram_result_lo)
        _write(output_dram_addr, half_bytes, gpr_out_addr, row_bytes, uram_result_hi)

        self.generate_instruction_add_inc(t_reg)
        self.loop_end()

        self.release_isa_reg()  # t_reg
        self.release_isa_reg()  # tmp_reg
        return 4 * M * N

    def rope_hf_core_dram_gqa(self, M: int, group_size: int, N: int, input_dram_addr: int, output_dram_addr: int, cos_dram_addr: int, sin_dram_addr: int, gpr_M_reg: Optional[int] = None,
                              gpr_input_addr: Optional[int] = None, gpr_out_addr: Optional[int] = None, gpr_cos_addr: Optional[int] = None,
                              gpr_N_reg: Optional[int] = None, gpr_group_reg: Optional[int] = None) -> int:
        """HF GQA RoPE DRAM entrypoint; dispatches based on ``gpr_N_reg`` / ``gpr_M_reg``:

        - Any runtime dimension/address GPR: :meth:`rope_hf_core_dram_gqa_dynamic`. Missing M/N
          registers are allocated and seeded from compile-time values; ``gpr_group_reg`` optionally
          makes group size runtime as well.
        - No runtime GPRs (default): :meth:`rope_hf_core_dram_gqa_legacy` — Python-unrolled rows.

        ``gpr_input_addr`` / ``gpr_out_addr`` / ``gpr_cos_addr`` (dynamic path only) optionally source
        the Q/output/cos-table bases from GPRs (word addr).
        """
        if any(r is not None for r in (gpr_M_reg, gpr_N_reg, gpr_group_reg,
                                       gpr_input_addr, gpr_out_addr, gpr_cos_addr)):
            seeded_regs = []
            if gpr_M_reg is None:
                gpr_M_reg = self.alloc_isa_reg(); seeded_regs.append(gpr_M_reg)
                self.generate_instruction_add_set(gpr_M_reg, M)
            if gpr_N_reg is None:
                gpr_N_reg = self.alloc_isa_reg(); seeded_regs.append(gpr_N_reg)
                self.generate_instruction_add_set(gpr_N_reg, N)
            result = self.rope_hf_core_dram_gqa_dynamic(
                M, group_size, N, input_dram_addr, output_dram_addr, cos_dram_addr, sin_dram_addr,
                gpr_M_reg=gpr_M_reg, gpr_N_reg=gpr_N_reg, gpr_group_reg=gpr_group_reg,
                gpr_input_addr=gpr_input_addr, gpr_out_addr=gpr_out_addr, gpr_cos_addr=gpr_cos_addr)
            for _ in seeded_regs:
                self.release_isa_reg()
            return result
        return self.rope_hf_core_dram_gqa_legacy(M, group_size, N, input_dram_addr, output_dram_addr, cos_dram_addr, sin_dram_addr)

    def rope_hf_core_dram_gqa_legacy(self, M: int, group_size: int, N: int, input_dram_addr: int, output_dram_addr: int, cos_dram_addr: int, sin_dram_addr: int) -> int:
        """Static grouped-query RoPE with dynamic-equivalent table reuse and DMA granularity."""
        assert M >= 1, "M must be at least 1"
        assert group_size >= 1, "group_size must be at least 1"
        assert N % UE_VECTOR_SIZE == 0 and N >= 64, f"N must be a multiple of {UE_VECTOR_SIZE} and >= 64"
        assert N % 2 == 0, "N must be even for RoPE half layout"
        assert N >= 128, "N must be >= 128 so half-vector SRAM offsets are 128-byte aligned"
        half = N // 2
        bytes_per_elem = 2
        row_bytes = N * bytes_per_elem
        rope_row_bytes = 2 * row_bytes
        sram_x = 0x00000
        sram_a = 0x20000
        sram_cos = 0x80000
        sram_sin = 0x80000 + N * bytes_per_elem
        sram_bc = 0x80000 + N * bytes_per_elem * 2
        for row_idx in range(M):
            # Dynamic loads the contiguous [cos|sin] row once and reuses it for every query group.
            self.accelerator_memory_to_sram(accelerator_dram_address=cos_dram_addr + row_idx * rope_row_bytes,
                                            sram_address=sram_cos, element_size=2 * N)
            for group_idx in range(group_size):
                q_row_idx = row_idx * group_size + group_idx
                self.accelerator_memory_to_sram(accelerator_dram_address=input_dram_addr + q_row_idx * row_bytes, sram_address=sram_x, element_size=N)
                self.eltwise_mul_core(vector_A_sram_start_addr=sram_x, vector_B_sram_start_addr=sram_cos, vector_C_sram_wb_addr=sram_a, element_size=N)
                self.eltwise_mul_core(vector_A_sram_start_addr=sram_x + half * bytes_per_elem, vector_B_sram_start_addr=sram_sin, vector_C_sram_wb_addr=sram_bc, element_size=half)
                self.eltwise_mul_core(vector_A_sram_start_addr=sram_x, vector_B_sram_start_addr=sram_sin + half * bytes_per_elem, vector_C_sram_wb_addr=sram_bc + half * bytes_per_elem, element_size=half)
                self.eltwise_add_core(vector_A_sram_start_addr=sram_a, vector_B_sram_start_addr=sram_bc, vector_C_sram_wb_addr=sram_x, element_size=N)
                self.sram_to_accelerator_memory(sram_address=sram_x, accelerator_dram_address=output_dram_addr + q_row_idx * row_bytes, element_size=N)
        return 4 * M * group_size * N

    def rope_hf_core_dram_gqa_pbi(self, M: int, group_size: int, N: int, input_dram_addr: int, output_dram_addr: int, cos_dram_addr: int, sin_dram_addr: int, gpr_M_reg: int = None,
                                  gpr_input_addr: Optional[int] = None, gpr_out_addr: Optional[int] = None, gpr_cos_addr: Optional[int] = None) -> int:
        # DEPRECATED REFERENCE ONLY: deliberately unreachable, even under ``python -O``.
        raise RuntimeError("rope_hf_core_dram_gqa_pbi is deprecated; use rope_hf_core_dram_gqa_dynamic")
        """PBI-backed grouped-query RoPE. Q rows are [M, group_size, N], RoPE rows are [M, N].

        **Batch dimension / M (dynamic, required):**
        ``gpr_M_reg`` is a **required** GPR index 1..15 holding the runtime outer-loop trip count.
        Caller must prime that register beforehand (typically with ``ADD_SET``). The inner
        ``group_size`` loop is still compile-time. ``M`` is FLOPs-accounting only — the captured
        program has no static reference to it.

        ``gpr_input_addr`` / ``gpr_out_addr`` / ``gpr_cos_addr`` optionally source the Q / output /
        cos-table bases from GPRs (word addr); sin stays contiguous after cos. See
        :meth:`rope_hf_core_dram_pbi`.
        """
        assert M >= 1, "M must be at least 1"
        assert group_size >= 1, "group_size must be at least 1"
        assert N % UE_VECTOR_SIZE == 0 and N >= 64, f"N must be a multiple of {UE_VECTOR_SIZE} and >= 64"
        assert N % 2 == 0, "N must be even for RoPE half layout"
        assert N >= 128, "N must be >= 128 so half-vector SRAM offsets are 128-byte aligned"
        if gpr_M_reg is None or not (1 <= gpr_M_reg <= 15):
            raise ValueError(f"rope_hf_core_dram_gqa_pbi: gpr_M_reg must be a GPR index 1..15, got {gpr_M_reg}")
        self._validate_addr_gprs("rope_hf_core_dram_gqa_pbi", gpr_M_reg, {
            "gpr_input_addr": gpr_input_addr, "gpr_out_addr": gpr_out_addr, "gpr_cos_addr": gpr_cos_addr,
        })

        half = N // 2
        bytes_per_elem = 2
        row_bytes = N * bytes_per_elem
        rope_row_bytes = 2 * row_bytes
        sram_x = 0x00000
        sram_a = 0x20000
        sram_d = 0x40000
        sram_cos = 0x80000
        sram_sin = 0x80000 + N * bytes_per_elem
        sram_bc = 0x80000 + N * bytes_per_elem * 2

        assert sin_dram_addr == cos_dram_addr + row_bytes, "PBI GQA RoPE expects contiguous [cos, sin] rows"
        x_uram_type, x_uram_addr = self.sram_address_to_uram_address(sram_x)
        rope_uram_type, rope_uram_addr = self.sram_address_to_uram_address(sram_cos)
        d_uram_type, d_uram_addr = self.sram_address_to_uram_address(sram_d)

        x_ptr = self.alloc_inst_ptr()
        rope_ptr = self.alloc_inst_ptr()
        out_ptr = self.alloc_inst_ptr()

        self.generate_instruction_pbi_init(
            dram_shared_addr=input_dram_addr,
            dma_length=row_bytes,
            output_size=0,
            uram_length=0,
            uram_a_start_addr=0,
            uram_b_start_addr=0,
            uram_wb_addr=0,
            uram_dst_addr=x_uram_addr,
            fmax_context_addr=0,
            inst_pointer_idx=x_ptr,
        )
        self.generate_instruction_pbi_init(
            dram_shared_addr=cos_dram_addr,
            dma_length=rope_row_bytes,
            output_size=0,
            uram_length=0,
            uram_a_start_addr=0,
            uram_b_start_addr=0,
            uram_wb_addr=0,
            uram_dst_addr=rope_uram_addr,
            fmax_context_addr=0,
            inst_pointer_idx=rope_ptr,
        )
        self.generate_instruction_pbi_init(
            dram_shared_addr=output_dram_addr,
            dma_length=row_bytes,
            output_size=0,
            uram_length=0,
            uram_a_start_addr=d_uram_addr,
            uram_b_start_addr=d_uram_addr,
            uram_wb_addr=0,
            uram_dst_addr=0,
            fmax_context_addr=0,
            inst_pointer_idx=out_ptr,
        )

        # Optional: source pointer bases from GPRs (word addr) once before the loop. The rope
        # pointer (cos base) reads [cos, sin] contiguously, so only the cos base is overridden.
        if gpr_input_addr is not None:
            self._pbi_override_dram_base_from_gpr(x_ptr, gpr_input_addr)
        if gpr_cos_addr is not None:
            self._pbi_override_dram_base_from_gpr(rope_ptr, gpr_cos_addr)
        if gpr_out_addr is not None:
            self._pbi_override_dram_base_from_gpr(out_ptr, gpr_out_addr)

        # Absolute jump keeps the ISA loop anchored at the current I-cache window.
        program_dram_start_addr = self.get_program_dram_addr()
        cur_inst_count = self.capture_count
        jump_target_word_addr = ue_35bit_addr_shifter(
            program_dram_start_addr + (cur_inst_count + 1) * INSTRUCTION_SIZE_BYTES
        )
        self.generate_instruction_jump_abs(jump_target_word_addr)
        self.loop_start(loop_cnt=M, gpr_loop_cnt=gpr_M_reg)
        self.accelerator_memory_to_sram(
            accelerator_dram_address=rope_row_bytes,
            sram_address=sram_cos,
            element_size=0,
            inst_pointer_idx=rope_ptr,
        )
        self.loop_start(group_size)
        self.accelerator_memory_to_sram(
            accelerator_dram_address=row_bytes,
            sram_address=sram_x,
            element_size=0,
            inst_pointer_idx=x_ptr,
        )
        self.eltwise_mul_core(vector_A_sram_start_addr=sram_x, vector_B_sram_start_addr=sram_cos, vector_C_sram_wb_addr=sram_a, element_size=N)
        self.eltwise_mul_core(vector_A_sram_start_addr=sram_x + half * bytes_per_elem, vector_B_sram_start_addr=sram_sin, vector_C_sram_wb_addr=sram_bc, element_size=half)
        self.eltwise_mul_core(vector_A_sram_start_addr=sram_x, vector_B_sram_start_addr=sram_sin + half * bytes_per_elem, vector_C_sram_wb_addr=sram_bc + half * bytes_per_elem, element_size=half)
        self.eltwise_add_core(vector_A_sram_start_addr=sram_a, vector_B_sram_start_addr=sram_bc, vector_C_sram_wb_addr=sram_d, element_size=N)
        self.sram_to_accelerator_memory(
            sram_address=0,
            accelerator_dram_address=row_bytes,
            element_size=0,
            inst_pointer_idx=out_ptr,
        )
        self.loop_end()
        self.loop_end()

        self.release_inst_ptr(out_ptr)
        self.release_inst_ptr(rope_ptr)
        self.release_inst_ptr(x_ptr)
        return 4 * M * group_size * N

    def rope_hf_core_dram_gqa_dynamic(self, M: int, group_size: int, N: int, input_dram_addr: int, output_dram_addr: int,
                                             cos_dram_addr: int, sin_dram_addr: int,
                                             gpr_M_reg: int = None, gpr_N_reg: int = None, gpr_group_reg: Optional[int] = None,
                                             gpr_input_addr: Optional[int] = None, gpr_out_addr: Optional[int] = None,
                                             gpr_cos_addr: Optional[int] = None) -> int:
        """Fast 4-PBI-pointer grouped-query HF RoPE with runtime M / group_size / N (head_dim>=128).

        The GQA analog of :meth:`rope_hf_core_dram_dynamic`. Q rows are
        ``[M, group_size, N]``, RoPE rows ``[M, N]``; one cos/sin row is loaded once per group and
        shared by all ``group_size`` Q rows (the group loop re-reads it from URAM_B, saving
        ``group_size-1`` rope loads/token vs a per-Q-row reload). Each Q row is roped with full-row
        runtime-length eltwise ops (one op spans the whole head_dim / half), not 64 elements at a time.

        **Four PBI pointers**: ``ptr_rope`` (cos+sin load, once per outer M iter -> URAM_B row 0),
        ``ptr_x`` (Q-row load, per group member -> URAM_A row 0), ``ptr_compute`` (shared by the 4
        eltwise ops), ``ptr_wb`` (out store, per group member). Because one Q row is processed at a
        time, every URAM address is a loop-invariant register (row 0 / n_rows / half etc.), not an
        advancing cursor — simpler than the batched non-GQA core.

        Per Q row (identical math to legacy; ``[cos|sin]`` table with sin lower half pre-negated):
        ``a = x*cos`` (full->URAM_A), ``b[:h]=x[h:]*sin[:h]`` + ``b[h:]=x[:h]*sin[h:]`` (half->URAM_B),
        ``out = a+b`` (full->URAM_A in place). URAM_A: x/out@0, a@n_rows; URAM_B: cos@0, sin@n_rows,
        b@2*n_rows -> URAM_B needs 3*(N/64) rows (must fit the 4095 row ceiling).

        ``gpr_M_reg`` (1..15) and ``gpr_N_reg`` (1..63) required; ``gpr_group_reg`` (1..15) optional
        (runtime group_size; ``None`` -> compile-time ``group_size``). Optional ``gpr_input_addr`` /
        ``gpr_out_addr`` / ``gpr_cos_addr`` source DRAM bases from GPRs (word addr = ``byte >> 3``).
        N>=128, multiple of 128; non-64-aligned head_dim via host padding, same as the non-GQA core.
        Caller brackets with start_capture() and primes the supplied registers.

        **Status: offline-validated only (HW-UNVALIDATED)**.
        """
        fn = "rope_hf_core_dram_gqa_dynamic"
        assert M >= 1 and group_size >= 1, f"{fn}: M and group_size must be at least 1"
        if gpr_M_reg is None or not (1 <= gpr_M_reg <= 15):
            raise ValueError(f"{fn}: gpr_M_reg must be a GPR index 1..15, got {gpr_M_reg}")
        if gpr_N_reg is None or not (1 <= gpr_N_reg <= 63):
            raise ValueError(f"{fn}: gpr_N_reg must be a GPR index 1..63, got {gpr_N_reg}")
        if gpr_group_reg is not None and not (1 <= gpr_group_reg <= 15):
            raise ValueError(f"{fn}: gpr_group_reg must be a GPR index 1..15 or None, got {gpr_group_reg}")
        _reg_set = [r for r in (gpr_M_reg, gpr_N_reg, gpr_group_reg) if r is not None]
        if len(set(_reg_set)) != len(_reg_set):
            raise ValueError(f"{fn}: gpr_M_reg/gpr_N_reg/gpr_group_reg must be distinct, got {_reg_set}")
        self._validate_addr_gprs(fn, gpr_M_reg, {
            "gpr_input_addr": gpr_input_addr, "gpr_out_addr": gpr_out_addr, "gpr_cos_addr": gpr_cos_addr,
        })
        for _nm, _rg in (("gpr_input_addr", gpr_input_addr), ("gpr_out_addr", gpr_out_addr), ("gpr_cos_addr", gpr_cos_addr)):
            if _rg is not None and _rg in (gpr_N_reg, gpr_group_reg):
                raise ValueError(f"{fn}: {_nm}={_rg} collides with gpr_N_reg/gpr_group_reg")
        assert N >= 128 and N % 128 == 0, (
            f"{fn} handles the N>=128 path with 64-row-aligned halves; N must be a multiple of 128, got N={N}.")
        assert sin_dram_addr == cos_dram_addr + N * 2, "GQA RoPE expects contiguous [cos, sin] rows"
        assert self.is_capture_on, f"{fn}() requires active capture"
        assert 3 * (N // UE_VECTOR_SIZE) <= URAM_NEAR_FULL_ADDR, (
            f"{fn}: N={N} too large — 3*(N/64)={3*(N//UE_VECTOR_SIZE)} rows exceeds "
            f"URAM_NEAR_FULL_ADDR={URAM_NEAR_FULL_ADDR}.")

        SRAM_A = 0x00000   # URAM_A row 0: x / out
        SRAM_B = 0x80000   # URAM_B row 0: cos (sin @ n_rows, b @ 2*n_rows)
        IN_BASE_W  = ue_35bit_addr_shifter(input_dram_addr)
        COS_BASE_W = ue_35bit_addr_shifter(cos_dram_addr)
        OUT_BASE_W = ue_35bit_addr_shifter(output_dram_addr)

        _alloc_list = []
        def _alloc():
            r = self.alloc_isa_reg(); _alloc_list.append(r); return r

        n_rows_reg     = _alloc()  # N/64  (a-region base in URAM_A; sin base in URAM_B; full row_size)
        half_rows_reg  = _alloc()  # N/128 (x2 row; rotate-half span)
        rope_rows_reg  = _alloc()  # N/32  (b-region base in URAM_B = 2*n_rows)
        row_words_reg  = _alloc()  # N/4   (x/out per-Q-row DRAM word stride)
        rope_row_words_reg = _alloc()  # N/2 (cos+sin per-outer-row DRAM word stride)
        x_row_bytes_reg  = _alloc()  # N*2   (x/out DMA length)
        rope_row_bytes_reg = _alloc()  # 2N*2 (cos+sin DMA length)
        sin_hi_row_reg = _alloc()  # n_rows + half_rows (sin[half:] URAM row)
        b_hi_row_reg   = _alloc()  # rope_rows + half_rows (b[half:] URAM row)
        za_reg         = _alloc()  # constant 0 (x/cos/out row 0; avoids relying on hard reg0 as PBI src)
        x_q_reg        = _alloc()  # Q-input DRAM word cursor (per group member; contiguous across M)
        out_q_reg      = _alloc()  # output DRAM word cursor
        cos_row_reg    = _alloc()  # cos/sin DRAM word cursor (per outer M iter)

        def _set(ptr, field, reg):
            self.generate_instruction_pbi_inc(general_reg_src=reg, pbi_field_select=field, inst_pointer_idx=ptr)

        def _seed(dst_reg, lit_base_w, gpr_base):
            if gpr_base is not None:
                self.generate_instruction_add_imm(src_reg_idx=gpr_base, immediate_value=0, dst_reg_idx=dst_reg)
            else:
                self.generate_instruction_add_set(dst_reg, lit_base_w)

        # ===== Phase 1: derive N-dependent counts / strides =====
        self.generate_instruction_shr(n_rows_reg, gpr_N_reg, 6)          # N/64
        self.generate_instruction_shr(half_rows_reg, gpr_N_reg, 7)       # N/128
        self.generate_instruction_shr(rope_rows_reg, gpr_N_reg, 5)       # N/32
        self.generate_instruction_shr(row_words_reg, gpr_N_reg, 2)       # N/4 words
        self.generate_instruction_shr(rope_row_words_reg, gpr_N_reg, 1)  # N/2 words
        self.generate_instruction_shl(x_row_bytes_reg, gpr_N_reg, 1)     # N*2 bytes
        self.generate_instruction_shl(rope_row_bytes_reg, gpr_N_reg, 2)  # 2N*2 bytes
        self.generate_instruction_add_reg(sin_hi_row_reg, n_rows_reg, half_rows_reg)     # n_rows + half
        self.generate_instruction_add_reg(b_hi_row_reg, rope_rows_reg, half_rows_reg)    # 2*n_rows + half
        self.generate_instruction_add_set(za_reg, 0)

        # ===== Phase 2: PBI pointer inits (outside all loops) =====
        ptr_rope    = self.alloc_inst_ptr()
        ptr_x       = self.alloc_inst_ptr()
        ptr_wb      = self.alloc_inst_ptr()
        ptr_compute = self.alloc_inst_ptr()
        self.generate_instruction_pbi_init(dram_shared_addr=cos_dram_addr, uram_dst_addr=0, inst_pointer_idx=ptr_rope)
        self.generate_instruction_pbi_init(dram_shared_addr=input_dram_addr, uram_dst_addr=0, inst_pointer_idx=ptr_x)
        self.generate_instruction_pbi_init(dram_shared_addr=output_dram_addr, uram_a_start_addr=0,
                                           uram_b_start_addr=0, inst_pointer_idx=ptr_wb)
        self.generate_instruction_pbi_init(inst_pointer_idx=ptr_compute)
        # DMA lengths are constant (one row / one rope row) -> set once here; only DRAM_ADDR moves.
        _set(ptr_rope, PBI_FIELD.DMA_LENGTH, rope_row_bytes_reg)
        _set(ptr_x, PBI_FIELD.DMA_LENGTH, x_row_bytes_reg)
        _set(ptr_wb, PBI_FIELD.DMA_LENGTH, x_row_bytes_reg)

        # ===== Phase 3: seed DRAM word cursors =====
        _seed(x_q_reg,    IN_BASE_W,  gpr_input_addr)
        _seed(cos_row_reg, COS_BASE_W, gpr_cos_addr)
        _seed(out_q_reg,  OUT_BASE_W, gpr_out_addr)

        def _compute_one_qrow():
            # 4 full-row/half runtime-length eltwise ops on the shared compute_ptr (URAM addrs fixed:
            # x@0, a@n_rows, cos@0, sin@n_rows, b@2*n_rows). Same op sequence as the non-GQA core.
            # op1: a = x*cos (full -> URAM_A a-region)
            _set(ptr_compute, PBI_FIELD.URAM_ROW_SIZE, n_rows_reg)
            _set(ptr_compute, PBI_FIELD.URAM_START_ADDR_Y, za_reg)         # A = x @ 0
            _set(ptr_compute, PBI_FIELD.URAM_START_ADDR_Z, za_reg)         # B = cos @ 0
            _set(ptr_compute, PBI_FIELD.URAM_WRITEB_ADDR, n_rows_reg)      # -> a @ n_rows
            self.start_queue_eltwise(UE_MODE.ELTWISE_MUL, 0, 0, 0, URAM_SECTION.URAM_A.value, 0, inst_pointer_idx=ptr_compute)
            # op2: b[:half] = x[half:] * sin[:half] (half -> URAM_B b-region). sin @ n_rows, b @ 2*n_rows.
            _set(ptr_compute, PBI_FIELD.URAM_ROW_SIZE, half_rows_reg)
            _set(ptr_compute, PBI_FIELD.URAM_START_ADDR_Y, half_rows_reg)  # A = x2 @ half_rows
            _set(ptr_compute, PBI_FIELD.URAM_START_ADDR_Z, n_rows_reg)     # B = sin_lo @ n_rows
            _set(ptr_compute, PBI_FIELD.URAM_WRITEB_ADDR, rope_rows_reg)   # -> b_lo @ 2*n_rows
            self.start_queue_eltwise(UE_MODE.ELTWISE_MUL, 0, 0, 0, URAM_SECTION.URAM_B.value, 0, inst_pointer_idx=ptr_compute)
            # op3: b[half:] = x[:half] * sin[half:] (half -> URAM_B). row_size stays half.
            _set(ptr_compute, PBI_FIELD.URAM_START_ADDR_Y, za_reg)         # A = x1 @ 0
            _set(ptr_compute, PBI_FIELD.URAM_START_ADDR_Z, sin_hi_row_reg) # B = sin_hi @ n_rows+half
            _set(ptr_compute, PBI_FIELD.URAM_WRITEB_ADDR, b_hi_row_reg)    # -> b_hi @ 2*n_rows+half
            self.start_queue_eltwise(UE_MODE.ELTWISE_MUL, 0, 0, 0, URAM_SECTION.URAM_B.value, 0, inst_pointer_idx=ptr_compute)
            # op4: out = a + b (full -> URAM_A in place over x). A = a @ n_rows, B = b @ 2*n_rows.
            _set(ptr_compute, PBI_FIELD.URAM_ROW_SIZE, n_rows_reg)
            _set(ptr_compute, PBI_FIELD.URAM_START_ADDR_Y, n_rows_reg)     # A = a
            _set(ptr_compute, PBI_FIELD.URAM_START_ADDR_Z, rope_rows_reg)  # B = b
            _set(ptr_compute, PBI_FIELD.URAM_WRITEB_ADDR, za_reg)          # -> out @ 0
            self.start_queue_eltwise(UE_MODE.ELTWISE_ADD, 0, 0, 0, URAM_SECTION.URAM_A.value, 0, inst_pointer_idx=ptr_compute)

        # ===== Phase 4: outer M loop / inner group loop (4 live PBI pointers) =====
        program_dram_start_addr = self.get_program_dram_addr()
        cur_inst_count = self.capture_count
        self.generate_instruction_jump_abs(ue_35bit_addr_shifter(
            program_dram_start_addr + (cur_inst_count + 1) * INSTRUCTION_SIZE_BYTES))
        self.loop_start(loop_cnt=M, gpr_loop_cnt=gpr_M_reg)

        # Load this token's cos+sin once -> URAM_B row 0 (shared by all group members).
        _set(ptr_rope, PBI_FIELD.DRAM_ADDR, cos_row_reg)
        self.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=SRAM_B, element_size=0, inst_pointer_idx=ptr_rope)

        # Inner group loop: rope each of the group_size Q rows (all share URAM_B cos/sin).
        if gpr_group_reg is not None:
            self.loop_start(loop_cnt=group_size, gpr_loop_cnt=gpr_group_reg)
        else:
            self.loop_start(loop_cnt=group_size)
        _set(ptr_x, PBI_FIELD.DRAM_ADDR, x_q_reg)
        self.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=SRAM_A, element_size=0, inst_pointer_idx=ptr_x)
        _compute_one_qrow()
        _set(ptr_wb, PBI_FIELD.DRAM_ADDR, out_q_reg)
        self.sram_to_accelerator_memory(sram_address=SRAM_A, accelerator_dram_address=0, element_size=0, inst_pointer_idx=ptr_wb)
        # Q-input / output cursors advance one row per group member (contiguous across the outer M).
        self.generate_instruction_add_reg(x_q_reg, x_q_reg, row_words_reg)
        self.generate_instruction_add_reg(out_q_reg, out_q_reg, row_words_reg)
        inner_loop_size = self.loop_end()

        # cos/sin cursor advances one rope row per outer iteration.
        self.generate_instruction_add_reg(cos_row_reg, cos_row_reg, rope_row_words_reg)
        outer_loop_size = self.loop_end()

        print(f"RoPE GQA phased loop sizes: outer={outer_loop_size}, inner={inner_loop_size} "
              f"(M=GPR[{gpr_M_reg}], group={'GPR[%d]' % gpr_group_reg if gpr_group_reg else group_size}, "
              f"N=GPR[{gpr_N_reg}], 4 PBI pointers)")
        assert outer_loop_size <= 256 and inner_loop_size <= 256, (
            f"{fn}: loop body sizes outer={outer_loop_size}, inner={inner_loop_size} exceed i-cache 256")

        self.release_inst_ptr(ptr_compute)
        self.release_inst_ptr(ptr_wb)
        self.release_inst_ptr(ptr_x)
        self.release_inst_ptr(ptr_rope)
        for _ in range(len(_alloc_list)):
            self.release_isa_reg()

        return 4 * M * group_size * N

    def start_queue_for_bf16_dequantize_operation(self, VECTOR_INPUT_DRAM_ADDR: int, SCALE_INPUT_DRAM_ADDR: int, data_type: TYPE,
                                                  output_sram_wb_addr: int, element_size: int) -> None:
        """
        Start queue for bf16 dequantize operation.
        Args:
            VECTOR_INPUT_DRAM_ADDR: DRAM address of input matrix
            data_type: data_type
            output_sram_wb_addr: SRAM address for output matrix
            element_size: number of elements in the matrix
        """

        output_uram_type, output_uram_start_addr = self.sram_address_to_uram_address(output_sram_wb_addr)
        row_size = (element_size + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE # round up to the nearest multiple of UE_VECTOR_SIZE - K is inner dimension of the matrix

        assert data_type in (TYPE.IF4, TYPE.IF8, TYPE.TQ4), f"data_type={data_type} must be one of TYPE.IF4, TYPE.IF8, TYPE.TQ4"

        if data_type == TYPE.IF4 or data_type == TYPE.TQ4:
            dma_length = element_size >> 1
        elif data_type == TYPE.IF8:
            dma_length = element_size

        self.accelerator_memory_to_scale_sram(accelerator_dram_address=SCALE_INPUT_DRAM_ADDR, element_size=row_size)

        self.ue_arithmetic_op(
            0,  # broadcast_mode
            0,  # clear_max_en
            1,  # stride_z
            0,  # lalu_a
            0,  # lalu_b
            LALU_MODE.BYPASS.value,  # lalu_mode
            0,  # lalu_scalar
            output_uram_type.value,  # uram_section
            0,  # uram_dst_addr
            output_uram_start_addr,  # uram_wb_addr
            URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
            UE_MODE.DEQUANTIZE,  # mode
            data_type.value,  # data_type (TYPE.IF4, TYPE.IF8)
            0,  # uram_a_start_addr
            0,  # uram_b_start_addr
            row_size,  # bram_length
            VECTOR_INPUT_DRAM_ADDR,  # dma_start_addr
            dma_length,  # dma_length
            row_size,  # output_size
        )

        return element_size # total flops is element_size

    def start_queue_for_quantize_operation(self, input_sram_addr: int, output_sram_addr: int,
                                               data_type: TYPE, element_size: int) -> int:
        """
        Start queue for bf16 quantize operation. Input data must already be in SRAM.

        Args:
            input_sram_addr: SRAM address of input bf16 vector
            output_sram_addr: SRAM address for quantized output
            data_type: quantization type. The on-chip quantize core only
                produces the FP4 variant of TYPE.IF4.
            element_size: number of elements to quantize
        """
        input_uram_type, input_uram_start_addr = self.sram_address_to_uram_address(input_sram_addr)
        assert input_uram_type == URAM_SECTION.URAM_A, f"input_sram_addr must be in URAM_A hex(input_uram_start_addr)={hex(input_uram_start_addr)}"
        output_uram_type, output_uram_start_addr = self.sram_address_to_uram_address(output_sram_addr)

        row_size = (element_size + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE

        assert data_type == TYPE.IF4, f"data_type={data_type} must be TYPE.IF4 (FP4 variant)"

        self.ue_arithmetic_op(
            0,  # broadcast_mode
            0,  # clear_max_en
            1,  # stride_z
            0,  # lalu_a
            0,  # lalu_b
            LALU_MODE.MODE_RECIP.value,  # lalu_mode
            0,  # lalu_scalar not used for quantize
            output_uram_type.value,  # uram_section
            0,  # uram_dst_addr
            output_uram_start_addr,  # uram_wb_addr
            URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
            UE_MODE.QUANTIZE,  # mode
            data_type.value,  # quantization_type
            input_uram_start_addr,  # uram_a_start_addr
            0,  # uram_b_start_addr
            row_size,  # bram_length
            0,  # dma_start_addr (N/A)
            0,  # dma_length (N/A)
            0,  # output_size (N/A)
        )

        return 2 * element_size # 2 FLOPS per element

    def prepare_tq4_codebook_dram(self, codebook_bf16: torch.Tensor) -> int:
        """Stage a 16-entry TQ4 codebook in params DRAM.

        The hardware auto-loads the lowest 256 bits of URAM-B[0] into the TQ4
        codebook register on the rising edge of every DEQUANTIZE / DOT_PRODUCT
        dma_start. Pad the 16 bf16 entries (32 bytes) up to a full 128-byte
        URAM row so a single 64-element bf16 transfer copies them into place.
        """
        assert codebook_bf16.dtype == torch.bfloat16, "codebook must be bf16"
        assert codebook_bf16.numel() == 16, "TQ4 codebook must contain exactly 16 entries"
        padded = torch.zeros(UE_VECTOR_SIZE, dtype=torch.bfloat16)
        padded[:16] = codebook_bf16.flatten()
        codebook_dram_addr = self.get_params_dram_addr()
        self.dma_write(DMA_DEVICE_H2C, codebook_dram_addr,
                       padded.view(torch.uint16), UE_VECTOR_SIZE * 2)
        self.allocate_params_dram(UE_VECTOR_SIZE * 2)
        return codebook_dram_addr

    def load_tq4_codebook(self, codebook_dram_addr: int) -> None:
        """Emit the captured DMA that copies the staged codebook into URAM-B[0].

        Must be called inside ``start_capture()`` before the first TQ4
        DEQUANTIZE / DOT_PRODUCT op so URAM-B[0] is primed when dma_start
        rises and the on-chip auto-load FSM latches the codebook register.
        """
        URAM_B_BASE_SRAM_ADDR = 0x80000
        self.accelerator_memory_to_sram(
            accelerator_dram_address=codebook_dram_addr,
            sram_address=URAM_B_BASE_SRAM_ADDR,
            element_size=UE_VECTOR_SIZE,
        )

    def matmat_mul_core(self, M: int, K: int, N: int, A_DRAM_ADDR: int, B_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, softmax_enable: bool = False, C_DRAM_ADDR: int = None, bias_mode: str = "broadcast_N",
                            is_B_quantized: bool = False, data_type: TYPE = None, SCALE_DRAM_ADDR: int = None, gelu_enable: bool = False, silu_enable: bool = False, sigmoid_enable: bool = False,
                            clamp_enable: bool = False, log_enable: bool = False,
                            clamp_min: float = 0.0, clamp_max: float = float("inf"),
                            debug_fmax: bool = False, ZERO_DRAM_ADDR: int = None, FMAX_DRAM_ADDR: int = None,
                            write_back_disable: bool = False, gpr_M_reg: int = None,
                            gpr_K_reg: int = None, gpr_N_reg: int = None,
                            gpr_a_addr: Optional[int] = None, gpr_b_addr: Optional[int] = None,
                            gpr_out_addr: Optional[int] = None, gpr_c_addr: Optional[int] = None,
                            gpr_scale_addr: Optional[int] = None) -> None:
        """Matrix multiply entrypoint; dispatches based on which dimensions are runtime registers:

        - any of ``gpr_M_reg`` / ``gpr_K_reg`` / ``gpr_N_reg`` provided: :meth:`matmat_mul_core_dynamic` —
          fully dynamic path. Missing dimension GPRs are auto-allocated, seeded from the compile-time
          scalar, and released after the call.
        - none provided (default): :meth:`matmat_mul_core_legacy` — compile-time M/K/N tiling.

        ``gpr_a_addr`` / ``gpr_b_addr`` / ``gpr_out_addr`` / ``gpr_c_addr`` / ``gpr_scale_addr``
        (dynamic path only) optionally source the A/B/output/bias/scale DRAM bases from GPRs
        (word addr) — see :meth:`matmat_mul_core_dynamic`.

        **Layout:** ``A`` is **M×K** (row-major). ``B`` is **N×K** (row-major); the accelerator uses ``B`` as above and
        applies an implicit transpose so the computed result is **A @ Bᵀ**, i.e. **M×N**, without a separate transpose pass.
        """
        _addr_gprs = (gpr_a_addr, gpr_b_addr, gpr_out_addr, gpr_c_addr, gpr_scale_addr)
        if gpr_M_reg is not None or gpr_K_reg is not None or gpr_N_reg is not None:
            allocated = []
            if gpr_K_reg is None:
                gpr_K_reg = self.alloc_isa_reg()
                self.generate_instruction_add_set(gpr_K_reg, K)
                allocated.append('K')
            if gpr_N_reg is None:
                gpr_N_reg = self.alloc_isa_reg()
                self.generate_instruction_add_set(gpr_N_reg, N)
                allocated.append('N')
            if gpr_M_reg is None:
                gpr_M_reg = self.alloc_isa_reg()
                self.generate_instruction_add_set(gpr_M_reg, M)
                allocated.append('M')
            flops = self.matmat_mul_core_dynamic(
                M, K, N, A_DRAM_ADDR, B_DRAM_ADDR, OUTPUT_DRAM_ADDR, softmax_enable, C_DRAM_ADDR, bias_mode,
                is_B_quantized, data_type, SCALE_DRAM_ADDR, gelu_enable, silu_enable, sigmoid_enable,
                clamp_enable, log_enable,
                clamp_min=clamp_min, clamp_max=clamp_max,
                write_back_disable=write_back_disable,
                gpr_M_reg=gpr_M_reg, gpr_K_reg=gpr_K_reg, gpr_N_reg=gpr_N_reg,
                gpr_a_addr=gpr_a_addr, gpr_b_addr=gpr_b_addr, gpr_out_addr=gpr_out_addr,
                gpr_c_addr=gpr_c_addr, gpr_scale_addr=gpr_scale_addr,
            )
            for _ in allocated:
                self.release_isa_reg()
            return flops
        if any(r is not None for r in _addr_gprs):
            raise ValueError("matmat_mul_core: gpr_*_addr require a dimension GPR (set gpr_M_reg, gpr_K_reg, or gpr_N_reg)")
        return self.matmat_mul_core_legacy(
            M, K, N, A_DRAM_ADDR, B_DRAM_ADDR, OUTPUT_DRAM_ADDR, softmax_enable, C_DRAM_ADDR, bias_mode,
            is_B_quantized, data_type, SCALE_DRAM_ADDR, gelu_enable, silu_enable, sigmoid_enable,
            clamp_enable, log_enable,
            clamp_min=clamp_min, clamp_max=clamp_max,
            debug_fmax=debug_fmax, ZERO_DRAM_ADDR=ZERO_DRAM_ADDR, FMAX_DRAM_ADDR=FMAX_DRAM_ADDR,
            write_back_disable=write_back_disable,
        )

    def matmat_mul_core_legacy(self, M: int, K: int, N: int, A_DRAM_ADDR: int, B_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, softmax_enable: bool = False, C_DRAM_ADDR: int = None, bias_mode: str = "broadcast_N",
                             is_B_quantized: bool = False, data_type: TYPE = None, SCALE_DRAM_ADDR: int = None, gelu_enable: bool = False, silu_enable: bool = False, sigmoid_enable: bool = False,
                             clamp_enable: bool = False, log_enable: bool = False,
                             clamp_min: float = 0.0, clamp_max: float = float("inf"),
                             debug_fmax: bool = False, ZERO_DRAM_ADDR: int = None, FMAX_DRAM_ADDR: int = None,
                             write_back_disable: bool = False) -> None:
        # Requirements: Based on these conditions M_chunk x K + M_chunk x N_chunk should fit in URAM_A and N_chunk x K should fit in URAM_B
        # 1. M_chunk can be any value between 1 and M
        # 2. N_chunk needs to be a multiple of UE_VECTOR_SIZE
        # 3. M_chunk * K + N_chunk * M_chunk must be less than or equal to URAM_FULL_ELEMENTS

        bytes_per_element = 2
        bias_enable = C_DRAM_ADDR is not None

        if bias_enable:
            assert bias_mode in ("broadcast_N", "full_matrix"), f"bias_mode={bias_mode} must be either 'broadcast_N' or 'full_matrix'"

        if is_B_quantized:
            assert data_type in (TYPE.IF4, TYPE.IF8), f"data_type={data_type} must be one of TYPE.IF4, TYPE.IF8"

        assert sum([gelu_enable, silu_enable, sigmoid_enable, clamp_enable, log_enable]) <= 1, "only one activation can be True"

        if softmax_enable:
            if debug_fmax:
                assert ZERO_DRAM_ADDR is not None, "ZERO_DRAM_ADDR must be provided when debug_fmax=True"
                assert FMAX_DRAM_ADDR is not None, "FMAX_DRAM_ADDR must be provided when debug_fmax=True"

        lalu_mode = LALU_MODE.BYPASS
        lalu_a = 0
        lalu_b = 0
        if gelu_enable:
            lalu_mode = LALU_MODE.ACT
            lalu_a = LALU_ACT_GELU_A
            lalu_b = LALU_ACT_GELU_B
        elif silu_enable:
            lalu_mode = LALU_MODE.ACT
            lalu_a = LALU_ACT_SILU_A
            lalu_b = LALU_ACT_SILU_B
        elif sigmoid_enable:
            lalu_mode = LALU_MODE.ACT_NO_X
            lalu_a = LALU_ACT_SIGMOID_A
            lalu_b = LALU_ACT_SIGMOID_B
        elif clamp_enable:
            lalu_mode = LALU_MODE.CLAMP
            lalu_a = self.float_to_bf16(clamp_min)
            lalu_b = self.float_to_bf16(clamp_max)
        elif log_enable:
            lalu_mode = LALU_MODE.LOG
            lalu_a = LALU_LOG_A
            lalu_b = LALU_LOG_B

        # Calculate N_chunk
        N_chunk = min(N, (URAM_NEAR_FULL_ELEMENTS // K) // UE_VECTOR_SIZE * UE_VECTOR_SIZE)
        N_chunk_aligned = None
        if N_chunk < UE_VECTOR_SIZE:
            # Try 32(2 bursts) and 16(1 burst)
            if (K * 32) <= URAM_NEAR_FULL_ELEMENTS:
                N_chunk = 32
            elif (K * 16) <= URAM_NEAR_FULL_ELEMENTS:
                N_chunk = 16
            else:
                assert False, f"N={N} is too large to fit in URAM_NEAR_FULL_ELEMENTS={URAM_NEAR_FULL_ELEMENTS}"
            N_chunk_aligned = UE_VECTOR_SIZE

        assert N_chunk <= N, f"N_chunk={N_chunk} must be less than or equal to N={N}"

        if N_chunk_aligned is None:
            # Calculate M_chunk
            if softmax_enable:
                M_chunk = min(UE_FMAX_CONTEXT_SIZE, M, URAM_FULL_ELEMENTS // (K + N_chunk)) # fmax_context is UE_FMAX_CONTEXT_SIZE elements
            else:
                M_chunk = min(M, URAM_FULL_ELEMENTS // (K + N_chunk))
        else:
            # Calculate M_chunk
            if softmax_enable:
                M_chunk = min(UE_FMAX_CONTEXT_SIZE, M, URAM_FULL_ELEMENTS // (K + N_chunk_aligned)) # fmax_context is UE_FMAX_CONTEXT_SIZE elements
            else:
                M_chunk = min(M, URAM_FULL_ELEMENTS // (K + N_chunk_aligned))

        assert M_chunk >= 1 and M_chunk <= M, f"M_chunk={M_chunk} must be greater than 0 and less than M={M}"

        print(f"M_chunk: {M_chunk}, N_chunk: {N_chunk}", f"N_chunk_aligned: {N_chunk_aligned}")

        if N_chunk_aligned is None:
            print(f"URAM_A usage: {100 * (M_chunk * K + M_chunk * N_chunk) / URAM_FULL_ELEMENTS:.2f}% of URAM_FULL_ELEMENTS")
            print(f"URAM_B usage: {100 * N_chunk * K / URAM_FULL_ELEMENTS:.2f}% of URAM_FULL_ELEMENTS")
        else:
            print(f"URAM_A usage: {100 * (M_chunk * K + M_chunk * N_chunk_aligned) / URAM_FULL_ELEMENTS:.2f}% of URAM_FULL_ELEMENTS")
            print(f"URAM_B usage: {100 * N_chunk * K / URAM_FULL_ELEMENTS:.2f}% of URAM_FULL_ELEMENTS")

        for i, m_take in self.chunk_ranges(M, M_chunk):
            self.accelerator_memory_to_sram(accelerator_dram_address=A_DRAM_ADDR + i * K * bytes_per_element,
                                            sram_address=0x00000,
                                            element_size=m_take * K)

            output_sram_wb_addr = 0x00000 + m_take * K * bytes_per_element
            assert output_sram_wb_addr < 0x80000, f"output_sram_wb_addr={output_sram_wb_addr} is greater than or equal to 0x80000, which is the size of URAM_B"

            clear_en = 1
            for j, n_take in self.chunk_ranges(N, N_chunk):
                if is_B_quantized:
                    if data_type == TYPE.IF4:
                        offset = j * K >> 1
                    elif data_type == TYPE.IF8:
                        offset = j * K
                    self.start_queue_for_bf16_dequantize_operation(VECTOR_INPUT_DRAM_ADDR=B_DRAM_ADDR + offset,
                                                                  SCALE_INPUT_DRAM_ADDR=SCALE_DRAM_ADDR + ((j * K) // UE_VECTOR_SIZE) * bytes_per_element,
                                                                  data_type=data_type,
                                                                  output_sram_wb_addr=0x80000,
                                                                  element_size=n_take * K)
                else:
                    self.accelerator_memory_to_sram(accelerator_dram_address=B_DRAM_ADDR + j * K * bytes_per_element,
                                                sram_address=0x80000,
                                                element_size=n_take * K)

                if bias_enable and bias_mode == "broadcast_N":
                    self.accelerator_memory_to_bias_sram(accelerator_dram_address=C_DRAM_ADDR + j * bytes_per_element, element_size=n_take)

                for output_row in range(m_take):

                    if bias_enable and bias_mode == "full_matrix":
                        self.accelerator_memory_to_bias_sram(accelerator_dram_address=C_DRAM_ADDR + ((i + output_row) * N + j) * bytes_per_element,
                                                             element_size=n_take)

                    in_sram_offset = output_row * K * bytes_per_element
                    if N_chunk_aligned is None:
                        out_sram_offset = output_row * n_take * bytes_per_element
                    else:
                        out_sram_offset = output_row * N_chunk_aligned * bytes_per_element

                    self.start_queue_for_bf16_matvec_operation(max_clear_en=clear_en,
                                                            fmax_context_addr=output_row,
                                                            vector_sram_start_addr=0x00000 + in_sram_offset,
                                                            matrix_sram_start_addr=0x80000,
                                                            output_sram_wb_addr=output_sram_wb_addr + out_sram_offset,
                                                            K=K,
                                                            N=n_take,
                                                            bias_enable=bias_enable,
                                                            lalu_mode=lalu_mode,
                                                            lalu_a=lalu_a,
                                                            lalu_b=lalu_b)
                    clear_en = 0

                # generated matrix is m_take x n_take
                start_dram_address_of_partial_matrix = OUTPUT_DRAM_ADDR + (i * N + j) * bytes_per_element

                # # LEGACY NOTE: This is the legacy way of copying m_take x n_take matrix to DRAM, check below for new way of copying with stride
                # for output_row in range(m_take):
                #     self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr + output_row * n_take * bytes_per_element, # every row is n_take elements
                #                                 accelerator_dram_address=start_dram_address_of_partial_matrix + output_row * N * bytes_per_element,
                #                                 element_size=n_take)

                # New way of copying with stride
                if not write_back_disable:
                    if N_chunk_aligned is None:
                        self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr,
                                                    accelerator_dram_address=start_dram_address_of_partial_matrix,
                                                    element_size=m_take * n_take,
                                                    stride_bytes_per_chunk=n_take * bytes_per_element,
                                                    stride_jump_bytes=N * bytes_per_element)
                    else: # this means output from matrix-vector operation is non-aligned - less than UE_VECTOR_SIZE elements
                        for o_row_idx in range(m_take):
                            self.ue_memcpy_to_dram(
                                memcpy_type=MEMCPY_TYPE.URAM.value,
                                uram_type=URAM_SECTION.URAM_A.value,
                                uram_src_addr=(m_take * K * bytes_per_element >> 7) + o_row_idx,
                                dram_dst_addr=OUTPUT_DRAM_ADDR + (i * N + j) * bytes_per_element + o_row_idx * N * bytes_per_element,
                                memcpy_length_bytes=n_take * 2,
                            )

            if softmax_enable:
                m_take_chunk_size = min(URAM_NEAR_FULL_ELEMENTS // N, m_take)
                assert m_take_chunk_size >= 1 and m_take_chunk_size <= m_take, f"m_take_chunk_size={m_take_chunk_size} must be greater than 0 and less than m_take={m_take}"

                for m_take_chunk_idx, m_take_chunk in self.chunk_ranges(m_take, m_take_chunk_size):
                    start_dram_address_of_partial_row_complete_matrix = OUTPUT_DRAM_ADDR + (i + m_take_chunk_idx) * N * bytes_per_element

                    input_sram_start_addr = 0x00000 # URAM_A start
                    self.accelerator_memory_to_sram(accelerator_dram_address=start_dram_address_of_partial_row_complete_matrix,
                                                sram_address=input_sram_start_addr,
                                                element_size=m_take_chunk * N)

                    output_sram_wb_addr = 0x80000 # URAM_B start
                    for row_idx in range(m_take_chunk):
                        if debug_fmax:
                            zero_sram_addr = input_sram_start_addr + m_take_chunk * N * bytes_per_element
                            self.accelerator_memory_to_sram(accelerator_dram_address=ZERO_DRAM_ADDR,
                                                            sram_address=zero_sram_addr,
                                                            element_size=UE_VECTOR_SIZE)

                            self.fmax_core(vector_sram_start_addr=zero_sram_addr,
                                        output_sram_wb_addr=zero_sram_addr,
                                        N=UE_VECTOR_SIZE,
                                        fmax_context_addr=row_idx + m_take_chunk_idx)
                            self.sram_to_accelerator_memory(sram_address=zero_sram_addr,
                                                            accelerator_dram_address=FMAX_DRAM_ADDR + (i + row_idx + m_take_chunk_idx) * UE_VECTOR_SIZE * bytes_per_element,
                                                            element_size=UE_VECTOR_SIZE)

                        self.start_queue_for_bf16_softmax_operation(fmax_context_addr=row_idx + m_take_chunk_idx,
                                                                vector_sram_start_addr=input_sram_start_addr + row_idx * N * bytes_per_element,
                                                                output_sram_wb_addr=output_sram_wb_addr + row_idx * N * bytes_per_element,
                                                                N=N)

                    self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr,
                                                accelerator_dram_address=start_dram_address_of_partial_row_complete_matrix,
                                                element_size=m_take_chunk * N)

        # Total Theoretical FLOPS: 2 * M * K * N + M * N * 5 (softmax)
        total_flops = 2 * M * K * N
        if softmax_enable:
            total_flops += M * N * 5
        if bias_enable:
            total_flops += M * N
        if gelu_enable or silu_enable:
            total_flops += 4 * M * N
        if clamp_enable:
            total_flops += M * N  # 1 compare + clamp per element
        if log_enable:
            total_flops += 2 * M * N  # clamp + log per element
        print(f"Total Theoretical FLOPS: {total_flops / 1e9:.6f} G")
        return total_flops

    def matmat_mul_dynamic_core(self, *args, **kwargs):
        """Deprecated compatibility alias for :meth:`matmat_mul_core_dynamic`."""
        import warnings

        warnings.warn(
            "matmat_mul_dynamic_core() is deprecated; use matmat_mul_core_dynamic()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.matmat_mul_core_dynamic(*args, **kwargs)

    def matmat_mul_core_dynamic(self, M: int, K: int, N: int, A_DRAM_ADDR: int, B_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                                softmax_enable: bool = False, C_DRAM_ADDR: int = None, bias_mode: str = "broadcast_N",
                                is_B_quantized: bool = False, data_type: TYPE = None, SCALE_DRAM_ADDR: int = None,
                                gelu_enable: bool = False, silu_enable: bool = False, sigmoid_enable: bool = False,
                                clamp_enable: bool = False, log_enable: bool = False,
                                clamp_min: float = 0.0, clamp_max: float = float("inf"),
                                write_back_disable: bool = False,
                                gpr_M_reg: int = None, gpr_K_reg: int = None, gpr_N_reg: int = None,
                                gpr_a_addr: Optional[int] = None, gpr_b_addr: Optional[int] = None,
                                gpr_out_addr: Optional[int] = None, gpr_c_addr: Optional[int] = None,
                                gpr_scale_addr: Optional[int] = None) -> int:
        """
        Fully dynamic M/K/N matmul captured as an ISA program (A @ Bᵀ -> M×N).

        This kernel computes every tiling constant in the ISA register file at startup using the integer
        ALU ops (``div_reg``, ``mul32_reg``, ``shr``, ``shl``, ``reg_min``) so K and N can be
        runtime registers. The software ``for i,j in chunk_ranges`` tile loops become ISA
        while-loops (``reg_sub`` + ``JNZ``); per-row URAM/DRAM cursors are tracked in registers
        because every PBI auto-advance delta (``K/64``, ``ceil(n_take/64)``, ``N*2``) is runtime.

        Args:
            M, K, N: Compile-time values used for FLOPS accounting and as fallback initial
                values when the matching ``gpr_*_reg`` is ``None``.
            gpr_M_reg / gpr_K_reg / gpr_N_reg: optional ISA register indices already holding the
                runtime M / K / N (caller primes them via ``ADD_SET`` before replay). When a GPR
                is ``None``, the corresponding dimension is seeded from the compile-time literal.
            gpr_a_addr / gpr_b_addr / gpr_out_addr / gpr_c_addr / gpr_scale_addr: optional ISA
                register indices holding the A / B / output / bias-C / scale DRAM **base** as a
                **word** address (``byte >> 3``). When given, the running cursor / per-tile address
                is sourced from the GPR instead of the compile-time literal, so one captured program
                serves any DRAM placement (re-prime the GPRs before each replay). ``gpr_c_addr``
                requires a bias (``C_DRAM_ADDR``); ``gpr_scale_addr`` requires ``is_B_quantized``.
                ``None`` -> literal (byte-for-byte identical program).

        Currently implements: LALU activations + softmax + broadcast_N bias + full_matrix bias
        (including softmax + full_matrix), and IF4/IF8 quantized-B (scale-BRAM + DEQUANTIZE).
        Loop-body optimizations per optimize_dynamic.md:
        running A/B DRAM cursors (opts 1–2), rows_done + m_tile_rows + n_row_words (opts 3–4).
        Strided writeback (opt 5) deferred — requires N_chunk>=64 dense layout match to legacy.
        """
        bytes_per_element = 2
        bias_enable = C_DRAM_ADDR is not None

        if is_B_quantized:
            assert data_type in (TYPE.IF4, TYPE.IF8), f"matmat_mul_core_dynamic quantized-B: data_type must be IF4 or IF8, got {data_type}"
            assert SCALE_DRAM_ADDR is not None, "matmat_mul_core_dynamic: SCALE_DRAM_ADDR required when is_B_quantized=True"
        if bias_enable:
            assert bias_mode in ("broadcast_N", "full_matrix"), (
                f"bias_mode={bias_mode} must be 'broadcast_N' or 'full_matrix'"
            )
        assert K % UE_VECTOR_SIZE == 0, f"K={K} must be a multiple of UE_VECTOR_SIZE={UE_VECTOR_SIZE}"
        assert sum([gelu_enable, silu_enable, sigmoid_enable, clamp_enable, log_enable]) <= 1, "only one activation can be True"
        if softmax_enable:
            assert N % UE_VECTOR_SIZE == 0, f"softmax requires N % {UE_VECTOR_SIZE} == 0, got N={N}"

        lalu_mode = LALU_MODE.BYPASS
        lalu_a = 0
        lalu_b = 0
        if gelu_enable:
            lalu_mode, lalu_a, lalu_b = LALU_MODE.ACT, LALU_ACT_GELU_A, LALU_ACT_GELU_B
        elif silu_enable:
            lalu_mode, lalu_a, lalu_b = LALU_MODE.ACT, LALU_ACT_SILU_A, LALU_ACT_SILU_B
        elif sigmoid_enable:
            lalu_mode, lalu_a, lalu_b = LALU_MODE.ACT_NO_X, LALU_ACT_SIGMOID_A, LALU_ACT_SIGMOID_B
        elif clamp_enable:
            lalu_mode, lalu_a, lalu_b = LALU_MODE.CLAMP, self.float_to_bf16(clamp_min), self.float_to_bf16(clamp_max)
        elif log_enable:
            lalu_mode, lalu_a, lalu_b = LALU_MODE.LOG, LALU_LOG_A, LALU_LOG_B

        assert (gpr_M_reg is not None and gpr_K_reg is not None and gpr_N_reg is not None), "Dynamic requires gpr inputs of m n k dims!"

        # Optional GPR-sourced DRAM bases (word addr = byte>>3). A/B/scale seed their running cursors
        # from the GPR; output/bias bases are register-added to the per-tile offset. None -> literal
        # (byte-for-byte identical program). The GPRs are caller-owned for the whole program, so they
        # must be distinct from the M/K/N dim GPRs and each other.
        self._validate_addr_gprs("matmat_mul_core_dynamic", gpr_M_reg, {
            "gpr_a_addr": gpr_a_addr, "gpr_b_addr": gpr_b_addr, "gpr_out_addr": gpr_out_addr,
            "gpr_c_addr": gpr_c_addr, "gpr_scale_addr": gpr_scale_addr,
        })
        for _nm, _rg in (("gpr_a_addr", gpr_a_addr), ("gpr_b_addr", gpr_b_addr), ("gpr_out_addr", gpr_out_addr),
                         ("gpr_c_addr", gpr_c_addr), ("gpr_scale_addr", gpr_scale_addr)):
            if _rg is not None and _rg in (gpr_K_reg, gpr_N_reg):
                raise ValueError(f"matmat_mul_core_dynamic: {_nm}={_rg} collides with gpr_K_reg/gpr_N_reg")
        if gpr_c_addr is not None and not bias_enable:
            raise ValueError("matmat_mul_core_dynamic: gpr_c_addr given but no bias (C_DRAM_ADDR is None)")
        if gpr_scale_addr is not None and not is_B_quantized:
            raise ValueError("matmat_mul_core_dynamic: gpr_scale_addr given but is_B_quantized=False")

        def _seed_cursor(cursor_reg, lit_base_w, gpr_base):
            # cursor = GPR base (copy, preserving the caller's reg) or the compile-time literal.
            if gpr_base is not None:
                self.generate_instruction_add_imm(src_reg_idx=gpr_base, immediate_value=0, dst_reg_idx=cursor_reg)
            else:
                self.generate_instruction_add_set(cursor_reg, lit_base_w)

        def _off_plus_base(off_reg, lit_base_w, gpr_base, dst_reg):
            # dst = off_reg + (GPR base or literal base). off_reg holds the per-tile word offset.
            if gpr_base is not None:
                self.generate_instruction_add_reg(dst_reg, off_reg, gpr_base)
            else:
                self.generate_instruction_add_imm(src_reg_idx=off_reg, immediate_value=lit_base_w, dst_reg_idx=dst_reg)

        # Strided writeback: one DMA per (M-tile, N-strip) whose DRAM row stride (N*2) is sourced from
        # a GPR at runtime (RTL stride-jump override), replacing the m_take per-row DMA loop. Requires
        # every n_take to be a multiple of 64 so the m_take output rows are contiguous in URAM (the
        # strided DMA reads URAM contiguously and writes DRAM in strided rows). That holds on the main
        # column-strip path; the sub-64 fallback (compile-time K large enough that a 64-wide strip
        # overflows the eff_z field) produces non-64 strips, so it keeps the per-row writeback. The
        # path is fixed by the compile-time K regime, so this is a compile-time choice (no mid-loop
        # branch). N must be a multiple of 64 (the runtime N must match this regime).
        _kr_ct = K // UE_VECTOR_SIZE
        _main_nchunk_ct = min(
            (URAM_NEAR_FULL_ELEMENTS // K) // UE_VECTOR_SIZE * UE_VECTOR_SIZE,
            (4095 // _kr_ct) * UE_VECTOR_SIZE,
        )
        use_strided_wb = (not write_back_disable) and (N % UE_VECTOR_SIZE == 0) and (_main_nchunk_ct >= UE_VECTOR_SIZE)

        URAM_B_ROW0 = (0x80000 >> 7) & 0xFFF  # URAM_B base row index (== 0)
        A_BASE_W = A_DRAM_ADDR >> 3            # DRAM word addresses (byte >> 3, PBI DRAM_ADDR format)
        B_BASE_W = B_DRAM_ADDR >> 3
        OUT_BASE_W = OUTPUT_DRAM_ADDR >> 3
        C_BASE_W = (C_DRAM_ADDR >> 3) if bias_enable else 0   # broadcast_N bias vector base
        SCALE_BASE_W = (SCALE_DRAM_ADDR >> 3) if is_B_quantized else 0

        # ----------------------------------------------------------------------
        # Register allocation. The ISA register file holds 63 GPRs (1..63); callers keep
        # gpr_M/K/N allocated for the whole program. Running DRAM cursors (A/B/output) and
        # precomputed strip strides avoid recomputing addresses each tile/strip/row (see
        # optimize_dynamic.md opts 1–4). softmax row_idx and full_matrix N/4 row stride
        # get dedicated registers so both bias modes work with softmax.
        # N and M totals are aliased onto the caller's gpr regs when provided (read-only).
        # ----------------------------------------------------------------------
        _alloc_list = []
        def _alloc():
            r = self.alloc_isa_reg()
            _alloc_list.append(r)
            return r

        K_rows_reg    = _alloc()  # K // 64 (URAM row count for one A/B vector)
        N_chunk_reg   = _alloc()  # column strip width (elements)
        M_chunk_reg   = _alloc()  # rows per M tile (URAM cap)
        gpr_M_counter = _alloc()  # remaining M rows
        m_take_reg    = _alloc()  # min(remaining, M_chunk); reused as inner loop counter
        N_counter_reg = _alloc()  # remaining N columns
        n_take_reg    = _alloc()  # min(remaining, N_chunk)
        s1 = _alloc()             # scratch / cursor (reused per phase)
        s2 = _alloc()             # scratch / cursor
        s3 = _alloc()             # scratch / increment
        s4 = _alloc()             # n_take_rows / N_chunks increment (survives a phase)
        a_dram_reg    = _alloc()  # running A tile DRAM word cursor
        b_dram_reg    = _alloc()  # running B strip DRAM word cursor (BF16 or quantized B data)
        scale_dram_reg = _alloc() if is_B_quantized else None  # running scale DRAM word cursor
        rows_done_reg = _alloc()  # M rows completed before current M-tile
        m_tile_rows_reg = _alloc()  # m_take for current M-tile (survives dot/wb inner loops)
        k_strip_word_stride_reg = _alloc()  # K_rows << 4 (B/A word advance per N/A row)
        n_row_words_reg = _alloc()  # N >> 2 (output row word stride; also full_matrix bias)
        n_stride_bytes_reg = _alloc() if use_strided_wb else None  # N*2 (DRAM row stride for strided wb)
        bias_full_row_reg = _alloc() if (bias_enable and bias_mode == "full_matrix") else None
        row_idx_reg = _alloc() if softmax_enable else None
        N_reg = gpr_N_reg        # caller-owned read-only alias
        M_total_reg = gpr_M_reg  # caller-owned read-only alias

        program_dram_start_addr = self.get_program_dram_addr()

        # ===== Phase 1: startup tiling arithmetic =============================
        self.generate_instruction_shr(K_rows_reg, gpr_K_reg, 6)         # K_rows = K >> 6

        # s1 = K (elements) = K_rows << 6
        self.generate_instruction_shl(s1, K_rows_reg, 6)

        # ---- runtime no-chunk fast path (skips the 3 startup divides for small problems) ----
        # When the whole problem fits one URAM tile + one column strip, the column-strip search
        # (two 32-cycle div_reg) and the M_chunk divide are pure overhead: N_chunk would clamp to N
        # and M_chunk would exceed M (single tile). A cheap feasibility test (mul + min + sub, no
        # divide) detects that case at runtime and a forward conditional jump skips the divide block,
        # leaving N_chunk = N and M_chunk = M (identical single-tile execution to the divide path).
        # Restricted to plain matmul: softmax needs the fmax-row cap and quantized-B a different B
        # layout, so those always take the divide path (this branch is not emitted for them).
        nochunk_eligible = (not softmax_enable) and (not is_B_quantized)
        nochunk_patch = None
        if nochunk_eligible:
            # n_ceil = ceil(N/64)*64 (URAM output rows are 64-padded); total_A = M*(K + n_ceil).
            self.generate_instruction_add_imm(src_reg_idx=N_reg, immediate_value=63, dst_reg_idx=s2)
            self.generate_instruction_shr(s2, s2, 6)
            self.generate_instruction_shl(s2, s2, 6)                       # n_ceil
            self.generate_instruction_add_reg(s2, s1, s2)                  # K + n_ceil
            self.generate_instruction_mul32_reg(s2, M_total_reg, s2)       # total_A = M*(K+n_ceil)
            self.generate_instruction_add_set(s3, URAM_FULL_ELEMENTS)
            self.generate_instruction_reg_min(s4, s2, s3)                  # min(total_A, URAM_FULL)
            self.generate_instruction_reg_sub(s2, s2, s4)                  # over_A (>0 iff A tile too big)
            self.generate_instruction_mul32_reg(s3, N_reg, K_rows_reg)     # eff_z need = N*K_rows
            self.generate_instruction_add_set(s4, 4095)
            self.generate_instruction_reg_min(s4, s3, s4)
            self.generate_instruction_reg_sub(s3, s3, s4)                  # over_Z (>0 iff eff_z field overflows)
            self.generate_instruction_add_reg(s2, s2, s3)                  # needs_chunk = over_A + over_Z (0 => fits)
            # optimistic single-tile values (overwritten by the divide block when needs_chunk != 0)
            self.generate_instruction_add_imm(src_reg_idx=N_reg, immediate_value=0, dst_reg_idx=N_chunk_reg)
            self.generate_instruction_add_imm(src_reg_idx=M_total_reg, immediate_value=0, dst_reg_idx=M_chunk_reg)
            nochunk_patch = self._emit_forward_skip_jz(s2, program_dram_start_addr)

        # N_chunk = floor(URAM_NEAR_FULL / K) rounded down to a multiple of 64,
        # then capped so N_chunk * K_rows <= 4095 (URAM_ROW_SIZE_Z is a 12-bit field).
        # hw_max = floor(4095 / K_rows) aligned down to 64.
        self.generate_instruction_add_set(s2, URAM_NEAR_FULL_ELEMENTS)
        self.generate_instruction_div_reg(s2, s2, s1)          # floor(URAM_NEAR_FULL / K)
        self.generate_instruction_shr(N_chunk_reg, s2, 6)
        self.generate_instruction_shl(N_chunk_reg, N_chunk_reg, 6)
        self.generate_instruction_add_set(s2, 4095)
        self.generate_instruction_div_reg(s2, s2, K_rows_reg)  # floor(4095 / K_rows)
        self.generate_instruction_shr(s2, s2, 6)
        self.generate_instruction_shl(s2, s2, 6)               # hw_max (64-aligned)
        self.generate_instruction_reg_min(N_chunk_reg, N_chunk_reg, s2)
        # If N_chunk == 0 (K too large for a 64-wide strip to fit the HW limit):
        # sub-64 fallback: use up to 32 columns but aligned to 16 (= 32 bytes = 1 AXI beat)
        # so the output DRAM address (cols_done >> 2 words) stays word-aligned every iteration.
        cur = self.capture_count
        skip_target = ue_35bit_addr_shifter(program_dram_start_addr + (cur + 7) * INSTRUCTION_SIZE_BYTES)
        self.generate_instruction_jump_abs_jnz(skip_target, N_chunk_reg)
        self.generate_instruction_add_set(s2, 4095)
        self.generate_instruction_div_reg(s2, s2, K_rows_reg)  # floor(4095 / K_rows)
        self.generate_instruction_add_set(N_chunk_reg, 32)
        self.generate_instruction_reg_min(N_chunk_reg, N_chunk_reg, s2)  # min(32, hw_limit)
        self.generate_instruction_shr(N_chunk_reg, N_chunk_reg, 4)       # align down to multiple of 16
        self.generate_instruction_shl(N_chunk_reg, N_chunk_reg, 4)
        # cap at N (shared between main path and sub-64 fallback)
        self.generate_instruction_reg_min(N_chunk_reg, N_reg, N_chunk_reg)

        # M_chunk = URAM_FULL // (K + N_chunk)
        self.generate_instruction_add_reg(s2, s1, N_chunk_reg)  # s2 = K + N_chunk
        self.generate_instruction_add_set(s3, URAM_FULL_ELEMENTS)
        self.generate_instruction_div_reg(M_chunk_reg, s3, s2)
        if softmax_enable:
            # cap by fmax-table rows and by one-softmax-slab-fits-URAM (URAM_NEAR_FULL // N)
            self.generate_instruction_add_set(s2, UE_FMAX_CONTEXT_SIZE)
            self.generate_instruction_reg_min(M_chunk_reg, M_chunk_reg, s2)
            self.generate_instruction_add_set(s2, URAM_NEAR_FULL_ELEMENTS)
            self.generate_instruction_div_reg(s2, s2, N_reg)
            self.generate_instruction_reg_min(M_chunk_reg, M_chunk_reg, s2)

        # No-chunk fast path rejoins here (its forward JZ skipped the divide block above).
        if nochunk_patch is not None:
            nochunk_patch()

        # Seed M counter and first m_take.
        self.generate_instruction_add_imm(src_reg_idx=M_total_reg, immediate_value=0, dst_reg_idx=gpr_M_counter)
        self.generate_instruction_reg_min(m_take_reg, gpr_M_counter, M_chunk_reg)

        # Precompute strip/row strides used throughout the loop body.
        self.generate_instruction_shl(k_strip_word_stride_reg, K_rows_reg, 4)  # K/4 words per matrix row
        self.generate_instruction_shr(n_row_words_reg, N_reg, 2)               # N/4 words per output row
        if use_strided_wb:
            self.generate_instruction_shl(n_stride_bytes_reg, N_reg, 1)        # N*2 bytes (strided wb DRAM row stride)

        # Running DRAM cursors (word addresses) — seeded from GPR base or literal.
        _seed_cursor(a_dram_reg, A_BASE_W, gpr_a_addr)
        _seed_cursor(b_dram_reg, B_BASE_W, gpr_b_addr)
        if is_B_quantized:
            _seed_cursor(scale_dram_reg, SCALE_BASE_W, gpr_scale_addr)

        # ===== Phase 2: PBI pointer inits (outside all loops) ================
        ptr_A   = self.alloc_inst_ptr()
        ptr_dot = self.alloc_inst_ptr()
        ptr_wb  = self.alloc_inst_ptr()
        self.generate_instruction_pbi_init(dram_shared_addr=A_DRAM_ADDR, inst_pointer_idx=ptr_A)
        if is_B_quantized:
            ptr_scale   = self.alloc_inst_ptr()
            ptr_B_quant = self.alloc_inst_ptr()
            self.generate_instruction_pbi_init(dram_shared_addr=SCALE_DRAM_ADDR, inst_pointer_idx=ptr_scale)
            self.generate_instruction_pbi_init(dram_shared_addr=B_DRAM_ADDR, uram_wb_addr=URAM_B_ROW0, inst_pointer_idx=ptr_B_quant)
        else:
            ptr_B = self.alloc_inst_ptr()
            self.generate_instruction_pbi_init(dram_shared_addr=B_DRAM_ADDR, uram_b_start_addr=URAM_B_ROW0, inst_pointer_idx=ptr_B)
        self.generate_instruction_pbi_init(dram_shared_addr=0, inst_pointer_idx=ptr_dot)
        self.generate_instruction_pbi_init(dram_shared_addr=OUTPUT_DRAM_ADDR, inst_pointer_idx=ptr_wb)
        if use_strided_wb:
            # Field 11 (stride_jump) = N*2 bytes — set once here; used by the strided writeback DMA.
            self.generate_instruction_pbi_inc(general_reg_src=n_stride_bytes_reg, pbi_field_select=PBI_FIELD.STRIDE_JUMP, inst_pointer_idx=ptr_wb)
        ptr_bias = None
        ptr_bias_full = None
        if bias_enable:
            if bias_mode == "broadcast_N":
                ptr_bias = self.alloc_inst_ptr()
                self.generate_instruction_pbi_init(dram_shared_addr=C_DRAM_ADDR, inst_pointer_idx=ptr_bias)
            else:  # full_matrix: per-row load inside the dot loop
                ptr_bias_full = self.alloc_inst_ptr()
                self.generate_instruction_pbi_init(dram_shared_addr=C_DRAM_ADDR, inst_pointer_idx=ptr_bias_full)
        if softmax_enable:
            ptr_sm_reload = self.alloc_inst_ptr()
            ptr_sm_exp    = self.alloc_inst_ptr()
            ptr_sm_mul    = self.alloc_inst_ptr()
            ptr_sm_wb     = self.alloc_inst_ptr()
            self.generate_instruction_pbi_init(dram_shared_addr=OUTPUT_DRAM_ADDR, inst_pointer_idx=ptr_sm_reload)
            self.generate_instruction_pbi_init(
                dram_shared_addr=0, lalu_scalar=self.float_to_bf19(1.0),
                inst_pointer_idx=ptr_sm_exp)
            self.generate_instruction_pbi_init(
                dram_shared_addr=0, uram_wb_addr=URAM_B_ROW0,
                lalu_scalar=self.float_to_bf19(1.0), inst_pointer_idx=ptr_sm_mul)
            self.generate_instruction_pbi_init(dram_shared_addr=OUTPUT_DRAM_ADDR, inst_pointer_idx=ptr_sm_wb)

        # ===== Phase 3: M while-loop body ===================================
        # Absolute jump re-anchors the i-cache so all relative backward jumps in the
        # loop body are within 512 slots of a known unconditional anchor.
        _prog_base = self.get_program_dram_addr()
        self.generate_instruction_jump_abs(
            ue_35bit_addr_shifter(_prog_base + (self.capture_count + 1) * INSTRUCTION_SIZE_BYTES)
        )
        body_start_inst_cnt = self.capture_count

        # ---- rows_done and m_take for this M-tile ----
        self.generate_instruction_reg_sub(rows_done_reg, M_total_reg, gpr_M_counter)
        self.generate_instruction_reg_min(m_take_reg, gpr_M_counter, M_chunk_reg)
        self.generate_instruction_add_imm(src_reg_idx=m_take_reg, immediate_value=0, dst_reg_idx=m_tile_rows_reg)

        # ---- A-tile load (m_take * K * 2 bytes) -> URAM_A row 0 ----
        self.generate_instruction_pbi_inc(general_reg_src=a_dram_reg, pbi_field_select=PBI_FIELD.DRAM_ADDR, inst_pointer_idx=ptr_A)
        # DMA_LENGTH = (m_take * K_rows) << 7  (== m_take*K*2 bytes).
        self.generate_instruction_mul32_shl_reg(s2, m_tile_rows_reg, K_rows_reg, 7)
        self.generate_instruction_pbi_inc(general_reg_src=s2, pbi_field_select=PBI_FIELD.DMA_LENGTH, inst_pointer_idx=ptr_A)
        self.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=0, element_size=0,
                                        memcpy_length_bytes=0, inst_pointer_idx=ptr_A)
        # Advance A cursor by m_take rows (m_take * K/4 words).
        self.generate_instruction_mul32_reg(s1, m_tile_rows_reg, k_strip_word_stride_reg)
        self.generate_instruction_add_reg(a_dram_reg, a_dram_reg, s1)

        # clear all fmax contexts for this M tile, set dot uram_length = K_rows
        self.generate_instruction_clear_fmax()
        self.generate_instruction_pbi_inc(general_reg_src=K_rows_reg, pbi_field_select=PBI_FIELD.URAM_ROW_SIZE, inst_pointer_idx=ptr_dot)

        # ---- N while-loop init (reset each M tile) ----
        _seed_cursor(b_dram_reg, B_BASE_W, gpr_b_addr)
        if is_B_quantized:
            _seed_cursor(scale_dram_reg, SCALE_BASE_W, gpr_scale_addr)
        self.generate_instruction_add_imm(src_reg_idx=N_reg, immediate_value=0, dst_reg_idx=N_counter_reg)
        self.generate_instruction_reg_min(n_take_reg, N_counter_reg, N_chunk_reg)

        n_body_start = self.capture_count

        # ---- B-strip load -> URAM_B row 0 ----
        if is_B_quantized:
            b_quant_dma_shift  = 5 if data_type == TYPE.IF4 else 6  # IF4: *32B; IF8: *64B per block
            b_quant_word_shift = 2 if data_type == TYPE.IF4 else 3  # IF4: *4W; IF8: *8W per block
            # 1. Load scale (n_take * K_rows blocks, 2 bytes each) -> BRAM
            self.generate_instruction_pbi_inc(general_reg_src=scale_dram_reg, pbi_field_select=PBI_FIELD.DRAM_ADDR, inst_pointer_idx=ptr_scale)
            self.generate_instruction_mul32_shl_reg(s2, n_take_reg, K_rows_reg, 1)  # n_take * K_rows * 2 bytes
            self.generate_instruction_pbi_inc(general_reg_src=s2, pbi_field_select=PBI_FIELD.DMA_LENGTH, inst_pointer_idx=ptr_scale)
            self.ue_memcpy_from_dram(0, 0, MEMCPY_TYPE.BRAM.value, 0, 0, inst_pointer_idx=ptr_scale)
            # 2. Dequantize quantized B (n_take * K_rows blocks) -> URAM_B
            self.generate_instruction_pbi_inc(general_reg_src=b_dram_reg, pbi_field_select=PBI_FIELD.DRAM_ADDR, inst_pointer_idx=ptr_B_quant)
            self.generate_instruction_mul32_shl_reg(s2, n_take_reg, K_rows_reg, b_quant_dma_shift)  # DMA_LENGTH in bytes
            self.generate_instruction_pbi_inc(general_reg_src=s2, pbi_field_select=PBI_FIELD.DMA_LENGTH, inst_pointer_idx=ptr_B_quant)
            self.generate_instruction_mul32_reg(s2, n_take_reg, K_rows_reg)  # recompute for uram/output size
            self.generate_instruction_pbi_inc(general_reg_src=s2, pbi_field_select=PBI_FIELD.URAM_ROW_SIZE, inst_pointer_idx=ptr_B_quant)
            self.generate_instruction_pbi_inc(general_reg_src=s2, pbi_field_select=PBI_FIELD.OUTPUT_SIZE, inst_pointer_idx=ptr_B_quant)
            self.ue_arithmetic_op(
                broadcast_mode=0, max_clear_en=0, stride_z=1, lalu_a=0, lalu_b=0,
                lalu_mode=LALU_MODE.BYPASS.value, scalar=0,
                uram_section=URAM_SECTION.URAM_B.value, uram_dst_addr=0, uram_wb_addr=0,
                uram_write_src=URAM_WRITE_SRC.URAM_WRITE_BACK.value,
                mode=UE_MODE.DEQUANTIZE, data_type=data_type.value,
                uram_a_start_addr=0, uram_b_start_addr=0, uram_length=0, dma_start_addr=0,
                dma_length=0, output_size=0, inst_pointer_idx=ptr_B_quant)
            # Advance scale cursor: n_take * K_rows * 2 bytes / 8 = n_take * K_rows / 4 words
            self.generate_instruction_mul32_shr_reg(s1, n_take_reg, K_rows_reg, 2)
            self.generate_instruction_add_reg(scale_dram_reg, scale_dram_reg, s1)
            # Advance B_quant cursor: n_take * K_rows * (32 or 64) bytes / 8 words
            self.generate_instruction_mul32_shl_reg(s1, n_take_reg, K_rows_reg, b_quant_word_shift)
            self.generate_instruction_add_reg(b_dram_reg, b_dram_reg, s1)
        else:
            # BF16 B: n_take * K * 2 bytes -> URAM_B
            self.generate_instruction_pbi_inc(general_reg_src=b_dram_reg, pbi_field_select=PBI_FIELD.DRAM_ADDR, inst_pointer_idx=ptr_B)
            self.generate_instruction_mul32_shl_reg(s2, n_take_reg, K_rows_reg, 7)   # n_take*K*2 bytes
            self.generate_instruction_pbi_inc(general_reg_src=s2, pbi_field_select=PBI_FIELD.DMA_LENGTH, inst_pointer_idx=ptr_B)
            self.ue_memcpy_from_dram(0, 0, MEMCPY_TYPE.URAM.value, 0, URAM_SECTION.URAM_B, inst_pointer_idx=ptr_B)
            # Advance B cursor by n_take matrix rows (n_take * K/4 words).
            self.generate_instruction_mul32_reg(s1, n_take_reg, k_strip_word_stride_reg)
            self.generate_instruction_add_reg(b_dram_reg, b_dram_reg, s1)

        # ---- dot-product setup ----
        self.generate_instruction_pbi_inc(general_reg_src=n_take_reg, pbi_field_select=PBI_FIELD.OUTPUT_SIZE, inst_pointer_idx=ptr_dot)
        self.generate_instruction_mul32_reg(s1, n_take_reg, K_rows_reg)
        self.generate_instruction_pbi_inc(general_reg_src=s1, pbi_field_select=PBI_FIELD.URAM_ROW_SIZE_Z, inst_pointer_idx=ptr_dot)
        # ---- bias load per N-strip ----
        if bias_enable and bias_mode == "broadcast_N":
            # One load of bias[cols_done .. cols_done+n_take) shared across all m_take rows.
            # bias word addr = C_BASE_W + cols_done >> 2; cols_done = N - N_counter.
            self.generate_instruction_reg_sub(s1, N_reg, N_counter_reg)     # cols_done
            self.generate_instruction_shr(s1, s1, 2)                        # cols_done*2>>3 -> words
            _off_plus_base(s1, C_BASE_W, gpr_c_addr, s1)
            self.generate_instruction_pbi_inc(general_reg_src=s1, pbi_field_select=PBI_FIELD.DRAM_ADDR, inst_pointer_idx=ptr_bias)
            self.generate_instruction_shl(s1, n_take_reg, 1)                # n_take*2 bytes
            self.generate_instruction_pbi_inc(general_reg_src=s1, pbi_field_select=PBI_FIELD.DMA_LENGTH, inst_pointer_idx=ptr_bias)
            self.ue_memcpy_from_dram(0, 0, MEMCPY_TYPE.BIAS_BRAM.value, 0, 0, inst_pointer_idx=ptr_bias)
        elif bias_enable and bias_mode == "full_matrix":
            # bias_full_row_reg = C_BASE_W + (rows_done*N + cols_done) >> 2
            self.generate_instruction_mul32_reg(s1, rows_done_reg, N_reg)
            self.generate_instruction_reg_sub(s2, N_reg, N_counter_reg)         # cols_done
            self.generate_instruction_add_reg(s1, s1, s2)
            self.generate_instruction_shr(s1, s1, 2)
            _off_plus_base(s1, C_BASE_W, gpr_c_addr, bias_full_row_reg)
            # DMA_LENGTH is constant per strip (n_take*2 bytes); set once, reused each M-row.
            self.generate_instruction_shl(s1, n_take_reg, 1)                    # n_take*2 bytes
            self.generate_instruction_pbi_inc(general_reg_src=s1, pbi_field_select=PBI_FIELD.DMA_LENGTH, inst_pointer_idx=ptr_bias_full)
        # n_take_rows = ceil(n_take / 64) = (n_take + 63) >> 6  (unifies main + aligned path)
        self.generate_instruction_add_imm(src_reg_idx=n_take_reg, immediate_value=63, dst_reg_idx=s4)
        self.generate_instruction_shr(s4, s4, 6)
        # URAM cursors: A row 0 of tile; output base = M_chunk * K_rows (constant across tiles)
        self.generate_instruction_add_set(s2, 0)                # uram_a_cur
        self.generate_instruction_mul32_reg(s3, M_chunk_reg, K_rows_reg)   # uram_wb_cur (base)
        if softmax_enable:
            self.generate_instruction_add_set(row_idx_reg, 0)            # row_idx (fmax context)

        # ---- M-row dot-product loop (counter = m_take_reg, re-derived after) ----
        dot_body = self.capture_count
        self.generate_instruction_pbi_inc(general_reg_src=s2, pbi_field_select=PBI_FIELD.URAM_START_ADDR_Y, inst_pointer_idx=ptr_dot)
        self.generate_instruction_pbi_inc(general_reg_src=s3, pbi_field_select=PBI_FIELD.URAM_WRITEB_ADDR, inst_pointer_idx=ptr_dot)
        if softmax_enable:
            self.generate_instruction_pbi_inc(general_reg_src=row_idx_reg, pbi_field_select=PBI_FIELD.FMX_CONTEXT, inst_pointer_idx=ptr_dot)
        if bias_enable and bias_mode == "full_matrix":
            # Load this M-row's bias slice [cols_done .. cols_done+n_take) into bias BRAM.
            self.generate_instruction_pbi_inc(general_reg_src=bias_full_row_reg, pbi_field_select=PBI_FIELD.DRAM_ADDR, inst_pointer_idx=ptr_bias_full)
            self.ue_memcpy_from_dram(0, 0, MEMCPY_TYPE.BIAS_BRAM.value, 0, 0, inst_pointer_idx=ptr_bias_full)
        self.ue_arithmetic_op(
            broadcast_mode=0, max_clear_en=0, stride_z=1, lalu_a=lalu_a, lalu_b=lalu_b, lalu_mode=lalu_mode.value,
            scalar=0, uram_section=URAM_SECTION.URAM_A.value, uram_dst_addr=0, uram_wb_addr=0,
            uram_write_src=URAM_WRITE_SRC.URAM_WRITE_BACK.value, mode=UE_MODE.BF16_DOT_PRODUCT, data_type=0,
            uram_a_start_addr=0, uram_b_start_addr=0, uram_length=0, dma_start_addr=0, dma_length=0,
            output_size=0, bias_adder_en=(1 if bias_enable else 0), fmax_context_addr=0, inst_pointer_idx=ptr_dot)
        self.generate_instruction_add_reg(s2, s2, K_rows_reg)   # uram_a_cur += K_rows
        self.generate_instruction_add_reg(s3, s3, s4)           # uram_wb_cur += n_take_rows
        if bias_enable and bias_mode == "full_matrix":
            self.generate_instruction_add_reg(bias_full_row_reg, bias_full_row_reg, n_row_words_reg)
        if softmax_enable:
            self.generate_instruction_add_inc(row_idx_reg)               # row_idx++
        dot_loop_sz = self.capture_count - dot_body + 2
        self.generate_instruction_add_dec(m_take_reg)
        self.generate_instruction_jump_rela_jnz(dot_loop_sz, m_take_reg)
        self.generate_instruction_add_imm(src_reg_idx=m_tile_rows_reg, immediate_value=0, dst_reg_idx=m_take_reg)

        # ---- writeback (URAM_A -> DRAM) ----
        if use_strided_wb:
            # One strided DMA for the whole m_take x n_take tile: the m_take rows are contiguous in
            # URAM (uram_wb cursor stepped by n_take_rows == n_take/64 per row) and map to DRAM rows
            # n_take wide at row stride N. chunk bytes = n_take*2 (OUTPUT_SIZE), total = m_take*n_take*2
            # (DMA_LENGTH), DRAM row stride = N*2 sourced from n_stride_bytes_reg at runtime. Replaces
            # the per-row DMA loop (one DMA per M-row) -- the dominant fixed cost on small/skinny shapes.
            self.generate_instruction_mul32_shl_reg(s1, m_take_reg, n_take_reg, 1)   # m_take*n_take*2 (total)
            self.generate_instruction_pbi_inc(general_reg_src=s1, pbi_field_select=PBI_FIELD.DMA_LENGTH, inst_pointer_idx=ptr_wb)
            self.generate_instruction_shl(s2, n_take_reg, 1)                          # n_take*2 (chunk bytes)
            self.generate_instruction_pbi_inc(general_reg_src=s2, pbi_field_select=PBI_FIELD.OUTPUT_SIZE, inst_pointer_idx=ptr_wb)
            self.generate_instruction_mul32_reg(s1, rows_done_reg, N_reg)
            self.generate_instruction_reg_sub(s2, N_reg, N_counter_reg)               # cols_done
            self.generate_instruction_add_reg(s1, s1, s2)
            self.generate_instruction_shr(s1, s1, 2)                                  # (rows_done*N+cols_done) words
            _off_plus_base(s1, OUT_BASE_W, gpr_out_addr, s1)
            self.generate_instruction_mul32_reg(s2, M_chunk_reg, K_rows_reg)          # uram_wb base (tile row 0)
            self.generate_instruction_pbi_inc(general_reg_src=s1, pbi_field_select=PBI_FIELD.DRAM_ADDR, inst_pointer_idx=ptr_wb)
            self.generate_instruction_pbi_inc(general_reg_src=s2, pbi_field_select=PBI_FIELD.URAM_START_ADDR_Y, inst_pointer_idx=ptr_wb)
            self.generate_instruction_pbi_inc(general_reg_src=s2, pbi_field_select=PBI_FIELD.URAM_START_ADDR_Z, inst_pointer_idx=ptr_wb)
            self.ue_memcpy_to_dram(memcpy_type=MEMCPY_TYPE.URAM.value, uram_type=URAM_SECTION.URAM_A.value,
                                   uram_src_addr=0, dram_dst_addr=0, memcpy_length_bytes=0,
                                   inst_pointer_idx=ptr_wb, pbi_stride_en=True)
        elif not write_back_disable:
            # ---- per-row writeback (sub-64 column-strip fallback: non-64 n_take, URAM rows padded) ----
            self.generate_instruction_shl(s1, n_take_reg, 1)            # n_take*2 bytes (DMA length)
            self.generate_instruction_pbi_inc(general_reg_src=s1, pbi_field_select=PBI_FIELD.DMA_LENGTH, inst_pointer_idx=ptr_wb)
            self.generate_instruction_mul32_reg(s1, rows_done_reg, N_reg)
            self.generate_instruction_reg_sub(s2, N_reg, N_counter_reg)        # cols_done
            self.generate_instruction_add_reg(s1, s1, s2)
            self.generate_instruction_shr(s1, s1, 2)
            _off_plus_base(s1, OUT_BASE_W, gpr_out_addr, s1)
            self.generate_instruction_mul32_reg(s2, M_chunk_reg, K_rows_reg)
            wb_body = self.capture_count
            self.generate_instruction_pbi_inc(general_reg_src=s1, pbi_field_select=PBI_FIELD.DRAM_ADDR, inst_pointer_idx=ptr_wb)
            self.generate_instruction_pbi_inc(general_reg_src=s2, pbi_field_select=PBI_FIELD.URAM_START_ADDR_Y, inst_pointer_idx=ptr_wb)
            self.generate_instruction_pbi_inc(general_reg_src=s2, pbi_field_select=PBI_FIELD.URAM_START_ADDR_Z, inst_pointer_idx=ptr_wb)
            self.ue_memcpy_to_dram(memcpy_type=MEMCPY_TYPE.URAM.value, uram_type=URAM_SECTION.URAM_A.value,
                                   uram_src_addr=0, dram_dst_addr=0, memcpy_length_bytes=0, inst_pointer_idx=ptr_wb)
            self.generate_instruction_add_reg(s1, s1, n_row_words_reg)
            self.generate_instruction_add_reg(s2, s2, s4)
            wb_loop_sz = self.capture_count - wb_body + 2
            self.generate_instruction_add_dec(m_take_reg)
            self.generate_instruction_jump_rela_jnz(wb_loop_sz, m_take_reg)
            self.generate_instruction_add_imm(src_reg_idx=m_tile_rows_reg, immediate_value=0, dst_reg_idx=m_take_reg)

        # ---- N-counter update / loop back ----
        self.generate_instruction_reg_sub(N_counter_reg, N_counter_reg, n_take_reg)
        self.generate_instruction_reg_min(n_take_reg, N_counter_reg, N_chunk_reg)
        n_loop_sz = self.capture_count - n_body_start + 2
        self.generate_instruction_jump_rela_jnz(n_loop_sz, N_counter_reg)

        # ===== Optional softmax slab over this M tile ========================
        # The triple M_chunk cap guarantees m_take * N fits URAM_A -> no inner chunk loop.
        if softmax_enable:
            # 1. reload m_take * N * 2 bytes from this tile's output (row 0) -> URAM_A row 0
            self.generate_instruction_add_imm(src_reg_idx=m_tile_rows_reg, immediate_value=0, dst_reg_idx=m_take_reg)
            self.generate_instruction_mul32_shl_reg(s1, m_take_reg, N_reg, 1)        # m_take*N*2 bytes
            self.generate_instruction_pbi_inc(general_reg_src=s1, pbi_field_select=PBI_FIELD.DMA_LENGTH, inst_pointer_idx=ptr_sm_reload)
            # tile row-0 word addr = OUT_BASE_W + rows_done*N/4
            self.generate_instruction_mul32_shr_reg(s2, rows_done_reg, N_reg, 2)
            _off_plus_base(s2, OUT_BASE_W, gpr_out_addr, s2)
            self.generate_instruction_pbi_inc(general_reg_src=s2, pbi_field_select=PBI_FIELD.DRAM_ADDR, inst_pointer_idx=ptr_sm_reload)
            self.ue_memcpy_from_dram(0, 0, MEMCPY_TYPE.URAM.value, 0, URAM_SECTION.URAM_A, inst_pointer_idx=ptr_sm_reload)

            # N_chunks = N // 64 URAM rows per output row; set EXP/MUL uram_length
            self.generate_instruction_shr(s4, N_reg, 6)
            self.generate_instruction_pbi_inc(general_reg_src=s4, pbi_field_select=PBI_FIELD.URAM_ROW_SIZE, inst_pointer_idx=ptr_sm_exp)
            self.generate_instruction_pbi_inc(general_reg_src=s4, pbi_field_select=PBI_FIELD.URAM_ROW_SIZE, inst_pointer_idx=ptr_sm_mul)

            # 2. per-row EXP+RECIP (BCAST_FMAX_NEGATE) -> URAM_A, then MUL_BROADCAST (BCAST_LALU_RESULT) -> URAM_B
            self.generate_instruction_add_set(row_idx_reg, 0)        # row_idx (fmax context)
            self.generate_instruction_add_set(s2, 0)        # URAM row cursor
            sm_body = self.capture_count
            self.generate_instruction_pbi_inc(general_reg_src=s2, pbi_field_select=PBI_FIELD.URAM_START_ADDR_Y, inst_pointer_idx=ptr_sm_exp)
            self.generate_instruction_pbi_inc(general_reg_src=s2, pbi_field_select=PBI_FIELD.URAM_WRITEB_ADDR, inst_pointer_idx=ptr_sm_exp)
            self.generate_instruction_pbi_inc(general_reg_src=row_idx_reg, pbi_field_select=PBI_FIELD.FMX_CONTEXT, inst_pointer_idx=ptr_sm_exp)
            self.ue_arithmetic_op(
                broadcast_mode=BROADCAST_MODE.FMAX_NEGATE.value, max_clear_en=0, stride_z=1, lalu_a=0, lalu_b=0,
                lalu_mode=LALU_MODE.MODE_RECIP.value, scalar=0,
                uram_section=URAM_SECTION.URAM_A.value, uram_dst_addr=0, uram_wb_addr=0,
                uram_write_src=URAM_WRITE_SRC.URAM_WRITE_BACK.value, mode=UE_MODE.EXP, data_type=0,
                uram_a_start_addr=0, uram_b_start_addr=0, uram_length=0, dma_start_addr=0, dma_length=0,
                output_size=0, fmax_context_addr=0, inst_pointer_idx=ptr_sm_exp)
            self.generate_instruction_pbi_inc(general_reg_src=s2, pbi_field_select=PBI_FIELD.URAM_START_ADDR_Y, inst_pointer_idx=ptr_sm_mul)
            self.generate_instruction_pbi_inc(general_reg_src=s2, pbi_field_select=PBI_FIELD.URAM_WRITEB_ADDR, inst_pointer_idx=ptr_sm_mul)
            self.ue_arithmetic_op(
                broadcast_mode=BROADCAST_MODE.LALU_RESULT.value, max_clear_en=0, stride_z=1, lalu_a=0, lalu_b=0,
                lalu_mode=LALU_MODE.BYPASS.value, scalar=0,
                uram_section=URAM_SECTION.URAM_B.value, uram_dst_addr=0, uram_wb_addr=0,
                uram_write_src=URAM_WRITE_SRC.URAM_WRITE_BACK.value, mode=UE_MODE.MUL_BROADCAST, data_type=0,
                uram_a_start_addr=0, uram_b_start_addr=0, uram_length=0, dma_start_addr=0, dma_length=0,
                output_size=0, fmax_context_addr=0, inst_pointer_idx=ptr_sm_mul)
            self.generate_instruction_add_reg(s2, s2, s4)   # cursor += N_chunks
            self.generate_instruction_add_inc(row_idx_reg)           # row_idx++
            sm_loop_sz = self.capture_count - sm_body + 2
            self.generate_instruction_add_dec(m_take_reg)
            self.generate_instruction_jump_rela_jnz(sm_loop_sz, m_take_reg)
            self.generate_instruction_add_imm(src_reg_idx=m_tile_rows_reg, immediate_value=0, dst_reg_idx=m_take_reg)

            # 3. writeback URAM_B -> DRAM as ONE contiguous DMA. The softmax MUL_BROADCAST laid the
            # m_take output rows out contiguously in URAM_B (cursor += N_chunks per row from row 0),
            # and they map to the contiguous DRAM block [OUT_BASE_W + rows_done*N/4 .. +m_take*N/4).
            # ptr_sm_wb was pbi_init'd with URAM_START_ADDR_Y/Z = 0 and is touched nowhere else, so the
            # URAM-B read base stays row 0; only DRAM_ADDR and DMA_LENGTH need a per-tile update.
            self.generate_instruction_mul32_shl_reg(s1, m_tile_rows_reg, N_reg, 1)   # m_take*N*2 bytes (length)
            self.generate_instruction_pbi_inc(general_reg_src=s1, pbi_field_select=PBI_FIELD.DMA_LENGTH, inst_pointer_idx=ptr_sm_wb)
            self.generate_instruction_mul32_shr_reg(s1, rows_done_reg, N_reg, 2)     # tile row-0 word addr offset
            _off_plus_base(s1, OUT_BASE_W, gpr_out_addr, s1)
            self.generate_instruction_pbi_inc(general_reg_src=s1, pbi_field_select=PBI_FIELD.DRAM_ADDR, inst_pointer_idx=ptr_sm_wb)
            self.ue_memcpy_to_dram(memcpy_type=MEMCPY_TYPE.URAM.value, uram_type=URAM_SECTION.URAM_B.value,
                                   uram_src_addr=0, dram_dst_addr=0, memcpy_length_bytes=0, inst_pointer_idx=ptr_sm_wb)

        # ---- M-counter update / loop back ----
        self.generate_instruction_add_imm(src_reg_idx=m_tile_rows_reg, immediate_value=0, dst_reg_idx=m_take_reg)
        self.generate_instruction_reg_sub(gpr_M_counter, gpr_M_counter, m_take_reg)
        self.generate_instruction_reg_min(m_take_reg, gpr_M_counter, M_chunk_reg)
        outer_loop_size = self.capture_count - body_start_inst_cnt + 1
        self.generate_instruction_jump_rela_jnz(outer_loop_size, gpr_M_counter)

        print(f"matmat_mul_core_dynamic while-loop body size: {outer_loop_size}")
        assert outer_loop_size <= 512, (
            f"Outer while-loop body size {outer_loop_size} exceeds i-cache limit of 512 instructions"
        )

        # release pointers and registers (LIFO)
        if softmax_enable:
            for ptr in (ptr_sm_wb, ptr_sm_mul, ptr_sm_exp, ptr_sm_reload):
                self.release_inst_ptr(ptr)
        if bias_enable:
            if bias_mode == "broadcast_N":
                self.release_inst_ptr(ptr_bias)
            else:
                self.release_inst_ptr(ptr_bias_full)
        if is_B_quantized:
            for ptr in (ptr_wb, ptr_dot, ptr_B_quant, ptr_scale, ptr_A):
                self.release_inst_ptr(ptr)
        else:
            for ptr in (ptr_wb, ptr_dot, ptr_B, ptr_A):
                self.release_inst_ptr(ptr)
        for _ in range(len(_alloc_list)):
            self.release_isa_reg()

        total_flops = 2 * M * K * N
        if bias_enable:
            total_flops += M * N
        if softmax_enable:
            total_flops += M * N * 5
        if gelu_enable or silu_enable or sigmoid_enable:
            total_flops += 4 * M * N
        if clamp_enable:
            total_flops += M * N
        if log_enable:
            total_flops += 2 * M * N
        print(f"Total Theoretical FLOPS: {total_flops / 1e9:.6f} G")
        return total_flops

    @staticmethod
    def matmat_mul_two_cores(ue0: "UnifiedEngine",
                             ue1: "UnifiedEngine",
                             M: int,
                             K: int,
                             N: int,
                             A_DRAM_ADDR: int,
                             B_DRAM_ADDR: int,
                             OUTPUT_DRAM_ADDR: int,
                             softmax_enable: bool = False,
                             C_DRAM_ADDR: int = None,
                             bias_mode: str = "broadcast_N",
                             is_B_quantized: bool = False,
                             data_type: TYPE = None,
                             SCALE_DRAM_ADDR: int = None,
                             gelu_enable: bool = False,
                             silu_enable: bool = False,
                             sigmoid_enable: bool = False,
                             clamp_enable: bool = False,
                             log_enable: bool = False,
                             m_engine0: int = None,
                             wait_timeout_seconds: float = 10.0,
                             dynamic: bool = False) -> int:
        """
        Run one matmul on two engines in parallel by sharding rows of A along M.

        ue0 computes top rows, ue1 computes bottom rows. Both engines use the
        same B matrix. Output is written contiguously to OUTPUT_DRAM_ADDR.

        Returns:
            Total theoretical FLOPs of the combined two-engine workload.
        """
        bytes_per_element = 2

        assert M >= 2, f"M must be at least 2 for two-core execution, got M={M}"
        if m_engine0 is None:
            m_engine0 = M // 2
        assert m_engine0 >= 1 and m_engine0 < M, f"m_engine0 must be in [1, M-1], got m_engine0={m_engine0}, M={M}"
        m_engine1 = M - m_engine0

        A0_DRAM_ADDR = A_DRAM_ADDR
        A1_DRAM_ADDR = A_DRAM_ADDR + m_engine0 * K * bytes_per_element
        OUT0_DRAM_ADDR = OUTPUT_DRAM_ADDR
        OUT1_DRAM_ADDR = OUTPUT_DRAM_ADDR + m_engine0 * N * bytes_per_element

        C0_DRAM_ADDR = C_DRAM_ADDR
        C1_DRAM_ADDR = C_DRAM_ADDR
        if C_DRAM_ADDR is not None and bias_mode == "full_matrix":
            C1_DRAM_ADDR = C_DRAM_ADDR + m_engine0 * N * bytes_per_element

        # Dynamic path primes M/K/N GPRs on each engine before capture.
        m_reg0 = k_reg0 = n_reg0 = None
        m_reg1 = k_reg1 = n_reg1 = None
        if dynamic:
            m_reg0 = ue0.alloc_isa_reg()
            k_reg0 = ue0.alloc_isa_reg()
            n_reg0 = ue0.alloc_isa_reg()
            m_reg1 = ue1.alloc_isa_reg()
            k_reg1 = ue1.alloc_isa_reg()
            n_reg1 = ue1.alloc_isa_reg()

        # Program engine0
        ue0.start_capture()
        ue0.generate_instruction_flag_clear()
        if dynamic:
            ue0.generate_instruction_add_set(m_reg0, m_engine0)
            ue0.generate_instruction_add_set(k_reg0, K)
            ue0.generate_instruction_add_set(n_reg0, N)
        flops_engine0 = ue0.matmat_mul_core(
            M=m_engine0,
            K=K,
            N=N,
            A_DRAM_ADDR=A0_DRAM_ADDR,
            B_DRAM_ADDR=B_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=OUT0_DRAM_ADDR,
            softmax_enable=softmax_enable,
            C_DRAM_ADDR=C0_DRAM_ADDR,
            bias_mode=bias_mode,
            is_B_quantized=is_B_quantized,
            data_type=data_type,
            SCALE_DRAM_ADDR=SCALE_DRAM_ADDR,
            gelu_enable=gelu_enable,
            silu_enable=silu_enable,
            sigmoid_enable=sigmoid_enable,
            clamp_enable=clamp_enable,
            log_enable=log_enable,
            gpr_M_reg=m_reg0,
            gpr_K_reg=k_reg0,
            gpr_N_reg=n_reg0,
        )
        ue0.generate_instruction_flag_set()
        ue0.generate_instruction_halt()
        ue0.stop_capture()
        if dynamic:
            ue0.release_isa_reg()
            ue0.release_isa_reg()
            ue0.release_isa_reg()

        engine0_program_dram_addr = ue0.get_program_dram_addr()
        ue0.write_captured_instructions_to_dram(engine0_program_dram_addr)
        ue0.allocate_program_dram(ue0.get_capture_instruction_size_bytes())

        # Program ue1
        ue1.start_capture()
        if dynamic:
            ue1.generate_instruction_add_set(m_reg1, m_engine1)
            ue1.generate_instruction_add_set(k_reg1, K)
            ue1.generate_instruction_add_set(n_reg1, N)
        flops_engine1 = ue1.matmat_mul_core(
            M=m_engine1,
            K=K,
            N=N,
            A_DRAM_ADDR=A1_DRAM_ADDR,
            B_DRAM_ADDR=B_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=OUT1_DRAM_ADDR,
            softmax_enable=softmax_enable,
            C_DRAM_ADDR=C1_DRAM_ADDR,
            bias_mode=bias_mode,
            is_B_quantized=is_B_quantized,
            data_type=data_type,
            SCALE_DRAM_ADDR=SCALE_DRAM_ADDR,
            gelu_enable=gelu_enable,
            silu_enable=silu_enable,
            sigmoid_enable=sigmoid_enable,
            clamp_enable=clamp_enable,
            log_enable=log_enable,
            gpr_M_reg=m_reg1,
            gpr_K_reg=k_reg1,
            gpr_N_reg=n_reg1,
        )
        ue1.generate_instruction_flag_check(target_engine_idx=0)
        ue1.generate_instruction_halt()
        ue1.stop_capture()
        if dynamic:
            ue1.release_isa_reg()
            ue1.release_isa_reg()
            ue1.release_isa_reg()

        engine1_program_dram_addr = ue1.get_program_dram_addr()
        ue1.write_captured_instructions_to_dram(engine1_program_dram_addr)
        ue1.allocate_program_dram(ue1.get_capture_instruction_size_bytes())

        # Launch both engines
        ue0.start_execute_from_dram(engine0_program_dram_addr)
        ue1.start_execute_from_dram(engine1_program_dram_addr)
        ue1.wait_queue(wait_timeout_seconds)

        total_flops = flops_engine0 + flops_engine1
        print(f"Total Theoretical FLOPS (two cores): {total_flops / 1e9:.6f} G")
        return total_flops

    def fmax_core(self, vector_sram_start_addr: int, output_sram_wb_addr: int, N: int, fmax_context_addr: int = 0) -> None:
        """get's the fmax from the fmax_context_addr and stores it in the output_sram_wb_addr.

        Uses ADD_BROADCAST with FMAX_NEGATE broadcast mode to subtract the tracked
        maximum (populated by a prior dot product / matvec with max_clear_en=1)
        from every element in the input vector.

        Args:
            vector_sram_start_addr: SRAM address of input vector (must be in URAM_A: 0x00000-0x7FFFF)
            output_sram_wb_addr: SRAM address for output writeback
            N: number of elements
            fmax_context_addr: fmax context slot (0-63) populated by a prior dot product (default: 0)
        """
        vector_uram_type, vector_uram_start_addr = self.sram_address_to_uram_address(vector_sram_start_addr)
        assert vector_uram_type == URAM_SECTION.URAM_A, f"vector_sram_start_addr must be in URAM_A, got {hex(vector_sram_start_addr)}"

        output_uram_type, output_uram_addr = self.sram_address_to_uram_address(output_sram_wb_addr)

        row_size = (N + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE

        self.ue_arithmetic_op(
            BROADCAST_MODE.FMAX_NEGATE.value,  # broadcast_mode: use -fmax from context
            0,  # max_clear_en
            1,  # stride_z
            0,  # lalu_a
            0,  # lalu_b
            LALU_MODE.BYPASS.value,  # lalu_mode
            0,  # scalar (not used with FMAX_NEGATE)
            output_uram_type.value,
            0,  # uram_dst_addr
            output_uram_addr,
            URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
            UE_MODE.ADD_BROADCAST,
            0,  # data_type
            vector_uram_start_addr,
            0,  # uram_b_start_addr (not used for broadcast)
            row_size,
            0,  # dma_start_addr
            0,  # dma_length
            0,  # output_size
            fmax_context_addr=fmax_context_addr
        )

    def bf16_transpose_core(self, M: int, N: int, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, IDENTITY_DRAM_ADDR: int = None, gpr_M_reg: int = None, gpr_N_reg: int = None,
                            gpr_input_addr: Optional[int] = None, gpr_out_addr: Optional[int] = None, gpr_identity_addr: Optional[int] = None) -> None:
        """Transpose ``M×N`` → ``N×M``. Dispatches to :meth:`bf16_transpose_core_dynamic` when any ``gpr_*`` is set, else :meth:`bf16_transpose_core_legacy`.

        ``gpr_input_addr`` / ``gpr_out_addr`` / ``gpr_identity_addr`` (dynamic path only) optionally
        source the X / Y / identity-matrix DRAM bases from GPRs (word addr). See
        :meth:`bf16_transpose_core_dynamic`."""
        if any(r is not None for r in (gpr_M_reg, gpr_N_reg, gpr_input_addr,
                                       gpr_out_addr, gpr_identity_addr)):
            seeded_regs = []
            if gpr_M_reg is None:
                gpr_M_reg = self.alloc_isa_reg(); seeded_regs.append(gpr_M_reg)
                self.generate_instruction_add_set(gpr_M_reg, M)
            result = self.bf16_transpose_core_dynamic(
                M, N, INPUT_DRAM_ADDR, OUTPUT_DRAM_ADDR, IDENTITY_DRAM_ADDR,
                gpr_M_reg, gpr_N_reg, gpr_input_addr=gpr_input_addr,
                gpr_out_addr=gpr_out_addr, gpr_identity_addr=gpr_identity_addr)
            for _ in seeded_regs:
                self.release_isa_reg()
            return result
        return self.bf16_transpose_core_legacy(M, N, INPUT_DRAM_ADDR, OUTPUT_DRAM_ADDR, IDENTITY_DRAM_ADDR)

    def bf16_transpose_core_legacy(self, M: int, N: int, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, IDENTITY_DRAM_ADDR: int = None) -> None:
        """
        Transposes a (M x N) input matrix X to produce an (N x M) output matrix Y = X^T.

        Uses the identity matrix trick: for each column j of X, computes
        Y[j, :] = e_j^T @ X = X[:, j] via bf16 matvec with a one-hot vector.
        Processes N in column blocks of UE_VECTOR_SIZE width.
        """
        bytes_per_element = 2
        # Allocate identity matrix of size UE_VECTOR_SIZE x UE_VECTOR_SIZE in URAM_A start
        identity_matrix_sram_start_addr = 0x00000
        if IDENTITY_DRAM_ADDR is not None:
            identity_matrix_dram_addr = IDENTITY_DRAM_ADDR
        else:
            identity_matrix_dram_addr = self.get_params_dram_addr()
            self.allocate_params_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * bytes_per_element)
            self.dma_write(DMA_DEVICE_H2C, identity_matrix_dram_addr, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16), UE_VECTOR_SIZE * UE_VECTOR_SIZE * bytes_per_element)

        identity_tensor = torch.eye(N, dtype=torch.bfloat16)

        # transfer identity matrix to URAM_A start
        self.accelerator_memory_to_sram(accelerator_dram_address=identity_matrix_dram_addr,
                                        sram_address=identity_matrix_sram_start_addr,
                                        element_size=UE_VECTOR_SIZE * UE_VECTOR_SIZE)

        usable_uram_a_start_addr = identity_matrix_sram_start_addr + UE_VECTOR_SIZE * UE_VECTOR_SIZE * bytes_per_element
        usable_uram_b_elements = URAM_NEAR_FULL_ELEMENTS
        M_chunk = min(M, (usable_uram_b_elements // N) // UE_VECTOR_SIZE * UE_VECTOR_SIZE)
        M_chunk_aligned = None
        if M_chunk < UE_VECTOR_SIZE:
            if (N * 32) <= usable_uram_b_elements:
                M_chunk = 32
            elif (N * 16) <= usable_uram_b_elements:
                M_chunk = 16
            else:
                assert False, f"N={N} is too large to fit in usable URAM elements={usable_uram_b_elements}"
            M_chunk_aligned = UE_VECTOR_SIZE

        usable_uram_a_elements = URAM_FULL_ELEMENTS - UE_VECTOR_SIZE * UE_VECTOR_SIZE
        output_M_size = M_chunk_aligned if M_chunk_aligned is not None else M_chunk
        N_chunk = min(N, usable_uram_a_elements // output_M_size)
        assert N_chunk % UE_VECTOR_SIZE == 0, f"N_chunk={N_chunk} must be a multiple of UE_VECTOR_SIZE={UE_VECTOR_SIZE}"
        assert N_chunk >= 1 and N_chunk <= N, f"N_chunk={N_chunk} must be greater than 0 and less than N={N}"

        print(f"M_chunk: {M_chunk}, N_chunk: {N_chunk}", f"M_chunk_aligned: {M_chunk_aligned}")
        print(f"URAM_A usage: {100 * (UE_VECTOR_SIZE * UE_VECTOR_SIZE + N_chunk * output_M_size) / URAM_FULL_ELEMENTS:.2f}% of URAM_NEAR_FULL_ELEMENTS")
        print(f"URAM_B usage: {100 * M_chunk * N / URAM_FULL_ELEMENTS:.2f}% of URAM_FULL_ELEMENTS")

        output_sram_wb_addr = usable_uram_a_start_addr
        uram_b_start_addr = 0x80000
        for i, n_take in self.chunk_ranges(N, N_chunk):
            for j, m_take in self.chunk_ranges(M, M_chunk):

                self.accelerator_memory_to_sram(accelerator_dram_address=INPUT_DRAM_ADDR + j * N * bytes_per_element,
                                            sram_address=uram_b_start_addr,
                                            element_size=m_take * N)

                for output_row in range(n_take):
                    if M_chunk_aligned is None:
                        out_sram_offset = output_row * m_take * bytes_per_element
                    else:
                        out_sram_offset = output_row * M_chunk_aligned * bytes_per_element

                    ones_idx = identity_tensor[output_row+i, :].reshape(-1, UE_VECTOR_SIZE).sum(axis=1).argmax(axis=0)
                    vector_idx = identity_tensor[output_row+i, :].reshape(-1, UE_VECTOR_SIZE)[ones_idx, :].argmax(axis=0)

                    self.start_queue_for_bf16_matvec_operation(max_clear_en=0,
                                                                fmax_context_addr=0,
                                                                vector_sram_start_addr=0x00000 + vector_idx * UE_VECTOR_SIZE * bytes_per_element,
                                                                matrix_sram_start_addr=uram_b_start_addr + ones_idx * UE_VECTOR_SIZE * bytes_per_element,
                                                                output_sram_wb_addr=output_sram_wb_addr + out_sram_offset,
                                                                K=UE_VECTOR_SIZE,
                                                                N=m_take,
                                                                stride_z=N)

                start_dram_address_of_partial_matrix = OUTPUT_DRAM_ADDR + i * M * bytes_per_element + j * bytes_per_element

                if M_chunk_aligned is None:
                    self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr,
                                                    accelerator_dram_address=start_dram_address_of_partial_matrix,
                                                    element_size=n_take * m_take,
                                                    stride_bytes_per_chunk=m_take * bytes_per_element,
                                                    stride_jump_bytes=M * bytes_per_element)
                else:
                    for o_row_idx in range(n_take):
                        self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr + o_row_idx * M_chunk_aligned * bytes_per_element,
                                                        accelerator_dram_address=start_dram_address_of_partial_matrix + o_row_idx * M * bytes_per_element,
                                                        element_size=m_take)

        # No FLOPS for this operation
    
    def bf16_transpose_core_dynamic(self, M: int, N: int,
                                    INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                                    IDENTITY_DRAM_ADDR: int = None,
                                    gpr_M_reg: int = None, gpr_N_reg: int = None,
                                    gpr_input_addr: Optional[int] = None, gpr_out_addr: Optional[int] = None,
                                    gpr_identity_addr: Optional[int] = None) -> None:
        """
        Transpose an (M x N) input matrix X into an (N x M) output Y = X^T, captured as a
        single replayable ISA program with **runtime (dynamic) M and N**.

        The Python ``for i, j in chunk_ranges`` tile loops of
        :meth:`bf16_transpose_core_legacy` are replaced by hardware ISA while-loops so one
        compiled body serves any M (and, when ``gpr_N_reg`` is given, any N up to 4032) within
        the URAM budget. M is always dynamic; N is dynamic when ``gpr_N_reg`` is supplied,
        otherwise it stays the compile-time ``N`` (backward-compatible path).

        Dynamic N hinges on the identity-matvec column read, whose URAM-B walk stride
        ``uram_row_stride_z = N/64`` is stored in pointer-row field 10 (``URAM_ROW_STRIDE_Z``).
        When ``gpr_N_reg`` is supplied the field is overridden with ``n_blocks_reg = N >> 6``
        right after ``ptr_dot`` init, so N no longer has to be baked in at capture time.
        The block trip count (N/64), the tile height ``M_chunk = floor(4095 / (N/64))`` and
        the per-tile DRAM/eff_z arithmetic are likewise derived in the ISA register file at
        startup.

        ``N`` must be a multiple of 64 and ``<= 4032`` (eff_z = M_chunk * N/64 must fit the
        12-bit ``URAM_ROW_SIZE_Z`` field); the same bound applies to the runtime N value.

        Algorithm (same identity-matvec trick as the legacy core): X[rows..rows+m_take, :]
        is loaded into URAM_B; column ``c`` of that tile (= row ``c`` of Y) is gathered by a
        BF16 dot product of the one-hot identity row ``c%64`` (URAM_A) against URAM-B block
        ``c//64`` strided by N/64, producing the ``m_take`` elements ``Y[c, rows..]``. Each
        Y row is written straight back to DRAM (per-column writeback is unavoidable because
        the N*2-byte output stride is dynamic and cannot be pointer-row backed).

        Pointer rows (one per independently advancing stream):
          * ``ptr_in``  — DRAM->URAM_B tile load (DRAM_ADDR, DMA_LENGTH)
          * ``ptr_dot`` — identity matvec (URAM_START_ADDR_Y=identity row, _Z=URAM-B block,
                          OUTPUT_SIZE=m_take, URAM_ROW_SIZE_Z=m_take*N/64, WRITEB=scratch)
          * ``ptr_wb``  — URAM_A scratch -> DRAM Y-row writeback (DRAM_ADDR, DMA_LENGTH)

        Note: ``URAM_START_ADDR_Y`` backs the descriptor's ``uram_a_start_addr`` (the
        identity vector in URAM_A) and ``URAM_START_ADDR_Z`` backs ``uram_b_start_addr``
        (the data in URAM_B) — there is no ``_X`` field.
        """
        bytes_per_element = 2

        assert gpr_M_reg is not None, "bf16_transpose_core_dynamic requires gpr_M_reg (dynamic M)."
        # Optional GPR-sourced DRAM bases (word addr = byte>>3) for input X and output Y, so one
        # captured program serves any placement. The identity matrix is a constant buffer and stays
        # literal. None -> literal (byte-for-byte identical program).
        self._validate_addr_gprs("bf16_transpose_core_dynamic", gpr_M_reg, {
            "gpr_input_addr": gpr_input_addr, "gpr_out_addr": gpr_out_addr,
        })
        if gpr_N_reg is not None:
            for _nm, _rg in (("gpr_input_addr", gpr_input_addr), ("gpr_out_addr", gpr_out_addr)):
                if _rg is not None and _rg == gpr_N_reg:
                    raise ValueError(f"bf16_transpose_core_dynamic: {_nm}={_rg} collides with gpr_N_reg={gpr_N_reg}")

        def _off_plus_base(off_reg, lit_base_w, gpr_base, dst_reg):
            # dst = off_reg + (GPR base or literal base). off_reg holds the per-tile word offset.
            if gpr_base is not None:
                self.generate_instruction_add_reg(dst_reg, off_reg, gpr_base)
            else:
                self.generate_instruction_add_imm(src_reg_idx=off_reg, immediate_value=lit_base_w, dst_reg_idx=dst_reg)

        # Dynamic N is enabled when the caller supplies gpr_N_reg. The column-gather matvec's
        # uram_row_stride_z (= N/64) is held in pointer-row field 10 (URAM_ROW_STRIDE_Z) and
        # overridden with n_blocks_reg after pbi_init, so N becomes a runtime value up to the 4032 cap.
        dynamic_N = gpr_N_reg is not None
        N_reg = gpr_N_reg

        assert N % UE_VECTOR_SIZE == 0, f"N={N} must be a multiple of UE_VECTOR_SIZE={UE_VECTOR_SIZE}"
        assert N <= 4032, (
            f"N={N} exceeds the 4032 transpose cap (eff_z = M_chunk * N/64 must fit the 12-bit "
            f"URAM_ROW_SIZE_Z field). For dynamic N, the runtime value must also be <= 4032."
        )
        N_blocks = N // UE_VECTOR_SIZE          # number of 64-col blocks (== dot stride in URAM rows)
        stride_z_rows = N // UE_VECTOR_SIZE     # uram_row_stride_z for the column-gather matvec
                                                # (compile-time fallback; overridden by reg when dynamic_N)

        # Rows of X that fit in URAM_B (m_take * N elements), rounded down to a 64-multiple
        # so every m_take is 64-aligned (keeps DMA lengths 128-byte aligned).
        M_chunk = (URAM_NEAR_FULL_ELEMENTS // N) // UE_VECTOR_SIZE * UE_VECTOR_SIZE
        if M_chunk == 0:
            M_chunk = UE_VECTOR_SIZE
        assert M_chunk * N <= URAM_FULL_ELEMENTS, f"N={N} too large: one 64-row tile does not fit URAM_B"
        assert M_chunk * stride_z_rows <= 0xFFF, (
            f"N={N} too large: eff_z = M_chunk({M_chunk}) * stride_z({stride_z_rows}) = "
            f"{M_chunk * stride_z_rows} overflows the 12-bit URAM_ROW_SIZE_Z field (max 4095). "
            f"Max supported N is 4032."
        )
        # Per-block strided writeback gathers a 64-column block (64 Y-rows of m_take elements) into a
        # contiguous URAM_A buffer that follows the 64-row identity, then writes it back with ONE
        # strided DMA (chunk = m_take*2, DRAM row stride = M*2 from a GPR) instead of 64 per-column
        # DMAs. The buffer is 64*m_take elements; it plus the identity must fit URAM_A.
        assert UE_VECTOR_SIZE * UE_VECTOR_SIZE + UE_VECTOR_SIZE * M_chunk <= URAM_FULL_ELEMENTS, (
            f"transpose per-block writeback buffer (identity + 64*M_chunk={UE_VECTOR_SIZE * M_chunk}) "
            f"does not fit URAM_A ({URAM_FULL_ELEMENTS}); reduce N's M_chunk."
        )

        # --- Identity matrix (64x64 one-hot rows) in URAM_A rows 0..63; scratch follows. ---
        identity_matrix_sram_start_addr = 0x00000
        if IDENTITY_DRAM_ADDR is not None:
            identity_matrix_dram_addr = IDENTITY_DRAM_ADDR
        else:
            identity_matrix_dram_addr = self.get_params_dram_addr()
            self.allocate_params_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * bytes_per_element)
            self.dma_write(DMA_DEVICE_H2C, identity_matrix_dram_addr, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16), UE_VECTOR_SIZE * UE_VECTOR_SIZE * bytes_per_element)

        # gpr_identity_addr (optional): source the identity-matrix DRAM base from a runtime GPR
        # (word addr) via the REG_REWRITE path instead of the literal, so a single captured program
        # can relocate the identity matrix. None -> literal (byte-for-byte unchanged).
        self.accelerator_memory_to_sram(accelerator_dram_address=identity_matrix_dram_addr,
                                        sram_address=identity_matrix_sram_start_addr,
                                        element_size=UE_VECTOR_SIZE * UE_VECTOR_SIZE,
                                        general_reg_src=gpr_identity_addr)

        URAM_B_ROW0 = (0x80000 >> 7) & 0xFFF    # URAM_B base row index (== 0)
        SCRATCH_LINE = UE_VECTOR_SIZE           # URAM_A row after the 64-row identity (one Y-row buffer)
        A_BASE_W = INPUT_DRAM_ADDR >> 3         # DRAM word addresses (byte >> 3, PBI DRAM_ADDR format)
        OUT_BASE_W = OUTPUT_DRAM_ADDR >> 3

        # --- Register allocation ---
        _alloc_list = []
        def _alloc():
            r = self.alloc_isa_reg()
            _alloc_list.append(r)
            return r

        M_chunk_reg   = _alloc()  # rows per M tile (held in a reg for reg_min; runtime when dynamic_N)
        M_counter     = _alloc()  # remaining M rows
        m_take_reg    = _alloc()  # min(remaining, M_chunk)
        rows_done_reg = _alloc()  # M rows completed before current tile
        eff_z_reg     = _alloc()  # m_take * (N/64)  -> URAM_ROW_SIZE_Z span for the matvec
        out_stride_reg = _alloc() # M >> 2 (output DRAM word advance per gathered column)
        out_dram_reg  = _alloc()  # running output DRAM word cursor (per 64-col block)
        n_blocks_reg  = _alloc() if dynamic_N else None  # N/64 == stride_z_rows == block trip count
        m_rows_reg    = _alloc()  # ceil(m_take/64) URAM rows per gathered Y-row (buffer cursor step)
        wb_cursor_reg = _alloc()  # URAM_A buffer write cursor (per column within a 64-col block)
        m_stride_bytes_reg = _alloc()  # M*2 (DRAM row stride for the per-block strided writeback)
        blk_adv_reg   = _alloc()  # 64*M>>2 words (output DRAM advance per 64-col block)
        s1 = _alloc()             # scratch
        s2 = _alloc()             # scratch

        def _set(ptr, field, reg):
            # PBI_MODE_REG: pointer-row <field> := GPR[reg] (absolute), other fields unchanged.
            self.generate_instruction_pbi_inc(general_reg_src=reg, pbi_field_select=field, inst_pointer_idx=ptr)

        # ===== Phase 1: startup tiling arithmetic =====
        if dynamic_N:
            # n_blocks = N >> 6 (== stride_z_rows). M_chunk = floor(4095 / n_blocks) aligned down to
            # 64. This single cap covers BOTH the 12-bit eff_z field (m_take*n_blocks <= 4095) and the
            # URAM_B capacity (m_take*N <= URAM_NEAR_FULL), which coincide since URAM_NEAR_FULL == 4095*64.
            self.generate_instruction_shr(n_blocks_reg, gpr_N_reg, 6)
            self.generate_instruction_add_set(s1, 0xFFF)
            self.generate_instruction_div_reg(s1, s1, n_blocks_reg)
            self.generate_instruction_shr(M_chunk_reg, s1, 6)
            self.generate_instruction_shl(M_chunk_reg, M_chunk_reg, 6)
        else:
            self.generate_instruction_add_set(M_chunk_reg, M_chunk)
        self.generate_instruction_shr(out_stride_reg, gpr_M_reg, 2)             # M*2 bytes >> 3 = M >> 2 words
        self.generate_instruction_shl(m_stride_bytes_reg, gpr_M_reg, 1)         # M*2 bytes (strided-wb DRAM row stride)

        # ===== Phase 2: PBI pointer-row inits (constants) =====
        ptr_in  = self.alloc_inst_ptr()
        ptr_dot = self.alloc_inst_ptr()
        ptr_wb  = self.alloc_inst_ptr()
        self.generate_instruction_pbi_init(dram_shared_addr=INPUT_DRAM_ADDR, inst_pointer_idx=ptr_in)
        # ptr_dot: uram_length=1 (K=64 -> 1 row), uram_a=identity row 0, uram_b=URAM_B row 0, wb=scratch.
        # Field 10 (uram_row_stride_z) initialised to stride_z_rows; overridden with n_blocks_reg below if dynamic_N.
        self.generate_instruction_pbi_init(uram_length=1, uram_a_start_addr=0, uram_b_start_addr=URAM_B_ROW0,
                                           uram_wb_addr=SCRATCH_LINE, uram_row_stride_z=stride_z_rows,
                                           inst_pointer_idx=ptr_dot)
        # ptr_wb: source = scratch (URAM_A read uses uram_start_addr_y; set z too to mirror matmul wb).
        # Field 11 (stride_jump) is M*2 bytes -- set once here since m_stride_bytes_reg is already live.
        self.generate_instruction_pbi_init(dram_shared_addr=OUTPUT_DRAM_ADDR, uram_a_start_addr=SCRATCH_LINE,
                                           uram_b_start_addr=SCRATCH_LINE, inst_pointer_idx=ptr_wb)
        _set(ptr_wb, PBI_FIELD.STRIDE_JUMP, m_stride_bytes_reg)
        if dynamic_N:
            _set(ptr_dot, PBI_FIELD.URAM_ROW_STRIDE_Z, n_blocks_reg)

        # ===== Phase 3: outer M while-loop (counter seeded outside body, decremented at tail) =====
        self.generate_instruction_add_imm(src_reg_idx=gpr_M_reg, immediate_value=0, dst_reg_idx=M_counter)

        # Absolute jump re-anchors the i-cache so all relative backward jumps in the
        # loop body are within 512 slots of a known unconditional anchor.
        _prog_base = self.get_program_dram_addr()
        self.generate_instruction_jump_abs(
            ue_35bit_addr_shifter(_prog_base + (self.capture_count + 1) * INSTRUCTION_SIZE_BYTES)
        )
        m_body_start = self.capture_count
        self.generate_instruction_reg_sub(rows_done_reg, gpr_M_reg, M_counter)  # rows_done = M - remaining
        self.generate_instruction_reg_min(m_take_reg, M_counter, M_chunk_reg)   # m_take = min(remaining, M_chunk)

        # ---- load X[rows_done:rows_done+m_take, :] -> URAM_B row 0 ----
        # rows_done*N (elements) *2 bytes >> 3 = rows_done*N >> 2 (words); m_take*N *2 bytes (DMA length).
        # When N is a register these become reg*reg, so the mul+shift fuses into one ALU word each.
        if dynamic_N:
            self.generate_instruction_mul32_shr_reg(s1, rows_done_reg, N_reg, 2)
        else:
            self.generate_instruction_mul32_imm(s1, rows_done_reg, N)
            self.generate_instruction_shr(s1, s1, 2)
        _off_plus_base(s1, A_BASE_W, gpr_input_addr, s1)
        _set(ptr_in, PBI_FIELD.DRAM_ADDR, s1)
        if dynamic_N:
            self.generate_instruction_mul32_shl_reg(s2, m_take_reg, N_reg, 1)
        else:
            self.generate_instruction_mul32_imm(s2, m_take_reg, N)
            self.generate_instruction_shl(s2, s2, 1)
        _set(ptr_in, PBI_FIELD.DMA_LENGTH, s2)
        self.ue_memcpy_from_dram(0, 0, MEMCPY_TYPE.URAM.value, 0, URAM_SECTION.URAM_B.value, inst_pointer_idx=ptr_in)

        # ---- dot-product per-tile setup ----
        _set(ptr_dot, PBI_FIELD.OUTPUT_SIZE, m_take_reg)                        # m_take outputs per gather
        if dynamic_N:
            self.generate_instruction_mul32_reg(eff_z_reg, m_take_reg, n_blocks_reg)    # m_take * (N/64)
        else:
            self.generate_instruction_mul32_imm(eff_z_reg, m_take_reg, stride_z_rows)   # m_take * (N/64)
        _set(ptr_dot, PBI_FIELD.URAM_ROW_SIZE_Z, eff_z_reg)                     # total URAM-B span (eff_z)
        self.generate_instruction_pbi_inc(general_reg_src=REGFILE_R0_ZERO, pbi_field_select=PBI_FIELD.URAM_START_ADDR_Z, inst_pointer_idx=ptr_dot)  # block := 0

        # ---- writeback per-tile setup (one strided DMA per 64-column block) ----
        self.generate_instruction_add_imm(src_reg_idx=m_take_reg, immediate_value=63, dst_reg_idx=m_rows_reg)
        self.generate_instruction_shr(m_rows_reg, m_rows_reg, 6)                 # ceil(m_take/64) URAM rows per Y-row
        self.generate_instruction_shl(s1, m_take_reg, 1)                         # m_take*2 (chunk bytes = OUTPUT_SIZE)
        _set(ptr_wb, PBI_FIELD.OUTPUT_SIZE, s1)
        self.generate_instruction_shl(s1, m_take_reg, 7)                         # 64*m_take*2 (total = DMA_LENGTH)
        _set(ptr_wb, PBI_FIELD.DMA_LENGTH, s1)
        self.generate_instruction_shr(s1, rows_done_reg, 2)                      # rows_done*2 bytes >> 3
        _off_plus_base(s1, OUT_BASE_W, gpr_out_addr, out_dram_reg)               # block-0 DRAM word addr
        self.generate_instruction_shl(blk_adv_reg, out_stride_reg, 6)            # 64 * (M>>2) words = output advance per block

        # ===== Phase 4: block loop (N/64): gather 64 columns -> URAM buffer, ONE strided DMA out =====
        # Block trip count is N/64: a register when dynamic_N, else the compile-time literal.
        if dynamic_N:
            blk_reg = self.loop_start(gpr_loop_cnt=n_blocks_reg)
        else:
            blk_reg = self.loop_start(loop_cnt=N_blocks)
        # identity row cursor := 0 and buffer write cursor := SCRATCH_LINE at the start of each block
        self.generate_instruction_pbi_inc(general_reg_src=REGFILE_R0_ZERO, pbi_field_select=PBI_FIELD.URAM_START_ADDR_Y, inst_pointer_idx=ptr_dot)
        self.generate_instruction_add_set(wb_cursor_reg, SCRATCH_LINE)

        win_reg = self.loop_start(loop_cnt=UE_VECTOR_SIZE)
        # gather column (one-hot identity row) into the URAM buffer at the running cursor. The URAM-B
        # read stride is N/64: sourced from pointer-row field 10 (uram_row_stride_z), which was initialised
        # to stride_z_rows above and overridden with n_blocks_reg when dynamic_N.
        _set(ptr_dot, PBI_FIELD.URAM_WRITEB_ADDR, wb_cursor_reg)
        self.ue_arithmetic_op(
            broadcast_mode=0, max_clear_en=0, stride_z=stride_z_rows, lalu_a=0, lalu_b=0,
            lalu_mode=LALU_MODE.BYPASS.value, scalar=0, uram_section=URAM_SECTION.URAM_A.value,
            uram_dst_addr=0, uram_wb_addr=0, uram_write_src=URAM_WRITE_SRC.URAM_WRITE_BACK.value,
            mode=UE_MODE.BF16_DOT_PRODUCT, data_type=0, uram_a_start_addr=0, uram_b_start_addr=0,
            uram_length=0, dma_start_addr=0, dma_length=0, output_size=0, bias_adder_en=0,
            fmax_context_addr=0, inst_pointer_idx=ptr_dot)
        # advance identity row (+1) and buffer write cursor (+m_rows)
        self.generate_instruction_pbi_inc(uram_a_start_addr=1, inst_pointer_idx=ptr_dot)
        self.generate_instruction_add_reg(wb_cursor_reg, wb_cursor_reg, m_rows_reg)
        self.loop_end()  # within (64 columns)

        # ONE strided DMA: 64 Y-rows from the URAM buffer (SCRATCH_LINE) -> DRAM, each m_take*2 bytes,
        # at DRAM row stride M*2 sourced from pointer-row field 11 (stride_jump, set after ptr_wb pbi_init).
        _set(ptr_wb, PBI_FIELD.DRAM_ADDR, out_dram_reg)
        self.ue_memcpy_to_dram(memcpy_type=MEMCPY_TYPE.URAM.value, uram_type=URAM_SECTION.URAM_A.value,
                               uram_src_addr=0, dram_dst_addr=0, memcpy_length_bytes=0,
                               inst_pointer_idx=ptr_wb, pbi_stride_en=True)
        # advance output DRAM cursor by one block and URAM-B block (+1 row) for the next 64 columns
        self.generate_instruction_add_reg(out_dram_reg, out_dram_reg, blk_adv_reg)
        self.generate_instruction_pbi_inc(uram_b_start_addr=1, inst_pointer_idx=ptr_dot)
        self.loop_end()  # block (N/64)

        # ===== M-counter update / loop back (mirror loop_end: decrement then JNZ, size +2) =====
        m_loop_sz = self.capture_count - m_body_start + 2
        self.generate_instruction_reg_sub(M_counter, M_counter, m_take_reg)
        self.generate_instruction_jump_rela_jnz(m_loop_sz, M_counter)

        assert m_loop_sz <= 512, f"Outer M-loop body size {m_loop_sz} exceeds i-cache limit of 512 instructions."

        # --- cleanup (LIFO) ---
        for ptr in (ptr_wb, ptr_dot, ptr_in):
            self.release_inst_ptr(ptr)
        for _ in range(len(_alloc_list)):
            self.release_isa_reg()



    def bf16_permute_core(self, dim_0: int, dim_1: int, dim_2: int,
                          INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int) -> int:
        """
        Permutes a (dim_0, dim_1, dim_2) matrix to (dim_1, dim_0, dim_2) by
        gathering rows in permuted order from DRAM into URAM, then writing back.

        dim_2 must be a multiple of UE_VECTOR_SIZE.

        Args:
            dim_0, dim_1, dim_2: input tensor dimensions
            INPUT_DRAM_ADDR: DRAM address of input (dim_0 * dim_1 * dim_2, bf16)
            OUTPUT_DRAM_ADDR: DRAM address for output (dim_1 * dim_0 * dim_2, bf16)

        Returns:
            total_flops (= total elements, 1 op per element for gather)
        """
        bytes_per_element = 2
        assert dim_2 % UE_VECTOR_SIZE == 0, f"dim_2={dim_2} must be a multiple of UE_VECTOR_SIZE={UE_VECTOR_SIZE}"

        permute_indices = torch.arange(0, dim_0 * dim_1, dtype=torch.int32).reshape(dim_0, dim_1, 1)
        permute_indices = permute_indices.permute(1, 0, 2).flatten()

        total_elements = dim_0 * dim_1 * dim_2
        row_bytes = dim_2 * bytes_per_element
        rows_per_uram_chunk = (URAM_NEAR_FULL_ELEMENTS // dim_2)

        output_dram_offset = OUTPUT_DRAM_ADDR
        row_idx = 0
        total_rows = dim_0 * dim_1

        while row_idx < total_rows:
            chunk_rows = min(rows_per_uram_chunk, total_rows - row_idx)

            for j in range(chunk_rows):
                src_row = permute_indices[row_idx + j].item()
                self.ue_memcpy_from_dram(INPUT_DRAM_ADDR + src_row * row_bytes,
                                        row_bytes, 0,
                                        URAM_START_ADDR + (j * dim_2) // UE_VECTOR_SIZE,
                                        URAM_SECTION.URAM_A.value)

            self.ue_memcpy_to_dram(MEMCPY_TYPE.URAM.value, URAM_SECTION.URAM_A.value,
                                   URAM_START_ADDR, output_dram_offset,
                                   chunk_rows * row_bytes)

            output_dram_offset += chunk_rows * row_bytes
            row_idx += chunk_rows

        # No FLOPS for this operation

    def patching_core(self, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                      matrix_dram_addrs: list, scale_dram_addrs: list,
                      C: int = 3, H: int = 384, W: int = 384,
                      patch_h: int = 4, patch_w: int = 4,
                      K: int = 1024, N: int = 64,
                      data_type: 'TYPE' = None) -> int:
        """
        Patching core: extracts (patch_h x patch_w x C) patches from a (C, H, W) image
        and projects each patch through quantized weight matrices.

        The image is divided into a grid of (H/patch_h) x (W/patch_w) patches.
        Patches along the W axis are grouped by len(matrix_dram_addrs), each using
        a different pre-quantized weight matrix (N, K) for the dot product.

        Each group loads C*patch_h URAM rows (one row per channel per pixel row),
        each containing patches_per_group*patch_w = UE_VECTOR_SIZE elements that
        span all patches in the group horizontally.

        URAM layout:
          URAM_A[0..C*patch_h-1]:  input patch data (C*patch_h rows of UE_VECTOR_SIZE)
          URAM_A[HALFWAY..]:       output rows for one patch group

        Args:
            INPUT_DRAM_ADDR: DRAM address of input image (C x H x W, bf16, CHW layout)
            OUTPUT_DRAM_ADDR: DRAM address for output (n_patches_h * n_patches_w, N, bf16)
            matrix_dram_addrs: list of DRAM addresses for quantized weight matrices
            scale_dram_addrs: list of DRAM addresses for weight scales
            C, H, W: image dimensions
            patch_h, patch_w: patch dimensions
            K: weight inner dimension (= C * patch_h * UE_VECTOR_SIZE)
            N: output dimension per patch
            data_type: quantization type (TYPE.IF4 or TYPE.IF8). The INT vs FP
                variant is selected per-block by the sign of the bf16 scale.

        Returns:
            total_flops
        """
        if data_type is None:
            data_type = TYPE.IF4

        bytes_per_element = 2
        n_patches_h = H // patch_h
        n_patches_w = W // patch_w
        patches_per_group = len(matrix_dram_addrs)
        n_groups_w = n_patches_w // patches_per_group

        assert len(matrix_dram_addrs) == len(scale_dram_addrs), "matrix and scale lists must match"
        assert n_patches_w % patches_per_group == 0, f"n_patches_w={n_patches_w} must be divisible by patches_per_group={patches_per_group}"
        assert patches_per_group * patch_w == UE_VECTOR_SIZE, \
            f"patches_per_group*patch_w={patches_per_group * patch_w} must equal UE_VECTOR_SIZE={UE_VECTOR_SIZE}"

        ALIGNMENT_SIZE = K * 2
        BRAM_SIZE_ALIGNED = (SCALE_BRAM_SIZE_BYTES // ALIGNMENT_SIZE) * ALIGNMENT_SIZE

        # Each URAM row holds patches_per_group * patch_w = 64 elements = 128 bytes
        input_bytes_per_uram_row = patches_per_group * patch_w * bytes_per_element

        output_row_uram_addr = URAM_HALFWAY_ADDR
        output_sram_addr = output_row_uram_addr * UE_VECTOR_SIZE * bytes_per_element
        batch_output_dram_addr = OUTPUT_DRAM_ADDR

        for p_i in range(n_patches_h):
            for p_j in range(n_groups_w):
                patch_offset_bytes = p_i * patch_h * W * bytes_per_element + \
                                     p_j * patches_per_group * patch_w * bytes_per_element

                for channel in range(C):
                    for row_idx in range(patch_h):
                        uram_addr = URAM_START_ADDR + channel * patch_h + row_idx
                        src_dram = INPUT_DRAM_ADDR + channel * H * W * bytes_per_element + \
                                   row_idx * W * bytes_per_element + patch_offset_bytes
                        self.ue_memcpy_from_dram(src_dram, input_bytes_per_uram_row, 0,
                                                uram_addr, URAM_SECTION.URAM_A.value)

                for p in range(patches_per_group):
                    scale_remaining_bytes = N * K * 2 // UE_VECTOR_SIZE
                    scale_addr = scale_dram_addrs[p]
                    wb_addr = output_row_uram_addr + p
                    dram_addr = matrix_dram_addrs[p]

                    while scale_remaining_bytes > 0:
                        chunk_scale_bytes = min(scale_remaining_bytes, BRAM_SIZE_ALIGNED)
                        number_of_elements = (chunk_scale_bytes * UE_VECTOR_SIZE) >> 1

                        self.ue_memcpy_from_dram(scale_addr, chunk_scale_bytes,
                                                MEMCPY_TYPE.BRAM.value, 0, 0)

                        if data_type == TYPE.IF4:
                            chunk_dma_bytes = number_of_elements >> 1
                        else:
                            chunk_dma_bytes = number_of_elements

                        N_chunk = number_of_elements // K

                        self.ue_arithmetic_op(
                            0,  # broadcast_mode
                            0,  # clear_max_en
                            1,  # stride_z
                            0,  # lalu_a
                            0,  # lalu_b
                            LALU_MODE.BYPASS.value,
                            0,  # scalar
                            URAM_SECTION.URAM_A.value,
                            0,  # uram_dst_addr
                            wb_addr,
                            URAM_WRITE_SRC.URAM_WRITE_BACK.value,
                            UE_MODE.DOT_PRODUCT,
                            data_type.value,
                            URAM_START_ADDR,
                            0,  # uram_b_start_addr
                            K >> 6,
                            dram_addr,
                            chunk_dma_bytes,
                            N_chunk,
                            0  # no bias
                        )

                        scale_addr += chunk_scale_bytes
                        scale_remaining_bytes -= chunk_scale_bytes
                        wb_addr += N_chunk // UE_VECTOR_SIZE
                        dram_addr += chunk_dma_bytes

                self.ue_memcpy_to_dram(MEMCPY_TYPE.URAM.value, URAM_SECTION.URAM_A.value,
                                       output_row_uram_addr, batch_output_dram_addr,
                                       patches_per_group * N * bytes_per_element)
                batch_output_dram_addr += patches_per_group * N * bytes_per_element

        # no FLOPS for this operation

    # =========================================================================
    # Unified attention: Q(batch, head), K/V(aligned_seq_len, head)
    # =========================================================================

    def unified_attention_core(
        self,
        batch: int,
        aligned_seq_len: int,
        head_dim: int,
        Q_DRAM_ADDR: int,
        K_DRAM_ADDR: int,
        V_DRAM_ADDR: int,
        BIAS_DRAM_ADDR: int,
        OUTPUT_DRAM_ADDR: int,
        SCRATCH_DRAM_ADDR: int,
        IDENTITY_DRAM_ADDR: int = None,
        gpr_batch_reg: int = None,
        gpr_aligned_seq_len_reg: int = None,
        gpr_q_addr: int = None,
        gpr_k_addr: int = None,
        gpr_v_addr: int = None,
        gpr_bias_addr: int = None,
        gpr_out_addr: int = None,
        gpr_head_dim_reg: int = None,
        gpr_scale_reg: int = None,
        gpr_scratch_addr: int = None,
        gpr_identity_addr: int = None,
    ) -> int:
        """Scaled dot-product attention with explicit batch and aligned KV length.

        Runtime-``head_dim`` extension (all optional, dynamic path only):
          * ``gpr_head_dim_reg`` — head_dim as a runtime register (drives the score/PV matmul K/N and,
            with ``gpr_scratch_addr``, the scratch sub-offsets).
          * ``gpr_scale_reg``    — bf16 bit-pattern of the Q pre-scale (``1/sqrt(head_dim)``), primed
            by the host at runtime (the ISA has no on-device rsqrt); fed to the scaled-Q broadcast via
            the field-11 ``lalu_scalar`` override. ``None`` → baked ``1/sqrt(head_dim)`` scalar.
          * ``gpr_scratch_addr`` / ``gpr_identity_addr`` — runtime SCRATCH / IDENTITY base overrides
            (word addr). The three scratch sub-buffers (V.T, score, scaled_q) are re-based off
            ``gpr_scratch_addr`` at runtime.

        Layout contract:
          Q    = ``[batch, head_dim]``
          K/V  = ``[aligned_seq_len, head_dim]``
          bias = ``[batch, aligned_seq_len]`` (full-matrix bias)
          out  = ``[batch, head_dim]``

        Scratch layout:
          ``SCRATCH + 0``                                      -> V.T      ``[head_dim, aligned_seq_len]``
          ``SCRATCH + head_dim*aligned_seq_len*2``             -> score/P  ``[aligned_seq_len, aligned_seq_len]``
          ``SCRATCH + (head_dim+aligned_seq_len)*aligned_seq_len*2``
                                                                 -> scaled_q ``[batch, head_dim]``

        When either dimension GPR is provided, the dynamic path is used. Optional address GPRs
        source Q/K/V/bias/output base addresses and require the dynamic path.
        """
        _addr_gprs = (gpr_q_addr, gpr_k_addr, gpr_v_addr, gpr_bias_addr, gpr_out_addr,
                      gpr_head_dim_reg, gpr_scale_reg, gpr_scratch_addr, gpr_identity_addr)
        if gpr_batch_reg is None and gpr_aligned_seq_len_reg is None and not any(g is not None for g in _addr_gprs):
            return self.unified_attention_core_legacy(
                batch=batch,
                aligned_seq_len=aligned_seq_len,
                head_dim=head_dim,
                Q_DRAM_ADDR=Q_DRAM_ADDR,
                K_DRAM_ADDR=K_DRAM_ADDR,
                V_DRAM_ADDR=V_DRAM_ADDR,
                BIAS_DRAM_ADDR=BIAS_DRAM_ADDR,
                OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                SCRATCH_DRAM_ADDR=SCRATCH_DRAM_ADDR,
                IDENTITY_DRAM_ADDR=IDENTITY_DRAM_ADDR,
            )
        if any(g is not None for g in _addr_gprs) and (gpr_batch_reg is None or gpr_aligned_seq_len_reg is None):
            raise ValueError("unified_attention_core: gpr_*_addr require both gpr_batch_reg and gpr_aligned_seq_len_reg")
        return self.unified_attention_core_dynamic(
            batch=batch,
            aligned_seq_len=aligned_seq_len,
            head_dim=head_dim,
            Q_DRAM_ADDR=Q_DRAM_ADDR,
            K_DRAM_ADDR=K_DRAM_ADDR,
            V_DRAM_ADDR=V_DRAM_ADDR,
            BIAS_DRAM_ADDR=BIAS_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
            SCRATCH_DRAM_ADDR=SCRATCH_DRAM_ADDR,
            IDENTITY_DRAM_ADDR=IDENTITY_DRAM_ADDR,
            gpr_batch_reg=gpr_batch_reg,
            gpr_aligned_seq_len_reg=gpr_aligned_seq_len_reg,
            gpr_q_addr=gpr_q_addr,
            gpr_k_addr=gpr_k_addr,
            gpr_v_addr=gpr_v_addr,
            gpr_bias_addr=gpr_bias_addr,
            gpr_out_addr=gpr_out_addr,
            gpr_head_dim_reg=gpr_head_dim_reg,
            gpr_scale_reg=gpr_scale_reg,
            gpr_scratch_addr=gpr_scratch_addr,
            gpr_identity_addr=gpr_identity_addr,
        )

    def unified_attention_core_legacy(
        self,
        batch: int,
        aligned_seq_len: int,
        head_dim: int,
        Q_DRAM_ADDR: int,
        K_DRAM_ADDR: int,
        V_DRAM_ADDR: int,
        BIAS_DRAM_ADDR: int,
        OUTPUT_DRAM_ADDR: int,
        SCRATCH_DRAM_ADDR: int,
        IDENTITY_DRAM_ADDR: int = None,
    ) -> int:
        bytes_per_element = 2
        if batch <= 0 or aligned_seq_len <= 0 or head_dim <= 0:
            raise ValueError(f"unified_attention_core_legacy: invalid dims batch={batch}, aligned_seq_len={aligned_seq_len}, head_dim={head_dim}")
        if batch > aligned_seq_len:
            raise ValueError(f"unified_attention_core_legacy: batch must be <= aligned_seq_len, got batch={batch}, aligned_seq_len={aligned_seq_len}")
        if aligned_seq_len % UE_VECTOR_SIZE != 0:
            raise ValueError(f"unified_attention_core_legacy: aligned_seq_len must be a multiple of {UE_VECTOR_SIZE}, got {aligned_seq_len}")

        v_t_dram_addr = SCRATCH_DRAM_ADDR
        score_dram_addr = v_t_dram_addr + head_dim * aligned_seq_len * bytes_per_element
        scaled_q_dram_addr = score_dram_addr + aligned_seq_len * aligned_seq_len * bytes_per_element

        self.bf16_transpose_core(
            M=aligned_seq_len,
            N=head_dim,
            INPUT_DRAM_ADDR=V_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=v_t_dram_addr,
            IDENTITY_DRAM_ADDR=IDENTITY_DRAM_ADDR,
        )
        total_flops = self.eltwise_core_dram(
            M=batch,
            N=head_dim,
            dram_a=Q_DRAM_ADDR,
            dram_b=None,
            dram_out=scaled_q_dram_addr,
            mode=UE_MODE.MUL_BROADCAST,
            scalar=1.0 / math.sqrt(head_dim),
        )
        total_flops += self.matmat_mul_core(
            M=batch,
            K=head_dim,
            N=aligned_seq_len,
            A_DRAM_ADDR=scaled_q_dram_addr,
            B_DRAM_ADDR=K_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=score_dram_addr,
            softmax_enable=True,
            C_DRAM_ADDR=BIAS_DRAM_ADDR,
            bias_mode="full_matrix",
        )
        total_flops += self.matmat_mul_core(
            M=batch,
            K=aligned_seq_len,
            N=head_dim,
            A_DRAM_ADDR=score_dram_addr,
            B_DRAM_ADDR=v_t_dram_addr,
            OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
        )
        print(f"unified_attention_core_legacy: batch={batch}, aligned_seq_len={aligned_seq_len}, head_dim={head_dim}, FLOPS={total_flops / 1e9:.6f} G")
        return total_flops

    def unified_attention_core_dynamic(
        self,
        batch: int,
        aligned_seq_len: int,
        head_dim: int,
        Q_DRAM_ADDR: int,
        K_DRAM_ADDR: int,
        V_DRAM_ADDR: int,
        BIAS_DRAM_ADDR: int,
        OUTPUT_DRAM_ADDR: int,
        SCRATCH_DRAM_ADDR: int,
        IDENTITY_DRAM_ADDR: int = None,
        gpr_batch_reg: int = None,
        gpr_aligned_seq_len_reg: int = None,
        gpr_q_addr: int = None,
        gpr_k_addr: int = None,
        gpr_v_addr: int = None,
        gpr_bias_addr: int = None,
        gpr_out_addr: int = None,
        gpr_head_dim_reg: int = None,
        gpr_scale_reg: int = None,
        gpr_scratch_addr: int = None,
        gpr_identity_addr: int = None,
    ) -> int:
        bytes_per_element = 2
        if batch > aligned_seq_len:
            raise ValueError(f"unified_attention_core_dynamic: batch must be <= aligned_seq_len, got batch={batch}, aligned_seq_len={aligned_seq_len}")
        if aligned_seq_len % UE_VECTOR_SIZE != 0:
            raise ValueError(f"unified_attention_core_dynamic: aligned_seq_len must be a multiple of {UE_VECTOR_SIZE}, got {aligned_seq_len}")
        allocated_regs = []
        def _alloc():
            r = self.alloc_isa_reg(); allocated_regs.append(r); return r
        if gpr_batch_reg is None:
            gpr_batch_reg = _alloc()
            self.generate_instruction_add_set(gpr_batch_reg, batch)
        if gpr_aligned_seq_len_reg is None:
            gpr_aligned_seq_len_reg = _alloc()
            self.generate_instruction_add_set(gpr_aligned_seq_len_reg, aligned_seq_len)

        # head_dim register. When the caller passes gpr_head_dim_reg, head_dim is *runtime* and the
        # score/PV matmuls plus the V transpose and Q pre-scale all take it as a register (the transpose
        # then uses its dynamic-N path and the pre-scale its dynamic-N eltwise). Otherwise head_dim is a
        # compile-time constant: head_dim_reg still holds it (the matmuls always take all three dims as
        # registers), but the transpose bakes N=head_dim (its validated dynamic-M / compile-time-N path)
        # and the Q pre-scale runs as an N-baked PBI broadcast, avoiding the HW-unvalidated dynamic-N
        # transpose for the common constant-head_dim case (e.g. gemma, head_dim=256).
        head_dim_runtime = gpr_head_dim_reg is not None
        if head_dim_runtime:
            head_dim_reg = gpr_head_dim_reg
        else:
            head_dim_reg = _alloc()
            self.generate_instruction_add_set(head_dim_reg, head_dim)
        sub_hd_n_reg = head_dim_reg if head_dim_runtime else None  # gpr_N_reg for transpose / pre-scale

        # Scratch sub-buffers. Offsets are compile-time (from the template head_dim / aligned_seq_len);
        # when gpr_scratch_addr is supplied they are re-based off the runtime scratch word address by
        # adding those constant word offsets, so one captured program can place the scratch anywhere.
        # NOTE: with a runtime aligned_seq_len the compile-time ``aligned_seq_len`` MUST be the maximum
        # the runtime value can take (e.g. gemma's FLASH_ATTN_ROWS), so these offsets reserve the max
        # footprint and the sub-buffers never overlap at a smaller runtime length.
        v_t_dram_addr = SCRATCH_DRAM_ADDR
        score_dram_addr = v_t_dram_addr + head_dim * aligned_seq_len * bytes_per_element
        scaled_q_dram_addr = score_dram_addr + aligned_seq_len * aligned_seq_len * bytes_per_element
        v_t_reg = score_reg = scaled_q_reg = None
        if gpr_scratch_addr is not None:
            off_score_w = (head_dim * aligned_seq_len * bytes_per_element) >> 3          # words
            off_scaled_q_w = off_score_w + ((aligned_seq_len * aligned_seq_len * bytes_per_element) >> 3)
            v_t_reg = _alloc();     self.generate_instruction_add_imm(src_reg_idx=gpr_scratch_addr, immediate_value=0, dst_reg_idx=v_t_reg)
            score_reg = _alloc();   self.generate_instruction_add_imm(src_reg_idx=gpr_scratch_addr, immediate_value=off_score_w, dst_reg_idx=score_reg)
            scaled_q_reg = _alloc(); self.generate_instruction_add_imm(src_reg_idx=gpr_scratch_addr, immediate_value=off_scaled_q_w, dst_reg_idx=scaled_q_reg)

        self.bf16_transpose_core(
            M=aligned_seq_len,
            N=head_dim,
            INPUT_DRAM_ADDR=V_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=v_t_dram_addr,
            IDENTITY_DRAM_ADDR=IDENTITY_DRAM_ADDR,
            gpr_M_reg=gpr_aligned_seq_len_reg,
            gpr_N_reg=sub_hd_n_reg,
            gpr_input_addr=gpr_v_addr,
            gpr_out_addr=v_t_reg,
            gpr_identity_addr=gpr_identity_addr,
        )
        total_flops = self.eltwise_core_dram(
            M=batch,
            N=head_dim,
            dram_a=Q_DRAM_ADDR,
            dram_b=None,
            dram_out=scaled_q_dram_addr,
            mode=UE_MODE.MUL_BROADCAST,
            scalar=1.0 / math.sqrt(head_dim),
            gpr_M_reg=gpr_batch_reg,
            gpr_N_reg=sub_hd_n_reg,   # runtime head_dim -> dynamic eltwise; else N-baked PBI broadcast
            gpr_a_addr=gpr_q_addr,
            gpr_out_addr=scaled_q_reg,
            gpr_scalar_reg=gpr_scale_reg,   # runtime 1/sqrt(head_dim) (host-primed); None -> baked
        )
        total_flops += self.matmat_mul_core(
            M=batch,
            K=head_dim,
            N=aligned_seq_len,
            A_DRAM_ADDR=scaled_q_dram_addr,
            B_DRAM_ADDR=K_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=score_dram_addr,
            softmax_enable=True,
            C_DRAM_ADDR=BIAS_DRAM_ADDR,
            bias_mode="full_matrix",
            gpr_M_reg=gpr_batch_reg,
            gpr_K_reg=head_dim_reg,
            gpr_N_reg=gpr_aligned_seq_len_reg,
            gpr_a_addr=scaled_q_reg,
            gpr_b_addr=gpr_k_addr,
            gpr_out_addr=score_reg,
            gpr_c_addr=gpr_bias_addr,
        )
        total_flops += self.matmat_mul_core(
            M=batch,
            K=aligned_seq_len,
            N=head_dim,
            A_DRAM_ADDR=score_dram_addr,
            B_DRAM_ADDR=v_t_dram_addr,
            OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
            gpr_M_reg=gpr_batch_reg,
            gpr_K_reg=gpr_aligned_seq_len_reg,
            gpr_N_reg=head_dim_reg,
            gpr_a_addr=score_reg,
            gpr_b_addr=v_t_reg,
            gpr_out_addr=gpr_out_addr,
        )
        for _ in allocated_regs:
            self.release_isa_reg()
        print(f"unified_attention_core_dynamic: batch={batch}, aligned_seq_len={aligned_seq_len}, head_dim={head_dim}, FLOPS={total_flops / 1e9:.6f} G")
        return total_flops

    def quantized_matmat_core(self, M: int, K: int, N: int, A_DRAM_ADDR: int, B_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, SCALE_DRAM_ADDR: int, C_DRAM_ADDR: int = None, bias_mode: str = "broadcast_N", data_type: TYPE = None, gelu_enable: bool = False, silu_enable: bool = False, sigmoid_enable: bool = False, clamp_enable: bool = False, log_enable: bool = False, write_back_disable: bool = False,
                              gpr_M_reg: Optional[int] = None, gpr_K_reg: Optional[int] = None, gpr_N_reg: Optional[int] = None,
                              gpr_a_addr: Optional[int] = None, gpr_b_addr: Optional[int] = None,
                              gpr_out_addr: Optional[int] = None, gpr_scale_addr: Optional[int] = None) -> None:
        """Quantized matrix-matrix multiplication core (1-pass streaming quantized dot).

        Dispatches on whether runtime-dimension GPRs are supplied:

        - any of ``gpr_M_reg`` / ``gpr_K_reg`` / ``gpr_N_reg`` given: :meth:`quantized_matmat_core_dynamic`
          — **fully dynamic** (runtime M/K/N + GPR DRAM bases), still 1-pass. M==1 is the folded gemma
          decoder path: one captured body serves every layer via per-layer GPR addresses, and the
          1-pass streaming dot keeps it at legacy throughput (vs the 2-pass dequantize path of
          :meth:`matmat_mul_core_dynamic`); this M==1 path is HW-validated. M>1 uses a newer general
          M-tiling path (see :meth:`quantized_matmat_core_dynamic`) that is EXPERIMENTAL /
          HW-UNVALIDATED. Missing dimension GPRs are auto-allocated + seeded.
        - none given (default): :meth:`quantized_matmat_core_legacy` — the compile-time-tiled static path.

        Args:
            M: batch dimension (number of input vectors)
            K: inner dimension (vector length), must be a multiple of UE_VECTOR_SIZE
            N: outer dimension (matrix height), must be a multiple of UE_VECTOR_SIZE
            A_DRAM_ADDR: DRAM address of input matrix
            B_DRAM_ADDR: DRAM address of weight matrix
            OUTPUT_DRAM_ADDR: DRAM address of output matrix
            C_DRAM_ADDR: DRAM address of bias matrix
            bias_enable: enable bias
            bias_mode: bias mode
            data_type: data type
            SCALE_DRAM_ADDR: DRAM address of scale matrix
            gelu_enable: enable gelu
            silu_enable: enable silu
            clamp_enable: enable clamp (relu via clamp(x, 0, +inf))
            log_enable: enable log (log(clamp(x, 1e-3, +inf)))
            gpr_*_reg / gpr_*_addr: runtime dims / DRAM bases (dynamic path); see
                :meth:`quantized_matmat_core_dynamic`.
        """
        _addr_gprs = (gpr_a_addr, gpr_b_addr, gpr_out_addr, gpr_scale_addr)
        if gpr_M_reg is not None or gpr_K_reg is not None or gpr_N_reg is not None:
            if C_DRAM_ADDR is not None:
                raise ValueError("quantized_matmat_core: bias (C_DRAM_ADDR) not supported on the dynamic path")
            allocated = []
            if gpr_K_reg is None:
                gpr_K_reg = self.alloc_isa_reg(); self.generate_instruction_add_set(gpr_K_reg, K); allocated.append('K')
            if gpr_N_reg is None:
                gpr_N_reg = self.alloc_isa_reg(); self.generate_instruction_add_set(gpr_N_reg, N); allocated.append('N')
            if gpr_M_reg is None:
                gpr_M_reg = self.alloc_isa_reg(); self.generate_instruction_add_set(gpr_M_reg, M); allocated.append('M')
            flops = self.quantized_matmat_core_dynamic(
                M=M, K=K, N=N, A_DRAM_ADDR=A_DRAM_ADDR, B_DRAM_ADDR=B_DRAM_ADDR,
                OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR, SCALE_DRAM_ADDR=SCALE_DRAM_ADDR,
                gpr_M_reg=gpr_M_reg, gpr_K_reg=gpr_K_reg, gpr_N_reg=gpr_N_reg,
                gpr_a_addr=gpr_a_addr, gpr_b_addr=gpr_b_addr, gpr_out_addr=gpr_out_addr, gpr_scale_addr=gpr_scale_addr,
                data_type=data_type, gelu_enable=gelu_enable, silu_enable=silu_enable, sigmoid_enable=sigmoid_enable,
                clamp_enable=clamp_enable, log_enable=log_enable, write_back_disable=write_back_disable,
            )
            for _ in allocated:
                self.release_isa_reg()
            return flops
        if any(r is not None for r in _addr_gprs):
            raise ValueError("quantized_matmat_core: gpr_*_addr require a dimension GPR (set gpr_M_reg/gpr_K_reg/gpr_N_reg)")
        return self.quantized_matmat_core_legacy(
            M=M, K=K, N=N, A_DRAM_ADDR=A_DRAM_ADDR, B_DRAM_ADDR=B_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR, SCALE_DRAM_ADDR=SCALE_DRAM_ADDR,
            C_DRAM_ADDR=C_DRAM_ADDR, bias_mode=bias_mode, data_type=data_type,
            gelu_enable=gelu_enable, silu_enable=silu_enable, sigmoid_enable=sigmoid_enable,
            clamp_enable=clamp_enable, log_enable=log_enable, write_back_disable=write_back_disable,
        )

    def quantized_matmat_core_legacy(self, M: int, K: int, N: int, A_DRAM_ADDR: int, B_DRAM_ADDR: int,
                                     OUTPUT_DRAM_ADDR: int, SCALE_DRAM_ADDR: int, C_DRAM_ADDR: int = None,
                                     bias_mode: str = "broadcast_N", data_type: TYPE = None,
                                     gelu_enable: bool = False, silu_enable: bool = False,
                                     sigmoid_enable: bool = False, clamp_enable: bool = False,
                                     log_enable: bool = False, write_back_disable: bool = False) -> int:
        """Compile-time-tiled static quantized matmul (A @ Bᵀ), 1-pass streaming quantized dot.

        The default (no runtime-dimension GPRs) path of :meth:`quantized_matmat_core`.

        Per (M-tile, N-strip) the scale strip is loaded into scale BRAM **once** and reused across all
        of the tile's rows (the scales are identical for every row of an N-strip and the dot never
        mutates scale BRAM), matching :meth:`quantized_matmat_core_dynamic`'s M>1 path — the scale DMA
        count is 1 per strip rather than 1 per row.
        """
        bytes_per_element = 2

        bias_enable = False
        if C_DRAM_ADDR is not None:
            bias_enable = True

        if bias_enable:
            assert bias_mode in ("broadcast_N", "full_matrix"), f"bias_mode={bias_mode} must be either 'broadcast_N' or 'full_matrix'"

        if data_type in (TYPE.IF4, TYPE.IF8, TYPE.TQ4):
            assert SCALE_DRAM_ADDR is not None, "SCALE_DRAM_ADDR must be provided when data_type is IF4, IF8 or TQ4"

        assert sum([gelu_enable, silu_enable, sigmoid_enable, clamp_enable, log_enable]) <= 1, "only one activation can be True"

        lalu_mode = LALU_MODE.BYPASS
        lalu_a = 0
        lalu_b = 0
        if gelu_enable:
            lalu_mode = LALU_MODE.ACT
            lalu_a = LALU_ACT_GELU_A
            lalu_b = LALU_ACT_GELU_B
        elif silu_enable:
            lalu_mode = LALU_MODE.ACT
            lalu_a = LALU_ACT_SILU_A
            lalu_b = LALU_ACT_SILU_B
        elif sigmoid_enable:
            lalu_mode = LALU_MODE.ACT_NO_X
            lalu_a = LALU_ACT_SIGMOID_A
            lalu_b = LALU_ACT_SIGMOID_B
        elif clamp_enable:
            lalu_mode = LALU_MODE.CLAMP
            lalu_a = LALU_CLAMP_RELU_A
            lalu_b = LALU_CLAMP_RELU_B
        elif log_enable:
            lalu_mode = LALU_MODE.LOG
            lalu_a = LALU_LOG_A
            lalu_b = LALU_LOG_B

        # We put entire input matrix into URAM_A, and entire output matrix into URAM_B

        M_chunk = min(M, (URAM_FULL_ELEMENTS // K))
        N_chunk = min((SCALE_BRAM_ELEMENTS // K) * UE_VECTOR_SIZE,
                       N,
                     ((URAM_FULL_ELEMENTS // M_chunk) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE,
                     BIAS_BRAM_ELEMENTS)

        assert (N * K) % UE_VECTOR_SIZE == 0, f"(N * K={N * K}) must be a multiple of UE_VECTOR_SIZE={UE_VECTOR_SIZE}"

        print(f"M_chunk={M_chunk}, N_chunk={N_chunk}")


        max_clear_en = 1
        for M_chunk_idx, M_chunk_size in self.chunk_ranges(M, M_chunk):
            # transfer M_chunk_size x K elements to URAM_A
            self.accelerator_memory_to_sram(accelerator_dram_address=A_DRAM_ADDR + M_chunk_idx * K * bytes_per_element,
                                            sram_address=0x00000,
                                            element_size=M_chunk_size * K)

            for N_chunk_idx, N_take_size in self.chunk_ranges(N, N_chunk):

                if bias_enable and bias_mode == "broadcast_N":
                    self.accelerator_memory_to_bias_sram(accelerator_dram_address=C_DRAM_ADDR + N_chunk_idx * bytes_per_element,
                                                        element_size=N_take_size)

                # Scale strip and the quantized-B DRAM offset are invariant across the rows of this
                # N-strip (they depend only on N_chunk_idx / N_take_size, and the dot never mutates
                # scale BRAM), so load the scales ONCE per strip instead of once per row — matching
                # quantized_matmat_core_dynamic's M>1 path and cutting the scale DMA count by
                # M_chunk_size.
                self.accelerator_memory_to_scale_sram(accelerator_dram_address=SCALE_DRAM_ADDR + ((N_chunk_idx * K) // UE_VECTOR_SIZE) * bytes_per_element,
                                                    element_size=(N_take_size * K) // UE_VECTOR_SIZE)

                if data_type == TYPE.IF4 or data_type == TYPE.TQ4:
                    dma_offset = N_chunk_idx * K // 2
                elif data_type == TYPE.IF8:
                    dma_offset = N_chunk_idx * K
                else:
                    assert False, f"data_type={data_type} is not supported"

                for vector_idx in range(M_chunk_size): # for each input vector

                    if bias_enable and bias_mode == "full_matrix":
                        self.accelerator_memory_to_bias_sram(accelerator_dram_address=C_DRAM_ADDR + ((M_chunk_idx + vector_idx) * N + N_chunk_idx) * bytes_per_element,
                                                            element_size=N_take_size)

                    self.start_queue_for_dot_product_operation(max_clear_en=max_clear_en,
                                                                fmax_context_addr=0,
                                                                vector_sram_start_addr=0x00000 + vector_idx * K * bytes_per_element,
                                                                output_sram_wb_addr=0x80000 + vector_idx * N_take_size * bytes_per_element,
                                                                K=K,
                                                                N=N_take_size,
                                                                dma_start_addr=B_DRAM_ADDR + dma_offset,
                                                                data_type=data_type,
                                                                bias_enable=bias_enable,
                                                                lalu_mode=lalu_mode,
                                                                lalu_a=lalu_a,
                                                                lalu_b=lalu_b)
                    max_clear_en = 0

                if not write_back_disable:
                    self.sram_to_accelerator_memory(sram_address=0x80000,
                                                    accelerator_dram_address=OUTPUT_DRAM_ADDR + (M_chunk_idx * N + N_chunk_idx) * bytes_per_element,
                                                    element_size=M_chunk_size * N_take_size,
                                                    stride_bytes_per_chunk=N_take_size * bytes_per_element,
                                                    stride_jump_bytes=N * bytes_per_element)

        total_flops = 2 * M * N * K
        if bias_enable:
            total_flops += M * N
        if gelu_enable or silu_enable:
            total_flops += 4 * M * N
        if clamp_enable:
            total_flops += M * N
        if log_enable:
            total_flops += 2 * M * N
        return total_flops

    def quantized_matmat_core_dynamic(self, M: int, K: int, N: int,
                                      A_DRAM_ADDR: int, B_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, SCALE_DRAM_ADDR: int,
                                      gpr_M_reg: int, gpr_K_reg: int, gpr_N_reg: int,
                                      gpr_a_addr: Optional[int] = None, gpr_b_addr: Optional[int] = None,
                                      gpr_out_addr: Optional[int] = None, gpr_scale_addr: Optional[int] = None,
                                      data_type: TYPE = None, gelu_enable: bool = False, silu_enable: bool = False,
                                      sigmoid_enable: bool = False, clamp_enable: bool = False, log_enable: bool = False,
                                      clamp_min: float = 0.0, clamp_max: float = float("inf"),
                                      write_back_disable: bool = False) -> int:
        """Fully-dynamic (runtime M/K/N + GPR DRAM addresses) **1-pass** quantized matmul: A @ Bᵀ.

        Unlike :meth:`matmat_mul_core_dynamic` — which DEQUANTIZES B to bf16 in URAM and then does a
        bf16 dot (**two passes** over the weight bytes) — this streams the quantized IF4/IF8 weights
        straight through the DOT_PRODUCT unit (inline fp4→bf19 unpack), **one pass**, exactly like the
        fast static :meth:`quantized_matmat_core`.

        Two internal code paths, selected on the **compile-time** ``M`` value (the runtime ``gpr_M_reg``
        is still what the captured program actually uses to size the loop):

        - ``M == 1``: the decode single-token path for the folded gemma decoder. Because every DRAM
          base (A / B / output / scale) *and* the K/N dims are runtime GPRs, one captured body serves
          all 26 layers, and the 1-pass dot restores legacy throughput (the 2-pass dynamic core is the
          ~2x "streaming regression" root cause). Per N-strip (column tile that fits the scale BRAM):
          load the strip's per-block scales, emit one streaming ``UE_MODE.DOT_PRODUCT`` op whose DMA
          streams the quantized B strip from DRAM and dots it against the single A vector in URAM_A ->
          ``n_take`` outputs accumulated into a running URAM_B row cursor. One final DMA writes all
          ``N`` outputs back to DRAM (batched writeback). Only **two** loop-live PBI pointers
          (scale / dot); the A-load and final-writeback pointers live outside the loop. When the whole
          ``N`` fits one column tile (``single_strip``) the strip loop and its entry abs-jump (i-cache
          refill) are dropped entirely for a straight-line body. HW-validated (production decode path).

        - ``M > 1``: general runtime-M path, mirroring :meth:`matmat_mul_core_dynamic`'s M-tile /
          N-strip / per-row nested while-loop skeleton (URAM-capacity tiling, strided writeback via
          ``PBI_FIELD.STRIDE_JUMP``), but keeping the 1-pass streaming quantized dot instead of the
          2-pass dequantize-then-bf16-dot body — so it needs only **four** distinct PBI pointers
          (A-load / scale / dot / writeback) instead of matmat's five (no dequantize-to-URAM_B step).
          Per (M-tile, N-strip): the strip's scale block is loaded once (not once per row — a
          deliberate efficiency improvement over :meth:`quantized_matmat_core`'s per-row reload, safe
          since the dot never mutates scale BRAM), then each of the tile's rows issues its own
          streaming DOT_PRODUCT call (B is re-streamed from DRAM per row, matching the static core's
          per-row behavior — the 1-pass design has no on-chip B cache to reuse across rows). Output
          rows land contiguously in URAM_B for the tile/strip and are flushed with one strided DMA.
          No single-tile/single-strip fast path in this first cut (always the full loop nest).
          **Experimental / HW-unvalidated** — new as of this addition; start with small M on real HW.

        ``gpr_M_reg`` / ``gpr_K_reg`` / ``gpr_N_reg`` are the runtime M / inner / outer dims.
        ``gpr_a_addr`` / ``gpr_b_addr`` / ``gpr_out_addr`` / ``gpr_scale_addr`` source the DRAM bases
        from GPRs (word addr = byte>>3); omit any to bake its literal base. One activation
        (gelu/silu/sigmoid/clamp/log) may be fused into the dot's LALU, identical to
        :meth:`quantized_matmat_core`. Bias is not supported on either path.
        """
        fn = "quantized_matmat_core_dynamic"
        assert self.is_capture_on, f"{fn}() requires active capture"
        if data_type not in (TYPE.IF4, TYPE.IF8, TYPE.TQ4):
            raise ValueError(f"{fn}: data_type must be IF4/IF8/TQ4, got {data_type!r}")
        if M < 1:
            raise ValueError(f"{fn}: M must be >= 1, got M={M}")
        if K % UE_VECTOR_SIZE != 0 or N % UE_VECTOR_SIZE != 0:
            raise ValueError(f"{fn}: K={K} and N={N} must be multiples of {UE_VECTOR_SIZE}")
        for _nm, _rg in (("gpr_M_reg", gpr_M_reg), ("gpr_K_reg", gpr_K_reg), ("gpr_N_reg", gpr_N_reg)):
            if _rg is None or not (1 <= _rg <= 63):
                raise ValueError(f"{fn}: {_nm} must be a GPR index 1..63, got {_rg}")
        if len({gpr_M_reg, gpr_K_reg, gpr_N_reg}) != 3:
            raise ValueError(f"{fn}: gpr_M_reg/gpr_K_reg/gpr_N_reg must be distinct")
        self._validate_addr_gprs(fn, gpr_M_reg, {
            "gpr_a_addr": gpr_a_addr, "gpr_b_addr": gpr_b_addr,
            "gpr_out_addr": gpr_out_addr, "gpr_scale_addr": gpr_scale_addr})
        for _nm, _rg in (("gpr_a_addr", gpr_a_addr), ("gpr_b_addr", gpr_b_addr),
                         ("gpr_out_addr", gpr_out_addr), ("gpr_scale_addr", gpr_scale_addr)):
            if _rg is not None and _rg in (gpr_K_reg, gpr_N_reg):
                raise ValueError(f"{fn}: {_nm}={_rg} collides with gpr_K_reg/gpr_N_reg")
        if sum([gelu_enable, silu_enable, sigmoid_enable, clamp_enable, log_enable]) > 1:
            raise ValueError(f"{fn}: at most one activation may be enabled")
        # Template feasibility: one 64-wide column strip's scales (K_rows blocks) must fit the scale BRAM.
        K_rows_ct = K // UE_VECTOR_SIZE
        if K_rows_ct == 0 or (SCALE_BRAM_ELEMENTS // K_rows_ct) < UE_VECTOR_SIZE:
            raise ValueError(f"{fn}: K={K} too large — a 64-wide column strip's scales exceed the scale BRAM")

        # Activation -> LALU (baked into the dot descriptor; identical mapping to quantized_matmat_core).
        lalu_mode, lalu_a, lalu_b = LALU_MODE.BYPASS, 0, 0
        if gelu_enable:
            lalu_mode, lalu_a, lalu_b = LALU_MODE.ACT, LALU_ACT_GELU_A, LALU_ACT_GELU_B
        elif silu_enable:
            lalu_mode, lalu_a, lalu_b = LALU_MODE.ACT, LALU_ACT_SILU_A, LALU_ACT_SILU_B
        elif sigmoid_enable:
            lalu_mode, lalu_a, lalu_b = LALU_MODE.ACT_NO_X, LALU_ACT_SIGMOID_A, LALU_ACT_SIGMOID_B
        elif clamp_enable:
            lalu_mode, lalu_a, lalu_b = LALU_MODE.CLAMP, self.float_to_bf16(clamp_min), self.float_to_bf16(clamp_max)
        elif log_enable:
            lalu_mode, lalu_a, lalu_b = LALU_MODE.LOG, LALU_LOG_A, LALU_LOG_B

        SRAM_A = 0x00000
        SRAM_B = 0x80000
        _, a_urow = self.sram_address_to_uram_address(SRAM_A)
        out_type, out_urow = self.sram_address_to_uram_address(SRAM_B)
        A_BASE_W = ue_35bit_addr_shifter(A_DRAM_ADDR)
        B_BASE_W = ue_35bit_addr_shifter(B_DRAM_ADDR)
        OUT_BASE_W = ue_35bit_addr_shifter(OUTPUT_DRAM_ADDR)
        SCALE_BASE_W = ue_35bit_addr_shifter(SCALE_DRAM_ADDR)
        # Quantized B stream bytes per (col, block): IF4/TQ4 -> 32 (= K/2 per col over K_rows blocks); IF8 -> 64.
        b_dma_shift = 5 if data_type in (TYPE.IF4, TYPE.TQ4) else 6   # n_take*K_rows << b_dma_shift bytes
        b_word_shift = b_dma_shift - 3                                 # bytes>>3 -> words (IF4:2, IF8:3)

        if M == 1:
            _alloc_list = []
            def _alloc():
                r = self.alloc_isa_reg(); _alloc_list.append(r); return r
            K_rows_reg    = _alloc()   # K/64 (dot depth)
            N_chunk_reg   = _alloc()   # column strip width (scale-BRAM bound, 64-aligned)
            N_counter_reg = _alloc()   # remaining columns
            n_take_reg    = _alloc()   # min(remaining, N_chunk)
            b_dram_reg    = _alloc()   # running quantized-B strip DRAM word cursor
            scale_dram_reg = _alloc()  # running scale strip DRAM word cursor
            out_urow_reg  = _alloc()   # running URAM_B output-row cursor (batched writeback)
            s1 = _alloc()
            s2 = _alloc()

            def _set(ptr, field, reg):
                self.generate_instruction_pbi_inc(general_reg_src=reg, pbi_field_select=field, inst_pointer_idx=ptr)
            def _seed(dst, lit_w, gpr_base):
                if gpr_base is not None:
                    self.generate_instruction_add_imm(src_reg_idx=gpr_base, immediate_value=0, dst_reg_idx=dst)
                else:
                    self.generate_instruction_add_set(dst, lit_w)

            # Compile-time N_chunk (the runtime tiling reproduces this from gpr_K_reg). ``single_strip``
            # collapses to one straight-line strip when the whole N fits one column tile — no entry
            # abs-jump / i-cache refill and no strip loop. Correct while runtime N <= this N_chunk, which
            # holds when runtime N == the compile-time template (the gemma constant-dim decode contract;
            # every dim it passes is a model constant equal to the baked value).
            ct_N_chunk = min(N, (SCALE_BRAM_ELEMENTS // K_rows_ct // UE_VECTOR_SIZE) * UE_VECTOR_SIZE)
            single_strip = (N <= ct_N_chunk)

            # ----- Phase 1: runtime tiling. K_rows is always needed; N_chunk (3 startup divides) only
            # for the multi-strip loop, so single_strip skips it. -----
            self.generate_instruction_shr(K_rows_reg, gpr_K_reg, 6)        # K_rows = K/64 (dot depth)
            if not single_strip:
                self.generate_instruction_add_set(s1, SCALE_BRAM_ELEMENTS)
                self.generate_instruction_div_reg(s1, s1, K_rows_reg)      # max cols such that cols*K_rows <= scale BRAM
                self.generate_instruction_shr(N_chunk_reg, s1, 6)
                self.generate_instruction_shl(N_chunk_reg, N_chunk_reg, 6) # align down to a multiple of 64
                self.generate_instruction_reg_min(N_chunk_reg, N_chunk_reg, gpr_N_reg)

            # ----- Phase 2: one-shot A vector (1xK) load into URAM_A (temp pointer, released before loop) -----
            ptr_A = self.alloc_inst_ptr()
            self.generate_instruction_pbi_init(dram_shared_addr=A_DRAM_ADDR, uram_dst_addr=a_urow, inst_pointer_idx=ptr_A)
            if gpr_a_addr is not None:
                self._pbi_override_dram_base_from_gpr(ptr_A, gpr_a_addr)
            self.generate_instruction_shl(s1, gpr_K_reg, 1)               # K*2 bytes (M==1)
            _set(ptr_A, PBI_FIELD.DMA_LENGTH, s1)
            self.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=SRAM_A, element_size=0, inst_pointer_idx=ptr_A)
            self.release_inst_ptr(ptr_A)

            # ----- Phase 3: loop-live pointers (scale + dot only; the writeback is a single DMA after
            # the loop, so its pointer is allocated later). Each strip's dot accumulates into a running
            # URAM_B row cursor; ONE final DMA writes all N outputs back to DRAM. Batching the writeback
            # replaces up to ~18 tiny per-strip DMAs with one large one — the small-transfer overhead is
            # invisible on a 256b AXI bus but decisive on a 512b bus (wide-AXI boards underutilise the
            # bus on sub-beat transfers). Loop body now touches 2 PBI pointers (was 3). -----
            ptr_scale = self.alloc_inst_ptr()
            ptr_dot   = self.alloc_inst_ptr()
            self.generate_instruction_pbi_init(dram_shared_addr=SCALE_DRAM_ADDR, inst_pointer_idx=ptr_scale)
            # dot descriptor baked once (A vector row 0 -> URAM_B at the out-row cursor, activation via
            # LALU); DRAM_ADDR / DMA_LENGTH / OUTPUT_SIZE / URAM_ROW_SIZE / URAM_WRITEB_ADDR come from the
            # pointer each strip.
            self.generate_instruction_pbi_init(dram_shared_addr=B_DRAM_ADDR, uram_a_start_addr=a_urow,
                                               uram_wb_addr=out_urow, inst_pointer_idx=ptr_dot)
            _seed(b_dram_reg, B_BASE_W, gpr_b_addr)
            _seed(scale_dram_reg, SCALE_BASE_W, gpr_scale_addr)
            self.generate_instruction_add_set(out_urow_reg, out_urow)     # running URAM_B output row

            def _emit_strip(nt_reg, advance):
                """One column strip: load its per-block scales, stream-dot the B strip vs A into the
                URAM_B out-row cursor. When ``advance`` (multi-strip loop), also bump the B/scale DRAM
                cursors and the URAM out-row for the next strip."""
                # (1) scale strip -> quant/scale BRAM (nt * K_rows blocks, 2 bytes each)
                self.generate_instruction_mul32_shl_reg(s1, nt_reg, K_rows_reg, 1)
                _set(ptr_scale, PBI_FIELD.DRAM_ADDR, scale_dram_reg)
                _set(ptr_scale, PBI_FIELD.DMA_LENGTH, s1)
                self.ue_memcpy_from_dram(0, 0, MEMCPY_TYPE.BRAM.value, 0, 0, inst_pointer_idx=ptr_scale)
                # (2) 1-pass streaming dot: DMA streams the B strip (nt*K/2 bytes) vs A -> URAM_B out-row
                self.generate_instruction_mul32_shl_reg(s1, nt_reg, K_rows_reg, b_dma_shift)
                _set(ptr_dot, PBI_FIELD.DRAM_ADDR, b_dram_reg)
                _set(ptr_dot, PBI_FIELD.DMA_LENGTH, s1)
                _set(ptr_dot, PBI_FIELD.OUTPUT_SIZE, nt_reg)
                _set(ptr_dot, PBI_FIELD.URAM_ROW_SIZE, K_rows_reg)
                _set(ptr_dot, PBI_FIELD.URAM_WRITEB_ADDR, out_urow_reg)
                self.ue_arithmetic_op(
                    0, 1, 1, lalu_a, lalu_b, lalu_mode.value, 0,
                    out_type.value, 0, out_urow, URAM_WRITE_SRC.URAM_WRITE_BACK.value,
                    UE_MODE.DOT_PRODUCT, data_type.value,
                    a_urow, 0, 0, 0, 0, 0, inst_pointer_idx=ptr_dot)
                if advance:
                    # B += nt*K/2 words ; scale += nt*K_rows*2 words ; out-row += nt/64 rows
                    self.generate_instruction_mul32_shl_reg(s1, nt_reg, K_rows_reg, b_word_shift)
                    self.generate_instruction_add_reg(b_dram_reg, b_dram_reg, s1)
                    self.generate_instruction_mul32_shr_reg(s1, nt_reg, K_rows_reg, 2)   # nt*K_rows*2>>3
                    self.generate_instruction_add_reg(scale_dram_reg, scale_dram_reg, s1)
                    self.generate_instruction_shr(s1, nt_reg, 6)                          # nt/64 rows
                    self.generate_instruction_add_reg(out_urow_reg, out_urow_reg, s1)

            def _emit_final_wb():
                """Single DMA: all N outputs (URAM_B base row) -> OUTPUT DRAM base (N*2 bytes)."""
                if write_back_disable:
                    return
                ptr_wb = self.alloc_inst_ptr()
                self.generate_instruction_pbi_init(dram_shared_addr=OUTPUT_DRAM_ADDR, inst_pointer_idx=ptr_wb)
                if gpr_out_addr is not None:
                    self._pbi_override_dram_base_from_gpr(ptr_wb, gpr_out_addr)
                self.generate_instruction_shl(s1, gpr_N_reg, 1)                           # N*2 bytes
                _set(ptr_wb, PBI_FIELD.DMA_LENGTH, s1)
                self.ue_memcpy_to_dram(memcpy_type=MEMCPY_TYPE.URAM.value, uram_type=out_type.value,
                                       uram_src_addr=out_urow, dram_dst_addr=0, memcpy_length_bytes=0,
                                       inst_pointer_idx=ptr_wb)
                self.release_inst_ptr(ptr_wb)

            if single_strip:
                # ----- Single-strip straight-line: no loop, no entry abs-jump (i-cache refill). -----
                _emit_strip(gpr_N_reg, advance=False)
                _emit_final_wb()
                print(f"{fn}: single-strip straight-line (M=1, K=GPR[{gpr_K_reg}], N=GPR[{gpr_N_reg}] "
                      f"<= N_chunk={ct_N_chunk}; no abs-jump, batched writeback, 2 loop PBI pointers)")
            else:
                # ----- Phase 4: N-strip while-loop (entry abs-jump anchors the resident backward
                # loop-back); the batched writeback fires ONCE after the loop. -----
                program_dram_start_addr = self.get_program_dram_addr()
                cur = self.capture_count
                self.generate_instruction_jump_abs(ue_35bit_addr_shifter(
                    program_dram_start_addr + (cur + 1) * INSTRUCTION_SIZE_BYTES))
                self.generate_instruction_add_imm(src_reg_idx=gpr_N_reg, immediate_value=0, dst_reg_idx=N_counter_reg)
                self.generate_instruction_reg_min(n_take_reg, N_counter_reg, N_chunk_reg)
                body_start = self.capture_count
                _emit_strip(n_take_reg, advance=True)
                # N-counter update / loop back
                self.generate_instruction_reg_sub(N_counter_reg, N_counter_reg, n_take_reg)
                self.generate_instruction_reg_min(n_take_reg, N_counter_reg, N_chunk_reg)
                loop_sz = self.capture_count - body_start + 2
                self.generate_instruction_jump_rela_jnz(loop_sz, N_counter_reg)
                assert loop_sz <= 256, f"{fn}: N-loop body {loop_sz} exceeds i-cache budget 256"
                _emit_final_wb()
                print(f"{fn}: N-loop body={loop_sz} (M=1, K=GPR[{gpr_K_reg}], N=GPR[{gpr_N_reg}] runtime, "
                      f"1-pass streaming quantized dot, batched writeback, 2 loop PBI pointers)")

            self.release_inst_ptr(ptr_dot)
            self.release_inst_ptr(ptr_scale)
            for _ in range(len(_alloc_list)):
                self.release_isa_reg()

        else:
            # ================================================================================
            # General M>1 path: M-tile / N-strip / per-row nested runtime while-loop, mirroring
            # matmat_mul_core_dynamic's tiling skeleton (URAM-capacity tiling + strided writeback
            # via PBI_FIELD.STRIDE_JUMP), but keeping the 1-pass streaming quantized dot instead of
            # matmat's 2-pass dequantize-then-bf16-dot body. Needs only 4 distinct PBI pointers
            # (A-load / scale / dot / writeback) since there is no dequantize-to-URAM_B step.
            # EXPERIMENTAL / HW-UNVALIDATED (see class docstring for the M==1 path, which is
            # untouched and remains the HW-validated production decode path).
            # ================================================================================
            _alloc_list = []
            def _alloc():
                r = self.alloc_isa_reg(); _alloc_list.append(r); return r

            K_rows_reg      = _alloc()   # K/64 (dot depth)
            M_chunk_reg     = _alloc()   # rows per M-tile (URAM_A capacity bound)
            N_chunk_reg     = _alloc()   # column strip width (scale-BRAM + output-capacity bound, 64-aligned)
            gpr_M_counter   = _alloc()   # remaining M rows
            m_take_reg      = _alloc()   # min(remaining, M_chunk); reused as the row-loop counter
            m_tile_rows_reg = _alloc()   # m_take snapshot that survives the N-strip/row loops
            rows_done_reg   = _alloc()   # M rows completed before the current M-tile
            N_counter_reg   = _alloc()   # remaining N columns within this M-tile
            n_take_reg      = _alloc()   # min(remaining, N_chunk)
            n_take_rows_reg = _alloc()   # n_take/64 (N is always 64-aligned; see feasibility check above)
            a_row_reg       = _alloc()   # URAM_A row cursor for the current row within the tile
            out_row_reg     = _alloc()   # URAM_B row cursor for the current row's output
            a_dram_reg      = _alloc()   # running A DRAM word cursor
            b_dram_reg      = _alloc()   # running quantized-B strip DRAM word cursor (reset per tile)
            scale_dram_reg  = _alloc()   # running scale strip DRAM word cursor (reset per tile)
            k_row_word_stride_reg = _alloc()  # K_rows << 4 (A-matrix row word-stride)
            n_stride_bytes_reg    = _alloc()  # N*2 bytes (DRAM row stride for the strided writeback)
            s1 = _alloc()
            s2 = _alloc()
            s3 = _alloc()

            def _set(ptr, field, reg):
                self.generate_instruction_pbi_inc(general_reg_src=reg, pbi_field_select=field, inst_pointer_idx=ptr)
            def _seed_cursor(cursor_reg, lit_base_w, gpr_base):
                if gpr_base is not None:
                    self.generate_instruction_add_imm(src_reg_idx=gpr_base, immediate_value=0, dst_reg_idx=cursor_reg)
                else:
                    self.generate_instruction_add_set(cursor_reg, lit_base_w)
            def _off_plus_base(off_reg, lit_base_w, gpr_base, dst_reg):
                if gpr_base is not None:
                    self.generate_instruction_add_reg(dst_reg, off_reg, gpr_base)
                else:
                    self.generate_instruction_add_imm(src_reg_idx=off_reg, immediate_value=lit_base_w, dst_reg_idx=dst_reg)

            # ----- Phase 1: tiling arithmetic (mirrors quantized_matmat_core's static formula:
            # M_chunk bound by URAM_A capacity, N_chunk bound by scale BRAM AND by the output tile
            # (M_chunk x N_chunk) fitting URAM_B) -----
            self.generate_instruction_shr(K_rows_reg, gpr_K_reg, 6)                    # K_rows = K/64
            self.generate_instruction_add_set(s1, URAM_FULL_ELEMENTS)
            self.generate_instruction_div_reg(M_chunk_reg, s1, gpr_K_reg)              # URAM_FULL // K
            self.generate_instruction_reg_min(M_chunk_reg, M_chunk_reg, gpr_M_reg)

            self.generate_instruction_add_set(s1, SCALE_BRAM_ELEMENTS)
            self.generate_instruction_div_reg(s1, s1, K_rows_reg)                      # scale-BRAM bound (elements)
            self.generate_instruction_shr(N_chunk_reg, s1, 6)
            self.generate_instruction_shl(N_chunk_reg, N_chunk_reg, 6)                 # aligned down to 64

            self.generate_instruction_add_set(s2, URAM_FULL_ELEMENTS)
            self.generate_instruction_div_reg(s2, s2, M_chunk_reg)                     # URAM_FULL // M_chunk
            self.generate_instruction_shr(s2, s2, 6)
            self.generate_instruction_shl(s2, s2, 6)
            self.generate_instruction_reg_min(N_chunk_reg, N_chunk_reg, s2)
            self.generate_instruction_reg_min(N_chunk_reg, N_chunk_reg, gpr_N_reg)

            self.generate_instruction_add_imm(src_reg_idx=gpr_M_reg, immediate_value=0, dst_reg_idx=gpr_M_counter)
            self.generate_instruction_reg_min(m_take_reg, gpr_M_counter, M_chunk_reg)

            self.generate_instruction_shl(k_row_word_stride_reg, K_rows_reg, 4)        # K_rows*16 words/A-row
            self.generate_instruction_shl(n_stride_bytes_reg, gpr_N_reg, 1)            # N*2 bytes

            _seed_cursor(a_dram_reg, A_BASE_W, gpr_a_addr)

            # ----- Phase 2: PBI pointer inits (outside all loops; 4 distinct pointers total) -----
            ptr_A     = self.alloc_inst_ptr()
            ptr_scale = self.alloc_inst_ptr()
            ptr_dot   = self.alloc_inst_ptr()
            ptr_wb    = self.alloc_inst_ptr()
            self.generate_instruction_pbi_init(dram_shared_addr=A_DRAM_ADDR, uram_dst_addr=a_urow, inst_pointer_idx=ptr_A)
            self.generate_instruction_pbi_init(dram_shared_addr=SCALE_DRAM_ADDR, inst_pointer_idx=ptr_scale)
            self.generate_instruction_pbi_init(dram_shared_addr=B_DRAM_ADDR, inst_pointer_idx=ptr_dot)
            self.generate_instruction_pbi_init(dram_shared_addr=OUTPUT_DRAM_ADDR, inst_pointer_idx=ptr_wb)
            _set(ptr_wb, PBI_FIELD.STRIDE_JUMP, n_stride_bytes_reg)

            # ----- Phase 3: M-tile while-loop (abs-jump anchors the resident backward loop-back) -----
            program_dram_start_addr = self.get_program_dram_addr()
            self.generate_instruction_jump_abs(ue_35bit_addr_shifter(
                program_dram_start_addr + (self.capture_count + 1) * INSTRUCTION_SIZE_BYTES))
            m_body_start = self.capture_count

            self.generate_instruction_reg_sub(rows_done_reg, gpr_M_reg, gpr_M_counter)
            self.generate_instruction_reg_min(m_take_reg, gpr_M_counter, M_chunk_reg)
            self.generate_instruction_add_imm(src_reg_idx=m_take_reg, immediate_value=0, dst_reg_idx=m_tile_rows_reg)

            # A-tile load: m_take*K*2 bytes -> URAM_A row 0
            _set(ptr_A, PBI_FIELD.DRAM_ADDR, a_dram_reg)
            self.generate_instruction_mul32_shl_reg(s1, m_tile_rows_reg, K_rows_reg, 7)   # m_take*K*2 bytes
            _set(ptr_A, PBI_FIELD.DMA_LENGTH, s1)
            self.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=SRAM_A, element_size=0,
                                            memcpy_length_bytes=0, inst_pointer_idx=ptr_A)
            self.generate_instruction_mul32_reg(s1, m_tile_rows_reg, k_row_word_stride_reg)
            self.generate_instruction_add_reg(a_dram_reg, a_dram_reg, s1)

            # ----- N-strip while-loop init (reset each M-tile) -----
            _seed_cursor(b_dram_reg, B_BASE_W, gpr_b_addr)
            _seed_cursor(scale_dram_reg, SCALE_BASE_W, gpr_scale_addr)
            self.generate_instruction_add_imm(src_reg_idx=gpr_N_reg, immediate_value=0, dst_reg_idx=N_counter_reg)
            self.generate_instruction_reg_min(n_take_reg, N_counter_reg, N_chunk_reg)

            n_body_start = self.capture_count

            # scale-strip load -> BRAM, once per strip (not per row: safe, BRAM isn't touched by the dot)
            _set(ptr_scale, PBI_FIELD.DRAM_ADDR, scale_dram_reg)
            self.generate_instruction_mul32_shl_reg(s1, n_take_reg, K_rows_reg, 1)
            _set(ptr_scale, PBI_FIELD.DMA_LENGTH, s1)
            self.ue_memcpy_from_dram(0, 0, MEMCPY_TYPE.BRAM.value, 0, 0, inst_pointer_idx=ptr_scale)

            # dot descriptor: constant for every row in this strip (B is re-streamed per row below,
            # matching quantized_matmat_core's per-row behavior -- the 1-pass dot has no on-chip B cache)
            _set(ptr_dot, PBI_FIELD.DRAM_ADDR, b_dram_reg)
            self.generate_instruction_mul32_shl_reg(s1, n_take_reg, K_rows_reg, b_dma_shift)
            _set(ptr_dot, PBI_FIELD.DMA_LENGTH, s1)
            _set(ptr_dot, PBI_FIELD.OUTPUT_SIZE, n_take_reg)
            _set(ptr_dot, PBI_FIELD.URAM_ROW_SIZE, K_rows_reg)
            self.generate_instruction_shr(n_take_rows_reg, n_take_reg, 6)   # exact: n_take always 64-aligned

            # ----- M-row dot loop (inner) -----
            self.generate_instruction_add_set(a_row_reg, a_urow)
            self.generate_instruction_add_set(out_row_reg, out_urow)
            row_body = self.capture_count
            _set(ptr_dot, PBI_FIELD.URAM_START_ADDR_Y, a_row_reg)
            _set(ptr_dot, PBI_FIELD.URAM_WRITEB_ADDR, out_row_reg)
            self.ue_arithmetic_op(
                0, 1, 1, lalu_a, lalu_b, lalu_mode.value, 0,
                out_type.value, 0, out_urow, URAM_WRITE_SRC.URAM_WRITE_BACK.value,
                UE_MODE.DOT_PRODUCT, data_type.value,
                a_urow, 0, 0, 0, 0, 0, inst_pointer_idx=ptr_dot)
            self.generate_instruction_add_reg(a_row_reg, a_row_reg, K_rows_reg)
            self.generate_instruction_add_reg(out_row_reg, out_row_reg, n_take_rows_reg)
            row_loop_sz = self.capture_count - row_body + 2
            self.generate_instruction_add_dec(m_take_reg)
            self.generate_instruction_jump_rela_jnz(row_loop_sz, m_take_reg)
            assert row_loop_sz <= 256, f"{fn}: M-row loop body {row_loop_sz} exceeds i-cache budget 256"
            self.generate_instruction_add_imm(src_reg_idx=m_tile_rows_reg, immediate_value=0, dst_reg_idx=m_take_reg)

            # advance B/scale cursors for the next strip
            self.generate_instruction_mul32_shl_reg(s1, n_take_reg, K_rows_reg, b_word_shift)
            self.generate_instruction_add_reg(b_dram_reg, b_dram_reg, s1)
            self.generate_instruction_mul32_shr_reg(s1, n_take_reg, K_rows_reg, 2)
            self.generate_instruction_add_reg(scale_dram_reg, scale_dram_reg, s1)

            # ----- strided writeback for this (M-tile, N-strip) -----
            if not write_back_disable:
                self.generate_instruction_mul32_shl_reg(s1, m_tile_rows_reg, n_take_reg, 1)   # m_take*n_take*2 (total)
                _set(ptr_wb, PBI_FIELD.DMA_LENGTH, s1)
                self.generate_instruction_shl(s2, n_take_reg, 1)                               # n_take*2 (chunk)
                _set(ptr_wb, PBI_FIELD.OUTPUT_SIZE, s2)
                self.generate_instruction_mul32_reg(s1, rows_done_reg, gpr_N_reg)
                self.generate_instruction_reg_sub(s2, gpr_N_reg, N_counter_reg)                # cols_done
                self.generate_instruction_add_reg(s1, s1, s2)
                self.generate_instruction_shr(s1, s1, 2)                                       # -> words
                _off_plus_base(s1, OUT_BASE_W, gpr_out_addr, s1)
                _set(ptr_wb, PBI_FIELD.DRAM_ADDR, s1)
                self.generate_instruction_add_set(s3, out_urow)
                _set(ptr_wb, PBI_FIELD.URAM_START_ADDR_Y, s3)
                _set(ptr_wb, PBI_FIELD.URAM_START_ADDR_Z, s3)
                self.ue_memcpy_to_dram(memcpy_type=MEMCPY_TYPE.URAM.value, uram_type=out_type.value,
                                       uram_src_addr=0, dram_dst_addr=0, memcpy_length_bytes=0,
                                       inst_pointer_idx=ptr_wb, pbi_stride_en=True)

            # ----- N-counter update / loop back -----
            self.generate_instruction_reg_sub(N_counter_reg, N_counter_reg, n_take_reg)
            self.generate_instruction_reg_min(n_take_reg, N_counter_reg, N_chunk_reg)
            n_loop_sz = self.capture_count - n_body_start + 2
            self.generate_instruction_jump_rela_jnz(n_loop_sz, N_counter_reg)
            assert n_loop_sz <= 512, f"{fn}: N-strip loop body {n_loop_sz} exceeds i-cache budget 512"

            # ----- M-tile counter update / loop back -----
            self.generate_instruction_add_imm(src_reg_idx=m_tile_rows_reg, immediate_value=0, dst_reg_idx=m_take_reg)
            self.generate_instruction_reg_sub(gpr_M_counter, gpr_M_counter, m_take_reg)
            self.generate_instruction_reg_min(m_take_reg, gpr_M_counter, M_chunk_reg)
            m_loop_sz = self.capture_count - m_body_start + 1
            self.generate_instruction_jump_rela_jnz(m_loop_sz, gpr_M_counter)
            assert m_loop_sz <= 512, f"{fn}: M-tile loop body {m_loop_sz} exceeds i-cache budget 512"

            print(f"{fn}: general M>1 path, M-tile/N-strip/row loop body={m_loop_sz} "
                  f"(M=GPR[{gpr_M_reg}], K=GPR[{gpr_K_reg}], N=GPR[{gpr_N_reg}] runtime, "
                  f"1-pass streaming quantized dot, strided writeback, 4 PBI pointers) "
                  f"[EXPERIMENTAL / HW-UNVALIDATED]")

            for ptr in (ptr_wb, ptr_dot, ptr_scale, ptr_A):
                self.release_inst_ptr(ptr)
            for _ in range(len(_alloc_list)):
                self.release_isa_reg()

        total_flops = 2 * M * N * K
        if gelu_enable or silu_enable or sigmoid_enable:
            total_flops += 4 * M * N
        elif clamp_enable:
            total_flops += M * N
        elif log_enable:
            total_flops += 2 * M * N
        return total_flops

    def _emit_pbi_scatter_per_token(self, *, read_base, read_stride_bytes,
                                    write_specs, sram_byte_addr, element_count,
                                    gpr_seq_len, template_seq_len: int = 0):
        """Emit one PBI runtime loop that staged-copies one ``element_count``-row
        per outer iteration from ``read_base`` to each (base, stride) in
        ``write_specs``. The outer trip count is taken from GPR ``gpr_seq_len`` so
        the body executes exactly ``actual_seq_len`` times at runtime — making
        the captured bin truly seq_len-agnostic up to ``MAX_CONTEXT_SIZE``.

        The read side computes addresses with ``reg_mul_imm`` + ``add_imm`` and
        applies them to a PBI pointer with a ``PBI_MODE_REG`` DRAM-field override.
        The write side uses ``pbi_init`` pointers + per-call DRAM-delta DMAs.

        Per-iteration t-counter is a locally-allocated GPR that increments by 1
        at end-of-body via ``add_inc``. Released after the loop.
        """
        bpe = self.bytes_per_element
        bytes_per_call = element_count * bpe
        _, sram_words = self.sram_address_to_uram_address(sram_byte_addr)

        # Allocate the read pointer and write pointers (one per destination stream).
        ptr_r = self.alloc_inst_ptr()
        self.generate_instruction_pbi_init(
            dram_shared_addr=read_base,
            dma_length=bytes_per_call,
            uram_dst_addr=sram_words,
            inst_pointer_idx=ptr_r,
        )
        ptr_ws = [self.alloc_inst_ptr() for _ in write_specs]
        for ptr_w, (dst_base, _stride) in zip(ptr_ws, write_specs):
            self.generate_instruction_pbi_init(
                dram_shared_addr=dst_base,
                dma_length=bytes_per_call,
                uram_a_start_addr=sram_words,
                uram_b_start_addr=sram_words,
                inst_pointer_idx=ptr_w,
            )

        # Per-token counter t (0, 1, 2, ...) — used to compute read DRAM addr.
        t_reg = self.alloc_isa_reg()
        self.generate_instruction_add_set(t_reg, 0)

        self.loop_start(loop_cnt=template_seq_len, gpr_loop_cnt=gpr_seq_len)
        # Read DRAM addr = read_base + t * read_stride_bytes
        self.generate_instruction_reg_mul_imm(
            self.TMP_REG, t_reg, ue_35bit_addr_shifter(read_stride_bytes))
        self.generate_instruction_add_imm(
            self.TMP_REG, ue_35bit_addr_shifter(read_base), self.TMP_REG)
        # Replace the read pointer's DRAM field with the runtime-computed address.
        self._pbi_override_dram_base_from_gpr(ptr_r, self.TMP_REG)
        self.accelerator_memory_to_sram(
            accelerator_dram_address=0,
            sram_address=0,
            element_size=element_count,
            inst_pointer_idx=ptr_r,
            memcpy_length_bytes=0,
        )
        # SRAM→DRAM via PBI pointers (each advances its DRAM addr by its stride).
        for ptr_w, (_base, dst_stride) in zip(ptr_ws, write_specs):
            self.sram_to_accelerator_memory(
                sram_address=0,
                accelerator_dram_address=dst_stride,
                element_size=element_count,
                inst_pointer_idx=ptr_w,
                memcpy_length_bytes=0,
            )
        # t += 1
        self.generate_instruction_add_inc(t_reg)
        self.loop_end()

        self.release_isa_reg()  # t_reg
        for ptr in reversed(ptr_ws):
            self.release_inst_ptr(ptr)
        self.release_inst_ptr(ptr_r)
            
    def start_capture(self):
        """Start capturing instructions instead of executing them"""
        self._inst_id = 0
        self.capture_count = 0
        self.capture_buffer = []
        self._capture_loop_stack = []
        self.is_capture_on = True
        print(f"Capture started. Buffer initialized, count={self.capture_count}")

    def stop_capture(self):
        """Stop capturing instructions"""
        self.is_capture_on = False
        print(f"Capture stopped. Total instructions captured: {self.capture_count}, size: {self.capture_count * 32} bytes")

    def clear_dram(self, chunk_size_bytes: int = 64 * 1024 * 1024) -> None:
        dram_total_bytes = DRAM_END_ADDR - DRAM_START_ADDR + 1
        fill = b'\xff' * chunk_size_bytes
        offset = 0
        bar_width = 40
        print(f"Clearing DRAM [{hex(DRAM_START_ADDR)}..{hex(DRAM_END_ADDR)}] ({dram_total_bytes / 1024**3:.2f} GB)")
        while offset < dram_total_bytes:
            write_size = min(chunk_size_bytes, dram_total_bytes - offset)
            self.dma_write(DMA_DEVICE_H2C, DRAM_START_ADDR + offset, fill[:write_size], write_size)
            offset += write_size
            pct = offset / dram_total_bytes
            filled = int(bar_width * pct)
            bar = '█' * filled + '░' * (bar_width - filled)
            print(f"\r  [{bar}] {pct*100:5.1f}%  {offset/1024**2:.0f}/{dram_total_bytes/1024**2:.0f} MB", end='', flush=True)
        print()

    def clear_capture_buffer(self):
        """Clear the capture buffer"""
        self.capture_buffer = []
        self.capture_count = 0
        self._capture_loop_stack = []

    def get_captured_instructions(self) -> list[Instructions]:
        """Get list of captured instructions"""
        if not self.capture_buffer:
            print("Warning: capture_buffer is empty! Call start_capture() first.")
            return []
        return self.capture_buffer

    def get_capture_count(self) -> int:
        """Get number of captured instructions"""
        return self.capture_count

    def get_capture_instruction_size_bytes(self) -> int:
        """Get the size of the captured instructions in bytes"""
        return self.capture_count * 32 # 32 bytes per instruction

    def parse_instruction(self, inst: Instructions, inst_index: int, inst_addr: int) -> None:
        """
        Parse and print an instruction for human-readable display

        Args:
            inst: Instructions object
            inst_index: Instruction index in capture buffer
            inst_addr: Physical DRAM address of instruction
        """
        w = inst.words
        _u32 = lambda x: int(x) & 0xFFFFFFFF

        # Print instruction header (capture buffer words are unsigned 32-bit raw)
        hex_str = " ".join(f"{_u32(w[j]):08X}" for j in range(8))
        print(f"  [{inst_index:4d}] @ {_u32(inst_addr):#010X} = {hex_str}")

        inst_type = _inst_desc_bits(w, 8, 11)
        transaction_id = _inst_desc_bits(w, 0, 7)

        # PBI pointer-row load / bump-only (inst_type 6/C). Same 256b payload layout as UE ops; do not decode as ISA.
        if inst_type == INSTRUCTION_PBI_SET:
            ptr_i = _inst_desc_bits(w, 12, 15)
            pbi_m = _inst_desc_bits(w, 16, 19)
            pbi_fsel = _inst_desc_bits(w, 20, 23)
            if pbi_m == PBI_MODE_INIT:
                pbi_label = "INIT"
            elif pbi_m == PBI_MODE_INC:
                pbi_label = "INC"
            elif pbi_m == PBI_MODE_REG:
                pbi_label = "REG"
            else:
                pbi_label = f"UNKNOWN_MODE({pbi_m})"
            dram_w = _inst_desc_bits(w, 32, 63)
            src_reg = _inst_desc_bits(w, 24, 29)  # pbi_general_reg_idx = descriptor[29:24]
            dma_len = _inst_desc_bits(w, 64, 95)
            out_sz = _inst_desc_bits(w, 156, 171)
            ur_rs = _inst_desc_bits(w, 96, 107)
            ur_rsz = _inst_desc_bits(w, 108, 119)
            ur_ay = _inst_desc_bits(w, 120, 131)
            ur_az = _inst_desc_bits(w, 132, 143)
            ur_wb = _inst_desc_bits(w, 144, 155)
            fmx = _inst_desc_bits(w, 223, 228)
            ur_md = _inst_desc_bits(w, 234, 245)
            reg_suffix = ""
            if pbi_m == PBI_MODE_REG:
                try:
                    field_name = PBI_FIELD(pbi_fsel).name.lower()
                except ValueError:
                    field_name = f"field_{pbi_fsel}"
                reg_suffix = f"  gr[{src_reg}]->{field_name}"
            result = (
                f"PBI_SET ({pbi_label}) inst_pointer_idx={ptr_i}{reg_suffix}\n"
                f"    inst_dram_addr[63:32] (word addr >>3 field): {_u32(dram_w):#010X}  dma_length: {dma_len}  output_size: {out_sz}\n"
                f"    uram_row_size: {ur_rs}  uram_row_size_z: {ur_rsz}  uram_a_y: {ur_ay}  uram_a_z: {ur_az}\n"
                f"    uram_wb: {ur_wb}  uram_memcpy_dst: {ur_md}  fmx_context: {fmx}\n"
                f"    inst_id: {transaction_id}"
            )
            for line in result.split("\n"):
                print(f"        {line}")
            return

        # UE engine operations.
        if inst_type in (INSTRUCTION_UE_OP, INSTRUCTION_UE_PBI):
            mode_sel = _inst_desc_bits(w, 172, 175)

            if mode_sel == 0xF:
                dram_src_addr = _inst_desc_bits(w, 32, 63)
                memcpy_length = _inst_desc_bits(w, 64, 95)
                uram_select = _inst_desc_bits(w, 176, 176)
                bram_uram_select = _inst_desc_bits(w, 232, 233)
                uram_memcpy_dst_addr = _inst_desc_bits(w, 234, 245)
                stride_en = _inst_desc_bits(w, 222, 222)
                lalu_scalar = _inst_desc_bits(w, 186, 206)
                output_size = _inst_desc_bits(w, 156, 171)

                memcpy_type_name = MEMCPY_TYPE(bram_uram_select).name if bram_uram_select in [e.value for e in MEMCPY_TYPE] else f"UNKNOWN({bram_uram_select})"
                memcpy_type_str = f"{memcpy_type_name}"
                if bram_uram_select == MEMCPY_TYPE.URAM.value:
                    uram_section_name = URAM_SECTION(uram_select).name if uram_select in [e.value for e in URAM_SECTION] else f"UNKNOWN({uram_select})"
                    memcpy_type_str += f" ({uram_section_name})"

                result = f"UE_MEMCPY_FROM_DRAM"
                result += f"\n    dram_src_addr: {_u32(dram_src_addr):#010X}"
                result += f"\n    memcpy_length: {memcpy_length} bytes"
                result += f"\n    uram_dst_addr: {uram_memcpy_dst_addr}"
                result += f"\n    memcpy_type: {memcpy_type_str}"
                if stride_en:
                    sj = lalu_scalar
                    result += f"\n    stride_bytes_per_chunk: {output_size}"
                    result += f"\n    stride_jump_bytes: {sj}"
                result += f"\n    inst_id: {transaction_id}"
                for line in result.split('\n'):
                    print(f"        {line}")
                return

            mode = mode_sel
            dma_start = _inst_desc_bits(w, 179, 179)
            uram_start = _inst_desc_bits(w, 180, 180)
            data_type = _inst_desc_bits(w, 181, 182)
            output_size = _inst_desc_bits(w, 156, 171)
            uram_length = _inst_desc_bits(w, 96, 107)
            uram_length_z = _inst_desc_bits(w, 108, 119)
            uram_a_start_addr = _inst_desc_bits(w, 120, 131)
            uram_b_start_addr = _inst_desc_bits(w, 132, 143)
            uram_wb_addr = _inst_desc_bits(w, 144, 155)
            uram_section = _inst_desc_bits(w, 176, 176)
            uram_write_src = _inst_desc_bits(w, 177, 178)
            uram_dst_addr = _inst_desc_bits(w, 234, 245)
            uram_bram = _inst_desc_bits(w, 232, 233)
            lalu_mode = _inst_desc_bits(w, 183, 185)
            scalar = _inst_desc_bits(w, 186, 206)
            wb_padding_control = _inst_desc_bits(w, 207, 207)
            max_clear_en = _inst_desc_bits(w, 208, 208)
            bias_adder_en = _inst_desc_bits(w, 209, 209)
            stride_z = _inst_desc_bits(w, 210, 221)
            stride_en = _inst_desc_bits(w, 222, 222)
            broadcast_mode = _inst_desc_bits(w, 229, 230)
            dram_addr = _inst_desc_bits(w, 32, 63)
            dma_length = _inst_desc_bits(w, 64, 95)
            uram_bram_wb_start = _inst_desc_bits(w, 246, 246)

            if mode == UE_MODE.URAM_DRAM_WRITEBACK.value:
                memcpy_type_name = MEMCPY_TYPE(uram_bram).name if uram_bram in [e.value for e in MEMCPY_TYPE] else f"UNKNOWN({uram_bram})"
                memcpy_type_str = f"{memcpy_type_name}"
                if uram_bram == MEMCPY_TYPE.URAM.value:
                    uram_section_name = URAM_SECTION(uram_section).name if uram_section in [e.value for e in URAM_SECTION] else f"UNKNOWN({uram_section})"
                    memcpy_type_str += f" ({uram_section_name})"

                result = f"UE_MEMCPY_TO_DRAM"
                result += f"\n    uram_src_addr: {uram_a_start_addr}"
                result += f"\n    dram_dst_addr: {_u32(dram_addr):#010X}"
                result += f"\n    memcpy_length: {uram_length} uram rows ({uram_length * 64 * 2} bytes)"
                result += f"\n    memcpy_type: {memcpy_type_str}"
                result += f"\n    uram_bram_wb_start: {uram_bram_wb_start}"
                if stride_en:
                    result += f"\n    stride_en: enabled"
                result += f"\n    inst_id: {transaction_id}"
                for line in result.split('\n'):
                    print(f"        {line}")
                return

            mode_name = UE_MODE(mode).name if mode in [e.value for e in UE_MODE] else f"UNKNOWN({mode})"
            result = f"UE_COMPUTE ({mode_name})"
            result += f"\n    uram_a_start: {uram_a_start_addr}, uram_b_start: {uram_b_start_addr}"
            result += f"\n    uram_length: {uram_length}, uram_wb_addr: {uram_wb_addr}"
            result += f"\n    output_size: {output_size}"
            data_type_name = TYPE(data_type).name if data_type in [e.value for e in TYPE] else f"UNKNOWN({data_type})"
            result += f"\n    data_type: {data_type_name}"
            if uram_start:
                result += f"\n    uram_start: enabled, uram_length: {uram_length}, uram_length_z: {uram_length_z}"
            if dma_start:
                result += f"\n    dma_start: enabled, dma_start_addr: {_u32(dram_addr):#010X}"
                result += f", dma_length: {dma_length}"
            if uram_bram == MEMCPY_TYPE.URAM.value:
                uram_section_name = URAM_SECTION(uram_section).name if uram_section in [e.value for e in URAM_SECTION] else f"UNKNOWN({uram_section})"
                result += f"\n    uram_type: URAM ({uram_section_name})"
            else:
                uram_type_name = MEMCPY_TYPE(uram_bram).name if uram_bram in [e.value for e in MEMCPY_TYPE] else f"UNKNOWN({uram_bram})"
                result += f"\n    uram_type: {uram_type_name}"
            if uram_write_src == URAM_WRITE_SRC.URAM_WRITE_BACK.value:
                result += f"\n    writeback: enabled, uram_dst_addr: {uram_dst_addr}"
            else:
                writeback_name = URAM_WRITE_SRC(uram_write_src).name if uram_write_src in [e.value for e in URAM_WRITE_SRC] else f"UNKNOWN({uram_write_src})"
                result += f"\n    writeback: disabled ({writeback_name})"
            result += f"\n    wb_padding_select: {wb_padding_control}"
            if bias_adder_en:
                result += f"\n    bias_adder: enabled"
            if max_clear_en:
                result += f"\n    max_clear: enabled"
            if lalu_mode != 0:
                lalu_mode_name = LALU_MODE(lalu_mode).name if lalu_mode in [e.value for e in LALU_MODE] else f"UNKNOWN({lalu_mode})"
                result += f"\n    lalu_mode: {lalu_mode_name}, scalar: {_u32(scalar):#010X}"
            if broadcast_mode != 0:
                broadcast_mode_name = BROADCAST_MODE(broadcast_mode).name if broadcast_mode in [e.value for e in BROADCAST_MODE] else f"UNKNOWN({broadcast_mode})"
                result += f"\n    broadcast_mode: {broadcast_mode_name}"
            result += f"\n    uram_row_stride_z: {stride_z}, stride_en: {stride_en}"
            result += f"\n    inst_id: {transaction_id}"
            for line in result.split('\n'):
                print(f"        {line}")
            return

        # ISA (non-UE): [85:32] micro-op fields
        isa_mode = _inst_desc_bits(w, 32, 35)
        src_reg_idx = _inst_desc_bits(w, 36, 41)
        dst_reg_idx = _inst_desc_bits(w, 42, 47)
        rst_reg_idx = _inst_desc_bits(w, 48, 53)
        immediate_value = _inst_desc_bits(w, 54, 85)

        if inst_type == INSTRUCTION_SWI:
            result = f"ISA_SWI"
            result += f"\n    transaction_id: {transaction_id}"
            for line in result.split('\n'):
                print(f"        {line}")
            return
        if inst_type == INSTRUCTION_HALT:
            result = f"ISA_HALT"
            result += f"\n    transaction_id: {transaction_id}"
            for line in result.split('\n'):
                print(f"        {line}")
            return
        if inst_type == INSTRUCTION_JUMP:
            jump_mode = isa_mode
            reg_id = src_reg_idx
            jump_mode_names = {
                JUMP_MODE_ABSOLUTE: "ABSOLUTE",
                JUMP_MODE_REG_ABS:  "REG_ABS",
                JUMP_MODE_JNZ: "JNZ",
                JUMP_MODE_JZ: "JZ",
                JUMP_MODE_RELATIVE: "RELATIVE",
                JUMP_MODE_RELA_JNZ: "RELA_JNZ",
                JUMP_MODE_RELA_JZ: "RELA_JZ",
                JUMP_MODE_REG_RELA: "REG_RELA",
            }
            jump_mode_name = jump_mode_names.get(jump_mode, f"UNKNOWN({jump_mode})")
            result = f"ISA_JUMP ({jump_mode_name})"
            if jump_mode in (JUMP_MODE_RELATIVE, JUMP_MODE_RELA_JNZ, JUMP_MODE_RELA_JZ):
                result += f"\n    relative_back: {immediate_value} inst words"
            elif jump_mode in (JUMP_MODE_REG_ABS, JUMP_MODE_REG_RELA):
                result += f"\n    rst_reg_idx: {rst_reg_idx}"
            else:
                result += f"\n    target_addr: {_u32(immediate_value):#010X}"
            result += f"\n    transaction_id: {transaction_id}"
            if jump_mode in (JUMP_MODE_JNZ, JUMP_MODE_JZ, JUMP_MODE_RELA_JNZ, JUMP_MODE_RELA_JZ):
                result += f"\n    reg_id: {reg_id}"
            for line in result.split('\n'):
                print(f"        {line}")
            return
        if inst_type == INSTRUCTION_REG_ALU:
            isa_mode_names = {
                ALU_MODE_INC:     "INC",
                ALU_MODE_DEC:     "DEC",
                ALU_MODE_ADD_REG: "ADD_REG",
                ALU_MODE_ADD_IMM: "ADD_IMM",
                ALU_MODE_SET:     "SET",
                ALU_MODE_MIN:     "MIN",
                ALU_MODE_SUB:     "SUB",
                ALU_MODE_SHR:       "SHR",
                ALU_MODE_SHL:       "SHL",
                ALU_MODE_MUL32_REG: "MUL32_REG",
                ALU_MODE_MUL32_IMM: "MUL32_IMM",
                ALU_MODE_DIV_REG:   "DIV_REG",
                ALU_MODE_MUL_SHL:   "MUL_SHL",
                ALU_MODE_MUL_SHR:   "MUL_SHR",
            }
            mode_name = isa_mode_names.get(isa_mode, f"UNKNOWN({isa_mode})")
            result = f"ISA_REG_ALU ({mode_name})"
            result += f"\n    transaction_id: {transaction_id}"
            if isa_mode == ALU_MODE_SET:
                result += f"\n    dst_reg: {dst_reg_idx}, value: {_u32(immediate_value):#010X}"
            elif isa_mode in (ALU_MODE_INC, ALU_MODE_DEC):
                result += f"\n    reg: {dst_reg_idx}"
            elif isa_mode in (ALU_MODE_ADD_IMM, ALU_MODE_MUL32_IMM,
                              ALU_MODE_SHR, ALU_MODE_SHL):
                result += f"\n    dst_reg: {dst_reg_idx}, src_reg: {src_reg_idx}, immediate: {_u32(immediate_value):#010X}"
            elif isa_mode in (ALU_MODE_ADD_REG, ALU_MODE_MIN, ALU_MODE_SUB,
                              ALU_MODE_MUL32_REG, ALU_MODE_DIV_REG):
                result += f"\n    dst_reg: {dst_reg_idx}, src_reg: {src_reg_idx}, rst_reg: {rst_reg_idx}"
            elif isa_mode in (ALU_MODE_MUL_SHL, ALU_MODE_MUL_SHR):
                result += f"\n    dst_reg: {dst_reg_idx}, src_reg: {src_reg_idx}, rst_reg: {rst_reg_idx}, shift: {immediate_value & 0x1F}"
            for line in result.split('\n'):
                print(f"        {line}")
            return
        if inst_type == INSTRUCTION_FLAG:
            flag_mode = isa_mode
            target_engine = src_reg_idx & 0x7
            flag_mode_names = {
                FLAG_MODE_SET: "SET",
                FLAG_MODE_CLEAR: "CLEAR",
                FLAG_MODE_CHECK: "CHECK",
            }
            mode_name = flag_mode_names.get(flag_mode, f"UNKNOWN({flag_mode})")
            result = f"ISA_FLAG ({mode_name})"
            result += f"\n    transaction_id: {transaction_id}"
            if flag_mode == FLAG_MODE_CHECK:
                result += f"\n    target_engine: {target_engine}"
            for line in result.split('\n'):
                print(f"        {line}")
            return
        if inst_type == INSTRUCTION_NOP:
            result = f"ISA_NOP"
            result += f"\n    transaction_id: {transaction_id}"
            for line in result.split('\n'):
                print(f"        {line}")
            return

        result = f"ISA_UNKNOWN (type=0x{inst_type:X})"
        result += f"\n    transaction_id: {transaction_id}"
        for line in result.split('\n'):
            print(f"        {line}")

    def ue_isa_descriptor(self, inst_type: int, immediate_value: int = 0,
                          isa_mode: int = 0, src_reg_idx: int = 0,
                          dst_reg_idx: int = 0, rst_reg_idx: int = 0):
        """
        Shared 256b instruction descriptor compiler for ISA micro-ops (JUMP / REG_ALU / SEMAPHORE / FLAG).

        Header [15:0]: [7:0] instruction index from :attr:`_inst_id`; [11:8] inst_type; [15:12] reserved.
        ISA [85:32]: [35:32] isa_mode; [41:36] src; [47:42] dst; [53:48] rst; [85:54] immediate.

        After append, :attr:`_inst_id` is incremented (same pattern as :meth:`ue_op_descriptor`).
        """
        if self.capture_buffer is None:
            print("ERROR: ue_isa_descriptor() called but capture_buffer is not initialized!")
            return

        if self.capture_count >= MAX_DECODER_INSTRUCTIONS:
            print(f"ERROR: ue_isa_descriptor() called but capture_count ({self.capture_count}) >= MAX ({MAX_DECODER_INSTRUCTIONS})!")
            return

        tid = self._inst_id

        inst = Instructions()
        w = inst.words

        # ISA descriptor: Header [15:0] (w[0][15:0])
        w[0] = (tid & 0xFF) | ((inst_type & 0xF) << 8)

        # ISA [85:32]: [35:32] isa_mode; [41:36] src; [47:42] dst; [53:48] rst; [85:54] immediate
        w[1] = ((isa_mode & 0xF) |
                ((src_reg_idx & 0x3F) << 4) |
                ((dst_reg_idx & 0x3F) << 10) |
                ((rst_reg_idx & 0x3F) << 16) |
                ((immediate_value & 0x3FF) << 22))
        w[2] = (immediate_value >> 10) & 0x3FFFFF

        self.capture_buffer.append(inst)
        self.capture_count += 1
        self._inst_id = tid + 1

    def generate_instruction_halt(self):
        """Generate a HALT instruction. Instruction index comes from :attr:`_inst_id` (then incremented).

        If the captured stream length after HALT is not a multiple of two instructions (64 bytes),
        appends one ``NOP`` so program boundaries stay aligned for 512-bit DRAM instruction fetch
        (same rule as :meth:`write_captured_instructions_to_dram`).
        """
        self.ue_isa_descriptor(INSTRUCTION_HALT)
        self.pad_capture_to_64b_boundary()

    def generate_instruction_swi(self):
        """Generate an SWI instruction (software interrupt). Same header layout as HALT."""
        self.ue_isa_descriptor(INSTRUCTION_SWI)

    def generate_instruction_nop(self):
        """Generate a NOP instruction (``INSTRUCTION_NOP`` / ``INST_TYPE_NOP = 4'hA``).

        The queue FSM falls straight through to ``STATE_FETCH`` so the PC advances
        by one instruction with no datapath side effects. Used as padding to keep
        absolute jump targets and program entry points 512-bit (64 B) aligned.
        """
        self.ue_isa_descriptor(INSTRUCTION_NOP)

    def pad_capture_to_64b_boundary(self) -> None:
        """If the captured instruction count is currently odd (32 B aligned but not 64 B),
        append a NOP so the next instruction lands on a 512-bit / 64 B / 2-instruction boundary.

        The hardware DRAM instruction fetch path issues 512-bit reads, so absolute jump targets
        and program boundaries must be 64 B aligned. This is the single shared helper used by:

        - :meth:`generate_instruction_halt` (program-end alignment)
        - :meth:`write_captured_instructions_to_dram` (DRAM-flush alignment)

        :meth:`_generate_instruction_jump` has its own related logic that adjusts the **jump
        target** address (not the current capture position), which is a different concern.
        """
        if self.capture_count % 2 != 0:
            self.generate_instruction_nop()

    def _generate_instruction_jump(
        self,
        immediate_value: int,
        jump_mode: int,
        src_reg_idx: int = 0,
    ) -> None:
        """
        Low-level ``INSTRUCTION_JUMP`` (same layout as ``andromeda.c`` ``generate_instruction_jump``).

        Prefer the ``generate_instruction_jump_*`` helpers so callers do not pass ``JUMP_MODE_*``
        explicitly.

        For absolute jump modes (``JUMP_MODE_ABSOLUTE`` / ``JUMP_MODE_JNZ`` / ``JUMP_MODE_JZ``)
        we enforce DRAM alignment of the target byte address (= ``immediate_value << 3``):

        - It must be at least 256-bit (one instruction = 32 B) aligned, otherwise
          assert. Anything finer than that is meaningless for an instruction PC.
        - It should be 512-bit (64 B = 2 instructions) aligned for the upcoming
          RTL. If only 32 B aligned, transparently pad: emit a ``NOP`` before the
          jump (so the jump itself shifts to the next slot) and bump the target
          by one instruction so it lands on a 64 B boundary. This preserves the
          common "fall through to next captured instruction" pattern used by the
          PBI helpers (rms / rope / matmat), since both the jump and its target
          shift by exactly one slot.
        """
        if jump_mode in (JUMP_MODE_ABSOLUTE, JUMP_MODE_JNZ, JUMP_MODE_JZ):
            target_byte_addr = int(immediate_value) << 3
            assert target_byte_addr % INSTRUCTION_SIZE_BYTES == 0, (
                f"Absolute jump target byte address 0x{target_byte_addr:X} is not "
                f"{INSTRUCTION_SIZE_BYTES}-byte (256-bit) aligned"
            )
            if target_byte_addr % (2 * INSTRUCTION_SIZE_BYTES) != 0:
                self.generate_instruction_nop()
                target_byte_addr += INSTRUCTION_SIZE_BYTES
                immediate_value = ue_35bit_addr_shifter(target_byte_addr)
        self.ue_isa_descriptor(
            INSTRUCTION_JUMP,
            immediate_value=immediate_value & 0xFFFFFFFF,
            isa_mode=jump_mode,
            src_reg_idx=src_reg_idx,
        )

    def generate_instruction_jump_abs(self, target_instruction_word_addr: int) -> None:
        """Unconditional absolute jump (``JUMP_MODE_ABSOLUTE``). ``target_instruction_word_addr`` is byte address ``>> 3``."""
        self._generate_instruction_jump(target_instruction_word_addr, JUMP_MODE_ABSOLUTE, 0)

    def generate_instruction_jump_abs_jnz(self, target_instruction_word_addr: int, reg_id: int) -> None:
        """Absolute jump if ``reg_id != 0`` (``JUMP_MODE_JNZ``)."""
        self._generate_instruction_jump(target_instruction_word_addr, JUMP_MODE_JNZ, reg_id)

    def generate_instruction_jump_abs_jz(self, target_instruction_word_addr: int, reg_id: int) -> None:
        """Absolute jump if ``reg_id == 0`` (``JUMP_MODE_JZ``)."""
        self._generate_instruction_jump(target_instruction_word_addr, JUMP_MODE_JZ, reg_id)

    def _emit_forward_skip_jz(self, cond_reg: int, program_dram_start_addr: int):
        """Emit a forward ``JZ cond_reg`` that skips the next captured block; returns a ``patch()``.

        Call ``patch()`` once the to-be-skipped block has been captured. It pads the stream to a
        512-bit (64 B / 2-instruction) boundary so the rejoin label is fetch-aligned, then rewrites
        the JZ's target immediate to the current capture position. The JZ is emitted with a
        64 B-aligned placeholder target so :meth:`_generate_instruction_jump` does NOT auto-insert an
        alignment NOP, which keeps the JZ at a fixed buffer index for in-place patching. When
        ``cond_reg == 0`` at run time the engine jumps over the block; otherwise it falls through.

        Mirrors the existing startup sub-64 skip pattern (a forward absolute conditional jump before
        the resident loop body); the only twist is a runtime-sized skipped block, handled by patching
        the target after the fact instead of hard-coding the instruction offset.
        """
        size_b = INSTRUCTION_SIZE_BYTES

        def aligned64(idx: int) -> bool:
            return (program_dram_start_addr + idx * size_b) % (2 * size_b) == 0

        placeholder_idx = self.capture_count + 2
        while not aligned64(placeholder_idx):
            placeholder_idx += 1
        placeholder = ue_35bit_addr_shifter(program_dram_start_addr + placeholder_idx * size_b)
        jz_idx = self.capture_count
        self.generate_instruction_jump_abs_jz(placeholder, cond_reg)
        assert self.capture_count == jz_idx + 1, (
            "forward-skip JZ unexpectedly inserted an alignment NOP (placeholder target not 64 B aligned)"
        )

        def patch():
            while not aligned64(self.capture_count):
                self.generate_instruction_nop()
            imm = ue_35bit_addr_shifter(program_dram_start_addr + self.capture_count * size_b) & 0xFFFFFFFF
            # ue_isa_descriptor JUMP layout: w[1] = [3:0]mode | [9:4]src | [31:22]imm[9:0];
            # w[2] = imm[31:10]. dst/rst stay 0 for a conditional jump.
            w = self.capture_buffer[jz_idx].words
            w[1] = (JUMP_MODE_JZ & 0xF) | ((cond_reg & 0x3F) << 4) | ((imm & 0x3FF) << 22)
            w[2] = (imm >> 10) & 0x3FFFFF

        return patch

    def generate_instruction_jump_rela(self, backward_instruction_words: int) -> None:
        """Unconditional relative backward jump (``JUMP_MODE_RELATIVE``); immediate is backward offset in instruction words."""
        self._generate_instruction_jump(backward_instruction_words, JUMP_MODE_RELATIVE, 0)

    def generate_instruction_jump_rela_jnz(self, backward_instruction_words: int, reg_id: int) -> None:
        """Relative backward jump if ``reg_id != 0`` (``JUMP_MODE_RELA_JNZ``)."""
        self._generate_instruction_jump(backward_instruction_words, JUMP_MODE_RELA_JNZ, reg_id)

    def generate_instruction_jump_rela_jz(self, backward_instruction_words: int, reg_id: int) -> None:
        """Relative backward jump if ``reg_id == 0`` (``JUMP_MODE_RELA_JZ``)."""
        self._generate_instruction_jump(backward_instruction_words, JUMP_MODE_RELA_JZ, reg_id)

    def generate_instruction_jump_reg_abs(self, rst_reg_idx: int) -> None:
        """Unconditional absolute jump to the UE word address held in ``rst_reg_idx`` (``JUMP_MODE_REG_ABS``).

        The register must contain the same word-address format used by
        :meth:`generate_instruction_jump_abs` (``byte_addr >> 3``), typically written
        with :meth:`generate_instruction_add_set` before the loop.  No alignment
        padding is applied because the runtime value is unknown at capture time.
        """
        self.ue_isa_descriptor(
            INSTRUCTION_JUMP,
            immediate_value=0,
            isa_mode=JUMP_MODE_REG_ABS,
            rst_reg_idx=rst_reg_idx,
        )

    def generate_instruction_jump_reg_rela(self, rst_reg_idx: int) -> None:
        """Unconditional relative backward jump; offset in instruction words taken from ``rst_reg_idx[9:0]`` (``JUMP_MODE_REG_RELA``).

        ``inst_ram_read_ptr`` is decremented by the register value (saturating at 0),
        mirroring :meth:`generate_instruction_jump_rela` but with a runtime-computed
        offset stored in a GPR.
        """
        self.ue_isa_descriptor(
            INSTRUCTION_JUMP,
            immediate_value=0,
            isa_mode=JUMP_MODE_REG_RELA,
            rst_reg_idx=rst_reg_idx,
        )

    def loop_start(self, loop_cnt: int = 0, gpr_loop_cnt: int = None) -> int:
        """
        Open a counted backward-branch loop during instruction capture (stack-nested).

        Pushes a new frame: allocates a counter register, emits ``ADD_SET`` with ``loop_cnt``, and
        records ``body_start_inst_cnt`` = :attr:`_inst_id` after that instruction (the transaction id
        of the next captured instruction, i.e. the loop-body head). Nested loops call
        ``loop_start`` again before the matching ``loop_end``; each frame has its own register and
        start index.

        Args:
            loop_cnt: Initial iteration count **when** ``gpr_loop_cnt`` is omitted (same semantics as
                :meth:`generate_instruction_add_set`). Still used when ``gpr_loop_cnt`` is set — for
                documentation / templates only; emission copies the iteration count from
                ``gpr_loop_cnt`` instead.
            gpr_loop_cnt: If not ``None``, non-zero GPR index holding the runtime loop trip count.
                Emits ``ADD_IMM`` into the allocated counter register as ``dst = src + 0`` so the
                counter starts from whatever value that GPR held at execution time (typically set with
                :meth:`generate_instruction_add_set` earlier in the same program).

        Returns:
            The allocated loop counter register index.

        Raises:
            RuntimeError: If capture is not active.
            ValueError: If ``gpr_loop_cnt == 0``.
        """
        if not self.is_capture_on or self.capture_buffer is None:
            raise RuntimeError("loop_start() requires an active capture (call start_capture() first).")
        reg = self.alloc_isa_reg()
        if gpr_loop_cnt is not None:
            if gpr_loop_cnt == 0:
                raise ValueError("loop_start(): gpr_loop_cnt must not be 0 (register 0 is hard-wired)")
            self.generate_instruction_add_imm(src_reg_idx=gpr_loop_cnt, immediate_value=0, dst_reg_idx=reg)
        else:
            self.generate_instruction_add_set(dst_reg_idx=reg, immediate_value=loop_cnt)
        body_start_inst_cnt = self.capture_count
        self._capture_loop_stack.append((reg, body_start_inst_cnt))
        return reg

    def loop_end(self) -> int:
        """
        Close the innermost loop opened with :meth:`loop_start` (LIFO).

        Pops one stack frame, computes ``backward_instruction_words`` = current ``_inst_id`` -
        ``body_start_inst_cnt`` + 2, then emits :meth:`generate_instruction_add_dec` and
        :meth:`generate_instruction_jump_rela_jnz`, and releases the counter register.

        Returns:
            The ``backward_instruction_words`` value passed to the jump.

        Raises:
            RuntimeError: If capture is not active or the loop stack is empty.
        """
        if not self.is_capture_on or self.capture_buffer is None:
            raise RuntimeError("loop_end() requires an active capture (call start_capture() first).")
        if not self._capture_loop_stack:
            raise RuntimeError("loop_end() without a matching loop_start().")
        reg, body_start_inst_cnt = self._capture_loop_stack.pop()
        loop_body_size = self.capture_count - body_start_inst_cnt + 2
        self.generate_instruction_add_dec(reg_idx=reg)
        self.generate_instruction_jump_rela_jnz(loop_body_size, reg)
        self.release_isa_reg()
        return loop_body_size

    def generate_instruction_add_inc(self, reg_idx: int):
        """
        Generate an ADD instruction to increment a register

        Args:
            reg_idx: Register index to increment (must not be 0)
        """
        if reg_idx == 0:
            print("ERROR: INSTRUCTION_REG_ALU overwriting reg_idx 0 (zero reg) not allowed")
            return
        self.ue_isa_descriptor(INSTRUCTION_REG_ALU, isa_mode=ALU_MODE_INC,
                               src_reg_idx=reg_idx, dst_reg_idx=reg_idx)

    def generate_instruction_add_dec(self, reg_idx: int):
        """
        Generate an ADD instruction to decrement a register

        Args:
            reg_idx: Register index to decrement (must not be 0)
        """
        if reg_idx == 0:
            print("ERROR: INSTRUCTION_REG_ALU overwriting reg_idx 0 (zero reg) not allowed")
            return
        self.ue_isa_descriptor(INSTRUCTION_REG_ALU, isa_mode=ALU_MODE_DEC,
                               src_reg_idx=reg_idx, dst_reg_idx=reg_idx)

    def generate_instruction_add_reg(self, dst_reg_idx: int, src_reg_idx: int, rst_reg_idx: int):
        """
        Generate an ADD instruction to add two registers

        Args:
            dst_reg_idx: Destination register index (must not be 0)
            src_reg_idx: Source register index
            rst_reg_idx: Reset register index
        """
        if dst_reg_idx == 0:
            print("ERROR: INSTRUCTION_REG_ALU overwriting reg_idx 0 (zero reg) not allowed")
            return
        self.ue_isa_descriptor(INSTRUCTION_REG_ALU, isa_mode=ALU_MODE_ADD_REG,
                               src_reg_idx=src_reg_idx, dst_reg_idx=dst_reg_idx,
                               rst_reg_idx=rst_reg_idx)

    def generate_instruction_add_imm(self, src_reg_idx: int, immediate_value: int, dst_reg_idx: Optional[int] = None):
        """
        Generate an ADD instruction to add immediate value to register

        Args:
            src_reg_idx: Source register index (must not be 0)
            immediate_value: 32-bit immediate value to add
            dst_reg_idx: Destination register index (defaults to src_reg_idx if None)
        """
        if src_reg_idx == 0:
            print("ERROR: INSTRUCTION_REG_ALU overwriting reg_idx 0 (zero reg) not allowed")
            return
        if dst_reg_idx is None:
            dst_reg_idx = src_reg_idx
        self.ue_isa_descriptor(INSTRUCTION_REG_ALU, immediate_value=immediate_value,
                               isa_mode=ALU_MODE_ADD_IMM, src_reg_idx=src_reg_idx,
                               dst_reg_idx=dst_reg_idx)

    def generate_instruction_add_set(self, dst_reg_idx: int, immediate_value: int):
        """
        Generate an ADD instruction to set register to immediate value

        Args:
            dst_reg_idx: Destination register index (must not be 0)
            immediate_value: 32-bit immediate value to set
        """
        if dst_reg_idx == 0:
            print("ERROR: INSTRUCTION_REG_ALU overwriting reg_idx 0 (zero reg) not allowed")
            return
        self.ue_isa_descriptor(INSTRUCTION_REG_ALU, immediate_value=immediate_value,
                               isa_mode=ALU_MODE_SET, src_reg_idx=dst_reg_idx,
                               dst_reg_idx=dst_reg_idx)

    def generate_instruction_reg_min(self, dst_reg_idx: int, src_reg_idx: int, rst_reg_idx: int):
        """
        Emit ALU_MODE_MIN: dst = min(src1, src2), unsigned.

        Args:
            dst_reg_idx: Destination register (must not be 0)
            src_reg_idx: First source register
            rst_reg_idx: Second source register
        """
        if dst_reg_idx == 0:
            print("ERROR: INSTRUCTION_REG_ALU overwriting reg_idx 0 (zero reg) not allowed")
            return
        self.ue_isa_descriptor(INSTRUCTION_REG_ALU, isa_mode=ALU_MODE_MIN,
                               src_reg_idx=src_reg_idx, dst_reg_idx=dst_reg_idx,
                               rst_reg_idx=rst_reg_idx)

    def generate_instruction_reg_sub(self, dst_reg_idx: int, src_reg_idx: int, rst_reg_idx: int):
        """
        Emit ALU_MODE_SUB: dst = src1 - src2, unsigned.

        Args:
            dst_reg_idx: Destination register (must not be 0)
            src_reg_idx: Minuend register (src1)
            rst_reg_idx: Subtrahend register (src2)
        """
        if dst_reg_idx == 0:
            print("ERROR: INSTRUCTION_REG_ALU overwriting reg_idx 0 (zero reg) not allowed")
            return
        self.ue_isa_descriptor(INSTRUCTION_REG_ALU, isa_mode=ALU_MODE_SUB,
                               src_reg_idx=src_reg_idx, dst_reg_idx=dst_reg_idx,
                               rst_reg_idx=rst_reg_idx)

    def generate_instruction_reg_mul_imm(
        self, dst_reg_idx: int, src_reg_idx: int, immediate_value: int
    ):
        """Emit a reg*imm multiply: dst = (src * imm)[31:0], unsigned.

        Uses ALU_MODE_MUL32_IMM (3-cycle pipelined DSP). The RTL zero-extends the
        immediate from 16 bits, so ``immediate_value`` must fit in 16 bits; the full
        32-bit ``src`` register is used for the product.
        """
        if dst_reg_idx == 0:
            print("ERROR: INSTRUCTION_REG_ALU overwriting reg_idx 0 (zero reg) not allowed")
            return
        assert immediate_value & 0xFFFF == immediate_value, "immediate_value must be less than 2**16"
        self.ue_isa_descriptor(
            INSTRUCTION_REG_ALU,
            immediate_value=immediate_value & 0xFFFFFFFF,
            isa_mode=ALU_MODE_MUL32_IMM,
            src_reg_idx=src_reg_idx,
            dst_reg_idx=dst_reg_idx,
        )

    def generate_instruction_shr(self, dst_reg_idx: int, src_reg_idx: int, immediate_value: int):
        """Emit ALU_MODE_SHR: dst = src >> immediate_value[4:0], logical right shift."""
        if dst_reg_idx == 0:
            print("ERROR: INSTRUCTION_REG_ALU overwriting reg_idx 0 (zero reg) not allowed")
            return
        assert 0 <= immediate_value <= 31, "shift amount must be 0..31"
        self.ue_isa_descriptor(INSTRUCTION_REG_ALU, isa_mode=ALU_MODE_SHR,
                               immediate_value=immediate_value,
                               src_reg_idx=src_reg_idx, dst_reg_idx=dst_reg_idx)

    def generate_instruction_shl(self, dst_reg_idx: int, src_reg_idx: int, immediate_value: int):
        """Emit ALU_MODE_SHL: dst = src << immediate_value[4:0], logical left shift."""
        if dst_reg_idx == 0:
            print("ERROR: INSTRUCTION_REG_ALU overwriting reg_idx 0 (zero reg) not allowed")
            return
        assert 0 <= immediate_value <= 31, "shift amount must be 0..31"
        self.ue_isa_descriptor(INSTRUCTION_REG_ALU, isa_mode=ALU_MODE_SHL,
                               immediate_value=immediate_value,
                               src_reg_idx=src_reg_idx, dst_reg_idx=dst_reg_idx)

    def generate_instruction_mul32_reg(self, dst_reg_idx: int, src_reg_idx: int, rst_reg_idx: int):
        """Emit ALU_MODE_MUL32_REG: dst = (src * rst)[31:0], 3-cycle pipelined (int_mult_pipe)."""
        if dst_reg_idx == 0:
            print("ERROR: INSTRUCTION_REG_ALU overwriting reg_idx 0 (zero reg) not allowed")
            return
        self.ue_isa_descriptor(INSTRUCTION_REG_ALU, isa_mode=ALU_MODE_MUL32_REG,
                               src_reg_idx=src_reg_idx, dst_reg_idx=dst_reg_idx,
                               rst_reg_idx=rst_reg_idx)

    def generate_instruction_mul32_imm(self, dst_reg_idx: int, src_reg_idx: int, immediate_value: int):
        """Emit ALU_MODE_MUL32_IMM: dst = (src * immediate_value)[31:0], 3-cycle pipelined."""
        if dst_reg_idx == 0:
            print("ERROR: INSTRUCTION_REG_ALU overwriting reg_idx 0 (zero reg) not allowed")
            return
        assert immediate_value & 0xFFFFFFFF == immediate_value, "immediate_value must fit in 32 bits"
        self.ue_isa_descriptor(INSTRUCTION_REG_ALU, isa_mode=ALU_MODE_MUL32_IMM,
                               immediate_value=immediate_value & 0xFFFFFFFF,
                               src_reg_idx=src_reg_idx, dst_reg_idx=dst_reg_idx)

    def generate_instruction_mul32_shl_reg(self, dst_reg_idx: int, src_reg_idx: int, rst_reg_idx: int, shift: int):
        """Emit ALU_MODE_MUL_SHL: dst = ((src * rst)[31:0]) << shift[4:0], one fused 3-cycle word.

        Collapses a ``mul32_reg`` followed by ``shl`` into a single ISA instruction, reusing the
        shared ``int_mult_pipe`` (no extra DSP). The shift amount rides in the immediate field;
        the two multiply operands are both registers (``src`` and ``rst``).
        """
        if dst_reg_idx == 0:
            print("ERROR: INSTRUCTION_REG_ALU overwriting reg_idx 0 (zero reg) not allowed")
            return
        assert 0 <= shift <= 31, "shift amount must be 0..31"
        self.ue_isa_descriptor(INSTRUCTION_REG_ALU, isa_mode=ALU_MODE_MUL_SHL,
                               immediate_value=shift,
                               src_reg_idx=src_reg_idx, dst_reg_idx=dst_reg_idx,
                               rst_reg_idx=rst_reg_idx)

    def generate_instruction_mul32_shr_reg(self, dst_reg_idx: int, src_reg_idx: int, rst_reg_idx: int, shift: int):
        """Emit ALU_MODE_MUL_SHR: dst = ((src * rst)[31:0]) >> shift[4:0], one fused 3-cycle word.

        Logical right shift of the 32-bit product. Collapses ``mul32_reg`` + ``shr`` into one
        word. The products it shifts here are non-negative byte/element counts, so logical and
        arithmetic right shift agree.
        """
        if dst_reg_idx == 0:
            print("ERROR: INSTRUCTION_REG_ALU overwriting reg_idx 0 (zero reg) not allowed")
            return
        assert 0 <= shift <= 31, "shift amount must be 0..31"
        self.ue_isa_descriptor(INSTRUCTION_REG_ALU, isa_mode=ALU_MODE_MUL_SHR,
                               immediate_value=shift,
                               src_reg_idx=src_reg_idx, dst_reg_idx=dst_reg_idx,
                               rst_reg_idx=rst_reg_idx)

    def generate_instruction_div_reg(self, dst_reg_idx: int, src_reg_idx: int, rst_reg_idx: int):
        """Emit ALU_MODE_DIV_REG: dst = src / rst (unsigned floor), 32-cycle sequential (int_divider)."""
        if dst_reg_idx == 0:
            print("ERROR: INSTRUCTION_REG_ALU overwriting reg_idx 0 (zero reg) not allowed")
            return
        self.ue_isa_descriptor(INSTRUCTION_REG_ALU, isa_mode=ALU_MODE_DIV_REG,
                               src_reg_idx=src_reg_idx, dst_reg_idx=dst_reg_idx,
                               rst_reg_idx=rst_reg_idx)

    def generate_instruction_flag_set(self):
        """Set this engine's flag to 1, signaling busy to other engines."""
        self.ue_isa_descriptor(INSTRUCTION_FLAG, isa_mode=FLAG_MODE_SET)

    def generate_instruction_flag_clear(self):
        """Clear this engine's flag to 0, signaling done to other engines."""
        self.ue_isa_descriptor(INSTRUCTION_FLAG, isa_mode=FLAG_MODE_CLEAR)

    def generate_instruction_flag_check(self, target_engine_idx: int):
        """
        Spin-wait until target engine's flag is 1 before proceeding.

        Args:
            target_engine_idx: Engine index (0-7) whose flag to wait on
        """
        if target_engine_idx < 0 or target_engine_idx > 7:
            print(f"ERROR: target_engine_idx must be 0-7, got {target_engine_idx}")
            return
        self.ue_isa_descriptor(INSTRUCTION_FLAG, isa_mode=FLAG_MODE_CHECK,
                               src_reg_idx=target_engine_idx)

    def _patch_jump_immediate(self, capture_idx: int, target_word_addr: int) -> None:
        """
        Patch the 32-bit immediate of an already-captured ``INSTRUCTION_JUMP`` descriptor in
        :attr:`capture_buffer`, preserving all other fields (``inst_id``, ``inst_type``,
        ``isa_mode``, ``src/dst/rst`` register indices).

        Layout (see :meth:`ue_isa_descriptor`): immediate occupies bits [85:54] split as
        ``w[1][31:22]`` (low 10) and ``w[2][21:0]`` (high 22).
        """
        if self.capture_buffer is None or capture_idx < 0 or capture_idx >= len(self.capture_buffer):
            raise IndexError(
                f"_patch_jump_immediate: capture_idx={capture_idx} out of range "
                f"(buffer size={len(self.capture_buffer) if self.capture_buffer else 0})"
            )
        target = int(target_word_addr) & 0xFFFFFFFF
        w = self.capture_buffer[capture_idx].words
        w[1] = (w[1] & 0x003FFFFF) | ((target & 0x3FF) << 22)
        w[2] = (w[2] & 0xFFC00000) | ((target >> 10) & 0x3FFFFF)

    def generate_instruction_pbi_init(
        self,
        dram_shared_addr: int = 0,
        dma_length: int = 0,
        output_size: int = 0,
        uram_length: int = 0,
        uram_a_start_addr: int = 0,
        uram_b_start_addr: int = 0,
        uram_wb_addr: int = 0,
        uram_dst_addr: int = 0,
        fmax_context_addr: int = 0,
        inst_pointer_idx: int = 0,
        uram_row_stride_z: int = 1,
        stride_jump: int = 0,
        lalu_scalar: Optional[int] = None,
    ) -> None:
        """
        PBI init via :meth:`ue_op_descriptor` with ``inst_type=INSTRUCTION_PBI_SET`` and
        ``pbi_mode=PBI_MODE_INIT``. Sets all pointer-register fields from the literal arguments;
        ``inst_pointer_idx`` (``w[0][15:12]``) selects which pointer row to write.

        ``uram_row_stride_z`` initialises pointer-row field 10 (URAM-B read stride, encoded in the
        descriptor as ``inst_uram_row_stride_z``). Defaults to 1 so existing callers that never
        set it continue to work correctly for standard dot-product kernels.

        Pointer-row field 11 mirrors the descriptor's shared scalar/stride field. Use
        ``stride_jump`` for memcpy/writeback pointers and ``lalu_scalar`` for arithmetic pointers;
        they are semantic aliases for the same bits. When ``lalu_scalar`` is provided it takes the
        field value and ``stride_jump`` must remain zero.
        """
        if self.capture_buffer is None:
            print("ERROR: generate_instruction_pbi_init() called but capture_buffer is not initialized!")
            return
        if self.capture_count >= MAX_DECODER_INSTRUCTIONS:
            print(f"ERROR: generate_instruction_pbi_init() capture_count >= MAX ({MAX_DECODER_INSTRUCTIONS})!")
            return
        if lalu_scalar is not None:
            if stride_jump != 0:
                raise ValueError("generate_instruction_pbi_init: stride_jump and lalu_scalar alias "
                                 "the same PBI field; specify only one")
            field_11 = lalu_scalar
        else:
            field_11 = stride_jump

        self.ue_op_descriptor(
            inst_type=INSTRUCTION_PBI_SET,
            inst_pointer_idx=inst_pointer_idx,
            pbi_mode=PBI_MODE_INIT,
            broadcast_mode=0,
            max_clear_en=0,
            stride_z=uram_row_stride_z,   # descriptor[221:210] → pointer-row field 10
            lalu_a=0,
            lalu_b=0,
            lalu_mode=LALU_MODE.BYPASS.value,
            scalar=field_11,               # descriptor[206:186] → shared pointer-row field 11
            uram_bram=0,
            uram_section=URAM_SECTION.URAM_A.value,
            uram_dst_addr=uram_dst_addr,
            dram_to_uram_cpy_start=0,
            uram_wb_addr=uram_wb_addr,
            uram_write_src=URAM_WRITE_SRC.URAM_WRITE_BACK.value,
            mode=UE_MODE.DOT_PRODUCT,
            data_type=0,
            uram_a_start_addr=uram_a_start_addr,
            uram_b_start_addr=uram_b_start_addr,
            uram_length=uram_length,
            dma_start_addr=dram_shared_addr,
            dma_length=dma_length,
            output_size=output_size,
            bias_adder_en=0,
            stride_bytes_per_chunk=0,
            stride_jump_bytes=0,
            fmax_context_addr=fmax_context_addr,
        )

    def generate_instruction_pbi_inc(
        self,
        dram_shared_addr: int = 0,
        dma_length: int = 0,
        output_size: int = 0,
        uram_length: int = 0,
        uram_a_start_addr: int = 0,
        uram_b_start_addr: int = 0,
        uram_wb_addr: int = 0,
        uram_dst_addr: int = 0,
        fmax_context_addr: int = 0,
        inst_pointer_idx: int = 0,
        general_reg_src: Optional[int] = None,
        pbi_field_select: PBI_FIELD = PBI_FIELD.DRAM_ADDR,
    ) -> None:
        """
        PBI increment update via :meth:`ue_op_descriptor` with ``inst_type=INSTRUCTION_PBI_SET``
        and ``inst_pointer_idx`` in ``w[0][15:12]``.

        When ``general_reg_src`` is ``None``: ``pbi_mode=PBI_MODE_INC`` (plain increment; every
        field becomes row + descriptor delta).
        When ``general_reg_src`` is provided: ``pbi_mode=PBI_MODE_REG`` — the field selected by
        ``pbi_field_select`` is written directly from GPR ``general_reg_src`` (its delta is
        discarded), while all other fields still take the increment path.
        """
        if self.capture_buffer is None:
            print("ERROR: generate_instruction_pbi_inc() called but capture_buffer is not initialized!")
            return
        if self.capture_count >= MAX_DECODER_INSTRUCTIONS:
            print(f"ERROR: generate_instruction_pbi_inc() capture_count >= MAX ({MAX_DECODER_INSTRUCTIONS})!")
            return

        pbi_mode = PBI_MODE_REG if general_reg_src is not None else PBI_MODE_INC
        self.ue_op_descriptor(
            inst_type=INSTRUCTION_PBI_SET,
            inst_pointer_idx=inst_pointer_idx,
            pbi_mode=pbi_mode,
            pbi_field_select=pbi_field_select,
            broadcast_mode=0,
            max_clear_en=0,
            stride_z=0,
            lalu_a=0,
            lalu_b=0,
            lalu_mode=LALU_MODE.BYPASS.value,
            scalar=0,
            uram_bram=0,
            uram_section=URAM_SECTION.URAM_A.value,
            uram_dst_addr=uram_dst_addr,
            dram_to_uram_cpy_start=0,
            uram_wb_addr=uram_wb_addr,
            uram_write_src=URAM_WRITE_SRC.URAM_WRITE_BACK.value,
            mode=UE_MODE.DOT_PRODUCT,
            data_type=0,
            uram_a_start_addr=uram_a_start_addr,
            uram_b_start_addr=uram_b_start_addr,
            uram_length=uram_length,
            dma_start_addr=dram_shared_addr,
            dma_length=dma_length,
            output_size=output_size,
            bias_adder_en=0,
            stride_bytes_per_chunk=0,
            stride_jump_bytes=0,
            general_reg_src=general_reg_src,
            fmax_context_addr=fmax_context_addr,
        )

    def generate_instruction_clear_fmax(self) -> None:
        """
        Generate an dummy pbi instruction using reserved register 0 to clear the fmax context
        """
        if self.capture_buffer is None:
            print("ERROR: generate_instruction_pbi_init() called but capture_buffer is not initialized!")
            return
        if self.capture_count >= MAX_DECODER_INSTRUCTIONS:
            print(f"ERROR: generate_instruction_pbi_init() capture_count >= MAX ({MAX_DECODER_INSTRUCTIONS})!")
            return

        self.ue_arithmetic_op(
            0,  # broadcast_mode (not used for bf16 matvec operation)
            1,  # max_clear_en
            1,  # stride_z
            0,  # lalu_a
            0,  # lalu_b
            LALU_MODE.BYPASS.value,  # lalu_mode
            0,  # scalar
            URAM_SECTION.URAM_A.value,
            0,  # uram_dst_addr
            0,
            URAM_WRITE_SRC.URAM_WB_DISABLE.value,  # uram_write_src
            UE_MODE.ELTWISE_ADD,
            0,  # data_type
            0,
            0,  # uram_b_start_addr
            1,
            0,  # dma_start_addr
            0,  # dma_length
            0,  # output_size
            bias_adder_en=0,
            fmax_context_addr=0,
        )

    def write_captured_instructions_to_dram(self, start_addr: int = DRAM_INSTRUCTION_ADDR) -> int:
        """
        Write all captured instructions to DRAM

        Args:
            start_addr: Starting DRAM address to write instructions (default: DRAM_INSTRUCTION_ADDR)

        Returns:
            Number of bytes written, or -1 on error
        """
        if start_addr is None:
            start_addr = DRAM_INSTRUCTION_ADDR

        if not self.capture_buffer:
            print("Warning: No captured instructions to write. Capture buffer is empty.")
            return 0

        if self.capture_count == 0:
            print("Warning: No captured instructions to write. Capture count is 0.")
            return 0

        # Pad the captured stream to a 64 B (512-bit) boundary so the next
        # program/instruction DRAM allocation also starts on that boundary.
        self.pad_capture_to_64b_boundary()

        # Each instruction is 32 bytes (256 bits)
        total_bytes = self.capture_count * 32

        # Combine all instructions into a single byte array
        all_instructions_bytes = bytearray()
        for inst in self.capture_buffer:
            all_instructions_bytes.extend(inst.get_bytes())

        # Convert to bytes
        instructions_bytes = bytes(all_instructions_bytes)

        # Write to DRAM
        print(f"Writing {self.capture_count} captured instructions ({total_bytes} bytes) to DRAM at 0x{start_addr:x}...")
        bytes_written = self.dma_write(DMA_DEVICE_H2C, start_addr, instructions_bytes, total_bytes)

        if bytes_written == total_bytes:
            print(f"Successfully wrote {bytes_written} bytes ({self.capture_count} instructions) to DRAM")
        else:
            print(f"Warning: Expected to write {total_bytes} bytes, but only wrote {bytes_written} bytes")

        return bytes_written

    def start_execute_from_dram(self, instruction_addr: Optional[int] = None):
        """
        Start executing instructions from DRAM

        This function configures the hardware to execute instructions stored in DRAM.
        It sets the instruction control register to enable execution and points to
        the DRAM address where instructions are stored.

        Args:
            instruction_addr: DRAM address where instructions are stored
                            (default: DRAM_INSTRUCTION_ADDR)
        """
        if instruction_addr is None:
            instruction_addr = DRAM_INSTRUCTION_ADDR
        self.write_reg32(UE_INSTRUCTION_ADDR, ue_35bit_addr_shifter(instruction_addr))

    def program_execute(self, program_start_addr: Optional[int] = None, timeout: float = 50.0, flops: float = None) -> None:
        """Execute compiled program from DRAM instruction memory.
        """
        if program_start_addr is None:
            program_start_addr = DRAM_INSTRUCTION_ADDR
        print(f"Execute program start at 0x{program_start_addr:X}")
        self.start_execute_from_dram(program_start_addr)
        latency, flop_rate_program = 0, 0
        if timeout == 0:
            print("Program started")
        else:
            self.wait_queue(timeout)
            latency = self.report_latency_in_us()
            print(f"    Total program execution latency = {latency} us")
            if flops is not None:
                flop_rate_program, _ = self.report_flop_rate_gflops(flops)
                print(f"Report FLOPS for program execution: {flop_rate_program:.2f} GFLOPS")
        return latency, flop_rate_program

    def software_reset(self):
        """
        Trigger a software reset of the engine via the UE_QUEUE_CTRL register
        and reinitialize all hardware and software state.

        Writes bit[31]=1 (reset command flag) and bit[15]=1 (sw_reset_pending)
        to UE_QUEUE_CTRL_ADDR. The hardware defers the actual reset pulse until
        the AXI write response completes and the AXI master is idle, avoiding
        protocol violations from aborting in-flight DMA bursts.
        """
        SW_RESET_CMD = 0x80008000
        self.write_reg32(UE_QUEUE_CTRL_ADDR, SW_RESET_CMD)
        self.wait_queue(1.0) # 1 seconds timeout
        self.init_unified_engine()
        self._inst_id = 0
        print("Software reset complete.")

    def report_timing_and_instruction_count(self):
        """
        Report timing and instruction count
        """
        latency = self.read_reg32(UE_LATENCY_COUNT_ADDR) * UE_PIPELINE_COUNTER_CLK_DIV
        instruction_count = self.read_reg32(UE_INSTRUCTION_CTL_ADDR)
        print(f"Latency: {latency * self._clock_period_ns / 1e3:.3f} us, Instruction count: {instruction_count}")
        print(f"Latency in cycles: {latency}")
        return latency, instruction_count

    def report_latency_in_us(self):
        """
        Report latency
        """
        return self.read_reg32(UE_LATENCY_COUNT_ADDR) * UE_PIPELINE_COUNTER_CLK_DIV * self._clock_period_ns / 1e3

    def report_flop_rate_gflops(self, num_flops: int):
        """
        Report flop rate and gflops ratio of peak throughput
        """
        cycles = self.read_reg32(UE_LATENCY_COUNT_ADDR) * UE_PIPELINE_COUNTER_CLK_DIV
        gflops_ratio = num_flops / 1.28 / cycles
        return num_flops / (cycles * self._clock_period_ns), gflops_ratio

    # Fixed FP8 E4M3FN lookup, indexed by byte code 0x00..0xFF. All 254
    # finite values listed exactly (powers-of-two and 1/8-step fractions
    # thereof; every entry is representable in bf16 without rounding).
    # Codes 0x7F / 0xFF are NaN under E4M3FN and must never be emitted by
    # quantization, so they're sentinels here (math.nan) and the snap-to-
    # grid table built below filters them out. Negative half (0x80..0xFF)
    # mirrors the positive half with 0x80 = -0.
    _FP8_E4M3FN_VALUE_BY_CODE: Tuple[float, ...] = (
        # 0x00..0x07  (positive subnormals, E=0)
        0.0, 1/512, 2/512, 3/512, 4/512, 5/512, 6/512, 7/512,
        # 0x08..0x0F  (E=1)
        1.0/64, 1.125/64, 1.25/64, 1.375/64, 1.5/64, 1.625/64, 1.75/64, 1.875/64,
        # 0x10..0x17  (E=2)
        1.0/32, 1.125/32, 1.25/32, 1.375/32, 1.5/32, 1.625/32, 1.75/32, 1.875/32,
        # 0x18..0x1F  (E=3)
        1.0/16, 1.125/16, 1.25/16, 1.375/16, 1.5/16, 1.625/16, 1.75/16, 1.875/16,
        # 0x20..0x27  (E=4)
        1.0/8, 1.125/8, 1.25/8, 1.375/8, 1.5/8, 1.625/8, 1.75/8, 1.875/8,
        # 0x28..0x2F  (E=5)
        1.0/4, 1.125/4, 1.25/4, 1.375/4, 1.5/4, 1.625/4, 1.75/4, 1.875/4,
        # 0x30..0x37  (E=6)
        1.0/2, 1.125/2, 1.25/2, 1.375/2, 1.5/2, 1.625/2, 1.75/2, 1.875/2,
        # 0x38..0x3F  (E=7, bias point: 1.0 .. 1.875)
        1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875,
        # 0x40..0x47  (E=8)
        2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75,
        # 0x48..0x4F  (E=9)
        4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,
        # 0x50..0x57  (E=10)
        8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        # 0x58..0x5F  (E=11)
        16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0,
        # 0x60..0x67  (E=12)
        32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0,
        # 0x68..0x6F  (E=13)
        64.0, 72.0, 80.0, 88.0, 96.0, 104.0, 112.0, 120.0,
        # 0x70..0x77  (E=14)
        128.0, 144.0, 160.0, 176.0, 192.0, 208.0, 224.0, 240.0,
        # 0x78..0x7F  (E=15; 0x7F = +NaN)
        256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, math.nan,
        # 0x80..0x87  (negative subnormals, E=0)
        -0.0, -1/512, -2/512, -3/512, -4/512, -5/512, -6/512, -7/512,
        # 0x88..0x8F
        -1.0/64, -1.125/64, -1.25/64, -1.375/64, -1.5/64, -1.625/64, -1.75/64, -1.875/64,
        # 0x90..0x97
        -1.0/32, -1.125/32, -1.25/32, -1.375/32, -1.5/32, -1.625/32, -1.75/32, -1.875/32,
        # 0x98..0x9F
        -1.0/16, -1.125/16, -1.25/16, -1.375/16, -1.5/16, -1.625/16, -1.75/16, -1.875/16,
        # 0xA0..0xA7
        -1.0/8, -1.125/8, -1.25/8, -1.375/8, -1.5/8, -1.625/8, -1.75/8, -1.875/8,
        # 0xA8..0xAF
        -1.0/4, -1.125/4, -1.25/4, -1.375/4, -1.5/4, -1.625/4, -1.75/4, -1.875/4,
        # 0xB0..0xB7
        -1.0/2, -1.125/2, -1.25/2, -1.375/2, -1.5/2, -1.625/2, -1.75/2, -1.875/2,
        # 0xB8..0xBF
        -1.0, -1.125, -1.25, -1.375, -1.5, -1.625, -1.75, -1.875,
        # 0xC0..0xC7
        -2.0, -2.25, -2.5, -2.75, -3.0, -3.25, -3.5, -3.75,
        # 0xC8..0xCF
        -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -7.0, -7.5,
        # 0xD0..0xD7
        -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0,
        # 0xD8..0xDF
        -16.0, -18.0, -20.0, -22.0, -24.0, -26.0, -28.0, -30.0,
        # 0xE0..0xE7
        -32.0, -36.0, -40.0, -44.0, -48.0, -52.0, -56.0, -60.0,
        # 0xE8..0xEF
        -64.0, -72.0, -80.0, -88.0, -96.0, -104.0, -112.0, -120.0,
        # 0xF0..0xF7
        -128.0, -144.0, -160.0, -176.0, -192.0, -208.0, -224.0, -240.0,
        # 0xF8..0xFF  (0xFF = -NaN)
        -256.0, -288.0, -320.0, -352.0, -384.0, -416.0, -448.0, math.nan,
    )

    @classmethod
    def _fp8_e4m3fn_value_code_table(cls, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Materialize the FP8 E4M3FN snap-to-grid lookup as torch tensors.

        Returns ``(values_t, codes_t)`` where ``values_t`` is a float32
        tensor of the 254 finite FP8 E4M3FN values from
        :attr:`_FP8_E4M3FN_VALUE_BY_CODE` and ``codes_t`` is the matching
        uint8 tensor of byte codes. The +/-NaN entries at 0x7F / 0xFF are
        filtered out so the table can be used directly for absmax
        snap-to-grid quantization without ever emitting a NaN code.
        """
        values = [v for v in cls._FP8_E4M3FN_VALUE_BY_CODE if not math.isnan(v)]
        codes = [c for c, v in enumerate(cls._FP8_E4M3FN_VALUE_BY_CODE) if not math.isnan(v)]
        return (torch.tensor(values, dtype=torch.float32, device=device),
                torch.tensor(codes, dtype=torch.uint8, device=device))

    def quantize_weight(self,
                        weight: torch.Tensor,
                        N: int,
                        K: int,
                        data_type: TYPE,
                        int_variant: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a bf16 weight matrix (N, K) with absmax quantization.

        Width is selected by ``data_type`` (TYPE.IF4 -> 4-bit packed nibbles,
        TYPE.IF8 -> 8-bit raw bytes). Within each width, the INT vs FP variant
        is selected by ``int_variant`` and is communicated to the hardware via
        the sign bit of the per-block bf16 scale: negative -> INT (two's
        complement codes), positive -> FP (E2M1 / E4M3 codes). The hardware
        always uses ``|scale|`` as the effective multiplier.
        """
        assert data_type in (TYPE.IF4, TYPE.IF8), \
            f"data_type={data_type} must be one of TYPE.IF4, TYPE.IF8"
        if N % UE_VECTOR_SIZE != 0:
            raise ValueError(f"N={N} must be a multiple of {UE_VECTOR_SIZE}")
        if K % UE_VECTOR_SIZE != 0:
            raise ValueError(f"K={K} must be a multiple of {UE_VECTOR_SIZE}")
        assert weight.dim() == 2, "Weight must be 2D"
        assert weight.dtype == torch.bfloat16, "Weight must be bfloat16"
        assert weight.shape[0] == N and weight.shape[1] == K, f"Weight shape {weight.shape} must match ({N}, {K})"

        matrix = weight.contiguous()  # (N, K)
        matrix_flat = matrix.flatten()
        num_elements = matrix_flat.numel()
        num_blocks = (num_elements + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE

        fp4_values = torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                    dtype=torch.bfloat16, device=matrix.device)
        # FP8 E4M3FN code-to-value table (excludes NaN). Built once outside the
        # per-block loop and only consulted on the IF8-FP path below.
        fp8_values_t, fp8_codes_t = self._fp8_e4m3fn_value_code_table(matrix.device)
        if data_type == TYPE.IF4:
            if int_variant:
                max_val_bf16 = torch.tensor(7.0, dtype=torch.bfloat16, device=matrix.device)
                clamp_min, clamp_max = -8, 7
            else:
                max_val_bf16 = torch.tensor(6.0, dtype=torch.bfloat16, device=matrix.device)
        else:  # TYPE.IF8
            if int_variant:
                max_val_bf16 = torch.tensor(127.0, dtype=torch.bfloat16, device=matrix.device)
                clamp_min, clamp_max = -128, 127
            else:
                # FP8 E4M3 finite max (excluding NaN at 0x7F/0xFF).
                max_val_bf16 = torch.tensor(448.0, dtype=torch.bfloat16, device=matrix.device)

        quantized_int8 = torch.zeros(num_elements, dtype=torch.int8, device=matrix.device)
        scales_bf16 = torch.zeros(num_blocks, dtype=torch.bfloat16, device=matrix.device)
        for i in range(num_blocks):
            start = i * UE_VECTOR_SIZE
            end = min(start + UE_VECTOR_SIZE, num_elements)
            block = matrix_flat[start:end]
            block_bf16 = block.to(torch.bfloat16)
            abs_block = block_bf16.abs()
            max_abs = abs_block.max()
            if float(max_abs.item()) == 0.0:
                scale_bf16 = torch.tensor(1.0, dtype=torch.bfloat16, device=matrix.device)
            else:
                scale_bf16 = max_abs / max_val_bf16
            scales_bf16[i] = scale_bf16
            if float(max_abs.item()) == 0.0:
                q_block = torch.zeros(end - start, dtype=torch.int8, device=matrix.device)
            else:
                scaled = block_bf16 / scale_bf16
                if data_type == TYPE.IF4 and not int_variant:
                    # FP4 E2M1: snap to the 16-entry value table.
                    scaled_expanded = scaled.unsqueeze(-1)
                    fp4_values_expanded = fp4_values.unsqueeze(0)
                    distances = torch.abs(scaled_expanded - fp4_values_expanded)
                    closest_indices = torch.argmin(distances, dim=1)
                    fp4_codes = torch.tensor([
                        0b1111, 0b1110, 0b1101, 0b1100, 0b1011, 0b1010, 0b1001, 0b1000,
                        0b0000, 0b0001, 0b0010, 0b0011, 0b0100, 0b0101, 0b0110, 0b0111,
                    ], dtype=torch.int8, device=matrix.device)
                    q_block = fp4_codes[closest_indices]
                elif data_type == TYPE.IF4 and int_variant:
                    rounded = torch.round(scaled)
                    clamped = rounded.clamp(clamp_min, clamp_max)
                    q_block = clamped.to(torch.int8)
                elif data_type == TYPE.IF8 and int_variant:
                    rounded = torch.round(scaled)
                    clamped = rounded.clamp(clamp_min, clamp_max)
                    q_block = clamped.to(torch.int8)
                else:
                    # IF8 + FP path: snap to the FP8 E4M3FN finite-value
                    # table and emit the matching byte code. The on-chip
                    # fp8_to_bf19 module (Vivado/hdl/fp8_to_bf19.sv) decodes
                    # the same byte to bf19 1:1, so the SW reference and HW
                    # output match exactly for non-NaN codes.
                    distances = torch.abs(scaled.to(torch.float32).unsqueeze(-1)
                                          - fp8_values_t.unsqueeze(0))
                    closest_indices = torch.argmin(distances, dim=-1)
                    # uint8 byte code reinterpreted into the int8 storage
                    # buffer (bit pattern is preserved; quantized_int8 is
                    # later viewed as uint8 for the DMA write).
                    q_block = fp8_codes_t[closest_indices].contiguous().view(torch.int8)
            quantized_int8[start:end] = q_block

        # Sign-bit encodes the INT vs FP variant for the hardware: negative
        # scale -> INT path. Magnitude is the effective multiplier.
        if int_variant:
            scales_bf16 = -scales_bf16

        if data_type == TYPE.IF8:
            # IF8: send raw bytes (no packing)
            quantized_bytes = quantized_int8.view(torch.uint8)  # reinterpret as uint8
            num_bytes = num_elements
            quantized_matrix_dram_addr = self.get_params_dram_addr()
            scale_dram_addr = quantized_matrix_dram_addr + num_bytes
            self.dma_write(DMA_DEVICE_H2C, quantized_matrix_dram_addr, quantized_bytes, num_bytes)
            self.dma_write(DMA_DEVICE_H2C, scale_dram_addr, scales_bf16.view(torch.uint16), num_blocks * 2)
            variant_str = "INT8" if int_variant else "FP8"
            print(f"IF8 ({variant_str}): wrote {num_bytes} raw bytes + {num_blocks*2} scale bytes")
            self.allocate_params_dram(num_bytes + num_blocks * 2)
        else:
            # IF4: pack two 4-bit codes per byte (low nibble first).
            num_packed_bytes = (num_elements + 1) // 2
            packed_int4 = torch.zeros(num_packed_bytes, dtype=torch.uint8, device=matrix.device)
            for i in range(0, num_elements, 2):
                byte_idx = i // 2
                if i + 1 < num_elements:
                    val1 = quantized_int8[i].item()
                    val2 = quantized_int8[i + 1].item()
                    packed_int4[byte_idx] = ((val2 & 0xF) << 4) | (val1 & 0xF)
                else:
                    val1 = quantized_int8[i].item()
                    packed_int4[byte_idx] = val1 & 0xF
            quantized_matrix_dram_addr = self.get_params_dram_addr()
            scale_dram_addr = quantized_matrix_dram_addr + num_packed_bytes
            self.dma_write(DMA_DEVICE_H2C, quantized_matrix_dram_addr, packed_int4, num_packed_bytes)
            self.dma_write(DMA_DEVICE_H2C, scale_dram_addr, scales_bf16.view(torch.uint16), num_blocks * 2)
            variant_str = "INT4" if int_variant else "FP4"
            print(f"IF4 ({variant_str}): wrote {num_packed_bytes} packed bytes + {num_blocks*2} scale bytes")
            self.allocate_params_dram(num_packed_bytes + num_blocks * 2)

        print(f"Quantized matrix and scales written to DRAM at 0x{quantized_matrix_dram_addr:x} and 0x{scale_dram_addr:x}")
        return quantized_matrix_dram_addr, scale_dram_addr

    def quantize_weight_simulate(self,
                                 weight: torch.Tensor,
                                 data_type: TYPE,
                                 int_variant: bool = False) -> torch.Tensor:
        """
        Software model of ``quantize_weight`` followed by the on-chip
        IF4 / IF8 dequantize path. Returns the bf16 matrix the hardware
        would emit -- ``code_table[byte] * |scale|`` per element -- using
        the same per-block absmax + snap-to-grid quantization rules.

        This is the comparison reference for IF4 / IF8 correctness tests:
        no DRAM round-trip is needed, and the result is bit-exact with
        the HW dequantize output for non-NaN codes (FP4 / FP8 lookup
        values fit losslessly in bf16; INT codes are integers).

        ``data_type``: TYPE.IF4 or TYPE.IF8.
        ``int_variant``: True selects the INT path (round-and-clamp);
        False selects the FP path (snap to NVFP4 / FP8 E4M3FN grid).
        """
        assert data_type in (TYPE.IF4, TYPE.IF8), \
            f"data_type={data_type} must be one of TYPE.IF4, TYPE.IF8"
        assert weight.dim() == 2, "Weight must be 2D"
        assert weight.dtype == torch.bfloat16, "Weight must be bfloat16"

        matrix = weight.contiguous()
        matrix_flat = matrix.flatten()
        num_elements = matrix_flat.numel()
        num_blocks = (num_elements + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE

        if data_type == TYPE.IF4:
            max_val = torch.tensor(7.0 if int_variant else 6.0,
                                   dtype=torch.bfloat16, device=matrix.device)
            fp_values = torch.tensor(
                [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0,
                  0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0],
                dtype=torch.bfloat16, device=matrix.device)
        else:  # TYPE.IF8
            max_val = torch.tensor(127.0 if int_variant else 448.0,
                                   dtype=torch.bfloat16, device=matrix.device)
            if not int_variant:
                fp8_values_t, _ = self._fp8_e4m3fn_value_code_table(matrix.device)

        out_flat = torch.zeros(num_elements, dtype=torch.bfloat16, device=matrix.device)
        for i in range(num_blocks):
            start = i * UE_VECTOR_SIZE
            end = min(start + UE_VECTOR_SIZE, num_elements)
            block = matrix_flat[start:end]
            max_abs = block.abs().max()
            if float(max_abs.item()) == 0.0:
                continue
            scale_bf16 = (max_abs / max_val).to(torch.bfloat16)
            scaled = block / scale_bf16
            if data_type == TYPE.IF4 and not int_variant:
                distances = torch.abs(scaled.unsqueeze(-1) - fp_values.unsqueeze(0))
                dequant = fp_values[torch.argmin(distances, dim=-1)]
            elif data_type == TYPE.IF4 and int_variant:
                dequant = scaled.round().clamp(-8, 7).to(torch.bfloat16)
            elif data_type == TYPE.IF8 and int_variant:
                dequant = scaled.round().clamp(-128, 127).to(torch.bfloat16)
            else:  # IF8 + FP
                distances = torch.abs(scaled.to(torch.float32).unsqueeze(-1)
                                      - fp8_values_t.unsqueeze(0))
                dequant = fp8_values_t[torch.argmin(distances, dim=-1)].to(torch.bfloat16)
            out_flat[start:end] = (dequant.to(torch.float32)
                                   * float(scale_bf16.item())).to(torch.bfloat16)
        return out_flat.reshape(matrix.shape)

    @staticmethod
    def float_to_bf19(f: float) -> int:
        """Convert float to bf19 format"""
        # Convert float to bits
        bits = struct.unpack('I', struct.pack('f', f))[0]

        sign = (bits >> 31) & 0x1       # IEEE754 sign
        exp8 = (bits >> 23) & 0xFF      # IEEE754 biased exponent (0..255)
        frac23 = bits & 0x7FFFFF        # IEEE754 23-bit fraction

        exn, exp_field, frac10 = 0, 0, 0

        if exp8 == 0xFF:                          # Inf / NaN
            exn = 0x2 if frac23 == 0 else 0x3     # 10: Inf, 11: NaN
            exp_field = 0
            frac10 = 0 if frac23 == 0 else (frac23 >> 13)
        elif exp8 == 0 and frac23 == 0:           # ±0
            exn = 0x0                              # 00: zero
            exp_field = 0
            frac10 = 0
        else:                                      # normals and subnormals
            exn = 0x1                               # 01: "normal" class
            exp_field = exp8                        # store IEEE raw exponent directly
            frac10 = frac23 >> 13                   # top 10 bits (truncate)

        # Pack: [20:19] exn | [18] sign | [17:10] exp8 | [9:0] frac10
        bf19 = ((exn << 19) |
                (sign << 18) |
                ((exp_field & 0xFF) << 10) |
                (frac10 & 0x3FF))

        return bf19 & 0x1FFFFF  # 21-bit value

    def float_to_bf16(self, f: float) -> int:
        """Convert float to bf16 format (truncate lower 16 bits of mantissa)"""
        # Convert float to bits
        bits = struct.unpack('I', struct.pack('f', f))[0]
        # BF16 is simply the upper 16 bits of IEEE754 float32
        return (bits >> 16) & 0xFFFF

    @staticmethod
    def bf19_to_float(bf19: int) -> float:
        """Convert bf19 to float (simplified)"""
        # Extract fields from bf19 format
        exn = (bf19 >> 19) & 0x3
        sign = (bf19 >> 18) & 0x1
        exp = (bf19 >> 10) & 0xFF
        frac = bf19 & 0x3FF

        # Simplified conversion - for full implementation see C code
        # This is a placeholder that approximates the value
        if exn == 0x0:  # Zero
            return 0.0 if sign == 0 else -0.0
        elif exn == 0x2:  # Inf
            return float('inf') if sign == 0 else float('-inf')
        elif exn == 0x3:  # NaN
            return float('nan')
        else:  # Normal
            # Approximate conversion (simplified)
            # In real implementation, this would properly reconstruct the float
            return float(exp) * (1.0 + frac / 1024.0) * (-1 if sign else 1)


def check_isa_jumps(
    instructions: list,
    base_addr: int,
    name: str = "",
    icache_size: int = 512,
) -> list:
    """Static checker for ISA JUMP instructions in a captured instruction buffer.

    Enforces two hardware constraints:
      1. Absolute jump targets (ABSOLUTE / JNZ / JZ) must be 64-byte (2-instruction) aligned.
      2. Relative backward jumps (RELATIVE / RELA_JNZ / RELA_JZ) must be within ``icache_size``
         instructions of the most recent unconditional absolute jump anchor.  The i-cache is
         reloaded when an unconditional absolute jump fires; any relative jump whose backward
         offset exceeds that distance would under-run the cache and fetch stale instructions.

    Args:
        instructions: list of ``Instructions`` objects (the capture buffer).
        base_addr:    DRAM byte address of ``instructions[0]``.  Must be 64-byte aligned.
        name:         Label prepended to every issue string.
        icache_size:  Hardware i-cache depth in instructions (default 512 = 16 KB / 32 B).

    Returns:
        List of issue strings.  Empty list means all checks passed.
    """
    INST_BYTES = 32
    ALIGN_BYTES = 64          # 2 instructions per 512-bit DRAM fetch
    ABS_MODES = {JUMP_MODE_ABSOLUTE, JUMP_MODE_JNZ, JUMP_MODE_JZ}
    REL_MODES = {JUMP_MODE_RELATIVE, JUMP_MODE_RELA_JNZ, JUMP_MODE_RELA_JZ}

    issues = []
    prefix = f"[{name}] " if name else ""

    # Index (in buffer) of the first instruction of the current i-cache window.
    # Updated whenever an unconditional absolute jump fires; None means unknown
    # (no anchor seen yet, or last anchor was outside this buffer).
    anchor_target_idx = None

    for i, inst in enumerate(instructions):
        w = inst.words
        inst_type = _inst_desc_bits(w, 8, 11)
        if inst_type != INSTRUCTION_JUMP:
            continue

        jump_mode     = _inst_desc_bits(w, 32, 35)
        # immediate_value occupies bits [85:54] of the 256-bit descriptor (must match
        # ue_isa_descriptor's encoding and queue_state_module.sv line 302
        # `inst_immediate_value = inst_descriptor[85:54]`):
        #   bits [63:54] → word[1] bits [31:22]  (10 bits, imm[9:0])
        #   bits [85:64] → word[2] bits [21:0]   (22 bits, imm[31:10])
        # (The old [82:51] window read 3 low bits of the `rst` field, scaling every
        #  relative backward offset by 8 and mis-locating absolute targets.)
        immediate     = (_inst_desc_bits(w, 54, 63)) | (_inst_desc_bits(w, 64, 85) << 10)
        inst_addr     = base_addr + i * INST_BYTES
        tag           = f"{prefix}inst[{i}] @ 0x{inst_addr:X}"

        if jump_mode in ABS_MODES:
            target_byte = immediate << 3
            # --- Concern 1: 64-byte alignment ---
            if target_byte % ALIGN_BYTES != 0:
                issues.append(
                    f"{tag}: ABS JUMP (mode={jump_mode}) target 0x{target_byte:X} "
                    f"is NOT 64-byte aligned (mod64={target_byte % ALIGN_BYTES})"
                )
            # Update i-cache anchor for unconditional absolute jumps only.
            # Conditional jumps (JNZ/JZ) may or may not fire; tracking them as
            # definitive anchors would give false safety for the non-firing path.
            if jump_mode == JUMP_MODE_ABSOLUTE:
                target_buf_idx = (target_byte - base_addr) // INST_BYTES
                if 0 <= target_buf_idx < len(instructions):
                    anchor_target_idx = target_buf_idx
                else:
                    # Jump exits this buffer (e.g. preamble → main binary).
                    # Relative jumps after this point are in the new region.
                    anchor_target_idx = None

        elif jump_mode in REL_MODES:
            backward = immediate  # instruction slots to step back
            # --- Concern 2a: within hardware immediate field limit ---
            if backward > icache_size:
                issues.append(
                    f"{tag}: REL JUMP (mode={jump_mode}) backward={backward} "
                    f"exceeds i-cache size {icache_size}"
                )
            # --- Concern 2b: must have a known unconditional anchor ---
            if anchor_target_idx is None:
                issues.append(
                    f"{tag}: REL JUMP (mode={jump_mode}) backward={backward} "
                    f"with NO preceding unconditional absolute anchor in this buffer"
                )
            else:
                # cache_pos: 0-based index of this instruction within the current window.
                cache_pos = i - anchor_target_idx
                # The RTL applies "ptr -= immediate" to the post-increment PC (already incremented
                # in STATE_FETCH), so the landing position is: cache_pos + 1 - backward.
                # Two constraints must hold:
                #   (a) this instruction itself is within the loaded i-cache window
                #   (b) the landing position is >= 0 (no under-run; hardware saturates at 0 but
                #       we require the emitter to compute the exact offset, not rely on saturation)
                if cache_pos >= icache_size:
                    issues.append(
                        f"{tag}: REL JUMP (mode={jump_mode}) backward={backward} "
                        f"— instruction is {cache_pos} slots from anchor[{anchor_target_idx}], "
                        f"OUTSIDE i-cache window ({icache_size} slots)"
                    )
                elif backward > cache_pos + 1:
                    issues.append(
                        f"{tag}: REL JUMP (mode={jump_mode}) backward={backward} "
                        f"would under-run cache (cache_pos={cache_pos}, "
                        f"landing at {cache_pos + 1 - backward}, expected >= 0)"
                    )

    return issues


def calculate_snr(reference, result) -> float:
    """SNR in dB between reference and result (handles DeviceTensor)."""
    if isinstance(reference, DeviceTensor):
        reference = reference.to_cpu()
    if isinstance(result, DeviceTensor):
        result = result.to_cpu()
    error = reference - result
    signal_power = (reference ** 2).mean()
    noise_power = (error ** 2).mean()
    if torch.isnan(reference).any() or torch.isnan(result).any():
        return float('-inf')
    if noise_power > 0:
        return (10 * torch.log10(signal_power / noise_power)).item()
    return float('inf')
