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
UE_PEAK_GFLOPS = 333.3 * 0.128
UE_TRACE_SIZE = 8192
UE_AXI_DATA_WIDTH_BITS = int(os.environ.get("UE_AXI_DATA_WIDTH_BITS", "256"))
# DMA device paths (can be overridden by command-line argument)
DMA_DEVICE_H2C = "/dev/xdma0_h2c_0"
DMA_DEVICE_C2H = "/dev/xdma0_c2h_0"
DMA_DEVICE_USER = "/dev/xdma0_user"  # AXI-Lite user interface for register access

def set_dma_device(device_name: str):
    """Set DMA device paths based on device name (e.g., 'xdma0' -> '/dev/xdma0_*').

    Also rebinds the names in any module that imported them by value
    (``from user_dma_core import DMA_DEVICE_H2C``), so callers that took the
    pre-set_dma_device snapshot still see the right device. Without this,
    models split across xdma0/xdma1 — methods on UnifiedEngine read the live
    module global (correct), while bare references in the model file read the
    stale import snapshot (wrong).
    """
    import sys as _sys
    global DMA_DEVICE_H2C, DMA_DEVICE_C2H, DMA_DEVICE_USER
    old_h2c, old_c2h, old_user = DMA_DEVICE_H2C, DMA_DEVICE_C2H, DMA_DEVICE_USER
    DMA_DEVICE_H2C = f"/dev/{device_name}_h2c_0"
    DMA_DEVICE_C2H = f"/dev/{device_name}_c2h_0"
    DMA_DEVICE_USER = f"/dev/{device_name}_user"
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

# Pipeline component latencies from timing.md (micro values)
# Old pipline depth values as a reference
# UE_PIPELINE_BF19_MULT = 1  # bf19_mult pipeline depth
# UE_PIPELINE_BF19_ADD = 2  # bf19_add pipeline depth
# UE_PIPELINE_CUSTOM_EXP = 3  # custom_exp pipeline depth
# UE_PIPELINE_ADDER_TREE = 16  # adder_tree pipeline depth
# UE_PIPELINE_BF20_ADD = 2  # bf20_adder pipeline depth
## bf20_adder_unit latencies
# UE_LATENCY_BF20_ITR2 = UE_PIPELINE_BF20_ADD + 1 - 1
# UE_LATENCY_BF20_ITR3 = 2*UE_PIPELINE_BF20_ADD + 2 - 1
# UE_LATENCY_BF20_ITRGT3 = 3*UE_PIPELINE_BF20_ADD + 2 - 1

# Pipeline component latencies from bf19_mult.sv, bf19_add.sv, custom_exp.sv, adder_tree.sv
UE_PIPELINE_BF19_MULT = 3  # +1: split bf19_mult rounding adder at bit 10
UE_PIPELINE_BF19_ADD = 3
UE_PIPELINE_CUSTOM_EXP = 6  # Was 5, +1 to split custom_exp multiply and add
UE_PIPELINE_ADDER_TREE = 12
BF20_ADDER_3_CYCLE = True
UE_LATENCY_BF20_ITR2, UE_LATENCY_BF20_ITR3, UE_LATENCY_BF20_ITRGT3 = ((3, 11, 11) if BF20_ADDER_3_CYCLE else (2, 5, 7))

# Pipeline stage counts from timing.md formulas
UE_PIPELINE_STAGES_INPUT_REG = 1  # Input register for DOT_PRODUCT_MODE
UE_PIPELINE_STAGES_DOT_P = 4  # Dot_P Pipeline Stages (Stage 1 + Stage 2 + Mult Result Reg + Adder Tree Input Reg)
UE_PIPELINE_STAGES_RMS = 4  # RMS Pipeline Stages
UE_PIPELINE_STAGES_EXP = 4  # EXP Pipeline Stages (Stage 1 + add_result_reg + custom_exp + exp_result_reg + Adder Tree Input Reg)
UE_PIPELINE_STAGES_MULT = 4  # Mult Pipeline Stages
UE_PIPELINE_STAGES_ADD = 3  # ADD Pipeline Stages

# Mode latencies calculated from timing.md formulas
UE_LATENCY_DOT_PRODUCT = UE_PIPELINE_STAGES_INPUT_REG + UE_PIPELINE_STAGES_DOT_P + UE_PIPELINE_BF19_MULT + UE_PIPELINE_ADDER_TREE - 1
UE_LATENCY_RMS = UE_PIPELINE_STAGES_RMS + UE_PIPELINE_BF19_MULT + UE_PIPELINE_ADDER_TREE - 1
UE_LATENCY_EXP = UE_PIPELINE_STAGES_EXP + UE_PIPELINE_BF19_ADD + UE_PIPELINE_CUSTOM_EXP + UE_PIPELINE_ADDER_TREE - 1
UE_LATENCY_ADD_REDUCE = UE_PIPELINE_STAGES_ADD + UE_PIPELINE_BF19_ADD + UE_PIPELINE_ADDER_TREE - 1
UE_LATENCY_ELTWISE_MUL = UE_PIPELINE_STAGES_MULT + UE_PIPELINE_BF19_MULT - 1
UE_LATENCY_ELTWISE_ADD = UE_PIPELINE_STAGES_ADD + UE_PIPELINE_BF19_ADD - 1
UE_LATENCY_ELTWISE_SUB = UE_LATENCY_ELTWISE_ADD  # SUB reuses ADD pipeline (Y sign-flipped)
UE_LATENCY_ADD_EXP = UE_PIPELINE_STAGES_EXP + UE_PIPELINE_BF19_ADD + UE_PIPELINE_CUSTOM_EXP - 1
UE_LATENCY_ROPE = 8  # Additional mode latency
# Legacy latency values as a reference
# UE_LATENCY_QUANTIZATION = 19  # Additional mode latency (pre timing pipeline update)
# UE_LATENCY_QSCALE = 12  # Additional mode latency (pre timing pipeline update)
UE_LATENCY_MEAN = 20  # Additional mode latency

# LALU pipeline component latencies from timing.md (micro values)
UE_LALU_PIPELINE_FPDIV = 4  # fpdiv pipeline depth (+1 SRT stage at q6 -> q5 boundary)
UE_LALU_PIPELINE_FPSQRT = 4  # fpsqrt pipeline depth (+1 final-stage split at T10/S9)
UE_LALU_PIPELINE_FACT = 10  # sample_1_plus_exp_bx depth: FPMult + custom_exp(6) + FPAdd

# LALU mode latencies calculated from timing.md formulas (shift register delay parameter)
# Pipeline stages are all 1 cycle (Input Reg + intermediate Reg + Output Reg - 1 for overlap)
UE_LALU_LATENCY_SOFTMAX = 1 + UE_LALU_PIPELINE_FPDIV
UE_LALU_LATENCY_RMS = 1 + UE_LALU_PIPELINE_FPSQRT + 1 + UE_LALU_PIPELINE_FPDIV
UE_LALU_LATENCY_ACT = 1 + UE_LALU_PIPELINE_FACT + 1 + UE_LALU_PIPELINE_FPDIV

# Quantization latency values (matching andromeda.c)
# +1 cycle from quantize_fmax BF16 register before bf16_to_bf19 conversion
UE_QUANTIZE_FMAX_PIPELINE = 8
UE_LATENCY_QSCALE = UE_LALU_PIPELINE_FPDIV + UE_QUANTIZE_FMAX_PIPELINE + 2
UE_QINPUT_DELAY = UE_LATENCY_QSCALE + 1
UE_LATENCY_QUANTIZATION = UE_LATENCY_QSCALE + UE_PIPELINE_BF19_MULT + 5

# ISA instruction type constants (matching queue_state_module.sv / andromeda.c)
INSTRUCTION_UE_OP = 0
INSTRUCTION_JUMP = 1
INSTRUCTION_REG_ALU = 2  # INST_TYPE_REG_ALU (general-purpose register ALU)
INSTRUCTION_ADD = INSTRUCTION_REG_ALU  # legacy alias (same as andromeda.c INSTRUCTION_ADD)
INSTRUCTION_REG_REWRITE = 3
INSTRUCTION_FLAG = 4
INSTRUCTION_UE_PBI = 5
INSTRUCTION_PBI_SET = 6
INSTRUCTION_SWI = 8
INSTRUCTION_HALT = 9
INSTRUCTION_NOP = 10  # 4'hA in queue_state_module.sv (INST_TYPE_NOP)

# PBI mode constants (match queue_state_module.sv)
PBI_MODE_INIT = 0
PBI_MODE_INC  = 1
PBI_MODE_REG  = 2

class PBI_FIELD(IntEnum):
    """Field selector for PBI_MODE_REG (pbi_field_select, inst_descriptor[23:20]).
    Matches the case statement in queue_state_module.sv STATE_PBI_UPDATE."""
    DRAM_ADDR            = 0  # next_dram_addr            = regfile_rdata1        [31:0]
    DMA_LENGTH           = 1  # next_dma_length           = regfile_rdata1        [31:0]
    OUTPUT_SIZE          = 2  # next_output_size          = regfile_rdata1        [15:0]
    URAM_ROW_SIZE        = 3  # next_uram_row_size        = regfile_rdata1        [11:0]
    URAM_START_ADDR_Y    = 4  # next_uram_start_addr_y    = regfile_rdata1        [11:0]
    URAM_START_ADDR_Z    = 5  # next_uram_start_addr_z    = regfile_rdata1        [11:0]
    URAM_WRITEB_ADDR     = 6  # next_uram_writeb_addr     = regfile_rdata1        [11:0]
    URAM_ROW_SIZE_Z      = 7  # next_uram_row_size_z      = regfile_rdata1        [11:0]
    URAM_MEMCPY_DST_ADDR = 8  # next_uram_memcpy_dst_addr = regfile_rdata1        [11:0]
    FMX_CONTEXT          = 9  # next_fmx_context          = regfile_rdata1        [5:0]

# FLAG instruction mode constants
FLAG_MODE_SET   = 0  # Assert this engine's flag (signal busy)
FLAG_MODE_CLEAR = 1  # De-assert this engine's flag (signal done)
FLAG_MODE_CHECK = 2  # Spin-wait until target engine's flag is 1

# JUMP mode constants (match queue_state_module.sv / andromeda.c)
JUMP_MODE_ABSOLUTE = 0
JUMP_MODE_JNZ = 2  # absolute if reg != 0
JUMP_MODE_JZ = 3   # absolute if reg == 0
JUMP_MODE_RELATIVE = 4  # inst cache read ptr -= immediate[9:0]
JUMP_MODE_RELA_JNZ = 5
JUMP_MODE_RELA_JZ = 6

# Register ALU sub-modes (isa_mode[3:0]; queue_state_module.sv ALU_MODE_*)
ALU_MODE_INC = 0  # dst = src + 1
ALU_MODE_DEC = 1  # dst = src - 1
ALU_MODE_ADD_REG = 2  # dst = src1 + src2
ALU_MODE_ADD_IMM = 3  # dst = src + immediate
ALU_MODE_SET = 4  # dst = immediate
ALU_MODE_MIN = 5  # dst = min(src1, src2), unsigned
ALU_MODE_SUB = 6  # dst = src1 - src2
# isa_mode 4'b0111 reserved (reg×reg multiply removed in RTL for timing)
ALU_MODE_MUL_IMM = 8  # dst = (src[15:0] * imm[15:0]) & 0xFFFFFFFF, unsigned

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
                 params_dram_base: int = DRAM_START_ADDR,
                 program_dram_base: int = DRAM_INSTRUCTION_ADDR,
                 tensor_dram_base: int = DRAM_ACTIVATION_ADDR,
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
        Allocate the next available general-purpose ISA register (1-15).
        Register 0 is hard-wired zero.

        Returns:
            The allocated register index (1-15).
        """
        if self._isa_reg_counter > 15:
            raise ValueError("Exceeded maximum number of general registers (15)")

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
        assert hw_version == 0x40faba47, f"HW version mismatch: got 0x{hw_version & 0xFFFFFFFF:08x}, expected 0x40faba47. Please update FPGA with commit update_3fa1735.bin using update_flash.py (public release v1.1)"

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

        # Initialize Unified Engine hardware
        print("init_unified_engine()")
        """Initialize Unified Engine hardware"""
        # Set output valid delay register
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

        # Configure delay for the last ALU (matching andromeda.c init_unified_engine)
        ue_lalu_delay = (
            ((UE_QINPUT_DELAY & 0x1F) << 22) +
            (UE_LATENCY_QUANTIZATION << 17) +
            (UE_LATENCY_QSCALE << 13) +
            (UE_LALU_LATENCY_ACT << 8) +
            (UE_LALU_LATENCY_RMS << 4) +
            (UE_LALU_LATENCY_SOFTMAX << 0)
        )
        self.write_reg32(UE_LALU_DELAY_ADDR, ue_lalu_delay)
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
        :meth:`generate_instruction_pbi_inc` when a GPR override is requested). RTL first accumulates
        the instruction delta (INC step), then overrides one field (selected by ``pbi_field_select``)
        from general register ``general_reg_src`` (``w[0][27:24]``).

        reg rewrite:
        - if ``general_reg_src`` is set and ``inst_type != INSTRUCTION_PBI_SET``, ``inst_type`` = ``INSTRUCTION_REG_REWRITE``
          and ``w[1][7:4]`` holds ``inst_src_reg_idx`` (``inst_descriptor[39:36]``).
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

            is_memcpy_mode = mode in (UE_MODE.MEMCPY_FROM_DRAM, UE_MODE.URAM_DRAM_WRITEBACK)
            stride_en = 1 if stride_jump_bytes > 0 else 0
            output_size = stride_bytes_per_chunk if stride_en else output_size
            scalar = stride_jump_bytes if stride_en else scalar

            # [7:0] inst_id; [11:8] inst_type; [15:12] inst_ptr; [19:16] pbi_mode for PBI_SET; [31:16] lalu_a otherwise
            w[0] = ((tid & 0xFF) |
                    ((inst_type & 0xF) << 8) |
                    (((inst_pointer_idx or 0) & 0xF) << 12))
            # PBI_SET: [19:16] pointer_mode, [23:20] field_select; [27:24] pbi_general_reg_idx when PBI_MODE_REG.
            # Other types: [31:16] = lalu_a.
            if int(inst_type) == int(INSTRUCTION_PBI_SET):
                w[0] |= (int(pbi_mode) & 0xF) << 16
                w[0] |= (int(pbi_field_select) & 0xF) << 20
                if int(pbi_mode) == PBI_MODE_REG:
                    assert general_reg_src is not None, "general_reg_src is required for PBI_MODE_REG"
                    w[0] |= (int(general_reg_src) & 0xF) << 24
            else:
                w[0] |= ((lalu_a & 0xFFFF) << 16)
            w[1] = ue_35bit_addr_shifter(dma_start_addr)
            if int(inst_type) == int(INSTRUCTION_REG_REWRITE):
                assert general_reg_src is not None, "general_reg_src is required for REG_REWRITE"
                w[1] |= (general_reg_src & 0xF) << 4
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
                            general_reg_src: Optional[int] = None,
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
            general_reg_src: REG_REWRITE source register index; when set (and inst_pointer_idx is None), emits INSTRUCTION_REG_REWRITE so the DRAM address is taken from this ISA register at runtime.
            inst_pointer_idx: when nonzero, emit PBI-style memcpy (pointer-backed registers are incremented by the immediate value specified in the instruction).
        """
        if inst_pointer_idx is not None:
            inst_type = INSTRUCTION_UE_PBI
            encoded_dram_addr = dram_src_addr
        elif general_reg_src is not None:
            inst_type = INSTRUCTION_REG_REWRITE
            encoded_dram_addr = 0
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
            general_reg_src=general_reg_src,
            fmax_context_addr=0,
        )

    def ue_memcpy_to_dram(self, memcpy_type: int, uram_type: int,
                         uram_src_addr: int, dram_dst_addr: int,
                         memcpy_length_bytes: int,
                         stride_bytes_per_chunk: int = 0,
                         stride_jump_bytes: int = 0,
                         general_reg_src: Optional[int] = None,
                         inst_pointer_idx: Optional[int] = None,
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
            general_reg_src: General purpose register source (default: 0)
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
        elif general_reg_src is not None:
            inst_type = INSTRUCTION_REG_REWRITE
            encoded_dram_addr = 0
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
            general_reg_src=general_reg_src,
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
                   inst_pointer_idx: Optional[int] = None
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

        ``general_reg_src``: when set (and inst_pointer_idx is None), emits INSTRUCTION_REG_REWRITE
        so the DRAM source address is taken from ISA register ``general_reg_src`` at runtime.
        """
        uram_type, uram_start_addr = self.sram_address_to_uram_address(sram_address)
        nbytes = element_size * 2 if memcpy_length_bytes is None else memcpy_length_bytes
        self.ue_memcpy_from_dram(
            accelerator_dram_address,
            nbytes,
            MEMCPY_TYPE.URAM.value,
            uram_start_addr,
            uram_type.value,
            stride_bytes_per_chunk=stride_bytes_per_chunk,
            stride_jump_bytes=stride_jump_bytes,
            general_reg_src=general_reg_src,
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

        ``general_reg_src``: when set (and inst_pointer_idx is None), emits INSTRUCTION_REG_REWRITE
        so the DRAM destination address is taken from ISA register ``general_reg_src`` at runtime.
        """
        uram_type, uram_start_addr = self.sram_address_to_uram_address(sram_address)
        axi_beat_bytes = UE_AXI_DATA_WIDTH_BITS // 8
        if stride_bytes_per_chunk != 0:
            assert stride_bytes_per_chunk % axi_beat_bytes == 0, (
                f"stride_bytes_per_chunk={stride_bytes_per_chunk} must be a multiple of "
                f"AXI beat bytes {axi_beat_bytes} (UE_AXI_DATA_WIDTH_BITS={UE_AXI_DATA_WIDTH_BITS})"
            )
        nbytes = element_size * 2 if memcpy_length_bytes is None else memcpy_length_bytes
        self.ue_memcpy_to_dram(
            MEMCPY_TYPE.URAM.value,
            uram_type.value,
            uram_start_addr,
            accelerator_dram_address,
            nbytes,
            stride_bytes_per_chunk=stride_bytes_per_chunk,
            stride_jump_bytes=stride_jump_bytes,
            general_reg_src=general_reg_src,
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

    def eltwise_core_dram(
        self,
        M: int,
        N: int,
        dram_a: int,
        dram_b: int,
        dram_out: int,
        mode: UE_MODE,
        gpr_M_reg: Optional[int] = None,
    ) -> int:
        """Dispatches based on ``gpr_M_reg``:

        - ``gpr_M_reg`` is a GPR index (1..15): :meth:`eltwise_core_dram_pbi` — one row per ISA
          iteration with runtime row count carried in that register. Captured program has no
          static reference to ``M``; ``M`` is FLOPs-accounting only.
        - ``gpr_M_reg is None`` (default): :meth:`eltwise_core_dram_legacy` — compile-time
          ``m_chunk`` tiling, no GPR needed.

        Returns ``M * N`` (one flop per output BF16 element).
        """
        if gpr_M_reg is not None:
            return self.eltwise_core_dram_pbi(
                M=M, N=N,
                dram_a=dram_a, dram_b=dram_b, dram_out=dram_out,
                mode=mode,
                gpr_M_reg=gpr_M_reg,
            )
        return self.eltwise_core_dram_legacy(
            M=M, N=N,
            dram_a=dram_a, dram_b=dram_b, dram_out=dram_out,
            mode=mode,
        )

    def eltwise_core_dram_legacy(
        self,
        M: int,
        N: int,
        dram_a: int,
        dram_b: int,
        dram_out: int,
        mode: UE_MODE,
    ) -> int:
        """
        Legacy BF16 element-wise ADD / MUL / SUB on ``[M, N]`` tensors in DRAM (row-major) with
        compile-time vertical tiling.

        Chooses ``m_chunk`` from ``URAM_NEAR_FULL_ELEMENTS // N`` (one UE vector of headroom vs
        a full bank) so a single memcpy+eltwise never fills **every** BF16 slot in URAM.

        Emits **ISA** ``loop_start`` / ``loop_end`` over ``M // m_chunk`` full tiles when that count
        is positive: three **PBI** streams (A load, B load, out store) with fixed
        ``dma_length = m_chunk * N * 2`` bytes per iteration and ``chunk_bytes`` as the per-iter
        DRAM stride. Body: PBI memcpy → PBI memcpy → eltwise on ``m_chunk * N`` BF16 → PBI store.

        If ``M % m_chunk`` is nonzero, one trailing **non-PBI** memcpy + eltwise + store covers the
        remaining rows (PBI cannot change ``dma_length`` per iteration inside one loop).

        For **dynamic seq_len / one row per loop iteration** (no ``m_chunk``), use
        :meth:`eltwise_core_dram_pbi`.

        **Staging (fixed):** A at ``0x00000`` (URAM_A), B at ``0x80000`` (URAM_B); output reuses the
        A buffer.

        Args:
            M, N: Logical row/column counts (``N`` multiple of ``UE_VECTOR_SIZE``,
                ``N <= URAM_NEAR_FULL_ELEMENTS`` so one row and every tile stay below a full bank).
            dram_a, dram_b, dram_out: Byte base addresses of A, B, and output.
            mode: ``UE_MODE.ELTWISE_ADD``, ``ELTWISE_MUL``, or ``ELTWISE_SUB``.

        Returns:
            ``M * N`` (one flop per logical output element).
        """
        if mode not in (UE_MODE.ELTWISE_ADD, UE_MODE.ELTWISE_MUL, UE_MODE.ELTWISE_SUB):
            raise ValueError(
                f"eltwise_core_dram_legacy: mode must be ELTWISE_ADD, ELTWISE_MUL, or ELTWISE_SUB, got {mode!r}"
            )
        if M < 1 or N < 1:
            raise ValueError(f"eltwise_core_dram_legacy: require M>=1 and N>=1, got M={M}, N={N}")
        if N % UE_VECTOR_SIZE != 0:
            raise ValueError(
                f"eltwise_core_dram_legacy: N must be a multiple of UE_VECTOR_SIZE={UE_VECTOR_SIZE}, got N={N}"
            )
        if N > URAM_NEAR_FULL_ELEMENTS:
            raise ValueError(
                f"eltwise_core_dram_legacy: N={N} exceeds near-full URAM row ({URAM_NEAR_FULL_ELEMENTS} BF16 slots); "
                "need N <= URAM_NEAR_FULL_ELEMENTS (full bank width is not supported here)."
            )

        def _emit_tile(elements: int) -> None:
            if mode == UE_MODE.ELTWISE_ADD:
                self.eltwise_add_core(0x00000, 0x80000, 0x00000, elements)
            elif mode == UE_MODE.ELTWISE_MUL:
                self.eltwise_mul_core(0x00000, 0x80000, 0x00000, elements)
            else:
                self.eltwise_sub_core(0x00000, 0x80000, 0x00000, elements)

        # Always tile with near-full capacity only (never URAM_FULL_ELEMENTS per op).
        m_chunk = URAM_NEAR_FULL_ELEMENTS // N
        if m_chunk < 1:
            raise ValueError(
                f"eltwise_core_dram_legacy: N={N} too large; need at least one row of BF16 within "
                f"URAM_NEAR_FULL_ELEMENTS={URAM_NEAR_FULL_ELEMENTS}."
            )
        chunk_bytes = m_chunk * N * 2
        tile_elements = m_chunk * N
        num_full = M // m_chunk
        rem_rows = M % m_chunk

        _, a_uram_row = self.sram_address_to_uram_address(0x00000)
        _, b_uram_row = self.sram_address_to_uram_address(0x80000)
        _, out_uram_row = self.sram_address_to_uram_address(0x00000)

        if num_full >= 1:
            ptr_a = self.alloc_inst_ptr()
            ptr_b = self.alloc_inst_ptr()
            ptr_out = self.alloc_inst_ptr()

            self.generate_instruction_pbi_init(
                dram_shared_addr=dram_a,
                dma_length=chunk_bytes,
                output_size=0,
                uram_length=0,
                uram_a_start_addr=0,
                uram_b_start_addr=0,
                uram_wb_addr=0,
                uram_dst_addr=a_uram_row,
                fmax_context_addr=0,
                inst_pointer_idx=ptr_a,
            )
            self.generate_instruction_pbi_init(
                dram_shared_addr=dram_b,
                dma_length=chunk_bytes,
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
                dma_length=chunk_bytes,
                output_size=0,
                uram_length=0,
                uram_a_start_addr=out_uram_row,
                uram_b_start_addr=out_uram_row,
                uram_wb_addr=0,
                uram_dst_addr=0,
                fmax_context_addr=0,
                inst_pointer_idx=ptr_out,
            )

            program_dram_start_addr = self.get_program_dram_addr()
            cur_inst_count = self.capture_count
            self.generate_instruction_jump_abs(
                ue_35bit_addr_shifter(
                    program_dram_start_addr + (cur_inst_count + 1) * INSTRUCTION_SIZE_BYTES
                )
            )
            self.loop_start(num_full)

            self.accelerator_memory_to_sram(
                accelerator_dram_address=chunk_bytes,
                sram_address=0x00000,
                element_size=0,
                inst_pointer_idx=ptr_a,
            )
            self.accelerator_memory_to_sram(
                accelerator_dram_address=chunk_bytes,
                sram_address=0x80000,
                element_size=0,
                inst_pointer_idx=ptr_b,
            )
            _emit_tile(tile_elements)
            self.sram_to_accelerator_memory(
                sram_address=0x00000,
                accelerator_dram_address=chunk_bytes,
                element_size=0,
                inst_pointer_idx=ptr_out,
            )

            outer_loop_size = self.loop_end()
            assert outer_loop_size <= 256, (
                f"eltwise_core_dram_legacy: outer loop body {outer_loop_size} instructions exceeds "
                "i-cache budget 256"
            )

            self.release_inst_ptr(ptr_out)
            self.release_inst_ptr(ptr_b)
            self.release_inst_ptr(ptr_a)

        if rem_rows > 0:
            byte_off = num_full * m_chunk * N * 2
            elements_rem = rem_rows * N
            self.accelerator_memory_to_sram(
                accelerator_dram_address=dram_a + byte_off,
                sram_address=0x00000,
                element_size=elements_rem,
            )
            self.accelerator_memory_to_sram(
                accelerator_dram_address=dram_b + byte_off,
                sram_address=0x80000,
                element_size=elements_rem,
            )
            _emit_tile(elements_rem)
            self.sram_to_accelerator_memory(
                sram_address=0x00000,
                accelerator_dram_address=dram_out + byte_off,
                element_size=elements_rem,
            )

        return M * N

    def eltwise_core_dram_pbi(
        self,
        M: int,
        N: int,
        dram_a: int,
        dram_b: int,
        dram_out: int,
        mode: UE_MODE,
        gpr_M_reg: int,
    ) -> int:
        """
        PBI BF16 element-wise ADD / MUL / SUB on ``[M, N]`` DRAM tensors (**no vertical tiling**).

        Loads/processes/stores **one row per ISA iteration**: ``dma_length = N * 2`` bytes per PBI op,
        DRAM strides ``row_bytes`` each iteration via hardware loop counters (:meth:`loop_start` /
        :meth:`loop_end`). No ``m_chunk`` / remnant tail path unlike :meth:`eltwise_core_dram`.

        Requires ``N <= URAM_NEAR_FULL_ELEMENTS`` so one row fits staging.

        **Staging (fixed):** A at ``0x00000``, B at ``0x80000``, output reuses A.

        **Batch dimension / M (dynamic, required):**
        ``gpr_M_reg`` is a **required** GPR index 1..15 holding the runtime row count. Caller
        must prime that register beforehand (typically with ``ADD_SET``). The hardware loop
        runs for whatever value that register holds at execute time; ``M`` is still required
        as a compile-time argument purely for FLOPs accounting / asserts — the captured
        program contains no static reference to ``M``.

        Returns:
            ``M * N`` (one flop per output BF16 element). Since ``M`` is FLOPs-accounting only,
            callers feeding this to :meth:`report_flop_rate_gflops` should pass the realized
            row count separately if it differs from the compile-time ``M``.
        """
        fn = "eltwise_core_dram_pbi"
        if mode not in (UE_MODE.ELTWISE_ADD, UE_MODE.ELTWISE_MUL, UE_MODE.ELTWISE_SUB):
            raise ValueError(
                f"{fn}: mode must be ELTWISE_ADD, ELTWISE_MUL, or ELTWISE_SUB, got {mode!r}"
            )
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

        row_bytes = N * 2

        def _emit_row() -> None:
            if mode == UE_MODE.ELTWISE_ADD:
                self.eltwise_add_core(0x00000, 0x80000, 0x00000, N)
            elif mode == UE_MODE.ELTWISE_MUL:
                self.eltwise_mul_core(0x00000, 0x80000, 0x00000, N)
            else:
                self.eltwise_sub_core(0x00000, 0x80000, 0x00000, N)

        _, a_uram_row = self.sram_address_to_uram_address(0x00000)
        _, b_uram_row = self.sram_address_to_uram_address(0x80000)
        _, out_uram_row = self.sram_address_to_uram_address(0x00000)

        ptr_a = self.alloc_inst_ptr()
        ptr_b = self.alloc_inst_ptr()
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

        self.release_inst_ptr(ptr_out)
        self.release_inst_ptr(ptr_b)
        self.release_inst_ptr(ptr_a)

        return M * N

    # Broadcast operations ------------------------------------------------------
    def start_queue_broadcast(self, mode: UE_MODE, broadcast_mode: BROADCAST_MODE,
                              uram_src_start_addr: int, uram_wb_start_addr: int,
                              element_size: int, scalar: float = None) -> Optional[torch.Tensor]:
        """
        Start queue for broadcast op: MUL_BROADCAST or ADD_BROADCAST.
        scalar: encoded scalar (e.g. bf16 from float_to_bf16 for MUL/ADD broadcast).
        Wraps ue_arithmetic_op with BROADCAST_MODE.SCALAR_IN_REG.
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
        )

    def broadcast_mul(self, scalar: float,
                            sram_start_addr: int, sram_wb_addr: int,
                            element_size: int) -> Optional[torch.Tensor]:
        """Start queue for broadcast multiply. Wraps ue_arithmetic_op with UE_MODE.MUL_BROADCAST."""
        self.start_queue_broadcast(
            UE_MODE.MUL_BROADCAST, BROADCAST_MODE.SCALAR_IN_REG,
            sram_start_addr, sram_wb_addr, element_size, scalar
        )

    def broadcast_add(self, scalar: float,
                            sram_start_addr: int, sram_wb_addr: int,
                            element_size: int) -> Optional[torch.Tensor]:
        """Start queue for broadcast add. Wraps ue_arithmetic_op with UE_MODE.ADD_BROADCAST."""
        self.start_queue_broadcast(
            UE_MODE.ADD_BROADCAST, BROADCAST_MODE.SCALAR_IN_REG,
            sram_start_addr, sram_wb_addr, element_size, scalar
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


    def rms_norm_core_dram_pbi(self, M: int, N: int, A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, GAMMA_DRAM_ADDR: int, gpr_M_reg: int) -> int:
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

        Caller must have start_capture() active.
        """
        assert M >= 1, "rms_norm_core_dram_pbi() requires M >= 1"
        if not (1 <= gpr_M_reg <= 15):
            raise ValueError(f"rms_norm_core_dram_pbi: gpr_M_reg must be a GPR index 1..15, got {gpr_M_reg}")

        vector_sram_addr = 0x00000
        gamma_sram_addr = 0x80000

        self.accelerator_memory_to_sram(accelerator_dram_address=GAMMA_DRAM_ADDR,
                                        sram_address=gamma_sram_addr,
                                        element_size=N)

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
                           gpr_M_reg: Optional[int] = None) -> None:
        """RMS norm DRAM entrypoint; dispatches based on ``gpr_M_reg``:

        - ``gpr_M_reg`` is a GPR index (1..15): :meth:`rms_norm_core_dram_pbi` — outer row loop trip
          count is taken from that register at runtime (caller must prime it via ``ADD_SET``). The
          captured program has no static reference to ``M``; ``M`` is FLOPs-accounting only.
        - ``gpr_M_reg is None`` (default): :meth:`rms_norm_core_dram_legacy` — compile-time
          ``chunk_size`` tiling, no GPR needed.
        """
        if gpr_M_reg is not None:
            return self.rms_norm_core_dram_pbi(
                M=M,
                N=N,
                A_DRAM_ADDR=A_DRAM_ADDR,
                OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                GAMMA_DRAM_ADDR=GAMMA_DRAM_ADDR,
                gpr_M_reg=gpr_M_reg,
            )

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

    def layer_norm_core_dram(self, M: int, N: int, A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, GAMMA_DRAM_ADDR: int = None, BETA_DRAM_ADDR: int = None) -> None:
        """
        Core layer norm: normalizes vector x -> x / rms(x).
        Args:
            M: number of rows in the input matrix
            N: number of columns in the input matrix
            A_DRAM_ADDR: DRAM address of input matrix
            OUTPUT_DRAM_ADDR: DRAM address for normalized output
            GAMMA_DRAM_ADDR: DRAM address for gamma
            BETA_DRAM_ADDR: DRAM address for beta
        """
        zeros_sram_addr = 0x80000

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
            self.accelerator_memory_to_sram(accelerator_dram_address=cos_dram_addr, sram_address=sram_cos, element_size=N)
            self.overwrite_instruction_with_general_register(tmp_reg)
            self.generate_instruction_add_imm(rope_size_reg, ue_35bit_addr_shifter(sin_dram_addr), tmp_reg)
            self.accelerator_memory_to_sram(accelerator_dram_address=sin_dram_addr, sram_address=sram_sin, element_size=N)
            self.overwrite_instruction_with_general_register(tmp_reg)
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
            self.sram_to_accelerator_memory(sram_address=sram_d, accelerator_dram_address=output_dram_addr, element_size=N)
            self.overwrite_instruction_with_general_register(tmp_reg)
        else:
            self.sram_to_accelerator_memory(sram_address=sram_d, accelerator_dram_address=output_dram_addr, element_size=N)
        if gr_weight_dram != 0:
            self.release_isa_reg()
        return 4 * N
        
    def rope_hf_core_dram(self, M: int, N: int, input_dram_addr: int, output_dram_addr: int, cos_dram_addr: int, sin_dram_addr: int, gpr_M_reg: Optional[int] = None, rope_size_reg: int = None, output_addr_inc_reg: int = None, tmp_reg: int = None) -> int:
        """HF RoPE DRAM entrypoint; dispatches based on ``gpr_M_reg``:

        - ``gpr_M_reg`` is a GPR index (1..15): :meth:`rope_hf_core_dram_pbi` — outer M loop uses
          runtime trip count from that register (caller must prime via ``ADD_SET``). ``M`` is
          FLOPs-accounting only; captured program has no static reference to it.
        - ``gpr_M_reg is None`` (default): :meth:`rope_hf_core_dram_legacy` — Python-unrolled rows.
        """
        if gpr_M_reg is not None:
            return self.rope_hf_core_dram_pbi(M, N, input_dram_addr, output_dram_addr, cos_dram_addr, sin_dram_addr, gpr_M_reg=gpr_M_reg)
        return self.rope_hf_core_dram_legacy(M, N, input_dram_addr, output_dram_addr, cos_dram_addr, sin_dram_addr)

    def rope_hf_core_dram_legacy(self, M: int, N: int, input_dram_addr: int, output_dram_addr: int, cos_dram_addr: int, sin_dram_addr: int) -> int:
        """RoPE (HuggingFace style). Caller must have start_capture() before and stop_capture() after."""
        assert M >= 1, "M must be at least 1"
        assert N % UE_VECTOR_SIZE == 0 and N >= 64, f"N must be a multiple of {UE_VECTOR_SIZE} and >= 64"
        assert N % 2 == 0, "N must be even for RoPE half layout"
        assert N >= 128, "N must be >= 128 so half-vector SRAM offsets are 128-byte aligned"
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
        for row_idx in range(M):
            self.accelerator_memory_to_sram(accelerator_dram_address=input_dram_addr + row_idx * row_bytes, sram_address=sram_x, element_size=N)
            self.accelerator_memory_to_sram(accelerator_dram_address=cos_dram_addr + row_idx * rope_row_bytes, sram_address=sram_cos, element_size=N)
            self.accelerator_memory_to_sram(accelerator_dram_address=sin_dram_addr + row_idx * rope_row_bytes, sram_address=sram_sin, element_size=N)
            self.eltwise_mul_core(vector_A_sram_start_addr=sram_x, vector_B_sram_start_addr=sram_cos, vector_C_sram_wb_addr=sram_a, element_size=N)
            self.eltwise_mul_core(vector_A_sram_start_addr=sram_x + half * bytes_per_elem, vector_B_sram_start_addr=sram_sin, vector_C_sram_wb_addr=sram_bc, element_size=half)
            self.eltwise_mul_core(vector_A_sram_start_addr=sram_x, vector_B_sram_start_addr=sram_sin + half * bytes_per_elem, vector_C_sram_wb_addr=sram_bc + half * bytes_per_elem, element_size=half)
            self.eltwise_add_core(vector_A_sram_start_addr=sram_a, vector_B_sram_start_addr=sram_bc, vector_C_sram_wb_addr=sram_d, element_size=N)
            self.sram_to_accelerator_memory(sram_address=sram_d, accelerator_dram_address=output_dram_addr + row_idx * row_bytes, element_size=N)
        return 4 * M * N

    def rope_hf_core_dram_pbi(self, M: int, N: int, input_dram_addr: int, output_dram_addr: int, cos_dram_addr: int, sin_dram_addr: int, gpr_M_reg: int = None) -> int:
        """PBI-backed HF RoPE over M rows. Caller must have start_capture() before and stop_capture() after.

        **Batch dimension / M (dynamic, required):**
        ``gpr_M_reg`` is a **required** GPR index 1..15 holding the runtime row count. Caller must
        prime that register beforehand (typically with ``ADD_SET``). The hardware loop runs for
        whatever value that register holds at execute time; ``M`` is FLOPs-accounting / template
        only — the captured program has no static reference to ``M``.
        """
        assert M >= 1, "M must be at least 1"
        assert N % UE_VECTOR_SIZE == 0 and N >= 64, f"N must be a multiple of {UE_VECTOR_SIZE} and >= 64"
        assert N % 2 == 0, "N must be even for RoPE half layout"
        assert N >= 128, "N must be >= 128 so half-vector SRAM offsets are 128-byte aligned"
        if gpr_M_reg is None or not (1 <= gpr_M_reg <= 15):
            raise ValueError(f"rope_hf_core_dram_pbi: gpr_M_reg must be a GPR index 1..15, got {gpr_M_reg}")

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

        self.release_inst_ptr(out_ptr)
        self.release_inst_ptr(rope_ptr)
        self.release_inst_ptr(x_ptr)
        return 4 * M * N

    def rope_hf_core_dram_gqa(self, M: int, group_size: int, N: int, input_dram_addr: int, output_dram_addr: int, cos_dram_addr: int, sin_dram_addr: int, gpr_M_reg: Optional[int] = None) -> int:
        """HF GQA RoPE DRAM entrypoint; dispatches based on ``gpr_M_reg``:

        - ``gpr_M_reg`` is a GPR index (1..15): :meth:`rope_hf_core_dram_gqa_pbi` — outer M loop uses
          runtime trip count from that register (caller must prime via ``ADD_SET``). Inner
          ``group_size`` loop remains compile-time. ``M`` is FLOPs-accounting only.
        - ``gpr_M_reg is None`` (default): :meth:`rope_hf_core_dram_gqa_legacy` — Python-unrolled rows.
        """
        if gpr_M_reg is not None:
            return self.rope_hf_core_dram_gqa_pbi(M, group_size, N, input_dram_addr, output_dram_addr, cos_dram_addr, sin_dram_addr, gpr_M_reg=gpr_M_reg)
        return self.rope_hf_core_dram_gqa_legacy(M, group_size, N, input_dram_addr, output_dram_addr, cos_dram_addr, sin_dram_addr)

    def rope_hf_core_dram_gqa_legacy(self, M: int, group_size: int, N: int, input_dram_addr: int, output_dram_addr: int, cos_dram_addr: int, sin_dram_addr: int) -> int:
        """Grouped-query RoPE: input/output rows are [M, group_size, N], RoPE rows are [M, N]."""
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
        sram_d = 0x40000
        sram_cos = 0x80000
        sram_sin = 0x80000 + N * bytes_per_elem
        sram_bc = 0x80000 + N * bytes_per_elem * 2
        for row_idx in range(M):
            self.accelerator_memory_to_sram(accelerator_dram_address=cos_dram_addr + row_idx * rope_row_bytes, sram_address=sram_cos, element_size=N)
            self.accelerator_memory_to_sram(accelerator_dram_address=sin_dram_addr + row_idx * rope_row_bytes, sram_address=sram_sin, element_size=N)
            for group_idx in range(group_size):
                q_row_idx = row_idx * group_size + group_idx
                self.accelerator_memory_to_sram(accelerator_dram_address=input_dram_addr + q_row_idx * row_bytes, sram_address=sram_x, element_size=N)
                self.eltwise_mul_core(vector_A_sram_start_addr=sram_x, vector_B_sram_start_addr=sram_cos, vector_C_sram_wb_addr=sram_a, element_size=N)
                self.eltwise_mul_core(vector_A_sram_start_addr=sram_x + half * bytes_per_elem, vector_B_sram_start_addr=sram_sin, vector_C_sram_wb_addr=sram_bc, element_size=half)
                self.eltwise_mul_core(vector_A_sram_start_addr=sram_x, vector_B_sram_start_addr=sram_sin + half * bytes_per_elem, vector_C_sram_wb_addr=sram_bc + half * bytes_per_elem, element_size=half)
                self.eltwise_add_core(vector_A_sram_start_addr=sram_a, vector_B_sram_start_addr=sram_bc, vector_C_sram_wb_addr=sram_d, element_size=N)
                self.sram_to_accelerator_memory(sram_address=sram_d, accelerator_dram_address=output_dram_addr + q_row_idx * row_bytes, element_size=N)
        return 4 * M * group_size * N

    def rope_hf_core_dram_gqa_pbi(self, M: int, group_size: int, N: int, input_dram_addr: int, output_dram_addr: int, cos_dram_addr: int, sin_dram_addr: int, gpr_M_reg: int = None) -> int:
        """PBI-backed grouped-query RoPE. Q rows are [M, group_size, N], RoPE rows are [M, N].

        **Batch dimension / M (dynamic, required):**
        ``gpr_M_reg`` is a **required** GPR index 1..15 holding the runtime outer-loop trip count.
        Caller must prime that register beforehand (typically with ``ADD_SET``). The inner
        ``group_size`` loop is still compile-time. ``M`` is FLOPs-accounting only — the captured
        program has no static reference to it.
        """
        assert M >= 1, "M must be at least 1"
        assert group_size >= 1, "group_size must be at least 1"
        assert N % UE_VECTOR_SIZE == 0 and N >= 64, f"N must be a multiple of {UE_VECTOR_SIZE} and >= 64"
        assert N % 2 == 0, "N must be even for RoPE half layout"
        assert N >= 128, "N must be >= 128 so half-vector SRAM offsets are 128-byte aligned"
        if gpr_M_reg is None or not (1 <= gpr_M_reg <= 15):
            raise ValueError(f"rope_hf_core_dram_gqa_pbi: gpr_M_reg must be a GPR index 1..15, got {gpr_M_reg}")

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
                            debug_fmax: bool = False, ZERO_DRAM_ADDR: int = None, FMAX_DRAM_ADDR: int = None,
                            write_back_disable: bool = False, gpr_M_reg: int = None) -> None:
        """Matrix multiply entrypoint; dispatches based on ``gpr_M_reg``:

        - ``gpr_M_reg`` is a GPR index (1..15): :meth:`matmat_mul_core_pbi` — outer M-tile loop trip
          count is taken from that register at runtime (caller must prime it via ``ADD_SET``). The
          captured program has no static reference to ``M``; ``M`` is FLOPs-accounting only.
        - ``gpr_M_reg is None`` (default): :meth:`matmat_mul_core_legacy` — compile-time M tiling.

        **Layout:** ``A`` is **M×K** (row-major). ``B`` is **N×K** (row-major); the accelerator uses ``B`` as above and
        applies an implicit transpose so the computed result is **A @ Bᵀ**, i.e. **M×N**, without a separate transpose pass.
        """
        if gpr_M_reg is not None:
            return self.matmat_mul_core_pbi(
                M, K, N, A_DRAM_ADDR, B_DRAM_ADDR, OUTPUT_DRAM_ADDR, softmax_enable, C_DRAM_ADDR, bias_mode,
                is_B_quantized, data_type, SCALE_DRAM_ADDR, gelu_enable, silu_enable, sigmoid_enable,
                clamp_enable, log_enable,
                debug_fmax, ZERO_DRAM_ADDR, FMAX_DRAM_ADDR,
                write_back_disable=write_back_disable, gpr_M_reg=gpr_M_reg,
            )
        return self.matmat_mul_core_legacy(
            M, K, N, A_DRAM_ADDR, B_DRAM_ADDR, OUTPUT_DRAM_ADDR, softmax_enable, C_DRAM_ADDR, bias_mode,
            is_B_quantized, data_type, SCALE_DRAM_ADDR, gelu_enable, silu_enable, sigmoid_enable,
            clamp_enable, log_enable,
            debug_fmax, ZERO_DRAM_ADDR, FMAX_DRAM_ADDR,
            write_back_disable=write_back_disable,
        )

    def matmat_mul_core_legacy(self, M: int, K: int, N: int, A_DRAM_ADDR: int, B_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, softmax_enable: bool = False, C_DRAM_ADDR: int = None, bias_mode: str = "broadcast_N",
                             is_B_quantized: bool = False, data_type: TYPE = None, SCALE_DRAM_ADDR: int = None, gelu_enable: bool = False, silu_enable: bool = False, sigmoid_enable: bool = False,
                             clamp_enable: bool = False, log_enable: bool = False,
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
            lalu_a = LALU_CLAMP_RELU_A
            lalu_b = LALU_CLAMP_RELU_B
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

    def matmat_mul_core_pbi(self, M: int, K: int, N: int, A_DRAM_ADDR: int, B_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, softmax_enable: bool = False, C_DRAM_ADDR: int = None, bias_mode: str = "broadcast_N",
                             is_B_quantized: bool = False, data_type: TYPE = None, SCALE_DRAM_ADDR: int = None, gelu_enable: bool = False, silu_enable: bool = False, sigmoid_enable: bool = False,
                             clamp_enable: bool = False, log_enable: bool = False,
                             debug_fmax: bool = False, ZERO_DRAM_ADDR: int = None, FMAX_DRAM_ADDR: int = None,
                             write_back_disable: bool = False, gpr_M_reg: int = None) -> int:
        """
        Matmul capture path with a hardware while-loop over the M dimension.

        Args:
            M: Compile-time row count, used only for FLOPS reporting and as the fallback
                initial value of the hardware loop counter when ``gpr_M_reg`` is not supplied.
            gpr_M_reg: Optional ISA register index already holding the **runtime** row count.
                When provided, ``gpr_M_counter`` is seeded by copying this register
                (``add_imm dst=gpr_M_counter, src=gpr_M_reg, imm=0``).
                When ``None``, ``gpr_M_counter`` is seeded directly from compile-time ``M``
                (``add_set gpr_M_counter, M``).

        M-tiling:
        - ``M_chunk`` is chosen purely from URAM capacity (no compile-time M bound).
          Each tile takes ``m_take = min(gpr_M_counter, M_chunk)`` rows, so the last tile
          is a partial tile when M is not a multiple of M_chunk.
        - The outer loop terminates via ``ALU_MODE_SUB + JUMP_RELA_JNZ`` on ``gpr_M_counter``.

        N-tiling:
        - **Uniform** strips — largest ``N_chunk`` that divides ``N`` and fits URAM.
          Multiples of 64 (``UE_VECTOR_SIZE``) on the main path; 16/32 on the legacy small path.

        Softmax (when ``softmax_enable``): ``M_chunk`` is also capped by
        ``URAM_NEAR_FULL_ELEMENTS // N`` so one softmax slab fits the near-full URAM row budget.
        """
        # Requirements: Based on these conditions M_chunk x K + M_chunk x N_chunk should fit in URAM_A and N_chunk x K should fit in URAM_B
        # 1. M_chunk can be any value between 1 and M
        # 2. On the main N path ``N_chunk`` is a multiple of ``UE_VECTOR_SIZE``; on the small path, 16 or 32.
        # 3. M_chunk * K + N_chunk * M_chunk must be less than or equal to URAM_FULL_ELEMENTS

        bytes_per_element = 2
        bias_enable = C_DRAM_ADDR is not None

        if bias_enable:
            assert bias_mode in ("broadcast_N", "full_matrix"), f"bias_mode={bias_mode} must be either 'broadcast_N' or 'full_matrix'"

        if is_B_quantized:
            assert data_type in (TYPE.IF4, TYPE.IF8), f"data_type={data_type} must be one of TYPE.IF4, TYPE.IF8"
            assert K % UE_VECTOR_SIZE == 0, (
                f"quantized B path uses scale DRAM stride linear in j; require K % UE_VECTOR_SIZE == 0, got K={K}"
            )

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
            lalu_a = LALU_CLAMP_RELU_A
            lalu_b = LALU_CLAMP_RELU_B
        elif log_enable:
            lalu_mode = LALU_MODE.LOG
            lalu_a = LALU_LOG_A
            lalu_b = LALU_LOG_B

        # URAM cap on column strip width (same as legacy first-pass N_chunk).
        def _largest_uniform_strip(
            total_n: int, strip_cap: int, step: int, min_strip: int
        ) -> int | None:
            """Largest ``w`` with ``w | total_n``, ``min_strip <= w <= strip_cap``, ``w % step == 0``."""
            if strip_cap < min_strip or total_n < 1:
                return None
            upper = min(strip_cap, total_n)
            upper = (upper // step) * step
            if upper < min_strip:
                return None
            for cand in range(upper, min_strip - 1, -step):
                if total_n % cand == 0:
                    return cand
            return None

        n_cap_main = min(N, (URAM_NEAR_FULL_ELEMENTS // K) // UE_VECTOR_SIZE * UE_VECTOR_SIZE)
        N_chunk_aligned = None
        num_full_n_tiles = None
        N_chunk = None

        if n_cap_main >= UE_VECTOR_SIZE:
            # Multiples-of-64 strips only; maximizes URAM B width subject to cap and ``N_chunk | N``.
            N_chunk = _largest_uniform_strip(N, n_cap_main, UE_VECTOR_SIZE, UE_VECTOR_SIZE)
            if N_chunk is not None:
                num_full_n_tiles = N // N_chunk

        if num_full_n_tiles is None:
            # Legacy small-``N_chunk`` path: 32 or 16 from K, ``N_chunk_aligned`` for URAM-A matvec rows.
            if (K * 32) <= URAM_NEAR_FULL_ELEMENTS:
                n_cap_small = 32
            elif (K * 16) <= URAM_NEAR_FULL_ELEMENTS:
                n_cap_small = 16
            else:
                assert False, f"N={N} is too large to fit in URAM_NEAR_FULL_ELEMENTS={URAM_NEAR_FULL_ELEMENTS}"
            N_chunk_aligned = UE_VECTOR_SIZE
            N_chunk = _largest_uniform_strip(N, n_cap_small, 16, 16)
            assert N_chunk is not None, (
                f"matmat_mul_core_pbi: no uniform N strip (multiple of 16, ≤{n_cap_small}) divides N={N}. "
                "Adjust N or drop gpr_M_reg from the wrapper call to route to the legacy core."
            )
            num_full_n_tiles = N // N_chunk

        if N_chunk_aligned is None:
            if softmax_enable:
                M_chunk = min(UE_FMAX_CONTEXT_SIZE, URAM_FULL_ELEMENTS // (K + N_chunk))
            else:
                M_chunk = URAM_FULL_ELEMENTS // (K + N_chunk)
        else:
            if softmax_enable:
                M_chunk = min(UE_FMAX_CONTEXT_SIZE, URAM_FULL_ELEMENTS // (K + N_chunk_aligned))
            else:
                M_chunk = URAM_FULL_ELEMENTS // (K + N_chunk_aligned)

        if softmax_enable:
            softmax_max_rows_per_slab = URAM_NEAR_FULL_ELEMENTS // N
            assert softmax_max_rows_per_slab >= 1, (
                f"matmat_mul_core_pbi softmax: row length N={N} exceeds URAM_NEAR_FULL_ELEMENTS={URAM_NEAR_FULL_ELEMENTS}"
            )
            M_chunk = min(M_chunk, softmax_max_rows_per_slab)

        assert M_chunk >= 1, f"M_chunk={M_chunk} must be >= 1 (K={K}, N_chunk={N_chunk} too large for URAM?)"

        print(
            f"M_chunk: {M_chunk} (URAM cap, dynamic M from gpr_M_reg={gpr_M_reg}), "
            f"N_chunk: {N_chunk} (uniform ×{num_full_n_tiles}), "
            f"n_cap_main: {n_cap_main}, N_chunk_aligned: {N_chunk_aligned}",
        )

        if N_chunk_aligned is None:
            print(f"URAM_A usage: {100 * (M_chunk * K + M_chunk * N_chunk) / URAM_FULL_ELEMENTS:.2f}% of URAM_FULL_ELEMENTS")
            print(f"URAM_B usage: {100 * N_chunk * K / URAM_FULL_ELEMENTS:.2f}% of URAM_FULL_ELEMENTS")
        else:
            print(f"URAM_A usage: {100 * (M_chunk * K + M_chunk * N_chunk_aligned) / URAM_FULL_ELEMENTS:.2f}% of URAM_FULL_ELEMENTS")
            print(f"URAM_B usage: {100 * N_chunk * K / URAM_FULL_ELEMENTS:.2f}% of URAM_FULL_ELEMENTS")

        pointer_idx = [self.alloc_inst_ptr() for _ in range(13)]

        M_chunk_reg  = self.alloc_isa_reg()   # holds compile-time M_chunk constant
        m_take_reg   = self.alloc_isa_reg()   # min(gpr_M_counter, M_chunk) per tile
        gpr_M_counter = self.alloc_isa_reg()   # remaining M rows, decremented each tile
        dma_bytes_reg = self.alloc_isa_reg()  # scratch: m_take_reg * stride bytes, reused per callsite

        self.generate_instruction_add_set(M_chunk_reg, M_chunk)
        if gpr_M_reg is not None:
            self.generate_instruction_add_imm(src_reg_idx=gpr_M_reg, immediate_value=0, dst_reg_idx=gpr_M_counter)
        else:
            self.generate_instruction_add_set(gpr_M_counter, M)
        self.generate_instruction_reg_min(m_take_reg, gpr_M_counter, M_chunk_reg)

        ISA_FOR_LOOP = 1

        # --- one-time PBI inits (outside the tile loop) ---
        if bias_enable and bias_mode == "full_matrix":
            self.generate_instruction_pbi_init(
                dram_shared_addr=C_DRAM_ADDR,
                dma_length=N_chunk * bytes_per_element,
                inst_pointer_idx=pointer_idx[0],
            )
        if softmax_enable:
            softmax_dram_row0 = OUTPUT_DRAM_ADDR
            self.generate_instruction_pbi_init(
                dram_shared_addr=softmax_dram_row0,
                inst_pointer_idx=pointer_idx[11],
            )
            self.generate_instruction_pbi_init(
                dram_shared_addr=softmax_dram_row0,
                inst_pointer_idx=pointer_idx[12],
            )
        if not write_back_disable:
            if N_chunk_aligned is None:
                self.generate_instruction_pbi_init(
                    dram_shared_addr=OUTPUT_DRAM_ADDR,
                    output_size=N_chunk * bytes_per_element,
                    uram_a_start_addr=(M_chunk * K * bytes_per_element) >> 7,
                    inst_pointer_idx=pointer_idx[5],
                )
            else:
                self.generate_instruction_pbi_init(
                    dram_shared_addr=OUTPUT_DRAM_ADDR,
                    dma_length=N_chunk * 2,
                    uram_a_start_addr=M_chunk * K * bytes_per_element >> 7,
                    uram_b_start_addr=M_chunk * K * bytes_per_element >> 7,
                    inst_pointer_idx=pointer_idx[5],
                )
        self.generate_instruction_pbi_init(
            dram_shared_addr=A_DRAM_ADDR,
            inst_pointer_idx=pointer_idx[10],
        )
        if softmax_enable and debug_fmax:
            uram_start_addr = (M_chunk * N * bytes_per_element) >> 7
            self.generate_instruction_pbi_init(
                dram_shared_addr=FMAX_DRAM_ADDR,
                dma_length=UE_VECTOR_SIZE * 2,
                uram_a_start_addr=uram_start_addr,
                uram_b_start_addr=uram_start_addr,
                inst_pointer_idx=pointer_idx[7],
            )

        # abs jump to flush i-cache so the while-loop body lands at the start of a fresh line
        program_dram_start_addr = self.get_program_dram_addr()
        cur_inst_count = self.capture_count
        jump_target_word_addr = ue_35bit_addr_shifter(program_dram_start_addr + (cur_inst_count + 1) * INSTRUCTION_SIZE_BYTES)
        self.generate_instruction_jump_abs(jump_target_word_addr)

        # ===== while-loop body start =====
        body_start_inst_cnt = self.capture_count

        self.generate_instruction_reg_mul_imm(dma_bytes_reg, m_take_reg, K * 2)
        self.generate_instruction_pbi_inc(
            general_reg_src=dma_bytes_reg,
            pbi_field_select=PBI_FIELD.DMA_LENGTH,
            inst_pointer_idx=pointer_idx[10],
        )
        self.accelerator_memory_to_sram(accelerator_dram_address=M_chunk * K * bytes_per_element,
                                            sram_address=0,
                                            element_size=0,
                                            inst_pointer_idx=pointer_idx[10],
                                            )

        # clear max for each M tile
        self.generate_instruction_clear_fmax()

        ue_b = UE_VECTOR_SIZE * bytes_per_element
        k_row_b = K * bytes_per_element
        output_sram_wb_addr = M_chunk * K * bytes_per_element
        row_full_b = (N_chunk if N_chunk_aligned is None else N_chunk_aligned) * bytes_per_element
        pbi_inc_dot_product_uram_start_addr = k_row_b // ue_b
        pbi_inc_dot_product_uram_wb_addr_full = row_full_b // ue_b

        pbi_base_dot_product_uram_wb_addr = output_sram_wb_addr // ue_b
        pbi_inc_dot_product_uram_wb_addr = pbi_inc_dot_product_uram_wb_addr_full

        # PBI over N: ``num_full_n_tiles`` strips of uniform width ``N_chunk``.
        if is_B_quantized:
            if data_type == TYPE.IF4:
                offset = (K * N_chunk) >> 1
            elif data_type == TYPE.IF8:
                offset = K * N_chunk

            self.generate_instruction_pbi_init(
                dram_shared_addr=SCALE_DRAM_ADDR,
                dma_length=((N_chunk * K + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE) * bytes_per_element,
                inst_pointer_idx=pointer_idx[2],
            )
            self.generate_instruction_pbi_init(
                dram_shared_addr=B_DRAM_ADDR,
                dma_length=N_chunk * K if data_type == TYPE.IF8 else N_chunk * K >> 1,
                output_size=(N_chunk * K + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE,
                uram_length=(N_chunk * K + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE,
                uram_wb_addr=0x80000 >> 7,
                inst_pointer_idx=pointer_idx[3],
            )
        else:
            self.generate_instruction_pbi_init(
                dram_shared_addr=B_DRAM_ADDR,
                dma_length=N_chunk * K * bytes_per_element,
                uram_b_start_addr=(0x80000 >> 7) & 0xFFF,
                inst_pointer_idx=pointer_idx[2],
            )
        if bias_enable and bias_mode == "broadcast_N":
            self.generate_instruction_pbi_init(
                dram_shared_addr=C_DRAM_ADDR,
                dma_length=N_chunk * bytes_per_element,
                inst_pointer_idx=pointer_idx[4],
            )

        for i in range(ISA_FOR_LOOP):
            # N strip loop
            self.loop_start(num_full_n_tiles)
            if is_B_quantized:
                self.ue_memcpy_from_dram(
                    (K // UE_VECTOR_SIZE) * bytes_per_element * N_chunk,
                    0,
                    MEMCPY_TYPE.BRAM.value, 0, 0,
                    inst_pointer_idx=pointer_idx[2])

                self.ue_arithmetic_op(
                    broadcast_mode=0,
                    max_clear_en=0,
                    stride_z=1,
                    lalu_a=0,
                    lalu_b=0,
                    lalu_mode=LALU_MODE.BYPASS.value,
                    scalar=0,
                    uram_section=URAM_SECTION.URAM_B.value,
                    uram_dst_addr=0,
                    uram_wb_addr=0,
                    uram_write_src=URAM_WRITE_SRC.URAM_WRITE_BACK.value,
                    mode=UE_MODE.DEQUANTIZE,
                    data_type=data_type.value,
                    uram_a_start_addr=0,
                    uram_b_start_addr=0,
                    uram_length=0,
                    dma_start_addr=offset,
                    dma_length=0,
                    output_size=0,
                    inst_pointer_idx=pointer_idx[3],
                )
            else:
                self.ue_memcpy_from_dram(
                    K * bytes_per_element * N_chunk,
                    0,
                    MEMCPY_TYPE.URAM.value,
                    0,
                    URAM_SECTION.URAM_B,
                    inst_pointer_idx=pointer_idx[2],
                )

            if bias_enable and bias_mode == "broadcast_N":
                self.ue_memcpy_from_dram(
                    bytes_per_element * N_chunk,
                    0,
                    MEMCPY_TYPE.BIAS_BRAM.value,
                    0,
                    0,
                    inst_pointer_idx=pointer_idx[4],
                )

            self.generate_instruction_pbi_init(
                dram_shared_addr=0,
                dma_length=K * N_chunk,
                output_size=N_chunk,
                uram_length=K // UE_VECTOR_SIZE,
                uram_a_start_addr=0,
                uram_b_start_addr=0,
                uram_wb_addr=pbi_base_dot_product_uram_wb_addr,
                uram_dst_addr=0,
                fmax_context_addr=0,
                inst_pointer_idx=pointer_idx[1],
            )

            for j in range(ISA_FOR_LOOP):
                self.loop_start(gpr_loop_cnt=m_take_reg)
                if bias_enable and bias_mode == "full_matrix":
                    self.accelerator_memory_to_bias_sram(
                        accelerator_dram_address=N * bytes_per_element,
                        element_size=0,
                        inst_pointer_idx=pointer_idx[0],
                    )

                self.ue_arithmetic_op(
                    broadcast_mode=0,
                    max_clear_en=0,
                    stride_z=1,
                    lalu_a=lalu_a,
                    lalu_b=lalu_b,
                    lalu_mode=lalu_mode.value,
                    scalar=0,
                    uram_section=URAM_SECTION.URAM_A.value,
                    uram_dst_addr=0,
                    uram_wb_addr=pbi_inc_dot_product_uram_wb_addr,
                    uram_write_src=URAM_WRITE_SRC.URAM_WRITE_BACK.value,
                    mode=UE_MODE.BF16_DOT_PRODUCT,
                    data_type=0,
                    uram_a_start_addr=pbi_inc_dot_product_uram_start_addr,
                    uram_b_start_addr=0,
                    uram_length=0,
                    dma_start_addr=0,
                    dma_length=0,
                    output_size=0,
                    bias_adder_en=bias_enable,
                    fmax_context_addr=1,
                    inst_pointer_idx=pointer_idx[1],
                )
                self.loop_end()

            if bias_enable and bias_mode == "full_matrix":
                # rewind that row stride m_take_reg times (ISA loop), then step by N_chunk for the next N strip.
                self.loop_start(gpr_loop_cnt=m_take_reg)
                self.generate_instruction_pbi_inc(
                    dram_shared_addr=-N * bytes_per_element,
                    inst_pointer_idx=pointer_idx[0],
                )
                self.loop_end()
                self.generate_instruction_pbi_inc(
                    dram_shared_addr=N_chunk * bytes_per_element,
                    inst_pointer_idx=pointer_idx[0],
                )

            if not write_back_disable:
                if N_chunk_aligned is None:
                    self.generate_instruction_reg_mul_imm(dma_bytes_reg, m_take_reg, N_chunk * 2)
                    self.generate_instruction_pbi_inc(
                        general_reg_src=dma_bytes_reg,
                        pbi_field_select=PBI_FIELD.DMA_LENGTH,
                        inst_pointer_idx=pointer_idx[5],
                    )
                    self.ue_memcpy_to_dram(
                        memcpy_type=MEMCPY_TYPE.URAM.value,
                        uram_type=URAM_SECTION.URAM_A.value,
                        uram_src_addr=0,
                        dram_dst_addr=bytes_per_element * N_chunk,
                        memcpy_length_bytes=0,
                        stride_bytes_per_chunk=0,
                        stride_jump_bytes=N * bytes_per_element,
                        inst_pointer_idx=pointer_idx[5],
                    )
                else:
                    self.loop_start(gpr_loop_cnt=m_take_reg)
                    self.ue_memcpy_to_dram(
                        memcpy_type=MEMCPY_TYPE.URAM.value,
                        uram_type=URAM_SECTION.URAM_A.value,
                        uram_src_addr=1,
                        dram_dst_addr=N * bytes_per_element,
                        memcpy_length_bytes=0,
                        inst_pointer_idx=pointer_idx[5],
                    )
                    self.loop_end()

                    # rewind m_take_reg row strides, then step by N_chunk for the next N strip.
                    self.loop_start(gpr_loop_cnt=m_take_reg)
                    self.generate_instruction_pbi_inc(
                        dram_shared_addr=-N * bytes_per_element,
                        uram_a_start_addr=-1,
                        uram_b_start_addr=-1,
                        inst_pointer_idx=pointer_idx[5],
                    )
                    self.loop_end()
                    self.generate_instruction_pbi_inc(
                        dram_shared_addr=N_chunk * bytes_per_element,
                        inst_pointer_idx=pointer_idx[5],
                    )
            self.loop_end()

        # Advance output PBI pointer by (m_take_reg - 1)*N*2 to the start of the next M strip.
        # Loop m_take_reg times (+N*2 each), then subtract one step (-N*2) for net (m_take_reg-1)*N*2.
        if not write_back_disable:
            self.loop_start(gpr_loop_cnt=m_take_reg)
            self.generate_instruction_pbi_inc(
                dram_shared_addr=N * bytes_per_element,
                inst_pointer_idx=pointer_idx[5],
            )
            self.loop_end()
            self.generate_instruction_pbi_inc(
                dram_shared_addr=-N * bytes_per_element,
                inst_pointer_idx=pointer_idx[5],
            )

        # Advance bias PBI pointer by (m_take_reg - 1)*N*2 to the start of the next M strip.
        if bias_enable and bias_mode == "full_matrix":
            self.loop_start(gpr_loop_cnt=m_take_reg)
            self.generate_instruction_pbi_inc(
                dram_shared_addr=N * bytes_per_element,
                inst_pointer_idx=pointer_idx[0],
            )
            self.loop_end()
            self.generate_instruction_pbi_inc(
                dram_shared_addr=-N * bytes_per_element,
                inst_pointer_idx=pointer_idx[0],
            )

        # Optional softmax operation
        if softmax_enable:
            assert N % UE_VECTOR_SIZE == 0, "N must be a multiple of UE_VECTOR_SIZE"
            # input_sram_start_addr = 0x00000, softmax_out_sram_wb_addr = 0x80000
            self.generate_instruction_reg_mul_imm(dma_bytes_reg, m_take_reg, N * 2)
            self.generate_instruction_pbi_inc(
                general_reg_src=dma_bytes_reg,
                pbi_field_select=PBI_FIELD.DMA_LENGTH,
                inst_pointer_idx=pointer_idx[11],
            )
            self.ue_memcpy_from_dram(
                dram_src_addr=M_chunk * N * bytes_per_element,
                memcpy_length_bytes=0,
                memcpy_type=MEMCPY_TYPE.URAM.value,
                uram_dst_addr=0,
                uram_type=URAM_SECTION.URAM_A.value,
                inst_pointer_idx=pointer_idx[11],
            )

            uram_wb_row_stride = N * bytes_per_element >> 7
            row_size = N // UE_VECTOR_SIZE
            if debug_fmax:
                uram_sfmx_trace_addr = (M_chunk * N * bytes_per_element) >> 7
                sfmx_dbg_nbytes = UE_VECTOR_SIZE * 2
                row_size_fmax = (UE_VECTOR_SIZE + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
                self.generate_instruction_pbi_init(
                    uram_length=row_size_fmax,
                    uram_a_start_addr=uram_sfmx_trace_addr,
                    uram_wb_addr=uram_sfmx_trace_addr,
                    inst_pointer_idx=pointer_idx[6],
                )
            self.generate_instruction_pbi_init(
                uram_length=row_size,
                inst_pointer_idx=pointer_idx[8],
            )
            self.generate_instruction_pbi_init(
                uram_length=row_size,
                uram_wb_addr=(0x80000 >> 7),
                inst_pointer_idx=pointer_idx[9],
            )

            for j in range(ISA_FOR_LOOP):
                self.loop_start(gpr_loop_cnt=m_take_reg)
                if debug_fmax:
                    self.ue_memcpy_from_dram(
                        ZERO_DRAM_ADDR,
                        sfmx_dbg_nbytes,
                        memcpy_type=MEMCPY_TYPE.URAM.value,
                        uram_dst_addr=uram_sfmx_trace_addr,
                        uram_type=URAM_SECTION.URAM_A.value,
                        stride_bytes_per_chunk=0,
                        stride_jump_bytes=0,
                    )

                    self.ue_arithmetic_op(
                        broadcast_mode=BROADCAST_MODE.FMAX_NEGATE.value,
                        max_clear_en=0,
                        stride_z=1,
                        lalu_a=0,
                        lalu_b=0,
                        lalu_mode=LALU_MODE.BYPASS.value,
                        scalar=0,
                        uram_section=URAM_SECTION.URAM_A.value,
                        uram_dst_addr=0,
                        uram_wb_addr=0,
                        uram_write_src=URAM_WRITE_SRC.URAM_WRITE_BACK.value,
                        mode=UE_MODE.ADD_BROADCAST,
                        data_type=0,
                        uram_a_start_addr=0,
                        uram_b_start_addr=0,
                        uram_length=0,
                        dma_start_addr=0,
                        dma_length=0,
                        output_size=0,
                        fmax_context_addr=1,
                        inst_pointer_idx=pointer_idx[6],
                    )

                    self.ue_memcpy_to_dram(
                        memcpy_type=MEMCPY_TYPE.URAM.value,
                        uram_type=URAM_SECTION.URAM_A.value,
                        uram_src_addr=0,
                        dram_dst_addr=UE_VECTOR_SIZE * bytes_per_element,
                        memcpy_length_bytes=0,
                        stride_bytes_per_chunk=0,
                        stride_jump_bytes=0,
                        inst_pointer_idx=pointer_idx[7],
                    )

                self.ue_arithmetic_op(
                    broadcast_mode=BROADCAST_MODE.FMAX_NEGATE.value,
                    max_clear_en=0,
                    stride_z=1,
                    lalu_a=0,
                    lalu_b=0,
                    lalu_mode=LALU_MODE.MODE_RECIP.value,
                    scalar=self.float_to_bf19(1.0),
                    uram_section=URAM_SECTION.URAM_A,
                    uram_dst_addr=0,
                    uram_wb_addr=uram_wb_row_stride,
                    uram_write_src=URAM_WRITE_SRC.URAM_WRITE_BACK.value,
                    mode=UE_MODE.EXP,
                    data_type=0,
                    uram_a_start_addr=uram_wb_row_stride,
                    uram_b_start_addr=0,
                    uram_length=0,
                    dma_start_addr=0,
                    dma_length=0,
                    output_size=0,
                    fmax_context_addr=1,
                    inst_pointer_idx=pointer_idx[8],
                )

                self.ue_arithmetic_op(
                    broadcast_mode=BROADCAST_MODE.LALU_RESULT.value,
                    max_clear_en=0,
                    stride_z=1,
                    lalu_a=0,
                    lalu_b=0,
                    lalu_mode=LALU_MODE.BYPASS.value,
                    scalar=self.float_to_bf19(1.0),
                    uram_section=URAM_SECTION.URAM_B,
                    uram_dst_addr=0,
                    uram_wb_addr=uram_wb_row_stride,
                    uram_write_src=URAM_WRITE_SRC.URAM_WRITE_BACK.value,
                    mode=UE_MODE.MUL_BROADCAST,
                    data_type=0,
                    uram_a_start_addr=uram_wb_row_stride,
                    uram_b_start_addr=0,
                    uram_length=0,
                    dma_start_addr=0,
                    dma_length=0,
                    output_size=0,
                    fmax_context_addr=0,
                    inst_pointer_idx=pointer_idx[9],
                )
                self.loop_end()
            self.generate_instruction_pbi_inc(
                general_reg_src=dma_bytes_reg,
                pbi_field_select=PBI_FIELD.DMA_LENGTH,
                inst_pointer_idx=pointer_idx[12],
            )
            self.ue_memcpy_to_dram(
                memcpy_type=MEMCPY_TYPE.URAM.value,
                uram_type=URAM_SECTION.URAM_B.value,
                uram_src_addr=0,
                dram_dst_addr=M_chunk * N * bytes_per_element,
                memcpy_length_bytes=0,
                inst_pointer_idx=pointer_idx[12],
            )
        # ===== while-loop end =====
        # Decrement remaining-row counter by actual m_take; jump back if rows remain.
        self.generate_instruction_reg_sub(gpr_M_counter, gpr_M_counter, m_take_reg)
        self.generate_instruction_reg_min(m_take_reg, gpr_M_counter, M_chunk_reg)
        outer_loop_size = self.capture_count - body_start_inst_cnt + 2
        self.generate_instruction_jump_rela_jnz(outer_loop_size, gpr_M_counter)

        self.release_isa_reg()  # dma_bytes_reg
        self.release_isa_reg()  # gpr_M_counter
        self.release_isa_reg()  # m_take_reg
        self.release_isa_reg()  # M_chunk_reg

        print(f"While-loop body size: {outer_loop_size}")
        assert outer_loop_size <= 512, (
            f"Outer while-loop body size {outer_loop_size} exceeds i-cache limit of 512 instructions"
        )

        for ptr in reversed(pointer_idx):
            self.release_inst_ptr(ptr)

        total_flops = 2 * M * K * N
        if softmax_enable:
            total_flops += M * N * 5
        if gelu_enable or silu_enable or sigmoid_enable:
            total_flops += M * N
        if clamp_enable:
            total_flops += M * N  # 1 compare + clamp per element
        if log_enable:
            total_flops += 2 * M * N  # clamp + log per element
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
                             use_pbi: bool = False) -> int:
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

        # PBI path is driven by gpr_M_reg; allocate + prime a GPR with M on each engine when use_pbi.
        m_reg0 = ue0.alloc_isa_reg() if use_pbi else None
        m_reg1 = ue1.alloc_isa_reg() if use_pbi else None

        # Program engine0
        ue0.start_capture()
        ue0.generate_instruction_flag_clear()
        if use_pbi:
            ue0.generate_instruction_add_set(m_reg0, m_engine0)
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
        )
        ue0.generate_instruction_flag_set()
        ue0.generate_instruction_halt()
        ue0.stop_capture()
        if use_pbi:
            ue0.release_isa_reg()

        engine0_program_dram_addr = ue0.get_program_dram_addr()
        ue0.write_captured_instructions_to_dram(engine0_program_dram_addr)
        ue0.allocate_program_dram(ue0.get_capture_instruction_size_bytes())

        # Program ue1
        ue1.start_capture()
        if use_pbi:
            ue1.generate_instruction_add_set(m_reg1, m_engine1)
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
        )
        ue1.generate_instruction_flag_check(target_engine_idx=0)
        ue1.generate_instruction_halt()
        ue1.stop_capture()
        if use_pbi:
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

    def bf16_transpose_core(self, M: int, N: int, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, use_pbi: bool = False, IDENTITY_DRAM_ADDR: int = None) -> None:
        """Transpose ``M×N`` → ``N×M``. ``use_pbi``: :meth:`bf16_transpose_core_pbi` vs :meth:`bf16_transpose_core_legacy`."""
        if use_pbi:
            return self.bf16_transpose_core_pbi(M, N, INPUT_DRAM_ADDR, OUTPUT_DRAM_ADDR, IDENTITY_DRAM_ADDR=IDENTITY_DRAM_ADDR)
        return self.bf16_transpose_core_legacy(M, N, INPUT_DRAM_ADDR, OUTPUT_DRAM_ADDR, IDENTITY_DRAM_ADDR=IDENTITY_DRAM_ADDR)

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
    
    def bf16_transpose_core_pbi(self, M: int, N: int, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, IDENTITY_DRAM_ADDR: int = None) -> None:
        """
        Transposes a (M x N) input matrix X to produce an (N x M) output matrix Y = X^T.
        Hybrid mode: Python outer loop for N (legacy style for bandwidth efficiency), 
        Hardware mapped loop for M (PBI optimized).
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
        
        # transfer identity matrix to URAM_A start
        self.accelerator_memory_to_sram(accelerator_dram_address=identity_matrix_dram_addr,
                                        sram_address=identity_matrix_sram_start_addr,
                                        element_size=UE_VECTOR_SIZE * UE_VECTOR_SIZE)

        usable_uram_a_start_addr = identity_matrix_sram_start_addr + UE_VECTOR_SIZE * UE_VECTOR_SIZE * bytes_per_element
        
        # 1. Calculate theoretical max M_chunk
        M_chunk = min(M, (URAM_NEAR_FULL_ELEMENTS // N) // UE_VECTOR_SIZE * UE_VECTOR_SIZE)
        
        # 2. Force M_chunk to be a perfect divisor of M so the static hardware loop drops no remainders
        while M_chunk > 0 and M % M_chunk != 0:
            M_chunk -= UE_VECTOR_SIZE
            
        if M_chunk == 0:
            raise ValueError(f"Cannot find an M_chunk that evenly divides M={M} with UE_VECTOR_SIZE={UE_VECTOR_SIZE}")

        M_chunk_aligned = None
        if M_chunk < UE_VECTOR_SIZE:
            if (N * 32) <= URAM_NEAR_FULL_ELEMENTS:
                M_chunk = 32
            elif (N * 16) <= URAM_NEAR_FULL_ELEMENTS:
                M_chunk = 16
            else:
                assert False, f"N={N} is too large to fit in usable URAM elements={URAM_NEAR_FULL_ELEMENTS}"
            M_chunk_aligned = UE_VECTOR_SIZE

        num_full_m_tiles = M // M_chunk

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

        # Local Pointer Row Allocation (Recommended Lifetime Rule)
        pointer_input = self.alloc_inst_ptr()
        pointer_compute = self.alloc_inst_ptr()
        pointer_output = self.alloc_inst_ptr()

        stride_in_rows = N // UE_VECTOR_SIZE
        init_dma_length = UE_VECTOR_SIZE * M_chunk if stride_in_rows == 1 else M_chunk * N
        init_vector_sram_addr = 0x00000

        row_stride_elements = M_chunk if M_chunk_aligned is None else M_chunk_aligned
        wb_line_delta = row_stride_elements // UE_VECTOR_SIZE

        # ===== Legacy Style Python Outer N Loop =====
        for n_offset, n_take in self.chunk_ranges(N, N_chunk):
            
            OUTPUT_DRAM_ADDR_LOCAL = OUTPUT_DRAM_ADDR + n_offset * M * bytes_per_element
            
            # Init Input Pointer
            self.generate_instruction_pbi_init(
                dram_shared_addr=INPUT_DRAM_ADDR,
                dma_length=M_chunk * N * bytes_per_element,
                uram_b_start_addr=(uram_b_start_addr >> 7) & 0xFFF,
                inst_pointer_idx=pointer_input,
            )

            # Init Output Pointer
            if M_chunk_aligned is None:
                self.generate_instruction_pbi_init(
                    dram_shared_addr=OUTPUT_DRAM_ADDR_LOCAL,
                    dma_length=n_take * M_chunk * bytes_per_element, # Total DMA bytes explicitly set here
                    output_size=M_chunk * bytes_per_element,         # Chunk size (prevents 16-bit overflow)
                    uram_a_start_addr=output_sram_wb_addr >> 7,
                    inst_pointer_idx=pointer_output,
                )
            else:
                self.generate_instruction_pbi_init(
                    dram_shared_addr=OUTPUT_DRAM_ADDR_LOCAL,
                    dma_length=M_chunk * 2,
                    uram_a_start_addr=output_sram_wb_addr >> 7,
                    uram_b_start_addr=output_sram_wb_addr >> 7,
                    inst_pointer_idx=pointer_output,
                )

            # Init Compute Pointer (Offsetting uram_b so we pull from the correct columns)
            compute_uram_b_offset = (n_offset // UE_VECTOR_SIZE)
            self.generate_instruction_pbi_init(
                dram_shared_addr=0,
                dma_length=init_dma_length,
                output_size=M_chunk,
                uram_length=1, 
                uram_a_start_addr=init_vector_sram_addr >> 7,
                uram_b_start_addr=(uram_b_start_addr >> 7) + compute_uram_b_offset,
                uram_wb_addr=output_sram_wb_addr >> 7,
                inst_pointer_idx=pointer_compute
            )

            # ===== Inner Hardware M Loop =====
            self.loop_start(num_full_m_tiles)

            # 1. Fetch exactly one horizontal M slice (M_chunk * N)
            self.ue_memcpy_from_dram(
                N * bytes_per_element * M_chunk,
                0,
                MEMCPY_TYPE.URAM.value,
                0,
                URAM_SECTION.URAM_B,
                inst_pointer_idx=pointer_input,
            )

            # 2. Compute `n_take` rows in increments of UE_VECTOR_SIZE
            for inner_n_offset, inner_n_take in self.chunk_ranges(n_take, UE_VECTOR_SIZE):
                
                self.loop_start(inner_n_take)
                
                self.ue_arithmetic_op(
                    broadcast_mode=0,
                    max_clear_en=0,
                    stride_z=stride_in_rows, 
                    lalu_a=0,
                    lalu_b=0,
                    lalu_mode=LALU_MODE.BYPASS.value,
                    scalar=0,
                    uram_section=URAM_SECTION.URAM_A.value,
                    uram_dst_addr=0,
                    uram_wb_addr=wb_line_delta,             
                    uram_write_src=URAM_WRITE_SRC.URAM_WRITE_BACK.value,
                    mode=UE_MODE.BF16_DOT_PRODUCT,
                    data_type=0,
                    uram_a_start_addr=1,                    # +1 line per iteration
                    uram_b_start_addr=0,                    # 0 (matrix pointer remains stationary for these rows)
                    uram_length=0,                          
                    dma_start_addr=0,
                    dma_length=0,                           
                    output_size=0,                          
                    bias_adder_en=False,
                    fmax_context_addr=0,
                    inst_pointer_idx=pointer_compute,
                )
                self.loop_end() 

                # Reset uram_a (identity matrix) and advance uram_b (matrix window) for next UE_VECTOR chunk
                self.generate_instruction_pbi_inc(
                    uram_a_start_addr=-inner_n_take,
                    uram_b_start_addr=1,
                    inst_pointer_idx=pointer_compute
                )

            # 3. Write back outputs to DRAM
            if M_chunk_aligned is None:
                self.ue_memcpy_to_dram(
                    memcpy_type=MEMCPY_TYPE.URAM.value,
                    uram_type=URAM_SECTION.URAM_A.value,
                    uram_src_addr=0,
                    dram_dst_addr=bytes_per_element * M_chunk,
                    memcpy_length_bytes=0,    # Falls back to PBI pointer's dma_length 
                    stride_bytes_per_chunk=0, # Falls back to PBI pointer's output_size 
                    stride_jump_bytes=M * bytes_per_element,
                    inst_pointer_idx=pointer_output,
                )
            else:
                self.loop_start(n_take)
                self.ue_memcpy_to_dram(
                    memcpy_type=MEMCPY_TYPE.URAM.value,
                    uram_type=URAM_SECTION.URAM_A.value,
                    uram_src_addr=1,
                    dram_dst_addr=M * bytes_per_element,
                    memcpy_length_bytes=0,
                    inst_pointer_idx=pointer_output,
                )
                self.loop_end()

                # Rewind and step
                self.loop_start(n_take)
                self.generate_instruction_pbi_inc(
                    dram_shared_addr=-M * bytes_per_element,
                    uram_a_start_addr=-1,
                    uram_b_start_addr=-1,
                    inst_pointer_idx=pointer_output,
                )
                self.loop_end()
                self.generate_instruction_pbi_inc(
                    dram_shared_addr=M_chunk * bytes_per_element,
                    inst_pointer_idx=pointer_output,
                )

            # 4. Rewind Compute Pointers (uram_b and uram_wb) for the next M tile
            for inner_n_offset, inner_n_take in self.chunk_ranges(n_take, UE_VECTOR_SIZE):
                self.generate_instruction_pbi_inc(
                    uram_b_start_addr=-1,
                    uram_wb_addr=-inner_n_take * wb_line_delta,
                    inst_pointer_idx=pointer_compute
                )

            self.loop_end()
            # ===== End Hardware M Loop =====

        # Release Local Pointer Tracking
        self.release_inst_ptr(pointer_input)
        self.release_inst_ptr(pointer_compute)
        self.release_inst_ptr(pointer_output)



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

    def flash_attention_core(self, head_dim: int, seq_len: int, Q_DRAM_ADDR: int, K_DRAM_ADDR: int, V_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, SCRATCH_DRAM_ADDR: int, IDENTITY_DRAM_ADDR: int = None, BIAS_DRAM_ADDR: int = None,
                            debug_mode: bool = False, SM_OUTPUT_DRAM_ADDR: int = None, ATTN_P_DRAM_ADDR: int = None,
                            gpr_bucket_idx: int = None, num_buckets: int = 8, use_pbi: bool = True):
        """Flash attention entrypoint; dispatches based on ``gpr_bucket_idx``:

        - ``gpr_bucket_idx`` is a GPR index (1..15): :meth:`flash_attention_core_pbi` — captured
          program is the ``num_buckets``-way dispatcher; caller must prime that register **with the
          1-based selector** ``bucket_idx = aligned_seq_len // UE_VECTOR_SIZE`` (via ``ADD_SET``)
          *before* program execution starts. So ``aligned_seq_len = 64 → 1``, ``128 → 2``,
          ``UE_VECTOR_SIZE*N → N``. Out-of-range values silently fall through into bucket_1
          (seq_len=UE_VECTOR_SIZE), which produces wrong activations rather than crashing — so
          double-check the selector matches the aligned (padded-to-UE_VECTOR_SIZE) seq_len, not the
          raw one. Requires ``ATTN_P_DRAM_ADDR``. ``seq_len`` is ignored here since the bucketized
          program covers all bucket sizes; the register is preserved across calls so caller can
          prime once and invoke many times.
        - ``gpr_bucket_idx is None`` (default): :meth:`flash_attention_core_legacy` — single static
          ``seq_len`` body, no dispatcher.

        Returns ``int`` total FLOPS for the legacy path, or ``list[int]`` per-bucket FLOPS for the
        PBI path (caller selects with ``bucket_flops[gpr_bucket_idx - 1]`` — Python list is 0-based
        even though the runtime selector is 1-based; see :meth:`flash_attention_core_pbi`).
        """
        if gpr_bucket_idx is not None:
            if ATTN_P_DRAM_ADDR is None:
                raise ValueError(
                    "flash_attention_core: gpr_bucket_idx-driven PBI path requires ATTN_P_DRAM_ADDR "
                    f"(allocate seq_len*seq_len BF16s, i.e. {seq_len * seq_len * 2} bytes)"
                )
            return self.flash_attention_core_pbi(
                head_dim=head_dim,
                Q_DRAM_ADDR=Q_DRAM_ADDR,
                K_DRAM_ADDR=K_DRAM_ADDR,
                V_DRAM_ADDR=V_DRAM_ADDR,
                OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                SCRATCH_DRAM_ADDR=SCRATCH_DRAM_ADDR,
                ATTN_P_DRAM_ADDR=ATTN_P_DRAM_ADDR,
                BIAS_DRAM_ADDR=BIAS_DRAM_ADDR,
                debug_mode=debug_mode,
                SM_OUTPUT_DRAM_ADDR=SM_OUTPUT_DRAM_ADDR,
                IDENTITY_DRAM_ADDR=IDENTITY_DRAM_ADDR,
                gpr_bucket_idx=gpr_bucket_idx,
                num_buckets=num_buckets,
                use_pbi=use_pbi,
            )

        return self.flash_attention_core_legacy(
            head_dim=head_dim,
            seq_len=seq_len,
            Q_DRAM_ADDR=Q_DRAM_ADDR,
            K_DRAM_ADDR=K_DRAM_ADDR,
            V_DRAM_ADDR=V_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
            SCRATCH_DRAM_ADDR=SCRATCH_DRAM_ADDR,
            BIAS_DRAM_ADDR=BIAS_DRAM_ADDR,
            debug_mode=debug_mode,
            SM_OUTPUT_DRAM_ADDR=SM_OUTPUT_DRAM_ADDR,
            IDENTITY_DRAM_ADDR=IDENTITY_DRAM_ADDR,
        )

    def flash_attention_core_legacy(self, head_dim: int, seq_len: int, Q_DRAM_ADDR: int, K_DRAM_ADDR: int, V_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, SCRATCH_DRAM_ADDR: int, IDENTITY_DRAM_ADDR: int, BIAS_DRAM_ADDR: int = None,
                            debug_mode: bool = False, SM_OUTPUT_DRAM_ADDR: int = None) -> None:

        bytes_per_element = 2
        bias_enable = BIAS_DRAM_ADDR is not None

        if debug_mode: # DEBUG only, needs to be allocated in DRAM
            assert SM_OUTPUT_DRAM_ADDR is not None, "SM_OUTPUT_DRAM_ADDR is not set for debug mode"

        # SCRATCH_DRAM_ADDR is used for V^T
        SCRATCH_DRAM_PARTIAL_SM = SCRATCH_DRAM_ADDR + head_dim * seq_len * bytes_per_element # used for partial softmax output

        # ----------------------------------------------------------------------------------------------------------------
        # I @ V^T: (head_dim, head_dim) @ (seq_len, head_dim)^T -> (head_dim, seq_len)
        # Convention: first matrix I is (M, K), second V^T is (K, N), output  (M, N)
        M = head_dim   # identity length (rows of I)
        K = head_dim  # identity dimension (inner product dim)
        N = seq_len   # V length (columns of V^T)

        # Load UE_VECTOR_SIZE x UE_VECTOR_SIZE identity block into URAM_A start
        identity_matrix_sram_start_addr = 0x00000

        # transfer identity matrix to URAM_A start
        self.accelerator_memory_to_sram(accelerator_dram_address=IDENTITY_DRAM_ADDR,
                                        sram_address=identity_matrix_sram_start_addr,
                                        element_size=UE_VECTOR_SIZE * UE_VECTOR_SIZE)

        usable_uram_a_start_addr = identity_matrix_sram_start_addr + UE_VECTOR_SIZE * UE_VECTOR_SIZE * bytes_per_element

        # URAM_B is used for V matrix, we need to chunk the V matrix into smaller chunks that can fit in URAM_B
        usable_uram_b_elements = URAM_NEAR_FULL_ELEMENTS
        N_chunk = min(N, (usable_uram_b_elements // K) // UE_VECTOR_SIZE * UE_VECTOR_SIZE)
        N_chunk_aligned = None
        if N_chunk < UE_VECTOR_SIZE:
            if (K * 32) <= usable_uram_b_elements:
                N_chunk = 32
            elif (K * 16) <= usable_uram_b_elements:
                N_chunk = 16
            else:
                assert False, f"K={K} is too large to fit in usable URAM elements={usable_uram_b_elements}"
            N_chunk_aligned = UE_VECTOR_SIZE

        usable_uram_a_elements = URAM_FULL_ELEMENTS - UE_VECTOR_SIZE * UE_VECTOR_SIZE
        output_N_size = N_chunk_aligned if N_chunk_aligned is not None else N_chunk
        M_chunk = min(M, usable_uram_a_elements // output_N_size)
        assert M_chunk >= 1 and M_chunk <= M, f"M_chunk={M_chunk} must be greater than 0 and less than M={M}"

        print(f"M_chunk: {M_chunk}, N_chunk: {N_chunk}", f"N_chunk_aligned: {N_chunk_aligned}")
        print(f"URAM_A usage: {100 * (UE_VECTOR_SIZE * UE_VECTOR_SIZE + M_chunk * output_N_size) / URAM_FULL_ELEMENTS:.2f}% of URAM_NEAR_FULL_ELEMENTS")
        print(f"URAM_B usage: {100 * N_chunk * K / URAM_FULL_ELEMENTS:.2f}% of URAM_FULL_ELEMENTS")

        output_sram_wb_addr = usable_uram_a_start_addr
        uram_b_start_addr = 0x80000
        for i, m_take in self.chunk_ranges(M, M_chunk):
            for j, n_take in self.chunk_ranges(N, N_chunk):

                self.accelerator_memory_to_sram(accelerator_dram_address=V_DRAM_ADDR + j * K * bytes_per_element,
                                            sram_address=uram_b_start_addr,
                                            element_size=n_take * K)

                for output_row in range(m_take):
                    if N_chunk_aligned is None:
                        out_sram_offset = output_row * n_take * bytes_per_element
                    else:
                        out_sram_offset = output_row * N_chunk_aligned * bytes_per_element

                    row = output_row + i
                    ones_idx = row // UE_VECTOR_SIZE
                    vector_idx = row % UE_VECTOR_SIZE

                    self.start_queue_for_bf16_matvec_operation(max_clear_en=0,
                                                            fmax_context_addr=0,
                                                            vector_sram_start_addr=0x00000 + vector_idx * UE_VECTOR_SIZE * bytes_per_element,
                                                            matrix_sram_start_addr=uram_b_start_addr + ones_idx * UE_VECTOR_SIZE * bytes_per_element,
                                                            output_sram_wb_addr=output_sram_wb_addr + out_sram_offset,
                                                            K=UE_VECTOR_SIZE,
                                                            N=n_take,
                                                            stride_z=m_take)

                start_dram_address_of_partial_matrix = SCRATCH_DRAM_ADDR + i * N * bytes_per_element + j * bytes_per_element # the space needed is head_dim x seq_len

                if N_chunk_aligned is None:
                    self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr,
                                                    accelerator_dram_address=start_dram_address_of_partial_matrix,
                                                    element_size=m_take * n_take,
                                                    stride_bytes_per_chunk=n_take * bytes_per_element,
                                                    stride_jump_bytes=N * bytes_per_element)
                else:
                    for o_row_idx in range(m_take):
                        self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr + o_row_idx * N_chunk_aligned * bytes_per_element,
                                                        accelerator_dram_address=start_dram_address_of_partial_matrix + o_row_idx * N * bytes_per_element,
                                                        element_size=n_take)

        # ----------------------------------------------------------------------------------------------------------------
        # Q @ K^T: (seq_len, head_dim) @ (head_dim, seq_len) -> (seq_len, seq_len)
        # Convention: first matrix Q is (M, K), second K^T is (K, N), output scores (M, N)
        M = seq_len   # query length (rows of Q)
        K = head_dim  # head dimension (inner product dim)
        N = seq_len   # key length (columns of K^T)
        # Calculate N_chunk
        usable_uram_b_elements = URAM_NEAR_FULL_ELEMENTS
        N_chunk = min(N, (usable_uram_b_elements // K) // UE_VECTOR_SIZE * UE_VECTOR_SIZE)
        N_chunk_aligned = None
        if N_chunk < UE_VECTOR_SIZE:
            if (K * 32) <= usable_uram_b_elements:
                N_chunk = 32
            elif (K * 16) <= usable_uram_b_elements:
                N_chunk = 16
            else:
                assert False, f"K={K} is too large to fit in usable URAM elements={usable_uram_b_elements}"
            N_chunk_aligned = UE_VECTOR_SIZE

        usable_uram_a_elements = URAM_FULL_ELEMENTS
        output_N_size = N_chunk_aligned if N_chunk_aligned is not None else N_chunk
        M_chunk = min(UE_FMAX_CONTEXT_SIZE, M, usable_uram_a_elements // (K + output_N_size))
        assert M_chunk >= 1 and M_chunk <= M, f"M_chunk={M_chunk} must be greater than 0 and less than M={M}"

        print(f"M_chunk: {M_chunk}, N_chunk: {N_chunk}", f"N_chunk_aligned: {N_chunk_aligned}")
        print(f"URAM_A usage: {100 * (M_chunk * K + M_chunk * output_N_size) / URAM_FULL_ELEMENTS:.2f}% of URAM_NEAR_FULL_ELEMENTS")
        print(f"URAM_B usage: {100 * N_chunk * K / URAM_FULL_ELEMENTS:.2f}% of URAM_FULL_ELEMENTS")

        uram_a_start_addr = 0x00000
        uram_b_start_addr = 0x80000
        for i, m_take in self.chunk_ranges(M, M_chunk):
            self.accelerator_memory_to_sram(accelerator_dram_address=Q_DRAM_ADDR + i * K * bytes_per_element,
                                            sram_address=uram_a_start_addr,
                                            element_size=m_take * K)

            self.broadcast_mul(scalar=1 / math.sqrt(head_dim),
                                    sram_start_addr=uram_a_start_addr,
                                    sram_wb_addr=uram_a_start_addr,
                                    element_size=m_take * K)

            output_sram_wb_addr = uram_a_start_addr + m_take * K * bytes_per_element

            assert output_sram_wb_addr < 0x80000, f"output_sram_wb_addr={output_sram_wb_addr} is greater than 0x80000, which is the size of URAM_B"

            clear_en = 1
            for j, n_take in self.chunk_ranges(N, N_chunk):
                self.accelerator_memory_to_sram(accelerator_dram_address=K_DRAM_ADDR + j * K * bytes_per_element,
                                            sram_address=uram_b_start_addr,
                                            element_size=n_take * K)

                assert m_take * K + n_take * m_take<= URAM_FULL_ELEMENTS

                for output_row in range(m_take):
                    if bias_enable:
                        self.accelerator_memory_to_bias_sram(accelerator_dram_address=BIAS_DRAM_ADDR + ((i + output_row) * N + j) * bytes_per_element,
                                                             element_size=n_take)

                    if N_chunk_aligned is None:
                        out_sram_offset = output_row * n_take * bytes_per_element
                    else:
                        out_sram_offset = output_row * N_chunk_aligned * bytes_per_element

                    self.start_queue_for_bf16_matvec_operation(max_clear_en=clear_en,
                                                            fmax_context_addr=output_row,
                                                            vector_sram_start_addr=uram_a_start_addr + output_row * K * bytes_per_element,
                                                            matrix_sram_start_addr=uram_b_start_addr,
                                                            output_sram_wb_addr=output_sram_wb_addr + out_sram_offset,
                                                            K=K,
                                                            N=n_take,
                                                            bias_enable=bias_enable)
                    clear_en = 0

                # TODO: if FMAX_CONTEXT_SIZE x seq_len can fit in URAM_A, then we can avoid copying to DRAM, create a special case for this
                start_dram_address_of_partial_matrix = SCRATCH_DRAM_PARTIAL_SM + j * bytes_per_element # the space needed is FMAX_CONTEXT_SIZE x seq_len

                if N_chunk_aligned is None:
                    self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr,
                                                    accelerator_dram_address=start_dram_address_of_partial_matrix,
                                                    element_size=m_take * n_take,
                                                    stride_bytes_per_chunk=n_take * bytes_per_element,
                                                    stride_jump_bytes=N * bytes_per_element)
                else:
                    for o_row_idx in range(m_take):
                        self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr + o_row_idx * N_chunk_aligned * bytes_per_element,
                                                        accelerator_dram_address=start_dram_address_of_partial_matrix + o_row_idx * N * bytes_per_element,
                                                        element_size=n_take)


            # SOFTMAX CALCULATION
            #print(f"softmax rows: {m_take * N} elements vs {URAM_FULL_ELEMENTS} elements")
            # DEBUG to get seq_len x seq_len sm(QK^T) results are copied to DRAM
            # start_dram_address_of_partial_row_complete_matrix = SM_OUTPUT_DRAM_ADDR + i * N * bytes_per_element #  make only FMAX_CONTEXT_SIZE x seq_len sm(QK^T) results are copied to DRAM

            # if m_take * N is greater than the space available in URAM_A, copy the matrix to DRAM
            max_m_take = min((URAM_FULL_ELEMENTS - UE_VECTOR_SIZE) // N, UE_FMAX_CONTEXT_SIZE) # worst case scenario, leave one row for output

            for m_take_chunk_idx, m_take_chunk_size in self.chunk_ranges(m_take, max_m_take):
                self.accelerator_memory_to_sram(accelerator_dram_address=SCRATCH_DRAM_PARTIAL_SM + m_take_chunk_idx * N * bytes_per_element,
                                            sram_address=uram_a_start_addr,
                                            element_size=m_take_chunk_size * N)

                # Reuse input sram_wb_addr for softmax output
                for row_idx in range(m_take_chunk_size):
                    self.start_queue_for_bf16_softmax_operation(fmax_context_addr=row_idx + m_take_chunk_idx,
                                                                vector_sram_start_addr=uram_a_start_addr + row_idx * N * bytes_per_element,
                                                                output_sram_wb_addr=uram_a_start_addr + row_idx * N * bytes_per_element,
                                                                N=N)


                # softmax output tap point - DEBUG only
                if debug_mode:
                    self.sram_to_accelerator_memory(sram_address=uram_a_start_addr,
                                    accelerator_dram_address=SM_OUTPUT_DRAM_ADDR + (i + m_take_chunk_idx) * N * bytes_per_element,
                                    element_size=m_take_chunk_size * N)

                v_tr_row_chunk_size = min((URAM_NEAR_FULL_ELEMENTS // seq_len // UE_VECTOR_SIZE) * UE_VECTOR_SIZE,
                                        ((URAM_FULL_ELEMENTS - m_take_chunk_size * seq_len) // m_take_chunk_size // UE_VECTOR_SIZE) * UE_VECTOR_SIZE,
                                        head_dim)

                v_tr_row_chunk_size_aligned = None
                if v_tr_row_chunk_size < UE_VECTOR_SIZE:
                    v_tr_row_chunk_size_aligned = UE_VECTOR_SIZE
                    if seq_len * 32 <= URAM_NEAR_FULL_ELEMENTS:
                        v_tr_row_chunk_size = 32
                    elif seq_len * 16 <= URAM_NEAR_FULL_ELEMENTS:
                        v_tr_row_chunk_size = 16
                    else:
                        assert False, f"v_tr_row_chunk_size={v_tr_row_chunk_size} is too large to fit in URAM_NEAR_FULL_ELEMENTS={URAM_NEAR_FULL_ELEMENTS}"

                v_t_sram_start_addr = 0x80000 # URAM_B start
                output_sram_wb_addr = uram_a_start_addr + m_take_chunk_size * seq_len * bytes_per_element

                for v_tr_column_idx, v_tr_column_take in self.chunk_ranges(head_dim, v_tr_row_chunk_size):
                    self.accelerator_memory_to_sram(accelerator_dram_address=SCRATCH_DRAM_ADDR + v_tr_column_idx * seq_len * bytes_per_element,
                                                sram_address=v_t_sram_start_addr,
                                                element_size=v_tr_column_take * seq_len)

                    for p_row_idx in range(m_take_chunk_size):
                        if v_tr_row_chunk_size_aligned is None:
                            output_sram_wb_offset = p_row_idx * v_tr_column_take * bytes_per_element
                        else:
                            output_sram_wb_offset = 0

                        self.start_queue_for_bf16_matvec_operation(max_clear_en=0,
                                                                fmax_context_addr=0,
                                                                vector_sram_start_addr=uram_a_start_addr + p_row_idx * seq_len * bytes_per_element,
                                                                matrix_sram_start_addr=v_t_sram_start_addr,
                                                                output_sram_wb_addr=output_sram_wb_addr + output_sram_wb_offset,
                                                                K=seq_len,
                                                                N=v_tr_column_take)

                        if v_tr_row_chunk_size_aligned is not None:
                            self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr + output_sram_wb_offset,
                                                            accelerator_dram_address=OUTPUT_DRAM_ADDR + (i + m_take_chunk_idx) * head_dim * bytes_per_element
                                                                                                        + v_tr_column_idx * bytes_per_element
                                                                                                        + p_row_idx * head_dim * bytes_per_element,
                                                            element_size=v_tr_column_take)


                    if v_tr_row_chunk_size_aligned is None:
                        self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr,
                                                        accelerator_dram_address=OUTPUT_DRAM_ADDR + (i + m_take_chunk_idx) * head_dim * bytes_per_element + v_tr_column_idx * bytes_per_element,
                                                        element_size=m_take_chunk_size * v_tr_column_take,
                                                        stride_bytes_per_chunk=v_tr_column_take * bytes_per_element,
                                                        stride_jump_bytes=head_dim * bytes_per_element)

        # Total Theoretical FLOPS: seq_len * head_dim + 2 * seq_len * head_dim * seq_len + seq_len * seq_len + seq_len * seq_len * 5 + 2 * seq_len * seq_len * head_dim
        total_flops = seq_len * head_dim # q_scale
        total_flops += 2 * seq_len * head_dim * seq_len # Q @ K^T
        if bias_enable:
            total_flops += seq_len * seq_len # bias
        total_flops += seq_len * seq_len * 5 # softmax
        total_flops += 2 * seq_len * seq_len * head_dim # sm @ v
        print(f"Total Theoretical FLOPS: {total_flops / 1e9:.6f} G")
        return total_flops

    def flash_attention_core_pbi(self, head_dim: int, Q_DRAM_ADDR: int, K_DRAM_ADDR: int, V_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, SCRATCH_DRAM_ADDR: int, ATTN_P_DRAM_ADDR: int, gpr_bucket_idx: int,
                            IDENTITY_DRAM_ADDR: int = None,
                            num_buckets: int = 8,
                            BIAS_DRAM_ADDR: int = None, debug_mode: bool = False, SM_OUTPUT_DRAM_ADDR: int = None,
                            use_pbi: bool = True):
        """
        **Vᵀ** at ``SCRATCH_DRAM_ADDR``: prefer ``bf16_transpose_core`` (PBI) when available; currently uses
        ``I @ V`` via :meth:`matmat_mul_core` ``use_pbi=True`` (**no identity fast path**—full GEMM cost).
        Scale **Q** in DRAM (``1/√d``, ISA ``loop_start(M)`` + two PBI row pointers). Fused Q@Kᵀ + optional
        full-matrix bias + row softmax via :meth:`matmat_mul_core` ``use_pbi=True`` into ``ATTN_P_DRAM_ADDR``;
        then **P @ V** with a second ``matmat_mul_core`` (``use_pbi=True``). ``debug_mode`` is not supported.

        Dynamic seq_len via bucketization is **mandatory** (``gpr_bucket_idx`` is a required GPR index
        1..15): emits a header that copies ``gpr_bucket_idx`` into a temp register and runs
        ``num_buckets`` pairs of ``(ADD_DEC temp, JZ -> bucket_i)`` on the temp, followed by
        ``num_buckets`` bucket bodies covering ``seq_len = UE_VECTOR_SIZE * i`` for ``i = 1..num_buckets``;
        each bucket ends with a ``JMP -> end`` label. **``gpr_bucket_idx`` itself is read-only and
        preserved across calls** — caller can prime it once and invoke flash_attention many times.

        **Bucket indexing is 1-based** (falls out of the "decrement-until-zero" cascade):
        ``gpr_bucket_idx = K`` selects the Kth bucket body whose compile-time seq_len is
        ``K * UE_VECTOR_SIZE``. So callers should compute::

            gpr_bucket_idx = aligned_seq_len // UE_VECTOR_SIZE  # NOT (aligned_seq_len // VEC) - 1

        Examples: ``aligned_seq_len = 64 → 1``, ``128 → 2``, ``UE_VECTOR_SIZE*N → N``.
        Caller must ensure ``gpr_bucket_idx ∈ [1, num_buckets]``; out-of-range values silently fall
        through into the bucket_1 body (seq_len=UE_VECTOR_SIZE), which usually produces wrong
        activations rather than crashing.

        The returned ``bucket_flops`` is a plain Python list, so caller-side FLOPs lookup uses the
        usual 0-based offset: ``bucket_flops[gpr_bucket_idx - 1]``.

        Q/K/V/OUTPUT/SCRATCH/ATTN_P buffers must be sized for the maximum bucket
        (``UE_VECTOR_SIZE * num_buckets``). The bucket step is fixed to ``UE_VECTOR_SIZE`` (hardware
        vector width).

        ``num_buckets`` defaults to the module-level ``FLASH_ATTENTION_NUM_BUCKETS`` constant but
        may be overridden per call (e.g. test cases use a smaller ``num_buckets`` to keep the captured
        program small).
        """
        bytes_per_element = 2
        bucket_step = UE_VECTOR_SIZE

        if debug_mode:
            raise RuntimeError(
                "flash_attention_core_pbi: debug_mode / SM_OUTPUT is not supported with fused "
                "matmul+softmax; set debug_mode=False."
            )

        if num_buckets < 1:
            raise ValueError(f"flash_attention_core_pbi: num_buckets={num_buckets} must be >= 1")

        program_dram_start_addr = self.get_program_dram_addr()

        # Bucket jump header: num_buckets pairs of (ADD_DEC temp, JZ -> bucket_i_placeholder). Placeholder target = 0
        bucket_scratch_reg = self.alloc_isa_reg()
        self.generate_instruction_add_imm(src_reg_idx=gpr_bucket_idx, immediate_value=0, dst_reg_idx=bucket_scratch_reg)
        jz_capture_indices: list[int] = []
        for _ in range(num_buckets):
            self.generate_instruction_add_dec(reg_idx=bucket_scratch_reg)
            jz_capture_indices.append(self.capture_count)
            self.generate_instruction_jump_abs_jz(
                target_instruction_word_addr=0, reg_id=bucket_scratch_reg
            )

        # Bucket bodies, each followed by a JMP-to-end placeholder (also target = 0).
        bucket_start_capture_indices: list[int] = []
        end_jmp_capture_indices: list[int] = []
        bucket_flops: list[int] = []
        for i in range(num_buckets):
            self.pad_capture_to_64b_boundary()
            bucket_start_capture_indices.append(self.capture_count)
            bucket_seq_len = bucket_step * (i + 1)
            if use_pbi:
                _body_flops = self._flash_attention_pbi_body(
                    head_dim=head_dim,
                    seq_len=bucket_seq_len,
                    Q_DRAM_ADDR=Q_DRAM_ADDR,
                    K_DRAM_ADDR=K_DRAM_ADDR,
                    V_DRAM_ADDR=V_DRAM_ADDR,
                    OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                    SCRATCH_DRAM_ADDR=SCRATCH_DRAM_ADDR,
                    ATTN_P_DRAM_ADDR=ATTN_P_DRAM_ADDR,
                    BIAS_DRAM_ADDR=BIAS_DRAM_ADDR,
                    _silent=True,
                    IDENTITY_TRANSPOSE_DRAM_ADDR=IDENTITY_DRAM_ADDR,
                )
            else:
                _body_flops = self.flash_attention_core_legacy(
                    head_dim=head_dim,
                    seq_len=bucket_seq_len,
                    Q_DRAM_ADDR=Q_DRAM_ADDR,
                    K_DRAM_ADDR=K_DRAM_ADDR,
                    V_DRAM_ADDR=V_DRAM_ADDR,
                    OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                    SCRATCH_DRAM_ADDR=SCRATCH_DRAM_ADDR,
                    IDENTITY_DRAM_ADDR=IDENTITY_DRAM_ADDR,
                    BIAS_DRAM_ADDR=BIAS_DRAM_ADDR,
                )
            bucket_flops.append(_body_flops)
            end_jmp_capture_indices.append(self.capture_count)
            self.generate_instruction_jump_abs(target_instruction_word_addr=0)

        self.pad_capture_to_64b_boundary()
        end_capture_count = self.capture_count
        end_word_addr = ue_35bit_addr_shifter(
            program_dram_start_addr + end_capture_count * INSTRUCTION_SIZE_BYTES
        )

        # Patch header JZs -> corresponding bucket entry.
        for jz_idx, bucket_idx in zip(jz_capture_indices, bucket_start_capture_indices):
            bucket_word_addr = ue_35bit_addr_shifter(
                program_dram_start_addr + bucket_idx * INSTRUCTION_SIZE_BYTES
            )
            self._patch_jump_immediate(jz_idx, bucket_word_addr)

        # Patch bucket-tail JMPs -> shared end label.
        for jmp_idx in end_jmp_capture_indices:
            self._patch_jump_immediate(jmp_idx, end_word_addr)

        self.release_isa_reg()  # bucket_scratch_reg

        print(
            f"flash_attention_core_pbi (bucketized): {num_buckets} buckets, "
            f"seq_len={bucket_step}..{num_buckets * bucket_step}, "
            f"Theoretical FLOPS min-bucket={bucket_flops[0] / 1e9:.6f} G, "
            f"max-bucket={bucket_flops[-1] / 1e9:.6f} G"
        )
        return bucket_flops

    def _flash_attention_pbi_body(self, head_dim: int, seq_len: int, Q_DRAM_ADDR: int, K_DRAM_ADDR: int,
                                   V_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, SCRATCH_DRAM_ADDR: int,
                                   ATTN_P_DRAM_ADDR: int, BIAS_DRAM_ADDR: int,
                                   _silent: bool = False,
                                   IDENTITY_TRANSPOSE_DRAM_ADDR: int = None) -> int:
        """Single concrete-``seq_len`` body of :meth:`flash_attention_core_pbi`.

        IDENTITY_TRANSPOSE_DRAM_ADDR: forwarded from IDENTITY_DRAM_ADDR of the outer call; passed
        to bf16_transpose_core_pbi to avoid a per-bucket allocate_params_dram + dma_write.
        """
        bytes_per_element = 2

        # matmat PBI path is now driven by gpr_M_reg; allocate a shared GPR for this bucket body
        # and re-prime it before each matmul (each uses a different compile-time M).
        m_reg = self.alloc_isa_reg()

        # Materialize Vᵀ efficiently using the dedicated dynamic transpose core.
        # V is (seq_len × head_dim) -> V^T at SCRATCH is (head_dim × seq_len)
        self.bf16_transpose_core_pbi(
            M=seq_len,
            N=head_dim,
            INPUT_DRAM_ADDR=V_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=SCRATCH_DRAM_ADDR,
            IDENTITY_DRAM_ADDR=IDENTITY_TRANSPOSE_DRAM_ADDR,
        )

        bias_enable = BIAS_DRAM_ADDR is not None

        M = seq_len   # attention matrix row count
        N = seq_len   # attention matrix column count (key index)
        qk_k = head_dim  # inner dim for Q and K

        if N % UE_VECTOR_SIZE != 0:
            raise ValueError(
                f"flash_attention_core_pbi: fused softmax requires seq_len % UE_VECTOR_SIZE == 0, got seq_len={N}"
            )

        # --- Q scale (attention): multiply every row of Q by 1/sqrt(head_dim) in DRAM in place -------
        attn_scale = 1.0 / math.sqrt(head_dim)
        row_bytes = qk_k * bytes_per_element
        vector_sram_addr = 0x00000
        if qk_k > URAM_NEAR_FULL_ELEMENTS:
            raise ValueError(
                f"flash_attention_core_pbi: head_dim={qk_k} exceeds URAM_NEAR_FULL_ELEMENTS={URAM_NEAR_FULL_ELEMENTS} "
                "(ISA Q-scale loop stages one full row in URAM)"
            )

        vector_uram_type, vector_uram_start_addr = self.sram_address_to_uram_address(vector_sram_addr)
        assert vector_uram_type == URAM_SECTION.URAM_A, "Q staging must be URAM_A"

        # Two PBI program pointers (same as rms_norm_core_dram_pbi): load row from Q_DRAM_ADDR,
        # write row back to the same buffer; each loop iteration advances both by row_bytes in DRAM.
        row_load_ptr = self.alloc_inst_ptr()
        row_store_ptr = self.alloc_inst_ptr()
        self.generate_instruction_pbi_init(
            dram_shared_addr=Q_DRAM_ADDR,
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
            dram_shared_addr=Q_DRAM_ADDR,
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

        # ISA loop over query rows (M == seq_len): anchor loop body in I-cache, then M iterations.
        program_dram_start_addr = self.get_program_dram_addr()
        cur_inst_count = self.capture_count
        self.generate_instruction_jump_abs(
            ue_35bit_addr_shifter(program_dram_start_addr + (cur_inst_count + 1) * INSTRUCTION_SIZE_BYTES)
        )
        self.loop_start(M)
        # Body per row: H2C one row → scalar multiply in URAM → C2H row back (PBI: element_size=0 uses ptrs).
        self.accelerator_memory_to_sram(
            accelerator_dram_address=row_bytes,
            sram_address=vector_sram_addr,
            element_size=0,
            inst_pointer_idx=row_load_ptr,
        )
        self.broadcast_mul(
            scalar=attn_scale,
            sram_start_addr=vector_sram_addr,
            sram_wb_addr=vector_sram_addr,
            element_size=qk_k,
        )
        self.sram_to_accelerator_memory(
            sram_address=vector_sram_addr,
            accelerator_dram_address=row_bytes,
            element_size=0,
            inst_pointer_idx=row_store_ptr,
        )
        q_scale_loop_size = self.loop_end()
        if not _silent:
            print(f"flash_attention_core_pbi Q-scale loop body: {q_scale_loop_size} instructions (M={M}, head_dim={qk_k})")
        assert q_scale_loop_size <= 256, (
            f"Q-scale ISA loop body {q_scale_loop_size} exceeds i-cache 256"
        )
        self.release_inst_ptr(row_store_ptr)
        self.release_inst_ptr(row_load_ptr)

        # Q @ K^T + optional full-matrix bias + row softmax → ATTN_P
        self.generate_instruction_add_set(m_reg, M)
        self.matmat_mul_core(
            M=M,
            K=qk_k,
            N=N,
            A_DRAM_ADDR=Q_DRAM_ADDR,
            B_DRAM_ADDR=K_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=ATTN_P_DRAM_ADDR,
            softmax_enable=True,
            C_DRAM_ADDR=BIAS_DRAM_ADDR if bias_enable else None,
            bias_mode="full_matrix",
            gpr_M_reg=m_reg,
        )

        # P @ V: A = P (M×K), B = V^T DRAM (N×K); matmat_mul_core yields A @ B^T → (seq_len, head_dim)
        self.generate_instruction_add_set(m_reg, seq_len)
        self.matmat_mul_core(
            M=seq_len,
            K=seq_len,
            N=head_dim,
            A_DRAM_ADDR=ATTN_P_DRAM_ADDR,
            B_DRAM_ADDR=SCRATCH_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
            gpr_M_reg=m_reg,
        )
        self.release_isa_reg()

        total_flops = seq_len * head_dim # q_scale
        total_flops += 2 * seq_len * head_dim * seq_len # Q @ K^T
        if bias_enable:
            total_flops += seq_len * seq_len # bias
        total_flops += seq_len * seq_len * 5 # softmax
        total_flops += 2 * seq_len * seq_len * head_dim # sm @ v
        if not _silent:
            print(f"Total Theoretical FLOPS: {total_flops / 1e9:.6f} G")
        return total_flops

    # =========================================================================
    # Decoder group attention (single-query GQA used by autoregressive decode)
    # =========================================================================
    def decoder_group_attention_core(
        self,
        group_size: int,
        head_dim: int,
        seq_len: int,
        Q_DRAM_ADDR: int,
        K_DRAM_ADDR: int,
        V_DRAM_ADDR: int,
        OUTPUT_DRAM_ADDR: int,
        SCRATCH_DRAM_ADDR: int,
        IDENTITY_DRAM_ADDR: int = None,
        BIAS_DRAM_ADDR: int = None,
        debug_mode: bool = False,
        SM_OUTPUT_DRAM_ADDR: int = None,
        gpr_bucket_idx: int = None,
        num_buckets: int = 8,
        use_pbi: bool = True,
    ):
        """Decoder group attention entrypoint; dispatches based on ``gpr_bucket_idx``:

        - ``gpr_bucket_idx`` is a GPR index (1..15): :meth:`decoder_group_attention_core_pbi` — emits
          a ``num_buckets``-way bucket dispatcher (each body sized for ``seq_len = K*UE_VECTOR_SIZE``).
          ``seq_len`` arg is ignored (bucket bodies cover the range). Returns a per-bucket FLOPS list.
        - ``gpr_bucket_idx is None`` (default): :meth:`decoder_group_attention_core_legacy` — single
          static-seq_len body. Returns int total FLOPS.
        """
        if gpr_bucket_idx is not None:
            return self.decoder_group_attention_core_pbi(
                group_size=group_size,
                head_dim=head_dim,
                Q_DRAM_ADDR=Q_DRAM_ADDR,
                K_DRAM_ADDR=K_DRAM_ADDR,
                V_DRAM_ADDR=V_DRAM_ADDR,
                OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                SCRATCH_DRAM_ADDR=SCRATCH_DRAM_ADDR,
                gpr_bucket_idx=gpr_bucket_idx,
                num_buckets=num_buckets,
                IDENTITY_DRAM_ADDR=IDENTITY_DRAM_ADDR,
                BIAS_DRAM_ADDR=BIAS_DRAM_ADDR,
                debug_mode=debug_mode,
                SM_OUTPUT_DRAM_ADDR=SM_OUTPUT_DRAM_ADDR,
                use_pbi=use_pbi,
            )
        return self.decoder_group_attention_core_legacy(
            group_size=group_size,
            head_dim=head_dim,
            seq_len=seq_len,
            Q_DRAM_ADDR=Q_DRAM_ADDR,
            K_DRAM_ADDR=K_DRAM_ADDR,
            V_DRAM_ADDR=V_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
            SCRATCH_DRAM_ADDR=SCRATCH_DRAM_ADDR,
            IDENTITY_DRAM_ADDR=IDENTITY_DRAM_ADDR,
            BIAS_DRAM_ADDR=BIAS_DRAM_ADDR,
            debug_mode=debug_mode,
            SM_OUTPUT_DRAM_ADDR=SM_OUTPUT_DRAM_ADDR,
        )

    def decoder_group_attention_core_legacy(
        self,
        group_size: int,
        head_dim: int,
        seq_len: int,
        Q_DRAM_ADDR: int,
        K_DRAM_ADDR: int,
        V_DRAM_ADDR: int,
        OUTPUT_DRAM_ADDR: int,
        SCRATCH_DRAM_ADDR: int,
        IDENTITY_DRAM_ADDR: int = None,
        BIAS_DRAM_ADDR: int = None,
        debug_mode: bool = False,
        SM_OUTPUT_DRAM_ADDR: int = None,
    ) -> int:
        """Legacy GQA decode attention: Q is ``group_size × head_dim`` (one new token replicated
        across the GQA query heads), K/V are ``seq_len × head_dim`` from the KV cache.

        Conceptually a per-group ``softmax(q·Kᵀ/√d) · V`` with ``M=1`` per group. Compiled as
        a Python ``for g in range(group_size)`` loop, each iteration emitting its own
        chunked V^T materialization + Q@Kᵀ + softmax + P@V sequence.
        """
        total_flops = 0
        group_stride_bytes = head_dim * self.bytes_per_element
        bytes_per_element = 2
        bias_enable = True if BIAS_DRAM_ADDR is not None else False

        if debug_mode: # DEBUG only, needs to be allocated in DRAM
            assert SM_OUTPUT_DRAM_ADDR is not None, "SM_OUTPUT_DRAM_ADDR is not set for debug mode"

        identity_tensor = torch.eye(head_dim, dtype=torch.bfloat16)
        for g in range(group_size):
            group_q_dram_addr = Q_DRAM_ADDR + g * group_stride_bytes
            group_output_dram_addr = OUTPUT_DRAM_ADDR + g * group_stride_bytes

            # SCRATCH_DRAM_ADDR is used for V^T
            SCRATCH_DRAM_PARTIAL_SM = SCRATCH_DRAM_ADDR + head_dim * seq_len * bytes_per_element # used for partial softmax output

            # ----------------------------------------------------------------------------------------------------------------
            # I @ V^T: (head_dim, head_dim) @ (seq_len, head_dim)^T -> (head_dim, seq_len)
            # Convention: first matrix I is (M, K), second V^T is (K, N), output  (M, N)
            M = head_dim   # identity length (rows of I)
            K = head_dim  # identity dimension (inner product dim)
            N = seq_len   # V length (columns of V^T)

            # transfer identity matrix to URAM_A start
            self.accelerator_memory_to_sram(accelerator_dram_address=IDENTITY_DRAM_ADDR,
                                            sram_address=0,
                                            element_size=UE_VECTOR_SIZE * UE_VECTOR_SIZE)

            usable_uram_a_start_addr = UE_VECTOR_SIZE * UE_VECTOR_SIZE * bytes_per_element

            # URAM_B is used for V matrix, we need to chunk the V matrix into smaller chunks that can fit in URAM_B
            usable_uram_b_elements = URAM_NEAR_FULL_ELEMENTS
            N_chunk = min(N, (usable_uram_b_elements // K) // UE_VECTOR_SIZE * UE_VECTOR_SIZE)
            N_chunk_aligned = None
            if N_chunk < UE_VECTOR_SIZE:
                if (K * 32) <= usable_uram_b_elements:
                    N_chunk = 32
                elif (K * 16) <= usable_uram_b_elements:
                    N_chunk = 16
                else:
                    assert False, f"K={K} is too large to fit in usable URAM elements={usable_uram_b_elements}"
                N_chunk_aligned = UE_VECTOR_SIZE

            usable_uram_a_elements = URAM_FULL_ELEMENTS - UE_VECTOR_SIZE * UE_VECTOR_SIZE
            output_N_size = N_chunk_aligned if N_chunk_aligned is not None else N_chunk
            M_chunk = min(M, usable_uram_a_elements // output_N_size)
            assert M_chunk >= 1 and M_chunk <= M, f"M_chunk={M_chunk} must be greater than 0 and less than M={M}"

            print(f"M_chunk: {M_chunk}, N_chunk: {N_chunk}", f"N_chunk_aligned: {N_chunk_aligned}")
            print(f"URAM_A usage: {100 * (UE_VECTOR_SIZE * UE_VECTOR_SIZE + M_chunk * output_N_size) / URAM_FULL_ELEMENTS:.2f}% of URAM_NEAR_FULL_ELEMENTS")
            print(f"URAM_B usage: {100 * N_chunk * K / URAM_FULL_ELEMENTS:.2f}% of URAM_FULL_ELEMENTS")

            output_sram_wb_addr = usable_uram_a_start_addr
            uram_b_start_addr = 0x80000
            for i, m_take in self.chunk_ranges(M, M_chunk):
                for j, n_take in self.chunk_ranges(N, N_chunk):

                    self.accelerator_memory_to_sram(accelerator_dram_address=V_DRAM_ADDR + j * K * bytes_per_element,
                                                sram_address=uram_b_start_addr,
                                                element_size=n_take * K)

                    for output_row in range(m_take):
                        if N_chunk_aligned is None:
                            out_sram_offset = output_row * n_take * bytes_per_element
                        else:
                            out_sram_offset = output_row * N_chunk_aligned * bytes_per_element

                        ones_idx = identity_tensor[output_row+i, :].reshape(-1, UE_VECTOR_SIZE).sum(axis=1).argmax(axis=0)
                        vector_idx = identity_tensor[output_row+i, :].reshape(-1, UE_VECTOR_SIZE)[ones_idx, :].argmax(axis=0)

                        self.start_queue_for_bf16_matvec_operation(max_clear_en=0,
                                                                fmax_context_addr=0,
                                                                vector_sram_start_addr=0x00000 + vector_idx * UE_VECTOR_SIZE * bytes_per_element,
                                                                matrix_sram_start_addr=uram_b_start_addr + ones_idx * UE_VECTOR_SIZE * bytes_per_element,
                                                                output_sram_wb_addr=output_sram_wb_addr + out_sram_offset,
                                                                K=UE_VECTOR_SIZE,
                                                                N=n_take,
                                                                stride_z=m_take)

                    start_dram_address_of_partial_matrix = SCRATCH_DRAM_ADDR + i * N * bytes_per_element + j * bytes_per_element # the space needed is head_dim x seq_len

                    if N_chunk_aligned is None:
                        self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr,
                                                        accelerator_dram_address=start_dram_address_of_partial_matrix,
                                                        element_size=m_take * n_take,
                                                        stride_bytes_per_chunk=n_take * bytes_per_element,
                                                        stride_jump_bytes=N * bytes_per_element)
                    else:
                        for o_row_idx in range(m_take):
                            self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr + o_row_idx * N_chunk_aligned * bytes_per_element,
                                                            accelerator_dram_address=start_dram_address_of_partial_matrix + o_row_idx * N * bytes_per_element,
                                                            element_size=n_take)

            # ----------------------------------------------------------------------------------------------------------------
            # Q @ K^T: (1, head_dim) @ (head_dim, seq_len) -> (1, seq_len)
            # Convention: first matrix Q is (M, K), second K^T is (K, N), output scores (M, N)
            M = 1         # query length (rows of Q)
            K = head_dim  # head dimension (inner product dim)
            N = seq_len   # key length (columns of K^T)
            # Calculate N_chunk
            usable_uram_b_elements = URAM_NEAR_FULL_ELEMENTS
            N_chunk = min(N, (usable_uram_b_elements // K) // UE_VECTOR_SIZE * UE_VECTOR_SIZE)
            N_chunk_aligned = None
            if N_chunk < UE_VECTOR_SIZE:
                if (K * 32) <= usable_uram_b_elements:
                    N_chunk = 32
                elif (K * 16) <= usable_uram_b_elements:
                    N_chunk = 16
                else:
                    assert False, f"K={K} is too large to fit in usable URAM elements={usable_uram_b_elements}"
                N_chunk_aligned = UE_VECTOR_SIZE

            usable_uram_a_elements = URAM_FULL_ELEMENTS
            output_N_size = N_chunk_aligned if N_chunk_aligned is not None else N_chunk
            M_chunk = min(UE_FMAX_CONTEXT_SIZE, M, usable_uram_a_elements // (K + output_N_size))
            assert M_chunk >= 1 and M_chunk <= M, f"M_chunk={M_chunk} must be greater than 0 and less than M={M}"

            print(f"M_chunk: {M_chunk}, N_chunk: {N_chunk}", f"N_chunk_aligned: {N_chunk_aligned}")
            print(f"URAM_A usage: {100 * (M_chunk * K + M_chunk * output_N_size) / URAM_FULL_ELEMENTS:.2f}% of URAM_NEAR_FULL_ELEMENTS")
            print(f"URAM_B usage: {100 * N_chunk * K / URAM_FULL_ELEMENTS:.2f}% of URAM_FULL_ELEMENTS")

            uram_a_start_addr = 0x00000
            uram_b_start_addr = 0x80000
            for i, m_take in self.chunk_ranges(M, M_chunk):
                self.accelerator_memory_to_sram(accelerator_dram_address=group_q_dram_addr + i * K * bytes_per_element,
                                                sram_address=uram_a_start_addr,
                                                element_size=m_take * K)

                self.broadcast_mul(scalar=1 / math.sqrt(head_dim),
                                        sram_start_addr=uram_a_start_addr,
                                        sram_wb_addr=uram_a_start_addr,
                                        element_size=m_take * K)

                output_sram_wb_addr = uram_a_start_addr + m_take * K * bytes_per_element

                assert output_sram_wb_addr < 0x80000, f"output_sram_wb_addr={output_sram_wb_addr} is greater than 0x80000, which is the size of URAM_B"

                clear_en = 1
                for j, n_take in self.chunk_ranges(N, N_chunk):
                    self.accelerator_memory_to_sram(accelerator_dram_address=K_DRAM_ADDR + j * K * bytes_per_element,
                                                sram_address=uram_b_start_addr,
                                                element_size=n_take * K)

                    if bias_enable:
                        self.accelerator_memory_to_bias_sram(accelerator_dram_address=BIAS_DRAM_ADDR + j * bytes_per_element,
                                                           element_size=n_take)

                    assert m_take * K + n_take * m_take<= URAM_FULL_ELEMENTS

                    for output_row in range(m_take):
                        # removed bias_enable as per causal mask drop

                        if N_chunk_aligned is None:
                            out_sram_offset = output_row * n_take * bytes_per_element
                        else:
                            out_sram_offset = output_row * N_chunk_aligned * bytes_per_element

                        self.start_queue_for_bf16_matvec_operation(max_clear_en=clear_en,
                                                                fmax_context_addr=output_row,
                                                                vector_sram_start_addr=uram_a_start_addr + output_row * K * bytes_per_element,
                                                                matrix_sram_start_addr=uram_b_start_addr,
                                                                output_sram_wb_addr=output_sram_wb_addr + out_sram_offset,
                                                                K=K,
                                                                N=n_take,
                                                                bias_enable=bias_enable)
                        clear_en = 0

                    # TODO: if FMAX_CONTEXT_SIZE x seq_len can fit in URAM_A, then we can avoid copying to DRAM, create a special case for this
                    start_dram_address_of_partial_matrix = SCRATCH_DRAM_PARTIAL_SM + j * bytes_per_element # the space needed is FMAX_CONTEXT_SIZE x seq_len

                    if N_chunk_aligned is None:
                        self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr,
                                                        accelerator_dram_address=start_dram_address_of_partial_matrix,
                                                        element_size=m_take * n_take,
                                                        stride_bytes_per_chunk=n_take * bytes_per_element,
                                                        stride_jump_bytes=N * bytes_per_element)
                    else:
                        for o_row_idx in range(m_take):
                            self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr + o_row_idx * N_chunk_aligned * bytes_per_element,
                                                            accelerator_dram_address=start_dram_address_of_partial_matrix + o_row_idx * N * bytes_per_element,
                                                            element_size=n_take)


                # SOFTMAX CALCULATION
                # if m_take * N is greater than the space available in URAM_A, copy the matrix to DRAM
                max_m_take = min((URAM_FULL_ELEMENTS - UE_VECTOR_SIZE) // N, UE_FMAX_CONTEXT_SIZE) # worst case scenario, leave one row for output

                for m_take_chunk_idx, m_take_chunk_size in self.chunk_ranges(m_take, max_m_take):
                    self.accelerator_memory_to_sram(accelerator_dram_address=SCRATCH_DRAM_PARTIAL_SM + m_take_chunk_idx * N * bytes_per_element,
                                                sram_address=uram_a_start_addr,
                                                element_size=m_take_chunk_size * N)

                    # Reuse input sram_wb_addr for softmax output
                    for row_idx in range(m_take_chunk_size):
                        self.start_queue_for_bf16_softmax_operation(fmax_context_addr=row_idx + m_take_chunk_idx,
                                                                    vector_sram_start_addr=uram_a_start_addr + row_idx * N * bytes_per_element,
                                                                    output_sram_wb_addr=uram_a_start_addr + row_idx * N * bytes_per_element,
                                                                    N=N)


                    # softmax output tap point - DEBUG only
                    if debug_mode:
                        self.sram_to_accelerator_memory(sram_address=uram_a_start_addr,
                                        accelerator_dram_address=SM_OUTPUT_DRAM_ADDR + (i + m_take_chunk_idx) * N * bytes_per_element,
                                        element_size=m_take_chunk_size * N)

                    v_tr_row_chunk_size = min((URAM_NEAR_FULL_ELEMENTS // seq_len // UE_VECTOR_SIZE) * UE_VECTOR_SIZE,
                                            ((URAM_FULL_ELEMENTS - m_take_chunk_size * seq_len) // m_take_chunk_size // UE_VECTOR_SIZE) * UE_VECTOR_SIZE,
                                            head_dim)

                    v_tr_row_chunk_size_aligned = None
                    if v_tr_row_chunk_size < UE_VECTOR_SIZE:
                        v_tr_row_chunk_size_aligned = UE_VECTOR_SIZE
                        if seq_len * 32 <= URAM_NEAR_FULL_ELEMENTS:
                            v_tr_row_chunk_size = 32
                        elif seq_len * 16 <= URAM_NEAR_FULL_ELEMENTS:
                            v_tr_row_chunk_size = 16
                        else:
                            assert False, f"v_tr_row_chunk_size={v_tr_row_chunk_size} is too large to fit in URAM_NEAR_FULL_ELEMENTS={URAM_NEAR_FULL_ELEMENTS}"

                    v_t_sram_start_addr = 0x80000 # URAM_B start
                    output_sram_wb_addr = uram_a_start_addr + m_take_chunk_size * seq_len * bytes_per_element

                    for v_tr_column_idx, v_tr_column_take in self.chunk_ranges(head_dim, v_tr_row_chunk_size):
                        self.accelerator_memory_to_sram(accelerator_dram_address=SCRATCH_DRAM_ADDR + v_tr_column_idx * seq_len * bytes_per_element,
                                                    sram_address=v_t_sram_start_addr,
                                                    element_size=v_tr_column_take * seq_len)

                        for p_row_idx in range(m_take_chunk_size):
                            if v_tr_row_chunk_size_aligned is None:
                                output_sram_wb_offset = p_row_idx * v_tr_column_take * bytes_per_element
                            else:
                                output_sram_wb_offset = 0

                            self.start_queue_for_bf16_matvec_operation(max_clear_en=0,
                                                                    fmax_context_addr=0,
                                                                    vector_sram_start_addr=uram_a_start_addr + p_row_idx * seq_len * bytes_per_element,
                                                                    matrix_sram_start_addr=v_t_sram_start_addr,
                                                                    output_sram_wb_addr=output_sram_wb_addr + output_sram_wb_offset,
                                                                    K=seq_len,
                                                                    N=v_tr_column_take)

                            if v_tr_row_chunk_size_aligned is not None:
                                self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr + output_sram_wb_offset,
                                                                accelerator_dram_address=group_output_dram_addr + (i + m_take_chunk_idx) * head_dim * bytes_per_element
                                                                                                            + v_tr_column_idx * bytes_per_element
                                                                                                            + p_row_idx * head_dim * bytes_per_element,
                                                                element_size=v_tr_column_take)


                        if v_tr_row_chunk_size_aligned is None:
                            self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr,
                                                            accelerator_dram_address=group_output_dram_addr + (i + m_take_chunk_idx) * head_dim * bytes_per_element + v_tr_column_idx * bytes_per_element,
                                                            element_size=m_take_chunk_size * v_tr_column_take,
                                                            stride_bytes_per_chunk=v_tr_column_take * bytes_per_element,
                                                            stride_jump_bytes=head_dim * bytes_per_element)

            # Total Theoretical FLOPS
            group_flops = 1 * head_dim # q_scale
            group_flops += 2 * 1 * head_dim * seq_len # Q @ K^T
            group_flops += 1 * seq_len * 5 # softmax
            group_flops += 2 * 1 * seq_len * head_dim # sm @ v
            print(f"Total Theoretical FLOPS: {group_flops}")
            total_flops += group_flops
        return total_flops

    def decoder_group_attention_core_pbi(
        self,
        group_size: int,
        head_dim: int,
        Q_DRAM_ADDR: int,
        K_DRAM_ADDR: int,
        V_DRAM_ADDR: int,
        OUTPUT_DRAM_ADDR: int,
        SCRATCH_DRAM_ADDR: int,
        gpr_bucket_idx: int,
        num_buckets: int = 8,
        IDENTITY_DRAM_ADDR: int = None,
        BIAS_DRAM_ADDR: int = None,
        debug_mode: bool = False,
        SM_OUTPUT_DRAM_ADDR: int = None,
        use_pbi: bool = True,
    ) -> list:
        """Bucketized decoder group attention. Mirrors the dispatcher shape of
        :meth:`flash_attention_core_pbi`: emits ``num_buckets`` complete bucket bodies (one per
        ``seq_len = UE_VECTOR_SIZE * i`` for ``i = 1..num_buckets``), with a JZ-cascade header
        that routes via the runtime ``gpr_bucket_idx`` GPR (1-based selector preserved across calls).

        ``group_size`` is the matmul ``M`` dimension and is always static (fixed by the model).

        Caller must size ``SCRATCH_DRAM_ADDR`` for the **maximum** bucket so the per-bucket offset
        arithmetic for ``score_dram_addr`` / ``scaled_q_dram_addr`` lands inside the allocation.

        Returns ``list[int]`` — per-bucket FLOPS; caller picks ``bucket_flops[gpr_bucket_idx - 1]``.
        """
        del debug_mode, SM_OUTPUT_DRAM_ADDR

        if not (1 <= gpr_bucket_idx <= 15):
            raise ValueError(
                f"decoder_group_attention_core_pbi: gpr_bucket_idx={gpr_bucket_idx} must be a GPR index in [1, 15]"
            )
        if num_buckets < 1:
            raise ValueError(
                f"decoder_group_attention_core_pbi: num_buckets={num_buckets} must be >= 1"
            )

        program_dram_start_addr = self.get_program_dram_addr()
        bucket_step = UE_VECTOR_SIZE

        # Bucket jump header: copy gpr_bucket_idx into a scratch reg so the JZ cascade leaves the
        # caller's bucket register untouched.
        bucket_scratch_reg = self.alloc_isa_reg()
        self.generate_instruction_add_imm(
            src_reg_idx=gpr_bucket_idx, immediate_value=0, dst_reg_idx=bucket_scratch_reg
        )
        jz_capture_indices: list = []
        for _ in range(num_buckets):
            self.generate_instruction_add_dec(reg_idx=bucket_scratch_reg)
            jz_capture_indices.append(self.capture_count)
            self.generate_instruction_jump_abs_jz(
                target_instruction_word_addr=0, reg_id=bucket_scratch_reg
            )

        # Bucket bodies, each followed by a JMP-to-end placeholder.
        bucket_start_capture_indices: list = []
        end_jmp_capture_indices: list = []
        bucket_flops: list = []
        for i in range(num_buckets):
            self.pad_capture_to_64b_boundary()
            bucket_start_capture_indices.append(self.capture_count)
            bucket_seq_len = bucket_step * (i + 1)
            if use_pbi:
                _body_flops = self._decoder_group_attention_pbi_body(
                    group_size=group_size,
                    head_dim=head_dim,
                    seq_len=bucket_seq_len,
                    Q_DRAM_ADDR=Q_DRAM_ADDR,
                    K_DRAM_ADDR=K_DRAM_ADDR,
                    V_DRAM_ADDR=V_DRAM_ADDR,
                    OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                    SCRATCH_DRAM_ADDR=SCRATCH_DRAM_ADDR,
                    BIAS_DRAM_ADDR=BIAS_DRAM_ADDR,
                    IDENTITY_TRANSPOSE_DRAM_ADDR=IDENTITY_DRAM_ADDR,
                )
            else:
                _body_flops = self.decoder_group_attention_core_legacy(
                    group_size=group_size,
                    head_dim=head_dim,
                    seq_len=bucket_seq_len,
                    Q_DRAM_ADDR=Q_DRAM_ADDR,
                    K_DRAM_ADDR=K_DRAM_ADDR,
                    V_DRAM_ADDR=V_DRAM_ADDR,
                    OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                    SCRATCH_DRAM_ADDR=SCRATCH_DRAM_ADDR,
                    IDENTITY_DRAM_ADDR=IDENTITY_DRAM_ADDR,
                    BIAS_DRAM_ADDR=BIAS_DRAM_ADDR,
                )
            bucket_flops.append(_body_flops)
            end_jmp_capture_indices.append(self.capture_count)
            self.generate_instruction_jump_abs(target_instruction_word_addr=0)

        self.pad_capture_to_64b_boundary()
        end_capture_count = self.capture_count
        end_word_addr = ue_35bit_addr_shifter(
            program_dram_start_addr + end_capture_count * INSTRUCTION_SIZE_BYTES
        )

        # Patch header JZs -> corresponding bucket entry.
        for jz_idx, bucket_start_idx in zip(jz_capture_indices, bucket_start_capture_indices):
            bucket_word_addr = ue_35bit_addr_shifter(
                program_dram_start_addr + bucket_start_idx * INSTRUCTION_SIZE_BYTES
            )
            self._patch_jump_immediate(jz_idx, bucket_word_addr)

        # Patch bucket-tail JMPs -> shared end label.
        for jmp_idx in end_jmp_capture_indices:
            self._patch_jump_immediate(jmp_idx, end_word_addr)

        self.release_isa_reg()  # bucket_scratch_reg

        print(
            f"decoder_group_attention_core_pbi (bucketized): {num_buckets} buckets, "
            f"seq_len={bucket_step}..{num_buckets * bucket_step}, "
            f"FLOPS min-bucket={bucket_flops[0] / 1e9:.6f} G, "
            f"max-bucket={bucket_flops[-1] / 1e9:.6f} G"
        )
        return bucket_flops

    def _decoder_group_attention_pbi_body(
        self,
        group_size: int,
        head_dim: int,
        seq_len: int,
        Q_DRAM_ADDR: int,
        K_DRAM_ADDR: int,
        V_DRAM_ADDR: int,
        OUTPUT_DRAM_ADDR: int,
        SCRATCH_DRAM_ADDR: int,
        BIAS_DRAM_ADDR: int = None,
        IDENTITY_TRANSPOSE_DRAM_ADDR: int = None,
    ) -> int:
        """Single concrete-``seq_len`` body of :meth:`decoder_group_attention_core_pbi`.

        All five DRAM addresses are absolute (caller-supplied); each bucket body writes its
        ``score`` / ``scaled_q`` scratch using its own static ``seq_len``-derived offsets within
        SCRATCH_DRAM_ADDR. As long as the caller has allocated SCRATCH_DRAM_ADDR for the maximum
        bucket's seq_len, the per-bucket offsets stay inside the allocation.
        IDENTITY_TRANSPOSE_DRAM_ADDR: forwarded from IDENTITY_DRAM_ADDR; passed to
        bf16_transpose_core_pbi to avoid a per-bucket allocate_params_dram + dma_write.
        """
        bytes_per_element = self.bytes_per_element
        score_dram_addr = SCRATCH_DRAM_ADDR + head_dim * seq_len * bytes_per_element
        scaled_q_dram_addr = score_dram_addr + group_size * seq_len * bytes_per_element

        # Materialize V^T once since K/V cache is shared across all query groups.
        self.bf16_transpose_core_pbi(
            M=seq_len,
            N=head_dim,
            INPUT_DRAM_ADDR=V_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=SCRATCH_DRAM_ADDR,
            IDENTITY_DRAM_ADDR=IDENTITY_TRANSPOSE_DRAM_ADDR,
        )

        self.accelerator_memory_to_sram(
            accelerator_dram_address=Q_DRAM_ADDR,
            sram_address=0x00000,
            element_size=group_size * head_dim,
        )
        self.broadcast_mul(
            scalar=1 / math.sqrt(head_dim),
            sram_start_addr=0x00000,
            sram_wb_addr=0x00000,
            element_size=group_size * head_dim,
        )
        self.sram_to_accelerator_memory(
            sram_address=0x00000,
            accelerator_dram_address=scaled_q_dram_addr,
            element_size=group_size * head_dim,
        )

        # PBI matmul path is driven by gpr_M_reg; allocate a GPR primed with group_size.
        m_reg = self.alloc_isa_reg()
        self.generate_instruction_add_set(m_reg, group_size)

        self.matmat_mul_core(
            M=group_size,
            K=head_dim,
            N=seq_len,
            A_DRAM_ADDR=scaled_q_dram_addr,
            B_DRAM_ADDR=K_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=score_dram_addr,
            softmax_enable=True,
            C_DRAM_ADDR=BIAS_DRAM_ADDR,
            gpr_M_reg=m_reg,
        )
        self.matmat_mul_core(
            M=group_size,
            K=seq_len,
            N=head_dim,
            A_DRAM_ADDR=score_dram_addr,
            B_DRAM_ADDR=SCRATCH_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
            gpr_M_reg=m_reg,
        )
        self.release_isa_reg()  # m_reg

        # Match legacy decoder_group_attention_core GFLOP accounting exactly.
        group_flops = 1 * head_dim
        group_flops += 2 * 1 * head_dim * seq_len
        group_flops += 1 * seq_len * 5
        group_flops += 2 * 1 * seq_len * head_dim
        return group_size * group_flops

    def quantized_matmat_core(self, M: int, K: int, N: int, A_DRAM_ADDR: int, B_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, SCALE_DRAM_ADDR: int, C_DRAM_ADDR: int = None, bias_mode: str = "broadcast_N", data_type: TYPE = None, gelu_enable: bool = False, silu_enable: bool = False, sigmoid_enable: bool = False, clamp_enable: bool = False, log_enable: bool = False) -> None:
        """Quantized matrix-matrix multiplication core.
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

                for vector_idx in range(M_chunk_size): # for each input vector

                    if bias_enable and bias_mode == "full_matrix":
                        self.accelerator_memory_to_bias_sram(accelerator_dram_address=C_DRAM_ADDR + ((M_chunk_idx + vector_idx) * N + N_chunk_idx) * bytes_per_element,
                                                            element_size=N_take_size)

                    self.accelerator_memory_to_scale_sram(accelerator_dram_address=SCALE_DRAM_ADDR + ((N_chunk_idx * K) // UE_VECTOR_SIZE) * bytes_per_element,
                                                        element_size=(N_take_size * K) // UE_VECTOR_SIZE)

                    if data_type == TYPE.IF4 or data_type == TYPE.TQ4:
                        dma_offset = N_chunk_idx * K // 2
                    elif data_type == TYPE.IF8:
                        dma_offset = N_chunk_idx * K
                    else:
                        assert False, f"data_type={data_type} is not supported"

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

    def _emit_pbi_scatter_per_token(self, *, read_base, read_stride_bytes,
                                    write_specs, sram_byte_addr, element_count,
                                    gpr_seq_len, template_seq_len):
        """Emit one PBI runtime loop that staged-copies one ``element_count``-row
        per outer iteration from ``read_base`` to each (base, stride) in
        ``write_specs``. The outer trip count is taken from GPR ``gpr_seq_len`` so
        the body executes exactly ``actual_seq_len`` times at runtime — making
        the captured bin truly seq_len-agnostic up to ``MAX_CONTEXT_SIZE``.

        Read side uses register-computed addresses (``reg_mul_imm`` + ``add_imm``
        + ``general_reg_src=TMP_REG``) — the same pattern used by the decoder
        bin. The write side uses gemma3-style ``pbi_init`` pointers + per-call
        DRAM-delta DMAs, which is the proven SRAM→DRAM PBI scatter shape.

        Per-iteration t-counter is a locally-allocated GPR that increments by 1
        at end-of-body via ``add_inc``. Released after the loop.
        """
        bpe = self.bytes_per_element
        bytes_per_call = element_count * bpe
        _, sram_words = self.sram_address_to_uram_address(sram_byte_addr)

        # Allocate write PBI pointers (one per destination stream).
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
        # DRAM→SRAM with runtime-computed source DRAM addr (delivered via TMP_REG).
        self.accelerator_memory_to_sram(
            accelerator_dram_address=0,
            sram_address=sram_byte_addr,
            element_size=element_count,
            general_reg_src=self.TMP_REG,
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
        dram_total_bytes = 0xffffffff - DRAM_START_ADDR + 1
        fill = b'\xff' * chunk_size_bytes
        offset = 0
        bar_width = 40
        print(f"Clearing DRAM [{hex(DRAM_START_ADDR)}..0xffffffff] ({dram_total_bytes / 1024**3:.2f} GB)")
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

        # C ue_memcpy REG_REWRITE path: memset then only w[0], w[1] (no UE payload in w[2:])
        if (inst_type == INSTRUCTION_REG_REWRITE and w[2] == 0 and w[3] == 0
                and w[4] == 0 and w[5] == 0 and w[6] == 0 and w[7] == 0):
            general_reg_src = _inst_desc_bits(w, 36, 39)
            result = (f"UE_MEMCPY_FROM_DRAM (REG_REWRITE, src_reg={general_reg_src})\n"
                      f"    inst_id: {transaction_id}")
            for line in result.split('\n'):
                print(f"        {line}")
            return

        # PBI pointer-row load / bump-only (inst_type 6). Same 256b payload layout as UE ops; do not decode as ISA.
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
            src_reg = _inst_desc_bits(w, 24, 27)  # pbi_general_reg_idx = descriptor[27:24]
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

        # UE engine ops: inst_type 0, or REG_REWRITE with full tail (ue_arithmetic_op / memcpy overwrites w[0..1] only)
        if inst_type in (0, INSTRUCTION_REG_REWRITE, INSTRUCTION_UE_PBI):
            mode_sel = _inst_desc_bits(w, 172, 175)
            reg_rewrite = inst_type == INSTRUCTION_REG_REWRITE
            general_reg_src = _inst_desc_bits(w, 36, 39) if reg_rewrite else 0

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
                if reg_rewrite:
                    result += f" (addr reset from general register {general_reg_src})"
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
                if reg_rewrite:
                    result += f" (addr from general register {general_reg_src})"
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
                if reg_rewrite:
                    result += f" (from general register {general_reg_src})"
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

        # ISA (non-UE): [79:32] micro-op fields
        isa_mode = _inst_desc_bits(w, 32, 35)
        src_reg_idx = _inst_desc_bits(w, 36, 39)
        dst_reg_idx = _inst_desc_bits(w, 40, 43)
        rst_reg_idx = _inst_desc_bits(w, 44, 47)
        immediate_value = _inst_desc_bits(w, 48, 79)

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
                JUMP_MODE_JNZ: "JNZ",
                JUMP_MODE_JZ: "JZ",
                JUMP_MODE_RELATIVE: "RELATIVE",
                JUMP_MODE_RELA_JNZ: "RELA_JNZ",
                JUMP_MODE_RELA_JZ: "RELA_JZ",
            }
            jump_mode_name = jump_mode_names.get(jump_mode, f"UNKNOWN({jump_mode})")
            result = f"ISA_JUMP ({jump_mode_name})"
            if jump_mode in (JUMP_MODE_RELATIVE, JUMP_MODE_RELA_JNZ, JUMP_MODE_RELA_JZ):
                result += f"\n    relative_back: {immediate_value} inst words"
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
                ALU_MODE_INC: "INC",
                ALU_MODE_DEC: "DEC",
                ALU_MODE_ADD_REG: "ADD_REG",
                ALU_MODE_ADD_IMM: "ADD_IMM",
                ALU_MODE_SET: "SET",
                ALU_MODE_MIN: "MIN",
                ALU_MODE_SUB: "SUB",
                ALU_MODE_MUL_IMM: "MUL_IMM",
            }
            mode_name = isa_mode_names.get(isa_mode, f"UNKNOWN({isa_mode})")
            result = f"ISA_REG_ALU ({mode_name})"
            result += f"\n    transaction_id: {transaction_id}"
            if isa_mode == ALU_MODE_SET:
                result += f"\n    dst_reg: {dst_reg_idx}, value: {_u32(immediate_value):#010X}"
            elif isa_mode in (ALU_MODE_INC, ALU_MODE_DEC):
                result += f"\n    reg: {dst_reg_idx}"
            elif isa_mode == ALU_MODE_ADD_IMM:
                result += f"\n    dst_reg: {dst_reg_idx}, src_reg: {src_reg_idx}, immediate: {_u32(immediate_value):#010X}"
            elif isa_mode == ALU_MODE_MUL_IMM:
                result += f"\n    dst_reg: {dst_reg_idx}, src_reg: {src_reg_idx}, immediate: {_u32(immediate_value):#010X}"
            elif isa_mode in (ALU_MODE_ADD_REG, ALU_MODE_MIN, ALU_MODE_SUB):
                result += f"\n    dst_reg: {dst_reg_idx}, src_reg: {src_reg_idx}, rst_reg: {rst_reg_idx}"
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
        Shared 256b instruction descriptor compiler for ISA micro-ops (JUMP / REG_ALU / REG_REWRITE / SEMAPHORE / FLAG).

        Header [15:0]: [7:0] instruction index from :attr:`_inst_id`; [11:8] inst_type; [15:12] reserved.
        ISA [79:32]: [35:32] isa_mode; [39:36] src; [43:40] dst; [47:44] rst; [79:48] immediate.

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

        # ISA [79:32]: [35:32] isa_mode; [39:36] src; [43:40] dst; [47:44] rst; [79:48] immediate
        w[1] = ((isa_mode & 0xF) |
                ((src_reg_idx & 0xF) << 4) |
                ((dst_reg_idx & 0xF) << 8) |
                ((rst_reg_idx & 0xF) << 12) |
                ((immediate_value & 0xFFFF) << 16))
        w[2] = (immediate_value >> 16) & 0xFFFF

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
        - bucket dispatchers (:meth:`flash_attention_core_pbi`,
          :meth:`decoder_group_attention_core_pbi`) before each bucket body's entry point.

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

    def generate_instruction_jump_rela(self, backward_instruction_words: int) -> None:
        """Unconditional relative backward jump (``JUMP_MODE_RELATIVE``); immediate is backward offset in instruction words."""
        self._generate_instruction_jump(backward_instruction_words, JUMP_MODE_RELATIVE, 0)

    def generate_instruction_jump_rela_jnz(self, backward_instruction_words: int, reg_id: int) -> None:
        """Relative backward jump if ``reg_id != 0`` (``JUMP_MODE_RELA_JNZ``)."""
        self._generate_instruction_jump(backward_instruction_words, JUMP_MODE_RELA_JNZ, reg_id)

    def generate_instruction_jump_rela_jz(self, backward_instruction_words: int, reg_id: int) -> None:
        """Relative backward jump if ``reg_id == 0`` (``JUMP_MODE_RELA_JZ``)."""
        self._generate_instruction_jump(backward_instruction_words, JUMP_MODE_RELA_JZ, reg_id)

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
        """Emit ALU_MODE_MUL_IMM: dst = (src[15:0] * imm[15:0]) mod 2**32, unsigned; upper bits ignored."""
        if dst_reg_idx == 0:
            print("ERROR: INSTRUCTION_REG_ALU overwriting reg_idx 0 (zero reg) not allowed")
            return
        assert immediate_value & 0xFFFF == immediate_value, "immediate_value must be less than 2**16"
        self.ue_isa_descriptor(
            INSTRUCTION_REG_ALU,
            immediate_value=immediate_value & 0xFFFFFFFF,
            isa_mode=ALU_MODE_MUL_IMM,
            src_reg_idx=src_reg_idx,
            dst_reg_idx=dst_reg_idx,
        )

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

    def overwrite_instruction_with_general_register(self, general_register: int) -> None:
        """
        Patch ``capture_buffer[capture_count - 1]`` so the DRAM address field is taken
        from the ISA regfile instead of ``w[1]``.

        Encoding (256b UE descriptor): inst_type = INSTRUCTION_REG_REWRITE in w[0][11:8];
        inst_src_reg_idx in w[1][7:4] (descriptor [39:36]). Preserves inst_id in w[0][7:0].
        Other descriptor words are unchanged.
        """
        if self.capture_buffer is None or len(self.capture_buffer) == 0:
            print("ERROR: overwrite_instruction_with_general_register() called but capture_buffer is empty!")
            return
        if self.capture_count == 0:
            print("ERROR: overwrite_instruction_with_general_register() called but capture_count is 0!")
            return
        if general_register <= 0 or general_register > 15:
            raise ValueError(f"general_register must be in [1, 15], got {general_register}")

        inst = self.capture_buffer[self.capture_count - 1]
        w = inst.words
        inst_id = w[0] & 0xFF

        # Overwrite word 0: preserve inst_id [7:0], set inst_type to INSTRUCTION_REG_REWRITE [11:8]
        w[0] = (inst_id & 0xFF) | ((INSTRUCTION_REG_REWRITE & 0xF) << 8)

        # Overwrite word 1: set inst_src_reg_idx [39:36] (bits 7:4 of w[1])
        w[1] = ((general_register & 0xF) << 4)

    def _patch_jump_immediate(self, capture_idx: int, target_word_addr: int) -> None:
        """
        Patch the 32-bit immediate of an already-captured ``INSTRUCTION_JUMP`` descriptor in
        :attr:`capture_buffer`, preserving all other fields (``inst_id``, ``inst_type``,
        ``isa_mode``, ``src/dst/rst`` register indices).

        Layout (see :meth:`ue_isa_descriptor`): immediate occupies bits [79:48] split as
        ``w[1][31:16]`` (low 16) and ``w[2][15:0]`` (high 16).

        Used by :meth:`flash_attention_core_pbi` bucketization to fill placeholder JZ /
        JMP targets after forward bucket-entry and end-label addresses become known.
        """
        if self.capture_buffer is None or capture_idx < 0 or capture_idx >= len(self.capture_buffer):
            raise IndexError(
                f"_patch_jump_immediate: capture_idx={capture_idx} out of range "
                f"(buffer size={len(self.capture_buffer) if self.capture_buffer else 0})"
            )
        target = int(target_word_addr) & 0xFFFFFFFF
        w = self.capture_buffer[capture_idx].words
        w[1] = (w[1] & 0x0000FFFF) | ((target & 0xFFFF) << 16)
        w[2] = (w[2] & 0xFFFF0000) | ((target >> 16) & 0xFFFF)

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
    ) -> None:
        """
        PBI init via :meth:`ue_op_descriptor` with ``inst_type=INSTRUCTION_PBI_SET`` and
        ``pbi_mode=PBI_MODE_INIT``. Sets all pointer-register fields from the literal arguments;
        ``inst_pointer_idx`` (``w[0][15:12]``) selects which pointer row to write.
        """
        if self.capture_buffer is None:
            print("ERROR: generate_instruction_pbi_init() called but capture_buffer is not initialized!")
            return
        if self.capture_count >= MAX_DECODER_INSTRUCTIONS:
            print(f"ERROR: generate_instruction_pbi_init() capture_count >= MAX ({MAX_DECODER_INSTRUCTIONS})!")
            return

        self.ue_op_descriptor(
            inst_type=INSTRUCTION_PBI_SET,
            inst_pointer_idx=inst_pointer_idx,
            pbi_mode=PBI_MODE_INIT,
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

        When ``general_reg_src`` is ``None``: ``pbi_mode=PBI_MODE_INC`` (plain increment).
        When ``general_reg_src`` is provided: ``pbi_mode=PBI_MODE_REG`` — RTL first accumulates
        the instruction delta (INC step) then overrides the field selected by ``pbi_field_select``
        with the value in GPR ``general_reg_src``.
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

    def start_execute_from_dram(self, instruction_addr: int = DRAM_INSTRUCTION_ADDR):
        """
        Start executing instructions from DRAM

        This function configures the hardware to execute instructions stored in DRAM.
        It sets the instruction control register to enable execution and points to
        the DRAM address where instructions are stored.

        Args:
            instruction_addr: DRAM address where instructions are stored
                            (default: DRAM_INSTRUCTION_ADDR)
        """
        self.write_reg32(UE_INSTRUCTION_ADDR, ue_35bit_addr_shifter(instruction_addr))

    def program_execute(self, program_start_addr: int = DRAM_INSTRUCTION_ADDR, timeout: float = 50.0, flops: float = None) -> None:
        """Execute compiled program from DRAM instruction memory.
        """
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
        latency = self.read_reg32(UE_LATENCY_COUNT_ADDR)
        instruction_count = self.read_reg32(UE_INSTRUCTION_CTL_ADDR)
        print(f"Latency: {latency * self._clock_period_ns / 1e3:.3f} us, Instruction count: {instruction_count}")
        print(f"Latency in cycles: {latency}")
        return latency, instruction_count

    def report_latency_in_us(self):
        """
        Report latency
        """
        return self.read_reg32(UE_LATENCY_COUNT_ADDR) * self._clock_period_ns / 1e3

    def report_flop_rate_gflops(self, num_flops: int):
        """
        Report flop rate and gflops ratio of peak throughput
        """
        gflops_ratio = num_flops / 1.28 / self.read_reg32(UE_LATENCY_COUNT_ADDR)
        return num_flops / (self.read_reg32(UE_LATENCY_COUNT_ADDR) * self._clock_period_ns), gflops_ratio

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
