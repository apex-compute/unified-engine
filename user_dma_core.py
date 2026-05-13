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
from typing import Optional, Tuple, Callable
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
# DMA device paths (can be overridden by command-line argument)
DMA_DEVICE_H2C = "/dev/xdma0_h2c_0"
DMA_DEVICE_C2H = "/dev/xdma0_c2h_0"
DMA_DEVICE_USER = "/dev/xdma0_user"  # AXI-Lite user interface for register access

def set_dma_device(device_name: str):
    """Set DMA device paths based on device name (e.g., 'xdma0' -> '/dev/xdma0_*')"""
    global DMA_DEVICE_H2C, DMA_DEVICE_C2H, DMA_DEVICE_USER
    DMA_DEVICE_H2C = f"/dev/{device_name}_h2c_0"
    DMA_DEVICE_C2H = f"/dev/{device_name}_c2h_0"
    DMA_DEVICE_USER = f"/dev/{device_name}_user"

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

# Pipeline component latencies from bf19_mult.vhdl, bf19_add.vhdl, custom_exp.vhdl, adder_tree.vhdl
UE_PIPELINE_BF19_MULT = 2
UE_PIPELINE_BF19_ADD = 3
UE_PIPELINE_CUSTOM_EXP = 5
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
UE_LALU_PIPELINE_FPDIV = 3  # fpdiv pipeline depth (from fpdiv.vhdl line 561)
UE_LALU_PIPELINE_FPSQRT = 3  # fpsqrt pipeline depth (from fpsqrt.vhdl line 9)
UE_LALU_PIPELINE_FACT = 8  # sample_1_plus_exp_bx pipeline depth (from sample_1_plus_exp_bx.vhdl line 9039)

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
INSTRUCTION_ADD = 2
INSTRUCTION_REG_REWRITE = 3
INSTRUCTION_FLAG = 4
INSTRUCTION_UE_PBI = 5
INSTRUCTION_PBI_SET = 6
INSTRUCTION_SWI = 8
INSTRUCTION_HALT = 9

# PBI mode constants (match queue_state_module.sv)
PBI_MODE_INIT = 0
PBI_MODE_INC = 1

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

# ADD mode constants
INST_ADD_INC = 0  # Increment destination register
INST_ADD_DEC = 1  # Decrement destination register
INST_ADD_REG = 2  # Add two source registers
INST_ADD_IMM = 3  # Add immediate to register
INST_ADD_SET = 4  # Set register to immediate value

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
                 clock_period_ns: float = None):
        self.device = device
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

        Args:
            size_bytes: Number of bytes to allocate
        """
        self._next_program_dram_addr = self._align_up(self._next_program_dram_addr, align_bytes)
        program_dram_addr = self._next_program_dram_addr
        self._next_program_dram_addr += size_bytes
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

    def init_unified_engine(self):
        # Test user device register access first
        print(f"{DMA_DEVICE_USER} register access...")
        hw_version = self.user_read_reg32(UE_FPGA_VERSION_ADDR)
        print(f"HW version via user device: 0x{hw_version & 0xFFFFFFFF:08x}")
        assert hw_version == 0x24618306, f"HW version mismatch: got 0x{hw_version & 0xFFFFFFFF:08x}, expected 0x24618306. Please update FPGA with commit update_2461830.bin using update_flash.py (public release v1.2)"

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
            ((UE_QINPUT_DELAY & 0x1F) << 21) +
            (UE_LATENCY_QUANTIZATION << 16) +
            (UE_LATENCY_QSCALE << 12) +
            (UE_LALU_LATENCY_ACT << 8) +
            (UE_LALU_LATENCY_RMS << 4) +
            (UE_LALU_LATENCY_SOFTMAX << 0)
        )
        self.write_reg32(UE_LALU_DELAY_ADDR, ue_lalu_delay)
        self.dram_inst_running(True)
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
        inst_pointer_idx: int = 0,
        pbi_mode: int = PBI_MODE_INIT,
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
        general_reg_src: int = 0,
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
        For ``INSTRUCTION_PBI_SET``, ``inst_pointer_idx`` fills ``w[0][15:12]`` and ``pbi_mode`` fills
        ``w[0][19:16]`` (no ``lalu_a`` in ``w[0][31:20]``).

        reg rewrite:
        - if ``general_reg_src`` is set, ``inst_type`` = ``INSTRUCTION_REG_REWRITE`` and ``w[1]`` carries ``src_reg_idx``.
        """

        uram_start = int((mode != UE_MODE.DOT_PRODUCT) and
                        (mode != UE_MODE.URAM_DRAM_WRITEBACK) and
                        (mode != UE_MODE.DEQUANTIZE) and
                        (mode != UE_MODE.MEMCPY_FROM_DRAM))
        uram_bram_wb_start = int(mode == UE_MODE.URAM_DRAM_WRITEBACK)
        dma_start = int((mode == UE_MODE.DOT_PRODUCT) or (mode == UE_MODE.DEQUANTIZE))
        uram_length_z = dma_length >> 6  # in 64 element units

        wb_padding_select = 1  # TODO: make this a parameter

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

            # [7:0] inst_id; [11:8] inst_type; [15:12] inst_ptr; [19:16] pbi_mode for PBI_SET; [31:16]/[31:20] lalu_a otherwise
            w[0] = ((tid & 0xFF) |
                    ((inst_type & 0xF) << 8) |
                    ((inst_pointer_idx & 0xF) << 12))
            # PBI_SET: RTL inst_pointer_mode = inst_descriptor[19:16] (queue_state_module.sv).
            # inst_bf16_lalu_a = [31:16] shares that nibble; keep [31:20] zero per readme_ISA_PBI.md.
            if int(inst_type) == int(INSTRUCTION_PBI_SET):
                w[0] |= (int(pbi_mode) & 0xF) << 16
                w[0] &= 0x000F_FFFF
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

            if general_reg_src != 0:
                w[0] = (tid & 0xFF) | ((INSTRUCTION_REG_REWRITE & 0xF) << 8)
                w[1] = ((general_reg_src & 0xF) << 4)

            self.capture_buffer.append(inst)
            self.capture_count += 1
            self._inst_id = tid + 1

    def ue_memcpy_from_dram(self, dram_src_addr: int, memcpy_length_bytes: int,
                            memcpy_type: int, uram_dst_addr: int,
                            uram_type: int,
                            stride_bytes_per_chunk: int = 0,
                            stride_jump_bytes: int = 0,
                            general_reg_src: int = 0,
                            inst_pointer_idx: int = 0
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
            general_reg_src: REG_REWRITE source register index when nonzero
            fmax_context_addr: 6-bit fmx_context for descriptor bit [223] when nonzero
            inst_pointer_idx: when nonzero, emit PBI-style memcpy (pointer-backed registers are incremented by the immediate value specified in the instruction).
        """
        is_pointer_mode = inst_pointer_idx != 0
        self.ue_op_descriptor(
            inst_type=INSTRUCTION_UE_PBI if is_pointer_mode else INSTRUCTION_UE_OP,
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
            dma_start_addr=dram_src_addr,
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
                         general_reg_src: int = 0,
                         inst_pointer_idx: int = 0
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
        is_pointer_mode = inst_pointer_idx != 0
        self.ue_op_descriptor(
            inst_type= INSTRUCTION_UE_PBI if is_pointer_mode else INSTRUCTION_UE_OP,
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
            dma_start_addr=dram_dst_addr,
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
                   general_reg_src: int = 0, fmax_context_addr: int = 0,
                   inst_pointer_idx: int = 0
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
        is_pointer_mode = inst_pointer_idx != 0
        self.ue_op_descriptor(
            inst_type= INSTRUCTION_UE_PBI if is_pointer_mode else INSTRUCTION_UE_OP,
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
            general_reg_src=general_reg_src,
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
        inst_pointer_idx: int = 0,
        memcpy_length_bytes: Optional[int] = None,
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
        inst_pointer_idx: int = 0,
        memcpy_length_bytes: Optional[int] = None,
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
        AXI_DATA_WIDTH = 256
        uram_type, uram_start_addr = self.sram_address_to_uram_address(sram_address)
        assert stride_bytes_per_chunk % (AXI_DATA_WIDTH // 8) == 0, "stride_bytes_per_chunk must be a multiple of AXI_DATA_WIDTH, TODO: support more non-aligned writes"
        nbytes = element_size * 2 if memcpy_length_bytes is None else memcpy_length_bytes
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

    def accelerator_memory_to_bias_sram(
        self,
        accelerator_dram_address: int,
        element_size: int,
        inst_pointer_idx: int = 0,
    ) -> None:
        """Copy from accelerator DRAM to bias BRAM. element_size is in elements (bf16 = 2 bytes per element).

        ``inst_pointer_idx``: when nonzero, emit PBI-style memcpy (pointer-backed increment).
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
        inst_pointer_idx: int = 0,
    ) -> Optional[torch.Tensor]:
        """
        Start queue for element-wise op: ELTWISE_ADD, ELTWISE_MUL, or ELTWISE_SUB.
        Wraps :meth:`ue_arithmetic_op` with standard eltwise parameters.

        ``inst_pointer_idx`` is forwarded when nonzero (PBI / UE_OP_REG path); default ``0`` keeps
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
        inst_pointer_idx: int = 0,
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


    def rms_norm_core_dram_pbi(self, M: int, N: int, A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, GAMMA_DRAM_ADDR: int) -> None:
        """
        Core RMS norm: normalizes vector x -> x / rms(x).
        Args:
            M: number of rows in the input matrix
            N: number of columns in the input matrix
            A_DRAM_ADDR: DRAM address of input matrix
            OUTPUT_DRAM_ADDR: DRAM address for normalized output
            GAMMA_DRAM_ADDR: DRAM address for gamma
            N: number of elements
        """
        vector_sram_addr = 0x00000
        gamma_sram_addr = 0x80000

        self.accelerator_memory_to_sram(accelerator_dram_address=GAMMA_DRAM_ADDR,
                                        sram_address=gamma_sram_addr,
                                        element_size=N)

        chunk_size = min(URAM_NEAR_FULL_ELEMENTS // N, M)
        print(f"Chunk size: {chunk_size} for M={M}, N={N}")
        assert chunk_size >= 1 and chunk_size <= M, f"chunk_size={chunk_size} must be greater than 0 and less than M={M}"
        assert self.is_capture_on, "rms_norm_core_dram_pbi() requires active capture"
        assert N % UE_VECTOR_SIZE == 0, f"rms_norm_core_dram_pbi() requires N to be a multiple of UE_VECTOR_SIZE, got N={N}"
        bytes_per_element = 2
        chunk_bytes = chunk_size * N * bytes_per_element
        num_full_m_tiles = M // chunk_size
        m_remainder = M % chunk_size
        print(f"M_chunk: {chunk_size}, M_remainder: {m_remainder}, num_full_m_tiles: {num_full_m_tiles}")
        row_size = (N + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
        vector_uram_type, vector_uram_start_addr = self.sram_address_to_uram_address(vector_sram_addr)
        gamma_uram_type, gamma_uram_start_addr = self.sram_address_to_uram_address(gamma_sram_addr)
        assert vector_uram_type == URAM_SECTION.URAM_A, f"vector_sram_addr must be in URAM_A, got {hex(vector_sram_addr)}"
        assert gamma_uram_type == URAM_SECTION.URAM_B, f"gamma_sram_addr must be in URAM_B, got {hex(gamma_sram_addr)}"

        row_stride = row_size
        full_chunk_load_ptr = self.alloc_inst_ptr()
        full_chunk_store_ptr = self.alloc_inst_ptr()
        rms_ptr = self.alloc_inst_ptr()
        broadcast_ptr = self.alloc_inst_ptr()
        gamma_mul_ptr = self.alloc_inst_ptr()

        def emit_m_tile(m_take: int, is_full_tile: bool) -> None:
            if is_full_tile:
                self.generate_instruction_pbi_init(
                    dram_shared_addr=A_DRAM_ADDR,
                    dma_length=chunk_bytes,
                    output_size=0,
                    uram_length=0,
                    uram_a_start_addr=0,
                    uram_b_start_addr=0,
                    uram_wb_addr=0,
                    uram_dst_addr=vector_uram_start_addr,
                    fmax_context_addr=0,
                    inst_pointer_idx=full_chunk_load_ptr,
                )
                self.generate_instruction_pbi_init(
                    dram_shared_addr=OUTPUT_DRAM_ADDR,
                    dma_length=chunk_bytes,
                    output_size=0,
                    uram_length=0,
                    uram_a_start_addr=vector_uram_start_addr,
                    uram_b_start_addr=vector_uram_start_addr,
                    uram_wb_addr=0,
                    uram_dst_addr=0,
                    fmax_context_addr=0,
                    inst_pointer_idx=full_chunk_store_ptr,
                )

                # Absolute jump keeps the outer ISA loop anchored at the current I-cache window.
                program_dram_start_addr = self.get_program_dram_addr()
                cur_inst_count = self.capture_count
                jump_target_word_addr = ue_35bit_addr_shifter(
                    program_dram_start_addr + (cur_inst_count + 1) * INSTRUCTION_SIZE_BYTES
                )
                self.generate_instruction_jump_abs(jump_target_word_addr)
                self.loop_start(num_full_m_tiles)

                self.accelerator_memory_to_sram(
                    accelerator_dram_address=chunk_bytes,
                    sram_address=vector_sram_addr,
                    element_size=0,
                    inst_pointer_idx=full_chunk_load_ptr,
                )
            else:
                residual_a_addr = A_DRAM_ADDR + num_full_m_tiles * chunk_bytes
                self.accelerator_memory_to_sram(
                    accelerator_dram_address=residual_a_addr,
                    sram_address=vector_sram_addr,
                    element_size=m_take * N,
                )

            self.generate_instruction_pbi_init(
                dram_shared_addr=0,
                dma_length=0,
                output_size=0,
                uram_length=row_size,
                uram_a_start_addr=vector_uram_start_addr,
                uram_b_start_addr=0,
                uram_wb_addr=0,
                uram_dst_addr=0,
                fmax_context_addr=0,
                inst_pointer_idx=rms_ptr,
            )
            self.generate_instruction_pbi_init(
                dram_shared_addr=0,
                dma_length=0,
                output_size=0,
                uram_length=row_size,
                uram_a_start_addr=vector_uram_start_addr,
                uram_b_start_addr=0,
                uram_wb_addr=vector_uram_start_addr,
                uram_dst_addr=0,
                fmax_context_addr=0,
                inst_pointer_idx=broadcast_ptr,
            )
            self.generate_instruction_pbi_init(
                dram_shared_addr=0,
                dma_length=0,
                output_size=0,
                uram_length=row_size,
                uram_a_start_addr=vector_uram_start_addr,
                uram_b_start_addr=gamma_uram_start_addr,
                uram_wb_addr=vector_uram_start_addr,
                uram_dst_addr=0,
                fmax_context_addr=0,
                inst_pointer_idx=gamma_mul_ptr,
            )

            self.loop_start(m_take)
            self.ue_arithmetic_op(
                broadcast_mode=0,
                max_clear_en=0,
                stride_z=1,
                lalu_a=0,
                lalu_b=0,
                lalu_mode=LALU_MODE.MODE_RSQRT.value,
                scalar=self.float_to_bf19(float(math.sqrt(N))),
                uram_section=URAM_SECTION.URAM_A.value,
                uram_dst_addr=0,
                uram_wb_addr=0,
                uram_write_src=URAM_WRITE_SRC.URAM_WB_DISABLE.value,
                mode=UE_MODE.RMS,
                data_type=0,
                uram_a_start_addr=row_stride,
                uram_b_start_addr=0,
                uram_length=0,
                dma_start_addr=0,
                dma_length=0,
                output_size=0,
                inst_pointer_idx=rms_ptr,
            )
            self.ue_arithmetic_op(
                broadcast_mode=BROADCAST_MODE.LALU_RESULT.value,
                max_clear_en=0,
                stride_z=1,
                lalu_a=0,
                lalu_b=0,
                lalu_mode=LALU_MODE.BYPASS.value,
                scalar=self.float_to_bf19(1.0),
                uram_section=URAM_SECTION.URAM_A.value,
                uram_dst_addr=0,
                uram_wb_addr=row_stride,
                uram_write_src=URAM_WRITE_SRC.URAM_WRITE_BACK.value,
                mode=UE_MODE.MUL_BROADCAST,
                data_type=0,
                uram_a_start_addr=row_stride,
                uram_b_start_addr=0,
                uram_length=0,
                dma_start_addr=0,
                dma_length=0,
                output_size=0,
                inst_pointer_idx=broadcast_ptr,
            )
            self.ue_arithmetic_op(
                broadcast_mode=0,
                max_clear_en=0,
                stride_z=1,
                lalu_a=0,
                lalu_b=0,
                lalu_mode=LALU_MODE.BYPASS.value,
                scalar=0,
                uram_section=URAM_SECTION.URAM_A.value,
                uram_dst_addr=0,
                uram_wb_addr=row_stride,
                uram_write_src=URAM_WRITE_SRC.URAM_WRITE_BACK.value,
                mode=UE_MODE.ELTWISE_MUL,
                data_type=0,
                uram_a_start_addr=row_stride,
                uram_b_start_addr=0,
                uram_length=0,
                dma_start_addr=0,
                dma_length=0,
                output_size=0,
                inst_pointer_idx=gamma_mul_ptr,
            )
            self.loop_end()

            if is_full_tile:
                self.sram_to_accelerator_memory(
                    sram_address=vector_sram_addr,
                    accelerator_dram_address=chunk_bytes,
                    element_size=0,
                    inst_pointer_idx=full_chunk_store_ptr,
                )
                outer_loop_size = self.loop_end()
                print(f"Loop body size: {outer_loop_size}")
                assert outer_loop_size <= 256, (
                    f"Outer loop body size {outer_loop_size} is greater than i-cache size of 256 instructions"
                )
            else:
                residual_output_addr = OUTPUT_DRAM_ADDR + num_full_m_tiles * chunk_bytes
                self.sram_to_accelerator_memory(
                    sram_address=vector_sram_addr,
                    accelerator_dram_address=residual_output_addr,
                    element_size=m_take * N,
                )

        emit_m_tile(chunk_size, is_full_tile=True)
        if m_remainder:
            emit_m_tile(m_remainder, is_full_tile=False)

        self.release_inst_ptr(gamma_mul_ptr)
        self.release_inst_ptr(broadcast_ptr)
        self.release_inst_ptr(rms_ptr)
        self.release_inst_ptr(full_chunk_store_ptr)
        self.release_inst_ptr(full_chunk_load_ptr)

        # Total Theoretical FLOPS: 3 * M * N + M * N (gamma)
        total_flops = 3 * M * N + M * N # exp(x), sum(x), broadcast_mul, eltwise_mul(gamma)
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
                           use_pbi: bool = False) -> None:
        """RMS norm DRAM entrypoint; switch between PBI and legacy core paths."""
        if use_pbi:
            return self.rms_norm_core_dram_pbi(
                M=M,
                N=N,
                A_DRAM_ADDR=A_DRAM_ADDR,
                OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR,
                GAMMA_DRAM_ADDR=GAMMA_DRAM_ADDR,
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

    def rope_core(self, N: int,
                  x_sram_addr: int,
                  cos_sram_addr: int,
                  sin_sram_addr: int,
                  output_sram_addr: int,
                  a_sram_addr: int,
                  bc_sram_addr: int) -> None:
        """
        Core HF-style RoPE on SRAM vectors.

        Computes: output = x * cos + cat(x[hi]*sin[lo], x[lo]*sin[hi])

        sin must have its first half pre-negated for the standard HF RoPE formula:
          rotate_half(x) * sin = cat(-x[hi], x[lo]) * sin

        SRAM constraints:
          - x, output, a must be in URAM_A
          - cos, sin, bc must be in URAM_B
          - bc must have room for N elements (b=N/2 then c=N/2, contiguous)

        Args:
            N: vector length (must be multiple of UE_VECTOR_SIZE and even)
            x_sram_addr: input vector x in URAM_A (N elements)
            cos_sram_addr: cosine values in URAM_B (N elements)
            sin_sram_addr: sine values in URAM_B (N elements, first half negated)
            output_sram_addr: output in URAM_A (N elements, may alias x_sram_addr)
            a_sram_addr: scratch in URAM_A for intermediate a = x*cos (N elements)
            bc_sram_addr: scratch in URAM_B for b and c (N elements total)
        """
        half_N = N // 2
        bytes_per_element = 2

        # Step 1: a = x * cos
        self.eltwise_mul_core(x_sram_addr, cos_sram_addr, a_sram_addr, N)

        # Step 2: b = x[hi] * sin[lo]  →  stored at bc_sram_addr (first half)
        self.eltwise_mul_core(x_sram_addr + half_N * bytes_per_element,
                              sin_sram_addr,
                              bc_sram_addr, half_N)

        # Step 3: c = x[lo] * sin[hi]  →  stored at bc_sram_addr + half (second half)
        self.eltwise_mul_core(x_sram_addr,
                              sin_sram_addr + half_N * bytes_per_element,
                              bc_sram_addr + half_N * bytes_per_element, half_N)

        # Step 4: output = a + cat(b, c)   (b and c are already contiguous at bc_sram_addr)
        self.eltwise_add_core(a_sram_addr, bc_sram_addr, output_sram_addr, N)

    def rope_core_dram(self, M: int, N: int,
                       X_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                       COS_DRAM_ADDR: int, SIN_DRAM_ADDR: int) -> int:
        """
        DRAM-level HF-style RoPE over M rows of N elements.

        cos and sin are loaded once into URAM_B and reused for every row.
        sin must have its first half pre-negated.

        URAM layout:
          URAM_A: [input rows chunk] [scratch a (N)]
          URAM_B: [cos (N)] [sin (N)] [scratch bc (N)]

        Args:
            M: number of rows
            N: vector length per row (must be multiple of UE_VECTOR_SIZE and even)
            X_DRAM_ADDR: DRAM address of input matrix (M x N, bf16)
            OUTPUT_DRAM_ADDR: DRAM address for output matrix (M x N, bf16)
            COS_DRAM_ADDR: DRAM address for cosine values (N elements, bf16)
            SIN_DRAM_ADDR: DRAM address for sine values (N elements, bf16, first half negated)

        Returns:
            total_flops: theoretical FLOP count (4 * M * N)
        """
        bytes_per_element = 2
        assert N % UE_VECTOR_SIZE == 0, f"N={N} must be a multiple of UE_VECTOR_SIZE={UE_VECTOR_SIZE}"
        assert N % 2 == 0, f"N={N} must be even for HF RoPE"

        # URAM_B: cos (N) + sin (N) + bc scratch (N) = 3N elements
        cos_sram_addr = 0x80000
        sin_sram_addr = cos_sram_addr + N * bytes_per_element
        bc_sram_addr = sin_sram_addr + N * bytes_per_element

        self.accelerator_memory_to_sram(accelerator_dram_address=COS_DRAM_ADDR,
                                        sram_address=cos_sram_addr,
                                        element_size=N)
        self.accelerator_memory_to_sram(accelerator_dram_address=SIN_DRAM_ADDR,
                                        sram_address=sin_sram_addr,
                                        element_size=N)

        # URAM_A: chunk of input rows + scratch a (N)
        # Reserve N elements at the end of URAM_A for scratch a
        usable_uram_a_elements = URAM_NEAR_FULL_ELEMENTS - N
        chunk_size = min(M, usable_uram_a_elements // N)
        assert chunk_size >= 1, f"N={N} too large: need at least 2*N elements in URAM_A"

        x_sram_base = 0x00000
        a_sram_addr = x_sram_base + chunk_size * N * bytes_per_element

        print(f"rope_core_dram: M={M}, N={N}, chunk_size={chunk_size}")

        for i, m_take in self.chunk_ranges(M, chunk_size):
            self.accelerator_memory_to_sram(accelerator_dram_address=X_DRAM_ADDR + i * N * bytes_per_element,
                                            sram_address=x_sram_base,
                                            element_size=m_take * N)

            for j in range(m_take):
                row_sram_addr = x_sram_base + j * N * bytes_per_element
                self.rope_core(N, row_sram_addr, cos_sram_addr, sin_sram_addr,
                               row_sram_addr, a_sram_addr, bc_sram_addr)

            self.sram_to_accelerator_memory(sram_address=x_sram_base,
                                            accelerator_dram_address=OUTPUT_DRAM_ADDR + i * N * bytes_per_element,
                                            element_size=m_take * N)

        total_flops = 3 * M * N
        return total_flops

    def rope_hf_core_dram(self, M: int, N: int, input_dram_addr: int, output_dram_addr: int, cos_dram_addr: int, sin_dram_addr: int, use_pbi: bool = False, rope_size_reg: int = None, output_addr_inc_reg: int = None, tmp_reg: int = None) -> int:
        if use_pbi:
            return self.rope_hf_core_dram_pbi(M, N, input_dram_addr, output_dram_addr, cos_dram_addr, sin_dram_addr)
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

    def rope_hf_core_dram_pbi(self, M: int, N: int, input_dram_addr: int, output_dram_addr: int, cos_dram_addr: int, sin_dram_addr: int) -> int:
        """PBI-backed HF RoPE over M rows. Caller must have start_capture() before and stop_capture() after."""
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
        self.loop_start(M)
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

    def rope_hf_core_dram_gqa(self, M: int, group_size: int, N: int, input_dram_addr: int, output_dram_addr: int, cos_dram_addr: int, sin_dram_addr: int, use_pbi: bool = False) -> int:
        if use_pbi:
            return self.rope_hf_core_dram_gqa_pbi(M, group_size, N, input_dram_addr, output_dram_addr, cos_dram_addr, sin_dram_addr)
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

    def rope_hf_core_dram_gqa_pbi(self, M: int, group_size: int, N: int, input_dram_addr: int, output_dram_addr: int, cos_dram_addr: int, sin_dram_addr: int) -> int:
        """PBI-backed grouped-query RoPE. Q rows are [M, group_size, N], RoPE rows are [M, N]."""
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
        self.loop_start(M)
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

    def matmat_mul_core_pbi(self, M: int, K: int, N: int, A_DRAM_ADDR: int, B_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, softmax_enable: bool = False, C_DRAM_ADDR: int = None, bias_mode: str = "broadcast_N",
                             is_B_quantized: bool = False, data_type: TYPE = None, SCALE_DRAM_ADDR: int = None, gelu_enable: bool = False, silu_enable: bool = False, sigmoid_enable: bool = False,
                             clamp_enable: bool = False, log_enable: bool = False,
                             debug_fmax: bool = False, ZERO_DRAM_ADDR: int = None, FMAX_DRAM_ADDR: int = None,
                             write_back_disable: bool = False) -> None:
        """
        Matmul capture path structured like a future pointer-backed-increment (PBI) ISA loop nest.

        - **M** is split into ``M // M_chunk`` full row tiles (PBI-sim: fixed loop count + A DRAM /
          output pointer bump by ``M_chunk * K`` / row stride) plus an optional **residual** tile
          ``M % M_chunk`` outside that loop (same order as :meth:`chunk_ranges`).
        - **N** is split into ``N // N_chunk`` full column tiles plus **residual** ``N % N_chunk``
          outside the fixed-count loop.
        - **Rows** within an M tile: always ``m_take`` matvecs (``m_take == M_chunk`` for full tiles,
          smaller on the last M slice).

        Today this only reorders Python control flow; emit sequence matches the original
        :meth:`matmat_mul_core`.
        """
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

        # M dimension: fixed-count PBI loop for full tiles; residual M % M_chunk separate.
        num_full_m_tiles = M // M_chunk
        m_remainder = M % M_chunk
        print(f"M_chunk: {M_chunk}, M_remainder: {m_remainder}, num_full_m_tiles: {num_full_m_tiles}")
        # N dimension: fixed-count PBI loop for full tiles; residual N % N_chunk separate.
        num_full_n_tiles = N // N_chunk
        n_remainder = N % N_chunk
        print(f"N_chunk: {N_chunk}, N_remainder: {n_remainder}, num_full_n_tiles: {num_full_n_tiles}")
        assert N_chunk_aligned is None or n_remainder == 0, f"Illegal case with N_chunk_aligned={N_chunk_aligned} and n_remainder={n_remainder}"

        if softmax_enable:
            M_softmax_chunk = min(URAM_NEAR_FULL_ELEMENTS // N, M_chunk)
            num_softmax_m_tiles = M_chunk // M_softmax_chunk
            m_softmax_remainder = M_chunk % M_softmax_chunk
            print(f"M_softmax_chunk: {M_softmax_chunk}, num_softmax_m_tiles: {num_softmax_m_tiles}, m_softmax_remainder: {m_softmax_remainder}")
            if m_remainder:
                M_softmax_chunk_remainder = min(URAM_NEAR_FULL_ELEMENTS // N, m_remainder)
                num_softmax_m_tiles_remainder = m_remainder // M_softmax_chunk_remainder
                m_softmax_remainder_remainder = m_remainder % M_softmax_chunk_remainder
                print(f"M_softmax_chunk_remainder: {M_softmax_chunk_remainder}, num_softmax_m_tiles_remainder: {num_softmax_m_tiles_remainder}, m_softmax_remainder_remainder: {m_softmax_remainder_remainder}")

        pointer_idx = [self.alloc_inst_ptr() for _ in range(13)]

        def emit_n_tile_pbi(is_full_tile: bool, n_take: int, m_take: int) -> None:
            """
            PBI N tile: ``is_full_tile`` is True for the primary ``N_chunk`` column strip (``j == 0``),
            False for the residual ``n_remainder`` strip. URAM dot-product layout follows ``m_take``.
            """
            ue_b = UE_VECTOR_SIZE * bytes_per_element
            k_row_b = K * bytes_per_element
            output_sram_wb_addr = m_take * K * bytes_per_element
            row_full_b = (N_chunk if N_chunk_aligned is None else N_chunk_aligned) * bytes_per_element
            row_res_b = (n_remainder * bytes_per_element if N_chunk_aligned is None else row_full_b)
            pbi_inc_dot_product_uram_start_addr = k_row_b // ue_b
            pbi_inc_dot_product_uram_wb_addr_full = row_full_b // ue_b
            diff_res = n_remainder and row_res_b != row_full_b
            pbi_inc_dot_product_uram_wb_addr_residual = row_res_b // ue_b if diff_res else pbi_inc_dot_product_uram_wb_addr_full

            pbi_base_dot_product_uram_wb_addr = output_sram_wb_addr // ue_b
            pbi_inc_dot_product_uram_wb_addr = pbi_inc_dot_product_uram_wb_addr_full if is_full_tile else pbi_inc_dot_product_uram_wb_addr_residual

            # pbi init for loop over N dimension (N_chunk)
            if is_B_quantized:
                if data_type == TYPE.IF4:
                    offset = ((K * N_chunk) >> 1) if is_full_tile else ((K * num_full_n_tiles * N_chunk) >> 1)
                elif data_type == TYPE.IF8:
                    offset = K * N_chunk if is_full_tile else K * num_full_n_tiles * N_chunk

                if is_full_tile:
                    self.generate_instruction_pbi_init(
                        dram_shared_addr=SCALE_DRAM_ADDR,
                        dma_length=((n_take * K + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE) * bytes_per_element,
                        output_size=0,
                        uram_length=0,
                        uram_a_start_addr=0,
                        uram_b_start_addr=0,
                        uram_wb_addr=0,
                        uram_dst_addr=0,
                        fmax_context_addr=0,
                        inst_pointer_idx=pointer_idx[2],
                    )
                    self.generate_instruction_pbi_init(
                            dram_shared_addr=B_DRAM_ADDR,
                            dma_length=n_take * K if data_type == TYPE.IF8 else n_take * K >> 1,
                            output_size=(n_take * K + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE,
                            uram_length=(n_take * K + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE,
                            uram_a_start_addr=0,
                            uram_b_start_addr=0,
                            uram_wb_addr=0x80000 >> 7,
                            uram_dst_addr=0,
                            fmax_context_addr=0,
                            inst_pointer_idx=pointer_idx[3],
                        )
            else:
                if is_full_tile:
                    self.generate_instruction_pbi_init(
                        dram_shared_addr=B_DRAM_ADDR,
                        dma_length=n_take * K * bytes_per_element,
                        output_size=0,
                        uram_length=0,
                        uram_a_start_addr=0,
                        uram_b_start_addr=(0x80000 >> 7) & 0xFFF,
                        uram_wb_addr=0,
                        uram_dst_addr=0,
                        fmax_context_addr=0,
                        inst_pointer_idx=pointer_idx[2],
                    )
            if bias_enable and bias_mode == "broadcast_N" and is_full_tile:
                # pbi_base settings for memcpy_c bias
                self.generate_instruction_pbi_init(
                    dram_shared_addr=C_DRAM_ADDR,
                    dma_length=n_take * bytes_per_element,
                    output_size=0,
                    uram_length=0,
                    uram_a_start_addr=0,
                    uram_b_start_addr=0,
                    uram_wb_addr=0,
                    uram_dst_addr=0,
                    fmax_context_addr=0,
                    inst_pointer_idx=pointer_idx[4],
                )

            # pbi loop body over N dimension (N_chunk)
            if is_full_tile:
                self.loop_start(num_full_n_tiles)
            if is_B_quantized:
                # start_queue_for_bf16_dequantize_operation()
                self.ue_memcpy_from_dram(
                    (K // UE_VECTOR_SIZE) * bytes_per_element * N_chunk if is_full_tile else SCALE_DRAM_ADDR + (K // UE_VECTOR_SIZE) * bytes_per_element * num_full_n_tiles * N_chunk,
                    0 if is_full_tile else ((n_take * K + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE) * bytes_per_element,
                    MEMCPY_TYPE.BRAM.value, 0, 0,
                    inst_pointer_idx=pointer_idx[2] if is_full_tile else 0)

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
                    uram_wb_addr=0 if is_full_tile else 0x80000 >> 7,
                    uram_write_src=URAM_WRITE_SRC.URAM_WRITE_BACK.value,
                    mode=UE_MODE.DEQUANTIZE,
                    data_type=data_type.value,
                    uram_a_start_addr=0,
                    uram_b_start_addr=0,
                    uram_length=0,
                    dma_start_addr=offset if is_full_tile else B_DRAM_ADDR + offset,
                    dma_length=0 if is_full_tile else n_take * K if data_type == TYPE.IF8 else n_take * K >> 1,
                    output_size=0 if is_full_tile else (n_take * K + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE,
                    inst_pointer_idx=pointer_idx[3] if is_full_tile else 0,
                )
            else:
                self.ue_memcpy_from_dram(
                    K * bytes_per_element * N_chunk if is_full_tile else B_DRAM_ADDR + num_full_n_tiles * N_chunk * K * bytes_per_element,
                    0 if is_full_tile else n_take * K * bytes_per_element,
                    MEMCPY_TYPE.URAM.value,
                    0 if is_full_tile else (0x80000 >> 7) & 0xFFF,
                    URAM_SECTION.URAM_B,
                    inst_pointer_idx=pointer_idx[2] if is_full_tile else 0,
                )

            if bias_enable and bias_mode == "broadcast_N":
                # accelerator_memory_to_bias_sram()
                self.ue_memcpy_from_dram(
                    bytes_per_element * N_chunk if is_full_tile else C_DRAM_ADDR + num_full_n_tiles * N_chunk * bytes_per_element,
                    0 if is_full_tile else n_take * bytes_per_element,
                    MEMCPY_TYPE.BIAS_BRAM.value,
                    0,
                    0,
                    inst_pointer_idx=pointer_idx[4] if is_full_tile else 0,
                )

            # pbi init for loop over innermost dimension (m_take)
            self.generate_instruction_pbi_init(
                dram_shared_addr=0,
                dma_length=K * n_take,
                output_size=n_take,
                uram_length=K // UE_VECTOR_SIZE,
                uram_a_start_addr=0,
                uram_b_start_addr=0,
                uram_wb_addr=pbi_base_dot_product_uram_wb_addr,
                uram_dst_addr=0,
                fmax_context_addr=0,
                inst_pointer_idx=pointer_idx[1],
            )

            # pbi loop body over innermost dimension (m_take)
            self.loop_start(m_take)
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

            # For the level-2 loop, reset the dram addr pointer through pbi increment op to handle the nested loop; for N_remainder, level-2 loop is skipped and reset the dram_addr and dram_length pointers for the level-3 loop
            if bias_enable and bias_mode == "full_matrix":
                self.accelerator_memory_to_bias_sram(
                    accelerator_dram_address=(
                        (N_chunk - m_take * N) * bytes_per_element
                        if is_full_tile
                        else (n_remainder - N) * bytes_per_element
                    ),
                    element_size=0 if is_full_tile else N_chunk - n_remainder,
                    inst_pointer_idx=pointer_idx[0],
                )

            if not write_back_disable:
                # result memcpy to DRAM with stride
                if N_chunk_aligned is None:
                    self.ue_memcpy_to_dram(
                        memcpy_type=MEMCPY_TYPE.URAM.value,
                        uram_type=URAM_SECTION.URAM_A.value,
                        uram_src_addr=0,
                        dram_dst_addr=bytes_per_element * n_take,
                        memcpy_length_bytes=0,
                        stride_bytes_per_chunk=0,
                        stride_jump_bytes=N * bytes_per_element,
                        inst_pointer_idx=pointer_idx[5],
                    )
                else:
                    # pbi loop body over innermost dimension (m_take)
                    self.loop_start(m_take)
                    self.ue_memcpy_to_dram(
                        memcpy_type=MEMCPY_TYPE.URAM.value,
                        uram_type=URAM_SECTION.URAM_A.value,
                        uram_src_addr=1,
                        dram_dst_addr= N * bytes_per_element,
                        memcpy_length_bytes=0,
                        inst_pointer_idx=pointer_idx[5],
                    )
                    self.loop_end()

                    self.ue_memcpy_to_dram(
                        memcpy_type=MEMCPY_TYPE.URAM.value,
                        uram_type=URAM_SECTION.URAM_A.value,
                        uram_src_addr=-m_take,
                        dram_dst_addr=(n_take - m_take * N) * bytes_per_element,
                        memcpy_length_bytes=0,
                        inst_pointer_idx=pointer_idx[5],
                    )

            if is_full_tile:
                self.loop_end()

            if not write_back_disable:
                # reset pointers for the level-3 loop
                if N_chunk_aligned is None:
                    memcpy_length_bytes = 0 if is_full_tile and n_remainder == 0 else m_take * (n_remainder - N_chunk) * 2 if is_full_tile else m_take * (N_chunk - n_remainder) * 2
                    stride_bytes_per_chunk = 0 if is_full_tile and n_remainder == 0 else (n_remainder - N_chunk) * 2 if is_full_tile else (N_chunk - n_remainder) * 2
                    self.generate_instruction_pbi_inc(
                        dram_shared_addr=(m_take - 1) * N * bytes_per_element if (not is_full_tile or n_remainder == 0) else 0,
                        dma_length=memcpy_length_bytes,
                        output_size=stride_bytes_per_chunk, # reuse stride_bytes_per_chunk for output_size
                        uram_length=0,
                        uram_a_start_addr=0,
                        uram_b_start_addr=0,
                        uram_wb_addr=0,
                        uram_dst_addr=0,
                        fmax_context_addr=0,
                        inst_pointer_idx=pointer_idx[5],
                    )
                else:
                    self.generate_instruction_pbi_inc(
                        dram_shared_addr=(m_take - 1) * N * bytes_per_element,
                        dma_length=0,
                        output_size=0, # reuse stride_bytes_per_chunk for output_size
                        uram_length=0,
                        uram_a_start_addr=0,
                        uram_b_start_addr=0,
                        uram_wb_addr=0,
                        uram_dst_addr=0,
                        fmax_context_addr=0,
                        inst_pointer_idx=pointer_idx[5],
                    )

        def emit_m_tile_softmax(m_take_chunk_idx: int, m_take_chunk: int) -> None:
            input_sram_start_addr = 0x00000 # URAM_A start
            nbytes = m_take_chunk * N * 2
            self.ue_memcpy_from_dram(
                dram_src_addr=m_take_chunk * N * bytes_per_element,
                memcpy_length_bytes=0,
                memcpy_type=MEMCPY_TYPE.URAM.value,
                uram_dst_addr=0,
                uram_type=URAM_SECTION.URAM_A.value,
                inst_pointer_idx=pointer_idx[11],
            )

            output_sram_wb_addr = 0x80000 # URAM_B start

            row_byte_stride = N * bytes_per_element
            vector_uram_row_base = input_sram_start_addr >> 7
            uram_wb_row_base = output_sram_wb_addr >> 7
            uram_wb_row_stride = row_byte_stride >> 7
            assert N % UE_VECTOR_SIZE == 0, "N must be a multiple of UE_VECTOR_SIZE"
            row_size = N // UE_VECTOR_SIZE

            if debug_fmax:
                uram_start_addr = (m_take_chunk * N * bytes_per_element) >> 7
                nbytes = UE_VECTOR_SIZE * 2
                row_size_fmax = (UE_VECTOR_SIZE + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE

            # pbi init for softmax loop:
            if debug_fmax:
                self.generate_instruction_pbi_init(
                dram_shared_addr=0,
                dma_length=0,
                output_size=0,
                uram_length=row_size_fmax,
                uram_a_start_addr=uram_start_addr,
                uram_b_start_addr=0,
                uram_wb_addr=uram_start_addr,
                uram_dst_addr=0,
                fmax_context_addr=m_take_chunk_idx,
                inst_pointer_idx=pointer_idx[6],
                )
            self.generate_instruction_pbi_init(
                    dram_shared_addr=0,
                    dma_length=0,
                    output_size=0,
                    uram_length=row_size,
                    uram_a_start_addr=vector_uram_row_base,
                    uram_b_start_addr=0,
                    uram_wb_addr=vector_uram_row_base,
                    uram_dst_addr=0,
                    fmax_context_addr=m_take_chunk_idx,
                    inst_pointer_idx=pointer_idx[8],
                    )
            self.generate_instruction_pbi_init(
                    dram_shared_addr=0,
                    dma_length=0,
                    output_size=0,
                    uram_length=row_size,
                    uram_a_start_addr=vector_uram_row_base,
                    uram_b_start_addr=0,
                    uram_wb_addr=uram_wb_row_base,
                    uram_dst_addr=0,
                    fmax_context_addr=0,
                    inst_pointer_idx=pointer_idx[9],
                    )

            # PBI loop over innermost m_take_chunk for softmax
            self.loop_start(m_take_chunk)
            if debug_fmax:
                self.ue_memcpy_from_dram(
                    ZERO_DRAM_ADDR,
                    nbytes,
                    memcpy_type=MEMCPY_TYPE.URAM.value,
                    uram_dst_addr=uram_start_addr,
                    uram_type=URAM_SECTION.URAM_A.value,
                    stride_bytes_per_chunk=0,
                    stride_jump_bytes=0,
                    inst_pointer_idx=0
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

            nbytes = m_take_chunk * N * 2
            self.ue_memcpy_to_dram(
                memcpy_type=MEMCPY_TYPE.URAM.value,
                uram_type=URAM_SECTION.URAM_B.value,
                uram_src_addr=0,
                dram_dst_addr=m_take_chunk * N * bytes_per_element,
                memcpy_length_bytes=0,
                inst_pointer_idx=pointer_idx[12],
            )

        def emit_m_tile(is_full_tile: bool = True) -> None:
            m_take = M_chunk if is_full_tile else m_remainder

            # pointer global initialization for all loops in PBI mode
            if is_full_tile:
                if bias_enable and bias_mode == "full_matrix":
                    self.generate_instruction_pbi_init(
                        dram_shared_addr=C_DRAM_ADDR,
                        dma_length=N_chunk * bytes_per_element,
                        output_size=0,
                        uram_length=0,
                        uram_a_start_addr=0,
                        uram_b_start_addr=0,
                        uram_wb_addr=0,
                        uram_dst_addr=0,
                        fmax_context_addr=0,
                        inst_pointer_idx=pointer_idx[0],
                    )
                if softmax_enable:
                    self.generate_instruction_pbi_init(
                    dram_shared_addr=OUTPUT_DRAM_ADDR,
                    dma_length=M_softmax_chunk * N * 2,
                    output_size=0,
                    uram_length=0,
                    uram_a_start_addr=0,
                    uram_b_start_addr=0,
                    uram_wb_addr=0,
                    uram_dst_addr=0,
                    fmax_context_addr=0,
                    inst_pointer_idx=pointer_idx[11],
                    )
                    self.generate_instruction_pbi_init(
                    dram_shared_addr=OUTPUT_DRAM_ADDR,
                    dma_length=M_softmax_chunk * N * 2,
                    output_size=0,
                    uram_length=0,
                    uram_a_start_addr=0,
                    uram_b_start_addr=0,
                    uram_wb_addr=0,
                    uram_dst_addr=0,
                    fmax_context_addr=0,
                    inst_pointer_idx=pointer_idx[12],
                    )

            m_rows = M_chunk if is_full_tile else m_remainder
            if not write_back_disable:
                if N_chunk_aligned is None:
                    self.generate_instruction_pbi_init(
                        dram_shared_addr=(
                            OUTPUT_DRAM_ADDR
                            if is_full_tile
                            else OUTPUT_DRAM_ADDR + M_chunk * num_full_m_tiles * N * bytes_per_element
                        ),
                        dma_length=m_rows * N_chunk * 2,
                        output_size=N_chunk * bytes_per_element,
                        uram_length=0,
                        uram_a_start_addr=(m_rows * K * bytes_per_element) >> 7,
                        uram_b_start_addr=0,
                        uram_wb_addr=0,
                        uram_dst_addr=0,
                        fmax_context_addr=0,
                        inst_pointer_idx=pointer_idx[5],
                    )
                else:
                    self.generate_instruction_pbi_init(
                        dram_shared_addr=(
                        OUTPUT_DRAM_ADDR if is_full_tile else OUTPUT_DRAM_ADDR + num_full_m_tiles * M_chunk * N * bytes_per_element
                        ),
                        dma_length=N_chunk * 2,
                        output_size=0,
                        uram_length=0,
                        uram_a_start_addr=m_take * K * bytes_per_element >> 7,
                        uram_b_start_addr=m_take * K * bytes_per_element >> 7,
                        uram_wb_addr=0,
                        uram_dst_addr=0,
                        fmax_context_addr=0,
                        inst_pointer_idx=pointer_idx[5],
                    )

            self.generate_instruction_pbi_init(
                dram_shared_addr=(
                    A_DRAM_ADDR
                    if is_full_tile
                    else A_DRAM_ADDR + num_full_m_tiles * M_chunk * K * bytes_per_element
                ),
                dma_length=m_rows * K * 2,
                output_size=0,
                uram_length=0,
                uram_a_start_addr=0,
                uram_b_start_addr=0,
                uram_wb_addr=0,
                uram_dst_addr=0,
                fmax_context_addr=0,
                inst_pointer_idx=pointer_idx[10],
            )
            if softmax_enable:
                if debug_fmax:
                    uram_start_addr = ((M_softmax_chunk if is_full_tile else M_softmax_chunk_remainder) * N * bytes_per_element) >> 7
                    self.generate_instruction_pbi_init(
                        dram_shared_addr=(
                            FMAX_DRAM_ADDR
                            if is_full_tile
                            else FMAX_DRAM_ADDR + num_full_m_tiles * M_chunk * UE_VECTOR_SIZE * bytes_per_element
                        ),
                        dma_length=UE_VECTOR_SIZE * 2,
                        output_size=0,
                        uram_length=0,
                        uram_a_start_addr=uram_start_addr,
                        uram_b_start_addr=uram_start_addr,
                        uram_wb_addr=0,
                        uram_dst_addr=0,
                        fmax_context_addr=0,
                        inst_pointer_idx=pointer_idx[7],
                    )

            # pbi loop body over outermost dimension M (M_chunk)
            if is_full_tile:
                # absolute jump before the outermost loop start to make sure the for loop starts at the beginning of the i-cache
                program_dram_start_addr = self.get_program_dram_addr()
                cur_inst_count = self.capture_count
                jump_target_word_addr = ue_35bit_addr_shifter(program_dram_start_addr + (cur_inst_count + 1) * INSTRUCTION_SIZE_BYTES)
                self.generate_instruction_jump_abs(jump_target_word_addr)
                self.loop_start(num_full_m_tiles)
            self.accelerator_memory_to_sram(accelerator_dram_address=M_chunk * K * bytes_per_element,
                                            sram_address=0,
                                            element_size=0,
                                            inst_pointer_idx=pointer_idx[10],
                                            )
            # clear max for each M tile
            self.generate_instruction_clear_fmax()

            emit_n_tile_pbi(True, N_chunk, m_take)
            if n_remainder:
                # For the level-3 loop, we need reset the dma_length pointer through pbi increment op for N_remainder chunk.
                if bias_enable and bias_mode == "full_matrix":
                    self.generate_instruction_pbi_inc(
                    dram_shared_addr=0,
                    dma_length=(n_remainder - N_chunk)*2,
                    output_size=0, # reuse stride_bytes_per_chunk for output_size
                    uram_length=0,
                    uram_a_start_addr=0,
                    uram_b_start_addr=0,
                    uram_wb_addr=0,
                    uram_dst_addr=0,
                    fmax_context_addr=0,
                    inst_pointer_idx=pointer_idx[0],
                    )
                emit_n_tile_pbi(False, n_remainder, m_take)
            else:
                # For the level-3 loop, we need reset the dram_addr pointer through pbi increment op for no N_remainder case.
                if bias_enable and bias_mode == "full_matrix":
                    self.generate_instruction_pbi_inc(
                    dram_shared_addr=(m_take * N - N) * bytes_per_element,
                    dma_length=0,
                    output_size=0, # reuse stride_bytes_per_chunk for output_size
                    uram_length=0,
                    uram_a_start_addr=0,
                    uram_b_start_addr=0,
                    uram_wb_addr=0,
                    uram_dst_addr=0,
                    fmax_context_addr=0,
                    inst_pointer_idx=pointer_idx[0],
                    )

            if softmax_enable:
                num_softmax_tiles = num_softmax_m_tiles if is_full_tile else num_softmax_m_tiles_remainder
                softmax_remainder = m_softmax_remainder if is_full_tile else m_softmax_remainder_remainder
                softmax_chunk = M_softmax_chunk if is_full_tile else M_softmax_chunk_remainder
                for softmax_tile_idx in range(num_softmax_tiles):
                    emit_m_tile_softmax(softmax_tile_idx * softmax_chunk, softmax_chunk)
                if softmax_remainder:
                    emit_m_tile_softmax(num_softmax_tiles * softmax_chunk, softmax_remainder)
            if is_full_tile:
                outer_loop_size = self.loop_end()
                print(f"Loop body size: {outer_loop_size}")
                loop_size_limit = 256 if m_remainder else 512
                assert outer_loop_size <= loop_size_limit, f"Outer loop body size {outer_loop_size} is greater than i-cache size of {loop_size_limit} instructions"

        # core function instruction starts here
        emit_m_tile(is_full_tile=True)
        if m_remainder:
            emit_m_tile(is_full_tile=False)
        for ptr in reversed(pointer_idx):
            self.release_inst_ptr(ptr)

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

    def matmat_mul_core(self, M: int, K: int, N: int, A_DRAM_ADDR: int, B_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, softmax_enable: bool = False, C_DRAM_ADDR: int = None, bias_mode: str = "broadcast_N",
                            is_B_quantized: bool = False, data_type: TYPE = None, SCALE_DRAM_ADDR: int = None, gelu_enable: bool = False, silu_enable: bool = False, sigmoid_enable: bool = False,
                            clamp_enable: bool = False, log_enable: bool = False,
                            debug_fmax: bool = False, ZERO_DRAM_ADDR: int = None, FMAX_DRAM_ADDR: int = None,
                            use_pbi: bool = False, write_back_disable: bool = False) -> None:
        """Matrix multiply entrypoint; switch between PBI and legacy core paths."""
        if use_pbi:
            return self.matmat_mul_core_pbi(
                M, K, N, A_DRAM_ADDR, B_DRAM_ADDR, OUTPUT_DRAM_ADDR, softmax_enable, C_DRAM_ADDR, bias_mode,
                is_B_quantized, data_type, SCALE_DRAM_ADDR, gelu_enable, silu_enable, sigmoid_enable,
                clamp_enable, log_enable,
                debug_fmax, ZERO_DRAM_ADDR, FMAX_DRAM_ADDR,
                write_back_disable=write_back_disable,
            )
        else:
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

        # Program engine0
        ue0.start_capture()
        ue0.generate_instruction_flag_clear()
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
            use_pbi=use_pbi,
        )
        ue0.generate_instruction_flag_set()
        ue0.generate_instruction_halt()
        ue0.stop_capture()

        engine0_program_dram_addr = ue0.get_program_dram_addr()
        ue0.write_captured_instructions_to_dram(engine0_program_dram_addr)
        ue0.allocate_program_dram(ue0.get_capture_instruction_size_bytes())

        # Program ue1
        ue1.start_capture()
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
            use_pbi=use_pbi,
        )
        ue1.generate_instruction_flag_check(target_engine_idx=0)
        ue1.generate_instruction_halt()
        ue1.stop_capture()

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

    def bf16_transpose_core(self, M: int, N: int, INPUT_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int) -> None:
        """
        Transposes a (M x N) input matrix X to produce an (N x M) output matrix Y = X^T.

        Uses the identity matrix trick: for each column j of X, computes
        Y[j, :] = e_j^T @ X = X[:, j] via bf16 matvec with a one-hot vector.
        Processes N in column blocks of UE_VECTOR_SIZE width.
        """
        bytes_per_element = 2
        # Allocate identity matrix of size UE_VECTOR_SIZE x UE_VECTOR_SIZE in URAM_A start
        identity_matrix_sram_start_addr = 0x00000
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

    def flash_attention_core_legacy(self, head_dim: int, seq_len: int, Q_DRAM_ADDR: int, K_DRAM_ADDR: int, V_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, SCRATCH_DRAM_ADDR: int, BIAS_DRAM_ADDR: int = None,
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

        # Allocate identity matrix of size UE_VECTOR_SIZE x UE_VECTOR_SIZE in URAM_A start
        identity_matrix_sram_start_addr = 0x00000
        identity_matrix_dram_addr = self.get_params_dram_addr()
        self.allocate_params_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * bytes_per_element)
        self.dma_write(DMA_DEVICE_H2C, identity_matrix_dram_addr, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16), UE_VECTOR_SIZE * UE_VECTOR_SIZE * bytes_per_element)

        identity_tensor = torch.eye(head_dim, dtype=torch.bfloat16)

        # transfer identity matrix to URAM_A start
        self.accelerator_memory_to_sram(accelerator_dram_address=identity_matrix_dram_addr,
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

    def flash_attention_core_pbi(self, head_dim: int, seq_len: int, Q_DRAM_ADDR: int, K_DRAM_ADDR: int, V_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, SCRATCH_DRAM_ADDR: int, BIAS_DRAM_ADDR: int = None,
                            debug_mode: bool = False, SM_OUTPUT_DRAM_ADDR: int = None) -> None:
        bytes_per_element = 2
        bias_enable = BIAS_DRAM_ADDR is not None

        if debug_mode:  # DEBUG only, needs to be allocated in DRAM
            assert SM_OUTPUT_DRAM_ADDR is not None, "SM_OUTPUT_DRAM_ADDR is not set for debug mode"

        # SCRATCH_DRAM_ADDR is used for V^T
        SCRATCH_DRAM_PARTIAL_SM = SCRATCH_DRAM_ADDR + head_dim * seq_len * bytes_per_element  # used for partial softmax output

        # ----------------------------------------------------------------------------------------------------------------
        # I @ V^T migrated to matmat_mul_core_pbi in phase 1. This materializes V^T in
        # SCRATCH_DRAM_ADDR so the remaining stages can keep using the legacy flow.
        identity_matrix_dram_addr = self.get_params_dram_addr()
        self.allocate_params_dram(head_dim * head_dim * bytes_per_element)
        self.dma_write(
            DMA_DEVICE_H2C,
            identity_matrix_dram_addr,
            torch.eye(head_dim, dtype=torch.bfloat16),
            head_dim * head_dim * bytes_per_element,
        )
        self.matmat_mul_core(
            M=head_dim,
            K=head_dim,
            N=seq_len,
            A_DRAM_ADDR=identity_matrix_dram_addr,
            B_DRAM_ADDR=V_DRAM_ADDR,
            OUTPUT_DRAM_ADDR=SCRATCH_DRAM_ADDR,
            use_pbi=True,
        )

        # ----------------------------------------------------------------------------------------------------------------
        # Fully inlined PBI second half:
        #   1. Q @ K^T into SCRATCH_DRAM_PARTIAL_SM
        #   2. Softmax in place on a full M tile
        #   3. Softmax-row @ V^T with row-driven PBI loops
        M = seq_len
        K = head_dim
        N = seq_len
        uram_a_start_addr = 0x00000
        uram_b_start_addr = 0x80000
        uram_b_base_words = (uram_b_start_addr >> 7) & 0xFFF

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

        output_N_size = N_chunk_aligned if N_chunk_aligned is not None else N_chunk
        qkt_m_chunk = min(UE_FMAX_CONTEXT_SIZE, M, URAM_FULL_ELEMENTS // (K + output_N_size))
        softmax_m_chunk = min((URAM_FULL_ELEMENTS - UE_VECTOR_SIZE) // N, UE_FMAX_CONTEXT_SIZE, M)
        M_chunk = min(qkt_m_chunk, softmax_m_chunk)
        assert M_chunk >= 1 and M_chunk <= M, f"M_chunk={M_chunk} must be greater than 0 and less than M={M}"

        print(f"M_chunk: {M_chunk}, N_chunk: {N_chunk}", f"N_chunk_aligned: {N_chunk_aligned}")
        print(f"URAM_A usage: {100 * (M_chunk * N + UE_VECTOR_SIZE) / URAM_FULL_ELEMENTS:.2f}% of URAM_FULL_ELEMENTS")
        print(f"URAM_B usage: {100 * N_chunk * K / URAM_FULL_ELEMENTS:.2f}% of URAM_FULL_ELEMENTS")

        num_full_m_tiles = M // M_chunk
        m_remainder = M % M_chunk
        print(f"M_chunk: {M_chunk}, M_remainder: {m_remainder}, num_full_m_tiles: {num_full_m_tiles}")
        num_full_n_tiles = N // N_chunk
        n_remainder = N % N_chunk
        print(f"N_chunk: {N_chunk}, N_remainder: {n_remainder}, num_full_n_tiles: {num_full_n_tiles}")
        assert N_chunk_aligned is None or n_remainder == 0, f"Illegal case with N_chunk_aligned={N_chunk_aligned} and n_remainder={n_remainder}"

        def compute_v_chunk() -> int:
            v_chunk = min(head_dim, (URAM_NEAR_FULL_ELEMENTS // seq_len // UE_VECTOR_SIZE) * UE_VECTOR_SIZE)
            if v_chunk >= UE_VECTOR_SIZE:
                return v_chunk
            for candidate in (32, 16):
                if candidate <= head_dim and seq_len * candidate <= URAM_NEAR_FULL_ELEMENTS:
                    return candidate
            assert False, f"Unable to fit V^T tile for head_dim={head_dim}, seq_len={seq_len}"

        V_chunk = compute_v_chunk()
        num_full_v_tiles = head_dim // V_chunk
        v_remainder = head_dim % V_chunk
        print(f"V_chunk: {V_chunk}, V_remainder: {v_remainder}, num_full_v_tiles: {num_full_v_tiles}")

        pointer_idx = [self.alloc_inst_ptr() for _ in range(12 if debug_mode else 11)]
        bias_ptr = pointer_idx[0]
        qkt_dot_ptr = pointer_idx[1]
        k_load_ptr = pointer_idx[2]
        qkt_wb_ptr = pointer_idx[3]
        q_load_ptr = pointer_idx[4]
        softmax_load_ptr = pointer_idx[5]
        softmax_exp_ptr = pointer_idx[6]
        softmax_mul_ptr = pointer_idx[7]
        v_load_ptr = pointer_idx[8]
        sm_dot_ptr = pointer_idx[9]
        output_wb_ptr = pointer_idx[10]
        debug_sm_wb_ptr = pointer_idx[11] if debug_mode else None

        def emit_qkt_n_tile_pbi(is_full_tile: bool, n_take: int, m_take: int) -> None:
            ue_b = UE_VECTOR_SIZE * bytes_per_element
            output_sram_wb_addr = m_take * K * bytes_per_element
            row_full_b = (N_chunk if N_chunk_aligned is None else N_chunk_aligned) * bytes_per_element
            row_res_b = (n_remainder * bytes_per_element if N_chunk_aligned is None else row_full_b)
            pbi_inc_dot_product_uram_start_addr = (K * bytes_per_element) // ue_b
            pbi_inc_dot_product_uram_wb_addr_full = row_full_b // ue_b
            diff_res = n_remainder and row_res_b != row_full_b
            pbi_inc_dot_product_uram_wb_addr_residual = row_res_b // ue_b if diff_res else pbi_inc_dot_product_uram_wb_addr_full
            pbi_base_dot_product_uram_wb_addr = output_sram_wb_addr // ue_b
            pbi_inc_dot_product_uram_wb_addr = (
                pbi_inc_dot_product_uram_wb_addr_full if is_full_tile else pbi_inc_dot_product_uram_wb_addr_residual
            )

            if is_full_tile:
                self.generate_instruction_pbi_init(
                    dram_shared_addr=K_DRAM_ADDR,
                    dma_length=n_take * K * bytes_per_element,
                    output_size=0,
                    uram_length=0,
                    uram_a_start_addr=0,
                    uram_b_start_addr=uram_b_base_words,
                    uram_wb_addr=0,
                    uram_dst_addr=0,
                    fmax_context_addr=0,
                    inst_pointer_idx=k_load_ptr,
                )

            if is_full_tile:
                self.loop_start(num_full_n_tiles)
                self.generate_instruction_pbi_init(
                    dram_shared_addr=0,
                    dma_length=K * n_take,
                    output_size=n_take,
                    uram_length=K // UE_VECTOR_SIZE,
                    uram_a_start_addr=0,
                    uram_b_start_addr=0,
                    uram_wb_addr=pbi_base_dot_product_uram_wb_addr,
                    uram_dst_addr=0,
                    fmax_context_addr=0,
                    inst_pointer_idx=qkt_dot_ptr,
                )
            else:
                self.generate_instruction_pbi_init(
                    dram_shared_addr=0,
                    dma_length=K * n_take,
                    output_size=n_take,
                    uram_length=K // UE_VECTOR_SIZE,
                    uram_a_start_addr=0,
                    uram_b_start_addr=0,
                    uram_wb_addr=pbi_base_dot_product_uram_wb_addr,
                    uram_dst_addr=0,
                    fmax_context_addr=0,
                    inst_pointer_idx=qkt_dot_ptr,
                )

            self.ue_memcpy_from_dram(
                K * bytes_per_element * N_chunk if is_full_tile else K_DRAM_ADDR + num_full_n_tiles * N_chunk * K * bytes_per_element,
                0 if is_full_tile else n_take * K * bytes_per_element,
                MEMCPY_TYPE.URAM.value,
                0 if is_full_tile else uram_b_base_words,
                URAM_SECTION.URAM_B,
                inst_pointer_idx=k_load_ptr if is_full_tile else 0,
            )

            self.loop_start(m_take)
            if bias_enable:
                self.accelerator_memory_to_bias_sram(
                    accelerator_dram_address=N * bytes_per_element,
                    element_size=0,
                    inst_pointer_idx=bias_ptr,
                )

            self.ue_arithmetic_op(
                broadcast_mode=0,
                max_clear_en=0,
                stride_z=1,
                lalu_a=0,
                lalu_b=0,
                lalu_mode=LALU_MODE.BYPASS.value,
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
                inst_pointer_idx=qkt_dot_ptr,
            )
            self.loop_end()

            if bias_enable:
                self.accelerator_memory_to_bias_sram(
                    accelerator_dram_address=(
                        (N_chunk - m_take * N) * bytes_per_element
                        if is_full_tile
                        else (n_remainder - N) * bytes_per_element
                    ),
                    element_size=0 if is_full_tile else N_chunk - n_remainder,
                    inst_pointer_idx=bias_ptr,
                )

            if N_chunk_aligned is None:
                self.ue_memcpy_to_dram(
                    memcpy_type=MEMCPY_TYPE.URAM.value,
                    uram_type=URAM_SECTION.URAM_A.value,
                    uram_src_addr=0,
                    dram_dst_addr=bytes_per_element * n_take,
                    memcpy_length_bytes=0,
                    stride_bytes_per_chunk=0,
                    stride_jump_bytes=N * bytes_per_element,
                    inst_pointer_idx=qkt_wb_ptr,
                )
            else:
                self.loop_start(m_take)
                self.ue_memcpy_to_dram(
                    memcpy_type=MEMCPY_TYPE.URAM.value,
                    uram_type=URAM_SECTION.URAM_A.value,
                    uram_src_addr=1,
                    dram_dst_addr=N * bytes_per_element,
                    memcpy_length_bytes=0,
                    inst_pointer_idx=qkt_wb_ptr,
                )
                self.loop_end()

                self.ue_memcpy_to_dram(
                    memcpy_type=MEMCPY_TYPE.URAM.value,
                    uram_type=URAM_SECTION.URAM_A.value,
                    uram_src_addr=-m_take,
                    dram_dst_addr=(n_take - m_take * N) * bytes_per_element,
                    memcpy_length_bytes=0,
                    inst_pointer_idx=qkt_wb_ptr,
                )

            if is_full_tile:
                self.loop_end()

            if N_chunk_aligned is None:
                memcpy_length_bytes = (
                    0 if is_full_tile and n_remainder == 0
                    else m_take * (n_remainder - N_chunk) * bytes_per_element if is_full_tile
                    else m_take * (N_chunk - n_remainder) * bytes_per_element
                )
                stride_bytes_per_chunk = (
                    0 if is_full_tile and n_remainder == 0
                    else (n_remainder - N_chunk) * bytes_per_element if is_full_tile
                    else (N_chunk - n_remainder) * bytes_per_element
                )
                self.generate_instruction_pbi_inc(
                    dram_shared_addr=(m_take - 1) * N * bytes_per_element if (not is_full_tile or n_remainder == 0) else 0,
                    dma_length=memcpy_length_bytes,
                    output_size=stride_bytes_per_chunk,
                    uram_length=0,
                    uram_a_start_addr=0,
                    uram_b_start_addr=0,
                    uram_wb_addr=0,
                    uram_dst_addr=0,
                    fmax_context_addr=0,
                    inst_pointer_idx=qkt_wb_ptr,
                )
            else:
                self.generate_instruction_pbi_inc(
                    dram_shared_addr=(m_take - 1) * N * bytes_per_element,
                    dma_length=0,
                    output_size=0,
                    uram_length=0,
                    uram_a_start_addr=0,
                    uram_b_start_addr=0,
                    uram_wb_addr=0,
                    uram_dst_addr=0,
                    fmax_context_addr=0,
                    inst_pointer_idx=qkt_wb_ptr,
                )

        def emit_softmax_rows_pbi(m_take: int) -> None:
            row_byte_stride = N * bytes_per_element
            row_stride_words = row_byte_stride >> 7
            row_size = N // UE_VECTOR_SIZE

            self.ue_memcpy_from_dram(
                dram_src_addr=m_take * N * bytes_per_element,
                memcpy_length_bytes=0,
                memcpy_type=MEMCPY_TYPE.URAM.value,
                uram_dst_addr=0,
                uram_type=URAM_SECTION.URAM_A.value,
                inst_pointer_idx=softmax_load_ptr,
            )

            self.generate_instruction_pbi_init(
                dram_shared_addr=0,
                dma_length=0,
                output_size=0,
                uram_length=row_size,
                uram_a_start_addr=0,
                uram_b_start_addr=0,
                uram_wb_addr=0,
                uram_dst_addr=0,
                fmax_context_addr=0,
                inst_pointer_idx=softmax_exp_ptr,
            )
            self.generate_instruction_pbi_init(
                dram_shared_addr=0,
                dma_length=0,
                output_size=0,
                uram_length=row_size,
                uram_a_start_addr=0,
                uram_b_start_addr=0,
                uram_wb_addr=0,
                uram_dst_addr=0,
                fmax_context_addr=0,
                inst_pointer_idx=softmax_mul_ptr,
            )

            self.loop_start(m_take)
            self.ue_arithmetic_op(
                broadcast_mode=BROADCAST_MODE.FMAX_NEGATE.value,
                max_clear_en=0,
                stride_z=1,
                lalu_a=0,
                lalu_b=0,
                lalu_mode=LALU_MODE.MODE_RECIP.value,
                scalar=self.float_to_bf19(1.0),
                uram_section=URAM_SECTION.URAM_A.value,
                uram_dst_addr=0,
                uram_wb_addr=row_stride_words,
                uram_write_src=URAM_WRITE_SRC.URAM_WRITE_BACK.value,
                mode=UE_MODE.EXP,
                data_type=0,
                uram_a_start_addr=row_stride_words,
                uram_b_start_addr=0,
                uram_length=0,
                dma_start_addr=0,
                dma_length=0,
                output_size=0,
                fmax_context_addr=1,
                inst_pointer_idx=softmax_exp_ptr,
            )
            self.ue_arithmetic_op(
                broadcast_mode=BROADCAST_MODE.LALU_RESULT.value,
                max_clear_en=0,
                stride_z=1,
                lalu_a=0,
                lalu_b=0,
                lalu_mode=LALU_MODE.BYPASS.value,
                scalar=self.float_to_bf19(1.0),
                uram_section=URAM_SECTION.URAM_A.value,
                uram_dst_addr=0,
                uram_wb_addr=row_stride_words,
                uram_write_src=URAM_WRITE_SRC.URAM_WRITE_BACK.value,
                mode=UE_MODE.MUL_BROADCAST,
                data_type=0,
                uram_a_start_addr=row_stride_words,
                uram_b_start_addr=0,
                uram_length=0,
                dma_start_addr=0,
                dma_length=0,
                output_size=0,
                fmax_context_addr=0,
                inst_pointer_idx=softmax_mul_ptr,
            )
            self.loop_end()

            if debug_mode:
                self.sram_to_accelerator_memory(
                    sram_address=uram_a_start_addr,
                    accelerator_dram_address=m_take * N * bytes_per_element,
                    element_size=0,
                    inst_pointer_idx=debug_sm_wb_ptr,
                    memcpy_length_bytes=m_take * N * bytes_per_element,
                )

        def emit_smv_rows_pbi(m_take: int) -> None:
            row_byte_stride = N * bytes_per_element
            row_stride_words = row_byte_stride >> 7
            output_row_buffer_addr = m_take * N * bytes_per_element
            output_row_buffer_words = output_row_buffer_addr >> 7
            assert output_row_buffer_addr + UE_VECTOR_SIZE * bytes_per_element <= 0x80000, (
                f"output row buffer spills out of URAM_A: addr={hex(output_row_buffer_addr)}"
            )

            self.generate_instruction_pbi_init(
                dram_shared_addr=0,
                dma_length=N * V_chunk,
                output_size=V_chunk,
                uram_length=N // UE_VECTOR_SIZE,
                uram_a_start_addr=0,
                uram_b_start_addr=0,
                uram_wb_addr=output_row_buffer_words,
                uram_dst_addr=0,
                fmax_context_addr=0,
                inst_pointer_idx=sm_dot_ptr,
            )

            self.loop_start(m_take)
            self.generate_instruction_pbi_init(
                dram_shared_addr=SCRATCH_DRAM_ADDR,
                dma_length=V_chunk * N * bytes_per_element,
                output_size=0,
                uram_length=0,
                uram_a_start_addr=0,
                uram_b_start_addr=uram_b_base_words,
                uram_wb_addr=0,
                uram_dst_addr=0,
                fmax_context_addr=0,
                inst_pointer_idx=v_load_ptr,
            )

            self.loop_start(num_full_v_tiles)
            self.ue_memcpy_from_dram(
                dram_src_addr=V_chunk * N * bytes_per_element,
                memcpy_length_bytes=0,
                memcpy_type=MEMCPY_TYPE.URAM.value,
                uram_dst_addr=0,
                uram_type=URAM_SECTION.URAM_B.value,
                inst_pointer_idx=v_load_ptr,
            )
            self.ue_arithmetic_op(
                broadcast_mode=0,
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
                mode=UE_MODE.BF16_DOT_PRODUCT,
                data_type=0,
                uram_a_start_addr=0,
                uram_b_start_addr=0,
                uram_length=0,
                dma_start_addr=0,
                dma_length=0,
                output_size=0,
                fmax_context_addr=0,
                inst_pointer_idx=sm_dot_ptr,
            )
            self.ue_memcpy_to_dram(
                memcpy_type=MEMCPY_TYPE.URAM.value,
                uram_type=URAM_SECTION.URAM_A.value,
                uram_src_addr=0,
                dram_dst_addr=V_chunk * bytes_per_element,
                memcpy_length_bytes=0,
                inst_pointer_idx=output_wb_ptr,
            )
            self.loop_end()

            if v_remainder:
                self.generate_instruction_pbi_inc(
                    dram_shared_addr=0,
                    dma_length=(v_remainder - V_chunk) * N,
                    output_size=v_remainder - V_chunk,
                    uram_length=0,
                    uram_a_start_addr=0,
                    uram_b_start_addr=0,
                    uram_wb_addr=0,
                    uram_dst_addr=0,
                    fmax_context_addr=0,
                    inst_pointer_idx=sm_dot_ptr,
                )
                self.generate_instruction_pbi_inc(
                    dram_shared_addr=0,
                    dma_length=(v_remainder - V_chunk) * bytes_per_element,
                    output_size=0,
                    uram_length=0,
                    uram_a_start_addr=0,
                    uram_b_start_addr=0,
                    uram_wb_addr=0,
                    uram_dst_addr=0,
                    fmax_context_addr=0,
                    inst_pointer_idx=output_wb_ptr,
                )

                self.ue_memcpy_from_dram(
                    dram_src_addr=SCRATCH_DRAM_ADDR + num_full_v_tiles * V_chunk * N * bytes_per_element,
                    memcpy_length_bytes=v_remainder * N * bytes_per_element,
                    memcpy_type=MEMCPY_TYPE.URAM.value,
                    uram_dst_addr=uram_b_start_addr,
                    uram_type=URAM_SECTION.URAM_B.value,
                )
                self.ue_arithmetic_op(
                    broadcast_mode=0,
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
                    mode=UE_MODE.BF16_DOT_PRODUCT,
                    data_type=0,
                    uram_a_start_addr=0,
                    uram_b_start_addr=0,
                    uram_length=0,
                    dma_start_addr=0,
                    dma_length=0,
                    output_size=0,
                    fmax_context_addr=0,
                    inst_pointer_idx=sm_dot_ptr,
                )
                self.ue_memcpy_to_dram(
                    memcpy_type=MEMCPY_TYPE.URAM.value,
                    uram_type=URAM_SECTION.URAM_A.value,
                    uram_src_addr=0,
                    dram_dst_addr=v_remainder * bytes_per_element,
                    memcpy_length_bytes=0,
                    inst_pointer_idx=output_wb_ptr,
                )

                self.generate_instruction_pbi_inc(
                    dram_shared_addr=0,
                    dma_length=(V_chunk - v_remainder) * N,
                    output_size=V_chunk - v_remainder,
                    uram_length=0,
                    uram_a_start_addr=row_stride_words,
                    uram_b_start_addr=0,
                    uram_wb_addr=0,
                    uram_dst_addr=0,
                    fmax_context_addr=0,
                    inst_pointer_idx=sm_dot_ptr,
                )
                self.generate_instruction_pbi_inc(
                    dram_shared_addr=0,
                    dma_length=(V_chunk - v_remainder) * bytes_per_element,
                    output_size=0,
                    uram_length=0,
                    uram_a_start_addr=0,
                    uram_b_start_addr=0,
                    uram_wb_addr=0,
                    uram_dst_addr=0,
                    fmax_context_addr=0,
                    inst_pointer_idx=output_wb_ptr,
                )
            else:
                self.generate_instruction_pbi_inc(
                    dram_shared_addr=0,
                    dma_length=0,
                    output_size=0,
                    uram_length=0,
                    uram_a_start_addr=row_stride_words,
                    uram_b_start_addr=0,
                    uram_wb_addr=0,
                    uram_dst_addr=0,
                    fmax_context_addr=0,
                    inst_pointer_idx=sm_dot_ptr,
                )
            self.loop_end()

        def emit_m_tile(is_full_tile: bool = True) -> None:
            m_take = M_chunk if is_full_tile else m_remainder
            assert m_take > 0

            q_tile_base_addr = Q_DRAM_ADDR if is_full_tile else Q_DRAM_ADDR + num_full_m_tiles * M_chunk * K * bytes_per_element
            score_tile_base_addr = SCRATCH_DRAM_PARTIAL_SM if is_full_tile else SCRATCH_DRAM_PARTIAL_SM + num_full_m_tiles * M_chunk * N * bytes_per_element
            bias_tile_base_addr = (
                BIAS_DRAM_ADDR
                if is_full_tile
                else BIAS_DRAM_ADDR + num_full_m_tiles * M_chunk * N * bytes_per_element
            ) if bias_enable else 0
            output_tile_base_addr = OUTPUT_DRAM_ADDR if is_full_tile else OUTPUT_DRAM_ADDR + num_full_m_tiles * M_chunk * head_dim * bytes_per_element
            output_sram_wb_addr = m_take * K * bytes_per_element
            output_row_buffer_addr = m_take * N * bytes_per_element

            if bias_enable:
                self.generate_instruction_pbi_init(
                    dram_shared_addr=bias_tile_base_addr,
                    dma_length=N_chunk * bytes_per_element,
                    output_size=0,
                    uram_length=0,
                    uram_a_start_addr=0,
                    uram_b_start_addr=0,
                    uram_wb_addr=0,
                    uram_dst_addr=0,
                    fmax_context_addr=0,
                    inst_pointer_idx=bias_ptr,
                )

            if N_chunk_aligned is None:
                self.generate_instruction_pbi_init(
                    dram_shared_addr=score_tile_base_addr,
                    dma_length=m_take * N_chunk * bytes_per_element,
                    output_size=N_chunk * bytes_per_element,
                    uram_length=0,
                    uram_a_start_addr=output_sram_wb_addr >> 7,
                    uram_b_start_addr=0,
                    uram_wb_addr=0,
                    uram_dst_addr=0,
                    fmax_context_addr=0,
                    inst_pointer_idx=qkt_wb_ptr,
                )
            else:
                self.generate_instruction_pbi_init(
                    dram_shared_addr=score_tile_base_addr,
                    dma_length=N_chunk * bytes_per_element,
                    output_size=0,
                    uram_length=0,
                    uram_a_start_addr=output_sram_wb_addr >> 7,
                    uram_b_start_addr=output_sram_wb_addr >> 7,
                    uram_wb_addr=0,
                    uram_dst_addr=0,
                    fmax_context_addr=0,
                    inst_pointer_idx=qkt_wb_ptr,
                )

            self.generate_instruction_pbi_init(
                dram_shared_addr=q_tile_base_addr,
                dma_length=m_take * K * bytes_per_element,
                output_size=0,
                uram_length=0,
                uram_a_start_addr=0,
                uram_b_start_addr=0,
                uram_wb_addr=0,
                uram_dst_addr=0,
                fmax_context_addr=0,
                inst_pointer_idx=q_load_ptr,
            )
            self.generate_instruction_pbi_init(
                dram_shared_addr=score_tile_base_addr,
                dma_length=m_take * N * bytes_per_element,
                output_size=0,
                uram_length=0,
                uram_a_start_addr=0,
                uram_b_start_addr=0,
                uram_wb_addr=0,
                uram_dst_addr=0,
                fmax_context_addr=0,
                inst_pointer_idx=softmax_load_ptr,
            )
            self.generate_instruction_pbi_init(
                dram_shared_addr=output_tile_base_addr,
                dma_length=V_chunk * bytes_per_element,
                output_size=0,
                uram_length=0,
                uram_a_start_addr=output_row_buffer_addr >> 7,
                uram_b_start_addr=output_row_buffer_addr >> 7,
                uram_wb_addr=0,
                uram_dst_addr=0,
                fmax_context_addr=0,
                inst_pointer_idx=output_wb_ptr,
            )
            if is_full_tile:
                self.loop_start(num_full_m_tiles)

            self.accelerator_memory_to_sram(
                accelerator_dram_address=m_take * K * bytes_per_element,
                sram_address=uram_a_start_addr,
                element_size=0,
                inst_pointer_idx=q_load_ptr,
            )
            self.broadcast_mul(
                scalar=1 / math.sqrt(head_dim),
                sram_start_addr=uram_a_start_addr,
                sram_wb_addr=uram_a_start_addr,
                element_size=m_take * K,
            )
            self.generate_instruction_clear_fmax()

            emit_qkt_n_tile_pbi(True, N_chunk, m_take)
            if n_remainder:
                if bias_enable:
                    self.generate_instruction_pbi_inc(
                        dram_shared_addr=0,
                        dma_length=(n_remainder - N_chunk) * bytes_per_element,
                        output_size=0,
                        uram_length=0,
                        uram_a_start_addr=0,
                        uram_b_start_addr=0,
                        uram_wb_addr=0,
                        uram_dst_addr=0,
                        fmax_context_addr=0,
                        inst_pointer_idx=bias_ptr,
                    )
                emit_qkt_n_tile_pbi(False, n_remainder, m_take)
            elif bias_enable:
                self.generate_instruction_pbi_inc(
                    dram_shared_addr=(m_take * N - N) * bytes_per_element,
                    dma_length=0,
                    output_size=0,
                    uram_length=0,
                    uram_a_start_addr=0,
                    uram_b_start_addr=0,
                    uram_wb_addr=0,
                    uram_dst_addr=0,
                    fmax_context_addr=0,
                    inst_pointer_idx=bias_ptr,
                )

            if debug_mode:
                self.generate_instruction_pbi_init(
                    dram_shared_addr=SM_OUTPUT_DRAM_ADDR if is_full_tile else SM_OUTPUT_DRAM_ADDR + num_full_m_tiles * M_chunk * N * bytes_per_element,
                    dma_length=m_take * N * bytes_per_element,
                    output_size=0,
                    uram_length=0,
                    uram_a_start_addr=0,
                    uram_b_start_addr=0,
                    uram_wb_addr=0,
                    uram_dst_addr=0,
                    fmax_context_addr=0,
                    inst_pointer_idx=debug_sm_wb_ptr,
                )

            emit_softmax_rows_pbi(m_take)
            emit_smv_rows_pbi(m_take)

            if is_full_tile:
                outer_loop_size = self.loop_end()
                print(f"Loop body size: {outer_loop_size}")
                loop_size_limit = 256 if m_remainder else 512
                assert outer_loop_size <= loop_size_limit, (
                    f"Outer loop body size {outer_loop_size} is greater than i-cache size of {loop_size_limit} instructions"
                )

        emit_m_tile(is_full_tile=True)
        if m_remainder:
            emit_m_tile(is_full_tile=False)

        for ptr in reversed(pointer_idx):
            self.release_inst_ptr(ptr)

        # Total Theoretical FLOPS: seq_len * head_dim + 2 * seq_len * head_dim * seq_len + seq_len * seq_len + seq_len * seq_len * 5 + 2 * seq_len * seq_len * head_dim
        total_flops = seq_len * head_dim # q_scale
        total_flops += 2 * seq_len * head_dim * seq_len # Q @ K^T
        if bias_enable:
            total_flops += seq_len * seq_len # bias
        total_flops += seq_len * seq_len * 5 # softmax
        total_flops += 2 * seq_len * seq_len * head_dim # sm @ v
        print(f"Total Theoretical FLOPS: {total_flops / 1e9:.6f} G")
        return total_flops

    def flash_attention_core(self, head_dim: int, seq_len: int, Q_DRAM_ADDR: int, K_DRAM_ADDR: int, V_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, SCRATCH_DRAM_ADDR: int, BIAS_DRAM_ADDR: int = None,
                            debug_mode: bool = False, SM_OUTPUT_DRAM_ADDR: int = None, use_pbi: bool = False) -> None:
        """Flash-attention entrypoint; switch between PBI and legacy core paths."""
        if use_pbi:
            return self.flash_attention_core_pbi(
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
        )

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
            if pbi_m == PBI_MODE_INIT:
                pbi_label = "INIT"
            elif pbi_m == PBI_MODE_INC:
                pbi_label = "INC"
            else:
                pbi_label = f"UNKNOWN_MODE({pbi_m})"
            dram_w = _inst_desc_bits(w, 32, 63)
            dma_len = _inst_desc_bits(w, 64, 95)
            out_sz = _inst_desc_bits(w, 156, 171)
            ur_rs = _inst_desc_bits(w, 96, 107)
            ur_rsz = _inst_desc_bits(w, 108, 119)
            ur_ay = _inst_desc_bits(w, 120, 131)
            ur_az = _inst_desc_bits(w, 132, 143)
            ur_wb = _inst_desc_bits(w, 144, 155)
            fmx = _inst_desc_bits(w, 223, 228)
            ur_md = _inst_desc_bits(w, 234, 245)
            result = (
                f"PBI_SET ({pbi_label}) inst_pointer_idx={ptr_i}\n"
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
        if inst_type == INSTRUCTION_ADD:
            isa_mode_names = {
                INST_ADD_INC: "INC",
                INST_ADD_DEC: "DEC",
                INST_ADD_REG: "REG",
                INST_ADD_IMM: "IMM",
                INST_ADD_SET: "SET"
            }
            mode_name = isa_mode_names.get(isa_mode, f"UNKNOWN({isa_mode})")
            result = f"ISA_ADD ({mode_name})"
            result += f"\n    transaction_id: {transaction_id}"
            if isa_mode == INST_ADD_SET:
                result += f"\n    dst_reg: {dst_reg_idx}, value: {_u32(immediate_value):#010X}"
            elif isa_mode == INST_ADD_INC:
                result += f"\n    reg: {dst_reg_idx}"
            elif isa_mode == INST_ADD_DEC:
                result += f"\n    reg: {dst_reg_idx}"
            elif isa_mode == INST_ADD_IMM:
                result += f"\n    reg: {dst_reg_idx}, immediate: {_u32(immediate_value):#010X}"
            elif isa_mode == INST_ADD_REG:
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

        result = f"ISA_UNKNOWN (type=0x{inst_type:X})"
        result += f"\n    transaction_id: {transaction_id}"
        for line in result.split('\n'):
            print(f"        {line}")

    def ue_isa_descriptor(self, inst_type: int, immediate_value: int = 0,
                          isa_mode: int = 0, src_reg_idx: int = 0,
                          dst_reg_idx: int = 0, rst_reg_idx: int = 0):
        """
        Shared 256b instruction descriptor compiler for ISA micro-ops (JUMP / ADD / REG_REWRITE / SEMAPHORE / FLAG).

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
        """Generate a HALT instruction. Instruction index comes from :attr:`_inst_id` (then incremented)."""
        self.ue_isa_descriptor(INSTRUCTION_HALT)

    def generate_instruction_swi(self):
        """Generate an SWI instruction (software interrupt). Same header layout as HALT."""
        self.ue_isa_descriptor(INSTRUCTION_SWI)

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
        """
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

    def loop_start(self, loop_cnt: int) -> int:
        """
        Open a counted backward-branch loop during instruction capture (stack-nested).

        Pushes a new frame: allocates a counter register, emits ``ADD_SET`` with ``loop_cnt``, and
        records ``body_start_inst_cnt`` = :attr:`_inst_id` after that instruction (the transaction id
        of the next captured instruction, i.e. the loop-body head). Nested loops call
        ``loop_start`` again before the matching ``loop_end``; each frame has its own register and
        start index.

        Args:
            loop_cnt: Initial iteration count (same semantics as :meth:`generate_instruction_add_set`).

        Returns:
            The allocated loop counter register index.

        Raises:
            RuntimeError: If capture is not active.
        """
        if not self.is_capture_on or self.capture_buffer is None:
            raise RuntimeError("loop_start() requires an active capture (call start_capture() first).")
        reg = self.alloc_isa_reg()
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
            print("ERROR: INSTRUCTION_ADD overwriting reg_idx 0 (zero reg) not allowed")
            return
        self.ue_isa_descriptor(INSTRUCTION_ADD, isa_mode=INST_ADD_INC,
                               src_reg_idx=reg_idx, dst_reg_idx=reg_idx)

    def generate_instruction_add_dec(self, reg_idx: int):
        """
        Generate an ADD instruction to decrement a register

        Args:
            reg_idx: Register index to decrement (must not be 0)
        """
        if reg_idx == 0:
            print("ERROR: INSTRUCTION_ADD overwriting reg_idx 0 (zero reg) not allowed")
            return
        self.ue_isa_descriptor(INSTRUCTION_ADD, isa_mode=INST_ADD_DEC,
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
            print("ERROR: INSTRUCTION_ADD overwriting reg_idx 0 (zero reg) not allowed")
            return
        self.ue_isa_descriptor(INSTRUCTION_ADD, isa_mode=INST_ADD_REG,
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
            print("ERROR: INSTRUCTION_ADD overwriting reg_idx 0 (zero reg) not allowed")
            return
        if dst_reg_idx is None:
            dst_reg_idx = src_reg_idx
        self.ue_isa_descriptor(INSTRUCTION_ADD, immediate_value=immediate_value,
                               isa_mode=INST_ADD_IMM, src_reg_idx=src_reg_idx,
                               dst_reg_idx=dst_reg_idx)

    def generate_instruction_add_set(self, dst_reg_idx: int, immediate_value: int):
        """
        Generate an ADD instruction to set register to immediate value

        Args:
            dst_reg_idx: Destination register index (must not be 0)
            immediate_value: 32-bit immediate value to set
        """
        if dst_reg_idx == 0:
            print("ERROR: INSTRUCTION_ADD overwriting reg_idx 0 (zero reg) not allowed")
            return
        self.ue_isa_descriptor(INSTRUCTION_ADD, immediate_value=immediate_value,
                               isa_mode=INST_ADD_SET, src_reg_idx=dst_reg_idx,
                               dst_reg_idx=dst_reg_idx)

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

    def generate_instruction_pbi_init(
        self,
        dram_shared_addr: int,
        dma_length: int,
        output_size: int,
        uram_length: int,
        uram_a_start_addr: int,
        uram_b_start_addr: int,
        uram_wb_addr: int,
        uram_dst_addr: int,
        fmax_context_addr: int,
        inst_pointer_idx: int,
    ) -> None:
        """
        PBI init via :meth:`ue_op_descriptor` with ``inst_type=INSTRUCTION_PBI_SET`` and ``inst_pointer_idx``
        in ``w[0][15:12]``. Remaining possible mismatch vs sparse C tail: ``w[5][31:12]`` (UE mode bits),
        ``w[7][21:10]`` vs compute-path mux when ``uram_dst_addr`` is nonzero.
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
            general_reg_src=0,
            fmax_context_addr=fmax_context_addr,
        )

    def generate_instruction_pbi_inc(
        self,
        dram_shared_addr: int,
        dma_length: int,
        output_size: int,
        uram_length: int,
        uram_a_start_addr: int,
        uram_b_start_addr: int,
        uram_wb_addr: int,
        uram_dst_addr: int,
        fmax_context_addr: int,
        inst_pointer_idx: int,
    ) -> None:
        """
        PBI increment-only update via :meth:`ue_op_descriptor` with ``inst_type=INSTRUCTION_PBI_SET``,
        ``inst_pointer_idx`` in ``w[0][15:12]``, and ``pbi_mode=PBI_MODE_INC`` in ``w[0][19:16]``.
        """
        if self.capture_buffer is None:
            print("ERROR: generate_instruction_pbi_inc() called but capture_buffer is not initialized!")
            return
        if self.capture_count >= MAX_DECODER_INSTRUCTIONS:
            print(f"ERROR: generate_instruction_pbi_inc() capture_count >= MAX ({MAX_DECODER_INSTRUCTIONS})!")
            return

        self.ue_op_descriptor(
            inst_type=INSTRUCTION_PBI_SET,
            inst_pointer_idx=inst_pointer_idx,
            pbi_mode=PBI_MODE_INC,
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
            general_reg_src=0,
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

    def dram_inst_running(self, enable: bool = True):
        """
        Run instructions from DRAM one by one
        """
        self.write_reg32(UE_INSTRUCTION_CTL_ADDR, 1 if enable else 0)

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
