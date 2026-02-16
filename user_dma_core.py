"""
Core DMA/engine ops: constants, register definitions, instruction-level APIs.

This module provides:
- Register address and constant definitions (USER_BASE_ADDR, UE_*_ADDR, etc.)
- Enums: UE_MODE, TYPE, URAM_SECTION, LALU_MODE, etc.
- Instructions, DeviceTensor, UnifiedEngine
- Low-level ops: start_queue, wait_queue, dma_read/dma_write, user_read_reg32, etc.
- High-level engine APIs built on them (eltwise_add, start_queue, matmat_mul, ...)

For tests and bf16 helpers (precompute_freqs_cis, calculate_snr, generic_tests),
use user_dma_tests_bf16 or the combined user_dma_ops.

Usage:
    from user_dma_core import UnifiedEngine, UE_MODE, DRAM_ACTIVATION_ADDR
    ue = UnifiedEngine(device='cpu')
    handler = ue.eltwise_add(size, input_a_dram_addr, input_b_dram_addr, output_dram_addr)
    result = handler(vec1, vec2)
"""

import struct
import os
import time
from typing import Optional, Tuple, Callable
from enum import IntEnum
import torch
import math

# Register address definitions (matches andromeda.c)
# User device base address (AXI-Lite BAR mapping)
USER_BASE_ADDR = 0x30000000
UE_FPGA_VERSION_ADDR = 0x30000000
UE_ISA_CTRL = 0x30000004  # Reserved for ISA debug
UE_DRAM_ADDR = 0x30000008  # Unified DMA address (bits [63:32])
UE_DMA_LENGTH_ADDR = 0x3000000C
UE_CONTROL_ADDR = 0x30000010
UE_STATUS_ADDR = 0x30000014
UE_OUTPUT_VALID_DELAY_ADDR = 0x3000001C
UE_URAM_LENGTH_ADDR = 0x30000020
UE_URAM_WRITEBACK_ADDR = 0x30000024
UE_LATENCY_COUNT_ADDR = 0x30000030
UE_DRAM_URAM_CTRL_ADDR = 0x30000034
UE_URAM_ROW_SIZE_ADDR = 0x30000040
UE_VALID_DELAY_EXTRA_ADDR = 0x30000044
UE_LALU_INST_ADDR = 0x30000048
UE_LALU_DELAY_ADDR = 0x3000004C
UE_SCALAR_ADDR = 0x30000050
UE_QUEUE_CTRL_ADDR = 0x30000054
UE_QUEUE_SIZE_ADDR = 0x30000058
UE_URAM_LENGTH_ADDR_Z = 0x3000005C
UE_BIAS_ADDER_EN_ADDR = 0x30000060
UE_URAM_WB_PADDING_ADDR = 0x30000064
UE_BROADCAST_MODE_ADDR = 0x30000068
UE_ARGMAX_INDEX = 0x3000006C
UE_INSTRUCTION_CTL_ADDR = 0x30000070
UE_TOTAL_BYTES_PER_STRIDE = 0x30000074
UE_INSTRUCTION_ADDR = 0x30000078
UE_TOTAL_STRIDE_BYTES = 0x3000007C
UE_FMAX_CONTEXT_ADDR = 0x30000080
UE_LAST_REG_ADDR = 0x30000080  # last reg address (same as FMAX for init loop)
# CLOCK_CYCLE_TIME_NS will be set based on target argument (default: 3, alveo: 2.5)
CLOCK_CYCLE_TIME_NS = 3

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
UE_FMAX_CONTEXT_SIZE = 64
SCALE_BRAM_ELEMENTS = 8192
SCALE_BRAM_SIZE_BYTES = SCALE_BRAM_ELEMENTS * 2
BIAS_BRAM_ELEMENTS = 8192
BIAS_BRAM_SIZE_BYTES = BIAS_BRAM_ELEMENTS * 2
DRAM_START_ADDR = 0x80000000 # 0 GB
DRAM_ACTIVATION_ADDR = 0xB0000000 # 512 MB reserved for intermediate results
DRAM_INSTRUCTION_ADDR = 0xD0000000  # 256*3 MB reserved for instructions

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

class TYPE(IntEnum):
    INT4 = 0
    INT8 = 1
    FP4 = 2
    BF16 = 3

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
    GELU = 1
    MODE_RECIP = 2
    MODE_RSQRT = 3
    SILU = 4

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
# UE_LATENCY_BF20_ITR2 = UE_PIPELINE_BF20_ADD + 1 - 1  # 2 + 1 - 1 = 2
# UE_LATENCY_BF20_ITR3 = 2*UE_PIPELINE_BF20_ADD + 2 - 1  # 2 * 2 + 2 - 1 = 5
# UE_LATENCY_BF20_ITRGT3 = 3*UE_PIPELINE_BF20_ADD + 2 - 1  # 3 * 2 + 2 - 1 = 7

# Pipeline component latencies from bf19_mult.vhdl, bf19_add.vhdl, custom_exp.vhdl, adder_tree.vhdl
UE_PIPELINE_BF19_MULT = 2
UE_PIPELINE_BF19_ADD = 3
UE_PIPELINE_CUSTOM_EXP = 4
UE_PIPELINE_ADDER_TREE = 21
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
UE_LATENCY_DOT_PRODUCT = UE_PIPELINE_STAGES_INPUT_REG + UE_PIPELINE_STAGES_DOT_P + UE_PIPELINE_BF19_MULT + UE_PIPELINE_ADDER_TREE - 1  # 1 + 4 + 1 + 16 - 1 = 21
UE_LATENCY_RMS = UE_PIPELINE_STAGES_RMS + UE_PIPELINE_BF19_MULT + UE_PIPELINE_ADDER_TREE - 1  # 4 + 1 + 16 - 1 = 20
UE_LATENCY_EXP = UE_PIPELINE_STAGES_EXP + UE_PIPELINE_BF19_ADD + UE_PIPELINE_CUSTOM_EXP + UE_PIPELINE_ADDER_TREE - 1  # 4 + 2 + 3 + 16 - 1 = 24
UE_LATENCY_ADD_REDUCE = UE_PIPELINE_STAGES_ADD + UE_PIPELINE_BF19_ADD + UE_PIPELINE_ADDER_TREE - 1  # 3 + 2 + 16 - 1 = 20
UE_LATENCY_ELTWISE_MUL = UE_PIPELINE_STAGES_MULT + UE_PIPELINE_BF19_MULT - 1  # 4 + 1 - 1 = 4
UE_LATENCY_ELTWISE_ADD = UE_PIPELINE_STAGES_ADD + UE_PIPELINE_BF19_ADD - 1  # 3 + 2 - 1 = 4
UE_LATENCY_ADD_EXP = UE_PIPELINE_STAGES_EXP + UE_PIPELINE_BF19_ADD + UE_PIPELINE_CUSTOM_EXP - 1  # 4 + 2 + 3 - 1 = 8
UE_LATENCY_ROPE = 8  # Additional mode latency
# Legacy latency values as a reference
# UE_LATENCY_QUANTIZATION = 18  # Additional mode latency
# UE_LATENCY_QSCALE = 12  # Additional mode latency
UE_LATENCY_MEAN = 20  # Additional mode latency

# LALU pipeline component latencies from timing.md (micro values)
UE_LALU_PIPELINE_FPDIV = 3  # fpdiv pipeline depth (from fpdiv.vhdl line 561)
UE_LALU_PIPELINE_FPSQRT = 2  # fpsqrt pipeline depth (from fpsqrt.vhdl line 9)
UE_LALU_PIPELINE_GELU_DENOM = 6  # gelu_denom pipeline depth (from gelu_denom.vhdl line 3981)
UE_LALU_PIPELINE_SILU_DENOM = 5  # silu_denom pipeline depth (from silu_denom.vhdl line 3810)

# LALU mode latencies calculated from timing.md formulas (shift register delay parameter)
# Pipeline stages are all 1 cycle (Input Reg + intermediate Reg + Output Reg - 1 for overlap)
UE_LALU_LATENCY_SOFTMAX = 1 + UE_LALU_PIPELINE_FPDIV  # RECIP: 1 + 3 = 4
UE_LALU_LATENCY_RMS = 1 + UE_LALU_PIPELINE_FPSQRT + 1 + UE_LALU_PIPELINE_FPDIV  # RSQRT: 1 + 2 + 1 + 3 = 7
UE_LALU_LATENCY_GELU = 1 + UE_LALU_PIPELINE_GELU_DENOM + 1 + UE_LALU_PIPELINE_FPDIV  # GELU: 1 + 6 + 1 + 3 = 11
UE_LALU_LATENCY_SILU = 1 + UE_LALU_PIPELINE_SILU_DENOM + 1 + UE_LALU_PIPELINE_FPDIV  # SILU: 1 + 5 + 1 + 3 = 10
UE_LALU_LATENCY_MULT = 2  # Additional LALU mode latency

# Quantization latency values (matching andromeda.c)
UE_QUANTIZE_FMAX_PIPELINE = 7
UE_LATENCY_QSCALE = UE_LALU_PIPELINE_FPDIV + UE_QUANTIZE_FMAX_PIPELINE + 2  # 12
UE_QINPUT_DELAY = UE_LATENCY_QSCALE + 1  # 13
UE_LATENCY_QUANTIZATION = UE_LATENCY_QSCALE + UE_PIPELINE_BF19_MULT + 5  # 19

# ISA instruction type constants (matching andromeda.c)
INSTRUCTION_JUMP = 1
INSTRUCTION_ADD = 2
INSTRUCTION_REG_REWRITE = 3
INSTRUCTION_HALT = 7

# JUMP mode constants
JUMP_MODE_ABSOLUTE = 0
JUMP_MODE_RELATIVE = 1
JUMP_MODE_JNZ = 2  # Jump if Not Zero
JUMP_MODE_JZ = 3   # Jump if Zero

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

class Instructions:
    """
    256-bit instruction descriptor (32 bytes exactly, matches RTL FIFO width)
    Bit layout matches axi_top.sv QUEUE_LOAD state
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
    def __init__(self, device: str = 'cpu'):
        self.device = device

        # Hardware state (no simulation - uses actual DMA)
        # Note: DRAM/URAM/BRAM are accessed via DMA, not stored locally

        # Latency tracking
        self.latency_count = 0

        # Decoder instruction capture state
        self.capture_buffer: list[Instructions] = []
        self.capture_count = 0
        self.is_capture_on = False

        # Incremental memory allocation tracking for auto-allocation
        # Params region starts from DRAM_START_ADDR
        self._next_params_dram_addr = DRAM_START_ADDR
        # Program/instruction region starts from DRAM_INSTRUCTION_ADDR
        self._next_program_dram_addr = DRAM_INSTRUCTION_ADDR

        self._tensor_dram_addr = DRAM_ACTIVATION_ADDR

        self._inst_id = 0

        # Initialize
        self.init_unified_engine()

    def allocate_params_dram(self, size_bytes: int):
        """
        Allocate memory from the params DRAM region incrementally.

        Args:
            size_bytes: Number of bytes to allocate
        """
        self._next_params_dram_addr += size_bytes

    def allocate_program_dram(self, size_bytes: int):
        """
        Allocate memory from the program/instruction DRAM region incrementally.

        Args:
            size_bytes: Number of bytes to allocate
        """
        self._next_program_dram_addr += size_bytes

    def get_params_dram_addr(self) -> int:
        return self._next_params_dram_addr

    def get_program_dram_addr(self) -> int:
        return self._next_program_dram_addr

    def get_params_dram_usage(self) -> int:
        """Get the total params DRAM memory used so far."""
        return self._next_params_dram_addr - DRAM_START_ADDR

    def get_program_dram_usage(self) -> int:
        """Get the total program DRAM memory used so far."""
        return self._next_program_dram_addr - DRAM_INSTRUCTION_ADDR

    def reset_params_dram_addr(self):
        self._next_params_dram_addr = DRAM_START_ADDR

    def reset_program_dram_addr(self):
        self._next_program_dram_addr = DRAM_INSTRUCTION_ADDR

    def allocate_tensor_dram(self, size_bytes: int) -> int:
        """Allocate memory from the tensor DRAM region incrementally."""
        tensor_dram_addr = self._tensor_dram_addr
        self._tensor_dram_addr += size_bytes
        return tensor_dram_addr

    def get_tensor_dram_usage(self) -> int:
        """Get the total tensor DRAM memory used so far."""
        return self._tensor_dram_addr - DRAM_ACTIVATION_ADDR

    def reset_tensor_dram_addr(self):
        self._tensor_dram_addr = DRAM_ACTIVATION_ADDR

    def get_tensor_dram_addr(self) -> int:
        return self._tensor_dram_addr

    def chunk_ranges(self, total: int, chunk_size: int):
        """Yield (start_index, chunk_size) for each chunk. Last chunk may be smaller."""
        start = 0
        while start < total:
            take = min(chunk_size, total - start)
            yield start, take
            start += take

    def init_unified_engine(self):
        # Test user device register access first
        print(f"{DMA_DEVICE_USER} register access...")
        hw_version = self.user_read_reg32(UE_FPGA_VERSION_ADDR)
        print(f"HW version via user device: 0x{hw_version & 0xFFFFFFFF:08x}")
        #assert hw_version == 0x2ae8dae9, "HW version is not 0x2ae8dae9 please check the FPGA version"

        addr = USER_BASE_ADDR
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

        # Configure delay for the last ALU
        ue_lalu_delay = (
            (UE_QINPUT_DELAY << 25) +
            (UE_LATENCY_QUANTIZATION << 20) +
            (UE_LATENCY_QSCALE << 16) +
            (UE_LALU_LATENCY_SILU << 12) +
            (UE_LALU_LATENCY_GELU << 8) +
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
                offset = address - USER_BASE_ADDR
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
                offset = address - USER_BASE_ADDR
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
        # Check bit [8] and bits [11:8] - queue busy is when ((queue_reg >> 8) & 0xF) == 1
        return ((queue_reg >> 8) & 0xF) == 1

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

    def ue_memcpy_from_dram(self, dram_src_addr: int, memcpy_length_bytes: int,
                            memcpy_type: int, uram_dst_addr: int,
                            uram_type: int, inst_id: int,
                            stride_bytes_per_chunk: int = 0,
                            stride_jump_bytes: int = 0,
                            general_reg_src: int = 0,
                            fmax_context_addr: int = 0):
        """
        Memory copy from DRAM to URAM/BRAM

        Supports stride mode: If stride mode is enabled via set_stride_mode(), the DMA
        will copy data in strides. The stride configuration is preserved during the
        operation.

        Args:
            dram_src_addr: Source address in DRAM
            memcpy_length_bytes: Number of bytes to copy
            memcpy_type: Type of memory (MEMCPY_TYPE.URAM, BRAM, BIAS_BRAM, SCALE_BRAM)
            uram_dst_addr: Destination address in URAM (only meaningful for URAM type)
            uram_type: URAM section (URAM_SECTION.URAM_A or URAM_B, only meaningful for URAM type)
            inst_id: Instruction ID
            stride_bytes_per_chunk: Bytes to copy per stride (0 = no stride mode)
            stride_jump_bytes: Distance in bytes between start of consecutive copies in DRAM
                              (0 = contiguous, use stride_bytes_per_chunk for the jump)
            general_reg_src: General purpose register source (default: 0)
            fmax_context_addr: Fmax context address packed in inst_reserved_2 (bits 15-20, 6 bits, default: 0)
        """
        # Capture mode: pack instruction instead of executing
        if not self.is_capture_on:
            # Clear remaining config registers
            # Virtual mode=4'b1111 for memcpy from dram to uram
            control_value = 0xF << 11
            self.write_reg32(UE_LALU_INST_ADDR, LALU_MODE.BYPASS.value)

            if stride_jump_bytes > 0:
                self.write_reg32(UE_TOTAL_STRIDE_BYTES, (1 << 21) | stride_bytes_per_chunk)
                self.write_reg32(UE_TOTAL_BYTES_PER_STRIDE, stride_jump_bytes)
            else:
                self.write_reg32(UE_TOTAL_STRIDE_BYTES, 0)
                self.write_reg32(UE_TOTAL_BYTES_PER_STRIDE, 0)

            self.write_reg32(UE_SCALAR_ADDR, 0)
            self.write_reg32(UE_CONTROL_ADDR, control_value)
            self.write_reg32(UE_DRAM_ADDR, dram_src_addr)
            self.write_reg32(UE_DMA_LENGTH_ADDR, memcpy_length_bytes)
            self.write_reg32(UE_URAM_LENGTH_ADDR, 0)
            self.write_reg32(UE_URAM_ROW_SIZE_ADDR, 0)
            self.write_reg32(UE_URAM_LENGTH_ADDR_Z, 0)
            self.write_reg32(UE_URAM_WB_PADDING_ADDR, 0)

            self.write_reg32(UE_URAM_WRITEBACK_ADDR, 0)  # Writeback address to URAM
            uram_ctrl = ((memcpy_type << 16) | (uram_dst_addr << 4) |
                        (URAM_WRITE_SRC.URAM_DRAM.value << 2) |
                        ((uram_type & 0x1) << 1) | 1)
            self.write_reg32(UE_DRAM_URAM_CTRL_ADDR, uram_ctrl)
            # Trigger queue
            self.write_reg32(UE_QUEUE_CTRL_ADDR, inst_id)
        else:
            # === CAPTURE: Pack into 256-bit instruction descriptor ===
            if self.capture_buffer is not None and self.capture_count < MAX_DECODER_INSTRUCTIONS:
                inst = Instructions()
                w = inst.words

                # Clear instruction first (all zeros - matches explicit zeroing in function)
                for i in range(8):
                    w[i] = 0

                # Memory copy stride mode is enabled if stride_bytes_per_chunk > 0
                stride_en = 1 if stride_jump_bytes > 0 else 0

                # Pack only the registers that are set for MEMCPY operation
                # Word 0 (bits 0-31): ISA control fields in [15:0], reserved in [31:16]
                inst_type = 0
                if general_reg_src != 0:
                    w[0] = ((0 & 0xF) << 0) | \
                           ((general_reg_src & 0xF) << 4) | \
                           ((0 & 0xF) << 8) | \
                           ((0 & 0xF) << 12)
                    inst_type = INSTRUCTION_REG_REWRITE
                else:
                    w[0] = 0  # ISA fields not used for regular memcpy
                    inst_type = 0
                
                # Word 1 (bits 32-63): unified DMA address (replaces separate source/target)
                w[1] = dram_src_addr
                
                w[2] = memcpy_length_bytes  # Word 2: dma_length
                # Word 3: mode_sel(4-7)=0xF (virtual mode), output_size(8-23)=stride_bytes_per_chunk
                w[3] = ((0xF << 4) |
                        ((stride_bytes_per_chunk & 0xFFFF) << 8))  # inst_output_size for bytes per stride
                w[4] = 0  # Word 4: All zeros
                # Word 5: start_memcpy_reg(168), uram_select(169), dr_mb_wb_mode(170-171),
                #         uram_memcpy_dst_addr(172-183), bram_uram_select(184-185),
                #         lalu_mode(186-188), lalu_scalar[2:0](189-191)
                w[5] = ((1 << 8) |  # start_memcpy_reg = 1
                        ((uram_type & 1) << 9) |  # uram_select
                        ((URAM_WRITE_SRC.URAM_DRAM.value & 3) << 10) |  # dr_mb_wb_mode = URAM_DRAM
                        ((uram_dst_addr & 0xFFF) << 12) |  # uram_memcpy_dst_addr
                        ((memcpy_type & 3) << 24) |  # bram_uram_select = memcpy_type
                        ((stride_jump_bytes & 0x7) << 29))  # lalu_scalar[2:0] = stride_jump_bytes lower 3 bits
                # Word 6: lalu_scalar[20:3](192-209), uram_row_size_z(210-221),
                #         uram_dram_wb_start_reg(222), wb_padding_sel(223)
                w[6] = ((stride_jump_bytes >> 3) & 0x3FFFF)  # lalu_scalar[20:3] = stride_jump_bytes upper 18 bits
                # Word 7 (bits 224-255): fmax_clear(224), inst_trans_row_cnt(225-234), inst_trans_col_cnt(235-244),
                #                      broadcast_mode(245-246), transaction_id(247-252), inst_type(253-255)
                w[7] = ((0 & 1) |                           # bit 0: fmax_clear = 0
                        (0 << 1) |                          # bits 1: use bias adder = 0
                        (0 << 2) |                          # bits 2-13: uram_row_stride_z = 0
                        (stride_en << 14) |                 # bit 14: dma_stride_en = stride_en
                        (fmax_context_addr << 15) |  # bits 15-20: inst_reserved_2 = fmax_context address
                        (0 << 21) |                         # bits 21-22: broadcast_mode = 0
                        ((inst_id & 0x3F) << 23) |          # bits 23-28: transaction_id (6 bits)
                        ((inst_type & 0x7) << 29))          # bits 29-31: inst_type (3 bits)

                self.capture_buffer.append(inst)
                self.capture_count += 1

    def ue_memcpy_to_dram(self, memcpy_type: int, uram_type: int,
                         uram_src_addr: int, dram_dst_addr: int,
                         memcpy_length_bytes: int, inst_id: int,
                         stride_bytes_per_chunk: int = 0,
                         stride_jump_bytes: int = 0,
                         general_reg_src: int = 0):
        """
        Memory copy from URAM to DRAM

        Supports stride mode: When stride parameters are provided, the DMA will write
        data in strides to non-contiguous DRAM locations.

        Args:
            memcpy_type: Type of memory (MEMCPY_TYPE.URAM, BRAM, etc.)
            uram_type: URAM section (URAM_SECTION.URAM_A or URAM_B)
            uram_src_addr: Source address in URAM
            dram_dst_addr: Destination address in DRAM
            memcpy_length_bytes: Total number of bytes to copy (should be multiple of 128)
            inst_id: Instruction ID
            stride_bytes_per_chunk: Bytes to write per stride (0 = no stride mode)
            stride_jump_bytes: Distance in bytes between start of consecutive writes in DRAM
                              (0 = contiguous, use stride_bytes_per_chunk for the jump)
            general_reg_src: General purpose register source (default: 0)

        Example (stride writeback):
            # Write 2 rows (128 bytes each) with 256 byte gaps between them in DRAM
            # Row 0 -> DRAM[0:128], Row 1 -> DRAM[256:384]
            ue.set_stride_mode(128, 256, enable=True)  # Enable stride mode first
            ue.ue_memcpy_to_dram(MEMCPY_TYPE.URAM, URAM_A, 0, dst_addr, 256, 0,
                                 stride_bytes_per_chunk=128, stride_jump_bytes=256)
            ue.wait_queue()
            ue.clear_stride_mode()
        """
        round_length_bytes = (memcpy_length_bytes + 127) & ~127

        # Use start_queue with URAM_DRAM_WRITEBACK mode
        self.start_queue(
            0,  # broadcast_mode
            0,  # clear_max_en
            1,  # stride_z
            LALU_MODE.BYPASS.value,  # lalu_mode
            0,  # scalar
            memcpy_type,  # uram_bram (URAM/BRAM type)
            uram_type,  # uram_section (URAM_A/B)
            0,  # uram_dst_addr
            0,  # dram_to_uram_cpy_start
            0,  # uram_wb_addr
            URAM_WRITE_SRC.URAM_DRAM.value,  # uram_write_src
            UE_MODE.URAM_DRAM_WRITEBACK,  # mode
            0,  # data_type
            uram_src_addr,  # uram_a_start_addr
            uram_src_addr,  # uram_b_start_addr
            round_length_bytes // UE_VECTOR_SIZE // 2,  # uram_length (bram_length)
            dram_dst_addr,  # dma_start_addr
            0,  # dma_length
            0,  # output_size
            inst_id,  # inst_id
            0,  # bias_adder_en
            stride_bytes_per_chunk,  # stride_bytes_per_chunk
            stride_jump_bytes,  # stride_jump_bytes
            general_reg_src  # general_reg_src
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
        inst_id = 0

        remaining_bytes = memcpy_length_bytes
        aligned_max_chunk_bytes = (URAM_NEAR_FULL_SIZE // stride_bytes_per_chunk) * stride_bytes_per_chunk
        input_dram_addr_offset = input_dram_addr
        output_dram_addr_offset = output_dram_addr
        while remaining_bytes > 0:
            chunk_bytes = min(remaining_bytes, aligned_max_chunk_bytes)

            stride_enabled_way = True
            if stride_enabled_way:
                self.ue_memcpy_from_dram(input_dram_addr_offset, chunk_bytes, MEMCPY_TYPE.URAM, URAM_START_ADDR, URAM_SECTION.URAM_A.value, inst_id, stride_bytes_per_chunk, stride_jump_bytes)
                self.wait_queue()
                inst_id += 1

                self.ue_memcpy_to_dram(MEMCPY_TYPE.URAM.value, URAM_SECTION.URAM_A.value, URAM_START_ADDR, output_dram_addr_offset, chunk_bytes, inst_id)
                self.wait_queue()
                inst_id += 1
            else:
                print(f"memcpy_length_bytes: {memcpy_length_bytes}")
                print(f"stride_bytes_per_chunk: {stride_bytes_per_chunk}")
                print(f"stride_jump_bytes: {stride_jump_bytes}")
                for i in range(memcpy_length_bytes // stride_bytes_per_chunk):
                    self.ue_memcpy_from_dram(input_dram_addr_offset + i * stride_jump_bytes, stride_bytes_per_chunk, MEMCPY_TYPE.URAM, URAM_START_ADDR + (i * stride_bytes_per_chunk) // UE_VECTOR_SIZE, URAM_SECTION.URAM_A.value, inst_id)
                    self.wait_queue()
                    inst_id += 1

                self.ue_memcpy_to_dram(MEMCPY_TYPE.URAM.value, URAM_SECTION.URAM_A.value, URAM_START_ADDR, output_dram_addr_offset, memcpy_length_bytes, inst_id)
                self.wait_queue()
                inst_id += 1

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
            print( f"flops: {self.report_flop_rate_gflops(memcpy_length_bytes / bytes_per_element):.3f} GFLOPS")

            if isinstance(matrix, DeviceTensor):
                return DeviceTensor((memcpy_length_bytes // stride_bytes_per_chunk, stride_bytes_per_chunk // bytes_per_element), ue=self, dram_addr=output_dram_addr)
            else:
                return DeviceTensor((memcpy_length_bytes // stride_bytes_per_chunk, stride_bytes_per_chunk // bytes_per_element), ue=self, dram_addr=output_dram_addr).data

        return handler

    def start_queue(self, broadcast_mode: int, max_clear_en: int, stride_z: int, lalu_mode: int, scalar: int,
                   uram_bram: int, uram_section: int, uram_dst_addr: int,
                   dram_to_uram_cpy_start: int, uram_wb_addr: int,
                   uram_write_src: int, mode: UE_MODE, data_type: int,
                   uram_a_start_addr: int, uram_b_start_addr: int,
                   uram_length: int, dma_start_addr: int, dma_length: int,
                   output_size: int, inst_id: int, bias_adder_en: int = 0,
                   stride_bytes_per_chunk: int = 0, stride_jump_bytes: int = 0,
                   general_reg_src: int = 0, fmax_context_addr: int = 0
                   ) -> Optional[torch.Tensor]:
        """
        Start queue operation - configures registers and executes operation

        This is the low-level function that configures the hardware and triggers
        operations. It stores configuration in registers and executes based on mode.

        Returns:
            Result tensor if operation produces output, None otherwise
        """
        # Configure UE engine
        # Match andromeda.c: uram_start excludes DOT_PRODUCT, URAM_DRAM_WRITEBACK, and DEQUANTIZE
        uram_start = int((mode != UE_MODE.DOT_PRODUCT) and
                        (mode != UE_MODE.URAM_DRAM_WRITEBACK) and
                        (mode != UE_MODE.DEQUANTIZE))
        uram_bram_wb_start = int(mode == UE_MODE.URAM_DRAM_WRITEBACK)
        # Match andromeda.c: dma_start for DOT_PRODUCT or DEQUANTIZE
        dma_start = int((mode == UE_MODE.DOT_PRODUCT) or (mode == UE_MODE.DEQUANTIZE))
        uram_length_z = dma_length >> 6  # in 64 element units

        # TODO: make this a parameter
        #define WB_PADDING_ZERO         0 // 0x0000
        #define WB_PADDING_NEG_INF      1 // 0xFF80
        wb_padding_select = 1 # TODO: make this a parameter

        wb_padding_control = 0
        if mode == UE_MODE.BF16_DOT_PRODUCT or mode == UE_MODE.DOT_PRODUCT:
            if max_clear_en == 1:
                wb_padding_control |= (1 << 1)
            if mode == UE_MODE.BF16_DOT_PRODUCT:
                wb_padding_control |= ((wb_padding_select & 1) << 0)  # 0 → use 0x0000 padding

        # Capture mode: pack instruction instead of writing to queue
        if not self.is_capture_on:
            # Configure broadcast mode
            self.write_reg32(UE_BROADCAST_MODE_ADDR, broadcast_mode)

            # Configure last ALU
            self.write_reg32(UE_LALU_INST_ADDR, lalu_mode)
            self.write_reg32(UE_SCALAR_ADDR, scalar)

            if stride_jump_bytes > 0:
                self.write_reg32(UE_TOTAL_STRIDE_BYTES, (1 << 21) | stride_bytes_per_chunk)
                self.write_reg32(UE_TOTAL_BYTES_PER_STRIDE, stride_jump_bytes)
            else:
                self.write_reg32(UE_TOTAL_STRIDE_BYTES, 0)
                self.write_reg32(UE_TOTAL_BYTES_PER_STRIDE, 0)

            # Configure URAM
            self.write_reg32(UE_URAM_WRITEBACK_ADDR, uram_wb_addr)
            uram_ctrl = ((uram_bram << 16) | (uram_dst_addr << 4) |
                        (uram_write_src << 2) | ((uram_section & 0x1) << 1) |
                        dram_to_uram_cpy_start)
            self.write_reg32(UE_DRAM_URAM_CTRL_ADDR, uram_ctrl)


            control_value = ((output_size << 16) | (mode.value << 11) |
                            (data_type << 8) | (uram_start << 4) |
                            (uram_bram_wb_start << 2) | dma_start)

            # Set DMA start address and length
            if dma_start:
                self.write_reg32(UE_DRAM_ADDR, dma_start_addr)
                self.write_reg32(UE_DMA_LENGTH_ADDR, dma_length)
            if uram_bram_wb_start:
                self.write_reg32(UE_DRAM_ADDR, dma_start_addr)
                # writeback length is defined by uram_length

            self.write_reg32(UE_URAM_LENGTH_ADDR,
                            (uram_b_start_addr << 12) | uram_a_start_addr)
            self.write_reg32(UE_URAM_ROW_SIZE_ADDR, uram_length)
            self.write_reg32(UE_URAM_LENGTH_ADDR_Z, (stride_z << 12) | uram_length_z)
            self.write_reg32(UE_CONTROL_ADDR, control_value)

            self.write_reg32(UE_URAM_WB_PADDING_ADDR, wb_padding_control)
            # Configure bias adder enable (matching andromeda.c)
            self.write_reg32(UE_BIAS_ADDER_EN_ADDR, bias_adder_en)
            # Trigger queue operation
            # Hardware will execute the operation based on the configured registers
            self.write_reg32(UE_QUEUE_CTRL_ADDR, inst_id)
        else:
            # === CAPTURE: Pack into 256-bit instruction descriptor ===
            if self.capture_buffer is not None and self.capture_count < MAX_DECODER_INSTRUCTIONS:
                inst = Instructions()
                w = inst.words

                stride_en = 1 if stride_jump_bytes > 0 else 0
                output_size = stride_bytes_per_chunk if stride_en else output_size
                scalar = stride_jump_bytes if stride_en else scalar

                # Pack according to RTL bit layout (axi_top.sv lines 928-951 + inst_type)
                # Word 0 (bits 0-31): ISA control fields in [15:0], reserved in [31:16]
                inst_type = 0
                if general_reg_src != 0:
                    w[0] = ((0 & 0xF) << 0) | \
                           ((general_reg_src & 0xF) << 4) | \
                           ((0 & 0xF) << 8) | \
                           ((0 & 0xF) << 12)
                    inst_type = INSTRUCTION_REG_REWRITE
                else:
                    w[0] = 0  # ISA fields not used for regular memcpy
                    inst_type = 0

                # Word 1 (bits 32-63): unified DMA address (replaces separate source/target)
                w[1] = (dma_start_addr if (dma_start or uram_bram_wb_start) else 0)

                # Word 2 (bits 64-95): dma_length
                w[2] = dma_length if dma_start else 0

                # Word 3 (bits 96-127): dma_start_reg(96), uram_start_reg(97), data_type(98-99),
                #                      mode_sel(100-103), output_size(104-119), uram_row_size[7:0](120-127)
                w[3] = ((dma_start & 1) |
                        ((uram_start & 1) << 1) |
                        ((data_type & 3) << 2) |
                        ((mode.value & 0xF) << 4) |
                        ((output_size & 0xFFFF) << 8) |
                        ((uram_length & 0xFF) << 24))

                # Word 4 (bits 128-159): uram_row_size[11:8](128-131), uram_start_addr_y(132-143),
                #                      uram_start_addr_z(144-155), uram_writeb_addr[3:0](156-159)
                w[4] = (((uram_length >> 8) & 0xF) |
                        ((uram_a_start_addr & 0xFFF) << 4) |
                        ((uram_b_start_addr & 0xFFF) << 16) |
                        ((uram_wb_addr & 0xF) << 28))

                # Word 5 (bits 160-191): uram_writeb_addr[11:4](160-167), start_memcpy_reg(168),
                #                      uram_select(169), dr_mb_wb_mode(170-171), uram_memcpy_dst_addr(172-183),
                #                      bram_uram_select(184-185), lalu_mode(186-188), lalu_scalar[2:0](189-191)
                w[5] = (((uram_wb_addr >> 4) & 0xFF) |
                        ((dram_to_uram_cpy_start & 1) << 8) |
                        ((uram_section & 1) << 9) |
                        ((uram_write_src & 3) << 10) |
                        ((uram_dst_addr & 0xFFF) << 12) |
                        ((uram_bram & 3) << 24) |
                        ((lalu_mode & 0x7) << 26) |
                        ((scalar & 0x7) << 29))

                # Word 6 (bits 192-223): lalu_scalar[20:3](192-209), uram_row_size_z(210-221),
                #                      uram_dram_wb_start_reg(222), wb_padding_sel(223)
                w[6] = (((scalar >> 3) & 0x3FFFF) |
                        ((uram_length_z & 0xFFF) << 18) |
                        ((uram_bram_wb_start & 1) << 30) |
                        ((wb_padding_control & 1) << 31))

                
                # Word 7 (bits 224-255): fmax_clear(224), use_bf19_bias_adder(225), uram_row_stride_z(226-237),
                #                      dma_stride_en(238), reserved_2(239-244)=fmax_context_addr, broadcast_mode(245-246), transaction_id(247-252), inst_type(253-255)
                w[7] = ((max_clear_en & 1) |
                        ((bias_adder_en & 1) << 1) |
                        ((stride_z & 0xFFF) << 2) |
                        ((stride_en & 0x1) << 14) |         # bit 14: dma_stride_en
                        (fmax_context_addr << 15) |  # bits 15-20: reserved_2 = fmax_context address
                        ((broadcast_mode & 0x3) << 21) |
                        ((inst_id & 0x3F) << 23) |
                        ((inst_type & 0x7) << 29))

                self.capture_buffer.append(inst)
                self.capture_count += 1

        # Note: Operations are executed by hardware, not simulated
        # Queue busy state is checked via is_queue_busy() which reads the hardware register
        # Results are read back via DMA when needed
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
    # Wrappers for UE eltwise and broadcast ops (wrap start_queue with fixed mode)
    # Used by user_dma_ops.py eltwise_op, eltwise_add, eltwise_mul, broadcast_op.
    # -------------------------------------------------------------------------

    def accelerator_memory_to_sram(self, accelerator_dram_address: int, sram_address: int, element_size: int, stride_bytes_per_chunk: int = 0, stride_jump_bytes: int = 0) -> None:
        """DMA data from accelerator memory to SRAM"""
        element_size_bytes = element_size * 2
        uram_type, uram_start_addr = self.sram_address_to_uram_address(sram_address)
        self.ue_memcpy_from_dram(accelerator_dram_address, element_size_bytes, MEMCPY_TYPE.URAM.value, uram_start_addr, uram_type.value, self._inst_id,
                               stride_bytes_per_chunk=stride_bytes_per_chunk, stride_jump_bytes=stride_jump_bytes) # 2 bytes per element
        self._inst_id += 1

    def sram_to_accelerator_memory(self, sram_address: int, accelerator_dram_address: int, element_size: int, stride_bytes_per_chunk: int = 0, stride_jump_bytes: int = 0) -> None:
        """DMA data from SRAM to accelerator memory"""
        element_size_bytes = element_size * 2 # TODO: non-aligned writes are not supported
        uram_type, uram_start_addr = self.sram_address_to_uram_address(sram_address)
        assert stride_bytes_per_chunk % UE_VECTOR_SIZE == 0, "stride_bytes_per_chunk must be a multiple of UE_VECTOR_SIZE, TODO: support non-aligned writes"
        self.ue_memcpy_to_dram(MEMCPY_TYPE.URAM.value, uram_type.value, uram_start_addr, accelerator_dram_address, element_size_bytes, self._inst_id,
                               stride_bytes_per_chunk=stride_bytes_per_chunk, stride_jump_bytes=stride_jump_bytes) # 2 bytes per element
        self._inst_id += 1

    def accelerator_memory_to_bias_sram(self, accelerator_dram_address: int, element_size: int) -> None:
        """Copy from accelerator DRAM to bias BRAM. element_size is in elements (bf16 = 2 bytes per element)."""
        size_bytes = element_size * 2
        assert size_bytes <= BIAS_BRAM_SIZE_BYTES, f"size_bytes={size_bytes} must be less than or equal to BIAS_BRAM_SIZE_BYTES={BIAS_BRAM_SIZE_BYTES}"
        self.ue_memcpy_from_dram(accelerator_dram_address, size_bytes, MEMCPY_TYPE.BIAS_BRAM.value, 0, 0, self._inst_id)
        self._inst_id += 1

    # Element-wise operations ---------------------------------------------------
    def start_queue_eltwise(self, mode: UE_MODE,
                            uram_a_start_addr: int, uram_b_start_addr: int,
                            uram_wb_addr: int, uram_section: int, row_size: int) -> Optional[torch.Tensor]:
        """
        Start queue for element-wise op: ELTWISE_ADD or ELTWISE_MUL.
        Wraps start_queue with standard eltwise parameters.
        """
        if mode not in (UE_MODE.ELTWISE_ADD, UE_MODE.ELTWISE_MUL):
            raise ValueError(f"Invalid eltwise mode: {mode}. Use ELTWISE_ADD or ELTWISE_MUL.")
        self.start_queue(
            0,  # broadcast_mode
            0,  # max_clear_en
            1,  # stride_z
            LALU_MODE.BYPASS.value,  # lalu_mode
            0,  # scalar (not used)
            0,  # uram_bram (URAM)
            uram_section,
            0,  # uram_dst_addr
            0,  # dram_to_uram_cpy_start
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
            self._inst_id,
        )
        self._inst_id += 1

    def eltwise_add_core(self, vector_A_sram_start_addr: int, vector_B_sram_start_addr: int, vector_C_sram_wb_addr: int,
                                element_size: int) -> Optional[torch.Tensor]:
        """Start queue for element-wise add. Wraps start_queue with UE_MODE.ELTWISE_ADD."""
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
        """Start queue for element-wise multiply. Wraps start_queue with UE_MODE.ELTWISE_MUL."""
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

    # Broadcast operations ------------------------------------------------------
    def start_queue_broadcast(self, mode: UE_MODE, broadcast_mode: BROADCAST_MODE,
                              uram_src_start_addr: int, uram_wb_start_addr: int, 
                              element_size: int, scalar: float = None) -> Optional[torch.Tensor]:
        """
        Start queue for broadcast op: MUL_BROADCAST or ADD_BROADCAST.
        scalar: encoded scalar (e.g. bf16 from float_to_bf16 for MUL/ADD broadcast).
        Wraps start_queue with BROADCAST_MODE.SCALAR_IN_REG.
        """
        if mode not in (UE_MODE.MUL_BROADCAST, UE_MODE.ADD_BROADCAST):
            raise ValueError(f"Invalid broadcast mode: {mode}. Use MUL_BROADCAST or ADD_BROADCAST.")

        if broadcast_mode == BROADCAST_MODE.SCALAR_IN_REG:
            assert scalar is not None, "scalar must be provided for SCALAR_IN_REG broadcast mode"

        uram_src_type, uram_src_start_addr = self.sram_address_to_uram_address(uram_src_start_addr)
        assert uram_src_type ==  URAM_SECTION.URAM_A, "uram_src_start_addr must be in URAM_A"

        uram_wb_type, uram_wb_addr = self.sram_address_to_uram_address(uram_wb_start_addr)

        row_size = (element_size + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE # round up to the nearest multiple of UE_VECTOR_SIZE

        self.start_queue(
            broadcast_mode.value,  # broadcast_mode
            0,  # max_clear_en
            1,  # stride_z
            LALU_MODE.BYPASS.value,  # lalu_mode
            self.float_to_bf16(scalar) if scalar is not None else self.float_to_bf19(1.0),
            0,  # uram_bram (URAM)
            uram_wb_type.value,
            0,  # uram_dst_addr
            0,  # dram_to_uram_cpy_start
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
            self._inst_id,
        )
        self._inst_id += 1

    def broadcast_mul(self, scalar: float,
                            sram_start_addr: int, sram_wb_addr: int,
                            element_size: int) -> Optional[torch.Tensor]:                           
        """Start queue for broadcast multiply. Wraps start_queue with UE_MODE.MUL_BROADCAST."""
        self.start_queue_broadcast(
            UE_MODE.MUL_BROADCAST, BROADCAST_MODE.SCALAR_IN_REG,
            sram_start_addr, sram_wb_addr, element_size, scalar
        )

    def broadcast_add(self, scalar: float,
                            sram_start_addr: int, sram_wb_addr: int,
                            element_size: int) -> Optional[torch.Tensor]:
        """Start queue for broadcast add. Wraps start_queue with UE_MODE.ADD_BROADCAST."""
        self.start_queue_broadcast(
            UE_MODE.ADD_BROADCAST, BROADCAST_MODE.SCALAR_IN_REG,
            sram_start_addr, sram_wb_addr, element_size, scalar
        )

    # Compute engine operations --------------------------------------------------
    def start_queue_for_bf16_matvec_operation(self, max_clear_en: int, fmax_context_addr: int, vector_sram_start_addr: int, matrix_sram_start_addr: int, output_sram_wb_addr: int, K: int, N: int, bias_enable: bool = False) -> None:
        """Start queue for compute engine. Wraps start_queue with UE_MODE.
        Args:
            max_clear_en: max_clear_en
            fmax_context_addr: fmax_context_addr
            vector_sram_start_addr: vector_sram_start_addr
            matrix_sram_start_addr: matrix_sram_start_addr
            output_sram_wb_addr: output_sram_wb_addr
            K: K
            N: N
        """
        vector_uram_type, vector_uram_start_addr = self.sram_address_to_uram_address(vector_sram_start_addr)
        matrix_uram_type, matrix_uram_start_addr = self.sram_address_to_uram_address(matrix_sram_start_addr)
        assert vector_uram_type ==  URAM_SECTION.URAM_A, f"vector_sram_start_addr must be in URAM_A hex(vector_uram_start_addr)={hex(vector_uram_start_addr)}"
        assert matrix_uram_type ==  URAM_SECTION.URAM_B, f"matrix_sram_start_addr must be in URAM_B hex(matrix_uram_start_addr)={hex(matrix_uram_start_addr)}"
        output_uram_type, output_uram_start_addr = self.sram_address_to_uram_address(output_sram_wb_addr)

        assert K % UE_VECTOR_SIZE == 0, f"K={K} must be a multiple of UE_VECTOR_SIZE={UE_VECTOR_SIZE}"
        assert N % UE_VECTOR_SIZE == 0, f"N={N} must be a multiple of UE_VECTOR_SIZE={UE_VECTOR_SIZE}"

        self.start_queue(
            0,  # broadcast_mode (not used for bf16 matvec operation)
            max_clear_en,  # max_clear_en
            1,  # stride_z
            LALU_MODE.BYPASS.value,  # lalu_mode
            0,  # scalar
            0,  # uram_bram (URAM)
            output_uram_type.value,
            0,  # uram_dst_addr
            0,  # dram_to_uram_cpy_start
            output_uram_start_addr,
            URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
            UE_MODE.BF16_DOT_PRODUCT,
            0,  # data_type
            vector_uram_start_addr,
            matrix_uram_start_addr,  # uram_b_start_addr (not used for compute)
            K // UE_VECTOR_SIZE,
            0,  # dma_start_addr
            K * N,  # dma_length
            N,  # output_size
            self._inst_id,
            bias_adder_en=bias_enable,
            fmax_context_addr=fmax_context_addr
        )
        self._inst_id += 1

    # Compute engine operations --------------------------------------------------
    def start_queue_for_bf16_softmax_operation(self, fmax_context_addr: int, vector_sram_start_addr: int, output_sram_wb_addr: int, N: int) -> None:
        """Start queue for compute engine. Wraps start_queue with UE_MODE.
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
        self.start_queue(
            BROADCAST_MODE.FMAX_NEGATE.value,  # broadcast_mode
            0,  # max_clear_en
            1,  # stride_z
            LALU_MODE.MODE_RECIP.value,  # lalu_mode: 1/sum
            self.float_to_bf19(1.0),  # scalar: 1.0 in bf19 format
            0,  # uram_bram (URAM)
            vector_uram_type.value,
            0,  # uram_dst_addr
            0,  # dram_to_uram_cpy_start
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
            self._inst_id,
            fmax_context_addr=fmax_context_addr
        )
        self._inst_id += 1

        self.start_queue_broadcast(UE_MODE.MUL_BROADCAST, BROADCAST_MODE.LALU_RESULT, # use 1/sum result from LALU
                            vector_sram_start_addr, output_sram_wb_addr, N)

    def start_queue_for_bf16_rms_mean(self, vector_sram_start_addr: int, N: int) -> None:
        """
        Start queue for RMS mean operation. Wraps start_queue with UE_MODE.
        Args:
            vector_uram_start_addr: vector_uram_start_addr
            N: number of elements in the vector
        """
        row_size = (N + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE

        vector_uram_type, vector_uram_start_addr = self.sram_address_to_uram_address(vector_sram_start_addr)
        assert vector_uram_type == URAM_SECTION.URAM_A, f"vector_uram_start_addr must be in URAM_A, got {hex(vector_uram_start_addr)}"

        self.start_queue(
            0,  # broadcast_mode
            0,  # max_clear_en
            1,  # stride_z
            LALU_MODE.MODE_RSQRT.value,  # lalu_mode gets sqrt(N) / sqrt(sum(x_i^2))
            self.float_to_bf19(float(math.sqrt(N))),  # BF19 scalar (sqrt(N))
            0,  # uram_bram (URAM)
            vector_uram_type,  # uram_section
            0,  # uram_dst_addr
            0,  # dram_to_uram_cpy_start
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
            self._inst_id
        )
        self._inst_id += 1

    def start_queue_for_bf16_layer_norm_mean(self, vector_sram_start_addr: int, zeros_sram_start_addr: int, N: int) -> None:
        """
        Start queue for layer norm mean operation. Wraps start_queue with UE_MODE.
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

        self.start_queue(
            0,  # broadcast_mode
            0,  # clear_max_en
            1,  # stride_z
            LALU_MODE.MODE_RECIP.value,  # lalu_mode (computes 1/sum)
            self.float_to_bf19(float(N)),  # BF19 scalar (n)
            0,  # uram_bram (URAM = 0)
            vector_uram_type.value,  # uram_section
            0,  # uram_dst_addr
            0,  # dram_to_uram_cpy_start
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
            self._inst_id
        )
        self._inst_id += 1        

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


    # rms_norm dram version of rms norm core
    def rms_norm_core_dram(self, M: int, N: int, A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, GAMMA_DRAM_ADDR: int) -> None:
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

    def rms_norm_core_dram_post_add(self, M: int, N: int, A_DRAM_ADDR: int, B_DRAM_ADDR: int, ADDOUTPUT_DRAM_ADDR: int, NORMOUTPUT_DRAM_ADDR: int, GAMMA_DRAM_ADDR: int = None) -> None:
        """
        Residual addition followed by RMS norm: rms_norm(A + B).
        Args:
            M: number of rows in the input matrix
            N: number of columns in the input matrix
            A_DRAM_ADDR: DRAM address of first input matrix
            B_DRAM_ADDR: DRAM address of second input matrix
            ADDOUTPUT_DRAM_ADDR: DRAM address for residual addition output
            NORMOUTPUT_DRAM_ADDR: DRAM address for normalized output
            GAMMA_DRAM_ADDR: DRAM address for gamma
        """
        gamma_sram_addr = 0x80000
        params_sram_addr = gamma_sram_addr

        if GAMMA_DRAM_ADDR is not None:
            self.accelerator_memory_to_sram(accelerator_dram_address=GAMMA_DRAM_ADDR,
                                        sram_address=gamma_sram_addr,
                                        element_size=N)
            params_sram_addr += N * 2
        else:
            gamma_sram_addr = None

        # A in URAM_A, B in URAM_B (after params) so eltwise_add_core sees different URAMs
        vector_A_sram_addr = 0x00000
        vector_B_sram_addr = params_sram_addr
        uram_b_remaining_elements = URAM_NEAR_FULL_ELEMENTS - (params_sram_addr - 0x80000) // 2
        chunk_size = min(URAM_NEAR_FULL_ELEMENTS // N, uram_b_remaining_elements // N, M)
        assert chunk_size >= 1 and chunk_size <= M, f"chunk_size={chunk_size} must be greater than 0 and less than M={M}"

        for i, m_take in self.chunk_ranges(M, chunk_size):
            chunk_elements = m_take * N

            # Load A into URAM_A, B into URAM_B (after params)
            self.accelerator_memory_to_sram(accelerator_dram_address=A_DRAM_ADDR + i * N * 2,
                                        sram_address=vector_A_sram_addr,
                                        element_size=chunk_elements)
            self.accelerator_memory_to_sram(accelerator_dram_address=B_DRAM_ADDR + i * N * 2,
                                        sram_address=vector_B_sram_addr,
                                        element_size=chunk_elements)

            # Residual add: A + B -> A (in-place into A's SRAM location)
            for j in range(m_take):
                self.eltwise_add_core(vector_A_sram_addr + j * N * 2, vector_B_sram_addr + j * N * 2, vector_A_sram_addr + j * N * 2, N)

            # Write residual addition result to DRAM
            self.sram_to_accelerator_memory(sram_address=vector_A_sram_addr,
                                            accelerator_dram_address=ADDOUTPUT_DRAM_ADDR + i * N * 2,
                                            element_size=chunk_elements)

            # RMS norm the add result (in-place)
            for j in range(m_take):
                self.rms_norm_core(vector_A_sram_addr + j * N * 2, vector_A_sram_addr + j * N * 2, N, gamma_sram_addr)

            # Write normalized result to DRAM
            self.sram_to_accelerator_memory(sram_address=vector_A_sram_addr,
                                            accelerator_dram_address=NORMOUTPUT_DRAM_ADDR + i * N * 2,
                                            element_size=chunk_elements)

        # Total Theoretical FLOPS:
        total_flops = M * N # add(A, B)
        total_flops += 3 * M * N # exp(x), sum(x), broadcast_mul
        if gamma_sram_addr is not None:
            total_flops += M * N # eltwise_mul(gamma)
        return total_flops

    def rms_norm_core_dram_pre_add(self, M: int, N: int, A_DRAM_ADDR: int, B_DRAM_ADDR: int, NORMOUTPUT_DRAM_ADDR: int, ADDOUTPUT_DRAM_ADDR: int, GAMMA_DRAM_ADDR: int = None) -> None:
        """
        RMS norm followed by residual addition: rms_norm(A) + B.
        Args:
            M: number of rows in the input matrix
            N: number of columns in the input matrix
            A_DRAM_ADDR: DRAM address of input matrix to normalize
            B_DRAM_ADDR: DRAM address of residual matrix to add after norm
            NORMOUTPUT_DRAM_ADDR: DRAM address for normalized output
            ADDOUTPUT_DRAM_ADDR: DRAM address for residual addition output
            GAMMA_DRAM_ADDR: DRAM address for gamma
        """
        gamma_sram_addr = 0x80000
        params_sram_addr = gamma_sram_addr

        if GAMMA_DRAM_ADDR is not None:
            self.accelerator_memory_to_sram(accelerator_dram_address=GAMMA_DRAM_ADDR,
                                        sram_address=gamma_sram_addr,
                                        element_size=N)
            params_sram_addr += N * 2
        else:
            gamma_sram_addr = None

        # A in URAM_A, B in URAM_B (after params) so eltwise_add_core sees different URAMs
        vector_A_sram_addr = 0x00000
        vector_B_sram_addr = params_sram_addr
        uram_b_remaining_elements = URAM_NEAR_FULL_ELEMENTS - (params_sram_addr - 0x80000) // 2
        chunk_size = min(URAM_NEAR_FULL_ELEMENTS // N, uram_b_remaining_elements // N, M)
        assert chunk_size >= 1 and chunk_size <= M, f"chunk_size={chunk_size} must be greater than 0 and less than M={M}"

        for i, m_take in self.chunk_ranges(M, chunk_size):
            chunk_elements = m_take * N

            # Load A chunk into URAM_A
            self.accelerator_memory_to_sram(accelerator_dram_address=A_DRAM_ADDR + i * N * 2,
                                        sram_address=vector_A_sram_addr,
                                        element_size=chunk_elements)

            # RMS norm A (in-place)
            for j in range(m_take):
                self.rms_norm_core(vector_A_sram_addr + j * N * 2, vector_A_sram_addr + j * N * 2, N, gamma_sram_addr)

            # Write normalized result to DRAM
            self.sram_to_accelerator_memory(sram_address=vector_A_sram_addr,
                                            accelerator_dram_address=NORMOUTPUT_DRAM_ADDR + i * N * 2,
                                            element_size=chunk_elements)

            # Load B chunk into URAM_B (after params)
            self.accelerator_memory_to_sram(accelerator_dram_address=B_DRAM_ADDR + i * N * 2,
                                        sram_address=vector_B_sram_addr,
                                        element_size=chunk_elements)

            # Residual add: rms_norm(A) + B -> A (in-place)
            for j in range(m_take):
                self.eltwise_add_core(vector_A_sram_addr + j * N * 2, vector_B_sram_addr + j * N * 2, vector_A_sram_addr + j * N * 2, N)

            # Write addition result to DRAM
            self.sram_to_accelerator_memory(sram_address=vector_A_sram_addr,
                                            accelerator_dram_address=ADDOUTPUT_DRAM_ADDR + i * N * 2,
                                            element_size=chunk_elements)

        # Total Theoretical FLOPS:
        total_flops = 3 * M * N # exp(x), sum(x), broadcast_mul
        if gamma_sram_addr is not None:
            total_flops += M * N # eltwise_mul(gamma)
        total_flops += M * N # add(rms_norm(A), B)
        return total_flops

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

    def layer_norm_core_dram_post_add(self, M: int, N: int, A_DRAM_ADDR: int, B_DRAM_ADDR: int, ADDOUTPUT_DRAM_ADDR: int, NORMOUTPUT_DRAM_ADDR: int, GAMMA_DRAM_ADDR: int = None, BETA_DRAM_ADDR: int = None) -> None:
        """
        Core layer norm: normalizes vector x -> x / rms(x).
        Args:
            M: number of rows in the input matrix
            N: number of columns in the input matrix
            A_DRAM_ADDR: DRAM address of first input matrix
            B_DRAM_ADDR: DRAM address of second input matrix
            ADDOUTPUT_DRAM_ADDR: DRAM address of residual addition
            NORMOUTPUT_DRAM_ADDR: DRAM address for normalized output
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

        # A in URAM_A, B in URAM_B (after params) so eltwise_add_core sees different URAMs
        vector_A_sram_addr = 0x00000
        vector_B_sram_addr = params_sram_addr
        uram_b_remaining_elements = URAM_NEAR_FULL_ELEMENTS - (params_sram_addr - 0x80000) // 2
        chunk_size = min(URAM_NEAR_FULL_ELEMENTS // N, uram_b_remaining_elements // N, M)
        assert chunk_size >= 1 and chunk_size <= M, f"chunk_size={chunk_size} must be greater than 0 and less than M={M}"

        for i, m_take in self.chunk_ranges(M, chunk_size):
            chunk_elements = m_take * N

            # Load A into URAM_A, B into URAM_B (after params)
            self.accelerator_memory_to_sram(accelerator_dram_address=A_DRAM_ADDR + i * N * 2,
                                        sram_address=vector_A_sram_addr,
                                        element_size=chunk_elements)
            self.accelerator_memory_to_sram(accelerator_dram_address=B_DRAM_ADDR + i * N * 2,
                                        sram_address=vector_B_sram_addr,
                                        element_size=chunk_elements)

            # Residual add: A + B -> A (in-place into A's SRAM location)
            for j in range(m_take):
                self.eltwise_add_core(vector_A_sram_addr + j * N * 2, vector_B_sram_addr + j * N * 2, vector_A_sram_addr + j * N * 2, N)

            # Write residual addition result to DRAM
            self.sram_to_accelerator_memory(sram_address=vector_A_sram_addr,
                                            accelerator_dram_address=ADDOUTPUT_DRAM_ADDR + i * N * 2,
                                            element_size=chunk_elements)

            # Layer norm the add result (in-place)
            for j in range(m_take):
                self.layer_norm_core(vector_A_sram_addr + j * N * 2, vector_A_sram_addr + j * N * 2, N, zeros_sram_addr, gamma_sram_addr, beta_sram_addr)

            # Write normalized result to DRAM
            self.sram_to_accelerator_memory(sram_address=vector_A_sram_addr,
                                            accelerator_dram_address=NORMOUTPUT_DRAM_ADDR + i * N * 2,
                                            element_size=chunk_elements)

        # Total Theoretical FLOPS:
        total_flops = M * N # add(A, B)
        total_flops += 5 * M * N # mean(N), subtract(N), variance(N), sum(N), rsqrt(1), scale(N)
        if gamma_sram_addr is not None:
            total_flops += M * N # mul(gamma) N times
        if beta_sram_addr is not None:
            total_flops += M * N # add(beta) N times
        return total_flops

    def layer_norm_core_dram_pre_add(self, M: int, N: int, A_DRAM_ADDR: int, B_DRAM_ADDR: int, NORMOUTPUT_DRAM_ADDR: int, ADDOUTPUT_DRAM_ADDR: int, GAMMA_DRAM_ADDR: int = None, BETA_DRAM_ADDR: int = None) -> None:
        """
        Layer norm followed by residual addition: norm(A) + B.
        Args:
            M: number of rows in the input matrix
            N: number of columns in the input matrix
            A_DRAM_ADDR: DRAM address of input matrix to normalize
            B_DRAM_ADDR: DRAM address of residual matrix to add after norm
            NORMOUTPUT_DRAM_ADDR: DRAM address for normalized output
            ADDOUTPUT_DRAM_ADDR: DRAM address for residual addition output
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

        # A in URAM_A, B in URAM_B (after params) so eltwise_add_core sees different URAMs
        vector_A_sram_addr = 0x00000
        vector_B_sram_addr = params_sram_addr
        uram_b_remaining_elements = URAM_NEAR_FULL_ELEMENTS - (params_sram_addr - 0x80000) // 2
        chunk_size = min(URAM_NEAR_FULL_ELEMENTS // N, uram_b_remaining_elements // N, M)
        assert chunk_size >= 1 and chunk_size <= M, f"chunk_size={chunk_size} must be greater than 0 and less than M={M}"

        for i, m_take in self.chunk_ranges(M, chunk_size):
            chunk_elements = m_take * N

            # Load A chunk into URAM_A
            self.accelerator_memory_to_sram(accelerator_dram_address=A_DRAM_ADDR + i * N * 2,
                                        sram_address=vector_A_sram_addr,
                                        element_size=chunk_elements)

            # Layer norm A (in-place)
            for j in range(m_take):
                self.layer_norm_core(vector_A_sram_addr + j * N * 2, vector_A_sram_addr + j * N * 2, N, zeros_sram_addr, gamma_sram_addr, beta_sram_addr)

            # Write normalized result to DRAM
            self.sram_to_accelerator_memory(sram_address=vector_A_sram_addr,
                                            accelerator_dram_address=NORMOUTPUT_DRAM_ADDR + i * N * 2,
                                            element_size=chunk_elements)

            # Load B chunk into URAM_B (after params)
            self.accelerator_memory_to_sram(accelerator_dram_address=B_DRAM_ADDR + i * N * 2,
                                        sram_address=vector_B_sram_addr,
                                        element_size=chunk_elements)

            # Residual add: norm(A) + B -> A (in-place)
            for j in range(m_take):
                self.eltwise_add_core(vector_A_sram_addr + j * N * 2, vector_B_sram_addr + j * N * 2, vector_A_sram_addr + j * N * 2, N)

            # Write addition result to DRAM
            self.sram_to_accelerator_memory(sram_address=vector_A_sram_addr,
                                            accelerator_dram_address=ADDOUTPUT_DRAM_ADDR + i * N * 2,
                                            element_size=chunk_elements)

        # Total Theoretical FLOPS:
        total_flops = 5 * M * N # mean(N), subtract(N), variance(N), sum(N), rsqrt(1), scale(N)
        if gamma_sram_addr is not None:
            total_flops += M * N # mul(gamma) N times
        if beta_sram_addr is not None:
            total_flops += M * N # add(beta) N times
        total_flops += M * N # add(norm(A), B)
        return total_flops


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

    def matmat_mul_core(self, M: int, K: int, N: int, A_DRAM_ADDR: int, B_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, softmax_enable: bool = False, C_DRAM_ADDR: int = None, bias_mode: str = "broadcast_N") -> None:
        # Requirements: Based on these conditions M_chunk x K + M_chunk x N_chunk should fit in URAM_A and N_chunk x K should fit in URAM_B
        # 1. M_chunk can be any value between 1 and M
        # 2. N_chunk needs to be a multiple of UE_VECTOR_SIZE
        # 3. M_chunk * K + N_chunk * M_chunk must be less than or equal to URAM_FULL_ELEMENTS

        bytes_per_element = 2
        bias_enable = C_DRAM_ADDR is not None

        if bias_enable:
            assert bias_mode in ("broadcast_N", "full_matrix"), f"bias_mode={bias_mode} must be either 'broadcast_N' or 'full_matrix'"

        # Calculate N_chunk
        N_chunk = min(N, (URAM_NEAR_FULL_ELEMENTS // K) // UE_VECTOR_SIZE * UE_VECTOR_SIZE)
        assert N_chunk >= 1 and N_chunk <= N, f"N_chunk={N_chunk} must be greater than 0 and less than N={N}"

        # Calculate M_chunk
        if softmax_enable:
            M_chunk = min(UE_FMAX_CONTEXT_SIZE, M, URAM_FULL_ELEMENTS // (K + N_chunk)) # fmax_context is UE_FMAX_CONTEXT_SIZE elements
        else:
            M_chunk = min(M, URAM_FULL_ELEMENTS // (K + N_chunk))

        assert M_chunk >= 1 and M_chunk <= M, f"M_chunk={M_chunk} must be greater than 0 and less than M={M}"

        print(f"M_chunk: {M_chunk}, N_chunk: {N_chunk}")
        print(f"URAM_A usage: {100 * (M_chunk * K + M_chunk * N_chunk) / URAM_FULL_ELEMENTS:.2f}% of URAM_FULL_ELEMENTS")
        print(f"URAM_B usage: {100 * N_chunk * K / URAM_FULL_ELEMENTS:.2f}% of URAM_FULL_ELEMENTS")

        for i, m_take in self.chunk_ranges(M, M_chunk):
            self.accelerator_memory_to_sram(accelerator_dram_address=A_DRAM_ADDR + i * K * bytes_per_element,
                                            sram_address=0x00000,
                                            element_size=m_take * K)

            output_sram_wb_addr = 0x00000 + m_take * K * bytes_per_element
            if output_sram_wb_addr >= 0x80000:
                assert False, f"output_sram_wb_addr={output_sram_wb_addr} is greater than 0x80000, which is the size of URAM_B"

            clear_en = 1
            for j, n_take in self.chunk_ranges(N, N_chunk):
                self.accelerator_memory_to_sram(accelerator_dram_address=B_DRAM_ADDR + j * K * bytes_per_element,
                                            sram_address=0x80000,
                                            element_size=n_take * K)

                if bias_enable and bias_mode == "broadcast_N":
                    self.accelerator_memory_to_bias_sram(accelerator_dram_address=C_DRAM_ADDR + j * bytes_per_element, element_size=n_take)

                for output_row in range(m_take):

                    if bias_enable and bias_mode == "full_matrix":
                        self.accelerator_memory_to_bias_sram(accelerator_dram_address=C_DRAM_ADDR + ((i + output_row) * N + j) * bytes_per_element,
                                                             element_size=n_take)

                    self.start_queue_for_bf16_matvec_operation(max_clear_en=clear_en,
                                                            fmax_context_addr=output_row,
                                                            vector_sram_start_addr=0x00000 + output_row * K * bytes_per_element,
                                                            matrix_sram_start_addr=0x80000,
                                                            output_sram_wb_addr=output_sram_wb_addr + output_row * n_take * bytes_per_element,
                                                            K=K,
                                                            N=n_take,
                                                            bias_enable=bias_enable)
                    clear_en = 0

                # generated matrix is m_take x n_take
                start_dram_address_of_partial_matrix = OUTPUT_DRAM_ADDR + (i * N + j) * bytes_per_element

                # # LEGACY NOTE: This is the legacy way of copying m_take x n_take matrix to DRAM, check below for new way of copying with stride
                # for output_row in range(m_take):
                #     self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr + output_row * n_take * bytes_per_element, # every row is n_take elements
                #                                 accelerator_dram_address=start_dram_address_of_partial_matrix + output_row * N * bytes_per_element,
                #                                 element_size=n_take)

                # New way of copying with stride 
                self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr,
                                            accelerator_dram_address=start_dram_address_of_partial_matrix,
                                            element_size=m_take * n_take,
                                            stride_bytes_per_chunk=n_take * bytes_per_element,
                                            stride_jump_bytes=N * bytes_per_element)

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
        print(f"Total Theoretical FLOPS: {total_flops}")
        return total_flops

    def flash_attention_core(self, head_dim: int, seq_len: int, Q_DRAM_ADDR: int, K_DRAM_ADDR: int, V_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int, SCRATCH_DRAM_ADDR: int, BIAS_DRAM_ADDR: int = None,
                            debug_mode: bool = False, SM_OUTPUT_DRAM_ADDR: int = None, V_TRANS_DRAM_ADDR: int = None) -> None:

        bytes_per_element = 2
        bias_enable = BIAS_DRAM_ADDR is not None

        if debug_mode: # DEBUG only, needs to be allocated in DRAM
            assert SM_OUTPUT_DRAM_ADDR is not None, "SM_OUTPUT_DRAM_ADDR is not set for debug mode"
            assert V_TRANS_DRAM_ADDR is not None, "V_TRANS_DRAM_ADDR is not set for debug mode"

        # Q @ K^T: (seq_len, head_dim) @ (head_dim, seq_len) -> (seq_len, seq_len)
        # Convention: first matrix Q is (M, K), second K^T is (K, N), output scores (M, N)
        M = seq_len   # query length (rows of Q)
        K = head_dim  # head dimension (inner product dim)
        N = seq_len   # key length (columns of K^T)

        # Allocate identity matrix of size UE_VECTOR_SIZE x UE_VECTOR_SIZE in URAM_A start
        identity_matrix_sram_start_addr = 0x00000
        usable_sram_start_addr = identity_matrix_sram_start_addr + UE_VECTOR_SIZE * UE_VECTOR_SIZE * bytes_per_element

        # Calculate N_chunk
        N_chunk = min(N, ((URAM_FULL_ELEMENTS - UE_VECTOR_SIZE * UE_VECTOR_SIZE) // K) // UE_VECTOR_SIZE * UE_VECTOR_SIZE)
        assert N_chunk >= 1 and N_chunk <= N, f"N_chunk={N_chunk} must be greater than 0 and less than N={N}"

        M_chunk = min(UE_FMAX_CONTEXT_SIZE, M, (URAM_FULL_ELEMENTS - UE_VECTOR_SIZE * UE_VECTOR_SIZE) // (K + N_chunk)) # fmax_context is UE_FMAX_CONTEXT_SIZE elements
        assert M_chunk >= 1 and M_chunk <= M, f"M_chunk={M_chunk} must be greater than 0 and less than M={M}"

        print(f"M_chunk: {M_chunk}, N_chunk: {N_chunk}")
        print(f"URAM_A usage: {100 * (M_chunk * K + M_chunk * N_chunk) / URAM_FULL_ELEMENTS:.2f}% of URAM_FULL_ELEMENTS")
        print(f"URAM_B usage: {100 * N_chunk * K / URAM_FULL_ELEMENTS:.2f}% of URAM_FULL_ELEMENTS")

        identity_matrix_dram_addr = self.get_params_dram_addr()
        self.allocate_params_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * bytes_per_element)
        self.dma_write(DMA_DEVICE_H2C, identity_matrix_dram_addr, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16), UE_VECTOR_SIZE * UE_VECTOR_SIZE * bytes_per_element)

        # transfer identity matrix to URAM_A start
        self.accelerator_memory_to_sram(accelerator_dram_address=identity_matrix_dram_addr,
                                        sram_address=identity_matrix_sram_start_addr,
                                        element_size=UE_VECTOR_SIZE * UE_VECTOR_SIZE)

        for i, m_take in self.chunk_ranges(M, M_chunk):
            self.accelerator_memory_to_sram(accelerator_dram_address=Q_DRAM_ADDR + i * K * bytes_per_element,
                                            sram_address=usable_sram_start_addr,
                                            element_size=m_take * K)

            self.broadcast_mul(scalar=1 / math.sqrt(head_dim),
                                    sram_start_addr=usable_sram_start_addr,
                                    sram_wb_addr=usable_sram_start_addr,
                                    element_size=m_take * K)

            output_sram_wb_addr = usable_sram_start_addr + m_take * K * bytes_per_element

            assert output_sram_wb_addr < 0x80000, f"output_sram_wb_addr={output_sram_wb_addr} is greater than 0x80000, which is the size of URAM_B"

            clear_en = 1
            for j, n_take in self.chunk_ranges(N, N_chunk):
                self.accelerator_memory_to_sram(accelerator_dram_address=K_DRAM_ADDR + j * K * bytes_per_element,
                                            sram_address=0x80000,
                                            element_size=n_take * K)

                for output_row in range(m_take):
                    if bias_enable:
                        self.accelerator_memory_to_bias_sram(accelerator_dram_address=BIAS_DRAM_ADDR + ((i + output_row) * N + j) * bytes_per_element,
                                                             element_size=n_take)

                    self.start_queue_for_bf16_matvec_operation(max_clear_en=clear_en,
                                                            fmax_context_addr=output_row,
                                                            vector_sram_start_addr=usable_sram_start_addr + output_row * K * bytes_per_element,
                                                            matrix_sram_start_addr=0x80000,
                                                            output_sram_wb_addr=output_sram_wb_addr + output_row * n_take * bytes_per_element,
                                                            K=K,
                                                            N=n_take,
                                                            bias_enable=bias_enable)
                    clear_en = 0

                # TODO: if FMAX_CONTEXT_SIZE x seq_len can fit in URAM_A, then we can avoid copying to DRAM, create a special case for this
                start_dram_address_of_partial_matrix = SCRATCH_DRAM_ADDR + j * bytes_per_element # the space needed is FMAX_CONTEXT_SIZE x seq_len

                # LEGACY NOTE: This is the legacy way of copying m_take x n_take matrix to DRAM, check below for new way of copying with stride
                # for output_row in range(m_take):
                #     self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr + output_row * n_take * bytes_per_element, # every row is n_take elements
                #                                 accelerator_dram_address=start_dram_address_of_partial_matrix + output_row * N * bytes_per_element,
                #                                 element_size=n_take)

                # New way of copying with stride
                self.sram_to_accelerator_memory(sram_address=output_sram_wb_addr,
                                                accelerator_dram_address=start_dram_address_of_partial_matrix,
                                                element_size=m_take * n_take,
                                                stride_bytes_per_chunk=n_take * bytes_per_element,
                                                stride_jump_bytes=N * bytes_per_element)


            # SOFTMAX CALCULATION
            #print(f"softmax rows: {m_take * N} elements vs {URAM_FULL_ELEMENTS} elements")
            # DEBUG to get seq_len x seq_len sm(QK^T) results are copied to DRAM
            # start_dram_address_of_partial_row_complete_matrix = SM_OUTPUT_DRAM_ADDR + i * N * bytes_per_element #  make only FMAX_CONTEXT_SIZE x seq_len sm(QK^T) results are copied to DRAM
            start_dram_address_of_partial_row_complete_matrix = SCRATCH_DRAM_ADDR # the space needed is FMAX_CONTEXT_SIZE x seq_len
            
            # if m_take * N is greater than the space available in URAM_A, copy the matrix to DRAM
            max_m_take = min((URAM_FULL_ELEMENTS - UE_VECTOR_SIZE * UE_VECTOR_SIZE) // N, UE_FMAX_CONTEXT_SIZE)

            for m_take_chunk_idx, m_take_chunk_size in self.chunk_ranges(m_take, max_m_take):
                self.accelerator_memory_to_sram(accelerator_dram_address=start_dram_address_of_partial_row_complete_matrix + m_take_chunk_idx * N * bytes_per_element,
                                            sram_address=usable_sram_start_addr,
                                            element_size=m_take_chunk_size * N)

                # Reuse input sram_wb_addr for softmax output
                for row_idx in range(m_take_chunk_size):
                    self.start_queue_for_bf16_softmax_operation(fmax_context_addr=row_idx + m_take_chunk_idx,
                                                                vector_sram_start_addr=usable_sram_start_addr + row_idx * N * bytes_per_element,
                                                                output_sram_wb_addr=usable_sram_start_addr + row_idx * N * bytes_per_element,
                                                                N=N)


                # softmax output tap point - DEBUG only
                if debug_mode:
                    self.sram_to_accelerator_memory(sram_address=usable_sram_start_addr,
                                    accelerator_dram_address=SM_OUTPUT_DRAM_ADDR + (i + m_take_chunk_idx) * N * bytes_per_element,
                                    element_size=m_take_chunk_size * N)

                # V transpose calculation starts herewith symmetric partitioning
                v_row_chunk_size = min(URAM_HALF_ELEMENTS // UE_VECTOR_SIZE, seq_len)
                v_column_chunk_size = min(URAM_HALF_ELEMENTS // seq_len, head_dim, UE_VECTOR_SIZE)

                partial_v_transpose_can_stay_in_uram_b = True
                # if v_column_chunk_size is less than UE_VECTOR_SIZE, it means data cannot stay in URAM_B
                if v_column_chunk_size < UE_VECTOR_SIZE:
                    partial_v_transpose_can_stay_in_uram_b = False
                    v_column_chunk_size = UE_VECTOR_SIZE
                    first_partition_size = URAM_FULL_ELEMENTS - seq_len * UE_VECTOR_SIZE
                    v_row_chunk_size = min(first_partition_size // UE_VECTOR_SIZE, seq_len)
                    assert v_row_chunk_size >= 1 and v_row_chunk_size <= seq_len, f"v_row_chunk_size={v_row_chunk_size} must be greater than 0 and less than seq_len={seq_len}"

                v_sram_start_addr = 0x80000 # URAM_B start
                v_transpose_sram_start_addr = v_sram_start_addr + v_row_chunk_size * UE_VECTOR_SIZE * bytes_per_element

                assert v_column_chunk_size >= 1 and v_column_chunk_size <= head_dim, f"v_column_chunk_size={v_column_chunk_size} must be greater than 0 and less than head_dim={head_dim}"

                for v_column_inx, v_column_take in self.chunk_ranges(head_dim, v_column_chunk_size):
                    for v_row_inx, v_row_take in self.chunk_ranges(seq_len, v_row_chunk_size):
                        # LEGACY NOTE: This is the legacy way of copying v_row_take x v_column_take matrix to SRAM, check below for new way of copying with stride
                        # for op_row in range(v_row_take):
                        #     self.accelerator_memory_to_sram(accelerator_dram_address=V_DRAM_ADDR + v_row_inx * head_dim * bytes_per_element + v_column_inx * bytes_per_element + op_row * head_dim * bytes_per_element,
                        #                                     sram_address=0x80000 + op_row * v_column_take * bytes_per_element,
                        #                                     element_size=v_column_take)

                        # New way of copying with stride
                        self.accelerator_memory_to_sram(accelerator_dram_address=V_DRAM_ADDR + v_row_inx * head_dim * bytes_per_element + v_column_inx * bytes_per_element,
                                                        sram_address=v_sram_start_addr,
                                                        element_size=v_row_take * v_column_take,
                                                        stride_bytes_per_chunk=v_column_take * bytes_per_element,
                                                        stride_jump_bytes=head_dim * bytes_per_element)

                        for op_row in range(v_column_take):
                            self.start_queue_for_bf16_matvec_operation(max_clear_en=0,
                                                                    fmax_context_addr=0,
                                                                    vector_sram_start_addr=identity_matrix_sram_start_addr + op_row * UE_VECTOR_SIZE * bytes_per_element,
                                                                    matrix_sram_start_addr=v_sram_start_addr,
                                                                    output_sram_wb_addr=v_transpose_sram_start_addr + op_row * v_row_take * bytes_per_element,
                                                                    K=UE_VECTOR_SIZE,
                                                                    N=v_row_take)
     

                        if partial_v_transpose_can_stay_in_uram_b == False:
                            self.sram_to_accelerator_memory(sram_address=v_transpose_sram_start_addr,
                                                            accelerator_dram_address=SCRATCH_DRAM_ADDR + v_row_inx * bytes_per_element,
                                                            element_size=v_row_take * v_column_take,
                                                            stride_bytes_per_chunk=v_row_take * bytes_per_element,
                                                            stride_jump_bytes=seq_len * bytes_per_element)       

                        # V^T tap point - DEBUG: Copy V^T to DRAM
                        if debug_mode:
                            self.sram_to_accelerator_memory(sram_address=v_transpose_sram_start_addr,
                                                            accelerator_dram_address=V_TRANS_DRAM_ADDR + (v_column_inx * seq_len + v_row_inx) * bytes_per_element,
                                                            element_size=v_row_take * v_column_take,
                                                            stride_bytes_per_chunk=v_row_take * bytes_per_element,
                                                            stride_jump_bytes=seq_len * bytes_per_element)                            

                    # # V^T tap point - DEBUG: Copy V^T to DRAM if full seq_len partial matrix is available
                    # if debug_mode:
                    #     self.sram_to_accelerator_memory(sram_address=v_transpose_sram_start_addr,
                    #                                     accelerator_dram_address=V_TRANS_DRAM_ADDR + v_column_inx * seq_len * bytes_per_element,
                    #                                     element_size=seq_len * v_column_take)

                    # Bring it back to URAM_B at start of URAM_B
                    # mini partition of V^T and Softmax which is in URAM_A
                    # matrix multiplicaion here m_take_chunk_size x seq_len @ seq_len x v_column_take = m_take_chunk_size x v_column_take
                    # m_take_chunk_size x seq_len is in URAM_A already and known size. Let's calculate the URAM_B size needed for this operation
                    # Known output size is m_take_chunk_size x v_column_take

                    final_output_sram_wb_addr = 0x80000 # URAM_B start

                    if partial_v_transpose_can_stay_in_uram_b == False:
                        v_t_transfered_from_dram = final_output_sram_wb_addr + m_take_chunk_size * v_column_take * bytes_per_element
                    else:
                        v_t_transfered_from_dram = v_transpose_sram_start_addr

                    # TODO: This is a temporary fix to avoid the error, need to fix this properly
                    assert (seq_len * v_column_take + m_take_chunk_size * v_column_take) <= URAM_FULL_ELEMENTS, f"Not able to transfer partial V^T and Final Output to URAM_B"

                    if partial_v_transpose_can_stay_in_uram_b == False:
                        # Now we have seq_len x v_column_take matrix in URAM_B
                        self.accelerator_memory_to_sram(accelerator_dram_address=SCRATCH_DRAM_ADDR,
                                                        sram_address=v_t_transfered_from_dram,
                                                        element_size=seq_len * v_column_take)

                    # After this m_take x v_column_take matrix is in 0xC0000
                    for p_row_idx in range(m_take_chunk_size):
                        self.start_queue_for_bf16_matvec_operation(max_clear_en=0,
                                                                fmax_context_addr=0,
                                                                vector_sram_start_addr=usable_sram_start_addr + p_row_idx * seq_len * bytes_per_element,
                                                                matrix_sram_start_addr=v_t_transfered_from_dram,
                                                                output_sram_wb_addr=final_output_sram_wb_addr + p_row_idx * v_column_take * bytes_per_element,
                                                                K=seq_len,
                                                                N=v_column_take)

                    # LEGACY NOTE: This is the legacy way of copying m_take x n_take matrix to DRAM, check below for new way of copying with stride
                    # for ii in range(m_take):
                    #     self.sram_to_accelerator_memory(sram_address=0x80000 + ii * v_column_take * bytes_per_element, # every row is n_take elements
                    #                                     accelerator_dram_address=OUTPUT_DRAM_ADDR + i * head_dim * bytes_per_element + v_column_inx * bytes_per_element + ii * head_dim * bytes_per_element,
                    #                                     element_size=v_column_take)

                    # New way of copying with stride - Abandoned for now since we don't have enough space in URAM_B to copy the matrix
                    self.sram_to_accelerator_memory(sram_address=final_output_sram_wb_addr,
                                                    accelerator_dram_address=OUTPUT_DRAM_ADDR + (i + m_take_chunk_idx) * head_dim * bytes_per_element + v_column_inx * bytes_per_element,
                                                    element_size=m_take_chunk_size * v_column_take,
                                                    stride_bytes_per_chunk=v_column_take * bytes_per_element,
                                                    stride_jump_bytes=head_dim * bytes_per_element)

        # Total Theoretical FLOPS: seq_len * head_dim + 2 * seq_len * head_dim * seq_len + seq_len * seq_len + seq_len * seq_len * 5 + 2 * seq_len * seq_len * head_dim
        total_flops = seq_len * head_dim # q_scale 
        total_flops += 2 * seq_len * head_dim * seq_len # Q @ K^T
        if bias_enable:
            total_flops += seq_len * seq_len # bias
        total_flops += seq_len * seq_len * 5 # softmax
        total_flops += 2 * seq_len * seq_len * head_dim # sm @ v
        print(f"Total Theoretical FLOPS: {total_flops}")
        return total_flops

    def start_capture(self):
        """Start capturing instructions instead of executing them"""
        self.capture_count = 0
        self.capture_buffer = []
        self.is_capture_on = True
        print(f"Capture started. Buffer initialized, count={self.capture_count}")

    def stop_capture(self):
        """Stop capturing instructions"""
        self.is_capture_on = False
        print(f"Capture stopped. Total instructions captured: {self.capture_count}")

    def clear_capture_buffer(self):
        """Clear the capture buffer"""
        self.capture_buffer = []
        self.capture_count = 0
        print(f"Capture buffer cleared. Total instructions captured: {self.capture_count}")

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
        inst_type = (w[7] >> 29) & 0x7
        
        # Print instruction header with hex dump
        hex_str = ' '.join([f"{w[j]:08X}" for j in range(8)])
        print(f"  [{inst_index:4d}] @ {hex(inst_addr):10s} = {hex_str}")
        
        # Parse and format instruction details
        if inst_type == 0 or inst_type == INSTRUCTION_REG_REWRITE:
            # UE op (Unified Engine operation)
            mode_sel = (w[3] >> 4) & 0xF
            
            if mode_sel == 0xF:
                # memcpy from dram
                # Word 0: ISA control fields (if REG_REWRITE) or 0
                inst_type = (w[7] >> 29) & 0x7
                if inst_type == INSTRUCTION_REG_REWRITE:
                    general_reg_src = (w[0] >> 4) & 0xF
                else:
                    general_reg_src = 0
                # Word 1: unified DMA address (dram_src_addr)
                dram_src_addr = w[1]
                memcpy_length = w[2]
                stride_bytes_per_chunk = (w[3] >> 8) & 0xFFFF
                uram_select = (w[5] >> 9) & 0x1
                dr_mb_wb_mode = (w[5] >> 10) & 0x3
                uram_memcpy_dst_addr = (w[5] >> 12) & 0xFFF
                bram_uram_select = (w[5] >> 24) & 0x3
                stride_jump_bytes_low = (w[5] >> 29) & 0x7
                stride_jump_bytes_high = w[6] & 0x3FFFF
                stride_jump_bytes = stride_jump_bytes_low | (stride_jump_bytes_high << 3)
                stride_en = (w[7] >> 14) & 0x1
                inst_id = (w[7] >> 23) & 0x3F
                
                # Parse memcpy_type
                memcpy_type_name = MEMCPY_TYPE(bram_uram_select).name if bram_uram_select in [e.value for e in MEMCPY_TYPE] else f"UNKNOWN({bram_uram_select})"
                memcpy_type_str = f"{memcpy_type_name}"
                if bram_uram_select == MEMCPY_TYPE.URAM.value:
                    # For URAM, also show URAM_A or URAM_B
                    uram_section_name = URAM_SECTION(uram_select).name if uram_select in [e.value for e in URAM_SECTION] else f"UNKNOWN({uram_select})"
                    memcpy_type_str += f" ({uram_section_name})"
                
                result = f"UE_MEMCPY_FROM_DRAM"
                if inst_type == INSTRUCTION_REG_REWRITE:
                        result += f" (addr reset from general register {general_reg_src})"
                result += f"\n    dram_src_addr: {hex(dram_src_addr)}"
                result += f"\n    memcpy_length: {memcpy_length} bytes"
                result += f"\n    uram_dst_addr: {uram_memcpy_dst_addr}"
                result += f"\n    memcpy_type: {memcpy_type_str}"
                if stride_en:
                    result += f"\n    stride_bytes_per_chunk: {stride_bytes_per_chunk}"
                    result += f"\n    stride_jump_bytes: {stride_jump_bytes}"
                result += f"\n    inst_id: {inst_id}"
                # Print formatted instruction details
                for line in result.split('\n'):
                    print(f"        {line}")
                return
            else:
                # Check if it's memcpy to dram or compute engine
                mode = (w[3] >> 4) & 0xF
                dma_start = w[3] & 0x1
                uram_start = (w[3] >> 1) & 0x1
                data_type = (w[3] >> 2) & 0x3
                output_size = (w[3] >> 8) & 0xFFFF
                uram_length = ((w[3] >> 24) & 0xFF) | (((w[4] >> 0) & 0xF) << 8)
                uram_a_start_addr = (w[4] >> 4) & 0xFFF
                uram_b_start_addr = (w[4] >> 16) & 0xFFF
                uram_wb_addr = ((w[4] >> 28) & 0xF) | (((w[5] >> 0) & 0xFF) << 4)
                dram_to_uram_cpy_start = (w[5] >> 8) & 0x1
                uram_section = (w[5] >> 9) & 0x1
                uram_write_src = (w[5] >> 10) & 0x3
                uram_dst_addr = (w[5] >> 12) & 0xFFF
                uram_bram = (w[5] >> 24) & 0x3
                lalu_mode = (w[5] >> 26) & 0x7
                scalar_low = (w[5] >> 29) & 0x7
                scalar_high = (w[6] >> 0) & 0x3FFFF
                scalar = scalar_low | (scalar_high << 3)
                uram_length_z = (w[6] >> 18) & 0xFFF
                uram_bram_wb_start = (w[6] >> 30) & 0x1
                wb_padding_control = (w[6] >> 31) & 0x1
                max_clear_en = w[7] & 0x1
                bias_adder_en = (w[7] >> 1) & 0x1
                stride_z = (w[7] >> 2) & 0xFFF
                stride_en = (w[7] >> 14) & 0x1
                broadcast_mode = (w[7] >> 21) & 0x3
                inst_id = (w[7] >> 23) & 0x3F
                # Word 0: ISA control fields (if REG_REWRITE) or 0
                inst_type = (w[7] >> 29) & 0x7
                if inst_type == INSTRUCTION_REG_REWRITE:
                    general_reg_src = (w[0] >> 4) & 0xF
                else:
                    general_reg_src = 0
                # Word 1: unified DMA address
                dram_addr = w[1]
                dma_length = w[2]
                
                if mode == UE_MODE.URAM_DRAM_WRITEBACK.value:
                    # memcpy to dram
                    # Parse memcpy_type
                    memcpy_type_name = MEMCPY_TYPE(uram_bram).name if uram_bram in [e.value for e in MEMCPY_TYPE] else f"UNKNOWN({uram_bram})"
                    memcpy_type_str = f"{memcpy_type_name}"
                    if uram_bram == MEMCPY_TYPE.URAM.value:
                        # For URAM, also show URAM_A or URAM_B
                        uram_section_name = URAM_SECTION(uram_section).name if uram_section in [e.value for e in URAM_SECTION] else f"UNKNOWN({uram_section})"
                        memcpy_type_str += f" ({uram_section_name})"
                    
                    result = f"UE_MEMCPY_TO_DRAM"
                    if inst_type == INSTRUCTION_REG_REWRITE:
                        result += f" (addr from general register {general_reg_src})"
                    result += f"\n    uram_src_addr: {uram_a_start_addr}"
                    result += f"\n    dram_dst_addr: {hex(dram_addr)}"
                    result += f"\n    memcpy_length: {uram_length} uram rows ({uram_length * 64 * 2} bytes)"
                    result += f"\n    memcpy_type: {memcpy_type_str}"
                    result += f"\n    uram_bram_wb_start: {uram_bram_wb_start}"
                    if stride_en:
                        result += f"\n    stride_en: enabled"
                    result += f"\n    inst_id: {inst_id}"
                    # Print formatted instruction details
                    for line in result.split('\n'):
                        print(f"        {line}")
                    return
                else:
                    # compute engine
                    mode_name = UE_MODE(mode).name if mode in [e.value for e in UE_MODE] else f"UNKNOWN({mode})"
                    result = f"UE_COMPUTE ({mode_name})"
                    result += f"\n    uram_a_start: {uram_a_start_addr}, uram_b_start: {uram_b_start_addr}"
                    result += f"\n    uram_length: {uram_length}, uram_wb_addr: {uram_wb_addr}"
                    result += f"\n    output_size: {output_size}"
                    
                    # data_type
                    data_type_name = TYPE(data_type).name if data_type in [e.value for e in TYPE] else f"UNKNOWN({data_type})"
                    result += f"\n    data_type: {data_type_name}"
                    
                    # uram_start or dram_start
                    if uram_start:
                        result += f"\n    uram_start: enabled, uram_length: {uram_length}, uram_length_z: {uram_length_z}"
                    if dma_start:
                        result += f"\n    dma_start: enabled, dma_start_addr: {hex(dram_addr)}"
                        if inst_type == INSTRUCTION_REG_REWRITE:
                            result += f" (from general register {general_reg_src})"
                        result += f", dma_length: {dma_length}"
                    
                    # uram_type based on uram_bram and uram_section
                    if uram_bram == MEMCPY_TYPE.URAM.value:
                        uram_section_name = URAM_SECTION(uram_section).name if uram_section in [e.value for e in URAM_SECTION] else f"UNKNOWN({uram_section})"
                        result += f"\n    uram_type: URAM ({uram_section_name})"
                    else:
                        uram_type_name = MEMCPY_TYPE(uram_bram).name if uram_bram in [e.value for e in MEMCPY_TYPE] else f"UNKNOWN({uram_bram})"
                        result += f"\n    uram_type: {uram_type_name}"
                    
                    # writeback enabled (uram_write_src = 0) otherwise writeback disabled
                    if uram_write_src == URAM_WRITE_SRC.URAM_WRITE_BACK.value:
                        result += f"\n    writeback: enabled, uram_dst_addr: {uram_dst_addr}"
                    else:
                        writeback_name = URAM_WRITE_SRC(uram_write_src).name if uram_write_src in [e.value for e in URAM_WRITE_SRC] else f"UNKNOWN({uram_write_src})"
                        result += f"\n    writeback: disabled ({writeback_name})"
                    
                    # wb_padding_select
                    result += f"\n    wb_padding_select: {wb_padding_control}"
                    
                    if bias_adder_en:
                        result += f"\n    bias_adder: enabled"
                    if max_clear_en:
                        result += f"\n    max_clear: enabled"
                    if lalu_mode != 0:
                        lalu_mode_name = LALU_MODE(lalu_mode).name if lalu_mode in [e.value for e in LALU_MODE] else f"UNKNOWN({lalu_mode})"
                        result += f"\n    lalu_mode: {lalu_mode_name}, scalar: {hex(scalar)}"
                    if broadcast_mode != 0:
                        broadcast_mode_name = BROADCAST_MODE(broadcast_mode).name if broadcast_mode in [e.value for e in BROADCAST_MODE] else f"UNKNOWN({broadcast_mode})"
                        result += f"\n    broadcast_mode: {broadcast_mode_name}"
                    result += f"\n    inst_id: {inst_id}"
                    # Print formatted instruction details
                    for line in result.split('\n'):
                        print(f"        {line}")
                    return
        else:
            # ISA instruction
            # Word 0 (bits 0-31): ISA control fields in [15:0]
            isa_mode = w[0] & 0xF
            src_reg_idx = (w[0] >> 4) & 0xF
            dst_reg_idx = (w[0] >> 8) & 0xF
            rst_reg_idx = (w[0] >> 12) & 0xF
            # Word 1 (bits 32-63): immediate_value (moved from [31:0] to [63:32])
            immediate_value = w[1]
            
            if inst_type == INSTRUCTION_HALT:
                result = f"ISA_HALT"
                for line in result.split('\n'):
                    print(f"        {line}")
                return
            elif inst_type == INSTRUCTION_JUMP:
                jump_mode = isa_mode
                reg_id = src_reg_idx
                jump_mode_name = ["ABSOLUTE", "RELATIVE", "JNZ"][jump_mode] if jump_mode < 3 else f"UNKNOWN({jump_mode})"
                result = f"ISA_JUMP ({jump_mode_name})"
                result += f"\n    target_addr: {hex(immediate_value)}"
                if jump_mode == 2:  # JNZ
                    result += f"\n    reg_id: {reg_id}"
                for line in result.split('\n'):
                    print(f"        {line}")
                return
            elif inst_type == INSTRUCTION_ADD:
                isa_mode_names = {
                    INST_ADD_INC: "INC",
                    INST_ADD_DEC: "DEC",
                    INST_ADD_REG: "REG",
                    INST_ADD_IMM: "IMM",
                    INST_ADD_SET: "SET"
                }
                mode_name = isa_mode_names.get(isa_mode, f"UNKNOWN({isa_mode})")
                result = f"ISA_ADD ({mode_name})"
                if isa_mode == INST_ADD_SET:  # SET
                    result += f"\n    dst_reg: {dst_reg_idx}, value: {hex(immediate_value)}"
                elif isa_mode == INST_ADD_INC:  # INC
                    result += f"\n    reg: {dst_reg_idx}"
                elif isa_mode == INST_ADD_DEC:  # DEC
                    result += f"\n    reg: {dst_reg_idx}"
                elif isa_mode == INST_ADD_IMM:  # IMM
                    result += f"\n    reg: {dst_reg_idx}, immediate: {hex(immediate_value)}"
                elif isa_mode == INST_ADD_REG:  # REG
                    result += f"\n    dst_reg: {dst_reg_idx}, src_reg: {src_reg_idx}, rst_reg: {rst_reg_idx}"
                for line in result.split('\n'):
                    print(f"        {line}")
                return
            else:
                result = f"ISA_UNKNOWN (type=0x{inst_type:X})"
                for line in result.split('\n'):
                    print(f"        {line}")
                return

    def generate_instruction(self, inst_type: int, immediate_value: int = 0,
                             isa_mode: int = 0, src_reg_idx: int = 0,
                             dst_reg_idx: int = 0, rst_reg_idx: int = 0):
        """
        Generate an instruction and add to capture buffer (base function matching C code)

        Args:
            inst_type: Instruction type (INSTRUCTION_HALT, INSTRUCTION_JUMP, INSTRUCTION_ADD, etc.)
            immediate_value: 32-bit immediate value (for JUMP: target_addr, for ADD: immediate value)
            isa_mode: ISA mode (4 bits) - for ADD: INST_ADD_INC, INST_ADD_DEC, etc.
            src_reg_idx: Source register index (4 bits)
            dst_reg_idx: Destination register index (4 bits)
            rst_reg_idx: Reset register index (4 bits)
        """
        # capture_buffer is initialized to [] in start_capture(); treat None as uninitialized
        if self.capture_buffer is None:
            print("ERROR: generate_instruction() called but capture_buffer is not initialized!")
            assert False
            return

        if self.capture_count >= MAX_DECODER_INSTRUCTIONS:
            print(f"ERROR: generate_instruction() called but capture_count ({self.capture_count}) >= MAX ({MAX_DECODER_INSTRUCTIONS})!")
            assert False
            return

        inst = Instructions()
        w = inst.words

        # Clear all 256 bits (8 words) to zero first
        for i in range(8):
            w[i] = 0

        # Word 0 (bits 0-31): ISA control fields in [15:0], reserved in [31:16]
        # bits [3:0]: isa_mode
        # bits [7:4]: src_reg_idx
        # bits [11:8]: dst_reg_idx
        # bits [15:12]: rst_reg_idx
        # bits [31:16]: reserved
        w[0] = ((isa_mode & 0xF) << 0) | \
               ((src_reg_idx & 0xF) << 4) | \
               ((dst_reg_idx & 0xF) << 8) | \
               ((rst_reg_idx & 0xF) << 12)

        # Word 1 (bits 32-63): immediate_value (moved from [31:0] to [63:32])
        # For JUMP: target_addr
        # For ADD: immediate value
        # For HALT: set to 0
        w[1] = immediate_value

        # Word 7 (bits 224-255): inst_type at bits 253-255 (bits 29-31 of word 7)
        w[7] = (inst_type & 0x7) << 29

        self.capture_buffer.append(inst)
        self.capture_count += 1

        # print(f"generate_instruction(inst_type=0x{inst_type:X}, imm=0x{immediate_value:X}, isa={isa_mode}, src={src_reg_idx}, dst={dst_reg_idx}, rst={rst_reg_idx}): wrote to buffer[{self.capture_count-1}]")

    def generate_instruction_halt(self):
        """Generate a HALT instruction"""
        self.generate_instruction(INSTRUCTION_HALT, 0, 0, 0, 0, 0)

    def generate_instruction_jump(self, target_instruction_addr: int, jump_mode: int, reg_id: int = 0):
        """
        Generate a JUMP instruction

        Args:
            target_instruction_addr: 32-bit target address for jump
            jump_mode: JUMP_MODE_ABSOLUTE, JUMP_MODE_RELATIVE, or JUMP_MODE_JNZ
            reg_id: Register ID for JNZ mode (4 bits) - only used when jump_mode == JUMP_MODE_JNZ
        """
        self.generate_instruction(INSTRUCTION_JUMP, target_instruction_addr, jump_mode, reg_id, 0, 0)

    def generate_instruction_add_inc(self, reg_idx: int):
        """
        Generate an ADD instruction to increment a register

        Args:
            reg_idx: Register index to increment (must not be 0)
        """
        if reg_idx == 0:
            print("ERROR: INSTRUCTION_ADD overwriting reg_idx 0 (zero reg) not allowed")
            return
        self.generate_instruction(INSTRUCTION_ADD, 0, INST_ADD_INC, reg_idx, reg_idx, 0)

    def generate_instruction_add_dec(self, reg_idx: int):
        """
        Generate an ADD instruction to decrement a register

        Args:
            reg_idx: Register index to decrement (must not be 0)
        """
        if reg_idx == 0:
            print("ERROR: INSTRUCTION_ADD overwriting reg_idx 0 (zero reg) not allowed")
            return
        self.generate_instruction(INSTRUCTION_ADD, 0, INST_ADD_DEC, reg_idx, reg_idx, 0)

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
        self.generate_instruction(INSTRUCTION_ADD, 0, INST_ADD_REG, src_reg_idx, dst_reg_idx, rst_reg_idx)

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
        self.generate_instruction(INSTRUCTION_ADD, immediate_value, INST_ADD_IMM, src_reg_idx, dst_reg_idx, 0)

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
        self.generate_instruction(INSTRUCTION_ADD, immediate_value, INST_ADD_SET, dst_reg_idx, dst_reg_idx, 0)


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
        self.write_reg32(UE_INSTRUCTION_ADDR, instruction_addr)

    def report_timing_and_instruction_count(self):
        """
        Report timing and instruction count
        """
        latency = self.read_reg32(UE_LATENCY_COUNT_ADDR)
        instruction_count = self.read_reg32(UE_INSTRUCTION_CTL_ADDR)
        print(f"Latency: {latency * CLOCK_CYCLE_TIME_NS / 1e3:.3f} us, Instruction count: {instruction_count}")
        print(f"Latency in cycles: {latency}")

    def report_latency_in_us(self):
        """
        Report latency
        """
        return self.read_reg32(UE_LATENCY_COUNT_ADDR) * CLOCK_CYCLE_TIME_NS / 1e3

    def report_flop_rate_gflops(self, num_flops: int):
        """
        Report flop rate
        """
        return num_flops / (self.read_reg32(UE_LATENCY_COUNT_ADDR) * CLOCK_CYCLE_TIME_NS)

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
