"""
reset_state_regs.py

Zero the AXI-Lite control registers that ``software_reset()`` does NOT touch
and that ``init_unified_engine()`` does NOT re-write. These registers retain
whatever the prior process last wrote into them, and the compiled bin may
not re-initialize them before use (relying on cold-boot zero).

This module performs ONLY direct AXI-Lite register writes — no URAM/SRAM
DMAs, no instruction capture, no queue activity. That keeps it safe to call
at any point after software_reset() / init_unified_engine().

Registers we LEAVE ALONE:
  0x00 UE_FPGA_VERSION          - RO version word
  0x14 UE_STATUS                - RO status
  0x1C UE_OUTPUT_VALID_DELAY    - set by init_unified_engine()
  0x30 UE_LATENCY_COUNT         - RO counter
  0x44 UE_VALID_DELAY_EXTRA     - set by init_unified_engine()
  0x4C UE_LALU_DELAY            - set by init_unified_engine()
  0x54 UE_QUEUE_CTRL            - touched by software_reset()
  0x8C-0x98 UE_ARGMAX1..4       - result registers, may be RO

Registers we ZERO (all suspected to carry process-spanning state):
  0x08 UE_DRAM
  0x0C UE_DMA_LENGTH
  0x10 UE_CONTROL
  0x20 UE_URAM_LENGTH
  0x24 UE_URAM_WRITEBACK
  0x34 UE_DRAM_URAM_CTRL
  0x38 UE_LALU_HYPERPARAMETERS
  0x40 UE_URAM_ROW_SIZE
  0x48 UE_LALU_INST
  0x50 UE_SCALAR
  0x60 UE_BIAS_ADDER_EN      <-- prime suspect for gpt2 cross-run NaN
  0x64 UE_URAM_WB_PADDING
  0x68 UE_BROADCAST_MODE
  0x70 UE_INSTRUCTION_CTL
  0x78 UE_INSTRUCTION
  0x80 UE_FMAX_CONTEXT
  0x84 UE_TRACE_BRAM
  0x9C UE_INT_REG            <-- write clears interrupt latch
"""

from user_dma_core import (
    UnifiedEngine,
    UE_DRAM_ADDR,
    UE_DMA_LENGTH_ADDR,
    UE_CONTROL_ADDR,
    UE_URAM_LENGTH_ADDR,
    UE_URAM_WRITEBACK_ADDR,
    UE_DRAM_URAM_CTRL_ADDR,
    UE_LALU_HYPERPARAMETERS_ADDR,
    UE_URAM_ROW_SIZE_ADDR,
    UE_LALU_INST_ADDR,
    UE_SCALAR_ADDR,
    UE_BIAS_ADDER_EN_ADDR,
    UE_URAM_WB_PADDING_ADDR,
    UE_BROADCAST_MODE_ADDR,
    UE_INSTRUCTION_CTL_ADDR,
    UE_INSTRUCTION_ADDR,
    UE_FMAX_CONTEXT_ADDR,
    UE_TRACE_BRAM_ADDR,
    UE_INT_REG,
)


# Narrowed list: only registers that look like per-op operands or mode flags,
# NOT persistent device-config registers. The full-zero list hung prefill
# because something (UE_INSTRUCTION_CTL / UE_FMAX_CONTEXT / UE_URAM_ROW_SIZE /
# UE_CONTROL / UE_DRAM_URAM_CTRL) needs a non-zero default that the bin
# doesn't re-initialize.
_REGS_TO_ZERO = (
    UE_LALU_HYPERPARAMETERS_ADDR,  # 0x38 - bf16 lalu_a/lalu_b operands
    UE_LALU_INST_ADDR,             # 0x48 - lalu instruction mode
    UE_SCALAR_ADDR,                # 0x50 - scalar operand
    UE_BIAS_ADDER_EN_ADDR,         # 0x60 - bias adder mode (gpt2 prime suspect)
    UE_BROADCAST_MODE_ADDR,        # 0x68 - broadcast mode flag
    UE_URAM_WB_PADDING_ADDR,       # 0x64 - URAM writeback padding mode
)


def reset_state_regs(ue: UnifiedEngine) -> None:
    """Write 0 to every AXI-Lite register that survives software_reset().

    Call AFTER ``software_reset()`` and BEFORE any DMA / queue activity.
    """
    for addr in _REGS_TO_ZERO:
        try:
            ue.write_reg32(addr, 0x00000000)
        except Exception as e:  # noqa: BLE001
            print(f"[reset_state_regs] write 0x{addr:02X} failed: {e}")

    # UE_INT_REG: write any value clears the interrupt-cause latch (bits [1:0]).
    try:
        ue.write_reg32(UE_INT_REG, 0xFFFFFFFF)
    except Exception as e:  # noqa: BLE001
        print(f"[reset_state_regs] write UE_INT_REG failed: {e}")
