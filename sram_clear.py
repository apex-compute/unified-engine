"""
sram_clear.py

Zero every on-chip memory that ``software_reset()`` does NOT touch. Call
AFTER ``software_reset()`` and BEFORE loading any model weights / tensors.

What software_reset() clears:
  - DMA queue state / pipeline latches
  - The latency/delay config registers (re-applied by init_unified_engine)
  - Python-side ``_inst_id`` counter

What it does NOT clear (and what THIS function zeros):
  - URAM A    (SRAM 0x00000-0x7FFFF, ~512 KB on-chip scratchpad)
  - URAM B    (SRAM 0x80000-0xFFFFF, ~512 KB on-chip scratchpad)
  - Scale SRAM (8192 bf16 elements, 16 KB)
  - Bias SRAM  (8192 bf16 elements, 16 KB)

If any of those banks hold NaN / garbage left over from the previous process,
the first kernel of the next model that READS from them before WRITING them
pulls the poison in, and NaN propagates through every downstream bf16
matmul / softmax. That is the cross-run state contamination we've been chasing.
"""

import torch

from user_dma_core import (
    UnifiedEngine,
    URAM_HALF_ELEMENTS,
    SCALE_BRAM_ELEMENTS,
    BIAS_BRAM_ELEMENTS,
    DRAM_START_ADDR,
    UE_ARGMAX1_INDEX,
    UE_ARGMAX2_INDEX,
    UE_ARGMAX3_INDEX,
    UE_ARGMAX4_INDEX,
    UE_INT_REG,
    UE_LAST_REG_ADDR,
    URAM_SECTION,
)


def clear_on_chip_sram(
    ue: UnifiedEngine,
    scratch_dram_addr: int = DRAM_START_ADDR,
    timeout_s: float = 5.0,
) -> None:
    """
    Mini-program that fills URAM A, URAM B, scale SRAM, and bias SRAM with 0.

    Steps:
      1. Host -> DRAM: DMA 256 KB of bf16 zeros to ``scratch_dram_addr``
         (URAM_HALF_ELEMENTS = 2048 * 64 = 131072 bf16 elements = 262144 bytes).
         Both URAM banks reuse the same DRAM buffer; scale/bias SRAM read a
         16 KB prefix of it.
      2. Capture 4 UE memcpy descriptors + HALT into a captured-instruction
         program, write it to program DRAM, kick the queue, wait for HALT.

    ``scratch_dram_addr`` defaults to DRAM_START_ADDR (0x80000000). Whatever
    you stage there is overwritten by your subsequent weight_init().
    """
    zeros = torch.zeros(URAM_HALF_ELEMENTS, dtype=torch.bfloat16)
    ue.dma_to_accelerator_memory(scratch_dram_addr, zeros)

    ue.clear_inst_id()
    ue.start_capture()

    ue.accelerator_memory_to_sram(scratch_dram_addr, 0x00000, URAM_HALF_ELEMENTS)
    ue.accelerator_memory_to_sram(scratch_dram_addr, 0x80000, URAM_HALF_ELEMENTS)
    ue.accelerator_memory_to_scale_sram(scratch_dram_addr, SCALE_BRAM_ELEMENTS)
    ue.accelerator_memory_to_bias_sram(scratch_dram_addr, BIAS_BRAM_ELEMENTS)

    ue.stop_capture()
    ue.generate_instruction_halt()

    program_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())
    ue.clear_capture_buffer()
    ue.start_execute_from_dram(program_addr)
    ue.wait_queue(timeout_s)


def probe_state(ue: UnifiedEngine, tag: str = "") -> None:
    """
    Read & print the FPGA state surfaces that survive software_reset() so we
    can see what is left over from a prior process. Call BEFORE and AFTER your
    clear helpers to confirm they did anything.

    Prints:
      - ARGMAX1..4 index registers (0x8C-0x98). These hold the last sampled
        token index. If the new model reads ARGMAX before writing, it sees
        the prior process's last token = stuck-token loop ("sui sui sui").
      - UE_INT_REG (0x9C). Interrupt-cause latch bits [1:0].
      - Full AXI-Lite config-register dump 0x00..UE_LAST_REG_ADDR.
      - Python-side state: _inst_id, _inst_ptr_counter, _isa_reg_counter,
        and the program/params DRAM bump pointers if present.
    """
    prefix = f"[probe_state {tag}]" if tag else "[probe_state]"
    print(f"{prefix} ARGMAX1=0x{ue.read_reg32(UE_ARGMAX1_INDEX):08x} "
          f"ARGMAX2=0x{ue.read_reg32(UE_ARGMAX2_INDEX):08x} "
          f"ARGMAX3=0x{ue.read_reg32(UE_ARGMAX3_INDEX):08x} "
          f"ARGMAX4=0x{ue.read_reg32(UE_ARGMAX4_INDEX):08x}")
    print(f"{prefix} UE_INT_REG(0x9C)=0x{ue.read_reg32(UE_INT_REG):08x}")

    print(f"{prefix} AXI-Lite register dump 0x00..0x{UE_LAST_REG_ADDR:02X}:")
    for addr in range(0x00, UE_LAST_REG_ADDR + 4, 4):
        try:
            val = ue.read_reg32(addr)
        except Exception as e:  # noqa: BLE001
            val = -1
            print(f"{prefix}   0x{addr:02X}: READ-FAILED ({e})")
            continue
        print(f"{prefix}   0x{addr:02X}: 0x{val:08x}")

    py_state = {
        "_inst_id": getattr(ue, "_inst_id", None),
        "_inst_ptr_counter": getattr(ue, "_inst_ptr_counter", None),
        "_isa_reg_counter": getattr(ue, "_isa_reg_counter", None),
        "_next_params_dram_addr": getattr(ue, "_next_params_dram_addr", None),
        "_next_program_dram_addr": getattr(ue, "_next_program_dram_addr", None),
        "_tensor_dram_addr": getattr(ue, "_tensor_dram_addr", None),
    }
    print(f"{prefix} python state: {py_state}")


def clear_argmax_and_pbi_regs(
    ue: UnifiedEngine,
    timeout_s: float = 5.0,
) -> None:
    """
    Zero state surfaces that survive software_reset() AND are NOT on-chip
    memories (so clear_on_chip_sram doesn't cover them):

      - UE_INT_REG (interrupt-cause latch, 0x9C): write any value -> clears
        the latch (per axi_lite_*_module.sv comment in user_dma_core.py).
      - UE_ARGMAX1..4 (0x8C-0x98): attempt write-0. These may be read-only
        result registers in the RTL; we try anyway and ``probe_state`` after
        will tell you whether it stuck.
      - PBI pointer rows 1..15: emit a PBI_INIT descriptor for each index
        with all-zero fields, captured into a mini-program and executed.
        This resets the per-pointer base/length state used by PBI memcpys.

    Call AFTER software_reset() (and after clear_on_chip_sram if you also
    use it) and BEFORE weight_init().
    """
    # Direct AXI-Lite register writes
    ue.write_reg32(UE_INT_REG, 0xFFFFFFFF)  # clear interrupt latch
    for argmax_addr in (UE_ARGMAX1_INDEX, UE_ARGMAX2_INDEX,
                        UE_ARGMAX3_INDEX, UE_ARGMAX4_INDEX):
        try:
            ue.write_reg32(argmax_addr, 0x00000000)
        except Exception as e:  # noqa: BLE001
            print(f"[clear_argmax_and_pbi_regs] write 0x{argmax_addr:02X} failed: {e}")

    # PBI pointer rows 1..15 -> emit PBI_INIT with all-zero fields, executed
    # via the queue (same pattern as isa_add_set_core in gpt2_run_from_bin.py).
    ue.clear_inst_id()
    ue.start_capture()
    for idx in range(1, 16):
        ue.generate_instruction_pbi_init(
            dram_shared_addr=0,
            dma_length=0,
            output_size=0,
            uram_length=0,
            uram_a_start_addr=0,
            uram_b_start_addr=0,
            uram_wb_addr=0,
            uram_dst_addr=0,
            fmax_context_addr=0,
            inst_pointer_idx=idx,
        )
    ue.stop_capture()
    ue.generate_instruction_halt()

    program_addr = ue.get_program_dram_addr()
    ue.write_captured_instructions_to_dram(program_addr)
    ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())
    ue.clear_capture_buffer()
    ue.start_execute_from_dram(program_addr)
    ue.wait_queue(timeout_s)

    # Reset Python-side PBI pointer counter so model-side allocators start
    # from a clean state.
    if hasattr(ue, "reset_inst_ptr_counter"):
        ue.reset_inst_ptr_counter()
