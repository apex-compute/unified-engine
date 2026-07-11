# Unified-engine ISA & PBI Compiler Guide

This document describes the queue-engine ISA and the pointer-backed increment
(PBI) mechanism from the software side: the instruction set the compiler targets,
the Python helpers that emit it, and the usage patterns. The **ISA Overview**,
**PBI Overview**, and **Software API** sections cover everything needed to build
and run programs.

> Hardware-level behavior (FSM, register files, per-field data path) lives in a
> separate internal RTL reference and is not part of this guide.

---

## ISA Overview

The queue engine consumes one **256-bit instruction descriptor** per instruction.
The low 16 bits are a common header; the rest of the word is interpreted by
instruction type.

Common header:

- `[7:0]` — transaction / instruction id
- `[11:8]` — instruction type (see table)
- `[15:12]` — pointer-row index; used by `UE_PBI` and `PBI_SET`

Type-specific header bits:

- **UE ops (non-`PBI_SET`):** `[31:16]` = `lalu_a`.
- **`PBI_SET`:** `[19:16]` = `pbi_mode`, `[23:20]` = `pbi_field_select`, `[29:24]` = GPR source index (for `PBI_MODE_REG`).
- **ISA micro-ops (jump / reg-alu / flag / …):** `[35:32]` = `isa_mode`, `[41:36]` = src reg, `[47:42]` = dst reg, `[53:48]` = rst reg, `[85:54]` = 32-bit immediate.

### Instruction types (`inst_descriptor[11:8]`)

| Value | Type | Python | Purpose |
| --- | --- | --- | --- |
| `0x0` | `UE_OP` | `INSTRUCTION_UE_OP` | UE compute / memcpy op, absolute descriptor fields |
| `0x1` | `JUMP` | `INSTRUCTION_JUMP` | control flow (absolute / relative / conditional) |
| `0x2` | `REG_ALU` (prefetchable) | `INSTRUCTION_REG_ALU` | general-purpose register ALU |
| `0x3` | `REG_REWRITE` | `INSTRUCTION_REG_REWRITE` | UE op whose DRAM address comes from a GPR at runtime |
| `0x4` | `SEMAPHORE` | `INSTRUCTION_FLAG` | cross-engine flag set / clear / check |
| `0x5` | `UE_PBI` | `INSTRUCTION_UE_PBI` | UE op that reads a pointer row, then increments it |
| `0x6` | `PBI_SET` (prefetchable) | `INSTRUCTION_PBI_SET` | pointer-row update |
| `0x7` | `REG_ALU` (non-prefetchable) | — | register ALU, serialized behind the UE engine |
| `0x8` | `SWI` | `INSTRUCTION_SWI` | software interrupt |
| `0x9` | `HALT` | `INSTRUCTION_HALT` | stop and raise an interrupt |
| `0xA` | `NOP` | `INSTRUCTION_NOP` | no-op; alignment padding |
| `0xB` | `PBI_SET` (non-prefetchable) | — | pointer-row update, serialized behind the UE engine |

---

## PBI Overview

PBI is a descriptor-reuse mechanism for repeated UE operations. Instead of
re-encoding full absolute addresses and sizes for every instruction, software
seeds a hardware **pointer row** once; UE ops then read that row for their
operands and advance it automatically after each op.

The four things you can do with a pointer row:

1. **Seed** it with absolute values.
2. **Use** it as the operand source for a UE op.
3. **Advance** it — automatically after each pointer-backed op, or standalone.
4. **Override** one field from a register at runtime.

PBI is a **pointer-field reuse** mechanism, not general descriptor indirection.
Only the twelve numeric operand fields in the pointer row are backed; all control
settings (`mode`, `data_type`, `broadcast_mode`, `lalu_mode`, `uram_select`, start
flags, `bf16_lalu_a/b`, …) always come from the descriptor.

---

## Software API

Programs are built by capturing descriptors into a buffer, flushing them to DRAM,
and replaying them from the queue:

`start_capture()` → emit instructions → `generate_instruction_halt()` →
`write_captured_instructions_to_dram()` → start the queue.

### UE operations

Compute and data-movement helpers emit UE ops:

- `ue_arithmetic_op(...)` — compute modes (dot product, eltwise, softmax, rms, …).
- `ue_memcpy_from_dram(...)` — DRAM → URAM / BRAM / BIAS / SCALE.
- `ue_memcpy_to_dram(...)` — URAM / BRAM → DRAM writeback.
- Wrappers: `accelerator_memory_to_sram(...)`, `sram_to_accelerator_memory(...)`, `start_queue_eltwise(...)`, and the matmul / softmax / rms / quantize `start_queue_*` helpers.

By default each helper encodes its addresses, lengths, and sizes **absolutely**
into the instruction. Pass a non-zero `inst_pointer_idx` to switch the same helper
into pointer-backed (PBI) mode — see below.

Low-level compilers (rarely called directly): `ue_op_descriptor(...)` builds any
UE / PBI descriptor; `ue_isa_descriptor(...)` builds ISA micro-ops.

### PBI operations

A **pointer row** is a small hardware register holding a UE op's numeric operands
— DRAM address, DMA length, output size, URAM addresses/sizes, and strides (twelve
fields). Set a row once, then let repeated ops read and advance it, so the loop
body stays tiny and identical every iteration.

**1. Allocate a row.**
`alloc_inst_ptr()` returns the next free index `1..15`; `release_inst_ptr(idx)`
frees the most recent (release in reverse order); `reset_inst_ptr_counter()`
discards all allocation state. Index `0` is reserved and cannot hold state. Use
**one row per independently advancing stream** — input load, output writeback, and
each inner URAM traversal advance by different amounts, so each needs its own row.

**2. Use it in a UE op (`UE_PBI`, auto-increment).**
Pass `inst_pointer_idx=p` to any UE helper (`ue_arithmetic_op`,
`ue_memcpy_from_dram/to_dram`, wrappers). Two things change:

- the op takes its operands (addresses / lengths / sizes) **from the row**, not from the arguments; and
- the numeric arguments you pass become **per-iteration deltas** — after the op runs, the row is advanced by them automatically.

So inside a loop, each pass reads the current row and then post-increments it by
your deltas. Pass `0` for any field that should stay put.

**3. Set or update a row (three forms; all emit `PBI_SET`, none run a UE op).**

- **Immediate set (`PBI_MODE_INIT`) — `generate_instruction_pbi_init(...)`.** Writes
  **every** field of the row with the absolute base values you pass. Use it to
  establish a stream's starting point. Signature:
  `generate_instruction_pbi_init(dram_shared_addr, dma_length, output_size,
  uram_length, uram_a_start_addr, uram_b_start_addr, uram_wb_addr, uram_dst_addr,
  fmax_context_addr, inst_pointer_idx, uram_row_stride_z=1, stride_jump=0)`.
- **Immediate increment (`PBI_MODE_INC`) — `generate_instruction_pbi_inc(...)`.**
  **Adds** the given deltas into the current row. Use it for carry corrections
  between nested loops (the auto-advance in step 2 does the same thing after each op).
- **GPR override (`PBI_MODE_REG`) — `generate_instruction_pbi_inc(...,
  general_reg_src=g, pbi_field_select=PBI_FIELD.X)`.** Same call as the increment
  form, but the one field you select is **replaced** by the value currently in GPR
  `g`; **every other field is processed exactly as the increment form**, so pass `0`
  for any field you don't want to touch. Use it when an address or size is only known at run time
  (compute it into a GPR with the register-ALU helpers first). Selectable fields
  (`PBI_FIELD` enum): `DRAM_ADDR`, `DMA_LENGTH`, `OUTPUT_SIZE`, `URAM_ROW_SIZE`,
  `URAM_START_ADDR_Y`, `URAM_START_ADDR_Z`, `URAM_WRITEB_ADDR`, `URAM_ROW_SIZE_Z`,
  `URAM_MEMCPY_DST_ADDR`, `FMX_CONTEXT`, `URAM_ROW_STRIDE_Z`, `STRIDE_JUMP`.

`generate_instruction_clear_fmax()` issues a harmless op on reserved row 0 to
clear the fmax context.

> **Prefetch / latency:** `pbi_init` / `pbi_inc` and the register-ALU helpers emit
> the *prefetchable* instruction variants by default, so their bookkeeping overlaps
> the previous UE op's execution and costs ~nothing on the critical path. Trade-off:
> a prefetchable `pbi_init` is **not** held behind a still-draining DMA, so don't
> chain a `pbi_init` reconfigure right behind an in-flight DMA on the same data —
> use a `general_reg_src` / `REG_REWRITE` override there instead. The
> non-prefetchable hardware variants exist for strict ordering but are not currently
> exposed as a Python option.

### ISA general controls

Instruction-emit helpers for control flow, register math, and synchronization:

- **Jumps:** `generate_instruction_jump_abs / jump_abs_jnz / jump_abs_jz / jump_reg_abs / jump_rela / jump_rela_jnz / jump_rela_jz / jump_reg_rela`.
- **Register ALU (32-bit GPRs):** `generate_instruction_add_set / add_inc / add_dec / add_reg / add_imm / reg_min / reg_sub / shr / shl / mul32_reg / mul32_imm / mul32_shl_reg / mul32_shr_reg / div_reg`.
- **Cross-engine flags:** `generate_instruction_flag_set / flag_clear / flag_check(target_engine_idx)`.
- **Misc:** `generate_instruction_swi / halt / nop`; `overwrite_instruction_with_general_register(gpr)` rewrites the last captured UE op so its DRAM address is taken from a GPR at run time.

**Counted loops.** `loop_start` / `loop_end` build a hardware-counted
backward-branch loop, so a short body replays in hardware while PBI advances the
addresses across iterations:

- `loop_start(loop_cnt=0, gpr_loop_cnt=None)` — allocates a counter GPR, emits `ADD_SET` with `loop_cnt` (or `ADD_IMM` from a runtime trip count in `gpr_loop_cnt`), records the body head, pushes a stack frame; returns the counter register.
- `loop_end()` — pops the frame, emits `ADD_DEC` then `JUMP_RELA_JNZ(body_size)`, releases the counter. Frames are LIFO, so nested loops just nest the calls.

> **Loops need an absolute-jump anchor.** The instruction cache holds one
> 512-instruction window that is only reloaded on an unconditional **absolute** jump;
> a relative backward jump (what `loop_end` emits) just steps back inside that
> window. So begin a looped program with an unconditional absolute jump into the
> loop region and keep the whole body within 512 instructions of it — otherwise the
> relative jump can under-run the cache and replay stale instructions.
> `check_isa_jumps()` statically verifies this.

**Nested loops with PBI.** There is no "restore pointer snapshot" opcode, so:

1. Put full tiles inside the hardware-counted loop; handle the remainder tile outside it.
2. Allocate outer-loop persistent pointer rows once and seed them (`pbi_init`) once — they carry forward across outer iterations automatically.
3. Inside the outer body, re-`pbi_init` any inner pointer rows that must restart from a local base each outer iteration.
4. Use `generate_instruction_pbi_inc(...)` only for genuine end-of-loop carry corrections; prefer a local re-seed when it is simpler.

### Typical code flow

```python
p = ue.alloc_inst_ptr()

ue.generate_instruction_pbi_init(               # seed absolute bases into row p
    dram_shared_addr=base_dram_addr,
    dma_length=base_length_bytes,
    uram_a_start_addr=base_uram_a,
    uram_wb_addr=base_uram_wb,
    inst_pointer_idx=p,
    # ... other bases ...
)

ue.loop_start(loop_cnt)
ue.accelerator_memory_to_sram(                  # reads row p; args below are per-iter deltas
    accelerator_dram_address=dram_delta, sram_address=dst, element_size=0,
    inst_pointer_idx=p,
)                                               # row p advances by the deltas after the op
ue.ue_arithmetic_op(..., uram_wb_addr=uram_wb_delta, inst_pointer_idx=p)
ue.loop_end()

ue.release_inst_ptr(p)
```

---

## Common Pitfalls

- Don't use pointer index `0` for state — row 0 reads zero and its writes are dropped in hardware.
- Don't share one pointer row across streams that advance by different deltas.
- Don't assume every descriptor field is pointer-backed — only the twelve operand fields are; control settings always come from the descriptor.
- In a pointer-backed (`UE_PBI`) op, the numeric arguments are **increment deltas**, not the op operands (the operand comes from the row).
- A register override (`PBI_MODE_REG` / `general_reg_src`) **replaces** the selected field with the raw GPR value; it does not add to it.
- Don't emit a relative backward jump without a preceding unconditional absolute anchor, and keep the loop body within the 512-instruction i-cache window.
- Don't call `reset_inst_ptr_counter()` from a helper that should only release its own local pointer rows.
