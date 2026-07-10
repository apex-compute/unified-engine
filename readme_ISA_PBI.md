# PBI Mechanism Guide

This document summarizes the current ISA shape and the pointer-backed increment (PBI) mechanism from RTL support in `queue_state_module.sv` to software usage in `user_dma_core.py`.

## ISA Summary

The queue engine consumes one 256-bit instruction descriptor per instruction. The low bits are a common header:

- `[7:0]`: transaction / instruction id
- `[11:8]`: instruction type
- `[15:12]`: pointer register file index (`inst_pointer_idx` in RTL; used for PBI and for other pointer-backed ISA ops)
- `[19:16]`: **PBI sub-mode**, meaningful only when `inst_type == INST_TYPE_PBI_SET` (`4'h6`):
  - `PBI_MODE_INIT = 4'b0000`: load the selected pointer row from the descriptor’s pointer-backed fields
  - `PBI_MODE_INC = 4'b0001`: add the descriptor’s pointer-backed field values to the current pointer row (same arithmetic as `INST_TYPE_UE_PBI` after an op)

For non–`PBI_SET` instructions, software should keep `[19:16]` at zero so those bits do not alias the low nibble of `inst_bf16_lalu_a` (`inst_descriptor[31:16]`) on UE paths.

The rest of the 256-bit word is interpreted by instruction type. The main classes are:

- UE instructions: arithmetic or memcpy work submitted to the UE datapath
- ISA control instructions: jump, add, register rewrite
- Synchronization / control instructions: semaphore, software interrupt, halt

Major software entry points:

- `ue_arithmetic_op()`: compute-centric UE descriptor emission
- `ue_memcpy_from_dram()`: DRAM to URAM/BRAM/BIAS/SCALE copy emission
- `ue_memcpy_to_dram()`: URAM/BRAM to DRAM writeback emission
- higher-level wrappers such as `accelerator_memory_to_sram()`, `sram_to_accelerator_memory()`, matmul helpers, softmax helpers, and loop helpers build on top of those primitives

## PBI Overview

PBI is a descriptor reuse mechanism for repeated UE operations.

Instead of re-encoding full absolute addresses and sizes for every repeated instruction, software:

1. Initializes a hardware pointer row with base values (`PBI_SET` + `PBI_MODE_INIT`), or bumps it without a UE op (`PBI_SET` + `PBI_MODE_INC`).
2. Emits normal UE instructions tagged with a non-zero pointer index (`UE_PBI`).
3. Lets hardware read supported fields from that pointer row for the op.
4. Lets hardware increment the same pointer row after each `UE_PBI` execution (and optionally between ops via `PBI_MODE_INC`).

## RTL Support

### Pointer resources

PBI uses a dedicated pointer register file:

- **Row select:** `inst_pointer_idx = inst_descriptor[15:12]` (4 bits → 16 rows, `0..15`).
- **PBI_SET only:** `inst_pointer_mode = inst_descriptor[19:16]` (`PBI_MODE_INIT` / `PBI_MODE_INC`), defined in `queue_state_module.sv` next to `INST_TYPE_PBI_SET`.

### Instruction types relevant to PBI

The relevant instruction types are:

- `INST_TYPE_UE_OP = 4'h0`: normal UE op, no pointer backing.
- `INST_TYPE_UE_PBI = 4'h5`: execute using pointer-backed fields, then increment the pointer row.
- `INST_TYPE_PBI_SET = 4'h6`: initialize a pointer row.

### Pointer-backed fields supported by RTL

The pointer row currently stores only the following fields:

- `dram_addr`
- `dma_length`
- `output_size`
- `uram_row_size`
- `uram_start_addr_y`
- `uram_start_addr_z`
- `uram_writeb_addr`
- `uram_row_size_z`
- `uram_memcpy_dst_addr`
- `fmx_context`

### Fields not supported by PBI

The following fields are still taken directly from each instruction descriptor and are not stored in the pointer row:

- `mode_sel`
- `data_type`
- `broadcast_mode`
- `lalu_mode`
- `lalu_a`
- `lalu_b`
- `lalu_scalar`
- `uram_select`
- `dr_mb_wb_mode`
- `dma_stride_en`
- `uram_row_stride_z`
- `use_bf19_bias_adder`
- UE start bits such as `dma_start`, `uram_start`, `start_memcpy`, `uram_dram_wb_start`

This means PBI should be viewed as a pointer-field reuse mechanism, not as a fully general descriptor indirection mechanism.

### RTL behavior for `INST_TYPE_PBI_SET`

`INST_TYPE_PBI_SET` does **not** run the UE datapath. It only updates the pointer register file in `INST_PBI_UPDATE`, then returns to fetch.

Behavior in `INST_PBI_UPDATE` (see `queue_state_module.sv`):

| Condition | Pointer row update |
| --- | --- |
| `inst_type == INST_TYPE_PBI_SET && inst_pointer_mode == PBI_MODE_INIT` | **Load:** `pointer_regfile_wdata` is the concatenation of the instruction’s pointer-backed fields (absolute seed values). |
| `inst_type == INST_TYPE_UE_PBI` | **Increment:** each supported field becomes `pbi_* + inst_*` (descriptor supplies **deltas**). |
| `inst_type == INST_TYPE_PBI_SET && inst_pointer_mode == PBI_MODE_INC` | **Increment:** same add form as `UE_PBI`, but **no** UE execute—only the pointer row writeback. |

So **`PBI_MODE_INIT`** is “set row from descriptor,” and **`PBI_MODE_INC`** is “bump row by descriptor deltas” without firing a compute/memcpy op.

### Pointer allocation

Software manages pointer allocation with:

- `alloc_inst_ptr()`
- `release_inst_ptr()`
- `reset_inst_ptr_counter()` (only when you intentionally want to discard the whole allocation state)

Pointer allocation starts at `1` and goes up to `15`.

In practice, software treats pointer index `0` as reserved / non-user. The docstring of `generate_instruction_clear_fmax()` explicitly refers to using reserved register `0`.

Recommended lifetime rule:

- Prefer **local allocation + local release** inside each kernel helper.
- Do **not** call `reset_inst_ptr_counter()` at the end of a helper unless that helper explicitly owns the full pointer-allocation lifetime.
- If a helper allocates `N` pointers, release exactly those `N` pointers before return, ideally in reverse allocation order.

### Software: `generate_instruction_pbi_init` and `generate_instruction_pbi_inc`

Both helpers emit `INSTRUCTION_PBI_SET = 6` via `ue_op_descriptor()`. They differ only in **`inst_descriptor[19:16]`** (`pbi_mode` in Python):

- **`generate_instruction_pbi_init(...)`** → `PBI_MODE_INIT` (`4'b0000`): load the pointer row.
- **`generate_instruction_pbi_inc(...)`** → `PBI_MODE_INC` (`4'b0001`): add descriptor deltas into the pointer row.

`user_dma_core.py` exposes matching software constants:

- `PBI_MODE_INIT = 0`
- `PBI_MODE_INC = 1`

For `INSTRUCTION_PBI_SET`, the descriptor compiler packs the low word as:

- `[7:0]` instruction id, `[11:8]` type `6`, `[15:12]` `inst_pointer_idx`, `[19:16]` `pbi_mode`, and **`[31:20]` zero** (no `lalu_a` in the upper half for this type).

Init example:

```python
generate_instruction_pbi_init(
    dram_shared_addr,
    dma_length,
    output_size,
    uram_length,
    uram_a_start_addr,
    uram_b_start_addr,
    uram_wb_addr,
    uram_dst_addr,
    fmax_context_addr,
    inst_pointer_idx,
)
```

Increment-only pointer row update (no UE op), same argument list:

```python
generate_instruction_pbi_inc(
    dram_shared_addr,
    dma_length,
    output_size,
    uram_length,
    uram_a_start_addr,
    uram_b_start_addr,
    uram_wb_addr,
    uram_dst_addr,
    fmax_context_addr,
    inst_pointer_idx,
)
```

Advanced / custom use: call `ue_op_descriptor(..., inst_type=INSTRUCTION_PBI_SET, inst_pointer_idx=..., pbi_mode=PBI_MODE_INIT | PBI_MODE_INC, ...)` with the same field layout as the helpers.

### Software pointer mode for regular helpers

Most regular helpers become pointer-backed automatically when `inst_pointer_idx != 0`.

Examples:

- `ue_memcpy_from_dram(...)`
- `ue_memcpy_to_dram(...)`
- `ue_arithmetic_op(...)`
- wrappers such as `accelerator_memory_to_sram(...)`, `sram_to_accelerator_memory(...)`, `accelerator_memory_to_bias_sram(...)`, and `start_queue_eltwise(...)`

When `inst_pointer_idx != 0`, software emits instruction type `5`, which the RTL interprets as pointer-backed `UE_PBI`.

### Pointer-mode semantics

Once `inst_pointer_idx != 0`, supported numeric fields become increments, not fresh absolute values.

For DRAM-to-SRAM copies:

- Normal mode: `accelerator_dram_address` is an absolute DRAM source address.
- Pointer mode: `accelerator_dram_address` is a DRAM address increment in bytes.

For SRAM-to-DRAM writeback:

- Normal mode: `accelerator_dram_address` is an absolute DRAM destination.
- Pointer mode: `accelerator_dram_address` is the DRAM destination increment.

For arithmetic ops:

- Supported address/size-style fields such as `uram_a_start_addr`, `uram_b_start_addr`, `uram_wb_addr`, `uram_length`, `dma_start_addr`, `dma_length`, `output_size`, and `fmax_context_addr` are interpreted as deltas against the pointer row.
- Unsupported control fields still come directly from the current descriptor.

### Multi-pointer design rule

For real kernels, the most important design rule is:

- **Use one pointer row per independently advancing stream.**

Do **not** assume one pointer index is enough for an entire kernel.

Typical independent streams are:

- DRAM input load for outer tiles
- DRAM output writeback for outer tiles
- inner-loop URAM read pointer for operand A
- inner-loop URAM read pointer for operand B
- inner-loop URAM writeback pointer
- optional bias / scale / fmax-context stream

If two operations advance different fields by different deltas, they should usually use **different** pointer rows.

Examples from current code:

- `matmat_mul_core_pbi()` uses many pointer rows because B-load, bias-load, row traversal, writeback, softmax staging, and outer-tile traversal all advance differently.
- `rms_norm_core_dram_pbi()` uses separate pointers for:
  - full-chunk DRAM load
  - full-chunk DRAM writeback
  - inner RMS row traversal
  - inner broadcast writeback traversal
  - inner gamma-multiply traversal

## Hardware-Based For Loop Support

PBI is independent from the loop mechanism, but the two are designed to be used together.

Software provides:

- `loop_start(loop_cnt)`
- `loop_end()`

These helpers build a counted hardware loop using ISA counter and relative backward jump instructions:

- `loop_start()` allocates a counter register and emits `ADD_SET`.
- `loop_end()` emits `ADD_DEC` and `JUMP_RELA_JNZ`.

This allows a short loop body to be replayed in hardware while PBI advances addresses and sizes across iterations.

### Nested-loop recipe with `INIT` and `INC` only

Because hardware currently supports only:

- `PBI_MODE_INIT`
- `PBI_MODE_INC`

there is no direct "restore previous pointer snapshot" operation.

That means nested-loop conversion should follow this pattern:

1. Put the **full tiles** into the hardware-counted loop.
2. Keep the **remainder tile** outside that loop.
3. For the outer loop, use pointer rows whose state should persist across outer iterations.
4. Inside the outer-loop body, re-`INIT` any inner-loop pointer rows that must restart from the same base every outer iteration.
5. Use inner `loop_start()` / `loop_end()` for the row loop.
6. If an inner pointer needs a manual end-of-loop correction, use `generate_instruction_pbi_inc(...)`.

This is the core technique that makes nested PBI loops practical without a reset opcode.

In other words:

- outer-loop persistent streams: allocate once, `INIT` once, then let UE/PBI updates carry them forward
- inner-loop restartable streams: `INIT` again at the start of each outer iteration
- remainder case: handle separately in plain absolute form or with a smaller dedicated sequence

## Legacy Kernel Migration Playbook

This section is the recommended process for converting an existing legacy software-loop kernel into PBI + ISA loops.

### Step 1: Identify the loop nest

Before writing any PBI code, write down:

- which dimensions are looped in Python
- which dimensions are full-tile repeatable
- which dimensions have a remainder
- which loops are worth moving to hardware

Good candidates:

- repeated full-tile outer loops
- repeated row loops with fixed per-iteration structure

Less attractive candidates:

- loops with heavy per-iteration branching
- loops whose remainder handling dominates the body

### Step 2: Separate legacy and PBI entry points

Follow the pattern used by current converted kernels:

- `kernel_core_pbi(...)`
- `kernel_core_legacy(...)`
- `kernel_core(..., use_pbi=False)`

This is strongly recommended even if the first PBI version is incomplete.

Benefits:

- preserves a known-good fallback
- makes A/B validation easy
- keeps the migration incremental

### Step 3: Partition state into independent pointer streams

For every repeated instruction in the loop body, ask:

- which fields must stay fixed?
- which fields must advance every iteration?
- which fields advance at different rates in nested loops?

Group together only the fields that truly share one progression.

If two instructions need different progression, assign different pointer rows.

### Step 4: Convert the innermost repeat first

The easiest safe migration path is:

1. convert the innermost repeated loop to `loop_start()` / `loop_end()`
2. introduce pointer rows for that loop
3. verify correctness
4. only then move the next outer loop into hardware

This is exactly how complex kernels should be approached in future upgrades.

### Step 5: Make full-tile and remainder paths explicit

For each tiled dimension, split it into:

- `num_full_tiles`
- `remainder`

Then implement:

- one hardware-counted path for the full tiles
- one explicit residual path outside that loop

Do not try to force remainder handling into the same PBI loop body unless the pointer math stays simple and obviously correct.

### Step 6: Use `PBI_MODE_INC` only for true correction / carry-forward needs

`generate_instruction_pbi_inc(...)` is best used when:

- a pointer row must be adjusted without issuing a UE operation
- nested loops leave the pointer row at a position that needs correction
- a residual path needs a small delta tweak before reuse

Avoid using `PBI_MODE_INC` as a substitute for clear loop structure. If re-`INIT` is simpler and local, prefer re-`INIT`.

### Step 7: Keep pointer ownership local

At helper scope:

- allocate only the pointer rows that helper needs
- release exactly those rows before return

Do not end a helper with `reset_inst_ptr_counter()` unless you intentionally want to invalidate all outer allocation state.

### Step 8: Check instruction-cache footprint

After converting a loop body, measure the captured instruction count and the loop-body size returned by `loop_end()`.

For outer loops, this matters a lot. The loop body must stay within the intended instruction-cache budget.

Current kernels already use checks like:

- `assert outer_loop_size <= 256`
- or slightly larger limits depending on remainder structure

Future migrations should keep similar guards.

### Step 9: Validate against legacy behavior

For every migrated kernel:

- run legacy path
- run PBI path
- compare numerical quality
- compare instruction count
- compare timing / throughput

At minimum, record:

- instruction count
- latency
- numerical agreement (for example SNR)

### Step 10: Preserve a clear code shape

Converted kernels should stay readable. A good target structure is:

- parameter / tiling calculation
- pointer allocation
- small local emit helpers
- full-tile hardware loop
- remainder handling
- pointer release
- FLOP accounting

## Worked Migration Pattern

The most reusable migration pattern today is:

1. keep one legacy implementation unchanged
2. build a PBI implementation that first converts the inner loop
3. if beneficial, move the outer full-tile loop into ISA using:
   - one persistent pointer for outer input traversal
   - one persistent pointer for outer output traversal
   - re-`INIT` of inner pointers inside the outer-loop body
4. leave the residual tile outside the outer loop
5. release local pointer rows before return

This is the pattern used by the current `rms_norm_core_dram_pbi()` transition and is a good default model for future upgrades.

## ISA Performance Analysis

This section gives a representative cycle model for the current queue-state RTL. It is meant as a front-end scheduling estimate, not a full datapath throughput proof.

### Global front-end cost

For most instructions, the fixed control-path cost is:

- instruction-cache refill / instruction DMA: `T_inst_dma` cycles when queue mode needs a new RAM line
- `INST_FETCH`: `1` cycle
- `INST_DECODE_TYPE`: `1` cycle

So the common baseline is:

- `Global = T_inst_dma + 1 + 1`
- If the instruction is already present in instruction RAM, use `T_inst_dma = 0`
- In single-step CSR mode, use `T_inst_dma = 0`

Notes:

- Queue-mode refill is handled by `INST_RAM_DMA_START` plus `INST_RAM_DMA_BUSY`, so `T_inst_dma` is variable and depends on the AXI / memcpy completion time.
- `INST_FETCH` and `INST_DECODE_TYPE` are each single-state, single-cycle front-end stages in the current RTL.

### Representative instruction costs

Let `Tbusy` denote backend completion time in `INST_BUSY`.

| Instruction type | Representative cycle model | Notes |
| --- | --- | --- |
| `INST_TYPE_HALT` | `Global + 1` | Enters `INST_HALT`, then returns to `INST_IDLE`. No UE backend busy phase. |
| `INST_TYPE_SWI` | `Global + 1` | `INST_SWI` is a one-cycle sideband event, then returns to fetch. |
| `INST_TYPE_ADD` | `Global + 1` | `INST_ADD` writes the regfile and returns to fetch in the same state. |
| `INST_TYPE_SEMAPHORE` set / clear | `Global + 1` | `SEMAPHORE_MODE_SET` and `SEMAPHORE_MODE_CLEAR` are single-cycle control operations. |
| `INST_TYPE_SEMAPHORE` check | `Global + N` | Polls in `INST_SEMAPHORE` until the target engine flag is observed high. |
| `INST_TYPE_JUMP` absolute / cond taken | `Global + 1 + T_inst_dma(next)` | One cycle in `INST_JUMP`, then typically restarts queue fetch through RAM DMA at the target address. |
| `INST_TYPE_JUMP` cond not taken | `Global + 1` | Falls through directly back to `INST_FETCH`. |
| relative `INST_TYPE_JUMP` | `Global + 2` | `INST_JUMP` plus `INST_JUMP_ADDR`, then back to fetch. |
| `INST_TYPE_REG_REWRITE` | `Global + 1 + 1 + 4 + Tbusy` | One cycle in `INST_REWRITE`, then behaves like a UE instruction through decode-reg, execute, and busy. |
| `INST_TYPE_PBI_SET` | `Global + 1` | `INST_PBI_UPDATE` runs once (load for `PBI_MODE_INIT`, add for `PBI_MODE_INC`) then returns to fetch. |
| `INST_TYPE_UE_OP` | `Global + 1 + 4 + Tbusy` | `INST_DECODE_REG` = 1 cycle, `INST_EXECUTE` = 4 cycles, then backend wait in `INST_BUSY`. |
| `INST_TYPE_UE_PBI` | `Global + 1 + 4 + 1 + Tbusy` | Same as `UE_OP`, plus one `INST_PBI_UPDATE` cycle to write back the incremented pointer row before entering `INST_BUSY`. |

### Representative decompositions

Using the shorthand above:

- `INST_TYPE_HALT = T_inst_dma + 1 fetch + 1 decode_type + 1 halt`
- `INST_TYPE_PBI_SET = T_inst_dma + 1 fetch + 1 decode_type + 1 pbi_update`
- `INST_TYPE_UE_OP = T_inst_dma + 1 fetch + 1 decode_type + 1 decode_reg + 4 execute + Tbusy`
- `INST_TYPE_UE_PBI = T_inst_dma + 1 fetch + 1 decode_type + 1 decode_reg + 4 execute + 1 pbi_update + Tbusy`
- `INST_TYPE_REG_REWRITE = T_inst_dma + 1 fetch + 1 decode_type + 1 rewrite + 1 decode_reg + 4 execute + Tbusy`

### What `Tbusy` means in practice

`Tbusy` is where almost all operation-specific latency lives:

- for DRAM-to-URAM / BRAM memcpy, it is dominated by DMA completion
- for URAM compute ops, it is dominated by the UE arithmetic engine completion
- for URAM-to-DRAM writeback, it is dominated by writeback engine completion
- for synchronization instructions like semaphore-check, it is the time spent waiting for the external condition

### PBI vs non-PBI front-end overhead

For a single representative UE instruction:

- `UE_OP` front-end cost after fetch/decode is `1 decode_reg + 4 execute = 5` cycles before busy wait
- `UE_PBI` front-end cost after fetch/decode is `1 decode_reg + 4 execute + 1 pbi_update = 6` cycles before busy wait

So PBI adds about `1` extra control cycle per instruction versus plain `UE_OP`

## Minimal Usage Pattern

The standard PBI pattern is:

1. Allocate a pointer index.
2. Emit one `PBI_SET` with **`PBI_MODE_INIT`** for that index (`generate_instruction_pbi_init`).
3. Emit regular helper/core instructions with the same `inst_pointer_idx` (`INST_TYPE_UE_PBI` in hardware).
4. In those pointer-backed instructions, pass deltas for PBI-supported fields.
5. Optionally emit **`PBI_SET` with `PBI_MODE_INC`** (`generate_instruction_pbi_inc`) when you need to advance the pointer row **without** scheduling a UE op in the same instruction.
6. If the body is repeated, wrap it with `loop_start()` and `loop_end()`.

Example:

```python
p = ue.alloc_inst_ptr()

ue.generate_instruction_pbi_init(
    dram_shared_addr=base_dram_addr,
    dma_length=base_length_bytes,
    output_size=base_output_size,
    uram_length=base_uram_length,
    uram_a_start_addr=base_uram_a,
    uram_b_start_addr=base_uram_b,
    uram_wb_addr=base_uram_wb,
    uram_dst_addr=base_memcpy_dst,
    fmax_context_addr=base_fmx_context,
    inst_pointer_idx=p,
)

ue.loop_start(loop_cnt)

ue.accelerator_memory_to_sram(
    accelerator_dram_address=dram_delta_bytes,
    sram_address=dst_sram_addr,
    element_size=0,
    inst_pointer_idx=p,
)

ue.ue_arithmetic_op(
    broadcast_mode=0,
    max_clear_en=0,
    stride_z=1,
    lalu_a=0,
    lalu_b=0,
    lalu_mode=0,
    scalar=0,
    uram_section=0,
    uram_dst_addr=0,
    uram_wb_addr=uram_wb_delta,
    uram_write_src=0,
    mode=some_mode,
    data_type=0,
    uram_a_start_addr=uram_a_delta,
    uram_b_start_addr=uram_b_delta,
    uram_length=uram_len_delta,
    dma_start_addr=dram_delta_bytes,
    dma_length=dma_len_delta,
    output_size=output_delta,
    inst_pointer_idx=p,
)

ue.loop_end()
```

That example is intentionally minimal. For real kernels, prefer the migration playbook above and assume you may need **multiple** pointer rows plus a separate residual path.

## Common Pitfalls

- Do not use pointer index `0` for user-managed PBI state.
- Do not use a single pointer row for multiple streams that advance differently.
- Do not assume every descriptor field is pointer-backed.
- Do not pass absolute DRAM addresses to pointer-backed memcpy helpers after `PBI_SET`; those arguments become increments.
- Do not assume PBI changes `mode_sel`, `data_type`, `broadcast_mode`, or activation control fields.
- Do not forget that `INST_TYPE_UE_PBI` (`4'h5`) both consumes the pointer row for the UE op and then updates it with increments.
- Do not fold full-tile and remainder logic together unless the pointer math is clearly correct.
- Do not forget to re-`INIT` inner-loop pointers when an outer-loop iteration needs them to restart from a local base.
- Do not call `reset_inst_ptr_counter()` from a helper that should only release its own local pointer rows.

## Summary

The current PBI mechanism is a hardware-supported pointer-row update system for a limited subset of descriptor fields.

- Type **`6`** (`PBI_SET`): updates the pointer row only—either **load** (`[19:16] == PBI_MODE_INIT`) or **increment** (`[19:16] == PBI_MODE_INC`) from the descriptor’s pointer-backed fields, selected by `[15:12]`.
- Type **`5`** (`UE_PBI`): runs a UE operation using pointer-backed values, then **increments** the row by the descriptor deltas.
- Software: `generate_instruction_pbi_init()` for init; `generate_instruction_pbi_inc()` for increment-only; then memcpy / `ue_arithmetic_op` / wrappers with non-zero `inst_pointer_idx` for `UE_PBI` ops.
- PBI is most useful when paired with hardware-counted loops.
