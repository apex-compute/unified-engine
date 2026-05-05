# PBI Mode — Pointer-Backed Instruction Compression

Everything learned through experimentation across gemma3, smolvlm2, and parakeet.

---

## 1. What PBI Is

PBI (Pointer-Backed Instruction) mode replaces hardcoded DRAM addresses in the instruction stream with **pointer registers** (registers 1–15). A `PBI_INIT` instruction loads a base address and auto-increment stride into a register. Each subsequent PBI-backed instruction uses that register's current address then advances it by the stride.

The practical effect: a Python-level `for` loop over tiles that would unroll into N explicit instructions in the binary instead becomes **one loop body + hardware loop control** (`ADD_SET` / `ADD_DEC` / `JUMP_RELA_JNZ`). The instruction stream shrinks proportionally.

---

## 2. The Entry Point — `matmat_mul_core_pbi`

Location: `user_dma_core.py`, `UnifiedEngine.matmat_mul_core_pbi()`

```
matmat_mul_core_pbi(M, K, N, A_DRAM_ADDR, B_DRAM_ADDR, OUTPUT_DRAM_ADDR,
                    is_B_quantized, data_type, SCALE_DRAM_ADDR,
                    silu_enable, gelu_enable, ...)
```

Also reachable via `matmat_mul_core(..., use_pbi=True)`.

The function implements a **3-level loop nest**:
- **Outer M-tile loop**: hardware loop over `num_full_m_tiles = M // M_chunk` row tiles
- **Middle N-tile loop**: hardware loop over `num_full_n_tiles = N // N_chunk` column tiles  
- **Inner K-reduction**: handled by a single hardware dot-product instruction per tile

It allocates **13 PBI pointer registers** upfront (`PBI_INIT` × 13), then emits one loop body. At runtime the hardware loops replay that body.

### Fixed overhead per call

Every call to `matmat_mul_core_pbi` emits these instructions unconditionally:
- 13 × `PBI_INIT` (one per pointer register)
- 1 × `generate_instruction_jump_abs` (i-cache anchor before the M-tile loop)
- 1 × `ADD_SET` (M-tile `loop_start`)
- 1 × `ADD_DEC` + 1 × `JUMP_RELA_JNZ` (M-tile `loop_end`)

**Total fixed overhead: ~17 instructions = 544 bytes** that must be recovered by tile collapsing.

---

## 3. Tiling Math — Where the Savings Come From

### M_chunk

```python
M_chunk = min(M, URAM_FULL_ELEMENTS // (K + N_chunk))
```

`URAM_FULL_ELEMENTS = 262144` elements (0x1000 rows × 64-element vector size).

### N_chunk

```python
N_chunk = min(N, (URAM_NEAR_FULL_ELEMENTS // K) // UE_VECTOR_SIZE * UE_VECTOR_SIZE)
# URAM_NEAR_FULL_ELEMENTS = 262080, UE_VECTOR_SIZE = 64
```

**Critical insight: N_chunk shrinks as K grows.** Larger K consumes more URAM per row, leaving fewer N columns per tile, forcing more N-tile iterations — more to collapse with the hardware loop.

### Savings formula

```
instructions saved = (num_full_n_tiles - 1) × body_size_instructions
                   + (num_full_m_tiles - 1) × (full_n_loop_size + overhead)
net gain           = instructions_saved - 17_fixed_overhead
```

PBI wins when `net gain > 0`.

---

## 4. Concrete Numbers for smolvlm2 and parakeet

### smolvlm2 decoder projections (K=1024 or K=2560)

| Projection | M | K    | N    | N_chunk | N-tiles | M-tiles | PBI verdict     |
|------------|---|------|------|---------|---------|---------|-----------------|
| q, o       | 1 | 1024 | 1024 | 256     | 4       | 1       | marginal / loss |
| k, v       | 1 | 1024 | 320  | 256     | 1       | 1       | **loss** (+17 overhead, 0 savings) |
| gate, up   | 1 | 1024 | 2560 | 256     | 10      | 1       | saves ~9 bodies  |
| down       | 1 | 2560 | 1024 | 64      | 16      | 1       | saves ~15 bodies |

**Result: applying PBI to all 7 projections grew decoder_program.bin from 7.5MB → 8.3MB.** k/v overhead (+17 per call × 2 × 24 layers = +816 instructions per bucket) dominated.

### Why M=1 decode is fundamentally hard for PBI

With M=1, `num_full_m_tiles = 1` — the outer loop runs exactly once. **Zero M-tile collapsing**. All savings must come from the N-tile loop alone. For small models with K=1024, N_chunk=256 is large, leaving few N-tiles. The 17-instruction fixed overhead exceeds the savings.

### Hypothetical large model (K=8192, N=8192 — e.g. Llama-70B)

```
N_chunk = (262080 // 8192) // 64 * 64 = 32 // 64 * 64 → 32 (falls to 32-element path)
num_full_n_tiles = 8192 // 32 = 256
```

PBI saves 255 × body vs 17 overhead → **massive win even at M=1 decode.**

### parakeet encoder (M=L_pad, large sequence)

```
M=L_pad (~256–2048), K=1024, N=1024 → N_chunk=256, N-tiles=4, M-tiles=L_pad/M_chunk
```

With L_pad=512 and M_chunk=256: `num_full_m_tiles = 2`. With L_pad=2048: `num_full_m_tiles = 8+`. The M-tile loop collapses hundreds of row iterations. **Ideal PBI target** — same shape as gemma3 prefill where PBI already works.

---

## 5. ISA Register Rules — Critical for Multi-Op Programs

### The collision bug

`loop_start()` calls `alloc_isa_reg()` which increments `_isa_reg_counter` starting from 1. If any model reserves fixed registers (e.g. smolvlm2 reserves regs 1, 2, 3 for `V_CACHE_SIZE_REG`, `ROPE_SIZE_REG`, `TMP_REG`), the first `loop_start` call allocates reg 1 and `ADD_SET reg=1, loop_count` **silently clobbers** the fixed register. In smolvlm2 this corrupted KV cache write addresses → "The The The" repetition output.

### The fix

Before each bucket compilation, reset the counter above all reserved registers:

```python
_pbi_reg_base = max(self.V_CACHE_SIZE_REG, self.ROPE_SIZE_REG, self.TMP_REG) + 1  # = 4
self._isa_reg_counter = _pbi_reg_base
```

### Register lifecycle — no leak across calls

`loop_end()` calls `release_isa_reg()` which decrements `_isa_reg_counter`. Every balanced `loop_start` / `loop_end` pair returns the counter to its pre-call value. Calling PBI matmul 168 times (7 projections × 24 layers) in one bucket does **not** exhaust registers — the counter stays at `_pbi_reg_base` after each call.

---

## 6. Absolute Jump Address Fix

`matmat_mul_core_pbi` emits `generate_instruction_jump_abs` to anchor the outer M-tile loop to a specific i-cache position. The address is computed at compile time via `get_program_dram_addr()`.

**The problem in smolvlm2**: the fused decode program loads as:
```
[reg_set 64B] [embed_raw N bytes] [bucket_raw]
```
at `_decode_scratch_addr`. But `get_program_dram_addr()` returns `_scratch_base` — the address before reg_set and embed are laid out. The absolute jump lands `64 + embed_size` bytes (4+ instructions) **before** the actual bucket start. The M-tile loop body is never reached → garbled output.

**The fix**: pre-allocate the reg_set + embed offset **before** compiling any bucket, so `get_program_dram_addr()` returns the correct runtime address during compilation:

```python
_max_embed = max(len(r) for r in self._decode_embed_raw)
_bucket_offset = 64 + _max_embed          # reg_set (64B) + embed
_scratch_base = self.get_program_dram_addr()
self.allocate_program_dram(_bucket_offset) # advance pointer NOW

for kv_len in kv_len_buckets:
    self._isa_reg_counter = _pbi_reg_base
    self.start_capture()
    # ... compile bucket — get_program_dram_addr() now returns correct base
```

Then after the bucket loop, only allocate the remaining `max_bucket` bytes (not re-allocate the offset):

```python
max_bucket = max(len(r) for r in self._decode_bucket_raw)
self._decode_scratch_addr = _scratch_base   # point to start of whole fused program
self.allocate_program_dram(max_bucket)      # only the bucket portion remains
```

---

## 7. BF16 Async Race — When PBI Store Fails

### The pattern that breaks

```
PBI memcpy (async DMA) → matvec (async compute) → PBI store (fires immediately)
```

PBI auto-increments and fires the store instruction before the async matvec has written its output to URAM. The store reads stale / wrong-channel data.

**Symptom**: channel 0 tiny norm, channel 1 contains channel 0's data, channels 2+ zeros. SNR ≈ -0.71 dB.

**Where this hits**: parakeet per-channel depthwise conv — each channel does a memcpy → 1×L_pad matvec → store. The PBI store loop fires ahead of the compute.

**Where this does NOT hit**: quantized matmul (`is_B_quantized=True`). The K-reduction accumulates into URAM across all K tiles and only stores **once at the end**, after the compute instruction has completed. No async gap between compute and store.

### BF16 matmul PBI is fine

Standard BF16 matmul with PBI (as used in gemma3 prefill) does **not** have this race. The store is integrated into the matmul compute instruction — no separate async gap. The race only appears in the manual memcpy → compute → store pattern of the DW-conv.

---

## 8. Decision Checklist — Should I Use PBI Here?

```
1. Is M large (prefill / encoder, M >> 1)?
        YES → strong candidate, M-tile loop collapses many iterations
        NO (M=1 decode) → go to step 2

2. Is K large enough to force small N_chunk?
        N_chunk = (262080 // K) // 64 * 64
        Compute: num_full_n_tiles = N // N_chunk
        Is (num_full_n_tiles - 1) × body_size > 17 ?
        YES → PBI helps even at M=1
        NO  → PBI hurts, stay with quantized_matmat_core / matmat_mul_core

3. Is the weight BF16 or quantized?
        Quantized (IF4/IF8) → safe, no async race
        BF16 standard matmul → safe (gemma3 confirms)
        BF16 manual memcpy→compute→store loop → UNSAFE, async race

4. Does the model reserve fixed ISA registers?
        YES → set _isa_reg_counter = max(reserved_regs) + 1 before each capture
        NO  → default _isa_reg_counter = 1 is fine

5. Does the bucket load at a non-zero offset from _scratch_base?
        YES (e.g. reg_set + embed prefix) → pre-allocate that offset before
              compiling any bucket so jump_abs addresses are correct
        NO  → no special handling needed
```

---

## 9. Model-by-Model Summary

| Model | Op | M | PBI useful? | Notes |
|-------|----|---|-------------|-------|
| gemma3 | prefill matmuls | seq_len (large) | **Yes** | Already implemented, working |
| gemma3 | decode matmuls | 1 | No | K=3072, small model |
| parakeet | encoder attn+FF matmuls | L_pad (large) | **Yes** | Unimplemented, clear next target |
| parakeet | DW-conv per channel | 1 | No | Async race, M=1 both |
| smolvlm2 | prefill matmuls | seq_len | Yes | Not yet implemented |
| smolvlm2 | decode projections | 1 | No | K=1024 too small, file grew 7.5→8.3MB |
| Llama-70B (hypothetical) | decode matmuls | 1 | Yes | K=8192 → N_chunk=32 → 256 N-tiles |

---

## 10. smolvlm2-Specific Implementation Notes

When adding PBI to smolvlm2's `compile_decoder`, all 4 changes must be applied together — they are interdependent:

1. **`lm_matmul` helper**: add `pbi=False` flag, route `pbi=True` through `matmat_mul_core(..., use_pbi=True)` instead of `quantized_matmat_core`
2. **ISA reg fix**: set `_isa_reg_counter = 4` before each bucket (regs 1–3 are smolvlm2 fixed regs)
3. **Jump address fix**: pre-allocate `64 + max_embed` bytes before bucket loop; save `_scratch_base`
4. **Scratch fix**: after bucket loop, set `_decode_scratch_addr = _scratch_base`; allocate only `max_bucket` remaining

Missing any one of these produces silent corruption: wrong KV cache addresses (reg collision) or M-tile loop never executing (jump mismatch). Both manifest as repetitive / degenerate text output ("The The The").

The smolvlm2 config (`smolvlm2_config.json`) defines:
```json
"V_CACHE_SIZE_REG": 1,
"ROPE_SIZE_REG": 2,
"TMP_REG": 3
```
Any model with fixed ISA registers needs the same register fence.
