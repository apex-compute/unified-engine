# Changelog

## [Unreleased] — PR #29

### PBI (Pointer-Backed Instruction) — Dynamic M / seq_len

Previously, a separate instruction program was compiled for every possible sequence length bucket. Programs now use GPR-driven (general-purpose register) iteration counts — the M / seq_len is written into a register at runtime, so **one compiled program handles any sequence length**. This eliminates the per-bucket instruction image and shrinks the instruction binary by roughly 10×.

Gemma3 is fully migrated to this mode. The prefill program is compiled once and dispatched for any prompt length up to `prefill_max_seq_len` (192 tokens). Three dedicated GPRs carry runtime state:

| GPR | Name | Purpose |
|-----|------|---------|
| 1 | `TMP_REG` | Scratch (address patching) |
| 2 | `GF_SEQ_LEN_REG` | Current sequence length (drives all matmul M counts) |
| 3 | `GF_Q_SEQ_LEN_REG` | Q-head RMS row count |
| 4 | `GF_BUCKET_IDX_REG` | Flash-attention bucket selector (`seq_len // 64`) |

The `decoder_seq_len_buckets` config field is removed; `prefill_max_seq_len` is raised from 33 → 192.

---

### API Changes — `gf_M_reg` replaces `use_pbi` flag

The boolean `use_pbi: bool` parameter on core functions is replaced by an explicit `gf_M_reg: int | None` parameter. Pass the GPR index holding the runtime row count, or `None` to use a compile-time constant M (legacy / non-PBI path).

**Affected functions in `user_dma_core.py`:**

| Function | Before | After |
|----------|--------|-------|
| `rms_norm_core_dram_pbi` | no `gf_M_reg` param | `gf_M_reg: int` (required) |
| `rope_hf_core_dram` | `use_pbi: bool = False` | `gf_M_reg: Optional[int] = None` |
| `rope_hf_core_dram_pbi` | no `gf_M_reg` param | `gf_M_reg: int = None` |
| `rope_hf_core_dram_gqa` | `use_pbi: bool = False` | `gf_M_reg: Optional[int] = None` |
| `rope_hf_core_dram_gqa_pbi` | no `gf_M_reg` param | `gf_M_reg: int = None` |
| `bf16_transpose_core` | no PBI support | `use_pbi: bool = False` added |
| `flash_attention_core_pbi` | `seq_len: int` arg | removed; replaced by `gf_bucket_idx: int` GPR |
| `loop_start` | `loop_cnt: int` (required) | `loop_cnt: int = 0, gf_loop_cnt: int = None` |

**Other API changes:**
- `load_instructions()` → renamed to `load_program_instructions_from_file()`
- `rope_hf_core` and `decoder_group_attention_core_legacy/pbi` moved from `gemma3_test.py` into `user_dma_core.py` (now shared across models)
- `eltwise_core_dram` added as a unified entry point; the old split between `_legacy` and `_pbi` variants is preserved for compatibility

---

### ISA Additions (hardware-level, mostly transparent to users)

New instruction generators added to `UnifiedEngine`:

| Method | Description |
|--------|-------------|
| `generate_instruction_nop()` | Emit a NOP |
| `pad_capture_to_64b_boundary()` | Align instruction buffer to 64-byte boundary |
| `generate_instruction_reg_min(dst, src, rst)` | GPR ← min(GPR, GPR) |
| `generate_instruction_reg_sub(dst, src, rst)` | GPR ← GPR − GPR |
| `generate_instruction_reg_mul_imm(dst, src, imm)` | GPR ← GPR × immediate |
| `_patch_jump_immediate(capture_idx, target)` | Retroactively patch a jump target |

`loop_start` now accepts `gf_loop_cnt` to drive the iteration count from a GPR instead of a compile-time integer, enabling loops whose trip count is not known until runtime.

---

### TurboQuant — `tq_utils.py` (new file)

New module implementing Lloyd-Max optimal codebook computation for TurboQuant quantization:

- `compute_lloyd_max_codebook(d, bits)` — solves continuous 1D k-means under the Beta distribution that describes each coordinate of a uniform random vector on S^{d−1}
- `get_codebook(d, bits)` / `get_codebook_tensors(d, bits, device, dtype)` — cached codebook lookup
- `generate_rotation_matrix(...)` — random rotation preprocessing for TurboQuant

The codebook is dimension-aware: for head dimension `d`, the per-coordinate distribution is `Beta((d−1)/2, (d−1)/2)` rescaled to `[−1, 1]`, converging to `N(0, 1/d)` for large `d`.

---

### Gemma3 Quantization

Weight quantization in `gemma3_test.py` uses `quant_schemas.quantize` dispatched via `QUANT_PRECISION = "if4"`. Per-K-block (size 64) min-MSE selection between INT4 (−8..7) and FP4 (NVFP4 E2M1) is applied to all quantizable weights (Q/K/V/O projections, MLP gate/up/down, LM head); the bf16 scale's sign bit selects the codebook on the FPGA. Norms and embeddings stay BF16. To switch the whole model to a different codebook, change `QUANT_PRECISION` (valid values: `int4`, `fp4`, `if4`).
