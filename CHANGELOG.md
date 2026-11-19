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

### Strassen matmul — `nn_lib.py` §10 (optional)

Replaces one large `matmat_mul_core` call with 7 half-size products instead of 8, nested up to 3 levels. Each level removes 12.5% of the multiplies and multiplies the weight storage by 1.75.

**Precision.** The input, the layer's weight and the output keep the model's formats. The sums and products in between are fp16, which keeps one level within 1 dB of the plain matmul (in bf16 each level cost 11 dB). On a 4-bit layer the leaf weights that are sums are stored in 8-bit (IF8) and the untouched blocks stay 4-bit: the sum of two 4-bit blocks does not fit a 4-bit grid, and squeezing it back cost 6 dB per level.

**Measured** (Alveo U50, one engine, the kernel as it is; time relative to the plain matmul of the same weights):

| Weights | Shape (rows x K -> N) | Level 1 | Level 2 | Level 3 |
|---------|-----------------------|---------|---------|---------|
| bf16 | 720 x 1152 -> 1152 | 0.95x | n/a | n/a |
| bf16 | 720 x 4352 -> 1152 | 0.88x | n/a | n/a |
| bf16 | 408 x 2048 -> 2048 | 0.90x | 0.88x | 0.97x |
| bf16 | 408 x 4608 -> 2048 | 0.86x | 0.79x | 0.80x |
| 4-bit | 408 x 2048 -> 2048 | 0.91x | 0.89x | 0.97x |
| 4-bit | 408 x 2048 -> 1024 | 0.93x | 0.93x | 1.17x |
| 4-bit | 408 x 2048 -> 4608 | 0.90x | 0.85x | 0.87x |
| 4-bit | 408 x 4608 -> 2048 | 0.89x | 0.82x | 0.82x |

| | Plain | Level 1 | Level 2 | Level 3 |
|---|-------|---------|---------|---------|
| bf16 layer, SNR vs the exact product | 55.1 dB | 54.5 dB | 52.5 dB | 48.5 dB |
| 4-bit layer, SNR vs the bf16 matmul | 20.1 dB | 20.0 dB | 19.8 dB | 18.9 dB |
| 4-bit layer, weight bytes | 1x | 2.9x | 5.7x | 10.3x |
| bf16 layer, weight bytes | 1x | 1.75x | 3.06x | 5.36x |

**No extra weight memory on a 4-bit layer (online leaves, level 1).** The weight is stored once, as its four 4-bit quarters. Before the matmul the board dequantizes two quarters to fp16 on chip, adds them and writes the sum to a scratch all layers share; the pieces are dealt out over the engines. Measured on 8 engines, 408 and 832 rows per engine: making the five sums takes 0.27 ms (2048 x 2048) to 0.61 ms (4608 x 2048), and the matmul runs at 0.92x / 0.91x of the plain 4-bit time against 0.91x / 0.89x with stored leaves. Same accuracy as the plain 4-bit matmul (20.1 dB vs bf16).

**When to use it.** Both widths 1024 or more and halving into whole 64-lane blocks, several hundred rows per call, no fused activation, and a matmul that is a large share of the model's time. Level 2 when one side is 2048 or wider, otherwise level 1. Level 3 never won. A width that needs padding to halve is not worth it.

**In models.** Alpamayo 2 Edge, vision `o_proj` / `fc2` and projector `fc2` at level 1 (on by default): 73.2 s -> 72.6 s per plan, same output. Parakeet, 128-row clips: 1.8%. Swin, last stage only: 0.4%. Both are off by default (`--strassen-level`). No 4-bit model uses it yet.

**API** (all in `nn_lib.py`):

| Function | Purpose |
|----------|---------|
| `strassen_levels(K, N, max_level)` | Deepest level the shape allows |
| `strassen_weight_tree(W, b, levels, store_w, store_b)` | Host, once: the fp16 leaf weights of a bf16 layer |
| `strassen_weight_tree_if4(data, scales, N, K, b, levels, store_pack, store_b)` | Host, once: the leaf weights of a 4-bit layer (IF4 blocks, IF8 sums) |
| `strassen_if4_cells` / `strassen_bf16_cells`, `strassen_online_tree`, `strassen_online_sums_core`, `strassen_online_items` | Online leaves, any level, 4-bit or bf16 layer: store the weight once as its cells, emit the sums, build the tree |
| `strassen_scratch_elements(M, K, N, levels)` | Scratch size per depth |
| `strassen_matmul_core_dram(ue, M, K, N, A, tree, OUT, scratch)` | Emits the program; row-major in and out, bias included |

**Rules.** K and N multiples of 128 * 2^(levels-1); rows a multiple of 2^levels and fixed at compile time (leftover rows go through the plain matmul); K below 8192; output address 128-byte aligned; scratch private per engine and zero-filled once. A model that also decodes one row at a time keeps its plain weight for that.

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
