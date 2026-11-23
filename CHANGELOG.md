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

**No extra weight memory (online leaves).** The weight is stored once, cut into its 4^levels cells. Before the matmul the board makes every leaf that is a sum on chip (a 4-bit cell is dequantized to fp16 as it is loaded), and writes it to a scratch all layers share; the pieces are dealt out over the engines. Measured on 8 engines, time relative to the plain matmul:

| Layer | Shape (rows per engine x K -> N) | Stored L1 | Online L1 | Stored L2 | Online L2 | Sums alone, L1 / L2 |
|-------|----------------------------------|-----------|-----------|-----------|-----------|---------------------|
| 4-bit | 408 x 2048 -> 2048 | 0.91x | 0.92x | 0.89x | 0.91x | 0.28 / 0.77 ms |
| 4-bit | 408 x 4608 -> 2048 | 0.89x | 0.91x | 0.82x | 0.84x | 0.62 / 1.72 ms |
| 4-bit | 832 x 4608 -> 2048 | 0.89x | 0.91x | 0.82x | 0.84x | same |
| bf16 | 408 x 4608 -> 2048 | 0.86x | 0.87x | 0.79x | 0.81x | 1.02 / 3.13 ms |
| bf16 | 720 x 4352 -> 1152 | 0.88x | 0.89x | n/a | n/a | 0.58 ms |

Online costs 1-2 points of speed against stored leaves and no weight memory (stored: 2.9x / 5.7x on a 4-bit layer, 1.75x / 3.06x on bf16). Accuracy: a 4-bit layer stays at the plain 4-bit matmul's 20.1 dB at both levels (stored IF8 sums: 20.0 / 19.8 dB); a bf16 layer is the same as stored leaves, 54.5 / 52.4 dB. The sum scratch is 5 (level 1) or 45 (level 2) cells of fp16, one buffer for the whole model.

**When to use it.** Both widths 1024 or more and halving into whole 64-lane blocks, several hundred rows per call, and a matmul that is a large share of the model's time. Level 2 when one side is 2048 or wider, otherwise level 1. Level 3 never won. A width that needs padding to halve is not worth it.

**Fused activation.** `activation=` (the `matmat_mul_core` arguments, e.g. `{"gelu_enable": True}`) with `identity_addr=` (a 64 x 64 bf16 identity in DRAM). The output quarters that end in a product take the activation fused. The rest (1/2 of the output at level 1, 3/4 at level 2) go through an on-chip identity product, because the hardware ignores the activation on elementwise ops. Measured cost of that pass: 3.1-4.1 ns per value, 55 dB for GELU, SiLU and ReLU. The pass is a hardware loop, a few dozen instructions per call instead of one per 64 values; board: bit-identical to the unrolled form for GELU, SiLU and ReLU, same time, 200,704 values in 39 instructions instead of 3,148. Board, one engine, bf16, time vs the native matmul with the same fused activation: GELU 720 x 1152 -> 4352 level 1 0.96x (53.8 dB, native 54.3); ReLU 408 x 2048 -> 4608 level 1 0.90x (54.5 dB), level 2 0.87x (52.6 dB, native 55.1). Online leaves, 8 engines, 408 rows per engine: GELU 2048 -> 4608 4-bit level 1 0.93x, level 2 0.90x (19.8 dB vs bf16, same as plain 4-bit); bf16 level 1 0.91x (53.7 dB), level 2 0.89x (51.9 dB, native 54.3). SiLU 2048 -> 2048 4-bit 0.94x at both levels.

**In models.** Alpamayo 2 Edge, vision `o_proj` / `fc2` and projector `fc2` at level 1 (on by default): 73.2 s -> 72.6 s per plan, same output. Parakeet, 128-row clips: 1.8%. Swin, last stage only: 0.4%. Both are off by default (`--strassen-level`). Alpamayo 2 Edge prefill (4-bit, online leaves, on by default): level 2 on fc1 / fc2, level 1 on q / k / v / o, 72.5 s -> 69.6 s per plan on 8 engines, same 23 tokens, trajectory within 0.026 m. The weights are stored once as input-column slabs that decode also reads (decode step 29.9 -> 30.3 ms); weight region 2731 of 2752 MB, program 80-87 MB per engine.

Strassen on vs off, same session (board, U50):

| Model | Engines | Gain | Default |
|---|---|---|---|
| Qwen2.5-Omni 7B `--speak`, Token2Wav DiT | 8 | 71.2 -> 64.0 s (10.1%) | on |
| pi05 prefix MLP | 8 | 13.5 -> 12.9 s total (4.4%) | on |
| Gemma 4 E4B / E2B prefill, 500-row prompt | 8 | 4.1% / 2.0% (decode 0.5% slower) | on |
| Qwen2.5-VL 3B prefill | 8 | 3.30 -> 3.19 s (3.3%) | on |
| FLUX.2 klein 4B DiT, 512 x 512 | 8 | 90.7 -> 88.4 s (2.5%) | on |
| LocateAnything 3B vision | 1 | 1.0% (256 rows) / 2.7% (896 rows) | vision on, LM off |
| BigCodec, Qwen3.5 2B vision, Parakeet, Swin | 1 | none or slower: one engine builds every call's weight sums | off |

**API** (all in `nn_lib.py`):

| Function | Purpose |
|----------|---------|
| `strassen_levels(K, N, max_level)` | Deepest level the shape allows |
| `strassen_weight_tree(W, b, levels, store_w, store_b)` | Host, once: the fp16 leaf weights of a bf16 layer |
| `strassen_weight_tree_if4(data, scales, N, K, b, levels, store_pack, store_b)` | Host, once: the leaf weights of a 4-bit layer (IF4 blocks, IF8 sums) |
| `strassen_if4_cells` / `strassen_bf16_cells`, `strassen_online_tree`, `strassen_online_sums_core`, `strassen_online_items` | Online leaves, any level, 4-bit or bf16 layer: store the weight once as its cells, emit the sums, build the tree |
| `strassen_scratch_elements(M, K, N, levels)` | Scratch size per depth |
| `strassen_matmul_core_dram(ue, M, K, N, A, tree, OUT, scratch)` | Emits the program; row-major in and out, bias included |

**Rules.** K and N multiples of 128 * 2^(levels-1); rows a multiple of 2^levels and fixed at compile time (leftover rows go through the plain matmul); K below 8192; output address 128-byte aligned; scratch private per engine and zero-filled once. A model that also decodes one row at a time can store a 4-bit weight as input-column slabs (`strassen_if4_slab_cells`) that both paths read.

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
