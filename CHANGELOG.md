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

A drop-in replacement for one large `matmat_mul_core` call that does 7 half-size products instead of 8, nested up to 3 levels. Alpamayo 2 Edge uses level 1 by default on its vision `o_proj` / `fc2` and projector `fc2`; no other model uses it.

**fp16 intermediates.** The input, the layer's weight and the output stay bf16; the input sums, the weight sums (leaf weights), the 7 products and the recombination are stored in fp16. In bf16 those sums cost about 11 dB per level; in fp16 one level is within 1 dB of the plain matmul at the same speed. Board, one engine, 720 x 4352 -> 1152 with a bias: plain 55.1 dB, level 1 54.5 dB at 0.88x of the plain time (43.9 dB when the intermediates were bf16); 408 x 4608 -> 2048: 54.5 / 52.4 / 48.4 dB at 0.86x / 0.79x / 0.80x for levels 1 / 2 / 3. `strassen_weight_tree` hands `store_w` / `store_b` fp16 tensors to upload as they are; `weight_format=None` keeps 4-bit leaves. The bf16 SNR column of the table below is the old all-bf16 kernel.

**4-bit (IF4) weights.** `strassen_weight_tree_if4` takes the layer's IF4 pack. The two leaves per level that are an untouched block of the weight are cut out of the pack and stay 4-bit; the five that are sums are formed from the dequantized weight and stored IF8, because the sum of two 4-bit blocks does not fit a 4-bit grid again. Board, one engine, 408 rows, level 1, against the plain IF4 matmul (20.1 dB vs bf16): 20.0 dB at 0.87-0.90x of its time for 2.9x the weight bytes (5.7x / 10.3x at levels 2 / 3). Re-quantizing every leaf to 4 bits instead gives 14.1 / 8.1 / 2.0 dB at levels 1 / 2 / 3.

**When it is worth trying:** a matmul with K and N of a thousand or more that halves into whole 64-lane blocks, several hundred rows per call, and a large share of the model's time. Do not use it on small matmuls or on a matmul with a fused activation. Measured in whole models: Alpamayo 2 Edge vision 73.2 -> 72.6 s; parakeet (128 rows) 1.8% and swin (2 of 24 layers) 0.4%, both left off by default.

**API** (all in `nn_lib.py`):

| Function | Purpose |
|----------|---------|
| `strassen_levels(K, N, max_level)` | Deepest level the shape allows |
| `strassen_weight_tree(W, b, levels, store_w, store_b)` | Host, at weight-init: forms and stores the 7^levels leaf weights |
| `strassen_weight_tree_if4(data, scales, N, K, b, levels, store_pack, store_b)` | Same for a 4-bit layer: untouched blocks stay IF4, sums are stored IF8 |
| `strassen_scratch_elements(M, K, N, levels)` | Scratch size per depth |
| `strassen_matmul_core_dram(ue, M, K, N, A, tree, OUT, scratch)` | Emits the program; row-major in, row-major out, bias included |

**How it works.** The weight is split into quarters and the 7 Strassen weight combinations are formed on the host in fp32, then stored (and quantized) one by one, so weight storage grows 1.75x per level. On the device, one level is 4 strided gathers of the input quarters, 5 elementwise sums, 7 half-size products, 4 elementwise sums and 4 strided writes of the output quarters. The last product of each output quarter adds the rest of that quarter through a full-matrix bias, so the quarters need no extra add. The formulas are in the `nn_lib.py` section header.

**Measured on the board** (Alveo U50, one matmul, 4096 x 7680 -> 8192):

| Level | Work | Weights | bf16 time | bf16 SNR | IF4 time | IF4 SNR |
|-------|------|---------|-----------|----------|----------|---------|
| 0 (plain) | 100% | 1x | 1.00x | 55.0 dB | 1.00x | ~20 dB |
| 1 | 87.5% | 1.75x | 0.82x | 43.9 dB | 0.88x | ~14 dB |
| 2 | 76.6% | 3.06x | 0.71x | 37.3 dB | 0.78x | ~8 dB |
| 3 | 67.0% | 5.36x | 0.64x | 31.2 dB | 0.70x | ~2 dB |

**Limitations found:**

- **Accuracy drops every level**: about 11 dB at level 1 and 6 dB per further level in bf16, and 6 dB per level with 4-bit weights. Errors of separate layers add up.
- **Small shapes get slower.** At K = N = 256 it is 1.26x slower. Across a 640-shape sweep it was faster on 215 shapes with one engine and on 121 with eight engines.
- **Shape rules.** K and N must be multiples of 128 * 2^(levels-1), M a multiple of 2^levels (the caller runs leftover rows through the plain matmul), K below 8192, and the output address 128-byte aligned. A half narrower than 64 lanes would have to be padded and costs more work than it saves.
- **No fused activation.** A matmul that carries GELU or a clamp cannot use it.
- **Scratch.** 9 input-quarter and 14 output-quarter buffers per depth, private per engine, zero-filled once.
- **Whole models.** Alpamayo 2 Edge at level 1: 80.4 s -> 77.8 s (3% faster), trajectory error 0.18 m -> 0.87 m; the 4-bit prefill layers cause the error, the bf16 vision layers do not. Qwen3-4B perplexity: 19.4 plain, 21.5 / 24.3 / 40.4 at levels 1 / 2 / 3.

**Status.** `nn_lib.py` holds the only implementation. It is the routine that ran on the board inside Alpamayo 2 Edge at level 1, lifted out; Alpamayo calls it (`models/alpamayo2_edge/alpamayo2_edge_test.py --strassen-level N --strassen-ops ...`, which lands with the Alpamayo 2 Edge model, is the working example of wiring it into a model). After the lift it has been checked on the host only (levels 1-3 against the exact product), not yet re-run on the board. The experiments behind the numbers (shape sweep, level probe, LLM quality test, CSVs, plots) were removed from the tree and are in git history: `git show 68189d7 --stat -- strassen/`.

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
