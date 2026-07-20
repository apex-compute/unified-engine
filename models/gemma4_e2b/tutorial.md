# Gemma4 E2B on the Unified Engine — Code Tutorial (`gemma4_e2b_test.py`)

A deep walkthrough of the **live** driver `gemma4_e2b_test.py`: the overall
architecture, the per-stage computation and memory-copy (DMA) flow for the
**build**, **prefill**, **encode** (vision / audio) and **decode** stages,
exactly where the code calls a *hardware core* versus where it computes on the
*host*, and a list of run-time inefficiencies found while reading. A final
section summarizes how the deploy-only `gemma4_e2b_run_from_bin.py` differs.

> Why `test.py`? It is the canonical, runnable end-to-end script: it builds the
> weight bin + instruction bin from the HF model on first run, then executes.
> `gemma4_e2b_run_from_bin.py` is an execute-only mirror whose `__main__` is
> currently **deprecated** (it still references attention cores that were
> removed in the `unified_attention_core` migration — see §11).
>
> Scope note. This tutorial covers the model glue in `models/gemma4_e2b/`. The
> hardware primitives it calls live in the repo-root `user_dma_core.py`
> (`UnifiedEngine`) and `quant_lib.py`; they are referenced but not
> re-documented in full.

---

## 0. TL;DR orientation (read this first)

* **One script, one build, one runtime.** `gemma4_e2b_test.py::main`
  ([:9170](gemma4_e2b_test.py:9170)) does: cold-start **build** (if the bins are
  missing) → **load** the instruction bin → optional **encoder** (vision/audio)
  → **prefill** → **decode**. `serve_openai.py` wraps this same runtime in an
  HTTP server; `gemma4_e2b_numeric.py` is a vision-encoder SNR verifier.

* **Everything is an "instruction bin."** The engine is programmed by *emitting
  ISA* into a capture buffer (`start_capture()` … `stop_capture()`), serializing
  it to `gemma4_e2b_bin/programs.bin`, DMA-ing that to card DRAM, and telling the
  FPGA to `start_execute_from_dram(addr)`. The Python **compiles a program once**
  and **replays** it — it does not stream ops one at a time at run time.

* **One prefill template, one dynamic decoder.** Thanks to PBI/GPR (§2.4) the
  prefill program is seq-len-agnostic (works for any prompt ≤ `prefill_max_seq_len`)
  and the decoder is a **single** program that serves every context length up to
  `max_context_size`, self-advancing its position counter each step.

* **Greedy by construction.** The LM head ends in a **hardware argmax**
  (`get_arg_max_index`); logits never cross PCIe. That is why the server rejects
  `temperature > 0`. An optional on-FPGA repetition penalty is applied as the
  LM-head matmul's bias term, still with no logit readback.

---

## 1. Directory map

| File | Role |
|---|---|
| `gemma4_e2b_config.json` | Model dims + DRAM region table + fixed ISA-register bindings. Single source of truth for shapes/offsets. |
| **`gemma4_e2b_test.py`** | **Live build+run.** Loads HF, quantizes weights → `params.bin`, emits `programs.bin`, then executes. Uses `unified_attention_core`. |
| `gemma4_e2b_run_from_bin.py` | Execute-only deploy mirror (deprecated `__main__`; see §11). |
| `serve_openai.py` | Long-running OpenAI-compatible chat server over the same runtime. |
| `audio_primitives.py` | Stateless ISA-emit helpers for the audio Conformer (GLU, SiLU, depthwise-conv-as-Toeplitz-matmul, chunked eltwise/copy). |
| `gemma4_e2b_numeric.py` | Vision-encoder numeric check: FPGA readbacks vs an IF4-quantized HF oracle and full-precision HF (`calculate_snr`). |
| `gemma4_e2b_bin/` | Artifacts: `programs.bin(.json)` (unified ISA, ~14.4 MiB full multimodal), `params.bin(.json)` (~7 GB weights: `[LM | vision | audio | host]`), `tokenizer/`. |

---

## 2. The execution model (how the Python talks to hardware)

Four mechanisms make every stage legible.

### 2.1 Program capture, not eager execution

`Gemma4_UnifiedEngine` ([:964](gemma4_e2b_test.py:964)) subclasses `UnifiedEngine`.
A "compile" method (`compile_prefill`, `compile_decoder`, `compile_vision_layer`,
…) does **not** run anything — it appends 32-byte ISA words into the capture
buffer. `compile_instruction_bin` ([:8185](gemma4_e2b_test.py:8185)) wraps all of
them in one `start_capture()`/`stop_capture()` session and writes
`programs.bin`. At run time, `load_instruction_bin` ([:8661](gemma4_e2b_test.py:8661))
DMAs the bin to the baked base address; each stage then sets a few registers and
calls `start_execute_from_dram(addr)`.

### 2.2 Two memories: DRAM (card) and URAM (on-chip SRAM)

* **DRAM** (up to 4 GB, byte-addressed `0x0…0xFFFFFFFF`) holds weights,
  activations, KV cache, and the program.
* **URAM** is the on-chip scratchpad the compute cores read/write, banked as
  `URAM_A` (base `0x00000`) and `URAM_B` (base `0x80000`). Capacities are
  `URAM_FULL_ELEMENTS` / `URAM_NEAR_FULL_ELEMENTS`
  ([user_dma_core.py:147](../../user_dma_core.py:147)).

Every compute op is really **DMA in → compute → DMA out**:

| Python call | Direction | Meaning |
|---|---|---|
| `accelerator_memory_to_sram(dram, sram, n)` | DRAM→URAM | load `n` bf16 elements ([user_dma_core.py:1607](../../user_dma_core.py:1607)) |
| `sram_to_accelerator_memory(sram, dram, n)` | URAM→DRAM | store; supports strided scatter |
| `accelerator_memcpy(src, dst, bytes)` | DRAM→DRAM | via URAM[0] staging ([user_dma_core.py:1693](../../user_dma_core.py:1693)) |
| `dma_to_accelerator_memory(dram, tensor)` | host→DRAM | PCIe upload |
| `dma_from_accelerator_memory(dram, shape)` | DRAM→host | PCIe readback |

The `_emit_sram_*_chunked` helpers ([:1164](gemma4_e2b_test.py:1164)) exist because
a row-block must fit a URAM bank; they tile a DRAM buffer into URAM-sized chunks
and loop load→op→store (each URAM bank holds 262144 bf16 elements, so the uniform
chunk is 131072).

### 2.3 Compute cores (the "hardware callsites")

These `UnifiedEngine` methods each emit one op's ISA — the lines to look for when
asking "where does the FPGA actually compute":

| Core | Computes | Fusions |
|---|---|---|
| `matmat_mul_core` / `quantized_matmat_core` | `A @ Bᵀ` (B stored `N×K`, implicit transpose) | GELU/SiLU/sigmoid/clamp/log, bias `C`, inline dequant of IF4/IF8 `B` |
| `rms_norm_core` (URAM) / `rms_norm_core_dram` (DRAM) | RMSNorm (optional γ) | — |
| `rope_hf_core_dram` / `rope_hf_core_dram_gqa` / `rope_hf_core_decode` | RoPE rotation from cos/sin | GPR-driven cos/sin base |
| `unified_attention_core` | full SDPA: Vᵀ, scaled-Q, `QKᵀ+bias`, softmax, `P·Vᵀ` | dynamic batch/aligned-len/head_dim/scale via GPRs ([user_dma_core.py:6998](../../user_dma_core.py:6998)) |
| `eltwise_add/mul/sub_core`, `broadcast_mul/add` | vector ops in URAM | — |
| `bf16_transpose_core`, `patching_core` | transpose / im2col | — |

Weights `B` are **block-quantized IF4** (4-bit; IF8 = 8-bit). The INT-vs-FP
variant is encoded in the sign bit of each block's bf16 scale
([TYPE](../../user_dma_core.py:176)). `matmat` dequantizes inline, so activations
stay bf16 and only weights are 4-bit in DRAM.

### 2.4 PBI / GPR — the "dynamic" (seq-len-agnostic) programs

The decisive mechanism on this branch. **The decoder is one program that works
for every context length**, and prefill is **one template** for every prompt
length ≤ `prefill_max_seq_len`:

* `matmat_mul_core(..., gpr_M_reg=self.gpr_seq_len)` makes the outer row-loop trip
  count a **runtime register**, not a compile-time constant. The captured program
  has no static `M`; the `M` argument is FLOP-accounting only.
* Address math ("V cache at `decode_pos`") is emitted as
  `reg_mul_imm(TMP, gpr_seq_len, stride)` + `add_imm(TMP, base)`, then the memcpy
  takes its DRAM address from the register (`general_reg_src=`).
* `loop_start(gpr_loop_cnt=…)` / `loop_end()` build hardware loops with a
  register trip count — used for GQA duplication and strided copies
  (`_emit_gqa_duplicate_pbi` [:7260](gemma4_e2b_test.py:7260),
  `_emit_strided_copy_pbi` [:7303](gemma4_e2b_test.py:7303)).
* The fixed register file is declared in config
  ([gemma4_e2b_config.json:2](gemma4_e2b_config.json:2)) and bound in `__init__`:
  `TMP_REG=1`, `GPR_SEQ_LEN_REG=2`, `GPR_Q_SEQ_LEN_REG=3`, `GPR_BUCKET_IDX_REG=4`,
  `GPR_ALIGNED_SEQ_LEN_REG=5`. The auto-allocator starts at index 5.

Consequence: the decoder ISA is captured **once** (templated at
`MAX_CONTEXT_SIZE`), `run_decoder` primes `gpr_seq_len` **once** to the prompt
length, and the program **self-advances** it with a trailing
`add_inc(gpr_seq_len)` each step ([:8164](gemma4_e2b_test.py:8164)).

---

## 3. Model architecture & DRAM map

Fixed 4 GB layout, chosen in `__init__` ([:967](gemma4_e2b_test.py:967)):

```
0x00000000 – 0x64000000  Weight: LM        (1600 MB)  ← params_dram_base
0x64000000 – 0x6c000000  Weight: Vision    (128 MB)
0x6c000000 – 0x78000000  Weight: Audio     (192 MB)
0x78000000 – 0x88000000  Activation scratch(256 MB)   ← tensor_dram_base
0x88000000 – 0x98000000  KV cache          (256 MB)
0x98000000 – 0xa0000000  ISA: audio        (128 MB)
0xa0000000 – 0x100000000 ISA: unified bin  (1.5 GB)   ← program_dram_base
```

Model dims (config `file_info` / `model`):

* **35 layers**, hidden 1536, **group_size 8** (GQA: 8 query heads share 1 KV head).
* `head_dim = 512` for **full-attention** layers `{4,9,14,19,24,29,34}`,
  `head_dim_sliding = 256` for the other 28 **sliding-window** layers (window
  512). KV cache is uniformly sized at `k_size = 512·2` bytes/token.
* MLP intermediate 6144, **widening to 12288 from layer 15** (`double_wide_mlp_first_layer`).
* **Dual RoPE**: *local* (θ=10000, full rotation over 256) and *global* (θ=1e6,
  partial rotary 0.25 → rotates 128 of 512) for the full-attn layers.
* **Per-layer input injection** (Gemma-4 feature): each layer mixes in a
  precomputed per-layer embedding via a gate/projection — see §6.4.
* Vocab 262144; embeddings are a zero-copy `torch.frombuffer` **mmap view** into
  `params.bin` ([weight_init:6969](gemma4_e2b_test.py:6969)) so a decode-token
  lookup touches ~3 KB, not 770 MB.

---

## 4. Build stage — HF weights → two bins

Runs only in `test.py` (needs the HF model, fetched by `_ensure_hf_model`).

### 4.1 `params.bin` — weights (`weight_bin_generate`, [:268](gemma4_e2b_test.py:268))

* **[host]** load HF `language_model`; scale embeddings by `√hidden_size`
  ([:309](gemma4_e2b_test.py:309)).
* **[host]** per layer, pad Q/K/V to the *max* (full-attn) sizes and MLP gate/up to
  the *wide* size so every layer has a uniform byte stride
  ([:344](gemma4_e2b_test.py:344)); KV-shared layers get zero placeholders for K/V
  ([:332](gemma4_e2b_test.py:332)).
* **[host]** block-quantize projection weights to **IF4** (`_parallel_quantize`
  [:239](gemma4_e2b_test.py:239)); norms/scalars stay bf16.
* Sibling builders fold the other bin sections: `_build_vision_section_bytes`
  ([:560](gemma4_e2b_test.py:560)), `_build_audio_section_bytes`
  ([:688](gemma4_e2b_test.py:688)), and `_build_host_section_bytes`
  ([:876](gemma4_e2b_test.py:876)) — the last holds tensors the *host* still needs
  at run time (per-layer embed table, per-layer model-projection, layer scalars,
  the KV-sharing map). `tokenizer_subset_extract` ([:844](gemma4_e2b_test.py:844))
  writes the minimal tokenizer/processor files.

### 4.2 `programs.bin` — the ISA (`compile_instruction_bin`, [:8185](gemma4_e2b_test.py:8185))

One capture session containing, in order:
1. the **prefill template** (`compile_prefill`),
2. the **dynamic decoder** (`compile_decoder`),
3. optionally the **vision** and **audio** encoder ISA + vision RoPE tables.

`main` builds the **complete LM+vision+audio** bin on first run (using
`DEFAULT_IMAGE`/`DEFAULT_AUDIO` only for their canonical *shape* — vision is forced
to 2520 patches, audio to a fixed `T_raw`; the actual pixels/mel are uploaded
fresh at run time) unless `GEMMA4_LM_ONLY_BIN=1`
([main:9271](gemma4_e2b_test.py:9271)). The manifest records each section's baked
absolute start address; **PBI JUMP targets are baked against
`get_program_dram_addr()` at capture time**, so the loader must DMA the bin to
that exact base.

### 4.3 DRAM population (`weight_init` [:6920](gemma4_e2b_test.py:6920), `tensor_init` [:7042](gemma4_e2b_test.py:7042))

Weights streamed to DRAM in 64 MB DMA chunks; RoPE cos/sin tables generated on
host and uploaded (`_load_rope_host` [:6848](gemma4_e2b_test.py:6848)); KV cache
zeroed; a 64×64 identity matrix uploaded once (used by attention's Vᵀ path) and
cached so redundant re-writes are suppressed (`dma_write` override
[:1153](gemma4_e2b_test.py:1153)).

---

## 5. Per-stage compute & memcpy flow

Legend: **[HW]** = FPGA compute-core callsite, **[DMA]** = memory copy, **[host]** =
torch on CPU.

### 5.1 Prefill (`compile_prefill` [:7334](gemma4_e2b_test.py:7334), `run_prefill` [:7746](gemma4_e2b_test.py:7746))

#### Host preamble (`run_prefill`)

1. **[host]** token → embedding lookup (mmap view).
2. **[host]** *multimodal merge*: overwrite image (`mm_type==1`)/audio
   (`mm_type==3`) placeholder rows with encoder soft-tokens.
3. **[DMA]** upload embeddings → `LAYER0_INPUT_DRAM`.
4. **[host]** per-layer inputs `[seq,35,256]` (`_compute_per_layer_inputs`
   [:7703](gemma4_e2b_test.py:7703)), permute to `[35,seq,256]` (layer-contiguous),
   **[DMA]** → `PER_LAYER_INPUTS_DRAM`.
5. **[host]** build **two** bias matrices — `full` (causal) and `sliding` (causal ∧
   within window), in q-position space — **[DMA]** upload both.
6. **[HW-preamble]** prime `gpr_seq_len`, `gpr_q_seq_len`, `gpr_bucket_idx`,
   `gpr_aligned_seq_len` with the actual (padded) lengths.
7. `program_execute(prefill_addr)` runs all 35 layers on-device.

`run_prefill_bucketed` ([:8919](gemma4_e2b_test.py:8919)) wraps this: pad the
prompt to `prefill_max_seq_len` with the last real token (the static template's
per-token loops need valid data at every row), `software_reset()` the queue,
re-zero KV/flash/identity (undo any encoder clobber), run, then restore
`self.seq_len` to the *true* length so decode starts at the right offset.

#### On-device per layer (`compile_prefill`)

All matmuls IF4 with `gpr_M_reg=gpr_seq_len`:

| Step | Op | Type |
|---|---|---|
| carry input (layers>0) | `LAYER0_OUTPUT → LAYER0_INPUT` chunked copy | [DMA] |
| pre-norm | `rms_norm_core_dram` | [HW] |
| Q proj / K,V proj (non-shared) | `matmat_mul_core` | [HW] |
| V-norm + scatter to KV cache | PBI loop: load → `rms_norm_core`(no γ) → scatter at `k_size` stride | [HW]+[DMA] |
| Q/K norm | `rms_norm_core_dram` | [HW] |
| K-RoPE, Q-RoPE | `rope_hf_core_dram` / `rope_hf_core_dram_gqa` **(PBI, `gpr_M_reg`)** + `_emit_strided_copy_pbi` to pack/unpack rope vs non-rotary dims ([:7521](gemma4_e2b_test.py:7521)) | [HW]+[DMA] |
| GQA duplicate K,V | `_emit_gqa_duplicate_pbi`: 1 KV head → 8 contiguous flash rows ([:7533](gemma4_e2b_test.py:7533)) | [DMA] |
| **attention** | `unified_attention_core(batch=aligned_seq_len, gpr_batch_reg=gpr_q_seq_len, q_scale=1.0)` ([:7548](gemma4_e2b_test.py:7548)) | **[HW]** |
| O proj | `matmat_mul_core` | [HW] |
| post-attn norm + residual | `rms_norm_core_dram` + chunked eltwise add | [HW]+[DMA] |
| pre-MLP norm | `rms_norm_core_dram` | [HW] |
| MLP gate (GELU-fused) + up | two `matmat_mul_core` | [HW] |
| gate⊙up | chunked `eltwise_mul` | [HW]+[DMA] |
| MLP down + post-MLP norm + residual | `matmat_mul_core`, `rms_norm_core_dram`, eltwise add | [HW]+[DMA] |
| **per-layer injection** | §6.4 | [HW]+[DMA] |

There is **no LM head in prefill** — logits are only needed for the first decode
token, which the decoder produces. Gemma4 uses attention scaling 1.0, so
`unified_attention_core` is called with `q_scale=1.0` (no Q pre-scale) — the
scaled-Q step becomes a no-op multiply ([wrapper:1075](gemma4_e2b_test.py:1075)).

### 5.2 Decode (`compile_decoder` [:7888](gemma4_e2b_test.py:7888), `run_decoder` [:9007](gemma4_e2b_test.py:9007))

**Single dynamic program.** `compile_decoder` captures **one** 35-layer body plus
the LM head, templated at `MAX_CONTEXT_SIZE`. Per layer mirrors prefill with `M=1`
(one new token), differing in three ways:

1. **Register-driven addresses.** KV scatter, K-RoPE output and cache reads compute
   their DRAM address from `gpr_seq_len` at run time (`reg_mul_imm`+`add_imm`→`TMP`,
   memcpy `general_reg_src=TMP`).
2. **Runtime KV gather + one attention call.** K/V are gathered from the
   `k_size`-strided cache into fixed contiguous flash buffers with
   `_emit_strided_copy_pbi` (trip count `gpr_aligned_seq_len` over the full
   template; rows past `decode_pos` are zero and bias-masked
   [:8047](gemma4_e2b_test.py:8047)). Attention is a single
   `unified_attention_core(batch=group_size=8, gpr_aligned_seq_len_reg=…,
   q_scale=1.0)` per layer ([:8059](gemma4_e2b_test.py:8059)) — the 8 query heads of
   a group are the batch; full vs sliding layers differ only in which bias address
   they read.
3. **Self-advance.** After the LM head, `add_inc(gpr_seq_len)` bumps the counter,
   then `HALT` ([:8164](gemma4_e2b_test.py:8164)).

**LM head on-device** (last layer only, [:8145](gemma4_e2b_test.py:8145)):
`rms_norm` → `matmat_mul_core` N=262144 (IF4, optional `C=PENALTY_BIAS_DRAM`) →
`LOGITS_DRAM`. A **hardware argmax** returns the top index; logits never leave the
card.

#### Per-token host loop (`run_decoder`)

Prime `gpr_seq_len` **once** ([:9059](gemma4_e2b_test.py:9059)), then per token:

1. **[HW]** set `gpr_bucket_idx = aligned_seq_len/64`.
2. **[host]** embed the previous token; **[DMA]** → `LAYER0_INPUT_DRAM`.
3. **[host]** per-layer inputs `[1,35,256]`; **[DMA]** → `PER_LAYER_INPUTS_DRAM`.
4. **[host]** build the two 1×aligned bias rows (full sees `[0,pos]`, sliding the
   last `window`); **[DMA]** upload both.
5. *(optional)* refresh the on-FPGA repetition-penalty bias (`_write_penalty_bias`
   [:1895](gemma4_e2b_test.py:1895), default OFF via `GEMMA4_PENALTY`) — added as the
   LM-head matmul's `C` term so the HW argmax returns the penalized token with **no
   logit readback**.
6. `program_execute(decoder_addr)`; **[HW]** `get_arg_max_index()` → next token.
7. Stop on token `1` or `end_of_turn` (106) ([:9148](gemma4_e2b_test.py:9148)).

Per token the host does ~5 tiny DMAs + one execute; the 35-layer math and the
argmax are entirely on-device. A live TTY status bar reports tok/s.

### 5.3 Vision encoder (`_run_vision_encoder_fpga`, [:1278](gemma4_e2b_test.py:1278)) — the performance-critical stage

This is the heaviest compute in the whole pipeline and the main optimization
target, so it gets its own detailed treatment. It runs *before* prefill in VLM
mode; its soft-tokens merge into the embedding stream (§5.1 step 2).

#### 5.3.0 Fixed shape (why everything is a compile-time constant)

Every image is resized to `VISION_CANONICAL_SIZE = 896×896`
([:58](gemma4_e2b_test.py:58)) so the Gemma4 processor always emits
`num_patches = VISION_FIXED_NUM_PATCHES = 2520`; a runtime assert enforces it
([:1333](gemma4_e2b_test.py:1333)). Fixed shape is what lets the entire 16-layer
encoder be captured **once** as a single one-shot ISA blob inside `programs.bin`.
Dimensions used throughout ([:2207](gemma4_e2b_test.py:2207)):

| Symbol | Value | Meaning |
|---|---|---|
| `S` | 2520 | patches (sequence length) |
| `aligned_S` | 2560 | `S` rounded up to a multiple of 64 (attention length) |
| `H` (`VIS_H`) | 768 | hidden size |
| `NH` / `HD` | 12 / 64 | heads / head_dim |
| `MLP` | 3072 | FFN intermediate |
| `VIS_LAYERS` | 16 | encoder layers |
| pool_k | 3 | avg-pool kernel → `2520/9 = 280` soft tokens |

Attention is **full bidirectional** (every patch attends to every patch; only
padding is masked) — so per head the score matrix is `aligned_S² = 2560² ≈ 6.55 M`
entries. **Approximate compute per encoder run** (bf16 activations, IF4 weights):
projections+MLP ≈ **0.77 TFLOP** (MLP dominates), attention `QKᵀ`+`PV` ≈
**0.32 TFLOP**. That is ~an order of magnitude more than a whole LM prefill — hence
the focus.

#### 5.3.1 Host reference vs hardware (`gemma4_e2b_numeric.py`)

There are **two host references** and the **one FPGA path**, compared by SNR in
`gemma4_e2b_numeric.py::build_references` ([gemma4_e2b_numeric.py:204](gemma4_e2b_numeric.py:204)):

* **`hf`** — the full-precision HF Gemma4 vision tower (ground truth).
* **`hostref`** — the *same* HF tower with every `nn.Linear` weight round-tripped
  through **IF4** block-quant (`quantize_vision_tower_`
  [gemma4_e2b_numeric.py:96](gemma4_e2b_numeric.py:96)). This *mimics the hardware
  numerics*: the quant gap is present on both sides, so `hostref`↔FPGA SNR isolates
  **hardware execution error** from the (benign, identical) quant gap.
* **FPGA** — `_run_vision_encoder_fpga` stashes three checkpoints in `self._vis_ckpt`
  ([:1516](gemma4_e2b_test.py:1516)): `A` = patch_embed `[S,768]`, `B` = encoder_out
  (pre-pool) `[S,768]`, `C` = image_features `[N_soft,1536]`.

Because the FPGA permutes Q/K/V channels **within each head** before RoPE
(`VIS_ROPE_PERM` = `[0:16, 32:48, 16:32, 48:64]`, [:577](gemma4_e2b_test.py:577)),
the reference has to reproduce that permutation and the head-major layout to
compare layer-0 internals (`_fpga_interleaved` / `_head_major`
[gemma4_e2b_numeric.py:147](gemma4_e2b_numeric.py:147)). `report()` masks padding
rows so the ~2264 zero rows don't dominate the norms. Set
`GEMMA4_VIS_L0_CKPT=1` to dump ~18 layer-0 intermediate buffers for a stage-by-stage
diff ([:1466](gemma4_e2b_test.py:1466)); `GEMMA4_VISION_STATS=1` prints
finite/NaN/rms of `encoder_out` and `image_features`.

#### 5.3.2 What runs where

| Sub-stage | Where | Notes |
|---|---|---|
| resize + patchify + tokenize | **host** (HF processor) | fixed to 2520 patches |
| pixel scale `2·(x−0.5)` | **host** | `vision_patch_embed` [:3848](gemma4_e2b_test.py:3848) |
| patch input-proj (IF4) | **[HW]** matmul | `[S,768]@[768,768]` |
| position-embedding gather + zero-pad | **host** | table lookup, small |
| patch_embed add | **[HW]** chunked eltwise | → `VIS_IO_A` |
| 16 encoder layers | **[HW]** one-shot | see 5.3.3 |
| 2D-RoPE table build | **host** once | reused for any image (depends only on the canonical grid) |
| pooler (avg-pool by position, mask, ×√H) | **host** | `<800` rows, indexed by position |
| embed RMSNorm + `embed_vision` proj (IF4) | **[HW]** | `vision_embed_project` [:3877](gemma4_e2b_test.py:3877) |

So the host does **only** preprocessing, the RoPE-table construction, and the
position-indexed pooler/gather (all cheap); every dense matmul, norm, RoPE apply,
transpose and the attention itself are on the FPGA.

#### 5.3.3 One encoder layer, op by op

`run_vision_layer` ([:3628](gemma4_e2b_test.py:3628)) calls four emitters in
sequence. `S`-major buffers ping-pong `VIS_IO_A`↔`VIS_IO_B` (16 layers, even → ends
in `A`).

**A. `compile_vision_layer` ([:3166](gemma4_e2b_test.py:3166)) — norm + QKV proj**
1. pre-norm (`_rms_norm_dram_pbi`, `M=S`).
2. For each of Q, K, V: **clip_in** (`_emit_clamp_dram_to_dram` — a *passthrough
   matmul* against a 64×64 identity, the only way to reach the LALU clamp,
   [:3047](gemma4_e2b_test.py:3047)) writes `VIS_NORM_OUT`→`VIS_INPUT_CLIP_H_SCRATCH`,
   then the IF4 projection matmul with a **fused output clamp**
   (`_matmul_with_output_clamp`).
3. Q-norm and K-norm (`M = S·NH = 30 240` rows of `HD=64`).

**B. `host_vision_v_norm_rope_gather` ([:3293](gemma4_e2b_test.py:3293)) — V-norm + RoPE + transpose**
1. V-norm (RMS with a constant-ones γ).
2. **2D RoPE** on Q and K (`_emit_vision_rope_2d` [:2964](gemma4_e2b_test.py:2964)) —
   see 5.3.4; `VIS_HOST_ROPE=1` swaps in a correct host computation
   (`_host_vision_rope_inplace` [:3273](gemma4_e2b_test.py:3273)).
3. Per-head transpose of Q/K/V from interleaved `(S·NH, HD)` to head-major
   `(NH, aligned_S, HD)` (`_emit_qkv_transpose_to_hm` [:2864](gemma4_e2b_test.py:2864)),
   into `VIS_FLASH_{Q,K,V}_HM`.

No Q pre-scale: attention passes `q_scale=1.0` (Gemma vision folds the query scaling
into q-norm).

**C. `run_vision_attention_all_heads` ([:3409](gemma4_e2b_test.py:3409)) — attention**
Production capture sets `_vis_s7 = True` ([:8403](gemma4_e2b_test.py:8403)), so for
each of the 12 heads:
1. **bounce** its `aligned_S·HD` slice from the HM buffer into the fixed
   `VIS_FLASH_{Q,K,V}` (a DRAM→SRAM→DRAM copy, `_bounce` [:3437](gemma4_e2b_test.py:3437)),
2. `_vis_emit_attn` → `unified_attention_core(batch=aligned_S,
   aligned_seq_len=aligned_S, q_scale=1.0)` — Vᵀ, `QKᵀ+bias`, softmax, `P·Vᵀ` all in
   `VIS_FLASH_SCRATCH` ([:3383](gemma4_e2b_test.py:3383)),
3. **bounce** the output back to `VIS_FLASH_OUT_HM`.
Then one inverse transpose HM→interleaved `VIS_Q_DRAM`
(`_emit_attn_out_transpose_to_interleaved` [:2903](gemma4_e2b_test.py:2903)).

**D. `compile_vision_layer_post_attn` ([:3485](gemma4_e2b_test.py:3485)) — O-proj + MLP**
O-proj (clip_in + IF4 + fused out-clamp) → post-attn norm → residual add (chunked)
→ pre-FFN norm → gate (clip_in + IF4+GELU + a **separate** out-clamp, since GELU and
CLAMP both need the LALU) → up (clip_in + IF4 + fused clamp) → `gate⊙up` (chunked)
→ down (clip_in + IF4 + fused clamp) → post-FFN norm → residual add → output buffer.

That is **8 clamp passes, 7 IF4 matmuls, 5 RMS-norms, 2 RoPE loops, 6
transposes/bounces, 12 attention calls — per layer, ×16.**

#### 5.3.4 The 2D-RoPE kernel (the hot loop)

`_emit_vision_rope_2d` ([:2964](gemma4_e2b_test.py:2964)) is a doubly-nested ISA
loop: **outer over `S=2520` patches, inner over `NH=12` heads** → **30 240
iterations**. Each iteration issues ~5 small strided-DMA loads + **6 eltwise
mul/add ops on 32 elements each** + 2 stores. It loads each patch's cos/−sin/sin_hi
row *once* and reuses it across the patch's 12 heads (RoPE opt A), and the rows are
32-wide not 64 (opt B) — but the body still processes only **32 bf16 (64 bytes) per
eltwise**, far below the 64-lane vector / URAM-bank width. Two RoPE calls (Q, K) per
layer × 16 layers ≈ **~1 M loop iterations / ~10 M tiny ISA ops** per encoder run.
This is almost certainly the dominant **latency** sink (as opposed to FLOP sink) —
see finding V1.

---

*(Audio is covered next; the vision-specific inefficiency findings are in §9A.)*

### 5.4 Audio encode (`_run_audio_encoder_fpga`, [:1573](gemma4_e2b_test.py:1573))

Conformer encoder; the most hybrid host/FPGA path.

* **[host]** load audio (resample 16 kHz, mono), HF feature extractor → mel
  `input_features`; assert `T_raw` matches the baked value.
* **[DMA/HW]** subsample: `audio_subsample_fpga` ([:4838](gemma4_e2b_test.py:4838))
  uploads mel and emits an **im2col + strided-conv** subsample (two 3×3 stride-2
  stages, `_emit_aud_sub_im2col_s0/s1` [:5083](gemma4_e2b_test.py:5083)) → projection
  chain, shrinking `T_raw`→`T_sub`.
* **[HW]** the encoder runs as **one on-device one-shot** over all Conformer
  layers. Each layer (`run_audio_layer` [:6683](gemma4_e2b_test.py:6683)): **FFN1
  (macaron ½-step)** → **self-attention** → **depthwise conv** → **FFN2 (½-step)**
  → **norm-out**. Building blocks come from `audio_primitives.py`:
  * GLU `a·σ(b)` / SiLU `x·σ(x)` via an identity-matmul-with-sigmoid trick
    ([audio_primitives.py:42](audio_primitives.py:42));
  * **depthwise 1D conv as a per-channel Toeplitz matmul** — host builds the
    `(D,L,L)` causal Toeplitz once (`build_toeplitz_for_depthwise`
    [audio_primitives.py:208](audio_primitives.py:208)); the FPGA runs `D`
    independent `(1,L)@(L,L)` matmuls (`depthwise_conv1d_core_dram`
    [audio_primitives.py:251](audio_primitives.py:251));
  * `half_step_residual` = `res + 0.5·ff`.
* **Relative-position attention** (Transformer-XL style) is emitted **on FPGA** as
  a chain: build K-context / V-context transposes, rel-K projection, AC & BD score
  matrices, a rel-shift, **tanh soft-cap + mask + softmax**, then value·scatter
  (`_emit_aud_attn_fpga_chain` [:6442](gemma4_e2b_test.py:6442) and the
  `_emit_aud_attn_*` helpers around [:6076–6510](gemma4_e2b_test.py:6076)). A
  **host** fallback (`_aud_chunked_attn_host` [:5710](gemma4_e2b_test.py:5710),
  `_aud_rel_shift_host` [:5809](gemma4_e2b_test.py:5809)) exists for
  verification/bring-up.
* **[HW]** `audio_embed_project_fpga` ([:5343](gemma4_e2b_test.py:5343)); **[DMA]**
  read `AUD_FEATURES_FINAL`, truncate/pad to the audio-slot count, restore LM state.

---

## 6. Architecture details worth calling out

### 6.1 GQA via physical duplication
One KV head per group; instead of a grouped kernel, prefill/decode **replicate** K
and V 8× (`_emit_gqa_duplicate_pbi`) into dense flash buffers so a standard
`batch × head_dim` attention core can run. Simpler kernel, more DRAM traffic.

### 6.2 Two attention regimes, two biases
Sliding (`head_dim=256`, window 512) and full (`head_dim=512`, global RoPE) layers
coexist. The host uploads **both** a full and a sliding bias every prefill/decode
step; each layer's compile picks the address. KV cache is uniformly
`head_dim=512`-sized; sliding layers gather with a stride-mismatch copy.

### 6.3 KV sharing
`_kv_shared_map` lets later layers reference an earlier layer's KV slot; only
`_num_kv_slots` unique slots are allocated (saves ~40 MB at 1024 context,
`tensor_init` [:7042](gemma4_e2b_test.py:7042)). Shared layers skip K/V proj and
their K/V weights are zero placeholders in `params.bin`.

### 6.4 Per-layer input injection (Gemma-4-specific)
After each layer's MLP residual, `_compile_per_layer_injection`
([:7173](gemma4_e2b_test.py:7173)) computes `gate = gelu(W_gate·h)` (1536→256),
`gated = gate ⊙ per_layer_input[layer]`, `proj = W_proj·gated` (256→1536), then a
**fused** `rmsnorm(proj) + h` scaled by a per-layer scalar — all in URAM in one
chunked pass (the fusion removes a DRAM write+read round trip). `per_layer_input`
is precomputed on host and streamed in per step.

### 6.5 The `unified_attention_core` decomposition
The wrapper ([:1069](gemma4_e2b_test.py:1069)) turns SDPA into four core calls
([:1075](gemma4_e2b_test.py:1075)): `bf16_transpose_core` (Vᵀ) →
`eltwise_core_dram` MUL_BROADCAST (scaled-Q by `q_scale`) → `matmat_mul_core` with
`softmax_enable` and `bias_mode="full_matrix"` (scores = scaled-Q·Kᵀ + bias,
softmax fused) → `matmat_mul_core` (P·Vᵀ). All dims/addresses can be GPRs, which is
what makes prefill and decode share one captured body across all lengths.

---

## 7. Hardware-callsite vs host-reference: the split at a glance

| Concern | On FPGA (HW core) | On host (torch) |
|---|---|---|
| Token embedding | — | mmap lookup |
| Per-layer inputs | — | `_compute_per_layer_inputs` |
| Attention biases / masks | consumed | built each step |
| RoPE tables | consumed | generated once (`_load_rope_host`); vision 2D per run |
| RMSNorm / matmul / RoPE apply | ✅ | numeric oracle only (`_host_rms_norm`, numeric.py) |
| Attention (SDPA) | ✅ `unified_attention_core` | audio has a host fallback path |
| Depthwise conv (audio) | ✅ Toeplitz matmul | Toeplitz built on host |
| Vision clamps | ✅ fused in matmul | clip ranges extracted at build |
| LM head + argmax | ✅ (greedy, no readback) | — |
| Repetition penalty | ✅ as matmul bias `C` (opt-in) | bias vector built on host |
| Weight quantization | — | `_parallel_quantize` at build |

Rule of thumb: **all dense math is on-device**; the host does I/O,
mask/table construction, small per-token control, and audio's optional
relative-position fallback.

---

## 8. Reading order in the source

1. `gemma4_e2b_config.json` — dims, regions, register bindings.
2. `__init__` → `weight_init` → `tensor_init`
   ([:967](gemma4_e2b_test.py:967), [:6920](gemma4_e2b_test.py:6920),
   [:7042](gemma4_e2b_test.py:7042)).
3. `weight_bin_generate` + `compile_instruction_bin` for the build stage
   ([:268](gemma4_e2b_test.py:268), [:8185](gemma4_e2b_test.py:8185)).
4. `compile_prefill` + `run_prefill` for one full layer's op/DMA sequence.
5. `compile_decoder` + `run_decoder` for the dynamic-PBI pattern.
6. `unified_attention_core` wrapper ([:1069](gemma4_e2b_test.py:1069)) and its
   library body ([user_dma_core.py:6998](../../user_dma_core.py:6998)).
7. `_run_vision_encoder_fpga` / `_run_audio_encoder_fpga` + `audio_primitives.py`.
8. `main` ([:9170](gemma4_e2b_test.py:9170)) for the end-to-end driver.

---

## 9A. Vision-encoder inefficiencies (the optimization focus)

Ordered by expected payoff. V1–V3 are about **latency / memory traffic** (op count,
not FLOPs); V4–V6 are **wasted FLOPs**; V7 is **correctness/accuracy**.

**V1 — The 2D-RoPE kernel is a 30 k-iteration scalar loop.**
`_emit_vision_rope_2d` ([:2964](gemma4_e2b_test.py:2964)) loops `S=2520 × NH=12 =
30 240` times per call, each doing 6 eltwise ops on **32 elements (64 bytes)** plus
~7 tiny strided DMAs — two calls (Q,K) per layer × 16 ≈ **~1 M iterations / ~10 M
ISA ops** that each touch a fraction of the 64-lane datapath. This is very likely
the single largest wall-clock cost in the encoder. **Ideas:** apply RoPE as a
batched vector op over the whole `(S·NH, 64)` tensor (one wide `eltwise_mul` /
`eltwise_add` per coefficient instead of per-row), or use a dedicated RoPE core over
DRAM with a `gpr_M_reg` row loop the way LM RoPE does (`rope_hf_core_dram`) rather
than hand-rolled per-row SRAM math. Even just processing the full 64-wide row per
op (instead of two 32-wide halves) roughly halves op count.

**V2 — Per-head attention "bounce" copies (`_vis_s7` path, on by default in the bin).**
The production capture sets `_vis_s7=True` ([:8403](gemma4_e2b_test.py:8403)), so each
head's Q/K/V is copied HM→`VIS_FLASH_{Q,K,V}` and the output copied back — **4
copies × 12 heads × 16 layers ≈ 768 full-head DRAM↔DRAM round trips** (each
`aligned_S·HD ≈ 164 k` bf16 ≈ 320 KB) purely to give `unified_attention_core` fixed
base addresses. The *other* branch in the same function
([:3468](gemma4_e2b_test.py:3468)) already points the core straight at the HM offset
with **zero bounces**. `unified_attention_core` also accepts `gpr_q/k/v/out_addr`
GPR address overrides ([user_dma_core.py:7012](../../user_dma_core.py:7012)), so a
per-head base computed into a register would keep the "fixed ISA" property *without*
the copies. Removing the bounces saves on the order of **hundreds of MB of DRAM
traffic per image**.

**V3 — Attention is 12 serial per-head calls.** Each of the 12 heads is a separate
`unified_attention_core` with its own Vᵀ transpose, score, softmax, PV and its own
GPR prime/alloc ([:3446](gemma4_e2b_test.py:3446)). The Vᵀ transpose and the SRAM
tiling setup repeat 12× per layer. FLOPs are inherent, but the per-call fixed
overhead and the repeated transpose are not; a head-batched formulation (or at least
hoisting the transpose/bias setup out of the head loop) would cut the per-call
overhead 12×.

**V4 — Clamp-as-passthrough-matmul, ×8 per layer.** Gemma4 vision `ClippableLinear`
clips inputs and outputs; the only HW route to the LALU clamp is a **full matmul
against a 64×64 identity** (`_emit_clamp_dram_to_dram` [:3047](gemma4_e2b_test.py:3047)).
Each layer emits ~8 of these (clip_in for q/k/v/o/gate/up/down + gate's separate
out-clamp), i.e. 8 passthrough GEMMs over `S×H` (or `S×MLP`) that compute *nothing*.
**Ideas:** clamp bounds are static per layer — fold the input clamp into the
*previous* op's fused output clamp wherever the producer is an FPGA op (pre-norm,
residual add), so `clip_in` disappears; or add an eltwise-clamp path so a clamp is a
vector op, not a GEMM.

**V5 — Q/K/V re-clamp the *same* `VIS_NORM_OUT` three times.** `clip_in_q`,
`clip_in_k`, `clip_in_v` ([:3199–3241](gemma4_e2b_test.py:3199)) each clamp the
identical pre-norm tensor into the same scratch. When the three input clip ranges
coincide (common), two of these passthrough GEMMs are pure waste — clamp once and
reuse.

**V6 — `aligned_S` padding inflates the quadratic term.** Attention runs at
`aligned_S=2560` vs `S=2520`; the score matmul and softmax cost scales with
`aligned_S²`, so ~3% of attention compute (and the KV/bias padding rows) is spent on
masked-out positions. Minor next to V1–V5, but free to reclaim if the 64-alignment
constraint can be relaxed for the last tile.

**V7 — The FPGA split-64 2D-RoPE is numerically wrong (~2.6× magnitude).** Code
comments flag `gemma4_e2b_vision_rope_bug`: the default FPGA RoPE inflates magnitude
~2.6× vs the correct host computation, with `VIS_HOST_ROPE=1` as the escape hatch
([:3330](gemma4_e2b_test.py:3330)). This is the prime suspect for the `hostref`↔FPGA
SNR gap in `numeric.py`; **fix correctness before optimizing V1**, since the right
kernel may have a different (cheaper) structure.

**V8 — The VLM path reloads the full HF model every run.** `_run_vision_encoder_fpga`
calls `_ensure_hf_model` + `vision_weight_init(hf_model)`
([:1296](gemma4_e2b_test.py:1296), [:1386](gemma4_e2b_test.py:1386)) to re-upload
vision weights and rebuild RoPE tables on **each** VLM invocation, even though the
weights already live in `params.bin`'s vision section and the ISA is baked. Loading
+ uploading a multi-GB model dominates cold latency; the `_vision_weight_init_from_combined_bin`
path ([:2214](gemma4_e2b_test.py:2214)) exists precisely to avoid it.

**V9 — Non-one-shot (bring-up / numeric) path is one FPGA dispatch per op.** When
`_oneshot_mode` is off, `_compile_and_run_single` ([:2754](gemma4_e2b_test.py:2754))
compiles→DMAs→executes→`wait_queue(120)` for **every** sub-op — ~30 ops × 16 layers ≈
**~480 round trips per image**. Fine for the production one-shot bin, but the
`gemma4_e2b_numeric.py` verification loop pays full PCIe+queue latency per op; batch
verified layers into one capture when profiling.

**V10 — Full-tensor re-DMA "kick" before the encoder one-shot.** Before
`start_execute_from_dram`, the whole vision ISA section is re-DMA'd to DRAM
([:1441](gemma4_e2b_test.py:1441)) to work around a stale-prefetch dispatch quirk,
re-transferring MBs already resident. A one-cacheline touch would serve the same
"kick" purpose.

---

## 9. Inefficiencies & findings (LM & general)

Ordered roughly by impact; several are acknowledged in code comments.

1. **GQA is materialized (8× K/V duplication).** Prefill and decode physically copy
   each KV head into 8 contiguous rows before attention (`_emit_gqa_duplicate_pbi`).
   At long context this is meaningful DRAM traffic and multiplies the flash-buffer
   footprint versus a group-aware kernel that reads one KV head and broadcasts
   internally.

2. **Sliding-layer KV is re-gathered every step over the full template.** Decode
   emits an `_emit_strided_copy_pbi` of length `MAX_CONTEXT_SIZE` per layer to
   re-pack `k_size`-strided cache into `head_dim=256` contiguous buffers
   ([:8047](gemma4_e2b_test.py:8047)), because sliding layers store 256 but the cache
   stride is 512. Storing sliding-layer KV at its native 256 stride (a second slot
   size) would remove the gather.

3. **Both biases uploaded every step.** `run_decoder` builds and DMAs *both* full and
   sliding bias rows each token even when `seq_len ≤ window` makes them identical
   (the code aliases the tensors but still DMAs twice). Guarding the sliding upload
   on `seq_len > sliding_window` removes a per-token PCIe transfer.

4. **Per-token host DMAs dominate short-context decode.** Each step re-sends
   embedding + per-layer-inputs + 2 biases before a tiny 1-row program. At small
   context the PCIe round trips likely outweigh 35 layers of `M=1` compute. Only the
   new token's per-layer-input row changes; caching the rest and skipping unchanged
   biases would cut host overhead.

5. **Prefill pads to `prefill_max_seq_len` (512) regardless of prompt length.** The
   static template runs at the full 512 even for a 20-token prompt
   ([run_prefill_bucketed:8919](gemma4_e2b_test.py:8919)), masking pad rows via bias
   and zeroing their KV. Correct but pays full prefill cost for short prompts. The
   matmuls/norms/RoPE are already PBI (`gpr_seq_len`), so the padding is mostly for
   any residual static loops — pushing those to PBI would make prefill
   length-proportional.

6. **`software_reset()` + full KV/flash/identity re-zero on every prefill.**
   `run_prefill_bucketed` re-zeros the entire KV cache and flash buffers and
   re-uploads identity each call to undo possible encoder clobber. In the pure-LM
   path nothing clobbered them, so this is a large redundant DMA on the hot path;
   gate it on "an encoder ran this session."

7. **Vision Q/K/V each re-clip the *same* pre-norm input separately.**
   `compile_vision_layer` ([:3166](gemma4_e2b_test.py:3166)) emits `clip_in_q`,
   `clip_in_k`, `clip_in_v` as three DRAM→DRAM clamp passes over identical
   `VIS_NORM_OUT` even when their input clip ranges coincide. Deduplicating equal
   ranges removes two full-tensor passes per layer.

8. **Vision attention is per-head.** `run_vision_attention_all_heads`
   ([:3409](gemma4_e2b_test.py:3409)) loops 12 heads, each a separate
   transpose+attention+inverse-transpose with its own head-major buffers. A batched
   `unified_attention_core` over all heads (as decode already does over group_size)
   would amortize the transposes.

9. **Audio keeps a full host attention fallback.** The rel-shift and AC/BD score
   construction exist in both host and FPGA form ([:5636–5809](gemma4_e2b_test.py:5636)).
   If the host path is ever on the live route it serializes the encoder on CPU tensor
   ops + PCIe; keeping it strictly a verification tool avoids that.

10. **Encoder "kick" re-DMA.** Both encoders re-DMA their already-resident program
    section right before `start_execute_from_dram` because a fresh process's
    instruction prefetch is otherwise stale. Documented reliability hack, but it
    re-transfers megabytes already in DRAM.

11. **Legacy `decoder_attention_core` still compiled in.** The class still defines
    the pre-migration per-head attention core ([:1937](gemma4_e2b_test.py:1937))
    which builds `torch.eye(head_dim)` on host per call. It appears unused on the
    live `unified_attention_core` path; dead code that inflates the file and invites
    accidental use.

12. **Doc/name drift.** `README.md` still lists `weights_gemma4_e2b_hf.bin` and
    `quant_schemas.py` while config + code use `params.bin` / `programs.bin` and
    `quant_lib.py` ([config:51](gemma4_e2b_config.json:51),
    [README.md:91](README.md:91)). Harmless but a foot-gun when assembling a deploy
    footprint by hand.

---

## 10. `serve_openai.py` and `gemma4_e2b_numeric.py` (brief)

* **`serve_openai.py`** loads the bins once (weights DMA to card DRAM at startup)
  and serves `POST /v1/chat/completions` (SSE streaming + non-streaming),
  `GET /v1/models`, `GET /health`. Single card, one request at a time (concurrent
  → 503). Decode is greedy by construction, so `temperature > 0` is rejected unless
  `--force-greedy`. It reuses `Gemma4_UnifiedEngine`'s prefill/decode from
  `gemma4_e2b_test`.
* **`gemma4_e2b_numeric.py`** verifies the **vision encoder** only: it SNR-compares
  FPGA readbacks (patch_embed, encoder_out, image_features) against two host
  references — an HF vision tower with **IF4-quantized** Linear weights (mimics
  hardware, isolates HW execution error from the quant gap) and full-precision HF
  (ground truth) — using `calculate_snr` from `user_dma_core`.

---

## 11. Main differences vs `gemma4_e2b_run_from_bin.py`

`run_from_bin.py` is the **execute-only deploy mirror** of `test.py`. It shares the
`Gemma4_UnifiedEngine` layer/prefill/decode structure almost line-for-line (its own
comments say each block "mirrors gemma4_e2b_test.py"), but differs as follows:

1. **⚠️ It is currently deprecated / non-runnable.** Its `__main__` raises
   `SystemExit` ([gemma4_e2b_run_from_bin.py:7814](gemma4_e2b_run_from_bin.py:7814))
   because its compile paths still call `flash_attention_core` /
   `decoder_group_attention_core`, which were **removed** from `user_dma_core.py`
   during the `unified_attention_core` migration. `test.py` uses the fused
   `unified_attention_core` and runs.

2. **No build machinery.** `run_from_bin.py` lacks `weight_bin_generate`,
   `_build_vision/audio/host_section_bytes`, `tokenizer_subset_extract`, and the HF
   import (`_ensure_hf_model`). It **refuses to compile** and errors if
   `programs.bin`/`params.bin` are missing. `test.py` builds them on first run.

3. **Different attention emitters.** `test.py` has `unified_attention_core` +
   `_unified_attention_core_dynamic_q_scale`; `run_from_bin.py` has
   `_flash_attention_core_cached` + `_emit_flash_subroutine` (the removed-core
   wrappers) and a shared sliding-attention subroutine (Trick 5) for decode. Net: the
   *same* prefill/decode dataflow, a *different* (now-stale) attention kernel.

4. **Thinner audio/vision path.** `test.py` carries the fuller on-FPGA audio
   attention chain (`_emit_aud_attn_fpga_chain`, `_emit_aud_softcap_tanh_dram`,
   `_emit_aud_attn_value_and_scatter`, `audio_embed_project_fpga`) and
   `compile_audio_encoder_bin`/`run_audio_encoder_oneshot`; `run_from_bin.py` keeps
   more host-side audio validators (`_aud_validate_matrix_*`) instead.

5. **Execute-only overrides.** `run_from_bin.py` re-defines small helpers as thin
   overrides (`get_arg_max_index`, `program_execute`, `allocate_params_dram`,
   `clear_inst_id`, `write_captured_instructions_to_file`, `isa_add_set_core`
   without the leading underscore) to keep the deploy surface minimal.

Practical takeaway: **read and run `gemma4_e2b_test.py`.** Treat `run_from_bin.py`
as a soon-to-be-refreshed deploy shell; if you need execute-only serving today, use
`serve_openai.py`, which builds on the live `test.py` engine.
