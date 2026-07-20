# Gemma‑4 E4B on the Unified Engine — Code Tutorial

A deep walk‑through of the Python that builds and runs Gemma‑4 **E4B**
inference on the FPGA accelerator. The primary subject is
[gemma4_e4b_test.py](models/gemma4_e4b/gemma4_e4b_test.py) — the **runnable,
authoritative** implementation: it owns the ISA compiler, uses the migrated
`unified_attention_core`, and its `main()` actually executes. This tutorial
covers the overall architecture, the exact compute / memcpy operations at every
stage (weight load, prefill, encode, decode), which work runs on the
**hardware cores** versus the **host reference path**, and closes with observed
run inefficiencies. The final section lists how the sibling
[gemma4_e4b_run_from_bin.py](models/gemma4_e4b/gemma4_e4b_run_from_bin.py)
differs.

Files (all under `models/gemma4_e4b/`):

| File | Role |
|---|---|
| [gemma4_e4b_test.py](models/gemma4_e4b/gemma4_e4b_test.py) | **Build + run.** `weight_bin_generate`, `compile_instruction_bin`, the `Gemma4_UnifiedEngine` class, and `main()`. Authoritative. |
| [gemma4_e4b_run_from_bin.py](models/gemma4_e4b/gemma4_e4b_run_from_bin.py) | Execute‑only "deploy" twin. Loads the bins and runs — never compiles. Currently deprecated (see §12). |
| [audio_primitives.py](models/gemma4_e4b/audio_primitives.py) | Stateless ISA‑emit helpers for the Conformer audio encoder (GLU, SiLU, depthwise conv, …). |
| [gemma4_e4b_config.json](models/gemma4_e4b/gemma4_e4b_config.json) | Model dims + fixed DRAM region map. |
| [README.md](models/gemma4_e4b/README.md) | User‑facing quickstart. |

---

## 1. The two‑script, compile‑once model

Inference is split into a **build stage** and a **deploy stage** so the
expensive ISA compilation happens exactly once. Both live in `test.py`:
`main()` ([9227](models/gemma4_e4b/gemma4_e4b_test.py#L9227)) *compiles the bin
if it is missing, then runs*; `run_from_bin.py` only runs.

```
gemma4_e4b_test.py  main()
  ├─ (cold start, bin missing)  compile_instruction_bin()  ─▶  gemma4_e4b_bin/
  │       weight_bin_generate()                                   programs.bin   (unified ISA)
  │                                                               programs.json  (manifest: addrs, sizes, flops, layout sig)
  │                                                               params.bin     (~10 GB combined weights)
  │                                                               params.json    (weight section manifest)
  │                                                               tokenizer/     (~32 MB)
  └─ (every run)  load_instruction_bin() → set_prefill_seq* → run_prefill_bucketed() → run_decoder()
```

The instruction bin is a single **unified** program image containing, back to
back: the LM prefill template, the LM decode program, the vision encoder, and
the audio encoder. `compile_instruction_bin`
([8081](models/gemma4_e4b/gemma4_e4b_test.py#L8081)) captures each section in its
own `start_capture`/`stop_capture` session (to stay under the instruction cap)
and concatenates the bytes; every internal `JUMP_ABS` target is baked against
one absolute base address. The loader therefore **must** DMA the bin to the
exact base recorded in the manifest (`load_instruction_bin`,
[8541](models/gemma4_e4b/gemma4_e4b_test.py#L8541)).

By default `main()` builds **LM + vision + audio** (the whole bin is only
~15 MiB since it holds ISA, not data); `GEMMA4_LM_ONLY_BIN=1` builds LM‑only.

### Compile once, run any length — "Dynamic PBI"

One captured prefill template and one decode program serve **all**
prompt/context lengths. Two mechanisms make that possible:

- **GPR‑parameterized instructions.** Sequence length, aligned length, and a
  bucket/aligned‑vector count are written to fixed general‑purpose registers
  before each run (`GPR_SEQ_LEN_REG`, `GPR_Q_SEQ_LEN_REG`, `GPR_BUCKET_IDX_REG`
  (aka `gpr_kv_nvec` at decode), `GPR_ALIGNED_SEQ_LEN_REG`; see `fixed_isa_regs`
  in the config and `_isa_add_set_core` at
  [1896](models/gemma4_e4b/gemma4_e4b_test.py#L1896)). Cores read `M` / trip
  counts / addresses from a register (`gpr_M_reg=...`) rather than a
  compile‑time constant.
- **PBI hardware loops.** Repetitive per‑token / per‑head copies are emitted as
  *hardware* `loop_start(gpr_loop_cnt=…)` / `loop_end` blocks (see the `§4.4`
  PBI helpers `_emit_strided_copy_pbi`, `_emit_vnorm_copy_pbi`,
  `_emit_gqa_duplicate_pbi`, [6992–7067](models/gemma4_e4b/gemma4_e4b_test.py#L6992)),
  so the instruction count is constant regardless of runtime length.

---

## 2. Model architecture (E4B dims)

From [gemma4_e4b_config.json](models/gemma4_e4b/gemma4_e4b_config.json) `file_info` / `model`:

| Property | E4B value | Note |
|---|---|---|
| `hidden_size` | **2560** | (E2B was 1536) |
| `num_layers` | **42** | (E2B 35) |
| `num_attention_heads` | **8** | |
| `num_key_value_heads` | **2** | GQA: `gs_per_kv = 8/2 = 4` query heads share one KV head |
| `head_dim` (full‑attn layers) | 512 | |
| `head_dim_sliding` | 256 | sliding‑window layers use half |
| `mlp_elements` (intermediate) | 10240 | `double_wide_mlp_first_layer=42` ⇒ wide path effectively unused for E4B |
| `embedding_vocab` | 262144 | |
| `max_context_size` | 1024 | |
| `max_prefill_seq_len` / `prefill_max_seq_len` | 320 | prompts capped here |
| `sliding_window` | 512 | |
| `full_attention_layers` / `rope_global_layers` | {5,11,17,23,29,35,41} | 1 global every 6 layers |

**Per‑layer structure (Gemma‑4 decoder block):**

```
x ─▶ RMSNorm(pre) ─▶ Q/K/V proj (IF4) ─▶ Q/K RMSNorm ─▶ RoPE ─▶ attention ─▶ O proj (IF4)
   ─▶ RMSNorm(post_attn) ─▶ + residual
   ─▶ RMSNorm(pre_ffn)  ─▶ [gate(GELU) ⊙ up] (IF4) ─▶ down (IF4) ─▶ RMSNorm(post_ffw) ─▶ + residual
   ─▶ PER‑LAYER INPUT INJECTION  (Gemma‑4‑specific, see §7)
```

Gemma‑4 specifics that recur throughout the code:

- **Dual normalization** — RMSNorm *after* attention and *after* the FFN
  (`POST_ATTENTION_NORM`, `POST_FFW_NORM`), plus **Q‑norm / K‑norm** per head
  before RoPE.
- **Per‑layer input injection** — every layer receives a token‑dependent gated
  projection of a per‑layer embedding table (§7).
- **Partial rotary** on global layers — RoPE rotates only the first
  `head_dim × 0.25 = 128` dims; the rest pass through.
- **Sliding vs full attention** alternation, each with its own bias matrix.
- **Attention scaling = 1.0** (Gemma‑4 uses no `1/√d`). `unified_attention_core`
  is called with `q_scale=1.0`.
- All linear weights are **IF4** (4‑bit, group‑quantized), stored as `_DATA`
  (packed nibbles) + `_SCALE` (per‑group scales).
- **GQA sizing** matters: E4B has `num_kv=2 ≠ num_attn=8`. A comment war in
  `_get_layer_attention_dims` ([6873](models/gemma4_e4b/gemma4_e4b_test.py#L6873))
  records the bug where an old `group_size` formula under‑counted Q heads.

---

## 3. Host vs hardware‑core split

The engine is an **instruction emitter**. Methods named `*_core` / `*_core_dram`
do **not** compute — they append accelerator ISA to a capture buffer that is
later dumped to `programs.bin` and executed on the FPGA. Plain Python/torch math
is the **host reference path** and runs on the CPU.

**Runs on the FPGA hardware cores (ISA emitted):**

| Core call | Used for |
|---|---|
| `matmat_mul_core` | GEMM: prefill Q/K/V/O/gate/up/down, LM head; decode `down`/`lm_head` (large‑K exception) |
| `quantized_matmat_core` | **decode GEMV** (M=1) for Q/K/V/O/gate/up (`§3h`) |
| `rms_norm_core` / `rms_norm_core_dram` | all RMSNorms (SRAM‑resident and DRAM‑to‑DRAM) |
| `rope_hf_core_dram_gqa` (prefill) / `rope_hf_core_decode` (decode) | RoPE rotation, GQA‑grouped |
| **`unified_attention_core`** ([1124](models/gemma4_e4b/gemma4_e4b_test.py#L1124)) | attention: `bf16_transpose_core(Vᵀ)` → Q·scale → `Q@Kᵀ` **with fused softmax + bias** → `P@V` |
| `bf16_transpose_core` | transpose V to `(head_dim, aligned_seq_len)` inside attention |
| `eltwise_add_core` / `eltwise_mul_core` / `eltwise_core_dram` | residual adds, gate⊙up, Q scale |
| `broadcast_mul` | per‑layer‑injection scalar |
| `accelerator_memory_to_sram` / `sram_to_accelerator_memory` | DRAM↔URAM staging (the "memcpy" ops) |
| `accelerator_memcpy` | DRAM↔DRAM (register‑addressed, for dynamic decode positions) |
| `loop_start`/`loop_end`, `generate_instruction_*` | PBI loops, flags, GPR arithmetic (`reg_mul_imm`, `add_imm`, `add_inc`), `halt` |

**Runs on the host (torch/CPU, reference / glue):**

| Host method | What it computes |
|---|---|
| `weight_bin_generate` ([281](models/gemma4_e4b/gemma4_e4b_test.py#L281)) | quantize HF weights → `params.bin` (build only; `_parallel_quantize`) |
| `_build_vision_section_bytes` / `_build_audio_section_bytes` / `_build_host_section_bytes` | pack encoder + host weights (build only) |
| `get_embedding_for_tokens` ([6506](models/gemma4_e4b/gemma4_e4b_test.py#L6506)) | token‑embedding lookup (zero‑copy mmap view of `params.bin`) |
| `_compute_per_layer_inputs` ([7492](models/gemma4_e4b/gemma4_e4b_test.py#L7492)) | per‑layer embed lookup + `embedding @ proj.T` (float32) + RMSNorm + scale — **every token** |
| prefill bias construction (`run_prefill`) | `full`/`sliding` attention bias matrices (same‑head + token‑causal; VLM image masking) |
| `_write_penalty_bias` ([1969](models/gemma4_e4b/gemma4_e4b_test.py#L1969)) | optional repetition‑penalty vector (host‑built, DMA'd; consumed as the LM‑head matmul `C` term) |
| `get_arg_max_index` | reads back the argmax token id after each decode step |
| audio: `audio_subsample_host`, `audio_rel_pos_host`, `_aud_chunked_attn_host*`, `_aud_depthwise_conv1d_host`, `build_toeplitz_for_depthwise` | host reference for parts of the Conformer encoder |

`_SILENT_MODE` / swapping `builtins.print` suppress library chatter during
capture; some setup calls are replayed with `capture_buffer=None` so their DMAs
prime DRAM without emitting ISA.

---

## 4. DRAM memory map & weight load

The FPGA DRAM layout is **fixed and baked into the bin**, so both engines
hard‑code identical bases (`__init__`,
[gemma4_e4b_test.py:994](models/gemma4_e4b/gemma4_e4b_test.py#L994)):

```
0x00000000 – 0xAC000000   Weights: LM        (2752 MB)   ← params_dram_base
0xAC000000 – 0xB4000000   Weights: Vision    (128 MB)
0xB4000000 – 0xC0000000   Weights: Audio     (192 MB)
0xC0000000 – 0xD4000000   Activation scratch (320 MB)    ← tensor_dram_base
0xD4000000 – 0xDA000000   ISA: Audio         (96 MB)
0xDA000000 – 0x100000000  ISA: Unified       (608 MB)    ← program_dram_base
```

`params.json` records `LAYER_WEIGHT_SIZE = 58,354,944` bytes/layer and each
tensor's `offset`/`size` inside the block (config `regions`, baked from
`0x50000000`, after the `vocab×hidden×2 = 0x50000000` token‑embedding table).

### `weight_bin_generate` (build, [281](models/gemma4_e4b/gemma4_e4b_test.py#L281))

Loads the HF model once, **IF4‑quantizes** every linear (`_parallel_quantize`),
lays out `[LM | vision | audio | host]` sections at their config offsets, and
writes `params.bin` + `params.json`. Run once on a beefy host.

### `weight_init` (runtime, [6586](models/gemma4_e4b/gemma4_e4b_test.py#L6586))

Memcpy‑heavy, once at startup:

1. `mmap` `params.bin` read‑only (RSS stays tiny; OS pages in only touched
   bytes). Token embedding is a **zero‑copy** `torch.frombuffer` view.
2. Load `params.json` (sections `lm | vision | audio | host`).
3. For each of 42 layers, for each `blk0_regions` tensor: slice the mmap and
   `dma_write(H2C, …)` to `layers_base + layer*LAYER_WEIGHT_SIZE + offset`.
   Layer‑0 addresses are cached on `self.DRAM_ADDR_LAYER0_*`; later layers add
   `layer_idx * LAYER_WEIGHT_SIZE`.
4. DMA non‑layer tensors (`OUTPUT_NORM`, `PER_LAYER_MODEL_PROJ`, `LM_HEAD`
   scale+data, …).
5. `_load_host_weights_from_combined_bin` mmaps the **host section** (per‑layer
   embed table, per‑layer model proj, proj norm, layer scalars, kv‑shared map).
6. `_load_rope_host` builds/loads the RoPE cos/sin tables.

### Activation scratch — `tensor_init` ([6708](models/gemma4_e4b/gemma4_e4b_test.py#L6708))

Allocates + zero‑pads every intermediate in the `0xC0000000` region:

- **KV cache with slot sharing** — only layers that own KV state get a slot;
  shared layers point at their reference layer's slot (`_kv_slot_for_layer`).
  K/V are stored at **`k_size` stride per token** (`k_size = num_kv × head_dim ×
  2`), uniform across layers.
- **Flash buffers sized for `max_prefill_seq_len`, not `max_context_size`** —
  they run at full width only during prefill; decode uses ≤ a bucket. Sizing for
  1024 would waste ~200 MB the decoder bin needs.
- Two bias buffers (`FLASH_BIAS_FULL`, `FLASH_BIAS_SLIDING`), an `ATTN_P`
  scratch, `LOGITS`, `PENALTY_BIAS`, per‑layer‑injection scratch, and a
  `FLASH_OUT_HEAD` temp used by the decode GQA marshaling.
- `_tensor_layout_signature` ([6856](models/gemma4_e4b/gemma4_e4b_test.py#L6856))
  snapshots the named addresses; `main()` refuses to run a cached bin whose
  baked layout differs (would hang the FPGA).

---

## 5. Prefill stage

`compile_prefill` ([7068](models/gemma4_e4b/gemma4_e4b_test.py#L7068)) emits one
GPR‑parameterized template of length `prefill_max_seq_len` (320);
`run_prefill_bucketed` → `run_prefill` execute it.

### 5.1 Host preamble (`run_prefill`, [7535](models/gemma4_e4b/gemma4_e4b_test.py#L7535))

1. Prefill processes `prefill_seq[:-1]`.
2. **Static template padding** (`run_prefill_bucketed`,
   [8798](models/gemma4_e4b/gemma4_e4b_test.py#L8798)): the prompt is padded with
   its *last real token* up to 320 so the fixed template always sees 320 rows;
   decode starts from the *actual* length so padded KV rows are never attended.
   `software_reset()` + full K/V‑cache zero + flash‑buffer zero + identity
   re‑DMA precede execution.
3. **Host**: `get_embedding_for_tokens` → `[seq_len, 2560]` bf16, scaled.
4. **Multimodal merge** (VLM/audio): overwrite image/audio placeholder rows with
   encoder soft‑token features (`mm_token_type_ids` 1=image, 3=audio).
5. DMA embeddings → `LAYER0_INPUT_DRAM`.
6. **Host**: `_compute_per_layer_inputs` → `[seq_len, 42, 256]`, permuted to
   `[42, seq_len, 256]` and DMA'd to `PER_LAYER_INPUTS_DRAM`.
7. **Host**: build `full_bias` + `sliding_bias` (`aligned_seq_len²` bf16). The
   mask is **same‑head AND token‑causal** — a flat causal `tril` would let KV
   group 1 read group 0's keys (catastrophic for E4B `num_kv=2`). VLM adds
   image‑block masking.
8. Set GPRs `gpr_seq_len`, `gpr_q_seq_len`, `gpr_aligned_seq_len`.
9. `program_execute(prefill_addr)` — blocks until halt (heartbeat prints).

### 5.2 On‑FPGA, per layer (order of emitted ops)

For `layer_idx` in `0..41`, `layer_off = layer_idx * LAYER_WEIGHT_SIZE`,
`num_attn=8`, `num_kv=2`, `gs_per_kv=4`:

| # | Op | Core call | Notes |
|---|---|---|---|
| 1 | copy prev output → input (layers>0) | SRAM copy chunked | `seq_len × 2560` |
| 2 | pre‑RMSNorm | `rms_norm_core_dram` (`gpr_seq_len`) | |
| 3 | Q proj | `matmat_mul_core` IF4 | N=`cur_q_size`=8·head_dim |
| 4/5 | K / V proj | `matmat_mul_core` IF4 | V → `FLASH_V` temp |
| 6 | **V‑norm + scatter to sloted KV** | `_emit_vnorm_copy_pbi` (PBI loop, `inner_count=num_kv`) | per‑row RMS (no γ), dense→sloted stride |
| 7 | Q norm (all 8 heads) | `rms_norm_core_dram` (`gpr_q_seq_len`) | |
| 8 | K norm (all 2 heads) | `rms_norm_core_dram` (`gpr_seq_len`) | |
| 9 | **RoPE** | `rope_hf_core_dram_gqa` (K group=num_kv, Q group=num_attn) | sliding = full rotary (1 call each); global = partial: strided‑copy → rope → strided‑copy back + passthrough, all via `_emit_strided_copy_pbi` |
| 10 | spread dense K → sloted cache | `_emit_strided_copy_pbi` | |
| 11 | **GQA duplicate K & V → flash layout** | `_emit_gqa_duplicate_pbi` ×2 | each KV head → `gs_per_kv` flash slots |
| 12 | **attention** | `unified_attention_core(q_scale=1.0)` | full or sliding bias per layer |
| 13 | O proj | `matmat_mul_core` IF4 | K=`cur_q_size` |
| 14 | post‑attn RMSNorm + residual add | `rms_norm_core_dram` + eltwise add | |
| 15 | pre‑FFN RMSNorm | `rms_norm_core_dram` | |
| 16 | gate (GELU fused) + up | two `matmat_mul_core` IF4 | N=`cur_mlp`=10240 |
| 17 | gate ⊙ up | eltwise mul chunked | |
| 18 | down proj | `matmat_mul_core` IF4 | K=10240 |
| 19 | post‑FFW RMSNorm + residual add | `rms_norm_core_dram` + eltwise add | |
| 20 | **per‑layer input injection** | §7 | |

After the last layer: `generate_instruction_halt()`. (Steps 6/9/10/11 are the
PBI‑loop rewrites of what the deprecated twin unrolls per token — see §12.)

> **`unified_attention_core`** ([1130](models/gemma4_e4b/gemma4_e4b_test.py#L1130))
> is one inline call per layer: `bf16_transpose_core` builds `Vᵀ` in scratch,
> Q is scaled by `q_scale` via `eltwise_core_dram`, then
> `matmat_mul_core(Q@Kᵀ, softmax_enable=True, C_DRAM_ADDR=bias,
> bias_mode="full_matrix")` fuses the bias add + softmax, and a second
> `matmat_mul_core` does `P@V`. Unlike the legacy `flash_attention_core`
> subroutine, it has no back‑to‑back‑call degradation, so all 42 layers run in
> the pipelined PBI path with no separate `√d` pre‑scale pass.

---

## 6. Decode stage

`compile_decoder` ([7745](models/gemma4_e4b/gemma4_e4b_test.py#L7745)) emits one
GEMV (M=1) program with a bucket dispatcher; `run_decoder`
([8994](models/gemma4_e4b/gemma4_e4b_test.py#L8994)) drives the token loop.

### 6.1 Host loop (per generated token)

```
while seq_len < max:
    seq_len += 1
    aligned_seq_len = round_up(seq_len, 64)
    set gpr_kv_nvec = aligned_seq_len // UE_VECTOR_SIZE   # ← bounds the KV gather
    host: embedding lookup for the new token → DMA to LAYER0_INPUT_DRAM
    host: _compute_per_layer_inputs([token]) → DMA to PER_LAYER_INPUTS_DRAM
    host: build 1×aligned bias rows (full + sliding, image-masked for VLM) → DMA
    (optional) host: _write_penalty_bias(generated) → DMA to PENALTY_BIAS_DRAM
    program_execute(decoder_addr)     # one pass: 42 layers + final norm + lm_head
    token_id = get_arg_max_index()    # HW argmax read-back
    print; stop on EOS / token 1
```

The **LM‑head argmax happens on the FPGA**; the host only reads back the winning
index. Repetition penalty (if `GEMMA4_PENALTY=1`) is baked as the LM‑head
matmul's `C` term (`C_DRAM_ADDR=PENALTY_BIAS_DRAM, bias_mode="broadcast_N"`,
[8056](models/gemma4_e4b/gemma4_e4b_test.py#L8056)) — all‑zero ⇒ pure greedy,
bit‑identical.

### 6.2 On‑FPGA, per layer (differences vs prefill)

- **Projections are `quantized_matmat_core` GEMV** (M=1) — except `down_proj`
  and `lm_head`, which stay `matmat_mul_core` because large K (up to 10240)
  would drive `quantized_matmat_core`'s `N_chunk` to 0 and hang ("§3h
  exception", [8017](models/gemma4_e4b/gemma4_e4b_test.py#L8017)).
- **Layer input avoids a copy**: layer 0 reads `LAYER0_INPUT_DRAM`; layers >0
  read `LAYER0_OUTPUT_DRAM` directly.
- **Dynamic KV write position**: V‑norm/scatter and K‑RoPE write to
  `slot_base + gpr_seq_len × k_size + kv_h × head_dim × 2`, computed at runtime
  (`reg_mul_imm` + `add_imm` → `TMP_REG`), then a register‑addressed store.
- **Bucket‑bounded KV gather** ([7767–7933](models/gemma4_e4b/gemma4_e4b_test.py#L7767)):
  for each of `num_kv=2` KV heads, `_emit_strided_copy_pbi` gathers **only
  `gpr_gather_n = gpr_kv_nvec × UE_VECTOR_SIZE = aligned_seq_len` rows** (what
  the attention actually reads) from the `k_size`‑strided slot into contiguous
  `FLASH_K`/`FLASH_V`. The comment flags the old "copy the full 1024 window every
  token for all 42 layers" as the previous *dominant decode cost*.
- **GQA marshaling (§7c)**: each KV head's `gs_per_kv=4` Q rows are copied to the
  fixed `FLASH_Q` base, `unified_attention_core` runs into `FLASH_OUT_HEAD`, and
  the result is drained back to that head's `q_off` slot.

`decoder_attention_core` ([2012](models/gemma4_e4b/gemma4_e4b_test.py#L2012)) is
a legacy single‑KV‑head kernel still present (I@Vᵀ identity‑matmul transpose,
Q@Kᵀ, softmax, P@V) but the **live** decode path is `unified_attention_core`.

---

## 7. Per‑layer input injection (Gemma‑4 specific)

Every layer ends with a token‑conditioned signal
(`_compile_per_layer_injection`, [6905](models/gemma4_e4b/gemma4_e4b_test.py#L6905)):

```
gate      = GELU( layer_out @ per_layer_input_gate )     # Linear 2560→256, matmat_mul_core gelu_enable
gated     = gate ⊙ per_layer_input[layer_idx]            # eltwise mul, host-precomputed signal
projected = per_layer_projection @ gated                 # Linear 256→2560, matmat_mul_core
out       = ( RMSNorm(projected) + layer_out ) × layer_scalar   # fused URAM pass
```

The **`per_layer_input[layer_idx]`** operand is built on the **host** in
`_compute_per_layer_inputs`: a per‑layer embedding‑table lookup plus
`embedding.float() @ per_layer_model_proj.T` (a float32 CPU matmul), RMSNorm,
add, scale — for the whole prompt at prefill and **once per token during
decode**. The RMSNorm+residual+scalar tail is fused into one URAM‑resident pass
to avoid a DRAM round trip.

---

## 8. Vision encode stage (VLM) — deep dive

Driver: `_run_vision_encoder_fpga` ([1322](models/gemma4_e4b/gemma4_e4b_test.py#L1322)).
The vision tower is a **SigLIP‑style ViT** with **bidirectional** (non‑causal)
attention and a **fixed‑shape** contract: every image is resized to
`VISION_CANONICAL_SIZE = (896, 896)` so the HF processor always emits
`num_patches = 2520` (constant, so the encoder ISA can be baked into the bin).

**Dims** (class constants, [2282](models/gemma4_e4b/gemma4_e4b_test.py#L2282)):

| | value |
|---|---|
| `VIS_H` (hidden) | 768 |
| `VIS_HEADS` (NH) | 12 |
| `VIS_HEAD_DIM` (HD) | 64 |
| `VIS_MLP` | 3072 |
| `VIS_LAYERS` | 16 |
| `num_patches` (S) | 2520 → `aligned_S = 2560` |
| projection out (`VIS_TEXT_H`) | LM embed width (read from `embedding_projection.weight`) |

All linears are **IF4** and, uniquely to the vision tower, `Gemma4ClippableLinear`:
each projection has *learned input and output clamp ranges* (`_vis_clip_ranges`)
that must be applied around the matmul.

### 8.1 What runs where — host reference vs FPGA hardware

This is the split you'll be optimizing. Vision has **no in‑process host
reference for the encoder body** — the old `get_image_features` host path was
removed; the 16 layers run only on the FPGA. What remains on the host is
**pre/post‑processing and setup**, not layer math.

| Stage | Host (CPU/torch) | FPGA hardware core |
|---|---|---|
| Preprocess | PIL resize → 896², HF `AutoProcessor` → `pixel_values [1,2520,768]`, `image_position_ids`, `padding_positions` | — |
| 2D‑RoPE tables | `_load_or_build_vision_rope_pads` ([2864](models/gemma4_e4b/gemma4_e4b_test.py#L2864), **build‑time**): calls HF `vt.encoder.rotary_emb`, applies `VIS_ROPE_PERM`, stores per‑patch 32‑wide tables into the bin | tables DMA'd to `VIS_ROPE_*_PAD_TILED` |
| Patch embed | pixel scale `2·(x−0.5)`; position‑embed **gather** `pos_table[0,x]+pos_table[1,y]`, zero padding rows ([3667](models/gemma4_e4b/gemma4_e4b_test.py#L3667)) | `input_proj` matmul (IF4) + chunked eltwise‑add of pos‑embed (`vision_patch_embed`, **runtime compile‑and‑run**) |
| **16 encoder layers** | *(none — pure FPGA, baked one‑shot)* | pre_norm, clip, Q/K/V/O proj, Q/K/V norm, 2D‑RoPE, transposes, per‑head attention, MLP, residuals |
| Pooler / embed proj | `vision_embed_project` ([3681](models/gemma4_e4b/gemma4_e4b_test.py#L3681)): mask padding, **avg‑pool by position**, `×√H`, strip masked rows (indexed ops, ~≤800 rows) | RMSNorm + `embedding_projection` matmul (IF4), **runtime compile‑and‑run** |
| Weight prep | `vision_weight_init` / `_build_vision_section_bytes`: IF4 quantize + `VIS_ROPE_PERM` permute (build‑time) | weights DMA'd to the vision weight region `0xAC000000` |

**Build vs runtime is important for performance work:** only the **16 layers**
are captured into `programs.bin` (in `_oneshot_mode`, see
`compile_instruction_bin` [8298](models/gemma4_e4b/gemma4_e4b_test.py#L8298))
and executed as one `start_execute_from_dram` shot at runtime. `vision_patch_embed`
and `vision_embed_project` are **not baked** — at runtime they emit ISA,
DMA it, and execute per sub‑op (`_compile_and_run_single`), interleaved with
host gathers. So the runtime critical path is:

```
host preprocess ─▶ vision_weight_init + vision_tensor_init + bias + RoPE-DMA (setup)
  ─▶ vision_patch_embed        (host scale/gather + a few FPGA compile-and-run ops)
  ─▶ re-DMA vision section      (~MB "kick" — FPGA dispatch workaround, §8.5)
  ─▶ start_execute(16 layers)   (one FPGA shot, the bulk)
  ─▶ vision_embed_project       (host pooler + 2 FPGA compile-and-run ops)
  ─▶ LM-state restore           (re-zero KV/flash, re-DMA IDENTITY)
```

### 8.2 Per‑layer op flow (baked)

`run_vision_layer` ([3584](models/gemma4_e4b/gemma4_e4b_test.py#L3584)) runs four
phases; in the bin they are emitted inline per layer, ping‑ponging between
`VIS_IO_A`/`VIS_IO_B`:

**Phase A — `compile_vision_layer` ([3187](models/gemma4_e4b/gemma4_e4b_test.py#L3187)):**
`pre_norm` (RMSNorm, PBI) → for each of Q/K/V: **input‑clamp pass** →
projection matmul (IF4) **with fused output clamp** → `q_norm`, `k_norm`
(per‑head RMSNorm, `M=S·NH`).

**Phase B — `host_vision_v_norm_rope_gather` ([3294](models/gemma4_e4b/gemma4_e4b_test.py#L3294))** (all FPGA despite the name):
`v_norm` (RMSNorm, ones‑γ) → **2D RoPE** on Q and K (`_emit_vision_rope_2d`) →
**per‑head transpose** of Q/K/V from interleaved `(S·NH, HD)` to head‑major
`(NH, aligned_S, HD)` buffers `VIS_FLASH_*_HM` (`_emit_qkv_transpose_to_hm`).

**Phase C — `run_vision_attention_all_heads` ([3375](models/gemma4_e4b/gemma4_e4b_test.py#L3375)):**
`for h in range(12)`: **SRAM‑bounce** each head's `Q/K/V` slice from the
head‑major buffer into the *fixed* `VIS_FLASH_Q/K/V` → `unified_attention_core`
(`batch=aligned_seq_len=2560`, `q_scale=1.0`, bidirectional `VIS_FLASH_BIAS`) →
bounce the result out to `VIS_FLASH_OUT_HM`. After all heads, one inverse
transpose back to interleaved `VIS_Q_DRAM` (`_emit_attn_out_transpose_to_interleaved`).

**Phase D — `compile_vision_layer_post_attn` ([3441](models/gemma4_e4b/gemma4_e4b_test.py#L3441)):**
O‑proj (clamp+matmul+clamp) → `post_attn_norm` → residual add → `pre_ffn_norm`
→ gate (clamp‑in, matmul+GELU, **separate clamp‑out** since GELU+CLAMP both use
the LALU) → up (clamp+matmul+clamp) → `gate ⊙ up` → down (clamp+matmul+clamp) →
`post_ffn_norm` → residual add → next `VIS_IO`.

### 8.3 Three mechanisms worth understanding

1. **Clamp = matmul‑by‑identity** (`_emit_clamp_dram_to_dram`,
   [3117](models/gemma4_e4b/gemma4_e4b_test.py#L3117)). The LALU's CLAMP unit
   only routes through the `DOT_PRODUCT` datapath, not eltwise/broadcast. So an
   input clamp is emitted as a `matmat_mul_core(M=N/64, K=64, N=64)` against a
   64×64 bf16 **identity** with `clamp_enable=True` — an identity multiply that
   passes the data through while the fused clamp applies. Every `ClippableLinear`
   therefore costs an **extra full‑tensor identity‑matmul pass** for the input
   clamp (output clamp is fused into the real projection).
2. **2D RoPE via the qwen split‑64 trick** (`_emit_vision_rope_2d`,
   [3024](models/gemma4_e4b/gemma4_e4b_test.py#L3024)). Weights are pre‑permuted
   (`VIS_ROPE_PERM`) so a 1‑D rotation reproduces the 2‑D result. It's a **PBI
   hardware loop** (patch‑outer / head‑inner): each patch's 32‑wide cos/neg_sin/
   sin_hi coefficients load once and are reused across its 12 heads; six
   `eltwise_*` ops per row produce the rotation.
3. **Per‑head attention with fixed‑buffer marshaling.** Attention is computed
   **one head at a time** (`unified_attention_core` expects a single head's
   contiguous Q/K/V at fixed base addresses), so Phase C bounces each head's
   `aligned_S×HD` slice through SRAM into `VIS_FLASH_Q/K/V` and back. Bias is a
   full `aligned_S × aligned_S` bf16 matrix (bidirectional: all‑zero except
   padding/alignment columns → `−inf`).

### 8.4 Vision‑specific inefficiencies (optimization targets)

Ordered by expected payoff:

1. **Per‑head attention serialization + marshaling.** 12 heads × 16 layers =
   **192** separate `unified_attention_core` calls, each wrapped in 3 input
   SRAM‑bounces + 1 output bounce of an `aligned_S×HD = 2560×64` slice
   (`run_vision_attention_all_heads` [3402](models/gemma4_e4b/gemma4_e4b_test.py#L3402)).
   That's ~768 large DRAM→SRAM→DRAM copies whose only purpose is to move each
   head to the *fixed* `VIS_FLASH_Q/K/V` base the core reads. If the core could
   take a head **stride/offset** (or GPR base address, as the LM decode path
   does via `gpr_*_addr`), the head‑major buffers could be read in place and all
   768 bounces plus the 4 transposes/layer disappear. This is the single biggest
   structural cost in the encoder.
2. **Clamp‑by‑identity passes.** Each layer runs ~7 input‑clamp identity‑matmuls
   (q,k,v,o,gate,up,down) plus a gate output‑clamp — full DRAM→DRAM passes over
   `S×H` or `S×MLP` that compute *nothing* but a clamp. 16 layers ⇒ ~128 clamp
   passes. A native clamp on the eltwise/writeback path (or folding the input
   clamp into the preceding norm/residual writeback) removes them. The `q/k/v`
   input clamps additionally **re‑read the identical `VIS_NORM_OUT`** three times
   into the same scratch with only the bounds differing ([3220‑3274](models/gemma4_e4b/gemma4_e4b_test.py#L3220)) — at minimum these three could share one pass.
3. **Per‑head transposes.** 3 forward (`_emit_qkv_transpose_to_hm`) + 1 inverse
   transpose per layer (64 total), each a strided full‑tensor gather, exist only
   to feed the per‑head attention. They vanish if #1 is fixed.
4. **Unrolled chunked residual/eltwise.** `_run_eltwise_add_chunked` /
   `_run_eltwise_mul_chunked` ([2770](models/gemma4_e4b/gemma4_e4b_test.py#L2770))
   split each `S×H` (≈1.94 M‑element) residual and the `S×MLP` gate⊙up into
   65536‑element chunks, emitting one program per chunk (~30 each) — pure
   instruction bloat vs a PBI‑looped eltwise like the LM path's
   `_emit_sram_eltwise_chunked`.
5. **Runtime "kick" re‑DMA of the whole vision section.** Before executing, the
   driver re‑DMAs the multi‑MB vision ISA to work around a stale‑prefetch
   dispatch quirk ([1470‑1487](models/gemma4_e4b/gemma4_e4b_test.py#L1470)) — a
   full section DMA on every VLM run.
6. **Bidirectional bias is a dense `2560²` bf16 matrix** (~13 MB) that is almost
   all zeros; only padding/alignment columns carry `−inf`. A per‑column mask
   vector applied in the softmax would avoid building and re‑uploading it.
7. **Encoder ↔ LM DRAM aliasing.** Vision reuses the LM tensor region, so every
   VLM run re‑zeros the KV cache + flash buffers and re‑DMAs IDENTITY afterward
   ([1540‑1585](models/gemma4_e4b/gemma4_e4b_test.py#L1540)) — and it is a
   recurring NaN‑leak hazard (the comments document the `num_kv` sizing bug).
8. **`patch_embed`/`embed_project` not baked.** They run compile‑and‑run at
   inference with host gathers (position‑embed lookup, pooler) on the critical
   path. Small next to the 16 layers, but fully avoidable by baking + moving the
   gathers on‑FPGA.

> **Note — stale in‑code warnings.** Several source comments still claim the
> FPGA vision output is "wrong until the vision‑QKV clamp bug is fixed"
> ([1387](models/gemma4_e4b/gemma4_e4b_test.py#L1387); `--fpga-encoder` help at
> [9276](models/gemma4_e4b/gemma4_e4b_test.py#L9276)). **These are out of date —
> the hardware vision encoder is verified working correctly.** The clamp path is
> a *performance* target (§8.4 items 2–3), not a correctness blocker. Keep
> `GEMMA4_VISION_STATS=1` (prints finite/NaN/rms of `encoder_out` /
> `image_features`) as a regression check when optimizing.

The soft tokens merge into the prompt embedding in `run_prefill` (§5.1 step 4).
E4B VLM quality is also documented as image‑dependent (repetition risk; run with
`GEMMA4_PENALTY=1`).

---

## 9. Audio encode stage

Driver: `_run_audio_encoder_fpga` ([1600](models/gemma4_e4b/gemma4_e4b_test.py#L1600)),
a **Conformer** encoder. Config via `audio_config_init`
([3791](models/gemma4_e4b/gemma4_e4b_test.py#L3791)): 12 layers, `H=1024`,
8 heads (`head_dim=128`), FFN 4096, conv kernel 5, chunked local attention
(`chunk=12`, left ctx 13), output dim 2560. Audio on hardware is a
debug/unverified path (`--fpga-encoder` / `GEMMA4_FPGA_AUDIO_FEATURES=1`);
otherwise parts run on the host reference.

Pipeline:

1. **Host**: HF processor → log‑mel `input_features [1, T_raw, 128]`.
2. **Subsampling** (`audio_subsample_fpga` [4651](models/gemma4_e4b/gemma4_e4b_test.py#L4651)):
   two stride‑2 convs as **im2col + matmul** (`_emit_aud_sub_im2col_s0/s1`),
   shrinking `T_raw → N1` frames and projecting to `H=1024`
   (`_emit_aud_embed_project_chain`). Host reference: `audio_subsample_host`.
3. **Relative‑position** table (`audio_rel_pos_host`, host).
4. **12 Conformer layers**, each a *macaron* block:
   `½·FFN → self‑attention (rel‑pos, chunked/blocked mask, logit soft‑cap) →
   depthwise conv → ½·FFN → LayerNorm` (`compile_audio_layer_ffn1/attn/conv/
   ffn2/norm_out`, `run_audio_layer` [6476](models/gemma4_e4b/gemma4_e4b_test.py#L6476)).
   Attention uses the rel‑pos "matrix AC / BD + rel‑shift" scheme
   (`_emit_aud_attn_matrix_ac`, `_emit_aud_attn_matrix_bd_unshifted`,
   `_emit_aud_attn_rel_shift`), with host references `_aud_chunked_attn_host*`.
5. `audio_embed_project_fpga`/`_host` → `[N_soft, 2560]`.

### `audio_primitives.py` helpers

Stateless ISA emitters ([audio_primitives.py](models/gemma4_e4b/audio_primitives.py)):

- `silu_core_dram` — `x·sigmoid(x)` via identity‑matmul + eltwise mul.
- `glu_core_dram` — `a·sigmoid(b)` on a split projection.
- `half_step_residual_core_dram` — `residual + 0.5·ff_out` (macaron half‑step).
- `depthwise_conv1d_core_dram` + `build_toeplitz_for_depthwise` — depthwise 1‑D
  conv as a **per‑channel Toeplitz matmul** (host builds the `(D, L, L)` causal
  Toeplitz; FPGA runs `D` independent `(1,L)@(L,L)` matmuls, **sequentially**).
- `eltwise_add_core_dram`, `copy_dram_to_dram_chunked` — chunked DRAM helpers.

Each chunks rows so a sub‑batch fits `URAM_A`/`URAM_B` (banks at `0x00000` /
`0x80000`), sized by `_row_chunk`.

---

## 10. Observed run inefficiencies

Ordered roughly by expected impact. (Several inefficiencies that exist in the
deprecated twin are already *fixed* here — see §12.)

### Decode

1. **Per‑token host recompute of per‑layer inputs.** `_compute_per_layer_inputs`
   runs a float32 `[1,2560]@[2560, 42·256]` CPU matmul + RMSNorm **for every
   decoded token** ([9-…], called at run_decoder), on the critical path between
   FPGA passes. The embedding lookup is cheap; the projection is not — it could
   be pushed onto the FPGA or cached.
2. **Host bias rebuild + DMA every token.** `full`/`sliding` bias rows are
   re‑allocated, filled, and DMA'd each step (the sliding/image branches also
   clone). Small but on the per‑token path.
3. **KV stored strided, then re‑gathered contiguous each layer.** K/V live at
   `k_size` stride but attention wants contiguous `head_dim`, so every layer
   PBI‑gathers into `FLASH_K/V`. Now bucket‑bounded (good), but a cache layout
   matching the core's stride would remove the gather entirely.
4. **Decode Q/K RoPE and GQA marshaling stay Python‑unrolled.** RoPE loops
   `for g in range(num_attn)` and `for kv_h in range(num_kv)` with per‑head
   `rope_hf_core_decode` + SRAM‑bounce copies, and each KV head marshals its Q
   group to/from the fixed `FLASH_Q`/`FLASH_OUT_HEAD` base
   ([7855–7969](models/gemma4_e4b/gemma4_e4b_test.py#L7855)). Counts are small
   compile‑time constants (8, 2) so not length‑dependent, but still extra
   copies every layer.

### Prefill

5. **Static‑template padding wastes compute.** Short prompts are padded to
   `prefill_max_seq_len=320`, so the FPGA always runs full 320‑row
   attention/MLP even for a 20‑token prompt. Bucketed prefill templates (like
   decode) would avoid it.
6. **Decode isn't pipelined across layers.** Each token executes one program
   pass; the M=1 GEMV regime is memory‑bound on weight reads with little reuse.
   (Inherent to autoregressive decode, but batching/speculative decode would
   amortize weight traffic.)

### Encoders / structural

7. **Vision encoder — see §8.4** for the detailed list (per‑head attention
   serialization + marshaling, clamp‑by‑identity passes, per‑head transposes,
   unrolled chunked eltwise, the runtime "kick" re‑DMA, and the dense bias
   matrix). The hardware path is verified correct; these are pure performance
   targets and the main VLM optimization surface.
8. **Encoder ↔ LM DRAM aliasing forces re‑initialization.** Vision/audio reuse
   the LM tensor region, so each VLM/audio run re‑zeros the KV cache + flash
   buffers and re‑DMAs IDENTITY before LM prefill — per‑run overhead and a
   recurring source of "vision floats leaked into LM cache → NaN" bugs the
   comments describe.
9. **Audio depthwise conv is `D` sequential per‑channel matmuls**
   (`depthwise_conv1d_core_dram`): latency scales linearly with channels, no
   batching. Audio attention also offloads substantial work to host references.
10. **Unused width provisioning.** `double_wide_mlp_first_layer=42 = num_layers`,
    so the wide‑MLP path never triggers for E4B, yet MLP buffers are sized for
    `max(mlp_elements, mlp_elements_wide)`. Minor DRAM slack.
11. **Legacy `decoder_attention_core` still carried.** The live path is
    `unified_attention_core`, but the ~280‑line legacy kernel remains in the
    class — dead weight / drift risk.
12. **Two ~9k‑line near‑duplicate files.** `run_from_bin.py` mirrors most of
    `test.py` by hand and is currently non‑runnable (§12) — maintenance cost and
    demonstrated drift (the GQA‑sizing bug history).

---

## 11. Quick reference — end‑to‑end call graph (LM)

```
main()                                                  # gemma4_e4b_test.py:9227
 ├─ Gemma4_UnifiedEngine()        __init__ → load_config → weight_init → tensor_init → _preallocate_identity_matrix
 ├─ (cold start) compile_instruction_bin()              # LM prefill+decode, +vision, +audio → programs.bin
 ├─ load_instruction_bin()                              # DMA bin to baked base; stale-layout guard
 ├─ set_prefill_seq[/_vlm/_audio]()                     # tokenize (+ run encoder for VLM/audio)
 ├─ run_prefill_bucketed() → run_prefill()              # host embed+per-layer+bias → DMA → program_execute(prefill_addr)
 └─ run_decoder()                                       # loop: host embed/per-layer/bias per token → program_execute(decoder_addr) → get_arg_max_index()
```

---

## 12. How `run_from_bin.py` differs

[gemma4_e4b_run_from_bin.py](models/gemma4_e4b/gemma4_e4b_run_from_bin.py) is the
execute‑only deploy twin. It shares the same `Gemma4_UnifiedEngine` shape, DRAM
map, config, weight/tensor init, `run_prefill`, `run_prefill_bucketed`, and
`run_decoder` **host** logic. The differences that matter:

1. **It is currently deprecated / non‑runnable.** Its `__main__`
   ([gemma4_e4b_run_from_bin.py:7548](models/gemma4_e4b/gemma4_e4b_run_from_bin.py#L7548))
   raises `SystemExit`: the class still references `flash_attention_core` /
   `decoder_group_attention_core`, which were **removed** from `user_dma_core.py`
   during the `unified_attention_core` migration. Use `test.py`.

2. **It never compiles.** `compile_instruction_bin`
   ([6783](models/gemma4_e4b/gemma4_e4b_run_from_bin.py#L6783)) raises
   `NotImplementedError`; `load_instruction_bin` only loads. The authoritative
   bin builder is `test.py`.

3. **Its `compile_prefill` is a legacy, non‑PBI, GQA‑incorrect stub.** Marked
   "LEGACY / REFERENCE ONLY" ([5918](models/gemma4_e4b/gemma4_e4b_run_from_bin.py#L5918)),
   it retains the retired dual‑engine / `group_size` structure that
   under‑counts Q heads for E4B, and it emits attention via the old
   `flash_attention_core` subroutine — pinned to **non‑PBI "LEGACY" mode**
   because of the PBI flash back‑to‑back HW bug
   ([6120](models/gemma4_e4b/gemma4_e4b_run_from_bin.py#L6120)).

4. **Per‑token Python‑unrolled loops instead of PBI hardware loops.** Where
   `test.py` uses `rope_hf_core_dram_gqa`, `_emit_strided_copy_pbi`,
   `_emit_vnorm_copy_pbi`, and `_emit_gqa_duplicate_pbi` (constant instruction
   count), the twin emits `seq_len`‑ and `seq_len×group_size`‑length unrolled
   RoPE / V‑norm / GQA‑duplicate / partial‑rotary copies
   ([6042–6106](models/gemma4_e4b/gemma4_e4b_run_from_bin.py#L6042)) — thousands
   of tiny ops per layer, a much larger prefill bin, and a separate `√d`
   pre‑scale pass that `unified_attention_core` makes unnecessary.

5. **Decode gathers the full `MAX_CONTEXT` window every token.** Its
   `compile_decoder` GQA loop copies all 1024 K/V rows per KV head per layer each
   step ([6665](models/gemma4_e4b/gemma4_e4b_run_from_bin.py#L6665)) via
   `decoder_group_attention_core`, versus `test.py`'s bucket‑bounded
   `gpr_gather_n` gather + inline `unified_attention_core`. This full‑window copy
   was the previous **dominant decode cost**.

6. **`main()` just loads + runs.** It requires `programs.bin` to already exist
   (no cold‑start build, no `_tensor_layout_signature` stale‑bin guard) and
   raises if the bin is missing.

In short: `run_from_bin.py` is the older, pre‑`unified_attention_core`,
pre‑PBI‑loop snapshot of the same pipeline, kept as an execute‑only reference.
For anything runnable — and for the efficient prefill/decode described above —
use [gemma4_e4b_test.py](models/gemma4_e4b/gemma4_e4b_test.py).
