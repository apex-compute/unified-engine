# Hardware Supported-Operations Manifest

This document is the ground truth for what the accelerator can and cannot do today. It pairs with `HARDWARE_BEHAVIORS.md` (which describes *why* the hardware is shaped the way it is) and with `user_dma_core.py` (which is the authoritative API — this manifest is descriptive, the Python file is normative). When the two disagree, `user_dma_core.py` wins and this manifest must be updated.

The manifest is organized into five layers of "can we do this":

- **PRIMITIVE** — the hardware has a direct instruction and `user_dma_core.py` exposes it. Usable today with zero new code beyond the `_core_dram` chunking wrappers in `model_lib_core.py`.
- **FUSED** — a primitive can optionally apply this as a post-op flag (bias, activation, softmax, etc.). No extra instructions; listed separately because the fused variants are cheaper than the composed equivalents.
- **COMPOSABLE** — not a primitive but expressible by chaining primitives. A known recipe exists.
- **COMPOSABLE-BUT-EXPENSIVE** — chaining works but the hardware advantage (fused kernels, low-latency paths) is lost. Flag for ASIC consideration.
- **INEXPRESSIBLE** — cannot be built from the current primitive set. Requires a new hardware feature.

Used by: architecture gap scans (see `MODEL_GAP_SCAN_*.md`) and new-model bring-up planning.

---

## 1. Hardware platform — one-paragraph recap

64 ALUs in lockstep (operations must present multiples of 64 elements along the consumed axis). Two 512 KB SRAM banks: URAM A (`0x00000`–`0x7FFFF`, activations) and URAM B (`0x80000`–`0xFFFFF`, weights). SRAM rows are 128 B (64 bf16 elements); no sub-row addressing. 4 GB DRAM, repartitionable across params / activations / instructions. Compile-then-execute model: no imperative dispatch, no data-dependent control flow unless the branches are pre-compiled. BF16 is the native compute dtype; INT4, INT8, FP4 E2M1 are supported for weight storage and matmul B-operands. See `HARDWARE_BEHAVIORS.md` for the full constraint set.

---

## 2. Primitive inventory

Every entry below is a direct `UnifiedEngine.*` method that triggers a hardware instruction. Names and signatures track `user_dma_core.py` as of the date at the top of `MODEL_GAP_SCAN_2026-04-13.md`. For `HELPER`-only methods (allocation, bookkeeping, instruction capture), see the full enumeration in that file's appendix or read `user_dma_core.py` directly — they are not part of the compute primitive set and not relevant to gap analysis.

### 2.1 Memory movement

| Primitive | Signature summary | Purpose |
|---|---|---|
| `accelerator_memory_to_sram` | `(dram_addr, sram_addr, element_size, stride_bytes_per_chunk?, stride_jump_bytes?)` | DRAM → SRAM (URAM-A or URAM-B depending on `sram_addr` range). Supports strided reads. |
| `sram_to_accelerator_memory` | `(sram_addr, dram_addr, element_size, stride_bytes_per_chunk?, stride_jump_bytes?)` | SRAM → DRAM. Strided writes. `stride_bytes_per_chunk` must be a multiple of 32 B (AXI width). |
| `accelerator_memory_to_bias_sram` | `(dram_addr, element_size)` | DRAM → bias BRAM (8192 elements, 16 KB). |
| `accelerator_memory_to_scale_sram` | `(dram_addr, element_size)` | DRAM → scale BRAM (8192 elements, 16 KB). |
| `dma_to_accelerator_memory` | `(dma_address, bf16 tensor)` | Host → DRAM. |
| `dma_from_accelerator_memory` | `(dma_address, shape)` | DRAM → host. |

**Shape constraints across the movement layer:** all SRAM addresses are 128-B aligned (§3 of `HARDWARE_BEHAVIORS.md`). `memcpy_length_bytes` is typically a multiple of 128. Strides must be multiples of 32 B.

### 2.2 Elementwise

| Primitive | Purpose | Operand placement |
|---|---|---|
| `eltwise_add_core(A, B, C_wb, element_size)` | C = A + B | A and B must be in **different URAM banks**. C can alias A. |
| `eltwise_mul_core(A, B, C_wb, element_size)` | C = A * B | Same placement rule. |

**Note:** no elementwise divide, no elementwise max/min, no elementwise reciprocal at the public primitive layer (reciprocal exists inside LALU but only as part of fused paths like softmax).

### 2.3 Broadcast

| Primitive | Purpose |
|---|---|
| `broadcast_mul(scalar, src_addr, wb_addr, element_size)` | In-place/out-of-place scalar * vector. Scalar encoded as bf16. |
| `broadcast_add(scalar, src_addr, wb_addr, element_size)` | Scalar + vector. |

**Broadcast modes available (see `BROADCAST_MODE` enum):** scalar-in-register, LALU result, negated LALU result, negated FMAX value. The last two are used internally by softmax and by the FMAX-subtract path; they are not typically invoked directly.

### 2.4 Matrix-vector and matrix-matrix

| Primitive | Shape | Fused flags |
|---|---|---|
| `matmat_mul_core(M, K, N, A, B, OUT, ...)` | A(M,K) × B(K,N) → OUT(M,N), both bf16 | `softmax_enable`, `bias_mode ∈ {"broadcast_N", "full_matrix"}`, `gelu_enable`, `silu_enable`, `sigmoid_enable`, `relu_enable` (mutually exclusive activations) |
| `matmat_mul_two_cores(ue0, ue1, ...)` | Row-sharded along M across two engines | Same fused flags |
| `quantized_matmat_core(M, K, N, A_bf16, B_quant, OUT, SCALE, ...)` | A bf16 × B quantized(INT4/INT8/FP4) → OUT bf16 | `bias_mode`, `gelu`, `silu`, `sigmoid`, `relu`. **No `softmax_enable`.** |
| `start_queue_for_dot_product_operation` | Matvec variant streaming B from DRAM, quantized | LALU activation + bias |
| `start_queue_for_bf16_matvec_operation` | bf16 matvec from SRAM | LALU activation + bias |

**Constraints:**
- `K % 64 == 0` and `N_chunk % 64 == 0` always.
- For `matmat_mul_core`: `M_chunk * K + M_chunk * N_chunk ≤ 1 M elements` (URAM full capacity).
- `softmax_enable` requires `M_chunk ≤ 64` (FMAX context size).
- Exactly one activation fused flag may be true.
- For quantized B: `SCALE_DRAM_ADDR` must be provided for INT4/INT8/FP4.

### 2.5 Attention

Single primitive: `flash_attention_core(head_dim, seq_len, Q, K, V, OUTPUT, SCRATCH, BIAS=None, debug_mode=False, SM_OUTPUT=None)`.

- Computes: softmax((Q K^T) / √d + BIAS) V, with the three-phase flash schedule (I @ V^T, Q @ K^T + softmax, SM @ V).
- **Constraints:** `head_dim % 64 == 0`, `seq_len % 64 == 0`. SCRATCH must accommodate (head_dim × seq_len) for V^T plus (seq_len × seq_len) for partial softmax.
- **Mask:** only via `BIAS_DRAM_ADDR`. No native causal, no native sliding-window, no native banded. User must pre-bake the full (seq_len × seq_len) bias tensor if any mask is desired.
- **Activation:** softmax only. No linear attention, no softmax-off path, no alternative nonlinearity.
- **Head structure:** the primitive operates on a single Q/K/V triple. For multi-head attention, the model calls `flash_attention_core` once per head (or uses `flash_attention_batched_core_dram` from `model_lib_core.py`). GQA/MQA is handled at the host-side loop, not inside the primitive.

### 2.6 Normalization

| Primitive | Signature | Notes |
|---|---|---|
| `rms_norm_core(x_sram, out_sram, N, gamma_sram?)` | In-place or out-of-place RMSNorm | Gamma optional. No beta (by definition). No epsilon knob exposed. |
| `rms_norm_core_dram(M, N, X_DRAM, OUT_DRAM, GAMMA_DRAM)` | DRAM-side loop over M rows | Thin wrapper over `rms_norm_core`. |
| `layer_norm_core(x_sram, out_sram, N, zeros_sram?, gamma_sram?, beta_sram?)` | Layer normalization | Requires a zeros vector staged in URAM-B; gamma and beta both optional. |
| `layer_norm_core_dram(M, N, X_DRAM, OUT_DRAM, GAMMA_DRAM?, BETA_DRAM?)` | DRAM-side loop | Same as above. |
| `start_queue_for_bf16_rms_mean` | Low-level RMS computation that leaves 1/RMS in LALU | Used internally; models rarely call this directly. |
| `start_queue_for_bf16_layer_norm_mean` | Low-level layer-norm mean subtract | Same. |

**Notes:** no QK-Norm primitive (it's RMSNorm applied to Q and K — composable, see §3.3). No GroupNorm, no InstanceNorm, no BatchNorm primitive (parakeet has a host-level `batch_norm_core_dram` helper but it's built from existing primitives, not a new instruction).

### 2.7 Softmax (standalone)

`start_queue_for_bf16_softmax_operation(fmax_context_addr, vector_sram_start_addr, output_sram_wb_addr, N)` — computes exp(x - max(x)) / sum(exp(·)) on a single N-vector in URAM-A. `N` constrained by the FMAX context size (64).

### 2.8 RoPE

`rope_core(N, x_sram, cos_sram, sin_sram, out_sram, a_sram, bc_sram)` and DRAM wrapper `rope_core_dram`.

- Computes: `out = x * cos + rot(x) * sin_prenegated`, where `rot(x) = cat(x[N/2:], x[:N/2])` and `sin` has the first half pre-negated.
- **Constraints:** `N % 64 == 0` AND `N` must be even (both conditions hold automatically if N is a multiple of 64 and > 0).
- **Shape convention:** this is HF-style RoPE (half-half rotation). Does NOT natively support the "pair-interleaved" RoPE used by some non-HF conventions — if a model uses interleaved RoPE, the weight layout can be permuted at load time.
- **Decoupled RoPE (as in MLA):** composable by calling `rope_core` on the positional subset of Q/K separately and concatenating, see §3.4.
- **2D RoPE (as in SAM):** composable by calling `rope_core` on each axis; sam has a custom `rope_2d_*` helper that does exactly this.

### 2.9 Permute / transpose

| Primitive | Shape transform |
|---|---|
| `bf16_transpose_core(M, N, IN, OUT)` | (M, N) → (N, M). Uses identity-matrix matvec trick. |
| `bf16_permute_core(d0, d1, d2, IN, OUT)` | (d0, d1, d2) → (d1, d0, d2). `d2 % 64 == 0`. |

**Not supported natively:** arbitrary 4D+ permutes, permutes that change the innermost dim. For those, models write their own SRAM-aware permute helpers (e.g. `bf16_smart_permute_core` in parakeet/smolvlm2, `chunked_transpose_core_dram` in parakeet, `multihead_reshape_dram` in sam). These stay model-local; see §3.11.

### 2.10 Quantize / dequantize

| Primitive | Formats | Direction |
|---|---|---|
| `start_queue_for_bf16_dequantize_operation` | INT4, INT8, FP4 E2M1 | Device-side dequant (vector in DRAM → bf16 in SRAM) |
| `start_queue_for_quantize_operation` | FP4 only | Device-side quant |
| `quantize_weight(weight, N, K, data_type)` | INT4, INT8, FP4 | Host-side quant (prepares DRAM layout for quantized matmul). Runs absmax quantization. |

**Supported dtypes:** `TYPE.INT4`, `TYPE.INT8`, `TYPE.FP4`, `TYPE.BF16` (see `TYPE` enum). **FP8 is not supported.**

### 2.11 Activations

| Activation | Availability |
|---|---|
| Sigmoid | LALU `ACT_NO_X` mode (via `lalu_a/b`), also `sigmoid_enable` fused flag on matmul |
| SiLU / Swish | LALU `ACT` mode (`silu_enable` on matmul) |
| GELU | LALU `ACT` mode (`gelu_enable` on matmul) |
| ReLU | LALU `RELU` mode (`relu_enable` on matmul) |
| Softmax | `softmax_enable` on matmul, or standalone `bf16_softmax_operation` |
| Reciprocal (1/x) | LALU `MODE_RECIP` — used internally by softmax and by post-norm scaling |
| Reciprocal sqrt (1/√x) | LALU `MODE_RSQRT` — used internally by RMS norm |
| Tanh | **Not a primitive.** Parakeet has a software composition (`tanh_core_dram`). |
| GELU-tanh / GELU-exact | Only the simplified sigmoid-approx GELU is available via LALU (b = -1.702). Exact GELU via erf is not available. |
| GLU / ReGLU / SwiGLU / GeGLU | Not primitives. Composed as `gate_matmul(silu_enable or sigmoid_enable) * up_matmul`. Models build this explicitly. |

### 2.12 Image patching

`patching_core(IN, OUT, matrix_dram_addrs, scale_dram_addrs, C, H, W, patch_h, patch_w, K, N, data_type)` — a ViT-style patch-embed primitive that combines patch extraction and quantized projection in one instruction. Constraint: `patches_per_group * patch_w == 64`. Only INT4/INT8/FP4 for the projection weight.

### 2.13 Argmax / top-k

| Mechanism | Details |
|---|---|
| `FMAX context` | 64 parallel FMAX contexts tracked during dot-product / matmul operations with `max_clear_en=1`. |
| `UE_ARGMAX1_INDEX` through `UE_ARGMAX4_INDEX` registers | **Top-4 argmax is read out natively** — four separate index registers capture the top-4 indices from the most recent max-tracking operation. |
| Top-k for k > 4 | **No native support.** Must be composed by: (a) running an argmax pass, (b) masking the winner in the scores tensor, (c) re-running. Cost is k sequential full-vector passes. |

This is the current top-4 ceiling. See §4.1 in the gap list for the ASIC implication.

### 2.14 Instruction / control flow (for the compile-then-execute layer)

| Primitive | Purpose |
|---|---|
| `generate_instruction_halt` | Emit HALT into the instruction stream |
| `generate_instruction_jump` | Emit absolute / conditional / relative JUMP. Supports `JMP`, `JNZ`, `JZ`. |
| `generate_instruction_add_*` (inc/dec/reg/imm/set) | ISA integer arithmetic on the engine's register file (R0–R3: zero, loop counter, DRAM-address, result) |
| `generate_instruction_flag_set/clear/check` | Multi-engine synchronization (set/clear this engine's flag; spin-wait on another engine's flag; 0–7 engine range) |
| `overwrite_instruction_with_general_register` | Replace the immediate DRAM address in the last captured instruction with a general-register reference, enabling parameterized loops |

**Implication:** compile-time loops and conditional branches exist at the ISA level, but the branch condition must be a *register* value. There is no way to branch on a *tensor value* (e.g. "if token_id == EOS then stop") without rolling the decision out to the host and restarting the instruction stream. This matters for models with data-dependent routing (see §4.2).

---

## 3. Composable patterns (known recipes)

These are operations that appear in published model architectures but are NOT primitives — they're built by composing the primitives above. Each pattern here should be considered "supported" for the purposes of gap analysis; none require hardware changes. Where a pattern is expensive, it is marked **COMPOSABLE-BUT-EXPENSIVE** with a note on why.

### 3.1 SwiGLU / GeGLU / ReGLU

Recipe: two parallel matmuls (`gate_proj`, `up_proj`) with different fused activations, then an elementwise multiply.
```
gate = matmat_mul_core(x, W_gate, silu_enable=True)    # SwiGLU
up   = matmat_mul_core(x, W_up)
hidden = eltwise_mul_core_dram(gate, up)
out = matmat_mul_core(hidden, W_down)
```
For GeGLU substitute `gelu_enable=True`; for ReGLU substitute `relu_enable=True`. **COMPOSABLE.** Every released LLM uses this.

### 3.2 RMSNorm with residual (pre-norm transformer block)

Pattern: `x = x + sublayer(rms_norm(x))`. The `rms_norm_core_dram_post_add` helper (now in `model_lib_core.py`) fuses the add with the norm, producing both the residual output and the normalized output in one pass. **COMPOSABLE, fused-helper available.**

### 3.3 QK-Norm

Recipe: apply `rms_norm_core` (or `layer_norm_core`) to Q and K tensors after the Q/K projections and before `flash_attention_core`. For per-head QK-norm, loop over heads.
```
Q = matmat_mul_core(x, W_Q)           # shape (seq_len, n_heads * head_dim)
K = matmat_mul_core(x, W_K)
# stage Q, K back to SRAM per-head, apply rms_norm_core, restage to DRAM
Q_normed = rms_norm_core_dram(Q, ...)
K_normed = rms_norm_core_dram(K, ...)
out = flash_attention_core(Q_normed, K_normed, V, ...)
```
Cost: 2 additional DRAM↔SRAM round trips per attention block. **COMPOSABLE.**

**Zero-centered RMSNorm QK-norm** (Qwen3-Coder-Next variant): "zero-centered" means the norm subtracts the mean before computing RMS. That's equivalent to `layer_norm_core` (which already subtracts the mean). Use `layer_norm_core_dram` instead of `rms_norm_core_dram`. **COMPOSABLE.**

### 3.4 Decoupled RoPE (for MLA)

MLA splits K into a positional half `K_pe` (gets RoPE) and a content half `K_nope` (no RoPE), then concatenates. Recipe:
```
K_pe = matmat_mul_core(latent_k, W_K_pe)           # (seq_len, rope_head_dim)
K_nope = matmat_mul_core(latent_k, W_K_nope)       # (seq_len, content_head_dim)
K_pe_rot = rope_core_dram(K_pe, cos, sin)          # apply RoPE only to the positional slice
K = concat_along_head_dim(K_pe_rot, K_nope)        # stage two DRAM buffers into one
# same for Q
```
The concat is a DRAM layout choice — allocate the final K tensor contiguously and write `K_pe_rot` and `K_nope` into non-overlapping slices. No primitive needed. **COMPOSABLE.**

Constraint: each of `rope_head_dim` and `content_head_dim` must independently satisfy `% 64 == 0` (because each goes through a separate `rope_core` / matmul call that requires the innermost dim be a multiple of 64). If the model's MLA dims violate this, pad per §7 of `HARDWARE_BEHAVIORS.md`.

### 3.5 Sliding-window attention

Recipe: pre-bake a banded bias tensor where positions outside the window are set to −∞ (bf16 `0xFF80`) and positions inside are 0. Pass to `flash_attention_core` as `BIAS_DRAM_ADDR`. Shape is (seq_len, seq_len), so memory scales quadratically — for seq_len=4096 that's 32 MB per batch, allocated from the params or activations region.

**COMPOSABLE.**

**Note on local:global ratios** (e.g. Trinity Large's 3:1): this is a layer-level scheduling decision. Three of every four layers call `flash_attention_core` with a windowed bias; the fourth calls it with no bias (or a NoPE-compatible full bias). No hardware involvement beyond picking which bias to pass.

### 3.6 Causal masking

Same as sliding-window: pre-bake a lower-triangular bias tensor (−∞ above the diagonal, 0 on/below). **COMPOSABLE.**

### 3.7 Gated attention (output gate)

Recipe: after `flash_attention_core`, compute a sigmoid-gated projection from the input and elementwise-multiply.
```
attn_out = flash_attention_core(Q, K, V, ...)
gate = matmat_mul_core(x, W_gate, sigmoid_enable=True)
gated = eltwise_mul_core_dram(attn_out, gate)
out = matmat_mul_core(gated, W_O)
```
**COMPOSABLE.**

### 3.8 Depth-scaled RMSNorm

Recipe: `rms_norm_core_dram` then `broadcast_mul_core_dram` with `scalar = 1/√L`. **COMPOSABLE.**

### 3.9 Parallel transformer blocks (Tiny Aya, older PaLM style)

Recipe: compute attention and MLP from the same normalized input independently, add both to the residual in one step. This is a scheduling change — run both branches, then two `eltwise_add_core_dram` calls or one three-input sum (add A to residual, add B to result). **COMPOSABLE.**

### 3.10 Multi-token prediction (MTP-3)

Recipe: at the final hidden state, run three additional prediction heads (each is a matmul to vocab followed by loss / sampling). No kernel change. **COMPOSABLE.**

For Step 3.5 Flash's inference-time MTP: the three extra heads are called every forward pass and the sampled tokens are used for speculative decoding. The speculative-decode verification loop is host-side.

### 3.11 Multi-head attention reshape / permute

Recipe: each model has its own SRAM-aware reshape from (seq_len, n_heads * head_dim) to per-head (n_heads, seq_len, head_dim) layouts. These are already implemented per-model (swin: `multihead_pad_and_permute`; sam: `multihead_reshape_dram`, `multihead_merge_dram`; parakeet: `bf16_smart_permute_core`). **COMPOSABLE, model-local.**

### 3.12 Embedding lookup

Recipe: no gather primitive exists. Models compute embedding by either (a) materializing a one-hot input and matmuling against the embedding weight (expensive), or (b) host-side indexing followed by a DMA of the selected row into DRAM. Approach (b) is what existing models use. **COMPOSABLE.**

### 3.13 Shared expert + routed experts (MoE)

Recipe: run the shared expert unconditionally (one matmul), compute router logits, apply argmax (top-4 native, top-8+ composed), run each selected expert, weight-sum the outputs. **COMPOSABLE if top-k ≤ 4**, **COMPOSABLE-BUT-EXPENSIVE if top-k ∈ {5..8}**.

### 3.14 Linear attention chunkwise form (Gated DeltaNet, Lightning Attention)

Linear attention — including the delta-rule variant in Gated DeltaNet and the lightning variant in Ling 2.5 — has a chunkwise parallel form that uses only matmuls and elementwise ops. No softmax, no state scan required *across chunks* if the chunk size is chosen well; within a chunk, the state update is a sequence of outer products that can be expressed as matmul(k^T, v) accumulated.

Recipe sketch for Gated DeltaNet (simplified):
```
# Per chunk of tokens:
# State S (size head_dim x head_dim) is maintained across chunks.
# Within a chunk: compute gated updates in parallel via triangular matmul.
S_chunk_contrib = matmat_mul_core(V_chunk.T, K_chunk) * alpha_mask
out_chunk = matmat_mul_core(Q_chunk, S_prev + S_chunk_contrib)
# Update S for next chunk
S = alpha * S + matmat_mul_core(V_chunk.T, K_chunk)
```
Every operation above is a primitive. No new instruction needed.

**COMPOSABLE-BUT-EXPENSIVE.** The "expensive" part is twofold: (a) linear attention's theoretical advantage (O(L) instead of O(L²)) is only realized on GPUs because of memory-bandwidth characteristics that don't hold on this accelerator; your `flash_attention_core` primitive already achieves O(L²) at full throughput. (b) The chunked scan introduces a sequential dependency between chunks (the state `S` must be passed forward), which the instruction scheduler must serialize. For small-to-moderate seq_len (<8k), you will NOT win vs. softmax attention on this hardware. For very long seq_len (>32k), linear attention becomes competitive even here. **Flag for ASIC:** a native linear-attention-scan kernel would be worthwhile if the linear-attention hybrid pattern becomes the dominant architecture — it currently dominates 3 of 12 models surveyed (Qwen3-Coder-Next, Qwen3.5, Ling 2.5).

### 3.15 MLA (Multi-Head Latent Attention)

MLA caches a compressed latent `c_kv` (shape `seq_len × kv_lora_rank`, where `kv_lora_rank` is typically 512–1024) and decompresses K and V per-layer per-forward-pass via two small matmuls. The Q side also has a low-rank compression (`q_lora_rank`). RoPE is applied to a decoupled positional slice (see §3.4).

Per-layer recipe (roughly):
```
# Compress:
c_q = matmat_mul_core(x, W_DQ)            # (seq_len, q_lora_rank)
c_kv = matmat_mul_core(x, W_DKV)           # (seq_len, kv_lora_rank)   <-- CACHED

# Decompress Q:
Q_nope = matmat_mul_core(c_q, W_UQ_nope)   # (seq_len, n_heads * head_dim_nope)
Q_pe   = matmat_mul_core(c_q, W_UQ_pe)     # (seq_len, n_heads * head_dim_pe)
Q_pe_rot = rope_core_dram(Q_pe, cos, sin)
Q = concat(Q_nope, Q_pe_rot)

# Decompress K:
K_nope = matmat_mul_core(c_kv, W_UK_nope)  # (seq_len, n_heads * head_dim_nope)
K_pe   = matmat_mul_core(x, W_KR)           # (seq_len, head_dim_pe)   shared across heads
K_pe_rot = rope_core_dram(K_pe, cos, sin)
K = concat(K_nope, broadcast K_pe_rot across heads)

# Decompress V:
V = matmat_mul_core(c_kv, W_UV)             # (seq_len, n_heads * head_dim_nope)

# Attention:
out = flash_attention_core(Q, K, V, ...)
```
Every step is a primitive. **COMPOSABLE.**

Constraints: `q_lora_rank % 64 == 0`, `kv_lora_rank % 64 == 0`, `head_dim_nope % 64 == 0`, `head_dim_pe % 64 == 0`. DeepSeek-V3 uses `q_lora_rank = 1536`, `kv_lora_rank = 512`, `head_dim_nope = 128`, `head_dim_pe = 64` — all multiples of 64, clean fit.

Cost note: the decompression matmuls run every forward pass per layer, which adds 3–4 matmuls per block vs. standard GQA. On this hardware, matmul is the cheap operation (the whole point of the 64-ALU array), so the cost is moderate. No ASIC change warranted for MLA itself.

### 3.16 Rotary scaling / YARN / linear / NTK-aware

These are all host-side transformations of the cos/sin tables fed into `rope_core`. The primitive doesn't care how cos/sin were computed. **COMPOSABLE.**

### 3.17 NoPE (no positional encoding) layers

Trivial — skip the `rope_core` call for NoPE layers. Trinity Large's global-attention layers use NoPE. **COMPOSABLE.**

### 3.18 Tied vs. untied embeddings

Weight-loading concern. When tied, reuse the embedding weight matrix for the LM head (same DRAM address). When untied, load two matrices. No kernel change. **COMPOSABLE.**

### 3.19 MoE top-4 routing

Use `matmat_mul_core` to compute router logits, then read `UE_ARGMAX1..4_INDEX` registers. **COMPOSABLE, cheap.**

### 3.20 MoE top-k for k ∈ {5, 6, 7, 8}

No native support. Compose via: (a) argmax read, (b) write bf16 `-inf` into the winning index in the logits tensor, (c) re-run argmax, repeat. Cost: k sequential argmax passes, each requiring a round trip from DRAM. For k=8 over a 256-expert logits vector, this is 8 × (DRAM→SRAM + argmax + SRAM→DRAM write) = on the order of 8 × O(256) bf16 operations plus the DMA overhead. Moderate but not catastrophic.

**COMPOSABLE-BUT-EXPENSIVE.** **Flag for ASIC:** if MoE models with k ≥ 8 become standard (GLM-5 is k=8 over 256 experts; DeepSeek-V3 is k=8 over 256), native top-8 argmax is the single highest-leverage additional primitive. Extending `UE_ARGMAX*_INDEX` registers from 4 to 8 (and the corresponding FMAX context width) is a small hardware change with large bring-up value.

### 3.21 Multimodal early fusion (Kimi K2.5, Qwen3.5)

Vision tokens produced by a vision encoder are concatenated with text token embeddings along the sequence dimension, then passed to the text transformer as a single long sequence. Implementation is just DRAM layout — write vision embeddings into the first M slots, text embeddings into slots M..L. No kernel change. **COMPOSABLE.**

### 3.22 Batch normalization

Parakeet's `batch_norm_core_dram` — built from `eltwise_add_core`, `eltwise_mul_core`, and `broadcast_mul`. Running-stats are baked into the weights at compile time (`batch_norm_fuse_params`). **COMPOSABLE.**

---

## 4. Known gaps — operations that are NOT expressible

These are the items to watch for new-model bring-up and to weigh against ASIC design decisions.

### 4.1 Native top-k argmax for k > 4

**Status:** COMPOSABLE-BUT-EXPENSIVE (see §3.20). Argmax register file tops out at four indices (`UE_ARGMAX1..4_INDEX`).

**Mechanism correction:** the argmax registers are populated as a *side-effect of a compute operation* (FMAX tracking during a dot-product / matmul with `max_clear_en=1`), not as a readable register bank. You cannot "read the next 4" without running another FMAX-tracking pass over the (now-masked) logits vector. Composing top-8 therefore costs:
1. The normal router matmul with `max_clear_en=1` — populates `ARGMAX1..4`, reads 4 winners. (This pass happens regardless.)
2. Scatter-write bf16 `-inf` into the 4 winning positions (indices come from the engine register file, so addresses are known — this IS expressible; 4 scalar SRAM writes).
3. An **additional** compute op that streams the masked logits through the FMAX tracker (e.g. a dummy dot-product against a 1-vector, or whatever the lowest-cost FMAX-populating op on a URAM-A vector is).
4. Read `ARGMAX1..4` for the next 4.

The extra op in step 3 is cheap per invocation (256-element scan is a few dozen cycles), but it runs on every token, every MoE layer — for GLM-5 that is ~78 layers × context_length routing decisions per forward pass. Aggregate overhead is plausibly low-single-digit percent of total latency.

**VERIFY before taking to hardware:** this description is inferred from the `max_clear_en` parameter and register naming. Confirm with the Verilog team that there is no cheaper "rescan vector for max" path already present — if there is, composed cost drops further.

**Blast radius today:** GLM-5 MoE routes top-8 of 256 experts. DeepSeek-V3.x (the upstream of GLM-5 and Kimi K2.5) is also top-8 of 256. Any future model inheriting the DeepSeek MoE topology hits this.

**ASIC recommendation (Tier-1):** widen the argmax register file and FMAX tracker to **top-16**. Top-8 is the minimum-viable ask that matches the current MoE standard; top-16 is the recommended target, on the reasoning that (a) the cost delta from 8 → 16 is small (additional MMIO index registers plus modest FMAX-tracker width), (b) MoE research is trending toward higher top-k, not lower, and (c) top-16 absorbs the occasional off-standard top-k (e.g. top-12) via a mask-the-extras cleanup that is cheaper than composing from top-4. Going beyond 16 (e.g. 32) is not justified by any model in the current survey.

### 4.2 Data-dependent gather / sparse indexing

**Status:** INEXPRESSIBLE.

**What's missing:** the hardware has no primitive for reading from DRAM at an address that is computed from tensor data at runtime. The compile-then-execute model requires every DRAM address to be known at compile time or readable from the engine's integer register file (which holds scalars, not tensor-derived index vectors).

**Blast radius today:** DeepSeek Sparse Attention (DSA, adopted by GLM-5) requires a per-query top-k index selection followed by a gather of the selected K/V rows. The gather indices vary *per query*, meaning the instruction stream would need to embed runtime-selected addresses, which it cannot. The vanilla dense flash-attention path through GLM-5 is expressible (MLA is fine, §3.15); only the sparse-attention optimization is not. You can bring up GLM-5 without DSA (use full-MLA attention) and lose the sparsity speedup, which is a performance penalty, not a correctness one. But if your target is to match DSA's compute profile on long context, that's inexpressible.

**Secondary affected patterns:** any model that uses runtime token pruning, content-addressed memory, or retrieval-augmented attention.

**ASIC recommendation:** a scatter/gather primitive keyed off a URAM-resident index vector. Tier-2 priority (DSA is currently the only architecture in the survey that needs it; but it is the only strictly inexpressible operation).

### 4.3 FP8 native dtype

**Status:** INEXPRESSIBLE natively; substitutable with INT8.

**What's missing:** no FP8 E4M3 or E5M2 in the `TYPE` enum. Quantized matmul supports INT4, INT8, FP4 E2M1.

**Blast radius today:** DSA uses FP8 for the indexer matmul. Any model shipping with an FP8 KV-cache (DeepSeek-V3.2 inference configs) would need to be re-quantized to INT8 at load time. Accuracy loss expected to be small but not zero.

**ASIC recommendation:** FP8 E4M3 support would be a nice-to-have. Tier-3 priority (currently no model in the survey strictly requires it; INT8 is an acceptable substitute).

### 4.4 Exact GELU (erf-based)

**Status:** COMPOSABLE-BUT-EXPENSIVE via a polynomial approximation; no primitive for erf.

**What's missing:** LALU's GELU uses the `1.702 × sigmoid` approximation (b = -1.702 in bf16). For models trained with exact GELU-erf, numerical differences are small but measurable.

**Blast radius today:** no model in the survey uses exact GELU — all of them either use SwiGLU or the sigmoid-GELU approximation.

**ASIC recommendation:** none. Current coverage is adequate.

### 4.5 Native causal / banded attention mask

**Status:** COMPOSABLE (§3.5, §3.6) via pre-baked bias tensors.

**What's missing:** no causal or windowed flag on `flash_attention_core`. The user must pre-compute and DRAM-resident the full bias tensor.

**Blast radius today:** every autoregressive model needs a causal mask; sliding-window models need a banded mask. Cost is the memory footprint: (seq_len × seq_len) bf16 per distinct mask pattern. For seq_len = 262k (Qwen3-Coder-Next native context) that's 137 GB — far exceeds 4 GB DRAM. For long-context models, the mask must be tiled per flash-attention chunk, which is an architectural complication but still expressible.

**ASIC recommendation:** a causal / banded flag on `flash_attention_core` would eliminate the mask storage for those two common cases. Tier-2 priority if long-context becomes a bring-up target.

### 4.6 Arbitrary high-dimensional permute

**Status:** COMPOSABLE per-model (§3.11); no general primitive.

**What's missing:** `bf16_permute_core` only does 3D (d0, d1, d2) → (d1, d0, d2). Permutes involving the innermost dim, 4D permutes, or transpose-within-head patterns require per-model helpers.

**Blast radius today:** every model that assembles multi-head attention tensors writes its own permute helper. Not a correctness gap, but a developer-velocity gap.

**ASIC recommendation:** none strictly needed; a more general permute primitive would save model-bring-up engineering time but doesn't unlock new architectures.

### 4.7 State-space models (SSMs, Mamba)

**Status:** COMPOSABLE-BUT-EXPENSIVE (no model in the survey uses pure SSM, but one variant — Lightning Attention in Ling 2.5 — is SSM-adjacent).

**What's missing:** the selective-scan primitive used by Mamba-family models is a sequential recurrence over sequence length. It has a chunkwise parallel form that's matmul-only, similar to Gated DeltaNet (§3.14).

**Blast radius today:** no pure-SSM model is in the survey. If Mamba-2 or a successor becomes a bring-up target, the pattern is buildable from primitives but loses efficiency vs. GPU.

**ASIC recommendation:** share the same "native linear-attention-scan kernel" fix as §3.14. Tier-3.

---

## 5. Shape-constraint cheat sheet

| Operation | Innermost dim constraint | Other constraints |
|---|---|---|
| Any SRAM access | multiple of 128 B (= 64 bf16 elements) | 128-B aligned address |
| `matmat_mul_core` | K % 64, N_chunk % 64 | M_chunk*K + M_chunk*N_chunk ≤ 1M elements |
| `matmat_mul_core` with `softmax_enable` | K % 64, N_chunk % 64 | M_chunk ≤ 64 |
| `flash_attention_core` | head_dim % 64, seq_len % 64 | SCRATCH ≥ (head_dim + seq_len) × seq_len × 2 B |
| `rope_core` | N % 64 (even enforced automatically) | sin first half pre-negated |
| `bf16_permute_core` | d2 % 64 | permutation limited to (d0,d1,d2)→(d1,d0,d2) |
| Eltwise ops | element_size % 64 recommended | A and B in different URAM banks |
| `patching_core` | patches_per_group × patch_w == 64 | n_patches_w % patches_per_group == 0 |
| `quantize_weight` | N % 64, K % 64 | weight must be 2D bf16 |
| FMAX / argmax | N bounded by FMAX context size (64) | top-k natively reads 4 indices |

See `HARDWARE_BEHAVIORS.md` §3 for the reasoning behind the 64-element / 128-byte constraint.

---

## 6. Data-type support

| Dtype | Read (matmul B) | Write | Convert to/from | Notes |
|---|---|---|---|---|
| BF16 | ✓ | ✓ | native compute | Primary working dtype |
| INT4 | ✓ (quantized_matmat_core) | via `quantize_weight` host-side | dequantize primitive exists | Absmax quantization, 64-block |
| INT8 | ✓ (quantized_matmat_core) | via `quantize_weight` host-side | dequantize primitive exists | Absmax, 64-block |
| FP4 E2M1 | ✓ (quantized_matmat_core) | `quantize_operation` primitive + host | dequantize primitive exists | 16 predefined table values |
| FP8 E4M3/E5M2 | ✗ | ✗ | ✗ | Not supported |
| FP16 | ✗ | ✗ | ✗ | Not supported |
| FP32 | ✗ at device | ✗ at device | host-side only | All host-side intermediates; device never consumes FP32 |
| BF19 | internal | internal | host converters exposed | Internal ALU format only |

---

## 7. Versioning

This manifest describes the state of `user_dma_core.py` at the commit noted in `MODEL_GAP_SCAN_2026-04-13.md`. When `user_dma_core.py` gains a primitive, add a row to §2. When a new composable pattern is discovered during a model bring-up, add it to §3. When a gap is resolved by an ASIC/FPGA change, move the entry from §4 to §2 and update the manifest date.

The manifest should be updated BEFORE a new primitive lands, as part of the review of the `user_dma_core.py` change.
