# SAM 3.1 Complete Compute Graph

Every arithmetic operation from input image to output masks.
Reshapes/views/permutes omitted unless they involve data movement (like window partition scatter/gather).
All dimensions assume batch=1 inference, no DAC, text-only prompts.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         INPUTS                                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Image:  [1, 3, 1008, 1008]  (bf16, normalized to [-1,1])                 ║
║  Text:   string  e.g. "car"                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
                    │                              │
                    ▼                              ▼
┌──────────────────────────────┐   ┌──────────────────────────────────────┐
│  VISION PATH                 │   │  TEXT PATH                           │
│  (Stages 1-2)                │   │  (Stage 3)                           │
└──────────────────────────────┘   └──────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
 STAGE 0: IMAGE PREPROCESSING (host-side)
═══════════════════════════════════════════════════════════════════════════════

  image [1, 3, 1008, 1008]
    │
    │  Resize to (1008, 1008)
    │  ToDtype float32
    │  Normalize: pixel = (pixel - 0.5) / 0.5
    │
    ▼
  image [1, 3, 1008, 1008] bf16


═══════════════════════════════════════════════════════════════════════════════
 STAGE 1: ViT-Det BACKBONE  (32 blocks, dim=1024, 16 heads, head_dim=64)
═══════════════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 1.0  PATCH EXTRACTION (host-side)                                      │
  │                                                                        │
  │   image [3, 1008, 1008]                                                │
  │     │                                                                  │
  │     │  Extract 72×72 grid of 14×14×3 patches                          │
  │     │  Each patch: flatten 14×14×3 = 588 elements                     │
  │     │  Zero-pad 588 → 640 (align to 64)                               │
  │     │                                                                  │
  │     ▼                                                                  │
  │   patches [5184, 640]  bf16  (written to DRAM via DMA)                │
  └─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 1.1  PATCH EMBEDDING                                                   │
  │                                                                        │
  │   MATMUL: patches @ patch_embed_weight^T                               │
  │     A = [5184, 640]                                                    │
  │     B = [1024, 640]   (Conv2d weight reshaped, padded)                │
  │     → [5184, 1024]                                                     │
  │                                                                        │
  │   (no bias — bias_patch_embed=False)                                   │
  └─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 1.2  ADD ABSOLUTE POSITIONAL EMBEDDING                                 │
  │                                                                        │
  │   pos_embed: pretrained [1, 577, 1024] → strip CLS → [1, 576, 1024]  │
  │              → reshape to [24, 24, 1024] → tile 3×3 → [72, 72, 1024] │
  │              → flatten → [5184, 1024]   (precomputed on host)         │
  │                                                                        │
  │   ELEMENTWISE ADD:  patches_out + pos_embed                            │
  │     [5184, 1024] + [5184, 1024] → [5184, 1024]                       │
  │     (5,308,416 element-wise additions)                                 │
  └─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 1.3  LN_PRE (LayerNorm, eps=1e-5)                                     │
  │                                                                        │
  │   For each of 5184 rows of 1024 elements:                             │
  │     μ = mean(x)                     (1024 adds + 1 div)               │
  │     σ² = mean((x - μ)²)            (1024 subs + 1024 muls + 1024 adds│
  │                                      + 1 div)                          │
  │     x_norm = (x - μ) / √(σ² + ε)  (1024 subs + 1 sqrt + 1024 divs) │
  │     out = x_norm * γ + β           (1024 muls + 1024 adds)           │
  │                                                                        │
  │   Total: 5184 × LayerNorm(1024)                                       │
  │   [5184, 1024] → [5184, 1024]                                        │
  └─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 1.4  ViT TRANSFORMER BLOCKS ×32                                        │
  │       28 LOCAL (window_size=24, blocks ∉ {7,15,23,31})                │
  │       4 GLOBAL (window_size=0, blocks ∈ {7,15,23,31})                 │
  └─────────────────────────────────────────────────────────────────────────┘

  FOR blk_idx = 0..31:
  ═══════════════════════════════════════════════════════════════════════════
  │
  │  is_global = (blk_idx ∈ {7, 15, 23, 31})
  │  LOCAL:  seq_len=576,  num_windows=9,  total_batches=144 (9×16)
  │  GLOBAL: seq_len=5184, num_windows=1,  total_batches=16
  │
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 1.4.1  PRE-ATTENTION LAYERNORM (norm1)                          │
  │  │                                                                  │
  │  │   LAYERNORM: x_in → x_normed                                    │
  │  │     [5184, 1024] → [5184, 1024]                                │
  │  │     5184 × LayerNorm(1024)  (γ=norm1.weight, β=norm1.bias)     │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 1.4.2  WINDOW PARTITION (LOCAL blocks only)                     │
  │  │                                                                  │
  │  │   Scatter-gather in DRAM:                                       │
  │  │     (72, 72, 1024) → (9 windows, 576, 1024)                   │
  │  │     = (5184, 1024) contiguous per-window                        │
  │  │                                                                  │
  │  │   For each of 9 windows:                                        │
  │  │     For each of 24 rows in window:                              │
  │  │       DMA gather: copy 24×1024 elements from strided source     │
  │  │       DMA write:  to contiguous output                          │
  │  │                                                                  │
  │  │   (No arithmetic — pure data movement)                          │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 1.4.3  Q, K, V PROJECTIONS (3 separate matmuls)                │
  │  │                                                                  │
  │  │   Q/K weights pre-permuted to rotate-half layout at load time   │
  │  │   (complex-pair [ri,ri,...] → [r,r,...,i,i,...] per head)       │
  │  │                                                                  │
  │  │   MATMUL Q: attn_input @ Q_weight^T + Q_bias                   │
  │  │     A = [5184, 1024],  B = [1024, 1024]                        │
  │  │     + bias [1024] broadcast                                     │
  │  │     → Q [5184, 1024]                                            │
  │  │                                                                  │
  │  │   MATMUL K: attn_input @ K_weight^T + K_bias                   │
  │  │     A = [5184, 1024],  B = [1024, 1024]                        │
  │  │     + bias [1024] broadcast                                     │
  │  │     → K [5184, 1024]                                            │
  │  │                                                                  │
  │  │   MATMUL V: attn_input @ V_weight^T + V_bias                   │
  │  │     A = [5184, 1024],  B = [1024, 1024]                        │
  │  │     + bias [1024] broadcast                                     │
  │  │     → V [5184, 1024]                                            │
  │  │                                                                  │
  │  │   3 × matmul(5184, 1024, 1024) + 3 × bias_add(5184, 1024)     │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 1.4.4  MULTI-HEAD RESHAPE (per-window permute)                  │
  │  │                                                                  │
  │  │   For Q, K, V separately:                                       │
  │  │     For each window w (9 or 1):                                 │
  │  │       bf16_permute_core:                                        │
  │  │         (seq_len, 16, 64) → (16, seq_len, 64)                  │
  │  │                                                                  │
  │  │   LOCAL:  9 windows × 3 tensors = 27 permutes                   │
  │  │           each: (576, 16, 64) → (16, 576, 64)                  │
  │  │   GLOBAL: 1 window × 3 tensors = 3 permutes                    │
  │  │           each: (5184, 16, 64) → (16, 5184, 64)               │
  │  │                                                                  │
  │  │   Result: Q,K,V as (total_batches, seq_len, 64)                │
  │  │     LOCAL:  (144, 576, 64)                                      │
  │  │     GLOBAL: (16, 5184, 64)                                      │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 1.4.5  2D RoPE (rotate-half, LOCAL blocks only*)                │
  │  │                                                                  │
  │  │  *Global blocks also use RoPE (scale=24/72=0.333) in reference  │
  │  │   code; HW skips global RoPE due to URAM capacity.             │
  │  │                                                                  │
  │  │   Tables precomputed (host-side) and stored in DRAM:            │
  │  │     cos_lo [576, 64 padded], cos_hi [576, 64 padded]           │
  │  │     sin_lo [576, 64 padded], sin_hi [576, 64 padded]           │
  │  │   (sin_lo is pre-negated: contains -sin values)                │
  │  │                                                                  │
  │  │   Bulk load all 4 tables into URAM_B (4 × 576 × 128B = 294KB) │
  │  │                                                                  │
  │  │   FOR Q then K:                                                 │
  │  │     FOR each batch b = 0..143:                                  │
  │  │       FOR each position t = 0..575:                             │
  │  │         Load x[0:32] (lo half) and x[32:64] (hi half) to URAM │
  │  │                                                                  │
  │  │         ELTWISE MUL: a_lo = x_lo * cos_lo[t]    (32 muls)     │
  │  │         ELTWISE MUL: a_hi = x_hi * cos_hi[t]    (32 muls)     │
  │  │         ELTWISE MUL: b_lo = x_hi * sin_lo[t]    (32 muls)     │
  │  │         ELTWISE MUL: b_hi = x_lo * sin_hi[t]    (32 muls)     │
  │  │         ELTWISE ADD: out_lo = a_lo + b_lo        (32 adds)     │
  │  │         ELTWISE ADD: out_hi = a_hi + b_hi        (32 adds)     │
  │  │                                                                  │
  │  │         Write out_lo, out_hi back to DRAM                       │
  │  │                                                                  │
  │  │   Applied to Q and K: 2 × 144 × 576 × (4×32 muls + 2×32 adds)│
  │  │   = 2 × 144 × 576 × 192 = 31,850,496 element-wise ops         │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 1.4.6  FLASH ATTENTION (batched)                                │
  │  │                                                                  │
  │  │   FOR each batch b = 0..total_batches-1:                        │
  │  │     flash_attention_core(head_dim=64, seq_len=seq_len):         │
  │  │                                                                  │
  │  │     a) MATMUL scores: Q[b] @ K[b]^T                            │
  │  │        [seq_len, 64] × [64, seq_len] → [seq_len, seq_len]     │
  │  │        (pre-scaled: Q multiplied by 1/√64 = 0.125 before matmul│
  │  │         via broadcast_mul on Q)                                 │
  │  │                                                                  │
  │  │     b) SOFTMAX (row-wise):                                      │
  │  │        For each of seq_len rows of seq_len elements:            │
  │  │          max_val = max(row)                                     │
  │  │          row = row - max_val           (seq_len subs)           │
  │  │          row = exp(row)                (seq_len exps)           │
  │  │          sum_val = sum(row)            (seq_len adds)           │
  │  │          row = row / sum_val           (seq_len divs)           │
  │  │                                                                  │
  │  │     c) MATMUL output: attn_probs @ V[b]                        │
  │  │        [seq_len, seq_len] × [seq_len, 64] → [seq_len, 64]     │
  │  │                                                                  │
  │  │   LOCAL:  144 batches × flash_attn(64, 576)                     │
  │  │     scores:  144 × 2×576×64×576  = 6.1B flops                  │
  │  │     softmax: 144 × 576 × (4×576) = 190M ops                    │
  │  │     output:  144 × 2×576×576×64  = 6.1B flops                  │
  │  │                                                                  │
  │  │   GLOBAL: 16 batches × flash_attn(64, 5184)                    │
  │  │     scores:  16 × 2×5184×64×5184  = 55B flops                  │
  │  │     softmax: 16 × 5184 × (4×5184) = 1.7B ops                   │
  │  │     output:  16 × 2×5184×5184×64  = 55B flops                  │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 1.4.7  MERGE HEADS (per-window permute)                         │
  │  │                                                                  │
  │  │   For each window w:                                             │
  │  │     bf16_permute_core:                                          │
  │  │       (16, seq_len, 64) → (seq_len, 16, 64) = (seq_len, 1024) │
  │  │                                                                  │
  │  │   LOCAL:  9 permutes of (16, 576, 64)                           │
  │  │   GLOBAL: 1 permute of (16, 5184, 64)                          │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 1.4.8  OUTPUT PROJECTION                                        │
  │  │                                                                  │
  │  │   MATMUL: merged @ proj_weight^T + proj_bias                    │
  │  │     A = [5184, 1024],  B = [1024, 1024]                        │
  │  │     + bias [1024] broadcast                                     │
  │  │     → [5184, 1024]                                              │
  │  │                                                                  │
  │  │   matmul(5184, 1024, 1024) + bias_add(5184, 1024)              │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 1.4.9  WINDOW UNPARTITION (LOCAL blocks only)                   │
  │  │                                                                  │
  │  │   Scatter in DRAM:                                               │
  │  │     (9 windows, 576, 1024) → (72, 72, 1024) = (5184, 1024)    │
  │  │                                                                  │
  │  │   For each of 9 windows:                                        │
  │  │     For each of 24 rows:                                        │
  │  │       DMA read contiguous → DMA write strided                   │
  │  │                                                                  │
  │  │   (No arithmetic — pure data movement)                          │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 1.4.10  RESIDUAL ADD 1                                          │
  │  │                                                                  │
  │  │   ELEMENTWISE ADD: x_in + attn_output → residual                │
  │  │     [5184, 1024] + [5184, 1024] → [5184, 1024]                │
  │  │     (5,308,416 additions)                                       │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 1.4.11  PRE-MLP LAYERNORM (norm2)                               │
  │  │                                                                  │
  │  │   LAYERNORM: residual → normed                                   │
  │  │     [5184, 1024] → [5184, 1024]                                │
  │  │     5184 × LayerNorm(1024)  (γ=norm2.weight, β=norm2.bias)     │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 1.4.12  MLP: fc1 + GELU + fc2                                  │
  │  │                                                                  │
  │  │   MATMUL fc1: normed @ fc1_weight^T + fc1_bias + GELU          │
  │  │     A = [5184, 1024],  B = [4736, 1024]                        │
  │  │     + bias [4736] broadcast                                     │
  │  │     + GELU activation (fused with matmul on HW)                 │
  │  │     → [5184, 4736]                                              │
  │  │                                                                  │
  │  │     GELU(x) = x · Φ(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))│
  │  │     (applied element-wise: 5184 × 4736 = 24,551,424 elements)  │
  │  │                                                                  │
  │  │   MATMUL fc2: gelu_out @ fc2_weight^T + fc2_bias               │
  │  │     A = [5184, 4736],  B = [1024, 4736]                        │
  │  │     + bias [1024] broadcast                                     │
  │  │     → [5184, 1024]                                              │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 1.4.13  RESIDUAL ADD 2                                          │
  │  │                                                                  │
  │  │   ELEMENTWISE ADD: residual + mlp_output → x_out                │
  │  │     [5184, 1024] + [5184, 1024] → [5184, 1024]                │
  │  │     (5,308,416 additions)                                       │
  │  │                                                                  │
  │  │   x_out becomes x_in for next block                             │
  │  └───────────────────────────────────────────────────────────────────┘
  │
  END FOR (32 blocks)
  │
  │  ViT output: [5184, 1024] = (72, 72, 1024)
  │


═══════════════════════════════════════════════════════════════════════════════
 STAGE 2: FPN NECK  (3 scales after scalp=1 removes 0.5x)
═══════════════════════════════════════════════════════════════════════════════

  ViT output [5184, 1024] = (72, 72, 1024)
    │
    ├─────────────────────────────────────────────────────────────────────
    │ SCALE 1x: (72, 72) → (72, 72, 256)
    │
    │  ┌──────────────────────────────────────────────────────────────┐
    │  │ 2.1  Conv1x1(1024 → 256)                                   │
    │  │                                                              │
    │  │   MATMUL: input @ weight^T + bias                            │
    │  │     A = [5184, 1024],  B = [256, 1024]                      │
    │  │     + bias [256] broadcast                                   │
    │  │     → [5184, 256]                                            │
    │  └──────────────────────────────────────────────────────────────┘
    │    │
    │    ▼
    │  ┌──────────────────────────────────────────────────────────────┐
    │  │ 2.2  Conv3x3(256 → 256, padding=1)                         │
    │  │                                                              │
    │  │   IM2COL: for each (h,w) in 72×72, gather 3×3 neighborhood  │
    │  │     For each pixel: 9 DMA reads of 256 elements             │
    │  │     with zero-padding at borders                             │
    │  │     → im2col buffer [5184, 2304]  (9×256)                   │
    │  │                                                              │
    │  │   MATMUL: im2col @ weight^T + bias                           │
    │  │     A = [5184, 2304],  B = [256, 2304]                      │
    │  │     + bias [256] broadcast                                   │
    │  │     → FPN_1x [5184, 256]                                    │
    │  └──────────────────────────────────────────────────────────────┘
    │
    ├─────────────────────────────────────────────────────────────────────
    │ SCALE 2x: (72, 72) → (144, 144, 256)
    │
    │  ┌──────────────────────────────────────────────────────────────┐
    │  │ 2.3  ConvTranspose2d(1024 → 512, k=2, s=2)                 │
    │  │                                                              │
    │  │   Decomposed into 4 matmuls (one per kernel position):       │
    │  │   FOR (ky, kx) in {(0,0), (0,1), (1,0), (1,1)}:            │
    │  │                                                              │
    │  │     MATMUL: input @ weight_slice[ky,kx]^T                   │
    │  │       A = [5184, 1024],  B = [512, 1024]                    │
    │  │       → temp [5184, 512]                                     │
    │  │                                                              │
    │  │     SCATTER: for each (h,w) in 72×72:                       │
    │  │       temp[h*72+w] → output[2h+ky, 2w+kx]                  │
    │  │       (5184 DMA read + 5184 DMA write per kernel pos)       │
    │  │                                                              │
    │  │     BIAS ADD (first kernel pos only via matmul bias):        │
    │  │       + bias [512] broadcast during (0,0) matmul             │
    │  │     BIAS ADD (remaining 3 positions):                        │
    │  │       For each of 5184 output positions:                     │
    │  │         ELTWISE ADD: output_pixel + bias → output_pixel      │
    │  │         (512 additions per pixel, 3 × 5184 pixels)           │
    │  │                                                              │
    │  │   → [20736, 512] = (144, 144, 512)                          │
    │  │   4 × matmul(5184, 1024, 512) + scatter + bias adds         │
    │  └──────────────────────────────────────────────────────────────┘
    │    │
    │    ▼
    │  ┌──────────────────────────────────────────────────────────────┐
    │  │ 2.4  Conv1x1(512 → 256)                                    │
    │  │                                                              │
    │  │   MATMUL: [20736, 512] @ [256, 512]^T + bias [256]         │
    │  │     → [20736, 256]                                           │
    │  └──────────────────────────────────────────────────────────────┘
    │    │
    │    ▼
    │  ┌──────────────────────────────────────────────────────────────┐
    │  │ 2.5  Conv3x3(256 → 256, padding=1)                         │
    │  │                                                              │
    │  │   IM2COL: 144×144 grid, 3×3 patches                        │
    │  │     → [20736, 2304]                                          │
    │  │   MATMUL: [20736, 2304] @ [256, 2304]^T + bias [256]       │
    │  │     → FPN_2x [20736, 256]                                   │
    │  └──────────────────────────────────────────────────────────────┘
    │
    ├─────────────────────────────────────────────────────────────────────
    │ SCALE 4x: (72, 72) → (288, 288, 256)
    │
    │  ┌──────────────────────────────────────────────────────────────┐
    │  │ 2.6  ConvTranspose2d(1024 → 512, k=2, s=2)                 │
    │  │                                                              │
    │  │   Same as 2.3: 4 matmuls + scatter + bias                   │
    │  │     4 × matmul(5184, 1024, 512) + scatter                   │
    │  │   → [20736, 512] = (144, 144, 512)                          │
    │  └──────────────────────────────────────────────────────────────┘
    │    │
    │    ▼
    │  ┌──────────────────────────────────────────────────────────────┐
    │  │ 2.7  GELU activation                                        │
    │  │                                                              │
    │  │   Applied element-wise on (144, 144, 512) = 10,616,832 elts │
    │  │   GELU(x) per element                                        │
    │  │   (On HW: via identity matmul with gelu_enable=True)        │
    │  └──────────────────────────────────────────────────────────────┘
    │    │
    │    ▼
    │  ┌──────────────────────────────────────────────────────────────┐
    │  │ 2.8  ConvTranspose2d(512 → 256, k=2, s=2)                  │
    │  │                                                              │
    │  │   4 matmuls + scatter + bias                                 │
    │  │     4 × matmul(20736, 512, 256) + scatter                   │
    │  │   → [82944, 256] = (288, 288, 256)                          │
    │  └──────────────────────────────────────────────────────────────┘
    │    │
    │    ▼
    │  ┌──────────────────────────────────────────────────────────────┐
    │  │ 2.9  Conv1x1(256 → 256)                                    │
    │  │                                                              │
    │  │   MATMUL: [82944, 256] @ [256, 256]^T + bias [256]         │
    │  │     → [82944, 256]                                           │
    │  └──────────────────────────────────────────────────────────────┘
    │    │
    │    ▼
    │  ┌──────────────────────────────────────────────────────────────┐
    │  │ 2.10 Conv3x3(256 → 256, padding=1)                         │
    │  │                                                              │
    │  │   IM2COL: 288×288 grid, 3×3 patches                        │
    │  │     → [82944, 2304]                                          │
    │  │   MATMUL: [82944, 2304] @ [256, 2304]^T + bias [256]       │
    │  │     → FPN_4x [82944, 256]                                   │
    │  └──────────────────────────────────────────────────────────────┘

  FPN OUTPUTS:
    FPN_4x: [82944, 256]  = (288, 288, 256)
    FPN_2x: [20736, 256]  = (144, 144, 256)
    FPN_1x: [5184, 256]   = (72, 72, 256)


═══════════════════════════════════════════════════════════════════════════════
 STAGE 3: TEXT ENCODER  (24 layers, dim=1024, 16 heads, head_dim=64, causal)
═══════════════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 3.0  TOKENIZATION (host-side)                                          │
  │                                                                        │
  │   BPE tokenize: "car" → [SOT, token_ids..., EOT, 0, 0, ...]         │
  │   → token_ids [1, 32]  (padded to context_length=32)                  │
  └─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 3.1  TOKEN EMBEDDING (host-side, results DMA'd to DRAM)               │
  │                                                                        │
  │   TABLE LOOKUP: token_embedding[token_ids]                             │
  │     Embedding table: [49408, 1024]                                     │
  │     → [32, 1024]                                                       │
  └─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 3.2  ADD POSITIONAL EMBEDDING                                          │
  │                                                                        │
  │   ELEMENTWISE ADD: token_embeds + pos_embedding                        │
  │     [32, 1024] + [32, 1024] → [32, 1024]                             │
  │     (32,768 additions)                                                 │
  └─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 3.3  CAUSAL MASK (precomputed, stored in DRAM)                         │
  │                                                                        │
  │   Upper-triangular [32, 32] matrix: 0 on/below diagonal, -inf above  │
  │   Used as additive bias in attention scores                            │
  └─────────────────────────────────────────────────────────────────────────┘
    │
    ▼

  FOR t_idx = 0..23 (24 text transformer layers):
  ═══════════════════════════════════════════════════════════════════════════
  │
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 3.4.1  LAYERNORM (ln_1)                                        │
  │  │                                                                  │
  │  │   LAYERNORM: x → normed                                         │
  │  │     [32, 1024] → [32, 1024]                                    │
  │  │     32 × LayerNorm(1024)                                        │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 3.4.2  QKV PROJECTION                                           │
  │  │                                                                  │
  │  │   MATMUL: normed @ in_proj_weight^T + in_proj_bias              │
  │  │     A = [32, 1024],  B = [3072, 1024]                          │
  │  │     + bias [3072] broadcast                                     │
  │  │     → QKV [32, 3072]                                            │
  │  │                                                                  │
  │  │   Split: Q = QKV[:, 0:1024]                                    │
  │  │          K = QKV[:, 1024:2048]                                  │
  │  │          V = QKV[:, 2048:3072]                                  │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 3.4.3  MULTI-HEAD RESHAPE                                       │
  │  │                                                                  │
  │  │   For Q, K, V:                                                   │
  │  │     bf16_permute_core: (32, 16, 64) → (16, 32, 64)             │
  │  │   3 permutes                                                     │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 3.4.4  CAUSAL SELF-ATTENTION (flash_attention with bias)        │
  │  │                                                                  │
  │  │   FOR each head h = 0..15:                                      │
  │  │     flash_attention_core(head_dim=64, seq_len=32):              │
  │  │                                                                  │
  │  │     a) MATMUL scores: Q[h] @ K[h]^T                            │
  │  │        [32, 64] × [64, 32] → [32, 32]                          │
  │  │        (Q pre-scaled by 1/√64 = 0.125)                         │
  │  │                                                                  │
  │  │     b) ADD CAUSAL BIAS: scores + causal_mask                    │
  │  │        [32, 32] + [32, 32] → [32, 32]                          │
  │  │        (1024 additions per head)                                 │
  │  │                                                                  │
  │  │     c) SOFTMAX (row-wise, 32 rows of 32):                      │
  │  │        max, sub, exp, sum, div per row                           │
  │  │                                                                  │
  │  │     d) MATMUL output: softmax @ V[h]                            │
  │  │        [32, 32] × [32, 64] → [32, 64]                          │
  │  │                                                                  │
  │  │   16 heads × flash_attn(64, 32)                                 │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 3.4.5  MERGE HEADS                                              │
  │  │                                                                  │
  │  │   bf16_permute_core: (16, 32, 64) → (32, 16, 64) = (32, 1024) │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 3.4.6  OUTPUT PROJECTION                                        │
  │  │                                                                  │
  │  │   MATMUL: merged @ out_proj_weight^T + out_proj_bias            │
  │  │     A = [32, 1024],  B = [1024, 1024]                          │
  │  │     + bias [1024] broadcast                                     │
  │  │     → [32, 1024]                                                │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 3.4.7  RESIDUAL ADD 1                                           │
  │  │                                                                  │
  │  │   ELEMENTWISE ADD: x_in + attn_output → residual                │
  │  │     [32, 1024] + [32, 1024] → [32, 1024]                      │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 3.4.8  LAYERNORM (ln_2)                                        │
  │  │                                                                  │
  │  │   LAYERNORM: residual → normed                                   │
  │  │     [32, 1024] → [32, 1024]                                    │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 3.4.9  MLP: fc + GELU + proj                                   │
  │  │                                                                  │
  │  │   MATMUL fc: normed @ c_fc_weight^T + c_fc_bias + GELU         │
  │  │     A = [32, 1024],  B = [4096, 1024]                          │
  │  │     + bias [4096] broadcast + GELU fused                        │
  │  │     → [32, 4096]                                                │
  │  │                                                                  │
  │  │   MATMUL proj: gelu_out @ c_proj_weight^T + c_proj_bias        │
  │  │     A = [32, 4096],  B = [1024, 4096]                          │
  │  │     + bias [1024] broadcast                                     │
  │  │     → [32, 1024]                                                │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 3.4.10  RESIDUAL ADD 2                                          │
  │  │                                                                  │
  │  │   ELEMENTWISE ADD: residual + mlp_output → x_out                │
  │  │     [32, 1024] + [32, 1024] → [32, 1024]                      │
  │  └───────────────────────────────────────────────────────────────────┘
  │
  END FOR (24 layers)
    │
    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 3.5  LN_FINAL                                                         │
  │                                                                        │
  │   LAYERNORM: [32, 1024] → [32, 1024]                                 │
  │     32 × LayerNorm(1024)                                               │
  └─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 3.6  RESIZER (Linear 1024 → 256)                                      │
  │                                                                        │
  │   MATMUL: text_tokens @ resizer_weight^T + resizer_bias               │
  │     A = [32, 1024],  B = [256, 1024]                                  │
  │     + bias [256] broadcast                                             │
  │     → text_features [32, 256]                                          │
  └─────────────────────────────────────────────────────────────────────────┘

  TEXT OUTPUT: text_features [32, 256],  text_mask [32] (True=padding)


═══════════════════════════════════════════════════════════════════════════════
 STAGE 4: GEOMETRY ENCODER  (3 layers, dim=256, 8 heads)
 (Text-only path: processes just 1 CLS token)
═══════════════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 4.1  CLS EMBED → FINAL_PROJ → NORM                                    │
  │                                                                        │
  │   COPY: cls_embed [1, 256] (learnable weight)                         │
  │                                                                        │
  │   MATMUL: cls @ final_proj_weight^T + final_proj_bias                 │
  │     A = [1, 256],  B = [256, 256]                                     │
  │     + bias [256] broadcast                                             │
  │     → [1, 256]                                                         │
  │                                                                        │
  │   LAYERNORM: [1, 256] → [1, 256]                                     │
  │     1 × LayerNorm(256)                                                 │
  └─────────────────────────────────────────────────────────────────────────┘
    │
    ▼

  FOR gi = 0..2 (3 geometry encoder layers):
  ═══════════════════════════════════════════════════════════════════════════
  │  (Each layer is a TransformerEncoderLayer: self-attn + cross-attn + FFN)
  │
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 4.2.1  NORM1 + SELF-ATTENTION                                   │
  │  │                                                                  │
  │  │   LAYERNORM: [1, 256] → [1, 256]                               │
  │  │                                                                  │
  │  │   Self-attention on 1 token = identity (trivial):               │
  │  │     QKV MATMUL: [1, 256] @ [768, 256]^T + bias [768]          │
  │  │       → [1, 768] → Q[1,256], K[1,256], V[1,256]               │
  │  │     scores = Q @ K^T / √32 = scalar                            │
  │  │     softmax(scalar) = 1.0                                       │
  │  │     output = 1.0 × V = V                                       │
  │  │     OUT PROJ: [1, 256] @ [256, 256]^T + bias → [1, 256]       │
  │  │                                                                  │
  │  │   RESIDUAL ADD: cls + self_attn_out → cls                       │
  │  │     [1, 256] + [1, 256] → [1, 256]                            │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 4.2.2  NORM2 + CROSS-ATTENTION (CLS queries image)             │
  │  │                                                                  │
  │  │   LAYERNORM: [1, 256] → [1, 256]                               │
  │  │                                                                  │
  │  │   Cross-attention: Q from CLS, K/V from FPN_1x features        │
  │  │     K/V source: FPN_1x [5184, 256] + position encoding         │
  │  │                                                                  │
  │  │   Q PROJ: [1, 256] @ Q_weight → [1, 256] → 8 heads × [1, 32] │
  │  │   K PROJ: [5184, 256] @ K_weight → [5184, 256] → 8 × [5184,32]│
  │  │   V PROJ: [5184, 256] @ V_weight → [5184, 256] → 8 × [5184,32]│
  │  │     (from packed in_proj_weight [768, 256])                     │
  │  │                                                                  │
  │  │   FOR each head h = 0..7:                                       │
  │  │     MATMUL: Q[h] @ K[h]^T                                      │
  │  │       [1, 32] × [32, 5184] → [1, 5184]                        │
  │  │     SCALAR MUL: scores × (1/√32)                                │
  │  │       [1, 5184] × scalar                                        │
  │  │     SOFTMAX: [1, 5184] (single row)                             │
  │  │     MATMUL: softmax @ V[h]                                      │
  │  │       [1, 5184] × [5184, 32] → [1, 32]                        │
  │  │                                                                  │
  │  │   MERGE + OUT PROJ: [1, 256] @ [256, 256]^T + bias → [1, 256] │
  │  │                                                                  │
  │  │   RESIDUAL ADD: cls + cross_attn_out → cls                      │
  │  │     [1, 256] + [1, 256] → [1, 256]                            │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 4.2.3  NORM3 + FFN                                              │
  │  │                                                                  │
  │  │   LAYERNORM: [1, 256] → [1, 256]                               │
  │  │                                                                  │
  │  │   MATMUL fc1: [1, 256] @ [2048, 256]^T + bias [2048]          │
  │  │     → [1, 2048]                                                 │
  │  │   RELU: element-wise max(0, x) on 2048 elements                 │
  │  │   MATMUL fc2: [1, 2048] @ [256, 2048]^T + bias [256]          │
  │  │     → [1, 256]                                                  │
  │  │                                                                  │
  │  │   RESIDUAL ADD: cls + ffn_out → cls                             │
  │  │     [1, 256] + [1, 256] → [1, 256]                            │
  │  └───────────────────────────────────────────────────────────────────┘
  │
  END FOR (3 layers)
    │
    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 4.3  ENCODE NORM                                                       │
  │                                                                        │
  │   LAYERNORM: [1, 256] → [1, 256]                                     │
  └─────────────────────────────────────────────────────────────────────────┘

  GEO OUTPUT: geo_cls [1, 256]


═══════════════════════════════════════════════════════════════════════════════
 STAGE 5: CONCATENATE PROMPT
═══════════════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 5.1  DRAM COPY: text_features [32, 256] → prompt[0:32]               │
  │ 5.2  DRAM COPY: geo_cls [1, 256] → prompt[32:33]                     │
  │                                                                        │
  │   prompt = [33, 256]                                                   │
  │   prompt_mask = [33]  (True for padding positions in text)            │
  └─────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
 STAGE 6: ENCODER FUSION  (6 layers, dim=256, 8 heads, head_dim=32→pad 64)
 Image self-attention + cross-attention to text prompt
═══════════════════════════════════════════════════════════════════════════════

  Input: image features FPN_1x [5184, 256]  (only 1x scale used, num_feature_levels=1)
  Prompt: [33, 256]

  FOR ei = 0..5 (6 encoder layers):
  ═══════════════════════════════════════════════════════════════════════════
  │
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 6.1  NORM1 + SELF-ATTENTION (image tokens attend to image)      │
  │  │                                                                  │
  │  │   LAYERNORM: [5184, 256] → [5184, 256]                         │
  │  │     5184 × LayerNorm(256)                                       │
  │  │                                                                  │
  │  │   Note: query = key = normed + position_encoding (added)        │
  │  │   ELEMENTWISE ADD: normed + pos_embed → query/key               │
  │  │     [5184, 256] + [5184, 256] → [5184, 256]                   │
  │  │                                                                  │
  │  │   QKV MATMUL: query @ in_proj_weight^T + in_proj_bias           │
  │  │     A = [5184, 256],  B = [768, 256]                           │
  │  │     + bias [768] broadcast                                      │
  │  │     → [5184, 768] → Q[5184,256], K[5184,256], V[5184,256]     │
  │  │     (V uses un-added normed, not query+pos)                     │
  │  │                                                                  │
  │  │   MULTI-HEAD RESHAPE with PAD: head_dim=32 → pad to 64         │
  │  │     For Q, K, V:                                                │
  │  │       Zero-fill padded buffer [5184, 8×64] = [5184, 512]      │
  │  │       For each head h, each row r:                              │
  │  │         copy 32 elements to padded position (DMA)               │
  │  │       Permute: (5184, 8, 64) → (8, 5184, 64)                  │
  │  │                                                                  │
  │  │   FLASH SELF-ATTENTION: 8 heads × flash_attn(64, 5184)         │
  │  │     FOR h = 0..7:                                               │
  │  │       MATMUL: Q[h] @ K[h]^T                                    │
  │  │         [5184, 64] × [64, 5184] → [5184, 5184]                │
  │  │       SCALAR MUL: scores × (1/√64)                              │
  │  │         5184 × 5184 = 26,873,856 scalar muls                   │
  │  │       SOFTMAX: 5184 rows × 5184 cols                            │
  │  │       MATMUL: softmax @ V[h]                                    │
  │  │         [5184, 5184] × [5184, 64] → [5184, 64]                │
  │  │                                                                  │
  │  │   MERGE HEADS + UNPAD:                                          │
  │  │     Permute: (8, 5184, 64) → (5184, 8, 64) = (5184, 512)     │
  │  │     MATMUL unpad: (5184, 512) @ unpad_weight^T → (5184, 256)  │
  │  │       unpad_weight is identity-select [256, 512]               │
  │  │                                                                  │
  │  │   OUT PROJ MATMUL: [5184, 256] @ out_weight^T + out_bias       │
  │  │     → [5184, 256]                                               │
  │  │                                                                  │
  │  │   RESIDUAL ADD: x_in + self_attn_out → residual                 │
  │  │     [5184, 256] + [5184, 256] → [5184, 256]                   │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 6.2  NORM2 + CROSS-ATTENTION (image queries text)               │
  │  │                                                                  │
  │  │   LAYERNORM: [5184, 256] → [5184, 256]                         │
  │  │                                                                  │
  │  │   Cross-attention: Q from image(5184), K/V from prompt(33)     │
  │  │                                                                  │
  │  │   Q PROJ from normed_image, K/V PROJ from prompt:               │
  │  │     (packed in_proj_weight [768, 256] split to Q/K/V)          │
  │  │                                                                  │
  │  │   FOR each head h = 0..7:                                       │
  │  │     MATMUL: Q[h] @ K[h]^T                                      │
  │  │       [5184, 32] × [32, 33] → [5184, 33]                      │
  │  │     SCALAR MUL: scores × (1/√32)                                │
  │  │       5184 × 33 = 171,072 scalar muls                          │
  │  │     SOFTMAX: 5184 rows × 33 cols                                │
  │  │     TRANSPOSE V: V[h]^T for matmat_mul_core                    │
  │  │       bf16_permute: [33, 32] → [32, 33]                       │
  │  │     MATMUL: softmax @ V[h]                                      │
  │  │       [5184, 33] × [33, 32] → [5184, 32]                      │
  │  │                                                                  │
  │  │   MERGE + OUT PROJ:                                              │
  │  │     → [5184, 256]                                               │
  │  │                                                                  │
  │  │   RESIDUAL ADD: residual + cross_attn_out → residual            │
  │  │     [5184, 256] + [5184, 256] → [5184, 256]                   │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 6.3  NORM3 + FFN                                                │
  │  │                                                                  │
  │  │   LAYERNORM: [5184, 256] → [5184, 256]                         │
  │  │                                                                  │
  │  │   MATMUL fc1: [5184, 256] @ [2048, 256]^T + bias [2048]       │
  │  │     → [5184, 2048]                                              │
  │  │   RELU: max(0, x) on 5184 × 2048 = 10,616,832 elements        │
  │  │   MATMUL fc2: [5184, 2048] @ [256, 2048]^T + bias [256]       │
  │  │     → [5184, 256]                                               │
  │  │                                                                  │
  │  │   RESIDUAL ADD: residual + ffn_out → x_out                     │
  │  │     [5184, 256] + [5184, 256] → [5184, 256]                   │
  │  └───────────────────────────────────────────────────────────────────┘
  │
  END FOR (6 layers)

  ENCODER OUTPUT: fused_memory [5184, 256]


═══════════════════════════════════════════════════════════════════════════════
 STAGE 7: DECODER  (6 layers, 200 queries, dim=256, 8 heads, head_dim=32)
 No DAC at inference. Presence token prepended during self-attention.
═══════════════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 7.0  INITIALIZE                                                        │
  │                                                                        │
  │   queries = query_embed.weight [200, 256]  (learnable)                │
  │   ref_boxes = sigmoid(reference_points.weight) [200, 4]  (learnable)  │
  │   presence_token = presence_token.weight [1, 256]  (learnable)        │
  └─────────────────────────────────────────────────────────────────────────┘
    │
    ▼

  FOR di = 0..5 (6 decoder layers):
  ═══════════════════════════════════════════════════════════════════════════
  │
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 7.1  CONDITIONAL QUERY POSITION (from reference boxes)          │
  │  │                                                                  │
  │  │  a) SINUSOIDAL EMBEDDING of ref_boxes:                          │
  │  │     For each query q, box = [cx, cy, w, h]:                    │
  │  │       x_embed = cx × 2π                                        │
  │  │       y_embed = cy × 2π                                        │
  │  │       dim_t = 10000^(2*(i//2)/128) for i=0..127               │
  │  │       pos_x = [sin(x/d0), cos(x/d1), sin(x/d2), cos(x/d3)...]│
  │  │       pos_y = [sin(y/d0), cos(y/d1), ...]                      │
  │  │       pos = concat(pos_y, pos_x, pos_w, pos_h)                │
  │  │     → query_sine_embed [200, 512]                               │
  │  │     (200 × 4 coords × 128 sin/cos = lots of transcendentals)  │
  │  │                                                                  │
  │  │  b) REF_POINT_HEAD MLP: sine_embed → query_pos                 │
  │  │     MATMUL L0: [200, 512] @ [256, 512]^T + bias [256]         │
  │  │       → [200, 256]                                              │
  │  │     RELU: 200 × 256 = 51,200 elements                          │
  │  │     MATMUL L1: [200, 256] @ [256, 256]^T + bias [256]         │
  │  │       → query_pos [200, 256]                                    │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 7.2  BOX RPB ATTENTION MASK                                     │
  │  │                                                                  │
  │  │  a) Convert ref_boxes from cxcywh → xyxy:                      │
  │  │     x1 = cx - 0.5w, y1 = cy - 0.5h                            │
  │  │     x2 = cx + 0.5w, y2 = cy + 0.5h                            │
  │  │     (200 × 4 muls + 200 × 4 adds/subs)                        │
  │  │                                                                  │
  │  │  b) Compute spatial coordinates:                                │
  │  │     coords_h = [0/72, 1/72, ..., 71/72]  (72 divs)            │
  │  │     coords_w = [0/72, 1/72, ..., 71/72]  (72 divs)            │
  │  │                                                                  │
  │  │  c) Compute signed distances to box edges:                      │
  │  │     deltas_y = coords_h - boxes_xyxy[:, 1::2]   [200, 72, 2]  │
  │  │     deltas_x = coords_w - boxes_xyxy[:, 0::2]   [200, 72, 2]  │
  │  │     (200 × 72 × 2 = 28,800 subtractions each)                 │
  │  │                                                                  │
  │  │  d) Log-scale encoding:                                         │
  │  │     deltas *= 8                          (scalar muls)          │
  │  │     deltas = sign(d) × log2(|d|+1) / log2(8)                  │
  │  │     (200 × 72 × 2 × {abs, add, log2, mul, sign, mul} each)    │
  │  │                                                                  │
  │  │  e) RPB MLPs (x and y separately):                              │
  │  │     boxRPB_embed_x:                                             │
  │  │       MATMUL L0: [200×72, 2] @ [256, 2]^T + bias              │
  │  │         → [200×72, 256]                                         │
  │  │       RELU: 200 × 72 × 256                                     │
  │  │       MATMUL L1: [200×72, 256] @ [8, 256]^T + bias            │
  │  │         → [200, 72, 8]                                          │
  │  │     boxRPB_embed_y: same dimensions                             │
  │  │       → [200, 72, 8]                                            │
  │  │                                                                  │
  │  │  f) Outer sum to build 2D bias:                                 │
  │  │     BROADCAST ADD: deltas_y[:,:,None,:] + deltas_x[:,None,:,:] │
  │  │       [200, 72, 1, 8] + [200, 1, 72, 8] → [200, 72, 72, 8]  │
  │  │     → [8, 200, 5184] attention bias per head                   │
  │  │                                                                  │
  │  │  g) Prepend zeros for presence token:                           │
  │  │     [8, 201, 5184]  (first row = zeros for presence token)     │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 7.3  SELF-ATTENTION (queries + presence token)                  │
  │  │                                                                  │
  │  │   Prepend presence token to queries:                            │
  │  │     tgt_o2o = cat(presence[1,256], queries[200,256]) = [201,256]│
  │  │     query_pos_o2o = cat(zeros[1,256], query_pos[200,256])      │
  │  │                                                                  │
  │  │   ELEMENTWISE ADD: q = k = tgt + query_pos                     │
  │  │     [201, 256] + [201, 256] → [201, 256]                      │
  │  │     (for both Q and K; V = tgt without pos)                    │
  │  │                                                                  │
  │  │   QKV PROJ (packed in_proj_weight [768, 256]):                  │
  │  │     MATMUL: [201, 256] @ [768, 256]^T + bias [768]            │
  │  │       → [201, 768] → Q,K from q; V from tgt                   │
  │  │     (actually: Q/K projected from tgt+pos, V from tgt)         │
  │  │                                                                  │
  │  │   MULTI-HEAD: 8 heads × head_dim=32                            │
  │  │                                                                  │
  │  │   FLASH SELF-ATTENTION: 8 heads × flash_attn(32, 201)          │
  │  │     FOR h = 0..7:                                               │
  │  │       MATMUL: Q[h] @ K[h]^T  [201,32]×[32,201] → [201,201]   │
  │  │       SCALAR MUL: × (1/√32)                                    │
  │  │       SOFTMAX: 201 rows × 201 cols                              │
  │  │       MATMUL: softmax @ V[h]  [201,201]×[201,32] → [201,32]   │
  │  │                                                                  │
  │  │   MERGE + OUT PROJ:                                              │
  │  │     MATMUL: [201, 256] @ out_weight^T + bias → [201, 256]     │
  │  │                                                                  │
  │  │   RESIDUAL ADD: tgt_o2o + self_attn_out → tgt_o2o              │
  │  │     [201, 256] + [201, 256] → [201, 256]                      │
  │  │                                                                  │
  │  │   NORM2 (LayerNorm):                                            │
  │  │     [201, 256] → [201, 256]                                    │
  │  │     201 × LayerNorm(256)                                        │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 7.4  CROSS-ATTENTION TO TEXT (ca_text)                          │
  │  │                                                                  │
  │  │   Q: tgt + query_pos [201, 256]                                │
  │  │   K, V: prompt [33, 256]  (text memory = prompt passed through)│
  │  │                                                                  │
  │  │   ELEMENTWISE ADD: tgt + query_pos → q_input [201, 256]       │
  │  │                                                                  │
  │  │   QKV from packed in_proj_weight of ca_text:                    │
  │  │     Q MATMUL: [201, 256] → [201, 256]                         │
  │  │     K MATMUL: [33, 256] → [33, 256]                           │
  │  │     V MATMUL: [33, 256] → [33, 256]                           │
  │  │                                                                  │
  │  │   CROSS-ATTENTION: 8 heads × (q_len=201, kv_len=33, hd=32)    │
  │  │     FOR h = 0..7:                                               │
  │  │       MATMUL: Q[h] @ K[h]^T  [201,32]×[32,33] → [201,33]     │
  │  │       SCALAR MUL: × (1/√32)                                    │
  │  │       (+ optional key_padding_mask for text padding positions) │
  │  │       SOFTMAX: 201 rows × 33 cols                               │
  │  │       MATMUL: softmax @ V[h]  [201,33]×[33,32] → [201,32]    │
  │  │                                                                  │
  │  │   MERGE + OUT PROJ:                                              │
  │  │     MATMUL: [201, 256] @ out_weight^T + bias → [201, 256]     │
  │  │                                                                  │
  │  │   RESIDUAL ADD: tgt + ca_text_out → tgt                        │
  │  │     [201, 256] + [201, 256] → [201, 256]                      │
  │  │                                                                  │
  │  │   CATEXT_NORM (LayerNorm):                                      │
  │  │     [201, 256] → [201, 256]                                    │
  │  │     201 × LayerNorm(256)                                        │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 7.5  CROSS-ATTENTION TO IMAGE (with Box RPB bias)              │
  │  │                                                                  │
  │  │   Q: tgt + query_pos [201, 256]                                │
  │  │   K: fused_memory + pos_embed [5184, 256]                      │
  │  │   V: fused_memory [5184, 256]                                  │
  │  │                                                                  │
  │  │   ELEMENTWISE ADD: tgt + query_pos → q [201, 256]             │
  │  │   ELEMENTWISE ADD: memory + pos → k [5184, 256]               │
  │  │                                                                  │
  │  │   QKV from packed in_proj_weight of cross_attn:                 │
  │  │     Q MATMUL: [201, 256] → [201, 256]                         │
  │  │     K MATMUL: [5184, 256] → [5184, 256]                       │
  │  │     V MATMUL: [5184, 256] → [5184, 256]                       │
  │  │                                                                  │
  │  │   CROSS-ATTENTION: 8 heads × (q=201, kv=5184, hd=32)          │
  │  │     FOR h = 0..7:                                               │
  │  │       MATMUL: Q[h] @ K[h]^T                                    │
  │  │         [201, 32] × [32, 5184] → [201, 5184]                  │
  │  │       SCALAR MUL: × (1/√32)                                    │
  │  │         201 × 5184 scalar muls                                  │
  │  │       ADD RPB BIAS: scores + boxRPB[h]                          │
  │  │         [201, 5184] + [201, 5184] → [201, 5184]               │
  │  │         (1,041,984 additions per head)                          │
  │  │       SOFTMAX: 201 rows × 5184 cols                             │
  │  │       MATMUL: softmax @ V[h]                                    │
  │  │         [201, 5184] × [5184, 32] → [201, 32]                  │
  │  │                                                                  │
  │  │   MERGE + OUT PROJ:                                              │
  │  │     MATMUL: [201, 256] @ out_weight^T + bias → [201, 256]     │
  │  │                                                                  │
  │  │   RESIDUAL ADD: tgt + cross_attn_out → tgt                     │
  │  │     [201, 256] + [201, 256] → [201, 256]                      │
  │  │                                                                  │
  │  │   NORM1 (LayerNorm):                                            │
  │  │     [201, 256] → [201, 256]                                    │
  │  │     201 × LayerNorm(256)                                        │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 7.6  FFN                                                        │
  │  │                                                                  │
  │  │   MATMUL fc1: [201, 256] @ [2048, 256]^T + bias [2048]        │
  │  │     → [201, 2048]                                               │
  │  │   RELU: max(0, x) on 201 × 2048 = 411,648 elements            │
  │  │   MATMUL fc2: [201, 2048] @ [256, 2048]^T + bias [256]        │
  │  │     → [201, 256]                                                │
  │  │                                                                  │
  │  │   RESIDUAL ADD: tgt + ffn_out → tgt                            │
  │  │     [201, 256] + [201, 256] → [201, 256]                      │
  │  │                                                                  │
  │  │   NORM3 (LayerNorm):                                            │
  │  │     [201, 256] → [201, 256]                                    │
  │  │     201 × LayerNorm(256)                                        │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 7.7  SPLIT PRESENCE TOKEN                                       │
  │  │                                                                  │
  │  │   presence_out = tgt[0]    → [1, 256]                          │
  │  │   tgt = tgt[1:]           → [200, 256]  (queries only)        │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 7.8  ITERATIVE BOX REFINEMENT                                   │
  │  │                                                                  │
  │  │   NORM: [200, 256] → [200, 256]                                │
  │  │     200 × LayerNorm(256)                                        │
  │  │                                                                  │
  │  │   BBOX_EMBED MLP (3 layers: 256→256→256→4):                    │
  │  │     MATMUL L0: [200, 256] @ [256, 256]^T + bias [256]         │
  │  │       → [200, 256]                                              │
  │  │     RELU: 200 × 256                                             │
  │  │     MATMUL L1: [200, 256] @ [256, 256]^T + bias [256]         │
  │  │       → [200, 256]                                              │
  │  │     RELU: 200 × 256                                             │
  │  │     MATMUL L2: [200, 256] @ [4, 256]^T + bias [4]             │
  │  │       → delta [200, 4]                                          │
  │  │                                                                  │
  │  │   INVERSE SIGMOID on ref_boxes:                                 │
  │  │     clamp(ref_boxes, eps, 1-eps)                                │
  │  │     inv_sig = log(x / (1-x))                                   │
  │  │     (200 × 4 × {clamp, div, sub, log})                        │
  │  │                                                                  │
  │  │   ELEMENTWISE ADD: delta + inv_sigmoid(ref_boxes)              │
  │  │     [200, 4] + [200, 4] → [200, 4]                            │
  │  │                                                                  │
  │  │   SIGMOID: new_ref_boxes = sigmoid(sum)                        │
  │  │     (200 × 4 × {neg, exp, add1, recip})                       │
  │  │     → ref_boxes [200, 4]  (for next layer)                    │
  │  └───────────────────────────────────────────────────────────────────┘
  │    │
  │    ▼
  │  ┌───────────────────────────────────────────────────────────────────┐
  │  │ 7.9  PRESENCE TOKEN SCORING                                     │
  │  │                                                                  │
  │  │   LAYERNORM (presence_token_out_norm):                          │
  │  │     [1, 256] → [1, 256]                                        │
  │  │                                                                  │
  │  │   PRESENCE_TOKEN_HEAD MLP (3 layers: 256→256→256→1):           │
  │  │     MATMUL L0: [1, 256] @ [256, 256]^T + bias → [1, 256]     │
  │  │     RELU: 256 elements                                          │
  │  │     MATMUL L1: [1, 256] @ [256, 256]^T + bias → [1, 256]     │
  │  │     RELU: 256 elements                                          │
  │  │     MATMUL L2: [1, 256] @ [1, 256]^T + bias → [1, 1]         │
  │  │     → presence_logit (scalar)                                   │
  │  └───────────────────────────────────────────────────────────────────┘
  │
  END FOR (6 layers)

  DECODER OUTPUTS:
    hs [6, 200, 256]           (intermediate query features, all layers)
    ref_boxes [7, 200, 4]      (iteratively refined boxes, layers 0-6)
    presence_logits [6, 1]     (presence score per layer)


═══════════════════════════════════════════════════════════════════════════════
 STAGE 8: DOT PRODUCT SCORING
═══════════════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 8.1  PROMPT MLP (on text prompt, residual=True)                        │
  │                                                                        │
  │   MATMUL L0: prompt [33, 256] @ [2048, 256]^T + bias [2048]          │
  │     → [33, 2048]                                                       │
  │   RELU: 33 × 2048 = 67,584 elements                                   │
  │   MATMUL L1: [33, 2048] @ [256, 2048]^T + bias [256]                 │
  │     → [33, 256]                                                        │
  │   RESIDUAL ADD: prompt + mlp_out → prompt_mlp_out                      │
  │     [33, 256] + [33, 256] → [33, 256]                                │
  │   LAYERNORM (out_norm): [33, 256] → [33, 256]                        │
  └─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 8.2  MEAN-POOL TEXT (masking padding tokens)                           │
  │                                                                        │
  │   is_valid = ~text_mask → [33, 1] float (1=valid, 0=pad)             │
  │   ELEMENTWISE MUL: prompt_mlp_out × is_valid                          │
  │     [33, 256] × [33, 1] broadcast → [33, 256]                        │
  │   SUM over seq dim: sum([33, 256], dim=0) → [256]                    │
  │     (33 × 256 = 8,448 additions)                                      │
  │   COUNT valid: sum(is_valid) → num_valid (scalar)                     │
  │   SCALAR DIV: summed / num_valid                                       │
  │     → pooled_prompt [256]                                              │
  └─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 8.3  PROJECT PROMPT + QUERIES                                          │
  │                                                                        │
  │   MATMUL prompt_proj: pooled [1, 256] @ [256, 256]^T + bias [256]    │
  │     → proj_prompt [1, 256]                                             │
  │                                                                        │
  │   MATMUL hs_proj: hs[-1] [200, 256] @ [256, 256]^T + bias [256]     │
  │     → proj_hs [200, 256]                                              │
  │     (using last decoder layer output only at inference)               │
  └─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 8.4  DOT PRODUCT + SCALE + CLAMP                                      │
  │                                                                        │
  │   MATMUL: proj_hs @ proj_prompt^T                                     │
  │     [200, 256] × [256, 1] → [200, 1]                                 │
  │                                                                        │
  │   SCALAR MUL: scores × (1/√256) = scores × 0.0625                    │
  │     200 scalar muls                                                    │
  │                                                                        │
  │   CLAMP: clamp(scores, -12, 12)                                        │
  │     200 clamp ops                                                      │
  │                                                                        │
  │   → out_logits [200, 1]                                                │
  └─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 8.5  FINAL SCORES                                                      │
  │                                                                        │
  │   SIGMOID: out_probs = sigmoid(out_logits)                             │
  │     200 × {neg, exp, add1, recip}                                     │
  │                                                                        │
  │   SIGMOID: presence_score = sigmoid(presence_logits[-1])               │
  │     1 × sigmoid                                                        │
  │                                                                        │
  │   SCALAR MUL: out_probs × presence_score                               │
  │     200 × 1 → [200] final scores                                      │
  └─────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
 STAGE 9: SEGMENTATION HEAD
═══════════════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 9.1  CROSS-ATTEND ENCODER OUTPUT TO TEXT PROMPT                        │
  │                                                                        │
  │   LAYERNORM (cross_attn_norm):                                        │
  │     fused_memory [5184, 256] → normed [5184, 256]                    │
  │     5184 × LayerNorm(256)                                              │
  │                                                                        │
  │   CROSS-ATTENTION: normed queries prompt                              │
  │     Q: normed [5184, 256]                                              │
  │     K, V: prompt [33, 256]                                             │
  │                                                                        │
  │     QKV PROJ (packed in_proj_weight):                                  │
  │       Q MATMUL: [5184, 256] → [5184, 256]                            │
  │       K MATMUL: [33, 256] → [33, 256]                                │
  │       V MATMUL: [33, 256] → [33, 256]                                │
  │                                                                        │
  │     8 heads × cross_attn(q=5184, kv=33, hd=32):                      │
  │       same pattern as 6.2                                              │
  │                                                                        │
  │     MERGE + OUT PROJ → [5184, 256]                                    │
  │                                                                        │
  │   RESIDUAL ADD: fused_memory + cross_attn_out → encoder_visual        │
  │     [5184, 256] + [5184, 256] → [5184, 256]                          │
  └─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 9.2  PIXEL DECODER (3-stage bottom-up FPN fusion)                      │
  │                                                                        │
  │   Inputs (coarse-to-fine):                                             │
  │     FPN_4x [82944, 256] = (288, 288, 256)                            │
  │     FPN_2x [20736, 256] = (144, 144, 256)                            │
  │     encoder_visual [5184, 256] = (72, 72, 256) (replaces FPN_1x)     │
  │                                                                        │
  │   prev = encoder_visual  (72×72)                                       │
  │                                                                        │
  │   ── Stage 0: Fuse with FPN_2x ──                                    │
  │                                                                        │
  │   NEAREST UPSAMPLE 2×: prev (72,72,256) → (144,144,256)              │
  │     For each pixel (h,w) in 72×72:                                    │
  │       DMA read [256]                                                   │
  │       DMA write to 4 output positions                                  │
  │       (5184 reads, 20736 writes — no arithmetic)                      │
  │                                                                        │
  │   ELEMENTWISE ADD: upsampled + FPN_2x                                  │
  │     [20736, 256] + [20736, 256] → [20736, 256]                       │
  │     (5,308,416 additions)                                              │
  │                                                                        │
  │   CONV3x3(256 → 256, padding=1):                                     │
  │     IM2COL: 144×144 grid, 3×3 patches → [20736, 2304]               │
  │     MATMUL: [20736, 2304] @ [256, 2304]^T + bias [256]              │
  │       → [20736, 256]                                                   │
  │                                                                        │
  │   GROUPNORM(8 groups, 256 channels):                                  │
  │     For each of 20736 spatial positions:                               │
  │       For each of 8 groups (32 channels each):                        │
  │         μ = mean(32 values)                                            │
  │         σ² = var(32 values)                                            │
  │         normalize, scale, shift                                        │
  │     20736 × 8 × GroupNorm(32)                                         │
  │                                                                        │
  │   RELU: max(0, x) on 20736 × 256 = 5,308,416 elements                │
  │                                                                        │
  │   ── Stage 1: Fuse with FPN_4x ──                                    │
  │                                                                        │
  │   NEAREST UPSAMPLE 2×: (144,144,256) → (288,288,256)                 │
  │     20736 reads, 82944 writes                                          │
  │                                                                        │
  │   ELEMENTWISE ADD: upsampled + FPN_4x                                  │
  │     [82944, 256] + [82944, 256] → [82944, 256]                       │
  │     (21,233,664 additions)                                             │
  │                                                                        │
  │   CONV3x3(256 → 256, padding=1):                                     │
  │     IM2COL: 288×288 grid → [82944, 2304]                             │
  │     MATMUL: [82944, 2304] @ [256, 2304]^T + bias [256]              │
  │       → [82944, 256]                                                   │
  │                                                                        │
  │   GROUPNORM(8 groups): 82944 × 8 × GroupNorm(32)                     │
  │                                                                        │
  │   RELU: max(0, x) on 82944 × 256 = 21,233,664 elements               │
  │                                                                        │
  │   → pixel_embed [82944, 256] = (288, 288, 256)                       │
  └─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 9.3  INSTANCE SEGMENTATION HEAD                                        │
  │                                                                        │
  │   CONV1x1(256 → 256):                                                │
  │     MATMUL: pixel_embed [82944, 256] @ [256, 256]^T + bias [256]    │
  │       → instance_embeds [82944, 256] = (288, 288, 256)               │
  └─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 9.4  MASK PREDICTOR                                                    │
  │                                                                        │
  │   MASK_EMBED MLP (3 layers: 256→256→256→256):                        │
  │     Using last decoder layer queries hs[-1] [200, 256]:              │
  │                                                                        │
  │     MATMUL L0: [200, 256] @ [256, 256]^T + bias [256]               │
  │       → [200, 256]                                                     │
  │     RELU: 200 × 256                                                    │
  │     MATMUL L1: [200, 256] @ [256, 256]^T + bias [256]               │
  │       → [200, 256]                                                     │
  │     RELU: 200 × 256                                                    │
  │     MATMUL L2: [200, 256] @ [256, 256]^T + bias [256]               │
  │       → mask_embed [200, 256]                                          │
  │                                                                        │
  │   EINSUM (dot product per query per pixel):                            │
  │     mask_preds = mask_embed @ instance_embeds^T                       │
  │     [200, 256] × [256, 82944] → [200, 82944]                         │
  │     = [200, 288, 288]                                                  │
  │                                                                        │
  │     This is a MATMUL: M=200, K=256, N=82944                          │
  │     FLOPs: 2 × 200 × 256 × 82944 ≈ 8.5B                             │
  └─────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
 STAGE 10: POST-PROCESSING
═══════════════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 10.1  FILTER BY CONFIDENCE                                             │
  │                                                                        │
  │   scores [200] from Stage 8.5                                          │
  │   keep = scores > 0.5  (200 comparisons)                              │
  │   N = count(keep)                                                      │
  │                                                                        │
  │   selected_scores [N]                                                  │
  │   selected_boxes [N, 4]                                                │
  │   selected_masks [N, 288, 288]                                         │
  └─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 10.2  BOX FORMAT CONVERSION (cxcywh → xyxy)                           │
  │                                                                        │
  │   For each of N boxes:                                                 │
  │     x1 = cx - 0.5 * w    (1 mul + 1 sub)                             │
  │     y1 = cy - 0.5 * h    (1 mul + 1 sub)                             │
  │     x2 = cx + 0.5 * w    (1 mul + 1 add)                             │
  │     y2 = cy + 0.5 * h    (1 mul + 1 add)                             │
  │     N × 4 muls + N × 4 adds                                           │
  │                                                                        │
  │   SCALE to original image size:                                        │
  │     box × [orig_w, orig_h, orig_w, orig_h]                           │
  │     N × 4 muls                                                        │
  └─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 10.3  MASK UPSAMPLE + THRESHOLD                                       │
  │                                                                        │
  │   BILINEAR INTERPOLATE: [N, 1, 288, 288] → [N, 1, orig_h, orig_w]   │
  │     (standard bilinear: 4 muls + 3 adds per output pixel)            │
  │                                                                        │
  │   SIGMOID: element-wise on all output pixels                           │
  │     N × orig_h × orig_w sigmoids                                      │
  │                                                                        │
  │   THRESHOLD: > 0.5 → bool mask                                        │
  │     N × orig_h × orig_w comparisons                                   │
  └─────────────────────────────────────────────────────────────────────────┘

╔══════════════════════════════════════════════════════════════════════════════╗
║                         OUTPUTS                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  masks:  [N, orig_h, orig_w]  bool  (per-instance binary masks)           ║
║  boxes:  [N, 4]  float  (x1, y1, x2, y2 in original image coords)       ║
║  scores: [N]  float  (confidence ∈ [0, 1])                               ║
╚══════════════════════════════════════════════════════════════════════════════╝


═══════════════════════════════════════════════════════════════════════════════
 OPERATION SUMMARY (per inference)
═══════════════════════════════════════════════════════════════════════════════

  STAGE 1 (ViT backbone, 32 blocks):
    28 local blocks × {3 matmul(5184,1024,1024) + 1 matmul(5184,1024,1024)
                        + 1 matmul(5184,1024,4736) + 1 matmul(5184,4736,1024)
                        + 144×flash_attn(64,576) + 2×144×576 RoPE
                        + 2 LayerNorm(5184,1024) + 2 eltwise_add(5184,1024)}
    4 global blocks × {same matmuls + 16×flash_attn(64,5184) + no RoPE(HW)
                        + 2 LN + 2 adds}
    + patch_embed matmul + pos_embed add + LN_pre

  STAGE 2 (FPN neck):
    Scale 1x: 1×matmul(5184,1024,256) + im2col+matmul(5184,2304,256)
    Scale 2x: 4×matmul(5184,1024,512) + scatter + matmul(20736,512,256)
              + im2col+matmul(20736,2304,256)
    Scale 4x: 4×matmul(5184,1024,512) + GELU + 4×matmul(20736,512,256)
              + scatter + matmul(82944,256,256) + im2col+matmul(82944,2304,256)

  STAGE 3 (Text encoder, 24 layers):
    24 × {matmul(32,1024,3072) + 16×flash_attn(64,32) + matmul(32,1024,1024)
          + matmul(32,1024,4096) + matmul(32,4096,1024) + 2 LN + 2 adds}
    + LN_final + resizer matmul(32,1024,256)

  STAGE 4 (Geometry encoder, 3 layers):
    3 × {self_attn(1 token) + cross_attn(1→5184) + FFN(256→2048→256)
         + 3 LN + 3 adds}

  STAGE 6 (Encoder fusion, 6 layers):
    6 × {8×flash_attn(64,5184) + 8×cross_attn(5184→33)
         + matmul(5184,256,2048) + matmul(5184,2048,256)
         + 3 LN + 3 adds + QKV/out proj matmuls}

  STAGE 7 (Decoder, 6 layers):
    6 × {8×flash_attn(32,201) + 8×cross_attn(201→33) + 8×cross_attn(201→5184)
         + FFN(201,256→2048→256) + bbox MLP + RPB computation + sine embed
         + 4 LN + 4 adds + QKV/out proj matmuls}

  STAGE 8 (Scoring):
    prompt MLP(33,256→2048→256) + pool + 2 proj matmuls + dot product

  STAGE 9 (Segmentation head):
    cross_attn(5184→33) + 2×{upsample + add + im2col+matmul + GN + ReLU}
    + conv1x1(82944,256,256) + mask_embed MLP + einsum(200,256,82944)
```
