# Parakeet-TDT-0.6B on Unified Engine (FPGA Accelerator)

## Intuition

Speech recognition converts audio into text. The challenge is that raw
audio is dense (16,000 samples per second) while text is sparse (~3 words
per second). The model's job is to compress the audio signal into a
sequence of meaningful features, then decode those features into tokens.

**Parakeet-TDT-0.6B** (NVIDIA, 600M params, bfloat16) does this in three stages:

1. **Compress**: A mel spectrogram converts raw audio into 128 frequency
   bands at ~100 frames/second. Three strided convolution stages reduce
   this 8x to ~12 frames/second, then a linear projection maps each frame
   to a 1024-dimensional feature vector. For 10 seconds of audio, this
   produces ~131 feature vectors.

2. **Understand**: 24 Conformer encoder layers process these features.
   Each layer combines local context (depthwise convolution with a 9-tap
   kernel), global context (multi-head self-attention with relative
   position encoding), and non-linear mixing (feed-forward blocks with
   SiLU activation). After 24 layers, each feature vector encodes the
   meaning of its surrounding audio context.

3. **Decode**: A Token-and-Duration Transducer (TDT) walks through the
   encoder output frame-by-frame. At each frame, a 2-layer LSTM predicts
   the next token, a joint network scores all possible tokens, and a
   duration predictor determines how many frames to skip. This lets the
   decoder jump over silence efficiently — for 131 frames, TDT typically
   needs only ~85 decode steps instead of 131.

**On this FPGA**: The encoder compiles to a single ~48 MB instruction
stream (~1.5M instructions) that runs autonomously on the Unified Engine.
The decoder requires host interaction at each step (the token ID determines
the next action). The main bottleneck is instruction fetch latency from
DRAM — ~40% of the encoder's wall-clock time is spent fetching 32-byte
instructions sequentially, not computing.

## Quick Start

```bash
cd ~/unified-engine/models/parakeet

python parakeet_test.py --audio ../../test_samples/1089-134686-0000.wav --dev xdma0

# Live mic streaming (3-second chunks)
python parakeet_stream.py --dev xdma0 --port 8000
```

---

## Model Architecture

Parakeet-TDT-0.6B is NVIDIA's speech recognition model: **Conformer encoder**
+ **Token-and-Duration Transducer** decoder. 600M parameters, bfloat16.

```
                          Audio Waveform (16 kHz mono)
                                    |
                      +-------------v--------------+
                      |    Mel Spectrogram (CPU)    |
                      |  STFT -> |mag|^2 -> mel FB  |
                      |  -> log -> normalize        |
                      +-------------+--------------+
                                    |
                              (B, T_mel, 128)
                                    |
                      +-------------v--------------+
                      |     Subsampling (4x)       |
                      |  Conv2d -> DWConv+PW x2    |
                      |  -> flatten -> Linear      |
                      +-------------+--------------+
                                    |
                              (L_pad, 1024)        L = T_mel / 8
                                    |
          +-------------------------v--------------------------+
          |              Conformer Encoder x 24                |
          |                                                    |
          |  For each layer:                                   |
          |    FF1 ──> Attention ──> ConvModule ──> FF2 ──> LN |
          |                                                    |
          +-------------------------+--------------------------+
                                    |
                              (L_pad, 1024)
                                    |
          +-------------------------v--------------------------+
          |          TDT Decoder (greedy loop)                 |
          |                                                    |
          |  Predictor: 2-layer LSTM (640-dim)                 |
          |  Joint: enc_proj + pred_proj -> ReLU -> logits     |
          |  Output: token IDs + duration (skip) values        |
          |                                                    |
          +-------------------------+--------------------------+
                                    |
                              Token IDs
                                    |
                      +-------------v--------------+
                      |   SentencePiece Decode     |
                      +----------------------------+
                                    |
                                  Text
```

### Key Dimensions

```
D       = 1024    model dimension (embed size)
FF      = 4096    feed-forward hidden dim (4x D)
H       = 8       attention heads
dk      = 128     head dimension (D / H)
K_conv  = 9       depthwise conv kernel size
pad     = 4       conv padding
L       = varies  sequence length (time frames after subsampling)
L_pad   = L rounded up to next multiple of 64
P       = 2L-1    relative positional embedding length
P_pad   = P rounded up to next multiple of 64
```

---

## Detailed Pipeline

### 1. Mel Spectrogram (CPU)

```
waveform (16kHz) ──> STFT(n_fft=512, hop=160, win=400)
                        |
                   complex spectrogram (257 freq bins)
                        |
                   |magnitude|^2  (power spectrum)
                        |
                   mel_filterbank(128, 257) @ power  -> (128, T_mel)
                        |
                   log(clamp(x, min=1e-5))
                        |
                   per-channel normalize: (x - mean) / std
                        |
                   transpose -> (T_mel, 128)  bf16
```

`T_mel = floor(num_samples / hop_length) + 1`. For 10.4s audio: T_mel = 1042.

### 2. Subsampling (4x temporal reduction)

Three stages downsample the time axis by 8x total (stride-2 three times):

```
(T_mel, 128)
     |
     |  Stage 0: Conv2d(1->256, k=3, s=2, p=1) + ReLU        [FPGA]
     |  im2col patches (N0, 64) @ W(64, 256) + bias
     v
(H0, W0, 256)    H0 = (T_mel+1)/2,  W0 = 64
     |
     |  Stage 1: DWConv2d(256, k=3, s=2) + PWConv2d(256) + ReLU   [FPGA]
     |  Host builds DW im2col patches (256, N1_pad, 64), uploads to DRAM
     |  FPGA: 256x per-channel matmul + bias + transpose + PW matmul + ReLU
     v
(H1, W1, 256)    H1 = (H0+1)/2,  W1 = 32
     |
     |  Stage 2: DWConv2d(256, k=3, s=2) + PWConv2d(256) + ReLU   [FPGA]
     |  Same pattern as stage 1
     v
(H2, W2, 256)    H2 = (H1+1)/2,  W2 = 16      <-- H2 = L (encoder seq len)
     |
     |  Flatten: (H2, 256*16) = (L, 4096)
     |
     |  Stage 3: Linear(4096, 1024) + bias                    [FPGA]
     |  matmul (L_pad, 4096) @ (4096, 1024)
     v
(L_pad, 1024)    <-- encoder input, zero-padded from L to L_pad
```

Stages 1-2 use `compile_sub_stage_dw_pw`: the host builds depthwise im2col
patches via `_im2col_conv2d` (using `F.unfold`), uploads them to DRAM, and
the FPGA executes per-channel DW matmuls, adds bias, transposes
(SC, N_out) → (N_out, SC), and runs the pointwise matmul with ReLU.
The im2col patches for stage 1 are ~262 MB (256 channels × 8384 spatial
positions × 64 padded kernel elements) — this is a large DMA transfer
but one-time per inference.

### 3. Conformer Encoder (24 layers)

Each conformer layer has 5 sub-blocks in the Macaron-style arrangement:

```
x ─────────────────────────────────────────────────────────────────> x
  |                                                               ^
  +---> [FF1 half-step] ---> [Self-Attention] ---> [ConvModule] --+
  |           |                     |                    |        |
  |     x + 0.5*FF1(x)    x + Attn(x)          x + Conv(x)      |
  |                                                               |
  +---> [FF2 half-step] ---> [Final LayerNorm] -------------------+
              |
        x + 0.5*FF2(x)
```

#### 3.1 Feed-Forward Block (FF1 and FF2) — K-Split

The FF weight matrices are split in half along the inner dimension at load time.
This avoids the K=4096 pathology where URAM_B overflow forces N_chunk=32 with
per-row DMA writes. With K=2048, N_chunk=64 fits, enabling strided bulk writes.

```
Math:  ff(x) = W2_lo · SiLU(W1_lo · LN(x)) + W2_hi · SiLU(W1_hi · LN(x))
       output = x + 0.5 * ff(x)

  Input x: (L_pad, 1024)
       |
       v
  LayerNorm(x)                        gamma, beta: (1024,)
       |                               Per-row normalize, scale, shift
       v
  (L_pad, 1024)
       |
       +--------- K-split up-projection ---------+
       |                                          |
       v                                          v
  matmul + SiLU                             matmul + SiLU
  (L_pad, 1024) @ (1024, 2048)             (L_pad, 1024) @ (1024, 2048)
  W1_lo: first 2048 output dims             W1_hi: last 2048 output dims
       |                                          |
       v                                          v
  FF_MID_LO (L_pad, 2048)                  FF_MID_HI (L_pad, 2048)
       |                                          |
       +--------- K-split down-projection --------+
       |                                          |
       v                                          v
  matmul                                    matmul
  (L_pad, 2048) @ (2048, 1024)             (L_pad, 2048) @ (2048, 1024)
  W2_lo: first 2048 input dims              W2_hi: last 2048 input dims
       |                                          |
       v                                          v
  partial_lo (L_pad, 1024)                  partial_hi (L_pad, 1024)
       |                                          |
       +---> eltwise_add (SRAM bulk) <------------+
                     |
                     v
               (L_pad, 1024)
                     |
                     v
  half_step_residual:
    output = x + 0.5 * ff_output
    [SRAM bulk: load ff -> broadcast_mul(0.5) -> load x -> eltwise_add -> store]
```

**Hardware instructions for FF block (K-split, L_pad=192):**
| Operation | HW Call | Dims (M,K,N) | Instructions |
|---|---|---|---|
| LayerNorm | `layer_norm_core_dram` | M=192, N=1024 | 197 |
| Up-proj LO + SiLU | `matmat_mul_core(silu)` | (192, 1024, 2048) | ~2135 |
| Up-proj HI + SiLU | `matmat_mul_core(silu)` | (192, 1024, 2048) | ~2135 |
| Down-proj LO | `matmat_mul_core` | (192, 2048, 1024) | ~3138 |
| Down-proj HI | `matmat_mul_core` | (192, 2048, 1024) | ~3138 |
| Accumulate | `eltwise_add` (SRAM) | L_pad × D | 4 |
| Residual | `half_step_residual_core_dram` | bulk | 5 |

Down-proj improved from ~12,420 to ~6,280 (2×3138+4) by crossing the
N_chunk>=64 threshold. Total FF block: ~10,752 (was ~17,851).

#### 3.2 Multi-Head Relative Self-Attention

```
Math:  Q, K, V = LN(x) @ Wq,  LN(x) @ Wk,  LN(x) @ Wv
       P = pos_emb @ Wp
       For each head h:
         content  = (Q_h + bias_u) @ K_h^T
         position = rel_shift((Q_h + bias_v) @ P_h^T)
         scores   = (content + position) / sqrt(dk)
         weights  = softmax(scores + mask)
         head_h   = weights @ V_h
       output = x + concat(heads) @ Wo

  Input x: (L_pad, 1024)
       |
       v
  LayerNorm(x)
       |
       +----------+----------+----------+
       v          v          v          v
   Q=x@Wq     K=x@Wk     V=x@Wv    P=pe@Wp
  (L,1024)   (L,1024)   (L,1024)  (P_pad,1024)
       |          |          |          |
       +----------+----------+----------+
                  |
          For h = 0..7:    (per-head, dk=128)
                  |
       +----------v-----------+
       |  Q_h = Q[:, h*128:(h+1)*128]    gathered via strided DMA
       |  K_h = K[:, h*128:(h+1)*128]    (stride=128*2, jump=1024*2)
       |  V_h = V[:, h*128:(h+1)*128]
       |  P_h = P[:, h*128:(h+1)*128]
       +----------+-----------+
                  |
  Content scores:                    Positional scores:
  (Q_h + bias_u) @ K_h^T            rel_shift((Q_h + bias_v) @ P_h^T)
  (L,128) @ (128,L) = (L,L)         (L,128) @ (128,P_pad) -> shift -> (L,L)
       |                                  |
       +-------------+-------------------+
                     |
              scores = (content + position) / sqrt(128)
              [SRAM: eltwise_add -> broadcast_mul(0.08839)]
                     |
              weights = softmax(scores + mask)
              [matmul (L,L) @ I(L,L) with softmax_enable, bias=mask]
                     |
              head_h = weights @ V_h^T
              (L,L) @ (L,128) = (L,128)
              [V_h transposed via chunked dot-product transpose]
                     |
              Write head_h to ATTN_OUT[:, h*128:(h+1)*128]
              [strided scatter DMA]

          End for h
                  |
                  v
  output_proj = concat_heads @ Wo
  (L_pad, 1024) @ (1024, 1024)
                  |
                  v
  residual = x + output_proj
  [SRAM bulk: eltwise_add]
```

**Relative position encoding**: The model uses additive relative position
biases (Shaw-style). `bias_u` and `bias_v` are per-head learnable vectors
(128 elements each), pre-tiled to `(L_pad, 128)` for bulk SRAM add.
The positional embeddings are sinusoidal:
`pe[t, 2i] = sin(t / 10000^(2i/D))`, `pe[t, 2i+1] = cos(...)`.

**Relative shift (skew trick)**: After computing `(Q+bias_v) @ P^T` which
gives an `(L, 2L-1)` matrix (all possible relative distances from `-L+1` to
`L-1`), `rel_shift` extracts the valid `(L, L)` sub-matrix where entry
`[i,j]` corresponds to relative distance `j-i`. Implemented as L per-row
DMA copies from computed offsets — pure data rearrangement, no arithmetic.

**Attention mask**: An `(L_pad, L_pad)` matrix where valid positions are `0.0`
and padding positions are `-1e38` (effectively -infinity). Added to scores
before softmax so padding positions get zero attention weight.

#### 3.3 Convolution Module

```
Math:  a, b = split(LN(x) @ W_pw1)    W_pw1: (2048, 1024)
       glu    = a * sigmoid(b)
       dw     = DWConv1d(glu, kernel=9)
       bn     = BatchNorm(dw)
       silu   = bn * sigmoid(bn)
       output = x + SiLU(BN(DWConv(GLU(PWConv1(LN(x)))))) @ W_pw2

  Input x: (L_pad, 1024)
       |
       v
  LayerNorm(x)
       |
       +---> matmul (L,1024)@(1024,1024) ---> a: (L_pad, 1024)   [CONV_A_DRAM]
       |              W_pw1a (first 1024 rows)
       +---> matmul (L,1024)@(1024,1024) ---> b: (L_pad, 1024)   [CONV_B_DRAM]
                      W_pw1b (last 1024 rows)
       |
       v
  GLU:  sigmoid(b) via matmul (L,1024) @ I(1024,1024) sigmoid_enable
        output = a * sigmoid(b)   [SRAM bulk eltwise_mul]
        Result: (L_pad, 1024) at CONV_A_DRAM
       |
       v
  Transpose (L_pad, 1024) -> (1024, L_pad)           [CONV_T_DRAM]
  [chunked_transpose: 16 chunks of 64 columns,
   each chunk: strided DMA gather -> dot-product transpose]
       |
       v
  DW Conv1d via Toeplitz matmul:
  For each of 1024 channels:
    (1, L_pad) @ Toeplitz(L_pad, L_pad) = (1, L_pad)
  [Toeplitz matrix encodes 9-tap kernel as diagonals:
   T[i,j] = kernel[j-i+4] if |j-i| <= 4, else 0]
       |
       v
  (1024, L_pad) at CONV_DW_DRAM
       |
       v
  Batch Norm (eval mode, fused):
    output[c,:] = input[c,:] * scale[c] + shift[c]
    where scale = gamma/sqrt(var+eps), shift = beta - mean*scale
  [SRAM bulk: load input -> eltwise_mul(tiled_scale) -> eltwise_add(tiled_shift)]
  [6 instructions total using pre-tiled (1024, L_pad) matrices]
       |
       v
  SiLU:
    sigmoid(x) via matmul (1024,L) @ I(L,L) sigmoid_enable
    output = x * sigmoid(x)   [SRAM bulk eltwise_mul]
       |
       v
  Transpose (1024, L_pad) -> (L_pad, 1024)           [CONV_T_DRAM]
       |
       v
  PW Conv2: matmul (L_pad, 1024) @ (1024, 1024)
       |
       v
  Residual: x + conv_output   [SRAM bulk eltwise_add]
```

**Why two transposes?** The depthwise conv operates on each channel
independently along the time axis. In `(L, D)` layout, channels are
columns — accessing a single channel requires strided reads across all L
rows. In `(D, L)` layout, each channel is a contiguous row, making the
per-channel Toeplitz matmul natural: `row @ Toeplitz`. After the conv +
BN + SiLU (all in channel-first layout), we transpose back to `(L, D)` for
the pointwise conv2 matmul.

**Toeplitz convolution matrix**: A 9-tap 1D convolution
`y[t] = sum_{k=0}^{8} w[k] * x[t+k-4]` is equivalent to
`y = T @ x` where T is an `(L, L)` banded matrix:

```
T = | w4  w5  w6  w7  w8   0   0  ...  0  |
    | w3  w4  w5  w6  w7  w8   0  ...  0  |
    | w2  w3  w4  w5  w6  w7  w8  ...  0  |
    | w1  w2  w3  w4  w5  w6  w7  ...  0  |
    | w0  w1  w2  w3  w4  w5  w6  ...  0  |
    |  0  w0  w1  w2  w3  w4  w5  ...  0  |
    |  :                              :    |
    |  0   0  ...  w0  w1  w2  w3  w4  w5  |
    |  0   0  ...   0  w0  w1  w2  w3  w4  |
```

Each channel has different weights, so there are 1024 different Toeplitz
matrices, each `(L_pad, L_pad)`. Pre-built once at initialization.

### 4. TDT Decoder (Token-and-Duration Transducer)

The decoder is **not** a single instruction stream. It is a host-driven
loop where the host executes small FPGA programs, reads back results, and
decides what to do next. This is unavoidable because the control flow
(which token was predicted, whether to emit or skip frames) depends on
the FPGA's output at each step.

There are **3 compiled FPGA programs**:

| Program | What it computes | Instructions | Why separate |
|---|---|---|---|
| `pred_prog` | 2-layer LSTM predictor step | ~50 | Host must save/restore LSTM state on blank |
| `tok_prog` | Joint network + token logits + argmax | ~20 | Host must read argmax before deciding next action |
| `dur_prog` | Duration logits + argmax | ~5 | Only needed after tok_prog determines the token |

#### Execution Flow

```
Host (CPU)                              FPGA
    |
    |  For each encoder frame t = 0, 1, ..., L-1:
    |
    |  ---- SAVE LSTM STATE ----
    |  DMA read h0, c0, h1, c1
    |  (save in case we need to
    |   undo on blank token)
    |
    |  ---- PREDICTOR ----
    |  DMA write embedding[last_token]
    |  to PRED_EMB_DRAM
    |                                    |
    |---execute pred_prog -------------->|
    |                                    |  LSTM layer 0:
    |                                    |    gates = emb @ W_ih0 + b_ih0
    |                                    |          + h0  @ W_hh0 + b_hh0
    |                                    |    i = sigmoid(gates[0:640])
    |                                    |    f = sigmoid(gates[640:1280])
    |                                    |    g = tanh(gates[1280:1920])
    |                                    |    o = sigmoid(gates[1920:2560])
    |                                    |    c0_new = f * c0 + i * g
    |                                    |    h0_new = o * tanh(c0_new)
    |                                    |
    |                                    |  LSTM layer 1:
    |                                    |    (same, using h0_new as input)
    |                                    |
    |                                    |  pred_out = h1_new
    |<---done----------------------------|
    |
    |  ---- JOINT (TOKEN) ----
    |  DMA copy enc_out[t] (1024 elems)
    |  to JOINT_ENC_DRAM
    |                                    |
    |---execute tok_prog --------------->|
    |                                    |  enc_proj  = enc_out[t] @ W_enc + b_enc
    |                                    |    (1,1024) @ (1024,640) = (1,640)
    |                                    |
    |                                    |  pred_proj = pred_out @ W_pred + b_pred
    |                                    |    (1,640) @ (640,640) = (1,640)
    |                                    |
    |                                    |  joint = ReLU(enc_proj + pred_proj)
    |                                    |    [eltwise_add -> identity matmul
    |                                    |     with relu_enable]
    |                                    |
    |                                    |  tok_logits = joint @ W_tok + b_tok
    |                                    |    (1,640) @ (640,8256) = (1,8256)
    |                                    |    [HW argmax tracks max during matmul]
    |                                    |
    |<---done----------------------------|
    |
    |  token_id = read HW argmax register
    |
    |  ---- DECISION ----
    |
    |  if token_id == <blank> (8192):
    |    |
    |    |  Restore saved LSTM state
    |    |  (DMA write saved h0,c0,h1,c1
    |    |   back to PRED_H0/C0/H1/C1_DRAM)
    |    |
    |    |---execute dur_prog ---------->|
    |    |                               |  dur_logits = joint @ W_dur + b_dur
    |    |                               |    (1,640) @ (640,64) = (1,64)
    |    |<--done------------------------|
    |    |
    |    |  duration = read HW argmax register
    |    |  t += max(duration, 1)
    |    |  (advance past frames with no speech)
    |    |
    |  else (real token):
    |    |
    |    |  EMIT token_id to output list
    |    |  last_token = token_id
    |    |
    |    |---execute dur_prog ---------->|
    |    |                               |  (same as above)
    |    |<--done------------------------|
    |    |
    |    |  duration = read HW argmax register
    |    |  if duration > 0:
    |    |    t += duration   (skip frames)
    |    |  else:
    |    |    loop back to PREDICTOR
    |    |    (emit another token at same t)
    |    |
```

#### Why 3 separate programs?

**pred_prog must be separate** because on blank tokens, the host needs to
**undo** the LSTM state update. If the predictor were merged with the joint
network, the LSTM hidden/cell states would already be corrupted before the
host knew the result was blank. By running the predictor first, the host
can save state beforehand and restore it if needed.

**tok_prog and dur_prog are separate** because the token result determines
whether LSTM state gets restored *before* the duration is read. On a blank
token, the sequence is: read token argmax -> restore LSTM -> run dur_prog.
If they were one program, the duration would be computed with the wrong
(un-restored) joint state.

**tok_prog could theoretically include dur_prog** for the non-blank path
(where no restore is needed), but keeping them separate simplifies the
single code path.

#### LSTM Gate Math (inside pred_prog)

Each LSTM layer computes 4 gates from a single (1, 2560) vector, which is
the sum of two matmuls:

```
gates = x @ W_ih + b_ih + h_prev @ W_hh + b_hh

  where x is (1, 640), W_ih is (640, 2560), h_prev is (1, 640), W_hh is (640, 2560)

Split into 4 slices of 640:
  gates[0:640]    -> i (input gate)    -> sigmoid    controls what new info to store
  gates[640:1280] -> f (forget gate)   -> sigmoid    controls what old info to keep
  gates[1280:1920]-> g (candidate)     -> tanh       new candidate values
  gates[1920:2560]-> o (output gate)   -> sigmoid    controls what to output

Cell update:   c_new = f * c_prev + i * g
Hidden update: h_new = o * tanh(c_new)
```

Sigmoid is implemented as `matmul (1,640) @ I(640,640) with sigmoid_enable`.
Tanh is implemented as `2 * sigmoid(2x) - 1`: broadcast_mul(2) -> sigmoid
via identity matmul -> broadcast_mul(2) -> broadcast_add(-1).

#### TDT vs Standard RNN-T

Standard RNN-T advances by exactly 1 frame on every blank token. For an
encoder output of L frames with ~50 real tokens, that means ~L decode
steps (one per frame), most of which are blanks.

TDT (Token-and-Duration Transducer) predicts a **duration** alongside each
token. On blank, `duration` can be 1-4, allowing the decoder to skip
multiple silent frames at once. On real tokens, `duration > 0` means
"this token spans multiple frames" (e.g., a long vowel).

For 10.4s audio (L=131 frames), standard RNN-T would need ~131 steps.
TDT typically needs **~85 steps** (50 tokens + 35 blank-with-skip), a ~1.5x
reduction in decoder iterations.

#### Decoder Timing

Each decode step involves:
- 4 DMA reads (save LSTM state): ~0.1ms
- 1 DMA write (embedding): ~0.05ms
- pred_prog execution: ~0.5ms (2 matmuls + gates)
- 1 DMA write (enc_out[t]): ~0.05ms
- tok_prog execution: ~0.5ms (3 matmuls + argmax)
- 1 register read (token argmax): ~0.01ms
- Optional: 4 DMA writes (restore LSTM): ~0.1ms
- dur_prog execution: ~0.1ms (1 small matmul + argmax)
- 1 register read (duration argmax): ~0.01ms

**Per step: ~1.4ms**. For 85 steps: **~120ms** on FPGA.

But the measured decoder time is **~2.9s** because the host-side DMA
round-trips (Python -> kernel driver -> PCIe -> FPGA -> PCIe -> kernel
driver -> Python) dominate. Each DMA call has ~10-30ms of Python/driver
overhead, and there are ~15 DMA operations per step x 85 steps = ~1,275
DMA round-trips.

---

## Hardware Mapping

### DRAM Memory Map (4 GB)

```
0x00000000 +---------------------------+
           |                           |
           |    Parameters (Weights)   |  ~2.93 GB
           |    - 24x conformer layers |
           |    - K-split FF weights   |
           |    - Identity matrices    |
           |    - Toeplitz matrices    |
           |    - Tiled BN/bias params |
           |    - Predictor + Joint    |
           |                           |
0xBB000000 +---------------------------+
           |                           |
           |    Tensor (Activations)   |  ~592 MB
           |    - Encoder intermediates|
           |    - Decoder buffers      |
           |    - Subsampling scratch  |
           |                           |
0xE0000000 +---------------------------+
           |                           |
           |    Programs (Instructions)|  512 MB
           |    - Encoder program      |
           |    - Decoder programs     |
           |    - Subsampling programs |
           |                           |
0xFFFFFFFF +---------------------------+
```

### SRAM (URAM) Layout

```
URAM_A (512 KB = 262,144 bf16 elements = 4096 rows of 64)
0x00000 +---------------------------+
        | A-matrix rows (input)     |   M_chunk * K elements
        |                           |
        +---------------------------+
        | Output sub-block          |   M_chunk * N_chunk elements
        |                           |
        +---------------------------+
        | (unused)                  |
0x7FFFF +---------------------------+

URAM_B (512 KB = 262,144 bf16 elements)
0x80000 +---------------------------+
        | B-matrix rows (weights)   |   N_chunk * K elements
        |                           |
        +---------------------------+
        | (unused / softmax output) |
0xFFFFF +---------------------------+
```

### Matmul Tiling Strategy

The hardware performs matrix multiplication as batched dot products:

```
C[i,j] = A[i,:] · B[j,:]     (dot product of row i of A with row j of B)

  For each M_chunk of A rows:
    Load M_chunk rows of A into URAM_A

    For each N_chunk of B rows:
      Load N_chunk rows of B into URAM_B

      For each output row i in M_chunk:
        HW dot product: A_row[i] · B_chunk  -> output[i, j..j+N_chunk]
        (K/64 vector MAC operations per dot product)

      DMA write output sub-block to DRAM (strided if N_chunk >= 64)
```

**Chunk sizing constraints**:
- `N_chunk = floor(URAM_B_capacity / K)`, rounded down to multiple of 64
- `M_chunk = floor(URAM_A_capacity / (K + N_chunk))`
- When `N_chunk < 64`: fallback to 32 or 16, output alignment requires per-row DMA

**Example: (192, 1024, 1024) — QKV projection**
```
N_chunk = min(1024, floor(262080/1024)/64*64) = 192
M_chunk = min(192, floor(262144/(1024+192))) = 192  (all fits!)
N_chunks = ceil(1024/192) = 6
Total: 1 A-load + 6 * (1 B-load + 192 dot-products + 1 strided-write)
     = 1,165 instructions
```

**Example: (192, 2048, 1024) — FF down-projection (K-split)**
```
N_chunk = floor(262080/2048)/64*64 = 64   (crosses the 64-element threshold!)
M_chunk = floor(262144/(2048+64)) = 124
M_chunks = 2 (124, 68)
N_chunks = 16
Per M-chunk: 1 A-load + 16 * (1 B-load + m_take dots + 1 strided-write)
Total: 3,138 instructions  (x2 halves + eltwise_add = ~6,280)
```

Before K-split, the original (192, 4096, 1024) matmul hit the N_chunk<64
pathology (N_chunk=32, per-row DMA writes), costing 12,420 instructions.
K-split halves K from 4096 to 2048, which makes N_chunk=64 fit in URAM_B,
enabling strided bulk writes. Two half-matmuls + accumulate = ~6,280 total.

### Activation Functions via LALU

The hardware has a **Look-Ahead Linear/Activation Unit (LALU)** that applies
non-linearities inline during matmul output writeback:

```
matmul output element z ──> LALU ──> stored value
                              |
          +-------------------+-------------------+
          |         |         |         |         |
        BYPASS    SiLU     GELU    SIGMOID     RELU
         z      z*σ(z)   z*Φ(z)    σ(z)     max(0,z)
```

For standalone activations (no matmul needed), we multiply by an identity
matrix to trigger the LALU: `sigmoid(x) = (x @ I) with sigmoid_enable`.
This is used for GLU's `sigmoid(b)`, SiLU in the conv module, LSTM gates,
and the attention softmax passthrough.

### Softmax Implementation

Softmax is a 3-phase operation built into `matmat_mul_core`:

```
Phase 1: Normal matmul with fmax tracking
  Each dot-product result updates a per-row running max register
  (indexed by fmax_context_addr = output_row, max 64 rows)

Phase 2: Bias addition (attention mask)
  bias_mode="full_matrix": load per-row mask from ATTN_MASK_DRAM
  Added to each output element before softmax

Phase 3: Softmax (per-row)
  For each row:
    1. EXP instruction: computes exp(x - row_max) for all elements
       (row_max subtraction via FMAX_NEGATE broadcast mode)
       LALU accumulates sum and computes 1/sum via RECIP
    2. MUL_BROADCAST: multiply all elements by 1/sum
       (using LALU_RESULT as the scalar)
  Result: row sums to 1.0, numerically stable via max subtraction
```

### LayerNorm Implementation

Per-row normalization, with gamma/beta loaded once into SRAM_B:

```
Setup (once per call):
  Load zeros, gamma, beta to SRAM_B: 3 DMA instructions

For each chunk of rows (chunk_size = floor(262080/N) = 255 for N=1024):
  Load chunk to SRAM_A: 1 DMA instruction

  For each row x of length N=1024:
    layer_norm_core hardware instruction:
      1. ADD_REDUCE + RECIP(N) -> mean
      2. ADD_BROADCAST(-mean) -> x_centered
      3. RMS + RSQRT -> 1/std
      4. MUL_BROADCAST(1/std) -> x_norm
      5. ELTWISE_MUL(gamma) -> x_scaled
      6. ELTWISE_ADD(beta) -> output
    (all 6 sub-ops are fused into 1 hardware instruction)

  Write chunk to DRAM: 1 DMA instruction

Total for M=192, N=1024: 3 + 1 + 192 + 1 = 197 instructions
```

### Transpose via Strided DMA + Dot-Product

The hardware has no native transpose instruction. Instead, it uses
strided DMA to gather column slices and identity-matrix dot products to
transpose each slice:

```
To transpose (M, N) -> (N, M):

Process N in 64-column chunks:
  For each chunk c of 64 columns:
    1. Strided DMA gather: ONE instruction collects all M rows'
       64-element column slice into contiguous SRAM (M, 64)
       (stride_bytes_per_chunk=64*2, stride_jump_bytes=N*2)

    2. Write to temp DRAM: ONE contiguous DMA write (M*64 elements)

    3. Dot-product transpose: for each of 64 output rows:
       dot_product(identity_row[i], temp_column[j]) = temp[j, i]
       This effectively reads column i of the temp buffer

    4. Write transposed chunk (64, M_aligned) to output DRAM

Per chunk: 2 (gather+write) + ~130 (permute dot-products + writes) = ~132
Total: 16 chunks for N=1024 -> ~2112 instructions
```

The strided DMA gather (step 1) replaced a per-row extraction loop that
previously did 2*M individual DMAs per chunk (~384 for M=192). This reduced
the transpose from ~8224 to ~2112 instructions for (192, 1024).

---

## Weight Staging Summary

All transformations applied during `weight_init()` before any execution:

```
Original Weight                 Staged Shape          Method & Reason
─────────────────────────────   ───────────────────   ──────────────────────────────
FF1/FF2 W1 (4096,1024)      -> W1_LO(2048,1024)      K-split: halve output dim
                              + W1_HI(2048,1024)      avoids K=4096 URAM_B overflow
FF1/FF2 W2 (1024,4096)      -> W2_LO(1024,2048)      K-split: halve input dim
                              + W2_HI(1024,2048)      enables N_chunk=64 (was 32)
DW Conv1d (1024,1,9)         -> (1024, 64)            im2col zero-pad K to block_size
Sub Conv2d (256,1,3,3)       -> (256, 64)             im2col zero-pad 9 to 64
PW Conv1 (2048,1024,1)       -> 2x (1024, 1024)      split at D for GLU + squeeze
PW Conv2 (1024,1024,1)       -> (1024, 1024)          squeeze trailing dim
Joint out (8198,640)         -> tok(8256,640)          row-pad to 64-multiple + split
                              + dur(64,640)            padding biases set to -inf
BN params (4 vectors)        -> scale(1024,), shift(1024,)  fuse gamma/sqrt(var+eps)
BN tiled  scale(1024,)       -> (1024, L_pad)         broadcast for bulk eltwise_mul
Attn bias (8,128)            -> 8x (L_pad, 128)       tile per-head for bulk add
DW kernel (1024,1,9)         -> (1024, L_pad, L_pad)  Toeplitz conv matrix
```

---

## Instruction Budget (L_pad=192)

Per conformer layer: **~62,500 instructions**

```
                    FF1         Attention      Conv Module     FF2        LN
                +-----------+  +-----------+  +-----------+  +---------+  +----+
LayerNorm       |     197   |  |     197   |  |     197   |  |    197  |  | 197|
Up-proj (K-split|   4,270   |  |           |  |           |  |  4,270  |  |    |
  2x matmul)    |           |  |           |  |           |  |         |  |    |
Down-proj(K-spl)|   6,280   |  |           |  |           |  |  6,280  |  |    |
  2x matmul+add)|           |  |           |  |           |  |         |  |    |
Matmul (D->D)   |           |  |   5,825   |  |  4,499    |  |         |  |    |
  (QKV+pos+out) |           |  |           |  | (PW1+PW2) |  |         |  |    |
Per-head ops    |           |  |  ~13,900  |  |           |  |         |  |    |
  (scores, rel_shift, softmax, V transpose)  |           |  |         |  |    |
Transpose       |           |  |           |  |   2,508   |  |         |  |    |
  (strided DMA) |           |  |           |  | (fwd+back)|  |         |  |    |
DW Conv (1024x) |           |  |           |  |  ~3,072   |  |         |  |    |
SiLU matmul     |           |  |           |  |   1,036   |  |         |  |    |
BN (tiled)      |           |  |           |  |       6   |  |         |  |    |
GLU             |           |  |           |  |   1,169   |  |         |  |    |
Residual/eltwise|       5   |  |       4   |  |       4   |  |      5  |  |    |
                +-----------+  +-----------+  +-----------+  +---------+  +----+
Subtotal:         10,752        ~19,926         12,491        10,752       197
```

**24 layers total: ~1.5M instructions** (measured: 1,499,978)

**Optimizations applied:**
- **K-split FF**: down-proj (L_pad,4096,1024) split into 2x(L_pad,2048,1024)
  + eltwise_add. Avoids N_chunk=32 pathology. Saves ~6,140 inst/block.
- **Strided DMA transpose**: column extraction uses one strided DMA instead
  of M*2 per-row DMAs. Saves ~6,100 inst per (192,1024) transpose.

**Top consumers**:
```
  17%  FF down-proj K-split (2x matmul + accumulate, 48 calls)
  17%  FF up-proj K-split (2x matmul, 48 calls)
  18%  Attention per-head ops (scores, softmax, V transpose, 8 heads)
   8%  QKV + output + pos projection matmuls (5 per layer)
   8%  Conv module PW1/PW2 + DW conv + GLU + SiLU
   4%  Conv module transposes (strided DMA + dot-product)
   7%  Softmax (per-row exp + normalize)
   7%  LayerNorm (5 per layer, per-row normalize)
   4%  Relative shift + attention bias ops
  10%  Everything else (BN, residuals, DMA scatter/gather)
```

**Timing breakdown (wall-clock vs HW counter):**
```
  HW execution (latency register):   ~13.4s  (2.38B cycles @ 5.63ns)
  Instruction fetch overhead:         ~8.9s  (~5.9us per instruction fetch)
  Wall-clock total:                  ~22.3s  (~14.9us per instruction)
```

~40% of wall-clock time is instruction fetch from DRAM — the FPGA fetches
each 32-byte instruction sequentially with no prefetch/double-buffering.
This is a hardware design bottleneck, not reducible in software.

---

## Host/FPGA Interaction and CPU Round-Trips

Not all computation can be a single uninterrupted FPGA instruction stream.
This model has several points where the host CPU must intervene between
FPGA program executions, each requiring PCIe round-trips that add latency.

### Where host intervention is required

```
Pipeline Phase        Host Role                          Why CPU is in the loop
──────────────────    ─────────────────────────────────  ────────────────────────────────
Mel spectrogram       Compute entirely on CPU            Signal processing (STFT, log,
                                                         normalize) — not worth FPGA port

Subsampling           Build im2col patches per stage,    im2col is a data rearrangement
stages 0-2            upload patches, execute FPGA       that depends on spatial layout;
                      program, read result back          FPGA has no im2col instruction

Subsampling           Host flattens (H2, W2, SC) to     Reordering from spatial to
flatten               (H2, 4096), uploads to DRAM        feature-vector layout

Encoder               Fully autonomous FPGA execution    Single instruction stream,
(24 layers)           — NO host intervention             no data-dependent control flow

Decoder               Host drives every step:            Control flow depends on FPGA
(greedy loop)         save LSTM state, run pred_prog,    output — the token ID determines
                      read argmax, decide blank/emit,    whether to restore LSTM state,
                      optionally restore state,          and the duration determines how
                      run dur_prog, read duration,       many frames to skip. These
                      advance frame pointer              decisions cannot be made on FPGA.
```

### Why the decoder is the worst offender

The encoder compiles to a single ~48 MB instruction stream that runs
autonomously on the FPGA for ~22 seconds with zero host interaction.
The decoder, by contrast, requires **~15 host DMA operations per step**
across **~85 steps** = **~1,275 PCIe round-trips**.

Each round-trip (Python → kernel driver → PCIe → FPGA → PCIe → Python)
takes ~10-30ms of host overhead, even though the actual FPGA work per
program is <1ms. This means:

```
Decoder FPGA compute:     85 steps × ~1.1ms = ~94ms
Decoder host overhead:    1,275 DMAs × ~2.2ms avg = ~2,800ms
                          ─────────
Measured decoder time:    ~2,856ms  (97% is host overhead!)
```

The decoder cannot be merged into one instruction stream because each
step's output determines the next step's input:
- **Token argmax** → decides blank vs real token → determines LSTM restore
- **Duration argmax** → decides frame advance → determines loop termination
- **LSTM state save/restore** → must happen between predictor and joint

This is inherent to autoregressive decoding — each token depends on all
previous tokens. The only way to reduce this overhead is to minimize
per-DMA host latency (faster driver, kernel bypass, or batched DMA ops).

### Subsampling host overhead

Subsampling stages 1-2 each require a host round-trip:
1. Read previous stage output from FPGA DRAM (~17 MB for stage 0 → 1)
2. Build im2col patches on CPU (~262 MB for stage 1)
3. Upload patches to FPGA DRAM (~262 MB DMA write)
4. Execute FPGA program
5. Read result

This adds ~200-500ms per stage due to the large DMA transfers. The
encoder's im2col is avoided entirely because it uses Toeplitz matrices
(pre-built once, stored in DRAM). The subsampling stages use 2D
convolutions with stride, which have different spatial patterns than the
1D depthwise convs in the encoder, requiring fresh im2col per inference.

---

## Streaming Mode

For live transcription, `parakeet_stream.py` uses **L_pad=64** (3-second
audio chunks produce L~38 frames):

```
  3s audio -> 301 mel frames -> subsampling -> 38 time frames -> pad to 64

  Instruction reduction vs L_pad=192:
    Attention (L^2): 9x smaller
    Everything else: 3x smaller
    Estimated encoder time: ~7s per chunk (vs 22.3s)
```

The streaming architecture decouples audio capture from FPGA processing:

```
  Browser mic ──WebSocket──> audio_buffer (lock-protected)
                                  |
  Transcription thread:           |
    loop:                         |
      sleep(0.1)                  |
      grab buffer ────────────────+
      if >= 48000 samples (3s):
        take exactly 48000
        send to FPGA pipeline
        print result
```

The WebSocket handler **never blocks** on transcription — it only appends
audio to the buffer. The dedicated transcription thread polls every 100ms
and processes complete 3-second windows. If the FPGA takes longer than 3
seconds, audio accumulates and is processed as the next chunk.
