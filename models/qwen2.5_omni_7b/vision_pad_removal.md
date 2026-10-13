# Qwen2.5-Omni vision padding removal — hardware-verified

Status: **done and verified on hardware.** Padding removal is unconditional —
there are no switches. `--vis-qkv-strip` and `--vis-mlp-real-n` have been
removed from the CLI; the encoder always projects Q/K/V at their real widths
with strip write-back, and gate/up always run at real N on a padded row pitch.

Board: `xdma0`, FPGA build `0x91d40cab` (non-aligned DMA), 333.33 MHz, AXI 256,
8 GiB, 8 engines, `--image` (`yosemite.jpg`, 576 patches -> 144 soft tokens).
Requires the non-aligned-DMA image; an older build or a `params.bin` without
compact `attn.qk`/`attn.v` weights is now a hard error, not a fallback.

## What is left: inner K only

Every removable padding is gone. The encoder spends **no FLOPs** on padding
except on the inner (reduction) K axis, which the 64-element quantization block
and the attention core require:

| Remaining padding | Real | Runs at | Why it cannot go |
| --- | --- | --- | --- |
| Attention `Q.K^T` head_dim | 80 | **128** | Reduction axis; 80 is not a multiple of the 64-element block, and the attention core needs a 64-aligned head stride. Largest remaining waste. |
| `P.V` reduction (= KV length) | Tkv | `ceil(Tkv/64)*64` | Reduction axis. This also pins `Q.K^T`'s **output** N to the same aligned width: the score matrix's N *is* `P.V`'s K, and the pad columns must be present and masked (-inf -> 0 through softmax), not skipped. |
| MLP `down` K | 3420 | **3456** | Reduction axis. The trailing 36 lanes stay zero through all 32 layers, which is what makes real-N gate/up safe. |
| Patch projection K | 1176 | **1216** | Reduction axis, IF4 and BF16 projection paths. |

Not FLOPs, but still padded **storage**: Q/K/V land in 128-lane attention slots
and gate/up rows keep the 3456 pitch. Compacting that storage is a DRAM
question (`params.bin` is 7244 of 8192 MiB), independent of this work.

Already real before this work, and unchanged: O projection contracts over
`VN*VD = 1280` (the real head width -- `head_dim` is 80, so 1280 is not a pad),
with `_vis_trim_pad_lanes` gathering 2048 -> 1280; `P.V` already computed only
its 80 real output columns; token/window ordering is already 64-aligned.

## Verified results

Measured on the board named above. Vision issued GFLOP and the padding
inventory are exact; millisecond figures carry run-to-run jitter, quantified
below.

| Config | Vision issued GFLOP | Useful | Vision FPGA ms | Vision % peak | programs.bin |
| --- | ---: | ---: | ---: | ---: | ---: |
| Padded (before) | 762.52 | 98.7% | 3322.5 | 67.2% | 142.75 MiB |
| real-N MLP only | 759.12 | 99.1% | 3312.5 | 67.1% | 142.75 MiB |
| QKV strip only | 762.52 | 98.7% | 3237.1 | 69.0% | 73.37 MiB |
| **Both (shipped)** | **759.12** | **99.1%** | **3220.7 - 3228.0** | **69.0 - 69.1%** | **73.37 MiB** |

### Issued GFLOP reduction (exact)

- Vision issued **762.52 -> 759.12 GFLOP = -3.40**, exactly the predicted
  3420-vs-3456 gate/up saving over 576 patches x 32 layers.
- Vision `Useful` (model GFLOP / issued) **98.7% -> 99.1%**.
- Whole-run issued **4819.57 -> 4816.17 GFLOP**. Model GFLOP is unchanged at
  4798.84, so total padding waste is now **17.33 GFLOP (0.36%)**, and all of it
  is the inner-K table above.
- Per-phase attribution (profile run): `o_proj+mlp` issued **550.26 -> 546.85**,
  Useful 99.0% -> 99.6%. No other phase's GFLOP moves, which is the proof that
  the saving is the MLP width and nothing else.

### Speedup (vision stage only)

- Vision **3322.5 -> 3220.7-3228.0 ms**, about **-95 ms (-2.9%)**, reproduced
  across three separate shipped-code runs.
- Vision % of peak **67.2% -> 69.0-69.1%**; `qkv_proj` alone **72.5% -> 83.2%**
  (% of peak is clock-invariant, so it is comparable to the 366.7 MHz profile
  baseline; milliseconds are not).
- **No whole-run speedup is claimed.** Total FPGA time was observed between
  18023.4 and 18205.3 ms on identical shipped code -- about +/-1%, which is
  wider than the 95 ms vision saving. The saving is real but only separable at
  the vision stage, not in the end-to-end total.
- The speedup is almost entirely the **QKV strip**, which saves *zero* FLOPs:
  it collapses 64 Q/K half-head matmuls + 16 V matmuls per layer into one
  2560-column Q+K and one 1280-column V call, i.e. 2560 fewer calls per image.
  That is a dispatch win, invisible to FLOP accounting -- the vision program
  section drops **121.39 -> 52.02 MiB** and `programs.bin` 142.75 -> 73.37 MiB,
  a 2.0x smaller program image.
- The real-N MLP contributes the smaller share (-10.0 ms), consistent with
  3.40 GFLOP at the measured vision rate.

### How this was verified, and what decode text is worth

Decode text is **not** a sound correctness criterion for this model, and it was
a mistake to treat it as one. Greedy decode is autoregressive over IF4 logits,
so one flipped near-tie rewrites every later token. The padded baseline code,
re-run unchanged hours later, produced a different caption than it had
originally -- board-state drift, no code involved. Text is only meaningful for
runs taken back to back.

What the pad-removal path was actually verified against:

- Issued GFLOP and `Useful` per stage and per phase (exact, reproducible).
- Program size: 76934752 bytes, identical before and after the dead-branch
  cleanup.
- A back-to-back A/B of the pre-cleanup and post-cleanup encoders: **identical
  full decode text** in consecutive runs, confirming the cleanup is equivalent.
- Note that `programs.bin` is **not** byte-reproducible run to run even for
  identical source, so its hash cannot be used as an equivalence check.

Caution on the committed profile baseline
(`..._image_profile_multi-core_8 baseline.md`): it was taken on build
`0xbaa4c63b` at **366.7 MHz**, so its millisecond column is not comparable to
anything here. Only `% of peak` and the GFLOP columns carry across clocks.

## Artifacts

- `qwen2.5_omni_7b_test_xdma0_image_multi-core_8.md` — shipped-code run summary.
- `qwen2.5_omni_7b_test_xdma0_image_profile_multi-core_8.md` — shipped-code
  per-phase profile (the `o_proj+mlp` and `qkv_proj` rows above).
- Run logs were kept on the verification host only; they are not committed.
