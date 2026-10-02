# pi0.5 baseline performance (before padding removal)

Baseline for the pad-removal work on `non-aligned-start-addr-dma`: the model **as it runs today, with every padding still in place**. All numbers are from one inference of `pi05_test.py` on Alveo, 8 engines.

## Setup

| Item | Value |
|---|---|
| Host / device | p2, `--dev xdma0 --device alveo` |
| HW_INFO | `0x89016eab`: queue on, AXI 256-bit, 8 GiB DRAM, **8 cores**, 366.67 MHz (2.727 ns) |
| Peak throughput | 366.67 MHz × 128 FLOP/cycle × 8 cores = **375.5 GFLOP/s** |
| Software | unified-engine `a919c157` (`non_align_model_test`, unaligned-DMA changes imported), `myvenv`, torch 2.12 CPU |
| Engines | `--engines 8`: vision and prefix row-sharded, denoise column-sharded |
| Padding knobs | `VIS_UNPAD=split80` (P·V N=80, O-proj K=1280), `VIS_QKV_LANES=80`, `VIS_I_PAD=4352` |
| Inputs | 3 image slots (`KEEP_MASKED_SLOTS`), prefix 832 tokens (valid 813), action horizon 10 of 64 padded rows, 10 denoise steps |
| Result sanity | action chunk (10, 7) finite: `nan=False inf=False` |

```bash
cd ~/unified-engine && myvenv/bin/python3 models/pi05/pi05_test.py --dev xdma0 --device alveo --engines 8 > models/pi05/pi05_baseline_runtime.log 2>&1
```

Compile is separate: all three stages precompile in 15.0 s, not in the times below.

## Stage summary

*Effective* = FLOPs at the model's real dimensions. *Issued* = FLOPs the FPGA actually executes on the padded dimensions. Throughput columns are the model's own printout (unrounded stage times); the Total row divides by the rounded 13.0 s. % peak uses issued.

| Stage | Time (s) | Share | Effective GFLOP | Issued GFLOP | Padding overhead | Effective GFLOP/s | Issued GFLOP/s | % peak (issued) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vision (SigLIP, 3 image slots) | 2.3 | 17.7% | 655.9 | 695.8 | 6.1% | 286.2 | 303.6 | 81% |
| Prefix (PaliGemma LM, 832 tokens) | 10.0 | 76.9% | 3,399.8 | 3,399.8 | 0.0% | 340.0 | 340.0 | 91% |
| Denoise (action expert, 10 steps) | 0.7 | 5.4% | 74.7 | 75.5 | 1.1% | 104.6 | 105.8 | 28% |
| **Total** | **13.0** | 100% | **4,130.4** | **4,171.1** | **1.0%** | 317.7 | 320.9 | 85% |

## Vision (SigLIP, 3 image slots)

| Op | Real dims | Padded dims (what runs) | Effective GFLOP | Issued GFLOP | Overhead | Share of stage issued |
|---|---|---|---:|---:|---:|---:|
| Q/K/V projection (3 matmuls) | M=256 K=1152 N=3x16x72=3456 | M=256 K=1152 N=3x16x80=3840 | 165.1 | 183.5 | +11.1% | 26.4% |
| attention Q.K^T | 16 heads M=256 K=72 N=256 | 16 heads M=256 K=128 N=256 | 12.2 | 21.7 | +77.8% | 3.1% |
| attention P.V | 16 heads M=256 K=256 N=72 | 16 heads M=256 K=256 N=80 | 12.2 | 13.6 | +11.1% | 2.0% |
| O projection | M=256 K=16x72=1152 N=1152 | M=256 K=16x80=1280 N=1152 | 55.0 | 61.2 | +11.1% | 8.8% |
| MLP fc1 | M=256 K=1152 N=4304 | M=256 K=1152 N=4352 | 205.6 | 207.9 | +1.1% | 29.9% |
| MLP fc2 | M=256 K=4304 N=1152 | M=256 K=4352 N=1152 | 205.6 | 207.9 | +1.1% | 29.9% |
| **Vision total** | | | **655.9** | **695.8** | **+6.1%** | 100% |

## Prefix (PaliGemma LM, 832 tokens)

| Op | Real dims | Padded dims (what runs) | Effective GFLOP | Issued GFLOP | Overhead | Share of stage issued |
|---|---|---|---:|---:|---:|---:|
| Q projection | M=832 K=2048 N=2048 | = real | 125.6 | 125.6 | 0 | 3.7% |
| K,V projection (MQA, 1 kv head) | M=832 K=2048 N=2x256 | = real | 31.4 | 31.4 | 0 | 0.9% |
| attention Q.K^T | 8 heads M=832 K=256 N=832 | = real | 51.0 | 51.0 | 0 | 1.5% |
| attention P.V | 8 heads M=832 K=832 N=256 | = real | 51.0 | 51.0 | 0 | 1.5% |
| O projection | M=832 K=2048 N=2048 | = real | 125.6 | 125.6 | 0 | 3.7% |
| gated MLP gate+up+down | M=832 K=2048 N=16384 (x3) | = real | 3,015.1 | 3,015.1 | 0 | 88.7% |
| **Prefix total** | | | **3,399.8** | **3,399.8** | **+0.0%** | 100% |

## Denoise (action expert, 10 steps)

| Op | Real dims | Padded dims (what runs) | Effective GFLOP | Issued GFLOP | Overhead | Share of stage issued |
|---|---|---|---:|---:|---:|---:|
| Q projection | M=10 K=1024 N=2048 | = real | 7.5 | 7.5 | 0 | 10.0% |
| K,V projection | M=10 K=1024 N=2x256 | = real | 1.9 | 1.9 | 0 | 2.5% |
| O projection | M=10 K=2048 N=1024 | = real | 7.5 | 7.5 | 0 | 10.0% |
| gated MLP gate+up+down | M=10 K=1024 N=4096 (x3) | = real | 45.3 | 45.3 | 0 | 60.0% |
| attention Q.K^T + P.V | 8 heads M=10 K=256 Tkv=842 | 8 heads M=10 K=256 Tkv=896 | 12.4 | 13.2 | +6.4% | 17.5% |
| action_in + action_out | M=10 W=32 | M=10 W=64 | 0.0 | 0.1 | +100.0% | 0.1% |
| **Denoise total** | | | **74.7** | **75.5** | **+1.1%** | 100% |

## Where the padding costs FLOPs

- Vision wastes **39.9 GFLOP** (5.7% of what it issues). By op: Q/K/V projection 18.4 (80 vs 72 lanes), attention Q·Kᵀ 9.5 (head_dim 128 vs 72), O projection 6.2 (K 1280 vs 1152), fc1 and fc2 2.3 each (4352 vs 4304), P·V 1.4 (80 vs 72).
- Denoise wastes **0.82 GFLOP**: attention over Tkv 842→896 and action_in/out width 32→64. The row padding (10→64) is already trimmed away.
- Prefix has no padding: every dimension is 64-aligned, so effective = issued.
- Estimate for the planned vision change (not measured): Q/K/V, P·V, O-proj and fc1/fc2 back to real dims removes about 30.4 GFLOP, about 0.10 s at the measured 303.6 GFLOP/s, under 1% of the 13.0 s total. Attention Q·Kᵀ keeps head_dim 128 (the 9.5 GFLOP stays). Time may not scale with FLOPs, so the real gain needs a run.

## What this baseline does not cover

- **Per-op time is not measured.** The model reports time per stage only; the per-op rows are FLOP counts from the model's own formulas (`_vision_flops`, `_prefix_flops`, `_denoise_flops`). The three stage totals reproduce the log exactly (655.9 / 695.8, 3,399.8, 74.7 / 75.5 GFLOP).
- The counts cover matmul and attention FLOPs only, as the model counts them. Not included in either column: patch embedding (K 588→640, K padding stays), the multimodal projector, norms, softmax, RoPE, AdaRMS conditioning, and vision/prefix memory traffic.
- One run, one sample observation; no repeats, so no variance is given.
- The padding-removal switches (`--vis_qkv_strip`, `--vis_unpad_lanes`, `--vis_fc1_real_n`) are in `pi05_test.py` but off by default and not verified on any hardware, because the boards' current images lack the new unaligned-DMA support. Every number in this file is the unmodified (baseline) software.
- Raw log: `models/pi05/pi05_baseline_runtime.log` on p2.
