# pi0.5 baseline performance on Kintex7 (before padding removal)

Baseline for the pad-removal work on the 2-core Kintex7 board, **unmodified model, every padding in place**. One inference of `pi05_test.py` launched by `model_auto_test.py`. The Alveo 8-engine baseline is in `pi05_performance.md`.

## Setup

| Item | Value |
|---|---|
| Host / device | p2, `--dev xdma1 --device kintex7` |
| HW_INFO | `0x8440c653`: queue on, AXI 256-bit, 4 GiB DRAM, **2 cores**, 198.32 MHz (5.042 ns) |
| FPGA version | `0xf6ca9b81` (RTL with the unaligned-DMA changes) |
| Peak throughput | 198.32 MHz × 128 FLOP/cycle × 2 cores = **50.8 GFLOP/s** |
| Software | unified-engine `a919c157` (`non_align_model_test`, model unmodified), `myvenv`, torch 2.12 CPU |
| Engines | `--engines 2` (set by the harness from the core count): vision and prefix row-sharded, denoise column-sharded |
| Padding knobs | `VIS_UNPAD=split80` (P·V N=80, O-proj K=1280), `VIS_QKV_LANES=80`, `VIS_I_PAD=4352` |
| Inputs | 3 image slots, prefix 832 tokens, action horizon 10 of 64 padded rows, 10 denoise steps |
| Result | harness PASS, action chunk (10, 7) finite (min -1.0078, max 0.0083) |

```bash
cd ~/unified-engine && myvenv/bin/python3 -u model_auto_test.py --only pi05 --dev xdma1 --device kintex7 --verbose
```

The harness resets the board, poisons all 4 GiB of DRAM, then runs pi05; the whole call took 136.3 s. Precompile took 8.3 s and is not in the stage times.

## Stage summary

*Effective* = FLOPs at the model's real dimensions. *Issued* = FLOPs the FPGA executes on the padded dimensions. Rates are the model's own printout. **The model's printed % of peak is wrong on this board** (it hard-codes a 366.67 MHz clock, so it prints 47% / 50% / 32%); the % peak below uses the real 198.32 MHz clock and the issued rate.

| Stage | Time (s) | Share | Effective GFLOP | Issued GFLOP | Padding overhead | Effective GFLOP/s | Issued GFLOP/s | % peak (issued) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vision (SigLIP, 3 image slots) | 15.8 | 17.4% | 655.9 | 695.8 | 6.1% | 41.4 | 43.9 | 86% |
| Prefix (PaliGemma LM, 832 tokens) | 72.3 | 79.7% | 3,399.8 | 3,399.8 | 0.0% | 47.0 | 47.0 | 93% |
| Denoise (action expert, 10 steps) | 2.5 | 2.8% | 74.7 | 75.5 | 1.1% | 29.6 | 29.9 | 59% |
| **Total** | **90.7** | 100% | **4,130.4** | **4,171.1** | **1.0%** | 45.5 | 46.0 | 91% |

## Against the 8-engine Alveo baseline

| Stage | Kintex7 2 engines (s) | Alveo 8 engines (s) | Ratio |
|---|---:|---:|---:|
| Vision | 15.8 | 2.3 | 6.9× |
| Prefix | 72.3 | 10.0 | 7.2× |
| Denoise | 2.5 | 0.7 | 3.6× |
| Peak | 50.8 GFLOP/s | 375.5 GFLOP/s | 7.4× |

Vision and prefix scale close to the peak ratio; denoise (10 rows per op, latency-bound) scales much less.

## Memory

Params 1,703.9 MB after the action expert (two vision weight copies fit), program region 4.76 MB at `0xE0000000`, 1,217.8 MB of headroom below the program base. The 4 GB address map fits in the board's 4 GiB.

## Per-op breakdown

FLOP counts do not depend on the board or engine count; they are the same as in `pi05_performance.md` (computed from the model's own `_vision_flops`, `_prefix_flops`, `_denoise_flops`, and they reproduce this run's stage totals: 655.9 / 695.8, 3,399.8, 74.7 / 75.5 GFLOP).

### Vision (SigLIP, 3 image slots)

| Op | Real dims | Padded dims (what runs) | Effective GFLOP | Issued GFLOP | Overhead | Share of stage issued |
|---|---|---|---:|---:|---:|---:|
| Q/K/V projection (3 matmuls) | M=256 K=1152 N=3x16x72=3456 | M=256 K=1152 N=3x16x80=3840 | 165.1 | 183.5 | +11.1% | 26.4% |
| attention Q.K^T | 16 heads M=256 K=72 N=256 | 16 heads M=256 K=128 N=256 | 12.2 | 21.7 | +77.8% | 3.1% |
| attention P.V | 16 heads M=256 K=256 N=72 | 16 heads M=256 K=256 N=80 | 12.2 | 13.6 | +11.1% | 2.0% |
| O projection | M=256 K=16x72=1152 N=1152 | M=256 K=16x80=1280 N=1152 | 55.0 | 61.2 | +11.1% | 8.8% |
| MLP fc1 | M=256 K=1152 N=4304 | M=256 K=1152 N=4352 | 205.6 | 207.9 | +1.1% | 29.9% |
| MLP fc2 | M=256 K=4304 N=1152 | M=256 K=4352 N=1152 | 205.6 | 207.9 | +1.1% | 29.9% |
| **Total** | | | **655.9** | **695.8** | **+6.1%** | 100% |

### Prefix (PaliGemma LM, 832 tokens)

| Op | Real dims | Padded dims (what runs) | Effective GFLOP | Issued GFLOP | Overhead | Share of stage issued |
|---|---|---|---:|---:|---:|---:|
| Q projection | M=832 K=2048 N=2048 | = real | 125.6 | 125.6 | 0 | 3.7% |
| K,V projection (MQA, 1 kv head) | M=832 K=2048 N=2x256 | = real | 31.4 | 31.4 | 0 | 0.9% |
| attention Q.K^T | 8 heads M=832 K=256 N=832 | = real | 51.0 | 51.0 | 0 | 1.5% |
| attention P.V | 8 heads M=832 K=832 N=256 | = real | 51.0 | 51.0 | 0 | 1.5% |
| O projection | M=832 K=2048 N=2048 | = real | 125.6 | 125.6 | 0 | 3.7% |
| gated MLP gate+up+down | M=832 K=2048 N=16384 (x3) | = real | 3,015.1 | 3,015.1 | 0 | 88.7% |
| **Total** | | | **3,399.8** | **3,399.8** | **+0.0%** | 100% |

### Denoise (action expert, 10 steps)

| Op | Real dims | Padded dims (what runs) | Effective GFLOP | Issued GFLOP | Overhead | Share of stage issued |
|---|---|---|---:|---:|---:|---:|
| Q projection | M=10 K=1024 N=2048 | = real | 7.5 | 7.5 | 0 | 10.0% |
| K,V projection | M=10 K=1024 N=2x256 | = real | 1.9 | 1.9 | 0 | 2.5% |
| O projection | M=10 K=2048 N=1024 | = real | 7.5 | 7.5 | 0 | 10.0% |
| gated MLP gate+up+down | M=10 K=1024 N=4096 (x3) | = real | 45.3 | 45.3 | 0 | 60.0% |
| attention Q.K^T + P.V | 8 heads M=10 K=256 Tkv=842 | 8 heads M=10 K=256 Tkv=896 | 12.4 | 13.2 | +6.4% | 17.5% |
| action_in + action_out | M=10 W=32 | M=10 W=64 | 0.0 | 0.1 | +100.0% | 0.1% |
| **Total** | | | **74.7** | **75.5** | **+1.1%** | 100% |

## What this baseline does not cover

- Per-op time is not measured; the model reports stage times only. Per-op rows are FLOP counts.
- Counts cover matmul and attention FLOPs as the model counts them (not patch embedding, projector, norms, softmax, RoPE, AdaRMS).
- One run, no repeats.
- The padding-removal switches (`--vis_qkv_strip`, `--vis_unpad_lanes`, `--vis_fc1_real_n`) are in `pi05_test.py` but off by default and not verified on any hardware, because the boards' current images lack the new unaligned-DMA support. Every number in this file is the unmodified (baseline) software.
- Raw log: `models/pi05/pi05_xdma1_baseline_runtime.log` on p2.
