# qwen2.5_omni_7b run summary

- **Mode:** image

## Hardware

- **HW version:** 0x472c631f
- **Device:** xdma0
- **Clock:** 3.0000 ns (333.3 MHz)
- **AXI data width:** 256 bits
- **DRAM:** 8 GiB
- **Engines in use:** 8 of 8 reported
- **Peak throughput:** 341.3 GFLOPS (333.3 MHz x 128 FLOP/cycle x 8 engines)
- **Per-engine peak:** 42.7 GFLOPS

## Weights and programs

- **Weight bin:** `params.bin` — 7243.9 MiB (validated against params.json)
- **LM weight DRAM:** 0.6 MiB (IF4 projections incl. V; BF16 embedding on host)
- **Program bin:** `programs.bin` — 73.40 MiB (24 sections)
  - **decode:** 3.31 MiB across 8 engine sections (master 905.6 KiB)
  - **prefill:** 18.06 MiB across 8 engine sections (master 4485.4 KiB)
  - **vision:** 52.03 MiB across 8 engine sections (master 6849.2 KiB)

## Stage summary

FPGA time is the HW counter; CPU wall is the host-side timer around the same stage. `x 1-engine peak` is the achieved rate divided by ONE engine's peak: the effective speedup the 8-engine split delivered, against a ceiling of 8.00x. `Model GFLOP` is what the architecture owes at its own dimensions -- true prompt length, true attention windows, matrix products only; `Effective GFLOPS` divides it by the same measured FPGA time, so it is comparable to any other implementation of this model on any hardware; `Useful` is how much of the issued work the model needed.

| Stage | Shape | Work (GFLOP) | Model GFLOP | Useful | FPGA time (ms) | Throughput (GFLOPS) | Effective GFLOPS | % of peak | x 1-engine peak | CPU wall (s) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vision encoder | 576 patches -> 144 soft tokens | 759.12 | 752.37 | 99.1% | 3220.7 | 235.7 | 233.6 | 69.1% | 5.52x | 3.34 |
| Prefill | 170 tokens | 2232.59 | 2224.43 | 99.6% | 7340.8 | 304.1 | 303.0 | 89.1% | 7.13x | 7.47 |
| **TOTAL** | 8 engines | **2991.71** | **2976.80** | **99.5%** | **10561.6** | **283.3** | **281.9** | **83.0%** | **6.64x** | **10.81** |

## Vision

- **Image:** `yosemite.jpg` -> (576 patches -> 144 soft tokens) = 144 soft tokens total
- **Work:** 759.1 GFLOP
- **HW latency:** 3220.7 ms
- **Throughput:** 235.7 GFLOPS (69.1% of peak)
- **End-to-end (CPU timer):** 3.34 s

## Prefill

- **Sequence length:** 170 tokens
- **Work:** 2232.6 GFLOP
- **HW latency:** 7340.8 ms
- **Throughput:** 304.1 GFLOPS (89.1% of peak)
- **End-to-end (CPU timer):** 7.47 s
- **Embedding:** host BF16 gather and row DMA are included in CPU time, not FPGA time

## Time to first token

- **TTFT (HW counter; vision + prefill):** 10561.6 ms
- **TTFT (CPU timer; vision + prefill):** 10.81 s
  (includes prefill host embedding lookup and DMA)

## Per-phase profile

Phase latencies come from FPGA hardware counters. Vision's patch_embed is a separate program; the remaining phases are timed between per-phase HALTs and include a stop/restart per phase. They exclude host time, so the SHARE column says which phase is worth sharding next. `*` marks phases that run on engine 0 only, whose % of peak is measured against ONE engine.

### Vision encoder

576 patches -> 144 soft tokens. patch_embed is a separate FPGA program before the encoder checkpoints.

| Phase | Calls | Total ms | Share | GFLOP | Model GFLOP | Useful | GFLOPS | Effective GFLOPS | % of peak |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| patch_embed | 1 | 45.12 | 1.4% | 1.79 | 1.73 | 96.7% | 39.7 | 38.4 | 11.6% |
| qkv_proj | 32 | 638.79 | 19.8% | 181.36 | 181.19 | 99.9% | 283.9 | 283.7 | 83.2% |
| permute_qkv * | 32 | 155.30 | 4.8% | 0.00 | 0.00 | 0.0% | 0.0 | 0.0 | 0.0% |
| rope | 32 | 78.10 | 2.4% | 0.30 | 0.00 | 0.0% | 3.9 | 0.0 | 1.1% |
| attention | 32 | 149.03 | 4.6% | 15.97 | 12.08 | 75.7% | 107.1 | 81.1 | 31.4% |
| unpermute+trim * | 32 | 73.04 | 2.3% | 0.00 | 0.00 | 0.0% | 0.0 | 0.0 | 0.0% |
| o_proj+mlp | 32 | 1753.30 | 54.4% | 546.85 | 544.53 | 99.6% | 311.9 | 310.6 | 91.4% |
| merger * | 1 | 328.04 | 10.2% | 12.84 | 12.83 | 99.9% | 39.1 | 39.1 | 91.8% |
| **TOTAL** | 194 | **3220.72** | 100.0% | **759.12** | **752.37** | **99.1%** | **235.7** | **233.6** | **69.1%** |

### Prefill

170 tokens.

| Phase | Calls | Total ms | Share | GFLOP | Model GFLOP | Useful | GFLOPS | Effective GFLOPS | % of peak |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| qkv_proj | 28 | 550.82 | 7.5% | 157.31 | 157.22 | 99.9% | 285.6 | 285.4 | 83.7% |
| rope+cache | 28 | 130.10 | 1.8% | 0.08 | 0.00 | 0.0% | 0.6 | 0.0 | 0.2% |
| attention | 28 | 96.24 | 1.3% | 13.27 | 5.83 | 44.0% | 137.9 | 60.6 | 40.4% |
| attn_permute | 28 | 15.07 | 0.2% | 0.00 | 0.00 | 0.0% | 0.0 | 0.0 | 0.0% |
| o_proj | 28 | 398.16 | 5.4% | 122.28 | 122.28 | 100.0% | 307.1 | 307.1 | 90.0% |
| mlp_proj | 28 | 6150.45 | 83.8% | 1939.64 | 1939.09 | 100.0% | 315.4 | 315.3 | 92.4% |
| **TOTAL** | 168 | **7340.85** | 100.0% | **2232.59** | **2224.43** | **99.6%** | **304.1** | **303.0** | **89.1%** |

### Decode - 1st token

Context 171 tokens (aligned 192).

| Phase | Calls | Total ms | Share | GFLOP | Model GFLOP | Useful | GFLOPS | Effective GFLOPS | % of peak |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| qkv_proj | 28 | 3.80 | 6.7% | 0.93 | 0.92 | 100.0% | 243.3 | 243.2 | 71.3% |
| rope+cache | 28 | 0.82 | 1.5% | 0.00 | 0.00 | 0.0% | 0.6 | 0.0 | 0.2% |
| attention | 28 | 5.27 | 9.3% | 0.08 | 0.07 | 88.0% | 14.8 | 13.0 | 4.3% |
| attn_permute | 28 | 0.05 | 0.1% | 0.00 | 0.00 | 0.0% | 0.0 | 0.0 | 0.0% |
| o_proj | 28 | 2.66 | 4.7% | 0.72 | 0.72 | 100.0% | 269.9 | 269.9 | 79.1% |
| mlp_norm | 28 | 0.37 | 0.6% | 0.00 | 0.00 | 0.0% | 1.4 | 0.0 | 0.4% |
| mlp_gate_up | 28 | 26.15 | 46.2% | 7.61 | 7.60 | 100.0% | 290.9 | 290.8 | 85.2% |
| mlp_proj | 28 | 13.64 | 24.1% | 3.80 | 3.80 | 100.0% | 278.8 | 278.8 | 81.7% |
| final_norm | 1 | 0.01 | 0.0% | 0.00 | 0.00 | 0.0% | 1.9 | 0.0 | 0.5% |
| lm_head | 1 | 3.77 | 6.7% | 1.09 | 1.09 | 100.0% | 288.8 | 288.7 | 84.6% |
| **TOTAL** | 226 | **56.55** | 100.0% | **14.22** | **14.21** | **99.9%** | **251.5** | **251.3** | **73.7%** |

## Prompt & output

### Prompt

```
Describe the picture in detail.
```

### Decoded text

```
(none)
```
