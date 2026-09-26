# qwen2.5_omni_7b run summary

- **Mode:** image

## Hardware

- **HW version:** 0xfc46ae6f
- **Device:** xdma0
- **Clock:** 2.7273 ns (366.7 MHz)
- **AXI data width:** 256 bits
- **DRAM:** 8 GiB
- **Engines in use:** 8 of 8 reported
- **Peak throughput:** 375.5 GFLOPS (366.7 MHz x 128 FLOP/cycle x 8 engines)
- **Per-engine peak:** 46.9 GFLOPS

## Weights and programs

- **Weight bin:** `params.bin` — 5989.5 MiB (validated against params.json)
- **LM weight DRAM:** 0.6 MiB (IF4 projections incl. V; BF16 embedding on host)
- **Program bin:** `programs.bin` — 142.39 MiB (24 sections)
  - **decode:** 3.31 MiB across 8 engine sections (master 903.8 KiB)
  - **prefill:** 17.95 MiB across 8 engine sections (master 4473.1 KiB)
  - **vision:** 121.13 MiB across 8 engine sections (master 15695.2 KiB)

## Stage summary

FPGA time is the HW counter; CPU wall is the host-side timer around the same stage. `x 1-engine peak` is the achieved rate divided by ONE engine's peak: the effective speedup the 8-engine split delivered, against a ceiling of 8.00x. `Model GFLOP` is what the architecture owes at its own dimensions -- true prompt length, true attention windows, matrix products only; `Effective GFLOPS` divides it by the same measured FPGA time, so it is comparable to any other implementation of this model on any hardware; `Useful` is how much of the issued work the model needed.

| Stage | Shape | Work (GFLOP) | Model GFLOP | Useful | FPGA time (ms) | Throughput (GFLOPS) | Effective GFLOPS | % of peak | x 1-engine peak | CPU wall (s) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vision encoder | 576 patches -> 144 soft tokens | 762.52 | 752.37 | 98.7% | 3008.9 | 253.4 | 250.0 | 67.5% | 5.40x | 3.11 |
| Prefill | 170 tokens | 2232.59 | 2224.43 | 99.6% | 6598.4 | 338.4 | 337.1 | 90.1% | 7.21x | 6.70 |
| **TOTAL** | 8 engines | **2995.11** | **2976.80** | **99.4%** | **9607.3** | **311.8** | **309.8** | **83.0%** | **6.64x** | **9.81** |

## Vision

- **Image:** `yosemite.jpg` -> 576 patches -> 144 soft tokens
- **Work:** 762.5 GFLOP
- **HW latency:** 3008.9 ms
- **Throughput:** 253.4 GFLOPS (67.5% of peak)
- **End-to-end (CPU timer):** 3.11 s

## Prefill

- **Sequence length:** 170 tokens
- **Work:** 2232.6 GFLOP
- **HW latency:** 6598.4 ms
- **Throughput:** 338.4 GFLOPS (90.1% of peak)
- **End-to-end (CPU timer):** 6.70 s
- **Embedding:** host BF16 gather and row DMA are included in CPU time, not FPGA time

## Time to first token

- **TTFT (HW counter; vision + prefill):** 9607.3 ms
- **TTFT (CPU timer; vision + prefill):** 9.81 s
  (includes prefill host embedding lookup and DMA)

## Per-phase profile

Phase latencies come from FPGA hardware counters. Vision's patch_embed is a separate program; the remaining phases are timed between per-phase HALTs and include a stop/restart per phase. They exclude host time, so the SHARE column says which phase is worth sharding next. `*` marks phases that run on engine 0 only, whose % of peak is measured against ONE engine.

### Vision encoder

576 patches -> 144 soft tokens. patch_embed is a separate FPGA program before the encoder checkpoints.

| Phase | Calls | Total ms | Share | GFLOP | Model GFLOP | Useful | GFLOPS | Effective GFLOPS | % of peak |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| patch_embed | 1 | 40.41 | 1.3% | 1.79 | 1.73 | 96.7% | 44.4 | 42.9 | 11.8% |
| qkv_proj | 32 | 666.11 | 22.1% | 181.36 | 181.19 | 99.9% | 272.3 | 272.0 | 72.5% |
| permute_qkv * | 32 | 149.00 | 5.0% | 0.00 | 0.00 | 0.0% | 0.0 | 0.0 | 0.0% |
| rope | 32 | 70.95 | 2.4% | 0.30 | 0.00 | 0.0% | 4.3 | 0.0 | 1.1% |
| attention | 32 | 131.79 | 4.4% | 15.97 | 12.08 | 75.7% | 121.2 | 91.7 | 32.3% |
| unpermute+trim * | 32 | 67.71 | 2.3% | 0.00 | 0.00 | 0.0% | 0.0 | 0.0 | 0.0% |
| o_proj+mlp | 32 | 1584.46 | 52.7% | 550.26 | 544.53 | 99.0% | 347.3 | 343.7 | 92.5% |
| merger * | 1 | 298.44 | 9.9% | 12.84 | 12.83 | 99.9% | 43.0 | 43.0 | 91.7% |
| **TOTAL** | 194 | **3008.87** | 100.0% | **762.52** | **752.37** | **98.7%** | **253.4** | **250.0** | **67.5%** |

### Prefill

170 tokens.

| Phase | Calls | Total ms | Share | GFLOP | Model GFLOP | Useful | GFLOPS | Effective GFLOPS | % of peak |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| qkv_proj | 28 | 494.96 | 7.5% | 157.31 | 157.22 | 99.9% | 317.8 | 317.6 | 84.6% |
| rope+cache | 28 | 119.78 | 1.8% | 0.08 | 0.00 | 0.0% | 0.7 | 0.0 | 0.2% |
| attention | 28 | 83.54 | 1.3% | 13.27 | 5.83 | 44.0% | 158.9 | 69.8 | 42.3% |
| attn_permute | 28 | 13.76 | 0.2% | 0.00 | 0.00 | 0.0% | 0.0 | 0.0 | 0.0% |
| o_proj | 28 | 357.27 | 5.4% | 122.28 | 122.28 | 100.0% | 342.3 | 342.3 | 91.2% |
| mlp_proj | 28 | 5529.07 | 83.8% | 1939.64 | 1939.09 | 100.0% | 350.8 | 350.7 | 93.4% |
| **TOTAL** | 168 | **6598.39** | 100.0% | **2232.59** | **2224.43** | **99.6%** | **338.4** | **337.1** | **90.1%** |

### Decode - 1st token

Context 171 tokens (aligned 192).

| Phase | Calls | Total ms | Share | GFLOP | Model GFLOP | Useful | GFLOPS | Effective GFLOPS | % of peak |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| qkv_proj | 28 | 3.48 | 6.6% | 0.93 | 0.92 | 100.0% | 265.7 | 265.6 | 70.8% |
| rope+cache | 28 | 0.76 | 1.5% | 0.00 | 0.00 | 0.0% | 0.6 | 0.0 | 0.2% |
| attention | 28 | 4.75 | 9.0% | 0.08 | 0.07 | 88.0% | 16.4 | 14.4 | 4.4% |
| attn_permute | 28 | 0.05 | 0.1% | 0.00 | 0.00 | 0.0% | 0.0 | 0.0 | 0.0% |
| o_proj | 28 | 2.49 | 4.7% | 0.72 | 0.72 | 100.0% | 289.0 | 289.0 | 77.0% |
| mlp_norm | 28 | 0.34 | 0.6% | 0.00 | 0.00 | 0.0% | 1.5 | 0.0 | 0.4% |
| mlp_gate_up | 28 | 24.33 | 46.3% | 7.60 | 7.60 | 100.0% | 312.5 | 312.5 | 83.2% |
| mlp_proj | 28 | 12.81 | 24.4% | 3.80 | 3.80 | 100.0% | 296.9 | 296.9 | 79.1% |
| final_norm | 1 | 0.01 | 0.0% | 0.00 | 0.00 | 0.0% | 2.0 | 0.0 | 0.5% |
| lm_head | 1 | 3.48 | 6.6% | 1.09 | 1.09 | 100.0% | 312.8 | 312.8 | 83.3% |
| **TOTAL** | 226 | **52.51** | 100.0% | **14.22** | **14.21** | **99.9%** | **270.8** | **270.6** | **72.1%** |

## Prompt & output

### Prompt

```
Describe the picture in detail.
```

### Decoded text

```
(none)
```
