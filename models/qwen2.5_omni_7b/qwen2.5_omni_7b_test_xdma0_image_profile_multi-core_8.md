# qwen2.5_omni_7b run summary

- **Mode:** image

## Hardware

- **HW version:** 0x92efba46
- **Device:** xdma0
- **Clock:** 2.7273 ns (366.7 MHz)
- **AXI data width:** 256 bits
- **DRAM:** 8 GiB
- **Engines in use:** 8 of 8 reported
- **Peak throughput:** 375.5 GFLOPS (366.7 MHz x 128 FLOP/cycle x 8 engines)
- **Per-engine peak:** 46.9 GFLOPS

## Weights and programs

- **Weight bin:** `params.bin` — 5608.3 MiB (validated against params.json)
- **Program bin:** `programs.bin` — 35.31 MiB (24 sections)
  - **decode:** 3.70 MiB across 8 engine sections (master 1059.4 KiB)
  - **prefill:** 17.34 MiB across 8 engine sections (master 4329.6 KiB)
  - **vision:** 14.27 MiB across 8 engine sections (master 2015.1 KiB)

### ISA usage

```
  core 0 ISA: 5.26 / 40 MiB
  core 1 ISA peak: 2.04 / 8 MiB (decode 0.55, prefill 2.04, vision 1.76 MiB)
  core 2 ISA peak: 2.04 / 8 MiB (decode 0.55, prefill 2.04, vision 1.76 MiB)
  core 3 ISA peak: 2.04 / 8 MiB (decode 0.55, prefill 2.04, vision 1.76 MiB)
  core 4 ISA peak: 1.76 / 8 MiB (decode 0.26, prefill 1.75, vision 1.76 MiB)
  core 5 ISA peak: 1.76 / 8 MiB (decode 0.26, prefill 1.75, vision 1.76 MiB)
  core 6 ISA peak: 1.76 / 8 MiB (decode 0.26, prefill 1.75, vision 1.76 MiB)
  core 7 ISA peak: 1.76 / 8 MiB (decode 0.26, prefill 1.75, vision 1.76 MiB)
```

## Stage summary

FPGA time is the HW counter; CPU wall is the host-side timer around the same stage. `x 1-engine peak` is the achieved rate divided by ONE engine's peak: the effective speedup the 8-engine split delivered, against a ceiling of 8.00x.

| Stage | Shape | Work (GFLOP) | FPGA time (ms) | Throughput (GFLOPS) | % of peak | x 1-engine peak | CPU wall (s) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vision encoder | 576 patches -> 144 soft tokens | 943.34 | 3437.1 | 274.5 | 73.1% | 5.85x | 3.57 |
| Prefill | 170 tokens | 2521.52 | 21028.0 | 119.9 | 31.9% | 2.55x | 21.14 |
| **TOTAL** | 8 engines | **3464.86** | **24465.1** | **141.6** | **37.7%** | **3.02x** | **24.72** |

## Effective throughput (model FLOPs)

`Model GFLOP` is what the architecture owes at its own dimensions -- true prompt length, true attention windows, matrix products only. `Issued GFLOP` is what this engine billed at the shapes it actually ran. `Effective GFLOPS` divides the first by the measured FPGA time, so it is comparable to any other implementation of this model on any hardware; `Useful` is how much of the issued work the model needed.

| Stage | Model GFLOP | Issued GFLOP | Useful | FPGA time (ms) | Effective GFLOPS | % of peak |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vision encoder | 752.37 | 943.34 | 79.8% | 3437.1 | 218.9 | 58.3% |
| Prefill | 2224.43 | 2521.52 | 88.2% | 21028.0 | 105.8 | 28.2% |
| **TOTAL** | **2976.80** | **3464.86** | **85.9%** | **24465.1** | **121.7** | **32.4%** |

## Vision

- **Image:** `yosemite.jpg` -> 576 patches -> 144 soft tokens
- **Work:** 943.3 GFLOP
- **HW latency:** 3437.1 ms
- **Throughput:** 274.5 GFLOPS (73.1% of peak)
- **End-to-end (CPU timer):** 3.57 s

## Prefill

- **Sequence length:** 170 tokens
- **Work:** 2521.5 GFLOP
- **HW latency:** 21028.0 ms
- **Throughput:** 119.9 GFLOPS (31.9% of peak)
- **End-to-end (CPU timer):** 21.14 s

## Time to first token

- **TTFT (HW counter; vision + prefill):** 24465.1 ms
- **TTFT (CPU timer; vision + prefill):** 24.72 s

## Multi-core scaling

This model requires exactly 8 engines, so no 1-engine baseline can be measured for comparison.  Speedup is therefore taken against one engine's PEAK (46.9 GFLOPS), which makes it a lower bound on the sharding's true benefit: a stage that is inefficient for reasons unrelated to sharding is charged for that here as well.

| Stage | Throughput (GFLOPS) | x 1-engine peak (max 8.00x) | Implied serial fraction |
| :--- | ---: | ---: | ---: |
| Vision encoder | 274.5 | 5.85x | 5.3% |
| Prefill | 119.9 | 2.55x | 30.4% |

Phases that ran on engine 0 alone, and their share of their stage's FPGA time:

| Stage | Phase | ms | Share of stage |
| :--- | :--- | ---: | ---: |
| Vision encoder | permute_qkv | 152.12 | 4.5% |
| Vision encoder | unpermute+trim | 72.60 | 2.1% |
| Vision encoder | merger | 298.70 | 8.8% |

## Per-phase profile

Phase latencies come from the HW counter between per-phase HALTs: they exclude host time but include one stop/restart per phase, so the SHARE column is the number to act on -- it says which phase is worth sharding next. `*` marks phases that run on engine 0 only, whose % of peak is measured against ONE engine.

### Vision encoder

576 patches -> 144 soft tokens.

| Phase | Calls | Total ms | Share | GFLOP | GFLOPS | % of peak |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| qkv_proj | 32 | 831.80 | 24.5% | 290.12 | 348.8 | 92.9% |
| permute_qkv * | 32 | 152.12 | 4.5% | 0.00 | 0.0 | 0.0% |
| rope | 32 | 75.25 | 2.2% | 0.30 | 4.0 | 1.1% |
| attention | 32 | 356.02 | 10.5% | 88.03 | 247.3 | 65.9% |
| unpermute+trim * | 32 | 72.60 | 2.1% | 0.00 | 0.0 | 0.0% |
| o_proj+mlp | 32 | 1610.17 | 47.4% | 550.26 | 341.7 | 91.0% |
| merger * | 1 | 298.70 | 8.8% | 12.84 | 43.0 | 91.6% |
| **TOTAL** | 193 | **3396.65** | 100.0% | **941.55** | **277.2** | **73.8%** |

### Prefill

170 tokens.

| Phase | Calls | Total ms | Share | GFLOP | GFLOPS | % of peak |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| qkv_proj | 28 | 570.82 | 2.7% | 177.67 | 311.3 | 82.9% |
| rope+cache | 28 | 135.32 | 0.6% | 0.09 | 0.7 | 0.2% |
| attention | 28 | 94.30 | 0.4% | 14.99 | 159.0 | 42.3% |
| attn_permute | 28 | 15.56 | 0.1% | 0.00 | 0.0 | 0.0% |
| o_proj | 28 | 415.59 | 2.0% | 138.11 | 332.3 | 88.5% |
| mlp_proj | 28 | 19796.41 | 94.1% | 2190.66 | 110.7 | 29.5% |
| **TOTAL** | 168 | **21028.00** | 100.0% | **2521.52** | **119.9** | **31.9%** |

### Decode - 1st token

Context 171 tokens (aligned 192).

| Phase | Calls | Total ms | Share | GFLOP | GFLOPS | % of peak |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| qkv_proj | 28 | 4.61 | 7.3% | 0.93 | 200.8 | 53.5% |
| rope+cache | 28 | 0.77 | 1.2% | 0.00 | 0.6 | 0.2% |
| attention | 28 | 5.13 | 8.1% | 0.83 | 162.0 | 43.1% |
| attn_permute | 28 | 0.05 | 0.1% | 0.00 | 0.0 | 0.0% |
| o_proj | 28 | 10.31 | 16.2% | 0.72 | 69.8 | 18.6% |
| mlp_norm | 28 | 0.34 | 0.5% | 0.00 | 1.5 | 0.4% |
| mlp_gate_up | 28 | 24.39 | 38.4% | 7.60 | 311.9 | 83.1% |
| mlp_proj | 28 | 13.07 | 20.6% | 3.80 | 290.8 | 77.5% |
| final_norm | 1 | 0.01 | 0.0% | 0.00 | 2.0 | 0.5% |
| lm_head | 1 | 4.81 | 7.6% | 1.11 | 230.7 | 61.5% |
| **TOTAL** | 226 | **63.49** | 100.0% | **14.99** | **236.2** | **62.9%** |

### Decode - at context

Context 2048 tokens (aligned 2048).

| Phase | Calls | Total ms | Share | GFLOP | GFLOPS | % of peak |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| qkv_proj | 28 | 4.61 | 4.8% | 0.93 | 200.6 | 53.4% |
| rope+cache | 28 | 0.77 | 0.8% | 0.00 | 0.6 | 0.2% |
| attention | 28 | 38.69 | 39.9% | 0.83 | 21.5 | 5.7% |
| attn_permute | 28 | 0.05 | 0.1% | 0.00 | 0.0 | 0.0% |
| o_proj | 28 | 10.31 | 10.6% | 0.72 | 69.8 | 18.6% |
| mlp_norm | 28 | 0.34 | 0.4% | 0.00 | 1.5 | 0.4% |
| mlp_gate_up | 28 | 24.39 | 25.1% | 7.60 | 311.9 | 83.1% |
| mlp_proj | 28 | 13.08 | 13.5% | 3.80 | 290.7 | 77.4% |
| final_norm | 1 | 0.01 | 0.0% | 0.00 | 2.0 | 0.5% |
| lm_head | 1 | 4.81 | 5.0% | 1.11 | 230.8 | 61.5% |
| **TOTAL** | 226 | **97.05** | 100.0% | **14.99** | **154.5** | **41.1%** |

## Prompt & output

### Prompt

```
Describe the picture in detail.
```

### Decoded text

```
(none)
```
