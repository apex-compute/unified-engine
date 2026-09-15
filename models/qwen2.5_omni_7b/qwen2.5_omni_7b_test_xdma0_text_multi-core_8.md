# qwen2.5_omni_7b run summary

- **Mode:** text

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
- **Program bin:** `programs.bin` — 20.69 MiB (16 sections)
  - **decode:** 3.69 MiB across 8 engine sections (master 1047.1 KiB)
  - **prefill:** 17.01 MiB across 8 engine sections (master 4278.9 KiB)

### ISA usage

```
  core 0 ISA: 5.20 / 40 MiB
  core 1 ISA peak: 2.00 / 8 MiB (decode 0.55, prefill 2.00 MiB)
  core 2 ISA peak: 2.00 / 8 MiB (decode 0.55, prefill 2.00 MiB)
  core 3 ISA peak: 2.00 / 8 MiB (decode 0.55, prefill 2.00 MiB)
  core 4 ISA peak: 1.71 / 8 MiB (decode 0.26, prefill 1.71 MiB)
  core 5 ISA peak: 1.71 / 8 MiB (decode 0.26, prefill 1.71 MiB)
  core 6 ISA peak: 1.71 / 8 MiB (decode 0.26, prefill 1.71 MiB)
  core 7 ISA peak: 1.71 / 8 MiB (decode 0.26, prefill 1.71 MiB)
```

## Stage summary

FPGA time is the HW counter; CPU wall is the host-side timer around the same stage. `x 1-engine peak` is the achieved rate divided by ONE engine's peak: the effective speedup the 8-engine split delivered, against a ceiling of 8.00x.

| Stage | Shape | Work (GFLOP) | FPGA time (ms) | Throughput (GFLOPS) | % of peak | x 1-engine peak | CPU wall (s) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill | 29 tokens | 837.18 | 7399.1 | 113.1 | 30.1% | 2.41x | 7.40 |
| Decode | 42 steps, 41 tokens kept | 596.07 | 2563.7 | 232.5 | 61.9% | 4.95x | 2.62 |
| **TOTAL** | 8 engines | **1433.25** | **9962.8** | **143.9** | **38.3%** | **3.07x** | **10.02** |

## Effective throughput (model FLOPs)

`Model GFLOP` is what the architecture owes at its own dimensions -- true prompt length, true attention windows, matrix products only. `Issued GFLOP` is what this engine billed at the shapes it actually ran. `Effective GFLOPS` divides the first by the measured FPGA time, so it is comparable to any other implementation of this model on any hardware; `Useful` is how much of the issued work the model needed.

| Stage | Model GFLOP | Issued GFLOP | Useful | FPGA time (ms) | Effective GFLOPS | % of peak |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill | 378.64 | 837.18 | 45.2% | 7399.1 | 51.2 | 13.6% |
| Decode | 594.76 | 596.07 | 99.8% | 2563.7 | 232.0 | 61.8% |
| **TOTAL** | **973.40** | **1433.25** | **67.9%** | **9962.8** | **97.7** | **26.0%** |

## Prefill

- **Sequence length:** 29 tokens
- **Work:** 837.2 GFLOP
- **HW latency:** 7399.1 ms
- **Throughput:** 113.1 GFLOPS (30.1% of peak)
- **End-to-end (CPU timer):** 7.40 s

## Time to first token

- **TTFT (HW counter; prefill):** 7399.1 ms
- **TTFT (CPU timer; prefill):** 7.40 s

## Decode

- **Steps:** 42 (kept 41 tokens, sequence total 71)
- **First-token speed (HW counter):** 16.43 tok/s (60.8 ms)
- **Average speed (HW counter):** 16.38 tok/s (61.0 ms/token)
- **Average speed (CPU timer):** 16.00 tok/s (62.5 ms/token)
- **Host overhead:** 2.3% of wall time outside the engines
- **Work per token:** 14.19 GFLOP
- **Throughput:** 232.5 GFLOPS (61.9% of peak)
- **End-to-end (CPU timer):** 2.62 s

## Multi-core scaling

This model requires exactly 8 engines, so no 1-engine baseline can be measured for comparison.  Speedup is therefore taken against one engine's PEAK (46.9 GFLOPS), which makes it a lower bound on the sharding's true benefit: a stage that is inefficient for reasons unrelated to sharding is charged for that here as well.

| Stage | Throughput (GFLOPS) | x 1-engine peak (max 8.00x) | Implied serial fraction |
| :--- | ---: | ---: | ---: |
| Prefill | 113.1 | 2.41x | 33.1% |
| Decode | 232.5 | 4.95x | 8.8% |

Run with `--profile` to attribute that serial fraction to named phases.

## Prompt & output

### Prompt

```
Explain why the sky is blue in one sentence.
```

### Decoded text

```
The blue color of the sky is caused by the scattering of sunlight by the Earth's atmosphere, which causes shorter wavelengths of light, such as blue, to scatter more than longer wavelengths, such as red.
```
