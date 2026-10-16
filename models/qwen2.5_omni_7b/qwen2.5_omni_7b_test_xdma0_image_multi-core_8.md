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
- **Program bin:** `programs.bin` — 73.37 MiB (24 sections)
  - **decode:** 3.30 MiB across 8 engine sections (master 893.2 KiB)
  - **prefill:** 18.05 MiB across 8 engine sections (master 4476.6 KiB)
  - **vision:** 52.02 MiB across 8 engine sections (master 6839.1 KiB)

## Stage summary

FPGA time is the HW counter; CPU wall is the host-side timer around the same stage. `x 1-engine peak` is the achieved rate divided by ONE engine's peak: the effective speedup the 8-engine split delivered, against a ceiling of 8.00x. `Model GFLOP` is what the architecture owes at its own dimensions -- true prompt length, true attention windows, matrix products only; `Effective GFLOPS` divides it by the same measured FPGA time, so it is comparable to any other implementation of this model on any hardware; `Useful` is how much of the issued work the model needed.

| Stage | Shape | Work (GFLOP) | Model GFLOP | Useful | FPGA time (ms) | Throughput (GFLOPS) | Effective GFLOPS | % of peak | x 1-engine peak | CPU wall (s) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vision encoder | 576 patches -> 144 soft tokens | 759.12 | 752.37 | 99.1% | 3221.1 | 235.7 | 233.6 | 69.0% | 5.52x | 3.22 |
| Prefill | 170 tokens | 2232.59 | 2224.43 | 99.6% | 7340.7 | 304.1 | 303.0 | 89.1% | 7.13x | 7.36 |
| Decode | 128 steps, 128 tokens kept | 1824.46 | 1822.04 | 99.9% | 7643.5 | 238.7 | 238.4 | 69.9% | 5.59x | 8.43 |
| **TOTAL** | 8 engines | **4816.17** | **4798.84** | **99.6%** | **18205.3** | **264.5** | **263.6** | **77.5%** | **6.20x** | **19.01** |

## Vision

- **Image:** `yosemite.jpg` -> (576 patches -> 144 soft tokens) = 144 soft tokens total
- **Work:** 759.1 GFLOP
- **HW latency:** 3221.1 ms
- **Throughput:** 235.7 GFLOPS (69.0% of peak)
- **End-to-end (CPU timer):** 3.22 s

## Prefill

- **Sequence length:** 170 tokens
- **Work:** 2232.6 GFLOP
- **HW latency:** 7340.7 ms
- **Throughput:** 304.1 GFLOPS (89.1% of peak)
- **End-to-end (CPU timer):** 7.36 s
- **Embedding:** host BF16 gather and row DMA are included in CPU time, not FPGA time

## Time to first token

- **TTFT (HW counter; vision + prefill):** 10561.8 ms
- **TTFT (CPU timer; vision + prefill):** 10.58 s
  (includes prefill host embedding lookup and DMA)

## Decode

- **Steps:** 128 (kept 128 tokens, sequence total 298)
- **First-token speed (HW counter):** 17.37 tok/s (57.6 ms)
- **Average speed (HW counter):** 16.75 tok/s (59.7 ms/token)
- **Average speed (CPU timer):** 15.18 tok/s (65.9 ms/token)
- **Host overhead:** 9.4% of wall time outside the engines
- **Work per token:** 14.25 GFLOP
- **Throughput:** 238.7 GFLOPS (69.9% of peak)
- **End-to-end (CPU timer):** 8.43 s

## Prompt & output

### Prompt

```
Describe the picture in detail.
```

### Decoded text

```
The image depicts a stunning landscape scene, likely taken in a mountainous region. The sun is positioned on the left side of the image, casting a warm, golden light that creates a starburst effect as it shines through the trees. The sky is a clear blue with a few scattered clouds, adding to the serene atmosphere.

In the foreground, there is a dense forest of evergreen trees, their green hues contrasting beautifully with the blue sky. The trees are lush and vibrant, indicating a healthy, thriving ecosystem. 

The middle ground features a valley with a river or stream running through it, though it is not clearly visible. The valley
```
