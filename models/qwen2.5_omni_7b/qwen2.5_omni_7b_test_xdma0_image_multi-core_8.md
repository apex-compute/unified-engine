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
- **Program bin:** `programs.bin` — 35.28 MiB (24 sections)
  - **decode:** 3.69 MiB across 8 engine sections (master 1047.1 KiB)
  - **prefill:** 17.33 MiB across 8 engine sections (master 4320.9 KiB)
  - **vision:** 14.26 MiB across 8 engine sections (master 2007.0 KiB)

### ISA usage

```
  core 0 ISA: 5.24 / 40 MiB
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
| Vision encoder | 576 patches -> 144 soft tokens | 943.34 | 3434.0 | 274.7 | 73.2% | 5.85x | 3.43 |
| Prefill | 170 tokens | 2521.52 | 21025.7 | 119.9 | 31.9% | 2.56x | 21.03 |
| Decode | 128 steps, 128 tokens kept | 1826.55 | 8239.0 | 221.7 | 59.0% | 4.72x | 8.43 |
| **TOTAL** | 8 engines | **5291.41** | **32698.7** | **161.8** | **43.1%** | **3.45x** | **32.89** |

## Effective throughput (model FLOPs)

`Model GFLOP` is what the architecture owes at its own dimensions -- true prompt length, true attention windows, matrix products only. `Issued GFLOP` is what this engine billed at the shapes it actually ran. `Effective GFLOPS` divides the first by the measured FPGA time, so it is comparable to any other implementation of this model on any hardware; `Useful` is how much of the issued work the model needed.

| Stage | Model GFLOP | Issued GFLOP | Useful | FPGA time (ms) | Effective GFLOPS | % of peak |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vision encoder | 752.37 | 943.34 | 79.8% | 3434.0 | 219.1 | 58.4% |
| Prefill | 2224.43 | 2521.52 | 88.2% | 21025.7 | 105.8 | 28.2% |
| Decode | 1822.04 | 1826.55 | 99.8% | 8239.0 | 221.1 | 58.9% |
| **TOTAL** | **4798.84** | **5291.41** | **90.7%** | **32698.7** | **146.8** | **39.1%** |

## Vision

- **Image:** `yosemite.jpg` -> 576 patches -> 144 soft tokens
- **Work:** 943.3 GFLOP
- **HW latency:** 3434.0 ms
- **Throughput:** 274.7 GFLOPS (73.2% of peak)
- **End-to-end (CPU timer):** 3.43 s

## Prefill

- **Sequence length:** 170 tokens
- **Work:** 2521.5 GFLOP
- **HW latency:** 21025.7 ms
- **Throughput:** 119.9 GFLOPS (31.9% of peak)
- **End-to-end (CPU timer):** 21.03 s

## Time to first token

- **TTFT (HW counter; vision + prefill):** 24459.7 ms
- **TTFT (CPU timer; vision + prefill):** 24.46 s

## Decode

- **Steps:** 128 (kept 128 tokens, sequence total 298)
- **First-token speed (HW counter):** 15.85 tok/s (63.1 ms)
- **Average speed (HW counter):** 15.54 tok/s (64.4 ms/token)
- **Average speed (CPU timer):** 15.19 tok/s (65.8 ms/token)
- **Host overhead:** 2.2% of wall time outside the engines
- **Work per token:** 14.27 GFLOP
- **Throughput:** 221.7 GFLOPS (59.0% of peak)
- **End-to-end (CPU timer):** 8.43 s

## Multi-core scaling

This model requires exactly 8 engines, so no 1-engine baseline can be measured for comparison.  Speedup is therefore taken against one engine's PEAK (46.9 GFLOPS), which makes it a lower bound on the sharding's true benefit: a stage that is inefficient for reasons unrelated to sharding is charged for that here as well.

| Stage | Throughput (GFLOPS) | x 1-engine peak (max 8.00x) | Implied serial fraction |
| :--- | ---: | ---: | ---: |
| Vision encoder | 274.7 | 5.85x | 5.2% |
| Prefill | 119.9 | 2.56x | 30.4% |
| Decode | 221.7 | 4.72x | 9.9% |

Run with `--profile` to attribute that serial fraction to named phases.

## Prompt & output

### Prompt

```
Describe the picture in detail.
```

### Decoded text

```
The image depicts a breathtaking landscape scene, likely taken in a national park or a similar natural setting. The focal point of the image is the sun, which is positioned on the left side of the frame, casting a warm, golden light across the scene. The sun's rays create a starburst effect, adding a dramatic and picturesque quality to the image. 

The landscape features a range of mountains or rock formations that dominate the background. These formations are covered with greenery, suggesting a healthy, thriving ecosystem. The foreground is filled with a dense forest of evergreen trees, their vibrant green hues contrasting beautifully with the blue sky above. The
```
