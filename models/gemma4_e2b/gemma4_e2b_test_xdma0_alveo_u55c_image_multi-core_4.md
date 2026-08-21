# gemma4_e2b_test run summary

- **HW version:** 0x68f0c76c
- **--dev:** xdma0
- **--device:** alveo_u55c
- **Clock / frequency:** 3.3 ns (300.0 MHz)
- **Cores (--multi-core):** 4
- **Peak throughput:** 153.6 GFLOPS (300.0 MHz × 128 × 4 core(s))
- **DRAM read speed:** 14414.1 MB/s (4-core aggregate, private regions)

## Weights

- **Weight bin:** `params.bin` — 7142.6 MB
- **Total weight DRAM (quantized, on FPGA):** 1544.4 MB

## Program image (`programs.bin`)

- **Total programs.bin size:** 19.2 MB
- **Vision section:** 1.9 MB
- **Prefill program:** 4156.1 KB
- **Decoder program:** 1456.5 KB
- **Program image:** vision fresh, prefill/decode fresh

## Vision

- **Vision kernel:** matmatmul
- **Vision tokens (soft tokens):** 256
- **Vision FPGA run time (HW latency):** 9993.7 ms (9993728.0 us)
- **Vision reported FLOPS:** 115.0 GFLOPS
- **Vision utilization (% peak):** 74.9%
- **Vision end-to-end (CPU timer):** 10.1 s

## Prefill

- **Prefill seq_len:** 272
- **Prefill FPGA run time (HW latency):** 8461.2 ms (8461235.1 us)
- **Prefill reported FLOPS:** 125.1 GFLOPS
- **Prefill utilization (% peak):** 81.5%
- **Prefill end-to-end (CPU timer):** 8.5 s

## Decode

- **Decoded tokens:** 427 generated (sequence total 699)
- **First-token speed (peak):** 5.8 tok/s
- **Average FLOPS:** 33.9 GFLOPS
- **Decode utilization (% peak):** 22.1%
- **Decode end-to-end (CPU timer):** 77.8 s
- **Decode average speed (CPU timer):** 5.5 tok/s

## Prompt & output

### Full prefill prompt

```
<bos><|turn>user
<|image><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><image|>Describe this image in detail.<turn|>
<|turn>model

```

### Decoded text

```
This is a stunning, high-contrast photograph featuring a classic **vanished or "sunburst" style image** with a dramatic sky.

**Sky and Light:**
The upper portion of the image is dominated by a bright, brilliant sky. The sun is positioned centrally, creating a powerful **sunburst effect**. Intense, brilliant light radiates outward from the sun, casting strong highlights, and the light is intensely bright, suggesting a clear day. The light is highly reflective, creating a strong lens flare and lens flare that streaks across the scene, with distinct, sharp rays emanating from the sun, characteristic of a "sunburst" effect, with rays radiating in all directions.

**Foreground/Foreground elements are heavily stylized with a distinct, almost digital or digital-looking overlay. There is a distinct, almost digital or digital-looking overlay that overlays the landscape, giving the scene a slightly surreal or stylized appearance. The overall color palette is vibrant, with deep greens in the trees and warm tones, especially in the lower portion of the image.**

**Landscape:**
The landscape consists of rolling hills and valleys.
*   **Hills and terrain** that appear to be covered in varied vegetation.
*   The terrain is rugged and undulating.
*   The land is covered in dense forest, suggesting a temperate or perhaps a mix of coniferous and deciduous trees.
*   The middle ground is filled with a dense expanse of dark green trees, indicating a lush, forested area.
*   The terrain is rolling, with valleys and ridges.
*   In the distance, the landscape stretches out toward a hazy horizon.

**Atmosphere:**
The atmosphere is bright, emphasizing the contrast between the bright sun and deep shadows in the valleys and shadowed areas, highlighting the texture of the landscape.

**Overall Impression:**
The image is dramatic due to the intense light from the sunburst effect in the sky, contrasting with the darker, shadowed areas in the valleys, while the forested areas in the middle ground. The composition draws the eye from the bright sky down into the dark, textured landscape below.
```
