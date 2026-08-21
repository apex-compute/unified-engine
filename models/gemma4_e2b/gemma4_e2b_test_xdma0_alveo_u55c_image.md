# gemma4_e2b_test run summary

- **HW version:** 0x68f0c76c
- **--dev:** xdma0
- **--device:** alveo_u55c
- **Clock / frequency:** 3.3 ns (300.0 MHz)
- **Cores (--multi-core):** 1
- **Peak throughput:** 38.4 GFLOPS (300.0 MHz × 128 × 1 core(s))
- **DRAM read speed:** 8288.7 MB/s (1-core aggregate, private regions)

## Weights

- **Weight bin:** `params.bin` — 7142.6 MB
- **Total weight DRAM (quantized, on FPGA):** 1544.4 MB

## Program image (`programs.bin`)

- **Total programs.bin size:** 10.5 MB
- **Vision section:** 3.2 MB
- **Prefill program:** 5989.2 KB
- **Decoder program:** 1456.5 KB
- **Program image:** vision fresh, prefill/decode fresh

## Vision

- **Vision kernel:** matmatmul
- **Vision tokens (soft tokens):** 256
- **Vision FPGA run time (HW latency):** 35621.2 ms (35621190.2 us)
- **Vision reported FLOPS:** 32.3 GFLOPS
- **Vision utilization (% peak):** 84.0%
- **Vision end-to-end (CPU timer):** 35.7 s

## Prefill

- **Prefill seq_len:** 272
- **Prefill FPGA run time (HW latency):** 31479.0 ms (31478952.4 us)
- **Prefill reported FLOPS:** 33.6 GFLOPS
- **Prefill utilization (% peak):** 87.6%
- **Prefill end-to-end (CPU timer):** 31.5 s

## Decode

- **Decoded tokens:** 384 generated (sequence total 656)
- **First-token speed (peak):** 5.6 tok/s
- **Average FLOPS:** 32.5 GFLOPS
- **Decode utilization (% peak):** 84.8%
- **Decode end-to-end (CPU timer):** 72.8 s
- **Decode average speed (CPU timer):** 5.3 tok/s

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
The upper portion of the image is dominated by a bright, brilliant sky. The sun is positioned behind a cloud formation, creating a dramatic effect where intense, brilliant light bursts through breaks in the clouds, creating a strong **sunburst** effect. The light radiates outward in sharp, defined rays (crepuscular rays) that fan out across the scene. The light is intensely bright, suggesting a sunny day, with strong highlights, particularly on the foreground elements.

**Foreground:**
The immediate foreground is dominated by a dark, silhouetted landscape. There is a prominent, rocky outcrop or cliff face on the left, which is heavily textured and appears to be made of reddish-brown rock, catching the intense sunlight, which emphasizes the texture of the rock.

**Midground:**
The middle ground consists of a vast expanse of **dense forest**. The trees, mostly coniferous (pines and evergreens, appearing deep green, suggesting a dense forest. The forest stretches across the middle of the image. The trees are densely packed, creating a deep, dark canopy that contrasts sharply with the bright sky and the illuminated areas.

**Background:**
The background shows a distant horizon line where the land meets the sky. The landscape is rolling or hilly, with patches of lighter green and darker patches of trees, suggesting rolling terrain. The overall impression is one of a vast, natural landscape.

**Overall Impression:**
The photograph is highly stylized due to the dramatic lighting. The contrast between the intensely bright sky and the darker, shadowed areas, which makes the forest appear deep and rich in texture. The composition draws the eye from the dramatic sky down to the dark foreground, with the bright sunburst effect being the focal point.
```
