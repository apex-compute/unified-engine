# Gemma 4 E2B on the Apex Compute Unified Engine — tiered benchmark

Low, Medium and High are eight-engine runs on the Alveo U50, driven by
`benchmark.py`. All FPGA times come from hardware counters; wall-clock times
are labelled as such. The tiers are the qwen2.5_omni_7b tiers (same frame
counts and total prompt lengths, audio in the same ranges), so the two models
can be compared tier by tier.

## Platform

| Board | Device | Engines | Clock | Eight-engine peak | HW version | Model weight bin |
| :--- | :--- | ---: | ---: | ---: | :--- | ---: |
| Alveo U50, 8 GiB HBM | `xdma0` | 8 of 8 | 366.7 MHz | 375.5 GFLOPS | `0x25511139` (flash image) | 7078.6 MB |

## Workloads

| Tier | Command | Camera frames | Spoken query (audio encoder) | Total prompt | Of which text | Decode budget |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| Low | `benchmark.py --low` | 0 | 2.55 s, 64 soft tokens | 850 | ~786 | 32 |
| Medium | `benchmark.py --medium` | 1 × 896×896, 256 soft tokens | 4.5 s, 113 soft tokens | 2048 | ~1679 | 64 |
| High | `benchmark.py --high` | 4 × 896×896, 1024 soft tokens | 6.0 s, 150 soft tokens | 6144 | ~4970 | 128 |

- The **total prompt** (media included) is the fixed part of the shape: the
  text is instruction filler grown until the prompt is exactly that long
  (`--target-prefill-tokens`). Gemma 4 E2B encodes an 896×896 frame into 256
  soft tokens (Qwen2.5-Omni: ~1024), so text makes up more of each prompt.
- The prompts ask for more than the decode budget, so decode runs to the cap
  (`--max-new-tokens`) rather than ending early on the end-of-turn token.
- Each frame is its own image block and its own FPGA vision-encoder run.

## Context: 8192 tokens

`max_context_size` is 8192 (was 4096). Prompts longer than the 512-row prefill
program are prefilled in chunks of up to 512 tokens. Each chunk is compiled
with its position offset (RoPE rows, K/V-cache writes) and attends the whole
cache prefix that the earlier chunks wrote.

| Check | Result |
| :--- | :--- |
| Tensor arena at 8192 context | 276.5 MB high-water of 480 MB (the same as 4096 before; attention buffers are sized per role, not as context²) |
| Decode step at position 8191 (`--profile --profile-context 8192`) | 7.39 ms per step vs 4.58 ms at position 18; attention 2.83 ms of it |
| 491-token prompt, one pass vs 2 chunks | 128 greedy tokens byte-identical |
| 2000-token prompt (4 chunks), fact placed at the start | Recalled correctly, coherent summary |

## Results

| Metric | Low | Medium | High |
| :--- | ---: | ---: | ---: |
| Vision encoder, HW / wall (s) | — | 4.72 / 4.77 | 19.19 / 19.42 (4 frames) |
| Audio encoder, HW / wall (s) | 2.80 / 2.95 | 5.43 / 5.71 | 8.01 / 8.45 |
| Prefill chunks | 2 | 4 | 12 |
| Prefill, HW / wall (s) | 12.32 / 12.59 | 30.97 / 31.47 | 106.02 / —¹ |
| Prefill speed, HW (tok/s) | 68.9 | 66.1 | 57.9 |
| Prefill model GFLOPS (% peak) | 267.7 (71.3%) | 262.6 (69.9%) | 244.4 (65.1%) |
| **TTFT, HW counters (s)** | **15.12** | **41.12** | **133.23** |
| TTFT, wall clock (s) | 15.54 | 41.95 | —¹ |
| Decoded tokens (budget) | 32 (32) | 64 (64) | 128 (128) |
| Decode first token, HW (tok/s) | 26.5 | 20.1 | 11.3 |
| Decode average, HW (tok/s) | 26.3 | 19.5 | 11.2 |
| Decode average, wall clock (tok/s) | 20.1 | 16.0 | 7.5 |
| Decode model GFLOPS (% peak) | 125.6 (33.4%) | 96.1 (25.6%) | 60.3 (16.0%) |
| Final context (tokens) | 881 | 2111 | 6271 |
| Die temperature, start → peak | 84.2 → 86.7 °C | 80.7 → 86.7 °C | 59.8 → 68.3 °C¹ |

¹ High is measured with cooling pauses (see [High and the board's
temperature](#high-and-the-boards-temperature)): its 12 prefill chunks run in
four groups of three, and the run waits for the die to cool to 60 °C before
each group. The HW times exclude the pauses; the wall-clock prefill and TTFT
include them (1610 s and 1638 s) and are not meaningful.

Outputs (greedy, cut at the decode budget):

- **Low:** "**Transcription:** Unified engine. Apex computes. **Explanation:**
  The audio clip contains two distinct phrases: "Unified engine" and "Apex
  computes."" (the clip's words are "Unified Engine, Apex Computes").
- **Medium:** "**Transcription:** Unified Engine. Apex Computes custom ML
  inference accelerator. **Image Description:** The image shows a landscape
  with a sun setting over a mountainous terrain. The sky has clouds and bright
  light emanating from the sun in the upper left. The terrain features steep
  mountains on t…" (the Yosemite sample image).
- **High:** "**Audio Transcription:** Unified Engine. Apex Computes custom ML
  inference accelerator. **Camera Frames:** * **Frame 1:** The scene shows a
  landscape with mountains, a valley, a forest, and a sun setting over a valley
  with sun rays. * The lighting is bright with a sun low in the sky casting
  rays across the sky. * The terrain includes mountains, a valley, and a dense
  forest. …"

### Wall clock against the HW counters

Compilation is a one-time cost and is outside every timer: all program images,
including every prefill chunk's, are compiled before the run and held in host
memory. What the wall clock adds on top of the hardware counters is runtime
host work only:

| Stage | Low gap | Medium gap | High gap | What the host does |
| :--- | ---: | ---: | ---: | :--- |
| Vision | — | 0.05 s | 0.22 s | read back the 256 soft tokens per frame |
| Audio | 0.14 s | 0.28 s | 0.35 s | upload mel features and scratch, DMA the program, read back the soft tokens |
| Prefill | 0.27 s | 0.50 s | 1.34 s | first chunk: zero the K/V cache, attention scratch and bias planes; per chunk: embedding and per-layer-embedding lookup and upload, the two bias planes, DMA of the chunk's LM and worker images (2 / 6 / 20 ms in total), worker start and drain |
| **TTFT** | **0.42 s** | **0.83 s** | **1.90 s** | sum of the above |
| Decode, per token | 11.7 ms | 11.4 ms | 29.3 ms | embedding lookups and uploads, two bias-row uploads (they grow with the context), worker priming and drain, the 8-candidate argmax readback, detokenization |

### High and the board's temperature

High is the first tier long enough to heat this card. Run continuously, it
passes only from a cool start; with cooling pauses between prefill chunk
groups it passes and peaks at 68 °C:

| Run | Die at start | Peak | Result |
| :--- | ---: | ---: | :--- |
| High, cooling to 60 °C before chunks 4, 7 and 10 | 59.8 °C | 68.3 °C | passed (the numbers above); pauses 382, 528 and 592 s |
| High, continuous, after an hour idle | 70.8 °C | 92.7 °C | passed: prefill 108.94 s HW, TTFT 135.86 s HW, same 128 tokens |
| High, 15 min after that run | 78.2 °C | 94.7 °C | device lost off PCIe in prefill chunk 12 of 12 |
| High, earlier, back-to-back | 86.7 °C | 95.2 °C | device lost off PCIe in prefill chunk 6 |
| 4000-token text prompt, back-to-back | 90.7 °C | 93.7 °C | device lost off PCIe in the first chunk |

Every failure came at 93.7–95.2 °C, and the continuous pass peaked at 92.7 °C.
This U50 idles at 80–84 °C and the 366.7 MHz image does not meet timing, so a
continuous High on this card depends on its starting temperature; the same
program on an Alveo U55C (290 MHz) peaks at 63 °C. The cooling pauses are a
debug-only measurement aid and are not in the repository. The paused run
decodes the same 128 tokens as the continuous one, and its prefill is 2.7%
faster on the HW counters (106.02 s against 108.94 s).

## Notes

- **TTFT** is media encoding plus LM prefill, before the first decode step.
  It excludes model initialization, weight loading and the first compile.
- **Compile time** (one-time, before the run): ~9 s for Low, ~23 s for Medium
  and ~97 s for High, mostly one LM image compile per prefill chunk.
- **Prefill is not linear in the prompt length.** Projections and the MLP are,
  but every chunk attends the whole cache prefix, so attention grows with the
  square of the length: 68.9 tok/s at 850 tokens, 65.9 at 2048. Chunks still
  attend the full prefix on the 28 sliding-window layers (masked by the bias)
  and transpose V once per head, which makes long prompts slower than they
  need to be: High's prefill runs at 63% of peak against Low's 71%.
