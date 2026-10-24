# Gemma 4 E4B on the Apex Compute Unified Engine — tiered benchmark

`benchmark.py --low | --medium | --high` runs the same workload shapes as
[gemma4_e2b](../gemma4_e2b/benchmark.md) (and qwen2.5_omni_7b) on E4B, eight
engines per run. All FPGA times come from hardware counters; wall-clock times
are labelled as such. E4B runs on the shared gemma4_e2b implementation (see
[README](README.md)).

## Platforms

| Board | Engines | Clock | Eight-engine peak | HW version |
| :--- | ---: | ---: | ---: | :--- |
| Alveo U50, 8 GiB HBM (p2) | 8 of 8 | 366.7 MHz | 375.5 GFLOPS | `0x25511139` (flash image) |
| Alveo U50, 8 GiB HBM (p2) | 8 of 8 | 333.3 MHz | 341.3 GFLOPS | `0x6798f78b` ([333.3 MHz results](#results-alveo-u50-3333-mhz)) |
| Alveo U55C, 16 GiB HBM (aztec) | 8 of 12 | 290.0 MHz | 297.0 GFLOPS | `0xa4fba9d9` |

Weights: 2482 MB of DRAM (171 MB shared + 2311 MB of per-engine slices).

## Workloads

| Tier | Camera frames (896×896) | Spoken query | Total prompt | Decode budget |
| :--- | :--- | :--- | ---: | ---: |
| Low | 0 | 2.55 s, 64 soft tokens | 850 | 32 |
| Medium | 1, 256 soft tokens | 4.5 s, 113 soft tokens | 2048 | 64 |
| High | 4, 1024 soft tokens | 6.0 s, 150 soft tokens | 6144 | 128 |

## Results: Alveo U55C (290 MHz)

| Metric | Low | Medium | High |
| :--- | ---: | ---: | ---: |
| Vision encoder, HW (s) | — | 5.98 | 23.91 (4 frames) |
| Audio encoder, HW (s) | 3.53 | 6.85 | 10.11 |
| Prefill chunks | 2 | 4 | 12 |
| Prefill, HW (s) | 30.65 | 75.27 | 248.53 |
| Prefill speed, HW (tok/s) | 27.7 | 27.2 | 24.7 |
| Prefill model GFLOPS (% peak) | 224.3 (75.5%) | 222.8 (75.0%) | 208.6 (70.2%) |
| **TTFT, HW counters (s)** | **34.18** | **88.10** | **282.55** |
| TTFT, wall clock (s) | 34.79 | 89.16 | 284.88 |
| Decoded tokens (budget) | 32 (32) | 64 (64) | 128 (128) |
| Decode first token, HW (tok/s) | 10.6 | 7.9 | 4.3 |
| Decode average, HW (tok/s) | 10.6 | 7.8 | 4.3 |
| Decode average, wall clock (tok/s) | 9.4 | 7.1 | 4.1 |
| Decode model GFLOPS (% peak) | 101.4 (34.1%) | 75.8 (25.5%) | 43.4 (14.6%) |
| Final context (tokens) | 881 | 2111 | 6271 |
| Die temperature, start → peak | 53.3 → 57.8 °C | 55.8 → 62.3 °C | 58.8 → 67.8 °C |

## Results: Alveo U50 (366.7 MHz)

| Metric | Low | Medium | High |
| :--- | ---: | ---: | ---: |
| Vision encoder, HW (s) | — | 4.81 | 19.23 (4 frames) |
| Audio encoder, HW (s) | 2.81 | 5.44 | 8.03 |
| Prefill, HW (s) | 24.31 | 59.73 | 197.31 |
| Prefill speed, HW (tok/s) | 34.9 | 34.3 | 31.1 |
| Prefill model GFLOPS (% peak) | 282.8 (75.3%) | 280.7 (74.8%) | 262.7 (70.0%) |
| **TTFT, HW counters (s)** | **27.12** | **69.98** | **224.57** |
| TTFT, wall clock (s) | 27.71 | 71.37 | —¹ |
| Decode first token, HW (tok/s) | 13.3 | 10.0 | 5.4 |
| Decode average, wall clock (tok/s) | 11.4 | 8.4 | 4.8 |
| Decode model GFLOPS (% peak) | 126.8 (33.8%) | 95.3 (25.4%) | 54.3 (14.5%) |
| Die temperature, start → peak | 72.3 → 79.2 °C | 70.3 → 84.2 °C | 60.8 → 72.3 °C¹ |

¹ Run continuously, a High prefill takes this U50 past 94 °C and the device
drops off PCIe (see [gemma4_e2b](../gemma4_e2b/benchmark.md#high-and-the-boards-temperature)),
so High is measured with cooling pauses: the 12 prefill chunks run in four
groups of three, and the run waits for the die to cool to 60 °C before each
group (598, 548 and 592 s). The HW times exclude the pauses; the wall-clock
TTFT includes them (1966 s) and is not meaningful. The pauses are a debug-only
measurement aid, not in the repository. The U55C runs High continuously at
68 °C. The U50 is 1.26× faster per tier (its clock ratio), e.g. Medium TTFT
69.98 s against 88.10 s and High 224.57 s against 282.55 s.

## Results: Alveo U50 (333.3 MHz)

`0x6798f78b`: the split35 floorplan with the engine clock lowered to
333.33 MHz (andromeda `alveo/u50-split35-floorplan-333mhz`, CI run
40031819605; routed WNS −0.540 ns). Low, Medium and High ran back to back with
no cooling pauses, in one series with E2B and Qwen-Omni Low/Medium/High
(2026-10-24, p2). Outputs match the ones below.

| Metric | Low | Medium | High |
| :--- | ---: | ---: | ---: |
| Vision encoder, HW (s) | — | 5.04 | 20.17 (4 frames) |
| Audio encoder, HW (s) | 3.04 | 5.89 | 8.70 |
| Prefill chunks | 2 | 4 | 12 |
| Prefill, HW (s) | 26.61 | 65.36 | 215.94 |
| Prefill speed, HW (tok/s) | 31.9 | 31.3 | 28.4 |
| Prefill model GFLOPS (% peak) | 258.4 (75.7%) | 256.5 (75.2%) | 240.1 (70.3%) |
| **TTFT, HW counters (s)** | **29.64** | **76.30** | **244.82** |
| TTFT, wall clock (s) | 30.21 | 77.31 | 246.02 |
| Decoded tokens (budget) | 32 (32) | 64 (64) | 128 (128) |
| Decode first token, HW (tok/s) | 12.4 | 9.3 | 5.0 |
| Decode average, HW (tok/s) | 12.4 | 9.1 | 4.9 |
| Decode average, wall clock (tok/s) | 10.4 | 8.3 | 4.8 |

At 366.7 MHz the U50 image misses timing by 0.74–0.78 ns, and that is the
image the U50 failures were measured on. On 2026-10-23/24 two 366.7 MHz
builds failed the same continuous High: `0x821ebde1` lost the board off PCIe
partway through the Qwen-Omni High prefill (3 of 3 runs, two U50s), and the
split35 floorplan `0xe0942acd` (WNS −0.742 ns) returned NaN/Inf from the audio
encoder (2 of 2) and one NaN in `user_hw_test.py`'s 8-engine matmul. The same
floorplan at 333.3 MHz (`0x6798f78b`, WNS −0.540 ns) and two 200 MHz
timing-clean images (`0xc8a797b8`, `0x0b8030a6`) ran every tier back to back
with no pauses and no failures.

## Outputs

- **Low:** "**Transcription:** "Unified Engine. Apex Computes." **Explanation:**
  The speaker is naming two entities: "Unified Engine" and "Apex Comput…"
- **Medium:** "**Transcription:** (No speech was provided.) **Image
  Description:** The image shows a landscape featuring a valley surrounded by
  mountains under a sky with clouds and a sun setting over the mountains. …"
- **High:** "**Transcription:** Unified Engine. **Camera Frame Descriptions:**
  **Frame 1:** The scene shows a valley with dense green forest, and the sun is
  setting over the mountains, … **Frame 2:** …"

### Medium's missing transcription is IF4, not the FPGA

On the same 2048-token Medium prompt (image + 4.5 s audio + filler text):

| Model | Answer |
| :--- | :--- |
| HF E4B, bf16 (host) | "**Speech Transcription:** Unified Engine, Apex Computes custom ML inference accelerator. …" |
| HF E4B, every LM projection IF4-round-tripped (host) | "**Transcription:** (No speech was provided in the input.) …" |
| FPGA (IF4) | "**Transcription:** (No speech was provided.) …" |

The FPGA reproduces the IF4 model. With a short prompt, the same image + audio
input transcribes correctly on the FPGA ("Unified Engine, Apex Computes custom
ML inference accelerator."), and audio alone does at any length. E4B's IF4
weights lose the audio under a long joint prompt; E2B's do not.

## Against E2B (same tiers, same boards)

| TTFT, HW (s) | Low | Medium | High |
| :--- | ---: | ---: | ---: |
| E2B, U50 | 15.12 | 41.12 | 133.23 |
| E4B, U50 | 27.12 | 69.98 | 224.57 |
| E2B, U55C | 19.04 | 51.79 | 170.86 |
| E4B, U55C | 34.18 | 88.10 | 282.55 |

E4B's prefill does about 2× E2B's work per token and runs at a similar or
higher share of peak (75% against 70–72%), so its TTFT is 1.65–1.8× E2B's;
decode is 2.0–2.1× slower.
