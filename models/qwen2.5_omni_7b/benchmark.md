# Qwen2.5-Omni-7B on the Apex Compute Unified Engine — tiered benchmark

Low, Medium and High are eight-engine model runs on the Alveo U50. All FPGA
times below come from hardware counters; CPU times are identified separately.
This revision includes the host-side BF16 embedding gather, the host-side LM
head argmax, and one private 64-aligned column shard per engine for every
projection weight (Q, K, V, O, gate, up, down), shared by prefill and decode.

## Platform and workloads

| Board | Device | Engines | Clock | Eight-engine peak | HW version | Model weight bin |
| :--- | :--- | ---: | ---: | ---: | :--- | ---: |
| Alveo U50, 8 GiB HBM | `xdma0` | 8 of 8 | 366.7 MHz | 375.5 GFLOPS | `0xfc46ae6f` | 5989.5 MiB |

| Tier | Command | Vision | Audio | LM prefill rows |
| :--- | :--- | :--- | :--- | ---: |
| Low | `benchmark.py --low` | None | 2.55 s, 64 soft tokens | 849 |
| Medium | `benchmark.py --medium` | 1 × 896×896, 1024 soft tokens | 5.10 s, 127 soft tokens | 2047 |
| High | `benchmark.py --high` | 4 × 896×896, 4096 soft tokens | 6.00 s, 150 soft tokens | 6144 |

High is measured on HW `0x25511139` (same 366.7 MHz U50). Run continuously,
its ~290 s prefill takes the die past 95 °C and the board drops off PCIe, so
High was measured with a debug-only split: the 28 LM layers ran as four 7-layer
programs, with a wait for the die to cool to 60 °C before each. Each part took
73.0–73.2 s and peaked at 76–84 °C. The pauses are excluded from the HW times,
and splitting adds no device work (it only relaunches the programs). The split
code is not part of the repository.

## Time to first token

This report follows the run summaries' convention: TTFT is media encoding plus
LM prefill, **before the first decode step**. It excludes model initialization,
weight loading, compilation, and host input preparation (including the host
BF16 embedding gather -- see each tier's own source report).

| Tier | Vision HW (ms) | Audio HW (ms) | Prefill HW (ms) | TTFT (s) | CPU-stage TTFT (s) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Low | — | 612.9 | 33481.0 | 34.09 | 34.12 |
| Medium | 22266.0 | 1180.5 | 84511.9 | 107.96 | 108.00 |
| High | 87398.2 | 1416.1 | 292688.2 | 381.50 | — (cooling pauses) |

## Stage throughput and utilization

| Tier | Stage | Issued GFLOP | Model GFLOP | FPGA ms | Issued GFLOPS | Issued % peak | Model GFLOPS | Model % peak |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Low | Audio | 166.1 | 164.9 | 612.9 | 271.0 | 72.2% | 269.1 | 71.6% |
| Low | Prefill | 11392.6 | 11224.8 | 33481.0 | 340.3 | 90.6% | 335.3 | 89.3% |
| Medium | Vision/image | 5811.8 | 5645.5 | 22266.0 | 261.0 | 69.5% | 253.5 | 67.5% |
| Medium | Audio | 336.5 | 328.8 | 1180.5 | 285.0 | 75.9% | 278.5 | 74.2% |
| Medium | Prefill | 28426.0 | 27555.9 | 84511.9 | 336.4 | 89.6% | 326.1 | 86.8% |
| High | Vision/image | 23150.3 | 22581.8 | 87398.2 | 264.9 | 70.5% | 258.4 | 68.8% |
| High | Audio | 400.1 | 387.3 | 1416.1 | 282.5 | 75.2% | 273.5 | 72.8% |
| High | Prefill | 95524.1 | 87744.8 | 292688.2 | 326.4 | 86.9% | 299.8 | 79.8% |

## Decode throughput

| Tier | Resident context | First-token HW | Average HW | CPU average | Latency per step |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Low | 863 tokens at final step | 15.34 tok/s | 14.90 tok/s | 13.63 tok/s | 67.1 ms HW average |
| Medium | 2079 tokens at final step | 9.86 tok/s | 9.75 tok/s | 9.15 tok/s | 102.5 ms HW average |
| High | 6163 tokens at final step | 6.29 tok/s | 6.26 tok/s | 4.63 tok/s | 159.6 ms HW average |

High's model stopped after 19 tokens: it transcribed the speech correctly
("unified engine apex computes custom ml inference accelerator unified
engine.") and ended without describing the frames.

## Speech output (Talker + Token2Wav)

`--speak` adds speech after Thinker decode: the Talker (24-layer Qwen2
decoder, IF4) turns the reply into codec tokens, and Token2Wav (flow-matching
DiT with 36 evaluations, then the BigVGAN vocoder, bf16) turns them into a
24 kHz waveform. All three run on the same eight engines. Both runs below hit
the 300-codec-token Token2Wav cap, which is 6.00 s of audio (50 codec tokens
per second of speech). Measured on HW `0xbaa4c63b` at 366.7 MHz, speaker
Chelsie. Real time is seconds of audio divided by device time.

| Stage | Device time | Throughput | Real time | Model % peak | Issued % peak |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Talker (prefix + decode) | 8.79 s | 39.0 codec tok/s decode | 0.68× | 29.3% | 29.4% |
| Token2Wav DiT | 62.05 s | 0.58 evaluations/s | 0.10× | 69.0% | 80.4% |
| Token2Wav BigVGAN | 8.91 s | — | 0.67× | 33.9% | 41.0% |
| **Speech total** | **79.75 s** | **6.00 s audio** | **0.075×** | **60.7%** | **70.3%** |

The table is the text-prompt run. With an image+audio prompt, the Talker
conditions over a longer prefix, which slows it to 20.69 s (23.8 codec tok/s
decode, 19.9% of peak). That gives a speech total of 92.91 s, 0.065× real
time, at 53.7% of peak.

Speech is not real time: generating 6.00 s of audio takes 80–93 s of device
time. The DiT is 67–78% of it and already runs at 69% of peak; the Talker
decodes at 0.78× real-time codec rate.

## Source reports and commands

| Tier | Generated report |
| :--- | :--- |
| Low | `qwen2.5_omni_7b_test_xdma0_audio_low_multi-core_8.md` |
| Medium | `qwen2.5_omni_7b_test_xdma0_image+audio_medium_multi-core_8.md` |
| High | `qwen2.5_omni_7b_test_xdma0_image+audio_high_multi-core_8.md` |
| Speech, text prompt | `qwen2.5_omni_7b_test_xdma0_text_speak_Chelsie_multi-core_8.md` |
| Speech, image+audio prompt | `qwen2.5_omni_7b_test_xdma0_image+audio_speak_Chelsie_multi-core_8.md` |

```bash
python models/qwen2.5_omni_7b/benchmark.py --low --dev xdma0 --multi-core 8
python models/qwen2.5_omni_7b/benchmark.py --medium --dev xdma0 --multi-core 8
python models/qwen2.5_omni_7b/benchmark.py --high --dev xdma0 --multi-core 8
python models/qwen2.5_omni_7b/qwen2.5_omni_7b_test.py --dev xdma0 --multi-core 8 --speak
python models/qwen2.5_omni_7b/qwen2.5_omni_7b_test.py --dev xdma0 --multi-core 8 --image --audio --speak
```

The generated source reports remain on the benchmark machine for inspection;
only this consolidated `benchmark.md` is versioned as the benchmark result.
