# Qwen2.5-Omni-7B on the Apex Compute Unified Engine — tiered benchmark

Three workload tiers on one Alveo U50 at eight engines: a voice command, a
single-camera query, and a multi-camera longer-context query.

All timings are FPGA hardware counters. The one exception is marked in place.

## Platform

| Board | Engines | Clock | Peak | Per-engine peak | HW version | Weights | Context |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- | ---: |
| Alveo U50, `xdma0`, 8 GiB HBM | 8 of 8 | 366.7 MHz | 375.5 GFLOPS | 46.9 GFLOPS | `0xe9cbe74b` | `params.bin`, 5608.3 MiB | 2500 tokens, 2560-row KV |

Identical in all three tiers, `params.bin` byte for byte.

## Tiers

| Tier | Command | Camera frames | Vision tokens | Speech | Audio tokens | Text tokens | Input tokens |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Low — voice command | `--low` | 0 | 0 | 2.55 s | 64 | 785 | 849 |
| Medium — single camera | `--medium` | 1 x 896x896 | 1024 | 5.10 s | 128 | 895 | 2047 |
| High — multi-camera | `--high` | 4 x 896x896 | 4096 | 6.00 s | 150 | 1898 | 6144 |

Audio encodes at 25.1 soft tokens per second of speech. High's 150 tokens over
6.00 s is the configured 600-mel-frame maximum, so that tier sits at the
ceiling of what the audio front end accepts in one request.

## Time to first token, by input modality

| Tier | Vision encoder (ms) | Audio encoder (ms) | LLM prefill (ms) | TTFT (ms) | TTFT (s) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Low | — | 614.0 | 35197.6 | 35811.6 | 35.81 |
| Medium | 36229.1 | 1172.3 | 83948.5 | 121349.9 | 121.35 |
| High | 145355.1 | 1400.8 | 251721.4 | 398477.3 | 398.48 |

| Tier | Vision share | Audio share | Prefill share |
| :--- | ---: | ---: | ---: |
| Low | — | 1.7% | 98.3% |
| Medium | 29.9% | 1.0% | 69.2% |
| High | 36.5% | 0.4% | 63.2% |

Host overhead is negligible: the CPU timer and the hardware counter agree to
within 0.1% at every tier.

### Unit costs

| Stage | Low | Medium | High | Unit cost |
| :--- | ---: | ---: | ---: | :--- |
| Vision, per soft token | — | 35.38 ms | 35.49 ms | ~35.4 ms |
| Vision, per 896x896 frame | — | 36.23 s | 36.34 s | ~36.3 s |
| Audio, per soft token | 9.59 ms | 9.16 ms | 9.34 ms | ~9.3 ms |
| Audio, per second of speech | 240.8 ms | 229.9 ms | 233.5 ms | ~0.23x realtime |
| Prefill, per input token | 41.46 ms | 41.01 ms | 40.97 ms | ~41.1 ms |

**Audio is never a design constraint.** At most 1.7% of TTFT, and 0.4% at High.
A 6-second query costs 1.4 s to encode.

**Vision decides multi-camera feasibility.** Each frame costs a flat 36.3 s.
Going from one camera to four adds 109.1 s before the LM sees a token.

**Prefill is linear in this range**, 41 ms per token from 849 to 2048 within
1.2%. The dense projections dominate and the attention term has not yet
asserted itself, so prefill cost follows token count alone.

### Efficiency by stage

`Issued` is the work billed at the shapes actually run; `effective` is what
the architecture owes at its own dimensions, so it is the figure comparable to
any other implementation of this model.

| Stage | Tier | Issued GFLOPS | % of peak | Effective GFLOPS | % of peak | Useful work |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Prefill | Low | 341.6 | 91.0% | 318.9 | 84.9% | 93.4% |
| Prefill | Medium | 338.8 | 90.2% | 328.2 | 87.4% | 96.9% |
| Prefill | High | 338.9 | 90.3% | 328.6 | 87.5% | 96.9% |
| Vision | Medium | 290.7 | 77.4% | 155.8 | 41.5% | 53.6% |
| Vision | High | 288.9 | 77.0% | 154.9 | 41.3% | 53.6% |
| Audio | Low | 270.6 | 72.1% | 268.6 | 71.5% | 99.3% |
| Audio | Medium | 288.1 | 76.7% | 281.6 | 75.0% | 97.7% |
| Audio | High | 279.4 | 74.4% | 270.5 | 72.0% | 96.8% |

Prefill is the best-utilised stage on the board; its share of TTFT is large
because the work is large, not because it runs badly. The vision encoder issues
53.6% padding, making that gap the largest recoverable inefficiency in TTFT.
Vision and audio's High-tier effective figures are not in the harness's own
printed summary -- `_write_high_summary` only turns `model_flops` into an
effective-GFLOPS column for prefill's monolithic estimate -- but every
phase's raw result already carries `model_flops`, so they are derived here
from the same run rather than left blank. High's per-frame/per-second effective
rate lands within 1% of Medium's for both stages, as it should: same shape,
replicated.

## Decode throughput

| Tier | Resident context | First token | Average | ms/token | GFLOP/token | GFLOPS | % of peak |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Low | 862 | 13.32 tok/s | 13.32 tok/s | 75.1 | 14.53 | 193.5 | 51.5% |
| Medium | 2158 | 8.85 tok/s | 8.74 tok/s | 114.5 | 15.03 | 131.3 | 35.0% |
| High | 6144 | 5.32 tok/s | 5.32 tok/s | 187.9 | 16.61 | 88.8 | 23.6% |

Generation degrades 60% from 862 to 6144 resident tokens — 13.32 to 5.32
tok/s, a 2.5x rise in per-token latency for 7.1x the context.

The cause is not arithmetic. Work per token rises 14% across that range, while
efficiency falls from 51.5% to 23.6% of peak. The decoder reads a KV cache
growing linearly with context while its matmul shapes stay fixed, so the added
time is memory traffic, not compute.

*Derived, not measured:* one decode step streams the full 5.88 GB weight set.
At Low's 75.1 ms that is 78.3 GB/s, within ~9% of the ~85 GB/s aggregate AXI
ceiling — short-context decode sits near the DRAM weight-streaming floor, and
the gap that opens at Medium and widens at High is attention and KV overhead
on top of that floor.

High's decode runs at the tier's true 6144-token resident depth, not a
shorter stand-in. Rows beyond the tier's real prefill chunk (2048 tokens) are
zero-filled placeholders in the KV cache, so decode's attention shape and DMA
cost are real at 6144 while its logits are not — consistent with the
numerics-unchecked contract stated for the High tier throughout this
document.

## Quantization

Set in `qwen2.5_omni_7b_config.json` under `precision`; not a per-run choice.

| Component | Precision | Size on device | Derivation |
| :--- | :--- | ---: | :--- |
| LM q, k, o, gate, up, down | IF4 | q_proj 171.5 MiB | 28 x 3584 x 3584 x 0.5 B |
| Vision encoder | IF4 | | |
| Audio encoder | IF4 | | |
| LM head | IF4 | 259.9 MiB | 152064 x 3584 x 0.5 B |
| Token embedding | **IF8** | 519.8 MiB | 152064 x 3584 x 1 B |
| LM `v_proj` | **BF16** | 98.0 MiB | 28 x 512 x 3584 x 2 B |
| `o_proj`, decode only | **BF16** | 686.0 MiB | 28 x 3584 x 3584 x 2 B |
| RMSNorm weights, biases | BF16 | | |
| Activations | BF16 | `lm.io` 17.5 MiB | 2560 x 3584 x 2 B |
| KV cache | BF16 | 70.0 MiB each | 28 x 4 x 2560 x 128 x 2 B |

**IF4** is block-adaptive 4-bit at block size 64 with a BF16 scale per block
whose *sign bit selects the block's format*: negative means INT4, positive
means FP4, magnitude is the effective scale. So "INT4/FP4 block-quantized" is
exact — both formats are present, chosen per 64-element block.

Three components are bold above because they depart from a pure-IF4
description and affect any model-size or FLOP-per-byte comparison. They are
identical across the tiers, so tier-to-tier comparability is unaffected. Every
size in that table was checked against the per-core DRAM layout the runs print.

## What ran on hardware

Low and Medium are single continuous runs, measured end to end, producing real
decoded text.

High cannot run as one resident context — that is why the tier exists. It runs
as four processes, each with its own eight-core reset and compilation:

| Component | ms | Status |
| :--- | ---: | :--- |
| Vision, 4 frames | 145355.1 | measured, all four on the FPGA |
| Audio, 6.0 s | 1400.8 | measured |
| Prefill, chunk 1 of 3 | 83907.1 | measured, a real 2048-token FPGA run |
| Prefill, chunks 2 and 3 | 167814.3 | chunk 1's measurement counted twice more |
| **TTFT** | **398477.3** | **57.9% elapsed on hardware, 42.1% replayed** |

Nothing there is modelled or curve-fit; what is synthetic is the chunk count,
not the rate. The x3 holds because the chunks are independent — identical
2048-token shape, no KV between them — so running all three would move the
number by measurement noise.

It is nonetheless a *lower* bound on a true 6144-token context, whose later
chunks would attend over a growing KV. The summary's separate monolithic
estimate, 267.1 s of prefill for 413.8 s TTFT, lands 3.9% higher for that
reason.

High is performance-only: numerics and coherence are unchecked, EOS is
suppressed so all 32 decode steps execute, and no claim is made that a
6144-token request fits in DRAM.

## Source

| Tier | Summary file |
| :--- | :--- |
| Low | `qwen2.5_omni_7b_test_xdma0_audio_low_multi-core_8.md` |
| Medium | `qwen2.5_omni_7b_test_xdma0_image+audio_medium_multi-core_8.md` |
| High | `qwen2.5_omni_7b_test_xdma0_image+audio_high_multi-core_8.md` |

```bash
python models/qwen2.5_omni_7b/qwen2.5_omni_7b_test.py --multi-core 8 --low
python models/qwen2.5_omni_7b/qwen2.5_omni_7b_test.py --multi-core 8 --medium
python models/qwen2.5_omni_7b/qwen2.5_omni_7b_test.py --multi-core 8 --high
```
