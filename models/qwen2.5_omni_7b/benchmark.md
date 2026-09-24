# Qwen2.5-Omni-7B on the Apex Compute Unified Engine — tiered benchmark

Low and Medium are continuous eight-engine model runs on the Alveo U50. High
measures four full-resolution vision passes and audio on the FPGA, then measures
individual eight-engine LM operations at High dimensions. High's LM and TTFT
totals are **derived**, not an end-to-end model run. All FPGA times below come
from hardware counters; CPU times are identified separately.

## Platform and workloads

| Board | Device | Engines | Clock | Eight-engine peak | HW version | Model weight bin |
| :--- | :--- | ---: | ---: | ---: | :--- | ---: |
| Alveo U50, 8 GiB HBM | `xdma0` | 8 of 8 | 366.7 MHz | 375.5 GFLOPS | `0x01f2b686` | 5560.3 MiB |

| Tier | Command | Vision | Audio | LM prefill rows | Measurement |
| :--- | :--- | :--- | :--- | ---: | :--- |
| Low | `benchmark.py --low` | None | 2.55 s, 64 soft tokens | 849 | Continuous model run |
| Medium | `benchmark.py --medium` | 1 × 896×896, 1024 soft tokens | 5.10 s, 127 soft tokens | 2047 | Continuous model run |
| High | `benchmark.py --high` | 4 × 896×896, 4096 soft tokens | 6.00 s, 150 soft tokens | 6144 | Measured media + derived LM |

High replays the **same** bundled 896×896 image four times. This measures four
same-shape vision executions, not four distinct camera images. Its 6144 LM rows
are the benchmark target, not a prompt that was run through the resident model.

## Time to first token

This report follows the run summaries' convention: TTFT is media encoding plus
LM prefill, **before the first decode step**. It excludes model initialization,
weight loading, compilation, and host input preparation. High also omits LM
operations outside the isolated-op coverage described below.

| Tier | Vision HW (ms) | Audio HW (ms) | Prefill HW/derived (ms) | TTFT HW/projected (ms) | TTFT (s) | CPU-stage TTFT (s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Low | — | 611.1 | 33368.9 | **33980.0** | 33.98 | 33.98 |
| Medium | 22233.0 | 1180.0 | 84071.0 | **107484.0** | 107.48 | 107.49 |
| High | 89000.9 | 1404.3 | 301111.1 | **391516.3 projected** | 391.52 projected | Not available |

Low and Medium TTFTs are sums of real model-stage hardware counters; their CPU
columns sum the corresponding stage timers. High's vision and audio are real
FPGA measurements, but its prefill is a sum of separately measured operation
latencies across 28 layers. Therefore **391.52 s is not an end-to-end or CPU
timer measurement**. The High TTFT split is 22.7% vision, 0.4% audio, and 76.9%
derived prefill (Low: 1.8% audio/98.2% prefill; Medium: 20.7% vision/1.1%
audio/78.2% prefill).

## Stage throughput and utilization

`Issued` counts the FPGA operations at their executed shapes; `model` counts
the architecture's useful matrix work. Both rates divide by the same
hardware-counter time. Utilization uses the full 375.5-GFLOPS eight-engine
peak, including stages that do not occupy all engines equally.

| Tier | Stage | Issued GFLOP | Model GFLOP | FPGA/derived ms | Issued GFLOPS | Issued % peak | Model GFLOPS | Model % peak |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Low | Audio | 166.1 | 164.9 | 611.1 | 271.8 | 72.4% | 269.9 | 71.9% |
| Low | Prefill | 11392.6 | 11224.8 | 33368.9 | 341.4 | 90.9% | 336.4 | 89.6% |
| Medium | Vision/image | 5811.8 | 5645.4 | 22233.0 | 261.4 | 69.6% | 253.9 | 67.6% |
| Medium | Audio | 336.5 | 328.8 | 1180.0 | 285.1 | 75.9% | 278.6 | 74.2% |
| Medium | Prefill | 28426.0 | 27555.9 | 84071.0 | 338.1 | 90.1% | 327.8 | 87.3% |
| High | Vision/image | 23247.0 | 22581.8 | 89000.9 | 261.2 | 69.6% | 253.7 | 67.6% |
| High | Audio | 400.1 | 387.3 | 1404.3 | 284.9 | 75.9% | 275.8 | 73.5% |
| High | Prefill, derived | 97703.7 | 87760.3 | 301111.1 | 324.5 | 86.4% | 291.5 | 77.6% |

The measured vision cost is 22.23 s for Medium's image and 22.25 s per High
replay. High audio costs 1.40 s. Derived High prefill is 20.40 input tokens/s;
the real Low and Medium prefill rates are 25.44 and 24.35 tokens/s. The High
prefill rate falls as the 6144-row attention workload grows; it should not be
replaced by a linear extrapolation from the shorter model runs.

## Decode throughput

| Tier | Resident context | First-token HW | Average HW | CPU average | Latency per step | Status |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| Low | 862 tokens at final step | 13.34 tok/s | 13.34 tok/s | 13.09 tok/s | 75.0 ms HW | Measured model run, 13 steps |
| Medium | 2079 tokens at final step | 8.88 tok/s | 8.80 tok/s | 8.68 tok/s | 113.6 ms HW average | Measured model run, 32 steps |
| High | 6144-token target | — | 6.66 tok/s raw | — | 150.23 ms raw | Isolated-op sum, not model decode |
| High, corrected estimate | 6144-token target | — | 6.05 tok/s | — | 165.26 ms | Raw latency × 1.10 empirical scale |

High's correction is the benchmark's one global scale, fitted to the Low and
Medium first-token hardware times versus isolated-op totals at comparable
contexts. It is an estimate, not a new hardware counter. The High decode
operation table includes an eight-engine LM head plus global argmax; its full
step still omits normalization, RoPE, residual/elementwise work, embeddings,
KV-cache setup, host transfers, and inter-operation scheduling. High prefill
has the same omitted-work limitation. No High decoded text or numerical
correctness claim is made.

## Source reports and commands

| Tier | Generated report |
| :--- | :--- |
| Low | `qwen2.5_omni_7b_test_xdma0_audio_low_multi-core_8.md` |
| Medium | `qwen2.5_omni_7b_test_xdma0_image+audio_medium_multi-core_8.md` |
| High | `qwen2.5_omni_7b_benchmark_high_op_by_op.md` |

```bash
python models/qwen2.5_omni_7b/benchmark.py --low --dev xdma0 --multi-core 8
python models/qwen2.5_omni_7b/benchmark.py --medium --dev xdma0 --multi-core 8
python models/qwen2.5_omni_7b/benchmark.py --high --dev xdma0 --multi-core 8
```

The generated source reports remain on the benchmark machine for inspection;
only this consolidated `benchmark.md` is versioned as the benchmark result.
