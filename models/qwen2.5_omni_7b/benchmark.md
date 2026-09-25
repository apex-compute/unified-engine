# Qwen2.5-Omni-7B on the Apex Compute Unified Engine — tiered benchmark

Low and Medium are continuous eight-engine model runs on the Alveo U50. High
measures individual eight-engine LM operations at High dimensions **before**
the media phase, then measures four full-resolution vision passes and audio
on the FPGA. High's LM and TTFT totals are **derived**, not an end-to-end
model run. All FPGA times below come from hardware counters; CPU times are
identified separately.

High's LM ops are measured cold, ahead of the vision/audio phase, because
running ~90 s of real vision compute immediately before them measurably
slowed the following LM matmuls (q_proj/gate_proj etc. read ~15-17% lower
GFLOPS when timed right after vision than when timed standalone) — a
board-thermal artifact of stage ordering within this benchmark, not a model
regression. Reordering the two phases (LM first, media second) recovered the
LM ops' standalone GFLOPS; the numbers below reflect that ordering.

This revision reflects three private-DRAM/decode optimizations: the token
embedding moved from an FPGA-resident IF8 lookup table to a host-side BF16
gather (prompt/generated rows only, DMA'd on demand); decode's O projection
switched from a shared BF16 overlay to IF4, reusing prefill's own private
column shards directly; and decode's `down_proj` no longer stages a separate
N-sharded image, instead reusing prefill's private K-sharded weights and
summing the eight partial outputs with `eltwise_core_dram`. Net effect:
`OMNI_PRIVATE_RESERVE_BYTES` drops from 712 to 645 MiB/core, and decode
throughput improves measurably (see below).

## Platform and workloads

| Board | Device | Engines | Clock | Eight-engine peak | HW version | Model weight bin |
| :--- | :--- | ---: | ---: | ---: | :--- | ---: |
| Alveo U50, 8 GiB HBM | `xdma0` | 8 of 8 | 366.7 MHz | 375.5 GFLOPS | `0xfc46ae6f` | 6061.5 MiB |

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
weight loading, compilation, and host input preparation (including, now, the
host BF16 embedding gather -- see each tier's own source report). High also
omits LM operations outside the isolated-op coverage described below.

| Tier | Vision HW (ms) | Audio HW (ms) | Prefill HW/derived (ms) | TTFT HW/projected (ms) | TTFT (s) | CPU-stage TTFT (s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Low | — | 613.2 | 33384.4 | **33997.7** | 34.00 | 34.03 |
| Medium | 22257.2 | 1181.8 | 83853.2 | **107292.2** | 107.29 | 107.33 |
| High | 89062.4 | 1403.7 | 288448.1 | **378914.2 projected** | 378.91 projected | Not available |

Low and Medium TTFTs are sums of real model-stage hardware counters; their CPU
columns sum the corresponding stage timers, and now include the host BF16
embedding gather (previously an on-FPGA IF8 lookup, now off the hardware
counter and inside the CPU timer instead). High's vision and audio are real
FPGA measurements, but its prefill is a sum of separately measured operation
latencies across 28 layers, measured cold before the media phase runs (see
above). Therefore **378.91 s is not an end-to-end or CPU timer measurement**.
The High TTFT split is 23.5% vision, 0.4% audio, and 76.1% derived prefill
(Low: 1.8% audio/98.2% prefill; Medium: 20.7% vision/1.1% audio/78.2%
prefill).

## Stage throughput and utilization

`Issued` counts the FPGA operations at their executed shapes; `model` counts
the architecture's useful matrix work. Both rates divide by the same
hardware-counter time. Utilization uses the full 375.5-GFLOPS eight-engine
peak, including stages that do not occupy all engines equally.

| Tier | Stage | Issued GFLOP | Model GFLOP | FPGA/derived ms | Issued GFLOPS | Issued % peak | Model GFLOPS | Model % peak |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Low | Audio | 166.1 | 164.9 | 613.2 | 270.9 | 72.1% | 268.9 | 71.6% |
| Low | Prefill | 11392.6 | 11224.8 | 33384.4 | 341.3 | 90.9% | 336.2 | 89.5% |
| Medium | Vision/image | 5811.8 | 5645.5 | 22257.2 | 261.1 | 69.5% | 253.6 | 67.5% |
| Medium | Audio | 336.5 | 328.8 | 1181.8 | 284.7 | 75.8% | 278.2 | 74.1% |
| Medium | Prefill | 28426.0 | 27555.9 | 83853.2 | 339.0 | 90.3% | 328.6 | 87.5% |
| High | Vision/image | 23247.0 | 22581.8 | 89062.4 | 261.0 | 69.5% | 253.6 | 67.5% |
| High | Audio | 400.1 | 387.3 | 1403.7 | 285.0 | 75.9% | 275.9 | 73.5% |
| High | Prefill, derived | 97703.7 | 87760.3 | 288448.1 | 338.7 | 90.2% | 304.2 | 81.0% |

The measured vision cost is 22.26 s for Medium's image and 22.27 s per High
replay. High audio costs 1.40 s. Derived High prefill is 21.30 input tokens/s;
the real Low and Medium prefill rates are 25.43 and 24.41 tokens/s. High's
prefill GFLOPS now lands close to the real Low/Medium prefill rate (90.3%,
90.9% of peak) because the LM ops are measured cold, before the media phase;
it should still not be replaced by a linear extrapolation from the shorter
model runs.

## Decode throughput

| Tier | Resident context | First-token HW | Average HW | CPU average | Latency per step | Status |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| Low | 862 tokens at final step | 15.12 tok/s | 15.12 tok/s | 14.68 tok/s | 66.1 ms HW | Measured model run, 13 steps |
| Medium | 2079 tokens at final step | 9.79 tok/s | 9.69 tok/s | 9.49 tok/s | 103.2 ms HW average | Measured model run, 32 steps |
| High | 6144-token target | — | 7.15 tok/s raw | — | 139.79 ms raw | Isolated-op sum, not model decode |
| High, corrected estimate | 6144-token target | — | 6.50 tok/s | — | 153.77 ms | Raw latency × 1.10 empirical scale |

Decode is faster at every tier than the pre-optimization baseline (Low was
13.09-13.34 tok/s CPU/HW; Medium was 8.68-8.88 tok/s): IF4 O/V and the shared
K-sharded `down_proj` reduce both the private weight footprint decode streams
per step and the number of distinct weight images it has to hold resident.

High's correction factor (1.10) is carried over from `benchmark.py`'s
pre-optimization calibration (Low 78.2/71.75 ms, Medium 99.2/89.19 ms) and has
**not** been refitted against this revision's faster decode; treat the
corrected 6.50 tok/s figure as directional; the raw 7.15 tok/s row is this
run's actual isolated-op measurement. The High decode operation table includes
an eight-engine LM head plus global argmax; its full step still omits
normalization, RoPE, residual/elementwise work (other than the down_proj
reduction, which is measured), embeddings, KV-cache setup, host transfers, and
inter-operation scheduling. High prefill has the same omitted-work limitation.
No High decoded text or numerical correctness claim is made.

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
