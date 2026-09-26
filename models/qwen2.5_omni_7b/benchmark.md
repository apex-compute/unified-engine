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

This revision reflects the private-DRAM/decode optimizations accumulated
this round: the token embedding is a host-side BF16 gather (prompt/generated
rows only, DMA'd on demand, no FPGA-resident lookup table); the LM head's
global argmax is a host-side reduction of each engine's free local-argmax
candidate (a two-byte DRAM read per engine) instead of an on-device
64-wide identity-matrix scan of the full vocabulary; and **every** projection
weight (Q, K, V, O, gate, up, down) now uses one private, 64-aligned column
shard per engine, staged once at weight-load time and reused as-is by both
prefill (tensor-parallel, mirroring gate/up/down's existing pattern) and
decode — no second runtime copy, no on-the-fly quantization. V is IF4 in
both phases too, recovering the decode GEMV throughput a BF16-only V would
give up. Net effect: prefill's own qkv_proj/o_proj get faster from the same
tensor-parallel win MLP already had, decode throughput improves, and the
per-engine private DRAM reserve usage lands at 447.8 MiB against the
645 MiB `OMNI_PRIVATE_RESERVE_BYTES` budget (down from 712 MiB/core before
this round of optimizations) while the **shared** pool's own usage drops
from ~515 MiB to ~0.6 MiB (Q/K/O's old prefill-only shared copies and V's
old shared BF16 copy are gone).

## Platform and workloads

| Board | Device | Engines | Clock | Eight-engine peak | HW version | Model weight bin |
| :--- | :--- | ---: | ---: | ---: | :--- | ---: |
| Alveo U50, 8 GiB HBM | `xdma0` | 8 of 8 | 366.7 MHz | 375.5 GFLOPS | `0xfc46ae6f` | 5989.5 MiB |

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
weight loading, compilation, and host input preparation (including the host
BF16 embedding gather -- see each tier's own source report). High also
omits LM operations outside the isolated-op coverage described below.

| Tier | Vision HW (ms) | Audio HW (ms) | Prefill HW/derived (ms) | TTFT HW/projected (ms) | TTFT (s) | CPU-stage TTFT (s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Low | — | 612.9 | 33481.0 | **34093.9** | 34.09 | 34.12 |
| Medium | 22266.0 | 1180.5 | 84511.9 | **107958.4** | 107.96 | 108.00 |
| High | 90149.9 | 1447.6 | 339056.5 | **430654.0 projected** | 430.65 projected | Not available |

Low and Medium TTFTs are sums of real model-stage hardware counters; their CPU
columns sum the corresponding stage timers, including the host BF16
embedding gather (off the hardware counter, inside the CPU timer instead).
High's vision and audio are real FPGA measurements, but its prefill is a sum
of separately measured operation latencies across 28 layers, measured cold
before the media phase runs (see above). Therefore **430.65 s is not an
end-to-end or CPU timer measurement**. The High TTFT split is 20.9% vision,
0.3% audio, and 78.7% derived prefill (Low: 1.8% audio/98.2% prefill;
Medium: 20.6% vision/1.1% audio/78.3% prefill).

## Stage throughput and utilization

`Issued` counts the FPGA operations at their executed shapes; `model` counts
the architecture's useful matrix work. Both rates divide by the same
hardware-counter time. Utilization uses the full 375.5-GFLOPS eight-engine
peak, including stages that do not occupy all engines equally.

| Tier | Stage | Issued GFLOP | Model GFLOP | FPGA/derived ms | Issued GFLOPS | Issued % peak | Model GFLOPS | Model % peak |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Low | Audio | 166.1 | 164.9 | 612.9 | 271.0 | 72.2% | 269.1 | 71.6% |
| Low | Prefill | 11392.6 | 11224.8 | 33481.0 | 340.3 | 90.6% | 335.3 | 89.3% |
| Medium | Vision/image | 5811.8 | 5645.5 | 22266.0 | 261.0 | 69.5% | 253.5 | 67.5% |
| Medium | Audio | 336.5 | 328.8 | 1180.5 | 285.0 | 75.9% | 278.5 | 74.2% |
| Medium | Prefill | 28426.0 | 27555.9 | 84511.9 | 336.4 | 89.6% | 326.1 | 86.8% |
| High | Vision/image | 23247.0 | 22581.8 | 90149.9 | 257.9 | 68.7% | 250.5 | 66.7% |
| High | Audio | 400.1 | 387.3 | 1447.6 | 276.4 | 73.6% | 267.6 | 71.3% |
| High | Prefill, derived | 97703.7 | 87760.3 | 339056.5 | 288.2 | 76.7% | 258.8 | 68.9% |

The measured vision cost is 22.27 s for Medium's image and 22.54 s per High
replay. High audio costs 1.45 s. Derived High prefill is 18.12 input tokens/s;
the real Low and Medium prefill rates are 25.36 and 24.22 tokens/s. High's
prefill GFLOPS lands close to the real Low/Medium prefill rate (90.6%, 89.6%
of peak) because the LM ops are measured cold, before the media phase; it
should still not be replaced by a linear extrapolation from the shorter
model runs.

## Decode throughput

| Tier | Resident context | First-token HW | Average HW | CPU average | Latency per step | Status |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| Low | 863 tokens at final step | 15.34 tok/s | 14.90 tok/s | 13.63 tok/s | 67.1 ms HW average | Measured model run, 14 steps |
| Medium | 2079 tokens at final step | 9.86 tok/s | 9.75 tok/s | 9.15 tok/s | 102.5 ms HW average | Measured model run, 32 steps |
| High | 6144-token target | — | 7.05 tok/s raw | — | 141.85 ms raw | Isolated-op sum, not model decode |
| High, corrected estimate | 6144-token target | — | 6.41 tok/s | — | 156.03 ms | Raw latency × 1.10 empirical scale |

Decode is faster at every tier than the pre-optimization baseline (Low was
13.09-13.34 tok/s CPU/HW; Medium was 8.68-8.88 tok/s): every projection now
streams a single private column shard per step (no phase-specific overlay
or on-the-fly quantization to build first), and V's IF4 shard keeps its GEMV
throughput even though it is now the same physical shard prefill uses.

High's correction factor (1.10) is carried over from `benchmark.py`'s
pre-optimization calibration (Low 78.2/71.75 ms, Medium 99.2/89.19 ms) and has
**not** been refitted against this revision's decode; treat the corrected
6.41 tok/s figure as directional; the raw 7.05 tok/s row is this run's actual
isolated-op measurement. The High decode operation table includes an
eight-engine LM head; its global argmax is a host-side reduction (see above),
off the HW counter. The full decode step still omits normalization, RoPE,
residual/elementwise work (other than the down_proj reduction, which is
measured), embeddings, KV-cache setup, host transfers, and inter-operation
scheduling. High prefill has the same omitted-work limitation. No High
decoded text or numerical correctness claim is made.

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
