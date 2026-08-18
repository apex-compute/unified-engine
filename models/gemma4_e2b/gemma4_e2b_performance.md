# Gemma4 E2B performance

## Test setup

Measurements were collected on `p620` with the Kintex-7 bitstream
`0x52a71442`, XDMA device `xdma0`, and a 5.042213410674164 ns clock period.
Both implementations used `test_samples/yosemite.jpg`, the prompt
`Describe this image in detail.`, the same `params.bin`, IF4 projection
weights, 35 LM layers, 16 vision layers, and 256 image soft tokens.

> **2026-08-17 refresh.** The Kintex, Kintex 2-core, and Alveo columns were
> re-measured from clean builds. Kintex used `--dev xdma1` with accepted HW
> version `0x3d04c689`; Alveo used `--device alveo` with accepted HW version
> `0x6bb5d25d`. Each run was preceded by `make clean`
> (`clean_program_bins.sh`) so every program image was recompiled from scratch.

Commands:

```bash
python models/gemma4_e2b/gemma4_e2b_test.py --dev xdma1 --image
python models/gemma4_e2b/gemma4_e2b_test.py --dev xdma1 --image --profile
python models/gemma4_e2b/gemma4_e2b_test.py --dev xdma1 --image --multi-core
python models/gemma4_e2b/gemma4_e2b_test.py --dev xdma1 --image --multi-core --profile
python models/gemma4_e2b/gemma4_e2b_test.py --device alveo --image
python models/gemma4_e2b/gemma4_e2b_test.py --device alveo --image --multi-core
python models/gemma4_e2b/gemma4_e2b_test.py --device alveo --image --multi-core 4
python models/gemma4_e2b/gemma4_e2b_test.py --device alveo --image --multi-core 8
```

## Performance comparison

| Metric | Legacy | Kintex | Kintex 2-core | rk | Alveo | Alveo 2-core | Alveo 4-core | Alveo 8-core |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vision FPGA execution | 55.54 s | 53.78 s | 27.77s | 32.56 s | **29.05 s** | **15.4 s** | **8.00 s** | **4.77 s** |
| Vision end-to-end path | not reported | 54.37 s | 28.58s | 33.03 s | **36.10 s** | **19.6 s** | **11.06 s** | **7.97 s** |
| Vision throughput | not reported | 21.37 GFLOPS | 41.38 GFLOPS | 35.00 GFLOPS | **39.56 GFLOPS** | **74.4 GFLOPS** | **143.61 GFLOPS** | **240.66 GFLOPS** |
| Vision soft tokens | 256 | 256 | 256 | 256 | 256 | 256 | 256 | 256 |
| LM prefill sequence executed | 512-token template | 272 actual tokens | 272 actual tokens | 273 actual tokens | 272 actual tokens | 272 actual tokens | 272 actual tokens | 272 actual tokens |
| LM prefill FPGA latency | 113.571 s | 51.52 s | 31.87s | 32.544 s | **28.51 s** | **17.3 s** | **11.60 s** | **8.79 s** |
| LM prefill throughput | 23.20 GFLOPS | 23.94 GFLOPS | 38.76 GFLOPS | 37.89 GFLOPS | **43.26 GFLOPS** | **71.22 GFLOPS** | **106.32 GFLOPS** | **140.34 GFLOPS** |
| Decode average throughput | 2.68 tok/s | 3.74 tok/s | 3.76 tok/s | 6.00 tok/s | **5.47 tok/s** | **5.49 tok/s** | **5.49 tok/s** | **5.56 tok/s** |
| Decode peak first-token throughput | 2.86 tok/s | 4.00 tok/s | 4.0 tok/s | 6.23 tok/s | **5.79 tok/s** | **5.88 tok/s** | **5.88 tok/s** | **5.89 tok/s** |
| Decode average hardware throughput | 13.42 GFLOPS | 18.78 GFLOPS | 18.85 GFLOPS | 30.45 GFLOPS | **33.82 GFLOPS** | **33.97 GFLOPS** | **33.98 GFLOPS** | **34.35 GFLOPS** |
| Vision program section | 3.58 MiB | 3.22 MiB | 2.39 MiB | / | 3.22 MiB | 2.39 MiB | 1.90 MiB | 1.75 MiB |
| Prefill program section | 5.71 MiB | 3.08 MiB | 3.46 MiB | / | 3.08 MiB | 3.46 MiB | 3.39 MiB | 3.41 MiB |
| Decode program section | | 1.42 MiB | 1.46 MiB | | 1.42 MiB | 1.46 MiB | 1.42 MiB | 1.42 MiB |
| Combined program image | 10.37 MiB | 28.45 MiB | 10.81 MiB | / | 28.45 MiB | 10.81 MiB | 27.44 MiB | 26.92 MiB |
| Weight image (`params.bin`) | 6.91 GiB | same | same | / | same | same | same | same |
| Correctness |  | coherent, total 724 | coherent, total 709 | not recorded | coherent, total 724 | coherent, total 709 | coherent, total 709 | coherent, total 709 |

The four Alveo columns were refreshed on 2026-08-17 against HW version
`0x6bb5d25d`, with `make clean` before every run. All four produced coherent
Yosemite descriptions and reached the stop token. Worker attention scratch now
aliases output-only regions that are dead during attention and fully overwritten
before later reads, allowing the complete eight-engine image path to fit in the
32-bit DRAM window without overlapping vision weights or worker ISA.

The Kintex 2-core column is from a 2026-08-17 `--multi-core` run with vision
attention heads sharded across both engines, replacing the older projection-only
multicore measurements. This nearly halves the vision encoder (53.78 → 28.09 s)
and substantially reduces prefill (51.52 → 31.78 s) versus single-core. The
current run produced a coherent Yosemite description and reached the stop token
on the accepted `0x3d04c689` bitstream. The combined program image includes both
engines; the vision and prefill section rows report the master sections.

## Refactor optimizations

- Split the control path into dedicated vision, LM, and audio modules with a
  small unified-engine orchestrator. Weight initialization, tensor
  initialization, compile, and run phases now have explicit ownership.
- Capture patch embedding and all 16 vision layers as one straight-line FPGA
  program. This removes per-layer host dispatch and shrinks the vision ISA by
  15.3%.
- Execute prefill at the actual aligned request length instead of performing
  all compute over the legacy 512-token template. For this 272-token request,
  that is the primary source of the 2.18x prefill speedup.
- Move per-layer input preparation to compact host-backed/memory-mapped data
  and upload only the rows needed by the active request.
- Use streaming prefill and decode kernels with dynamic PBI dimensions and
  inline unified attention call chains.
- Replace the older vision permutation staging path with direct
  `bf16_permute_dram_core` operation and keep generated vision constants in
  stable parameter memory.
- Store named vision and LM sections in a combined manifest so an unchanged
  section can be reused independently. The LM section is reduced despite a
  slightly larger decoder section.
- Compact the LM KV cache to 15 unique shared slots. The refactor reports 18
  MiB for K+V and 313 MiB total tensor DRAM use.
- Memory-map the 4.38 GiB per-layer host embedding table instead of eagerly
  materializing it. The complete shared weight image remains 6.91 GiB.
- Add checkpointed profile images for vision, prefill, and decode without
  changing the normal execution image.

Profile tables below are fresh `--profile` runs (`--dev xdma1`, 2026-08-17), rows
in **compile order**. `--multi-core 2` uses per-phase multi-core profiling: the
master's per-segment HW latency counter, with checkpoints at the sharded-region
boundaries so each region's segment measures its fork-to-join wall-time.
**Vision** now shards three regions — Projection, **Attention (head-sharded)**,
and Post-attention — so all three ~halve (this is the change behind the faster
28.09 s Kintex 2-core vision; the old table had vision attention engine-invariant).
**Prefill** shards only its two matmul-heavy regions (QKV/V + MLP); its attention
and per-layer prep stay master-only. The norm/RoPE/permute/gather phases are
master-only and engine-invariant in both stages. Decode is single-engine (never
sharded), so its multi-core breakdown equals single-core.

## Profile backup: vision encoder

Vision multi-core segments tile cleanly (they sum to the single-shot master total),
so the per-phase `--multi-core 2` numbers are exact.

| Phase | Single-core | --multi-core 2 |
|---|---:|---:|
| Patch embedding | 123.6 ms | 123.6 ms |
| Projection (Q/K/V) † | 6,682.4 ms | **3,361.0 ms** |
| RoPE | 1,072.8 ms | **740.0 ms** |
| Permute | 269.6 ms | 269.6 ms |
| Attention † | 16,642.2 ms | **8,664.7 ms** |
| Post-attention (O + MLP) † | 28,911.4 ms | **14,541.1 ms** |
| Pooler tail | 75.3 ms | 75.3 ms |
| Tail | 0.0 ms | 0.0 ms |
| **Total** | **53,777.3 ms** | **27,775 ms** |

† row-sharded across 2 engines

## Profile backup: LM prefill

Prefill folds O projection into the MLP phase so single-core matches the multi-core
O+MLP sharded region. Only **QKV/V projection** and **MLP (incl. O)** are sharded
(†) — the rest are master-only and engine-invariant. Under two-engine the master's
per-segment counter mis-times the two *long* master-only phases (per-layer prep,
attention), so those rows show their single-core value (which is the true
multi-core value — they run identically on the master). The reconstructed
multi-core total (31,782 ms) matches the single-shot master counter (31,782.7 ms
from the Kintex 2-core prefill run), confirming it.

| Phase | Single-core | --multi-core 2 |
|---|---:|---:|
| Per-layer preparation ‡ | 596.0 ms | 596.0 ms |
| QKV/V projection † | 3,302.5 ms | **1,682.3 ms** |
| RoPE | 162.6 ms | 162.6 ms |
| KV gather | 70.0 ms | 70.0 ms |
| Attention ‡ | 9,268.7 ms | 9,268.7 ms |
| MLP (incl. O projection) † | 37,426.6 ms | **19,309.4 ms** |
| Injection | 694.2 ms | 694.2 ms |
| Tail halt | 0.0 ms | 0.0 ms |
| **Total** | **51,520.6 ms** | **31,782.4 ms** |

† row-sharded across 2 engines (these ~halve).

## Profile backup: one decode token at position 272

Decode is single-engine; `--multi-core 2` is identical (245.3 ms total).

| Phase | Single-core |
|---|---:|
| Per-layer preparation | 5.6 ms |
| QKV/V projection | 13.2 ms |
| RoPE | 1.0 ms |
| KV gather | 0.1 ms |
| Attention | 30.5 ms |
| O projection | 11.8 ms |
| MLP | 135.8 ms |
| Injection | 12.6 ms |
| LM head | 34.8 ms |
| Tail increment | 0.0 ms |
| **Total** | **245.3 ms** |

The profiled single-token throughput is ~4.08 tok/s. Checkpoint HALTs add
instrumentation control flow, so the refreshed normal-run average (3.74 tok/s
single-core, 3.75 tok/s Kintex 2-core) remains the end-to-end decode result to use for
user-visible throughput.
