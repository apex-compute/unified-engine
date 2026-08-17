# Gemma4 E2B performance

## Test setup

Measurements were collected on `p620` with the Kintex-7 bitstream
`0x52a71442`, XDMA device `xdma0`, and a 5.042213410674164 ns clock period.
Both implementations used `test_samples/yosemite.jpg`, the prompt
`Describe this image in detail.`, the same `params.bin`, IF4 projection
weights, 35 LM layers, 16 vision layers, and 256 image soft tokens.

> **2026-08-17 refresh (Kintex + Kintex 2-core columns and all profile tables).**
> These were re-measured on `--dev xdma1`, whose flashed bitstream reports HW
> version `0xbb859676` — this does **not** match the version this software
> expects (`0xcf133b89`, `update_cf133b89.bin`). The `hw_version` assertion in
> `user_dma_core.init_unified_engine` was temporarily commented out to let the
> runs proceed against the on-board bitstream. Treat the refreshed numbers, and
> especially the Kintex 2-core correctness result, with that mismatch in mind;
> the other columns (Legacy, Refactored, rk, Alveo*) are unchanged from the
> original setup above. Each of the four runs was preceded by `make clean`
> (`clean_program_bins.sh`) so every program image was recompiled from scratch.

Commands:

```bash
python models/gemma4_e2b/gemma4_e2b_test.py --dev xdma1 --image 
python models/gemma4_e2b/gemma4_e2b_test.py --dev xdma1 --image --profile
python models/gemma4_e2b/gemma4_e2b_test.py --dev xdma1 --image --prefill-kernel matmatmul
python models/gemma4_e2b/gemma4_e2b_test.py --dev xdma1 --image --multi-core
python models/gemma4_e2b/gemma4_e2b_test.py --dev xdma1 --image --multi-core --profile
python models/gemma4_e2b/gemma4_e2b_test.py --device alveo --image
python models/gemma4_e2b/gemma4_e2b_test.py --device alveo --image --multi-core
```

## Performance comparison

| Metric | Legacy | Kintex | Refactored matmatmul prefill | Kintex 2-core | rk | Alveo | Alveo 2-core | Alveo 4-core (unvalidated) | Alveo 8-core (unvalidated) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vision FPGA execution | 55.54 s | 53.78 s | 54.85 s | **28.09 s** | 32.56 s | 36.15 s | **19.89 s** | **10.96 s** | **7.19 s** |
| Vision end-to-end path | not reported | 54.32 s | 56.10 s | **28.83 s** | 33.03 s | 36.66 s | **20.56 s** | **12.00 s** | **8.86 s** |
| Vision throughput | not reported | 21.37 GFLOPS | 20.76 GFLOPS | **40.91 GFLOPS** | 35.00 GFLOPS | 26.30 GFLOPS | **47.82 GFLOPS** | **86.78 GFLOPS** | **132.27 GFLOPS** |
| Vision soft tokens | 256 | 256 | 256 | 256 | 256 | 256 | 256 | 256 | 256 |
| LM prefill sequence executed | 512-token template | 273 actual tokens | 272 actual tokens | 273 actual tokens | 273 actual tokens | 272 actual tokens | 272 actual tokens | 272 actual tokens | 272 actual tokens |
| LM prefill FPGA latency | 113.571 s | 51.52 s | 52.578 s | **31.78 s** | 32.544 s | 34.54 s | **21.09 s** | **14.27 s** | **10.88 s** |
| LM prefill useful-work throughput | 23.20 GFLOPS | 23.94 GFLOPS | 23.45 GFLOPS | 38.80 GFLOPS | 37.89 GFLOPS | 29.79 GFLOPS | **48.84 GFLOPS** | **72.31 GFLOPS** | **95.23 GFLOPS** |
| Decode average throughput | 2.68 tok/s | 3.75 tok/s | 3.85 tok/s | **3.83 tok/s** | 6.00 tok/s | 5.60 tok/s | 5.62 tok/s | 5.68 tok/s | 5.46 tok/s |
| Decode peak first-token throughput | 2.86 tok/s | 4.01 tok/s | 4.01 tok/s | **4.00 tok/s** | 6.23 tok/s | 5.91 tok/s | 5.90 tok/s | 5.79 tok/s | 5.75 tok/s |
| Decode average hardware throughput | 13.42 GFLOPS | 18.78 GFLOPS | 19.20 GFLOPS | **19.31 GFLOPS** | 30.45 GFLOPS | 23.61 GFLOPS | 23.68 GFLOPS | 23.91 GFLOPS | 23.05 GFLOPS |
| Vision program section | 3.58 MiB | 3.22 MiB | 3.03 MiB | **2.32 MiB (master)** | / | 3.05 MiB | 4.15 MiB incl. worker | 6.18 MiB incl. workers | not reported |
| Prefill program section | 5.71 MiB | 3.08 MiB | 3.21 MiB | **3.38 MiB (master)** | / | 3.08 MiB | 4.70 MiB incl. worker | 7.37 MiB incl. workers | not reported |
| Combined program image | 10.37 MiB | 7.73 MiB | 7.67 MiB | **7.13 MiB (master)** | / | 7.55 MiB | 10.27 MiB | 14.98 MiB | not reported |
| Weight image (`params.bin`) | 6.91 GiB | same | same | same | / | same | same | same | same |
| Correctness | legacy baseline | coherent | coherent | **failed: unrelated (barcode)** | not recorded | coherent; stop token | coherent; stop token | **failed: unrelated description** | **failed: unrelated description** |

The Alveo one- and two-core runs produced coherent Yosemite descriptions and
reached the stop token. The experimental four- and eight-core columns record
measured execution numbers, but both decoded unrelated image descriptions and
are therefore explicitly marked unvalidated. See `gemma4_e2b_alveo_4core.log`
and `gemma4_e2b_alveo_8core.log` for those correctness failures.

The Kintex 2-core column is from a 2026-08-17 `--multi-core` run with vision
attention heads sharded across both engines, replacing the older projection-only
multicore measurements. This refresh nearly halves the vision encoder (53.78 →
28.09 s) and prefill (51.52 → 31.78 s) versus single-core, but its **decode was
incoherent**: with the same Yosemite image that single-core described correctly,
the two-engine run produced an unrelated description of a "barcode" (reaching the
stop token). This is a correctness regression on the sharded path, observed here
on the mismatched `0xbb859676` bitstream (see the 2026-08-17 refresh note above),
so it is not yet clear whether the fault is in the head-sharded attention path or
an artifact of the bitstream/software version mismatch — it needs re-confirmation
on the matching `0xcf133b89` bitstream before the sharded path can be trusted.
The 2-core program-section sizes above are the **master** engine's stored sections
from this run (the sharded design splits work per engine, so the master program
shrank); the worker sections were not separately itemized in the run log, so these
are not directly comparable to the older "incl. worker" Alveo figures.

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
| Projection (Q/K/V) † | 6,682.4 ms | **3,360.0 ms** |
| RoPE | 1,072.8 ms | 1,072.8 ms |
| Permute | 269.6 ms | 269.6 ms |
| Attention † | 16,642.2 ms | **8,645.1 ms** |
| Post-attention (O + MLP) † | 28,911.4 ms | **14,539.1 ms** |
| Pooler tail | 75.3 ms | 75.3 ms |
| Tail | 0.0 ms | 0.0 ms |
| **Total** | **53,777.3 ms** | **28,085.5 ms** |

† row-sharded across 2 engines (Projection, Attention head-sharded, and
Post-attention); these ~halve. The master-only phases (patch embed, RoPE,
permute, pooler tail) are unchanged, as expected. The multi-core total
(28,085.5 ms) matches the Kintex 2-core vision FPGA execution (28.09 s) above.

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

† row-sharded across 2 engines (these ~halve). ‡ master-only; the profiler's
per-segment counter mis-times these two under two-engine, so the (identical)
single-core value is shown.

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
instrumentation control flow, so the normal-run average (3.75 tok/s single-core,
3.83 tok/s Kintex 2-core) remains the end-to-end decode result to use for
user-visible throughput.
