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

| Metric | Kintex | Kintex 2-core | rk | Alveo | Alveo 2-core | Alveo 4-core | Alveo 8-core |
|---|---:|---:|---:|---:|---:|---:|---:|
| Peak throughput | 25.39 GFLOPS | 50.77 GFLOPS | 42.67 GFLOPS | 46.93 GFLOPS | 93.87 GFLOPS | 187.73 GFLOPS | 375.47 GFLOPS |
| Vision FPGA execution | 53.78 s | 27.78 s | 32.05 s | **29.12 s** | **15.43 s** | **8.08 s** | **4.95 s** |
| Vision end-to-end path | 54.43 s | 28.54 s | 32.62 s | **29.66 s** | **16.16 s** | **9.22 s** | **7.03 s** |
| Vision throughput | 21.37 GFLOPS | 41.41 GFLOPS | 35.85 GFLOPS | **39.46 GFLOPS** | **74.47 GFLOPS** | **142.21 GFLOPS** | **232.17 GFLOPS** |
| — utilization (% peak) | 84.2% | 81.6% | 84.0% | **84.1%** | **79.3%** | **75.8%** | **61.8%** |
| Vision soft tokens | 256 | 256 | 256 | 256 | 256 | 256 | 256 |
| LM prefill sequence executed | 272 actual tokens | 272 actual tokens | 272 actual tokens | 272 actual tokens | 272 actual tokens | 272 actual tokens | 272 actual tokens |
| LM prefill FPGA latency | 43.95 s | 23.46 s | 28.02 s | **24.92 s** | **12.84 s** | **6.94 s** | **4.02 s** |
| LM prefill throughput | 24.08 GFLOPS | 45.13 GFLOPS | 37.78 GFLOPS | **42.48 GFLOPS** | **82.47 GFLOPS** | **152.64 GFLOPS** | **263.39 GFLOPS** |
| — utilization (% peak) | 94.8% | 88.9% | 88.5% | **90.5%** | **87.9%** | **81.3%** | **70.1%** |
| Decode average throughput | 3.75 tok/s | 3.76 tok/s | 5.85 tok/s | **6.59 tok/s** | **6.52 tok/s** | **6.56 tok/s** | **6.56 tok/s** |
| Decode peak first-token throughput | 4.00 tok/s | 4.00 tok/s | 6.21 tok/s | **6.95 tok/s** | **6.98 tok/s** | **6.94 tok/s** | **6.93 tok/s** |
| Decode average hardware throughput | 18.97 GFLOPS | 22.95 GFLOPS | 36.47 GFLOPS | **40.77 GFLOPS** | **40.56 GFLOPS** | **40.56 GFLOPS** | **40.57 GFLOPS** |
| — utilization (% peak) | 74.7% | 45.2% | 85.5% | **86.9%** | **43.2%** | **21.6%** | **10.8%** |
| Vision program section | 3.22 MiB | 2.39 MiB | 3.22 MiB | 3.22 MiB | 2.39 MiB | 1.90 MiB | 1.75 MiB |
| Prefill program section | 5.85 MiB | 4.77 MiB | 5.85 MiB | 5.85 MiB | 4.77 MiB | 4.06 MiB | 3.72 MiB |
| Decode program section | 1.42 MiB | 1.42 MiB | 1.42 MiB | 1.42 MiB | 1.42 MiB | 1.42 MiB | 1.42 MiB |
| Combined program image | 10.49 MiB | 13.73 MiB | 10.49 MiB | 10.49 MiB | 10.81 MiB | 19.20 MiB | 30.40 MiB |
| Weight image (`params.bin`) | 6.91 GiB | same | same | same | same | same | same |
| Correctness | coherent, total 656 | coherent, total 699 | coherent, total 656 | coherent, total 656 | coherent, total 699 | coherent, total 699 | coherent, total 699 |

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

## Profile backup: vision encoder (kintex7)

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

## Profile backup: LM prefill (kintex7)

| Phase | Single-core | --multi-core 2 | Single-core GQA | --multi-core 2 GQA |
|---|---:|---:|---:|---:|
| Per-layer preparation ‡ | 596.0 ms | 596.0 ms | 596.0 ms | 0.0 ms |
| QKV/V projection † | 3,302.5 ms | **1,682.3 ms** | 3,302.5 ms | **1,693.6 ms** |
| RoPE | 162.6 ms | 162.6 ms | 162.6 ms | 162.5 ms |
| KV gather | 70.0 ms | 70.0 ms | 0.1 ms | 0.1 ms |
| Q permute | — | — | 75.2 ms | 75.3 ms |
| Attention † | 9,268.7 ms | 9,268.7 ms | 1,697.2 ms | **906.3 ms** |
| MLP (incl. O projection) † | 37,426.6 ms | **19,309.4 ms** | 37,426.7 ms | **19,331.5 ms** |
| Injection | 694.2 ms | 694.2 ms | 694.2 ms | 694.2 ms |
| Tail halt | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| **Total** | **51,520.6 ms** | **31,782.4 ms** | **43,954.5 ms** | **22,863.3 ms** |

† row-sharded across 2 engines (these ~halve).

## Profile backup: one decode token at position 272 (kintex7) no multicore usecase in decoder

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
