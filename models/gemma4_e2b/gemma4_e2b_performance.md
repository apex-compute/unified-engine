# Gemma4 E2B performance

## Test setup

Measurements were collected on `p620` with the Kintex-7 bitstream
`0x52a71442`, XDMA device `xdma0`, and a 5.042213410674164 ns clock period.
Both implementations used `test_samples/yosemite.jpg`, the prompt
`Describe this image in detail.`, the same `params.bin`, IF4 projection
weights, 35 LM layers, 16 vision layers, and 256 image soft tokens.

Commands:

```bash
python models/gemma4_e2b/gemma4_e2b_test.py --image 
python models/gemma4_e2b/gemma4_e2b_test.py --image --profile
python models/gemma4_e2b/gemma4_e2b_test.py --image --prefill-kernel matmatmul
python models/gemma4_e2b/gemma4_e2b_test.py --image --multi-core
python models/gemma4_e2b/gemma4_e2b_test.py --image --multi-core --profile
```

## Performance comparison

| Metric | Legacy | Refactored streaming prefill | Refactored matmatmul prefill | Multicore (2 engines) | rk |
|---|---:|---:|---:|---:|---:|
| Vision FPGA execution | 55.54 s | 54.60 s | 54.85 s | 37.20 s | 32.56 s |
| Vision end-to-end path | not reported | 55.81 s | 56.10 s | 38.77 s | 33.03 s |
| Vision throughput | not reported | 20.86 GFLOPS | 20.76 GFLOPS | 30.63 GFLOPS | 35.00 GFLOPS |
| Vision soft tokens | 256 | 256 | 256 | 256 | 256 |
| LM prefill sequence executed | 512-token template | 272 actual tokens | 272 actual tokens | 272 actual tokens | 273 actual tokens |
| LM prefill FPGA latency | 113.571 s | 51.63 s | 52.578 s | 31.93 s | 32.544 s |
| LM prefill useful-work throughput | 23.20 GFLOPS | 23.94 GFLOPS | 23.45 GFLOPS | 38.80 GFLOPS | 37.89 GFLOPS |
| Decode average throughput | 2.68 tok/s | 3.82 tok/s | 3.85 tok/s | 3.60 tok/s | 6.00 tok/s |
| Decode peak first-token throughput | 2.86 tok/s | 4.07 tok/s | 4.01 tok/s | 4.02 tok/s | 6.23 tok/s |
| Decode average hardware throughput | 13.42 GFLOPS | 19.11 GFLOPS | 19.20 GFLOPS | 17.96 GFLOPS | 30.45 GFLOPS |
| Vision program section | 3.58 MiB | 3.03 MiB | 3.03 MiB | 4.09 MiB incl. worker | / |
| Prefill program section | 5.71 MiB | 3.10 MiB | 3.21 MiB | 3.44 MiB incl. worker | / |
| Combined program image | 10.37 MiB | 7.54 MiB | 7.67 MiB | 8.95 MiB | / |
| Weight image (`params.bin`) | 6.91 GiB | same | same | same | / |

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

Profile tables below are fresh `--profile` runs (`--dev xdma1`), rows in
**compile order**. `--multi-core 2` uses per-phase multi-core profiling: the
master's per-segment HW latency counter, with checkpoints at the sharded-region
boundaries so each region's segment measures its fork-to-join wall-time. Each stage
row-shards its **two matmul-heavy projection regions** across the two engines
(vision: Projection + Post-attention; prefill: QKV/V + MLP) — these ~halve; the
norm/RoPE/attention phases are master-only and engine-invariant. Decode is
single-engine (never sharded), so its multi-core breakdown equals single-core.

## Profile backup: vision encoder

Vision multi-core segments tile cleanly (they sum to the single-shot master total),
so the per-phase `--multi-core 2` numbers are exact.

| Phase | Single-core | --multi-core 2 |
|---|---:|---:|
| Patch embedding | 123.6 ms | 123.6 ms |
| Projection (Q/K/V) † | 6,682.5 ms | **3,359.2 ms** |
| RoPE gather | 2,520.5 ms | 2,520.5 ms |
| Attention | 16,642.2 ms | 16,642.3 ms |
| Post-attention (O + MLP) † | 28,911.9 ms | **14,540.7 ms** |
| Pooler tail | 75.3 ms | 75.3 ms |
| Tail | 0.0 ms | 0.0 ms |
| **Total** | **54,956.0 ms** | **37,261.6 ms** |

† row-sharded across 2 engines (the two fork-join regions); these ~halve. The
master-only phases are unchanged, as expected.

## Profile backup: LM prefill

Prefill folds O projection into the MLP phase so single-core matches the multi-core
O+MLP sharded region. Only **QKV/V projection** and **MLP (incl. O)** are sharded
(†) — the rest are master-only and engine-invariant. Under two-engine the master's
per-segment counter mis-times the two *long* master-only phases (per-layer prep,
attention), so those rows show their single-core value (which is the true
multi-core value — they run identically on the master). The reconstructed
multi-core total (31,782 ms) matches the single-shot master counter (31,781.9 ms),
confirming it.

| Phase | Single-core | --multi-core 2 |
|---|---:|---:|
| Per-layer preparation ‡ | 596.0 ms | 596.0 ms |
| QKV/V projection † | 3,302.5 ms | **1,681.8 ms** |
| RoPE | 162.6 ms | 162.6 ms |
| KV gather | 70.0 ms | 70.0 ms |
| Attention ‡ | 9,268.7 ms | 9,268.7 ms |
| MLP (incl. O projection) † | 37,426.6 ms | **19,309.3 ms** |
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
instrumentation control flow, so the normal-run ~3.7 tok/s average remains the
end-to-end decode result to use for user-visible throughput.
