# Gemma4 E2B performance

## Test setup

Measurements were collected on `p620` with the Kintex-7 bitstream
`0x52a71442`, XDMA device `xdma0`, and a 5.042213410674164 ns clock period.
Both implementations used `test_samples/yosemite.jpg`, the prompt
`Describe this image in detail.`, the same `params.bin`, IF4 projection
weights, 35 LM layers, 16 vision layers, and 256 image soft tokens.

Commands:

```bash
python models/gemma4_e2b/gemma4_e2b_test.py --image --dev xdma0
python models/gemma4_e2b/gemma4_e2b_test.py --image --profile --dev xdma0
```

## Performance comparison

| Metric | Legacy | Refactored |
|---|---:|---:|
| Vision FPGA execution | 55.54 s | 54.60 s |
| Vision end-to-end path | not reported | 55.81 s |
| Vision throughput | not reported | 20.86 GFLOPS |
| Vision soft tokens | 256 | 256 |
| LM prefill sequence executed | 512-token template | 272 actual tokens |
| LM prefill FPGA latency | 113.571 s | 51.932 s |
| LM prefill useful-work throughput | 23.20 GFLOPS | 23.75 GFLOPS |
| Decode average throughput | 2.68 tok/s | 3.88 tok/s |
| Decode average hardware throughput | 13.42 GFLOPS | 19.22 GFLOPS |
| Decode wall time | 147.50 s / 396 tokens | 80.91 s / 314 tokens |
| Vision program section | 3.58 MiB | 3.03 MiB |
| Prefill program section | 5.71 MiB | 3.10 MiB |
| Combined program image | 10.37 MiB | 7.54 MiB |
| Weight image (`params.bin`) | 6.91 GiB | same |

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

## Profile backup: vision encoder

| Phase | Total latency | Share | Count | Per layer |
|---|---:|---:|---:|---:|
| Post-attention | 28,860.249 ms | 52.8% | 16 | 1,803.7656 ms |
| Attention | 16,616.577 ms | 30.4% | 16 | 1,038.5361 ms |
| Projection | 6,674.010 ms | 12.2% | 16 | 417.1256 ms |
| RoPE gather | 2,358.445 ms | 4.3% | 16 | 147.4028 ms |
| Patch embedding | 123.501 ms | 0.2% | 1 | 123.5009 ms |
| Tail | 0.003 ms | 0.0% | 1 | 0.0029 ms |
| **Total** | **54,632.785 ms** | **100%** | | |

Vision host wall-clock was 58.22 s: 54.73 s FPGA execution (94.0%), 2.38 s
HF model load (4.1%), 0.52 s host instruction emission (0.9%), 0.26 s setup,
0.21 s readback, 0.11 s preprocessing, and 0.01 s pool/project.

## Profile backup: LM prefill

| Phase | Total latency | Share | Count | Per layer |
|---|---:|---:|---:|---:|
| MLP | 34,935.889 ms | 67.2% | 35 | 998.1683 ms |
| Attention | 9,345.133 ms | 18.0% | 35 | 267.0038 ms |
| QKV/V projection | 3,298.324 ms | 6.3% | 35 | 94.2378 ms |
| O projection | 2,930.012 ms | 5.6% | 35 | 83.7146 ms |
| Injection | 694.165 ms | 1.3% | 35 | 19.8333 ms |
| Per-layer preparation | 595.937 ms | 1.1% | 1 | 595.9371 ms |
| RoPE | 156.245 ms | 0.3% | 35 | 4.4641 ms |
| KV gather | 64.567 ms | 0.1% | 35 | 1.8448 ms |
| Tail halt | 0.003 ms | 0.0% | 1 | 0.0029 ms |
| **Total** | **52,020.275 ms** | **100%** | | |

## Profile backup: one decode token at position 272

| Phase | Total latency | Share | Count | Per layer |
|---|---:|---:|---:|---:|
| MLP | 135.547 ms | 55.4% | 35 | 3.8728 ms |
| LM head | 34.761 ms | 14.2% | 1 | 34.7609 ms |
| Attention | 30.397 ms | 12.4% | 35 | 0.8685 ms |
| QKV/V projection | 13.111 ms | 5.4% | 35 | 0.3746 ms |
| Injection | 12.604 ms | 5.1% | 35 | 0.3601 ms |
| O projection | 11.746 ms | 4.8% | 35 | 0.3356 ms |
| Per-layer preparation | 5.548 ms | 2.3% | 1 | 5.5482 ms |
| RoPE | 0.970 ms | 0.4% | 35 | 0.0277 ms |
| KV gather | 0.102 ms | 0.0% | 35 | 0.0029 ms |
| Tail increment | 0.003 ms | 0.0% | 1 | 0.0029 ms |
| **Total** | **244.788 ms** | **100%** | | |

The profiled single-token throughput is 4.09 tok/s. Checkpoint HALTs add
instrumentation control flow, so the normal-run 3.88 tok/s average remains the
end-to-end decode result to use for user-visible throughput.
