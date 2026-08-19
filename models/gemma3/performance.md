# Gemma3 Performance

Baseline numbers captured from the per-run summaries written by
`gemma3_test.py` / `gemma3_test_IF8.py` (see `write_run_summary`). Both runs
compiled the program image **from scratch** (default; `--bin-reuse` off).

- **HW version:** `0x3d04c689`
- **--dev / --device:** `xdma1` / `kintex7`
- **Clock / frequency:** 5.0422 ns (198.3 MHz)
- **Cores:** 1
- **Prompt:** `x+3=5, what is x?`
- **Architecture:** 26 layers, hidden 1152, head_dim 256, group_size 4, MLP 6912,
  global-RoPE layers [5, 11, 17, 23].
- **Source summaries:** [gemma3_test_xdma1_kintex7.md](gemma3_test_xdma1_kintex7.md) (IF4),
  [gemma3_test_IF8_xdma1_kintex7.md](gemma3_test_IF8_xdma1_kintex7.md) (IF8).

## Gemma3 IF4

| Metric | Baseline | proper-gqa |
|---|---:|---:|
| Peak throughput | 25.39 GFLOPS | 25.39 GFLOPS |
| Weight bin (`weights_gemma3_hf.bin`) | 1083.77 MB | 1083.77 MB |
| Weight DRAM (quantized, on FPGA) | 507.8 MB | 507.8 MB |
| Prefill program | 53.6 KB | 88.9 KB |
| Decoder program | 66.6 KB | 66.6 KB |
| Combined program image | 120.2 KB | 155.6 KB |
| Prefill tokens (seq_len) | 19 | 19 |
| Prefill FPGA time (HW latency) | 1118.80 ms | 1128.10 ms |
| Prefill throughput | ~ GFLOPS | 23.65 GFLOPS |
| Prefill end-to-end (CPU) | 1.12 s | 1.13 s |
| Decoded tokens | 80 generated (total 99) | 76 generated (total 95) |
| Decode 1st-token speed (peak) | 10.60 tok/s | 10.60 tok/s |
| Decode average speed | 10.29 tok/s | 10.32 tok/s |
| Decode average throughput | 21.50 GFLOPS | 21.52 GFLOPS |
| Decode 1st-token HW latency | 94.4 ms/tok | 94.4 ms/tok |
| Decode average HW latency | 95.6 ms/tok | 95.5 ms/tok |
| Correctness | coherent (solves x = 2) | coherent (solves x = 2) |

## Gemma3 IF8

| Metric | Baseline | proper-gqa |
|---|---:|---:|
| Peak throughput | 25.39 GFLOPS | 25.39 GFLOPS |
| Weight bin (`weights_gemma3_hf.bin`) | 1560.49 MB | 1560.49 MB |
| Weight DRAM (quantized, on FPGA) | 984.5 MB | 984.5 MB |
| Prefill program | 53.6 KB | 88.9 KB |
| Decoder program | 66.6 KB | 66.6 KB |
| Combined program image | 120.2 KB | 155.6 KB |
| Prefill tokens (seq_len) | 19 | 19 |
| Prefill FPGA time (HW latency) | 2163.20 ms | 2172.52 ms |
| Prefill throughput | ~ GFLOPS | 12.28 GFLOPS |
| Prefill end-to-end (CPU) | 2.17 s | 2.18 s |
| Decoded tokens | 74 generated (total 93) | 74 generated (total 93) |
| Decode 1st-token speed (peak) | 5.78 tok/s | 5.78 tok/s |
| Decode average speed | 5.69 tok/s | 5.69 tok/s |
| Decode average throughput | 11.80 GFLOPS | 11.80 GFLOPS |
| Decode 1st-token HW latency | 173.1 ms/tok | 173.1 ms/tok |
| Decode average HW latency | 174.3 ms/tok | 174.3 ms/tok |
| Correctness | coherent (solves x = 2) | coherent (solves x = 2) |
