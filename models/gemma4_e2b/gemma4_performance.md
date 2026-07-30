# Gemma4 E2B Performance

## Baseline

Measured on p620 using the Kintex-7 target (`xdma0`, 198.3256 MHz), IF4 LM
weights, the default Yosemite image, and a 1,024-token context. Host vision was
used to isolate LM performance:

```bash
python models/gemma4_e2b/gemma4_e2b_refactor.py --image --vision-host
python models/gemma4_e2b/gemma4_e2b_refactor.py --image --vision-host --profile
python models/gemma4_e2b/gemma4_e2b_refactor.py --image --vision-host \
  --prefill-kernel matmatmul --decode-kernel streaming
python models/gemma4_e2b/gemma4_e2b_refactor.py --image --vision-host \
  --prefill-kernel streaming --decode-kernel streaming
```

The VLM prompt contains 273 tokens: 256 image soft tokens and 17 text/special
tokens. Prefill executes 272 tokens.

| Measurement | Baseline | Matmatmul prefill + streaming decode | Streaming prefill + streaming decode |
|---|---:|---:|---:|
| Host vision, complete path | 6.90 s | 6.94 s | 6.89 s |
| Host vision, model forward | 4.38 s | 4.37 s | 4.38 s |
| LM profile compile | 0.60 s | 0.61 s | 0.60 s |
| Prefill host preparation | 0.08 s | — | — |
| Prefill FPGA execution | 51.948 s | 51.947 s | 51.332 s |
| Prefill wall time | 52.11 s | 52.11 s | 51.49 s |
| Prefill throughput | 5.22 tok/s | 5.22 tok/s | 5.28 tok/s |
| Prefill FPGA throughput | 23.46 GFLOPS | 23.46 GFLOPS | 23.75 GFLOPS |
| Decode first-token wall speed | 2.84 tok/s | 3.27 tok/s | 3.27 tok/s |
| Decode first-token FPGA speed | 2.95 tok/s | — | — |
| Decode average wall speed | 2.59 tok/s | 2.96 tok/s | 2.93 tok/s |
| Decode average FPGA throughput | 13.29 GFLOPS | 15.32 GFLOPS | 15.13 GFLOPS |

Baseline kernel setup:

- Prefill: two-pass `matmat_mul_core` for all quantized projections.
- Decode: streaming `quantized_matmat_core` for Q/K/V, O, MLP gate, and MLP up;
  two-pass `matmat_mul_core` for MLP down and LM head.

The **Matmatmul prefill + streaming decode** run preserves the baseline
two-pass prefill while accelerating decode with streaming projections, except
for wide MLP-down operations with `K=12,288`, which retain the two-pass core.

The **Streaming prefill + streaming decode** run additionally converts all
supported prefill projections to the one-pass streaming core while retaining
the same `K=12,288` wide MLP-down fallback in both phases.

Profiled FPGA phase breakdown:

| Phase | Prefill | Decode, one token |
|---|---:|---:|
| MLP | 35.401 s (68.1%) | 180.49 ms (53.3%) |
| Attention | 9.257 s (17.8%) | 30.40 ms (9.0%) |
| QKV/V projection | 3.332 s (6.4%) | 13.11 ms (3.9%) |
| O projection | 3.048 s (5.9%) | 11.83 ms (3.5%) |
| LM head | — | 70.26 ms (20.7%) |
| KV gather | 64.57 ms (0.1%) | 20.48 ms (6.0%) |
| Per-layer injection | 686.18 ms (1.3%) | 11.34 ms (3.3%) |
| RoPE | 160.11 ms (0.3%) | 0.97 ms (0.3%) |
| **Total** | **51.948 s** | **338.89 ms** |
