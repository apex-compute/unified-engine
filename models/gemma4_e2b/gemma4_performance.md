# Gemma4 E2B Performance

## Performance

Measured on p620 using the Kintex-7 target (`xdma0`, 198.3256 MHz), IF4 LM
weights, the default Yosemite image, and a 1,024-token context. Host vision was
used to isolate LM performance:

```bash
python models/gemma4_e2b/gemma4_e2b_refactor.py --image --vision-host
python models/gemma4_e2b/gemma4_e2b_refactor.py --image --vision-host --profile
```

The VLM prompt contains 273 tokens: 256 image soft tokens and 17 text/special
tokens. Prefill executes 272 tokens.

| Measurement | Streaming MLP down | Compact KV direct attention | FPGA per-layer preparation |
|---|---:|---:|---:|
| Host vision, complete path | 6.89 s | 6.93 s | 6.87 s |
| Host vision, model forward | 4.38 s | 4.38 s | 4.38 s |
| LM profile compile | 0.60 s | 0.59 s | 0.94 s |
| Prefill host preparation | — | 0.08 s | 0.06 s |
| Prefill FPGA execution | 51.332 s | 51.332 s | 51.653 s |
| Prefill wall time | 51.49 s | 51.47 s | 51.76 s |
| Prefill throughput | 5.28 tok/s | 5.28 tok/s | 5.27 tok/s |
| Prefill FPGA throughput | 23.75 GFLOPS | 23.75 GFLOPS | 23.74 GFLOPS |
| Decode first-token wall speed | 3.67 tok/s | 4.00 tok/s | 4.04 tok/s |
| Decode first-token FPGA speed | 3.86 tok/s | 4.20 tok/s | 4.11 tok/s |
| Decode average wall speed | 3.29 tok/s | 3.69 tok/s | 3.82 tok/s |
| Decode average FPGA throughput | 17.10 GFLOPS | 19.29 GFLOPS | 18.97 GFLOPS |

The **Streaming MLP down** run moves the decoder's wide `K=12,288` MLP-down
projection from the two-pass core to the streaming core.

The **Compact KV direct attention** run retains decoder residual fusion and
stores each unique KV slot at its actual 256- or 512-element row width. Decoder
attention consumes these caches directly, eliminating the KV gather copies and
reducing K+V cache storage from 30 MB to 18 MB.

The **FPGA per-layer preparation** run additionally moves the shared BF16
`1536 -> 8960` projection, 256-wide RMSNorm, per-layer embedding add, and input
scaling onto the FPGA. The host performs only token-indexed row lookup and DMA;
the FPGA preparation costs 321.035 ms per prefill and 5.530 ms per decode token.

Profiled FPGA decode breakdown at position 272:

| Phase | Streaming MLP down | Compact KV direct attention | FPGA per-layer preparation |
|---|---:|---:|---:|
| MLP | 135.945 ms (52.5%) | 135.553 ms (57.0%) | 135.546 ms (55.7%) |
| LM head | 34.761 ms (13.4%) | 34.762 ms (14.6%) | 34.761 ms (14.3%) |
| Attention | 30.393 ms (11.7%) | 30.395 ms (12.8%) | 30.393 ms (12.5%) |
| KV gather | 20.479 ms (7.9%) | 0.102 ms (0.0%, checkpoint only) | 0.103 ms (0.0%, checkpoint only) |
| QKV/V projection | 13.113 ms (5.1%) | 13.115 ms (5.5%) | 13.107 ms (5.4%) |
| O projection | 11.832 ms (4.6%) | 11.747 ms (4.9%) | 11.745 ms (4.8%) |
| Per-layer injection | 11.336 ms (4.4%) | 11.337 ms (4.8%) | 11.339 ms (4.7%) |
| Per-layer preparation | — | — | 5.530 ms (2.3%) |
| RoPE | 0.969 ms (0.4%) | 0.970 ms (0.4%) | 0.969 ms (0.4%) |
| Tail add/increment | 0.003 ms (0.0%) | 0.003 ms (0.0%) | 0.003 ms (0.0%) |
| **Total** | **258.831 ms (100.0%)** | **237.984 ms (100.0%)** | **243.495 ms (100.0%)** |
| **Decode FPGA throughput** | **3.86 tok/s** | **4.20 tok/s** | **4.11 tok/s** |
