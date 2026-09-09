# Gemma3-1B host GPU performance — BF16

## Test setup

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 |
| Architecture / compute capability | Blackwell / 12.0 |
| CUDA cores / Tensor cores | 6,144 / 192 (5th generation) |
| CUDA / PyTorch | 13.0 / 2.13.0+cu130 |
| Advertised peak AI throughput | 988 AI TOPS (FP4 sparse) |
| VRAM / theoretical DRAM bandwidth | 11.50 GiB GDDR7 / 672 GB/s |
| Weight / activation precision | BF16 / BF16 |
| Prompt | x+3=5, what is x? |
| Host input IDs / FPGA prefill compute tokens | 20 / 19 |
| Generated / total host tokens | 73 / 93 |
| Timed runs | 3 median, CUDA synchronized |
| Peak allocated VRAM | 1.95 GiB (17.0%) |

## Performance

| Stage | Metric |
|---|---:|
| Prefill forward | 0.0123 s |
| End-to-end first token | 0.0132 s (75.79 tok/s) |
| Decode after prefill (73 tokens) | 0.8193 s (89.10 tok/s) |
| Full generation (73 tokens) | 0.8315 s (87.79 tok/s) |

The FPGA runner processes all but the final prompt token in its prefill program;
that final token enters its decoder. The host model receives the full chat-template
input, so both token counts are shown explicitly.
