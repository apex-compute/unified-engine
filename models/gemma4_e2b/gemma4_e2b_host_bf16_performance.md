# Gemma4 E2B host GPU performance — BF16

## Test setup

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 |
| CUDA / PyTorch | 13.0 / 2.13.0+cu130 |
| Architecture / compute capability | Blackwell / 12.0 |
| CUDA cores | 6,144 |
| Tensor cores | 192, 5th generation |
| Advertised peak AI throughput | 988 AI TOPS (FP4 sparse) |
| VRAM | 11.50 GiB GDDR7 |
| Theoretical DRAM bandwidth | 672 GB/s |
| Weight / activation precision | BF16 / BF16 |
| Image | `/home/siqinliu/unified-engine/test_samples/yosemite.jpg`; resized to 896×896 |
| Vision patches / soft tokens | 2520 / 256 |
| Prompt | Describe this image in detail. |
| Prefill / generated / total tokens | 273 / 384 / 657 |
| Timed runs | 3 median, CUDA synchronized |
| Peak allocated VRAM | 9.63 GiB (83.8% of device memory) |

## Performance

| Stage | Metric |
|---|---:|
| Vision encoder | 0.0354 s |
| VLM prefill + first token | 0.0671 s (14.90 tok/s) |
| Estimated LM prefill + first token | 0.0318 s |
| Decode (remaining 383 tokens) | 6.4762 s (59.14 tok/s) |
| Full generation (384 tokens) | 6.5433 s (58.69 tok/s) |

Vision is measured independently. The VLM first-token time includes vision; subtracting the independent vision measurement gives the displayed estimated LM-prefill figure.
