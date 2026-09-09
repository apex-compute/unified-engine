# Qwen2.5-VL-3B host GPU performance — BF16

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
| Image | `/home/siqinliu/unified-engine/test_samples/yosemite.jpg`; resized to 336×336 |
| Vision patches / merged tokens | 576 / 144 |
| Prompt | Describe the picture in details. |
| Prefill / generated / total tokens | 171 / 256 / 427 |
| Timed runs | 3 median, CUDA synchronized |
| Peak allocated VRAM | 7.17 GiB (62.3%) |

## Performance

| Stage | Metric |
|---|---:|
| Vision encoder | 0.0231 s |
| VLM prefill + first token | 0.0487 s (20.53 tok/s) |
| Estimated LM prefill + first token | 0.0256 s |
| Decode (255 remaining tokens) | 3.1565 s (80.79 tok/s) |
| Full generation (256 tokens) | 3.2052 s (79.87 tok/s) |

Vision is measured independently. The estimated LM-prefill figure subtracts
that vision measurement from the end-to-end first-token measurement.
