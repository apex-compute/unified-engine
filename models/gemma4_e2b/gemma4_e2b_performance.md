# Gemma4 E2B performance

## Alveo U55C scaling

Measured on Alveo U55C (`xdma0`, HW version `0x68f0c76c`) at 300 MHz using
the current single-core and 2/4/8/10/12-core image-run summaries.

| Metric | 1 core | 2 cores | 4 cores | 8 cores | 10 cores | 12 cores |
|---|---:|---:|---:|---:|---:|---:|
| Peak throughput (GFLOPS) | 38.4 | 76.8 | 153.6 | 307.2 | 384.0 | 460.8 |
| DRAM read speed (MB/s) | 8,288.7 | 8,953.0 | 14,414.1 | 25,483.9 | 31,793.0 | 38,250.6 |
| Vision throughput (GFLOPS) | 32.3 | 60.8 | 115.0 | 187.6 | 199.2 | 248.7 |
| Vision utilization (% peak) | 84.0% | 79.2% | 74.9% | 61.1% | 51.9% | 54.0% |
| Vision FPGA execution (s) | 35.6 | 18.9 | 10.0 | 6.1 | 5.8 | 4.6 |
| Prefill throughput (GFLOPS) | 33.6 | 67.8 | 125.1 | 217.7 | 203.7 | 218.8 |
| Prefill utilization (% peak) | 87.6% | 88.3% | 81.5% | 70.9% | 53.1% | 47.5% |
| Prefill FPGA execution (s) | 31.5 | 15.6 | 8.5 | 4.9 | 5.2 | 4.8 |
| Decode first-token speed (tok/s) | 5.6 | 5.8 | 5.8 | 5.8 | 5.5 | 5.5 |
| Decode average speed (tok/s, CPU timer) | 5.3 | 5.5 | 5.5 | 5.4 | 5.2 | 5.2 |

## Test setup

All implementations used `test_samples/yosemite.jpg`, the prompt
`Describe this image in detail.`, the same `params.bin`, IF4 projection
weights, 35 LM layers, 16 vision layers, and 256 image soft tokens.

Commands:

```bash
python models/gemma4_e2b/gemma4_e2b_test.py --device rk_256 --dev xdma0 --image

python models/gemma4_e2b/gemma4_e2b_test.py --device kintex7 --dev xdma1 --image
python models/gemma4_e2b/gemma4_e2b_test.py --device kintex7 --dev xdma1 --image --multi-core 2
python models/gemma4_e2b/gemma4_e2b_test.py --device alveo --dev xdma0 --image
python models/gemma4_e2b/gemma4_e2b_test.py --device alveo --dev xdma0 --image --multi-core 2
python models/gemma4_e2b/gemma4_e2b_test.py --device alveo --dev xdma0 --image --multi-core 4
python models/gemma4_e2b/gemma4_e2b_test.py --device alveo --dev xdma0 --image --multi-core 8

python models/gemma4_e2b/gemma4_e2b_test.py --device kintex7 --dev xdma1 --image --profile
python models/gemma4_e2b/gemma4_e2b_test.py --device kintex7 --dev xdma1 --image --multi-core --profile
```

## Performance comparison
> `--device alveo` with HW version `0x68f0c76c`
> `--dev xdma1` (kintex7) with HW version `0x6ee171b8`
> `--device rk_256` with HW version `0x3d04c689`

| Metric | Kintex | Kintex 2-core | rk_256 | Alveo | Alveo 2-core | Alveo 4-core | Alveo 8-core |
|---|---:|---:|---:|---:|---:|---:|---:|
| Peak throughput (GFLOPS) | 25.4 | 50.8 | 42.7 | 46.9 | 93.9 | 187.7 | 375.5 |
| DRAM read speed (MB/s) | 5875.9 | 6980.4 | 9272.2 | **10484.2** | **11176.7** | **15094.1** | **22918.3** |
| **Vision** |
| Vision soft tokens | 256 | 256 | 256 | 256 | 256 | 256 | 256 |
| Vision throughput (GFLOPS) | 21.4 | 41.4 | 35.9 | **39.5** | **74.9** | **142.2** | **232.2** |
| — utilization (% peak) | 84.2% | 81.6% | 84.0% | **84.1%** | **79.8%** | **75.7%** | **61.9%** |
| Vision FPGA execution (s)| 53.8 | 27.7 | 32.1 | **29.1** | **15.3** | **8.1** | **4.9** |
| Vision end-to-end (CPU)(s) | 53.8 | 27.8 | 32.1 | **29.2** | **15.4** | **8.2** | **5.1** |
| **LM PREFILL** |
| LM prefill seq length | 272 | 272 | 272 | 272 | 272 | 272 | 272 |
| LM prefill throughput (GFLOPS) | 24.1 | 45.1 | 37.8 | **42.5** | **82.3** | **152.7** | **263.4** |
| — utilization (% peak) | 94.9% | 88.9% | 88.5% | **90.5%** | **87.7%** | **81.3%** | **70.2%** |
| Prefill FPGA execution (s)| 44.0 | 23.5 | 28.0 | **24.9** | **12.9** | **6.9** | **4.0** |
| Prefill end-to-end (CPU)(s) | 44.1 | 23.6 | 28.1 | **25.0** | **13.0** | **7.1** | **4.3** |
| **LM DECODE** |
| Decode 1-st tok throughput (tok/s)| 4.0 | 3.9 | 6.2 | **6.8** | **6.7** | **6.6** | **6.8** |
| Decode average throughput (GFLOPS) | 18.3 | 18.2 | 28.9 | **32.3** | **32.2** | **32.2** | **32.2** |
| — utilization (% peak) | 71.9% | 35.8% | 67.7% | **68.8%** | **34.3%** | **17.1%** | **8.6%** |
| Decode end-to-end (CPU)(s) | 104.0 | 115.9 | 66.3 | **60.3** | **67.3** | **67.1** | **67.5** |
| Decode average throughput (CPU)(tok/s)| 3.7 | 3.7 | 5.8 | **6.4** | **6.3** | **6.4** | **6.3** |
| **ARTIFACT** |
| Vision program section (MiB) | 3.2 | 2.4 | 3.2 | 3.2 | 2.4 | 1.9 | 1.8 |
| Prefill program section (MiB)| 5.8 | 4.8 | 5.8 | 5.8 | 4.8 | 4.1 | 3.7 |
| Decode program section (MiB)| 1.4 | 1.4 | 1.4 | 1.4 | 1.4 | 1.4 | 1.4 |
| Combined program image (MiB)| 10.5 | 13.7 | 10.5 | 10.5 | 13.7 | 19.2 | 30.4 |
| Weight image (`params.bin`) (GiB) | 6.9 | same | same | same | same | same | same |
| Correctness | coherent, total 656 | coherent, total 699 | coherent, total 656 | coherent, total 656 | coherent, total 699 | coherent, total 699 | coherent, total 699 |

## Profile backup: vision encoder (kintex7)

Vision multi-core segments tile cleanly (they sum to the single-shot master total).

| Phase | Work (GFLOPs) | Samples | Single-core | | | `--multi-core 2` | | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| | | | FPGA execution time (ms) | Throughput (GFLOPS) | Utilization (% peak) | FPGA execution time (ms) | Throughput (GFLOPS) | Utilization (% peak) |
| patch_embed | 2.97 | 1 | 123.6 | 24.1 | 94.8% | 123.6 | 24.1 | 47.4% |
| proj † | 147.15 | 16 | 6,682.5 | 22.0 | 86.7% | 3,361.0 | 43.8 | 86.2% |
| rope † | 8.24 | 16 | 1,072.9 | 7.7 | 30.2% | 738.2 | 11.2 | 22.0% |
| permute | 0.00 | 16 | 269.6 | 0.0 | 0.0% | 269.6 | 0.0 | 0.0% |
| attention † | 329.70 | 16 | 16,642.3 | 19.8 | 78.0% | 8,636.4 | 38.2 | 75.2% |
| post_attn † | 659.51 | 16 | 28,912.0 | 22.8 | 89.9% | 14,539.7 | 45.4 | 89.3% |
| pooler_tail | 1.51 | 1 | 75.3 | 20.1 | 79.0% | 75.3 | 20.1 | 39.5% |
| **TOTAL** | **1149.1** | | **53,778.1** | **21.4** | **84.2%** | **27,743.7** | **41.4** | **81.6%** |

† Multicore implemented: `proj` and `post_attn` are row-sharded, `rope` is row-sharded, and `attention` is head-sharded across the two engines. `patch_embed`, `permute`, and `pooler_tail` run on core 0 only.

## Profile backup: LM prefill (kintex7)

| Phase | Work (GFLOPs) | Samples | Single-core | | | `--multi-core 2` | | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| | | | FPGA execution time (ms) | Throughput (GFLOPS) | Utilization (% peak) | FPGA execution time (ms) | Throughput (GFLOPS) | Utilization (% peak) |
| per_layer_prepare | 14.11 | 1 | 595.9 | 23.7 | 93.3% | 595.9 | 23.7 | 93.3% |
| qkv_vproj † | 79.73 | 35 | 3,302.6 | 24.1 | 95.1% | 1,693.6 | 47.1 | 92.7% |
| rope | 0.08 | 35 | 162.6 | 0.5 | 1.9% | 162.4 | 0.5 | 1.0% |
| q_permute | 0.00 | 35 | 75.2 | 0.0 | 0.0% | 75.2 | 0.0 | 0.0% |
| attention † | 30.12 | 35 | 1,697.2 | 17.7 | 69.9% | 906.5 | 33.2 | 65.4% |
| mlp † | 919.50 | 35 | 37,426.7 | 24.6 | 96.8% | 19,333.1 | 47.6 | 93.7% |
| inject | 15.07 | 35 | 694.2 | 21.7 | 85.5% | 694.2 | 21.7 | 42.8% |
| **TOTAL** | **1058.6** | | **43,954.4** | **24.1** | **94.9%** | **23,460.9** | **45.1** | **88.9%** |

† Multicore implemented: `qkv_vproj`, `attention`, and `mlp` are sharded across the two engines. `per_layer_prepare`, `rope`, `q_permute`, and `inject` run on core 0 only.

## Profile backup: decode tokens (kintex7; no multicore use case)

| Phase | Samples | First decode step (position 272) | | | | 1024th token (position 1023) | | | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | | Work (GFLOPs) | FPGA execution time (ms) | Throughput (GFLOPS) | Utilization (% peak) | Work (GFLOPs) | FPGA execution time (ms) | Throughput (GFLOPS) | Utilization (% peak) |
| per_layer_prepare | 1 | 0.03 | 5.6 | 5.0 | 19.6% | 0.03 | 5.6 | 5.0 | 19.6% |
| qkv_vproj | 35 | 0.29 | 13.2 | 22.3 | 87.7% | 0.29 | 13.2 | 22.3 | 87.7% |
| rope | 35 | 0.00 | 1.0 | 0.6 | 2.4% | 0.00 | 1.0 | 0.6 | 2.4% |
| attention | 35 | 0.11 | 30.6 | 3.6 | 14.3% | 0.35 | 92.5 | 3.8 | 15.1% |
| o_proj | 35 | 0.26 | 11.8 | 22.5 | 88.6% | 0.26 | 11.8 | 22.5 | 88.5% |
| mlp | 35 | 3.12 | 135.8 | 22.9 | 90.4% | 3.12 | 135.8 | 22.9 | 90.4% |
| inject | 35 | 0.06 | 12.6 | 4.4 | 17.3% | 0.06 | 12.6 | 4.4 | 17.3% |
| lm_head | 1 | 0.81 | 34.8 | 23.1 | 91.1% | 0.81 | 34.8 | 23.1 | 91.1% |
| **TOTAL** | | **4.7** | **245.3** | **19.1** | **75.0%** | **4.9** | **307.2** | **16.0** | **63.0%** |

### Ideal attention prediction (100% peak throughput)

All non-attention phase latencies remain measured values. Predicted attention latency is `attention work / 25.4 GFLOPS`.

| Phase | First decode step (position 272) | | | | 1024th token (position 1023) | | | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| | Work (GFLOPs) | FPGA execution time (ms) | Throughput (GFLOPS) | Utilization (% peak) | Work (GFLOPs) | FPGA execution time (ms) | Throughput (GFLOPS) | Utilization (% peak) |
| attention | 0.11 | 4.4 | 25.4 | 100.0% | 0.35 | 13.9 | 25.4 | 100.0% |
| **TOTAL** | **4.7** | **219.1** | **21.3** | **84.0%** | **4.9** | **228.6** | **21.5** | **84.7%** |
