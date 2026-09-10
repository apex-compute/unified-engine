# Gemma4 E2B performance

All numbers below are Alveo, `--dev xdma0`, HW version `0x6e7aca2e`, 2.7 ns
(366.7 MHz). Peak throughput is `366.7 MHz × 128 × cores`: **46.9 GFLOPS** at one
core, **375.5 GFLOPS** at eight.

## Test setup

`test_samples/yosemite.jpg`, the prompt `Describe this image in detail.`, the same
`params.bin` (7078.6 MB on disk, 1544.4 MB of quantized weights on the FPGA), IF4
projection weights, 35 LM layers, 16 vision layers, 256 image soft tokens.

```bash
python models/gemma4_e2b/gemma4_e2b_test.py --device alveo --dev xdma0 --image
python models/gemma4_e2b/gemma4_e2b_test.py --device alveo --dev xdma0 --image --multi-core 8

python models/gemma4_e2b/gemma4_e2b_test.py --device alveo --dev xdma0 --image --profile
python models/gemma4_e2b/gemma4_e2b_test.py --device alveo --dev xdma0 --image --multi-core 8 --profile
```

## Performance comparison

| Metric | 1 core | 8 cores | Speedup |
|---|---:|---:|---:|
| Peak throughput (GFLOPS) | 46.9 | 375.5 | 8.00× |
| DRAM read speed (MB/s) | 10,802.2 | **86,128.9** | 7.97× |
| **Vision** |
| Vision soft tokens | 256 | 256 | |
| Vision throughput (GFLOPS) | 39.5 | **242.8** | 6.15× |
| — utilization (% peak) | 84.1% | 64.7% | |
| Vision FPGA execution (s) | 29.1 | **4.7** | 6.15× |
| Vision end-to-end (CPU) (s) | 29.2 | **4.8** | 6.08× |
| **LM PREFILL** |
| LM prefill seq length | 272 | 272 | |
| LM prefill throughput (GFLOPS) | 42.5 | **265.2** | 6.24× |
| — utilization (% peak) | 90.5% | 70.6% | |
| Prefill FPGA execution (s) | 24.9 | **4.0** | 6.24× |
| Prefill end-to-end (CPU) (s) | 25.0 | **4.2** | 5.95× |
| Time to first token (TTFT, CPU; vision + prefill) (s) | 54.2 | **9.0** | 6.02× |
| Time to first token (TTFT, HW counter; vision + prefill) (s) | 54.0 | **8.7** | 6.19× |
| **LM DECODE** |
| Decode 1st-token speed (tok/s, HW counter) | 7.2 | **31.1** | 4.32× |
| Decode average throughput (GFLOPS) | 32.3 | **138.1** | 4.28× |
| — utilization (% peak) | 68.8% | 36.8% | |
| Decode average speed (CPU timer) (tok/s) | 6.5 | **24.2** | 3.72× |
| Decode end-to-end (CPU) (s) † | 58.8 | 18.4 | |
| **ARTIFACT** |
| Vision program section (MB) | 3.2 | 1.6 | |
| Prefill program (KB) | 5989.2 | 3593.9 | |
| Decoder program (KB) | 1456.5 | 1076.7 | |
| Combined program image (MB) | 10.5 | 28.1 | |
| Weight image (`params.bin`) (MB) | 7078.6 | same | |
| Correctness | coherent, total 656 | coherent, total 716 | |

† Not a like-for-like ratio: the two runs generated different token counts (384 vs
444), so compare the tok/s rows instead.

## Profile: vision encoder

| Phase | Work (GFLOPs) | Samples | 1 core | | | 8 cores | | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| | | | FPGA (ms) | GFLOPS | % peak | FPGA (ms) | GFLOPS | % peak |
| patch_embed | 2.97 | 1 | 66.8 | 44.5 | 94.8% | 66.8 | 44.5 | 11.9% |
| proj † | 147.15 | 16 | 3,610.0 | 40.8 | 86.9% | 473.7 | 310.6 | 82.7% |
| rope † | 8.24 | 16 | 575.1 | 14.3 | 30.5% | 249.8 | 33.0 | 8.8% |
| permute | 0.00 | 16 | 183.1 | 0.0 | 0.0% | 183.1 | 0.0 | 0.0% |
| attention † | 329.70 | 16 | 9,052.0 | 36.4 | 77.6% | 1,664.4 | 198.1 | 52.8% |
| post_attn † | 659.51 | 16 | 15,591.2 | 42.3 | 90.1% | 2,057.6 | 320.5 | 85.4% |
| pooler_tail | 1.51 | 1 | 40.7 | 37.1 | 79.1% | 40.7 | 37.1 | 9.9% |
| **TOTAL** | **1149.1** | | **29,119.0** | **39.5** | **84.1%** | **4,736.2** | **242.6** | **64.6%** |

† Sharded: `proj`, `post_attn` and `rope` are row-sharded, `attention` is
head-sharded. `patch_embed`, `permute` and `pooler_tail` run on core 0 only — their
times are identical in both columns, and their 8-core `% peak` is scored against the
full 375.5 GFLOPS, which is why it collapses.

## Profile: LM prefill

| Phase | Work (GFLOPs) | Samples | 1 core | | | 8 cores | | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| | | | FPGA (ms) | GFLOPS | % peak | FPGA (ms) | GFLOPS | % peak |
| per_layer_prepare | 14.11 | 1 | 322.6 | 43.7 | 93.2% | 322.6 | 43.7 | 93.2% |
| qkv_vproj † | 79.73 | 35 | 1,876.8 | 42.5 | 90.5% | 247.9 | 321.6 | 85.7% |
| rope | 0.08 | 35 | 91.8 | 0.9 | 1.8% | 91.8 | 0.9 | 0.2% |
| q_permute | 0.00 | 35 | 43.9 | 0.0 | 0.0% | 43.9 | 0.0 | 0.0% |
| attention † | 30.12 | 35 | 920.8 | 32.7 | 69.7% | 171.7 | 175.4 | 46.7% |
| mlp † | 919.50 | 35 | 21,280.6 | 43.2 | 92.1% | 2,739.4 | 335.7 | 89.4% |
| inject | 15.07 | 35 | 376.8 | 40.0 | 85.2% | 376.8 | 40.0 | 10.7% |
| **TOTAL** | **1058.6** | | **24,913.3** | **42.5** | **90.5%** | **3,994.2** | **265.0** | **70.6%** |

† Sharded: `qkv_vproj`, `attention` and `mlp`. `per_layer_prepare`, `rope`,
`q_permute` and `inject` run on core 0 only.

## Profile: LM decode

### First decode step (position 272)

| Phase | Work (GFLOPs) | Samples | 1 core | | | 8 cores | | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| | | | FPGA (ms) | GFLOPS | % peak | FPGA (ms) | GFLOPS | % peak |
| per_layer_prepare | 0.03 | 1 | 3.1 | 8.8 | 18.8% | 3.1 | 8.8 | 18.8% |
| qkv_vproj † | 0.29 | 35 | 7.5 | 38.9 | 82.9% | 1.7 | 175.0 | 46.6% |
| rope | 0.00 | 35 | 0.6 | 1.1 | 2.2% | 0.6 | 1.1 | 0.3% |
| attention † | 0.11 | 35 | 16.6 | 6.7 | 14.2% | 5.7 | 19.4 | 5.2% |
| o_proj † | 0.26 | 35 | 6.7 | 39.5 | 84.1% | 1.2 | 229.4 | 61.1% |
| mlp † | 3.12 | 35 | 77.6 | 40.2 | 85.6% | 10.7 | 292.1 | 77.8% |
| inject | 0.06 | 35 | 7.1 | 7.8 | 16.6% | 7.1 | 7.8 | 2.1% |
| lm_head † | 0.81 | 1 | 19.9 | 40.5 | 86.4% | 2.5 | 322.6 | 85.9% |
| **TOTAL** | **4.7** | | **139.0** | **33.6** | **71.6%** | **32.5** | **143.7** | **38.3%** |

### 1024th token (position 1023)

| Phase | Work (GFLOPs) | Samples | 1 core | | | 8 cores | | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| | | | FPGA (ms) | GFLOPS | % peak | FPGA (ms) | GFLOPS | % peak |
| per_layer_prepare | 0.03 | 1 | 3.1 | 8.8 | 18.8% | 3.1 | 8.8 | 18.8% |
| qkv_vproj † | 0.29 | 35 | 7.5 | 38.9 | 83.0% | 1.7 | 174.6 | 46.5% |
| rope | 0.00 | 35 | 0.6 | 1.1 | 2.3% | 0.6 | 1.1 | 0.3% |
| attention † | 0.35 | 35 | 50.2 | 7.1 | 15.0% | 13.3 | 26.6 | 7.1% |
| o_proj † | 0.26 | 35 | 6.7 | 39.5 | 84.2% | 1.2 | 229.7 | 61.2% |
| mlp † | 3.12 | 35 | 77.6 | 40.2 | 85.6% | 10.7 | 292.0 | 77.8% |
| inject | 0.06 | 35 | 7.1 | 7.8 | 16.6% | 7.1 | 7.8 | 2.1% |
| lm_head † | 0.81 | 1 | 19.9 | 40.5 | 86.4% | 2.5 | 322.7 | 85.9% |
| **TOTAL** | **4.9** | | **172.6** | **28.5** | **60.7%** | **40.1** | **122.4** | **32.6%** |

† Sharded: `qkv_vproj`, `o_proj`, `mlp` and `lm_head` are column-sharded weights.
`attention` is partly sharded — its V transpose is split over the workers by input
rows and P@V^T over its output columns, but Q@K^T stays whole on core 0 (softmax is
a row reduction over exactly the axis an N-shard would split). `per_layer_prepare`,
`rope` and `inject` run on core 0 only.

Amdahl, at the 1024th token: the three unsharded phases total 10.8 ms either way,
which is 6% of the 1-core step (172.6 ms) but 27% of the 8-core one (40.1 ms) --
`inject` alone is 7.1 ms of it. The largest single phase in the 8-core run is
`attention` at 13.3 ms, ahead of `mlp` at 10.7, because only part of it is split.
