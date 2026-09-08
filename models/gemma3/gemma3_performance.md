# Gemma3 performance

All numbers below are Alveo, `--dev xdma0`, HW version `0x6e7aca2e`, 2.7273 ns
(366.7 MHz). Peak throughput is `366.7 MHz × 128 × cores`: **46.93 GFLOPS** at one
core, **375.47 GFLOPS** at eight.

## Test setup

Prompt `x+3=5, what is x?`; IF4 weights (`weights_gemma3_hf.bin`, 1083.77 MB on
disk, 507.8 MB of quantized weights on the FPGA); 26 layers, hidden 1152, head_dim
256, group_size 4, MLP 6912, global-RoPE layers [5, 11, 17, 23]. Both runs compiled
the program image from scratch (`--bin-reuse` off).

```bash
python models/gemma3/gemma3_test.py --dev xdma0
python models/gemma3/gemma3_test.py --dev xdma0 --multi-core 8

python models/gemma3/gemma3_test.py --dev xdma0 --profile
python models/gemma3/gemma3_test.py --dev xdma0 --multi-core 8 --profile
```

## Performance comparison

| Metric | 1 core | 8 cores | Speedup |
|---|---:|---:|---:|
| Peak throughput (GFLOPS) | 46.93 | 375.47 | 8.00× |
| **PREFILL** † |
| Prefill tokens (seq_len) | 19 | 19 | |
| Prefill FPGA execution (ms) | 638.73 | 638.87 | 1.00× |
| Prefill throughput (GFLOPS) | 42.18 | 42.17 | 1.00× |
| — utilization (% peak) | 89.9% | 11.2% † | |
| Prefill end-to-end (CPU) (s) | 0.64 | 0.64 | |
| **DECODE** |
| Decoded tokens | 76 (total 95) | 76 (total 95) | |
| Decode 1st-token speed (tok/s, HW counter) | 18.64 | **85.91** | 4.61× |
| Decode average throughput (GFLOPS) | 37.03 | **171.70** | 4.64× |
| — utilization (% peak) | 78.9% | 45.7% | |
| Decode average speed (CPU timer) (tok/s) | 18.03 | **72.80** | 4.04× |
| **ARTIFACT** |
| Prefill program (KB) | 88.9 | 88.9 | |
| Decoder program (KB) | 66.6 | 38.4 | |
| Combined program image (KB) | 155.5 | 127.3 | |
| Weight bin (`weights_gemma3_hf.bin`) (MB) | 1083.77 | same | |
| Weight DRAM (quantized, on FPGA) (MB) | 507.8 | same | |
| Correctness | coherent (solves x = 2) | coherent (solves x = 2) | |

† **Prefill is not sharded.** `--multi-core 8` shards the decoder only, so prefill
runs on core 0 in both configurations and the two times are within 0.02% of each
other. Its 8-core `% peak` of 11.2% is that same single-engine work scored against
the eight-engine peak — it is not a regression, and the honest figure is the 89.9%
in the 1-core column.

## Profile: prefill (seq_len = 19)

Same program on core 0 either way; the two columns differ only by run-to-run noise.

| Step | GFLOP | 1 core | | | 8 cores | | |
|---|---:|---:|---:|---:|---:|---:|---:|
| | | FPGA (ms) | GFLOPS | % peak | FPGA (ms) | GFLOPS | % peak |
| pre_norm | 0.002 | 0.730 | 3.12 | 6.6% | 0.728 | 3.13 | 6.7% |
| qkv_proj_vcache | 1.748 | 41.079 | 42.56 | 90.7% | 41.094 | 42.54 | 90.6% |
| qk_norm_rope | 0.005 | 3.640 | 1.39 | 3.0% | 3.631 | 1.39 | 3.0% |
| attention | 0.130 | 18.345 | 7.11 | 15.1% | 18.340 | 7.11 | 15.2% |
| o_proj_post_attn_norm_residual | 1.168 | 28.210 | 41.42 | 88.2% | 28.218 | 41.40 | 88.2% |
| pre_ffn_norm | 0.002 | 0.666 | 3.42 | 7.3% | 0.668 | 3.41 | 7.3% |
| mlp_gateup_gelu_mul | 15.751 | 363.663 | 43.31 | 92.3% | 363.567 | 43.32 | 92.3% |
| mlp_down_post_ffn_norm_residual | 7.870 | 182.874 | 43.03 | 91.7% | 182.897 | 43.03 | 91.7% |
| **Total** | **26.678** | **639.206** | **41.74** | **88.9%** | **639.142** | **41.74** | **88.9%** |

## Profile: decode, first token (aligned KV = 64)

| Step | GFLOP | 1 core | | | 8 cores | | |
|---|---:|---:|---:|---:|---:|---:|---:|
| | | FPGA (ms) | GFLOPS | % peak | FPGA (ms) | GFLOPS | % peak |
| pre_norm | 0.000 | 0.221 | 0.54 | 1.2% | 0.222 | 0.54 | 1.2% |
| qkv_proj_vcache † | 0.092 | 2.449 | 37.57 | 80.0% | 1.124 | 81.87 | 21.8% |
| qk_norm_rope | 0.000 | 0.393 | 0.68 | 1.4% | 0.388 | 0.69 | 1.5% |
| attention ‡ | 0.007 | 3.135 | 2.19 | 4.7% | 2.696 | 2.54 | 1.4% |
| o_proj_post_attn_norm_residual † | 0.061 | 1.759 | 34.93 | 74.4% | 0.474 | 129.76 | 34.6% |
| pre_ffn_norm | 0.000 | 0.137 | 0.87 | 1.9% | 0.136 | 0.88 | 1.9% |
| mlp_gateup_gelu_mul † | 0.829 | 20.742 | 39.96 | 85.1% | 2.977 | 278.18 | 74.1% |
| mlp_down_post_ffn_norm_residual † | 0.414 | 10.489 | 39.49 | 84.1% | 2.124 | 194.97 | 51.9% |
| output_norm_lm_head † | 0.604 | 14.716 | 41.04 | 87.5% | 1.868 | 323.38 | 86.1% |
| **Total** | **2.008** | **54.041** | **37.15** | **79.2%** | **12.009** | **167.14** | **44.7%** |

## Profile: decode, token 512 (aligned KV = 512)

| Step | GFLOP | 1 core | | | 8 cores | | |
|---|---:|---:|---:|---:|---:|---:|---:|
| | | FPGA (ms) | GFLOPS | % peak | FPGA (ms) | GFLOPS | % peak |
| pre_norm | 0.000 | 0.219 | 0.55 | 1.2% | 0.223 | 0.54 | 1.1% |
| qkv_proj_vcache † | 0.092 | 2.452 | 37.53 | 80.0% | 1.124 | 81.87 | 21.8% |
| qk_norm_rope | 0.000 | 0.393 | 0.68 | 1.4% | 0.391 | 0.68 | 1.5% |
| attention ‡ | 0.055 | 14.329 | 3.83 | 8.2% | 5.189 | 10.58 | 5.6% |
| o_proj_post_attn_norm_residual † | 0.061 | 1.756 | 35.00 | 74.6% | 0.475 | 129.53 | 34.5% |
| pre_ffn_norm | 0.000 | 0.135 | 0.89 | 1.9% | 0.134 | 0.89 | 1.9% |
| mlp_gateup_gelu_mul † | 0.829 | 20.737 | 39.97 | 85.2% | 2.975 | 278.40 | 74.1% |
| mlp_down_post_ffn_norm_residual † | 0.414 | 10.492 | 39.48 | 84.1% | 2.126 | 194.77 | 51.9% |
| output_norm_lm_head † | 0.604 | 14.712 | 41.05 | 87.5% | 1.864 | 323.97 | 86.3% |
| **Total** | **2.056** | **65.224** | **31.52** | **67.2%** | **14.500** | **141.73** | **38.8%** |

† Column-sharded weights across all 8 engines: Q/K/V, o_proj, MLP gate/up, MLP
down, LM head.

‡ `attention` is only partly sharded, and only 4-way. Its V transpose is split over
the workers by input rows and P@V^T over its output columns — but P@V^T's N is
head_dim = 256, i.e. four 64-column blocks, so four engines is the ceiling and its
`% peak` is scored against 4 engines, not 8. Q@K^T stays whole on core 0, because
softmax is a row reduction over exactly the axis an N-shard would split.

`pre_norm`, `qk_norm_rope` and `pre_ffn_norm` are unsharded and run on core 0.

## Where the 8-core decode time goes

At token 512 the decoder is **4.50× faster** on 8 cores (65.224 → 14.500 ms), and
attention is what now dominates: 22.0% of the 1-core step but **35.8% of the 8-core
one**, having sped up only 2.76× while the five fully sharded weight steps around it
went 50.149 → 8.564 ms, i.e. 5.86× together. The three unsharded norms total just 0.748 ms (5.2%), so Amdahl is not
the binding constraint here yet — the partly-sharded attention is.
