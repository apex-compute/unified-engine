# Gemma4 E2B performance

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
> `--device alveo` with HW version `0x3d04c689`
> `--dev xdma1` (kintex7) with HW version `0x6ee171b8`
> `--device rk_256` with HW version `0x3d04c689`

| Metric | Kintex | Kintex 2-core | rk_256 | Alveo | Alveo 2-core | Alveo 4-core | Alveo 8-core |
|---|---:|---:|---:|---:|---:|---:|---:|
| Peak throughput (GFLOPS) | 25.4 | 50.8 | 42.7 | 46.9 | 93.9 | 187.7 | 375.5 |
| DRAM read speed (MB/s) | 5875.9 | 6884.3 | 9272.2 | **10484.2** | **11271.0** | **15132.2** | **22940.2** |
| **Vision** |
| Vision soft tokens | 256 | 256 | 256 | 256 | 256 | 256 | 256 |
| Vision throughput (GFLOPS) | 21.4 | 41.4 | 35.9 | **39.5** | **74.3** | **142.0** | **231.9** |
| — utilization (% peak) | 84.2% | 81.6% | 84.0% | **84.1%** | **79.1%** | **75.7%** | **61.8%** |
| Vision FPGA execution (s)| 53.8 | 27.7 | 32.1 | **29.1** | **15.5** | **8.1** | **5.0** |
| Vision end-to-end (CPU)(s) | 53.8 | 27.8 | 32.1 | **29.2** | **15.5** | **8.2** | **5.1** |
| **LM PREFILL** |
| LM prefill seq length | 272 | 272 | 272 | 272 | 272 | 272 | 272 |
| LM prefill throughput (GFLOPS) | 24.1 | 45.1 | 37.8 | **42.5** | **82.2** | **152.6** | **263.6** |
| — utilization (% peak) | 94.9% | 88.9% | 88.6% | **90.5%** | **87.6%** | **81.3%** | **70.2%** |
| Prefill FPGA execution (s)| 44.0 | 23.5 | 28.0 | **24.9** | **12.9** | **6.9** | **4.0** |
| Prefill end-to-end (CPU)(s) | 44.1 | 23.6 | 28.1 | **25.0** | **13.0** | **7.1** | **4.2** |
| **LM DECODE** |
| Decode 1-st tok throughput (tok/s)| 3.9 | 3.9 | 6.2 | **6.8** | **6.7** | **6.9** | **6.7** |
| Decode average throughput (GFLOPS) | 23.1 | 22.9 | 36.5 | **40.8** | **40.6** | **40.6** | **40.6** |
| — utilization (% peak) | 90.8% | 45.2% | 85.5% | **86.9%** | **43.2%** | **21.6%** | **10.8%** |
| Decode end-to-end (CPU)(s) | 104.5 | 116.1 | 66.6 | **60.2** | **68.0** | **67.8** | **67.3** |
| Decode average throughput (CPU)(tok/s)| 3.7 | 3.7 | 5.8 | **6.4** | **6.3** | **6.3** | **6.3** |
| **ARTIFACT** |
| Vision program section (MiB) | 3.2 | 2.4 | 3.2 | 3.2 | 2.4 | 1.9 | 1.8 |
| Prefill program section (MiB)| 5.8 | 4.8 | 5.8 | 5.8 | 4.8 | 4.1 | 3.7 |
| Decode program section (MiB)| 1.4 | 1.4 | 1.4 | 1.4 | 1.4 | 1.4 | 1.4 |
| Combined program image (MiB)| 10.5 | 13.7 | 10.5 | 10.5 | 13.7 | 19.2 | 30.4 |
| Weight image (`params.bin`) (GiB) | 6.9 | same | same | same | same | same | same |
| Correctness | coherent, total 656 | coherent, total 699 | coherent, total 656 | coherent, total 656 | coherent, total 699 | coherent, total 699 | coherent, total 699 |

## Profile backup: vision encoder (kintex7)

Vision multi-core segments tile cleanly (they sum to the single-shot master total)

| Phase | Single-core | --multi-core 2 |
|---|---:|---:|
| Patch embedding | 123.6 ms | 123.6 ms |
| Projection (Q/K/V) † | 6,682.4 ms | **3,361.0 ms** |
| RoPE | 1,072.8 ms | **740.0 ms** |
| Permute | 269.6 ms | 269.6 ms |
| Attention † | 16,642.2 ms | **8,664.7 ms** |
| Post-attention (O + MLP) † | 28,911.4 ms | **14,541.1 ms** |
| Pooler tail | 75.3 ms | 75.3 ms |
| Tail | 0.0 ms | 0.0 ms |
| **Total** | **53,777.3 ms** | **27,775 ms** |

† row-sharded across 2 engines

## Profile backup: LM prefill (kintex7)

| Phase | Single-core | --multi-core 2 | Single-core GQA | --multi-core 2 GQA |
|---|---:|---:|---:|---:|
| Per-layer preparation ‡ | 596.0 ms | 596.0 ms | 596.0 ms | 0.0 ms |
| QKV/V projection † | 3,302.5 ms | **1,682.3 ms** | 3,302.5 ms | **1,693.6 ms** |
| RoPE | 162.6 ms | 162.6 ms | 162.6 ms | 162.5 ms |
| KV gather | 70.0 ms | 70.0 ms | 0.1 ms | 0.1 ms |
| Q permute | — | — | 75.2 ms | 75.3 ms |
| Attention † | 9,268.7 ms | 9,268.7 ms | 1,697.2 ms | **906.3 ms** |
| MLP (incl. O projection) † | 37,426.6 ms | **19,309.4 ms** | 37,426.7 ms | **19,331.5 ms** |
| Injection | 694.2 ms | 694.2 ms | 694.2 ms | 694.2 ms |
| Tail halt | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| **Total** | **51,520.6 ms** | **31,782.4 ms** | **43,954.5 ms** | **22,863.3 ms** |

† row-sharded across 2 engines (these ~halve).

## Profile backup: one decode token at position 272 (kintex7) no multicore usecase in decoder

| Phase | Single-core | 
|---|---:|
| Per-layer preparation | 5.6 ms |
| QKV/V projection | 13.2 ms |
| RoPE | 1.0 ms |
| KV gather | 0.1 ms |
| Attention | 30.5 ms |
| O projection | 11.8 ms |
| MLP | 135.8 ms |
| Injection | 12.6 ms |
| LM head | 34.8 ms |
| Tail increment | 0.0 ms |
| **Total** | **245.3 ms** |
