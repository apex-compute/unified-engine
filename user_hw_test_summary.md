| Test                                     | Dimensions                            | SNR (dB) | GFLOPS | MB/s    | Inst Bytes | SNR diff  | GFLOPS diff |
| ---------------------------------------- | ------------------------------------- | -------- | ------ | ------- | ---------- | --------- | ----------- |
| software_reset                           | n/a                                   | n/a      | n/a    | n/a     | n/a        |           |             |
| dram_read_speed                          | elements=262080                       | n/a      | n/a    | 5959.24 | n/a        |           |             |
| dram_write_speed                         | elements=262080                       | inf      | n/a    | 5674.33 | n/a        |           |             |
| isa_rela_loop                            | loop_cnt=4, n_elem=256                | n/a      | n/a    | n/a     | n/a        |           |             |
| isa_rela_loop_reg_rela                   | loop_cnt=5, bwd_offset=4              | n/a      | n/a    | n/a     | n/a        |           |             |
| isa_abs_loop                             | loop_cnt=6                            | n/a      | n/a    | n/a     | n/a        |           |             |
| isa_abs_loop_reg_abs                     | loop_cnt=4                            | n/a      | n/a    | n/a     | n/a        |           |             |
| isa_reg_min_sub_mul                      |                                       | n/a      | n/a    | n/a     | n/a        |           |             |
| isa_new_alu_ops                          |                                       | n/a      | n/a    | n/a     | n/a        |           |             |
| ue_int_reg_read                          | n/a                                   | n/a      | n/a    | n/a     | n/a        |           |             |
| fmax                                     | length=256                            | inf      | n/a    | n/a     | n/a        |           |             |
| packing                                  | M=1024 mode=16 wb=16                  | inf      | n/a    | n/a     | n/a        |           |             |
| packing                                  | M=1024 mode=32 wb=32                  | inf      | n/a    | n/a     | n/a        |           |             |
| packing                                  | M=1024 mode=48 wb=48                  | inf      | n/a    | n/a     | n/a        |           |             |
| packing                                  | M=1024 mode=64 wb=64                  | inf      | n/a    | n/a     | n/a        |           |             |
| padding_zero                             | M=128, N=48, N_aligned=64             | inf      | n/a    | n/a     | n/a        |           |             |
| slicing                                  | M=5 N=64 slice=16 wb=16               | inf      | n/a    | n/a     | n/a        |           |             |
| quantized_fp4                            | N=64, K=2048                          | n/a      | 3.87   | n/a     | n/a        |           |             |
| if4_if8-IF4-FP4                          | M=512, scale=1.0, all_q_values        | inf      | n/a    | n/a     | n/a        |           |             |
| if4_if8-IF4-INT4                         | M=512, scale=-1.0, all_q_values       | inf      | n/a    | n/a     | n/a        |           |             |
| if4_if8-IF8-FP8                          | M=512, scale=1.0, all_q_values        | inf      | n/a    | n/a     | n/a        |           |             |
| if4_if8-IF8-INT8                         | M=512, scale=-1.0, all_q_values       | inf      | n/a    | n/a     | n/a        |           |             |
| if4_if8_mixed-IF4-alt                    | M=512, num_blocks=8                   | inf      | n/a    | n/a     | n/a        |           |             |
| if4_if8_mixed-IF4-fp_int                 | M=512, num_blocks=8                   | inf      | n/a    | n/a     | n/a        |           |             |
| if4_if8_mixed-IF4-int_fp                 | M=512, num_blocks=8                   | inf      | n/a    | n/a     | n/a        |           |             |
| if4_if8_mixed-IF8-alt                    | M=512, num_blocks=8                   | inf      | n/a    | n/a     | n/a        |           |             |
| if4_if8_mixed-IF8-fp_int                 | M=512, num_blocks=8                   | inf      | n/a    | n/a     | n/a        |           |             |
| if4_if8_mixed-IF8-int_fp                 | M=512, num_blocks=8                   | inf      | n/a    | n/a     | n/a        |           |             |
| tq4_dequantize-pos_unit                  | M=512, num_blocks=8                   | inf      | n/a    | n/a     | n/a        |           |             |
| tq4_dequantize-varying                   | M=512, num_blocks=8                   | inf      | n/a    | n/a     | n/a        |           |             |
| tq4_dequantize-wide_range                | M=512, num_blocks=8                   | inf      | n/a    | n/a     | n/a        |           |             |
| tq4_dot_product-uniform                  | K=64, N=64                            | 50.00    | n/a    | n/a     | n/a        |           |             |
| tq4_dot_product-varying                  | K=64, N=64                            | 52.50    | n/a    | n/a     | n/a        |           |             |
| tq4_dot_product-uniform                  | K=128, N=128                          | 49.75    | n/a    | n/a     | n/a        |           |             |
| tq4_dot_product-varying                  | K=128, N=128                          | 51.00    | n/a    | n/a     | n/a        |           |             |
| tq4_dequantize_variant-tiny              | M=512, num_blocks=8                   | inf      | n/a    | n/a     | n/a        |           |             |
| tq4_dequantize_variant-ramp_pow2         | M=512, num_blocks=8                   | inf      | n/a    | n/a     | n/a        |           |             |
| tq4_dpv-64x128-ones                      | K=64, N=128                           | inf      | n/a    | n/a     | n/a        |           |             |
| tq4_dpv-128x64-alt                       | K=128, N=64                           | inf      | n/a    | n/a     | n/a        |           |             |
| tq4_dpv-192x64-ones-zero-pow2            | K=192, N=64                           | inf      | n/a    | n/a     | n/a        |           |             |
| tq4_dpv-64x256-altsign-max-pow2          | K=64, N=256                           | inf      | n/a    | n/a     | n/a        |           |             |
| tq4_dot_product_onehot_oracle            | K=128, N=64, pos=0                    | inf      | n/a    | n/a     | n/a        |           |             |
| tq4_dot_product_onehot_oracle            | K=128, N=64, pos=1                    | inf      | n/a    | n/a     | n/a        |           |             |
| tq4_dot_product_onehot_oracle            | K=128, N=64, pos=2                    | inf      | n/a    | n/a     | n/a        |           |             |
| tq4_dot_product_onehot_oracle            | K=128, N=64, pos=31                   | inf      | n/a    | n/a     | n/a        |           |             |
| tq4_dot_product_onehot_oracle            | K=128, N=64, pos=62                   | inf      | n/a    | n/a     | n/a        |           |             |
| tq4_dot_product_onehot_oracle            | K=128, N=64, pos=63                   | inf      | n/a    | n/a     | n/a        |           |             |
| tq4_dot_product_onehot_oracle            | K=128, N=64, pos=64                   | inf      | n/a    | n/a     | n/a        |           |             |
| tq4_dot_product_onehot_oracle            | K=128, N=64, pos=65                   | inf      | n/a    | n/a     | n/a        |           |             |
| tq4_dot_product_onehot_oracle            | K=128, N=64, pos=95                   | inf      | n/a    | n/a     | n/a        |           |             |
| tq4_dot_product_onehot_oracle            | K=128, N=64, pos=126                  | inf      | n/a    | n/a     | n/a        |           |             |
| tq4_dot_product_onehot_oracle            | K=128, N=64, pos=127                  | inf      | n/a    | n/a     | n/a        |           |             |
| tq4_codebook_reload                      | cb0: K=64, N=64                       | inf      | n/a    | n/a     | n/a        |           |             |
| tq4_codebook_reload                      | cb1: K=64, N=64                       | inf      | n/a    | n/a     | n/a        |           |             |
| if4_if8_dot_product-IF4-FP               | K=64, N=64                            | 58.50    | n/a    | n/a     | n/a        |           |             |
| if4_if8_dot_product-IF4-INT              | K=64, N=64                            | 62.00    | n/a    | n/a     | n/a        |           |             |
| if4_if8_dot_product-IF8-FP               | K=64, N=64                            | 34.50    | n/a    | n/a     | n/a        |           |             |
| if4_if8_dot_product-IF8-INT              | K=64, N=64                            | 33.00    | n/a    | n/a     | n/a        |           |             |
| if4_if8_dot_product-IF4-FP               | K=128, N=128                          | 55.25    | n/a    | n/a     | n/a        |           |             |
| if4_if8_dot_product-IF4-INT              | K=128, N=128                          | 57.50    | n/a    | n/a     | n/a        |           |             |
| dequantize-IF4-INT                       | M=64, N=128                           | 23.12    | 1.09   | n/a     | n/a        |           |             |
| dequantize-IF4-FP                        | M=64, N=128                           | 19.50    | 1.08   | n/a     | n/a        |           |             |
| dequantize-IF8-INT                       | M=64, N=128                           | 47.00    | 1.00   | n/a     | n/a        |           |             |
| dequantize-IF8-FP                        | M=64, N=128                           | 32.00    | 1.00   | n/a     | n/a        |           |             |
| matmat_mul_non_aligned_writeback         | M=2, K=256, N=32                      | 57.00    | n/a    | n/a     | n/a        |           |             |
| rope_hf_core_dram                        | M=64, N=512                           | 54.75    | 0.78   | n/a     | 16448      |           |             |
| rope_hf_core_dram+pbi                    | M=64, N=512                           | 55.00    | 0.91   | n/a     | 576        |           |             |
| bf16_permute                             | dim_0=144, dim_1=48, dim_2=64         | inf      | n/a    | n/a     | n/a        |           |             |
| patching                                 | C=3 H=384 W=384 patch=4x4 K=1024 N=64 | 57.75    | n/a    | n/a     | n/a        |           |             |
| mix_of_broadcast_eltwise_add_eltwise_mul | dim=8192                              | 41.25    | n/a    | n/a     | n/a        |           |             |
| eltwise_core_dram_legacy_mul             | M=64,N=512,elements=32768             | 57.25    | 0.84   | n/a     | 192        |           |             |
| eltwise_core_dram_legacy_add             | M=64,N=512,elements=32768             | 66.00    | 0.84   | n/a     | 192        |           |             |
| eltwise_core_dram_legacy_sub             | M=64,N=512,elements=32768             | 66.00    | 0.84   | n/a     | 192        |           |             |
| bf16_transpose+dynamic                   | M=64, N=64                            | inf      | 0.15   | 305.49  | 1792       | identical | -18.3%      |
| bf16_transpose+legacy                    | M=64, N=64                            | inf      | 0.19   | 373.77  | 2176       |           |             |
| bf16_transpose+dynamic                   | M=256, N=256                          | inf      | 0.29   | 570.44  | 1792       | identical | -4.5%       |
| bf16_transpose+legacy                    | M=256, N=256                          | inf      | 0.30   | 597.54  | 8320       |           |             |
| bf16_transpose+dynamic                   | M=512, N=2048                         | inf      | 0.19   | 387.07  | 1792       | identical | -7.6%       |
| bf16_transpose+legacy                    | M=512, N=2048                         | inf      | 0.21   | 418.93  | 524864     |           |             |
| bf16_transpose+dynamic                   | M=1024, N=4032                        | inf      | 0.19   | 387.47  | 1792       |           |             |

**Dynamic vs Legacy:**
Throughput: dynamic is -10.3% vs legacy (geomean over 3 paired tests)
SNR: identical to legacy

**Status: INCOMPLETE (failed or aborted)**

**RNG State:**
seed: 0
start: py=46f264538534,torch=42bff94b75a5
end: py=d50110ddf8a6,torch=7a86c5e8176d
