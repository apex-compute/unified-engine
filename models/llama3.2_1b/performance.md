# LLaMA 3.2 1B Performance

Hardware: Andromeda FPGA (HW version 0x25e4082c), clock cycle 5.15 ns (194 MHz), device kintex7.

Architecture: 16 layers, hidden 2048, GQA (8 KV heads × 1 Q/KV = 8 Q heads), actual_head_dim 64, MLP 8192, IF4 quantization.

## Instruction Size

| Stage   | Baseline (kB) | Current — streaming LM head (kB) |
|---------|--------------|----------------------------------|
| Prefill | 842.2        | 842.2                            |
| Decoder | 877.0        | 666.4                            |
| Total   | 1,719.2      | 1,508.7                          |

## Run Performance

| Metric | Baseline | Current — streaming LM head |
|--------|----------|-----------------------------|
| Prompt tokens (prefill) | 44 | 44 |
| Decode tokens generated | 71 | 71 |
| **Prefill HW time (ms)** | 3,944.0 | 3,944.0 |
| **Prefill CPU time (ms)** | 3,943.5 | 3,948.8 |
| **Decode 1st token HW (ms/tok)** | 144.2 | 120.6 |
| **Decode 1st token (tok/s)** | 6.93 | 8.29 |
| **Decode avg HW time (ms/tok)** | 146.8 | 123.9 |
| **Decode avg CPU time (ms/tok)** | 153.3 | 130.3 |
| Decode throughput (inc. host detokenizer) (tok/s) | 6.52 | 7.67 |

## Decoder Step Breakdown (1st token, all 16 layers)

| Step | Baseline (ms) | Current — streaming LM head (ms) |
|------|--------------|----------------------------------|
| pre_norm | 0.14 | 0.14 |
| qkv_proj | 8.95 | 8.95 |
| rope | 0.24 | 0.24 |
| kv_scatter_attn | 11.01 | 11.01 |
| o_proj_residual | 6.02 | 6.02 |
| pre_ffn_norm | 0.11 | 0.11 |
| mlp_gateup_silu | 47.53 | 47.53 |
| mlp_down_residual | 23.74 | 23.74 |
| norm_lm_head | 46.49 | 23.20 |
| **Total** | **144.24** | **120.95** |
