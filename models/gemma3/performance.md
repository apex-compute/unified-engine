# Gemma3 Performance

Hardware: Andromeda FPGA (HW version 0x25e4082c), clock cycle 5.15 ns (194 MHz), device kintex7.

Architecture: 26 layers, hidden 2048, GQA (4 KV heads × 4 Q/KV = 16 Q heads), head_dim 256, MLP 16384, IF4 quantization.

## Instruction Size

| Stage   | Baseline (kB) | Quantized LM head (kB) | + flash attn reuse (kB) |
|---------|--------------|----------------------|------------------------|
| Prefill | 1,655.8      | 1,655.8              | **382.5**              |
| Decoder | 1,150.4      | 1,077.3              | **326.9**              |
| Total   | 2,806.2      | 2,733.1              | **709.4**              |

Prefill: 4.3× smaller (26 flash_attention_core copies → 1 shared subroutine + 26 call stubs).
Decoder: 3.2× smaller (26 decoder_group_attention_core copies → 1 shared subroutine + 26 KV-staging + call stubs).
Total: 3.8× smaller vs quantized LM head baseline.

## Run Performance

| Metric | Baseline | Quantized LM head | + flash attn reuse |
|--------|----------|-------------------|-------------------|
| Prompt tokens (prefill) | 19 | 19 | 19 |
| Decode tokens generated | 77 | 113 | 77 |
| **Prefill HW time (ms)** | 1,224.0 | 1,224.0 | 1,224.1 |
| **Prefill CPU time (ms)** | 1,241.1 | 1,232.6 | 1,236.0 |
| **Decode 1st token HW (ms/tok)** | 120.2 | 95.3 | 95.8 |
| **Decode 1st token (tok/s)** | 8.32 | 10.50 | 10.44 |
| **Decode avg HW time (ms/tok)** | 121.4 | 97.2 | 97.3 |
| **Decode avg CPU time (ms/tok)** | 128.7 | 105.2 | 104.7 |
| Decode throughput (inc. host detokenizer) (tok/s) | 7.77 | 9.51 | 9.55 |

## Decoder Step Breakdown (1st token, all 26 layers)

| Step | Baseline (ms) | Quantized LM head (ms) | + flash attn reuse (ms) |
|------|--------------|----------------------|------------------------|
| pre_norm | 0.19 | 0.19 | 0.19 |
| qkv_proj_vcache | 4.27 | 4.27 | 4.27 |
| qk_norm_rope | 0.53 | 0.53 | 0.52 |
| attention | 5.55 | 5.55 | 5.55 |
| o_proj_post_attn_norm_residual | 2.96 | 2.96 | 2.96 |
| pre_ffn_norm | 0.15 | 0.15 | 0.15 |
| mlp_gateup_gelu_mul | 36.97 | 36.97 | 36.97 |
| mlp_down_post_ffn_norm_residual | 18.59 | 18.59 | 18.59 |
| norm_lm_head | 51.62 | 26.68 | 26.54 |
| **Total** | **120.84** | **95.90** | **95.75** |

---

## Gemma3-IF8 Performance (+ flash attn reuse)

Same subroutine reuse applied to both prefill and decoder.

### Instruction Size

| Stage   | + flash attn reuse (kB) |
|---------|------------------------|
| Prefill | 382.5                  |
| Decoder | 418.3                  |
| Total   | 800.8                  |

### Run Performance

| Metric | + flash attn reuse |
|--------|--------------------|
| Prompt tokens (prefill) | 19 |
| Decode tokens generated | 76 |
| **Prefill HW time (ms)** | 1,280.4 |
| **Decode avg HW time (ms/tok)** | 220.0 |
| Decode throughput (tok/s) | 4.77 |
