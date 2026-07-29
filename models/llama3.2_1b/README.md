# LLaMA 3.2 1B

This folder contains the LLaMA 3.2 1B accelerator inference.

## Layout

- **llama3.2_1b_test.py** – Prefill + decode loop on accelerator.
- **llama3.2_1b_IF8.py** – IF8 weight variant with the same optimized execution flow.
- **llama3.2_1b_config.json** – Model and layout config.
- **llama3.2_1b_bin/** – IF4 weights, HF model, and programs (generated at runtime).
- **llama3.2_1b_if8_bin/** – IF8 weights and programs (generated at runtime).

## Prerequisites

- Run from the **repo root directory** so that `user_dma_core` is on the path.
- Python with `torch`, `transformers`, and DMA device access.

## Usage

From the repo root directory:

```bash
# Prefill + decode (default prompt)
python models/llama3.2_1b/llama3.2_1b_test.py

# Custom prompt
python models/llama3.2_1b/llama3.2_1b_test.py --prompt "What is 2+2?"

# IF8 uses the same prompt, prefill, decode, profile, and cache flow
python models/llama3.2_1b/llama3.2_1b_IF8.py --prompt "What is 2+2?"

# Restore Meta's full dated system block (the default is the lower-latency minimal chat wrapper)
python models/llama3.2_1b/llama3.2_1b_test.py --prompt "What is 2+2?" --standard-chat-template

# Compatibility A/B: restore the slower dequantize-then-BF16 prefill matmuls
python models/llama3.2_1b/llama3.2_1b_test.py --prompt "What is 2+2?" --two-pass-prefill

# IF8 A/B: test one-pass streaming instead of its measured-faster dequantize/BF16 default
python models/llama3.2_1b/llama3.2_1b_IF8.py --prompt "What is 2+2?" --stream-prefill

# Decoder-only A/B: use dequantize/BF16 matmuls instead of the faster streaming decoder
python models/llama3.2_1b/llama3.2_1b_test.py --prompt "What is 2+2?" --decoder-matmatmul

# Override the board clock when needed (Kintex-7 default: 5.0422 ns / 198.33 MHz)
python models/llama3.2_1b/llama3.2_1b_test.py --cycle 5.0422
```

Both precision variants use a 4096-token decode context and 4096-position RoPE
tables. The prompt/prefill limit remains 128 tokens.

## Measured prefill performance

Kintex-7 at 198.3256 MHz, hardware `0x884c96b9`, prompt `"x^2=-1"`:

| Path | Prefill tokens | FPGA time | Rate |
| --- | ---: | ---: | ---: |
| Previous Llama, dated system template | 39 | 3,432.9 ms | 11.36 tok/s |
| Optimized IF4, minimal chat template | 14 | **1,158.2 ms** | **12.09 tok/s** |
| Optimized IF8, dequantize/BF16 prefill | 14 | **1,349.8 ms** | **10.38 tok/s** |
| IF8, one-pass streaming A/B | 14 | 2,231.3 ms | 6.28 tok/s |
| Gemma3, equivalent 14-token prefill | 14 | 884.0 ms | 15.84 tok/s |

The default minimal wrapper removes the tokenizer's automatically injected 25-token
system/date block. The prefill compiler keeps layer state in one recurrent DRAM
buffer and scales the full query tensor once. IF4 uses the one-pass streaming
matmul; IF8 follows Gemma IF8 and uses the measured-faster dequantize/BF16
matmul. `--decoder-matmatmul` affects only decoder projections; the legacy
`--matmatmul` spelling remains accepted as an alias. Use
`--standard-chat-template` when the dated system metadata is required.

## Measured decode performance

Kintex-7 at 1066 / 5.375 = 198.3256 MHz, hardware `0x884c96b9`, prompt `"x^2=-1"`:

| Metric | Before | Optimized |
| --- | ---: | ---: |
| End-to-end decode | 7.89 tok/s | 8.29 tok/s |
| CPU time | 126.8 ms/token | 120.6 ms/token |
| FPGA time | 123.4 ms/token | 119.8 ms/token |
| Decoder instructions | 1,865.1 KiB | 1,633.2 KiB |

The optimized decoder reads each per-head K/V cache directly, hoists query
scaling once per layer, writes attention output into its final slot, avoids the
inter-layer hidden-state copy, prebuilds position dispatch entries, and reuses
host buffers. Generated instruction binaries are fingerprinted against the
compiler sources and config, so stale binaries rebuild automatically.

Gemma3 remains faster for this prompt (9.71 tok/s) because this Llama decoder
performs 3.022 GFLOP/token versus Gemma3's 2.056 GFLOP/token, about 47% more
model work. Llama's measured FPGA throughput is actually higher per unit of
work; the remaining absolute gap is architectural rather than host overhead.
The IF8 variant measures 4.55 tok/s because its quantized weight payload is
twice the IF4 payload; it no longer performs the old redundant cache and
inter-layer copies.
