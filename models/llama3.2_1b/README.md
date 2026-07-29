# LLaMA 3.2 1B

This folder contains the LLaMA 3.2 1B accelerator inference.

## Layout

- **llama3.2_1b_test.py** – Prefill + decode loop on accelerator.
- **llama3.2_1b_config.json** – Model and layout config.
- **llama3.2_1b_bin/** – Weights, HF model, and decoder binaries (generated at runtime).

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

# Override the board clock when needed (Kintex-7 default: 5.0422 ns / 198.33 MHz)
python models/llama3.2_1b/llama3.2_1b_test.py --cycle 5.0422
```

## Measured decode performance

Kintex-7 at 1066 / 5.375 = 198.3256 MHz, hardware `0x884c96b9`, prompt `"x^2=-1"`:

| Metric | Before | Optimized |
| --- | ---: | ---: |
| End-to-end decode | 7.89 tok/s | 8.21 tok/s |
| CPU time | 126.8 ms/token | 121.8 ms/token |
| FPGA time | 123.4 ms/token | 121.2 ms/token |
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
