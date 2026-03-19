# LLaMA 3.2 1B example

This folder contains the LLaMA 3.2 1B accelerator inference example.

## Layout

- **llama32_1b_test.py** – Prefill + decode loop on accelerator.
- **llama32_1b_config.json** – Model and layout config.
- **llama32_1b_bin/** – Weights, HF model, and decoder binaries (generated at runtime).

## Prerequisites

- Run from the **parent directory** so that `user_dma_core` is on the path:
- Python with `torch`, `transformers`, and DMA device access.

## Usage

From the parent directory:

```bash
# Prefill + decode (default prompt)
python llama3.2_1b_example/llama32_1b_test.py

# Custom prompt
python llama3.2_1b_example/llama32_1b_test.py --prompt "What is 2+2?"
```
