# Qwen3 1.7B example

This folder contains the Qwen3 1.7B accelerator inference example.

## Layout

- **qwen3_1.7b_test.py** – Prefill + decode loop on accelerator.
- **qwen3_1.7b_config.json** – Model and layout config.
- **qwen3_1.7b_bin/** – Weights, HF model, and decoder binaries (generated at runtime).

## Prerequisites

- Run from the **parent directory** so that `user_dma_core` is on the path:
- Python with `torch`, `transformers`, and DMA device access.

## Usage

From the parent directory:

```bash
# Prefill + decode (default prompt)
python qwen3_1.7b_example/qwen3_1.7b_test.py

# Custom prompt
python qwen3_1.7b_example/qwen3_1.7b_test.py --prompt "What is 2+2?"
```
