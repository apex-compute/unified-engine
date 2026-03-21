# Qwen3 1.7B

This folder contains the Qwen3 1.7B accelerator inference.

## Layout

- **qwen3_1.7b_test.py** – Prefill + decode loop on accelerator.
- **qwen3_1.7b_config.json** – Model and layout config.
- **qwen3_1.7b_bin/** – Weights, HF model, and decoder binaries (generated at runtime).

## Prerequisites

- Run from the **repo root directory** so that `user_dma_core` is on the path:
- Python with `torch`, `transformers`, and DMA device access.

## Usage

From the repo root directory:

```bash
# Prefill + decode (default prompt)
python models/qwen3_1.7b/qwen3_1.7b_test.py

# Custom prompt
python models/qwen3_1.7b/qwen3_1.7b_test.py --prompt "What is 2+2?"
```
