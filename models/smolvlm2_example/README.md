# SmolVLM2 example

This folder contains the SmolVLM2-500M vision-language accelerator inference example.

## Layout

- **smolvlm2_test.py** – Prefill + decode loop on accelerator (fp4_e2m1 vision, q4_64 language).
- **smolvlm2_config.json** – Model and layout config.
- **smolvlm2_bin/** – Weights, HF model, and decoder binaries (generated at runtime).

## Prerequisites

- Run from the **repo root directory** so that `user_dma_core` is on the path.
- Python with `torch`, `transformers`, and DMA device access.

## Usage

From the repo root directory:

```bash
# Prefill + decode (default prompt)
python models/smolvlm2_example/smolvlm2_test.py

# Custom prompt and image
python models/smolvlm2_example/smolvlm2_test.py --prompt "Describe this image." --image test_samples/vette.jpg
```
