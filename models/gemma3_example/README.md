# Gemma3 example

This folder contains the Gemma3 accelerator inference example and numeric verification.

## Layout

- **gemma3_test.py** – Prefill + decode loop on accelerator (single or dual engine).
- **gemma3_numeric.py** – Numeric verification with torch reference (prefill + decoder).
- **gemma3_config.json** – Model and layout config.
- **decoder_program.json** / **decoder_program_slave.json** – Decoder program metadata (written on first decoder compile).
- **gemma3_bin/** – Weights, HF model, and decoder binaries. Contains:
  - `weights_gemma3_hf.bin` or `full_model_weights.bin`
  - `gemma-3-1b-it/` (Hugging Face model, or set via config)
  - `decoder_program.bin` (and `decoder_program_slave.bin` when using dual engine)

## Prerequisites

- Run from the **repo root directory** so that `user_dma_core` is on the path:
  ```bash
  python models/gemma3_example/gemma3_test.py
  ```
- Python with `torch`, `transformers`, and DMA device access.

## Usage

From the **repo root** directory:

```bash
# Prefill + decode (default prompt)
python models/gemma3_example/gemma3_test.py

# Custom prompt
python models/gemma3_example/gemma3_test.py --prompt "Your prompt here"

# DMA device and clock cycle time (default: xdma0, 5.62 ns)
python models/gemma3_example/gemma3_test.py --dev xdma0 --cycle 5.62

# Use local full-model weights bin
python models/gemma3_example/gemma3_test.py --local-weights

# Dual engine (master + slave)
python models/gemma3_example/gemma3_test.py --dual-engine
```