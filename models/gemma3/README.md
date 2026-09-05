# Gemma3 example

This folder contains the Gemma3 accelerator inference example and numeric verification.

> [!WARNING]
> `gemma3_test_IF8.py` is deprecated and IF8 is currently not working. It is
> retained only for historical/debugging reference, excluded from
> `model_auto_test.py`, and must not be treated as a supported inference path.
> Use `gemma3_test.py` (IF4).

## Layout

- **gemma3_test.py** – Prefill + decode loop on accelerator (single engine).
- **gemma3_test_IF8.py** – **Deprecated and currently non-working** IF8 experiment.
- **gemma3_numeric.py** – Numeric verification with torch reference (prefill + decoder).
- **gemma3_config.json** – Model and layout config.
- **decoder_program.json** – Decoder program metadata (written on first decoder compile).
- **gemma3_bin/** – Weights, HF model, and decoder binaries. Contains:
  - `weights_gemma3_hf.bin` or `full_model_weights.bin`
  - `gemma-3-1b-it/` (Hugging Face model, or set via config)
  - `decoder_program.bin`

## Prerequisites

- Run from the **repository root**:
  ```bash
  python models/gemma3/gemma3_test.py
  ```
- Python with `torch`, `transformers`, and DMA device access.

## Usage

From the repository root:

```bash
# Prefill + decode (default prompt)
python models/gemma3/gemma3_test.py

# Custom prompt
python models/gemma3/gemma3_test.py --prompt "Your prompt here"

# DMA device and clock (Kintex-7: xdma0, 1066 / 5.375 = 198.3256 MHz, 5.0422 ns)
python models/gemma3/gemma3_test.py --dev xdma0 --cycle 5.0422

# Use local full-model weights bin
python models/gemma3/gemma3_test.py --local-weights

```
