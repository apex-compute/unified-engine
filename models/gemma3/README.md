# Gemma3 example

This folder contains the Gemma3 accelerator inference example and numeric verification.

## Layout

- **gemma3_test.py** – Prefill + decode loop on accelerator (IF4 codec; single or dual engine).
- **gemma3_test_IF8.py** – Same flow, IF8 (8-bit adaptive block-scaled) codec. See `notes/notes_gemma3_if8.md` for design points.
- **gemma3_numeric.py** – Numeric verification with torch reference (prefill + decoder).
- **gemma3_config.json** – Model and layout config (shared by IF4 and IF8; the IF8 script re-packs offsets at load time).
- **decoder_program.json** / **decoder_program_slave.json** – Decoder program metadata (written on first decoder compile).
- **gemma3_bin/** – Weights, HF model, and decoder binaries. Contains:
  - `weights_gemma3_hf.bin` or `full_model_weights.bin`
  - `gemma-3-1b-it/` (Hugging Face model, or set via config)
  - `decoder_program.bin` (and `decoder_program_slave.bin` when using dual engine)

## Prerequisites

- Run from the **parent directory** (`pcie_utils`) so that `user_dma_core` is on the path:
  ```bash
  cd pcie_utils
  python gemma3_example/gemma3_test.py
  ```
- Python with `torch`, `transformers`, and DMA device access.

## Usage

From the **pcie_utils** (parent) directory:

```bash
# Prefill + decode (default prompt)
python gemma3_example/gemma3_test.py            # IF4 codec
python gemma3_example/gemma3_test_IF8.py        # IF8 codec

# Custom prompt
python gemma3_example/gemma3_test.py     --prompt "Your prompt here"
python gemma3_example/gemma3_test_IF8.py --prompt "Your prompt here"

# DMA device and clock cycle time (default: xdma0, 5.62 ns)
python gemma3_example/gemma3_test.py --dev xdma0 --cycle 5.62

# Use local full-model weights bin
python gemma3_example/gemma3_test.py --local-weights

# Dual engine (master + slave)
python gemma3_example/gemma3_test.py --dual-engine
```

The IF4 and IF8 scripts use **separate** cache files (`weights_gemma3_hf.bin`
+ `gemma3_instruction.bin` for IF4; `weights_gemma3_hf_if8.bin` +
`gemma3_instruction_if8.bin` for IF8), so they do not collide and you can
switch between codecs without manually clearing caches. IF8 first-run
quantization takes longer (~245 s on a typical CPU) because the FP8
snap-to-grid is more expensive than IF4; subsequent runs skip the
quantization step.