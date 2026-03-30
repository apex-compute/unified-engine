# Parakeet-TDT-0.6B

This folder contains the Parakeet-TDT-0.6B accelerator inference.

## Layout

- **parakeet_test.py** – Encoder + TDT decode loop on accelerator.
- **parakeet_config.json** – Model and layout config.
- **parakeet_bin/** – Weights, HF model, and decoder binaries (generated at runtime).
- **arch.md** – Detailed architecture notes (Conformer encoder, TDT decoder, mel frontend).

## Prerequisites

- Run from the **repo root directory** so that `user_dma_core` is on the path.
- Python with `torch`, `transformers`, `soundfile`, and DMA device access.

## Usage

From the repo root directory:

```bash
# Default audio file
python models/parakeet/parakeet_test.py

# Custom audio
python models/parakeet/parakeet_test.py --audio path/to/audio.wav
```