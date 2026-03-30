# Swin Transformer (Swin-Large, patch4, window12, 384, ImageNet-22k)

This folder contains the Swin Transformer accelerator inference.

## Layout

- **swin_test.py** – Forward pass on accelerator.
- **swin_config.json** – Model and layout config.
- **swin_bin/** – Weights, HF model, and program binaries (generated at runtime).

## Prerequisites

- Run from the **repo root directory** so that `user_dma_core` is on the path.
- Python with `torch`, `transformers`, `Pillow`, `torchvision`, and DMA device access.

## Usage

From the repo root directory:

```bash
# Default image
python models/swin/swin_test.py

# Custom image
python models/swin/swin_test.py --image path/to/image.jpg
```

## Weight Binary

Generated automatically on first run from HuggingFace model
(`microsoft/swin-large-patch4-window12-384-in22k`). Stored at `swin_bin/`.
INT4 quantized weights.
