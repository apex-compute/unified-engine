# Gemma4 E2B

This folder contains the Gemma4 E2B accelerator inference. Both text-only
prompts and image + text (VLM) prompts are supported. The full pipeline —
patch embedder, 16-layer vision encoder, pooler, embed_vision projection,
35-layer language-model prefill, and decoder loop — runs on the accelerator.

## Layout

- **gemma4_e2b_test.py** – Prefill + decode loop on accelerator (text or VLM, single or dual engine).
- **gemma4_e2b_config.json** – Model and layout config.
- **decoder_program.json** – Decoder program metadata (written on first decoder compile).
- **gemma4_e2b_bin/** – Weights, HF model, and decoder binaries. Contains:
  - `weights_gemma4_e2b_hf.bin` (generated from the HF model on first run)
  - `gemma-4-E2B-it/` (Hugging Face model, or set via config)
  - `decoder_program.bin`

## Model Summary

| Parameter | Value |
|-----------|-------|
| Architecture | Gemma4ForConditionalGeneration |
| Language model layers | 35 (28 sliding-attention + 7 full-attention) |
| Hidden size | 1536 |
| Attention heads / KV heads | 8 / 1 (GQA, group size 8) |
| Head dim (sliding / full) | 256 / 512 |
| MLP intermediate | 6144 (layers 0–14), 12288 (layers 15–34) |
| Vocab size | 262 144 |
| Vision encoder layers | 16 |
| Vision hidden size | 768 |
| Image features | 256 soft tokens × 1536 |
| Max context | 512 |
| Codec | IF4 (MixMSE_64) for matmul weights; BF16 for norms / scales / embeddings |

## Prerequisites

- Run from the **repo root directory** so that `user_dma_core` is on the path:
  ```bash
  python models/gemma4_e2b/gemma4_e2b_test.py
  ```
- Python with `torch`, `transformers >= 5.5.0`, and DMA device access.
- The HF model (`google/gemma-4-E2B-it`) is downloaded automatically into `gemma4_e2b_bin/gemma-4-E2B-it/` on first run (requires network access and a Hugging Face account with access to the Gemma license).

## Usage

From the **repo root** directory. Gemma4 E2B supports three modes — text only, image + text (VLM), and audio + text. Exactly one encoder modality can be active per run.

### Language model (text only)

```bash
# Default prompt
python models/gemma4_e2b/gemma4_e2b_test.py

# Custom prompt
python models/gemma4_e2b/gemma4_e2b_test.py --prompt "Your prompt here"
```

### Vision + text (VLM)

```bash
# Use the shipped example image (../../test_samples/yosemite.jpg) with default prompt
python models/gemma4_e2b/gemma4_e2b_test.py --vision-enable

# Custom image and prompt
python models/gemma4_e2b/gemma4_e2b_test.py --image path/to/image.jpg --prompt "What is in this image?"
```

### Audio + text

```bash
# Use the shipped example audio (../../test_samples/apex.wav) with default prompt
python models/gemma4_e2b/gemma4_e2b_test.py --audio-enable

# Custom audio and prompt
python models/gemma4_e2b/gemma4_e2b_test.py --audio path/to/file.wav --prompt "Transcribe this exactly."
```

### Common flags

```bash
# DMA device and clock cycle time (default: xdma0, 5.62 ns)
python models/gemma4_e2b/gemma4_e2b_test.py --dev xdma0 --cycle 5.62

# Use local full-model weights bin
python models/gemma4_e2b/gemma4_e2b_test.py --local-weights
```

### Notes

- Passing `--image PATH` implies `--vision-enable`. Passing `--vision-enable` alone uses the shipped default image.
- Passing `--audio PATH` implies `--audio-enable`. Passing `--audio-enable` alone uses the shipped default audio.
- `--vision-enable/--image` and `--audio-enable/--audio` are mutually exclusive (one modality per run).
- Audio input requires `soundfile` (`pip install soundfile`).
