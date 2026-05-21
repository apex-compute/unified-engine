# Gemma4 E2B example

This folder contains the Gemma4 E2B accelerator inference example. Both
text-only prompts and image + text (VLM) prompts are supported. The full
pipeline — patch embedder, 16-layer vision encoder, pooler, embed_vision
projection, 35-layer language-model prefill, and decoder loop — runs on
the accelerator.

## Layout

- **gemma4_e2b_test.py** – Compile + run. Builds `gemma4_instruction.bin` on first invocation (or extends it for VLM), then runs.
- **gemma4_e2b_run_from_bin.py** – Execute-only. Loads the pre-compiled bin and runs; refuses to compile.
- **gemma4_e2b_config.json** – Model and layout config.
- **gemma4_e2b_bin/** – Weights, tokenizer, side-cache, and the single unified instruction bin. Contains:
  - `weights_gemma4_e2b_hf.bin`
  - `host_weights.bin` + `host_weights.json`
  - `gemma4_instruction.bin` + `gemma4_instruction.json`
  - `gemma-4-E2B-it/` (Hugging Face tokenizer files)

## Prerequisites

- Run from the **repo root directory** so that `user_dma_core` is on the path:
  ```bash
  python models/gemma4_e2b/gemma4_e2b_test.py
  ```
- Python with `torch`, `transformers >= 5.5.0`, and DMA device access.
- The HF model (`google/gemma-4-E2B-it`) is downloaded automatically into `gemma4_e2b_bin/gemma-4-E2B-it/` on first run (requires network access and a Hugging Face account with access to the Gemma license).

## Usage

From the **repo root** directory. Gemma4 E2B supports three modes — image
+ text (VLM), text only, and audio + text. Exactly one encoder modality
can be active per run.

### Recommended first run — VLM mode

The first run builds the unified `gemma4_instruction.bin` containing
both the LM and the vision encoder. Every subsequent run in any mode
loads this same bin and skips compilation, so doing VLM first sets you
up for everything afterward.

```bash
# Default image (../../test_samples/yosemite.jpg) + default prompt
python models/gemma4_e2b/gemma4_e2b_test.py --vision-enable

# Custom image and prompt
python models/gemma4_e2b/gemma4_e2b_test.py --image path/to/image.jpg --prompt "What is in this image?"
```

### Language model (text only)

```bash
# Default prompt
python models/gemma4_e2b/gemma4_e2b_test.py

# Custom prompt
python models/gemma4_e2b/gemma4_e2b_test.py --prompt "Your prompt here"
```

### Audio + text

```bash
# Default audio (../../test_samples/apex.wav) + default prompt
python models/gemma4_e2b/gemma4_e2b_test.py --audio-enable

# Custom audio and prompt
python models/gemma4_e2b/gemma4_e2b_test.py --audio path/to/file.wav --prompt "Transcribe this exactly."
```

### Fast execute-only path

After the first compile, use `gemma4_e2b_run_from_bin.py` for fast
runs — it loads the pre-built bin and never compiles.

```bash
python models/gemma4_e2b/gemma4_e2b_run_from_bin.py                                     # LM only
python models/gemma4_e2b/gemma4_e2b_run_from_bin.py --vision-enable                     # VLM, default image
python models/gemma4_e2b/gemma4_e2b_run_from_bin.py --image my_photo.jpg --prompt "?"   # VLM, custom image
```

`gemma4_e2b_run_from_bin.py` is fully self-contained — it does not
import from `gemma4_e2b_test.py` and does not need the full
`gemma-4-E2B-it/` HF model directory at run time (a bundled
tokenizer subset in `gemma4_e2b_bin/tokenizer/` is used instead).
LM-only deploys need only:

```
gemma4_e2b_run_from_bin.py
gemma4_e2b_config.json
user_dma_core.py + quant_schemas.py    (FPGA driver + codec lib)
gemma4_e2b_bin/
  weights_gemma4_e2b_hf.bin
  host_weights.bin + .json
  gemma4_instruction.bin + .json
  tokenizer/                            (~32 MB)
```

VLM mode currently still loads the HF model directory for vision encoder
weights — fold-in to the bin is a future enhancement.

### Common flags

```bash
# DMA device and clock cycle time (default: xdma0, 5.62 ns)
python models/gemma4_e2b/gemma4_e2b_test.py --dev xdma0 --cycle 5.62

# Use local full-model weights bin
python models/gemma4_e2b/gemma4_e2b_test.py --local-weights
```

### Mode notes

- Passing `--image PATH` implies `--vision-enable`. Passing `--vision-enable` alone uses the shipped default image.
- Passing `--audio PATH` implies `--audio-enable`. Passing `--audio-enable` alone uses the shipped default audio.
- `--vision-enable/--image` and `--audio-enable/--audio` are mutually exclusive (one modality per run).
- Audio input requires `soundfile` (`pip install soundfile`).

To force a clean rebuild after changing buckets in the config or pulling
new ISA semantics:

```bash
rm gemma4_e2b_bin/gemma4_instruction.bin gemma4_e2b_bin/gemma4_instruction.json
python models/gemma4_e2b/gemma4_e2b_test.py --vision-enable
```

For design details (unified-bin layout, host-RAM optimisation, vision
fixed-shape contract, low-memory deploy workflow), see
`src/template/notes/notes_gemma4_e2b.md`.
